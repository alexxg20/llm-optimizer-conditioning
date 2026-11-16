#!/usr/bin/env python3
"""Validation loss of a saved checkpoint at several validation chunk lengths.

Builds the pinned model the way train_feasibility.py does, loads the checkpoint's weights and runs the
harness's validation loop over the same val_tokens once per --chunk-tokens value.

Attention is already limited to the checkpoint step's window (1792 tokens at the end of training) and to the
current document. A chunk cut therefore only affects tokens within that window of a chunk start whose document
crosses the boundary; chunk sizes at or below the training micro-batch measure this context effect.

Chunk sizes above the training micro-batch put the model on positions it never saw. The pinned Rotary computes
its angles in bf16 (inv_freq is cast with the model), so for the highest-frequency component a position p is
rounded to steps of 2^(floor(log2 p) - 7), which destroys relative position between nearby tokens. Losses at
those sizes are dominated by this rounding, not by context.

Thermal pause/stop as in the harness; clocks and power are never changed.
Run from pretraining/:  .venv/bin/python scripts/eval_checkpoint.py checkpoints/<run_id>/ckpt_step001530.pt
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Imported before torch so its allocator setting applies.
import train_feasibility as tf  # noqa: E402
import torch  # noqa: E402

DEFAULT_CHUNK_TOKENS = (4096, 16384, 32768, 65536)
REFERENCE_TOLERANCE = 1e-3
EXIT_REFERENCE_MISMATCH = 5


def chunk_steps(val_tokens, chunk_tokens):
    """Number of chunks per size; every size must tile val_tokens exactly so all sizes score the same targets."""
    steps = {}
    for T in chunk_tokens:
        if T <= 0 or T % 128:
            raise ValueError(f"chunk size {T} must be a positive multiple of 128 (FlexAttention block size)")
        if val_tokens % T:
            raise ValueError(f"chunk size {T} must divide val_tokens={val_tokens:,}")
        steps[T] = val_tokens // T
    return steps


def attn_blocksize(step, num_iterations):
    return int(64*((step/num_iterations * (1792 - 64) + 64)//64))


def default_output(checkpoint, step):
    run_id = Path(checkpoint).resolve().parent.name
    return tf.PROJECT_DIR / "reports" / run_id / f"eval_chunks_{step:06d}.json"


def default_report(checkpoint):
    return tf.PROJECT_DIR / "reports" / Path(checkpoint).resolve().parent.name / f"{tf.REPORT_NAME}.json"


def logged_val_loss(report_path, step):
    path = Path(report_path)
    if not path.exists():
        return None
    evals = json.loads(path.read_text()).get("measurements", {}).get("evals", [])
    return next((e["val_loss"] for e in evals if e["step"] == step), None)


def reference_check(results, chunk_tokens, logged, tolerance=REFERENCE_TOLERANCE):
    """Compares the loss at the training micro-batch size with the loss the training run logged at that step."""
    check = {"chunk_tokens": chunk_tokens, "logged": logged, "tolerance": tolerance}
    measured = next((r["val_loss"] for r in results if r["chunk_tokens"] == chunk_tokens and r["status"] == "ok"), None)
    if logged is None:
        return {**check, "status": "not-verified", "detail": "no logged val_loss for this step"}
    if measured is None:
        return {**check, "status": "not-verified", "detail": f"no successful evaluation at {chunk_tokens} tokens"}
    diff = abs(measured - logged)
    return {**check, "measured": measured, "abs_diff": diff, "status": "ok" if diff <= tolerance else "mismatch"}


def exit_code(results, check):
    statuses = {r["status"] for r in results}
    if "aborted-thermal" in statuses:
        return tf.EXIT_THERMAL
    if "oom" in statuses:
        return tf.EXIT_OOM
    if check["status"] == "mismatch":
        return EXIT_REFERENCE_MISMATCH
    return tf.EXIT_OK


def build_model(up, state_dict):
    model = tf.construct_model(up)
    model = model.cuda().bfloat16()
    for m in model.modules():
        if isinstance(m, up.CastedLinear):
            m.float()
    model.load_state_dict(state_dict, strict=True)
    if hasattr(up.config, "coordinate_descent_tuning"):
        up.config.coordinate_descent_tuning = True
    return torch.compile(model)


def validate(model, val_loader, val_steps, attn_blocksize, cool):
    """The harness's validation loop (train_feasibility.train); returns (loss, first-chunk seconds, pause seconds)."""
    torch.cuda.synchronize()
    t_eval = time.monotonic()
    model.eval()
    val_loader.reset()
    val_loss = 0.0
    first_mb_s = None
    pause = 0.0
    for _ in range(val_steps):
        pause += cool()
        with torch.no_grad():
            x_val, y_val = val_loader.next_batch()
            val_loss += model(x_val, y_val, attn_blocksize=attn_blocksize)
        if first_mb_s is None:
            torch.cuda.synchronize()
            first_mb_s = time.monotonic() - t_eval - pause
    val_loss /= val_steps
    return val_loss.item(), first_mb_s, pause


class ThermalAbort(RuntimeError):
    pass


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="ckpt_step*.pt written by train_feasibility.py")
    p.add_argument("--chunk-tokens", type=int, nargs="+", default=list(DEFAULT_CHUNK_TOKENS),
                   help="validation chunk lengths in tokens (4096 16384 32768 65536)")
    p.add_argument("--output", default=None, help="JSON output (default: reports/<run_id>/eval_chunks_<step>.json)")
    p.add_argument("--report", default=None,
                   help="training report JSON with the logged val_loss (default: "
                        f"reports/<run_id>/{tf.REPORT_NAME}.json)")
    p.add_argument("--cool-at", type=float, default=82.0, help="pause between chunks at this GPU temperature in °C (82)")
    p.add_argument("--resume-at", type=float, default=77.0, help="resume once cooled to this temperature in °C (77)")
    p.add_argument("--max-temp", type=float, default=87.0, help="abort above this GPU temperature in °C (87)")
    p.add_argument("--max-cool-wait", type=float, default=900.0, help="abort if one pause exceeds this many seconds (900)")
    p.add_argument("--temp-poll-interval", type=float, default=2.0, help="seconds between telemetry samples (2)")
    a = p.parse_args(argv)
    if not a.resume_at < a.cool_at < a.max_temp:
        p.error("temperatures must satisfy --resume-at < --cool-at < --max-temp")
    return a


def main(argv=None):
    cli = parse_args(argv)
    up = tf.load_upstream()
    args = up.args
    try:
        steps_by_T = chunk_steps(args.val_tokens, cli.chunk_tokens)
    except ValueError as e:
        raise SystemExit(str(e))
    assert all(T * n == args.val_tokens for T, n in steps_by_T.items())

    ckpt = torch.load(cli.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    step = int(ckpt["step"])
    T_ref = int(ckpt["config"]["micro_batch_tokens"])
    run_id = Path(cli.checkpoint).resolve().parent.name
    out_path = Path(cli.output) if cli.output else default_output(cli.checkpoint, step)
    report_path = Path(cli.report) if cli.report else default_report(cli.checkpoint)
    reference = logged_val_loss(report_path, step)
    if ckpt.get("upstream_sha256") != tf.UPSTREAM_SHA256:
        print(f"WARNING: checkpoint was trained with upstream sha256 {ckpt.get('upstream_sha256')}, not the pinned file.")

    telemetry = tf.GpuTelemetry(tf.nvml_index())
    stop = tf.StopController()
    telemetry.start(cli.temp_poll_interval, tf.ThermalGovernor(cli.cool_at, cli.resume_at, cli.max_temp, cli.max_cool_wait), stop)
    log = lambda msg: print(msg, flush=True)

    def cool():
        p = tf.wait_for_cooling(telemetry.cooling, lambda: stop.requested, telemetry.latest_temp,
                                cli.cool_at, cli.resume_at, log, torch.cuda.synchronize)
        if stop.requested:
            raise ThermalAbort(stop.detail)
        return 0.0 if p is None else p["seconds"]

    model = build_model(up, ckpt["model"])
    del ckpt
    window = attn_blocksize(step, args.num_iterations)
    window_t = torch.tensor(window, dtype=torch.int, device='cuda')
    result = {
        "checkpoint": str(Path(cli.checkpoint).resolve()), "run_id": run_id, "step": step,
        "attn_blocksize": window, "training_micro_batch_tokens": T_ref, "val_tokens": args.val_tokens,
        "training_report": str(report_path), "logged_val_loss": reference,
        "gpu": telemetry.static_info().get("name"), "torch": torch.__version__, "time_utc": tf.utc_now(),
        "method": "train_feasibility validation loop (model.eval, torch.no_grad, val_loader.reset, mean of per-chunk "
                  "losses); torch._dynamo.reset() before each chunk size so each compiles static shapes as in training",
        "results": [],
    }
    log(f"{run_id} step {step}: attn_blocksize {window}, training micro-batch {T_ref:,}, {args.val_tokens:,} val tokens, "
        f"logged val_loss {reference}")

    def save():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tf.atomic_write(out_path, json.dumps(result, indent=2))

    try:
        for T, val_steps in steps_by_T.items():
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            rec = {"chunk_tokens": T, "chunks": val_steps, "positions_beyond_training": T > T_ref}
            log(f"\nchunk {T:,} tokens x {val_steps} chunks (first chunk includes compilation)...")
            t0 = time.monotonic()
            try:
                val_loader = up.DistributedDataLoader(str(tf.UPSTREAM_DIR / args.input_val_bin), T, 0, 1)
                loss, first_s, pause = validate(model, val_loader, val_steps, window_t, cool)
                torch.cuda.synchronize()
                rec.update(status="ok", val_loss=loss, wall_time_s=time.monotonic() - t0 - pause,
                           first_chunk_s=first_s, pause_s=pause)
            except ThermalAbort as e:
                rec.update(status="aborted-thermal", detail=str(e))
            except Exception as e:
                if not tf.is_oom(e):
                    raise
                rec.update(status="oom", message=str(e).strip().splitlines()[0][:300] if str(e).strip() else type(e).__name__)
            rec.update(peak_allocated_gib=torch.cuda.max_memory_allocated() / tf.GIB,
                       peak_reserved_gib=torch.cuda.max_memory_reserved() / tf.GIB,
                       max_temp_c=telemetry.window(t0, time.monotonic())["temp_max_c"])
            val_loader = None
            result["results"].append(rec)
            save()
            if rec["status"] == "ok":
                log(f"chunk {T:,}: val_loss {rec['val_loss']:.4f} ({rec['wall_time_s']:.1f} s, "
                    f"peak {rec['peak_allocated_gib']:.2f} GiB)")
            else:
                log(f"chunk {T:,}: {rec['status']} {rec.get('message') or rec.get('detail') or ''}")
            if rec["status"] == "aborted-thermal":
                break
    finally:
        telemetry.stop()

    check = result["reference_check"] = reference_check(result["results"], T_ref, reference)
    save()
    if check["status"] == "mismatch":
        print("\n" + "!" * 80 + f"\nWARNING: {T_ref}-token val_loss {check['measured']:.5f} does not reproduce the logged "
              f"{reference:.5f} (|diff| {check['abs_diff']:.2e} > {REFERENCE_TOLERANCE:g}).\n"
              "The other chunk sizes are not comparable with the training log.\n" + "!" * 80, flush=True)
    elif check["status"] == "not-verified":
        print("\n" + "!" * 80 + f"\nWARNING: NOT VERIFIED against the training log ({check['detail']}; "
              f"report {report_path}).\n" + "!" * 80, flush=True)
    log(f"\nwrote {out_path}")
    print(f"\n{'chunk tokens':>12}  {'val_loss':>8}  {'chunks':>6}  {'time s':>7}  {'peak GiB':>8}")
    for r in result["results"]:
        loss = f"{r['val_loss']:.4f}" if r["status"] == "ok" else r["status"]
        secs = f"{r['wall_time_s']:.1f}" if r["status"] == "ok" else "-"
        note = "  positions beyond training (bf16 RoPE)" if r["positions_beyond_training"] else ""
        print(f"{r['chunk_tokens']:>12,}  {loss:>8}  {r['chunks']:>6}  {secs:>7}  {r['peak_allocated_gib']:>8.2f}{note}")
    return exit_code(result["results"], check)


if __name__ == "__main__":
    sys.exit(main())
