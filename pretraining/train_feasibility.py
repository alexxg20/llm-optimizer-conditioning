#!/usr/bin/env python3
"""Real-training throughput / ETA harness for modded-nanogpt @ 9730304 on one laptop GPU.

The model, the Muon optimizer, and the data loader are executed from the unmodified pinned
file (upstream/train_gpt2.py, sha256-verified); the pinned main loop is replicated below with
live statistics, a memory preflight, thermal/OOM guards and report generation.
Run with --help, and see README.md for the exact commands.
"""
import os

# Must be set before CUDA initialises; many micro-batch-sized buffers fragment the default allocator.
if "PYTORCH_ALLOC_CONF" not in os.environ and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    ALLOC_CONF_SET_BY_SCRIPT = True
else:
    ALLOC_CONF_SET_BY_SCRIPT = False

import argparse
import contextlib
import dataclasses
import datetime
import glob
import hashlib
import json
import math
import platform
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
import types
import uuid
from collections import deque
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent
UPSTREAM_DIR = PROJECT_DIR / "upstream"
UPSTREAM_FILE = UPSTREAM_DIR / "train_gpt2.py"
UPSTREAM_REPO = "https://github.com/KellerJordan/modded-nanogpt"
UPSTREAM_COMMIT = "973030408364f8738b4ad9e8f912d8cbbf56e4d4"
UPSTREAM_SHA256 = "e74ceea94ff30807ab5e4f6a3d7d8cf3e1bee4bfd3b5b84e01f57d8e5f8a4e57"
UPSTREAM_EXEC_END = "args = Hyperparameters()\n"
DOWNLOAD_CMD = ".venv/bin/python upstream/data/cached_fineweb10B.py 9"
REPORT_NAME = "training_feasibility_report"
DESIGNED_WORLD_SIZE = 8  # upstream run.sh: torchrun --nproc_per_node=8

# From the 40 logs in records/120424_ValueEmbed/ at the pinned commit (final "step:1530/1530 val_loss" lines).
REFERENCE_8XH100 = {
    "hardware": "8x NVIDIA H100 80GB HBM3",
    "torch": "2.6.0.dev20241203+cu124",
    "runs": 40,
    "train_time_s_mean": 264.6,
    "train_time_s_min": 264.3,
    "train_time_s_max": 265.0,
    "final_val_loss_mean": 3.2777,
    "final_val_loss_sd": 0.0040,
    "source": "records/120424_ValueEmbed/*.txt at commit 9730304",
}

# The upstream clock restarts at step 10, so its train_time covers 1520 of the 1530 steps.
RECORD_FULL_RUN_S = REFERENCE_8XH100["train_time_s_mean"] * 1530 / 1520
SHARD_TOKENS = 100_000_000
CALIBRATION_MICRO_BATCHES = 8
CALIBRATION_WINDOWS = (896, 1792)
CALIBRATION_METHOD = ("forward+backward only, no optimizer update, real data, after training stopped: median time of "
                      f"{CALIBRATION_MICRO_BATCHES} micro-batches at the last measured window, 896 and 1792 tokens; "
                      "the per-micro-batch difference is interpolated linearly between those windows (clamped outside), "
                      "multiplied by the accumulation steps and averaged over the real 1530-step window schedule")
MIN_STEPS_FOR_FIRM_VERDICT = 10

GIB = 2**30
CUDA_CONTEXT_RESERVE_GIB = 0.5
FRAGMENTATION_FACTOR = 1.05
# Saved activations per token: (block tensors in units of n_embd bf16 values per layer,
# bytes per vocab entry for logits/loss/grad-of-logits at the backward peak).
ACTIVATION_COEFFS = {"lower": (10, 4), "central": (16, 8), "upper": (24, 12)}
# Unavoidable even with perfect recomputation: matmul inputs h, o, h2 (3d) + relu^2 output (4d)
# + flex-attention q, k, v (3d) per layer, and one materialised bf16 lm_head output.
MIN_SAVED_BLOCK, MIN_HEAD_BYTES = 10, 2
CANDIDATE_MICRO_BATCHES = (65536, 16384, 8192, 4096, 2048)
RUNNABLE_VERDICTS = ("plausible-tight", "expected-to-fit")
EXIT_OK, EXIT_ERROR, EXIT_REFUSED, EXIT_OOM, EXIT_THERMAL, EXIT_ABORTED = 0, 1, 2, 3, 4, 130


# ----------------------------------------------------------------------------- upstream


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_upstream():
    actual = sha256_of(UPSTREAM_FILE)
    listed = {}
    for line in (UPSTREAM_DIR / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            listed[name.strip()] = digest
    if actual != UPSTREAM_SHA256 or listed.get("train_gpt2.py") != UPSTREAM_SHA256:
        raise SystemExit(f"upstream/train_gpt2.py does not match commit {UPSTREAM_COMMIT[:7]} "
                         f"(sha256 {actual}); restore the pristine file.")
    return actual


def exec_upstream(module_name="upstream_train_gpt2"):
    """Executes the pinned file up to `args = Hyperparameters()` (definitions only, no side effects)."""
    verify_upstream()
    source = UPSTREAM_FILE.read_text()
    end = source.index(UPSTREAM_EXEC_END) + len(UPSTREAM_EXEC_END)
    module = types.ModuleType(module_name)
    module.__file__ = str(UPSTREAM_FILE)
    sys.modules[module_name] = module
    # Line 3 of the pinned file reads sys.argv[0] to log its own source.
    saved_argv0 = sys.argv[0] if sys.argv else None
    if sys.argv:
        sys.argv[0] = str(UPSTREAM_FILE)
    else:
        sys.argv.append(str(UPSTREAM_FILE))
    try:
        exec(compile(source[:end], str(UPSTREAM_FILE), "exec"), module.__dict__)
    finally:
        if saved_argv0 is None:
            sys.argv.pop()
        else:
            sys.argv[0] = saved_argv0
    return module


_UPSTREAM = None


def load_upstream():
    global _UPSTREAM
    if _UPSTREAM is None:
        _UPSTREAM = exec_upstream()
    return _UPSTREAM


def original_config(up):
    hp = dataclasses.asdict(up.args)
    tokens_per_step = hp["batch_size"] * hp["sequence_length"]
    return {
        **hp,
        "tokens_per_step": tokens_per_step,
        "target_tokens": tokens_per_step * hp["num_iterations"],
        "micro_batch_tokens": hp["sequence_length"],
        "accumulation_steps_on_1_gpu": hp["batch_size"],
        "use_ddp": True,
        "designed_world_size": DESIGNED_WORLD_SIZE,
    }


def resolve_config(orig, micro_batch_tokens=None, no_ddp=False, val_loss_every=None, world_size=1):
    T = orig["sequence_length"] if micro_batch_tokens is None else micro_batch_tokens
    problems = []
    if T <= 0 or T % 128:
        problems.append(f"--micro-batch-tokens must be a positive multiple of 128 (FlexAttention block size), got {T}")
    elif orig["tokens_per_step"] % (T * world_size):
        problems.append(f"--micro-batch-tokens {T} must divide the {orig['tokens_per_step']:,} tokens per optimizer step")
    elif orig["val_tokens"] % (T * world_size):
        problems.append(f"--micro-batch-tokens {T} must divide val_tokens={orig['val_tokens']:,}")
    if val_loss_every is not None and val_loss_every < 0:
        problems.append("--val-loss-every must be >= 0")
    if problems:
        raise ValueError("; ".join(problems))
    return {
        "micro_batch_tokens": T,
        "world_size": world_size,
        "train_accumulation_steps": orig["tokens_per_step"] // (T * world_size),
        "val_steps": orig["val_tokens"] // (T * world_size),
        "tokens_per_step": orig["tokens_per_step"],
        "target_tokens": orig["target_tokens"],
        "num_iterations": orig["num_iterations"],
        "use_ddp": not no_ddp,
        "val_loss_every": orig["val_loss_every"] if val_loss_every is None else val_loss_every,
    }


def list_deviations(cfg, orig):
    """Settings that differ from the pinned configuration; each one needs --accept-deviations."""
    out = []
    if cfg["micro_batch_tokens"] != orig["sequence_length"]:
        out.append({
            "setting": "sequence_length (tokens per packed FlexAttention micro-batch)",
            "original": orig["sequence_length"],
            "used": cfg["micro_batch_tokens"],
            "consequence": (f"gradient accumulation {orig['accumulation_steps_on_1_gpu']} -> "
                            f"{cfg['train_accumulation_steps']} micro-batches per optimizer step on 1 GPU; "
                            f"tokens per step unchanged ({cfg['tokens_per_step']:,}); validation in "
                            f"{cfg['val_steps']} chunks of {cfg['micro_batch_tokens']} instead of "
                            f"{orig['val_tokens'] // orig['sequence_length']} chunks of {orig['sequence_length']}"),
            "impact": ("Model, optimizers, tokens per step, number of steps and LR/momentum/window schedules are "
                       "unchanged; the loader reads the same token stream, apart from the shard-tail tokens it skips "
                       f"(~{skipped_tail(orig['sequence_length']) / 1000:.0f}k tokens per shard at "
                       f"{orig['sequence_length']:,}, {skipped_tail(cfg['micro_batch_tokens']):,} at "
                       f"{cfg['micro_batch_tokens']:,}). Attention context is cut at every micro-batch boundary "
                       "(attention never spans more than 1792 tokens or crosses documents anyway), so documents "
                       "straddling a boundary lose earlier context: the loss curve and the validation loss are not "
                       "strictly comparable with the record. Gradients of the bf16 parameters (embeddings, scalars) "
                       f"are accumulated in bf16 over {cfg['train_accumulation_steps']} micro-batches instead of "
                       f"{orig['accumulation_steps_on_1_gpu']}, which adds rounding error to those gradients. More "
                       "micro-batches add per-micro-batch overhead (kernel launches, the original loader's synchronous "
                       "host-to-device copy, create_block_mask), so measured throughput is slightly pessimistic; "
                       "attention FLOPs are slightly lower."),
        })
    if not cfg["use_ddp"]:
        out.append({
            "setting": "DistributedDataParallel wrapper",
            "original": "model = DDP(model, device_ids=[ddp_local_rank])",
            "used": "no wrapper (single process, world_size=1)",
            "consequence": "frees DDP's 1.40 GiB duplicate gradient buckets; one compiled graph instead of per-bucket graphs",
            "impact": ("With world_size=1 DDP's gradient all-reduce is an identity, so gradients and updates are "
                       "numerically the same. Muon's own dist.all_reduce still runs unchanged. Removing DDP also "
                       "removes its per-bucket graph splitting and gradient copies, so steps can be faster than the "
                       "DDP-wrapped code would be on one GPU: with this change the timing bias is not guaranteed to "
                       "be pessimistic."),
        })
    if cfg["val_loss_every"] != orig["val_loss_every"]:
        out.append({
            "setting": "val_loss_every",
            "original": orig["val_loss_every"],
            "used": cfg["val_loss_every"],
            "consequence": "validation runs at a different frequency (0 = only at the final step)",
            "impact": "Training computation is identical; only time spent on validation changes.",
        })
    return out


def skipped_tail(T, ntok=SHARD_TOKENS):
    return ntok - shard_usable_tokens(ntok, T)


def config_label(c):
    return (f"{c['micro_batch_tokens']:,}-token micro-batches × {c['train_accumulation_steps']} accumulation steps, "
            f"{'DDP' if c['use_ddp'] else 'no DDP'}")


def non_training_differences(cfg):
    out = [
        f"Runs on 1 GPU (world_size=1) instead of the designed 8; the pinned code itself handles this via "
        f"gradient accumulation (batch_size // world_size). All Muon Newton-Schulz work happens on one GPU.",
        "Started as a plain Python process (RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT set in-process) instead of "
        "torchrun; same NCCL process group.",
        f"PyTorch {torch.__version__} (CUDA {torch.version.cuda}) instead of {REFERENCE_8XH100['torch']}: the "
        f"2024 nightly is not a stable release and has no kernels for newer GPUs (Blackwell sm_120 needs CUDA >= 12.8).",
        "Each optimizer step is timed with torch.cuda.synchronize() before and after (data loading, all "
        f"micro-batches, optimizer steps); the first {cfg.get('warmup_steps', '?')} steps (compilation/warm-up) are "
        "excluded from throughput (the original excludes its first 10 steps).",
        "Validation and checkpoint time are measured separately and excluded from step time (as in the original).",
        "Model checkpoint: saved only with --save-checkpoint-on-exit (the original always saves at the final step).",
        "Stops gracefully on Ctrl+C / SIGTERM / temperature limit after the current micro-batch; the partial "
        "optimizer step is discarded and not counted.",
        "Logs and reports are written to reports/<run_id>/ instead of logs/<uuid>.txt.",
    ]
    if cfg.get("window_calibration", True):
        out.append("After a graceful Ctrl+C/SIGTERM stop, a short calibration times real-data micro-batches at three "
                   f"attention windows ({CALIBRATION_METHOD}). It never updates the model.")
    if ALLOC_CONF_SET_BY_SCRIPT:
        out.append("Allocator setting PYTORCH_ALLOC_CONF=expandable_segments:True (reduces fragmentation; "
                   "does not change any computation).")
    return out


def build_optimizers(raw_model, Muon):
    # Verbatim from upstream/train_gpt2.py lines 450-457.
    optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight, raw_model.transformer.vte.weight], lr=0.6, betas=(0.8, 0.95), fused=True)
    optimizer2 = torch.optim.Adam([raw_model.lm_head.weight], lr=0.008, betas=(0.8, 0.95), fused=True)
    params = list(raw_model.transformer.h.parameters())
    matrix_params = [p for p in params if p.ndim == 2]
    scalar_params = [p for p in params if p.ndim < 2] + [raw_model.skip_weights]
    optimizer3 = Muon(matrix_params, lr=0.05, momentum=0.95)
    optimizer4 = torch.optim.Adam(scalar_params, lr=0.04, betas=(0.8, 0.95), fused=True)
    optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]
    return optimizers


def optimizer_partition(raw_model):
    """(Adam-managed, Muon-managed) parameters, grouped as in build_optimizers()."""
    params = list(raw_model.transformer.h.parameters())
    matrix_params = [p for p in params if p.ndim == 2]
    scalar_params = [p for p in params if p.ndim < 2] + [raw_model.skip_weights]
    adam = [raw_model.transformer.wte.weight, raw_model.transformer.vte.weight, raw_model.lm_head.weight] + scalar_params
    return adam, matrix_params


def make_get_lr(args):
    # Verbatim from upstream/train_gpt2.py lines 459-470.
    def get_lr(it):
        assert it <= args.num_iterations
        if it < args.warmup_iters:
            return (it+1) / args.warmup_iters
        elif it < args.num_iterations - args.cooldown_iters:
            return 1.0
        else:
            decay_ratio = (args.num_iterations - it) / args.cooldown_iters
            return decay_ratio
    return get_lr


def construct_model(up):
    GPT, GPTConfig = up.GPT, up.GPTConfig
    # Verbatim from upstream/train_gpt2.py lines 436-437.
    num_vocab = 50304
    model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768))
    return model


# ----------------------------------------------------------------------------- memory preflight


def model_memory(up):
    """Exact parameter/optimizer-state accounting from a meta-device model (no allocation, no CUDA)."""
    with torch.device("meta"):
        model = construct_model(up)
    model = model.bfloat16()
    for m in model.modules():
        if isinstance(m, up.CastedLinear):
            m.float()

    def nbytes(ps):
        return sum(p.numel() * p.element_size() for p in ps)

    params = list(model.parameters())
    adam, muon = optimizer_partition(model)
    by_dtype = {}
    for p in params:
        name = str(p.dtype).replace("torch.", "")
        by_dtype[name] = by_dtype.get(name, 0) + p.numel()
    matmul = sum(m.weight.numel() for m in model.modules() if isinstance(m, up.CastedLinear))
    wte, vte = model.transformer.wte, model.transformer.vte
    return {
        "n_params": sum(p.numel() for p in params),
        "params_by_dtype": by_dtype,
        "param_bytes": nbytes(params),
        "adam_state_bytes": 2 * nbytes(adam),
        "muon_state_bytes": nbytes(muon),
        "muon_update_buffer_bytes": 2 * sum(p.numel() for p in muon),
        "embedding_params": wte.weight.numel() + vte.weight.numel(),
        "matmul_params": matmul,
        "flops_per_token_matmul": 6 * matmul,
        "n_layer": len(model.transformer.h),
        "n_embd": wte.embedding_dim,
        "n_head": model.transformer.h[0].attn.n_head,
        "vocab_size": wte.num_embeddings,
        "value_embed_width": vte.embedding_dim,
    }


def activation_bytes_per_token(mm, c_block, head_bytes):
    d = mm["n_embd"]
    return mm["n_layer"] * c_block * d * 2 + head_bytes * mm["vocab_size"] + (mm["value_embed_width"] + 2 * d) * 2


def estimate_memory(mm, micro_batch_tokens, use_ddp):
    P = mm["param_bytes"]
    ddp = P if use_ddp else 0
    persistent = 2 * P + ddp + mm["adam_state_bytes"] + mm["muon_state_bytes"]
    optimizer_step_peak = persistent + mm["muon_update_buffer_bytes"]
    # Dense backward of wte + vte: an fp32 V x (12d + d) buffer plus its bf16 copy while every gradient is alive.
    embedding_backward_peak = persistent + mm["vocab_size"] * (mm["value_embed_width"] + mm["n_embd"]) * (4 + 2)
    est = {name: max(persistent + micro_batch_tokens * activation_bytes_per_token(mm, c, h), optimizer_step_peak,
                     embedding_backward_peak)
           for name, (c, h) in ACTIVATION_COEFFS.items()}
    # First micro-batch of the first step: no gradients and no optimizer state exist yet.
    lower_bound = P + ddp + micro_batch_tokens * activation_bytes_per_token(mm, MIN_SAVED_BLOCK, MIN_HEAD_BYTES)
    return {"persistent_bytes": persistent, "lower_bound_bytes": lower_bound, **est,
            "embedding_backward_transient_bytes": embedding_backward_peak - persistent,
            "logits_bf16_bytes": micro_batch_tokens * mm["vocab_size"] * 2}


def memory_verdict(est, budget_bytes):
    if est["lower_bound_bytes"] > budget_bytes:
        return "impossible"
    if est["central"] * FRAGMENTATION_FACTOR > budget_bytes:
        return "implausible"
    if est["upper"] * FRAGMENTATION_FACTOR > budget_bytes:
        return "plausible-tight"
    return "expected-to-fit"


def run_preflight(mm, vram, cfg, orig):
    budget = vram["total_bytes"] - vram["used_bytes"] - CUDA_CONTEXT_RESERVE_GIB * GIB
    rows = []
    for T in sorted(set(CANDIDATE_MICRO_BATCHES) | {cfg["micro_batch_tokens"]}, reverse=True):
        for use_ddp in (True, False):
            est = estimate_memory(mm, T, use_ddp)
            rows.append({
                "micro_batch_tokens": T,
                "use_ddp": use_ddp,
                "accumulation_steps": cfg["tokens_per_step"] // T,
                "lower_bound_gib": est["lower_bound_bytes"] / GIB,
                "lower_gib": est["lower"] / GIB,
                "central_gib": est["central"] / GIB,
                "upper_gib": est["upper"] / GIB,
                "persistent_gib": est["persistent_bytes"] / GIB,
                "logits_bf16_gib": est["logits_bf16_bytes"] / GIB,
                "verdict": memory_verdict(est, budget),
                "is_original": T == orig["sequence_length"] and use_ddp,
                "is_selected": T == cfg["micro_batch_tokens"] and use_ddp == cfg["use_ddp"],
            })
    P = mm["param_bytes"]
    return {
        "method": "analytic (no GPU allocation); exact parameter/optimizer-state bytes, estimated activations",
        "vram_total_gib": vram["total_bytes"] / GIB,
        "vram_used_by_others_gib": vram["used_bytes"] / GIB,
        "cuda_context_reserve_gib": CUDA_CONTEXT_RESERVE_GIB,
        "fragmentation_factor": FRAGMENTATION_FACTOR,
        "budget_gib": budget / GIB,
        "activation_coefficients": {k: {"block_n_embd_values_per_layer": c, "head_bytes_per_vocab_entry": h}
                                    for k, (c, h) in ACTIVATION_COEFFS.items()},
        "activation_mib_per_token": {k: activation_bytes_per_token(mm, c, h) / 2**20
                                     for k, (c, h) in ACTIVATION_COEFFS.items()},
        "persistent_breakdown_gib": {
            "params": P / GIB, "grads": P / GIB, "ddp_buckets": P / GIB,
            "adam_states": mm["adam_state_bytes"] / GIB, "muon_momentum": mm["muon_state_bytes"] / GIB,
        },
        "embedding_backward_transient_gib": estimate_memory(mm, cfg["micro_batch_tokens"], True)["embedding_backward_transient_bytes"] / GIB,
        "rows": rows,
        "selected": next(r for r in rows if r["is_selected"]),
        "original": next(r for r in rows if r["is_original"]),
    }


def format_preflight(pf, gpu_name):
    b = pf["persistent_breakdown_gib"]
    lines = [
        f"Memory preflight ({pf['method']})",
        f"  GPU: {gpu_name}: {pf['vram_total_gib']:.2f} GiB total, {pf['vram_used_by_others_gib']:.2f} GiB in use by other processes",
        f"  budget for PyTorch = total - in use - {pf['cuda_context_reserve_gib']:.2f} GiB CUDA context = {pf['budget_gib']:.2f} GiB "
        f"(estimates x{pf['fragmentation_factor']} for fragmentation before comparing)",
        f"  persistent state from step 2 on: params {b['params']:.2f} + grads {b['grads']:.2f} + DDP buckets {b['ddp_buckets']:.2f} "
        f"+ Adam {b['adam_states']:.2f} + Muon {b['muon_momentum']:.2f} = "
        f"{b['params'] + b['grads'] + b['ddp_buckets'] + b['adam_states'] + b['muon_momentum']:.2f} GiB "
        f"({b['params'] + b['grads'] + b['adam_states'] + b['muon_momentum']:.2f} without DDP)",
        "  activations per token (MiB): " + ", ".join(f"{k} {v:.3f}" for k, v in pf["activation_mib_per_token"].items())
        + f"; embedding-backward transient {pf['embedding_backward_transient_gib']:.2f} GiB on top of the persistent state",
        "",
        "  micro-batch  accum  DDP  hard-lower-bound  central  upper   verdict",
    ]
    for r in pf["rows"]:
        tags = [t for t, on in (("original", r["is_original"]), ("selected", r["is_selected"])) if on]
        lines.append((f"  {r['micro_batch_tokens']:>11}  {r['accumulation_steps']:>5}  {'yes' if r['use_ddp'] else 'no ':<3}"
                      f"  {r['lower_bound_gib']:>12.2f} GiB  {r['central_gib']:>7.2f}  {r['upper_gib']:>5.2f}   "
                      f"{r['verdict']:<16}{'  <- ' + ', '.join(tags) if tags else ''}").rstrip())
    return "\n".join(lines)


# ----------------------------------------------------------------------------- telemetry & stop control


class StopController:
    def __init__(self):
        self.reason = None
        self.detail = ""
        self._signals = 0
        self._lock = threading.Lock()

    @property
    def requested(self):
        return self.reason is not None

    def request(self, reason, detail=""):
        with self._lock:
            if self.reason is not None:
                return False
            self.reason, self.detail = reason, detail
            return True

    def install(self):
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    def _on_signal(self, signum, frame):
        self._signals += 1
        if self._signals > 1:
            raise KeyboardInterrupt
        name = signal.Signals(signum).name
        self.request("user-interrupt" if signum == signal.SIGINT else "sigterm", f"{name} received")
        print(f"\n{name} received: stopping after the current micro-batch (the partial optimizer step is discarded). "
              "During compilation this waits until compilation finishes. A second Ctrl+C aborts as soon as Python "
              "regains control (during the short post-stop window calibration it only skips the calibration).",
              flush=True)


THROTTLE_REASONS = {"power_cap": 0x4, "hw_slowdown": 0x8, "sw_thermal": 0x20, "hw_thermal": 0x40, "hw_power_brake": 0x80}


class GpuTelemetry:
    """Read-only GPU telemetry via NVML (fallback: nvidia-smi). Never changes clocks, power or voltage."""

    def __init__(self, index=0):
        self.index = index
        self.backend = None
        self._nvml = self._handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            self._nvml = pynvml
            self.backend = "nvml"
        except Exception:
            if shutil.which("nvidia-smi"):
                self.backend = "nvidia-smi"
        self._samples = deque(maxlen=8192)
        self._lock = threading.Lock()
        self._halt = threading.Event()
        self._thread = None
        self.max_temp_seen = None

    def _call(self, name, *args):
        try:
            value = getattr(self._nvml, name)(self._handle, *args) if self._handle is not None else None
            return value.decode() if isinstance(value, bytes) else value
        except Exception:
            return None

    def _smi(self, fields):
        try:
            out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits",
                                  "-i", str(self.index)], capture_output=True, text=True, timeout=10).stdout
            vals = [v.strip() for v in out.strip().split(",")]
            return dict(zip(fields, vals)) if len(vals) == len(fields) else {}
        except Exception:
            return {}

    @staticmethod
    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def static_info(self):
        if self.backend == "nvml":
            n = self._nvml
            mem = self._call("nvmlDeviceGetMemoryInfo")
            cc = self._call("nvmlDeviceGetCudaComputeCapability")
            cuda_drv = None
            with contextlib.suppress(Exception):
                v = n.nvmlSystemGetCudaDriverVersion()
                cuda_drv = f"{v // 1000}.{(v % 1000) // 10}"
            driver = None
            with contextlib.suppress(Exception):
                driver = n.nvmlSystemGetDriverVersion()
                driver = driver.decode() if isinstance(driver, bytes) else driver
            mw = lambda x: None if x is None else x / 1000
            return {
                "name": self._call("nvmlDeviceGetName"),
                "vram_total_mib": None if mem is None else mem.total / 2**20,
                "compute_capability": None if cc is None else f"{cc[0]}.{cc[1]}",
                "driver_version": driver,
                "cuda_driver_api": cuda_drv,
                "cuda_cores": self._call("nvmlDeviceGetNumGpuCores"),
                "memory_bus_width_bits": self._call("nvmlDeviceGetMemoryBusWidth"),
                "max_sm_clock_mhz": self._call("nvmlDeviceGetMaxClockInfo", n.NVML_CLOCK_SM),
                "max_mem_clock_mhz": self._call("nvmlDeviceGetMaxClockInfo", n.NVML_CLOCK_MEM),
                "enforced_power_limit_w": mw(self._call("nvmlDeviceGetEnforcedPowerLimit")),
                "default_power_limit_w": mw(self._call("nvmlDeviceGetPowerManagementDefaultLimit")),
                "telemetry_backend": "nvml",
            }
        if self.backend == "nvidia-smi":
            f = self._smi(["name", "memory.total", "compute_cap", "driver_version", "clocks.max.sm", "power.limit"])
            return {"name": f.get("name"), "vram_total_mib": self._num(f.get("memory.total")),
                    "compute_capability": f.get("compute_cap"), "driver_version": f.get("driver_version"),
                    "max_sm_clock_mhz": self._num(f.get("clocks.max.sm")),
                    "enforced_power_limit_w": self._num(f.get("power.limit")), "telemetry_backend": "nvidia-smi"}
        return {"name": None, "telemetry_backend": None}

    def memory(self):
        if self.backend == "nvml":
            mem = self._call("nvmlDeviceGetMemoryInfo")
            if mem is not None:
                return {"total_bytes": mem.total, "used_bytes": mem.used}
        if self.backend == "nvidia-smi":
            f = self._smi(["memory.total", "memory.used"])
            if f:
                return {"total_bytes": float(f["memory.total"]) * 2**20, "used_bytes": float(f["memory.used"]) * 2**20}
        return None

    def sample(self):
        s = {"t": time.monotonic(), "temp_c": None, "util_pct": None, "power_w": None, "sm_clock_mhz": None, "throttle": []}
        if self.backend == "nvml":
            n = self._nvml
            if hasattr(n, "nvmlDeviceGetTemperatureV"):
                s["temp_c"] = self._call("nvmlDeviceGetTemperatureV", n.NVML_TEMPERATURE_GPU)
            if s["temp_c"] is None:
                s["temp_c"] = self._call("nvmlDeviceGetTemperature", n.NVML_TEMPERATURE_GPU)
            util = self._call("nvmlDeviceGetUtilizationRates")
            s["util_pct"] = None if util is None else util.gpu
            power = self._call("nvmlDeviceGetPowerUsage")
            s["power_w"] = None if power is None else power / 1000
            s["sm_clock_mhz"] = self._call("nvmlDeviceGetClockInfo", n.NVML_CLOCK_SM)
            mask = self._call("nvmlDeviceGetCurrentClocksEventReasons")
            if mask is None:
                mask = self._call("nvmlDeviceGetCurrentClocksThrottleReasons")
            if mask is not None:
                s["throttle"] = [k for k, bit in THROTTLE_REASONS.items() if mask & bit]
        elif self.backend == "nvidia-smi":
            f = self._smi(["temperature.gpu", "utilization.gpu", "power.draw", "clocks.sm"])
            s.update(temp_c=self._num(f.get("temperature.gpu")), util_pct=self._num(f.get("utilization.gpu")),
                     power_w=self._num(f.get("power.draw")), sm_clock_mhz=self._num(f.get("clocks.sm")))
        if s["temp_c"] is not None:
            self.max_temp_seen = max(self.max_temp_seen or s["temp_c"], s["temp_c"])
        return s

    def start(self, interval_s, max_temp_c, stop):
        if self.backend is None:
            print("warning: GPU temperature monitoring unavailable (no NVML / nvidia-smi); thermal stop disabled.")
            return

        def loop():
            near_warned = failure_warned = False
            failures = 0
            while True:
                s = self.sample()
                with self._lock:
                    self._samples.append(s)
                temp = s["temp_c"]
                if temp is None:
                    failures += 1
                    if failures >= 5 and not failure_warned:
                        print("\nWARNING: 5 consecutive GPU temperature reads failed; the thermal stop cannot "
                              "trigger until reads recover. Watch the temperature yourself or press Ctrl+C.", flush=True)
                        failure_warned = True
                else:
                    failures = 0
                    if temp > max_temp_c:
                        if stop.request("thermal", f"GPU temperature {temp} °C exceeded --max-temp {max_temp_c} °C"):
                            print(f"\nWARNING: GPU temperature {temp} °C > limit {max_temp_c} °C: stopping after the "
                                  "current micro-batch.", flush=True)
                    elif temp >= max_temp_c - 5 and not near_warned:
                        print(f"\nwarning: GPU at {temp} °C, within 5 °C of the {max_temp_c} °C stop limit.", flush=True)
                        near_warned = True
                    elif temp < max_temp_c - 8:
                        near_warned = False
                if self._halt.wait(interval_s):
                    return

        self._thread = threading.Thread(target=loop, name="gpu-telemetry", daemon=True)
        self._thread.start()

    def stop(self):
        self._halt.set()
        if self._thread is not None:
            self._thread.join(timeout=15)

    def window(self, t0, t1):
        with self._lock:
            ss = [s for s in self._samples if t0 <= s["t"] <= t1]
            if not ss and self._samples:
                ss = [self._samples[-1]]

        def mean(key):
            v = [s[key] for s in ss if s[key] is not None]
            return statistics.fmean(v) if v else None

        temps = [s["temp_c"] for s in ss if s["temp_c"] is not None]
        throttled = lambda keys: (100 * sum(any(k in s["throttle"] for k in keys) for s in ss) / len(ss)) if ss else None
        return {
            "gpu_util_mean_pct": mean("util_pct"),
            "temp_max_c": max(temps) if temps else None,
            "power_mean_w": mean("power_w"),
            "sm_clock_mean_mhz": mean("sm_clock_mhz"),
            "power_capped_pct": throttled(("power_cap", "hw_power_brake")),
            "thermal_throttled_pct": throttled(("sw_thermal", "hw_thermal", "hw_slowdown")),
            "telemetry_samples": len(ss),
        }


def nvml_index():
    first = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return int(first) if first.isdigit() else 0


def power_source():
    info = {"on_ac_power": None, "battery_status": None, "battery_capacity_pct": None}
    for dev in glob.glob("/sys/class/power_supply/*"):
        read = lambda name: Path(dev, name).read_text().strip() if Path(dev, name).exists() else None
        with contextlib.suppress(OSError):
            kind = read("type")
            if kind == "Mains" and read("online") is not None:
                info["on_ac_power"] = bool(info["on_ac_power"]) or read("online") == "1"
            elif kind == "Battery":
                info["battery_status"] = read("status")
                cap = read("capacity")
                info["battery_capacity_pct"] = int(cap) if cap and cap.isdigit() else None
    return info


def host_environment():
    cpu = None
    with contextlib.suppress(OSError):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    ram = None
    with contextlib.suppress(OSError, ValueError, IndexError):
        ram = int(Path("/proc/meminfo").read_text().split("\n")[0].split()[1]) / 2**20
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "platform": platform.platform(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "ram_gib": ram,
        "allocator_conf": os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        **power_source(),
    }


# ----------------------------------------------------------------------------- data & OOM registry


def shard_usable_tokens(ntok, T):
    # The pinned loader reads T+1 tokens at positions 0, T, 2T, ... and moves on once pos + T >= ntok.
    return (math.ceil(ntok / T) - 1) * T


def check_data(up, cfg, orig):
    train_files = sorted(glob.glob(str(UPSTREAM_DIR / orig["input_bin"])))
    val_files = sorted(glob.glob(str(UPSTREAM_DIR / orig["input_val_bin"])))
    info = {"train_shards": len(train_files), "val_shards": len(val_files), "train_tokens": 0,
            "steps_covered": 0, "wraps_at_step": None, "error": None, "warning": None}
    if not train_files or not val_files:
        info["error"] = f"FineWeb shards missing under upstream/data/fineweb10B/; download them with: {DOWNLOAD_CMD}"
        return info
    T = cfg["micro_batch_tokens"]
    ntoks = [int(up._peek_data_shard(f)) for f in train_files]
    info["train_tokens"] = sum(ntoks)
    micro_batches = sum(shard_usable_tokens(n, T) // T for n in ntoks)
    info["steps_covered"] = micro_batches // cfg["train_accumulation_steps"]
    if info["steps_covered"] < cfg["num_iterations"]:
        info["wraps_at_step"] = info["steps_covered"] + 1
        info["warning"] = (f"only {len(train_files)} train shard(s) ({info['train_tokens']:,} tokens) present; the "
                           f"original loader wraps around to the first shard at step ~{info['wraps_at_step']} of "
                           f"{cfg['num_iterations']}. Throughput is unaffected, but a run past that point repeats data. "
                           f"Full budget: {DOWNLOAD_CMD}")
    return info


def oom_key(cfg, gpu):
    return {"micro_batch_tokens": cfg["micro_batch_tokens"], "use_ddp": cfg["use_ddp"], "gpu": gpu.get("name"),
            "vram_total_mib": round(gpu.get("vram_total_mib") or 0), "torch": torch.__version__}


def find_oom_failure(registry_path, key):
    with contextlib.suppress(OSError, ValueError):
        for entry in json.loads(Path(registry_path).read_text()):
            if entry.get("key") == key:
                return entry
    return None


def record_oom_failure(registry_path, key, details):
    entries = []
    with contextlib.suppress(OSError, ValueError):
        entries = json.loads(Path(registry_path).read_text())
    entries.append({"key": key, "time_utc": utc_now(), **details})
    atomic_write(Path(registry_path), json.dumps(entries, indent=2))


def is_oom(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, torch.OutOfMemoryError):
            return True
        if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
            return True
        exc = exc.__cause__ or exc.__context__
    return False


# ----------------------------------------------------------------------------- statistics & estimates


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def fmt_duration(seconds):
    if seconds is None:
        return "n/a"
    return f"{seconds / 3600:.2f} h ({seconds / 86400:.2f} d)"


def n_evals_full_run(num_iterations, val_loss_every):
    if val_loss_every <= 0:
        return 1
    return len(set(range(0, num_iterations + 1, val_loss_every)) | {num_iterations})


def summarize(steps, warmup_steps, ma_window, target_tokens, flops_per_token=None):
    """Throughput, ETA and uncertainty from completed optimizer steps (token-based, warm-up excluded)."""
    tokens_done = sum(s["tokens"] for s in steps)
    timed = steps[warmup_steps:]
    out = {"completed_steps": len(steps), "tokens_done": tokens_done, "warmup_steps_excluded": min(len(steps), warmup_steps),
           "timed_steps": len(timed), "fraction_of_budget_done": tokens_done / target_tokens}
    if not timed:
        return out
    times = [s["step_time_s"] for s in timed]
    n = len(times)

    def tput(recs):
        return sum(r["tokens"] for r in recs) / sum(r["step_time_s"] for r in recs)

    window = timed[-ma_window:]
    tput_ma, tput_all = tput(window), tput(timed)
    remaining = max(target_tokens - tokens_done, 0)
    w = min(ma_window, n)
    window_tputs = [tput(timed[i:i + w]) for i in range(n - w + 1)]
    sd = statistics.stdev(times) if n > 1 else 0.0
    out.update({
        "step_time_mean_s": statistics.fmean(times),
        "step_time_median_s": statistics.median(times),
        "step_time_sd_s": sd,
        "step_time_cv": sd / statistics.fmean(times),
        "step_time_min_s": min(times),
        "step_time_max_s": max(times),
        "ma_window": len(window),
        "step_time_ma_s": statistics.fmean(s["step_time_s"] for s in window),
        "throughput_ma_tok_s": tput_ma,
        "throughput_mean_tok_s": tput_all,
        "eta_remaining_s": remaining / tput_ma,
        "total_steady_state_s": target_tokens / tput_ma,
        "total_steady_state_cumulative_s": target_tokens / tput_all,
        "total_fastest_window_s": target_tokens / max(window_tputs),
        "total_slowest_window_s": target_tokens / min(window_tputs),
        "total_ci95_s": None,
        "drift_pct": None,
    })
    if len(window) > 1:
        # Centred on the MA estimate with the MA window's sd and n; normal approximation, independent step times.
        wt = [r["step_time_s"] for r in window]
        rel_half = 1.96 * statistics.stdev(wt) / math.sqrt(len(wt)) / statistics.fmean(wt)
        out["step_time_ma_sd_s"] = statistics.stdev(wt)
        out["total_ci95_s"] = [out["total_steady_state_s"] * max(1 - rel_half, 0.0), out["total_steady_state_s"] * (1 + rel_half)]
    if n >= 6:
        k = n // 3
        out["drift_pct"] = 100 * (statistics.fmean(times[-k:]) / statistics.fmean(times[:k]) - 1)
    if flops_per_token:
        out["approx_tflops_matmul"] = flops_per_token * tput_ma / 1e12
    return out


def schedule_windows(num_iterations):
    """attn_blocksize of every training step, computed as in the pinned loop (upstream line 490)."""
    return [int(64*((step/num_iterations * (1792 - 64) + 64)//64)) for step in range(num_iterations)]


def interpolate(points, w):
    """Piecewise-linear t(w) through sorted (window, seconds) points, clamped outside the measured range."""
    if w <= points[0][0]:
        return points[0][1]
    for (w0, t0), (w1, t1) in zip(points, points[1:]):
        if w <= w1:
            return t0 + (t1 - t0) * (w - w0) / (w1 - w0)
    return points[-1][1]


def window_correction(ma_step_s, accum, points, w_measured, num_iterations, steps_done=0):
    """Additive per-step correction for the attention window growing over the schedule."""
    unique = {}
    for p in points:
        unique.setdefault(p["window"], p["median_s"])
    pts = sorted(unique.items())
    t_ref = interpolate(pts, w_measured)
    per_step = [ma_step_s + accum * (interpolate(pts, w) - t_ref) for w in schedule_windows(num_iterations)]
    total = num_iterations * statistics.fmean(per_step)
    return {"mean_step_s": statistics.fmean(per_step), "total_s": total, "remaining_s": sum(per_step[steps_done:]),
            "delta_total_s": total - num_iterations * ma_step_s, "w_measured": w_measured,
            "per_micro_batch_s": {str(w): t for w, t in pts}}


def steady_eval_seconds(e):
    # The first validation micro-batch carries the eval-graph compilation; extrapolate from the others.
    n = e.get("val_steps") or 0
    if e.get("first_microbatch_s") is None or n < 2:
        return e["duration_s"]
    return (e["duration_s"] - e["first_microbatch_s"]) * n / (n - 1)


def project_wall_clock(training_s, summary, steps, evals, setup_time_s, warmup_steps, n_evals):
    if "throughput_mean_tok_s" not in summary:
        return None
    warm = steps[:warmup_steps]
    compile_overhead = max(sum(s["step_time_s"] for s in warm) - sum(s["tokens"] for s in warm) / summary["throughput_mean_tok_s"], 0.0)
    steady = [steady_eval_seconds(e) for e in evals]
    eval_s = statistics.median(steady) if steady else None
    eval_compile_excess = max(evals[0]["duration_s"] - steady[0], 0.0) if evals else 0.0
    total = training_s + (setup_time_s or 0.0) + compile_overhead + (eval_s or 0.0) * n_evals + eval_compile_excess
    return {
        "total_s": total,
        "training_s": training_s,
        "setup_s": setup_time_s,
        "compile_warmup_overhead_s": compile_overhead,
        "eval_seconds_each": eval_s,
        "eval_compile_excess_s": eval_compile_excess,
        "evals_in_full_run": n_evals,
        "validation_not_measured": eval_s is None,
    }


def feasibility(report):
    est = report.get("estimates") or {}
    total = (est.get("wall_clock") or {}).get("total_s")
    training = est.get("total_training_s")
    timed = (report.get("summary") or {}).get("timed_steps", 0)
    status = report.get("status")
    orig_verdict = (report.get("preflight") or {}).get("original", {}).get("verdict")
    original_note = ("" if orig_verdict in RUNNABLE_VERDICTS else
                     "The original configuration (65,536-token micro-batches) cannot run on this GPU at all "
                     f"(preflight verdict: {orig_verdict}); no reliable runtime estimate can be obtained for it.")
    oom_verdict = None
    if status == "oom":
        oom_verdict = (f"did not complete (out of memory at step {(report.get('oom') or {}).get('step')}); "
                       f"throughput measured over {timed} steps")
    if total is None:
        if status == "oom":
            text = ("The selected configuration ran out of GPU memory before a timed step completed. No reliable "
                    "runtime estimate can be obtained for it.")
        elif status in ("refused-preflight", "preflight-only"):
            text = "No training step was executed, so no runtime has been measured."
        else:
            text = "Not enough completed steps to estimate the runtime."
        return {"verdict": oom_verdict or "undetermined", "text": f"{original_note} {text}".strip()}
    hours = total / 3600
    corrected = "window-corrected" if est.get("window_correction_applied") else "not window-corrected, likely low"
    run_text = (f"One complete pretraining run would take about {fmt_duration(total)} wall-clock "
                f"({fmt_duration(training)} of training only, {corrected}, plus validation and compilation) of "
                "continuous full GPU load.")
    if hours <= 24:
        verdict = "feasible-single-run"
        text = (f"{run_text} That is technically feasible for a single run, but it ties up the laptop for that long, "
                "and any research use (ablations, seeds, hyperparameter sweeps) multiplies it.")
    elif hours <= 24 * 7:
        verdict = "marginal"
        text = f"{run_text} That is possible only as a multi-day unattended run and is impractical for iterative research."
    else:
        verdict = "not-practical"
        text = f"{run_text} That is not practical on a laptop."
    if timed < MIN_STEPS_FOR_FIRM_VERDICT:
        verdict += " (provisional)"
        text += f" Provisional: based on only {timed} timed steps."
    if oom_verdict:
        verdict = oom_verdict
        text = f"The run ran out of GPU memory at step {(report.get('oom') or {}).get('step')}. {text}"
    applies = (" The estimate applies to the memory-compatible configuration listed in section 3."
               if report.get("deviations") else "")
    return {"verdict": verdict, "text": f"{text} {original_note}{applies}".replace("  ", " ")}


def uncertainty_notes(report):
    s = report.get("summary") or {}
    steps = (report.get("measurements") or {}).get("steps") or []
    notes = []
    if s.get("timed_steps", 0) and s["timed_steps"] < 20:
        notes.append(f"Only {s['timed_steps']} timed steps; the moving average is still noisy (let it run for 20+ steps).")
    if s.get("drift_pct") is not None:
        notes.append(f"Step time drift from the first to the last third of timed steps: {s['drift_pct']:+.1f}% "
                     "(positive = slowing down, e.g. thermal or power throttling).")
    if s.get("step_time_cv") is not None:
        notes.append(f"Step-time coefficient of variation: {100 * s['step_time_cv']:.1f}%.")
    if s.get("fraction_of_budget_done") is not None and steps:
        notes.append(f"Measured {100 * s['fraction_of_budget_done']:.2f}% of the token budget; the rest is extrapolated.")
    therm = [r["thermal_throttled_pct"] for r in steps if r.get("thermal_throttled_pct") is not None]
    power = [r["power_capped_pct"] for r in steps if r.get("power_capped_pct") is not None]
    if therm or power:
        notes.append("Clock-limiting reasons reported by the GPU, share of telemetry samples: thermal slowdown "
                     f"{_f(statistics.fmean(therm) if therm else None, '.0f', '%')}, power cap "
                     f"{_f(statistics.fmean(power) if power else None, '.0f', '%')}.")
    env = report.get("environment") or {}
    if env.get("on_ac_power") is False:
        notes.append("The laptop was on battery power; GPU power limits are usually lower on battery.")
    est = report.get("estimates") or {}
    wc = est.get("wall_clock") or {}
    evals = (report.get("measurements") or {}).get("evals") or []
    if len(evals) == 1:
        notes.append("Steady validation time is extrapolated from the first evaluation with its first (compiling) "
                     "micro-batch removed; that compile excess is added once.")
    if "total_training_s" in est and not est.get("window_correction_applied"):
        calib = report.get("window_calibration") or {}
        notes.append("The training-time estimate is NOT window-corrected "
                     f"({calib.get('status')}: {calib.get('reason')}); measured steps used early-schedule attention "
                     f"windows (mean over the schedule is {statistics.fmean(schedule_windows(1530)):.0f} tokens), so "
                     "it is likely low.")
    if wc.get("validation_not_measured"):
        notes.append("Validation time was not measured; the wall-clock projection excludes it.")
    data = report.get("data") or {}
    if data.get("wraps_at_step"):
        notes.append(f"Local data covers {data['steps_covered']} steps; a full run would wrap around (more shards needed).")
    return notes


BASE_LIMITATIONS = [
    "The estimate assumes the throughput of the most recent moving-average window holds for the whole run. Laptop "
    "GPUs can slow down further as the chassis heat-soaks over hours; the observed drift is reported above.",
    "Single GPU only: this says nothing about multi-GPU scaling or communication cost on a cluster.",
    "PyTorch/compiler version differs from the record (needed for Blackwell); kernel selection differs.",
    "The 95% interval assumes independent step times; step times are autocorrelated (thermal state), so it "
    "understates the true uncertainty. The fastest/slowest-window range is a more honest spread.",
    "Background load on the laptop (browser, display, other GPU users) and the power source affect throughput.",
]


def limitations(report):
    out = list(BASE_LIMITATIONS)
    if (report.get("estimates") or {}).get("window_correction_applied"):
        window = ("The attention window grows from 64 to 1792 tokens over the 1530 steps; measured steps sit early in "
                  "that schedule. The training estimate is corrected with a calibration measured "
                  f"{CALIBRATION_METHOD}. It excludes any window effect on the optimizer step and assumes the "
                  "per-micro-batch cost is linear between the calibrated windows.")
    else:
        window = ("The attention window grows from 64 to 1792 tokens over the 1530 steps and measured steps sit early in "
                  "that schedule; without the post-stop calibration the training estimate is likely low.")
    out.insert(1, window)
    if report.get("deviations"):
        out.insert(2, "The measured configuration differs from the original as listed in section 3, so kernel shapes, "
                      "per-micro-batch overheads and attention context at micro-batch boundaries differ from the "
                      "original 65,536-token micro-batches.")
    return out


def finalize(report):
    """Recomputes all derived sections of the report from the raw measurements."""
    cfg = report["effective_config"]
    meas = report["measurements"]
    report["summary"] = summarize(meas["steps"], cfg["warmup_steps"], cfg["ma_window"], cfg["target_tokens"],
                                  report["model"]["flops_per_token_matmul"])
    s = report["summary"]
    est = {"target_tokens": cfg["target_tokens"], "tokens_done": s["tokens_done"],
           "remaining_tokens": max(cfg["target_tokens"] - s["tokens_done"], 0)}
    if "throughput_ma_tok_s" in s:
        calib = report.get("window_calibration") or {}
        corr = None
        if calib.get("status") == "done":
            corr = window_correction(s["step_time_ma_s"], cfg["train_accumulation_steps"], calib["points"],
                                     calib["w_measured"], cfg["num_iterations"], s["completed_steps"])
        delta = corr["delta_total_s"] if corr else 0.0
        training = s["total_steady_state_s"] + delta
        shift = lambda pair: None if pair is None else [v + delta for v in pair]
        wall = project_wall_clock(training, s, meas["steps"], meas["evals"], meas.get("setup_time_s"),
                                  cfg["warmup_steps"], n_evals_full_run(cfg["num_iterations"], cfg["val_loss_every"]))
        est.update({
            "throughput_tok_s": s["throughput_ma_tok_s"],
            "window_correction_applied": corr is not None,
            "window_correction": corr,
            "total_training_s": training,
            "total_steady_state_s": s["total_steady_state_s"],
            "eta_remaining_s": corr["remaining_s"] if corr else s["eta_remaining_s"],
            "eta_remaining_uncorrected_s": s["eta_remaining_s"],
            "total_steady_state_cumulative_s": s["total_steady_state_cumulative_s"] + delta,
            "total_ci95_s": shift(s["total_ci95_s"]),
            "total_range_s": shift([s["total_fastest_window_s"], s["total_slowest_window_s"]]),
            "wall_clock": wall,
            "record_full_run_s": RECORD_FULL_RUN_S,
            "slowdown_vs_8xH100": training / RECORD_FULL_RUN_S,
            "time_for_record_significance_runs_s": REFERENCE_8XH100["runs"] * wall["total_s"],
        })
    report["estimates"] = est
    report["feasibility"] = feasibility(report)
    report["uncertainty_notes"] = uncertainty_notes(report)
    report["limitations"] = limitations(report)
    report["updated_utc"] = utc_now()
    return report


# ----------------------------------------------------------------------------- report rendering


def _f(v, spec=".2f", unit=""):
    return "n/a" if v is None else f"{v:{spec}}{unit}"


def render_markdown(report):
    o, c = report["original_config"], report["effective_config"]
    mm, pf, env, gpu = report["model"], report.get("preflight") or {}, report["environment"], report["gpu"]
    s, est, meas = report["summary"], report["estimates"], report["measurements"]
    ref = REFERENCE_8XH100
    measured = "throughput_tok_s" in est
    L = [f"# Training feasibility report: modded-nanogpt on {gpu.get('name') or 'unknown GPU'}", "",
         f"Run `{report['run_id']}` · status **{report['status']}**"
         + (f" (stop reason: {report['stop_reason']}: {report.get('stop_detail') or ''})" if report.get("stop_reason") else "")
         + f" · created {report['created_utc']} · updated {report['updated_utc']}", ""]

    L += ["## 1. Original training requirements", "",
          f"- Code: [{UPSTREAM_REPO}]({UPSTREAM_REPO}) `train_gpt2.py` at commit `{UPSTREAM_COMMIT}` "
          f"(sha256 `{report['upstream']['sha256']}`), record #14 \"Value Embeddings\".",
          f"- Model: {mm['n_layer']} layers, {mm['n_head']} heads, d={mm['n_embd']}, vocab {mm['vocab_size']:,}; "
          f"{mm['n_params']:,} parameters ({mm['embedding_params']:,} in token/value embeddings, "
          f"{mm['matmul_params']:,} in matmuls); {mm['param_bytes'] / GIB:.2f} GiB of weights "
          f"({', '.join(f'{k}: {v:,}' for k, v in mm['params_by_dtype'].items())}).",
          "- Optimizers: Adam (embeddings, lr 0.6), Adam (lm_head, lr 0.008), Muon (block matrices, lr 0.05, "
          "momentum 0.85→0.95), Adam (scalars, lr 0.04); betas (0.8, 0.95).",
          f"- Batch: {o['batch_size']} sequences × {o['sequence_length']:,} tokens = {o['tokens_per_step']:,} tokens per "
          f"step; {o['num_iterations']} steps (cooldown {o['cooldown_iters']}); "
          f"**target budget {o['target_tokens']:,} tokens** of FineWeb (GPT-2 tokens, 100M-token shards).",
          f"- Validation every {o['val_loss_every']} steps on {o['val_tokens']:,} tokens.",
          f"- Designed for {o['designed_world_size']} GPUs (`torchrun --nproc_per_node=8`). Reference: {ref['runs']} logged "
          f"runs on {ref['hardware']} took {ref['train_time_s_mean']:.1f} s of training time on average "
          f"(min {ref['train_time_s_min']}, max {ref['train_time_s_max']}), final val loss "
          f"{ref['final_val_loss_mean']} ± {ref['final_val_loss_sd']} ({ref['source']}); the upstream clock starts "
          f"after step 10, so all 1530 steps take ≈ {RECORD_FULL_RUN_S:.1f} s.",
          f"- Approximate compute: 6 × {mm['matmul_params']:,} ≈ {mm['flops_per_token_matmul'] / 1e9:.2f} GFLOP per "
          f"token (matmuls only) → {mm['flops_per_token_matmul'] * o['target_tokens'] / 1e18:.2f} EFLOP for the budget.", ""]

    L += ["## 2. Laptop GPU and environment", "",
          "| Property | Value |", "|---|---|"]
    rows = [("GPU", gpu.get("name")), ("VRAM", _f((gpu.get("vram_total_mib") or 0) / 1024, ".2f", " GiB") if gpu.get("vram_total_mib") else "n/a"),
            ("Compute capability", gpu.get("compute_capability")), ("Streaming multiprocessors", gpu.get("sm_count")),
            ("CUDA cores", gpu.get("cuda_cores")), ("Memory bus", _f(gpu.get("memory_bus_width_bits"), ".0f", " bit")),
            ("Max SM / memory clock (NVML)", f"{_f(gpu.get('max_sm_clock_mhz'), '.0f')} / {_f(gpu.get('max_mem_clock_mhz'), '.0f')} MHz"),
            ("Enforced power limit (read-only)", _f(gpu.get("enforced_power_limit_w"), ".0f", " W")),
            ("Driver / CUDA driver API", f"{gpu.get('driver_version')} / {gpu.get('cuda_driver_api')}"),
            ("PyTorch / CUDA runtime", f"{env['torch']} / {env['torch_cuda']}"), ("Python", env["python"]),
            ("CPU / RAM", f"{env.get('cpu')} ({env.get('cpu_count')} threads) / {_f(env.get('ram_gib'), '.1f', ' GiB')}"),
            ("Power source", {True: "AC adapter", False: "battery"}.get(env.get("on_ac_power"), "unknown")
             + (f" (battery {env.get('battery_capacity_pct')}%, {env.get('battery_status')})" if env.get("battery_status") else "")),
            ("OS", env["platform"])]
    L += [f"| {k} | {v if v is not None else 'n/a'} |" for k, v in rows] + [""]

    L += ["## 3. Changes made to fit the model", ""]
    if pf:
        orig_row = pf["original"]
        L += [f"**Original configuration:** preflight verdict **{orig_row['verdict']}**. Even the first micro-batch of "
              f"{o['sequence_length']:,} tokens needs at least {orig_row['lower_bound_gib']:.1f} GiB (weights, DDP "
              f"buckets, one bf16 logits tensor of {orig_row['logits_bf16_gib']:.2f} GiB and the minimum saved activations); "
              f"the central estimate for a steady-state step is {orig_row['central_gib']:.1f} GiB, against a budget of "
              f"{pf['budget_gib']:.2f} GiB on this GPU. The original was sized for 80 GB H100s.", ""]
        L += ["| micro-batch tokens | accumulation | DDP | hard lower bound (GiB) | central (GiB) | upper (GiB) | verdict |",
              "|---:|---:|:-:|---:|---:|---:|---|"]
        for r in pf["rows"]:
            tag = " (original)" if r["is_original"] else (" (selected)" if r["is_selected"] else "")
            L.append(f"| {r['micro_batch_tokens']:,}{tag} | {r['accumulation_steps']} | {'yes' if r['use_ddp'] else 'no'} | "
                     f"{r['lower_bound_gib']:.2f} | {r['central_gib']:.2f} | {r['upper_gib']:.2f} | {r['verdict']} |")
        L += ["", f"Budget = {pf['vram_total_gib']:.2f} GiB total − {pf['vram_used_by_others_gib']:.2f} GiB used by other "
              f"processes − {pf['cuda_context_reserve_gib']} GiB CUDA context; estimates are multiplied by "
              f"{pf['fragmentation_factor']} before comparison. Weights, gradients and optimizer states are exact; "
              "activation sizes are estimates (central/upper bracket what torch.compile typically saves).", ""]
    if report["deviations"]:
        L += ["**Training-configuration changes" + (" (explicitly accepted with `--accept-deviations`)" if c.get("deviations_accepted") else "") + ":**", ""]
        for d in report["deviations"]:
            L += [f"- **{d['setting']}**: `{d['original']}` → `{d['used']}`. {d['consequence']}. {d['impact']}"]
        L.append("")
    else:
        L += ["No training-configuration changes: the effective configuration equals the pinned one.", ""]
    stream = ("the same FineWeb token stream (apart from the skipped shard tails noted above)"
              if c["micro_batch_tokens"] != o["sequence_length"] else "FineWeb data and token order")
    L += ["Unchanged: model architecture and size, Muon/Adam implementations and parameter groups, learning rates and "
          f"schedules, {stream}, next-token cross-entropy loss, the 1530-step / "
          f"{o['target_tokens']:,}-token budget.", "", "Non-training differences:", ""]
    L += [f"- {d}" for d in report["non_training_differences"]] + [""]

    L += ["## 4. Measured throughput and peak memory", ""]
    if not meas["steps"]:
        L.append("No training step was completed in this run, so nothing was measured.")
        if pf and pf["original"]["verdict"] not in RUNNABLE_VERDICTS:
            L.append("**No reliable runtime estimate can be obtained for the original configuration**"
                     + (" or for the selected configuration." if report["status"] == "oom" else "."))
        L.append("")
    else:
        last = meas["steps"][-1]
        L += ["| Metric | Value |", "|---|---|",
              f"| Completed optimizer steps | {s['completed_steps']} of {c['num_iterations']} (first {s['warmup_steps_excluded']} excluded as warm-up) |",
              f"| Tokens processed | {s['tokens_done']:,} ({100 * s['fraction_of_budget_done']:.2f}% of budget) |",
              f"| Micro-batch / accumulation | {c['micro_batch_tokens']:,} tokens × {c['train_accumulation_steps']} = {c['tokens_per_step']:,} tokens per step |"]
        if s.get("timed_steps"):
            L += [f"| Throughput (last {s['ma_window']} steps) | {s['throughput_ma_tok_s']:,.0f} tokens/s |",
                  f"| Throughput (all timed steps) | {s['throughput_mean_tok_s']:,.0f} tokens/s |",
                  f"| Step time mean / median / sd | {s['step_time_mean_s']:.2f} / {s['step_time_median_s']:.2f} / {s['step_time_sd_s']:.2f} s |",
                  f"| Step time min / max | {s['step_time_min_s']:.2f} / {s['step_time_max_s']:.2f} s |",
                  f"| Approx. achieved matmul TFLOPS | {s.get('approx_tflops_matmul', 0):.1f} |"]
        first = meas["steps"][0]
        L += [f"| First (compilation) step | {first['step_time_s']:.1f} s |",
              f"| Peak memory allocated: overall / steady-state (after warm-up) | {_f(meas.get('peak_allocated_gib'))} / "
              f"{_f(meas.get('steady_peak_allocated_gib'))} GiB (reserved peak {_f(meas.get('peak_reserved_gib'))} GiB) |",
              f"| Latest train loss | {last['train_loss']:.4f} (step {last['step']}) |"]
        temps = [r["temp_max_c"] for r in meas["steps"] if r.get("temp_max_c") is not None]
        utils = [r["gpu_util_mean_pct"] for r in meas["steps"][s["warmup_steps_excluded"]:] if r.get("gpu_util_mean_pct") is not None]
        powers = [r["power_mean_w"] for r in meas["steps"][s["warmup_steps_excluded"]:] if r.get("power_mean_w") is not None]
        clocks = [r["sm_clock_mean_mhz"] for r in meas["steps"][s["warmup_steps_excluded"]:] if r.get("sm_clock_mean_mhz") is not None]
        L += [f"| GPU utilisation (timed steps, mean) | {_f(statistics.fmean(utils) if utils else None, '.0f', '%')} |",
              f"| GPU temperature max | {_f(max(temps) if temps else None, '.0f', ' °C')} |",
              f"| GPU power / SM clock (timed steps, mean) | {_f(statistics.fmean(powers) if powers else None, '.0f', ' W')} / "
              f"{_f(statistics.fmean(clocks) if clocks else None, '.0f', ' MHz')} |"]
        for e in meas["evals"]:
            L.append(f"| Validation at step {e['step']} | loss {e['val_loss']:.4f}, {e['duration_s']:.1f} s "
                     f"(first micro-batch {_f(e.get('first_microbatch_s'), '.1f', ' s')}"
                     f"{', includes compilation' if e['includes_compile'] else ''}; steady ≈ {steady_eval_seconds(e):.1f} s) |")
        calib = report.get("window_calibration") or {}
        if calib.get("status") == "done":
            L.append("| Window calibration (median forward+backward per micro-batch) | "
                     + ", ".join(f"window {p['window']}: {1000 * p['median_s']:.1f} ms" for p in calib["points"]) + " |")
        L.append("")
    if report.get("oom"):
        oo = report["oom"]
        L += [f"**Out of memory** during {oo.get('phase')} (step {oo.get('step')}, micro-batch {oo.get('micro_batch')}); "
              f"peak allocated {_f(oo.get('peak_allocated_gib'), '.2f', ' GiB')}. Message: `{oo.get('message')}`", ""]

    L += ["## 5. Estimated complete pretraining duration", ""]
    if measured:
        wc = est.get("wall_clock") or {}
        ci = est.get("total_ci95_s")
        calib = report.get("window_calibration") or {}
        if est["window_correction_applied"]:
            corr = est["window_correction"]
            training_line = (f"- **Training only, window-corrected: {fmt_duration(est['total_training_s'])}** for the "
                             f"{cfg_tokens(c)}-token budget (correction {corr['delta_total_s'] / 3600:+.2f} h; {CALIBRATION_METHOD}).")
        else:
            training_line = (f"- **Training only, NOT window-corrected: {fmt_duration(est['total_training_s'])}** "
                             f"(calibration {calib.get('status')}: {calib.get('reason')}); likely low because measured "
                             "steps used early-schedule attention windows.")
        L += [f"- **Projected wall-clock for one complete run: {fmt_duration(wc.get('total_s'))}** = training "
              f"{fmt_duration(wc.get('training_s'))} + {wc.get('evals_in_full_run')} validation passes "
              f"({_f(wc.get('eval_seconds_each'), '.0f', ' s')} each, steady) + validation compile "
              f"{_f(wc.get('eval_compile_excess_s'), '.0f', ' s')} + training compile/warm-up "
              f"{_f(wc.get('compile_warmup_overhead_s'), '.0f', ' s')} + setup {_f(wc.get('setup_s'), '.0f', ' s')}. "
              f"Measured with {config_label(c)}.",
              training_line,
              f"- Training only, uncorrected (basis of the live ETA; throughput of the last {s['ma_window']} timed steps): "
              f"{fmt_duration(est['total_steady_state_s'])}.",
              f"- Using all timed steps instead: {fmt_duration(est['total_steady_state_cumulative_s'])}.",
              f"- Range over sliding {s['ma_window']}-step windows (fastest – slowest): "
              f"{fmt_duration(est['total_range_s'][0])} – {fmt_duration(est['total_range_s'][1])}.",
              f"- 95% interval around the moving-average estimate (statistical only, last {s['ma_window']} steps): "
              + (f"{fmt_duration(ci[0])} – {fmt_duration(ci[1])}." if ci else "needs ≥ 2 timed steps."),
              f"- Remaining at the time of the report: {fmt_duration(est['eta_remaining_s'])} for {est['remaining_tokens']:,} tokens.",
              f"- Compared with {ref['hardware']} (≈ {RECORD_FULL_RUN_S:.1f} s of training for all 1530 steps): "
              f"**{est['slowdown_vs_8xH100']:,.0f}× slower** (training only)."]
        if pf and pf["original"]["verdict"] not in RUNNABLE_VERDICTS:
            L.append("- No reliable runtime estimate can be obtained for the original configuration itself (it cannot "
                     "complete a training step here); the figures above are for the configuration in section 3, which "
                     "keeps the model, optimizers, data and token budget.")
        L.append("")
    else:
        if pf and pf["original"]["verdict"] not in RUNNABLE_VERDICTS:
            L.append("**No reliable runtime estimate can be obtained for the original configuration**: it cannot "
                     "complete a single training step on this GPU (section 3).")
        L += ["No runtime estimate is available from this run: " + (
            "no timed optimizer step completed." if meas["steps"] or report["status"] not in ("preflight-only", "refused-preflight")
            else "no training was executed."), ""]
    L += ["Uncertainty:", ""] + [f"- {n}" for n in report["uncertainty_notes"] or ["No measurements yet."]] + [""]

    L += ["## 6. Limitations of the extrapolation", ""] + [f"- {x}" for x in report["limitations"]] + [""]

    L += ["## 7. Is full pretraining on this laptop feasible?", "", f"**Verdict: {report['feasibility']['verdict']}.** "
          f"{report['feasibility']['text']}", ""]

    L += ["## 8. Why a GPU cluster is preferable", ""]
    orig_row = pf.get("original") if pf else None
    if orig_row and orig_row["verdict"] not in RUNNABLE_VERDICTS:
        L.append(f"- Memory: the original configuration needs well over {orig_row['lower_bound_gib']:.0f} GiB per GPU "
                 f"(central estimate {orig_row['central_gib']:.0f} GiB); this laptop has {pf['vram_total_gib']:.1f} GiB, so it "
                 "can only run a modified micro-batch setup. 80 GB GPUs such as the H100s used for the record run the original "
                 "configuration unchanged.")
    if measured:
        L += [f"- Time: one run takes {fmt_duration(est['wall_clock']['total_s'])} wall-clock here "
              f"({fmt_duration(est['total_training_s'])} of training, measured with {config_label(c)}) versus about "
              f"{RECORD_FULL_RUN_S / 60:.1f} minutes of training on 8×H100 ({est['slowdown_vs_8xH100']:,.0f}× slower).",
              f"- Research needs many runs: the record at this commit was backed by {ref['runs']} runs for its "
              f"significance test; {ref['runs']} runs here would take {fmt_duration(est['time_for_record_significance_runs_s'])} wall-clock."]
    else:
        L.append(f"- Research needs many runs: the record at this commit was backed by {ref['runs']} runs on 8×H100 "
                 f"for its significance test ({ref['runs']} × {ref['train_time_s_mean'] / 60:.1f} min).")
    L += ["- Multi-GPU nodes allow the designed 8-way data parallelism and its exact micro-batch layout, which this "
          "laptop cannot reproduce.",
          "- Sustained multi-hour full load on a laptop is thermally and mechanically stressful and blocks the machine "
          "for other work; cluster nodes are built for continuous load and can be scheduled unattended.", ""]
    return "\n".join(L)


def cfg_tokens(c):
    return f"{c['target_tokens']:,}"


def atomic_write(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def timed_steps_in(json_path):
    """(timed steps, run_id) of an existing report, or (None, None) when there is none readable."""
    with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
        data = json.loads(Path(json_path).read_text())
        return int(data["summary"].get("timed_steps", 0)), data.get("run_id")
    return None, None


class Run:
    """Owns the run directory, the log file and the report, which is rewritten after every step."""

    def __init__(self, output_dir, report, t_start=None):
        self.output_dir = Path(output_dir)
        self.report = report
        self.t_start = time.monotonic() if t_start is None else t_start
        self.dir = self.output_dir / "reports" / report["run_id"]
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "train.log"
        self.kept_root = None
        self._write_warned = False

    def log(self, s, console=True):
        if console:
            print(s, flush=True)
        with open(self.log_path, "a") as f:
            f.write(s + "\n")

    def save(self, final=False):
        finalize(self.report)
        try:
            payload = json.dumps(self.report, indent=2, default=str)
            markdown = render_markdown(self.report)
            atomic_write(self.dir / f"{REPORT_NAME}.json", payload)
            atomic_write(self.dir / f"{REPORT_NAME}.md", markdown)
            latest_json = self.output_dir / f"{REPORT_NAME}.json"
            existing, existing_run = timed_steps_in(latest_json)
            if existing is None or self.report["summary"]["timed_steps"] >= existing:
                atomic_write(latest_json, payload)
                atomic_write(self.output_dir / f"{REPORT_NAME}.md", markdown)
                self.kept_root = None
                return [self.dir, self.output_dir]
            self.kept_root = self.output_dir / "reports" / str(existing_run) / f"{REPORT_NAME}.md"
            return [self.dir]
        except OSError as e:
            if final or not self._write_warned:
                print(f"WARNING: could not write the report ({e}); training continues.", flush=True)
                self._write_warned = True
            return []


# ----------------------------------------------------------------------------- training


def format_live(rec, s, num_iterations, warmup_steps, target_tokens):
    head = (f"step {rec['step']:>4}/{num_iterations} ({100 * rec['step'] / num_iterations:4.1f}%) | "
            f"loss {rec['train_loss']:.4f} | step {rec['step_time_s']:.2f} s")
    if s.get("timed_steps"):
        head += (f" | avg({s['ma_window']}) {s['step_time_ma_s']:.2f} s | {s['throughput_ma_tok_s']:,.0f} tok/s"
                 f" | ~{s.get('approx_tflops_matmul', 0):.1f} TFLOPS")
        eta = f"ETA {fmt_duration(s['eta_remaining_s'])} | total {fmt_duration(s['total_steady_state_s'])}"
    else:
        head += f" | warming up {s['completed_steps']}/{warmup_steps} | {rec['tokens_per_s']:,.0f} tok/s"
        if rec["step"] >= 2:
            remaining = max(target_tokens - s["tokens_done"], 0)
            eta = (f"ETA ~{fmt_duration(remaining / rec['tokens_per_s'])} | total "
                   f"~{fmt_duration(target_tokens / rec['tokens_per_s'])} (provisional)")
        else:
            eta = "ETA n/a (compilation step)"
    gpu = (f"GPU {_f(rec.get('gpu_util_mean_pct'), '.0f', '%')} {_f(rec.get('temp_max_c'), '.0f', ' °C')} "
           f"{_f(rec.get('power_mean_w'), '.0f', ' W')} {_f(rec.get('sm_clock_mean_mhz'), '.0f', ' MHz')}")
    mem = (f"mem {rec['mem_allocated_gib']:.2f} GiB alloc, {rec['mem_peak_allocated_gib']:.2f} peak, "
           f"{rec['mem_reserved_gib']:.2f} reserved")
    return f"{head}\n    {mem} | {gpu} | {eta}"


def calibrate_windows(model, train_loader, x, y, windows, use_ddp, k=CALIBRATION_MICRO_BATCHES):
    """Median seconds per real-data forward+backward micro-batch at each attention window; no optimizer step."""
    points = []
    with (model.no_sync() if use_ddp else contextlib.nullcontext()):
        for w in windows:
            attn_blocksize = torch.tensor(w, dtype=torch.int, device='cuda')
            samples = []
            for _ in range(k):
                torch.cuda.synchronize()
                t = time.monotonic()
                loss = model(x, y, attn_blocksize=attn_blocksize)
                x, y = train_loader.next_batch()
                loss.backward()
                torch.cuda.synchronize()
                samples.append(time.monotonic() - t)
            points.append({"window": w, "median_s": statistics.median(samples), "samples_s": samples})
    model.zero_grad(set_to_none=True)
    return points


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def train(up, cfg, run, telemetry, stop, registry_path, gpu):
    """The pinned main loop (upstream lines 380-560) with configurable micro-batch/DDP and measurement."""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    CastedLinear, Muon, DistributedDataLoader, config, args = up.CastedLinear, up.Muon, up.DistributedDataLoader, up.config, up.args
    report = run.report
    meas = report["measurements"]
    steps, evals = meas["steps"], meas["evals"]
    W = cfg["warmup_steps"]
    T = cfg["micro_batch_tokens"]
    val_steps, train_accumulation_steps = cfg["val_steps"], cfg["train_accumulation_steps"]
    val_loss_every = cfg["val_loss_every"]
    where = {"phase": "setup", "step": 0, "micro_batch": 0}

    for k, v in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1"), ("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", str(free_port()))):
        os.environ.setdefault(k, v)
    if int(os.environ["WORLD_SIZE"]) != 1:
        raise SystemExit("This harness measures a single GPU; run it without torchrun (WORLD_SIZE must be 1).")
    dist.init_process_group(backend='nccl')
    model = raw_model = optimizers = None
    try:
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        props = torch.cuda.get_device_properties(device)
        gpu.update(sm_count=props.multi_processor_count, torch_total_memory_gib=props.total_memory / GIB,
                   l2_cache_mib=getattr(props, "L2_cache_size", 0) / 2**20)

        smi = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
        run.log(up.code + "=" * 100 + "\n" + Path(__file__).read_text() + "=" * 100, console=False)
        run.log(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n{smi}\n" + "=" * 100, console=False)

        train_loader = DistributedDataLoader(str(UPSTREAM_DIR / args.input_bin), T, ddp_rank, ddp_world_size)
        val_loader = DistributedDataLoader(str(UPSTREAM_DIR / args.input_val_bin), T, ddp_rank, ddp_world_size)
        run.log(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
        run.log(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
        x, y = train_loader.next_batch()

        model = construct_model(up)
        model = model.cuda().bfloat16()
        for m in model.modules():
            if isinstance(m, CastedLinear):
                m.float()
        if hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = True
        model = torch.compile(model)
        if cfg["use_ddp"]:
            model = DDP(model, device_ids=[ddp_local_rank])
            raw_model = model.module
        else:
            raw_model = model
        optimizers = build_optimizers(raw_model, Muon)
        optimizer3 = optimizers[2]
        get_lr = make_get_lr(args)
        schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

        run.log(f"\nTraining: {cfg['train_accumulation_steps']} micro-batches × {T:,} tokens = {cfg['tokens_per_step']:,} "
                f"tokens/step, {args.num_iterations} steps, target {cfg['target_tokens']:,} tokens. "
                f"Press Ctrl+C once to stop and write the report.")
        first_wider = next(i for i, w in enumerate(schedule_windows(args.num_iterations)) if w > 64)
        run.log(f"Note: the live ETA extrapolates from early-schedule attention windows (64 tokens until step "
                f"{first_wider}, 1792 at the end), so it is likely low. "
                + ("After Ctrl+C, the final report applies a measured window-cost correction."
                   if cfg["window_calibration"] else
                   "--no-window-calibration is set, so the final report flags this bias without correcting it."))
        torch.cuda.synchronize()
        meas["setup_time_s"] = time.monotonic() - run.t_start
        train_time_s = 0.0
        for step in range(args.num_iterations + 1):
            last_step = (step == args.num_iterations)
            attn_blocksize = torch.tensor(64*((step/args.num_iterations * (1792 - 64) + 64)//64), dtype=torch.int, device='cuda')

            if (last_step or (val_loss_every > 0 and step % val_loss_every == 0)):
                if stop.requested:
                    break
                where.update(phase="validation", step=step, micro_batch=0)
                first_eval = not evals
                if first_eval:
                    run.log(f"step {step}: validation on {val_steps} micro-batches (the first evaluation includes compilation)...")
                torch.cuda.synchronize()
                t_eval = time.monotonic()
                model.eval()
                val_loader.reset()
                val_loss = 0.0
                first_mb_s = None
                for _ in range(val_steps):
                    if stop.requested:
                        break
                    with torch.no_grad():
                        x_val, y_val = val_loader.next_batch()
                        val_loss += model(x_val, y_val, attn_blocksize=attn_blocksize)
                    if first_mb_s is None:
                        torch.cuda.synchronize()
                        first_mb_s = time.monotonic() - t_eval
                if stop.requested:
                    run.log("validation interrupted; it is not recorded")
                    break
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                val_loss /= val_steps
                torch.cuda.synchronize()
                eval_s = time.monotonic() - t_eval
                evals.append({"step": step, "val_loss": val_loss.item(), "duration_s": eval_s, "first_microbatch_s": first_mb_s,
                              "val_steps": val_steps, "includes_compile": first_eval})
                timed = max(len(steps) - W, 0)
                run.log(f'step:{step}/{args.num_iterations} val_loss:{val_loss.item():.4f} train_time:{1000 * train_time_s:.0f}ms '
                        f'step_avg:{1000 * train_time_s / timed if timed else float("nan"):.2f}ms (validation took {eval_s:.1f} s)')
                run.save()

            if last_step:
                if cfg["save_checkpoint_on_exit"]:
                    save_checkpoint(run, up, raw_model, optimizers, step)
                break

            model.train()
            if step == 0:
                run.log("step 1: compiling (torch.compile + coordinate-descent tuning); this can take several minutes...")
            torch.cuda.synchronize()
            t0 = time.monotonic()
            shard_before = train_loader.current_shard
            completed = True
            for i in range(1, train_accumulation_steps+1):
                if stop.requested:
                    completed = False
                    break
                where.update(phase="forward/backward", step=step + 1, micro_batch=i)
                ctx = model.no_sync() if (cfg["use_ddp"] and i < train_accumulation_steps) else contextlib.nullcontext()
                with ctx:
                    loss = model(x, y, attn_blocksize=attn_blocksize)
                    x, y = train_loader.next_batch()
                    loss.backward()
                train_loss = loss.detach()
            if not completed:
                model.zero_grad(set_to_none=True)
                run.log(f"stopped during step {step + 1} after {i - 1}/{train_accumulation_steps} micro-batches; "
                        "this partial step is discarded")
                break
            where.update(phase="optimizer", micro_batch=0)
            for p in model.parameters():
                p.grad /= train_accumulation_steps
            frac = min(step/300, 1)
            optimizer3.param_groups[0]['momentum'] = (1 - frac) * 0.85 + frac * 0.95
            for opt, sched in zip(optimizers, schedulers):
                opt.step()
                sched.step()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t1 = time.monotonic()

            step_time = t1 - t0
            tokens = train_accumulation_steps * T * ddp_world_size
            rec = {"step": step + 1, "train_loss": train_loss.item(), "step_time_s": step_time, "tokens": tokens,
                   "tokens_per_s": tokens / step_time, "warmup": len(steps) < W, "attn_blocksize": int(attn_blocksize),
                   "mem_allocated_gib": torch.cuda.memory_allocated() / GIB,
                   "mem_peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
                   "mem_peak_since": "start" if len(steps) <= W else "end of warm-up",
                   "mem_reserved_gib": torch.cuda.memory_reserved() / GIB,
                   **telemetry.window(t0, t1)}
            steps.append(rec)
            meas["peak_allocated_gib"] = max(meas["peak_allocated_gib"] or 0.0, rec["mem_peak_allocated_gib"])
            meas["peak_reserved_gib"] = max(meas["peak_reserved_gib"] or 0.0, torch.cuda.max_memory_reserved() / GIB)
            if len(steps) > W:
                meas["steady_peak_allocated_gib"] = rec["mem_peak_allocated_gib"]
            elif len(steps) == W:
                torch.cuda.reset_peak_memory_stats()
            if not rec["warmup"]:
                train_time_s += step_time
            if train_loader.current_shard < shard_before and meas.get("data_wrapped_at_step") is None:
                meas["data_wrapped_at_step"] = step + 1
                run.log(f"note: the training data wrapped around to the first shard during step {step + 1}")
            s = summarize(steps, W, cfg["ma_window"], cfg["target_tokens"], report["model"]["flops_per_token_matmul"])
            print(format_live(rec, s, args.num_iterations, W, cfg["target_tokens"]), flush=True)
            timed = max(len(steps) - W, 0)
            run.log(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{1000 * train_time_s:.0f}ms "
                    f"step_avg:{1000 * train_time_s / timed if timed else float('nan'):.2f}ms", console=False)
            run.save()

        if stop.requested:
            report["status"] = "stopped-thermal" if stop.reason == "thermal" else "interrupted"
            if stop.reason in ("user-interrupt", "sigterm") and steps and cfg["window_calibration"]:
                where.update(phase="window calibration", micro_batch=0)
                w_measured = steps[-1]["attn_blocksize"]
                windows = sorted({w_measured, *CALIBRATION_WINDOWS})
                run.log(f"\nWindow calibration: {CALIBRATION_MICRO_BATCHES} real-data forward+backward micro-batches at "
                        f"attention windows {windows} (no optimizer update; Ctrl+C to skip)...")
                try:
                    # A stop during validation leaves eval mode, which would select a different compiled graph.
                    model.train()
                    points = calibrate_windows(model, train_loader, x, y, windows, cfg["use_ddp"])
                    report["window_calibration"] = {"status": "done", "w_measured": w_measured, "points": points,
                                                    "micro_batches_per_window": CALIBRATION_MICRO_BATCHES,
                                                    "method": CALIBRATION_METHOD}
                    run.log("window calibration: " + ", ".join(f"window {p['window']}: {1000 * p['median_s']:.1f} ms"
                                                               for p in points))
                except (KeyboardInterrupt, Exception) as exc:
                    with contextlib.suppress(Exception):
                        model.zero_grad(set_to_none=True)
                    reason = "second Ctrl+C" if isinstance(exc, KeyboardInterrupt) else f"{type(exc).__name__}: {str(exc)[:200]}"
                    report["window_calibration"] = {"status": "skipped", "reason": reason, "w_measured": w_measured}
                    run.log(f"window calibration skipped ({reason}); the report flags the uncorrected estimate")
            elif cfg["window_calibration"]:
                report["window_calibration"] = {"status": "not-run", "reason": "only after a Ctrl+C/SIGTERM stop with "
                                                "completed steps (not after a thermal stop)"}
            if cfg["save_checkpoint_on_exit"] and steps:
                save_checkpoint(run, up, raw_model, optimizers, len(steps))
            return EXIT_THERMAL if stop.reason == "thermal" else EXIT_OK
        report["status"] = "completed"
        return EXIT_OK
    except Exception as exc:
        if is_oom(exc):
            peak = max(meas["peak_allocated_gib"] or 0.0, torch.cuda.max_memory_allocated() / GIB)
            msg = str(exc).strip().splitlines()[0][:500] if str(exc).strip() else type(exc).__name__
            report["status"] = "oom"
            report["oom"] = {**where, "peak_allocated_gib": peak, "message": msg}
            record_oom_failure(registry_path, oom_key(cfg, gpu), {"run_id": report["run_id"], **report["oom"]})
            run.log(f"\nCUDA out of memory during {where['phase']} (step {where['step']}, micro-batch {where['micro_batch']}); "
                    f"peak allocated {peak:.2f} GiB.\n{msg}\nThis configuration is recorded in {registry_path} and will not "
                    "be retried automatically. Choose a smaller --micro-batch-tokens (a new deviation that needs approval).")
            return EXIT_OOM
        if stop.requested:
            thermal = stop.reason == "thermal"
            report["status"] = "stopped-thermal" if thermal else "interrupted"
            report["stop_detail"] = f"{stop.detail}; {type(exc).__name__} raised while stopping: {str(exc)[:300]}"
            report["error"] = "".join(traceback.format_exception(exc))[-4000:]
            return EXIT_THERMAL if thermal else EXIT_OK
        raise
    finally:
        with contextlib.suppress(Exception):
            dist.destroy_process_group()


def save_checkpoint(run, up, raw_model, optimizers, step):
    path = run.dir / ('state_step%06d.pt' % step)
    run.log(f"saving checkpoint to {path} (several GB)...")
    log = dict(step=step, code=up.code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
    tmp = path.with_name(path.name + ".tmp")
    torch.save(log, tmp)
    os.replace(tmp, path)
    run.report["checkpoint"] = {"path": str(path), "step": step, "size_gib": path.stat().st_size / GIB}


# ----------------------------------------------------------------------------- main


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preflight-only", action="store_true", help="print the memory analysis and write a report; no CUDA")
    p.add_argument("--micro-batch-tokens", type=int, default=None,
                   help="tokens per micro-batch (original: 65536); smaller values raise gradient accumulation "
                        "so tokens per step stay 524,288. A deviation: needs --accept-deviations")
    p.add_argument("--no-ddp", action="store_true", help="skip the DDP wrapper (identity at world_size=1). A deviation")
    p.add_argument("--val-loss-every", type=int, default=None, help="validation frequency (original: 125). A deviation")
    p.add_argument("--accept-deviations", action="store_true", help="confirm the listed changes from the pinned config")
    p.add_argument("--max-temp", type=float, default=82.0, help="stop gracefully above this GPU temperature in °C (82)")
    p.add_argument("--temp-poll-interval", type=float, default=2.0, help="seconds between telemetry samples (2)")
    p.add_argument("--warmup-steps", type=int, default=3, help="optimizer steps excluded from throughput (compilation; 3)")
    p.add_argument("--ma-window", type=int, default=10, help="moving-average window in optimizer steps (10)")
    p.add_argument("--save-checkpoint-on-exit", action="store_true", help="save model+optimizer state on exit (~4 GB)")
    p.add_argument("--no-window-calibration", action="store_true",
                   help="skip the short post-stop attention-window calibration (the estimate is then flagged as likely low)")
    p.add_argument("--output-dir", default=str(PROJECT_DIR), help="where reports/, the latest report and oom_failures.json go")
    p.add_argument("--retry-failed-config", action="store_true", help="allow a configuration recorded as OOM to run again")
    a = p.parse_args(argv)
    if a.warmup_steps < 1:
        p.error("--warmup-steps must be >= 1 (the first step includes compilation)")
    if a.ma_window < 1 or a.temp_poll_interval <= 0:
        p.error("--ma-window must be >= 1 and --temp-poll-interval > 0")
    return a


def new_report(run_id, orig, cfg, deviations, mm, gpu, env, preflight):
    return {
        "schema": f"{REPORT_NAME}/v1",
        "run_id": run_id,
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "status": "initialised",
        "stop_reason": None,
        "stop_detail": None,
        "upstream": {"repo": UPSTREAM_REPO, "commit": UPSTREAM_COMMIT, "file": "train_gpt2.py",
                     "sha256": UPSTREAM_SHA256, "reference_8xH100": REFERENCE_8XH100},
        "original_config": orig,
        "effective_config": cfg,
        "deviations": deviations,
        "non_training_differences": non_training_differences(cfg),
        "model": mm,
        "gpu": gpu,
        "environment": env,
        "preflight": preflight,
        "data": None,
        "measurements": {"steps": [], "evals": [], "setup_time_s": None, "peak_allocated_gib": None,
                         "steady_peak_allocated_gib": None, "peak_reserved_gib": None, "data_wrapped_at_step": None},
        "window_calibration": ({"status": "not-run", "reason": "training has not stopped via Ctrl+C/SIGTERM"}
                               if cfg.get("window_calibration", True) else
                               {"status": "disabled", "reason": "--no-window-calibration"}),
        "oom": None,
        "error": None,
        "checkpoint": None,
    }


def print_exit_summary(report, paths, kept_root=None):
    est, s, c = report["estimates"], report["summary"], report["effective_config"]
    print("\n" + "=" * 100)
    print(f"status: {report['status']}" + (f" ({report.get('stop_detail')})" if report.get("stop_detail") else ""))
    if "total_training_s" in est:
        ci = est.get("total_ci95_s")
        calib = report.get("window_calibration") or {}
        print(f"completed {s['completed_steps']} steps ({s['timed_steps']} timed), {est['throughput_tok_s']:,.0f} tok/s "
              f"(last {s['ma_window']} steps), measured with {config_label(c)}")
        label = "window-corrected" if est["window_correction_applied"] else "NOT window-corrected, likely low"
        print(f"ESTIMATED TOTAL TRAINING TIME ({label}; {config_label(c)}): {fmt_duration(est['total_training_s'])}  "
              f"[window range {fmt_duration(est['total_range_s'][0])} – {fmt_duration(est['total_range_s'][1])}"
              + (f"; 95% CI {fmt_duration(ci[0])} – {fmt_duration(ci[1])}" if ci else "") + "]")
        print(f"  uncorrected (live-ETA basis, early-schedule windows): {fmt_duration(est['total_steady_state_s'])}")
        print("  window correction: " + (CALIBRATION_METHOD if est["window_correction_applied"]
                                         else f"{calib.get('status')} ({calib.get('reason')})"))
        wc = est.get("wall_clock") or {}
        print(f"PROJECTED WALL-CLOCK FOR ONE COMPLETE RUN (training + validation + compile): {fmt_duration(wc.get('total_s'))}")
        print(f"feasibility verdict: {report['feasibility']['verdict']}")
        for n in report["uncertainty_notes"]:
            print(f"  - {n}")
    else:
        print(f"feasibility verdict: {report['feasibility']['verdict']}. {report['feasibility']['text']}")
    for d in paths:
        print(f"report: {Path(d) / (REPORT_NAME + '.md')}")
    if kept_root:
        print(f"latest root report kept (it has more timed steps): {kept_root}")
    print("=" * 100)


def main(argv=None):
    t_start = time.monotonic()
    cli = parse_args(argv)
    up = load_upstream()
    orig = original_config(up)
    try:
        cfg = resolve_config(orig, cli.micro_batch_tokens, cli.no_ddp, cli.val_loss_every)
    except ValueError as e:
        print(f"error: {e}")
        return EXIT_REFUSED
    cfg.update(warmup_steps=cli.warmup_steps, ma_window=cli.ma_window, max_temp_c=cli.max_temp,
               temp_poll_interval_s=cli.temp_poll_interval, save_checkpoint_on_exit=cli.save_checkpoint_on_exit,
               window_calibration=not cli.no_window_calibration, deviations_accepted=cli.accept_deviations)
    deviations = list_deviations(cfg, orig)
    output_dir = Path(cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = output_dir / "oom_failures.json"

    telemetry = GpuTelemetry(nvml_index())
    gpu = telemetry.static_info()
    vram = telemetry.memory()
    if vram is None:
        print("error: cannot read GPU memory (no NVML / nvidia-smi), so the memory preflight cannot run.")
        return EXIT_REFUSED
    mm = model_memory(up)
    preflight = run_preflight(mm, vram, cfg, orig)
    print(format_preflight(preflight, gpu.get("name")))
    print()

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    report = new_report(run_id, orig, cfg, deviations, mm, gpu, host_environment(), preflight)
    report["data"] = check_data(up, cfg, orig)
    d = report["data"]
    print(f"Data: {d['train_shards']} train shard(s) with {d['train_tokens']:,} tokens, {d['val_shards']} val shard(s); "
          f"covers {d['steps_covered']} of {cfg['num_iterations']} steps at this micro-batch size.\n")
    sel = preflight["selected"]

    if cli.preflight_only:
        report["status"] = "preflight-only"
        run = Run(output_dir, report)
        print_exit_summary(report, run.save(final=True), run.kept_root)
        return EXIT_OK

    if sel["verdict"] not in RUNNABLE_VERDICTS:
        report["status"] = "refused-preflight"
        report["stop_detail"] = f"preflight verdict for the selected configuration: {sel['verdict']}"
        need = (f"it needs at least {sel['lower_bound_gib']:.1f} GiB even for its first micro-batch (central estimate "
                f"{sel['central_gib']:.1f} GiB)" if sel["lower_bound_gib"] > preflight["budget_gib"] else
                f"its central estimate is {sel['central_gib']:.1f} GiB (×{FRAGMENTATION_FACTOR} for fragmentation)")
        print(f"REFUSED: the selected configuration (micro-batch {sel['micro_batch_tokens']:,} tokens, "
              f"DDP {'on' if sel['use_ddp'] else 'off'}) is '{sel['verdict']}' on this GPU: {need}, but the budget is "
              f"{preflight['budget_gib']:.2f} GiB. It is not attempted.")
        if sel["is_original"]:
            print("No reliable runtime estimate can be obtained for the original configuration on this GPU.")
        fits = [r for r in preflight["rows"] if r["verdict"] in RUNNABLE_VERDICTS]
        if fits:
            print("Memory-compatible alternatives keeping the full model, optimizers, data and token budget "
                  "(each is a deviation that needs your approval):")
            for r in fits:
                print(f"  --micro-batch-tokens {r['micro_batch_tokens']}{' --no-ddp' if not r['use_ddp'] else ''}"
                      f"  ({r['verdict']}, central {r['central_gib']:.1f} GiB, upper {r['upper_gib']:.1f} GiB)")
        run = Run(output_dir, report)
        print_exit_summary(report, run.save(final=True), run.kept_root)
        return EXIT_REFUSED

    if deviations and not cli.accept_deviations:
        print("This configuration differs from the pinned one:")
        for d in deviations:
            print(f"  - {d['setting']}: {d['original']} -> {d['used']}\n      {d['consequence']}\n      {d['impact']}")
        print("Nothing was run. If you approve these changes, re-run with --accept-deviations.")
        return EXIT_REFUSED

    prior = find_oom_failure(registry_path, oom_key(cfg, gpu))
    if prior and not cli.retry_failed_config:
        print(f"REFUSED: this configuration already ran out of memory (run {prior.get('run_id')}, {prior.get('time_utc')}); "
              f"see {registry_path}. Pick a smaller --micro-batch-tokens, or pass --retry-failed-config if something "
              "changed (e.g. other GPU applications were closed).")
        return EXIT_REFUSED

    data = report["data"]
    if data["error"]:
        print(f"error: {data['error']}")
        return EXIT_REFUSED
    if data["warning"]:
        print(f"warning: {data['warning']}\n")

    if not torch.cuda.is_available():
        print("error: CUDA is not available to PyTorch (CUDA_VISIBLE_DEVICES hidden or driver problem); nothing was run.")
        return EXIT_REFUSED

    if deviations:
        print("Accepted deviations from the pinned configuration:")
        for d in deviations:
            print(f"  - {d['setting']}: {d['original']} -> {d['used']}")
    first = telemetry.sample()
    report["environment"]["thermal_stop_available"] = first["temp_c"] is not None
    if first["temp_c"] is None:
        print("!" * 100 + "\nWARNING: the GPU temperature cannot be read, so the --max-temp thermal stop is UNAVAILABLE "
              "for this run.\nWatch the temperature yourself (e.g. `watch -n 2 nvidia-smi`) and press Ctrl+C to stop.\n"
              + "!" * 100)
    else:
        print(f"GPU temperature before training: {first['temp_c']} °C (stop limit {cli.max_temp} °C)")
    run = Run(output_dir, report, t_start)
    report["status"] = "running"
    stop = StopController()
    stop.install()
    telemetry.start(cli.temp_poll_interval, cli.max_temp, stop)
    code = EXIT_ERROR
    try:
        code = train(up, cfg, run, telemetry, stop, registry_path, gpu)
    except KeyboardInterrupt:
        report["status"] = "aborted"
        code = EXIT_ABORTED
    except BaseException as exc:
        report["status"] = "error"
        report["error"] = "".join(traceback.format_exception(exc))[-4000:]
        raise
    finally:
        # Further Ctrl+C presses must not interrupt the final report write.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        telemetry.stop()
        report["stop_reason"] = stop.reason
        report["stop_detail"] = report.get("stop_detail") or stop.detail or None
        report["environment"]["max_gpu_temp_seen_c"] = telemetry.max_temp_seen
        print_exit_summary(report, run.save(final=True), run.kept_root)
    return code


if __name__ == "__main__":
    sys.exit(main())
