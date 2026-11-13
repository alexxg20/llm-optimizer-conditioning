"""CPU-only tests for train_feasibility.py. The GPU is hidden before torch is imported."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import contextlib
import io
import json
import math
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_feasibility as tf  # noqa: E402

UP = tf.load_upstream()
ORIG = tf.original_config(UP)
MM = tf.model_memory(UP)
VRAM_12GB = {"total_bytes": 12227 * 2**20, "used_bytes": 14 * 2**20}


def fake_step(i, seconds, tokens=524288, window=64):
    return {"step": i, "train_loss": 10.0 - 0.01 * i, "step_time_s": seconds, "tokens": tokens,
            "tokens_per_s": tokens / seconds, "warmup": i <= 3, "attn_blocksize": window,
            "mem_allocated_gib": 6.5, "mem_peak_allocated_gib": 8.1, "mem_reserved_gib": 8.6,
            "gpu_util_mean_pct": 98.0, "temp_max_c": 74.0, "power_mean_w": 90.0, "sm_clock_mean_mhz": 2100.0,
            "power_capped_pct": 40.0, "thermal_throttled_pct": 0.0, "telemetry_samples": 10}


def fake_eval(step, duration, first_mb, val_steps=2560):
    return {"step": step, "val_loss": 10.83, "duration_s": duration, "first_microbatch_s": first_mb,
            "val_steps": val_steps, "includes_compile": step == 0}


def synthetic_report(micro_batch_tokens=4096, no_ddp=True, steps=(), evals=(), status="interrupted", calibration=None):
    cfg = tf.resolve_config(ORIG, micro_batch_tokens, no_ddp)
    cfg.update(warmup_steps=3, ma_window=10, max_temp_c=82.0, temp_poll_interval_s=2.0,
               save_checkpoint_on_exit=False, window_calibration=True, deviations_accepted=True)
    pf = tf.run_preflight(MM, VRAM_12GB, cfg, ORIG)
    gpu = {"name": "Test GPU", "vram_total_mib": 12227, "compute_capability": "12.0"}
    report = tf.new_report("test-run", ORIG, cfg, tf.list_deviations(cfg, ORIG), MM, gpu, tf.host_environment(), pf)
    report["status"] = status
    report["measurements"].update(steps=list(steps), evals=list(evals), setup_time_s=30.0, peak_allocated_gib=8.1,
                                  steady_peak_allocated_gib=7.9, peak_reserved_gib=8.6)
    if calibration:
        report["window_calibration"] = calibration
    return tf.finalize(report)


def code_lines(lines):
    """(indent, code) pairs without comments, blank lines or comment-only lines."""
    out = []
    for line in lines:
        code = line.split(" #")[0].rstrip()
        if code.strip() and not code.strip().startswith("#"):
            out.append((len(code) - len(code.lstrip()), code.strip()))
    return out


def ordered_match(upstream, ours, check_indent):
    """True if `upstream` appears in `ours` in order (other lines may be interleaved), optionally with the same
    indentation relative to the first matched line."""
    for start, (_, text) in enumerate(ours):
        if text != upstream[0][1]:
            continue
        base_u, base_o, j, ok = upstream[0][0], ours[start][0], start, True
        for indent, code in upstream:
            while j < len(ours) and not (ours[j][1] == code and (not check_indent or ours[j][0] - base_o == indent - base_u)):
                j += 1
            if j == len(ours):
                ok = False
                break
            j += 1
        if ok:
            return True
    return False


class UpstreamTests(unittest.TestCase):
    def test_hyperparameters_come_from_pinned_file(self):
        self.assertEqual(tf.sha256_of(tf.UPSTREAM_FILE), tf.UPSTREAM_SHA256)
        self.assertEqual((ORIG["batch_size"], ORIG["sequence_length"], ORIG["num_iterations"]), (8, 65536, 1530))
        self.assertEqual(ORIG["tokens_per_step"], 524_288)
        self.assertEqual(ORIG["target_tokens"], 802_160_640)
        self.assertEqual(ORIG["val_tokens"], 10_485_760)
        self.assertEqual(ORIG["val_loss_every"], 125)
        self.assertEqual(UP.code, tf.UPSTREAM_FILE.read_text())

    def test_parameter_accounting(self):
        self.assertEqual(MM["n_params"], 625_803_306)
        self.assertEqual(MM["params_by_dtype"], {"bfloat16": 502_235_178, "float32": 123_568_128})
        self.assertEqual(MM["param_bytes"], 502_235_178 * 2 + 123_568_128 * 4)
        self.assertEqual(MM["matmul_params"], 123_568_128)
        self.assertEqual(MM["muon_state_bytes"], 84_934_656 * 4)
        # Adam + Muon must cover every parameter exactly once.
        self.assertEqual(MM["adam_state_bytes"], 2 * (MM["param_bytes"] - MM["muon_state_bytes"]))

    def test_partition_matches_real_optimizers(self):
        model = UP.GPT(UP.GPTConfig(vocab_size=256, n_layer=12, n_head=2, n_embd=64)).bfloat16()
        for m in model.modules():
            if isinstance(m, UP.CastedLinear):
                m.float()
        opts = tf.build_optimizers(model, UP.Muon)
        ids = lambda opt: {id(p) for g in opt.param_groups for p in g["params"]}
        adam, muon = tf.optimizer_partition(model)
        self.assertEqual(ids(opts[0]) | ids(opts[1]) | ids(opts[3]), {id(p) for p in adam})
        self.assertEqual(ids(opts[2]), {id(p) for p in muon})
        self.assertIsInstance(opts[2], UP.Muon)
        self.assertEqual(len(adam) + len(muon), len(list(model.parameters())))

    def test_replicated_lines_are_verbatim_and_in_order(self):
        upstream = tf.UPSTREAM_FILE.read_text().splitlines()
        ours = code_lines(Path(tf.__file__).read_text().splitlines())
        # Loop bodies: order and relative indentation must match. Line 534 (the no_sync line) is intentionally changed.
        strict = [(498, 506), (532, 533), (535, 553)]
        loose = [(382, 387), (436, 447), (450, 471), (490, 490)]
        for (start, end), check_indent in [(r, True) for r in strict] + [(r, False) for r in loose]:
            block = code_lines(upstream[start - 1:end])
            self.assertTrue(ordered_match(block, ours, check_indent),
                            f"upstream lines {start}-{end} are not reproduced verbatim and in order")

    def test_ordered_match_rejects_reordering_and_reindenting(self):
        up = code_lines(["    a = 1", "    for i in x:", "        b(i)", "    c()"])
        self.assertTrue(ordered_match(up, code_lines(["        a = 1", "        z()", "        for i in x:",
                                                      "            b(i)", "        c()"]), True))
        self.assertFalse(ordered_match(up, code_lines(["a = 1", "c()", "for i in x:", "    b(i)"]), True))
        self.assertFalse(ordered_match(up, code_lines(["a = 1", "for i in x:", "    b(i)", "    c()"]), True))

    def test_pinned_model_forward_on_cpu(self):
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        mod = tf.exec_upstream("upstream_cpu_smoke")
        mod.create_block_mask = lambda *a, **k: create_block_mask(*a, **{**k, "device": "cpu", "_compile": False})
        mod.flex_attention = flex_attention
        model = mod.GPT(mod.GPTConfig(vocab_size=50304, n_layer=12, n_head=2, n_embd=64)).bfloat16()
        for m in model.modules():
            if isinstance(m, mod.CastedLinear):
                m.float()
        idx = torch.randint(0, 50257, (257,))
        idx[100] = 50256
        with torch.no_grad():
            loss = model(idx[:-1], idx[1:], attn_blocksize=torch.tensor(128, dtype=torch.int))
        # lm_head is zero-initialised, so the first loss is exactly ln(vocab).
        self.assertAlmostEqual(loss.item(), math.log(50304), places=2)


class ConfigTests(unittest.TestCase):
    def test_original_is_unchanged(self):
        cfg = tf.resolve_config(ORIG)
        self.assertEqual((cfg["micro_batch_tokens"], cfg["train_accumulation_steps"], cfg["val_steps"]), (65536, 8, 160))
        self.assertTrue(cfg["use_ddp"])
        self.assertEqual(tf.list_deviations(cfg, ORIG), [])

    def test_smaller_micro_batch_keeps_tokens_per_step(self):
        cfg = tf.resolve_config(ORIG, 4096, no_ddp=True)
        self.assertEqual((cfg["train_accumulation_steps"], cfg["val_steps"]), (128, 2560))
        self.assertEqual(cfg["train_accumulation_steps"] * 4096, ORIG["tokens_per_step"])
        devs = tf.list_deviations(cfg, ORIG)
        self.assertEqual(len(devs), 2)
        self.assertTrue(devs[0]["setting"].startswith("sequence_length"))
        self.assertIn("~58k tokens per shard at 65,536, 256 at 4,096", devs[0]["impact"])
        self.assertIn("bf16 over 128 micro-batches instead of 8", devs[0]["impact"])
        self.assertEqual(devs[1]["setting"], "DistributedDataParallel wrapper")
        self.assertIn("not guaranteed to be pessimistic", devs[1]["impact"])

    def test_val_frequency_is_a_deviation(self):
        cfg = tf.resolve_config(ORIG, val_loss_every=0)
        self.assertEqual([d["setting"] for d in tf.list_deviations(cfg, ORIG)], ["val_loss_every"])

    def test_invalid_micro_batches_rejected(self):
        for bad in (0, 1000, 384, -128):
            with self.assertRaises(ValueError):
                tf.resolve_config(ORIG, bad)


class MemoryTests(unittest.TestCase):
    budget = VRAM_12GB["total_bytes"] - VRAM_12GB["used_bytes"] - tf.CUDA_CONTEXT_RESERVE_GIB * tf.GIB

    def test_original_is_impossible(self):
        est = tf.estimate_memory(MM, 65536, use_ddp=True)
        self.assertEqual(est["logits_bf16_bytes"], 65536 * 50304 * 2)
        self.assertGreater(est["lower_bound_bytes"], 20 * tf.GIB)
        self.assertEqual(tf.memory_verdict(est, self.budget), "impossible")

    def test_recommended_alternatives_fit(self):
        self.assertEqual(tf.memory_verdict(tf.estimate_memory(MM, 4096, use_ddp=False), self.budget), "expected-to-fit")
        self.assertEqual(tf.memory_verdict(tf.estimate_memory(MM, 2048, use_ddp=True), self.budget), "expected-to-fit")
        self.assertEqual(tf.memory_verdict(tf.estimate_memory(MM, 16384, use_ddp=False), self.budget), "implausible")

    def test_persistent_state_and_embedding_transient(self):
        est = tf.estimate_memory(MM, 4096, use_ddp=True)
        self.assertAlmostEqual(est["persistent_bytes"] / tf.GIB, 6.66, places=2)
        self.assertAlmostEqual(tf.estimate_memory(MM, 4096, use_ddp=False)["persistent_bytes"] / tf.GIB, 5.27, places=2)
        self.assertEqual(est["embedding_backward_transient_bytes"], 50304 * (9216 + 768) * 6)
        small = tf.estimate_memory(MM, 128, use_ddp=True)
        self.assertEqual(small["central"], small["persistent_bytes"] + small["embedding_backward_transient_bytes"])

    def test_estimates_are_ordered_and_monotonic(self):
        prev = 0
        for T in (2048, 4096, 8192, 16384, 65536):
            est = tf.estimate_memory(MM, T, use_ddp=True)
            self.assertLessEqual(est["lower_bound_bytes"], est["lower"])
            self.assertLessEqual(est["lower"], est["central"])
            self.assertLessEqual(est["central"], est["upper"])
            self.assertGreaterEqual(est["central"], prev)
            prev = est["central"]

    def test_preflight_marks_original_and_selected(self):
        cfg = tf.resolve_config(ORIG, 4096, no_ddp=True)
        pf = tf.run_preflight(MM, VRAM_12GB, cfg, ORIG)
        self.assertEqual(pf["original"]["micro_batch_tokens"], 65536)
        self.assertEqual(pf["selected"]["micro_batch_tokens"], 4096)
        self.assertFalse(pf["selected"]["use_ddp"])
        self.assertIn("impossible", tf.format_preflight(pf, "Test GPU"))


class StatsTests(unittest.TestCase):
    target = ORIG["target_tokens"]

    def test_constant_steps(self):
        steps = [fake_step(1, 300.0), fake_step(2, 25.0), fake_step(3, 21.0)] + [fake_step(i, 20.0) for i in range(4, 14)]
        s = tf.summarize(steps, 3, 10, self.target, MM["flops_per_token_matmul"])
        self.assertEqual((s["completed_steps"], s["timed_steps"]), (13, 10))
        self.assertAlmostEqual(s["throughput_ma_tok_s"], 524288 / 20)
        self.assertAlmostEqual(s["total_steady_state_s"], 1530 * 20.0)
        self.assertAlmostEqual(s["eta_remaining_s"], (1530 - 13) * 20.0)
        self.assertAlmostEqual(s["drift_pct"], 0.0)
        self.assertEqual(s["total_ci95_s"], [1530 * 20.0, 1530 * 20.0])
        self.assertAlmostEqual(s["approx_tflops_matmul"], 6 * 123_568_128 * 524288 / 20 / 1e12)

    def test_eta_is_token_based(self):
        steps = [fake_step(i, 10.0, tokens=262144) for i in range(1, 8)]
        s = tf.summarize(steps, 1, 10, self.target)
        self.assertAlmostEqual(s["eta_remaining_s"], (self.target - 7 * 262144) / 26214.4)

    def test_ci_is_centred_on_moving_average(self):
        steps = [fake_step(i, 20.0 + i) for i in range(1, 31)]
        s = tf.summarize(steps, 3, 10, self.target)
        self.assertGreater(s["drift_pct"], 0)
        self.assertLess(s["total_fastest_window_s"], s["total_slowest_window_s"])
        self.assertAlmostEqual(s["total_slowest_window_s"], s["total_steady_state_s"])
        window = [20.0 + i for i in range(21, 31)]
        half = 1.96 * statistics.stdev(window) / math.sqrt(10) / statistics.fmean(window)
        lo, hi = s["total_ci95_s"]
        self.assertAlmostEqual(lo, s["total_steady_state_s"] * (1 - half))
        self.assertAlmostEqual(hi, s["total_steady_state_s"] * (1 + half))

    def test_warmup_only_has_no_estimate(self):
        s = tf.summarize([fake_step(1, 300.0)], 3, 10, self.target)
        self.assertNotIn("throughput_ma_tok_s", s)

    def test_eval_time_projection(self):
        steps = [fake_step(1, 300.0), fake_step(2, 20.0), fake_step(3, 20.0)] + [fake_step(i, 20.0) for i in range(4, 10)]
        s = tf.summarize(steps, 3, 10, self.target)
        evals = [fake_eval(0, 400.0, 300.0)]
        steady = (400.0 - 300.0) * 2560 / 2559
        self.assertAlmostEqual(tf.steady_eval_seconds(evals[0]), steady)
        wc = tf.project_wall_clock(30600.0, s, steps, evals, 30.0, 3, 14)
        self.assertAlmostEqual(wc["compile_warmup_overhead_s"], 280.0)
        self.assertAlmostEqual(wc["eval_seconds_each"], steady)
        self.assertAlmostEqual(wc["eval_compile_excess_s"], 400.0 - steady)
        self.assertAlmostEqual(wc["total_s"], 30600.0 + 30.0 + 280.0 + 14 * steady + (400.0 - steady))

    def test_eval_count_and_formatting(self):
        self.assertEqual(tf.n_evals_full_run(1530, 125), 14)
        self.assertEqual(tf.n_evals_full_run(1530, 0), 1)
        self.assertEqual(tf.fmt_duration(86400), "24.00 h (1.00 d)")
        self.assertEqual(tf.shard_usable_tokens(100_000_000, 4096), 24414 * 4096)
        self.assertEqual(tf.skipped_tail(4096), 256)
        self.assertEqual(tf.skipped_tail(65536), 57600)


class CalibrationTests(unittest.TestCase):
    def test_schedule_matches_pinned_formula(self):
        ws = tf.schedule_windows(1530)
        self.assertEqual(len(ws), 1530)
        self.assertEqual((ws[0], ws[56], ws[57], ws[-1]), (64, 64, 128, 1728))
        self.assertEqual(ws, sorted(ws))

    def test_interpolation(self):
        pts = [(64, 1.0), (896, 2.0), (1792, 4.0)]
        self.assertEqual(tf.interpolate(pts, 32), 1.0)
        self.assertAlmostEqual(tf.interpolate(pts, 480), 1.5)
        self.assertAlmostEqual(tf.interpolate(pts, 1344), 3.0)
        self.assertEqual(tf.interpolate(pts, 4000), 4.0)

    def test_correction_averages_over_schedule(self):
        a, b, accum, ma = 0.150, 2e-5, 128, 20.0
        points = [{"window": w, "median_s": a + b * w} for w in (64, 896, 1792, 64)]
        corr = tf.window_correction(ma, accum, points, 64, 1530, steps_done=30)
        ws = tf.schedule_windows(1530)
        expected_mean = ma + accum * b * (statistics.fmean(ws) - 64)
        self.assertAlmostEqual(corr["mean_step_s"], expected_mean)
        self.assertAlmostEqual(corr["total_s"], 1530 * expected_mean)
        self.assertAlmostEqual(corr["delta_total_s"], 1530 * (expected_mean - ma))
        self.assertAlmostEqual(corr["remaining_s"], sum(ma + accum * b * (w - 64) for w in ws[30:]))

    def test_report_headlines_corrected_figure(self):
        steps = [fake_step(1, 300.0)] + [fake_step(i, 20.0) for i in range(2, 30)]
        calib = {"status": "done", "w_measured": 64, "method": tf.CALIBRATION_METHOD,
                 "points": [{"window": 64, "median_s": 0.150}, {"window": 896, "median_s": 0.160},
                            {"window": 1792, "median_s": 0.175}]}
        report = synthetic_report(steps=steps, evals=[fake_eval(0, 400.0, 300.0)], calibration=calib)
        est = report["estimates"]
        self.assertTrue(est["window_correction_applied"])
        self.assertGreater(est["total_training_s"], est["total_steady_state_s"])
        self.assertAlmostEqual(est["wall_clock"]["training_s"], est["total_training_s"])
        md = tf.render_markdown(report)
        self.assertIn("Training only, window-corrected", md)
        self.assertIn("forward+backward only, no optimizer update, real data, after training stopped", md)
        self.assertIn("Training only, uncorrected", md)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            tf.print_exit_summary(report, [])
        self.assertIn("ESTIMATED TOTAL TRAINING TIME (window-corrected; 4,096-token micro-batches × 128", out.getvalue())

    def test_uncorrected_estimate_is_flagged(self):
        report = synthetic_report(steps=[fake_step(i, 20.0) for i in range(1, 20)])
        self.assertFalse(report["estimates"]["window_correction_applied"])
        self.assertTrue(any("NOT window-corrected" in n for n in report["uncertainty_notes"]))
        self.assertIn("not window-corrected, likely low", report["feasibility"]["text"])


class SafetyTests(unittest.TestCase):
    def test_oom_detection(self):
        self.assertTrue(tf.is_oom(torch.OutOfMemoryError("CUDA out of memory")))
        self.assertTrue(tf.is_oom(RuntimeError("CUDA error: out of memory")))
        wrapped = ValueError("compile failed")
        wrapped.__cause__ = RuntimeError("Triton: out of memory")
        self.assertTrue(tf.is_oom(wrapped))
        self.assertFalse(tf.is_oom(RuntimeError("out of resource: shared memory")))

    def test_oom_registry(self):
        with tempfile.TemporaryDirectory() as d:
            reg = Path(d) / "oom_failures.json"
            cfg = tf.resolve_config(ORIG, 4096, no_ddp=True)
            key = tf.oom_key(cfg, {"name": "Test GPU", "vram_total_mib": 12227})
            self.assertIsNone(tf.find_oom_failure(reg, key))
            tf.record_oom_failure(reg, key, {"run_id": "r1", "phase": "forward/backward"})
            self.assertEqual(tf.find_oom_failure(reg, key)["run_id"], "r1")
            other = tf.oom_key(tf.resolve_config(ORIG, 2048), {"name": "Test GPU", "vram_total_mib": 12227})
            self.assertIsNone(tf.find_oom_failure(reg, other))

    def test_stop_controller_keeps_first_reason(self):
        stop = tf.StopController()
        self.assertTrue(stop.request("thermal", "hot"))
        self.assertFalse(stop.request("user-interrupt"))
        self.assertEqual(stop.reason, "thermal")


class ThermalTests(unittest.TestCase):
    def governor(self):
        return tf.ThermalGovernor(82.0, 77.0, 87.0, 900.0)

    def test_hysteresis(self):
        g = self.governor()
        seq = [(80, None), (82, "cool"), (85, None), (78, None), (77, "resume"), (81, None), (82, "cool")]
        for t, (temp, expected) in enumerate(seq):
            self.assertEqual(g.update(temp, float(t)), expected, f"temp {temp}")
        self.assertTrue(g.cooling)

    def test_emergency_stop_is_strictly_above_max_temp_and_overrides_cooling(self):
        g = self.governor()
        self.assertEqual(g.update(87, 0.0), "cool")
        self.assertIsNone(g.update(87, 1.0))
        self.assertEqual(g.update(88, 2.0), "stop")
        self.assertIn("--max-temp emergency limit 87", g.detail)

    def test_cooling_timeout(self):
        g = self.governor()
        g.update(83, 0.0)
        self.assertIsNone(g.update(80, 900.0))
        self.assertEqual(g.update(80, 901.0), "timeout")
        self.assertIn("--max-cool-wait 900", g.detail)

    def test_failed_reads_are_ignored_but_cannot_extend_a_pause(self):
        g = self.governor()
        self.assertIsNone(g.update(None, 0.0))
        self.assertFalse(g.cooling)
        g.update(83, 10.0)
        self.assertIsNone(g.update(None, 500.0))
        self.assertTrue(g.cooling)
        self.assertEqual(g.update(None, 911.0), "timeout")

    def test_invalid_thresholds(self):
        for args in [(77.0, 82.0, 87.0, 900.0), (82.0, 77.0, 80.0, 900.0), (82.0, 77.0, 87.0, 0.0)]:
            with self.assertRaises(ValueError):
                tf.ThermalGovernor(*args)

    def test_cli_defaults_and_validation(self):
        a = tf.parse_args([])
        self.assertEqual((a.cool_at, a.resume_at, a.max_temp, a.max_cool_wait), (82.0, 77.0, 87.0, 900.0))
        for bad in (["--resume-at", "83"], ["--max-temp", "80"], ["--max-cool-wait", "0"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                tf.parse_args(bad)

    def test_wait_for_cooling(self):
        cooling, logs, syncs = threading.Event(), [], []
        args = (lambda: False, lambda: 83, 82.0, 77.0, logs.append, lambda: syncs.append(1))
        self.assertIsNone(tf.wait_for_cooling(cooling, *args, poll_s=0.01))
        self.assertEqual((logs, syncs), ([], []))
        cooling.set()
        threading.Timer(0.2, cooling.clear).start()
        p = tf.wait_for_cooling(cooling, *args, poll_s=0.01)
        self.assertGreaterEqual(p["seconds"], 0.15)
        self.assertFalse(p["ended_by_stop"])
        self.assertEqual(syncs, [1])
        self.assertEqual(logs[0], "cooling pause: GPU 83 °C >= 82 °C, waiting for <= 77 °C ...")
        self.assertTrue(logs[1].startswith("resumed after"))

    def test_wait_for_cooling_ends_on_stop(self):
        cooling, stopped = threading.Event(), threading.Event()
        cooling.set()
        threading.Timer(0.1, stopped.set).start()
        p = tf.wait_for_cooling(cooling, stopped.is_set, lambda: 84, 82.0, 77.0, lambda s: None, poll_s=0.01)
        self.assertTrue(p["ended_by_stop"])
        self.assertIsNone(tf.wait_for_cooling(cooling, stopped.is_set, lambda: 84, 82.0, 77.0, lambda s: None))

    def test_telemetry_thread_drives_events_and_stop(self):
        temps = iter([80, 82, 79, 76, 88] + [88] * 100)
        with mock.patch.dict(sys.modules, {"pynvml": None}):
            tel = tf.GpuTelemetry()
        tel.backend = "fake"
        tel.sample = lambda: {"t": time.monotonic(), "temp_c": next(temps), "util_pct": None,
                              "power_w": None, "sm_clock_mhz": None, "throttle": []}
        stop, seen = tf.StopController(), []
        orig_update = tf.ThermalGovernor.update

        class Recording(tf.ThermalGovernor):
            def update(self, temp_c, now):
                action = orig_update(self, temp_c, now)
                seen.append((temp_c, action, tel.cooling.is_set()))
                return action

        with contextlib.redirect_stdout(io.StringIO()):
            tel.start(0.01, Recording(82.0, 77.0, 87.0, 900.0), stop)
            deadline = time.monotonic() + 5
            while not stop.requested and time.monotonic() < deadline:
                time.sleep(0.01)
            tel.stop()
        self.assertEqual([a for _, a, _ in seen[:5]], [None, "cool", None, "resume", "stop"])
        self.assertEqual(stop.reason, "thermal")
        self.assertTrue(tel.emergency.is_set())
        self.assertFalse(tel.cooling.is_set())

    def test_cooling_stats_exclude_warmup_steps(self):
        steps = [fake_step(1, 300.0), fake_step(2, 20.0), fake_step(3, 20.0)] + [fake_step(i, 20.0) for i in range(4, 14)]
        c = tf.cooling_stats(steps, [fake_eval(0, 100.0, 50.0)], [{"seconds": 30.0}, {"seconds": 30.0}])
        self.assertEqual(c["pauses"], 2)
        self.assertAlmostEqual(c["active_s"], 10 * 20.0 + 100.0)
        self.assertAlmostEqual(c["pause_per_compute_s"], 60.0 / 300.0)
        self.assertAlmostEqual(c["pause_fraction_of_wall_clock"], 60.0 / 360.0)
        self.assertIsNone(tf.cooling_stats([], [], [])["pause_per_compute_s"])

    def test_projection_includes_cooling(self):
        steps = [fake_step(i, 20.0) for i in range(1, 10)]
        s = tf.summarize(steps, 3, 10, ORIG["target_tokens"])
        evals = [fake_eval(0, 400.0, 300.0)]
        base = tf.project_wall_clock(30600.0, s, steps, evals, 30.0, 3, 14)
        cooled = tf.project_wall_clock(30600.0, s, steps, evals, 30.0, 3, 14, cooling_per_compute_s=0.1)
        self.assertEqual(base["cooling_s"], 0.0)
        self.assertAlmostEqual(cooled["cooling_s"], 0.1 * (30600.0 + 14 * base["eval_seconds_each"]))
        self.assertAlmostEqual(cooled["total_s"] - base["total_s"], cooled["cooling_s"])

    def test_live_eta_includes_cooling(self):
        steps = [fake_step(i, 20.0) for i in range(1, 14)]
        s = tf.summarize(steps, 3, 10, ORIG["target_tokens"])
        rec = {**steps[-1], "mem_peak_allocated_gib": 8.1}
        plain = tf.format_live(rec, s, 1530, 3, ORIG["target_tokens"])
        cooled = tf.format_live(rec, s, 1530, 3, ORIG["target_tokens"],
                                {"pauses": 2, "pause_total_s": 120.0, "pause_per_compute_s": 0.5})
        self.assertIn(f"ETA {tf.fmt_duration(s['eta_remaining_s'])}", plain)
        self.assertIn(f"ETA {tf.fmt_duration(1.5 * s['eta_remaining_s'])}", cooled)
        self.assertIn("2 pause(s), 2.0 min", cooled)

    def test_report_shows_thresholds_and_pauses(self):
        steps = [fake_step(i, 20.0) for i in range(1, 14)]
        report = synthetic_report(steps=steps, evals=[fake_eval(0, 400.0, 300.0)])
        report["effective_config"].update(cool_at_c=82.0, resume_at_c=77.0, max_temp_c=87.0, max_cool_wait_s=900.0)
        md = tf.render_markdown(tf.finalize(report))
        self.assertIn("pause at ≥ 82 °C until ≤ 77 °C (at most 900 s per pause); emergency stop above 87 °C", md)
        self.assertIn("No cooling pause so far", md)
        report["measurements"]["cooling_pauses"] = [{"phase": "training", "step": 5, "micro_batch": 3, "seconds": 50.0,
                                                     "start_temp_c": 82, "end_temp_c": 77, "ended_by_stop": False}]
        tf.finalize(report)
        c = report["cooling"]
        self.assertEqual(c["pauses"], 1)
        self.assertAlmostEqual(c["pause_per_compute_s"], 50.0 / (10 * 20.0 + 400.0))
        wc = report["estimates"]["wall_clock"]
        self.assertGreater(wc["cooling_s"], 0)
        self.assertAlmostEqual(report["estimates"]["eta_remaining_incl_cooling_s"],
                               report["estimates"]["eta_remaining_s"] * (1 + c["pause_per_compute_s"]))
        md = tf.render_markdown(report)
        self.assertIn("| Cooling pauses (excluded from step and validation times) | 1, 0.8 min in total", md)
        self.assertIn("+ cooling pauses", md)
        self.assertIn("extrapolates the heat balance seen so far", md)


class ReportTests(unittest.TestCase):
    SECTIONS = ["## 1. Original training requirements", "## 2. Laptop GPU", "## 3. Changes made to fit the model",
                "## 4. Measured throughput and peak memory", "## 5. Estimated complete pretraining duration",
                "## 6. Limitations of the extrapolation", "## 7. Is full pretraining on this laptop feasible?",
                "## 8. Why a GPU cluster is preferable"]
    NO_ESTIMATE = "No reliable runtime estimate can be obtained for the original configuration"

    def test_measured_report_has_all_sections(self):
        steps = [fake_step(1, 300.0), fake_step(2, 22.0), fake_step(3, 21.0)] + [fake_step(i, 20.0 + 0.05 * i) for i in range(4, 40)]
        report = synthetic_report(steps=steps, evals=[fake_eval(0, 380.0, 300.0)])
        md = tf.render_markdown(report)
        for heading in self.SECTIONS:
            self.assertIn(heading, md)
        self.assertIn(self.NO_ESTIMATE, md)
        self.assertIn("Projected wall-clock for one complete run", md)
        self.assertIn("steady-state (after warm-up)", md)
        self.assertIn("--accept-deviations", md)
        self.assertEqual(report["feasibility"]["verdict"], "feasible-single-run")
        self.assertIn("wall-clock", report["feasibility"]["text"])
        self.assertAlmostEqual(report["estimates"]["record_full_run_s"], 264.6 * 1530 / 1520)
        self.assertGreater(report["estimates"]["slowdown_vs_8xH100"], 100)
        json.dumps(report, default=str)

    def test_verdict_is_provisional_with_few_timed_steps(self):
        report = synthetic_report(steps=[fake_step(i, 20.0) for i in range(1, 9)])
        self.assertEqual(report["summary"]["timed_steps"], 5)
        self.assertTrue(report["feasibility"]["verdict"].endswith("(provisional)"))
        firm = synthetic_report(steps=[fake_step(i, 20.0) for i in range(1, 14)])
        self.assertNotIn("provisional", firm["feasibility"]["verdict"])

    def test_oom_after_some_steps(self):
        report = synthetic_report(steps=[fake_step(i, 20.0) for i in range(1, 16)], status="oom")
        report["oom"] = {"phase": "forward/backward", "step": 16, "micro_batch": 7, "peak_allocated_gib": 11.2,
                         "message": "CUDA out of memory"}
        tf.finalize(report)
        self.assertEqual(report["feasibility"]["verdict"],
                         "did not complete (out of memory at step 16); throughput measured over 12 steps")
        md = tf.render_markdown(report)
        self.assertIn("**Out of memory** during forward/backward (step 16, micro-batch 7)", md)
        self.assertIn("Projected wall-clock", md)

    def test_refused_original_report(self):
        report = synthetic_report(65536, False, status="refused-preflight")
        md = tf.render_markdown(report)
        for heading in self.SECTIONS:
            self.assertIn(heading, md)
        self.assertIn(self.NO_ESTIMATE, md)
        self.assertEqual(report["feasibility"]["verdict"], "undetermined")

    def test_root_report_needs_at_least_as_many_timed_steps(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "training_feasibility_report.json"
            long_run = synthetic_report(steps=[fake_step(i, 20.0) for i in range(1, 14)])
            tf.Run(d, long_run).save()
            short = synthetic_report(steps=[fake_step(i, 20.0) for i in range(1, 6)])
            short["run_id"] = "short-run"
            run = tf.Run(d, short)
            self.assertEqual(len(run.save()), 1)
            self.assertEqual(json.loads(root.read_text())["run_id"], "test-run")
            self.assertEqual(run.kept_root, Path(d) / "reports" / "test-run" / "training_feasibility_report.md")
            self.assertTrue((Path(d) / "reports" / "short-run" / "training_feasibility_report.md").exists())
            equal = synthetic_report(steps=[fake_step(i, 21.0) for i in range(1, 14)])
            equal["run_id"] = "equal-run"
            self.assertEqual(len(tf.Run(d, equal).save()), 2)
            self.assertEqual(json.loads(root.read_text())["run_id"], "equal-run")


def write_shard(path, tokens):
    header = np.zeros(256, dtype=np.int32)
    header[:3] = (20240520, 1, len(tokens))
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(np.asarray(tokens, dtype=np.uint16).tobytes())


@contextlib.contextmanager
def cuda_as_identity():
    saved = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    try:
        yield
    finally:
        torch.Tensor.cuda = saved


class CheckpointPlanTests(unittest.TestCase):
    def cfg(self, fraction=0.1, weights_only=False, on_exit=False):
        cfg = tf.resolve_config(ORIG, 4096, no_ddp=True)
        cfg.update(checkpoint_every_fraction=fraction, checkpoint_weights_only=weights_only,
                   save_checkpoint_on_exit=on_exit, checkpoint_dir="/tmp/ckpt")
        return cfg

    def test_schedule(self):
        self.assertEqual(tf.checkpoint_steps(1530, 0.1), list(range(0, 1531, 153)))
        self.assertEqual(len(tf.checkpoint_steps(1530, 0.1)), 11)
        self.assertEqual(tf.checkpoint_steps(1530, 0), [])
        self.assertEqual(tf.checkpoint_steps(1530, 0.15)[-3:], [1150, 1380, 1530])
        self.assertEqual(tf.checkpoint_steps(1530, 1), [0, 1530])
        for bad in (-0.1, 1.5, 1e-5):
            with self.assertRaises(ValueError):
                tf.checkpoint_steps(1530, bad)

    def test_size_estimate(self):
        full_state = MM["param_bytes"] + MM["adam_state_bytes"] + MM["muon_state_bytes"]
        data = 1530 * 128 * (4097 * 2 + 20)
        plan = tf.checkpoint_plan(self.cfg(), MM)
        self.assertEqual((plan["count"], plan["interval"]), (11, 153))
        self.assertEqual(plan["bytes_per_checkpoint"], full_state)
        self.assertEqual(plan["data_bytes"], data)
        self.assertEqual(plan["total_bytes"], 11 * full_state + data)
        self.assertEqual(tf.checkpoint_plan(self.cfg(weights_only=True), MM)["bytes_per_checkpoint"], MM["param_bytes"])
        self.assertEqual(tf.checkpoint_plan(self.cfg(on_exit=True), MM)["count"], 12)
        off = tf.checkpoint_plan(self.cfg(fraction=0), MM)
        self.assertFalse(off["enabled"])
        self.assertEqual(off["total_bytes"], 0)
        exit_only = tf.checkpoint_plan(self.cfg(fraction=0, on_exit=True), MM)
        self.assertEqual((exit_only["count"], exit_only["data_bytes"]), (1, data))

    def test_disk_rule(self):
        self.assertTrue(tf.disk_sufficient(10, 10))
        self.assertFalse(tf.disk_sufficient(11, 10))
        with tempfile.TemporaryDirectory() as d:
            self.assertGreater(tf.free_bytes_at(Path(d) / "not" / "created" / "yet"), 0)

    def test_description_and_projection(self):
        text = tf.checkpoint_description(self.cfg())
        self.assertIn("every 153 steps (11 checkpoints: steps 0, 153, ..., 1530)", text)
        self.assertIn("No checkpoints", tf.checkpoint_description(self.cfg(fraction=0)))
        steps = [fake_step(i, 20.0) for i in range(1, 10)]
        s = tf.summarize(steps, 3, 10, ORIG["target_tokens"])
        base = tf.project_wall_clock(30600.0, s, steps, [], 30.0, 3, 14)
        with_ckpt = tf.project_wall_clock(30600.0, s, steps, [], 30.0, 3, 14, checkpoint_s_each=20.0, n_checkpoints=11)
        self.assertAlmostEqual(with_ckpt["total_s"] - base["total_s"], 220.0)

    def test_report_lists_checkpoints_and_projects_their_time(self):
        report = synthetic_report(steps=[fake_step(i, 20.0) for i in range(1, 14)])
        report["checkpoint_plan"] = tf.checkpoint_plan(self.cfg(), MM)
        report["checkpoints"] = [{"step": 0, "path": "/tmp/ckpt/ckpt_step000000.pt", "data_file": None, "seconds": 5.0,
                                  "size_bytes": 2 * tf.GIB},
                                 {"step": 153, "path": "/tmp/ckpt/ckpt_step000153.pt", "data_file": "/tmp/ckpt/d.npz",
                                  "seconds": 12.0, "size_bytes": 4 * tf.GIB}]
        tf.finalize(report)
        wc = report["estimates"]["wall_clock"]
        self.assertEqual((wc["checkpoint_seconds_each"], wc["checkpoints_in_full_run"]), (12.0, 11))
        md = tf.render_markdown(report)
        self.assertIn("| Checkpoints written | 2 in /tmp/ckpt (steps 0, 153;", md)
        self.assertIn("+ 11 checkpoints (12 s each)", md)


class DataRecorderTests(unittest.TestCase):
    T = 8

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.shards = [np.arange(50, dtype=np.uint16) + 1000 * k for k in range(2)]
        for k, tokens in enumerate(self.shards):
            write_shard(d / f"fineweb_train_{k + 1:06d}.bin", tokens)
        self.pattern = str(d / "fineweb_train_*.bin")

    def tearDown(self):
        self.tmp.cleanup()

    def recorder(self, enabled=True):
        return tf.DataRecorder(UP.DistributedDataLoader(self.pattern, self.T, 0, 1), enabled)

    def test_positions_and_tokens_match_the_pinned_loader(self):
        seen = []
        with cuda_as_identity():
            rec = self.recorder()
            for _ in range(14):
                x, y = rec.next_batch()
                shard, offset, window = rec.pending
                np.testing.assert_array_equal(window[:-1], x.numpy())
                np.testing.assert_array_equal(window[1:], y.numpy())
                np.testing.assert_array_equal(window, self.shards[shard][offset:offset + self.T + 1])
                seen.append((shard, offset))
        # 6 windows fit in a 50-token shard at T=8; then the next shard, then the wrap back to the first.
        self.assertEqual(seen, [(0, o) for o in range(0, 41, 8)] + [(1, o) for o in range(0, 41, 8)] + [(0, 0), (0, 8)])
        self.assertEqual(rec.current_shard, 0)

    def test_consumed_bookkeeping_and_record_files(self):
        with cuda_as_identity():
            rec = self.recorder()
            x, y = rec.next_batch()
            for step in (1, 2):
                rec.begin_step()
                for i in (1, 2, 3):
                    rec.consume(step, i)
                    x, y = rec.next_batch()
                rec.commit_step()
            rec.begin_step()
            rec.consume(3, 1)
            x, y = rec.next_batch()
            rec.discard_step()
        self.assertEqual(len(rec.buffer), 6)
        self.assertEqual(rec.resume_position(), (1, 0))
        meta = {"run_id": "r", "upstream_sha256": "u", "harness_sha256": "h",
                "shard_sha256": {"fineweb_train_000001.bin": "abc"}}
        with tempfile.TemporaryDirectory() as out:
            npz, sidecar = rec.flush(out, 1, 2, meta)
            self.assertEqual(npz.name, "data_steps000001-000002.npz")
            with np.load(npz) as data:
                self.assertEqual(sorted(data.files), ["micro_batch", "offset", "shard_index", "step", "tokens"])
                self.assertEqual((data["tokens"].shape, data["tokens"].dtype), ((6, 9), np.uint16))
                self.assertEqual(data["step"].tolist(), [1, 1, 1, 2, 2, 2])
                self.assertEqual(data["micro_batch"].tolist(), [1, 2, 3, 1, 2, 3])
                self.assertEqual(data["shard_index"].tolist(), [0] * 6)
                self.assertEqual(data["offset"].tolist(), list(range(0, 41, 8)))
                for row, off in zip(data["tokens"], data["offset"]):
                    np.testing.assert_array_equal(row, self.shards[0][off:off + 9])
            on_disk = json.loads(npz.with_suffix(".json").read_text())
            self.assertEqual(on_disk, json.loads(json.dumps(sidecar)))
            self.assertEqual((on_disk["T"], on_disk["steps"], on_disk["n_micro_batches"]), (8, [1, 2], 6))
            self.assertEqual([s["sha256"] for s in on_disk["shards"]], ["abc", None])
            self.assertIn("x = tokens[i, :-1]", on_disk["definition"])
            self.assertEqual((on_disk["run_id"], on_disk["harness_sha256"]), ("r", "h"))
            self.assertNotIn("shard_sha256", on_disk)
            self.assertEqual(list(Path(out).glob("*.tmp")), [])
        self.assertEqual(rec.buffer, [])

    def test_disabled_recorder_keeps_no_tokens(self):
        with cuda_as_identity():
            rec = self.recorder(enabled=False)
            rec.next_batch()
            rec.begin_step()
            rec.consume(1, 1)
            rec.next_batch()
            rec.commit_step()
        self.assertIsNone(rec.pending[2])
        self.assertEqual(rec.buffer, [])


class GateTests(unittest.TestCase):
    """main() in-process with fixed VRAM numbers; train() is replaced so it can never start."""

    def run_main(self, *argv, free_bytes=None):
        out = io.StringIO()

        def forbidden_train(*args, **kwargs):
            raise AssertionError("train() must not be reached in tests")

        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(out):
            saved = tf.GpuTelemetry.memory, tf.train, tf.free_bytes_at
            tf.GpuTelemetry.memory = lambda self: VRAM_12GB
            tf.train = forbidden_train
            if free_bytes is not None:
                tf.free_bytes_at = lambda path: free_bytes
            try:
                code = tf.main([*argv, "--output-dir", d])
            finally:
                tf.GpuTelemetry.memory, tf.train, tf.free_bytes_at = saved
            files = sorted(str(p.relative_to(d)) for p in Path(d).rglob("*.json"))
        return code, out.getvalue(), files

    def test_disk_preflight_refuses_when_checkpoints_do_not_fit(self):
        code, out, _ = self.run_main("--micro-batch-tokens", "4096", "--no-ddp", "--accept-deviations",
                                     "--checkpoint-every-fraction", "0.1", free_bytes=10 * tf.GIB)
        self.assertEqual(code, tf.EXIT_REFUSED)
        self.assertIn("REFUSED: the checkpoints of a full run need", out)
        self.assertIn("quota may be lower than the free space shown", out)

    def test_preflight_only_prints_checkpoint_plan(self):
        code, out, _ = self.run_main("--preflight-only", "--checkpoint-every-fraction", "0.1")
        self.assertEqual(code, tf.EXIT_OK)
        self.assertIn("Checkpoints: 11 ×", out)

    def test_original_refused_by_preflight(self):
        code, out, files = self.run_main()
        self.assertEqual(code, tf.EXIT_REFUSED)
        self.assertIn("REFUSED", out)
        self.assertIn("needs at least", out)
        self.assertIn("No reliable runtime estimate can be obtained for the original configuration", out)
        self.assertIn("training_feasibility_report.json", files)

    def test_deviation_gate(self):
        code, out, files = self.run_main("--micro-batch-tokens", "4096", "--no-ddp")
        self.assertEqual(code, tf.EXIT_REFUSED)
        self.assertIn("re-run with --accept-deviations", out)
        self.assertEqual(files, [])

    def test_accepted_run_stops_before_gpu_when_cuda_hidden(self):
        code, out, _ = self.run_main("--micro-batch-tokens", "4096", "--no-ddp", "--accept-deviations")
        self.assertEqual(code, tf.EXIT_REFUSED)
        self.assertTrue("CUDA is not available" in out or "shards missing" in out, out)
        self.assertFalse(torch.cuda.is_initialized())


class CliTests(unittest.TestCase):
    def test_preflight_only_subprocess(self):
        with tempfile.TemporaryDirectory() as d:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
            res = subprocess.run([sys.executable, str(ROOT / "train_feasibility.py"), "--preflight-only", "--output-dir", d],
                                 capture_output=True, text=True, env=env, timeout=300)
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertIn("Memory preflight", res.stdout)
            report = json.loads((Path(d) / "training_feasibility_report.json").read_text())
            self.assertEqual(report["status"], "preflight-only")
            self.assertEqual(report["preflight"]["original"]["verdict"], "impossible")


if __name__ == "__main__":
    unittest.main()
