"""CPU-only tests for scripts/eval_checkpoint.py. The GPU is hidden before torch is imported."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_checkpoint as ec  # noqa: E402
from test_feasibility_math import UP, cuda_as_identity, write_shard  # noqa: E402


class ChunkTests(unittest.TestCase):
    def test_default_sizes_tile_the_validation_tokens(self):
        steps = ec.chunk_steps(UP.args.val_tokens, ec.DEFAULT_CHUNK_TOKENS)
        self.assertEqual(steps, {4096: 2560, 16384: 640, 32768: 320, 65536: 160})

    def test_rejects_sizes_that_do_not_tile_or_break_flex_blocks(self):
        with self.assertRaisesRegex(ValueError, "divide"):
            ec.chunk_steps(UP.args.val_tokens, [4096, 3 * 4096])
        with self.assertRaisesRegex(ValueError, "multiple of 128"):
            ec.chunk_steps(UP.args.val_tokens, [4160 - 32])
        with self.assertRaisesRegex(ValueError, "multiple of 128"):
            ec.chunk_steps(UP.args.val_tokens, [0])

    def test_window_matches_the_training_schedule(self):
        n = UP.args.num_iterations
        schedule = ec.tf.schedule_windows(n)
        self.assertEqual([ec.attn_blocksize(s, n) for s in range(n)], schedule)
        self.assertEqual(ec.attn_blocksize(n, n), 1792)

    def test_default_output_next_to_the_run_report(self):
        out = ec.default_output("checkpoints/20261115-232515-94a608/ckpt_step001530.pt", 1530)
        self.assertEqual(out, ec.tf.PROJECT_DIR / "reports" / "20261115-232515-94a608" / "eval_chunks_001530.json")

    def test_every_chunk_size_scores_the_same_targets(self):
        val_tokens = 64
        tokens = np.arange(200, dtype=np.uint16)
        with tempfile.TemporaryDirectory() as d, cuda_as_identity():
            write_shard(Path(d) / "fineweb_val_000000.bin", tokens)
            for T in (8, 16, 32, 64):
                n = val_tokens // T
                loader = UP.DistributedDataLoader(str(Path(d) / "fineweb_val_*.bin"), T, 0, 1)
                loader.reset()
                xs, ys = zip(*(loader.next_batch() for _ in range(n)))
                np.testing.assert_array_equal(torch.cat(xs).numpy(), tokens[:val_tokens])
                np.testing.assert_array_equal(torch.cat(ys).numpy(), tokens[1:val_tokens + 1])


class ReferenceCheckTests(unittest.TestCase):
    RESULTS = [{"chunk_tokens": 2048, "status": "ok", "val_loss": 3.40},
               {"chunk_tokens": 4096, "status": "ok", "val_loss": 3.3195},
               {"chunk_tokens": 65536, "status": "oom"}]

    def test_pass_within_tolerance(self):
        check = ec.reference_check(self.RESULTS, 4096, 3.3193)
        self.assertEqual(check["status"], "ok")
        self.assertAlmostEqual(check["abs_diff"], 2e-4)
        self.assertEqual(ec.exit_code(self.RESULTS[:2], check), ec.tf.EXIT_OK)

    def test_mismatch_beyond_tolerance(self):
        check = ec.reference_check(self.RESULTS, 4096, 3.3150)
        self.assertEqual(check["status"], "mismatch")
        self.assertEqual(ec.exit_code(self.RESULTS[:2], check), ec.EXIT_REFERENCE_MISMATCH)

    def test_missing_reference_or_measurement_is_not_verified(self):
        self.assertEqual(ec.reference_check(self.RESULTS, 4096, None)["status"], "not-verified")
        self.assertEqual(ec.reference_check(self.RESULTS, 65536, 3.3)["status"], "not-verified")
        self.assertEqual(ec.reference_check(self.RESULTS, 1024, 3.3)["status"], "not-verified")

    def test_thermal_and_oom_exit_codes_take_precedence(self):
        check = ec.reference_check(self.RESULTS, 4096, 3.0)
        self.assertEqual(ec.exit_code(self.RESULTS, check), ec.tf.EXIT_OOM)
        aborted = self.RESULTS + [{"chunk_tokens": 8192, "status": "aborted-thermal"}]
        self.assertEqual(ec.exit_code(aborted, check), ec.tf.EXIT_THERMAL)

    def test_logged_val_loss_from_report(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "report.json"
            path.write_text('{"measurements": {"evals": [{"step": 0, "val_loss": 10.8}, {"step": 1530, "val_loss": 3.3}]}}')
            self.assertEqual(ec.logged_val_loss(path, 1530), 3.3)
            self.assertIsNone(ec.logged_val_loss(path, 1500))
            self.assertIsNone(ec.logged_val_loss(Path(d) / "missing.json", 1530))


if __name__ == "__main__":
    unittest.main()
