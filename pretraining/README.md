# Single-GPU pretraining harness: modded-nanogpt @ 9730304

Runs the **real** pinned training (`upstream/train_gpt2.py`, record #14 "Value Embeddings": same
model, Muon/Adam optimizers, FineWeb data, loss and 802,160,640-token budget) on one GPU until you
press Ctrl+C, shows a live ETA, and writes `training_feasibility_report.{md,json}`.

`train_feasibility.py` verifies the sha256 of the pinned file and executes its definitions
(model, Muon, data loader, hyperparameters) unmodified; only the main loop is replicated, with
measurement and safety hooks added.

All commands below run from `pretraining/`.

## Setup (once)

`setup.sh` creates `.venv`, reads the GPU's compute capability with `nvidia-smi` and installs
torch 2.14.1 from the matching wheel index (≥ 7.5 → cu130; 7.0, e.g. V100 → cu126; override with
`TORCH_INDEX=<url>`), then `requirements.txt`. It finally checks that the build has kernels for this GPU.

Laptop, with mise (Python 3.12 from `mise.toml`):

```bash
cd pretraining
mise trust && mise install              # Python 3.12, creates .venv
mise run setup                          # bash setup.sh
.venv/bin/python scripts/check_gpu.py   # ~1-2 min kernel compatibility check, does not train
mise run data                           # FineWeb10B: val + 9 train shards (~2 GB) into upstream/data/fineweb10B/
mise run test                           # CPU-only tests, GPU hidden
mise run preflight                      # memory analysis, no training
```

Without mise (any Python 3.12/3.13 with `venv`):

```bash
cd pretraining
PYTHON=python3 bash setup.sh
.venv/bin/python scripts/check_gpu.py
.venv/bin/python upstream/data/cached_fineweb10B.py 9
CUDA_VISIBLE_DEVICES= .venv/bin/python -m unittest discover -s tests
.venv/bin/python train_feasibility.py --preflight-only
```

## Why the original configuration does not run on 12 GB

The pinned config packs 65,536 tokens into each micro-batch. Even its first micro-batch needs
≈20 GiB (weights + DDP buckets + one 6.1 GiB bf16 logits tensor + minimum saved activations); a
steady-state step is estimated at ≈50 GiB. `.venv/bin/python train_feasibility.py` (no flags) therefore
refuses to run it and writes a report saying no runtime estimate can be obtained for it.

## Training commands (each needs your explicit approval of the listed changes)

Option B (recommended): 4096-token micro-batches × 128 accumulation steps, no DDP wrapper
(identity at world_size=1, saves 1.4 GiB):

```bash
.venv/bin/python train_feasibility.py --micro-batch-tokens 4096 --no-ddp --accept-deviations
```

Option A (one change only): 2048-token micro-batches × 256 accumulation steps, DDP kept:

```bash
.venv/bin/python train_feasibility.py --micro-batch-tokens 2048 --accept-deviations
```

Both keep 524,288 tokens per optimizer step, 1530 steps, all schedules and the same token stream
(apart from the few shard-tail tokens the loader skips).
Without `--accept-deviations` the script prints the exact differences and exits.

- **Ctrl+C once**: stops after the current micro-batch, discards the partial step, runs a short
  attention-window calibration (3 windows × 8 real-data forward+backward micro-batches, no optimizer
  update), then writes the reports. A second Ctrl+C aborts as soon as Python regains control (during
  the calibration it only skips it); the reports are still written.
- The live ETA extrapolates from early-schedule attention windows (64 tokens until step 57, 1792 at the
  end) and is likely low; the final report headlines the window-corrected figure and the projected
  wall-clock (training + validation + compilation). `--no-window-calibration` skips the correction.
- The first step compiles for several minutes; step 0 also runs the original validation pass
  (10.5M tokens) before training starts. ETA appears after `--warmup-steps` (3) steps.
- `--max-temp 82` stops gracefully above 82 °C. Clocks, power and voltage are never changed.
- Checkpoints are off by default. `--checkpoint-every-fraction 0.1` writes one every 153 steps
  (steps 0, 153, …, 1530); `--save-checkpoint-on-exit` also checkpoints the last completed step when
  you stop. See "Checkpoints and consumed data" below.
- A configuration that ran out of memory is recorded in `oom_failures.json` and refused next time
  (`--retry-failed-config` overrides, e.g. after closing other GPU applications).
- Run as a plain `python` process, not with `torchrun` (torchrun turns Ctrl+C into SIGTERM).

## Checkpoints and consumed data

The loss is a function of both the parameters and the data, so every checkpoint comes with the exact
tokens consumed since the previous one. In `checkpoints/<run_id>/` (or `--checkpoint-dir`):

- `ckpt_step{s:06d}.pt` — state after `s` completed optimizer steps: unwrapped model `state_dict`,
  optimizer and scheduler state (omitted with `--checkpoint-weights-only`), the position (shard file,
  index, offset) of the next unconsumed micro-batch for a future resume, the effective config, and
  the sha256 of the pinned file and of the harness.
- `data_steps{a:06d}-{b:06d}.npz` — every micro-batch of steps `a..b`: `tokens` (uint16,
  `[n, T+1]`, with `x = tokens[i, :-1]`, `y = tokens[i, 1:]`), `step`, `micro_batch`, `shard_index`,
  `offset`; the `.json` sidecar lists the shard files with their sha256 (hashed once at startup).
  Micro-batches of a discarded partial step are not recorded.

Before training, the script estimates the disk space for the full run and refuses to start if the
checkpoint directory has less free space. At `--checkpoint-every-fraction 0.1` that is 11 × 3.87 GiB
with optimizer state (11 × 1.40 GiB weights-only) plus 1.50 GiB of data records: ≈ 44.1 GiB
(≈ 16.9 GiB weights-only). On a network filesystem the quota can be lower than the
free space shown. Checkpoint time is excluded from step time and added to the wall-clock projection.

## Outputs (git-ignored)

- `training_feasibility_report.md` / `.json` — the latest report; a run only replaces it if it has at
  least as many timed steps.
- `reports/<run_id>/` — that run's reports (rewritten after every step) and `train.log`
  (pinned code, this script, `nvidia-smi`, original-format step lines).

## Layout

- `upstream/` — unmodified files from the pinned commit + `SHA256SUMS` (read-only); the FineWeb
  shards are downloaded into `upstream/data/fineweb10B/`.
- `train_feasibility.py` — entrypoint. `setup.sh` — environment. `scripts/check_gpu.py` — kernel
  compatibility check. `tests/` — CPU-only tests.
