# Single-GPU pretraining: modded-nanogpt record #14

Runs the pinned [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) training
(`upstream/train_gpt2.py`, commit `9730304`, record #14 "Value Embeddings") on one GPU, with a live
ETA, checkpoints every N % of training, a record of the exact tokens consumed between checkpoints,
and a report at the end.

Model, Muon/Adam optimizers, learning rates and schedules, tokens per step (524,288), number of steps
(1530) and the FineWeb token stream are those of the original. `train_feasibility.py` checks the sha256
of the pinned file and executes its definitions unmodified; only the main loop is replicated, with
measurement, checkpointing and safety hooks added.

- Results of the run we did: [`RESULTS.md`](RESULTS.md).
- Its checkpoints and data: [Hugging Face](https://huggingface.co/epfl-optimizer-conditioning/modded-nanogpt-1gpu)
  (private, `epfl-optimizer-conditioning` members). The run was produced by the code at tag
  `run-20261115-232515`.

All commands run from `pretraining/`.

## Requirements

- An NVIDIA GPU with native bf16 (compute capability ≥ 8.0) and about 10 GiB of free memory.
  Without native bf16, `torch.compile` skips the model's bf16 graphs: it then runs uncompiled, far
  slower, and out of memory (observed on a V100).
- About 2 GB for the FineWeb shards, and up to 44 GiB for checkpoints (below).

## Setup

`setup.sh` creates `.venv`, installs PyTorch 2.14.1 for the detected GPU (override with
`TORCH_INDEX=<url>`) and `requirements.txt`, then checks that the build has kernels for the GPU.

```bash
mise trust && mise install               # with mise: Python 3.12 from mise.toml
bash setup.sh                            # without mise: PYTHON=python3.12 bash setup.sh
.venv/bin/python scripts/check_gpu.py    # ~1-2 min: bf16, FlexAttention, rms_norm, fused Adam
.venv/bin/python upstream/data/cached_fineweb10B.py 9    # validation + 9 training shards (~2 GB)
CUDA_VISIBLE_DEVICES= .venv/bin/python -m unittest discover -s tests   # CPU-only tests
```

With mise, `mise run setup`, `mise run data`, `mise run test` and `mise run preflight` do the same.

## Why the original batch size does not fit

The original runs on 8 GPUs, each processing one micro-batch of 65,536 tokens per step. A single
micro-batch of that size needs about 20 GiB (weights, a 6.1 GiB bf16 logits tensor and the saved
activations), and a steady-state step about 50 GiB. On a ~12 GB GPU, the script therefore refuses the
original configuration; `--preflight-only` prints the memory analysis without training.

The configuration we use keeps 524,288 tokens per optimizer step by accumulating gradients:

| | original | used here |
|---|---|---|
| micro-batch | 65,536 tokens | 4,096 tokens (`--micro-batch-tokens 4096`) |
| micro-batches per step | 8 (one per GPU) | 128 (gradient accumulation) |
| DDP wrapper | yes | no (`--no-ddp`; it does nothing on one GPU and saves 1.4 GiB) |

Each micro-batch is the mean loss over the same number of tokens, so averaging 128 of them gives the
same gradient as the original 8, up to floating-point rounding. The differences: attention context is
cut at every 4,096-token boundary instead of every 65,536, and the bf16 gradients of the embeddings and
scalars are summed over 128 micro-batches instead of 8. `RESULTS.md` measures their effect.

## Training

```bash
.venv/bin/python train_feasibility.py --micro-batch-tokens 4096 --no-ddp --accept-deviations \
    --checkpoint-every-fraction 0.1
```

Without `--accept-deviations` the script lists the differences from the original and exits.

- The first step compiles for a few minutes; step 0 also runs the original validation pass (10.5M
  tokens). The ETA appears after 3 steps.
- **Ctrl+C once** stops after the current micro-batch, discards the partial step and writes the
  report. It first runs a short calibration, because early steps use a small attention window (64
  tokens, growing to 1,792) and would otherwise make the estimate too low. A second Ctrl+C aborts.
- Run it as a plain `python` process, not with `torchrun` (which turns Ctrl+C into SIGTERM).
- A configuration that ran out of memory is recorded in `oom_failures.json` and refused next time
  (`--retry-failed-config` overrides).

### Long runs

To keep a run going after the terminal closes:

```bash
bash scripts/run_detached.sh --micro-batch-tokens 4096 --no-ddp --accept-deviations --checkpoint-every-fraction 0.1
tail -f logs/latest.log      # Ctrl+C here only stops tail
bash scripts/stop_run.sh     # same as one Ctrl+C; call it again to abort
```

The machine must not suspend during the run.

### Temperature

When the GPU reaches `--cool-at` (82 °C), training pauses after the current micro-batch until it has
cooled to `--resume-at` (77 °C). A pause only waits between micro-batches, so the trained model is the
same; pause time is excluded from step times and added to the ETA. Above `--max-temp` (87 °C), or if
one pause lasts longer than `--max-cool-wait` (900 s), the run stops cleanly. Clocks, power limits and
voltage are never changed.

## Checkpoints and consumed data

`--checkpoint-every-fraction 0.1` saves at steps 0, 153, 306, …, 1530 into `checkpoints/<run_id>/`:

- `ckpt_stepNNNNNN.pt`: the model weights after that many optimizer steps, the state of the four
  optimizers and schedulers, the data position, the configuration and the sha256 of the code
  (3.87 GiB each; 1.40 GiB with `--checkpoint-weights-only`).
- `data_stepsAAAAAA-BBBBBB.npz` (+ `.json`): every 4,097-token window consumed by steps A..B, in order,
  with its shard, offset, step and micro-batch (`x = tokens[i, :-1]`, `y = tokens[i, 1:]`). The `.json`
  describes the format and lists the sha256 of each shard.

The script checks the free disk space before starting (≈ 44 GiB for the full run, ≈ 17 GiB
weights-only).

For our run these are published on Hugging Face:
[`weights/`](https://huggingface.co/epfl-optimizer-conditioning/modded-nanogpt-1gpu/tree/main/weights)
(weights only, safetensors),
[`full/`](https://huggingface.co/epfl-optimizer-conditioning/modded-nanogpt-1gpu/tree/main/full)
(full checkpoints with optimizer state) and
[`data/`](https://huggingface.co/epfl-optimizer-conditioning/modded-nanogpt-1gpu/tree/main/data)
(consumed tokens). The model card there shows how to load them.

`scripts/eval_checkpoint.py <ckpt>` re-runs the validation of a checkpoint at chosen chunk lengths
(`--chunk-tokens`).

## Files

| path | contents |
|---|---|
| `upstream/` | unmodified files from the pinned commit, with `SHA256SUMS`; shards download to `upstream/data/fineweb10B/` |
| `train_feasibility.py` | the training script |
| `setup.sh`, `requirements.txt`, `mise.toml` | environment |
| `scripts/` | `check_gpu.py`, `eval_checkpoint.py`, `run_detached.sh`, `stop_run.sh` |
| `tests/` | CPU-only tests |

Git-ignored outputs: `training_feasibility_report.{md,json}` (latest report), `reports/<run_id>/`
(per-run reports and `train.log`), `checkpoints/<run_id>/`, `logs/`, `run.pid`.
