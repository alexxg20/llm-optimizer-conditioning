# Optimizer conditioning in nanoGPT: curvature of the local quadratic model

We measure the conditioning of the loss landscape along a single training run of a modded-nanogpt
model, starting with the **smoothness constant of gradient descent**: the largest eigenvalue of the
Gauss–Newton matrix G, i.e. the Hessian of the *prox* loss (cross-entropy of the network linearized at
a checkpoint) from

> A. Meterez, P. A. Nair, D. Morwani, C. Pehlevan, S. Kakade, A. Damian,
> **A Defense of the Quadratic Model**, 2026. Paper: https://arxiv.org/pdf/2607.21716

Gradient descent on the local quadratic is stable iff η < 2/λ_max.

The model and checkpoints come from our single-GPU reproduction of modded-nanogpt record #14,
published on Hugging Face: `epfl-optimizer-conditioning/modded-nanogpt-1gpu` (625.8M parameters,
1,530 steps × 524,288 tokens, checkpoints every 153 steps).

## Current results (lm_head block, FineWeb-10B validation, 128 windows × 4,096 tokens)

| step | 0 | 153 | 306 | 459 | 612 | 765 | 918 | 1071 | 1224 | 1377 | 1530 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| λ_max | 0.0006 | 8.87 | 11.39 | 11.78 | 12.79 | **13.07** | 11.71 | 10.80 | 10.89 | 10.52 | 10.16 |

Curvature rises during the first half of training (progressive sharpening), peaks at 50% and decreases
afterwards. Step 0 is degenerate (lm_head is zero-initialized, so predictions are uniform).

How we know the numbers are reliable:
- **Number of windows** (`results/W_study_918`): at step 918, λ_max settles from W = 16 on
  (11.71–11.88, sampling noise ≈ ±1%); W = 128 (one optimizer step of tokens) is used for the sweep.
- **Warm starts** (`results/warm_check`, and seed 1 in `W_study_918`): independent random starts at
  steps 765, 918 and 1530 reproduce the warm-started values exactly (13.0665, 11.7062, 10.1562).
- **Pipeline checks** (development runs, not included here): two random starts agree, and the cached
  lm_head path matches the full-model computation.

Each run in `results/<experiment>/` has a `.json` with its settings, window positions and full convergence
history; `summary.csv` and the overview plots summarise each experiment; `results/index.csv` lists all runs.

## How it works

For each checkpoint θ_ref and token window, G·v = Jᵀ H_ℓ J v is computed without forming J:
a JVP (forward mode) gives J v, the softmax Hessian gives u = (p ⊙ Jv − p (pᵀJv)) / S, and a VJP
(reverse mode) gives Jᵀ u. Products are averaged over windows and fed to a power method
(stop when ‖Gv − λv‖/λ < 1e-3, at least 15 iterations, warm start from the previous checkpoint).

The full 626M-parameter space does not fit on our 9.3 GiB V100, so `--params` selects a parameter
block (regular expression on parameter names); all other parameters stay frozen. λ_max of a block is a
lower bound on λ_max of the full G. For `lm_head`, hidden states are cached once per window (exact).

## Files

| file | role |
|---|---|
| `gpt_model.py` | model definition from the pinned `train_gpt2.py`; attention rewritten explicitly (needed for forward-mode autodiff), `forward(..., return_logits=True)` |
| `load.py` | loads a `.safetensors` checkpoint (strict key match) |
| `linearize.py` | linearized model and its checks (exactness, adjoint test, finite differences) |
| `prox_hvp.py` | G·v for one window, average over windows, cached-hidden-state version for lm_head |
| `power_method.py` | power iteration with convergence history and warm start |
| `run_power.py` | driver for one checkpoint: data windows, parameter block, results + `results/index.csv` |
| `mu_block.py` | μ (smallest, or smallest nonzero, eigenvalue of G) of one block: `lm_head`, `wte` or one attention layer |
| `mu_float64.py` | μ of `lm_head` in float64, with a residual bound and a full-rank check |
| `jacobian_nullspace.py` | exact rank of the lm_head Jacobian (before the softmax) from the eigenvalues of its 768 × 768 row blocks, and its null vectors |
| `linf_smooth.py` | L∞ smoothness of one block: max of xᵀGx over sign vectors, by sign power iteration (`--mode small` checks it against brute force) |
| `compare.py` | tables and plots of one experiment from the index |
| `val_check.py` | sanity check: validation loss of a checkpoint |
| `download_shards.py`, `cached_fineweb100B.py` | FineWeb shard download (10B sample used for final results) |
| `scripts/` | `run_all.sh` (sweep), `run_resume.sh` (resume the sweep), `run_wstudy.sh` (window study), `run_check_now.sh` (random-start checks) |
| `results/<experiment>/` | `checkpoints` (main sweep), `W_study_918` (number of windows), `warm_check` (random starts): one `.json` per run, `summary.csv`, overview plots |
| `results/index.csv` | one row per run, all experiments |
| `logs/` | raw logs of the runs |

Eigenvectors (`*_eigvec.npy`, 154 MB each), data shards and model weights are not in the repository.

## Running

Reference environment: EPFL Gnoto with a Tesla V100 (9.3 GiB, compute capability 7.0). Its image
provides Python 3.13 and PyTorch 2.9.1 + CUDA 12.8; the other packages are pinned in `requirements.txt`.
The published results were computed with Python 3.12 and PyTorch 2.5.1 + CUDA 12.1.

```bash
git clone https://github.com/alexxg20/llm-optimizer-conditioning.git
cd llm-optimizer-conditioning/analysis
python -m venv --system-site-packages .venv   # reuses the image's PyTorch
.venv/bin/pip install -r requirements.txt
source env.sh                                 # venv, $W8 / $D10 / $VAL10, cd into analysis/
python download_shards.py 1 2 10 11           # validation shard + some training shards -> fineweb10B/
hf auth login                                 # the weights repository is private (organization members)
hf download epfl-optimizer-conditioning/modded-nanogpt-1gpu \
    weights/model_step000918.safetensors --local-dir .   # -> weights/
# all 11 checkpoints (~15.4 GiB), needed by scripts/run_all.sh:
#   hf download epfl-optimizer-conditioning/modded-nanogpt-1gpu --include "weights/*" --local-dir .
python run_power.py --ckpt $W8/model_step000918.safetensors --step 918 --shard $VAL10 \
    --S 4096 --W 128 --params lm_head --exp my_test
python compare.py --exp my_test --x step
python linf_smooth.py --ckpt $W8/model_step000918.safetensors --step 918 --shard $VAL10 \
    --block lm_head --mode small
setsid nohup bash scripts/run_all.sh > logs/run_all.txt 2>&1 < /dev/null &   # full sweep, ~4 h
```

- `source <clone>/analysis/env.sh` works from any directory. To keep weights, shards or a venv
  elsewhere, set `VENV`, `W8`, `D10` or `VAL10` in `analysis/env.local.sh` (gitignored), e.g.
  `W8=/scratch/weights`, and download there (`--local-dir /scratch` for `hf download`).
- Without the Gnoto image, create the venv without `--system-site-packages` and install PyTorch first:
  `.venv/bin/pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128`.
- `val_check.py` reads the FineWeb-100B validation shard: `python cached_fineweb100B.py 0`.

## Open points

- Dependence of λ_max on the data (seen vs unseen text, other shards) at the late checkpoints: not yet measured.
- Only the lm_head block so far (lower bound on the full L); next: attention/MLP and embedding blocks,
  Adam-preconditioned curvature, Muon.
- The quadratic model is a weak predictor of the dynamics before ≈ 50–60% of training (paper, Table 1).
