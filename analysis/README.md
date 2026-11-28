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

### Other geometries at step 918 (lm_head)

- **L∞ smoothness** (`results/linf`): max of xᵀGx over sign vectors x ∈ {±1}ⁿ is at least 4,964 (best of
  10 restarts of sign power iteration, `linf_smooth.py --mode full`). The trivial upper bound n·λ_max is 4.5 × 10⁸.
- **One step on the quadratic** (`results/direction_curvature`): with the best step size for each direction,
  sign descent lowers the loss 86.5× more than gradient descent, Adam's next update 80× more and Adam without
  momentum 104× more. The quadratic predicts the actual loss change within 5% for every direction.

### μ of lm_head at step 1530 (`results/mu_structure/step001530`)

G is full rank, but μ = λ_min(G) ≤ 2.586 × 10⁻¹⁴, so κ = L/μ ≥ 3.9 × 10¹⁴. The bounds come from 768 × 768
blocks of G and small projections of it, not from an eigensolver on the full matrix. "Target count" means
the number of times a token is the target in the 524,288 evaluation tokens.

- **Full rank** (`mu_structure.py`): every softmax probability is positive, so Gx = 0 would need all capped
  logits to move by the same amount at every token. For two rows j, k that needs two 768-dimensional
  subspaces of ℝ^524,288 (the hidden states scaled by each row's soft-cap slopes) to share a direction.
  For rows 13 and 11 the cosine of their smallest principal angle is 1 − 1.9 × 10⁻⁵, against a pessimistic
  rounding error of 1.1 × 10⁻⁷, so the kernel is {0}. Without the soft cap it would be the 768 shift
  directions ΔW = 1vᵀ.
- **Upper bounds**: the row block of token 36130 (target count 0) gives μ ≤ 2.586 × 10⁻¹⁴. 269 rows are
  bitwise identical; since lm_head starts at zero, they cannot have been a target during training (a target
  would have separated them). Differences of identical rows are exact eigenvectors of G, which brings the
  eigenvalues of a 768 × 768 matrix into G, each repeated 268 times; the smallest is 2.686 × 10⁻¹⁴. In all,
  at least 272 eigenvalues lie below 3 × 10⁻¹⁴ and at least 69,493 below 10⁻¹³. So λ₂ − λ₁ < 3 × 10⁻¹⁴
  (3 × 10⁻¹⁵ of λ_max), and no eigensolver can separate λ₁ from the rest; whether λ₁ is far below this
  cluster would need a lower bound. The shift directions have curvature 8.0 × 10⁻¹⁰ to 4.6 × 10⁻⁵.
- **By target count** (`mu_by_frequency.py`, `by_frequency.json`): restricted to the rows of tokens with
  target count at least 1, 10, 100 or 1,000, μ ≤ 2.2 × 10⁻¹³, 4.0 × 10⁻¹¹, 1.9 × 10⁻⁹, 1.1 × 10⁻⁸. The
  restricted block's λ_max is at least 9.97 (its sharpest row), so κ ≥ 9.1 × 10⁸ even for the 44 most
  frequent tokens.
- **Eigensolvers** (`lanczos_feasibility.py`, `lanczos_feasibility.json`): float32 products of G are off by
  6.4 × 10⁻⁷ relative (`mu_float64.log`), far above μ, so float32 cannot resolve it; `mu_block.py`'s
  2.77 × 10⁻⁷ (971 products) is not μ. On a model of the spectrum (row weights × hidden-state eigenvalues,
  38.6M eigenvalues, minimum 4.5 × 10⁻¹⁴), float64 Lanczos without reorthogonalization is at 3.0 × 10⁻¹⁰
  after 60,000 products; extrapolating the last two points (slope −2.7 in log–log) gives about 2 × 10⁶
  products to reach 2.6 × 10⁻¹⁴. ARPACK on a 2M-eigenvalue sample of the model reports a false convergence
  at 5.8 × 10⁻⁷ after 93 products with `mu_block.py`'s settings, and does not converge in 25,528 products
  with `mu_float64.py`'s settings.

Run `mu_structure.py` first (≈ 8 min on an RTX 5070 Ti); it writes `spectrum.npz`, which
`mu_by_frequency.py` and `lanczos_feasibility.py` read.

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
| `mu_structure.py` | μ of lm_head from the structure of G: full-rank certificate, row-block and identical-row bounds, Rayleigh–Ritz, shift directions |
| `mu_by_frequency.py` | upper bounds on μ of lm_head restricted to the rows of frequent tokens |
| `lanczos_feasibility.py` | Lanczos and ARPACK on a matrix with the spectrum shape of lm_head's G and a known minimum |
| `jacobian_nullspace.py` | exact rank of the lm_head Jacobian (before the softmax) from the eigenvalues of its 768 × 768 row blocks, and its null vectors |
| `linf_smooth.py` | L∞ smoothness of one block: max of xᵀGx over sign vectors, by sign power iteration (`--mode small` checks it against brute force) |
| `direction_curvature.py` | one step of GD, sign descent and Adam on the quadratic model of lm_head: best step size, predicted and actual loss decrease (needs the full checkpoint with the optimizer state) |
| `compare.py` | tables and plots of one experiment from the index |
| `val_check.py` | sanity check: validation loss of a checkpoint |
| `download_shards.py`, `cached_fineweb100B.py` | FineWeb shard download (10B sample used for final results) |
| `scripts/` | `run_all.sh` (sweep), `run_resume.sh` (resume the sweep), `run_wstudy.sh` (window study), `run_check_now.sh` (random-start checks) |
| `results/<experiment>/` | `checkpoints` (main sweep), `W_study_918` (number of windows), `warm_check` (random starts): one `.json` per run, `summary.csv`, overview plots; `linf`, `direction_curvature`: one `.json` per run; `mu_structure/step001530`: μ of lm_head |
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
- Only the lm_head block so far (lower bound on the full L); next: one attention block, then MLP and
  embedding blocks, Adam-preconditioned curvature, Muon.
- μ of lm_head has only an upper bound. A lower bound needs a preconditioned eigensolver (diagonal row
  blocks plus a correction for the shift directions).
- The quadratic model is a weak predictor of the dynamics before ≈ 50–60% of training (paper, Table 1).
