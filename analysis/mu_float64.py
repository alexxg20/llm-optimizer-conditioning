"""mu for the output layer (lm_head) at one checkpoint, in float64, with a checked error bound.

Steps:
  1. load the model, the 128 windows and the final hidden states (as in mu_block.py)
  2. build G in float64, one chunk of tokens at a time
  3. check the float64 product against the float32 product, and check symmetry
  4. Lanczos for the smallest eigenvalue, with a tight tolerance; save the eigenvector
  5. residual check: a true eigenvalue lies within ||r|| of the Rayleigh quotient
  6. decide: full rank (mu = smallest eigenvalue) or numerically singular
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh, ArpackNoConvergence

from load import load_model
from prox_hvp import cache_hidden, make_matvec_lmhead

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--step", type=int, required=True)
p.add_argument("--shard", required=True)
p.add_argument("--lam_max", type=float, required=True, help="lambda_max from run_power.py (sets the shift)")
p.add_argument("--S", type=int, default=4096, help="tokens per window")
p.add_argument("--W", type=int, default=128, help="number of windows")
p.add_argument("--chunk", type=int, default=256, help="tokens per float64 chunk (lower it if GPU memory runs out)")
p.add_argument("--tol", type=float, default=1e-9, help="relative tolerance of the eigensolver")
p.add_argument("--maxiter", type=int, default=1500, help="maximum number of products")
p.add_argument("--zero_tol", type=float, default=1e-9, help="zero threshold, relative to lam_max")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="results")
p.add_argument("--exp", default="mu_float64")
args = p.parse_args()


def load_shard(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520 and header[1] == 1, "not a valid shard"
    return np.fromfile(path, dtype=np.uint16, count=int(header[2]), offset=256 * 4)


# ------------------------------------------------ 1. windows, model, hidden states
attn_blocksize = int(64 * ((args.step / 1530 * (1792 - 64) + 64) // 64))
tokens = load_shard(args.shard)
starts = np.linspace(0, len(tokens) - args.S - 1, args.W).astype(int)
windows = [torch.from_numpy(tokens[s : s + args.S].astype(np.int64)).cuda() for s in starts]

model = load_model(args.ckpt, bf16_embeddings=True)
model.requires_grad_(False)
W32 = model.lm_head.weight.detach().float().clone()           # float32 copy on the GPU
V, d = W32.shape
n = V * d
hs = cache_hidden(model, windows, attn_blocksize)              # list of (S, d) float32, on the CPU
del model, windows
torch.cuda.empty_cache()
print(f"lm_head: V={V}, d={d}, n={n:,}; {len(hs)} windows", flush=True)

S, nW, CH = args.S, len(hs), args.chunk
W64 = W32.double()
count = {"products": 0}
t0 = time.time()


# ------------------------------------------------ 2. G in float64, token chunk by token chunk
def G64(vec):
    """Gauss-Newton matrix (averaged over windows) times a flat vector of length n, in float64."""
    count["products"] += 1
    if count["products"] % 10 == 0:
        print(f"   product {count['products']}, {time.time() - t0:.0f} s", flush=True)
    v = torch.as_tensor(np.asarray(vec, dtype=np.float64)).reshape(V, d).cuda()
    acc = torch.zeros(V, d, dtype=torch.float64, device="cuda")
    for h_cpu in hs:
        for c0 in range(0, h_cpu.shape[0], CH):
            H = h_cpu[c0 : c0 + CH].cuda().double()             # (C, d) hidden states of this chunk
            T = torch.tanh((H @ W64.T) / 30)                    # tanh(z / 30), z = pre-cap logits
            F = 30 * T                                          # soft-capped logits
            D = 1 - T * T                                       # derivative of the soft cap
            Av = D * (H @ v.T)                                  # J v for each token, (C, V)
            P = torch.softmax(F, dim=-1)
            dot = (P * Av).sum(dim=-1, keepdim=True)
            U = (P * Av - P * dot) / S                          # u, including the 1/S
            acc += (D * U).T @ H                                # J^T u, summed over the tokens
    return (acc / nW).reshape(-1).cpu().numpy()


# ------------------------------------------------ 3. checks
rng = np.random.default_rng(args.seed)
x = rng.standard_normal(n)
y64 = G64(x)
mv32, _ = make_matvec_lmhead(W32, hs)                           # the float32 operator used before
y32 = mv32(x.astype(np.float32)).astype(np.float64)
rel_prec = np.linalg.norm(y64 - y32) / np.linalg.norm(y64)
print(f"[check] float64 vs float32 product, relative difference: {rel_prec:.3e}", flush=True)

y = rng.standard_normal(n)
Gy = G64(y)
sym_a, sym_b = float(x @ Gy), float(y @ y64)                    # x^T G y and y^T G x: must agree
print(f"[check] symmetry, relative difference: {abs(sym_a - sym_b) / abs(sym_a):.3e}", flush=True)

# ------------------------------------------------ 4. smallest eigenvalue (Lanczos on c*I - G)
lam = args.lam_max
c = 2.0 * lam


def B_apply(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return c * x - G64(x)


B = LinearOperator((n, n), matvec=B_apply, dtype=np.float64)
try:
    vals, vecs = eigsh(B, k=1, which="LA", tol=args.tol, maxiter=args.maxiter)
    converged = True
except ArpackNoConvergence as e:
    print("[eig] did not converge within maxiter; keeping the best approximation", flush=True)
    vals, vecs = e.eigenvalues, e.eigenvectors
    converged = False
    if len(vals) == 0:
        raise SystemExit("no eigenvalue converged: increase --maxiter")

v = vecs[:, 0] / np.linalg.norm(vecs[:, 0])

# ------------------------------------------------ 5. residual: the error bound
Gv = G64(v)
g = float(v @ Gv)                                               # Rayleigh quotient
res = float(np.linalg.norm(Gv - g * v))                         # ||r||: a true eigenvalue lies within this
print(f"\n[eig] converged: {converged}")
print(f"[eig] smallest eigenvalue estimate g = {g:.6e}  (relative to lam: {g / lam:.3e})")
print(f"[eig] residual ||G v - g v|| = {res:.3e}")
print(f"[eig] a true eigenvalue lies in [{g - res:.6e}, {g + res:.6e}]")

# ------------------------------------------------ 6. decision
thr = args.zero_tol * lam
if g - res > thr:
    verdict = "full rank: mu = smallest eigenvalue, which lies in the interval above"
elif g + res <= thr:
    verdict = "numerically singular even in float64: mu needs the smallest nonzero eigenvalue"
else:
    verdict = "undecided: the interval contains the zero threshold; more products or precision needed"
print(f"[decision] zero threshold = {thr:.3e}")
print(f"[decision] {verdict}")

# ------------------------------------------------ 7. save
out = Path(args.out) / args.exp
out.mkdir(parents=True, exist_ok=True)
stamp = time.strftime("%Y%m%d-%H%M%S")
tag = f"lm_head_step{args.step:06d}_W{args.W}_{stamp}"
np.save(out / f"{tag}_eigvec.npy", v.astype(np.float32))
result = {"config": vars(args) | {"n": n}, "precision_check": rel_prec, "symmetry_check": abs(sym_a - sym_b) / abs(sym_a),
          "converged": converged, "g": g, "residual": res, "interval": [g - res, g + res],
          "lam_max": lam, "zero_threshold": thr, "verdict": verdict, "products": count["products"],
          "seconds": time.time() - t0}
(out / f"{tag}.json").write_text(json.dumps(result, indent=2, default=float))
print(f"\nsaved {out / (tag + '.json')}")