"""Can Lanczos find mu of lm_head? A test on a model of its spectrum, where the minimum is known.

Model spectrum: lambda_{j,i} = s_j * gamma_i (row curvature weight x eigenvalue of mean h h^T), 38.6M
eigenvalues, from spectrum.npz (mu_structure.py). It ignores the p p^T coupling between rows, and its
minimum (4.5e-14 at step 1530) is above the bound on the real mu. A diagonal matrix with this spectrum
has correctly rounded float64 products, so the test isolates convergence from the rounding of G.

A. unrestarted Lanczos without reorthogonalization on the full model spectrum, GPU. In exact arithmetic
   this is the best a Krylov method can do with k products; in floating point it is somewhat slower.
B. ARPACK eigsh on c*I - diag (as mu_block.py and mu_float64.py call it), with their settings, on a
   2M-eigenvalue random sample plus the extremes, CPU; products are capped at --cap

  python lanczos_feasibility.py --step 1530
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import LinearOperator, eigsh, ArpackNoConvergence

p = argparse.ArgumentParser()
p.add_argument("--step", type=int, required=True)
p.add_argument("--kmax", type=int, default=60000, help="Lanczos steps in part A")
p.add_argument("--sample", type=int, default=2_000_000, help="eigenvalues in the ARPACK test of part B")
p.add_argument("--cap", type=int, default=30000, help="maximum products per ARPACK run")
p.add_argument("--out", default="results/mu_structure")
args = p.parse_args()
outdir = Path(args.out) / f"step{args.step:06d}"

z = np.load(outdir / "spectrum.npz")
lam = np.multiply.outer(z["s"], z["gam"]).ravel()
n, lmax, lmin = lam.size, float(lam.max()), float(lam.min())
print(f"model spectrum: n = {n:,}, lambda_max = {lmax:.4f}, lambda_min = {lmin:.3e}, trace = {lam.sum():.1f}")
x = np.random.default_rng(0).standard_normal(n)
summary = {"n": n, "lambda_max": lmax, "lambda_min": lmin, "trace": float(lam.sum()),
           "random_rayleigh_quotient": float(np.sum(lam * x * x) / np.sum(x * x)), "config": vars(args)}
horizon = lmax / args.step
summary["eigenvalues_above_L_over_T"] = int((lam > horizon).sum())
print(f"  eigenvalues above L/T = {horizon:.2e} (T = {args.step} steps): {summary['eigenvalues_above_L_over_T']:,}")

# ------------------------------------------------ A. unrestarted Lanczos, float64 on the GPU
L = torch.from_numpy(lam).cuda()
q = torch.from_numpy(x).cuda()
q /= q.norm()
q_prev = torch.zeros_like(q)
beta = 0.0
al, be = [], []
checks = sorted({c for c in [10, 30, 100, 300, 1000, 3000, 10000, 30000, 60000] if c <= args.kmax} | {args.kmax})
lanczos = []
t0 = time.time()
print("\nA. unrestarted Lanczos")
print(f"{'k':>7} {'smallest Ritz value':>20} {'/ lambda_min':>13}")
for k in range(1, args.kmax + 1):
    w = L * q - beta * q_prev
    a = torch.dot(q, w).item()
    w -= a * q
    b = w.norm().item()
    al.append(a)
    be.append(b)
    q_prev, q, beta = q, w / b, b
    if k in checks:
        th = eigh_tridiagonal(np.array(al), np.array(be[:-1]), eigvals_only=True, select="i", select_range=(0, 0))[0]
        lanczos.append({"k": k, "ritz_min": float(th)})
        print(f"{k:7d} {th:20.3e} {th / lmin:13.2e}   ({time.time() - t0:.0f} s)", flush=True)
summary["lanczos"] = lanczos
del L, q, q_prev, w
torch.cuda.empty_cache()

# ------------------------------------------------ B. ARPACK with the settings of mu_block.py and mu_float64.py
rng = np.random.default_rng(1)
samp = np.concatenate([[lmin, lmax], rng.choice(lam, args.sample, replace=False)])
del lam
c = 2 * lmax
cnt = {"k": 0}


class Cap(Exception):
    pass


def B(v):
    cnt["k"] += 1
    if cnt["k"] > args.cap:
        raise Cap
    return (c - samp) * v.ravel()


op = LinearOperator((samp.size, samp.size), matvec=B, dtype=np.float64)
print(f"\nB. ARPACK on c*I - diag (c = 2 lambda_max), n = {samp.size:,}, true minimum {samp.min():.3e}, cap {args.cap:,} products")
arpack = []
for name, tol, maxiter in [("mu_block.py settings (tol 1e-6, default maxiter)", 1e-6, None),
                           ("mu_float64.py settings (tol 1e-9, maxiter 1500)", 1e-9, 1500)]:
    cnt["k"] = 0
    t0 = time.time()
    res = {"settings": name, "true_min": float(samp.min())}
    try:
        vals, vecs = eigsh(op, k=1, which="LA", tol=tol, maxiter=maxiter, v0=rng.uniform(-1, 1, samp.size))
        v = vecs[:, 0] / np.linalg.norm(vecs[:, 0])
        res.update(converged=True, products=cnt["k"], reported_min=float(c - vals[0]),
                   rayleigh_quotient=float(np.sum(samp * v * v)))
    except ArpackNoConvergence:
        res.update(converged=False, products=cnt["k"])
    except Cap:
        res.update(converged=False, products=cnt["k"] - 1, capped=True)
    res["seconds"] = time.time() - t0
    arpack.append(res)
    print(f"  {name}: " + ", ".join(f"{k} = {v:.3e}" if isinstance(v, float) else f"{k} = {v}"
                                    for k, v in res.items() if k != "settings"), flush=True)
summary["arpack"] = arpack
(outdir / "lanczos_feasibility.json").write_text(json.dumps(summary, indent=2))
print(f"saved {outdir / 'lanczos_feasibility.json'}")
