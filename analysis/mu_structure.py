"""mu of lm_head from the structure of G: full-rank certificate, upper bounds on the bottom eigenvalues.

G = (1/N) sum_t J_t^T (diag p_t - p_t p_t^T) J_t,  (J_t x)_k = D_tk h_t . x_k,  D = softcap slope,
so x^T G x = (1/N) sum_t Var_{p_t}(u_t) with u_tk = D_tk h_t . x_k (how x moves the capped logits).

1. Full rank. Every p_tk > 0 (the cap keeps p >= e^-60 / V), so G x = 0 forces u_t to be constant
   over k at every t: D_k * (H x_k) = c for every row k. For two rows j, k this needs the subspaces
   diag(D_j) col(H) and diag(D_k) col(H) of R^N to share a direction. If their smallest principal angle
   is > 0, then c = 0, so H x_k = 0 for every row and x = 0 (H has full column rank).
2. Row blocks G_jj = (1/N) sum_t p(1-p) D^2 h h^T for the --nsel rows with the smallest
   s_j = (1/N) sum_t p(1-p) D^2: float32 screen, the best 16 again in float64.
   lambda_min(G) <= lambda_min(G_jj) (Cauchy interlacing).
3. Identical rows: the difference of two identical rows (times y) is an exact eigenvector of G when y is
   an eigenvector of Gt = (1/N) sum_t p D^2 h h^T, so lambda_min(Gt) is an exact eigenvalue of G,
   repeated (group size - 1) times.
4. Rayleigh-Ritz on X = [e_j (x) y_j] (y_j bottom eigenvector of G_jj, non-identical rows):
   lambda_i(G) <= theta_i(X^T G X). The identical-row eigenvectors span an invariant subspace orthogonal to X,
   so x^T G z = 0 for x in X, z in it, and the two sets of upper bounds combine.
5. Residual r = G x - theta x of the best row vector, and one Rayleigh-Ritz step on span{x, r}.
6. Block-Jacobi ratio x^T G x / x^T B x on shift directions 1_S v^T, B = blockdiag(G_jj): how far
   a block-diagonal preconditioner is from G (relevant for a two-sided bracket).

  python mu_structure.py --ckpt $W8/model_step001530.safetensors --step 1530 --shard $VAL10 --lam_max 10.15625
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import torch

from load import load_model
from prox_hvp import cache_hidden

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--step", type=int, required=True)
p.add_argument("--shard", required=True)
p.add_argument("--lam_max", type=float, required=True, help="lambda_max from run_power.py (for kappa)")
p.add_argument("--S", type=int, default=4096)
p.add_argument("--W", type=int, default=128)
p.add_argument("--chunk", type=int, default=1024, help="tokens per chunk")
p.add_argument("--nsel", type=int, default=2048, help="rows whose blocks are screened")
p.add_argument("--batch", type=int, default=256, help="rows per screening batch (lower it if GPU memory runs out)")
p.add_argument("--out", default="results/mu_structure")
args = p.parse_args()
MS = [10, 100, 1000, 10000]


def load_shard(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520 and header[1] == 1, "not a valid shard"
    return np.fromfile(path, dtype=np.uint16, count=int(header[2]), offset=256 * 4)


t0 = time.time()
log = lambda msg: print(f"[{time.time() - t0:5.0f} s] {msg}", flush=True)

# ------------------------------------------------ setup (same windows as run_power.py)
attn_blocksize = int(64 * ((args.step / 1530 * (1792 - 64) + 64) // 64))
tokens = load_shard(args.shard)
starts = np.linspace(0, len(tokens) - args.S - 1, args.W).astype(int)
windows = [torch.from_numpy(tokens[s : s + args.S].astype(np.int64)).cuda() for s in starts]
targets = np.concatenate([tokens[s + 1 : s + args.S + 1] for s in starts]).astype(np.int64)

model = load_model(args.ckpt, bf16_embeddings=True)
model.requires_grad_(False)
Wt = model.lm_head.weight.detach().float().clone()
hs = cache_hidden(model, windows, attn_blocksize)
del model, windows
torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = False          # the float32 screen needs full float32 products
torch.set_float32_matmul_precision("highest")
Hall = torch.cat(hs).cuda()
del hs
V, d = Wt.shape
N = Hall.shape[0]
freq = np.bincount(targets, minlength=V)
chunks = [(a, min(a + args.chunk, N)) for a in range(0, N, args.chunk)]

_, inv, cnt = np.unique(Wt.cpu().numpy(), axis=0, return_inverse=True, return_counts=True)
inv = inv.ravel()
groups = [np.flatnonzero(inv == g) for g in np.flatnonzero(cnt > 1)]
identical = np.zeros(V, dtype=bool)
for g in groups:
    identical[g] = True
log(f"setup; V = {V:,}, d = {d}, N = {N:,}; {identical.sum()} identical rows in {len(groups)} group(s), "
    f"largest {max(map(len, groups), default=0)}; target counts of identical rows: max {freq[identical].max(initial=0)}")

# ------------------------------------------------ pass A: s_j, lse, p_min, C, shift ratios, certificate Gram blocks
f_order = np.argsort(-freq, kind="stable")
never = np.flatnonzero((freq == 0) & ~identical)
nz = np.flatnonzero(freq > 0)
med = nz[np.argsort(freq[nz], kind="stable")][len(nz) // 2 : len(nz) // 2 + 2]
pairs = [(int(f_order[0]), int(f_order[1])), (int(f_order[0]), int(never[0])), (int(med[0]), int(med[1]))]
prow = sorted({j for pr in pairs for j in pr})
pidx = {j: i for i, j in enumerate(prow)}
pkeys = sorted({key for j, k in pairs for key in [(j, j), (j, k), (k, k)]})
Apair = {key: torch.zeros(d, d, dtype=torch.float64, device="cuda") for key in pkeys}
Mf = torch.zeros(V, len(MS), device="cuda")
for i, m in enumerate(MS):
    Mf[torch.from_numpy(f_order[:m]).cuda(), i] = 1

s = torch.zeros(V, dtype=torch.float64, device="cuda")
C = torch.zeros(d, d, dtype=torch.float64, device="cuda")
Knum = torch.zeros(len(MS) + 1, d, d, dtype=torch.float64, device="cuda")
Kden = torch.zeros(len(MS) + 1, d, d, dtype=torch.float64, device="cuda")
lse_all = torch.empty(N, device="cuda")
pmin = 1.0
for a, b in chunks:
    H = Hall[a:b]
    Hd = H.double()
    T = torch.tanh(H @ Wt.T / 30)
    F, D = 30 * T, 1 - T * T
    lse = torch.logsumexp(F, dim=1, keepdim=True)
    lse_all[a:b] = lse.squeeze(1)
    P = torch.exp(F - lse)
    pmin = min(pmin, P.min().item())
    PD = P * D
    PDD = PD * D
    w = PDD * (1 - P)
    s += w.sum(0, dtype=torch.float64)
    mD = PD.sum(1, keepdim=True)
    num = [(P * (D - mD) ** 2).sum(1, dtype=torch.float64)]          # Var_p(D): shift of all rows
    den = [w.sum(1, dtype=torch.float64)]
    e2, e1, wd = (PDD @ Mf).double(), (PD @ Mf).double(), (w @ Mf).double()
    num += list((e2 - e1 ** 2).T)                                     # Var_p(D * 1_S): shift of the rows in S
    den += list(wd.T)
    for i in range(len(MS) + 1):
        Knum[i] += (Hd.T * num[i]) @ Hd
        Kden[i] += (Hd.T * den[i]) @ Hd
    C += Hd.T @ Hd
    Dp = D[:, prow].double()
    for key in pkeys:
        Apair[key] += (Hd.T * (Dp[:, pidx[key[0]]] * Dp[:, pidx[key[1]]])) @ Hd
    del T, F, D, P, PD, PDD, w
s /= N; C /= N; Knum /= N; Kden /= N
for key in Apair:
    Apair[key] /= N
s_np = s.cpu().numpy()
evC = np.linalg.eigvalsh(C.cpu().numpy())
log("pass A done")

print(f"\nC = mean h h^T: lambda_min {evC[0]:.4e}, lambda_max {evC[-1]:.4e}")
print(f"smallest p_tk: {pmin:.3e}   (soft-cap floor e^-60/V = {np.exp(-60) / V:.3e})")
print("\n1. Full rank: smallest principal angle between diag(D_j) col(H) and diag(D_k) col(H)")
cert = []
for j, k in pairs:
    Ajj, Ajk, Akk = (Apair[x].cpu().numpy() for x in [(j, j), (j, k), (k, k)])
    Lj, Lk = np.linalg.cholesky(Ajj), np.linalg.cholesky(Akk)
    M = sla.solve_triangular(Lj, sla.solve_triangular(Lk, Ajk.T, lower=True).T, lower=True)
    cos = float(np.linalg.svd(M, compute_uv=False)[0])
    kap = float(max(np.linalg.cond(Ajj), np.linalg.cond(Akk)))
    # pessimistic error on cos: Cholesky and accumulation in float64 (d * cond * eps), plus the float32
    # rounding of D, which turns each subspace by an angle of order 1e-7
    err = d * kap * 2.2e-16 + 1e-7
    cert.append({"rows": [j, k], "targets": [int(freq[j]), int(freq[k])], "cos": cos, "one_minus_cos": 1 - cos,
                 "angle_rad": float(np.arccos(min(cos, 1.0))), "cond": kap, "cos_error_estimate": err})
    print(f"   rows ({j:5d}, {k:5d}), targets ({freq[j]:6d}, {freq[k]:6d}): 1 - cos = {1 - cos:.3e} "
          f"(angle {cert[-1]['angle_rad']:.3e} rad); pessimistic error on cos {err:.1e}")
full_rank = all(c["one_minus_cos"] > 10 * c["cos_error_estimate"] for c in cert) and evC[0] > 0 and pmin > 0
print(f"   -> {'G is nonsingular (kernel = {0})' if full_rank else 'no certificate'}")

print("\n6. Block-Jacobi ratio x^T G x / x^T B x on shift directions 1_S v^T:")
shift = []
for i, name in enumerate(["all rows"] + [f"top {m} targets" for m in MS]):
    g = sla.eigh(Knum[i].cpu().numpy(), Kden[i].cpu().numpy(), eigvals_only=True)
    shift.append({"S": name, "ratio_min": float(g[0]), "ratio_max": float(g[-1])})
    print(f"   S = {name:18s}: min {g[0]:.3e}, max {g[-1]:.3e}")
evA = np.linalg.eigvalsh(Knum[0].cpu().numpy()) / V
print(f"   curvature of the 768 shift directions (eigenvalues of A = K / V): [{evA[0]:.3e}, {evA[-1]:.3e}]")

# ------------------------------------------------ pass B: float32 screen of G_jj, non-identical rows
cand = np.array([j for j in np.argsort(s_np, kind="stable") if not identical[j]][: args.nsel])
nsel = len(cand)
lam8 = np.zeros((nsel, 8))
Y = np.zeros((nsel, d))
for b0 in range(0, nsel, args.batch):
    rows = torch.from_numpy(cand[b0 : b0 + args.batch]).cuda()
    Wb = Wt[rows]
    G = torch.zeros(len(rows), d, d, dtype=torch.float64, device="cuda")
    for a, b in chunks:
        H = Hall[a:b]
        Tb = torch.tanh(H @ Wb.T / 30)
        Pb = torch.exp(30 * Tb - lse_all[a:b, None])
        wb = Pb * (1 - Pb) * (1 - Tb * Tb) ** 2
        G += torch.matmul((wb.T[:, :, None] * H[None]).transpose(1, 2), H).double()
    G = (G / N).cpu().numpy()
    for i in range(G.shape[0]):
        ev, vec = sla.eigh(G[i], subset_by_index=[0, 7])
        lam8[b0 + i], Y[b0 + i] = ev, vec[:, 0]
    log(f"pass B rows {b0 + G.shape[0]}/{nsel}, running min {lam8[: b0 + G.shape[0], 0].min():.4e}")
Y /= np.linalg.norm(Y, axis=1, keepdims=True)

# ------------------------------------------------ pass C: float64 blocks (best rows, identical groups), X^T G X
NTOP = min(16, nsel)
top = np.argsort(lam8[:, 0])[:NTOP]
reps = np.array([g[0] for g in groups], dtype=np.int64)
rows_t = torch.from_numpy(np.concatenate([cand[top], reps])).cuda()
nt = len(rows_t)
rows_a = torch.from_numpy(cand).cuda()
Yg = torch.from_numpy(Y).cuda()
Gt64 = torch.zeros(nt, d, d, dtype=torch.float64, device="cuda")        # G_jj
Gtil = torch.zeros(len(reps), d, d, dtype=torch.float64, device="cuda")  # (1/N) sum p D^2 h h^T, identical groups
dg = torch.zeros(nsel, dtype=torch.float64, device="cuda")
Q = torch.zeros(nsel, nsel, dtype=torch.float64, device="cuda")
for a, b in chunks:
    H = Hall[a:b]
    Hd = H.double()
    Ta = torch.tanh(H @ Wt[rows_a].T / 30)
    Pa = torch.exp((30 * Ta - lse_all[a:b, None]).double())
    g = (1 - Ta * Ta).double() * (Hd @ Yg.T)
    q = Pa * g
    dg += (q * g).sum(0)
    Q += q.T @ q
    Tt = torch.tanh(H @ Wt[rows_t].T / 30).double()
    Pt = torch.exp(30 * Tt - lse_all[a:b, None].double())
    PDt = Pt * (1 - Tt * Tt) ** 2
    Gt64 += torch.matmul(((PDt * (1 - Pt)).T[:, :, None] * Hd[None]).transpose(1, 2), Hd)
    if len(reps):
        Gtil += torch.matmul((PDt[:, NTOP:].T[:, :, None] * Hd[None]).transpose(1, 2), Hd)
XGX = (torch.diag(dg) - Q).cpu().numpy() / N
Gt64 = Gt64.cpu().numpy() / N
Gtil = Gtil.cpu().numpy() / N
log("pass C done")

ev64 = np.array([sla.eigh(Gt64[i], subset_by_index=[0, 0])[0][0] for i in range(nt)])
print(f"\n2. Row blocks: lambda_min(G_jj), float32 screen of {nsel} rows (s_j up to {s_np[cand[-1]]:.3e})")
best_rows = []
for i in np.argsort(ev64[:NTOP])[:8]:
    j = int(cand[top[i]])
    best_rows.append({"row": j, "targets": int(freq[j]), "s_j": float(s_np[j]),
                      "lambda_min_float32": float(lam8[top[i], 0]), "lambda_min_float64": float(ev64[i])})
    print(f"   row {j:5d} (targets {freq[j]:4d}): s_j {s_np[j]:.3e}, float32 {lam8[top[i], 0]:.5e}, float64 {ev64[i]:.5e}")
counts_rows = {f"{t:.0e}": int((lam8[:, 0] < t).sum()) for t in [3e-14, 1e-13, 1e-12]}
print(f"   rows with lambda_min(G_jj) below 3e-14, 1e-13, 1e-12: {list(counts_rows.values())}")

print("\n3. Identical rows: exact eigenvalues of G")
ident = []
for gi, grp in enumerate(groups):
    evt = np.linalg.eigvalsh(Gtil[gi])                   # each eigenvalue of Gt is one of G, (size - 1) times
    lt = float(evt[0])
    ident.append({"size": int(len(grp)), "first_rows": grp[:12].tolist(), "lambda_min_Gtilde": lt,
                  "Gtilde_eigs_first10": evt[:10].tolist(), "multiplicity": int(len(grp) - 1),
                  "below": {f"{t:.0e}": int((evt < t).sum()) for t in [3e-14, 1e-13, 1e-12]},
                  "lambda_min_Gjj": float(ev64[NTOP + gi])})
    print(f"   group of {len(grp)} rows (e.g. {grp[:6].tolist()}): eigenvalue {lt:.5e}, repeated >= {len(grp) - 1} times "
          f"(its own row block: {ev64[NTOP + gi]:.5e})")

th = np.linalg.eigvalsh(XGX)
rq = np.diag(XGX)
off = XGX - np.diag(rq)
counts = {}
for t in [3e-14, 1e-13, 1e-12]:
    counts[f"{t:.0e}"] = int((th < t).sum()) + sum(g["multiplicity"] * g["below"][f"{t:.0e}"] for g in ident)
print(f"\n4. Rayleigh-Ritz on {nsel} row vectors: lambda_i(G) <= theta_i")
print(f"   theta_1..5 = {np.array2string(th[:5], formatter={'float_kind': lambda v: f'{v:.5e}'})}")
print(f"   theta_10, 100, 1000, {nsel}: {th[9]:.3e}, {th[99]:.3e}, {th[999]:.3e}, {th[-1]:.3e}")
print(f"   theta_1 below the best single-row quotient by {(rq.min() - th[0]) / rq.min():.2e} (relative); "
      f"largest coupling / sqrt(diag diag) = {np.max(np.abs(off) / np.sqrt(np.outer(rq, rq))):.2e}")
print(f"   with the identical-row eigenvectors, eigenvalues of G below 3e-14, 1e-13, 1e-12: at least {list(counts.values())}")

# ------------------------------------------------ pass D: residual of the best row vector
ib = int(np.argmin(ev64[:NTOP]))
jb = int(cand[top[ib]])
yb = np.linalg.eigh(Gt64[ib])[1][:, 0]
theta = float(yb @ Gt64[ib] @ yb)
ybg = torch.from_numpy(yb).cuda()
R = torch.zeros(V, d, dtype=torch.float64, device="cuda")
for a, b in chunks:
    H = Hall[a:b]
    T = torch.tanh(H @ Wt.T / 30)
    P = torch.exp(30 * T - lse_all[a:b, None])
    D = 1 - T * T
    c = (P[:, jb].double() * D[:, jb].double() * (H.double() @ ybg)).float()
    R -= ((P * D) * c[:, None]).T.matmul(H).double()          # (G x)_k for k != jb
    del T, P, D
R /= N
R[jb] = torch.from_numpy(Gt64[ib] @ yb - theta * yb).cuda()
beta = R.norm().item()

# ------------------------------------------------ pass E: r^T G r for the unit residual
Rh = (R / beta).float()
acc = 0.0
for a, b in chunks:
    H = Hall[a:b]
    T = torch.tanh(H @ Wt.T / 30)
    P = torch.exp(30 * T - lse_all[a:b, None])
    U = (1 - T * T) * (H @ Rh.T)
    ub = (P * U).sum(1, keepdim=True)
    acc += (P * (U - ub) ** 2).sum(dtype=torch.float64).item()
    del T, P, U
rho = acc / N
th2 = theta - 2 * beta ** 2 / ((rho - theta) + np.sqrt((rho - theta) ** 2 + 4 * beta ** 2))
log("passes D, E done")
print(f"\n5. Best row {jb}: theta = {theta:.6e}, ||G x - theta x|| = {beta:.3e}, r^T G r / r^T r = {rho:.3e}")
print(f"   one Rayleigh-Ritz step on span(x, r): {th2:.10e} (lower by {(theta - th2) / theta:.1e}, relative)")

ub_mu = min(float(th[0]), min((g["lambda_min_Gtilde"] for g in ident), default=np.inf), float(th2))
print(f"\nmu = lambda_min(G) <= {ub_mu:.4e}; kappa >= {args.lam_max / ub_mu:.2e} (lambda_max = {args.lam_max})")

outdir = Path(args.out) / f"step{args.step:06d}"
outdir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(outdir / "spectrum.npz", s=s_np, gam=evC, freq=freq.astype(np.int32))
summary = {"mu_upper_bound": ub_mu, "kappa_lower_bound": args.lam_max / ub_mu, "lam_max": args.lam_max,
           "full_rank_certificate": full_rank, "pairs": cert, "p_min": pmin,
           "C_eig_min": float(evC[0]), "C_eig_max": float(evC[-1]),
           "shift_eig_A": [float(evA[0]), float(evA[-1])], "block_jacobi_shift_ratios": shift,
           "best_rows": best_rows, "rows_below": counts_rows, "identical_groups": ident,
           "ritz_theta_first10": th[:10].tolist(), "ritz_theta_100_1000_last": [float(th[99]), float(th[999]), float(th[-1])],
           "eigenvalues_below_at_least": counts,
           "residual": {"row": jb, "theta": theta, "norm": beta, "rho": rho, "theta_after_step": th2},
           "seconds": time.time() - t0, "config": vars(args)}
(outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
print(f"saved {outdir / 'summary.json'} and {outdir / 'spectrum.npz'}")
