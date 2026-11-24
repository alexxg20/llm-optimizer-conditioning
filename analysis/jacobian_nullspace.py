"""Exact rank of the lm_head Jacobian J (before the softmax), and its null vectors.

J^T J is block diagonal over vocabulary rows j:  (J^T J v)_j = M_j v_j,
  M_j = (1/N) sum_t D_tj^2 h_t h_t^T,   D = softcap slope,   N = number of tokens.

Pass 1: all eigenvalues of every block -> rank and nullity of J.
Pass 2: only for blocks with near-zero eigenvalues, recompute M_j and save the null
        eigenvectors u. Then x = e_j (x) u (zero except row j) satisfies J x = 0,
        so G x = 0 for the actual Gauss-Newton matrix G.
Resumable: pass 1 saves each chunk to disk.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch

from load import load_model
from prox_hvp import cache_hidden

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--step", type=int, required=True)
p.add_argument("--shard", required=True)
p.add_argument("--S", type=int, default=4096)
p.add_argument("--W", type=int, default=128)
p.add_argument("--tchunk", type=int, default=1024, help="tokens per chunk")
p.add_argument("--jchunk", type=int, default=64, help="vocabulary rows per chunk")
p.add_argument("--zero_tol", type=float, default=1e-6, help="zero threshold, relative to the largest eigenvalue")
p.add_argument("--out", default="results/jacobian_nullspace")
args = p.parse_args()


def load_shard(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520 and header[1] == 1, "not a valid shard"
    return np.fromfile(path, dtype=np.uint16, count=int(header[2]), offset=256 * 4)


# ------------------------------------------------ setup (same as the other scripts)
attn_blocksize = int(64 * ((args.step / 1530 * (1792 - 64) + 64) // 64))
tokens = load_shard(args.shard)
starts = np.linspace(0, len(tokens) - args.S - 1, args.W).astype(int)
windows = [torch.from_numpy(tokens[s : s + args.S].astype(np.int64)).cuda() for s in starts]

model = load_model(args.ckpt, bf16_embeddings=True)
model.requires_grad_(False)
W = dict(model.named_parameters())["lm_head.weight"].detach().float().cuda()   # (V, d)
V, d = W.shape
hs = cache_hidden(model, windows, attn_blocksize)     # per window (S, d), on the CPU
del model
torch.cuda.empty_cache()
nW, S = len(hs), args.S
N = nW * S
n = V * d
print(f"lm_head: V = {V:,}, d = {d}, n = V*d = {n:,}; {nW} windows x {S} tokens", flush=True)

outdir = Path(args.out) / f"step{args.step:06d}"
eig_dir = outdir / "eigs"
eig_dir.mkdir(parents=True, exist_ok=True)
t0 = time.time()


def block_mats(idx):
    """M_j for the vocabulary rows in idx (a cuda LongTensor), as float64 (len(idx), d, d)."""
    Wc = W[idx]
    Mtot = torch.zeros(len(idx), d, d, device="cuda", dtype=torch.float64)
    for h_cpu in hs:
        for c0 in range(0, S, args.tchunk):
            H = h_cpu[c0 : c0 + args.tchunk].cuda().float()
            T = torch.tanh(H @ Wc.T / 30)
            w = (1 - T * T) ** 2                              # D^2, shape (tokens, rows)
            A = w.T.unsqueeze(-1) * H.unsqueeze(0)            # (rows, tokens, d)
            Mtot += (A.transpose(1, 2) @ H).double()          # sum_t D^2 h h^T
    return Mtot / N


# ------------------------------------------------ pass 1: all eigenvalues of all blocks
for j0 in range(0, V, args.jchunk):
    j1 = min(V, j0 + args.jchunk)
    f = eig_dir / f"eig_{j0:06d}.npy"
    if f.exists():
        continue
    idx = torch.arange(j0, j1, device="cuda")
    np.save(f, torch.linalg.eigvalsh(block_mats(idx)).cpu().numpy())   # (rows, d), ascending
    print(f"  pass 1: rows {j0}-{j1 - 1} done, {time.time() - t0:.0f} s", flush=True)

all_eig = np.concatenate([np.load(eig_dir / f"eig_{j0:06d}.npy") for j0 in range(0, V, args.jchunk)])
largest = float(all_eig.max())
thr = args.zero_tol * largest
rank = int((all_eig > thr).sum())
nullity = n - rank
positive = all_eig[all_eig > thr]
smallest_pos = float(positive.min())
print(f"\n[eig] largest eigenvalue of J^T J: {largest:.6e}  (sigma_max = {np.sqrt(largest):.6e})")
print(f"[eig] smallest eigenvalue above threshold: {smallest_pos:.6e}  (sigma_min = {np.sqrt(smallest_pos):.6e})")
print(f"[eig] smallest eigenvalue overall: {float(all_eig.min()):.6e}")
print(f"[decision] zero threshold (relative {args.zero_tol:g}): {thr:.3e}")
print(f"[decision] rank(J) = {rank:,} of {n:,};  nullity = {nullity:,}")
verdict = ("J full column rank (numerically)" if nullity == 0
           else f"J rank-deficient: nullity {nullity:,} (numerically)")
print(f"[decision] {verdict}", flush=True)

# ------------------------------------------------ pass 2: null eigenvectors, flagged rows only
flagged = np.where((all_eig <= thr).any(axis=1))[0]
null_rows, null_vals, null_vecs = [], [], []
for r0 in range(0, len(flagged), args.jchunk):
    rows = flagged[r0 : r0 + args.jchunk]
    idx = torch.as_tensor(rows, device="cuda")
    vals, vecs = torch.linalg.eigh(block_mats(idx))
    vals, vecs = vals.cpu().numpy(), vecs.cpu().numpy()
    max_diff = np.abs(vals - all_eig[rows]).max()     # consistency check against pass 1
    for k, r in enumerate(rows):
        for i in np.where(vals[k] <= thr)[0]:
            null_rows.append(int(r))
            null_vals.append(float(vals[k, i]))
            null_vecs.append(vecs[k, :, i].astype(np.float32))
    print(f"  pass 2: {len(rows)} flagged rows, max difference from pass 1 = {max_diff:.2e}", flush=True)

np.savez(outdir / "null_vectors.npz",
         rows=np.array(null_rows, dtype=np.int64),
         values=np.array(null_vals, dtype=np.float64),
         vectors=np.array(null_vecs, dtype=np.float32).reshape(-1, d))   # unit vectors in R^d

summary = {"n": n, "rank": rank, "nullity": nullity, "largest_eig_JtJ": largest,
           "smallest_positive_eig_JtJ": smallest_pos, "smallest_eig_JtJ": float(all_eig.min()),
           "zero_threshold": thr, "verdict": verdict, "null_rows": null_rows,
           "seconds": time.time() - t0, "config": vars(args)}
(outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
print(f"saved {outdir / 'summary.json'} and {outdir / 'null_vectors.npz'}")