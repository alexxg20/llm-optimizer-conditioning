# ~/ML_project/analysis/mu_block.py
"""mu for one parameter block of the Gauss-Newton matrix G.

  G is full rank      ->  mu = smallest eigenvalue of G
  G is not full rank  ->  mu = smallest nonzero eigenvalue of G

Blocks:
  --block lm_head   unembedding, lm_head.weight
  --block wte       embedding, transformer.wte.weight
  --block attn      query, key and value weights of one attention layer (--layer L)

Examples:
  python mu_block.py --ckpt $W8/model_step001530.safetensors --step 1530 --shard $VAL10 \
      --block lm_head --lam_max <lambda_max of lm_head at step 1530>
  python mu_block.py --ckpt $W8/model_step001530.safetensors --step 1530 --shard $VAL10 \
      --block attn --layer 0 --lam_max <lambda_max of that block> --W 16
"""
import argparse, json, re, time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh

from load import load_model
from prox_hvp import cache_hidden, make_matvec_lmhead, make_matvec_windows

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--step", type=int, required=True)
p.add_argument("--shard", required=True)
p.add_argument("--block", required=True, choices=["lm_head", "wte", "attn"])
p.add_argument("--layer", type=int, default=0, help="attention layer, for --block attn")
p.add_argument("--S", type=int, default=4096, help="tokens per window")
p.add_argument("--W", type=int, default=128, help="number of windows")
p.add_argument("--set", type=int, default=0)
p.add_argument("--lam_max", type=float, required=True,
               help="lambda_max of this block from run_power.py (sets the shift)")
p.add_argument("--k", type=int, default=10, help="number of smallest eigenvalues to compute")
p.add_argument("--rank_tol", type=float, default=1e-10,
               help="lm_head: hidden-state eigenvalues below rank_tol * largest count as zero")
p.add_argument("--zero_tol", type=float, default=1e-6,
               help="an eigenvalue below zero_tol * lam_max counts as zero")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="results")
p.add_argument("--exp", default="mu_block")
args = p.parse_args()


def load_shard(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520 and header[1] == 1, "not a valid shard"
    return np.fromfile(path, dtype=np.uint16, count=int(header[2]), offset=256 * 4)


#  1. windows and the parameter block
attn_blocksize = int(64 * ((args.step / 1530 * (1792 - 64) + 64) // 64))
tokens = load_shard(args.shard)
starts = np.linspace(0, len(tokens) - args.S - 1 - args.set * args.S, args.W).astype(int) + args.set * args.S
assert np.all(np.diff(starts) >= args.S), "windows overlap"
windows = [torch.from_numpy(tokens[s : s + args.S].astype(np.int64)).cuda() for s in starts]

patterns = {
    "lm_head": r"^lm_head\.weight$",
    "wte": r"^transformer\.wte\.weight$",
    "attn": rf"^transformer\.h\.{args.layer}\.attn\.c_[qkv]\.weight$",
}
model = load_model(args.ckpt, bf16_embeddings=(args.block != "wte"))
model.requires_grad_(False)
theta_ref = {k: q.detach().float() for k, q in model.named_parameters()
             if re.search(patterns[args.block], k)}
assert theta_ref, "no parameter matched"
names = list(theta_ref)
shapes = {k: tuple(theta_ref[k].shape) for k in names}
offsets, acc = {}, 0
for k in names:
    offsets[k] = acc                       # start of this tensor in the flat vector
    acc += theta_ref[k].numel()
n = acc
print(f"block {args.block}: {len(names)} tensor(s), n = {n:,} parameters")
for k in names:
    print("   ", k, shapes[k])

# 2. the operator G (never formed)
## For lm_head it uses the cached hidden states, which is fast. For wte and attn it runs the full model for each window, which is slow.

if args.block == "lm_head":
    W_ref = theta_ref["lm_head.weight"]
    hs = cache_hidden(model, windows, attn_blocksize)
    matvec_raw, _ = make_matvec_lmhead(W_ref, hs)                  # fast
else:
    matvec_raw, _ = make_matvec_windows(model, theta_ref, windows, attn_blocksize)   # slow


def G(x):
    """Gauss-Newton matrix (averaged over windows) times a flat vector of length n."""
    return matvec_raw(np.asarray(x, dtype=np.float32).reshape(-1)).astype(np.float64)


# 3. exact null directions: projector Pn
rng = np.random.default_rng(args.seed)
probe = None                # one exact null vector, used to measure the noise floor
lam = args.lam_max

if args.block == "lm_head":
    V, d = W_ref.shape
    M = np.zeros((d, d), dtype=np.float64)
    for h in hs:
        hd = h.double().numpy()
        M += hd.T @ hd                                             # sum over tokens of h h^T
    evals, evecs = np.linalg.eigh(M)
    rel = evals / evals[-1]
    N = evecs[:, rel <= args.rank_tol]                             # d x m: unused hidden directions
    nullity = V * N.shape[1]
    print(f"[structure] hidden-state rank = {d - N.shape[1]} of {d}; exact nullity = {nullity:,}")

    def Pn(x):
        X = x.reshape(V, d)
        return (X @ N @ N.T).reshape(-1)

    if N.shape[1] > 0:
        probe = np.outer(rng.standard_normal(V), N[:, 0]).ravel()

elif args.block == "wte":
    V, d = shapes[names[0]]
    used = np.unique(np.concatenate([w.cpu().numpy() for w in windows]))
    unused = np.ones(V, dtype=bool)
    unused[used] = False
    nullity = int(unused.sum()) * d
    print(f"[structure] tokens used: {len(used):,} of {V:,}; unused: {int(unused.sum()):,}; "
          f"exact nullity = {nullity:,}")

    def Pn(x):
        X = x.reshape(V, d)
        return np.where(unused[:, None], X, 0.0).reshape(-1)       # keep unused rows only

    if nullity > 0:
        i = int(np.argmax(unused))
        probe = np.zeros(n)
        probe[i * d : (i + 1) * d] = rng.standard_normal(d)        # row of an unused token

else:  # attn: scaling one head's W_q or W_k leaves the QK-normalised output unchanged
    n_head = model.transformer.h[args.layer].attn.n_head
    heads = []                                                     # (start, end, unit vector)
    for k in names:
        if "c_q.weight" in k or "c_k.weight" in k:
            out_dim, in_dim = shapes[k]
            hd_ = out_dim // n_head
            W_t = theta_ref[k].cpu().numpy().astype(np.float64)
            for h in range(n_head):
                vec = W_t[h * hd_ : (h + 1) * hd_].ravel()
                s = offsets[k] + h * hd_ * in_dim
                heads.append((s, s + hd_ * in_dim, vec / np.linalg.norm(vec)))
    nullity = len(heads)
    print(f"[structure] exact nullity = {nullity} (one scale direction per head of W_q and W_k)")

    def Pn(x):
        y = np.zeros_like(x)
        for s, e, u in heads:
            y[s:e] += (x[s:e] @ u) * u                             # supports are disjoint
        return y

    if nullity > 0:
        s, e, u = heads[0]
        probe = np.zeros(n)
        probe[s:e] = u

# noise floor: G on an exact null direction should be zero, so this is rounding error
if probe is not None:
    floor = float(np.linalg.norm(G(probe)) / (lam * np.linalg.norm(probe)))
    print(f"[floor] relative size of G applied to an exact null direction: {floor:.3e}")
else:
    floor = None
    print("[floor] no exact null direction for this block")

# 4. operator on the complement of the null space
def Pc(x):
    return x - Pn(x)


c = 2.0 * lam          # value placed on the null space, so it sits at the bottom of the operator
count = {"products": 0}
t0 = time.time()


def Bop(x):
    """c*I - G restricted to the complement of the null space; zero on the null space."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    count["products"] += 1
    if count["products"] % 10 == 0:
        print(f"   product {count['products']}, {time.time() - t0:.0f} s", flush=True)
    y = Pc(x)
    if not np.any(y):
        return np.zeros(n)
    return c * y - Pc(G(y))


# 5. smallest eigenvalues of G on the complement
B = LinearOperator((n, n), matvec=Bop, dtype=np.float64)
vals, _ = eigsh(B, k=args.k, which="LA", tol=1e-6)               # largest eigenvalues of Bop (Lanzcos)
g = np.sort(c - vals)                                             # smallest eigenvalues of G
print(f"\n[7d] {count['products']} products in {time.time() - t0:.0f} s")
print("[7d] smallest eigenvalues of G on the complement:")
for m in g:
    print(f"      {m:.6e}   (relative to lam: {m / lam:.3e})")

# 6. full rank, and mu
thr = args.zero_tol * lam
n_zero = int(np.sum(g < thr))
full_rank = (nullity == 0) and (g[0] >= thr)
if full_rank:
    mu = float(g[0])
    mu_def = "smallest eigenvalue (G is full rank)"
else:
    nz = g[g >= thr]
    mu = float(nz[0]) if len(nz) else None
    mu_def = "smallest nonzero eigenvalue (G is not full rank)"

print(f"\n[rank] exact nullity = {nullity:,}")
print(f"[rank] zero threshold = {thr:.3e}; computed eigenvalues below it: {n_zero} of {args.k}")
print(f"[rank] full rank: {full_rank}")
if mu is None:
    print("[mu] no nonzero eigenvalue among the k computed: rerun with a larger --k")
else:
    print(f"[mu] {mu_def} = {mu:.6e}  (relative {mu / lam:.3e}); lam / mu = {lam / mu:.3e}")

# 7. save
out = Path(args.out) / args.exp
out.mkdir(parents=True, exist_ok=True)
stamp = time.strftime("%Y%m%d-%H%M%S")
suffix = f"_L{args.layer}" if args.block == "attn" else ""
tag = f"{args.block}{suffix}_step{args.step:06d}_W{args.W}_{stamp}"
results = {"config": vars(args) | {"n": n}, "tensors": {k: list(shapes[k]) for k in names},
           "exact_nullity": nullity, "noise_floor": floor, "lam_max": lam, "shift_c": c,
           "smallest_eigs": g.tolist(), "relative": (g / lam).tolist(),
           "zero_threshold": thr, "full_rank": full_rank, "mu": mu, "mu_definition": mu_def,
           "products": count["products"]}
(out / f"{tag}.json").write_text(json.dumps(results, indent=2, default=float))
print(f"\nsaved {out / (tag + '.json')}")