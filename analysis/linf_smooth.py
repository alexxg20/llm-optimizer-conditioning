# ~/ML_project/analysis/linf_smooth.py
"""L_inf smoothness of one block: sup of x^T G x over sign vectors x in {-1,+1}^n.

  --mode small  validate the heuristic on a random subset of n_small coordinates,
                where the exact maximum is found by checking all 2^n sign vectors (testing) 
  --mode full   run the heuristic on the whole block (a lower bound on the sup).

Method: sign power iteration with random restarts. For PSD G each step does not decrease
x^T G x, so each restart ends at a local maximum.
"""
import argparse, json, re, time
from pathlib import Path

import numpy as np
import torch

from load import load_model
from prox_hvp import cache_hidden, make_matvec_lmhead, make_matvec_windows

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--step", type=int, required=True)
p.add_argument("--shard", required=True)
p.add_argument("--block", required=True, choices=["lm_head", "wte", "attn"])
p.add_argument("--layer", type=int, default=0, help="attention layer, for --block attn")
p.add_argument("--mode", required=True, choices=["small", "full"])
p.add_argument("--n_small", type=int, default=20, help="coordinates in the small test (at most 22)")
p.add_argument("--restarts", type=int, default=10)
p.add_argument("--iters", type=int, default=50, help="maximum sign-power steps per restart")
p.add_argument("--lam_max", type=float, default=None, help="lambda_max of the block, for the bound n*lam_max")
p.add_argument("--S", type=int, default=4096)
p.add_argument("--W", type=int, default=128)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="results")
p.add_argument("--exp", default="linf")
args = p.parse_args()


def load_shard(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520 and header[1] == 1, "not a valid shard"
    return np.fromfile(path, dtype=np.uint16, count=int(header[2]), offset=256 * 4)


# 1. windows, model and block 
attn_blocksize = int(64 * ((args.step / 1530 * (1792 - 64) + 64) // 64))
tokens = load_shard(args.shard)
starts = np.linspace(0, len(tokens) - args.S - 1, args.W).astype(int)
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
n = sum(theta_ref[k].numel() for k in names)
print(f"block {args.block}: {len(names)} tensor(s), n = {n:,} parameters")

# 2. the operator G
if args.block == "lm_head":
    W_ref = theta_ref["lm_head.weight"]
    hs = cache_hidden(model, windows, attn_blocksize) # we should do this for others too...  
    matvec_raw, _ = make_matvec_lmhead(W_ref, hs)
else:
    matvec_raw, _ = make_matvec_windows(model, theta_ref, windows, attn_blocksize)


def G(x):
    return matvec_raw(np.asarray(x, dtype=np.float32).reshape(-1)).astype(np.float64)


rng = np.random.default_rng(args.seed)


# 3. the heuristic
def sign_power(Gop, n, restarts, iters, rng):
    """Maximise x^T G x over sign vectors by sign power iteration with random restarts.

    Step: x_new = sign(G x). For PSD G the value never decreases, so each restart
    ends at a local maximum. Returns the best value, its sign vector and all restart values.
    """
    best_val, best_x = -np.inf, None
    values = []
    for r in range(restarts):
        x = rng.choice(np.array([-1.0, 1.0]), size=n)
        y = Gop(x)                              # y = G x
        val = float(x @ y)
        steps = 0
        for steps in range(1, iters + 1):
            x_new = np.where(y >= 0, 1.0, -1.0)  # sign vector that best matches G x
            if np.array_equal(x_new, x):
                steps -= 1
                break                            # fixed point: a local maximum
            y_new = Gop(x_new)
            val_new = float(x_new @ y_new)
            if val_new < val - 1e-9 * max(1.0, abs(val)):
                print("   warning: value decreased; numerical precision?", flush=True)
            x, y, val = x_new, y_new, val_new
        values.append(val)
        print(f"  restart {r + 1}/{restarts}: value {val:.6e}, steps {steps}", flush=True)
        if val > best_val:
            best_val, best_x = val, x.copy()
    return best_val, best_x, values


def exact_max(Gs): # for validation of small hessians
    """Brute force over all 2^n sign vectors (feasible for n up to about 22)."""
    m = Gs.shape[0]
    best, best_x = -np.inf, None
    chunk = 1 << 16
    for start in range(0, 1 << m, chunk):
        idx = np.arange(start, min(start + chunk, 1 << m), dtype=np.int64)
        bits = (idx[:, None] >> np.arange(m)) & 1
        X = 2.0 * bits - 1.0                             # rows are sign vectors
        vals = np.einsum("ij,ij->i", X @ Gs, X)          # x^T G x for every row
        k = int(np.argmax(vals))
        if vals[k] > best:
            best, best_x = float(vals[k]), X[k].copy()
    return best, best_x


# 4. run
out = Path(args.out) / args.exp
out.mkdir(parents=True, exist_ok=True)
stamp = time.strftime("%Y%m%d-%H%M%S")
suffix = f"_L{args.layer}" if args.block == "attn" else ""
result = {"config": vars(args) | {"n": n}}

if args.mode == "small":
    assert args.n_small <= 22, "exact enumeration needs n_small <= 22"
    S_idx = np.sort(rng.choice(n, size=args.n_small, replace=False))
    
    # Build the small block G_SS one column at a time: column j is G e_j, keep the rows in S.
    print(f"\n[small] building the {args.n_small} x {args.n_small} block: {args.n_small} products")
    Gs = np.zeros((args.n_small, args.n_small))
    e = np.zeros(n, dtype=np.float32)
    for c, j in enumerate(S_idx):
        e[j] = 1.0
        Gs[:, c] = G(e)[S_idx]
        e[j] = 0.0
    Gs = 0.5 * (Gs + Gs.T)                                       # remove rounding asymmetry

    exact_val, exact_x = exact_max(Gs)
    print(f"[small] exact maximum over all 2^{args.n_small} sign vectors: {exact_val:.6e}")
    heur_val, heur_x, values = sign_power(lambda x: Gs @ x, args.n_small,
                                          args.restarts, args.iters, rng)
    hits = sum(abs(v - exact_val) <= 1e-7 * abs(exact_val) for v in values)
    print(f"[small] heuristic best: {heur_val:.6e}  (ratio to exact {heur_val / exact_val:.6f})")
    print(f"[small] restarts that reached the exact maximum: {hits} of {args.restarts}")
    result.update({"exact_max": exact_val, "heuristic_best": heur_val, "restart_values": values,
                   "restarts_hitting_exact": hits, "subset": S_idx.tolist()})
    tag = f"small_{args.block}{suffix}_n{args.n_small}_seed{args.seed}_{stamp}"
else:
    print(f"\n[full] sign power iteration on all {n:,} coordinates")
    t0 = time.time()
    best_val, best_x, values = sign_power(G, n, args.restarts, args.iters, rng)
    print(f"[full] best value (lower bound on the sup): {best_val:.6e}  ({time.time() - t0:.0f} s)")
    result.update({"best_value": best_val, "restart_values": values})
    if args.lam_max is not None:
        bound = n * args.lam_max
        print(f"[full] upper bound n * lam_max = {bound:.6e};  best / bound = {best_val / bound:.4f}")
        result["upper_bound_n_lam_max"] = bound
    np.save(out / f"signs_{args.block}{suffix}_{stamp}.npy", best_x.astype(np.int8))
    tag = f"full_{args.block}{suffix}_{stamp}"

(out / f"{tag}.json").write_text(json.dumps(result, indent=2, default=float))
print(f"\nsaved {out / (tag + '.json')}")