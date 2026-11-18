# ~/ML_project/analysis/run_power.py
"""Largest eigenvalue of the Gauss-Newton matrix (Hessian of the prox loss) at one checkpoint.

Example:
    python run_power.py --ckpt $W8/model_step000918.safetensors --step 918 --shard $VAL10 \
        --S 4096 --W 128 --params lm_head --exp checkpoints [--init prev_eigvec.npy]
"""
import argparse, json, re, csv, time, subprocess, platform
from pathlib import Path
import numpy as np
import torch
from load import load_model
from prox_hvp import make_matvec_windows, cache_hidden, make_matvec_lmhead
from power_method import power_iteration, plot_history

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--step", type=int, required=True)             # sets the attention window
p.add_argument("--shard", required=True)                      # a .bin file with the tokens
p.add_argument("--S", type=int, default=4096)                 # window length
p.add_argument("--W", type=int, default=128)                  # number of windows
p.add_argument("--set", type=int, default=0, help="shift window starts by set*S tokens (disjoint window sets)")
p.add_argument("--params", default="lm_head",                 # regex: which parameters to analyse
               help="e.g. 'lm_head' or 'transformer\\.h\\..*\\.weight$'")
p.add_argument("--iters", type=int, default=300)
p.add_argument("--tol", type=float, default=1e-3)             # stop when the relative residual < tol
p.add_argument("--min_iters", type=int, default=15)           # never stop before this many iterations
p.add_argument("--seed", type=int, default=0)
p.add_argument("--init", default=None, help=".npy file with a starting vector (warm start)")
p.add_argument("--out", default="results")
p.add_argument("--exp", default="misc", help="experiment name, e.g. S_study, W_study, checkpoints")
args = p.parse_args()


def load_shard(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520 and header[1] == 1, "not a valid shard"
    return np.fromfile(path, dtype=np.uint16, count=int(header[2]), offset=256 * 4)


# attention window for this step, as in the training script
attn_blocksize = int(64 * ((args.step / 1530 * (1792 - 64) + 64) // 64))

# token windows: W windows of S tokens, evenly spread, shifted by set*S
tokens = load_shard(args.shard)
starts = np.linspace(0, len(tokens) - args.S - 1 - args.set * args.S, args.W).astype(int) + args.set * args.S
assert np.all(np.diff(starts) >= args.S), "windows overlap: use fewer windows or a smaller --set"
windows = [torch.from_numpy(tokens[s : s + args.S].astype(np.int64)).cuda() for s in starts]

# which parameters to analyse; all others stay fixed at the checkpoint
pattern = re.compile(args.params)
embeddings_analysed = any(pattern.search(k) for k in ("transformer.wte.weight", "transformer.vte.weight"))
model = load_model(args.ckpt, bf16_embeddings=not embeddings_analysed)   # fp32 if we analyse them
model.requires_grad_(False)                                              # frozen weights: no autograd bookkeeping

theta_ref = {k: p.detach() for k, p in model.named_parameters() if pattern.search(k)}
assert theta_ref, f"no parameters match {args.params!r}"
print(f"analysing {len(theta_ref)} tensors, {sum(p.numel() for p in theta_ref.values())/1e6:.1f}M parameters:")
for k in theta_ref:
    print("   ", k, tuple(theta_ref[k].shape))

if args.params == "lm_head":
    print("using cached hidden states for lm_head")
    hs = cache_hidden(model, windows, attn_blocksize)                    # blocks run once per window
    matvec_raw, n = make_matvec_lmhead(theta_ref["lm_head.weight"], hs)
else:
    matvec_raw, n = make_matvec_windows(model, theta_ref, windows, attn_blocksize)


def matvec(v):
    out = matvec_raw(v)
    if matvec.first:
        matvec.first = False
        print(f"peak GPU memory after first product: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    return out
matvec.first = True

v0 = None
if args.init:
    v0 = np.load(args.init)
    assert v0.shape == (n,), f"--init vector has length {v0.shape[0]}, expected {n} (different --params?)"
    print("warm start from", args.init)

hist, v_final = power_iteration(matvec, n, iters=args.iters, tol=args.tol, seed=args.seed,
                                v0=v0, min_iters=args.min_iters)

# ------------------------------------------------------------------ save results
stamp = time.strftime("%Y%m%d-%H%M%S")
out = Path(args.out) / args.exp
out.mkdir(parents=True, exist_ok=True)
params_tag = re.sub(r"[^A-Za-z0-9]+", "-", args.params).strip("-")
tag = f"step{args.step:06d}_{params_tag}_S{args.S}_W{args.W}_seed{args.seed}_set{args.set}_{stamp}"

try:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or None
except Exception:
    commit = None

config = vars(args) | {
    "attn_blocksize": attn_blocksize,
    "n_params": n,
    "lambda_max": hist["lambda"][-1],
    "two_over_lambda": 2 / hist["lambda"][-1],
    "iterations": hist["iter"][-1],
    "final_resid": hist["rel_resid"][-1],
    "converged": hist["rel_resid"][-1] < args.tol,
    "seconds_total": hist["seconds"][-1],
    "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
    "window_starts": starts.tolist(),
    "timestamp": stamp,
    "git_commit": commit,
    "torch": torch.__version__,
    "host": platform.node(),
}

(out / f"{tag}.json").write_text(json.dumps({"config": config, "history": hist}, indent=2))
np.save(out / f"{tag}_eigvec.npy", v_final)
plot_history(hist, out / f"{tag}.png", title=f"power method, {tag}")

# one summary row per run in results/index.csv
index = Path(args.out) / "index.csv"
row = {k: config[k] for k in ["timestamp", "exp", "step", "params", "S", "W", "seed", "set", "init",
                              "lambda_max", "two_over_lambda", "iterations", "final_resid",
                              "converged", "seconds_total", "peak_gpu_gib", "git_commit"]}
row["json"] = str(out / f"{tag}.json")
new = not index.exists()
with index.open("a", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(row))
    if new:
        w.writeheader()
    w.writerow(row)

print("saved", out / f"{tag}.json", out / f"{tag}.png", "and", out / f"{tag}_eigvec.npy")