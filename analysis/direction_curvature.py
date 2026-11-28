"""One step of GD, sign descent and Adam on the local quadratic model of the lm_head block.

For a direction d, the quadratic model predicts  L(W - eta d) ~ L - eta g^T d + eta^2 / 2 d^T G d,
so the best step is eta* = g^T d / d^T G d and it lowers the loss by (g^T d)^2 / (2 d^T G d).
Directions, all from the same gradient g (mean loss over the windows, as in training):
  gd           d = g
  sign         d = sign(g)
  adam         d = m'_hat / (sqrt(v'_hat) + eps)   Adam's next update if its next gradient were g:
                                                   m' = b1 m + (1-b1) g, v' = b2 v + (1-b2) g^2, step t+1,
                                                   with m, v from the full checkpoint
  adam_no_mom  d = g / (sqrt(v'_hat) + eps)        the same preconditioner without the momentum
v' must include g: dividing by the stored v alone blows up coordinates whose stored v is ~0.
Each predicted decrease is compared with the actual loss change on the same tokens (lm_head only,
other parameters fixed), which also measures how good the Gauss-Newton quadratic is here.

  python direction_curvature.py --ckpt $W8/model_step000918.safetensors \
      --full full/ckpt_step000918.pt --step 918 --shard $VAL10 --lam_max 11.70624 --linf 4963.94
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from load import load_model
from prox_hvp import cache_hidden, make_matvec_lmhead

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True, help="weights (.safetensors) at this step")
p.add_argument("--full", required=True, help="full checkpoint (.pt) at the same step, with the optimizer states")
p.add_argument("--step", type=int, required=True)
p.add_argument("--shard", required=True)
p.add_argument("--S", type=int, default=4096)
p.add_argument("--W", type=int, default=128)
p.add_argument("--chunk", type=int, default=1024, help="tokens per chunk")
p.add_argument("--lam_max", type=float, required=True, help="lambda_max of lm_head at this step (run_power.py)")
p.add_argument("--linf", type=float, default=None, help="L_inf lower bound at this step (linf_smooth.py)")
p.add_argument("--split", action="store_true",
               help="also split each best step over rows grouped by how often their token is a target")
p.add_argument("--out", default="results/direction_curvature")
args = p.parse_args()


def load_shard(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520 and header[1] == 1, "not a valid shard"
    return np.fromfile(path, dtype=np.uint16, count=int(header[2]), offset=256 * 4)


# 1. windows (same as run_power.py --set 0), hidden states, lm_head weights
t0 = time.time()
attn_blocksize = int(64 * ((args.step / 1530 * (1792 - 64) + 64) // 64))
tokens = load_shard(args.shard)
starts = np.linspace(0, len(tokens) - args.S - 1, args.W).astype(int)
windows = [torch.from_numpy(tokens[s : s + args.S].astype(np.int64)).cuda() for s in starts]
targets = [torch.from_numpy(tokens[s + 1 : s + args.S + 1].astype(np.int64)).cuda() for s in starts]

model = load_model(args.ckpt, bf16_embeddings=True)
model.requires_grad_(False)
W = model.lm_head.weight.detach().float().clone()
hs = cache_hidden(model, windows, attn_blocksize)
del model, windows
torch.cuda.empty_cache()
V, d = W.shape
n, N, CH = V * d, args.W * args.S, args.chunk

# 2. Adam's state for lm_head: optimizers[1] in build_optimizers() (pretraining/train_feasibility.py)
full = torch.load(args.full, map_location="cpu", mmap=True, weights_only=True)
assert full["step"] == args.step, f"full checkpoint is at step {full['step']}, not {args.step}"
W_full = full["model"]["lm_head.weight"].float().cuda()
w_diff = float((W_full - W).abs().max())
assert w_diff == 0.0, f"lm_head differs between --ckpt and --full (max |diff| = {w_diff:.3e})"
opt = full["optimizers"][1]
(state,) = opt["state"].values()
group = opt["param_groups"][0]
beta1, beta2 = group["betas"]
eps, lr, t = group["eps"], group["lr"], float(state["step"])
m = state["exp_avg"].float().cuda()
v = state["exp_avg_sq"].float().cuda()
assert m.shape == W.shape and v.shape == W.shape
del full, W_full
print(f"setup {time.time() - t0:.0f} s; Adam state at t = {t:.0f}, lr = {lr:.3e}, eps = {eps:g}", flush=True)


def chunk_loss_grad(H, y, Wt, want_grad):
    """Cross-entropy of the soft-capped logits at each position of one chunk, and the chunk's summed gradient in W."""
    T = torch.tanh(H @ Wt.T / 30)
    Fz = 30 * T
    lse = torch.logsumexp(Fz, dim=1)
    lt = (lse - Fz.gather(1, y[:, None]).squeeze(1)).double()
    if not want_grad:
        return lt, None
    R = torch.exp(Fz - lse[:, None])
    R[torch.arange(len(y), device=y.device), y] -= 1                # softmax minus one-hot
    return lt, (R * (1 - T * T)).T @ H


def loss_and_grad(Wt, want_grad=False):
    loss = torch.zeros((), dtype=torch.float64, device="cuda")
    g = torch.zeros_like(Wt) if want_grad else None
    for h_cpu, y in zip(hs, targets):
        for c0 in range(0, args.S, CH):
            lt, gc = chunk_loss_grad(h_cpu[c0 : c0 + CH].cuda(), y[c0 : c0 + CH], Wt, want_grad)
            loss += lt.sum()
            if want_grad:
                g += gc
    return float(loss) / N, (g / N if want_grad else None)


def loss_by_target_group(Wt, group_of_token, n_groups):
    """Each group's part of the mean loss: the loss summed over positions whose target is in the group, over N."""
    acc = torch.zeros(n_groups, dtype=torch.float64, device="cuda")
    for h_cpu, y in zip(hs, targets):
        for c0 in range(0, args.S, CH):
            yc = y[c0 : c0 + CH]
            lt, _ = chunk_loss_grad(h_cpu[c0 : c0 + CH].cuda(), yc, Wt, False)
            acc.scatter_add_(0, group_of_token[yc], lt)
    return (acc / N).cpu().numpy()


# 3. the gradient, checked against autograd on one chunk
H0, y0 = hs[0][:CH].cuda(), targets[0][:CH]
Wa = W.clone().requires_grad_(True)
F.cross_entropy(30 * torch.tanh(H0 @ Wa.T / 30), y0, reduction="sum").backward()
_, g0 = chunk_loss_grad(H0, y0, W, True)
grad_check = float((g0 - Wa.grad).norm() / Wa.grad.norm())
print(f"[check] gradient vs autograd on one chunk, relative difference: {grad_check:.2e}", flush=True)
assert grad_check < 1e-4
del Wa, g0

L0, g = loss_and_grad(W, want_grad=True)
g64 = g.double()
g1, g2sq = float(g64.abs().sum()), float((g64 * g64).sum())
phi = g1 ** 2 / (n * g2sq)
print(f"loss at the checkpoint: {L0:.6f};  ||g||_2 = {g2sq ** 0.5:.4e}, ||g||_1 = {g1:.4e}")
print(f"gradient density phi = ||g||_1^2 / (n ||g||_2^2) = {phi:.4e}", flush=True)
v_hat = v / (1 - beta2 ** t)
print(f"Adam: ||sqrt(v_hat)||_2 = {float(v_hat.double().sum()) ** 0.5:.4e} (RMS of training gradients); "
      f"coordinates with sqrt(v_hat) < eps: {float((v_hat.sqrt() < eps).double().mean()):.2%}", flush=True)
del v_hat
t1 = t + 1
m1_hat = (beta1 * m + (1 - beta1) * g) / (1 - beta1 ** t1)
denom = ((beta2 * v + (1 - beta2) * g * g) / (1 - beta2 ** t1)).sqrt() + eps
del m, v

# 4. curvature along each direction, predicted and actual decrease
matvec, _ = make_matvec_lmhead(W, hs)


def apply_G(dirn):
    return torch.from_numpy(matvec(dirn.reshape(-1).cpu().numpy())).cuda().view_as(dirn)


directions = {"gd": g, "sign": torch.sign(g), "adam": m1_hat / denom, "adam_no_mom": g / denom}
rows = []

if args.split:
    # a token's group serves both for its row of W and for the positions where it is the target
    counts = torch.bincount(torch.cat(targets), minlength=V)
    bins = [("count 0", 0, 0), ("count 1-9", 1, 9), ("count 10-99", 10, 99), ("count >=100", 100, None)]
    token_group = torch.full((V,), -1, dtype=torch.long, device="cuda")
    for k, (_, lo, hi) in enumerate(bins):
        token_group[(counts >= lo) if hi is None else (counts >= lo) & (counts <= hi)] = k
    assert bool((token_group >= 0).all())
    n_rows = [int((token_group == k).sum()) for k in range(len(bins))]
    target_share = [float(counts[token_group == k].sum()) / N for k in range(len(bins))]
    base_by_target = loss_by_target_group(W, token_group, len(bins))
    split_rows = []


def split_step(name, dirn, Gd, eta, total_pred, total_actual):
    """Splits one step's decrease over row groups (exactly, in the quadratic model) and over target groups."""
    lin = (g64 * dirn.double()).sum(1)                           # g_j . d_j for each row j
    quad = (dirn.double() * Gd.double()).sum(1)                  # d_j . (G d)_j: sums to d^T G d
    after_by_target = loss_by_target_group(W - eta * dirn, token_group, len(bins))
    parts = []
    for k, (gname, _, _) in enumerate(bins):
        in_k = token_group == k
        parts.append({"direction": name, "group": gname, "rows": n_rows[k], "target_share": target_share[k],
                      "predicted": float(eta * lin[in_k].sum() - 0.5 * eta ** 2 * quad[in_k].sum()),
                      "actual_moving_only_these_rows": L0 - loss_and_grad(W - eta * dirn * in_k[:, None])[0],
                      "actual_at_these_targets": float(base_by_target[k] - after_by_target[k])})
    sum_pred = sum(r["predicted"] for r in parts)
    sum_targets = sum(r["actual_at_these_targets"] for r in parts)
    assert abs(sum_pred - total_pred) <= 1e-6 * abs(total_pred), (sum_pred, total_pred)
    assert abs(sum_targets - total_actual) <= 1e-6 * abs(total_actual) + 1e-12, (sum_targets, total_actual)
    split_rows.extend(parts)


def add_row(name, dirn, gTd, dGd, eta, step_rule):
    pred = eta * gTd - 0.5 * eta ** 2 * dGd
    actual = L0 - loss_and_grad(W - eta * dirn)[0]
    rows.append({"direction": name, "step": step_rule, "eta": eta, "gTd": gTd, "dGd": dGd,
                 "curvature_per_unit_length": dGd / float((dirn.double() ** 2).sum()),
                 "predicted_decrease": pred, "actual_decrease": actual})
    print(f"  {name:11s} {step_rule:12s} eta = {eta:.3e}  predicted {pred:.4e}  actual {actual:.4e}", flush=True)


for name, dirn in directions.items():
    gTd = float((g64 * dirn.double()).sum())
    Gd = apply_G(dirn)
    dGd = float((dirn.double() * Gd.double()).sum())
    add_row(name, dirn, gTd, dGd, gTd / dGd, "best")
    if args.split:
        split_step(name, dirn, Gd, gTd / dGd, rows[-1]["predicted_decrease"], rows[-1]["actual_decrease"])
    del Gd
    if name == "gd":
        add_row(name, dirn, gTd, dGd, 1 / args.lam_max, "1/lam_max")
    if name.startswith("adam"):
        add_row(name, dirn, gTd, dGd, lr, "adam lr")

# 5. summary
print(f"\n{'direction':11s} {'step':12s} {'eta':>10s} {'curv/|d|^2':>11s} {'predicted':>11s} {'actual':>11s}")
for r in rows:
    print(f"{r['direction']:11s} {r['step']:12s} {r['eta']:10.3e} {r['curvature_per_unit_length']:11.3e} "
          f"{r['predicted_decrease']:11.4e} {r['actual_decrease']:11.4e}")
best = {r["direction"]: r["predicted_decrease"] for r in rows if r["step"] == "best"}
print(f"\nbest-step decrease relative to GD: " + ", ".join(f"{k} {v / best['gd']:.3g}x" for k, v in best.items()))
rho = None
if args.linf is not None:
    rho = args.linf / (n * args.lam_max)
    print(f"rho = L_inf / (n lam_max) >= {rho:.3e};  worst-case sign/GD ratio phi / rho <= {phi / rho:.3g}")
if args.split:
    print(f"\nbest step split by how often a token is a target in the {N:,} positions")
    print(f"{'direction':11s} {'group':12s} {'rows':>6s} {'targets':>8s} {'predicted':>11s} "
          f"{'only rows':>11s} {'at targets':>11s}")
    for r in split_rows:
        print(f"{r['direction']:11s} {r['group']:12s} {r['rows']:6d} {r['target_share']:8.2%} {r['predicted']:11.4e} "
              f"{r['actual_moving_only_these_rows']:11.4e} {r['actual_at_these_targets']:11.4e}")

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
tag = f"step{args.step:06d}_{time.strftime('%Y%m%d-%H%M%S')}"
result = {"config": vars(args) | {"n": n, "N": N}, "loss": L0, "grad_norm_2": g2sq ** 0.5, "grad_norm_1": g1,
          "phi": phi, "rho_lower_bound": rho, "grad_check": grad_check,
          "adam": {"t": t, "lr": lr, "eps": eps, "betas": [beta1, beta2]}, "rows": rows,
          "split": split_rows if args.split else None, "seconds": time.time() - t0}
(out / f"{tag}.json").write_text(json.dumps(result, indent=2, default=float))
print(f"\nsaved {out / (tag + '.json')}  ({time.time() - t0:.0f} s)")
