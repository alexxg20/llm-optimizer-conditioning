"""Upper bounds on mu of lm_head restricted to the rows of tokens seen >= m times as targets.

For the rows R kept, the principal block G_RR has two exact families of test vectors:
  one row j in R:        lambda_min(G_jj),  G_jj = (1/N) sum_t p(1-p) D^2 h h^T
  common shift of R:     Delta w_k = v for k in R  ->  v^T A_R v / v^T v,
                         A_R = (1/(N |R|)) sum_t Var_p(D * 1_R) h h^T
The bound is the smaller of the two; it uses the 16 flattest rows (by s_j) of each R.
kappa(G_RR) >= lambda_max(G_jj) / bound, with j the row of R with the largest s_j (interlacing again).
Needs spectrum.npz (s_j, target counts) from mu_structure.py.

  python mu_by_frequency.py --ckpt $W8/model_step001530.safetensors --step 1530 --shard $VAL10 --lam_max 10.15625
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
p.add_argument("--lam_max", type=float, required=True, help="lambda_max from run_power.py (for kappa)")
p.add_argument("--S", type=int, default=4096)
p.add_argument("--W", type=int, default=128)
p.add_argument("--chunk", type=int, default=1024, help="tokens per chunk")
p.add_argument("--per", type=int, default=16, help="candidate rows per cutoff")
p.add_argument("--cutoffs", type=int, nargs="+", default=[1, 10, 100, 1000])
p.add_argument("--out", default="results/mu_structure")
args = p.parse_args()
outdir = Path(args.out) / f"step{args.step:06d}"


def load_shard(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520 and header[1] == 1, "not a valid shard"
    return np.fromfile(path, dtype=np.uint16, count=int(header[2]), offset=256 * 4)


t0 = time.time()
z = np.load(outdir / "spectrum.npz")
s_np, freq = z["s"], z["freq"]
attn_blocksize = int(64 * ((args.step / 1530 * (1792 - 64) + 64) // 64))
tokens = load_shard(args.shard)
starts = np.linspace(0, len(tokens) - args.S - 1, args.W).astype(int)
windows = [torch.from_numpy(tokens[s : s + args.S].astype(np.int64)).cuda() for s in starts]
targets = np.concatenate([tokens[s + 1 : s + args.S + 1] for s in starts]).astype(np.int64)
assert (np.bincount(targets, minlength=len(freq)) == freq).all(), "spectrum.npz is from other windows"
model = load_model(args.ckpt, bf16_embeddings=True)
model.requires_grad_(False)
Wt = model.lm_head.weight.detach().float().clone()
hs = cache_hidden(model, windows, attn_blocksize)
del model, windows
torch.cuda.empty_cache()
V, d = Wt.shape
N = len(hs) * args.S
MS = args.cutoffs

sel, owner, sharp = [], [], []
for i, m in enumerate(MS):
    R = np.nonzero(freq >= m)[0]
    pick = R[np.argsort(s_np[R], kind="stable")[: args.per]]
    sel += pick.tolist()
    owner += [i] * len(pick)
    sharp.append(int(R[np.argmax(s_np[R])]))
nflat = len(sel)
sel += sharp
sel_t = torch.tensor(sel, device="cuda")
masks = torch.stack([torch.from_numpy(freq >= m) for m in MS]).cuda().float()      # (len(MS), V)
sizes = masks.sum(1).double()
Gsel = torch.zeros(len(sel), d, d, dtype=torch.float64, device="cuda")
AR = torch.zeros(len(MS), d, d, dtype=torch.float64, device="cuda")
for h_cpu in hs:
    for c0 in range(0, args.S, args.chunk):
        H = h_cpu[c0 : c0 + args.chunk].cuda()
        T = torch.tanh(H @ Wt.T / 30)
        F, D = 30 * T, 1 - T * T
        P = torch.softmax(F, dim=1)
        Hd = H.double()
        for i in range(len(MS)):
            DR = D * masks[i]
            mR = (P * DR).sum(1, keepdim=True)
            AR[i] += (Hd.T * (P * (DR - mR) ** 2).sum(1, dtype=torch.float64)) @ Hd     # Var_p(D * 1_R), two-pass
        w = (P[:, sel_t] * (1 - P[:, sel_t]) * D[:, sel_t] ** 2).double()
        Gsel += (w.T.unsqueeze(-1) * Hd.unsqueeze(0)).transpose(1, 2) @ Hd
        del T, F, D, P
Gsel /= N
AR /= N * sizes[:, None, None]
evall = torch.linalg.eigvalsh(Gsel).cpu().numpy()
ev, evtop = evall[:nflat, 0], evall[nflat:, -1]
evA = torch.linalg.eigvalsh(AR)[:, 0].cpu().numpy()

rows = []
print(f"{'tokens seen >= m':>16} {'rows':>7} {'min lmin(G_jj)':>15} {'(row, count)':>16} {'lmin(A_R)':>11} "
      f"{'mu_R <=':>10} {'lmax(G_RR) >=':>14} {'kappa_R >=':>11}")
for i, m in enumerate(MS):
    idx = [k for k, o in enumerate(owner) if o == i]
    k = idx[int(np.argmin(ev[idx]))]
    bnd = float(min(ev[k], evA[i]))
    lmx = float(evtop[i])
    rows.append({"min_count": m, "rows": int(sizes[i]), "row_bound": float(ev[k]), "flattest_row": sel[k],
                 "its_count": int(freq[sel[k]]), "shift_bound": float(evA[i]), "mu_upper_bound": bnd,
                 "sharpest_row": sharp[i], "lambda_max_lower_bound": lmx, "kappa_lower_bound": lmx / bnd,
                 "L_over_mu_R": args.lam_max / bnd})
    print(f"{m:>16} {int(sizes[i]):>7} {ev[k]:15.3e} {f'({sel[k]}, {freq[sel[k]]})':>16} {evA[i]:11.3e} "
          f"{bnd:10.3e} {lmx:14.4f} {lmx / bnd:11.2e}")
(outdir / "by_frequency.json").write_text(json.dumps({"cutoffs": rows, "seconds": time.time() - t0,
                                                      "config": vars(args)}, indent=2))
print(f"saved {outdir / 'by_frequency.json'} ({time.time() - t0:.0f} s)")
