# ~/ML_project/analysis/compare.py
"""Compare power-method runs recorded in results/index.csv.

Usage:
    python compare.py --exp S_study --x S
    python compare.py --exp W_study_918 --x W
    python compare.py --exp checkpoints --x step
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # no display needed on the cluster
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--results", default="results", help="folder that contains index.csv")
p.add_argument("--exp", required=True, help="experiment name to compare (the --exp used in run_power.py)")
p.add_argument("--x", default="step", help="column for the x axis: step, S or W")
args = p.parse_args()

df = pd.read_csv(Path(args.results) / "index.csv")
df = df[df["exp"] == args.exp]
if df.empty:
    raise SystemExit(f"no runs with exp == {args.exp!r} in {args.results}/index.csv")

# keep the latest run for each setting
keys = [args.x, "params", "seed"] + (["set"] if "set" in df.columns else [])
df = df.sort_values(keys + ["timestamp"]).groupby(keys).tail(1)
series = [k for k in keys if k != args.x]          # one line per combination of these

out = Path(args.results) / args.exp
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "summary.csv", index=False)

cols = keys + ["lambda_max", "two_over_lambda", "iterations", "final_resid",
               "converged", "seconds_total", "peak_gpu_gib"]
print(df[[c for c in cols if c in df.columns]].to_string(index=False))

# 1) lambda_max and 2/lambda_max against the chosen variable
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for name, g in df.groupby(series):
    name = name if isinstance(name, tuple) else (name,)
    lab = ", ".join(f"{k}={v}" for k, v in zip(series, name))
    ax[0].plot(g[args.x], g["lambda_max"], marker="o", label=lab)
    ax[1].plot(g[args.x], g["two_over_lambda"], marker="o", label=lab)
ax[0].set_ylabel("lambda_max")
ax[1].set_ylabel("2 / lambda_max (GD stability limit)")
for a in ax:
    a.set_xlabel(args.x)
    a.grid(alpha=0.3)
    a.legend(fontsize=7)
fig.suptitle(f"{args.exp}: lambda_max vs {args.x}")
fig.tight_layout()
fig.savefig(out / f"lambda_vs_{args.x}.png", dpi=130)
plt.close(fig)

# 2) convergence curves of all runs overlaid
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for _, r in df.iterrows():
    h = json.loads(Path(r["json"]).read_text())["history"]
    lab = f"{args.x}={r[args.x]}"
    ax[0].plot(h["iter"], h["lambda"], label=lab)
    ax[1].semilogy(h["iter"], h["rel_resid"], label=lab)
ax[0].set_title("lambda estimate (Rayleigh quotient)")
ax[1].set_title("relative residual ||Gv - lambda v|| / |lambda|")
for a in ax:
    a.set_xlabel("power iteration")
    a.grid(alpha=0.3)
    a.legend(fontsize=7)
fig.suptitle(f"{args.exp}: convergence")
fig.tight_layout()
fig.savefig(out / "convergence_overlay.png", dpi=130)
plt.close(fig)

print("saved", out / "summary.csv", out / f"lambda_vs_{args.x}.png", out / "convergence_overlay.png")