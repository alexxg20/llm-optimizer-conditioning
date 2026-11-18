# ~/ML_project/analysis/make_github_zip.py
"""Build ~/ML_project/analysis_for_github.zip with only the relevant files, inside an analysis/ folder.

Keeps: code, scripts, README, .gitignore, env.example.sh, logs of the kept experiments, and the
results of `checkpoints`, `W_study_918` and `warm_check` (per-run .json, summary.csv, overview plots).
Drops: exploratory experiments (smoke_tests, cache_check, S_study, W_check_val) and per-run .png
files (each can be redrawn from its .json). Only files tracked by git are considered, so .npy files,
data shards, env.sh and _trash can never be included.
Run from ~/ML_project/analysis:   python make_github_zip.py
"""
import csv, io, re, subprocess, zipfile
from pathlib import Path

KEEP_EXPS = {"checkpoints", "W_study_918", "warm_check"}
DROP_LOGS = set()                                       # all five logs belong to kept experiments
OUT = Path.home() / "ML_project" / "analysis_for_github.zip"

tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split()
extra = ["README.md", ".gitignore", "env.example.sh", "make_github_zip.py"]   # take current versions
files = sorted(set(tracked) | {f for f in extra if Path(f).exists()})


def keep(f):
    parts = Path(f).parts
    if parts[0] == "results":
        if f == "results/index.csv":
            return True
        if len(parts) < 3 or parts[1] not in KEEP_EXPS:
            return False
        name = parts[2]
        if name.endswith(".png") and name.startswith("step"):   # per-run plot, redrawable from its .json
            return False
        return True
    if parts[0] == "logs":
        return Path(f).name not in DROP_LOGS
    return True


selected = [f for f in files if keep(f)]

# index.csv restricted to the kept experiments
rows = list(csv.DictReader(open("results/index.csv")))
kept_rows = [r for r in rows if r["exp"] in KEEP_EXPS]
buf = io.StringIO()
w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
w.writeheader()
w.writerows(kept_rows)

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in selected:
        if f == "results/index.csv":
            z.writestr("analysis/results/index.csv", buf.getvalue())
        else:
            z.write(f, "analysis/" + f)

print("\n".join(selected))
print(f"\n{len(selected)} files -> {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")
print(f"index.csv: kept {len(kept_rows)} of {len(rows)} rows ({', '.join(sorted(KEEP_EXPS))})")
