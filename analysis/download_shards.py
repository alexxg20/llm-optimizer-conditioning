import os, sys
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "kjj0/fineweb10B-gpt2"          # the dataset the training run used
OUT = os.environ.get("D10") or Path(__file__).resolve().parent / "fineweb10B"   # D10 is set by env.sh

files = ["fineweb_val_000000.bin"]
files += [f"fineweb_train_{i:06d}.bin" for i in map(int, sys.argv[1:])]
for f in files:
    path = hf_hub_download(repo_id=REPO, filename=f, repo_type="dataset", local_dir=OUT)
    print("ok", path)