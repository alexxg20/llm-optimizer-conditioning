# ~/ML_project/analysis/val_check.py
import numpy as np
import torch
from pathlib import Path
from load import load_model

#Used to validate if the taken weights are correct

CKPT = "/home/ML_project/nano_gpt/weights/model_step000153.safetensors"
VAL = Path(__file__).resolve().parent / "fineweb100B" / "fineweb_val_000000.bin"
S = 1024
ATTN_BLOCKSIZE = 192
N_WINDOWS = 8

def load_shard(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520 and header[1] == 1, f"not a valid shard: {path}"
    print("shard tokens:", header[2])
    return np.fromfile(path, dtype=np.uint16, count=int(header[2]), offset=256 * 4)

def eval_loss(model, tokens, start):
    chunk = torch.from_numpy(tokens[start : start + S + 1].astype(np.int64)).cuda()
    x, y = chunk[:-1], chunk[1:] 
    with torch.no_grad():
        return model(x, target=y, attn_blocksize=ATTN_BLOCKSIZE).item()

# checks
assert CKPT and Path(CKPT).exists(), f"missing checkpoint: {CKPT}"
assert VAL.exists(), f"missing validation shard: {VAL}"
tokens = load_shard(VAL)

# validation
model = load_model(CKPT)
starts = np.linspace(0, len(tokens) - S - 1, N_WINDOWS).astype(int)
losses = [eval_loss(model, tokens, s) for s in starts]
print("per-window losses:", [round(l, 3) for l in losses])
print(f"mean validation loss over {N_WINDOWS} windows: {np.mean(losses):.4f}")