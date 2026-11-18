# ~/ML_project/analysis/load.py
import sys
from pathlib import Path
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpt_model import GPT, GPTConfig

def read_state(ckpt_path):
    if ckpt_path.endswith(".safetensors"):
        state = load_file(ckpt_path, device="cpu")          # plain dict of tensors
        step = int(Path(ckpt_path).stem.split("step")[-1])   # step is in the file name
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        step = ckpt.get("step")
    return state, step

def load_model(ckpt_path, device="cuda", bf16_embeddings=True):
    config = GPTConfig(vocab_size=50304, n_layer=12, n_head=6, n_embd=768)
    state, step = read_state(ckpt_path)
    print("checkpoint step:", step)
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()}
    model = GPT(config)
    model.load_state_dict(state, strict=True)
    model = model.float().to(device).eval()
    if bf16_embeddings:
        # the checkpoint stores these in bf16, so keeping them in bf16 loses nothing
        model.transformer.wte.to(torch.bfloat16)
        model.transformer.vte.to(torch.bfloat16)
    print(f"loaded {ckpt_path} with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    return model