#!/usr/bin/env python3
"""GPU compatibility check for the kernels the pinned modded-nanogpt model needs. It does not train.

Runs each operation in bfloat16, the way train_gpt2.py uses it, compares it with an fp32 reference, and prints
PASS/FAIL per item with the error text. Takes about 1-2 minutes (mostly torch.compile).
Run from pretraining/:  .venv/bin/python scripts/check_gpu.py
"""
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

DEVICE = "cuda"
BF16 = torch.bfloat16


def rel_err(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-12)).item()


def check_bf16_matmul():
    n = 4096
    a = torch.randn(n, n, device=DEVICE, dtype=BF16)
    b = torch.randn(n, n, device=DEVICE, dtype=BF16)
    err = rel_err(a @ b, a.float() @ b.float())
    assert err < 2e-2, f"relative error {err:.2e} vs fp32"
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(10):
        a @ b
    torch.cuda.synchronize()
    tflops = 10 * 2 * n**3 / (time.perf_counter() - t) / 1e12
    return f"relative error {err:.1e}; {tflops:.1f} TFLOPS for a {n}^3 bf16 matmul"


def check_flex_attention():
    # 6 heads of dim 128 and a causal + same-document + sliding-window mask, as in the pinned GPT.
    S, H, D = 2048, 6, 128
    torch.manual_seed(0)
    idx = torch.randint(0, 50257, (S,), device=DEVICE)
    idx[torch.randint(0, S, (8,), device=DEVICE)] = 50256
    docs = (idx == 50256).cumsum(0)
    attn_blocksize = torch.tensor(1024, dtype=torch.int, device=DEVICE)

    def document_causal_mask(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = docs[q_idx] == docs[kv_idx]
        window_mask = q_idx - kv_idx < attn_blocksize
        return causal_mask & document_mask & window_mask

    compiled_mask = torch.compile(create_block_mask, dynamic=False)
    compiled_attention = torch.compile(flex_attention, dynamic=False)
    block_mask = compiled_mask(document_causal_mask, None, None, S, S, device=DEVICE, _compile=True)
    q, k, v = (torch.randn(1, H, S, D, device=DEVICE, dtype=BF16, requires_grad=True) for _ in range(3))
    out = compiled_attention(q, k, v, block_mask=block_mask)
    out.float().square().sum().backward()

    pos = torch.arange(S, device=DEVICE)
    dense = (pos[:, None] >= pos[None, :]) & (docs[:, None] == docs[None, :]) & (pos[:, None] - pos[None, :] < 1024)
    q32, k32, v32 = (t.detach().float().requires_grad_() for t in (q, k, v))
    ref = F.scaled_dot_product_attention(q32, k32, v32, attn_mask=dense)
    ref.square().sum().backward()
    errs = {"out": rel_err(out, ref), "dq": rel_err(q.grad, q32.grad), "dk": rel_err(k.grad, k32.grad),
            "dv": rel_err(v.grad, v32.grad)}
    bad = {k: v for k, v in errs.items() if not v < 5e-2}
    assert not bad, f"relative error vs fp32 too large: {bad}"
    return "relative errors vs fp32: " + ", ".join(f"{k} {v:.1e}" for k, v in errs.items())


def check_rms_norm():
    x = torch.randn(4096, 768, device=DEVICE, dtype=BF16, requires_grad=True)

    def norm(t):
        return F.rms_norm(t, (t.size(-1),))

    eps = torch.finfo(BF16).eps
    x32 = x.detach().float()
    ref = x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + eps)
    eager = norm(x)
    compiled = torch.compile(norm)(x)
    compiled.float().sum().backward()
    errs = {"eager": rel_err(eager, ref), "compiled": rel_err(compiled, ref)}
    assert all(v < 3e-2 for v in errs.values()), f"relative error vs fp32 too large: {errs}"
    assert torch.isfinite(x.grad).all(), "non-finite gradient from the compiled backward"
    return f"relative errors vs fp32: eager {errs['eager']:.1e}, compiled {errs['compiled']:.1e}; compiled backward finite"


def check_fused_adam():
    # Same shape and hyperparameters as the token-embedding Adam group in the pinned script.
    p = torch.nn.Parameter(torch.randn(50304, 768, device=DEVICE, dtype=BF16))
    before = p.detach().float().clone()
    opt = torch.optim.Adam([p], lr=0.6, betas=(0.8, 0.95), fused=True)
    p.grad = torch.randn_like(p)
    opt.step()
    torch.cuda.synchronize()
    assert torch.isfinite(p).all(), "non-finite parameters after the step"
    # Bias correction makes the first Adam step lr * g / (|g| + eps), i.e. about lr * sign(g).
    dev = ((before - p.detach().float()) - 0.6 * p.grad.float().sign()).abs().max().item()
    assert dev < 0.05, f"first step deviates from lr*sign(grad) by {dev:.3f}"
    return f"first step matches lr*sign(grad) within {dev:.3f}"


CHECKS = [
    ("bf16 matmul", check_bf16_matmul),
    ("torch.compile'd FlexAttention forward+backward (bf16, causal+document+window block mask)", check_flex_attention),
    ("F.rms_norm, eager and torch.compile'd with backward (bf16)", check_rms_norm),
    ("fused Adam step on a bf16 parameter", check_fused_adam),
]


def main():
    if not torch.cuda.is_available():
        print("FAIL: CUDA is not available to PyTorch (check the driver, CUDA_VISIBLE_DEVICES and the torch build).")
        return 2
    name = torch.cuda.get_device_name()
    major, minor = torch.cuda.get_device_capability()
    print(f"GPU: {name}, compute capability {major}.{minor}")
    print(f"torch {torch.__version__}, CUDA runtime {torch.version.cuda}, compiled for {' '.join(torch.cuda.get_arch_list())}")
    print(f"bf16 supported natively: {torch.cuda.is_bf16_supported(including_emulation=False)}, "
          f"with emulation: {torch.cuda.is_bf16_supported(including_emulation=True)}\n")
    failed = 0
    for label, fn in CHECKS:
        t = time.perf_counter()
        try:
            result, status = fn(), "PASS"
        except Exception as e:
            result, status = f"{type(e).__name__}: {str(e)[:3000]}", "FAIL"
            failed += 1
        print(f"[{status}] {label} ({time.perf_counter() - t:.1f} s)\n       {result}", flush=True)
        torch.cuda.empty_cache()
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
