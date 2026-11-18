# ~/ML_project/analysis/prox_hvp.py
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp, vjp

def prox_hvp(model, theta_ref, x, attn_blocksize, v):
    """Gauss–Newton (= Hessian of L_prox) times v, for one window x.

    theta_ref, v: dicts with the same keys (the parameters being analysed).
    Parameters not in theta_ref are taken from the model and held fixed.
    """
    S = x.shape[-1]
    f = lambda params: functional_call(model, params, (x, None, attn_blocksize, True))[0]

    f0, Jv = jvp(f, (theta_ref,), (v,))                 # forward mode: nothing stored for later
    p = torch.softmax(f0.float(), dim=-1)
    u = (p * Jv - p * (p * Jv).sum(-1, keepdim=True)) / S
    del f0, Jv, p                                       # free the logit-sized tensors

    _, vjp_fn = vjp(f, theta_ref)                       # only now build the backward graph
    (Hv,) = vjp_fn(u)
    del vjp_fn, u
    return Hv


def make_matvec_windows(model, theta_ref, windows, attn_blocksize):
    """v ↦ average over windows of H_i v, for the power method. Vectors live on the CPU."""
    names = list(theta_ref.keys())
    shapes = [theta_ref[n].shape for n in names]
    sizes = [theta_ref[n].numel() for n in names]
    n_total = sum(sizes)

    def matvec(vec_np):
        parts = np.split(vec_np, np.cumsum(sizes)[:-1])
        v = {n: torch.from_numpy(np.ascontiguousarray(p)).view(s).cuda()
             for n, p, s in zip(names, parts, shapes)}
        acc = np.zeros(n_total, dtype=np.float32)
        for x in windows:
            Hv = prox_hvp(model, theta_ref, x, attn_blocksize, v)
            acc += np.concatenate([Hv[n].reshape(-1).cpu().numpy() for n in names])
            del Hv
        del v
        return acc / len(windows)

    return matvec, n_total

# add to prox_hvp.py
import torch.nn.functional as F

def cache_hidden(model, windows, attn_blocksize):
    """Final normalised hidden states (input to lm_head) for each window, stored on the CPU."""
    store = {}
    hook = model.lm_head.register_forward_hook(lambda m, inp, out: store.__setitem__("h", inp[0].detach()))
    hs = []
    with torch.no_grad():
        for x in windows:
            model(x, None, attn_blocksize, True)
            hs.append(store["h"][0].float().cpu())        # (S, 768)
    hook.remove()
    return hs

def make_matvec_lmhead(W_ref, hs):
    """Same operator as make_matvec_windows(..., params='lm_head'), using cached hidden states."""
    shape, n = W_ref.shape, W_ref.numel()

    def matvec(vec_np):
        v = torch.from_numpy(np.ascontiguousarray(vec_np)).view(shape).cuda()
        acc = torch.zeros(shape, device="cuda")
        for h_cpu in hs:
            h = h_cpu.cuda()
            S = h.shape[0]
            f = lambda Wt: 30 * torch.tanh(F.linear(h, Wt) / 30)
            z, Jv = jvp(f, (W_ref,), (v,))
            p = torch.softmax(z, dim=-1)
            u = (p * Jv - p * (p * Jv).sum(-1, keepdim=True)) / S
            del z, Jv, p
            _, vjp_fn = vjp(f, W_ref)
            acc += vjp_fn(u)[0]
            del u, vjp_fn, h
        return (acc / len(hs)).reshape(-1).cpu().numpy()

    return matvec, n