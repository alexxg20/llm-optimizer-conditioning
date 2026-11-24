import torch
from torch.func import functional_call, jvp, vjp

class LinearizedModel:
    """Linearization of the model's logits around theta_ref, for one input window x."""

    def __init__(self, model, theta_ref, x, attn_blocksize):
        self.model = model
        self.theta_ref = {k: p.detach() for k, p in theta_ref.items()}
        self.x = x                              # token IDs, shape (S,); no targets needed
        self.attn_blocksize = attn_blocksize
        # f0 = f(theta_ref), and vjp_fn is the backward map at theta_ref
        self.f0, self._vjp_fn = vjp(self._f, self.theta_ref)

    def _f(self, params):
        # logits of shape (S, V) as a function of the parameter dictionary
        return functional_call(self.model, params, (self.x, None, self.attn_blocksize, True))[0]

    def jvp_op(self, v):
        """J v: how the logits change when the parameters move by v. Returns (S, V)."""
        _, Jv = jvp(self._f, (self.theta_ref,), (v,))
        return Jv

    def vjp_op(self, u):
        """Jᵀ u: the parameter-space vector corresponding to a weighting u of shape (S, V)."""
        (g,) = self._vjp_fn(u)
        return g

    def linear_logits(self, theta):
        """f_lin(theta) = f(theta_ref) + J (theta - theta_ref)."""
        delta = {k: theta[k] - self.theta_ref[k] for k in self.theta_ref}
        return self.f0 + self.jvp_op(delta)


def check_linearization(lin, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    v = {k: torch.randn(p.shape, generator=g, device="cuda") for k, p in lin.theta_ref.items()}
    u = torch.randn(lin.f0.shape, generator=g, device="cuda")

    # 1. the linearization is exact at the reference point
    print("f_lin(theta_ref) == f0 :", torch.allclose(lin.linear_logits(lin.theta_ref), lin.f0))

    # 2. adjoint test: <u, Jv> must equal <Jᵀu, v>
    lhs = (u * lin.jvp_op(v)).sum().item()
    g_u = lin.vjp_op(u)
    rhs = sum((g_u[k] * v[k]).sum() for k in v).item()
    print(f"adjoint test: <u,Jv>={lhs:.6e}  <Jᵀu,v>={rhs:.6e}  rel.diff={abs(lhs-rhs)/abs(lhs):.2e}")

    # 3. finite difference: (f(θ+εv) − f(θ)) / ε ≈ Jv
    eps = 1e-3
    with torch.no_grad():
        theta_eps = {k: lin.theta_ref[k] + eps * v[k] for k in v}
        fd = (lin._f(theta_eps) - lin.f0) / eps
    Jv = lin.jvp_op(v)
    print("finite-difference rel. error:", ((fd - Jv).norm() / Jv.norm()).item())