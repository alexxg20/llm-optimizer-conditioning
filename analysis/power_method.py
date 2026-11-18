# ~/ML_project/analysis/power_method.py
import time
import numpy as np


def power_iteration(matvec, n, iters=100, tol=1e-3, seed=0, v0=None, min_iters=15):
    """Largest eigenvalue of a symmetric PSD operator, from matvec(v) alone.

    Stops when the relative residual ||Gv - lambda v|| / |lambda| < tol, but never before
    min_iters iterations (gives a warm start time to leave a non-top eigenvector).
    v0: optional starting vector (warm start); otherwise a random vector from seed.

    Records at every iteration:
      lambda       Rayleigh quotient v^T G v (estimate of lambda_max, a lower bound for PSD G)
      rel_change   |lambda_k - lambda_{k-1}| / |lambda_k|
      rel_resid    ||Gv - lambda v|| / |lambda|  (zero exactly when v is an eigenvector)
    """
    if v0 is None:
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(n).astype(np.float32)
    else:
        v = np.asarray(v0, dtype=np.float32).copy()
    v /= np.linalg.norm(v)

    hist = {"iter": [], "lambda": [], "rel_change": [], "rel_resid": [], "seconds": []}
    lam_prev = None
    t0 = time.time()

    for k in range(1, iters + 1):
        w = matvec(v)                                   # one averaged Gauss-Newton-vector product
        lam = float(np.dot(v, w))                       # Rayleigh quotient
        resid = float(np.linalg.norm(w - lam * v)) / abs(lam)
        rel = abs(lam - lam_prev) / abs(lam) if lam_prev is not None else float("nan")

        hist["iter"].append(k)
        hist["lambda"].append(lam)
        hist["rel_change"].append(rel)
        hist["rel_resid"].append(resid)
        hist["seconds"].append(time.time() - t0)
        print(f"iter {k:4d}  lambda={lam:.6g}  rel_change={rel:.2e}  rel_resid={resid:.2e}", flush=True)

        if lam < 0:
            print("warning: negative Rayleigh quotient, the operator is not PSD or there is a bug")

        if resid < tol and k >= min_iters:
            print(f"converged after {k} iterations")
            break

        norm_w = np.linalg.norm(w)
        if norm_w == 0:
            raise RuntimeError("matvec returned zero: the vector is in the null space")
        v = w / norm_w
        lam_prev = lam

    return hist, v


def plot_history(hist, out_png, title):
    import matplotlib
    matplotlib.use("Agg")                               # no display needed on the cluster
    import matplotlib.pyplot as plt

    it = hist["iter"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].plot(it, hist["lambda"], marker=".")
    ax[0].set_title("lambda estimate (Rayleigh quotient)")
    ax[0].set_xlabel("power iteration")

    ax[1].semilogy(it, hist["rel_change"], marker=".")
    ax[1].set_title("relative change in lambda")
    ax[1].set_xlabel("power iteration")

    ax[2].semilogy(it, hist["rel_resid"], marker=".")
    ax[2].set_title("relative residual ||Gv - lambda v|| / |lambda|")
    ax[2].set_xlabel("power iteration")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)