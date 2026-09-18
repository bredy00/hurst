"""
The bug that led to Davies-Harte in the fixtures -- reproduced and debugged once.

Session A's first fBm fixture was a "cheap Riemann-Liouville" helper:

    kern = (np.arange(1, n + 1)) ** (H - 0.5)      # (j+1)^(H-1/2), j = 0..n-1
    out  = np.convolve(white, kern)[:n]            # X_t = sum_j kern_j w_{t-j}

Fed to the structure-function estimator with a planted H = 0.12 it returned
H = 0.266, which looked like an estimator bug and was not. This script shows
what was actually wrong, quantifies it, and tries the textbook fixes so the
diagnosis can be checked rather than believed.

    python debug_fbm_helper.py
"""

import math

import numpy as np

import volsurf_core as vc

H = 0.12
ALPHA = H - 0.5            # kernel exponent, (t-s)^alpha, singular at s -> t
N = 2 ** 17
RNG = np.random.default_rng(1)


def helper_original(w, H):
    """The Session A helper, verbatim."""
    n = len(w)
    kern = (np.arange(1, n + 1)) ** (H - 0.5)
    kern /= kern[0]
    out = np.convolve(w, kern)[:n]
    return out / np.std(np.diff(out))


def rl_with_kernel(w, kern):
    """Same Riemann-Liouville sum, arbitrary discrete kernel."""
    n = len(w)
    return np.convolve(w, kern)[:n]


def kern_point(n, a):
    """Point-sampled at the RIGHT end of each cell: (j+1)^a. The original."""
    return (np.arange(1, n + 1, dtype=float)) ** a


def kern_cell_mean(n, a):
    """Cell average of the kernel: int_j^{j+1} u^a du. Finite at j = 0."""
    j = np.arange(0, n, dtype=float)
    return ((j + 1) ** (a + 1) - j ** (a + 1)) / (a + 1)


def kern_blp_optimal(n, a):
    """
    Bennedsen-Lunde-Pakkanen (2017) optimal evaluation points
        b_j* = ((j^(a+1) - (j-1)^(a+1)) / (a+1))^(1/a),  j >= 1
    which minimise the L2 error of a Riemann sum against a power kernel.
    The j = 0 cell still has no finite point value, so it is left as the cell
    mean here -- that is the "kappa = 0" hybrid scheme.
    """
    j = np.arange(1, n, dtype=float)
    b = ((j + 1) ** (a + 1) - j ** (a + 1)) / (a + 1)
    b = b ** (1.0 / a)
    return np.concatenate([[kern_cell_mean(1, a)[0]], b ** a])


def hybrid_kappa1(rng, n, a):
    """
    Hybrid scheme, kappa = 1: the singular first cell is simulated EXACTLY as
    the Gaussian pair (dW, int_{t-1}^t (t-s)^a dW_s) with covariance
        Var[dW] = 1,  Var[W~] = 1/(2a+1),  Cov[dW, W~] = 1/(a+1),
    and cells j >= 1 use the BLP optimal points. Bennedsen, Lunde & Pakkanen,
    "Hybrid scheme for Brownian semistationary processes", Finance Stoch 2017.
    """
    cov = np.array([[1.0, 1.0 / (a + 1)], [1.0 / (a + 1), 1.0 / (2 * a + 1)]])
    L = np.linalg.cholesky(cov)
    z = rng.normal(size=(n, 2)) @ L.T
    dw, wt = z[:, 0], z[:, 1]
    k = kern_blp_optimal(n, a)
    k[0] = 0.0                       # first cell handled exactly by wt
    return np.convolve(dw, k)[:n] + wt


def davies_harte(n, H, rng):
    """Exact fBm: the project's shared generator (models/fbm.py, Session I)."""
    from models.fbm import fbm
    return fbm(n, H, rng)


def rho1(path):
    """Lag-1 autocorrelation of the increments."""
    d = np.diff(path)
    d = d - d.mean()
    return float(np.dot(d[1:], d[:-1]) / np.dot(d, d))


def slope(path, deltas, q=1.0):
    m = vc.structure_function(path, deltas, q)
    z, _, r2, _ = vc.zeta_regression(deltas, m)
    return z, r2


def main():
    print("=" * 74)
    print(f"The fBm helper bug, planted H = {H}   (n = {N})")
    print("=" * 74)

    w = RNG.normal(size=N)
    paths = {
        "ORIGINAL helper, point kernel (j+1)^a": helper_original(w, H),
        "cell-mean kernel  int_j^{j+1} u^a du":  rl_with_kernel(w, kern_cell_mean(N, ALPHA)),
        "BLP optimal points (hybrid, kappa=0)":  rl_with_kernel(w, kern_blp_optimal(N, ALPHA)),
        "hybrid scheme, kappa=1 (first cell exact)": hybrid_kappa1(np.random.default_rng(2), N, ALPHA),
        "Davies-Harte, exact fBm":                davies_harte(N, H, np.random.default_rng(3)),
    }

    rho_exact = 0.5 * (2 ** (2 * H) - 2)
    print(f"\n1. What the estimator reads (lags 1..60, the fixture's setting)")
    print(f"   exact fGn lag-1 autocorrelation at H={H}: {rho_exact:+.4f}")
    print(f"   {'generator':44} {'H_hat':>7} {'rho(1)':>8} {'r2':>8}")
    for name, p in paths.items():
        res = vc.hurst_from_structure(p)
        print(f"   {name:44} {res['H']:7.4f} {rho1(p):+8.4f} {res['linearity_r2']:8.5f}")

    print(f"\n2. Where the error lives: zeta(1) slope by lag window")
    print(f"   (a correct generator gives H = {H} in every window)")
    windows = {"1-8": np.arange(1, 9), "8-64": np.arange(8, 65, 4),
               "64-512": np.arange(64, 513, 32), "512-4096": np.arange(512, 4097, 256)}
    print(f"   {'generator':44} " + " ".join(f"{k:>9}" for k in windows))
    for name, p in paths.items():
        row = []
        for d in windows.values():
            z, _ = slope(p, d)
            row.append(f"{z:9.4f}")
        print(f"   {name:44} " + " ".join(row))

    print(f"\n3. Is the type-II (non-stationary start) part of it?")
    p = paths["ORIGINAL helper, point kernel (j+1)^a"]
    full = vc.hurst_from_structure(p)['H']
    burnt = vc.hurst_from_structure(p[20000:])['H']
    print(f"   original helper, all samples : H = {full:.4f}")
    print(f"   original helper, drop 20000  : H = {burnt:.4f}   "
          f"(gap {abs(full-burnt):.4f} -> transient is not the cause)")

    print(f"\n4. The kernel itself at the first few lags, alpha = {ALPHA:+.2f}")
    a = ALPHA
    print(f"   {'j':>3} {'(j+1)^a  [orig]':>16} {'cell mean':>12} {'BLP b_j*^a':>12}")
    kp, kc, kb = kern_point(6, a), kern_cell_mean(6, a), kern_blp_optimal(6, a)
    for j in range(6):
        print(f"   {j:3d} {kp[j]:16.4f} {kc[j]:12.4f} {kb[j]:12.4f}")
    print(f"   the true kernel (t-s)^a -> infinity as s -> t; the original caps "
          f"it at 1.0000\n   and so under-weights the most recent shock, which is "
          f"the whole source of roughness.")


if __name__ == "__main__":
    main()
