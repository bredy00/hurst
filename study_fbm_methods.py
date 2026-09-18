"""
How the project simulates fBm, and how that compares with the other methods (Session I).

  A  exactness, computed rather than sampled: the implied covariance of each linear
     map z -> sample against the fGn Toeplitz matrix -- Davies-Harte on Shevchenko's
     eq. (5), Wood-Chan with power-of-two padding, Cholesky, Hosking; the repo's old
     fixture copy; Shevchenko's step 5 as printed
  B  the embedding's smallest eigenvalue across H in (0, 1): eq. (5) vs the old copies'
     zero-middle row
  C  cost per sample against n for the exact methods; and whether step 1's power-of-two
     padding still buys anything with NumPy's FFT
  D  the estimator side: Breuer-Major's variance for sqrt(N)(V_N - 1) against simulation
     across H, and the loss of the central limit above H = 3/4

    python study_fbm_methods.py    (~2 minutes) -> captures/fbm_methods.json, .png
"""

import json
import math
import pathlib
import time

import numpy as np

import models.fbm as fb

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "captures" / "fbm_methods.json"


def toeplitz(n, H):
    return fb.autocovariance(np.subtract.outer(np.arange(n), np.arange(n)), H)


class _Identity:
    """An rng whose standard_normal returns the identity: turns a generator into its matrix."""

    def __init__(self, n):
        self.n = n

    def standard_normal(self, shape):
        return np.eye(self.n)


def implied_cov(method, n, H):
    """Covariance implied by a generator's linear map, exactly."""
    T = toeplitz(n, H)
    if method == "cholesky":
        L = np.linalg.cholesky(T)
        return L @ L.T
    if method == "hosking":
        L = fb._hosking(n, H, _Identity(n), n).T          # rows of the output are L's columns
        return L @ L.T
    if method in ("davies-harte", "wood-chan pow2", "old repo copy"):
        if method == "davies-harte":
            lam = np.array(fb.circulant_eigenvalues(n, H))
            M, s = len(lam), 1.0
        elif method == "wood-chan pow2":
            M = 1 << int(math.ceil(math.log2(2 * (n - 1))))
            g = fb.autocovariance(np.arange(M // 2 + 1), H)
            c = np.concatenate([g, g[-2:0:-1]])
            lam = np.maximum(np.fft.fft(c).real, 0.0)
            s = 1.0
        else:
            g = fb.autocovariance(np.arange(n), H)
            c = np.concatenate([g, [0.0], g[:0:-1]])
            lam = np.maximum(np.fft.fft(c).real, 0.0)
            M, s = len(c), 0.5                               # sqrt(lam / 2M): half the variance
        F = np.exp(-2j * np.pi * np.outer(np.arange(M), np.arange(M)) / M)[:n]
        A = F * np.sqrt(s * lam / M)[None, :]
        return A.real @ A.real.T + A.imag @ A.imag.T
    if method == "shevchenko step 5 as printed":
        lam = np.array(fb.circulant_eigenvalues(n, H))
        M = len(lam)
        F = np.exp(-2j * np.pi * np.outer(np.arange(M), np.arange(M)) / M)
        B = np.real(F @ np.diag(np.sqrt(lam)) @ np.real(np.conj(F) / M))[:n]
        return B @ B.T
    raise ValueError(method)


def section_a():
    out = {}
    methods = ("davies-harte", "wood-chan pow2", "cholesky", "hosking", "old repo copy", "shevchenko step 5 as printed")
    for method in methods:
        rows = {}
        for n, H in ((64, 0.1), (64, 0.5), (65, 0.75), (48, 0.95)):
            T = toeplitz(n, H)
            C = implied_cov(method, n, H)
            rows[f"n={n}, H={H}"] = {"max_abs_error": float(np.abs(C - T).max()),
                                     "variance_first": float(C[0, 0]), "variance_middle": float(C[n // 2, n // 2])}
        out[method] = rows
        worst = max(r["max_abs_error"] for r in rows.values())
        print(f"A  {method:30s} worst |implied cov - Toeplitz| {worst:.2e}", flush=True)
    return out


def section_b():
    Hs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.49, 0.5, 0.51, 0.6, 0.7, 0.8, 0.9, 0.93, 0.95, 0.97, 0.99, 0.995, 0.999]
    n = 2 ** 15
    eq5, old = [], []
    for H in Hs:
        g = fb.autocovariance(np.arange(n), H)
        a = np.fft.fft(np.concatenate([g, g[-2:0:-1]])).real
        b = np.fft.fft(np.concatenate([g, [0.0], g[:0:-1]])).real
        eq5.append(float(a.min() / a.max()))
        old.append(float(b.min() / b.max()))
    first_neg = next((H for H, v in zip(Hs, old) if v < 0), None)
    print(f"B  eq. (5): min eigen/max >= {min(eq5):+.2e} over H in [0.01, 0.999]; old zero-middle row first negative at H = {first_neg}", flush=True)
    return {"H": Hs, "n": n, "eq5_min_over_max": eq5, "old_min_over_max": old, "old_first_negative_H": first_neg}


def _best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def section_c():
    rng = np.random.default_rng(0)
    H = 0.1
    out = {"H": H, "n": [], "davies-harte": [], "cholesky": [], "hosking": []}
    for q in range(6, 21):
        n = 2 ** q
        out["n"].append(n)
        fb.circulant_eigenvalues.cache_clear()
        out["davies-harte"].append(_best(lambda: fb.fgn(n, H, rng)))           # includes the eigenvalues once
        out["cholesky"].append(_best(lambda: fb.fgn(n, H, rng, method="cholesky")) if n <= 4096 else None)
        out["hosking"].append(_best(lambda: fb.fgn(n, H, rng, method="hosking"), reps=1) if n <= 16384 else None)
    # step 1's power-of-two padding: n values from M = 2(n-1) vs the next power of two
    pad = []
    for n in (1500, 3000, 6000, 12001, 50001, 100003):
        g = fb.autocovariance(np.arange(n), H)
        c_min = np.concatenate([g, g[-2:0:-1]])
        M2 = 1 << int(math.ceil(math.log2(2 * (n - 1))))
        g2 = fb.autocovariance(np.arange(M2 // 2 + 1), H)
        c_pow = np.concatenate([g2, g2[-2:0:-1]])
        z1 = rng.standard_normal(len(c_min)) + 1j * rng.standard_normal(len(c_min))
        z2 = rng.standard_normal(M2) + 1j * rng.standard_normal(M2)
        t_min = _best(lambda: np.fft.fft(z1), reps=7)
        t_pow = _best(lambda: np.fft.fft(z2), reps=7)
        pad.append({"n": n, "M_minimal": len(c_min), "M_pow2": M2, "fft_minimal_s": t_min, "fft_pow2_s": t_pow})
    out["padding"] = pad
    print("C  seconds per sample: " + ", ".join(f"n=2^{int(math.log2(n))}: DH {d*1e3:.2f} ms"
                                              + (f", Chol {c*1e3:.0f} ms" if c else "") + (f", Hosk {h*1e3:.0f} ms" if h else "")
                                              for n, d, c, h in zip(out["n"], out["davies-harte"], out["cholesky"], out["hosking"])
                                              if n in (256, 4096, 16384, 2 ** 20)), flush=True)
    print("C  FFT of the minimal embedding vs power-of-two padding: " + ", ".join(
        f"n={p['n']}: {p['fft_minimal_s']*1e6:.0f} vs {p['fft_pow2_s']*1e6:.0f} us" for p in pad), flush=True)
    return out


def section_d():
    rng = np.random.default_rng(4)
    rows = []
    for H in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        N = 4096
        x = fb.fgn(N, H, rng, size=2000)
        s = np.sqrt(N) * (np.mean(x ** 2, axis=1) - 1.0)
        pred = fb.breuer_major_variance(H)
        rows.append({"H": H, "simulated": float(np.var(s)), "breuer_major": pred})
    growth = []
    for H in (0.8, 0.9):
        v = []
        for N in (512, 2048, 8192, 32768):
            x = fb.fgn(N, H, rng, size=400)
            v.append(float(np.var(np.sqrt(N) * (np.mean(x ** 2, axis=1) - 1.0))))
        growth.append({"H": H, "N": [512, 2048, 8192, 32768], "normalised_variance": v})
    print("D  simulated / Breuer-Major variance: " + ", ".join(f"H={r['H']}: {r['simulated']/r['breuer_major']:.3f}" for r in rows), flush=True)
    print("D  above 3/4 the normalised variance grows: " + "; ".join(
        f"H={g['H']}: " + " -> ".join(f"{v:.1f}" for v in g["normalised_variance"]) for g in growth), flush=True)
    return {"rows": rows, "growth": growth}


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))
    a = ax[0, 0]
    B = res["B"]
    Hs = np.array(B["H"])
    a.semilogy(Hs, np.abs(B["eq5_min_over_max"]), "o-", color="#1f77b4", label="Shevchenko eq. (5): all positive")
    old = np.array(B["old_min_over_max"])
    a.semilogy(Hs[old >= 0], old[old >= 0], "s", color="#ff7f0e", label="old repo row (0 in the middle), positive")
    a.semilogy(Hs[old < 0], -old[old < 0], "x", color="#d62728", ms=9, mew=2, label="old repo row, NEGATIVE")
    a.axvline(0.95, color="0.8", lw=0.8)
    a.set_xlabel("H")
    a.set_ylabel("|smallest eigenvalue| / largest")
    a.set_title("(a) the circulant embedding, n = 2^15")
    a.legend(fontsize=8)
    a = ax[0, 1]
    C = res["C"]
    n = np.array(C["n"], float)
    for key, col in (("davies-harte", "#1f77b4"), ("hosking", "#2ca02c"), ("cholesky", "#d62728")):
        y = np.array([np.nan if v is None else v for v in C[key]], float)
        a.loglog(n, y, "o-", color=col, label={"davies-harte": "Davies-Harte, O(n log n)", "hosking": "Hosking, O(n^2)",
                                                "cholesky": "Cholesky, O(n^3)"}[key])
    a.set_xlabel("n (values of fGn)")
    a.set_ylabel("seconds per sample")
    a.set_title("(b) cost of an exact sample")
    a.legend(fontsize=8)
    a = ax[1, 0]
    nn, H = 64, 0.3
    k = np.arange(nn)
    for method, col, lab in (("davies-harte", "#1f77b4", "Davies-Harte (eq. (5), sqrt(lam/M))"),
                             ("old repo copy", "#ff7f0e", "old repo copy: variance 1/2"),
                             ("shevchenko step 5 as printed", "#d62728", "step 5 as printed: non-stationary")):
        a.plot(k, np.diag(implied_cov(method, nn, H)), color=col, lw=1.8, label=lab)
    a.axhline(1.0, color="0.6", ls=":")
    a.set_ylim(0, 1.3)
    a.set_xlabel("position k along the sample")
    a.set_ylabel("implied Var(xi_k), exact")
    a.set_title("(c) what the implementation details do to the variance (H = 0.3)")
    a.legend(fontsize=8)
    a = ax[1, 1]
    D = res["D"]
    Hd = [r["H"] for r in D["rows"]]
    a.plot(Hd, [r["simulated"] for r in D["rows"]], "o", color="#1f77b4", label="simulated Var sqrt(N)(V_N - 1), N = 4096")
    a.plot(Hd, [r["breuer_major"] for r in D["rows"]], "-", color="k", label="Breuer-Major: 2 sum_n rho(n)^2")
    for g, mk in zip(D["growth"], ("^", "v")):
        a.plot([g["H"]] * len(g["N"]), g["normalised_variance"], mk, color="#d62728", alpha=0.7,
               label=f"H = {g['H']}: N = 512 ... 32768 (grows)")
    a.axvline(0.75, color="0.6", ls="--")
    a.text(0.755, 2.4, "H = 3/4", fontsize=8, color="0.4")
    a.set_yscale("log")
    a.set_xlabel("H")
    a.set_ylabel("variance")
    a.set_title("(d) the estimator side: Breuer-Major holds below 3/4 only")
    a.legend(fontsize=7)
    fig.suptitle("Simulating fBm: what the project does, and the alternatives", fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "captures" / "fbm_methods.png", dpi=120)


def main():
    res = {}
    for key, fn in (("A", section_a), ("B", section_b), ("C", section_c), ("D", section_d)):
        res[key] = fn()
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    plot(res)
    return res


if __name__ == "__main__":
    main()
