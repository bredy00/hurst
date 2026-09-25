"""
Session I -- exact simulation of fractional Gaussian noise and fBm (models/fbm.py).

  embedding     the circulant of Shevchenko eq. (5) is nonnegative at every H tested
                in (0, 1); the zero-middle row the old copies used is not, from H ~ 0.95
  exactness     the Davies-Harte map's implied covariance IS the Toeplitz matrix
                (computed, not sampled), real and imaginary parts are independent; the
                old copies' map gave half of it
  agreement     Davies-Harte, Cholesky and Hosking give the same second moments
  scaling       Var B_n = n^{2H}; on [0, T], Var B_T = T^{2H} (Shevchenko steps 9-10)
  estimation    quadratic-variation H recovers the planted H; the Breuer-Major variance
                of sqrt(N)(V_N - 1) matches the simulated one for H < 3/4 and the
                normalised variance grows with N above it (no central limit there)
  erratum       Shevchenko's step 5 as printed ("take the real part of the inverse FFT")
                gives a non-stationary covariance; his Matlab code, which keeps the
                complex inverse FFT, gives the exact one

    python test_fbm.py
"""

import math
import sys

import numpy as np

import models.fbm as fb

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))


def toeplitz(n, H):
    return fb.autocovariance(np.subtract.outer(np.arange(n), np.arange(n)), H)


def dh_implied_covariance(n, H):
    """Covariances of (Re Y, Im Y) for Y = FFT(sqrt(lam/M)(Z1 + i Z2)), exactly."""
    lam = fb.circulant_eigenvalues(n, H)
    M = len(lam)
    F = np.exp(-2j * np.pi * np.outer(np.arange(M), np.arange(M)) / M)[:n]
    A = F * np.sqrt(lam / M)[None, :]
    # Re Y = A.re Z1 - A.im Z2,  Im Y = A.im Z1 + A.re Z2
    re_re = A.real @ A.real.T + A.imag @ A.imag.T
    im_im = A.imag @ A.imag.T + A.real @ A.real.T
    re_im = A.real @ A.imag.T - A.imag @ A.real.T
    return re_re, im_im, re_im


def test_embedding():
    print("\nThe circulant embedding")
    worst = min(float(fb.circulant_eigenvalues(2 ** 12 + 1, H).min() / fb.circulant_eigenvalues(2 ** 12 + 1, H).max())
                for H in (0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.999))
    check("eq. (5) embedding nonnegative at H from 0.01 to 0.999 (n = 2^12 + 1)", worst >= 0.0, f"min eigen / max {worst:+.2e}")
    g = fb.autocovariance(np.arange(2 ** 12), 0.99)
    old = np.fft.fft(np.concatenate([g, [0.0], g[:0:-1]])).real
    check("the old copies' zero-middle row is NOT nonnegative at H = 0.99 (why they raised there)",
          float(old.min()) < 0.0, f"min eigenvalue {old.min():.3f}")
    try:
        fb.fgn(10, 1.0)
        refused = False
    except ValueError:
        refused = True
    check("H outside (0, 1) is refused", refused)


def test_exactness():
    print("\nExactness of the Davies-Harte map (computed, not sampled)")
    worst, cross = 0.0, 0.0
    for n, H in ((33, 0.1), (64, 0.3), (65, 0.5), (48, 0.75), (40, 0.95)):
        rr, ii, ri = dh_implied_covariance(n, H)
        T = toeplitz(n, H)
        worst = max(worst, float(np.abs(rr - T).max()), float(np.abs(ii - T).max()))
        cross = max(cross, float(np.abs(ri).max()))
    check("real and imaginary parts each have covariance exactly T (to 1e-12)", worst < 1e-12, f"max error {worst:.1e}")
    check("...and are independent of each other", cross < 1e-12, f"max cross-covariance {cross:.1e}")
    # the old copies: sqrt(lam / 2M) on the zero-middle embedding
    n, H = 64, 0.3
    g = fb.autocovariance(np.arange(n), H)
    c = np.concatenate([g, [0.0], g[:0:-1]])
    m = len(c)
    lam = np.fft.fft(c).real
    F = np.exp(-2j * np.pi * np.outer(np.arange(m), np.arange(m)) / m)[:n]
    A = F * np.sqrt(lam / (2 * m))[None, :]
    old = A.real @ A.real.T + A.imag @ A.imag.T
    check("the old copies' map gave exactly half the covariance (variance 1/2)",
          float(np.abs(old - toeplitz(n, H) / 2).max()) < 1e-12, f"diagonal {old[0, 0]:.6f}")


def test_methods_agree():
    print("\nThree exact methods, the same second moments")
    n, H, size = 12, 0.2, 60000
    T = toeplitz(n, H)
    rng = np.random.default_rng(1)
    for method in fb.METHODS:
        x = fb.fgn(n, H, rng, size=size, method=method)
        emp = x.T @ x / size
        se = np.sqrt((1.0 + T ** 2) / size)                 # Var of a product of two unit Gaussians
        z = float(np.max(np.abs(emp - T) / se))
        check(f"{method}: empirical covariance within 4.5 SE of T everywhere (60k samples)", z < 4.5, f"max |z| {z:.2f}")


def test_scaling():
    print("\nfBm scaling")
    rng = np.random.default_rng(2)
    n, H = 256, 0.3
    paths = fb.fbm(n, H, rng, size=20000)
    v = float(np.var(paths[:, -1]))
    check("Var B_n = n^{2H} on the unit grid", abs(v / n ** (2 * H) - 1) < 0.04, f"{v:.3f} vs {n ** (2 * H):.3f}")
    paths = fb.fbm(n, H, rng, T=2.0, size=20000)
    v = float(np.var(paths[:, -1]))
    check("Var B_T = T^{2H} on [0, T] (steps 9-10)", abs(v / 2.0 ** (2 * H) - 1) < 0.04, f"{v:.4f} vs {2.0 ** (2 * H):.4f}")


def test_estimation():
    print("\nEstimating H, and the Breuer-Major variance")
    rng = np.random.default_rng(3)
    worst = 0.0
    for H in (0.05, 0.1, 0.3, 0.5, 0.7):
        worst = max(worst, abs(fb.quadratic_variation_hurst(fb.fbm(2 ** 16, H, rng)) - H))
    check("quadratic-variation H recovers H from 0.05 to 0.7 within 0.02 (n = 2^16)", worst < 0.02, f"worst error {worst:.4f}")
    # Above 3/4 first differences leave long memory in the squares (sum rho^2 = inf):
    # the error shrinks like N^{2H-2} instead of N^{-1/2}. Shevchenko Remark 5.6
    # recommends a second-order filter there; this estimator is only held to 0.05.
    e9 = abs(fb.quadratic_variation_hurst(fb.fbm(2 ** 16, 0.9, rng)) - 0.9)
    check("H = 0.9 (> 3/4, slower non-central regime): within 0.05", e9 < 0.05, f"error {e9:.4f}")
    for H in (0.12, 0.3, 0.6):
        N = 4096
        x = fb.fgn(N, H, rng, size=3000)
        s = np.sqrt(N) * (np.mean(x ** 2, axis=1) - 1.0)
        pred = fb.breuer_major_variance(H)
        ratio = float(np.var(s) / pred)
        check(f"H = {H}: Var of sqrt(N)(V_N - 1) matches Breuer-Major 2 sum rho(n)^2 = {pred:.3f}",
              abs(ratio - 1) < 0.12, f"simulated / predicted {ratio:.3f}")
    grow = []
    for N in (1024, 16384):
        x = fb.fgn(N, 0.9, rng, size=800)
        grow.append(float(np.var(np.sqrt(N) * (np.mean(x ** 2, axis=1) - 1.0))))
    check("H = 0.9 > 3/4: the normalised variance grows with N (no central limit)", grow[1] > 2.0 * grow[0],
          f"{grow[0]:.1f} at N = 1024 -> {grow[1]:.1f} at N = 16384")


def test_erratum():
    print("\nShevchenko Sec. 6, step 5 as printed vs his Matlab code")
    n, H = 17, 0.3
    lam = fb.circulant_eigenvalues(n, H)
    M = len(lam)
    F = np.exp(-2j * np.pi * np.outer(np.arange(M), np.arange(M)) / M)
    Finv = np.conj(F) / M
    D = np.diag(np.sqrt(lam))
    code = np.real(F @ D @ Finv)[:n]                   # Re FFT( sqrt(lam) * IFFT(zeta) ), zeta real
    text = np.real(F @ D @ np.real(Finv))[:n]          # step 5 as printed: real part of the IFFT first
    T = toeplitz(n, H)
    e_code = float(np.abs(code @ code.T - T).max())
    e_text = float(np.abs(text @ text.T - T).max())
    check("the Matlab code's steps give covariance exactly T", e_code < 1e-12, f"max error {e_code:.1e}")
    d = np.diag(text @ text.T)
    check("step 5 as printed does not: the variances change along the path", e_text > 0.1,
          f"max error {e_text:.3f}; variance {d[0]:.3f} at k = 0, {d[n // 2]:.3f} at k = {n // 2}")


def test_stationary_gaussian():
    """Session L: the same circulant embedding for any stationary covariance."""
    print("\nL -- stationary_gaussian: Shevchenko's embedding for a general covariance")
    for H in (0.1, 0.7):
        a = fb.fgn(3000, H, rng=np.random.default_rng(3), size=5)
        b = fb.stationary_gaussian(lambda k: fb.autocovariance(k, H), 3000, rng=np.random.default_rng(3), size=5)
        check(f"with fGn's covariance it IS fgn(), draw for draw (H = {H})", np.array_equal(a, b))
    # an AR(1) covariance: known, and not fGn
    phi, n, m = 0.8, 2048, 4000
    x = fb.stationary_gaussian(lambda k: phi ** np.abs(k) / (1 - phi * phi), n, rng=np.random.default_rng(4), size=m)
    lag0 = float(np.mean(x * x))
    lag1 = float(np.mean(x[:, 1:] * x[:, :-1]))
    se0 = math.sqrt(2.0 / (1 - phi * phi) ** 2 * (1 + phi * phi) / (1 - phi * phi) / (n * m))
    check("an AR(1) covariance is reproduced at lags 0 and 1",
          abs(lag0 - 1 / (1 - phi * phi)) < 4 * se0 and abs(lag1 - phi / (1 - phi * phi)) < 4 * se0,
          f"lag 0 {lag0:.4f} (exact {1 / (1 - phi * phi):.4f}), lag 1 {lag1:.4f} (exact {phi / (1 - phi * phi):.4f}), SE ~{se0:.4f}")
    try:
        fb.stationary_gaussian(lambda k: np.where(np.abs(k) == 1, 0.9, np.where(k == 0, 1.0, 0.0)), 64,
                               rng=np.random.default_rng(0))
        refused = False
    except ArithmeticError:
        refused = True
    check("a covariance with no nonnegative embedding is refused, not truncated", refused)


if __name__ == "__main__":
    print("=" * 74)
    print("Session I -- fGn and fBm, exactly")
    print("=" * 74)
    test_embedding()
    test_exactness()
    test_methods_agree()
    test_scaling()
    test_estimation()
    test_erratum()
    test_stationary_gaussian()
    print("\n" + "=" * 74)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    print("=" * 74)
    sys.exit(1 if FAIL else 0)
