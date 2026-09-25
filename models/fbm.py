"""
Exact simulation of fractional Gaussian noise (fGn) and fractional Brownian motion (Session I).

fBm B^H on the grid t_k = k T / n is (T/n)^H times the cumulative sum of fGn, the
stationary Gaussian sequence xi_k = B^H_k - B^H_{k-1} with unit variance and covariance

    rho_H(k) = ( |k+1|^{2H} - 2 |k|^{2H} + |k-1|^{2H} ) / 2.

Simulating it means finding a square root S of the n x n Toeplitz covariance
T = [rho_H(|i-j|)] (S S' = T) and returning S z for standard Gaussian z. Three exact ways:

  "davies-harte"  (default)  circulant embedding, O(n log n). Embed T in the
      M x M circulant C = circ(c), M = 2(N'-1), c = (rho(0), ..., rho(N'-1), rho(N'-2), ...,
      rho(1)), N' >= n (N' = n unless 2(n-1) needs padding to a fast FFT size; see
      circulant_eigenvalues) -- Shevchenko (2014, "Fractional Brownian motion in a nutshell",
      arXiv:1406.1956, Sec. 6, eq. (5)); Davies & Harte (1987); Wood & Chan (1994);
      Dietrich & Newsam (1997). C = Q Lambda Q* with Lambda = FFT(c), so
      S = Q Lambda^{1/2} Q* is real whenever Lambda >= 0, which for fGn holds at every H
      in (0, 1) (Craigmile 2003 for H <= 1/2; Dietrich & Newsam 1997 and Perrin et al.
      2002 for H >= 1/2). Wood and Chan's fallbacks -- enlarging M, or truncating
      negative eigenvalues -- are therefore never needed for fGn, and this is refused
      rather than patched if it ever happens. One complex FFT of sqrt(Lambda/M)(Z1 + iZ2)
      gives TWO independent exact samples, its real and imaginary parts.
  "cholesky"      T = L L', O(n^3): the reference, for tests and small n.
  "hosking"       Durbin-Levinson recursion, O(n^2): exact sequential conditionals.

Until Session I three copies of a Davies-Harte routine lived in test_surface.py,
healthcheck.py and capture_v3.py, with two defects found against Shevchenko Sec. 6:

  1. the middle entry of the circulant row was 0, not rho(n-1). It sits outside the
     Toeplitz block, so draws stayed exact wherever the embedding stayed nonnegative,
     but with the 0 it does not from H ~ 0.93 at n = 2^15 (min eigenvalue / max = -9.8e-6 at
     H = 0.95, -1.5e-5 at 0.99; the copies raised there and blamed fBm). With (5) it
     is positive at every H tested up to 0.999.
  2. the complex trick was scaled by sqrt(Lambda / 2M) instead of sqrt(Lambda / M), so
     the "fGn" had variance 1/2: the right correlations, the wrong scale (checked
     exactly: implied covariance = T / 2 to 1.5e-15).

Neither touched a reported number: every fixture used H <= 0.5, and every estimator
they fed (structure function, log-log slopes) is scale-free.

Breuer-Major is not a simulation method. It is the central limit theorem behind the
ESTIMATORS of H: for g of Hermite rank p with sum_n |rho(n)|^p < inf,
N^{-1/2} sum_k g(xi_k) -> N(0, sum_{q >= p} g_q^2 q! sum_{n in Z} rho(n)^q). With
g(x) = x^2 - 1 (rank 2) it gives the sqrt(N) rate of quadratic-variation estimators
iff H < 3/4 for first differences (Shevchenko Thm 5.5); breuer_major_variance and
test_fbm.py check that prediction against the simulated noise.
"""

import functools
import math

import numpy as np

METHODS = ("davies-harte", "cholesky", "hosking")


def autocovariance(k, H):
    """rho_H(k) of unit-variance fGn, vectorised over k."""
    k = np.abs(np.asarray(k, dtype=float))
    return 0.5 * (np.abs(k + 1.0) ** (2 * H) - 2.0 * k ** (2 * H) + np.abs(k - 1.0) ** (2 * H))


def _fast_len(m):
    """Smallest 2^a 3^b 5^c >= m: sizes NumPy's FFT handles without Bluestein's detour."""
    best = 1 << max(m - 1, 0).bit_length()
    f5 = 1
    while f5 < best:
        f35 = f5
        while f35 < best:
            f = f35
            while f < m:
                f *= 2
            best = min(best, f)
            f35 *= 3
        f5 *= 5
    return best


@functools.lru_cache(maxsize=32)
def circulant_eigenvalues(n, H):
    """
    Eigenvalues of the circulant embedding of n values of fGn, as a read-only array.

    Shevchenko's eq. (5) for N' = M/2 + 1 >= n values, with M = 2 * fast_len(n - 1): the
    minimal size 2(n - 1) when that has no large prime factor, padded otherwise. Step 1
    of his algorithm pads to a power of two; that still pays with NumPy's FFT (measured:
    n = 1500 gives M = 2998 = 2 x 1499, 4.4x slower than 4096), but a 5-smooth size is
    enough. Padding means simulating N' values and keeping the first n, so the
    embedding is still eq. (5)'s and still nonnegative. Raises if an eigenvalue is
    negative beyond rounding -- for fGn that would contradict the theorem, so it would
    mean a bug, not a need for Wood-Chan's approximate fallback.
    """
    if n < 2:
        return np.ones(max(n, 1))
    M = 2 * _fast_len(n - 1)
    g = autocovariance(np.arange(M // 2 + 1), H)
    c = np.concatenate([g, g[-2:0:-1]])                 # length M, eq. (5) for N' = M/2 + 1
    lam = np.fft.fft(c).real
    floor = -1e-10 * float(np.abs(lam).max())
    if float(lam.min()) < floor:
        raise ArithmeticError(f"circulant embedding of fGn not nonnegative at H={H}, n={n}: "
                              f"min eigenvalue {lam.min():.3e} (max {lam.max():.3e})")
    lam = np.maximum(lam, 0.0)
    lam.setflags(write=False)
    return lam


def _circulant_draw(lam, n, rng, size):
    """Steps 3-8 of Shevchenko's algorithm for an embedding with eigenvalues lam >= 0."""
    M = len(lam)
    scale = np.sqrt(lam / M)
    pairs = (size + 1) // 2
    Z = rng.standard_normal((pairs, M)) + 1j * rng.standard_normal((pairs, M))
    Y = np.fft.fft(scale[None, :] * Z, axis=1)[:, :n]
    return np.concatenate([Y.real, Y.imag], axis=0)[:size]


def _davies_harte(n, H, rng, size):
    lam = circulant_eigenvalues(n, H)
    if n < 2:
        return rng.standard_normal((size, n))
    return _circulant_draw(lam, n, rng, size)


def stationary_gaussian(acov, n, rng=None, size=None):
    """
    n values of the stationary Gaussian sequence with autocovariance acov(k), by the same
    circulant embedding (Shevchenko's eq. (5) and steps 3-8), for any covariance whose
    embedding is nonnegative: shape (n,), or (size, n). `acov` is a callable on integer
    lags. Unlike fGn's, a general embedding can have negative eigenvalues; this raises
    then, as circulant_eigenvalues does, rather than truncating them (Wood-Chan's
    approximate fallback). Session L uses it for the lifted kernel's stationary increments.
    """
    rng = np.random.default_rng() if rng is None else rng
    k = 1 if size is None else int(size)
    M = 2 * _fast_len(max(int(n) - 1, 1))
    g = np.asarray(acov(np.arange(M // 2 + 1)), dtype=float)
    c = np.concatenate([g, g[-2:0:-1]])
    lam = np.fft.fft(c).real
    if float(lam.min()) < -1e-10 * float(np.abs(lam).max()):
        raise ArithmeticError(f"circulant embedding not nonnegative at n={n}: min eigenvalue "
                              f"{lam.min():.3e} (max {lam.max():.3e})")
    out = _circulant_draw(np.maximum(lam, 0.0), int(n), rng, k)
    return out[0] if size is None else out


def _cholesky(n, H, rng, size):
    T = autocovariance(np.subtract.outer(np.arange(n), np.arange(n)), H)
    L = np.linalg.cholesky(T)
    return rng.standard_normal((size, n)) @ L.T


def _hosking(n, H, rng, size):
    """Durbin-Levinson: xi_k | xi_{k-1..0} ~ N(phi_k . past, v_k), exactly."""
    r = autocovariance(np.arange(n), H)
    out = np.empty((size, n))
    z = rng.standard_normal((size, n))
    out[:, 0] = z[:, 0]
    phi = np.zeros(n)
    v = 1.0
    for k in range(1, n):
        # reflection coefficient, then the order-k predictor from the order-(k-1) one
        a = (r[k] - phi[:k - 1] @ r[k - 1:0:-1]) / v if k > 1 else r[1]
        if k > 1:
            phi[:k - 1] = phi[:k - 1] - a * phi[:k - 1][::-1]
        phi[k - 1] = a
        v *= (1.0 - a * a)
        out[:, k] = out[:, k - 1::-1] @ phi[:k] + math.sqrt(v) * z[:, k]
    return out


def fgn(n, H, rng=None, size=None, method="davies-harte"):
    """
    n values of unit-variance fGn at Hurst index H in (0, 1): shape (n,), or (size, n).
    """
    if not 0.0 < H < 1.0:
        raise ValueError(f"H must be in (0, 1), got {H}")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    rng = np.random.default_rng() if rng is None else rng
    k = 1 if size is None else int(size)
    draw = {"davies-harte": _davies_harte, "cholesky": _cholesky, "hosking": _hosking}[method]
    out = draw(int(n), float(H), rng, k)
    return out[0] if size is None else out


def fbm(n, H, rng=None, T=None, size=None, method="davies-harte"):
    """
    fBm at t_k = k T / n, k = 1..n (B_0 = 0 is not included): the cumulative sum of fGn,
    times (T/n)^H when T is given (Shevchenko steps 9-10); on the unit grid otherwise.
    """
    x = fgn(n, H, rng, size, method)
    path = np.cumsum(x, axis=-1)
    return path if T is None else path * (float(T) / n) ** H


def quadratic_variation_hurst(path):
    """
    Shevchenko's standard estimator (Sec. 5, filter 'Increments 1', dilations 1 and 2):
    H = (1/2) log2( V(d^2) / V(d) ), V the mean squared increment at lag 1 and 2.
    Scale-free, strongly consistent; sqrt(N)-normal for H < 3/4 (Breuer-Major).
    """
    b = np.asarray(path, dtype=float)
    v1 = np.mean(np.diff(b) ** 2)
    v2 = np.mean((b[2:] - b[:-2]) ** 2)
    return 0.5 * math.log2(v2 / v1)


def breuer_major_variance(H, lags=200_000):
    """
    Breuer-Major variance of N^{-1/2} sum_k (xi_k^2 - 1) for fGn: g(x) = x^2 - 1 has
    Hermite rank 2 with g_2 = 1, so sigma^2 = 2! sum_{n in Z} rho(n)^2, finite iff
    H < 3/4 (the tail rho(n)^2 ~ n^{4H-4} is summed to `lags` and extrapolated).
    """
    if H >= 0.75:
        return float("inf")
    n = np.arange(1, lags + 1)
    r2 = autocovariance(n, H) ** 2
    tail_c = (H * (2 * H - 1)) ** 2               # rho(n) ~ H(2H-1) n^{2H-2}
    p = 4.0 - 4.0 * H                              # tail ~ tail_c n^{-p}, p > 1
    tail = tail_c * lags ** (1.0 - p) / (p - 1.0) if H != 0.5 else 0.0
    return 2.0 * (1.0 + 2.0 * (float(r2.sum()) + tail))
