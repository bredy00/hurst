"""
Fourier option pricing from a characteristic function.

Two independent routes, deliberately:

  Lewis       one integral, no damping parameter to choose. This is the default
              because it needs no alpha, but note it is the FUSSIER of the two
              numerically: its integrand carries no 1/v^2 damping, so it needs
              more quadrature to reach the same accuracy at short maturities.
  Carr-Madan  the classic damped transform, plus an FFT variant that produces a
              whole strip of strikes in one pass -- which is what a calibrator
              wants, since it prices a full slice per objective evaluation. The
              damping denominator decays like 1/v^2 on top of the transform's own
              decay, which makes it markedly better conditioned: measured against
              the exact Black-Scholes cf it reaches 2.6e-16 where Lewis, at the
              same automatic settings, reached 1.6e-07 at one day.

They share nothing but the characteristic function, so agreement between them is
real evidence rather than a restatement. test_pricing.py requires 1e-8.

Conventions throughout: X = log(S_tau / F), k = log(K / F), and every function
returns the UNDISCOUNTED call value in units of the forward. The actual price is
`df * F * value`. Keeping the normalisation out of the quadrature means the
integrands are O(1) whatever the underlying is worth.

Numpy only.
"""

import functools
import math

import numpy as np


PANEL_ORDER = 64


@functools.lru_cache(maxsize=8)
def _leggauss(n):
    """
    Cached Gauss-Legendre rule on [-1, 1].

    np.polynomial.legendre.leggauss solves an eigenvalue problem and costs O(n^2):
    n = 4000 measured at 8.4 SECONDS on this machine. A calibrator prices a whole
    surface per objective evaluation, thousands of times, so this must be cached --
    and the order must stay small, which is why the quadrature below is composite
    rather than one enormous rule.
    """
    return np.polynomial.legendre.leggauss(n)


def _composite_gl(a, b, n_panels, order=PANEL_ORDER):
    """
    Composite Gauss-Legendre: `n_panels` panels of `order` points each.

    Resolution then scales linearly in the number of panels at no construction
    cost, since only the one small rule is ever built. A single high-order rule
    cannot do this -- doubling its accuracy quadruples the time to build it.
    """
    x, w = _leggauss(int(order))
    edges = np.linspace(a, b, int(n_panels) + 1)
    lo, hi = edges[:-1], edges[1:]
    half = 0.5 * (hi - lo)
    nodes = (half[:, None] * (x[None, :] + 1.0) + lo[:, None]).ravel()
    weights = (half[:, None] * w[None, :]).ravel()
    return nodes, weights


def _auto_u_max(cf, shift, tol=1e-14, u_start=25.0, u_cap=1.0e5):
    """
    Find where the transform integrand has decayed below `tol`.

    A fixed truncation cannot serve every maturity. |phi(u - i/2)|/(u^2+1/4) for
    a typical Heston parameter set is 7e-21 at u = 200 for a one-year option but
    still 2.9e-06 at u = 200 for a one-day option -- truncating there loses real
    mass at the short end, which is exactly where the surface is most interesting.
    Decay is super-exponential once it starts, so doubling until it is negligible
    costs only a handful of evaluations.
    """
    u = float(u_start)
    while u < u_cap:
        probe = np.array([u * 0.75, u])
        val = float(np.max(np.abs(cf(probe + shift)) / (probe * probe)))
        if val < tol:
            return u
        u *= 2.0
    return u_cap


def _auto_panels(u_max, k_abs_max, order=PANEL_ORDER, lo=16, hi=8000):
    """
    Panels needed over [0, u_max]. Two separate demands, added.

    The oscillatory one is obvious: e^{-i u k} turns over u_max*|k|/(2 pi) times,
    so deep strikes need more nodes.

    The other is easy to under-provision and was. At a SHORT maturity the
    transform decays like exp(-sigma^2 tau u^2 / 2) with a tiny coefficient, so
    it stays near flat across a wide range and then falls off a cliff; that cliff
    needs resolving wherever it sits. Sizing only from the oscillation gave 46
    panels at tau = 1 day where 200 were needed -- 1.7e-07 error against
    Black-76, versus 7.3e-15 once resolved. Ten nodes per unit of u covers it,
    and the cost is linear because the rule is composite.
    """
    n_nodes = 10.0 * u_max + 8.0 * u_max * abs(k_abs_max)
    return int(min(max(math.ceil(n_nodes / order), lo), hi))


def lewis_call(k, tau, cf, u_max=None, n_panels=None, tol=1e-14):
    """
    Lewis (2001):

        c(k) = 1 - (e^{k/2} / pi) * Int_0^inf Re[ e^{-i u k} phi(u - i/2) ] / (u^2 + 1/4) du

    `cf` is a callable cf(u) -> phi(u) accepting a complex array. The 1/(u^2+1/4)
    factor gives the integrand a head start on decay, which is why this needs no
    damping parameter.

    Truncation and resolution are chosen automatically unless given: a fixed pair
    silently fails at one day (too little range) and deep in the money (too little
    resolution), and both are regimes the surface actually contains.
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    if u_max is None:
        u_max = _auto_u_max(cf, -0.5j, tol)
    if n_panels is None:
        n_panels = _auto_panels(u_max, float(np.max(np.abs(k))))

    u, w = _composite_gl(0.0, u_max, n_panels)
    dens = cf(u - 0.5j) / (u * u + 0.25)
    integrand = np.real(np.exp(-1j * np.outer(k, u)) * dens[None, :])
    out = 1.0 - np.exp(0.5 * k) / math.pi * (integrand @ w)
    return out if out.size > 1 else float(out[0])


def carr_madan_call(k, tau, cf, alpha=1.5, u_max=None, n_panels=None, tol=1e-14):
    """
    Carr-Madan (1999), evaluated directly rather than by FFT:

        c(k) = e^{-alpha k} / pi * Int_0^inf Re[ e^{-i v k} psi(v) ] dv
        psi(v) = phi(v - (alpha+1) i) / (alpha^2 + alpha - v^2 + i (2 alpha + 1) v)

    alpha damps the non-integrable payoff. It must satisfy E[S^{1+alpha}] < inf;
    1.5 is safe for Heston at ordinary parameters, and test_pricing.py checks the
    price is insensitive to it across 0.75 to 3.0 -- if it is not, the moment
    condition is being violated and the answer is not to be trusted.
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    if u_max is None:
        u_max = _auto_u_max(cf, -(alpha + 1.0) * 1j, tol)
    if n_panels is None:
        n_panels = _auto_panels(u_max, float(np.max(np.abs(k))))

    v, w = _composite_gl(0.0, u_max, n_panels)
    denom = alpha * alpha + alpha - v * v + 1j * (2.0 * alpha + 1.0) * v
    psi = cf(v - (alpha + 1.0) * 1j) / denom
    integrand = np.real(np.exp(-1j * np.outer(k, v)) * psi[None, :])
    out = np.exp(-alpha * k) / math.pi * (integrand @ w)
    return out if out.size > 1 else float(out[0])


def carr_madan_fft(tau, cf, alpha=1.5, n=4096, eta=0.25):
    """
    The FFT form: one transform yields a strip of log-strikes.

    Returns (k_grid, call_values). Grid spacing is lambda = 2 pi / (n eta), so a
    finer strike grid needs a smaller eta, and a wider one needs a larger n. The
    Simpson weights recover the order the trapezoid rule would otherwise lose.

    This is the form a calibrator wants: an objective evaluation needs a whole
    slice, and paying one FFT for it instead of one integral per strike is the
    difference between a fit that takes seconds and one that takes minutes.
    """
    v = np.arange(n) * eta
    lam = 2.0 * math.pi / (n * eta)
    b = 0.5 * n * lam
    k = -b + lam * np.arange(n)

    phi = cf(v - (alpha + 1.0) * 1j)
    denom = alpha * alpha + alpha - v * v + 1j * (2.0 * alpha + 1.0) * v
    psi = phi / denom

    # Simpson: 1, 4, 2, 4, ..., 4, 1 over 3
    simpson = np.ones(n)
    simpson[1:-1:2] = 4.0
    simpson[2:-1:2] = 2.0
    simpson *= eta / 3.0

    x = np.exp(1j * b * v) * psi * simpson
    y = np.fft.fft(x)
    calls = np.real(np.exp(-alpha * k) / math.pi * y)
    return k, calls


def implied_vol_from_call(c, k, tau, lo=1e-4, hi=5.0, tol=1e-12, maxiter=100):
    """
    Invert a normalised (F = 1, undiscounted) call value for Black-76 vol.

    Reuses the same Newton-on-vega-with-bisection-guard as volsurf_core rather
    than pulling in a second solver, and returns None outside the no-arbitrage
    bounds instead of a fabricated number.
    """
    import volsurf_core as vc

    K = math.exp(k)
    intrinsic = max(1.0 - K, 0.0)
    if not np.isfinite(c) or c <= intrinsic + 1e-14 or c >= 1.0:
        return None

    def f(s):
        return float(vc.bs_price(1.0, K, s, tau, 1.0, 'C')) - c

    if f(lo) > 0.0 or f(hi) < 0.0:
        return None
    a, b = lo, hi
    s = min(max(0.2, lo), hi)
    for _ in range(maxiter):
        fs = f(s)
        if abs(fs) < tol:
            return float(s)
        if fs > 0.0:
            b = s
        else:
            a = s
        if b - a < 1e-14:
            return float(0.5 * (a + b))
        vega = float(vc.bs_vega(1.0, K, s, tau, 1.0))
        step = s - fs / vega if vega > 1e-14 else None
        s = step if (step is not None and a < step < b) else 0.5 * (a + b)
    return float(s)


def smile(ks, tau, cf, pricer=lewis_call, **kw):
    """Implied vols across a strip of log-moneyness. None where inversion fails."""
    cs = np.atleast_1d(pricer(ks, tau, cf, **kw))
    return [implied_vol_from_call(float(c), float(k), tau) for c, k in zip(cs, np.atleast_1d(ks))]
