"""
The Heston (1993) stochastic volatility model.

    dS_t / S_t = sqrt(v_t) dW1
    dv_t       = kappa (theta - v_t) dt + xi sqrt(v_t) dW2
    d<W1,W2>   = rho dt

Everything here is expressed in LOG-MONEYNESS X = log(S_tau / F), i.e. under the
forward measure with the carry already inside F. That makes the martingale
condition phi(-i) = 1 rather than phi(-i) = F, keeps the numbers O(1) instead of
O(650), and matches the coordinate the surface already uses.

Numpy only. scipy is not needed for any of this and stays out.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HestonParams:
    v0: float = 0.04         # initial variance
    kappa: float = 2.0       # mean reversion speed
    theta: float = 0.045     # long-run variance
    xi: float = 0.5          # vol of vol
    rho: float = -0.7        # spot/vol correlation

    NAMES = ("v0", "kappa", "theta", "xi", "rho")

    def as_tuple(self):
        return (self.v0, self.kappa, self.theta, self.xi, self.rho)

    @property
    def feller(self):
        """
        2 kappa theta > xi^2. Reported, never enforced: real calibrations violate
        it routinely, and a fit that silently clamps to satisfy it is lying about
        what the market is saying.
        """
        return 2.0 * self.kappa * self.theta > self.xi ** 2

    def valid(self):
        return (self.v0 > 0 and self.kappa > 0 and self.theta > 0
                and self.xi > 0 and abs(self.rho) < 1)


XI_TAYLOR = 1e-4     # below this vol-of-vol the log term uses its Taylor series


def _clog1p(z):
    """
    log(1 + z) for complex z, accurate when |z| is small.

    numpy's complex log1p is NOT: its ufunc loop computes log(hypot(1+x, y)),
    which throws away every digit of a small z. The real-part identity
    log|1+z| = 0.5 log1p(2x + x^2 + y^2) keeps them, and atan2 handles the angle.
    """
    z = np.asarray(z, dtype=complex)
    x, y = z.real, z.imag
    return 0.5 * np.log1p(2.0 * x + x * x + y * y) + 1j * np.arctan2(y, 1.0 + x)


def char_func(u, tau, p):
    """
    phi(u) = E[ exp(i u X_tau) ],  X = log(S_tau / F).

    Uses the Albrecher "little Heston trap" formulation:

        a  = kappa - rho xi i u
        d  = sqrt(a^2 + xi^2 (u^2 + i u))
        g  = (a - d) / (a + d)                 <-- NOT its reciprocal

        C  = (kappa theta / xi^2) [ (a-d) tau - 2 log((1 - g e^{-d tau})/(1 - g)) ]
        D  = ((a-d) / xi^2) (1 - e^{-d tau}) / (1 - g e^{-d tau})
        phi = exp(C + D v0)

    The choice of g is the whole game. The original 1993 paper uses 1/g, which
    sends the argument of the complex logarithm across its branch cut and makes
    phi jump discontinuously in tau. Measured on (kappa=0.5, xi=0.9, rho=-0.7) at
    u = 8, the largest step between neighbouring tau on a 4000-point grid is
    9.5e-04 for the form above and 2.04e-01 for the reciprocal -- a 215x
    difference, and visible as price jumps past about a year. `char_func_trap`
    below keeps the broken version so the test suite can show this rather than
    assert it on authority.

    SMALL xi, and why the formulas above are not evaluated literally. Written as
    printed, kappa*theta/xi^2 and (a-d)/xi^2 both diverge as xi -> 0 while their
    combination stays finite, and the cancellation costs precision: measured
    against the exact Black-Scholes cf the literal form is off by 5.9e-09 at
    xi = 1e-4, 3.4e-05 at 1e-6 and is 100% wrong by 1e-10 (`char_func_naive`,
    kept below for the health check). Two identities remove it entirely:

        (a - d)(a + d) = -xi^2 (u^2 + iu)   =>   (a-d)/xi^2 = -(u^2+iu)/(a+d)

    which is exact and has no cancellation, and for the log term

        log((1 - g e^{-d tau})/(1 - g)) = log1p( g (1 - e^{-d tau}) / (1 - g) )

    a single log1p of an O(xi^2) quantity, with 1 - e^{-d tau} from expm1. Its
    Taylor series in xi^2 -- log1p(z)/xi^2 = zeta (1 - z/2 + z^2/3 - z^3/4),
    z = xi^2 zeta -- is used below XI_TAYLOR = 1e-4, where four terms are exact
    to machine precision; above it an accurate complex log1p is used. The two
    branches agree to 3e-16 across the switch, and the result matches the exact
    Black-Scholes cf to 2e-16 at xi = 1e-8, 1e-10 and 1e-12. At xi = 0 exactly the
    deterministic-variance limit is returned.
    """
    u = np.asarray(u, dtype=complex)
    v0, kappa, theta, xi, rho = p.as_tuple()

    iu = 1j * u
    b = u * u + iu                              # u^2 + i u

    if xi == 0.0:
        # dv = kappa (theta - v) dt exactly: integrated variance is deterministic
        relax = -math.expm1(-kappa * tau) / kappa if kappa > 0 else tau
        V = v0 * relax + theta * (tau - relax)
        return np.exp(-0.5 * b * V)

    a = kappa - rho * xi * iu
    d = np.sqrt(a * a + (xi * xi) * b)

    apd = a + d
    # a + d = 0 only in degenerate corners; fall back to the other root there
    bad = np.abs(apd) < 1e-300
    if np.any(bad):
        d = np.where(bad, -d, d)
        apd = a + d

    amd_xi2 = -b / apd                          # (a - d) / xi^2, exactly
    gam = amd_xi2 / apd                         # g / xi^2
    one_m_e = -np.expm1(-d * tau)               # 1 - e^{-d tau}
    zeta = gam * one_m_e / (1.0 - (xi * xi) * gam)
    z = (xi * xi) * zeta
    if xi < XI_TAYLOR:
        log_term_xi2 = zeta * (1.0 - z / 2.0 + z * z / 3.0 - z * z * z / 4.0)
    else:
        log_term_xi2 = _clog1p(z) / (xi * xi)

    C = kappa * theta * (amd_xi2 * tau - 2.0 * log_term_xi2)
    D = amd_xi2 * one_m_e / (1.0 - (xi * xi) * gam * (1.0 - one_m_e))
    return np.exp(C + D * v0)


def char_func_naive(u, tau, p):
    """
    The literal textbook evaluation, kept ONLY so the cancellation as xi -> 0 can
    be measured rather than described. Identical to `char_func` above 1e-3 or
    so; do not price with it.
    """
    u = np.asarray(u, dtype=complex)
    v0, kappa, theta, xi, rho = p.as_tuple()

    iu = 1j * u
    a = kappa - rho * xi * iu
    d = np.sqrt(a * a + (xi * xi) * (u * u + iu))

    denom = a + d
    bad = np.abs(denom) < 1e-300
    if np.any(bad):
        d = np.where(bad, -d, d)
        denom = a + d

    g = (a - d) / denom
    edt = np.exp(-d * tau)
    one_m_ge = 1.0 - g * edt

    C = (kappa * theta / (xi * xi)) * ((a - d) * tau
                                       - 2.0 * np.log(one_m_ge / (1.0 - g)))
    D = ((a - d) / (xi * xi)) * (1.0 - edt) / one_m_ge
    return np.exp(C + D * v0)


def char_func_trap(u, tau, p):
    """
    The WRONG formulation, kept deliberately: g = (a + d)/(a - d), the reciprocal.

    This exists so the test suite can demonstrate that the branch-cut problem is
    real and that the shipped version does not have it, rather than asserting it
    on authority. Do not price with this.
    """
    u = np.asarray(u, dtype=complex)
    v0, kappa, theta, xi, rho = p.as_tuple()
    iu = 1j * u
    a = kappa - rho * xi * iu
    d = np.sqrt(a * a + (xi * xi) * (u * u + iu))
    g = (a + d) / (a - d)
    edt = np.exp(d * tau)
    C = (kappa * theta / (xi * xi)) * ((a + d) * tau
                                       - 2.0 * np.log((1.0 - g * edt) / (1.0 - g)))
    D = ((a + d) / (xi * xi)) * (1.0 - edt) / (1.0 - g * edt)
    return np.exp(C + D * v0)


def variance_swap_rate(p, tau):
    """
    Fair variance strike E[(1/tau) int_0^tau v_t dt] under Heston:

        theta + (v0 - theta) (1 - e^{-kappa tau}) / (kappa tau)

    exact, because the variance drift is affine. Useful two ways: as a check on
    a fit (the model's variance-swap curve must lie near the market's), and as
    the cleanest external handle on kappa when the option surface cannot
    determine it -- the curve's approach to theta is governed by kappa alone.
    """
    if p.kappa * tau < 1e-12:
        return p.v0
    relax = -math.expm1(-p.kappa * tau) / (p.kappa * tau)
    return p.theta + (p.v0 - p.theta) * relax


def kappa_from_variance_swap(v0, theta, tau, rate, lo=1e-4, hi=1e3):
    """
    Invert variance_swap_rate for kappa by bisection, given v0 and theta.

    The rate is monotone in kappa whenever v0 != theta (it moves from v0 at
    kappa = 0 towards theta as kappa grows), so the root is unique when it
    exists. Returns None if `rate` is not between v0 and theta -- no kappa can
    produce it, and inventing one would be worse than saying so. Feed the
    result to `calibrate(..., prior={'kappa': k}, prior_weight=...)`.
    """
    if abs(v0 - theta) < 1e-14:
        return None
    if not (min(v0, theta) - 1e-14 <= rate <= max(v0, theta) + 1e-14):
        return None

    def f(k):
        return variance_swap_rate(HestonParams(v0, k, theta, 1e-3, 0.0), tau) - rate

    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return lo if abs(flo) < abs(fhi) else hi
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        fm = f(mid)
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
        if hi / lo < 1.0 + 1e-12:
            break
    return math.sqrt(lo * hi)


def simulate(p, tau, n_paths=200_000, n_steps=250, seed=0, antithetic=True):
    """
    Full-truncation Euler simulation of X = log(S/F), for validating the pricer.

    Full truncation (Lord et al.) is the standard fix for the fact that Euler
    on the variance process goes negative: the variance is floored at zero
    wherever it appears, but the state itself is allowed to go negative and
    recover. Plain reflection biases the variance upward.
    """
    rng = np.random.default_rng(seed)
    v0, kappa, theta, xi, rho = p.as_tuple()
    dt = tau / n_steps
    sdt = math.sqrt(dt)

    m = n_paths // 2 if antithetic else n_paths
    z1 = rng.standard_normal((n_steps, m))
    z2 = rng.standard_normal((n_steps, m))
    if antithetic:
        z1 = np.concatenate([z1, -z1], axis=1)
        z2 = np.concatenate([z2, -z2], axis=1)

    n = z1.shape[1]
    x = np.zeros(n)
    v = np.full(n, v0)
    for i in range(n_steps):
        vp = np.maximum(v, 0.0)
        sv = np.sqrt(vp)
        dw1 = z1[i] * sdt
        dw2 = (rho * z1[i] + math.sqrt(1.0 - rho * rho) * z2[i]) * sdt
        x += -0.5 * vp * dt + sv * dw1
        v += kappa * (theta - vp) * dt + xi * sv * dw2
    return x


def mc_call(p, tau, k, n_paths=200_000, n_steps=250, seed=0):
    """
    Undiscounted forward call value in units of F, plus its standard error.
    k is log-moneyness log(K/F).
    """
    x = simulate(p, tau, n_paths, n_steps, seed)
    payoff = np.maximum(np.exp(x) - math.exp(k), 0.0)
    return float(payoff.mean()), float(payoff.std(ddof=1) / math.sqrt(len(payoff)))
