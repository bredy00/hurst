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

    KNOWN LIMITATION -- small xi. Both kappa*theta/xi^2 and (a-d)/xi^2 diverge as
    xi -> 0 while their combination stays finite, so the expression loses
    precision by cancellation. Against the exact Black-Scholes characteristic
    function the error is ~1e-10 at xi = 1e-4, 2e-07 at xi = 1e-5 and 3.5e-06 at
    xi = 1e-6. This never bites at calibration-realistic parameters (xi ~ 0.1 to
    1.0) but it does mean the xi -> 0 degeneracy test has to be run at a moderate
    xi, not an arbitrarily tiny one.
    """
    u = np.asarray(u, dtype=complex)
    v0, kappa, theta, xi, rho = p.as_tuple()

    iu = 1j * u
    a = kappa - rho * xi * iu
    d = np.sqrt(a * a + (xi * xi) * (u * u + iu))

    denom = a + d
    # a + d = 0 only in degenerate corners; fall back to the other root there
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
