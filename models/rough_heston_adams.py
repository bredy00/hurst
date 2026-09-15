"""
The TRUE rough Heston characteristic function, with no lift: a reference.

El Euch & Rosenbaum (2019): with K(t) = t^(alpha-1) / Gamma(alpha), alpha = H + 1/2,

    psi(t) = int_0^t K(t - s) F(psi(s)) ds                (fractional Riccati)
    F(x)   = -1/2 (u^2 + i u) + (i u rho xi - kappa) x + 1/2 xi^2 x^2
    log cf(u, T) = int_0^T [ kappa theta psi(s) + v0 F(psi(s)) ] ds

(the same parametrisation as models.rough_heston; the lifted Riccati system
converges to this as N -> infinity). Solved with the fractional Adams
predictor-corrector of Diethelm, Ford & Freed (2002), on a uniform grid:

  predictor  psi^P_{k+1} = h^a / Gamma(a+1) sum_{j<=k} ((k+1-j)^a - (k-j)^a) F_j
  corrector  psi_{k+1}   = h^a / Gamma(a+2) [ sum_{j<=k} a_{j,k+1} F_j + F(psi^P_{k+1}) ]
             a_{0,k+1} = k^(a+1) - (k-a)(k+1)^a
             a_{j,k+1} = (k-j+2)^(a+1) + (k-j)^(a+1) - 2 (k-j+1)^(a+1),  1 <= j <= k

It costs O(M^2) per maturity -- 2000 steps is ~2x10^6 weighted sums -- so it is a
VALIDATION tool for the lift (study_lewis_isometry.py, Session G), never a pricer
for calibration. Its own error is measured by doubling M.
"""

import math

import numpy as np


def adams_log_cf(u, T, p, M=2000):
    """log cf(u, T) of rough Heston by fractional Adams with M uniform steps."""
    u = np.atleast_1d(np.asarray(u, dtype=complex))
    a = p.H + 0.5
    h = T / M
    iu = 1j * u
    b0 = -0.5 * (u * u + iu)
    lin = iu * p.rho * p.xi - p.kappa
    q = 0.5 * p.xi * p.xi

    def F(x):
        return b0 + lin * x + q * x * x

    c_pred = h ** a / math.gamma(a + 1.0)
    c_corr = h ** a / math.gamma(a + 2.0)
    psi = np.zeros((M + 1, len(u)), dtype=complex)
    Fv = np.zeros((M + 1, len(u)), dtype=complex)
    Fv[0] = F(psi[0])
    for k in range(M):
        j = np.arange(k + 1, dtype=float)
        bw = (k + 1 - j) ** a - (k - j) ** a
        pred = c_pred * (bw @ Fv[:k + 1])
        aw = (k - j + 2) ** (a + 1) + (k - j) ** (a + 1) - 2.0 * (k - j + 1) ** (a + 1)
        aw[0] = k ** (a + 1) - (k - a) * (k + 1) ** a
        psi[k + 1] = c_corr * (aw @ Fv[:k + 1] + F(pred))
        Fv[k + 1] = F(psi[k + 1])
    integrand = p.kappa * p.theta * psi + p.v0 * Fv
    return h * (0.5 * integrand[0] + integrand[1:-1].sum(axis=0) + 0.5 * integrand[-1])


def lewis_prices_from_logcf(ks, logcf_on, u_max, k_abs_max=None):
    """
    Undiscounted call prices (units of F) on the Lewis contour from a function
    returning log cf at given complex u. Same quadrature grid as rough_heston's
    Lewis pricer, so two cfs priced through it differ only by their cfs.
    """
    import models.rough_heston as rh
    import pricing.fourier as fo
    ks = np.atleast_1d(np.asarray(ks, float))
    kmax = float(np.max(np.abs(ks))) if k_abs_max is None else k_abs_max
    v, w = rh.lewis_grid(u_max, kmax)
    phi = rh._safe_exp(logcf_on(v - 0.5j))
    dens = phi / (v * v + 0.25)
    return 1.0 - np.exp(0.5 * ks) / math.pi * fo._real_transform(ks, v, dens, w)
