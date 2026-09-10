"""
The calibration objective: model surface vs market surface, in implied vol.

Fitting in IMPLIED VOL rather than price is the whole design decision here. Price
errors are dominated by the at-the-money contracts, which are worth a hundred
times what a wing contract is worth; a price-space least squares therefore fits
the middle of the surface almost exclusively and is nearly blind to the skew --
which is the part we actually care about. In vol space every quote speaks at the
same scale, and the vega weighting then re-introduces the differences in how
reliably each one was measured.

Weights come from `volsurf_core.quote_weight` -- inverse variance of the implied
vol, 1/(half_spread/vega)^2 -- so this is the same notion of quote quality the
live surface already uses, not a second one invented for the fit.
"""

import math
from dataclasses import dataclass, field

import numpy as np

import pricing.fourier as fo

# Calibration does not need 1e-14 quadrature. Measured on a 30-day Heston slice:
# tol=1e-14 costs 6.76 ms, tol=1e-12 costs 3.23 ms, and the two agree to 2.8e-15.
# Five times faster than the original settings for no accuracy that matters.
CALIB_TOL = 1e-12


@dataclass
class MarketSurface:
    """
    A surface to fit: one row per quote.

    tau, k (log-moneyness vs that expiry's forward), iv, and weight. Flat arrays
    rather than nested slices, with an index by tau built once, because the fit
    walks the whole thing thousands of times.
    """
    tau: np.ndarray
    k: np.ndarray
    iv: np.ndarray
    weight: np.ndarray
    _groups: list = field(default=None, repr=False)

    def __post_init__(self):
        self.tau = np.asarray(self.tau, dtype=float)
        self.k = np.asarray(self.k, dtype=float)
        self.iv = np.asarray(self.iv, dtype=float)
        self.weight = np.asarray(self.weight, dtype=float)
        n = len(self.tau)
        if not (len(self.k) == len(self.iv) == len(self.weight) == n):
            raise ValueError("tau, k, iv and weight must be the same length")
        # One characteristic function evaluation serves a whole expiry, so group
        # once here rather than rediscovering the structure on every call.
        self._groups = [(float(t), np.where(self.tau == t)[0])
                        for t in np.unique(self.tau)]

    def __len__(self):
        return len(self.tau)

    @property
    def n_expiries(self):
        return len(self._groups)

    @classmethod
    def from_points(cls, points, ctxs):
        """Build from volatility_surface_3.surface_points() output."""
        tau, k, iv, w = [], [], [], []
        for exp, rows in points.items():
            for r in rows:
                tau.append(ctxs[exp].tau)
                k.append(r['k'])
                iv.append(r['iv'])
                w.append(r['weight'])
        return cls(np.array(tau), np.array(k), np.array(iv), np.array(w))


def model_ivs(params, surface, cf_factory, pricer=fo.carr_madan_call,
              tol=CALIB_TOL):
    """
    Model implied vols at every quote on the surface.

    Returns (ivs, n_failed). A point whose price cannot be inverted -- far enough
    out that the model value is below a tick -- comes back as NaN rather than a
    fabricated number, and the caller decides what to do with it.
    """
    out = np.full(len(surface), np.nan)
    failed = 0
    for tau, idx in surface._groups:
        cf = cf_factory(params, tau)
        ks = surface.k[idx]
        prices = np.atleast_1d(pricer(ks, tau, cf, tol=tol))
        # Whole strip at once. One-at-a-time inversion was measured at half the
        # cost of an entire objective evaluation, nearly all of it numpy call
        # overhead rather than arithmetic.
        vols = fo.implied_vols_from_calls(prices, ks, tau)
        failed += int(np.count_nonzero(~np.isfinite(vols)))
        out[idx] = vols
    return out, failed


def residuals(params, surface, cf_factory, pricer=fo.carr_madan_call,
              tol=CALIB_TOL):
    """
    sqrt(w) * (model_iv - market_iv), the vector Levenberg-Marquardt minimises.

    Points the model cannot price contribute exactly zero rather than being
    dropped: the residual vector has to keep a fixed length for the Jacobian to
    make sense, and a zero is the honest statement that this quote carried no
    information about the fit.
    """
    mv, failed = model_ivs(params, surface, cf_factory, pricer, tol)
    r = np.sqrt(surface.weight) * (mv - surface.iv)
    r[~np.isfinite(r)] = 0.0
    return r, failed


def loss(params, surface, cf_factory, pricer=fo.carr_madan_call, tol=CALIB_TOL):
    """Weighted sum of squared vol errors."""
    r, _ = residuals(params, surface, cf_factory, pricer, tol)
    return float(r @ r)


def rmse_vol(params, surface, cf_factory, pricer=fo.carr_madan_call,
             tol=CALIB_TOL):
    """
    Unweighted RMSE in vol points -- the number to quote to a human.

    The weighted loss is what the optimiser minimises, but it is in units nobody
    has intuition for. This is "the fit is off by N vol points on average".
    """
    mv, _ = model_ivs(params, surface, cf_factory, pricer, tol)
    d = mv - surface.iv
    d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d * d))) if d.size else float('nan')


def per_expiry_report(params, surface, cf_factory, pricer=fo.carr_madan_call,
                      tol=CALIB_TOL):
    """RMSE and worst error by expiry -- where the model is failing, not just how much."""
    mv, _ = model_ivs(params, surface, cf_factory, pricer, tol)
    rows = []
    for tau, idx in surface._groups:
        d = mv[idx] - surface.iv[idx]
        d = d[np.isfinite(d)]
        if d.size:
            rows.append({'tau': tau, 'n': int(d.size),
                         'rmse': float(np.sqrt(np.mean(d * d))),
                         'bias': float(np.mean(d)),
                         'worst': float(np.max(np.abs(d)))})
    return rows
