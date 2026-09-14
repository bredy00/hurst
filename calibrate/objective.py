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
    by_expiry: list = field(default=None, repr=False)
    # Optional short-end skew anchors (calibrate.weights.SkewAnchor): one extra
    # residual row each, appended after the quotes. Session G.
    anchors: list = field(default_factory=list, repr=False)

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
        self.by_expiry = [(float(t), np.where(self.tau == t)[0])
                          for t in np.unique(self.tau)]

    def __len__(self):
        return len(self.tau)

    @property
    def n_expiries(self):
        return len(self.by_expiry)

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


# Failure classes returned by model_ivs(..., return_status=True)
OK, BELOW_RESOLUTION, MODEL_FAILURE = 0, 1, 2

# The implied-vol inversion searches [1e-4, 5.0]. A quote the model cannot price
# is charged as if the model had quoted the ceiling: no finite, priceable
# outcome can cost more, so failing is never a way to lower the objective.
FAILURE_IV = 5.0
RESOLUTION = 1e-14


def model_ivs(params, surface, cf_factory, pricer=fo.carr_madan_call,
              tol=CALIB_TOL, return_status=False):
    """
    Model implied vols at every quote on the surface.

    Returns (ivs, n_failed), or (ivs, n_failed, status) with return_status. A
    point whose price cannot be inverted comes back as NaN rather than a
    fabricated number, and `status` says why:

      BELOW_RESOLUTION  the model's time value is below 1e-14 -- far enough out
                        that the model value is effectively zero
      MODEL_FAILURE     the price is non-finite or outside the no-arbitrage band
                        (a broken solve, or a pricer that declined to answer)
    """
    out = np.full(len(surface), np.nan)
    status = np.zeros(len(surface), dtype=int)
    failed = 0
    for tau, idx in surface.by_expiry:
        cf = cf_factory(params, tau)
        ks = surface.k[idx]
        prices = np.atleast_1d(np.asarray(pricer(ks, tau, cf, tol=tol), dtype=float))
        # Whole strip at once. One-at-a-time inversion was measured at half the
        # cost of an entire objective evaluation, nearly all of it numpy call
        # overhead rather than arithmetic.
        vols = fo.implied_vols_from_calls(prices, ks, tau)
        bad = ~np.isfinite(vols)
        failed += int(np.count_nonzero(bad))
        intrinsic = np.maximum(1.0 - np.exp(ks), 0.0)
        below = bad & np.isfinite(prices) & (prices <= intrinsic + RESOLUTION) & (prices > intrinsic - 1e-9)
        st = np.where(bad, MODEL_FAILURE, OK)
        st[below] = BELOW_RESOLUTION
        status[idx] = st
        out[idx] = vols
    if return_status:
        return out, failed, status
    return out, failed


def residuals(params, surface, cf_factory, pricer=fo.carr_madan_call,
              tol=CALIB_TOL):
    """
    sqrt(w) * (model_iv - market_iv), the vector Levenberg-Marquardt minimises.

    The vector keeps a fixed length for the Jacobian, so unpriceable points
    must be given a value, and which value is not a detail. An earlier version
    gave every one of them exactly zero -- "no information" -- and a rough
    Heston fit used that: it walked to parameters where two whole maturities
    came back unpriceable, their residuals vanished, and the cost FELL, ending
    at H = 0.02 and theta = 0.88 with garbage prices of 1e18 at 30 and 90 days.

    Now:
      * the model says ~zero time value AND the market quote is itself below
        resolution: genuinely no information, 0;
      * the model says ~zero time value where the market prices real time value:
        the model quoted zero vol, residual sqrt(w) * (0 - market_iv);
      * the model failed (non-finite or out-of-band price): charged at
        FAILURE_IV, which no priceable outcome can exceed.
    """
    import volsurf_core as vc

    mv, failed, status = model_ivs(params, surface, cf_factory, pricer, tol,
                                   return_status=True)
    sw = np.sqrt(surface.weight)
    r = sw * (mv - surface.iv)
    if failed:
        below = status == BELOW_RESOLUTION
        if np.any(below):
            K = np.exp(surface.k[below])
            c_mkt = np.array([float(vc.bs_price(1.0, Ki, max(v, 1e-8), t, 1.0, 'C'))
                              for Ki, v, t in zip(K, surface.iv[below], surface.tau[below])])
            informative = c_mkt - np.maximum(1.0 - K, 0.0) > RESOLUTION
            rb = np.where(informative, sw[below] * (0.0 - surface.iv[below]), 0.0)
            r[below] = rb
        fail = status == MODEL_FAILURE
        r[fail] = sw[fail] * (FAILURE_IV - surface.iv[fail])
    # By here every model-side failure has a finite charge; anything still
    # non-finite comes from the MARKET row itself (a NaN quote or weight), which
    # carries no information.
    r[~np.isfinite(r)] = 0.0
    if surface.anchors:
        # Anchor rows read the same model vols the quote rows charged, including
        # the failure charge, so a failed strip cannot make its skew look right.
        with np.errstate(divide="ignore", invalid="ignore"):
            mv_eff = np.where(sw > 0, surface.iv + r / np.where(sw > 0, sw, 1.0), mv)
        mv_eff = np.where(np.isfinite(mv_eff), mv_eff, FAILURE_IV)
        extra = [np.sqrt(an.weight) * (float(an.a @ mv_eff[an.idx]) - an.target) / an.se
                 for an in surface.anchors]
        r = np.concatenate([r, np.asarray(extra, dtype=float)])
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
    for tau, idx in surface.by_expiry:
        d = mv[idx] - surface.iv[idx]
        d = d[np.isfinite(d)]
        if d.size:
            rows.append({'tau': tau, 'n': int(d.size),
                         'rmse': float(np.sqrt(np.mean(d * d))),
                         'bias': float(np.mean(d)),
                         'worst': float(np.max(np.abs(d)))})
    return rows
