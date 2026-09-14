"""
Quote weights for the implied-vol objective, and the short-end skew anchor
(Session G, the H weighting item of the analytics review).

What the Session E study actually showed, restated because it decides the
design. On the 20-quote test surface, vega^2 weights (a constant half-spread in
PRICE) left 2.3 effective quotes, and SE(H) was 0.20. Equal weights in vol gave
SE(H) = 0.047. corr(H, xi) was 0.98-0.99 under EVERY weighting -- 0.9776 is the
equal-weight value -- so the H-xi trade-off belongs to the surface design, and
re-weighting alone cannot remove it. Weights fix the standard error; only
information that separates the level of the skew (xi) from its decay rate in tau
(H) fixes the correlation.

Three schemes:

  vega2             w = (vega / h)^2 with one price half-spread h for every
                    quote: Session E's weights. Correct only if quote noise
                    really is a constant in price, which makes IV noise explode
                    in the short-dated wings -- so those quotes are ignored.
  inverse_variance  w = 1 / (sigma_iv^2 + floor^2), sigma_iv = half_spread / vega
                    from each quote's OWN spread. The Gauss-Markov weights when the
                    noise model is right. The floor stands for model error: without
                    it a long-dated ATM quote with a one-cent spread gets a weight
                    no model can honour.
  hybrid            inverse variance, re-balanced so every expiry carries the same
                    total weight, PLUS one residual per short expiry on its ATM
                    skew: (model skew - market skew) / SE(skew). The skew is the
                    observable H acts on (it scales like tau^(H - 1/2)), and an
                    anchor row makes the fit match it explicitly rather than as a
                    by-product of 9 quotes' levels.

An honest caveat, measured in study_h_weighting.py rather than assumed: when the
model is RIGHT and the weights match the noise, inverse variance already attains
the Cramer-Rao bound, and an anchor built from the same quotes cannot add
information -- it can only move weight around. The case for the hybrid is
misspecification: a real surface the model cannot fit everywhere, where vega^2
or even inverse-variance weights let the long end decide H.
"""

import math
from dataclasses import dataclass

import numpy as np

import volsurf_core as vc

IV_FLOOR = 0.001            # 0.1 vol point of model error
SHORT_END = 30.0 / 365.0    # anchors on expiries up to a month


def realistic_half_spread(price, tick=0.005 / 650.0, proportional=0.025):
    """
    Half-spread in units of the forward: the larger of half a one-cent tick on a
    ~650 underlying and 2.5% of the option's value. A stand-in for a real book,
    for studies only; recorded chains carry their own bid and ask.
    """
    return np.maximum(tick, proportional * np.asarray(price, float))


def iv_noise_sd(half_spread, vega, floor=0.0):
    """sqrt((half_spread / vega)^2 + floor^2)."""
    s = np.asarray(vc.iv_uncertainty(half_spread, vega), float)
    return np.sqrt(s * s + floor * floor)


def vega2_weights(vega, half_spread_const=0.005):
    return (np.asarray(vega, float) / half_spread_const) ** 2


def inverse_variance_weights(half_spread, vega, floor=IV_FLOOR):
    s = iv_noise_sd(half_spread, vega, floor)
    return np.where(np.isfinite(s) & (s > 0), 1.0 / (s * s), 0.0)


def balance_by_expiry(tau, w):
    """Rescale so each expiry's weights sum to the same total; the grand total is kept."""
    tau = np.asarray(tau, float)
    w = np.asarray(w, float).copy()
    taus = np.unique(tau)
    total = float(w.sum())
    for t in taus:
        idx = tau == t
        s = float(w[idx].sum())
        if s > 0:
            w[idx] *= (total / len(taus)) / s
    return w


def maturity_taper(tau, short_end=SHORT_END, power=1.0):
    """1 up to `short_end`, then (short_end / tau)^power: the long end still counts, less."""
    tau = np.asarray(tau, float)
    return np.where(tau <= short_end, 1.0, (short_end / np.maximum(tau, 1e-12)) ** power)


def effective_quotes(w):
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w * w).sum()) if np.any(w > 0) else 0.0


def skew_functional(k, z, quote_weights=None, window=1.5, degree=2):
    """
    The vector a with  skew = a @ iv  for the local ATM skew estimator of
    volsurf_core.atm_skew_from_slice (tricube kernel in z times quote weight,
    weighted polynomial fit in k, linear coefficient). Because the estimator is
    linear in the vols, the SAME a turns model vols and market vols into skews,
    so any bias of the estimator itself cancels in the residual.
    Returns (a, used_mask); a is None when the fit cannot be formed.
    """
    k = np.asarray(k, float)
    z = np.asarray(z, float)
    qw = np.ones_like(k) if quote_weights is None else np.asarray(quote_weights, float)
    wts = qw * vc.tricube(z / float(window))
    used = wts > 0
    if int(used.sum()) < degree + 2:
        return None, used
    X = np.vander(k[used], degree + 1)            # columns k^degree ... k^0, as polyfit
    Wd = wts[used]
    G = X.T @ (Wd[:, None] * X)
    coef_map = np.linalg.solve(G, X.T * Wd[None, :])   # (degree+1, n_used)
    a = np.zeros_like(k)
    a[used] = coef_map[-2]                          # d/dk at k = 0
    return a, used


@dataclass
class SkewAnchor:
    """One extra residual row: sqrt(weight) * (a @ model_iv[idx] - target) / se."""
    idx: np.ndarray
    a: np.ndarray
    target: float
    se: float
    weight: float
    tau: float


def build_anchors(tau, k, iv, noise_sd, short_end=SHORT_END, window=1.5, weight=None):
    """
    ATM-skew anchors for every expiry with tau <= short_end.

    z uses each expiry's own ATM vol (the quote nearest k = 0). SE(skew) is
    sqrt(sum_i a_i^2 sigma_i^2) from the same noise model the weights use.
    `weight` defaults to the number of quotes in that expiry: one anchor then
    counts as much as the whole slice would under unit weights.
    """
    tau, k, iv, noise_sd = (np.asarray(x, float) for x in (tau, k, iv, noise_sd))
    anchors = []
    for t in np.unique(tau):
        if t > short_end:
            continue
        idx = np.where(tau == t)[0]
        atm = idx[np.argmin(np.abs(k[idx]))]
        z = k[idx] / (iv[atm] * math.sqrt(t))
        inv_var = np.where(noise_sd[idx] > 0, 1.0 / noise_sd[idx] ** 2, 0.0)
        a_loc, _ = skew_functional(k[idx], z, inv_var, window)
        if a_loc is None:
            continue
        se = float(math.sqrt(np.sum(a_loc ** 2 * noise_sd[idx] ** 2)))
        if not (se > 0):
            continue
        anchors.append(SkewAnchor(idx=idx, a=a_loc, target=float(a_loc @ iv[idx]), se=se,
                                  weight=float(len(idx) if weight is None else weight), tau=float(t)))
    return anchors


def residual_map(n, w, anchors=()):
    """
    The matrix M with  residuals = M @ (model_iv - market_iv):  diag(sqrt w) on
    top, one row per anchor below. The fit's effective weight matrix is M'M,
    which is what the Gauss-Newton study needs.
    """
    rows = [np.diag(np.sqrt(np.asarray(w, float)))]
    for an in anchors:
        r = np.zeros(n)
        r[an.idx] = math.sqrt(an.weight) * an.a / an.se
        rows.append(r[None, :])
    return np.vstack(rows)


def scheme_weights(scheme, tau, k, iv, vega, half_spread, noise_sd=None, floor=IV_FLOOR,
                   short_end=SHORT_END, anchor_weight=None):
    """(w, anchors) for 'vega2', 'inverse_variance' or 'hybrid'."""
    if scheme == "vega2":
        return vega2_weights(vega), []
    w = inverse_variance_weights(half_spread, vega, floor)
    if scheme == "inverse_variance":
        return w, []
    if scheme != "hybrid":
        raise ValueError(f"unknown weighting scheme {scheme!r}")
    sd = iv_noise_sd(half_spread, vega, floor) if noise_sd is None else noise_sd
    return balance_by_expiry(tau, w), build_anchors(tau, k, iv, sd, short_end, weight=anchor_weight)
