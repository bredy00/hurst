"""
volsurf_core -- the pure maths. No IO, no network, no matplotlib.

This is the seed of the layered package: everything here is a function of its
arguments, so it can be tested against known answers with TWS switched off.

Conventions
-----------
tau         time increment T - t0 in years, never absolute time. The surface is
            a function of how far ahead we are looking, not of the calendar.
F           the forward for that expiry. When you pass F, the carry is already
            inside it, so `drift` is 0. When you pass spot instead, drift = r-q.
k           log-moneyness ln(K/F).
z           k normalised by the ATM move, k / (sigma_atm * sqrt(tau)). This is
            the coordinate the grid should be built on: it is the number of
            standard deviations, so it means the same thing at every expiry.
w           total variance sigma^2 * tau. This is the quantity that must be
            monotone in tau (no calendar arbitrage) and convex in k (no
            butterfly arbitrage), which sigma itself is not.
"""

import datetime
import math

import numpy as np

SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0
SQRT_2PI = math.sqrt(2.0 * math.pi)
SQRT_2 = math.sqrt(2.0)

# scipy is deliberately NOT imported here. `import scipy.stats` alone costs
# ~5.8s on this machine, and this module is on the startup path of a tool whose
# most common outcome is "TWS is not running". The two functions we need are
# three lines each, and test_core.py checks them against scipy to 1e-15.
_erf = np.vectorize(math.erf, otypes=[float])


def norm_pdf(x):
    # Scalar fast path: np.vectorize carries a fixed per-CALL overhead that
    # dominates when the argument is a single number, and the implied-vol solver
    # calls this one value at a time thousands of times per calibration.
    if isinstance(x, (int, float)):
        return math.exp(-0.5 * x * x) / SQRT_2PI
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / SQRT_2PI


def norm_cdf(x):
    if isinstance(x, (int, float)):
        return 0.5 * (1.0 + math.erf(x / SQRT_2))
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + _erf(x / SQRT_2))


# --- time -------------------------------------------------------------------
def us_dst(day):
    """
    Is US daylight saving in force on this date?

    Second Sunday of March to first Sunday of November. The switch happens at
    02:00 local, so for anything at the 16:00 close the calendar date decides it
    exactly. Written out rather than read from zoneinfo so this module stays
    free of a timezone database (test_core checks it against zoneinfo).
    """
    def nth_sunday(month, n):
        first = datetime.date(day.year, month, 1)
        return first + datetime.timedelta(days=(6 - first.weekday()) % 7 + 7 * (n - 1))
    return nth_sunday(3, 2) <= day < nth_sunday(11, 1)


def us_close_utc(day):
    """The 16:00 New York close of `day`, as a naive UTC datetime."""
    return datetime.datetime.combine(day, datetime.time(20 if us_dst(day) else 21, 0))


def _as_naive_utc(t):
    """Aware datetimes are converted to UTC. Naive ones are taken to BE UTC."""
    if t.tzinfo is not None:
        t = t.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return t


def tau_years(t0, expiry, min_tau=1e-6):
    """
    Time increment T - t0 in years, ACT/365.

    t0 and expiry may be date or datetime. A date expiry is taken at the 16:00
    New York close in UTC -- 20:00 in summer, 21:00 in winter -- so an expiry
    later today is a real fraction of a day rather than zero.

    Naive datetimes mean UTC. Pass `datetime.datetime.now(datetime.timezone.utc)`,
    never a bare `now()`: that is local wall-clock time, and on a UTC+3 machine
    it made every live tau three hours short (Session G; see the tutorial).
    """
    if isinstance(expiry, datetime.datetime):
        t_exp = _as_naive_utc(expiry)
    else:
        t_exp = us_close_utc(expiry)
    if isinstance(t0, datetime.datetime):
        t_now = _as_naive_utc(t0)
    else:
        t_now = datetime.datetime.combine(t0, datetime.time(14, 30))
    return max((t_exp - t_now).total_seconds() / SECONDS_PER_YEAR, min_tau)


def new_york_date(t):
    """The New York calendar date of an instant (aware, or naive meaning UTC)."""
    u = _as_naive_utc(t)
    # 04:00 UTC is midnight in New York in both seasons' worst case; use the
    # UTC date's DST flag, which is wrong only between midnight and 02:00 local
    # on the two switch days -- not a time anything here is recorded.
    offset = 4 if us_dst(u.date()) else 5
    return (u - datetime.timedelta(hours=offset)).date()


def parse_ib_date(yyyymmdd):
    return datetime.date(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]))


# --- Black-Scholes ----------------------------------------------------------
def d1_d2(F, K, sigma, tau, drift=0.0):
    """
    d1 = (ln(F/K) + (drift + sigma^2/2) * tau) / (sigma * sqrt(tau))
    d2 = d1 - sigma * sqrt(tau)

    Pass the forward with drift=0 (the carry is already in F), or pass spot with
    drift = r - q. Both are the same formula; only where the carry lives differs.
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    tau = np.asarray(tau, dtype=float)
    srt = sigma * np.sqrt(tau)
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(F / K) + (drift + 0.5 * sigma * sigma) * tau) / srt
        d2 = d1 - srt
    return d1, d2


def bs_vega(F, K, sigma, tau, df=1.0, drift=0.0):
    """dPrice/dSigma, per 1.0 of vol (divide by 100 for 'per vol point')."""
    d1, _ = d1_d2(F, K, sigma, tau, drift)
    return df * F * np.sqrt(tau) * norm_pdf(d1)


def bs_price(F, K, sigma, tau, df=1.0, right='C', drift=0.0):
    """Undiscounted-forward Black-76 price, then discounted by df."""
    d1, d2 = d1_d2(F, K, sigma, tau, drift)
    if right == 'C':
        return df * (F * norm_cdf(d1) - K * norm_cdf(d2))
    return df * (K * norm_cdf(-d2) - F * norm_cdf(-d1))


def otm_right(K, F):
    """The out-of-the-money side at this strike. Split at the FORWARD, not spot."""
    return 'C' if K >= F else 'P'


def implied_vol(price, F, K, tau, df=1.0, right=None, lo=1e-4, hi=5.0):
    """
    Invert Black-76 for sigma. Returns None when the price is outside the
    no-arbitrage bounds, rather than raising or returning a fabricated number.
    """
    if right is None:
        right = otm_right(K, F)
    if price is None or not np.isfinite(price) or price <= 0 or tau <= 0:
        return None
    intrinsic = df * max(F - K, 0.0) if right == 'C' else df * max(K - F, 0.0)
    upper = df * F if right == 'C' else df * K
    if price <= intrinsic + 1e-12 or price >= upper:
        return None

    def f(s):
        return float(bs_price(F, K, s, tau, df, right)) - price

    # Bracket first, then Newton on vega with a bisection guard. Price is
    # monotone in sigma, so the bracket can never be lost; vega is exactly the
    # derivative we need and we already have it, which makes this converge in a
    # handful of steps without pulling in scipy.optimize.
    if f(lo) > 0.0 or f(hi) < 0.0:
        return None
    a, b = lo, hi
    s = min(max(0.20, lo), hi)
    for _ in range(100):
        fs = f(s)
        if abs(fs) < 1e-12:
            return float(s)
        if fs > 0.0:
            b = s
        else:
            a = s
        if b - a < 1e-12:
            return float(0.5 * (a + b))
        v = float(bs_vega(F, K, s, tau, df))
        step = s - fs / v if v > 1e-12 else None
        s = step if (step is not None and a < step < b) else 0.5 * (a + b)
    return float(s)


# --- the forward, implied from the market -----------------------------------
def forward_from_parity(strikes, call_mid, put_mid, min_points=3):
    """
    Recover (F, discount_factor, r2) from put-call parity, per expiry.

        C - K_disc = P + F_disc   ->   C - P = df * (F - K)

    Regressing (C-P) on K gives slope = -df and intercept = df*F, so both the
    discount factor and the forward come out of the market. No dividend model,
    no rate assumption, no inherited broker guess.

    r2 doubles as a chain-quality score: a clean SPY expiry sits at ~1.0.
    """
    K = np.asarray(strikes, dtype=float)
    y = np.asarray(call_mid, dtype=float) - np.asarray(put_mid, dtype=float)
    ok = np.isfinite(K) & np.isfinite(y)
    if ok.sum() < min_points:
        return None, None, 0.0
    K, y = K[ok], y[ok]
    if np.ptp(K) <= 0:
        return None, None, 0.0

    slope, intercept = np.polyfit(K, y, 1)
    df = -slope
    if not np.isfinite(df) or df <= 0:
        return None, None, 0.0
    F = intercept / df

    resid = y - (slope * K + intercept)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else 0.0
    return float(F), float(df), float(r2)


# --- coordinates ------------------------------------------------------------
def log_moneyness(K, F):
    return np.log(np.asarray(K, dtype=float) / F)


def normalised_moneyness(K, F, sigma_atm, tau):
    """
    z = ln(K/F) / (sigma_atm * sqrt(tau)) -- strikes measured in standard
    deviations of the move to expiry.

    This is the coordinate that makes the grid mean the same thing at every
    maturity. A fixed +/-2% band has a width in z of 0.02/(sigma*sqrt(tau)),
    which diverges as tau -> 0: at half a day it is ~11 sigma (no market exists
    out there), at 12 days it is ~0.8 sigma (you never leave the ATM region).
    """
    denom = sigma_atm * np.sqrt(tau)
    if denom <= 0:
        return np.full(np.shape(K), np.nan)
    return log_moneyness(K, F) / denom


def total_variance(sigma, tau):
    """w = sigma^2 * tau."""
    return np.asarray(sigma, dtype=float) ** 2 * tau


def sigma_from_total_variance(w, tau):
    return np.sqrt(np.asarray(w, dtype=float) / tau)


def select_strikes_by_sigma(strikes, F, sigma_atm, tau, n_sigma=3.0, max_strikes=None):
    """
    Keep strikes within +/- n_sigma standard deviations of the forward.

    Unlike a fixed percentage band, this widens with sqrt(tau), so every expiry
    contributes the same slice of the distribution and every contract in the
    grid has comparable vega. That is what makes the quote noise homoskedastic
    instead of exploding in the short-dated wings.
    """
    width = n_sigma * sigma_atm * math.sqrt(tau)
    keep = [float(K) for K in sorted(strikes)
            if K > 0 and abs(math.log(K / F)) <= width]
    if max_strikes and len(keep) > max_strikes:
        # Thin evenly in z so the wings survive, rather than truncating them
        idx = np.linspace(0, len(keep) - 1, max_strikes).round().astype(int)
        keep = [keep[i] for i in sorted(set(idx))]
    return keep


# --- quality ----------------------------------------------------------------
def iv_uncertainty(half_spread, vega):
    """
    The standard error of an implied vol implied by the quoted spread.

    sigma_err = half_spread / vega. This is the single number that says how much
    a quote is worth. It is the reason a wing contract must not be given the
    same weight as an ATM one.
    """
    vega = np.asarray(vega, dtype=float)
    out = np.full(np.broadcast(vega, np.asarray(half_spread)).shape, np.inf, dtype=float)
    good = vega > 0
    np.divide(np.asarray(half_spread, dtype=float), vega, out=out, where=good)
    return out if out.shape else float(out)


def quote_weight(half_spread, vega):
    """Inverse-variance weight, 1 / sigma_err^2. Zero for a worthless quote."""
    err = iv_uncertainty(half_spread, vega)
    with np.errstate(divide='ignore'):
        return np.where(np.isfinite(err) & (err > 0), 1.0 / (err * err), 0.0)


# --- roughness --------------------------------------------------------------
def tricube(u):
    """LOESS kernel: (1 - |u|^3)^3 on |u| < 1, zero outside."""
    u = np.abs(np.asarray(u, dtype=float))
    return np.where(u < 1.0, (1.0 - u ** 3) ** 3, 0.0)


def atm_skew_from_slice(ks, sigmas, zs=None, weights=None, window=1.5, degree=2):
    """
    d(sigma)/dk at the money, from a LOCAL weighted polynomial fit.

    A global quadratic across the whole +/-3 sigma slice is a regression, not a
    derivative. Real smiles carry cubic and higher structure, so the wings bias
    the ATM slope. Measured on a synthetic slice with cubic curvature, the bias
    falls from 3.5e-2 at +/-3 sigma to 5e-4 at +/-0.5 sigma.

    Points are weighted by (inverse-variance quote weight) x (tricube kernel in
    z, centred at the money) so the estimate tapers smoothly rather than
    hard-cutting at the window edge.

    Returns (slope, stderr, n_used); (None, None, 0) when it cannot be formed.
    """
    k = np.asarray(ks, dtype=float)
    y = np.asarray(sigmas, dtype=float)
    ok = np.isfinite(k) & np.isfinite(y)

    if zs is None:
        z = k / (np.std(k) if np.std(k) > 0 else 1.0)
    else:
        z = np.asarray(zs, dtype=float)
        ok &= np.isfinite(z)

    w = np.ones_like(k)
    if weights is not None:
        qw = np.asarray(weights, dtype=float)
        ok &= np.isfinite(qw) & (qw > 0)
        w = np.where(np.isfinite(qw) & (qw > 0), qw, 0.0)

    kern = tricube(z / float(window))
    w = w * kern
    ok &= w > 0

    n = int(ok.sum())
    if n < degree + 2:
        return None, None, 0
    k, y, w = k[ok], y[ok], w[ok]

    try:
        coeffs, cov = np.polyfit(k, y, degree, w=np.sqrt(w), cov=True)
    except (np.linalg.LinAlgError, ValueError):
        return None, None, 0
    slope = float(coeffs[-2])                       # d/dk evaluated at k = 0
    var = float(cov[-2, -2])
    stderr = math.sqrt(var) if np.isfinite(var) and var > 0 else None
    return slope, stderr, n


def estimate_hurst(taus, skews, skew_errs=None):
    """
    Fit log|ATM skew| = c + (H - 1/2) log(tau), weighted least squares.

    The market's ATM skew explodes at the short end like tau^(H-1/2) with
    H ~ 0.1. A classical diffusion (Heston included) forces that exponent to 0
    as tau -> 0, so it cannot reproduce this; reading the slope off a log-log
    plot is the cheapest live test of whether a surface is rough.

    If skew_errs is given, each point is weighted by the inverse variance of
    log|skew|, which by the delta method is (|skew| / err)^2. Short expiries
    carry fewer strikes inside the ATM window, so their skew is measured less
    precisely, and equal weighting lets that noise into H.

    Returns (H, H_stderr, intercept, r2).
    """
    t = np.asarray(taus, dtype=float)
    s = np.abs(np.asarray(skews, dtype=float))
    ok = np.isfinite(t) & np.isfinite(s) & (t > 0) & (s > 0)

    w = None
    if skew_errs is not None:
        e = np.asarray([np.nan if v is None else v for v in skew_errs], dtype=float)
        good = np.isfinite(e) & (e > 0)
        if good.any():
            w = np.where(good, s / np.where(good, e, 1.0), np.nan)   # 1/sd of log|skew|
            ok &= np.isfinite(w) & (w > 0)

    if ok.sum() < 3:
        return None, None, None, 0.0
    x, y = np.log(t[ok]), np.log(s[ok])
    ww = w[ok] if w is not None else None

    try:
        coeffs, cov = np.polyfit(x, y, 1, w=ww, cov=True)
    except (np.linalg.LinAlgError, ValueError):
        return None, None, None, 0.0
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    var = float(cov[0, 0])
    h_err = math.sqrt(var) if np.isfinite(var) and var > 0 else None

    resid = y - (slope * x + intercept)
    if ww is not None:
        ss_res = np.sum(ww ** 2 * resid ** 2)
        ybar = np.sum(ww ** 2 * y) / np.sum(ww ** 2)
        ss_tot = np.sum(ww ** 2 * (y - ybar) ** 2)
    else:
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope + 0.5), h_err, intercept, float(r2)


# --- structure function: H from the realised path, not from the skew --------
def structure_function(x, deltas, q=1.0):
    """
    m_hat(q, Delta) = 1/(N - Delta) * sum_t |X_{t+Delta} - X_t|^q,
    the q-th order structure function of X_t = log(sigma_t).

    This is the second, independent route to H. The skew route reads the
    option surface; this one reads the realised volatility path. When the two
    disagree, the surface and the underlying are telling different stories.
    """
    X = np.asarray(x, dtype=float)
    X = X[np.isfinite(X)]
    out = []
    for d in np.asarray(deltas, dtype=int):
        if d < 1 or d >= len(X):
            out.append(np.nan)
            continue
        inc = np.abs(X[d:] - X[:-d])
        out.append(float(np.mean(inc ** q)))
    return np.array(out, dtype=float)


def zeta_regression(deltas, m_hat):
    """
    log m(q, Delta) = log C_q + zeta(q) log Delta.

    Returns (zeta, log_Cq, r2, zeta_stderr). On SPX the lecture's fit gives
    zeta_hat(1) ~ 0.1555 with R^2 ~ 0.93.
    """
    d = np.asarray(deltas, dtype=float)
    m = np.asarray(m_hat, dtype=float)
    ok = np.isfinite(d) & np.isfinite(m) & (d > 0) & (m > 0)
    if ok.sum() < 3:
        return None, None, 0.0, None
    x, y = np.log(d[ok]), np.log(m[ok])
    try:
        coeffs, cov = np.polyfit(x, y, 1, cov=True)
    except (np.linalg.LinAlgError, ValueError):
        return None, None, 0.0, None
    zeta, log_cq = float(coeffs[0]), float(coeffs[1])
    var = float(cov[0, 0])
    resid = y - (zeta * x + log_cq)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else 0.0
    return zeta, log_cq, float(r2), (math.sqrt(var) if var > 0 else None)


def hurst_from_structure(log_sigma, deltas=None, qs=(0.5, 1.0, 1.5, 2.0, 3.0)):
    """
    Estimate H from the monofractal scaling zeta(q) = q * H.

    Regress zeta(q) on q through the origin across several q. A monofractal
    (fBm-like) process gives a straight line through 0; curvature in zeta(q) is
    evidence of multifractality, which this returns as `linearity_r2` so it can
    be seen rather than assumed away.

    Returns dict with H, per-q zetas, and the monofractality check.
    """
    if deltas is None:
        deltas = np.unique(np.round(np.logspace(0, np.log10(60), 18)).astype(int))
    out = {'deltas': np.asarray(deltas), 'per_q': {}}
    zs, qq = [], []
    for q in qs:
        m = structure_function(log_sigma, deltas, q)
        zeta, log_cq, r2, se = zeta_regression(deltas, m)
        out['per_q'][q] = {'zeta': zeta, 'log_Cq': log_cq, 'r2': r2, 'stderr': se}
        if zeta is not None:
            zs.append(zeta)
            qq.append(q)
    if len(zs) < 2:
        out['H'] = None
        out['linearity_r2'] = 0.0
        return out
    qq, zs = np.array(qq), np.array(zs)
    H = float(np.sum(qq * zs) / np.sum(qq * qq))     # least squares through origin
    resid = zs - H * qq
    ss_tot = np.sum((zs - zs.mean()) ** 2)
    out['H'] = H
    out['linearity_r2'] = float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else 1.0
    return out


# --- finite differences -----------------------------------------------------
def fd_first(x, y):
    """
    Central first derivative on a possibly non-uniform grid; one-sided at the
    ends. Option grids are never evenly spaced, so the uniform-grid formula
    would carry an O(h) bias exactly where the strikes thin out.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    d = np.full(n, np.nan)
    if n < 2:
        return d
    for i in range(1, n - 1):
        h1, h2 = x[i] - x[i - 1], x[i + 1] - x[i]
        if h1 <= 0 or h2 <= 0:
            continue
        d[i] = (-h2 / (h1 * (h1 + h2)) * y[i - 1]
                + (h2 - h1) / (h1 * h2) * y[i]
                + h1 / (h2 * (h1 + h2)) * y[i + 1])
    d[0] = (y[1] - y[0]) / (x[1] - x[0])
    d[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return d


def fd_second(x, y):
    """
    Central second derivative on a possibly non-uniform grid; NaN at the ends.

    CAVEAT worth knowing before trusting a risk-neutral density: this 3-point
    stencil is exact for cubics on a UNIFORM grid, but on a non-uniform grid it
    is only exact for quadratics and drops to first-order accuracy, with error
    ~ (h2 - h1) * f''' / 3. Real strike grids are non-uniform -- SPY is $1 near
    the money and $5 in the wings -- so a Breeden-Litzenberger density computed
    straight off raw strikes carries a real bias exactly where the spacing
    changes. Interpolate onto an even grid first, or fit a smooth slice and
    differentiate that, when the density itself is the deliverable.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    d = np.full(n, np.nan)
    for i in range(1, n - 1):
        h1, h2 = x[i] - x[i - 1], x[i + 1] - x[i]
        if h1 <= 0 or h2 <= 0:
            continue
        d[i] = 2.0 * (h2 * y[i - 1] - (h1 + h2) * y[i] + h1 * y[i + 1]) / (h1 * h2 * (h1 + h2))
    return d


def fd_weights(x0, xs, m):
    """
    Fornberg (1988) weights: d^m/dx^m at x0 of the polynomial through the nodes
    xs, for ANY node placement. Returns w with f^(m)(x0) ~ sum_j w_j f(x_j).

    Exact for polynomials up to degree len(xs) - 1, which is the point: five
    nodes make the second derivative exact for quartics on any grid, so the
    (h2 - h1) f-triple-prime term that costs the 3-point stencil an order on
    non-uniform spacing simply is not there.
    """
    xs = np.asarray(xs, dtype=float)
    n = len(xs)
    c = np.zeros((n, m + 1))
    c1, c4 = 1.0, xs[0] - x0
    c[0, 0] = 1.0
    for i in range(1, n):
        mn = min(i, m)
        c2, c5, c4 = 1.0, c4, xs[i] - x0
        for j in range(i):
            c3 = xs[i] - xs[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i, k] = c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2
    return c[:, m]


def fd_second_poly(x, y, width=5):
    """
    Second derivative from the local polynomial through `width` neighbouring
    points (Fornberg weights). NaN at the two endpoints, like fd_second.

    This is the fix for fd_second's order loss on non-uniform grids, and it was
    chosen by measurement rather than argument (debug_fd_methods.py). Against the
    exact lognormal density, observed convergence order of the max error:

        grid                       3-pt   mapped 3-pt   log-map   5-pt poly
        smooth stretch             2.00      2.00         1.99       3.99
        geometric                  2.00      2.00         2.00       3.98
        $1 / $5 kink (SPY-like)    1.07      0.01         0.87       3.06
        random spacing             0.84     -0.12         0.84       2.76

    The coordinate-mapping idea -- differentiate in an index coordinate where the
    grid is uniform and chain-rule back -- is second order only when the grid
    function K(x) is itself smooth; at a kink its numerically differenced metric
    K'' is O(1) wrong and the method does not converge at all (44% error at every
    resolution). On smooth grids the plain 3-point stencil is already O(h^2)
    because h2 - h1 is O(h^2) there. The local polynomial needs no assumption
    about the grid and is the only one of the four that holds second order on
    the grids that actually cause the problem.

    Uniform grids take a vectorised fast path (the classical -1 16 -30 16 -1 /
    12h^2 stencil); everything else builds per-point weights.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    d = np.full(n, np.nan)
    if n < 3:
        return d
    width = int(min(max(width, 3), n))
    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("fd_second_poly needs a strictly increasing grid")
    uniform = (h.max() - h.min()) <= 1e-12 * h.mean()
    if uniform and width == 5 and n >= 5:
        hh = h.mean()
        d[2:-2] = (-y[:-4] + 16.0 * y[1:-3] - 30.0 * y[2:-2]
                   + 16.0 * y[3:-1] - y[4:]) / (12.0 * hh * hh)
        for i in (1, n - 2):
            lo = max(0, min(i - 2, n - 5))
            d[i] = float(fd_weights(x[i], x[lo:lo + 5], 2) @ y[lo:lo + 5])
        return d
    half = width // 2
    for i in range(1, n - 1):
        lo = max(0, min(i - half, n - width))
        w = fd_weights(x[i], x[lo:lo + width], 2)
        d[i] = float(w @ y[lo:lo + width])
    return d


def breeden_litzenberger(strikes, call_prices, df=1.0, stencil="poly5"):
    """
    Risk-neutral density q(K) = (1/df) * d2C/dK2.

    The ultimate quality gate on a surface: a negative density anywhere is a
    butterfly arbitrage, i.e. bad data or a bad fit. Returns (K, density) with
    NaN at the two endpoints where a central second difference does not exist.

    `stencil` is "poly5" (five-point local polynomial, second order on any grid;
    see fd_second_poly) or "3pt" (the classical stencil, first order on
    non-uniform strikes -- kept so the difference can be measured).

    This is the RAW density and stays raw on purpose: its sign is the arbitrage
    signal the panel exists to show. Anything that samples from it should go
    through `density_for_sampling`, which floors and renormalises.
    """
    K = np.asarray(strikes, dtype=float)
    C = np.asarray(call_prices, dtype=float)
    order = np.argsort(K)
    K, C = K[order], C[order]
    if stencil == "3pt":
        return K, fd_second(K, C) / df
    if stencil != "poly5":
        raise ValueError(f"unknown stencil {stencil!r}")
    return K, fd_second_poly(K, C) / df


def density_for_sampling(strikes, density, floor=None):
    """
    A copy of a risk-neutral density made safe for anything that SAMPLES it:
    NaNs dropped, floored at `floor` (default: machine epsilon), renormalised to
    unit mass under the trapezoid rule.

    Deliberately separate from breeden_litzenberger. Flooring hides the sign,
    and the sign is the diagnostic -- a Monte Carlo that draws from a density
    with -7e-10 in its tail evaluates a negative probability, but a panel that
    shows that -7e-10 is doing its job. Use the raw density to look, this one to
    draw. Returns (K, q, info) where info records what the floor removed, so
    that a floor hiding a real arbitrage shows up in a log rather than nowhere.
    """
    K = np.asarray(strikes, dtype=float)
    q = np.asarray(density, dtype=float)
    ok = np.isfinite(q) & np.isfinite(K)
    K, q = K[ok], q[ok]
    if len(K) < 2:
        raise ValueError("density_for_sampling needs at least two finite points")
    if floor is None:
        floor = float(np.finfo(float).eps)
    raw_mass = float(np.trapezoid(q, K))
    negative_mass = float(np.trapezoid(np.minimum(q, 0.0), K))
    qf = np.maximum(q, floor)
    mass = float(np.trapezoid(qf, K))
    if mass <= 0.0:
        raise ValueError("density has no positive mass to renormalise")
    # Renormalise, then floor again: dividing by a mass a hair above one would
    # leave the floored points a rounding error BELOW the floor. The second
    # floor moves the mass by at most n_floored * floor * dK, i.e. ~1e-14,
    # which is as "exactly one" as floating point offers.
    qf = np.maximum(qf / mass, floor)
    info = {"raw_mass": raw_mass, "negative_mass": negative_mass,
            "n_floored": int(np.count_nonzero(q < floor)),
            "renorm_factor": 1.0 / mass, "floor": floor}
    return K, qf, info


# --- arbitrage --------------------------------------------------------------
def butterfly_violations(ks, ws):
    """
    Indices where total variance is not convex in k on the RAW grid.

    This is a data-sanity heuristic, not the arbitrage condition. The rigorous
    no-butterfly test is Gatheral's g(k) >= 0 (see fit/svi.py) or, equivalently,
    a non-negative risk-neutral density; convexity of w in k is neither
    necessary nor sufficient for it. What this is good for is exactly what it
    is used for here: catching a single bad print that dents an otherwise smooth
    slice. Flag it, never smooth it away.
    """
    k = np.asarray(ks, dtype=float)
    w = np.asarray(ws, dtype=float)
    order = np.argsort(k)
    k, w = k[order], w[order]
    bad = []
    for i in range(1, len(k) - 1):
        if not all(np.isfinite([w[i - 1], w[i], w[i + 1]])):
            continue
        h1, h2 = k[i] - k[i - 1], k[i + 1] - k[i]
        if h1 <= 0 or h2 <= 0:
            continue
        second = (w[i + 1] - w[i]) / h2 - (w[i] - w[i - 1]) / h1
        if second < 0:
            bad.append(int(order[i]))
    return bad


def calendar_violations(taus, ws_at_fixed_k):
    """
    Indices where total variance decreases with maturity at a fixed k, which is
    a calendar arbitrage. w must be non-decreasing in tau.
    """
    t = np.asarray(taus, dtype=float)
    w = np.asarray(ws_at_fixed_k, dtype=float)
    order = np.argsort(t)
    bad = []
    for a, b in zip(order[:-1], order[1:]):
        if np.isfinite(w[a]) and np.isfinite(w[b]) and w[b] < w[a] - 1e-12:
            bad.append(int(b))
    return bad
