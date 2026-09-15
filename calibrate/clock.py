"""
Which clock does the option market price short-dated variance on? (Session H)

Calendar time (ACT/365 on seconds) says a Friday-to-Monday option has three days of
variance to go; a market that expects little variance over the weekend prices
about one. For a one-day option the overnight hours are 17.5 of 30. Every short-end
number built on tau -- implied vol, the ATM skew term structure, H -- inherits the
choice.

The trading clock (volsurf_core.variance_time) counts session seconds at 1 and
overnight / weekend seconds at omega. It is measured from the chain itself:

  1. ATM total variance w_j = sigma_j^2 tau_j per expiry, read from prices with
     tau = 1 -- Black's price depends on sigma^2 tau only, so this involves no clock.
  2. From the snapshot instant t0 to each expiry's close the seconds split into
     session S_j, one-night gaps O_j and weekend or holiday gaps W_j.
  3. With a flat forward variance per unit of variance time over the short window,

         w_j = a (S_j + omega (O_j + W_j)) + sum_k e_k [j >= k] + noise_j

     a is variance per session-second, omega the price of a night-second relative
     to it, e_k the extra variance the chain prices on an event day (a central-bank
     decision, a data release) -- a step in total variance. For fixed omega the model
     is linear in (a, e), so omega is PROFILED: generalised least squares at each
     omega, noise proportional to w (a constant relative error in implied variance),
     residual sums pooled over snapshots with one a per snapshot. The interval is
     where the pooled residual is within F(1, dof)_0.95 noise variances of its minimum.
  4. Events are found, not assumed: a step is added at the expiry where it lowers
     the residual most and kept if that is a better-than-4-sigma improvement, at most
     three per snapshot (the September 2026 chains open on an FOMC day).

What identifies omega: consecutive weekday expiries each add one session and one
17.5-hour night under ANY clock, so together they only fix a (6.5 h + 17.5 h omega).
Weekend spans (one session, 65.5 hours of gap) and the partial first day pull the
two apart. The first version of this module scored a clock by how constant the
forward rates between neighbouring expiries were, with the median absolute deviation
of log rates as a robust loss -- which treats exactly those weekend spans as
outliers. On noise-free chains planted at omega = 0.1 it returned 0.187, at 0.5 it
returned 0.017 (Session H; test_clock.py keeps the case).

Caveat: omega absorbs any day-of-week pattern in session variance. A market that
prices Mondays as busier than Tuesdays reads here as weekend variance.
"""

import copy
import math

import numpy as np

import volsurf_core as vc

EVENT_Z = 4.0
MAX_EVENTS = 3


def atm_total_variance(app, ctxs, n_near=5):
    """
    {expiry: (w_atm, expiry_date)} from a replayed snapshot: OTM mids near the
    forward -> total variance by inverting Black with tau = 1 -> local quadratic in
    log-moneyness evaluated at k = 0.
    """
    by_exp = {}
    for q in app.quotes.values():
        ctx = ctxs.get(q.expiry)
        if ctx is None or not np.isfinite(ctx.forward) or q.right != vc.otm_right(q.strike, ctx.forward):
            continue
        mid = q.mid
        if not (np.isfinite(mid) and mid > 0):
            continue
        s1 = vc.implied_vol(mid, ctx.forward, q.strike, 1.0, ctx.discount, q.right, hi=5.0)
        if s1 is None or not np.isfinite(s1) or s1 <= 0:
            continue
        by_exp.setdefault(q.expiry, []).append((math.log(q.strike / ctx.forward), s1 * s1))
    out = {}
    for exp, rows in by_exp.items():
        rows.sort(key=lambda r: abs(r[0]))
        near = np.array(rows[:n_near])
        if len(near) < 3:
            continue
        k, w = near[:, 0], near[:, 1]
        deg = 2 if len(near) >= 4 else 1
        coef = np.polyfit(k, w, deg)
        out[exp] = (float(np.polyval(coef, 0.0)), vc.parse_ib_date(exp))
    return out


class Snapshot:
    """One chain's short end: expiry dates, ATM total variances and the seconds split (in years)."""

    def __init__(self, t0, atm, max_days=15):
        today = vc.new_york_date(t0)
        items = sorted(((d, w) for w, d in atm.values() if np.isfinite(w) and w > 0), key=lambda r: r[0])
        items = [(d, w) for d, w in items if 0 <= (d - today).days <= max_days]
        self.t0 = t0
        self.dates = [d for d, _ in items]
        self.w = np.array([w for _, w in items], float)
        split = np.array([vc.time_split(t0, vc.expiry_utc(d)) for d in self.dates], float).reshape(-1, 3)
        split = split / vc.SECONDS_PER_YEAR
        self.S, self.O, self.W = split[:, 0], split[:, 1], split[:, 2]
        self.events = []                      # indices k: a step in w from expiry k on

    def __len__(self):
        return len(self.w)

    def steps(self):
        n = len(self)
        return np.array([[1.0 if j >= k else 0.0 for k in self.events] for j in range(n)]).reshape(n, len(self.events))

    def identified(self):
        """Weekend or holiday time must vary across expiries, or omega is not separable from a."""
        return len(self) >= 4 and float(np.ptp(self.W)) > 0 and float(np.ptp(self.S)) > 0


def _gls(X, w):
    """min || (w - X beta) / w ||^2: (beta, residual sum, (X'WX)^-1)."""
    Xw = X / w[:, None]
    beta, *_ = np.linalg.lstsq(Xw, np.ones(len(w)), rcond=None)
    r = 1.0 - Xw @ beta
    try:
        cov = np.linalg.inv(Xw.T @ Xw)
    except np.linalg.LinAlgError:
        cov = np.full((X.shape[1], X.shape[1]), np.nan)
    return beta, float(r @ r), cov


def _free_design(sn, split_weekend=False):
    base = [sn.S, sn.O, sn.W] if split_weekend else [sn.S, sn.O + sn.W]
    return np.column_stack(base + ([sn.steps()] if sn.events else []))


def find_events(sn, z=EVENT_Z, max_events=MAX_EVENTS):
    """Greedy step detection on the clock-free design [S, O + W]; sets sn.events and returns them."""
    sn.events = []
    while len(sn.events) < max_events:
        X0 = _free_design(sn)
        _, sse0, _ = _gls(X0, sn.w)
        best = None
        for k in range(1, len(sn)):
            if k in sn.events:
                continue
            step = np.array([1.0 if j >= k else 0.0 for j in range(len(sn))])
            _, sse, _ = _gls(np.column_stack([X0, step]), sn.w)
            if best is None or sse < best[1]:
                best = (k, sse)
        dof = len(sn) - X0.shape[1] - 1
        if best is None or dof < 2:
            break
        s2 = best[1] / dof
        if s2 <= 0 or (sse0 - best[1]) / s2 < z * z:
            break
        sn.events.append(best[0])
    sn.events.sort()
    return sn.events


def fit_snapshot(sn):
    """Unconstrained GLS on [S, O + W, steps]: omega = b / a with a delta-method SE; and with O, W apart."""
    out = {"n": len(sn), "events": [sn.dates[k].isoformat() for k in sn.events], "identified": sn.identified()}
    if not sn.identified():
        return dict(out, omega=float("nan"), se=float("nan"))
    X = _free_design(sn)
    beta, sse, cov = _gls(X, sn.w)
    dof = max(len(sn) - X.shape[1], 1)
    s2 = sse / dof
    a, b = beta[0], beta[1]
    g = np.array([-b / a ** 2, 1.0 / a])
    se = float(math.sqrt(max(g @ (cov[:2, :2] * s2) @ g, 0.0)))
    out.update(omega=float(b / a), se=se, a=float(a), rel_noise=float(math.sqrt(s2)),
               event_variance=[float(e) for e in beta[2:]])
    if len(sn) - 3 - len(sn.events) >= 2 and float(np.ptp(sn.O)) > 0:
        b3, _, _ = _gls(_free_design(sn, split_weekend=True), sn.w)
        out.update(omega_overnight=float(b3[1] / b3[0]), omega_weekend=float(b3[2] / b3[0]))
    return out


def profile(snaps, grid):
    """Pooled GLS residual sum at each omega (one level a and its own event steps per snapshot)."""
    total = np.zeros(len(grid))
    n_obs = n_par = 0
    for sn in snaps:
        steps = sn.steps()
        for i, om in enumerate(grid):
            X = np.column_stack([sn.S + om * (sn.O + sn.W)] + ([steps] if sn.events else []))
            total[i] += _gls(X, sn.w)[1]
        n_obs += len(sn)
        n_par += 1 + len(sn.events)
    return total, n_obs, n_par


def estimate_omega(snapshots, grid=None, max_days=15, detect_events=True):
    """
    snapshots: list of (t0, atm) pairs, atm as from atm_total_variance. Returns

      omega, ci95          profile estimate pooled over snapshots
      grid, chi2           the profile, as (residual - min) / noise variance
      calendar_chi2        how far the calendar clock (omega = 1) is from the best fit
      per_snapshot         each snapshot's own unconstrained fit (omega, se, events,
                           omega_overnight / omega_weekend when separable)
    """
    grid = np.geomspace(0.002, 2.0, 121) if grid is None else np.asarray(grid, float)
    snaps = [Snapshot(t0, atm, max_days) for t0, atm in snapshots]
    snaps = [sn for sn in snaps if sn.identified()]
    if not snaps:
        return {"omega": float("nan"), "ci95": (float("nan"), float("nan")), "identified": False,
                "reason": "no snapshot has weekend or holiday time inside the window"}
    if detect_events:
        for sn in snaps:
            find_events(sn)
    per = [fit_snapshot(sn) for sn in snaps]
    from scipy.optimize import brentq, minimize_scalar
    sse, n_obs, n_par = profile(snaps, grid)
    lg = np.log(grid)
    j = int(np.argmin(sse))

    def f(log_om):
        return profile(snaps, [math.exp(log_om)])[0][0]
    # the grid is a bracket, not the answer: at 1% noise the interval is +-3%, finer than
    # its 6% spacing, and a grid-point interval covered the truth 7-28% of the time
    lo_b, hi_b = lg[max(j - 1, 0)], lg[min(j + 1, len(grid) - 1)]
    res = minimize_scalar(f, bounds=(lo_b, hi_b), method="bounded", options={"xatol": 1e-7})
    l_hat, s_min = (float(res.x), float(res.fun)) if res.fun <= sse[j] else (float(lg[j]), float(sse[j]))
    dof = max(n_obs - n_par - 1, 1)
    s2 = s_min / dof
    chi2 = (sse - s_min) / s2 if s2 > 0 else np.zeros_like(sse)
    # the noise variance is estimated from ~a dozen residuals, so the cut is F(1, dof),
    # not chi2(1): 4.7 at dof 12 instead of 3.84 (measured coverage 0.87-0.93 -> see test_clock)
    from scipy.stats import f as f_dist
    cut = float(f_dist.ppf(0.95, 1, dof))

    def edge(side):
        idx = np.arange(0, j) if side < 0 else np.arange(j + 1, len(grid))
        out = idx[chi2[idx] > cut]
        if s2 <= 0 or len(out) == 0:
            return float(grid[0] if side < 0 else grid[-1])        # not bounded inside the grid
        k = out.max() if side < 0 else out.min()
        return float(math.exp(brentq(lambda z: (f(z) - s_min) / s2 - cut, *sorted((lg[k], l_hat)))))
    ci = (edge(-1), edge(+1))
    cal = float((f(0.0) - s_min) / s2) if s2 > 0 else float("inf")
    return {"omega": float(math.exp(l_hat)), "ci95": ci, "identified": True, "grid": grid.tolist(),
            "chi2": chi2.tolist(), "calendar_chi2": cal, "per_snapshot": per, "n_expiries": n_obs,
            "rel_noise": float(math.sqrt(s2))}


def forward_rates(t0, atm, omega, max_days=15):
    """Forward variance per year of variance time between consecutive expiries (for plots)."""
    sn = Snapshot(t0, atm, max_days)
    tau = np.array([vc.variance_years((s, o, w_), omega) for s, o, w_ in
                    zip(sn.S * vc.SECONDS_PER_YEAR, sn.O * vc.SECONDS_PER_YEAR, sn.W * vc.SECONDS_PER_YEAR)])
    dtau, dw = np.diff(np.concatenate([[0.0], tau])), np.diff(np.concatenate([[0.0], sn.w]))
    ok = dtau > 0
    return dw[ok] / dtau[ok], [d for d, k in zip(sn.dates, ok) if k]


def snapshot_atm(path, n_near=5):
    """(t0 in UTC, atm total variances) for a recorded snapshot."""
    import sources.replay as replay
    app, ctxs = replay.load(path)
    return replay.recorded_utc(app, ctxs), atm_total_variance(app, ctxs, n_near)


def retime(ctxs, t0, omega):
    """A copy of the expiry contexts with tau on the trading clock (for calibration and H)."""
    out = {}
    for exp, ctx in ctxs.items():
        c = copy.copy(ctx)
        c.tau = vc.variance_time(t0, vc.parse_ib_date(exp), omega)
        out[exp] = c
    return out
