"""
The dual Kalman filter of the trial: dual estimation (Wan & Nelson 1997) used to calibrate.

Two filters run side by side, each using the other's latest estimate:

  state filter      x_k = (alpha, b_MKT, b_SMB, b_HML, b_MOM), the portfolio's exposures in
                    signal period k: a random walk driven by the known effect of the last
                    parameter change (the control input J_x d theta), measured by the
                    period's FF4 fit (falsify) with its Newey-West variances as the noise.
  parameter filter  theta_k = (H, log c, log delta [, log delta_eq]), a random walk, measured
                    by a pseudo-observation of what is WANTED rather than what happened:
                        z = (beta*, 0) = h(theta) + v,    h = (b_MKT(theta), cost(theta)),
                    linearised with the Jacobian of the period's realised b_MKT and cost,
                    measured on that same period by counterfactual portfolios: the database
                    holds every return, so the portfolio built with theta + d theta can be run
                    over the same days from the same holdings. The b_MKT the update starts
                    from is the state filter's, not the raw fit: that is the coupling.

The noise of the pseudo-observation is the exchange rate between the two goals, fixed
before a run and not tuned: eps_beta on the beta (plus the state filter's own variance) and
eps_cost on the cost in % a year. The parameter update is a Gauss-Newton step on
(b_MKT - beta*)^2 / eps_beta^2 + cost^2 / eps_cost^2, damped by the filter's covariance.
"""

import math
from dataclasses import dataclass

import numpy as np

import portfolio.factor_fit as ff

LOG_BOUNDS = (math.log(0.05), math.log(20.0))
BOUNDS = {"H": (0.02, 0.49), "confidence": LOG_BOUNDS, "delta": (math.log(0.5), math.log(20.0)),
          "delta_eq": (math.log(0.5), math.log(20.0))}
FD_STEP = {"H": 0.02, "confidence": 0.10, "delta": 0.05, "delta_eq": 0.05}
Q_STEP = {"H": 0.01, "confidence": 0.05, "delta": 0.03, "delta_eq": 0.03}       # per period, sd
P0_SD = {"H": 0.10, "confidence": 0.50, "delta": 0.30, "delta_eq": 0.30}


@dataclass
class Gaussian:
    x: np.ndarray
    P: np.ndarray


def kalman_update(g, z, H, R, h=None):
    """Update g with z = H x + v (or z = h + H (x - x_prior) + v when h is given)."""
    pred = H @ g.x if h is None else h
    S = H @ g.P @ H.T + R
    K = np.linalg.solve(S, H @ g.P).T
    x = g.x + K @ (z - pred)
    P = (np.eye(len(g.x)) - K @ H) @ g.P
    return Gaussian(x, 0.5 * (P + P.T)), float((z - pred) @ np.linalg.solve(S, z - pred))


def theta_of(cfg, names):
    return np.array([cfg.H if n == "H" else math.log(getattr(cfg, n)) for n in names])


def clip(theta, names):
    return np.array([min(max(v, BOUNDS[n][0]), BOUNDS[n][1]) for v, n in zip(theta, names)])


def run(engine, cfg0, names, target_beta, start, end, eps_beta=0.05, eps_cost=0.25,
        use_falsify=True, x_q=(1e-5, 0.02, 0.02, 0.02, 0.02), r_mode="nw", q_scale=1.0):
    """
    The dual filter from `start` to `end`, one step per signal period. Returns the daily
    results of the portfolio it actually held, the per-period fits and the filter paths.

    r_mode "nw" (the pre-registered filter) measures each period with its own Newey-West
    variances. On a volatility-managed portfolio that weighting is biased: the calm periods,
    where exposure is highest, are the ones whose beta is least precise (factor variance is
    low), so precision weighting underweights high exposure (trial: corr(b, se) = 0.55, the
    precision-weighted mean beta 0.73 against a simple mean 0.88). "constant" uses the running
    median of the variances instead -- equal weights. q_scale multiplies the parameters' drift.
    """
    m = engine.m
    d = len(names)
    periods = [int(p) for p in engine.period_starts if start <= p < end]
    theta = theta_of(cfg0, names)
    pf = Gaussian(theta.copy(), np.diag([P0_SD[n] ** 2 for n in names]))
    Qt = np.diag([(q_scale * Q_STEP[n]) ** 2 for n in names])
    se_hist = []
    st = Gaussian(np.array([0.0, target_beta, 0.0, 0.0, 0.0]), np.diag([1e-6, 0.25, 0.25, 0.25, 0.25]))
    Qx = np.diag(np.asarray(x_q) ** 2)
    w = None
    daily, fits, path = [], [], []
    Jx_prev, dtheta_prev = np.zeros((5, d)), np.zeros(d)
    for k, p0 in enumerate(periods):
        p1 = periods[k + 1] if k + 1 < len(periods) else end
        cfg = cfg0.with_theta(theta, names)
        res = engine.run(cfg, p0, p1, w0=w)
        F = m.F[p0:p1]
        f = ff.fit(res["net"], F, use_falsify)
        f["period"], f["first_day"] = k, p0
        fits.append(f)
        # state filter: predict with the control effect of the last parameter change, then update
        st = Gaussian(st.x + Jx_prev @ dtheta_prev, st.P + Qx)
        zx = np.concatenate([[f["alpha"]], f["betas"]])
        v = np.concatenate([[f["se_alpha"]], f["se"]]) ** 2 + 1e-12
        se_hist.append(v)
        Rx = np.diag(v if r_mode == "nw" else np.median(np.array(se_hist), axis=0))
        st, _ = kalman_update(st, zx, np.eye(5), Rx)
        # counterfactuals on the same days from the same holdings: the Jacobian in theta
        base = ff.ols_nw(res["net"], F)[0]
        cost0 = 100 * 252 * float(res["cost"].sum()) / (p1 - p0)
        J = np.zeros((2, d))
        Jx = np.zeros((5, d))
        for i, n in enumerate(names):
            th = theta.copy()
            th[i] += FD_STEP[n]
            r2 = engine.run(cfg0.with_theta(th, names), p0, p1, w0=w)
            c2 = ff.ols_nw(r2["net"], F)[0]
            Jx[:, i] = (c2 - base) / FD_STEP[n]
            J[0, i] = Jx[1, i]
            J[1, i] = (100 * 252 * float(r2["cost"].sum()) / (p1 - p0) - cost0) / FD_STEP[n]
        # parameter filter: the pseudo-observation of the target
        pf = Gaussian(pf.x, pf.P + Qt)
        h = np.array([st.x[1], cost0])
        R = np.diag([eps_beta ** 2 + st.P[1, 1], eps_cost ** 2])
        # the beta channel's normalised innovation: ~N(0, 1) if the filter is consistent. The
        # cost channel's is not a check: a cost of zero is a goal the pseudo-observation never
        # reaches, so its innovation stays positive by construction.
        S_bb = float(J[0] @ pf.P @ J[0] + R[0, 0])
        z_beta = (target_beta - h[0]) / math.sqrt(S_bb)
        new, nis = kalman_update(pf, np.array([target_beta, 0.0]), J, R, h=h)
        new_theta = clip(new.x, names)
        dtheta_prev, Jx_prev = new_theta - theta, Jx
        theta = new_theta
        pf = Gaussian(new_theta, new.P)
        w = res["w_end"]
        daily.append(res)
        path.append({"period": k, "first_day": p0, "theta": theta.tolist(),
                     "sd": np.sqrt(np.diag(pf.P)).tolist(), "beta_mkt": float(st.x[1]),
                     "beta_mkt_sd": float(math.sqrt(st.P[1, 1])), "b_mkt_fit": float(f["betas"][0]),
                     "cost_pct": cost0, "J": J.tolist(), "nis": nis, "z_beta": z_beta})
    out = {k: np.concatenate([r[k] for r in daily]) for k in
           ("gross", "cost", "net", "turnover", "beta_true", "gross_exposure", "days")}
    out["fits"], out["path"], out["names"] = fits, path, list(names)
    return out
