"""
Session L trial -- the Black-Litterman portfolio, its demo market, and the dual Kalman filter
(package portfolio/).

  market        the simulated factors have the planted premia, vols and correlations; the RV
                proxy is level-unbiased; assets carry NO alpha, and their loadings are the
                planted ones; the database round-trips exactly
  bl            the update form equals the textbook information form; zero confidence returns
                the equilibrium and infinite confidence the views; Pi = delta Sigma w_mkt makes
                w_mkt optimal for an investor of that risk aversion (He & Litterman's identity)
  forecast      the GJR weights sum to one, are positive, and reach further back as H falls;
                at H = 1/2 the forecast is the last observation; the variogram estimator
                recovers a planted H from exact fBm
  backtest      the portfolio uses no information from the future (a run to day t is unchanged
                when later returns are replaced); costs equal turnover x the rate; the gross cap
                binds; a lower H gives a smoother exposure path and less turnover
  falsify       its factor regression and this package's own agree to rounding
  filter        the Kalman update equals the textbook gain form; the parameter filter moves
                downhill on its objective; bounds hold

    python test_portfolio.py
"""

import math
import sys
import tempfile
import pathlib

import numpy as np

import models.fbm as fbm
import portfolio.backtest as bt
import portfolio.black_litterman as bl
import portfolio.demo_market as dm
import portfolio.dual_kalman as dk
import portfolio.factor_fit as ff
import portfolio.rough_forecast as rf

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


SPEC = dm.MarketSpec(n_days=1512, n_assets=25, seed=7)
M = dm.simulate(SPEC)


def test_market():
    print("\nthe demo market")
    ann_vol = M.F.std(axis=0) * math.sqrt(252)
    # the realised vol of a rough-vol factor is its long-run level up to the sample's own spread
    check("factor volatilities are within 15% of the planted long-run levels",
          bool(np.all(np.abs(ann_vol / np.array(SPEC.vol) - 1) < 0.15)),
          ", ".join(f"{n} {v:.3f} (planted {p})" for n, v, p in zip(dm.FACTORS, ann_vol, SPEC.vol)))
    C = np.corrcoef(M.F.T)
    # 1/sqrt(n) is the nominal SE of a sample correlation; stochastic volatility makes the
    # effective sample smaller, so the worst of the six off-diagonals is judged at 4 of them
    tol = 4.0 / math.sqrt(M.T)
    check("the factor correlation matrix is the planted one within 4 nominal SE",
          float(np.max(np.abs(C - np.array(SPEC.corr)))) < tol,
          f"worst {np.max(np.abs(C - np.array(SPEC.corr))):.3f} against {tol:.3f}")
    bias = M.RV.mean(axis=0) / (M.VOL ** 2 / 252).mean(axis=0)
    check("the realised-variance proxy is level-unbiased (E RV = sigma^2 dt), within 5%",
          bool(np.all(np.abs(bias - 1) < 0.05)), ", ".join(f"{b:.3f}" for b in bias))
    # no alpha: each asset's return is exactly its loadings times the factors, plus noise
    resid = M.R - M.F @ M.B.T
    t = resid.mean(axis=0) / (resid.std(axis=0, ddof=1) / math.sqrt(M.T))
    check("no asset carries alpha: every intercept is within 4 SE of zero", bool(np.all(np.abs(t) < 4.0)),
          f"worst |t| {np.max(np.abs(t)):.2f} over {M.N} assets")
    X = np.column_stack([np.ones(M.T), M.F])
    fit = np.linalg.lstsq(X, M.R, rcond=None)[0]
    coef = fit[1:].T
    res = M.R - X @ fit
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X))[1:][:, None] * res.var(axis=0, ddof=5)[None, :]).T
    check("OLS on the factors recovers every planted loading within 4 SE",
          float(np.max(np.abs((coef - M.B) / se))) < 4.0,
          f"worst |t| {np.max(np.abs((coef - M.B) / se)):.2f} over {4 * M.N} loadings; "
          f"worst absolute error {np.max(np.abs(coef - M.B)):.3f}, median SE {np.median(se):.3f}")
    w = M.w_mkt(100)
    check("market weights are cap shares and sum to one", abs(w.sum() - 1) < 1e-12 and bool(np.all(w > 0)))


def test_database():
    print("\nthe database")
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "m.sqlite"
        dm.write(M, p)
        back = dm.load(p)
        same = all(np.array_equal(getattr(M, a), getattr(back, a)) for a in ("F", "RV", "VOL", "R", "B", "size",
                                                                            "value", "mom_load", "idio", "cap0", "regime"))
        check("the market round-trips through SQLite exactly", same)
        check("...and so does its spec", back.spec == M.spec, f"{back.spec.name}")
        dm.write(M, p)
        n = dm.load(p).T
        check("writing twice leaves one copy, not two", n == M.T, f"{n} days")


def test_black_litterman():
    print("\nBlack-Litterman")
    rng = np.random.default_rng(0)
    n = 8
    A = rng.standard_normal((n, n))
    Sigma = A @ A.T / n + np.eye(n) * 0.05
    w_mkt = np.abs(rng.random(n))
    w_mkt /= w_mkt.sum()
    P = np.array([np.concatenate([[1, -1], np.zeros(n - 2)]), np.concatenate([np.zeros(n - 4), [0.5, 0.5, -0.5, -0.5]])])
    Q = np.array([0.03, 0.02])
    Pi = bl.equilibrium(Sigma, w_mkt, 2.5)
    Om = bl.omega(P, Sigma, 0.05, 1.0)
    mu, Mp = bl.posterior(Pi, Sigma, P, Q, Om, 0.05)
    mu2, M2 = bl.posterior_information_form(Pi, Sigma, P, Q, Om, 0.05)
    check("the update form equals the information form (mean and covariance)",
          float(np.max(np.abs(mu - mu2))) < 1e-10 and float(np.max(np.abs(Mp - M2))) < 1e-10,
          f"mean {np.max(np.abs(mu - mu2)):.1e}, covariance {np.max(np.abs(Mp - M2)):.1e}")
    # He & Litterman's identity: with no views, the equilibrium weights ARE the market
    w0 = bl.weights(Pi, Sigma, np.zeros((n, n)), 2.5)
    check("with no views the optimal portfolio is the market portfolio", float(np.max(np.abs(w0 - w_mkt))) < 1e-10,
          f"{np.max(np.abs(w0 - w_mkt)):.1e}")
    mu_lo, _ = bl.posterior(Pi, Sigma, P, Q, bl.omega(P, Sigma, 0.05, 1e-8), 0.05)
    mu_hi, _ = bl.posterior(Pi, Sigma, P, Q, bl.omega(P, Sigma, 0.05, 1e8), 0.05)
    check("confidence -> 0 gives the prior back", float(np.max(np.abs(mu_lo - Pi))) < 1e-6,
          f"{np.max(np.abs(mu_lo - Pi)):.1e}")
    check("confidence -> infinity imposes the views exactly (P mu = Q)",
          float(np.max(np.abs(P @ mu_hi - Q))) < 1e-6, f"{np.max(np.abs(P @ mu_hi - Q)):.1e}")
    # the posterior mean is between the prior and the views, view by view
    between = np.all((P @ mu - P @ Pi) * (Q - P @ Pi) >= 0) and np.all(np.abs(P @ mu - P @ Pi) <= np.abs(Q - P @ Pi))
    check("each view's posterior sits between the equilibrium and the view", bool(between))


def test_forecast():
    print("\nthe rough-volatility forecast")
    prev = None
    for H in (0.02, 0.10, 0.30, 0.49):
        a = rf.gjr_weights(H, 5, 504)
        mean_lag = float(np.arange(1, len(a) + 1) @ a)
        ok = abs(a.sum() - 1) < 1e-12 and bool(np.all(a > 0))
        check(f"H = {H}: the GJR weights are positive and sum to one", ok, f"mean lag {mean_lag:.1f} days")
        if prev is not None:
            check(f"...and reach less far back than at H = {prev[0]}", mean_lag < prev[1],
                  f"{mean_lag:.1f} vs {prev[1]:.1f} days")
        prev = (H, mean_lag)
    a = rf.gjr_weights(0.49, 5, 504)
    check("at H -> 1/2 the forecast is (nearly) the last observation alone", a[0] > 0.97, f"a_1 = {a[0]:.3f}")
    y = np.arange(20.0)
    w = np.array([0.5, 0.3, 0.2])
    f = rf.forecast_log(y, w)
    check("the predictor is the weighted sum of the past, aligned one step ahead",
          abs(f[10] - (0.5 * y[9] + 0.3 * y[8] + 0.2 * y[7])) < 1e-12 and not np.isfinite(f[2]),
          f"f[10] = {f[10]:.3f}")
    # the variogram estimator on exact fBm with known noise
    rng = np.random.default_rng(3)
    for H in (0.08, 0.25):
        est = [rf.variogram_hurst(fbm.fbm(4000, H, rng=rng) + 0.1 * rng.standard_normal(4000), noise_var=0.01)["H"]
               for _ in range(6)]
        check(f"the variogram estimator recovers a planted H = {H} from noisy exact fBm",
              abs(float(np.mean(est)) - H) < 0.03, f"mean {np.mean(est):.3f}, sd {np.std(est, ddof=1):.3f} over 6 paths")


def test_backtest():
    print("\nthe backtest")
    E = bt.Engine(M)
    cfg = bt.Config()
    r = E.run(cfg, bt.START, bt.START + 200)
    # no look-ahead: replace every return from day t on and the run up to t is unchanged
    cut = bt.START + 100
    M2 = dm.Market(M.spec, M.F.copy(), M.RV.copy(), M.VOL.copy(), M.R.copy(), M.B, M.size, M.value, M.mom_load,
                   M.idio, M.cap0, M.regime)
    rng = np.random.default_rng(1)
    M2.R[cut:] = rng.standard_normal(M2.R[cut:].shape) * 0.02
    M2.F[cut:] = rng.standard_normal(M2.F[cut:].shape) * 0.01
    M2.RV[cut:] = np.abs(rng.standard_normal(M2.RV[cut:].shape)) * 1e-4
    E2 = bt.Engine(M2)
    r2 = E2.run(cfg, bt.START, bt.START + 200)
    k = cut - bt.START
    check("no look-ahead: replacing every return from day t on leaves the run up to t identical",
          float(np.max(np.abs(r["net"][:k] - r2["net"][:k]))) == 0.0,
          f"worst difference {np.max(np.abs(r['net'][:k] - r2['net'][:k])):.1e} over {k} days")
    check("...and it does change the run after t", float(np.max(np.abs(r["net"][k:] - r2["net"][k:]))) > 0)
    check("cost = turnover x the rate", abs(float(r["cost"].sum()) - float(r["turnover"].sum()) * cfg.cost_bp * 1e-4) < 1e-15)
    # the cap is on the TARGET; a held position drifts with returns until the next rebalance
    tight = bt.Config(gross_cap=0.5)
    tgt = [float(np.abs(E.target(t, tight)).sum()) for t in E.days[:20]]
    held = E.run(tight, bt.START, bt.START + 60)
    check("the gross cap binds on every rebalance target, and a held position drifts only slightly",
          max(tgt) <= 0.5 + 1e-9 and float(np.max(held["gross_exposure"])) < 0.52,
          f"worst target {max(tgt):.4f}, worst held {np.max(held['gross_exposure']):.4f}")
    free = E.run(bt.Config(band=0.0), bt.START, bt.START + 300)
    banded = E.run(bt.Config(band=0.01), bt.START, bt.START + 300)
    check("a no-trade band cuts turnover", banded["turnover"].sum() < free["turnover"].sum(),
          f"{banded['turnover'].sum():.1f} vs {free['turnover'].sum():.1f}")
    lo = E.run(bt.Config(H=0.02), bt.START, bt.START + 400)
    hi = E.run(bt.Config(H=0.45), bt.START, bt.START + 400)
    check("a lower H makes a smoother, cheaper exposure path (the 'jaggedness' knob works)",
          lo["turnover"].sum() < hi["turnover"].sum() and np.std(np.diff(lo["beta_true"])) < np.std(np.diff(hi["beta_true"])),
          f"turnover {lo['turnover'].sum():.1f} vs {hi['turnover'].sum():.1f}; daily exposure change sd "
          f"{np.std(np.diff(lo['beta_true'])):.4f} vs {np.std(np.diff(hi['beta_true'])):.4f}")
    w = E.target(bt.START, cfg)
    check("the target weights are finite and the views tilt them away from the market",
          bool(np.all(np.isfinite(w))) and float(np.max(np.abs(w / w.sum() - M.w_mkt(bt.START)))) > 1e-3)


def test_factor_fit():
    print("\nthe factor regression")
    rng = np.random.default_rng(4)
    n = 600
    F = rng.standard_normal((n, 4)) * 0.01
    y = 0.0002 + F @ np.array([0.8, -0.3, 0.2, 0.1]) + 0.004 * rng.standard_normal(n)
    a, se, _ = ff.ols_nw(y, F)
    check("the internal regression recovers the planted loadings within 3 SE",
          bool(np.all(np.abs((a[1:] - np.array([0.8, -0.3, 0.2, 0.1])) / se[1:]) < 3)),
          ", ".join(f"{v:+.3f}" for v in a[1:]))
    check("Newey-West lag follows floor(4 (n/100)^(2/9))", ff.nw_lag(1008) == 6 and ff.nw_lag(252) == 4,
          f"n=1008 -> {ff.nw_lag(1008)}, n=252 -> {ff.nw_lag(252)}")
    if ff.FALSIFY:
        fa = ff.fit(y, F, use_falsify=True)
        fb = ff.fit(y, F, use_falsify=False)
        d = max(float(np.max(np.abs(fa["betas"] - fb["betas"]))), abs(fa["alpha"] - fb["alpha"]),
                float(np.max(np.abs(fa["se"] - fb["se"]))))
        check("falsify's factor regression and this package's agree to rounding", d < 1e-10,
              f"worst difference {d:.1e} (falsify at {ff.FALSIFY})")
        d2, src = ff.deflated_sharpe(y, [0.01, 0.02, 0.03, 0.04])
        check("falsify's deflated Sharpe returns a probability", src == "falsify" and 0 <= d2 <= 1, f"DSR {d2:.3f}")
    else:
        check("falsify not present: the internal fallbacks are used", ff.fit(y, F)["source"] == "internal")


def test_dual_kalman():
    print("\nthe dual Kalman filter")
    g = dk.Gaussian(np.array([1.0, 2.0]), np.array([[0.5, 0.1], [0.1, 0.4]]))
    H = np.array([[1.0, 0.5]])
    R = np.array([[0.2]])
    z = np.array([2.5])
    upd, _ = dk.kalman_update(g, z, H, R)
    S = H @ g.P @ H.T + R
    K = g.P @ H.T @ np.linalg.inv(S)
    check("the update equals the textbook gain form",
          float(np.max(np.abs(upd.x - (g.x + K @ (z - H @ g.x))))) < 1e-12
          and float(np.max(np.abs(upd.P - (np.eye(2) - K @ H) @ g.P))) < 1e-12)
    check("the update reduces the covariance", float(np.trace(upd.P)) < float(np.trace(g.P)))
    E = bt.Engine(M)
    r = dk.run(E, bt.Config(), ("H", "confidence", "delta"), 0.8, bt.START, bt.START + 21 * 12)
    names = r["names"]
    th = np.array([p["theta"] for p in r["path"]])
    inside = all(dk.BOUNDS[n][0] - 1e-12 <= v <= dk.BOUNDS[n][1] + 1e-12 for row in th for n, v in zip(names, row))
    check("every parameter stays inside its bounds", inside)
    check("the state filter's market beta tracks the per-period fits",
          abs(np.mean([p["beta_mkt"] for p in r["path"]]) - np.mean([p["b_mkt_fit"] for p in r["path"]])) < 0.15,
          f"filter {np.mean([p['beta_mkt'] for p in r['path']]):.3f}, fits {np.mean([p['b_mkt_fit'] for p in r['path']]):.3f}")
    # the Jacobian's own sign: more risk aversion means less exposure, a rougher model costs more
    J = np.array(r["path"][2]["J"])
    check("the measured Jacobian has the signs the model implies (d beta / d log delta < 0, d cost / dH > 0)",
          J[0, names.index("delta")] < 0 and J[1, names.index("H")] > 0,
          f"d beta/d log delta {J[0, names.index('delta')]:+.2f}, d cost/dH {J[1, names.index('H')]:+.1f}")
    moves = [abs(p["beta_mkt"] - 0.8) for p in r["path"]]
    check("the filter's tracking error falls over the run", np.mean(moves[-4:]) < np.mean(moves[:4]),
          f"first four periods {np.mean(moves[:4]):.3f}, last four {np.mean(moves[-4:]):.3f}")


if __name__ == "__main__":
    print("=" * 74)
    print("Session L trial -- Black-Litterman, FF4 and the dual Kalman filter")
    print("=" * 74)
    test_market()
    test_database()
    test_black_litterman()
    test_forecast()
    test_backtest()
    test_factor_fit()
    test_dual_kalman()
    print("\n" + "=" * 74)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    print("=" * 74)
    sys.exit(1 if FAIL else 0)
