"""
The Session L trial (25 September 2026): calibrate the risk model's Hurst index -- and the
Black-Litterman views' confidence and risk aversion -- with a dual Kalman filter, so that a
Black-Litterman portfolio holds a target market exposure at low cost. Is that practical?

Akin's idea, as stated: build a default portfolio with Black-Litterman at all times on a
demo market and database; run a Fama-French four-factor regression on it; report the betas
per signal period from backtests with falsify; feed them to a dual Kalman filter that
calibrates H ("the jaggedness") toward a desired market exposure while cutting costs; try the
same on the confidence matrix Omega and on the equilibrium.

The pieces (package portfolio/, revertible as a unit):
  demo_market.py      12 years, 40 assets, MKT/SMB/HML/MOM with rough factor volatility from
                      exact fGn; no alpha anywhere; SQLite
  backtest.py         weekly Black-Litterman rebalancing, 10 bp a unit of turnover; the risk
                      model forecasts factor volatility with the rough predictor at index H
  factor_fit.py       FF4 with Newey-West errors, deflated Sharpe and CSCV from falsify
  dual_kalman.py      the state filter (exposures) and the parameter filter (theta)

Two markets: stationary (H_true = 0.10 throughout) and shift (H_true 0.08, then 0.25 from
the train/test boundary on, so the whole test period runs at the new roughness). Target: FF4
market beta 0.8. Exchange rate,
fixed before any run: 0.05 of beta error weighs like 0.25% a year of cost.

Train on years 5-8 (the first four are warm-up for the risk model), test on 9-12.
Strategies scored on the test years:
  S0 default            H = 0.10, c = 1, delta = 2.5 (the textbook settings)
  S1 statistical H      H estimated from the training years' log realised variance (the
                        variogram with the known RV noise); delta set on the training years
                        to hit the target
  S1r rolling H         S1, with H re-estimated every signal period from the last two years
  S2 grid-best          the best of 54 fixed (H, c, delta) on the training objective
  S3 dual Kalman        the filter on (H, log c, log delta), started at S0 and run throughout
  S3e ...+ equilibrium  the filter also moving log delta_eq, Pi's scale
  S4 no-trade band      S1 with a band sized on the training years to S3's cost

Pre-registered reading (written before the test years were scored):
  the idea is practical if S3 has the lowest test objective of S1, S2, S4 in BOTH markets,
  with its filter stable (parameters off their bounds most of the time, the beta channel's
  normalised innovation with mean near 0 and sd near 1; the cost channel's cannot be, since
  a zero cost is a goal, not a measurement -- noted on the dry run, before any scoring);
  if a no-trade band (S4) matches it at the same cost, H is not the lever to pull for costs;
  and falsify's deflation says how much of any Sharpe difference is the search.

    python study_trial_bl_hurst.py          (~10 minutes)
    -> captures/trial/demo_market_<scenario>.sqlite (gitignored), captures/trial_bl_hurst.json,
       captures/trial_bl_hurst.log, captures/trial_bl_hurst.png
"""

import datetime
import json
import math
import pathlib
import sqlite3
import time

import numpy as np

import models.fbm as fbm
import portfolio.backtest as bt
import portfolio.demo_market as dm
import portfolio.dual_kalman as dk
import portfolio.factor_fit as ff
import portfolio.rough_forecast as rf

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "captures" / "trial_bl_hurst.json"
DB_DIR = ROOT / "captures" / "trial"
SCENARIOS = {"stationary": dm.MarketSpec(), "shift": dm.MarketSpec(H_true=(0.08, 0.25))}
TARGET = 0.8
EPS_BETA, EPS_COST = 0.05, 0.25
SPLIT = bt.START + 4 * 252
GRID = [(H, c, d) for H in (0.02, 0.05, 0.10, 0.20, 0.30, 0.49) for c in (0.25, 1.0, 4.0)
        for d in (2.5, 3.0, 3.75)]
NAMES3 = ("H", "confidence", "delta")
NAMES4 = ("H", "confidence", "delta", "delta_eq")


def say(msg):
    print(msg, flush=True)


class Ledger:
    """Every configuration evaluated, in the database: the N falsify deflates by."""

    def __init__(self, path):
        self.con = sqlite3.connect(str(path))
        for t in ("runs", "portfolio_daily", "period_fits", "kalman"):
            self.con.execute(f"DELETE FROM {t}")
        self.n = 0

    def record(self, strategy, cfg, phase, res, fits):
        self.n += 1
        run = self.n
        params = {k: getattr(cfg, k) for k in ("H", "confidence", "delta", "delta_eq", "band", "tau", "cost_bp")} \
            if cfg is not None else {}
        self.con.execute("INSERT INTO runs VALUES (?,?,?,?,?)",
                         (run, strategy, json.dumps(params), phase, datetime.datetime.now().isoformat()))
        self.con.executemany("INSERT INTO portfolio_daily VALUES (?,?,?,?,?,?,?,?)",
                             [(run, int(d), float(g), float(c), float(n), float(t), float(b), float(x))
                              for d, g, c, n, t, b, x in zip(res["days"], res["gross"], res["cost"], res["net"],
                                                             res["turnover"], res["beta_true"], res["gross_exposure"])])
        self.con.executemany("INSERT INTO period_fits VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                             [(run, f["period"], f["source"], f["first_day"], f["n"], f["alpha"], *map(float, f["betas"]),
                               f["se_alpha"], *map(float, f["se"])) for f in fits])
        return run

    def kalman(self, run, path, names):
        rows = []
        for p in path:
            th = dict(zip(names, p["theta"]))
            sd = dict(zip(names, p["sd"]))
            rows.append((run, p["period"], th.get("H"), th.get("confidence"), th.get("delta"), th.get("delta_eq"),
                         sd.get("H"), sd.get("confidence"), sd.get("delta"), sd.get("delta_eq"),
                         p["beta_mkt"], p["beta_mkt_sd"], p["cost_pct"]))
        self.con.executemany("INSERT INTO kalman VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)

    def close(self):
        self.con.commit()
        self.con.close()


def sharpe(x):
    return float(np.mean(x) / np.std(x, ddof=1) * math.sqrt(252)) if np.std(x) > 0 else 0.0


def score(res, fits, lo, hi):
    """The objective and its parts on days [lo, hi): per-period FF4 beta vs the target, cost."""
    sel = (res["days"] >= lo) & (res["days"] < hi)
    b = np.array([f["betas"][0] for f in fits if lo <= f["first_day"] < hi])
    cost = 100 * 252 * float(res["cost"][sel].mean())
    true_b = []
    days = res["days"][sel]
    bt_ = res["beta_true"][sel]
    for p0 in range(lo, hi, bt.PERIOD):
        s = (days >= p0) & (days < p0 + bt.PERIOD)
        if s.sum() >= 10:
            true_b.append(bt_[s].mean())
    true_b = np.array(true_b)
    net = res["net"][sel]
    ffit = ff.fit(net, np.asarray(res["_F"])[sel])
    J = float(np.mean((b - TARGET) ** 2) / EPS_BETA ** 2 + cost ** 2 / EPS_COST ** 2)
    return {"J": J, "beta_rmse": float(np.sqrt(np.mean((b - TARGET) ** 2))), "beta_mean": float(b.mean()),
            "true_beta_rmse": float(np.sqrt(np.mean((true_b - TARGET) ** 2))), "cost_pct": cost,
            "turnover": float(res["turnover"][sel].sum() / sel.sum() * 252), "sharpe_net": sharpe(net),
            "gross_exposure": float(res["gross_exposure"][sel].mean()),
            "alpha_pct": 100 * 252 * ffit["alpha"], "alpha_t": ffit["alpha"] / ffit["se_alpha"],
            "exposure_path_H": exposure_roughness(res, lo, hi)}


def exposure_roughness(res, lo, hi):
    """Shevchenko's QV estimator on the weekly exposure path (the 'jaggedness' of the portfolio)."""
    days = res["days"]
    sel = (days >= lo) & (days < hi) & ((days - bt.START) % bt.REBALANCE_EVERY == 0)
    path = res["beta_true"][sel]
    return float(fbm.quadratic_variation_hurst(path)) if len(path) > 20 else float("nan")


def run_fixed(E, cfg, lo, hi, use_falsify=True):
    res = E.run(cfg, lo, hi)
    res["_F"] = E.m.F[res["days"]]
    fits = ff.period_fits(res["net"], E.m.F, res["days"], [p for p in E.period_starts if lo <= p < hi], use_falsify)
    return res, fits


def calibrate_delta(E, cfg, lo, hi, ledger, deltas=np.linspace(2.0, 4.5, 11)):
    """The delta whose training-years mean FF4 beta is closest to the target (a 1-D search, in the ledger)."""
    best = None
    tries = []
    for d in deltas:
        c = bt.Config(**{**cfg.__dict__, "delta": float(d)})
        res, fits = run_fixed(E, c, lo, hi)
        ledger.record("delta-search", c, "train", res, fits)
        b = np.mean([f["betas"][0] for f in fits])
        tries.append((float(d), float(b), sharpe(res["net"])))
        if best is None or abs(b - TARGET) < best[1]:
            best = (float(d), abs(b - TARGET))
    return best[0], tries


def band_for_cost(E, cfg, lo, hi, target_cost):
    """No-trade band (bisection) whose training-years cost matches target_cost."""
    a, b = 0.0, 0.05
    for _ in range(14):
        mid = 0.5 * (a + b)
        res, _ = run_fixed(E, bt.Config(**{**cfg.__dict__, "band": mid}), lo, hi, use_falsify=False)
        c = 100 * 252 * float(res["cost"].mean())
        if c > target_cost:
            a = mid
        else:
            b = mid
    return 0.5 * (a + b)


def rolling_h_cfgs(E, cfg, lo, hi, noise_var):
    """S1r: H re-estimated at each signal period from the last two years of log RV."""
    by_period = {}
    for p0 in E.period_starts:
        if lo <= p0 < hi:
            y = E.logrv[p0 - 504:p0, 0]
            by_period[int(p0)] = min(max(rf.variogram_hurst(y, noise_var=noise_var)["H"], 0.02), 0.49)
    starts = np.array(sorted(by_period))

    def path(t):
        p0 = starts[starts <= t][-1]
        return bt.Config(**{**cfg.__dict__, "H": by_period[int(p0)]})
    return path, by_period


def scenario(name, spec):
    t0 = time.perf_counter()
    DB_DIR.mkdir(parents=True, exist_ok=True)
    db = DB_DIR / f"demo_market_{name}.sqlite"
    market = dm.simulate(spec)
    dm.write(market, db)
    m = dm.load(db)                                         # the trial runs on what the database holds
    assert np.array_equal(m.R, market.R) and np.array_equal(m.F, market.F)
    E = bt.Engine(m)
    L = Ledger(db)
    T = m.T
    noise = spec.rv_noise ** 2
    say(f"\n=== {name}: {m.N} assets, {T} days, H_true {spec.H_true} (switch at day {int(T * spec.shift_at)}), "
        f"database {db.name} ({db.stat().st_size / 1e6:.1f} MB); train [{bt.START}, {SPLIT}), test [{SPLIT}, {T})")
    out = {"spec": {k: v for k, v in spec.__dict__.items()}, "db": str(db.relative_to(ROOT))}

    # ---- training: the grid (the "Hurst parameter optimisation" by search), all in the ledger
    grid_rows = []
    grid_nets = []
    for H, c, d in GRID:
        cfg = bt.Config(H=H, confidence=c, delta=d)
        res, fits = run_fixed(E, cfg, bt.START, SPLIT)
        L.record("grid", cfg, "train", res, fits)
        s = score(res, fits, bt.START, SPLIT)
        grid_rows.append({"H": H, "confidence": c, "delta": d, **s})
        grid_nets.append(res["net"])
    grid_nets = np.column_stack(grid_nets)
    best = min(range(len(GRID)), key=lambda i: grid_rows[i]["J"])
    H2, c2, d2 = GRID[best]
    say(f"grid ({len(GRID)} configurations, training years): best J {grid_rows[best]['J']:.2f} at H = {H2}, c = {c2}, "
        f"delta = {d2} (beta RMSE {grid_rows[best]['beta_rmse']:.3f}, cost {grid_rows[best]['cost_pct']:.2f}%/yr)")
    by_h = {}
    for r in grid_rows:
        by_h.setdefault(r["H"], []).append(r)
    say("  by H (best over c, delta): " + "; ".join(
        f"H {h}: J {min(z['J'] for z in rs):.1f}, cost {min(rs, key=lambda z: z['J'])['cost_pct']:.2f}%"
        for h, rs in by_h.items()))

    # ---- S1: the statistical H, delta set on the training years
    Hs = min(max(rf.variogram_hurst(E.logrv[:SPLIT, 0], noise_var=noise)["H"], 0.02), 0.49)
    d1, d_tries = calibrate_delta(E, bt.Config(H=Hs), bt.START, SPLIT, L)
    say(f"S1: variogram H of the training years' log RV (noise known) {Hs:.3f}; delta for beta {TARGET}: {d1:.2f}")

    # ---- S3: the dual Kalman filters, from the start of the training years to the end
    kal = {}
    for label, names in (("S3 dual Kalman", NAMES3), ("S3e + equilibrium", NAMES4)):
        tk = time.perf_counter()
        r = dk.run(E, bt.Config(), names, TARGET, bt.START, T, EPS_BETA, EPS_COST)
        r["_F"] = m.F[r["days"]]
        run = L.record(label, None, "train+test", r, r["fits"])
        L.kalman(run, r["path"], names)
        kal[label] = r
        th_split = [p for p in r["path"] if p["first_day"] <= SPLIT][-1]["theta"]
        th_end = r["path"][-1]["theta"]
        fmt = lambda th: ", ".join(f"{n} {v if n == 'H' else math.exp(v):.3f}" for n, v in zip(names, th))
        zb = np.array([p["z_beta"] for p in r["path"]])
        at_bound = {n: float(np.mean([abs(p["theta"][i] - dk.BOUNDS[n][0]) < 1e-9 or abs(p["theta"][i] - dk.BOUNDS[n][1]) < 1e-9
                                      for p in r["path"]])) for i, n in enumerate(names)}
        say(f"{label} ({time.perf_counter() - tk:.0f}s): theta at the split {fmt(th_split)}; at the end {fmt(th_end)}; "
            f"beta-channel innovation mean {zb.mean():+.2f}, sd {zb.std():.2f} (0 and 1 if consistent); "
            f"share of periods at a bound: " + ", ".join(f"{n} {100 * v:.0f}%" for n, v in at_bound.items()))
        r["at_bound"], r["z_beta_sd"], r["z_beta_mean"] = at_bound, float(zb.std()), float(zb.mean())

    # ---- S4: S1 with a no-trade band at S3's training cost
    s3_train_cost = score(kal["S3 dual Kalman"], kal["S3 dual Kalman"]["fits"], bt.START, SPLIT)["cost_pct"]
    band = band_for_cost(E, bt.Config(H=Hs, delta=d1), bt.START, SPLIT, s3_train_cost)
    say(f"S4: no-trade band {band:.4f} per asset gives S1 the dual filter's training cost ({s3_train_cost:.2f}%/yr)")

    # ---- the test years
    cfgs = {"S0 default": bt.Config(), "S1 statistical H": bt.Config(H=Hs, delta=d1),
            "S2 grid-best": bt.Config(H=H2, confidence=c2, delta=d2),
            "S4 no-trade band": bt.Config(H=Hs, delta=d1, band=band)}
    test = {}
    for label, cfg in cfgs.items():
        res, fits = run_fixed(E, cfg, SPLIT, T)
        L.record(label, cfg, "test", res, fits)
        test[label] = score(res, fits, SPLIT, T)
        test[label]["_net"] = res["net"]
    path, hs_roll = rolling_h_cfgs(E, bt.Config(H=Hs, delta=d1), SPLIT, T, noise)
    res = E.run(bt.Config(H=Hs, delta=d1), SPLIT, T, cfg_path=path)
    res["_F"] = m.F[res["days"]]
    fits = ff.period_fits(res["net"], m.F, res["days"], [p for p in E.period_starts if SPLIT <= p < T])
    L.record("S1r rolling H", None, "test", res, fits)
    test["S1r rolling H"] = score(res, fits, SPLIT, T)
    test["S1r rolling H"]["_net"] = res["net"]
    for label, r in kal.items():
        test[label] = score(r, r["fits"], SPLIT, T)
        sel = r["days"] >= SPLIT
        test[label]["_net"] = r["net"][sel]
    # out-of-sample J of every grid configuration: where does the training pick land?
    grid_test_J = []
    for H, c, d in GRID:
        res_g, fits_g = run_fixed(E, bt.Config(H=H, confidence=c, delta=d), SPLIT, T, use_falsify=False)
        grid_test_J.append(score(res_g, fits_g, SPLIT, T)["J"])
    rank = float(np.mean(np.array(grid_test_J) < grid_test_J[best]))

    # ---- falsify: deflation and overfitting
    trial_sr = [np.mean(grid_nets[:, j]) / np.std(grid_nets[:, j], ddof=1) for j in range(grid_nets.shape[1])]
    dsr_s2, src = ff.deflated_sharpe(grid_nets[:, best], trial_sr)
    p = ff.pbo(grid_nets, n_blocks=16)
    psr = {}
    for label, s in test.items():
        psr[label], _ = ff.deflated_sharpe(s["_net"], [0.0])      # one trial: PSR(0)
    say(f"falsify ({src}): the grid's best training Sharpe, deflated by {len(GRID)} trials: DSR {dsr_s2:.3f}; "
        + (f"PBO (ArgMax on Sharpe, CSCV 16 blocks) {p['pbo']:.3f}, degradation {p['degradation']:+.3f}" if p else "no PBO")
        + f"; the objective-selected configuration ranks {100 * rank:.0f}% from the top of {len(GRID)} out of sample")

    say(f"\ntest years [{SPLIT}, {T}), target beta {TARGET}; J = mean((b - {TARGET})^2)/{EPS_BETA}^2 + cost^2/{EPS_COST}^2:")
    say(f"  {'strategy':22s} {'J':>8s} {'beta RMSE':>10s} {'true-beta RMSE':>15s} {'mean beta':>10s} {'cost %/yr':>10s} "
        f"{'turnover':>9s} {'net SR':>7s} {'alpha %':>8s} {'alpha t':>8s} {'PSR':>6s} {'path H':>7s}")
    for label in sorted(test, key=lambda k: test[k]["J"]):
        s = test[label]
        say(f"  {label:22s} {s['J']:8.2f} {s['beta_rmse']:10.3f} {s['true_beta_rmse']:15.3f} {s['beta_mean']:10.3f} "
            f"{s['cost_pct']:10.2f} {s['turnover']:9.1f} {s['sharpe_net']:7.2f} {s['alpha_pct']:8.2f} {s['alpha_t']:8.2f} "
            f"{psr[label]:6.3f} {s['exposure_path_H']:7.3f}")
    L.close()
    out.update({"grid": grid_rows, "grid_best": {"H": H2, "confidence": c2, "delta": d2, "index": best},
                "grid_test_J": grid_test_J, "grid_pick_oos_rank": rank,
                "S1": {"H": Hs, "delta": d1, "delta_tries": d_tries}, "S1r_H_by_period": hs_roll, "S4_band": band,
                "kalman": {k: {"path": v["path"], "names": v["names"], "at_bound": v["at_bound"],
                               "z_beta_sd": v["z_beta_sd"], "z_beta_mean": v["z_beta_mean"]} for k, v in kal.items()},
                "test": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in test.items()},
                "psr_test": psr, "falsify": {"source": src, "dsr_grid_best_train": dsr_s2, "pbo": p,
                                            "n_trials_ledger": L.n},
                "seconds": time.perf_counter() - t0})
    return out


def post_hoc(results):
    """
    NOT pre-registered: written after the test years were scored, to ask whether the dual
    filter's gap to the grid is two fixable details or the idea itself. S3c measures each
    period with constant (running-median) variances instead of its own Newey-West ones, and
    lets the parameters drift four times slower. Scored on both markets, and then -- because
    a variant chosen after seeing results is itself a search -- on a FRESH market (a new seed)
    where the grid is re-run on its own training years.
    """
    out = {}
    specs = dict(SCENARIOS, **{"confirmation (new seed)": dm.MarketSpec(seed=20260926)})
    for name, spec in specs.items():
        m = dm.simulate(spec)
        E = bt.Engine(m)
        rows = {}
        if name in results:
            g = results[name]["grid_best"]
            best = (g["H"], g["confidence"], g["delta"])
        else:
            J = []
            for H, c, d in GRID:
                res, fits = run_fixed(E, bt.Config(H=H, confidence=c, delta=d), bt.START, SPLIT, use_falsify=False)
                J.append(score(res, fits, bt.START, SPLIT)["J"])
            best = GRID[int(np.argmin(J))]
        res, fits = run_fixed(E, bt.Config(H=best[0], confidence=best[1], delta=best[2]), SPLIT, m.T)
        rows["S2 grid-best"] = score(res, fits, SPLIT, m.T)
        for label, kw in (("S3 dual Kalman", {}), ("S3c constant R, slow drift", {"r_mode": "constant", "q_scale": 0.25})):
            r = dk.run(E, bt.Config(), NAMES3, TARGET, bt.START, m.T, EPS_BETA, EPS_COST, **kw)
            r["_F"] = m.F[r["days"]]
            rows[label] = score(r, r["fits"], SPLIT, m.T)
            rows[label]["H_at_bound"] = float(np.mean([abs(p["theta"][0] - dk.BOUNDS["H"][0]) < 1e-9 for p in r["path"]]))
            zb = np.array([p["z_beta"] for p in r["path"]])
            rows[label]["z_beta_mean"], rows[label]["z_beta_sd"] = float(zb.mean()), float(zb.std())
        out[name] = {"grid_best": best, "rows": rows}
        say(f"\n[post hoc] {name}: grid pick H {best[0]}, c {best[1]}, delta {best[2]}")
        for label, s_ in rows.items():
            extra = (f"; H at its bound {100 * s_['H_at_bound']:.0f}% of periods, beta innovation "
                     f"{s_['z_beta_mean']:+.2f} +/- {s_['z_beta_sd']:.2f}") if "H_at_bound" in s_ else ""
            say(f"  {label:28s} J {s_['J']:6.2f}  beta RMSE {s_['beta_rmse']:.3f}  mean beta {s_['beta_mean']:.3f}  "
                f"cost {s_['cost_pct']:.2f}%/yr{extra}")
    return out


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ui import theme
    tok = theme.apply("light")
    cat = [theme.series(i) for i in range(8)]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.4))
    for row, name in enumerate(("stationary", "shift")):
        r = res[name]
        k = r["kalman"]["S3 dual Kalman"]
        days = np.array([p["first_day"] for p in k["path"]]) / 252
        th = np.array([p["theta"] for p in k["path"]])
        split_y = SPLIT / 252
        shifted = r["spec"]["H_true"][0] != r["spec"]["H_true"][1]
        h_true = r["spec"]["H_true"]

        # (1) H. The story is that it sits ON its lower bound, so the bound and the market's
        # own H are drawn: a flat line at 0.02 over an empty 0-0.5 axis says nothing by itself.
        a = ax[row, 0]
        lo, hi = dk.BOUNDS["H"]
        a.axhspan(0.0, lo, color=tok["neutral"], zorder=0)
        a.axhline(lo, color=tok["muted"], lw=1.0, ls="-")
        a.text(days[0], lo + 0.004, f" filter's lower bound {lo}", color=tok["ink2"], fontsize=8, va="bottom")
        a.step([days[0], split_y, days[-1]], [h_true[0], h_true[1], h_true[1]], where="post",
               color=cat[3], lw=1.4, ls="--", label="the market's own H")
        a.plot(days, th[:, 0], color=cat[0], label="H the filter chose")
        a.axvline(split_y, color=tok["axis"], lw=0.8, ls="--")
        share = 100 * r["kalman"]["S3 dual Kalman"]["at_bound"]["H"]
        a.set_title(f"{name}: the risk model's H -- at its bound on {share:.0f}% of periods")
        a.set_ylim(0, 0.32)
        a.set_ylabel("Hurst index")
        a.set_xlabel("year")
        a.legend(loc="upper right")

        a = ax[row, 1]
        a.plot(days, np.exp(th[:, 1]), color=cat[1], label="view confidence c")
        a.plot(days, np.exp(th[:, 2]), color=cat[2], label="risk aversion delta")
        a.axvline(split_y, color=tok["axis"], lw=0.8, ls="--")
        a.set_title(f"{name}: view confidence and risk aversion")
        a.set_ylabel("multiple")
        a.set_ylim(0, 4.8)
        a.legend(loc="lower right")
        a.set_xlabel("year")

        a = ax[row, 2]
        b_fit = np.array([p["b_mkt_fit"] for p in k["path"]])
        a.plot(days, b_fit, color=tok["muted"], lw=0.8, label="FF4 beta per period (falsify)")
        a.plot(days, [p["beta_mkt"] for p in k["path"]], color=cat[0], label="the state filter's")
        a.axhline(TARGET, color=cat[3], lw=1.0, ls="--", label=f"target {TARGET}")
        a.axvline(split_y, color=tok["axis"], lw=0.8, ls="--")
        a.set_title(f"{name}: FF4 market beta -- the swing is the strategy, not the setting")
        a.set_ylabel("beta on MKT")
        a.set_ylim(0.2, 2.1)
        a.set_xlabel("year")
        a.legend(loc="upper left", ncol=3, fontsize=7.5)
        if row == 0:
            a.text(days[-1], 1.78, f"per-period sd {b_fit.std(ddof=1):.2f}, of which\nmeasurement noise ~0.09  ",
                   color=tok["ink2"], fontsize=8, ha="right", va="top")
    fig.suptitle("The dual Kalman filter on the demo market  (vertical dashed line: training | test years)")
    fig.tight_layout()
    fig.savefig(ROOT / "captures" / "trial_bl_hurst.png", dpi=120)


def main():
    t0 = time.perf_counter()
    say(f"falsify: {ff.FALSIFY or 'not found -- internal fallbacks'}")
    res = {name: scenario(name, spec) for name, spec in SCENARIOS.items()}
    res["post_hoc"] = post_hoc(res)
    res["meta"] = {"target_beta": TARGET, "eps_beta": EPS_BETA, "eps_cost": EPS_COST, "split": SPLIT,
                   "start": bt.START, "falsify": ff.FALSIFY is not None}
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    plot(res)
    say(f"\n[done in {time.perf_counter() - t0:.0f}s] -> {OUT}")


if __name__ == "__main__":
    main()
