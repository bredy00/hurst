"""
Session G -- recording real IBKR chains and history, verified offline.

Everything runs against fake_ib.py: EClient's outbound calls are answered
locally and callbacks arrive on a worker thread, as TWS's reader thread would
deliver them. Four real-data traps are pinned here, each found while building
the recorder, before a single real quote had been seen:

  1  delayed data sends greeks on tick 83, which v3 dropped (no IVs at all)
  2  IBKR vega is per vol point; unconverted, iv_error is 100x too big
  3  a request id reused across cycles lets a late tick corrupt another contract
  4  tau used local wall-clock time as UTC (checked in test_core.py)

    python test_recording.py
"""

import contextlib
import datetime
import io
import json
import math
import pathlib
import sys
import tempfile
import time

import numpy as np

import fake_ib
import record_chains as rc
import record_history as rh
import sources.history as hist
import sources.replay as replay
import volatility_surface_3 as v3
import volsurf_core as vc

PASS, FAIL = [], []
TODAY = datetime.date(2026, 9, 15)


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))


@contextlib.contextmanager
def offline():
    """No rate limiting, no TCP probe, and the sweeps' chatter captured."""
    acquire, probe = v3.RateLimiter.acquire, v3.probe_tws
    v3.RateLimiter.acquire = lambda self: None
    v3.probe_tws = lambda *a, **k: True
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            yield buf
    finally:
        v3.RateLimiter.acquire, v3.probe_tws = acquire, probe


def fake_app(mdt=1, **kw):
    cls = fake_ib.make_fake(v3.LiveSurfaceApp, **kw)
    return v3.connect_app("SPY", port=4001, market_data_type=mdt,
                          app_factory=cls, spot_timeout=2.0)


# --- expiry selection ---------------------------------------------------------
def test_expiry_selection():
    print("\nExpiries by target maturity, not the first n")
    exps = fake_ib.spy_like_expirations(TODAY)
    chosen = v3.select_expiries_by_target(exps, TODAY, rc.TARGET_DAYS, 1)
    dte = [(vc.parse_ib_date(e) - TODAY).days for e in chosen]
    check("unique and sorted", chosen == sorted(set(chosen)))
    check("never inside a day", min(dte) >= 1, f"min {min(dte)} d")
    check("spans the term structure: 12+ expiries out past 300 days",
          len(chosen) >= 12 and max(dte) >= 300, f"{len(chosen)} expiries, {dte}")
    first_n = [e for e in exps if (vc.parse_ib_date(e) - TODAY).days >= 1][:16]
    span = (vc.parse_ib_date(first_n[-1]) - TODAY).days
    check("the old 'first n' rule would have stopped inside a month on dailies",
          span < 30, f"first 16 expiries end at {span} d")


# --- tick handling ------------------------------------------------------------
def test_ticks_and_units():
    print("\nDelayed greeks, vega units, unset sentinels")
    app = v3.LiveSurfaceApp()
    app.id_map[7] = ("20261016", 650.0, "C")
    app.tickOptionComputation(7, 83, 0, 0.15, 0.5, 12.0, 0.0, 0.01, 0.744, -0.05, 650.0)
    q = app.quotes.get(7)
    check("tick 83 (delayed model greeks) now records an IV", q is not None and q.iv == 0.15)
    check("vega stored per 1.00 of vol (IBKR sends per vol point)",
          q is not None and abs(q.vega - 74.4) < 1e-9, f"{q.vega if q else None}")
    app.id_map[8] = ("20261016", 660.0, "C")
    app.tickOptionComputation(8, 13, 0, 0.14, 0.4, 8.0, 0.0, 0.01, 1.7976931348623157e308, -0.05, 650.0)
    check("Double.MAX_VALUE vega is treated as unset, not as a huge vega",
          not np.isfinite(app.quotes[8].vega))

    # A 30-day 25-delta put with a 3-cent half-spread and IBKR vega 0.40
    q = v3.OptionQuote("20261016", 620.0, "P", iv=0.19, vega=0.40 * v3.IB_VEGA_PER_UNIT_VOL,
                       bid=4.17, ask=4.23)
    raw = v3.OptionQuote("20261016", 620.0, "P", iv=0.19, vega=0.40, bid=4.17, ask=4.23)
    check("with the conversion a tight OTM put is usable (0.075 vol points)",
          q.usable(), f"iv_error {q.iv_error:.5f}")
    check("without it the same quote would have been thrown away (7.5 'vol points')",
          not raw.usable(), f"iv_error {raw.iv_error:.4f}")


# --- a recording cycle --------------------------------------------------------
def test_record_cycle():
    print("\nRecording cycles against the fake exchange")
    targets = (1, 7, 30, 90)
    with tempfile.TemporaryDirectory() as d, offline():
        app = fake_app()
        exps = v3.select_expiries_by_target(app.expirations, vc.new_york_date(
            datetime.datetime.now(datetime.timezone.utc)), targets, 1)
        path1, st1 = rc.record_cycle(app, "SPY", exps, d, 0, max_strikes=9)
        written = path1.exists() and path1.stat().st_size > 0
        app2, ctxs2 = replay.load(path1)
        now = datetime.datetime.now(datetime.timezone.utc)

    check("a snapshot file is written", written, path1.name)
    check("every quote carries an IV", st1["quotes"] > 0 and st1["with_iv"] == st1["quotes"],
          f"{st1['with_iv']} of {st1['quotes']}")
    r2 = min(r["parity_r2"] for r in st1["expiries"].values())
    check("put-call parity holds on every expiry (r2 > 0.9999)", r2 > 0.9999, f"min r2 {r2:.6f}")
    worst_f = max(abs(r["forward"] / app.fake_forward(r["tau_days"] / 365.0) - 1)
                  for r in st1["expiries"].values())
    check("recovered forwards match the book's to 1e-4", worst_f < 1e-4, f"{worst_f:.2e}")
    worst_tau = max(abs(ctxs2[e].tau - vc.tau_years(now, vc.parse_ib_date(e))) * 365 * 86400
                    for e in ctxs2)
    check("tau is measured on the UTC clock (within a minute of now)", worst_tau < 60,
          f"worst {worst_tau:.1f} s")
    check("cycle statistics are stored in the snapshot", app2.meta.get("usable") == st1["usable"])
    pts = v3.surface_points(app2, ctxs2, max_age=1e9)
    check("the replayed snapshot yields surface points on every expiry",
          set(pts) == set(exps), f"{sorted(pts)} vs {exps}")


def test_cycles_do_not_leak():
    print("\nCycle isolation: new forwards, and late ticks cannot land on other contracts")
    targets = (7, 30)
    with tempfile.TemporaryDirectory() as d, offline():
        app = fake_app()
        today = vc.new_york_date(datetime.datetime.now(datetime.timezone.utc))
        exps = v3.select_expiries_by_target(app.expirations, today, targets, 1)
        _, st1 = rc.record_cycle(app, "SPY", exps, d, 0, max_strikes=9)
        old_grid_id = max(app.id_map)                  # a cycle-0 grid request id
        app.fake_spot = 663.0
        app.later(lambda: app.tickOptionComputation(old_grid_id, 13, 0, 9.99, 0.5, 1.0,
                                                    0.0, 0.01, 0.5, -0.05, 663.0))
        path2, st2 = rc.record_cycle(app, "SPY", exps, d, 1, max_strikes=9)
        snap2 = json.loads(pathlib.Path(path2).read_text(encoding="utf-8"))

        # Control: the same late tick when ids are reused across cycles
        app.fake_spot = 650.0
        _, _ = rc.record_cycle(app, "SPY", exps, d, 5, max_strikes=9)
        reused = max(app.id_map)
        app.later(lambda: app.tickOptionComputation(reused, 13, 0, 9.99, 0.5, 1.0,
                                                    0.0, 0.01, 0.5, -0.05, 650.0))
        time.sleep(0.05)
        corrupted = app.quotes.get(reused)

    f1 = {e: r["forward"] for e, r in st1["expiries"].items()}
    f2 = {e: r["forward"] for e, r in st2["expiries"].items()}
    ratio = [f2[e] / f1[e] for e in f1]
    check("the second cycle re-derives forwards from the moved market",
          all(abs(x - 663.0 / 650.0) < 1e-4 for x in ratio), f"F2/F1 {np.round(ratio, 6)}")
    check("a late tick for a previous cycle's request is ignored",
          all(q["iv"] != 9.99 for q in snap2["quotes"]))
    check("control: with ids reused, the same late tick lands on a live contract",
          corrupted is not None and corrupted.iv == 9.99)


def test_delayed_session():
    print("\nDelayed data end to end, and the live -> delayed fallback")
    with tempfile.TemporaryDirectory() as d, offline():
        cls = fake_ib.make_fake(v3.LiveSurfaceApp, delayed=True)
        app = v3.connect_app("SPY", port=4001, market_data_type=3, app_factory=cls, spot_timeout=2.0)
        exps = v3.select_expiries_by_target(app.expirations, vc.new_york_date(
            datetime.datetime.now(datetime.timezone.utc)), (7, 30), 1)
        _, st = rc.record_cycle(app, "SPY", exps, d, 0, max_strikes=7)

        class NoLive(fake_ib.make_fake(v3.LiveSurfaceApp)):
            def reqMktData(self, reqId, contract, *rest):
                if reqId == self.SPOT_REQ_ID and self.sent_market_data_type == 1:
                    return                         # live spot never arrives
                super().reqMktData(reqId, contract, *rest)
        app2 = v3.connect_app("SPY", port=4001, market_data_type=1, app_factory=NoLive, spot_timeout=0.6)

    check("a delayed-data session records implied vols (tick 83)",
          st["with_iv"] == st["quotes"] > 0, f"{st['with_iv']} of {st['quotes']}")
    check("the snapshot says which data it holds", st["market_data_type"] == "delayed")
    check("no live spot -> falls back to delayed and records that",
          app2.market_data_type == 3 and app2.sent_market_data_type == 3)


def test_check_mode():
    print("\n--check smoke test")
    with offline() as buf:
        app = fake_app()
        ok = rc.check(app, "SPY")
    out = buf.getvalue()
    check("the smoke test passes against a well-formed book", ok)
    check("and verifies vega units against Black-Scholes", "(consistent)" in out,
          next((l for l in out.splitlines() if "vega units" in l), "no units line"))
    lines = rc.explain_errors({354: 3, 2104: 1, 10197: 1})
    check("error codes come with an actionable hint; benign 2104 is not shown",
          any("delayed" in l for l in lines) and not any("2104" in l for l in lines)
          and any("phone" in l for l in lines))


# --- history --------------------------------------------------------------------
def test_history():
    print("\nHistory pull and the three variance views")
    with offline():
        cls = fake_ib.make_fake(rh.HistoryApp)
        app = cls()
        app.connect("127.0.0.1", 4001, 38)
        logs = []
        data = rh.pull(app, "SPY", years=1, rv_months=1, pause=0.0, log=logs.append)
    check("daily, IV and HV series arrive", len(data["daily"]) > 200 and len(data["iv30"]) > 200
          and len(data["hv30"]) > 200, f"{len(data['daily'])} / {len(data['iv30'])} / {len(data['hv30'])}")
    check("an unpermissioned VIX is skipped, not fatal",
          data["vix"] == [] and any("skipped" in l for l in logs))
    weeks = int(round(1 * 52 / 12))
    check("5-minute bars are pulled a week at a time", len([r for r in app.requests
                                                           if isinstance(r[1], tuple) and r[1][3] == "5 mins"]) == weeks)
    rv = hist.realised_variance(data["bars5m"])
    check("realised variance per day recovers 0.15^2 to 15%",
          len(rv["date"]) >= 15 and abs(rv["rv"].mean() / 0.0225 - 1) < 0.15,
          f"{len(rv['date'])} days, mean {rv['rv'].mean():.5f}")
    a = {"date": "20260901  09:35:00", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 0}
    b = dict(a, date="20260901  09:40:00")
    c = dict(a, date="20260901  09:45:00")
    check("overlapping chunks are stitched once each, in order",
          [x["date"] for x in hist.merge_bars([[b, a], [c, b]])] == [a["date"], b["date"], c["date"]])

    rng = np.random.default_rng(11)
    n_days, steps, sigma = 4000, 390, 0.20
    paths = np.cumsum(rng.normal(0, sigma / math.sqrt(252 * steps), (n_days, steps)), axis=1)
    paths = np.concatenate([np.zeros((n_days, 1)), paths], axis=1)
    px = 100 * np.exp(paths)
    # A high/low read off 390 samples a day is short of the continuous one by
    # ~0.5826 sigma sqrt(dt) at each end (Broadie-Glasserman-Kou), which biases
    # Garman-Klass ~8% low. Correct the simulation, not the estimator: real daily
    # highs and lows come from every trade.
    shift = 0.5826 * sigma / math.sqrt(252 * steps)
    raw = hist.garman_klass(px[:, 0], px.max(1), px.min(1), px[:, -1])
    gk = hist.garman_klass(px[:, 0], px.max(1) * math.exp(shift), px.min(1) * math.exp(-shift), px[:, -1])
    se = gk.std(ddof=1) / math.sqrt(n_days)
    check("Garman-Klass is unbiased for a driftless diffusion (within 4 SE)",
          abs(gk.mean() - sigma ** 2) < 4 * se,
          f"{gk.mean():.5f} vs {sigma**2:.5f} (se {se:.5f}); without the continuity "
          f"correction {raw.mean():.5f}, {100*(raw.mean()/sigma**2-1):+.1f}%")
    cc = 252 * np.log(px[:, -1] / px[:, 0]) ** 2
    eff = cc.var() / gk.var()
    check("...and several times more efficient than a squared daily return", eff > 4.0,
          f"variance ratio {eff:.1f} (theory 7.4 continuous)")

    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "h.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        back = hist.load(p)
    check("history file round-trips", back["bars5m"] == data["bars5m"])


def test_pipeline_synthetic():
    print("\nThe real-data pipeline, end to end on a recording with a known answer")
    import run_real_data as rrd
    with tempfile.TemporaryDirectory() as d:
        t0 = time.perf_counter()
        truth, snap, history = rrd.synthetic_inputs(d)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report, out_dir = rrd.run(snap, history, "hybrid", out_dir=pathlib.Path(d) / "report", synthetic=True,
                                      quick=True)
        wrote = all((out_dir / f).exists() for f in ("report.json", "report.md", "figure.png"))
    c = report["chain"]["checks"]
    check("recorded chain passes the unit and convention checks (vega ratio, IV under our tau)",
          c["vega_units_ok"] and abs(c["iv_mid_minus_ibkr_median_vp"]) < 0.05 and not c["weak_parity"],
          f"vega ratio {c['vega_ratio_median']:.3f}, IV gap {c['iv_mid_minus_ibkr_median_vp']:+.3f} vp")
    sf = report["chain"]["surface"]
    check("the surface keeps the quotes and builds short-end anchors", sf["quotes"] > 50 and sf["anchors"] >= 2,
          f"{sf['quotes']} quotes, {sf['anchors']} anchors")
    fp = report["H"]["filter_profile"]
    # Session L: judged against the estimator's MEASURED spread on 500 days (0.079 over 21
    # seeds), not the profile curvature's SE, which reads a median 0.028 and understates it
    # threefold -- see the same correction in test_filters.test_learn_h. What this check is
    # for is that the pipeline runs end to end and lands near the planted answer; the
    # estimator's own precision is measured there.
    check("the history filter recovers the planted H = 0.10 within 2 measured sd (0.079)",
          abs(fp["H"] - truth.H) < 2 * 0.079,
          f"H {fp['H']:.3f} (profile se {fp['se']:.3f})  ({time.perf_counter()-t0:.0f}s)")
    check("report.json, report.md and figure.png are written", wrote)


if __name__ == "__main__":
    print("=" * 74)
    print("Session G -- recording IBKR chains and history (offline, fake exchange)")
    print("=" * 74)
    t0 = time.perf_counter()
    test_expiry_selection()
    test_ticks_and_units()
    test_record_cycle()
    test_cycles_do_not_leak()
    test_delayed_session()
    test_check_mode()
    test_history()
    test_pipeline_synthetic()
    print("\n" + "=" * 74)
    print(f"{len(PASS)} passed, {len(FAIL)} failed   ({time.perf_counter()-t0:.0f}s)")
    for f in FAIL:
        print(f"  FAILED: {f}")
    print("=" * 74)
    sys.exit(1 if FAIL else 0)
