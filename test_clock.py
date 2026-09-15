"""
Session H -- the trading-time clock for short-dated options.

  calendar      Easter, NYSE holidays and early closes by rule, against the
                published 2022, 2026 and 2027 calendars
  seconds       session / overnight / weekend splits across a plain night, a weekend,
                a holiday weekend, both DST switches and an early close; the three
                parts always add up to the elapsed seconds
  variance time omega = 1 is ACT/365 exactly; any weighting integrates to one year
                over the reference year; a stopped clock does not count a weekend
  estimator     planted clocks recovered from noisy ATM total variance, with honest
                intervals; an event day found and kept out of omega; the median-
                absolute-deviation loss this replaced shown blind to the weekend spans
  replay        snapshot timestamps: the instant is recovered from the taus, and a
                timestamp without an offset is refused rather than used

    python test_clock.py
"""

import datetime
import math
import sys

import numpy as np

import calibrate.clock as ck
import volsurf_core as vc

PASS, FAIL = [], []
D, DT = datetime.date, datetime.datetime


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))


def hours(split):
    return tuple(round(s / 3600.0, 6) for s in split)


# --- calendar -----------------------------------------------------------------
def test_calendar():
    print("\nEaster, holidays and early closes by rule")
    known_easter = {2019: D(2019, 4, 21), 2024: D(2024, 3, 31), 2025: D(2025, 4, 20),
                    2026: D(2026, 4, 5), 2027: D(2027, 3, 28), 2038: D(2038, 4, 25)}
    bad = {y: vc._easter(y) for y, d in known_easter.items() if vc._easter(y) != d}
    check("Easter Sunday on six known years (2038 is the latest possible, April 25)", not bad, str(bad))
    nyse_2026 = {D(2026, 1, 1), D(2026, 1, 19), D(2026, 2, 16), D(2026, 4, 3), D(2026, 5, 25),
                 D(2026, 6, 19), D(2026, 7, 3), D(2026, 9, 7), D(2026, 11, 26), D(2026, 12, 25)}
    got = set(vc.nyse_holidays(2026))
    check("2026 NYSE holidays (Independence Day observed Friday July 3)", got == nyse_2026,
          f"missing {sorted(nyse_2026 - got)}, extra {sorted(got - nyse_2026)}")
    nyse_2027 = {D(2027, 1, 1), D(2027, 1, 18), D(2027, 2, 15), D(2027, 3, 26), D(2027, 5, 31),
                 D(2027, 6, 18), D(2027, 7, 5), D(2027, 9, 6), D(2027, 11, 25), D(2027, 12, 24)}
    got = set(vc.nyse_holidays(2027))
    check("2027 NYSE holidays (Juneteenth and Christmas observed on Fridays, July 4 on Monday)",
          got == nyse_2027, f"missing {sorted(nyse_2027 - got)}, extra {sorted(got - nyse_2027)}")
    nyse_2022 = {D(2022, 1, 17), D(2022, 2, 21), D(2022, 4, 15), D(2022, 5, 30), D(2022, 6, 20),
                 D(2022, 7, 4), D(2022, 9, 5), D(2022, 11, 24), D(2022, 12, 26)}
    got = set(vc.nyse_holidays(2022))
    check("2022: New Year's Day on a Saturday is not observed (no Dec 31, 2021 closure)",
          got == nyse_2022 and D(2021, 12, 31) not in vc.nyse_holidays(2021),
          f"missing {sorted(nyse_2022 - got)}, extra {sorted(got - nyse_2022)}")
    check("early closes 2025: July 3, November 28, December 24",
          set(vc.nyse_early_closes(2025)) == {D(2025, 7, 3), D(2025, 11, 28), D(2025, 12, 24)},
          str(sorted(vc.nyse_early_closes(2025))))
    check("early closes 2026: no July 2/3 (July 3 is the holiday); November 27, December 24",
          set(vc.nyse_early_closes(2026)) == {D(2026, 11, 27), D(2026, 12, 24)}, str(sorted(vc.nyse_early_closes(2026))))
    check("early closes 2027: December 24 is the observed Christmas, not an early close",
          set(vc.nyse_early_closes(2027)) == {D(2027, 11, 26)}, str(sorted(vc.nyse_early_closes(2027))))


# --- seconds ------------------------------------------------------------------
def test_time_split():
    print("\nSession, overnight and weekend seconds")
    cases = [
        ("a session (Wed 16 Sep 2026, EDT)", DT(2026, 9, 16, 13, 30), DT(2026, 9, 16, 20, 0), (6.5, 0.0, 0.0)),
        ("a night (Tue close -> Wed open)", DT(2026, 9, 15, 20, 0), DT(2026, 9, 16, 13, 30), (0.0, 17.5, 0.0)),
        ("a weekend (Fri close -> Mon open)", DT(2026, 9, 18, 20, 0), DT(2026, 9, 21, 13, 30), (0.0, 0.0, 65.5)),
        ("Labor Day weekend (Fri 4 -> Tue 8 Sep 2026)", DT(2026, 9, 4, 20, 0), DT(2026, 9, 8, 13, 30), (0.0, 0.0, 89.5)),
        ("DST ends (Fri 30 Oct close 20:00 UTC -> Mon 2 Nov open 14:30 UTC)", DT(2026, 10, 30, 20, 0),
         DT(2026, 11, 2, 14, 30), (0.0, 0.0, 66.5)),
        ("DST starts (Fri 6 Mar close 21:00 UTC -> Mon 9 Mar open 13:30 UTC)", DT(2026, 3, 6, 21, 0),
         DT(2026, 3, 9, 13, 30), (0.0, 0.0, 64.5)),
        ("early close (Fri 27 Nov 2026: 09:30-13:00 EST)", DT(2026, 11, 27, 14, 30), DT(2026, 11, 27, 21, 0), (3.5, 0.0, 3.0)),
        ("Thursday close -> Friday 16:00 across the early close", DT(2026, 11, 25, 21, 0), DT(2026, 11, 27, 21, 0),
         (3.5, 0.0, 44.5)),
    ]
    for name, a, b, want in cases:
        got = hours(vc.time_split(a, b))
        check(name, got == want, f"got {got} h, want {want} h")
    rng = np.random.default_rng(0)
    worst = 0.0
    base = DT(2026, 1, 1)
    for _ in range(300):
        a = base + datetime.timedelta(seconds=float(rng.uniform(0, 365 * 86400)))
        b = a + datetime.timedelta(seconds=float(rng.uniform(0, 40 * 86400)))
        worst = max(worst, abs(sum(vc.time_split(a, b)) - (b - a).total_seconds()))
    check("the three parts add up to the elapsed seconds (300 random intervals up to 40 days)", worst < 1e-6,
          f"worst {worst:.2e} s")
    aware = datetime.datetime(2026, 9, 16, 9, 30, tzinfo=datetime.timezone(datetime.timedelta(hours=-4)))
    check("aware datetimes are converted, not read as UTC", hours(vc.time_split(aware, DT(2026, 9, 16, 20, 0))) == (6.5, 0.0, 0.0))


# --- variance time --------------------------------------------------------------
def test_variance_time():
    print("\nVariance time")
    pairs = [(DT(2026, 9, 15, 15, 0), D(2026, 9, 16)), (DT(2026, 9, 18, 19, 0), D(2026, 9, 21)),
             (DT(2026, 11, 25, 16, 0), D(2026, 12, 18)), (DT(2026, 10, 30, 12, 0), D(2027, 3, 19)),
             (DT(2026, 9, 15, 15, 0), D(2027, 9, 17))]
    worst = max(abs(vc.variance_time(t, e, 1.0) / vc.tau_years(t, e) - 1.0) for t, e in pairs)
    check("omega = 1 is ACT/365 to 1e-12 (nights, weekends, holidays, DST, a year)", worst < 1e-12, f"{worst:.1e}")
    y25 = [vc.variance_time(DT(2025, 1, 1), DT(2026, 1, 1), om, ow) for om, ow in ((0.05, None), (0.3, None), (1.0, None), (0.1, 0.5))]
    check("any weighting integrates to exactly one year over the reference year",
          max(abs(v - 1) for v in y25) < 1e-12, str([round(v, 14) for v in y25]))
    y27 = [vc.variance_time(DT(2027, 1, 1), DT(2028, 1, 1), om) for om in (0.05, 0.3)]
    check("...and to within 1% over another year (2027)", max(abs(v - 1) for v in y27) < 0.01, str([round(v, 5) for v in y27]))
    yt, yo, yw = vc.reference_year_split()
    check("reference year: 1622.5 session, 3412.5 overnight, 3725 weekend/holiday hours",
          hours((yt, yo, yw)) == (1622.5, 3412.5, 3725.0), str(hours((yt, yo, yw))))
    stopped = 1e-9
    fri = vc.variance_time(DT(2026, 9, 18, 20, 0), D(2026, 9, 21), stopped)
    mon = vc.variance_time(DT(2026, 9, 21, 13, 30), D(2026, 9, 21), stopped)
    check("a stopped clock: Friday close -> Monday close is Monday's session alone", abs(fri / mon - 1) < 1e-6,
          f"{fri * 252:.6f} vs {mon * 252:.6f} (variance years x 252)")
    check("...while the calendar clock counts 3 days", abs(vc.tau_years(DT(2026, 9, 18, 20, 0), D(2026, 9, 21)) * 365 - 3.0) < 1e-9)


# --- the estimator --------------------------------------------------------------
def planted_chain(t0, days, omega, noise, rng, event=None):
    atm = {}
    for d in days:
        w = 0.16 ** 2 * vc.variance_time(t0, d, omega)
        if event is not None and d >= event[0]:
            w += event[1]
        atm[d.strftime("%Y%m%d")] = (w * (1.0 + noise * rng.standard_normal()), d)
    return atm


def test_estimator():
    print("\nThe clock from ATM total variance (planted clocks)")
    t0 = DT(2026, 9, 15, 15, 0)                               # Tuesday 11:00 New York
    days = [d for d in (D(2026, 9, 15) + datetime.timedelta(i) for i in range(22)) if vc.session_utc(d)]
    rng = np.random.default_rng(3)
    for om in (0.05, 0.2, 1.0):
        r = ck.estimate_omega([(t0, planted_chain(t0, days, om, 0.0, rng))])
        check(f"noise-free chain planted at omega = {om}: recovered to 1e-4", abs(r["omega"] / om - 1) < 1e-4,
              f"{r['omega']:.6f}")
    R = 60
    for om, noise in ((0.2, 0.01), (0.2, 0.03), (1.0, 0.02)):
        est, cover = [], 0
        for _ in range(R):
            r = ck.estimate_omega([(t0, planted_chain(t0, days, om, noise, rng))])
            est.append(r["omega"])
            cover += r["ci95"][0] <= om <= r["ci95"][1]
        est = np.array(est)
        check(f"omega = {om}, {noise:.0%} noise in w: median within 3%, spread as expected",
              abs(np.median(est) / om - 1) < 0.03 and np.std(np.log(est)) < 4.0 * noise,
              f"median {np.median(est):.4f}, log-sd {np.std(np.log(est)):.3f} over {R} chains")
        check(f"omega = {om}, {noise:.0%} noise: the 95% interval covers the truth 85-100% of the time",
              0.85 <= cover / R <= 1.0, f"{cover}/{R}")
    r = ck.estimate_omega([(t0, planted_chain(t0, days, 0.2, 0.01, rng))])
    check("calendar clock rejected when the chain is on a trading clock (omega = 0.2)", r["calendar_chi2"] > 20,
          f"calendar clock {r['calendar_chi2']:.0f} noise variances worse")
    r = ck.estimate_omega([(t0, planted_chain(t0, days, 1.0, 0.01, rng))])
    check("...and not rejected when the chain is on the calendar clock", r["calendar_chi2"] < 6, f"{r['calendar_chi2']:.1f}")

    fomc = (D(2026, 9, 16), 0.16 ** 2 / 252)                   # one extra session of variance on Sep 16
    found, with_ev, without = 0, [], []
    for _ in range(20):
        atm = planted_chain(t0, days, 0.2, 0.01, rng, event=fomc)
        r = ck.estimate_omega([(t0, atm)])
        r0 = ck.estimate_omega([(t0, atm)], detect_events=False)
        found += r["per_snapshot"][0]["events"] == ["2026-09-16"]
        with_ev.append(r["omega"])
        without.append(r0["omega"])
    check("an FOMC-day step is found at its expiry (and nowhere else) in 18+ of 20 chains", found >= 18, f"{found}/20")
    check("with the event found, omega stays within 3% of the truth", abs(np.median(with_ev) / 0.2 - 1) < 0.03,
          f"{np.median(with_ev):.4f} (ignoring the event: {np.median(without):.4f})")

    snaps = [(t0 + datetime.timedelta(hours=h), None) for h in (0, 2, 4)]
    snaps = [(t, planted_chain(t, days, 0.2, 0.02, rng)) for t, _ in snaps]
    pooled = ck.estimate_omega(snaps)
    single = ck.estimate_omega(snaps[:1])
    width = lambda r: math.log(r["ci95"][1] / r["ci95"][0])
    check("three snapshots through the session pool into a narrower interval than one",
          width(pooled) < width(single), f"log-width {width(pooled):.3f} vs {width(single):.3f}")

    # the loss this estimator replaced: MAD of log forward rates between neighbours
    def mad_loss(atm, om):
        sn = ck.Snapshot(t0, atm)
        tau = sn.S + om * (sn.O + sn.W)
        rates = np.diff(np.concatenate([[0.0], sn.w])) / np.diff(np.concatenate([[0.0], tau]))
        lr = np.log(rates)
        return float(np.median(np.abs(lr - np.median(lr))))
    atm = planted_chain(t0, days, 0.1, 0.0, rng)
    grid = np.geomspace(0.005, 1.0, 61)
    losses = np.array([mad_loss(atm, om) for om in grid])
    check("the old MAD loss is flat at zero across clocks on a noise-free chain (weekends outvoted)",
          np.sum(losses < 1e-12) > 40, f"zero at {np.sum(losses < 1e-12)} of 61 clocks; the new estimator: "
          f"{ck.estimate_omega([(t0, atm)])['omega']:.4f}")
    few = {d.strftime("%Y%m%d"): (0.16 ** 2 * vc.variance_time(t0, d, 0.2), d) for d in days[:4]}
    r = ck.estimate_omega([(t0, few)])
    check("no weekend inside the window -> reported as not identified, no number", r["identified"] is False,
          r.get("reason", ""))


# --- replay timestamps ------------------------------------------------------------
def test_recorded_instant():
    print("\nSnapshot instants")
    import types
    import sources.replay as replay
    t0 = DT(2026, 9, 15, 17, 42, 11)
    ctxs = {e: types.SimpleNamespace(tau=vc.tau_years(t0, vc.parse_ib_date(e))) for e in ("20260916", "20260918", "20261016")}
    app = types.SimpleNamespace(recorded_at="2026-09-15T20:42:11")               # local UTC+3, no offset
    got = replay.recorded_utc(app, ctxs)
    check("the instant is recovered from the taus, not the local timestamp", abs((got - t0).total_seconds()) < 1e-3,
          f"{got.isoformat()} vs {t0.isoformat()}")
    try:
        replay.recorded_utc(app, None)
        refused = False
    except ValueError:
        refused = True
    check("a timestamp without an offset and no taus is refused", refused)
    app2 = types.SimpleNamespace(recorded_at="2026-09-15T17:42:11+00:00")
    check("an aware timestamp is used when there are no taus", replay.recorded_utc(app2, None) == t0)


def test_pipeline_clock():
    print("\nThe pipeline's clock on fake-exchange recordings priced on a planted trading clock")
    import tempfile
    import time
    import fake_ib
    import record_chains as rc
    import run_real_data as rrd
    import volatility_surface_3 as v3
    from test_recording import offline
    planted = 0.2
    flat = lambda k, tau, H: 0.16 + 1.0 * k * k                  # flat ATM forward variance on the planted clock
    with tempfile.TemporaryDirectory() as d, offline():
        cls = fake_ib.make_fake(v3.LiveSurfaceApp, iv_fn=flat,
                                variance_clock=lambda now, day: vc.variance_time(now, day, planted))
        app = v3.connect_app("SPY", port=4001, market_data_type=1, app_factory=cls, spot_timeout=2.0)
        today = vc.new_york_date(datetime.datetime.now(datetime.timezone.utc))
        exps = v3.select_expiries_by_target(app.expirations, today, tuple(range(1, 16)), 1)
        paths = []
        for cycle in range(2):
            path, _ = rc.record_cycle(app, "SPY", exps, d, cycle, max_strikes=9)
            paths.append(path)
            time.sleep(1.1)                                      # distinct file names
        clock, t0 = rrd.trading_clock(paths[-1])
    pc = clock["pooled"]
    check("both recordings of the day are pooled", clock["n_snapshots"] == 2, str(clock["n_snapshots"]))
    check("omega read from recorded quotes within 10% of the planted 0.2",
          pc.get("identified") and abs(pc["omega"] / planted - 1) < 0.10,
          f"{pc.get('omega', float('nan')):.4f} (95% {pc.get('ci95')}), {pc.get('n_expiries')} expiries")
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    check("the snapshot instant is the recording's UTC instant (within two minutes)",
          abs((t0 - now).total_seconds()) < 120, f"{t0.isoformat()} vs now {now.isoformat(timespec='seconds')}")


if __name__ == "__main__":
    print("=" * 74)
    print("trading-time clock")
    print("=" * 74)
    test_calendar()
    test_time_split()
    test_variance_time()
    test_estimator()
    test_recorded_instant()
    test_pipeline_clock()
    print("\n" + "=" * 74)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    print("=" * 74)
    sys.exit(1 if FAIL else 0)
