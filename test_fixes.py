"""
Proof for the six runtime fixes in volatility_surface_2.py.

Everything here runs with no TWS and no network: EClient's outbound calls are
stubbed, and the wrapper callbacks are driven directly, the same way the reader
thread would drive them.

    python test_fixes.py
"""

import contextlib
import datetime
import io
import sys
import threading
import time

import volatility_surface_2 as v2


PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))


class StubApp(v2.LiveSurfaceApp):
    """LiveSurfaceApp with every outbound socket call replaced by a recorder."""

    def __init__(self):
        super().__init__()
        self.sent = []        # (reqId, genericTicks)
        self.cancelled = []
        self.concurrent_peak = 0
        self._open = set()
        self.msg_times = []
        self._slock = threading.Lock()

    def reqMktData(self, reqId, contract, genericTickList, snapshot, regSnapshot, opts):
        with self._slock:
            self.sent.append((reqId, genericTickList))
            self.msg_times.append(time.monotonic())
            self._open.add(reqId)
            self.concurrent_peak = max(self.concurrent_peak, len(self._open))

    def cancelMktData(self, reqId):
        with self._slock:
            self.cancelled.append(reqId)
            self.msg_times.append(time.monotonic())
            self._open.discard(reqId)


# --- Fix 1 -----------------------------------------------------------------
def test_error_signatures():
    print("\nFix 1 -- version-agnostic error() callback")
    # Observe what the handler ITSELF decided, by capturing what it printed.
    # (Re-deriving the parse inside the test would pass even if error() broke.)
    for label, args in [
        ("ibapi <=9.81  (reqId, code, msg)",
         (7, 200, "No security definition")),
        ("ibapi 10.10+  (reqId, code, msg, json)",
         (7, 200, "No security definition", "")),
        ("ibapi 10.19+  (reqId, errorTime, code, msg, json)",
         (7, 1789169000000, 200, "No security definition", "")),
    ]:
        app = StubApp()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            app.error(*args)
        line = buf.getvalue().strip()
        want = "IBKR Msg 7: 200 - No security definition"
        check(label, line == want, f"handler printed {line!r}")

    # A timestamp must never be mistaken for an error code
    app = StubApp()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        app.error(-1, 1789169000000, 2104, "Market data farm OK", "")
    check("5-arg benign code suppressed (timestamp not read as code)",
          buf.getvalue().strip() == "", f"printed {buf.getvalue()!r}")

    # Too-short payloads must not raise
    app = StubApp()
    with contextlib.redirect_stdout(io.StringIO()):
        app.error(7, 200)
    check("2-arg payload does not raise", True)

    # The 502 path must release every waiter so nothing can block forever
    app = StubApp()
    app.error(-1, 1789169000000, 502, "Couldn't connect to TWS", "")
    check("502 (5-arg form) trips connect_failed",
          app.connect_failed.is_set())
    check("502 releases resolved/chain/spot waiters",
          app.resolved.is_set() and app.chain_resolved.is_set() and app.spot_ready.is_set())

    app = StubApp()
    app.error(-1, 502, "Couldn't connect to TWS")
    check("502 (3-arg form) trips connect_failed", app.connect_failed.is_set())

    app = StubApp()
    app.error(1, 2104, "Market data farm connection is OK")
    check("benign 2104 does not trip failure", not app.connect_failed.is_set())


# --- Fix 2 -----------------------------------------------------------------
def test_spot_fallbacks():
    print("\nFix 2 -- spot resolution and fallbacks")
    R = v2.LiveSurfaceApp.SPOT_REQ_ID

    app = StubApp()
    app.tickPrice(R, v2.TICK_LAST, 650.25, None)
    check("LAST (4) accepted", app.spot_price == 650.25 and app.spot_ready.is_set())

    app = StubApp()
    app.tickPrice(R, v2.TICK_BID, 649.90, None)
    check("BID alone is not enough", not app.spot_ready.is_set())
    app.tickPrice(R, v2.TICK_ASK, 650.10, None)
    check("BID+ASK -> mid", abs(app.spot_price - 650.00) < 1e-9, f"spot={app.spot_price}")

    app = StubApp()
    app.tickPrice(R, v2.TICK_CLOSE, 648.00, None)
    check("CLOSE (9) accepted when nothing else", app.spot_price == 648.00)

    for tt, name in [(v2.TICK_DELAYED_LAST, "DELAYED_LAST (68)"),
                     (v2.TICK_DELAYED_CLOSE, "DELAYED_CLOSE (75)")]:
        app = StubApp()
        app.tickPrice(R, tt, 651.0, None)
        check(f"{name} accepted", app.spot_price == 651.0)

    app = StubApp()
    app.tickPrice(R, v2.TICK_DELAYED_BID, 650.0, None)
    app.tickPrice(R, v2.TICK_DELAYED_ASK, 651.0, None)
    check("DELAYED_BID/ASK (66/67) -> mid", abs(app.spot_price - 650.5) < 1e-9)

    # Preference order: a real LAST must win over a stale close
    app = StubApp()
    app.tickPrice(R, v2.TICK_CLOSE, 600.0, None)
    app.tickPrice(R, v2.TICK_LAST, 655.0, None)
    check("LAST outranks CLOSE", app.spot_price == 655.0)

    app = StubApp()
    app.tickPrice(R, v2.TICK_LAST, -1.0, None)
    check("negative price rejected", not app.spot_ready.is_set())
    app.tickPrice(999999, v2.TICK_LAST, 650.0, None)
    check("foreign reqId ignored", not app.spot_ready.is_set())

    # The alpha's failure mode: nothing but a non-price tick, forever.
    app = StubApp()
    app.tickPrice(R, 6, 700.0, None)   # HIGH, not in the accepted set
    check("unusable tick does not falsely set spot", not app.spot_ready.is_set())


# --- Fix 3 -----------------------------------------------------------------
def test_pacing_and_lines():
    print("\nFix 3 -- line cap, pacing, cancellation")

    lim = v2.RateLimiter(40)
    t0 = time.monotonic()
    for _ in range(20):
        lim.acquire()
    elapsed = time.monotonic() - t0
    check("RateLimiter(40/s): 20 msgs take >= ~0.475s",
          elapsed >= 0.475, f"{elapsed:.3f}s")

    app = StubApp()
    contracts = []
    for i in range(162):                      # the real SPY count from the alpha
        app.id_map[1000 + i] = ("20260911", 637.0 + (i % 27))
        contracts.append((1000 + i, object()))

    sweeper = v2.ChainSweeper(app, contracts, max_lines=90,
                              req_per_sec=2000, quote_timeout=0.25)

    # Deliver an IV for every second contract; the rest must time out and cancel.
    def feed():
        time.sleep(0.05)
        for idx, (rid, _) in enumerate(contracts):
            if idx % 2 == 0:
                app.tickOptionComputation(rid, v2.TICK_MODEL_OPTION, 0,
                                          0.15, 0.5, 1.0, 0.0, 0.01, 0.2, -0.1, 650.0)
            time.sleep(0.001)

    threading.Thread(target=feed, daemon=True).start()
    sweeper._sweep()

    check("never exceeded 90 concurrent lines",
          app.concurrent_peak <= 90, f"peak={app.concurrent_peak}")
    check("every subscription was cancelled",
          len(app.cancelled) == len(app.sent) == 162,
          f"sent={len(app.sent)} cancelled={len(app.cancelled)}")
    check("no line left open", len(app._open) == 0)

    # Staleness: quotes must age out, since the sweeper cycles lines
    app2 = StubApp()
    app2.id_map[1000] = ("20260911", 650.0)
    app2.iv_dict[1000] = (0.15, time.monotonic() - 120.0)
    app2.id_map[1001] = ("20260911", 651.0)
    app2.iv_dict[1001] = (0.16, time.monotonic())
    fresh = app2.fresh_quotes(max_age=30.0)
    check("stale quote (120s) dropped, fresh kept",
          len(fresh) == 1 and fresh[0]['Strike'] == 651.0, f"{fresh}")


# --- Fix 4 -----------------------------------------------------------------
def test_chain_filter():
    print("\nFix 4 -- tradingClass / multiplier filter")
    exps, strikes = {"20260911", "20260916"}, {640.0, 650.0, 660.0}

    def fresh():
        a = StubApp()
        a._symbol = "SPY"
        return a

    a = fresh()
    a.securityDefinitionOptionParameter(2, "SMART", 1, "SPY", "100", exps, strikes)
    check("accepts SMART / SPY / x100",
          a.trading_class == "SPY" and len(a.expirations) == 2)

    a = fresh()
    a.securityDefinitionOptionParameter(2, "SMART", 1, "SPY7", "10", exps, strikes)
    check("rejects mini-option (multiplier 10)", not a.expirations)

    a = fresh()
    a.securityDefinitionOptionParameter(2, "SMART", 1, "SPY1", "100", exps, strikes)
    check("rejects foreign tradingClass SPY1", not a.expirations)

    a = fresh()
    a.securityDefinitionOptionParameter(2, "BOX", 1, "SPY", "100", exps, strikes)
    check("rejects non-SMART exchange", not a.expirations)

    a = fresh()
    a.securityDefinitionOptionParameter(2, "BOX", 1, "SPY7", "10", exps, strikes)
    a.securityDefinitionOptionParameter(2, "SMART", 1, "SPY", "100", exps, strikes)
    check("picks the right class out of a mixed stream", a.trading_class == "SPY")

    a = fresh()
    a.securityDefinitionOptionParameterEnd(2)
    check("End releases the waiter even with no match", a.chain_resolved.is_set())


# --- Fix 5 -----------------------------------------------------------------
def test_generic_ticks():
    print("\nFix 5 -- generic tick list")
    app = StubApp()
    contracts = [(1000, object())]
    app.id_map[1000] = ("20260911", 650.0)
    sweeper = v2.ChainSweeper(app, contracts, max_lines=5,
                              req_per_sec=2000, quote_timeout=0.05)
    sweeper._sweep()
    ticks = {g for _, g in app.sent}
    check('option subscriptions send genericTickList=""',
          ticks == {""}, f"observed {ticks}")

    from ibapi.ticktype import TickTypeEnum
    check("tick 13 is MODEL_OPTION", TickTypeEnum.to_str(13) == "MODEL_OPTION")
    check("tick 24 is OPTION_IMPLIED_VOL (what generic 106 delivers)",
          TickTypeEnum.to_str(24) == "OPTION_IMPLIED_VOL")

    # A non-model computation tick must not be recorded as the surface IV
    app = StubApp()
    app.tickOptionComputation(1000, 10, 0, 0.99, 0.5, 1.0, 0.0, 0.01, 0.2, -0.1, 650.0)
    check("BID_OPTION_COMPUTATION (10) ignored", 1000 not in app.iv_dict)
    app.tickOptionComputation(1000, 13, 0, 0.15, 0.5, 1.0, 0.0, 0.01, 0.2, -0.1, 650.0)
    check("MODEL_OPTION (13) recorded", app.iv_dict[1000][0] == 0.15)
    app.tickOptionComputation(1001, 13, 0, None, 0.5, 1.0, 0.0, 0.01, 0.2, -0.1, 650.0)
    check("None IV rejected", 1001 not in app.iv_dict)
    app.tickOptionComputation(1002, 13, 0, float('nan'), 0.5, 1.0, 0.0, 0.01, 0.2, -0.1, 650.0)
    check("NaN IV rejected", 1002 not in app.iv_dict)
    app.tickOptionComputation(1003, 13, 0, -0.5, 0.5, 1.0, 0.0, 0.01, 0.2, -0.1, 650.0)
    check("negative IV rejected", 1003 not in app.iv_dict)


# --- Fix 6 -----------------------------------------------------------------
def test_zero_dte():
    print("\nFix 6 -- 0DTE excluded")
    today = datetime.date(2026, 9, 4)
    exps = ["20260904", "20260907", "20260909", "20260911",
            "20260914", "20260916", "20260918", "20260921"]

    kept, skipped = select_expiries_wrapper(exps, today)
    check("today's expiry is skipped", "20260904" not in kept and "20260904" in skipped)
    check("returns 6 expiries", len(kept) == 6, f"{kept}")
    check("earliest kept is T>0", kept[0] == "20260907")

    # The alpha's rule, for contrast
    alpha_kept = [e for e in exps if e >= today.strftime('%Y%m%d')][:6]
    check("alpha's rule would have included 0DTE", alpha_kept[0] == "20260904")

    kept2, _ = v2.select_expiries(exps, today, min_days_to_expiry=3, n_expiries=6)
    check("min_days_to_expiry=3 honoured", kept2[0] == "20260907")

    past = ["20260901", "20260902"] + exps
    kept3, skipped3 = v2.select_expiries(past, today, 1, 6)
    check("already-expired dates excluded",
          not any(e in kept3 for e in ("20260901", "20260902")))
    check("expired dates are not reported as skipped-0DTE",
          not any(e in skipped3 for e in ("20260901", "20260902")))

    strikes = [float(s) for s in range(600, 701)]
    sel = v2.select_strikes(strikes, 650.0, 0.02)
    check("strike band +/-2% -> 637..663", sel[0] == 637.0 and sel[-1] == 663.0,
          f"{len(sel)} strikes")


def select_expiries_wrapper(exps, today):
    return v2.select_expiries(exps, today, min_days_to_expiry=1, n_expiries=6)


if __name__ == "__main__":
    print("=" * 68)
    print("Runtime fix verification -- volatility_surface_2.py")
    print("=" * 68)
    test_error_signatures()
    test_spot_fallbacks()
    test_pacing_and_lines()
    test_chain_filter()
    test_generic_ticks()
    test_zero_dte()
    print("\n" + "=" * 68)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
    print("=" * 68)
    sys.exit(1 if FAIL else 0)
