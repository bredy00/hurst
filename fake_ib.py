"""
A deterministic stand-in for TWS / IB Gateway, for offline tests and CI.

make_fake(app_class) returns a subclass whose outbound EClient calls are
answered locally instead of over a socket. Callbacks are delivered on a
worker thread a couple of milliseconds later, as TWS's reader thread would
deliver them -- which matters: the chain sweeper stamps each request AFTER
reqMktData returns, so a synchronous reply would look stale to it.

The book is priced from a smile whose ATM skew scales like tau^(H - 1/2), with
exact put-call parity, IBKR's units (vega per vol POINT) and IBKR's tick types
(delayed data on 66/67/83).
"""

import math
import queue
import threading
import time

import numpy as np

import volsurf_core as vc

CONID = 756733


def smile_iv(k, tau, H=0.1):
    """sigma(k, tau): a level, a rough ATM skew, and some curvature."""
    tau = max(tau, 1.0 / 365.0)
    skew = -0.14 * min(tau ** (H - 0.5), 12.0)
    return max(0.155 + 0.02 * math.sqrt(tau) + skew * k + 1.2 * k * k, 0.04)


def spy_like_expirations(today, n_days=730):
    """Dailies for three weeks, Fridays to three months, then third Fridays."""
    import datetime
    out = []
    for i in range(0, n_days):
        d = today + datetime.timedelta(days=i)
        if d.weekday() >= 5:
            continue
        third_friday = d.weekday() == 4 and 15 <= d.day <= 21
        if i <= 21 or (i <= 95 and d.weekday() == 4) or third_friday:
            out.append(d.strftime("%Y%m%d"))
    return out


class _Bar:
    def __init__(self, date, o, h, l, c, v):
        self.date, self.open, self.high, self.low, self.close, self.volume = date, o, h, l, c, v


def make_fake(app_class, spot=650.0, rate=0.04, carry=0.012, delayed=False, H=0.1,
              today=None, bars_fn=None, iv_fn=None, variance_clock=None):
    import datetime
    base_today = today or datetime.date(2026, 9, 15)

    class FakeIB(app_class):
        def __init__(self):
            super().__init__()
            self.fake_spot = spot
            self.fake_rate, self.fake_carry, self.fake_delayed = rate, carry, delayed
            self.fake_today = base_today
            self.fake_expirations = spy_like_expirations(base_today)
            self.fake_strikes = [float(k) for k in range(300, 1001)]
            self.sent_market_data_type = None
            self.requests = []
            self._connected = False
            self._q = queue.Queue()
            self._worker = threading.Thread(target=self._deliver, daemon=True)
            self._worker.start()

        # --- plumbing ------------------------------------------------------
        def _deliver(self):
            while True:
                fn = self._q.get()
                if fn is None:
                    return
                time.sleep(0.002)
                fn()

        def later(self, fn):
            self._q.put(fn)

        def connect(self, host, port, clientId):
            self._connected = True
            self.later(self.connectAck)

        def isConnected(self):
            return self._connected

        def run(self):
            return None

        def disconnect(self):
            self._connected = False

        def serverVersion(self):
            return 157

        def twsConnectionTime(self):
            return "20260915 10:00:00 US/Eastern"

        # --- reference data -------------------------------------------------
        def reqContractDetails(self, reqId, contract):
            class CD:
                pass
            cd = CD()
            cd.contract = type("C", (), {"conId": CONID})()
            self.later(lambda: (self.contractDetails(reqId, cd), self.contractDetailsEnd(reqId)))

        def reqSecDefOptParams(self, reqId, symbol, fut_exchange, sec_type, conid):
            self.later(lambda: (
                self.securityDefinitionOptionParameter(
                    reqId, "SMART", CONID, symbol, "100",
                    set(self.fake_expirations), set(self.fake_strikes)),
                self.securityDefinitionOptionParameterEnd(reqId)))

        def reqMarketDataType(self, t):
            self.sent_market_data_type = int(t)

        # --- quotes -----------------------------------------------------------
        def fake_forward(self, tau):
            return self.fake_spot * math.exp((self.fake_rate - self.fake_carry) * tau)

        def reqMktData(self, reqId, contract, generic, snapshot, reg_snapshot, opts):
            self.requests.append((reqId, contract))
            d = 60 if self.fake_delayed else 0         # 1/2/4 live, 66/67/68 delayed
            if reqId == self.SPOT_REQ_ID:
                self.later(lambda: self.tickPrice(reqId, 4 + (64 if d else 0), self.fake_spot, None))
                return
            exp = contract.lastTradeDateOrContractMonth
            K, right = float(contract.strike), contract.right
            now = datetime.datetime.now(datetime.timezone.utc)
            tau = vc.tau_years(now, vc.parse_ib_date(exp))
            F, df = self.fake_forward(tau), math.exp(-self.fake_rate * tau)
            if variance_clock is not None:
                # price total variance on the planted clock: sigma(tau_var)^2 tau_var
                tau_v = variance_clock(now, vc.parse_ib_date(exp))
                sig = (iv_fn or smile_iv)(math.log(K / F), tau_v, H) * math.sqrt(tau_v / tau)
            else:
                sig = (iv_fn or smile_iv)(math.log(K / F), tau, H)
            price = float(vc.bs_price(F, K, sig, tau, df, right))
            half = max(0.01, 0.015 * price)
            bid, ask = max(price - half, 0.0), price + half
            vega_ib = float(vc.bs_vega(F, K, sig, tau, df)) / 100.0   # IBKR: per vol point
            delta = 0.5
            model_tick = 83 if self.fake_delayed else 13

            def send():
                self.tickPrice(reqId, 1 + (65 if d else 0), bid, None)
                self.tickPrice(reqId, 2 + (65 if d else 0), ask, None)
                self.tickOptionComputation(reqId, model_tick, 0, sig, delta, price, 0.0,
                                           0.01, vega_ib, -0.05, self.fake_spot)
            self.later(send)

        def cancelMktData(self, reqId):
            pass

        # --- history ------------------------------------------------------------
        def reqHistoricalData(self, reqId, contract, end, duration, bar_size, what,
                              use_rth, fmt, keep_up, chart_options):
            self.requests.append((reqId, (contract.symbol, end, duration, bar_size, what)))
            bars = (bars_fn or default_bars)(contract, end, duration, bar_size, what)
            if bars is None:
                self.later(lambda: self.error(reqId, 162, "HMDS query returned no data"))
                return

            def send():
                for b in bars:
                    self.historicalData(reqId, b)
                self.historicalDataEnd(reqId, "", "")
            self.later(send)

    return FakeIB


def default_bars(contract, end, duration, bar_size, what, seed=7):
    """Deterministic bars: 400 daily bars, or one week of 78-a-day 5-minute bars."""
    import datetime
    if contract.symbol == "VIX":
        return None                       # "not permissioned", as on many accounts
    rng = np.random.default_rng(seed + (sum(map(ord, end)) if end else 0))  # hash() is salted per process
    if bar_size == "1 day":
        start = datetime.date(2025, 1, 2)
        out, px = [], 500.0
        for i in range(400):
            d = start + datetime.timedelta(days=i)
            if d.weekday() >= 5:
                continue
            if what == "TRADES":
                o = px
                c = o * math.exp(rng.normal(0, 0.01))
                h, l = max(o, c) * 1.004, min(o, c) * 0.996
                px = c
                out.append(_Bar(d.strftime("%Y%m%d"), o, h, l, c, 1e6))
            else:
                iv = 0.15 + 0.02 * math.sin(i / 20.0)
                out.append(_Bar(d.strftime("%Y%m%d"), iv, iv, iv, iv, 0))
        return out
    if bar_size == "5 mins":
        end_day = datetime.date(int(end[:4]), int(end[4:6]), int(end[6:8]))
        out = []
        for back in range(7):
            d = end_day - datetime.timedelta(days=back)
            if d.weekday() >= 5:
                continue
            px = 600.0
            for j in range(78):
                t = datetime.datetime.combine(d, datetime.time(9, 30)) + datetime.timedelta(minutes=5 * j)
                c = px * math.exp(rng.normal(0, 0.15 / math.sqrt(252 * 78)))
                out.append(_Bar(t.strftime("%Y%m%d  %H:%M:%S"), px, max(px, c), min(px, c), c, 1000))
                px = c
        return out
    return []
