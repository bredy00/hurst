"""
Synthetic stand-ins for a real recording, with a KNOWN answer (Session G).

  model_iv_fn(p)        an implied-vol function (k, tau) -> sigma priced by the lifted
                        rough Heston, for fake_ib's book: a recorded chain whose
                        generating parameters are known
  synthetic_history(p)  a history file in record_history.py's format, from a QE
                        simulation at 5-minute resolution: daily OHLC, 5-minute bars
                        and a 30-day implied variance, with a known H

run_real_data.py --synthetic runs the real-data pipeline end to end on these, so
every stage that will touch real data is exercised against a truth first.
"""

import datetime
import math

import numpy as np

import models.rough_heston as rh
import pricing.fourier as fo
import sources.history as hist


def model_iv_fn(p, n_k=121):
    """(k, tau, H) -> implied vol under the lifted rough Heston p, interpolated per maturity."""
    cache = {}

    def iv(k, tau, _H=None):
        key = round(float(tau), 6)
        if key not in cache:
            half = 4.0 * math.sqrt(max(p.v0, p.theta) * max(tau, 1.0 / 365))
            ks = np.linspace(-half, half, n_k)
            c, info = rh.call_prices(ks, tau, p)
            vols = fo.implied_vols_from_calls(c, ks, tau)
            good = np.isfinite(vols)
            cache[key] = (ks[good], vols[good])
        ks, vols = cache[key]
        return float(np.interp(k, ks, vols))
    return iv


def synthetic_history(p, n_days=500, bars_per_day=78, seed=0, start=datetime.date(2024, 9, 16), spot=500.0):
    """
    A volsurf-history/1 dict simulated from rough Heston p with the positivity-
    preserving step: 5-minute bars during the session, overnight gaps ignored
    (the variance keeps evolving overnight at the same scheme, without returns).
    iv30 is the square root of the model's expected average variance over the next
    30 trading days from each day's closing state -- the quantity a VIX-style index
    measures under this model.
    """
    rng = np.random.default_rng(seed)
    w, x = rh.lift_nodes(p.H, rh.N_DEFAULT)
    h = 1.0 / (252 * bars_per_day)
    st = rh.LiftedAffineStep(w, x, p.v0, p.kappa, p.theta, p.xi, h, p.rho)
    st30 = rh.LiftedAffineStep(w, x, p.v0, p.kappa, p.theta, p.xi, 30.0 / 252, p.rho)
    y = np.zeros((len(w), 1))
    logS = math.log(spot)
    srho = math.sqrt(1.0 - p.rho * p.rho)
    daily, bars, iv30, true_daily_var, close_var = [], [], [], [], []
    day = start
    while len(daily) < n_days:
        if day.weekday() >= 5:
            day += datetime.timedelta(days=1)
            continue
        o = math.exp(logS)
        hi = lo = o
        ivar = 0.0
        t = datetime.datetime.combine(day, datetime.time(9, 30))
        for j in range(bars_per_day):
            y, V, dI, dZ, fix = st.step_y(y, rng, with_log_price=True)
            prev = math.exp(logS)
            logS += float(-0.5 * dI[0] + p.rho * dZ[0] + srho * math.sqrt(dI[0]) * rng.standard_normal() + fix[0])
            c = math.exp(logS)
            hi, lo = max(hi, prev, c), min(lo, prev, c)
            ivar += float(dI[0])
            t += datetime.timedelta(minutes=5)
            bars.append({"date": t.strftime("%Y%m%d  %H:%M:%S"), "open": prev, "high": max(prev, c),
                         "low": min(prev, c), "close": c, "volume": 1000.0})
        d = day.strftime("%Y%m%d")
        daily.append({"date": d, "open": o, "high": hi, "low": lo, "close": math.exp(logS), "volume": 1e6})
        ev = max(st30.EI_const + float(st30.EI_lin @ y[:, 0]), 0.0) / (30.0 / 252)
        iv30.append({"date": d, "open": math.sqrt(ev), "high": math.sqrt(ev), "low": math.sqrt(ev),
                     "close": math.sqrt(ev), "volume": 0.0})
        true_daily_var.append(252.0 * ivar)
        close_var.append(float(V[0]))
        day += datetime.timedelta(days=1)
    return {"format": hist.FORMAT, "symbol": "SYNTH", "synthetic": True,
            "truth": {k: getattr(p, k) for k in ("v0", "kappa", "theta", "xi", "rho", "H")},
            "true_daily_integrated_variance": true_daily_var, "true_close_variance": close_var,
            "daily": daily, "iv30": iv30, "hv30": [], "vix": [], "bars5m": bars}
