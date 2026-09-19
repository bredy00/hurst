"""
Volatility surface -- v3.

Keeps all six runtime fixes from v2, and changes the four things that were
actually wrong with the surface itself:

  A  TCP pre-probe, so an absent TWS fails in well under a second instead of
     sitting inside EClient.connect()'s own socket timeout for ten.
  B  The greeks are kept. IBKR delivers delta/gamma/vega/theta and the model
     price on every tick and both earlier versions bound them and threw them
     away -- including the one number the weighting needs.
  C  The grid is built on z = ln(K/F) / (sigma_atm * sqrt(tau)) and the surface
     plots total variance w = sigma^2 * tau. The forward comes from put-call
     parity, so no rate or dividend is assumed.
  D  No bfill/ffill. Where there is no market there is a hole, and the hole is
     drawn as a hole.

Why C is not cosmetic: a fixed +/-2% band has a width in standard deviations of
0.02 / (sigma * sqrt(tau)), which diverges as tau -> 0. On a SPY chain that is
~4.7 sigma at half a day and ~0.7 sigma at twelve days -- simultaneously far too
wide and far too narrow. Vega then decays like exp(-d1^2 / 2), so the implied
vol uncertainty in the short-dated wings runs to thousands of vol points while
the back months sit at hundredths. Sampling at fixed z makes every contract in
the grid carry comparable vega, which is what makes the noise uniform.
"""

import collections
import math
import pathlib
import datetime
import socket
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper

import volsurf_core as vc

# matplotlib is imported lazily inside live_desktop_plot. Pulling it in at module
# scope costs ~2.4s, and the most common outcome of running this file is "TWS is
# not running", which should not take eight seconds of imports to discover.


# --- tick types (verified against ibapi.ticktype.TickTypeEnum) ---
TICK_BID, TICK_ASK, TICK_LAST, TICK_CLOSE = 1, 2, 4, 9
TICK_DELAYED_BID, TICK_DELAYED_ASK = 66, 67
TICK_DELAYED_LAST, TICK_DELAYED_CLOSE = 68, 75
TICK_BID_OPTION, TICK_ASK_OPTION, TICK_MODEL_OPTION = 10, 11, 13
# Delayed data sends its greeks on 83, not 13. Filtering on 13 alone meant a
# delayed-data session recorded bid/ask but never a single implied vol.
TICK_DELAYED_MODEL_OPTION = 83
MODEL_OPTION_TICKS = (TICK_MODEL_OPTION, TICK_DELAYED_MODEL_OPTION)
MARKET_DATA_TYPES = {1: 'live', 2: 'frozen', 3: 'delayed', 4: 'delayed-frozen'}
# IBKR quotes vega per ONE VOL POINT (a 0.01 move in implied vol). Everything in
# this package -- iv_error, quote_weight, MAX_IV_UNCERTAINTY -- is per 1.00 of
# vol, as bs_vega is. Unconverted, every live iv_error comes out 100x too big and
# the 5-vol-point usability filter drops nearly the whole chain. The stand-in
# feeds used bs_vega, so no test could see it (Session G). `record_chains.py
# --check` measures the ratio on the first live quote.
IB_VEGA_PER_UNIT_VOL = 100.0
# Newer servers send Double.MAX_VALUE for a greek they did not compute, and
# np.isfinite(1.8e308) is True.
IB_UNSET = 1e300

SPOT_TICK_ROLES = {
    TICK_LAST: 'last',   TICK_DELAYED_LAST: 'last',
    TICK_BID: 'bid',     TICK_DELAYED_BID: 'bid',
    TICK_ASK: 'ask',     TICK_DELAYED_ASK: 'ask',
    TICK_CLOSE: 'close', TICK_DELAYED_CLOSE: 'close',
}

BENIGN_CODES = {2104, 2106, 2107, 2119, 2158}
MAX_MSG_PER_SEC = 40
MAX_MARKET_DATA_LINES = 90

# A quote whose implied vol is uncertain by more than this is not information.
MAX_IV_UNCERTAINTY = 0.05      # 5 vol points
N_SIGMA_BAND = 3.0
# Half-width, in sigma units, of the local window used to measure the ATM skew.
ATM_WINDOW_Z = 1.5


@dataclass
class OptionQuote:
    """Everything the callback gives us. Fix B: none of it is discarded."""
    expiry: str
    strike: float
    right: str
    iv: float = float('nan')
    delta: float = float('nan')
    gamma: float = float('nan')
    vega: float = float('nan')
    theta: float = float('nan')
    model_price: float = float('nan')
    und_price: float = float('nan')
    bid: float = float('nan')
    ask: float = float('nan')
    ts: float = 0.0

    @property
    def mid(self):
        if np.isfinite(self.bid) and np.isfinite(self.ask) and self.ask >= self.bid > 0:
            return 0.5 * (self.bid + self.ask)
        return self.model_price

    @property
    def half_spread(self):
        if np.isfinite(self.bid) and np.isfinite(self.ask) and self.ask >= self.bid:
            return max(0.5 * (self.ask - self.bid), 0.005)
        return 0.005

    @property
    def iv_error(self):
        """Standard error of this quote's implied vol: half-spread / vega."""
        if not np.isfinite(self.vega) or self.vega <= 0:
            return float('inf')
        return self.half_spread / self.vega

    def usable(self, max_err=MAX_IV_UNCERTAINTY):
        return np.isfinite(self.iv) and self.iv > 0 and self.iv_error <= max_err


@dataclass
class ExpiryContext:
    """Per-expiry state: everything needed to put a strike on the grid."""
    expiry: str
    tau: float
    forward: float = float('nan')
    discount: float = 1.0
    parity_r2: float = 0.0
    sigma_atm: float = float('nan')
    strikes: list = field(default_factory=list)


class RateLimiter:
    def __init__(self, per_second):
        self._interval = 1.0 / float(per_second)
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            if now < self._next_slot:
                time.sleep(self._next_slot - now)
            self._next_slot = time.monotonic() + self._interval


class ConnectionError_(RuntimeError):
    pass


def probe_tws(host, port, timeout=0.25):
    """
    Fix A. EClient.connect() takes ~10s to give up on a dead port.

    Note this always burns the full timeout rather than returning instantly:
    packets to a closed TWS port on this machine are dropped, not refused, so
    the probe raises TimeoutError rather than ConnectionRefusedError. 250 ms is
    ample for a loopback connection and 40x faster than letting EClient decide.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class LiveSurfaceApp(EWrapper, EClient):
    SPOT_REQ_ID = 999

    def __init__(self):
        EClient.__init__(self, self)
        self._lock = threading.RLock()
        self.quotes = {}            # reqId -> OptionQuote
        self.id_map = {}            # reqId -> (expiry, strike, right)
        self.ctx = {}               # expiry -> ExpiryContext
        self.expirations, self.strikes = [], []
        self.trading_class = None
        self.spot_price = 0.0
        self._spot_quotes = {}
        self.underlying_conId = 0
        self._symbol = None
        self.resolved = threading.Event()
        self.chain_resolved = threading.Event()
        self.spot_ready = threading.Event()
        self.connect_failed = threading.Event()
        self.error_counts = collections.Counter()
        self.data_type_seen = {}    # reqId -> market data type IBKR says it is sending

    def connectAck(self):
        print("TWS Acknowledged Connection")

    # Fix 1 (carried from v2): every ibapi signature shape
    def error(self, *args):
        if len(args) < 3:
            print(f"IBKR Msg (unparsed): {args}")
            return
        if len(args) >= 5:
            req_id, code, msg = args[0], args[2], args[3]
        else:
            req_id, code, msg = args[0], args[1], args[2]
        self.error_counts[code] += 1
        if code == 502:
            self.connect_failed.set()
            for ev in (self.resolved, self.chain_resolved, self.spot_ready):
                ev.set()
        if code not in BENIGN_CODES:
            print(f"IBKR Msg {req_id}: {code} - {msg}")

    def marketDataType(self, reqId, marketDataType):
        self.data_type_seen[reqId] = int(marketDataType)

    def contractDetails(self, reqId, contractDetails):
        self.underlying_conId = contractDetails.contract.conId
        self.resolved.set()

    def contractDetailsEnd(self, reqId):
        self.resolved.set()

    # Fix 2 (carried): every price tick that can stand in for spot
    def tickPrice(self, reqId, tickType, price, attrib):
        if price <= 0:
            return
        if reqId == self.SPOT_REQ_ID:
            role = SPOT_TICK_ROLES.get(tickType)
            if role is None:
                return
            with self._lock:
                self._spot_quotes[role] = price
                resolved = self._resolve_spot()
                if resolved > 0:
                    self.spot_price = resolved
                    self.spot_ready.set()
            return
        # Option bid/ask: needed for put-call parity and for the real spread
        if tickType in (TICK_BID, TICK_ASK, TICK_DELAYED_BID, TICK_DELAYED_ASK):
            with self._lock:
                q = self._quote(reqId)
                if q is None:
                    return
                if tickType in (TICK_BID, TICK_DELAYED_BID):
                    q.bid = price
                else:
                    q.ask = price
                q.ts = time.monotonic()

    def _resolve_spot(self):
        q = self._spot_quotes
        if q.get('last', 0) > 0:
            return q['last']
        if q.get('bid', 0) > 0 and q.get('ask', 0) > 0:
            return 0.5 * (q['bid'] + q['ask'])
        return q.get('close', 0.0)

    # Fix 4 (carried): one trading class, multiplier 100, SMART only
    def securityDefinitionOptionParameter(self, reqId, exchange, underlyingConId,
                                          tradingClass, multiplier, expirations, strikes):
        if exchange != "SMART" or str(multiplier) != "100":
            return
        if self._symbol is not None and tradingClass != self._symbol:
            return
        with self._lock:
            self.trading_class = tradingClass
            self.expirations = sorted(list(expirations))
            self.strikes = sorted(list(strikes))
        self.chain_resolved.set()

    def securityDefinitionOptionParameterEnd(self, reqId):
        self.chain_resolved.set()

    def _quote(self, reqId):
        mapped = self.id_map.get(reqId)
        if mapped is None:
            return None
        q = self.quotes.get(reqId)
        if q is None:
            q = OptionQuote(expiry=mapped[0], strike=mapped[1], right=mapped[2])
            self.quotes[reqId] = q
        return q

    # --- Fix B: keep the greeks -------------------------------------------
    def tickOptionComputation(self, reqId, tickType, tickAttrib, impliedVol, delta,
                              optPrice, pvDividend, gamma, vega, theta, undPrice):
        if tickType not in MODEL_OPTION_TICKS:
            return
        if impliedVol is None or not np.isfinite(impliedVol) or not 0 < impliedVol < IB_UNSET:
            return
        with self._lock:
            q = self._quote(reqId)
            if q is None:
                return
            q.iv = float(impliedVol)
            for name, val, scale in (('delta', delta, 1.0), ('gamma', gamma, 1.0),
                                     ('vega', vega, IB_VEGA_PER_UNIT_VOL), ('theta', theta, 1.0),
                                     ('model_price', optPrice, 1.0), ('und_price', undPrice, 1.0)):
                if val is not None and np.isfinite(val) and abs(val) < IB_UNSET:
                    setattr(q, name, float(val) * scale)
            q.ts = time.monotonic()

    def quote_arrived_after(self, req_id, t0):
        with self._lock:
            q = self.quotes.get(req_id)
        return q is not None and q.ts >= t0 and np.isfinite(q.iv)

    def fresh(self, max_age=30.0):
        now = time.monotonic()
        with self._lock:
            return [q for q in self.quotes.values() if now - q.ts <= max_age]


# Fix 3 (carried from v2): line cap, pacing, cancellation
class ChainSweeper(threading.Thread):
    def __init__(self, app, contracts, max_lines=MAX_MARKET_DATA_LINES,
                 req_per_sec=MAX_MSG_PER_SEC, quote_timeout=8.0):
        super().__init__(daemon=True)
        self.app, self.contracts = app, contracts
        self.max_lines, self.quote_timeout = max_lines, quote_timeout
        self.limiter = RateLimiter(req_per_sec)
        self._stop_flag = threading.Event()
        self._active = {}
        self.sweeps = 0

    def stop(self):
        self._stop_flag.set()

    def cancel_all(self):
        for rid in list(self._active):
            try:
                self.app.cancelMktData(rid)
            except Exception:
                pass
        self._active.clear()

    def run(self):
        while not self._stop_flag.is_set():
            self.sweep()
            self.sweeps += 1

    def sweep(self):
        pending = list(self.contracts)
        self._active = {}
        while (pending or self._active) and not self._stop_flag.is_set():
            while pending and len(self._active) < self.max_lines:
                req_id, contract = pending.pop(0)
                self.limiter.acquire()
                # Fix 5 (carried): empty generic tick list
                self.app.reqMktData(req_id, contract, "", False, False, [])
                self._active[req_id] = time.monotonic()
            now = time.monotonic()
            done = [rid for rid, t0 in self._active.items()
                    if self.app.quote_arrived_after(rid, t0) or (now - t0) > self.quote_timeout]
            for rid in done:
                self.limiter.acquire()
                self.app.cancelMktData(rid)
                self._active.pop(rid, None)
            time.sleep(0.05)


def make_option(symbol, expiry, strike, right, trading_class=None):
    o = Contract()
    o.symbol, o.secType = symbol, 'OPT'
    o.exchange, o.currency = 'SMART', 'USD'
    o.lastTradeDateOrContractMonth = expiry
    o.strike, o.right = strike, right
    if trading_class:
        o.tradingClass = trading_class
    return o


# --- Fix C: build the grid in sigma units -----------------------------------
def build_expiry_contexts(app, expiries, t0, spot):
    """
    Seed pass. For each expiry take the strikes nearest spot in BOTH rights,
    recover (F, df) from put-call parity, and read sigma_atm off the result.
    Only then can a sigma-normalised band be defined at all -- the band width
    depends on the very quantity we are trying to measure, so it takes two
    passes, exactly as a desk would do it.
    """
    ctxs = {}
    for exp in expiries:
        ctx = ExpiryContext(expiry=exp, tau=vc.tau_years(t0, vc.parse_ib_date(exp)))
        near = sorted(app.strikes, key=lambda s: abs(s - spot))[:9]
        ctx.strikes = sorted(near)
        ctxs[exp] = ctx
    return ctxs


def resolve_forwards(app, ctxs, spot):
    """Run put-call parity per expiry on whatever seed quotes have arrived."""
    by_exp = {}
    for q in app.fresh(max_age=60.0):
        by_exp.setdefault(q.expiry, {}).setdefault(q.strike, {})[q.right] = q

    for exp, ctx in ctxs.items():
        ks, cs, ps = [], [], []
        for strike, sides in sorted(by_exp.get(exp, {}).items()):
            c_q, p_q = sides.get('C'), sides.get('P')
            if c_q is None or p_q is None:
                continue
            cm, pm = c_q.mid, p_q.mid
            if not (np.isfinite(cm) and np.isfinite(pm)):
                continue
            ks.append(strike)
            cs.append(cm)
            ps.append(pm)

        F, df, r2 = vc.forward_from_parity(ks, cs, ps)
        if F is None or not (0.5 * spot < F < 2.0 * spot):
            # Parity failed or produced nonsense -- fall back to spot and say so
            ctx.forward, ctx.discount, ctx.parity_r2 = spot, 1.0, 0.0
        else:
            ctx.forward, ctx.discount, ctx.parity_r2 = F, df, r2

        # sigma_atm from the quote closest to the forward
        best, best_d = None, 1e18
        for strike, sides in by_exp.get(exp, {}).items():
            for q in sides.values():
                if np.isfinite(q.iv) and abs(strike - ctx.forward) < best_d:
                    best, best_d = q, abs(strike - ctx.forward)
        ctx.sigma_atm = best.iv if best is not None else 0.20
    return ctxs


def grid_contracts(app, ctxs, symbol, n_sigma=N_SIGMA_BAND, max_strikes=21, id_base=2000):
    """OTM contracts on a sigma-normalised band, split at the FORWARD."""
    contracts, req_id = [], id_base
    for exp, ctx in ctxs.items():
        if not np.isfinite(ctx.forward) or not np.isfinite(ctx.sigma_atm):
            continue
        sel = vc.select_strikes_by_sigma(app.strikes, ctx.forward, ctx.sigma_atm,
                                         ctx.tau, n_sigma, max_strikes)
        ctx.strikes = sel
        for K in sel:
            right = vc.otm_right(K, ctx.forward)
            app.id_map[req_id] = (exp, K, right)
            contracts.append((req_id, make_option(symbol, exp, K, right, app.trading_class)))
            req_id += 1
    return contracts


def surface_points(app, ctxs, max_age=30.0, max_err=MAX_IV_UNCERTAINTY):
    """
    Turn live quotes into (tau, z, w, weight) with every unusable quote dropped
    rather than filled in. Returns a dict keyed by expiry.
    """
    out = {}
    for q in app.fresh(max_age):
        ctx = ctxs.get(q.expiry)
        if ctx is None or not np.isfinite(ctx.forward) or not np.isfinite(ctx.sigma_atm):
            continue
        if not q.usable(max_err):
            continue
        z = vc.normalised_moneyness(q.strike, ctx.forward, ctx.sigma_atm, ctx.tau)
        if not np.isfinite(z):
            continue
        out.setdefault(q.expiry, []).append({
            'z': float(z),
            'k': float(vc.log_moneyness(q.strike, ctx.forward)),
            'w': float(vc.total_variance(q.iv, ctx.tau)),
            'iv': q.iv,
            'weight': float(vc.quote_weight(q.half_spread, q.vega)),
            'strike': q.strike,
        })
    for exp in out:
        out[exp].sort(key=lambda r: r['z'])
    return out


def build_grid(points, ctxs, z_grid):
    """
    Interpolate each expiry's total variance onto a common z grid.

    Fix D: outside the observed z range the value is NaN, and stays NaN.
    matplotlib draws NaN as a hole, so a gap in the market renders as a gap in
    the surface instead of a flat extrapolation that looks like data.
    """
    exps = sorted(points, key=lambda e: ctxs[e].tau)
    taus, Z = [], []
    for exp in exps:
        rows = points[exp]
        if len(rows) < 3:
            continue
        zs = np.array([r['z'] for r in rows])
        ws = np.array([r['w'] for r in rows])
        row = np.interp(z_grid, zs, ws, left=np.nan, right=np.nan)
        row[(z_grid < zs.min()) | (z_grid > zs.max())] = np.nan
        taus.append(ctxs[exp].tau)
        Z.append(row)
    return np.array(taus), np.array(Z), [e for e in exps if e in points]


def roughness(points, ctxs, window=ATM_WINDOW_Z):
    """
    ATM skew per expiry from a LOCAL fit, then H from the weighted log-log slope.

    The skew must be measured near the money: a global fit across the whole
    +/-3 sigma slice lets the wings bias what is meant to be a derivative at a
    point. Each expiry's skew also carries a standard error, and H is fitted
    weighted by it -- short expiries hold fewer strikes inside the window, so
    their skew is the least certain and must not count equally.
    """
    taus, skews, errs = [], [], []
    for exp, rows in points.items():
        if len(rows) < 4:
            continue
        s, se, _ = vc.atm_skew_from_slice(
            [r['k'] for r in rows], [r['iv'] for r in rows],
            zs=[r['z'] for r in rows], weights=[r['weight'] for r in rows],
            window=window)
        if s is None or not np.isfinite(s) or s == 0:
            continue
        taus.append(ctxs[exp].tau)
        skews.append(s)
        errs.append(se)
    if len(taus) < 3:
        return None, [], [], []
    H, H_err, _, r2 = vc.estimate_hurst(taus, skews, errs)
    return (H, H_err, r2), taus, skews, errs


def slice_density(svi_params, tau, forward, discount=1.0, n=1201, n_sigma=6.0):
    """
    Risk-neutral density of one fitted SVI slice, via Breeden-Litzenberger.

    Differentiate the FITTED slice on a dense UNIFORM strike grid, never the raw
    quotes: fd_second loses an order of accuracy on non-uniform spacing (see the
    caveat in volsurf_core), and real strike grids are $1 near the money and $5
    in the wings -- precisely where the density would then be wrong.

    Returns (K, density). Density is NaN at the two endpoints.
    """
    import fit.svi as svi

    par = {p: svi_params[p] for p in svi.PARAM_NAMES}
    w0 = float(svi.raw_svi(0.0, **par))
    s_atm = math.sqrt(max(w0, 1e-12) / tau)

    def density_on(half_width, npts):
        K = np.linspace(forward * math.exp(-half_width),
                        forward * math.exp(half_width), npts)
        k = np.log(K / forward)
        w = svi.raw_svi(k, **par)
        sig = np.sqrt(np.maximum(w, 1e-12) / tau)
        C = np.array([float(vc.bs_price(forward, Ki, si, tau, discount, 'C'))
                      for Ki, si in zip(K, sig)])
        return vc.breeden_litzenberger(K, C, discount)

    # Widen until the density actually integrates to one, and grow the grid with
    # it so the spacing stays fine.
    #
    # Two traps avoided here. Sizing from the wing volatility does not work:
    # SVI total variance grows linearly in |k|, so half = n_sigma*sigma(half)*
    # sqrt(tau) is a fixed point iteration settling near n_sigma^2*b(1+|rho|),
    # a grid spanning a thousand times spot. Sizing from the edge density does
    # not work either: a skewed slice has a fat left tail that is still 4e-3 of
    # peak when the mass is already 0.9996, so an edge test keeps widening until
    # dK is so coarse the mass falls back. Mass is the thing we want, so test it.
    # A slice that violates Durrleman is not a probability density at all, so
    # its mass may never reach one however far the grid is widened. Without a
    # cap the loop runs away -- observed reaching K = 0 to 5.9e8 with dK = 33282,
    # which hides the very negative region the panel exists to show. So: stop at
    # a hard cap, and stop as soon as widening stops buying mass.
    half = n_sigma * s_atm * math.sqrt(tau)
    half_cap = 4.0 * half
    npts = n
    K, dens = density_on(half, npts)
    prev_mass = -np.inf
    for _ in range(8):
        good = np.isfinite(dens)
        if not good.any():
            break
        mass = float(np.trapezoid(dens[good], K[good]))
        if mass >= 1.0 - 1e-3 or mass <= prev_mass + 1e-4 or half >= half_cap:
            break
        prev_mass = mass
        half = min(half * 1.4, half_cap)
        npts = min(int(npts * 1.4), 20001)     # keep dK from blowing up
        K, dens = density_on(half, npts)
    return K, dens


def hurst_status(log_sigma, min_samples=500):
    """
    H from the realised log-volatility path, or an honest statement that there
    is not yet enough history.

    The structure function measures increments across lags of many samples, so
    a few minutes of a live session cannot support it. Returning a number from
    50 points would be worse than returning nothing.
    """
    x = np.asarray(log_sigma, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < min_samples:
        return {'ready': False, 'H': None, 'linearity_r2': 0.0,
                'n': int(len(x)), 'needed': int(min_samples - len(x))}
    res = vc.hurst_from_structure(x)
    return {'ready': res['H'] is not None, 'H': res['H'],
            'linearity_r2': res['linearity_r2'], 'n': int(len(x)), 'needed': 0,
            'per_q': res['per_q']}


def hurst_agreement(h_skew, h_structure, tol=0.05):
    """
    Do the two independent roughness estimates agree?

    The skew estimate reads the option surface; the structure function reads the
    realised path. When they diverge, the market is pricing a roughness the
    underlying is not currently showing -- which is information, not an error,
    so it is surfaced rather than reconciled.
    """
    if h_skew is None or h_structure is None:
        return {'agree': None, 'gap': None}
    gap = abs(float(h_skew) - float(h_structure))
    return {'agree': gap <= tol, 'gap': gap}


def audit_surface(points, ctxs, k_probes=(-0.06, -0.03, 0.0, 0.03, 0.06)):
    """
    Static-arbitrage audit. Flags, never fixes.

    Butterfly: total variance must be convex in k within a slice. A concave
    patch implies a negative risk-neutral density, i.e. bad data or a bad fit.

    Calendar: w must be non-decreasing in tau at fixed LOG-MONEYNESS k. Note
    this probes fixed k, not fixed z: the no-calendar-arbitrage condition is
    dw/dtau >= 0 at k = ln(K/F_tau), and z rescales by sigma*sqrt(tau), so
    comparing slices at a common z compares different moneyness at each
    maturity and is not the arbitrage statement at all.

    Returns {expiry: {'butterfly': [row indices], 'calendar': [k values]}}.
    """
    out = {e: {'butterfly': [], 'calendar': []} for e in points}

    for exp, rows in points.items():
        if len(rows) < 3:
            continue
        out[exp]['butterfly'] = vc.butterfly_violations(
            [r['k'] for r in rows], [r['w'] for r in rows])

    exps = sorted(points, key=lambda e: ctxs[e].tau)
    for k0 in k_probes:
        taus, ws, owners = [], [], []
        for e in exps:
            rows = points[e]
            ks = np.array([r['k'] for r in rows])
            if len(ks) < 2 or k0 < ks.min() or k0 > ks.max():
                continue                      # do not extrapolate to make a flag
            order = np.argsort(ks)
            taus.append(ctxs[e].tau)
            ws.append(float(np.interp(k0, ks[order],
                                      np.array([r['w'] for r in rows])[order])))
            owners.append(e)
        for i in vc.calendar_violations(taus, ws):
            out[owners[i]]['calendar'].append(k0)
    return out


def select_expiries_by_target(expirations, today, target_days, min_days_to_expiry=1):
    """
    One expiry per target maturity: the listed expiry nearest each target, in
    days, never below `min_days_to_expiry`, duplicates dropped.

    Taking the first n expiries is wrong for a chain with dailies: on SPY the
    first twelve all sit inside three weeks, which leaves no term structure to
    calibrate H against.
    """
    dated = [(e, (vc.parse_ib_date(e) - today).days) for e in sorted(expirations)]
    dated = [(e, d) for e, d in dated if d >= min_days_to_expiry]
    chosen = []
    for t in target_days:
        if not dated:
            break
        e, _ = min(dated, key=lambda ed: (abs(ed[1] - t), -ed[1]))
        if e not in chosen:
            chosen.append(e)
    return sorted(chosen)


def connect_app(symbol='SPY', host='127.0.0.1', port=7497, client_id=36,
                spot_timeout=15.0, market_data_type=1, app_factory=None):
    """
    Probe, connect, resolve the underlying and its option chain, start spot.

    market_data_type: 1 live, 2 frozen (last quotes at the close), 3 delayed,
    4 delayed-frozen. If no spot arrives, live falls back to delayed and frozen
    to delayed-frozen, and the type actually in use is kept on the app.
    """
    # Fix A: answer "is anything listening?" in milliseconds
    t_probe = time.monotonic()
    if not probe_tws(host, port):
        raise ConnectionError_(
            f"Nothing is listening on {host}:{port} "
            f"(TCP probe failed in {1000*(time.monotonic()-t_probe):.0f} ms). "
            "Start TWS or IB Gateway and enable "
            "Global Configuration > API > Settings > 'Enable ActiveX and Socket Clients'."
        )

    app = (app_factory or LiveSurfaceApp)()
    app._symbol = symbol
    app.connect(host, port, clientId=client_id)
    threading.Thread(target=app.run, daemon=True).start()

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and not app.isConnected():
        if app.connect_failed.is_set():
            raise ConnectionError_(f"TWS refused the API connection on {host}:{port}.")
        time.sleep(0.05)
    if not app.isConnected():
        raise ConnectionError_(f"Timed out connecting to {host}:{port}.")

    under = Contract()
    under.symbol, under.secType = symbol, 'STK'
    under.exchange, under.currency = 'SMART', 'USD'
    app.underlying = under

    app.reqContractDetails(1, under)
    if not app.resolved.wait(timeout=10) or app.underlying_conId == 0:
        raise ConnectionError_(f"Could not resolve a conId for {symbol}.")

    app.market_data_type = int(market_data_type)
    app.reqMarketDataType(app.market_data_type)
    app.reqMktData(app.SPOT_REQ_ID, under, "", False, False, [])
    if not app.spot_ready.wait(timeout=spot_timeout / 2):
        fallback = {1: 3, 2: 4}.get(app.market_data_type)
        if fallback is None:
            raise ConnectionError_(f"No usable price for {symbol} after {spot_timeout / 2:.0f}s.")
        print(f"No {MARKET_DATA_TYPES[app.market_data_type]} spot tick; "
              f"falling back to {MARKET_DATA_TYPES[fallback]} market data (type {fallback}).")
        # The type applies to requests made AFTER it is set; the request already
        # sent under the old type is dead (354/10089). Re-send it, or the fallback
        # can never produce a price -- carried unnoticed from v2 (Session G).
        app.market_data_type = fallback
        app.cancelMktData(app.SPOT_REQ_ID)
        app.reqMarketDataType(fallback)
        app.reqMktData(app.SPOT_REQ_ID, under, "", False, False, [])
        if not app.spot_ready.wait(timeout=spot_timeout / 2):
            raise ConnectionError_(f"No usable price for {symbol} after {spot_timeout:.0f}s.")
    print(f"Spot {symbol} = {app.spot_price:.2f}  "
          f"({MARKET_DATA_TYPES[app.market_data_type]} data)")

    app.reqSecDefOptParams(2, symbol, "", "STK", app.underlying_conId)
    if not app.chain_resolved.wait(timeout=15) or not app.expirations:
        raise ConnectionError_("No SMART / multiplier-100 chain returned.")
    print(f"Trading class '{app.trading_class}': "
          f"{len(app.expirations)} expiries, {len(app.strikes)} strikes")
    return app


def seed_and_grid(app, symbol, exps, t0=None, n_sigma=N_SIGMA_BAND, max_strikes=21,
                  seed_timeout=5.0, sweep_grid=True, id_base=1000):
    """
    One self-consistent pass over the chain: both rights near the money for
    put-call parity, then the OTM grid on a sigma band around THOSE forwards.

    t0 must be an aware datetime (or naive UTC); it defaults to now in UTC.
    Seed ids start at id_base and grid ids at id_base + 1000. A recorder moves
    id_base on every cycle, so a late tick for a cancelled request can never
    land on a different contract that happens to reuse its id.
    """
    t0 = t0 or datetime.datetime.now(datetime.timezone.utc)
    spot = app.spot_price
    ctxs = build_expiry_contexts(app, exps, t0, spot)
    seed, rid = [], id_base
    for exp, ctx in ctxs.items():
        for K in ctx.strikes:
            for right in ('C', 'P'):
                app.id_map[rid] = (exp, K, right)
                seed.append((rid, make_option(symbol, exp, K, right, app.trading_class)))
                rid += 1
    if rid > id_base + 1000:
        raise ValueError("seed request ids would collide with the grid's (2000+)")
    print(f"Seed pass: {len(seed)} contracts (both rights) for put-call parity")
    ChainSweeper(app, seed, quote_timeout=seed_timeout).sweep()

    resolve_forwards(app, ctxs, spot)
    for exp in sorted(ctxs, key=lambda e: ctxs[e].tau):
        c = ctxs[exp]
        print(f"  {exp}  tau={c.tau*365:6.2f}d  F={c.forward:8.2f}  "
              f"df={c.discount:.5f}  r2={c.parity_r2:.4f}  sigma_atm={c.sigma_atm:.4f}")

    contracts = grid_contracts(app, ctxs, symbol, n_sigma=n_sigma, max_strikes=max_strikes,
                               id_base=id_base + 1000)
    print(f"Grid pass: {len(contracts)} OTM contracts on a +/-{n_sigma:g} sigma band")
    if sweep_grid:
        ChainSweeper(app, contracts).sweep()
    return ctxs, contracts


def start_app(symbol='SPY', host='127.0.0.1', port=7497, client_id=36,
              n_expiries=6, min_days_to_expiry=1, spot_timeout=15.0,
              market_data_type=1, target_days=None, app_factory=None):
    app = connect_app(symbol, host, port, client_id, spot_timeout, market_data_type,
                      app_factory=app_factory)

    # Aware UTC: a bare now() is local wall-clock time (Session G bug)
    t0 = datetime.datetime.now(datetime.timezone.utc)
    today = vc.new_york_date(t0)
    if target_days:
        exps = select_expiries_by_target(app.expirations, today, target_days,
                                         min_days_to_expiry)
    else:
        # Fix 6 (carried): no 0DTE
        dated = [(e, (vc.parse_ib_date(e) - today).days) for e in app.expirations]
        exps = [e for e, d in dated if d >= min_days_to_expiry][:n_expiries]
        skipped = [e for e, d in dated if 0 <= d < min_days_to_expiry]
        if skipped:
            print(f"Skipping 0DTE: {skipped}")
    if not exps:
        raise ConnectionError_("No expiries left after the 0DTE filter.")

    ctxs, contracts = seed_and_grid(app, symbol, exps, t0, sweep_grid=False)
    sweeper = ChainSweeper(app, contracts)
    sweeper.start()
    app.sweeper, app.ctxs = sweeper, ctxs
    return app


def _draw_frame(fig, ax, app, ctxs, max_age, z_grid, log_vol_series, state):
    """
    One frame of the dashboard. Every number comes from the same functions as before
    (surface_points, audit_surface, svi.fit_slice, roughness, slice_density); only the
    presentation is Session I's: the ui.theme dark tokens, one hue per entity, reference
    lines in muted ink, and every state in the status line with an icon and a label.
    """
    import fit.svi as svi
    from ui import theme as th
    t = th.tokens("dark")
    blue, orange = th.series(0, "dark"), th.series(1, "dark")
    ax_3d, ax_skew, ax_rough, ax_dens = ax["3d"], ax["skew"], ax["rough"], ax["dens"]
    pts = surface_points(app, ctxs, max_age)
    taus, Z, exps = build_grid(pts, ctxs, z_grid)
    if len(taus) < 2:
        return False
    status = []

    # --- the surface: total variance, one hue for magnitude -------------------------
    elev, azim = ax_3d.elev, ax_3d.azim
    ax_3d.clear()
    ax_3d.set_facecolor(t["surface"])
    for pane in (ax_3d.xaxis.pane, ax_3d.yaxis.pane, ax_3d.zaxis.pane):
        pane.set_facecolor(t["surface"])
        pane.set_edgecolor(t["grid"])
    X, Y = np.meshgrid(z_grid, np.array(taus) * 365.0)
    # NaN renders as a hole. A gap in the market is a gap here.
    ax_3d.plot_surface(X, Y, Z, cmap=th.sequential_cmap("dark"), edgecolor=t["surface"],
                       lw=0.25, alpha=0.95, rstride=1, cstride=1)
    ax_3d.set_xlabel("z = ln(K/F) / (σ√τ)", color=t["ink2"], fontsize=9, labelpad=6)
    ax_3d.set_ylabel("τ (days)", color=t["ink2"], fontsize=9, labelpad=6)
    ax_3d.set_zlabel("w = σ²τ", color=t["ink2"], fontsize=9, labelpad=6)
    ax_3d.tick_params(colors=t["ink2"], labelsize=8)
    ax_3d.set_title("Total variance surface", color=t["ink"], fontsize=11, loc="left", pad=4)
    ax_3d.view_init(elev=elev, azim=azim)
    n_quotes = sum(len(v) for v in pts.values())
    holes = int(np.isnan(Z).sum())
    # A1: static-arbitrage audit, reported not repaired
    flags = audit_surface(pts, ctxs)
    n_bf = sum(len(f['butterfly']) for f in flags.values())
    n_cal = sum(len(f['calendar']) for f in flags.values())
    status.append(("critical", f"static arbitrage: {n_bf} butterfly, {n_cal} calendar")
                  if (n_bf or n_cal) else ("good", "no static arbitrage"))

    # --- front slice, in sigma units, with a vega-weighted error bar ----------------
    ax_skew.clear()
    front = exps[0]
    rows = pts[front]
    fc = ctxs[front]
    zs = [r['z'] for r in rows]
    ivs = [r['iv'] for r in rows]
    errs = [1.0 / np.sqrt(r['weight']) if r['weight'] > 0 else 0.0 for r in rows]
    ax_skew.axvline(0, color=t["muted"], ls="--", lw=1)
    ax_skew.errorbar(zs, ivs, yerr=errs, fmt='o', color=blue, ecolor=t["muted"], elinewidth=1,
                     capsize=2, ms=5, mec=t["surface"], mew=1.0, label="quotes (±1/√weight)")
    # A2: fit the slice, and draw the fit rather than joining dots
    fit_p = svi.fit_slice([r['k'] for r in rows], [r['w'] for r in rows],
                          weights=[r['weight'] for r in rows], tau=fc.tau)
    if fit_p is not None:
        kk = np.linspace(min(r['k'] for r in rows), max(r['k'] for r in rows), 200)
        ww = svi.raw_svi(kk, **{p: fit_p[p] for p in svi.PARAM_NAMES})
        zz = kk / (fc.sigma_atm * math.sqrt(fc.tau))
        ax_skew.plot(zz, np.sqrt(np.maximum(ww, 1e-12) / fc.tau), '-', color=blue, lw=2.0, alpha=0.9,
                     label=f"SVI fit (rmse {fit_p['rmse']:.1e})")
        ok, wide = fit_p.get('durrleman'), fit_p.get('durrleman_wide')
        if not ok:
            status.append(("critical", "SVI: Durrleman condition violated"))
        elif wide is False:
            status.append(("warning", "SVI: wings unconstrained"))
        else:
            status.append(("good", "SVI: Durrleman holds"))
    else:
        status.append(("warning", "SVI: no fit on the front slice"))
    # A1: mark the quotes the audit flagged
    bad = [i for i in flags[front]['butterfly'] if 0 <= i < len(rows)]
    if bad:
        ax_skew.plot([rows[i]['z'] for i in bad], [rows[i]['iv'] for i in bad], 'x', color=th.STATUS["critical"],
                     ms=9, mew=2, label="butterfly flag")
    ax_skew.set_xlabel("z (σ from the forward)")
    ax_skew.set_ylabel("implied vol")
    ax_skew.set_title(f"Front slice {front}   F {fc.forward:.2f}   parity r² {fc.parity_r2:.3f}")
    ax_skew.legend(loc="upper right")

    # --- roughness: the log-log ATM skew slope is H - 1/2 ----------------------------
    ax_rough.clear()
    res, rt, rs, re = roughness(pts, ctxs)
    if res is not None:
        H, H_err, r2 = res
        rt_d = np.array(rt) * 365.0
        rs_a = np.abs(np.array(rs))
        # Error bars come from the local fit, so a badly determined short-dated skew
        # looks badly determined
        yerr = np.array([e if (e is not None and np.isfinite(e)) else 0.0 for e in re])
        ax_rough.errorbar(rt_d, rs_a, yerr=yerr, fmt='o', color=orange, ecolor=t["muted"], elinewidth=1,
                          capsize=2, ms=6, mec=t["surface"], mew=1.0, label="ATM skew per expiry")
        xs = np.linspace(min(rt), max(rt), 60)
        # Anchor the line on the fitted intercept, not on one point
        _, _, icpt, _ = vc.estimate_hurst(rt, rs, re)
        ax_rough.plot(xs * 365.0, np.exp(icpt) * xs ** (H - 0.5), '-', color=orange, lw=2.0, alpha=0.85,
                      label=f"τ^(H−½), H = {H:.3f}" + (f" ± {H_err:.3f}" if H_err else ""))
        ax_rough.set_xscale('log')
        ax_rough.set_yscale('log')
        ax_rough.set_title(f"ATM skew term structure   Ĥ_skew {H:.3f}   r² {r2:.3f}")
        ax_rough.legend(loc="upper right")
        # A4: the second, independent estimate from the realised path
        st = hurst_status(log_vol_series) if log_vol_series is not None else {'ready': False, 'H': None,
                                                                               'needed': None}
        if st['ready']:
            agr = hurst_agreement(H, st['H'])
            status.append(("good", f"Ĥ_skew and Ĥ_path agree (gap {agr['gap']:.3f})") if agr['agree']
                          else ("warning", f"Ĥ_skew {H:.3f} and Ĥ_path {st['H']:.3f} diverge"))
        else:
            need = st.get('needed')
            status.append((None, f"Ĥ_path: need {need} more samples" if need else "Ĥ_path: no history"))
    else:
        ax_rough.set_title("ATM skew term structure: needs 3+ expiries")
    ax_rough.set_xlabel("τ (days)")
    ax_rough.set_ylabel("|∂σ/∂k|")
    # Default log minor labels collide over a narrow decade range
    for axis in (ax_rough.xaxis, ax_rough.yaxis):
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())

    # --- A3: risk-neutral density of the fitted front slice --------------------------
    # Differentiated on a dense uniform grid, never on raw strikes -- fd_second loses
    # an order on uneven spacing.
    ax_dens.clear()
    if fit_p is not None:
        Kd, dens = slice_density(fit_p, fc.tau, fc.forward, fc.discount)
        good = np.isfinite(dens)
        mass = float(np.trapezoid(dens[good], Kd[good])) if good.any() else 0.0
        neg = float(np.nanmin(dens)) if good.any() else 0.0
        ax_dens.axhline(0.0, color=t["axis"], lw=0.8)
        ax_dens.axvline(fc.forward, color=t["muted"], ls="--", lw=1)
        ax_dens.fill_between(Kd[good], 0, dens[good], color=blue, alpha=0.18, lw=0)
        ax_dens.plot(Kd[good], dens[good], '-', color=blue, lw=2.0)
        if neg < -1e-10:
            badd = good & (dens < 0)
            ax_dens.plot(Kd[badd], dens[badd], 'o', color=th.STATUS["critical"], ms=4)
            status.append(("critical", "density negative: butterfly arbitrage"))
        else:
            status.append(("good", f"density non-negative, ∫ = {mass:.4f}"))
        ax_dens.set_xlim(fc.forward * 0.75, fc.forward * 1.25)
        ax_dens.set_title(f"Risk-neutral density, {front}")
    else:
        ax_dens.set_title("Risk-neutral density: no SVI fit")
    ax_dens.set_xlabel("strike")
    ax_dens.set_ylabel("q(K)")

    # --- header and status line --------------------------------------------------------
    for art in state.get("chrome", []):
        art.remove()
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S UTC")
    mdt = {1: "live", 2: "frozen", 3: "delayed", 4: "delayed-frozen"}.get(getattr(app, "market_data_type", 1), "live")
    title = fig.text(0.012, 0.965, f"{getattr(app, '_symbol', 'SPY')}  ·  live volatility surface",
                     color=t["ink"], fontsize=15, fontweight="semibold", va="baseline")
    sub = fig.text(0.012, 0.94, f"updated {now}   ·   {n_quotes} usable quotes on {len(exps)} expiries   "
                                f"·   {holes} grid holes, not filled   ·   market data: {mdt}"
                                + ("   ·   LOCKED" if state["locked"] else ""),
                   color=t["ink2"], fontsize=9.5, va="baseline")
    bar = th.status_line(fig, status, x=0.012, y=0.015, mode="dark", fontsize=9.5)
    state["chrome"] = [title, sub, bar]
    state["status"] = status
    return True


def live_desktop_plot(app, ctxs=None, max_age=30.0, log_vol_series=None, frames=None, save=None):
    """
    The live dashboard. frames / save (Session I): draw that many frames, write the last
    one to `save` as a PNG and return -- for an offline demo (--demo) or a check.
    Keys: L locks the view, S saves a snapshot to captures/live/.
    """
    global plt, ScalarFormatter, NullFormatter
    import matplotlib
    if save and frames:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    from ui import theme as th
    t = th.apply("dark")

    if log_vol_series is None:
        log_vol_series = getattr(app, 'log_vol_history', None)

    ctxs = ctxs if ctxs is not None else app.ctxs
    if not (save and frames):
        plt.ion()
    fig = plt.figure(figsize=(18, 11))
    try:
        fig.canvas.manager.set_window_title('Live volatility surface')
    except AttributeError:
        pass
    ax = {"3d": plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=2, projection='3d'),
          "skew": plt.subplot2grid((3, 3), (0, 2)),
          "rough": plt.subplot2grid((3, 3), (1, 2)),
          "dens": plt.subplot2grid((3, 3), (2, 2))}
    fig.subplots_adjust(left=0.01, right=0.975, top=0.9, bottom=0.08, hspace=0.62, wspace=0.22)
    state = {"locked": False, "chrome": []}
    snap_dir = pathlib.Path(__file__).parent / "captures" / "live"
    # matplotlib binds 's' to its save dialog and 'l' / 'L' / 'k' to log axes: release them,
    # or one key press would do two things
    for km, drop in (("keymap.save", ("s",)), ("keymap.yscale", ("l",)), ("keymap.xscale", ("k", "L"))):
        plt.rcParams[km] = [k for k in plt.rcParams[km] if k not in drop]

    from matplotlib.widgets import Button
    buttons = {}
    for key, rect, label in (("lock", [0.845, 0.009, 0.062, 0.032], "Lock  (L)"),
                             ("snap", [0.912, 0.009, 0.076, 0.032], "Snapshot  (S)")):
        b = Button(fig.add_axes(rect), label, color=t["surface"], hovercolor=t["grid"])
        b.label.set_color(t["ink"])
        b.label.set_fontsize(9)
        for sp in b.ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(t["axis"])
        buttons[key] = b
    state["buttons"] = buttons            # widgets must stay referenced to stay live

    def toggle(_=None):
        state["locked"] = not state["locked"]
        buttons["lock"].label.set_text("Unlock  (L)" if state["locked"] else "Lock  (L)")
        print("view locked" if state["locked"] else "view unlocked")
        fig.canvas.draw_idle()

    def snapshot(_=None):
        snap_dir.mkdir(parents=True, exist_ok=True)
        path = snap_dir / f"surface_{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%SZ}.png"
        fig.savefig(path, dpi=120)
        print(f"snapshot saved: {path}")

    def on_key(event):
        if event.key in ("l", "L"):
            toggle()
        elif event.key in ("s", "S"):
            snapshot()

    buttons["lock"].on_clicked(toggle)
    buttons["snap"].on_clicked(snapshot)
    fig.canvas.mpl_connect("key_press_event", on_key)
    z_grid = np.linspace(-N_SIGMA_BAND, N_SIGMA_BAND, 41)
    print("--- live dashboard (L lock, S snapshot, Ctrl+C quit) ---")
    drawn = 0
    try:
        while True:
            if not state["locked"]:
                if _draw_frame(fig, ax, app, ctxs, max_age, z_grid, log_vol_series, state):
                    drawn += 1
            if save and frames:
                if drawn >= frames:
                    fig.savefig(save, dpi=110)
                    plt.close(fig)
                    return state.get("status", [])
                time.sleep(0.5)
            else:
                plt.pause(0.5)

    except KeyboardInterrupt:
        print("\nShutting down...")
        sw = getattr(app, 'sweeper', None)
        if sw is not None:
            sw.stop()
            sw.cancel_all()
        try:
            app.cancelMktData(app.SPOT_REQ_ID)
        except Exception:
            pass
        app.disconnect()
        plt.close()


def _demo_app(target_days):
    """The fake TWS of the tests, for the dashboard without IB Gateway (--demo)."""
    import fake_ib
    RateLimiter.acquire = lambda self: None
    global probe_tws
    probe_tws = lambda *a, **k: True
    today = vc.new_york_date(datetime.datetime.now(datetime.timezone.utc))
    cls = fake_ib.make_fake(LiveSurfaceApp, today=today)
    return start_app('SPY', port=4001, app_factory=cls, spot_timeout=2.0, target_days=target_days)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Live SPY volatility surface (TWS / IB Gateway), or --demo offline.")
    ap.add_argument("--port", type=int, default=7497, help="7497 TWS paper, 7496 TWS live, 4001/4002 Gateway")
    ap.add_argument("--demo", action="store_true", help="no TWS: the tests' fake market, priced from a rough smile")
    ap.add_argument("--frames", type=int, default=None, help="with --save: draw this many frames, then exit")
    ap.add_argument("--save", default=None, help="write the last frame to this PNG")
    args = ap.parse_args()
    targets = (2, 5, 10, 21, 45, 90, 180, 365)
    try:
        instance = _demo_app(targets) if args.demo else start_app('SPY', port=args.port)
    except ConnectionError_ as exc:
        print(f"\nStartup failed: {exc}")
        raise SystemExit(1)
    print("App Started")
    time.sleep(3 if args.demo else 10)
    live_desktop_plot(instance, frames=args.frames, save=args.save)
