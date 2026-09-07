"""
Volatility surface -- v2.

Scope of this revision: ONLY the six issues that break at runtime. The maths,
the coordinate system and the plotting are deliberately left as they were in the
alpha so the two versions can be compared like for like.

  Fix 1  version-agnostic error() callback
  Fix 2  bounded wait for spot, with bid/ask + delayed-data fallbacks
  Fix 3  market-data line cap + message pacing + cancellation
  Fix 4  tradingClass / multiplier filter on the option chain
  Fix 5  correct generic tick list (106 is the underlying's 30-day IV, not per-contract)
  Fix 6  0DTE excluded

The alpha is preserved byte-identical at alpha/volatility_surface_1_ALPHA.py.
"""

# Import threading for non-blocking API communication
import threading

# Import time for sleep intervals and timestamps
import time

# Import datetime for real expiry arithmetic (Fix 6)
import datetime

# Import pandas for data manipulation and pivot tables
import pandas as pd

# Import numpy for grid generation and numerical operations
import numpy as np

# Import matplotlib for 2D and 3D visualization
import matplotlib.pyplot as plt
# Import Button widget for UI interactivity
from matplotlib.widgets import Button

# Import Interactive Brokers API components
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

# Use dark background style for a professional terminal aesthetic
plt.style.use('dark_background')


# --- Tick type constants (verified against ibapi.ticktype.TickTypeEnum) ---
TICK_BID, TICK_ASK, TICK_LAST, TICK_CLOSE = 1, 2, 4, 9
TICK_DELAYED_BID, TICK_DELAYED_ASK = 66, 67
TICK_DELAYED_LAST, TICK_DELAYED_CLOSE = 68, 75
TICK_MODEL_OPTION = 13

# Map every price tick we are willing to accept for the underlying onto a role.
# Fix 2: the alpha accepted only LAST(4) and CLOSE(9), which is why it hung
# outside regular trading hours and on any delayed-data entitlement.
SPOT_TICK_ROLES = {
    TICK_LAST: 'last',   TICK_DELAYED_LAST: 'last',
    TICK_BID: 'bid',     TICK_DELAYED_BID: 'bid',
    TICK_ASK: 'ask',     TICK_DELAYED_ASK: 'ask',
    TICK_CLOSE: 'close', TICK_DELAYED_CLOSE: 'close',
}

# Connectivity notices that are informational, not failures.
BENIGN_CODES = {2104, 2106, 2107, 2119, 2158}

# IB's documented ceiling is 50 messages/second; stay well under it.
MAX_MSG_PER_SEC = 40
# IB's default entitlement is 100 concurrent market data lines.
MAX_MARKET_DATA_LINES = 90


class RateLimiter:
    """Token bucket used to keep every outbound request under IB's pacing limit."""

    def __init__(self, per_second):
        self._interval = 1.0 / float(per_second)
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self):
        # Serialise callers so two threads cannot both claim the same slot
        with self._lock:
            now = time.monotonic()
            if now < self._next_slot:
                time.sleep(self._next_slot - now)
            self._next_slot = time.monotonic() + self._interval


class ConnectionError_(RuntimeError):
    """Raised when TWS/Gateway is unreachable, instead of spinning forever."""


class LiveSurfaceApp(EWrapper, EClient):
    # Reserved request id for the underlying's market data stream
    SPOT_REQ_ID = 999

    def __init__(self):
        # Initialize the API client
        EClient.__init__(self, self)
        # Guards every structure written by the reader thread and read by the UI
        self._lock = threading.RLock()
        # reqId -> (implied vol, monotonic timestamp) so quotes can be aged out
        self.iv_dict = {}
        # Map request IDs to (Expiration, Strike) tuples
        self.id_map = {}
        # Lists to hold available option chain parameters
        self.expirations = []
        self.strikes = []
        # The trading class we actually locked onto (Fix 4)
        self.trading_class = None
        # Variables to track the underlying asset state
        self.spot_price = 0.0
        self._spot_quotes = {}
        self.underlying_conId = 0
        # The symbol we are resolving, needed to filter the chain (Fix 4)
        self._symbol = None
        # Threading events to manage asynchronous data flow
        self.resolved = threading.Event()
        self.chain_resolved = threading.Event()
        self.spot_ready = threading.Event()
        self.connect_failed = threading.Event()

    # Callback triggered when connection to TWS/Gateway is successful
    def connectAck(self):
        print("TWS Acknowledged Connection")

    # --- Fix 1 -------------------------------------------------------------
    # ibapi has changed this signature twice and PyPI only serves 9.81.1, while
    # IBKR's own installer ships 10.19+. Accept every shape rather than betting
    # on one:
    #   <=9.81   (reqId, code, msg)
    #   10.10+   (reqId, code, msg, advancedOrderRejectJson)
    #   10.19+   (reqId, errorTime, code, msg, advancedOrderRejectJson)
    def error(self, *args):
        if len(args) < 3:
            print(f"IBKR Msg (unparsed): {args}")
            return
        # The 5-arg form is the only one carrying an epoch-ms timestamp at [1]
        if len(args) >= 5:
            req_id, code, msg = args[0], args[2], args[3]
        else:
            req_id, code, msg = args[0], args[1], args[2]

        # 502 means TWS is not listening; unblock every waiter rather than hang
        if code == 502:
            self.connect_failed.set()
            self.resolved.set()
            self.chain_resolved.set()
            self.spot_ready.set()

        if code not in BENIGN_CODES:
            print(f"IBKR Msg {req_id}: {code} - {msg}")

    # Callback receiving contract details like the unique contract ID (conId)
    def contractDetails(self, reqId, contractDetails):
        self.underlying_conId = contractDetails.contract.conId
        self.resolved.set()

    def contractDetailsEnd(self, reqId):
        # Always release the waiter, even if no details ever arrived
        self.resolved.set()

    # --- Fix 2 -------------------------------------------------------------
    # Accept last, bid/ask mid, close, and every delayed equivalent.
    def tickPrice(self, reqId, tickType, price, attrib):
        if reqId != self.SPOT_REQ_ID or price <= 0:
            return
        role = SPOT_TICK_ROLES.get(tickType)
        if role is None:
            return
        with self._lock:
            self._spot_quotes[role] = price
            resolved = self._resolve_spot()
            if resolved > 0:
                self.spot_price = resolved
                self.spot_ready.set()

    def _resolve_spot(self):
        """Best available spot, in order of preference."""
        q = self._spot_quotes
        if q.get('last', 0) > 0:
            return q['last']
        if q.get('bid', 0) > 0 and q.get('ask', 0) > 0:
            return 0.5 * (q['bid'] + q['ask'])
        return q.get('close', 0.0)

    # --- Fix 4 -------------------------------------------------------------
    # SPY publishes several trading classes (weeklies, minis, adjusted
    # contracts) with different multipliers. Accepting all of them silently
    # mixes incompatible contracts into one grid.
    def securityDefinitionOptionParameter(self, reqId, exchange, underlyingConId,
                                          tradingClass, multiplier, expirations, strikes):
        if exchange != "SMART":
            return
        if str(multiplier) != "100":
            return
        if self._symbol is not None and tradingClass != self._symbol:
            return
        with self._lock:
            self.trading_class = tradingClass
            self.expirations = sorted(list(expirations))
            self.strikes = sorted(list(strikes))
        self.chain_resolved.set()

    def securityDefinitionOptionParameterEnd(self, reqId):
        # Release the waiter even when nothing matched the filter
        self.chain_resolved.set()

    # Callback receiving calculated greeks and IV from the IBKR model
    def tickOptionComputation(self, reqId, tickType, tickAttrib, impliedVol, delta,
                              optPrice, pvDividend, gamma, vega, theta, undPrice):
        # tickType 13 is the model's Implied Volatility calculation
        if tickType != TICK_MODEL_OPTION:
            return
        if impliedVol is None or not np.isfinite(impliedVol) or impliedVol <= 0:
            return
        with self._lock:
            self.iv_dict[reqId] = (float(impliedVol), time.monotonic())

    def iv_arrived_after(self, req_id, t0):
        """True once a quote for req_id landed after this subscription started."""
        with self._lock:
            rec = self.iv_dict.get(req_id)
        return rec is not None and rec[1] >= t0

    def fresh_quotes(self, max_age):
        """Snapshot of (expiry, strike, iv) for quotes newer than max_age seconds."""
        now = time.monotonic()
        with self._lock:
            items = list(self.iv_dict.items())
            id_map = dict(self.id_map)
        out = []
        for rid, (iv, ts) in items:
            if now - ts > max_age:
                continue
            mapped = id_map.get(rid)
            if mapped is None:
                continue
            out.append({'Expiry': mapped[0], 'Strike': mapped[1], 'IV': iv})
        return out


# --- Fix 3 -----------------------------------------------------------------
class ChainSweeper(threading.Thread):
    """
    Continuously refreshes the chain while respecting IB's two hard limits.

    The alpha fired ~162 concurrent reqMktData calls at 100/second. IB allows
    ~100 concurrent lines and ~50 messages/second, and the alpha never called
    cancelMktData, so lines leaked until the session was exhausted.

    This keeps at most `max_lines` subscriptions open at once, paces every
    outbound message, and cancels each line as soon as its IV lands (or once it
    is clear no quote is coming).
    """

    def __init__(self, app, contracts, max_lines=MAX_MARKET_DATA_LINES,
                 req_per_sec=MAX_MSG_PER_SEC, quote_timeout=8.0):
        super().__init__(daemon=True)
        self.app = app
        self.contracts = contracts
        self.max_lines = max_lines
        self.quote_timeout = quote_timeout
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
            self._sweep()
            self.sweeps += 1

    def _sweep(self):
        pending = list(self.contracts)
        self._active = {}
        while (pending or self._active) and not self._stop_flag.is_set():
            # Open new lines only up to the cap
            while pending and len(self._active) < self.max_lines:
                req_id, contract = pending.pop(0)
                self.limiter.acquire()
                # Fix 5: empty generic tick list. Option computations (10-13)
                # stream automatically on an OPT subscription; generic tick 106
                # is OPTION_IMPLIED_VOL (tick 24), the underlying's 30-day IV,
                # which is not what this grid wants.
                self.app.reqMktData(req_id, contract, "", False, False, [])
                self._active[req_id] = time.monotonic()

            # Retire lines that have delivered, or that clearly will not
            now = time.monotonic()
            finished = [rid for rid, t0 in self._active.items()
                        if self.app.iv_arrived_after(rid, t0)
                        or (now - t0) > self.quote_timeout]
            for rid in finished:
                self.limiter.acquire()
                self.app.cancelMktData(rid)
                self._active.pop(rid, None)

            time.sleep(0.05)


# Wrapper function to run the client messaging loop
def run_loop(app):
    app.run()


def _to_date(yyyymmdd):
    return datetime.date(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]))


def select_expiries(expirations, today, min_days_to_expiry=1, n_expiries=6):
    """
    Fix 6: keep the next n expiries that are at least min_days_to_expiry away.

    Returns (kept, skipped). The alpha used `e >= today`, which admits today's
    expiry: T = 0, so the model IV degenerates and ln(K/F)/(sigma*sqrt(T)) is a
    division by zero.
    """
    dated = [(e, (_to_date(e) - today).days) for e in sorted(expirations)]
    kept = [e for e, d in dated if d >= min_days_to_expiry][:n_expiries]
    skipped = [e for e, d in dated if 0 <= d < min_days_to_expiry]
    return kept, skipped


def select_strikes(strikes, spot, band=0.02):
    """Strikes inside +/- band of spot."""
    return [s for s in sorted(strikes) if spot * (1 - band) <= s <= spot * (1 + band)]


def start_app(symbol='SPY', host='127.0.0.1', port=7497, client_id=35,
              n_expiries=6, strike_band=0.02, min_days_to_expiry=1,
              connect_timeout=10.0, spot_timeout=15.0):
    """Connect, resolve the chain, and start a paced sweeper. Never hangs."""
    app = LiveSurfaceApp()
    app._symbol = symbol
    app.connect(host, port, clientId=client_id)

    # Start the API message loop in a background thread
    api_thread = threading.Thread(target=run_loop, args=(app,), daemon=True)
    api_thread.start()

    # Fix 2 (part 1): bounded wait for the socket instead of a bare sleep(1)
    deadline = time.monotonic() + connect_timeout
    while time.monotonic() < deadline:
        if app.connect_failed.is_set():
            raise ConnectionError_(
                f"TWS/Gateway is not accepting API connections on {host}:{port}. "
                "Start TWS or IB Gateway and enable "
                "Global Configuration > API > Settings > 'Enable ActiveX and Socket Clients'."
            )
        if app.isConnected():
            break
        time.sleep(0.1)
    else:
        raise ConnectionError_(f"Timed out connecting to {host}:{port} after {connect_timeout}s.")

    # Define the underlying stock contract
    underlying = Contract()
    underlying.symbol = symbol
    underlying.secType = 'STK'
    underlying.exchange = 'SMART'
    underlying.currency = 'USD'

    # Request contract details to find the internal conId
    app.reqContractDetails(1, underlying)
    if not app.resolved.wait(timeout=10) or app.underlying_conId == 0:
        raise ConnectionError_(f"Could not resolve a conId for {symbol}.")

    # Fix 2 (part 2): try live data, then fall back to delayed rather than spin
    app.reqMarketDataType(1)
    app.reqMktData(app.SPOT_REQ_ID, underlying, "", False, False, [])
    if not app.spot_ready.wait(timeout=spot_timeout / 2):
        print("No live spot tick; falling back to delayed market data (type 3).")
        app.reqMarketDataType(3)
        if not app.spot_ready.wait(timeout=spot_timeout / 2):
            raise ConnectionError_(
                f"No usable price for {symbol} on live or delayed data after "
                f"{spot_timeout:.0f}s. Check the market data subscription."
            )
    if app.connect_failed.is_set():
        raise ConnectionError_("Lost the connection to TWS while resolving spot.")

    spot = app.spot_price
    print(f"Spot {symbol} = {spot:.2f}")

    # Request the option chain parameters (strikes/expirations)
    app.reqSecDefOptParams(2, symbol, "", "STK", app.underlying_conId)
    if not app.chain_resolved.wait(timeout=15) or not app.expirations:
        raise ConnectionError_(
            f"No SMART / multiplier-100 / tradingClass=={symbol} option chain returned."
        )
    print(f"Locked onto trading class '{app.trading_class}' "
          f"({len(app.expirations)} expiries, {len(app.strikes)} strikes)")

    # Fix 6: drop 0DTE. At T=0 the model IV degenerates and every moneyness
    # calculation divides by zero.
    target_exps, skipped = select_expiries(
        app.expirations, datetime.date.today(), min_days_to_expiry, n_expiries)
    if skipped:
        print(f"Skipping {len(skipped)} expiry/expiries inside {min_days_to_expiry}d: {skipped}")

    target_strikes = select_strikes(app.strikes, spot, strike_band)
    if not target_exps or not target_strikes:
        raise ConnectionError_("Chain filter left nothing to subscribe to.")

    # Build the contract list once; the sweeper owns subscribe/cancel from here
    contracts = []
    req_id = 1000
    for exp in target_exps:
        for strike in target_strikes:
            opt = Contract()
            opt.symbol = symbol
            opt.secType = 'OPT'
            opt.exchange = 'SMART'
            opt.currency = 'USD'
            opt.lastTradeDateOrContractMonth = exp
            opt.strike = strike
            # Use the OTM wing on each side of spot
            opt.right = 'C' if strike >= spot else 'P'
            # Fix 4: pin the class so IB cannot resolve an ambiguous contract
            if app.trading_class:
                opt.tradingClass = app.trading_class
            app.id_map[req_id] = (exp, strike)
            contracts.append((req_id, opt))
            req_id += 1

    n = len(contracts)
    print(f"{n} contracts | {len(target_exps)} expiries x {len(target_strikes)} strikes")
    print(f"Sweeping at most {MAX_MARKET_DATA_LINES} concurrent lines, "
          f"<= {MAX_MSG_PER_SEC} msg/s "
          f"(~{2 * n / MAX_MSG_PER_SEC:.0f}s per full sweep)")

    sweeper = ChainSweeper(app, contracts)
    sweeper.start()
    app.sweeper = sweeper
    return app


# Helper class to manage UI state for the plot
class PlotState:
    def __init__(self, button):
        self.is_locked = False
        self.button = button

    # Function to toggle data updates on/off via button
    def toggle(self, event):
        self.is_locked = not self.is_locked
        self.button.label.set_text("UNLOCK UPDATES" if self.is_locked else "LOCK UPDATES")
        plt.draw()


# Main visualization loop
def live_desktop_plot(app, max_quote_age=30.0):
    # Enable interactive mode for real-time updates
    plt.ion()
    fig = plt.figure(figsize=(16, 9))
    fig.canvas.manager.set_window_title('Quant Guild - Live Volatility Surface')
    fig.patch.set_facecolor('#0b0d0f')

    # Define subplots: 3D Surface on left, 2D Skew on right
    ax_3d = plt.subplot2grid((1, 3), (0, 0), colspan=2, projection='3d')
    ax_skew = plt.subplot2grid((1, 3), (0, 2))

    # Initialize UI Button for locking/unlocking the view
    ax_button = plt.axes([0.42, 0.03, 0.12, 0.04])
    btn = Button(ax_button, 'LOCK UPDATES', color='#1f2329', hovercolor='#2d333b')
    btn.label.set_color('white')
    btn.label.set_fontsize(9)
    state = PlotState(btn)
    btn.on_clicked(state.toggle)

    print("--- Quant Guild Desktop Live Mode Started ---")

    try:
        while True:
            # Check if UI is not locked before refreshing data
            if not state.is_locked:
                # Because the sweeper cycles lines, quotes must be aged out or
                # the grid slowly fills with stale values from earlier sweeps.
                current_data = app.fresh_quotes(max_quote_age)

                # Minimum data threshold to build a meaningful surface
                if len(current_data) > 10:
                    # Convert to DataFrame and Pivot to create a grid
                    df = pd.DataFrame(current_data)
                    pivot = df.pivot_table(index='Expiry', columns='Strike', values='IV')
                    pivot = pivot.sort_index().sort_index(axis=1)
                    # Fill missing data points using linear interpolation
                    pivot = pivot.interpolate(method='linear', axis=0).bfill().ffill()

                    # Create coordinate meshes for 3D plotting
                    X, Y_idx = np.meshgrid(pivot.columns, np.arange(len(pivot.index)))
                    Z = pivot.values

                    # Save current camera angle to prevent reset during redraw
                    curr_elev, curr_azim = ax_3d.elev, ax_3d.azim

                    # Redraw the 3D IV Surface
                    ax_3d.clear()
                    ax_3d.set_facecolor('#0b0d0f')
                    ax_3d.plot_surface(X, Y_idx, Z, cmap='magma',
                                       edgecolor='white', lw=0.1, alpha=0.9)

                    # Format 3D axis labels and title
                    ax_3d.set_yticks(np.arange(len(pivot.index)))
                    ax_3d.set_yticklabels(pivot.index, fontsize=8)
                    ax_3d.set_title(
                        f"LIVE IV SURFACE | {time.strftime('%H:%M:%S')} | "
                        f"{len(current_data)} fresh quotes", color='white')
                    ax_3d.view_init(elev=curr_elev, azim=curr_azim)

                    # Redraw the 2D Skew for the nearest expiration
                    ax_skew.clear()
                    ax_skew.set_facecolor('#161b22')
                    nearest_exp = pivot.index[0]
                    skew_data = pivot.iloc[0]

                    # Plot IV vs Strike and indicate the spot price line
                    ax_skew.plot(skew_data.index, skew_data.values, marker='o', color='#00f2ff')
                    ax_skew.axvline(x=app.spot_price, color='#ff3e3e', linestyle='--')
                    ax_skew.set_title(f"FRONT-MONTH SKEW: {nearest_exp}", color='white')

            # Pause to allow Matplotlib to process the drawing events
            plt.pause(0.5)

    except KeyboardInterrupt:
        # Graceful shutdown: retire every open line before dropping the socket
        print("\nShutting down...")
        sweeper = getattr(app, 'sweeper', None)
        if sweeper is not None:
            sweeper.stop()
            sweeper.cancel_all()
        try:
            app.cancelMktData(app.SPOT_REQ_ID)
        except Exception:
            pass
        app.disconnect()
        plt.close()


# Main entry point
if __name__ == "__main__":
    try:
        app_instance = start_app('SPY')
    except ConnectionError_ as exc:
        # Fix 2: fail loudly and immediately instead of spinning forever
        print(f"\nStartup failed: {exc}")
        raise SystemExit(1)

    print("App Started")
    # Buffer time to allow the first sweep to populate
    time.sleep(10)
    live_desktop_plot(app_instance)
