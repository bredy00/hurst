"""
The demo market (Session L trial): a synthetic equity market with the Fama-French-Carhart
factors, rough factor volatility and a SQLite database, where every answer is known.

  factors   MKT, SMB, HML, MOM, daily: f_t = mu / 252 + diag(sigma_t / sqrt 252) L e_t, with
            L L' = C a constant correlation and e_t iid N(0, I). Each factor's volatility is
            rough, log sigma_k,t = log sigmabar_k + eta_k Y_k,t, Y a fractional Ornstein-
            Uhlenbeck process, dY = -kappa Y dt + dB^H, driven by EXACT fractional Gaussian
            noise (models/fbm.py: Shevchenko's circulant embedding). The Hurst index can
            switch once (at shift_at, two thirds in by default: inside the out-of-sample
            years), to give an adaptive method something to adapt to.
  RV        a daily realised variance per factor, sigma^2 dt exp(omega z - omega^2 / 2): the
            measurement a real run would have, level-unbiased, ~35% noise (omega = 0.35).
  assets    r_i,t = beta_i' f_t + s_i u_i,t with NO alpha anywhere: whatever alpha a backtest
            reports is noise or cost. Loadings follow characteristics: size -> SMB (small
            caps load positively), value -> HML, a momentum loading -> MOM.
  caps      market capitalisations compound with total returns; w_mkt is their share.

The database (sqlite3, standard library) holds the market and, once the trial writes
them, its runs, daily portfolio returns, per-period factor fits and the Kalman paths:

  meta(key, value)                                  the spec, as JSON, and the seed
  assets(asset, size, value, mom_load, beta_mkt, beta_smb, beta_hml, beta_mom, idio_vol, cap0)
  factors(day, mkt, smb, hml, mom, rv_*, vol_*, regime)
  returns(day, asset, ret)
  runs, portfolio_daily, period_fits, kalman        written by portfolio.trial
"""

import json
import math
import sqlite3
from dataclasses import asdict, dataclass, field

import numpy as np

import models.fbm as fbm

FACTORS = ("MKT", "SMB", "HML", "MOM")
DT = 1.0 / 252


@dataclass(frozen=True)
class MarketSpec:
    n_assets: int = 40
    n_days: int = 3024                                  # twelve years
    seed: int = 20260925
    mu: tuple = (0.06, 0.01, 0.02, 0.04)                # annual factor premia
    vol: tuple = (0.16, 0.08, 0.09, 0.12)               # long-run factor vols, annual
    corr: tuple = ((1.0, 0.2, -0.1, -0.2), (0.2, 1.0, -0.1, 0.0),
                   (-0.1, -0.1, 1.0, -0.3), (-0.2, 0.0, -0.3, 1.0))
    eta: tuple = (0.5, 0.4, 0.4, 0.45)                  # vol of log-vol, per year^H
    kappa: float = 2.0                                  # log-vol mean reversion, per year
    H_true: tuple = (0.10, 0.10)                        # before and after shift_at
    shift_at: float = 2.0 / 3.0                         # fraction of the sample
    rv_noise: float = 0.35
    idio_vol: tuple = (0.15, 0.35)
    burn: int = 504

    @property
    def name(self):
        h1, h2 = self.H_true
        return "stationary" if h1 == h2 else f"shift {h1:g}->{h2:g}"


@dataclass
class Market:
    spec: MarketSpec
    F: np.ndarray            # (T, 4) factor returns
    RV: np.ndarray           # (T, 4) realised-variance proxies (daily units)
    VOL: np.ndarray          # (T, 4) true factor vols (annual)
    R: np.ndarray            # (T, N) asset returns
    B: np.ndarray            # (N, 4) true loadings
    size: np.ndarray
    value: np.ndarray
    mom_load: np.ndarray
    idio: np.ndarray         # (N,) annual idiosyncratic vol
    cap0: np.ndarray
    regime: np.ndarray       # (T,) 0 / 1
    caps: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.caps is None:
            self.caps = self.cap0[None, :] * np.cumprod(1.0 + self.R, axis=0)

    @property
    def T(self):
        return self.R.shape[0]

    @property
    def N(self):
        return self.R.shape[1]

    def w_mkt(self, t):
        """Market weights at the START of day t (caps after day t-1)."""
        c = self.cap0 if t == 0 else self.caps[t - 1]
        return c / c.sum()


def _fou(n, H, eta, kappa, rng, y0=0.0):
    """Euler steps of dY = -kappa Y dt + eta dB^H on the daily grid, from exact fGn."""
    xi = fbm.fgn(n, H, rng=rng) * DT ** H
    y = np.empty(n)
    prev = y0
    a = 1.0 - kappa * DT
    for t in range(n):
        prev = a * prev + eta * xi[t]
        y[t] = prev
    return y


def simulate(spec=MarketSpec()):
    rng = np.random.default_rng(spec.seed)
    n, T, N = spec.burn + spec.n_days, spec.n_days, spec.n_assets
    half = spec.burn + int(T * spec.shift_at)
    VOL = np.empty((T, 4))
    for k in range(4):
        y1 = _fou(half, spec.H_true[0], spec.eta[k], spec.kappa, rng)
        y2 = _fou(n - half, spec.H_true[1], spec.eta[k], spec.kappa, rng, y0=y1[-1])
        y = np.concatenate([y1, y2])[spec.burn:]
        # centre log-vol so the long-run level is the spec's (E e^{2 eta Y} over the sample)
        VOL[:, k] = spec.vol[k] * np.exp(y - 0.5 * np.log(np.mean(np.exp(2 * y))))
    L = np.linalg.cholesky(np.array(spec.corr))
    e = rng.standard_normal((T, 4)) @ L.T
    F = np.array(spec.mu)[None, :] * DT + VOL * math.sqrt(DT) * e
    om = spec.rv_noise
    RV = VOL ** 2 * DT * np.exp(om * rng.standard_normal((T, 4)) - 0.5 * om * om)

    size = rng.standard_normal(N)
    value = rng.standard_normal(N)
    mom_load = rng.standard_normal(N)
    B = np.column_stack([np.clip(1.0 + 0.25 * rng.standard_normal(N), 0.4, 1.6),
                         -0.4 * size, 0.4 * value, 0.3 * mom_load])
    idio = rng.uniform(*spec.idio_vol, N)
    R = F @ B.T + (idio * math.sqrt(DT))[None, :] * rng.standard_normal((T, N))
    cap0 = np.exp(10.0 + 1.0 * size)
    regime = ((np.arange(T) >= int(T * spec.shift_at)).astype(int) if spec.H_true[0] != spec.H_true[1]
              else np.zeros(T, int))
    return Market(spec, F, RV, VOL, R, B, size, value, mom_load, idio, cap0, regime)


# ------------------------------------------------------------------ the database
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS assets(asset INTEGER PRIMARY KEY, size REAL, value REAL, mom_load REAL,
    beta_mkt REAL, beta_smb REAL, beta_hml REAL, beta_mom REAL, idio_vol REAL, cap0 REAL);
CREATE TABLE IF NOT EXISTS factors(day INTEGER PRIMARY KEY, mkt REAL, smb REAL, hml REAL, mom REAL,
    rv_mkt REAL, rv_smb REAL, rv_hml REAL, rv_mom REAL,
    vol_mkt REAL, vol_smb REAL, vol_hml REAL, vol_mom REAL, regime INTEGER);
CREATE TABLE IF NOT EXISTS returns(day INTEGER, asset INTEGER, ret REAL, PRIMARY KEY(day, asset));
CREATE TABLE IF NOT EXISTS runs(run INTEGER PRIMARY KEY, strategy TEXT, params TEXT, phase TEXT,
    created TEXT);
CREATE TABLE IF NOT EXISTS portfolio_daily(run INTEGER, day INTEGER, gross REAL, cost REAL, net REAL,
    turnover REAL, beta_true REAL, gross_exposure REAL, PRIMARY KEY(run, day));
CREATE TABLE IF NOT EXISTS period_fits(run INTEGER, period INTEGER, source TEXT, first_day INTEGER,
    n INTEGER, alpha REAL, b_mkt REAL, b_smb REAL, b_hml REAL, b_mom REAL,
    se_alpha REAL, se_mkt REAL, se_smb REAL, se_hml REAL, se_mom REAL,
    PRIMARY KEY(run, period, source));
CREATE TABLE IF NOT EXISTS kalman(run INTEGER, period INTEGER, H REAL, log_c REAL, log_delta REAL,
    log_delta_eq REAL, sd_H REAL, sd_log_c REAL, sd_log_delta REAL, sd_log_delta_eq REAL,
    beta_mkt REAL, beta_mkt_sd REAL, cost_pct REAL, PRIMARY KEY(run, period));
"""


def write(market, path):
    """Write the market to a fresh SQLite file at `path` (replacing any market in it)."""
    con = sqlite3.connect(str(path))
    try:
        con.executescript(SCHEMA)
        for tbl in ("meta", "assets", "factors", "returns"):
            con.execute(f"DELETE FROM {tbl}")
        spec = asdict(market.spec)
        con.executemany("INSERT INTO meta VALUES (?, ?)",
                        [("spec", json.dumps(spec)), ("name", market.spec.name), ("schema", "demo-market/1")])
        con.executemany("INSERT INTO assets VALUES (?,?,?,?,?,?,?,?,?,?)",
                        [(i, float(market.size[i]), float(market.value[i]), float(market.mom_load[i]),
                          *map(float, market.B[i]), float(market.idio[i]), float(market.cap0[i]))
                         for i in range(market.N)])
        con.executemany("INSERT INTO factors VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        [(t, *map(float, market.F[t]), *map(float, market.RV[t]), *map(float, market.VOL[t]),
                          int(market.regime[t])) for t in range(market.T)])
        con.executemany("INSERT INTO returns VALUES (?,?,?)",
                        [(t, i, float(market.R[t, i])) for t in range(market.T) for i in range(market.N)])
        con.commit()
    finally:
        con.close()


def load(path):
    """Read a market back from its database."""
    con = sqlite3.connect(str(path))
    try:
        spec_d = json.loads(con.execute("SELECT value FROM meta WHERE key = 'spec'").fetchone()[0])
        spec = MarketSpec(**{k: (tuple(tuple(r) if isinstance(r, list) else r for r in v)
                                 if isinstance(v, list) else v) for k, v in spec_d.items()})
        a = np.array(con.execute("SELECT * FROM assets ORDER BY asset").fetchall(), dtype=float)
        f = np.array(con.execute("SELECT * FROM factors ORDER BY day").fetchall(), dtype=float)
        rows = np.array(con.execute("SELECT day, asset, ret FROM returns").fetchall(), dtype=float)
    finally:
        con.close()
    T, N = int(f.shape[0]), int(a.shape[0])
    R = np.empty((T, N))
    R[rows[:, 0].astype(int), rows[:, 1].astype(int)] = rows[:, 2]
    return Market(spec, F=f[:, 1:5], RV=f[:, 5:9], VOL=f[:, 9:13], R=R, B=a[:, 4:8], size=a[:, 1],
                  value=a[:, 2], mom_load=a[:, 3], idio=a[:, 8], cap0=a[:, 9], regime=f[:, 13].astype(int))
