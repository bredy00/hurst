"""
Self-exciting jump arrivals: the Hawkes process with an exponential kernel.

    lambda(t) = mu + sum_{t_i < t} alpha e^{-beta (t - t_i)}

Every event lifts the intensity by alpha, and the lift decays at rate beta. A
constant-intensity Poisson process is the alpha = 0 case. Built once here and
attached to three volatility hosts in models/jump_hosts.py -- an OU log-variance,
classical Heston and the lifted rough Heston -- because the clustering is a
property of the arrivals, not of whichever diffusion they land on.

Two facts carry most of what follows.

  Stationarity. The mean number of children of one event is the kernel's
  integral, the branching ratio n = alpha / beta. The process is stationary
  iff n < 1, with mean intensity lambda_bar = mu / (1 - n). HawkesParams
  refuses n >= 1 rather than simulating an explosion.

  Mean dynamics are linear. d E[lambda] / dt = beta mu - (beta - alpha) E[lambda],
  so an extra event's EXPECTED excess intensity is alpha e^{-(beta - alpha) s}:
  it decays at beta - alpha, not beta, because each child spawns children.
  Everything analytic in this module and in jump_hosts follows from that line.

Time is in years throughout, rates per year. Numpy only, except the maximum-
likelihood fit, which uses scipy.optimize (allowed outside volsurf_core).
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HawkesParams:
    mu: float           # baseline intensity, events per year
    alpha: float        # jump in intensity per event, per year
    beta: float         # decay rate of that jump, per year

    def __post_init__(self):
        if not self.mu > 0:
            raise ValueError(f"mu must be positive, got {self.mu}")
        if self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        if not self.beta > 0:
            raise ValueError(f"beta must be positive, got {self.beta}")
        if self.alpha >= self.beta:
            raise ValueError(
                f"non-stationary: alpha = {self.alpha} >= beta = {self.beta}. The branching "
                f"ratio alpha/beta = {self.alpha / self.beta:.3f} is the expected number of "
                f"children per event; at or above one the event count explodes.")

    @property
    def branching_ratio(self):
        return self.alpha / self.beta

    @property
    def stationary_intensity(self):
        return self.mu / (1.0 - self.alpha / self.beta)

    @property
    def excess_decay(self):
        """Rate at which an extra event's EXPECTED excess intensity decays: beta - alpha."""
        return self.beta - self.alpha

    @classmethod
    def poisson(cls, rate):
        return cls(float(rate), 0.0, 1.0)

    def matched_poisson(self):
        """The constant-intensity process with the same mean event rate."""
        return HawkesParams.poisson(self.stationary_intensity)


# ------------------------------------------------------------ theory
def count_variance(p, window):
    """
    Var[N(t, t + window)] for the stationary process, exact.

    From Hawkes (1971): with the exponential kernel the covariance density of
    the counting process is lambda_bar delta(tau) + C e^{-g |tau|}, where
    g = beta - alpha and C = alpha lambda_bar (2 beta - alpha) / (2 g). Integrating
    twice over the window:

        Var = lambda_bar w + 2 C (w / g - (1 - e^{-g w}) / g^2)

    which tends to lambda_bar w / (1 - n)^2 for long windows -- the familiar
    over-dispersion -- and is lambda w for Poisson.
    """
    lam = p.stationary_intensity
    if p.alpha == 0.0:
        return lam * window
    g = p.beta - p.alpha
    C = p.alpha * lam * (2.0 * p.beta - p.alpha) / (2.0 * g)
    return lam * window + 2.0 * C * (window / g - (1.0 - math.exp(-g * window)) / (g * g))


def fano_factor(p, window):
    """Var[N] / E[N] over a window. 1 for Poisson, > 1 for any clustering."""
    return count_variance(p, window) / (p.stationary_intensity * window)


def expected_intensity(p, t, planted=()):
    """
    E[lambda(t)] starting from stationarity, with events forced at `planted`.

        E[lambda(t)] = lambda_bar + sum_{t_i <= t} alpha e^{-(beta - alpha)(t - t_i)}
    """
    t = np.asarray(t, dtype=float)
    out = np.full(t.shape, p.stationary_intensity)
    g = p.excess_decay
    for ti in planted:
        out = out + np.where(t >= ti, p.alpha * np.exp(-g * np.clip(t - ti, 0.0, None)), 0.0)
    return out


def expected_count(p, a, b, planted=()):
    """E[number of events in (a, b]] including the planted ones, from stationarity."""
    g = p.excess_decay
    total = p.stationary_intensity * (b - a)
    for ti in planted:
        if a < ti <= b:
            total += 1.0
        lo = max(a, ti)
        if b > lo:
            if g > 0:
                total += (p.alpha / g) * (math.exp(-g * (lo - ti)) - math.exp(-g * (b - ti)))
    return total


# ------------------------------------------------------------ simulation
@dataclass
class EventPaths:
    """Event times per path, padded with +inf. Times may be negative (burn-in)."""
    times: np.ndarray           # (n_paths, cap)
    planted: np.ndarray         # (n_paths, cap) bool: was this event forced
    T: float
    params: HawkesParams

    @property
    def n_paths(self):
        return self.times.shape[0]

    def counts(self, a=0.0, b=None):
        b = self.T if b is None else b
        return np.count_nonzero((self.times > a) & (self.times <= b), axis=1)

    def path(self, i, include_burn_in=False):
        t = self.times[i]
        m = np.isfinite(t) if include_burn_in else (np.isfinite(t) & (t >= 0.0))
        return t[m]

    def binned(self, n_steps, T=None):
        """(n_paths, n_steps) event counts on a uniform grid of (0, T]."""
        T = self.T if T is None else T
        dt = T / n_steps
        m = np.isfinite(self.times) & (self.times > 0.0) & (self.times <= T)
        rows, cols = np.nonzero(m)
        k = np.minimum((self.times[rows, cols] / dt).astype(int), n_steps - 1)
        out = np.zeros((self.n_paths, n_steps), dtype=int)
        np.add.at(out, (rows, k), 1)
        return out

    def intensity_on_grid(self, n_steps, T=None):
        """
        lambda at the grid points t_k = k dt, k = 0..n_steps, exactly: each event's
        contribution is decayed from its own time to the next grid point, then the
        excitation is carried forward by e^{-beta dt}. O(events + paths x steps).
        """
        T = self.T if T is None else T
        p = self.params
        dt = T / n_steps
        exc0 = np.zeros(self.n_paths)
        pre = np.isfinite(self.times) & (self.times <= 0.0)
        if pre.any():
            decay = np.where(pre, np.exp(-p.beta * (0.0 - np.where(pre, self.times, 0.0))), 0.0)
            exc0 = p.alpha * decay.sum(axis=1)
        m = np.isfinite(self.times) & (self.times > 0.0) & (self.times <= T)
        rows, cols = np.nonzero(m)
        tt = self.times[rows, cols]
        k = np.minimum(np.ceil(tt / dt - 1e-12).astype(int), n_steps)
        add = np.zeros((self.n_paths, n_steps + 1))
        np.add.at(add, (rows, k), p.alpha * np.exp(-p.beta * (k * dt - tt)))
        lam = np.empty((self.n_paths, n_steps + 1))
        exc = exc0 + add[:, 0]
        lam[:, 0] = p.mu + exc
        f = math.exp(-p.beta * dt)
        for j in range(1, n_steps + 1):
            exc = exc * f + add[:, j]
            lam[:, j] = p.mu + exc
        return lam


def simulate(p, T, n_paths=1, seed=None, rng=None, planted=(), burn_in=0.0):
    """
    Exact event times on (-burn_in, T] by Ogata thinning, every path advanced
    together. `planted` are events forced at the same times in every path; they
    excite the intensity exactly like natural events (for Poisson, alpha = 0,
    they excite nothing -- which is the point of comparing the two).

    Thinning is exact because between events the intensity only decays, so its
    value just after the current time dominates it until the next event. A
    candidate that would jump past a planted event is discarded and the clock
    restarted at the planted time, which the exponential clock's memorylessness
    permits.

    Start from stationarity with `burn_in` of several 1/(beta - alpha): events in
    the burn-in are kept (with negative times) because their excitation carries
    into the window.
    """
    rng = np.random.default_rng(seed) if rng is None else rng
    n = int(n_paths)
    P = np.append(np.sort(np.asarray(planted, dtype=float)) + burn_in, np.inf)
    T_end = T + burn_in
    cap = int(max(16, 1.5 * p.stationary_intensity * T_end + len(planted) + 16))
    times = np.full((n, cap), np.inf)
    flags = np.zeros((n, cap), dtype=bool)
    cnt = np.zeros(n, dtype=int)
    t = np.zeros(n)
    exc = np.zeros(n)
    ip = np.zeros(n, dtype=int)
    active = np.ones(n, dtype=bool)

    def record(j, tt, is_planted):
        nonlocal times, flags, cap
        if j.size == 0:
            return
        need = int(cnt[j].max()) + 1
        if need > cap:
            new = max(need, 2 * cap)
            times = np.concatenate([times, np.full((n, new - cap), np.inf)], axis=1)
            flags = np.concatenate([flags, np.zeros((n, new - cap), dtype=bool)], axis=1)
            cap = new
        times[j, cnt[j]] = tt
        flags[j, cnt[j]] = is_planted
        cnt[j] += 1

    while active.any():
        idx = np.nonzero(active)[0]
        lam_bar = p.mu + exc[idx]
        w = t[idx] + rng.exponential(size=idx.size) / lam_bar
        tp = P[ip[idx]]
        hit = tp <= w
        if hit.any():
            j = idx[hit]
            exc[j] = exc[j] * np.exp(-p.beta * (tp[hit] - t[j])) + p.alpha
            t[j] = tp[hit]
            record(j, t[j].copy(), True)
            ip[j] += 1
        k = idx[~hit]
        wk = w[~hit]
        done = wk > T_end
        active[k[done]] = False
        k, wk = k[~done], wk[~done]
        if k.size:
            exc_w = exc[k] * np.exp(-p.beta * (wk - t[k]))
            accept = rng.random(k.size) * (p.mu + exc[k]) <= p.mu + exc_w
            exc[k] = exc_w + p.alpha * accept
            t[k] = wk
            record(k[accept], wk[accept], False)

    return EventPaths(times=times - burn_in, planted=flags, T=float(T), params=p)


# ------------------------------------------------------------ likelihood
def _excitation_sums(p, t):
    """A_i = sum_{j<i} e^{-beta (t_i - t_j)} by the O(n) recursion."""
    n = len(t)
    A = np.zeros(n)
    if n > 1:
        d = np.exp(-p.beta * np.diff(t))
        a = 0.0
        for i in range(1, n):
            a = d[i - 1] * (1.0 + a)
            A[i] = a
    return A


def log_likelihood(p, times, T):
    """
    log L = sum_i log lambda(t_i) - int_0^T lambda dt, for events on (0, T]:

        sum_i log(mu + alpha A_i) - mu T - (alpha / beta) sum_i (1 - e^{-beta (T - t_i)})
    """
    t = np.asarray(times, dtype=float)
    if t.size == 0:
        return -p.mu * T
    A = _excitation_sums(p, t)
    lam = p.mu + p.alpha * A
    if np.any(lam <= 0):
        return -np.inf
    return float(np.sum(np.log(lam)) - p.mu * T
                 - (p.alpha / p.beta) * np.sum(1.0 - np.exp(-p.beta * (T - t))))


def rescaled_intervals(p, times):
    """
    Time-rescaling theorem: if the model is right, the compensator increments
    Lambda(t_i) - Lambda(t_{i-1}) are i.i.d. Exp(1). With the recursion above,
    Lambda(t_i) = mu t_i + (alpha / beta) (i - A_i).
    """
    t = np.asarray(times, dtype=float)
    A = _excitation_sums(p, t)
    Lam = p.mu * t + (p.alpha / p.beta) * (np.arange(len(t)) - A)
    return np.diff(np.concatenate([[0.0], Lam]))


def fit_mle(times, T, n_starts=4, poisson=False):
    """
    Maximum-likelihood (mu, alpha, beta) from one event path, with standard
    errors from the observed information (numerical Hessian of -log L).

    Optimised in (log mu, logit n, log beta) so every trial point is a valid,
    stationary process; several starting decay rates because the likelihood is
    flat along beta when events are sparse. `poisson=True` fits alpha = 0.

    Returns dict(params, loglik, se: dict, cov, converged).
    """
    from scipy.optimize import minimize

    t = np.asarray(times, dtype=float)
    n_ev = len(t)
    rate = max(n_ev / T, 1e-9)
    if poisson:
        p = HawkesParams.poisson(rate)
        se = {"mu": math.sqrt(rate / T)}
        return {"params": p, "loglik": log_likelihood(p, t, T), "se": se,
                "cov": np.array([[se["mu"] ** 2]]), "converged": True}

    def unpack(z):
        mu = math.exp(z[0])
        nb = 1.0 / (1.0 + math.exp(-z[1]))
        beta = math.exp(z[2])
        return HawkesParams(mu, nb * beta, beta)

    def nll(z):
        try:
            return -log_likelihood(unpack(z), t, T)
        except (ValueError, OverflowError):
            return 1e300

    best = None
    mean_gap = T / max(n_ev, 1)
    for k in range(n_starts):
        beta0 = (2.0 ** (k - 1)) / mean_gap
        z0 = np.array([math.log(0.5 * rate), 0.0, math.log(beta0)])
        r = minimize(nll, z0, method="Nelder-Mead",
                     options={"xatol": 1e-8, "fatol": 1e-9, "maxiter": 4000, "maxfev": 8000})
        if best is None or r.fun < best.fun:
            best = r
    p = unpack(best.x)

    # Observed information in natural coordinates (mu, alpha, beta)
    x0 = np.array([p.mu, p.alpha, p.beta])
    h = np.maximum(1e-4 * np.abs(x0), 1e-6)

    def f(x):
        try:
            return -log_likelihood(HawkesParams(*x), t, T)
        except ValueError:
            return np.inf

    Hm = np.empty((3, 3))
    f0 = f(x0)
    for i in range(3):
        for j in range(i, 3):
            ei = np.zeros(3)
            ej = np.zeros(3)
            ei[i] = h[i]
            ej[j] = h[j]
            val = (f(x0 + ei + ej) - f(x0 + ei - ej) - f(x0 - ei + ej) + f(x0 - ei - ej)) / (4 * h[i] * h[j])
            Hm[i, j] = Hm[j, i] = val
    try:
        cov = np.linalg.inv(Hm)
        se = {name: (math.sqrt(cov[i, i]) if cov[i, i] > 0 else float("nan"))
              for i, name in enumerate(("mu", "alpha", "beta"))}
    except np.linalg.LinAlgError:
        cov = np.full((3, 3), np.nan)
        se = {name: float("nan") for name in ("mu", "alpha", "beta")}
    return {"params": p, "loglik": -float(best.fun), "se": se, "cov": cov,
            "converged": bool(best.success), "f0": f0}
