"""
One jump component, three volatility hosts.

The Hawkes arrivals in models/hawkes.py are attached, unchanged, to:

    OUHost      x = log V,  dx = kappa (theta - x) dt + sigma dW_v + J_x dN
    HestonHost  dv = kappa (theta - v) dt + xi sqrt(v) dW_v + J_v dN
    RoughHost   the lifted rough Heston of models/rough_heston.py,
                dU_i = (-x_i U_i + kappa (theta - V)) dt + xi sqrt(V) dW_v + J_v dN,
                V = v0 + sum_i w_i U_i

and in every host the log-price co-jumps:

    dX = -1/2 V dt + sqrt(V) dW + J_s dN,      d<W, W_v> = rho dt

For the rough host the jump enters the Volterra DRIVER, like the drift and the
diffusion do, so its effect on V decays through the fractional kernel
K(t - s) -- a power law -- rather than exponentially. That is the Volterra-
consistent choice, and it has a consequence measured below: a jump's variance
response at a fixed lag decays much faster at first than on the exponential
hosts, which changes when clustered shocks can compound.

EXPECTATIONS ARE EXACT. Every host's conditional mean is linear in its state and
in the jump arrivals, and E[number of events per step] has a closed form
(hawkes.expected_count). So `expected_path` iterates the host's own discrete
update with the noise removed and gives the Monte Carlo's expectation with no
discretisation gap between the two -- the simulator is checked against it, not
against a continuous-time approximation.

THE SECOND-SPIKE PROPERTY, made precise. By linearity, the expected state after
shocks at t1 and t2 = t1 + d is the sum of single-shock responses r(s). So

    E[increment after the 2nd] - E[increment after the 1st] = r(d + W) - r(d)

where an increment is the state W after a shock minus the state just before it.
The second spike is larger iff the single-shock response is still RISING
across [d, d + W]. For a constant-intensity process r is the host's own jump
decay, strictly decreasing, so the second increment is always SMALLER -- the
property is impossible, not merely unlikely. For Hawkes on the exponential
hosts r'(0) = eta (alpha - kappa): at short gaps the property holds iff the
excitation per event outpaces mean reversion. Measured on the rough host, the
kernel's power-law decay has to be overcome instead.

A LEVEL comparison (state after the second shock vs after the first) is not a
test of any of this: the first shock has not decayed, so the level is higher
after the second shock under Poisson too.
"""

import math
from dataclasses import dataclass

import numpy as np

import models.hawkes as hk


@dataclass(frozen=True)
class JumpSizes:
    state_mean: float = 0.0     # mean of the exponential jump in the host's state
    price_mean: float = 0.0     # mean log-price jump (Gaussian)
    price_std: float = 0.0      # std of the log-price jump


# ------------------------------------------------------------------ hosts
class OUHost:
    """x = log V with exact OU transitions. The state measured is x itself."""

    name = "OU (log-variance)"

    def __init__(self, kappa=3.0, theta=math.log(0.04), sigma=1.0):
        self.kappa, self.theta, self.sigma = float(kappa), float(theta), float(sigma)

    def _coef(self, dt):
        e = math.exp(-self.kappa * dt)
        sd = self.sigma * math.sqrt((1.0 - e * e) / (2.0 * self.kappa))
        return e, sd

    def stationary_state(self, jump_mean, rate, dt):
        """Discrete fixed point of the mean recursion under a constant event rate."""
        e, _ = self._coef(dt)
        return self.theta + jump_mean * rate * dt / (1.0 - e)

    def init(self, n_paths, x0):
        return np.full(n_paths, float(x0))

    def variance(self, x):
        return np.exp(x)

    def measure(self, x):
        return x

    def step(self, x, dt, z_v, jump_sum):
        e, sd = self._coef(dt)
        return self.theta + (x - self.theta) * e + sd * z_v + jump_sum

    def mean_step(self, m, dt, expected_jump):
        e, _ = self._coef(dt)
        return self.theta + (m - self.theta) * e + expected_jump


class HestonHost:
    """CIR variance, full-truncation Euler (Lord et al.). The state measured is v."""

    name = "classical Heston (variance)"

    def __init__(self, kappa=3.0, theta=0.04, xi=0.3):
        self.kappa, self.theta, self.xi = float(kappa), float(theta), float(xi)

    def stationary_state(self, jump_mean, rate, dt):
        return self.theta + jump_mean * rate / self.kappa

    def init(self, n_paths, v0):
        return np.full(n_paths, float(v0))

    def variance(self, v):
        return np.maximum(v, 0.0)

    def measure(self, v):
        return v

    def step(self, v, dt, z_v, jump_sum):
        vp = np.maximum(v, 0.0)
        return v + self.kappa * (self.theta - vp) * dt + self.xi * np.sqrt(vp * dt) * z_v + jump_sum

    def mean_step(self, m, dt, expected_jump):
        return m + self.kappa * (self.theta - m) * dt + expected_jump


def mittag_leffler(z, a, b, tol=1e-16, max_terms=400):
    """E_{a,b}(z) = sum_k z^k / Gamma(a k + b), by its series (|z| up to ~5 in double precision)."""
    if abs(z) > 5.0:
        raise ValueError(f"series Mittag-Leffler used at |z| = {abs(z):.2f} > 5: cancellation")
    total, k = 0.0, 0
    while k < max_terms:
        term = z ** k / math.gamma(a * k + b)
        total += term
        if k > 4 and abs(term) < tol * max(abs(total), 1e-300):
            break
        k += 1
    return total


def driver_impact_integral(t, H, kappa):
    """
    int_0^t of the exact continuous response of V to a unit driver impulse:
    t^alpha E_{alpha,alpha+1}(-kappa t^alpha), alpha = H + 1/2 (the resolvent of the
    rough Heston mean reversion; the lift and the grid approximate it).
    """
    a = H + 0.5
    return t ** a * mittag_leffler(-kappa * t ** a, a, a + 1.0)


class RoughHost:
    """
    Lifted rough Heston variance with Hawkes jumps through the Volterra DRIVER: a jump
    enters like the drift and the diffusion do, so its effect on V is J K(t - s) with the
    mean reversion's resolvent -- a power law. Each factor steps with its exact
    propagator e^{-x_i dt} and takes the step's driver increment through the cell mean
    phi_1(-x_i dt), the discretisation Session D chose because it keeps the kernel's
    short-lag weight.

    A JUMP'S SIZE IS ITS INTEGRATED VARIANCE IMPACT (Session I). In the exact rough model
    the kernel is infinite at zero, so a driver jump has no instantaneous size in V; on a
    lift the instantaneous move is finite but depends on the grid and the lift (40.2 per
    unit driver at 5-minute steps on the Sessions A-H lift, 17.8 at 1-hour steps), while
    the one-day integrated impact moves by at most 5.4% (docs/comparison-jump-modes.md,
    section 2). So JumpSizes.state_mean is read here as the jump's AVERAGE EXTRA VARIANCE
    over the first `impact_horizon` (default one trading day), and converted to a driver
    increment through the exact continuous response:

        driver increment = impact * horizon / int_0^horizon r(t) dt,
        int_0^t r = t^alpha E_{alpha,alpha+1}(-kappa t^alpha)        (driver_impact_integral)

    so the parameter means the same thing on every grid and every lift. The realised
    one-day impact is within 4.3% of it on the default lift at 5-minute to quarter-day
    steps, and within 5.4% on the Sessions A-H lift; the worst case on both is 1-hour
    steps (test_hawkes.py).

    Until Session I a second mode, "direct", added a separately decaying component D to
    V. It was dropped: while D is high the mean reversion pushes the rough factors down,
    and they remember that through the kernel after D has gone, so V undershoots its
    no-jump path (after ~20 days at a 25/y decay). study_jump_modes.py keeps a private
    copy of it so the Session H comparison stays reproducible.

    The state is the (N, n_paths) matrix of lifted factors. The quantity measured is V.
    """

    def __init__(self, kappa=3.0, theta=0.04, xi=0.3, H=0.12, v0=0.04, N=None, impact_horizon=1.0 / 252):
        import models.rough_heston as rh
        self.kappa, self.theta, self.xi, self.H, self.v0 = map(float, (kappa, theta, xi, H, v0))
        self.N = rh.N_DEFAULT if N is None else int(N)
        self.w, self.x = rh.lift_nodes(self.H, self.N)
        self._phi1 = rh._phi1
        self.impact_horizon = float(impact_horizon)
        self.driver_per_impact = self.impact_horizon / driver_impact_integral(self.impact_horizon, self.H, self.kappa)
        self.name = "rough Heston (variance, driver jumps)"

    def _coef(self, dt):
        return np.exp(-self.x * dt), self._phi1(-self.x * dt)

    def stationary_state(self, jump_mean, rate, dt):
        """State with E[state] a fixed point of the mean recursion under a constant rate."""
        E, g = self._coef(dt)
        n = len(self.w)
        A = np.eye(n) - np.diag(E) + self.kappa * dt * np.outer(g, self.w)
        b = g * (self.kappa * dt * (self.theta - self.v0) + jump_mean * self.driver_per_impact * rate * dt)
        return np.linalg.solve(A, b)

    def init(self, n_paths, state0):
        return np.repeat(np.asarray(state0, dtype=float)[:, None], n_paths, axis=1)

    def _V(self, S):
        return self.v0 + self.w @ S

    def variance(self, S):
        return np.maximum(self._V(S), 0.0)

    def measure(self, S):
        return self._V(S)

    def step(self, S, dt, z_v, jump_sum):
        E, g = self._coef(dt)
        Vp = np.maximum(self._V(S), 0.0)
        drive = (self.kappa * (self.theta - Vp) * dt + self.xi * np.sqrt(Vp * dt) * z_v
                 + jump_sum * self.driver_per_impact)
        return E[:, None] * S + g[:, None] * drive[None, :]

    def mean_step(self, M, dt, expected_jump):
        E, g = self._coef(dt)
        V = self.v0 + self.w @ M
        drive = self.kappa * (self.theta - V) * dt + expected_jump * self.driver_per_impact
        return E * M + g * drive


class ConstantHost:
    """Constant variance: the Merton / lecture jump diffusion, for closed-form checks."""

    name = "constant variance (Merton)"

    def __init__(self, v=0.04):
        self.v = float(v)

    def stationary_state(self, jump_mean, rate, dt):
        return self.v

    def init(self, n_paths, v0):
        return np.full(n_paths, self.v)

    def variance(self, s):
        return s

    def measure(self, s):
        return s

    def step(self, s, dt, z_v, jump_sum):
        return s

    def mean_step(self, m, dt, expected_jump):
        return m


# ------------------------------------------------------------------ simulation
def default_state(host, hawkes, jumps, dt):
    return host.stationary_state(jumps.state_mean, hawkes.stationary_intensity, dt)


def simulate(host, hawkes, jumps, T, n_steps, n_paths, seed=0, planted=(), rho=-0.7,
             burn_in=1.0, state0=None, compensate=False, record_every=1):
    """
    Paths of (X, measured state, V) on a uniform grid of n_steps over (0, T].

    Events are simulated EXACTLY (hawkes.simulate, stationary via burn_in) and
    binned into steps; all events of a step act at its end. `planted` shocks
    should sit strictly inside steps (e.g. (j + 0.5) dt) so there is no floating
    ambiguity about which step they belong to. `compensate` subtracts
    lambda(t) E[e^J - 1] dt from the log-price -- the risk-neutral drift; off
    by default because the statistical tests here are under the real-world
    measure, where it is only a random drift.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    events = hk.simulate(hawkes, T, n_paths=n_paths, rng=rng, planted=planted, burn_in=burn_in)
    counts = events.binned(n_steps, T)
    lam = events.intensity_on_grid(n_steps, T) if compensate else None
    s = host.init(n_paths, default_state(host, hawkes, jumps, dt) if state0 is None else state0)
    X = np.zeros(n_paths)
    sq = math.sqrt(dt)
    srho = math.sqrt(1.0 - rho * rho)
    k_comp = math.exp(jumps.price_mean + 0.5 * jumps.price_std ** 2) - 1.0
    n_rec = n_steps // record_every + 1
    out_state = np.empty((n_rec, n_paths))
    out_X = np.empty((n_rec, n_paths))
    out_state[0] = host.measure(s)
    out_X[0] = X
    r = 1
    for k in range(n_steps):
        V = host.variance(s)
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        z_v = rho * z1 + srho * z2
        nk = counts[:, k]
        jump_state = np.zeros(n_paths)
        jump_price = np.zeros(n_paths)
        hit = nk > 0
        if hit.any():
            m = nk[hit]
            if jumps.state_mean > 0:
                jump_state[hit] = rng.gamma(m, jumps.state_mean)
            if jumps.price_std > 0 or jumps.price_mean != 0:
                jump_price[hit] = rng.normal(m * jumps.price_mean, np.sqrt(m) * jumps.price_std)
        X = X - 0.5 * V * dt + np.sqrt(V) * sq * z1 + jump_price
        if compensate:
            X = X - lam[:, k] * k_comp * dt
        s = host.step(s, dt, z_v, jump_state)
        if (k + 1) % record_every == 0:
            out_state[r] = host.measure(s)
            out_X[r] = X
            r += 1
    t = np.arange(n_rec) * dt * record_every
    return {"t": t, "X": out_X, "state": out_state, "counts": counts, "events": events, "dt": dt}


def expected_path(host, hawkes, jumps, T, n_steps, planted=(), state0=None):
    """
    E[measured state] at every grid point, exact for the discrete model: the
    host's own update with the noise removed and each step's jump replaced by
    state_mean * E[events in the step], which hawkes.expected_count gives in
    closed form (planted events included).
    """
    dt = T / n_steps
    m = default_state(host, hawkes, jumps, dt) if state0 is None else np.asarray(state0, dtype=float)
    out = np.empty(n_steps + 1)
    out[0] = float(host.measure(m if np.ndim(m) else np.asarray(m)))
    for k in range(n_steps):
        en = hk.expected_count(hawkes, k * dt, (k + 1) * dt, planted)
        m = host.mean_step(m, dt, jumps.state_mean * en)
        out[k + 1] = float(host.measure(m))
    return out


def shock_response(host, hawkes, jumps, T, n_steps, shock_step):
    """
    r(s): the expected state with one extra (planted) event in step `shock_step`
    minus without, on the grid. Linear superposition of these is exact.
    """
    dt = T / n_steps
    t_shock = (shock_step + 0.5) * dt
    with_shock = expected_path(host, hawkes, jumps, T, n_steps, planted=(t_shock,))
    base = expected_path(host, hawkes, jumps, T, n_steps)
    return with_shock - base


def second_spike(host, hawkes, jumps, dt, gap_steps, window_steps, lead_steps=10):
    """
    Exact expected increments after two planted shocks `gap_steps` apart.

    Returns dict(inc1, inc2, level1, level2, r_d, r_dW) where incN is the state
    `window_steps` after shock N minus the state just before it, levelN the state
    `window_steps` after it, and r_d, r_dW the single-shock response at the gap
    and at gap + window -- inc2 - inc1 = r_dW - r_d identically.

    The window must be STRICTLY shorter than the gap. A first draft measured a
    five-day window after shocks two days apart, so the second shock landed
    inside the first one's window, inflated the first increment, and made every
    host look as if the second spike were smaller; the superposition identity
    above caught it, off by 0.5. The equal case is contaminated too, one level
    down: the second shock's jump acts at the end of its step, index j2 + 1,
    which is exactly where a window of `gap` steps after the first shock ends.
    """
    if window_steps >= gap_steps:
        raise ValueError(f"window ({window_steps} steps) must be shorter than the gap ({gap_steps}): "
                         f"otherwise the second shock's jump lands inside the first spike's window")
    j1 = lead_steps
    j2 = j1 + gap_steps
    n_steps = j2 + window_steps + 2
    T = n_steps * dt
    planted = ((j1 + 0.5) * dt, (j2 + 0.5) * dt)
    path = expected_path(host, hawkes, jumps, T, n_steps, planted=planted)
    r = shock_response(host, hawkes, jumps, T, n_steps, j1)
    # the jump of shock N acts at the end of its step: index jN + 1
    inc1 = path[j1 + 1 + window_steps] - path[j1]
    inc2 = path[j2 + 1 + window_steps] - path[j2]
    return {"inc1": float(inc1), "inc2": float(inc2),
            "level1": float(path[j1 + 1 + window_steps]), "level2": float(path[j2 + 1 + window_steps]),
            "r_d": float(r[j1 + gap_steps]), "r_dW": float(r[j1 + 1 + gap_steps + window_steps]),
            "planted": planted, "T": T, "n_steps": n_steps, "j1": j1, "j2": j2}


def excess_kurtosis_theory(v, hawkes, price_std, window):
    """
    Constant variance v, zero-mean Gaussian price jumps, no compensator: given
    N events the return is N(0, v w + N s^2), so the return is a scale mixture
    of normals and its excess kurtosis is exactly

        3 s^4 Var[N] / (v w + E[N] s^2)^2

    with Var[N] from hawkes.count_variance -- lambda w for Poisson, larger for
    any clustering at the same mean rate.
    """
    EN = hawkes.stationary_intensity * window
    VN = hk.count_variance(hawkes, window)
    s2 = price_std * price_std
    return 3.0 * s2 * s2 * VN / (v * window + EN * s2) ** 2


def excess_kurtosis(x, batches=40):
    """
    Sample excess kurtosis of the rows of x (paths x periods) pooled, with a
    batch standard error: paths are split into `batches` groups because returns
    WITHIN a path are dependent (clustering, stochastic variance) and paths are
    the independent unit.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    groups = np.array_split(np.arange(n), batches)

    def kurt(v):
        v = v.ravel()
        v = v - v.mean()
        m2 = np.mean(v * v)
        return float(np.mean(v ** 4) / (m2 * m2) - 3.0)

    ks = np.array([kurt(x[g]) for g in groups])
    return kurt(x), float(ks.std(ddof=1) / math.sqrt(batches))
