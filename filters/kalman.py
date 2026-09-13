"""
Kalman, adaptive Kalman and dual Kalman filtering, built once and applied to
three hosts.

The lecture's state space (the VIX filter) is the OU case:

    x_k = F x_{k-1} + B u + w_k,   F = e^{-kappa dt},  B = 1 - e^{-kappa dt},  u = theta,
    Q = (sigma^2 / 2 kappa)(1 - e^{-2 kappa dt}),       y_k = x_k + v_k,  v ~ N(0, R),
    K = P_{k|k-1} / (P_{k|k-1} + R)

Every host here is a linear-in-the-state model  x_k = A x_{k-1} + b + w_k,
y_k = H x_k + c + v_k,  with process noise Q that may depend on the state:

  OUModel         the lecture's filter, exactly linear-Gaussian: optimal.
  CIRModel        classical Heston variance. The CIR conditional MEAN is the OU
                  one, so A and b are identical; the conditional variance is
                  affine in the state, v xi^2 (e^{-k dt} - e^{-2k dt})/k +
                  theta xi^2 (1 - e^{-k dt})^2 / (2k), and is evaluated at the
                  filtered estimate -- the standard quasi-maximum-likelihood
                  Kalman filter for affine models.
  LiftedRoughModel  rough Heston. The state is the vector of N lifted factors;
                  with the lift's exponential-Euler step
                      U_k = diag(e^{-x dt}) U_{k-1} + g (kappa (theta - V) dt + xi sqrt(V) dB),
                      V = v0 + w'U,  g = phi_1(-x dt),
                  the drift is linear in U and the noise is RANK ONE,
                  Q = xi^2 V dt g g'. A rough variance is not Markov and cannot
                  be Kalman-filtered as a scalar; the lifted factors are Markov,
                  and are. That is the point of Session D's lift made concrete.

Filters (the lecture's comparison, plus one):

  strict      fixed gain dynamics: ignores bad prints, lags a regime change.
  adaptive    'dynamic K': a surprise |nu| > c sqrt(S) inflates the prior
              covariance, additively along the direction shocks enter the
              state, until the innovation variance matches what was seen -- so
              the gain jumps. Catches a regime change at once, and overreacts to
              a single bad print the same way (17x the normal error on the OU
              host, 40 seeds, against strict's 2.3x).
  robust      tells the two apart by PERSISTENCE: one large innovation is
              down-weighted as a bad print (R inflated for that step), a run of
              `persistence` large same-signed innovations is a regime change and
              inflates the prior like `adaptive`. On OU and Heston it re-converges
              in one step and ignores the print (1.7x); on rough Heston there is
              little regime to catch -- a shock relaxes like a power law.

Parameters, three ways:

  fit_mle        offline maximum likelihood through the filter (and the lecture's
                 AR(1) regression, which noisy quotes bias badly: kappa 61-65
                 for a true 5 at the test design).
  dual_kalman    Wan & Nelson's dual EKF: a parameter filter beside the state
                 filter, linearised through the recurrent derivative dx/dw.
                 joint_ekf, the augmented-state EKF, agrees with it to 0.14% --
                 they make the same omission.
  recursive_mle  the dual filter with Ljung's (1979) correction: full sensitivity
                 equations for x, P and the gain, Gauss-Newton steps on the
                 innovation likelihood. The omission above biases the dual EKF by
                 up to 2 SE; this removes it (total deviation from the MLE 2.5 SE
                 against 4.6 SE over eight host-seeds).

Numpy only, except the offline maximum-likelihood fit (scipy.optimize).
"""

import math
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------- models
class OUModel:
    """The lecture's scalar OU state space. params: kappa, theta, sigma, R."""

    param_names = ("kappa", "theta", "sigma", "R")
    positive = ("kappa", "sigma", "R")
    dim = 1

    def __init__(self, dt):
        self.dt = float(dt)

    def transition(self, p):
        e = math.exp(-p["kappa"] * self.dt)
        return np.array([[e]]), np.array([p["theta"] * (1.0 - e)])

    def process_cov(self, p, x):
        e2 = math.exp(-2.0 * p["kappa"] * self.dt)
        return np.array([[p["sigma"] ** 2 * (1.0 - e2) / (2.0 * p["kappa"])]])

    def observation(self, p):
        return np.array([1.0]), 0.0

    def initial(self, p):
        return np.array([p["theta"]]), np.array([[p["sigma"] ** 2 / (2.0 * p["kappa"])]])

    def measure(self, x):
        return x[..., 0]


class CIRModel:
    """Heston variance, CIR. params: kappa, theta, xi, R. Q depends on the filtered state."""

    param_names = ("kappa", "theta", "xi", "R")
    positive = ("kappa", "theta", "xi", "R")
    dim = 1

    def __init__(self, dt):
        self.dt = float(dt)

    def transition(self, p):
        e = math.exp(-p["kappa"] * self.dt)
        return np.array([[e]]), np.array([p["theta"] * (1.0 - e)])

    def process_cov(self, p, x):
        k, th, xi = p["kappa"], p["theta"], p["xi"]
        e = math.exp(-k * self.dt)
        v = max(float(x[0]), 0.0)
        q = v * xi * xi * (e - e * e) / k + th * xi * xi * (1.0 - e) ** 2 / (2.0 * k)
        return np.array([[max(q, 1e-18)]])

    def observation(self, p):
        return np.array([1.0]), 0.0

    def initial(self, p):
        return np.array([p["theta"]]), np.array([[p["theta"] * p["xi"] ** 2 / (2.0 * p["kappa"])]])

    def measure(self, x):
        return x[..., 0]


class LiftedRoughModel:
    """
    Lifted rough Heston variance, state U in R^N, observed V = v0 + w'U plus noise.
    params: kappa, theta, xi, R (H and v0 fixed at construction: they define the
    factors themselves, so a parameter filter cannot move them without changing
    what the state means).
    """

    param_names = ("kappa", "theta", "xi", "R")
    positive = ("kappa", "theta", "xi", "R")

    def __init__(self, dt, H=0.12, v0=0.04, N=None):
        import models.rough_heston as rh
        self.dt, self.H, self.v0 = float(dt), float(H), float(v0)
        self.N = rh.N_DEFAULT if N is None else int(N)
        self.w, self.x = rh.lift_nodes(self.H, self.N)
        self.E = np.exp(-self.x * self.dt)
        self.g = rh._phi1(-self.x * self.dt)
        self.dim = len(self.w)

    def transition(self, p):
        A = np.diag(self.E) - p["kappa"] * self.dt * np.outer(self.g, self.w)
        b = p["kappa"] * self.dt * (p["theta"] - self.v0) * self.g
        return A, b

    def process_cov(self, p, U):
        V = max(self.v0 + float(self.w @ U), 0.0)
        return (p["xi"] ** 2) * V * self.dt * np.outer(self.g, self.g)

    def observation(self, p):
        return self.w.copy(), self.v0

    def initial(self, p):
        A, b = self.transition(p)
        U = np.linalg.solve(np.eye(self.dim) - A, b)
        P = np.zeros((self.dim, self.dim))
        Q = self.process_cov(p, U)
        for _ in range(4000):              # discrete Lyapunov by iteration; A is stable
            Pn = A @ P @ A.T + Q
            if np.max(np.abs(Pn - P)) < 1e-14 * max(1.0, np.max(np.abs(Pn))):
                P = Pn
                break
            P = Pn
        return U, P

    def measure(self, U):
        return self.v0 + U @ self.w


# ----------------------------------------------------------------- simulation
def simulate_ou(kappa, theta, sigma, dt, n, x0=None, seed=0, jumps=None):
    """Exact OU transitions. `jumps` maps step index -> additive state jump."""
    rng = np.random.default_rng(seed)
    e = math.exp(-kappa * dt)
    sd = sigma * math.sqrt((1 - e * e) / (2 * kappa))
    x = np.empty(n)
    x[0] = theta if x0 is None else x0
    z = rng.standard_normal(n)
    for k in range(1, n):
        x[k] = theta + (x[k - 1] - theta) * e + sd * z[k]
        if jumps and k in jumps:
            x[k] += jumps[k]
    return x


def simulate_cir(kappa, theta, xi, dt, n, v0=None, seed=0, jumps=None, substeps=10):
    """Full-truncation Euler with substeps; `jumps` maps step -> additive variance jump."""
    rng = np.random.default_rng(seed)
    h = dt / substeps
    v = theta if v0 is None else v0
    out = np.empty(n)
    out[0] = v
    for k in range(1, n):
        for _ in range(substeps):
            vp = max(v, 0.0)
            v = v + kappa * (theta - vp) * h + xi * math.sqrt(vp * h) * rng.standard_normal()
        if jumps and k in jumps:
            v += jumps[k]
        out[k] = max(v, 0.0)
    return out


def simulate_rough(model, p, n, seed=0, jumps=None, substeps=4, theta_after=None):
    """
    The lifted variance path on the model's own grid (dt / substeps inner steps,
    same exponential-Euler scheme). `jumps` maps step -> a jump of that size in
    V, delivered through the driver direction g so that V moves by exactly that
    amount at once and then relaxes through the kernel. `theta_after` =
    (step, theta) moves the long-run level from that step on -- the only kind of
    regime change that PERSISTS on a rough host: a single shock relaxes like a
    power law and is largely gone by the next day.
    Returns (V, U) with U of shape (n, N).
    """
    import models.rough_heston as rh
    rng = np.random.default_rng(seed)
    h = model.dt / substeps
    E = np.exp(-model.x * h)
    g = rh._phi1(-model.x * h)
    U, _ = model.initial(p)
    U = U.copy()
    Us = np.empty((n, model.dim))
    Us[0] = U
    theta = p["theta"]
    for k in range(1, n):
        if theta_after is not None and k == theta_after[0]:
            theta = theta_after[1]
        for _ in range(substeps):
            V = max(model.v0 + float(model.w @ U), 0.0)
            drive = p["kappa"] * (theta - V) * h + p["xi"] * math.sqrt(V * h) * rng.standard_normal()
            U = E * U + g * drive
        if jumps and k in jumps:
            U = U + g * (jumps[k] / float(model.w @ g))
        Us[k] = U
    return model.v0 + Us @ model.w, Us


# ----------------------------------------------------------------- filters
def inflate(P_prior, H, R, nu, u):
    """
    Additive covariance matching along direction u (H u = 1): the smallest change
    that makes the innovation variance H P H' + R equal the innovation actually
    seen, P + delta u u', delta = nu^2 - R - H P H'. Multiplicative inflation
    (scaling P by nu^2 / H P H') was tried first and breaks on the rough model:
    when the filtered variance touches zero the process noise vanishes, H P H'
    falls to ~1e-20 and the factor reaches 1e17 -- the covariance overflows.
    """
    hph = float(H @ P_prior @ H)
    delta = nu * nu - R - hph
    if delta <= 0.0:
        return P_prior
    return P_prior + delta * np.outer(u, u)


@dataclass
class Strict:
    """No adaptation: the plain filter."""
    name: str = "strict"

    def reset(self):
        pass

    def adjust(self, nu, S, P_prior, H, R, u):
        return P_prior, R


@dataclass
class Adaptive:
    """
    The lecture's dynamic gain. When the normalised innovation exceeds `threshold`,
    the prior covariance is inflated (along the direction shocks enter the state)
    until the innovation's own variance matches what was seen -- and the gain
    jumps. Catches a regime change at once; overreacts to a bad print the same way.
    """
    threshold: float = 3.0
    name: str = "adaptive"

    def reset(self):
        pass

    def adjust(self, nu, S, P_prior, H, R, u):
        if nu * nu / S > self.threshold ** 2:
            return inflate(P_prior, H, R, nu, u), R
        return P_prior, R


@dataclass
class Robust:
    """
    Outliers by magnitude, regime changes by persistence. A normalised innovation
    beyond `threshold` is treated as a bad print (R inflated so the update moves
    by at most `threshold` standard deviations of the innovation), unless the last
    `persistence` innovations were all beyond it with the same sign -- then it is
    a regime change and the prior is inflated as in Adaptive.
    """
    threshold: float = 3.0
    persistence: int = 2
    name: str = "robust"
    history: list = field(default_factory=list)

    def reset(self):
        self.history = []

    def adjust(self, nu, S, P_prior, H, R, u):
        eps = nu / math.sqrt(S)
        big = abs(eps) > self.threshold
        self.history.append(math.copysign(1.0, eps) if big else 0.0)
        self.history = self.history[-self.persistence:]
        if not big:
            return P_prior, R
        if len(self.history) == self.persistence and abs(sum(self.history)) == self.persistence:
            return inflate(P_prior, H, R, nu, u), R
        # bad print: Huber-style, inflate R so the standardised innovation is `threshold`
        hph = float(H @ P_prior @ H)
        R_eff = max(nu * nu / self.threshold ** 2 - hph, R)
        return P_prior, R_eff


def _shock_direction(model, p, x, H):
    """
    The direction in state space along which a surprise is attributed: the
    process-noise direction Q H / (H Q H), normalised so H u = 1. For the scalar
    hosts that is 1; for the lifted rough model Q is proportional to g g', so u is
    g / (w'g) -- a shock enters every factor through the kernel's cell weights,
    exactly as the model says real shocks do. Falls back to H / (H'H).
    """
    Q = model.process_cov(p, x)
    qh = Q @ H
    hqh = float(H @ qh)
    if hqh > 1e-300:
        return qh / hqh
    return H / float(H @ H)


def kalman_filter(model, p, y, policy=None, x0=None, P0=None):
    """
    Run the filter over observations y (1-D). Returns a dict of arrays:
    x_prior, x_post (n, dim), P_post (n, dim, dim), innov, S, gain (n, dim),
    and loglik (the prediction-error decomposition, constant included).

    Joseph-form covariance update, (I - K H) P (I - K H)' + K R K', so P stays
    symmetric positive semi-definite for the 24-factor rough model too.
    """
    policy = Strict() if policy is None else policy
    policy.reset()
    y = np.asarray(y, dtype=float)
    n, d = len(y), model.dim
    if d == 1 and isinstance(policy, Strict) and x0 is None and P0 is None:
        return _scalar_filter(model, p, y)
    A, b = model.transition(p)
    H, c = model.observation(p)
    R = p["R"]
    x, P = model.initial(p)
    if x0 is not None:
        x = np.asarray(x0, dtype=float).copy()
    if P0 is not None:
        P = np.asarray(P0, dtype=float).copy()
    I = np.eye(d)
    out_prior = np.empty((n, d))
    out_post = np.empty((n, d))
    out_P = np.empty((n, d, d))
    innov = np.empty(n)
    S_arr = np.empty(n)
    gain = np.empty((n, d))
    ll = 0.0
    for k in range(n):
        if k > 0:
            Q = model.process_cov(p, x)
            x = A @ x + b
            P = A @ P @ A.T + Q
        out_prior[k] = x
        nu = y[k] - (float(H @ x) + c)
        S = float(H @ P @ H) + R
        P_adj, R_eff = policy.adjust(nu, S, P, H, R, _shock_direction(model, p, x, H))
        S_eff = float(H @ P_adj @ H) + R_eff
        K = (P_adj @ H) / S_eff
        x = x + K * nu
        IKH = I - np.outer(K, H)
        P = IKH @ P_adj @ IKH.T + R_eff * np.outer(K, K)
        ll += -0.5 * (math.log(2.0 * math.pi * S) + nu * nu / S)
        out_post[k] = x
        out_P[k] = P
        innov[k] = nu
        S_arr[k] = S
        gain[k] = K
    return {"x_prior": out_prior, "x_post": out_post, "P_post": out_P, "innov": innov,
            "S": S_arr, "gain": gain, "loglik": ll,
            "estimate": model.measure(out_post)}


def _scalar_filter(model, p, y):
    """
    The same recursion on Python floats, for one-dimensional states without an
    adaptation policy. Numpy's per-call overhead made the 1x1 matrix version
    ~20 microseconds a step -- a maximum-likelihood fit of a 5000-step series
    took 49 s; this path gives identical numbers (checked in test_filters.py).
    """
    A, b = model.transition(p)
    H, c = model.observation(p)
    a, bb, h, R = float(A[0, 0]), float(b[0]), float(H[0]), float(p["R"])
    x0, P0 = model.initial(p)
    x, P = float(x0[0]), float(P0[0, 0])
    state_q = isinstance(model, CIRModel)
    q_const = None if state_q else float(model.process_cov(p, x0)[0, 0])
    n = len(y)
    prior = np.empty(n)
    post = np.empty(n)
    Ps = np.empty(n)
    innov = np.empty(n)
    S_arr = np.empty(n)
    gain = np.empty(n)
    ll = 0.0
    log2pi = math.log(2.0 * math.pi)
    if state_q:
        k_, th, xi = p["kappa"], p["theta"], p["xi"]
        e = math.exp(-k_ * model.dt)
        c1 = xi * xi * (e - e * e) / k_
        c0 = th * xi * xi * (1.0 - e) ** 2 / (2.0 * k_)
    yl = y.tolist()
    for k in range(n):
        if k > 0:
            q = (max(x, 0.0) * c1 + c0) if state_q else q_const
            x = a * x + bb
            P = a * a * P + max(q, 1e-18)
        prior[k] = x
        nu = yl[k] - (h * x + c)
        S = h * h * P + R
        K = P * h / S
        x += K * nu
        P = (1.0 - K * h) ** 2 * P + R * K * K
        ll -= 0.5 * (log2pi + math.log(S) + nu * nu / S)
        post[k] = x
        Ps[k] = P
        innov[k] = nu
        S_arr[k] = S
        gain[k] = K
    return {"x_prior": prior[:, None], "x_post": post[:, None], "P_post": Ps[:, None, None],
            "innov": innov, "S": S_arr, "gain": gain[:, None], "loglik": ll, "estimate": post}


def steady_state_ou(p, dt):
    """
    Scalar steady state of the lecture's filter: with a = e^{-kappa dt}, q = Q,
    the prior variance M solves M = a^2 M R / (M + R) + q. Returns (M, K, P).
    """
    a2 = math.exp(-2.0 * p["kappa"] * dt)
    q = p["sigma"] ** 2 * (1.0 - a2) / (2.0 * p["kappa"])
    R = p["R"]
    # M^2 + M (R - a2 R - q) - q R = 0
    bq = R - a2 * R - q
    M = 0.5 * (-bq + math.sqrt(bq * bq + 4.0 * q * R))
    K = M / (M + R)
    return M, K, (1.0 - K) * M


# ----------------------------------------------------------------- calibration
def ar1_calibration(y, dt):
    """
    The lecture's offline step: regress y_k on y_{k-1}, then kappa = -ln(a)/dt,
    theta = b/(1 - a), sigma^2 = s^2 2 kappa / (1 - a^2). Correct for a noiseless
    OU. With observation noise the regressor is measured with error and a is
    attenuated towards zero, so kappa is biased UP -- measured, not assumed.
    """
    y = np.asarray(y, dtype=float)
    X, Y = y[:-1], y[1:]
    a, b = np.polyfit(X, Y, 1)
    resid = Y - (a * X + b)
    s2 = float(np.var(resid, ddof=2))
    a = float(min(max(a, 1e-9), 1 - 1e-12))
    kappa = -math.log(a) / dt
    return {"kappa": kappa, "theta": b / (1.0 - a),
            "sigma": math.sqrt(s2 * 2.0 * kappa / (1.0 - a * a)), "a": a}


def fit_mle(model, y, start, names=None, fixed=None):
    """
    Maximum-likelihood parameters from the filter's own likelihood, with SEs from
    the numerical Hessian. Positive parameters optimised in log space.
    """
    from scipy.optimize import minimize

    names = tuple(model.param_names if names is None else names)
    fixed = dict(fixed or {})

    def to_z(p):
        return np.array([math.log(p[k]) if k in model.positive else p[k] for k in names])

    def from_z(z):
        p = dict(start)
        p.update(fixed)
        for i, k in enumerate(names):
            p[k] = math.exp(z[i]) if k in model.positive else float(z[i])
        return p

    def nll(z):
        try:
            v = -kalman_filter(model, from_z(z), y)["loglik"]
            return v if np.isfinite(v) else 1e300
        except (ValueError, OverflowError, np.linalg.LinAlgError):
            return 1e300

    r = minimize(nll, to_z(start), method="Nelder-Mead",
                 options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 4000, "maxfev": 8000})
    p = from_z(r.x)
    x0 = np.array([p[k] for k in names])
    h = np.maximum(1e-4 * np.abs(x0), 1e-7)

    def f(xv):
        q = dict(p)
        for i, k in enumerate(names):
            q[k] = float(xv[i])
        if any(q[k] <= 0 for k in names if k in model.positive):
            return np.inf
        return -kalman_filter(model, q, y)["loglik"]

    m = len(names)
    Hm = np.empty((m, m))
    for i in range(m):
        for j in range(i, m):
            ei = np.zeros(m)
            ej = np.zeros(m)
            ei[i] = h[i]
            ej[j] = h[j]
            Hm[i, j] = Hm[j, i] = (f(x0 + ei + ej) - f(x0 + ei - ej) - f(x0 - ei + ej)
                                   + f(x0 - ei - ej)) / (4.0 * h[i] * h[j])
    try:
        cov = np.linalg.inv(Hm)
        se = {k: (math.sqrt(cov[i, i]) if cov[i, i] > 0 else float("nan")) for i, k in enumerate(names)}
    except np.linalg.LinAlgError:
        se = {k: float("nan") for k in names}
    return {"params": p, "se": se, "loglik": -float(r.fun), "converged": bool(r.success)}


# ----------------------------------------------------------------- dual & joint
def _transform(model, name, value):
    return math.log(value) if name in model.positive else value


def _untransform(model, name, z):
    return math.exp(z) if name in model.positive else z


def dual_kalman(model, p_start, y, learn=("kappa",), q_param=1e-6, p0_param=1.0, fd=1e-6):
    """
    Wan & Nelson dual extended Kalman filter.

    Parameters in `learn` (log-transformed when positive) follow a random walk
    with variance `q_param` per step. Each step:

      1. parameter time update         w- = w,  Pw- = Pw + Qw
      2. state time update with w-     x- = A(w) x + b(w),  P- = A P A' + Q
      3. recurrent derivative          dx-/dw = dA/dw x + A dx/dw + db/dw
      4. state measurement update      x = x- + K nu,  P = (I - K H) P- ...
      5. parameter measurement update  C = H dx-/dw,  Kw = Pw- C' / (C Pw- C' + S)
      6. derivative after the update   dx/dw = (I - K H) dx-/dw

    dA/dw and db/dw by central differences on the model, so the same code serves
    all three hosts. Returns the state estimate and the parameter path.
    """
    y = np.asarray(y, dtype=float)
    n, d = len(y), model.dim
    p = dict(p_start)
    names = tuple(learn)
    m = len(names)
    w = np.array([_transform(model, k, p[k]) for k in names])
    Pw = np.eye(m) * p0_param
    Qw = np.eye(m) * q_param
    x, P = model.initial(p)
    dx = np.zeros((d, m))
    H, c = model.observation(p)
    I = np.eye(d)
    est = np.empty(n)
    path = np.empty((n, m))

    def params_at(wv):
        q = dict(p)
        for i, k in enumerate(names):
            q[k] = _untransform(model, k, wv[i])
        return q

    for k in range(n):
        pw = params_at(w)
        Pw_prior = Pw + Qw
        if k > 0:
            A, b = model.transition(pw)
            Q = model.process_cov(pw, x)
            dA = np.empty((m, d, d))
            db = np.empty((m, d))
            for i in range(m):
                wp = w.copy()
                wm = w.copy()
                wp[i] += fd
                wm[i] -= fd
                Ap, bp = model.transition(params_at(wp))
                Am, bm = model.transition(params_at(wm))
                dA[i] = (Ap - Am) / (2 * fd)
                db[i] = (bp - bm) / (2 * fd)
            dx_prior = np.stack([dA[i] @ x + db[i] for i in range(m)], axis=1) + A @ dx
            x = A @ x + b
            P = A @ P @ A.T + Q
        else:
            dx_prior = dx
        nu = y[k] - (float(H @ x) + c)
        R = pw["R"]
        S = float(H @ P @ H) + R
        K = (P @ H) / S
        x = x + K * nu
        IKH = I - np.outer(K, H)
        P = IKH @ P @ IKH.T + R * np.outer(K, K)
        C = H @ dx_prior                                   # (m,)
        Sw = float(C @ Pw_prior @ C) + S
        Kw = (Pw_prior @ C) / Sw
        w = w + Kw * nu
        Pw = Pw_prior - np.outer(Kw, C @ Pw_prior)
        Pw = 0.5 * (Pw + Pw.T)
        dx = IKH @ dx_prior
        est[k] = float(model.measure(x))
        path[k] = [_untransform(model, names[i], w[i]) for i in range(m)]
    return {"estimate": est, "params": path, "final": dict(zip(names, path[-1])), "Pw": Pw}


def recursive_mle(model, p_start, y, learn=("kappa",), p0_param=1.0, fd=1e-6, max_step=0.25,
                  warmup=20):
    """
    The dual filter with Ljung's correction: a recursive prediction-error
    (recursive maximum-likelihood) parameter filter.

    Wan-Nelson's dual EKF and the joint EKF both drop the dependence of the
    Kalman GAIN and COVARIANCE on the parameters, and Ljung (1979) showed that
    this makes the parameter estimates converge to the wrong point in general.
    Measured here: both land 0.6-2 SE below the full-sample MLE, whatever the
    parameter noise q (1e-6 to 0 changes nothing), and agree with each other to
    0.14% -- they share the same omission.

    This carries the full sensitivity equations along with the filter, for each
    learned parameter w_i:

        dx-_i = dA_i x + A dx_i + db_i
        dP-_i = dA_i P A' + A dP_i A' + A P dA_i' + dQ_i        (Riccati sensitivity)
        dnu_i = -H dx-_i,       dS_i = H dP-_i H'
        dK_i  = dP-_i H / S - P- H dS_i / S^2
        dx_i  = dx-_i + dK_i nu + K dnu_i
        dP_i  = dP-_i - dK_i H P- - K H dP-_i

    so the score of each innovation's log-density is exact, and the parameters
    take Gauss-Newton steps with the accumulated Fisher information
    (dnu dnu' / S + 1/2 dS dS' / S^2). For a correctly specified Gaussian model
    this converges to the maximum-likelihood estimate. dQ_i includes Q's
    dependence on the state through dx_i, by a directional difference.
    """
    y = np.asarray(y, dtype=float)
    n, d = len(y), model.dim
    p = dict(p_start)
    names = tuple(learn)
    m = len(names)
    w = np.array([_transform(model, k, p[k]) for k in names])
    info = np.eye(m) / p0_param
    x, P = model.initial(p)
    dx = np.zeros((m, d))
    dP = np.zeros((m, d, d))
    H, c = model.observation(p)
    I = np.eye(d)
    est = np.empty(n)
    path = np.empty((n, m))

    def params_at(wv):
        q = dict(p)
        for i, k in enumerate(names):
            q[k] = _untransform(model, k, wv[i])
        return q

    for k in range(n):
        pw = params_at(w)
        R = pw["R"]
        if k > 0:
            A, b = model.transition(pw)
            Q = model.process_cov(pw, x)
            dxp = np.empty((m, d))
            dPp = np.empty((m, d, d))
            for i in range(m):
                wp, wm = w.copy(), w.copy()
                wp[i] += fd
                wm[i] -= fd
                Ap, bp = model.transition(params_at(wp))
                Am, bm = model.transition(params_at(wm))
                dA = (Ap - Am) / (2 * fd)
                db = (bp - bm) / (2 * fd)
                dQ = (model.process_cov(params_at(wp), x + fd * dx[i])
                      - model.process_cov(params_at(wm), x - fd * dx[i])) / (2 * fd)
                dxp[i] = dA @ x + A @ dx[i] + db
                dPp[i] = dA @ P @ A.T + A @ dP[i] @ A.T + A @ P @ dA.T + dQ
            x = A @ x + b
            P = A @ P @ A.T + Q
        else:
            dxp, dPp = dx, dP
        nu = y[k] - (float(H @ x) + c)
        S = float(H @ P @ H) + R
        K = (P @ H) / S
        dnu = -(dxp @ H)                                    # (m,)
        dS = np.array([float(H @ dPp[i] @ H) for i in range(m)])
        if "R" in names:
            dS[names.index("R")] += R                       # d R / d log R
        dK = np.stack([(dPp[i] @ H) / S - (P @ H) * dS[i] / (S * S) for i in range(m)])
        # score of log N(nu; 0, S) and its Fisher information
        score = -0.5 * (dS / S + 2.0 * nu * dnu / S - nu * nu * dS / (S * S))
        if k >= warmup:
            info = info + np.outer(dnu, dnu) / S + 0.5 * np.outer(dS, dS) / (S * S)
            step = np.linalg.solve(info, score)
            w = w + np.clip(step, -max_step, max_step)
        # state update and its sensitivities (with the pre-update parameters)
        x_new = x + K * nu
        IKH = I - np.outer(K, H)
        P_new = IKH @ P @ IKH.T + R * np.outer(K, K)
        for i in range(m):
            dx[i] = dxp[i] + dK[i] * nu + K * dnu[i]
            dP[i] = dPp[i] - np.outer(dK[i], H @ P) - np.outer(K, H @ dPp[i])
            dP[i] = 0.5 * (dP[i] + dP[i].T)
        x, P = x_new, P_new
        est[k] = float(model.measure(x))
        path[k] = [_untransform(model, names[i], w[i]) for i in range(m)]
    cov_w = np.linalg.inv(info)
    return {"estimate": est, "params": path, "final": dict(zip(names, path[-1])), "info": info,
            "se_log": np.sqrt(np.diag(cov_w))}


def joint_ekf(model, p_start, y, learn=("kappa",), q_param=1e-6, p0_param=1.0, fd=1e-6):
    """
    Augmented-state EKF: z = [x, w], f(z) = [A(w) x + b(w), w], Jacobian by central
    differences. The independent route the dual filter is checked against.
    """
    y = np.asarray(y, dtype=float)
    n, d = len(y), model.dim
    p = dict(p_start)
    names = tuple(learn)
    m = len(names)
    x, P = model.initial(p)
    w = np.array([_transform(model, k, p[k]) for k in names])
    z = np.concatenate([x, w])
    Pz = np.zeros((d + m, d + m))
    Pz[:d, :d] = P
    Pz[d:, d:] = np.eye(m) * p0_param
    H, c = model.observation(p)
    Hz = np.concatenate([H, np.zeros(m)])
    est = np.empty(n)
    path = np.empty((n, m))

    def params_at(wv):
        q = dict(p)
        for i, k in enumerate(names):
            q[k] = _untransform(model, k, wv[i])
        return q

    def f(zv):
        A, b = model.transition(params_at(zv[d:]))
        return np.concatenate([A @ zv[:d] + b, zv[d:]])

    for k in range(n):
        if k > 0:
            F = np.eye(d + m)
            A, _ = model.transition(params_at(z[d:]))
            F[:d, :d] = A
            for i in range(m):
                zp = z.copy()
                zm = z.copy()
                zp[d + i] += fd
                zm[d + i] -= fd
                F[:d, d + i] = (f(zp)[:d] - f(zm)[:d]) / (2 * fd)
            Qz = np.zeros((d + m, d + m))
            Qz[:d, :d] = model.process_cov(params_at(z[d:]), z[:d])
            Qz[d:, d:] = np.eye(m) * q_param
            z = f(z)
            Pz = F @ Pz @ F.T + Qz
        R = params_at(z[d:])["R"]
        nu = y[k] - (float(Hz @ z) + c)
        S = float(Hz @ Pz @ Hz) + R
        K = (Pz @ Hz) / S
        z = z + K * nu
        IKH = np.eye(d + m) - np.outer(K, Hz)
        Pz = IKH @ Pz @ IKH.T + R * np.outer(K, K)
        est[k] = float(model.measure(z[:d]))
        path[k] = [_untransform(model, names[i], z[d + i]) for i in range(m)]
    return {"estimate": est, "params": path, "final": dict(zip(names, path[-1]))}


# ----------------------------------------------------------------- diagnostics
def reconvergence_steps(estimate, truth, start, band, hold=5, limit=None):
    """Steps after `start` until |estimate - truth| < band for `hold` consecutive steps."""
    err = np.abs(np.asarray(estimate) - np.asarray(truth))
    limit = len(err) if limit is None else min(len(err), start + limit)
    run = 0
    for k in range(start, limit):
        run = run + 1 if err[k] < band else 0
        if run >= hold:
            return k - hold + 1 - start
    return None


def excursion(estimate, truth, start, length=10):
    """Largest |estimate - truth| over the `length` steps from `start`."""
    err = np.abs(np.asarray(estimate) - np.asarray(truth))
    return float(err[start:start + length].max())
