"""
EGARCH with Student-t innovations, its score-driven cousin, and GARCH as the benchmark
(Session L, 25 September 2026: a returns-only volatility model to keep in the codebase for
when realised volatility arrives).

Three families, one interface (Spec, fit, loglik, variance_path, forecast, simulate, scan):

  "nelson"  EGARCH(p, q) of Nelson (1991):
                log s2_t = omega + sum_j beta_j log s2_{t-j}
                           + sum_i [ alpha_i (|z_{t-i}| - E|z|) + gamma_i z_{t-i} ],
            z_t = r_t / s_t, normal or standardised Student-t (unit variance, nu > 2).
            gamma < 0 is the leverage effect: a fall raises volatility more than a rise.
  "beta-t"  Beta-t-EGARCH(1,1) of Harvey & Chakravarty (2008) and Harvey (2013): the log
            scale lambda_t (r_t = e^{lambda_t} eps_t, eps_t ~ t_nu) moves with the SCORE of
            the Student-t,
                lambda_{t+1} = omega (1 - phi) + phi lambda_t + kappa u_t
                               + kappa_s sgn(-r_t) (u_t + 1),
                u_t = (nu + 1) r_t^2 / (nu e^{2 lambda_t} + r_t^2) - 1,   in [-1, nu].
            Nelson's update is linear in |z|, so one extreme return moves log-variance
            without bound; here it moves it by at most kappa nu + |kappa_s| (nu + 1). That
            bounded response is why it is the backup to keep for noisy, jumpy data.
  "garch"   GARCH(p, q) of Bollerslev (1986), s2_t = omega + sum_i alpha_i r_{t-i}^2
            + sum_j beta_j s2_{t-j}: the benchmark every volatility model is scored against.

Estimation is Gaussian or Student-t maximum likelihood on the natural recursion, through
maps that make every admissible parameter reachable and nothing else: stationarity of
the log-variance autoregression through its partial autocorrelations (Barndorff-Nielsen &
Schou 1973, Monahan 1984), GARCH's positivity and persistence < 1 through a softmax, and
nu > NU_MIN = 2.05 through NU_MIN + e^x. Standard errors come from the numerical Hessian in the natural
parameters. Returns are in whatever unit they are given (daily log returns throughout
this project); variances are in that unit squared.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize

FAMILIES = ("nelson", "beta-t", "garch")
DISTS = ("normal", "t")
LOG_S2_CLAMP = 60.0            # log-variance kept in [-60, 60] while the optimiser explores
NU_MIN, NU_MAX = 2.05, 500.0   # nu -> 2 makes the variance, and every variance forecast, infinite


@dataclass(frozen=True)
class Spec:
    family: str = "nelson"
    p: int = 1                 # lags of the (log-)variance
    q: int = 1                 # lags of the shock
    dist: str = "t"

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        if self.dist not in DISTS:
            raise ValueError(f"dist must be one of {DISTS}")
        if self.family == "beta-t" and (self.p, self.q, self.dist) != (1, 1, "t"):
            raise ValueError("beta-t is the (1, 1) Student-t model")
        if self.p < 1 or self.q < 1:
            raise ValueError("p and q must be >= 1")

    @property
    def name(self):
        if self.family == "beta-t":
            return "Beta-t-EGARCH(1,1)"
        base = "EGARCH" if self.family == "nelson" else "GARCH"
        return f"{base}({self.p},{self.q})-{self.dist}"

    @property
    def param_names(self):
        if self.family == "beta-t":
            return ("omega", "phi", "kappa", "kappa_s", "nu")
        a = tuple(f"alpha{i + 1}" for i in range(self.q))
        b = tuple(f"beta{j + 1}" for j in range(self.p))
        g = tuple(f"gamma{i + 1}" for i in range(self.q)) if self.family == "nelson" else ()
        return ("omega",) + a + g + b + (("nu",) if self.dist == "t" else ())

    @property
    def k(self):
        return len(self.param_names)


# ------------------------------------------------------------------ distributions
def abs_moment(dist, nu=None):
    """E|z| for the unit-variance innovation."""
    if dist == "normal":
        return math.sqrt(2.0 / math.pi)
    return math.sqrt(nu - 2.0) * math.exp(math.lgamma((nu - 1) / 2) - math.lgamma(nu / 2)) / math.sqrt(math.pi)


def log_density(z, dist, nu=None):
    """log f(z) of the unit-variance innovation, vectorised."""
    z = np.asarray(z, dtype=float)
    if dist == "normal":
        return -0.5 * math.log(2 * math.pi) - 0.5 * z * z
    c = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
    return c - 0.5 * (nu + 1) * np.log1p(z * z / (nu - 2))


def draw(dist, nu, size, rng):
    """Unit-variance innovations."""
    if dist == "normal":
        return rng.standard_normal(size)
    return rng.standard_t(nu, size) * math.sqrt((nu - 2) / nu)


# ------------------------------------------------------------------ parameter maps
def _pacf_to_ar(pac):
    """Durbin-Levinson: partial autocorrelations in (-1, 1)^p -> a stationary AR(p)."""
    phi = np.zeros(0)
    for k, a in enumerate(pac):
        phi = np.concatenate([phi - a * phi[::-1], [a]]) if k else np.array([a])
    return phi


def _ar_to_pacf(phi):
    """The inverse map (step-down recursion); raises if phi is not stationary."""
    phi = np.array(phi, dtype=float)
    pac = np.zeros(len(phi))
    for k in range(len(phi) - 1, -1, -1):
        a = phi[k]
        if abs(a) >= 1:
            raise ValueError("not stationary")
        pac[k] = a
        if k:
            phi = (phi[:k] + a * phi[:k][::-1]) / (1 - a * a)
    return pac


def to_natural(x, spec):
    """Unconstrained optimiser coordinates -> natural parameters (a vector in param_names order)."""
    x = np.asarray(x, dtype=float)
    nu = [NU_MIN + min(math.exp(min(x[-1], 50.0)), NU_MAX - NU_MIN)] if spec.dist == "t" else []
    if spec.family == "beta-t":
        return np.array([x[0], math.tanh(x[1]), x[2], x[3]] + nu)
    p, q = spec.p, spec.q
    if spec.family == "nelson":
        beta = _pacf_to_ar(np.tanh(x[1 + 2 * q:1 + 2 * q + p]))
        return np.concatenate([[x[0]], x[1:1 + 2 * q], beta, nu])
    # garch: omega > 0; (alpha, beta, slack) on the simplex, so persistence < 1
    e = np.exp(np.concatenate([x[1:1 + q + p], [0.0]]) - max(0.0, float(np.max(x[1:1 + q + p]))))
    s = e / e.sum()
    return np.concatenate([[math.exp(x[0])], s[:q + p], nu])


def to_unconstrained(theta, spec):
    theta = np.asarray(theta, dtype=float)
    nu = [math.log(theta[-1] - NU_MIN)] if spec.dist == "t" else []
    if spec.family == "beta-t":
        return np.array([theta[0], math.atanh(theta[1]), theta[2], theta[3]] + nu)
    p, q = spec.p, spec.q
    if spec.family == "nelson":
        pac = _ar_to_pacf(theta[1 + 2 * q:1 + 2 * q + p])
        return np.concatenate([[theta[0]], theta[1:1 + 2 * q], np.arctanh(pac), nu])
    ab = theta[1:1 + q + p]
    slack = 1.0 - float(ab.sum())
    if slack <= 0 or np.any(ab <= 0):
        raise ValueError("GARCH needs alpha, beta > 0 and persistence < 1")
    return np.concatenate([[math.log(theta[0])], np.log(ab / slack), nu])


def persistence(theta, spec):
    theta = np.asarray(theta, dtype=float)
    if spec.family == "beta-t":
        return float(theta[1])
    p, q = spec.p, spec.q
    if spec.family == "nelson":
        return float(np.sum(theta[1 + 2 * q:1 + 2 * q + p]))
    return float(np.sum(theta[1:1 + q + p]))


# ------------------------------------------------------------------ filters
def _filter(theta, r, spec, need_path=False):
    """
    (loglik contributions, conditional variances s2_t for t = 0..T, the last being the
    one-step forecast). Every variance is the one known at the START of its day.
    """
    theta = [float(v) for v in theta]
    r = np.asarray(r, dtype=float)
    T = len(r)
    rl = r.tolist()
    s2_path = np.empty(T + 1) if need_path else None
    ll = np.empty(T)
    v0 = float(np.var(r)) if T > 1 else 1e-4
    if spec.family == "beta-t":
        omega, phi, kappa, ks, nu = theta
        c = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(math.pi * nu)
        lam = omega
        vr = nu / (nu - 2)
        for t in range(T):
            e2l = math.exp(2 * lam)
            y2 = rl[t] * rl[t]
            ll[t] = c - lam - 0.5 * (nu + 1) * math.log1p(y2 / (nu * e2l))
            if need_path:
                s2_path[t] = e2l * vr
            u = (nu + 1) * y2 / (nu * e2l + y2) - 1.0
            sg = -1.0 if rl[t] > 0 else (1.0 if rl[t] < 0 else 0.0)
            lam = omega * (1 - phi) + phi * lam + kappa * u + ks * sg * (u + 1.0)
            lam = max(min(lam, LOG_S2_CLAMP / 2), -LOG_S2_CLAMP / 2)
        if need_path:
            s2_path[T] = math.exp(2 * lam) * vr
        return ll, s2_path

    p, q = spec.p, spec.q
    nu = theta[-1] if spec.dist == "t" else None
    if spec.dist == "t":
        cz = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
    omega = theta[0]
    alpha = theta[1:1 + q]
    if spec.family == "nelson":
        gamma = theta[1 + q:1 + 2 * q]
        beta = theta[1 + 2 * q:1 + 2 * q + p]
        ez = abs_moment(spec.dist, nu)
        hist_l = [math.log(v0)] * p           # log s2, most recent first
        hist_z = [0.0] * q                    # pre-sample shocks at their mean: no contribution
        hist_a = [ez] * q                     # |z| at its mean
        for t in range(T + 1):
            ls = omega
            for j in range(p):
                ls += beta[j] * hist_l[j]
            for i in range(q):
                ls += alpha[i] * (hist_a[i] - ez) + gamma[i] * hist_z[i]
            ls = max(min(ls, LOG_S2_CLAMP), -LOG_S2_CLAMP)
            if need_path:
                s2_path[t] = math.exp(ls)
            if t == T:
                break
            z = rl[t] * math.exp(-0.5 * ls)
            if spec.dist == "t":
                ll[t] = cz - 0.5 * (nu + 1) * math.log1p(z * z / (nu - 2)) - 0.5 * ls
            else:
                ll[t] = -0.9189385332046727 - 0.5 * z * z - 0.5 * ls
            hist_l = [ls] + hist_l[:-1]
            hist_z = [z] + hist_z[:-1]
            hist_a = [abs(z)] + hist_a[:-1]
        return ll, s2_path

    beta = theta[1 + q:1 + q + p]
    hist_s = [v0] * p
    hist_r2 = [v0] * q
    for t in range(T + 1):
        s2 = omega
        for i in range(q):
            s2 += alpha[i] * hist_r2[i]
        for j in range(p):
            s2 += beta[j] * hist_s[j]
        s2 = max(s2, 1e-300)
        if need_path:
            s2_path[t] = s2
        if t == T:
            break
        y2 = rl[t] * rl[t]
        if spec.dist == "t":
            ll[t] = cz - 0.5 * (nu + 1) * math.log1p(y2 / (s2 * (nu - 2))) - 0.5 * math.log(s2)
        else:
            ll[t] = -0.9189385332046727 - 0.5 * y2 / s2 - 0.5 * math.log(s2)
        hist_s = [s2] + hist_s[:-1]
        hist_r2 = [y2] + hist_r2[:-1]
    return ll, s2_path


def loglik(theta, r, spec):
    ll, _ = _filter(theta, r, spec)
    return float(np.sum(ll))


def variance_path(theta, r, spec):
    """Conditional variances s2_0..s2_T: s2_t uses returns before t; s2_T is the forecast."""
    return _filter(theta, r, spec, need_path=True)[1]


# ------------------------------------------------------------------ estimation
def default_start(r, spec):
    v = float(np.var(r))
    nu = [8.0] if spec.dist == "t" else []
    if spec.family == "beta-t":
        return np.array([0.5 * math.log(v * 6 / 8), 0.97, 0.04, 0.02, 8.0])
    p, q = spec.p, spec.q
    if spec.family == "nelson":
        beta = [0.97] if p == 1 else list(_pacf_to_ar([0.97] + [0.1] * (p - 1)))
        per = float(np.sum(beta))
        return np.array([math.log(v) * (1 - per)] + [0.12 / q] * q + [-0.06 / q] * q + beta + nu)
    alpha = [0.06 / q] * q
    beta = [0.9 / p] * p
    return np.array([v * 0.04] + alpha + beta + nu)


def _hessian(f, x, rel=1e-4):
    k = len(x)
    h = rel * np.maximum(np.abs(x), 1e-2)
    H = np.empty((k, k))
    f0 = f(x)
    for i in range(k):
        for j in range(i, k):
            if i == j:
                xp, xm = x.copy(), x.copy()
                xp[i] += h[i]
                xm[i] -= h[i]
                H[i, i] = (f(xp) - 2 * f0 + f(xm)) / (h[i] * h[i])
            else:
                xpp, xpm, xmp, xmm = x.copy(), x.copy(), x.copy(), x.copy()
                xpp[i] += h[i]; xpp[j] += h[j]
                xpm[i] += h[i]; xpm[j] -= h[j]
                xmp[i] -= h[i]; xmp[j] += h[j]
                xmm[i] -= h[i]; xmm[j] -= h[j]
                H[i, j] = H[j, i] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h[i] * h[j])
    return H


def fit(r, spec, starts=None, n_random=2, seed=0, se=True, maxiter=400):
    """
    Maximum likelihood. Returns a dict: theta (natural, param_names order), se, loglik,
    aic, bic, persistence, half_life (in observations), nu, n_obs, converged, spec.
    """
    r = np.asarray(r, dtype=float)
    rng = np.random.default_rng(seed)
    x_starts = [to_unconstrained(s, spec) for s in (starts or [default_start(r, spec)])]
    for _ in range(n_random):
        x_starts.append(x_starts[0] + 0.3 * rng.standard_normal(len(x_starts[0])))

    def nll(x):
        try:
            v = -loglik(to_natural(x, spec), r, spec)
        except (OverflowError, ValueError, ZeroDivisionError):
            return 1e300
        return v if math.isfinite(v) else 1e300

    best = None
    for x0 in x_starts:
        res = optimize.minimize(nll, x0, method="L-BFGS-B", bounds=[(-30.0, 30.0)] * len(x0),
                                options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res
    theta = to_natural(best.x, spec)
    out = {"spec": spec, "theta": theta, "loglik": -float(best.fun), "n_obs": len(r),
           "converged": bool(best.success), "persistence": persistence(theta, spec)}
    k = spec.k
    out["aic"] = 2 * k - 2 * out["loglik"]
    out["bic"] = k * math.log(len(r)) - 2 * out["loglik"]
    per = out["persistence"]
    out["half_life"] = math.log(0.5) / math.log(abs(per)) if 0 < abs(per) < 1 else float("inf")
    out["nu"] = float(theta[-1]) if spec.dist == "t" else float("inf")
    # a t fit with nu near 2 matches the shape of the data while its variance, (nu/(nu-2))
    # times the scale squared for Beta-t, is set by the tail index: a density fit, not a
    # variance forecaster (study_egarch.py)
    out["nu_near_2"] = bool(spec.dist == "t" and out["nu"] < 2.5)
    if se:
        H = _hessian(lambda th: -loglik(th, r, spec), theta.copy())
        try:
            cov = np.linalg.inv(H)
            out["se"] = np.sqrt(np.maximum(np.diag(cov), 0.0))
        except np.linalg.LinAlgError:
            out["se"] = np.full(k, np.nan)
    return out


# ------------------------------------------------------------------ simulation and forecasts
def simulate(theta, spec, T, rng, burn=500):
    """T returns from the model (after `burn` discarded), and their conditional variances."""
    theta = np.asarray(theta, dtype=float)
    nu = float(theta[-1]) if spec.dist == "t" else None
    n = T + burn
    eps = draw(spec.dist, nu, n, rng)
    r = np.empty(n)
    s2 = np.empty(n)
    if spec.family == "beta-t":
        omega, phi, kappa, ks, nu = [float(v) for v in theta]
        raw = eps * math.sqrt(nu / (nu - 2))              # t_nu, scale 1
        lam = omega
        for t in range(n):
            r[t] = math.exp(lam) * raw[t]
            s2[t] = math.exp(2 * lam) * nu / (nu - 2)
            y2 = r[t] * r[t]
            u = (nu + 1) * y2 / (nu * math.exp(2 * lam) + y2) - 1.0
            sg = -1.0 if r[t] > 0 else (1.0 if r[t] < 0 else 0.0)
            lam = omega * (1 - phi) + phi * lam + kappa * u + ks * sg * (u + 1.0)
        return r[burn:], s2[burn:]
    p, q = spec.p, spec.q
    if spec.family == "nelson":
        omega, alpha, gamma = theta[0], theta[1:1 + q], theta[1 + q:1 + 2 * q]
        beta = theta[1 + 2 * q:1 + 2 * q + p]
        ez = abs_moment(spec.dist, nu)
        per = float(np.sum(beta))
        hl = [omega / (1 - per)] * p
        hz = [0.0] * q
        ha = [ez] * q
        for t in range(n):
            ls = omega + sum(beta[j] * hl[j] for j in range(p)) + sum(
                alpha[i] * (ha[i] - ez) + gamma[i] * hz[i] for i in range(q))
            s2[t] = math.exp(ls)
            r[t] = math.sqrt(s2[t]) * eps[t]
            hl = [ls] + hl[:-1]
            hz = [eps[t]] + hz[:-1]
            ha = [abs(eps[t])] + ha[:-1]
        return r[burn:], s2[burn:]
    omega, alpha, beta = theta[0], theta[1:1 + q], theta[1 + q:1 + q + p]
    v = omega / (1 - persistence(theta, spec))
    hs, hr = [v] * p, [v] * q
    for t in range(n):
        s2[t] = omega + sum(alpha[i] * hr[i] for i in range(q)) + sum(beta[j] * hs[j] for j in range(p))
        r[t] = math.sqrt(s2[t]) * eps[t]
        hs = [s2[t]] + hs[:-1]
        hr = [r[t] * r[t]] + hr[:-1]
    return r[burn:], s2[burn:]


def forecast(theta, r, spec, horizon, n_sims=20000, seed=0):
    """
    E[s2_{T+k} | returns to T] for k = 1..horizon: closed form for GARCH, by simulation from
    the filtered state for the EGARCH families (E exp(log s2) is not exp(E log s2)).
    """
    theta = np.asarray(theta, dtype=float)
    path = variance_path(theta, r, spec)
    s2_next = float(path[-1])
    if spec.family == "garch" and spec.p == 1 and spec.q == 1:
        omega, a, b = theta[0], theta[1], theta[2]
        vbar = omega / (1 - a - b)
        return vbar + (a + b) ** np.arange(horizon) * (s2_next - vbar)
    rng = np.random.default_rng(seed)
    nu = float(theta[-1]) if spec.dist == "t" else None
    out = np.zeros(horizon)
    if spec.family == "beta-t":
        omega, phi, kappa, ks, nu = [float(v) for v in theta]
        lam = np.full(n_sims, 0.5 * math.log(s2_next * (nu - 2) / nu))
        for k in range(horizon):
            e2l = np.exp(2 * lam)
            out[k] = float(np.mean(e2l)) * nu / (nu - 2)
            y = np.exp(lam) * rng.standard_t(nu, n_sims)
            y2 = y * y
            u = (nu + 1) * y2 / (nu * e2l + y2) - 1.0
            lam = omega * (1 - phi) + phi * lam + kappa * u + ks * np.sign(-y) * (u + 1.0)
        return out
    p, q = spec.p, spec.q
    if spec.family == "nelson":
        omega, alpha, gamma = theta[0], theta[1:1 + q], theta[1 + q:1 + 2 * q]
        beta = theta[1 + 2 * q:1 + 2 * q + p]
        ez = abs_moment(spec.dist, nu)
        # the filtered state: the last p log-variances (the forecast first) and q shocks,
        # most recent first; each simulated day draws its shock and pushes it on
        s2_hist = path[-p - 1:-1][::-1]
        z_obs = (np.asarray(r[-q:], float) / np.sqrt(path[-q - 1:-1]))[::-1]
        hl = [np.full(n_sims, math.log(s2_next))] + [np.full(n_sims, math.log(v)) for v in s2_hist[:p - 1]]
        hz = [np.full(n_sims, v) for v in z_obs]
        for k in range(horizon):
            out[k] = float(np.mean(np.exp(hl[0])))
            hz = [draw(spec.dist, nu, n_sims, rng)] + hz[:q - 1]
            ls = omega + sum(beta[j] * hl[j] for j in range(p)) + sum(
                alpha[i] * (np.abs(hz[i]) - ez) + gamma[i] * hz[i] for i in range(q))
            hl = [ls] + hl[:p - 1]
        return out
    raise NotImplementedError("closed-form forecasts are GARCH(1,1) only")


# ------------------------------------------------------------------ the scan
DEFAULT_SCAN = tuple(
    [Spec("nelson", p, q, d) for d in DISTS for (p, q) in ((1, 1), (1, 2), (2, 1), (2, 2))]
    + [Spec("beta-t")] + [Spec("garch", 1, 1, d) for d in DISTS])


def qlike(target, s2):
    """Patton's (2011) QLIKE: robust to noise in the variance proxy; 0 at a perfect forecast."""
    ratio = np.asarray(target, float) / np.asarray(s2, float)
    ratio = np.maximum(ratio, 1e-300)
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def scan(r, specs=DEFAULT_SCAN, split=None, targets=None, **fit_kw):
    """
    Fit every spec on r[:split] (all of r if split is None) and rank by BIC. With `split`,
    each fit then filters the whole series with its parameters fixed and scores its
    one-step variance forecasts on r[split:] against every target in `targets` (a dict of
    name -> variance proxy on r's index: squared returns, realised variance, or, in a
    simulation, the true integrated variance). Returns rows sorted by BIC.
    """
    r = np.asarray(r, dtype=float)
    train = r if split is None else r[:split]
    rows = []
    for spec in specs:
        f = fit(train, spec, **fit_kw)
        row = {"spec": spec.name, "fit": f, "loglik": f["loglik"], "k": spec.k, "aic": f["aic"],
               "bic": f["bic"], "persistence": f["persistence"], "nu": f["nu"]}
        if split is not None:
            s2 = variance_path(f["theta"], r, spec)[:len(r)]
            row["oos"] = {name: qlike(np.asarray(tg)[split:], s2[split:]) for name, tg in (targets or {}).items()}
        rows.append(row)
    rows.sort(key=lambda z: z["bic"])
    return rows
