"""
Non-Gaussian filtering of rough variance at the zero boundary (Session H).

Session G left one flag on the rough host: where variance sits at zero on ~13% of
days, a Gaussian quasi-likelihood (the exact-moment Kalman filter) is unreliable --
kappa came out +1.79 SE with R = 0.01^2 and -0.65 SE with R = 0.002^2, spread ~2.
Matching the first two moments is not enough when the law of V_{t+1} has an atom
or a pole at zero. Two non-Gaussian answers are built here and compared in
study_zero_boundary.py:

ParticleFilter
    A bootstrap particle filter whose transition IS the positivity-preserving QE step
    (models.rough_heston.LiftedAffineStep) with `substeps` per observation. The
    likelihood estimate log p(y_1..T) = sum_t log mean_i p(y_t | particle_i) is
    unbiased on the natural scale for any number of particles. Two details make it
    usable for estimation, not only for filtering:
      - common random numbers: every draw comes from one seeded stream with shapes
        that do not depend on the parameters (LiftedAffineStep.step_given), so
        likelihoods at nearby parameters share their Monte Carlo error;
      - sorted systematic resampling: particles are ordered by V before a single-
        uniform systematic resample, so the resampling map moves continuously with
        the weights and the likelihood surface is smooth enough to profile.

FourierFilter  (filters/fourier.py)
    The characteristic-function route, after Bates (2006).

Observations: "spot" (y = V + noise) or "rv" (daily realised variance: the day's
integrated variance / dt plus R plus 2/M times its square, as in
filters.kalman.LiftedRoughRVModel).
"""

import math

import numpy as np

import models.rough_heston as rh


def sorted_systematic(key, logw, u):
    """Indices after sorting by `key` and systematic resampling with one uniform u."""
    order = np.argsort(key, kind="stable")
    lw = logw[order]
    w = np.exp(lw - lw.max())
    c = np.cumsum(w)
    c /= c[-1]
    P = len(w)
    idx = np.searchsorted(c, (u + np.arange(P)) / P)
    return order[np.minimum(idx, P - 1)]


def exp_gauss_density(beta, y, R):
    """
    Density at y of Exponential(rate beta) + N(0, R), evaluated in logs:

        beta exp(-beta y + beta^2 R / 2) Phi((y - beta R) / sqrt(R)).

    Written directly, a small conditional mean makes beta huge, exp(beta^2 R / 2)
    overflows while Phi underflows, and the product is nan -- which the adapted
    filter then charged as a -1e6 likelihood day (found on 64-substep data with 16
    filter substeps, Session H). Below x = -25 the Mills-ratio expansion replaces the
    cancelling pair; as beta -> infinity the density tends to N(y; 0, R).
    """
    from scipy.special import log_ndtr
    beta = np.asarray(beta, float)
    x = (y - beta * R) / math.sqrt(R)
    tail = x < -25.0
    logc = np.empty_like(beta)
    bn, xn = beta[~tail], x[~tail]
    logc[~tail] = np.log(bn) - bn * y + 0.5 * bn * bn * R + log_ndtr(xn)
    bt, xt = beta[tail], x[tail]
    logc[tail] = (np.log(bt) - 0.5 * y * y / R - np.log(math.sqrt(2.0 * math.pi) * (-xt))
                  + np.log1p(-1.0 / xt ** 2 + 3.0 / xt ** 4))
    return np.exp(logc)


class ParticleFilter:
    """
    Particle filter for the lifted rough variance. params: kappa, theta, xi, R.
    H, v0 and N define the lift and are fixed at construction, as in
    filters.kalman.LiftedRoughModel.
    """

    def __init__(self, dt, H=0.12, v0=0.04, N=None, n_particles=2000, substeps=4,
                 observation="spot", bars_per_day=78, seed=0, burn_days=40):
        if observation not in ("spot", "rv"):
            raise ValueError("observation must be 'spot' or 'rv'")
        self.dt, self.H, self.v0 = float(dt), float(H), float(v0)
        self.N = rh.N_DEFAULT if N is None else int(N)
        self.w, self.x = rh.lift_nodes(self.H, self.N)
        self.P, self.substeps = int(n_particles), int(substeps)
        self.observation, self.bars_per_day = observation, int(bars_per_day)
        self.seed, self.burn_days = int(seed), int(burn_days)

    def _noise(self, rng):
        n, d = self.P, len(self.w)
        return rng.standard_normal(n), rng.random(n), rng.standard_normal((d, n))

    def run(self, p, y, keep_path=False):
        """
        Filter y. Returns {"loglik", "ess" (per step), "mean_V", "q05", "q95"} with the
        filtered mean and 5/95% quantiles of V (or of the day's variance for RV).
        """
        y = np.asarray(y, float)
        st = rh.LiftedAffineStep(self.w, self.x, self.v0, p["kappa"], p["theta"], p["xi"],
                                 self.dt / self.substeps)
        rng = np.random.default_rng(self.seed)
        Y = np.repeat(st.y_star[:, None], self.P, axis=1)
        for _ in range(self.burn_days * self.substeps):
            z, u, zp = self._noise(rng)
            Y, _, _ = st.step_given(Y, z, u, zp)
        R = float(p["R"])
        ll = 0.0
        T = len(y)
        ess = np.empty(T)
        mean_V = np.empty(T) if keep_path else None
        q05 = np.empty(T) if keep_path else None
        q95 = np.empty(T) if keep_path else None
        log2pi = math.log(2.0 * math.pi)
        for t in range(T):
            I = np.zeros(self.P)
            for _ in range(self.substeps):
                z, u, zp = self._noise(rng)
                Y, V, dI = st.step_given(Y, z, u, zp)
                I += dI
            if self.observation == "spot":
                pred, var = V, np.full(self.P, R)
            else:
                pred = I / self.dt
                var = R + 2.0 / self.bars_per_day * pred * pred
            logw = -0.5 * ((y[t] - pred) ** 2 / var + np.log(var) + log2pi)
            top = logw.max()
            wt = np.exp(logw - top)
            ll += top + math.log(wt.mean())
            wn = wt / wt.sum()
            ess[t] = 1.0 / float(np.sum(wn * wn))
            if keep_path:
                order = np.argsort(pred)
                cw = np.cumsum(wn[order])
                mean_V[t] = float(wn @ pred)
                q05[t] = float(pred[order][np.searchsorted(cw, 0.05)])
                q95[t] = float(pred[order][min(np.searchsorted(cw, 0.95), self.P - 1)])
            idx = sorted_systematic(pred, logw, rng.random())
            Y = Y[:, idx]
        out = {"loglik": ll, "ess": ess}
        if keep_path:
            out.update(mean_V=mean_V, q05=q05, q95=q95)
        return out

    def loglik(self, p, y):
        return self.run(p, y)["loglik"]


class AdaptedParticleFilter(ParticleFilter):
    """
    The same filter with the observation built into the last substep of each day
    (spot observations): a fully adapted auxiliary particle filter.

    Measured on the bootstrap filter first: near zero its likelihood jumped by +/-3
    between neighbouring kappa values even under common random numbers, with the
    effective sample size falling to 1 on some days -- too noisy to estimate with.
    Here, for each particle the QE law of V over the last substep is known in
    closed form, so the predictive likelihood of the observation,

        L_i = int QE(v; m_i, s_i^2) N(y; v, R) dv,

    is computed per particle BEFORE anything is drawn (Gauss-Legendre in s = sqrt(v)
    for the quadratic branch, whose density is smooth in s; closed form for the
    exponential branch, atom included). The log-likelihood increment is
    log mean_i L_i; particles are resampled by L_i; V is then drawn from each
    particle's exact posterior QE(v) N(y; v, R) and the factors follow V along the
    step's regression direction, as in the unconditional step.
    """

    K = 64          # midpoint cells in s = sqrt(v), used for both the likelihood and the draw

    def run(self, p, y, keep_path=False):
        from scipy.special import ndtr, ndtri
        if self.observation != "spot":
            return super().run(p, y, keep_path)
        y = np.asarray(y, float)
        st = rh.LiftedAffineStep(self.w, self.x, self.v0, p["kappa"], p["theta"], p["xi"],
                                 self.dt / self.substeps)
        rng = np.random.default_rng(self.seed)
        Y = np.repeat(st.y_star[:, None], self.P, axis=1)
        for _ in range(self.burn_days * self.substeps):
            z, u, zp = self._noise(rng)
            Y, _, _ = st.step_given(Y, z, u, zp)
        R = float(p["R"])
        sR = math.sqrt(R)
        norm = 1.0 / (sR * math.sqrt(2.0 * math.pi))
        T = len(y)
        ll = 0.0
        ess = np.empty(T)
        mean_V = np.empty(T) if keep_path else None
        q05 = np.empty(T) if keep_path else None
        q95 = np.empty(T) if keep_path else None
        r = st.L_perp.shape[1]
        P = self.P
        ar = np.arange(P)
        for t in range(T):
            for _ in range(self.substeps - 1):
                z, u, zp = self._noise(rng)
                Y, _, _ = st.step_given(Y, z, u, zp)
            yt = float(y[t])
            mu = st.e[:, None] * Y + st.mean_add[:, None]
            m = st.v0 + st.c @ mu
            s2 = np.maximum(st.s2_const + st.s2_lin @ Y, 0.0)
            pos = m > 0
            psi = np.where(pos, s2 / np.where(pos, m * m, 1.0), np.inf)
            quad = pos & (psi <= st.PSI_C)
            expo = pos & ~quad
            L = np.zeros(P)
            L[~pos] = norm * math.exp(-0.5 * yt * yt / R)
            p_atom = np.zeros(P)
            beta_e = np.ones(P)
            if np.any(expo):
                ps_ = psi[expo]
                pa = (ps_ - 1.0) / (ps_ + 1.0)
                be = (1.0 - pa) / m[expo]
                atom = pa * norm * math.exp(-0.5 * yt * yt / R)
                cont = (1.0 - pa) * exp_gauss_density(be, yt, R)
                L[expo] = atom + cont
                p_atom[expo] = atom / np.maximum(atom + cont, 1e-300)
                beta_e[expo] = be
            lo = math.sqrt(max(yt - 7.0 * sR, 0.0))
            hi = math.sqrt(max(yt + 7.0 * sR, 1e-12))
            edges = np.linspace(lo, hi, self.K + 1)
            mid = 0.5 * (edges[1:] + edges[:-1])
            ds = edges[1] - edges[0]
            qi = np.flatnonzero(quad)
            if qi.size:
                ip = 2.0 / np.maximum(psi[qi], 1e-300)
                b2 = ip - 1.0 + np.sqrt(ip) * np.sqrt(ip - 1.0)
                cq = np.sqrt(b2)[:, None]
                ra = np.sqrt(m[qi] / (1.0 + b2))[:, None]
                g = mid[None, :] / ra
                dens = (np.exp(-0.5 * (g - cq) ** 2) + np.exp(-0.5 * (g + cq) ** 2)) / (ra * math.sqrt(2.0 * math.pi))
                mass = dens * (norm * np.exp(-0.5 * (yt - mid * mid) ** 2 / R) * ds)[None, :]
                cum = np.cumsum(mass, axis=1)
                L[qi] = cum[:, -1]
                row_of = np.full(P, -1)
                row_of[qi] = np.arange(qi.size)
            Lsum = float(L.sum())
            if not (Lsum > 0):
                ll += -1e6
                L = np.full(P, 1.0)
                Lsum = float(P)
            ll += math.log(Lsum / P)
            wn = L / Lsum
            ess[t] = 1.0 / float(np.sum(wn * wn))
            idx = sorted_systematic(m, np.log(np.maximum(L, 1e-300)), rng.random())
            uu = rng.random(P)
            V = np.zeros(P)
            e_r = expo[idx]
            if np.any(e_r):
                src = idx[e_r]
                be = beta_e[src]
                mu_tn = yt - be * R
                lo_cdf = ndtr(-mu_tn / sR)
                u1 = uu[e_r]
                pat = p_atom[src]
                u2 = np.clip((u1 - pat) / np.maximum(1.0 - pat, 1e-300), 0.0, 1.0)
                draw = mu_tn + sR * ndtri(np.clip(lo_cdf + u2 * (1.0 - lo_cdf), 1e-15, 1.0 - 1e-15))
                V[e_r] = np.where(u1 <= pat, 0.0, np.maximum(draw, 0.0))
            q_r = quad[idx]
            if np.any(q_r):
                rows = row_of[idx[q_r]]
                cw = cum[rows]
                target = uu[q_r] * cw[:, -1]
                k = np.minimum((cw < target[:, None]).sum(axis=1), self.K - 1)
                prev = np.where(k > 0, cw[np.arange(k.size), np.maximum(k - 1, 0)], 0.0)
                cell = cw[np.arange(k.size), k] - prev
                frac = np.where(cell > 0, (target - prev) / np.where(cell > 0, cell, 1.0), 0.5)
                V[q_r] = (edges[k] + frac * ds) ** 2
            Y = Y[:, idx]
            m_r, s2_r, mu_r = m[idx], s2[idx], mu[:, idx]
            zp = rng.standard_normal((len(self.w), P))
            Vbar = s2_r / (st.xi ** 2 * st.cSc)
            Y = mu_r + st.reg[:, None] * (V - m_r)[None, :] + (st.L_perp @ zp[:r]) * (st.xi * np.sqrt(Vbar))[None, :]
            if keep_path:
                mean_V[t] = float(V.mean())
                q05[t], q95[t] = np.quantile(V, [0.05, 0.95])
        out = {"loglik": ll, "ess": ess}
        if keep_path:
            out.update(mean_V=mean_V, q05=q05, q95=q95)
        return out


def profile_1d(loglik_fn, values):
    """
    Log-likelihood on a grid of one parameter; the maximiser and a Wald SE from a
    quadratic through the best three points (in the parameter's own units).
    """
    values = np.asarray(values, float)
    ll = np.array([loglik_fn(v) for v in values])
    j = int(np.argmax(ll))
    hat, se = float(values[j]), float("nan")
    lo, hi = max(j - 1, 0), min(j + 2, len(values))
    if hi - lo == 3:
        a, b, c = np.polyfit(values[lo:hi], ll[lo:hi], 2)
        if a < 0:
            hat, se = float(-b / (2 * a)), float(1.0 / math.sqrt(-2 * a))
    return {"values": values, "loglik": ll, "hat": hat, "se": se}
