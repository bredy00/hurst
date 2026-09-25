"""
A characteristic-function filter for the lifted rough variance (Session H).

Bates (2006, "Maximum likelihood estimation of latent affine processes") filters a
scalar latent variance by carrying its conditional characteristic function through
time: predict with the affine transition CF, update by Bayes' rule in Fourier space,
and approximate the posterior by a gamma law matched to its first two moments. The
likelihood of each observation is an exact Fourier integral, so the non-Gaussian
shape of the predictive distribution -- the part a Kalman filter throws away -- is
kept. The lifted state here has 44 factors, not one, so the posterior is carried as

    V_t ~ Gamma(k_t, s_t)                                   (the boundary lives here)
    U_t | V_t ~ N(m_t + beta_t (V_t - E V_t), P_t),  P_t w = 0 (so v0 + w'U_t = V_t)

Transition CF of the variance, exact for the continuous lift:

    E[exp(i z V_{t+h}) | U_t] = exp(i z v0 + a(h, z) + b(h, z)' U_t)
    b_i' = -x_i b_i + w_i F(S),   S = sum_i b_i,   F(S) = -kappa S + xi^2 S^2 / 2
    a'   = kappa (theta - v0) S + xi^2 v0 S^2 / 2,    b(0) = i z w,  a(0) = 0

(derived from the generator; solved as psi_i = b_i / w_i with the same exponential
RK4 as the pricing cf, on a graded mesh -- at h -> 0 the kernel mass K_N(0) = 854 makes
S = i z K_N(0) large). The coefficients depend on (h, z) only, so they are solved once
per parameter set. Each day then needs only matrix-vector products on a z-grid:

    log Phi(z) = i z v0 + a + b'(m - beta E V) + b'P b / 2 - k log(1 - s b'beta)
    p(y)       = (1/pi) int_0^Z Re[Phi(z) e^{-i z y}] e^{-R z^2 / 2} dz
    E[V|y], E[V^2|y] from Phi' and Phi'' the same way (z-derivatives by differences)

The observation noise e^{-R z^2/2} bounds the integration range, Z ~ 9 / sqrt(R).
Observations: "spot" (y = V + N(0, R)) or, since Session I, "rv": daily realised
variance, y = I / dt + N(0, R + 2/M E[y]^2) with I the day's integrated variance and
the sampling noise evaluated at the prior mean, as in filters.kalman.LiftedRoughRVModel.
The RV update needs the JOINT transform of (V_{t+1}, I), which is the same Riccati with
a running term (variance_cf_coefficients, zeta), inverted over the observation's
frequency omega = zeta dt, with z-derivatives at z = 0 for the posterior moments of
V_{t+1}. Because RV noise is multiplicative, a day with a small predicted RV is a
PRECISE observation and needs high frequencies: omega up to 9 / sqrt(R_eff), zeta up
to ~2e7 on a 4%-vol day. There the transform's steady state |S| ~ sqrt(2 zeta) / xi makes
the explicit part stiff (h^alpha / Gamma(1 + alpha) * (xi sqrt(2 zeta) + kappa) <= 2, the
pricing solver's ETDRK4 edge), so frequencies come in doubling panels, each with its
own grade-2 mesh of max(240, 6 x its stability count) steps, solved only when a day
first reaches that panel. Measured against an 8x finer mesh: log-transform errors
5e-10 at zeta = 1e2, 5e-6 at 1e5, 2e-5 at 1e6, 2e-6 at 1e7.
"""

import math

import numpy as np

import models.rough_heston as rh


def variance_cf_coefficients(z, h, w, x, v0, kappa, theta, xi, M=480, grade=3.0, zeta=None):
    """
    (a, b) with b of shape (N, n_z):

        E[exp(i z V_h + i zeta I_h) | U_0] = exp(i z v0 + a + b'U_0),   I_h = int_0^h V_s ds.

    zeta (broadcast against z; default 0, the spot transform) is a running term: it
    adds i zeta to every psi_i' and i zeta v0 to a' (the generator of the joint
    transform; Session I, realised-variance observations).

    M = 480 graded steps by default: 240 until Session I, but on the 40-node lift that
    gave one-day moments of V to 1.1e-5 only (480: 6.5e-7, 960: 4.1e-8 -- fourth order).
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n_u = len(z)
    lz = np.zeros(n_u) if zeta is None else np.broadcast_to(np.asarray(zeta, dtype=float), z.shape)
    run_b = 1j * lz                    # the running term of psi' (and of the stages' N)
    run_a = 1j * lz * v0               # ... and of a'
    mesh = h * (np.arange(M + 1) / M) ** grade
    hs = np.diff(mesh)
    c = -np.outer(hs, x)
    E = np.exp(c)
    E2 = np.exp(0.5 * c)
    f1h = rh._phi123(0.5 * c)[0]
    p1, p2, p3 = rh._phi123(c)
    A1 = hs[:, None] * (p1 - 3.0 * p2 + 4.0 * p3)
    A2 = hs[:, None] * (2.0 * p2 - 4.0 * p3)
    A3 = hs[:, None] * (-p2 + 4.0 * p3)
    s_h = 0.5 * hs * (f1h @ w)
    s_h2 = 0.5 * hs * ((E2 * f1h) @ w)
    wE = E * w[None, :]
    wE2 = E2 * w[None, :]
    wA1, wA2, wA3 = A1 @ w, A2 @ w, A3 @ w
    q = 0.5 * xi * xi
    kt = kappa * (theta - v0)

    def F(S):
        return -kappa * S + q * S * S + run_b

    def G(S):
        return kt * S + v0 * q * S * S + run_a

    psi = np.repeat((1j * z)[None, :], len(w), axis=0).astype(complex)
    Psi = (w @ psi.real) * 0.0 + 1j * z * float(w.sum())
    a = np.zeros(n_u, dtype=complex)
    for j in range(M):
        hj = hs[j]
        S2 = wE2[j] @ psi
        S1 = wE[j] @ psi
        Nu = F(Psi)
        Pa = S2 + s_h[j] * Nu
        Na = F(Pa)
        Pb = S2 + s_h[j] * Na
        Nb = F(Pb)
        Pc = S1 + s_h2[j] * Nu + s_h[j] * (2.0 * Nb - Nu)
        Nc = F(Pc)
        a += (hj / 6.0) * (G(Psi) + 2.0 * (G(Pa) + G(Pb)) + G(Pc))
        psi = E[j][:, None] * psi + np.outer(A1[j], Nu) + np.outer(A2[j], Na + Nb) + np.outer(A3[j], Nc)
        Psi = S1 + wA1[j] * Nu + wA2[j] * (Na + Nb) + wA3[j] * Nc
    return a, w[:, None] * psi


class FourierFilter:
    """
    CF filter for the lifted rough variance, observed as spot variance or as daily
    realised variance (observation="rv", Session I). params: kappa, theta, xi, R.
    H, v0, N fixed at construction.

    For "rv" the frequency panels' meshes are sized for xi <= xi_cap and
    kappa <= kappa_cap, so the likelihood is one smooth function of the parameters
    (a mesh that followed xi would jump between step counts); run() refuses
    parameters beyond the caps rather than solve past the stability edge.
    """

    PANEL_NODES = 32            # Gauss-Legendre nodes per frequency panel ("rv")
    ZETA_MAX = 3e7              # frequencies beyond this are not solved; days needing them are counted

    def __init__(self, dt, H=0.12, v0=0.04, N=None, n_z=384, z_sd=9.0, riccati_steps=480, grade=3.0,
                 observation="spot", bars_per_day=78, xi_cap=2.0, kappa_cap=20.0, rv_noise="prior"):
        if observation not in ("spot", "rv"):
            raise ValueError("observation must be 'spot' or 'rv'")
        if rv_noise not in ("prior", "observed"):
            raise ValueError("rv_noise must be 'prior' or 'observed'")
        # RV's sampling variance 2/M x^2 at the prior mean of x (Session G's Kalman model) or
        # at the observed RV itself (Barndorff-Nielsen & Shephard's feasible version)
        self.rv_noise = rv_noise
        self.dt, self.H, self.v0 = float(dt), float(H), float(v0)
        self.N = rh.N_DEFAULT if N is None else int(N)
        self.w, self.x = rh.lift_nodes(self.H, self.N)
        self.n_z, self.z_sd, self.M, self.grade = int(n_z), float(z_sd), int(riccati_steps), float(grade)
        self.observation, self.bars_per_day = observation, int(bars_per_day)
        self.xi_cap, self.kappa_cap = float(xi_cap), float(kappa_cap)
        self._cache_key, self._cache = None, None
        self._panels = None

    # --- realised variance: frequency panels, solved lazily ------------------------
    def _panel_steps(self, zeta_hi):
        """Grade-2 mesh size for frequencies up to zeta_hi: max(240, 6 x the stability count)."""
        al = self.H + 0.5
        h_stab = (2.0 * math.gamma(1.0 + al) / (self.xi_cap * math.sqrt(2.0 * zeta_hi) + self.kappa_cap)) ** (1.0 / al)
        return int(max(240, math.ceil(6.0 * self.dt / h_stab)))

    class _Panels:
        """The omega nodes of one parameter set, extended panel by panel as days need them."""

        def __init__(self, flt, p, omega_1):
            self.f, self.p, self.edge = flt, p, [0.0, omega_1]
            self.om = np.zeros(0)
            self.ow = np.zeros(0)
            self.cols = {k: None for k in ("a", "b", "a_z", "b_z", "a_zz", "b_zz")}
            self.capped = False
            self._add()

        def _add(self):
            f, p = self.f, self.p
            lo, hi = self.edge[-2], self.edge[-1]
            gx, gw = np.polynomial.legendre.leggauss(f.PANEL_NODES)
            om = lo + 0.5 * (hi - lo) * (gx + 1.0)
            ow = 0.5 * (hi - lo) * gw
            zeta = om / f.dt
            dz = 1e-3 / max(float(p["theta"]), 1e-4)            # z-derivatives at z = 0, on V's scale
            M = f._panel_steps(float(zeta.max()))
            z3 = np.concatenate([np.zeros_like(om), np.full_like(om, dz), np.full_like(om, -dz)])
            a3, b3 = variance_cf_coefficients(z3, f.dt, f.w, f.x, f.v0, p["kappa"], p["theta"], p["xi"],
                                              M=M, grade=2.0, zeta=np.tile(zeta, 3))
            n = len(om)
            a0, ap, am = a3[:n], a3[n:2 * n], a3[2 * n:]
            b0, bp, bm = b3[:, :n], b3[:, n:2 * n], b3[:, 2 * n:]
            new = {"a": a0, "b": b0, "a_z": (ap - am) / (2 * dz), "b_z": (bp - bm) / (2 * dz),
                   "a_zz": (ap - 2 * a0 + am) / dz ** 2, "b_zz": (bp - 2 * b0 + bm) / dz ** 2}
            for k, v in new.items():
                old = self.cols[k]
                self.cols[k] = v if old is None else np.concatenate([old, v], axis=-1)
            self.om = np.concatenate([self.om, om])
            self.ow = np.concatenate([self.ow, ow])

        def upto(self, omega):
            """
            Number of nodes in the panels that reach below omega, solving new panels if
            needed. Whole panels, so every day integrates with complete Gauss rules.
            Returns (n, capped): capped if omega lies beyond ZETA_MAX.
            """
            capped = False
            while self.edge[-1] < omega:
                if 2.0 * self.edge[-1] / self.f.dt > self.f.ZETA_MAX:
                    capped = True
                    break
                self.edge.append(2.0 * self.edge[-1])
                self._add()
            n_pan = int(np.searchsorted(np.asarray(self.edge[:-1]), omega, side="left"))
            return max(n_pan, 1) * self.f.PANEL_NODES, capped

    def _run_rv(self, p, y, keep_path=False):
        """
        The RV filter. Same posterior representation as the spot filter; the update uses
        the joint transform of (V_{t+1}, x = I / dt) and moves the factors along BOTH.

        With y = x + eps, eps ~ N(0, R_eff) independent of the state, every posterior moment
        needed comes from integrals on the same omega grid (Tweedie's formula and its
        Gaussian integration by parts): with p, p', p'' the predictive density of y and its
        y-derivatives, and q = E[V_{t+1}; y],

            E[x | y]      = y + R_eff p'/p,     Var(x | y) = R_eff + R_eff^2 (p''/p - (p'/p)^2),
            Cov(V, x | y) = R_eff (q' - E[V|y] p') / p.

        The factors then follow the Gaussian regression of U_{t+1} on (V_{t+1}, x) under the
        prior's exact affine moments -- U = m + B_V (V - E V) + B_x (x - E x) + e -- with x
        given (V, y) linear in V. The spot filter moves them along V's regression only;
        realised variance also carries the day's integral, which is what informs the slow
        factors. A day whose Fourier density is not positive (a numerical failure far in
        the predictive tail) takes the Gaussian update instead and is counted, rather than
        charged a -1e6.
        """
        if p["xi"] > self.xi_cap or p["kappa"] > self.kappa_cap:
            raise ValueError(f"xi {p['xi']:.3g} / kappa {p['kappa']:.3g} beyond the meshes' caps "
                             f"({self.xi_cap}, {self.kappa_cap}): construct with larger xi_cap / kappa_cap")
        y = np.asarray(y, float)
        dt, v0, w = self.dt, self.v0, self.w
        st = rh.LiftedAffineStep(w, self.x, v0, p["kappa"], p["theta"], p["xi"], dt)
        A_U, b_U = st.transition_U()
        EI_U = st.EI_lin @ st.T                                   # E[I | U] = EI_const + EI_U @ U
        mom = st.integrated_moments()
        key = (p["kappa"], p["theta"], p["xi"], float(np.max(y)))
        if self._panels is None or self._panels[0] != key:
            # the first panel ends where the largest observation turns e^{-i omega y} by 1/2 rad
            self._panels = (key, self._Panels(self, p, 0.5 / max(float(np.max(y)), 1e-12)))
        pan = self._panels[1]
        R, twoM = float(p["R"]), 2.0 / self.bars_per_day
        U_star = np.linalg.solve(np.eye(len(w)) - A_U, b_U)
        C = st.stationary_cov_U(U_star)
        mV = v0 + float(w @ U_star)
        varV = float(w @ C @ w)
        beta = (C @ w) / varV
        Pp = C - np.outer(beta, beta) * varV
        m = U_star.copy()
        k_g, s_g = mV * mV / varV, varV / mV
        T = len(y)
        ll, capped_days, fallback_days = 0.0, 0, 0
        ll_t = np.zeros(T)
        path = np.empty((T, 4)) if keep_path else None
        for t in range(T):
            # --- the prior's exact affine moments of (U_{t+1}, V_{t+1}, x)
            Sig = Pp + np.outer(beta, beta) * varV                # Cov(U_t) under the posterior
            y0 = st.T @ m
            Cov = A_U @ Sig @ A_U.T + st.cov_U(m)
            C_UI = A_U @ Sig @ EI_U + st.Tinv @ (mom["C_yI_const"] + mom["C_yI_lin"] @ y0)
            var_I = float(EI_U @ Sig @ EI_U) + max(mom["V_I_const"] + float(mom["V_I_lin"] @ y0), 0.0)
            m_pred = A_U @ m + b_U
            mV_pred = v0 + float(w @ m_pred)
            Ex = max((st.EI_const + float(EI_U @ m)) / dt, 0.0)
            C_Ux = C_UI / dt
            var_x = max(var_I / dt ** 2, 1e-300)
            varV_pred = max(float(w @ Cov @ w), 1e-300)
            cov_Vx = float(w @ C_Ux)
            SZ = np.array([[varV_pred, cov_Vx], [cov_Vx, var_x]])
            CUZ = np.column_stack([Cov @ w, C_Ux])
            det = SZ[0, 0] * SZ[1, 1] - SZ[0, 1] ** 2
            if det > 1e-14 * SZ[0, 0] * SZ[1, 1]:
                B = CUZ @ np.linalg.inv(SZ)
            else:                                                 # x carries nothing beyond V
                B = np.column_stack([(Cov @ w) / varV_pred, np.zeros(len(w))])
            P_Z = Cov - B @ SZ @ B.T
            P_Z = 0.5 * (P_Z + P_Z.T)

            R_eff = R + twoM * (Ex * Ex if self.rv_noise == "prior" else y[t] * y[t])
            n, capped = pan.upto(self.z_sd / math.sqrt(R_eff))
            capped_days += int(capped)
            om, ow = pan.om[:n], pan.ow[:n]
            cols = pan.cols
            a, b = cols["a"][:n], cols["b"][:, :n]
            a_z, b_z, a_zz, b_zz = cols["a_z"][:n], cols["b_z"][:, :n], cols["a_zz"][:n], cols["b_zz"][:, :n]
            base = m - beta * mV
            s = beta @ b
            s_z, s_zz = beta @ b_z, beta @ b_zz
            Pb, Pbz = Pp @ b, Pp @ b_z
            one = 1.0 - s_g * s
            one = np.where(one.real > 1e-12, one, 1e-12 + 1j * one.imag)
            logPhi = a + base @ b + 0.5 * np.einsum("iz,iz->z", b, Pb) - k_g * np.log(one) - 0.5 * R_eff * om * om
            L1 = 1j * v0 + a_z + base @ b_z + np.einsum("iz,iz->z", b_z, Pb) + k_g * s_g * s_z / one
            L2 = (a_zz + base @ b_zz + np.einsum("iz,iz->z", b_zz, Pb) + np.einsum("iz,iz->z", b_z, Pbz)
                  + k_g * s_g * s_zz / one + k_g * s_g * s_g * s_z * s_z / (one * one))
            Phi = np.exp(logPhi - 1j * om * y[t])
            iw = -1j * om
            py = float(ow @ Phi.real) / math.pi
            p1 = float(ow @ (iw * Phi).real) / math.pi               # p'(y)
            p2 = float(ow @ (iw * iw * Phi).real) / math.pi          # p''(y)
            q0 = float(ow @ (-1j * Phi * L1).real) / math.pi         # E[V; y]
            q1 = float(ow @ (iw * (-1j) * Phi * L1).real) / math.pi  # its y-derivative
            M2 = float(ow @ (-Phi * (L2 + L1 * L1)).real) / math.pi
            ok = py > 1e-300 and math.isfinite(py)
            if ok:
                EV = q0 / py
                VV = M2 / py - EV * EV
                Exy = y[t] + R_eff * p1 / py
                Vxy = R_eff + R_eff ** 2 * (p2 / py - (p1 / py) ** 2)
                Cvx = R_eff * (q1 - EV * p1) / py
                ok = EV > 0 and VV > 0 and Vxy > 0 and all(map(math.isfinite, (EV, VV, Exy, Vxy, Cvx)))
            if ok:
                ll_t[t] = math.log(py)
                ll += ll_t[t]
            else:                                                 # Gaussian (Kalman) update for this day
                fallback_days += 1
                S = var_x + R_eff
                nu = y[t] - Ex
                ll_t[t] = -0.5 * (math.log(2.0 * math.pi * S) + nu * nu / S)
                ll += ll_t[t]
                gZ = SZ[:, 1] / S
                EV = max(mV_pred + gZ[0] * nu, 1e-10)
                Exy = Ex + gZ[1] * nu
                post = SZ - np.outer(SZ[:, 1], SZ[:, 1]) / S
                VV, Vxy, Cvx = post[0, 0], max(post[1, 1], 0.0), post[0, 1]
            VV = max(VV, 1e-12 * EV * EV + 1e-20)
            c_xv = Cvx / VV
            m = m_pred + B[:, 0] * (EV - mV_pred) + B[:, 1] * (Exy - Ex)
            beta = B[:, 0] + B[:, 1] * c_xv
            Pp = P_Z + np.outer(B[:, 1], B[:, 1]) * max(Vxy - Cvx * c_xv, 0.0)
            mV, varV = EV, VV
            k_g, s_g = EV * EV / VV, VV / EV
            if keep_path:
                path[t] = (EV, math.sqrt(VV), Ex, Exy)
        out = {"loglik": ll, "loglik_t": ll_t, "capped_days": capped_days,
               "fallback_days": fallback_days, "panels": len(pan.edge) - 1}
        if keep_path:
            out["mean_V"], out["sd_V"] = path[:, 0], path[:, 1]
            out["prior_mean_rv"], out["post_mean_rv"] = path[:, 2], path[:, 3]
        return out

    def _coefficients(self, p):
        key = (p["kappa"], p["theta"], p["xi"], p["R"])
        if key == self._cache_key:
            return self._cache
        Z = self.z_sd / math.sqrt(p["R"])
        gx, gw = np.polynomial.legendre.leggauss(self.n_z)
        z = 0.5 * Z * (gx + 1.0)
        zw = 0.5 * Z * gw
        dz = 1e-4 * Z
        args = (self.dt, self.w, self.x, self.v0, p["kappa"], p["theta"], p["xi"], self.M, self.grade)
        a0, b0 = variance_cf_coefficients(z, *args)
        ap, bp = variance_cf_coefficients(z + dz, *args)
        am, bm = variance_cf_coefficients(z - dz, *args)
        coef = {"z": z, "zw": zw, "a": a0, "b": b0,
                "a_z": (ap - am) / (2 * dz), "b_z": (bp - bm) / (2 * dz),
                "a_zz": (ap - 2 * a0 + am) / dz ** 2, "b_zz": (bp - 2 * b0 + bm) / dz ** 2,
                "damp": np.exp(-0.5 * p["R"] * z * z)}
        self._cache_key, self._cache = key, coef
        return coef

    def run(self, p, y, keep_path=False):
        if self.observation == "rv":
            return self._run_rv(p, y, keep_path)
        y = np.asarray(y, float)
        st = rh.LiftedAffineStep(self.w, self.x, self.v0, p["kappa"], p["theta"], p["xi"], self.dt)
        A_U, b_U = st.transition_U()
        cf = self._coefficients(p)
        z, zw, damp = cf["z"], cf["zw"], cf["damp"]
        a, b, a_z, b_z, a_zz, b_zz = cf["a"], cf["b"], cf["a_z"], cf["b_z"], cf["a_zz"], cf["b_zz"]
        w = self.w
        # start from the stationary law: mean U*, Gaussian covariance split along w
        U_star = np.linalg.solve(np.eye(len(w)) - A_U, b_U)
        C = st.stationary_cov_U(U_star)
        mV = self.v0 + float(w @ U_star)
        varV = float(w @ C @ w)
        beta = (C @ w) / varV
        Pp = C - np.outer(beta, beta) * varV
        m = U_star.copy()
        k_g, s_g = mV * mV / varV, varV / mV
        T = len(y)
        ll = 0.0
        path = np.empty((T, 2)) if keep_path else None
        for t in range(T):
            # --- predict: CF of V_{t+1} mixed over the posterior at t
            base = m - beta * mV
            s = beta @ b                                     # (n_z,)
            s_z, s_zz = beta @ b_z, beta @ b_zz
            Pb, Pbz = Pp @ b, Pp @ b_z
            one = 1.0 - s_g * s
            one = np.where(one.real > 1e-12, one, 1e-12 + 1j * one.imag)
            logPhi = 1j * z * self.v0 + a + base @ b + 0.5 * np.einsum("iz,iz->z", b, Pb) - k_g * np.log(one)
            L1 = 1j * self.v0 + a_z + base @ b_z + np.einsum("iz,iz->z", b_z, Pb) + k_g * s_g * s_z / one
            L2 = (a_zz + base @ b_zz + np.einsum("iz,iz->z", b_zz, Pb) + np.einsum("iz,iz->z", b_z, Pbz)
                  + k_g * s_g * s_zz / one + k_g * s_g * s_g * s_z * s_z / (one * one))
            Phi = np.exp(logPhi)
            # --- the new state's Gaussian part, from exact affine moments
            Cov = A_U @ (Pp + np.outer(beta, beta) * varV) @ A_U.T + st.cov_U(m)
            m_pred = A_U @ m + b_U
            mV_pred = self.v0 + float(w @ m_pred)
            varV_pred = max(float(w @ Cov @ w), 1e-300)
            beta_new = (Cov @ w) / varV_pred
            Pp_new = Cov - np.outer(beta_new, beta_new) * varV_pred
            # --- update: exact Fourier likelihood and posterior moments of V
            ph = np.exp(-1j * z * y[t]) * damp
            py = float(np.sum(zw * (Phi * ph).real)) / math.pi
            M1 = float(np.sum(zw * ((-1j * Phi * L1) * ph).real)) / math.pi
            M2 = float(np.sum(zw * ((-Phi * (L2 + L1 * L1)) * ph).real)) / math.pi
            if not (py > 1e-300):
                ll += -1e6
                EV, VV = mV_pred, varV_pred
            else:
                ll += math.log(py)
                EV = max(M1 / py, 1e-10)
                VV = max(M2 / py - EV * EV, 1e-12 * EV * EV + 1e-20)
            m = m_pred + beta_new * (EV - mV_pred)
            beta, Pp, mV, varV = beta_new, Pp_new, EV, VV
            k_g, s_g = EV * EV / VV, VV / EV
            if keep_path:
                path[t] = (EV, math.sqrt(VV))
        out = {"loglik": ll}
        if keep_path:
            out["mean_V"], out["sd_V"] = path[:, 0], path[:, 1]
        return out

    def loglik(self, p, y):
        return self.run(p, y)["loglik"]
