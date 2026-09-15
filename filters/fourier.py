"""
A characteristic-function filter for the lifted rough variance (Session H).

Bates (2006, "Maximum likelihood estimation of latent affine processes") filters a
scalar latent variance by carrying its conditional characteristic function through
time: predict with the affine transition CF, update by Bayes' rule in Fourier space,
and approximate the posterior by a gamma law matched to its first two moments. The
likelihood of each observation is an exact Fourier integral, so the non-Gaussian
shape of the predictive distribution -- the part a Kalman filter throws away -- is
kept. The lifted state here has 24 factors, not one, so the posterior is carried as

    V_t ~ Gamma(k_t, s_t)                                   (the boundary lives here)
    U_t | V_t ~ N(m_t + beta_t (V_t - E V_t), P_t),  P_t w = 0 (so v0 + w'U_t = V_t)

Transition CF of the variance, exact for the continuous lift:

    E[exp(i z V_{t+h}) | U_t] = exp(i z v0 + a(h, z) + b(h, z)' U_t)
    b_i' = -x_i b_i + w_i F(S),   S = sum_i b_i,   F(S) = -kappa S + xi^2 S^2 / 2
    a'   = kappa (theta - v0) S + xi^2 v0 S^2 / 2,    b(0) = i z w,  a(0) = 0

(derived from the generator; solved as psi_i = b_i / w_i with the same exponential
RK4 as the pricing cf, on a graded mesh -- at h -> 0 the kernel mass K_N(0) = 62 makes
S = i z K_N(0) large). The coefficients depend on (h, z) only, so they are solved once
per parameter set. Each day then needs only matrix-vector products on a z-grid:

    log Phi(z) = i z v0 + a + b'(m - beta E V) + b'P b / 2 - k log(1 - s b'beta)
    p(y)       = (1/pi) int_0^Z Re[Phi(z) e^{-i z y}] e^{-R z^2 / 2} dz
    E[V|y], E[V^2|y] from Phi' and Phi'' the same way (z-derivatives by differences)

The observation noise e^{-R z^2/2} bounds the integration range, Z ~ 9 / sqrt(R).
Spot observations only (y = V + N(0, R)); realised variance would need the CF of the
day's integral, a straightforward extension not built yet.
"""

import math

import numpy as np

import models.rough_heston as rh


def variance_cf_coefficients(z, h, w, x, v0, kappa, theta, xi, M=240, grade=3.0):
    """(a, b) with b of shape (N, n_z): E[exp(i z V_h) | U_0] = exp(i z v0 + a + b'U_0)."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n_u = len(z)
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
        return -kappa * S + q * S * S

    def G(S):
        return kt * S + v0 * q * S * S

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
    CF filter for spot observations of the lifted rough variance. params: kappa,
    theta, xi, R. H, v0, N fixed at construction.
    """

    def __init__(self, dt, H=0.12, v0=0.04, N=None, n_z=384, z_sd=9.0, riccati_steps=240, grade=3.0):
        self.dt, self.H, self.v0 = float(dt), float(H), float(v0)
        self.N = rh.N_DEFAULT if N is None else int(N)
        self.w, self.x = rh.lift_nodes(self.H, self.N)
        self.n_z, self.z_sd, self.M, self.grade = int(n_z), float(z_sd), int(riccati_steps), float(grade)
        self._cache_key, self._cache = None, None

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
