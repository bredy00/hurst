"""
Rough Heston, made Markovian by lifting the fractional kernel.

The model (El Euch & Rosenbaum 2019), written with Heston's own parameters so
that H = 1/2 IS vanilla Heston rather than something that resembles it:

    dS_t / S_t = sqrt(V_t) dW_t
    V_t = V_0 + int_0^t K(t-s) [ kappa (theta - V_s) ds + xi sqrt(V_s) dB_s ]
    K(t) = t^(alpha-1) / Gamma(alpha),   alpha = H + 1/2,   d<W,B> = rho dt

For H < 1/2 the kernel is singular at 0 and V is neither Markov nor a
semimartingale, which is what produces the tau^(H-1/2) explosion of the ATM
skew that vanilla Heston cannot make (Phase 1 baseline: 58% of the market's
one-day skew).

The lift (Abi Jaber & El Euch 2019). The kernel is completely monotone,

    K(t) = int_0^inf e^{-xt} mu(dx),   mu(dx) = x^(-alpha) / (Gamma(alpha) Gamma(1-alpha)) dx

so a quadrature of mu gives K(t) ~ sum_i w_i e^{-x_i t}, and with that kernel the
variance is a weighted sum of N ordinary mean-reverting factors:

    V_t = V_0 + sum_i w_i U^i_t,
    dU^i_t = ( -x_i U^i_t + kappa (theta - V_t) ) dt + xi sqrt(V_t) dB_t,   U^i_0 = 0

which is Markovian in (S, U^1..U^N) and affine, so the characteristic function
is exp of the solution of N coupled ordinary Riccati equations (see
log_char_func). With N = 1 and x_1 = 0, w_1 = 1 the kernel is identically 1 and
every formula below reduces to models.heston line by line; test_rough.py checks
that numerically, and that the H -> 1/2 limit of the full N = 20 lift converges
to Heston.

Conventions match the rest of the package: X = log(S_tau / F) under the forward
measure, cf(u) = E[exp(i u X_tau)], so cf(0) = 1 and cf(-i) = 1. Numpy only.
"""

import math
from dataclasses import dataclass

import numpy as np

from models.heston import HestonParams


@dataclass(frozen=True)
class RoughHestonParams:
    v0: float = 0.04
    kappa: float = 2.0
    theta: float = 0.045
    xi: float = 0.5
    rho: float = -0.7
    H: float = 0.12

    NAMES = ("v0", "kappa", "theta", "xi", "rho", "H")

    def as_tuple(self):
        return (self.v0, self.kappa, self.theta, self.xi, self.rho, self.H)

    @property
    def alpha(self):
        return self.H + 0.5

    @property
    def feller(self):
        """Reported for reference only; for H < 1/2 it is not the relevant condition."""
        return 2.0 * self.kappa * self.theta > self.xi ** 2

    def valid(self):
        return (self.v0 > 0 and self.kappa > 0 and self.theta > 0
                and self.xi > 0 and abs(self.rho) < 1 and 0.0 < self.H <= 0.5)

    def heston(self):
        """The same parameters with the roughness dropped."""
        return HestonParams(self.v0, self.kappa, self.theta, self.xi, self.rho)


# --------------------------------------------------------------- D1: the kernel
def fractional_kernel(t, H):
    """K(t) = t^(alpha-1) / Gamma(alpha), alpha = H + 1/2."""
    a = H + 0.5
    return np.asarray(t, dtype=float) ** (a - 1.0) / math.gamma(a)


def mu_density(x, H):
    """Density of the completely-monotone measure: x^(-alpha) / (Gamma(alpha) Gamma(1-alpha))."""
    a = H + 0.5
    return np.asarray(x, dtype=float) ** (-a) / (math.gamma(a) * math.gamma(1.0 - a))


def mu_mass(lo, hi, H):
    """int_lo^hi mu(dx), closed form."""
    a = H + 0.5
    c = 1.0 / (math.gamma(a) * math.gamma(1.0 - a))
    return c * (np.asarray(hi, float) ** (1.0 - a) - np.asarray(lo, float) ** (1.0 - a)) / (1.0 - a)


def mu_first_moment(lo, hi, H):
    """int_lo^hi x mu(dx), closed form."""
    a = H + 0.5
    c = 1.0 / (math.gamma(a) * math.gamma(1.0 - a))
    return c * (np.asarray(hi, float) ** (2.0 - a) - np.asarray(lo, float) ** (2.0 - a)) / (2.0 - a)


# ------------------------------------------------------ D2: sum of exponentials
N_DEFAULT = 24


def lift_nodes(H, N=N_DEFAULT, eta_1=0.1, eta_N=1.0e5):
    """
    Nodes x_i and weights w_i with K(t) ~ sum_i w_i e^{-x_i t}.

    Abi Jaber-El Euch's construction: partition the x axis into cells, give each
    cell its mu-mass as the weight and its mu-mean as the node,

        w_i = int_{cell_i} mu(dx),    x_i = (1/w_i) int_{cell_i} x mu(dx).

    The cells here are [0, eta_1] followed by N-1 geometric cells up to eta_N.
    Three choices in that sentence matter and were measured (Session D notes):

    - The first cell starts at ZERO. mu has most of its mass at small x -- at
      t = 1 year about 90% of K(t) comes from x < 1 -- and a partition that
      starts at 1/T simply loses it. Starting at zero puts that mass on one slow
      node, exact to O((eta_1 t)^2).
    - eta_N sets the shortest time scale the kernel is right at, roughly
      1/eta_N. The plan's acceptance range was [1 day, 2 years], which
      eta_N = 3000 meets at 0.73% with N = 20 -- but a ONE-DAY option lives
      entirely inside the first day, where that kernel is 31% off at one hour.
      eta_N = 1e5 with N = 24 gives 0.87% on [1 hour, 2 years] (and the same
      on [1 day, 2 years]); below five minutes it is still 12% off, which is
      documented rather than hidden. N = 40 reaches 0.37%.
    - The published r_n = 1 + 10 n^(-0.9) geometric rule, designed for the
      n -> infinity limit, gives 27% at n = 20 on the pricing range and is not
      used.

    Beyond two years the kernel's long memory is reproduced to about 4% (eta_1
    = 0.1 caps the slowest node at x ~ 0.03); that is why the Laplace-domain
    check on z in [0.1, 100] the plan asked for -- which probes t ~ 10 years --
    does not reach 1% and is reported as such.

    H = 1/2 returns the single node (w, x) = (1, 0): the kernel is 1 and the
    lift is vanilla Heston exactly.
    """
    if H >= 0.5 or N == 1:
        return np.array([1.0]), np.array([0.0])
    edges = np.concatenate([[0.0], np.exp(np.linspace(math.log(eta_1),
                                                       math.log(eta_N), N))])
    lo, hi = edges[:-1], edges[1:]
    w = mu_mass(lo, hi, H)
    x = mu_first_moment(lo, hi, H) / w
    return w, x


_LIFT_NODES = lift_nodes          # the construction itself, whatever using_lift has swapped in


class using_lift:
    """
    Context manager: every consumer of the lift -- pricer, calibration, Kalman,
    particle and characteristic-function filters, simulators, jump hosts -- reads
    lift_nodes through this module at call time, so inside

        with rh.using_lift(40, 1e8):
            ...

    the whole stack runs on that lift. Used to measure what adopting a lift would do
    (study_finer_lift.py) and to run the real-data pipeline on either lift
    (run_real_data.py --lift) while the default is undecided. A request for the
    one-node lift (N = 1, vanilla Heston) is passed through unchanged.
    """

    def __init__(self, N, eta_N):
        self.N, self.eta_N = int(N), float(eta_N)
        self._saved = None

    def __enter__(self):
        global lift_nodes
        exact = _LIFT_NODES
        N, eta_N = self.N, self.eta_N

        def patched(H, *args, **kwargs):
            n_req = args[0] if args else kwargs.get("N", N_DEFAULT)
            if n_req == 1:
                return exact(H, 1)
            return exact(H, N, eta_1=kwargs.get("eta_1", 0.1), eta_N=eta_N)
        self._saved = lift_nodes                     # restores an outer using_lift on exit
        lift_nodes = patched
        return self

    def __exit__(self, *exc):
        global lift_nodes
        lift_nodes = self._saved


def kernel_approx(t, weights, nodes):
    """sum_i w_i e^{-x_i t}, vectorised over t."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    return (weights[None, :] * np.exp(-np.outer(t, nodes))).sum(axis=1)


def kernel_error(H, N=N_DEFAULT, t_lo=1.0 / 365.0, t_hi=2.0, n=400, **kw):
    """(max relative error, L2 relative error) of the lift's kernel on [t_lo, t_hi]."""
    w, x = lift_nodes(H, N, **kw)
    ts = np.exp(np.linspace(math.log(t_lo), math.log(t_hi), n))
    exact = fractional_kernel(ts, H)
    rel = np.abs(kernel_approx(ts, w, x) - exact) / exact
    return float(rel.max()), float(math.sqrt(np.mean(rel ** 2)))


def laplace_kernel(z, H):
    """Laplace transform of K: z^(-alpha)."""
    return np.asarray(z, dtype=float) ** (-(H + 0.5))


def laplace_approx(z, weights, nodes):
    """Laplace transform of the lifted kernel: sum_i w_i / (z + x_i)."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    return (weights[None, :] / (z[:, None] + nodes[None, :])).sum(axis=1)


# ------------------------------------------------------------ D3: simulation
def _phi1(c):
    """(e^c - 1)/c for real c <= 0, with the c = 0 limit."""
    c = np.asarray(c, dtype=float)
    out = np.ones_like(c)
    nz = np.abs(c) > 1e-14
    out[nz] = np.expm1(c[nz]) / c[nz]
    return out


def simulate_lifted_gaussian(weights, nodes, dZ, dt, cell_mean=False):
    """
    The linear lift, driven by given increments dZ of shape (n_steps,) or
    (n_steps, n_paths):   U_i <- e^{-x_i dt} (U_i + dZ_k),   V = sum_i w_i U_i.

    That recursion is an exact evaluation of the convolution sum
        V(t_n) = sum_{k<n} K_N(t_n - t_k) dZ_k,    K_N = sum_i w_i e^{-x_i t}
    (the semigroup identity e^{-x(t_n - t_k)} = e^{-x dt} ... e^{-x dt}), which
    test_rough.py checks to rounding. With cell_mean=True the shock enters
    through phi_1(-x_i dt) instead, so the recursion evaluates the convolution
    with the CELL-AVERAGED kernel (1/dt) int_{j dt}^{(j+1) dt} K_N -- the
    discretisation the fBm fixture bug (docs/fbm_helper_bug.md) showed is the
    one that keeps the short-lag roughness, and the right thing to compare
    against a cell-averaged true kernel.

    Returns V with the same shape as dZ.
    """
    dZ = np.asarray(dZ, dtype=float)
    E = np.exp(-nodes * dt)
    gain = _phi1(-nodes * dt) if cell_mean else E
    shape = (slice(None),) + (None,) * (dZ.ndim - 1)
    U = np.zeros(nodes.shape + dZ.shape[1:])
    out = np.empty_like(dZ)
    for k in range(dZ.shape[0]):
        U = E[shape] * U + gain[shape] * dZ[k]
        out[k] = np.tensordot(weights, U, axes=(0, 0))
    return out


def convolve_kernel(kernel_values, dZ):
    """
    Direct Volterra convolution V(t_n) = sum_{k<=n} kernel[n-k] dZ_k for a kernel
    sampled (or cell-averaged) at lags dt, 2dt, ...; O(n^2), used only as the
    reference the lift is checked against.
    """
    dZ = np.asarray(dZ, dtype=float)
    n = dZ.shape[0]
    out = np.empty_like(dZ)
    for m in range(n):
        out[m] = np.tensordot(kernel_values[:m + 1][::-1], dZ[:m + 1], axes=(0, 0))
    return out


def simulate(p, tau, n_paths=100_000, n_steps=500, seed=0, N=N_DEFAULT, antithetic=True):
    """
    Full-truncation exponential-Euler simulation of the lifted rough Heston,
    X = log(S/F), for validating the characteristic function independently of
    the Riccati derivation.

    Each factor is stepped with its exact linear propagator e^{-x_i dt} and the
    drift/diffusion enter through phi_1(-x_i dt) -- the stiff fast factors
    (x_i up to a few thousand) would otherwise force a tiny dt. The variance is
    floored at zero wherever it is used (Lord et al.), never reflected.
    """
    rng = np.random.default_rng(seed)
    v0, kappa, theta, xi, rho, H = p.as_tuple()
    w, x = lift_nodes(H, N)
    dt = tau / n_steps
    sdt = math.sqrt(dt)
    E = np.exp(-x * dt)[:, None]
    G = _phi1(-x * dt)[:, None]

    m = n_paths // 2 if antithetic else n_paths
    z1 = rng.standard_normal((n_steps, m))
    z2 = rng.standard_normal((n_steps, m))
    if antithetic:
        z1 = np.concatenate([z1, -z1], axis=1)
        z2 = np.concatenate([z2, -z2], axis=1)
    n = z1.shape[1]
    X = np.zeros(n)
    U = np.zeros((len(w), n))
    V = np.full(n, v0)
    srho = math.sqrt(1.0 - rho * rho)
    for i in range(n_steps):
        vp = np.maximum(V, 0.0)
        sv = np.sqrt(vp)
        dW = z1[i] * sdt
        dB = (rho * z1[i] + srho * z2[i]) * sdt
        X += -0.5 * vp * dt + sv * dW
        shock = kappa * (theta - vp) * dt + xi * sv * dB      # same for every factor
        U = E * U + G * shock[None, :]
        V = v0 + w @ U
    return X


def mc_call(p, tau, k, n_paths=100_000, n_steps=500, seed=0, N=N_DEFAULT, scheme="euler"):
    """Undiscounted call value in units of F and its standard error; k = log(K/F)."""
    X = simulate(p, tau, n_paths, n_steps, seed, N) if scheme == "euler" else \
        simulate_qe(p, tau, n_paths, n_steps, seed, N)
    pay = np.maximum(np.exp(X) - math.exp(k), 0.0)
    return float(pay.mean()), float(pay.std(ddof=1) / math.sqrt(len(pay)))


# ------------------------------------ G: exact moments and a positivity-preserving step
def _phi_k(c, k):
    """
    phi_1 or phi_2 of a real array c <= 0, and only that one: Horner series below
    |c| = 0.1 (ten terms, exact to rounding there), expm1 above. _phi123 computes
    all three on the full array, which was 64% of a LiftedAffineStep build.
    """
    c = np.minimum(np.asarray(c, dtype=float), 0.0)
    out = np.empty_like(c)
    small = np.abs(c) < 0.1
    if np.any(small):
        cs = c[small]
        acc = np.full_like(cs, 1.0 / _FACT[k + 11])
        for j in range(10, -1, -1):
            acc = acc * cs + 1.0 / _FACT[j + k]
        out[small] = acc
    big = ~small
    if np.any(big):
        cb = c[big]
        e = np.expm1(cb)
        out[big] = e / cb if k == 1 else (e - cb) / (cb * cb)
    return out


def _phi12_nonpos(c):
    """phi_1, phi_2 of a real array c <= 0 (series near zero, see _phi_k)."""
    return _phi_k(c, 1), _phi_k(c, 2)


class LiftedAffineStep:
    """
    Exact conditional moments of the lifted rough Heston over one step h, and the
    positivity-preserving step built on them (Session G, flag 3).

    Why a new step. The exponential-Euler step draws a Gaussian V-increment of
    size ~ xi sqrt(V h) (1/h) int_0^h K_N, which at h = 1/1008 and H = 0.12 is
    15x xi sqrt(V h): variance at theta = 0.04 goes below zero on ~13% of days.
    The CONTINUOUS lifted model cannot (its kernel is completely monotone; Abi
    Jaber, Larsson & Pulido 2019), so the negativity is the scheme's, and the
    floor that hides it is a non-linearity that biased the filter's QML kappa by
    2-4 SE.

    The model is affine, and in the right coordinates it is diagonal. With
    D = diag(sqrt w), the drift matrix A = diag(x) + kappa 1 w' satisfies
    D A D^-1 = diag(x) + kappa sqrt(w) sqrt(w)' = Q diag(lambda) Q' (symmetric), so in
    y = Q' D U every coordinate is a scalar OU driven by the SAME sqrt(V) dB:

        dy_k = (-lambda_k y_k + alpha_k) dt + xi c_k sqrt(V) dB,   V = v0 + c'y,
        c = Q' sqrt(w),   alpha = kappa (theta - v0) c.

    Every conditional moment over [0, h] is then a closed-form integral of
    exponentials against E[V_s], itself a sum of exponentials in s. In
    particular the covariance is the Ito isometry, evaluated exactly:

        Cov(y_i, y_j) = xi^2 c_i c_j int_0^h e^{-(lambda_i + lambda_j)(h - s)} E[V_s] ds.

    The step (Andersen's QE, adapted to the lift):
      1. m = E[V_h], s^2 = Var[V_h]: exact, affine in the current state, O(N) per path;
      2. V_h drawn from Andersen's quadratic-exponential scheme, which matches
         (m, s^2) exactly and is non-negative by construction;
      3. the factors take V_h's surprise along the conditional regression
         direction, plus Gaussian noise ORTHOGONAL to V (its projection on c is
         identically zero), so V_h = v0 + c'y_h holds exactly. The regression
         direction and the orthogonal noise shape are evaluated at the
         stationary state -- V's own moments are exact at every state, the split
         of its surprise across the factors is exact at stationarity.

    moments_U() gives the exact mean and covariance in the factor coordinates U,
    which is what the Kalman filter now uses instead of the Euler transition.
    """

    PSI_C = 1.5       # Andersen's switching value

    def __init__(self, w, x, v0, kappa, theta, xi, h, rho=0.0):
        self.w, self.x = np.asarray(w, float), np.asarray(x, float)
        self.v0, self.kappa, self.theta, self.xi, self.h, self.rho = map(
            float, (v0, kappa, theta, xi, h, rho))
        sw = np.sqrt(self.w)
        lam, Q = np.linalg.eigh(np.diag(self.x) + self.kappa * np.outer(sw, sw))
        lam = np.maximum(lam, 0.0)
        self.lam = lam
        self.T = Q.T * sw[None, :]                # y = T U
        self.Tinv = Q / sw[:, None]               # U = Tinv y
        c = Q.T @ sw
        self.c = c
        kt = self.kappa * (self.theta - self.v0)
        self.alpha = kt * c
        h = self.h
        self.e = np.exp(-lam * h)
        p1, p2 = _phi12_nonpos(-lam * h)
        self.mean_add = self.alpha * h * p1       # E[y_h] = e * y + mean_add

        r = lam[:, None] + lam[None, :]
        I0 = h * _phi_k(-r * h, 1)                                           # (N, N)
        J = self._J(r[:, :, None], lam[None, None, :])                      # (N, N, N)
        L = self._L(I0[:, :, None], J, r[:, :, None], lam[None, None, :])  # (N, N, N)
        c2 = c * c
        P0 = self.v0 * I0 + kt * (L @ c2)                                  # state-free part
        P1 = J * c[None, None, :]                                          # times y_k
        cc = np.outer(c, c)
        self._cov_const = self.xi ** 2 * cc * P0
        self._cov_lin = self.xi ** 2 * cc[:, :, None] * P1                 # (N, N, N)
        # the same covariance in factor coordinates, affine in U: CU0 + CU1 @ U
        self._covU_const = self.Tinv @ self._cov_const @ self.Tinv.T
        self._covU_lin = np.einsum("ia,abk,jb,kl->ijl", self.Tinv, self._cov_lin, self.Tinv, self.T,
                                   optimize=True)
        self.s2_const = float(self.xi ** 2 * (c2 @ P0 @ c2))
        self.s2_lin = self.xi ** 2 * np.einsum("i,j,ijk->k", c2, c2, P1)

        # Moments of the integrated variance and of Z = int sqrt(V) dB, for log S
        self.EI_const = self.v0 * h + kt * float(c2 @ (h * h * p2))
        self.EI_lin = c * h * p1
        I0d = h * p1                                                       # I0(lambda_i)
        Jd = self._J(lam[:, None], lam[None, :])                           # (N, N)
        Ld = self._L(I0d[:, None], Jd, lam[:, None], lam[None, :])
        self.zeta_const = self.xi * float(c2 @ (self.v0 * I0d + kt * (Ld @ c2)))
        self.zeta_lin = self.xi * c * (c2 @ Jd)

        # Stationary state, and the fixed shapes used to split V's surprise
        with np.errstate(divide="ignore", invalid="ignore"):
            self.y_star = np.where(lam > 0, self.alpha / np.where(lam > 0, lam, 1.0), 0.0)
        self.V_star = self.v0 + float(c @ self.y_star)
        S_ref = cc * I0                                                    # per unit xi^2 V
        Sc = S_ref @ c
        cSc = float(c @ Sc)
        self.reg = Sc / cSc                                                # c'reg = 1
        S_perp = S_ref - np.outer(Sc, Sc) / cSc
        ev, evec = np.linalg.eigh(0.5 * (S_perp + S_perp.T))
        keep = ev > 1e-12 * max(float(ev.max()), 1e-300)
        self.L_perp = evec[:, keep] * np.sqrt(ev[keep])[None, :]
        self.L_perp -= np.outer(self.reg, c @ self.L_perp)                 # exact c-orthogonality
        self.cSc = cSc
        # Leverage. Every factor's innovation comes from the same dB, so the
        # factors orthogonal to V_h are correlated with the return's dZ -- that
        # is how today's return reaches FUTURE variance through the slow factors.
        # Drawing that part independently lost the skew (Session G, measured:
        # +2.0e-3 on a k = +0.08 call at 250 steps). At the reference state,
        # Cov(y_h, dZ) = xi V c * I0(lambda); its part orthogonal to V_h is
        # Cov(V_h, dZ) * d_ref, and the joint draw uses b_ref with
        # L_perp b_ref = d_ref.
        z_ref = c * I0d
        d_ref = z_ref / float(c @ z_ref) - self.reg
        self.b_ref = np.linalg.lstsq(self.L_perp, d_ref, rcond=None)[0]

    # closed-form exponential integrals, all on [0, h]
    def _J(self, r, lam):
        """int_0^h e^{-r(h-s)} e^{-lam s} ds = h e^{-min(r, lam) h} phi_1(-|r - lam| h)."""
        h = self.h
        return h * np.exp(-np.minimum(r, lam) * h) * _phi_k(-np.abs(r - lam) * h, 1)

    def _L(self, I0, J, r, lam):
        """int_0^h e^{-r(h-s)} s phi_1(-lam s) ds = (I0(r) - J(r, lam)) / lam."""
        h = self.h
        small = lam * h < 1e-7
        with np.errstate(divide="ignore", invalid="ignore"):
            direct = (I0 - J) / np.where(small, 1.0, lam)
        limit = h * h * _phi_k(-np.broadcast_to(r, np.broadcast(r, lam).shape) * h, 2)
        return np.where(small, limit, direct)

    # --- the Kalman filter's view: exact moments in factor coordinates --------
    def transition_U(self):
        """E[U_h | U_0] = A U_0 + b, exactly."""
        A = self.Tinv @ (self.e[:, None] * self.T)
        b = self.Tinv @ self.mean_add
        return A, b

    def cov_U(self, U):
        """
        Cov[U_h | U_0]: exact for an admissible state, and always clipped to the
        PSD cone. A FILTERED state can be inadmissible (its implied E[V_s] dips
        below zero), where the affine formula is indefinite. Clipping only when a
        check trips was tried first: whether the check trips switches on and off
        as the parameters move, and one such switch put a 0.34 jump into the
        log-likelihood between kappa = 3.40 and 3.42 -- enough to break
        Nelder-Mead and the Hessian SE. Always clipping is continuous.
        """
        U = np.asarray(U, float)
        CU = self._covU_const + self._covU_lin @ U
        ev, vec = np.linalg.eigh(0.5 * (CU + CU.T))
        return (vec * np.maximum(ev, 0.0)) @ vec.T

    def stationary_cov_U(self, U):
        """
        P solving P = A P A' + Q(U): in y coordinates A is diagonal, so
        P_ij = Q_ij / (1 - e_i e_j) exactly -- no iteration (the slowest factor
        decays at ~0.9996 a day, so iterating to 1e-14 would take ~40,000 steps).
        """
        Qy = self.T @ self.cov_U(U) @ self.T.T
        Py = Qy / (1.0 - np.outer(self.e, self.e))
        return self.Tinv @ Py @ self.Tinv.T

    # --- simulation ---------------------------------------------------------------
    def qe_draw(self, m, s2, z, u):
        """Andersen's QE with moments (m, s2): non-negative, both moments exact."""
        m = np.asarray(m, float)
        s2 = np.maximum(np.asarray(s2, float), 0.0)
        out = np.zeros_like(m)
        pos = m > 0
        psi = np.where(pos, s2 / np.where(pos, m * m, 1.0), np.inf)
        quad = pos & (psi <= self.PSI_C)
        if np.any(quad):
            ip = 2.0 / np.maximum(psi[quad], 1e-300)
            b2 = ip - 1.0 + np.sqrt(ip) * np.sqrt(ip - 1.0)
            a = m[quad] / (1.0 + b2)
            out[quad] = a * (np.sqrt(b2) + z[quad]) ** 2
        expo = pos & ~quad
        if np.any(expo):
            ps = psi[expo]
            p = (ps - 1.0) / (ps + 1.0)
            beta = (1.0 - p) / m[expo]
            uu = u[expo]
            out[expo] = np.where(uu <= p, 0.0,
                                 np.log((1.0 - p) / np.maximum(1.0 - uu, 1e-300)) / beta)
        return out

    def qe_log_mgf(self, A, m, s2):
        """
        log E[exp(A V)] for the QE draw with moments (m, s2), A <= 0 (closed form):
        quadratic branch V = a (b + Z)^2 gives exp(A a b^2 / (1 - 2 A a)) / sqrt(1 - 2 A a);
        exponential branch gives p + (1 - p) beta / (beta - A).
        """
        m = np.asarray(m, float)
        s2 = np.maximum(np.asarray(s2, float), 0.0)
        A = np.minimum(np.asarray(A, float), 0.0)
        out = np.zeros_like(m)
        pos = m > 0
        psi = np.where(pos, s2 / np.where(pos, m * m, 1.0), np.inf)
        quad = pos & (psi <= self.PSI_C)
        if np.any(quad):
            ip = 2.0 / np.maximum(psi[quad], 1e-300)
            b2 = ip - 1.0 + np.sqrt(ip) * np.sqrt(ip - 1.0)
            a = m[quad] / (1.0 + b2)
            d = 1.0 - 2.0 * A[quad] * a
            out[quad] = A[quad] * a * b2 / d - 0.5 * np.log(d)
        expo = pos & ~quad
        if np.any(expo):
            ps = psi[expo]
            p = (ps - 1.0) / (ps + 1.0)
            beta = (1.0 - p) / m[expo]
            out[expo] = np.log(p + (1.0 - p) * beta / (beta - A[expo]))
        return out

    def step_given(self, y, z, u, zp_full):
        """
        The variance step with the noise supplied: z, u of shape (n,), zp_full of
        shape (N, n) of which the first L_perp.shape[1] rows are used. Returns
        (y_h, V_h, dI). The shapes do not depend on the parameters, so a particle
        filter can run common random numbers across parameter values (Session H).
        """
        mu = self.e[:, None] * y + self.mean_add[:, None]
        m = self.v0 + self.c @ mu
        s2 = np.maximum(self.s2_const + self.s2_lin @ y, 0.0)
        V = self.qe_draw(m, s2, z, u)
        Vbar = s2 / (self.xi ** 2 * self.cSc)
        r = self.L_perp.shape[1]
        y_new = mu + self.reg[:, None] * (V - m)[None, :] + \
            (self.L_perp @ zp_full[:r]) * (self.xi * np.sqrt(Vbar))[None, :]
        EI = np.maximum(self.EI_const + self.EI_lin @ y, 0.0)
        dI = np.maximum(EI + 0.5 * self.h * (V - m), 0.0)
        return y_new, V, dI

    def step_y(self, y, rng, with_log_price=False):
        """
        One QE step for y of shape (N, n_paths). Returns y_h, V_h and, if asked,
        (dI, dZ) -- the integrated variance and int sqrt(V) dB over the step.
        """
        n = y.shape[1]
        mu = self.e[:, None] * y + self.mean_add[:, None]
        m = self.v0 + self.c @ mu
        s2 = np.maximum(self.s2_const + self.s2_lin @ y, 0.0)
        z = rng.standard_normal(n)
        u = rng.random(n)
        V = self.qe_draw(m, s2, z, u)
        Vbar = s2 / (self.xi ** 2 * self.cSc)
        zp = rng.standard_normal((self.L_perp.shape[1], n))
        scale = self.xi * np.sqrt(Vbar)
        y_new = mu + self.reg[:, None] * (V - m)[None, :] + (self.L_perp @ zp) * scale[None, :]
        if not with_log_price:
            return y_new, V
        EI = np.maximum(self.EI_const + self.EI_lin @ y, 0.0)
        dI = np.maximum(EI + 0.5 * self.h * (V - m), 0.0)
        # Cov(V_h, dZ) is >= 0 for any admissible state and bounded by Cauchy-Schwarz
        cov = np.clip(self.zeta_const + self.zeta_lin @ y, 0.0, np.sqrt(s2 * EI))
        with np.errstate(divide="ignore", invalid="ignore"):
            gam = np.where(s2 > 0, cov / np.where(s2 > 0, s2, 1.0), 0.0)
            lev = np.where(scale > 0, cov / np.where(scale > 0, scale, 1.0), 0.0)
        # dZ = gam (V_h - m) + [part correlated with the orthogonal factor noise] + independent rest
        corr_part = (self.b_ref @ zp) * lev
        resid = np.maximum(EI - gam * cov - (self.b_ref @ self.b_ref) * lev * lev, 0.0)
        dZ = gam * (V - m) + corr_part + np.sqrt(resid) * rng.standard_normal(n)
        # Martingale correction (Andersen 2008): conditional on V_h and the factor
        # noise, E[exp(dX)] = exp(A (V_h - m) + rho corr_part + c0), with
        # A = rho gam - rho^2 h / 4 and c0 = -rho^2 EI / 2 + rho^2 resid / 2. The
        # factor-noise part is Gaussian (its mgf is exact); V_h is not, so its
        # QE mgf is used.
        rho = self.rho
        A = rho * gam - 0.25 * rho * rho * self.h
        c0 = -0.5 * rho * rho * EI + 0.5 * rho * rho * resid +             0.5 * rho * rho * (self.b_ref @ self.b_ref) * lev * lev
        drift_fix = -(c0 - A * m + self.qe_log_mgf(A, m, s2))
        return y_new, V, dI, dZ, drift_fix

    def integrated_moments(self, n_panels=36, order=16):
        """
        Affine coefficients of the joint conditional moments of the factors and the
        integrated variance I = int_0^h V_s ds over the step (Session G, realised
        variance observations). With tau = h - r and G(tau) = sum_k c_k^2 (1 - e^{-lambda_k tau}) / lambda_k,

            I - E[I]          = xi int_0^h G(h - r) sqrt(V_r) dB_r
            Cov(y_i(h), I)    = xi^2 c_i int_0^h e^{-lambda_i tau} G(tau) E[V_{h - tau}] dtau
            Var(I)            = xi^2 int_0^h G(tau)^2 E[V_{h - tau}] dtau

        and E[V_r] is affine in the starting state, so each is const + lin @ y0. The
        integrals are evaluated once by Gauss-Legendre on panels graded
        geometrically towards both ends of the step (fast factors live within
        1/lambda of either end); check_quadrature() compares the same grid with the
        closed-form factor covariance. Cached on the step.
        """
        if getattr(self, "_int_mom", None) is not None:
            return self._int_mom
        import pricing.fourier as fo
        h, lam, c = self.h, self.lam, self.c
        gl_x, gl_w = fo._leggauss(order)
        g = np.geomspace(1e-11, 0.5, n_panels // 2)
        edges = np.unique(np.concatenate([[0.0, h], h * g, h - h * g]))
        lo, hi = edges[:-1], edges[1:]
        half = 0.5 * (hi - lo)
        tau = (half[:, None] * (gl_x[None, :] + 1.0) + lo[:, None]).ravel()
        wq = (half[:, None] * gl_w[None, :]).ravel()
        r = h - tau
        kt = self.kappa * (self.theta - self.v0)
        # E[V_r] = E0(r) + sum_m y0_m B_m(r)
        E0 = self.v0 + ((kt * c * c)[None, :] * (r[:, None] * _phi_k(-np.outer(r, lam), 1))).sum(axis=1)
        B = c[None, :] * np.exp(-np.outer(r, lam))                        # (Q, N)
        with np.errstate(divide="ignore", invalid="ignore"):
            gk = tau[:, None] * _phi_k(-np.outer(tau, lam), 1)            # (1 - e^{-lam tau}) / lam
        G = gk @ (c * c)                                                  # (Q,)
        El = np.exp(-np.outer(tau, lam))                                  # (Q, N)
        xi2 = self.xi ** 2
        wG = wq * G
        C_const = xi2 * c * ((El * (wG * E0)[:, None]).sum(axis=0))
        C_lin = xi2 * c[:, None] * (El.T @ (wG[:, None] * B))              # (N, N): [i, m]
        V_const = xi2 * float(np.sum(wq * G * G * E0))
        V_lin = xi2 * ((wq * G * G) @ B)                                   # (N,)
        self._int_mom = {"C_yI_const": C_const, "C_yI_lin": C_lin, "V_I_const": V_const, "V_I_lin": V_lin,
                         "nodes": tau, "weights": wq, "E0": E0, "B": B}
        return self._int_mom

    def check_quadrature(self, y0):
        """Largest relative error of the quadrature grid on Cov(y, y), against the closed form."""
        m = self.integrated_moments()
        tau, wq, E0, B = m["nodes"], m["weights"], m["E0"], m["B"]
        EV = E0 + B @ y0
        El = np.exp(-np.outer(tau, self.lam))
        Cq = self.xi ** 2 * np.outer(self.c, self.c) * ((El * (wq * EV)[:, None]).T @ El)
        Cx = self._cov_const + self._cov_lin @ y0
        return float(np.max(np.abs(Cq - Cx)) / np.max(np.abs(Cx)))

    def y_from_U(self, U):
        return self.T @ U

    def U_from_y(self, y):
        return self.Tinv @ y


def simulate_qe(p, tau, n_paths=100_000, n_steps=500, seed=0, N=N_DEFAULT, return_V=False,
                martingale=True):
    """
    X = log(S/F) under the lifted rough Heston with the positivity-preserving
    step (LiftedAffineStep). The variance is never negative and is never floored.

        dX = -1/2 dI + rho dZ + sqrt(1 - rho^2) sqrt(dI) dW_perp

    with dI and dZ = int sqrt(V) dB drawn consistently with the V draw: dZ's
    conditional mean given V_h is the exact regression Cov(V_h, Z)/Var(V_h).
    """
    rng = np.random.default_rng(seed)
    w, x = lift_nodes(p.H, N)
    st = LiftedAffineStep(w, x, p.v0, p.kappa, p.theta, p.xi, tau / n_steps, p.rho)
    y = np.zeros((len(w), n_paths))            # U_0 = 0, V_0 = v0
    X = np.zeros(n_paths)
    srho = math.sqrt(1.0 - p.rho * p.rho)
    Vmin = np.inf
    for _ in range(n_steps):
        y, V, dI, dZ, fix = st.step_y(y, rng, with_log_price=True)
        X += -0.5 * dI + p.rho * dZ + srho * np.sqrt(dI) * rng.standard_normal(n_paths)             + (fix if martingale else 0.0)
        Vmin = min(Vmin, float(V.min()))
    if return_V:
        return X, V, Vmin
    return X


# ------------------------------------------- D4: the characteristic function
_FACT = [math.factorial(k) for k in range(16)]


def _phi123(c):
    """
    phi_1, phi_2, phi_3 of a real array c (here c = -x h <= 0):
        phi_1 = (e^c - 1)/c,  phi_2 = (e^c - 1 - c)/c^2,  phi_3 = (e^c - 1 - c - c^2/2)/c^3.
    Direct formulas cancel catastrophically for small |c| (phi_3 loses 6 eps/c^2,
    i.e. everything below |c| ~ 1e-4), so |c| < 0.1 uses the series
    phi_k(c) = sum_j c^j / (j+k)!, twelve terms, exact to machine precision there.
    """
    c = np.asarray(c, dtype=float)
    small = np.abs(c) < 0.1
    cs = np.where(small, c, 0.0)
    p1 = np.zeros_like(c)
    p2 = np.zeros_like(c)
    p3 = np.zeros_like(c)
    term = np.ones_like(c)
    for j in range(12):
        p1 += term / _FACT[j + 1]
        p2 += term / _FACT[j + 2]
        p3 += term / _FACT[j + 3]
        term = term * cs
    cb = np.where(small, 1.0, c)
    e = np.expm1(cb)
    q1 = e / cb
    q2 = (e - cb) / (cb * cb)
    q3 = (e - cb - 0.5 * cb * cb) / (cb ** 3)
    return (np.where(small, p1, q1), np.where(small, p2, q2), np.where(small, p3, q3))


# Stability constants on z = h^alpha / Gamma(1+alpha) * (xi (1+|rho|) |u|_max + kappa),
# one per scheme, both MEASURED as the edge where |phi(u - i/2)| first exceeds one
# over u in [0, 1200] (it may never exceed one for a correct solve):
#   etdrk4   stable at 2.65, blows up at 4.0                     -> 2.0 used
#   exptrap  stable at z = 10.3 .. 14.7, blows up at 24.6 .. 30.3 -> 12.0 used
# measured at H = 0.02 and 0.12, maturities 30 d to 1 y.
C_STAB = {"etdrk4": 2.0, "exptrap": 12.0}


class StabilityError(ValueError):
    """
    Raised, never warned: a Riccati solve was requested with a step beyond the
    measured stability edge for its |u| range, or a solve returned |phi(u)| > 1
    on the strip -1 <= Im u <= 0, where no martingale model can (Session G).

    Before this, an under-stepped solve was only caught inside lewis_prices,
    which refines it. Any other caller -- char_func with explicit steps, a
    Carr-Madan pricer, a study script -- got the number: 120 exptrap steps at
    u = 1200, H = 0.12, 90 days returned |phi| = 1.21e36 without complaint.
    """


def max_stable_step(p, u_abs_max, scheme="etdrk4", c_stab=None):
    """Largest step h with h^alpha / Gamma(1+alpha) * (xi (1+|rho|) |u| + kappa) <= C_STAB."""
    cs = C_STAB[scheme] if c_stab is None else c_stab
    alpha = p.H + 0.5
    G = p.xi * (1.0 + abs(p.rho)) * float(u_abs_max) + p.kappa
    return (cs * math.gamma(1.0 + alpha) / G) ** (1.0 / alpha)


def stability_steps(tau, u_abs_max, p, grade=1.0, min_steps=24, c_stab=None,
                    scheme="etdrk4"):
    """
    Number of steps for a stable solve over the u range requested.

    The linear part -x_i psi_i is integrated exactly in both schemes, so the
    stiff fast factors cost nothing. What limits the step is the treatment of
    F(Psi): the Riccati Jacobian |dF/dPsi| grows like xi |u| (1 + |rho|) +
    kappa, and the kernel feeds it back over one step with weight
    int_0^h K(s) ds = h^alpha / Gamma(1+alpha). Stability then requires

        h^alpha / Gamma(1+alpha) * G  <  C_STAB[scheme],   G = xi (1+|rho|) |u|_max + kappa.

    For H = 1/2 and ETDRK4 this is the ordinary RK4 limit h G < 2. The rough
    kernel makes it much stricter at the same u, because h^alpha >> h for
    small h.

    The implicit scheme is NOT unconditionally stable. An earlier version of
    this module said it was, on the strength of tests that only probed u up to
    each maturity's pricing range; driven to u = 1200 at H = 0.12 and 90 days,
    120 implicit steps return |phi(u - i/2)| = 1e36 where the true value is
    below one. Its constant is six times better than ETDRK4's, which at
    alpha = 0.62 is eighteen times fewer steps -- not infinitely many.
    """
    h_max = max_stable_step(p, u_abs_max, scheme, c_stab)
    return int(max(min_steps, math.ceil(grade * tau / h_max)))


def u_max_for(p, tau, tol=1e-10, safety=1.3):
    """
    A starting estimate of where |cf(u)| has decayed below `tol`, from the two
    known regimes of a Heston-type characteristic function.

    Small u: the Gaussian envelope exp(-V u^2 / 2), V ~ integrated variance.
    Large u: these cfs decay EXPONENTIALLY, not like a Gaussian -- for
    |u| >> 1/(xi tau) the Riccati solution is D ~ -u sqrt(1-rho^2)/xi, so
    |cf| ~ exp(-u sqrt(1-rho^2) (v0 + kappa theta tau) / xi). Truncating from
    the Gaussian estimate alone gave u_max = 53 at one year where the tail is
    still 5e-05. The larger of the two, times `safety`.

    It is only a start. Measured, the ROUGH cf's tail is fatter than Heston's
    at short maturities -- at 7 days and the u where Heston has decayed to
    1e-18, rough Heston is still at 1e-9 -- so `call_prices` checks the tail
    it actually computed and extends the range when it is not below `tol`.
    Used instead of the pricer's doubling probe because for this model every
    probe is a full Riccati solve.
    """
    V = 0.5 * min(p.v0, p.theta) * tau
    u_gauss = math.sqrt(2.0 * math.log(1.0 / tol) / V)
    rate = math.sqrt(1.0 - p.rho * p.rho) * (p.v0 + p.kappa * p.theta * tau) / p.xi
    u_exp = math.log(1.0 / tol) / rate
    # Below two weeks the rough tail is fattest relative to the envelope
    # (measured: the check in call_prices had to extend at 2, 3 and 7 days with
    # safety 1.3, never at 1 day or beyond 14). Starting wider there is cheaper
    # than a second solve, since ETDRK4's step count grows with u_max.
    if tau < 14.0 / 365.0:
        safety = max(safety, 1.6)
    return safety * max(u_gauss, u_exp)


def n_panels_for(u_max, k_abs_max, per_unit=1.5, order=64, lo=2, hi=8000):
    """
    Composite Gauss-Legendre panels (64 points each) for the pricer.

    Measured on the Heston cf against a 1e-14 reference, 1e-10 price accuracy
    needs 16 panels at one day (u_max ~ 1200), 6 at a week or a month, 2 at a
    year -- about 0.9 nodes per unit u throughout. per_unit = 1.5 is that with
    a 1.7x margin. The oscillation e^{-iuk} adds u_max |k| / 2 pi periods at
    ~9 nodes each. The pricer's own default (10 per unit u plus 8 u |k|) is
    sized for 1e-14 and costs 3000-8000 nodes per expiry, which no Riccati
    solve here could justify.
    """
    n_nodes = per_unit * u_max + 1.5 * u_max * abs(k_abs_max)
    return int(min(max(math.ceil(n_nodes / order), lo), hi))


AUTO_EXPTRAP_STEPS = 120      # accuracy floor for exptrap + Richardson in 'auto'
# Relative cost per step, measured at N = 20, n_u = 300: ETDRK4 277 us, exptrap
# 203 us. Richardson runs exptrap at M and 2M, i.e. 3M steps.
COST_PER_STEP = {"etdrk4": 1.36, "exptrap": 1.0}


def log_char_func(u, tau, p, N=N_DEFAULT, steps=None, steps_mult=1.0, grade=1.0,
                  scheme="etdrk4", c_stab=None, check_stability=True, eta_N=1.0e5):
    """
    log cf(u) for the lifted rough Heston, from the N-factor Riccati system

        psi_i' = -x_i psi_i + F(Psi),      Psi = sum_i w_i psi_i,     psi_i(0) = 0
        phi'   = kappa theta Psi + v0 F(Psi),                         phi(0)   = 0
        F(P)   = -1/2 (u^2 + i u) + (i u rho xi - kappa) P + 1/2 xi^2 P^2

    with cf = exp(phi(tau)). Derivation: write E[e^{iuX_T} | F_t] = exp(iu X_t
    + phi(T-t) + sum_i psi_i(T-t) U^i_t), apply Ito, substitute V = V_0 +
    sum_i w_i U^i and collect the constant and the U^i terms. For N = 1, x = 0
    this is Heston's Riccati pair C' = kappa theta D, D' = F(D) exactly. In the
    limit N -> infinity, Psi = int_0^t K(t-s) F(Psi(s)) ds, i.e. the fractional
    Riccati equation D^alpha Psi = F(Psi) of El Euch-Rosenbaum, and phi(T) =
    kappa theta I^1 Psi(T) + v0 I^{1-alpha} Psi(T), their formula.

    Two schemes, both exact on the linear part (variation of constants with
    e^{-x_i h}, so the fast factors are never stiff):

    'etdrk4'   Cox-Matthews exponential RK4. Fourth order (measured 3.99-4.05
               against closed-form Heston). The explicit stages are stable only
               below a step set by |u| (stability_steps), which for a rough
               kernel is ten times stricter than for Heston.
    'exptrap'  Exponential trapezoidal rule with the nonlinear term IMPLICIT.
               Because F is quadratic, the implicit equation for Psi^{n+1} is a
               scalar quadratic per u with a closed-form root, and a step costs
               about three quarters of an ETDRK4 step. Its stability constant is
               six times ETDRK4's but it is NOT unconditionally stable -- see
               stability_steps for the measurement that corrected that claim.
               Second order; use it with Richardson (see char_func).

    The only coupling between factors is through the scalar Psi, so stages are
    never formed factor by factor: each needs a weighted sum of the state (two
    matvecs per step) and the update is one elementwise pass plus a rank-3
    product. The phi-functions depend only on x_i h, computed once per mesh.

    check_stability (default True) is a hard precondition: if the largest step
    of the mesh exceeds max_stable_step for this |u| range and scheme, the solve
    is refused with StabilityError before any work is done. Only convergence and
    blow-up studies that probe beyond the edge on purpose turn it off.
    """
    u = np.atleast_1d(np.asarray(u, dtype=complex))
    v0, kappa, theta, xi, rho, H = p.as_tuple()
    w, x = lift_nodes(H, N, eta_N=eta_N)
    if steps is None:
        u_abs = float(np.max(np.abs(u)))
        if scheme == "etdrk4":
            steps = stability_steps(tau, u_abs, p, grade, c_stab=c_stab, scheme="etdrk4")
        else:
            steps = max(AUTO_EXPTRAP_STEPS,
                        stability_steps(tau, u_abs, p, grade, c_stab=c_stab, scheme="exptrap"))
    M = int(max(1, round(steps * steps_mult)))

    iu = 1j * u
    b0 = -0.5 * (u * u + iu)
    lin = iu * rho * xi - kappa
    q = 0.5 * xi * xi
    kt = kappa * theta

    def F(P):
        return b0 + lin * P + q * P * P

    mesh = tau * (np.arange(M + 1) / M) ** grade
    hs = np.diff(mesh)
    if check_stability and scheme in C_STAB:
        u_abs = float(np.max(np.abs(u)))
        h_max = max_stable_step(p, u_abs, scheme, c_stab)
        if float(hs.max()) > h_max * (1.0 + 1e-9):
            raise StabilityError(
                f"{scheme}: step {hs.max():.3e} y exceeds the stable step {h_max:.3e} y at "
                f"|u| <= {u_abs:.0f} (H={H:.3f}, xi={xi:.3f}, rho={rho:.3f}, kappa={kappa:.3f}); "
                f"needs >= {stability_steps(tau, u_abs, p, grade, min_steps=1, c_stab=c_stab, scheme=scheme)}"
                f" steps, got {M}")
    c = -np.outer(hs, x)                     # (M, N)
    E = np.exp(c)
    n_u = len(u)
    psi = np.zeros((len(w), n_u), dtype=complex)
    phi = np.zeros(n_u, dtype=complex)
    Psi = np.zeros(n_u, dtype=complex)

    if scheme == "etdrk4":
        E2 = np.exp(0.5 * c)
        f1h = _phi123(0.5 * c)[0]
        p1, p2, p3 = _phi123(c)
        A1 = hs[:, None] * (p1 - 3.0 * p2 + 4.0 * p3)
        A2 = hs[:, None] * (2.0 * p2 - 4.0 * p3)
        A3 = hs[:, None] * (-p2 + 4.0 * p3)
        s_h = 0.5 * hs * (f1h @ w)
        s_h2 = 0.5 * hs * ((E2 * f1h) @ w)
        wE = E * w[None, :]
        wE2 = E2 * w[None, :]
        wA1, wA2, wA3 = A1 @ w, A2 @ w, A3 @ w
        # The two products in this loop are chosen by measurement, per size.
        # Contraction w @ psi: OpenBLAS pays ~80 us of thread dispatch per call
        # whatever the size, so einsum wins below ~1000 u-nodes (39 vs 79 us at
        # 512) and matmul above (85 vs 143 us at 2048). Rank-3 update: one
        # real @ complex product on a pre-stacked (N, 3) matrix, 23 us at 512 --
        # broadcasting A1[:, None] * Nu + ... was tried and cost 438 us.
        use_einsum = n_u <= 1024
        Astack = np.stack([A1, A2, A3], axis=2)          # (M, N, 3)
        Vbuf = np.empty((3, n_u), dtype=complex)
        for j in range(M):
            h = hs[j]
            if use_einsum:
                S2 = np.einsum("i,ij->j", wE2[j], psi)
                S1 = np.einsum("i,ij->j", wE[j], psi)
            else:
                S2 = wE2[j] @ psi
                S1 = wE[j] @ psi
            Nu = F(Psi)
            Gu = kt * Psi + v0 * Nu
            Pa = S2 + s_h[j] * Nu
            Na = F(Pa)
            Pb = S2 + s_h[j] * Na
            Nb = F(Pb)
            Pc = S1 + s_h2[j] * Nu + s_h[j] * (2.0 * Nb - Nu)
            Nc = F(Pc)
            Nab = Na + Nb
            psi *= E[j][:, None]
            Vbuf[0] = Nu
            Vbuf[1] = Nab
            Vbuf[2] = Nc
            psi += Astack[j] @ Vbuf
            Psi = S1 + wA1[j] * Nu + wA2[j] * Nab + wA3[j] * Nc
            phi += (h / 6.0) * (Gu + 2.0 * (kt * (Pa + Pb) + v0 * Nab)
                                + kt * Pc + v0 * Nc)
        return phi.reshape(np.shape(u)) if np.ndim(u) else phi

    if scheme != "exptrap":
        raise ValueError(f"unknown scheme {scheme!r}")

    # Exponential trapezoidal rule, implicit in F:
    #   psi_i^{n+1} = E_i psi_i^n + h (phi1_i - phi2_i) F^n + h phi2_i F^{n+1}
    # Summing with weights w_i gives Psi^{n+1} = S1 + a0 F^n + a1 F(Psi^{n+1}),
    # a scalar quadratic in Psi^{n+1}: a1 q P^2 + (a1 lin - 1) P + (S1 + a0 F^n
    # + a1 b0) = 0. The root continuous with the explicit predictor is taken.
    p1, p2, _ = _phi123(c)
    B0 = hs[:, None] * (p1 - p2)             # (M, N) weights on F^n
    B1 = hs[:, None] * p2                    # weights on F^{n+1}
    wE = E * w[None, :]
    a0 = B0 @ w
    a1 = B1 @ w
    Fn = F(Psi)
    use_einsum = n_u <= 1024          # see the ETDRK4 loop for the measurement
    for j in range(M):
        h = hs[j]
        S1 = np.einsum("i,ij->j", wE[j], psi) if use_einsum else wE[j] @ psi
        A = a1[j] * q
        Bq = a1[j] * lin - 1.0
        Cq = S1 + a0[j] * Fn + a1[j] * b0
        pred = S1 + (a0[j] + a1[j]) * Fn
        disc = np.sqrt(Bq * Bq - 4.0 * A * Cq)
        r1 = (-Bq + disc) / (2.0 * A)
        r2 = (-Bq - disc) / (2.0 * A)
        Pn1 = np.where(np.abs(r1 - pred) < np.abs(r2 - pred), r1, r2)
        Fn1 = F(Pn1)
        psi *= E[j][:, None]
        psi += np.outer(B0[j], Fn) + np.outer(B1[j], Fn1)
        # trapezoid on phi' = kt Psi + v0 F(Psi)
        phi += 0.5 * h * ((kt * Psi + v0 * Fn) + (kt * Pn1 + v0 * Fn1))
        Psi, Fn = Pn1, Fn1
    return phi.reshape(np.shape(u)) if np.ndim(u) else phi


def choose_scheme(tau, u_abs_max, p, grade=1.0):
    """
    ('etdrk4', M) or ('exptrap', M) -- whichever is cheaper at its own stable
    step count, with exptrap carrying Richardson (3M steps) and an accuracy
    floor of AUTO_EXPTRAP_STEPS.
    """
    m_e = stability_steps(tau, u_abs_max, p, grade, scheme="etdrk4")
    m_t = max(AUTO_EXPTRAP_STEPS, stability_steps(tau, u_abs_max, p, grade, scheme="exptrap"))
    if COST_PER_STEP["etdrk4"] * m_e <= COST_PER_STEP["exptrap"] * 3 * m_t:
        return "etdrk4", m_e
    return "exptrap", m_t


def char_func(u, tau, p, N=N_DEFAULT, scheme="auto", richardson=False, check_invariant=True, **kw):
    """
    cf(u) = E[exp(i u X_tau)] for the lifted rough Heston.

    scheme 'auto' takes whichever of ETDRK4 and exptrap-with-Richardson is
    cheaper at its own stable step count for this u range (choose_scheme):
    ETDRK4 at short maturities, where its step count is small, exptrap at long
    ones, where ETDRK4's step is pinned by |u|.

    `richardson=True` with scheme='exptrap' runs M and 2M steps and combines
    (4 cf_2M - cf_M) / 3, which cancels the leading O(h^2) error: measured
    against closed-form Heston the order goes from 2.00 to 4.00. The two solves
    also give an error estimate for free, which the Lewis pricer uses.

    check_invariant (default True) is a hard postcondition: for every u on the
    strip -1 <= Im u <= 0, |cf(u)| <= E[S^(-Im u)] <= 1 by Jensen, for ANY
    martingale model and any parameters. A value above 1 + PHI_BOUND_TOL there is
    a broken solve, and raises StabilityError instead of being returned.
    """
    if scheme == "auto":
        scheme, m = choose_scheme(tau, float(np.max(np.abs(u))), p, kw.get("grade", 1.0))
        if scheme == "exptrap":
            richardson = True
        kw.setdefault("steps", m)
    if richardson and scheme == "exptrap":
        steps = kw.pop("steps", None)
        if steps is None:
            steps = max(AUTO_EXPTRAP_STEPS,
                        stability_steps(tau, float(np.max(np.abs(u))), p,
                                        kw.get("grade", 1.0), scheme="exptrap"))
        a = log_char_func(u, tau, p, N=N, steps=steps, scheme=scheme, **kw)
        b = log_char_func(u, tau, p, N=N, steps=2 * steps, scheme=scheme, **kw)
        out = (4.0 * _safe_exp(b) - _safe_exp(a)) / 3.0
    else:
        out = _safe_exp(log_char_func(u, tau, p, N=N, scheme=scheme, **kw))
    if check_invariant:
        ua = np.atleast_1d(np.asarray(u, dtype=complex))
        strip = (ua.imag <= 1e-12) & (ua.imag >= -1.0 - 1e-12)
        if np.any(strip):
            worst = float(np.max(np.abs(np.atleast_1d(out)[strip])))
            if worst > 1.0 + PHI_BOUND_TOL:
                raise StabilityError(
                    f"|cf| = {worst:.3e} > 1 on the strip -1 <= Im u <= 0 at tau = {tau:.4f} "
                    f"(H={p.H:.3f}, xi={p.xi:.3f}): a broken solve, not a model value")
    if np.ndim(u) == 0:
        return out.reshape(())[()]
    return out


def _safe_exp(z):
    """exp of a complex array without overflow: a blown-up solve returns a huge
    finite number the checks can see, never inf or nan."""
    z = np.asarray(z, dtype=complex)
    return np.exp(np.clip(z.real, -745.0, 700.0) + 1j * np.nan_to_num(z.imag))


def cf_factory(p, tau, N=N_DEFAULT, **kw):
    """
    The calibrator's interface: params, tau -> callable cf(u). The callable
    carries `.params` so the rough pricers can size their own quadrature, and
    accepts keyword overrides (scheme, steps, ...) so a pricer can pin them.
    """
    def cf(u, **override):
        return char_func(u, tau, p, N=N, **{**kw, **override})
    cf.params = p
    cf.tau = tau
    cf.N = N
    return cf


def pricer_settings(p, tau, k_abs_max, tol=1e-10, per_unit=1.5):
    """
    (u_max, n_panels) for pricing.fourier.carr_madan_call / lewis_call at this
    maturity, chosen analytically so the pricer's doubling probe -- a Riccati
    solve per probe -- is never run. Pass as u_max=..., n_panels=... .
    """
    um = u_max_for(p, tau, tol)
    return um, n_panels_for(um, k_abs_max, per_unit)


def call_prices(ks, tau, p, tol=1e-9, N=N_DEFAULT, steps_mult=1.0, **kw):
    """
    Undiscounted call values (units of F) for a strip of log-moneyness at one
    maturity: (prices, info). The single entry point for pricing this model;
    see lewis_prices for the contour, the checks and the refinement.

    An earlier version priced on the Carr-Madan contour with alpha = 1.5. That
    needs E[S^2.5] < inf, a calibration walked to parameters where it is not,
    and the pricer returned 1e18 there. Lewis needs only E[S^(1/2)].
    """
    return lewis_prices(ks, tau, p, tol=tol, N=N, steps_mult=steps_mult, **kw)


PHI_BOUND_TOL = 1e-6        # |phi(u - i/2)| may exceed 1 by at most this
ETDRK4_MARGIN = 1.5         # steps above the stability edge, for accuracy at 1 day


def lewis_grid(u_max, k_abs_max, order=64, first=2.0, periods=6.0):
    """
    Gauss-Legendre nodes on [0, u_max] in panels that DOUBLE in width, capped so
    no panel holds more than `periods` turns of e^{-iuk}.

    The Lewis integrand phi(u - i/2) / (u^2 + 1/4) carries nearly all of its
    mass at small u and is smooth and tiny far out, so a uniform grid spends
    most of its nodes where nothing is. Measured against the uniform composite
    rule at the same u_max: 2176 -> 640 nodes at one day, 4608 -> 704 at the
    runaway parameters, IV differences at the level of the cf's own error
    (< 2e-3 vol points away from the one-day wings).
    """
    import pricing.fourier as fo

    x, w = fo._leggauss(int(order))
    cap = periods * 2.0 * math.pi / max(float(k_abs_max), 1e-3)
    edges = [0.0]
    width = float(first)
    while edges[-1] < u_max:
        edges.append(min(edges[-1] + min(width, cap), float(u_max)))
        width *= 2.0
    e = np.asarray(edges)
    lo, hi = e[:-1], e[1:]
    half = 0.5 * (hi - lo)
    nodes = (half[:, None] * (x[None, :] + 1.0) + lo[:, None]).ravel()
    weights = (half[:, None] * w[None, :]).ravel()
    return nodes, weights


def _lewis_solve(ks, tau, p, u_max, n_panels, N, steps, tol_price, steps_mult=1.0):
    """
    One Lewis pricing pass on a fixed grid with a-posteriori error control.

    Returns (prices, info). The cf is computed at M and 2M exptrap steps (or
    ETDRK4 at M and 2M when that is cheaper) so every pass carries its own error
    estimate, and two checks decide whether the pass is trusted:

      invariant   |phi(u - i/2)| <= E[S^(1/2)] <= 1 for ANY martingale model
                  (Jensen). A solve that breaks it is wrong, full stop.
      error       the Lewis integral of |phi_2M - phi_M| / (u^2 + 1/4) / 3 -- the
                  error of the 2M solve -- below tol_price. That is an
                  INSTABILITY detector, not an accuracy target: the Richardson
                  value returned is orders of magnitude better than the 2M solve
                  (measured 1.2e-6 -> 1.9e-11 at H = 1/2), while an unstable
                  solve differs between M and 2M at O(1). A 1e-8 target was
                  tried first and forced 3-4 refinements at the TRUE parameters,
                  a 15x slowdown for accuracy nothing downstream can see.
    """
    import pricing.fourier as fo

    v, w = lewis_grid(u_max, float(np.max(np.abs(ks))) if np.size(ks) else 0.0)
    z = v - 0.5j
    kern = 1.0 / (v * v + 0.25)
    scheme, m_stab = choose_scheme(tau, u_max, p)
    if scheme == "exptrap":
        # Richardson needs M and 2M anyway, so the error estimate is free. The
        # extrapolated value's error is far below |phi_2M - phi_M| / 3; using
        # the latter is deliberately conservative.
        M = int(math.ceil(steps_mult * max(int(steps), m_stab)))
        ea = _safe_exp(log_char_func(z, tau, p, N=N, steps=M, scheme=scheme))
        eb = _safe_exp(log_char_func(z, tau, p, N=N, steps=2 * M, scheme=scheme))
        phi = (4.0 * eb - ea) / 3.0
        err_price = float(np.sum(w * np.abs(eb - ea) * kern)) / (3.0 * math.pi)
        modmax = float(max(np.max(np.abs(eb)), np.max(np.abs(phi))))
    else:
        # ETDRK4 at its stability step count was correct in every measured case
        # (H = 0.02 and 0.12, u to 1200, 30 d to 1 y); one solve plus the
        # invariant. `steps` only ever raises M, when a refinement asks for it.
        M = max(int(math.ceil(steps_mult * ETDRK4_MARGIN * m_stab)),
                int(steps) if int(steps) > AUTO_EXPTRAP_STEPS else 0)
        phi = _safe_exp(log_char_func(z, tau, p, N=N, steps=M, scheme=scheme))
        err_price = 0.0
        modmax = float(np.max(np.abs(phi)))
    # The truncated mass is bounded by the integrand AT the cut-off. With
    # doubling panels the last panel starts at u_max / 2, so reading all 64 of
    # its nodes -- as the uniform-grid version did -- sees values from halfway
    # in and extends the range for nothing (measured: 6 needless extensions
    # across ten expiries at the true parameters).
    tail = float(np.max(np.abs(phi[-8:]) * kern[-8:]))
    dens = phi * kern
    ks = np.atleast_1d(np.asarray(ks, dtype=float))
    prices = 1.0 - np.exp(0.5 * ks) / math.pi * fo._real_transform(ks, v, dens, w)
    ok = (np.all(np.isfinite(prices)) and modmax <= 1.0 + PHI_BOUND_TOL
          and err_price <= tol_price)
    return prices, {"scheme": scheme, "steps": M, "err_price": err_price,
                    "phi_max": modmax, "tail": tail, "ok": bool(ok),
                    "u_max": u_max, "n_panels": n_panels, "n_nodes": len(v)}


def lewis_prices(ks, tau, p, tol=1e-9, tol_price=1e-6, N=N_DEFAULT, max_refine=3,
                 max_extend=3, u_max=None, steps_mult=1.0):
    """
    Undiscounted call values (units of F) for a strip of log-moneyness, on the
    Lewis contour Im u = -1/2, with the quadrature range and the ODE step count
    both CHECKED rather than trusted. Returns (prices, info).

    Why Lewis and not Carr-Madan for this model. Carr-Madan with damping alpha
    evaluates phi on Im u = -(1 + alpha), which needs E[S^(1+alpha)] < inf; for
    rough and vanilla Heston that moment explodes in finite time when vol-of-vol
    is high relative to mean reversion and correlation, and a calibration WILL
    walk there -- it did, to (xi, rho, kappa) = (0.63, -0.56, 0.19), where the
    Heston discriminant for p = 2.5 is -0.36. Lewis needs only E[S^(1/2)], which
    is finite for every martingale model, and it makes |phi| <= 1 a hard check.
    The two agree to 2.5e-13 in price where both are valid.

    Refinement: if the invariant or the step-halving error fails, the step count
    doubles (up to max_refine times); if the u-tail has not decayed below `tol`,
    the range grows 1.5x (up to max_extend times). If a pass still fails, the
    prices come back NaN and info['ok'] is False: the caller is told the model
    could not be priced there, instead of being handed a number.
    """
    ks = np.atleast_1d(np.asarray(ks, dtype=float))
    k_abs = float(np.max(np.abs(ks))) if ks.size else 0.0
    um = u_max_for(p, tau, tol) if u_max is None else float(u_max)
    steps = AUTO_EXPTRAP_STEPS
    refined = extended = 0
    while True:
        prices, info = _lewis_solve(ks, tau, p, um, n_panels_for(um, k_abs), N, steps, tol_price,
                                    steps_mult)
        bad_solve = (info["phi_max"] > 1.0 + PHI_BOUND_TOL or info["err_price"] > tol_price
                     or not np.all(np.isfinite(prices)))
        if bad_solve and refined < max_refine:
            steps = 2 * info["steps"]
            refined += 1
            continue
        if not bad_solve and info["tail"] > tol and extended < max_extend:
            um *= 1.5
            extended += 1
            continue
        break
    info.update(refined=refined, extended=extended)
    if not info["ok"]:
        prices = np.full(ks.shape, np.nan)
    return prices, info


class RoughPricer:
    """
    The pricer to hand a CALIBRATION: the signature of pricing.fourier's pricers
    (ks, tau, cf, tol) with `cf` built by rough_heston.cf_factory, pricing on the
    Lewis contour with every solve checked (see lewis_prices).

    The u-range for each maturity is sized once, on first use, with a margin,
    and then frozen. Speed: sizing means a tail check, and a failed check means
    a second Riccati solve. Smoothness: an optimiser differentiates the
    objective by finite differences, and a grid that re-sizes itself as the
    parameters move puts a kink under every step. The tail is still checked on
    every call and the frozen range grows if the parameters have wandered
    somewhere fatter-tailed, so the freeze is a default, not a promise.
    `stats` records what happened, including how many strips came back
    unpriceable -- which the objective turns into a penalty, never a zero.
    """

    def __init__(self, tol=1e-9, tol_price=1e-6, N=N_DEFAULT, widen=1.25, **_ignored):
        self.tol, self.tol_price, self.N, self.widen = tol, tol_price, N, widen
        self.grids = {}
        self.stats = {"solves": 0, "refined": 0, "extended": 0, "failed_strips": 0}

    def __call__(self, ks, tau, cf, tol=None):
        p = getattr(cf, "params", None)
        if p is None:
            raise TypeError("RoughPricer needs a cf built by rough_heston.cf_factory")
        ks = np.atleast_1d(np.asarray(ks, dtype=float))
        key = round(float(tau), 12)
        if key not in self.grids:
            self.grids[key] = self.widen * u_max_for(p, tau, self.tol)
        prices, info = lewis_prices(ks, tau, p, tol=self.tol, tol_price=self.tol_price,
                                    N=getattr(cf, "N", self.N), u_max=self.grids[key])
        self.grids[key] = max(self.grids[key], info["u_max"])
        self.stats["solves"] += 1 + info["refined"] + info["extended"]
        self.stats["refined"] += info["refined"]
        self.stats["extended"] += info["extended"]
        if not info["ok"]:
            self.stats["failed_strips"] += 1
        return prices if prices.size > 1 else float(prices[0])
