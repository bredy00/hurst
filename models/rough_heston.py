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


def mc_call(p, tau, k, n_paths=100_000, n_steps=500, seed=0, N=N_DEFAULT):
    """Undiscounted call value in units of F and its standard error; k = log(K/F)."""
    X = simulate(p, tau, n_paths, n_steps, seed, N)
    pay = np.maximum(np.exp(X) - math.exp(k), 0.0)
    return float(pay.mean()), float(pay.std(ddof=1) / math.sqrt(len(pay)))


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
    cs = C_STAB[scheme] if c_stab is None else c_stab
    alpha = p.H + 0.5
    G = p.xi * (1.0 + abs(p.rho)) * float(u_abs_max) + p.kappa
    h_max = (cs * math.gamma(1.0 + alpha) / G) ** (1.0 / alpha)
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
                  scheme="etdrk4", c_stab=None):
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
    """
    u = np.atleast_1d(np.asarray(u, dtype=complex))
    v0, kappa, theta, xi, rho, H = p.as_tuple()
    w, x = lift_nodes(H, N)
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


def char_func(u, tau, p, N=N_DEFAULT, scheme="auto", richardson=False, **kw):
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
