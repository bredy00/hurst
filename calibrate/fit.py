"""
Levenberg-Marquardt calibration, numpy only.

Two design choices worth stating.

**Bounded, unconstrained parameters.** The optimiser never sees v0, kappa,
theta, xi or rho directly. Each is the image of a logistic on a finite box, so
the bounds cannot be violated, no projection or penalty is needed, and the search
directions are well scaled -- kappa ~ 2 and v0 ~ 0.04 differ by two orders of
magnitude in the raw parameterisation, which is exactly the sort of thing that
makes a Jacobian ill-conditioned. See BOUNDS for why the box is finite rather
than a bare log or tanh map.

**Generic over the model.** Nothing here knows what Heston is. A model supplies
a characteristic-function factory and a `Transform`, and the same driver fits it.
That is the point: Session E's rough Heston must be calibrated by this identical
code for its comparison against Phase 1 to mean anything.
"""

import math
from dataclasses import dataclass

import numpy as np

from calibrate.objective import CALIB_TOL, residuals, rmse_vol
from models.heston import HestonParams, char_func
import models.rough_heston as rh


@dataclass(frozen=True)
class Transform:
    """Maps unconstrained optimiser coordinates to model parameters and back."""
    names: tuple
    to_x: callable        # params -> unconstrained vector
    from_x: callable      # unconstrained vector -> params


# Every parameter is mapped through a bounded logistic. The two obvious
# alternatives were both tried and both fail.
#
# A bare tanh for rho SATURATES: math.tanh(40) is exactly 1.0 in float64, so the
# optimiser can hand the model rho = +-1, which is outside its domain -- there
# sqrt(1 - rho^2) is zero and the simulator degenerates.
#
# A bare log for the positive parameters is unbounded above, and Levenberg-
# Marquardt does run away with it:
# asked to fit a ROUGH surface -- a shape Heston structurally cannot make at the
# short end -- the optimiser pushes kappa outward chasing a fit that does not
# exist. Two things then break. math.exp overflows past x ~ 709, which is merely
# a crash. Worse, well before that the characteristic function stops decaying, so
# `_auto_u_max` runs to its cap and a single objective evaluation needs half a
# million quadrature nodes; the fit does not fail, it just never finishes.
#
# Bounding each parameter to a financially sensible box fixes both, keeps the
# optimiser unconstrained in x, and costs nothing: no real calibration wants
# kappa = 1e13.
BOUNDS = {
    "v0":    (1e-6, 4.0),      # variance, so vol up to 200%
    "kappa": (1e-3, 50.0),     # 1/kappa from 20 years down to a week
    "theta": (1e-6, 4.0),
    "xi":    (1e-3, 10.0),
    "rho":   (-0.9999, 0.9999),
}


def _to_x(v, lo, hi, eps=1e-12):
    """logit of the position within [lo, hi]."""
    z = (float(v) - lo) / (hi - lo)
    z = min(max(z, eps), 1.0 - eps)
    return math.log(z / (1.0 - z))


def _from_x(x, lo, hi):
    """Bounded logistic. Saturates gracefully instead of overflowing."""
    x = float(x)
    # exp(-x) overflows for very negative x; the limit is simply the lower bound
    if x < -700.0:
        return lo
    if x > 700.0:
        return hi
    return lo + (hi - lo) / (1.0 + math.exp(-x))


def _heston_to_x(p):
    return np.array([_to_x(getattr(p, n), *BOUNDS[n]) for n in HestonParams.NAMES])


def _heston_from_x(x):
    vals = {n: _from_x(x[i], *BOUNDS[n]) for i, n in enumerate(HestonParams.NAMES)}
    return HestonParams(**vals)


HESTON_TRANSFORM = Transform(names=HestonParams.NAMES,
                             to_x=_heston_to_x, from_x=_heston_from_x)


def heston_cf_factory(p, tau):
    return lambda u: char_func(u, tau, p)


def _jacobian(fun, x, r0, step=1e-5):
    """
    Forward-difference Jacobian.

    Forward rather than central: central doubles the cost of the most expensive
    thing in the loop -- a full surface repricing per parameter -- and LM is
    tolerant of the extra O(step) error because the damping already handles an
    imperfect step direction. In unconstrained coordinates every x is O(1), so a
    single relative step size is appropriate throughout.
    """
    n, m = len(x), len(r0)
    J = np.empty((m, n))
    for j in range(n):
        h = step * max(abs(x[j]), 1.0)
        xp = x.copy()
        xp[j] += h
        J[:, j] = (fun(xp) - r0) / h
    return J


def levenberg_marquardt(fun, x0, max_iter=120, ftol=1e-10, xtol=1e-12,
                        lam0=1e-3, lam_up=4.0, lam_down=3.0, step=1e-5,
                        max_inner=40, max_seconds=None, callback=None):
    """
    Minimise ||fun(x)||^2. Returns (x, cost, info).

    Damping is scaled by diag(J^T J) rather than the identity (Marquardt's own
    refinement): with parameters of very different curvature, identity damping
    penalises every direction equally and stalls along the flat ones.

    `max_seconds` is not a nicety. When the model cannot represent the target --
    fitting vanilla Heston to a rough surface, which is exactly what Phase 1 is
    for -- the optimiser does not diverge and does not converge; it grinds, with
    lambda climbing through 1e8 while the cost improves in the sixth decimal.
    Measured at 225 ms per objective evaluation on a 91-quote surface, the
    default iteration cap alone would run for thirteen minutes to buy nothing.
    """
    import time as _time

    t_start = _time.perf_counter()
    x = np.array(x0, dtype=float)
    r = np.asarray(fun(x), dtype=float)
    cost = float(r @ r)
    lam = float(lam0)
    n_fev, it, reason = 1, 0, "max_iter"

    for it in range(1, max_iter + 1):
        if max_seconds is not None and _time.perf_counter() - t_start > max_seconds:
            reason = "timeout"
            break
        J = _jacobian(fun, x, r, step)
        n_fev += len(x)
        JtJ = J.T @ J
        Jtr = J.T @ r
        scale = np.maximum(np.diag(JtJ), 1e-12)

        if float(np.max(np.abs(Jtr))) < 1e-14:
            reason = "gradient"
            break

        accepted = False
        for _ in range(max_inner):
            try:
                dx = np.linalg.solve(JtJ + lam * np.diag(scale), -Jtr)
            except np.linalg.LinAlgError:
                lam *= lam_up
                continue
            xn = x + dx
            rn = np.asarray(fun(xn), dtype=float)
            n_fev += 1
            cn = float(rn @ rn)
            if cn < cost:
                gain = cost - cn
                x, r, cost = xn, rn, cn
                lam = max(lam / lam_down, 1e-12)
                accepted = True
                break
            lam *= lam_up
            if lam > 1e14:
                break

        if callback is not None:
            callback(it, x, cost, lam)
        if not accepted:
            reason = "no_descent"
            break
        if gain <= ftol * max(cost, 1e-300):
            reason = "ftol"
            break
        if float(np.max(np.abs(dx))) <= xtol:
            reason = "xtol"
            break

    return x, cost, {"iterations": it, "n_fev": n_fev, "lambda": lam,
                     "reason": reason,
                     "seconds": _time.perf_counter() - t_start}


DEFAULT_STARTS = (
    HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.5),
    HestonParams(v0=0.09, kappa=0.8, theta=0.06, xi=1.0, rho=-0.7),
    HestonParams(v0=0.02, kappa=5.0, theta=0.03, xi=0.3, rho=-0.3),
    HestonParams(v0=0.06, kappa=1.5, theta=0.09, xi=0.7, rho=-0.85),
)


def prior_residuals(params, transform, prior, prior_weight):
    """
    Tikhonov rows: sqrt(lambda_j) * (p_j - prior_j) / |prior_j| for each named
    parameter in `prior`. Returns an empty array when there is no prior.

    Relative deviation, not absolute, so kappa ~ 2 and v0 ~ 0.04 are penalised on
    the same footing. Units: one data residual is (model - market vol) / vol
    error, i.e. "standard errors of one quote". A weight lambda therefore means
    "a 100% deviation from the prior costs as much as lambda quotes each off by
    one standard error". With 130 quotes, lambda ~ 10 is a light hand and
    lambda ~ 1000 is a pin.
    """
    if not prior:
        return np.zeros(0)
    if prior_weight is None:
        prior_weight = 1.0
    rows = []
    for name, target in prior.items():
        if name not in transform.names:
            raise KeyError(f"prior on unknown parameter {name!r}")
        lam = (prior_weight[name] if isinstance(prior_weight, dict)
               else float(prior_weight))
        if lam <= 0.0:
            continue
        scale = abs(float(target)) if abs(float(target)) > 1e-12 else 1.0
        rows.append(math.sqrt(lam) * (getattr(params, name) - float(target)) / scale)
    return np.asarray(rows, dtype=float)


def calibrate(surface, cf_factory, transform, starts, tol=CALIB_TOL,
              prior=None, prior_weight=None, pricer=None, **lm_kw):
    """
    Fit a model to a surface from several starting points, keep the best.

    Multi-start is not optional for this problem. The Heston objective has
    genuine local minima -- a fit can settle into a low vol-of-vol, high
    mean-reversion basin that matches the level and misses the skew entirely --
    and a single start silently returns whichever basin it happened to land in.
    Four starts spread across the plausible region is cheap insurance.

    `prior` / `prior_weight` add Tikhonov regularisation towards named parameter
    values (see prior_residuals), appended as extra rows of the residual vector
    so the Levenberg-Marquardt machinery is unchanged. This is the honest answer
    to the kappa identifiability finding: a surface that stops before the
    relaxation time 1/kappa cannot determine kappa, so either say so, or supply
    the missing information explicitly -- a historical moving average of fitted
    kappas, or a value pinned from the variance-swap term structure
    (models.heston.kappa_from_variance_swap) -- and report what it cost. The
    result carries `data_cost` and `prior_cost` separately for that reason: a
    prior that is paying a lot is a prior that disagrees with the market.
    """
    pk = {} if pricer is None else {"pricer": pricer}

    def make_fun():
        def fun(x):
            p = transform.from_x(x)
            r, _ = residuals(p, surface, cf_factory, tol=tol, **pk)
            if prior:
                r = np.concatenate([r, prior_residuals(p, transform, prior, prior_weight)])
            return r
        return fun

    fun = make_fun()
    best = None
    trials = []
    for s in starts:
        x0 = transform.to_x(s)
        x, cost, info = levenberg_marquardt(fun, x0, **lm_kw)
        p = transform.from_x(x)
        trials.append({"start": s, "params": p, "cost": cost, **info})
        if best is None or cost < best["cost"]:
            best = {"params": p, "cost": cost, "x": x, **info}

    best["rmse_vol"] = rmse_vol(best["params"], surface, cf_factory, tol=tol, **pk)
    best["trials"] = trials
    best["n_starts"] = len(starts)
    pr = prior_residuals(best["params"], transform, prior, prior_weight) if prior else np.zeros(0)
    best["prior_cost"] = float(pr @ pr)
    best["data_cost"] = best["cost"] - best["prior_cost"]
    # How much the answer depended on where we started -- if this is large, the
    # objective is multi-modal and a single-start fit would have been a coin flip.
    costs = [t["cost"] for t in trials]
    best["cost_spread"] = float(max(costs) / max(min(costs), 1e-300))
    return best


def calibrate_heston(surface, starts=DEFAULT_STARTS, tol=CALIB_TOL, prior=None,
                     prior_weight=None, **lm_kw):
    """Fit vanilla Heston. Feller is reported in the result, never enforced."""
    res = calibrate(surface, heston_cf_factory, HESTON_TRANSFORM, starts,
                    tol=tol, prior=prior, prior_weight=prior_weight, **lm_kw)
    res["feller"] = res["params"].feller
    p = res["params"]
    res["feller_margin"] = 2.0 * p.kappa * p.theta - p.xi * p.xi
    return res


# ------------------------------------------------------------ rough Heston
# H is bounded away from 0 (the kernel t^(H-1/2) is not integrable-squared
# there and the lift's node range was built for H >= 0.05) and capped at 1/2,
# where the model IS Heston. Everything else shares Heston's box.
ROUGH_BOUNDS = dict(BOUNDS, H=(0.02, 0.5))


def _rough_to_x(p):
    return np.array([_to_x(getattr(p, n), *ROUGH_BOUNDS[n]) for n in rh.RoughHestonParams.NAMES])


def _rough_from_x(x):
    vals = {n: _from_x(x[i], *ROUGH_BOUNDS[n]) for i, n in enumerate(rh.RoughHestonParams.NAMES)}
    return rh.RoughHestonParams(**vals)


ROUGH_TRANSFORM = Transform(names=rh.RoughHestonParams.NAMES,
                            to_x=_rough_to_x, from_x=_rough_from_x)

DEFAULT_ROUGH_STARTS = (
    rh.RoughHestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.5, H=0.15),
    rh.RoughHestonParams(v0=0.09, kappa=0.8, theta=0.06, xi=1.0, rho=-0.7, H=0.10),
)


def blas_single_thread():
    """
    A context that pins OpenBLAS to one thread, or does nothing without threadpoolctl.

    Measured in Session G (benchmark_riccati.py): every threaded BLAS call on the
    Riccati step's small (24 x n_u) products pays 80-130 us of thread dispatch,
    single-threaded matmul is the fastest contraction at every n_u from 16 to
    8192, and a 10-expiry rough objective evaluation drops from 0.717 s to
    0.641 s (median of 7). Pinned once around a whole calibration, so the
    context's own cost is paid once, not per solve.
    """
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1, user_api="blas")
    except ImportError:                                  # pragma: no cover
        import contextlib
        return contextlib.nullcontext()


KERNEL_TOL = 0.01           # max relative kernel error on [1 day, 2 y], strictly below (review target)
REFINE_N = 32               # N = 24 breaches KERNEL_TOL below H ~ 0.08; N = 32 holds on the whole box


def verify_stability(p, surface, N=rh.N_DEFAULT, tail_tol=1e-9):
    """
    Post-fit hard check (Session G): re-price every expiry at the fitted
    parameters with a FRESH Lewis solve -- no frozen grid, no warm refinement --
    and require every strip to pass the |phi(u - i/2)| <= 1 invariant and the
    step-halving check. Every solve inside already ran through the stability
    precondition (rough_heston.StabilityError). A fit that cannot be re-priced
    cleanly is reported as not ok, never returned as if it were.
    """
    worst, failed = 0.0, []
    for tau, idx in surface.by_expiry:
        _, info = rh.lewis_prices(surface.k[idx], tau, p, tol=tail_tol, N=N)
        worst = max(worst, float(info["phi_max"]))
        if not info["ok"]:
            failed.append(float(tau))
    return {"ok": not failed and worst <= 1.0 + rh.PHI_BOUND_TOL, "phi_max": worst,
            "failed_expiries": failed}


def calibrate_rough_heston(surface, starts=DEFAULT_ROUGH_STARTS, tail_tol=1e-9,
                           N=rh.N_DEFAULT, prior=None, prior_weight=None, refine_N=REFINE_N,
                           **lm_kw):
    """
    Fit the lifted rough Heston with the SAME driver as vanilla Heston -- that
    is the point of the shared `char_func` interface; nothing in
    levenberg_marquardt or the objective knows which model it is fitting.

    Three things differ from calibrate_heston, all on the pricing side:

    - the pricer is a RoughPricer, which sizes its quadrature from the model
      (the generic pricer's doubling probe would cost a Riccati solve per probe)
      and freezes the grid per maturity so the objective is smooth;
    - the finite-difference step for the Jacobian is 1e-3 rather than 1e-5:
      the ODE step count changes discretely with the parameters, which puts
      ~1e-8 kinks in the characteristic function, and a 1e-5 step would divide
      that noise by too little;
    - `tail_tol` is the pricer's tail tolerance, not a quadrature tolerance.

    Two hard checks on the answer (Session G), both in the result:
    - kernel_ok: the lift's kernel error at the FITTED H is below KERNEL_TOL.
      N = 24 meets it for H >= 0.08 only; a fit that lands lower is polished
      again from where it stopped with refine_N nodes.
    - stability: every expiry re-priced from scratch at the fitted parameters
      passes the invariant checks (verify_stability).
    res["ok"] is both.

    Cost: about 0.1-0.4 s per objective evaluation for a 3-expiry test
    surface and ~1.5 s for the ten-expiry rough surface, so a full multi-start
    fit is minutes, not seconds. `max_seconds` is honoured.
    """
    lm_kw.setdefault("step", 1e-3)
    pricer = RoughPricerFactory(tail_tol, N)
    with blas_single_thread():
        res = calibrate(surface, pricer.cf_factory, ROUGH_TRANSFORM, starts,
                        tol=tail_tol, prior=prior, prior_weight=prior_weight,
                        pricer=pricer.pricer, **lm_kw)
    p = res["params"]
    res["kernel_error"] = rh.kernel_error(p.H, N)[0]
    res["pricer_stats"] = dict(pricer.pricer.stats)
    res["N"] = N
    if res["kernel_error"] >= KERNEL_TOL and refine_N and refine_N > N:
        polish = dict(lm_kw)
        polish["max_iter"] = min(int(polish.get("max_iter", 20)), 20)
        again = calibrate_rough_heston(surface, starts=(p,), tail_tol=tail_tol, N=refine_N, prior=prior,
                                       prior_weight=prior_weight, refine_N=None, **polish)
        again["refined_from"] = {"N": N, "params": p, "kernel_error": res["kernel_error"]}
        return again
    res["kernel_ok"] = res["kernel_error"] < KERNEL_TOL
    res["stability"] = verify_stability(p, surface, N, tail_tol)
    res["ok"] = bool(res["kernel_ok"] and res["stability"]["ok"])
    return res


class RoughPricerFactory:
    """Binds one RoughPricer and one cf factory (same N) for a calibration run."""

    def __init__(self, tail_tol, N):
        self.N = N
        self.pricer = rh.RoughPricer(tol=tail_tol, N=N)

    def cf_factory(self, p, tau):
        return rh.cf_factory(p, tau, N=self.N)
