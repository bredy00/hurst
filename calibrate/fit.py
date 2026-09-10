"""
Levenberg-Marquardt calibration, numpy only.

Two design choices worth stating.

**Unconstrained parameters.** The optimiser never sees v0, kappa, theta, xi or
rho directly. It works on log(positive) and atanh(rho), so the bounds cannot be
violated, no projection or penalty is needed, and the search directions are
well scaled -- kappa ~ 2 and v0 ~ 0.04 differ by two orders of magnitude in the
raw parameterisation, which is exactly the sort of thing that makes a Jacobian
ill-conditioned.

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


@dataclass(frozen=True)
class Transform:
    """Maps unconstrained optimiser coordinates to model parameters and back."""
    names: tuple
    to_x: callable        # params -> unconstrained vector
    from_x: callable      # unconstrained vector -> params


# tanh SATURATES: math.tanh(40) is exactly 1.0 in float64, so a bare tanh map
# can hand the model rho = +-1, which is outside its domain -- sqrt(1 - rho^2)
# is then zero and the simulator degenerates. Scaling by a cap strictly below 1
# makes that unreachable however far the optimiser wanders. The inverse divides
# by the same cap so the round trip stays exact.
# Every parameter is mapped through a bounded logistic rather than a bare log.
#
# A log map is unbounded above, and Levenberg-Marquardt does run away with it:
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


def calibrate(surface, cf_factory, transform, starts, tol=CALIB_TOL, **lm_kw):
    """
    Fit a model to a surface from several starting points, keep the best.

    Multi-start is not optional for this problem. The Heston objective has
    genuine local minima -- a fit can settle into a low vol-of-vol, high
    mean-reversion basin that matches the level and misses the skew entirely --
    and a single start silently returns whichever basin it happened to land in.
    Four starts spread across the plausible region is cheap insurance.
    """
    def make_fun():
        def fun(x):
            p = transform.from_x(x)
            r, _ = residuals(p, surface, cf_factory, tol=tol)
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

    best["rmse_vol"] = rmse_vol(best["params"], surface, cf_factory, tol=tol)
    best["trials"] = trials
    best["n_starts"] = len(starts)
    # How much the answer depended on where we started -- if this is large, the
    # objective is multi-modal and a single-start fit would have been a coin flip.
    costs = [t["cost"] for t in trials]
    best["cost_spread"] = float(max(costs) / max(min(costs), 1e-300))
    return best


def calibrate_heston(surface, starts=DEFAULT_STARTS, tol=CALIB_TOL, **lm_kw):
    """Fit vanilla Heston. Feller is reported in the result, never enforced."""
    res = calibrate(surface, heston_cf_factory, HESTON_TRANSFORM, starts,
                    tol=tol, **lm_kw)
    res["feller"] = res["params"].feller
    p = res["params"]
    res["feller_margin"] = 2.0 * p.kappa * p.theta - p.xi * p.xi
    return res
