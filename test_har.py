"""
Session N -- HAR-RV as a first-class model (models/har.py).

  design       the regressors are the trailing 1-, 5- and 22-day means known at each close,
               and the targets the mean over the NEXT h days -- no look-ahead either way
  recovery     coefficients planted in a simulated HAR process are recovered within 3 SE,
               with Newey-West errors that widen for overlapping multi-day targets
  log-HAR      the lognormal correction is what makes its level forecast unbiased
  qlike        zero at a perfect forecast, positive otherwise, asymmetric as Patton's is

    python test_har.py
"""

import math
import sys

import numpy as np

import models.har as har

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def simulate(n=6000, b=(1e-5, 0.4, 0.3, 0.2), sd=2e-5, seed=0):
    """A HAR process in levels with multiplicative-free noise, kept positive by a floor."""
    rng = np.random.default_rng(seed)
    rv = np.full(n, b[0] / (1 - sum(b[1:])))
    for t in range(22, n):
        rv[t] = max(b[0] + b[1] * rv[t - 1] + b[2] * rv[t - 5:t].mean() + b[3] * rv[t - 22:t].mean()
                    + sd * rng.standard_normal(), 1e-8)
    return rv


def test_design():
    print("\nthe design: what is known when")
    rv = np.arange(1.0, 41.0)
    X = har.regressors(rv)
    check("row t holds the trailing means ending AT day t", X[30, 1] == rv[30] and
          abs(X[30, 2] - rv[26:31].mean()) < 1e-12 and abs(X[30, 3] - rv[9:31].mean()) < 1e-12)
    check("...and is NaN until the 22-day window is full", np.isnan(X[20, 3]) and np.isfinite(X[21, 3]))
    y = har.targets(rv, horizon=5)
    check("the target is the mean of the NEXT h days, never the current one",
          abs(y[10] - rv[11:16].mean()) < 1e-12 and np.isnan(y[-1]))


def test_recovery():
    print("\nrecovering a planted HAR")
    b = (1e-5, 0.4, 0.3, 0.2)
    rv = simulate(b=b)
    m = har.fit(rv, horizon=1)
    z = (m.coef - np.array(b)) / m.se
    check("every coefficient within 3 SE of the planted value", bool(np.all(np.abs(z) < 3)),
          m.describe() + f"; z {np.round(z, 2).tolist()}")
    # a 5-day target's residuals overlap (MA(4)); averaging also shrinks them, so the honest
    # comparison is Newey-West against a naive OLS error on the SAME 5-day fit, not against h=1
    m5 = har.fit(rv, horizon=5)
    naive = har.fit(rv, horizon=5, lag=0)
    check("for an overlapping 5-day target, Newey-West errors exceed the naive ones",
          bool(np.all(m5.se[1:] > naive.se[1:])),
          f"daily coefficient se {naive.se[1]:.4f} naive -> {m5.se[1]:.4f} Newey-West")
    f = har.forecast(m, rv)
    y = har.targets(rv, 1)
    ok = np.isfinite(f) & np.isfinite(y)
    bias = float(np.mean(y[ok] - f[ok]) / np.std(y[ok] - f[ok]) * math.sqrt(ok.sum()))
    check("the in-sample forecast is unbiased (mean error within 3 SE)", abs(bias) < 3, f"t {bias:+.2f}")
    try:
        har.fit(rv, train=30)            # rows 21..28: 8 days for 4 coefficients
        refused = False
    except ValueError:
        refused = True
    check("too few days for the coefficients is refused, not fitted", refused)


def test_log_har():
    print("\nthe log-HAR and its correction")
    rng = np.random.default_rng(3)
    n = 5000
    z = np.zeros(n)
    for t in range(22, n):
        z[t] = -9.0 * 0.1 + 0.5 * z[t - 1] + 0.2 * z[t - 5:t].mean() + 0.2 * z[t - 22:t].mean() \
            + 0.4 * rng.standard_normal()
    rv = np.exp(z)
    m = har.fit(rv, horizon=1, log=True)
    f = har.forecast(m, rv)
    raw = har.HAR(m.coef, m.se, 1, True, m.windows, 0.0, m.n_obs, m.names)
    f0 = har.forecast(raw, rv)
    y = har.targets(rv, 1)
    ok = np.isfinite(f) & np.isfinite(y)
    # the correction rests on one assumption, checkable on the residuals: E exp(e) = exp(s^2/2).
    # (A ratio of sample MEANS of the levels is not a usable test: the levels are so skewed
    # that a handful of days carries both means.)
    e = np.log(y[ok]) - np.log(f0[ok])
    check("the log residuals satisfy the lognormal identity E exp(e) = exp(s^2/2) within 2%",
          abs(float(np.mean(np.exp(e))) / math.exp(0.5 * m.resid_var) - 1) < 0.02,
          f"{np.mean(np.exp(e)):.4f} against {math.exp(0.5 * m.resid_var):.4f}")
    q1, q0 = har.qlike(y[ok], f[ok]), har.qlike(y[ok], f0[ok])
    check("...and the corrected forecast scores better in QLIKE than the uncorrected one",
          q1 < q0, f"QLIKE {q1:.4f} corrected against {q0:.4f}")
    try:
        har.fit(np.array([1.0, -1.0] * 100), log=True)
        refused = False
    except ValueError:
        refused = True
    check("a non-positive realised variance is refused by the log-HAR", refused)


def test_qlike():
    print("\nQLIKE")
    t = np.array([1.0, 2.0, 0.5])
    check("zero at a perfect forecast", har.qlike(t, t) == 0.0)
    check("positive otherwise", har.qlike(t, 1.1 * t) > 0 and har.qlike(t, 0.9 * t) > 0)
    check("under-prediction costs more than over-prediction by the same factor",
          har.qlike(t, t / 1.5) > har.qlike(t, t * 1.5),
          f"{har.qlike(t, t / 1.5):.4f} against {har.qlike(t, t * 1.5):.4f}")


if __name__ == "__main__":
    print("=" * 74)
    print("Session N -- HAR-RV")
    print("=" * 74)
    test_design()
    test_recovery()
    test_log_har()
    test_qlike()
    print("\n" + "=" * 74)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    print("=" * 74)
    sys.exit(1 if FAIL else 0)
