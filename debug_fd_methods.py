"""
Which second-derivative method actually holds second order on non-uniform
strike grids? Measured, not argued. This is the study behind fd_second_poly.

Four candidates against the exact lognormal density obtained by differentiating
Black-76 call prices:

    raw 3-pt        the classical non-uniform stencil (volsurf_core.fd_second)
    mapped 3-pt     the coordinate-map idea: x = f(K) with the grid uniform in
                    x (the index), uniform differences in x, chain rule back
                    with metrics K', K'' from central differences on the index
    log-map         the same with the analytic map s = ln K
    Fornberg 5-pt   local quartic through five nodes (volsurf_core.fd_second_poly)

on four grids: smooth tanh stretch, geometric, a $1-then-$5 kink like SPY, and
random spacing. The observed order is log2 of the error ratio per doubling.

    python debug_fd_methods.py
"""

import math

import numpy as np

import volsurf_core as vc


def fd2_mapped(x, y):
    """Index map with numerically differenced metrics."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    d = np.full(len(x), np.nan)
    yx = 0.5 * (y[2:] - y[:-2])
    yxx = y[2:] - 2 * y[1:-1] + y[:-2]
    kx = 0.5 * (x[2:] - x[:-2])
    kxx = x[2:] - 2 * x[1:-1] + x[:-2]
    d[1:-1] = (yxx - (yx / kx) * kxx) / (kx * kx)
    return d


def fd2_logmap(x, y):
    """Analytic map s = ln K, exact metrics, 3-point stencil in s."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    s = np.log(x)
    d = np.full(len(x), np.nan)
    ys, yss = vc.fd_first(s, y), vc.fd_second(s, y)
    d[1:-1] = ((yss - ys) / (x * x))[1:-1]
    return d


def grids(n, kind):
    if kind == "smooth":
        s = np.linspace(-1, 1, n)
        return 1.0 + 0.6 * np.tanh(1.5 * s) / math.tanh(1.5)
    if kind == "geometric":
        return np.exp(np.linspace(math.log(0.5), math.log(1.8), n))
    if kind == "kink":
        a = np.arange(0.40, 1.0001, 0.6 / (n * 0.7))
        b = np.arange(a[-1] + 5 * (a[1] - a[0]), 1.8, 5 * (a[1] - a[0]))
        return np.concatenate([a, b])
    if kind == "random":
        rng = np.random.default_rng(0)
        h = 1.0 + 0.4 * rng.uniform(-1, 1, n - 1)
        return 0.4 + 1.4 * np.concatenate([[0], np.cumsum(h)]) / h.sum()
    raise ValueError(kind)


def bs_call(K, sig=0.25, tau=0.25):
    return np.array([float(vc.bs_price(1.0, k, sig, tau, 1.0, 'C')) for k in K])


def bs_density(K, sig=0.25, tau=0.25):
    return (np.exp(-(np.log(K) + 0.5 * sig * sig * tau) ** 2 / (2 * sig * sig * tau))
            / (K * sig * math.sqrt(2 * math.pi * tau)))


METHODS = {
    "raw 3-pt": vc.fd_second,
    "mapped 3-pt (index metrics)": fd2_mapped,
    "log-map (analytic)": fd2_logmap,
    "Fornberg 5-pt": vc.fd_second_poly,
}
SIZES = (101, 201, 401, 801)


def study(kind, verbose=True):
    """Returns {method: (errors, mean order)} for one grid family."""
    out = {}
    for name, f in METHODS.items():
        errs = []
        for n in SIZES:
            K = grids(n, kind)
            d = f(K, bs_call(K))
            q = bs_density(K)
            ok = np.isfinite(d)
            errs.append(float(np.max(np.abs(d[ok] - q[ok])) / q.max()))
        orders = [math.log2(errs[i] / errs[i + 1]) for i in range(len(SIZES) - 1)]
        out[name] = (errs, float(np.mean(orders)))
        if verbose:
            print(f"  {name:30}" + "".join(f"{e:11.2e}" for e in errs)
                  + f"   order {np.mean(orders):5.2f}")
    return out


def main():
    for kind in ("smooth", "geometric", "kink", "random"):
        print(f"\n--- grid: {kind} ---   max |q_fd - q_exact| / peak at n = {SIZES}")
        study(kind)
    x = np.linspace(0.5, 3.0, 200)
    h = x[1] - x[0]
    e2 = float(np.nanmax(np.abs(vc.fd_second(x, x ** 2)[1:-1] - 2.0)))
    print(f"\nThe uniform quadratic health check: measured {e2:.2e}; the rounding "
          f"floor 4 eps max|y| / h^2 is {4 * np.finfo(float).eps * 9 / h ** 2:.2e}. "
          f"That number is rounding, not truncation.")


if __name__ == "__main__":
    main()
