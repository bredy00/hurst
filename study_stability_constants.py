"""
Re-measure the Riccati stability constants C_STAB on a given lift (Session I).

C_STAB bounds z = h^alpha / Gamma(1+alpha) * (xi (1+|rho|) |u|_max + kappa): a solve is
stable when z <= C_STAB. The shipped constants (etdrk4 2.0, exptrap 12.0) were measured
in Session G on the N = 24, eta_N = 1e5 lift as the edge where max |phi(u - i/2)| over
u in [0, 1200] first exceeds one. The finer lift puts more of the kernel's weight into
the first seconds, so the edge is measured again here: for each scheme, H and maturity,
the fewest steps M that keep max |phi| <= 1 + 1e-9, converted to z.

    python study_stability_constants.py [N eta_N]      (default 40 1e8)
"""

import math
import sys

import numpy as np

import models.rough_heston as rh


def max_phi(u, tau, P, M, scheme, N, eta):
    lp = rh.log_char_func(u, tau, P, N=N, eta_N=eta, steps=M, scheme=scheme, check_stability=False)
    with np.errstate(over="ignore", invalid="ignore"):
        v = np.abs(np.exp(lp))
    return float(np.nanmax(np.where(np.isfinite(v), v, np.inf)))


def edge(tau, P, scheme, N, eta, umax=1200.0):
    u = np.linspace(0.0, umax, 1200) - 0.5j
    lo, hi = 2, 4096                       # hi assumed stable
    if max_phi(u, tau, P, hi, scheme, N, eta) > 1 + 1e-9:
        return None
    while hi - lo > 1:                     # smallest stable M, assuming stability is monotone in M
        mid = (lo + hi) // 2
        if max_phi(u, tau, P, mid, scheme, N, eta) <= 1 + 1e-9:
            hi = mid
        else:
            lo = mid
    a = P.H + 0.5
    G = P.xi * (1 + abs(P.rho)) * umax + P.kappa
    z_stable = (tau / hi) ** a / math.gamma(1 + a) * G
    z_unstable = (tau / lo) ** a / math.gamma(1 + a) * G
    return hi, z_stable, z_unstable


def main(N=40, eta=1e8):
    print(f"lift N = {N}, eta_N = {eta:g}; shipped constants {rh.C_STAB}")
    out = {}
    for scheme in ("etdrk4", "exptrap"):
        zs = []
        for H in (0.02, 0.12):
            P = rh.RoughHestonParams(0.04, 2.0, 0.045, 0.5, -0.7, H)
            for d_ in (30, 90, 365):
                r = edge(d_ / 365, P, scheme, N, eta)
                if r is None:
                    print(f"  {scheme} H={H} {d_}d: unstable even at 4096 steps")
                    continue
                M, zs_, zu = r
                zs.append(zs_)
                print(f"  {scheme:7s} H={H:4.2f} {d_:3d}d: stable from M = {M:4d} steps; z stable {zs_:6.2f}, "
                      f"one step fewer z {zu:6.2f} blows up", flush=True)
        out[scheme] = min(zs) if zs else None
        print(f"  {scheme}: smallest stable edge z = {out[scheme]:.2f} (constant in use {rh.C_STAB[scheme]})", flush=True)
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    main(int(args[0]), float(args[1])) if args else main()
