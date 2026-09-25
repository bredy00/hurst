"""
The implementation detail that mattered in Session E, measured (Session G).

The ETDRK4 Riccati step for the lifted rough Heston does, per step, two
contractions S = (w * E_j) @ psi over the N = 24 factors, with psi of shape
(N, n_u) complex, and one rank-3 update psi += A1 Nu + A2 Nab + A3 Nc.

  contraction   matmul goes through OpenBLAS, which wakes its thread pool on
                every call whatever the size; einsum stays in numpy's own loop.
  rank-3 update broadcasting A_k[:, None] * v_k three times allocates three
                (N, n_u) temporaries; one real (N, 3) @ complex (3, n_u) product
                on a pre-stacked matrix does it in a single call.

This script times each variant against n_u, with OpenBLAS at its default
thread count and pinned to one thread, and the full cf solve per ODE step
with the shipped code against a copy forced onto the old paths. It writes
captures/riccati_benchmark.png and captures/riccati_benchmark.json.

    python benchmark_riccati.py      (~1 minute)
"""

import json
import pathlib
import platform
import time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

import models.rough_heston as rh

ROOT = pathlib.Path(__file__).parent
N = 24
SIZES = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def timeit(fn, min_time=0.15, min_reps=20):
    """Median seconds per call over repeated batches (warm)."""
    for _ in range(3):
        fn()
    reps, t0 = 0, time.perf_counter()
    while True:
        fn()
        reps += 1
        if reps >= min_reps and time.perf_counter() - t0 > min_time:
            break
    batch = max(1, reps // 10)
    samples = []
    for _ in range(9):
        t1 = time.perf_counter()
        for _ in range(batch):
            fn()
        samples.append((time.perf_counter() - t1) / batch)
    return float(np.median(samples))


def micro(n_u, rng):
    w = rng.random(N)
    psi = rng.standard_normal((N, n_u)) + 1j * rng.standard_normal((N, n_u))
    A = [rng.random(N) for _ in range(3)]
    V = [rng.standard_normal(n_u) + 1j * rng.standard_normal(n_u) for _ in range(3)]
    Astack = np.stack(A, axis=1)                 # (N, 3)
    Vbuf = np.stack(V, axis=0)                   # (3, n_u)
    out = {}
    out["matmul"] = timeit(lambda: w @ psi)
    out["einsum"] = timeit(lambda: np.einsum("i,ij->j", w, psi))
    with threadpool_limits(limits=1, user_api="blas"):
        out["matmul_1thread"] = timeit(lambda: w @ psi)
    out["rank3_broadcast"] = timeit(lambda: A[0][:, None] * V[0] + A[1][:, None] * V[1] + A[2][:, None] * V[2])
    out["rank3_session_d"] = timeit(lambda: np.column_stack(A) @ np.vstack(V))
    out["rank3_stacked"] = timeit(lambda: Astack @ Vbuf)
    with threadpool_limits(limits=1, user_api="blas"):
        out["rank3_stacked_1thread"] = timeit(lambda: Astack @ Vbuf)
    return out


U_MAX = 100.0     # 64 ETDRK4 steps over 30 days are stable to |u| ~ 128: time real arithmetic, not NaNs


def full_step(n_u, p, tau=30 / 365, threads=None):
    """Seconds per ODE step of the shipped ETDRK4 solve (optionally with BLAS pinned)."""
    u = np.linspace(0.0, U_MAX, n_u) - 0.5j
    M = 64
    run = lambda: rh.log_char_func(u, tau, p, steps=M, scheme="etdrk4")
    if threads:
        with threadpool_limits(limits=threads, user_api="blas"):
            return timeit(run, min_time=0.3, min_reps=3) / M
    return timeit(run, min_time=0.3, min_reps=3) / M


def old_step_time(n_u, p, tau=30 / 365, M=64):
    """
    The Session D loop body exactly as committed in 104f74d: matmul contractions
    and the rank-3 update as np.column_stack(...) @ np.vstack(...) built afresh
    every step, around the same precomputation, timed per step.
    """
    u = np.linspace(0.0, U_MAX, n_u) - 0.5j
    v0, kappa, theta, xi, rho, H = p.as_tuple()
    w, x = rh.lift_nodes(H, N)
    iu = 1j * u
    b0 = -0.5 * (u * u + iu)
    lin = iu * rho * xi - kappa
    q = 0.5 * xi * xi
    kt = kappa * theta

    def F(P):
        return b0 + lin * P + q * P * P
    hs = np.full(M, tau / M)
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
    wE, wE2 = E * w[None, :], E2 * w[None, :]
    wA1, wA2, wA3 = A1 @ w, A2 @ w, A3 @ w

    def run():
        psi = np.zeros((N, n_u), dtype=complex)
        phi = np.zeros(n_u, dtype=complex)
        Psi = np.zeros(n_u, dtype=complex)
        for j in range(M):
            h = hs[j]
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
            psi += np.column_stack([A1[j], A2[j], A3[j]]) @ np.vstack([Nu, Nab, Nc])
            Psi = S1 + wA1[j] * Nu + wA2[j] * Nab + wA3[j] * Nc
            phi += (h / 6.0) * (Gu + 2.0 * (kt * (Pa + Pb) + v0 * Nab) + kt * Pc + v0 * Nc)
        return phi
    return timeit(run, min_time=0.3, min_reps=3) / M


def main():
    rng = np.random.default_rng(0)
    p = rh.RoughHestonParams(0.04, 2.0, 0.045, 0.5, -0.7, 0.12)
    info = [{k: d.get(k) for k in ("internal_api", "version", "num_threads")} for d in threadpool_info()]
    print(f"{platform.processor() or platform.machine()}; BLAS: {info}")
    rows = []
    for n_u in SIZES:
        m = micro(n_u, rng)
        new = full_step(n_u, p)
        new1 = full_step(n_u, p, threads=1)
        old = old_step_time(n_u, p)
        rows.append({"n_u": n_u, **m, "step_new": new, "step_new_1thread": new1, "step_old": old})
        print(f"n_u {n_u:5d}:  contraction matmul {1e6*m['matmul']:7.1f} us  einsum {1e6*m['einsum']:7.1f} us  "
              f"matmul@1thread {1e6*m['matmul_1thread']:7.1f} us  |  rank-3 broadcast {1e6*m['rank3_broadcast']:7.1f} us  "
              f"stacked {1e6*m['rank3_stacked']:7.1f} us  Session D {1e6*m['rank3_session_d']:7.1f} us  |  full ETDRK4 step old {1e6*old:8.1f} us  new {1e6*new:8.1f} us  "
              f"({old/new:.1f}x)  new, BLAS 1 thread {1e6*new1:8.1f} us ({old/new1:.1f}x)", flush=True)
    out = {"machine": platform.processor() or platform.machine(), "blas": info, "N": N, "rows": rows}
    (ROOT / "captures" / "riccati_benchmark.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    plot(rows)
    return out


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = np.array([r["n_u"] for r in rows])
    g = lambda k: np.array([r[k] for r in rows]) * 1e6
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    a = ax[0]
    a.loglog(n, g("matmul"), "o-", color="#d62728", label="w @ psi (OpenBLAS, 8 threads)")
    a.loglog(n, g("matmul_1thread"), "s--", color="#ff9896", label="w @ psi (OpenBLAS, 1 thread)")
    a.loglog(n, g("einsum"), "o-", color="#1f77b4", label="einsum('i,ij->j')")
    a.axvline(1024, color="0.5", ls=":", lw=1)
    a.text(1100, a.get_ylim()[0] * 1.5, "switch at 1024", color="0.4", fontsize=8)
    a.set_xlabel("u-nodes per solve (n_u)")
    a.set_ylabel("microseconds per call")
    a.set_title("(a) the contraction, N = 24 factors")
    a.legend(fontsize=8)
    a = ax[1]
    a.loglog(n, g("rank3_session_d"), "o-", color="#d62728", label="column_stack @ vstack, per step (Session D)")
    a.loglog(n, g("rank3_broadcast"), "^:", color="#9467bd", label="A1 Nu + A2 Nab + A3 Nc (broadcast)")
    a.loglog(n, g("rank3_stacked"), "o-", color="#1f77b4", label="Astack @ Vbuf (one real x complex call)")
    a.loglog(n, g("rank3_stacked_1thread"), "s--", color="#aec7e8", label="Astack @ Vbuf, 1 thread")
    a.set_xlabel("u-nodes per solve (n_u)")
    a.set_title("(b) the rank-3 update")
    a.legend(fontsize=8)
    a = ax[2]
    sp = g("step_old") / g("step_new")
    a.semilogx(n, g("step_old") / g("step_new_1thread"), "s--", color="#98df8a", label="shipped code, BLAS pinned to 1 thread")
    a.semilogx(n, sp, "o-", color="#2ca02c", label="shipped code, default threads")
    a.legend(fontsize=8)
    for x_, y_ in zip(n, sp):
        a.annotate(f"{y_:.1f}x", (x_, y_), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    a.set_xlabel("u-nodes per solve (n_u)")
    a.set_ylabel("old / new time per ETDRK4 step")
    a.set_title("(c) full Riccati step: Session D code / shipped code")
    a.axhline(1.0, color="0.6", lw=0.8)
    fig.suptitle("Where the Riccati step's time goes -- lifted rough Heston, N = 24", fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "captures" / "riccati_benchmark.png", dpi=130)


if __name__ == "__main__":
    main()
