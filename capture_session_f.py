"""
Session F in one figure: Hawkes arrivals on three hosts, and the filters.

  (a) Expected intensity after two planted shocks, Hawkes vs Poisson, with one
      simulated path -- the second peak is higher by alpha e^{-(beta-alpha) d}.
  (b) The single-shock mean response r(s) on every host. The second spike is
      larger iff r is still RISING across the window: true for Hawkes on OU and
      Heston, never for Poisson, never for rough Heston with jumps in the
      Volterra driver at this branching ratio.
  (c) Excess kurtosis of daily returns, Hawkes vs Poisson at the same mean rate.
  (d) The lecture's regime change and bad print: strict, adaptive, robust.
  (e) Parameter learning: Wan-Nelson dual EKF and the recursive MLE with Ljung's
      sensitivities, against the offline MLE on the same data.
  (f) Rough host: 5-day forecasts from the true lifted filter vs a CIR filter
      fitted by maximum likelihood (kappa ~ 50).

    python capture_session_f.py
"""

import math
import pathlib
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import models.hawkes as hk
import models.jump_hosts as jh
from study_jump_modes import DirectJumpHost, DriverIncrementHost   # the Session F semantics (Session I)
import filters.kalman as kf

OUT = pathlib.Path(__file__).parent / "captures"
OUT.mkdir(exist_ok=True)
np.seterr(all="ignore")

H = hk.HawkesParams(mu=5.0, alpha=150.0, beta=250.0)
P = H.matched_poisson()
DT = 1.0 / (252 * 4)
DAY = 1.0 / 252
C = {"hawkes": "#ff5fd2", "poisson": "#00f2ff", "ou": "#ffb020", "heston": "#7ee787",
     "rough_d": "#ff6b6b", "rough_x": "#c792ea", "truth": "#e6edf3", "strict": "#00f2ff",
     "adaptive": "#ffb020", "robust": "#7ee787", "mle": "#e6edf3"}


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor("#161b22")
    ax.set_title(title, color="white", fontsize=10)
    ax.set_xlabel(xlabel, color="#9aa4b2", fontsize=9)
    ax.set_ylabel(ylabel, color="#9aa4b2", fontsize=9)
    ax.tick_params(colors="#9aa4b2", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#2d333b")


def main():
    t_all = time.perf_counter()
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor("#0b0d0f")

    # (a) intensity
    ax = axes[0, 0]
    t1, t2 = 0.10, 0.10 + 5 * DAY
    grid = np.linspace(0.0, 0.25, 2001)
    ev = hk.simulate(H, 0.25, n_paths=1, seed=4, planted=(t1, t2), burn_in=1.0)
    lam = ev.intensity_on_grid(2000, 0.25)[0]
    ax.plot(grid * 252, lam, color=C["hawkes"], lw=0.8, alpha=0.55, label="one Hawkes path")
    ax.plot(grid * 252, hk.expected_intensity(H, grid, (t1, t2)), color=C["hawkes"], lw=2.2,
            label="E[λ], Hawkes (exact)")
    ax.plot(grid * 252, hk.expected_intensity(P, grid, (t1, t2)), color=C["poisson"], lw=2.2,
            label="E[λ], Poisson, same mean rate")
    for t in (t1, t2):
        ax.axvline(t * 252, color="#ff3e3e", ls=":", lw=1)
    style(ax, "(a) two planted shocks, 5 days apart: intensity", "time (trading days)", "λ(t)  per year")
    ax.legend(fontsize=8, facecolor="#0b0d0f", edgecolor="#2d333b")

    # (b) single-shock responses
    ax = axes[0, 1]
    n = 60 * 4 + 20
    # OU and Heston share kappa and the arrivals, and both are linear in the mean,
    # so their NORMALISED responses coincide exactly; OU is drawn dashed on top.
    hosts = [
        ("Heston variance (Hawkes)", jh.HestonHost(kappa=3.0, theta=0.04, xi=0.3),
         jh.JumpSizes(state_mean=0.01), H, C["heston"], "-"),
        ("OU log-variance (Hawkes) — identical once normalised", jh.OUHost(kappa=3.0, theta=math.log(0.04), sigma=1.0),
         jh.JumpSizes(state_mean=0.05), H, C["ou"], "--"),
        ("rough, driver jumps (Hawkes)", DriverIncrementHost(),
         jh.JumpSizes(state_mean=0.002), H, C["rough_d"], "-"),
        ("rough, direct jumps (Hawkes)", DirectJumpHost(),
         jh.JumpSizes(state_mean=0.01), H, C["rough_x"], "-"),
        ("Heston variance (Poisson)", jh.HestonHost(kappa=3.0, theta=0.04, xi=0.3),
         jh.JumpSizes(state_mean=0.01), P, C["poisson"], "--"),
    ]
    s_days = (np.arange(n + 1) - 10) * DT * 252
    for label, host, J, proc, col, ls in hosts:
        r = jh.shock_response(host, proc, J, n * DT, n, 10)
        rr = r[11:] / r[11]
        ax.plot(s_days[11:], rr, ls, color=col, lw=2, label=label)
    ax.axhline(1.0, color="#2d333b", lw=1)
    style(ax, "(b) mean response to ONE shock, normalised: rising ⇒ second spike larger",
          "days after the shock", "r(s) / r(0+)")
    ax.legend(fontsize=7.5, facecolor="#0b0d0f", edgecolor="#2d333b")

    # (c) kurtosis
    ax = axes[0, 2]
    s = 0.03
    labels, kh, kp, eh, ep = [], [], [], [], []
    for label, host, J0 in (("constant\n(Merton)", jh.ConstantHost(0.04), jh.JumpSizes()),
                            ("OU", jh.OUHost(kappa=3.0, theta=math.log(0.04), sigma=1.0), jh.JumpSizes(state_mean=0.05)),
                            ("Heston", jh.HestonHost(kappa=3.0, theta=0.04, xi=0.3), jh.JumpSizes(state_mean=0.01)),
                            ("rough\n(driver)", DriverIncrementHost(), jh.JumpSizes(state_mean=0.002)),
                            ("rough\n(direct)", DirectJumpHost(), jh.JumpSizes(state_mean=0.01))):
        J = jh.JumpSizes(state_mean=J0.state_mean, price_std=s)
        vals = {}
        for name, proc in (("h", H), ("p", P)):
            sim = jh.simulate(host, proc, J, 2.0, 504 * 4, 2000, seed=7, record_every=4)
            vals[name] = jh.excess_kurtosis(np.diff(sim["X"], axis=0).T)
        labels.append(label)
        kh.append(vals["h"][0]); eh.append(vals["h"][1])
        kp.append(vals["p"][0]); ep.append(vals["p"][1])
    xs = np.arange(len(labels))
    ax.bar(xs - 0.2, kh, 0.4, yerr=np.array(eh) * 2, color=C["hawkes"], label="Hawkes")
    ax.bar(xs + 0.2, kp, 0.4, yerr=np.array(ep) * 2, color=C["poisson"], label="Poisson, same rate")
    th_h = jh.excess_kurtosis_theory(0.04, H, s, DAY)
    th_p = jh.excess_kurtosis_theory(0.04, P, s, DAY)
    ax.plot([-0.4, 0.0], [th_h, th_h], color="white", lw=1.5)
    ax.plot([0.0, 0.4], [th_p, th_p], color="white", lw=1.5, label="closed form (constant variance)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    style(ax, "(c) excess kurtosis of daily returns (±2 SE)", "", "excess kurtosis")
    ax.legend(fontsize=8, facecolor="#0b0d0f", edgecolor="#2d333b")

    # (d) strict / adaptive / robust on the lecture's example
    ax = axes[1, 0]
    nn, kj, ko = 500, 150, 380
    p = dict(kappa=0.5, theta=100.0, sigma=8.0, R=25.0)
    truth = kf.simulate_ou(p["kappa"], p["theta"], p["sigma"], DAY, nn, seed=3, jumps={kj: -28.0})
    y = truth + np.random.default_rng(5003).normal(0.0, 5.0, nn)
    y[ko] = truth[ko] - 30.0
    ax.plot(np.arange(nn), y, ".", color="#6e7681", ms=2.5, label="quotes (one bad print at 380)")
    ax.plot(np.arange(nn), truth, color=C["truth"], lw=1.0, ls="--", label="true level")
    for name, pol in (("strict", kf.Strict()), ("adaptive", kf.Adaptive(3.0)), ("robust", kf.Robust(3.0, 2))):
        est = kf.kalman_filter(kf.OUModel(DAY), p, y, policy=pol)["estimate"]
        ax.plot(np.arange(nn), est, color=C[name], lw=1.8, label=name)
    ax.set_ylim(55, 125)
    style(ax, "(d) regime change at 150, bad print at 380 (the lecture's two panels)", "trading days", "level")
    ax.legend(fontsize=8, facecolor="#0b0d0f", edgecolor="#2d333b", ncol=2)

    # (e) parameter learning
    ax = axes[1, 1]
    m = kf.OUModel(DAY)
    pp = dict(kappa=5.0, theta=0.2, sigma=0.3, R=0.05 ** 2)
    x = kf.simulate_ou(5.0, 0.2, 0.3, DAY, 5000, seed=2)
    yy = x + np.random.default_rng(1002).normal(0.0, 0.05, 5000)
    fit = kf.fit_mle(m, yy, dict(pp, kappa=1.5), names=("kappa",))
    du = kf.dual_kalman(m, dict(pp, kappa=1.5), yy, learn=("kappa",))
    rp = kf.recursive_mle(m, dict(pp, kappa=1.5), yy, learn=("kappa",))
    ax.plot(du["params"][:, 0], color=C["adaptive"], lw=1.6, label=f"Wan–Nelson dual EKF → {du['final']['kappa']:.2f}")
    ax.plot(rp["params"][:, 0], color=C["robust"], lw=1.6, label=f"recursive MLE (Ljung) → {rp['final']['kappa']:.2f}")
    k_mle, k_se = fit["params"]["kappa"], fit["se"]["kappa"]
    ax.axhline(k_mle, color=C["mle"], lw=1.2, label=f"offline MLE {k_mle:.2f} ± {k_se:.2f}")
    ax.axhspan(k_mle - k_se, k_mle + k_se, color="#e6edf3", alpha=0.08)
    ax.axhline(5.0, color="#ff3e3e", ls=":", lw=1, label="true κ = 5")
    ax.set_ylim(0, 10)
    style(ax, "(e) learning κ online from a wrong start (κ₀ = 1.5)", "observations", "κ estimate")
    ax.legend(fontsize=8, facecolor="#0b0d0f", edgecolor="#2d333b")

    # (f) rough forecasting
    ax = axes[1, 2]
    rm = kf.LiftedRoughModel(DAY)
    pr = dict(kappa=3.0, theta=0.04, xi=0.3, R=0.03 ** 2)
    V, _ = kf.simulate_rough(rm, pr, 2000, seed=11)
    yv = V + np.random.default_rng(201).normal(0.0, 0.03, 2000)
    cm = kf.CIRModel(DAY)
    pc = kf.fit_mle(cm, yv, dict(kappa=3.0, theta=float(np.mean(yv)), xi=0.3, R=pr["R"]))["params"]
    rl = kf.kalman_filter(rm, pr, yv)
    rc = kf.kalman_filter(cm, pc, yv)
    Al, bl = rm.transition(pr)
    Ac, bc = cm.transition(pc)
    h = 5
    fl, fc = [], []
    for k in range(len(yv) - h):
        xl = rl["x_post"][k].copy()
        xc = rc["x_post"][k].copy()
        for _ in range(h):
            xl = Al @ xl + bl
            xc = Ac @ xc + bc
        fl.append(float(rm.measure(xl)))
        fc.append(float(cm.measure(xc)))
    lo, hi = 1300, 1500
    ax.plot(np.arange(lo, hi), V[lo:hi], color=C["truth"], lw=1.0, label="rough variance V")
    ax.plot(np.arange(lo, hi), np.array(fl)[lo - h:hi - h], color=C["rough_d"], lw=1.5, label="5-day forecast, true lifted model")
    ax.plot(np.arange(lo, hi), np.array(fc)[lo - h:hi - h], color=C["poisson"], lw=1.5,
            label=f"5-day forecast, fitted CIR (κ = {pc['kappa']:.0f})")
    el = math.sqrt(np.mean((np.array(fl[300:]) - V[300 + h:]) ** 2))
    ec = math.sqrt(np.mean((np.array(fc[300:]) - V[300 + h:]) ** 2))
    style(ax, f"(f) forecasting rough variance: RMSE lifted {el:.4f} vs CIR {ec:.4f}", "trading days", "variance")
    ax.legend(fontsize=8, facecolor="#0b0d0f", edgecolor="#2d333b")

    fig.subplots_adjust(left=0.05, right=0.985, top=0.94, bottom=0.07, hspace=0.32, wspace=0.2)
    out = OUT / "session_f.png"
    fig.savefig(out, dpi=115, facecolor=fig.get_facecolor())
    print(f"figure -> {out}  ({time.perf_counter()-t_all:.0f}s)")


if __name__ == "__main__":
    main()
