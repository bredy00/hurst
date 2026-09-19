"""
Session F, task F1: Hawkes jump arrivals, attached to three volatility hosts.

The load-bearing test is the second-spike property, and it is asserted in its
exact form rather than as a Monte Carlo inequality:

    E[increment after the 2nd shock] - E[increment after the 1st] = r(d + W) - r(d)

where r is the single-shock mean response, so the second spike is larger iff r
is still rising across the window. Poisson makes that impossible on every host;
Hawkes delivers it on the exponential hosts iff alpha > kappa at short gaps; on
rough Heston with jumps in the Volterra driver it needs near-critical clustering.
Since Session I the driver is the rough host's only jump mode, and a jump's size is
its integrated variance impact (test_rough_jump_size).
Monte Carlo is used to check the simulator against the exact expectations, not
to establish the property.

    python test_hawkes.py
"""

import math
import sys
import time

import numpy as np
from scipy import stats

import models.hawkes as hk
import models.jump_hosts as jh


PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


np.seterr(all="ignore")
H = hk.HawkesParams(mu=5.0, alpha=150.0, beta=250.0)      # n = 0.6, lambda_bar = 12.5 / yr
P = H.matched_poisson()
DT = 1.0 / (252 * 4)                                      # a quarter of a trading day
DAY = 1.0 / 252


def hosts():
    return [
        (jh.OUHost(kappa=3.0, theta=math.log(0.04), sigma=1.0), jh.JumpSizes(state_mean=0.05)),
        (jh.HestonHost(kappa=3.0, theta=0.04, xi=0.3), jh.JumpSizes(state_mean=0.01)),
        # driver jumps sized by integrated impact (Session I): 0.017 extra variance on
        # average over the first day, the Sessions F-H driver increment of 0.002
        (jh.RoughHost(kappa=3.0, theta=0.04, xi=0.3, H=0.12, v0=0.04), jh.JumpSizes(state_mean=0.017)),
    ]


# --- the process ------------------------------------------------------------
def test_parameters():
    print("\nF1 -- parameters and stationarity")
    for bad in ((1.0, 2.0, 2.0), (1.0, 3.0, 2.0), (0.0, 0.5, 1.0), (1.0, -0.1, 1.0), (1.0, 0.5, 0.0)):
        try:
            hk.HawkesParams(*bad)
            ok = False
        except ValueError:
            ok = True
        check(f"constructor rejects (mu, alpha, beta) = {bad}", ok)
    check("alpha < beta is accepted, and Poisson is alpha = 0",
          hk.HawkesParams(1.0, 0.99, 1.0).branching_ratio == 0.99 and P.alpha == 0.0)
    check("stationary intensity is mu / (1 - alpha / beta)",
          abs(H.stationary_intensity - 5.0 / (1 - 0.6)) < 1e-12, f"{H.stationary_intensity}")
    check("the matched Poisson process has the same mean rate",
          P.stationary_intensity == H.stationary_intensity)


def test_simulation_against_theory():
    print("\nF1 -- exact simulation against closed forms")
    n_paths, T = 6000, 5.0
    t0 = time.perf_counter()
    ev = hk.simulate(H, T, n_paths=n_paths, seed=1, burn_in=1.0)
    N = ev.counts()
    rate = N.mean() / T
    se = math.sqrt(hk.count_variance(H, T) / n_paths) / T
    check("simulated mean rate within 3 SE of mu / (1 - n)",
          abs(rate - H.stationary_intensity) < 3 * se,
          f"{rate:.4f} vs {H.stationary_intensity} (SE {se:.4f}; {time.perf_counter()-t0:.1f}s)")
    check("...and within 2% (the plan's bar; 3 SE is 1.1% at this sample size)",
          abs(rate / H.stationary_intensity - 1) < 0.02, f"{(rate/H.stationary_intensity-1)*100:+.2f}%")

    # Count variance, Hawkes (1971). SE of a sample variance by batching paths.
    for w in (0.05, 0.25, 1.0):
        c = ev.counts(0.0, w).astype(float)
        v = c.var(ddof=1)
        batches = np.array_split(c, 30)
        se_v = np.std([b.var(ddof=1) for b in batches], ddof=1) / math.sqrt(30)
        th = hk.count_variance(H, w)
        check(f"count variance over {w} y matches the closed form within 4 SE",
              abs(v - th) < 4 * se_v, f"{v:.3f} vs {th:.3f} (SE {se_v:.3f}), Fano {v/c.mean():.2f}")
    evp = hk.simulate(P, T, n_paths=n_paths, seed=2)
    c = evp.counts(0.0, 1.0).astype(float)
    check("Poisson counts are equidispersed (Fano 1 within 0.05)", abs(c.var(ddof=1) / c.mean() - 1) < 0.05,
          f"Fano {c.var(ddof=1)/c.mean():.3f}")
    check("Hawkes over-dispersion at 1 year is (1-n)^-2 in the limit: theory Fano > 3.5",
          hk.fano_factor(H, 1.0) > 3.5, f"{hk.fano_factor(H, 1.0):.3f} (limit {1/(1-0.6)**2:.2f})")


def test_intensity_peaks():
    """The plan's statement, in expectation and exactly: E[lambda] just after the second of two
    shocks exceeds that after the first by alpha e^{-(beta-alpha) d}; for Poisson it is equal."""
    print("\nF1 -- two planted shocks: intensity peaks")
    d = 5 * DAY
    t1, t2 = 1.0, 1.0 + d
    e1 = float(hk.expected_intensity(H, t1, (t1, t2)))
    e2 = float(hk.expected_intensity(H, t2, (t1, t2)))
    check("exact: E[lambda(t2+)] - E[lambda(t1+)] = alpha e^{-(beta - alpha) d} > 0",
          abs((e2 - e1) - H.alpha * math.exp(-H.excess_decay * d)) < 1e-9 and e2 > e1,
          f"{e1:.3f} -> {e2:.3f}")
    p1 = float(hk.expected_intensity(P, t1, (t1, t2)))
    p2 = float(hk.expected_intensity(P, t2, (t1, t2)))
    check("Poisson: the two peaks are identical -- shocks do not excite", p1 == p2, f"{p1} = {p2}")

    ev = hk.simulate(H, 1.1, n_paths=20000, seed=3, planted=(t1, t2), burn_in=1.0)
    n = 440
    lam = ev.intensity_on_grid(n, 1.1)
    grid = np.linspace(0.0, 1.1, n + 1)
    th = hk.expected_intensity(H, grid, (t1, t2))
    se = lam.std(axis=0, ddof=1) / math.sqrt(lam.shape[0])
    z = np.abs(lam.mean(axis=0) - th) / np.maximum(se, 1e-12)
    check("Monte Carlo E[lambda(t)] follows the exact curve (max |z| over the grid < 4)",
          float(z.max()) < 4.0, f"max |z| {float(z.max()):.2f} over {n+1} points")


def test_likelihood():
    print("\nF1 -- maximum likelihood and the time-rescaling test")
    covered = 0
    last = None
    for seed in range(3):
        one = hk.simulate(H, 300.0, n_paths=1, seed=10 + seed, burn_in=1.0)
        times = one.path(0)
        fit = hk.fit_mle(times, 300.0)
        q = fit["params"]
        z = [abs(getattr(q, k) - getattr(H, k)) / fit["se"][k] for k in ("mu", "alpha", "beta")]
        covered += sum(1 for v in z if v < 3.0)
        last = (times, fit)
        print(f"      seed {seed}: {len(times)} events  mu {q.mu:.2f}+/-{fit['se']['mu']:.2f}  "
              f"alpha {q.alpha:.1f}+/-{fit['se']['alpha']:.1f}  beta {q.beta:.1f}+/-{fit['se']['beta']:.1f}")
    check("MLE recovers (mu, alpha, beta) within 3 SE in at least 8 of 9 cases",
          covered >= 8, f"{covered}/9 within 3 SE")
    times, fit = last
    ks_h = stats.kstest(hk.rescaled_intervals(fit["params"], times), "expon").pvalue
    fp = hk.fit_mle(times, 300.0, poisson=True)
    ks_p = stats.kstest(hk.rescaled_intervals(fp["params"], times), "expon").pvalue
    check("rescaled intervals under the fitted Hawkes model are Exp(1) (KS p > 0.01)",
          ks_h > 0.01, f"p = {ks_h:.3f}")
    check("...under a Poisson fit to the SAME events they are not (KS p < 1e-6)",
          ks_p < 1e-6, f"p = {ks_p:.1e}; log L Hawkes {fit['loglik']:.0f} vs Poisson {fp['loglik']:.0f}")
    poi = hk.simulate(P, 300.0, n_paths=1, seed=20).path(0)
    fpp = hk.fit_mle(poi, 300.0, poisson=True)
    check("Poisson events under a Poisson fit pass the same test (control)",
          stats.kstest(hk.rescaled_intervals(fpp["params"], poi), "expon").pvalue > 0.01)


# --- the hosts: second spike --------------------------------------------------
def test_second_spike_exact():
    print("\nF1 -- the second-spike property, exactly")
    try:
        jh.second_spike(hosts()[0][0], H, hosts()[0][1], DT, 8, 8)
        raised = False
    except ValueError:
        raised = True
    check("a window as long as the gap is refused (the second jump would land inside it)", raised)

    for host, J in hosts():
        a = jh.second_spike(host, H, J, DT, 20, 8)
        err = abs((a["inc2"] - a["inc1"]) - (a["r_dW"] - a["r_d"]))
        check(f"{host.name}: superposition identity inc2 - inc1 = r(d+W) - r(d)",
              err < 1e-12 * max(1.0, abs(a["inc1"])), f"error {err:.1e}")

    for host, J in hosts():
        a = jh.second_spike(host, P, J, DT, 20, 8)
        n = 200
        r = jh.shock_response(host, P, J, n * DT, n, 10)[11:]
        check(f"{host.name}, Poisson: second increment SMALLER, response strictly decreasing",
              a["inc2"] < a["inc1"] and np.all(np.diff(r) < 0),
              f"inc {a['inc1']:.5f} -> {a['inc2']:.5f}")
        check(f"{host.name}, Poisson: yet the LEVEL after the second shock is higher (not a test)",
              a["level2"] > a["level1"], f"level {a['level1']:.4f} -> {a['level2']:.4f}")

    ou, heston, rough_driver = hosts()
    for host, J in (ou, heston):
        a = jh.second_spike(host, H, J, DT, 8, 4)
        b = jh.second_spike(host, H, J, DT, 40, 8)
        check(f"{host.name}, Hawkes: second spike larger at a 2-day gap, not at a 10-day gap",
              a["inc2"] > a["inc1"] and b["inc2"] < b["inc1"],
              f"2 d: {a['inc1']:.4f} -> {a['inc2']:.4f};  10 d: {b['inc1']:.4f} -> {b['inc2']:.4f}")
        kap = host.kappa
        above = hk.HawkesParams(mu=5.0, alpha=1.03 * kap, beta=1.03 * kap / 0.6)
        below = hk.HawkesParams(mu=5.0, alpha=0.97 * kap, beta=0.97 * kap / 0.6)
        ya = jh.second_spike(host, above, J, DT, 2, 1)
        yb = jh.second_spike(host, below, J, DT, 2, 1)
        check(f"{host.name}: at short gaps the property switches on at alpha = kappa (r'(0) = eta (alpha - kappa))",
              ya["inc2"] > ya["inc1"] and yb["inc2"] < yb["inc1"],
              f"alpha = 1.03 kappa: {ya['inc2']-ya['inc1']:+.1e};  0.97 kappa: {yb['inc2']-yb['inc1']:+.1e}")

    grid = [(d, w) for d in (2, 4, 8, 20, 40, 80) for w in (1, 2, 4, 8, 20, 40) if w < d]
    host, J = rough_driver
    held = sum(1 for d, w in grid if (lambda a: a["inc2"] > a["inc1"])(jh.second_spike(host, H, J, DT, d, w)))
    check("rough Heston, jumps in the Volterra driver, n = 0.6: property holds NOWHERE on the grid",
          held == 0, f"{held} of {len(grid)} (gap, window) pairs -- the kernel's power-law decay wins")
    critical = hk.HawkesParams(mu=12.5 * 0.05, alpha=0.95 * 250.0, beta=250.0)
    held_c = sum(1 for d, w in grid if (lambda a: a["inc2"] > a["inc1"])(jh.second_spike(host, critical, J, DT, d, w)))
    check("...but it appears near criticality (n = 0.95, same mean rate)",
          held_c > len(grid) // 3, f"{held_c} of {len(grid)} pairs")


def test_rough_jump_size():
    """Session I: one rough jump mode (driver), sized by its integrated variance impact."""
    print("\nI -- rough jumps: driver only, sized by integrated variance impact")
    try:
        jh.RoughHost(jump_mode="direct")
        refused = False
    except TypeError:
        refused = True
    check("the direct mode is gone: RoughHost takes no jump_mode", refused)
    import models.rough_heston as rh
    impact = 0.02
    quiet = hk.HawkesParams.poisson(1e-9)
    rows, worst = {}, {}
    for lift in ((40, 1e8), (24, 1e5)):
        rows[lift], worst[lift] = [], 0.0
        with rh.using_lift(*lift):
            host = jh.RoughHost(kappa=3.0, theta=0.04, xi=0.3, H=0.12, v0=0.04)
            for label, bars in (("5 min", 78), ("1 h", 6.5), ("1/4 day", 4)):
                dt = DAY / bars
                k1d = int(round(bars))
                n = 2 + k1d + 1
                r = jh.shock_response(host, quiet, jh.JumpSizes(state_mean=impact), n * dt, n, 2)[3:]
                got = float(np.sum(r[:k1d]) * dt / DAY)          # average extra variance over the day
                worst[lift] = max(worst[lift], abs(got / impact - 1.0))
                rows[lift].append(f"{label} {got / impact - 1:+.1%}")
    # the grid's cell-mean discretisation of the driver, worst at 1-hour steps
    check("default lift: the realised one-day impact matches the parameter within 5% at 5-minute, "
          "1-hour and quarter-day steps", worst[(40, 1e8)] < 0.05, "; ".join(rows[(40, 1e8)]))
    check("...and within 6% on the Sessions A-H lift (24 nodes to 1e5/y)", worst[(24, 1e5)] < 0.06,
          "; ".join(rows[(24, 1e5)]))
    host = jh.RoughHost(kappa=3.0, theta=0.04, xi=0.3, H=0.12, v0=0.04)
    exact = jh.driver_impact_integral(DAY, 0.12, 3.0)
    a = 0.62
    series = sum((-3.0) ** k * DAY ** (a * (k + 1)) / math.gamma(a * k + a + 1.0) for k in range(4))
    check("the conversion uses the exact response integral t^a E_{a,a+1}(-kappa t^a) "
          "(four series terms agree to 1e-4)", abs(exact / series - 1.0) < 1e-4,
          f"{exact:.6f} vs {series:.6f}; driver increment per unit impact {host.driver_per_impact:.5f}")


def test_second_spike_monte_carlo():
    print("\nF1 -- the simulator against the exact expectations")
    for host, J in hosts():
        a = jh.second_spike(host, H, J, DT, 20, 8)
        t0 = time.perf_counter()
        sim = jh.simulate(host, H, J, a["T"], a["n_steps"], 20000, seed=21, planted=a["planted"])
        st = sim["state"]
        ex = jh.expected_path(host, H, J, a["T"], a["n_steps"], planted=a["planted"])
        z = np.abs(st.mean(axis=1) - ex)[1:] / (st.std(axis=1, ddof=1)[1:] / math.sqrt(st.shape[1]))
        check(f"{host.name}: Monte Carlo mean path within 4 SE of the exact one everywhere",
              float(z.max()) < 4.0, f"max |z| {float(z.max()):.2f} over {len(z)} points "
                                    f"({time.perf_counter()-t0:.1f}s)")

    # One powered empirical confirmation of the property itself
    host = jh.OUHost(kappa=3.0, theta=math.log(0.04), sigma=0.3)
    J = jh.JumpSizes(state_mean=0.05)
    strong = hk.HawkesParams(mu=2.5, alpha=400.0, beta=500.0)
    a = jh.second_spike(host, strong, J, DT, 8, 4)
    sim = jh.simulate(host, strong, J, a["T"], a["n_steps"], 40000, seed=31, planted=a["planted"])
    st = sim["state"]
    j1, j2 = a["j1"], a["j2"]
    diff = (st[j2 + 5] - st[j2]) - (st[j1 + 5] - st[j1])
    se = diff.std(ddof=1) / math.sqrt(len(diff))
    check("OU host, strong clustering: Monte Carlo sees the second spike larger by > 4 SE",
          diff.mean() > 4 * se, f"inc2 - inc1 = {diff.mean():+.4f} +/- {se:.4f} (exact {a['inc2']-a['inc1']:+.4f})")


# --- kurtosis -----------------------------------------------------------------
def test_kurtosis():
    print("\nF1 -- excess kurtosis of daily returns at equal mean jump rate")
    s = 0.03
    Jc = jh.JumpSizes(price_std=s)
    for label, proc in (("Hawkes", H), ("Poisson", P)):
        sim = jh.simulate(jh.ConstantHost(0.04), proc, Jc, 2.0, 504, 4000, seed=5, rho=0.0)
        k, se = jh.excess_kurtosis(np.diff(sim["X"], axis=0).T)
        th = jh.excess_kurtosis_theory(0.04, proc, s, DAY)
        check(f"constant variance, {label}: simulated kurtosis matches the scale-mixture closed form",
              abs(k - th) < 4 * se, f"{k:.3f} +/- {se:.3f} vs {th:.3f}")
    ratio = jh.excess_kurtosis_theory(0.04, H, s, DAY) / jh.excess_kurtosis_theory(0.04, P, s, DAY)
    check("closed form: the kurtosis ratio IS the daily Fano factor",
          abs(ratio - hk.fano_factor(H, DAY)) < 1e-12, f"{ratio:.4f} = Fano {hk.fano_factor(H, DAY):.4f}")

    for host, J0 in hosts():
        J = jh.JumpSizes(state_mean=J0.state_mean, price_std=s)
        res = {}
        t0 = time.perf_counter()
        for label, proc in (("Hawkes", H), ("Poisson", P)):
            sim = jh.simulate(host, proc, J, 2.0, 504 * 4, 2000, seed=7, record_every=4)
            res[label] = jh.excess_kurtosis(np.diff(sim["X"], axis=0).T)
        (kh, sh), (kp, sp) = res["Hawkes"], res["Poisson"]
        z = (kh - kp) / math.hypot(sh, sp)
        check(f"{host.name}: Hawkes returns more kurtotic than Poisson by > 4 SE",
              z > 4.0, f"{kh:.2f} +/- {sh:.2f} vs {kp:.2f} +/- {sp:.2f} (z {z:+.1f}, {time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    print("=" * 74)
    print("Session F / F1 -- Hawkes jumps on OU, Heston and rough Heston")
    print("=" * 74)
    t0 = time.perf_counter()
    test_parameters()
    test_simulation_against_theory()
    test_intensity_peaks()
    test_likelihood()
    test_second_spike_exact()
    test_rough_jump_size()
    test_second_spike_monte_carlo()
    test_kurtosis()
    print("\n" + "=" * 74)
    print(f"{len(PASS)} passed, {len(FAIL)} failed   ({time.perf_counter()-t0:.0f}s)")
    for f in FAIL:
        print(f"  FAILED: {f}")
    print("=" * 74)
    sys.exit(1 if FAIL else 0)
