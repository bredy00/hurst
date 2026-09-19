"""
Rough jump modes: jumps through the Volterra driver, or added directly to V?
(Session H decision brief; both modes built in Session F, models/jump_hosts.py.)

  A  response shapes: expected V after one jump (no excitation), normalised to the same
     extra variance over the first trading day; share of the 30-day impact inside the
     first hour / day / week; the 5-minute peak over the 1-day level
  B  what a driver jump is on a lift: immediate V impact and 1-day integrated impact per
     unit J across step sizes and lifts, against the exact continuous response
     J t^(alpha-1) E_{alpha,alpha}(-kappa t^alpha); a direct jump moves V by J, full stop
  C  the second-spike property on Session F's 21 (gap, window) pairs at branching ratios
     0.6 and 0.95, for driver (shipped and finer lift) and direct (decay 3/y and 25/y)
  D  can data tell them apart? An event study of daily realised variance after large-
     return days -- power law vs exponential decay of the excess -- on simulated 3-year
     histories of each mode, scored by how often the right shape wins. The same function
     runs on the recorded SPY history.

    python study_jump_modes.py        (~10 minutes) -> captures/jump_modes.json, .log

Session I adopted driver mode, sized by integrated impact, and retired direct mode from
the model. The two classes below keep the Session H semantics so this brief reproduces:
DriverIncrementHost reads jump sizes as raw driver increments, DirectJumpHost is the
retired mode, and the "shipped" rows run on the Sessions A-H lift (24 nodes to 1e5/y).
"""

import json
import math
import pathlib
import time

import numpy as np

import models.hawkes as hk
import models.jump_hosts as jh
import models.rough_heston as rh

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "captures" / "jump_modes.json"
KW = dict(kappa=3.0, theta=0.04, xi=0.3, H=0.12, v0=0.04)
BARS = 78
DAY = 1.0 / 252
QUIET = hk.HawkesParams.poisson(1e-9)          # a single planted event, nothing else arriving
SHIPPED = (24, 1e5)                             # the Sessions A-H lift, the "shipped" rows below


class DriverIncrementHost(jh.RoughHost):
    """
    Sessions F-H driver jumps: JumpSizes.state_mean is the raw driver increment J.
    Since Session I the model reads a rough jump's size as its integrated variance
    impact (models.jump_hosts.RoughHost); this class keeps the old reading so the
    comparison here reproduces its "per unit J" numbers. Not part of the model.
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.driver_per_impact = 1.0
        self.jump_mode = "driver"


class DirectJumpHost(jh.RoughHost):
    """
    The Sessions F-H "direct" mode, retired in Session I: a component D added to V,
    decaying at `jump_decay` (default kappa); the state is (N + 1, n), D last. Kept
    only so this comparison reproduces. Not part of the model.
    """

    def __init__(self, jump_decay=None, **kw):
        super().__init__(**kw)
        self.jump_mode = "direct"
        self.jump_decay = self.kappa if jump_decay is None else float(jump_decay)
        self.name = "rough Heston (variance, direct jumps)"

    def stationary_state(self, jump_mean, rate, dt):
        E, g = self._coef(dt)
        n = len(self.w)
        D = jump_mean * rate * dt / (1.0 - math.exp(-self.jump_decay * dt))
        A = np.eye(n) - np.diag(E) + self.kappa * dt * np.outer(g, self.w)
        b = g * (self.kappa * dt * (self.theta - self.v0 - D))
        return np.concatenate([np.linalg.solve(A, b), [D]])

    def _V(self, S):
        return self.v0 + self.w @ S[:-1] + S[-1]

    def step(self, S, dt, z_v, jump_sum):
        E, g = self._coef(dt)
        Vp = np.maximum(self._V(S), 0.0)
        drive = self.kappa * (self.theta - Vp) * dt + self.xi * np.sqrt(Vp * dt) * z_v
        out = np.empty_like(S)
        out[-1] = S[-1] * math.exp(-self.jump_decay * dt) + jump_sum
        out[:-1] = E[:, None] * S[:-1] + g[:, None] * drive[None, :]
        return out

    def mean_step(self, M, dt, expected_jump):
        E, g = self._coef(dt)
        V = self.v0 + self.w @ M[:-1] + M[-1]
        out = np.empty_like(M)
        out[-1] = M[-1] * math.exp(-self.jump_decay * dt) + expected_jump
        out[:-1] = E * M[:-1] + g * (self.kappa * (self.theta - V) * dt)
        return out


def say(msg):
    print(msg, flush=True)


def response(host, dt, days=30, lead=2, J=1e-3, hawkes=QUIET):
    n = lead + int(round(days / (dt / DAY))) + 1
    r = jh.shock_response(host, hawkes, jh.JumpSizes(state_mean=J), n * dt, n, lead)
    return r[lead + 1:] / J                                    # per unit J, from the first step after the jump


def mittag_leffler(z, a, b, terms=200):
    return sum(z ** k / math.gamma(a * k + b) for k in range(terms))


def exact_driver_integral(t, H=0.12, kappa=3.0):
    """int_0^t of the exact response to a unit driver impulse: t^alpha E_{alpha,alpha+1}(-kappa t^alpha)."""
    a = H + 0.5
    return t ** a * mittag_leffler(-kappa * t ** a, a, a + 1.0)


def section_a():
    dt = DAY / BARS
    variants = {
        "driver, shipped lift (N=24, 1e5)": (lambda: DriverIncrementHost(**KW), SHIPPED),
        "driver, finer lift (N=40, 1e8)": (lambda: DriverIncrementHost(**KW), (40, 1e8)),
        "direct, decay kappa = 3/y": (lambda: DirectJumpHost(**KW), SHIPPED),
        "direct, decay 25/y (10-day e-fold)": (lambda: DirectJumpHost(**KW, jump_decay=25.0), SHIPPED),
        "Heston host (reference)": (lambda: jh.HestonHost(kappa=3.0, theta=0.04, xi=0.3), SHIPPED),
    }
    out = {}
    for name, (make, lift) in variants.items():
        with rh.using_lift(*lift):
            r = response(make(), dt)
        cum = np.cumsum(r) * dt
        day1 = cum[BARS - 1]
        total = cum[-1]
        row = {"impact_1d_per_J": float(day1), "share_1h": float(cum[11] / total), "share_1d": float(day1 / total),
               "share_1w": float(cum[5 * BARS - 1] / total), "peak_over_1d_level": float(r[0] / r[BARS - 1]),
               "level_1w_over_1d": float(r[5 * BARS - 1] / r[BARS - 1]), "level_1m_over_1d": float(r[-1] / r[BARS - 1]),
               "shape_days": (np.arange(1, len(r) + 1) * dt / DAY)[::6].tolist(), "shape": (r / day1 * DAY)[::6].tolist()}
        out[name] = row
        say(f"A  {name}: of the 30-day impact {100*row['share_1h']:.1f}% lands in the first hour, {100*row['share_1d']:.1f}% "
            f"in the first day, {100*row['share_1w']:.1f}% in the first week; V at 5 min / V at 1 day = "
            f"{row['peak_over_1d_level']:.1f}; at 1 week {row['level_1w_over_1d']:.2f}, at 30 days {row['level_1m_over_1d']:.2f} of the 1-day level")
    return out


def section_b():
    out = {"exact_1d_integral_per_J": exact_driver_integral(DAY)}
    for lift_name, (N, eta) in (("shipped (N=24, 1e5)", (24, 1e5)), ("finer (N=40, 1e8)", (40, 1e8))):
        for step_name, dt in (("5 min", DAY / 78), ("1 h", DAY / 6.5), ("1/4 day", DAY / 4)):
            with rh.using_lift(N, eta):
                host = DriverIncrementHost(**KW)
                r = response(host, dt, days=2)
            k1d = int(round(DAY / dt))
            row = {"immediate_V_per_J": float(r[0]), "impact_1d_per_J": float(np.sum(r[:k1d]) * dt)}
            out[f"{lift_name}, step {step_name}"] = row
            say(f"B  driver jump, {lift_name}, step {step_name}: V moves {row['immediate_V_per_J']:.1f} J at once; "
                f"1-day integrated impact {row['impact_1d_per_J']:.5f} J (exact continuous {out['exact_1d_integral_per_J']:.5f} J)")
    with rh.using_lift(*SHIPPED):
        host = DirectJumpHost(**KW)
        r = response(host, DAY / 78, days=2)
    out["direct, any step"] = {"immediate_V_per_J": float(r[0]), "impact_1d_per_J": float(np.sum(r[:78]) * DAY / 78)}
    say(f"B  direct jump: V moves {r[0]:.3f} J at once at any step; 1-day impact {out['direct, any step']['impact_1d_per_J']:.5f} J")
    return out


def section_c():
    DT = 1.0 / (252 * 4)
    grid = [(d, w) for d in (2, 4, 8, 20, 40, 80) for w in (1, 2, 4, 8, 20, 40) if w < d]
    procs = {"n = 0.6": hk.HawkesParams(mu=5.0, alpha=150.0, beta=250.0),
             "n = 0.95": hk.HawkesParams(mu=12.5 * 0.05, alpha=0.95 * 250.0, beta=250.0)}
    variants = {"driver, shipped lift": (lambda: DriverIncrementHost(**KW), SHIPPED),
                "driver, finer lift": (lambda: DriverIncrementHost(**KW), (40, 1e8)),
                "direct, decay 3/y": (lambda: DirectJumpHost(**KW), SHIPPED),
                "direct, decay 25/y": (lambda: DirectJumpHost(**KW, jump_decay=25.0), SHIPPED)}
    out = {}
    for vname, (make, lift) in variants.items():
        for pname, proc in procs.items():
            def run():
                host = make()
                J = jh.JumpSizes(state_mean=0.002 if host.jump_mode == "driver" else 0.01)
                return [(d, w) for d, w in grid if (lambda a: a["inc2"] > a["inc1"])(jh.second_spike(host, proc, J, DT, d, w))]
            if lift:
                with rh.using_lift(*lift):
                    held = run()
            else:
                held = run()
            out[f"{vname}, {pname}"] = {"held": len(held), "of": len(grid),
                                         "pairs_quarter_days": [list(p) for p in held]}
            say(f"C  {vname}, {pname}: second spike larger on {len(held)} of {len(grid)} (gap, window) pairs"
                + (f"; gaps held: {sorted({d / 4 for d, _ in held})} days" if held else ""))
    return out


# --------------------------------------------------------------- D: event study
def daily_from_path(X, bars):
    """Daily returns and realised variance (annualised) from a log-price path sampled `bars` times a day."""
    n_days = (len(X) - 1) // bars
    Xd = X[: n_days * bars + 1]
    inc = np.diff(Xd)
    R = Xd[bars::bars] - Xd[:-1:bars][:n_days]
    RV = 252.0 * (inc.reshape(n_days, bars) ** 2).sum(axis=1)
    return R, RV


def event_study(R, RV, horizon=30, z=3.0, trail=60, base=20, gap=10):
    """
    Mean excess realised variance after large-return days, s = 1..horizon, and which
    decay shape fits its logarithm better: power law a - p log s or exponential a - lam s.
    """
    events, last = [], -10 ** 9
    for t in range(trail, len(R) - horizon):
        sd = np.std(R[t - trail:t])
        if abs(R[t]) > z * sd and t - last > gap:
            events.append(t)
            last = t
    if len(events) < 3:
        return {"n_events": len(events), "winner": None}
    post = np.array([RV[t + 1:t + 1 + horizon] for t in events]).mean(axis=0)
    pre = float(np.mean([RV[t - base:t].mean() for t in events]))
    e = post / pre - 1.0
    s = np.arange(1, horizon + 1, dtype=float)
    ok = e > 0
    if ok.sum() < 6:
        return {"n_events": len(events), "winner": None, "excess": e.tolist()}
    ls, le = np.log(s[ok]), np.log(e[ok])
    cp = np.polyfit(ls, le, 1)
    ce = np.polyfit(s[ok], le, 1)
    sse_p = float(np.sum((le - np.polyval(cp, ls)) ** 2))
    sse_e = float(np.sum((le - np.polyval(ce, s[ok])) ** 2))
    return {"n_events": len(events), "excess": e.tolist(), "power_p": float(-cp[0]), "exp_rate_per_day": float(-ce[0]),
            "sse_power": sse_p, "sse_exp": sse_e, "winner": "power" if sse_p < sse_e else "exponential"}


def section_d(n_hist=12, years=3):
    hawkes = hk.HawkesParams(mu=5.0, alpha=150.0, beta=250.0)
    dt = DAY / BARS
    with rh.using_lift(*SHIPPED):                 # the hosts read the lift at construction
        direct = DirectJumpHost(**KW, jump_decay=25.0)
        driver = DriverIncrementHost(**KW)
    # match the extra variance over the first day of a jump: J_driver = J_direct * direct / driver per unit J
    k1d = BARS
    per_direct = float(np.sum(response(direct, dt, days=2)[:k1d]) * dt)
    per_driver = float(np.sum(response(driver, dt, days=2)[:k1d]) * dt)
    J_direct = 0.02
    J_driver = J_direct * per_direct / per_driver
    out = {"J_direct": J_direct, "J_driver": J_driver, "histories_per_mode": n_hist, "years": years}
    n_steps = int(years * 252 * BARS)
    for name, host, J in (("direct (25/y)", direct, J_direct), ("driver (shipped lift)", driver, J_driver)):
        t0 = time.perf_counter()
        sim = jh.simulate(host, hawkes, jh.JumpSizes(state_mean=J, price_mean=-0.01, price_std=0.02),
                          years, n_steps, n_hist, seed=41, rho=-0.7)
        rows, wins = [], 0
        pooled_R, pooled_RV = [], []
        for i in range(n_hist):
            R, RV = daily_from_path(sim["X"][:, i], BARS)
            es = event_study(R, RV)
            rows.append({k: es.get(k) for k in ("n_events", "winner", "power_p", "exp_rate_per_day")})
            want = "exponential" if "direct" in name else "power"
            wins += es.get("winner") == want
            pooled_R.append(R)
            pooled_RV.append(RV)
        big = event_study(np.concatenate(pooled_R), np.concatenate(pooled_RV))
        out[name] = {"per_history": rows, "right_shape": wins, "pooled_36y": {k: big.get(k) for k in
                     ("n_events", "winner", "power_p", "exp_rate_per_day", "sse_power", "sse_exp", "excess")}}
        say(f"D  {name}: the right decay shape wins on {wins} of {n_hist} three-year histories "
            f"(events per history {np.median([r['n_events'] for r in rows]):.0f}); pooled over {n_hist * years} years: "
            f"{big.get('winner')} (power p {big.get('power_p', float('nan')):.2f}, exp rate {big.get('exp_rate_per_day', float('nan')):.3f}/day); "
            f"{time.perf_counter()-t0:.0f}s")
    return out


def main():
    res = {}
    for key, fn in (("A", section_a), ("B", section_b), ("C", section_c), ("D", section_d)):
        t0 = time.perf_counter()
        res[key] = fn()
        say(f"[{key} done in {time.perf_counter()-t0:.0f}s]")
        OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


if __name__ == "__main__":
    main()


def plot(res=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = res or json.loads(OUT.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))
    cols = {"driver, shipped lift (N=24, 1e5)": "#1f77b4", "driver, finer lift (N=40, 1e8)": "#17becf",
            "direct, decay kappa = 3/y": "#d62728", "direct, decay 25/y (10-day e-fold)": "#ff7f0e",
            "Heston host (reference)": "0.5"}
    a = ax[0, 0]
    for name, row in res["A"].items():
        d, s = np.array(row["shape_days"]), np.array(row["shape"])
        ok = s > 0
        a.loglog(d[ok], s[ok], color=cols.get(name, "k"), lw=1.6, label=name)
    a.set_xlabel("days after the jump")
    a.set_ylabel("V response (same extra variance over day 1)")
    a.set_title("(a) one jump, no excitation: power law vs exponential decay")
    a.legend(fontsize=7)
    a = ax[0, 1]
    B = res["B"]
    labels, imm, integ = [], [], []
    for k, v in B.items():
        if k.startswith(("shipped", "finer")):
            labels.append(k.replace(", step", "\nstep"))
            imm.append(v["immediate_V_per_J"])
            integ.append(v["impact_1d_per_J"] / B["exact_1d_integral_per_J"])
    x = np.arange(len(labels))
    a.bar(x - 0.2, imm, 0.4, color="#1f77b4", label="V moves at once (x J)")
    a2 = a.twinx()
    a2.plot(x + 0.2, integ, "o", color="#d62728", label="1-day impact / exact")
    a2.set_ylim(0.9, 1.05)
    a.set_xticks(x)
    a.set_xticklabels(labels, fontsize=7)
    a.set_ylabel("immediate V move per unit driver jump")
    a2.set_ylabel("1-day integrated impact / exact continuous")
    a.set_title("(b) a driver jump has no size, only an integrated impact")
    a.legend(loc="upper left", fontsize=8)
    a2.legend(loc="upper right", fontsize=8)
    a = ax[1, 0]
    C = res["C"]
    names = ["driver, shipped lift", "driver, finer lift", "direct, decay 3/y", "direct, decay 25/y"]
    for i, n in enumerate(("n = 0.6", "n = 0.95")):
        a.bar(np.arange(4) + (i - 0.5) * 0.38, [C[f"{v}, {n}"]["held"] for v in names], 0.38,
              color=["#9ecae1", "#08519c"][i], label=f"branching ratio {n[4:]}")
    a.set_xticks(np.arange(4))
    a.set_xticklabels(names, fontsize=8)
    a.set_ylabel("(gap, window) pairs with a larger second spike, of 21")
    a.set_title("(c) the second-spike property by jump mode")
    a.legend(fontsize=8)
    a = ax[1, 1]
    D = res["D"]
    for name, c in (("direct (25/y)", "#ff7f0e"), ("driver (shipped lift)", "#1f77b4")):
        e = np.array(D[name]["pooled_36y"]["excess"])
        s = np.arange(1, len(e) + 1)
        ok = e > 0
        a.loglog(s[ok], e[ok], "o-", ms=3, color=c,
                 label=f"{name}: {D[name]['right_shape']}/{D['histories_per_mode']} 3-y histories pick the right shape")
    a.set_xlabel("days after a large-return day")
    a.set_ylabel("excess realised variance (36 simulated years)")
    a.set_title("(d) the test for the recorded history: event study of RV")
    a.legend(fontsize=7)
    fig.suptitle("Rough jump modes: through the Volterra driver vs added directly to V", fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "captures" / "jump_modes.png", dpi=120)
