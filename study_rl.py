"""
The reinforcement-learning framework, graded (Session M, 26 September 2026).

Akin asked for an RL layer on the parts of this project that need progressive learning --
Markov properties, the argmax of a return function, absorbing states -- off by default, for
someone who later wants to customise the code. The framework is `rl/`. This is the study
that says what it is worth, and it is built so it can say "less than what is already here",
because both problems it is pointed at have a known optimum:

  A  the answer key. A small MDP solved exactly by value iteration; every agent must recover
     that Q and that policy. An agent that cannot do this has no business on a hard problem,
     and this section runs first for that reason.
  M  the Markov property, tested rather than assumed, on both states.
  H  hedging. The agents against the Black-Scholes delta, against that delta snapped to the
     agents' own action grid (their real floor), and against Hedged Monte Carlo -- which
     solves this exact problem analytically by backward regression (Sessions J-K).
  F  filter selection at the zero boundary, learned offline from logged filter runs, against
     the written protocol of Session I and against the converged 64-substep particle filter.
  C  what the framework costs, beside the analytic machinery it sits next to.

    python study_rl.py             (A M H C; ~8 minutes)
    python study_rl.py F           (the filtering section; ~15 minutes, cached afterwards)
    python study_rl.py all
    -> captures/rl.json, captures/rl.log, captures/rl.png
"""

import json
import pathlib
import sys
import time

import numpy as np

import models.hedging as hd
import models.rough_heston as rh
import rl
import rl.agents as ag
import rl.hedging_env as he
import rl.mdp as mdp
from rl.markov import markov_test, report

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "captures" / "rl.json"
P_HEDGE = rh.RoughHestonParams(0.04, 2.0, 0.04, 0.3, -0.7, 0.12)
T_OPT, N_STEPS, K_STRIKE, EVERY = 1.0 / 12, 128, 1.0, 4
N_TRAIN, N_TEST, N_ACTIONS = 32_000, 16_000, 21


def say(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ A: the answer key
def section_a():
    """
    A five-state chain with a terminal reward and a shortcut, solved exactly, then learned.

    The chain is small enough that value iteration is the truth and every agent's Q can be
    compared to it entry by entry. The batch covers every (state, action) pair, so a failure
    here is the learner's and not the data's.
    """
    n_s, n_a = 5, 2
    P = np.zeros((n_s, n_a, n_s))
    R = np.zeros((n_s, n_a))
    for s in range(4):
        P[s, 0, s + 1] = 1.0                       # walk on
        R[s, 0] = -1.0
        P[s, 1, 4] = 1.0                           # jump to the end
        R[s, 1] = -3.0
    P[4, :, 4] = 1.0
    absorbing = np.array([False] * 4 + [True])
    M = mdp.TabularMDP(P, R, absorbing, gamma=1.0, action_names=("walk", "jump"))
    Q, V, pi, info = M.value_iteration()
    say(f"A  exact value iteration: V {np.round(V, 3).tolist()}, policy "
        f"{[M.action_names[a] for a in pi[:4]]} ({info['iterations']} iterations, "
        f"residual {info['residual']:.1e})")

    # a batch covering every (s, a), one-hot features so a linear agent is exactly tabular
    s_l, a_l, r_l, sp_l, ab_l, ep_l, st_l = [], [], [], [], [], [], []
    ep = 0
    for s in range(4):
        for a in range(n_a):
            for rep in range(200):
                sp = int(np.argmax(P[s, a]))
                s_l.append(s); a_l.append(a); r_l.append(R[s, a]); sp_l.append(sp)
                ab_l.append(bool(absorbing[sp])); ep_l.append(ep); st_l.append(0)
                ep += 1
    eye = np.eye(n_s)
    batch = ag.Batch(eye[s_l], np.array(a_l), np.array(r_l), eye[sp_l], np.array(ab_l), n_a,
                     episode=np.array(ep_l), step=np.array(st_l))
    out = {"V_exact": V.tolist(), "pi_exact": pi.tolist(), "agents": {}}
    for name, model in (("fitted-q", ag.FittedQ(sweeps=200, gamma=1.0, ridge=1e-10)),
                        ("lspi", ag.LSPI(iters=50, gamma=1.0, ridge=1e-10)),
                        ("tabular-q", ag.TabularQ(n_s, lambda f: int(np.argmax(f)), gamma=1.0,
                                                  epochs=300, alpha=0.6))):
        model.fit(batch)
        Qh = model.q(eye)
        err = float(np.max(np.abs(Qh[:4] - Q[:4])))
        same = bool(np.array_equal(mdp.greedy(Qh)[:4], pi[:4]))
        out["agents"][name] = {"max_Q_error": err, "policy_matches": same}
        say(f"A  {name:10s}: worst |Q - Q*| on the live states {err:.2e}; recovers the exact policy: {same}")
    return out


# ------------------------------------------------------------------ the hedging batch
def hedging_data(n_train=N_TRAIN, n_test=N_TEST, n_actions=N_ACTIONS, policy="grid-uniform"):
    tr = hd.simulate_paths(P_HEDGE, T_OPT, N_STEPS, n_train, seed=1)
    te = hd.simulate_paths(P_HEDGE, T_OPT, N_STEPS, n_test, seed=99)
    batch, grid, dates = he.build_batch(tr, K_STRIKE, EVERY, n_actions=n_actions, seed=0, policy=policy)
    price = float(rh.call_prices(np.array([0.0]), T_OPT, P_HEDGE)[0][0])
    return tr, te, batch, grid, dates, price


# ------------------------------------------------------------------ M: is the state Markov?
def section_m(batch, tr):
    feats, lags, tgt, names = he.markov_inputs(batch, tr, K_STRIKE, EVERY)
    res = markov_test(feats, lags, tgt, names)
    rep = report(res)
    say(f"M  hedging state (S, sigma_hat, tau), {len(feats)} transitions with a predecessor:")
    for line in rep["lines"]:
        say(f"     {line}")
    say(f"   verdict: {'Markov enough to bootstrap through' if rep['ok'] else 'NOT Markov'}")
    return {"hedging": {k: v for k, v in res.items()}, "hedging_ok": rep["ok"]}


# ------------------------------------------------------------------ H: hedging
def section_h(tr, te, batch, grid, dates, price):
    out = {"price": price, "n_train": tr["S"].shape[1], "n_test": te["S"].shape[1],
            "n_actions": len(grid), "every": EVERY}
    base = {}
    base["Black-Scholes delta"] = hd.hedge_error(te, K_STRIKE, EVERY, price, "bs_model")
    base["...snapped to the grid"] = he.grid_floor(te, K_STRIKE, EVERY, price, len(grid))
    t0 = time.perf_counter()
    fit = hd.hmc_fit(tr, K_STRIKE, EVERY)
    hmc_s = time.perf_counter() - t0
    base["Hedged Monte Carlo"] = hd.hedge_error(te, K_STRIKE, EVERY, price, "hmc", fit=fit)

    rows = {}
    for label, e in base.items():
        rows[label] = {"sd": float(e.std() / price), "seconds": hmc_s if "Monte Carlo" in label else 0.0}
    agents = {}
    for label, model in (("fitted-Q, myopic (gamma = 0)", ag.FittedQ(sweeps=1, gamma=0.0)),
                         ("fitted-Q, bootstrapped", ag.FittedQ(sweeps=30, gamma=1.0)),
                         ("LSPI", ag.LSPI(iters=15, gamma=1.0))):
        t0 = time.perf_counter()
        model.fit(batch)
        secs = time.perf_counter() - t0
        e = he.apply_policy(te, K_STRIKE, EVERY, he.agent_policy(model, grid), price)
        rows[label] = {"sd": float(e.std() / price), "seconds": secs,
                       "info": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                for k, v in model.info.items() if k != "rows_per_action"}}
        agents[label] = model
    say(f"H  one-month ATM call, rebalanced every {EVERY} steps of {N_STEPS}; "
        f"{out['n_train']} training and {out['n_test']} test paths, {len(grid)} hedge ratios")
    say(f"   {'rule':34s} {'residual sd / price':>20s} {'fit seconds':>12s}")
    for label in sorted(rows, key=lambda z: rows[z]["sd"]):
        say(f"   {label:34s} {rows[label]['sd']:20.4f} {rows[label]['seconds']:12.1f}")
    out["rules"] = rows

    # the argmax the agent chose against the hedge the analytic rule chose, state by state
    k = dates[len(dates) // 3]
    f = he.phi_features(te["S"][k], hd.model_sigma(te, k), te["tau"][k], K_STRIKE)
    d_bs = hd.bs_delta(te["S"][k], K_STRIKE, hd.model_sigma(te, k), te["tau"][k])
    best = agents["fitted-Q, myopic (gamma = 0)"]
    a_rl = grid[best.policy(f)]
    psi, chi, chi_m = hd._features(te["S"][k], hd.model_sigma(te, k), te["tau"][k], K_STRIKE)
    coef = fit["coefs"][k][1]
    a_hmc = chi @ coef[:chi.shape[1]]
    order = np.argsort(d_bs)
    q = [int(len(order) * f_) for f_ in (0.1, 0.3, 0.5, 0.7, 0.9)]
    out["policy_slice"] = {"step": int(k), "tau": float(te["tau"][k]),
                           "bs_delta": d_bs[order][q].tolist(), "hmc": a_hmc[order][q].tolist(),
                           "rl": a_rl[order][q].tolist()}
    say(f"   at step {k} (tau {te['tau'][k]:.3f} y), the hedge each rule chooses across the smile:")
    say("     BS delta  " + "  ".join(f"{v:6.3f}" for v in d_bs[order][q]))
    say("     HMC       " + "  ".join(f"{v:6.3f}" for v in a_hmc[order][q]))
    say("     fitted-Q  " + "  ".join(f"{v:6.3f}" for v in a_rl[order][q]))
    out["hmc_vs_rl_mean_abs"] = float(np.mean(np.abs(a_hmc - a_rl)))
    out["bs_vs_rl_mean_abs"] = float(np.mean(np.abs(d_bs - a_rl)))
    say(f"   mean |RL - HMC| {out['hmc_vs_rl_mean_abs']:.4f}, mean |RL - BS| {out['bs_vs_rl_mean_abs']:.4f} "
        f"(the grid's own step is {1 / (len(grid) - 1):.3f})")
    return out


def section_h_scaling(te, price):
    """Does the gap close with data, or with a finer grid? Both are measured, not argued."""
    out = {"paths": [], "actions": []}
    say("H2 how the agent's residual moves with the batch and with the action grid:")
    for n_paths in (2_000, 8_000, 32_000, 128_000):
        tr = hd.simulate_paths(P_HEDGE, T_OPT, N_STEPS, n_paths, seed=1)
        b, grid, _ = he.build_batch(tr, K_STRIKE, EVERY, n_actions=N_ACTIONS, seed=0)
        m = ag.FittedQ(sweeps=1, gamma=0.0).fit(b)
        e = he.apply_policy(te, K_STRIKE, EVERY, he.agent_policy(m, grid), price)
        out["paths"].append({"n_paths": n_paths, "rows_per_action": len(b) // N_ACTIONS,
                             "sd": float(e.std() / price)})
        say(f"     {n_paths:7d} paths ({len(b) // N_ACTIONS:6d} rows per action): {e.std() / price:.4f}")
    tr = hd.simulate_paths(P_HEDGE, T_OPT, N_STEPS, 32_000, seed=1)
    for na in (3, 6, 11, 21, 41):
        b, grid, _ = he.build_batch(tr, K_STRIKE, EVERY, n_actions=na, seed=0)
        m = ag.FittedQ(sweeps=1, gamma=0.0).fit(b)
        e = he.apply_policy(te, K_STRIKE, EVERY, he.agent_policy(m, grid), price)
        floor = he.grid_floor(te, K_STRIKE, EVERY, price, na)
        out["actions"].append({"n_actions": na, "sd": float(e.std() / price),
                               "grid_floor": float(floor.std() / price)})
        say(f"     {na:3d} hedge ratios: agent {e.std() / price:.4f}, BS on the same grid "
            f"{floor.std() / price:.4f}, difference {e.std() / price - floor.std() / price:+.4f}")
    return out


# ------------------------------------------------------------------ F: filter selection
# The second host has to be genuinely AWAY from the boundary or the protocol degenerates to
# "always cf" and there is nothing to select. Measured on the Kalman filter's own diagnostic:
# theta 0.045 / xi 0.3 flags 100% of days, 0.06 / 0.2 flags 94%, 0.08 / 0.15 flags 3.3%, and
# 0.09 / 0.12 flags none. The predictive sd has to sit below half the mean, which needs a
# much quieter host than the stationary sd sqrt(xi^2 theta / 2 kappa) alone suggests.
HOSTS = [("at the boundary (theta 0.02, xi 0.9)",
          rh.RoughHestonParams(0.02, 3.0, 0.02, 0.9, -0.7, 0.12),
          dict(kappa=3.0, theta=0.02, xi=0.9, R=1e-10)),
         ("away from it (theta 0.09, xi 0.12)",
          rh.RoughHestonParams(0.09, 3.0, 0.09, 0.12, -0.7, 0.12),
          dict(kappa=3.0, theta=0.09, xi=0.12, R=1e-10))]


def section_f(n_days=250, seeds=(1, 2, 3), cost_weight=1.0):
    import rl.filter_env as fe
    import rl.filter_logs as fl
    say("F  logging each filter over every series (the expensive part; cached afterwards)")
    logs = fl.build_logs(HOSTS, n_days=n_days, seeds=seeds)
    batch = fe.build_batch(logs, cost_weight=cost_weight)
    feats, lags, tgt, names = fe.markov_inputs(batch)
    rep = report(markov_test(feats, lags, tgt, names))
    say(f"   the filtering state, {len(feats)} transitions with a predecessor:")
    for line in rep["lines"]:
        say(f"     {line}")

    # the chosen filter does not change tomorrow's state either -- the features come from the
    # Kalman filter's own run, whichever filter estimated the day -- so this is a bandit too,
    # and the myopic agent is the one the structure calls for
    boot = ag.FittedQ(sweeps=40, gamma=1.0).fit(batch)
    myopic = ag.FittedQ(sweeps=1, gamma=0.0).fit(batch)
    same_next = float(np.max(np.abs(batch.phi_next[batch.action == 0]
                                    - batch.phi_next[batch.action == 1])))
    say(f"   the next state is identical across actions to {same_next:.1e}: a bandit, as the hedge was")
    policies = {
        "always kalman": lambda f: fe.always("kalman", len(f)),
        "always cf": lambda f: fe.always("cf", len(f)),
        "always particle (16)": lambda f: fe.always("particle", len(f)),
        "the Session I protocol": fe.protocol_policy,
        "learned, bootstrapped": boot.policy,
        "learned, myopic": myopic.policy,
    }
    rows = {}
    for label, fn in policies.items():
        rows[label] = fe.score(logs, fn, cost_weight=cost_weight)
    say(f"   {len(logs)} series x {n_days} days; reward = day's log-likelihood - "
        f"{cost_weight} x seconds. The reference is the 64-substep particle filter.")
    say(f"   {'policy':24s} {'loglik':>10s} {'seconds':>9s} {'objective':>11s} "
        f"{'nats/day vs reference':>22s}  filter shares")
    for label in sorted(rows, key=lambda z: -rows[z]["objective"]):
        r = rows[label]
        share = " ".join(f"{k[:4]} {100 * v:.0f}%" for k, v in r["share"].items() if v > 0)
        gap = f"{r.get('gap_per_day', float('nan')):+.3f}"
        say(f"   {label:24s} {r['loglik']:10.1f} {r['seconds']:9.1f} {r['objective']:11.1f} "
            f"{gap:>22s}  {share}")
    return {"rows": rows, "markov": rep["lines"], "markov_ok": rep["ok"], "next_state_spread": same_next,
            "hosts": [h[0] for h in HOSTS], "n_days": n_days, "seeds": list(seeds),
            "cost_weight": cost_weight}


# ------------------------------------------------------------------ C: what it costs
def section_c(batch):
    out = {}
    t0 = time.perf_counter()
    tr = hd.simulate_paths(P_HEDGE, T_OPT, N_STEPS, 32_000, seed=1)
    out["simulate_32k_paths_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    hd.hmc_fit(tr, K_STRIKE, EVERY)
    out["hmc_fit_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    he.build_batch(tr, K_STRIKE, EVERY, n_actions=N_ACTIONS, seed=0)
    out["build_batch_s"] = time.perf_counter() - t0
    for label, model in (("fitted-q myopic", ag.FittedQ(sweeps=1, gamma=0.0)),
                         ("fitted-q bootstrapped", ag.FittedQ(sweeps=30, gamma=1.0)),
                         ("lspi", ag.LSPI(iters=15, gamma=1.0))):
        t0 = time.perf_counter()
        model.fit(batch)
        out[f"{label}_fit_s"] = time.perf_counter() - t0
    say("C  cost of the same hedging answer, on 32 000 paths:")
    say(f"     simulating the paths          {out['simulate_32k_paths_s']:6.1f} s  (both pay this)")
    say(f"     Hedged Monte Carlo            {out['hmc_fit_s']:6.1f} s  -> the analytic optimum")
    say(f"     building the RL batch         {out['build_batch_s']:6.1f} s")
    for label in ("fitted-q myopic", "fitted-q bootstrapped", "lspi"):
        say(f"     {label:29s} {out[f'{label}_fit_s']:6.1f} s")
    rl_total = out["build_batch_s"] + out["fitted-q myopic_fit_s"]
    out["rl_over_hmc"] = rl_total / max(out["hmc_fit_s"], 1e-9)
    say(f"   the cheapest agent costs {out['rl_over_hmc']:.1f}x Hedged Monte Carlo and does not reach it.")
    return out


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ui import theme
    tok = theme.apply("light")
    cat = [theme.series(i) for i in range(8)]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    a = ax[0]
    rows = res["H"]["rules"]
    order = sorted(rows, key=lambda z: rows[z]["sd"])
    colors = [cat[5] if "Monte Carlo" in z else (cat[0] if "fitted-Q" in z or "LSPI" in z else tok["muted"])
              for z in order]
    vals = [rows[z]["sd"] for z in order]
    a.barh(range(len(order)), vals, color=colors)
    for i, v in enumerate(vals):          # bars start at zero, which is honest and compresses
        a.text(v - 0.005, i, f"{v:.4f}", va="center", ha="right", fontsize=8, color="white")
    a.set_yticks(range(len(order)))
    a.set_yticklabels([z.replace("fitted-Q, ", "fitted-Q\n") for z in order], fontsize=7.5)
    a.invert_yaxis()
    a.set_xlabel("residual sd / price  (lower is better)")
    a.set_title("Hedging: learned against derived")
    a.grid(axis="y", visible=False)

    a = ax[1]
    sc = res["H2"]["actions"]
    na = [r["n_actions"] for r in sc]
    a.plot(na, [r["sd"] for r in sc], "o-", color=cat[0], label="the agent")
    a.plot(na, [r["grid_floor"] for r in sc], "s--", color=tok["muted"], label="BS delta on the same grid")
    a.axhline(res["H"]["rules"]["Hedged Monte Carlo"]["sd"], color=cat[5], lw=1.4, ls="-",
              label="Hedged Monte Carlo")
    a.set_xscale("log")
    a.set_xticks(na)
    a.set_xticklabels([str(v) for v in na])
    a.xaxis.set_minor_locator(plt.NullLocator())    # the log minor ticks land on the labels
    a.set_xlabel("hedge ratios on offer")
    a.set_ylabel("residual sd / price")
    a.set_title("The grid is not what binds")
    a.legend(fontsize=8)

    a = ax[2]
    sl = res["H"]["policy_slice"]
    x = np.arange(len(sl["bs_delta"]))
    a.plot(x, sl["bs_delta"], "o-", color=tok["muted"], label="BS delta")
    a.plot(x, sl["hmc"], "s-", color=cat[5], label="HMC")
    a.plot(x, sl["rl"], "^-", color=cat[0], label="fitted-Q (on the grid)")
    a.set_xticks(x)
    a.set_xticklabels(["10%", "30%", "50%", "70%", "90%"])
    a.set_xlabel("path, by Black-Scholes delta")
    a.set_ylabel("hedge ratio")
    a.set_title(f"What each rule holds (tau {sl['tau']:.3f} y)")
    a.legend(fontsize=8)
    fig.suptitle("The reinforcement-learning framework, against the answers this project already has")
    fig.tight_layout()
    fig.savefig(ROOT / "captures" / "rl.png", dpi=120)


def main(argv):
    rl.enable()
    want = [a.upper() for a in argv] or ["A", "M", "H", "C"]
    if want == ["ALL"]:
        want = ["A", "M", "H", "C", "F"]
    res = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    t0 = time.perf_counter()
    need_hedging = {"M", "H", "C"} & set(want)
    if need_hedging:
        say(f"   building the hedging batch ({N_TRAIN} training paths)...")
        tr, te, batch, grid, dates, price = hedging_data()
    if "A" in want:
        res["A"] = section_a()
    if "M" in want:
        res["M"] = section_m(batch, tr)
    if "H" in want:
        res["H"] = section_h(tr, te, batch, grid, dates, price)
        res["H2"] = section_h_scaling(te, price)
    if "C" in want:
        res["C"] = section_c(batch)
    if "F" in want:
        res["F"] = section_f()
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    if "H" in res and "H2" in res:
        plot(res)
    say(f"\n[done in {time.perf_counter() - t0:.0f}s] -> {OUT}")
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
