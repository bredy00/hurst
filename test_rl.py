"""
Session M -- the reinforcement-learning framework (`rl/`).

  disabled      every entry point refuses until the framework is turned on, and the refusal
                names the switch
  mdp           value iteration is exact on a chain with a closed form; absorbing states
                have zero value; gamma = 1 with an unreachable terminal is REFUSED rather
                than run to a divergence; the argmax breaks ties deterministically
  agents        all three recover the exact Q and the exact policy on a solvable MDP, which
                is the answer key the hard problems have no version of
  markov        the test rejects a state that hides a lag, accepts one that does not, and
                its effective-n correction reacts to serially correlated residuals
  hedging       the environment's reward is the analytic one-step risk; the next state does
                NOT depend on the action (so the problem is a bandit); the grid floor is
                reachable; no look-ahead
  filtering     the offline batch carries one row per (day, filter); the protocol policy
                reproduces `filters.protocol`'s own threshold rule
  registry      a user agent joins by decorator and is scored like the rest

    python test_rl.py
"""

import math
import sys

import numpy as np

import rl
import rl.agents as ag
import rl.mdp as mdp

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def test_disabled():
    print("\noff by default")
    rl.disable()
    check("the framework is off in a fresh process", not rl.is_enabled())
    refused = []
    for what, fn in (("the MDP framework", mdp.require),
                     ("an agent", lambda: rl.agent("fitted-q")),
                     ("fitting", lambda: ag.FittedQ().fit(_toy_batch()))):
        try:
            fn()
            refused.append(None)
        except rl.Disabled as e:
            refused.append(str(e))
    check("every entry point refuses while it is off", all(r is not None for r in refused))
    check("...and the refusal names the switch",
          all("rl.enable()" in r and "VOLSURF_RL" in r for r in refused if r),
          (refused[0] or "")[:70] + "...")
    rl.enable()
    check("enable() turns it on", rl.is_enabled())


def _chain(n=5, step_cost=1.0, jump_cost=3.0):
    """A walk-or-jump chain whose value has a closed form: V(s) = -min(n-1-s, jump_cost)."""
    P = np.zeros((n, 2, n))
    R = np.zeros((n, 2))
    for s in range(n - 1):
        P[s, 0, s + 1] = 1.0
        R[s, 0] = -step_cost
        P[s, 1, n - 1] = 1.0
        R[s, 1] = -jump_cost
    P[n - 1, :, n - 1] = 1.0
    absorbing = np.array([False] * (n - 1) + [True])
    return mdp.TabularMDP(P, R, absorbing, gamma=1.0, action_names=("walk", "jump"))


def _toy_batch(n_per=50):
    M = _chain()
    n_s, n_a = M.n_states, M.n_actions
    eye = np.eye(n_s)
    s, a, r, sp = [], [], [], []
    for st in range(n_s - 1):
        for act in range(n_a):
            for _ in range(n_per):
                nxt = int(np.argmax(M.P[st, act]))
                s.append(st); a.append(act); r.append(M.R[st, act]); sp.append(nxt)
    return ag.Batch(eye[s], np.array(a), np.array(r), eye[sp],
                    np.asarray(M.absorbing)[sp], n_a, episode=np.arange(len(s)))


def test_mdp():
    print("\nthe MDP: exact values, absorbing states, the argmax")
    M = _chain(5, 1.0, 3.0)
    Q, V, pi, info = M.value_iteration()
    exact = np.array([-min(4 - s, 3.0) for s in range(5)])
    exact[4] = 0.0
    check("value iteration matches the closed form V(s) = -min(steps to the end, the jump's cost)",
          float(np.max(np.abs(V - exact))) < 1e-12,
          f"V {np.round(V, 3).tolist()} against {exact.tolist()}")
    check("an absorbing state has zero value under every action", float(np.max(np.abs(Q[4]))) == 0.0)
    check("the policy jumps only where jumping is cheaper",
          pi[0] == 1 and pi[1] == 0 and pi[2] == 0 and pi[3] == 0,
          f"{[M.action_names[a] for a in pi[:4]]} (at state 1 walking and jumping tie at -3; "
          "the tie-break takes the lower index)")
    Vpi = M.policy_evaluation(pi)
    check("policy evaluation of the greedy policy returns its value",
          float(np.max(np.abs(Vpi - V))) < 1e-10, f"{np.max(np.abs(Vpi - V)):.1e}")

    # gamma = 1 with no way out must be refused, not run to a divergence
    P = np.zeros((2, 1, 2)); P[0, 0, 0] = 1.0; P[1, 0, 1] = 1.0
    try:
        mdp.TabularMDP(P, np.array([[1.0], [0.0]]), [False, True], gamma=1.0).value_iteration()
        refused = False
    except ValueError as e:
        refused = "cannot reach an absorbing state" in str(e)
    check("gamma = 1 with an unreachable terminal is refused, not diverged", refused)

    Q2 = np.array([[1.0, 1.0, 0.5], [0.2, 0.9, 0.9]])
    check("the argmax breaks ties by the lowest index, deterministically",
          mdp.greedy(Q2).tolist() == [0, 1], f"{mdp.greedy(Q2).tolist()}")
    check("...and a tolerance widens the tie", mdp.greedy(np.array([[1.0, 0.99]]), tol=0.02).tolist() == [0])

    # building an MDP by counting logged transitions recovers the model it was logged from
    b = _toy_batch(400)
    s = np.argmax(b.phi, axis=1)
    sp = np.argmax(b.phi_next, axis=1)
    M2, info2 = mdp.from_transitions(s, b.action, b.reward, sp, 5, 2, M.absorbing, gamma=1.0)
    Q2, V2, pi2, _ = M2.value_iteration()
    check("an MDP counted from logged transitions recovers the exact values",
          float(np.max(np.abs(V2 - V))) < 1e-12 and np.array_equal(pi2[:4], pi[:4]),
          f"worst |V - V*| {np.max(np.abs(V2 - V)):.1e}, {info2['unvisited']} unvisited (s, a)")


def test_agents():
    print("\nthe agents against the answer key")
    M = _chain()
    Q, V, pi, _ = M.value_iteration()
    b = _toy_batch(200)
    eye = np.eye(5)
    for name, model in (("fitted-q", ag.FittedQ(sweeps=200, gamma=1.0, ridge=1e-10)),
                        ("lspi", ag.LSPI(iters=50, gamma=1.0, ridge=1e-10)),
                        ("tabular-q", ag.TabularQ(5, lambda f: int(np.argmax(f)), gamma=1.0,
                                                  epochs=300, alpha=0.6))):
        model.fit(b)
        Qh = model.q(eye)
        err = float(np.max(np.abs(Qh[:4] - Q[:4])))
        check(f"{name} recovers the exact Q on the live states (1e-8)", err < 1e-8, f"{err:.1e}")
        check(f"...and the exact policy", np.array_equal(mdp.greedy(Qh)[:4], pi[:4]))
    check("an unfitted agent's weights are not silently zero-valued",
          ag.FittedQ().W is None)


def test_markov():
    print("\nthe Markov property, tested rather than assumed")
    from rl.markov import lagged_design, markov_test, report
    rng = np.random.default_rng(0)
    n = 4000
    # a genuinely Markov chain: x_{t+1} depends on x_t alone
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + rng.standard_normal()
    ep = np.zeros(n, int)
    st = np.arange(n)
    rows, prev, f_t, f_prev = lagged_design(x[:, None], ep, st)
    tgt = x[np.minimum(rows + 1, n - 1)][:, None]
    res = markov_test(f_t, f_prev, tgt, ["next x"])
    check("a first-order chain passes", res["next x"]["markov"],
          f"the lag explains {100 * res['next x']['partial_r2']:.3f}%, p {res['next x']['p_effective']:.2f}")

    # a second-order chain seen through a first-order state must FAIL
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.5 * y[t - 1] - 0.45 * y[t - 2] + rng.standard_normal()
    rows, prev, g_t, g_prev = lagged_design(y[:, None], ep, st)
    tgt2 = y[np.minimum(rows + 1, n - 1)][:, None]
    res2 = markov_test(g_t, g_prev, tgt2, ["next y"])
    check("a second-order chain seen through a one-lag state fails", not res2["next y"]["markov"],
          f"the lag explains {100 * res2['next y']['partial_r2']:.1f}%, p {res2['next y']['p_effective']:.1e}")
    check("the report's verdict follows the per-target ones",
          report(res)["ok"] and not report(res2)["ok"])
    check("the effective sample size reacts to correlated residuals",
          res2["next y"]["n_effective"] != res2["next y"]["n"])

    # the pairing must survive a batch that is NOT stored in episode-then-time order
    ep2 = np.tile(np.arange(100), 40)
    st2 = np.repeat(np.arange(40), 100)
    rows_a, prev_a, _, _ = lagged_design(np.zeros((4000, 1)), ep2, st2)
    ok = bool(len(rows_a) == 3900 and np.all(ep2[rows_a] == ep2[prev_a])
              and np.all(st2[rows_a] == st2[prev_a] + 1))
    check("lag pairing is on (episode, step), not on row adjacency", ok,
          f"{len(rows_a)} pairs from a date-major batch of 4000 (row adjacency would find 0)")


def test_hedging_env():
    print("\nthe hedging environment")
    import models.hedging as hd
    import models.rough_heston as rh
    import rl.hedging_env as he
    P = rh.RoughHestonParams(0.04, 2.0, 0.04, 0.3, -0.7, 0.12)
    paths = hd.simulate_paths(P, 1 / 12, 64, 3000, seed=5)
    b, grid, dates = he.build_batch(paths, 1.0, 4, n_actions=11, seed=0)
    check("one transition per path per rebalancing date",
          len(b) == len(dates) * paths["S"].shape[1], f"{len(b)} = {len(dates)} x {paths['S'].shape[1]}")
    check("the terminal transitions are marked absorbing",
          int(b.absorbing.sum()) == paths["S"].shape[1], f"{int(b.absorbing.sum())}")
    check("...and their next-state features are zeroed, not left at a diverging moneyness",
          float(np.abs(b.phi_next[b.absorbing]).max()) == 0.0)
    check("the moneyness feature is finite everywhere", bool(np.all(np.isfinite(b.phi))),
          f"worst |m| {np.abs(b.phi[:, 3]).max():.1f} against the clip at {he.M_CLIP}")

    # the writer's hedge does not move the market: the next state must not depend on the action
    first = b.step == 0
    means = [b.phi_next[first & (b.action == a), 3].mean() for a in (0, 5, 10)]
    spread = float(np.max(means) - np.min(means))
    sd = float(b.phi_next[first, 3].std() / math.sqrt((first & (b.action == 0)).sum()))
    check("the next state does not depend on the action, so the problem is a bandit",
          spread < 4 * sd, f"spread across actions {spread:.4f} against {4 * sd:.4f} (4 SE)")

    # the reward is the analytic one-step risk -- compared PATH BY PATH, since the logging
    # policy gives each action a random subset of paths and two subset means differ by noise
    k, a_idx = dates[2], 5
    rows = np.flatnonzero((b.step == 2) & (b.action == a_idx))
    which = b.episode[rows]
    k2 = k + 4
    dC = hd.bs_call(paths["S"][k2], 1.0, hd.model_sigma(paths, k2), paths["tau"][k2]) \
        - hd.bs_call(paths["S"][k], 1.0, hd.model_sigma(paths, k), paths["tau"][k])
    dW = (dC - grid[a_idx] * (paths["S"][k2] - paths["S"][k]))[which]
    check("the reward is -(one-step wealth change)^2, the quantity the analytic hedge minimises",
          float(np.max(np.abs(b.reward[rows] + dW * dW))) < 1e-15,
          f"{len(rows)} paths took this action; worst difference "
          f"{np.max(np.abs(b.reward[rows] + dW * dW)):.1e}")

    # a policy that IS the analytic delta on the grid must reproduce the grid floor
    price = 0.0216
    floor = he.grid_floor(paths, 1.0, 4, price, 11)

    def snapped(f, k_):
        d = hd.bs_delta(paths["S"][k_], 1.0, hd.model_sigma(paths, k_), paths["tau"][k_])
        return grid[np.clip(np.rint(d * 10).astype(int), 0, 10)]

    same = he.apply_policy(paths, 1.0, 4, snapped, price)
    check("the grid floor is exactly what a grid-snapped analytic delta achieves",
          float(np.max(np.abs(same - floor))) < 1e-12)


def test_filter_env():
    print("\nthe filter-selection environment")
    import filters.protocol as proto
    import rl.filter_env as fe
    rng = np.random.default_rng(1)
    T = 120
    mean_V = np.abs(0.04 + 0.02 * rng.standard_normal(T))
    sd_V = 0.01 + 0.01 * rng.random(T)
    y = np.abs(mean_V + 0.01 * rng.standard_normal(T))
    f = fe.day_features(mean_V, sd_V, y, np.arange(T) / (T - 1))
    logs = [{"features": f,
             "loglik": {k: rng.standard_normal(T) for k in fe.FILTERS},
             "seconds": {"kalman": 0.1, "cf": 10.0, "particle": 100.0},
             "truth_loglik": rng.standard_normal(T)}]
    b = fe.build_batch(logs, cost_weight=1.0)
    check("one row per (day, filter): the counterfactual is a logged row, not a guess",
          len(b) == T * len(fe.FILTERS), f"{len(b)} = {T} x {len(fe.FILTERS)}")
    check("the last day of every filter's run is absorbing",
          int(b.absorbing.sum()) == len(fe.FILTERS))
    check("each filter's run is its own episode, so a lag pairs two days of one run",
          len(np.unique(b.episode)) == len(fe.FILTERS))

    # the policy form of the protocol must be the protocol
    km_flags = f[:, 1] <= 2.0
    pol = fe.protocol_policy(f, threshold=2.0)
    check("the protocol policy is exactly the Session I threshold rule",
          np.array_equal(pol == fe.FILTERS.index("cf"), km_flags),
          f"the boundary binds on {100 * km_flags.mean():.0f}% of days; "
          f"protocol.BOUNDARY_SHARE is {proto.BOUNDARY_SHARE}")
    s = fe.score(logs, lambda ff: fe.always("kalman", len(ff)), cost_weight=1.0)
    check("scoring an always-one-filter policy charges that filter's whole cost",
          abs(s["seconds"] - 0.1) < 1e-9 and s["share"]["kalman"] == 1.0,
          f"{s['seconds']:.3f} s, shares {s['share']}")
    check("...and reports the gap to the reference when the logs carry one", "gap_per_day" in s)


def test_registry():
    print("\nthe registry")

    @rl.register_agent("test-constant")
    class Constant:
        def __init__(self, which=0):
            self.which = which

        def fit(self, batch):
            self.n = batch.n_actions
            return self

        def q(self, phi):
            out = np.zeros((len(np.atleast_2d(phi)), self.n))
            out[:, self.which] = 1.0
            return out

        def policy(self, phi):
            return mdp.greedy(self.q(phi))

    m = rl.agent("test-constant", which=1).fit(_toy_batch(10))
    check("a user agent joins by decorator and is constructed by name",
          m.policy(np.eye(5)).tolist() == [1] * 5)
    check("...and it is listed", "test-constant" in rl.AGENTS)
    try:
        rl.agent("no-such-agent")
        missing = False
    except KeyError as e:
        missing = "registered" in str(e)
    check("an unknown name lists what is registered", missing)
    try:
        rl.register_agent("fitted-q")(Constant)
        clashed = False
    except ValueError:
        clashed = True
    check("registering over an existing name is refused", clashed)
    rl.AGENTS.pop("test-constant", None)


if __name__ == "__main__":
    print("=" * 74)
    print("Session M -- the reinforcement-learning framework")
    print("=" * 74)
    test_disabled()
    test_mdp()
    test_agents()
    test_markov()
    test_hedging_env()
    test_filter_env()
    test_registry()
    print("\n" + "=" * 74)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    print("=" * 74)
    sys.exit(1 if FAIL else 0)
