# `rl/` — reinforcement learning, off by default

    import rl
    rl.enable()             # or VOLSURF_RL=1, or `python volsurf.py rl`

**What it is for.** Two things in this project are already dynamic programs with known
optima — the discrete hedge (Sessions J–K) and the zero-boundary filter protocol
(Session I). This package lets you learn those policies instead of deriving them, on the
same paths, so the comparison is like for like. It exists because someone customising this
code may want the learning route; it is off by default because on both problems the derived
answer is better and cheaper, and this file says by how much.

**The honest headline.** On the hedging problem, the best agent here leaves a residual of
**0.354** of the option price against Hedged Monte Carlo's **0.305** — 16% worse — after
which more data does not help and a finer action grid does not help. That gap is structural,
and §3 says what it is (corrected in Session N: not the argmax).

---

## 1. The three primitives

| | where | what it is for |
|---|---|---|
| Markov property | `rl.markov` | An MDP's one assumption, tested on logged trajectories before any learning. A nested F test on whether lagged information still predicts, with the sample size discounted for serially correlated residuals, and the lag block's partial R² reported beside its p-value — significance is not size. |
| argmax of the return function | `rl.mdp.greedy` | The policy is `argmax_a Q(s, a)`. Ties break deterministically on the lowest index, and a tolerance can widen a tie so a fitted Q does not report a policy that flips between two indistinguishable actions. |
| absorbing states | `rl.mdp.TabularMDP` | Carried explicitly, not as a zero-reward self-loop. `Q(s, a) = 0` there, the terminal payoff is earned on the transition *into* the state, and `value_iteration` refuses `gamma = 1` when the absorbing set is unreachable rather than diverging quietly. |

## 2. The answer key comes first

`study_rl.py A` builds a five-state chain whose value has a closed form, solves it exactly
by value iteration, and requires every agent to recover that Q and that policy:

| agent | worst \|Q − Q*\| | recovers the optimal policy |
|---|---|---|
| fitted-Q | 5.6e-13 | yes |
| LSPI | 9.3e-13 | yes |
| tabular-Q | 3.8e-14 | yes |

Run this before pointing an agent at anything hard. An agent that cannot solve a problem
with a known answer is not evidence about a problem without one.

## 3. Hedging: what the agents actually do

A one-month at-the-money call, rebalanced every 4 steps of 128, 32 000 training and 16 000
test paths, 21 hedge ratios on offer:

| rule | residual sd / price | fit |
|---|---|---|
| **Hedged Monte Carlo** (the analytic optimum) | **0.3050** | 0.6 s |
| fitted-Q, myopic (γ = 0) | 0.3543 | 0.4 s |
| fitted-Q, bootstrapped (γ = 1) | 0.3568 | 8.7 s |
| LSPI | 0.3573 | 10.4 s |
| Black–Scholes delta | 0.3582 | — |
| …snapped to the agents' own grid | 0.3596 | — |

Four things follow, and the third is the one worth keeping.

**The agent learns something real.** It beats the Black–Scholes delta, and the policy says
why. At τ = 0.057 y, across the smile:

```
BS delta   0.224   0.412   0.549   0.670   0.802
HMC        0.133   0.329   0.475   0.595   0.715
fitted-Q   0.150   0.300   0.400   0.500   0.650
```

HMC holds *less* than the Black–Scholes delta — that is the leverage adjustment, ρ = −0.7,
a fall in the price raising volatility. The agent goes the same way and gets about half way:
mean |RL − HMC| 0.062 against mean |RL − BS| 0.124, with the grid's own step at 0.050.

**The problem is a bandit, and bootstrapping costs you.** The writer's hedge does not move
the market, so the next state does not depend on the action (asserted in `test_rl.py` at
4 SE). Then `Q(s, a) = R(s, a) + γ·E[V(s′)|s]` and the second term is the same for every
action, so `argmax_a Q = argmax_a R`. The myopic agent is both better (0.3543 against
0.3568) and **twenty times cheaper** (0.4 s against 8.7 s) than the bootstrapped one. This
is the clearest practical lesson here: check whether your MDP is an MDP before paying for
one.

**The gap to the optimum is structural, not a budget.** Neither more data nor a finer grid
closes it:

| training paths | 2 000 | 8 000 | 32 000 | 128 000 |
|---|---|---|---|---|
| residual | 0.3533 | 0.3561 | 0.3543 | 0.3569 |

| hedge ratios | 3 | 6 | 11 | 21 | 41 |
|---|---|---|---|---|---|
| agent | 0.4663 | 0.3816 | 0.3612 | 0.3543 | 0.3550 |
| BS on the same grid | 0.4955 | 0.3902 | 0.3661 | 0.3596 | 0.3581 |

**It is not the argmax** — Session M said it was, and Session N measured otherwise. An agent
that fits Q as an explicit quadratic in the action and acts at the vertex (`quadratic-q`)
scores 0.3541 against the argmax's 0.3543. `study_rl.py D` attributes the whole gap:

| step | residual / price | share of the gap |
|---|---|---|
| fitted-Q, argmax | 0.3543 | |
| quadratic-Q, at the vertex | 0.3541 | 0.4% |
| known-reward regression (Black–Scholes mark) | 0.3276 | **53.7%** |
| …with a martingale mark | 0.3091 | **37.4%** |
| …one coefficient per date | 0.3068 | 4.8% |
| …on Hedged Monte Carlo's basis | 0.3048 | 3.9% |
| Hedged Monte Carlo | 0.3050 | |

The agent learns the reward −(ΔC − aΔS)² as a black box; a regression of ΔC on basis·ΔS
(`hedging_env.known_reward_fit`) uses its form and never models E[ΔS² | s]. And the
environment's Black–Scholes mark is not a martingale under the rough model, so one-step
risks do not add up to the total; `build_batch(..., mark=hedging_env.hmc_mark(fit, paths, K))`
swaps in one that is. Run the other way round — the martingale mark given to the agents —
the mark alone is worth 44.3% and the known reward alone 54.2%, additive to within 6.9%.
**When you know the reward's structure, estimate it; and make the mark a martingale before
you sum one-step rewards.**

**The state is only approximately Markov.** On 960 000 transitions the lag block is
statistically significant for every target and explains 0.02%, 0.65% and 0.21% of the
remaining variance. The true Markov state of the lifted model is the 44 factors, not
(S, σ̂, τ) — this is that leakage, measured. It is small enough to bootstrap through, which
is what the verdict says, and it is one more reason the myopic agent does no worse.

## 4. Filter selection

Learned offline from logged filter runs: each filter is run once over each series, its
per-day log-likelihood and cost recorded (`filters.*` gained a `loglik_t` array for this,
which is a useful diagnostic in its own right), and the MDP is built over that table.

| policy | objective | what it chose |
|---|---|---|
| always cf | 5377.5 | cf 100% |
| **the Session I protocol** | **5374.0** | Kalman 50%, cf 50% |
| learned (bootstrapped and myopic are identical) | 5332.3 | Kalman 5%, cf 69%, particle 26% |
| always Kalman | 3706.0 | Kalman 100% |
| always particle (16 substeps) | −31756.2 | particle 100% |

The written rule is within 0.07% of the best policy available. The learned one takes the
16-substep particle filter on a quarter of days, which is ruinous at the boundary; a linear
Q over five features cannot represent that cliff, while the protocol encodes it as a
threshold because Session I diagnosed its cause. **A rule derived from a mechanism beat a
rule fitted to the data, on the data.**

This problem is a bandit too, and provably: the state comes from the Kalman filter's own run
whichever filter estimated the day, so the next state is identical across actions to
**0.0e+00** and the two agents return byte-identical policies.

**The caveat, stated before the result:** each day's likelihood is credited to the filter
that produced it in *its own full run*, so a learned switching policy is an upper bound on
what a real implementation could reach — filters carry state, and switching means
re-initialising from a posterior the next filter does not represent exactly.
`switch_penalty` exists to charge for that.

## 4b. Transaction costs: the first genuine MDP (Session O)

`rl/cost_env.py`. With a proportional cost on trading, the holding carried forward is the
action, so the next state depends on it — not a bandit — and the optimum is a no-trade band.
Under Black–Scholes the Bellman equation is solved on a grid (`solve_dp`), which is the answer
key; `KnownCostFQI` keeps the known cost exact and learns the rest from transitions.

| 32 dates, 20 bp | objective / premium | gap to the DP |
|---|---|---|
| **the Bellman equation on a grid** | **−0.4460** | |
| known cost, risk by moments | −0.4469 | **−0.0009 ± 0.0001** |
| known cost, myopic | −0.4495 | −0.0035 |
| generic fitted-Q (black box) | −0.5367 | −0.0906 |
| Black–Scholes delta, every date | −0.4630 | −0.0170 |
| Whalley–Wilmott band | −0.5349 | −0.0889 |

The learner that keeps the cost exact **reaches the optimum**, model-free, and bootstrapping
pays (+0.0026 here, +0.118 at 128 dates). Whalley–Wilmott's closed-form band loses to plain
delta hedging at 32 dates, because its half-width is smaller than the delta's move between
dates; at 128 dates it wins over the delta and is still 10% short of the DP. Full table and
the basis finding at 128 dates: `docs/rl-framework.md`.

```python
import rl, rl.cost_env as ce
rl.enable()
P = ce.CostProblem(cost=0.002, n_dates=32)
dp = ce.solve_dp(P)                                            # the answer key
tr = ce.transitions(P, ce.simulate(P, 20000, seed=1))          # random-action logs
agent = ce.KnownCostFQI(P).fit(tr)
te = ce.simulate(P, 20000, seed=99)
print(ce.evaluate(P, te, agent.rule())["J"], ce.evaluate(P, te, dp["policy"])["J"])
```

## 5. Adding your own agent

One decorator; `docs/customising.md` §4 has a working example.

```python
@rl.register_agent("my-agent")
class MyAgent:
    def fit(self, batch): ...
    def q(self, phi): ...          # (n, n_actions)
    def policy(self, phi): ...     # rl.mdp.greedy(self.q(phi))
```

Run it against the answer key first, and run `rl.markov.markov_test` on your state before
you bootstrap through it.
