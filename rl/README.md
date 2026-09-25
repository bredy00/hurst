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
and §3 says what it is.

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

The reward is quadratic in the action, so its optimum is the vertex of a parabola. Hedged
Monte Carlo *solves for that vertex* by least squares, using every path at once and the
quadratic structure. The agent estimates each arm separately and compares neighbours, which
throws the structure away: the argmax has to resolve differences between adjacent actions
that are small next to the noise in `(dW)²`. Adding arms makes each one noisier, adding data
shrinks the noise but not the estimator's blindness to the shape. **When you know the
reward's structure, estimate it; the argmax is what you use when you do not.**

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
