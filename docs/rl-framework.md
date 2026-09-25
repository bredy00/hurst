# A reinforcement-learning framework, and what it is worth here

*Session M, 26 September 2026. `rl/` (7 modules), `study_rl.py`, `test_rl.py` (41 checks),
four standing health checks, `captures/rl.{json,log,png}`. Quick reference: `rl/README.md`.
Revert: `git rm -r rl study_rl.py test_rl.py docs/rl-framework.md` — nothing else imports it.*

## What was asked for

> Introduce, for people who later want to mess with the code and customise it, a reinforced
> learning framework on top of the things that require progressive learning. It just takes
> some Markov properties and argmax of a generational function and absorbing states, and
> it's doable. So it's not gonna cut on the computational costs of it. And it should be
> disabled by default, but if the user plays around with it then he should be able to enable
> it and have access to the machine learning module that it was supposed to contradict to.

So: the three primitives, off by default, and pointed at the parts of the project that are
already progressive. The last clause is the one that shaped the design — the analytic
machinery here was built as the alternative to learning, so the framework's job is to make
the two comparable rather than to replace one with the other.

## Why this project can grade an agent honestly

Most RL write-ups compare agents to other agents. Both control problems here have a **known
optimum**, which is the difference between a finding and a demonstration:

| | the optimum | where it came from |
|---|---|---|
| discrete hedging | Föllmer–Schweizer local risk minimisation, solved by backward regression | `models/hedging.py`, Sessions J–K |
| filter selection | the written protocol, chosen against the converged 64-substep particle filter | `filters/protocol.py`, Session I |

And the framework starts with an **answer key**: a five-state chain whose value has a closed
form, solved exactly by value iteration. All three agents recover that Q to 1e-13 and the
exact policy. An agent that cannot do that is not evidence about a problem where nobody
knows the answer, so this section runs first.

## The three primitives

**Markov** (`rl.markov`). The one assumption an MDP cannot do without, tested rather than
assumed. A nested F test on logged trajectories: does lagged information still predict the
next state and the reward, once the current state and action are in the regression? Two
refinements, both because the naive version would mislead:

- transitions from one trajectory are not independent, so the p-value is computed on an
  **effective sample size**, n scaled by the residuals' lag-1 autocorrelation;
- **significance is not size.** With 960,000 transitions everything is significant, so the
  lag block's partial R² is reported beside p and the verdict uses both.

**argmax of the return function** (`rl.mdp.greedy`). The policy is argmax_a Q(s, a). Ties
break deterministically on the lowest index; a tolerance can widen a tie, so a fitted Q does
not report a policy flipping between two indistinguishable actions.

**Absorbing states** (`rl.mdp.TabularMDP`). Carried explicitly, not as a zero-reward
self-loop. Q(s, a) = 0 there, the terminal payoff is earned on the transition *into* the
state, and `gamma = 1` with an unreachable absorbing set is **refused** rather than run to a
divergence.

## Hedging: the agent learns something real, and loses

One-month at-the-money call, rebalanced every 4 steps of 128, 32,000 training and 16,000
test paths, 21 hedge ratios:

| rule | residual sd / price | fit |
|---|---|---|
| **Hedged Monte Carlo** | **0.3050** | 0.6 s |
| fitted-Q, myopic (γ = 0) | 0.3543 | 0.4 s |
| fitted-Q, bootstrapped (γ = 1) | 0.3568 | 8.7 s |
| LSPI | 0.3573 | 10.4 s |
| Black–Scholes delta | 0.3582 | — |
| …snapped to the agents' grid | 0.3596 | — |

### 1. It learns the leverage adjustment

At τ = 0.057 y, the hedge each rule holds across the smile:

```
BS delta   0.224   0.412   0.549   0.670   0.802
HMC        0.133   0.329   0.475   0.595   0.715
fitted-Q   0.150   0.300   0.400   0.500   0.650
```

HMC holds **less** than the Black–Scholes delta throughout — the ρ = −0.7 leverage effect,
where a fall in the price raises volatility. The agent goes the same way and gets about half
the distance: mean |RL − HMC| 0.062 against mean |RL − BS| 0.124, with the grid step at
0.050. That is a policy you can read, which was part of the brief.

### 2. The problem is a bandit, and bootstrapping costs twenty times nothing

The writer's hedge does not move the market, so the next state does not depend on the
action — asserted in `test_rl.py` at 4 SE. Then

    Q(s, a) = R(s, a) + γ·E[V(s′) | s]

and the second term is identical across actions, so **argmax_a Q = argmax_a R**. The
sequential machinery is solving a problem that is not sequential. Measured: the myopic agent
is both better (0.3543 against 0.3568) and **twenty times cheaper** (0.4 s against 8.7 s).

This is the most portable lesson in the session. Before paying for an MDP, check that you
have one.

### 3. The gap to the optimum is structural

Neither more data nor a finer grid closes it:

| training paths | 2,000 | 8,000 | 32,000 | 128,000 |
|---|---|---|---|---|
| residual | 0.3533 | 0.3561 | 0.3543 | 0.3569 |

| hedge ratios | 3 | 6 | 11 | 21 | 41 |
|---|---|---|---|---|---|
| agent | 0.4663 | 0.3816 | 0.3612 | 0.3543 | 0.3550 |
| BS on the same grid | 0.4955 | 0.3902 | 0.3661 | 0.3596 | 0.3581 |

The agent is converged from 2,000 paths, and the grid stops binding above 11 actions. The
agent beats BS-on-the-same-grid at every resolution, so it is genuinely learning; it simply
stops 16% short of the derived answer.

**The reason is the argmax.** The reward is quadratic in the action, so its optimum is a
parabola's vertex. Hedged Monte Carlo *solves for that vertex* by least squares, using every
path at once and the quadratic structure. The agent fits each arm separately and compares
neighbours, which throws the structure away: it must resolve differences between adjacent
actions that are small beside the noise in (dW)². More arms make each noisier; more data
shrinks the noise but not the estimator's blindness to the shape.

**When you know the reward's structure, estimate it. The argmax is what you use when you do
not.** That is the honest boundary of this framework's usefulness, and it is why the
framework is off by default rather than wired into the pipeline.

### 4. The state is only approximately Markov

On 960,000 transitions the lag block is significant for all three targets and explains
0.02%, 0.65% and 0.21% of the remaining variance. The lifted model's true Markov state is
its 44 factors, not (S, σ̂, τ) — this is that leakage, measured rather than assumed away. It
is small, which is why bootstrapping is merely wasteful here rather than wrong, and it is a
second reason the myopic agent does no worse.

## Filter selection

Learned **offline** from logged runs: each filter is run once over each series, its per-day
log-likelihood and wall clock recorded, and the MDP built over that table. The agent never
calls a filter. This is how the problem presents itself on a desk — you have last month's
runs, not a simulator of them — and it made the study cheap enough to re-run.

It also needed a small library change: none of the three filters exposed a per-day
log-likelihood, only a total. `filters.kalman`, `filters.fourier` and `filters.particle` now
all return `loglik_t`, summing to `loglik` by construction. That is a useful diagnostic
independently of RL — a filter's total hides *where* it loses, and the zero-boundary work is
exactly a question about particular days.

**Choosing the hosts was itself a finding.** A selection problem needs days where the answer
differs. The Kalman filter's boundary diagnostic flags a day when its predictive mean sits
within two predictive sd of zero, and that is far more sensitive than the stationary
standard deviation √(ξ²θ/2κ) suggests: θ = 0.045, ξ = 0.3 flags **100%** of days, 0.06 / 0.2
flags 94%, 0.08 / 0.15 flags 3.3%, and 0.09 / 0.12 flags none. The first attempt used
θ = 0.045 as the "away from the boundary" host, every day flagged, the protocol degenerated
to "always cf", and there was nothing to select. The numbers below use 0.09 / 0.12, three
seeds each, 250 days, reward = the day's log-likelihood − 1.0 × seconds:

| policy | log-likelihood | seconds | objective | what it chose |
|---|---|---|---|---|
| always cf | 5401.8 | 24.3 | **5377.5** | cf 100% |
| **the Session I protocol** | 5396.8 | 22.8 | **5374.0** | Kalman 50%, cf 50% |
| learned, bootstrapped | 5360.0 | 27.6 | 5332.3 | Kalman 5%, cf 69%, particle 26% |
| learned, myopic | 5360.0 | 27.6 | 5332.3 | *identical* |
| always Kalman | 3706.6 | 0.6 | 3706.0 | Kalman 100% |
| always particle (16 substeps) | −31732.4 | 23.8 | −31756.2 | particle 100% |

**The written rule is within 0.07% of the best policy available**, while using the cheap
filter on half the days. The learned one is 0.8% behind it, and its mistake is legible: it
takes the 16-substep particle filter on 26% of days, which is ruinous at the boundary
(−11,721 nats on one series) and merely adequate away from it. A linear Q over five features
cannot represent that cliff; the protocol encodes it as a threshold because Session I
*diagnosed* it — the QE step's atom at zero makes the 16-substep discretisation the wrong
law exactly where the boundary binds. **A rule derived from a mechanism beat a rule fitted
to the data, on the data.**

**This problem is a bandit too, and here provably so.** The state features come from the
Kalman filter's own run whichever filter estimated the day, so the next state is identical
across actions **to 0.0e+00** — not merely independent in distribution, as in the hedging
case, but the same numbers. Then argmax_a Q = argmax_a R exactly, and the bootstrapped and
myopic agents return byte-identical policies. In hedging the independence is only
distributional (each path saw one action), so there the bootstrap does not cancel — it adds
estimation noise, and cost 20× for a slightly worse answer.

**The Markov test failed here, and was right to.** On the filtering state the lag block
explains 2.61% of the next boundary diagnostic and 1.09% of the next RV surprise — both
flagged. The reward, though, is Markov (0.04%): the state predicts today's payoff fine, it
is the state's *own dynamics* that carry memory. For a bandit that distinction is exactly
what matters, and it is why the failure costs nothing here.

**The caveat, stated before the result.** Each day's likelihood is credited to the filter
that produced it in *its own full run*, so a learned switching policy is an **upper bound**
on what a real implementation could achieve: filters carry state, and switching means
re-initialising from a posterior the next filter does not represent exactly.
`switch_penalty` exists to charge for it, and is zero in the headline numbers.

## What it costs

On 32,000 paths, for the same hedging answer:

| | seconds |
|---|---|
| simulating the paths (both pay this) | 7.0 |
| **Hedged Monte Carlo → the optimum** | **0.6** |
| building the RL batch | 0.5 |
| fitted-Q, myopic | 0.4 |
| fitted-Q, bootstrapped | 8.7 |
| LSPI | 10.4 |

The cheapest agent costs 1.6× Hedged Monte Carlo and does not reach it; the sequential ones
cost 18–25×. Akin's "it's not gonna cut on the computational costs" is confirmed with a
number.

## What it is good for

Not nothing, and the boundary is sharp:

- **when there is no analytic optimum.** Both problems here have one, which is exactly why
  they make good tests and bad applications. A cost model with a no-trade region, an
  execution schedule with market impact, a switching rule with real switching costs — those
  have no backward regression to beat.
- **the Markov test on its own.** It is useful before any RL: it says whether a state you
  were about to build a model on screens off the past, and it found that this project's own
  hedging state does so only approximately.
- **the bandit check.** Two lines of reasoning saved 20× the compute here and would in most
  places where an MDP is assumed rather than verified.

## Next

- A cost model with a genuine no-trade region, where the optimum is a threshold nobody has
  written down. That is the first problem in this codebase an agent could win.
- Charge `switch_penalty` properly and re-run the filtering comparison against a switching
  implementation rather than against logged full runs.
- The argmax finding suggests a hybrid worth measuring: fit Q(s, a) as an explicit quadratic
  in a and solve for the vertex. That is Hedged Monte Carlo again, arrived at from the RL
  side, which would be a tidy way to show they are the same object.
