# Changing things

*Session M, 26 September 2026. Every example here runs, and `test_customising.py` executes
the same seams so they cannot rot silently.*

This project is a research instrument: the point is to change it and see what moves. The
seams below are the places designed to be changed, in rough order of how often you would
want to. Nothing here needs you to fork a file — they are all arguments and registries.

Start with `python volsurf.py doctor`, which tells you whether the checkout can run at all.

---

## 1. Run the model on a different lift

Every consumer of the Markovian lift — pricer, calibration, Kalman, particle and cf
filters, simulators — reads `rough_heston.lift_nodes` at call time, so one context manager
moves the whole stack:

```python
import models.rough_heston as rh

with rh.using_lift(24, 1e5):          # the Sessions A-H lift
    price, info = rh.call_prices(ks, tau, params)
```

The default is 44 nodes to 1e8/y. `run_real_data.py --lift 24:1e5` does the same for the
pipeline. What the choice costs and buys is measured in `docs/study-lift-44.md`; the short
version is that the node count sets the kernel's sup-norm error and the *fastest node* sets
the increment law, and they are different criteria.

## 2. Add a hedging rule

`models/hedging.py` scores rules by name. A rule is anything that produces a hedge ratio per
path per date, so adding one means adding a branch to `hedge_error` — or, without touching
the file, scoring it yourself through the same accounting:

```python
import numpy as np, models.hedging as hd

paths = hd.simulate_paths(params, T=1/12, n_steps=128, n_paths=8000, seed=2)
price = 0.0216

def half_delta(k):                     # a deliberately bad rule, to see the machinery work
    return 0.5 * hd.bs_delta(paths["S"][k], 1.0, hd.model_sigma(paths, k), paths["tau"][k])

n, gains = len(paths["t"]) - 1, np.zeros(paths["S"].shape[1])
for k in range(0, n, 4):
    k2 = min(k + 4, n)
    gains += half_delta(k) * (paths["S"][k2] - paths["S"][k])
err = price + gains - np.maximum(paths["S"][n] - 1.0, 0.0)
print(err.std() / price)               # against 0.305 for Hedged Monte Carlo
```

The number to beat is in `docs/study-hedging.md`. Anything above ~0.30 at these settings is
not competitive with the analytic hedge, and the variance swap is what takes it to 0.06.

## 3. Add a volatility model to the scan

`models/egarch.py` takes a `Spec(family, p, q, dist)` and `scan` ranks whatever list you
hand it. To add your own family you implement its filter and register the name:

```python
import models.egarch as eg

rows = eg.scan(returns, specs=(eg.Spec("nelson", 1, 1, "t"),
                               eg.Spec("beta-t"),
                               eg.Spec("garch", 2, 1, "normal")),
               split=1764, targets={"realised variance": rv})
for r in rows:
    print(r["spec"], r["bic"], r["oos"])
```

Rank by the out-of-sample QLIKE in `r["oos"]`, not by `r["bic"]`. `docs/study-egarch.md`
records why: on this project's own rough Heston, BIC puts the Student-t models first and
they forecast worst, because a tail index fitted to the density sets the variance forecast
through ν/(ν−2).

## 4. Register a reinforcement-learning agent

The RL framework is **off by default**. Turn it on, then one decorator puts your agent into
every comparison:

```python
import numpy as np, rl
from rl.mdp import greedy

rl.enable()

@rl.register_agent("my-agent")
class MyAgent:
    def fit(self, batch):
        # batch.phi (n, k), batch.action (n,), batch.reward (n,), batch.phi_next, batch.absorbing
        self.W = np.zeros((batch.n_actions, batch.k))
        for a in range(batch.n_actions):
            rows = batch.action == a
            self.W[a], *_ = np.linalg.lstsq(batch.phi[rows], batch.reward[rows], rcond=None)
        return self

    def q(self, phi):
        return np.atleast_2d(phi) @ self.W.T

    def policy(self, phi):
        return greedy(self.q(phi))

model = rl.agent("my-agent").fit(batch)
```

Before you trust it, run it against the answer key — `study_rl.py A` solves a small MDP
exactly by value iteration and every agent must recover that Q and that policy. An agent
that cannot do that has no business on a hard problem, and the three shipped ones match it
to 1e-13.

And run the Markov test on your state before you bootstrap through it
(`rl.markov.markov_test`). It is the one assumption an MDP cannot do without, and this
project's own hedging state only passes it approximately.

## 5. Change the portfolio trial

`portfolio/` is self-contained. `backtest.Config` carries every knob, and
`dual_kalman.run` takes the list of parameters the filter is allowed to move:

```python
import portfolio.backtest as bt, portfolio.demo_market as dm, portfolio.dual_kalman as dk

market = dm.simulate(dm.MarketSpec(n_assets=60, H_true=(0.05, 0.30)))
engine = bt.Engine(market)
res = engine.run(bt.Config(H=0.02, confidence=0.5, delta=3.0, band=0.002))
learned = dk.run(engine, bt.Config(), ("H", "confidence", "delta"), target_beta=0.8,
                 start=bt.START, end=market.T)
```

Read `docs/trial-bl-hurst.md` first. The objective there is monotone in H, so anything that
calibrates H against cost will walk to the bound — that is a property of the problem, not
of the method, and you will rediscover it.

## 6. Swap the data source

`sources/` has three that emit the same `ChainSnapshot`: `synthetic`, `replay` (from a
recorded JSON) and the live IBKR path. The pipeline takes whichever:

```bash
python volsurf.py pipeline              # synthetic, end to end, with a known answer
python volsurf.py record --check        # the IBKR smoke test (needs IB Gateway)
```

---

## What to run after you change something

```bash
python volsurf.py test quick      # ~15 min, 589 checks
python volsurf.py health          # 128 standing analytical checks, with the numbers
python volsurf.py health --trend  # ...and how they moved against previous runs
```

The health check is the one to watch. It reports measured quantities rather than pass/fail,
and its trend rules compare a run against previous runs **within one model configuration**,
so a deliberate change (a different lift, say) opens a new segment instead of flagging
every series at once.

## The two habits this project runs on

**Grade against something that knows the answer.** Every study here is built around a
quantity whose true value is available: a planted parameter, a closed form, a converged
reference, an exactly solvable MDP. A result with nothing to compare against is a plot.

**Say what did not work.** The three most useful findings in the last two sessions were all
negative — BIC picks the worst volatility forecaster, the Hurst index has no interior
optimum for a cost objective, and the RL agent is beaten by the backward regression it was
pointed at. They are in the docs at the same size as everything else.
