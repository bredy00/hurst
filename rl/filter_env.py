"""
Filter selection as an MDP, learned OFFLINE from logged filter runs (Session M).

Akin asked in Session J for "Kalman filter selection using deep reinforced learning
frameworks"; this is that, with two deliberate departures from the literature it points at.

**First: offline, from a log.** Running a particle filter inside a learning loop costs
minutes per episode and would dominate everything. Instead each filter is run ONCE over a
set of series, its per-day log-likelihood contribution and its wall-clock cost recorded,
and the MDP is built over that table. The agent never calls a filter; it learns from what
the filters did. This is how the problem actually presents itself on a desk -- you have
last month's runs, not a simulator of them -- and it makes the whole study cheap enough to
re-run.

**Second: it is graded against a written rule that already exists.** `filters/protocol.py`
is the Session I policy: Kalman everywhere, the cf filter where the zero boundary binds,
the particle filter to confirm. It was chosen against the converged 256-substep particle
filter. So the learned policy has both a baseline to beat and a truth to be scored on,
which is the difference between a finding and a demonstration.

**The MDP.**
  state      per day: the Kalman filter's predictive mean and sd of the spot variance, how
             many predictive sd zero sits below the mean (the boundary diagnostic the
             protocol thresholds), the observed RV relative to its forecast, and the
             fraction of the series elapsed.
  action     which filter states this day: kalman (cheap), cf (accurate near zero), or
             particle (the reference, expensive).
  reward     the day's log-likelihood under the chosen filter, minus `cost_weight` times its
             cost in seconds. That single knob is the exchange rate between accuracy and
             compute, and it is set before the run, not tuned to the answer.
  absorbing  the end of the series.
  gamma      1: a series is a finite episode.

**The honest caveat, stated before the result.** Choosing a filter for one day and then
another for the next is not free in reality -- filters carry state, and switching means
re-initialising from a posterior the new filter does not represent exactly. The reward here
credits each day's likelihood to the filter that produced it in ITS OWN full run, so the
learned policy is an upper bound on what a real switching implementation would achieve.
`docs/rl-framework.md` says so, and `switch_penalty` exists to charge for it.
"""

import numpy as np

import rl
from rl.agents import Batch

FILTERS = ("kalman", "cf", "particle")


def day_features(mean_V, sd_V, y, t_frac):
    """
    The state, one row per day: [1, z, log10 mean_V, rv/forecast - 1, t_frac], with
    z = mean_V / sd_V the number of predictive sd zero sits below the mean.

    z is exactly what `protocol.boundary_days` thresholds (it flags z <= 2), so the learned
    policy and the written one see the same quantity and any difference between them is a
    difference of policy, not of information.
    """
    mean_V = np.asarray(mean_V, float)
    sd_V = np.maximum(np.asarray(sd_V, float), 1e-12)
    z = mean_V / sd_V
    fore = np.maximum(mean_V, 1e-12)
    return np.column_stack([np.ones_like(z), z, np.log10(np.maximum(mean_V, 1e-12)),
                            np.asarray(y, float) / fore - 1.0, np.asarray(t_frac, float)])


def build_batch(logs, cost_weight=1.0, switch_penalty=0.0):
    """
    Build the offline batch from logged filter runs.

    `logs` is a list, one entry per series, each a dict:
        features      (T, k) from `day_features`
        loglik        {filter name: (T,) per-day log-likelihood contribution}
        seconds       {filter name: float, the whole run's wall clock}
        truth_loglik  (T,) optional: the converged reference, for scoring

    Every (day, filter) pair becomes a transition, which is what makes an offline batch from
    three full runs: the counterfactual "what if this day had used the cf filter" is not a
    guess, it is a row of the log.
    """
    rl.require_enabled("the filter-selection environment")
    phi_l, act_l, rew_l, phin_l, abs_l, ep_l, st_l = [], [], [], [], [], [], []
    for e, log in enumerate(logs):
        f = np.asarray(log["features"], float)
        T = len(f)
        per_day_cost = {k: log["seconds"][k] / max(T, 1) for k in FILTERS}
        f_next = np.vstack([f[1:], f[-1:]])
        for a, name in enumerate(FILTERS):
            ll = np.asarray(log["loglik"][name], float)
            r = ll - cost_weight * per_day_cost[name]
            if switch_penalty:
                r = r - switch_penalty            # charged on every day; see the caveat above
            phi_l.append(f)
            act_l.append(np.full(T, a))
            rew_l.append(r)
            phin_l.append(f_next)
            absorb = np.zeros(T, bool)
            absorb[-1] = True
            abs_l.append(absorb)
            # one episode per (series, filter): the log of a filter's OWN full run, so a
            # lag pairs two days of the same run rather than two different filters' days
            ep_l.append(np.full(T, e * len(FILTERS) + a))
            st_l.append(np.arange(T))
    return Batch(np.vstack(phi_l), np.concatenate(act_l), np.concatenate(rew_l),
                 np.vstack(phin_l), np.concatenate(abs_l), len(FILTERS),
                 episode=np.concatenate(ep_l), step=np.concatenate(st_l))


def protocol_policy(features, threshold=2.0):
    """
    The Session I rule as a policy over the same state: the cf filter where the boundary
    binds (z <= threshold), the Kalman filter elsewhere. The particle filter is the
    confirmation step, not a state estimator, so it is never chosen.
    """
    z = np.asarray(features, float)[:, 1]
    return np.where(z <= threshold, FILTERS.index("cf"), FILTERS.index("kalman"))


def always(name, n):
    return np.full(int(n), FILTERS.index(name))


def score(logs, policy_fn, cost_weight=1.0):
    """
    Score a policy on the logs: total log-likelihood, total cost, the objective, and -- when
    the logs carry `truth_loglik` -- the gap to the converged reference.

    `policy_fn(features) -> (T,)` action indices.
    """
    tot_ll = tot_cost = tot_truth = 0.0
    days = 0
    per_filter = {k: 0 for k in FILTERS}
    for log in logs:
        f = np.asarray(log["features"], float)
        T = len(f)
        a = np.asarray(policy_fn(f), int)
        if len(a) != T:
            raise ValueError(f"policy returned {len(a)} actions for {T} days")
        ll = np.column_stack([np.asarray(log["loglik"][k], float) for k in FILTERS])
        tot_ll += float(ll[np.arange(T), a].sum())
        for i, k in enumerate(FILTERS):
            n_i = int((a == i).sum())
            per_filter[k] += n_i
            tot_cost += log["seconds"][k] / max(T, 1) * n_i
        if "truth_loglik" in log:
            tot_truth += float(np.asarray(log["truth_loglik"], float).sum())
        days += T
    out = {"loglik": tot_ll, "seconds": tot_cost, "objective": tot_ll - cost_weight * tot_cost,
           "days": days, "share": {k: v / max(days, 1) for k, v in per_filter.items()}}
    if tot_truth:
        out["gap_to_truth"] = tot_ll - tot_truth
        out["gap_per_day"] = (tot_ll - tot_truth) / max(days, 1)
    return out


def markov_inputs(batch):
    """(features, lagged, targets, names) for `rl.markov` on the filtering state."""
    from rl.markov import lagged_design
    rows, prev, f_t, f_prev = lagged_design(batch.phi[:, 1:], batch.episode, batch.step)
    act = batch.action[rows][:, None].astype(float)
    act_prev = batch.action[prev][:, None].astype(float)
    tgt = np.column_stack([batch.phi_next[rows, 1], batch.phi_next[rows, 3], batch.reward[rows]])
    return (np.hstack([f_t, act]), np.hstack([f_prev, act_prev]), tgt,
            ["next boundary z", "next RV surprise", "reward"])
