"""
A reinforcement-learning framework for the parts of this project that are already dynamic
programs (Session M, 26 September 2026).

**Off by default.** Nothing here runs until you turn it on, and every entry point refuses
with an explanation until you do:

    import rl
    rl.enable()                       # or set VOLSURF_RL=1 in the environment

It is off because it costs more, not less. The analytic machinery it sits beside --
Hedged Monte Carlo for the hedge, the zero-boundary protocol for the filter -- solves the
same problems by backward regression and by a written rule, and both are cheaper. This
module exists so that someone who wants to learn the policy instead of deriving it can,
on the same paths, against the same answers.

**Why this project can grade an RL agent honestly.** Both control problems here have a
known optimum:

  hedging    the risk-minimising hedge IS Foellmer-Schweizer local risk minimisation, which
             `models/hedging.py` solves by backward regression; in the near-deterministic
             control it is the Black-Scholes delta and the residual follows the
             Bertsimas-Kogan-Lo law. An agent can be scored against the right answer, not
             against another agent.
  filtering  the zero-boundary protocol (Session I) is a written policy, and the converged
             256-substep particle filter is the answer it was written against.

So every agent in here reports its gap to a known optimum, and `rl/README.md` records
what those gaps are. An RL result with nothing to compare against is a plot, not a finding.

**The three primitives**, which is all an MDP needs:

  Markov     `rl.markov`: does the state you chose actually screen off the past? A
             conditional-independence test on logged trajectories, run BEFORE any learning.
             If it fails, no agent can succeed and the honest move is to enlarge the state.
  argmax     `rl.mdp.greedy`: the policy is the argmax over actions of the action-value
             (return) function Q(s, a) -- what the state-action pair goes on to generate.
  absorbing  states where the episode ends and the value is the terminal payoff: expiry for
             the hedge, the end of the series for the filter. `TabularMDP` carries them
             explicitly, because getting them wrong is the commonest way a value iteration
             silently converges to the wrong thing.

Customising it: `docs/customising.md`. Registering your own agent takes one decorator.
"""

import os

__all__ = ["enable", "disable", "is_enabled", "require_enabled", "Disabled",
           "register_agent", "agent", "AGENTS"]

_ENABLED = os.environ.get("VOLSURF_RL", "").strip().lower() in ("1", "true", "yes", "on")


class Disabled(RuntimeError):
    """
    Raised by every entry point in `rl` while the framework is off.

    A refusal, not a warning: an RL run that started by accident would spend minutes and
    hand back a policy nobody asked for, which is worse than an error naming the switch.
    """


def enable():
    """Turn the framework on for this process."""
    global _ENABLED
    _ENABLED = True


def disable():
    global _ENABLED
    _ENABLED = False


def is_enabled():
    return _ENABLED


def require_enabled(what="this"):
    if not _ENABLED:
        raise Disabled(
            f"{what} is part of the reinforcement-learning framework, which is off by default "
            "because it costs more than the analytic machinery beside it. Turn it on with "
            "`import rl; rl.enable()`, or set VOLSURF_RL=1 in the environment. See rl/README.md "
            "for what it buys and what it does not.")


# ------------------------------------------------------------------ the agent registry
AGENTS = {}


def register_agent(name):
    """
    Decorator registering an agent class under `name`, so a study can name it in a string
    and someone else's agent can join the comparison without editing the study:

        @rl.register_agent("my-agent")
        class MyAgent:
            def fit(self, data): ...
            def q(self, features): ...      # (n, n_actions)

    Every agent in the comparison is scored the same way against the same known optimum.
    """
    def deco(cls):
        if name in AGENTS and AGENTS[name] is not cls:
            raise ValueError(f"an agent named {name!r} is already registered ({AGENTS[name].__name__})")
        AGENTS[name] = cls
        cls.agent_name = name
        return cls
    return deco


def agent(name, *args, **kwargs):
    """Construct a registered agent by name."""
    require_enabled(f"the agent {name!r}")
    if name not in AGENTS:
        raise KeyError(f"no agent named {name!r}; registered: {sorted(AGENTS)}")
    return AGENTS[name](*args, **kwargs)
