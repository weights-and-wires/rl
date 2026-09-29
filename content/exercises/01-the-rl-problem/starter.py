"""Agents, rewards, and returns.

Fill in the three functions in order. Together they run the agent-environment
loop, turn a list of rewards into returns, and estimate how good a policy is
by averaging the returns of many episodes.

The environment is an `rlenvs.TabularMDP`. `mdp.reset(seed)` starts an
episode and returns the first state. `mdp.step(a)` returns
`(next_state, reward, terminated, truncated)`. A policy `pi` is an array of
shape (n_states, n_actions) whose row `pi[s]` holds the probabilities of each
action in state `s`.
"""

import numpy as np

import rlenvs

# A 3 by 3 room with a pillar in the middle. Every step costs 1, and the
# episode ends on reaching G in the far corner.
ROOM = [
    "S..",
    ".#.",
    "..G",
]


def discounted_returns(rewards, gamma):
    """The discounted return G_t from every step of one episode.

    G_t = r_t + gamma * r_(t+1) + gamma^2 * r_(t+2) + ... up to the last reward.

    Args:
        rewards: the rewards r_0, ..., r_(T-1) of one episode, shape (T,).
        gamma: the discount factor, in [0, 1].

    Returns:
        Shape (T,). Entry t is G_t.
    """
    raise NotImplementedError("discounted_returns")


def run_episode(mdp, pi, rng):
    """Run one episode of policy `pi` in `mdp`.

    Start with `mdp.reset(seed=int(rng.integers(2**31)))`, so the whole episode
    is reproducible from `rng`. In each state s, draw an action from `pi[s]`
    with `rng`, then call `mdp.step`. Stop when the step reports terminated or
    truncated.

    Args:
        mdp: an `rlenvs.TabularMDP`.
        pi: the policy, shape (n_states, n_actions).
        rng: a numpy Generator.

    Returns:
        A tuple `(states, actions, rewards, terminated)`. `states[t]` is the
        state in which `actions[t]` was taken and `rewards[t]` received, so all
        three arrays have shape (T,). `terminated` is True if the episode ended
        in a terminal state and False if it hit the time limit.
    """
    raise NotImplementedError("run_episode")


def mc_value_estimate(mdp, pi, n_episodes, seed):
    """Estimate the value of the start state by averaging returns.

    Make one Generator with `np.random.default_rng(seed)` and pass it to
    `run_episode` for every episode. Compute each episode's return G_0 with
    `discounted_returns` and `mdp.gamma`.

    Args:
        mdp: an `rlenvs.TabularMDP`.
        pi: the policy, shape (n_states, n_actions).
        n_episodes: how many episodes to average, at least 2.
        seed: an int.

    Returns:
        A tuple `(estimate, standard_error)` of floats: the mean of the G_0
        values, and their sample standard deviation (with ddof=1) divided by
        sqrt(n_episodes).
    """
    raise NotImplementedError("mc_value_estimate")


if __name__ == "__main__":
    mdp = rlenvs.gridworld(ROOM, gamma=0.9)
    pi = rlenvs.uniform_policy(mdp)

    rng = np.random.default_rng(0)
    states, actions, rewards, terminated = run_episode(mdp, pi, rng)
    print(f"one episode of the random policy: {len(rewards)} steps, terminated={terminated}")
    print("  returns G_t for the first five steps:", np.round(discounted_returns(rewards, mdp.gamma)[:5], 3))

    exact = rlenvs.policy_evaluation(mdp, pi)[mdp.names["start"]]
    for n in (100, 400, 1600):
        estimate, se = mc_value_estimate(mdp, pi, n, seed=1)
        print(f"  {n:5d} episodes: V(start) about {estimate:.3f} +- {se:.3f}   (exact {exact:.3f})")
