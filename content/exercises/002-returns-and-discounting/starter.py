"""Returns and discounting.

Fill in the three functions in order. Together they weight rewards by a
discount, turn the rewards of an episode into returns, and estimate how good a
policy is by averaging the returns of many episodes.

`run_episode` is given in full, because you wrote it in the previous chapter.
The environment is an `rlenvs.TabularMDP`, and a policy `pi` is an array of
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


def run_episode(mdp, pi, rng):
    """Run one episode of policy `pi` in `mdp`. Given in full: you wrote it in the previous chapter.

    Returns `(states, actions, rewards, terminated)`, where `states[t]`,
    `actions[t]`, and `rewards[t]` all belong to step t.
    """
    s = mdp.reset(seed=int(rng.integers(2**31)))
    states, actions, rewards = [], [], []
    while True:
        a = int(rng.choice(mdp.n_actions, p=pi[s]))
        s2, r, terminated, truncated = mdp.step(a)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = s2
        if terminated or truncated:
            break
    return np.array(states), np.array(actions), np.array(rewards, dtype=float), bool(terminated)


def discount_weights(gamma, n):
    """The weight that gamma gives to a reward k steps ahead, for k = 0 to n - 1.

    The weight is gamma ** k, so the reward now has weight 1.

    Args:
        gamma: the discount factor, in [0, 1].
        n: how many weights to return.

    Returns:
        Shape (n,). Entry k is gamma ** k.
    """
    raise NotImplementedError("discount_weights")


def discounted_returns(rewards, gamma):
    """The discounted return G_t from every step of one episode.

    G_t = r_t + gamma * r_(t+1) + gamma^2 * r_(t+2) + ... up to the last reward.
    Use the recursion G_t = r_t + gamma * G_(t+1) with G_T = 0, so one backward
    pass gives every return.

    Args:
        rewards: the rewards r_0, ..., r_(T-1) of one episode, shape (T,).
        gamma: the discount factor, in [0, 1].

    Returns:
        Shape (T,). Entry t is G_t.
    """
    raise NotImplementedError("discounted_returns")


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
    print("weights on rewards 0 to 4 steps ahead:")
    for gamma in (0.5, 0.9):
        print(f"  gamma={gamma}:", np.round(discount_weights(gamma, 5), 3))

    mdp = rlenvs.gridworld(ROOM, gamma=0.9)
    pi = rlenvs.uniform_policy(mdp)
    print("returns for rewards (1, 0, 2):", discounted_returns([1.0, 0.0, 2.0], 0.9))

    exact = rlenvs.policy_evaluation(mdp, pi)[mdp.names["start"]]
    for n in (100, 400, 1600):
        estimate, se = mc_value_estimate(mdp, pi, n, seed=1)
        print(f"  {n:5d} episodes: V(start) about {estimate:.3f} +- {se:.3f}   (exact {exact:.3f})")
