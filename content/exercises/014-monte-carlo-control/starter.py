"""Epsilon-soft Monte Carlo control.

`discounted_returns` is given: it is the backward pass from the returns
chapter, and you used it in the Monte Carlo prediction chapter.
"""

import numpy as np

import rlenvs


def discounted_returns(rewards, gamma):
    """The return G_t from every step of one episode, by a backward pass (given).

    Args:
        rewards: shape (T,), where rewards[t] is r_t.
        gamma: the discount factor.

    Returns:
        Shape (T,), where entry t is G_t = r_t + gamma * G_(t+1), with G_T = 0.
    """
    rewards = np.asarray(rewards, dtype=float)
    G = np.zeros_like(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        G[t] = running
    return G


def epsilon_soft(Q, epsilon):
    """The epsilon-greedy policy for an action-value table.

    Args:
        Q: shape (S, A).
        epsilon: the exploration rate, in [0, 1].

    Returns:
        pi, shape (S, A). In each row, the first action with the largest Q
        (the one np.argmax picks) gets 1 - epsilon + epsilon / A, and every
        other action gets epsilon / A.
    """
    raise NotImplementedError("epsilon_soft")


def epsilon_soft_world(mdp, epsilon):
    """The modified world in which every chosen action is replaced, with probability epsilon, by a uniformly random one.

    Playing greedily in this world is the same as playing epsilon-greedily in
    `mdp`. So the optimal policy of this world is worth what the best
    epsilon-soft policy of `mdp` is worth.

    Args:
        mdp: an `rlenvs.TabularMDP`.
        epsilon: the replacement probability, in [0, 1].

    Returns:
        A new `rlenvs.TabularMDP` with the same `gamma`, `mu0`, and `terminal`
        as `mdp`, whose `P` and `R` mix each action's outcome with the average
        over all actions.
    """
    raise NotImplementedError("epsilon_soft_world")


def mc_control(mdp, n_episodes, epsilon, alpha, seed):
    """On-policy first-visit Monte Carlo control with an epsilon-greedy policy.

    Start from Q = 0. For each episode, act epsilon-greedily on the current Q
    (break ties between best actions at random), then move Q[s, a] a
    fraction alpha toward the return that followed the first visit to each
    pair (s, a) in the episode.

    Args:
        mdp: an rlenvs.TabularMDP. Use mdp.mu0 for the start state,
            mdp.sample(s, a, rng) for each step, mdp.terminal to stop, and
            mdp.max_steps as a cap on the episode length.
        n_episodes: how many episodes to run.
        epsilon: the exploration rate.
        alpha: the constant step size.
        seed: seed for np.random.default_rng, the only source of randomness.

    Returns:
        Q, shape (S, A).
    """
    raise NotImplementedError("mc_control")


if __name__ == "__main__":
    grid = rlenvs.gridworld(["....", ".#..", "S..G"])
    start = grid.names["start"]
    epsilon = 0.2
    best = rlenvs.value_iteration(epsilon_soft_world(grid, epsilon))[start]
    optimal = rlenvs.value_iteration(grid)[start]
    Q = mc_control(grid, n_episodes=1000, epsilon=epsilon, alpha=0.1, seed=0)
    v = rlenvs.policy_evaluation(grid, epsilon_soft(Q, epsilon))
    print(f"optimal value from the start:           {optimal:.3f}")
    print(f"best epsilon-soft value from the start: {best:.3f}")
    print(f"your epsilon-soft policy's value:       {v[start]:.3f}")
