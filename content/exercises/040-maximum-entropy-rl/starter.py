"""Maximum-entropy reinforcement learning on tabular MDPs.

Fill in the functions below. The agent earns the reward plus alpha times the
entropy of its policy in every state it acts in, where alpha > 0 is the
temperature. The hard max of ordinary value iteration becomes a soft max,
alpha * log sum_a exp(Q / alpha), and the optimal policy becomes a Boltzmann
distribution over the soft action values.

Q arrays have shape (S, A). A policy `pi` has shape (S, A), rows summing to
one. A terminal state's value is zero by definition, soft or not.
"""

import numpy as np

import rlenvs


def soft_value(q, alpha):
    """The soft maximum over actions: V(s) = alpha * log sum_a exp(Q(s, a) / alpha).

    Args:
        q: action values, shape (S, A). It can hold values near 1000.
        alpha: the temperature, a positive float. It can be as small as 1e-4.

    Returns:
        V, shape (S,). Subtract each row's maximum before calling `np.exp`.
    """
    raise NotImplementedError("soft_value")


def boltzmann_policy(q, alpha):
    """The Boltzmann policy pi(a | s) = exp((Q(s, a) - V(s)) / alpha).

    Args:
        q: action values, shape (S, A).
        alpha: the temperature.

    Returns:
        pi, shape (S, A), rows summing to one.
    """
    raise NotImplementedError("boltzmann_policy")


def soft_bellman_backup(mdp, q, alpha):
    """One soft Bellman optimality backup.

    new Q(s, a) = R(s, a) + gamma * sum_s2 P(s2 | s, a) * V(s2), with
    V = soft_value(q, alpha) in live states and V = 0 in terminal states.
    New Q is zero in terminal states, as `rlenvs.q_from_v` makes it.

    Args:
        mdp: a `rlenvs.TabularMDP`.
        q: the current action values, shape (S, A).
        alpha: the temperature.

    Returns:
        The backed-up Q, shape (S, A).
    """
    raise NotImplementedError("soft_bellman_backup")


def soft_value_iteration(mdp, alpha, tol=1e-12, max_iter=100_000):
    """Iterate `soft_bellman_backup` from Q = 0 until it stops changing.

    Args:
        mdp: a `rlenvs.TabularMDP` with gamma < 1.
        alpha: the temperature.
        tol: stop when successive Q arrays differ by less than this everywhere.
        max_iter: a safety cap on the number of backups.

    Returns:
        The soft-optimal Q*, shape (S, A).
    """
    raise NotImplementedError("soft_value_iteration")


def soft_bellman_residual(q, s, a, r, s2, terminated, gamma, alpha):
    """The sampled soft Bellman residual for a batch of transitions.

    delta = r + gamma * (1 - terminated) * soft_value(q[s2]) - q[s, a].

    Args:
        q: action values, shape (S, A).
        s, a, s2: integer arrays, shape (N,).
        r: rewards, shape (N,). r is the reward for taking a in s.
        terminated: shape (N,), 1.0 where s2 is terminal.
        gamma: the discount factor.
        alpha: the temperature.

    Returns:
        delta, shape (N,).
    """
    raise NotImplementedError("soft_bellman_residual")


if __name__ == "__main__":
    lake = rlenvs.frozen_lake()
    start = lake.names["start"]
    v_hard = rlenvs.value_iteration(lake)
    print("Frozen lake: the soft optimum approaches the hard one as alpha shrinks")
    print(" alpha    V_soft(start)   V*(start)   entropy of pi*(. | start)")
    for alpha in [1.0, 0.1, 0.01, 0.001]:
        q = soft_value_iteration(lake, alpha)
        pi = boltzmann_policy(q, alpha)
        entropy = -np.sum(pi[start] * np.log(pi[start]))
        v = soft_value(q, alpha)[start]
        print(f" {alpha:<6}   {v:.6f}        {v_hard[start]:.6f}    {entropy:.4f} nats")
