"""Reference solution for the maximum-entropy reinforcement learning exercise."""

import numpy as np

import rlenvs


def soft_value(q, alpha):
    q = np.asarray(q, dtype=float)
    top = q.max(axis=-1)
    return top + alpha * np.log(np.exp((q - top[..., None]) / alpha).sum(axis=-1))


def boltzmann_policy(q, alpha):
    q = np.asarray(q, dtype=float)
    return np.exp((q - soft_value(q, alpha)[..., None]) / alpha)


def soft_bellman_backup(mdp, q, alpha):
    v = np.where(mdp.terminal, 0.0, soft_value(q, alpha))
    return rlenvs.q_from_v(mdp, v)


def soft_value_iteration(mdp, alpha, tol=1e-12, max_iter=100_000):
    q = np.zeros((mdp.n_states, mdp.n_actions))
    for _ in range(max_iter):
        new = soft_bellman_backup(mdp, q, alpha)
        if np.max(np.abs(new - q)) < tol:
            return new
        q = new
    return q


def soft_bellman_residual(q, s, a, r, s2, terminated, gamma, alpha):
    q = np.asarray(q, dtype=float)
    v_next = soft_value(q[s2], alpha)
    target = np.asarray(r, dtype=float) + gamma * (1.0 - np.asarray(terminated, dtype=float)) * v_next
    return target - q[s, a]


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
