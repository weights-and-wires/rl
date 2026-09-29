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


def soft_policy_evaluation(mdp, pi, alpha):
    pi = np.asarray(pi, dtype=float)
    with np.errstate(divide="ignore"):
        log_pi = np.where(pi > 0, np.log(pi), 0.0)
    # The entropy bonus is a reward of -alpha * log pi(a | s) for acting in s.
    bonus = rlenvs.TabularMDP(
        P=mdp.P, R=mdp.R - alpha * log_pi, gamma=mdp.gamma, mu0=mdp.mu0, terminal=mdp.terminal
    )
    v = rlenvs.policy_evaluation(bonus, pi)
    return rlenvs.q_from_v(mdp, v)


def soft_bellman_residual(q, s, a, r, s2, terminated, gamma, alpha):
    q = np.asarray(q, dtype=float)
    v_next = soft_value(q[s2], alpha)
    target = np.asarray(r, dtype=float) + gamma * (1.0 - np.asarray(terminated, dtype=float)) * v_next
    return target - q[s, a]


def soft_policy_iteration(mdp, alpha, iters):
    pi = rlenvs.uniform_policy(mdp)
    history = np.zeros((iters + 1, mdp.n_states, mdp.n_actions))
    for k in range(iters + 1):
        history[k] = soft_policy_evaluation(mdp, pi, alpha)
        pi = boltzmann_policy(history[k], alpha)
    return history


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

    alpha = 0.1
    history = soft_policy_iteration(lake, alpha, iters=6)
    q_star = soft_value_iteration(lake, alpha)
    print(f"\nSoft policy iteration at alpha = {alpha}: distance to soft value iteration's Q*")
    for k, q in enumerate(history):
        print(f"  iteration {k}: max |Q - Q*| = {np.max(np.abs(q - q_star)):.2e}")
