"""Reference solution for the Markov decision processes exercise."""

import numpy as np

import rlenvs


def two_state_mdp(gamma=0.9):
    """The lesson's example: in each of two states, stay or go to the other one.

    Staying in state 0 pays 1, staying in state 1 pays 2, and going pays 0.
    Action 0 is stay and action 1 is go.
    """
    P = np.zeros((2, 2, 2))
    P[0, 0, 0] = 1.0
    P[0, 1, 1] = 1.0
    P[1, 0, 1] = 1.0
    P[1, 1, 0] = 1.0
    R = np.array([[1.0, 0.0], [2.0, 0.0]])
    return rlenvs.TabularMDP(P=P, R=R, gamma=gamma, mu0=np.array([1.0, 0.0]))


def policy_matrices(mdp, pi):
    P_pi = np.einsum("sa,sat->st", pi, mdp.P)
    r_pi = (pi * mdp.R).sum(axis=1)
    live = ~mdp.terminal
    return P_pi * live[:, None], r_pi * live


def evaluate_exact(mdp, pi):
    P_pi, r_pi = policy_matrices(mdp, pi)
    return np.linalg.solve(np.eye(mdp.n_states) - mdp.gamma * P_pi, r_pi)


def q_from_v(mdp, v):
    q = mdp.R + mdp.gamma * mdp.P @ v
    q[mdp.terminal] = 0.0
    return q


def bellman_optimality_backup(mdp, v):
    return q_from_v(mdp, v).max(axis=1)


def discounted_visitation(mdp, pi):
    P_pi, _ = policy_matrices(mdp, pi)
    d = (1.0 - mdp.gamma) * np.linalg.solve((np.eye(mdp.n_states) - mdp.gamma * P_pi).T, mdp.mu0)
    return d / d.sum()


if __name__ == "__main__":
    mdp = two_state_mdp()
    pi = rlenvs.uniform_policy(mdp)
    P_pi, r_pi = policy_matrices(mdp, pi)
    v = evaluate_exact(mdp, pi)
    print("Two-state MDP, uniform random policy")
    print("  P_pi =", P_pi.tolist(), " r_pi =", r_pi.tolist())
    print("  V =", np.round(v, 4), " Q =", np.round(q_from_v(mdp, v), 4).tolist())
    v_star = np.array([18.0, 20.0])
    print("  Bellman optimality backup of (18, 20):", bellman_optimality_backup(mdp, v_star))
    print("  discounted visitation from state 0:", np.round(discounted_visitation(mdp, pi), 4))

    mdp = rlenvs.random_mdp(n_states=6, n_actions=3, gamma=0.9, seed=0)
    pi = rlenvs.uniform_policy(mdp)
    gap = np.max(np.abs(evaluate_exact(mdp, pi) - rlenvs.policy_evaluation(mdp, pi)))
    print(f"Random 6-state MDP: largest gap from rlenvs.policy_evaluation = {gap:.2e}")
