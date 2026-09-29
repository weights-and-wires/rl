"""Reference solution for the dynamic programming exercise."""

import math

import numpy as np

import rlenvs


def two_state_mdp(gamma=0.9):
    """The example from lesson 04: in each of two states, stay or go to the other one.

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
    """P_pi of shape (S, S) and r_pi of shape (S,), with terminal rows set to zero (from lesson 04)."""
    live = ~mdp.terminal
    P_pi = np.einsum("sa,sat->st", pi, mdp.P) * live[:, None]
    r_pi = (pi * mdp.R).sum(axis=1) * live
    return P_pi, r_pi


def iterative_policy_evaluation(mdp, pi, tol):
    P_pi, r_pi = policy_matrices(mdp, pi)
    v = np.zeros(mdp.n_states)
    sweeps = 0
    while True:
        new = r_pi + mdp.gamma * P_pi @ v
        sweeps += 1
        if np.max(np.abs(new - v)) < tol:
            return new, sweeps
        v = new


def greedy_policy(mdp, v):
    q = rlenvs.q_from_v(mdp, v)
    pi = np.zeros_like(q)
    pi[np.arange(mdp.n_states), q.argmax(axis=1)] = 1.0
    return pi


def value_iteration(mdp, tol, v0=None):
    v = np.zeros(mdp.n_states) if v0 is None else np.asarray(v0, dtype=float).copy()
    history = [v]
    while True:
        new = rlenvs.q_from_v(mdp, v).max(axis=1)
        history.append(new)
        if np.max(np.abs(new - v)) < tol:
            return new, np.array(history)
        v = new


def sweeps_needed(gamma, initial_error, eps):
    if initial_error <= eps:
        return 0
    return math.ceil(math.log(eps / initial_error) / math.log(gamma))


def policy_iteration(mdp, pi0=None):
    if pi0 is None:
        pi0 = np.zeros((mdp.n_states, mdp.n_actions))
        pi0[:, 0] = 1.0
    pi = np.asarray(pi0, dtype=float)
    values = []
    while True:
        P_pi, r_pi = policy_matrices(mdp, pi)
        v = np.linalg.solve(np.eye(mdp.n_states) - mdp.gamma * P_pi, r_pi)
        values.append(v)
        new = greedy_policy(mdp, v)
        if np.array_equal(new, pi):
            return pi, v, np.array(values)
        pi = new


if __name__ == "__main__":
    mdp = two_state_mdp()
    v, history = value_iteration(mdp, tol=1e-10)
    print("Value iteration on the two-state MDP, from V = (0, 0):")
    for k in range(1, 5):
        err = np.max(np.abs(history[k] - v))
        print(f"  k={k}: V = {np.round(history[k], 3)}, error {err:.3f}, bound 20 * 0.9^k = {20 * 0.9**k:.3f}")
    print(f"  converged after {len(history) - 1} sweeps to {np.round(v, 6)}")
    print(f"  sweeps_needed(0.9, 20, 0.01) = {sweeps_needed(0.9, 20.0, 0.01)}")

    pi, v, values = policy_iteration(mdp)
    print("Policy iteration from 'stay everywhere':", [np.round(x, 3).tolist() for x in values])
    print("  final policy (rows are states, columns stay and go):", pi.tolist())

    mdp = rlenvs.random_mdp(n_states=20, n_actions=4, gamma=0.95, seed=0)
    pi, v, values = policy_iteration(mdp)
    v_vi, history = value_iteration(mdp, tol=1e-10)
    print(f"Random 20-state MDP: policy iteration took {len(values)} evaluations, "
          f"value iteration {len(history) - 1} sweeps; values agree to {np.max(np.abs(v - v_vi)):.1e}")
