"""Reference solution for the policy evaluation exercise."""

import numpy as np

import rlenvs


def hallway_mdp(gamma=0.9):
    """Four cells in a row. Cell 3 is the exit and ends the episode.

    Action 0 is left and action 1 is right. A move changes the cell by one, and
    moving left from cell 0 bumps the wall and stays in cell 0. Every move from
    cells 0, 1, and 2 pays -1. The episode starts in cell 0.
    """
    P = np.zeros((4, 2, 4))
    R = np.zeros((4, 2))
    for s in range(3):
        P[s, 0, max(s - 1, 0)] = 1.0
        P[s, 1, s + 1] = 1.0
        R[s, :] = -1.0
    P[3, :, 3] = 1.0
    return rlenvs.TabularMDP(
        P=P, R=R, gamma=gamma,
        mu0=np.array([1.0, 0.0, 0.0, 0.0]),
        terminal=np.array([False, False, False, True]),
    )


def policy_matrices(mdp, pi):
    """P_pi of shape (S, S) and r_pi of shape (S,), with terminal rows set to zero.

    Given in full from an earlier chapter.
    """
    live = ~mdp.terminal
    P_pi = np.einsum("sa,sat->st", pi, mdp.P) * live[:, None]
    r_pi = (pi * mdp.R).sum(axis=1) * live
    return P_pi, r_pi


def bellman_sweep(mdp, pi, v):
    P_pi, r_pi = policy_matrices(mdp, pi)
    return r_pi + mdp.gamma * P_pi @ np.asarray(v, dtype=float)


def iterative_policy_evaluation(mdp, pi, tol):
    v = np.zeros(mdp.n_states)
    sweeps = 0
    while True:
        new = bellman_sweep(mdp, pi, v)
        sweeps += 1
        if np.max(np.abs(new - v)) < tol:
            return new, sweeps
        v = new


def stopping_bound(gamma, last_change):
    return float(gamma / (1.0 - gamma) * last_change)


if __name__ == "__main__":
    mdp = hallway_mdp()
    uniform = rlenvs.uniform_policy(mdp)
    exact = rlenvs.policy_evaluation(mdp, uniform)

    v = np.zeros(mdp.n_states)
    print("Sweeps of the uniform policy from zeros (exact:", np.round(exact, 4), ")")
    for k in range(1, 4):
        v = bellman_sweep(mdp, uniform, v)
        print(f"  sweep {k}:", np.round(v, 4))

    for tol in (1e-1, 1e-3, 1e-6):
        v, sweeps = iterative_policy_evaluation(mdp, uniform, tol)
        err = np.max(np.abs(v - exact))
        print(f"tol {tol:g}: {sweeps} sweeps, true error {err:.2e}, bound {stopping_bound(mdp.gamma, tol):.2e}")
