"""Policy evaluation by sweeping.

Fill in the functions in order. The model is known, so every function works
directly on the arrays of an `rlenvs.TabularMDP`: `P` of shape (S, A, S), `R`
of shape (S, A), `gamma`, and `terminal` of shape (S,). A policy `pi` has shape
(S, A), and each row sums to one.

`hallway_mdp` and `policy_matrices` come from earlier chapters. They are
complete; don't change them. For comparisons, `rlenvs.policy_evaluation(mdp, pi)`
returns the exact values from the linear solve.
"""

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
    """One synchronous sweep of the Bellman equation for `pi`.

    Every entry of the result is computed from the old vector `v`:
    r_pi + gamma * P_pi @ v.

    Args:
        mdp: an `rlenvs.TabularMDP`.
        pi: the policy, shape (S, A).
        v: the current value guess, shape (S,). Don't modify it.

    Returns:
        The swept values, shape (S,). Terminal states come out as 0.
    """
    raise NotImplementedError("bellman_sweep")


def iterative_policy_evaluation(mdp, pi, tol):
    """Evaluate `pi` by repeated sweeps from V = 0.

    Stop after the first sweep whose largest change, max |V_new - V|, is below
    `tol`.

    Args:
        mdp: an `rlenvs.TabularMDP` with gamma < 1.
        pi: the policy, shape (S, A).
        tol: the stopping threshold, a positive float.

    Returns:
        A tuple `(v, sweeps)`: the last sweep's values, shape (S,), and the
        number of sweeps performed, an int.
    """
    raise NotImplementedError("iterative_policy_evaluation")


def stopping_bound(gamma, last_change):
    """The guaranteed bound on the error after a sweep that changed values by `last_change`.

    Args:
        gamma: the discount, in [0, 1).
        last_change: the largest absolute change in the last sweep, a float.

    Returns:
        gamma / (1 - gamma) * last_change, a float.
    """
    raise NotImplementedError("stopping_bound")


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
