"""Value iteration and its error bound.

The model is known, so every function works directly on the arrays of an
`rlenvs.TabularMDP`: `P` of shape (S, A, S), `R` of shape (S, A), `gamma`, and
`terminal` of shape (S,). A policy `pi` has shape (S, A), and each row sums to
one.

`two_state_mdp` and `greedy_policy` are given. You wrote `greedy_policy` in the
chapter on policy iteration. You can also call `rlenvs.q_from_v(mdp, v)`, which
returns R + gamma * P @ v with terminal rows set to zero.
"""

import math

import numpy as np

import rlenvs


def two_state_mdp(gamma=0.9):
    """The running example: in each of two states, stay or go to the other one.

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


def greedy_policy(mdp, v):
    """The one-hot policy that is greedy with respect to `v`, from the policy iteration chapter.

    Picks argmax_a Q(s, a) with Q = R + gamma * P @ v, and the lowest index among ties.
    """
    q = rlenvs.q_from_v(mdp, v)
    pi = np.zeros_like(q)
    pi[np.arange(mdp.n_states), q.argmax(axis=1)] = 1.0
    return pi


def value_iteration(mdp, tol, v0=None):
    """Iterate the Bellman optimality backup V_(k+1) = max_a [R + gamma * P @ V_k].

    Start from `v0`, or from zeros when `v0` is None. Sweep synchronously, which
    means the whole new vector comes from the old one. Stop after the first
    sweep whose largest change, max |V_(k+1) - V_k|, is below `tol`.

    Args:
        mdp: an `rlenvs.TabularMDP` with gamma < 1.
        tol: the stopping threshold, a positive float.
        v0: optional starting values, shape (S,).

    Returns:
        A tuple `(v, history)`. `v` is the last iterate, shape (S,). `history`
        has shape (K + 1, S) after K sweeps: row 0 is V_0 and row K is `v`.
    """
    raise NotImplementedError("value_iteration")


def sweeps_needed(gamma, initial_error, eps):
    """The number of sweeps that the gamma^k bound says reach error `eps`.

    Args:
        gamma: the discount factor, in (0, 1).
        initial_error: max_s |V_0(s) - V*(s)|, a positive float.
        eps: the target error, a positive float.

    Returns:
        The smallest int k >= 0 with gamma^k * initial_error <= eps.
    """
    raise NotImplementedError("sweeps_needed")


def stopping_threshold(gamma, eps):
    """How small the last sweep's change must be to guarantee an error of at most `eps`.

    The error after a sweep is at most gamma / (1 - gamma) times that sweep's
    largest change. Solve gamma / (1 - gamma) * change <= eps for the change.

    Args:
        gamma: the discount factor, in (0, 1).
        eps: the error you want to guarantee, a positive float.

    Returns:
        The largest change, a float, that still guarantees the error.
    """
    raise NotImplementedError("stopping_threshold")


if __name__ == "__main__":
    mdp = two_state_mdp()
    v, history = value_iteration(mdp, tol=1e-10)
    print("Value iteration on the two-state MDP, from V = (0, 0):")
    for k in range(1, 5):
        err = np.max(np.abs(history[k] - v))
        print(f"  k={k}: V = {np.round(history[k], 3)}, error {err:.3f}, bound 20 * 0.9^k = {20 * 0.9**k:.3f}")
    print(f"  converged after {len(history) - 1} sweeps to {np.round(v, 6)}")
    print(f"  sweeps_needed(0.9, 20, 0.01) = {sweeps_needed(0.9, 20.0, 0.01)}")
    print(f"  to guarantee error 0.01, stop when a sweep changes less than {stopping_threshold(0.9, 0.01):.5f}")

    mdp = rlenvs.random_mdp(n_states=20, n_actions=4, gamma=0.95, seed=0)
    v, history = value_iteration(mdp, tol=1e-10)
    print(f"Random 20-state MDP: {len(history) - 1} sweeps; greedy policy actions {greedy_policy(mdp, v).argmax(axis=1).tolist()}")
