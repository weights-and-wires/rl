"""Dynamic programming: policy evaluation, policy iteration, and value iteration.

Fill in the functions in order. The model is known, so every function works
directly on the arrays of an `rlenvs.TabularMDP`: `P` of shape (S, A, S), `R`
of shape (S, A), `gamma`, and `terminal` of shape (S,). A policy `pi` has shape
(S, A), and each row sums to one.

`policy_matrices` and `two_state_mdp` are given. You can also call
`rlenvs.q_from_v(mdp, v)`, which returns R + gamma * P @ v with terminal rows
set to zero.
"""

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
    """Evaluate `pi` by repeated synchronous Bellman expectation sweeps from V = 0.

    Each sweep computes the whole new vector from the old one:
    V_new = r_pi + gamma * P_pi @ V. Stop after the first sweep whose largest
    change, max |V_new - V|, is below `tol`.

    Args:
        mdp: an `rlenvs.TabularMDP` with gamma < 1.
        pi: the policy, shape (S, A).
        tol: the stopping threshold, a positive float.

    Returns:
        A tuple `(v, sweeps)`: the last sweep's values, shape (S,), and the
        number of sweeps performed, an int.
    """
    raise NotImplementedError("iterative_policy_evaluation")


def greedy_policy(mdp, v):
    """The deterministic policy that is greedy with respect to state values `v`.

    Args:
        mdp: an `rlenvs.TabularMDP`.
        v: state values, shape (S,).

    Returns:
        A one-hot policy, shape (S, A), that picks argmax_a Q(s, a) with
        Q = R + gamma * P @ v. Among tied actions, pick the lowest index.
    """
    raise NotImplementedError("greedy_policy")


def value_iteration(mdp, tol, v0=None):
    """Iterate the Bellman optimality backup V_(k+1) = max_a [R + gamma * P @ V_k].

    Start from `v0`, or from zeros when `v0` is None. Sweep synchronously, and
    stop after the first sweep whose largest change is below `tol`.

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


def policy_iteration(mdp, pi0=None):
    """Alternate exact policy evaluation and greedy improvement until the policy is stable.

    Start from `pi0`, or from "action 0 in every state" when `pi0` is None.
    Each round: evaluate the policy exactly with a linear solve, then take the
    greedy policy with respect to that value. Stop when the greedy policy
    equals the current one.

    Args:
        mdp: an `rlenvs.TabularMDP` with gamma < 1.
        pi0: optional starting policy, shape (S, A).

    Returns:
        A tuple `(pi, v, values)`: the final policy, shape (S, A); its value,
        shape (S,); and the value of every policy evaluated, in order, shape
        (n_evaluations, S).
    """
    raise NotImplementedError("policy_iteration")


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
