"""Reference solution: greedy improvement and policy iteration.

The model is known, so every function works directly on the arrays of an
`rlenvs.TabularMDP`: `P` of shape (S, A, S), `R` of shape (S, A), `gamma`, and
`terminal` of shape (S,). A policy `pi` has shape (S, A), and each row sums to
one.

`two_state_mdp` and `policy_matrices` are given. You can also call
`rlenvs.q_from_v(mdp, v)`, which returns R + gamma * P @ v with terminal rows
set to zero.
"""

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


def policy_matrices(mdp, pi):
    """P_pi of shape (S, S) and r_pi of shape (S,), with terminal rows set to zero."""
    live = ~mdp.terminal
    P_pi = np.einsum("sa,sat->st", pi, mdp.P) * live[:, None]
    r_pi = (pi * mdp.R).sum(axis=1) * live
    return P_pi, r_pi


def greedy_policy(mdp, v):
    """The deterministic policy that is greedy with respect to state values `v`.

    Args:
        mdp: an `rlenvs.TabularMDP`.
        v: state values, shape (S,).

    Returns:
        A one-hot policy, shape (S, A), that picks argmax_a Q(s, a) with
        Q = R + gamma * P @ v. Among tied actions, pick the lowest index.
    """
    q = rlenvs.q_from_v(mdp, v)
    pi = np.zeros_like(q)
    pi[np.arange(mdp.n_states), q.argmax(axis=1)] = 1.0
    return pi


def improve_policy(mdp, pi):
    """One round of policy iteration: evaluate `pi` exactly, then act greedily on it.

    Solve (I - gamma * P_pi) v = r_pi for the value of `pi`, then build the
    greedy policy with respect to that value.

    Args:
        mdp: an `rlenvs.TabularMDP` with gamma < 1.
        pi: the current policy, shape (S, A).

    Returns:
        A tuple `(new_pi, v)`: the greedy policy, shape (S, A), and the value
        of the policy you started with, `v`, shape (S,).
    """
    P_pi, r_pi = policy_matrices(mdp, pi)
    v = np.linalg.solve(np.eye(mdp.n_states) - mdp.gamma * P_pi, r_pi)
    return greedy_policy(mdp, v), v


def policy_iteration(mdp, pi0=None):
    """Alternate exact evaluation and greedy improvement until the policy is stable.

    Start from `pi0`, or from "action 0 in every state" when `pi0` is None.
    Call `improve_policy` in a loop, and stop when the greedy policy equals
    the current one.

    Args:
        mdp: an `rlenvs.TabularMDP` with gamma < 1.
        pi0: optional starting policy, shape (S, A).

    Returns:
        A tuple `(pi, v, values)`: the final policy, shape (S, A); its value,
        shape (S,); and the value of every policy evaluated, in order, shape
        (n_evaluations, S). The last row of `values` is `v`.
    """
    if pi0 is None:
        pi0 = np.zeros((mdp.n_states, mdp.n_actions))
        pi0[:, 0] = 1.0
    pi = np.asarray(pi0, dtype=float)
    values = []
    while True:
        new_pi, v = improve_policy(mdp, pi)
        values.append(v)
        if np.array_equal(new_pi, pi):
            return pi, v, np.array(values)
        pi = new_pi


if __name__ == "__main__":
    mdp = two_state_mdp()
    pi, v, values = policy_iteration(mdp)
    print("Policy iteration on the two-state MDP, from 'stay everywhere':")
    for k, row in enumerate(values):
        print(f"  evaluation {k + 1}: V = {np.round(row, 3).tolist()}")
    print("  final policy (rows are states, columns stay and go):", pi.tolist())

    mdp = rlenvs.random_mdp(n_states=20, n_actions=4, gamma=0.95, seed=0)
    pi, v, values = policy_iteration(mdp)
    v_star = rlenvs.value_iteration(mdp)
    print(f"Random 20-state MDP: {len(values)} evaluations, "
          f"largest gap to the optimal values {np.max(np.abs(v - v_star)):.1e}")
