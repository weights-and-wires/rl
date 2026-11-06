"""The Bellman equation: solve for a policy's values exactly.

Fill in the two functions. An `rlenvs.TabularMDP` carries `P` of shape
(S, A, S), `R` of shape (S, A), `gamma`, `mu0` of shape (S,), and `terminal` of
shape (S,). A policy `pi` has shape (S, A), and each row sums to one.

`hallway_mdp` and `policy_matrices` come from the previous chapter. They are
complete; don't change them.
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

    Given in full from the previous chapter.
    """
    live = ~mdp.terminal
    P_pi = np.einsum("sa,sat->st", pi, mdp.P) * live[:, None]
    r_pi = (pi * mdp.R).sum(axis=1) * live
    return P_pi, r_pi


def evaluate_exact(mdp, pi):
    """The exact value of `pi` from the linear system (I - gamma P_pi) V = r_pi.

    Args:
        mdp: an `rlenvs.TabularMDP` with gamma < 1.
        pi: the policy, shape (S, A).

    Returns:
        V, shape (S,). Terminal states get 0.
    """
    raise NotImplementedError("evaluate_exact")


def q_from_v(mdp, v):
    """Action values from state values: Q(s, a) = R(s, a) + gamma sum_s2 P(s2 | s, a) v(s2).

    Args:
        mdp: an `rlenvs.TabularMDP`.
        v: state values, shape (S,).

    Returns:
        Q, shape (S, A), with every entry of a terminal state's row set to 0.
    """
    raise NotImplementedError("q_from_v")


if __name__ == "__main__":
    mdp = hallway_mdp()
    right = np.tile([0.0, 1.0], (4, 1))
    print("Always right: V =", np.round(evaluate_exact(mdp, right), 4))

    uniform = rlenvs.uniform_policy(mdp)
    v = evaluate_exact(mdp, uniform)
    print("Uniform policy: V =", np.round(v, 4))
    print("Action values (columns: left, right):")
    print(np.round(q_from_v(mdp, v), 4))

    big = rlenvs.random_mdp(n_states=6, n_actions=3, gamma=0.9, seed=0)
    pi = rlenvs.uniform_policy(big)
    gap = np.max(np.abs(evaluate_exact(big, pi) - rlenvs.policy_evaluation(big, pi)))
    print(f"Random 6-state MDP: largest gap from rlenvs.policy_evaluation = {gap:.2e}")
