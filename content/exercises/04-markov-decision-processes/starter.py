"""Markov decision processes: Bellman equations as linear algebra.

Fill in the functions in order. Each one is a few lines of numpy. They turn a
policy into a Markov chain, solve for its values exactly, turn state values
into action values, apply the Bellman optimality backup, and compute where
the policy spends its (discounted) time.

An `rlenvs.TabularMDP` carries `P` of shape (S, A, S), `R` of shape (S, A),
`gamma`, `mu0` of shape (S,), and `terminal` of shape (S,). A policy `pi` has
shape (S, A), and each row sums to one.
"""

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
    """The Markov chain and expected rewards that a policy induces.

    Args:
        mdp: an `rlenvs.TabularMDP`.
        pi: the policy, shape (S, A).

    Returns:
        A tuple `(P_pi, r_pi)`. `P_pi[s, s2]` is sum_a pi(a | s) P(s2 | s, a),
        shape (S, S). `r_pi[s]` is sum_a pi(a | s) R(s, a), shape (S,). Rows of
        terminal states are all zero in both, because nothing happens after
        an episode ends.
    """
    raise NotImplementedError("policy_matrices")


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


def bellman_optimality_backup(mdp, v):
    """One application of the Bellman optimality operator: max over a of Q(s, a).

    Args:
        mdp: an `rlenvs.TabularMDP`.
        v: state values, shape (S,).

    Returns:
        (T* v)(s) = max_a [R(s, a) + gamma sum_s2 P(s2 | s, a) v(s2)], shape (S,).
    """
    raise NotImplementedError("bellman_optimality_backup")


def discounted_visitation(mdp, pi):
    """The normalised discounted state-visitation distribution of `pi` from `mu0`.

    d(s) = (1 - gamma) sum_t gamma^t Pr(s_t = s), which in matrix form is
    d = (1 - gamma) mu0^T (I - gamma P_pi)^-1.

    Args:
        mdp: an `rlenvs.TabularMDP` with gamma < 1.
        pi: the policy, shape (S, A).

    Returns:
        d, shape (S,), divided by its sum so that it sums to one.
    """
    raise NotImplementedError("discounted_visitation")


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
