"""The policy gradient theorem, computed exactly.

`softmax_policy`, `log_policy_grad`, `objective`, and `two_state_mdp` are given
in full, because you wrote the first three in the previous chapter. You
write the discounted visitation, the advantage table, and the theorem's
gradient, in two forms, then check them against finite differences of J.
"""

import numpy as np

import rlenvs


def softmax_policy(theta):
    """Given. The tabular softmax policy, shape (S, A), from logits theta, shape (S, A)."""
    theta = np.asarray(theta, dtype=float)
    z = theta - theta.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def log_policy_grad(theta, s, a):
    """Given. The score: the gradient of log pi(a | s) with respect to every logit, shape (S, A)."""
    pi = softmax_policy(theta)
    grad = np.zeros_like(pi)
    grad[s] = -pi[s]
    grad[s, a] += 1.0
    return grad


def objective(mdp, theta):
    """Given. The exact objective J(theta) = sum_s mu0(s) V^pi(s), as a float."""
    v = rlenvs.policy_evaluation(mdp, softmax_policy(theta))
    return float(mdp.mu0 @ v)


def two_state_mdp():
    """Given. Stay or switch between two states; staying in state 1 pays 1."""
    P = np.zeros((2, 2, 2))
    P[0, 0, 0] = 1.0
    P[0, 1, 1] = 1.0
    P[1, 0, 1] = 1.0
    P[1, 1, 0] = 1.0
    R = np.array([[0.0, 0.0], [1.0, 0.0]])
    return rlenvs.TabularMDP(P=P, R=R, gamma=0.9, mu0=np.array([1.0, 0.0]))


def discounted_visitation(mdp, pi):
    """The unnormalised discounted visitation d(s) = sum_t gamma^t Pr(s_t = s).

    It is the expected discounted number of visits to each state. It solves
    d = mu0 + gamma * P_pi^T d, where P_pi[s, s2] = sum_a pi[s, a] P[s, a, s2].
    Terminal states pass nothing on: zero their rows of P_pi.

    Args:
        mdp: a `rlenvs.TabularMDP` with gamma < 1.
        pi: the policy, shape (S, A).

    Returns:
        d, shape (S,). With no terminal states it sums to 1 / (1 - gamma).
    """
    raise NotImplementedError("discounted_visitation")


def advantages(mdp, pi):
    """The advantage table A(s, a) = Q(s, a) - V(s).

    Args:
        mdp: a `rlenvs.TabularMDP`.
        pi: the policy, shape (S, A).

    Returns:
        A, shape (S, A). Each row averages to zero under pi[s].
    """
    raise NotImplementedError("advantages")


def exact_gradient(mdp, theta):
    """The policy gradient theorem for a tabular softmax policy.

    dJ / dtheta[s, a] = d(s) * pi(a | s) * A(s, a).

    Args:
        mdp: a `rlenvs.TabularMDP` with gamma < 1.
        theta: logits, shape (S, A).

    Returns:
        The gradient, shape (S, A).
    """
    raise NotImplementedError("exact_gradient")


def score_form_gradient(mdp, theta):
    """The theorem before it is simplified.

    sum_s d(s) sum_a pi(a | s) Q(s, a) grad log pi(a | s), with the scores
    from `log_policy_grad`. Loop over every state and action and add up the
    (S, A) terms. It must equal `exact_gradient`.

    Returns:
        The gradient, shape (S, A).
    """
    raise NotImplementedError("score_form_gradient")


if __name__ == "__main__":
    mdp = two_state_mdp()
    theta = np.zeros((2, 2))
    pi = softmax_policy(theta)
    print("visitation d:", np.round(discounted_visitation(mdp, pi), 4))
    print("advantages:  ", np.round(advantages(mdp, pi), 4).tolist())
    print("gradient:    ", np.round(exact_gradient(mdp, theta), 5).tolist())
    print("score form:  ", np.round(score_form_gradient(mdp, theta), 5).tolist())

    mdp = rlenvs.random_mdp(n_states=3, n_actions=2, gamma=0.9, seed=0)
    theta = np.random.default_rng(1).normal(size=(3, 2))
    exact = exact_gradient(mdp, theta)
    eps = 1e-6
    fd = np.zeros_like(theta)
    for s in range(3):
        for a in range(2):
            up, down = theta.copy(), theta.copy()
            up[s, a] += eps
            down[s, a] -= eps
            fd[s, a] = (objective(mdp, up) - objective(mdp, down)) / (2 * eps)
    print(f"largest gap from finite differences on a random MDP: {np.abs(exact - fd).max():.2e}")
