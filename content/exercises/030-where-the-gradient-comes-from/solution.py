"""Reference solution for the policy gradient theorem exercise."""

import numpy as np

import rlenvs


def softmax_policy(theta):
    theta = np.asarray(theta, dtype=float)
    z = theta - theta.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def log_policy_grad(theta, s, a):
    pi = softmax_policy(theta)
    grad = np.zeros_like(pi)
    grad[s] = -pi[s]
    grad[s, a] += 1.0
    return grad


def objective(mdp, theta):
    v = rlenvs.policy_evaluation(mdp, softmax_policy(theta))
    return float(mdp.mu0 @ v)


def discounted_visitation(mdp, pi):
    pi = np.asarray(pi, dtype=float)
    # Terminal states pass nothing on, so their rows of P_pi are zero.
    P_pi = np.einsum("sa,sat->st", pi, mdp.P) * (~mdp.terminal)[:, None]
    # d = mu0 + gamma * P_pi^T d, so (I - gamma P_pi)^T d = mu0.
    return np.linalg.solve((np.eye(mdp.n_states) - mdp.gamma * P_pi).T, mdp.mu0)


def advantages(mdp, pi):
    v = rlenvs.policy_evaluation(mdp, pi)
    return rlenvs.q_from_v(mdp, v) - v[:, None]


def exact_gradient(mdp, theta):
    pi = softmax_policy(theta)
    d = discounted_visitation(mdp, pi)
    return d[:, None] * pi * advantages(mdp, pi)


def score_form_gradient(mdp, theta):
    pi = softmax_policy(theta)
    d = discounted_visitation(mdp, pi)
    v = rlenvs.policy_evaluation(mdp, pi)
    q = rlenvs.q_from_v(mdp, v)
    grad = np.zeros_like(pi)
    for s in range(pi.shape[0]):
        for a in range(pi.shape[1]):
            grad += d[s] * pi[s, a] * q[s, a] * log_policy_grad(theta, s, a)
    return grad


def two_state_mdp():
    """Stay or switch between two states; staying in state 1 pays 1."""
    P = np.zeros((2, 2, 2))
    P[0, 0, 0] = 1.0
    P[0, 1, 1] = 1.0
    P[1, 0, 1] = 1.0
    P[1, 1, 0] = 1.0
    R = np.array([[0.0, 0.0], [1.0, 0.0]])
    return rlenvs.TabularMDP(P=P, R=R, gamma=0.9, mu0=np.array([1.0, 0.0]))


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
