"""Checks for the policy gradient theorem exercise."""

import numpy as np

import rlenvs
from rlcheck import check, close, fail, finite, finite_difference, has, shape


def _two_state_mdp():
    """The lesson's worked example: stay or switch between two states."""
    P = np.zeros((2, 2, 2))
    P[0, 0, 0] = 1.0  # state 0, stay
    P[0, 1, 1] = 1.0  # state 0, switch
    P[1, 0, 1] = 1.0  # state 1, stay
    P[1, 1, 0] = 1.0  # state 1, switch
    R = np.array([[0.0, 0.0], [1.0, 0.0]])
    return rlenvs.TabularMDP(P=P, R=R, gamma=0.9, mu0=np.array([1.0, 0.0]))


@check("softmax_policy turns each row of logits into probabilities")
def test_softmax_policy(sub):
    softmax_policy = has(sub, "softmax_policy")
    theta = np.array([[0.0, np.log(3.0)], [2.0, 2.0]])
    pi = softmax_policy(theta)
    shape(pi, (2, 2), label="softmax_policy output")
    close(pi, np.array([[0.25, 0.75], [0.5, 0.5]]), atol=1e-12, label="softmax of [0, log 3] and [2, 2]")

    rng = np.random.default_rng(0)
    theta = rng.normal(size=(4, 3))
    shifted = theta + rng.normal(size=(4, 1)) * 5.0
    close(softmax_policy(shifted), softmax_policy(theta), atol=1e-12,
          label="adding a constant to one state's logits must not change its row")
    big = softmax_policy(np.array([[1000.0, 0.0, -1000.0]]))
    close(big, np.array([[1.0, 0.0, 0.0]]), atol=1e-12,
          label="logits of +-1000 (subtract the row max before exp)")


@check("log_policy_grad matches finite differences, and its expectation over actions is zero")
def test_log_policy_grad(sub):
    softmax_policy = has(sub, "softmax_policy")
    log_policy_grad = has(sub, "log_policy_grad")
    rng = np.random.default_rng(1)
    theta = rng.normal(size=(3, 4))
    for s, a in [(0, 0), (1, 2), (2, 3)]:
        got = log_policy_grad(theta, s, a)
        shape(got, (3, 4), label=f"log_policy_grad(theta, {s}, {a})")
        fd = finite_difference(lambda th: np.log(softmax_policy(th.reshape(3, 4))[s, a]), theta.copy())
        close(got, fd, atol=1e-7, label=f"gradient of log pi({a} | {s})")

    pi = softmax_policy(theta)
    expected = sum(pi[1, a] * log_policy_grad(theta, 1, a) for a in range(4))
    close(expected, np.zeros((3, 4)), atol=1e-12,
          label="sum over a of pi(a | s) * grad log pi(a | s)")


@check("objective gives J = 2.25 on the two-state example and mu0 @ V on a random MDP")
def test_objective(sub):
    objective = has(sub, "objective")
    mdp = _two_state_mdp()
    close(objective(mdp, np.zeros((2, 2))), 2.25, atol=1e-10, label="J under the uniform policy")

    mdp = rlenvs.random_mdp(n_states=4, n_actions=3, gamma=0.8, seed=3)
    theta = np.random.default_rng(3).normal(size=(4, 3))
    pi = np.exp(theta) / np.exp(theta).sum(axis=1, keepdims=True)
    P_pi = np.einsum("sa,sat->st", pi, mdp.P)
    v = np.linalg.solve(np.eye(4) - 0.8 * P_pi, (pi * mdp.R).sum(axis=1))
    close(objective(mdp, theta), mdp.mu0 @ v, atol=1e-10, label="J on a random 4-state MDP")


@check("exact_gradient reproduces the lesson's hand-worked two-state gradient")
def test_exact_gradient_worked_example(sub):
    exact_gradient = has(sub, "exact_gradient")
    got = exact_gradient(_two_state_mdp(), np.zeros((2, 2)))
    shape(got, (2, 2), label="exact_gradient output")
    # d = [5.5, 4.5], pi = 0.5, A = [[-0.225, 0.225], [0.725, -0.725]].
    want = np.array([[-0.61875, 0.61875], [1.63125, -1.63125]])
    close(got, want, atol=1e-10, label="d(s) * pi(a | s) * A(s, a)")


@check("exact_gradient (the policy gradient theorem) matches finite differences of J")
def test_exact_gradient_finite_difference(sub):
    objective = has(sub, "objective")
    exact_gradient = has(sub, "exact_gradient")
    for seed in range(4):
        mdp = rlenvs.random_mdp(n_states=3 + seed, n_actions=2 + seed % 2, gamma=0.9, seed=seed)
        S, A = mdp.n_states, mdp.n_actions
        theta = np.random.default_rng(10 + seed).normal(size=(S, A))
        fd = finite_difference(lambda th: objective(mdp, th.reshape(S, A)), theta.copy())
        got = exact_gradient(mdp, theta)
        shape(got, (S, A), label=f"exact_gradient on MDP seed {seed}")
        close(got, fd, atol=1e-7, label=f"theorem against finite differences, MDP seed {seed}")


@check("discounted_returns computes G_t backwards along the last axis")
def test_discounted_returns(sub):
    discounted_returns = has(sub, "discounted_returns")
    got = discounted_returns(np.array([0.0, 1.0, 0.0]), 0.9)
    close(got, np.array([0.9, 1.0, 0.0]), atol=1e-12, label="rewards [0, 1, 0], gamma 0.9")
    got = discounted_returns(np.array([1.0, 2.0, 3.0]), 0.5)
    close(got, np.array([2.75, 3.5, 3.0]), atol=1e-12, label="rewards [1, 2, 3], gamma 0.5")
    batch = discounted_returns(np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 3.0]]), 0.5)
    close(batch, np.array([[0.5, 1.0, 0.0], [2.75, 3.5, 3.0]]), atol=1e-12,
          label="a batch of two episodes, shape (2, 3)")


@check("reinforce_estimate on one hand-worked episode weights each score by gamma^t G_t")
def test_reinforce_estimate(sub):
    reinforce_estimate = has(sub, "reinforce_estimate")
    # s = [0, 1, 1], a = [switch, stay, switch], r = [0, 1, 0], gamma = 0.9.
    # G = [0.9, 1.0, 0.0], gamma^t G_t = [0.9, 0.9, 0.0].
    got = reinforce_estimate(np.zeros((2, 2)), [0, 1, 1], [1, 0, 1], [0.0, 1.0, 0.0], 0.9)
    shape(got, (2, 2), label="reinforce_estimate output")
    want = np.array([[-0.45, 0.45], [0.45, -0.45]])
    if np.allclose(got, np.array([[-0.45, 0.45], [0.5, -0.5]]), atol=1e-9):
        fail("the second step used G_1 = 1.0 without the gamma^1 = 0.9 weight; "
             f"expected {want.tolist()}")
    close(got, want, atol=1e-12, label="single-episode REINFORCE estimate")


@check("REINFORCE's mean lies within 4 standard errors of the exact gradient in every component")
def test_reinforce_unbiased(sub):
    exact_gradient = has(sub, "exact_gradient")
    reinforce_gradient = has(sub, "reinforce_gradient")
    mdp = rlenvs.random_mdp(n_states=3, n_actions=2, gamma=0.9, seed=0)
    theta = np.random.default_rng(1).normal(size=(3, 2))
    N = 2000
    est = np.asarray(reinforce_gradient(mdp, theta, episodes=N, horizon=100, seed=0))
    shape(est, (N, 3, 2), label="reinforce_gradient output")
    finite(est, label="reinforce_gradient output")
    exact = exact_gradient(mdp, theta)
    mean = est.mean(axis=0)
    stderr = est.std(axis=0, ddof=1) / np.sqrt(N)
    if np.any(stderr <= 0):
        fail("the per-episode estimates do not vary; each row should come from its own episode")
    z = (mean - exact) / stderr
    worst = np.unravel_index(np.argmax(np.abs(z)), z.shape)
    if np.abs(z[worst]) > 4.0:
        fail(
            f"component (s={worst[0]}, a={worst[1]}): REINFORCE mean {mean[worst]:.4f}, "
            f"exact {exact[worst]:.4f}, standard error {stderr[worst]:.4f}, so z = {z[worst]:.1f}. "
            "A biased estimate often means the gamma^t weight is missing."
        )
