"""Checks for the policy gradient theorem exercise."""

import numpy as np

import rlenvs
from rlcheck import almost, check, close, fail, finite_difference, has, shape


def _two_state_mdp():
    """The chapter's worked example: stay or switch between two states."""
    P = np.zeros((2, 2, 2))
    P[0, 0, 0] = 1.0  # state 0, stay
    P[0, 1, 1] = 1.0  # state 0, switch
    P[1, 0, 1] = 1.0  # state 1, stay
    P[1, 1, 0] = 1.0  # state 1, switch
    R = np.array([[0.0, 0.0], [1.0, 0.0]])
    return rlenvs.TabularMDP(P=P, R=R, gamma=0.9, mu0=np.array([1.0, 0.0]))


@check("discounted_visitation gives d = (5.5, 4.5) on the two-state example, summing to 1 / (1 - gamma)")
def test_visitation_example(sub):
    discounted_visitation = has(sub, "discounted_visitation")
    mdp = _two_state_mdp()
    d = np.asarray(discounted_visitation(mdp, np.full((2, 2), 0.5)), dtype=float)
    shape(d, (2,), label="discounted_visitation output")
    close(d, [5.5, 4.5], atol=1e-10, label="visitation under the uniform policy")
    almost(d.sum(), 10.0, 1e-10, label="total visitation")


@check("discounted_visitation equals the series sum over t of gamma^t mu0 P_pi^t and obeys the flow equation")
def test_visitation_series(sub):
    discounted_visitation = has(sub, "discounted_visitation")
    mdp = rlenvs.random_mdp(n_states=4, n_actions=3, gamma=0.8, seed=2)
    pi = np.random.default_rng(2).dirichlet(np.ones(3), size=4)
    d = np.asarray(discounted_visitation(mdp, pi), dtype=float)
    P_pi = np.einsum("sa,sat->st", pi, mdp.P)
    series, p_t = np.zeros(4), mdp.mu0.copy()
    for t in range(400):
        series += 0.8**t * p_t
        p_t = p_t @ P_pi
    close(d, series, atol=1e-9, label="visitation against the sum of gamma^t Pr(s_t = s) over 400 steps")
    close(d, mdp.mu0 + 0.8 * P_pi.T @ d, atol=1e-10, label="flow equation d = mu0 + gamma P_pi^T d")


@check("the visitation turns J into a weighted reward: J = sum_s d(s) r_pi(s)")
def test_visitation_return(sub):
    discounted_visitation = has(sub, "discounted_visitation")
    mdp = rlenvs.random_mdp(n_states=5, n_actions=2, gamma=0.9, seed=5)
    pi = np.random.default_rng(5).dirichlet(np.ones(2), size=5)
    d = np.asarray(discounted_visitation(mdp, pi), dtype=float)
    v = rlenvs.policy_evaluation(mdp, pi)
    close(d @ (pi * mdp.R).sum(axis=1), mdp.mu0 @ v, atol=1e-9,
          label="sum of d(s) * r_pi(s) against mu0 @ V")


@check("advantages match the hand table and average to zero under the policy")
def test_advantages(sub):
    advantages = has(sub, "advantages")
    adv = np.asarray(advantages(_two_state_mdp(), np.full((2, 2), 0.5)), dtype=float)
    shape(adv, (2, 2), label="advantages output")
    close(adv, [[-0.225, 0.225], [0.725, -0.725]], atol=1e-10, label="advantages on the two-state example")
    mdp = rlenvs.random_mdp(n_states=4, n_actions=3, gamma=0.9, seed=1)
    pi = np.random.default_rng(1).dirichlet(np.ones(3), size=4)
    close((pi * np.asarray(advantages(mdp, pi))).sum(axis=1), np.zeros(4), atol=1e-10,
          label="sum over a of pi(a | s) * A(s, a)")


@check("exact_gradient reproduces the hand-worked two-state gradient")
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


@check("score_form_gradient, with explicit scores and Q-values, equals the simplified d * pi * A")
def test_score_form(sub):
    score_form_gradient = has(sub, "score_form_gradient")
    exact_gradient = has(sub, "exact_gradient")
    for seed in range(3):
        mdp = rlenvs.random_mdp(n_states=4, n_actions=3, gamma=0.9, seed=seed)
        theta = np.random.default_rng(20 + seed).normal(size=(4, 3))
        got = np.asarray(score_form_gradient(mdp, theta), dtype=float)
        shape(got, (4, 3), label="score_form_gradient output")
        if not np.allclose(got, exact_gradient(mdp, theta), atol=1e-9):
            fail("sum of d(s) pi(a | s) Q(s, a) grad log pi(a | s) should equal d(s) pi(a | s) A(s, a), "
                 "because the score averages to zero and Q - V leaves the same gradient. "
                 f"On MDP seed {seed} the largest gap is {np.abs(got - exact_gradient(mdp, theta)).max():.3g}")
