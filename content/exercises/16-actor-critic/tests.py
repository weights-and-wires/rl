"""Checks for the baselines and actor-critic exercise."""

import numpy as np

import rlenvs
from rlcheck import check, close, fail, finite, finite_difference, has, shape

THETA = np.log(np.array([0.25, 0.25, 0.5]))
MEANS = np.array([1.0, 0.0, 2.0])


def _softmax(x):
    z = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def _pull(theta, means, sigma, n, seed):
    rng = np.random.default_rng(seed)
    pi = _softmax(theta)
    arms = rng.choice(len(pi), size=n, p=pi)
    rewards = means[arms] + sigma * rng.standard_normal(n)
    return arms, rewards


@check("bandit_gradient_samples scales each sample's score by its reward minus the baseline")
def test_bandit_gradient_samples(sub):
    samples = has(sub, "bandit_gradient_samples")
    got = samples(THETA, np.array([2, 0]), np.array([3.0, -1.0]), 1.0)
    shape(got, (2, 3), label="two samples over three arms")
    # (3 - 1) * ([0, 0, 1] - pi) and (-1 - 1) * ([1, 0, 0] - pi).
    want = np.array([[-0.5, -0.5, 1.0], [-1.5, 0.5, 1.0]])
    close(got, want, atol=1e-12, label="per-sample estimates")


@check("exact_bandit_gradient matches finite differences of J(theta) = sum_a pi(a) mean(a)")
def test_exact_bandit_gradient(sub):
    exact = has(sub, "exact_bandit_gradient")
    close(exact(THETA, MEANS), np.array([-0.0625, -0.3125, 0.375]), atol=1e-12,
          label="the lesson's three-arm example")
    rng = np.random.default_rng(1)
    theta, means = rng.normal(size=5), rng.normal(size=5)
    fd = finite_difference(lambda t: _softmax(t) @ means, theta.copy())
    close(exact(theta, means), fd, atol=1e-8, label="exact gradient against finite differences")


@check("With no baseline, b = V, or any other constant, the sample mean stays within 4 standard errors of the gradient")
def test_baseline_unbiased(sub):
    samples = has(sub, "bandit_gradient_samples")
    exact = has(sub, "exact_bandit_gradient")
    arms, rewards = _pull(THETA, MEANS, 1.0, 40000, seed=2)
    g = exact(THETA, MEANS)
    for b in (0.0, 1.25, 0.95, -3.0):
        est = np.asarray(samples(THETA, arms, rewards, b))
        finite(est, label=f"samples with baseline {b}")
        mean = est.mean(axis=0)
        se = est.std(axis=0, ddof=1) / np.sqrt(len(est))
        z = np.abs(mean - g) / se
        if not np.all(z < 4.0):
            fail(f"with baseline {b} the mean {np.round(mean, 4).tolist()} is "
                 f"{z.max():.1f} standard errors from the exact {g.tolist()}")


@check("estimator_variance matches the lesson's example and the variance of sampled estimates")
def test_estimator_variance(sub):
    variance = has(sub, "estimator_variance")
    samples = has(sub, "bandit_gradient_samples")
    close(variance(THETA, MEANS, 1.0, 0.0), 1.3515625, atol=1e-10, label="no baseline")
    close(variance(THETA, MEANS, 1.0, 1.25), 0.84375, atol=1e-10, label="baseline b = V = 1.25")

    rng = np.random.default_rng(3)
    theta, means, sigma = rng.normal(size=4), rng.normal(size=4) * 2, 0.5
    arms, rewards = _pull(theta, means, sigma, 200000, seed=4)
    for b in (0.0, 1.0):
        est = samples(theta, arms, rewards, b)
        empirical = est.var(axis=0, ddof=1).sum()
        close(variance(theta, means, sigma, b), empirical, rtol=0.03,
              label=f"exact total variance against 200000 samples, baseline {b}")


@check("optimal_baseline gives b* = 0.95 on the example and minimises the variance")
def test_optimal_baseline(sub):
    optimal = has(sub, "optimal_baseline")
    variance = has(sub, "estimator_variance")
    close(optimal(THETA, MEANS), 0.95, atol=1e-10, label="b* for pi = [0.25, 0.25, 0.5], means [1, 0, 2]")
    rng = np.random.default_rng(5)
    for _ in range(3):
        theta, means = rng.normal(size=4), rng.normal(size=4) * 2
        b = optimal(theta, means)
        slope = finite_difference(lambda x: variance(theta, means, 1.0, float(x[0])), np.array([b]))
        close(slope, np.zeros(1), atol=1e-6, label="d variance / d b at b*")
        mean_reward = float(_softmax(theta) @ means)
        if not variance(theta, means, 1.0, b) <= variance(theta, means, 1.0, mean_reward) + 1e-12:
            fail("the variance at b* should be no larger than at b = V")


@check("expected_td_error with the true V^pi equals the advantage table A^pi")
def test_expected_td_error(sub):
    expected_td_error = has(sub, "expected_td_error")
    mdp = rlenvs.random_mdp(n_states=4, n_actions=3, gamma=0.9, seed=6)
    pi = _softmax(np.random.default_rng(6).normal(size=(4, 3)))
    v = rlenvs.policy_evaluation(mdp, pi)
    adv = rlenvs.q_from_v(mdp, v) - v[:, None]
    got = expected_td_error(mdp, v)
    shape(got, (4, 3), label="expected_td_error output")
    close(got, adv, atol=1e-10, label="E[delta | s, a] with v = V^pi")
    close(expected_td_error(mdp, np.zeros(4)), mdp.R, atol=1e-12, label="with v = 0 it is R")


@check("expected_actor_update is (1 - gamma) times the gradient of J with the true critic, even shifted by a constant")
def test_expected_actor_update(sub):
    update = has(sub, "expected_actor_update")
    mdp = rlenvs.random_mdp(n_states=4, n_actions=3, gamma=0.9, seed=7)
    theta = np.random.default_rng(7).normal(size=(4, 3))
    pi = _softmax(theta)
    v = rlenvs.policy_evaluation(mdp, pi)

    def J(th):
        return mdp.mu0 @ rlenvs.policy_evaluation(mdp, _softmax(th.reshape(4, 3)))

    target = (1 - mdp.gamma) * finite_difference(J, theta.copy())
    close(update(mdp, theta, v), target, atol=1e-8, label="critic v = V^pi")
    close(update(mdp, theta, v + 3.0), target, atol=1e-8,
          label="critic v = V^pi + 3 (a constant error acts as a baseline)")
    noisy = v + np.random.default_rng(8).normal(size=4)
    gap = np.abs(update(mdp, theta, noisy) - target).max()
    if not gap > 1e-3:
        fail(f"a critic with state-dependent errors should bias the update, but the gap is {gap:.2g}")


@check("actor_critic with alpha_actor = 0 leaves theta at zero and learns V^pi of the uniform policy")
def test_actor_critic_critic_only(sub):
    actor_critic = has(sub, "actor_critic")
    mdp = rlenvs.random_mdp(n_states=5, n_actions=3, gamma=0.9, seed=0)
    theta, v = actor_critic(mdp, steps=20000, alpha_actor=0.0, alpha_critic=0.01, seed=0)
    shape(theta, (5, 3), label="theta")
    shape(v, (5,), label="v")
    close(theta, np.zeros((5, 3)), atol=0.0, label="theta with alpha_actor = 0")
    want = rlenvs.policy_evaluation(mdp, rlenvs.uniform_policy(mdp))
    err = np.abs(np.asarray(v) - want).max()
    if not err < 0.4:
        fail(f"the critic should reach V^pi = {np.round(want, 3).tolist()} within 0.4, "
             f"got {np.round(v, 3).tolist()} (largest error {err:.3f})")


@check("actor_critic closes at least 90% of the gap from the uniform policy to the optimum")
def test_actor_critic_learns(sub):
    actor_critic = has(sub, "actor_critic")
    mdp = rlenvs.random_mdp(n_states=5, n_actions=3, gamma=0.9, seed=0)
    theta, _ = actor_critic(mdp, steps=5000, alpha_actor=0.5, alpha_critic=0.1, seed=0)
    finite(theta, label="theta")
    start = mdp.mu0 @ rlenvs.policy_evaluation(mdp, rlenvs.uniform_policy(mdp))
    best = mdp.mu0 @ rlenvs.value_iteration(mdp)
    got = mdp.mu0 @ rlenvs.policy_evaluation(mdp, _softmax(np.asarray(theta)))
    frac = (got - start) / (best - start)
    if not frac >= 0.9:
        fail(f"J went from {start:.3f} to {got:.3f}, against an optimum of {best:.3f}: "
             f"{100 * frac:.0f}% of the gap, below 90%")
