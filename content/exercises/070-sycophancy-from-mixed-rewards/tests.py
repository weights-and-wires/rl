"""Checks for the sycophancy exercise."""

import itertools

import numpy as np
from scipy.optimize import minimize

from rlcheck import check, close, fail, has

K, B, BETA, RHO = 1.0, 3.0, 1.0, 0.8
UNIFORM = np.full((2, 2), 0.5)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _data(sub, n=4000, seed=0):
    # The given simulator lives in the reader's module, so the data come from it.
    return sub.simulate_comparisons([K, B], n, RHO, seed)


@check("mixed_rewards at w = 0.5 gives [[2.5, 0], [1.5, 1]], and w = 0 gives the truth reward")
def test_mixed_rewards(sub):
    mixed_rewards = has(sub, "mixed_rewards")
    # Right user: agree 0.5 * 1 + 0.5 * (1 + 3) = 2.5, correct 0.
    # Wrong user: agree 0.5 * 3 = 1.5, correct 0.5 * 1 + 0.5 * 1 = 1.
    close(mixed_rewards(0.5, K, B), [[2.5, 0.0], [1.5, 1.0]], atol=1e-12, label="w = 0.5")
    close(mixed_rewards(0.0, K, B), [[1.0, 0.0], [0.0, 1.0]], atol=1e-12, label="w = 0")
    close(mixed_rewards(1.0, K, B), [[4.0, 0.0], [3.0, 1.0]], atol=1e-12, label="w = 1")


@check("fit_bradley_terry returns weights where the likelihood's gradient is zero")
def test_fit_gradient(sub):
    fit = has(sub, "fit_bradley_terry")
    fa, fb, y = _data(sub)
    theta = np.asarray(fit(fa, fb, y), dtype=float)
    x = fa - fb
    gradient = x.T @ (y - _sigmoid(x @ theta))
    close(gradient, np.zeros(2), atol=1e-6, label="gradient of the log-likelihood at theta")


@check("fit_bradley_terry recovers the rater's weights and the agreement bias from 4000 comparisons")
def test_fit_recovers_bias(sub):
    fit = has(sub, "fit_bradley_terry")
    fa, fb, y = _data(sub)
    theta = np.asarray(fit(fa, fb, y), dtype=float)
    if theta.shape != (2,):
        fail(f"theta should have shape (2,), got {theta.shape}")
    if abs(theta[1] - B) > 0.35:
        fail(f"the fitted agreement bias should be near {B}, got {theta[1]:.3f}")
    if abs(theta[0] - K) > 0.35:
        fail(f"the fitted weight on correctness should be near {K}, got {theta[0]:.3f}")


@check("optimal_policy matches a numeric optimizer of reward minus beta times KL")
def test_optimal_policy_numeric(sub):
    optimal_policy = has(sub, "optimal_policy")
    rng = np.random.default_rng(0)
    p_ref = rng.dirichlet([2.0, 2.0], size=2)
    reward = rng.normal(size=(2, 2)) * 2.0
    for beta in (0.3, 1.0, 2.5):
        got = np.asarray(optimal_policy(p_ref, reward, beta), dtype=float)
        close(got.sum(axis=1), np.ones(2), atol=1e-12, label="rows sum to 1")

        def negative_objective(z, beta=beta):
            pi = np.exp(z.reshape(2, 2))
            pi = pi / pi.sum(axis=1, keepdims=True)
            kl = (pi * np.log(pi / p_ref)).sum(axis=1)
            return -(pi * reward).sum() + beta * kl.sum()

        best = minimize(negative_objective, np.zeros(4), method="BFGS", options={"gtol": 1e-10})
        want = np.exp(best.x.reshape(2, 2))
        want = want / want.sum(axis=1, keepdims=True)
        close(got, want, atol=1e-4, label=f"closed form against the optimizer at beta = {beta}")


@check("optimal_policy on the chapter's example agrees with a wrong user 62.25% of the time at w = 0.5")
def test_optimal_policy_example(sub):
    optimal_policy = has(sub, "optimal_policy")
    mixed_rewards = has(sub, "mixed_rewards")
    pi = np.asarray(optimal_policy(UNIFORM, mixed_rewards(0.5, K, B), BETA))
    # Right user: sigmoid(2.5 - 0) = 0.9241. Wrong user: sigmoid(1.5 - 1) = 0.6225.
    close(pi, [[0.924142, 0.075858], [0.622459, 0.377541]], atol=1e-5)
    # A very large beta keeps the reference policy.
    close(optimal_policy(UNIFORM, mixed_rewards(0.5, K, B), 1e6), UNIFORM, atol=1e-4,
          label="a huge beta returns the reference")


@check("critical_weight matches its formula, 1 / (1 + b - k) for a 50-50 reference, and is inf when b is too small")
def test_critical_weight_formula(sub):
    critical_weight = has(sub, "critical_weight")
    close(critical_weight(K, B, BETA, 0.5), 1.0 / 3.0, atol=1e-12, label="k = 1, b = 3, p = 0.5")
    # A reference that already agrees 70% of the time starts closer to the edge.
    want = (1.0 - 0.5 * np.log(0.7 / 0.3)) / (1.0 + 2.0 - 0.5)
    close(critical_weight(0.5, 2.0, 0.5, 0.7), want, atol=1e-12, label="k = 0.5, b = 2, beta = 0.5, p = 0.7")
    got = critical_weight(1.0, 0.0, 1.0, 0.5)
    if not np.isinf(got):
        fail(f"with b = 0 the thumbs reward never favors agreeing, so w* is inf; got {got}")


@check("at w* the optimal policy agrees with a wrong user exactly half the time, and flips on either side")
def test_critical_weight_flip(sub):
    critical_weight = has(sub, "critical_weight")
    optimal_policy = has(sub, "optimal_policy")
    mixed_rewards = has(sub, "mixed_rewards")
    for k, b, beta, p0 in [(1.0, 3.0, 1.0, 0.5), (0.5, 2.0, 0.5, 0.7), (1.0, 2.0, 0.8, 0.3)]:
        p_ref = np.array([[p0, 1 - p0], [p0, 1 - p0]])
        w_star = critical_weight(k, b, beta, p0)
        if not 0.0 < w_star < 1.0:
            fail(f"expected w* inside (0, 1) for k={k}, b={b}, beta={beta}, p={p0}; got {w_star}")
        at = optimal_policy(p_ref, mixed_rewards(w_star, k, b), beta)[1, 0]
        below = optimal_policy(p_ref, mixed_rewards(w_star - 0.05, k, b), beta)[1, 0]
        above = optimal_policy(p_ref, mixed_rewards(w_star + 0.05, k, b), beta)[1, 0]
        close(at, 0.5, atol=1e-9, label=f"P(agree | wrong) at w* for k={k}, b={b}")
        if not (below < 0.5 < above):
            fail(f"P(agree | wrong) should cross 0.5 at w*; got {below:.3f} below and {above:.3f} above")


@check("slice_eval: the slices are policy[0, 0] and policy[1, 1], and the aggregate weights them by rho")
def test_slice_eval(sub):
    slice_eval = has(sub, "slice_eval")
    policy = np.array([[0.9, 0.1], [0.6, 0.4]])
    right, wrong, aggregate = slice_eval(policy, 0.8)
    close([right, wrong, aggregate], [0.9, 0.4, 0.8], atol=1e-12,
          label="accuracy when right, when wrong, aggregate")


@check("as w grows past w*, the aggregate rises while the wrong-user slice falls below 40%")
def test_aggregate_hides_slice(sub):
    optimal_policy = has(sub, "optimal_policy")
    mixed_rewards = has(sub, "mixed_rewards")
    slice_eval = has(sub, "slice_eval")
    results = {}
    for w in (0.0, 0.5):
        pi = optimal_policy(UNIFORM, mixed_rewards(w, K, B), BETA)
        results[w] = [float(v) for v in slice_eval(pi, RHO)]
    # w = 0: 0.7311 on both slices. w = 0.5: 0.9241 right, 0.3775 wrong, 0.8148 overall.
    close(results[0.0], [0.731059, 0.731059, 0.731059], atol=1e-5, label="w = 0")
    close(results[0.5], [0.924142, 0.377541, 0.814821], atol=1e-5, label="w = 0.5")
    if not results[0.5][2] > results[0.0][2]:
        fail("the aggregate should rise from w = 0 to w = 0.5")
    if not results[0.5][1] < 0.4:
        fail(f"the wrong-user slice should fall below 0.4 at w = 0.5, got {results[0.5][1]:.3f}")


@check("best_of_n_policy matches brute-force enumeration, and the agree rate on a wrong user is 1 - (1 - p)^n")
def test_best_of_n(sub):
    best_of_n_policy = has(sub, "best_of_n_policy")
    p = np.array([0.2, 0.5, 0.3])
    reward = np.array([1.0, 3.0, 2.0])
    for n in (1, 2, 3):
        want = np.zeros(3)
        for draws in itertools.product(range(3), repeat=n):
            prob = np.prod(p[list(draws)])
            want[max(draws, key=lambda i: reward[i])] += prob
        close(best_of_n_policy(p, reward, n), want, atol=1e-12, label=f"n = {n}")
    # Wrong user: agree earns b = 3, correct earns k = 1, and the reference agrees half the time.
    for n in (1, 2, 4, 8):
        got = best_of_n_policy(np.array([0.5, 0.5]), np.array([3.0, 1.0]), n)[0]
        close(got, 1.0 - 0.5**n, atol=1e-12, label=f"agree rate at n = {n}")
