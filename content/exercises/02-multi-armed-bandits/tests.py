"""Checks for the multi-armed bandits exercise."""

import numpy as np

import rlenvs
from rlcheck import check, close, equal, fail, has, shape

MEANS = [0.1, 0.5, 0.9]


def _mean_curve(run_bandit, method, T, seeds, **kwargs):
    curves = []
    for s in seeds:
        curve = np.asarray(run_bandit(rlenvs.BernoulliBandit(MEANS, seed=s), method, T, seed=s, **kwargs), dtype=float)
        shape(curve, (T,), label=f"run_bandit(..., {method!r}, T={T})")
        curves.append(curve)
    return np.mean(curves, axis=0)


@check("update_estimate with alpha=None keeps the exact running mean")
def test_incremental_mean(sub):
    update_estimate = has(sub, "update_estimate")
    q, got = 0.0, []
    for n, r in enumerate([1.0, 0.0, 1.0, 1.0]):
        q = update_estimate(q, n, r)
        got.append(q)
    close(got, [1.0, 0.5, 2.0 / 3.0, 0.75], atol=1e-12, label="estimates after rewards 1, 0, 1, 1")

    rewards = np.random.default_rng(0).normal(2.0, 3.0, size=200)
    q, got = 0.0, []
    for n, r in enumerate(rewards):
        q = update_estimate(q, n, r)
        got.append(q)
    want = np.cumsum(rewards) / np.arange(1, 201)
    close(got, want, atol=1e-10, label="running estimate against np.cumsum(rewards) / n")


@check("update_estimate with a constant alpha is the exponentially weighted average")
def test_constant_step(sub):
    update_estimate = has(sub, "update_estimate")
    q, got = 0.0, []
    for n, r in enumerate([1.0, 0.0, 1.0, 1.0]):
        q = update_estimate(q, n, r, alpha=0.5)
        got.append(q)
    close(got, [0.5, 0.25, 0.625, 0.8125], atol=1e-12, label="alpha=0.5 estimates after rewards 1, 0, 1, 1")

    rng = np.random.default_rng(1)
    rewards = rng.normal(size=30)
    q1, alpha, q = 2.0, 0.2, 2.0
    for n, r in enumerate(rewards):
        q = update_estimate(q, n, r, alpha=alpha)
    i = np.arange(1, 31)
    want = (1 - alpha) ** 30 * q1 + np.sum(alpha * (1 - alpha) ** (30 - i) * rewards)
    close(q, want, atol=1e-12, label="(1 - alpha)^n Q_1 + sum_i alpha (1 - alpha)^(n - i) r_i")


@check("epsilon_greedy_action explores with probability epsilon and splits ties evenly")
def test_epsilon_greedy(sub):
    epsilon_greedy_action = has(sub, "epsilon_greedy_action")
    rng = np.random.default_rng(2)
    n = 30000
    picks = np.array([epsilon_greedy_action(np.array([0.2, 0.7, 0.1]), 0.1, rng) for _ in range(n)])
    freq = np.bincount(picks, minlength=3) / n
    want = np.array([0.1 / 3, 0.9 + 0.1 / 3, 0.1 / 3])
    se = np.sqrt(want * (1 - want) / n)
    if np.any(np.abs(freq - want) > 4 * se):
        fail(f"with epsilon=0.1 and a clear best arm, expected frequencies {np.round(want, 4)}, got {np.round(freq, 4)}")

    picks = np.array([epsilon_greedy_action(np.array([0.5, 0.5, 0.1]), 0.0, rng) for _ in range(4000)])
    share = np.mean(picks == 0)
    if np.any(picks == 2) or abs(share - 0.5) > 4 * np.sqrt(0.25 / 4000):
        fail(
            "with epsilon=0 and arms 0 and 1 tied for best, expected each about half the "
            f"time and arm 2 never; got arm 0 {share:.3f} of the time, arm 2 {np.mean(picks == 2):.3f}"
        )


@check("ucb1_scores matches the lesson's table and gives unpulled arms infinity")
def test_ucb1_scores(sub):
    ucb1_scores = has(sub, "ucb1_scores")
    got = ucb1_scores(np.array([0.6, 0.4]), np.array([5, 2]), 7)
    close(got, [0.6 + np.sqrt(2 * np.log(7) / 5), 0.4 + np.sqrt(2 * np.log(7) / 2)], atol=1e-12,
          label="scores for means (0.6, 0.4), counts (5, 2), t = 7")
    close(got, [1.482249, 1.794959], atol=1e-6, label="the lesson's table")
    got = np.asarray(ucb1_scores(np.array([0.3, 0.0, 0.8]), np.array([4, 0, 3]), 7), dtype=float)
    if not np.isposinf(got[1]):
        fail(f"an arm with count 0 must score +inf, got {got[1]}")


@check("thompson_action plays Beta(4, 2) over Beta(2, 2) with probability 5/7")
def test_thompson_action(sub):
    thompson_action = has(sub, "thompson_action")
    rng = np.random.default_rng(3)
    n = 20000
    picks = np.array([thompson_action(np.array([3, 1]), np.array([1, 1]), rng) for _ in range(n)])
    share = np.mean(picks == 0)
    se = np.sqrt((5 / 7) * (2 / 7) / n)
    if abs(share - 5 / 7) > 4 * se:
        fail(
            "arm 0 has 3 successes and 1 failure, arm 1 has 1 of each. With a uniform prior, "
            f"Thompson sampling should play arm 0 with probability 5/7 = 0.7143; it did {share:.4f} of the time"
        )


@check("run_bandit's regret curve for uniform play (epsilon=1) grows at the mean gap")
def test_uniform_regret(sub):
    run_bandit = has(sub, "run_bandit")
    curve = np.asarray(run_bandit(rlenvs.BernoulliBandit(MEANS, seed=0), "epsilon-greedy", 2000, seed=0, epsilon=1.0), dtype=float)
    shape(curve, (2000,), label="run_bandit output")
    if np.any(np.diff(curve) < -1e-12) or curve[0] < 0:
        fail("the cumulative regret must never decrease: add bandit.regret(arm), never a negative number")
    # Each pull costs gap 0.8, 0.4, or 0 with probability 1/3: mean 0.4, variance 0.8/3 - 0.16.
    want, sd = 2000 * 0.4, np.sqrt(2000 * (0.8 / 3 - 0.16))
    if abs(curve[-1] - want) > 4 * sd:
        fail(f"uniform play over 2000 pulls should cost about {want:.0f} +- {4 * sd:.0f} regret, got {curve[-1]:.1f}")


@check("UCB1 and Thompson sampling have far less regret than epsilon-greedy at T = 2000")
def test_regret_comparison(sub):
    run_bandit = has(sub, "run_bandit")
    seeds = range(4)
    eps = _mean_curve(run_bandit, "epsilon-greedy", 2000, seeds, epsilon=0.1)[-1]
    ucb = _mean_curve(run_bandit, "ucb1", 2000, seeds)[-1]
    ts = _mean_curve(run_bandit, "thompson", 2000, seeds)[-1]
    print(f"mean regret at T=2000: epsilon-greedy {eps:.1f}, UCB1 {ucb:.1f}, Thompson {ts:.1f}")
    if ucb > 0.75 * eps:
        fail(f"UCB1's mean regret {ucb:.1f} should be well under epsilon-greedy's {eps:.1f} (below 75 percent of it)")
    if ts > 0.3 * eps:
        fail(f"Thompson sampling's mean regret {ts:.1f} should be far under epsilon-greedy's {eps:.1f} (below 30 percent of it)")


@check("UCB1 and Thompson regret grow sublinearly from T to 4T, epsilon-greedy's almost linearly")
def test_growth(sub):
    run_bandit = has(sub, "run_bandit")
    seeds = range(4)
    report = []
    for method, kwargs in [("ucb1", {}), ("thompson", {}), ("epsilon-greedy", {"epsilon": 0.1})]:
        curve = _mean_curve(run_bandit, method, 8000, seeds, **kwargs)
        ratio = curve[-1] / max(curve[1999], 1e-12)
        report.append(f"{method} {ratio:.2f}")
        if method != "epsilon-greedy" and ratio > 2.2:
            fail(f"{method}'s regret grew {ratio:.2f} times from T=2000 to T=8000. Logarithmic growth gives well under 2.2")
        if method == "epsilon-greedy" and ratio < 2.8:
            fail(f"epsilon-greedy's regret grew only {ratio:.2f} times from T=2000 to T=8000. A fixed epsilon should grow close to 4 times")
    print("regret(8000) / regret(2000): " + ", ".join(report))
