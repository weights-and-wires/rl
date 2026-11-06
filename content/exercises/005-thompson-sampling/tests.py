"""Checks for the Thompson sampling exercise."""

import numpy as np

import rlenvs
from rlcheck import check, close, fail, has, shape

MEANS = [0.1, 0.5, 0.9]


def _mean_curve(run_bandit, method, T, seeds, **kwargs):
    curves = []
    for s in seeds:
        curve, _ = run_bandit(rlenvs.BernoulliBandit(MEANS, seed=s), method, T, seed=s, **kwargs)
        curve = np.asarray(curve, dtype=float)
        shape(curve, (T,), label=f"run_bandit(..., {method!r}, T={T})")
        curves.append(curve)
    return np.mean(curves, axis=0)


@check("beta_posterior adds one to each count: Beta(1 + S, 1 + F)")
def test_beta_posterior(sub):
    beta_posterior = has(sub, "beta_posterior")
    a, b = beta_posterior(np.array([3, 1, 0]), np.array([1, 1, 0]))
    shape(np.asarray(a), (3,), label="first Beta parameter")
    close(a, [4.0, 2.0, 1.0], atol=1e-12, label="first parameters 1 + successes")
    close(b, [2.0, 2.0, 1.0], atol=1e-12, label="second parameters 1 + failures")
    close(np.asarray(a) / (np.asarray(a) + np.asarray(b)), [2 / 3, 0.5, 0.5], atol=1e-12, label="posterior means a / (a + b)")


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


@check("thompson_action stops exploring an arm that the evidence has ruled out")
def test_thompson_confident(sub):
    thompson_action = has(sub, "thompson_action")
    rng = np.random.default_rng(4)
    picks = np.array([thompson_action(np.array([80, 5]), np.array([20, 95]), rng) for _ in range(2000)])
    if np.mean(picks == 0) < 0.99:
        fail(
            "arm 0 paid 80 of 100 pulls and arm 1 paid 5 of 100, so Thompson sampling should almost "
            f"never pick arm 1; it picked arm 0 only {np.mean(picks == 0):.3f} of the time"
        )
    picks = np.array([thompson_action(np.array([0, 0, 0]), np.array([0, 0, 0]), rng) for _ in range(3000)])
    freq = np.bincount(picks, minlength=3) / 3000
    if np.any(np.abs(freq - 1 / 3) > 4 * np.sqrt(2 / 9 / 3000)):
        fail(f"with no data, all three arms are equally plausible, so each should be picked a third of the time; got {np.round(freq, 3)}")


@check("prob_best returns probabilities that add to 1 and give 5/7 to the Beta(4, 2) arm")
def test_prob_best(sub):
    prob_best = has(sub, "prob_best")
    rng = np.random.default_rng(5)
    got = np.asarray(prob_best(np.array([3, 1]), np.array([1, 1]), 40000, rng), dtype=float)
    shape(got, (2,), label="prob_best output")
    close(got.sum(), 1.0, atol=1e-12, label="the probabilities add to 1")
    se = np.sqrt((5 / 7) * (2 / 7) / 40000)
    if abs(got[0] - 5 / 7) > 4 * se:
        fail(f"the chance that Beta(4, 2) beats Beta(2, 2) is 5/7 = 0.7143, but prob_best gave {got[0]:.4f}")


@check("Thompson sampling plays each arm with the probability that it is the best arm")
def test_matches_prob_best(sub):
    thompson_action = has(sub, "thompson_action")
    prob_best = has(sub, "prob_best")
    successes, failures = np.array([6, 3, 1]), np.array([4, 4, 1])
    rng = np.random.default_rng(6)
    n = 20000
    picks = np.array([thompson_action(successes, failures, rng) for _ in range(n)])
    freq = np.bincount(picks, minlength=3) / n
    want = np.asarray(prob_best(successes, failures, 200000, np.random.default_rng(7)), dtype=float)
    if np.any(np.abs(freq - want) > 4 * np.sqrt(want * (1 - want) / n) + 0.004):
        fail(f"the frequencies of Thompson's picks {np.round(freq, 3)} should match each arm's chance of being best {np.round(want, 3)}")


@check("Thompson sampling has far less regret than epsilon-greedy and UCB1 at T = 2000")
def test_regret_comparison(sub):
    run_bandit = has(sub, "run_bandit")
    seeds = range(4)
    eps = _mean_curve(run_bandit, "epsilon-greedy", 2000, seeds, epsilon=0.1)[-1]
    ucb = _mean_curve(run_bandit, "ucb1", 2000, seeds)[-1]
    ts = _mean_curve(run_bandit, "thompson", 2000, seeds)[-1]
    print(f"mean regret at T=2000: epsilon-greedy {eps:.1f}, UCB1 {ucb:.1f}, Thompson {ts:.1f}")
    if ts > 0.3 * eps:
        fail(f"Thompson sampling's mean regret {ts:.1f} should be far under epsilon-greedy's {eps:.1f} (below 30 percent of it)")
    if ts > ucb:
        fail(f"Thompson sampling's mean regret {ts:.1f} should be below UCB1's {ucb:.1f} on this bandit")


@check("Thompson regret grows only logarithmically from T=2000 to T=8000")
def test_growth(sub):
    run_bandit = has(sub, "run_bandit")
    curve = _mean_curve(run_bandit, "thompson", 8000, range(4))
    ratio = curve[-1] / max(curve[1999], 1e-12)
    print(f"regret(8000) / regret(2000) = {ratio:.2f} (linear growth gives 4, ln 8000 / ln 2000 is 1.18)")
    if ratio > 2.2:
        fail(f"Thompson's regret grew {ratio:.2f} times from T=2000 to T=8000. Logarithmic growth gives well under 2.2")
