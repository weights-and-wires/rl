"""Checks for the gradient bandit exercise."""

import numpy as np

import rlenvs
from rlcheck import check, close, fail, finite_difference, has, shape


def _softmax(H):
    z = np.exp(H - H.max())
    return z / z.sum()


@check("softmax gives probabilities, ignores a shared shift, and survives huge preferences")
def test_softmax(sub):
    softmax = has(sub, "softmax")
    close(softmax(np.array([0.0, np.log(3.0)])), [0.25, 0.75], atol=1e-12, label="softmax of (0, log 3)")
    H = np.random.default_rng(0).normal(size=6)
    p = softmax(H)
    shape(p, (6,), label="softmax output")
    close(np.sum(p), 1.0, atol=1e-12, label="sum of probabilities")
    close(softmax(H + 50.0), p, atol=1e-12, label="softmax(H + 50) against softmax(H)")
    close(softmax(np.array([1000.0, 0.0])), [1.0, 0.0], atol=1e-12,
          label="softmax of (1000, 0): subtract the max before np.exp")


@check("gradient_bandit_update matches the lesson's worked example")
def test_gradient_bandit_update(sub):
    update = has(sub, "gradient_bandit_update")
    softmax = has(sub, "softmax")
    H = update(np.zeros(3), 0, 1.0, 0.0, 0.1)
    shape(H, (3,), label="updated preferences")
    close(H, [0.1 * 2 / 3, -0.1 / 3, -0.1 / 3], atol=1e-12, label="preferences after arm 0 pays 1 with baseline 0")
    close(softmax(H), [0.355913, 0.322043, 0.322043], atol=1e-6, label="the new policy")
    H = update(np.array([0.5, -0.2, 0.1]), 2, 0.3, 1.0, 0.5)
    pi = _softmax(np.array([0.5, -0.2, 0.1]))
    close(H, np.array([0.5, -0.2, 0.1]) + 0.5 * (0.3 - 1.0) * (np.eye(3)[2] - pi), atol=1e-12,
          label="a reward below the baseline pushes the pulled arm down")


@check("gradient_bandit_update leaves H untouched and keeps the sum of preferences fixed")
def test_update_is_zero_sum(sub):
    update = has(sub, "gradient_bandit_update")
    H = np.array([0.5, -0.2, 0.1, 1.3])
    before = H.copy()
    new = np.asarray(update(H, 1, 2.5, 0.5, 0.3), dtype=float)
    close(H, before, atol=0.0, label="the input H, which the update must not modify")
    close(new.sum(), before.sum(), atol=1e-12,
          label="sum of preferences after the update: the changes (1{a = arm} - pi(a)) add up to zero")
    if not new[1] > before[1] or not np.all(new[[0, 2, 3]] < before[[0, 2, 3]]):
        fail("a reward above the baseline should raise the pulled arm and lower every other arm")


@check("expected_update equals the exact gradient of expected reward, for any baseline")
def test_expected_update(sub):
    expected_update = has(sub, "expected_update")
    close(expected_update(np.zeros(3), np.array([1.0, 2.0, 3.0]), 0.0), [-1 / 3, 0.0, 1 / 3], atol=1e-12,
          label="uniform policy, means (1, 2, 3): pi(a) * (q(a) - 2)")
    rng = np.random.default_rng(1)
    for trial in range(3):
        H = rng.normal(size=5)
        q = rng.normal(size=5) + 4.0
        exact = finite_difference(lambda h: float(_softmax(h) @ q), H.copy())
        for baseline in (0.0, 4.0, -7.5):
            got = expected_update(H, q, baseline)
            shape(got, (5,), label="expected_update output")
            close(got, exact, atol=1e-7, label=f"expected update with baseline {baseline} against finite differences of J(H)")


@check("run_gradient_bandit with a baseline learns the best arm on means near +4, and without one lags")
def test_baseline_matters(sub):
    run_gradient_bandit = has(sub, "run_gradient_bandit")
    means = np.random.default_rng(2).normal(4.0, 1.0, size=10)
    best = int(np.argmax(means))
    share = {}
    for use_baseline in (True, False):
        runs = []
        for s in range(20):
            arms = np.asarray(run_gradient_bandit(rlenvs.GaussianBandit(means, seed=s), 1000, 0.1, use_baseline, seed=s))
            shape(arms, (1000,), label="arms pulled")
            runs.append(np.mean(arms[500:] == best))
        share[use_baseline] = float(np.mean(runs))
    print(f"share of best-arm pulls in steps 500-1000: with baseline {share[True]:.3f}, without {share[False]:.3f}")
    if share[True] < 0.75:
        fail(f"with a baseline, the best arm should take over 75 percent of late pulls, got {share[True]:.3f}")
    if share[False] > share[True] - 0.2:
        fail(
            "without a baseline, every reward near +4 pushes up whichever arm was pulled, so learning "
            f"should lag well behind: got {share[False]:.3f} without against {share[True]:.3f} with"
        )
