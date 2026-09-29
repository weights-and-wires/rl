"""Checks for the gradient and contextual bandits exercise."""

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


@check("sherman_morrison keeps A^-1 exact through twenty rank-one updates")
def test_sherman_morrison(sub):
    sherman_morrison = has(sub, "sherman_morrison")
    rng = np.random.default_rng(3)
    A, A_inv = np.eye(4), np.eye(4)
    for _ in range(20):
        x = rng.normal(size=4)
        A = A + np.outer(x, x)
        A_inv = sherman_morrison(A_inv, x)
        shape(A_inv, (4, 4), label="updated inverse")
    close(A_inv, np.linalg.inv(A), atol=1e-10, label="after 20 updates, against np.linalg.inv(I + sum x x^T)")


@check("linucb_terms gives the ridge prediction and the confidence width in closed form")
def test_linucb_terms(sub):
    linucb_terms = has(sub, "linucb_terms")
    A_inv = np.linalg.inv(np.array([[3.0, 1.0], [1.0, 2.0]]))
    mean, width = linucb_terms(A_inv, np.array([1.0, 0.0]), np.array([1.0, 0.5]))
    close(mean, 0.3, atol=1e-12, label="the lesson's mean, theta_hat = (0.4, -0.2)")
    close(width, np.sqrt(0.35), atol=1e-12, label="the lesson's width, sqrt(0.35)")

    rng = np.random.default_rng(4)
    X, y, x = rng.normal(size=(30, 3)), rng.normal(size=30), rng.normal(size=3)
    mean, width = linucb_terms(np.linalg.inv(np.eye(3) + X.T @ X), X.T @ y, x)
    close(mean, x @ np.linalg.solve(X.T @ X + np.eye(3), X.T @ y), atol=1e-10, label="ridge prediction")

    # Seeing the same x n times gives A = I + n x x^T, so x^T A^-1 x = |x|^2 / (1 + n |x|^2).
    x = np.array([0.6, -0.8, 0.0])
    for n in (1, 10, 100):
        _, width = linucb_terms(np.linalg.inv(np.eye(3) + n * np.outer(x, x)), np.zeros(3), x)
        close(width, np.sqrt(1.0 / (1.0 + n)), atol=1e-10, label=f"width after seeing a unit x {n} times")


@check("run_linucb's regret stays far below the best context-blind arm's, and flattens out")
def test_run_linucb(sub):
    run_linucb = has(sub, "run_linucb")
    LinearContextualBandit = has(sub, "LinearContextualBandit")
    theta = np.random.default_rng(1).normal(size=(3, 4))
    curve = np.asarray(run_linucb(LinearContextualBandit(theta, seed=0), 2000, 1.0), dtype=float)
    shape(curve, (2000,), label="run_linucb output")
    if np.any(np.diff(curve) < -1e-12):
        fail("the cumulative regret must never decrease")
    # The best single arm, ignoring context, loses about 0.61 per round here.
    X = np.concatenate([np.ones((100000, 1)), np.random.default_rng(99).standard_normal((100000, 3))], axis=1)
    M = X @ theta.T
    blind = 2000 * float(np.mean(M.max(axis=1) - M[:, np.argmax(M.mean(axis=0))]))
    print(f"LinUCB regret {curve[-1]:.1f}; best context-blind arm about {blind:.0f}")
    if curve[-1] > 0.1 * blind:
        fail(f"LinUCB's regret after 2000 rounds is {curve[-1]:.1f}, above a tenth of the best fixed arm's {blind:.0f}")
    first, second = curve[999], curve[-1] - curve[999]
    if second > 0.75 * first:
        fail(f"LinUCB should learn: regret in rounds 1000-2000 ({second:.1f}) should be well below rounds 1-1000 ({first:.1f})")
