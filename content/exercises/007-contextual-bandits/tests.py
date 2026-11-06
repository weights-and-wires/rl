"""Checks for the LinUCB exercise."""

import numpy as np

from rlcheck import check, close, fail, has, shape


@check("linucb_terms matches the lesson's worked example: prediction 0.3, width sqrt(0.35)")
def test_linucb_worked_example(sub):
    linucb_terms = has(sub, "linucb_terms")
    A_inv = np.linalg.inv(np.array([[3.0, 1.0], [1.0, 2.0]]))
    mean, width = linucb_terms(A_inv, np.array([1.0, 0.0]), np.array([1.0, 0.5]))
    close(mean, 0.3, atol=1e-12, label="the lesson's mean, theta_hat = (0.4, -0.2)")
    close(width, np.sqrt(0.35), atol=1e-12, label="the lesson's width, sqrt(0.35)")


@check("linucb_terms gives a ridge prediction, and a width of sqrt(1 / (1 + n)) after n visits")
def test_linucb_terms(sub):
    linucb_terms = has(sub, "linucb_terms")
    rng = np.random.default_rng(4)
    X, y, x = rng.normal(size=(30, 3)), rng.normal(size=30), rng.normal(size=3)
    mean, width = linucb_terms(np.linalg.inv(np.eye(3) + X.T @ X), X.T @ y, x)
    close(mean, x @ np.linalg.solve(X.T @ X + np.eye(3), X.T @ y), atol=1e-10, label="ridge prediction")

    # Seeing the same x n times gives A = I + n x x^T, so x^T A^-1 x = |x|^2 / (1 + n |x|^2).
    x = np.array([0.6, -0.8, 0.0])
    for n in (1, 10, 100):
        _, width = linucb_terms(np.linalg.inv(np.eye(3) + n * np.outer(x, x)), np.zeros(3), x)
        close(width, np.sqrt(1.0 / (1.0 + n)), atol=1e-10, label=f"width after seeing a unit x {n} times")


@check("sherman_morrison reproduces the lesson's worked update and shrinks that context's width")
def test_sherman_morrison_example(sub):
    sherman_morrison = has(sub, "sherman_morrison")
    A_inv = np.linalg.inv(np.array([[3.0, 1.0], [1.0, 2.0]]))
    x = np.array([1.0, 0.5])
    new = np.asarray(sherman_morrison(A_inv, x), dtype=float)
    shape(new, (2, 2), label="updated inverse")
    close(new, [[1 / 3, -2 / 9], [-2 / 9, 16 / 27]], atol=1e-12, label="the lesson's updated inverse")
    close(np.sqrt(x @ new @ x), np.sqrt(0.35 / 1.35), atol=1e-12, label="the width after seeing x once")
    close(A_inv, np.linalg.inv(np.array([[3.0, 1.0], [1.0, 2.0]])), atol=0.0, label="the input A_inv, which must not change")


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
