"""LinUCB for contextual bandits.

Fill in the functions in order. You score one arm's prediction and confidence
width, update its inverse matrix after one new data point, and then run LinUCB
on a bandit whose best arm depends on a context vector.

`LinearContextualBandit` is the environment. It is complete; don't change it.
"""

import numpy as np


class LinearContextualBandit:
    """A contextual bandit whose arm a pays theta[a] @ x plus Gaussian noise.

    Each round, `observe()` draws a fresh context x of shape (d,): a leading 1
    for an intercept, then d - 1 standard-normal features. `pull(arm)` pays
    theta[arm] @ x + noise * N(0, 1) for the context last observed, and
    `regret(arm)` is max_a theta[a] @ x - theta[arm] @ x for that context.
    """

    def __init__(self, theta, noise=0.5, seed=None):
        self.theta = np.asarray(theta, dtype=float)
        self.k, self.d = self.theta.shape
        self.noise = float(noise)
        self.rng = np.random.default_rng(seed)
        self.x = None

    def observe(self):
        self.x = np.concatenate([[1.0], self.rng.standard_normal(self.d - 1)])
        return self.x.copy()

    def pull(self, arm):
        return float(self.theta[arm] @ self.x + self.noise * self.rng.standard_normal())

    def regret(self, arm):
        means = self.theta @ self.x
        return float(means.max() - means[arm])


def linucb_terms(A_inv, b, x):
    """LinUCB's prediction and confidence width for one arm.

    Args:
        A_inv: the inverse of A = I + sum of x x^T over this arm's past
            contexts, shape (d, d).
        b: the sum of reward * x over this arm's past rounds, shape (d,).
        x: the current context, shape (d,).

    Returns:
        A tuple `(mean, width)` of floats. `mean` is x @ theta_hat with
        theta_hat = A_inv @ b, the ridge regression estimate. `width` is
        sqrt(x @ A_inv @ x).
    """
    raise NotImplementedError("linucb_terms")


def sherman_morrison(A_inv, x):
    """The inverse of A + x x^T, given the inverse of A.

    Args:
        A_inv: the inverse of a symmetric positive-definite matrix A, shape (d, d).
        x: a vector, shape (d,).

    Returns:
        The inverse of A + outer(x, x), shape (d, d), in O(d^2) work.
    """
    raise NotImplementedError("sherman_morrison")


def run_linucb(bandit, T, alpha):
    """Run disjoint LinUCB on a `LinearContextualBandit` for T rounds.

    Keep one A_inv (starting at the identity) and one b (starting at zero) per
    arm. Each round, call `bandit.observe()` for the context x, score each arm
    as mean + alpha * width, pull the arm with the largest score, and update
    only that arm's A_inv and b.

    Args:
        bandit: a `LinearContextualBandit`.
        T: the number of rounds.
        alpha: the width multiplier, a float.

    Returns:
        Shape (T,). Entry t is the sum of `bandit.regret(arm)` over rounds 0 to t.
    """
    raise NotImplementedError("run_linucb")


if __name__ == "__main__":
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    A_inv = np.linalg.inv(A)
    mean, width = linucb_terms(A_inv, np.array([1.0, 0.0]), np.array([1.0, 0.5]))
    print(f"Worked example: prediction {mean:.4f}, width {width:.4f}")
    print("After seeing that context once, the inverse is")
    print(np.round(sherman_morrison(A_inv, np.array([1.0, 0.5])), 4))

    theta = np.random.default_rng(1).normal(size=(3, 4))
    curve = run_linucb(LinearContextualBandit(theta, seed=0), 2000, alpha=1.0)
    print("LinUCB cumulative regret at t = 250, 500, 1000, 2000:", np.round(curve[[249, 499, 999, 1999]], 1))
