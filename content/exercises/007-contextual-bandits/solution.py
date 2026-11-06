"""Reference solution for the LinUCB exercise."""

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
    theta_hat = A_inv @ b
    return float(theta_hat @ x), float(np.sqrt(x @ A_inv @ x))


def sherman_morrison(A_inv, x):
    A_inv = np.asarray(A_inv, dtype=float)
    x = np.asarray(x, dtype=float)
    Ax = A_inv @ x
    return A_inv - np.outer(Ax, Ax) / (1.0 + x @ Ax)


def run_linucb(bandit, T, alpha):
    k, d = bandit.k, bandit.d
    A_inv = np.stack([np.eye(d)] * k)
    b = np.zeros((k, d))
    regret = np.zeros(T)
    total = 0.0
    for t in range(T):
        x = bandit.observe()
        scores = []
        for a in range(k):
            mean, width = linucb_terms(A_inv[a], b[a], x)
            scores.append(mean + alpha * width)
        arm = int(np.argmax(scores))
        reward = bandit.pull(arm)
        A_inv[arm] = sherman_morrison(A_inv[arm], x)
        b[arm] += reward * x
        total += bandit.regret(arm)
        regret[t] = total
    return regret


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
