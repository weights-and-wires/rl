"""Reference solution for the gradient and contextual bandits exercise."""

import numpy as np

import rlenvs


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


def softmax(H):
    H = np.asarray(H, dtype=float)
    z = np.exp(H - H.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def gradient_bandit_update(H, arm, reward, baseline, alpha):
    H = np.asarray(H, dtype=float)
    onehot = np.zeros_like(H)
    onehot[arm] = 1.0
    return H + alpha * (reward - baseline) * (onehot - softmax(H))


def expected_update(H, q, baseline):
    H = np.asarray(H, dtype=float)
    pi = softmax(H)
    step = np.zeros_like(H)
    for x in range(len(H)):
        step += pi[x] * (gradient_bandit_update(H, x, q[x], baseline, alpha=1.0) - H)
    return step


def run_gradient_bandit(bandit, T, alpha, use_baseline, seed):
    rng = np.random.default_rng(seed)
    H = np.zeros(bandit.k)
    mean_reward = 0.0
    arms = np.zeros(T, dtype=int)
    for t in range(T):
        arm = int(rng.choice(bandit.k, p=softmax(H)))
        reward = bandit.pull(arm)
        baseline = mean_reward if use_baseline else 0.0
        H = gradient_bandit_update(H, arm, reward, baseline, alpha)
        mean_reward += (reward - mean_reward) / (t + 1)
        arms[t] = arm
    return arms


def sherman_morrison(A_inv, x):
    A_inv = np.asarray(A_inv, dtype=float)
    x = np.asarray(x, dtype=float)
    Ax = A_inv @ x
    return A_inv - np.outer(Ax, Ax) / (1.0 + x @ Ax)


def linucb_terms(A_inv, b, x):
    theta_hat = A_inv @ b
    return float(theta_hat @ x), float(np.sqrt(x @ A_inv @ x))


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
    means = np.random.default_rng(0).normal(4.0, 1.0, size=10)
    best = int(np.argmax(means))
    print("Gradient bandit on 10 arms with means near +4, fraction of optimal pulls in steps 800-1000:")
    for use_baseline in (True, False):
        share = np.mean([
            np.mean(run_gradient_bandit(rlenvs.GaussianBandit(means, seed=s), 1000, 0.1, use_baseline, seed=s)[800:] == best)
            for s in range(10)
        ])
        print(f"  baseline={use_baseline!s:5}  {share:.2f}")

    theta = np.random.default_rng(1).normal(size=(3, 4))
    curve = run_linucb(LinearContextualBandit(theta, seed=0), 2000, alpha=1.0)
    print("LinUCB cumulative regret at t = 250, 500, 1000, 2000:", np.round(curve[[249, 499, 999, 1999]], 1))
