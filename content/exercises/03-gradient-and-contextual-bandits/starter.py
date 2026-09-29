"""Gradient and contextual bandits.

Fill in the functions in order. The first four build the gradient bandit: a
softmax policy over arm preferences, its update, the update's expected value,
and a learning run. The last three build LinUCB for a bandit whose best arm
depends on a context vector.

`LinearContextualBandit` is the environment for the second half. It is
complete; don't change it.
"""

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
    """Turn preferences into probabilities.

    Args:
        H: preferences, shape (k,).

    Returns:
        exp(H) / sum(exp(H)), shape (k,). Subtract H.max() before np.exp so that
        large preferences don't overflow.
    """
    raise NotImplementedError("softmax")


def gradient_bandit_update(H, arm, reward, baseline, alpha):
    """One gradient bandit step after pulling `arm` and receiving `reward`.

    Every preference moves by alpha * (reward - baseline) * (1{a = arm} - pi(a)),
    where pi = softmax(H) is the policy before the update.

    Args:
        H: preferences, shape (k,).
        arm: the arm that was pulled, an int.
        reward: the reward it paid, a float.
        baseline: the reference level the reward is compared against, a float.
        alpha: the step size, a float.

    Returns:
        The new preferences, shape (k,). Don't modify H in place.
    """
    raise NotImplementedError("gradient_bandit_update")


def expected_update(H, q, baseline):
    """The average of the step-size-1 update over the policy's own arm choices.

    Arm x is pulled with probability pi(x) = softmax(H)[x] and pays q[x] on
    average. Weight each arm's update (with alpha = 1) by pi(x) and add them up.

    Args:
        H: preferences, shape (k,).
        q: each arm's mean reward, shape (k,).
        baseline: the baseline, a float.

    Returns:
        The expected change in H per unit of alpha, shape (k,).
    """
    raise NotImplementedError("expected_update")


def run_gradient_bandit(bandit, T, alpha, use_baseline, seed):
    """Run the gradient bandit for T steps from H = 0.

    Each step, draw an arm from softmax(H) with a Generator made from `seed`,
    pull it, and update H. With `use_baseline`, the baseline is the mean of all
    rewards received before this step (0 before the first). Without it, the
    baseline is 0.

    Args:
        bandit: an `rlenvs.GaussianBandit` or `rlenvs.BernoulliBandit`.
        T: the number of steps.
        alpha: the step size.
        use_baseline: whether to compare rewards against their running mean.
        seed: an int.

    Returns:
        The arms pulled, an int array of shape (T,).
    """
    raise NotImplementedError("run_gradient_bandit")


def sherman_morrison(A_inv, x):
    """The inverse of A + x x^T, given the inverse of A.

    Args:
        A_inv: the inverse of a symmetric positive-definite matrix A, shape (d, d).
        x: a vector, shape (d,).

    Returns:
        The inverse of A + outer(x, x), shape (d, d), in O(d^2) work.
    """
    raise NotImplementedError("sherman_morrison")


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
