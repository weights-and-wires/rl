"""Gradient bandits.

Fill in the functions in order. You build a softmax policy over arm
preferences, its update after one pull, the update's expected value, and a
learning run on a bandit whose rewards sit far from zero.
"""

import numpy as np

import rlenvs


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
        The updated preferences, shape (k,). Don't modify H in place.
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


if __name__ == "__main__":
    H = gradient_bandit_update(np.zeros(3), arm=0, reward=1.0, baseline=0.0, alpha=0.1)
    print("One update after arm 0 pays 1:", np.round(H, 4), "-> policy", np.round(softmax(H), 4))
    print("Expected update at H = 0 with q = (1, 2, 3):", np.round(expected_update(np.zeros(3), np.array([1.0, 2.0, 3.0]), 0.0), 4))

    means = np.random.default_rng(0).normal(4.0, 1.0, size=10)
    best = int(np.argmax(means))
    print("Gradient bandit on 10 arms with means near +4, fraction of best-arm pulls in steps 800-1000:")
    for use_baseline in (True, False):
        share = np.mean([
            np.mean(run_gradient_bandit(rlenvs.GaussianBandit(means, seed=s), 1000, 0.1, use_baseline, seed=s)[800:] == best)
            for s in range(10)
        ])
        print(f"  baseline={use_baseline!s:5}  {share:.2f}")
