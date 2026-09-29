"""Multi-armed bandits: incremental means, epsilon-greedy, UCB1, and Thompson sampling.

Fill in the functions in order. Each one is a few lines of numpy. The last one,
`run_bandit`, plays a bandit for T pulls with one of the three strategies and
returns the cumulative regret after every pull.

Arms are numbered 0 to k - 1. A bandit from `rlenvs` has `.k` arms,
`.pull(arm)` returns a reward, and `.regret(arm)` returns the gap between the
best arm's mean and this arm's mean.
"""

import numpy as np

import rlenvs


def update_estimate(q, n, reward, alpha=None):
    """Move an arm's value estimate toward one new reward.

    Args:
        q: the current estimate, a float.
        n: how many rewards are already averaged into `q`, an int >= 0.
        reward: the new reward, a float.
        alpha: the step size. With `alpha=None`, use 1 / (n + 1), so the result
            is the exact mean of all n + 1 rewards. With a float, use that
            constant step size.

    Returns:
        The new estimate, a float.
    """
    raise NotImplementedError("update_estimate")


def epsilon_greedy_action(Q, epsilon, rng):
    """Pick an arm the epsilon-greedy way.

    With probability `epsilon`, pick an arm uniformly at random. Otherwise pick
    an arm with the largest estimate, breaking ties uniformly at random.

    Args:
        Q: value estimates, shape (k,).
        epsilon: the exploration probability, in [0, 1].
        rng: a numpy Generator. Draw every random number from it.

    Returns:
        The chosen arm, an int.
    """
    raise NotImplementedError("epsilon_greedy_action")


def ucb1_scores(means, counts, t):
    """UCB1's score for every arm.

    The score is the arm's mean reward plus sqrt(2 ln t / N), where N is the
    arm's pull count and t is the total number of pulls so far.

    Args:
        means: each arm's average reward so far, shape (k,).
        counts: each arm's pull count, shape (k,). May hold zeros.
        t: the total number of pulls so far, an int >= 1.

    Returns:
        Scores, shape (k,). An arm with count 0 scores +inf.
    """
    raise NotImplementedError("ucb1_scores")


def thompson_action(successes, failures, rng):
    """Pick an arm by Thompson sampling for Bernoulli arms with a Beta(1, 1) prior.

    Arm a's posterior is Beta(1 + successes[a], 1 + failures[a]). Draw one
    sample from each arm's posterior and pick the arm with the largest sample.

    Args:
        successes: each arm's count of reward-1 pulls, shape (k,).
        failures: each arm's count of reward-0 pulls, shape (k,).
        rng: a numpy Generator.

    Returns:
        The chosen arm, an int.
    """
    raise NotImplementedError("thompson_action")


def run_bandit(bandit, method, T, seed, epsilon=0.1):
    """Play `bandit` for T pulls and record the cumulative regret.

    Args:
        bandit: an `rlenvs.BernoulliBandit`.
        method: "epsilon-greedy", "ucb1", or "thompson".
        T: the number of pulls.
        seed: seeds the Generator that makes the strategy's random choices.
        epsilon: the exploration probability for "epsilon-greedy".

    For "ucb1", pick the arm with the largest `ucb1_scores(Q, N, t)`, where t
    is the number of pulls made so far (use 1 on the very first pull), and
    break ties at random. Keep every arm's estimate with `update_estimate`.

    Returns:
        Shape (T,). Entry t is the sum of `bandit.regret(arm)` over the first
        t + 1 pulls.
    """
    raise NotImplementedError("run_bandit")


if __name__ == "__main__":
    means = [0.1, 0.5, 0.9]
    T = 8000
    print(f"Bernoulli arms {means}, mean regret over 3 seeds")
    for method in ["epsilon-greedy", "ucb1", "thompson"]:
        curves = [run_bandit(rlenvs.BernoulliBandit(means, seed=s), method, T, seed=s) for s in range(3)]
        mean = np.mean(curves, axis=0)
        marks = "  ".join(f"T={t}: {mean[t - 1]:6.1f}" for t in (500, 2000, 8000))
        print(f"{method:>15}  {marks}   growth from 2000 to 8000: {mean[-1] / mean[1999]:.2f}x")
