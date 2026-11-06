"""Optimism in the face of uncertainty: UCB1 on a row of slot machines.

Fill in the four functions above the line marked "Given in full". The bandit
loop below that line, `run_bandit`, is given in full: you wrote its
epsilon-greedy half in the previous chapter, and its UCB1 half calls your
`ucb1_action`.

Arms are numbered 0 to k - 1. A bandit from `rlenvs` has `.k` arms,
`.pull(arm)` returns a reward, and `.regret(arm)` returns the gap between the
best arm's mean and this arm's mean.
"""

import numpy as np

import rlenvs


def ucb_bonus(counts, t):
    """UCB1's uncertainty bonus for every arm.

    The bonus is sqrt(2 ln t / N), where N is the arm's pull count and t is the
    total number of pulls so far. An arm that was never pulled has an infinite
    bonus, so it gets tried before any arm is tried twice.

    Args:
        counts: each arm's pull count, shape (k,). May hold zeros.
        t: the total number of pulls so far, an int >= 1.

    Returns:
        Shape (k,). Entry a is the bonus of arm a, or `np.inf` where the count is 0.
    """
    raise NotImplementedError("ucb_bonus")


def ucb1_scores(means, counts, t):
    """UCB1's score for every arm: the mean reward plus the bonus.

    Args:
        means: each arm's average reward so far, shape (k,).
        counts: each arm's pull count, shape (k,). May hold zeros.
        t: the total number of pulls so far, an int >= 1.

    Returns:
        Shape (k,). Entry a is means[a] + the bonus of arm a, or `np.inf` for
        an unpulled arm.
    """
    raise NotImplementedError("ucb1_scores")


def ucb1_action(means, counts, t, rng):
    """Pick the arm with the largest UCB1 score, breaking ties at random.

    Args:
        means: each arm's average reward so far, shape (k,).
        counts: each arm's pull count, shape (k,).
        t: the total number of pulls so far, an int >= 1.
        rng: a numpy Generator. Draw the tie-break from it.

    Returns:
        The chosen arm, an int. When several arms share the top score (for
        example, several unpulled arms), each is chosen with equal probability.
    """
    raise NotImplementedError("ucb1_action")


def ucb1_regret_bound(gaps, T):
    """The upper bound on UCB1's expected regret after T pulls (Auer and others, 2002).

    The bound is 8 * sum over arms with gap > 0 of ln(T) / gap, plus
    (1 + pi^2 / 3) * sum over all arms of gap.

    Args:
        gaps: each arm's gap, shape (k,). The best arm has gap 0 and adds
            nothing to the first sum.
        T: the number of pulls.

    Returns:
        The bound, a float.
    """
    raise NotImplementedError("ucb1_regret_bound")


# Given in full: the bandit loop from the previous chapter, extended with UCB1.
# `run_bandit` calls the functions above, so it works once you fill them in.


def update_estimate(q, n, reward, alpha=None):
    if alpha is None:
        alpha = 1.0 / (n + 1)
    return q + alpha * (reward - q)


def epsilon_greedy_action(Q, epsilon, rng):
    Q = np.asarray(Q, dtype=float)
    if rng.random() < epsilon:
        return int(rng.integers(len(Q)))
    best = np.flatnonzero(Q == Q.max())
    return int(rng.choice(best))


def run_bandit(bandit, method, T, seed, epsilon=0.1):
    rng = np.random.default_rng(seed)
    k = bandit.k
    Q = np.zeros(k)
    N = np.zeros(k, dtype=int)
    regret = np.zeros(T)
    total = 0.0
    for t in range(T):
        if method == "epsilon-greedy":
            arm = epsilon_greedy_action(Q, epsilon, rng)
        elif method == "ucb1":
            arm = ucb1_action(Q, N, max(t, 1), rng)
        else:
            raise ValueError(f"unknown method {method!r}")
        reward = bandit.pull(arm)
        Q[arm] = update_estimate(Q[arm], N[arm], reward)
        N[arm] += 1
        total += bandit.regret(arm)
        regret[t] = total
    return regret, N


if __name__ == "__main__":
    means = [0.1, 0.5, 0.9]
    gaps = np.max(means) - np.array(means)
    print("scores after 7 pulls, means (0.6, 0.5), counts (5, 2):", np.round(ucb1_scores([0.6, 0.5], [5, 2], 7), 4))
    T = 8000
    print(f"slot-machine row {means}, mean regret over 3 seeds; UCB1 bound at T=2000: {ucb1_regret_bound(gaps, 2000):.0f}")
    for method in ["epsilon-greedy", "ucb1"]:
        runs = [run_bandit(rlenvs.BernoulliBandit(means, seed=s), method, T, seed=s)[0] for s in range(3)]
        mean = np.mean(runs, axis=0)
        marks = "  ".join(f"T={t}: {mean[t - 1]:6.1f}" for t in (500, 2000, 8000))
        print(f"{method:>15}  {marks}   growth from 2000 to 8000: {mean[-1] / mean[1999]:.2f}x")
