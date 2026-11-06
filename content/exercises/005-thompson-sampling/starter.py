"""Thompson sampling: sample your beliefs, then act on the sample.

Fill in the three functions above the line marked "Given in full". The bandit
loop below that line, `run_bandit`, is given in full: you wrote its
epsilon-greedy and UCB1 halves in the earlier chapters, and its Thompson half
calls your `thompson_action`.

Arms are numbered 0 to k - 1. A bandit from `rlenvs` has `.k` arms,
`.pull(arm)` returns a reward (0 or 1), and `.regret(arm)` returns the gap
between the best arm's mean and this arm's mean.
"""

import numpy as np

import rlenvs


def beta_posterior(successes, failures):
    """The Beta posterior over each arm's mean, starting from a uniform Beta(1, 1) prior.

    After S pulls that paid 1 and F pulls that paid 0, an arm's belief is
    Beta(1 + S, 1 + F).

    Args:
        successes: how many pulls of each arm paid 1, shape (k,).
        failures: how many pulls of each arm paid 0, shape (k,).

    Returns:
        A tuple `(a, b)` of float arrays of shape (k,): the two Beta parameters
        of every arm.
    """
    raise NotImplementedError("beta_posterior")


def thompson_action(successes, failures, rng):
    """Pick an arm by Thompson sampling: draw one plausible mean per arm, play the largest.

    Draw all k samples in one call to `rng.beta` with array arguments, then
    take the argmax of the samples. Do not take the argmax of the posterior
    means: that would be greedy.

    Args:
        successes: how many pulls of each arm paid 1, shape (k,).
        failures: how many pulls of each arm paid 0, shape (k,).
        rng: a numpy Generator. Draw every random number from it.

    Returns:
        The chosen arm, an int.
    """
    raise NotImplementedError("thompson_action")


def prob_best(successes, failures, n_samples, rng):
    """Estimate, by sampling, the probability that each arm has the highest mean.

    Draw `n_samples` rounds of one posterior sample per arm, count how often
    each arm wins its round, and divide by `n_samples`.

    Args:
        successes: how many pulls of each arm paid 1, shape (k,).
        failures: how many pulls of each arm paid 0, shape (k,).
        n_samples: how many rounds to draw.
        rng: a numpy Generator. Draw every random number from it.

    Returns:
        Shape (k,). Entry a is the fraction of rounds that arm a won. The
        entries add up to 1.
    """
    raise NotImplementedError("prob_best")


# Given in full: the bandit loop from the earlier chapters, extended with Thompson sampling.
# `run_bandit` calls `thompson_action`, so it works once you fill that in.


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


def ucb1_action(means, counts, t, rng):
    counts = np.asarray(counts, dtype=float)
    scores = np.full(counts.shape, np.inf)
    pulled = counts > 0
    scores[pulled] = np.asarray(means, dtype=float)[pulled] + np.sqrt(2.0 * np.log(t) / counts[pulled])
    best = np.flatnonzero(scores == scores.max())
    return int(rng.choice(best))


def run_bandit(bandit, method, T, seed, epsilon=0.1):
    rng = np.random.default_rng(seed)
    k = bandit.k
    Q = np.zeros(k)
    N = np.zeros(k, dtype=int)
    successes = np.zeros(k)
    regret = np.zeros(T)
    total = 0.0
    for t in range(T):
        if method == "epsilon-greedy":
            arm = epsilon_greedy_action(Q, epsilon, rng)
        elif method == "ucb1":
            arm = ucb1_action(Q, N, max(t, 1), rng)
        elif method == "thompson":
            arm = thompson_action(successes, N - successes, rng)
        else:
            raise ValueError(f"unknown method {method!r}")
        reward = bandit.pull(arm)
        Q[arm] = update_estimate(Q[arm], N[arm], reward)
        N[arm] += 1
        successes[arm] += reward
        total += bandit.regret(arm)
        regret[t] = total
    return regret, N


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("posterior parameters after (3 successes, 1 failure) and (1, 1):", beta_posterior([3, 1], [1, 1]))
    print("probability each arm is best:", np.round(prob_best([3, 1], [1, 1], 20000, rng), 3), "(exact 5/7, 2/7)")
    means = [0.1, 0.5, 0.9]
    T = 8000
    print(f"slot-machine row {means}, mean regret over 3 seeds")
    for method in ["epsilon-greedy", "ucb1", "thompson"]:
        runs = [run_bandit(rlenvs.BernoulliBandit(means, seed=s), method, T, seed=s)[0] for s in range(3)]
        mean = np.mean(runs, axis=0)
        marks = "  ".join(f"T={t}: {mean[t - 1]:6.1f}" for t in (500, 2000, 8000))
        print(f"{method:>15}  {marks}   growth from 2000 to 8000: {mean[-1] / mean[1999]:.2f}x")
