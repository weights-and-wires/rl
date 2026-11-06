"""Reference solution for the Thompson sampling exercise."""

import numpy as np

import rlenvs


def beta_posterior(successes, failures):
    return 1.0 + np.asarray(successes, dtype=float), 1.0 + np.asarray(failures, dtype=float)


def thompson_action(successes, failures, rng):
    a, b = beta_posterior(successes, failures)
    samples = rng.beta(a, b)
    return int(np.argmax(samples))


def prob_best(successes, failures, n_samples, rng):
    a, b = beta_posterior(successes, failures)
    samples = rng.beta(a, b, size=(n_samples, len(a)))
    winners = np.argmax(samples, axis=1)
    return np.bincount(winners, minlength=len(a)) / n_samples


# Given in full: the bandit loop from the earlier chapters, extended with Thompson sampling.


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
