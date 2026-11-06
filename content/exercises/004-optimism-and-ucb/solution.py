"""Reference solution for the optimism and UCB exercise."""

import numpy as np

import rlenvs


def ucb_bonus(counts, t):
    counts = np.asarray(counts, dtype=float)
    bonus = np.full(counts.shape, np.inf)
    pulled = counts > 0
    bonus[pulled] = np.sqrt(2.0 * np.log(t) / counts[pulled])
    return bonus


def ucb1_scores(means, counts, t):
    return np.asarray(means, dtype=float) + ucb_bonus(counts, t)


def ucb1_action(means, counts, t, rng):
    scores = ucb1_scores(means, counts, t)
    best = np.flatnonzero(scores == scores.max())
    return int(rng.choice(best))


def ucb1_regret_bound(gaps, T):
    gaps = np.asarray(gaps, dtype=float)
    bad = gaps[gaps > 0]
    return float(8.0 * np.sum(np.log(T) / bad) + (1.0 + np.pi**2 / 3.0) * np.sum(gaps))


# Given: the bandit loop from the previous chapter, extended with UCB1.


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
