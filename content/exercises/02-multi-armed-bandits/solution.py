"""Reference solution for the multi-armed bandits exercise."""

import numpy as np

import rlenvs


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


def ucb1_scores(means, counts, t):
    means = np.asarray(means, dtype=float)
    counts = np.asarray(counts, dtype=float)
    scores = np.full(means.shape, np.inf)
    pulled = counts > 0
    scores[pulled] = means[pulled] + np.sqrt(2.0 * np.log(t) / counts[pulled])
    return scores


def thompson_action(successes, failures, rng):
    samples = rng.beta(1.0 + np.asarray(successes), 1.0 + np.asarray(failures))
    return int(np.argmax(samples))


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
            scores = ucb1_scores(Q, N, max(t, 1))
            best = np.flatnonzero(scores == scores.max())
            arm = int(rng.choice(best))
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
    return regret


if __name__ == "__main__":
    means = [0.1, 0.5, 0.9]
    T = 8000
    print(f"Bernoulli arms {means}, mean regret over 3 seeds")
    for method in ["epsilon-greedy", "ucb1", "thompson"]:
        curves = [run_bandit(rlenvs.BernoulliBandit(means, seed=s), method, T, seed=s) for s in range(3)]
        mean = np.mean(curves, axis=0)
        marks = "  ".join(f"T={t}: {mean[t - 1]:6.1f}" for t in (500, 2000, 8000))
        print(f"{method:>15}  {marks}   growth from 2000 to 8000: {mean[-1] / mean[1999]:.2f}x")
