"""Reference solution for the gradient bandits exercise."""

import numpy as np

import rlenvs


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
