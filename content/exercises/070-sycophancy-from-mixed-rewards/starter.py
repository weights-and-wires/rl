"""Sycophancy from mixed rewards: a toy world with an exact answer.

A prompt holds a user's claim, and the claim is right with probability `rho`.
The model has two replies: agree (action 0) or correct the user (action 1).
The state is 0 when the user is right and 1 when the user is wrong, so every
table below is indexed `[state, action]`.

You write `fit_bradley_terry`, `mixed_rewards`, `optimal_policy`,
`critical_weight`, `slice_eval`, and `best_of_n_policy`. `simulate_comparisons`
is given: it plays the part of a human rater.

A response has two features, `(correct, agree)`. The rater's hidden reward is
`k * correct + b * agree`, and the rater prefers response a to response b with
probability `sigmoid(reward(a) - reward(b))`. The number `b` is the agreement
bias.
"""

import numpy as np

K_TRUE = 1.0  # how much the rater values a correct reply
B_TRUE = 3.0  # the rater's bias toward replies that agree

# Features of each (state, action) pair, as (correct, agree).
FEATURES = np.array(
    [
        [[1.0, 1.0], [0.0, 0.0]],  # user right: agree is correct, correcting is wrong
        [[0.0, 1.0], [1.0, 0.0]],  # user wrong: agree is wrong, correcting is right
    ]
)

# The truth reward is +1 for a correct reply and 0 otherwise.
TRUTH = FEATURES[:, :, 0]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Given ----------------------------------------------------------------------


def simulate_comparisons(theta, n, rho, seed):
    """Simulate `n` rater comparisons between an agreeing and a correcting reply.

    Returns `(feats_a, feats_b, a_wins)`. The first two have shape `(n, 2)`, with
    columns `(correct, agree)`. `a_wins` has shape `(n,)` and holds 1.0 when the
    rater preferred response a. Which reply is "a" is a fair coin flip.
    """
    rng = np.random.default_rng(seed)
    state = (rng.random(n) >= rho).astype(int)
    agree, correct = FEATURES[state, 0], FEATURES[state, 1]
    swap = rng.random(n) < 0.5
    feats_a = np.where(swap[:, None], correct, agree)
    feats_b = np.where(swap[:, None], agree, correct)
    p_a = sigmoid((feats_a - feats_b) @ np.asarray(theta, dtype=float))
    a_wins = (rng.random(n) < p_a).astype(float)
    return feats_a, feats_b, a_wins


# Your functions -------------------------------------------------------------


def fit_bradley_terry(feats_a, feats_b, a_wins, iters=25):
    """Fit the Bradley-Terry reward weights `theta` by maximum likelihood.

    The model says P(a preferred) = sigmoid(theta . (feats_a - feats_b)).
    Newton's method from `theta = 0` converges in a few steps.

    Args:
        feats_a, feats_b: arrays of shape `(n, d)`.
        a_wins: array of shape `(n,)`, 1.0 where response a was preferred.
        iters: number of Newton steps.

    Returns:
        `theta`, an array of shape `(d,)`.
    """
    raise NotImplementedError("fit_bradley_terry")


def mixed_rewards(w, k, b):
    """The mixed reward `(1 - w) * truth + w * thumbs` for every (state, action).

    The thumbs reward is `k * correct + b * agree`. Use `TRUTH` and `FEATURES`.

    Returns:
        An array of shape `(2, 2)`, indexed `[state, action]`.
    """
    raise NotImplementedError("mixed_rewards")


def optimal_policy(p_ref, reward, beta):
    """The optimum of expected reward minus `beta` times KL to `p_ref`.

    Args:
        p_ref: reference policy, shape `(2, 2)`, each row sums to 1.
        reward: reward table, shape `(2, 2)`.
        beta: the KL price, a positive float.

    Returns:
        An array of shape `(2, 2)` whose rows sum to 1.
    """
    raise NotImplementedError("optimal_policy")


def critical_weight(k, b, beta, p_agree_ref):
    """The mixing weight w* at which the policy agrees with a wrong user half the time.

    Args:
        k, b: the thumbs reward's weights on correct and agree.
        beta: the KL price.
        p_agree_ref: the reference policy's probability of agreeing, the same
            for right and wrong users.

    Returns:
        w* as a float, or `np.inf` if no weight reaches one half.
    """
    raise NotImplementedError("critical_weight")


def slice_eval(policy, rho):
    """Accuracy of a policy overall and on each slice of users.

    Args:
        policy: array of shape `(2, 2)`, indexed `[state, action]`.
        rho: probability that the user is right.

    Returns:
        `(acc_right, acc_wrong, aggregate)`: the probability of a correct reply
        when the user is right, when the user is wrong, and averaged over users.
    """
    raise NotImplementedError("slice_eval")


def best_of_n_policy(p, reward, n):
    """The distribution of the best of `n` draws from `p`, ranked by `reward`.

    This is the exact order-statistic formula from the chapter on reward
    hacking. Assume the rewards are distinct.

    Args:
        p: 1-D array of probabilities.
        reward: 1-D array of rewards, the same length.
        n: number of draws, a positive integer.

    Returns:
        An array shaped like `p`, in the original order.
    """
    raise NotImplementedError("best_of_n_policy")


if __name__ == "__main__":
    rho, beta = 0.8, 1.0
    p_ref = np.full((2, 2), 0.5)
    fa, fb, y = simulate_comparisons([K_TRUE, B_TRUE], 4000, rho, seed=0)
    k_hat, b_hat = fit_bradley_terry(fa, fb, y)
    print(f"rater weights: true ({K_TRUE}, {B_TRUE}), fitted ({k_hat:.2f}, {b_hat:.2f})")
    w_star = critical_weight(k_hat, b_hat, beta, 0.5)
    print(f"critical weight w* = {w_star:.3f}")
    print("   w   right  wrong  aggregate")
    for w in (0.0, 0.2, w_star, 0.5, 0.8):
        pi = optimal_policy(p_ref, mixed_rewards(w, k_hat, b_hat), beta)
        r, wr, agg = slice_eval(pi, rho)
        print(f"{w:5.2f}  {r:.3f}  {wr:.3f}  {agg:.3f}")
    print("best-of-n agree rate on a wrong user:",
          [round(float(best_of_n_policy(np.array([0.5, 0.5]), np.array([3.0, 1.0]), n)[0]), 4)
           for n in (1, 2, 4, 8)])
