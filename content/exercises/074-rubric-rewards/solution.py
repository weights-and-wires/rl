"""Rubric rewards: a recall grader, a precision grader, and their mix.

A response is a subset of K candidate claims. Claim i has rubric weight
``w[i]`` and probability ``p[i]`` of being true. The rubric grader pays for
every claim you include. The claims grader charges ``c`` for every false claim
you include.
"""

import numpy as np


def rubric_reward(include, w):
    """Recall-style reward: the total weight of the claims you include."""
    return float(np.sum(np.asarray(include, dtype=float) * np.asarray(w, dtype=float)))


def claims_reward(include, truth, c):
    """Precision-style reward: minus ``c`` for every false claim included."""
    include = np.asarray(include, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return float(-c * np.sum(include * (1.0 - truth)))


def gated_reward(tests_passed, rubric_score):
    """Gate a rubric score behind a test result."""
    return float(tests_passed) * float(rubric_score)


def _net_value(w, p, c, alpha, beta):
    """Expected value of including each claim: alpha*w - beta*c*(1-p)."""
    w = np.asarray(w, dtype=float)
    p = np.asarray(p, dtype=float)
    return alpha * w - beta * c * (1.0 - p)


def optimal_subset(w, p, c, alpha, beta):
    """Include claim i exactly when alpha*w[i] > beta*c*(1 - p[i])."""
    return _net_value(w, p, c, alpha, beta) > 0.0


def expected_reward(include, w, p, c, alpha, beta):
    """Expected mixed reward, linear in the inclusion vector."""
    include = np.asarray(include, dtype=float)
    return float(np.sum(include * _net_value(w, p, c, alpha, beta)))


def policy_gradient(theta, w, p, c, alpha, beta):
    """Exact gradient: net value times dq/dtheta = q * (1 - q)."""
    q = 1.0 / (1.0 + np.exp(-np.asarray(theta, dtype=float)))
    return _net_value(w, p, c, alpha, beta) * q * (1.0 - q)


def train_policy(w, p, c, alpha, beta, lr=1.0, steps=600, theta0=None):
    """Gradient ascent on the expected reward."""
    w = np.asarray(w, dtype=float)
    theta = np.zeros(len(w)) if theta0 is None else np.array(theta0, dtype=float)
    for _ in range(steps):
        theta = theta + lr * policy_gradient(theta, w, p, c, alpha, beta)
    return theta


if __name__ == "__main__":
    w = np.array([3.0, 2.0, 2.0, 1.0])
    p = np.array([0.9, 0.7, 0.5, 0.2])
    c, alpha, beta = 5.0, 1.0, 1.0
    keep = optimal_subset(w, p, c, alpha, beta)
    print("optimal subset:", keep.astype(int))
    print("brain dump:", expected_reward(np.ones(4), w, p, c, alpha, beta))
    print("optimal:", expected_reward(keep.astype(float), w, p, c, alpha, beta))
    theta = train_policy(w, p, c, alpha, beta)
    print("trained inclusion probabilities:", np.round(1 / (1 + np.exp(-theta)), 3))
