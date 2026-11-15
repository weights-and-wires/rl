"""Rubric rewards: a recall grader, a precision grader, and their mix.

A response is a subset of K candidate claims. Claim i has rubric weight
``w[i]`` and probability ``p[i]`` of being true. The rubric grader pays for
every claim you include. The claims grader charges ``c`` for every false claim
you include. Fill in the seven functions. The checks call them in the order
they appear here.
"""

import numpy as np


def rubric_reward(include, w):
    """Recall-style reward: the total weight of the claims you include.

    Args:
        include: array of shape ``(K,)``, 1 where the response states claim i
            and 0 where it does not.
        w: array of shape ``(K,)``, the rubric weight of each claim.

    Returns:
        A float, the sum of ``w[i]`` over the included claims.
    """
    raise NotImplementedError("rubric_reward")


def claims_reward(include, truth, c):
    """Precision-style reward: minus ``c`` for every false claim included.

    Args:
        include: array of shape ``(K,)``, 1 for an included claim.
        truth: array of shape ``(K,)``, 1 where claim i is true and 0 where it
            is false.
        c: the cost of one false claim, a positive float.

    Returns:
        A float, ``-c`` times the number of included claims that are false.
        True claims and omitted claims cost nothing.
    """
    raise NotImplementedError("claims_reward")


def gated_reward(tests_passed, rubric_score):
    """Gate a rubric score behind a test result.

    Args:
        tests_passed: True or False, whether the program's tests all passed.
        rubric_score: a float, the rubric reward of the response.

    Returns:
        The product of the two. A failed test always earns 0.
    """
    raise NotImplementedError("gated_reward")


def optimal_subset(w, p, c, alpha, beta):
    """The subset of claims that maximizes the expected mixed reward.

    The mixed reward is ``alpha * rubric_reward + beta * claims_reward``.
    Because it is a sum over claims, each claim is decided on its own.

    Args:
        w: shape ``(K,)``, rubric weights.
        p: shape ``(K,)``, probability that each claim is true.
        c: the cost of one false claim.
        alpha: the weight on the rubric reward.
        beta: the weight on the claims reward.

    Returns:
        A boolean array of shape ``(K,)``, True for a claim worth including.
        A claim whose expected net value is exactly 0 stays out.
    """
    raise NotImplementedError("optimal_subset")


def expected_reward(include, w, p, c, alpha, beta):
    """Expected mixed reward when claim i is included with weight ``include[i]``.

    ``include`` can be a 0/1 mask or a vector of inclusion probabilities. The
    expectation is over which claims turn out to be true.

    Args:
        include: shape ``(K,)``, values in ``[0, 1]``.
        w, p, c, alpha, beta: as in ``optimal_subset``.

    Returns:
        A float, the expected value of ``alpha * rubric_reward`` plus
        ``beta * claims_reward``.
    """
    raise NotImplementedError("expected_reward")


def policy_gradient(theta, w, p, c, alpha, beta):
    """Exact gradient of the expected reward for a Bernoulli policy.

    The policy includes claim i with probability ``q[i] = sigmoid(theta[i])``,
    independently of the other claims. Do not sample: differentiate the
    expectation.

    Args:
        theta: shape ``(K,)``, one logit per claim.
        w, p, c, alpha, beta: as in ``optimal_subset``.

    Returns:
        An array of shape ``(K,)``, the derivative of the expected reward with
        respect to each ``theta[i]``.
    """
    raise NotImplementedError("policy_gradient")


def train_policy(w, p, c, alpha, beta, lr=1.0, steps=600, theta0=None):
    """Gradient ascent on the expected reward, starting from ``theta0``.

    Args:
        w, p, c, alpha, beta: as in ``optimal_subset``.
        lr: the step size.
        steps: the number of ascent steps.
        theta0: shape ``(K,)``, the starting logits. Zeros when None.

    Returns:
        The final logits, shape ``(K,)``. Do not change ``theta0`` in place.
    """
    raise NotImplementedError("train_policy")


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
