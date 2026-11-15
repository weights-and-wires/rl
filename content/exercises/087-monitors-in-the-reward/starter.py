"""A chain-of-thought monitor in the reward, and random audits instead.

One coding prompt, four actions. The policy picks one of them:

    0  honest   solve the task honestly. The tests pass with probability p.
    1  admit    say the task failed. Reward 0, never flagged.
    2  verbal   hack the tests and say so in the chain of thought. The tests
                always pass. The monitor flags it with probability r_v.
    3  obfus    hack the tests and hide it. The tests always pass, it costs eps
                in effort, and the monitor flags it with probability r_o.

A monitor reads the chain of thought and returns a flag. You compare two
rewards. Reward (a) pays only for passing, and the monitor only measures.
Reward (b) pays for passing and subtracts lam for each flag. You also write
the closed forms that say when a penalty, or a random audit, makes honesty
the best action.

Every distribution is a 1-D numpy array of shape (4,) that sums to one.
"""

import numpy as np

HONEST, ADMIT, VERBAL, OBFUS = 0, 1, 2, 3


def softmax(z):
    """A numerically stable softmax over the last axis. This helper is given."""
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def expected_rewards(p, eps, r_v, r_o, lam=0.0, q=0.0, c=0.0):
    """The expected reward of each of the four actions.

    The reward is the pass indicator, minus eps for the obfuscated hack, minus
    lam for each monitor flag, minus c for a hack that a random audit catches.
    An audit happens with probability q and always catches a hack, whatever the
    chain of thought says. Honest solving is never flagged and never audited
    as a hack, and "admit" earns 0.

    Args:
        p: the probability that an honest solution passes.
        eps: the effort cost of the obfuscated hack.
        r_v: the monitor's flag probability on the verbalized hack.
        r_o: the monitor's flag probability on the obfuscated hack.
        lam: the penalty per flag.
        q: the probability that a rollout is audited.
        c: the penalty for a hack that an audit catches.

    Returns:
        Array of shape (4,), in the order honest, admit, verbal, obfus.
    """
    raise NotImplementedError("expected_rewards")


def kl_optimum(ref, rewards, beta):
    """The policy that maximizes E[reward] - beta * KL(pi || ref).

    Args:
        ref: the reference policy, shape (4,).
        rewards: expected reward per action, shape (4,).
        beta: the KL coefficient, greater than 0.

    Returns:
        pi, shape (4,), proportional to ref * exp(rewards / beta).
    """
    raise NotImplementedError("kl_optimum")


def detection_rates(pi, r_v, r_o):
    """How often the monitor flags a hack, and how often a hack goes unflagged.

    Args:
        pi: the policy, shape (4,).
        r_v: the flag probability on the verbalized hack.
        r_o: the flag probability on the obfuscated hack.

    Returns:
        (flagged, undetected): the probability per rollout of a hack that the
        monitor flags, and of a hack that it misses.
    """
    raise NotImplementedError("detection_rates")


def lambda_star(p, cost, r):
    """The smallest flag penalty at which honest solving beats a hack.

    A hack with effort cost `cost` that the monitor flags with probability r
    earns 1 - cost - lam * r. Honest solving earns p.

    Returns:
        The penalty lam at which the two earn the same, as a float.
    """
    raise NotImplementedError("lambda_star")


def audit_threshold(p, cost, c):
    """The smallest audit probability at which honest solving beats a hack.

    A hack with effort cost `cost` that an audit catches with probability q
    earns 1 - cost - q * c. Honest solving earns p.

    Returns:
        The probability q at which the two earn the same, as a float.
    """
    raise NotImplementedError("audit_threshold")


def policy_gradient(theta, rewards):
    """The exact gradient of E[reward] with respect to softmax logits theta.

    Args:
        theta: logits, shape (4,).
        rewards: expected reward per action, shape (4,).

    Returns:
        Array of shape (4,): the derivative of J = sum_a pi_a * rewards_a
        with respect to each logit, where pi = softmax(theta).
    """
    raise NotImplementedError("policy_gradient")


def train(ref, rewards, lr=5.0, steps=6000):
    """Gradient ascent on E[reward] from the reference policy.

    Start at theta = log(ref) and take `steps` steps of
    theta += lr * policy_gradient(theta, rewards). There is no KL term.

    Returns:
        The final policy, shape (4,).
    """
    raise NotImplementedError("train")


if __name__ == "__main__":
    p, eps, r_v, r_o = 0.3, 0.05, 0.95, 0.6
    ref = np.array([0.55, 0.25, 0.18, 0.02])
    beta = 0.05
    names = ["honest", "admit", "verbal", "obfus"]

    for label, kwargs in [("(a) pass only", {}), ("(b) pass - 0.5 * flag", {"lam": 0.5}),
                          ("(c) pass, audit q=0.2 c=5", {"q": 0.2, "c": 5.0})]:
        R = expected_rewards(p, eps, r_v, r_o, **kwargs)
        pi = kl_optimum(ref, R, beta)
        flagged, undetected = detection_rates(pi, r_v, r_o)
        print(label)
        print("  pi*:", {n: round(float(x), 4) for n, x in zip(names, pi)})
        print(f"  flagged hacks {flagged:.4f}   undetected hacks {undetected:.4f}")

    print("lambda* to beat verbal:", round(lambda_star(p, 0.0, r_v), 4),
          " to beat obfus:", round(lambda_star(p, eps, r_o), 4))
    print("audit q* to beat verbal:", round(audit_threshold(p, 0.0, 5.0), 4),
          " to beat obfus:", round(audit_threshold(p, eps, 5.0), 4))
    pi = train(ref, expected_rewards(p, eps, r_v, r_o, lam=0.5))
    print("trained under (b):", {n: round(float(x), 3) for n, x in zip(names, pi)})
