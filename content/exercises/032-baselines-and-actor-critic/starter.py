"""Baselines and a one-step actor-critic.

Fill in the functions below. Part 1 works with a softmax policy over k arms,
where you can compute the gradient estimator's mean and variance exactly and
find the constant baseline that minimises the variance. Part 2 replaces the
sampled return with a critic's one-step TD error on a tabular MDP, checks that
the error averages to the advantage, and then trains an actor-critic.

`softmax` is given.
"""

import numpy as np

import rlenvs


def softmax(logits):
    """Softmax along the last axis. Given."""
    logits = np.asarray(logits, dtype=float)
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# Part 1: a softmax policy over k arms --------------------------------------


def bandit_gradient_samples(theta, arms, rewards, baseline):
    """Per-sample score-function estimates of the gradient, with a baseline.

    Sample i's estimate is (rewards[i] - baseline) * grad log pi(arms[i]),
    where grad log pi(a) = e_a - pi for the softmax pi = softmax(theta).

    Args:
        theta: logits, shape (k,).
        arms: the arms pulled, shape (N,), integers.
        rewards: the rewards received, shape (N,).
        baseline: a constant b.

    Returns:
        The estimates, shape (N, k), one row per sample.
    """
    raise NotImplementedError("bandit_gradient_samples")


def exact_bandit_gradient(theta, means):
    """The exact gradient of J(theta) = sum_a pi(a) * means[a].

    Args:
        theta: logits, shape (k,).
        means: each arm's expected reward, shape (k,).

    Returns:
        The gradient, shape (k,).
    """
    raise NotImplementedError("exact_bandit_gradient")


def estimator_variance(theta, means, sigma, baseline):
    """The exact total variance of one sample from bandit_gradient_samples.

    Arm a pays means[a] plus Gaussian noise with standard deviation sigma.
    "Total variance" is the trace of the covariance matrix: the expected
    squared distance of one estimate from the true gradient.

    Args:
        theta: logits, shape (k,).
        means: each arm's expected reward, shape (k,).
        sigma: the reward noise's standard deviation.
        baseline: a constant b.

    Returns:
        The total variance, a float.
    """
    raise NotImplementedError("estimator_variance")


def optimal_baseline(theta, means):
    """The constant baseline that minimises estimator_variance.

    Args:
        theta: logits, shape (k,).
        means: each arm's expected reward, shape (k,).

    Returns:
        b*, a float. It doesn't depend on sigma.
    """
    raise NotImplementedError("optimal_baseline")


# Part 2: a critic on a tabular MDP -----------------------------------------


def expected_td_error(mdp, v):
    """The expected one-step TD error for every state and action.

    E[delta | s, a] = R[s, a] + gamma * sum_s2 P[s, a, s2] v[s2] - v[s].

    Args:
        mdp: a `rlenvs.TabularMDP` with no terminal states.
        v: the critic's values, shape (S,).

    Returns:
        An array of shape (S, A).
    """
    raise NotImplementedError("expected_td_error")




def actor_critic(mdp, steps, alpha_actor, alpha_critic, seed):
    """Train a tabular softmax actor and a tabular critic, one step at a time.

    Start from theta = 0 and v = 0, and from a state drawn from mdp.mu0.
    At every step, sample a from pi(. | s), sample (s2, r) with
    `mdp.sample(s, a, rng)`, compute delta = r + gamma * v[s2] - v[s], move
    v[s] by alpha_critic * delta, and move theta[s] by
    alpha_actor * delta * grad log pi(a | s). Then continue from s2, except
    that with probability 1 - gamma you restart from a fresh draw of mu0.

    Args:
        mdp: a `rlenvs.TabularMDP` with no terminal states and gamma < 1.
        steps: the number of steps.
        alpha_actor: the actor's step size.
        alpha_critic: the critic's step size.
        seed: seeds `np.random.default_rng`, used for every draw.

    Returns:
        (theta, v), shapes (S, A) and (S,).
    """
    raise NotImplementedError("actor_critic")


if __name__ == "__main__":
    theta = np.log(np.array([0.25, 0.25, 0.5]))
    means = np.array([1.0, 0.0, 2.0])
    b_star = optimal_baseline(theta, means)
    print("Three arms, pi = [0.25, 0.25, 0.5], means = [1, 0, 2], sigma = 1")
    for name, b in [("no baseline", 0.0), ("b = V", float(softmax(theta) @ means)), ("b = b*", b_star)]:
        print(f"  {name:12s} b = {b:.3f}  variance = {estimator_variance(theta, means, 1.0, b):.4f}")

    mdp = rlenvs.random_mdp(n_states=5, n_actions=3, gamma=0.9, seed=0)
    start = float(mdp.mu0 @ rlenvs.policy_evaluation(mdp, rlenvs.uniform_policy(mdp)))
    best = float(mdp.mu0 @ rlenvs.value_iteration(mdp))
    theta, v = actor_critic(mdp, steps=5000, alpha_actor=0.5, alpha_critic=0.1, seed=0)
    learned = float(mdp.mu0 @ rlenvs.policy_evaluation(mdp, softmax(theta)))
    print(f"\nActor-critic on a random 5-state MDP after 5000 steps:")
    print(f"  J(uniform) = {start:.3f}, J(learned) = {learned:.3f}, J* = {best:.3f}")
