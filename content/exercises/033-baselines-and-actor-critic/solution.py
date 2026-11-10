"""Reference solution for the baselines and actor-critic exercise."""

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
    pi = softmax(theta)
    arms = np.asarray(arms, dtype=int)
    rewards = np.asarray(rewards, dtype=float)
    scores = -np.tile(pi, (len(arms), 1))
    scores[np.arange(len(arms)), arms] += 1.0
    return (rewards - baseline)[:, None] * scores


def exact_bandit_gradient(theta, means):
    pi = softmax(theta)
    means = np.asarray(means, dtype=float)
    return pi * (means - pi @ means)


def estimator_variance(theta, means, sigma, baseline):
    pi = softmax(theta)
    means = np.asarray(means, dtype=float)
    k = len(pi)
    score_sq = ((np.eye(k) - pi) ** 2).sum(axis=1)  # |e_a - pi|^2 for each arm
    second_moment = np.sum(pi * ((means - baseline) ** 2 + sigma**2) * score_sq)
    g = exact_bandit_gradient(theta, means)
    return float(second_moment - g @ g)


def optimal_baseline(theta, means):
    pi = softmax(theta)
    means = np.asarray(means, dtype=float)
    k = len(pi)
    weight = pi * ((np.eye(k) - pi) ** 2).sum(axis=1)
    return float(weight @ means / weight.sum())


# Part 2: a critic on a tabular MDP -----------------------------------------


def expected_td_error(mdp, v):
    v = np.asarray(v, dtype=float)
    return mdp.R + mdp.gamma * mdp.P @ v - v[:, None]




def actor_critic(mdp, steps, alpha_actor, alpha_critic, seed):
    rng = np.random.default_rng(seed)
    S, A = mdp.n_states, mdp.n_actions
    theta = np.zeros((S, A))
    v = np.zeros(S)
    s = int(rng.choice(S, p=mdp.mu0))
    for _ in range(steps):
        pi = softmax(theta[s])
        a = int(rng.choice(A, p=pi))
        s2, r = mdp.sample(s, a, rng)
        delta = r + mdp.gamma * v[s2] - v[s]
        v[s] += alpha_critic * delta
        score = -pi
        score[a] += 1.0
        theta[s] += alpha_actor * delta * score
        # Restarting with probability 1 - gamma makes the visited states follow
        # the discounted visitation from mu0. The TD target above still uses
        # the true next state s2.
        s = int(rng.choice(S, p=mdp.mu0)) if rng.random() < 1.0 - mdp.gamma else s2
    return theta, v


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
