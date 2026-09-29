"""Reference solution for the research frontier exercise."""

import numpy as np


def softmax(logits):
    """Softmax over the last axis."""
    z = np.asarray(logits, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def majority_vote(answers):
    """The most common answer, breaking ties toward the smallest value."""
    values, counts = np.unique(np.asarray(answers), return_counts=True)
    return values[np.argmax(counts)]


def ttrl_rewards(answers):
    """TTRL's pseudo-reward: 1 for agreeing with the majority, 0 otherwise."""
    answers = np.asarray(answers)
    return (answers == majority_vote(answers)).astype(float)


def reward_and_label_accuracy(answers, truth):
    """How often the pseudo-reward equals the true reward, and whether the majority is right."""
    answers = np.asarray(answers)
    pseudo = ttrl_rewards(answers)
    true = (answers == truth).astype(float)
    return float(np.mean(pseudo == true)), float(majority_vote(answers) == truth)


def self_certainty(logits):
    """INTUITOR's self-certainty: the mean over positions of KL(uniform || p)."""
    logits = np.asarray(logits, dtype=float)
    V = logits.shape[-1]
    log_p = logits - logits.max(axis=-1, keepdims=True)
    log_p = log_p - np.log(np.exp(log_p).sum(axis=-1, keepdims=True))
    # KL(U || p) = sum_j (1/V) log((1/V) / p_j) = -log V - mean_j log p_j.
    kl = -np.log(V) - log_p.mean(axis=-1)
    return kl.mean(axis=-1)


def group_advantages(rewards):
    """GRPO advantages: standardise within the group, or zeros when all rewards tie."""
    rewards = np.asarray(rewards, dtype=float)
    std = rewards.std()
    if std < 1e-12:
        return np.zeros_like(rewards)
    return (rewards - rewards.mean()) / std


def policy_gradient(theta, actions, advantages):
    """Gradient in theta of mean_i advantages[i] * log softmax(theta)[actions[i]]."""
    pi = softmax(theta)
    actions = np.asarray(actions)
    advantages = np.asarray(advantages, dtype=float)
    onehot = np.zeros((actions.size, theta.size))
    onehot[np.arange(actions.size), actions] = 1.0
    return (advantages[:, None] * (onehot - pi)).mean(axis=0)


def ttrl_train(theta0, group_size=16, steps=200, lr=1.0, seed=0):
    """Test-time RL on one question: majority-vote rewards and GRPO advantages."""
    rng = np.random.default_rng(seed)
    theta = np.array(theta0, dtype=float)
    for _ in range(steps):
        answers = rng.choice(theta.size, size=group_size, p=softmax(theta))
        adv = group_advantages(ttrl_rewards(answers))
        theta = theta + lr * policy_gradient(theta, answers, adv)
    return theta


def intuitor_train(path_logits0, answer_logits, group_size=16, steps=200, lr=1.0, seed=0):
    """INTUITOR on a two-step policy: learn which method to use, rewarded by self-certainty."""
    rng = np.random.default_rng(seed)
    theta = np.array(path_logits0, dtype=float)
    answer_logits = np.asarray(answer_logits, dtype=float)
    certainty = self_certainty(answer_logits[:, None, :])  # one position per method
    for _ in range(steps):
        paths = rng.choice(theta.size, size=group_size, p=softmax(theta))
        # A sample's self-certainty depends on its method, not on the answer
        # it then draws, so the answers never need sampling here.
        adv = group_advantages(certainty[paths])
        theta = theta + lr * policy_gradient(theta, paths, adv)
    return theta


if __name__ == "__main__":
    answers = np.array([3, 3, 4, 6, 7, 8, 9, 5])
    print("majority", majority_vote(answers), "rewards", ttrl_rewards(answers))
    print("reward and label accuracy with truth 5:", reward_and_label_accuracy(answers, 5))
    print("self-certainty of [0.7, 0.1, 0.1, 0.1]:",
          round(float(self_certainty(np.log([[0.7, 0.1, 0.1, 0.1]]))), 4))

    for name, p0 in (("right plurality", [0.45, 0.25, 0.1, 0.1, 0.1]),
                     ("wrong plurality", [0.25, 0.45, 0.1, 0.1, 0.1])):
        theta = ttrl_train(np.log(p0), seed=0)
        print(f"TTRL, {name}: P(correct answer 0) {p0[0]:.2f} -> {softmax(theta)[0]:.3f}")

    table = np.log(np.array([[0.70, 0.10, 0.10, 0.05, 0.05],     # careful
                             [0.02, 0.95, 0.01, 0.01, 0.01],     # confident, wrong
                             [0.20, 0.20, 0.20, 0.20, 0.20]]))   # guessing
    theta = intuitor_train(np.zeros(3), table, seed=0)
    pi = softmax(theta)
    acc0 = softmax(np.zeros(3)) @ softmax(table)[:, 0]
    print(f"INTUITOR: method probabilities {np.round(pi, 3)}, "
          f"accuracy {acc0:.3f} -> {pi @ softmax(table)[:, 0]:.3f}")
