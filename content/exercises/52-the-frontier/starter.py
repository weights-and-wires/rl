"""The research frontier: rewards a model gives itself, and where they collapse.

Test-time RL (TTRL) rewards each sampled answer for agreeing with the
majority vote of its group. INTUITOR rewards each output for the model's own
confidence, its self-certainty. Both then use GRPO-style group advantages.
You build both rewards and run each on a toy policy small enough to watch
exactly what it reinforces.

A toy policy here is a 1-D array of logits theta over K answers (or methods),
with probabilities softmax(theta).
"""

import numpy as np


def softmax(logits):
    """Softmax over the last axis. This helper is given."""
    z = np.asarray(logits, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def majority_vote(answers):
    """The most common answer in a group.

    Args:
        answers: a 1-D integer array of sampled answers.

    Returns:
        The most frequent value. Break ties toward the smallest value.
    """
    raise NotImplementedError("majority_vote")


def ttrl_rewards(answers):
    """TTRL's pseudo-reward for each sample.

    Args:
        answers: a 1-D integer array of sampled answers, shape (G,).

    Returns:
        A float array of shape (G,): 1.0 where the answer equals
        majority_vote(answers), 0.0 elsewhere.
    """
    raise NotImplementedError("ttrl_rewards")


def reward_and_label_accuracy(answers, truth):
    """How good TTRL's pseudo-rewards are, measured against the true answer.

    Args:
        answers: a 1-D integer array of sampled answers, shape (G,).
        truth: the correct answer.

    Returns:
        A tuple (reward_accuracy, label_accuracy) of floats.
        reward_accuracy is the fraction of samples whose pseudo-reward equals
        their true reward, 1.0 if answer == truth else 0.0.
        label_accuracy is 1.0 if the majority vote equals truth, else 0.0.
    """
    raise NotImplementedError("reward_and_label_accuracy")


def self_certainty(logits):
    """INTUITOR's self-certainty of an output.

    For each position t, KL(U || p_t), where U is uniform over the V tokens
    and p_t = softmax(logits[..., t, :]). Then the mean over positions.
    Work with log-probabilities, so large logits don't overflow.

    Args:
        logits: shape (..., T, V).

    Returns:
        Shape (...): a float for a single (T, V) input.
    """
    raise NotImplementedError("self_certainty")


def group_advantages(rewards):
    """GRPO advantages for one group of rewards.

    (r - mean(r)) / std(r), with the population standard deviation
    (np.std's default). When every reward is the same, return zeros.

    Args:
        rewards: shape (G,).

    Returns:
        Shape (G,).
    """
    raise NotImplementedError("group_advantages")


def policy_gradient(theta, actions, advantages):
    """Gradient of the policy-gradient surrogate for a softmax policy.

    The surrogate is mean_i advantages[i] * log softmax(theta)[actions[i]].

    Args:
        theta: logits, shape (K,).
        actions: sampled indices, shape (G,).
        advantages: shape (G,), treated as constants.

    Returns:
        The gradient in theta, shape (K,).
    """
    raise NotImplementedError("policy_gradient")


def ttrl_train(theta0, group_size=16, steps=200, lr=1.0, seed=0):
    """Test-time RL on one question with no labels.

    Use rng = np.random.default_rng(seed). Each step: draw group_size answers
    from softmax(theta) with rng.choice, score them with ttrl_rewards, turn
    the scores into group_advantages, and take
    theta += lr * policy_gradient(theta, answers, advantages).

    Args:
        theta0: initial logits over K answers, shape (K,).
        group_size, steps, lr, seed: as named.

    Returns:
        The final logits, shape (K,).
    """
    raise NotImplementedError("ttrl_train")


def intuitor_train(path_logits0, answer_logits, group_size=16, steps=200, lr=1.0, seed=0):
    """INTUITOR on a two-step toy policy.

    Step one picks a method m from softmax(theta). Step two answers with the
    fixed distribution softmax(answer_logits[m]), which is never trained.
    Every output shares the step-one term of its self-certainty, and the
    group standardisation removes a shared term, so score each sample by
    the self-certainty of its step-two distribution alone:
    self_certainty(answer_logits[m][None, :]).

    Use rng = np.random.default_rng(seed). Each step: draw group_size methods
    from softmax(theta), score each, turn the scores into group_advantages,
    and take theta += lr * policy_gradient(theta, methods, advantages).

    Args:
        path_logits0: initial logits over M methods, shape (M,).
        answer_logits: shape (M, K), one row of answer logits per method.
        group_size, steps, lr, seed: as named.

    Returns:
        The final method logits, shape (M,).
    """
    raise NotImplementedError("intuitor_train")


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
