"""Checks for the research frontier exercise."""

import numpy as np

from rlcheck import almost, check, close, equal, fail, finite_difference, has, shape

CAREFUL_WRONG = np.log(np.array([
    [0.70, 0.10, 0.10, 0.05, 0.05],  # careful: usually right, never sure
    [0.02, 0.95, 0.01, 0.01, 0.01],  # confident and wrong
    [0.20, 0.20, 0.20, 0.20, 0.20],  # guessing
]))
CONFIDENT_RIGHT = np.log(np.array([
    [0.95, 0.02, 0.01, 0.01, 0.01],  # confident and right
    [0.30, 0.40, 0.10, 0.10, 0.10],
    [0.20, 0.20, 0.20, 0.20, 0.20],
]))


def _softmax(z):
    z = np.asarray(z, dtype=float) - np.max(z)
    e = np.exp(z)
    return e / e.sum()


@check("majority_vote finds the mode and breaks ties toward the smallest answer")
def test_majority(sub):
    majority_vote = has(sub, "majority_vote")
    equal(int(majority_vote(np.array([3, 3, 4, 6, 7, 8, 9, 5]))), 3, label="mode of the lesson's samples")
    equal(int(majority_vote(np.array([2, 1, 2, 1]))), 1, label="a tie between 1 and 2")
    equal(int(majority_vote(np.array([7]))), 7, label="one sample")


@check("ttrl_rewards and the two accuracies match the lesson's lucky-hit example")
def test_ttrl_rewards(sub):
    ttrl_rewards = has(sub, "ttrl_rewards")
    accuracy = has(sub, "reward_and_label_accuracy")
    answers = np.array([3, 3, 4, 6, 7, 8, 9, 5])
    close(ttrl_rewards(answers), np.array([1, 1, 0, 0, 0, 0, 0, 0]), atol=0, label="rewards")
    reward_acc, label_acc = accuracy(answers, 5)
    almost(reward_acc, 0.625, 1e-12, label="reward accuracy with truth 5")
    almost(label_acc, 0.0, 1e-12, label="label accuracy with truth 5")
    reward_acc, label_acc = accuracy(answers, 3)
    almost(reward_acc, 1.0, 1e-12, label="reward accuracy with truth 3")
    almost(label_acc, 1.0, 1e-12, label="label accuracy with truth 3")


@check("With a systematic wrong answer, reward accuracy stays high while label accuracy is low")
def test_accuracy_gap(sub):
    accuracy = has(sub, "reward_and_label_accuracy")
    rng = np.random.default_rng(0)
    p = np.array([0.25, 0.35] + [0.05] * 8)  # answer 0 is right, answer 1 is a common mistake
    rewards, labels = [], []
    for _ in range(1000):
        r, l = accuracy(rng.choice(10, size=16, p=p), 0)
        rewards.append(r)
        labels.append(l)
    if not (np.mean(rewards) > 0.55 and np.mean(labels) < 0.42):
        fail(f"over 1,000 simulated questions expected reward accuracy above 0.55 and label "
             f"accuracy below 0.42, got {np.mean(rewards):.3f} and {np.mean(labels):.3f}")


@check("self_certainty gives 0.4298 for [0.7, 0.1, 0.1, 0.1] and 0 for a uniform distribution")
def test_self_certainty_values(sub):
    self_certainty = has(sub, "self_certainty")
    p = np.array([0.7, 0.1, 0.1, 0.1])
    want = 0.25 * (np.log(0.25 / 0.7) + 3 * np.log(2.5))
    almost(float(self_certainty(np.log(p)[None, :])), want, 1e-10, label="one position")
    almost(float(self_certainty(np.zeros((3, 6)))), 0.0, 1e-12, label="uniform logits")
    two = np.stack([np.log(p), np.zeros(4)])
    almost(float(self_certainty(two)), want / 2, 1e-10, label="mean over two positions")


@check("self_certainty ignores a constant added to logits, handles batches, and stays finite")
def test_self_certainty_props(sub):
    self_certainty = has(sub, "self_certainty")
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(2, 3, 5, 7))
    got = np.asarray(self_certainty(logits))
    shape(got, (2, 3), label="batched self-certainty")
    close(np.asarray(self_certainty(logits + 50.0)), got, atol=1e-10,
          label="after adding 50 to every logit")
    if np.any(got < -1e-12):
        fail(f"self-certainty is a KL divergence and can't be negative, got min {got.min():.3g}")
    big = np.array([[1000.0, 0.0, 0.0]])
    value = float(self_certainty(big))
    # log p is about [0, -1000, -1000], so KL(U || p) = -log 3 + 2000 / 3.
    almost(value, 2000.0 / 3.0 - np.log(3.0), 1e-6, label="logits [1000, 0, 0]")


@check("group_advantages standardises the group, and a tied group gets all zeros")
def test_group_advantages(sub):
    group_advantages = has(sub, "group_advantages")
    adv = group_advantages(np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    close(adv, np.array([np.sqrt(3.0)] * 2 + [-1 / np.sqrt(3.0)] * 6), atol=1e-12,
          label="advantages for rewards 1, 1, 0, 0, 0, 0, 0, 0")
    close(group_advantages(np.ones(5)), np.zeros(5), atol=0, label="a group that all agrees")


@check("policy_gradient matches finite differences of mean_i A_i log pi(a_i)")
def test_policy_gradient(sub):
    policy_gradient = has(sub, "policy_gradient")
    rng = np.random.default_rng(2)
    theta0 = rng.normal(size=5)
    actions = np.array([0, 3, 3, 1, 4, 3])
    adv = rng.normal(size=6)

    def surrogate(theta):
        return float(np.mean(adv * np.log(_softmax(theta)[actions])))

    got = policy_gradient(theta0, actions, adv)
    shape(got, (5,), label="gradient")
    close(got, finite_difference(surrogate, theta0.copy()), rtol=1e-6, atol=1e-9,
          label="policy_gradient against finite differences")


@check("TTRL locks onto whichever answer starts as the plurality, right or wrong")
def test_ttrl_train(sub):
    ttrl_train = has(sub, "ttrl_train")
    right = _softmax(ttrl_train(np.log([0.45, 0.25, 0.1, 0.1, 0.1]), 16, 200, 1.0, seed=0))
    if right[0] < 0.95:
        fail(f"starting with the right answer (0) as the plurality, TTRL should end with "
             f"P(answer 0) above 0.95, got {np.round(right, 3)}")
    wrong = _softmax(ttrl_train(np.log([0.25, 0.45, 0.1, 0.1, 0.1]), 16, 200, 1.0, seed=0))
    if wrong[1] < 0.95:
        fail(f"starting with a wrong answer (1) as the plurality, TTRL should collapse onto it "
             f"with P(answer 1) above 0.95, got {np.round(wrong, 3)}")
    entropy = -np.sum(wrong * np.log(np.maximum(wrong, 1e-300)))
    if entropy > 0.1:
        fail(f"the collapsed policy should have entropy below 0.1 nats, got {entropy:.3f}")


@check("INTUITOR picks the most confident method, which helps only when confidence is right")
def test_intuitor(sub):
    intuitor_train = has(sub, "intuitor_train")
    for table, name, want_method in ((CAREFUL_WRONG, "confident-and-wrong", 1),
                                     (CONFIDENT_RIGHT, "confident-and-right", 0)):
        pi = _softmax(intuitor_train(np.zeros(3), table, 16, 200, 1.0, seed=0))
        if pi[want_method] < 0.95:
            fail(f"with the {name} table INTUITOR should put over 0.95 on method "
                 f"{want_method}, the most confident one; got {np.round(pi, 3)}")
    accuracy = _softmax(intuitor_train(np.zeros(3), CAREFUL_WRONG, 16, 200, 1.0, seed=0)) @ \
        np.exp(CAREFUL_WRONG)[:, 0]
    if accuracy > 0.05:
        fail(f"INTUITOR on the confident-and-wrong table should drive accuracy from 0.307 to "
             f"below 0.05, got {accuracy:.3f}")
