"""Reference solution: a task metric becomes a reward.

Every function takes numpy arrays or plain Python lists and returns numpy
values. Three small worlds share one idea: a score you care about becomes a
reward, and a policy gradient follows it.

Vision: score a predicted box against the true box with intersection over
union (IoU), then stop a policy from spraying boxes.
Speech: score a transcript with word error rate, then train on it with the
minimum word error rate (MWER) loss.
Robots: add a bounded correction to a fixed controller, and estimate its
gradient with REINFORCE and a leave-one-out baseline.
"""

import numpy as np



def iou(box_a, box_b):
    box_a = np.asarray(box_a, dtype=float)
    box_b = np.asarray(box_b, dtype=float)
    # A box whose far edge sits before its near edge has zero width, not a negative one.
    area_a = np.clip(box_a[..., 2] - box_a[..., 0], 0, None) * np.clip(
        box_a[..., 3] - box_a[..., 1], 0, None
    )
    area_b = np.clip(box_b[..., 2] - box_b[..., 0], 0, None) * np.clip(
        box_b[..., 3] - box_b[..., 1], 0, None
    )
    inter_w = np.minimum(box_a[..., 2], box_b[..., 2]) - np.maximum(box_a[..., 0], box_b[..., 0])
    inter_h = np.minimum(box_a[..., 3], box_b[..., 3]) - np.maximum(box_a[..., 1], box_b[..., 1])
    inter = np.clip(inter_w, 0, None) * np.clip(inter_h, 0, None)
    union = area_a + area_b - inter
    safe = np.where(union > 0, union, 1.0)
    return np.where(union > 0, inter / safe, 0.0)


def detection_reward(pred_boxes, gt_boxes, length_penalty=True):
    pred = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    gt = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    if len(pred) == 0 or len(gt) == 0:
        return 0.0
    # Every ground-truth box takes credit from its best-matching prediction.
    overlaps = iou(pred[:, None, :], gt[None, :, :])  # (K, M)
    recall = float(np.mean(overlaps.max(axis=0)))
    if not length_penalty:
        return recall
    return min(1.0, len(gt) / len(pred)) * recall


def edit_distance(ref, hyp):
    ref, hyp = list(ref), list(hyp)
    # prev[j] is the distance between the first i-1 reference words and hyp[:j].
    prev = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        cur = [i] + [0] * len(hyp)
        for j in range(1, len(hyp) + 1):
            substitute = prev[j - 1] + (ref[i - 1] != hyp[j - 1])
            delete = prev[j] + 1
            insert = cur[j - 1] + 1
            cur[j] = min(substitute, delete, insert)
        prev = cur
    return prev[-1]


def wer(ref, hyp):
    ref_words, hyp_words = ref.split(), hyp.split()
    return edit_distance(ref_words, hyp_words) / len(ref_words)


def _renormalise(logscores):
    z = np.asarray(logscores, dtype=float)
    p = np.exp(z - z.max())
    return p / p.sum()


def mwer_loss(logscores, errors):
    p_hat = _renormalise(logscores)
    errors = np.asarray(errors, dtype=float)
    # The paper's W-bar: the average error count over the list, a constant.
    w_bar = errors.mean()
    return float(np.sum(p_hat * (errors - w_bar)))


def mwer_grad(logscores, errors):
    p_hat = _renormalise(logscores)
    errors = np.asarray(errors, dtype=float)
    expected = np.sum(p_hat * errors)
    return p_hat * (errors - expected)


def residual_action(base_action, raw, bound):
    return base_action + bound * np.tanh(raw)


def reinforce_gradient(rewards, features, eps, sigma):
    n = rewards.shape[0]
    # Reward-to-go: G[i, t] = sum of rewards from step t to the end of episode i.
    to_go = np.cumsum(rewards[:, ::-1], axis=1)[:, ::-1]
    # Leave-one-out baseline: the mean reward-to-go of the other episodes.
    baseline = (to_go.sum(axis=0, keepdims=True) - to_go) / (n - 1)
    # Score of a Gaussian pre-squash action: (raw - mean) / sigma^2 * phi = eps / sigma * phi.
    score = eps[:, :, None] * features / sigma
    return np.sum((to_go - baseline)[:, :, None] * score, axis=(0, 1)) / n


if __name__ == "__main__":
    print("IoU of (0,0,2,2) and (1,1,3,3):", round(float(iou([0, 0, 2, 2], [1, 1, 3, 3])), 4))
    target = np.array([[0.1, 0.1, 0.4, 0.4]])
    spray = np.array([[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6], [0.7, 0.1, 0.9, 0.2], [0.0, 0.8, 0.2, 1.0]])
    print("spray reward without / with the length penalty:",
          detection_reward(spray, target, length_penalty=False),
          detection_reward(spray, target, length_penalty=True))
    print("WER of 'the cat sit' against 'the cat sat':", round(wer("the cat sat", "the cat sit"), 4))
    logscores, errors = np.array([2.0, 1.0, 0.5, -1.0]), np.array([0.0, 1.0, 2.0, 3.0])
    print("MWER loss:", round(mwer_loss(logscores, errors), 4))
    print("MWER gradient:", np.round(mwer_grad(logscores, errors), 4))
    print("residual action, base 0.8, raw 0.5, bound 1:", round(float(residual_action(0.8, 0.5, 1.0)), 4))
    rng = np.random.default_rng(0)
    n, sigma = 20000, 0.1
    eps = rng.standard_normal((n, 1))
    rewards = -((0.8 + sigma * eps[:, 0] - 1.0) ** 2)[:, None]
    grad = reinforce_gradient(rewards, np.ones((n, 1, 1)), eps, sigma)
    print("REINFORCE gradient at w = 0 (exact value 0.4):", np.round(grad, 3))
