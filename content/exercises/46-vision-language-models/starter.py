"""Aligning vision-language models: IoU rewards, mDPO, and GRPO for grounding.

Fill in the functions under "The exercise". The toy world above them is
given: `make_scenes` draws images that each hold one bright rectangle,
`box_features` is a frozen "vision encoder", and `BoxPolicy` is a small
Gaussian policy that proposes one box per image.

Boxes are always `(x1, y1, x2, y2)`: the left, top, right, and bottom edges,
in normalised coordinates between 0 and 1.
"""

import numpy as np
import torch

torch.set_num_threads(1)

# The toy grounding world. The same code sits in starter.py. -------------

SIZE = 16  # the image is SIZE by SIZE pixels


def make_scenes(n, rng):
    """Draw `n` images, each holding one bright rectangle on a noisy background.

    Returns `(images, boxes)`. `images` has shape `(n, SIZE, SIZE)`. `boxes`
    has shape `(n, 4)` and holds each rectangle as `(x1, y1, x2, y2)` in
    normalised coordinates: pixel edges divided by SIZE, so every value is in
    `[0, 1]`, and `x2 > x1`, `y2 > y1`.
    """
    w = rng.integers(3, 9, n)
    h = rng.integers(3, 9, n)
    x1 = rng.integers(0, SIZE - w + 1)
    y1 = rng.integers(0, SIZE - h + 1)
    images = np.zeros((n, SIZE, SIZE))
    for i in range(n):
        images[i, y1[i] : y1[i] + h[i], x1[i] : x1[i] + w[i]] = 1.0
    images += 0.1 * rng.standard_normal(images.shape)
    boxes = np.stack([x1, y1, x1 + w, y1 + h], axis=1) / SIZE
    return images, boxes


def box_features(images):
    """A frozen "vision encoder": the brightest pixel in every column, then in every row.

    Returns a float32 tensor of shape `(n, 2 * SIZE)`.
    """
    feats = np.concatenate([images.max(axis=1), images.max(axis=2)], axis=1)
    return torch.tensor(feats, dtype=torch.float32)


class BoxPolicy(torch.nn.Module):
    """A Gaussian policy over one box, `(x1, y1, x2, y2)`, given image features.

    The mean is a small MLP squashed into `[0, 1]`. The log standard deviation
    is one learned number per coordinate, the same for every image.
    """

    def __init__(self, hidden=64, init_std=0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2 * SIZE, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 4)
        )
        self.log_std = torch.nn.Parameter(torch.full((4,), float(np.log(init_std))))

    def forward(self, features):
        mean = torch.sigmoid(self.net(features))
        return mean, self.log_std.expand_as(mean)


# The exercise. ----------------------------------------------------------


def iou(box_a, box_b):
    """Intersection over union of axis-aligned boxes, with broadcasting.

    Args:
        box_a: boxes of shape `(..., 4)`, each `(x1, y1, x2, y2)`.
        box_b: boxes of a shape that broadcasts against `box_a`, `(..., 4)`.

    Returns:
        An array of the broadcast batch shape (a 0-d array for two single
        boxes). A box with `x2 <= x1` or `y2 <= y1` has zero area. When the
        union of two boxes has zero area, the IoU is 0. Never NaN.
    """
    raise NotImplementedError("iou")


def dense_logprob(token_logps, corrected, gamma=5.0):
    """RLHF-V's dense score: a weighted mean of token log-probabilities.

    Args:
        token_logps: log p(y_i | x, y_<i) for each token, shape `(..., T)`.
        corrected: boolean array of the same shape, true for tokens that the
            annotator rewrote.
        gamma: the weight on corrected tokens. Unchanged tokens weigh 1.

    Returns:
        sum(w * token_logps) / sum(w) along the last axis, shape `(...)`,
        where `w` is gamma on corrected tokens and 1 elsewhere.
    """
    raise NotImplementedError("dense_logprob")


def mdpo_loss(chosen_logratio, rejected_logratio, chosen_corrupt_logratio, beta, delta=0.0):
    """The three terms of mDPO, each averaged over the batch.

    Every input is a log-ratio log pi_theta(y | image, q) - log pi_ref(y | image, q),
    shape `(n,)`.

    Args:
        chosen_logratio: the chosen response y_w on the real image m_w.
        rejected_logratio: the rejected response y_l on the real image m_w.
        chosen_corrupt_logratio: the chosen response y_w on the corrupted image m_l.
        beta: the KL coefficient.
        delta: the anchor's target for the chosen implicit reward.

    Returns:
        A tuple of three floats `(dpo, copo, anchor)`:
        dpo = mean of -log sigmoid(beta * chosen - beta * rejected),
        copo = mean of -log sigmoid(beta * chosen - beta * chosen_corrupt),
        anchor = mean of -log sigmoid(beta * chosen - delta).
        Compute -log sigmoid without overflow for large arguments.
    """
    raise NotImplementedError("mdpo_loss")


def detection_reward(pred_boxes, gt_boxes, length_penalty=True):
    """A recall-style detection reward, with VLM-R1's length penalty.

    Args:
        pred_boxes: the policy's boxes, shape `(K, 4)`. K can be 0.
        gt_boxes: the ground-truth boxes, shape `(M, 4)`, with M at least 1.
        length_penalty: when true, multiply by min(1, M / K).

    Returns:
        A float: for each ground-truth box, the best IoU over all predicted
        boxes, averaged over the M ground-truth boxes, then penalised. Zero
        when there are no predictions.
    """
    raise NotImplementedError("detection_reward")


def group_advantages(rewards, eps=1e-6):
    """GRPO's group-relative advantages along the last axis.

    Args:
        rewards: shape `(..., G)`, one group of G rewards per prompt.
        eps: added to the standard deviation to avoid dividing by zero.

    Returns:
        (rewards - group mean) / (group standard deviation + eps), the same
        shape. Use the population standard deviation (numpy's default).
    """
    raise NotImplementedError("group_advantages")


def gaussian_logprob(actions, mean, log_std):
    """Log-density of a diagonal Gaussian, summed over the last axis.

    Args:
        actions: a torch tensor of shape `(..., 4)`.
        mean: a torch tensor that broadcasts against `actions`.
        log_std: a torch tensor that broadcasts against `actions`.

    Returns:
        A torch tensor of shape `(...)`, differentiable with respect to
        `mean` and `log_std`.
    """
    raise NotImplementedError("gaussian_logprob")


def train_grounding(steps=300, batch=32, group=8, lr=0.01, seed=0):
    """Train a `BoxPolicy` with GRPO against the IoU verifier.

    Seed numpy with `np.random.default_rng(seed)` and torch with
    `torch.manual_seed(seed)`, then build the policy and an Adam optimiser.
    Each update: draw `batch` scenes, sample `group` boxes per image from the
    policy, reward each box by its IoU with the image's true box, turn the
    rewards into group advantages, and step on -(advantage * log-prob).mean().

    Returns:
        `(policy, history)`, where `history` is a list of `steps` floats: the
        mean IoU of the policy's mean box on each update's batch.
    """
    raise NotImplementedError("train_grounding")


if __name__ == "__main__":
    print("IoU of (0,0,2,2) and (1,1,3,3):", iou([0, 0, 2, 2], [1, 1, 3, 3]))
    print("Group advantages of [0, 0.5, 1, 1]:", np.round(group_advantages([0, 0.5, 1, 1]), 4))

    policy, history = train_grounding(steps=300, seed=0)
    test_images, test_boxes = make_scenes(200, np.random.default_rng(99))
    with torch.no_grad():
        mean, _ = policy(box_features(test_images))
    print("step   mean IoU of the policy's mean box (training batch)")
    for t in range(0, 300, 50):
        print(f"{t:4d}   {np.mean(history[t : t + 10]):.3f}")
    print(f"held-out mean IoU after training: {iou(mean.numpy(), test_boxes).mean():.3f}")
