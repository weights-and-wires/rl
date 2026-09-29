"""Reference solution for the vision-language model exercise."""

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


def dense_logprob(token_logps, corrected, gamma=5.0):
    token_logps = np.asarray(token_logps, dtype=float)
    weights = np.where(np.asarray(corrected, dtype=bool), gamma, 1.0)
    return np.sum(weights * token_logps, axis=-1) / np.sum(weights, axis=-1)


def _neg_log_sigmoid(z):
    # -log sigmoid(z) = log(1 + exp(-z)), stable for large |z|.
    return np.logaddexp(0.0, -np.asarray(z, dtype=float))


def mdpo_loss(chosen_logratio, rejected_logratio, chosen_corrupt_logratio, beta, delta=0.0):
    chosen = np.asarray(chosen_logratio, dtype=float)
    rejected = np.asarray(rejected_logratio, dtype=float)
    corrupt = np.asarray(chosen_corrupt_logratio, dtype=float)
    dpo = np.mean(_neg_log_sigmoid(beta * chosen - beta * rejected))
    copo = np.mean(_neg_log_sigmoid(beta * chosen - beta * corrupt))
    anchor = np.mean(_neg_log_sigmoid(beta * chosen - delta))
    return float(dpo), float(copo), float(anchor)


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


def group_advantages(rewards, eps=1e-6):
    rewards = np.asarray(rewards, dtype=float)
    mean = rewards.mean(axis=-1, keepdims=True)
    std = rewards.std(axis=-1, keepdims=True)
    return (rewards - mean) / (std + eps)


def gaussian_logprob(actions, mean, log_std):
    z = (actions - mean) / torch.exp(log_std)
    per_dim = -0.5 * z**2 - log_std - 0.5 * np.log(2 * np.pi)
    return per_dim.sum(dim=-1)


def train_grounding(steps=300, batch=32, group=8, lr=0.01, seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    policy = BoxPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    history = []
    for _ in range(steps):
        images, gt = make_scenes(batch, rng)
        mean, log_std = policy(box_features(images))
        with torch.no_grad():
            noise = torch.randn(batch, group, 4)
            boxes = mean[:, None, :] + torch.exp(log_std)[:, None, :] * noise
        # The verifier: IoU against the ground-truth box, one group per image.
        rewards = iou(boxes.numpy(), gt[:, None, :])
        adv = torch.tensor(group_advantages(rewards), dtype=torch.float32)
        logp = gaussian_logprob(boxes, mean[:, None, :], log_std[:, None, :])
        loss = -(adv * logp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(iou(mean.detach().numpy(), gt).mean()))
    return policy, history


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
