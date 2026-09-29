"""Checks for the vision-language model exercise."""

import numpy as np
import torch

from rlcheck import almost, check, close, fail, has, shape

torch.set_num_threads(1)


def _iou_reference(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    wa = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    wb = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    iw = max(min(a[2], b[2]) - max(a[0], b[0]), 0.0)
    ih = max(min(a[3], b[3]) - max(a[1], b[1]), 0.0)
    inter = iw * ih
    union = wa + wb - inter
    return inter / union if union > 0 else 0.0


@check("iou gives 1 for identical boxes, 0 for disjoint ones, and 1/7 for the offset squares")
def test_iou_known_values(sub):
    iou = has(sub, "iou")
    almost(iou([0, 0, 2, 2], [0, 0, 2, 2]), 1.0, 1e-12, label="identical boxes")
    almost(iou([0, 0, 1, 1], [2, 2, 3, 3]), 0.0, 1e-12, label="disjoint boxes")
    almost(iou([0, 0, 1, 1], [1, 0, 2, 1]), 0.0, 1e-12, label="boxes that share only an edge")
    almost(iou([0, 0, 2, 2], [1, 1, 3, 3]), 1 / 7, 1e-12, label="(0,0,2,2) against (1,1,3,3)")
    almost(iou([0, 0, 4, 2], [1, 0, 3, 2]), 0.5, 1e-12, label="a box inside a box twice its area")


@check("iou broadcasts over batches, is symmetric, and returns 0 for degenerate boxes")
def test_iou_batched_and_degenerate(sub):
    iou = has(sub, "iou")
    rng = np.random.default_rng(0)
    corners = rng.uniform(0, 1, size=(6, 5, 2, 2))
    a = np.concatenate([corners[..., 0, :].min(-1, keepdims=True), corners[..., 1, :].min(-1, keepdims=True),
                        corners[..., 0, :].max(-1, keepdims=True), corners[..., 1, :].max(-1, keepdims=True)], -1)
    b = rng.uniform(0, 1, size=(5, 4))
    b = np.concatenate([np.minimum(b[:, :2], b[:, 2:]), np.maximum(b[:, :2], b[:, 2:])], -1)
    got = np.asarray(iou(a, b))
    shape(got, (6, 5), label="iou of (6, 5, 4) boxes against (5, 4) boxes")
    want = np.array([[_iou_reference(a[i, j], b[j]) for j in range(5)] for i in range(6)])
    close(got, want, atol=1e-12, label="batched iou")
    close(np.asarray(iou(b, a)), want, atol=1e-12, label="iou with the arguments swapped")

    degenerate = np.array([
        [0.5, 0.5, 0.5, 0.5],  # a point
        [0.2, 0.2, 0.2, 0.9],  # zero width
        [0.8, 0.8, 0.2, 0.2],  # corners swapped: far edges before near edges
    ])
    target = np.array([0.1, 0.1, 0.9, 0.9])
    got = np.asarray(iou(degenerate, target))
    if not np.all(np.isfinite(got)):
        fail(f"iou returned NaN or infinity for degenerate boxes: {got}")
    close(got, np.zeros(3), atol=1e-12, label="iou of degenerate boxes against a real box")
    almost(float(iou([0.3, 0.3, 0.3, 0.3], [0.3, 0.3, 0.3, 0.3])), 0.0, 1e-12,
           label="two identical points, whose union has zero area")


@check("dense_logprob reproduces the RLHF-V example: -0.525 with gamma 5, the plain mean with gamma 1")
def test_dense_logprob(sub):
    dense_logprob = has(sub, "dense_logprob")
    # 20 unchanged tokens summing to -6 and 4 corrected tokens summing to -3.
    logps = np.concatenate([np.full(20, -0.3), np.full(4, -0.75)])
    corrected = np.concatenate([np.zeros(20, bool), np.ones(4, bool)])
    almost(dense_logprob(logps, corrected, 5.0), -0.525, 1e-12, label="gamma = 5")
    almost(dense_logprob(logps, corrected, 1.0), -0.375, 1e-12, label="gamma = 1")
    rng = np.random.default_rng(1)
    batch = -rng.uniform(0, 2, size=(3, 7))
    mask = rng.random((3, 7)) < 0.4
    w = np.where(mask, 2.5, 1.0)
    close(dense_logprob(batch, mask, 2.5), (w * batch).sum(1) / w.sum(1), atol=1e-12,
          label="a batch of three responses")


@check("mdpo_loss matches hand values, and its conditional term is exactly log 2 for an image-blind policy")
def test_mdpo_loss_values(sub):
    mdpo_loss = has(sub, "mdpo_loss")
    # The worked example: chosen +0.3, rejected -0.2, chosen on the cropped image +0.3.
    dpo, copo, anchor = mdpo_loss(np.array([0.3]), np.array([-0.2]), np.array([0.3]), beta=1.0, delta=0.0)
    almost(dpo, np.log1p(np.exp(-0.5)), 1e-9, label="DPO term")
    almost(copo, np.log(2.0), 1e-9, label="conditional term when the image does not matter")
    almost(anchor, np.log1p(np.exp(-0.3)), 1e-9, label="anchor term with delta = 0")
    almost(anchor, 0.554355, 1e-6, label="anchor term, four decimals")

    rng = np.random.default_rng(2)
    cw, cl, cc = rng.normal(size=(3, 5))
    beta, delta = 0.1, 0.05
    dpo, copo, anchor = mdpo_loss(cw, cl, cc, beta=beta, delta=delta)
    nls = lambda z: np.log1p(np.exp(-z))
    almost(dpo, np.mean(nls(beta * (cw - cl))), 1e-9, label="DPO term, batch of five")
    almost(copo, np.mean(nls(beta * (cw - cc))), 1e-9, label="conditional term, batch of five")
    almost(anchor, np.mean(nls(beta * cw - delta)), 1e-9, label="anchor term, batch of five")

    # Stability: a huge margin must not overflow.
    dpo, _, _ = mdpo_loss(np.array([-800.0]), np.array([800.0]), np.array([0.0]), beta=1.0)
    almost(dpo, 1600.0, 1e-6, label="DPO term for a confidently wrong pair")


@check("detection_reward is hacked by extra boxes without the length penalty and repaired with it")
def test_detection_reward(sub):
    detection_reward = has(sub, "detection_reward")
    gt = np.array([[0.1, 0.1, 0.4, 0.4]])
    junk = np.array([[0.5, 0.5, 0.6, 0.6], [0.7, 0.1, 0.9, 0.2], [0.0, 0.8, 0.2, 1.0]])
    exact = gt.copy()
    spray = np.concatenate([exact, junk])
    almost(detection_reward(exact, gt, length_penalty=True), 1.0, 1e-12, label="one exact box")
    almost(detection_reward(spray, gt, length_penalty=False), 1.0, 1e-12,
           label="exact box plus three junk boxes, no penalty")
    almost(detection_reward(spray, gt, length_penalty=True), 0.25, 1e-12,
           label="exact box plus three junk boxes, with min(1, 1/4)")
    # Two targets, three predictions: best matches 1 and 1/7, penalty 2/3.
    gt2 = np.array([[0, 0, 2, 2], [10, 10, 12, 12]], float)
    pred = np.array([[0, 0, 2, 2], [11, 11, 13, 13], [5, 5, 6, 6]], float)
    almost(detection_reward(pred, gt2, length_penalty=False), (1 + 1 / 7) / 2, 1e-12,
           label="two targets, recall")
    almost(detection_reward(pred, gt2, length_penalty=True), (2 / 3) * (1 + 1 / 7) / 2, 1e-12,
           label="two targets, penalised")
    almost(detection_reward(np.zeros((0, 4)), gt2), 0.0, 1e-12, label="no predictions at all")


@check("group_advantages standardises each group and gives exactly zero for a group of equal rewards")
def test_group_advantages(sub):
    group_advantages = has(sub, "group_advantages")
    got = np.asarray(group_advantages(np.array([0.0, 0.5, 1.0, 1.0])))
    r = np.array([0.0, 0.5, 1.0, 1.0])
    close(got, (r - r.mean()) / (r.std() + 1e-6), atol=1e-9, label="one group of four")
    rng = np.random.default_rng(3)
    R = rng.uniform(size=(5, 8))
    A = np.asarray(group_advantages(R))
    shape(A, (5, 8), label="advantages for 5 groups of 8")
    close(A.mean(axis=1), np.zeros(5), atol=1e-9, label="mean advantage in each group")
    close(A.std(axis=1), np.ones(5), atol=1e-4, label="spread of advantages in each group")
    close(np.asarray(group_advantages(np.full(8, 0.3))), np.zeros(8), atol=1e-12,
          label="a group whose rewards are all equal")


@check("gaussian_logprob matches torch.distributions.Normal summed over the four coordinates")
def test_gaussian_logprob(sub):
    gaussian_logprob = has(sub, "gaussian_logprob")
    torch.manual_seed(0)
    mean = torch.rand(6, 4)
    log_std = torch.randn(6, 4) * 0.3 - 2.0
    actions = mean + torch.randn(6, 4) * 0.2
    got = gaussian_logprob(actions, mean, log_std)
    want = torch.distributions.Normal(mean, log_std.exp()).log_prob(actions).sum(-1)
    shape(got, (6,), label="one log-probability per box")
    close(got.detach().numpy(), want.numpy(), rtol=1e-5, atol=1e-5, label="log-probability")
    # The gradient must flow to the mean, or training cannot work.
    mean.requires_grad_(True)
    gaussian_logprob(actions, mean, log_std).sum().backward()
    if mean.grad is None or not torch.all(torch.isfinite(mean.grad)):
        fail("gaussian_logprob must be differentiable with respect to the mean")
    close(mean.grad.numpy(), ((actions - mean) / log_std.exp() ** 2).detach().numpy(),
          rtol=1e-4, atol=1e-4, label="gradient with respect to the mean, (a - mu) / sigma^2")


@check("train_grounding lifts held-out IoU from below 0.1 to above 0.6 in 300 GRPO updates")
def test_train_grounding(sub):
    train_grounding = has(sub, "train_grounding")
    make_scenes = has(sub, "make_scenes")
    box_features = has(sub, "box_features")
    BoxPolicy = has(sub, "BoxPolicy")
    iou = has(sub, "iou")
    images, boxes = make_scenes(200, np.random.default_rng(99))
    feats = box_features(images)

    torch.manual_seed(0)
    fresh = BoxPolicy()
    with torch.no_grad():
        before = float(np.mean([_iou_reference(p, g) for p, g in zip(fresh(feats)[0].numpy(), boxes)]))
    policy, history = train_grounding(steps=300, batch=32, group=8, lr=0.01, seed=0)
    if len(history) != 300:
        fail(f"train_grounding should return one history entry per update, 300, got {len(history)}")
    with torch.no_grad():
        pred = policy(feats)[0].numpy()
    after = float(np.mean([_iou_reference(p, g) for p, g in zip(pred, boxes)]))
    if before >= 0.1:
        fail(f"an untrained policy should score below 0.1 held-out IoU, got {before:.3f}")
    if after <= 0.6:
        fail(f"after 300 updates the held-out IoU should exceed 0.6, got {after:.3f} (reference: about 0.73)")
