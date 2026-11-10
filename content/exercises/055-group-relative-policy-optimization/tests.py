"""Checks for the group-relative policy optimization exercise."""

import itertools
import math

import numpy as np
import torch

from rlcheck import almost, check, close, fail, finite, has, shape

torch.set_num_threads(1)
D = torch.float64


def _t(x):
    return torch.tensor(x, dtype=D)


@check("rloo_advantages subtracts the mean of the other rewards in the group")
def test_rloo(sub):
    rloo = has(sub, "rloo_advantages")
    got = rloo(_t([1.0, 0.0, 0.0, 1.0]), 4)
    shape(got, (4,), label="rloo_advantages output")
    # Response 0: 1 - (0 + 0 + 1) / 3 = 0.6667.
    close(got, [2 / 3, -2 / 3, -2 / 3, 2 / 3], atol=1e-12, label="the group (1, 0, 0, 1)")

    rng = np.random.default_rng(0)
    r = rng.normal(size=(3, 5))
    want = np.array([[r[p, i] - np.delete(r[p], i).mean() for i in range(5)] for p in range(3)])
    close(rloo(_t(r.reshape(-1)), 5), want.reshape(-1), atol=1e-12,
          label="three groups of five, against an explicit leave-one-out loop")


@check("grpo_advantages matches the hand-computed group of eight, and a tied group gives zeros")
def test_grpo_advantages(sub):
    grpo = has(sub, "grpo_advantages")
    r = _t([1, 0, 0, 1, 1, 0, 0, 0])
    # Mean 0.375, population std sqrt(0.375 * 0.625) = 0.484123.
    std = math.sqrt(0.375 * 0.625)
    want = [(x - 0.375) / std for x in [1, 0, 0, 1, 1, 0, 0, 0]]
    close(grpo(r, 8), want, atol=1e-5, label="std-scaled advantages")
    almost(float(grpo(r, 8)[0]), 1.290994, tol=1e-5, label="a correct answer's advantage")
    close(grpo(r, 8, scale_by_std=False), [x - 0.375 for x in [1, 0, 0, 1, 1, 0, 0, 0]],
          atol=1e-12, label="mean-centred advantages (Dr. GRPO)")
    tied = grpo(_t([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]), 4)
    finite(tied.detach().numpy(), label="advantages of two tied groups")
    close(tied, np.zeros(8), atol=1e-12, label="two groups whose rewards all tie")


@check("Exact expectation over every group: RLOO is unbiased, mean-centred GRPO is (G-1)/G of it, std-scaled is not parallel")
def test_group_baselines_exactly(sub):
    rloo = has(sub, "rloo_advantages")
    grpo = has(sub, "grpo_advantages")
    probs = np.array([0.4, 0.3, 0.2, 0.1])
    reward = np.array([1.0, 0.0, 0.5, 2.0])
    true_grad = probs * (reward - probs @ reward)   # d E[r] / d logits of a softmax
    k = 3
    want = {"rloo": np.zeros(4), "mean": np.zeros(4), "std": np.zeros(4)}
    eye = np.eye(4)
    for group in itertools.product(range(4), repeat=k):
        weight = np.prod(probs[list(group)])
        r = _t(reward[list(group)])
        score = eye[list(group)] - probs            # d log pi(y_i) / d logits, (k, 4)
        for name, adv in (("rloo", rloo(r, k)), ("mean", grpo(r, k, scale_by_std=False)),
                          ("std", grpo(r, k))):
            want[name] += weight * (adv.detach().numpy()[:, None] * score).mean(axis=0)
    close(want["rloo"], true_grad, atol=1e-12, label="expected RLOO gradient against the true gradient")
    close(want["mean"], (k - 1) / k * true_grad, atol=1e-12,
          label="expected mean-centred GRPO gradient against (G-1)/G times the true gradient")
    g = want["std"]
    cosine = g @ true_grad / (np.linalg.norm(g) * np.linalg.norm(true_grad))
    if not 0.5 < cosine < 1 - 1e-4:
        fail(f"the std-scaled estimator should point roughly, not exactly, along the true "
             f"gradient; its cosine similarity is {cosine:.6f}")


@check("clipped_surrogate stops the gradient past 1 + eps_high for A > 0 and below 1 - eps_low for A < 0")
def test_clip_higher(sub):
    surrogate = has(sub, "clipped_surrogate")
    old = math.log(0.01)

    def grad_at(ratio, adv, eps_low, eps_high):
        new = torch.tensor([[old + math.log(ratio)]], dtype=D, requires_grad=True)
        value = surrogate(new, _t([[old]]), _t([adv]), eps_low, eps_high)
        shape(value.detach(), (1, 1), label="clipped_surrogate output")
        (g,) = torch.autograd.grad(value.sum(), new)
        return float(g)

    # d(rho * A) / d log pi_new = rho * A when unclipped, and 0 once clipped.
    cases = [
        (1.25, +1, 0.2, 0.2, 0.0, "rho 1.25, A +1, symmetric 0.2: clipped"),
        (1.25, +1, 0.2, 0.28, 1.25, "rho 1.25, A +1, clip-higher 0.28: still moving"),
        (1.30, +1, 0.2, 0.28, 0.0, "rho 1.30, A +1, clip-higher 0.28: clipped"),
        (0.75, -1, 0.2, 0.28, 0.0, "rho 0.75, A -1: clipped at 1 - eps_low"),
        (0.85, -1, 0.2, 0.28, -0.85, "rho 0.85, A -1: unclipped"),
        (1.50, -1, 0.2, 0.28, -1.5, "rho 1.50, A -1: the min keeps the unclipped, worse value"),
    ]
    for ratio, adv, lo, hi, want, label in cases:
        almost(grad_at(ratio, adv, lo, hi), want, tol=1e-9, label=label)


@check("aggregate: per-sequence mean, token mean, and constant normaliser on unequal lengths")
def test_aggregate(sub):
    aggregate = has(sub, "aggregate")
    per_token = _t([[1.0, 1.0, 9.0, 9.0, 9.0, 9.0], [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]])
    mask = _t([[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]])
    # Sequence means are 1 and -1, so their mean is 0.
    almost(float(aggregate(per_token, mask, "sequence_mean")), 0.0, tol=1e-12, label="sequence_mean")
    # Token mean: (2 - 6) / 8 = -0.5. The long response carries 6 of the 8 tokens.
    almost(float(aggregate(per_token, mask, "token_mean")), -0.5, tol=1e-12, label="token_mean")
    # Constant: ((2) / 10 + (-6) / 10) / 2 = -0.2.
    almost(float(aggregate(per_token, mask, "constant", max_tokens=10)), -0.2, tol=1e-12,
           label="constant with max_tokens 10")
    equal_mask = torch.ones(2, 6, dtype=D)
    almost(float(aggregate(per_token, equal_mask, "sequence_mean")),
           float(aggregate(per_token, equal_mask, "token_mean")), tol=1e-12,
           label="sequence_mean and token_mean at equal lengths")


@check("grpo_loss on-policy has gradient -(A - beta (1 - pi_ref/pi)) / (B |o_i|) per token")
def test_grpo_loss_gradient(sub):
    loss_fn = has(sub, "grpo_loss")
    torch.manual_seed(0)
    B, T, beta = 4, 5, 0.04
    logp = (-torch.rand(B, T, dtype=D) * 3).requires_grad_(True)
    ref = -torch.rand(B, T, dtype=D) * 3
    mask = _t([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0]])
    adv = _t([1.2, -0.4, 0.3, -1.1])
    loss = loss_fn(logp, logp.detach(), adv, mask, ref_logp=ref, beta=beta, mode="sequence_mean")
    (g,) = torch.autograd.grad(loss, logp)
    lengths = mask.sum(dim=1, keepdim=True)
    k3_grad = 1.0 - torch.exp(ref - logp.detach())
    want = -(adv[:, None] - beta * k3_grad) / (B * lengths) * mask
    close(g.numpy(), want.numpy(), atol=1e-12, label="gradient with respect to logp_new")
    # On-policy the ratio is 1, so without the KL the loss is minus the mean advantage.
    plain = loss_fn(logp.detach(), logp.detach(), adv, mask, mode="sequence_mean")
    almost(float(plain), -float(adv.mean()), tol=1e-12, label="on-policy loss value without KL")


@check("length_bias_toy: GRPO makes wrong answers longer and right ones shorter; Dr. GRPO leaves both at 4.5")
def test_length_bias(sub):
    toy = has(sub, "length_bias_toy")
    wrong, right = toy("sequence_mean", steps=300, lr=0.5)
    almost(wrong, 5.3868, tol=2e-3, label="GRPO: mean length when wrong")
    almost(right, 2.3607, tol=2e-3, label="GRPO: mean length when right")
    wrong, right = toy("constant", steps=300, lr=0.5)
    almost(wrong, 4.5, tol=1e-6, label="Dr. GRPO: mean length when wrong")
    almost(right, 4.5, tol=1e-6, label="Dr. GRPO: mean length when right")


@check("gspo_ratio is the geometric mean of the token ratios, over response tokens only")
def test_gspo_ratio(sub):
    gspo = has(sub, "gspo_ratio")
    # Wrong implementation 1, "summed_log_ratio": exp of the SUM of the masked log-ratios,
    #   which is the whole-sequence likelihood ratio and is not divided by length.
    # Wrong implementation 2, "padded_mean": divides by T, so padding counts as a token.
    torch.manual_seed(0)
    old = -torch.rand(3, 4, dtype=D) * 3
    # Token ratios: row 0 is (4, 1) with 2 padded slots, row 1 is (8, 1, 1, 1), row 2 is (3,).
    ratios = [[4.0, 1.0, 5.0, 5.0], [8.0, 1.0, 1.0, 1.0], [3.0, 5.0, 5.0, 5.0]]
    mask = _t([[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]])
    new = (old + torch.log(_t(ratios))).requires_grad_(True)
    got = gspo(new, old, mask)
    shape(got.detach(), (3,), label="gspo_ratio output")
    # Row 0: sqrt(4 * 1) = 2. Row 1: 8 ** (1 / 4) = 1.681793. Row 2: 3.
    close(got.detach(), [2.0, 8.0 ** 0.25, 3.0], atol=1e-12, label="unequal lengths with padding")

    # A response whose tokens all have ratio 1.2 has sequence ratio 1.2 at any length.
    for n in (1, 3, 7):
        lo = -torch.rand(1, 7, dtype=D)
        m = torch.zeros(1, 7, dtype=D)
        m[0, :n] = 1.0
        almost(float(gspo(lo + math.log(1.2), lo, m)[0]), 1.2, tol=1e-12,
               label=f"equal token ratios 1.2, length {n}")

    # Each valid token moves s_i by s_i / |o_i|, and padding does not move it at all.
    (g,) = torch.autograd.grad(got.sum(), new)
    want = torch.tensor([[1.0, 1.0, 0, 0], [8.0 ** 0.25 / 4] * 4,
                         [3.0, 0, 0, 0]], dtype=D)
    close(g.numpy(), want.numpy(), atol=1e-12, label="gradient of the sequence ratios")
