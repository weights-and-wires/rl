"""Checks for the reasoning-with-verifiable-rewards exercise."""

import itertools
import math
from fractions import Fraction

import numpy as np
import torch

from rlcheck import almost, check, close, fail, finite, has, shape

torch.set_num_threads(1)
D = torch.float64


def _exact_pass(n, c, k):
    return float(1 - Fraction(math.comb(n - c, k), math.comb(n, k)))


@check("pass_at_k matches 1 - C(n-c, k) / C(n, k) exactly, including the worked example")
def test_pass_at_k_exact(sub):
    pass_at_k = has(sub, "pass_at_k")
    # n = 10, c = 3, k = 5: 1 - 21 / 252 = 0.916667.
    almost(pass_at_k(10, 3, 5), 1 - 21 / 252, tol=1e-12, label="worked example")
    almost(pass_at_k(10, 0, 5), 0.0, tol=1e-12, label="no correct samples")
    almost(pass_at_k(10, 6, 5), 1.0, tol=1e-12, label="fewer than k wrong samples")
    for n in range(1, 13):
        for c in range(0, n + 1):
            for k in range(1, n + 1):
                almost(pass_at_k(n, c, k), _exact_pass(n, c, k), tol=1e-12,
                       label=f"n={n}, c={c}, k={k}")


@check("pass_at_k stays accurate where the binomial coefficients overflow a float")
def test_pass_at_k_stable(sub):
    pass_at_k = has(sub, "pass_at_k")
    # C(5000, 1000) has over a thousand digits; a float version overflows.
    for c, k in ((10, 1000), (3, 100), (1, 1)):
        got = pass_at_k(5000, c, k)
        finite(got, label=f"n=5000, c={c}, k={k}")
        almost(got, _exact_pass(5000, c, k), tol=1e-10, label=f"n=5000, c={c}, k={k}")


@check("answer_reward pays only a well-formed, exactly correct <answer> after </think>")
def test_answer_reward(sub):
    reward = has(sub, "answer_reward")
    cases = [
        (" 6 * 7 = 42 </think> <answer> 42 </answer>", 42, 1.0, "a correct answer"),
        (" 6 * 7 = 41 </think> <answer>41</answer>", 42, 0.0, "a wrong answer"),
        (" think </think>\n<answer>-3</answer>", -3, 1.0, "a negative answer"),
        (" <answer>42</answer>", 42, 0.0, "no </think> before the answer"),
        (" </think> <answer>42", 42, 0.0, "an unclosed answer tag"),
        (" </think> <answer> forty-two </answer>", 42, 0.0, "a number in words"),
        (" </think> <answer>42.0</answer>", 42, 0.0, "not an integer"),
        (" </think> <answer>42 or 43</answer>", 42, 0.0, "two numbers"),
        (" the answer is 42", 42, 0.0, "no tags at all"),
    ]
    for text, target, want, label in cases:
        got = reward(text, target)
        if float(got) != want:
            fail(f"{label}: answer_reward({text!r}, {target}) returned {got}, expected {want}")
    # The GPU run closes its reasoning with a plain-text tag instead.
    tagged = [
        (" 6 * 7 = 42 </reasoning> <answer>42</answer>", 1.0, "a correct answer after </reasoning>"),
        (" 6 * 7 = 42 </think> <answer>42</answer>", 0.0, "</think> when the close tag is </reasoning>"),
    ]
    for text, want, label in tagged:
        got = reward(text, 42, close_tag="</reasoning>")
        if float(got) != want:
            fail(f"{label}: answer_reward({text!r}, 42, close_tag='</reasoning>') "
                 f"returned {got}, expected {want}")


@check("pg_loss has gradient -A_i / (total tokens) on each unmasked token")
def test_pg_loss(sub):
    loss_fn = has(sub, "pg_loss")
    torch.manual_seed(0)
    logp = (-torch.rand(3, 4, dtype=D) * 2).requires_grad_(True)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=D)
    adv = torch.tensor([0.5, -1.0, 2.0], dtype=D)
    loss = loss_fn(logp, mask, adv)
    (g,) = torch.autograd.grad(loss, logp)
    close(g.numpy(), (-adv[:, None] * mask / 7.0).numpy(), atol=1e-12, label="gradient of pg_loss")
    padded = logp.detach().clone()
    padded[mask == 0] = float("-inf")
    almost(float(loss_fn(padded, mask, adv)), float(loss), tol=1e-12,
           label="pg_loss with -inf in the padding")


@check("token_entropy is log V for uniform logits, 0 for a sure token, ignores -inf and masked positions")
def test_token_entropy(sub):
    entropy = has(sub, "token_entropy")
    logits = torch.zeros(2, 3, 8, dtype=D)
    logits[1, :, :] = float("-inf")
    logits[1, :, 2] = 0.0                 # row 1 is a sure token
    logits[0, 2, 4:] = float("-inf")      # position (0, 2) is uniform over 4 tokens
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=D)
    got = float(entropy(logits, mask))
    # Unmasked: two positions at log 8 and three at 0, so the mean is 2 log 8 / 5.
    almost(got, 2 * math.log(8) / 5, tol=1e-12, label="mean entropy over the mask")
    mask2 = torch.tensor([[0, 0, 1], [0, 0, 0]], dtype=D)
    almost(float(entropy(logits, mask2)), math.log(4), tol=1e-12, label="uniform over four allowed tokens")


@check("entropy_change_estimate matches the exact entropy change to first order")
def test_entropy_change(sub):
    estimate = has(sub, "entropy_change_estimate")

    def exact_h(z):
        logp = torch.log_softmax(z, -1)
        return -(logp.exp() * logp).sum(-1)

    # The lesson's example: pi = (0.7, 0.2, 0.1), natural-gradient step 0.1 * A.
    z = torch.log(torch.tensor([0.7, 0.2, 0.1], dtype=D))
    for adv in ((1.0, -1.0, 0.0), (-1.0, 1.0, 0.0)):
        dz = 0.1 * torch.tensor(adv, dtype=D)
        est = float(estimate(z, dz))
        exact = float(exact_h(z + dz) - exact_h(z))
        if abs(est - exact) > 0.005:
            fail(f"for A = {adv}, the estimate {est:.4f} and the exact change {exact:.4f} "
                 "should agree within 0.005")
    almost(float(estimate(z, 0.1 * torch.tensor([1.0, -1.0, 0.0], dtype=D))), -0.0473, tol=5e-4,
           label="estimate for A = (+1, -1, 0)")

    torch.manual_seed(0)
    z = torch.randn(5, 8, dtype=D)
    dz = torch.randn(5, 8, dtype=D)
    dz = 1e-3 * dz / dz.norm(dim=-1, keepdim=True)
    est = estimate(z, dz)
    shape(est.detach(), (5,), label="one estimate per row")
    exact = exact_h(z + dz) - exact_h(z)
    rel = ((est - exact).abs() / exact.abs()).max()
    if not float(rel) < 1e-2:
        fail(f"on random logits with a step of norm 1e-3, the relative error should be "
             f"below 1e-2; the largest was {float(rel):.3g}")


@check("train_rlvr_toy with SGD: pass@1 and reward rise, entropy collapses, and pass@64 falls below the base")
def test_rlvr_sgd(sub):
    train = has(sub, "train_rlvr_toy")
    hist = train(steps=200, group_size=8, lr=1.0, optimizer="sgd", seed=0, ks=(1, 8, 64))
    pk = np.asarray(hist["pass@k"])
    ent = np.asarray(hist["entropy"])
    shape(pk, (201, 3), label="pass@k history")
    shape(ent, (200,), label="entropy history")
    almost(pk[0, 0], 0.2015, tol=1e-3, label="the base model's pass@1")
    if not pk[-1, 0] > 0.45:
        fail(f"pass@1 should rise from {pk[0, 0]:.3f} to above 0.45; it reached {pk[-1, 0]:.3f}")
    if not ent[-1] < 0.6 * ent[0]:
        fail(f"entropy should fall below 60% of its start, {ent[0]:.3f}; it ended at {ent[-1]:.3f}")
    if not pk[-1, 2] < pk[0, 2] - 0.15:
        fail(f"pass@64 should fall well below the base model's {pk[0, 2]:.3f} as the habit "
             f"sharpens; it ended at {pk[-1, 2]:.3f}")
    if not pk[-1, 1] < pk[0, 1]:
        fail(f"pass@8 should already favour the base model: base {pk[0, 1]:.3f}, "
             f"trained {pk[-1, 1]:.3f}")


@check("train_rlvr_toy with Adam: the rare correct paths get learned, and pass@64 no longer falls")
def test_rlvr_adam(sub):
    train = has(sub, "train_rlvr_toy")
    hist = train(steps=200, group_size=8, lr=0.1, optimizer="adam", seed=0, ks=(1, 64))
    pk = np.asarray(hist["pass@k"])
    if not pk[-1, 0] > 0.9:
        fail(f"with Adam, pass@1 should pass 0.9; it reached {pk[-1, 0]:.3f}")
    if not pk[-1, 1] > pk[0, 1] - 1e-3:
        fail(f"with Adam, pass@64 should not fall: base {pk[0, 1]:.4f}, trained {pk[-1, 1]:.4f}")
