"""Checks for the capstone post-training exercise."""

import math

import numpy as np
import torch

from rlcheck import almost, check, close, fail, finite_difference, has, shape

torch.set_num_threads(1)
D = torch.float64


def _t(x):
    return torch.tensor(x, dtype=D)


def _layer(seed=0, out=5, inp=4, r=2, batch=3):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, inp, generator=g, dtype=D)
    W = torch.randn(out, inp, generator=g, dtype=D)
    A = torch.randn(r, inp, generator=g, dtype=D)
    B = torch.randn(out, r, generator=g, dtype=D)
    return x, W, A, B


@check("lora_linear equals the base layer when B is zero, and x @ (W + (alpha / r) B A).T otherwise")
def test_lora_linear(sub):
    lora_linear = has(sub, "lora_linear")
    x, W, A, B = _layer()
    got = lora_linear(x, W, A, torch.zeros_like(B), 8.0, 2)
    shape(got, (3, 5), label="lora_linear output")
    close(got, x @ W.T, atol=1e-12, label="at B = 0, the layer must equal the base layer")
    # Named wrong implementation: forgetting the alpha / r scale. At alpha = 8 and
    # r = 2 the scale is 4, so an unscaled adapter path is off by a factor of 4.
    want = x @ (W + (8.0 / 2) * B @ A).T
    close(lora_linear(x, W, A, B, 8.0, 2), want, atol=1e-10, label="alpha = 8, r = 2, B set")
    close(lora_linear(x[0], W, A, B, 8.0, 2), want[0], atol=1e-10, label="a single row")


@check("init_lora returns B as zeros and A as N(0, 1/in) draws, so the start equals the base model")
def test_init_lora(sub):
    init_lora = has(sub, "init_lora")
    lora_linear = has(sub, "lora_linear")
    A, B = init_lora(400, 30, 50, 0)
    shape(A, (50, 400), label="A")
    shape(B, (30, 50), label="B")
    # Named wrong implementations: B random with A zero, or both random. Either
    # way the adapter is not a zero update at the start, or not trainable.
    if float(torch.as_tensor(B).abs().max()) != 0.0:
        fail(f"B must start at exactly zero so the start equals the base; its largest entry is {float(torch.as_tensor(B).abs().max()):.4g}")
    std = float(torch.as_tensor(A).std())
    if not 0.85 / math.sqrt(400) < std < 1.15 / math.sqrt(400):
        fail(f"A should hold draws with standard deviation 1/sqrt(in) = {1 / math.sqrt(400):.4f}, got {std:.4f}")
    A2, _ = init_lora(400, 30, 50, 0)
    close(A, A2, atol=0.0, label="the same seed must give the same A")
    A3, _ = init_lora(400, 30, 50, 1)
    if torch.allclose(torch.as_tensor(A), torch.as_tensor(A3)):
        fail("different seeds should give different A")
    x, W, _, _ = _layer(seed=3, out=6, inp=7, r=3)
    A, B = init_lora(7, 6, 3, 5)
    close(lora_linear(x, W, A, B, 6.0, 3), x @ W.T, atol=0.0, label="the adapted layer at initialization")


@check("lora_param_count is r * (in + out) per layer, and matches the adapters init_lora builds")
def test_param_count(sub):
    lora_param_count = has(sub, "lora_param_count")
    init_lora = has(sub, "init_lora")
    got = lora_param_count([(1024, 1024)], 8)
    if got != 16384:
        fail(f"one 1024 x 1024 layer at rank 8 has 8 * 2048 = 16384 parameters, got {got}")
    got = lora_param_count([(64, 32), (32, 64), (128, 32)], 4)
    # 4 * (96 + 96 + 160) = 1408.
    if got != 1408:
        fail(f"three layers at rank 4 should have 1408 parameters, got {got}")
    built = 0
    for out, inp in [(64, 32), (32, 64), (128, 32)]:
        A, B = init_lora(inp, out, 4, 0)
        built += torch.as_tensor(A).numel() + torch.as_tensor(B).numel()
    if built != got:
        fail(f"init_lora builds {built} parameters but lora_param_count says {got}")
    ratio = lora_param_count([(1024, 1024)], 8) / (1024 * 1024)
    almost(ratio, 1 / 64, 1e-12, label="the fraction of one square layer's weights trained at rank 8")


@check("lora_grads matches finite differences for A and B, and dA is zero while B is zero")
def test_lora_grads(sub):
    lora_grads = has(sub, "lora_grads")
    x, W, A, B = _layer(seed=1)
    g = torch.Generator().manual_seed(9)
    y = torch.randn(3, 5, generator=g, dtype=D)
    alpha, r = 6.0, 2
    xn, Wn, yn = x.numpy(), W.numpy(), y.numpy()

    def loss_of(An, Bn):
        out = xn @ Wn.T + (alpha / r) * (xn @ An.T) @ Bn.T
        return 0.5 * np.sum((out - yn) ** 2)

    dA, dB = lora_grads(x, W, A, B, alpha, r, y)
    shape(dA, (2, 4), label="dA")
    shape(dB, (5, 2), label="dB")
    close(dA, finite_difference(lambda a: loss_of(a, B.numpy()), A.numpy()), atol=1e-6, rtol=1e-5,
          label="dA against finite differences")
    close(dB, finite_difference(lambda b: loss_of(A.numpy(), b), B.numpy()), atol=1e-6, rtol=1e-5,
          label="dB against finite differences")
    # The base is frozen: a W that requires gradients must receive none.
    Wg = W.clone().requires_grad_(True)
    lora_grads(x, Wg, A, B, alpha, r, y)
    if Wg.grad is not None and float(Wg.grad.abs().sum()) != 0.0:
        fail("lora_grads must not send a gradient into the base weight W")
    # At B = 0 the adapter output is zero whatever A is, so A gets no gradient yet.
    dA0, dB0 = lora_grads(x, W, A, torch.zeros_like(B), alpha, r, y)
    close(dA0, np.zeros((2, 4)), atol=1e-12, label="dA at B = 0")
    if float(torch.as_tensor(dB0).abs().max()) < 1e-3:
        fail("dB at B = 0 should be nonzero: the first step moves B")


@check("train_adapter lowers a loss by moving only A and B, and leaves W and its inputs unchanged")
def test_train_adapter(sub):
    train_adapter = has(sub, "train_adapter")
    init_lora = has(sub, "init_lora")
    g = torch.Generator().manual_seed(0)
    W = torch.randn(4, 4, generator=g, dtype=D) * 0.5
    W0 = W.clone()
    # A target the rank-2 adapter can reach exactly.
    At = torch.randn(2, 4, generator=g, dtype=D)
    Bt = torch.randn(4, 2, generator=g, dtype=D)
    target = W.T + (4.0 / 2) * (torch.eye(4, dtype=D) @ At.T) @ Bt.T
    A0, B0 = init_lora(4, 4, 2, 0)
    A0c, B0c = A0.clone(), B0.clone()

    def loss_fn(table):
        return 0.5 * ((table - target) ** 2).sum()

    start = float(0.5 * ((W.T - target) ** 2).sum())
    A, B = train_adapter(loss_fn, W, A0, B0, 4.0, 2, 400, 0.05)
    close(W, W0, atol=0.0, label="the frozen weight W")
    close(A0, A0c, atol=0.0, label="the A that was passed in")
    close(B0, B0c, atol=0.0, label="the B that was passed in")
    table = W.T + 2.0 * (torch.eye(4, dtype=D) @ A.T) @ B.T
    end = float(loss_fn(table))
    if not end < 0.01 * start:
        fail(f"400 steps should cut the loss to under 1% of its start {start:.3f}, got {end:.4f}")
    if float(torch.as_tensor(B).abs().max()) == 0.0:
        fail("B never moved")


@check("sequence_kl is zero for equal distributions and matches the hand value")
def test_sequence_kl(sub):
    sequence_kl = has(sub, "sequence_kl")
    p = _t([0.5, 0.5]).log()
    q = _t([0.25, 0.75]).log()
    got = sequence_kl(p, q)
    # 0.5 ln(0.5 / 0.25) + 0.5 ln(0.5 / 0.75) = 0.5 ln 2 + 0.5 ln(2/3) = 0.143841.
    almost(got, 0.5 * math.log(2) + 0.5 * math.log(2 / 3), 1e-12, label="KL((.5, .5) || (.25, .75))")
    almost(sequence_kl(p, p), 0.0, 1e-12, label="KL of a distribution with itself")
    rows = torch.stack([p, _t([0.9, 0.1]).log()])
    refs = torch.stack([q, _t([0.5, 0.5]).log()])
    want = (got + 0.9 * math.log(1.8) + 0.1 * math.log(0.2)) / 2
    almost(sequence_kl(rows, refs), want, 1e-12, label="the mean over two rows")


@check("pass_at_k is the unbiased estimator, not the plug-in 1 - (1 - c/n)^k")
def test_pass_at_k(sub):
    pass_at_k = has(sub, "pass_at_k")
    # n = 5, c = 2, k = 2: 1 - C(3, 2) / C(5, 2) = 1 - 3 / 10 = 0.7.
    # Named wrong implementation: the plug-in 1 - (1 - 2/5)^2 = 0.64.
    almost(pass_at_k(5, 2, 2), 0.7, 1e-12, label="pass@2 with 2 of 5 correct")
    almost(pass_at_k(5, 2, 1), 0.4, 1e-12, label="pass@1 equals c / n")
    almost(pass_at_k(8, 0, 4), 0.0, 1e-12, label="no correct samples")
    almost(pass_at_k(8, 6, 4), 1.0, 1e-12, label="fewer than k wrong samples")
    got = pass_at_k(8, np.array([0, 1, 4, 8]), 4)
    # c = 1: 1 - C(7, 4) / C(8, 4) = 1 - 35 / 70 = 0.5. c = 4: 1 - 1 / 70.
    close(got, [0.0, 0.5, 1 - 1 / 70, 1.0], atol=1e-12, label="a vector of counts")


@check("judge_win_rate averages both orders, and a position-biased judge scores 0.5 with consistency 0")
def test_judge_win_rate(sub):
    judge_win_rate = has(sub, "judge_win_rate")
    # A wins in both orders on 3 pairs, loses in both on 1, and ties on 1.
    # With B shown first, "B wins" is +1 in the verdict, so A winning is -1.
    ab = [1, 1, 1, -1, 0]
    ba = [-1, -1, -1, 1, 0]
    win, cons = judge_win_rate(ab, ba)
    almost(win, 3.5 / 5, 1e-12, label="win rate, ties counting half")
    almost(cons, 1.0, 1e-12, label="swap consistency of a consistent judge")
    # Named wrong implementation: scoring one order only. A judge that always
    # picks the first response gives A a win rate of 1.0 in that single order.
    win, cons = judge_win_rate([1, 1, 1, 1], [1, 1, 1, 1])
    almost(win, 0.5, 1e-12, label="win rate of a judge that always picks the first position")
    almost(cons, 0.0, 1e-12, label="swap consistency of that judge")
    # Pair 0 and pair 2 flip with the order, which is position bias. Pair 1 is a
    # consistent win for A and pair 3 a consistent tie.
    win, cons = judge_win_rate([1, 1, -1, 0], [1, -1, -1, 0])
    almost(cons, 0.5, 1e-12, label="two of four pairs flip")
    almost(win, (0.5 + 1.0 + 0.5 + 0.5) / 4, 1e-12, label="win rate with flips")


@check("the toy pipeline starts equal to the base, SFT lowers the demonstration NLL, and W stays frozen")
def test_pipeline_sft(sub):
    run = has(sub, "run_toy_pipeline")
    for seed in range(5):
        out = run(seed)
        if out["init_gap"] != 0.0:
            fail(f"seed {seed}: at initialization the adapted model must equal the base, gap {out['init_gap']:.3g}")
        drop = out["sft_nll_before"] - out["sft_nll_after"]
        if not drop > 0.1:
            fail(f"seed {seed}: SFT should lower the demonstration NLL by more than 0.1, "
                 f"went {out['sft_nll_before']:.3f} -> {out['sft_nll_after']:.3f}")
        if out["w_drift"] != 0.0:
            fail(f"seed {seed}: the frozen base weight changed by {out['w_drift']:.3g}")


@check("DPO after SFT raises the chosen-over-rejected margin and the reward, and the KL to SFT stays under 2 nats")
def test_pipeline_dpo(sub):
    run = has(sub, "run_toy_pipeline")
    for seed in range(5):
        out = run(seed)
        gain = out["margin_after"] - out["margin_before"]
        if not gain > 0.4:
            fail(f"seed {seed}: the held-out margin should rise by more than 0.4, "
                 f"went {out['margin_before']:.3f} -> {out['margin_after']:.3f}")
        if not out["reward_after"] > out["reward_before"] + 0.03:
            fail(f"seed {seed}: expected reward should rise, "
                 f"went {out['reward_before']:.3f} -> {out['reward_after']:.3f}")
        if not 0.0 < out["kl_to_sft"] < 2.0:
            fail(f"seed {seed}: the KL to the SFT policy should be in (0, 2) nats, got {out['kl_to_sft']:.3f}")
