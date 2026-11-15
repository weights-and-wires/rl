"""Checks for the AI-feedback and self-improvement exercise."""

import itertools
import math

import numpy as np
import torch

from rlcheck import almost, check, close, equal, fail, finite_difference, has, shape

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

V, T = 8, 4
SEQS = np.array(list(itertools.product(range(V), repeat=T)))


def _policy(seed):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(T, V))
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def _success(p, target):
    probs = np.prod(p[np.arange(T), SEQS], axis=1)
    return float(np.sum(probs * (SEQS.sum(axis=1) == target)))


def _token_logps(logits, seqs):
    logp = torch.log_softmax(logits, dim=-1)
    return logp[torch.arange(seqs.shape[1]).expand_as(seqs), seqs]


def _pairs_batch(logp, prev):
    """Every (real, synthetic) pair of a 1-D distribution, flattened row-major."""
    n = len(logp)
    return (logp[:, None].expand(n, n).reshape(-1), logp[None, :].expand(n, n).reshape(-1),
            prev[:, None].expand(n, n).reshape(-1), prev[None, :].expand(n, n).reshape(-1))


@check("filtered_sft_loss averages the kept responses' NLL and ignores masked tokens")
def test_filtered_sft(sub):
    loss_fn = has(sub, "filtered_sft_loss")
    token_logps = torch.tensor([
        [-1.0, -1.0, -9.0],
        [-2.0, -1.0, -9.0],
        [-0.5, -1.0, -9.0],
        [-3.0, -1.0, -9.0],
    ])
    mask = torch.tensor([[1.0, 1.0, 0.0]] * 4)
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    # Kept sequence log-probs are -2 and -1.5, so the loss is 1.75.
    almost(float(loss_fn(token_logps, mask, rewards)), 1.75, 1e-12,
           label="the lesson's four-sample example")
    none = loss_fn(token_logps, mask, torch.zeros(4))
    almost(float(none), 0.0, 1e-12, label="no response kept")


@check("Expert iteration is RWR with a 0/1 reward: same loss up to 1/mean(r), same gradient direction")
def test_filtered_is_rwr(sub):
    sft = has(sub, "filtered_sft_loss")
    rwr = has(sub, "rwr_loss")
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(T, V, generator=g, requires_grad=True)
    seqs = torch.randint(0, V, (32, T), generator=g)
    mask = (torch.rand(32, T, generator=g) > 0.2).double()
    rewards = (torch.rand(32, generator=g) > 0.6).double()
    lp = _token_logps(logits, seqs)
    a = sft(lp, mask, rewards)
    b = rwr(lp, mask, rewards)
    almost(float(a), float(b) / float(rewards.mean()), 1e-12,
           label="filtered_sft_loss against rwr_loss / mean(r)")
    (ga,) = torch.autograd.grad(a, logits, retain_graph=True)
    (gb,) = torch.autograd.grad(b, logits)
    close(ga, gb / rewards.mean(), atol=1e-12, label="their gradients, rescaled")


@check("At theta = theta_old, the RWR gradient over pi_old equals minus the gradient of the success rate")
def test_rwr_is_policy_gradient(sub):
    rwr = has(sub, "rwr_loss")
    target = 14
    seqs = torch.as_tensor(SEQS)
    mask = torch.ones(seqs.shape, dtype=torch.float64)
    reward = torch.as_tensor((SEQS.sum(axis=1) == target).astype(float))
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(T, V, generator=g).requires_grad_(True)
    with torch.no_grad():
        p_old = _token_logps(logits, seqs).sum(dim=1).exp()
    # An exact expectation over pi_old: weight each response by B * pi_old * r.
    weights = len(seqs) * p_old * reward
    (g_rwr,) = torch.autograd.grad(rwr(_token_logps(logits, seqs), mask, weights), logits)
    success = (_token_logps(logits, seqs).sum(dim=1).exp() * reward).sum()
    (g_pg,) = torch.autograd.grad(success, logits)
    close(g_rwr, -g_pg, atol=1e-12, label="RWR gradient against -grad P(success)")


@check("em_step returns the per-step marginals of the policy conditioned on success")
def test_em_step(sub):
    em_step = has(sub, "em_step")
    p = _policy(2)
    for target in (5, 14, 20):
        got = np.asarray(em_step(p.copy(), target))
        shape(got, (T, V), label="em_step output")
        probs = np.prod(p[np.arange(T), SEQS], axis=1) * (SEQS.sum(axis=1) == target)
        probs = probs / probs.sum()
        want = np.stack([[probs[SEQS[:, t] == a].sum() for a in range(V)] for t in range(T)])
        close(got, want, atol=1e-12, label=f"marginals of pi(. | success), target {target}")


@check("Iterating em_step never lowers the success rate, and ten steps pass 0.99")
def test_em_monotone(sub):
    em_step = has(sub, "em_step")
    for seed, target in ((0, 14), (1, 24), (2, 5)):
        p = _policy(seed)
        rates = [_success(p, target)]
        for _ in range(10):
            p = np.asarray(em_step(p, target))
            rates.append(_success(p, target))
        if np.any(np.diff(rates) < -1e-12):
            fail(f"EM must never lower the success rate; seed {seed}, target {target} "
                 f"gave {np.round(rates, 4)}")
        if rates[-1] < 0.99:
            fail(f"after ten EM steps, success should pass 0.99; seed {seed}, "
                 f"target {target} reached {rates[-1]:.4f}")


@check("spin_loss is log 2 at theta = theta_t, equals DPO on (real, synthetic), and matches finite differences")
def test_spin_loss(sub):
    spin = has(sub, "spin_loss")
    g = torch.Generator().manual_seed(3)
    real, synth = -3 * torch.rand(6, generator=g), -3 * torch.rand(6, generator=g)
    prev_r, prev_s = -3 * torch.rand(6, generator=g), -3 * torch.rand(6, generator=g)
    got = spin(real, synth, real, synth, 0.7)
    shape(got, (6,), label="spin_loss output")
    close(got, torch.full((6,), math.log(2.0)), atol=1e-12, label="at theta = theta_t")

    almost(float(spin(torch.tensor([-2.0]), torch.tensor([-1.2]), torch.tensor([-2.5]),
                      torch.tensor([-1.0]), 1.0)[0]),
           math.log1p(math.exp(-0.7)), 1e-12, label="the lesson's worked example")

    lam = 0.4
    margin = lam * (real - prev_r) - lam * (synth - prev_s)
    dpo = -torch.nn.functional.logsigmoid(margin)
    close(spin(real, synth, prev_r, prev_s, lam), dpo, atol=1e-12,
          label="DPO with beta = lambda, chosen = real, rejected = synthetic")

    x0 = torch.cat([real, synth]).numpy().copy()

    def f(x):
        x = torch.as_tensor(x)
        return float(spin(x[:6], x[6:], prev_r, prev_s, lam).sum())

    x = torch.cat([real, synth]).clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(spin(x[:6], x[6:], prev_r, prev_s, lam).sum(), x)
    close(grad.numpy(), finite_difference(f, x0), atol=1e-7,
          label="gradient against finite differences")


@check("At theta = theta_t, the SPIN gradient on the logits is (lambda / 2)(p_t - p_data)")
def test_spin_first_gradient(sub):
    spin = has(sub, "spin_loss")
    g = torch.Generator().manual_seed(4)
    n, lam = 16, 0.8
    p_data = torch.softmax(1.5 * torch.randn(n, generator=g), dim=0)
    for p_t in (torch.softmax(torch.randn(n, generator=g), dim=0), p_data.clone()):
        theta = torch.log(p_t).clone().requires_grad_(True)
        prev = torch.log_softmax(theta.detach(), dim=0)
        logp = torch.log_softmax(theta, dim=0)
        losses = spin(*_pairs_batch(logp, prev), lam)
        weights = (p_data[:, None] * p_t[None, :]).reshape(-1)
        (grad,) = torch.autograd.grad((weights * losses).sum(), theta)
        close(grad, lam / 2 * (p_t - p_data), atol=1e-12,
              label="gradient of the exact SPIN objective at theta = theta_t")


@check("Five SPIN iterations bring the model within 0.01 of the data distribution")
def test_spin_iterations(sub):
    spin = has(sub, "spin_loss")
    g = torch.Generator().manual_seed(0)
    n = 16
    p_data = torch.softmax(1.5 * torch.randn(n, generator=g), dim=0)
    theta = torch.zeros(n)
    tvs = []
    for _ in range(5):
        prev = torch.log_softmax(theta, dim=0)
        weights = (p_data[:, None] * prev.exp()[None, :]).reshape(-1)
        theta = theta.clone().requires_grad_(True)
        for _ in range(100):
            losses = spin(*_pairs_batch(torch.log_softmax(theta, dim=0), prev), 1.0)
            (grad,) = torch.autograd.grad((weights * losses).sum(), theta)
            theta = (theta - grad).detach().requires_grad_(True)
        theta = theta.detach()
        tvs.append(0.5 * float((torch.softmax(theta, dim=0) - p_data).abs().sum()))
    if tvs[-1] > 0.01 or tvs[-1] >= tvs[0]:
        fail(f"total variation to p_data after each SPIN iteration: {np.round(tvs, 4)}. "
             "Expected it to fall below 0.01 by the fifth.")
