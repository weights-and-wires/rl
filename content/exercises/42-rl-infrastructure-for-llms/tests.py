"""Checks for the RL-infrastructure exercise."""

import itertools
import math

import torch

from rlcheck import almost, check, close, fail, has, shape

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

MU = torch.tensor([0.1, 0.1, 0.8])
PI = torch.tensor([0.1, 0.3, 0.6])
F = torch.tensor([1.0, 1.0, 0.0])

# A small autoregressive toy: V = 4 tokens, T = 3, and a bigram policy whose
# logits are indexed by (step, previous token), with previous token V at t = 0.
V, T = 4, 3
SEQS = torch.tensor(list(itertools.product(range(V), repeat=T)))
PREV = torch.cat([torch.full((len(SEQS), 1), V), SEQS[:, :-1]], dim=1)
STEPS = torch.arange(T).expand_as(SEQS)
REWARD = (SEQS == 3).sum(dim=1).double() + (SEQS[:, 0] == SEQS[:, 2]).double()


def _token_logps(theta):
    return torch.log_softmax(theta, dim=-1)[STEPS, PREV, SEQS]


def _true_gradient(theta):
    theta = theta.clone().requires_grad_(True)
    value = (_token_logps(theta).sum(dim=1).exp() * REWARD).sum()
    (grad,) = torch.autograd.grad(value, theta)
    return grad


def _expected_gradient(loss_fn, theta, theta_behavior, **kw):
    """E_{y ~ mu}[gradient of the reader's loss], exactly, over all 64 sequences.

    The loss is linear in the advantages, so scaling each sequence's
    advantage by B * mu(y) turns the batch mean into an expectation under mu.
    """
    with torch.no_grad():
        logp_mu = _token_logps(theta_behavior)
        mu = logp_mu.sum(dim=1).exp()
    theta = theta.clone().requires_grad_(True)
    adv = len(SEQS) * mu * REWARD
    loss = loss_fn(_token_logps(theta), logp_mu, torch.ones(SEQS.shape), adv, **kw)
    (grad,) = torch.autograd.grad(loss, theta)
    return -grad


def _cos(a, b):
    return float((a * b).sum() / (a.norm() * b.norm()))


@check("tis_weights and mis_weights reproduce the lesson's engine example")
def test_weights(sub):
    tis = has(sub, "tis_weights")
    mis = has(sub, "mis_weights")
    ratio = torch.tensor([0.08 / 0.02, 0.08 / 0.01, 1.0, 0.3])
    close(tis(ratio, 2.0), torch.tensor([2.0, 2.0, 1.0, 0.3]), atol=1e-12,
          label="TIS with C = 2")
    close(mis(ratio, 0.5, 5.0), torch.tensor([4.0, 0.0, 1.0, 0.0]), atol=1e-12,
          label="IcePop's band [0.5, 5]")


@check("estimator_bias_variance matches the lesson's table for no correction, IS, and TIS")
def test_bias_variance_table(sub):
    bv = has(sub, "estimator_bias_variance")
    tis = has(sub, "tis_weights")
    rho = PI / MU
    rows = {
        "no correction": (torch.ones(3), -0.2, 0.16),
        "IS": (rho, 0.0, 0.84),
        "TIS, C = 2": (tis(rho, 2.0), -0.1, 0.41),
    }
    for name, (w, bias, var) in rows.items():
        got_bias, got_var = bv(MU, PI, F, w)
        almost(got_bias, bias, 1e-12, label=f"{name} bias")
        almost(got_var, var, 1e-12, label=f"{name} variance")


@check("Plain IS is exactly unbiased, and TIS's bias is -E_mu[(rho - C)^+ f], on random distributions")
def test_is_unbiased_tis_bias(sub):
    bv = has(sub, "estimator_bias_variance")
    tis = has(sub, "tis_weights")
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        mu = torch.softmax(torch.randn(12, generator=g), dim=0)
        pi = torch.softmax(torch.randn(12, generator=g), dim=0)
        f = torch.randn(12, generator=g)
        rho = pi / mu
        bias, _ = bv(mu, pi, f, rho)
        almost(bias, 0.0, 1e-12, label="IS bias")
        bias, _ = bv(mu, pi, f, torch.ones(12))
        almost(bias, float((mu * f).sum() - (pi * f).sum()), 1e-12,
               label="bias with no correction, E_mu[f] - E_pi[f]")
        cap = 1.5
        bias, _ = bv(mu, pi, f, tis(rho, cap))
        almost(bias, float(-(mu * torch.clamp(rho - cap, min=0) * f).sum()), 1e-12,
               label="TIS bias")


@check("For n samples, TIS has the lowest MSE at n = 16 and IS the lowest at n = 1000")
def test_mse_tradeoff(sub):
    bv = has(sub, "estimator_bias_variance")
    tis = has(sub, "tis_weights")
    rho = PI / MU
    options = {"none": torch.ones(3), "IS": rho, "TIS": tis(rho, 2.0)}
    for n, best in ((16, "TIS"), (1000, "IS")):
        mse = {}
        for name, w in options.items():
            b, v = bv(MU, PI, F, w)
            mse[name] = b**2 + v / n
        winner = min(mse, key=mse.get)
        if winner != best:
            fail(f"at n = {n}, expected {best} to have the lowest mean squared error, "
                 f"got {', '.join(f'{k} {v:.4f}' for k, v in mse.items())}")


@check("sequence_ratio averages 1 under mu, and its second moment grows as (E_mu[rho^2])^T")
def test_sequence_ratio(sub):
    seq_ratio = has(sub, "sequence_ratio")
    mu = torch.tensor([0.4, 0.3, 0.2, 0.1])
    pi = torch.tensor([0.25, 0.25, 0.25, 0.25])
    one_step = float((mu * (pi / mu) ** 2).sum())
    for length in (1, 2, 4, 6):
        seqs = torch.tensor(list(itertools.product(range(4), repeat=length)))
        lp_mu, lp_pi = torch.log(mu)[seqs], torch.log(pi)[seqs]
        prob = lp_mu.sum(dim=1).exp()
        rho = seq_ratio(lp_pi, lp_mu, torch.ones(seqs.shape))
        shape(rho, (len(seqs),), label="sequence_ratio output")
        almost(float((prob * rho).sum()), 1.0, 1e-10, label=f"E_mu[rho], T = {length}")
        almost(float((prob * rho**2).sum()), one_step**length, 1e-9,
               label=f"E_mu[rho^2], T = {length}")
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    lp_t, lp_b = torch.tensor([[-1.0, -2.0, -30.0]]), torch.tensor([[-1.5, -1.0, 0.0]])
    almost(float(seq_ratio(lp_t, lp_b, mask)[0]), math.exp(-0.5), 1e-12,
           label="masked positions are left out")


@check("k3_mismatch is zero for identical engines, non-negative, and averages to KL(mu || pi)")
def test_k3(sub):
    k3 = has(sub, "k3_mismatch")
    g = torch.Generator().manual_seed(2)
    lp = torch.log_softmax(torch.randn(5, 7, generator=g), dim=-1)
    close(k3(lp, lp, torch.ones(5, 7)), torch.zeros(5), atol=0, label="identical engines")
    logits = torch.randn(10, generator=g)
    pi = torch.log_softmax(logits, dim=0)
    mu = torch.log_softmax(logits + 0.3 * torch.randn(10, generator=g), dim=0)
    # One-token sequences, one per outcome, weighted by mu.
    per = k3(pi[:, None], mu[:, None], torch.ones(10, 1))
    if torch.any(per < 0):
        fail("k3 = rho - 1 - log rho is never negative")
    kl = float((mu.exp() * (mu - pi)).sum())
    almost(float((mu.exp() * per).sum()), kl, 1e-12, label="E_mu[k3] against KL(mu || pi)")


@check("With exact sequence-level IS, the expected gradient from mu's samples is the on-policy gradient")
def test_sequence_is_unbiased(sub):
    loss_fn = has(sub, "is_pg_loss")
    g = torch.Generator().manual_seed(3)
    theta = torch.randn(T, V + 1, V, generator=g)
    stale = theta + 0.5 * torch.randn(T, V + 1, V, generator=g)
    want = _true_gradient(theta)
    close(_expected_gradient(loss_fn, theta, theta, level="token"), want, atol=1e-10,
          label="on-policy samples, token level")
    close(_expected_gradient(loss_fn, theta, stale, level="sequence"), want, atol=1e-10,
          label="off-policy samples, sequence-level IS")
    token = _expected_gradient(loss_fn, theta, stale, level="token")
    err = float((token - want).norm() / want.norm())
    if err < 0.05:
        fail("token-level ratios ignore the prefix and suffix, so their expected "
             f"gradient should be off by more than 5%; the relative error was {err:.4f}")


@check("Uncorrected gradients from stale samples drift further from the true gradient as the lag grows")
def test_staleness(sub):
    loss_fn = has(sub, "is_pg_loss")
    g = torch.Generator().manual_seed(4)
    path = [torch.randn(T, V + 1, V, generator=g)]
    for _ in range(8):
        path.append(path[-1] + 1.0 * _true_gradient(path[-1]))
    theta = path[-1]
    want = _true_gradient(theta)
    cosines = []
    for lag in (0, 2, 4, 8):
        mu_theta = path[-1 - lag]
        # No correction: pass the policy's own log-probs as the behaviour, so rho = 1.
        grad = _expected_gradient(
            lambda lp, _b, m, a, **kw: loss_fn(lp, lp.detach(), m, a, **kw), theta, mu_theta)
        cosines.append(_cos(grad, want))
    almost(cosines[0], 1.0, 1e-10, label="cosine at lag 0")
    if not all(a > b for a, b in zip(cosines, cosines[1:])) or cosines[-1] > 0.9:
        fail("the uncorrected gradient's cosine with the true gradient should fall "
             f"with the lag, below 0.9 at lag 8; got {[round(c, 4) for c in cosines]}")
    corrected = _expected_gradient(loss_fn, theta, path[0], level="sequence")
    almost(_cos(corrected, want), 1.0, 1e-10, label="sequence-level IS at lag 8")


@check("engine_logprobs: float64 gives ratios of exactly 1, and FP16 mismatches 20x less than BF16")
def test_precision(sub):
    engine = has(sub, "engine_logprobs")
    g = torch.Generator().manual_seed(0)
    logits = 3.0 * torch.randn(2000, 16, generator=g)
    train = torch.log_softmax(logits, dim=-1)
    same = engine(logits, torch.float64)
    shape(same, (2000, 16), label="engine_logprobs output")
    if not torch.equal(same, train):
        fail("an engine in float64 should reproduce the trainer's log-probs exactly")
    kl = {}
    for dtype in (torch.bfloat16, torch.float16):
        infer = engine(logits, dtype)
        kl[dtype] = float((infer.exp() * (infer - train)).sum(dim=-1).mean())
    if not kl[torch.float16] * 20 < kl[torch.bfloat16]:
        fail(f"mean KL(mu || pi) was {kl[torch.bfloat16]:.3g} for bfloat16 and "
             f"{kl[torch.float16]:.3g} for float16; expected float16 to be at least "
             "20 times smaller, since it keeps 3 more mantissa bits")
