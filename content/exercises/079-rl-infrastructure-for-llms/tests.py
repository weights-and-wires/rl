"""Checks for the asynchronous-RL exercise: staleness, mismatch, and packing."""

import itertools
import math

import torch

from rlcheck import almost, check, close, equal, fail, has, shape

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


@check("estimator_bias_variance matches the lesson's table, and TIS wins at n = 16 while IS wins at n = 1000")
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


@check("async_schedule: batch staleness never exceeds K, and K = 0 is the synchronous loop")
def test_async_staleness(sub):
    schedule = has(sub, "async_schedule")
    # Hand-worked: generation 1, training 2, four updates, K = 1.
    out = schedule(4, 1.0, 2.0, 1)
    equal(list(out["staleness"]), [0, 1, 1, 1], label="staleness with K = 1")
    equal(list(out["versions"]), [0, 0, 1, 2], label="generating versions with K = 1")
    # Without a cap the sampler runs ahead of a slow trainer (named wrong
    # implementation: ignoring the cap, which gives [0, 1, 2, 2] here).
    free = schedule(4, 1.0, 2.0, 10)
    equal(list(free["staleness"]), [0, 1, 2, 2], label="staleness with no effective cap")
    for g, t in ((1.0, 2.0), (3.0, 2.0), (2.0, 2.0), (0.5, 3.0)):
        for k in (0, 1, 2, 3):
            got = schedule(12, g, t, k)["staleness"]
            if max(got) > k:
                fail(f"gen {g}, train {t}, K = {k}: no batch may be more than K versions "
                     f"old, got staleness {list(got)}")
    sync = schedule(5, 3.0, 2.0, 0)
    equal(list(sync["staleness"]), [0] * 5, label="staleness with K = 0")
    almost(sync["makespan"], 5 * (3.0 + 2.0), 1e-12, label="K = 0 makespan, n * (gen + train)")


@check("async_schedule: utilization is computed exactly and asynchrony beats the synchronous loop")
def test_async_utilization(sub):
    schedule = has(sub, "async_schedule")
    n, g, t = 6, 1.0, 2.0
    sync = schedule(n, g, t, 0)
    almost(sync["trainer_utilization"], t / (g + t), 1e-12, label="sync trainer utilization")
    almost(sync["sampler_utilization"], g / (g + t), 1e-12, label="sync sampler utilization")
    # Trainer-bound: the trainer starts at time g and never waits again.
    for k in (1, 2, 5):
        out = schedule(n, g, t, k)
        almost(out["makespan"], g + n * t, 1e-12, label=f"makespan, K = {k}")
        almost(out["trainer_utilization"], n * t / (g + n * t), 1e-12,
               label=f"trainer utilization, K = {k}")
        almost(out["sampler_utilization"], n * g / (g + n * t), 1e-12,
               label=f"sampler utilization, K = {k}")
    # Sampler-bound: the sampler never waits, and the last update trails it by t.
    out = schedule(n, 3.0, 2.0, 2)
    almost(out["makespan"], n * 3.0 + 2.0, 1e-12, label="sampler-bound makespan")
    almost(out["sampler_utilization"], n * 3.0 / (n * 3.0 + 2.0), 1e-12,
           label="sampler-bound sampler utilization")


@check("logprob_mismatch: zero for equal logits, second-order in the perturbation, and TIS removes the bf16 bias")
def test_logprob_mismatch(sub):
    mismatch = has(sub, "logprob_mismatch")
    tis = has(sub, "tis_weights")
    bv = has(sub, "estimator_bias_variance")
    g = torch.Generator().manual_seed(0)
    logits = 3.0 * torch.randn(50, 32, generator=g)
    kl, max_abs = mismatch(logits, logits.clone())
    shape(kl, (50,), label="kl")
    almost(float(kl.abs().max()), 0.0, 1e-12, label="KL of identical logits")
    almost(max_abs, 0.0, 1e-12, label="max gap of identical logits")
    # KL(mu || pi) is about half the pi-variance of the logit change delta.
    pi = torch.softmax(logits, dim=-1)
    delta = 1e-3 * torch.randn(50, 32, generator=g)
    kl, _ = mismatch(logits, logits + delta)
    want = 0.5 * ((pi * delta**2).sum(-1) - (pi * delta).sum(-1) ** 2)
    close(kl, want, rtol=0.02, atol=0.0, label="KL for a small perturbation")
    # The sampler rounds its logits to bfloat16, the trainer keeps float32.
    low = logits.to(torch.bfloat16).to(torch.float64)
    kl, max_abs = mismatch(logits, low)
    if not (float(kl.min()) >= 0.0 and 0.02 < max_abs < 0.2):
        fail(f"bf16 rounding of logits of scale 3 should give KL >= 0 and a max "
             f"log-prob gap between 0.02 and 0.2, got min KL {float(kl.min()):.2e}, "
             f"max gap {max_abs:.4f}")
    plain, truncated = [], []
    for i in range(50):
        f = 2.0 * torch.randn(32, generator=g)
        mu, p = torch.softmax(low[i], -1), torch.softmax(logits[i], -1)
        rho = p / mu
        almost(bv(mu, p, f, rho)[0], 0.0, 1e-12, label="IS bias after the mismatch")
        plain.append(abs(bv(mu, p, f, torch.ones(32))[0]))
        truncated.append(abs(bv(mu, p, f, tis(rho, 1.02))[0]))
    if sum(truncated) > 0.05 * sum(plain):
        fail("truncated weights with C = 1.02 should remove at least 95% of the bias "
             f"from bf16 rounding, got mean |bias| {sum(truncated) / 50:.2e} "
             f"against {sum(plain) / 50:.2e} uncorrected")


@check("pack_rollouts: block-diagonal mask, restarted positions, and the packed loss equals the unpacked sum to 1e-10")
def test_packing(sub):
    pack = has(sub, "pack_rollouts")
    loss_fn = has(sub, "rollout_loss")
    rows = [torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6, 7]), torch.tensor([2, 3, 1, 4, 6])]
    resp = [torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 1]), torch.tensor([0, 0, 0, 1, 1])]
    adv = [1.0, -0.5, 2.0]
    tokens, pos, mask, targets, weights = pack(rows, resp, adv)
    equal(tokens.tolist(), [1, 2, 3, 4, 5, 6, 7, 2, 3, 1, 4, 6], label="packed tokens")
    equal(pos.tolist(), [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4], label="restarted positions")
    shape(mask, (12, 12), label="attention mask")
    want = torch.zeros(12, 12)
    for a, b in ((0, 4), (4, 7), (7, 12)):
        want[a:b, a:b] = torch.tril(torch.ones(b - a, b - a))
    close(mask, want, atol=0.0, label="block-diagonal causal mask")
    # Position 3 is the last of rollout 0: no next token, so no weight.
    # Position 2 predicts the response token 4, and rollout 0 has two response tokens.
    almost(float(weights[2]), 1.0 / 2, 1e-12, label="weight at a response target")
    almost(float(weights[3]), 0.0, 1e-12, label="weight at a rollout's last position")
    almost(float(weights[4]), -0.5 / 2, 1e-12, label="weight on a prompt-side position that predicts a response token")
    almost(float(weights[0]), 0.0, 1e-12, label="weight at a prompt target")
    packed = float(loss_fn(sub.toy_logits(tokens, pos, mask), targets, weights))
    separate = 0.0
    for r, s, a in zip(rows, resp, adv):
        one = pack([r], [s], [a])
        separate += float(loss_fn(sub.toy_logits(*one[:3]), one[3], one[4]))
    almost(packed, separate, 1e-10, label="packed loss against the sum of unpacked losses")
    # Named wrong implementation: one causal mask over the whole row, so the
    # second rollout attends to the first.
    leaky = torch.tril(torch.ones(12, 12))
    bad = float(loss_fn(sub.toy_logits(tokens, pos, leaky), targets, weights))
    if abs(bad - separate) < 1e-3:
        fail("the test model should show a cross-rollout leak with a single causal "
             f"mask, got {bad:.6f} against {separate:.6f}")
