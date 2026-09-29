"""Checks for the RLHF-with-PPO exercise. CPU only, toy-sized."""

import numpy as np
import torch

from rlcheck import almost, check, close, fail, has, shape

torch.set_num_threads(1)

D = torch.float64


def _t(x):
    return torch.tensor(x, dtype=D)


@check("kl_shaped_rewards charges -beta * log-ratio per token and pays r on the last token")
def test_shaped_rewards(sub):
    shaped = has(sub, "kl_shaped_rewards")
    logp = _t([[-0.5, -1.0, -0.2, 0.0], [-0.3, -0.4, -0.1, -2.0]])
    ref = _t([[-0.7, -0.9, -0.6, 0.0], [-0.3, -0.6, -0.1, -1.0]])
    mask = _t([[1, 1, 1, 0], [1, 1, 1, 1]])
    got = shaped(logp, ref, _t([1.0, -0.5]), mask, 0.1)
    shape(got, (2, 4), label="kl_shaped_rewards")
    close(got, [[-0.02, 0.01, 0.96, 0.0], [0.0, -0.02, 0.0, -0.4]], atol=1e-12,
          label="shaped rewards (the task reward belongs on each row's last response token)")

    # Padding may hold anything; it must not leak into the rewards.
    logp_pad = logp.clone()
    logp_pad[0, 3] = 5.0
    close(shaped(logp_pad, ref, _t([1.0, -0.5]), mask, 0.1), got, atol=1e-12,
          label="a padded position with a stray log-prob")

    rng = np.random.default_rng(0)
    lengths = rng.integers(1, 7, size=5)
    mask = _t(np.arange(6)[None, :] < lengths[:, None])
    logp, ref, r = _t(rng.normal(size=(5, 6))), _t(rng.normal(size=(5, 6))), _t(rng.normal(size=5))
    got = shaped(logp, ref, r, mask, 0.3)
    want = r - 0.3 * ((logp - ref) * mask).sum(dim=1)
    close(got.sum(dim=1), want, atol=1e-12, label="each row sums to r - beta * log(pi / pi_ref)")


@check("kl_estimators: k1 and k3 are unbiased for KL(q || p), k2 isn't, and k3 is never negative")
def test_estimator_values(sub):
    kl_estimators = has(sub, "kl_estimators")
    q = _t([0.5, 0.3, 0.2])
    p = _t([0.3, 0.3, 0.4])
    k1, k2, k3 = kl_estimators(torch.log(q), torch.log(p))
    close(k1, [0.51083, 0.0, -0.69315], atol=1e-5, label="k1 on the lesson's example")
    close(k2, [0.13047, 0.0, 0.24023], atol=1e-5, label="k2 on the lesson's example")
    close(k3, [0.11083, 0.0, 0.30685], atol=1e-5, label="k3 on the lesson's example")

    gen = torch.Generator().manual_seed(0)
    for trial in range(5):
        q = torch.softmax(torch.randn(8, generator=gen, dtype=D), 0)
        p = torch.softmax(torch.randn(8, generator=gen, dtype=D), 0)
        kl = float((q * (q.log() - p.log())).sum())
        k1, k2, k3 = kl_estimators(q.log(), p.log())
        almost(float((q * k1).sum()), kl, tol=1e-12, label=f"E_q[k1] against KL (trial {trial})")
        almost(float((q * k3).sum()), kl, tol=1e-12, label=f"E_q[k3] against KL (trial {trial})")
        if abs(float((q * k2).sum()) - kl) < 1e-4:
            fail("E_q[k2] matched the KL almost exactly; k2 should be biased for distant q and p")
        if float(k3.min()) < -1e-15:
            fail(f"k3 must be non-negative, got {float(k3.min()):.3g}")
    if float(k1.min()) >= 0:
        fail("k1 should be negative for some tokens (any where p > q)")

    # Close distributions: k3 has far lower variance than k1.
    q = torch.softmax(torch.randn(8, generator=gen, dtype=D), 0)
    p = torch.softmax(q.log() + 0.1 * torch.randn(8, generator=gen, dtype=D), 0)
    k1, _, k3 = kl_estimators(q.log(), p.log())
    var = lambda k: float((q * (k - (q * k).sum()) ** 2).sum())
    if not var(k3) < 0.1 * var(k1):
        fail(f"for close q and p, Var[k3] = {var(k3):.3g} should be far below Var[k1] = {var(k1):.3g}")


@check("As losses, k1 has zero expected gradient, k2 gives grad KL(q||p), and k3 gives grad KL(p||q)")
def test_estimator_gradients(sub):
    kl_estimators = has(sub, "kl_estimators")
    gen = torch.Generator().manual_seed(1)
    theta0 = torch.randn(8, generator=gen, dtype=D)
    p = torch.softmax(torch.randn(8, generator=gen, dtype=D), 0)

    def expected_grad(which):
        theta = theta0.clone().requires_grad_(True)
        logq = torch.log_softmax(theta, 0)
        k = kl_estimators(logq, p.log())[which]
        # Sampled tokens are constants: weight each outcome by q, detached.
        loss = (logq.exp().detach() * k).sum()
        (g,) = torch.autograd.grad(loss, theta)
        return g

    def true_grad(direction):
        theta = theta0.clone().requires_grad_(True)
        logq = torch.log_softmax(theta, 0)
        q = logq.exp()
        kl = (q * (logq - p.log())).sum() if direction == "qp" else (p * (p.log() - logq)).sum()
        (g,) = torch.autograd.grad(kl, theta)
        return g

    close(expected_grad(0), torch.zeros(8), atol=1e-12, label="expected gradient of k1 as a loss")
    close(expected_grad(1), true_grad("qp"), atol=1e-12, label="k2 against grad KL(q || p)")
    close(expected_grad(2), true_grad("pq"), atol=1e-12, label="k3 against grad KL(p || q)")
    gap = float((true_grad("pq") - true_grad("qp")).abs().max())
    if gap < 1e-3:
        fail("test setup: the two KL gradients should differ")


@check("gae on tokens matches the worked example and both lambda limits, with no bootstrap past the end")
def test_gae(sub):
    gae = has(sub, "gae")
    rewards = _t([[-0.02, 0.01, 0.96, 0.0]])
    values = _t([[0.5, 0.6, 0.8, 7.0]])
    mask = _t([[1, 1, 1, 0]])
    adv, ret = gae(rewards, values, mask, gamma=1.0, lam=0.95)
    shape(adv, (1, 4), label="advantages")
    close(adv, [[0.4239, 0.362, 0.16, 0.0]], atol=1e-10, label="advantages (value 7.0 is padding)")
    close(ret, [[0.9239, 0.962, 0.96, 0.0]], atol=1e-10, label="returns = advantages + values")

    rng = np.random.default_rng(0)
    lengths = rng.integers(1, 8, size=6)
    mask = _t(np.arange(8)[None, :] < lengths[:, None])
    rewards = _t(rng.normal(size=(6, 8))) * mask
    values = _t(rng.normal(size=(6, 8)))
    adv1, _ = gae(rewards, values, mask, gamma=1.0, lam=1.0)
    to_go = torch.flip(torch.cumsum(torch.flip(rewards, [1]), 1), [1])
    close(adv1, (to_go - values) * mask, atol=1e-10, label="lam = 1 against reward-to-go minus V")
    adv0, _ = gae(rewards, values, mask, gamma=0.9, lam=0.0)
    nxt = torch.cat([values[:, 1:] * mask[:, 1:], torch.zeros(6, 1, dtype=D)], dim=1)
    close(adv0, (rewards + 0.9 * nxt - values) * mask, atol=1e-10, label="lam = 0 against TD errors")


@check("masked_whiten gives mean 0 and variance 1 over response tokens, and 0 on padding")
def test_whiten(sub):
    whiten = has(sub, "masked_whiten")
    x = _t([[1.0, 2.0, 3.0, 100.0], [4.0, 5.0, -50.0, 9.0]])
    mask = _t([[1, 1, 1, 0], [1, 1, 0, 0]])
    got = whiten(x, mask)
    vals = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=D)
    want = (vals - 3.0) / torch.sqrt(torch.tensor(2.0, dtype=D) + 1e-8)
    close(got[mask.bool()], want, atol=1e-7, label="whitened response entries (variance over N)")
    close(got[~mask.bool()], torch.zeros(3), atol=1e-12, label="padded entries")


@check("ppo_policy_loss clips exactly as the lesson's table says, and the gradient stops where it should")
def test_policy_loss(sub):
    loss_fn = has(sub, "ppo_policy_loss")
    ratios = _t([1.3, 0.7, 0.7, 1.3])
    adv = _t([1.0, 1.0, -1.0, -1.0])
    # Rows: one token each, so each loss is minus that token's surrogate.
    want_value = [-1.2, -0.7, 0.8, 1.3]
    want_grad = [0.0, -0.7, 0.0, 1.3]    # d loss / d logp = -ratio * A when unclipped
    for i in range(4):
        logp = torch.zeros(1, 1, dtype=D, requires_grad=True)
        old = -torch.log(ratios[i]).reshape(1, 1)
        loss = loss_fn(logp, old, adv[i].reshape(1, 1), torch.ones(1, 1, dtype=D), 0.2)
        almost(float(loss), want_value[i], tol=1e-12, label=f"loss for ratio {ratios[i]:.1f}, A {adv[i]:+.0f}")
        (g,) = torch.autograd.grad(loss, logp)
        almost(float(g), want_grad[i], tol=1e-12,
               label=f"gradient for ratio {ratios[i]:.1f}, A {adv[i]:+.0f}")

    # At ratio 1 the loss is minus the mean advantage over response tokens only.
    adv = _t([[1.0, 2.0, 99.0], [3.0, -1.0, 4.0]])
    mask = _t([[1, 1, 0], [1, 1, 1]])
    logp = torch.full((2, 3), -1.0, dtype=D)
    almost(float(loss_fn(logp, logp.clone(), adv, mask, 0.2)), -9.0 / 5.0, tol=1e-12,
           label="per-token mean over the 5 response tokens")


@check("value_loss is half the mean of the larger of the clipped and unclipped squared errors")
def test_value_loss(sub):
    value_loss = has(sub, "value_loss")
    got = value_loss(_t([[1.5]]), _t([[1.0]]), _t([[2.0]]), _t([[1.0]]), 0.2)
    almost(float(got), 0.32, tol=1e-12, label="V 1.5, V_old 1.0, R 2.0, eps 0.2")
    got = value_loss(_t([[1.1, 0.0, 5.0]]), _t([[1.0, 0.0, 0.0]]), _t([[2.0, 1.0, 0.0]]),
                     _t([[1, 1, 0]]), 0.2)
    almost(float(got), 0.5 * (0.81 + 1.0) / 2, tol=1e-12, label="two tokens and one padded")


def _ref_losses(logp, old, adv, values, old_values, returns, mask, eps, vf):
    ratio = torch.exp(logp - old)
    surr = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv)
    pl = -(surr * mask).sum() / mask.sum()
    clipped = old_values + torch.clamp(values - old_values, -eps, eps)
    vl = 0.5 * (torch.maximum((values - returns) ** 2, (clipped - returns) ** 2) * mask).sum() / mask.sum()
    return pl + vf * vl


@check("ppo_update with one epoch and one minibatch takes exactly one step on policy + vf_coef * value loss")
def test_ppo_update_step(sub):
    ppo_update = has(sub, "ppo_update")
    gen = torch.Generator().manual_seed(0)
    B, T, V = 6, 4, 5
    tokens = torch.randint(0, V, (B, T), generator=gen)
    mask = (torch.arange(T)[None, :] < torch.tensor([4, 3, 2, 4, 1, 3])[:, None]).to(D)
    logits = torch.nn.Parameter(torch.randn(T, V, generator=gen, dtype=D))
    vparam = torch.nn.Parameter(torch.randn(T, generator=gen, dtype=D))

    def forward(idx):
        logp = torch.log_softmax(logits, -1)[torch.arange(T)[None, :], tokens[idx]]
        return logp, vparam.expand(len(idx), T)

    with torch.no_grad():
        old_logp, old_values = forward(torch.arange(B))
        old_logp = old_logp - 0.3 * torch.randn(B, T, generator=gen, dtype=D)
        old_values = old_values.clone()
    batch = {
        "old_logp": old_logp,
        "old_values": old_values,
        "advantages": torch.randn(B, T, generator=gen, dtype=D),
        "returns": torch.randn(B, T, generator=gen, dtype=D),
        "mask": mask,
    }
    start = [logits.detach().clone(), vparam.detach().clone()]
    logp, values = forward(torch.arange(B))
    total = _ref_losses(logp, old_logp, batch["advantages"], values, old_values,
                        batch["returns"], mask, 0.2, 0.5)
    grads = torch.autograd.grad(total, [logits, vparam])
    optimizer = torch.optim.SGD([logits, vparam], lr=0.1)
    stats = ppo_update(forward, optimizer, batch, epochs=1, minibatch_size=B, clip_eps=0.2,
                       vf_coef=0.5, max_grad_norm=1e9, generator=torch.Generator().manual_seed(0))
    close(logits.detach(), start[0] - 0.1 * grads[0], atol=1e-12, label="policy parameters after one step")
    close(vparam.detach(), start[1] - 0.1 * grads[1], atol=1e-12, label="value parameters after one step")
    for key in ("policy_loss", "value_loss", "clip_frac"):
        if key not in stats:
            fail(f"ppo_update should return a dict with {key!r}; got keys {sorted(stats)}")

    # A tiny max_grad_norm must cap the step's size.
    logits.data.copy_(start[0])
    vparam.data.copy_(start[1])
    ppo_update(forward, optimizer, batch, epochs=1, minibatch_size=B, max_grad_norm=1e-3,
               generator=torch.Generator().manual_seed(0))
    moved = torch.sqrt(((logits.detach() - start[0]) ** 2).sum() + ((vparam.detach() - start[1]) ** 2).sum())
    almost(float(moved), 0.1 * 1e-3, tol=1e-9, label="step length with max_grad_norm = 1e-3 and lr = 0.1")


def _soft_optimum(ref_logits, beta):
    """beta * log Z for the rising reward, by soft backward induction over (t, prev)."""
    T, _, V = ref_logits.shape
    log_ref = torch.log_softmax(ref_logits, -1)
    up = (torch.arange(V)[None, :] > torch.arange(V + 1)[:, None]).to(D)
    up[V] = 0.0
    v_next = torch.zeros(V, dtype=D)
    for t in reversed(range(T)):
        step = up / (T - 1) if t > 0 else torch.zeros(V + 1, V, dtype=D)
        v = beta * torch.logsumexp(log_ref[t] + (step + v_next[None, :]) / beta, dim=-1)
        v_next = v[:V]
    return float(v[V])


def _objective(logits, ref_logits, beta):
    """Exact E[r] - beta * KL for a tabular policy, by propagating the state distribution."""
    T, _, V = logits.shape
    log_pi, log_ref = torch.log_softmax(logits, -1), torch.log_softmax(ref_logits, -1)
    up = (torch.arange(V)[None, :] > torch.arange(V + 1)[:, None]).to(D)
    up[V] = 0.0
    dist = torch.zeros(V + 1, dtype=D)
    dist[V] = 1.0
    total = 0.0
    for t in range(T):
        pi = log_pi[t].exp()
        step = up / (T - 1) if t > 0 else torch.zeros(V + 1, V, dtype=D)
        total += float((dist[:, None] * pi * (step - beta * (log_pi[t] - log_ref[t]))).sum())
        dist = torch.cat([(dist[:, None] * pi).sum(0), torch.zeros(1, dtype=D)])
    return total


class _Toy(torch.nn.Module):
    """The same tabular policy with a value table as the starter's ToyPolicy."""

    def __init__(self, seed, vocab=8, length=6):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.vocab, self.length = vocab, length
        self.logits = torch.nn.Parameter(torch.randn(length, vocab + 1, vocab, generator=gen, dtype=D))
        self.values = torch.nn.Parameter(torch.zeros(length, vocab + 1, dtype=D))

    def forward(self, tokens):
        batch, steps = tokens.shape
        prev = torch.cat([torch.full((batch, 1), self.vocab), tokens[:, :-1]], dim=1)
        pos = torch.arange(steps).expand(batch, steps)
        logp = torch.log_softmax(self.logits[pos, prev], -1).gather(-1, tokens[..., None])[..., 0]
        return logp, self.values[pos, prev]

    @torch.no_grad()
    def sample(self, n, gen):
        tokens = torch.zeros(n, self.length, dtype=torch.long)
        prev = torch.full((n,), self.vocab)
        for t in range(self.length):
            tokens[:, t] = torch.multinomial(torch.softmax(self.logits[t, prev], -1), 1, generator=gen)[:, 0]
            prev = tokens[:, t]
        return tokens


@check("PPO built from your functions closes 85% of the gap to the exact KL-regularised optimum")
def test_ppo_toy(sub):
    shaped = has(sub, "kl_shaped_rewards")
    gae = has(sub, "gae")
    whiten = has(sub, "masked_whiten")
    ppo_update = has(sub, "ppo_update")
    seed, beta = 0, 0.05
    gen = torch.Generator().manual_seed(seed)
    ref, policy = _Toy(seed), _Toy(seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.05)
    first = None
    for _ in range(40):
        tokens = policy.sample(256, gen)
        mask = torch.ones(tokens.shape, dtype=D)
        with torch.no_grad():
            old_logp, old_values = policy(tokens)
            ref_logp, _ = ref(tokens)
        task = (tokens[:, 1:] > tokens[:, :-1]).to(D).mean(dim=1)
        first = float(task.mean()) if first is None else first
        adv, ret = gae(shaped(old_logp, ref_logp, task, mask, beta), old_values, mask, 1.0, 0.95)
        batch = {"old_logp": old_logp, "old_values": old_values, "advantages": whiten(adv, mask),
                 "returns": ret, "mask": mask}
        ppo_update(lambda idx: policy(tokens[idx]), optimizer, batch, epochs=4, minibatch_size=64,
                   clip_eps=0.2, vf_coef=0.5, max_grad_norm=1.0, generator=gen)
    best = _soft_optimum(ref.logits.detach(), beta)
    start = _objective(ref.logits.detach(), ref.logits.detach(), beta)
    reached = _objective(policy.logits.detach(), ref.logits.detach(), beta)
    frac = (reached - start) / (best - start)
    if not frac > 0.85:
        fail(f"J went from {start:.4f} to {reached:.4f}; the exact optimum beta log Z is {best:.4f}. "
             f"That closes {100 * frac:.0f}% of the gap, below 85%. (Mean task reward started at "
             f"{first:.3f}.)")
    if reached > best + 1e-9:
        fail(f"J = {reached:.6f} is above beta log Z = {best:.6f}, which is impossible")
