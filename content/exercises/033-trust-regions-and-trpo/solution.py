"""Reference solution for the natural gradients and TRPO exercise."""

import numpy as np
import torch

torch.set_num_threads(1)
DTYPE = torch.float64


def make_contextual_bandit(n_contexts=8, n_features=4, n_actions=3, seed=0):
    """A small contextual bandit and a linear softmax policy over it. Given.

    Returns:
        features: shape (C, D), one feature vector per context.
        rewards: shape (C, A), the deterministic reward of each action in each
            context.
        logits_fn: a function mapping flat parameters, shape (A * D,), to
            logits for every context, shape (C, A).
    """
    rng = np.random.default_rng(seed)
    features = torch.tensor(rng.normal(size=(n_contexts, n_features)), dtype=DTYPE)
    rewards = torch.tensor(rng.normal(size=(n_contexts, n_actions)), dtype=DTYPE)

    def logits_fn(params):
        return features @ params.view(n_actions, n_features).T

    return features, rewards, logits_fn


def flat_grad(y, params, create_graph=False):
    """The gradient of scalar y with respect to params, as a flat vector. Given."""
    (g,) = torch.autograd.grad(y, params, create_graph=create_graph)
    return g.reshape(-1)


def mean_kl(logits_old, logits_new):
    logp_old = torch.log_softmax(logits_old, dim=-1)
    logp_new = torch.log_softmax(logits_new, dim=-1)
    return (logp_old.exp() * (logp_old - logp_new)).sum(dim=-1).mean()


def fisher_vector_product(logits_fn, params, v, damping=0.0):
    params = params.detach().requires_grad_(True)
    logits = logits_fn(params)
    kl = mean_kl(logits.detach(), logits)
    grad = flat_grad(kl, params, create_graph=True)
    hvp = flat_grad(grad @ v, params)
    return hvp + damping * v


def conjugate_gradient(Avp, b, iters=10, tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rr = r @ r
    for _ in range(iters):
        if rr <= tol:
            break
        Ap = Avp(p)
        alpha = rr / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = r @ r
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x


def surrogate(logits, actions, advantages, logp_old):
    logp = torch.log_softmax(logits, dim=-1).gather(-1, actions[:, None]).squeeze(-1)
    return (torch.exp(logp - logp_old) * advantages).mean()


def natural_step(g, fvp, max_kl, cg_iters=10):
    x = conjugate_gradient(fvp, g, iters=cg_iters)
    shs = x @ fvp(x)
    return torch.sqrt(2.0 * max_kl / shs) * x


def line_search(surrogate_fn, kl_fn, params, full_step, max_kl, backtracks=10, shrink=0.5):
    with torch.no_grad():
        base = surrogate_fn(params)
        fraction = 1.0
        for _ in range(backtracks):
            candidate = params + fraction * full_step
            if kl_fn(candidate) <= max_kl and surrogate_fn(candidate) > base:
                return candidate, fraction
            fraction *= shrink
    return params.clone(), 0.0


def trpo_update(logits_fn, params, contexts, actions, advantages, max_kl=0.01, damping=0.1):
    params = params.detach()
    with torch.no_grad():
        logits_old = logits_fn(params)[contexts]
        logp_old = torch.log_softmax(logits_old, dim=-1).gather(-1, actions[:, None]).squeeze(-1)

    def surrogate_fn(p):
        return surrogate(logits_fn(p)[contexts], actions, advantages, logp_old)

    def kl_fn(p):
        return mean_kl(logits_old, logits_fn(p)[contexts])

    p = params.clone().requires_grad_(True)
    g = flat_grad(surrogate_fn(p), p)
    fvp = lambda v: fisher_vector_product(lambda q: logits_fn(q)[contexts], params, v, damping)
    step = natural_step(g, fvp, max_kl)
    new_params, _ = line_search(surrogate_fn, kl_fn, params, step, max_kl)
    return new_params


def sample_batch(logits_fn, params, rewards, batch, generator):
    """Sample contexts uniformly and actions from the policy. Given.

    Returns (contexts, actions, advantages). The advantage is exact:
    the action's reward minus the policy's expected reward in that context.
    """
    with torch.no_grad():
        pi = torch.softmax(logits_fn(params), dim=-1)
        contexts = torch.randint(len(rewards), (batch,), generator=generator)
        actions = torch.multinomial(pi[contexts], 1, generator=generator).squeeze(-1)
        values = (pi * rewards).sum(dim=-1)
        advantages = rewards[contexts, actions] - values[contexts]
    return contexts, actions, advantages


def expected_reward(logits_fn, params, rewards):
    """The exact objective: the mean over contexts of sum_a pi(a | c) r(c, a). Given."""
    with torch.no_grad():
        return float((torch.softmax(logits_fn(params), dim=-1) * rewards).sum(dim=-1).mean())


if __name__ == "__main__":
    features, rewards, logits_fn = make_contextual_bandit(seed=0)
    params = torch.zeros(3 * 4, dtype=DTYPE)
    gen = torch.Generator().manual_seed(0)
    best = float(rewards.max(dim=-1).values.mean())
    print(f"best possible expected reward: {best:.4f}")
    print("iter  expected reward   KL to previous policy")
    for it in range(15):
        contexts, actions, advantages = sample_batch(logits_fn, params, rewards, 256, gen)
        new = trpo_update(logits_fn, params, contexts, actions, advantages, max_kl=0.02)
        kl = float(mean_kl(logits_fn(params), logits_fn(new)))
        params = new
        print(f"{it:4d}  {expected_reward(logits_fn, params, rewards):14.4f}   {kl:.4f}")
