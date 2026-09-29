"""Checks for the natural gradients and TRPO exercise."""

import numpy as np
import torch

from rlcheck import check, close, fail, finite, has, shape

torch.set_num_threads(1)
DTYPE = torch.float64


def _problem(seed=0):
    rng = np.random.default_rng(seed)
    C, D, A = 8, 4, 3
    features = torch.tensor(rng.normal(size=(C, D)), dtype=DTYPE)
    rewards = torch.tensor(rng.normal(size=(C, A)), dtype=DTYPE)
    params = torch.tensor(rng.normal(size=A * D) * 0.5, dtype=DTYPE)

    def logits_fn(p):
        return features @ p.view(A, D).T

    return features, rewards, params, logits_fn


def _explicit_fisher(logits_fn, params):
    """mean over contexts of sum_a pi(a | c) score score^T, by enumerating actions."""
    logp = torch.log_softmax(logits_fn(params), dim=-1).detach()
    jac = torch.autograd.functional.jacobian(lambda p: torch.log_softmax(logits_fn(p), dim=-1), params)
    C, A, P = jac.shape
    F = torch.zeros(P, P, dtype=DTYPE)
    for c in range(C):
        for a in range(A):
            F += logp[c, a].exp() * torch.outer(jac[c, a], jac[c, a])
    return F / C


def _spd(n, seed):
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n))
    return M @ M.T + n * np.eye(n)


@check("mean_kl is zero for equal policies and matches a hand computation")
def test_mean_kl(sub):
    mean_kl = has(sub, "mean_kl")
    old = torch.log(torch.tensor([[0.5, 0.5], [0.2, 0.8]], dtype=DTYPE))
    new = torch.log(torch.tensor([[0.25, 0.75], [0.2, 0.8]], dtype=DTYPE))
    got = mean_kl(old, new)
    # Row 0: 0.5 log 2 + 0.5 log(2/3). Row 1: 0.
    want = 0.5 * (0.5 * np.log(2.0) + 0.5 * np.log(2.0 / 3.0))
    close(float(got), want, atol=1e-12, label="mean KL(old || new) over two rows")
    close(float(mean_kl(old, old)), 0.0, atol=1e-12, label="KL of a policy with itself")
    close(float(mean_kl(old + 5.0, new - 2.0)), want, atol=1e-12,
          label="KL from logits shifted by a constant per row")
    if float(mean_kl(new, old)) == float(got):
        fail("KL is not symmetric: mean_kl(new, old) should differ from mean_kl(old, new) here")


@check("fisher_vector_product equals the explicitly built Fisher matrix times v")
def test_fvp_explicit(sub):
    fvp = has(sub, "fisher_vector_product")
    _, _, params, logits_fn = _problem(0)
    F = _explicit_fisher(logits_fn, params)
    rng = np.random.default_rng(1)
    for _ in range(3):
        v = torch.tensor(rng.normal(size=params.shape[0]), dtype=DTYPE)
        got = fvp(logits_fn, params, v)
        shape(got, tuple(params.shape), label="fisher_vector_product output")
        close(got.detach().numpy(), (F @ v).numpy(), atol=1e-10, label="F v against the explicit Fisher")
    v = torch.tensor(rng.normal(size=params.shape[0]), dtype=DTYPE)
    close(fvp(logits_fn, params, v, damping=0.1).detach().numpy(), (F @ v + 0.1 * v).numpy(), atol=1e-10,
          label="with damping 0.1, (F + 0.1 I) v")


@check("fisher_vector_product equals the Hessian of the mean KL at theta_old times v")
def test_fvp_hessian(sub):
    fvp = has(sub, "fisher_vector_product")
    _, _, params, logits_fn = _problem(2)
    old = logits_fn(params).detach()

    def kl(p):
        lo, ln = torch.log_softmax(old, -1), torch.log_softmax(logits_fn(p), -1)
        return (lo.exp() * (lo - ln)).sum(-1).mean()

    H = torch.autograd.functional.hessian(kl, params)
    v = torch.tensor(np.random.default_rng(3).normal(size=params.shape[0]), dtype=DTYPE)
    close(fvp(logits_fn, params, v).detach().numpy(), (H @ v).numpy(), atol=1e-10, label="Hessian-vector product")


@check("conjugate_gradient matches np.linalg.solve on symmetric positive definite systems")
def test_conjugate_gradient(sub):
    cg = has(sub, "conjugate_gradient")
    for n, seed in [(10, 0), (5, 1)]:
        A = _spd(n, seed)
        b = np.random.default_rng(seed + 10).normal(size=n)
        At = torch.tensor(A, dtype=DTYPE)
        calls = []

        def Avp(p):
            calls.append(1)
            return At @ p

        x = cg(Avp, torch.tensor(b, dtype=DTYPE), iters=n)
        close(np.asarray(x, dtype=float), np.linalg.solve(A, b), atol=1e-6,
              label=f"{n} by {n} system after {n} iterations")
        if len(calls) > n + 1:
            fail(f"conjugate_gradient called Avp {len(calls)} times for iters={n}; "
                 "it needs one product per iteration")
    A = _spd(10, 4)
    b = np.random.default_rng(5).normal(size=10)
    x2 = np.asarray(cg(lambda p: torch.tensor(A, dtype=DTYPE) @ p, torch.tensor(b, dtype=DTYPE), iters=2), dtype=float)
    x6 = np.asarray(cg(lambda p: torch.tensor(A, dtype=DTYPE) @ p, torch.tensor(b, dtype=DTYPE), iters=6), dtype=float)
    r2, r6 = np.linalg.norm(A @ x2 - b), np.linalg.norm(A @ x6 - b)
    if not r6 < r2:
        fail(f"more iterations should shrink the residual: {r2:.3g} after 2, {r6:.3g} after 6")


@check("surrogate equals the mean advantage at theta_old, and its gradient there is the policy gradient")
def test_surrogate(sub):
    surrogate = has(sub, "surrogate")
    _, _, params, logits_fn = _problem(4)
    g = torch.Generator().manual_seed(0)
    contexts = torch.randint(8, (50,), generator=g)
    actions = torch.randint(3, (50,), generator=g)
    advantages = torch.randn(50, generator=g, dtype=DTYPE)
    with torch.no_grad():
        logp_old = torch.log_softmax(logits_fn(params)[contexts], -1).gather(-1, actions[:, None]).squeeze(-1)
    p = params.clone().requires_grad_(True)
    val = surrogate(logits_fn(p)[contexts], actions, advantages, logp_old)
    close(float(val), float(advantages.mean()), atol=1e-12, label="surrogate at theta = theta_old")
    (grad,) = torch.autograd.grad(val, p)
    q = params.clone().requires_grad_(True)
    logp = torch.log_softmax(logits_fn(q)[contexts], -1).gather(-1, actions[:, None]).squeeze(-1)
    (pg,) = torch.autograd.grad((logp * advantages).mean(), q)
    close(grad.numpy(), pg.numpy(), atol=1e-12, label="gradient of the surrogate against mean(A * grad log pi)")


@check("natural_step gives sqrt(2 delta / g^T F^-1 g) F^-1 g, and its quadratic KL equals delta")
def test_natural_step(sub):
    natural_step = has(sub, "natural_step")
    _, _, params, logits_fn = _problem(5)
    F = _explicit_fisher(logits_fn, params) + 0.1 * torch.eye(12, dtype=DTYPE)
    g = torch.tensor(np.random.default_rng(6).normal(size=12), dtype=DTYPE)
    step = natural_step(g, lambda v: F @ v, max_kl=0.01, cg_iters=12)
    shape(step, (12,), label="natural_step output")
    x = torch.linalg.solve(F, g)
    want = torch.sqrt(2 * 0.01 / (g @ x)) * x
    close(np.asarray(step.detach(), dtype=float), want.numpy(), atol=1e-8, label="the natural step")
    close(float(0.5 * step @ F @ step), 0.01, atol=1e-10, label="0.5 * s^T F s")


@check("line_search halves the step until the KL constraint holds and the surrogate improves")
def test_line_search(sub):
    line_search = has(sub, "line_search")
    params = torch.tensor([0.0], dtype=DTYPE)
    step = torch.tensor([1.0], dtype=DTYPE)
    new, frac = line_search(lambda p: float(3 * p[0]), lambda p: float(p[0] ** 2), params, step, max_kl=0.1)
    close(float(frac), 0.25, atol=0.0, label="accepted fraction when KL(fraction) = fraction^2 and delta = 0.1")
    close(np.asarray(new, dtype=float), [0.25], atol=1e-12, label="accepted parameters")
    new, frac = line_search(lambda p: float(-p[0]), lambda p: float(p[0] ** 2), params, step, max_kl=0.1)
    close(float(frac), 0.0, atol=0.0, label="fraction when the surrogate never improves")
    close(np.asarray(new, dtype=float), [0.0], atol=0.0, label="parameters when no step is accepted")
    new, frac = line_search(lambda p: float(p[0] - 2 * p[0] ** 2), lambda p: float(p[0] ** 2) * 0.01,
                            params, step, max_kl=0.1)
    close(float(frac), 0.25, atol=0.0,
          label="fraction when the KL holds at once but the surrogate first improves at 0.25")


@check("trpo_update keeps every step inside the trust region and raises the expected reward")
def test_trpo_update(sub):
    trpo_update = has(sub, "trpo_update")
    mean_kl = has(sub, "mean_kl")
    _, rewards, _, logits_fn = _problem(0)
    params = torch.zeros(12, dtype=DTYPE)
    gen = torch.Generator().manual_seed(0)

    def value(p):
        with torch.no_grad():
            return float((torch.softmax(logits_fn(p), -1) * rewards).sum(-1).mean())

    start = value(params)
    for it in range(10):
        with torch.no_grad():
            pi = torch.softmax(logits_fn(params), -1)
            contexts = torch.randint(8, (256,), generator=gen)
            actions = torch.multinomial(pi[contexts], 1, generator=gen).squeeze(-1)
            advantages = rewards[contexts, actions] - (pi * rewards).sum(-1)[contexts]
        new = trpo_update(logits_fn, params, contexts, actions, advantages, max_kl=0.02, damping=0.1)
        new = torch.as_tensor(new, dtype=DTYPE).detach()
        finite(new.numpy(), label=f"parameters after update {it}")
        kl = float(mean_kl(logits_fn(params)[contexts], logits_fn(new)[contexts]))
        if not kl <= 0.02 + 1e-9:
            fail(f"update {it}: KL on the batch is {kl:.4f}, above max_kl = 0.02")
        params = new
    end = value(params)
    if not end > start + 0.25:
        fail(f"10 TRPO updates took the expected reward from {start:.3f} only to {end:.3f}; "
             "expected a gain of more than 0.25")
