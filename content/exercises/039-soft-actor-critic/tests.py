"""Checks for the soft actor-critic exercise."""

import math

import numpy as np
import torch
from scipy.special import ndtr, ndtri

from rlcheck import almost, check, close, fail, finite_difference, has, shape

torch.set_num_threads(1)


class _FixedPolicy(torch.nn.Module):
    """A state-independent tanh-Gaussian policy with learnable mean and log std, in float64."""

    def __init__(self, mean, log_std, max_action):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.tensor([mean], dtype=torch.float64))
        self.ls = torch.nn.Parameter(torch.tensor([log_std], dtype=torch.float64))
        self.max_action = max_action

    def forward(self, s):
        n = s.shape[0]
        return self.mu.expand(n, 1), self.ls.expand(n, 1)


def _critic(centre, scale):
    """A known critic Q(s, a) = -scale * (a - centre)^2, summed over action dimensions."""
    return lambda s, a: -scale * ((a - centre) ** 2).sum(dim=-1)


def _t(x):
    return torch.as_tensor(np.asarray(x, dtype=float), dtype=torch.float64)





@check("tanh_log_det equals log(1 - tanh(u)^2) summed over dimensions, and stays finite at |u| = 20 in float32")
def test_tanh_log_det(sub):
    tanh_log_det = has(sub, "tanh_log_det")
    close(tanh_log_det(_t([[0.5]])).numpy(), [math.log(1 - math.tanh(0.5) ** 2)], atol=1e-12, label="u = 0.5")
    u = torch.linspace(-5, 5, 101, dtype=torch.float64).reshape(-1, 1).repeat(1, 2)
    naive = torch.log(1 - torch.tanh(u) ** 2).sum(-1)
    close(tanh_log_det(u).numpy(), naive.numpy(), atol=1e-8, label="against the naive formula for |u| <= 5")
    big = tanh_log_det(torch.tensor([[20.0], [-20.0]], dtype=torch.float32))
    if not torch.isfinite(big).all():
        fail(
            "tanh_log_det returned -inf at u = +-20 in float32, where tanh(u) rounds to 1. Use "
            "log(1 - tanh(u)^2) = 2 * (log 2 - u - softplus(-2u))."
        )
    want = 2 * (math.log(2) - 20 - math.log1p(math.exp(-40)))
    close(big.double().numpy(), [want, want], atol=1e-4, label="u = +-20 in float32")


@check("sample_action's log_prob matches the density from a numerical change of variables")
def test_change_of_variables(sub):
    sample_action = has(sub, "sample_action")
    mu, sigma, m = 0.3, 0.8, 2.0
    eps = _t(np.linspace(-3.0, 3.0, 25).reshape(-1, 1))
    a, log_prob = sample_action(_t(np.full((25, 1), mu)), _t(np.full((25, 1), math.log(sigma))), eps, m)
    shape(a, (25, 1), label="the action")
    shape(log_prob, (25,), label="log_prob")
    close(a.numpy(), m * np.tanh(mu + sigma * eps.numpy()), atol=1e-12, label="a = max_action * tanh(mean + std * eps)")
    # P(A <= a) = Phi((atanh(a / m) - mu) / sigma), so the density is its derivative in a.
    a_np, h = a.numpy().ravel(), 1e-6
    cdf = lambda x: ndtr((np.arctanh(x / m) - mu) / sigma)
    density = (cdf(a_np + h) - cdf(a_np - h)) / (2 * h)
    close(log_prob.exp().numpy(), density, rtol=1e-5, atol=1e-9,
          label="exp(log_prob) against the derivative of the action's CDF")

    grid = np.linspace(-m + 1e-9, m - 1e-9, 200_001)
    u = np.arctanh(grid / m).reshape(-1, 1)
    _, lp = sample_action(_t(np.full_like(u, mu)), _t(np.full_like(u, math.log(sigma))),
                          _t((u - mu) / sigma), m)
    density = lp.exp().numpy()
    total = float(np.sum(0.5 * (density[1:] + density[:-1]) * np.diff(grid)))
    almost(total, 1.0, tol=1e-4, label="the density integrated over (-max_action, max_action)")

    means = _t([[0.1, -0.4]])
    log_stds = _t([[0.2, -0.3]])
    e = _t([[0.7, -1.2]])
    _, lp2 = sample_action(means, log_stds, e, 1.0)
    base = torch.distributions.Normal(means, log_stds.exp())
    dist = torch.distributions.TransformedDistribution(base, [torch.distributions.transforms.TanhTransform()])
    a2 = torch.tanh(means + log_stds.exp() * e)
    close(lp2.numpy(), dist.log_prob(a2).sum(-1).numpy(), atol=1e-8,
          label="a 2-D action against torch's TransformedDistribution with TanhTransform")


@check("sample_action is reparameterised: gradients flow from the action and log_prob back to mean and log_std")
def test_reparameterised(sub):
    sample_action = has(sub, "sample_action")
    mean = _t([[0.2, -0.5]]).requires_grad_(True)
    log_std = _t([[-0.1, 0.3]]).requires_grad_(True)
    eps = _t([[0.4, 1.1]])
    a, lp = sample_action(mean, log_std, eps, 1.5)
    g = torch.autograd.grad(a.sum() + lp.sum(), [mean, log_std], allow_unused=True)
    if g[0] is None or g[1] is None:
        fail("the action and log_prob must depend on mean and log_std through autograd, not detached samples")

    def f(x):
        a, lp = sample_action(_t(x[:2].reshape(1, 2)), _t(x[2:].reshape(1, 2)), eps, 1.5)
        return float(a.sum() + lp.sum())

    fd = finite_difference(f, np.array([0.2, -0.5, -0.1, 0.3]))
    close(np.concatenate([g[0].numpy().ravel(), g[1].numpy().ravel()]), fd, atol=1e-6,
          label="the gradient of sum(a) + sum(log_prob) against finite differences")


@check("actor_loss is mean(alpha * log_prob - min(Q1, Q2)) at reparameterised actions")
def test_actor_loss_value(sub):
    actor_loss = has(sub, "actor_loss")
    sample_action = has(sub, "sample_action")
    policy = _FixedPolicy(0.1, -0.2, 2.0)
    q1, q2 = _critic(0.5, 1.0), _critic(-0.2, 1.5)
    s = torch.zeros(64, 3, dtype=torch.float64)
    eps = torch.randn(64, 1, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    alpha = 0.3
    got = actor_loss(policy, q1, q2, s, eps, alpha)
    with torch.no_grad():
        mean, log_std = policy(s)
        a, lp = sample_action(mean, log_std, eps, 2.0)
        want = (alpha * lp - torch.minimum(q1(s, a), q2(s, a))).mean()
        only_q1 = (alpha * lp - q1(s, a)).mean()
    if abs(float(want) - float(only_q1)) > 1e-3 and abs(float(got.detach()) - float(only_q1)) < 1e-9:
        fail("actor_loss used Q1 alone. SAC's actor uses the smaller of the two critics.")
    almost(float(got.detach()), float(want), tol=1e-10, label="the loss on 64 fixed noise draws")


@check("actor_loss's reparameterised gradient matches the gradient of the exact expected loss")
def test_actor_loss_gradient(sub):
    actor_loss = has(sub, "actor_loss")
    m, alpha, centre = 2.0, 0.4, 0.6
    q = _critic(centre, 1.0)
    # Stratified noise: normal quantiles at the midpoints of N equal-probability bins.
    n = 20_000
    eps = _t(ndtri((np.arange(n) + 0.5) / n).reshape(-1, 1))
    s = torch.zeros(n, 1, dtype=torch.float64)

    def expected_loss(x):
        mu, sigma = x[0], math.exp(x[1])
        nodes, weights = np.polynomial.hermite_e.hermegauss(120)
        weights = weights / math.sqrt(2 * math.pi)
        u = mu + sigma * nodes
        log_prob = -0.5 * nodes**2 - x[1] - 0.5 * math.log(2 * math.pi) - 2 * (math.log(2) - u - np.logaddexp(0.0, -2 * u)) - math.log(m)
        a = m * np.tanh(u)
        return float(np.sum(weights * (alpha * log_prob + (a - centre) ** 2)))

    for mu0, ls0 in [(0.2, -0.5), (-0.4, 0.1)]:
        policy = _FixedPolicy(mu0, ls0, m)
        loss = actor_loss(policy, q, q, s, eps, alpha)
        grad = torch.autograd.grad(loss, [policy.mu, policy.ls])
        got = np.array([float(grad[0]), float(grad[1])])
        want = finite_difference(expected_loss, np.array([mu0, ls0]), eps=1e-5)
        close(got, want, atol=1e-4, label=f"d loss / d (mean, log_std) at mean {mu0}, log_std {ls0}")


@check("critic_target is r + gamma (1 - terminated) (min(Q1, Q2) - alpha log pi(a' | s'))")
def test_critic_target(sub):
    critic_target = has(sub, "critic_target")
    got = critic_target(_t([1.0, 1.0, 0.0]), _t([0.0, 1.0, 0.0]), _t([3.0, 3.0, -1.0]), _t([2.5, 2.5, 2.0]),
                        _t([-0.8037, -0.8037, 1.0]), 0.99, 0.2)
    shape(got, (3,), label="critic_target output")
    close(got.numpy(), [1 + 0.99 * (2.5 + 0.2 * 0.8037), 1.0, 0.99 * (-1.0 - 0.2)], atol=1e-10,
          label="the lesson's example, a terminal transition, and a case where Q1 is smaller")


@check("temperature_loss has gradient alpha * (H - target) in log_alpha, and sends none to log_prob")
def test_temperature_loss(sub):
    temperature_loss = has(sub, "temperature_loss")
    gen = torch.Generator().manual_seed(4)
    log_prob = (torch.randn(128, generator=gen, dtype=torch.float64) - 0.5).requires_grad_(True)
    target = -1.0
    entropy = -float(log_prob.detach().mean())
    for la in (-1.0, 0.5):
        log_alpha = torch.tensor(la, dtype=torch.float64, requires_grad=True)
        loss = temperature_loss(log_alpha, log_prob, target)
        almost(float(loss.detach()), math.exp(la) * (entropy - target), tol=1e-10,
               label=f"the loss at log_alpha = {la}, which should be alpha * (H - target)")
        loss.backward()
        almost(float(log_alpha.grad), math.exp(la) * (entropy - target), tol=1e-10,
               label=f"d loss / d log_alpha at log_alpha = {la}")
        if log_prob.grad is not None and log_prob.grad.abs().sum() > 0:
            fail("temperature_loss sent gradient into log_prob. Detach it: the temperature step must not move the policy.")
    low = torch.full((8,), 2.0, dtype=torch.float64)
    log_alpha = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    temperature_loss(log_alpha, low, target).backward()
    if not float(log_alpha.grad) < 0:
        fail("with entropy -2 below the target -1, the gradient must be negative, so a descent step raises alpha")


@check("gradient steps on temperature_loss drive a best-responding policy's entropy to the target")
def test_temperature_tuning(sub):
    temperature_loss = has(sub, "temperature_loss")
    # Against Q(a) = -a^2 / 2, the best Gaussian policy at temperature alpha has variance alpha.
    target = 1.0
    alpha_star = math.exp(2 * target) / (2 * math.pi * math.e)
    log_alpha = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([log_alpha], lr=0.05)
    for _ in range(400):
        entropy = 0.5 * math.log(2 * math.pi * math.e * float(log_alpha.detach().exp()))
        loss = temperature_loss(log_alpha, torch.tensor([-entropy], dtype=torch.float64), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    alpha = float(log_alpha.detach().exp())
    if abs(alpha / alpha_star - 1) > 0.01:
        fail(f"alpha settled at {alpha:.5f}, but the entropy matches the target only at {alpha_star:.5f}")
