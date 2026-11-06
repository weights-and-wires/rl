"""Checks for the distributional reinforcement learning exercise."""

import numpy as np
import torch

from rlcheck import almost, check, close, fail, finite_difference, has, shape

torch.set_num_threads(1)


def _np(x):
    return torch.as_tensor(x).detach().numpy().astype(float) if torch.is_tensor(x) else np.asarray(x, dtype=float)


@check("categorical_support spaces n_atoms evenly from v_min to v_max")
def test_support(sub):
    categorical_support = has(sub, "categorical_support")
    close(categorical_support(-2.0, 2.0, 5), [-2.0, -1.0, 0.0, 1.0, 2.0], atol=1e-12, label="(-2, 2, 5)")
    z = _np(categorical_support(-10.0, 10.0, 51))
    shape(z, (51,), label="C51's support")
    close(np.diff(z), np.full(50, 0.4), atol=1e-12, label="C51's atom spacing")


@check("c51_project reproduces the worked example, including the mass clipped at v_max")
def test_project_example(sub):
    c51_project = has(sub, "c51_project")
    support = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    p = np.array([[0.1, 0.2, 0.4, 0.2, 0.1]])
    m = _np(c51_project(p, np.array([1.0]), np.array([0.0]), 0.9, support))
    close(m, [[0.0, 0.08, 0.20, 0.44, 0.28]], atol=1e-12, label="projected probabilities")
    # Clipping at v_max = 2 pulls the mean from 1 + 0.9 * 0 = 1 down to 0.92.
    almost(m[0] @ support, 0.92, 1e-12, label="mean of the projection")


@check("c51_project keeps all the mass when targets land exactly on atoms")
def test_project_integer(sub):
    c51_project = has(sub, "c51_project")
    support = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    p = np.array([[0.1, 0.2, 0.4, 0.2, 0.1], [0.3, 0.1, 0.2, 0.1, 0.3]])
    # r = 1, gamma = 1 shifts every atom by exactly one: b is always an integer.
    m = _np(c51_project(p, np.array([1.0, -1.0]), np.array([0.0, 0.0]), 1.0, support))
    close(m, [[0.0, 0.1, 0.2, 0.4, 0.3], [0.4, 0.2, 0.1, 0.3, 0.0]], atol=1e-12,
          label="projection with integer b (the l == u case)")
    # Terminal with r = 0.5: every atom goes to 0.5, split between 0 and 1.
    m = _np(c51_project(p[:1], np.array([0.5]), np.array([1.0]), 0.9, support))
    close(m, [[0.0, 0.0, 0.5, 0.5, 0.0]], atol=1e-12, label="terminal step with r = 0.5")


@check("c51_project conserves mass always, and the mean r + gamma * mean wherever nothing is clipped")
def test_project_moments(sub):
    c51_project = has(sub, "c51_project")
    rng = np.random.default_rng(0)
    support = np.linspace(-10.0, 10.0, 51)
    p = rng.dirichlet(np.ones(51), size=64)
    r = rng.uniform(-0.9, 0.9, size=64)  # |r| + 0.9 * 10 <= 10, so nothing is clipped
    term = (rng.random(64) < 0.25).astype(float)
    m = _np(c51_project(p, r, term, 0.9, support))
    shape(m, (64, 51), label="projection of a batch of 64")
    if np.any(m < -1e-12):
        fail(f"projected probabilities must be nonnegative; the smallest is {m.min():.3g}")
    close(m.sum(axis=1), np.ones(64), atol=1e-12, label="total mass per row")
    want = r + 0.9 * (1 - term) * (p @ support)
    close(m @ support, want, atol=1e-10, label="projected mean against r + gamma * (1 - term) * mean")
    big = _np(c51_project(p, rng.uniform(-30, 30, size=64), np.zeros(64), 0.99, support))
    close(big.sum(axis=1), np.ones(64), atol=1e-12, label="total mass when many atoms are clipped")


@check("c51_loss is the mean cross-entropy, with gradient (softmax(logits) - m) / B")
def test_c51_loss(sub):
    c51_loss = has(sub, "c51_loss")
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(3, 5))
    m = rng.dirichlet(np.ones(5), size=3)
    log_p = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    t = torch.tensor(logits, requires_grad=True)
    loss = c51_loss(t, m)
    almost(float(loss), float(-(m * log_p).mean(axis=0).sum()), 1e-10, label="loss value")
    loss.backward()
    softmax = np.exp(log_p)
    close(t.grad.numpy(), (softmax - m) / 3, atol=1e-10, label="gradient against (softmax - m) / B")
    fd = finite_difference(lambda x: float(c51_loss(torch.tensor(x), m)), logits.copy())
    close(t.grad.numpy(), fd, atol=1e-7, label="gradient against finite differences")


@check("quantile_midpoints are (2i - 1) / 2N, and quantile_huber_loss matches the hand example")
def test_quantile_loss_values(sub):
    quantile_midpoints = has(sub, "quantile_midpoints")
    quantile_huber_loss = has(sub, "quantile_huber_loss")
    close(quantile_midpoints(4), [0.125, 0.375, 0.625, 0.875], atol=1e-12, label="quantile_midpoints(4)")
    pred = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    target = torch.tensor([[0.5, 2.0]], dtype=torch.float64)
    almost(float(quantile_huber_loss(pred, target, [0.25, 0.75], 1.0)), 0.40625, 1e-12,
           label="kappa = 1 hand example")
    almost(float(quantile_huber_loss(pred, target, [0.25, 0.75], 0.0)), 0.75, 1e-12,
           label="kappa = 0 hand example")


@check("quantile_huber_loss's gradient matches finite differences, and at kappa = 0 equals P(target < theta) - tau")
def test_quantile_loss_grad(sub):
    quantile_huber_loss = has(sub, "quantile_huber_loss")
    rng = np.random.default_rng(3)
    B, N, M = 2, 3, 5
    pred0 = rng.normal(size=(B, N))
    target = torch.tensor(rng.normal(size=(B, M)))
    taus = (2 * np.arange(1, N + 1) - 1) / (2 * N)
    t = torch.tensor(pred0, requires_grad=True)
    quantile_huber_loss(t, target, taus, 1.0).backward()
    fd = finite_difference(lambda x: float(quantile_huber_loss(torch.tensor(x), target, taus, 1.0)),
                           pred0.copy())
    close(t.grad.numpy(), fd, atol=1e-6, label="kappa = 1 gradient against finite differences")
    t = torch.tensor(pred0, requires_grad=True)
    quantile_huber_loss(t, target, taus, 0.0).backward()
    below = (target.numpy()[:, None, :] < pred0[:, :, None]).mean(axis=2)
    close(t.grad.numpy(), (below - taus[None, :]) / B, atol=1e-12,
          label="kappa = 0 gradient against (fraction of targets below theta_i - tau_i) / B")


@check("Quantile regression on 10,000 exponential samples recovers the quantiles at the midpoints")
def test_quantile_regression(sub):
    quantile_huber_loss = has(sub, "quantile_huber_loss")
    quantile_midpoints = has(sub, "quantile_midpoints")
    taus = np.asarray(quantile_midpoints(4), dtype=float)
    torch.manual_seed(0)
    samples = torch.distributions.Exponential(torch.tensor(1.0, dtype=torch.float64)).sample((1, 10000))
    theta = torch.zeros(1, 4, dtype=torch.float64, requires_grad=True)
    optimiser = torch.optim.Adam([theta], lr=0.05)
    for k in range(800):
        for group in optimiser.param_groups:
            group["lr"] = 0.05 * (1 - k / 800)
        loss = quantile_huber_loss(theta, samples, taus, 0.0)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    got = theta.detach().numpy()[0]
    empirical = np.quantile(samples.numpy()[0], taus)
    close(got, empirical, atol=0.01, label="learned locations against the sample quantiles")
    close(got, -np.log(1 - taus), atol=0.1, label="learned locations against the true quantiles -log(1 - tau)")
