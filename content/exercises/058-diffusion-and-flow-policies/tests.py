"""Checks for the diffusion-and-flow-policies exercise."""

import math

import torch

from rlcheck import almost, check, close, fail, finite_difference, has, shape

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)


def _toy(sub):
    """The shared toy pieces from the reader's module."""
    return (
        has(sub, "T"),
        has(sub, "BETAS"),
        has(sub, "ALPHAS"),
        has(sub, "ALPHA_BARS"),
        has(sub, "SIGMAS"),
        has(sub, "ToyDenoiser"),
    )


def _reference_rollout(sub, model, x_T, noise):
    """An independent rollout, written from the two-coefficient posterior mean."""
    T, BETAS, ALPHAS, ALPHA_BARS, SIGMAS, _ = _toy(sub)
    xs = [x_T]
    x = x_T
    for k in range(T):
        t = T - k
        abar = ALPHA_BARS[t - 1]
        abar_prev = ALPHA_BARS[t - 2] if t > 1 else torch.tensor(1.0)
        eps = model.eps(x, t)
        x0_hat = (x - torch.sqrt(1 - abar) * eps) / torch.sqrt(abar)
        mean = (torch.sqrt(abar_prev) * BETAS[t - 1] / (1 - abar)) * x0_hat + (
            torch.sqrt(ALPHAS[t - 1]) * (1 - abar_prev) / (1 - abar)
        ) * x
        x = mean + SIGMAS[t - 1] * noise[k]
        xs.append(x)
    return torch.stack(xs)


@check("gaussian_logprob gives 0.1905 for x = 0.3 under N(0.5, 0.2^2) and matches torch.distributions")
def test_gaussian_logprob(sub):
    lp = has(sub, "gaussian_logprob")
    got = lp(torch.tensor(0.3), torch.tensor(0.5), 0.2)
    almost(float(got), -0.5 - math.log(0.2) - 0.5 * math.log(2 * math.pi), 1e-10,
           label="log N(0.3; 0.5, 0.2^2)")
    g = torch.Generator().manual_seed(0)
    x = torch.randn(50, generator=g)
    mu = torch.randn(50, generator=g)
    sd = torch.rand(50, generator=g) + 0.1
    want = torch.distributions.Normal(mu, sd).log_prob(x)
    close(lp(x, mu, sd), want, atol=1e-10, label="gaussian_logprob on random inputs")


@check("ddpm_mean equals the posterior mean of q(x_{t-1} | x_t, x_0) at every step")
def test_ddpm_mean(sub):
    T, BETAS, ALPHAS, ALPHA_BARS, _, _ = _toy(sub)
    ddpm_mean = has(sub, "ddpm_mean")
    g = torch.Generator().manual_seed(1)
    x0 = torch.randn(20, generator=g)
    eps = torch.randn(20, generator=g)
    for t in range(1, T + 1):
        abar = ALPHA_BARS[t - 1]
        abar_prev = ALPHA_BARS[t - 2] if t > 1 else torch.tensor(1.0)
        x_t = torch.sqrt(abar) * x0 + torch.sqrt(1 - abar) * eps
        want = (torch.sqrt(abar_prev) * BETAS[t - 1] / (1 - abar)) * x0 + (
            torch.sqrt(ALPHAS[t - 1]) * (1 - abar_prev) / (1 - abar)
        ) * x_t
        close(ddpm_mean(x_t, eps, t), want, atol=1e-10, label=f"ddpm_mean at t = {t}")


@check("rollout runs the denoising MDP from x_T to x_0 with the given noise")
def test_rollout(sub):
    T, *_, ToyDenoiser = _toy(sub)
    rollout = has(sub, "rollout")
    model = ToyDenoiser()
    with torch.no_grad():
        model.w += 0.1
        model.b -= 0.2
    g = torch.Generator().manual_seed(2)
    x_T = torch.randn(7, generator=g)
    noise = torch.randn(T, 7, generator=g)
    with torch.no_grad():
        xs = rollout(model, x_T, noise)
        want = _reference_rollout(sub, model, x_T, noise)
    shape(xs, (T + 1, 7), label="rollout output")
    close(xs[0], x_T, atol=0, label="xs[0] must be x_T")
    close(xs, want, atol=1e-10, label="rollout against an independent posterior-mean rollout")


@check("trajectory_logprob of a fresh rollout equals log N(noise; 0, 1) - log sigma_t at every step")
def test_trajectory_logprob(sub):
    T, *_, SIGMAS, ToyDenoiser = _toy(sub)
    traj = has(sub, "trajectory_logprob")
    model = ToyDenoiser()
    with torch.no_grad():
        model.w -= 0.05
        model.b += 0.3
    g = torch.Generator().manual_seed(3)
    x_T = torch.randn(9, generator=g)
    noise = torch.randn(T, 9, generator=g)
    with torch.no_grad():
        xs = _reference_rollout(sub, model, x_T, noise)
        got = traj(model, xs)
    shape(got, (T, 9), label="trajectory_logprob output")
    steps = torch.arange(T, 0, -1)
    sig = SIGMAS[steps - 1][:, None]
    want = -0.5 * noise**2 - torch.log(sig) - 0.5 * math.log(2 * math.pi)
    close(got, want, atol=1e-9,
          label="row k is log p(x_{T-k-1} | x_{T-k}); it depends only on the noise drawn")


@check("ddpo_loss clips the per-step ratio the PPO way and matches hand values")
def test_ddpo_loss(sub):
    ddpo_loss = has(sub, "ddpo_loss")
    # Two steps, three trajectories.
    logp_old = torch.zeros(2, 3)
    logp = torch.tensor([[0.1905 - 0.4093, math.log(1.5), math.log(1.5)],
                         [0.0, math.log(0.5), math.log(0.5)]])
    adv = torch.tensor([1.0, 1.0, -1.0])
    got = float(ddpo_loss(logp, logp_old, adv, 0.2))
    # Per-element objectives: min(ratio * A, clip(ratio) * A).
    terms = [math.exp(0.1905 - 0.4093), 1.2, -1.5, 1.0, 0.5, -0.8]
    almost(got, -sum(terms) / 6, 1e-10, label="ddpo_loss on the hand example")

    # At the sampling policy, the gradient is the REINFORCE one: -A / (T n).
    g = torch.Generator().manual_seed(4)
    old = torch.randn(4, 5, generator=g)
    a = torch.randn(5, generator=g)
    new = old.clone().requires_grad_(True)
    ddpo_loss(new, old, a, 0.2).backward()
    close(new.grad, (-a / 20).expand(4, 5), atol=1e-12,
          label="d loss / d logp at ratio 1")


@check("diffusion_dpo_loss is log 2 at the reference, 0.5130 on the worked pair, and its gradient matches finite differences")
def test_diffusion_dpo_loss(sub):
    loss = has(sub, "diffusion_dpo_loss")
    e = torch.tensor([0.3, 0.7, 1.1])
    almost(float(loss(e, e, e + 0.2, e + 0.2, 5.0)), math.log(2), 1e-12, label="loss when model = reference")
    almost(float(loss(torch.tensor([0.30]), torch.tensor([0.50]), torch.tensor([0.60]),
                      torch.tensor([0.40]), 1.0)),
           -math.log(1 / (1 + math.exp(-0.4))), 1e-10, label="worked pair")
    g = torch.Generator().manual_seed(5)
    x0 = torch.rand(16, generator=g)

    def f(x):
        x = torch.as_tensor(x)
        return float(loss(x[0:4], x[4:8], x[8:12], x[12:16], 2.0))

    var = x0.clone().requires_grad_(True)
    loss(var[0:4], var[4:8], var[8:12], var[12:16], 2.0).backward()
    close(var.grad, torch.as_tensor(finite_difference(f, x0.numpy().copy())), atol=1e-6, label="gradient against finite differences")


@check("ddpo_train lifts P(x_0 > 1) from about 0.44 to above 0.9 in 20 iterations")
def test_ddpo_train(sub):
    T, *_, ToyDenoiser = _toy(sub)
    rollout = has(sub, "rollout")
    reward = has(sub, "reward")
    ddpo_train = has(sub, "ddpo_train")
    model = ToyDenoiser()
    gen = torch.Generator().manual_seed(7)
    history = ddpo_train(model, reward, iterations=20, batch_size=256, epochs=4,
                         lr=0.02, clip_eps=0.2, generator=gen)
    if len(history) != 20:
        fail(f"ddpo_train should return 20 mean rewards, one per iteration, got {len(history)}")
    g = torch.Generator().manual_seed(8)
    with torch.no_grad():
        xs = rollout(model, torch.randn(4000, generator=g), torch.randn(T, 4000, generator=g))
    frac = float((xs[-1] > 1).double().mean())
    if frac < 0.9:
        fail(f"after training, P(x_0 > 1) should exceed 0.9, got {frac:.3f}")
    if history[-1] <= history[0] + 0.3:
        fail(f"the mean reward should rise by more than 0.3, went {history[0]:.3f} -> {history[-1]:.3f}")
