"""Checks for the diffusion-models-as-policies exercise."""

import math

import torch

from rlcheck import almost, check, close, fail, has, shape

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


@check("The DDPO score-function gradient agrees with the reward-backprop gradient within 4 standard errors")
def test_score_function_matches_pathwise(sub):
    T, *_, ToyDenoiser = _toy(sub)
    rollout = has(sub, "rollout")
    traj = has(sub, "trajectory_logprob")
    reward = has(sub, "reward")
    g = torch.Generator().manual_seed(5)
    sf, pw = [], []
    for _ in range(10):
        model = ToyDenoiser()
        x_T = torch.randn(4000, generator=g)
        noise = torch.randn(T, 4000, generator=g)
        xs = rollout(model, x_T, noise)
        r = reward(xs[-1])
        (grad_b,) = torch.autograd.grad(r.mean(), [model.b])
        pw.append(float(grad_b.sum()))
        r = r.detach()
        surrogate = (traj(model, xs.detach()).sum(0) * (r - r.mean())).mean()
        (grad_b,) = torch.autograd.grad(surrogate, [model.b])
        sf.append(float(grad_b.sum()))
    sf, pw = torch.tensor(sf), torch.tensor(pw)
    se = math.sqrt(float(sf.var() + pw.var()) / 10)
    gap = abs(float(sf.mean() - pw.mean()))
    # The derivative of E[r] as every b_t rises together. Pushing eps up pulls
    # samples left, so it is negative.
    if float(pw.mean()) >= 0:
        fail(f"the pathwise derivative should be negative, got {float(pw.mean()):.4f}")
    if gap > 4 * se:
        fail(
            f"score-function estimate {float(sf.mean()):.4f} and pathwise estimate "
            f"{float(pw.mean()):.4f} differ by {gap:.4f}, more than 4 standard errors ({4 * se:.4f})"
        )


@check("rollout with grad_steps = K backpropagates through only the last K steps (DRaFT-K)")
def test_draft_k(sub):
    T, *_, ToyDenoiser = _toy(sub)
    rollout = has(sub, "rollout")
    reward = has(sub, "reward")
    g = torch.Generator().manual_seed(6)
    x_T = torch.randn(200, generator=g)
    noise = torch.randn(T, 200, generator=g)

    def grads(k):
        model = ToyDenoiser()
        xs = rollout(model, x_T, noise, grad_steps=k)
        reward(xs[-1]).mean().backward()
        return model.b.grad.clone()

    full = grads(None)
    # Finite differences of the same deterministic function of b.
    eps = 1e-6
    fd = torch.zeros(T)
    for i in range(T):
        vals = []
        for sgn in (1, -1):
            model = ToyDenoiser()
            with torch.no_grad():
                model.b[i] += sgn * eps
                vals.append(float(reward(rollout(model, x_T, noise)[-1]).mean()))
        fd[i] = (vals[0] - vals[1]) / (2 * eps)
    close(full, fd, rtol=1e-4, atol=1e-8, label="full reward-backprop gradient vs finite differences")
    close(grads(T), full, atol=1e-12, label="grad_steps = T must equal full backprop")
    one = grads(1)
    close(one[0], full[0], atol=1e-12, label="grad_steps = 1: gradient on b_1, the last step")
    close(one[1:], torch.zeros(T - 1), atol=0, label="grad_steps = 1: gradient on b_2 .. b_T")
    three = grads(3)
    close(three[:3], full[:3], atol=1e-12, label="grad_steps = 3: gradient on b_1 .. b_3")
    close(three[3:], torch.zeros(T - 3), atol=0, label="grad_steps = 3: gradient on b_4 .. b_T")


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
