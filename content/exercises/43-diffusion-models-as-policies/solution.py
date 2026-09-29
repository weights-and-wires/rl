"""Reference solution: a tiny diffusion model trained as a policy."""

import math

import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# The toy diffusion model. This block is identical in starter.py.
# ---------------------------------------------------------------------------

T = 10
BETAS = torch.linspace(0.02, 0.6, T)          # beta_t for t = 1..T, at index t - 1
ALPHAS = 1.0 - BETAS
ALPHA_BARS = torch.cumprod(ALPHAS, dim=0)
SIGMAS = torch.sqrt(BETAS)                    # the DDPM choice sigma_t^2 = beta_t

MIX_WEIGHTS = torch.tensor([0.5, 0.5])
MIX_MEANS = torch.tensor([-1.5, 1.5])
MIX_STD = 0.3


def eps_pretrained(x: torch.Tensor, t: int) -> torch.Tensor:
    """The exact noise prediction for the two-bump data, at step t in 1..T.

    x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps, so x_t is a mixture of
    Gaussians and eps*(x, t) = -sqrt(1 - abar_t) * d/dx log p_t(x).
    """
    abar = ALPHA_BARS[t - 1]
    means = torch.sqrt(abar) * MIX_MEANS
    var = abar * MIX_STD**2 + 1.0 - abar
    diff = x[..., None] - means
    resp = torch.softmax(torch.log(MIX_WEIGHTS) - 0.5 * diff**2 / var, dim=-1)
    score = -(resp * diff).sum(-1) / var
    return -torch.sqrt(1.0 - abar) * score


class ToyDenoiser:
    """eps_theta(x, t) = eps_pretrained(x, t) + w[t-1] * x + b[t-1].

    Twenty trainable numbers, all zero at the start, so the untrained model
    is exactly the pretrained one.
    """

    def __init__(self) -> None:
        self.w = torch.zeros(T, requires_grad=True)
        self.b = torch.zeros(T, requires_grad=True)

    def parameters(self) -> list[torch.Tensor]:
        return [self.w, self.b]

    def eps(self, x: torch.Tensor, t: int) -> torch.Tensor:
        return eps_pretrained(x, t) + self.w[t - 1] * x + self.b[t - 1]


def reward(x0: torch.Tensor) -> torch.Tensor:
    """A smooth "land to the right of 1" score in (0, 1)."""
    return torch.sigmoid(4.0 * (x0 - 1.0))


# ---------------------------------------------------------------------------
# Your functions.
# ---------------------------------------------------------------------------


def gaussian_logprob(x: torch.Tensor, mean: torch.Tensor, std) -> torch.Tensor:
    std = torch.as_tensor(std, dtype=torch.float64)
    z = (x - mean) / std
    return -0.5 * z**2 - torch.log(std) - 0.5 * math.log(2.0 * math.pi)


def ddpm_mean(x_t: torch.Tensor, eps: torch.Tensor, t: int) -> torch.Tensor:
    beta = BETAS[t - 1]
    return (x_t - beta / torch.sqrt(1.0 - ALPHA_BARS[t - 1]) * eps) / torch.sqrt(ALPHAS[t - 1])


def _step(model: ToyDenoiser, x: torch.Tensor, t: int, noise: torch.Tensor) -> torch.Tensor:
    return ddpm_mean(x, model.eps(x, t), t) + SIGMAS[t - 1] * noise


def rollout(model, x_T, noise, grad_steps=None):
    xs = [x_T]
    x = x_T
    first_graded = 0 if grad_steps is None else T - grad_steps
    for k in range(T):
        t = T - k
        if k < first_graded:
            with torch.no_grad():
                x = _step(model, x, t, noise[k])
        else:
            if k == first_graded:
                x = x.detach()                # DRaFT's stop_grad
            x = _step(model, x, t, noise[k])
        xs.append(x)
    return torch.stack(xs)


def trajectory_logprob(model, xs):
    xs = xs.detach()
    rows = []
    for k in range(T):
        t = T - k
        mean = ddpm_mean(xs[k], model.eps(xs[k], t), t)
        rows.append(gaussian_logprob(xs[k + 1], mean, SIGMAS[t - 1]))
    return torch.stack(rows)


def ddpo_loss(logp, logp_old, advantages, clip_eps):
    ratio = torch.exp(logp - logp_old)
    adv = advantages[None, :]
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    return -torch.minimum(unclipped, clipped).mean()


def ddpo_train(model, reward_fn, iterations, batch_size, epochs, lr, clip_eps, generator):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for _ in range(iterations):
        with torch.no_grad():
            x_T = torch.randn(batch_size, generator=generator)
            noise = torch.randn(T, batch_size, generator=generator)
            xs = rollout(model, x_T, noise)
            logp_old = trajectory_logprob(model, xs)
            r = reward_fn(xs[-1])
            adv = (r - r.mean()) / (r.std() + 1e-8)
        history.append(float(r.mean()))
        for _ in range(epochs):
            loss = ddpo_loss(trajectory_logprob(model, xs), logp_old, adv, clip_eps)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return history


if __name__ == "__main__":
    gen = torch.Generator().manual_seed(0)

    # One denoising step by hand.
    lp = gaussian_logprob(torch.tensor(0.3), torch.tensor(0.5), 0.2)
    print(f"log N(0.3; 0.5, 0.2^2) = {float(lp):.4f}")

    model = ToyDenoiser()
    with torch.no_grad():
        xs = rollout(model, torch.randn(4000, generator=gen), torch.randn(T, 4000, generator=gen))
    print(f"pretrained model: P(x_0 > 1) = {float((xs[-1] > 1).double().mean()):.3f}")

    history = ddpo_train(model, reward, iterations=20, batch_size=256, epochs=4,
                         lr=0.02, clip_eps=0.2, generator=gen)
    print("DDPO mean reward by iteration:")
    for i in range(0, len(history), 4):
        print(f"  iteration {i:2d}: {history[i]:.3f}")

    with torch.no_grad():
        xs = rollout(model, torch.randn(4000, generator=gen), torch.randn(T, 4000, generator=gen))
    print(f"after DDPO: P(x_0 > 1) = {float((xs[-1] > 1).double().mean()):.3f}")

    # Reward backpropagation: one gradient through the whole chain.
    fresh = ToyDenoiser()
    xs = rollout(fresh, torch.randn(1000, generator=gen), torch.randn(T, 1000, generator=gen))
    reward(xs[-1]).mean().backward()
    print("d E[r] / d b_t by reward backprop, t = 1..T:")
    print("  " + " ".join(f"{g:+.3f}" for g in fresh.b.grad.tolist()))
