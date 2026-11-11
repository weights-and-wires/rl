"""A tiny diffusion model, trained as a policy.

The toy model below denoises in one dimension with T = 10 steps. Its
pretrained noise prediction is exact for data made of two bumps, at -1.5 and
+1.5. You treat each denoising step as an action, score the final sample,
and train the model with DDPO's policy gradient. You also write the
Diffusion-DPO loss, which learns from preference pairs instead of a reward.

Step t runs from x_t to x_{t-1}. A trajectory `xs` has shape (T + 1, n):
xs[0] is x_T, pure noise, and xs[T] is x_0, the sample. Row k of anything
per-step belongs to the step from xs[k] to xs[k + 1], which is step
t = T - k. Every tensor is float64.
"""

import math

import torch
import torch.nn.functional as F

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
    """Elementwise log-density of x under N(mean, std^2).

    Args:
        x: values, any shape.
        mean: means, broadcastable to x.
        std: standard deviations, a float or a tensor broadcastable to x.

    Returns:
        log N(x; mean, std^2), the broadcast shape of the inputs.
    """
    raise NotImplementedError("gaussian_logprob")


def ddpm_mean(x_t: torch.Tensor, eps: torch.Tensor, t: int) -> torch.Tensor:
    """The mean of one DDPM denoising step, from x_t and a noise prediction.

    Args:
        x_t: the current noisy values, shape (n,).
        eps: the predicted noise at x_t, shape (n,).
        t: the step, an integer in 1..T. Use BETAS[t - 1], ALPHAS[t - 1], and
            ALPHA_BARS[t - 1].

    Returns:
        The mean of x_{t-1}, shape (n,).
    """
    raise NotImplementedError("ddpm_mean")


def rollout(model: ToyDenoiser, x_T: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Run the sampler from x_T to x_0 with the noise you are given.

    Step k (t = T - k) sets x_{t-1} = ddpm_mean(x_t, model.eps(x_t, t), t)
    + SIGMAS[t - 1] * noise[k].

    Args:
        model: a ToyDenoiser.
        x_T: the starting noise, shape (n,).
        noise: the noise for each step, shape (T, n).

    Returns:
        The trajectory, shape (T + 1, n), with xs[0] = x_T and xs[T] = x_0.
    """
    raise NotImplementedError("rollout")


def trajectory_logprob(model: ToyDenoiser, xs: torch.Tensor) -> torch.Tensor:
    """Per-step log-probabilities of a stored trajectory under the model.

    Treat xs as fixed data: gradients flow only through the model's
    parameters, never through xs.

    Args:
        model: a ToyDenoiser.
        xs: a trajectory, shape (T + 1, n).

    Returns:
        Shape (T, n). Row k is log p_theta(xs[k + 1] | xs[k]) for step
        t = T - k, a Gaussian with mean ddpm_mean(...) and std SIGMAS[t - 1].
    """
    raise NotImplementedError("trajectory_logprob")


def ddpo_loss(logp: torch.Tensor, logp_old: torch.Tensor, advantages: torch.Tensor,
              clip_eps: float) -> torch.Tensor:
    """DDPO's clipped surrogate, as a loss to minimise.

    Args:
        logp: per-step log-probabilities under the current model, shape (T, n).
        logp_old: the same under the model that sampled, shape (T, n).
        advantages: one advantage per trajectory, shape (n,), shared by all
            of that trajectory's steps.
        clip_eps: the clip range. Ratios are clipped to [1 - clip_eps, 1 + clip_eps].

    Returns:
        A scalar: minus the mean over steps and trajectories of
        min(ratio * A, clip(ratio) * A).
    """
    raise NotImplementedError("ddpo_loss")


def diffusion_dpo_loss(err_w: torch.Tensor, err_w_ref: torch.Tensor, err_l: torch.Tensor,
                       err_l_ref: torch.Tensor, beta: float) -> torch.Tensor:
    """The Diffusion-DPO loss on a batch of preference pairs.

    Args:
        err_w, err_w_ref: squared denoising error of the model and of the
            reference on the winning image, shape (n,).
        err_l, err_l_ref: the same on the losing image, shape (n,).
        beta: the folded scale beta * T * omega.

    Returns:
        A scalar: -mean(log sigmoid(-beta * ((err_w - err_w_ref) - (err_l - err_l_ref)))).
    """
    raise NotImplementedError("diffusion_dpo_loss")


def ddpo_train(model: ToyDenoiser, reward_fn, iterations: int, batch_size: int,
               epochs: int, lr: float, clip_eps: float,
               generator: torch.Generator) -> list[float]:
    """Train the model with DDPO and return the mean reward of each batch.

    Each iteration draws batch_size trajectories from the current model with
    torch.randn(..., generator=generator), records their log-probabilities,
    normalises the rewards over the batch to get advantages, and takes
    `epochs` Adam steps (learning rate lr) on ddpo_loss.

    Args:
        model: a ToyDenoiser, trained in place.
        reward_fn: maps x_0, shape (n,), to rewards, shape (n,).
        iterations: the number of sample-then-update rounds.
        batch_size: trajectories per round.
        epochs: gradient steps per round on the same trajectories.
        lr: the Adam learning rate.
        clip_eps: the clip range for ddpo_loss.
        generator: the random source for every sample.

    Returns:
        A list of `iterations` floats, the mean reward of each round's
        batch, measured before that round's update.
    """
    raise NotImplementedError("ddpo_train")


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

    # Diffusion-DPO at the reference: the loss starts at log 2.
    e = torch.tensor([0.3, 0.7])
    print(f"diffusion_dpo_loss at the reference = {float(diffusion_dpo_loss(e, e, e + 0.2, e + 0.2, 1.0)):.4f}")
    print(f"diffusion_dpo_loss on the worked pair = "
          f"{float(diffusion_dpo_loss(torch.tensor([0.3]), torch.tensor([0.5]), torch.tensor([0.6]), torch.tensor([0.4]), 1.0)):.4f}")
