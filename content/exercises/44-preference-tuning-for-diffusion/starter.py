"""Path KL, Diffusion-DPO, and reward hacking on a toy diffusion model.

You first measure KL divergences on a four-step linear-Gaussian chain, where
every answer is exact: the KL between whole sampling paths and the far
smaller KL between final samples. Then you return to the 10-step toy
diffusion model from the previous lesson. You train it from preference
pairs with Diffusion-DPO, and you watch reward backpropagation hack a proxy
reward unless a path-KL penalty holds it back.

A trajectory `xs` has shape (T + 1, n): xs[0] is x_T and xs[T] is x_0. Row
k of anything per-step belongs to diffusion step t = T - k. Every tensor is
float64.
"""

import math

import torch
import torch.nn.functional as F

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# The toy diffusion model from the previous lesson. Identical in starter.py.
# ---------------------------------------------------------------------------

T = 10
BETAS = torch.linspace(0.02, 0.6, T)          # beta_t for t = 1..T, at index t - 1
ALPHAS = 1.0 - BETAS
ALPHA_BARS = torch.cumprod(ALPHAS, dim=0)
SIGMAS = torch.sqrt(BETAS)

MIX_WEIGHTS = torch.tensor([0.5, 0.5])
MIX_MEANS = torch.tensor([-1.5, 1.5])
MIX_STD = 0.3


def eps_pretrained(x, t):
    """The exact noise prediction for the two-bump data.

    t is an int in 1..T, or an integer tensor broadcastable to x.
    """
    abar = torch.as_tensor(ALPHA_BARS[torch.as_tensor(t) - 1])[..., None]
    means = torch.sqrt(abar) * MIX_MEANS
    var = abar * MIX_STD**2 + 1.0 - abar
    diff = x[..., None] - means
    resp = torch.softmax(torch.log(MIX_WEIGHTS) - 0.5 * diff**2 / var, dim=-1)
    score = -(resp * diff / var).sum(-1)
    return -torch.sqrt(1.0 - abar[..., 0]) * score


class ToyDenoiser:
    """eps_theta(x, t) = eps_pretrained(x, t) + w[t-1] * x + b[t-1]."""

    def __init__(self) -> None:
        self.w = torch.zeros(T, requires_grad=True)
        self.b = torch.zeros(T, requires_grad=True)

    def parameters(self):
        return [self.w, self.b]

    def eps(self, x, t):
        t = torch.as_tensor(t)
        return eps_pretrained(x, t) + self.w[t - 1] * x + self.b[t - 1]


def ddpm_mean(x_t, eps, t):
    t = torch.as_tensor(t)
    return (x_t - BETAS[t - 1] / torch.sqrt(1.0 - ALPHA_BARS[t - 1]) * eps) / torch.sqrt(ALPHAS[t - 1])


def rollout(model, x_T, noise):
    """Differentiable sampler: xs[0] = x_T, xs[T] = x_0, shape (T + 1, n)."""
    xs = [x_T]
    x = x_T
    for k in range(T):
        t = T - k
        x = ddpm_mean(x, model.eps(x, t), t) + SIGMAS[t - 1] * noise[k]
        xs.append(x)
    return torch.stack(xs)


def sample(model, n, generator):
    """n trajectories from the model, without gradient, shape (T + 1, n)."""
    with torch.no_grad():
        return rollout(model, torch.randn(n, generator=generator),
                       torch.randn(T, n, generator=generator))


def true_reward(x0):
    """What you actually want: samples near 2."""
    return -((x0 - 2.0) ** 2)


def proxy_reward(x0):
    """A reward model that learned "further right is better" from reference samples."""
    return x0


# ---------------------------------------------------------------------------
# Your functions.
# ---------------------------------------------------------------------------


def gaussian_kl(mu1, std1, mu2, std2):
    """KL(N(mu1, std1^2) || N(mu2, std2^2)), elementwise.

    Args:
        mu1, std1: mean and standard deviation of the first Gaussian.
        mu2, std2: mean and standard deviation of the second Gaussian.
            Each argument is a float or a tensor; they broadcast together.

    Returns:
        A tensor of the broadcast shape.
    """
    raise NotImplementedError("gaussian_kl")


def chain_path_kl(a_theta, b_theta, a_ref, b_ref, s):
    """Exact KL between the path distributions of two linear-Gaussian chains.

    Both chains start from x ~ N(0, 1) and take K steps in the order given,
    x_next = a[k] * x + b[k] + s[k] * z, with z ~ N(0, 1). They share s.

    Args:
        a_theta, b_theta: the fine-tuned chain's coefficients, K floats each.
        a_ref, b_ref: the reference chain's coefficients, K floats each.
        s: the shared step standard deviations, K floats.

    Returns:
        A scalar tensor: the sum over steps of the expected per-step KL,
        with the expectation over x drawn from the fine-tuned chain.
    """
    raise NotImplementedError("chain_path_kl")


def chain_marginal_kl(a_theta, b_theta, a_ref, b_ref, s):
    """Exact KL between the two chains' final samples.

    Both final samples are Gaussian. Propagate each chain's mean and variance
    through its K steps and compare the two final Gaussians.

    Args:
        The same as chain_path_kl.

    Returns:
        A scalar tensor, KL(final fine-tuned sample || final reference sample).
    """
    raise NotImplementedError("chain_marginal_kl")


def chain_path_kl_mc(a_theta, b_theta, a_ref, b_ref, s, n, generator):
    """Monte Carlo estimate of the path KL from summed per-step log-ratios.

    Draw n paths from the fine-tuned chain with torch.randn(..., generator=
    generator): first the n starting values, then n step noises per step.

    Args:
        The first five are the same as chain_path_kl.
        n: the number of paths.
        generator: a torch.Generator.

    Returns:
        A tuple (estimate, standard_error) of floats: the mean over paths of
        the summed log p_theta / p_ref of each step, and its standard error.
    """
    raise NotImplementedError("chain_path_kl_mc")


def ddpm_step_kl(model, xs):
    """Per-step KL between the model's denoising step and the pretrained one.

    Args:
        model: a ToyDenoiser.
        xs: a trajectory, shape (T + 1, n), usually sampled from the model.

    Returns:
        Shape (T, n). Row k is KL(p_theta(. | xs[k]) || p_pretrained(. | xs[k]))
        for step t = T - k. Both steps share SIGMAS[t - 1]. Keep the graph,
        so the result can be differentiated.
    """
    raise NotImplementedError("ddpm_step_kl")


def denoising_errors(model, x0, t, eps):
    """Squared noise-prediction errors of the model and the pretrained network.

    Noise x0 to x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps, then score both
    networks on that same x_t.

    Args:
        model: a ToyDenoiser.
        x0: clean samples, shape (n,).
        t: an integer tensor of steps in 1..T, shape (n,).
        eps: the noise, shape (n,).

    Returns:
        A tuple (err_model, err_ref), each shape (n,): (eps - eps_theta)^2 with
        gradient, and (eps - eps_pretrained)^2 without.
    """
    raise NotImplementedError("denoising_errors")


def diffusion_dpo_loss(err_w, err_w_ref, err_l, err_l_ref, beta):
    """The Diffusion-DPO loss from per-pair denoising errors.

    Args:
        err_w, err_w_ref: the model's and the reference's errors on the
            preferred samples, shape (n,).
        err_l, err_l_ref: the same on the rejected samples, shape (n,).
        beta: the positive scale inside the sigmoid. It stands for the
            paper's beta * T * omega(lambda_t).

    Returns:
        A scalar: the mean over pairs of
        -log sigmoid(-beta * ((err_w - err_w_ref) - (err_l - err_l_ref))).
    """
    raise NotImplementedError("diffusion_dpo_loss")


def diffusion_dpo_train(model, x0_w, x0_l, beta, steps, batch_size, lr, generator):
    """Fine-tune the model on preference pairs with Diffusion-DPO.

    Each step draws batch_size pair indices, one step t per pair, and fresh
    noise for the winner and for the loser, all from `generator`, then takes
    one Adam step (learning rate lr) on diffusion_dpo_loss.

    Args:
        model: a ToyDenoiser, trained in place.
        x0_w: preferred samples, shape (n_pairs,).
        x0_l: rejected samples, shape (n_pairs,).
        beta: the loss scale.
        steps: the number of gradient steps.
        batch_size: pairs per step.
        lr: the Adam learning rate.
        generator: a torch.Generator.

    Returns:
        A list of `steps` floats, the loss at each step before its update.
    """
    raise NotImplementedError("diffusion_dpo_train")


def reward_backprop_train(model, reward_fn, kl_coef, steps, batch_size, lr, generator):
    """Maximise reward_fn(x_0) - kl_coef * path KL by backpropagating through rollout.

    Each step samples batch_size trajectories with rollout (x_T and the noise
    drawn from `generator`), computes the mean reward of x_0 and the mean
    summed ddpm_step_kl, and takes one Adam step on -reward + kl_coef * KL.

    Args:
        model: a ToyDenoiser, trained in place.
        reward_fn: maps x_0, shape (n,), to rewards, shape (n,).
        kl_coef: the weight on the path KL. 0 turns the penalty off.
        steps: the number of gradient steps.
        batch_size: trajectories per step.
        lr: the Adam learning rate.
        generator: a torch.Generator.

    Returns:
        A tuple (samples, kls): a detached tensor of shape (steps, batch_size)
        holding each step's x_0 batch, and a list of `steps` floats, the mean
        path KL at each step.
    """
    raise NotImplementedError("reward_backprop_train")


if __name__ == "__main__":
    a_theta = [0.90, 0.80, 0.85, 0.95]
    b_theta = [0.10, 0.20, 0.00, 0.30]
    a_ref = [0.90, 0.70, 0.85, 1.00]
    b_ref = [0.00, 0.20, 0.10, 0.00]
    s = [0.50, 0.40, 0.30, 0.20]
    gen = torch.Generator().manual_seed(0)
    path = float(chain_path_kl(a_theta, b_theta, a_ref, b_ref, s))
    mc, se = chain_path_kl_mc(a_theta, b_theta, a_ref, b_ref, s, 200_000, gen)
    marginal = float(chain_marginal_kl(a_theta, b_theta, a_ref, b_ref, s))
    print(f"path KL {path:.5f}   Monte Carlo {mc:.4f} +- {se:.4f}   final-sample KL {marginal:.5f}")

    ref = ToyDenoiser()
    x_a, x_b = sample(ref, 2000, gen)[-1], sample(ref, 2000, gen)[-1]
    a_wins = true_reward(x_a) >= true_reward(x_b)
    x0_w, x0_l = torch.where(a_wins, x_a, x_b), torch.where(a_wins, x_b, x_a)
    model = ToyDenoiser()
    losses = diffusion_dpo_train(model, x0_w, x0_l, beta=5.0, steps=150, batch_size=256,
                                 lr=0.05, generator=gen)
    before = float(true_reward(sample(ref, 4000, gen)[-1]).mean())
    after = float(true_reward(sample(model, 4000, gen)[-1]).mean())
    print(f"Diffusion-DPO: loss {losses[0]:.4f} -> {losses[-1]:.4f}, "
          f"true reward {before:.3f} -> {after:.3f}")

    for kl_coef in (0.0, 0.3):
        model = ToyDenoiser()
        x0s, kls = reward_backprop_train(model, proxy_reward, kl_coef, steps=60,
                                         batch_size=256, lr=0.02, generator=gen)
        print(f"reward backprop, kl_coef = {kl_coef}:")
        for i in range(0, 60, 10):
            print(f"  step {i:2d}: proxy {float(proxy_reward(x0s[i]).mean()):7.2f}   "
                  f"true {float(true_reward(x0s[i]).mean()):8.2f}   path KL {kls[i]:7.2f}")
