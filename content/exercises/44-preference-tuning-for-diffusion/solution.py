"""Reference solution: path KL, Diffusion-DPO, and reward hacking on a toy."""

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
    mu1, std1, mu2, std2 = (torch.as_tensor(v, dtype=torch.float64) for v in (mu1, std1, mu2, std2))
    return torch.log(std2 / std1) + (std1**2 + (mu1 - mu2) ** 2) / (2 * std2**2) - 0.5


def _chain_moments(a, b, s):
    mean, var = torch.tensor(0.0), torch.tensor(1.0)
    moments = []
    for a_k, b_k, s_k in zip(a, b, s):
        moments.append((mean, var))
        mean, var = a_k * mean + b_k, a_k**2 * var + s_k**2
    return moments, (mean, var)


def chain_path_kl(a_theta, b_theta, a_ref, b_ref, s):
    a_theta, b_theta, a_ref, b_ref, s = (torch.as_tensor(v, dtype=torch.float64)
                                         for v in (a_theta, b_theta, a_ref, b_ref, s))
    moments, _ = _chain_moments(a_theta, b_theta, s)
    total = torch.tensor(0.0)
    for k, (mean, var) in enumerate(moments):
        da, db = a_theta[k] - a_ref[k], b_theta[k] - b_ref[k]
        # E[(da * x + db)^2] with x ~ N(mean, var), divided by 2 s^2.
        total = total + (da**2 * (var + mean**2) + 2 * da * db * mean + db**2) / (2 * s[k] ** 2)
    return total


def chain_marginal_kl(a_theta, b_theta, a_ref, b_ref, s):
    a_theta, b_theta, a_ref, b_ref, s = (torch.as_tensor(v, dtype=torch.float64)
                                         for v in (a_theta, b_theta, a_ref, b_ref, s))
    _, (m1, v1) = _chain_moments(a_theta, b_theta, s)
    _, (m2, v2) = _chain_moments(a_ref, b_ref, s)
    return gaussian_kl(m1, torch.sqrt(v1), m2, torch.sqrt(v2))


def chain_path_kl_mc(a_theta, b_theta, a_ref, b_ref, s, n, generator):
    a_theta, b_theta, a_ref, b_ref, s = (torch.as_tensor(v, dtype=torch.float64)
                                         for v in (a_theta, b_theta, a_ref, b_ref, s))
    x = torch.randn(n, generator=generator)
    log_ratio = torch.zeros(n)
    for k in range(len(s)):
        mean_theta = a_theta[k] * x + b_theta[k]
        mean_ref = a_ref[k] * x + b_ref[k]
        x_next = mean_theta + s[k] * torch.randn(n, generator=generator)
        log_ratio += (-(x_next - mean_theta) ** 2 + (x_next - mean_ref) ** 2) / (2 * s[k] ** 2)
        x = x_next
    return float(log_ratio.mean()), float(log_ratio.std() / math.sqrt(n))


def ddpm_step_kl(model, xs):
    rows = []
    for k in range(T):
        t = T - k
        x = xs[k]
        mean_theta = ddpm_mean(x, model.eps(x, t), t)
        mean_ref = ddpm_mean(x, eps_pretrained(x, t), t)
        rows.append(gaussian_kl(mean_theta, SIGMAS[t - 1], mean_ref, SIGMAS[t - 1]))
    return torch.stack(rows)


def denoising_errors(model, x0, t, eps):
    abar = ALPHA_BARS[t - 1]
    x_t = torch.sqrt(abar) * x0 + torch.sqrt(1.0 - abar) * eps
    err_model = (eps - model.eps(x_t, t)) ** 2
    with torch.no_grad():
        err_ref = (eps - eps_pretrained(x_t, t)) ** 2
    return err_model, err_ref


def diffusion_dpo_loss(err_w, err_w_ref, err_l, err_l_ref, beta):
    inside = -beta * ((err_w - err_w_ref) - (err_l - err_l_ref))
    return -F.logsigmoid(inside).mean()


def diffusion_dpo_train(model, x0_w, x0_l, beta, steps, batch_size, lr, generator):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    n_pairs = x0_w.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n_pairs, (batch_size,), generator=generator)
        t = torch.randint(1, T + 1, (batch_size,), generator=generator)   # one t per pair
        eps_w = torch.randn(batch_size, generator=generator)
        eps_l = torch.randn(batch_size, generator=generator)
        err_w, err_w_ref = denoising_errors(model, x0_w[idx], t, eps_w)
        err_l, err_l_ref = denoising_errors(model, x0_l[idx], t, eps_l)
        loss = diffusion_dpo_loss(err_w, err_w_ref, err_l, err_l_ref, beta)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def reward_backprop_train(model, reward_fn, kl_coef, steps, batch_size, lr, generator):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    samples, kls = [], []
    for _ in range(steps):
        xs = rollout(model, torch.randn(batch_size, generator=generator),
                     torch.randn(T, batch_size, generator=generator))
        path_kl = ddpm_step_kl(model, xs).sum(0).mean()
        loss = -reward_fn(xs[-1]).mean() + kl_coef * path_kl
        opt.zero_grad()
        loss.backward()
        opt.step()
        samples.append(xs[-1].detach())
        kls.append(float(path_kl.detach()))
    return torch.stack(samples), kls


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
