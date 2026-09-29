"""Reference solution: marginal-preserving SDE sampling and Flow-GRPO."""

import math

import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# The toy flows. This block is identical in starter.py.
#
# Time runs from t = 1 (noise) to t = 0 (data), and x_t = (1 - t) x_0 + t eps.
# ---------------------------------------------------------------------------

DATA_MEAN = 2.0      # the one-dimensional data is N(2, 0.5^2)
DATA_STD = 0.5

CENTRES = torch.tensor([[1.5, 1.5], [-1.5, 1.5], [-1.5, -1.5], [1.5, -1.5]])
CENTRE_STD = 0.4     # the two-dimensional data is four round bumps
N_PROMPTS = 4        # prompt c asks for a sample near CENTRES[c]


def time_grid(n_steps: int) -> list[float]:
    """n_steps + 1 evenly spaced times from 1 down to 0."""
    return torch.linspace(1.0, 0.0, n_steps + 1).tolist()


def reference_velocity(x: torch.Tensor, t: float) -> torch.Tensor:
    """The exact rectified-flow velocity for the four-bump data, shape (n, 2)."""
    var = (1 - t) ** 2 * CENTRE_STD**2 + t**2
    diff = x[:, None, :] - (1 - t) * CENTRES                    # (n, 4, 2)
    resp = torch.softmax(-0.5 * (diff**2).sum(-1) / var, dim=-1)  # (n, 4)
    per_bump = (t - (1 - t) * CENTRE_STD**2) / var * diff - CENTRES
    return (resp[..., None] * per_bump).sum(1)


class Policy(torch.nn.Module):
    """v_theta(x, t, c) = reference_velocity(x, t) + a small MLP correction.

    The MLP's last layer starts at zero, so the untrained policy is the
    reference flow exactly.
    """

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2 + 1 + N_PROMPTS, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, 2),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, t: float, c: torch.Tensor) -> torch.Tensor:
        onehot = torch.nn.functional.one_hot(c, N_PROMPTS).to(x.dtype)
        tt = torch.full((x.shape[0], 1), float(t))
        return reference_velocity(x, t) + self.net(torch.cat([x, tt, onehot], dim=1))


def prompt_reward(x0: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """exp(-||x_0 - CENTRES[c]||^2): 1 on the asked-for bump, near 0 elsewhere."""
    return torch.exp(-((x0 - CENTRES[c]) ** 2).sum(-1))


# ---------------------------------------------------------------------------
# Your functions.
# ---------------------------------------------------------------------------


def score_from_velocity(x, t, v):
    return -(x + (1 - t) * v) / t


def gaussian_velocity(x, t, m, s):
    return (t - (1 - t) * s**2) / ((1 - t) ** 2 * s**2 + t**2) * (x - (1 - t) * m) - m


def sde_step(x, t, h, v, a, t_cap):
    sigma = a * math.sqrt(t / (1 - min(t, t_cap)))
    mean = x - h * (v + sigma**2 / (2 * t) * (x + (1 - t) * v))
    return mean, sigma * math.sqrt(h)


def sample_sde(velocity_fn, x1, grid, a, generator, return_path=False):
    t_cap = grid[1]
    x = x1
    path = [x]
    for t, t_next in zip(grid[:-1], grid[1:]):
        mean, std = sde_step(x, t, t - t_next, velocity_fn(x, t), a, t_cap)
        x = mean + std * torch.randn(x.shape, generator=generator)
        if return_path:
            path.append(x)
    return torch.stack(path) if return_path else x


def flow_step_kl(v_theta, v_ref, t, h, a, t_cap):
    sigma = a * math.sqrt(t / (1 - min(t, t_cap)))
    coef = h / 2 * (sigma * (1 - t) / (2 * t) + 1 / sigma) ** 2
    return coef * ((v_theta - v_ref) ** 2).sum(-1)


def group_advantages(rewards, eps):
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, unbiased=False, keepdim=True)
    return (rewards - mean) / (std + eps)


def flow_logprob(policy, xs, c, grid, a):
    xs = xs.detach()
    t_cap = grid[1]
    logps, kls = [], []
    for k, (t, t_next) in enumerate(zip(grid[:-1], grid[1:])):
        h = t - t_next
        v = policy(xs[k], t, c)
        mean, std = sde_step(xs[k], t, h, v, a, t_cap)
        z = (xs[k + 1] - mean) / std
        logps.append((-0.5 * z**2 - math.log(std) - 0.5 * math.log(2 * math.pi)).sum(-1))
        kls.append(flow_step_kl(v, reference_velocity(xs[k], t), t, h, a, t_cap))
    return torch.stack(logps), torch.stack(kls)


def flow_grpo_loss(logp, logp_old, advantages, clip_eps, kl, beta):
    ratio = torch.exp(logp - logp_old)
    adv = advantages[None, :]
    objective = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
    return -objective.mean() + beta * kl.mean()


def flow_grpo_train(policy, iterations, group_size, epochs, lr, clip_eps, beta, a,
                    n_steps, generator):
    grid = time_grid(n_steps)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    c = torch.arange(N_PROMPTS).repeat_interleave(group_size)
    history = []
    for _ in range(iterations):
        with torch.no_grad():
            x1 = torch.randn(c.shape[0], 2, generator=generator)
            xs = sample_sde(lambda x, t: policy(x, t, c), x1, grid, a, generator, return_path=True)
            logp_old, kl_old = flow_logprob(policy, xs, c, grid, a)
            r = prompt_reward(xs[-1], c)
            adv = group_advantages(r.view(N_PROMPTS, group_size), 1e-4).reshape(-1)
        history.append((float(r.mean()), float(kl_old.sum(0).mean())))
        for _ in range(epochs):
            logp, kl = flow_logprob(policy, xs, c, grid, a)
            loss = flow_grpo_loss(logp, logp_old, adv, clip_eps, kl, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return history


if __name__ == "__main__":
    gen = torch.Generator().manual_seed(0)
    grid = time_grid(250)
    x1 = torch.randn(40_000, generator=gen)
    print("Same starting noise, four noise levels, data N(2, 0.5^2):")
    for a in (0.0, 0.3, 0.7, math.sqrt(2)):
        x0 = sample_sde(lambda x, t: gaussian_velocity(x, t, DATA_MEAN, DATA_STD), x1, grid, a, gen)
        corr = float(torch.corrcoef(torch.stack([x1, x0]))[0, 1])
        print(f"  a = {a:.3f}: mean {float(x0.mean()):.3f}, std {float(x0.std()):.3f}, "
              f"corr(noise, sample) {corr:+.3f}")

    r = torch.tensor([[0.801, 0.802, 0.800, 0.803]])
    print("GRPO advantages for near-equal rewards:", group_advantages(r, 0.0).tolist()[0])

    torch.manual_seed(0)
    policy = Policy()
    history = flow_grpo_train(policy, iterations=80, group_size=16, epochs=2, lr=3e-3,
                              clip_eps=0.2, beta=0.01, a=0.7, n_steps=10, generator=gen)
    print("Flow-GRPO on the four-bump flow:")
    for i in range(0, len(history), 10):
        print(f"  iteration {i:2d}: mean reward {history[i][0]:.3f}, path KL {history[i][1]:.3f}")
