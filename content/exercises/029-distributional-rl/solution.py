"""Reference solution for the distributional reinforcement learning exercise."""

import numpy as np
import torch

torch.set_num_threads(1)


def categorical_support(v_min: float, v_max: float, n_atoms: int) -> np.ndarray:
    return np.linspace(v_min, v_max, n_atoms)


def c51_project(next_probs, rewards, terminated, gamma: float, support) -> np.ndarray:
    next_probs = np.asarray(next_probs, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    live = 1.0 - np.asarray(terminated, dtype=float)
    support = np.asarray(support, dtype=float)
    v_min, v_max, n = support[0], support[-1], len(support)
    dz = (v_max - v_min) / (n - 1)

    # Shift and shrink every atom, then clip it into the support's range.
    tz = np.clip(rewards[:, None] + gamma * live[:, None] * support[None, :], v_min, v_max)
    b = (tz - v_min) / dz  # fractional atom index, in [0, n - 1]
    lower = np.floor(b).astype(int)
    upper = np.ceil(b).astype(int)
    # When b is an integer, lower == upper and the formulas below would give
    # that atom (u - b) + (b - l) = 0. Nudge so all the mass lands on it.
    lower[(upper > 0) & (lower == upper)] -= 1
    upper[(lower < n - 1) & (lower == upper)] += 1

    m = np.zeros_like(next_probs)
    rows = np.repeat(np.arange(len(rewards))[:, None], n, axis=1)
    np.add.at(m, (rows, lower), next_probs * (upper - b))
    np.add.at(m, (rows, upper), next_probs * (b - lower))
    return m


def c51_loss(logits: torch.Tensor, target_probs) -> torch.Tensor:
    target_probs = torch.as_tensor(target_probs, dtype=logits.dtype)
    log_p = torch.log_softmax(logits, dim=1)
    return -(target_probs * log_p).sum(dim=1).mean()


def quantile_midpoints(n: int) -> np.ndarray:
    return (2 * np.arange(1, n + 1) - 1) / (2 * n)


def quantile_huber_loss(pred: torch.Tensor, target: torch.Tensor, taus, kappa: float = 1.0) -> torch.Tensor:
    taus = torch.as_tensor(taus, dtype=pred.dtype)
    # u[b, i, j] = target_j - pred_i: prediction quantiles on axis 1.
    u = target[:, None, :] - pred[:, :, None]
    abs_u = u.abs()
    if kappa > 0:
        huber = torch.where(abs_u <= kappa, 0.5 * u**2, kappa * (abs_u - 0.5 * kappa))
    else:
        huber = abs_u
    weight = (taus[None, :, None] - (u.detach() < 0).to(pred.dtype)).abs()
    return (weight * huber).mean(dim=2).sum(dim=1).mean()


if __name__ == "__main__":
    # 1. The C51 projection on the worked example.
    support = categorical_support(-2.0, 2.0, 5)
    p = np.array([[0.1, 0.2, 0.4, 0.2, 0.1]])
    m = c51_project(p, np.array([1.0]), np.array([0.0]), 0.9, support)
    print(f"projected m = {np.round(m[0], 4)}, mass {m.sum():.4f}, mean {m[0] @ support:.4f}")

    # 2. Quantile regression on samples from an exponential distribution.
    torch.manual_seed(0)
    samples = torch.distributions.Exponential(1.0).sample((1, 4000)).double()
    taus = quantile_midpoints(4)
    theta = torch.zeros(1, 4, dtype=torch.float64, requires_grad=True)
    optimiser = torch.optim.Adam([theta], lr=0.05)
    for _ in range(600):
        loss = quantile_huber_loss(theta, samples, taus, kappa=0.0)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    print(f"learned quantiles {np.round(theta.detach().numpy()[0], 3)}, "
          f"true {np.round(-np.log(1 - taus), 3)}")
