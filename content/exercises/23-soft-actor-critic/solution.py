"""Reference solution for the soft actor-critic exercise."""

import math

import torch
import torch.nn.functional as F

torch.set_num_threads(1)


# Given: the policy network ------------------------------------------------------


class TanhGaussianPolicy(torch.nn.Module):
    """Maps states to the mean and log standard deviation of the pre-squash Gaussian.

    `forward(s)` returns `(mean, log_std)`, each of shape (N, act_dim). The
    action is `max_action * tanh(u)` with `u ~ Normal(mean, exp(log_std)**2)`.
    """

    def __init__(self, obs_dim, act_dim, max_action, hidden=64):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        )
        self.mean = torch.nn.Linear(hidden, act_dim)
        self.log_std = torch.nn.Linear(hidden, act_dim)
        self.max_action = max_action

    def forward(self, s):
        h = self.body(s)
        return self.mean(h), self.log_std(h).clamp(-5.0, 2.0)


# The exercise -------------------------------------------------------------------


def gaussian_log_prob(u, mean, log_std):
    z = (u - mean) * torch.exp(-log_std)
    return (-0.5 * z**2 - log_std - 0.5 * math.log(2 * math.pi)).sum(dim=-1)


def tanh_log_det(u):
    # log(1 - tanh(u)^2) = 2 * (log 2 - u - softplus(-2u)), which stays finite for large |u|.
    return (2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))).sum(dim=-1)


def sample_action(mean, log_std, eps, max_action):
    u = mean + torch.exp(log_std) * eps
    a = max_action * torch.tanh(u)
    D = u.shape[-1]
    log_prob = gaussian_log_prob(u, mean, log_std) - tanh_log_det(u) - D * math.log(max_action)
    return a, log_prob


def actor_loss(policy, critic1, critic2, s, eps, alpha):
    mean, log_std = policy(s)
    a, log_prob = sample_action(mean, log_std, eps, policy.max_action)
    q = torch.minimum(critic1(s, a), critic2(s, a))
    return (alpha * log_prob - q).mean()


def critic_target(r, terminated, q1_next, q2_next, next_log_prob, gamma, alpha):
    soft_next = torch.minimum(q1_next, q2_next) - alpha * next_log_prob
    return r + gamma * (1.0 - terminated) * soft_next


def temperature_loss(log_alpha, log_prob, target_entropy):
    return -(log_alpha.exp() * (log_prob + target_entropy).detach()).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    mean, log_std, max_action = torch.tensor([[0.3]]), torch.tensor([[math.log(0.8)]]), 2.0
    eps = torch.randn(200_000, 1)
    a, _ = sample_action(mean.expand(200_000, 1), log_std.expand(200_000, 1), eps, max_action)
    edges = torch.linspace(-2.0, 2.0, 21)
    counts = torch.histc(a.squeeze(-1), bins=20, min=-2.0, max=2.0)
    width = float(edges[1] - edges[0])
    print("Tanh-Gaussian with mean 0.3, std 0.8, bound 2: histogram against exp(log_prob)")
    print("    a      histogram   formula")
    for k in range(0, 20, 2):
        centre = 0.5 * (edges[k] + edges[k + 1])
        u = torch.atanh(centre / max_action).reshape(1, 1)
        logp = gaussian_log_prob(u, mean, log_std) - tanh_log_det(u) - math.log(max_action)
        print(f"  {float(centre):+.2f}    {float(counts[k]) / (200_000 * width):.4f}     {float(logp.exp()):.4f}")

    print("\nAutomatic temperature: a Gaussian policy that best-responds to Q(a) = -a^2 / 2")
    target = 1.0
    log_alpha = torch.zeros((), requires_grad=True)
    opt = torch.optim.Adam([log_alpha], lr=0.05)
    for step in range(401):
        alpha = float(log_alpha.detach().exp())
        entropy = 0.5 * math.log(2 * math.pi * math.e * alpha)  # the best response has variance alpha
        if step % 50 == 0:
            print(f"  step {step:>3}: alpha = {alpha:.4f}, policy entropy = {entropy:+.4f} (target {target})")
        loss = temperature_loss(log_alpha, torch.tensor([-entropy]), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
