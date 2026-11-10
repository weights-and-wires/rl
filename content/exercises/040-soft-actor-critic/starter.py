"""Soft actor-critic: the squashed Gaussian, the reparameterised actor, and the temperature.

Fill in the functions under "The exercise", except `gaussian_log_prob`, which is
given. SAC's policy draws a Gaussian sample u = mean + std * eps and outputs the bounded action
a = max_action * tanh(u). You compute the log-probability of that action with
its change-of-variables correction, build the reparameterised actor loss and
the soft critic target, and write the loss that tunes the temperature alpha
toward a target entropy.

Shapes: `mean`, `log_std`, `u`, `eps`, and actions are (N, act_dim).
Log-probabilities, rewards, termination flags, and critic outputs are (N,).
A critic is called as `critic(s, a)`.
"""

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
    """log Normal(u; mean, exp(log_std)**2), summed over the last dimension.

    Given in full: you wrote this kind of density in earlier chapters.

    Args:
        u, mean, log_std: tensors of shape (N, D).

    Returns:
        A tensor of shape (N,).
    """
    z = (u - mean) * torch.exp(-log_std)
    return (-0.5 * z**2 - log_std - 0.5 * math.log(2 * math.pi)).sum(dim=-1)


def tanh_log_det(u):
    """sum_i log(1 - tanh(u_i)**2), the log of the tanh's Jacobian determinant.

    It must stay finite in float32 for |u| as large as 20, where tanh(u)
    rounds to exactly 1.

    Args:
        u: pre-squash values, shape (N, D).

    Returns:
        A tensor of shape (N,).
    """
    raise NotImplementedError("tanh_log_det")


def sample_action(mean, log_std, eps, max_action):
    """A reparameterised tanh-Gaussian sample and its log-probability.

    u = mean + exp(log_std) * eps, and a = max_action * tanh(u). The density
    of a picks up the change-of-variables correction for tanh and for the
    scaling by max_action.

    Args:
        mean, log_std: the Gaussian's parameters, shape (N, D).
        eps: standard normal draws, shape (N, D).
        max_action: the action bound, a positive float.

    Returns:
        (a, log_prob): a of shape (N, D), log_prob of shape (N,). Both must
        carry gradients back to mean and log_std.
    """
    raise NotImplementedError("sample_action")


def actor_loss(policy, critic1, critic2, s, eps, alpha):
    """The reparameterised SAC actor loss.

    mean over the batch of alpha * log pi(a | s) - min(Q1(s, a), Q2(s, a)),
    with (a, log pi) from `sample_action` on `policy(s)` and `eps`.

    Args:
        policy: a module whose call `policy(s)` returns (mean, log_std), with a
            `max_action` attribute.
        critic1, critic2: callables `critic(s, a)` returning shape (N,).
        s: states, shape (N, obs_dim).
        eps: standard normal draws, shape (N, act_dim).
        alpha: the temperature, a float.

    Returns:
        A scalar tensor.
    """
    raise NotImplementedError("actor_loss")


def critic_target(r, terminated, q1_next, q2_next, next_log_prob, gamma, alpha):
    """The soft critic target.

    y = r + gamma * (1 - terminated) * (min(q1_next, q2_next) - alpha * next_log_prob),
    where the next action was sampled from the current policy at s2.

    Args:
        r, terminated, q1_next, q2_next, next_log_prob: tensors of shape (N,).
        gamma: the discount factor.
        alpha: the temperature, a float.

    Returns:
        y, shape (N,).
    """
    raise NotImplementedError("critic_target")


def temperature_loss(log_alpha, log_prob, target_entropy):
    """The loss whose gradient steps tune alpha = exp(log_alpha).

    -mean(alpha * (log_prob + target_entropy)), with log_prob treated as a
    constant.

    Args:
        log_alpha: a scalar tensor with requires_grad=True.
        log_prob: log pi(a | s) for actions sampled from the policy, shape (N,).
        target_entropy: the entropy to aim for, a float, such as -act_dim.

    Returns:
        A scalar tensor.
    """
    raise NotImplementedError("temperature_loss")


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
