"""Marginal-preserving SDE sampling and Flow-GRPO on toy flows.

A rectified-flow model samples by integrating an ODE, which has no per-step
density. You turn it into an SDE with the same marginals, check that on a
one-dimensional Gaussian flow whose velocity is exact, and then train a
two-dimensional flow with Flow-GRPO: groups of rollouts per prompt,
group-normalised rewards, a clipped per-step ratio, and a KL penalty.

Time runs from t = 1 (noise) to t = 0 (data). A step goes from t to t - h
with h > 0. A path `xs` has shape (n_steps + 1, n, ...): xs[0] is the noise
and xs[-1] is the sample. Row k of anything per-step belongs to the step
from grid[k] to grid[k + 1]. Every tensor is float64.
"""

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
    """The score, d/dx log p_t(x), of a rectified flow, from its velocity.

    Args:
        x: positions, any shape.
        t: the time, a float in (0, 1].
        v: the velocity at x and t, the same shape as x.

    Returns:
        The score, the same shape as x.
    """
    raise NotImplementedError("score_from_velocity")


def gaussian_velocity(x, t, m, s):
    """The exact rectified-flow velocity E[eps - x_0 | x_t = x] for data N(m, s^2).

    Args:
        x: positions, any shape.
        t: the time, a float in [0, 1].
        m, s: the data mean and standard deviation, floats.

    Returns:
        The velocity, the same shape as x.
    """
    raise NotImplementedError("gaussian_velocity")


def sde_step(x, t, h, v, a, t_cap):
    """Mean and standard deviation of one Flow-GRPO SDE step from t to t - h.

    The noise level is sigma_t = a * sqrt(t / (1 - min(t, t_cap))). The cap
    only matters at t = 1, where the uncapped formula divides by zero.

    Args:
        x: positions, any shape.
        t: the current time, a float in (0, 1].
        h: the step size, a positive float.
        v: the velocity at x and t, the same shape as x.
        a: the noise-level multiplier. a = 0 gives the plain Euler ODE step.
        t_cap: the largest t used in the denominator of sigma_t.

    Returns:
        A tuple (mean, std): the step's mean, shaped like x, and its standard
        deviation sigma_t * sqrt(h) as a float.
    """
    raise NotImplementedError("sde_step")


def sample_sde(velocity_fn, x1, grid, a, generator, return_path=False):
    """Run the SDE sampler from noise at grid[0] = 1 down to grid[-1] = 0.

    Use t_cap = grid[1]. Draw each step's noise with
    torch.randn(x.shape, generator=generator).

    Args:
        velocity_fn: a function (x, t) -> velocity, shaped like x.
        x1: the starting noise, shape (n,) or (n, d).
        grid: a list of decreasing times, grid[0] = 1 and grid[-1] = 0.
        a: the noise-level multiplier.
        generator: a torch.Generator.
        return_path: if True, return every state, not only the last.

    Returns:
        The final sample, shaped like x1, or with return_path the whole path,
        shape (len(grid),) + x1.shape.
    """
    raise NotImplementedError("sample_sde")


def flow_step_kl(v_theta, v_ref, t, h, a, t_cap):
    """Flow-GRPO's closed-form KL between two SDE steps that share x_t.

    Args:
        v_theta, v_ref: the two velocities at x_t, shape (..., d).
        t, h, a, t_cap: as in sde_step.

    Returns:
        (h / 2) (sigma_t (1 - t) / (2 t) + 1 / sigma_t)^2 ||v_theta - v_ref||^2,
        with the squared norm summed over the last axis, shape (...).
    """
    raise NotImplementedError("flow_step_kl")


def group_advantages(rewards, eps):
    """GRPO advantages: normalise rewards within each group.

    Args:
        rewards: shape (groups, group_size).
        eps: a floor added to the population standard deviation.

    Returns:
        (rewards - group mean) / (group std + eps), shape (groups, group_size).
        Use the population standard deviation (divide by group_size).
    """
    raise NotImplementedError("group_advantages")


def flow_logprob(policy, xs, c, grid, a):
    """Per-step log-probabilities and KLs of a stored path under the policy.

    Treat xs as fixed data. Use t_cap = grid[1].

    Args:
        policy: a Policy.
        xs: a path, shape (len(grid), n, 2).
        c: the prompt of each path, an integer tensor of shape (n,).
        grid: the time grid the path was sampled on.
        a: the noise-level multiplier it was sampled with.

    Returns:
        A tuple (logp, kl), each shape (len(grid) - 1, n). Row k of logp is
        log p_theta(xs[k + 1] | xs[k]), summed over both coordinates. Row k
        of kl is flow_step_kl between the policy and reference_velocity at
        xs[k].
    """
    raise NotImplementedError("flow_logprob")


def flow_grpo_loss(logp, logp_old, advantages, clip_eps, kl, beta):
    """The Flow-GRPO loss to minimise.

    Args:
        logp, logp_old: per-step log-probabilities, shape (steps, n).
        advantages: one advantage per path, shape (n,).
        clip_eps: ratios are clipped to [1 - clip_eps, 1 + clip_eps].
        kl: per-step KLs to the reference, shape (steps, n).
        beta: the KL coefficient.

    Returns:
        A scalar: minus the mean over steps and paths of
        min(ratio * A, clip(ratio) * A), plus beta times the mean of kl.
    """
    raise NotImplementedError("flow_grpo_loss")


def flow_grpo_train(policy, iterations, group_size, epochs, lr, clip_eps, beta, a,
                    n_steps, generator):
    """Train the policy with Flow-GRPO and return its progress.

    Each iteration samples group_size paths for each of the N_PROMPTS prompts
    on time_grid(n_steps), with noise from `generator`. It records the old
    log-probabilities, computes prompt_reward and group advantages (eps =
    1e-4), and takes `epochs` Adam steps (learning rate lr) on flow_grpo_loss.

    Args:
        policy: a Policy, trained in place.
        iterations: the number of sample-then-update rounds.
        group_size: paths per prompt per round.
        epochs: gradient steps per round.
        lr: the Adam learning rate.
        clip_eps: the clip range.
        beta: the KL coefficient.
        a: the noise-level multiplier.
        n_steps: sampling steps per path.
        generator: a torch.Generator.

    Returns:
        A list of `iterations` tuples (mean reward, mean path KL), measured on
        each round's paths before its update. The path KL is the per-step KL
        summed over steps, averaged over paths.
    """
    raise NotImplementedError("flow_grpo_train")


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
