"""Continuous PPO pieces: a diagonal Gaussian, normalizers, and one step.

The empty `RunningMeanStd` state is given. Fill in every function that raises
`NotImplementedError`.
"""

import numpy as np

from rlenvs import Pendulum  # each environment in vectorized_rollout is a Pendulum


def gaussian_logprob_entropy_kl(mean, log_std, actions, old_mean, old_log_std):
    """Return log-probability, entropy, and KL for a diagonal Gaussian.

    Each result has shape `(N,)` and is summed over action dimensions. `mean`
    and `old_mean` have shape `(D,)` or `(N, D)`. `log_std` and `old_log_std`
    have shape `(D,)` or `(N, D)`. The KL is KL(old || current).

    Args:
        mean: current mean.
        log_std: current log standard deviation.
        actions: actions of shape `(N, D)`.
        old_mean: mean of the policy that collected the actions.
        old_log_std: log standard deviation of that policy.

    Returns:
        A tuple `(logp, entropy, kl)`.
    """
    raise NotImplementedError("gaussian_logprob_entropy_kl")


class RunningMeanStd:
    """Running population mean and variance.

    `mean` and `var` have the observation shape. `count` starts at 0.
    """

    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.zeros(shape, dtype=np.float64)
        self.count = 0.0

    def update(self, batch):
        """Fold `batch`, shape `(N, *shape)`, into the running moments."""
        raise NotImplementedError("RunningMeanStd.update")

    def merge(self, other):
        """Fold another `RunningMeanStd` with Chan's parallel formula."""
        raise NotImplementedError("RunningMeanStd.merge")

    def normalize(self, x, epsilon=1e-8):
        """Return `(x - mean) / sqrt(var + epsilon)`."""
        raise NotImplementedError("RunningMeanStd.normalize")


def scale_reward_stream(rewards, dones, gamma, epsilon=1e-8):
    """Divide each reward by the running standard deviation of the return.

    `rewards` and `dones` have shape `(T, N)`. Update `G = gamma * G + r`,
    fold `G` into a scalar running variance, divide the reward, then set `G`
    to 0 where `dones` is 1. Do not subtract the mean.

    Args:
        rewards: rewards of shape `(T, N)`.
        dones: 1 where that step ends an episode.
        gamma: discount used to build the return.
        epsilon: added inside the square root.

    Returns:
        Scaled rewards of shape `(T, N)`.
    """
    raise NotImplementedError("scale_reward_stream")


def normalize_advantages(advantages, epsilon=1e-8):
    """Center one minibatch and divide by its population standard deviation.

    Args:
        advantages: a batch of advantages.
        epsilon: added to the standard deviation.

    Returns:
        The normalized batch, same shape as `advantages`.
    """
    raise NotImplementedError("normalize_advantages")


def anneal_learning_rate(update, total_updates, initial_lr):
    """Return the linear rate after `update` updates have already finished.

    Args:
        update: updates already finished.
        total_updates: updates in the schedule.
        initial_lr: rate at update 0.

    Returns:
        `initial_lr * (1 - update / total_updates)`.
    """
    raise NotImplementedError("anneal_learning_rate")


def clipped_value_loss(values, old_values, targets, clip_eps):
    """Return the mean of the worse of the two squared value errors.

    Args:
        values: current value predictions.
        old_values: values that produced the advantages.
        targets: returns the critic is fitting.
        clip_eps: half-width of the band around `old_values`.

    Returns:
        A scalar loss.
    """
    raise NotImplementedError("clipped_value_loss")


def vectorized_rollout(n_envs, actions, seed, max_steps=200):
    """Step `n_envs` Pendulums and reset each one when it ends.

    Environment `i` starts with `reset(seed + i)`. A later reset calls
    `reset()` with no seed. `final_obs` is the state before that reset, and
    `next_obs` is the state the next step sees.

    Args:
        n_envs: number of environments.
        actions: actions of shape `(T, n_envs)`.
        seed: base seed.
        max_steps: truncation length of each Pendulum.

    Returns:
        A dict with `obs`, `rewards`, `final_obs`, `next_obs`, `terminated`,
        and `truncated`.
    """
    raise NotImplementedError("vectorized_rollout")


def ppo_continuous_step(
    mean,
    log_std,
    actions,
    old_mean,
    old_log_std,
    old_logp,
    advantages,
    returns,
    values,
    old_values,
    rewards,
    clip_eps=0.2,
    value_coef=0.5,
    ent_coef=0.01,
):
    """Take one continuous PPO step and record the reward sum.

    Normalize the advantages, form the ratio from `old_logp`, and minimize
    the negative clipped surrogate plus `value_coef` times the clipped value
    loss minus `ent_coef` times the mean entropy. Fit the value to `returns`.

    Args:
        mean: current mean, shape `(N, D)`.
        log_std: current log standard deviation.
        actions: actions of shape `(N, D)`.
        old_mean: mean of the policy that collected the actions.
        old_log_std: log standard deviation of that policy.
        old_logp: stored log-probability of each action, shape `(N,)`.
        advantages: advantages, shape `(N,)`.
        returns: value targets, shape `(N,)`.
        values: current values, shape `(N,)`.
        old_values: values that produced the advantages, shape `(N,)`.
        rewards: rewards that produced the batch.
        clip_eps: clip width for the ratio and the value.
        value_coef: weight on the value loss.
        ent_coef: weight on the entropy.

    Returns:
        A dict of floats: `loss`, `policy_loss`, `value_loss`, `entropy`,
        `approx_kl`, `clip_fraction`, and `rollout_return`.
    """
    raise NotImplementedError("ppo_continuous_step")


if __name__ == "__main__":
    print(gaussian_logprob_entropy_kl(0, 0, 0, 0, 0))
