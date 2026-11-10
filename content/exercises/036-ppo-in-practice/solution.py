"""Continuous PPO pieces: a diagonal Gaussian, normalizers, and one step."""

import numpy as np

from rlenvs import Pendulum


def _rows(value, actions):
    """Broadcast `value` to the action shape `(N, D)`."""
    value = np.asarray(value, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if value.shape != actions.shape:
        value = np.broadcast_to(value, actions.shape)
    return np.array(value, dtype=np.float64, copy=True)


def gaussian_logprob_entropy_kl(mean, log_std, actions, old_mean, old_log_std):
    """Return log-probability, entropy, and KL for a diagonal Gaussian.

    Each result has shape `(N,)` and is summed over action dimensions. The KL
    is KL(old || current).

    Args:
        mean: current mean, shape `(D,)` or `(N, D)`.
        log_std: current log standard deviation, shape `(D,)` or `(N, D)`.
        actions: actions of shape `(N, D)`.
        old_mean: mean of the policy that collected the actions.
        old_log_std: log standard deviation of that policy.

    Returns:
        A tuple `(logp, entropy, kl)`.
    """
    actions = np.asarray(actions, dtype=np.float64)
    mean = _rows(mean, actions)
    log_std = _rows(log_std, actions)
    old_mean = _rows(old_mean, actions)
    old_log_std = _rows(old_log_std, actions)
    var = np.exp(2.0 * log_std)
    old_var = np.exp(2.0 * old_log_std)
    z = (actions - mean) / np.exp(log_std)
    logp = (-0.5 * np.log(2.0 * np.pi) - log_std - 0.5 * z**2).sum(axis=-1)
    entropy = (0.5 * (np.log(2.0 * np.pi) + 1.0) + log_std).sum(axis=-1)
    kl = (
        log_std
        - old_log_std
        + (old_var + (old_mean - mean) ** 2) / (2.0 * var)
        - 0.5
    ).sum(axis=-1)
    return logp, entropy, kl


class RunningMeanStd:
    """Running population mean and variance.

    `mean` and `var` have the observation shape. `count` starts at 0.
    """

    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.zeros(shape, dtype=np.float64)
        self.count = 0.0

    def _as_batch(self, batch):
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == self.mean.ndim:
            batch = batch.reshape((1, *self.mean.shape))
        elif batch.ndim != self.mean.ndim + 1 or batch.shape[1:] != self.mean.shape:
            raise ValueError(
                f"expected batch shape (N, {self.mean.shape}), got {batch.shape}"
            )
        return batch

    def update(self, batch):
        """Fold `batch`, shape `(N, *shape)`, into the running moments."""
        batch = self._as_batch(batch)
        batch_count = float(batch.shape[0])
        if batch_count == 0.0:
            return
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        self._combine(batch_mean, batch_var, batch_count)

    def merge(self, other):
        """Fold another `RunningMeanStd` with Chan's parallel formula."""
        if other.count == 0.0:
            return
        self._combine(other.mean, other.var, float(other.count))

    def _combine(self, batch_mean, batch_var, batch_count):
        if self.count == 0.0:
            self.mean = np.array(batch_mean, dtype=np.float64, copy=True)
            self.var = np.array(batch_var, dtype=np.float64, copy=True)
            self.count = batch_count
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count
        m2 = m2 + delta**2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x, epsilon=1e-8):
        """Return `(x - mean) / sqrt(var + epsilon)`."""
        x = np.asarray(x, dtype=np.float64)
        return (x - self.mean) / np.sqrt(self.var + epsilon)


def scale_reward_stream(rewards, dones, gamma, epsilon=1e-8):
    """Divide each reward by the running standard deviation of the return.

    Args:
        rewards: rewards of shape `(T, N)`.
        dones: 1 where that step ends an episode.
        gamma: discount used to build the return.
        epsilon: added inside the square root.

    Returns:
        Scaled rewards of shape `(T, N)`.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    scaled = np.zeros_like(rewards)
    running = RunningMeanStd(())
    ret = np.zeros(rewards.shape[1], dtype=np.float64)
    for t in range(rewards.shape[0]):
        ret = ret * gamma + rewards[t]
        running.update(ret)
        scaled[t] = rewards[t] / np.sqrt(running.var + epsilon)
        ret = ret * (1.0 - dones[t])
    return scaled


def normalize_advantages(advantages, epsilon=1e-8):
    """Center one minibatch and divide by its population standard deviation.

    Args:
        advantages: a batch of advantages.
        epsilon: added to the standard deviation.

    Returns:
        The normalized batch.
    """
    advantages = np.asarray(advantages, dtype=np.float64)
    return (advantages - advantages.mean()) / (advantages.std() + epsilon)


def anneal_learning_rate(update, total_updates, initial_lr):
    """Return the linear rate after `update` updates have already finished.

    Args:
        update: updates already finished.
        total_updates: updates in the schedule.
        initial_lr: rate at update 0.

    Returns:
        The rate for the next update.
    """
    return float(initial_lr) * (1.0 - float(update) / float(total_updates))


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
    values = np.asarray(values, dtype=np.float64)
    old_values = np.asarray(old_values, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    clipped = old_values + np.clip(values - old_values, -clip_eps, clip_eps)
    unclipped = (values - targets) ** 2
    clipped_sq = (clipped - targets) ** 2
    return float(np.mean(np.maximum(unclipped, clipped_sq)))


def vectorized_rollout(n_envs, actions, seed, max_steps=200):
    """Step `n_envs` Pendulums and reset each one when it ends.

    Args:
        n_envs: number of environments.
        actions: actions of shape `(T, n_envs)`.
        seed: base seed.
        max_steps: truncation length of each Pendulum.

    Returns:
        A dict of arrays covering the rollout.
    """
    actions = np.asarray(actions, dtype=np.float64)
    envs = [Pendulum(max_steps=max_steps) for _ in range(n_envs)]
    state = [
        np.asarray(env.reset(seed=seed + i), dtype=np.float64)
        for i, env in enumerate(envs)
    ]
    obs, rewards, final_obs, next_obs, terminated, truncated = [], [], [], [], [], []
    for t in range(actions.shape[0]):
        obs.append(np.stack([row.copy() for row in state]))
        step_final, step_next = [], []
        step_reward, step_term, step_trunc = [], [], []
        for i, env in enumerate(envs):
            nxt, reward, term, trunc = env.step(float(actions[t, i]))
            nxt = np.asarray(nxt, dtype=np.float64)
            step_final.append(nxt.copy())
            step_reward.append(float(reward))
            step_term.append(float(term))
            step_trunc.append(float(trunc))
            if term or trunc:
                nxt = np.asarray(env.reset(), dtype=np.float64)
            state[i] = nxt
            step_next.append(nxt.copy())
        final_obs.append(np.stack(step_final))
        next_obs.append(np.stack(step_next))
        rewards.append(np.asarray(step_reward, dtype=np.float64))
        terminated.append(np.asarray(step_term, dtype=np.float64))
        truncated.append(np.asarray(step_trunc, dtype=np.float64))
    return {
        "obs": np.stack(obs),
        "rewards": np.stack(rewards),
        "final_obs": np.stack(final_obs),
        "next_obs": np.stack(next_obs),
        "terminated": np.stack(terminated),
        "truncated": np.stack(truncated),
    }


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
        A dict of floats for the loss pieces and the reward sum.
    """
    logp, entropy, kl = gaussian_logprob_entropy_kl(
        mean, log_std, actions, old_mean, old_log_std
    )
    adv = normalize_advantages(advantages)
    ratio = np.exp(logp - np.asarray(old_logp, dtype=np.float64))
    unclipped = ratio * adv
    clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss = float(-np.mean(np.minimum(unclipped, clipped)))
    value_loss = clipped_value_loss(values, old_values, returns, clip_eps)
    entropy_mean = float(np.mean(entropy))
    loss = policy_loss + value_coef * value_loss - ent_coef * entropy_mean
    return {
        "loss": float(loss),
        "policy_loss": policy_loss,
        "value_loss": float(value_loss),
        "entropy": entropy_mean,
        "approx_kl": float(np.mean(kl)),
        "clip_fraction": float(np.mean(np.abs(ratio - 1.0) > clip_eps)),
        "rollout_return": float(np.sum(rewards)),
    }


if __name__ == "__main__":
    logp, entropy, kl = gaussian_logprob_entropy_kl(
        np.array([[0.0, 1.0]]),
        np.array([0.0, np.log(0.5)]),
        np.array([[0.0, 0.0]]),
        np.array([[0.5, 1.0]]),
        np.array([0.0, 0.0]),
    )
    print("logp", float(logp.sum()), "entropy", float(entropy.sum()), "kl", float(kl.sum()))
    print("lr", [anneal_learning_rate(i, 4, 0.001) for i in range(5)])
    print("value", clipped_value_loss(np.array([1.5]), np.array([1.0]), np.array([2.0]), 0.2))
