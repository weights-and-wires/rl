"""Checks for the continuous PPO exercise.

Wrong implementations this file catches:

- `gaussian_logprob_entropy_kl` that omits `-0.5 * log(2π)` on each action
  dimension. The probability ratio of two such policies still matches, and
  the log-probability does not.
- `normalize_advantages` that divides by the standard deviation of the
  returns instead of the standard deviation of the advantages.
"""

import numpy as np
import torch

from rlenvs import Pendulum
from rlcheck import almost, check, close, fail, finite_difference, has

torch.set_num_threads(1)
torch.manual_seed(0)

_MEAN = np.array([[0.0, 1.0]])
_LOG_STD = np.array([0.0, np.log(0.5)])
_ACTION = np.array([[0.0, 0.0]])
_OLD_MEAN = np.array([[0.5, 1.0]])
_OLD_LOG_STD = np.array([0.0, 0.0])


def _rows(value, actions):
    value = np.asarray(value, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if value.shape != actions.shape:
        value = np.broadcast_to(value, actions.shape)
    return np.array(value, dtype=np.float64, copy=True)


def _closed_form(mean, log_std, actions, old_mean, old_log_std):
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


def _torch_oracle(mean, log_std, actions, old_mean, old_log_std):
    actions_t = torch.tensor(np.asarray(actions, dtype=np.float64))
    current = torch.distributions.Normal(
        torch.tensor(_rows(mean, actions)),
        torch.exp(torch.tensor(_rows(log_std, actions))),
    )
    old = torch.distributions.Normal(
        torch.tensor(_rows(old_mean, actions)),
        torch.exp(torch.tensor(_rows(old_log_std, actions))),
    )
    logp = current.log_prob(actions_t).sum(dim=-1)
    entropy = current.entropy().sum(dim=-1)
    kl = torch.distributions.kl_divergence(old, current).sum(dim=-1)
    return logp.numpy(), entropy.numpy(), kl.numpy()


def _call_gaussian(sub, mean, log_std, actions, old_mean, old_log_std):
    fn = has(sub, "gaussian_logprob_entropy_kl")
    out = fn(mean, log_std, actions, old_mean, old_log_std)
    if not isinstance(out, tuple) or len(out) != 3:
        fail(
            f"expected a tuple of three arrays, got {type(out).__name__} "
            f"of length {len(out) if isinstance(out, tuple) else 'n/a'}"
        )
    return out


@check("the Gaussian log-probability keeps the constant term")
def test_logprob(sub):
    logp, _, _ = _call_gaussian(sub, _MEAN, _LOG_STD, _ACTION, _OLD_MEAN, _OLD_LOG_STD)
    want, _, _ = _closed_form(_MEAN, _LOG_STD, _ACTION, _OLD_MEAN, _OLD_LOG_STD)
    # A version that drops -0.5*log(2π) per dimension is high by that constant.
    forgot = want + 0.5 * np.log(2.0 * np.pi) * _ACTION.shape[-1]
    got = np.asarray(logp, dtype=np.float64)
    if np.allclose(got, forgot, atol=1e-6):
        fail(
            "log-probability matches the version that drops -0.5*log(2π) "
            f"per dimension. expected {np.array2string(want, precision=6)}, "
            f"got {np.array2string(got, precision=6)}"
        )
    close(got, want, atol=1e-8, label="logprob")
    almost(float(np.sum(got)), -3.1447, 5e-5, label="lesson table sum")


@check("entropy and KL match the closed form, and the log-probability gradient matches a finite difference")
def test_entropy_kl_and_gradient(sub):
    logp, entropy, kl = _call_gaussian(
        sub, _MEAN, _LOG_STD, _ACTION, _OLD_MEAN, _OLD_LOG_STD
    )
    _, want_h, want_kl = _closed_form(_MEAN, _LOG_STD, _ACTION, _OLD_MEAN, _OLD_LOG_STD)
    torch_logp, torch_h, torch_kl = _torch_oracle(
        _MEAN, _LOG_STD, _ACTION, _OLD_MEAN, _OLD_LOG_STD
    )
    close(entropy, want_h, atol=1e-8, label="entropy")
    close(kl, want_kl, atol=1e-8, label="kl")
    close(logp, torch_logp, atol=1e-8, label="logprob vs torch")
    close(entropy, torch_h, atol=1e-8, label="entropy vs torch")
    close(kl, torch_kl, atol=1e-8, label="kl vs torch")
    almost(float(np.sum(entropy)), 2.1447, 5e-5, label="lesson entropy sum")
    almost(float(np.sum(kl)), 0.9319, 5e-5, label="lesson kl sum")

    fn = has(sub, "gaussian_logprob_entropy_kl")

    def sum_logp_of_mean(mean_vec):
        got, _, _ = fn(mean_vec, _LOG_STD, _ACTION, _OLD_MEAN, _OLD_LOG_STD)
        return float(np.sum(got))

    def sum_logp_of_log_std(log_std_vec):
        got, _, _ = fn(_MEAN, log_std_vec, _ACTION, _OLD_MEAN, _OLD_LOG_STD)
        return float(np.sum(got))

    mean_grad = finite_difference(sum_logp_of_mean, np.array([0.0, 1.0]))
    log_std_grad = finite_difference(sum_logp_of_log_std, _LOG_STD.copy())
    close(mean_grad, np.array([0.0, -4.0]), atol=1e-4, label="d logp / d mean")
    close(log_std_grad, np.array([-1.0, 3.0]), atol=1e-4, label="d logp / d log_std")


@check("merging two running estimates matches one pass")
def test_running_mean(sub):
    cls = has(sub, "RunningMeanStd")
    left = cls(())
    right = cls(())
    left.update(np.array([1.0, 3.0]))
    right.update(np.array([2.0, 6.0]))
    left.merge(right)
    data = np.array([1.0, 3.0, 2.0, 6.0])
    almost(float(left.mean), float(np.mean(data)), 1e-12, label="merged mean")
    almost(float(left.var), float(np.var(data)), 1e-12, label="merged var")
    almost(float(left.count), 4.0, 1e-12, label="merged count")

    rng = np.random.default_rng(0)
    batch = rng.normal(size=(20, 3))
    one = cls((3,))
    one.update(batch)
    two = cls((3,))
    other = cls((3,))
    two.update(batch[:7])
    other.update(batch[7:])
    two.merge(other)
    close(one.mean, batch.mean(axis=0), atol=1e-12, label="one-pass mean")
    close(one.var, batch.var(axis=0), atol=1e-12, label="one-pass var")
    close(two.mean, batch.mean(axis=0), atol=1e-12, label="merged batch mean")
    close(two.var, batch.var(axis=0), atol=1e-12, label="merged batch var")
    close(two.normalize(two.mean), np.zeros(3), atol=1e-8, label="normalize at the mean")


def _scale_reference(rewards, dones, gamma, epsilon=1e-8):
    rewards = np.asarray(rewards, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    scaled = np.zeros_like(rewards)
    seen = []
    ret = np.zeros(rewards.shape[1], dtype=np.float64)
    for t in range(rewards.shape[0]):
        ret = ret * gamma + rewards[t]
        seen.extend(ret.tolist())
        scaled[t] = rewards[t] / np.sqrt(np.var(seen) + epsilon)
        ret = ret * (1.0 - dones[t])
    return scaled


@check("reward scaling divides by the running return standard deviation")
def test_reward_scale(sub):
    fn = has(sub, "scale_reward_stream")
    rewards = np.array([[2.0], [0.0], [4.0]])
    dones = np.zeros_like(rewards)
    got = np.asarray(fn(rewards, dones, 0.5), dtype=np.float64)
    want = _scale_reference(rewards, dones, 0.5)
    close(got, want, atol=1e-8, label="lesson stream")
    almost(float(got[0, 0]), 20000.0, 1e-6, label="first scaled reward")
    almost(float(got[2, 0]), 2.7175, 5e-4, label="third scaled reward")

    rng = np.random.default_rng(1)
    stream = rng.normal(size=(6, 2))
    ends = np.zeros_like(stream)
    ends[2, 0] = 1.0
    ends[4, 1] = 1.0
    got = np.asarray(fn(stream, ends, 0.9), dtype=np.float64)
    want = _scale_reference(stream, ends, 0.9)
    close(got, want, atol=1e-8, label="stream with episode ends")


@check("advantage normalization divides by the standard deviation of the advantages")
def test_advantage_norm(sub):
    fn = has(sub, "normalize_advantages")
    advantages = np.array([1.0, 3.0, 5.0])
    returns = np.array([0.0, 6.0, 12.0])
    got = np.asarray(fn(advantages), dtype=np.float64)
    right = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    # Wrong: divide the centered advantages by std(returns), which is 3x larger.
    wrong = (advantages - advantages.mean()) / (returns.std() + 1e-8)
    if np.allclose(got, wrong, atol=1e-6) and not np.allclose(got, right, atol=1e-6):
        fail(
            "divided by std(returns) "
            f"{returns.std():.6g}. expected {np.array2string(right, precision=6)}, "
            f"got {np.array2string(got, precision=6)}"
        )
    close(got, right, atol=1e-8, label="normalized advantages")
    almost(float(got[0]), -1.2247, 5e-4, label="lesson first advantage")


@check("the learning rate falls in a straight line to zero")
def test_anneal(sub):
    fn = has(sub, "anneal_learning_rate")
    got = [fn(update, 4, 0.001) for update in range(5)]
    want = [0.001, 0.00075, 0.0005, 0.00025, 0.0]
    close(got, want, atol=1e-12, label="schedule")
    almost(0.4 - fn(0, 4, 0.001) * 2.0, 0.398, 1e-12, label="first step")
    almost(0.4 - fn(2, 4, 0.001) * 2.0, 0.399, 1e-12, label="later step")


def _value_reference(values, old_values, targets, clip_eps):
    values = np.asarray(values, dtype=np.float64)
    old_values = np.asarray(old_values, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    clipped = old_values + np.clip(values - old_values, -clip_eps, clip_eps)
    return float(np.mean(np.maximum((values - targets) ** 2, (clipped - targets) ** 2)))


@check("the clipped value loss takes the worse square, and the gradient vanishes outside the band")
def test_value_clip(sub):
    fn = has(sub, "clipped_value_loss")
    outside = fn(np.array([1.5]), np.array([1.0]), np.array([2.0]), 0.2)
    inside = fn(np.array([1.1]), np.array([1.0]), np.array([2.0]), 0.2)
    almost(outside, 0.64, 1e-12, label="outside the band")
    almost(inside, 0.81, 1e-12, label="inside the band")

    def at(value):
        def loss(arr):
            return fn(arr, np.array([1.0]), np.array([2.0]), 0.2)

        return finite_difference(loss, np.array([value], dtype=np.float64))

    close(at(1.1), np.array([-1.8]), atol=1e-4, label="gradient inside the band")
    close(at(1.5), np.array([0.0]), atol=1e-4, label="gradient outside the band")


def _pendulum_reference(n_envs, actions, seed, max_steps):
    actions = np.asarray(actions, dtype=np.float64)
    envs = [Pendulum(max_steps=max_steps) for _ in range(n_envs)]
    state = [
        np.asarray(env.reset(seed=seed + i), dtype=np.float64)
        for i, env in enumerate(envs)
    ]
    obs, rewards, final_obs, next_obs, terminated, truncated = [], [], [], [], [], []
    for t in range(actions.shape[0]):
        obs.append(np.stack([row.copy() for row in state]))
        step_final, step_next, step_reward, step_term, step_trunc = [], [], [], [], []
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
            step_next.append(np.asarray(nxt, dtype=np.float64).copy())
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


@check("a vectorized rollout matches independent Pendulums, including a reset that leaves the others running")
def test_rollout(sub):
    fn = has(sub, "vectorized_rollout")
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1.5, 1.5, size=(5, 3))
    got = fn(3, actions, seed=0, max_steps=2)
    want = _pendulum_reference(3, actions, seed=0, max_steps=2)
    if not isinstance(got, dict):
        fail(f"expected a dict of arrays, got {type(got).__name__}")
    for key, expected in want.items():
        if key not in got:
            fail(
                f"missing {key}. expected shape {expected.shape}, "
                f"got keys {sorted(got)}"
            )
        close(got[key], expected, atol=1e-8, label=key)
    # max_steps=2 truncates on steps 1 and 3. The reset state is next_obs.
    for step in (1, 3):
        if np.allclose(got["final_obs"][step], got["next_obs"][step]):
            fail(
                f"step {step} stored the same state in final_obs and next_obs. "
                f"expected a reset, got {np.array2string(got['final_obs'][step, 0], precision=4)}"
            )


def _step_reference(
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
    logp, entropy, kl = _closed_form(mean, log_std, actions, old_mean, old_log_std)
    advantages = np.asarray(advantages, dtype=np.float64)
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    ratio = np.exp(logp - np.asarray(old_logp, dtype=np.float64))
    surrogate = np.minimum(
        ratio * adv, np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    )
    policy_loss = float(-np.mean(surrogate))
    value_loss = _value_reference(values, old_values, returns, clip_eps)
    entropy_mean = float(np.mean(entropy))
    return {
        "loss": policy_loss + value_coef * value_loss - ent_coef * entropy_mean,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy_mean,
        "approx_kl": float(np.mean(kl)),
        "clip_fraction": float(np.mean(np.abs(ratio - 1.0) > clip_eps)),
        "rollout_return": float(np.sum(rewards)),
    }


def _require_step(got, want):
    if not isinstance(got, dict):
        fail(f"expected a dict of floats, got {type(got).__name__}")
    for key, expected in want.items():
        if key not in got:
            fail(f"missing {key}. expected {expected:.6g}, got keys {sorted(got)}")
        almost(got[key], expected, 1e-6, label=key)


@check("one continuous PPO step matches the assembled loss, and the recorded return is the reward sum")
def test_step(sub):
    fn = has(sub, "ppo_continuous_step")
    # Two of the four ratios leave [0.8, 1.2]. sigma is 1, so the log-prob
    # difference is -0.5 * ((a - mean)^2 - (a - old_mean)^2).
    actions = np.array([[1.0], [0.0], [0.0], [0.0]])
    mean = np.array([[1.0], [0.0], [-1.0], [0.2]])
    old_mean = np.zeros_like(mean)
    log_std = np.array([0.0])
    old_logp, _, _ = _closed_form(old_mean, log_std, actions, old_mean, log_std)
    advantages = np.array([1.0, -1.0, 0.5, 2.0])
    returns = np.array([0.5, 0.0, 1.0, -0.5])
    values = np.array([0.4, 0.2, 0.9, -0.2])
    old_values = np.array([0.0, 0.0, 1.0, 0.0])
    rng = np.random.default_rng(2)
    env = Pendulum(max_steps=8)
    env.reset(seed=2)
    rewards = []
    for action in rng.uniform(-1.5, 1.5, size=6):
        _, reward, _, _ = env.step(float(action))
        rewards.append(reward)
    rewards = np.asarray(rewards, dtype=np.float64)
    want = _step_reference(
        mean, log_std, actions, old_mean, log_std, old_logp,
        advantages, returns, values, old_values, rewards,
    )
    if want["clip_fraction"] != 0.5:
        fail(
            "the fixture no longer leaves the clip band. "
            f"expected clip fraction 0.5, got {want['clip_fraction']}"
        )
    got = fn(
        mean, log_std, actions, old_mean, log_std, old_logp,
        advantages, returns, values, old_values, rewards,
    )
    _require_step(got, want)

    # On policy, the ratio is 1 and a small move stays inside the clip.
    on_actions = np.array([[0.5], [-0.25], [0.8], [0.1]])
    on_mean = np.array([[0.0], [0.2], [-0.1], [0.3]])
    on_logp, _, _ = _closed_form(on_mean, log_std, on_actions, on_mean, log_std)
    on_adv = np.array([0.2, -0.4, 0.6, -0.1])
    on_returns = np.array([1.0, 0.5, 0.0, -0.2])
    on_values = np.array([0.9, 0.4, 0.1, -0.1])
    on_old_values = on_values.copy()

    def loss_of_mean(flat):
        out = fn(
            flat.reshape(4, 1), log_std, on_actions, on_mean, log_std, on_logp,
            on_adv, on_returns, on_values, on_old_values, rewards,
        )
        return float(out["loss"])

    grad = finite_difference(loss_of_mean, on_mean.reshape(-1).copy())
    adv = (on_adv - on_adv.mean()) / (on_adv.std() + 1e-8)
    score = (on_actions[:, 0] - on_mean[:, 0]) / np.exp(2.0 * log_std)
    analytical = (-adv / on_adv.shape[0]) * score
    close(grad, analytical, atol=1e-4, label="d loss / d mean")
