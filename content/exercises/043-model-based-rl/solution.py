"""Reference solution for the model-based reinforcement learning exercise."""

import numpy as np

import rlenvs


def fit_linear_model(states, actions, next_states, ridge=0.0):
    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=float).reshape(len(states), -1)
    next_states = np.asarray(next_states, dtype=float)
    X = np.concatenate([states, actions, np.ones((len(states), 1))], axis=1)
    gram = X.T @ X + ridge * np.eye(X.shape[1])
    return np.linalg.solve(gram, X.T @ next_states)


def model_step(W, states, actions):
    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=float).reshape(len(states), -1)
    X = np.concatenate([states, actions, np.ones((len(states), 1))], axis=1)
    return X @ W


def rollout(step_fn, s0, actions):
    s0 = np.asarray(s0, dtype=float)
    actions = np.asarray(actions, dtype=float)
    horizon = actions.shape[1]
    states = [s0]
    for t in range(horizon):
        states.append(step_fn(states[-1], actions[:, t]))
    return np.stack(states, axis=1)


def k_step_errors(true_step, learned_step, s0, actions):
    real = rollout(true_step, s0, actions)
    imagined = rollout(learned_step, s0, actions)
    gaps = np.linalg.norm(imagined[:, 1:] - real[:, 1:], axis=-1)
    return gaps.mean(axis=0)


def cem_update(samples, scores, n_elite):
    samples = np.asarray(samples, dtype=float)
    scores = np.asarray(scores, dtype=float)
    elite = samples[np.argsort(-scores, kind="stable")[:n_elite]]
    return elite.mean(axis=0), elite.std(axis=0)


def cem_optimize(score_fn, mean, std, n_samples, n_elite, n_iters, rng):
    mean = np.array(mean, dtype=float)
    std = np.array(std, dtype=float)
    for _ in range(n_iters):
        samples = mean + std * rng.standard_normal((n_samples,) + mean.shape)
        mean, std = cem_update(samples, score_fn(samples), n_elite)
    return mean


def mpc_action(step_fn, reward_fn, state, horizon, rng, n_samples=64, n_elite=8, n_iters=4):
    state = np.asarray(state, dtype=float)

    def score(plans):
        n = plans.shape[0]
        actions = (plans > 0).astype(float)[:, :, None]
        starts = np.repeat(state[None, :], n, axis=0)
        traj = rollout(step_fn, starts, actions)
        return reward_fn(traj[:, 1:]).sum(axis=1)

    plan = cem_optimize(
        score, np.zeros(horizon), np.ones(horizon), n_samples, n_elite, n_iters, rng
    )
    return int(plan[0] > 0)


# Helpers shared by the checks and the demo -----------------------------------


def cartpole_true_step(states, actions):
    """The true CartPole physics, taking actions of shape (N, 1) with values 0 or 1."""
    return rlenvs.CartPole.dynamics(states, np.asarray(actions)[:, 0] > 0.5)


def cartpole_reward(states):
    """Per-step planning reward, shape (N, H): high while upright and centred."""
    x, theta = states[..., 0], states[..., 2]
    upright = 1.0 - (theta / 0.21) ** 2 - 0.1 * (x / 2.4) ** 2
    return np.where(rlenvs.CartPole.failed(states), -10.0, upright)


def cartpole_dataset(n, rng):
    """One-step transitions from random states near upright and random pushes."""
    scale = np.array([0.5, 0.5, 0.1, 0.5])
    states = rng.uniform(-1.0, 1.0, size=(n, 4)) * scale
    actions = rng.integers(0, 2, size=(n, 1)).astype(float)
    return states, actions, cartpole_true_step(states, actions)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    states, actions, next_states = cartpole_dataset(2000, rng)
    W = fit_linear_model(states, actions, next_states, ridge=1e-6)

    def learned(s, a):
        return model_step(W, s, a)

    s0 = rng.uniform(-0.05, 0.05, size=(100, 4))
    plans = rng.integers(0, 2, size=(100, 30, 1)).astype(float)
    errors = k_step_errors(cartpole_true_step, learned, s0, plans)
    print("k-step prediction error of the learned linear model")
    for k in (1, 2, 5, 10, 20, 30):
        print(f"  k = {k:2d}: {errors[k - 1]:.5f}")

    env = rlenvs.CartPole(max_steps=200, seed=0)
    s = env.reset()
    steps = 0
    while True:
        a = mpc_action(learned, cartpole_reward, s, horizon=15, rng=rng)
        s, _, terminated, truncated = env.step(a)
        steps += 1
        if terminated or truncated:
            break
    print(f"MPC with the learned model balanced the pole for {steps} steps")
