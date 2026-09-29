"""Checks for the robotics and embodied agents exercise."""

import numpy as np

import rlenvs
from rlcheck import almost, check, close, fail, has, shape

KP, KD, HORIZON = 10.0, 1.5, 100
TEST_OFFSETS = np.linspace(-1.5, 1.5, 7)
_TRAINED = {}


def _env_return(w, offset, n=16, bound=1.0):
    """Mean return of base + residual on rlenvs.Pendulum itself, noise off."""
    rng = np.random.default_rng(2024)
    theta0, theta_dot0 = rng.uniform(-0.3, 0.3, n), rng.uniform(-0.3, 0.3, n)
    total = 0.0
    for i in range(n):
        env = rlenvs.Pendulum(max_steps=HORIZON)
        env.theta, env.theta_dot, env.t = float(theta0[i]), float(theta_dot0[i]), 0
        obs = env._obs()
        for _ in range(HORIZON):
            s, thd = obs[1], obs[2]
            u = -(KP * s + KD * thd) + bound * np.tanh(np.array([s, thd, 1.0]) @ w)
            obs, r, _, _ = env.step(u + offset)
            total += r
    return total / n


def _trained(sub):
    if not _TRAINED:
        train_residual = has(sub, "train_residual")
        _TRAINED["nominal"] = np.asarray(train_residual(1.0, 1.0, seed=0), dtype=float)
        _TRAINED["random"] = np.asarray(train_residual(-1.5, 1.5, seed=0), dtype=float)
        for key, w in _TRAINED.items():
            if w.shape != (3,) or not np.all(np.isfinite(w)):
                _TRAINED.clear()
                fail(f"train_residual should return three finite weights, got {w}")
    return _TRAINED


@check("pendulum_step matches rlenvs.Pendulum, including torque and speed clipping")
def test_pendulum_step_nominal(sub):
    pendulum_step = has(sub, "pendulum_step")
    rng = np.random.default_rng(0)
    theta = rng.uniform(-4, 4, 40)
    theta_dot = rng.uniform(-8, 8, 40)
    torque = rng.uniform(-3.5, 3.5, 40)  # some beyond the +-2 limit
    got = pendulum_step(theta, theta_dot, torque)
    if len(got) != 3:
        fail("pendulum_step should return (theta, theta_dot, reward)")
    want = np.zeros((3, 40))
    for i in range(40):
        env = rlenvs.Pendulum()
        env.theta, env.theta_dot = float(theta[i]), float(theta_dot[i])
        _, r, _, _ = env.step(torque[i])
        want[:, i] = env.theta, env.theta_dot, r
    for k, name in enumerate(["theta", "theta_dot", "reward"]):
        shape(got[k], (40,), label=name)
        close(np.asarray(got[k]), want[k], atol=1e-12, label=f"{name} after one step")


@check("pendulum_step with m = 1.3 and l = 0.8 matches rlenvs.Pendulum with those parameters")
def test_pendulum_step_randomised(sub):
    pendulum_step = has(sub, "pendulum_step")
    rng = np.random.default_rng(1)
    theta, theta_dot, torque = rng.uniform(-1, 1, 20), rng.uniform(-2, 2, 20), rng.uniform(-2, 2, 20)
    got = pendulum_step(theta, theta_dot, torque, m=1.3, l=0.8)
    for i in range(20):
        env = rlenvs.Pendulum()
        env.m, env.l = 1.3, 0.8  # instance attributes shadow the class defaults
        env.theta, env.theta_dot = float(theta[i]), float(theta_dot[i])
        _, r, _, _ = env.step(torque[i])
        close(np.array([got[0][i], got[1][i], got[2][i]]), np.array([env.theta, env.theta_dot, r]),
              atol=1e-12, label=f"state and reward for sample {i}")


@check("residual_action returns the base action at raw 0 and never moves further than bound from it")
def test_residual_action(sub):
    residual_action = has(sub, "residual_action")
    base = np.array([-1.2, 0.0, 0.7, 1.9])
    close(np.asarray(residual_action(base, np.zeros(4), 0.5)), base, atol=1e-15, label="raw = 0")
    raw = np.array([-50.0, -0.3, 0.4, 50.0])
    got = np.asarray(residual_action(base, raw, 0.5))
    close(got, base + 0.5 * np.tanh(raw), atol=1e-12, label="base + bound * tanh(raw)")
    if np.max(np.abs(got - base)) > 0.5 + 1e-12:
        fail(f"the residual moved the action by {np.max(np.abs(got - base)):.4f}, more than the bound 0.5")


@check("reinforce_gradient matches a hand computation with reward-to-go and a leave-one-out baseline")
def test_reinforce_gradient_hand(sub):
    reinforce_gradient = has(sub, "reinforce_gradient")
    rng = np.random.default_rng(2)
    n, horizon, sigma = 3, 2, 0.5
    rewards = rng.normal(size=(n, horizon))
    features = rng.normal(size=(n, horizon, 3))
    eps = rng.normal(size=(n, horizon))
    want = np.zeros(3)
    for i in range(n):
        for t in range(horizon):
            g_it = rewards[i, t:].sum()
            b_it = np.mean([rewards[j, t:].sum() for j in range(n) if j != i])
            want += (g_it - b_it) * eps[i, t] / sigma * features[i, t]
    want /= n
    got = np.asarray(reinforce_gradient(rewards, features, eps, sigma))
    shape(got, (3,), label="gradient")
    close(got, want, atol=1e-12, label="gradient on a 3-episode, 2-step batch")
    shifted = np.asarray(reinforce_gradient(rewards + 4.0, features, eps, sigma))
    close(shifted, got, atol=1e-10, label="gradient after adding 4 to every reward")


@check("reinforce_gradient recovers the worked example: base action 0.8, target 1, gradient 0.4")
def test_reinforce_gradient_worked(sub):
    reinforce_gradient = has(sub, "reinforce_gradient")
    rng = np.random.default_rng(3)
    n, sigma = 100_000, 0.1
    eps = rng.standard_normal((n, 1))
    action = 0.8 + 0.0 + sigma * eps[:, 0]  # base action plus a residual w = 0, plus noise
    rewards = -((action - 1.0) ** 2)[:, None]
    got = np.asarray(reinforce_gradient(rewards, np.ones((n, 1, 1)), eps, sigma))
    almost(got[0], 0.4, 0.02, label="d E[r] / d w at w = 0, exactly -2 (0.8 - 1)")


@check("rollout with zero weights and zero noise reproduces the base controller on rlenvs.Pendulum")
def test_rollout_base(sub):
    rollout = has(sub, "rollout")
    rng = np.random.default_rng(4)
    n = 5
    theta0, theta_dot0 = rng.uniform(-0.3, 0.3, n), rng.uniform(-0.3, 0.3, n)
    offsets = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    rewards, features = rollout(np.zeros(3), offsets, theta0, theta_dot0, np.zeros((n, 30)), 0.3)
    shape(rewards, (n, 30), label="rewards")
    shape(features, (n, 30, 3), label="features")
    for i in range(n):
        env = rlenvs.Pendulum()
        env.theta, env.theta_dot, env.t = float(theta0[i]), float(theta_dot0[i]), 0
        obs = env._obs()
        want = []
        for _ in range(30):
            if abs(obs[1] - np.sin(env.theta)) > 1e-12:
                fail("internal: observation mismatch")
            u = -(KP * obs[1] + KD * obs[2])
            obs, r, _, _ = env.step(u + offsets[i])
            want.append(r)
        close(rewards[i], np.array(want), atol=1e-10, label=f"rewards of episode {i} (offset {offsets[i]:+.1f})")
    close(features[:, 0, 0], np.sin(theta0), atol=1e-12, label="first feature at t = 0, sin(theta0)")
    close(features[:, :, 2], np.ones((n, 30)), atol=0, label="the constant feature")


@check("clip_reward reproduces the goal-baseline example: 0.60 with alpha 0 and 0.64 with alpha 1")
def test_clip_reward(sub):
    clip_reward = has(sub, "clip_reward")
    g, b, e = np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.6, 0.8])
    almost(float(clip_reward(e, g, b, 0.0)), 0.60, 1e-12, label="alpha = 0")
    almost(float(clip_reward(e, g, b, 1.0)), 0.64, 1e-12, label="alpha = 1")
    almost(float(clip_reward(e, g, b, 0.5)), 1 - 0.5 * (0.5**2 + 0.7**2), 1e-12, label="alpha = 0.5")
    rng = np.random.default_rng(5)
    s = rng.normal(size=(6, 8))
    g8, b8 = rng.normal(size=8), rng.normal(size=8)
    cos = (s / np.linalg.norm(s, axis=1, keepdims=True)) @ (g8 / np.linalg.norm(g8))
    close(np.asarray(clip_reward(s, g8, b8, 0.0)), cos, atol=1e-12,
          label="alpha = 0 equals cosine similarity, for a batch of unnormalised embeddings")


@check("With alpha 0 a texture-changing state outscores the goal; the goal-baseline projection fixes the order")
def test_clip_reward_hack(sub):
    clip_reward = has(sub, "clip_reward")
    pose_goal, pose_base, texture = np.eye(3)
    goal = pose_goal + 1.5 * texture      # the goal prompt also evokes a floor texture
    baseline = pose_base + 1.5 * texture  # and so does the baseline prompt
    kneeling = pose_goal                  # the task done, on a plain floor
    standing_on_texture = pose_base + 1.5 * texture
    r0 = np.asarray(clip_reward(np.stack([kneeling, standing_on_texture]), goal, baseline, 0.0))
    r1 = np.asarray(clip_reward(np.stack([kneeling, standing_on_texture]), goal, baseline, 1.0))
    close(r0, np.array([1 / np.sqrt(3.25), 2.25 / 3.25]), atol=1e-12, label="rewards with alpha = 0")
    if not r0[1] > r0[0]:
        fail(f"with alpha = 0 the texture state should score higher: {r0}")
    if not r1[0] > r1[1] + 0.2:
        fail(f"with alpha = 1 the goal state should win by over 0.2: {r1}")


@check("Residual RL at the nominal offset lifts the base controller's return from -4.3 to above -1.5")
def test_residual_nominal(sub):
    trained = _trained(sub)
    base = _env_return(np.zeros(3), 1.0)
    got = _env_return(trained["nominal"], 1.0)
    if got < -1.5:
        fail(f"the nominal residual should score above -1.5 at offset 1.0 (base: {base:.2f}), got {got:.2f}")


@check("Domain randomisation trades nominal return for a far better worst case across offsets")
def test_domain_randomisation(sub):
    trained = _trained(sub)
    nominal_curve = [_env_return(trained["nominal"], d) for d in TEST_OFFSETS]
    random_curve = [_env_return(trained["random"], d) for d in TEST_OFFSETS]
    worst_nominal, worst_random = min(nominal_curve), min(random_curve)
    if worst_random < -20.0 or worst_random < worst_nominal + 100.0:
        fail(
            f"over offsets {np.round(TEST_OFFSETS, 2).tolist()}, the randomised residual's worst return "
            f"should be above -20 and over 100 better than the nominal one's: got {worst_random:.1f} "
            f"against {worst_nominal:.1f}"
        )
    at_nominal_n = _env_return(trained["nominal"], 1.0)
    at_nominal_r = _env_return(trained["random"], 1.0)
    if at_nominal_n < at_nominal_r + 1.0:
        fail(
            f"at the nominal offset the nominal residual should beat the randomised one by over 1.0: "
            f"{at_nominal_n:.2f} against {at_nominal_r:.2f}"
        )
