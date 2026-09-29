"""Reference solution for the robotics and embodied agents exercise."""

import numpy as np

import rlenvs

# The robot, its fixed base controller, and helpers. The same code sits in
# starter.py. ---------------------------------------------------------------

# The task: hold the pendulum upright (theta = 0) for HORIZON steps, starting
# near the top. The motor has an unknown calibration offset: every command u
# comes out as u + offset. The simulator's best guess is NOMINAL_OFFSET.
HORIZON = 100
NOMINAL_OFFSET = 1.0
KP, KD = 10.0, 1.5


def base_controller(theta, theta_dot):
    """A PD controller tuned without knowing about the motor offset."""
    return -(KP * np.sin(theta) + KD * theta_dot)


def residual_features(theta, theta_dot):
    """What the residual policy sees: [sin(theta), theta_dot, 1], shape (..., 3)."""
    theta = np.asarray(theta, dtype=float)
    return np.stack([np.sin(theta), np.asarray(theta_dot, dtype=float), np.ones_like(theta)], axis=-1)


def start_states(n, rng):
    """`n` start states near upright: theta and theta_dot uniform in [-0.3, 0.3]."""
    return rng.uniform(-0.3, 0.3, n), rng.uniform(-0.3, 0.3, n)


def adam_step(w, grad, state, lr):
    """One Adam step that *ascends* `grad`. `state` is a dict, empty at the start."""
    state.setdefault("m", np.zeros_like(w))
    state.setdefault("v", np.zeros_like(w))
    state["t"] = state.get("t", 0) + 1
    state["m"] = 0.9 * state["m"] + 0.1 * grad
    state["v"] = 0.999 * state["v"] + 0.001 * grad**2
    m_hat = state["m"] / (1 - 0.9 ** state["t"])
    v_hat = state["v"] / (1 - 0.999 ** state["t"])
    return w + lr * m_hat / (np.sqrt(v_hat) + 1e-8)


# The exercise. ----------------------------------------------------------


def pendulum_step(theta, theta_dot, torque, m=1.0, l=1.0):
    g, dt = rlenvs.Pendulum.g, rlenvs.Pendulum.dt
    u = np.clip(torque, -rlenvs.Pendulum.max_torque, rlenvs.Pendulum.max_torque)
    angle = rlenvs.Pendulum.angle_normalize(theta)
    reward = -(angle**2 + 0.1 * theta_dot**2 + 0.001 * u**2)
    new_theta_dot = theta_dot + (3 * g / (2 * l) * np.sin(theta) + 3.0 / (m * l**2) * u) * dt
    new_theta_dot = np.clip(new_theta_dot, -rlenvs.Pendulum.max_speed, rlenvs.Pendulum.max_speed)
    new_theta = theta + new_theta_dot * dt
    return new_theta, new_theta_dot, reward


def residual_action(base_action, raw, bound):
    return base_action + bound * np.tanh(raw)


def rollout(w, offsets, theta0, theta_dot0, eps, sigma, bound=1.0):
    theta = np.array(theta0, dtype=float)
    theta_dot = np.array(theta_dot0, dtype=float)
    n, horizon = eps.shape
    rewards = np.zeros((n, horizon))
    features = np.zeros((n, horizon, 3))
    for t in range(horizon):
        phi = residual_features(theta, theta_dot)
        raw = phi @ w + sigma * eps[:, t]
        command = residual_action(base_controller(theta, theta_dot), raw, bound)
        # The motor adds its calibration offset to whatever it is told.
        theta, theta_dot, rewards[:, t] = pendulum_step(theta, theta_dot, command + offsets)
        features[:, t] = phi
    return rewards, features


def reinforce_gradient(rewards, features, eps, sigma):
    n = rewards.shape[0]
    # Reward-to-go: G[i, t] = sum of rewards from step t to the end of episode i.
    to_go = np.cumsum(rewards[:, ::-1], axis=1)[:, ::-1]
    # Leave-one-out baseline: the mean reward-to-go of the other episodes.
    baseline = (to_go.sum(axis=0, keepdims=True) - to_go) / (n - 1)
    # Score of a Gaussian pre-squash action: (raw - mean) / sigma^2 * phi = eps / sigma * phi.
    score = eps[:, :, None] * features / sigma
    return np.sum((to_go - baseline)[:, :, None] * score, axis=(0, 1)) / n


def train_residual(offset_low, offset_high, seed, iters=150, n_episodes=64, sigma=0.3, bound=1.0, lr=0.05):
    rng = np.random.default_rng(seed)
    w = np.zeros(3)
    state = {}
    for _ in range(iters):
        # Domain randomisation: spread the offsets evenly over the range.
        offsets = rng.permutation(np.linspace(offset_low, offset_high, n_episodes))
        theta0, theta_dot0 = start_states(n_episodes, rng)
        eps = rng.standard_normal((n_episodes, HORIZON))
        rewards, features = rollout(w, offsets, theta0, theta_dot0, eps, sigma, bound)
        w = adam_step(w, reinforce_gradient(rewards, features, eps, sigma), state, lr)
    return w


def clip_reward(state_emb, goal_emb, baseline_emb, alpha):
    s = np.asarray(state_emb, dtype=float)
    s = s / np.linalg.norm(s, axis=-1, keepdims=True)
    g = np.asarray(goal_emb, dtype=float)
    g = g / np.linalg.norm(g)
    b = np.asarray(baseline_emb, dtype=float)
    b = b / np.linalg.norm(b)
    d = (g - b) / np.linalg.norm(g - b)
    # Project s onto the line through b and g.
    proj = b + ((s - b) @ d)[..., None] * d
    mixed = alpha * proj + (1 - alpha) * s
    return 1.0 - 0.5 * np.sum((mixed - g) ** 2, axis=-1)


if __name__ == "__main__":
    def evaluate(w, offset, n=50):
        theta0, theta_dot0 = start_states(n, np.random.default_rng(123))
        rewards, _ = rollout(w, np.full(n, offset), theta0, theta_dot0, np.zeros((n, HORIZON)), 0.0)
        return rewards.sum(axis=1).mean()

    w_nominal = train_residual(NOMINAL_OFFSET, NOMINAL_OFFSET, seed=0)
    w_random = train_residual(-1.5, 1.5, seed=0)
    print("residual weights, nominal:", np.round(w_nominal, 3), " randomised:", np.round(w_random, 3))
    print("\nmotor offset   base     nominal residual   randomised residual")
    for offset in np.linspace(-1.5, 1.5, 7):
        print(f"{offset:+11.2f} {evaluate(np.zeros(3), offset):8.2f} "
              f"{evaluate(w_nominal, offset):18.2f} {evaluate(w_random, offset):21.2f}")

    g, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    e = np.array([0.6, 0.8])
    print("\nVLM-RM reward of (0.6, 0.8): alpha=0 ->", round(float(clip_reward(e, g, b, 0.0)), 4),
          " alpha=1 ->", round(float(clip_reward(e, g, b, 1.0)), 4))
