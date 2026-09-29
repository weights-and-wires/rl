"""Robotics: residual RL on a fixed controller, domain randomisation, and VLM rewards.

Fill in the functions under "The exercise". The robot above them is given: a
pendulum that must stay upright, a PD `base_controller` that knows nothing of
the motor's calibration offset, the `residual_features` the learned
correction sees, `start_states`, and an `adam_step` helper.
"""

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
    """One step of rlenvs.Pendulum's physics for a whole batch at once.

    Args:
        theta, theta_dot, torque: arrays of the same shape `(N,)`. `theta` is
            the angle from upright, not wrapped. `torque` is the motor's
            output before the +-2 clip.
        m, l: the pendulum's mass and length, floats or `(N,)` arrays, so a
            batch can hold many randomised pendulums.

    Returns:
        `(theta, theta_dot, reward)` after the step, each shape `(N,)`,
        identical to what `rlenvs.Pendulum.step` gives one pendulum at a time.
    """
    raise NotImplementedError("pendulum_step")


def residual_action(base_action, raw, bound):
    """The base action plus a residual squashed into (-bound, bound).

    Returns base_action + bound * tanh(raw), elementwise.
    """
    raise NotImplementedError("residual_action")


def rollout(w, offsets, theta0, theta_dot0, eps, sigma, bound=1.0):
    """Run N episodes of base controller plus residual policy, in parallel.

    At each step t, for every episode: phi = residual_features(theta, theta_dot),
    raw = phi @ w + sigma * eps[:, t], command = residual_action(base, raw, bound)
    with base = base_controller(theta, theta_dot), and the motor applies
    command + offsets through `pendulum_step` with m = l = 1.

    Args:
        w: residual weights, shape `(3,)`.
        offsets: each episode's motor offset, shape `(N,)`.
        theta0, theta_dot0: start states, shape `(N,)`.
        eps: standard normal exploration noise, shape `(N, T)`. T sets the
            number of steps.
        sigma: the exploration noise scale.
        bound: the residual's bound.

    Returns:
        `(rewards, features)`: rewards of shape `(N, T)`, where rewards[:, t]
        is paid for the step taken at t, and the features phi seen at each
        step, shape `(N, T, 3)`.
    """
    raise NotImplementedError("rollout")


def reinforce_gradient(rewards, features, eps, sigma):
    """REINFORCE estimate of the gradient of the expected return with respect to w.

    Uses the reward-to-go G[i, t] = sum_{t' >= t} rewards[i, t'], and for each
    episode i a leave-one-out baseline b[i, t], the mean of G[j, t] over the
    other episodes j != i.

    Args:
        rewards: shape `(N, T)`.
        features: the phi at each step, shape `(N, T, d)`.
        eps: the noise that produced each raw action, shape `(N, T)`.
        sigma: the noise scale.

    Returns:
        Shape `(d,)`: (1 / N) * sum_i sum_t (G[i, t] - b[i, t]) * eps[i, t] / sigma * features[i, t].
    """
    raise NotImplementedError("reinforce_gradient")


def train_residual(offset_low, offset_high, seed, iters=150, n_episodes=64, sigma=0.3, bound=1.0, lr=0.05):
    """Train residual weights with REINFORCE and Adam, starting from w = 0.

    Seed `np.random.default_rng(seed)`. Each iteration: spread the offsets
    evenly with rng.permutation(np.linspace(offset_low, offset_high, n_episodes)),
    draw start states with `start_states`, draw noise of shape
    `(n_episodes, HORIZON)`, run `rollout`, and take `adam_step` up the
    `reinforce_gradient`. Equal `offset_low` and `offset_high` mean no
    randomisation.

    Returns:
        The trained weights, shape `(3,)`.
    """
    raise NotImplementedError("train_residual")


def clip_reward(state_emb, goal_emb, baseline_emb, alpha):
    """VLM-RM's goal-baseline regularised CLIP reward.

    Normalise all three embeddings to unit length. Let L be the line through
    the baseline b and the goal g. The reward is
    1 - 0.5 * |alpha * proj_L(s) + (1 - alpha) * s - g|^2.

    Args:
        state_emb: image embeddings, shape `(..., d)`.
        goal_emb: the goal prompt's embedding, shape `(d,)`.
        baseline_emb: the baseline prompt's embedding, shape `(d,)`.
        alpha: the regularisation strength, in [0, 1].

    Returns:
        The rewards, shape `(...)`.
    """
    raise NotImplementedError("clip_reward")


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
