"""Model-based reinforcement learning: learn a model, measure its drift, plan with it.

Fill in the functions below, in order. You fit a linear dynamics model by least
squares, roll it forward on its own predictions, measure how its error grows
with the horizon, and then plan through it with the cross-entropy method (CEM)
inside a model-predictive control (MPC) loop that balances CartPole.

Shapes used throughout:
    N  number of states in a batch
    d  state dimension (4 for CartPole)
    k  action dimension (1 for CartPole, where the action is 0.0 or 1.0)
    H  planning or prediction horizon
"""

import numpy as np

import rlenvs


def fit_linear_model(states, actions, next_states, ridge=0.0):
    """Fit next_state = W^T [state, action, 1] by ridge-regularised least squares.

    Args:
        states: shape (N, d).
        actions: shape (N, k). A shape (N,) array is read as k = 1.
        next_states: shape (N, d).
        ridge: the penalty lam >= 0 on every entry of W, including the bias row.

    Returns:
        W, shape (d + k + 1, d), minimising
        0.5 * ||X W - next_states||^2 + 0.5 * ridge * ||W||^2, where each row of
        X is [state, action, 1]. Rows 0..d-1 of W hold A^T, the next k rows hold
        B^T, and the last row holds the bias c.
    """
    raise NotImplementedError("fit_linear_model")


def model_step(W, states, actions):
    """Predict next states with a fitted linear model.

    Args:
        W: shape (d + k + 1, d), from fit_linear_model.
        states: shape (N, d).
        actions: shape (N, k), or (N,) for k = 1.

    Returns:
        Predicted next states, shape (N, d).
    """
    raise NotImplementedError("model_step")


def rollout(step_fn, s0, actions):
    """Run a model open loop: each prediction becomes the next input.

    Args:
        step_fn: a function (states (N, d), actions (N, k)) -> next states (N, d).
        s0: start states, shape (N, d).
        actions: an action sequence per start state, shape (N, H, k).

    Returns:
        The visited states, shape (N, H + 1, d). Entry [:, 0] is s0 and entry
        [:, t + 1] is step_fn applied to entry [:, t] and actions[:, t].
    """
    raise NotImplementedError("rollout")


def k_step_errors(true_step, learned_step, s0, actions):
    """How far a learned model drifts from the truth over an open-loop rollout.

    Args:
        true_step: the real dynamics, same signature as step_fn in rollout.
        learned_step: the model, same signature.
        s0: start states, shape (N, d).
        actions: shape (N, H, k), the same actions fed to both.

    Returns:
        Shape (H,). Entry k - 1 is the mean over the batch of the Euclidean
        distance between the model's and the real state after k steps.
    """
    raise NotImplementedError("k_step_errors")


def cem_update(samples, scores, n_elite):
    """One cross-entropy-method refit.

    Args:
        samples: candidate solutions, shape (n, D).
        scores: shape (n,). Higher is better.
        n_elite: how many of the best samples to keep.

    Returns:
        (mean, std), each shape (D,): the mean and the population standard
        deviation (ddof=0) of the n_elite highest-scoring samples.
    """
    raise NotImplementedError("cem_update")


def cem_optimize(score_fn, mean, std, n_samples, n_elite, n_iters, rng):
    """Maximise score_fn with the cross-entropy method.

    Args:
        score_fn: a function from samples (n_samples, D) to scores (n_samples,).
        mean, std: the starting Gaussian, each shape (D,).
        n_samples, n_elite, n_iters: population size, elite count, iterations.
        rng: a numpy Generator. Draw each population as
            mean + std * rng.standard_normal((n_samples, D)).

    Returns:
        The final mean, shape (D,).
    """
    raise NotImplementedError("cem_optimize")


def mpc_action(step_fn, reward_fn, state, horizon, rng, n_samples=64, n_elite=8, n_iters=4):
    """Plan with CEM through a model and return the first action (model-predictive control).

    Each plan is a real vector u of length `horizon`, starting from a standard
    Gaussian (mean 0, std 1). Plan u pushes right at step t, action 1.0, when
    u[t] > 0, and left, action 0.0, otherwise.

    Args:
        step_fn: the model, (states (N, 4), actions (N, 1)) -> next states (N, 4).
        reward_fn: a function from states (N, H, 4) to per-step rewards (N, H).
        state: the current real state, shape (4,).
        horizon: the planning horizon H.
        rng: a numpy Generator, passed to cem_optimize.

    Returns:
        The first action of the best plan, as the int 0 or 1. A plan's score is
        the sum of reward_fn over the H states it visits after the start.
    """
    raise NotImplementedError("mpc_action")


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
