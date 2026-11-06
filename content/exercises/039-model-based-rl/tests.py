"""Checks for the model-based reinforcement learning exercise."""

import numpy as np

import rlenvs
from rlcheck import almost, check, close, fail, has, shape


def _affine_system():
    A = np.array([[1.0, 0.1], [-0.2, 0.95]])
    B = np.array([[0.0], [0.5]])
    c = np.array([0.05, -0.02])
    return A, B, c


def _true_cartpole(states, actions):
    return rlenvs.CartPole.dynamics(states, np.asarray(actions)[:, 0] > 0.5)


def _cartpole_reward(states):
    x, theta = states[..., 0], states[..., 2]
    upright = 1.0 - (theta / 0.21) ** 2 - 0.1 * (x / 2.4) ** 2
    return np.where(rlenvs.CartPole.failed(states), -10.0, upright)


def _cartpole_data(n, rng):
    scale = np.array([0.5, 0.5, 0.1, 0.5])
    states = rng.uniform(-1.0, 1.0, size=(n, 4)) * scale
    actions = rng.integers(0, 2, size=(n, 1)).astype(float)
    return states, actions, _true_cartpole(states, actions)


def _run_mpc(sub, step_fn, seed):
    mpc_action = has(sub, "mpc_action")
    rng = np.random.default_rng(seed)
    env = rlenvs.CartPole(max_steps=200, seed=seed)
    s = env.reset()
    steps, worst = 0, 0.0
    while True:
        a = mpc_action(step_fn, _cartpole_reward, s, horizon=15, rng=rng,
                       n_samples=64, n_elite=8, n_iters=4)
        if int(a) not in (0, 1):
            fail(f"mpc_action must return 0 or 1, got {a!r}")
        s, _, terminated, truncated = env.step(int(a))
        steps += 1
        worst = max(worst, abs(s[2]))
        if terminated or truncated:
            return steps, worst


@check("fit_linear_model recovers a known affine system exactly, and with ridge > 0 zeroes the ridge gradient")
def test_fit(sub):
    fit = has(sub, "fit_linear_model")
    A, B, c = _affine_system()
    rng = np.random.default_rng(0)
    states = rng.standard_normal((50, 2))
    actions = rng.standard_normal((50, 1))
    next_states = states @ A.T + actions @ B.T + c
    W = fit(states, actions, next_states, ridge=0.0)
    shape(W, (4, 2), label="W for 2 state and 1 action dimensions")
    want = np.concatenate([A.T, B.T, c[None, :]], axis=0)
    close(W, want, atol=1e-9, label="W stacked as [A^T; B^T; c]")
    rng = np.random.default_rng(1)
    states = rng.standard_normal((30, 3))
    actions = rng.standard_normal((30, 2))
    next_states = rng.standard_normal((30, 3))
    lam = 0.7
    W = fit(states, actions, next_states, ridge=lam)
    X = np.concatenate([states, actions, np.ones((30, 1))], axis=1)
    # Gradient of 0.5 * ||XW - Y||^2 + 0.5 * lam * ||W||^2.
    grad = X.T @ (X @ W - next_states) + lam * W
    close(grad, np.zeros_like(grad), atol=1e-9, label="gradient of the ridge objective at your W")


@check("model_step applies W to the features [s, a, 1]")
def test_model_step(sub):
    model_step = has(sub, "model_step")
    A, B, c = _affine_system()
    W = np.concatenate([A.T, B.T, c[None, :]], axis=0)
    s = np.array([[1.0, 2.0], [0.0, -1.0]])
    a = np.array([[1.0], [-2.0]])
    got = model_step(W, s, a)
    shape(got, (2, 2), label="model_step output")
    close(got, s @ A.T + a @ B.T + c, atol=1e-12, label="predicted next states")


@check("rollout feeds each prediction back in and matches the closed form A^k s0 + sum A^(k-1-j) B a_j")
def test_rollout(sub):
    rollout = has(sub, "rollout")
    A, B, _ = _affine_system()
    rng = np.random.default_rng(2)
    s0 = rng.standard_normal((3, 2))
    actions = rng.standard_normal((3, 6, 1))
    got = rollout(lambda s, a: s @ A.T + a @ B.T, s0, actions)
    shape(got, (3, 7, 2), label="rollout output for N=3, H=6, d=2")
    want = np.zeros((3, 7, 2))
    for k in range(7):
        total = s0 @ np.linalg.matrix_power(A, k).T
        for j in range(k):
            total = total + actions[:, j] @ (np.linalg.matrix_power(A, k - 1 - j) @ B).T
        want[:, k] = total
    close(got, want, atol=1e-10, label="rolled-out states")


@check("k_step_errors matches 1.01^k - 1 for a model that overestimates a unit growth rate by 1%")
def test_k_step_scalar(sub):
    k_step_errors = has(sub, "k_step_errors")
    s0 = np.ones((1, 1))
    actions = np.zeros((1, 50, 1))
    got = k_step_errors(lambda s, a: 1.0 * s, lambda s, a: 1.01 * s, s0, actions)
    shape(got, (50,), label="errors for H=50")
    k = np.arange(1, 51)
    close(got, 1.01 ** k - 1.0, rtol=1e-9, atol=1e-12, label="error after k steps")
    almost(got[0], 0.01, 1e-12, label="one-step error")
    almost(got[49], 1.01 ** 50 - 1.0, 1e-9, label="50-step error, about 0.645")


@check("On CartPole, the learned linear model's 30-step error is over 100 times its one-step error")
def test_k_step_cartpole(sub):
    fit = has(sub, "fit_linear_model")
    model_step = has(sub, "model_step")
    k_step_errors = has(sub, "k_step_errors")
    rng = np.random.default_rng(0)
    states, actions, next_states = _cartpole_data(2000, rng)
    W = fit(states, actions, next_states, ridge=1e-6)
    s0 = rng.uniform(-0.05, 0.05, size=(100, 4))
    plans = rng.integers(0, 2, size=(100, 30, 1)).astype(float)
    got = k_step_errors(_true_cartpole, lambda s, a: model_step(W, s, a), s0, plans)
    shape(got, (30,), label="errors for H=30")

    # An independent open-loop rollout of both models.
    real, imag = s0.copy(), s0.copy()
    want = []
    for t in range(30):
        real = _true_cartpole(real, plans[:, t])
        X = np.concatenate([imag, plans[:, t], np.ones((100, 1))], axis=1)
        imag = X @ W
        want.append(np.linalg.norm(imag - real, axis=1).mean())
    close(got, np.array(want), rtol=1e-6, atol=1e-10, label="mean k-step error")
    if got[29] < 100 * got[0]:
        fail(f"expected the error to compound: 1-step {got[0]:.2e}, 30-step {got[29]:.2e}")


@check("cem_update refits the mean and standard deviation to the top n_elite samples")
def test_cem_update(sub):
    cem_update = has(sub, "cem_update")
    samples = np.array([[-1.0], [0.5], [2.0], [1.2], [0.3]])
    scores = -((samples[:, 0] - 1.0) ** 2)
    mean, std = cem_update(samples, scores, n_elite=2)
    # Elites are 1.2 (score -0.04) and 0.5 (score -0.25).
    close(np.asarray(mean).ravel(), [0.85], atol=1e-12, label="elite mean")
    close(np.asarray(std).ravel(), [0.35], atol=1e-12, label="elite standard deviation (ddof=0)")


@check("cem_optimize finds the maximiser of a concave quadratic within 1e-3")
def test_cem_optimize(sub):
    cem_optimize = has(sub, "cem_optimize")
    target = np.array([1.0, -2.0, 0.5])
    rng = np.random.default_rng(0)
    got = cem_optimize(lambda x: -((x - target) ** 2).sum(axis=1), np.zeros(3), 2.0 * np.ones(3),
                       n_samples=100, n_elite=10, n_iters=30, rng=rng)
    shape(got, (3,), label="optimised mean")
    close(got, target, atol=1e-3, label="CEM's final mean")


@check("mpc_action with a linear model fitted to 2000 transitions also balances for 200 steps")
def test_mpc_learned(sub):
    fit = has(sub, "fit_linear_model")
    model_step = has(sub, "model_step")
    rng = np.random.default_rng(1)
    states, actions, next_states = _cartpole_data(2000, rng)
    W = fit(states, actions, next_states, ridge=1e-6)
    steps, worst = _run_mpc(sub, lambda s, a: model_step(W, s, a), seed=1)
    if steps < 200:
        fail(f"with the learned model the pole fell after {steps} steps; expected 200")
    if worst > 0.15:
        fail(f"the pole leaned to {worst:.3f} rad; expected MPC to keep it under 0.15")
