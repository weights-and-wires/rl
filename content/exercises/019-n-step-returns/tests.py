"""Checks for the n-step returns exercise."""

import numpy as np

import rlenvs
from rlcheck import almost, check, close, fail, has, shape


def _walk_runs(n_runs, n_episodes, seed):
    # Episodes drawn with plain coin flips on the 19-state walk, start in state 10.
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(n_runs):
        episodes = []
        for _ in range(n_episodes):
            s = 10
            states, rewards = [s], []
            while 0 < s < 20:
                s = s + (1 if rng.random() < 0.5 else -1)
                states.append(s)
                rewards.append(1.0 if s == 20 else (-1.0 if s == 0 else 0.0))
            episodes.append((np.array(states), np.array(rewards)))
        runs.append(episodes)
    return runs


_TRUTH = (np.arange(1, 20) - 10) / 10


def _rms(V):
    V = np.asarray(V, dtype=float)
    return float(np.sqrt(np.mean((V[1:20] - _TRUTH) ** 2)))


@check("n_step_return gives 1.9, 1.648, and 2.62 on the worked episode, and G_t once t + n reaches T")
def test_n_step_return(sub):
    f = has(sub, "n_step_return")
    r = np.array([1.0, 0.0, 2.0])
    v = np.array([0.5, 1.0, 0.8])
    almost(f(r, v, 0, 1, 0.9), 1.9, 1e-12, label="G_{0:1}")
    almost(f(r, v, 0, 2, 0.9), 1.648, 1e-12, label="G_{0:2}")
    almost(f(r, v, 0, 3, 0.9), 2.62, 1e-12, label="G_{0:3}")
    almost(f(r, v, 0, 10, 0.9), 2.62, 1e-12, label="G_{0:10}, past the end")
    # From t = 1: 0 + 0.9 * 0.8 = 0.72, then 0 + 0.9 * 2 = 1.8.
    almost(f(r, v, 1, 1, 0.9), 0.72, 1e-12, label="G_{1:2}")
    almost(f(r, v, 1, 2, 0.9), 1.8, 1e-12, label="G_{1:3}")
    # At the last step there is no next state, so the return is the reward alone.
    almost(f(r, v, 2, 1, 0.9), 2.0, 1e-12, label="G_{2:3}, the last step")


@check("expected_n_step_return matches the exact formula and shrinks the error by gamma^n")
def test_error_reduction(sub):
    f = has(sub, "expected_n_step_return")
    mdp = rlenvs.random_mdp(n_states=6, n_actions=3, gamma=0.9, seed=1)
    pi = rlenvs.uniform_policy(mdp)
    P_pi = np.einsum("sa,sat->st", pi, mdp.P)
    r_pi = (pi * mdp.R).sum(axis=1)
    v_pi = rlenvs.policy_evaluation(mdp, pi)
    V = v_pi + 2.0 * np.random.default_rng(0).normal(size=6)
    err = np.max(np.abs(V - v_pi))
    for n in range(1, 7):
        got = np.asarray(f(P_pi, r_pi, V, n, 0.9), dtype=float)
        want = sum(0.9**k * np.linalg.matrix_power(P_pi, k) @ r_pi for k in range(n))
        want = want + 0.9**n * np.linalg.matrix_power(P_pi, n) @ V
        close(got, want, atol=1e-10, label=f"expected {n}-step return")
        worst = np.max(np.abs(got - v_pi))
        if worst > 0.9**n * err + 1e-10:
            fail(f"n = {n}: error {worst:.4f} exceeds gamma^n times the starting error, {0.9**n * err:.4f}")


@check("n_step_td on one episode gives the TD(0), two-step, and Monte Carlo updates for n = 1, 2, 3")
def test_n_step_td_by_hand(sub):
    f = has(sub, "n_step_td")
    # States 0, 1, 2, then the end (3). Rewards 1, 0, 2. With alpha = 1 and
    # every value starting at 0.5, each update replaces V with its target.
    episodes = [(np.array([0, 1, 2, 3]), np.array([1.0, 0.0, 2.0]))]
    # n = 1: V[0] = 1 + 0.9 * 0.5 = 1.45, V[1] = 0 + 0.9 * 0.5 = 0.45, V[2] = 2.
    V = f(episodes, 4, 1, 1.0, 0.9, v_init=0.5)
    shape(V, (4,), label="n_step_td output")
    close(V, np.array([1.45, 0.45, 2.0, 0.0]), atol=1e-12, label="n = 1")
    # n = 2: V[0] = 1 + 0 + 0.81 * 0.5 = 1.405, then the windows reach the end.
    V = f(episodes, 4, 2, 1.0, 0.9, v_init=0.5)
    close(V, np.array([1.405, 1.8, 2.0, 0.0]), atol=1e-12, label="n = 2")
    # n = 3 reaches the end from every state: the Monte Carlo returns 2.62, 1.8, 2.
    V = f(episodes, 4, 3, 1.0, 0.9, v_init=0.5)
    close(V, np.array([2.62, 1.8, 2.0, 0.0]), atol=1e-12, label="n = 3")


@check("n_step_td with n = 4 beats n = 1 and n = 64 on the 19-state walk (30 runs of 10 episodes)")
def test_n_step_td(sub):
    f = has(sub, "n_step_td")
    runs = _walk_runs(30, 10, seed=7)
    err = {}
    for n in (1, 4, 64):
        values = []
        for episodes in runs:
            V = f(episodes, 21, n, 0.2, 1.0)
            shape(V, (21,), label="n_step_td output")
            values.append(_rms(V))
        err[n] = float(np.mean(values))
    # Seeded replications give about 0.41, 0.18, and 0.51.
    if not (err[4] < err[1] - 0.1 and err[4] < err[64] - 0.1):
        fail(
            "with alpha = 0.2, expected RMS errors near 0.41 for n = 1, 0.18 for n = 4, and "
            f"0.51 for n = 64; got {err[1]:.3f}, {err[4]:.3f}, and {err[64]:.3f}"
        )
