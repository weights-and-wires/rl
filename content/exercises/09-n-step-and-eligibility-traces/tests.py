"""Checks for the n-step returns and eligibility traces exercise."""

import numpy as np

import rlenvs
from rlcheck import almost, check, close, fail, has, shape


def _walk19():
    n = 19
    S = n + 2
    P = np.zeros((S, 1, S))
    R3 = np.zeros((S, 1, S))
    for s in range(1, n + 1):
        P[s, 0, s - 1] = 0.5
        P[s, 0, s + 1] = 0.5
    P[0, 0, 0] = 1.0
    P[S - 1, 0, S - 1] = 1.0
    R3[1, 0, 0] = -1.0
    R3[n, 0, n + 1] = 1.0
    terminal = np.zeros(S, dtype=bool)
    terminal[[0, S - 1]] = True
    mu0 = np.zeros(S)
    mu0[10] = 1.0
    R = (P * R3).sum(axis=2)
    return rlenvs.TabularMDP(P=P, R=R, gamma=1.0, mu0=mu0, terminal=terminal, R3=R3)


def _walk_runs(n_runs, n_episodes, seed):
    # Episodes drawn with plain coin flips: the same walk, quicker to sample.
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


def _forward_lambda_returns(rewards, values, gamma, lam):
    # The lambda-return as the explicit weighted sum of n-step returns.
    T = len(rewards)
    out = np.zeros(T)
    for t in range(T):
        def G(n):
            end = min(t + n, T)
            g = sum(gamma ** (k - t) * rewards[k] for k in range(t, end))
            if t + n < T:
                g += gamma**n * values[t + n]
            return g
        out[t] = (1 - lam) * sum(lam ** (n - 1) * G(n) for n in range(1, T - t)) + lam ** (T - t - 1) * G(T - t)
    return out


def _random_episode(seed, n_states=5, T=12):
    rng = np.random.default_rng(seed)
    states = np.concatenate([rng.integers(n_states, size=T), [n_states]])
    rewards = rng.normal(size=T)
    V = np.concatenate([rng.normal(size=n_states), [0.0]])
    return states, rewards, V


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


@check("lambda_return gives 2.017 on the worked episode and matches the weighted sum of n-step returns")
def test_lambda_return(sub):
    f = has(sub, "lambda_return")
    r = np.array([1.0, 0.0, 2.0])
    v = np.array([0.5, 1.0, 0.8])
    got = f(r, v, 0.9, 0.5)
    shape(got, (3,), label="lambda_return output")
    almost(got[0], 2.017, 1e-12, label="G^lambda_0")
    close(f(r, v, 0.9, 0.0), np.array([1.9, 0.72, 2.0]), atol=1e-12, label="lambda = 0, one-step targets")
    close(f(r, v, 0.9, 1.0), np.array([2.62, 1.8, 2.0]), atol=1e-12, label="lambda = 1, Monte Carlo returns")
    states, rewards, V = _random_episode(0)
    values = V[states[:-1]]
    close(f(rewards, values, 0.9, 0.7), _forward_lambda_returns(rewards, values, 0.9, 0.7),
          atol=1e-12, label="a random 12-step episode, gamma = 0.9, lambda = 0.7")


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


@check("td_lambda_offline equals the forward-view lambda-return update to 1e-12")
def test_offline_equivalence(sub):
    f = has(sub, "td_lambda_offline")
    for seed, lam in ((0, 0.7), (1, 0.3), (2, 0.95)):
        states, rewards, V = _random_episode(seed)
        values = V[states[:-1]]
        G = _forward_lambda_returns(rewards, values, 0.9, lam)
        want = np.zeros(len(V))
        np.add.at(want, states[:-1], 0.1 * (G - values))
        got = f(states, rewards, V.copy(), 0.1, 0.9, lam)
        shape(got, (len(V),), label="td_lambda_offline output")
        close(got, want, atol=1e-12, label=f"total increment, seed {seed}, lambda = {lam}")


@check("td_lambda_offline sums TD(0) increments at lambda = 0 and Monte Carlo increments at lambda = 1")
def test_offline_limits(sub):
    f = has(sub, "td_lambda_offline")
    states, rewards, V = _random_episode(3)
    T = len(rewards)
    values = V[states[:-1]]
    nxt = np.concatenate([values[1:], [0.0]])
    td0 = np.zeros(len(V))
    np.add.at(td0, states[:-1], 0.1 * (rewards + 0.9 * nxt - values))
    close(f(states, rewards, V.copy(), 0.1, 0.9, 0.0), td0, atol=1e-12, label="lambda = 0")
    G = np.zeros(T)
    run = 0.0
    for t in range(T - 1, -1, -1):
        run = rewards[t] + 0.9 * run
        G[t] = run
    mc = np.zeros(len(V))
    np.add.at(mc, states[:-1], 0.1 * (G - values))
    close(f(states, rewards, V.copy(), 0.1, 0.9, 1.0), mc, atol=1e-12, label="lambda = 1")


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


@check("Online td_lambda matches TD(0) at lambda = 0, and lambda = 0.8 beats it on the 19-state walk")
def test_td_lambda_online(sub):
    f = has(sub, "td_lambda")
    n_step = has(sub, "n_step_td")
    runs = _walk_runs(30, 10, seed=8)
    close(f(runs[0], 21, 0.2, 1.0, 0.0), n_step(runs[0], 21, 1, 0.2, 1.0), atol=1e-12,
          label="td_lambda at lambda = 0 against n_step_td at n = 1")
    e0 = float(np.mean([_rms(f(eps, 21, 0.2, 1.0, 0.0)) for eps in runs]))
    e8 = float(np.mean([_rms(f(eps, 21, 0.2, 1.0, 0.8)) for eps in runs]))
    # Seeded replications give about 0.41 and 0.16.
    if not (e8 < e0 - 0.1):
        fail(f"with alpha = 0.2, expected RMS errors near 0.41 at lambda = 0 and 0.16 at lambda = 0.8; got {e0:.3f} and {e8:.3f}")
