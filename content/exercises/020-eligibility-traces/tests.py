"""Checks for the lambda-return and eligibility traces exercise."""

import numpy as np

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


def _td0(episodes, n_states, alpha, gamma):
    # Plain online TD(0), for comparison with TD(lambda) at lambda = 0.
    V = np.zeros(n_states)
    for states, rewards in episodes:
        T = len(rewards)
        for t in range(T):
            nxt = 0.0 if t == T - 1 else V[states[t + 1]]
            V[states[t]] += alpha * (rewards[t] + gamma * nxt - V[states[t]])
    return V


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


@check("lambda_return gives 2.017 on the worked episode, and the one-step and Monte Carlo targets at the limits")
def test_lambda_return_by_hand(sub):
    f = has(sub, "lambda_return")
    r = np.array([1.0, 0.0, 2.0])
    v = np.array([0.5, 1.0, 0.8])
    got = f(r, v, 0.9, 0.5)
    shape(got, (3,), label="lambda_return output")
    almost(got[0], 2.017, 1e-12, label="G^lambda_0")
    close(f(r, v, 0.9, 0.0), np.array([1.9, 0.72, 2.0]), atol=1e-12, label="lambda = 0, one-step targets")
    close(f(r, v, 0.9, 1.0), np.array([2.62, 1.8, 2.0]), atol=1e-12, label="lambda = 1, Monte Carlo returns")


@check("lambda_return matches the explicit weighted sum of n-step returns on a random episode")
def test_lambda_return_forward(sub):
    f = has(sub, "lambda_return")
    states, rewards, V = _random_episode(0)
    values = V[states[:-1]]
    close(f(rewards, values, 0.9, 0.7), _forward_lambda_returns(rewards, values, 0.9, 0.7),
          atol=1e-12, label="a random 12-step episode, gamma = 0.9, lambda = 0.7")


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


@check("td_lambda_offline leaves V unchanged")
def test_offline_leaves_V(sub):
    f = has(sub, "td_lambda_offline")
    states, rewards, V = _random_episode(4)
    before = V.copy()
    f(states, rewards, V, 0.1, 0.9, 0.5)
    close(V, before, atol=0.0, label="V after td_lambda_offline")


@check("Online td_lambda at lambda = 0 is plain TD(0)")
def test_td_lambda_limit(sub):
    f = has(sub, "td_lambda")
    runs = _walk_runs(2, 5, seed=8)
    for episodes in runs:
        got = f(episodes, 21, 0.2, 1.0, 0.0)
        shape(got, (21,), label="td_lambda output")
        close(got, _td0(episodes, 21, 0.2, 1.0), atol=1e-12, label="td_lambda at lambda = 0 against plain TD(0)")


@check("Online td_lambda at lambda = 0.8 updates V at every step and starts each episode with a fresh trace")
def test_td_lambda_online_reference(sub):
    f = has(sub, "td_lambda")
    # Hand-written online TD(lambda): V changes at every step, and z resets each episode.
    episodes = [_random_episode(seed)[:2] for seed in (5, 6)]
    want = np.zeros(6)
    for states, rewards in episodes:
        z = np.zeros(6)
        T = len(rewards)
        for t in range(T):
            z *= 0.9 * 0.8
            z[states[t]] += 1.0
            nxt = 0.0 if t == T - 1 else want[states[t + 1]]
            want += 0.1 * (rewards[t] + 0.9 * nxt - want[states[t]]) * z
    close(f(episodes, 6, 0.1, 0.9, 0.8), want, atol=1e-12, label="two random episodes, gamma = 0.9, lambda = 0.8")


@check("Online TD(lambda) at lambda = 0.8 beats lambda = 0 on the 19-state walk (30 runs of 10 episodes)")
def test_td_lambda_online(sub):
    f = has(sub, "td_lambda")
    runs = _walk_runs(30, 10, seed=8)
    e0 = float(np.mean([_rms(f(eps, 21, 0.2, 1.0, 0.0)) for eps in runs]))
    e8 = float(np.mean([_rms(f(eps, 21, 0.2, 1.0, 0.8)) for eps in runs]))
    # Seeded replications give about 0.41 and 0.16.
    if not (e8 < e0 - 0.1):
        fail(f"with alpha = 0.2, expected RMS errors near 0.41 at lambda = 0 and 0.16 at lambda = 0.8; got {e0:.3f} and {e8:.3f}")
