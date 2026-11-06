"""Checks for the temporal-difference learning exercise."""

import numpy as np

import rlenvs
from rlcheck import almost, check, close, fail, has, shape


def _random_walk(n=5):
    S = n + 2
    P = np.zeros((S, 1, S))
    R3 = np.zeros((S, 1, S))
    for s in range(1, n + 1):
        P[s, 0, s - 1] = 0.5
        P[s, 0, s + 1] = 0.5
    P[0, 0, 0] = 1.0
    P[S - 1, 0, S - 1] = 1.0
    R3[n, 0, n + 1] = 1.0
    terminal = np.zeros(S, dtype=bool)
    terminal[[0, S - 1]] = True
    mu0 = np.zeros(S)
    mu0[(n + 1) // 2] = 1.0
    R = (P * R3).sum(axis=2)
    return rlenvs.TabularMDP(P=P, R=R, gamma=1.0, mu0=mu0, terminal=terminal, R3=R3)


def _chain():
    # States 0 -> 1 -> 2, with state 2 terminal. The reward is 0, then 1.
    P = np.zeros((3, 1, 3))
    P[0, 0, 1] = 1.0
    P[1, 0, 2] = 1.0
    P[2, 0, 2] = 1.0
    R3 = np.zeros((3, 1, 3))
    R3[1, 0, 2] = 1.0
    R = (P * R3).sum(axis=2)
    return rlenvs.TabularMDP(
        P=P, R=R, gamma=1.0, mu0=np.array([1.0, 0.0, 0.0]),
        terminal=np.array([False, False, True]), R3=R3,
    )


def _mc_constant_alpha(mdp, n_episodes, alpha, seed):
    # Constant-step every-visit Monte Carlo, for comparison. On the walk the
    # only reward comes at the end and gamma = 1, so every visit sees the
    # episode's final reward as its return.
    rng = np.random.default_rng(seed)
    V = np.where(mdp.terminal, 0.0, 0.5)
    for _ in range(n_episodes):
        s = 3
        visited, G = [], 0.0
        while not mdp.terminal[s]:
            s2, r = mdp.sample(s, 0, rng)
            visited.append(s)
            G += r
            s = s2
        for x in visited:
            V[x] += alpha * (G - V[x])
    return V


# Sutton and Barto's Example 6.4: A = 0, B = 1, terminal = 2.
_AB_DATA = (
    [(np.array([0, 1, 2]), np.array([0.0, 0.0]))]
    + [(np.array([1, 2]), np.array([1.0])) for _ in range(6)]
    + [(np.array([1, 2]), np.array([0.0]))]
)


@check("td0_update moves V[s] a fraction alpha toward r + gamma * V[s2], and ignores V[s2] at the end")
def test_td0_update(sub):
    f = has(sub, "td0_update")
    V = np.array([0.0, 0.5, 1.0])
    # Target 1 + 0.9 * 1.0 = 1.9, error 1.4, step 0.14.
    got = f(V.copy(), s=1, r=1.0, s2=2, terminated=False, alpha=0.1, gamma=0.9)
    close(got, np.array([0.0, 0.64, 1.0]), atol=1e-12, label="a non-terminal step")
    # On termination the target is r alone: 0.5 + 0.1 * (1 - 0.5) = 0.55.
    got = f(V.copy(), s=1, r=1.0, s2=2, terminated=True, alpha=0.1, gamma=0.9)
    close(got, np.array([0.0, 0.55, 1.0]), atol=1e-12, label="a terminal step")


@check("td0_prediction on a three-state chain passes news back one state per episode")
def test_chain(sub):
    f = has(sub, "td0_prediction")
    mdp = _chain()
    pi = np.ones((3, 1))
    # With alpha = 1 and every value starting at 0.5, the first episode sets
    # V[0] = 0 + V[1] = 0.5, then V[1] = 1. Only the second episode lets V[0]
    # see the new V[1].
    V1 = np.asarray(f(mdp, pi, n_episodes=1, alpha=1.0, seed=0), dtype=float)
    shape(V1, (3,), label="td0_prediction output")
    close(V1, np.array([0.5, 1.0, 0.0]), atol=1e-12, label="values after one episode")
    V2 = np.asarray(f(mdp, pi, n_episodes=2, alpha=1.0, seed=0), dtype=float)
    close(V2, np.array([1.0, 1.0, 0.0]), atol=1e-12, label="values after two episodes")


@check("batch_td0 converges to V(A) = V(B) = 0.75 on the eight-episode dataset")
def test_batch_td0_limit(sub):
    f = has(sub, "batch_td0")
    V = f(_AB_DATA, n_states=3, gamma=1.0, alpha=0.05, n_sweeps=1000)
    shape(V, (3,), label="batch_td0 output")
    close(V, np.array([0.75, 0.75, 0.0]), atol=1e-6, label="V for (A, B, terminal)")


@check("batch_td0 holds V fixed within a sweep, then applies the summed increments")
def test_batch_td0_fixed(sub):
    f = has(sub, "batch_td0")
    # Sweep 1 from V = 0: only the six reward-1 episodes move V[B], by 6 * 0.5 = 3.
    V1 = f(_AB_DATA, n_states=3, gamma=1.0, alpha=0.5, n_sweeps=1)
    close(V1, np.array([0.0, 3.0, 0.0]), atol=1e-12, label="values after one sweep")
    # Sweep 2 uses V[B] = 3 for every step: A moves 0.5 * 3 = 1.5, and B is
    # left eight times: 6 * 0.5 * (1 - 3) + 2 * 0.5 * (0 - 3) = -9, from 3 to -6.
    V2 = f(_AB_DATA, n_states=3, gamma=1.0, alpha=0.5, n_sweeps=2)
    close(V2, np.array([1.5, -6.0, 0.0]), atol=1e-12, label="values after two sweeps")


@check("td0_prediction on the random walk keeps terminal values at zero and lands near k / 6")
def test_walk_values(sub):
    f = has(sub, "td0_prediction")
    mdp = _random_walk()
    pi = np.ones((mdp.n_states, 1))
    V = np.asarray(f(mdp, pi, n_episodes=300, alpha=0.05, seed=0), dtype=float)
    shape(V, (mdp.n_states,), label="td0_prediction output")
    close(V[[0, 6]], np.zeros(2), atol=1e-12, label="terminal values")
    truth = np.arange(1, 6) / 6
    err = float(np.sqrt(np.mean((V[1:6] - truth) ** 2)))
    if not (err < 0.1):
        fail(f"after 300 episodes at alpha = 0.05 the RMS error should be below 0.1, got {err:.3f}")


@check("td0_prediction on the random walk has lower RMS error than constant-step MC after 100 episodes")
def test_td_beats_mc(sub):
    f = has(sub, "td0_prediction")
    mdp = _random_walk()
    pi = np.ones((mdp.n_states, 1))
    truth = np.arange(1, 6) / 6
    td, mc = [], []
    for run in range(40):
        V = np.asarray(f(mdp, pi, n_episodes=100, alpha=0.05, seed=run), dtype=float)
        td.append(np.sqrt(np.mean((V[1:6] - truth) ** 2)))
        W = _mc_constant_alpha(mdp, 100, 0.02, seed=run)
        mc.append(np.sqrt(np.mean((W[1:6] - truth) ** 2)))
    td, mc = float(np.mean(td)), float(np.mean(mc))
    if not (td < 0.055):
        fail(f"TD(0) with alpha = 0.05 should reach an RMS error near 0.036, below 0.055; got {td:.4f}")
    if not (td < mc):
        fail(f"TD(0)'s RMS error {td:.4f} should be below constant-step MC's {mc:.4f}")
