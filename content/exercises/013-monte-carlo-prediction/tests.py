"""Checks for the Monte Carlo prediction exercise."""

import numpy as np

import rlenvs
from rlcheck import check, close, fail, has, shape


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


def _episode(mdp, pi, rng, max_steps=1000):
    states, actions, rewards = [], [], []
    s = int(rng.choice(mdp.n_states, p=mdp.mu0))
    for _ in range(max_steps):
        if mdp.terminal[s]:
            break
        a = int(rng.choice(mdp.n_actions, p=pi[s]))
        s2, r = mdp.sample(s, a, rng)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = s2
    return np.array(states), np.array(actions), np.array(rewards, dtype=float)


def _walk_episodes(n_episodes, seed=0):
    mdp = _random_walk()
    rng = np.random.default_rng(seed)
    pi = np.ones((mdp.n_states, 1))
    return mdp, [_episode(mdp, pi, rng) for _ in range(n_episodes)]


@check("mc_prediction gives 3 for A on first visit and 2.5 on every visit in the A, B, A episode")
def test_first_and_every_visit(sub):
    f = has(sub, "mc_prediction")
    # States A = 0 and B = 1; the episode visits A, B, A with rewards 1, 0, 2.
    episode = (np.array([0, 1, 0]), np.array([0, 0, 0]), np.array([1.0, 0.0, 2.0]))
    first = f([episode], n_states=3, gamma=1.0, first_visit=True)
    every = f([episode], n_states=3, gamma=1.0, first_visit=False)
    shape(first, (3,), label="mc_prediction output")
    close(first, np.array([3.0, 2.0, 0.0]), atol=1e-12, label="first-visit values (A, B, unvisited)")
    close(every, np.array([2.5, 2.0, 0.0]), atol=1e-12, label="every-visit values (A, B, unvisited)")


@check("mc_prediction averages across episodes, and a state that no episode visits gets 0")
def test_average_over_episodes(sub):
    f = has(sub, "mc_prediction")
    e1 = (np.array([0, 1]), np.array([0, 0]), np.array([1.0, 1.0]))  # G = (2, 1)
    e2 = (np.array([0]), np.array([0]), np.array([5.0]))             # G = (5)
    e3 = (np.array([1, 0]), np.array([0, 0]), np.array([0.0, 3.0]))  # G = (3, 3)
    V = f([e1, e2, e3], n_states=3, gamma=1.0, first_visit=True)
    # State 0: returns 2, 5, 3 -> 10 / 3. State 1: returns 1, 3 -> 2. State 2: never visited.
    close(V, np.array([10.0 / 3.0, 2.0, 0.0]), atol=1e-12, label="values from three episodes")


@check("mc_prediction discounts: rewards (1, 0, 2) with gamma = 0.9 give first-visit V(A) = 2.62")
def test_discounted(sub):
    f = has(sub, "mc_prediction")
    episode = (np.array([0, 1, 0]), np.array([0, 0, 0]), np.array([1.0, 0.0, 2.0]))
    # G_2 = 2, G_1 = 0 + 0.9 * 2 = 1.8, G_0 = 1 + 0.9 * 1.8 = 2.62.
    first = f([episode], n_states=2, gamma=0.9, first_visit=True)
    every = f([episode], n_states=2, gamma=0.9, first_visit=False)
    close(first, np.array([2.62, 1.8]), atol=1e-12, label="first-visit values with gamma = 0.9")
    close(every, np.array([(2.62 + 2.0) / 2, 1.8]), atol=1e-12, label="every-visit values with gamma = 0.9")


@check("First-visit mc_prediction on the random walk lands within four standard errors of k / 6")
def test_random_walk(sub):
    f = has(sub, "mc_prediction")
    mdp, episodes = _walk_episodes(2000)
    V = np.asarray(f(episodes, n_states=mdp.n_states, gamma=1.0, first_visit=True), dtype=float)
    truth = np.arange(1, 6) / 6
    # Each return is 0 or 1, so a first-visit average over n visits has
    # standard error sqrt(v (1 - v) / n).
    visits = np.zeros(mdp.n_states)
    for states, _, _ in episodes:
        visits[np.unique(states)] += 1
    se = np.sqrt(truth * (1 - truth) / visits[1:6])
    worst = np.max(np.abs(V[1:6] - truth) / se)
    if worst > 4.0:
        fail(
            f"expected values near {np.round(truth, 3)}, got {np.round(V[1:6], 3)}: "
            f"{worst:.1f} standard errors off at worst"
        )
    close(V[[0, 6]], np.zeros(2), atol=1e-12, label="the two terminal states")


@check("Every-visit mc_prediction also lands near k / 6 on the random walk")
def test_random_walk_every_visit(sub):
    f = has(sub, "mc_prediction")
    mdp, episodes = _walk_episodes(2000)
    V = np.asarray(f(episodes, n_states=mdp.n_states, gamma=1.0, first_visit=False), dtype=float)
    truth = np.arange(1, 6) / 6
    err = np.max(np.abs(V[1:6] - truth))
    if err > 0.05:
        fail(f"expected values near {np.round(truth, 3)}, got {np.round(V[1:6], 3)}: {err:.3f} off at worst")


@check("The error of the middle state shrinks as you average more episodes")
def test_error_shrinks(sub):
    f = has(sub, "mc_prediction")
    mdp = _random_walk()
    pi = np.ones((mdp.n_states, 1))
    small, large = [], []
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        episodes = [_episode(mdp, pi, rng) for _ in range(1000)]
        for n, errs in ((10, small), (1000, large)):
            V = f(episodes[:n], n_states=mdp.n_states, gamma=1.0, first_visit=True)
            errs.append(abs(V[3] - 0.5))
    if not (np.mean(large) < 0.5 * np.mean(small)):
        fail(f"mean error of V[3] over 20 runs should fall a lot from 10 to 1000 episodes, "
             f"got {np.mean(small):.3f} then {np.mean(large):.3f}")
