"""Checks for the control and average-reward exercise."""

import numpy as np

from rlcheck import almost, check, close, equal, fail, has, shape


def _stationary(P):
    S = P.shape[0]
    M = np.vstack([P.T - np.eye(S), np.ones((1, S))])
    rhs = np.zeros(S + 1)
    rhs[-1] = 1.0
    mu = np.linalg.lstsq(M, rhs, rcond=None)[0]
    return mu / mu.sum()


def _lazy_ring():
    """Four states in a ring. State i stays with probability stay[i], else moves to i + 1."""
    stay = np.array([0.5, 0.2, 0.2, 0.2])
    P = np.zeros((4, 4))
    for i in range(4):
        P[i, i] = stay[i]
        P[i, (i + 1) % 4] = 1.0 - stay[i]
    return P, np.array([0.0, 0.0, 0.0, 4.0])


def _cycle():
    """A deterministic four-state cycle with reward 4 in state 3 and 0 elsewhere."""
    return np.roll(np.eye(4), 1, axis=1), np.array([0.0, 0.0, 0.0, 4.0])


@check("fourier_features gives cos(pi * c . x) for every coefficient vector, in product order")
def test_fourier(sub):
    fourier_features = has(sub, "fourier_features")
    # One input, order 2: cos(0), cos(pi / 4), cos(pi / 2).
    got = np.asarray(fourier_features(np.array([0.25]), 2), dtype=float)
    shape(got, (3,), label="fourier_features([0.25], order=2)")
    close(got, np.array([1.0, np.sqrt(0.5), 0.0]), atol=1e-12, label="fourier_features([0.25], 2)")
    # Two inputs, order 1: c = (0,0), (0,1), (1,0), (1,1) at x = (0.25, 0.5).
    got = np.asarray(fourier_features(np.array([0.25, 0.5]), 1), dtype=float)
    shape(got, (4,), label="fourier_features([0.25, 0.5], order=1)")
    close(got, np.array([1.0, 0.0, np.sqrt(0.5), -np.sqrt(0.5)]), atol=1e-12,
          label="fourier_features([0.25, 0.5], 1), ordered (0,0), (0,1), (1,0), (1,1)")
    got = np.asarray(fourier_features(np.array([0.1, 0.2, 0.3]), 2), dtype=float)
    shape(got, (27,), label="fourier_features with d = 3 and order 2 should have (order + 1) ** d entries")


@check("rbf_features is a Gaussian bump of the distance to each center")
def test_rbf(sub):
    rbf_features = has(sub, "rbf_features")
    centers = np.array([[0.0], [0.5], [1.0]])
    got = np.asarray(rbf_features(np.array([0.5]), centers, 0.5), dtype=float)
    shape(got, (3,), label="rbf_features([0.5], three centers)")
    # Distances 0.5, 0, 0.5 with sigma 0.5: exp(-0.25 / 0.5) = exp(-0.5).
    close(got, np.array([np.exp(-0.5), 1.0, np.exp(-0.5)]), atol=1e-12, label="rbf_features([0.5], ..., sigma=0.5)")
    centers2 = np.array([[0.0, 0.0], [3.0, 4.0]])
    got = np.asarray(rbf_features(np.array([0.0, 0.0]), centers2, 5.0), dtype=float)
    # Squared distance 25 and sigma 5: exp(-25 / 50).
    close(got, np.array([1.0, np.exp(-0.5)]), atol=1e-12, label="rbf_features in two dimensions")


@check("n_step_return sums discounted rewards and bootstraps only when the episode has not ended")
def test_n_step_return(sub):
    n_step_return = has(sub, "n_step_return")
    # 1 + 0.5 * 2 + 0.25 * 3 + 0.125 * 10 = 4.
    almost(float(n_step_return([1.0, 2.0, 3.0], 10.0, 0.5, False)), 1 + 1 + 0.75 + 1.25, 1e-12,
           label="3-step return that bootstraps from q = 10")
    # Terminated: the last window holds 2 rewards and no bootstrap term.
    almost(float(n_step_return([1.0, 2.0], 10.0, 0.5, True)), 2.0, 1e-12,
           label="terminal return, which must ignore q_boot")
    almost(float(n_step_return([-1.0], 4.0, 1.0, False)), 3.0, 1e-12, label="1-step return with gamma = 1")


# Plausible wrong implementations that the checks below catch:
#   1. n_step_return that bootstraps with gamma ** n for a short terminal window or ignores the
#      terminal flag: test_n_step_return fails on the terminal case.
#   2. differential_td_update that moves r_bar toward r alone (r_bar += beta * (r - r_bar)), or
#      that keeps a discount: test_differential_update fails on the hand-computed update, and
#      test_differential_fixed_point fails because the exact values are no longer a fixed point.
LEARN_EPISODES = 30


@check("n_step_sarsa with n = 8 beats n = 1 at the same step size on five seeds")
def test_n_step_learns(sub):
    n_step_sarsa = has(sub, "n_step_sarsa")
    cap = 1000
    for seed in range(5):
        _, one = n_step_sarsa(1, LEARN_EPISODES, seed=seed, max_steps=cap)
        w, eight = n_step_sarsa(8, LEARN_EPISODES, seed=seed, max_steps=cap)
        one, eight = np.asarray(one), np.asarray(eight)
        shape(eight, (LEARN_EPISODES,), label=f"steps per episode for seed {seed}")
        if eight.max() > cap or eight.min() < 1:
            fail(f"seed {seed}: episode lengths must be between 1 and {cap}; got min {eight.min()}, max {eight.max()}")
        tail = float(eight[-5:].mean())
        if tail > 220:
            fail(f"seed {seed}: n = 8 should average under 220 steps in the last 5 of {LEARN_EPISODES} "
                 f"episodes; got {tail:.1f} (first 3 episodes: {eight[:3].tolist()})")
        if eight.mean() > 0.92 * one.mean():
            fail(f"seed {seed}: n = 8 should average under 0.92 times the steps of n = 1 over "
                 f"{LEARN_EPISODES} episodes; got {eight.mean():.1f} against {one.mean():.1f}")


@check("average_reward is the stationary mean of the reward, matching the closed form")
def test_average_reward(sub):
    average_reward = has(sub, "average_reward")
    P, r = _cycle()
    almost(float(average_reward(P, r)), 1.0, 1e-9, label="average reward of the four-cycle (4 / 4)")
    P, r = _lazy_ring()
    # Stationary mass is proportional to the expected stay, 1 / (1 - stay): 2, 1.25, 1.25, 1.25.
    expected = 4.0 * 1.25 / 5.75
    almost(float(average_reward(P, r)), expected, 1e-9, label="average reward of the lazy ring (5 / 5.75)")


@check("differential_td_update measures reward against r_bar and moves w and r_bar by the same error")
def test_differential_update(sub):
    update = has(sub, "differential_td_update")
    w = np.array([1.0, 2.0, 0.0])
    x = np.array([1.0, 0.0, 1.0])
    x2 = np.array([0.0, 1.0, 1.0])
    # v(s) = 1, v(s2) = 2, r = 3, r_bar = 1.5: delta = 3 - 1.5 + 2 - 1 = 2.5.
    w2, rb2 = update(w, 1.5, x, 3.0, x2, 0.1, 0.02)
    close(np.asarray(w2, dtype=float), np.array([1.25, 2.0, 0.25]), atol=1e-12, label="weights after the update")
    almost(float(rb2), 1.5 + 0.02 * 2.5, 1e-12, label="r_bar after the update (it moves by beta * delta)")
    equal(w, np.array([1.0, 2.0, 0.0]), label="the input weights, which must not change in place")


@check("The exact differential values are a fixed point of differential TD on a deterministic cycle")
def test_differential_fixed_point(sub):
    update = has(sub, "differential_td_update")
    P, r = _cycle()
    mu = _stationary(P)
    r_bar = float(mu @ r)
    # Differential values solve h = r - r_bar + P h with mu @ h = 0. Here h - h[0] = [0, 1, 2, 3].
    h = np.linalg.solve(np.eye(4) - P + np.outer(np.ones(4), mu), r - r_bar)
    close(h - h[0], np.array([0.0, 1.0, 2.0, 3.0]), atol=1e-9, label="exact differential values minus h[0]")
    Phi = np.eye(4)
    w, rb = h.copy(), r_bar
    s = 0
    for _ in range(12):
        s2 = (s + 1) % 4
        w, rb = update(w, rb, Phi[s], float(r[s]), Phi[s2], 0.3, 0.05)
        s = s2
    close(np.asarray(w, dtype=float), h, atol=1e-10, label="weights after three laps from the exact values")
    almost(float(rb), r_bar, 1e-10, label="r_bar after three laps from the exact values")


@check("differential_td recovers the average reward and the relative values of the cycle")
def test_differential_cycle(sub):
    differential_td = has(sub, "differential_td")
    P, r = _cycle()
    w, rb = differential_td(P, r, np.eye(4), 4000, 0.1, 0.01, seed=0)
    almost(float(rb), 1.0, 1e-6, label="learned average reward on the deterministic cycle")
    w = np.asarray(w, dtype=float)
    close(w - w[0], np.array([0.0, 1.0, 2.0, 3.0]), atol=1e-5, label="learned values minus the value of state 0")


@check("differential_td on the lazy ring lands near the closed-form average reward on five seeds")
def test_differential_ring(sub):
    differential_td = has(sub, "differential_td")
    P, r = _lazy_ring()
    exact = float(_stationary(P) @ r)
    for seed in range(5):
        _, rb = differential_td(P, r, np.eye(4), 20000, 0.05, 0.002, seed=seed)
        if abs(rb - exact) > 0.15:
            fail(f"seed {seed}: average reward should be within 0.15 of {exact:.4f} after 20,000 steps; got {rb:.4f}")
