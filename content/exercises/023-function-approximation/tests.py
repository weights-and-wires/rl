"""Checks for the value-function approximation exercise."""

import numpy as np

import rlenvs
from rlcheck import almost, check, close, equal, fail, has, shape

def _stationary(P_pi):
    S = P_pi.shape[0]
    M = np.vstack([P_pi.T - np.eye(S), np.ones((1, S))])
    rhs = np.zeros(S + 1)
    rhs[-1] = 1.0
    mu = np.linalg.lstsq(M, rhs, rcond=None)[0]
    return mu / mu.sum()


def _problem(seed, n_states=6, n_features=3, gamma=0.9):
    """A random MDP, a random policy, its chain, and random linear features."""
    mdp = rlenvs.random_mdp(n_states=n_states, n_actions=2, gamma=gamma, seed=seed)
    rng = np.random.default_rng(seed + 100)
    pi = rng.dirichlet(np.ones(2), size=n_states)
    P_pi = np.einsum("sa,sat->st", pi, mdp.P)
    r_pi = (pi * mdp.R).sum(axis=1)
    d = _stationary(P_pi)
    Phi = rng.standard_normal((n_states, n_features))
    return mdp, pi, P_pi, r_pi, d, Phi


def _trajectory(mdp, pi, steps, seed):
    rng = np.random.default_rng(seed)
    s = 0
    S = np.zeros(steps, dtype=int)
    R = np.zeros(steps)
    S2 = np.zeros(steps, dtype=int)
    for t in range(steps):
        a = rng.choice(pi.shape[1], p=pi[s])
        s2, r = mdp.sample(s, a, rng)
        S[t], R[t], S2[t] = s, r, s2
        s = s2
    return S, R, S2


def _td_fixed_point(Phi, P_pi, r_pi, d, gamma):
    # The point where the expected semi-gradient update is zero. Chapter 24 derives it.
    D = np.diag(d)
    A = Phi.T @ D @ (np.eye(len(d)) - gamma * P_pi) @ Phi
    return np.linalg.solve(A, Phi.T @ D @ r_pi)


@check("tile_features activates one tile per tiling, and matches the hand example")
def test_tile_features(sub):
    tile_features = has(sub, "tile_features")
    f = np.asarray(tile_features(0.3, 0.0, 1.0, 4, 2), dtype=float)
    shape(f, (10,), label="tile_features(0.3, 0, 1, n_tiles=4, n_tilings=2)")
    equal(np.flatnonzero(f), np.array([1, 6]), label="active tiles for x = 0.3")
    equal(np.flatnonzero(tile_features(0.4, 0.0, 1.0, 4, 2)), np.array([1, 7]),
          label="active tiles for x = 0.4")
    rng = np.random.default_rng(0)
    for x in rng.uniform(-0.5, 1.5, size=50):
        g = np.asarray(tile_features(x, 0.0, 1.0, 5, 4), dtype=float)
        if g.shape != (24,) or not np.all((g == 0) | (g == 1)) or g.sum() != 4:
            fail(f"x = {x:.3f} with 5 tiles and 4 tilings should give 24 binary features "
                 f"with exactly 4 ones; got shape {g.shape} and {g.sum():g} ones")
    equal(np.flatnonzero(tile_features(-3.0, 0.0, 1.0, 4, 2)), np.array([0, 5]),
          label="a point below the range is clipped to low")


@check("semi_gradient_td_update moves w along x by alpha times the TD error, masking terminals")
def test_semi_gradient_td_update(sub):
    update = has(sub, "semi_gradient_td_update")
    w = np.array([1.0, 2.0, 0.0])
    x = np.array([1.0, 0.0, 1.0])
    x2 = np.array([0.0, 1.0, 1.0])
    # v(s) = 1, v(s2) = 2, target 0.5 + 0.9 * 2 = 2.3, error 1.3, step 0.13 * x.
    got = update(w.copy(), x, 0.5, x2, False, 0.1, 0.9)
    close(got, np.array([1.13, 2.0, 0.13]), atol=1e-12, label="non-terminal step")
    # Terminal: target is r alone, 0.5, error -0.5, step -0.05 * x.
    got = update(w.copy(), x, 0.5, x2, True, 0.1, 0.9)
    close(got, np.array([0.95, 2.0, -0.05]), atol=1e-12, label="terminal step")


@check("With tile features and alpha = 1/n_tilings, one update reaches the target exactly")
def test_tile_one_trial_learning(sub):
    tile_features = has(sub, "tile_features")
    update = has(sub, "semi_gradient_td_update")
    n_tiles, n_tilings = 4, 2
    x = tile_features(0.3, 0.0, 1.0, n_tiles, n_tilings)
    near = tile_features(0.4, 0.0, 1.0, n_tiles, n_tilings)
    far = tile_features(0.9, 0.0, 1.0, n_tiles, n_tilings)
    rng = np.random.default_rng(1)
    w = rng.standard_normal(x.shape[0])
    before = (w @ x, w @ near, w @ far)
    # A terminal transition with reward 5 makes the target exactly 5.
    w2 = update(w.copy(), x, 5.0, np.zeros_like(x), True, 1.0 / n_tilings, 0.9)
    almost(w2 @ x, 5.0, 1e-10, label="value at the trained point")
    error = 5.0 - before[0]
    # 0.4 shares one of two tiles with 0.3, so it moves half as far.
    almost(w2 @ near - before[1], error / 2, 1e-10, label="change at x = 0.4")
    almost(w2 @ far - before[2], 0.0, 1e-10, label="change at x = 0.9, which shares no tile")


@check("value_error is the d-weighted mean squared gap between Phi @ w and the true values")
def test_value_error(sub):
    value_error = has(sub, "value_error")
    Phi = np.array([[1.0], [2.0]])
    # Estimates [0.5, 1.0] against values [1, 3]: errors -0.5 and -2.
    got = float(value_error(Phi, np.array([0.5]), np.array([1.0, 3.0]), np.array([0.25, 0.75])))
    almost(got, 0.25 * 0.25 + 0.75 * 4.0, 1e-12, label="value_error on the two-state example")
    mdp, pi, P_pi, r_pi, d, Phi = _problem(seed=0)
    v = rlenvs.policy_evaluation(mdp, pi)
    almost(float(value_error(np.eye(6), v, v, d)), 0.0, 1e-12, label="value_error at the true values")


@check("Semi-gradient TD with a shrinking step size approaches the TD fixed point")
def test_semi_gradient_converges(sub):
    update = has(sub, "semi_gradient_td_update")
    mdp, pi, P_pi, r_pi, d, Phi = _problem(seed=3)
    S, R, S2 = _trajectory(mdp, pi, 20000, seed=0)
    w = np.zeros(3)
    for t in range(len(S)):
        alpha = 0.1 / (1 + t / 1000)
        w = update(w, Phi[S[t]], R[t], Phi[S2[t]], False, alpha, mdp.gamma)
    w_td = _td_fixed_point(Phi, P_pi, r_pi, d, mdp.gamma)
    gap = float(np.max(np.abs(Phi @ (w - w_td))))
    start = float(np.max(np.abs(Phi @ w_td)))
    if gap > 0.4 * start:
        fail(f"after 20,000 semi-gradient TD steps the values should be within "
             f"{0.4 * start:.3f} of the TD fixed point; the largest gap is {gap:.3f}")
