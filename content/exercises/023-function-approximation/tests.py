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


# Plausible wrong implementations that the checks below catch:
#   1. Features that ignore the action, so one weight vector is shared by all three actions.
#      The block-placement check and the "only action a changes" check fail, and the learning
#      check fails because the car never learns to push in the useful direction.
#   2. Bootstrapping with the current action a instead of the next action a2.
#      The hand-computed SARSA update fails.
SARSA_ALPHA = 0.05


@check("tile_features_2d and state_action_features give one tile per tiling in the action's own block")
def test_state_action_features(sub):
    tile_features_2d = has(sub, "tile_features_2d")
    saf = has(sub, "state_action_features")
    # Position -0.095 scales to 2.6 tiles and velocity -0.0105 to 1.7 (4 tiles per axis).
    # Tiling 0 has no shift: tile (2, 1), entry 2 * 5 + 1 = 11.
    # Tiling 1 shifts by 0.5 in both axes: tile (3, 2), entry 25 + 3 * 5 + 2 = 42.
    f = np.asarray(tile_features_2d(-0.095, -0.0105, 4, 2), dtype=float)
    shape(f, (50,), label="tile_features_2d(-0.095, -0.0105, n_tiles=4, n_tilings=2)")
    equal(np.flatnonzero(f), np.array([11, 42]), label="active tiles for (-0.095, -0.0105)")
    g = np.asarray(tile_features_2d(-9.0, -9.0, 4, 2), dtype=float)
    equal(np.flatnonzero(g), np.array([0, 25]), label="a point below both ranges is clipped to the low corner")
    per_action = 8 * 81
    base = np.flatnonzero(np.asarray(saf(-0.5, 0.01, 0), dtype=float))
    for a in range(3):
        x = np.asarray(saf(-0.5, 0.01, a), dtype=float)
        shape(x, (3 * per_action,), label=f"state_action_features(-0.5, 0.01, {a})")
        equal(np.flatnonzero(x), base + a * per_action,
              label=f"active entries for action {a}, which must sit in block {a}")


@check("semi_gradient_sarsa_update bootstraps from the next action and changes only the taken action's weights")
def test_sarsa_update(sub):
    saf = has(sub, "state_action_features")
    update = has(sub, "semi_gradient_sarsa_update")
    s, a, s2, a2 = (-0.5, 0.0), 2, (-0.45, 0.01), 0
    x = np.asarray(saf(s[0], s[1], a), dtype=float)
    x2 = np.asarray(saf(s2[0], s2[1], a2), dtype=float)
    # q(s, a) = 8 * 0.25 = 2 and q(s2, a2) = 8 * 0.5 = 4. Action 2 and action 0 use separate blocks.
    w = 0.25 * x + 0.5 * x2
    # Target -1 + 0.9 * 4 = 2.6, error 0.6, each of the 8 active weights moves by 0.05 * 0.6 = 0.03.
    got = update(w.copy(), s, a, -1.0, s2, a2, False, SARSA_ALPHA, 0.9)
    close(got, w + 0.03 * x, atol=1e-12, label="non-terminal step (q(s,a) goes from 2 to 2.24)")
    changed = np.flatnonzero(np.abs(got - w) > 1e-12)
    equal(changed, np.flatnonzero(x), label="entries that changed, which must be those of (s, a) only")
    # Terminal: target -1, error -3, each active weight moves by 0.05 * -3 = -0.15.
    got = update(w.copy(), s, a, -1.0, s2, a2, True, SARSA_ALPHA, 0.9)
    close(got, w - 0.15 * x, atol=1e-12, label="terminal step")


@check("semi_gradient_sarsa learns Mountain Car: the last 10 episodes take far fewer than 1000 steps")
def test_sarsa_learns(sub):
    sarsa = has(sub, "semi_gradient_sarsa")
    cap, limit = 1000, 250
    for seed in range(5):
        w, steps = sarsa(80, seed=seed, max_steps=cap)
        steps = np.asarray(steps)
        shape(steps, (80,), label=f"steps per episode for seed {seed}")
        if steps.max() > cap:
            fail(f"seed {seed}: an episode took {steps.max()} steps, more than the cap of {cap}")
        tail = float(steps[-10:].mean())
        if tail > limit:
            fail(f"seed {seed}: the mean of the last 10 episodes should be under {limit} steps "
                 f"after 80 episodes; got {tail:.1f} (first 3 episodes: {steps[:3].tolist()})")
