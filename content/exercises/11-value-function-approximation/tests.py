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


@check("td_fixed_point recovers V^pi exactly with one-hot features, and solves the projected Bellman equation")
def test_td_fixed_point_exact(sub):
    td_fixed_point = has(sub, "td_fixed_point")
    mdp, pi, P_pi, r_pi, d, Phi = _problem(seed=0)
    v = rlenvs.policy_evaluation(mdp, pi)
    w = td_fixed_point(np.eye(6), P_pi, r_pi, d, mdp.gamma)
    close(w, v, atol=1e-9, label="one-hot features: w should equal V^pi")
    w = np.asarray(td_fixed_point(Phi, P_pi, r_pi, d, mdp.gamma), dtype=float)
    shape(w, (3,), label="td_fixed_point weights")
    # At the TD fixed point, the Bellman residual is D-orthogonal to every feature.
    residual = r_pi + mdp.gamma * P_pi @ Phi @ w - Phi @ w
    close(Phi.T @ (d * residual), np.zeros(3), atol=1e-10,
          label="Phi^T D (Bellman residual) at the fixed point")


@check("The TD fixed point's error is within 1/(1 - gamma) of the best linear fit on 40 random problems")
def test_td_bound(sub):
    td_fixed_point = has(sub, "td_fixed_point")
    for seed in range(40):
        mdp, pi, P_pi, r_pi, d, Phi = _problem(seed=seed)
        v = rlenvs.policy_evaluation(mdp, pi)
        w = td_fixed_point(Phi, P_pi, r_pi, d, mdp.gamma)
        ve_td = float(d @ (Phi @ w - v) ** 2)
        D = np.diag(d)
        w_best = np.linalg.solve(Phi.T @ D @ Phi, Phi.T @ D @ v)
        ve_best = float(d @ (Phi @ w_best - v) ** 2)
        if ve_td < ve_best - 1e-10:
            fail(f"seed {seed}: VE at your fixed point, {ve_td:.5f}, is below the best "
                 f"possible {ve_best:.5f}, so it is not a least-squares-consistent answer")
        if ve_td > ve_best / (1 - mdp.gamma) + 1e-10:
            fail(f"seed {seed}: VE at your fixed point, {ve_td:.5f}, breaks the bound "
                 f"{ve_best / (1 - mdp.gamma):.5f} = best VE / (1 - gamma)")


@check("lstd with one-hot features equals the value of the empirical model, with terminal masking")
def test_lstd_tabular(sub):
    lstd = has(sub, "lstd")
    # Five transitions over three states. State 2's transition terminates.
    S = np.array([0, 0, 1, 1, 2])
    R = np.array([1.0, 0.0, 2.0, -1.0, 3.0])
    S2 = np.array([1, 0, 2, 0, 0])
    term = np.array([False, False, False, False, True])
    gamma = 0.5
    X, X2 = np.eye(3)[S], np.eye(3)[S2]
    w = lstd(X, R, X2, term, gamma)
    # Empirical model: P_hat from counts (terminal transitions lead nowhere), r_hat means.
    P_hat = np.zeros((3, 3))
    r_hat = np.zeros(3)
    for s, r, s2, t in zip(S, R, S2, term):
        n = np.sum(S == s)
        r_hat[s] += r / n
        if not t:
            P_hat[s, s2] += 1.0 / n
    want = np.linalg.solve(np.eye(3) - gamma * P_hat, r_hat)
    close(w, want, atol=1e-6, label="LSTD weights against (I - gamma P_hat)^-1 r_hat")


@check("lstd on a 20,000-step trajectory lands near the exact TD fixed point")
def test_lstd_converges(sub):
    lstd = has(sub, "lstd")
    td_fixed_point = has(sub, "td_fixed_point")
    mdp, pi, P_pi, r_pi, d, Phi = _problem(seed=3)
    S, R, S2 = _trajectory(mdp, pi, 20000, seed=0)
    w = lstd(Phi[S], R, Phi[S2], np.zeros(len(S), dtype=bool), mdp.gamma)
    w_td = td_fixed_point(Phi, P_pi, r_pi, d, mdp.gamma)
    gap = float(np.max(np.abs(Phi @ (np.asarray(w) - w_td))))
    if gap > 0.2:
        fail(f"LSTD's values should be within 0.2 of the TD fixed point's in every state "
             f"after 20,000 steps; the largest gap is {gap:.3f}")


@check("Semi-gradient TD with a shrinking step size approaches the TD fixed point")
def test_semi_gradient_converges(sub):
    update = has(sub, "semi_gradient_td_update")
    td_fixed_point = has(sub, "td_fixed_point")
    mdp, pi, P_pi, r_pi, d, Phi = _problem(seed=3)
    S, R, S2 = _trajectory(mdp, pi, 20000, seed=0)
    w = np.zeros(3)
    for t in range(len(S)):
        alpha = 0.1 / (1 + t / 1000)
        w = update(w, Phi[S[t]], R[t], Phi[S2[t]], False, alpha, mdp.gamma)
    w_td = td_fixed_point(Phi, P_pi, r_pi, d, mdp.gamma)
    gap = float(np.max(np.abs(Phi @ (w - w_td))))
    start = float(np.max(np.abs(Phi @ w_td)))
    if gap > 0.4 * start:
        fail(f"after 20,000 semi-gradient TD steps the values should be within "
             f"{0.4 * start:.3f} of the TD fixed point; the largest gap is {gap:.3f}")


@check("baird_features builds the 7-by-8 feature matrix of Baird's counterexample")
def test_baird_features(sub):
    baird_features = has(sub, "baird_features")
    Phi = np.asarray(baird_features(), dtype=float)
    shape(Phi, (7, 8), label="baird_features()")
    want = np.zeros((7, 8))
    want[np.arange(6), np.arange(6)] = 2.0
    want[:6, 7] = 1.0
    want[6, 6], want[6, 7] = 1.0, 2.0
    close(Phi, want, atol=0, label="baird_features()")


@check("baird_expected_update diverges with uniform weights and follows the exact decay on-policy")
def test_baird_divergence(sub):
    update = has(sub, "baird_expected_update")
    Phi = np.zeros((7, 8))
    Phi[np.arange(6), np.arange(6)] = 2.0
    Phi[:6, 7] = 1.0
    Phi[6, 6], Phi[6, 7] = 1.0, 2.0
    alpha, gamma = 0.01, 0.99
    uniform = np.full(7, 1 / 7)
    # One update against the linear map w + alpha Phi^T D (gamma 1 x_7^T - Phi) w.
    rng = np.random.default_rng(2)
    w = rng.standard_normal(8)
    target = gamma * np.tile(Phi[6], (7, 1))
    want = w + alpha * Phi.T @ np.diag(uniform) @ (target - Phi) @ w
    close(update(w.copy(), alpha, gamma, uniform), want, atol=1e-12, label="one uniform update")

    w0 = np.array([1, 1, 1, 1, 1, 1, 10, 1], dtype=float)
    w = w0.copy()
    for _ in range(1000):
        w = update(w, alpha, gamma, uniform)
    if not np.linalg.norm(w) > 10 * np.linalg.norm(w0):
        fail(f"with uniform weights |w| should grow past ten times its start, "
             f"{10 * np.linalg.norm(w0):.1f}, in 1000 sweeps; got {np.linalg.norm(w):.1f}")

    on_policy = np.eye(7)[6]
    w = w0.copy()
    for _ in range(1000):
        w = update(w, alpha, gamma, on_policy)
    # Only state 6 is updated. Its value shrinks by 1 - alpha (1 - gamma) |x_7|^2 each sweep.
    want_v6 = 12.0 * (1 - alpha * (1 - gamma) * 5.0) ** 1000
    almost(Phi[6] @ w, want_v6, 1e-8, label="on-policy value of state 6 after 1000 sweeps")
