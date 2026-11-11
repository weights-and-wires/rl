"""Checks for the sim-to-real exercise: randomize, identify, and adapt."""

import numpy as np

from rlcheck import almost, check, close, fail, has, shape

RANGES = {"mass": (0.1, 2.0), "friction": (0.1, 1.5)}


@check("randomized_params returns an (n, 2) array of [mass, friction] inside the ranges")
def test_randomized_bounds(sub):
    randomized_params = has(sub, "randomized_params")
    out = np.asarray(randomized_params(np.random.default_rng(0), RANGES, 500))
    shape(out, (500, 2), label="randomized_params(rng, ranges, 500)")
    if out[:, 0].min() < 0.1 or out[:, 0].max() > 2.0:
        fail(f"mass must stay in (0.1, 2.0), got [{out[:, 0].min():.3f}, {out[:, 0].max():.3f}]")
    if out[:, 1].min() < 0.1 or out[:, 1].max() > 1.5:
        fail(f"friction must stay in (0.1, 1.5), got [{out[:, 1].min():.3f}, {out[:, 1].max():.3f}]")


# Wrong implementation caught here: building a fresh `default_rng(0)` for every
# draw (the same seed per episode), so every row is the same physics and
# domain randomization silently becomes nominal training on one wrong robot.
@check("randomized_params draws a different physics for every row and every call, log-uniformly")
def test_randomized_varies(sub):
    randomized_params = has(sub, "randomized_params")
    rng = np.random.default_rng(1)
    a = np.asarray(randomized_params(rng, RANGES, 2000))
    b = np.asarray(randomized_params(rng, RANGES, 2000))
    if len(np.unique(a[:, 0])) < 1990:
        fail(f"2000 draws should give about 2000 distinct masses, got {len(np.unique(a[:, 0]))} (one seed reused per draw?)")
    if np.allclose(a, b):
        fail("two calls on the same generator returned the same draws; draw from the rng you are given")
    # Log-uniform: the mean of log(mass) is the middle of the log range.
    want = 0.5 * (np.log(0.1) + np.log(2.0))
    almost(np.log(a[:, 0]).mean(), want, 0.06, label="mean of log(mass)")
    want_f = 0.5 * (np.log(0.1) + np.log(1.5))
    almost(np.log(a[:, 1]).mean(), want_f, 0.06, label="mean of log(friction)")


# Wrong implementation caught here: regressing the next position (or the
# position change) on [force, velocity] instead of the acceleration
# (v' - v) / dt. The coefficients then absorb dt and come out wrong.
@check("sysid_fit recovers mass and friction exactly from noiseless transitions")
def test_sysid_exact(sub):
    sysid_fit = has(sub, "sysid_fit")
    collect = has(sub, "collect_transitions")
    rng = np.random.default_rng(2)
    for true in ([1.0, 0.5], [0.3, 1.2], [1.8, 0.1]):
        S, U, S2 = collect(true, 50, rng)
        mass, friction = sysid_fit(S, U, S2)
        almost(mass, true[0], 1e-8, label=f"mass for true {true}")
        almost(friction, true[1], 1e-8, label=f"friction for true {true}")


@check("sysid_fit stays within 2 percent with measurement noise on the velocity")
def test_sysid_noisy(sub):
    sysid_fit = has(sub, "sysid_fit")
    collect = has(sub, "collect_transitions")
    for seed in range(5):
        rng = np.random.default_rng(10 + seed)
        for true in ([0.15, 0.9], [1.0, 0.5], [2.0, 1.4]):
            mass, friction = sysid_fit(*collect(true, 300, rng, noise=0.002))
            if abs(mass - true[0]) > 0.02 * true[0]:
                fail(f"seed {seed}, true {true}: expected mass within 2% of {true[0]}, got {mass:.4f}")
            if abs(friction - true[1]) > 0.04 * true[1] + 0.02:
                fail(f"seed {seed}, true {true}: expected friction near {true[1]}, got {friction:.4f}")


@check("history_estimate uses only the last window transitions, and falls back to the prior when it has too few")
def test_history_estimate(sub):
    history_estimate = has(sub, "history_estimate")
    collect = has(sub, "collect_transitions")
    rng = np.random.default_rng(3)
    old = collect([1.5, 0.2], 40, rng)   # the robot before its payload changed
    new = collect([0.4, 0.9], 40, rng)   # the robot after
    S, U, S2 = (np.concatenate([o, n]) for o, n in zip(old, new))
    prior = np.array([1.0, 0.5])
    got = np.asarray(history_estimate(S, U, S2, 20, prior))
    close(got, [0.4, 0.9], atol=1e-8, label="estimate from the last 20 transitions")
    mixed = np.asarray(history_estimate(S, U, S2, 80, prior))
    if abs(mixed[0] - 0.4) < 0.05:
        fail(f"a window of all 80 transitions spans both robots, so it should not recover the new mass 0.4, got {mixed[0]:.3f}")
    close(history_estimate(S, U, S2, 2, prior), prior, atol=0, label="window of 2 returns the prior")
    close(history_estimate(S[:2], U[:2], S2[:2], 20, prior), prior, atol=0, label="two transitions so far returns the prior")


@check("adaptive_action makes any robot follow x'' = -kx x - kv v when the estimate is exact")
def test_adaptive_action(sub):
    adaptive_action = has(sub, "adaptive_action")
    step = has(sub, "step")
    rng = np.random.default_rng(4)
    params = np.stack([rng.uniform(0.2, 2.0, 25), rng.uniform(0.1, 1.5, 25)], axis=1)
    state = rng.uniform(-0.5, 0.5, size=(25, 2))
    gains = np.array([16.0, 8.0])
    u = np.asarray(adaptive_action(state, params, gains))
    shape(u, (25,), label="adaptive_action on a batch of 25")
    accel = (step(state, u, params)[:, 1] - state[:, 1]) / 0.05
    want = -16.0 * state[:, 0] - 8.0 * state[:, 1]
    close(accel, want, atol=1e-9, label="acceleration of the closed loop")


@check("tune_gain on one physics lands within 5 percent of the best gain on a grid")
def test_tune_gain(sub):
    tune_gain = has(sub, "tune_gain")
    batch_cost = has(sub, "batch_cost")
    nominal = np.array([[1.0, 0.5]])
    grid = np.array([[kx, kv] for kx in np.linspace(2, 30, 15) for kv in np.linspace(1, 12, 12)])
    best = batch_cost(grid, nominal).min()
    g = np.asarray(tune_gain(nominal, 0))
    got = float(batch_cost(g[None], nominal)[0, 0])
    if got > 1.05 * best:
        fail(f"expected a mean cost within 5% of the grid best {best:.4f}, got {got:.4f} at gain {g.round(2)}")


@check("a gain tuned on randomized physics has a much smaller worst-case cost and a slightly worse nominal cost")
def test_robust_tradeoff(sub):
    tune_gain = has(sub, "tune_gain")
    randomized_params = has(sub, "randomized_params")
    batch_cost = has(sub, "batch_cost")
    nominal = np.array([[1.0, 0.5]])
    real = np.array([[m, b] for m in np.linspace(0.15, 2.0, 6) for b in np.linspace(0.1, 1.5, 5)])
    for seed in range(5):
        g_nom = np.asarray(tune_gain(nominal, seed))
        g_rob = np.asarray(tune_gain(randomized_params(np.random.default_rng(100 + seed), RANGES, 128), seed))
        real_nom, real_rob = batch_cost(np.stack([g_nom, g_rob]), real).max(axis=1)
        sim_nom, sim_rob = batch_cost(np.stack([g_nom, g_rob]), nominal)[:, 0]
        if not real_rob < 0.25 * real_nom:
            fail(f"seed {seed}: expected the randomized worst case below 25% of the nominal one ({real_nom:.3f}), got {real_rob:.3f}")
        if not sim_rob > 1.02 * sim_nom:
            fail(f"seed {seed}: expected the randomized gain to pay at least 2% on the nominal robot ({sim_nom:.4f}), got {sim_rob:.4f}")
        if sim_rob > 1.5 * sim_nom:
            fail(f"seed {seed}: expected the nominal cost to rise by less than 50% ({sim_nom:.4f}), got {sim_rob:.4f}")


@check("an adaptive controller using identified parameters beats the robust gain on every real robot's worst case and on average")
def test_adaptive_beats_robust(sub):
    tune_gain = has(sub, "tune_gain")
    randomized_params = has(sub, "randomized_params")
    sysid_fit = has(sub, "sysid_fit")
    collect = has(sub, "collect_transitions")
    batch_cost = has(sub, "batch_cost")
    adaptive_cost = has(sub, "adaptive_cost")
    real = np.array([[m, b] for m in np.linspace(0.15, 2.0, 6) for b in np.linspace(0.1, 1.5, 5)])
    for seed in range(5):
        rng = np.random.default_rng(200 + seed)
        g_rob = np.asarray(tune_gain(randomized_params(rng, RANGES, 128), seed))
        robust = batch_cost(g_rob[None], real)[0]
        est = np.array([sysid_fit(*collect(p, 200, rng, noise=0.002)) for p in real])
        adaptive = adaptive_cost(real, est)
        if not adaptive.max() < 0.95 * robust.max():
            fail(f"seed {seed}: expected the adaptive worst case below 95% of the robust {robust.max():.4f}, got {adaptive.max():.4f}")
        if not adaptive.mean() < 0.99 * robust.mean():
            fail(f"seed {seed}: expected the adaptive mean below 99% of the robust {robust.mean():.4f}, got {adaptive.mean():.4f}")
