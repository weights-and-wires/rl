"""Checks for the options and hierarchical RL exercise."""

import numpy as np

from rlcheck import almost, check, close, equal, fail, has

GAMMA = 0.9
START_CELL = (1, 1)


def _setup(sub):
    env = has(sub, "four_rooms")()
    prims = has(sub, "primitive_options")(env)
    halls = has(sub, "hallway_options")(env)
    return env, prims, halls


def _find(options, name):
    return next(i for i, o in enumerate(options) if o.name == name)


def _reference_v(env):
    """Primitive value iteration, written independently of the reader's code."""
    V = np.full(len(env.cells), -10.0)
    V[env.goal] = 0.0
    for _ in range(3000):
        V_new = (env.R + env.gamma * env.P @ V).max(axis=1)
        V_new[env.goal] = 0.0
        if np.abs(V_new - V).max() < 1e-13:
            break
        V = V_new
    return V_new


@check("discounted_return weights reward t by gamma^t")
def test_discounted_return(sub):
    discounted_return = has(sub, "discounted_return")
    # -1 - 0.9 - 0.81 = -2.71. The undiscounted sum is -3.
    almost(discounted_return([-1.0, -1.0, -1.0], 0.9), -2.71, 1e-9, label="three steps at -1")
    # WRONG IMPLEMENTATION: summing the rewards inside an option without discounting.
    almost(discounted_return([-1.0] * 5, 0.9), -4.0951, 1e-9, label="five steps at -1")
    almost(discounted_return([0.0, 2.0, 0.0, 1.0], 0.5), 1.125, 1e-9, label="rewards 0, 2, 0, 1 at gamma 0.5")


@check("option_model of a one-step option is the MDP's own R[:, a] and gamma * P[:, a, :]")
def test_option_model_primitive(sub):
    env, prims, _ = _setup(sub)
    option_model = has(sub, "option_model")
    for a in range(4):
        R_o, P_o = option_model(env, prims[a])
        close(R_o, env.R[:, a], atol=1e-10, label=f"R_o for action {a}")
        close(P_o, GAMMA * env.P[:, a, :], atol=1e-10, label=f"P_o for action {a}")


@check("option_model matches the closed form one step from the hallway")
def test_option_model_closed_form(sub):
    env, _, halls = _setup(sub)
    option_model = has(sub, "option_model")
    o = _find(halls, "room0->(3, 6)")
    s = env.index[(3, 5)]
    hall = env.index[(3, 6)]
    R_o, P_o = option_model(env, halls[o])
    # From (3, 5) each step reaches the hallway with probability 0.8 and otherwise stays.
    # R_o = -1 + 0.2 * 0.9 * R_o, so R_o = -1 / 0.82. P_o[hall] = 0.8 * 0.9 + 0.2 * 0.9 * P_o[hall].
    almost(R_o[s], -1.0 / 0.82, 1e-9, label="R_o at (3, 5)")
    almost(P_o[s, hall], 0.72 / 0.82, 1e-9, label="P_o at (3, 5) to the hallway")
    almost(P_o[s].sum(), 0.72 / 0.82, 1e-9, label="every other entry of the row is 0")
    if np.any(P_o < -1e-12) or np.any(P_o.sum(axis=1) > GAMMA + 1e-9):
        fail("P_o must be nonnegative with every row summing to at most gamma = 0.9")


def _monte_carlo(env, option, s0, n, seed):
    """Run n copies of the option from s0 and average the discounted return and gamma^k."""
    rng = np.random.default_rng(seed)
    cum = np.cumsum(env.P, axis=2)
    s = np.full(n, s0)
    alive = np.ones(n, bool)
    ret = np.zeros(n)
    disc = np.ones(n)
    end = np.zeros(n, int)
    while alive.any():
        a = option.pi[s]
        ret += np.where(alive, disc * env.R[s, a], 0.0)
        u = rng.random(n)
        s2 = (u[:, None] > cum[s, a]).sum(axis=1)
        disc = np.where(alive, disc * GAMMA, disc)
        s = np.where(alive, s2, s)
        stop = alive & ((rng.random(n) < option.beta[s]) | (s == env.goal))
        end[stop] = s[stop]
        alive &= ~stop
    P_hat = np.zeros(len(env.cells))
    np.add.at(P_hat, end, disc)
    return ret.mean(), P_hat / n


@check("option_model agrees with Monte Carlo rollouts of two hallway options")
def test_option_model_monte_carlo(sub):
    env, _, halls = _setup(sub)
    option_model = has(sub, "option_model")
    for name, cell in (("room0->(3, 6)", START_CELL), ("room3->(7, 9)", (10, 10))):
        o = halls[_find(halls, name)]
        R_o, P_o = option_model(env, o)
        s0 = env.index[cell]
        r_hat, p_hat = _monte_carlo(env, o, s0, 8000, seed=0)
        if abs(R_o[s0] - r_hat) > 0.08:
            fail(f"{name} from {cell}: R_o is {R_o[s0]:.4f} but Monte Carlo gives {r_hat:.4f}")
        if np.abs(P_o[s0] - p_hat).max() > 0.012:
            fail(f"{name} from {cell}: P_o differs from Monte Carlo by {np.abs(P_o[s0] - p_hat).max():.4f} "
                 f"(expected sum {p_hat.sum():.4f}, got {P_o[s0].sum():.4f})")


@check("smdp_value_iteration over primitives alone finds the optimal values in 41 sweeps")
def test_smdp_vi_primitives(sub):
    env, prims, _ = _setup(sub)
    smdp_value_iteration = has(sub, "smdp_value_iteration")
    V_ref = _reference_v(env)
    V, _ = smdp_value_iteration(env, prims, tol=1e-12)
    close(V, V_ref, atol=1e-8, label="V* from smdp_value_iteration with primitives")
    almost(V_ref[env.start], -9.2581, 5e-5, label="V*(start)")
    _, sweeps = smdp_value_iteration(env, prims)
    equal(sweeps, 41, label="sweeps with tol 1e-6")


@check("adding hallway options keeps V* exactly and cuts the sweeps from 41 to 25")
def test_smdp_vi_with_options(sub):
    env, prims, halls = _setup(sub)
    smdp_value_iteration = has(sub, "smdp_value_iteration")
    V_ref = _reference_v(env)
    V, _ = smdp_value_iteration(env, prims + halls, tol=1e-12)
    close(V, V_ref, atol=1e-8, label="V* with primitives and hallway options")
    _, sweeps = smdp_value_iteration(env, prims + halls)
    equal(sweeps, 25, label="sweeps with tol 1e-6")


@check("smdp_q_update bootstraps with gamma^k, not gamma")
def test_smdp_q_update(sub):
    smdp_q_update = has(sub, "smdp_q_update")
    Q = np.zeros((3, 2))
    Q[1] = [-6.0, 5.0]
    avail = np.array([[True, True], [True, False], [True, True]])
    # The option ran 5 steps at -1 each: ret = -4.0951. Target = -4.0951 + 0.9^5 * (-6) = -7.63804.
    # Option 1 is not available in state 1, so its Q of 5 must not enter the max.
    # WRONG IMPLEMENTATION: bootstrapping with gamma instead of gamma^k gives target -9.4951.
    new = smdp_q_update(Q.copy(), s=0, o=0, ret=-4.0951, k=5, s2=1, terminated=False,
                        avail=avail, alpha=0.5, gamma=GAMMA)
    almost(new[0, 0], 0.5 * (-4.0951 + 0.59049 * -6.0), 1e-9, label="Q[0, 0] after the update")
    close(new[1:], Q[1:], atol=0, label="every other row is unchanged")
    done = smdp_q_update(Q.copy(), s=0, o=0, ret=-4.0951, k=5, s2=1, terminated=True,
                         avail=avail, alpha=0.5, gamma=GAMMA)
    almost(done[0, 0], 0.5 * -4.0951, 1e-9, label="Q[0, 0] when the option ended the episode")


@check("intra_option_q_update updates every option that would take the action, and only those")
def test_intra_option_update(sub):
    env, prims, halls = _setup(sub)
    intra_option_q_update = has(sub, "intra_option_q_update")
    options = prims + halls
    s, s2 = env.index[(3, 3)], env.index[(3, 4)]
    Q = np.zeros((len(env.cells), len(options)))
    Q[s2] = -(np.arange(len(options)) + 1.0)
    h = _find(options, "room0->(3, 6)")
    # Action 1 (right) in (3, 3): the primitive "right" and the hallway option toward (3, 6).
    # The primitive terminates, so its target is -1 + 0.9 * max over Q[s2] = -1 + 0.9 * (-1) = -1.9.
    # The hallway option continues at (3, 4), so its target is -1 + 0.9 * Q[s2, h] = -1 + 0.9 * (-5) = -5.5.
    new = intra_option_q_update(Q.copy(), s, 1, -1.0, s2, False, options, 0.5, GAMMA)
    changed = {(int(i), int(j)) for i, j in zip(*np.nonzero(new != Q))}
    want = {(s, 1), (s, h)}
    if changed != want:
        got = sorted((i, options[j].name) for i, j in changed)
        fail(f"expected updates to options a1 and {options[h].name} in state {s} only, got {got}")
    almost(new[s, 1], -0.95, 1e-9, label="Q[s, primitive right]")
    almost(new[s, h], -2.75, 1e-9, label="Q[s, hallway option]")
    done = intra_option_q_update(Q.copy(), s, 1, -1.0, s2, True, options, 0.5, GAMMA)
    almost(done[s, h], -0.5, 1e-9, label="Q[s, hallway option] when the step ended the episode")


@check("interrupted_beta ends an option where another option is strictly better")
def test_interrupted_beta(sub):
    env, prims, halls = _setup(sub)
    interrupted_beta = has(sub, "interrupted_beta")
    options = prims + halls
    h = _find(options, "room0->(3, 6)")
    # Every option is worth -3 everywhere, except that primitive "right" is worth -2 at (2, 2).
    Q = np.full((len(env.cells), len(options)), -3.0)
    better, tied = env.index[(2, 2)], env.index[(2, 3)]
    Q[better, 1] = -2.0
    beta = interrupted_beta(options, Q, h)
    equal(beta.shape, (len(env.cells),), label="shape")
    equal(float(beta[better]), 1.0, label="beta where a primitive is strictly better")
    # WRONG IMPLEMENTATION: interrupting on ties (>= instead of >) would also end the option here.
    equal(float(beta[tied]), 0.0, label="beta where the best alternative only ties")
    equal(float(beta[env.index[(3, 6)]]), 1.0, label="beta at the hallway keeps its own value 1")
    equal(float(beta[env.index[(8, 8)]]), 1.0, label="beta outside the option's room keeps its own value 1")


@check("intra-option learning is closer to the optimal option values than SMDP learning after 100000 steps")
def test_intra_beats_smdp(sub):
    env, prims, halls = _setup(sub)
    options = prims + halls
    learn = has(sub, "learn")
    V = _reference_v(env)
    target = np.full((len(env.cells), len(options)), np.nan)
    for i, o in enumerate(options):
        rows = np.arange(len(env.cells))
        P_pi, r_pi = env.P[rows, o.pi], env.R[rows, o.pi]
        M = np.eye(len(rows)) - GAMMA * P_pi * (1 - o.beta)[None, :]
        r_o = np.linalg.solve(M, r_pi)
        p_o = np.linalg.solve(M, GAMMA * P_pi * o.beta[None, :])
        target[:, i] = np.where(o.init, r_o + p_o @ V, np.nan)
    errors = {}
    for mode in ("smdp", "intra"):
        errs = [np.nanmean(np.abs(learn(env, options, mode, 100000, 0.1, 0.1, seed) - target))
                for seed in (0, 1)]
        errors[mode] = float(np.mean(errs))
    if errors["intra"] > 0.4:
        fail(f"intra-option learning should reach a mean error under 0.4, got {errors['intra']:.3f}")
    if errors["smdp"] < 2.0 * errors["intra"]:
        fail(f"SMDP learning should be at least twice as far from optimal as intra-option learning: "
             f"SMDP error {errors['smdp']:.3f}, intra-option error {errors['intra']:.3f}")
