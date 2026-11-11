"""Checks for the cooperative multi-agent RL exercise."""

import numpy as np

from rlcheck import almost as _almost, check, close, equal, fail, finite, finite_difference, has, shape


def almost(got, want, tol, label=None):
    """rlcheck.almost, except that NaN or infinity fails instead of slipping past the comparison."""
    finite(got, label=label)
    _almost(got, want, tol, label=label)


CLIMBING = np.array([[11.0, -30.0, 0.0], [-30.0, 7.0, 6.0], [0.0, 0.0, 5.0]])
AND = np.array([[0.0, 0.0], [0.0, 1.0]])
XOR = np.array([[1.0, 0.0], [0.0, 1.0]])


def hyper_params(rng, n_agents, hidden, state_dim):
    """Random hypernetwork parameters with both signs, so a missing abs shows up."""
    return {
        "w1_A": rng.normal(size=(state_dim, n_agents * hidden)),
        "w1_c": rng.normal(size=n_agents * hidden),
        "b1_A": rng.normal(size=(state_dim, hidden)),
        "b1_c": rng.normal(size=hidden),
        "w2_A": rng.normal(size=(state_dim, hidden)),
        "w2_c": rng.normal(size=hidden),
        "b2_A": rng.normal(size=(state_dim, 1)),
        "b2_c": rng.normal(size=1),
    }


@check("vdn_q_tot sums the agents' utilities: [[1, 2], [3, -1]] at actions (1, 0) is 2 + 3 = 5")
def test_vdn_q_tot(sub):
    vdn_q_tot = has(sub, "vdn_q_tot")
    # Named wrong implementation: VDN that averages instead of summing. It still
    # satisfies IGM, so only a check on the value itself catches it (here: 2.5, not 5).
    almost(vdn_q_tot(np.array([[1.0, 2.0], [3.0, -1.0]]), np.array([1, 0])), 5.0, tol=1e-12, label="two agents")
    qs = np.array([[0.5, -1.0, 2.0], [1.0, 1.5, -2.0], [0.0, 3.0, 0.25]])
    almost(vdn_q_tot(qs, np.array([2, 1, 0])), 2.0 + 1.5 + 0.0, tol=1e-12, label="three agents")
    almost(vdn_q_tot(qs, np.array([0, 2, 1])), 0.5 - 2.0 + 3.0, tol=1e-12, label="three agents, other joint action")


@check("greedy_actions picks each agent's own argmax")
def test_greedy_actions(sub):
    greedy_actions = has(sub, "greedy_actions")
    out = np.asarray(greedy_actions(np.array([[1.0, 3.0, 0.0], [2.0, -1.0, 4.0], [-5.0, -6.0, -7.0]])))
    equal(out, np.array([1, 2, 0]), label="greedy actions")


@check("IGM: the best VDN joint action equals the tuple of per-agent greedy actions")
def test_igm(sub):
    best_joint_action, greedy_actions = has(sub, "best_joint_action"), has(sub, "greedy_actions")
    rng = np.random.default_rng(0)
    for trial in range(30):
        qs = rng.normal(size=(3, 4))
        got = tuple(int(a) for a in best_joint_action(qs))
        want = tuple(int(a) for a in greedy_actions(qs))
        if got != want:
            fail(f"trial {trial}: best joint action {got}, per-agent greedy actions {want}")
    got = tuple(int(a) for a in best_joint_action(np.array([[1.0, 3.0, 0.0], [2.0, -1.0, 4.0]])))
    if got != (1, 2):
        fail(f"expected joint action (1, 2) for [[1, 3, 0], [2, -1, 4]], got {got}")


@check("qmix_mixer matches the hand computation, with abs on both weight layers and state-dependent weights")
def test_qmix_values(sub):
    qmix_mixer = has(sub, "qmix_mixer")
    # N = 2 agents, H = 1 hidden unit, S = 1 state feature. Raw weights are negative on purpose.
    hyper = {
        "w1_A": np.array([[-0.5, 1.0]]), "w1_c": np.zeros(2),
        "b1_A": np.array([[0.25]]), "b1_c": np.zeros(1),
        "w2_A": np.array([[-2.0]]), "w2_c": np.zeros(1),
        "b2_A": np.array([[0.5]]), "b2_c": np.zeros(1),
    }
    # state 1: W1 = |(-0.5, 1)| = (0.5, 1), b1 = 0.25, w2 = 2, b2 = 0.5.
    #   qs (1, 2): pre = 0.5 + 2 + 0.25 = 2.75, elu = 2.75, Q = 2 * 2.75 + 0.5 = 6.
    #   qs (-1, -2): pre = -2.25, elu = exp(-2.25) - 1, Q = 2 * (exp(-2.25) - 1) + 0.5.
    # state 2: W1 = (1, 2), b1 = 0.5, w2 = 4, b2 = 1. qs (1, 2): pre = 5.5, Q = 4 * 5.5 + 1 = 23.
    qs = np.array([[1.0, 2.0], [-1.0, -2.0], [1.0, 2.0]])
    state = np.array([[1.0], [1.0], [2.0]])
    want = np.array([6.0, 2.0 * (np.exp(-2.25) - 1.0) + 0.5, 23.0])
    got = np.asarray(qmix_mixer(qs, state, hyper))
    shape(got, (3,), label="Q_tot")
    close(got, want, atol=1e-9, label="Q_tot")


@check("QMIX is monotone: dQ_tot / dQ_i >= 0 for every agent, on random states and random hypernetworks")
def test_qmix_monotone(sub):
    qmix_mixer = has(sub, "qmix_mixer")
    rng = np.random.default_rng(1)
    eps = 1e-5
    for trial in range(10):
        n_agents, hidden, state_dim = 3, 4, 3
        hyper = hyper_params(rng, n_agents, hidden, state_dim)
        qs = rng.normal(size=(6, n_agents))
        state = rng.normal(size=(6, state_dim))
        for i in range(n_agents):
            up, down = qs.copy(), qs.copy()
            up[:, i] += eps
            down[:, i] -= eps
            slope = (np.asarray(qmix_mixer(up, state, hyper)) - np.asarray(qmix_mixer(down, state, hyper))) / (2 * eps)
            # Named wrong implementation: mixing weights without abs. A raw weight below
            # zero makes some slope negative, and this check fails.
            if slope.min() < -1e-7:
                fail(f"trial {trial}, agent {i}: dQ_tot/dQ_i = {slope.min():.4f} < 0, so the mixer is not monotone")
    # The hypernetworks must use the state: moving the state moves Q_tot.
    hyper = hyper_params(rng, 2, 3, 2)
    a = np.asarray(qmix_mixer(np.ones((1, 2)), np.array([[1.0, 0.0]]), hyper))
    b = np.asarray(qmix_mixer(np.ones((1, 2)), np.array([[0.0, 1.0]]), hyper))
    if abs(float(a[0]) - float(b[0])) < 1e-9:
        fail("Q_tot did not change when the state changed; the weights must come from the state")


@check("vdn_fit is the exact least-squares additive fit: AND has mse 1/16, an additive table has mse 0")
def test_vdn_fit(sub):
    vdn_fit = has(sub, "vdn_fit")
    q1, q2, mse = vdn_fit(AND)
    close(np.asarray(q1)[:, None] + np.asarray(q2)[None, :], np.array([[-0.25, 0.25], [0.25, 0.75]]), atol=1e-12, label="fitted AND table")
    almost(mse, 1 / 16, tol=1e-12, label="AND mse")
    almost(vdn_fit(XOR)[2], 0.25, tol=1e-12, label="XOR mse")
    additive = np.array([1.0, 2.0, 4.0])[:, None] + np.array([0.0, -1.0])[None, :]
    almost(vdn_fit(additive)[2], 0.0, tol=1e-12, label="an additive 3 x 2 table")
    # The normal equations: the gradient of the mse in (q1, q2) is zero at the fit.
    table = np.random.default_rng(2).normal(size=(3, 4))
    q1, q2, _ = vdn_fit(table)

    def mse_of(theta):
        return float(((table - (theta[:3, None] + theta[None, 3:])) ** 2).mean())

    grad = finite_difference(mse_of, np.concatenate([q1, q2]))
    close(grad, np.zeros(7), atol=1e-7, label="finite-difference gradient at the fit")


@check("fit_qmix represents a monotone table: AND is fit to mse < 0.01, where VDN is stuck at 1/16")
def test_qmix_and(sub):
    fit_qmix = has(sub, "fit_qmix")
    for seed in range(3):
        fitted, mse = fit_qmix(AND, seed=seed)
        shape(np.asarray(fitted), (2, 2), label="fitted table")
        if not mse < 0.01:
            fail(f"seed {seed}: QMIX mse on AND is {mse:.4f}; a monotone mixer can represent AND, so expected < 0.01")
        close(np.asarray(fitted), AND, atol=0.2, label=f"seed {seed} fitted table")


@check("neither VDN nor QMIX represents XOR: both keep mse above 0.1, because no monotone mixer can")
def test_xor(sub):
    fit_qmix = has(sub, "fit_qmix")
    # Row 0 says agent 2 prefers action 0, and row 1 says it prefers action 1.
    # One utility table cannot hold both orderings, so no monotone f fits the table.
    for seed in range(3):
        _, mse = fit_qmix(XOR, seed=seed)
        if not mse > 0.1:
            fail(f"seed {seed}: QMIX mse on XOR is {mse:.4f}; XOR is not monotone, so expected > 0.1")


@check("expected_individual_values: against a uniform partner the safe action 2 looks best on the climbing game")
def test_individual_values(sub):
    expected_individual_values = has(sub, "expected_individual_values")
    close(np.asarray(expected_individual_values(CLIMBING, np.ones(3) / 3)), np.array([-19 / 3, -17 / 3, 5 / 3]), atol=1e-12, label="uniform partner")
    close(np.asarray(expected_individual_values(CLIMBING, np.array([0.0, 0.0, 1.0]))), np.array([0.0, 6.0, 5.0]), atol=1e-12, label="partner plays action 2")
    close(np.asarray(expected_individual_values(CLIMBING, np.array([1.0, 0.0, 0.0]))), np.array([11.0, -30.0, 0.0]), atol=1e-12, label="partner plays action 0")


@check("on the climbing game, independent learners settle on the safe (2, 2) in 5 seeds; the joint learner finds (0, 0)")
def test_climbing(sub):
    independent_learners, joint_learner = has(sub, "independent_learners"), has(sub, "joint_learner")
    for seed in range(5):
        Q1, Q2 = independent_learners(CLIMBING, seed=seed)
        acts = (int(np.argmax(Q1)), int(np.argmax(Q2)))
        if acts != (2, 2):
            fail(f"seed {seed}: independent learners' greedy joint action is {acts}, expected (2, 2) worth 5")
        J = np.asarray(joint_learner(CLIMBING, seed=seed))
        shape(J, (3, 3), label="joint Q")
        best = tuple(int(i) for i in np.unravel_index(np.argmax(J), J.shape))
        if best != (0, 0):
            fail(f"seed {seed}: joint learner's greedy joint action is {best}, expected (0, 0) worth 11")
