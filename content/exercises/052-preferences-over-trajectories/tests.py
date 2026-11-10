"""Checks for the preferences-over-trajectories exercise."""

import numpy as np

import rlenvs
from rlcheck import almost, check, close, equal, fail, finite, finite_difference, has, shape

LAYOUT = ["S...", ".#..", "..H.", "...G"]


# The test's own copy of the world, so the checks never depend on the reader's helpers.
def _world():
    g = rlenvs.gridworld(LAYOUT, gamma=0.9)
    P = g.P.copy()
    cells = g.names["cells"]
    goal = [k for k, (i, j) in enumerate(cells) if LAYOUT[i][j] == "G"][0]
    hole = [k for k, (i, j) in enumerate(cells) if LAYOUT[i][j] == "H"][0]
    for s in (goal, hole):
        P[s] = 0.0
        P[s, :, s] = 1.0
    S = g.n_states
    mdp = rlenvs.TabularMDP(P=P, R=np.zeros((S, 4)), gamma=0.9, mu0=g.mu0,
                            terminal=np.zeros(S, dtype=bool))
    gi, gj = cells[goal]
    r = np.array([-0.15 * (abs(i - gi) + abs(j - gj)) for (i, j) in cells])
    r[goal], r[hole] = 1.0, -1.0
    return mdp, r


def _with_reward(mdp, r):
    R3 = np.broadcast_to(np.asarray(r, dtype=float)[None, None, :], mdp.P.shape).copy()
    return rlenvs.TabularMDP(P=mdp.P, R=(mdp.P * R3).sum(axis=2), gamma=mdp.gamma,
                             mu0=mdp.mu0, terminal=mdp.terminal, R3=R3)


def _segments(mdp, n, length, rng):
    S, A = mdp.n_states, mdp.n_actions
    segs = np.zeros((n, length), dtype=int)
    for i in range(n):
        s = int(rng.integers(S))
        for t in range(length):
            s = int(rng.choice(S, p=mdp.P[s, int(rng.integers(A))]))
            segs[i, t] = s
    return segs


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _dataset(mdp, r, seed, n=300, length=6, beta=3.0):
    """Segment pairs and Boltzmann labels, drawn with the test's own labeler."""
    rng = np.random.default_rng(seed)
    a = _segments(mdp, n, length, rng)
    b = _segments(mdp, n, length, rng)
    gap = r[a].sum(axis=1) - r[b].sum(axis=1)
    labels = (rng.random(n) < _sigmoid(beta * gap)).astype(float)
    return a, b, labels


def _regret(mdp, r_true, r_hat):
    true_mdp, learned = _with_reward(mdp, r_true), _with_reward(mdp, r_hat)
    q_hat = rlenvs.q_from_v(learned, rlenvs.value_iteration(learned))
    v_pi = rlenvs.policy_evaluation(true_mdp, rlenvs.greedy(q_hat))
    return float(np.max(rlenvs.value_iteration(true_mdp) - v_pi))


# The hand-worked example: four states with rewards 0.2, 0.5, 1.0, 0.3.
R4 = np.array([0.2, 0.5, 1.0, 0.3])


@check("segment_return sums the per-step rewards, so a longer segment can be worth more")
def test_segment_return(sub):
    segment_return = has(sub, "segment_return")
    reward_fn = lambda states: R4[states]
    got_a = segment_return(reward_fn, np.array([0, 1, 2]))
    got_c = segment_return(reward_fn, np.array([2]))
    almost(got_a, 1.7, 1e-12, label="return of the 3-step segment 0.2 + 0.5 + 1.0")
    # Wrong implementation caught here: comparing the MEAN reward of unequal-length segments.
    # The mean says the 1-step segment wins (1.0 vs 0.567); the sum says the 3-step one does (1.7 vs 1.0).
    almost(got_c, 1.0, 1e-12, label="return of the 1-step segment (a mean would give the same 1.0)")
    if not got_a > got_c:
        fail(f"expected the 3-step segment (1.7) to beat the 1-step segment (1.0), got {got_a} and {got_c}")


@check("preference_prob is the sigmoid of the return gap, and stays finite for huge gaps")
def test_preference_prob(sub):
    preference_prob = has(sub, "preference_prob")
    almost(preference_prob(1.0, 1.0), 0.5, 1e-12, label="equal returns")
    almost(preference_prob(1.7, 1.3), 0.598687660112452, 1e-9, label="sigmoid(0.4)")
    almost(preference_prob(1.3, 1.7), 1.0 - 0.598687660112452, 1e-9, label="swapping A and B")
    big, small = preference_prob(400.0, -400.0), preference_prob(-400.0, 400.0)
    finite(np.array([big, small]), label="probabilities at gaps of +-800")
    almost(big, 1.0, 1e-12, label="gap +800")
    almost(small, 0.0, 1e-12, label="gap -800")


@check("a constant added to every step leaves equal-length preferences alone, and moves unequal-length ones")
def test_shift_invariance(sub):
    segment_return = has(sub, "segment_return")
    preference_prob = has(sub, "preference_prob")
    rng = np.random.default_rng(0)
    r = rng.normal(size=6)
    a, b = rng.integers(0, 6, size=5), rng.integers(0, 6, size=5)
    base = preference_prob(segment_return(lambda s: r[s], a), segment_return(lambda s: r[s], b))
    for c in (-2.0, 0.7, 40.0):
        shifted = preference_prob(
            segment_return(lambda s: r[s] + c, a), segment_return(lambda s: r[s] + c, b)
        )
        almost(shifted, base, 1e-9, label=f"P(A preferred) after adding {c} per step, equal lengths")
    short = b[:3]
    p0 = preference_prob(segment_return(lambda s: r[s], a), segment_return(lambda s: r[s], short))
    p1 = preference_prob(segment_return(lambda s: r[s] + 2.0, a), segment_return(lambda s: r[s] + 2.0, short))
    if abs(p1 - p0) < 0.05:
        fail(f"with lengths 5 and 3, adding 2 per step must change P(A preferred); got {p0:.4f} then {p1:.4f}")


@check("bt_segment_loss matches the hand-worked pair, and the label says which segment won")
def test_bt_loss_by_hand(sub):
    bt_segment_loss = has(sub, "bt_segment_loss")
    X = np.eye(4)
    a, b = np.array([[0, 1, 2]]), np.array([[1, 1, 3]])
    # Return of A is 0.2 + 0.5 + 1.0 = 1.7. Return of B is 0.5 + 0.5 + 0.3 = 1.3. The gap is 0.4.
    loss1, grad1 = bt_segment_loss(R4, X, a, b, np.array([1.0]))
    almost(loss1, -np.log(0.598687660112452), 1e-9, label="loss when A is labeled the winner: -ln sigmoid(0.4)")
    # Wrong implementation caught here: a sign-flipped label gives -ln(1 - sigmoid(0.4)) = 0.913 here.
    loss0, _ = bt_segment_loss(R4, X, a, b, np.array([0.0]))
    almost(loss0, -np.log(1.0 - 0.598687660112452), 1e-9, label="loss when B is labeled the winner")
    # Feature counts: A visits states 0, 1, 2 once. B visits state 1 twice and state 3 once.
    want = (0.598687660112452 - 1.0) * np.array([1.0, -1.0, 1.0, -1.0])
    close(grad1, want, atol=1e-9, label="gradient (p - y) * (counts of A - counts of B)")


@check("bt_segment_loss's gradient matches finite differences on random pairs")
def test_bt_gradient(sub):
    bt_segment_loss = has(sub, "bt_segment_loss")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(7, 4))
    a, b = rng.integers(0, 7, size=(25, 5)), rng.integers(0, 7, size=(25, 5))
    labels = rng.integers(0, 2, size=25).astype(float)
    w = rng.normal(size=4)
    _, grad = bt_segment_loss(w, X, a, b, labels)
    numeric = finite_difference(lambda v: bt_segment_loss(v, X, a, b, labels)[0], w)
    close(grad, numeric, atol=1e-6, rtol=1e-5, label="analytic gradient against finite differences")
    shape(grad, (4,), label="gradient")


@check("simulate_preference follows the Boltzmann probability, never flips, and draws one number")
def test_simulate_preference(sub):
    simulate_preference = has(sub, "simulate_preference")
    a, b = np.array([0, 1, 2]), np.array([1, 1, 3])
    rng = np.random.default_rng(0)
    labels = np.array([simulate_preference(R4, a, b, 2.0, rng) for _ in range(4000)])
    # The gap is 0.4 and beta = 2, so A is preferred with probability sigmoid(0.8) = 0.690.
    almost(labels.mean(), _sigmoid(0.8), 0.03, label="frequency of 'A preferred' over 4000 draws")
    rng = np.random.default_rng(1)
    sure = [simulate_preference(R4, a, b, 1000.0, rng) for _ in range(20)]
    swapped = [simulate_preference(R4, b, a, 1000.0, rng) for _ in range(20)]
    equal(int(sum(sure)), 20, label="A is clearly better, so every label should be 1")
    equal(int(sum(swapped)), 0, label="B is clearly better when the pair is swapped, so every label should be 0")
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    simulate_preference(R4, a, b, 2.0, r1)
    r2.random()
    close(r1.random(3), r2.random(3), atol=0, label="the rng stream after one call (it must draw exactly one number)")


@check("fit_reward learns a reward that tracks the truth, on 5 seeds")
def test_fit_reward(sub):
    fit_reward = has(sub, "fit_reward")
    bt_segment_loss = has(sub, "bt_segment_loss")
    mdp, r = _world()
    X = np.eye(mdp.n_states)
    for seed in range(5):
        a, b, y = _dataset(mdp, r, seed)
        w = fit_reward(X, a, b, y)
        shape(w, (mdp.n_states,), label="learned weights")
        finite(w, label="learned weights")
        loss, _ = bt_segment_loss(w, X, a, b, y)
        if not loss < np.log(2.0) - 0.05:
            fail(f"seed {seed}: expected the fitted loss to drop well below ln 2 = 0.693, got {loss:.4f}")
        corr = float(np.corrcoef(w, r)[0, 1])
        if corr < 0.7:
            fail(f"seed {seed}: expected the learned reward to correlate above 0.7 with the true one, got {corr:.3f}")


@check("the learned reward induces an optimal policy as good as the true one, and a shift cannot change it")
def test_policy_from_learned_reward(sub):
    fit_reward = has(sub, "fit_reward")
    mdp, r = _world()
    X = np.eye(mdp.n_states)
    for seed in range(5):
        a, b, y = _dataset(mdp, r, seed)
        w = fit_reward(X, a, b, y)
        regret = _regret(mdp, r, X @ w)
        if regret > 0.05:
            fail(f"seed {seed}: expected the learned reward's greedy policy to lose under 0.05 of true value, got {regret:.3f}")
        shifted = _regret(mdp, r, X @ w + 5.0)
        almost(shifted, regret, 1e-9, label=f"seed {seed}: regret after adding 5 per step to the learned reward")
    # With every label flipped the learner pursues the hole. That is what a sign error does.
    a, b, y = _dataset(mdp, r, 0)
    bad = _regret(mdp, r, X @ fit_reward(X, a, b, 1.0 - y))
    if bad < 1.0:
        fail(f"expected flipped labels to give a clearly worse policy (regret above 1), got {bad:.3f}")


@check("disagreement_query picks the pair the ensemble disagrees about most, not the biggest gap")
def test_disagreement_query(sub):
    disagreement_query = has(sub, "disagreement_query")
    X = np.eye(4)
    # Three candidate pairs, state 3 as the common opponent. Member returns gaps, one column per pair:
    #   pair 0: (4, 4, 4)      all agree, and strongly
    #   pair 1: (0.1, 0, -0.1) all near a coin flip, so they agree too
    #   pair 2: (3, -3, 0)     the members split: probabilities 0.953, 0.047, 0.5
    ws = np.array([[4.0, 0.1, 3.0, 0.0], [4.0, 0.0, -3.0, 0.0], [4.0, -0.1, 0.0, 0.0]])
    a = np.array([[0], [1], [2]])
    b = np.array([[3], [3], [3]])
    got = disagreement_query(ws, X, a, b)
    equal(int(got), 2, label="index of the highest-variance pair (variances 0, about 0.0004, about 0.14)")
    got = disagreement_query(ws, X, a[::-1], b[::-1])
    equal(int(got), 0, label="the same pairs in reverse order, so the answer is an index into the arrays you pass")
    # Wrong implementation: the variance of the return gaps (logits) instead of
    # the probabilities. Pair 0 has gaps (5, 15): the logits differ a lot, but
    # both members are near-certain A wins (variance about 1e-5). Pair 1 has
    # gaps (1, 3): probabilities 0.731 and 0.953 (variance about 0.012).
    ws = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    X = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    got = disagreement_query(ws, X, np.array([[2], [1]]), np.array([[0], [0]]))
    equal(int(got), 1, label="saturated pair against a split pair: the variance is over probabilities, not return gaps")
