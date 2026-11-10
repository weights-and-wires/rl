"""Checks for the offline reinforcement learning exercise."""

import numpy as np

import rlenvs
from rlcheck import almost as _almost, check, close, fail, finite, finite_difference, has, shape


def almost(got, want, tol, label=None):
    """rlcheck.almost, except that NaN or infinity fails instead of slipping past the comparison."""
    finite(got, label=label)
    _almost(got, want, tol, label=label)


def make_dataset(mdp, behavior, n_episodes, seed, max_steps=500):
    """The same helper the starter provides, so the checks don't depend on the reader's copy."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_episodes):
        s = int(rng.choice(mdp.n_states, p=mdp.mu0))
        for _ in range(max_steps):
            a = int(rng.choice(mdp.n_actions, p=behavior[s]))
            s2, r = mdp.sample(s, a, rng)
            done = bool(mdp.terminal[s2])
            rows.append((s, a, r, s2, float(done)))
            if done:
                break
            s = s2
    s, a, r, s2, done = (np.array(col) for col in zip(*rows))
    return {"s": s.astype(int), "a": a.astype(int), "r": r.astype(float), "s2": s2.astype(int), "done": done}


def cautious_behavior(mdp):
    """Heads for the goal, but never steps toward the cliff: no down in row 2, no right at the start."""
    S, A = mdp.n_states, mdp.n_actions
    allowed = np.ones((S, A), dtype=bool)
    for k, (i, j) in enumerate(mdp.names["cells"]):
        if i == 2 and 1 <= j <= 10:
            allowed[k, rlenvs.DOWN] = False
    allowed[mdp.names["start"], rlenvs.RIGHT] = False
    q = rlenvs.q_from_v(mdp, rlenvs.value_iteration(mdp))
    best = np.where(allowed, q, -np.inf).argmax(axis=1)
    pi = 0.4 * allowed / allowed.sum(axis=1, keepdims=True)
    pi[np.arange(S), best] += 0.6
    return pi


def _corridor():
    mdp = rlenvs.gridworld(["S.G"], gamma=0.9)
    behavior = np.zeros((3, 4))
    behavior[:, rlenvs.RIGHT] = 1.0
    return mdp, make_dataset(mdp, behavior, 3, seed=0)


def _cliff(seed=0):
    mdp = rlenvs.cliff_walking(gamma=0.9)
    return mdp, make_dataset(mdp, cautious_behavior(mdp), 100, seed=seed)


def _counts(data, S, A):
    counts = np.zeros((S, A))
    np.add.at(counts, (data["s"], data["a"]), 1.0)
    return counts


def _restricted_optimum(mdp, allowed):
    """Value iteration that may only use the actions in `allowed`.

    States with no allowed action keep value 0. No allowed action reaches
    them, so they never affect the states the data covers.
    """
    has_action = allowed.any(axis=1)
    v = np.zeros(mdp.n_states)
    for _ in range(3000):
        q = np.where(allowed, rlenvs.q_from_v(mdp, v), -np.inf)
        v = np.where(has_action & ~mdp.terminal, q.max(axis=1), 0.0)
    if not np.all(np.isfinite(v)):
        raise RuntimeError("the restricted optimum should be finite everywhere")
    return v


@check("offline_q_iteration on the corridor leaves unseen actions at 0 and prefers them")
def test_offline_q_corridor(sub):
    offline_q_iteration = has(sub, "offline_q_iteration")
    mdp, data = _corridor()
    Q = offline_q_iteration(data, 3, 4, 0.9, iters=200)
    shape(Q, (3, 4), label="Q")
    want = np.zeros((3, 4))
    want[0, rlenvs.RIGHT] = -1.0  # -1 + 0.9 * max(0, -1, 0, 0): the unseen zeros win the max
    want[1, rlenvs.RIGHT] = -1.0  # -1 into the goal, no bootstrap
    close(Q, want, atol=1e-12, label="offline Q on the corridor")
    true_q = rlenvs.q_from_v(mdp, rlenvs.value_iteration(mdp))
    if not np.all(Q[:2] >= true_q[:2]):
        fail("every entry of the offline Q should sit at or above the true Q* here")
    v = rlenvs.policy_evaluation(mdp, rlenvs.greedy(Q))
    almost(v[0], -10.0, tol=1e-9, label="true value of the greedy policy, which bumps a wall forever")


@check("offline_q_iteration on the cliff matches the backup fixed point, and its greedy policy falls off")
def test_offline_q_cliff(sub):
    offline_q_iteration = has(sub, "offline_q_iteration")
    mdp, data = _cliff()
    S, A = mdp.n_states, mdp.n_actions
    seen = _counts(data, S, A) > 0
    Q = offline_q_iteration(data, S, A, mdp.gamma, iters=500)
    # The data are deterministic, so on seen pairs the target is R + gamma * max Q(s2).
    want = np.zeros((S, A))
    nxt = mdp.P.argmax(axis=2)
    for _ in range(500):
        backed = mdp.R + mdp.gamma * (~mdp.terminal[nxt]) * want[nxt].max(axis=2)
        want = np.where(seen, backed, 0.0)
    close(Q, want, atol=1e-8, label="Q against the fixed point with unseen actions held at 0")
    true_q = rlenvs.q_from_v(mdp, rlenvs.value_iteration(mdp))
    visited = seen.any(axis=1)
    gap = (Q - true_q)[visited[:, None] & ~seen]
    if gap.min() < 5.0:
        fail(f"unseen actions should be overestimated by at least 5, the smallest gap is {gap.min():.3f}")
    start = mdp.names["start"]
    v = rlenvs.policy_evaluation(mdp, rlenvs.greedy(Q))[start]
    if v > -100.0:
        fail(f"the greedy policy should keep stepping into the cliff (value near -1000), got {v:.2f}")


@check("cql_penalty is logsumexp minus the data action's Q: 2.4076 for [1, 2, 3] and action 0")
def test_cql_penalty(sub):
    cql_penalty = has(sub, "cql_penalty")
    almost(cql_penalty(np.array([[1.0, 2.0, 3.0]]), np.array([0]), np.array([0])),
           np.log(np.e + np.e**2 + np.e**3) - 1.0, tol=1e-12, label="penalty for Q = [1, 2, 3], action 0")
    almost(cql_penalty(np.array([[0.0, -1.0, 0.0, 0.0]]), np.array([0]), np.array([1])),
           np.log(3 + np.exp(-1.0)) + 1.0, tol=1e-12, label="penalty at the corridor's start state")
    big = cql_penalty(np.array([[1000.0, 999.0]]), np.array([0]), np.array([0]))
    almost(big, np.log1p(np.exp(-1.0)), tol=1e-9, label="penalty at Q = [1000, 999] (subtract the max)")
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(4, 3))
    s, a = rng.integers(0, 4, 20), rng.integers(0, 3, 20)
    rows = Q[s]
    want = np.mean(np.log(np.exp(rows).sum(axis=1)) - Q[s, a])
    almost(cql_penalty(Q, s, a), want, tol=1e-12, label="mean penalty over a batch of 20")


@check("cql_penalty_grad matches finite differences")
def test_cql_penalty_grad(sub):
    cql_penalty = has(sub, "cql_penalty")
    cql_penalty_grad = has(sub, "cql_penalty_grad")
    rng = np.random.default_rng(1)
    Q = rng.normal(size=(5, 3))
    s, a = rng.integers(0, 4, 30), rng.integers(0, 3, 30)
    got = cql_penalty_grad(Q, s, a)
    shape(got, (5, 3), label="gradient")
    fd = finite_difference(lambda q: cql_penalty(q.reshape(5, 3), s, a), Q.copy())
    close(got, fd, rtol=1e-6, atol=1e-9, label="gradient against finite differences")


@check("cql_q_iteration sinks unseen actions, so the greedy policy stays on the data and walks the safe path")
def test_cql(sub):
    cql_q_iteration = has(sub, "cql_q_iteration")
    mdp, data = _corridor()
    Q = cql_q_iteration(data, 3, 4, 0.9, alpha=0.5, iters=1000, lr=1.0)
    if not (Q[0].argmax() == rlenvs.RIGHT and Q[1].argmax() == rlenvs.RIGHT):
        fail(f"on the corridor, right should have the highest Q in both cells, got rows {Q[0]} and {Q[1]}")

    mdp, data = _cliff()
    S, A = mdp.n_states, mdp.n_actions
    seen = _counts(data, S, A) > 0
    Q = cql_q_iteration(data, S, A, mdp.gamma, alpha=0.5, iters=1000, lr=1.0)
    finite(Q, label="CQL's Q")
    visited = seen.any(axis=1)
    greedy = Q.argmax(axis=1)
    off = [int(k) for k in np.flatnonzero(visited) if not seen[k, greedy[k]]]
    if off:
        fail(f"in states {off[:5]} the greedy action never appears in the data")
    naive_unseen = 0.0
    worst = Q[visited[:, None] & ~seen].max()
    if worst > naive_unseen - 3.0:
        fail(f"CQL should push unseen actions well below 0, the highest is {worst:.3f}")
    start = mdp.names["start"]
    v = rlenvs.policy_evaluation(mdp, rlenvs.greedy(Q))[start]
    best = _restricted_optimum(mdp, seen)[start]
    almost(v, best, tol=1e-6, label="true value of CQL's greedy policy against the best in-data policy")


@check("expectile_loss is |tau - 1(u < 0)| u^2, and expectile_loss_grad matches finite differences")
def test_expectile_loss(sub):
    expectile_loss = has(sub, "expectile_loss")
    expectile_loss_grad = has(sub, "expectile_loss_grad")
    almost(expectile_loss(np.array([1.0]), 0.7), 0.7, tol=1e-12, label="L at u = 1, tau = 0.7")
    almost(expectile_loss(np.array([-1.0]), 0.7), 0.3, tol=1e-12, label="L at u = -1, tau = 0.7")
    almost(expectile_loss(np.array([2.0, -3.0]), 0.9), (0.9 * 4 + 0.1 * 9) / 2, tol=1e-12,
           label="mean L at u = [2, -3], tau = 0.9")
    u = np.random.default_rng(2).normal(size=12)
    for tau in (0.5, 0.8):
        fd = finite_difference(lambda x: expectile_loss(x, tau), u.copy())
        close(expectile_loss_grad(u, tau), fd, rtol=1e-6, atol=1e-9,
              label=f"gradient at tau = {tau} against finite differences")
    u5 = np.array([1.5, -0.5, 2.0])
    close(expectile_loss_grad(u5, 0.5), u5 / 3, atol=1e-12,
          label="at tau = 0.5 the gradient is u / n, half that of the mean squared error")


@check("expectile solves the weighted loss: the mean at tau = 0.5, rising toward the max as tau grows")
def test_expectile(sub):
    expectile = has(sub, "expectile")
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4, 5))
    w = rng.integers(1, 6, size=(4, 5)).astype(float)
    close(expectile(x, w, 0.5), (w * x).sum(axis=1) / w.sum(axis=1), atol=1e-10,
          label="expectile at tau = 0.5 against the weighted mean")
    for tau in (0.7, 0.95):
        v = expectile(x, w, tau)
        shape(v, (4,), label="expectile output")
        for i in range(4):
            def loss(vv):
                u = x[i] - vv[0]
                return np.sum(w[i] * np.where(u < 0, 1 - tau, tau) * u**2)
            g = finite_difference(loss, np.array([v[i]]))
            if not abs(g[0]) <= 1e-5:
                fail(f"at tau = {tau} row {i}, the weighted loss still has slope {g[0]:.2e} at your expectile")
    taus = [0.5, 0.7, 0.9, 0.99, 0.9999]
    vals = np.stack([expectile(x, w, t) for t in taus])
    if np.any(np.diff(vals, axis=0) < -1e-12):
        fail("the expectile should never fall as tau grows")
    close(vals[-1], x.max(axis=1), atol=0.02, label="expectile at tau = 0.9999 against the max")


@check("iql_values at tau = 0.5 equals the exact value of the data's own behaviour policy")
def test_iql_behavior_value(sub):
    iql_values = has(sub, "iql_values")
    mdp, data = _cliff()
    S, A = mdp.n_states, mdp.n_actions
    counts = _counts(data, S, A)
    visited = counts.sum(axis=1) > 0
    Q, V = iql_values(data, S, A, mdp.gamma, 0.5, iters=500)
    shape(Q, (S, A), label="Q")
    shape(V, (S,), label="V")
    pi_hat = np.where(visited[:, None], counts / np.maximum(counts.sum(axis=1, keepdims=True), 1), 0.25)
    v_exact = rlenvs.policy_evaluation(mdp, pi_hat)
    close(V[visited], v_exact[visited], atol=1e-8, label="IQL V at tau = 0.5 against policy evaluation")


@check("iql_values climbs with tau toward the best value the data supports, never using unseen actions")
def test_iql_tau(sub):
    iql_values = has(sub, "iql_values")
    mdp, data = _cliff()
    S, A = mdp.n_states, mdp.n_actions
    seen = _counts(data, S, A) > 0
    visited = seen.any(axis=1)
    best = _restricted_optimum(mdp, seen)
    starts = []
    for tau in (0.5, 0.9, 0.99):
        _, V = iql_values(data, S, A, mdp.gamma, tau, iters=500)
        starts.append(V[mdp.names["start"]])
        if np.any(V[visited] > best[visited] + 1e-8):
            fail(f"at tau = {tau}, V exceeds the best in-data value: IQL must not use unseen actions")
    if not (starts[0] < starts[1] < starts[2]):
        fail(f"V(start) should rise with tau, got {np.round(starts, 3)}")
    gap = np.abs(V[visited] - best[visited]).max()
    if not gap <= 0.1:
        fail(f"at tau = 0.99, V should be within 0.1 of the best in-data value, the gap is {gap:.3f}")


def _bc_reference(pi, data_actions, q_pi, q_data, alpha):
    """The paper's minimized actor loss, with λ held constant."""
    lam = float(alpha) / float(np.mean(np.abs(q_data)))
    return float(np.mean((pi - data_actions) ** 2) - lam * np.mean(q_pi))


# Wrong implementation this check catches: using alpha as lambda, without
# dividing by the mean absolute Q. On the three-row batch that loss is -1.82,
# not 0.08.
@check("bc_regularized_actor_loss is the mean squared gap minus lambda times the mean Q")
def test_bc_regularized_actor_loss(sub):
    bc_regularized_actor_loss = has(sub, "bc_regularized_actor_loss")
    pi = np.array([0.5, 0.0, 0.1])
    logged = np.array([0.0, 0.2, -0.4])
    q_pi = 4.0 * pi
    q_data = np.array([10.0, -20.0, 30.0])
    got = bc_regularized_actor_loss(pi, logged, q_pi, q_data, 2.5)
    almost(got, 0.08, tol=1e-12, label="loss at alpha = 2.5 on the three-row batch")
    almost(bc_regularized_actor_loss(pi, logged, q_pi, q_data, 1.0), 0.14, tol=1e-12,
           label="loss at alpha = 1 on the three-row batch")
    rng = np.random.default_rng(0)
    pi_r = rng.normal(size=6)
    logged_r = rng.normal(size=6)
    q_pi_r = rng.normal(size=6)
    q_data_r = rng.normal(size=6)
    want = _bc_reference(pi_r, logged_r, q_pi_r, q_data_r, 1.7)
    almost(bc_regularized_actor_loss(pi_r, logged_r, q_pi_r, q_data_r, 1.7), want, tol=1e-12,
           label="loss on a seeded batch of 6 at alpha = 1.7")


@check("bc_regularized_actor_loss gradient matches a finite difference, with lambda held fixed")
def test_bc_regularized_actor_grad(sub):
    bc_regularized_actor_loss = has(sub, "bc_regularized_actor_loss")
    pi = np.array([0.5, 0.0, 0.1])
    logged = np.array([0.0, 0.2, -0.4])
    q_data = np.array([10.0, -20.0, 30.0])

    def lesson_loss(x):
        return bc_regularized_actor_loss(x, logged, 4.0 * x, q_data, 2.5)

    fd = finite_difference(lesson_loss, pi.copy())
    lam = 2.5 / np.mean(np.abs(q_data))
    want = 2.0 * (pi - logged) / pi.size - lam * 4.0 / pi.size
    close(fd, want, rtol=1e-6, atol=1e-8, label="gradient on the three-row batch")
    close(fd, np.array([1.0 / 6.0, -0.3, 1.0 / 6.0]), atol=1e-8,
          label="gradient against 1/6, -0.3, 1/6")

    rng = np.random.default_rng(1)
    pi0 = rng.normal(size=5)
    logged_r = rng.normal(size=5)
    q_data_r = rng.normal(size=5)
    slope = 1.5

    def seeded_loss(x):
        return bc_regularized_actor_loss(x, logged_r, slope * x + 0.25, q_data_r, 2.5)

    fd2 = finite_difference(seeded_loss, pi0.copy())
    lam2 = 2.5 / np.mean(np.abs(q_data_r))
    want2 = 2.0 * (pi0 - logged_r) / pi0.size - lam2 * slope / pi0.size
    close(fd2, want2, rtol=1e-6, atol=1e-8, label="gradient on a seeded batch with Q = 1.5 pi + 0.25")

