"""Checks for the reward design exercise."""

import numpy as np

import rlenvs
from rlcheck import almost as _almost, check, close, equal, fail, finite, has, shape


def almost(got, want, tol, label=None):
    """rlcheck.almost, except that NaN or infinity fails instead of slipping past the comparison."""
    finite(got, label=label)
    _almost(got, want, tol, label=label)


def _chain(gamma=0.9):
    """Five states in a row. Action 0 moves left, action 1 moves right, entering state 4 pays 1."""
    S, A = 5, 2
    P = np.zeros((S, A, S))
    R3 = np.zeros((S, A, S))
    for s in range(4):
        P[s, 0, max(s - 1, 0)] = 1.0
        P[s, 1, s + 1] = 1.0
    P[4, :, 4] = 1.0
    R3[:4, :, 4] = 1.0
    mu0 = np.zeros(S)
    mu0[0] = 1.0
    terminal = np.array([False, False, False, False, True])
    return rlenvs.TabularMDP(P=P, R=(P * R3).sum(axis=2), gamma=gamma, mu0=mu0, terminal=terminal, R3=R3)


def _qstar(mdp):
    return rlenvs.q_from_v(mdp, rlenvs.value_iteration(mdp))


def _maze():
    return rlenvs.gridworld(["S.....", ".####.", "......", ".####.", ".....G"], gamma=0.95)


def _distance_potential(mdp):
    cells, grid = mdp.names["cells"], mdp.names["grid"]
    goal = next(c for c in cells if grid[c[0]][c[1]] == "G")
    return -np.array([abs(i - goal[0]) + abs(j - goal[1]) for i, j in cells], dtype=float)


@check("optimal_actions marks every action within tol of its row's best")
def test_optimal_actions(sub):
    optimal_actions = has(sub, "optimal_actions")
    q = np.array([[1.0, 2.0, 2.0], [0.5, 0.5 - 1e-12, -3.0], [-1.0, -2.0, -1.5]])
    want = np.array([[False, True, True], [True, True, False], [True, False, False]])
    equal(optimal_actions(q, tol=1e-9), want, label="optimal action mask")


@check("shaped_mdp adds gamma * Phi(s') - Phi(s): the chain's shaped rewards by hand")
def test_shaped_rewards(sub):
    shaped_mdp = has(sub, "shaped_mdp")
    mdp = _chain()
    phi = np.array([0.0, 1.0, 2.0, 3.0, 0.0])
    shaped = shaped_mdp(mdp, phi)
    shape(shaped.R, (5, 2), label="shaped R")
    # From state 1, left lands in 0: 0.9 * 0 - 1 = -1. Right lands in 2: 0.9 * 2 - 1 = 0.8.
    # From state 3, right enters the goal: 1 + 0.9 * 0 - 3 = -2.
    want = np.array([[0.0, 0.9], [-1.0, 0.8], [-1.1, 0.7], [-1.2, -2.0], [0.0, 0.0]])
    close(shaped.R[:4], want[:4], atol=1e-12, label="shaped expected rewards in the live states")
    close(shaped.P, mdp.P, atol=0, label="shaping must leave the transitions alone")
    if shaped is mdp or shaped.R is mdp.R:
        fail("shaped_mdp must return a new MDP and leave the original unchanged")
    close(mdp.R[3], np.array([0.0, 1.0]), atol=0, label="the original chain's rewards after shaping")


@check("potential-based shaping gives Q*' = Q* - Phi(s) and the same optimal actions on 10 random MDPs")
def test_shaping_invariance_random(sub):
    shaped_mdp = has(sub, "shaped_mdp")
    optimal_actions = has(sub, "optimal_actions")
    for seed in range(10):
        mdp = rlenvs.random_mdp(n_states=6, n_actions=3, gamma=0.9, seed=seed)
        phi = np.random.default_rng(100 + seed).normal(scale=5.0, size=6)
        q = _qstar(mdp)
        q_shaped = _qstar(shaped_mdp(mdp, phi))
        err = np.abs(q_shaped - (q - phi[:, None])).max()
        if not err <= 1e-8:
            fail(f"seed {seed}: max |Q*_shaped - (Q* - Phi)| is {err:.2e}, expected below 1e-8")
        equal(optimal_actions(q_shaped), optimal_actions(q), label=f"optimal actions, seed {seed}")


@check("with Phi = 0 at the goal, shaping the maze keeps its optimal actions and Q* - Phi")
def test_shaping_invariance_maze(sub):
    shaped_mdp = has(sub, "shaped_mdp")
    optimal_actions = has(sub, "optimal_actions")
    mdp = _maze()
    rng = np.random.default_rng(7)
    for phi in (_distance_potential(mdp), np.where(mdp.terminal, 0.0, rng.normal(scale=3.0, size=mdp.n_states))):
        q = _qstar(mdp)
        q_shaped = _qstar(shaped_mdp(mdp, phi))
        live = ~mdp.terminal
        close(q_shaped[live], (q - phi[:, None])[live], atol=1e-8, label="Q*_shaped against Q* - Phi")
        equal(optimal_actions(q_shaped)[live], optimal_actions(q)[live], label="optimal actions in the maze")


@check("a nonzero potential at terminal states breaks the guarantee: the agent jumps into the hole")
def test_terminal_potential(sub):
    shaped_mdp = has(sub, "shaped_mdp")
    optimal_actions = has(sub, "optimal_actions")
    mdp = rlenvs.gridworld(["HS...G"], gamma=0.9, step_reward=0.0, goal_reward=1.0)
    start = mdp.names["start"]
    q = _qstar(mdp)
    if not optimal_actions(q)[start, rlenvs.RIGHT]:
        fail("without shaping, moving right toward the goal should be optimal at the start")
    phi = np.full(mdp.n_states, 10.0)
    q_bad = _qstar(shaped_mdp(mdp, phi))
    # Every step now pays (0.9 - 1) * 10 = -1, so ending at once in the hole is best.
    close(q_bad[start], np.array([-1.9, -2.71, -1.9, -1.0]), atol=1e-9,
          label="shaped Q* at the start with Phi = 10 everywhere")
    phi_ok = np.where(mdp.terminal, 0.0, 10.0)
    q_ok = _qstar(shaped_mdp(mdp, phi_ok))
    equal(optimal_actions(q_ok)[start], optimal_actions(q)[start],
          label="optimal actions at the start with Phi = 0 at both terminals")


@check("a +0.3 bonus for entering state 1 makes the chain's agent loop and never reach the goal")
def test_bonus_breaks_policy(sub):
    bonus_mdp = has(sub, "bonus_mdp")
    optimal_actions = has(sub, "optimal_actions")
    mdp = _chain()
    close(_qstar(mdp)[0], np.array([0.6561, 0.729]), atol=1e-9, label="Q*(0, .) without the bonus")
    q = _qstar(bonus_mdp(mdp, 1, 0.3))
    loop = 0.3 / (1 - 0.81)
    # Looping 0 -> 1 -> 0 earns 0.3 every two steps: V(0) = 0.3 / (1 - 0.81) = 1.5789.
    close(q[0], np.array([0.9 * loop, loop]), atol=1e-9, label="Q*(0, .) with the bonus")
    mask = optimal_actions(q)
    if mask[2, 1] or mask[3, 1]:
        fail("with the bonus, left should be the only optimal action in states 2 and 3")
    v = rlenvs.policy_evaluation(mdp, rlenvs.greedy(q))
    almost(v[0], 0.0, tol=1e-12, label="true value of the bonus-optimal policy, which never reaches the goal")


@check("q_learning_updates applies the shaped TD update step by step")
def test_q_learning_updates(sub):
    q_learning_updates = has(sub, "q_learning_updates")
    Q0 = np.zeros((3, 2))
    tr = {"s": np.array([0, 1]), "a": np.array([1, 0]), "r": np.array([-1.0, 2.0]),
          "s2": np.array([1, 2]), "done": np.array([0.0, 1.0])}
    Q = q_learning_updates(Q0, tr, alpha=0.5, gamma=0.9)
    close(Q, np.array([[0.0, -0.5], [1.0, 0.0], [0.0, 0.0]]), atol=1e-12, label="two unshaped updates")
    close(Q0, np.zeros((3, 2)), atol=0, label="Q0 after the call (copy it, don't modify it)")
    phi = np.array([1.0, 2.0, 0.0])
    Q = q_learning_updates(Q0, tr, alpha=0.5, gamma=0.9, phi=phi)
    # Step 1: r' = -1 + 0.9 * 2 - 1 = -0.2. Step 2: r' = 2 + 0 - 2 = 0, no bootstrap.
    close(Q, np.array([[0.0, -0.1], [0.0, 0.0], [0.0, 0.0]]), atol=1e-12, label="two shaped updates")


@check("shaping equals starting Q-learning at Q0 + Phi(s): the tables differ by Phi(s) after every update")
def test_wiewiora(sub):
    q_learning_updates = has(sub, "q_learning_updates")
    rng = np.random.default_rng(3)
    S, A, N = 6, 3, 400
    tr = {"s": rng.integers(0, 5, N), "a": rng.integers(0, A, N), "r": rng.normal(size=N),
          "s2": rng.integers(0, S, N)}
    tr["done"] = (tr["s2"] == 5).astype(float)
    phi = rng.normal(scale=4.0, size=S)
    phi[5] = 0.0  # the terminal state
    Q0 = rng.normal(size=(S, A))
    shaped = q_learning_updates(Q0, tr, alpha=0.3, gamma=0.9, phi=phi)
    plain = q_learning_updates(Q0 + phi[:, None], tr, alpha=0.3, gamma=0.9)
    close(shaped, plain - phi[:, None], atol=1e-9, label="shaped Q against (Q started at Q0 + Phi) - Phi")


@check("run_q_learning with Phi = -distance to the goal needs under half the steps over 30 episodes")
def test_shaping_speeds_learning(sub):
    run_q_learning = has(sub, "run_q_learning")
    optimal_actions = has(sub, "optimal_actions")
    mdp = _maze()
    phi = _distance_potential(mdp)
    _, plain = run_q_learning(mdp, 30, 0.5, 0.1, seed=0)
    Q, shaped = run_q_learning(mdp, 30, 0.5, 0.1, seed=0, phi=phi)
    shape(plain, (30,), label="steps per episode")
    if shaped.sum() > 0.5 * plain.sum():
        fail(f"shaped Q-learning took {shaped.sum()} steps against {plain.sum()} without shaping; "
             "expected under half")
    # Q + Phi(s) estimates the unshaped Q*. Shifting a row doesn't move its argmax.
    v = rlenvs.policy_evaluation(mdp, rlenvs.greedy(Q + phi[:, None]))
    v_star = rlenvs.value_iteration(mdp)
    start = mdp.names["start"]
    almost(v[start], v_star[start], tol=1e-9, label="true value of the shaped learner's greedy policy")
