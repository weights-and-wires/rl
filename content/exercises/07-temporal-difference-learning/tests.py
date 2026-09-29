"""Checks for the temporal-difference learning exercise."""

import numpy as np

import rlenvs
from rlcheck import almost, check, close, fail, has, shape


def _random_walk(n=5):
    S = n + 2
    P = np.zeros((S, 1, S))
    R3 = np.zeros((S, 1, S))
    for s in range(1, n + 1):
        P[s, 0, s - 1] = 0.5
        P[s, 0, s + 1] = 0.5
    P[0, 0, 0] = 1.0
    P[S - 1, 0, S - 1] = 1.0
    R3[n, 0, n + 1] = 1.0
    terminal = np.zeros(S, dtype=bool)
    terminal[[0, S - 1]] = True
    mu0 = np.zeros(S)
    mu0[(n + 1) // 2] = 1.0
    R = (P * R3).sum(axis=2)
    return rlenvs.TabularMDP(P=P, R=R, gamma=1.0, mu0=mu0, terminal=terminal, R3=R3)


def _walk_episodes(n_episodes, seed):
    mdp = _random_walk()
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n_episodes):
        s = 3
        states, rewards = [s], []
        while not mdp.terminal[s]:
            s, r = mdp.sample(s, 0, rng)
            states.append(s)
            rewards.append(r)
        data.append((np.array(states), np.array(rewards)))
    return data


def _mc_constant_alpha(mdp, n_episodes, alpha, seed):
    # Constant-step every-visit Monte Carlo, for comparison. On the walk the
    # only reward comes at the end and gamma = 1, so every visit sees the
    # episode's final reward as its return.
    rng = np.random.default_rng(seed)
    V = np.where(mdp.terminal, 0.0, 0.5)
    for _ in range(n_episodes):
        s = 3
        visited, G = [], 0.0
        while not mdp.terminal[s]:
            s2, r = mdp.sample(s, 0, rng)
            visited.append(s)
            G += r
            s = s2
        for x in visited:
            V[x] += alpha * (G - V[x])
    return V


# Sutton and Barto's Example 6.4: A = 0, B = 1, terminal = 2.
_AB_DATA = (
    [(np.array([0, 1, 2]), np.array([0.0, 0.0]))]
    + [(np.array([1, 2]), np.array([1.0])) for _ in range(6)]
    + [(np.array([1, 2]), np.array([0.0]))]
)


@check("td0_update moves V[s] a fraction alpha toward r + gamma * V[s2], and ignores V[s2] at the end")
def test_td0_update(sub):
    f = has(sub, "td0_update")
    V = np.array([0.0, 0.5, 1.0])
    # Target 1 + 0.9 * 1.0 = 1.9, error 1.4, step 0.14.
    got = f(V.copy(), s=1, r=1.0, s2=2, terminated=False, alpha=0.1, gamma=0.9)
    close(got, np.array([0.0, 0.64, 1.0]), atol=1e-12, label="a non-terminal step")
    # On termination the target is r alone: 0.5 + 0.1 * (1 - 0.5) = 0.55.
    got = f(V.copy(), s=1, r=1.0, s2=2, terminated=True, alpha=0.1, gamma=0.9)
    close(got, np.array([0.0, 0.55, 1.0]), atol=1e-12, label="a terminal step")


@check("certainty_equivalence gives V(A) = V(B) = 0.75 on the eight-episode dataset")
def test_certainty_equivalence(sub):
    f = has(sub, "certainty_equivalence")
    V = f(_AB_DATA, n_states=3, gamma=1.0)
    shape(V, (3,), label="certainty_equivalence output")
    close(V, np.array([0.75, 0.75, 0.0]), atol=1e-12, label="V for (A, B, terminal)")


@check("batch_td0 converges to the certainty-equivalence values, on Example 6.4 and on random-walk data")
def test_batch_td0(sub):
    f = has(sub, "batch_td0")
    ce = has(sub, "certainty_equivalence")
    V = f(_AB_DATA, n_states=3, gamma=1.0, alpha=0.05, n_sweeps=1000)
    close(V, np.array([0.75, 0.75, 0.0]), atol=1e-6, label="batch TD(0) on Example 6.4")
    data = _walk_episodes(10, seed=3)
    want = ce(data, n_states=7, gamma=1.0)
    got = f(data, n_states=7, gamma=1.0, alpha=0.02, n_sweeps=1000)
    close(got, want, atol=1e-6, label="batch TD(0) against certainty_equivalence on 10 walk episodes")


@check("td0_prediction on the random walk has lower RMS error than constant-step MC after 100 episodes")
def test_td_beats_mc(sub):
    f = has(sub, "td0_prediction")
    mdp = _random_walk()
    pi = np.ones((mdp.n_states, 1))
    truth = np.arange(1, 6) / 6
    td, mc = [], []
    for run in range(40):
        V = np.asarray(f(mdp, pi, n_episodes=100, alpha=0.05, seed=run), dtype=float)
        shape(V, (mdp.n_states,), label="td0_prediction output")
        td.append(np.sqrt(np.mean((V[1:6] - truth) ** 2)))
        W = _mc_constant_alpha(mdp, 100, 0.02, seed=run)
        mc.append(np.sqrt(np.mean((W[1:6] - truth) ** 2)))
    td, mc = float(np.mean(td)), float(np.mean(mc))
    if not (td < 0.055):
        fail(f"TD(0) with alpha = 0.05 should reach an RMS error near 0.036, below 0.055; got {td:.4f}")
    if not (td < mc):
        fail(f"TD(0)'s RMS error {td:.4f} should be below constant-step MC's {mc:.4f}")


@check("sarsa_update and q_learning_update give 0.5 and 2.0 on the worked example")
def test_control_updates(sub):
    sarsa_update = has(sub, "sarsa_update")
    q_update = has(sub, "q_learning_update")
    Q = np.array([[0.0, 0.0], [2.0, 5.0]])
    # SARSA's next action is action 0, worth 2: 0 + 0.5 * (-1 + 2 - 0) = 0.5.
    got = sarsa_update(Q.copy(), 0, 0, -1.0, 1, 0, False, 0.5, 1.0)
    almost(got[0, 0], 0.5, 1e-12, label="SARSA with next action 0")
    # Q-learning uses the max, 5, whatever action comes next: 0.5 * (-1 + 5) = 2.
    got = q_update(Q.copy(), 0, 0, -1.0, 1, False, 0.5, 1.0)
    almost(got[0, 0], 2.0, 1e-12, label="Q-learning")
    got = q_update(Q.copy(), 0, 0, -1.0, 1, True, 0.5, 1.0)
    almost(got[0, 0], -0.5, 1e-12, label="Q-learning on a terminal step")
    got = sarsa_update(Q.copy(), 0, 0, -1.0, 1, 1, True, 0.5, 1.0)
    almost(got[0, 0], -0.5, 1e-12, label="SARSA on a terminal step")


@check("epsilon_greedy_action explores with probability epsilon and breaks ties at random")
def test_epsilon_greedy_action(sub):
    f = has(sub, "epsilon_greedy_action")
    rng = np.random.default_rng(0)
    q_row = np.array([0.0, 1.0, 1.0, -2.0])
    n = 20000
    counts = np.bincount([f(q_row, 0.2, rng) for _ in range(n)], minlength=4) / n
    # Actions 1 and 2 tie, so each gets (1 - 0.2) / 2 + 0.2 / 4 = 0.45.
    want = np.array([0.05, 0.45, 0.45, 0.05])
    if np.max(np.abs(counts - want)) > 0.02:
        fail(f"expected action frequencies near {want}, got {np.round(counts, 3)}")


@check("Q-learning's greedy policy walks the cliff edge: its value from the start is exactly -13")
def test_q_learning_cliff(sub):
    f = has(sub, "q_learning")
    mdp = rlenvs.cliff_walking()
    Q, returns = f(mdp, n_episodes=500, alpha=0.1, epsilon=0.1, seed=0)
    shape(Q, (mdp.n_states, mdp.n_actions), label="Q")
    shape(returns, (500,), label="returns")
    v = rlenvs.policy_evaluation(mdp, rlenvs.greedy(Q))
    start = mdp.names["start"]
    # policy_evaluation returns nan for states that never terminate, so test
    # with isclose: a comparison with nan is always false.
    if not np.isclose(v[start], -13.0, atol=1e-9):
        fail(f"the greedy policy's value at the start should be -13, got {v[start]:.3f}")


@check("SARSA's greedy policy takes a safer, longer path: finite but below -13 (two of three seeds)")
def test_sarsa_cliff(sub):
    f = has(sub, "sarsa")
    mdp = rlenvs.cliff_walking()
    start = mdp.names["start"]
    values = []
    for seed in range(3):
        Q, _ = f(mdp, n_episodes=500, alpha=0.1, epsilon=0.1, seed=seed)
        values.append(rlenvs.policy_evaluation(mdp, rlenvs.greedy(Q))[start])
    values = np.array(values)
    # A nan fails both comparisons, so a looping greedy policy does not count.
    safe = (values < -13.5) & (values > -40.0)
    if safe.sum() < 2:
        fail(
            "expected SARSA's greedy value from the start to be finite and below -13, "
            f"around -15 or -17, in at least two of three runs; got {np.round(values, 2)}"
        )


@check("During training SARSA earns more than Q-learning, whose exploration falls off the edge")
def test_online_returns(sub):
    sarsa = has(sub, "sarsa")
    q_learning = has(sub, "q_learning")
    mdp = rlenvs.cliff_walking()
    _, rs = sarsa(mdp, n_episodes=500, alpha=0.1, epsilon=0.1, seed=0)
    _, rq = q_learning(mdp, n_episodes=500, alpha=0.1, epsilon=0.1, seed=0)
    s_mean, q_mean = float(np.mean(rs[-100:])), float(np.mean(rq[-100:]))
    if not (s_mean > q_mean + 5.0):
        fail(
            "over the last 100 training episodes, SARSA's mean return should beat "
            f"Q-learning's by more than 5; got SARSA {s_mean:.1f} and Q-learning {q_mean:.1f}"
        )
