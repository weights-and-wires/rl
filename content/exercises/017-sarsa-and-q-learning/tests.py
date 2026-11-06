"""Checks for the SARSA and Q-learning exercise."""

import numpy as np

import rlenvs
from rlcheck import almost, check, fail, has, shape


@check("sarsa_update uses the next action's value: 0.5 on the worked example, and the reward alone at the end")
def test_sarsa_update(sub):
    f = has(sub, "sarsa_update")
    Q = np.array([[0.0, 0.0], [2.0, 5.0]])
    # The next action is action 0, worth 2: 0 + 0.5 * (-1 + 2 - 0) = 0.5.
    got = f(Q.copy(), 0, 0, -1.0, 1, 0, False, 0.5, 1.0)
    almost(got[0, 0], 0.5, 1e-12, label="SARSA with next action 0")
    got = f(Q.copy(), 0, 0, -1.0, 1, 1, False, 0.5, 1.0)
    almost(got[0, 0], 2.0, 1e-12, label="SARSA with next action 1")
    got = f(Q.copy(), 0, 0, -1.0, 1, 1, True, 0.5, 1.0)
    almost(got[0, 0], -0.5, 1e-12, label="SARSA on a terminal step")


@check("q_learning_update uses the max over next actions: 2.0 on the worked example, whatever comes next")
def test_q_learning_update(sub):
    f = has(sub, "q_learning_update")
    Q = np.array([[0.0, 0.0], [2.0, 5.0]])
    # The max is 5, so 0.5 * (-1 + 5) = 2.
    got = f(Q.copy(), 0, 0, -1.0, 1, False, 0.5, 1.0)
    almost(got[0, 0], 2.0, 1e-12, label="Q-learning")
    got = f(Q.copy(), 0, 0, -1.0, 1, True, 0.5, 1.0)
    almost(got[0, 0], -0.5, 1e-12, label="Q-learning on a terminal step")
    got = f(Q.copy(), 0, 0, -1.0, 1, False, 0.5, 0.9)
    almost(got[0, 0], 0.5 * (-1.0 + 0.9 * 5.0), 1e-12, label="Q-learning with gamma = 0.9")


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
