"""Checks for the Monte Carlo control exercise."""

import numpy as np

import rlenvs
from rlcheck import check, close, fail, has, shape

GRID = ["....", ".#..", "S..G"]


def _bandit_mdp():
    # State 0 has two actions and both end the episode in state 1.
    # Action 0 pays 1 and action 1 pays -1.
    P = np.zeros((2, 2, 2))
    P[:, :, 1] = 1.0
    R = np.array([[1.0, -1.0], [0.0, 0.0]])
    return rlenvs.TabularMDP(
        P=P, R=R, gamma=1.0, mu0=np.array([1.0, 0.0]), terminal=np.array([False, True])
    )


@check("epsilon_soft puts 1 - epsilon + epsilon / A on the first best action and epsilon / A elsewhere")
def test_epsilon_soft(sub):
    f = has(sub, "epsilon_soft")
    Q = np.array([[0.5, 2.0, 1.0, -1.0], [3.0, 1.0, 3.0, 0.0]])
    pi = f(Q, 0.2)
    shape(pi, (2, 4), label="epsilon_soft output")
    # The second row ties actions 0 and 2; the first maximiser wins, as in rlenvs.greedy.
    want = np.array([[0.05, 0.85, 0.05, 0.05], [0.85, 0.05, 0.05, 0.05]])
    close(pi, want, atol=1e-12, label="epsilon_soft with epsilon = 0.2")


@check("epsilon_soft with epsilon = 0 is greedy, with epsilon = 1 is uniform, and every row sums to one")
def test_epsilon_soft_limits(sub):
    f = has(sub, "epsilon_soft")
    Q = np.random.default_rng(0).normal(size=(6, 3))
    close(f(Q, 0.0), rlenvs.greedy(Q), atol=1e-12, label="epsilon_soft with epsilon = 0")
    close(f(Q, 1.0), np.full((6, 3), 1 / 3), atol=1e-12, label="epsilon_soft with epsilon = 1")
    pi = np.asarray(f(Q, 0.3), dtype=float)
    close(pi.sum(axis=1), np.ones(6), atol=1e-12, label="row sums")
    if pi.min() < 0.3 / 3 - 1e-12:
        fail(f"every action should get at least epsilon / A = 0.1, but the smallest probability is {pi.min():.3g}")


@check("epsilon_soft_world mixes each action with the average over actions, and rows stay probabilities")
def test_soft_world(sub):
    f = has(sub, "epsilon_soft_world")
    grid = rlenvs.gridworld(GRID)
    soft = f(grid, 0.2)
    shape(soft.P, grid.P.shape, label="P of the soft world")
    close(soft.P.sum(axis=2), np.ones(grid.P.shape[:2]), atol=1e-12, label="rows of P")
    want_R = 0.8 * grid.R + 0.2 * grid.R.mean(axis=1, keepdims=True)
    close(soft.R, want_R, atol=1e-12, label="R of the soft world")
    close(f(grid, 0.0).P, grid.P, atol=1e-12, label="epsilon = 0 leaves P unchanged")


@check("the best epsilon-soft policy on the grid is worth -3.818, and it is the greedy policy of the soft world")
def test_best_soft_value(sub):
    f = has(sub, "epsilon_soft_world")
    soft_fn = has(sub, "epsilon_soft")
    grid = rlenvs.gridworld(GRID)
    start = grid.names["start"]
    epsilon = 0.2
    soft = f(grid, epsilon)
    v_soft = rlenvs.value_iteration(soft)
    if abs(v_soft[start] - (-3.818)) > 5e-4:
        fail(f"the optimal value of the soft world from the start should be about -3.818, got {v_soft[start]:.4f}")
    if not (v_soft[start] < rlenvs.value_iteration(grid)[start] - 0.5):
        fail("exploring should cost something: the soft world's value should sit below the optimal value of -3")
    # Playing epsilon-greedy on Q* of the soft world, in the real MDP, has the soft world's optimal value.
    q_soft = rlenvs.q_from_v(soft, v_soft)
    v_real = rlenvs.policy_evaluation(grid, soft_fn(q_soft, epsilon))
    close(v_real, v_soft, atol=1e-8, label="epsilon-soft play in the real grid against greedy play in the soft world")


@check("mc_control learns Q = (1, -1) on a two-armed one-step problem with alpha = 1")
def test_mc_control_bandit(sub):
    f = has(sub, "mc_control")
    mdp = _bandit_mdp()
    Q = np.asarray(f(mdp, n_episodes=200, epsilon=0.5, alpha=1.0, seed=0), dtype=float)
    shape(Q, (2, 2), label="mc_control output")
    # alpha = 1 sets Q[s, a] to the latest return, so both arms end at their reward.
    close(Q, np.array([[1.0, -1.0], [0.0, 0.0]]), atol=1e-12, label="Q after 200 episodes")


@check("mc_control reaches the best epsilon-soft value on a small grid (median of five seeds)")
def test_mc_control(sub):
    f = has(sub, "mc_control")
    soft = has(sub, "epsilon_soft")
    grid = rlenvs.gridworld(GRID)
    start = grid.names["start"]
    epsilon = 0.2
    # The modified world of the chapter: each chosen action is replaced, with probability epsilon, by a random one.
    P = (1 - epsilon) * grid.P + epsilon * grid.P.mean(axis=1, keepdims=True)
    R = (1 - epsilon) * grid.R + epsilon * grid.R.mean(axis=1, keepdims=True)
    modified = rlenvs.TabularMDP(P=P, R=R, gamma=grid.gamma, mu0=grid.mu0, terminal=grid.terminal)
    best = rlenvs.value_iteration(modified)[start]
    got = []
    for seed in range(5):
        Q = np.asarray(f(grid, n_episodes=1000, epsilon=epsilon, alpha=0.1, seed=seed), dtype=float)
        shape(Q, (grid.n_states, grid.n_actions), label="mc_control output")
        v = rlenvs.policy_evaluation(grid, soft(Q, epsilon))
        got.append(v[start])
    median = float(np.median(got))
    # nan fails this comparison too, so a policy that never ends is caught.
    if not (median >= best - 0.5):
        fail(
            f"the best epsilon-soft policy is worth {best:.3f} from the start. Your "
            f"five runs gave {np.round(got, 3)}, median {median:.3f}, more than 0.5 below it."
        )
