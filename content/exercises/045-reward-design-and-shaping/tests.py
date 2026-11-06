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

