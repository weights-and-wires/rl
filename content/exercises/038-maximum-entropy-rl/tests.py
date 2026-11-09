"""Checks for the maximum-entropy reinforcement learning exercise."""

import numpy as np

import rlenvs
from rlcheck import almost, check, close, fail, has, shape


def _two_state_mdp():
    """The lesson's worked example: stay or switch, and staying in state 1 pays 1."""
    P = np.zeros((2, 2, 2))
    P[0, 0, 0] = P[0, 1, 1] = P[1, 0, 1] = P[1, 1, 0] = 1.0
    R = np.array([[0.0, 0.0], [1.0, 0.0]])
    return rlenvs.TabularMDP(P=P, R=R, gamma=0.9, mu0=np.array([1.0, 0.0]))


def _entropy(p):
    return -np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0), axis=-1)


@check("soft_value is alpha * logsumexp(Q / alpha): 2.0635 for Q = (1, 2) and alpha = 0.5, without overflow")
def test_soft_value(sub):
    soft_value = has(sub, "soft_value")
    got = soft_value(np.array([[1.0, 2.0]]), 0.5)
    shape(got, (1,), label="soft_value of one row")
    close(got, [0.5 * np.log(np.exp(2.0) + np.exp(4.0))], atol=1e-12, label="the lesson's example")

    big = soft_value(np.array([[1000.0, 999.0, 998.0]]), 0.01)
    close(big, [1000.0 + 0.01 * np.log(1 + np.exp(-100) + np.exp(-200))], atol=1e-9,
          label="Q near 1000 at alpha = 0.01 (subtract the row maximum before exp)")

    q = np.random.default_rng(0).normal(size=(5, 4))
    close(soft_value(q, 1e-4), q.max(axis=1), atol=1e-3, label="alpha -> 0, which should give max Q")
    close(soft_value(q, 1e4), q.mean(axis=1) + 1e4 * np.log(4), atol=1e-3,
          label="alpha -> infinity, which should give mean Q + alpha log A")


@check("boltzmann_policy is exp((Q - V) / alpha), and it beats every other policy on Q + alpha * entropy")
def test_boltzmann_policy(sub):
    soft_value = has(sub, "soft_value")
    boltzmann_policy = has(sub, "boltzmann_policy")
    got = boltzmann_policy(np.array([[1.0, 2.0]]), 0.5)
    shape(got, (1, 2), label="boltzmann_policy of one row")
    close(got, [[1 / (1 + np.e**2), np.e**2 / (1 + np.e**2)]], atol=1e-12, label="the lesson's example")

    rng = np.random.default_rng(1)
    q = rng.normal(size=(3, 4)) * 2
    alpha = 0.7
    pi = boltzmann_policy(q, alpha)
    close(pi.sum(axis=1), np.ones(3), atol=1e-12, label="row sums")
    score = (pi * q).sum(axis=1) + alpha * _entropy(pi)
    close(score, soft_value(q, alpha), atol=1e-10,
          label="sum_a pi(a) Q(a) + alpha H(pi), which should equal the soft value")
    rivals = rng.dirichlet(np.ones(4), size=(3000, 3))
    rival_scores = (rivals * q).sum(axis=2) + alpha * _entropy(rivals)
    if np.any(rival_scores > score + 1e-12):
        fail("a random policy scored higher on Q + alpha * entropy than the Boltzmann policy")


@check("soft_bellman_backup matches the worked example and gives terminal states zero value")
def test_soft_bellman_backup(sub):
    backup = has(sub, "soft_bellman_backup")
    mdp = _two_state_mdp()
    q1 = backup(mdp, np.zeros((2, 2)), 1.0)
    shape(q1, (2, 2), label="soft_bellman_backup output")
    b = 0.9 * np.log(2.0)
    close(q1, [[b, b], [1 + b, b]], atol=1e-12, label="one backup from Q = 0 at alpha = 1")
    v0 = np.log(2 * np.exp(b))
    v1 = np.log(np.exp(1 + b) + np.exp(b))
    close(backup(mdp, q1, 1.0), [[0.9 * v0, 0.9 * v1], [1 + 0.9 * v1, 0.9 * v0]], atol=1e-12,
          label="a second backup")

    lake = rlenvs.frozen_lake()
    q = np.random.default_rng(2).normal(size=(lake.n_states, 4))
    got = backup(lake, q, 0.5)
    close(got[lake.terminal], np.zeros((lake.terminal.sum(), 4)), atol=1e-12,
          label="Q in the lake's terminal states")
    v = 0.5 * np.log(np.exp(q / 0.5).sum(axis=1))
    v[lake.terminal] = 0.0
    want = lake.R + lake.gamma * lake.P @ v
    want[lake.terminal] = 0.0
    close(got, want, atol=1e-10,
          label="the backup on the frozen lake, where a terminal state's soft value must be 0, not alpha log A")


@check("soft_value_iteration reaches a fixed point of the soft Bellman equation")
def test_soft_value_iteration(sub):
    soft_value = has(sub, "soft_value")
    svi = has(sub, "soft_value_iteration")
    q = svi(_two_state_mdp(), 1.0)
    shape(q, (2, 2), label="soft_value_iteration output")
    V = np.log(np.exp(q).sum(axis=1))
    close(q, [[0.9 * V[0], 0.9 * V[1]], [1 + 0.9 * V[1], 0.9 * V[0]]], atol=1e-9,
          label="the two-state fixed point at alpha = 1")
    close(soft_value(q, 1.0), [10.937683, 11.699795], atol=1e-5, label="V at alpha = 1")

    for mdp, name in [(rlenvs.random_mdp(5, 3, 0.9, seed=3), "a random MDP"), (rlenvs.frozen_lake(), "the frozen lake")]:
        for alpha in (1.0, 0.1):
            q = svi(mdp, alpha)
            v = np.where(mdp.terminal, 0.0, alpha * np.log(np.exp((q - q.max(1, keepdims=True)) / alpha).sum(1)) + q.max(1))
            target = rlenvs.q_from_v(mdp, v)
            gap = np.max(np.abs(q - target))
            if gap > 1e-9:
                fail(f"on {name} at alpha = {alpha}, Q differs from its soft backup by {gap:.3g}")


@check("soft_value_iteration recovers rlenvs.value_iteration as alpha -> 0, within alpha log A / (1 - gamma)")
def test_soft_to_hard(sub):
    soft_value = has(sub, "soft_value")
    svi = has(sub, "soft_value_iteration")
    for mdp, name in [(rlenvs.random_mdp(5, 3, 0.9, seed=4), "a random MDP"), (rlenvs.frozen_lake(), "the frozen lake")]:
        v_hard = rlenvs.value_iteration(mdp)
        A = mdp.n_actions
        for alpha in (1.0, 0.1, 0.01, 0.001):
            v_soft = np.where(mdp.terminal, 0.0, soft_value(svi(mdp, alpha), alpha))
            excess = v_soft - v_hard
            limit = alpha * np.log(A) / (1 - mdp.gamma)
            if excess.min() < -1e-9 or excess.max() > limit + 1e-9:
                fail(
                    f"on {name} at alpha = {alpha}, V_soft - V* ranges over [{excess.min():.3g}, "
                    f"{excess.max():.3g}], outside [0, {limit:.3g}]"
                )
        if excess.max() > 1e-3:
            fail(f"on {name} at alpha = 0.001, V_soft should be within 0.001 of V*, got {excess.max():.3g}")





@check("soft_bellman_residual averages to exactly zero over next states at Q*, and not at a wrong Q")
def test_soft_bellman_residual(sub):
    svi = has(sub, "soft_value_iteration")
    residual = has(sub, "soft_bellman_residual")
    lake = rlenvs.frozen_lake()
    alpha = 0.1
    q_star = svi(lake, alpha)
    S = lake.n_states
    s_all, a_all, s2_all = np.meshgrid(np.arange(S), np.arange(4), np.arange(S), indexing="ij")
    s_all, a_all, s2_all = s_all.ravel(), a_all.ravel(), s2_all.ravel()
    live = ~lake.terminal[s_all]
    s_all, a_all, s2_all = s_all[live], a_all[live], s2_all[live]
    r = lake.R3[s_all, a_all, s2_all]
    done = lake.terminal[s2_all].astype(float)
    weight = lake.P[s_all, a_all, s2_all]

    delta = residual(q_star, s_all, a_all, r, s2_all, done, lake.gamma, alpha)
    shape(delta, s_all.shape, label="soft_bellman_residual output")
    expected = np.zeros((S, 4))
    np.add.at(expected, (s_all, a_all), weight * delta)
    worst = np.max(np.abs(expected))
    if worst > 1e-9:
        fail(
            f"at Q* the residual, averaged over next states, should be 0 for every (s, a); the worst was "
            f"{worst:.3g}. Did you drop the (1 - terminated) mask on the next state's soft value?"
        )
    wrong = q_star + 0.1
    delta = residual(wrong, s_all, a_all, r, s2_all, done, lake.gamma, alpha)
    expected = np.zeros((S, 4))
    np.add.at(expected, (s_all, a_all), weight * delta)
    if np.max(np.abs(expected)) < 1e-3:
        fail("shifting Q* by 0.1 should leave a nonzero expected residual")
