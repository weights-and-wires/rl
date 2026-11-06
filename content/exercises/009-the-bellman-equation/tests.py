"""Checks for the Bellman equation exercise."""

import numpy as np

import rlenvs
from rlcheck import check, close, fail, has, shape


def _hallway(gamma=0.9):
    P = np.zeros((4, 2, 4))
    R = np.zeros((4, 2))
    for s in range(3):
        P[s, 0, max(s - 1, 0)] = 1.0
        P[s, 1, s + 1] = 1.0
        R[s, :] = -1.0
    P[3, :, 3] = 1.0
    return rlenvs.TabularMDP(
        P=P, R=R, gamma=gamma,
        mu0=np.array([1.0, 0.0, 0.0, 0.0]),
        terminal=np.array([False, False, False, True]),
    )


def _matrices(mdp, pi):
    live = ~mdp.terminal
    return np.einsum("sa,sat->st", pi, mdp.P) * live[:, None], (pi * mdp.R).sum(axis=1) * live


V_UNIFORM = np.array([-6.14081525, -5.28321863, -3.37744839, 0.0])


@check("evaluate_exact gives the always-right values (-2.71, -1.9, -1, 0) that the lesson works out backward")
def test_always_right(sub):
    evaluate_exact = has(sub, "evaluate_exact")
    mdp = _hallway()
    v = evaluate_exact(mdp, np.tile([0.0, 1.0], (4, 1)))
    shape(v, (4,), label="evaluate_exact output")
    close(v, [-2.71, -1.9, -1.0, 0.0], atol=1e-10, label="values of the always-right policy")


@check("evaluate_exact solves the lesson's three equations for the uniform policy")
def test_uniform_hallway(sub):
    evaluate_exact = has(sub, "evaluate_exact")
    mdp = _hallway()
    v = np.asarray(evaluate_exact(mdp, rlenvs.uniform_policy(mdp)), dtype=float)
    close(v, V_UNIFORM, atol=1e-6, label="values of the uniform policy")
    close(v[0], -1 + 0.45 * v[0] + 0.45 * v[1], atol=1e-10, label="V(0) = -1 + 0.45 V(0) + 0.45 V(1)")
    close(v[1], -1 + 0.45 * v[0] + 0.45 * v[2], atol=1e-10, label="V(1) = -1 + 0.45 V(0) + 0.45 V(2)")
    close(v[2], -1 + 0.45 * v[1], atol=1e-10, label="V(2) = -1 + 0.45 V(1)")


@check("evaluate_exact agrees with rlenvs.policy_evaluation on random MDPs and on the frozen lake")
def test_against_reference(sub):
    evaluate_exact = has(sub, "evaluate_exact")
    rng = np.random.default_rng(0)
    for seed in range(3):
        mdp = rlenvs.random_mdp(n_states=7, n_actions=3, gamma=0.9, seed=seed)
        pi = rng.dirichlet(np.ones(3), size=7)
        close(evaluate_exact(mdp, pi), rlenvs.policy_evaluation(mdp, pi), atol=1e-9, label=f"random MDP, seed {seed}")
    lake = rlenvs.frozen_lake()
    pi = rlenvs.uniform_policy(lake)
    close(evaluate_exact(lake, pi), rlenvs.policy_evaluation(lake, pi), atol=1e-9, label="frozen lake, uniform policy")


@check("the values satisfy the Bellman equation V = r_pi + gamma P_pi V, and terminal states stay at 0")
def test_bellman_residual(sub):
    evaluate_exact = has(sub, "evaluate_exact")
    lake = rlenvs.frozen_lake()
    pi = np.random.default_rng(1).dirichlet(np.ones(4), size=lake.n_states)
    v = np.asarray(evaluate_exact(lake, pi), dtype=float)
    P_pi, r_pi = _matrices(lake, pi)
    close(v, r_pi + lake.gamma * P_pi @ v, atol=1e-9, label="V against r_pi + gamma P_pi V on the frozen lake")
    close(v[lake.terminal], np.zeros(int(lake.terminal.sum())), atol=1e-12, label="values of terminal states")


@check("q_from_v reproduces the lesson's action-value table for the uniform policy")
def test_q_table(sub):
    q_from_v = has(sub, "q_from_v")
    mdp = _hallway()
    Q = np.asarray(q_from_v(mdp, V_UNIFORM), dtype=float)
    shape(Q, (4, 2), label="q_from_v output")
    want = np.array([
        [-1 + 0.9 * V_UNIFORM[0], -1 + 0.9 * V_UNIFORM[1]],
        [-1 + 0.9 * V_UNIFORM[0], -1 + 0.9 * V_UNIFORM[2]],
        [-1 + 0.9 * V_UNIFORM[1], -1.0],
        [0.0, 0.0],
    ])
    close(Q, want, atol=1e-6, label="Q for the uniform policy")
    close(Q[:3], [[-6.5267, -5.7549], [-6.5267, -4.0397], [-5.7549, -1.0]], atol=1e-4, label="the lesson's rounded table")


@check("averaging Q over the policy's own action choices gives back V")
def test_q_averages_to_v(sub):
    evaluate_exact = has(sub, "evaluate_exact")
    q_from_v = has(sub, "q_from_v")
    mdp = rlenvs.random_mdp(n_states=6, n_actions=3, gamma=0.9, seed=5)
    pi = np.random.default_rng(2).dirichlet(np.ones(3), size=6)
    v = evaluate_exact(mdp, pi)
    Q = np.asarray(q_from_v(mdp, v), dtype=float)
    close((pi * Q).sum(axis=1), v, atol=1e-9, label="sum_a pi(a | s) Q(s, a) against V(s)")

    lake = rlenvs.frozen_lake()
    Q = np.asarray(q_from_v(lake, rlenvs.policy_evaluation(lake, rlenvs.uniform_policy(lake))), dtype=float)
    done = lake.terminal
    close(Q[done], np.zeros((int(done.sum()), lake.n_actions)), atol=0.0, label="rows of Q at terminal states")


@check("the exact value agrees with the average return of simulated episodes")
def test_against_simulation(sub):
    evaluate_exact = has(sub, "evaluate_exact")
    mdp = _hallway()
    pi = rlenvs.uniform_policy(mdp)
    v = evaluate_exact(mdp, pi)
    rng = np.random.default_rng(0)
    returns = []
    for _ in range(3000):
        s, g, discount = 0, 0.0, 1.0
        while not mdp.terminal[s]:
            a = int(rng.choice(2, p=pi[s]))
            s, r = mdp.sample(s, a, rng)
            g += discount * r
            discount *= mdp.gamma
        returns.append(g)
    mean = float(np.mean(returns))
    print(f"average return over 3000 episodes: {mean:.3f}; exact value of cell 0: {v[0]:.3f}")
    if abs(mean - v[0]) > 0.25:
        fail(f"the average simulated return {mean:.3f} should be close to the exact value {v[0]:.3f}")
