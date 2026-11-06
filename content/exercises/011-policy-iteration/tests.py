"""Checks for the policy iteration exercise."""

import numpy as np

import rlenvs
from rlcheck import check, close, equal, fail, has, shape

STAY, GO = 0, 1
V_STAR = np.array([18.0, 20.0])


def _two_state(gamma=0.9):
    P = np.zeros((2, 2, 2))
    P[0, STAY, 0] = P[0, GO, 1] = P[1, STAY, 1] = P[1, GO, 0] = 1.0
    R = np.array([[1.0, 0.0], [2.0, 0.0]])
    return rlenvs.TabularMDP(P=P, R=R, gamma=gamma, mu0=np.array([1.0, 0.0]))


def _big(seed=0):
    return rlenvs.random_mdp(n_states=20, n_actions=4, gamma=0.95, seed=seed)


@check("greedy_policy picks go in state 0 and stay in state 1 from V* = (18, 20)")
def test_greedy_policy(sub):
    greedy_policy = has(sub, "greedy_policy")
    pi = greedy_policy(_two_state(), V_STAR)
    shape(pi, (2, 2), label="greedy_policy output")
    equal(np.asarray(pi, dtype=float), np.array([[0.0, 1.0], [1.0, 0.0]]), label="greedy policy from V* = (18, 20)")


@check("greedy_policy breaks ties by the first action")
def test_greedy_ties(sub):
    greedy_policy = has(sub, "greedy_policy")
    base = rlenvs.random_mdp(n_states=5, n_actions=2, gamma=0.9, seed=3)
    P = base.P.copy()
    R = base.R.copy()
    P[:, 1], R[:, 1] = P[:, 0], R[:, 0]  # two identical actions
    twin = rlenvs.TabularMDP(P=P, R=R, gamma=0.9, mu0=base.mu0)
    pi = np.asarray(greedy_policy(twin, np.random.default_rng(0).normal(size=5)), dtype=float)
    equal(pi, np.tile([1.0, 0.0], (5, 1)), label="identical actions: pick action 0 everywhere")


@check("improve_policy turns 'stay everywhere' (value (10, 20)) into 'go, then stay'")
def test_improve_policy(sub):
    improve_policy = has(sub, "improve_policy")
    stay = np.array([[1.0, 0.0], [1.0, 0.0]])
    new_pi, v = improve_policy(_two_state(), stay)
    # Staying forever pays 1 / 0.1 = 10 in state 0 and 2 / 0.1 = 20 in state 1.
    close(v, [10.0, 20.0], atol=1e-10, label="value of the policy you started with")
    equal(np.asarray(new_pi, dtype=float), np.array([[0.0, 1.0], [1.0, 0.0]]), label="the improved policy")


@check("one round of improvement never lowers any state's value, on ten random MDPs")
def test_improvement_theorem(sub):
    improve_policy = has(sub, "improve_policy")
    for seed in range(10):
        mdp = _big(seed)
        pi = rlenvs.uniform_policy(mdp)
        for _ in range(3):
            new_pi, v = improve_policy(mdp, pi)
            v_new = rlenvs.policy_evaluation(mdp, new_pi)
            worst = float(np.min(v_new - v))
            if worst < -1e-9:
                fail(f"seed {seed}: the greedy policy is worse than its parent in some state, by {-worst:.3g}")
            pi = new_pi


@check("policy_iteration evaluates (10, 20), then (18, 20), and stops")
def test_policy_iteration_worked(sub):
    policy_iteration = has(sub, "policy_iteration")
    pi, v, values = policy_iteration(_two_state())
    close(np.asarray(values, dtype=float), [[10.0, 20.0], [18.0, 20.0]], atol=1e-10,
          label="values of each evaluated policy, from 'stay everywhere'")
    equal(np.asarray(pi, dtype=float), np.array([[0.0, 1.0], [1.0, 0.0]]), label="the final policy")
    close(v, V_STAR, atol=1e-10, label="the final value")


@check("policy_iteration from 'go everywhere' takes three evaluations: (0, 0), (10, 20), (18, 20)")
def test_policy_iteration_other_start(sub):
    policy_iteration = has(sub, "policy_iteration")
    go = np.array([[0.0, 1.0], [0.0, 1.0]])
    pi, v, values = policy_iteration(_two_state(), go)
    close(np.asarray(values, dtype=float), [[0.0, 0.0], [10.0, 20.0], [18.0, 20.0]], atol=1e-10,
          label="values of each evaluated policy, from 'go everywhere'")


@check("policy_iteration finds V* on a 20-state MDP, and never lowers any state's value")
def test_policy_iteration_random(sub):
    policy_iteration = has(sub, "policy_iteration")
    greedy_policy = has(sub, "greedy_policy")
    mdp = _big()
    v_star = rlenvs.value_iteration(mdp)
    pi, v, values = policy_iteration(mdp)
    values = np.asarray(values, dtype=float)
    close(v, v_star, atol=1e-8, label="policy iteration's value against V*")
    equal(np.asarray(pi, dtype=float), np.asarray(greedy_policy(mdp, v_star), dtype=float), label="policy against greedy(V*)")
    drops = np.diff(values, axis=0)
    if drops.size and drops.min() < -1e-10:
        fail(f"policy improvement lowered some state's value by {-drops.min():.3g}")
    print(f"policy iteration used {len(values)} evaluations")
