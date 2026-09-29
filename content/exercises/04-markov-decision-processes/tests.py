"""Checks for the Markov decision processes exercise."""

import numpy as np

import rlenvs
from rlcheck import check, close, fail, has, shape

STAY, GO = 0, 1
ROOM = ["S..", ".#.", "..G"]


def _two_state(gamma=0.9):
    P = np.zeros((2, 2, 2))
    P[0, STAY, 0] = P[0, GO, 1] = P[1, STAY, 1] = P[1, GO, 0] = 1.0
    R = np.array([[1.0, 0.0], [2.0, 0.0]])
    return rlenvs.TabularMDP(P=P, R=R, gamma=gamma, mu0=np.array([1.0, 0.0]))


def _random_policy(mdp, seed):
    logits = np.random.default_rng(seed).normal(size=(mdp.n_states, mdp.n_actions))
    pi = np.exp(logits)
    return pi / pi.sum(axis=1, keepdims=True)


@check("policy_matrices averages P and R over the policy's actions")
def test_policy_matrices(sub):
    policy_matrices = has(sub, "policy_matrices")
    mdp = _two_state()
    P_pi, r_pi = policy_matrices(mdp, rlenvs.uniform_policy(mdp))
    shape(P_pi, (2, 2), label="P_pi")
    shape(r_pi, (2,), label="r_pi")
    close(P_pi, [[0.5, 0.5], [0.5, 0.5]], atol=1e-12, label="P_pi for the uniform policy")
    close(r_pi, [0.5, 1.0], atol=1e-12, label="r_pi for the uniform policy")

    mdp = rlenvs.random_mdp(n_states=6, n_actions=3, gamma=0.9, seed=0)
    pi = _random_policy(mdp, 0)
    P_pi, r_pi = policy_matrices(mdp, pi)
    close(P_pi.sum(axis=1), np.ones(6), atol=1e-12, label="row sums of P_pi")
    s, s2 = 2, 4
    close(P_pi[s, s2], sum(pi[s, a] * mdp.P[s, a, s2] for a in range(3)), atol=1e-12,
          label="P_pi[2, 4] = sum_a pi(a | 2) P(4 | 2, a)")

    mdp = rlenvs.gridworld(ROOM, gamma=0.9)
    P_pi, r_pi = policy_matrices(mdp, rlenvs.uniform_policy(mdp))
    goal = int(np.flatnonzero(mdp.terminal)[0])
    close(P_pi[goal], np.zeros(mdp.n_states), atol=0, label="the terminal state's row of P_pi")
    close(r_pi[goal], 0.0, atol=0, label="the terminal state's r_pi")


@check("evaluate_exact gives V = (7.25, 7.75) for the uniform policy on the two-state MDP")
def test_evaluate_worked(sub):
    evaluate_exact = has(sub, "evaluate_exact")
    mdp = _two_state()
    close(evaluate_exact(mdp, rlenvs.uniform_policy(mdp)), [7.25, 7.75], atol=1e-10, label="V for the uniform policy")
    close(evaluate_exact(mdp, np.array([[0.0, 1.0], [1.0, 0.0]])), [18.0, 20.0], atol=1e-10,
          label="V for go in state 0, stay in state 1")


@check("evaluate_exact solves the Bellman equation on random MDPs and a grid with a goal")
def test_evaluate_random(sub):
    evaluate_exact = has(sub, "evaluate_exact")
    policy_matrices = has(sub, "policy_matrices")
    for seed in range(3):
        mdp = rlenvs.random_mdp(n_states=6, n_actions=3, gamma=0.9, seed=seed)
        pi = _random_policy(mdp, seed)
        v = evaluate_exact(mdp, pi)
        shape(v, (6,), label="evaluate_exact output")
        close(v, rlenvs.policy_evaluation(mdp, pi), atol=1e-10, label=f"against rlenvs.policy_evaluation, seed {seed}")
        P_pi, r_pi = policy_matrices(mdp, pi)
        residual = np.max(np.abs(r_pi + mdp.gamma * P_pi @ v - v))
        if residual > 1e-10:
            fail(f"the Bellman residual max |r_pi + gamma P_pi V - V| should be below 1e-10, got {residual:.2e}")
    mdp = rlenvs.gridworld(ROOM, gamma=0.9)
    pi = rlenvs.uniform_policy(mdp)
    close(evaluate_exact(mdp, pi), rlenvs.policy_evaluation(mdp, pi), atol=1e-10,
          label="the 3 by 3 room, whose goal is terminal")


@check("q_from_v gives the lesson's action values, and averaging Q over pi recovers V")
def test_q_from_v(sub):
    q_from_v = has(sub, "q_from_v")
    evaluate_exact = has(sub, "evaluate_exact")
    mdp = _two_state()
    q = q_from_v(mdp, np.array([7.25, 7.75]))
    shape(q, (2, 2), label="q_from_v output")
    close(q, [[7.525, 6.975], [8.975, 6.525]], atol=1e-10, label="Q for the uniform policy")
    close(q_from_v(mdp, np.array([18.0, 20.0])), [[17.2, 18.0], [20.0, 16.2]], atol=1e-10, label="Q*")

    mdp = rlenvs.random_mdp(n_states=6, n_actions=3, gamma=0.9, seed=1)
    pi = _random_policy(mdp, 5)
    v = evaluate_exact(mdp, pi)
    close((pi * q_from_v(mdp, v)).sum(axis=1), v, atol=1e-10, label="sum_a pi(a | s) Q(s, a) against V(s)")

    mdp = rlenvs.gridworld(ROOM, gamma=0.9)
    v = np.arange(mdp.n_states, dtype=float)
    close(q_from_v(mdp, v), rlenvs.q_from_v(mdp, v), atol=1e-12, label="the room, with its terminal row set to zero")


@check("bellman_optimality_backup leaves V* = (18, 20) fixed and moves anything else")
def test_backup_fixed_point(sub):
    backup = has(sub, "bellman_optimality_backup")
    mdp = _two_state()
    close(backup(mdp, np.array([18.0, 20.0])), [18.0, 20.0], atol=1e-10, label="backup of V*")
    close(backup(mdp, np.array([0.0, 0.0])), [1.0, 2.0], atol=1e-12, label="backup of (0, 0)")
    close(backup(mdp, np.array([10.0, 20.0])), [18.0, 20.0], atol=1e-12, label="backup of (10, 20)")
    mdp = rlenvs.random_mdp(n_states=6, n_actions=3, gamma=0.9, seed=2)
    v_star = rlenvs.value_iteration(mdp)
    close(backup(mdp, v_star), v_star, atol=1e-9, label="backup of V* on a random MDP")


@check("bellman_optimality_backup is a gamma-contraction in the max norm")
def test_contraction(sub):
    backup = has(sub, "bellman_optimality_backup")
    mdp = rlenvs.random_mdp(n_states=6, n_actions=3, gamma=0.9, seed=0)
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(200):
        u, v = rng.normal(scale=10.0, size=(2, 6))
        ratio = np.max(np.abs(backup(mdp, u) - backup(mdp, v))) / np.max(np.abs(u - v))
        worst = max(worst, ratio)
    print(f"largest ratio over 200 random pairs: {worst:.4f} (gamma = 0.9)")
    if worst > 0.9 + 1e-12:
        fail(f"|T*U - T*V| / |U - V| reached {worst:.4f}, above gamma = 0.9")
    if worst < 0.3:
        fail(f"the largest ratio was only {worst:.4f}: the backup should depend on V through gamma * P @ V")


@check("discounted_visitation sums to one and satisfies d = (1 - gamma) mu0 + gamma d P_pi")
def test_visitation(sub):
    discounted_visitation = has(sub, "discounted_visitation")
    mdp = _two_state()
    close(discounted_visitation(mdp, rlenvs.uniform_policy(mdp)), [0.55, 0.45], atol=1e-12, label="uniform policy from state 0")
    close(discounted_visitation(mdp, np.array([[0.0, 1.0], [1.0, 0.0]])), [0.1, 0.9], atol=1e-12,
          label="the optimal policy from state 0")

    mdp = rlenvs.random_mdp(n_states=6, n_actions=3, gamma=0.9, seed=4)
    mdp.mu0 = np.random.default_rng(4).dirichlet(np.ones(6))
    pi = _random_policy(mdp, 4)
    d = discounted_visitation(mdp, pi)
    shape(d, (6,), label="discounted_visitation output")
    close(d.sum(), 1.0, atol=1e-12, label="sum of d")
    P_pi = np.einsum("sa,sat->st", pi, mdp.P)
    close(d, (1 - mdp.gamma) * mdp.mu0 + mdp.gamma * d @ P_pi, atol=1e-12, label="the flow equation")
    close(d, rlenvs.state_distribution(mdp, pi), atol=1e-12, label="against rlenvs.state_distribution")


@check("expected return from mu0 equals the visitation-weighted reward over (1 - gamma)")
def test_objective_identity(sub):
    discounted_visitation = has(sub, "discounted_visitation")
    evaluate_exact = has(sub, "evaluate_exact")
    mdp = rlenvs.random_mdp(n_states=6, n_actions=3, gamma=0.8, seed=5)
    pi = _random_policy(mdp, 6)
    d = discounted_visitation(mdp, pi)
    r_pi = (pi * mdp.R).sum(axis=1)
    close(mdp.mu0 @ evaluate_exact(mdp, pi), d @ r_pi / (1 - mdp.gamma), atol=1e-10,
          label="mu0 @ V against (d @ r_pi) / (1 - gamma)")
