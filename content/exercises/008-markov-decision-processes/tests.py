"""Checks for the Markov decision process exercise."""

import numpy as np

import rlenvs
from rlcheck import check, close, equal, fail, has, shape


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


UNIFORM_P = np.array([
    [0.5, 0.5, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0],
    [0.0, 0.5, 0.0, 0.5],
    [0.0, 0.0, 0.0, 0.0],
])


@check("sample_episode follows the always-right policy through the hallway: cells 0, 1, 2, 3 and return -2.71")
def test_always_right(sub):
    sample_episode = has(sub, "sample_episode")
    mdp = _hallway()
    always_right = np.tile([0.0, 1.0], (4, 1))
    states, actions, rewards = sample_episode(mdp, always_right, np.random.default_rng(0))
    equal(list(states), [0, 1, 2, 3], label="states visited")
    equal(list(actions), [1, 1, 1], label="actions taken")
    close(rewards, [-1.0, -1.0, -1.0], atol=1e-12, label="rewards")
    G = sum(0.9**t * r for t, r in enumerate(rewards))
    close(G, -2.71, atol=1e-12, label="discounted return -1 - 0.9 - 0.81")


@check("sample_episode returns valid uniform-policy episodes that start in cell 0 and end at the exit")
def test_valid_episodes(sub):
    sample_episode = has(sub, "sample_episode")
    mdp = _hallway()
    pi = rlenvs.uniform_policy(mdp)
    rng = np.random.default_rng(1)
    lengths = []
    for _ in range(200):
        states, actions, rewards = sample_episode(mdp, pi, rng)
        if len(states) != len(actions) + 1 or len(rewards) != len(actions):
            fail(f"states must have one more entry than actions and rewards; got {len(states)}, {len(actions)}, {len(rewards)}")
        if states[0] != 0:
            fail(f"the episode should start in cell 0, the support of mu0; it started in {states[0]}")
        if not mdp.terminal[states[-1]]:
            fail(f"the episode should end in the terminal cell 3; it ended in {states[-1]}")
        for t, (s, a, r, s2) in enumerate(zip(states[:-1], actions, rewards, states[1:])):
            if mdp.P[s, a, s2] <= 0:
                fail(f"step {t}: the model gives no chance of moving from cell {s} to {s2} with action {a}")
            close(r, mdp.R[s, a], atol=1e-12, label=f"reward at step {t}")
        lengths.append(len(actions))
    if len(set(lengths)) < 3:
        fail("a random policy should produce episodes of several different lengths")


@check("sample_episode stops after max_steps when the policy never reaches the exit")
def test_max_steps(sub):
    sample_episode = has(sub, "sample_episode")
    mdp = _hallway()
    always_left = np.tile([1.0, 0.0], (4, 1))
    states, actions, rewards = sample_episode(mdp, always_left, np.random.default_rng(0), max_steps=7)
    equal(len(actions), 7, label="number of steps taken")
    equal(len(states), 8, label="number of states recorded")
    equal(list(states), [0] * 8, label="always left bumps the wall forever")


@check("policy_matrices reproduces the lesson's P_pi and r_pi for the uniform policy")
def test_hallway_matrices(sub):
    policy_matrices = has(sub, "policy_matrices")
    mdp = _hallway()
    P_pi, r_pi = policy_matrices(mdp, rlenvs.uniform_policy(mdp))
    shape(P_pi, (4, 4), label="P_pi")
    shape(r_pi, (4,), label="r_pi")
    close(P_pi, UNIFORM_P, atol=1e-12, label="P_pi for the uniform policy")
    close(r_pi, [-1.0, -1.0, -1.0, 0.0], atol=1e-12, label="r_pi for the uniform policy")
    always_right = np.tile([0.0, 1.0], (4, 1))
    P_pi, _ = policy_matrices(mdp, always_right)
    close(P_pi, [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]], atol=1e-12, label="P_pi for always right")


@check("policy_matrices matches the definition, summed action by action, on a random MDP")
def test_definition(sub):
    policy_matrices = has(sub, "policy_matrices")
    mdp = rlenvs.random_mdp(n_states=5, n_actions=3, gamma=0.9, seed=2)
    pi = np.random.default_rng(0).dirichlet(np.ones(3), size=5)
    P_pi, r_pi = policy_matrices(mdp, pi)
    shape(P_pi, (5, 5), label="P_pi")
    want_P = np.zeros((5, 5))
    want_r = np.zeros(5)
    for s in range(5):
        for a in range(3):
            want_P[s] += pi[s, a] * mdp.P[s, a]
            want_r[s] += pi[s, a] * mdp.R[s, a]
    close(P_pi, want_P, atol=1e-12, label="P_pi[s, s2] = sum_a pi(a | s) P(s2 | s, a)")
    close(r_pi, want_r, atol=1e-12, label="r_pi[s] = sum_a pi(a | s) R(s, a)")
    close(P_pi.sum(axis=1), np.ones(5), atol=1e-12, label="row sums of P_pi")


@check("policy_matrices zeroes the rows of terminal states")
def test_terminal_rows(sub):
    policy_matrices = has(sub, "policy_matrices")
    mdp = rlenvs.frozen_lake()
    P_pi, r_pi = policy_matrices(mdp, rlenvs.uniform_policy(mdp))
    done = np.flatnonzero(mdp.terminal)
    if len(done) == 0:
        fail("the frozen lake should have terminal states")
    close(P_pi[done], np.zeros((len(done), mdp.n_states)), atol=0.0, label="rows of P_pi at terminal states")
    close(r_pi[done], np.zeros(len(done)), atol=0.0, label="r_pi at terminal states")
    live = np.flatnonzero(~mdp.terminal)
    close(P_pi[live].sum(axis=1), np.ones(len(live)), atol=1e-12, label="row sums at non-terminal states")


@check("the frequencies of sampled steps match the rows of P_pi")
def test_sampler_matches_matrix(sub):
    sample_episode = has(sub, "sample_episode")
    policy_matrices = has(sub, "policy_matrices")
    mdp = _hallway()
    pi = rlenvs.uniform_policy(mdp)
    P_pi, _ = policy_matrices(mdp, pi)
    rng = np.random.default_rng(0)
    counts = np.zeros((4, 4))
    for _ in range(600):
        states, _, _ = sample_episode(mdp, pi, rng)
        for s, s2 in zip(states[:-1], states[1:]):
            counts[s, s2] += 1
    for s in range(3):
        freq = counts[s] / counts[s].sum()
        if np.max(np.abs(freq - P_pi[s])) > 0.04:
            fail(f"from cell {s}, sampled next-cell frequencies {np.round(freq, 3).tolist()} differ from P_pi row {P_pi[s].tolist()}")
