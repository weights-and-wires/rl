"""Checks for the agent-environment loop exercise."""

import numpy as np

import rlenvs
from rlcheck import check, close, equal, fail, has, shape

ROOM = [
    "S..",
    ".#.",
    "..G",
]
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3


def _room(**kwargs):
    return rlenvs.gridworld(ROOM, gamma=0.9, **kwargs)


def _shortest_path_policy(mdp):
    """Right along the top row, then down the right column. Elsewhere, head for G too."""
    pi = np.zeros((mdp.n_states, 4))
    for s, (i, j) in enumerate(mdp.names["cells"]):
        if i == 2 or (i == 0 and j < 2):
            pi[s, RIGHT] = 1.0
        else:
            pi[s, DOWN] = 1.0
    return pi


@check("uniform_random_policy gives every action probability 1 / n_actions in every state")
def test_uniform_policy(sub):
    uniform_random_policy = has(sub, "uniform_random_policy")
    pi = np.asarray(uniform_random_policy(8, 4), dtype=float)
    shape(pi, (8, 4), label="uniform_random_policy(8, 4)")
    close(pi, np.full((8, 4), 0.25), atol=1e-12, label="each entry should be 1/4")
    close(np.asarray(uniform_random_policy(3, 5)).sum(axis=1), np.ones(3), atol=1e-12, label="row sums")


@check("run_episode walks the shortest path exactly under a deterministic policy")
def test_run_episode_deterministic(sub):
    run_episode = has(sub, "run_episode")
    mdp = _room()
    out = run_episode(mdp, _shortest_path_policy(mdp), np.random.default_rng(0))
    if not isinstance(out, tuple) or len(out) != 4:
        fail("run_episode must return a tuple (states, actions, rewards, terminated)")
    states, actions, rewards, terminated = out
    equal(np.asarray(states), np.array([0, 1, 2, 4]), label="states visited before each action")
    equal(np.asarray(actions), np.array([RIGHT, RIGHT, DOWN, DOWN]), label="actions")
    close(rewards, [-1.0, -1.0, -1.0, -1.0], atol=0, label="rewards")
    if terminated is not True:
        fail(f"the episode ends in the goal, so terminated must be True, got {terminated!r}")


@check("run_episode is reproducible from the generator, and differs between seeds")
def test_run_episode_seed(sub):
    run_episode = has(sub, "run_episode")
    mdp = _room()
    pi = rlenvs.uniform_policy(mdp)
    a = run_episode(mdp, pi, np.random.default_rng(7))
    b = run_episode(mdp, pi, np.random.default_rng(7))
    for name, x, y in zip(["states", "actions", "rewards"], a[:3], b[:3]):
        equal(np.asarray(x), np.asarray(y), label=f"{name} from two generators seeded 7")
    c = run_episode(mdp, pi, np.random.default_rng(8))
    if len(c[1]) == len(a[1]) and np.array_equal(c[1], a[1]):
        fail("generators seeded 7 and 8 gave the same random episode: draw actions from rng")


@check("run_episode stops at the time limit and reports terminated=False")
def test_run_episode_truncation(sub):
    run_episode = has(sub, "run_episode")
    mdp = _room(max_steps=5)
    pi = np.zeros((mdp.n_states, 4))
    pi[:, UP] = 1.0  # walk into the top wall forever
    states, actions, rewards, terminated = run_episode(mdp, pi, np.random.default_rng(0))
    if len(rewards) != 5 or terminated:
        fail(
            "with max_steps=5 and a policy that never reaches the goal, the episode must stop "
            f"after 5 steps with terminated=False; got {len(rewards)} steps, terminated={terminated}"
        )


@check("run_episode draws actions with the policy's probabilities")
def test_run_episode_frequencies(sub):
    run_episode = has(sub, "run_episode")
    mdp = _room()
    pi = _shortest_path_policy(mdp)
    pi[0] = [0.1, 0.2, 0.3, 0.4]  # at the start: up and left bump a wall and stay
    rng = np.random.default_rng(1)
    taken = []
    for _ in range(3000):
        states, actions, _, _ = run_episode(mdp, pi, rng)
        taken.extend(np.asarray(actions)[np.asarray(states) == 0])
    freq = np.bincount(taken, minlength=4) / len(taken)
    se = np.sqrt(pi[0] * (1 - pi[0]) / len(taken))
    if np.any(np.abs(freq - pi[0]) > 4 * se):
        fail(f"at the start, actions should follow {pi[0]}, but their frequencies were {np.round(freq, 3)}")


@check("visit_counts counts every state, including the ones never visited")
def test_visit_counts(sub):
    visit_counts = has(sub, "visit_counts")
    got = np.asarray(visit_counts(np.array([0, 1, 1, 1, 0, 0, 3, 5]), 8))
    shape(got, (8,), label="visit_counts output")
    equal(got, np.array([3, 3, 0, 1, 0, 1, 0, 0]), label="counts for the states 0, 1, 1, 1, 0, 0, 3, 5")


@check("mean_episode_length is 4 on the shortest path and about 32 for the random agent")
def test_mean_episode_length(sub):
    mean_episode_length = has(sub, "mean_episode_length")
    mdp = _room()
    got = mean_episode_length(mdp, _shortest_path_policy(mdp), 20, seed=0)
    close(got, 4.0, atol=1e-12, label="mean length of the four-step route")
    got = mean_episode_length(mdp, rlenvs.uniform_policy(mdp), 2000, seed=0)
    print(f"random agent's mean episode length over 2000 episodes: {got:.2f} (exact 32)")
    if abs(got - 32.0) > 3.0:
        fail(f"the random agent takes 32 steps on average to leave the room, but your mean was {got:.2f}")
