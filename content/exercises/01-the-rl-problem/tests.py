"""Checks for the agents, rewards, and returns exercise."""

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


@check("discounted_returns gives G = (2.62, 1.8, 2) for rewards (1, 0, 2) and gamma 0.9")
def test_returns_worked_example(sub):
    discounted_returns = has(sub, "discounted_returns")
    got = discounted_returns(np.array([1.0, 0.0, 2.0]), 0.9)
    shape(got, (3,), label="discounted_returns output")
    close(got, [2.62, 1.8, 2.0], atol=1e-12, label="returns for rewards (1, 0, 2)")


@check("discounted_returns matches the direct sum of gamma^k r_(t+k) at every t")
def test_returns_direct_sum(sub):
    discounted_returns = has(sub, "discounted_returns")
    rewards = np.random.default_rng(0).normal(size=40)
    for gamma in (0.0, 0.5, 0.97, 1.0):
        want = np.array([sum(gamma**k * rewards[t + k] for k in range(40 - t)) for t in range(40)])
        close(discounted_returns(rewards, gamma), want, atol=1e-10, label=f"gamma = {gamma}")


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


@check("run_episode is reproducible from the generator, and stops at the time limit")
def test_run_episode_seed_and_truncation(sub):
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


@check("mc_value_estimate is exact, with zero standard error, when nothing is random")
def test_mc_deterministic(sub):
    mc_value_estimate = has(sub, "mc_value_estimate")
    mdp = _room()
    estimate, se = mc_value_estimate(mdp, _shortest_path_policy(mdp), 20, seed=0)
    # Four steps of -1: -(1 + 0.9 + 0.81 + 0.729).
    close(estimate, -3.439, atol=1e-12, label="value of the four-step path")
    close(se, 0.0, atol=1e-12, label="standard error with identical returns")


@check("mc_value_estimate lands within four standard errors of the exact value")
def test_mc_random_policy(sub):
    mc_value_estimate = has(sub, "mc_value_estimate")
    mdp = _room()
    pi = rlenvs.uniform_policy(mdp)
    exact = rlenvs.policy_evaluation(mdp, pi)[mdp.names["start"]]
    estimate, se = mc_value_estimate(mdp, pi, 1000, seed=0)
    print(f"estimate {estimate:.4f} +- {se:.4f}, exact {exact:.4f}")
    if not 0.0 < se < 0.2:
        fail(f"the standard error over 1000 episodes should be small but positive, got {se:.4f}")
    if abs(estimate - exact) > 4 * se:
        fail(
            f"the random policy's exact value at the start is {exact:.4f}; the estimate "
            f"{estimate:.4f} is {abs(estimate - exact) / se:.1f} standard errors away"
        )


@check("mc_value_estimate's standard error halves when the episode count quadruples")
def test_mc_standard_error(sub):
    mc_value_estimate = has(sub, "mc_value_estimate")
    mdp = _room()
    pi = rlenvs.uniform_policy(mdp)
    _, se_small = mc_value_estimate(mdp, pi, 250, seed=100)
    _, se_large = mc_value_estimate(mdp, pi, 1000, seed=0)
    ratio = se_small / se_large
    if not 1.6 < ratio < 2.5:
        fail(
            "the standard error is the sample standard deviation over sqrt(n), so 250 episodes "
            f"should have about twice the error of 1000. The ratio was {ratio:.2f}"
        )
