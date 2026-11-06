"""Checks for the returns and discounting exercise."""

import numpy as np

import rlenvs
from rlcheck import check, close, fail, has, shape

ROOM = [
    "S..",
    ".#.",
    "..G",
]
RIGHT, DOWN = 1, 2


def _room():
    return rlenvs.gridworld(ROOM, gamma=0.9)


def _shortest_path_policy(mdp):
    pi = np.zeros((mdp.n_states, 4))
    for s, (i, j) in enumerate(mdp.names["cells"]):
        if i == 2 or (i == 0 and j < 2):
            pi[s, RIGHT] = 1.0
        else:
            pi[s, DOWN] = 1.0
    return pi


@check("discount_weights gives 1, gamma, gamma^2, ... and a weight that vanishes for a distant reward")
def test_weights(sub):
    discount_weights = has(sub, "discount_weights")
    got = np.asarray(discount_weights(0.9, 4), dtype=float)
    shape(got, (4,), label="discount_weights(0.9, 4)")
    close(got, [1.0, 0.9, 0.81, 0.729], atol=1e-12, label="weights for gamma 0.9")
    close(discount_weights(1.0, 3), [1.0, 1.0, 1.0], atol=1e-12, label="gamma 1 weights every reward the same")
    close(discount_weights(0.0, 3), [1.0, 0.0, 0.0], atol=1e-12, label="gamma 0 weights only the reward now")
    if discount_weights(0.9, 51)[-1] > 0.006:
        fail("0.9 ** 50 is about 0.005, so the weight on a reward 50 steps ahead should be small")


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


@check("discounted_returns satisfies G_t = r_t + gamma G_(t+1) and ends with G_(T-1) = r_(T-1)")
def test_returns_recursion(sub):
    discounted_returns = has(sub, "discounted_returns")
    rewards = np.random.default_rng(1).normal(size=25)
    g = np.asarray(discounted_returns(rewards, 0.8), dtype=float)
    close(g[-1], rewards[-1], atol=1e-12, label="the last return is the last reward")
    close(g[:-1], rewards[:-1] + 0.8 * g[1:], atol=1e-10, label="G_t against r_t + 0.8 G_(t+1)")


@check("discounted_returns on a constant reward of 1 approaches 1 / (1 - gamma)")
def test_returns_geometric(sub):
    discounted_returns = has(sub, "discounted_returns")
    g = np.asarray(discounted_returns(np.ones(300), 0.9), dtype=float)
    close(g[0], 10.0, atol=1e-6, label="G_0 for 300 rewards of 1 at gamma 0.9 (about 1 / (1 - 0.9))")
    close(g[-1], 1.0, atol=1e-12, label="the last return is one reward")


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
