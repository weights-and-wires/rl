"""Checks for the off-policy importance sampling exercise."""

import numpy as np

import rlenvs
from rlcheck import almost, check, close, fail, has, shape


def _one_step_problem():
    # State 0 has two actions and both end the episode in state 1.
    # Action 0 ("L") pays 1 and action 1 ("R") pays 0.
    P = np.zeros((2, 2, 2))
    P[:, :, 1] = 1.0
    R = np.array([[1.0, 0.0], [0.0, 0.0]])
    return rlenvs.TabularMDP(
        P=P, R=R, gamma=1.0, mu0=np.array([1.0, 0.0]), terminal=np.array([False, True])
    )


@check("importance_ratios returns the tail products rho_{t:T-1} of pi / b")
def test_importance_ratios(sub):
    f = has(sub, "importance_ratios")
    pi = np.array([[0.9, 0.1], [0.2, 0.8]])
    b = np.array([[0.5, 0.5], [0.5, 0.5]])
    states = np.array([0, 1, 0])
    actions = np.array([0, 1, 1])
    # Step ratios 1.8, 1.6, 0.2, so the tails are 0.576, 0.32, 0.2.
    got = f(states, actions, pi, b)
    shape(got, (3,), label="importance_ratios output")
    close(got, np.array([0.576, 0.32, 0.2]), atol=1e-12, label="rho_{t:T-1}")


@check("importance_ratios is all ones when the target policy is the behaviour policy")
def test_ratios_same_policy(sub):
    f = has(sub, "importance_ratios")
    b = np.array([[0.3, 0.7], [0.6, 0.4]])
    got = f(np.array([0, 1, 1, 0]), np.array([1, 0, 1, 0]), b, b)
    close(got, np.ones(4), atol=1e-12, label="rho when pi = b")


@check("importance_ratios is zero from every step up to the last action that the target policy never takes")
def test_ratios_zero(sub):
    f = has(sub, "importance_ratios")
    pi = np.array([[1.0, 0.0], [0.5, 0.5]])
    b = np.array([[0.5, 0.5], [0.5, 0.5]])
    # The target never takes action 1 in state 0, so step 1 has ratio 0.
    got = f(np.array([1, 0, 1]), np.array([0, 1, 1]), pi, b)
    close(got, np.array([0.0, 0.0, 1.0]), atol=1e-12, label="rho_{t:T-1} with a zero step ratio in the middle")


@check("importance_sampling_estimates gives OIS 1.2 and WIS 0.947 on the three-episode example")
def test_is_worked_example(sub):
    f = has(sub, "importance_sampling_estimates")
    pi = np.array([[0.9, 0.1], [0.5, 0.5]])
    b = np.array([[0.5, 0.5], [0.5, 0.5]])
    L = (np.array([0]), np.array([0]), np.array([1.0]))
    R = (np.array([0]), np.array([1]), np.array([0.0]))
    ois, wis = f([L, R, L], pi, b, 1.0)
    almost(ois, 1.2, 1e-12, label="ordinary importance sampling")
    almost(wis, 3.6 / 3.8, 1e-12, label="weighted importance sampling")


@check("With one episode, WIS returns that episode's return, and with every weight zero it returns 0.0")
def test_wis_edge_cases(sub):
    f = has(sub, "importance_sampling_estimates")
    pi = np.array([[0.9, 0.1], [0.5, 0.5]])
    b = np.array([[0.5, 0.5], [0.5, 0.5]])
    L = (np.array([0]), np.array([0]), np.array([1.0]))
    _, wis = f([L], pi, b, 1.0)
    almost(wis, 1.0, 1e-12, label="WIS from one episode")
    never = np.array([[0.0, 1.0], [0.5, 0.5]])
    ois, wis = f([L, L], never, b, 1.0)
    almost(ois, 0.0, 1e-12, label="OIS when every weight is zero")
    almost(wis, 0.0, 1e-12, label="WIS when every weight is zero")


@check("importance_sampling_estimates discounts and weights multi-step episodes")
def test_multi_step(sub):
    f = has(sub, "importance_sampling_estimates")
    pi = np.array([[0.9, 0.1], [0.2, 0.8]])
    b = np.array([[0.5, 0.5], [0.5, 0.5]])
    e1 = (np.array([0, 1]), np.array([0, 1]), np.array([1.0, 2.0]))  # G_0 = 1 + 0.5 * 2 = 2, rho = 1.8 * 1.6 = 2.88
    e2 = (np.array([0, 1]), np.array([1, 0]), np.array([0.0, 4.0]))  # G_0 = 2, rho = 0.2 * 0.4 = 0.08
    ois, wis = f([e1, e2], pi, b, 0.5)
    almost(ois, (2.88 * 2.0 + 0.08 * 2.0) / 2, 1e-12, label="OIS")
    almost(wis, 2.0, 1e-12, label="WIS")


@check("Over 2,000 batches OIS is unbiased for V = 0.9, and WIS has lower variance and stays in [0, 1]")
def test_is_statistics(sub):
    f = has(sub, "importance_sampling_estimates")
    mdp = _one_step_problem()
    pi = np.array([[0.9, 0.1], [0.5, 0.5]])
    b = np.array([[0.5, 0.5], [0.5, 0.5]])
    rng = np.random.default_rng(1)
    # Draw every behaviour action at once: 2,000 batches of 10 one-step episodes.
    acts = rng.integers(2, size=(2000, 10))
    ois, wis = [], []
    for row in acts:
        episodes = [(np.array([0]), np.array([a]), np.array([mdp.R[0, a]])) for a in row]
        o, w = f(episodes, pi, b, 1.0)
        ois.append(o)
        wis.append(w)
    ois, wis = np.array(ois), np.array(wis)
    se = ois.std(ddof=1) / np.sqrt(len(ois))
    if abs(ois.mean() - 0.9) > 3 * se:
        fail(f"the mean OIS estimate should be 0.9 within {3 * se:.4f}, got {ois.mean():.4f}")
    if not (wis.var() < ois.var()):
        fail(f"WIS variance {wis.var():.4f} should be below OIS variance {ois.var():.4f}")
    if wis.min() < -1e-12 or wis.max() > 1 + 1e-12:
        fail(f"every WIS estimate should lie in [0, 1], got a range {wis.min():.3f} to {wis.max():.3f}")
    if ois.max() <= 1.0:
        fail("some OIS estimates should exceed 1, the largest possible return")
