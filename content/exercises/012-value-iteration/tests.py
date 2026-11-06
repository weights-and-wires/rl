"""Checks for the value iteration exercise."""

import numpy as np

import rlenvs
from rlcheck import check, close, equal, fail, has

STAY, GO = 0, 1
V_STAR = np.array([18.0, 20.0])


def _two_state(gamma=0.9):
    P = np.zeros((2, 2, 2))
    P[0, STAY, 0] = P[0, GO, 1] = P[1, STAY, 1] = P[1, GO, 0] = 1.0
    R = np.array([[1.0, 0.0], [2.0, 0.0]])
    return rlenvs.TabularMDP(P=P, R=R, gamma=gamma, mu0=np.array([1.0, 0.0]))


def _big():
    return rlenvs.random_mdp(n_states=20, n_actions=4, gamma=0.95, seed=0)


def _errors(history, v_star):
    return np.max(np.abs(np.asarray(history, dtype=float) - v_star), axis=1)


@check("value_iteration reproduces the chapter's table for k = 0 to 4")
def test_value_iteration_table(sub):
    value_iteration = has(sub, "value_iteration")
    v, history = value_iteration(_two_state(), 1e-10)
    history = np.asarray(history, dtype=float)
    if history.ndim != 2 or history.shape[1] != 2 or history.shape[0] < 5:
        fail(f"history should have shape (sweeps + 1, 2), starting with V_0; got shape {history.shape}")
    want = np.array([[0.0, 0.0], [1.0, 2.0], [1.9, 3.8], [3.42, 5.42], [4.878, 6.878]])
    close(history[:5], want, atol=1e-12, label="V_0 to V_4")
    close(v, V_STAR, atol=1e-8, label="the converged value")
    close(history[-1], v, atol=0, label="the last history row against the returned value")


@check("value_iteration converges to the optimal values of a random 20-state MDP")
def test_value_iteration_random(sub):
    value_iteration = has(sub, "value_iteration")
    mdp = _big()
    v, history = value_iteration(mdp, 1e-10)
    close(v, rlenvs.value_iteration(mdp), atol=1e-7, label="value iteration against the exact optimal values")
    last = np.max(np.abs(history[-1] - history[-2]))
    if last >= 1e-10:
        fail(f"the last sweep should change no value by 1e-10 or more, but it changed one by {last:.3g}")


@check("every iterate obeys |V_k - V*| <= gamma^k |V_0 - V*|, from any starting point")
def test_gamma_k_bound(sub):
    value_iteration = has(sub, "value_iteration")
    mdp = _big()
    v_star = rlenvs.value_iteration(mdp)
    v0 = np.random.default_rng(1).normal(scale=5.0, size=20)
    _, history = value_iteration(mdp, 1e-10, v0=v0)
    history = np.asarray(history, dtype=float)
    close(history[0], v0, atol=0, label="history[0] against v0")
    err = _errors(history, v_star)
    k = np.arange(len(err))
    bound = mdp.gamma**k * err[0]
    worst = int(np.argmax(err - bound))
    if err[worst] > bound[worst] + 1e-9:
        fail(f"at k={worst}, |V_k - V*| = {err[worst]:.4g} exceeds gamma^k |V_0 - V*| = {bound[worst]:.4g}")
    two = _errors(value_iteration(_two_state(), 1e-10)[1], V_STAR)[:10]
    close(two, 20 * 0.9 ** np.arange(10), atol=1e-9, label="two-state errors, where the bound is exactly tight")


@check("the stopping bound gamma / (1 - gamma) |V_k+1 - V_k| is never below the true error")
def test_stopping_bound(sub):
    value_iteration = has(sub, "value_iteration")
    mdp = _big()
    v_star = rlenvs.value_iteration(mdp)
    _, history = value_iteration(mdp, 1e-10)
    history = np.asarray(history, dtype=float)
    err = _errors(history[1:], v_star)
    step = np.max(np.abs(np.diff(history, axis=0)), axis=1)
    bound = mdp.gamma / (1 - mdp.gamma) * step
    worst = int(np.argmax(err - bound))
    if err[worst] > bound[worst] + 1e-9:
        fail(f"after sweep {worst + 1}, the error {err[worst]:.4g} exceeds the bound {bound[worst]:.4g}")


@check("sweeps_needed gives 73 for the two-state MDP at 0.01, exactly the sweep where the error drops below it")
def test_sweeps_needed(sub):
    sweeps_needed = has(sub, "sweeps_needed")
    value_iteration = has(sub, "value_iteration")
    n = sweeps_needed(0.9, 20.0, 0.01)
    if int(n) != 73:
        fail(f"the smallest k with 20 * 0.9^k <= 0.01 is 73, got {n}")
    if int(sweeps_needed(0.9, 0.005, 0.01)) != 0:
        fail("when the initial error is already below eps, no sweeps are needed")
    err = _errors(value_iteration(_two_state(), 1e-10)[1], V_STAR)
    if not (err[73] <= 0.01 < err[72]):
        fail(f"errors at sweeps 72 and 73 should straddle 0.01, got {err[72]:.5f} and {err[73]:.5f}")

    mdp = _big()
    v_star = rlenvs.value_iteration(mdp)
    _, history = value_iteration(mdp, 1e-10)
    err = _errors(history, v_star)
    k = int(sweeps_needed(mdp.gamma, err[0], 1e-3))
    if err[k] > 1e-3:
        fail(f"on the random MDP, {k} sweeps should reach error 1e-3, but the error was {err[k]:.3g}")


@check("stopping_threshold gives the change that guarantees an error: 0.01 * (1 - 0.9) / 0.9")
def test_stopping_threshold(sub):
    stopping_threshold = has(sub, "stopping_threshold")
    value_iteration = has(sub, "value_iteration")
    close(np.array([stopping_threshold(0.9, 0.01)]), np.array([0.01 / 9.0]), atol=1e-15,
          label="stopping_threshold(0.9, 0.01)")
    mdp = _big()
    v_star = rlenvs.value_iteration(mdp)
    eps = 1e-4
    tol = stopping_threshold(mdp.gamma, eps)
    v, _ = value_iteration(mdp, tol)
    err = float(np.max(np.abs(v - v_star)))
    if err > eps:
        fail(f"stopping at a change below {tol:.3g} should guarantee an error of at most {eps:g}, but the error is {err:.3g}")


@check("greedy_policy switches state 0 to go from V_2 on, long before the values converge")
def test_greedy_flips_early(sub):
    greedy_policy = has(sub, "greedy_policy")
    value_iteration = has(sub, "value_iteration")
    mdp = _two_state()
    _, history = value_iteration(mdp, 1e-10)
    actions = [int(np.argmax(np.asarray(greedy_policy(mdp, history[k]))[0])) for k in range(1, 6)]
    equal(np.array(actions), np.array([STAY, GO, GO, GO, GO]), label="greedy action in state 0 for V_1 to V_5")
