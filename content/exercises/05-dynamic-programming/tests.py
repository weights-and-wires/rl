"""Checks for the dynamic programming exercise."""

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


def _big():
    return rlenvs.random_mdp(n_states=20, n_actions=4, gamma=0.95, seed=0)


def _errors(history, v_star):
    return np.max(np.abs(np.asarray(history, dtype=float) - v_star), axis=1)


@check("iterative_policy_evaluation lands within gamma / (1 - gamma) * tol of the exact value")
def test_iterative_policy_evaluation(sub):
    ipe = has(sub, "iterative_policy_evaluation")
    mdp = _two_state()
    v, sweeps = ipe(mdp, rlenvs.uniform_policy(mdp), 1e-10)
    shape(v, (2,), label="iterative_policy_evaluation value")
    close(v, [7.25, 7.75], atol=1e-8, label="uniform policy on the two-state MDP")

    mdp = _big()
    pi = rlenvs.uniform_policy(mdp)
    tol = 1e-8
    v, sweeps = ipe(mdp, pi, tol)
    err = np.max(np.abs(v - rlenvs.policy_evaluation(mdp, pi)))
    bound = mdp.gamma / (1 - mdp.gamma) * tol
    print(f"{sweeps} sweeps, true error {err:.2e}, tol {tol:.0e}, bound {bound:.2e}")
    if err > bound:
        fail(f"after stopping at tol={tol:g}, the error {err:.3g} exceeds gamma / (1 - gamma) * tol = {bound:.3g}")


@check("greedy_policy picks go in state 0 and stay in state 1 from V*, and breaks ties by the first action")
def test_greedy_policy(sub):
    greedy_policy = has(sub, "greedy_policy")
    mdp = _two_state()
    pi = greedy_policy(mdp, V_STAR)
    shape(pi, (2, 2), label="greedy_policy output")
    equal(np.asarray(pi, dtype=float), np.array([[0.0, 1.0], [1.0, 0.0]]), label="greedy policy from V* = (18, 20)")

    base = rlenvs.random_mdp(n_states=5, n_actions=2, gamma=0.9, seed=3)
    P = base.P.copy()
    R = base.R.copy()
    P[:, 1], R[:, 1] = P[:, 0], R[:, 0]  # two identical actions
    twin = rlenvs.TabularMDP(P=P, R=R, gamma=0.9, mu0=base.mu0)
    pi = np.asarray(greedy_policy(twin, np.random.default_rng(0).normal(size=5)), dtype=float)
    equal(pi, np.tile([1.0, 0.0], (5, 1)), label="identical actions: pick action 0 everywhere")


@check("value_iteration reproduces the lesson's table for k = 0 to 4")
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


@check("every value iteration iterate obeys |V_k - V*| <= gamma^k |V_0 - V*|")
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


@check("greedy_policy switches state 0 to go from V_2 on, long before the values converge")
def test_greedy_flips_early(sub):
    greedy_policy = has(sub, "greedy_policy")
    value_iteration = has(sub, "value_iteration")
    mdp = _two_state()
    _, history = value_iteration(mdp, 1e-10)
    actions = [int(np.argmax(np.asarray(greedy_policy(mdp, history[k]))[0])) for k in range(1, 6)]
    equal(np.array(actions), np.array([STAY, GO, GO, GO, GO]), label="greedy action in state 0 for V_1 to V_5")


@check("policy_iteration evaluates (10, 20), then (18, 20), and stops")
def test_policy_iteration_worked(sub):
    policy_iteration = has(sub, "policy_iteration")
    pi, v, values = policy_iteration(_two_state())
    close(np.asarray(values, dtype=float), [[10.0, 20.0], [18.0, 20.0]], atol=1e-10,
          label="values of each evaluated policy, from 'stay everywhere'")
    equal(np.asarray(pi, dtype=float), np.array([[0.0, 1.0], [1.0, 0.0]]), label="the final policy")
    close(v, V_STAR, atol=1e-10, label="the final value")


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
