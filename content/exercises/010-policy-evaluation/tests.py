"""Checks for the policy evaluation exercise."""

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


def _big():
    return rlenvs.random_mdp(n_states=20, n_actions=4, gamma=0.95, seed=0)


@check("bellman_sweep reproduces the lesson's three hand sweeps for the uniform policy")
def test_hand_sweeps(sub):
    bellman_sweep = has(sub, "bellman_sweep")
    mdp = _hallway()
    pi = rlenvs.uniform_policy(mdp)
    v = np.zeros(4)
    want = [
        [-1.0, -1.0, -1.0, 0.0],
        [-1.9, -1.9, -1.45, 0.0],
        [-2.71, -2.5075, -1.855, 0.0],
    ]
    for k, expected in enumerate(want, start=1):
        v = np.asarray(bellman_sweep(mdp, pi, v), dtype=float)
        shape(v, (4,), label=f"sweep {k} output")
        close(v, expected, atol=1e-10, label=f"values after sweep {k}")


@check("bellman_sweep is synchronous, leaves its input alone, and keeps terminal values at 0")
def test_sweep_contract(sub):
    bellman_sweep = has(sub, "bellman_sweep")
    mdp = _hallway()
    pi = np.random.default_rng(0).dirichlet(np.ones(2), size=4)
    v = np.array([3.0, -2.0, 1.5, 5.0])
    before = v.copy()
    new = np.asarray(bellman_sweep(mdp, pi, v), dtype=float)
    close(v, before, atol=0.0, label="the input v, which the sweep must not modify")
    r = (pi * mdp.R).sum(axis=1)
    P = np.einsum("sa,sat->st", pi, mdp.P)
    want = np.array([r[s] + 0.9 * P[s] @ before for s in range(3)] + [0.0])
    close(new, want, atol=1e-10, label="r_pi + gamma P_pi v, with every entry computed from the old v")
    equal(float(new[3]), 0.0, label="value of the terminal exit, even though the input held 5")


@check("the exact values are a fixed point of bellman_sweep")
def test_fixed_point(sub):
    bellman_sweep = has(sub, "bellman_sweep")
    for seed in range(3):
        mdp = rlenvs.random_mdp(n_states=6, n_actions=3, gamma=0.9, seed=seed)
        pi = np.random.default_rng(seed).dirichlet(np.ones(3), size=6)
        v = rlenvs.policy_evaluation(mdp, pi)
        close(bellman_sweep(mdp, pi, v), v, atol=1e-9, label=f"one sweep from the exact values, random MDP {seed}")


@check("always-right is fully evaluated after four sweeps: the first three change values, the fourth changes nothing")
def test_always_right(sub):
    ipe = has(sub, "iterative_policy_evaluation")
    mdp = _hallway()
    v, sweeps = ipe(mdp, np.tile([0.0, 1.0], (4, 1)), 1e-12)
    shape(np.asarray(v), (4,), label="iterative_policy_evaluation value")
    close(v, [-2.71, -1.9, -1.0, 0.0], atol=1e-10, label="values of the always-right policy")
    equal(int(sweeps), 4, label="number of sweeps")


@check("iterative_policy_evaluation lands near the exact values for the uniform policy, and a smaller tol takes more sweeps")
def test_uniform_hallway(sub):
    ipe = has(sub, "iterative_policy_evaluation")
    mdp = _hallway()
    pi = rlenvs.uniform_policy(mdp)
    exact = rlenvs.policy_evaluation(mdp, pi)
    counts = []
    for tol in (1e-2, 1e-6, 1e-10):
        v, sweeps = ipe(mdp, pi, tol)
        err = np.max(np.abs(v - exact))
        if err > 0.9 / 0.1 * tol:
            fail(f"with tol={tol:g}, the error {err:.3g} exceeds gamma / (1 - gamma) * tol = {9 * tol:.3g}")
        counts.append(int(sweeps))
    if not counts[0] < counts[1] < counts[2]:
        fail(f"a smaller tol should take strictly more sweeps; got {counts} for tol = 1e-2, 1e-6, 1e-10")
    close(v, exact, atol=1e-8, label="values at tol = 1e-10")


@check("on a random 20-state MDP, the result is within gamma / (1 - gamma) * tol of the exact values")
def test_random_mdp(sub):
    ipe = has(sub, "iterative_policy_evaluation")
    mdp = _big()
    pi = rlenvs.uniform_policy(mdp)
    tol = 1e-8
    v, sweeps = ipe(mdp, pi, tol)
    err = np.max(np.abs(v - rlenvs.policy_evaluation(mdp, pi)))
    bound = mdp.gamma / (1 - mdp.gamma) * tol
    print(f"{sweeps} sweeps, true error {err:.2e}, tol {tol:.0e}, bound {bound:.2e}")
    if err > bound:
        fail(f"after stopping at tol={tol:g}, the error {err:.3g} exceeds gamma / (1 - gamma) * tol = {bound:.3g}")


@check("stopping_bound is gamma / (1 - gamma) times the last change, and it bounds the true error")
def test_stopping_bound(sub):
    stopping_bound = has(sub, "stopping_bound")
    ipe = has(sub, "iterative_policy_evaluation")
    close(stopping_bound(0.9, 1e-3), 9e-3, atol=1e-12, label="stopping_bound(0.9, 1e-3)")
    close(stopping_bound(0.95, 2.0), 38.0, atol=1e-9, label="stopping_bound(0.95, 2)")
    mdp = _hallway()
    pi = rlenvs.uniform_policy(mdp)
    exact = rlenvs.policy_evaluation(mdp, pi)
    for tol in (1e-1, 1e-3, 1e-6):
        v, _ = ipe(mdp, pi, tol)
        err = np.max(np.abs(v - exact))
        if err > stopping_bound(mdp.gamma, tol):
            fail(f"with tol={tol:g}, the true error {err:.3g} should stay under stopping_bound = {stopping_bound(mdp.gamma, tol):.3g}")
