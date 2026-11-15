"""Checks for the reward-hacking and constrained-RL exercise."""

import numpy as np
from scipy.optimize import linprog

from rlcheck import almost, check, close, fail, has, shape

NS = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])


def _world(sub, seed=0):
    make_world = has(sub, "make_world")
    gold, proxies = make_world(n_outputs=20000, n_members=3, seed=seed)
    base = np.full(gold.shape, 1.0 / gold.size)
    return gold, proxies, base


def _lp_value(reward, cost, budget):
    """The exact constrained optimum, by linear programming over arm probabilities."""
    reward, cost = np.asarray(reward, float), np.asarray(cost, float)
    res = linprog(-reward, A_ub=cost[None, :], b_ub=[budget], A_eq=np.ones((1, len(reward))),
                  b_eq=[1.0], bounds=[(0, 1)] * len(reward))
    return float(-res.fun)


@check("bon_kl gives log n - (n - 1) / n: 0 at n = 1, 0.1931 at n = 2, 3.1745 at n = 64")
def test_bon_kl(sub):
    bon_kl = has(sub, "bon_kl")
    got = np.asarray(bon_kl(np.array([1, 2, 4, 16, 64])), dtype=float)
    want = np.array([0.0, 0.19314718, 0.63629436, 1.83508872, 3.17450808])
    close(got, want, atol=1e-7, label="bon_kl at n = 1, 2, 4, 16, 64")


@check("best_of_n_policy matches F(i)^n - F(i-1)^n on hand-sized examples, in the original order")
def test_best_of_n_small(sub):
    best_of_n_policy = has(sub, "best_of_n_policy")
    got = best_of_n_policy(np.array([0.0, 1.0, 2.0, 3.0]), 2, np.full(4, 0.25))
    shape(got, (4,), label="best_of_n_policy output")
    close(got, np.array([1, 3, 5, 7]) / 16, atol=1e-12, label="n = 2 on four uniform outputs")
    base = np.array([0.5, 0.3, 0.2])
    got = best_of_n_policy(np.array([2.0, 0.0, 1.0]), 2, base)
    close(got, np.array([0.75, 0.09, 0.16]), atol=1e-12,
          label="n = 2, proxy [2, 0, 1], base [0.5, 0.3, 0.2]")
    close(best_of_n_policy(np.array([2.0, 0.0, 1.0]), 1, base), base, atol=1e-12,
          label="n = 1 must return the base policy")


@check("kl_divergence matches a hand computation and treats 0 log 0 as 0")
def test_kl(sub):
    kl_divergence = has(sub, "kl_divergence")
    p = np.array([0.04, 0.12, 0.2, 0.28, 0.36])
    q = np.full(5, 0.2)
    almost(kl_divergence(p, q), float(np.sum(p * np.log(p / 0.2))), 1e-12,
           label="KL of best-of-2 on five outputs")
    almost(kl_divergence(np.array([0.0, 1.0]), np.array([0.5, 0.5])), np.log(2.0), 1e-12,
           label="KL([0, 1] || [0.5, 0.5])")


@check("goodhart_curve: the proxy climbs with every n, but gold peaks inside the range and falls")
def test_goodhart_curve(sub):
    goodhart_curve = has(sub, "goodhart_curve")
    bon_kl = has(sub, "bon_kl")
    gold, proxies, base = _world(sub)
    d, g, p = goodhart_curve(gold, proxies[0], base, NS)
    for arr, name in ((d, "d"), (g, "gold score"), (p, "proxy score")):
        shape(arr, (len(NS),), label=name)
    close(d[1:] ** 2, bon_kl(NS[1:]), rtol=0.005, label="d squared against bon_kl")
    if not np.all(np.diff(p) > 0):
        fail(f"the proxy score must rise with n, got {np.round(p, 3)}")
    peak = int(np.argmax(g))
    if peak in (0, len(NS) - 1):
        fail(f"gold should peak strictly inside n = 1 to 4096, but peaks at n = {NS[peak]}")
    if g[peak] - g[-1] < 0.6:
        fail(f"gold should fall by at least 0.6 after its peak; peak {g[peak]:.3f}, "
             f"at n = 4096 {g[-1]:.3f}")


@check("Mean and min ensembles push the gold peak to larger n and raise it")
def test_ensembles(sub):
    goodhart_curve = has(sub, "goodhart_curve")
    ensemble_proxy = has(sub, "ensemble_proxy")
    gold, proxies, base = _world(sub)
    close(ensemble_proxy(proxies, "mean"), proxies.mean(axis=0), atol=1e-12, label="mean")
    close(ensemble_proxy(proxies, "min"), proxies.min(axis=0), atol=1e-12, label="min")
    peaks, heights = {}, {}
    for name, proxy in (("single", proxies[0]),
                        ("mean", ensemble_proxy(proxies, "mean")),
                        ("min", ensemble_proxy(proxies, "min"))):
        _, g, _ = goodhart_curve(gold, proxy, base, NS)
        peaks[name], heights[name] = int(NS[np.argmax(g)]), float(g.max())
    if peaks["mean"] < 2 * peaks["single"]:
        fail(f"the mean ensemble should peak at least twice as late as one member: "
             f"one member peaks at n = {peaks['single']}, the mean at n = {peaks['mean']}")
    if peaks["min"] < peaks["mean"]:
        fail(f"the min ensemble should peak no earlier than the mean: min at "
             f"n = {peaks['min']}, mean at n = {peaks['mean']}")
    if not heights["min"] > heights["mean"] > heights["single"] + 0.2:
        fail("peak gold should rise from one member to the mean to the min, got "
             f"{heights['single']:.3f}, {heights['mean']:.3f}, {heights['min']:.3f}")


@check("dual_value gives 1, 0.65, 0.3, and 0.6 at lam = 0, 0.5, 1, 2 on the one-state example")
def test_dual_value(sub):
    dual_value = has(sub, "dual_value")
    for lam, want in ((0.0, 1.0), (0.5, 0.65), (1.0, 0.3), (2.0, 0.6)):
        almost(dual_value([1.0, 0.0], [1.0, 0.0], 0.3, lam), want, 1e-12, label=f"g({lam})")


@check("dual_value stays above the constrained optimum for every lam, and touches it at the best lam")
def test_duality(sub):
    dual_value = has(sub, "dual_value")
    reward, cost, budget = [1.0, 0.6, 0.0], [1.0, 0.4, 0.0], 0.5
    best = _lp_value(reward, cost, budget)
    lams = np.linspace(0.0, 3.0, 3001)
    values = np.array([dual_value(reward, cost, budget, lam) for lam in lams])
    if np.any(values < best - 1e-9):
        fail(f"weak duality: g(lam) must be at least the constrained optimum {best:.4f}, "
             f"but the smallest g was {values.min():.4f}")
    almost(float(values.min()), best, 1e-3, label="the smallest g(lam) against the LP optimum")


@check("dual_step raises lam when over budget, lowers it when under, and never goes below 0")
def test_dual_step(sub):
    dual_step = has(sub, "dual_step")
    almost(dual_step(1.0, 0.5, 0.3, 0.1), 1.02, 1e-12, label="over budget by 0.2")
    almost(dual_step(1.0, 0.1, 0.3, 0.1), 0.98, 1e-12, label="under budget by 0.2")
    almost(dual_step(0.01, 0.0, 0.3, 0.1), 0.0, 1e-12, label="projected at 0")


@check("primal_dual lands on the constrained optimum: play the costly arm with probability 0.3, lam near 1")
def test_primal_dual(sub):
    primal_dual = has(sub, "primal_dual")
    p, lam = primal_dual([1.0, 0.0], [1.0, 0.0], 0.3)
    shape(np.asarray(p), (2,), label="averaged policy")
    almost(float(p[0]), 0.3, 0.03, label="probability of the costly arm")
    almost(float(lam), 1.0, 0.15, label="averaged multiplier")
