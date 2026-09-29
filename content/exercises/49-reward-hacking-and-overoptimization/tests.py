"""Checks for the reward overoptimisation exercise."""

import numpy as np

from rlcheck import almost, check, close, fail, has, shape

NS = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])


def _world(sub, seed=0):
    make_world = has(sub, "make_world")
    gold, proxies = make_world(n_outputs=20000, n_members=3, seed=seed)
    base = np.full(gold.shape, 1.0 / gold.size)
    return gold, proxies, base


@check("bon_kl gives log n - (n - 1) / n: 0 at n = 1, 0.1931 at n = 2, 3.1745 at n = 64")
def test_bon_kl(sub):
    bon_kl = has(sub, "bon_kl")
    got = np.asarray(bon_kl(np.array([1, 2, 4, 16, 64])), dtype=float)
    want = np.array([0.0, 0.19314718, 0.63629436, 1.83508872, 3.17450808])
    close(got, want, atol=1e-7, label="bon_kl at n = 1, 2, 4, 16, 64")


@check("best_of_n_policy matches F(i)^n - F(i-1)^n on hand-sized examples")
def test_best_of_n_small(sub):
    best_of_n_policy = has(sub, "best_of_n_policy")
    # Four equally likely outputs, already sorted by proxy: n = 2 gives 1, 3, 5, 7 sixteenths.
    got = best_of_n_policy(np.array([0.0, 1.0, 2.0, 3.0]), 2, np.full(4, 0.25))
    shape(got, (4,), label="best_of_n_policy output")
    close(got, np.array([1, 3, 5, 7]) / 16, atol=1e-12, label="n = 2 on four uniform outputs")

    # Unsorted proxy and an uneven base: sorted order is outputs 1, 2, 0.
    base = np.array([0.5, 0.3, 0.2])
    got = best_of_n_policy(np.array([2.0, 0.0, 1.0]), 2, base)
    close(got, np.array([0.75, 0.09, 0.16]), atol=1e-12,
          label="n = 2, proxy [2, 0, 1], base [0.5, 0.3, 0.2]")
    close(best_of_n_policy(np.array([2.0, 0.0, 1.0]), 1, base), base, atol=1e-12,
          label="n = 1 must return the base policy")


@check("best_of_n_policy agrees with 200,000 simulated best-of-3 draws")
def test_best_of_n_monte_carlo(sub):
    best_of_n_policy = has(sub, "best_of_n_policy")
    rng = np.random.default_rng(1)
    base = rng.dirichlet(np.ones(6))
    proxy = rng.standard_normal(6)
    draws = rng.choice(6, size=(200000, 3), p=base)
    picked = draws[np.arange(draws.shape[0]), np.argmax(proxy[draws], axis=1)]
    empirical = np.bincount(picked, minlength=6) / draws.shape[0]
    got = best_of_n_policy(proxy, 3, base)
    # Standard error of each frequency is at most 0.0012; allow about five of them.
    close(got, empirical, atol=0.006, label="exact best-of-3 against simulation")


@check("kl_divergence matches a hand computation and treats 0 log 0 as 0")
def test_kl(sub):
    kl_divergence = has(sub, "kl_divergence")
    p = np.array([0.04, 0.12, 0.2, 0.28, 0.36])
    q = np.full(5, 0.2)
    almost(kl_divergence(p, q), float(np.sum(p * np.log(p / 0.2))), 1e-12,
           label="KL of best-of-2 on five outputs")
    almost(kl_divergence(np.array([0.0, 1.0]), np.array([0.5, 0.5])), np.log(2.0), 1e-12,
           label="KL([0, 1] || [0.5, 0.5])")


@check("On 20,000 outputs the exact best-of-n KL sits within 0.5% of the formula, and never above it")
def test_kl_bound(sub):
    best_of_n_policy = has(sub, "best_of_n_policy")
    kl_divergence = has(sub, "kl_divergence")
    bon_kl = has(sub, "bon_kl")
    _, proxies, base = _world(sub)
    for n in (2, 4, 16, 64, 256, 1024):
        exact = kl_divergence(best_of_n_policy(proxies[0], n, base), base)
        formula = float(bon_kl(n))
        if exact > formula + 1e-9:
            fail(f"at n = {n} the exact KL {exact:.6f} is above the formula {formula:.6f}, "
                 "but the formula is an upper bound on a discrete space")
        if abs(exact / formula - 1.0) > 0.005:
            fail(f"at n = {n} the exact KL {exact:.6f} is more than 0.5% from {formula:.6f}")


@check("tilted_policy is base * exp(proxy / beta) normalised, and returns the base as beta grows")
def test_tilted(sub):
    tilted_policy = has(sub, "tilted_policy")
    base = np.array([0.5, 0.3, 0.2])
    proxy = np.array([1.0, 0.0, 0.0])
    got = tilted_policy(proxy, 1.0, base)
    want = base * np.exp(proxy)
    close(got, want / want.sum(), atol=1e-12, label="beta = 1")
    close(tilted_policy(proxy, 1e8, base), base, atol=1e-7, label="beta = 1e8")
    close(tilted_policy(np.array([800.0, 0.0, 0.0]), 1.0, base), np.array([1.0, 0.0, 0.0]),
          atol=1e-12, label="a proxy of 800 must not overflow")


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


@check("fit_gao_bon and fit_gao_rl recover planted coefficients exactly")
def test_fits_exact(sub):
    fit_gao_bon = has(sub, "fit_gao_bon")
    fit_gao_rl = has(sub, "fit_gao_rl")
    d = np.linspace(0.1, 3.0, 12)
    a, b = fit_gao_bon(d, d * (1.3 - 0.4 * d))
    almost(a, 1.3, 1e-9, label="best-of-n a")
    almost(b, 0.4, 1e-9, label="best-of-n b")
    a, b = fit_gao_rl(d, d * (0.7 - 0.25 * np.log(d)))
    almost(a, 0.7, 1e-9, label="RL a")
    almost(b, 0.25, 1e-9, label="RL b")


@check("The best-of-n fit on the simulated curve has b > 0 and puts the peak near the true one")
def test_fit_simulated(sub):
    goodhart_curve = has(sub, "goodhart_curve")
    fit_gao_bon = has(sub, "fit_gao_bon")
    gold, proxies, base = _world(sub)
    d, g, _ = goodhart_curve(gold, proxies[0], base, NS)
    a, b = fit_gao_bon(d, g)
    if not (a > 0 and b > 0):
        fail(f"expected a > 0 and b > 0 for a rise-then-fall curve, got a = {a:.3f}, b = {b:.3f}")
    d_star = a / (2 * b)
    d_peak = d[int(np.argmax(g))]
    if abs(d_star - d_peak) > 0.6:
        fail(f"the fitted peak d* = a / (2b) = {d_star:.3f} is far from the simulated "
             f"peak at d = {d_peak:.3f}")


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
