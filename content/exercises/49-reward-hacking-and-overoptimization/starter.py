"""Reward hacking and overoptimisation: the Goodhart curve, measured exactly.

A "world" here is a discrete set of N outputs, such as N possible answers to
one prompt. The base policy picks each with probability base[i]. Every
output has a gold score, the reward you want, and one or more proxy scores,
the rewards a learned model would give it. You optimise the proxy with
best-of-n sampling, measure what happens to gold, fit the Gao, Schulman,
and Hilton (2022) curve, and try reward ensembles.

All distributions are 1-D numpy arrays of shape (N,) that sum to one.
"""

import numpy as np


def make_world(n_outputs=20000, n_members=3, seed=0, df=3.0, scale=0.5, shared=0.3):
    """A gold reward and an ensemble of noisy proxy rewards over a discrete output space.

    Each output has a gold score drawn from a standard normal. Proxy member k
    scores it as gold + scale * (shared * e0 + (1 - shared) * e_k), where e0
    is an error every member shares and e_k is member k's own error. Both
    errors are Student-t with `df` degrees of freedom, so a few outputs fool a
    member by a lot.

    This helper is given. You don't need to change it.

    Returns:
        gold: shape (n_outputs,).
        proxies: shape (n_members, n_outputs).
    """
    rng = np.random.default_rng(seed)
    gold = rng.standard_normal(n_outputs)
    common = rng.standard_t(df, size=n_outputs)
    own = rng.standard_t(df, size=(n_members, n_outputs))
    proxies = gold + scale * (shared * common + (1.0 - shared) * own)
    return gold, proxies


def bon_kl(n):
    """The analytic best-of-n KL divergence from the base policy.

    Args:
        n: an integer or an array of integers, each at least 1.

    Returns:
        log(n) - (n - 1) / n, elementwise, as a float or float array.
    """
    raise NotImplementedError("bon_kl")


def best_of_n_policy(proxy, n, base):
    """The exact distribution of best-of-n sampling on a discrete space.

    Draw n outputs independently from `base` and keep the one with the
    highest proxy score. Assume the proxy has no ties.

    Args:
        proxy: proxy score of each output, shape (N,). Not necessarily sorted.
        n: the number of draws, an integer at least 1.
        base: the base policy, shape (N,), summing to one.

    Returns:
        The probability that each output is the one kept, shape (N,), in the
        same order as `proxy`.
    """
    raise NotImplementedError("best_of_n_policy")


def kl_divergence(p, q):
    """KL(p || q) = sum_i p_i log(p_i / q_i), with terms where p_i == 0 counted as 0.

    Args:
        p, q: distributions of shape (N,), with q_i > 0 wherever p_i > 0.

    Returns:
        A float.
    """
    raise NotImplementedError("kl_divergence")


def tilted_policy(proxy, beta, base):
    """The optimum of E[proxy] - beta * KL(pi || base): base * exp(proxy / beta), normalised.

    Args:
        proxy: shape (N,).
        beta: the KL coefficient, positive.
        base: shape (N,), strictly positive.

    Returns:
        A distribution of shape (N,). It must not overflow for large proxy / beta.
    """
    raise NotImplementedError("tilted_policy")


def goodhart_curve(gold, proxy, base, ns):
    """Trace best-of-n on the proxy, scoring each policy with gold and with the proxy.

    Args:
        gold: shape (N,).
        proxy: shape (N,), the reward best-of-n optimises.
        base: shape (N,).
        ns: a sequence of integers n.

    Returns:
        A tuple (d, gold_score, proxy_score), each of shape (len(ns),).
        d[j] = sqrt(KL(pi_j || base)), using the exact KL of the exact
        best-of-n policy pi_j for n = ns[j]. gold_score[j] and proxy_score[j]
        are the expected gold and proxy scores under pi_j.
    """
    raise NotImplementedError("goodhart_curve")


def fit_gao_bon(d, R):
    """Least-squares fit of the best-of-n form R(d) = d * (a - b * d).

    Args:
        d: distances, shape (m,).
        R: gold scores measured relative to the base policy, shape (m,).

    Returns:
        A tuple (a, b) of floats.
    """
    raise NotImplementedError("fit_gao_bon")


def fit_gao_rl(d, R):
    """Least-squares fit of the RL form R(d) = d * (a - b * log d).

    Args:
        d: distances, shape (m,), all strictly positive.
        R: gold scores measured relative to the base policy, shape (m,).

    Returns:
        A tuple (a, b) of floats.
    """
    raise NotImplementedError("fit_gao_rl")


def ensemble_proxy(proxies, how):
    """Combine an ensemble's scores into one proxy.

    Args:
        proxies: shape (k, N), one row per member.
        how: "mean" to average the members, "min" to take the lowest score.

    Returns:
        Shape (N,).
    """
    raise NotImplementedError("ensemble_proxy")


if __name__ == "__main__":
    gold, proxies = make_world(seed=0)
    base = np.full(gold.shape, 1.0 / gold.size)
    ns = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])

    d, g, p = goodhart_curve(gold, proxies[0], base, ns)
    a, b = fit_gao_bon(d[1:], g[1:])
    print("One proxy, best-of-n")
    print("    n      d   proxy    gold")
    for n, di, pi, gi in zip(ns, d, p, g):
        print(f"{n:5d}  {di:5.2f}  {pi:6.2f}  {gi:6.2f}")
    print(f"Gao fit: a = {a:.3f}, b = {b:.3f}, peak at d = {a / (2 * b):.2f}")

    for how in ("mean", "min"):
        _, g_ens, _ = goodhart_curve(gold, ensemble_proxy(proxies, how), base, ns)
        print(f"ensemble {how:4s}: gold peaks at n = {ns[g_ens.argmax()]}, "
              f"peak gold {g_ens.max():.2f}")
