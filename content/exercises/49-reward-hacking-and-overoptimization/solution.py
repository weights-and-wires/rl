"""Reference solution for the reward overoptimisation exercise."""

import numpy as np


def make_world(n_outputs=20000, n_members=3, seed=0, df=3.0, scale=0.5, shared=0.3):
    """A gold reward and an ensemble of noisy proxy rewards over a discrete output space.

    Each output has a gold score drawn from a standard normal. Proxy member k
    scores it as gold + scale * (shared * e0 + (1 - shared) * e_k), where e0
    is an error every member shares and e_k is member k's own error. Both
    errors are Student-t with `df` degrees of freedom, so a few outputs fool a
    member by a lot.

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
    """The analytic best-of-n KL, log n - (n - 1) / n, elementwise over n."""
    n = np.asarray(n, dtype=float)
    return np.log(n) - (n - 1.0) / n


def best_of_n_policy(proxy, n, base):
    """The exact distribution of the proxy-best of n draws from `base`."""
    proxy = np.asarray(proxy, dtype=float)
    base = np.asarray(base, dtype=float)
    order = np.argsort(proxy, kind="stable")
    cdf = np.cumsum(base[order])
    below = cdf - base[order]
    cdf = np.clip(cdf, 0.0, 1.0)
    below = np.clip(below, 0.0, 1.0)
    policy = np.empty_like(base)
    policy[order] = cdf**n - below**n
    return policy


def kl_divergence(p, q):
    """KL(p || q) for two distributions over the same outputs, with 0 log 0 = 0."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def tilted_policy(proxy, beta, base):
    """The KL-regularised optimum: base * exp(proxy / beta), normalised."""
    proxy = np.asarray(proxy, dtype=float)
    base = np.asarray(base, dtype=float)
    logits = np.log(base) + proxy / beta
    logits = logits - logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


def goodhart_curve(gold, proxy, base, ns):
    """Distance, gold score, and proxy score of best-of-n for each n in `ns`."""
    gold = np.asarray(gold, dtype=float)
    proxy = np.asarray(proxy, dtype=float)
    dist, gold_score, proxy_score = [], [], []
    for n in ns:
        policy = best_of_n_policy(proxy, n, base)
        dist.append(np.sqrt(max(kl_divergence(policy, base), 0.0)))
        gold_score.append(policy @ gold)
        proxy_score.append(policy @ proxy)
    return np.array(dist), np.array(gold_score), np.array(proxy_score)


def fit_gao_bon(d, R):
    """Least-squares fit of R = d * (a - b * d). Returns (a, b)."""
    d = np.asarray(d, dtype=float)
    X = np.stack([d, -d**2], axis=1)
    (a, b), *_ = np.linalg.lstsq(X, np.asarray(R, dtype=float), rcond=None)
    return float(a), float(b)


def fit_gao_rl(d, R):
    """Least-squares fit of R = d * (a - b * log d), for d > 0. Returns (a, b)."""
    d = np.asarray(d, dtype=float)
    X = np.stack([d, -d * np.log(d)], axis=1)
    (a, b), *_ = np.linalg.lstsq(X, np.asarray(R, dtype=float), rcond=None)
    return float(a), float(b)


def ensemble_proxy(proxies, how):
    """Combine member scores, shape (k, N), into one proxy by "mean" or "min"."""
    proxies = np.asarray(proxies, dtype=float)
    if how == "mean":
        return proxies.mean(axis=0)
    if how == "min":
        return proxies.min(axis=0)
    raise ValueError(f"how must be 'mean' or 'min', got {how!r}")


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
