"""Reference solution: reward hacking under best-of-n, and a budget.

Part 1, reward hacking. A "world" is a discrete set of N outputs, such as N
possible answers to one prompt. The base policy picks each with probability
base[i]. Every output has a gold score, the reward you want, and one or more
proxy scores, the rewards a learned model would give it. You optimize the
proxy with best-of-n sampling, measure what happens to gold, and try reward
ensembles. All distributions are 1-D numpy arrays of shape (N,) that sum to
one.

Part 2, a budget. A one-state constrained problem has K arms with rewards and
costs. You write the dual function and a primal-dual loop that finds the
policy and the multiplier together.
"""

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


def ensemble_proxy(proxies, how):
    """Combine member scores, shape (k, N), into one proxy by "mean" or "min"."""
    proxies = np.asarray(proxies, dtype=float)
    if how == "mean":
        return proxies.mean(axis=0)
    if how == "min":
        return proxies.min(axis=0)
    raise ValueError(f"how must be 'mean' or 'min', got {how!r}")


def dual_value(reward, cost, budget, lam):
    u = np.asarray(reward, dtype=float) - lam * np.asarray(cost, dtype=float)
    return float(u.max() + lam * budget)



def dual_step(lam, expected_cost, budget, lr):
    return float(max(0.0, lam + lr * (expected_cost - budget)))



def primal_dual(reward, cost, budget, steps=3000, lr_theta=0.5, lr_lam=0.05):
    reward = np.asarray(reward, dtype=float)
    cost = np.asarray(cost, dtype=float)
    theta = np.zeros(len(reward))
    lam = 0.0
    p_sum = np.zeros(len(reward))
    lam_sum = 0.0
    start = steps // 2
    for k in range(steps):
        z = theta - theta.max()
        p = np.exp(z) / np.exp(z).sum()
        u = reward - lam * cost
        if k >= start:
            p_sum += p
            lam_sum += lam
        theta = theta + lr_theta * p * (u - np.sum(p * u))
        lam = dual_step(lam, float(np.sum(p * cost)), budget, lr_lam)
    n = steps - start
    return p_sum / n, lam_sum / n



if __name__ == "__main__":
    gold, proxies = make_world(seed=0)
    base = np.full(gold.shape, 1.0 / gold.size)
    ns = np.array([1, 4, 16, 64, 256, 1024, 4096])
    d, g, p = goodhart_curve(gold, proxies[0], base, ns)
    print("One proxy, best-of-n")
    print("    n      d   proxy    gold")
    for n, di, pi, gi in zip(ns, d, p, g):
        print(f"{n:5d}  {di:5.2f}  {pi:6.2f}  {gi:6.2f}")
    for how in ("mean", "min"):
        _, g_ens, _ = goodhart_curve(gold, ensemble_proxy(proxies, how), base, ns)
        print(f"ensemble {how:4s}: gold peaks at n = {ns[g_ens.argmax()]}, peak gold {g_ens.max():.2f}")

    reward, cost, budget = [1.0, 0.0], [1.0, 0.0], 0.3
    print("dual function g(lam) at 0, 0.5, 1, 2:",
          [round(dual_value(reward, cost, budget, lam), 3) for lam in (0.0, 0.5, 1.0, 2.0)])
    policy, lam = primal_dual(reward, cost, budget)
    print("primal-dual policy:", np.round(policy, 3), " multiplier:", round(lam, 3))
