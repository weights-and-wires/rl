"""Advantage-weighted policy improvement: AWR, MPO's temperature, and AWAC.

A policy improves by reweighting what it already does. The functions here
build that step in pieces: the weights, the closed-form improved policy, the
temperature that sets the step's size, the weighted maximum-likelihood fit,
and a full AWAC loop on a fixed dataset.

The helpers `make_dataset`, `action_counts`, `cautious_behavior`, and
`behavior_cloning` are written for you (the first three are the ones from the
offline RL chapter). Fill in the functions after them.
"""

import numpy as np

import rlenvs


def make_dataset(mdp, behavior: np.ndarray, n_episodes: int, seed: int, max_steps: int = 500) -> dict:
    """Roll out `behavior` for `n_episodes` episodes and store every transition.

    Returns a dict of equal-length arrays: `s`, `a`, `r`, `s2` (integers or
    floats, shape (N,)), and `done`, 1.0 where `s2` is terminal. Episodes run
    until they reach a terminal state or `max_steps` steps.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_episodes):
        s = int(rng.choice(mdp.n_states, p=mdp.mu0))
        for _ in range(max_steps):
            a = int(rng.choice(mdp.n_actions, p=behavior[s]))
            s2, r = mdp.sample(s, a, rng)
            done = bool(mdp.terminal[s2])
            rows.append((s, a, r, s2, float(done)))
            if done:
                break
            s = s2
    s, a, r, s2, done = (np.array(col) for col in zip(*rows))
    return {"s": s.astype(int), "a": a.astype(int), "r": r.astype(float), "s2": s2.astype(int), "done": done}


def action_counts(data: dict, n_states: int, n_actions: int) -> np.ndarray:
    """How many times each (s, a) pair appears in the data, shape (S, A)."""
    counts = np.zeros((n_states, n_actions))
    np.add.at(counts, (data["s"], data["a"]), 1.0)
    return counts


def cautious_behavior(mdp) -> np.ndarray:
    """A demonstrator that heads for the goal but never steps down toward the cliff edge."""
    S, A = mdp.n_states, mdp.n_actions
    cells = mdp.names["cells"]
    allowed = np.ones((S, A), dtype=bool)
    for k, (i, j) in enumerate(cells):
        if i == 2 and 1 <= j <= 10:
            allowed[k, rlenvs.DOWN] = False
    allowed[mdp.names["start"], rlenvs.RIGHT] = False
    q = rlenvs.q_from_v(mdp, rlenvs.value_iteration(mdp))
    q = np.where(allowed, q, -np.inf)
    best = q.argmax(axis=1)
    pi = 0.4 * allowed / allowed.sum(axis=1, keepdims=True)
    pi[np.arange(S), best] += 0.6
    return pi


def behavior_cloning(data: dict, n_states: int, n_actions: int) -> np.ndarray:
    """The tabular maximum-likelihood policy: each action's share of the data at that state.

    A state with no data gets the uniform policy. Shape (S, A), rows sum to 1.
    """
    counts = action_counts(data, n_states, n_actions)
    totals = counts.sum(axis=1, keepdims=True)
    return np.where(totals > 0, counts / np.maximum(totals, 1.0), 1.0 / n_actions)


def _softmax(z: np.ndarray) -> np.ndarray:
    """Softmax along the last axis with the row max subtracted."""
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def awr_weights(adv: np.ndarray, beta: float, clip: float) -> np.ndarray:
    """AWR's sample weights: exp(adv / beta), capped at `clip`.

    Args:
        adv: advantages, shape (N,).
        beta: the temperature, positive. Small beta means a sharp weighting.
        clip: the largest weight allowed.

    Returns:
        Shape (N,). Must not overflow, even for an advantage of 1e4.
    """
    log_w = np.minimum(np.asarray(adv, dtype=float) / beta, np.log(clip))
    return np.exp(log_w)


def kl_improved_policy(pi_old: np.ndarray, adv: np.ndarray, beta: float) -> np.ndarray:
    """The policy that maximizes E_pi[adv] - beta * KL(pi || pi_old).

    It is pi_old * exp(adv / beta) / Z, where Z makes it sum to one.

    Args:
        pi_old: probabilities over actions, shape (A,), all positive.
        adv: advantage of each action, shape (A,).
        beta: the temperature, positive.

    Returns:
        Shape (A,), summing to 1. Must be stable when adv is shifted by 1000.
    """
    return _softmax(np.log(pi_old) + np.asarray(adv, dtype=float) / beta)


def improvement_objective(pi: np.ndarray, pi_old: np.ndarray, adv: np.ndarray, beta: float) -> float:
    """E_pi[adv] - beta * KL(pi || pi_old), as a float.

    A zero entry of `pi` contributes nothing to the KL (0 log 0 = 0).
    """
    pi = np.asarray(pi, dtype=float)
    pos = pi > 0
    kl = np.sum(pi[pos] * (np.log(pi[pos]) - np.log(pi_old[pos])))
    return float(np.sum(pi * adv) - beta * kl)


def mpo_weights(q: np.ndarray, eta: float) -> np.ndarray:
    """MPO's E-step weights on N sampled actions: softmax(q / eta), shape (N,).

    They sum to one over the batch and must not overflow for q near 1000.
    """
    return _softmax(np.asarray(q, dtype=float) / eta)


def weight_kl(w: np.ndarray) -> float:
    """KL(w || uniform) for weights w over N samples: sum_i w_i log(N w_i)."""
    pos = w > 0
    return float(np.sum(w[pos] * np.log(w.size * w[pos])))


def mpo_dual(q: np.ndarray, eta: float, eps: float) -> float:
    """MPO's temperature dual: eta * eps + eta * log mean_i exp(q_i / eta).

    Use a stable log-mean-exp. Its derivative in eta is eps - KL(w || uniform),
    where w = mpo_weights(q, eta).
    """
    q = np.asarray(q, dtype=float)
    top = q.max()
    log_mean_exp = top / eta + np.log(np.mean(np.exp((q - top) / eta)))
    return float(eta * eps + eta * log_mean_exp)


def mpo_temperature(q: np.ndarray, eps: float) -> float:
    """The eta > 0 that minimizes `mpo_dual`, found by a one-dimensional search.

    At the minimum, the weights softmax(q / eta) sit exactly eps away from
    uniform: weight_kl(mpo_weights(q, eta)) == eps. The KL falls as eta grows,
    so bisect on it in log eta.

    Args:
        q: action values of N sampled actions, shape (N,), not all equal.
        eps: the KL budget, positive and below log N.
    """
    q = np.asarray(q, dtype=float)
    spread = max(np.ptp(q), 1e-12)
    lo, hi = np.log(1e-8 * spread), np.log(1e8 * spread)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if weight_kl(mpo_weights(q, np.exp(mid))) > eps:
            lo = mid  # the weights are too sharp, so raise eta
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))


def awr_loss(logits: np.ndarray, actions: np.ndarray, weights: np.ndarray) -> float:
    """The weighted negative log-likelihood of a batch of discrete actions.

    -(1/N) sum_i weights_i * log softmax(logits_i)[actions_i].

    Args:
        logits: shape (N, A).
        actions: integer actions, shape (N,).
        weights: non-negative sample weights, shape (N,), treated as constants.
    """
    logits = np.asarray(logits, dtype=float)
    top = logits.max(axis=1, keepdims=True)
    logp = logits - top - np.log(np.exp(logits - top).sum(axis=1, keepdims=True))
    return float(-np.mean(weights * logp[np.arange(len(actions)), actions]))


def awr_loss_grad(logits: np.ndarray, actions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Gradient of `awr_loss` with respect to `logits`, shape (N, A)."""
    p = _softmax(np.asarray(logits, dtype=float))
    onehot = np.zeros_like(p)
    onehot[np.arange(len(actions)), actions] = 1.0
    return weights[:, None] * (p - onehot) / len(actions)


def weighted_gaussian_mle(actions: np.ndarray, weights: np.ndarray) -> tuple:
    """The Gaussian that maximizes sum_i weights_i * log N(actions_i; mu, sigma^2).

    Args:
        actions: shape (N,).
        weights: non-negative, shape (N,). They need not sum to one.

    Returns:
        (mu, sigma), the weighted mean and weighted standard deviation.
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    mu = np.sum(w * actions)
    return float(mu), float(np.sqrt(np.sum(w * (actions - mu) ** 2)))


def mpo_gaussian_step(actions: np.ndarray, q: np.ndarray, eps: float) -> tuple:
    """One MPO step for a Gaussian policy from N sampled actions and their values.

    E-step: choose eta with `mpo_temperature`, then weight with `mpo_weights`.
    M-step: fit a Gaussian to the weighted actions with `weighted_gaussian_mle`.

    Returns:
        (mu, sigma, eta).
    """
    eta = mpo_temperature(q, eps)
    mu, sigma = weighted_gaussian_mle(actions, mpo_weights(q, eta))
    return mu, sigma, eta


def awac_tabular(
    data: dict,
    n_states: int,
    n_actions: int,
    gamma: float,
    beta: float,
    clip: float = 20.0,
    rounds: int = 10,
    eval_iters: int = 300,
) -> np.ndarray:
    """Tabular AWAC on a fixed dataset. Returns the final policy, shape (S, A).

    Start from `behavior_cloning`. Then repeat `rounds` times:
      1. Critic: starting from zeros, run `eval_iters` sweeps of
         Q[s, a] <- mean over the data's (s, a) transitions of
         r + gamma * (1 - done) * sum_a2 pi[s2, a2] * Q[s2, a2].
         Pairs the data never took stay at 0.
      2. Advantage: for each transition, Q[s, a] - sum_a2 pi[s, a2] * Q[s, a2].
      3. Weights: awr_weights(advantage, beta, clip).
      4. Actor: the weighted maximum-likelihood tabular policy,
         pi[s, a] proportional to the sum of the weights of the transitions
         with that (s, a). A state with no data keeps the uniform policy.
    """
    s, a, r, s2, done = data["s"], data["a"], data["r"], data["s2"], data["done"]
    counts = action_counts(data, n_states, n_actions)
    seen = counts > 0
    has_data = seen.any(axis=1, keepdims=True)
    pi = behavior_cloning(data, n_states, n_actions)
    for _ in range(rounds):
        Q = np.zeros((n_states, n_actions))
        for _ in range(eval_iters):
            v = (pi * Q).sum(axis=1)
            y = r + gamma * (1.0 - done) * v[s2]
            total = np.zeros((n_states, n_actions))
            np.add.at(total, (s, a), y)
            Q = np.where(seen, total / np.maximum(counts, 1.0), 0.0)
        v = (pi * Q).sum(axis=1)
        w = awr_weights(Q[s, a] - v[s], beta, clip)
        wsum = np.zeros((n_states, n_actions))
        np.add.at(wsum, (s, a), w)
        pi = np.where(has_data, wsum / np.maximum(wsum.sum(axis=1, keepdims=True), 1e-300), 1.0 / n_actions)
    return pi


if __name__ == "__main__":
    pi_old = np.array([0.5, 0.3, 0.2])
    adv = np.array([1.0, 0.0, 2.0])
    print("improved policy, beta=1:", np.round(kl_improved_policy(pi_old, adv, 1.0), 4))
    print("AWR weights, beta=0.5, clip=20:", np.round(awr_weights(adv, 0.5, 20.0), 3))

    rng = np.random.default_rng(0)
    q = rng.normal(size=64)
    eta = mpo_temperature(q, 0.1)
    print(f"MPO temperature for eps=0.1: {eta:.4f}, KL of weights: {weight_kl(mpo_weights(q, eta)):.4f}")

    mdp = rlenvs.cliff_walking(gamma=0.9)
    S, A = mdp.n_states, mdp.n_actions
    start = mdp.names["start"]
    data = make_dataset(mdp, cautious_behavior(mdp), n_episodes=100, seed=0)
    bc = behavior_cloning(data, S, A)
    print(f"behavior cloning value at the start: {rlenvs.policy_evaluation(mdp, bc)[start]:.3f}")
    for beta in (3.0, 1.0, 0.3):
        pi = awac_tabular(data, S, A, mdp.gamma, beta)
        print(f"AWAC beta={beta:<4} value at the start: {rlenvs.policy_evaluation(mdp, pi)[start]:.3f}")
