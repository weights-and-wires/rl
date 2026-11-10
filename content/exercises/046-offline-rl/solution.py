"""Reference solution for the offline reinforcement learning exercise."""

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


def _mean_per_pair(data: dict, values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    total = np.zeros_like(counts)
    np.add.at(total, (data["s"], data["a"]), values)
    return np.divide(total, counts, out=np.zeros_like(total), where=counts > 0)


def offline_q_iteration(data: dict, n_states: int, n_actions: int, gamma: float, iters: int = 500) -> np.ndarray:
    counts = action_counts(data, n_states, n_actions)
    seen = counts > 0
    Q = np.zeros((n_states, n_actions))
    for _ in range(iters):
        y = data["r"] + gamma * (1.0 - data["done"]) * Q[data["s2"]].max(axis=1)
        Q = np.where(seen, _mean_per_pair(data, y, counts), Q)
    return Q


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def cql_penalty(Q: np.ndarray, s: np.ndarray, a: np.ndarray) -> float:
    rows = Q[s]
    top = rows.max(axis=1)
    lse = top + np.log(np.exp(rows - top[:, None]).sum(axis=1))
    return float(np.mean(lse - Q[s, a]))


def cql_penalty_grad(Q: np.ndarray, s: np.ndarray, a: np.ndarray) -> np.ndarray:
    n = len(s)
    grad = np.zeros_like(Q, dtype=float)
    np.add.at(grad, s, _softmax(Q[s]) / n)
    np.add.at(grad, (s, a), -1.0 / n)
    return grad


def cql_q_iteration(
    data: dict,
    n_states: int,
    n_actions: int,
    gamma: float,
    alpha: float = 0.5,
    iters: int = 1000,
    lr: float = 1.0,
) -> np.ndarray:
    s, a = data["s"], data["a"]
    n = len(s)
    counts = action_counts(data, n_states, n_actions)
    state_share = counts.sum(axis=1) / n
    visited = state_share > 0
    Q = np.zeros((n_states, n_actions))
    for _ in range(iters):
        y = data["r"] + gamma * (1.0 - data["done"]) * Q[data["s2"]].max(axis=1)
        td_grad = np.zeros_like(Q)
        np.add.at(td_grad, (s, a), (Q[s, a] - y) / n)
        grad = alpha * cql_penalty_grad(Q, s, a) + td_grad
        # Divide each visited state's row by its share of the data, so rare
        # states learn as fast as common ones. The minimiser doesn't move.
        grad[visited] /= state_share[visited, None]
        Q = Q - lr * grad
    return Q


def expectile_loss(u: np.ndarray, tau: float) -> float:
    u = np.asarray(u, dtype=float)
    weight = np.where(u < 0, 1.0 - tau, tau)
    return float(np.mean(weight * u**2))


def expectile_loss_grad(u: np.ndarray, tau: float) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    weight = np.where(u < 0, 1.0 - tau, tau)
    return 2.0 * weight * u / u.size


def expectile(values: np.ndarray, weights: np.ndarray, tau: float, iters: int = 20) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = weights.sum(axis=-1)
    safe = np.where(total > 0, total, 1.0)
    v = (weights * values).sum(axis=-1) / safe
    for _ in range(iters):
        c = weights * np.where(values >= v[..., None], tau, 1.0 - tau)
        v = (c * values).sum(axis=-1) / np.where(c.sum(axis=-1) > 0, c.sum(axis=-1), 1.0)
    return np.where(total > 0, v, 0.0)


def iql_values(
    data: dict,
    n_states: int,
    n_actions: int,
    gamma: float,
    tau: float,
    iters: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    counts = action_counts(data, n_states, n_actions)
    Q = np.zeros((n_states, n_actions))
    V = np.zeros(n_states)
    for _ in range(iters):
        V = expectile(Q, counts, tau)
        y = data["r"] + gamma * (1.0 - data["done"]) * V[data["s2"]]
        Q = _mean_per_pair(data, y, counts)
    return Q, V


def bc_regularized_actor_loss(
    pi_actions: np.ndarray,
    data_actions: np.ndarray,
    q_pi: np.ndarray,
    q_data: np.ndarray,
    alpha: float = 2.5,
) -> float:
    pi_actions = np.asarray(pi_actions, dtype=float)
    data_actions = np.asarray(data_actions, dtype=float)
    q_pi = np.asarray(q_pi, dtype=float)
    q_data = np.asarray(q_data, dtype=float)
    # Fujimoto and Gu, equation 5: λ uses Q at the logged actions and is not differentiated.
    lam = float(alpha) / float(np.mean(np.abs(q_data)))
    return float(np.mean((pi_actions - data_actions) ** 2) - lam * float(np.mean(q_pi)))


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


if __name__ == "__main__":
    mdp = rlenvs.cliff_walking(gamma=0.9)
    S, A = mdp.n_states, mdp.n_actions
    behavior = cautious_behavior(mdp)
    data = make_dataset(mdp, behavior, n_episodes=100, seed=0)
    start = mdp.names["start"]
    print(f"{len(data['s'])} transitions; behaviour value at the start: "
          f"{rlenvs.policy_evaluation(mdp, behavior)[start]:.3f}")

    def value_of(Q):
        return rlenvs.policy_evaluation(mdp, rlenvs.greedy(Q))[start]

    Q_naive = offline_q_iteration(data, S, A, mdp.gamma)
    Q_cql = cql_q_iteration(data, S, A, mdp.gamma, alpha=0.5)
    print(f"offline Q-learning, greedy value: {value_of(Q_naive):8.3f}")
    print(f"CQL,                greedy value: {value_of(Q_cql):8.3f}")
    for tau in (0.5, 0.7, 0.9, 0.99):
        Q, V = iql_values(data, S, A, mdp.gamma, tau)
        print(f"IQL tau={tau:<4}  V(start)={V[start]:7.3f}")

    pi = np.array([0.5, 0.0, 0.1])
    logged = np.array([0.0, 0.2, -0.4])
    loss = bc_regularized_actor_loss(pi, logged, 4.0 * pi, np.array([10.0, -20.0, 30.0]), alpha=2.5)
    print(f"TD3+BC actor loss: {loss:.2f}")
