"""Offline reinforcement learning: extrapolation error, CQL, and IQL in a table.

You learn from a fixed dataset of transitions and never touch the
environment. The dataset is a dict of equal-length arrays:

    data["s"], data["a"]   integer state and action, shape (N,)
    data["r"]              reward, shape (N,)
    data["s2"]             integer next state, shape (N,)
    data["done"]           1.0 where s2 is terminal, else 0.0, shape (N,)

The helpers `make_dataset`, `action_counts`, and `cautious_behavior` are
written for you. Fill in the functions after them. Every Q table has shape
(S, A) and starts at zero.
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


def offline_q_iteration(data: dict, n_states: int, n_actions: int, gamma: float, iters: int = 500) -> np.ndarray:
    """Q-learning on a fixed dataset, run to its fixed point.

    Each iteration computes every transition's target
    y = r + gamma * (1 - done) * max_a2 Q[s2, a2], then sets Q[s, a] to the
    mean target over the transitions with that (s, a). Pairs that never appear
    in the data keep their starting value of 0.

    Returns:
        Q, shape (S, A).
    """
    raise NotImplementedError("offline_q_iteration")


def cql_penalty(Q: np.ndarray, s: np.ndarray, a: np.ndarray) -> float:
    """The CQL penalty: the batch mean of logsumexp_a2 Q[s, a2] - Q[s, a].

    Args:
        Q: shape (S, A).
        s, a: the batch's states and actions, shape (N,).

    Returns:
        A float. Subtract each row's max before exponentiating.
    """
    raise NotImplementedError("cql_penalty")


def cql_penalty_grad(Q: np.ndarray, s: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Gradient of cql_penalty with respect to Q, shape (S, A)."""
    raise NotImplementedError("cql_penalty_grad")


def cql_q_iteration(
    data: dict,
    n_states: int,
    n_actions: int,
    gamma: float,
    alpha: float = 0.5,
    iters: int = 1000,
    lr: float = 1.0,
) -> np.ndarray:
    """Conservative Q-learning in a table, by preconditioned gradient descent.

    Each iteration:
      1. computes the targets y = r + gamma * (1 - done) * max_a2 Q[s2, a2],
         held fixed for this step;
      2. takes the gradient of alpha * cql_penalty + (1/2) mean (Q[s, a] - y)^2
         with respect to Q;
      3. divides each visited state's row of that gradient by the fraction of
         transitions that start in that state;
      4. sets Q <- Q - lr * gradient.

    Returns:
        Q, shape (S, A).
    """
    raise NotImplementedError("cql_q_iteration")


def expectile_loss(u: np.ndarray, tau: float) -> float:
    """The expectile loss: the mean of |tau - 1(u < 0)| * u^2 over the elements of u."""
    raise NotImplementedError("expectile_loss")


def expectile_loss_grad(u: np.ndarray, tau: float) -> np.ndarray:
    """Gradient of expectile_loss with respect to u, the same shape as u."""
    raise NotImplementedError("expectile_loss_grad")


def expectile(values: np.ndarray, weights: np.ndarray, tau: float, iters: int = 20) -> np.ndarray:
    """The weighted tau-expectile along the last axis.

    It is the v that minimises sum_i weights_i * |tau - 1(values_i < v)| * (values_i - v)^2.
    Start from the weighted mean and repeat: give each value the weight
    weights_i * (tau if values_i >= v else 1 - tau), and set v to the
    weighted mean under those weights.

    Args:
        values: shape (..., n).
        weights: non-negative, same shape as values.
        tau: in (0, 1).

    Returns:
        Shape (...). Rows whose weights are all zero get 0.
    """
    raise NotImplementedError("expectile")


def iql_values(
    data: dict,
    n_states: int,
    n_actions: int,
    gamma: float,
    tau: float,
    iters: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Implicit Q-learning in a table.

    Each iteration sets V[s] to the tau-expectile of Q[s, :] weighted by the
    action counts at s, then sets Q[s, a] to the mean over the data of
    r + gamma * (1 - done) * V[s2]. No max over actions appears anywhere.

    Returns:
        (Q, V) with shapes (S, A) and (S,). Unseen pairs have Q = 0.
    """
    raise NotImplementedError("iql_values")


def iql_policy(Q: np.ndarray, V: np.ndarray, counts: np.ndarray, beta: float) -> np.ndarray:
    """Advantage-weighted policy extraction on the data's actions.

    pi(a | s) is proportional to counts[s, a] * exp(beta * (Q[s, a] - V[s])).
    Unseen actions get probability 0. States with no data get a uniform row.
    Subtract each row's largest advantage before exponentiating.

    Returns:
        pi, shape (S, A), rows summing to one.
    """
    raise NotImplementedError("iql_policy")


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
    counts = action_counts(data, S, A)
    for tau in (0.5, 0.7, 0.9, 0.99):
        Q, V = iql_values(data, S, A, mdp.gamma, tau)
        pi = iql_policy(Q, V, counts, beta=10.0)
        v_pi = rlenvs.policy_evaluation(mdp, pi)[start]
        print(f"IQL tau={tau:<4}  V(start)={V[start]:7.3f}  policy value={v_pi:7.3f}")
