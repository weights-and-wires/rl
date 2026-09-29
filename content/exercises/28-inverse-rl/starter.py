"""Inverse reinforcement learning: maximum-entropy IRL and a GAIL discriminator.

Fill in the functions below. The first five build maximum-entropy IRL on a
small grid world: a backward pass (soft value iteration), a forward pass
(state visitation), feature counts, the log-likelihood gradient, and the
gradient-ascent loop. The last three build the GAIL discriminator over a
table of state-action pairs.

Shapes: S states, A actions, K features, T the horizon. `P` is the transition
array `(S, A, S)` from an `rlenvs.TabularMDP`, and `features` is `(S, K)`.
The reward of a state is `r = features @ theta`, paid on every step spent in
that state, for T steps, with no discount.
"""

import numpy as np

import rlenvs


def soft_value_iteration(P: np.ndarray, r: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """The backward pass: soft values and the maximum-entropy policy over a finite horizon.

    Q_t(s, a) = r(s) + sum_s2 P[s, a, s2] V_{t+1}(s2), with V_T = 0.
    V_t(s) = log sum_a exp(Q_t(s, a)), and pi_t(a | s) = exp(Q_t(s, a) - V_t(s)).

    Args:
        P: transitions, shape (S, A, S).
        r: reward of each state, shape (S,).
        horizon: T, the number of steps.

    Returns:
        (V, pi): V has shape (T + 1, S) with V[T] = 0, and pi has shape
        (T, S, A) with every pi[t, s] summing to one.
    """
    raise NotImplementedError("soft_value_iteration")


def state_visitation(P: np.ndarray, pi: np.ndarray, mu0: np.ndarray) -> np.ndarray:
    """The forward pass: the probability of being in each state at each step.

    D[0] = mu0, and D[t + 1, s2] = sum_{s, a} D[t, s] pi[t, s, a] P[s, a, s2].

    Args:
        P: transitions, shape (S, A, S).
        pi: a time-indexed policy, shape (T, S, A).
        mu0: the start distribution, shape (S,).

    Returns:
        D, shape (T, S). Each row sums to one.
    """
    raise NotImplementedError("state_visitation")


def feature_expectations(D: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Expected feature counts over the horizon: sum_t sum_s D[t, s] features[s].

    Args:
        D: state visitation, shape (T, S).
        features: shape (S, K).

    Returns:
        Shape (K,).
    """
    raise NotImplementedError("feature_expectations")


def empirical_feature_counts(states: np.ndarray, features: np.ndarray) -> np.ndarray:
    """The expert's feature counts: the mean over trajectories of sum_t features[s_t].

    Args:
        states: integer states of N demonstrations, shape (N, T).
        features: shape (S, K).

    Returns:
        Shape (K,).
    """
    raise NotImplementedError("empirical_feature_counts")


def maxent_irl_gradient(
    P: np.ndarray,
    features: np.ndarray,
    mu0: np.ndarray,
    horizon: int,
    theta: np.ndarray,
    f_expert: np.ndarray,
) -> np.ndarray:
    """Gradient of the demonstrations' mean log-likelihood with respect to theta.

    It is the expert's feature counts minus the counts the soft-optimal policy
    for r = features @ theta expects: run the backward pass, then the forward
    pass, then subtract.

    Args:
        P: transitions, shape (S, A, S). The checks use deterministic worlds.
        features: shape (S, K).
        mu0: start distribution, shape (S,).
        horizon: T.
        theta: reward weights, shape (K,).
        f_expert: the expert's feature counts, shape (K,).

    Returns:
        Shape (K,).
    """
    raise NotImplementedError("maxent_irl_gradient")


def maxent_irl(
    P: np.ndarray,
    features: np.ndarray,
    mu0: np.ndarray,
    horizon: int,
    f_expert: np.ndarray,
    lr: float = 0.2,
    iters: int = 1000,
) -> np.ndarray:
    """Gradient ascent on the log-likelihood from theta = 0.

    Each step is theta <- theta + lr * maxent_irl_gradient(...).

    Returns:
        The learned theta, shape (K,).
    """
    raise NotImplementedError("maxent_irl")


def discriminator_loss(logits: np.ndarray, rho_expert: np.ndarray, rho_policy: np.ndarray) -> float:
    """The GAIL discriminator's cross-entropy loss over a table of state-action pairs.

    D = sigmoid(logits) is the probability that a pair came from the expert.
    loss = -sum rho_expert * log D - sum rho_policy * log(1 - D).

    Args:
        logits: one logit per pair, any shape.
        rho_expert: the expert's occupancy of each pair, same shape, sums to one.
        rho_policy: the learner's occupancy of each pair, same shape, sums to one.

    Returns:
        The scalar loss. Compute log sigmoid without overflow.
    """
    raise NotImplementedError("discriminator_loss")


def discriminator_grad(logits: np.ndarray, rho_expert: np.ndarray, rho_policy: np.ndarray) -> np.ndarray:
    """Gradient of discriminator_loss with respect to the logits. Same shape as logits."""
    raise NotImplementedError("discriminator_grad")


def gail_reward(logits: np.ndarray) -> np.ndarray:
    """The reward GAIL gives the learner: -log(1 - D), with D = sigmoid(logits).

    Must stay finite for logits of size 800 in either direction.
    """
    raise NotImplementedError("gail_reward")


if __name__ == "__main__":
    mdp = rlenvs.gridworld(["S...", ".#..", "...."])
    S, cells = mdp.n_states, mdp.names["cells"]
    features = np.eye(S)  # one indicator feature per cell
    horizon = 8

    # A hidden expert: each step in the bottom-right corner is worth 3.
    theta_true = np.zeros(S)
    theta_true[cells.index((2, 3))] = 3.0
    _, pi_true = soft_value_iteration(mdp.P, features @ theta_true, horizon)

    # Record 500 demonstrations from the expert.
    rng = np.random.default_rng(0)
    states = np.zeros((500, horizon), dtype=int)
    for i in range(500):
        s = mdp.names["start"]
        for t in range(horizon):
            states[i, t] = s
            a = rng.choice(4, p=pi_true[t, s])
            s = rng.choice(S, p=mdp.P[s, a])
    f_expert = empirical_feature_counts(states, features)

    theta = maxent_irl(mdp.P, features, mdp.mu0, horizon, f_expert, lr=0.2, iters=1000)
    _, pi_learned = soft_value_iteration(mdp.P, features @ theta, horizon)
    f_learned = feature_expectations(state_visitation(mdp.P, pi_learned, mdp.mu0), features)
    print("cell     expert visits  learned visits  learned reward")
    for k, cell in enumerate(cells):
        print(f"{str(cell):8} {f_expert[k]:13.3f}  {f_learned[k]:14.3f}  {theta[k]:14.3f}")

    # A tabular GAIL discriminator over four state-action pairs.
    rho_e = np.array([0.5, 0.3, 0.2, 0.0])
    rho_p = np.array([0.1, 0.2, 0.3, 0.4])
    logits = np.zeros(4)
    for _ in range(2000):
        logits -= 2.0 * discriminator_grad(logits, rho_e, rho_p)
    print("trained D:           ", np.round(1 / (1 + np.exp(-logits)), 3))
    print("best D, rho_E / (rho_E + rho_pi):", np.round(rho_e / (rho_e + rho_p), 3))
    print("GAIL reward -log(1 - D):", np.round(gail_reward(logits), 3))
