"""Reference solution for the inverse reinforcement learning exercise."""

import numpy as np

import rlenvs


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    top = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(top, axis=axis) + np.log(np.sum(np.exp(x - top), axis=axis))


def soft_value_iteration(P: np.ndarray, r: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    S, A, _ = P.shape
    V = np.zeros((horizon + 1, S))
    pi = np.zeros((horizon, S, A))
    for t in range(horizon - 1, -1, -1):
        Q = r[:, None] + P @ V[t + 1]
        V[t] = _logsumexp(Q, axis=1)
        pi[t] = np.exp(Q - V[t][:, None])
    return V, pi


def state_visitation(P: np.ndarray, pi: np.ndarray, mu0: np.ndarray) -> np.ndarray:
    horizon, S, _ = pi.shape
    D = np.zeros((horizon, S))
    D[0] = mu0
    for t in range(horizon - 1):
        D[t + 1] = np.einsum("s,sa,sap->p", D[t], pi[t], P)
    return D


def feature_expectations(D: np.ndarray, features: np.ndarray) -> np.ndarray:
    return D.sum(axis=0) @ features


def empirical_feature_counts(states: np.ndarray, features: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=int)
    return features[states].sum(axis=1).mean(axis=0)


def maxent_irl_gradient(
    P: np.ndarray,
    features: np.ndarray,
    mu0: np.ndarray,
    horizon: int,
    theta: np.ndarray,
    f_expert: np.ndarray,
) -> np.ndarray:
    r = features @ theta
    _, pi = soft_value_iteration(P, r, horizon)
    D = state_visitation(P, pi, mu0)
    return f_expert - feature_expectations(D, features)


def maxent_irl(
    P: np.ndarray,
    features: np.ndarray,
    mu0: np.ndarray,
    horizon: int,
    f_expert: np.ndarray,
    lr: float = 0.2,
    iters: int = 1000,
) -> np.ndarray:
    theta = np.zeros(features.shape[1])
    for _ in range(iters):
        theta = theta + lr * maxent_irl_gradient(P, features, mu0, horizon, theta, f_expert)
    return theta


def _log_sigmoid(z: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -z)


def discriminator_loss(logits: np.ndarray, rho_expert: np.ndarray, rho_policy: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=float)
    # D = sigmoid(logits) is the probability that (s, a) came from the expert.
    return float(
        -np.sum(rho_expert * _log_sigmoid(logits)) - np.sum(rho_policy * _log_sigmoid(-logits))
    )


def discriminator_grad(logits: np.ndarray, rho_expert: np.ndarray, rho_policy: np.ndarray) -> np.ndarray:
    D = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float)))
    return rho_policy * D - rho_expert * (1.0 - D)


def gail_reward(logits: np.ndarray) -> np.ndarray:
    # -log(1 - sigmoid(l)) = log(1 + exp(l)), computed without overflow.
    return np.logaddexp(0.0, np.asarray(logits, dtype=float))


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
