"""Reference solution for the value-function approximation exercise."""

import numpy as np

import rlenvs

def stationary_distribution(P_pi: np.ndarray) -> np.ndarray:
    """The stationary distribution mu of a state-to-state matrix, with mu P_pi = mu.

    Already written for you. Solves (P_pi^T - I) mu = 0 together with
    sum(mu) = 1 by least squares, which is exact when the chain has a single
    stationary distribution.
    """
    S = P_pi.shape[0]
    M = np.vstack([P_pi.T - np.eye(S), np.ones((1, S))])
    rhs = np.zeros(S + 1)
    rhs[-1] = 1.0
    mu = np.linalg.lstsq(M, rhs, rcond=None)[0]
    return mu / mu.sum()


def tile_features(x: float, low: float, high: float, n_tiles: int, n_tilings: int) -> np.ndarray:
    width = (high - low) / n_tiles
    x = min(max(float(x), low), high)
    per_tiling = n_tiles + 1
    features = np.zeros(n_tilings * per_tiling)
    for k in range(n_tilings):
        # Tiling k is shifted left by k / n_tilings of a tile width.
        tile = int(np.floor((x - low) / width + k / n_tilings))
        features[k * per_tiling + tile] = 1.0
    return features


def semi_gradient_td_update(
    w: np.ndarray,
    x: np.ndarray,
    r: float,
    x2: np.ndarray,
    terminated: bool,
    alpha: float,
    gamma: float,
) -> np.ndarray:
    # The target r + gamma * v(s2) is treated as a constant: no gradient through it.
    target = r + gamma * (0.0 if terminated else float(w @ x2))
    delta = target - float(w @ x)
    return w + alpha * delta * x


def value_error(Phi: np.ndarray, w: np.ndarray, v: np.ndarray, d: np.ndarray) -> float:
    return float(d @ (Phi @ w - v) ** 2)


if __name__ == "__main__":
    # 1. Tile coding: nearby points share tiles, far points share none.
    for x in (0.30, 0.40, 0.90):
        active = np.flatnonzero(tile_features(x, 0.0, 1.0, n_tiles=4, n_tilings=2))
        print(f"x = {x:.2f}: active tiles {active}")

    # 2. Semi-gradient TD on a random MDP, with three random features per state.
    mdp = rlenvs.random_mdp(n_states=6, n_actions=2, gamma=0.9, seed=3)
    rng = np.random.default_rng(0)
    pi = rng.dirichlet(np.ones(2), size=6)
    P_pi = np.einsum("sa,sat->st", pi, mdp.P)
    d = stationary_distribution(P_pi)
    Phi = rng.standard_normal((6, 3))
    v = rlenvs.policy_evaluation(mdp, pi)
    w = np.zeros(3)
    print(f"value error before learning: {value_error(Phi, w, v, d):.4f}")
    s = 0
    for t in range(20000):
        a = rng.choice(2, p=pi[s])
        s2, r = mdp.sample(s, a, rng)
        w = semi_gradient_td_update(w, Phi[s], r, Phi[s2], False, 0.1 / (1 + t / 1000), mdp.gamma)
        s = s2
    print(f"value error after 20,000 steps: {value_error(Phi, w, v, d):.4f}")
