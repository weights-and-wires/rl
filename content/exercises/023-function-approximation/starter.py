"""Value-function approximation: tile coding and semi-gradient TD.

Fill in the functions below, in order. A value estimate here is always linear
in features: v(s) = w @ x(s), where x(s) is the feature vector of state s and w
is the weight vector you learn.

`stationary_distribution` is already written for you. It gives the fraction of
time a policy spends in each state, which is how `value_error` weights states.
"""

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
    """Binary tile-coding features for a scalar input.

    The range [low, high] is cut into n_tiles tiles of width
    (high - low) / n_tiles. There are n_tilings copies of that grid. Copy k,
    for k = 0, ..., n_tilings - 1, is shifted left by k / n_tilings of a tile
    width, so each copy needs n_tiles + 1 tiles to cover the range. In copy k,
    x falls in tile floor((x - low) / width + k / n_tilings).

    Clip x to [low, high] first.

    Args:
        x: the input, a float.
        low, high: the range the tiles cover.
        n_tiles: tiles per tiling across the range.
        n_tilings: number of shifted tilings.

    Returns:
        A float array of shape (n_tilings * (n_tiles + 1),) with exactly one 1
        per tiling. Tiling k owns entries k * (n_tiles + 1) up to
        (k + 1) * (n_tiles + 1) - 1.
    """
    raise NotImplementedError("tile_features")


def semi_gradient_td_update(
    w: np.ndarray,
    x: np.ndarray,
    r: float,
    x2: np.ndarray,
    terminated: bool,
    alpha: float,
    gamma: float,
) -> np.ndarray:
    """One linear semi-gradient TD(0) step.

    Args:
        w: weights, shape (d,).
        x: features of the state s_t, shape (d,).
        r: the reward r_t for the step from s_t.
        x2: features of the next state s_{t+1}, shape (d,).
        terminated: true when s_{t+1} is terminal, so its value counts as 0.
        alpha: step size.
        gamma: discount.

    Returns:
        The updated weights, shape (d,).
    """
    raise NotImplementedError("semi_gradient_td_update")


def value_error(Phi: np.ndarray, w: np.ndarray, v: np.ndarray, d: np.ndarray) -> float:
    """The mean squared value error of linear estimates against the true values.

    Args:
        Phi: feature matrix, shape (S, d). Row s is x(s).
        w: weights, shape (d,).
        v: the true values V^pi, shape (S,).
        d: how much each state counts, shape (S,), summing to 1.

    Returns:
        The float sum over states of d[s] * (v[s] - Phi[s] @ w) ** 2.
    """
    raise NotImplementedError("value_error")


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
