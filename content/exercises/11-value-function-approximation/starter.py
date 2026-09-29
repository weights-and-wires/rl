"""Value-function approximation: tile coding, semi-gradient TD, LSTD, and Baird.

Fill in the functions below, in order. A value estimate here is always linear
in features: v(s) = w @ x(s), where x(s) is the feature vector of state s and w
is the weight vector you learn.

`stationary_distribution` is already written for you. The checks use the same
computation to weight states by how often the policy visits them.
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


def td_fixed_point(
    Phi: np.ndarray, P_pi: np.ndarray, r_pi: np.ndarray, d: np.ndarray, gamma: float
) -> np.ndarray:
    """The weights linear TD(0) converges to, found by one linear solve.

    Args:
        Phi: feature matrix, shape (S, d). Row s is x(s).
        P_pi: state-to-state transition matrix under the policy, shape (S, S).
        r_pi: expected reward in each state under the policy, shape (S,).
        d: the state weighting, shape (S,), usually the stationary distribution.
        gamma: discount, below 1.

    Returns:
        w_TD, shape (d,).
    """
    raise NotImplementedError("td_fixed_point")


def lstd(
    X: np.ndarray,
    rewards: np.ndarray,
    X2: np.ndarray,
    terminated: np.ndarray,
    gamma: float,
    reg: float = 1e-8,
) -> np.ndarray:
    """Least-squares TD: the TD fixed point of a batch of transitions.

    Args:
        X: features of s_t for each transition, shape (T, d).
        rewards: r_t for each transition, shape (T,).
        X2: features of s_{t+1}, shape (T, d).
        terminated: shape (T,), true where s_{t+1} is terminal.
        gamma: discount.
        reg: a small ridge term added to the diagonal of A, so the solve
            works even when some feature never appears.

    Returns:
        The weights w solving (A + reg I) w = b, shape (d,).
    """
    raise NotImplementedError("lstd")


def baird_features() -> np.ndarray:
    """The feature matrix of Baird's counterexample, shape (7, 8).

    States 0 to 5 are the upper states: state i has value 2 w[i] + w[7].
    State 6 is the lower state, with value w[6] + 2 w[7].
    """
    raise NotImplementedError("baird_features")


def baird_expected_update(
    w: np.ndarray, alpha: float, gamma: float, weights: np.ndarray
) -> np.ndarray:
    """One expected semi-gradient update on Baird's counterexample.

    The target policy always takes the solid action, which moves every state
    to state 6 with reward 0. Every state is updated at once, and state s's
    contribution is scaled by weights[s].

    Args:
        w: weights, shape (8,).
        alpha: step size.
        gamma: discount.
        weights: how much each state counts, shape (7,), summing to 1.
            Uniform weights, 1/7 each, reproduce Sutton and Barto's
            semi-gradient DP update (11.9).

    Returns:
        The updated weights, shape (8,).
    """
    raise NotImplementedError("baird_expected_update")


if __name__ == "__main__":
    # 1. Tile coding: nearby points share tiles, far points share none.
    for x in (0.30, 0.40, 0.90):
        active = np.flatnonzero(tile_features(x, 0.0, 1.0, n_tiles=4, n_tilings=2))
        print(f"x = {x:.2f}: active tiles {active}")

    # 2. The TD fixed point against the best possible fit, on a random MDP.
    mdp = rlenvs.random_mdp(n_states=6, n_actions=2, gamma=0.9, seed=3)
    rng = np.random.default_rng(0)
    pi = rng.dirichlet(np.ones(2), size=6)
    P_pi = np.einsum("sa,sat->st", pi, mdp.P)
    r_pi = (pi * mdp.R).sum(axis=1)
    d = stationary_distribution(P_pi)
    Phi = rng.standard_normal((6, 3))
    v = rlenvs.policy_evaluation(mdp, pi)
    w_td = td_fixed_point(Phi, P_pi, r_pi, d, mdp.gamma)
    w_best = np.linalg.solve(Phi.T @ np.diag(d) @ Phi, Phi.T @ np.diag(d) @ v)
    ve_td = d @ (Phi @ w_td - v) ** 2
    ve_best = d @ (Phi @ w_best - v) ** 2
    print(f"VE at the TD fixed point {ve_td:.4f}, best possible {ve_best:.4f}, "
          f"ratio {ve_td / ve_best:.2f} (bound {1 / (1 - mdp.gamma):.0f})")

    # 3. LSTD from one long trajectory.
    s, rows = 0, []
    for _ in range(5000):
        a = rng.choice(2, p=pi[s])
        s2, r = mdp.sample(s, a, rng)
        rows.append((s, r, s2))
        s = s2
    S, R, S2 = (np.array(c) for c in zip(*rows))
    w_lstd = lstd(Phi[S], R, Phi[S2], np.zeros(len(S), dtype=bool), mdp.gamma)
    print(f"LSTD weights {np.round(w_lstd, 3)}, TD fixed point {np.round(w_td, 3)}")

    # 4. Baird's counterexample: uniform updates diverge, on-policy updates do not.
    for name, weights in (("uniform", np.full(7, 1 / 7)), ("on-policy", np.eye(7)[6])):
        w = np.array([1, 1, 1, 1, 1, 1, 10, 1], dtype=float)
        for _ in range(1000):
            w = baird_expected_update(w, alpha=0.01, gamma=0.99, weights=weights)
        print(f"Baird, {name} weighting: |w| after 1000 sweeps = {np.linalg.norm(w):.1f}")
