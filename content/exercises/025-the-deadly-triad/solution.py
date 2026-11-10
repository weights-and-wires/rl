"""Reference solution for the deadly triad exercise."""

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


def value_error(Phi: np.ndarray, w: np.ndarray, v: np.ndarray, d: np.ndarray) -> float:
    """The d-weighted mean squared gap between Phi @ w and v. Already written for you."""
    return float(d @ (Phi @ w - v) ** 2)


def td_fixed_point(
    Phi: np.ndarray, P_pi: np.ndarray, r_pi: np.ndarray, d: np.ndarray, gamma: float
) -> np.ndarray:
    D = np.diag(d)
    A = Phi.T @ D @ (np.eye(len(d)) - gamma * P_pi) @ Phi
    b = Phi.T @ D @ r_pi
    return np.linalg.solve(A, b)


def lstd(
    X: np.ndarray,
    rewards: np.ndarray,
    X2: np.ndarray,
    terminated: np.ndarray,
    gamma: float,
    reg: float = 1e-8,
) -> np.ndarray:
    live = 1.0 - np.asarray(terminated, dtype=float)
    A = X.T @ (X - gamma * live[:, None] * X2) + reg * np.eye(X.shape[1])
    b = X.T @ np.asarray(rewards, dtype=float)
    return np.linalg.solve(A, b)


def baird_features() -> np.ndarray:
    Phi = np.zeros((7, 8))
    for i in range(6):
        Phi[i, i] = 2.0
        Phi[i, 7] = 1.0
    Phi[6, 6] = 1.0
    Phi[6, 7] = 2.0
    return Phi


def baird_expected_update(
    w: np.ndarray, alpha: float, gamma: float, weights: np.ndarray
) -> np.ndarray:
    Phi = baird_features()
    v = Phi @ w
    # The target policy always takes the solid action, which lands in state 6
    # with reward 0, so every state's expected target is gamma * v[6].
    delta = gamma * v[6] - v
    return w + alpha * Phi.T @ (weights * delta)


if __name__ == "__main__":
    # 1. The TD fixed point against the best possible fit, on a random MDP.
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
    ve_td = value_error(Phi, w_td, v, d)
    ve_best = value_error(Phi, w_best, v, d)
    print(f"VE at the TD fixed point {ve_td:.4f}, best possible {ve_best:.4f}, "
          f"ratio {ve_td / ve_best:.2f} (bound {1 / (1 - mdp.gamma):.0f})")

    # 2. LSTD from one long trajectory.
    s, rows = 0, []
    for _ in range(5000):
        a = rng.choice(2, p=pi[s])
        s2, r = mdp.sample(s, a, rng)
        rows.append((s, r, s2))
        s = s2
    S, R, S2 = (np.array(c) for c in zip(*rows))
    w_lstd = lstd(Phi[S], R, Phi[S2], np.zeros(len(S), dtype=bool), mdp.gamma)
    print(f"LSTD weights {np.round(w_lstd, 3)}, TD fixed point {np.round(w_td, 3)}")

    # 3. Baird's counterexample: uniform updates diverge, on-policy updates do not.
    for name, weights in (("uniform", np.full(7, 1 / 7)), ("on-policy", np.eye(7)[6])):
        w = np.array([1, 1, 1, 1, 1, 1, 10, 1], dtype=float)
        for _ in range(1000):
            w = baird_expected_update(w, alpha=0.01, gamma=0.99, weights=weights)
        print(f"Baird, {name} weighting: |w| after 1000 sweeps = {np.linalg.norm(w):.1f}")
