"""The deadly triad: the TD fixed point, LSTD, and Baird's counterexample.

Fill in the functions below, in order. A value estimate is linear in features:
v(s) = w @ x(s). `stationary_distribution` and `value_error` are already
written for you, because you wrote them in the previous chapter.
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


def value_error(Phi: np.ndarray, w: np.ndarray, v: np.ndarray, d: np.ndarray) -> float:
    """The d-weighted mean squared gap between Phi @ w and v. Already written for you."""
    return float(d @ (Phi @ w - v) ** 2)


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
