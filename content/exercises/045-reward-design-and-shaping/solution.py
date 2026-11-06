"""Reference solution for the reward design exercise."""

import numpy as np

import rlenvs


def full_rewards(mdp) -> np.ndarray:
    """The reward for each (s, a, s2), shape (S, A, S)."""
    if mdp.R3 is not None:
        return np.array(mdp.R3, dtype=float)
    return np.broadcast_to(mdp.R[:, :, None], mdp.P.shape).astype(float)


def with_rewards(mdp, R3: np.ndarray):
    """A copy of `mdp` that pays `R3[s, a, s2]`, with `R` recomputed as its expectation."""
    return rlenvs.TabularMDP(
        P=mdp.P,
        R=(mdp.P * R3).sum(axis=2),
        gamma=mdp.gamma,
        mu0=mdp.mu0,
        terminal=mdp.terminal,
        R3=R3,
        max_steps=mdp.max_steps,
        names=dict(mdp.names),
    )


def shaped_mdp(mdp, phi: np.ndarray):
    phi = np.asarray(phi, dtype=float)
    F = mdp.gamma * phi[None, None, :] - phi[:, None, None]
    return with_rewards(mdp, full_rewards(mdp) + F)


def bonus_mdp(mdp, state: int, bonus: float):
    R3 = full_rewards(mdp).copy()
    R3[:, :, state] += bonus
    return with_rewards(mdp, R3)


def optimal_actions(q: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return q >= q.max(axis=1, keepdims=True) - tol


if __name__ == "__main__":
    # A five-state chain: entering state 4, the goal, pays 1.
    P = np.zeros((5, 2, 5))
    for s in range(4):
        P[s, 0, max(s - 1, 0)] = 1.0
        P[s, 1, s + 1] = 1.0
    P[4, :, 4] = 1.0
    R3 = np.zeros((5, 2, 5))
    R3[:4, :, 4] = 1.0
    mu0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    terminal = np.array([False, False, False, False, True])
    mdp = rlenvs.TabularMDP(P=P, R=(P * R3).sum(axis=2), gamma=0.9, mu0=mu0, terminal=terminal, R3=R3)

    def qstar(m):
        return rlenvs.q_from_v(m, rlenvs.value_iteration(m))

    q = qstar(mdp)
    phi = np.array([0.0, 1.0, 2.0, 3.0, 0.0])
    q_shaped = qstar(shaped_mdp(mdp, phi))
    print("max |Q*_shaped - (Q* - Phi)| =", f"{np.abs(q_shaped - (q - phi[:, None]))[:4].max():.2e}")
    print("same optimal actions:",
          bool(np.array_equal(optimal_actions(q)[:4], optimal_actions(q_shaped)[:4])))

    q_bonus = qstar(bonus_mdp(mdp, 1, 0.3))
    print("Q* at state 0 without the bonus:", np.round(q[0], 4))
    print("Q* at state 0 with +0.3 for entering state 1:", np.round(q_bonus[0], 4))
