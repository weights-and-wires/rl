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


def q_learning_updates(
    Q0: np.ndarray,
    transitions: dict,
    alpha: float,
    gamma: float,
    phi: np.ndarray | None = None,
) -> np.ndarray:
    Q = np.array(Q0, dtype=float)
    for s, a, r, s2, done in zip(
        transitions["s"], transitions["a"], transitions["r"], transitions["s2"], transitions["done"]
    ):
        if phi is not None:
            r = r + gamma * phi[s2] - phi[s]
        target = r + (0.0 if done else gamma * Q[s2].max())
        Q[s, a] += alpha * (target - Q[s, a])
    return Q


def run_q_learning(
    mdp,
    episodes: int,
    alpha: float,
    epsilon: float,
    seed: int,
    phi: np.ndarray | None = None,
    max_steps: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    Q = np.zeros((mdp.n_states, mdp.n_actions))
    steps = np.zeros(episodes, dtype=int)
    for ep in range(episodes):
        s = mdp.reset(seed=int(rng.integers(2**31)))
        for t in range(max_steps):
            if rng.random() < epsilon:
                a = int(rng.integers(mdp.n_actions))
            else:
                best = np.flatnonzero(Q[s] == Q[s].max())
                a = int(rng.choice(best))
            s2, r, terminated, truncated = mdp.step(a)
            if phi is not None:
                r = r + mdp.gamma * phi[s2] - phi[s]
            target = r + (0.0 if terminated else mdp.gamma * Q[s2].max())
            Q[s, a] += alpha * (target - Q[s, a])
            s = s2
            if terminated:
                break
        steps[ep] = t + 1
    return Q, steps


def distance_potential(mdp) -> np.ndarray:
    """Phi(s) = minus the Manhattan distance from s to the goal, which is 0 at the goal."""
    cells = mdp.names["cells"]
    grid = mdp.names["grid"]
    goal = next(c for c in cells if grid[c[0]][c[1]] == "G")
    return -np.array([abs(i - goal[0]) + abs(j - goal[1]) for i, j in cells], dtype=float)


if __name__ == "__main__":
    layout = [
        "S.....",
        ".####.",
        "......",
        ".####.",
        ".....G",
    ]
    mdp = rlenvs.gridworld(layout, gamma=0.95)
    phi = distance_potential(mdp)
    q = rlenvs.q_from_v(mdp, rlenvs.value_iteration(mdp))
    q_shaped = rlenvs.q_from_v(shaped_mdp(mdp, phi), rlenvs.value_iteration(shaped_mdp(mdp, phi)))
    print("max |Q*_shaped - (Q* - Phi)| =", f"{np.abs(q_shaped - (q - phi[:, None])).max():.2e}")
    print("same optimal actions:", bool(np.array_equal(optimal_actions(q), optimal_actions(q_shaped))))

    print("\nQ-learning, steps per episode (mean of 5 seeds)")
    plain = np.mean([run_q_learning(mdp, 30, 0.5, 0.1, seed)[1] for seed in range(5)], axis=0)
    shaped = np.mean([run_q_learning(mdp, 30, 0.5, 0.1, seed, phi)[1] for seed in range(5)], axis=0)
    for ep in (0, 1, 2, 5, 10, 29):
        print(f"  episode {ep + 1:2d}: plain {plain[ep]:6.1f}   shaped {shaped[ep]:6.1f}")

    # A bonus of 2 for entering the cell next to the start is not potential-based.
    bonus = bonus_mdp(mdp, mdp.names["cells"].index((0, 1)), 2.0)
    q_bonus = rlenvs.q_from_v(bonus, rlenvs.value_iteration(bonus))
    changed = np.flatnonzero(~np.all(optimal_actions(q) == optimal_actions(q_bonus), axis=1))
    print(f"\nwith a +2 bonus next to the start, {len(changed)} states change their optimal actions")
