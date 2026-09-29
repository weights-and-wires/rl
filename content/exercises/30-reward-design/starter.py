"""Reward design: potential-based shaping, and a bonus that breaks the task.

Fill in the functions after the helpers. The helpers `full_rewards`,
`with_rewards`, and `distance_potential` are written for you.

A potential `phi` is an array of shape (S,), one number per state. Every MDP
here is an `rlenvs.TabularMDP`. To get its optimal action values, use
`rlenvs.q_from_v(mdp, rlenvs.value_iteration(mdp))`.
"""

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


def distance_potential(mdp) -> np.ndarray:
    """Phi(s) = minus the Manhattan distance from s to the goal, which is 0 at the goal."""
    cells = mdp.names["cells"]
    grid = mdp.names["grid"]
    goal = next(c for c in cells if grid[c[0]][c[1]] == "G")
    return -np.array([abs(i - goal[0]) + abs(j - goal[1]) for i, j in cells], dtype=float)


def shaped_mdp(mdp, phi: np.ndarray):
    """A copy of `mdp` whose every transition also pays F = gamma * phi[s2] - phi[s].

    Build the (S, A, S) reward table with `full_rewards`, add F to entry
    [s, a, s2], and return `with_rewards(mdp, ...)`. Don't modify `mdp`.
    """
    raise NotImplementedError("shaped_mdp")


def bonus_mdp(mdp, state: int, bonus: float):
    """A copy of `mdp` that pays an extra `bonus` on every transition into `state`."""
    raise NotImplementedError("bonus_mdp")


def optimal_actions(q: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """A boolean mask, shape (S, A): True where q[s, a] is within tol of max_a q[s, a]."""
    raise NotImplementedError("optimal_actions")


def q_learning_updates(
    Q0: np.ndarray,
    transitions: dict,
    alpha: float,
    gamma: float,
    phi: np.ndarray | None = None,
) -> np.ndarray:
    """Tabular Q-learning over a fixed sequence of transitions, in order.

    `transitions` holds equal-length arrays "s", "a", "r", "s2", and "done".
    For each one, the reward is r, plus gamma * phi[s2] - phi[s] when phi is
    given. The target adds gamma * max_a2 Q[s2, a2] unless done is 1. Then
    Q[s, a] moves alpha of the way toward the target.

    Returns:
        The final Q, shape (S, A). Work on a copy of Q0.
    """
    raise NotImplementedError("q_learning_updates")


def run_q_learning(
    mdp,
    episodes: int,
    alpha: float,
    epsilon: float,
    seed: int,
    phi: np.ndarray | None = None,
    max_steps: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Epsilon-greedy Q-learning on `mdp`, optionally with shaping by `phi`.

    Make one generator, rng = np.random.default_rng(seed). Start each episode
    with mdp.reset(seed=int(rng.integers(2**31))). With probability epsilon
    take a random action; otherwise take a greedy one, breaking ties uniformly
    at random. Use the same update as q_learning_updates, with the episode's
    `terminated` flag as done. Q starts at zero.

    Returns:
        (Q, steps): the final Q, shape (S, A), and the number of steps each
        episode took, shape (episodes,).
    """
    raise NotImplementedError("run_q_learning")


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
