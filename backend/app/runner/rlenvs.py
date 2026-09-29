"""Small environments that every exercise in the course shares.

The harness puts this file's directory on the import path, so a starter file,
a solution, and a test suite all write `import rlenvs` and get the same worlds.
Everything here is plain numpy, runs in microseconds per step, and is exactly
reproducible from a seed.

The module holds four kinds of thing:

- Bandits: `BernoulliBandit` and `GaussianBandit`.
- Tabular Markov decision processes: `TabularMDP`, which carries its full
  model (transition and reward arrays) and a Gym-style `reset`/`step` loop,
  and the builders `random_mdp`, `gridworld`, `cliff_walking`, `frozen_lake`,
  and `river_swim`.
- Exact answers for tabular problems, for checks to compare against:
  `policy_evaluation`, `q_from_v`, `value_iteration`, `greedy`,
  `state_distribution`, and `uniform_policy`.
- Continuous control: `CartPole` and `Pendulum`, with the classic Gym physics.

A policy for a tabular MDP is always an array `pi` of shape `(S, A)` whose rows
are probability distributions over actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "BernoulliBandit",
    "GaussianBandit",
    "TabularMDP",
    "random_mdp",
    "gridworld",
    "cliff_walking",
    "frozen_lake",
    "river_swim",
    "uniform_policy",
    "policy_evaluation",
    "q_from_v",
    "value_iteration",
    "greedy",
    "state_distribution",
    "CartPole",
    "Pendulum",
]


# Bandits ----------------------------------------------------------------


class BernoulliBandit:
    """A k-armed bandit whose arm `a` pays 1 with probability `means[a]`, else 0."""

    def __init__(self, means, seed: int | None = None) -> None:
        self.means = np.asarray(means, dtype=float)
        self.k = len(self.means)
        self.rng = np.random.default_rng(seed)

    @property
    def best_mean(self) -> float:
        return float(self.means.max())

    def pull(self, arm: int) -> float:
        return float(self.rng.random() < self.means[arm])

    def regret(self, arm: int) -> float:
        """The expected reward lost by pulling `arm` instead of the best arm."""
        return self.best_mean - float(self.means[arm])


class GaussianBandit:
    """A k-armed bandit whose arm `a` pays a draw from Normal(means[a], sigma**2)."""

    def __init__(self, means, sigma: float = 1.0, seed: int | None = None) -> None:
        self.means = np.asarray(means, dtype=float)
        self.sigma = float(sigma)
        self.k = len(self.means)
        self.rng = np.random.default_rng(seed)

    @property
    def best_mean(self) -> float:
        return float(self.means.max())

    def pull(self, arm: int) -> float:
        return float(self.means[arm] + self.sigma * self.rng.standard_normal())

    def regret(self, arm: int) -> float:
        return self.best_mean - float(self.means[arm])


# Tabular MDPs -----------------------------------------------------------


@dataclass
class TabularMDP:
    """A finite MDP with its whole model in memory.

    Attributes:
        P: transition probabilities, shape `(S, A, S)`. `P[s, a, s2]` is the
            probability of landing in `s2` after taking `a` in `s`.
        R: expected reward for taking `a` in `s`, shape `(S, A)`.
        gamma: the discount factor, in `[0, 1]`.
        mu0: the start-state distribution, shape `(S,)`.
        terminal: shape `(S,)`, true for absorbing states that end an episode.
            A terminal state's value is zero by definition.
        R3: optional reward that depends on the landing state too, shape
            `(S, A, S)`. When present, `step` pays `R3[s, a, s2]`, and `R` is
            its expectation under `P`.
        outcomes: optional explicit outcome table for worlds where two ways of
            reaching the same next state pay different rewards. A tuple of
            three `(S, A, K)` arrays: probabilities, next states, and rewards.
            When present, `step` and `sample` draw one of the `K` outcomes.
        max_steps: `step` reports truncation after this many steps.
    """

    P: np.ndarray
    R: np.ndarray
    gamma: float
    mu0: np.ndarray
    terminal: np.ndarray | None = None
    R3: np.ndarray | None = None
    max_steps: int = 1000
    names: dict = field(default_factory=dict)
    outcomes: tuple | None = None

    def __post_init__(self) -> None:
        self.P = np.asarray(self.P, dtype=float)
        self.R = np.asarray(self.R, dtype=float)
        self.mu0 = np.asarray(self.mu0, dtype=float)
        S, A, S2 = self.P.shape
        if S != S2 or self.R.shape != (S, A) or self.mu0.shape != (S,):
            raise ValueError("P must be (S, A, S), R must be (S, A), mu0 must be (S,)")
        if not np.allclose(self.P.sum(axis=2), 1.0):
            raise ValueError("every row P[s, a, :] must sum to one")
        if self.terminal is None:
            self.terminal = np.zeros(S, dtype=bool)
        self.terminal = np.asarray(self.terminal, dtype=bool)
        self._rng = np.random.default_rng()
        self._state = 0
        self._t = 0

    @property
    def n_states(self) -> int:
        return self.P.shape[0]

    @property
    def n_actions(self) -> int:
        return self.P.shape[1]

    def reset(self, seed: int | None = None) -> int:
        """Start an episode and return the first state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = int(self._rng.choice(self.n_states, p=self.mu0))
        self._t = 0
        return self._state

    def step(self, action: int) -> tuple[int, float, bool, bool]:
        """Take `action`. Returns `(next_state, reward, terminated, truncated)`."""
        s2, reward = self.sample(self._state, int(action), self._rng)
        self._state = s2
        self._t += 1
        terminated = bool(self.terminal[s2])
        truncated = (not terminated) and self._t >= self.max_steps
        return s2, reward, terminated, truncated

    def sample(self, s: int, a: int, rng: np.random.Generator) -> tuple[int, float]:
        """Sample one transition from `(s, a)` without touching the episode state."""
        if self.outcomes is not None:
            probs, nexts, rewards = self.outcomes
            k = int(rng.choice(probs.shape[2], p=probs[s, a]))
            return int(nexts[s, a, k]), float(rewards[s, a, k])
        s2 = int(rng.choice(self.n_states, p=self.P[s, a]))
        reward = float(self.R3[s, a, s2]) if self.R3 is not None else float(self.R[s, a])
        return s2, reward


def _finish_rewards(P: np.ndarray, R3: np.ndarray) -> np.ndarray:
    return (P * R3).sum(axis=2)


def random_mdp(
    n_states: int = 5,
    n_actions: int = 3,
    gamma: float = 0.9,
    seed: int = 0,
    branching: int | None = None,
) -> TabularMDP:
    """A random MDP with Dirichlet transitions and standard-normal rewards.

    With `branching` set, each `(s, a)` reaches only that many next states.
    """
    rng = np.random.default_rng(seed)
    P = np.zeros((n_states, n_actions, n_states))
    for s in range(n_states):
        for a in range(n_actions):
            if branching is None:
                P[s, a] = rng.dirichlet(np.ones(n_states))
            else:
                support = rng.choice(n_states, size=branching, replace=False)
                P[s, a, support] = rng.dirichlet(np.ones(branching))
    R = rng.standard_normal((n_states, n_actions))
    mu0 = np.ones(n_states) / n_states
    return TabularMDP(P=P, R=R, gamma=gamma, mu0=mu0)


# Actions are numbered clockwise from up.
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
_MOVES = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}


def gridworld(
    layout: list[str],
    gamma: float = 1.0,
    slip: float = 0.0,
    step_reward: float = -1.0,
    goal_reward: float = 0.0,
    cliff_reward: float = -100.0,
    hole_reward: float = 0.0,
    max_steps: int = 500,
) -> TabularMDP:
    """Build a grid world from rows of characters.

    Cells: `.` open, `#` wall, `S` start, `G` goal (terminal), `H` hole
    (terminal), `C` cliff (not terminal: stepping onto it sends you back to
    `S`). Every open, start, goal, hole, and cliff cell is a state; walls are
    not. Moving into a wall or off the grid leaves you where you are.

    With probability `1 - slip` the chosen action happens; with probability
    `slip` a uniformly random action happens instead (which may be the chosen
    one). Every move pays `step_reward`, entering `G` pays `goal_reward` on
    top, entering `H` pays `hole_reward` on top, and entering `C` pays
    `cliff_reward` instead of `step_reward`.

    The returned MDP carries `names["cells"]`, the `(row, col)` of each state,
    and `names["grid"]`, the layout.
    """
    rows = [list(r) for r in layout]
    height, width = len(rows), len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError("every row of the layout must be the same length")
    cells = [(i, j) for i in range(height) for j in range(width) if rows[i][j] != "#"]
    index = {c: k for k, c in enumerate(cells)}
    S, A = len(cells), 4
    starts = [index[c] for c in cells if rows[c[0]][c[1]] == "S"]
    if len(starts) != 1:
        raise ValueError("the layout needs exactly one S")
    start = starts[0]

    K = A + 1
    probs = np.zeros((S, A, K))
    nexts = np.zeros((S, A, K), dtype=int)
    rewards = np.zeros((S, A, K))
    terminal = np.array([rows[i][j] in "GH" for (i, j) in cells])
    for s, (i, j) in enumerate(cells):
        for a in range(A):
            if terminal[s]:
                probs[s, a, 0], nexts[s, a, :] = 1.0, s
                continue
            outcomes = [(1.0 - slip, a)] + [(slip / A, b) for b in range(A)]
            for k, (prob, b) in enumerate(outcomes):
                di, dj = _MOVES[b]
                ni, nj = i + di, j + dj
                if not (0 <= ni < height and 0 <= nj < width) or rows[ni][nj] == "#":
                    ni, nj = i, j
                kind = rows[ni][nj]
                if kind == "C":
                    s2, r = start, cliff_reward
                else:
                    s2, r = index[(ni, nj)], step_reward
                    if kind == "G":
                        r += goal_reward
                    elif kind == "H":
                        r += hole_reward
                probs[s, a, k], nexts[s, a, k], rewards[s, a, k] = prob, s2, r

    # Fold the outcome table into the model arrays. Two outcomes can reach the
    # same next state with different rewards (from S, falling off the cliff
    # and bumping a wall both land on S), so R3 holds the mean reward given
    # the landing state, and `step` draws from the outcome table itself.
    P = np.zeros((S, A, S))
    paid = np.zeros((S, A, S))
    for k in range(K):
        np.add.at(P, (np.arange(S)[:, None], np.arange(A)[None, :], nexts[:, :, k]), probs[:, :, k])
        np.add.at(
            paid,
            (np.arange(S)[:, None], np.arange(A)[None, :], nexts[:, :, k]),
            probs[:, :, k] * rewards[:, :, k],
        )
    R3 = np.divide(paid, P, out=np.zeros_like(paid), where=P > 0)
    mu0 = np.zeros(S)
    mu0[start] = 1.0
    return TabularMDP(
        P=P,
        R=_finish_rewards(P, R3),
        gamma=gamma,
        mu0=mu0,
        terminal=terminal,
        R3=R3,
        max_steps=max_steps,
        names={"cells": cells, "grid": list(layout), "start": start},
        outcomes=(probs, nexts, rewards),
    )


CLIFF_LAYOUT = [
    "............",
    "............",
    "............",
    "SCCCCCCCCCCG",
]


def cliff_walking(gamma: float = 1.0, slip: float = 0.0) -> TabularMDP:
    """Sutton and Barto's cliff walk (Example 6.6): 4 by 12, -1 per step, -100 for the cliff."""
    return gridworld(CLIFF_LAYOUT, gamma=gamma, slip=slip)


FROZEN_LAKE_LAYOUT = [
    "S...",
    ".H.H",
    "...H",
    "H..G",
]


def frozen_lake(gamma: float = 0.95, slip: float = 2 / 3) -> TabularMDP:
    """A 4 by 4 slippery lake: reward 1 for reaching G, 0 otherwise, holes end the episode.

    The map is Gym's FrozenLake 4x4, but the slip differs: here a slip picks any
    of the four actions uniformly, where Gym only slips sideways.

    `slip=2/3` means the chosen action happens with probability 1/3 + 1/6 = 1/2.
    """
    return gridworld(
        FROZEN_LAKE_LAYOUT,
        gamma=gamma,
        slip=slip,
        step_reward=0.0,
        goal_reward=1.0,
        max_steps=100,
    )


def river_swim(n: int = 6, gamma: float = 0.95) -> TabularMDP:
    """A RiverSwim-style chain that punishes shallow exploration.

    Action 0 (swim left, with the current) always moves one state left and pays
    0.005 in state 0. Action 1 (swim right, against the current) moves right
    with probability 0.35, stays with probability 0.6, and slips left with
    probability 0.05; at the two ends the probability of leaving the chain is
    folded into staying. Swimming right in the last state pays 1. The episode
    starts in state 0 or 1 with equal probability.
    """
    S, A = n, 2
    P = np.zeros((S, A, S))
    R = np.zeros((S, A))
    for s in range(S):
        P[s, 0, max(s - 1, 0)] = 1.0
        right, stay, left = min(s + 1, S - 1), s, max(s - 1, 0)
        P[s, 1, right] += 0.35
        P[s, 1, stay] += 0.60
        P[s, 1, left] += 0.05
    R[0, 0] = 0.005
    R[S - 1, 1] = 1.0
    mu0 = np.zeros(S)
    mu0[:2] = 0.5
    return TabularMDP(P=P, R=R, gamma=gamma, mu0=mu0, max_steps=200)


# Exact answers ----------------------------------------------------------


def uniform_policy(mdp: TabularMDP) -> np.ndarray:
    return np.full((mdp.n_states, mdp.n_actions), 1.0 / mdp.n_actions)


def policy_evaluation(mdp: TabularMDP, pi: np.ndarray) -> np.ndarray:
    """The exact value of policy `pi`, by solving the Bellman equation as a linear system.

    Solves `(I - gamma P_pi) v = r_pi` with terminal states fixed at `v = 0`.

    With `gamma < 1` every state has a finite value. With `gamma = 1` a state
    has one only if, from it, `pi` reaches a terminal state with probability
    one. States where it does not, such as a state whose greedy action walks
    into a wall forever, get `nan` rather than making the whole solve fail, so
    a policy that is fine from the start state can still be evaluated there.
    """
    pi = np.asarray(pi, dtype=float)
    S = mdp.n_states
    P_pi = np.einsum("sa,sat->st", pi, mdp.P)
    r_pi = (pi * mdp.R).sum(axis=1)
    live = ~mdp.terminal
    P_pi = P_pi * live[:, None]
    r_pi = r_pi * live
    if mdp.gamma < 1.0:
        return np.linalg.solve(np.eye(S) - mdp.gamma * P_pi, r_pi)

    # With no discount, keep the states that cannot reach a closed loop of
    # live states: those reach a terminal state with probability one, and the
    # system restricted to them is nonsingular.
    edges = P_pi > 0
    can_end = mdp.terminal.copy()
    while True:
        grown = can_end | (edges & can_end[None, :]).any(axis=1)
        if (grown == can_end).all():
            break
        can_end = grown
    doomed = ~can_end
    reaches_doom = doomed.copy()
    while True:
        grown = reaches_doom | (edges & reaches_doom[None, :]).any(axis=1)
        if (grown == reaches_doom).all():
            break
        reaches_doom = grown
    proper = ~reaches_doom
    v = np.full(S, np.nan)
    idx = np.flatnonzero(proper)
    M = np.eye(len(idx)) - P_pi[np.ix_(idx, idx)]
    v[idx] = np.linalg.solve(M, r_pi[idx])
    return v


def q_from_v(mdp: TabularMDP, v: np.ndarray) -> np.ndarray:
    """`q[s, a] = R[s, a] + gamma * sum_s2 P[s, a, s2] v[s2]`, zero in terminal states."""
    q = mdp.R + mdp.gamma * mdp.P @ np.asarray(v, dtype=float)
    q[mdp.terminal] = 0.0
    return q


def value_iteration(mdp: TabularMDP, tol: float = 1e-12, max_iter: int = 100_000) -> np.ndarray:
    """The optimal state values, iterated until successive sweeps differ by less than `tol`."""
    v = np.zeros(mdp.n_states)
    for _ in range(max_iter):
        new = q_from_v(mdp, v).max(axis=1)
        if np.max(np.abs(new - v)) < tol:
            return new
        v = new
    return v


def greedy(q: np.ndarray) -> np.ndarray:
    """A deterministic policy, shape `(S, A)`, that picks the first maximising action."""
    q = np.asarray(q)
    pi = np.zeros_like(q, dtype=float)
    pi[np.arange(q.shape[0]), q.argmax(axis=1)] = 1.0
    return pi


def state_distribution(mdp: TabularMDP, pi: np.ndarray) -> np.ndarray:
    """The normalised discounted state-visitation distribution `d_pi` from `mu0`.

    `d_pi = (1 - gamma) mu0^T (I - gamma P_pi)^{-1}`, with terminal states
    absorbing nothing. Needs `gamma < 1`.
    """
    if mdp.gamma >= 1.0:
        raise ValueError("the discounted visitation distribution needs gamma < 1")
    P_pi = np.einsum("sa,sat->st", pi, mdp.P) * (~mdp.terminal)[:, None]
    M = np.eye(mdp.n_states) - mdp.gamma * P_pi
    d = np.linalg.solve(M.T, mdp.mu0)
    return d / d.sum()


# Continuous control -----------------------------------------------------


class CartPole:
    """CartPole-v1 with the Gym physics, in numpy.

    Observation: `[x, x_dot, theta, theta_dot]`. Actions: 0 pushes left, 1
    pushes right. Every step pays 1, including the last. The episode
    terminates when `|theta| > 12` degrees or `|x| > 2.4`, and truncates after
    `max_steps` steps.
    """

    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    length = 0.5  # half the pole's length
    force_mag = 10.0
    tau = 0.02
    theta_threshold = 12 * 2 * np.pi / 360
    x_threshold = 2.4
    n_actions = 2
    obs_dim = 4

    def __init__(self, max_steps: int = 500, seed: int | None = None) -> None:
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(4)
        self.t = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.rng.uniform(-0.05, 0.05, size=4)
        self.t = 0
        return self.state.copy()

    @classmethod
    def dynamics(cls, state: np.ndarray, action) -> np.ndarray:
        """One Euler step for a state (shape `(4,)` or `(N, 4)`) and action(s)."""
        x, x_dot, theta, theta_dot = np.moveaxis(np.asarray(state, dtype=float), -1, 0)
        force = np.where(np.asarray(action) == 1, cls.force_mag, -cls.force_mag)
        total_mass = cls.masscart + cls.masspole
        polemass_length = cls.masspole * cls.length
        cos, sin = np.cos(theta), np.sin(theta)
        temp = (force + polemass_length * theta_dot**2 * sin) / total_mass
        theta_acc = (cls.gravity * sin - cos * temp) / (
            cls.length * (4.0 / 3.0 - cls.masspole * cos**2 / total_mass)
        )
        x_acc = temp - polemass_length * theta_acc * cos / total_mass
        x = x + cls.tau * x_dot
        x_dot = x_dot + cls.tau * x_acc
        theta = theta + cls.tau * theta_dot
        theta_dot = theta_dot + cls.tau * theta_acc
        return np.stack([x, x_dot, theta, theta_dot], axis=-1)

    @classmethod
    def failed(cls, state: np.ndarray):
        state = np.asarray(state)
        return (np.abs(state[..., 0]) > cls.x_threshold) | (
            np.abs(state[..., 2]) > cls.theta_threshold
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool]:
        self.state = self.dynamics(self.state, action)
        self.t += 1
        terminated = bool(self.failed(self.state))
        truncated = (not terminated) and self.t >= self.max_steps
        return self.state.copy(), 1.0, terminated, truncated


class Pendulum:
    """Pendulum-v1 with the Gym physics, in numpy.

    Observation: `[cos(theta), sin(theta), theta_dot]`, with `theta = 0`
    upright. Action: a torque in `[-2, 2]`. The reward is
    `-(angle**2 + 0.1 * theta_dot**2 + 0.001 * torque**2)`, where `angle` is
    `theta` wrapped to `[-pi, pi)`. There is no terminal state; episodes
    truncate after `max_steps` steps.
    """

    max_speed = 8.0
    max_torque = 2.0
    dt = 0.05
    g = 10.0
    m = 1.0
    l = 1.0
    obs_dim = 3
    act_dim = 1

    def __init__(self, max_steps: int = 200, seed: int | None = None) -> None:
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.theta = 0.0
        self.theta_dot = 0.0
        self.t = 0

    def _obs(self) -> np.ndarray:
        return np.array([np.cos(self.theta), np.sin(self.theta), self.theta_dot])

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.theta = float(self.rng.uniform(-np.pi, np.pi))
        self.theta_dot = float(self.rng.uniform(-1.0, 1.0))
        self.t = 0
        return self._obs()

    @staticmethod
    def angle_normalize(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def step(self, action) -> tuple[np.ndarray, float, bool, bool]:
        u = float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], -self.max_torque, self.max_torque))
        th, thdot = self.theta, self.theta_dot
        cost = self.angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * u**2
        thdot = thdot + (3 * self.g / (2 * self.l) * np.sin(th) + 3.0 / (self.m * self.l**2) * u) * self.dt
        thdot = float(np.clip(thdot, -self.max_speed, self.max_speed))
        self.theta = float(th + thdot * self.dt)
        self.theta_dot = thdot
        self.t += 1
        return self._obs(), float(-cost), False, self.t >= self.max_steps
