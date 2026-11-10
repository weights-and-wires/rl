"""Value-function approximation: tile coding and semi-gradient TD.

Fill in the functions below, in order. A value estimate here is always linear
in features: v(s) = w @ x(s), where x(s) is the feature vector of state s and w
is the weight vector you learn.

`stationary_distribution` is already written for you. It gives the fraction of
time a policy spends in each state, which is how `value_error` weights states.
"""

import math

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


# ---------------------------------------------------------------------------
# Control: episodic semi-gradient SARSA on Mountain Car (Sutton and Barto,
# Example 10.1). This block is identical in starter.py.
# ---------------------------------------------------------------------------

POS_LOW, POS_HIGH = -1.2, 0.5
VEL_LOW, VEL_HIGH = -0.07, 0.07
N_ACTIONS = 3  # throttle -1, 0, +1
N_TILES = 8  # tiles per dimension in one tiling
N_TILINGS = 8
PER_TILING = (N_TILES + 1) ** 2
PER_ACTION = N_TILINGS * PER_TILING


class MountainCar:
    """An underpowered car in a valley. Reward is -1 per step until it reaches the flag."""

    def reset(self, rng: np.random.Generator) -> tuple[float, float]:
        self.pos = float(rng.uniform(-0.6, -0.4))
        self.vel = 0.0
        return self.pos, self.vel

    def step(self, action: int) -> tuple[tuple[float, float], float, bool]:
        vel = self.vel + 0.001 * (action - 1) - 0.0025 * math.cos(3 * self.pos)
        self.vel = min(max(vel, VEL_LOW), VEL_HIGH)
        self.pos = min(max(self.pos + self.vel, POS_LOW), POS_HIGH)
        if self.pos <= POS_LOW:
            self.vel = 0.0  # the left wall stops the car
        return (self.pos, self.vel), -1.0, self.pos >= POS_HIGH


def tile_features_2d(pos: float, vel: float, n_tiles: int, n_tilings: int) -> np.ndarray:
    """Binary tile-coding features for a (position, velocity) pair.

    This generalizes `tile_features` to two inputs. Scale position over
    [POS_LOW, POS_HIGH] and velocity over [VEL_LOW, VEL_HIGH] to [0, n_tiles],
    clipping both first. Each tiling is a grid of (n_tiles + 1) by (n_tiles + 1)
    tiles. Tiling k is shifted left by k / n_tilings of a tile in position and by
    (3 * k % n_tilings) / n_tilings of a tile in velocity. In tiling k, the point
    falls in tile (i, j) with i = floor(pos_scaled + position_shift) and
    j = floor(vel_scaled + velocity_shift).

    Returns:
        A float array of shape (n_tilings * (n_tiles + 1) ** 2,) with exactly one 1
        per tiling. Tiling k owns entries k * (n_tiles + 1) ** 2 up to
        (k + 1) * (n_tiles + 1) ** 2 - 1, and tile (i, j) is entry
        i * (n_tiles + 1) + j inside that block.
    """
    raise NotImplementedError("tile_features_2d")


def state_action_features(pos: float, vel: float, action: int) -> np.ndarray:
    """Features of a state-action pair, shape (N_ACTIONS * PER_ACTION,).

    Compute `tile_features_2d(pos, vel, N_TILES, N_TILINGS)` and put it in the
    block of entries that belongs to `action`, entries action * PER_ACTION up to
    (action + 1) * PER_ACTION - 1. Every other entry is 0. Then q(s, a) = w @ x
    reads only the weights of the action taken.
    """
    raise NotImplementedError("state_action_features")


def semi_gradient_sarsa_update(
    w: np.ndarray,
    s: tuple[float, float],
    a: int,
    r: float,
    s2: tuple[float, float],
    a2: int,
    terminated: bool,
    alpha: float,
    gamma: float,
) -> np.ndarray:
    """One semi-gradient SARSA step for the transition (s, a, r, s2, a2).

    Args:
        w: weights, shape (N_ACTIONS * PER_ACTION,).
        s, s2: (position, velocity) before and after the step.
        a, a2: the action taken in s, and the action the agent will take in s2.
        r: the reward for the step.
        terminated: true when s2 is terminal, so its value counts as 0.
        alpha: step size.
        gamma: discount.

    Returns:
        The updated weights. Only the weights of action a change.
    """
    raise NotImplementedError("semi_gradient_sarsa_update")


def semi_gradient_sarsa(
    n_episodes: int,
    alpha: float = 0.5 / N_TILINGS,
    gamma: float = 1.0,
    epsilon: float = 0.0,
    max_steps: int = 1000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Episodic semi-gradient SARSA on MountainCar.

    Start from zero weights, which is optimistic because every reward is -1. Each
    episode: reset the car with `rng`, pick an action epsilon-greedily from
    q(s, a) = w @ state_action_features(s, a), then loop: step, pick the next
    action a2 from the new state, call `semi_gradient_sarsa_update`, and move on.
    Stop the episode when it terminates or after max_steps steps. Use
    np.random.default_rng(seed) for the car's start and for exploration.

    Returns:
        (w, steps), where steps[i] is the number of steps episode i took.
    """
    raise NotImplementedError("semi_gradient_sarsa")


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

    # 3. Semi-gradient SARSA on Mountain Car. Episodes should get much shorter than 1000 steps.
    w, steps = semi_gradient_sarsa(40, seed=0)
    print(f"steps in episodes 1 to 3: {steps[:3]}, in the last 5: {steps[-5:]}")
