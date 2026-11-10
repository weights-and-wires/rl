"""Reference solution for the value-function approximation exercise."""

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
    pos_u = (min(max(pos, POS_LOW), POS_HIGH) - POS_LOW) / (POS_HIGH - POS_LOW) * n_tiles
    vel_u = (min(max(vel, VEL_LOW), VEL_HIGH) - VEL_LOW) / (VEL_HIGH - VEL_LOW) * n_tiles
    per_tiling = (n_tiles + 1) ** 2
    features = np.zeros(n_tilings * per_tiling)
    for k in range(n_tilings):
        # Tiling k is shifted left by k / n_tilings of a tile in position and by
        # (3k mod n_tilings) / n_tilings in velocity, so the grids do not line up.
        i = int(np.floor(pos_u + k / n_tilings))
        j = int(np.floor(vel_u + (3 * k % n_tilings) / n_tilings))
        features[k * per_tiling + i * (n_tiles + 1) + j] = 1.0
    return features


def state_action_features(pos: float, vel: float, action: int) -> np.ndarray:
    x = tile_features_2d(pos, vel, N_TILES, N_TILINGS)
    features = np.zeros(N_ACTIONS * x.shape[0])
    # Each action owns its own block, so the weights of other actions never move.
    features[action * x.shape[0] : (action + 1) * x.shape[0]] = x
    return features


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
    x = state_action_features(s[0], s[1], a)
    # The target uses the next action a2 that the agent will take, not the current a.
    target = r + gamma * (0.0 if terminated else float(w @ state_action_features(s2[0], s2[1], a2)))
    return w + alpha * (target - float(w @ x)) * x


def semi_gradient_sarsa(
    n_episodes: int,
    alpha: float = 0.5 / N_TILINGS,
    gamma: float = 1.0,
    epsilon: float = 0.0,
    max_steps: int = 1000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    env = MountainCar()
    w = np.zeros(N_ACTIONS * PER_ACTION)  # zero is optimistic: every reward is -1

    def pick(s):
        if rng.random() < epsilon:
            return int(rng.integers(N_ACTIONS))
        x = tile_features_2d(s[0], s[1], N_TILES, N_TILINGS)
        return int(np.argmax(w.reshape(N_ACTIONS, -1) @ x))

    steps = np.zeros(n_episodes, dtype=int)
    for ep in range(n_episodes):
        s = env.reset(rng)
        a = pick(s)
        for t in range(max_steps):
            s2, r, terminated = env.step(a)
            a2 = pick(s2)
            w = semi_gradient_sarsa_update(w, s, a, r, s2, a2, terminated, alpha, gamma)
            steps[ep] = t + 1
            if terminated:
                break
            s, a = s2, a2
    return w, steps


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
