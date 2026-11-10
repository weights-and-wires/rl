"""Reference solution for the control and average-reward exercise."""

import itertools
import math

import numpy as np


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """The stationary distribution mu of a state-to-state matrix, with mu P = mu.

    Already written for you. Solves (P^T - I) mu = 0 together with sum(mu) = 1
    by least squares, which is exact when the chain has one stationary distribution.
    """
    S = P.shape[0]
    M = np.vstack([P.T - np.eye(S), np.ones((1, S))])
    rhs = np.zeros(S + 1)
    rhs[-1] = 1.0
    mu = np.linalg.lstsq(M, rhs, rcond=None)[0]
    return mu / mu.sum()


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def fourier_features(x: np.ndarray, order: int) -> np.ndarray:
    x = np.atleast_1d(np.asarray(x, dtype=float))
    # One row per coefficient vector c in {0, ..., order}^d, last entry varying fastest.
    C = np.array(list(itertools.product(range(order + 1), repeat=x.shape[0])), dtype=float)
    return np.cos(math.pi * (C @ x))


def rbf_features(x: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    x = np.atleast_1d(np.asarray(x, dtype=float))
    sq = np.sum((np.asarray(centers, dtype=float) - x) ** 2, axis=1)
    return np.exp(-sq / (2.0 * sigma**2))


# ---------------------------------------------------------------------------
# Mountain Car and tile features from the previous chapter. Already written.
# ---------------------------------------------------------------------------

POS_LOW, POS_HIGH = -1.2, 0.5
VEL_LOW, VEL_HIGH = -0.07, 0.07
N_ACTIONS = 3  # throttle -1, 0, +1
N_TILES = 8
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


def state_action_features(pos: float, vel: float, action: int) -> np.ndarray:
    """Already written. 8 tilings of 9 x 9 tiles, copied into the block of `action`."""
    pos_u = (min(max(pos, POS_LOW), POS_HIGH) - POS_LOW) / (POS_HIGH - POS_LOW) * N_TILES
    vel_u = (min(max(vel, VEL_LOW), VEL_HIGH) - VEL_LOW) / (VEL_HIGH - VEL_LOW) * N_TILES
    features = np.zeros(N_ACTIONS * PER_ACTION)
    base = action * PER_ACTION
    for k in range(N_TILINGS):
        i = int(np.floor(pos_u + k / N_TILINGS))
        j = int(np.floor(vel_u + (3 * k % N_TILINGS) / N_TILINGS))
        features[base + k * PER_TILING + i * (N_TILES + 1) + j] = 1.0
    return features


# ---------------------------------------------------------------------------
# n-step semi-gradient SARSA
# ---------------------------------------------------------------------------


def n_step_return(rewards: list[float], q_boot: float, gamma: float, terminated: bool) -> float:
    G = sum(gamma**i * r for i, r in enumerate(rewards))
    if not terminated:
        G += gamma ** len(rewards) * q_boot
    return float(G)


def n_step_sarsa(
    n: int,
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
        x = state_action_features(s[0], s[1], 0)[:PER_ACTION]
        return int(np.argmax(w.reshape(N_ACTIONS, -1) @ x))

    steps = np.zeros(n_episodes, dtype=int)
    for ep in range(n_episodes):
        states = [env.reset(rng)]
        actions = [pick(states[0])]
        rewards: list[float] = []
        T = None  # the step at which the episode ends, once known
        t = 0
        while True:
            if T is None:
                s2, r, terminated = env.step(actions[t])
                states.append(s2)
                rewards.append(r)
                if terminated:
                    T = t + 1
                else:
                    actions.append(pick(s2))
                    if t + 1 >= max_steps:
                        break  # truncated: stop without a terminal update
            tau = t - n + 1
            if tau >= 0:
                end = min(tau + n, T) if T is not None else tau + n
                q_boot = 0.0
                if T is None or tau + n < T:
                    s_b = states[tau + n]
                    q_boot = float(w @ state_action_features(s_b[0], s_b[1], actions[tau + n]))
                G = n_step_return(rewards[tau:end], q_boot, gamma, T is not None and tau + n >= T)
                x = state_action_features(states[tau][0], states[tau][1], actions[tau])
                w = w + alpha * (G - float(w @ x)) * x
            if T is not None and tau >= T - 1:
                break
            t += 1
        steps[ep] = len(rewards)
    return w, steps


# ---------------------------------------------------------------------------
# Average reward and differential TD
# ---------------------------------------------------------------------------


def average_reward(P: np.ndarray, r: np.ndarray) -> float:
    return float(stationary_distribution(P) @ r)


def differential_td_update(
    w: np.ndarray,
    r_bar: float,
    x: np.ndarray,
    r: float,
    x2: np.ndarray,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, float]:
    # No discount and no terminal mask: the task never ends. The reward is
    # measured against the average reward r_bar.
    delta = r - r_bar + float(w @ x2) - float(w @ x)
    return w + alpha * delta * x, r_bar + beta * delta


def differential_td(
    P: np.ndarray,
    r: np.ndarray,
    Phi: np.ndarray,
    n_steps: int,
    alpha: float,
    beta: float,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    w = np.zeros(Phi.shape[1])
    r_bar = 0.0
    s = 0
    for _ in range(n_steps):
        s2 = int(rng.choice(P.shape[0], p=P[s]))
        w, r_bar = differential_td_update(w, r_bar, Phi[s], float(r[s]), Phi[s2], alpha, beta)
        s = s2
    return w, r_bar


if __name__ == "__main__":
    # 1. Features: Fourier and RBF values for the same point.
    print("fourier order 2 at x = 0.25:", np.round(fourier_features([0.25], 2), 4))
    print("rbf at x = 0.5:", np.round(rbf_features([0.5], np.array([[0.0], [0.5], [1.0]]), 0.5), 4))

    # 2. n-step SARSA against one-step SARSA on Mountain Car.
    for n in (1, 4):
        _, steps = n_step_sarsa(n, 40, alpha=0.5 / N_TILINGS, seed=0)
        print(f"n = {n}: mean steps over 40 episodes {steps.mean():.0f}, last 5 {steps[-5:]}")

    # 3. Differential TD on a four-state ring that lingers in some states.
    stay = np.array([0.5, 0.2, 0.2, 0.2])
    P = np.zeros((4, 4))
    for i in range(4):
        P[i, i] = stay[i]
        P[i, (i + 1) % 4] = 1 - stay[i]
    r = np.array([0.0, 0.0, 0.0, 4.0])
    w, r_bar = differential_td(P, r, np.eye(4), 20000, alpha=0.05, beta=0.005, seed=0)
    print(f"learned average reward {r_bar:.3f}, exact {average_reward(P, r):.3f}")
