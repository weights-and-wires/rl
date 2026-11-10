"""Control and average reward: features, n-step SARSA, and differential TD.

Fill in the functions below, in order. `stationary_distribution`, `MountainCar`,
and `state_action_features` are already written for you.
"""

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
    """Fourier cosine features of a point x in [0, 1]^d (Konidaris et al., 2011).

    x is a 1-D array of length d. Build one feature for each coefficient vector
    c in {0, 1, ..., order}^d, in the order of itertools.product(range(order + 1),
    repeat=d) (the last coefficient changes fastest). The feature for c is
    cos(pi * c @ x). Returns a float array of length (order + 1) ** d.
    """
    raise NotImplementedError


def rbf_features(x: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian radial basis features of the point x.

    x has shape (d,) and centers has shape (k, d). Feature i is
    exp(-||x - centers[i]||^2 / (2 * sigma^2)). Returns a float array of shape (k,).
    """
    raise NotImplementedError


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
    """The n-step SARSA return from one starting step.

    rewards holds the (up to n) rewards r_tau, ..., in order, and q_boot is the
    estimate q(s_{tau+n}, a_{tau+n}) to bootstrap from. The return is
    sum_i gamma**i * rewards[i], plus gamma ** len(rewards) * q_boot unless
    terminated is true (the episode ended inside the window, so nothing is left to
    bootstrap from).
    """
    raise NotImplementedError


def n_step_sarsa(
    n: int,
    n_episodes: int,
    alpha: float = 0.5 / N_TILINGS,
    gamma: float = 1.0,
    epsilon: float = 0.0,
    max_steps: int = 1000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Episodic n-step semi-gradient SARSA on Mountain Car (Sutton and Barto, 10.2).

    Keep the states, actions, and rewards of the episode. At each step t, once
    tau = t - n + 1 is at least 0, update the weights of (s_tau, a_tau) toward
    n_step_return(...), bootstrapping from (s_{tau+n}, a_{tau+n}) when the episode
    has not ended by then. After the episode ends at step T, keep updating for the
    remaining tau up to T - 1. Choose actions epsilon-greedily from the current
    weights, and stop an episode that reaches max_steps steps without a terminal
    update. Weights start at zero.

    Returns (w, steps), where w has shape (N_ACTIONS * PER_ACTION,) and steps is
    an int array with the number of steps in each episode.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Average reward and differential TD
# ---------------------------------------------------------------------------


def average_reward(P: np.ndarray, r: np.ndarray) -> float:
    """The average reward of the chain P with reward r[s] in state s.

    P has shape (S, S) and r has shape (S,). Returns the stationary-distribution
    mean of r, the sum over s of mu[s] * r[s]. Use stationary_distribution.
    """
    raise NotImplementedError


def differential_td_update(
    w: np.ndarray,
    r_bar: float,
    x: np.ndarray,
    r: float,
    x2: np.ndarray,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, float]:
    """One differential semi-gradient TD(0) step (Sutton and Barto, 10.3).

    Estimates are v(s) = w @ x. The differential TD error is
    delta = r - r_bar + w @ x2 - w @ x, with no discount and no terminal mask,
    because the task never ends. Then w moves by alpha * delta * x and r_bar moves
    by beta * delta. Returns the new (w, r_bar) without changing the inputs.
    """
    raise NotImplementedError


def differential_td(
    P: np.ndarray,
    r: np.ndarray,
    Phi: np.ndarray,
    n_steps: int,
    alpha: float,
    beta: float,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Differential TD(0) prediction on a continuing chain.

    P has shape (S, S), r[s] is the reward received in state s, and Phi has shape
    (S, d) with the feature vector of state s in row s. Start in state 0 with
    w = zeros(d) and r_bar = 0. For n_steps steps, sample the next state from
    P[s], apply differential_td_update, and move on. Draw from
    np.random.default_rng(seed). Returns (w, r_bar).
    """
    raise NotImplementedError


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
