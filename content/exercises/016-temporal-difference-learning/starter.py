"""TD(0) on the random walk.

Fill in the functions below, in order. `random_walk` is ready to use: it
builds the five-state random walk, whose true values are k / 6.

Batch functions take episodes as (states, rewards) pairs, where states has
one more entry than rewards: it ends with the terminal state. The learning
function plays its own episodes with one np.random.default_rng(seed).
"""

import numpy as np

import rlenvs


def random_walk(n: int = 5) -> rlenvs.TabularMDP:
    """Sutton and Barto's random walk (Example 6.2) as a `TabularMDP`.

    States `1` to `n` are the walk and states `0` and `n + 1` are the two
    exits, both terminal. The one action moves left or right with probability
    one half each. Leaving on the right pays 1 and every other step pays 0.
    Episodes start in the middle, and `gamma = 1`, so the true value of state
    `k` is `k / (n + 1)`.
    """
    S = n + 2
    P = np.zeros((S, 1, S))
    R3 = np.zeros((S, 1, S))
    for s in range(1, n + 1):
        P[s, 0, s - 1] = 0.5
        P[s, 0, s + 1] = 0.5
    P[0, 0, 0] = 1.0
    P[S - 1, 0, S - 1] = 1.0
    R3[n, 0, n + 1] = 1.0
    terminal = np.zeros(S, dtype=bool)
    terminal[[0, S - 1]] = True
    mu0 = np.zeros(S)
    mu0[(n + 1) // 2] = 1.0
    R = (P * R3).sum(axis=2)
    return rlenvs.TabularMDP(P=P, R=R, gamma=1.0, mu0=mu0, terminal=terminal, R3=R3)


def td0_update(V, s, r, s2, terminated, alpha, gamma):
    """Update `V[s]` in place toward the one-step target, and return `V`.

    The target is `r + gamma * V[s2]`, or `r` alone when `terminated` is true,
    because a terminal state has no future. `V[s]` moves a fraction `alpha`
    of the way to the target.
    """
    raise NotImplementedError("compute the TD target, then move V[s] a fraction alpha toward it")


def td0_prediction(mdp, pi, n_episodes, alpha, seed, v_init=0.5):
    """Estimate V for the policy `pi` by TD(0), and return it with shape `(S,)`.

    `pi` has shape `(S, A)`. Start terminal states at 0 and every other state
    at `v_init`. Use one `np.random.default_rng(seed)`: draw the start state
    from `mdp.mu0`, each action from `pi[s]`, and each step with
    `mdp.sample(s, a, rng)`. Call `td0_update` after every step.
    """
    raise NotImplementedError("play episodes and call td0_update after every step")


def batch_td0(episodes, n_states, gamma, alpha, n_sweeps):
    """Run batch TD(0) over fixed episodes and return `V`, shape `(n_states,)`.

    Start `V` at zero. In each sweep, hold `V` fixed: add `alpha * delta` for
    every step of every episode into a separate array, then add that array to
    `V` once. The last step of an episode has no next state to bootstrap from.
    """
    raise NotImplementedError("sweep the episodes with V held fixed, then apply the total")


if __name__ == "__main__":
    walk = random_walk()
    V = td0_prediction(walk, np.ones((walk.n_states, 1)), n_episodes=100, alpha=0.05, seed=0)
    print("TD(0) on the random walk after 100 episodes:", np.round(V[1:-1], 3))
    print("true values:                               ", np.round(np.arange(1, 6) / 6, 3))
