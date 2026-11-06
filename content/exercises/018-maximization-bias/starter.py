"""Maximization bias, double Q-learning, and the Expected SARSA target.

Fill in the functions below, in order. `MaxBiasMDP` and
`epsilon_greedy_action` are ready to use. In `MaxBiasMDP`, a table is a list
of rows from `env.new_q()`, so write `Q[s][a]` rather than `Q[s, a]`.
"""

import numpy as np


class MaxBiasMDP:
    """Sutton and Barto's maximization-bias MDP (Example 6.7).

    From state A (0), action RIGHT (1) ends the episode with reward 0, and
    action LEFT (0) moves to state B (1) with reward 0. State B has
    `n_b_actions` actions, and each one ends the episode with a reward drawn
    from Normal(-0.1, 1). Going left is worth -0.1, so RIGHT is optimal in A.
    State 2 is terminal. There is no discount.

    Different states have different numbers of actions, so an action-value
    table is a list of rows: `Q[s]` is an array of length `n_actions(s)`.
    """

    A, B, TERMINAL = 0, 1, 2
    LEFT, RIGHT = 0, 1

    def __init__(self, n_b_actions: int = 10, mean: float = -0.1, sigma: float = 1.0):
        self.n_b_actions = n_b_actions
        self.mean = mean
        self.sigma = sigma
        self.gamma = 1.0

    def n_actions(self, s: int) -> int:
        return 2 if s == self.A else self.n_b_actions

    def new_q(self) -> list:
        """A zero action-value table: a list with one row per state."""
        return [np.zeros(2), np.zeros(self.n_b_actions), np.zeros(1)]

    def reset(self) -> int:
        return self.A

    def step(self, s: int, a: int, rng: np.random.Generator) -> tuple[int, float, bool]:
        """Take action `a` in state `s`. Returns `(next_state, reward, terminated)`."""
        if s == self.A:
            if a == self.RIGHT:
                return self.TERMINAL, 0.0, True
            return self.B, 0.0, False
        return self.TERMINAL, float(self.mean + self.sigma * rng.standard_normal()), True


def epsilon_greedy_action(q_row, epsilon, rng):
    """An epsilon-greedy action from one row of Q, breaking ties at random.

    You wrote this function in the SARSA and Q-learning chapter.
    """
    if rng.random() < epsilon:
        return int(rng.integers(len(q_row)))
    return int(rng.choice(np.flatnonzero(q_row == q_row.max())))


def single_estimate(means):
    """Return the max over the last axis: input `(..., A)`, output `(...)`.

    This is what Q-learning's target does with noisy action values.
    """
    raise NotImplementedError("take the max over the last axis")


def double_estimate(means_a, means_b):
    """The double estimator: input two `(..., A)` arrays, output `(...)`.

    Use `means_a` to choose the best action (first argmax on ties), and read
    that action's value from `means_b`.
    """
    raise NotImplementedError("argmax one array, then index the other at that action")


def double_q_target(q_select, q_evaluate, r, terminated, gamma):
    """Return the double Q-learning target as a float.

    `q_select` and `q_evaluate` are the two tables' rows for the next state,
    each of shape `(A,)`. `q_select` picks the action, and `q_evaluate`
    scores it: `r + gamma * q_evaluate[argmax(q_select)]`. When `terminated`
    is true the target is `r` alone.
    """
    raise NotImplementedError("choose with one row and score with the other")


def q_learning(env, n_episodes, alpha, epsilon, seed):
    """Run Q-learning on a `MaxBiasMDP`.

    Returns a boolean array of shape `(n_episodes,)`, `True` where the episode
    went left from A. Use one `np.random.default_rng(seed)`, a table from
    `env.new_q()`, and `epsilon_greedy_action`. Record the action taken in
    `env.A` before you step. The target is `r`, or `r + gamma * Q[s2].max()`.
    """
    raise NotImplementedError("choose, step, update toward the max, and record left from A")


def double_q_learning(env, n_episodes, alpha, epsilon, seed):
    """Run double Q-learning on a `MaxBiasMDP`, with the same output as `q_learning`.

    Keep two tables, `Q1` and `Q2`. Act epsilon-greedily on `Q1[s] + Q2[s]`.
    After each step, flip a fair coin with `rng.random() < 0.5`: on heads
    update `Q1` toward `double_q_target(Q1[s2], Q2[s2], ...)`, otherwise
    update `Q2` toward `double_q_target(Q2[s2], Q1[s2], ...)`.
    """
    raise NotImplementedError("act on the sum, flip a coin, and update one table with the other's score")


def epsilon_greedy_probs(q_row, epsilon):
    """Return the epsilon-greedy action probabilities for one row, shape `(A,)`.

    Every action gets `epsilon / A`, and the greedy actions share the other
    `1 - epsilon` equally, so ties split it evenly.
    """
    raise NotImplementedError("epsilon / A plus (1 - epsilon) split across the ties")


def expected_sarsa_target(q_next, r, terminated, gamma, epsilon):
    """Return the Expected SARSA target as a float.

    It is `r + gamma * sum_a pi(a) * q_next[a]` for the epsilon-greedy policy,
    or `r` alone when `terminated` is true. Use `epsilon_greedy_probs`.
    """
    raise NotImplementedError("average q_next under the epsilon-greedy probabilities")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Ten actions, all truly worth 0, each estimated from 5 noisy samples.
    X = rng.standard_normal((20000, 5, 10)).mean(axis=1)
    Y = rng.standard_normal((20000, 5, 10)).mean(axis=1)
    print(f"single estimate of max value 0: {single_estimate(X).mean():+.3f}")
    print(f"double estimate of max value 0: {double_estimate(X, Y).mean():+.3f}")

    env = MaxBiasMDP()
    runs = 100
    q = np.mean([q_learning(env, 100, 0.1, 0.1, seed) for seed in range(runs)], axis=0)
    dq = np.mean([double_q_learning(env, 100, 0.1, 0.1, seed) for seed in range(runs)], axis=0)
    print("fraction of runs going left from A (optimal is 0.05 under epsilon = 0.1)")
    for lo, hi in ((0, 10), (10, 30), (80, 100)):
        print(f"  episodes {lo + 1:3d}-{hi:3d}: Q-learning {q[lo:hi].mean():.2f}, double Q {dq[lo:hi].mean():.2f}")
