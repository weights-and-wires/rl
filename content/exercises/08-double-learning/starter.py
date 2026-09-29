"""Maximisation bias, double Q-learning, and Expected SARSA.

Fill in the functions below, in order. `MaxBiasMDP` is Sutton and Barto's
Example 6.7, ready to use, and `epsilon_greedy_action` is the action picker
from the previous lesson.

In MaxBiasMDP, states have different numbers of actions, so an action-value
table is a list of rows: Q[s] is an array and Q[s][a] is one value. The
target functions take whole next-state rows, so they work for both that list
and an ordinary (S, A) array.
"""

import numpy as np

import rlenvs


class MaxBiasMDP:
    """Sutton and Barto's maximisation-bias MDP (Example 6.7).

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

    This is the function you wrote for Temporal-difference learning.
    """
    if rng.random() < epsilon:
        return int(rng.integers(len(q_row)))
    return int(rng.choice(np.flatnonzero(q_row == q_row.max())))


def single_estimate(means):
    """The single estimator of the largest true value: the max of the estimates.

    Args:
        means: estimated action values, shape (..., A).

    Returns:
        The max over the last axis, shape (...).
    """
    raise NotImplementedError("single_estimate")


def double_estimate(means_a, means_b):
    """The double estimator: choose with one set of estimates, score with the other.

    Args:
        means_a: estimates used to choose the action, shape (..., A).
        means_b: independent estimates used to score it, same shape.

    Returns:
        Shape (...): means_b at the index of the first maximum of means_a
        along the last axis (the index np.argmax returns).
    """
    raise NotImplementedError("double_estimate")


def epsilon_greedy_probs(q_row, epsilon):
    """The probabilities an epsilon-greedy policy gives each action.

    Args:
        q_row: shape (A,), the action values in one state.
        epsilon: the exploration rate.

    Returns:
        Shape (A,), summing to one. Every action gets epsilon / A, and the
        remaining 1 - epsilon is split evenly among the actions tied for the
        largest value.
    """
    raise NotImplementedError("epsilon_greedy_probs")


def expected_sarsa_target(q_next, r, terminated, gamma, epsilon):
    """The Expected SARSA target for one step.

    Args:
        q_next: shape (A,), the action values in the next state.
        r: the reward for the step.
        terminated: True if the next state is terminal.
        gamma: the discount factor.
        epsilon: the exploration rate of the epsilon-greedy target policy.

    Returns:
        A float: r plus gamma times the expected next value under the
        epsilon-greedy policy, or r alone if terminated.
    """
    raise NotImplementedError("expected_sarsa_target")


def double_q_target(q_select, q_evaluate, r, terminated, gamma):
    """The double Q-learning target for one step.

    Args:
        q_select: shape (A,), next-state values from the table being updated.
            Its first argmax chooses the next action.
        q_evaluate: shape (A,), next-state values from the other table. It
            scores the chosen action.
        r: the reward for the step.
        terminated: True if the next state is terminal.
        gamma: the discount factor.

    Returns:
        A float: r + gamma * q_evaluate[argmax(q_select)], or r alone if
        terminated.
    """
    raise NotImplementedError("double_q_target")


def q_learning(env, n_episodes, alpha, epsilon, seed):
    """Q-learning on a MaxBiasMDP, recording where each episode goes from A.

    Start from Q = env.new_q(). Act with epsilon_greedy_action and step with
    env.step(s, a, rng), using one np.random.default_rng(seed).

    Args:
        env: a MaxBiasMDP.
        n_episodes: how many episodes to run.
        alpha: the step size.
        epsilon: the exploration rate.
        seed: the seed for the random number generator.

    Returns:
        A boolean array of shape (n_episodes,): True where the episode took
        LEFT in state A.
    """
    raise NotImplementedError("q_learning")


def double_q_learning(env, n_episodes, alpha, epsilon, seed):
    """Double Q-learning on a MaxBiasMDP, recording where each episode goes from A.

    Keep two tables, Q1 and Q2, both from env.new_q(). Act epsilon-greedily
    on Q1[s] + Q2[s]. After each step, with probability one half update Q1
    with double_q_target(Q1[s2], Q2[s2], ...), and otherwise update Q2 with
    the roles swapped.

    Arguments and return value are the same as for q_learning.
    """
    raise NotImplementedError("double_q_learning")


def expected_sarsa(mdp, n_episodes, alpha, epsilon, seed):
    """Expected SARSA with an epsilon-greedy policy on a TabularMDP, from Q = 0.

    Args:
        mdp: an rlenvs.TabularMDP. Draw the start from mdp.mu0 and step with
            mdp.sample(s, a, rng). Stop at a terminal state or after
            mdp.max_steps steps.
        n_episodes: how many episodes to run.
        alpha: the step size.
        epsilon: the exploration rate, for acting and for the target.
        seed: seed for np.random.default_rng, the only source of randomness.

    Returns:
        (Q, returns): Q has shape (S, A), and returns has shape (n_episodes,),
        the total reward of each training episode.
    """
    raise NotImplementedError("expected_sarsa")


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

    cliff = rlenvs.cliff_walking()
    Q, returns = expected_sarsa(cliff, 300, alpha=1.0, epsilon=0.1, seed=0)
    v = rlenvs.policy_evaluation(cliff, rlenvs.greedy(Q))[cliff.names["start"]]
    print(f"Expected SARSA on the cliff with alpha = 1: greedy value {v:.1f}, "
          f"last-100 training return {returns[-100:].mean():.1f}")
