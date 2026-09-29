"""Reference solution for the maximisation bias and double learning exercise."""

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
    return np.max(np.asarray(means, dtype=float), axis=-1)


def double_estimate(means_a, means_b):
    means_a = np.asarray(means_a, dtype=float)
    means_b = np.asarray(means_b, dtype=float)
    pick = np.argmax(means_a, axis=-1)
    return np.take_along_axis(means_b, pick[..., None], axis=-1)[..., 0]


def epsilon_greedy_probs(q_row, epsilon):
    q_row = np.asarray(q_row, dtype=float)
    A = len(q_row)
    best = q_row == q_row.max()
    return epsilon / A + (1.0 - epsilon) * best / best.sum()


def expected_sarsa_target(q_next, r, terminated, gamma, epsilon):
    if terminated:
        return float(r)
    return float(r + gamma * epsilon_greedy_probs(q_next, epsilon) @ q_next)


def double_q_target(q_select, q_evaluate, r, terminated, gamma):
    if terminated:
        return float(r)
    return float(r + gamma * q_evaluate[np.argmax(q_select)])


def q_learning(env, n_episodes, alpha, epsilon, seed):
    rng = np.random.default_rng(seed)
    Q = env.new_q()
    went_left = np.zeros(n_episodes, dtype=bool)
    for i in range(n_episodes):
        s = env.reset()
        done = False
        while not done:
            a = epsilon_greedy_action(Q[s], epsilon, rng)
            if s == env.A:
                went_left[i] = a == env.LEFT
            s2, r, done = env.step(s, a, rng)
            target = r if done else r + env.gamma * Q[s2].max()
            Q[s][a] += alpha * (target - Q[s][a])
            s = s2
    return went_left


def double_q_learning(env, n_episodes, alpha, epsilon, seed):
    rng = np.random.default_rng(seed)
    Q1, Q2 = env.new_q(), env.new_q()
    went_left = np.zeros(n_episodes, dtype=bool)
    for i in range(n_episodes):
        s = env.reset()
        done = False
        while not done:
            # Behave epsilon-greedily on the sum of the two tables.
            a = epsilon_greedy_action(Q1[s] + Q2[s], epsilon, rng)
            if s == env.A:
                went_left[i] = a == env.LEFT
            s2, r, done = env.step(s, a, rng)
            # A fair coin decides which table to update; the other one scores.
            if rng.random() < 0.5:
                target = double_q_target(Q1[s2], Q2[s2], r, done, env.gamma)
                Q1[s][a] += alpha * (target - Q1[s][a])
            else:
                target = double_q_target(Q2[s2], Q1[s2], r, done, env.gamma)
                Q2[s][a] += alpha * (target - Q2[s][a])
            s = s2
    return went_left


def expected_sarsa(mdp, n_episodes, alpha, epsilon, seed):
    rng = np.random.default_rng(seed)
    Q = np.zeros((mdp.n_states, mdp.n_actions))
    returns = np.zeros(n_episodes)
    for i in range(n_episodes):
        s = int(rng.choice(mdp.n_states, p=mdp.mu0))
        for _ in range(mdp.max_steps):
            a = epsilon_greedy_action(Q[s], epsilon, rng)
            s2, r = mdp.sample(s, a, rng)
            returns[i] += r
            done = bool(mdp.terminal[s2])
            target = expected_sarsa_target(Q[s2], r, done, mdp.gamma, epsilon)
            Q[s, a] += alpha * (target - Q[s, a])
            s = s2
            if done:
                break
    return Q, returns


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
