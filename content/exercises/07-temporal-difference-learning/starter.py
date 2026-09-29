"""TD(0), batch TD, SARSA, and Q-learning.

Fill in the functions below, in order. `random_walk` is ready to use: it
builds the five-state random walk, whose true values are k / 6.

Batch functions take episodes as (states, rewards) pairs, where states has
one more entry than rewards: it ends with the terminal state. The learning
functions play their own episodes with one np.random.default_rng(seed).
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
    """One TD(0) update of a state-value table, in place.

    Args:
        V: state values, shape (S,).
        s: the state the step started in.
        r: the reward r_t for the step.
        s2: the state the step landed in.
        terminated: True if s2 is terminal. Then the target is r alone.
        alpha: the step size.
        gamma: the discount factor.

    Returns:
        V, with V[s] moved a fraction alpha toward r + gamma * V[s2].
    """
    raise NotImplementedError("td0_update")


def certainty_equivalence(episodes, n_states, gamma):
    """The exact values of the maximum-likelihood model fitted to the data.

    Args:
        episodes: a list of (states, rewards) pairs. states has length T + 1
            and ends with the terminal state; rewards has length T, and
            rewards[t] is paid on the step from states[t] to states[t + 1].
        n_states: the number of states S.
        gamma: the discount factor.

    Returns:
        V, shape (S,). Estimate each transition probability as the fraction
        of times states[t] = s was followed by s2, and each reward as the
        mean reward paid when leaving s. Then solve the Bellman equation of
        that model exactly. States that are never left get value 0.
    """
    raise NotImplementedError("certainty_equivalence")


def batch_td0(episodes, n_states, gamma, alpha, n_sweeps):
    """Batch TD(0): sweep a fixed dataset repeatedly until the values settle.

    Start from V = 0. In each sweep, hold V fixed, add up the TD(0) increment
    alpha * delta_t for every step of every episode, and then apply the sum.

    Args:
        episodes: a list of (states, rewards) pairs, as in
            certainty_equivalence. The last step of each episode ends in a
            terminal state.
        n_states: the number of states S.
        gamma: the discount factor.
        alpha: the step size.
        n_sweeps: how many sweeps over the dataset to make.

    Returns:
        V, shape (S,).
    """
    raise NotImplementedError("batch_td0")


def td0_prediction(mdp, pi, n_episodes, alpha, seed, v_init=0.5):
    """Online TD(0) prediction of V for policy pi.

    Start with V = v_init in every non-terminal state and 0 in terminal
    ones. Run n_episodes episodes, calling td0_update after every step.

    Args:
        mdp: an rlenvs.TabularMDP. Draw the start from mdp.mu0 and step with
            mdp.sample(s, a, rng). Stop at a terminal state or after
            mdp.max_steps steps.
        pi: the policy, shape (S, A).
        n_episodes: how many episodes to run.
        alpha: the step size.
        seed: seed for np.random.default_rng, the only source of randomness.
        v_init: the starting value of every non-terminal state.

    Returns:
        V, shape (S,).
    """
    raise NotImplementedError("td0_prediction")


def epsilon_greedy_action(q_row, epsilon, rng):
    """Pick an action epsilon-greedily from one row of Q.

    With probability epsilon, return a uniformly random action. Otherwise
    return a greedy action, chosen uniformly at random among ties.

    Args:
        q_row: shape (A,), the action values in the current state.
        epsilon: the exploration rate.
        rng: a np.random.Generator.

    Returns:
        An int action.
    """
    raise NotImplementedError("epsilon_greedy_action")


def sarsa_update(Q, s, a, r, s2, a2, terminated, alpha, gamma):
    """One SARSA update of an action-value table, in place.

    The target is r + gamma * Q[s2, a2], or r alone if terminated.

    Returns:
        Q, shape (S, A), with Q[s, a] moved a fraction alpha toward the target.
    """
    raise NotImplementedError("sarsa_update")


def q_learning_update(Q, s, a, r, s2, terminated, alpha, gamma):
    """One Q-learning update of an action-value table, in place.

    The target is r + gamma * max over actions of Q[s2], or r alone if
    terminated.

    Returns:
        Q, shape (S, A), with Q[s, a] moved a fraction alpha toward the target.
    """
    raise NotImplementedError("q_learning_update")


def sarsa(mdp, n_episodes, alpha, epsilon, seed):
    """SARSA with an epsilon-greedy policy, starting from Q = 0.

    Choose the first action before the loop. After each step, choose the
    next action a2 epsilon-greedily, update with it, and then take it.

    Args:
        mdp: an rlenvs.TabularMDP. Draw the start from mdp.mu0 and step with
            mdp.sample(s, a, rng). Stop at a terminal state or after
            mdp.max_steps steps.
        n_episodes: how many episodes to run.
        alpha: the step size.
        epsilon: the exploration rate.
        seed: seed for np.random.default_rng, the only source of randomness.

    Returns:
        (Q, returns): Q has shape (S, A), and returns has shape (n_episodes,),
        the undiscounted sum of rewards earned in each training episode.
    """
    raise NotImplementedError("sarsa")


def q_learning(mdp, n_episodes, alpha, epsilon, seed):
    """Q-learning with an epsilon-greedy behaviour policy, starting from Q = 0.

    Arguments and return values are the same as for sarsa.
    """
    raise NotImplementedError("q_learning")


if __name__ == "__main__":
    walk = random_walk()
    V = td0_prediction(walk, np.ones((walk.n_states, 1)), n_episodes=100, alpha=0.05, seed=0)
    print("TD(0) on the random walk after 100 episodes:", np.round(V[1:-1], 3))
    print("true values:                               ", np.round(np.arange(1, 6) / 6, 3))

    cliff = rlenvs.cliff_walking()
    start = cliff.names["start"]
    for name, learn in (("SARSA", sarsa), ("Q-learning", q_learning)):
        Q, returns = learn(cliff, n_episodes=500, alpha=0.1, epsilon=0.1, seed=0)
        v = rlenvs.policy_evaluation(cliff, rlenvs.greedy(Q))
        print(
            f"{name:>10}: greedy value from the start {v[start]:7.2f}, "
            f"mean training return over the last 100 episodes {returns[-100:].mean():7.2f}"
        )
