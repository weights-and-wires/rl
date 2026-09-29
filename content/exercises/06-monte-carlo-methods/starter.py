"""Monte Carlo prediction, control, and importance sampling.

Fill in the functions below, in order. `random_walk` and `generate_episode`
are ready to use: the first builds the five-state random walk, and the second
plays one episode of a policy and returns its states, actions, and rewards.

An episode is always a tuple of three arrays of the same length T:
states[t] is s_t, actions[t] is a_t, and rewards[t] is r_t, the reward for
taking a_t in s_t. The terminal state that ends the episode is not stored.
"""

import numpy as np

import rlenvs


def random_walk(n: int = 5) -> rlenvs.TabularMDP:
    """Sutton and Barto's random walk (Example 6.2) as a `TabularMDP`.

    States `1` to `n` are the walk and states `0` and `n + 1` are the two
    exits, both terminal. There is one action, and it moves left or right with
    probability one half each. Leaving on the right pays 1, and every other
    step pays 0. Episodes start in the middle state, and `gamma = 1`, so the
    true value of state `k` is `k / (n + 1)`.
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


def generate_episode(mdp, pi, rng, max_steps=1000):
    """Run one episode of policy `pi` in `mdp`, drawing randomness from `rng`.

    Returns `(states, actions, rewards)`, three arrays of the same length `T`.
    `states[t]` is `s_t`, `actions[t]` is `a_t`, and `rewards[t]` is `r_t`, the
    reward for taking `a_t` in `s_t`. The terminal state is not included.
    """
    states, actions, rewards = [], [], []
    s = int(rng.choice(mdp.n_states, p=mdp.mu0))
    for _ in range(max_steps):
        if mdp.terminal[s]:
            break
        a = int(rng.choice(mdp.n_actions, p=pi[s]))
        s2, r = mdp.sample(s, a, rng)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = s2
    return np.array(states, dtype=int), np.array(actions, dtype=int), np.array(rewards, dtype=float)


def discounted_returns(rewards, gamma):
    """The return G_t from every step of one episode.

    Args:
        rewards: shape (T,), where rewards[t] is r_t.
        gamma: the discount factor.

    Returns:
        Shape (T,), where entry t is G_t = r_t + gamma * r_{t+1} + ... +
        gamma^(T-1-t) * r_{T-1}.
    """
    raise NotImplementedError("discounted_returns")


def mc_prediction(episodes, n_states, gamma, first_visit=True):
    """Monte Carlo estimate of V from complete episodes.

    Args:
        episodes: a list of (states, actions, rewards) tuples, each three
            arrays of length T, as generate_episode returns them.
        n_states: the number of states S.
        gamma: the discount factor.
        first_visit: if True, average the return that follows only the
            first visit to each state in each episode. If False, average the
            returns that follow every visit.

    Returns:
        V, shape (S,): the average return for each state. A state that no
        episode visits gets 0.
    """
    raise NotImplementedError("mc_prediction")


def epsilon_soft(Q, epsilon):
    """The epsilon-greedy policy for an action-value table.

    Args:
        Q: shape (S, A).
        epsilon: the exploration rate, in [0, 1].

    Returns:
        pi, shape (S, A). In each row, the first action with the largest Q
        (the one np.argmax picks) gets 1 - epsilon + epsilon / A, and every
        other action gets epsilon / A.
    """
    raise NotImplementedError("epsilon_soft")


def mc_control(mdp, n_episodes, epsilon, alpha, seed):
    """On-policy first-visit Monte Carlo control with an epsilon-greedy policy.

    Start from Q = 0. For each episode, act epsilon-greedily on the current Q
    (break ties between best actions at random), then move Q[s, a] a
    fraction alpha toward the return that followed the first visit to each
    pair (s, a) in the episode.

    Args:
        mdp: an rlenvs.TabularMDP. Use mdp.mu0 for the start state,
            mdp.sample(s, a, rng) for each step, mdp.terminal to stop, and
            mdp.max_steps as a cap on the episode length.
        n_episodes: how many episodes to run.
        epsilon: the exploration rate.
        alpha: the constant step size.
        seed: seed for np.random.default_rng, the only source of randomness.

    Returns:
        Q, shape (S, A).
    """
    raise NotImplementedError("mc_control")


def importance_ratios(states, actions, pi, b):
    """The importance-sampling ratio from every step to the end of an episode.

    Args:
        states: shape (T,), the states s_0, ..., s_{T-1}.
        actions: shape (T,), the actions a_0, ..., a_{T-1}.
        pi: the target policy, shape (S, A).
        b: the behaviour policy that chose the actions, shape (S, A).

    Returns:
        Shape (T,), where entry t is rho_{t:T-1}, the product over k from t
        to T-1 of pi(a_k | s_k) / b(a_k | s_k).
    """
    raise NotImplementedError("importance_ratios")


def importance_sampling_estimates(episodes, pi, b, gamma):
    """Ordinary and weighted importance-sampling estimates of the start value.

    Every episode was generated by b from the same start state. Each one
    contributes its full return G_0, weighted by rho_{0:T-1}.

    Args:
        episodes: a list of (states, actions, rewards) tuples.
        pi: the target policy, shape (S, A).
        b: the behaviour policy, shape (S, A).
        gamma: the discount factor.

    Returns:
        (ois, wis), two floats. ois is the mean of rho * G over episodes.
        wis is the sum of rho * G divided by the sum of rho, or 0.0 when
        every weight is zero.
    """
    raise NotImplementedError("importance_sampling_estimates")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    walk = random_walk()
    one_action = np.ones((walk.n_states, 1))
    episodes = [generate_episode(walk, one_action, rng) for _ in range(1000)]
    V = mc_prediction(episodes, walk.n_states, gamma=1.0)
    print("first-visit MC on the random walk after 1000 episodes")
    print("  estimate:", np.round(V[1:-1], 3))
    print("  truth:   ", np.round(np.arange(1, 6) / 6, 3))

    grid = rlenvs.gridworld(["....", ".#..", "S..G"])
    Q = mc_control(grid, n_episodes=1000, epsilon=0.2, alpha=0.1, seed=0)
    v = rlenvs.policy_evaluation(grid, epsilon_soft(Q, 0.2))
    start = grid.names["start"]
    print(f"epsilon-soft MC control: value from the start {v[start]:.3f}")
