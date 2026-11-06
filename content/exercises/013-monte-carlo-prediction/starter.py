"""Monte Carlo prediction: estimate V by averaging the returns of complete episodes.

An episode is always a tuple of three arrays of the same length T:
states[t] is s_t, actions[t] is a_t, and rewards[t] is r_t, the reward for
taking a_t in s_t. The terminal state that ends the episode is not stored.

`random_walk`, `generate_episode`, and `discounted_returns` are given.
`discounted_returns` is the backward pass from the returns chapter.
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
    """The return G_t from every step of one episode, by a backward pass (given).

    Args:
        rewards: shape (T,), where rewards[t] is r_t.
        gamma: the discount factor.

    Returns:
        Shape (T,), where entry t is G_t = r_t + gamma * G_(t+1), with G_T = 0.
    """
    rewards = np.asarray(rewards, dtype=float)
    G = np.zeros_like(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        G[t] = running
    return G


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


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    walk = random_walk()
    one_action = np.ones((walk.n_states, 1))
    episodes = [generate_episode(walk, one_action, rng) for _ in range(2000)]
    print("first episode: states", episodes[0][0].tolist(), "rewards", episodes[0][2].tolist())
    for first in (True, False):
        V = mc_prediction(episodes, walk.n_states, gamma=1.0, first_visit=first)
        print("first-visit" if first else "every-visit", "MC after 2000 episodes")
        print("  estimate:", np.round(V[1:-1], 3))
    print("  truth:   ", np.round(np.arange(1, 6) / 6, 3))
