"""Reference solution for the temporal-difference learning exercise."""

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
    # A terminal next state has no future, so its value is left out of the target.
    target = r + gamma * V[s2] * (not terminated)
    V[s] += alpha * (target - V[s])
    return V


def td0_prediction(mdp, pi, n_episodes, alpha, seed, v_init=0.5):
    rng = np.random.default_rng(seed)
    V = np.where(mdp.terminal, 0.0, v_init)
    for _ in range(n_episodes):
        s = int(rng.choice(mdp.n_states, p=mdp.mu0))
        for _ in range(mdp.max_steps):
            a = int(rng.choice(mdp.n_actions, p=pi[s]))
            s2, r = mdp.sample(s, a, rng)
            done = bool(mdp.terminal[s2])
            td0_update(V, s, r, s2, done, alpha, mdp.gamma)
            s = s2
            if done:
                break
    return V


def batch_td0(episodes, n_states, gamma, alpha, n_sweeps):
    V = np.zeros(n_states)
    for _ in range(n_sweeps):
        # Sum every TD increment with V held fixed, then apply the total.
        delta = np.zeros(n_states)
        for states, rewards in episodes:
            T = len(rewards)
            for t in range(T):
                s, s2 = states[t], states[t + 1]
                done = t == T - 1
                delta[s] += alpha * (rewards[t] + gamma * V[s2] * (not done) - V[s])
        V += delta
    return V


if __name__ == "__main__":
    walk = random_walk()
    V = td0_prediction(walk, np.ones((walk.n_states, 1)), n_episodes=100, alpha=0.05, seed=0)
    print("TD(0) on the random walk after 100 episodes:", np.round(V[1:-1], 3))
    print("true values:                               ", np.round(np.arange(1, 6) / 6, 3))
