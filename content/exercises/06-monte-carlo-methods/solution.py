"""Reference solution for the Monte Carlo methods exercise."""

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
    rewards = np.asarray(rewards, dtype=float)
    G = np.zeros_like(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        G[t] = running
    return G


def mc_prediction(episodes, n_states, gamma, first_visit=True):
    total = np.zeros(n_states)
    count = np.zeros(n_states)
    for states, _, rewards in episodes:
        G = discounted_returns(rewards, gamma)
        seen = set()
        for t, s in enumerate(states):
            if first_visit:
                if s in seen:
                    continue
                seen.add(s)
            total[s] += G[t]
            count[s] += 1
    return np.divide(total, count, out=np.zeros(n_states), where=count > 0)


def epsilon_soft(Q, epsilon):
    Q = np.asarray(Q, dtype=float)
    S, A = Q.shape
    pi = np.full((S, A), epsilon / A)
    pi[np.arange(S), Q.argmax(axis=1)] += 1.0 - epsilon
    return pi


def mc_control(mdp, n_episodes, epsilon, alpha, seed):
    rng = np.random.default_rng(seed)
    S, A = mdp.n_states, mdp.n_actions
    Q = np.zeros((S, A))
    for _ in range(n_episodes):
        # Act epsilon-greedily on the current Q, breaking ties at random.
        states, actions, rewards = [], [], []
        s = int(rng.choice(S, p=mdp.mu0))
        for _ in range(mdp.max_steps):
            if mdp.terminal[s]:
                break
            if rng.random() < epsilon:
                a = int(rng.integers(A))
            else:
                a = int(rng.choice(np.flatnonzero(Q[s] == Q[s].max())))
            s2, r = mdp.sample(s, a, rng)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            s = s2
        # Move each first-visited pair a fraction alpha toward its return.
        G = discounted_returns(rewards, mdp.gamma)
        seen = set()
        for t, (s, a) in enumerate(zip(states, actions)):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            Q[s, a] += alpha * (G[t] - Q[s, a])
    return Q


def importance_ratios(states, actions, pi, b):
    step = pi[states, actions] / b[states, actions]
    # rho_{t:T-1} is the product of the step ratios from t to the end.
    return np.cumprod(step[::-1])[::-1]


def importance_sampling_estimates(episodes, pi, b, gamma):
    weights, returns = [], []
    for states, actions, rewards in episodes:
        weights.append(importance_ratios(states, actions, pi, b)[0])
        returns.append(discounted_returns(rewards, gamma)[0])
    weights = np.array(weights)
    returns = np.array(returns)
    ois = float(np.mean(weights * returns))
    total = weights.sum()
    wis = float((weights * returns).sum() / total) if total > 0 else 0.0
    return ois, wis


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
