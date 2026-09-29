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
    target = r + gamma * V[s2] * (not terminated)
    V[s] += alpha * (target - V[s])
    return V


def certainty_equivalence(episodes, n_states, gamma):
    # Count transitions and total rewards to fit the maximum-likelihood model.
    counts = np.zeros((n_states, n_states))
    reward_sum = np.zeros(n_states)
    for states, rewards in episodes:
        for t, r in enumerate(rewards):
            counts[states[t], states[t + 1]] += 1
            reward_sum[states[t]] += r
    visits = counts.sum(axis=1)
    seen = visits > 0
    P = np.zeros((n_states, n_states))
    P[seen] = counts[seen] / visits[seen, None]
    r_hat = np.zeros(n_states)
    r_hat[seen] = reward_sum[seen] / visits[seen]
    # States never left (terminal ones) keep value 0.
    idx = np.flatnonzero(seen)
    V = np.zeros(n_states)
    M = np.eye(len(idx)) - gamma * P[np.ix_(idx, idx)]
    V[idx] = np.linalg.solve(M, r_hat[idx])
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


def epsilon_greedy_action(q_row, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(len(q_row)))
    best = np.flatnonzero(q_row == q_row.max())
    return int(rng.choice(best))


def sarsa_update(Q, s, a, r, s2, a2, terminated, alpha, gamma):
    target = r + gamma * Q[s2, a2] * (not terminated)
    Q[s, a] += alpha * (target - Q[s, a])
    return Q


def q_learning_update(Q, s, a, r, s2, terminated, alpha, gamma):
    target = r + gamma * Q[s2].max() * (not terminated)
    Q[s, a] += alpha * (target - Q[s, a])
    return Q


def sarsa(mdp, n_episodes, alpha, epsilon, seed):
    rng = np.random.default_rng(seed)
    Q = np.zeros((mdp.n_states, mdp.n_actions))
    returns = np.zeros(n_episodes)
    for i in range(n_episodes):
        s = int(rng.choice(mdp.n_states, p=mdp.mu0))
        a = epsilon_greedy_action(Q[s], epsilon, rng)
        for _ in range(mdp.max_steps):
            s2, r = mdp.sample(s, a, rng)
            returns[i] += r
            done = bool(mdp.terminal[s2])
            # Pick the next action first: SARSA's target uses the action it takes.
            a2 = epsilon_greedy_action(Q[s2], epsilon, rng)
            sarsa_update(Q, s, a, r, s2, a2, done, alpha, mdp.gamma)
            s, a = s2, a2
            if done:
                break
    return Q, returns


def q_learning(mdp, n_episodes, alpha, epsilon, seed):
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
            q_learning_update(Q, s, a, r, s2, done, alpha, mdp.gamma)
            s = s2
            if done:
                break
    return Q, returns


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
