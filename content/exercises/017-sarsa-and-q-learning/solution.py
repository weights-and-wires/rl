"""Reference solution for the SARSA and Q-learning exercise."""

import numpy as np

import rlenvs


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
    cliff = rlenvs.cliff_walking()
    start = cliff.names["start"]
    for name, learn in (("SARSA", sarsa), ("Q-learning", q_learning)):
        Q, returns = learn(cliff, n_episodes=500, alpha=0.1, epsilon=0.1, seed=0)
        v = rlenvs.policy_evaluation(cliff, rlenvs.greedy(Q))
        print(
            f"{name:>10}: greedy value from the start {v[start]:7.2f}, "
            f"mean training return over the last 100 episodes {returns[-100:].mean():7.2f}"
        )
