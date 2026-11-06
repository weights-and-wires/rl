"""Reference solution for the lambda-return and eligibility traces exercise."""

import numpy as np

import rlenvs


def random_walk(n: int = 19, left_reward: float = -1.0) -> rlenvs.TabularMDP:
    """A random walk of `n` states between two terminal exits, as a `TabularMDP`.

    States `1` to `n` are the walk, and states `0` and `n + 1` are the exits.
    The one action moves left or right with probability one half. Leaving on
    the left pays `left_reward`, leaving on the right pays 1, and every other
    step pays 0. Episodes start in the middle, and `gamma = 1`.

    With the defaults, this is Sutton and Barto's 19-state walk (Figure 7.2),
    and the true value of state `k` is `(k - 10) / 10`.
    """
    S = n + 2
    P = np.zeros((S, 1, S))
    R3 = np.zeros((S, 1, S))
    for s in range(1, n + 1):
        P[s, 0, s - 1] = 0.5
        P[s, 0, s + 1] = 0.5
    P[0, 0, 0] = 1.0
    P[S - 1, 0, S - 1] = 1.0
    R3[1, 0, 0] = left_reward
    R3[n, 0, n + 1] = 1.0
    terminal = np.zeros(S, dtype=bool)
    terminal[[0, S - 1]] = True
    mu0 = np.zeros(S)
    mu0[(n + 1) // 2] = 1.0
    R = (P * R3).sum(axis=2)
    return rlenvs.TabularMDP(P=P, R=R, gamma=1.0, mu0=mu0, terminal=terminal, R3=R3)


def generate_episode(mdp, rng, pi=None):
    """Play one episode and return `(states, rewards)`.

    `states` has length `T + 1` and ends with the terminal state. `rewards`
    has length `T`, and `rewards[t]` is paid on the step from `states[t]`.
    The policy `pi` defaults to action 0 everywhere.
    """
    s = int(rng.choice(mdp.n_states, p=mdp.mu0))
    states, rewards = [s], []
    while not mdp.terminal[s] and len(rewards) < mdp.max_steps:
        a = 0 if pi is None else int(rng.choice(mdp.n_actions, p=pi[s]))
        s, r = mdp.sample(s, a, rng)
        states.append(s)
        rewards.append(r)
    return np.array(states), np.array(rewards, dtype=float)


def lambda_return(rewards, values, gamma, lam):
    rewards = np.asarray(rewards, dtype=float)
    values = np.asarray(values, dtype=float)
    T = len(rewards)
    G = np.zeros(T)
    # G^lam_t = r_t + gamma * ((1 - lam) * V(s_{t+1}) + lam * G^lam_{t+1}),
    # with the terminal state's value and lambda-return both zero.
    next_value, next_G = 0.0, 0.0
    for t in range(T - 1, -1, -1):
        G[t] = rewards[t] + gamma * ((1 - lam) * next_value + lam * next_G)
        next_value, next_G = values[t], G[t]
    return G


def td_lambda_offline(states, rewards, V, alpha, gamma, lam):
    V = np.asarray(V, dtype=float)
    z = np.zeros_like(V)
    delta_V = np.zeros_like(V)
    T = len(rewards)
    for t in range(T):
        s, s2 = states[t], states[t + 1]
        z *= gamma * lam
        z[s] += 1.0
        bootstrap = V[s2] if t < T - 1 else 0.0
        delta = rewards[t] + gamma * bootstrap - V[s]
        delta_V += alpha * delta * z
    return delta_V


def td_lambda(episodes, n_states, alpha, gamma, lam, v_init=0.0):
    V = np.full(n_states, float(v_init))
    for states, _ in episodes:
        V[states[-1]] = 0.0
    for states, rewards in episodes:
        z = np.zeros(n_states)
        T = len(rewards)
        for t in range(T):
            s, s2 = states[t], states[t + 1]
            z *= gamma * lam
            z[s] += 1.0
            bootstrap = V[s2] if t < T - 1 else 0.0
            delta = rewards[t] + gamma * bootstrap - V[s]
            V += alpha * delta * z
    return V


if __name__ == "__main__":
    r = np.array([1.0, 0.0, 2.0])
    v = np.array([0.5, 1.0, 0.8])
    print("lambda-returns with lambda = 0.5:", np.round(lambda_return(r, v, 0.9, 0.5), 4))

    walk = random_walk()
    truth = (np.arange(1, 20) - 10) / 10
    rng = np.random.default_rng(0)
    runs = [[generate_episode(walk, rng) for _ in range(10)] for _ in range(20)]
    print("RMS error after 10 episodes on the 19-state walk, averaged over 20 runs:")
    for lam in (0.0, 0.4, 0.8):
        err = np.mean([np.sqrt(np.mean((td_lambda(eps, walk.n_states, 0.2, 1.0, lam)[1:20] - truth) ** 2)) for eps in runs])
        print(f"  TD(lambda), lambda = {lam:.1f}, alpha = 0.2: {err:.3f}")
