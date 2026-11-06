"""Reference solution for the n-step returns exercise."""

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


def n_step_return(rewards, values, t, n, gamma):
    rewards = np.asarray(rewards, dtype=float)
    T = len(rewards)
    end = min(t + n, T)
    G = sum(gamma ** (k - t) * rewards[k] for k in range(t, end))
    if t + n < T:
        G += gamma**n * values[t + n]
    return float(G)


def expected_n_step_return(P_pi, r_pi, V, n, gamma):
    P_pi = np.asarray(P_pi, dtype=float)
    out = np.zeros(len(r_pi))
    step = np.eye(len(r_pi))  # P_pi^k
    for k in range(n):
        out += gamma**k * step @ r_pi
        step = step @ P_pi
    return out + gamma**n * step @ V


def n_step_td(episodes, n_states, n, alpha, gamma, v_init=0.0):
    V = np.full(n_states, float(v_init))
    # Terminal states end the episodes; their value stays zero.
    for states, _ in episodes:
        V[states[-1]] = 0.0
    for states, rewards in episodes:
        T = len(rewards)
        # Updating s_tau in order, each with the current V, matches the online
        # algorithm: s_tau's update happens once r_{tau+n-1} has arrived.
        for tau in range(T):
            end = min(tau + n, T)
            G = sum(gamma ** (k - tau) * rewards[k] for k in range(tau, end))
            if tau + n < T:
                G += gamma**n * V[states[tau + n]]
            V[states[tau]] += alpha * (G - V[states[tau]])
    return V


if __name__ == "__main__":
    r = np.array([1.0, 0.0, 2.0])
    v = np.array([0.5, 1.0, 0.8])
    print("n-step returns from t = 0:", [round(n_step_return(r, v, 0, n, 0.9), 4) for n in (1, 2, 3)])

    walk = random_walk()
    truth = (np.arange(1, 20) - 10) / 10
    rng = np.random.default_rng(0)
    runs = [[generate_episode(walk, rng) for _ in range(10)] for _ in range(20)]
    print("RMS error after 10 episodes on the 19-state walk, averaged over 20 runs:")
    for n in (1, 4, 16):
        err = np.mean([np.sqrt(np.mean((n_step_td(eps, walk.n_states, n, 0.2, 1.0)[1:20] - truth) ** 2)) for eps in runs])
        print(f"  n-step TD, n = {n:2d}, alpha = 0.2: {err:.3f}")
