"""n-step returns and n-step TD.

Fill in the functions below, in order. `random_walk` and `generate_episode`
are ready to use. Episodes are `(states, rewards)` pairs, with `states` one
entry longer than `rewards` because it ends at the terminal state.
"""

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
    """Return the n-step return from time `t` as a float.

    `rewards` holds the `T` rewards of one episode, and `values` holds the
    current estimates `V(s_0)` to `V(s_{T-1})`, one per visited state. The
    return is `rewards[t] + gamma * rewards[t + 1] + ...` for `n` terms, plus
    `gamma ** n * values[t + n]`. When `t + n >= T` the episode ends first,
    there is nothing to bootstrap from, and the result is the full return.
    """
    raise NotImplementedError("sum the rewards in the window, then add the bootstrap if the episode has not ended")


def expected_n_step_return(P_pi, r_pi, V, n, gamma):
    """Return the expected n-step return from every state, shape `(S,)`.

    `P_pi` is the `(S, S)` state-to-state matrix under the policy, `r_pi` the
    `(S,)` expected reward, and `V` the `(S,)` estimates to bootstrap from.
    The answer is `sum_{k<n} gamma**k * P_pi^k @ r_pi + gamma**n * P_pi^n @ V`.
    """
    raise NotImplementedError("add the discounted reward terms, then the bootstrap term, using powers of P_pi")


def n_step_td(episodes, n_states, n, alpha, gamma, v_init=0.0):
    """Run n-step TD over stored episodes and return `V`, shape `(n_states,)`.

    Each episode is `(states, rewards)`, with `states` one entry longer than
    `rewards`: it ends with the terminal state. Start `V` at `v_init`, except
    that every state an episode ends in has value 0. Loop over the steps
    `tau` of each episode in order, and move `V[states[tau]]` a fraction
    `alpha` toward the n-step return computed from the current `V`.
    """
    raise NotImplementedError("for each step, compute the n-step return from the current V and update")


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
