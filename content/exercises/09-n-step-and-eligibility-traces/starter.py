"""n-step returns, lambda-returns, and TD(lambda) with eligibility traces.

Fill in the functions below, in order. `random_walk` builds the 19-state walk,
and `generate_episode` plays one episode of it.

An episode is a pair (states, rewards). states has length T + 1 and ends
with the terminal state; rewards has length T, and rewards[t] is r_t, paid on
the step from states[t] to states[t + 1]. The return functions take `values`,
the estimates V(s_0), ..., V(s_{T-1}) along the episode.
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
    """The n-step return G_{t:t+n} from one episode.

    Args:
        rewards: shape (T,), where rewards[t] is r_t.
        values: shape (T,), where values[k] is the current estimate V(s_k).
            The episode ends in a terminal state s_T whose value is 0, so it
            is not stored.
        t: the step to start from, 0 <= t < T.
        n: the number of rewards to add before bootstrapping, n >= 1.
        gamma: the discount factor.

    Returns:
        A float: the sum of gamma^k * r_{t+k} for k from 0 to n - 1, plus
        gamma^n * V(s_{t+n}) if t + n < T. If t + n >= T, the full return G_t.
    """
    raise NotImplementedError("n_step_return")


def lambda_return(rewards, values, gamma, lam):
    """The lambda-return G^lambda_t for every step of one episode.

    Args:
        rewards: shape (T,).
        values: shape (T,), the estimates V(s_0), ..., V(s_{T-1}). The
            terminal state's value is 0.
        gamma: the discount factor.
        lam: lambda, in [0, 1].

    Returns:
        Shape (T,). Entry t is (1 - lam) times the sum over n from 1 to
        T - t - 1 of lam^(n-1) * G_{t:t+n}, plus lam^(T-t-1) * G_t.
    """
    raise NotImplementedError("lambda_return")


def expected_n_step_return(P_pi, r_pi, V, n, gamma):
    """The exact expected n-step return from every state, for a known model.

    Args:
        P_pi: shape (S, S), the state-to-state transition matrix under pi.
        r_pi: shape (S,), the expected reward from each state under pi.
        V: shape (S,), the value estimate to bootstrap from.
        n: the number of steps, n >= 1.
        gamma: the discount factor.

    Returns:
        Shape (S,): the sum over k from 0 to n - 1 of gamma^k P_pi^k r_pi,
        plus gamma^n P_pi^n V.
    """
    raise NotImplementedError("expected_n_step_return")


def td_lambda_offline(states, rewards, V, alpha, gamma, lam):
    """The total TD(lambda) increment over one episode, with V held fixed.

    Use an accumulating eligibility trace z, starting at zero: at each step,
    decay z by gamma * lam, add 1 to z[s_t], and add alpha * delta_t * z to
    the running total. Do not change V during the episode.

    Args:
        states: shape (T + 1,), ending with the terminal state.
        rewards: shape (T,).
        V: shape (S,), the fixed value estimates. Treat the terminal state's
            value as 0, whatever V holds there.
        alpha: the step size.
        gamma: the discount factor.
        lam: lambda.

    Returns:
        Shape (S,): the sum over t of alpha * delta_t * z_t.
    """
    raise NotImplementedError("td_lambda_offline")


def n_step_td(episodes, n_states, n, alpha, gamma, v_init=0.0):
    """Online n-step TD prediction over a list of episodes.

    Start with V = v_init everywhere, except 0 in each episode's final,
    terminal state. For each episode, for tau = 0, 1, ..., T - 1 in order,
    move V[s_tau] a fraction alpha toward G_{tau:tau+n} computed with the
    current V. That order gives exactly the updates of the online algorithm.

    Args:
        episodes: a list of (states, rewards) pairs, states ending with the
            terminal state.
        n_states: the number of states S.
        n: the number of steps before bootstrapping.
        alpha: the step size.
        gamma: the discount factor.
        v_init: the starting value.

    Returns:
        V, shape (S,).
    """
    raise NotImplementedError("n_step_td")


def td_lambda(episodes, n_states, alpha, gamma, lam, v_init=0.0):
    """Online TD(lambda) with accumulating traces over a list of episodes.

    Start as in n_step_td. Reset the trace z to zero at the start of each
    episode. At each step, decay z by gamma * lam, add 1 to z[s_t], compute
    delta_t with the current V, and update every state: V += alpha * delta_t * z.

    Args:
        episodes: a list of (states, rewards) pairs.
        n_states: the number of states S.
        alpha: the step size.
        gamma: the discount factor.
        lam: lambda.
        v_init: the starting value.

    Returns:
        V, shape (S,).
    """
    raise NotImplementedError("td_lambda")


if __name__ == "__main__":
    r = np.array([1.0, 0.0, 2.0])
    v = np.array([0.5, 1.0, 0.8])
    print("n-step returns from t = 0:", [round(n_step_return(r, v, 0, n, 0.9), 4) for n in (1, 2, 3)])
    print("lambda-returns with lambda = 0.5:", np.round(lambda_return(r, v, 0.9, 0.5), 4))

    walk = random_walk()
    truth = (np.arange(1, 20) - 10) / 10
    rng = np.random.default_rng(0)
    runs = [[generate_episode(walk, rng) for _ in range(10)] for _ in range(20)]
    print("RMS error after 10 episodes on the 19-state walk, averaged over 20 runs:")
    for n in (1, 4, 16):
        err = np.mean([np.sqrt(np.mean((n_step_td(eps, walk.n_states, n, 0.2, 1.0)[1:20] - truth) ** 2)) for eps in runs])
        print(f"  n-step TD, n = {n:2d}, alpha = 0.2: {err:.3f}")
    for lam in (0.0, 0.4, 0.8):
        err = np.mean([np.sqrt(np.mean((td_lambda(eps, walk.n_states, 0.2, 1.0, lam)[1:20] - truth) ** 2)) for eps in runs])
        print(f"  TD(lambda), lambda = {lam:.1f}, alpha = 0.2: {err:.3f}")
