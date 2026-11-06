"""The lambda-return and TD(lambda) with eligibility traces.

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


def lambda_return(rewards, values, gamma, lam):
    """Return the lambda-return from every step of one episode, shape `(T,)`.

    `rewards` has length `T`, and `values` holds the current estimates
    `V(s_0)` to `V(s_{T-1})`. Use the backward recursion
    `G[t] = rewards[t] + gamma * ((1 - lam) * V(s_{t+1}) + lam * G[t + 1])`,
    where the terminal state's value and the return after the last step are 0.
    """
    raise NotImplementedError("run the recursion backward from the last step")


def td_lambda_offline(states, rewards, V, alpha, gamma, lam):
    """Return the total TD(lambda) increment for one episode, shape `(S,)`.

    Hold `V` fixed and do not change it. At each step, decay the trace by
    `gamma * lam`, add 1 at the current state, compute the TD error from the
    fixed `V` (the last step has no next state to bootstrap from), and add
    `alpha * delta * z` to a separate total. `states` has length `T + 1` and
    ends with the terminal state.
    """
    raise NotImplementedError("decay the trace, bump the current state, and credit the TD error to every state in proportion to its trace")


def td_lambda(episodes, n_states, alpha, gamma, lam, v_init=0.0):
    """Run online TD(lambda) over stored episodes and return `V`, shape `(n_states,)`.

    Start `V` at `v_init`, except that every state an episode ends in has
    value 0. Reset the trace to zeros at the start of each episode. At every
    step, decay and bump the trace as in `td_lambda_offline`, but add
    `alpha * delta * z` straight into `V`, so later steps see the update.
    """
    raise NotImplementedError("the same loop as td_lambda_offline, adding into V at every step")


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
