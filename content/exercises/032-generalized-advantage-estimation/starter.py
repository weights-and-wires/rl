"""Generalised advantage estimation.

Fill in the functions below. Every function works on arrays whose first axis
is time: shape (T,) for one environment, or (T, N) for N environments side by
side. The flags follow the Gym convention of `rlenvs`:

- terminated[t] is True when the step from s_t reached a terminal state. There
  is no future after it, so nothing bootstraps.
- truncated[t] is True when a time limit or the end of the data cut the
  episode off after step t. The future still exists, so you bootstrap from
  next_values[t].

Either flag ends the episode, so advantages never flow back across it.
"""

import numpy as np

import rlenvs


def td_errors(rewards, values, next_values, terminated, gamma):
    """One-step TD errors, delta_t = r_t + gamma * V(s_{t+1}) * (1 - terminated_t) - V(s_t).

    Args:
        rewards: r_t, shape (T,) or (T, N).
        values: V(s_t), same shape.
        next_values: V(s_{t+1}), the value of the state each step reached,
            same shape.
        terminated: booleans, same shape.
        gamma: the discount factor.

    Returns:
        delta, the same shape as rewards.
    """
    raise NotImplementedError("td_errors")


def next_state_values(values, last_value, done, final_values):
    """Build next_values for a rollout from an environment that resets itself.

    When an episode ends at step t, the environment resets, so values[t + 1]
    is the value of the next episode's first state. The value of the state
    step t actually reached is final_values[t].

    Args:
        values: V(s_t) for t = 0..T-1, shape (T,).
        last_value: V of the observation after the last step, a float. Used
            when the last step did not end an episode.
        done: booleans, shape (T,): terminated or truncated at step t.
        final_values: shape (T,). final_values[t] is the value of the state
            step t reached. Only entries where done[t] is True are meaningful.

    Returns:
        next_values, shape (T,).
    """
    raise NotImplementedError("next_state_values")


def gae(rewards, values, next_values, terminated, truncated, gamma, lam):
    """Generalised advantage estimates by backward recursion.

    A_t = delta_t + gamma * lam * A_{t+1}, except that the recursion restarts
    from zero wherever an episode ended at step t, and A_T = 0 past the end.

    Args:
        rewards, values, next_values: shape (T,) or (T, N).
        terminated, truncated: booleans, same shape.
        gamma: the discount factor.
        lam: the GAE parameter lambda, in [0, 1].

    Returns:
        The advantages, the same shape as rewards.
    """
    raise NotImplementedError("gae")


def lambda_returns(rewards, values, next_values, terminated, truncated, gamma, lam):
    """The value targets that go with gae: the advantages plus the values.

    Args:
        The same as gae.

    Returns:
        An array the same shape as rewards.
    """
    raise NotImplementedError("lambda_returns")


if __name__ == "__main__":
    rewards = np.array([1.0, 0.0, 2.0])
    values = np.array([0.5, 1.0, 1.5])
    gamma, lam = 0.9, 0.8

    # The episode ends at the third step. Either the world ended it, or a time
    # limit cut it off at a state whose value is 1.0.
    next_values = np.array([1.0, 1.5, 1.0])
    never = np.array([False, False, False])
    at_end = np.array([False, False, True])
    print("delta (terminated):   ", td_errors(rewards, values, next_values, at_end, gamma))
    print("GAE   (terminated):   ", gae(rewards, values, next_values, at_end, never, gamma, lam))
    print("GAE   (truncated):    ", gae(rewards, values, next_values, never, at_end, gamma, lam))
    print("lambda=1 (terminated):", gae(rewards, values, next_values, at_end, never, gamma, 1.0))
    print("lambda-returns:       ", lambda_returns(rewards, values, next_values, at_end, never, gamma, lam))

    # With the true V^pi as the critic, GAE averages to A^pi(s, a) for any lambda.
    mdp = rlenvs.random_mdp(n_states=4, n_actions=2, gamma=0.9, seed=0)
    pi = rlenvs.uniform_policy(mdp)
    v = rlenvs.policy_evaluation(mdp, pi)
    adv = rlenvs.q_from_v(mdp, v) - v[:, None]
    rng = np.random.default_rng(0)
    H, N, s0, a0 = 6, 4000, 0, 1
    firsts = {0.0: [], 0.5: [], 1.0: []}
    for _ in range(N):
        s, a = s0, a0
        rs, vs, nvs = np.zeros(H), np.zeros(H), np.zeros(H)
        for t in range(H):
            s2, r = mdp.sample(s, a, rng)
            rs[t], vs[t], nvs[t] = r, v[s], v[s2]
            s, a = s2, int(rng.choice(2, p=pi[s2]))
        term = np.zeros(H, dtype=bool)
        trunc = np.zeros(H, dtype=bool)
        trunc[-1] = True  # a time limit cuts the rollout after H steps
        for lam_ in firsts:
            firsts[lam_].append(gae(rs, vs, nvs, term, trunc, 0.9, lam_)[0])
    print(f"\nA^pi({s0}, {a0}) = {adv[s0, a0]:+.4f}")
    for lam_, xs in firsts.items():
        xs = np.array(xs)
        print(f"  lambda = {lam_:.1f}: mean GAE {xs.mean():+.4f}, std err {xs.std(ddof=1) / np.sqrt(N):.4f}")
