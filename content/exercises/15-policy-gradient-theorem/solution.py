"""Reference solution for the policy gradient theorem exercise."""

import numpy as np

import rlenvs


def sample_episodes(mdp, pi, episodes, horizon, rng):
    """Roll out `episodes` episodes of `horizon` steps each, all at once.

    Given; you don't need to change it. Each step draws an action from
    `pi[s]` and a next state from `mdp.P[s, a]` by inverting the cumulative
    distribution, for every episode in parallel.

    Args:
        mdp: a `rlenvs.TabularMDP` with no terminal states.
        pi: the policy, shape (S, A), rows summing to one.
        episodes: the number of episodes, N.
        horizon: the number of steps per episode, H.
        rng: a `np.random.Generator`.

    Returns:
        (states, actions, rewards), each shape (N, H). `rewards[n, t]` is
        `mdp.R[states[n, t], actions[n, t]]`, the reward r_t for a_t in s_t.
    """
    S, A = pi.shape
    states = np.zeros((episodes, horizon), dtype=int)
    actions = np.zeros((episodes, horizon), dtype=int)
    rewards = np.zeros((episodes, horizon))
    s = (rng.random(episodes)[:, None] > np.cumsum(mdp.mu0)[None, :]).sum(axis=1)
    s = np.minimum(s, S - 1)
    for t in range(horizon):
        a = (rng.random(episodes)[:, None] > np.cumsum(pi[s], axis=1)).sum(axis=1)
        a = np.minimum(a, A - 1)
        states[:, t], actions[:, t], rewards[:, t] = s, a, mdp.R[s, a]
        s2 = (rng.random(episodes)[:, None] > np.cumsum(mdp.P[s, a], axis=1)).sum(axis=1)
        s = np.minimum(s2, S - 1)
    return states, actions, rewards


def softmax_policy(theta):
    theta = np.asarray(theta, dtype=float)
    z = theta - theta.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def log_policy_grad(theta, s, a):
    pi = softmax_policy(theta)
    grad = np.zeros_like(pi)
    grad[s] = -pi[s]
    grad[s, a] += 1.0
    return grad


def objective(mdp, theta):
    v = rlenvs.policy_evaluation(mdp, softmax_policy(theta))
    return float(mdp.mu0 @ v)


def exact_gradient(mdp, theta):
    pi = softmax_policy(theta)
    v = rlenvs.policy_evaluation(mdp, pi)
    adv = rlenvs.q_from_v(mdp, v) - v[:, None]
    # state_distribution is normalised to sum to one. With no terminal
    # states the unnormalised visitation sums to 1 / (1 - gamma).
    d = rlenvs.state_distribution(mdp, pi) / (1.0 - mdp.gamma)
    return d[:, None] * pi * adv


def discounted_returns(rewards, gamma):
    rewards = np.asarray(rewards, dtype=float)
    returns = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[:-1])
    for t in range(rewards.shape[-1] - 1, -1, -1):
        running = rewards[..., t] + gamma * running
        returns[..., t] = running
    return returns


def reinforce_estimate(theta, states, actions, rewards, gamma):
    pi = softmax_policy(theta)
    states = np.asarray(states, dtype=int)
    actions = np.asarray(actions, dtype=int)
    T = len(states)
    weights = gamma ** np.arange(T) * discounted_returns(rewards, gamma)
    scores = -pi[states]
    scores[np.arange(T), actions] += 1.0
    grad = np.zeros_like(pi)
    np.add.at(grad, states, weights[:, None] * scores)
    return grad


def reinforce_gradient(mdp, theta, episodes, horizon, seed):
    rng = np.random.default_rng(seed)
    states, actions, rewards = sample_episodes(
        mdp, softmax_policy(theta), episodes, horizon, rng
    )
    return np.stack(
        [
            reinforce_estimate(theta, states[n], actions[n], rewards[n], mdp.gamma)
            for n in range(episodes)
        ]
    )


if __name__ == "__main__":
    mdp = rlenvs.random_mdp(n_states=3, n_actions=2, gamma=0.9, seed=0)
    theta = np.random.default_rng(1).normal(size=(3, 2))

    exact = exact_gradient(mdp, theta)
    estimates = reinforce_gradient(mdp, theta, episodes=2000, horizon=100, seed=0)
    mean = estimates.mean(axis=0)
    stderr = estimates.std(axis=0, ddof=1) / np.sqrt(len(estimates))

    print(f"J(theta) = {objective(mdp, theta):.4f}")
    print("state action   exact    REINFORCE   std err   z")
    for s in range(3):
        for a in range(2):
            z = (mean[s, a] - exact[s, a]) / stderr[s, a]
            print(
                f"  {s}     {a}    {exact[s, a]:+.4f}   {mean[s, a]:+.4f}    "
                f"{stderr[s, a]:.4f}  {z:+.2f}"
            )
