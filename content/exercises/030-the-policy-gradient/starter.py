"""The policy gradient and REINFORCE.

Fill in the functions below. A tabular softmax policy has one logit per
state and action, theta of shape (S, A), and pi(a | s) is the softmax of row
theta[s]. You write the policy and its score, the objective J(theta), and
the REINFORCE estimator, then check that REINFORCE's average matches a
gradient measured by nudging each logit and watching J change.

`sample_episodes` is given. Use it inside `reinforce_gradient`.
"""

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
    """The tabular softmax policy.

    Args:
        theta: logits, shape (S, A).

    Returns:
        pi, shape (S, A), where pi[s] is the softmax of theta[s]. Each row
        sums to one. Subtract each row's maximum before exponentiating, so
        large logits don't overflow.
    """
    raise NotImplementedError("softmax_policy")


def log_policy_grad(theta, s, a):
    """The score: the gradient of log pi(a | s) with respect to every logit.

    Args:
        theta: logits, shape (S, A).
        s: a state index.
        a: an action index.

    Returns:
        An array of shape (S, A). Only row s is nonzero.
    """
    raise NotImplementedError("log_policy_grad")


def objective(mdp, theta):
    """The exact objective J(theta) = sum_s mu0(s) V^pi(s).

    Args:
        mdp: a `rlenvs.TabularMDP`.
        theta: logits, shape (S, A).

    Returns:
        J as a float. Use `rlenvs.policy_evaluation` for V^pi.
    """
    raise NotImplementedError("objective")


def discounted_returns(rewards, gamma):
    """Returns-to-go G_t = r_t + gamma * G_{t+1}, with G = 0 after the last step.

    Args:
        rewards: shape (T,) for one episode, or (N, T) for a batch. Time runs
            along the last axis.
        gamma: the discount factor.

    Returns:
        G, the same shape as rewards.
    """
    raise NotImplementedError("discounted_returns")


def reinforce_estimate(theta, states, actions, rewards, gamma):
    """One episode's REINFORCE estimate of the gradient of J.

    sum_t gamma^t * G_t * grad log pi(a_t | s_t).

    Args:
        theta: logits, shape (S, A).
        states: s_0, ..., s_{T-1}, shape (T,), integers.
        actions: a_0, ..., a_{T-1}, shape (T,), integers.
        rewards: r_0, ..., r_{T-1}, shape (T,). r_t is the reward for a_t in s_t.
        gamma: the discount factor.

    Returns:
        The estimate, shape (S, A).
    """
    raise NotImplementedError("reinforce_estimate")


def reinforce_gradient(mdp, theta, episodes, horizon, seed):
    """Per-episode REINFORCE estimates from fresh episodes of the policy.

    Args:
        mdp: a `rlenvs.TabularMDP` with no terminal states.
        theta: logits, shape (S, A).
        episodes: the number of episodes, N.
        horizon: the number of steps per episode, H.
        seed: seeds `np.random.default_rng`, which you pass to
            `sample_episodes`.

    Returns:
        An array of shape (N, S, A). Row n is episode n's estimate, so the
        mean over axis 0 estimates the gradient and the spread gives its
        standard error.
    """
    raise NotImplementedError("reinforce_gradient")


if __name__ == "__main__":
    mdp = rlenvs.random_mdp(n_states=3, n_actions=2, gamma=0.9, seed=0)
    theta = np.random.default_rng(1).normal(size=(3, 2))

    # The reference: nudge each logit by a tiny amount and watch J change.
    eps = 1e-6
    reference = np.zeros_like(theta)
    for s in range(3):
        for a in range(2):
            up, down = theta.copy(), theta.copy()
            up[s, a] += eps
            down[s, a] -= eps
            reference[s, a] = (objective(mdp, up) - objective(mdp, down)) / (2 * eps)

    estimates = reinforce_gradient(mdp, theta, episodes=2000, horizon=100, seed=0)
    mean = estimates.mean(axis=0)
    stderr = estimates.std(axis=0, ddof=1) / np.sqrt(len(estimates))

    print(f"J(theta) = {objective(mdp, theta):.4f}")
    print("state action  finite diff  REINFORCE   std err   z")
    for s in range(3):
        for a in range(2):
            z = (mean[s, a] - reference[s, a]) / stderr[s, a]
            print(
                f"  {s}     {a}    {reference[s, a]:+.4f}     {mean[s, a]:+.4f}    "
                f"{stderr[s, a]:.4f}  {z:+.2f}"
            )
