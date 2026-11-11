"""Off-policy evaluation from logs: IPS, doubly robust, trajectories, and FQE.

Logged contextual-bandit data is four arrays of length N: contexts x, actions a,
rewards r, and the behavior policy's table b of shape (K, A), where b[x, a] is
the probability that the logger chose a in context x. A target policy pi is a
(K, A) table too.

Logged trajectories are three (N, H) arrays: states[i, t] is s_t, actions[i, t]
is a_t, and rewards[i, t] is r_t. Every trajectory has exactly H steps.
"""

import numpy as np

import rlenvs


def finite_horizon_q(mdp, pi, horizon):
    """Exact action values with `horizon` steps left (given).

    Returns an array q of shape (H, S, A). q[t, s, a] is the expected discounted
    reward from step t to step H-1 if you take a in s at step t and follow pi
    afterward, with discount mdp.gamma. Step H-1 has no future, so
    q[H-1] = mdp.R.
    """
    q = np.zeros((horizon, mdp.n_states, mdp.n_actions))
    q[horizon - 1] = mdp.R
    for t in range(horizon - 2, -1, -1):
        v_next = (pi * q[t + 1]).sum(axis=1)
        q[t] = mdp.R + mdp.gamma * mdp.P @ v_next
    return q


def sample_episodes(mdp, b, n, horizon, rng):
    """Roll out the behavior policy b for n trajectories of H steps (given).

    Returns (states, actions, rewards), each of shape (n, H).
    """
    def draw(probs):
        u = rng.random(len(probs))
        k = (u[:, None] > np.cumsum(probs, axis=1)).sum(axis=1)
        return np.minimum(k, probs.shape[1] - 1)

    states = np.zeros((n, horizon), dtype=int)
    actions = np.zeros((n, horizon), dtype=int)
    rewards = np.zeros((n, horizon))
    s = rng.choice(mdp.n_states, size=n, p=mdp.mu0)
    for t in range(horizon):
        a = draw(b[s])
        states[:, t], actions[:, t], rewards[:, t] = s, a, mdp.R[s, a]
        s = draw(mdp.P[s, a])
    return states, actions, rewards


def effective_sample_size(w):
    """The number of equally weighted samples that carry the same information.

    Args:
        w: shape (N,), nonnegative importance weights.

    Returns:
        (sum of w) squared over (sum of w squared), as a float. Returns 0.0
        when every weight is zero.
    """
    raise NotImplementedError("effective_sample_size")


def ips(x, a, r, pi, b):
    """The inverse propensity score estimate of the target policy's value.

    Returns the mean over samples of r_i * pi[x_i, a_i] / b[x_i, a_i], a float.
    """
    raise NotImplementedError("ips")


def snips(x, a, r, pi, b):
    """The self-normalized IPS estimate: sum of w * r over sum of w.

    Returns a float, or 0.0 when every weight is zero.
    """
    raise NotImplementedError("snips")


def doubly_robust(x, a, r, pi, b, q_hat):
    """The doubly robust estimate: a model baseline plus a weighted correction.

    Args:
        q_hat: shape (K, A), a model of the expected reward in each context
            and action. It is fixed in advance, not fitted on this sample.

    Returns:
        The mean over samples of V_hat(x_i) + w_i * (r_i - q_hat[x_i, a_i]),
        where V_hat(x) = sum_a pi[x, a] * q_hat[x, a] and w_i = pi / b.
    """
    raise NotImplementedError("doubly_robust")


def per_decision_is(states, actions, rewards, pi, b, gamma):
    """Per-decision importance sampling for each trajectory.

    Returns shape (N,), where entry i is the sum over t of
    gamma**t * rho_{0:t} * r_t, and rho_{0:t} is the product of
    pi / b over steps 0 to t, inclusive.
    """
    raise NotImplementedError("per_decision_is")


def doubly_robust_trajectory(states, actions, rewards, pi, b, gamma, q):
    """The recursive doubly robust estimate of Jiang and Li (2016).

    Args:
        q: shape (H, S, A), a model of the action value with H - t steps left.

    Returns:
        Shape (N,). Start with 0 after the last step. Going backward, set
        est = V_hat[t, s_t] + rho_t * (r_t + gamma * est - q[t, s_t, a_t]),
        where rho_t = pi(a_t | s_t) / b(a_t | s_t) is one step's ratio.
    """
    raise NotImplementedError("doubly_robust_trajectory")


def fqe(s, a, r, s2, pi, gamma, n_states, n_actions, iters=300):
    """Tabular fitted Q evaluation of the target policy pi on logged transitions.

    Args:
        s, a, r, s2: shape (N,), one logged transition per entry. The MDP has
            no terminal states.
        pi: shape (S, A), the target policy.

    Returns:
        Shape (S, A). Start at zero. Each iteration sets every pair that
        appears in the data to the mean of r + gamma * sum_a' pi[s2, a'] * Q[s2, a']
        over its transitions. Pairs that never appear stay at zero.
    """
    raise NotImplementedError("fqe")


if __name__ == "__main__":
    # Logged bandit: three contexts, three actions, reward depends on both.
    rng = np.random.default_rng(0)
    K, A = 3, 3
    mu = rng.uniform(0, 1, size=(K, A))
    b = np.full((K, A), 1.0 / A)
    pi = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    p_x = np.array([0.5, 0.3, 0.2])
    truth = float(p_x @ (pi * mu).sum(axis=1))
    q_hat = mu + rng.normal(0, 0.1, size=mu.shape)
    est = {"ips": [], "snips": [], "dr": []}
    ess = []
    for _ in range(500):
        x = rng.choice(K, size=100, p=p_x)
        act = rng.choice(A, size=100)
        rew = mu[x, act] + rng.normal(0, 0.3, size=100)
        est["ips"].append(ips(x, act, rew, pi, b))
        est["snips"].append(snips(x, act, rew, pi, b))
        est["dr"].append(doubly_robust(x, act, rew, pi, b, q_hat))
        ess.append(effective_sample_size(pi[x, act] / b[x, act]))
    print(f"true value of pi: {truth:.3f}   mean effective sample size: {np.mean(ess):.1f} of 100")
    for name, v in est.items():
        print(f"{name:6s} mean {np.mean(v):.3f}   std {np.std(v):.3f}")

    # Logged trajectories on a random MDP.
    mdp = rlenvs.random_mdp(4, 2, gamma=0.9, seed=1)
    H = 5
    beh = np.array([[0.5, 0.5]] * 4)
    tgt = np.array([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]])
    q_exact = finite_horizon_q(mdp, tgt, H)
    exact = float(mdp.mu0 @ (tgt * q_exact[0]).sum(axis=1))
    S_, A_, R_ = sample_episodes(mdp, beh, 5000, H, rng)
    pd = per_decision_is(S_, A_, R_, tgt, beh, mdp.gamma)
    dr = doubly_robust_trajectory(S_, A_, R_, tgt, beh, mdp.gamma, q_exact)
    print(f"exact {H}-step value: {exact:.3f}")
    print(f"per-decision IS mean {pd.mean():.3f} std {pd.std():.3f}")
    print(f"doubly robust   mean {dr.mean():.3f} std {dr.std():.3f}")
