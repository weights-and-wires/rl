"""V-trace: off-policy targets for actors that lag behind the learner.

Every function works on one rollout of length T from one environment, so
arrays have shape (T,). The rollout comes from a behavior policy mu, the
policy an actor used when it acted. The learner holds a newer policy pi.

- log_rhos[t] is log pi(a_t | x_t) - log mu(a_t | x_t).
- terminated[t] is True when step t ended the episode. The environment then
  resets, so values[t + 1] belongs to the next episode and must not leak back.
- bootstrap_value is V(x_T), the value of the observation after the last step.
"""

import numpy as np

import rlenvs


def clipped_ratios(log_rhos, rho_bar, c_bar):
    """Truncated importance weights.

    Args:
        log_rhos: log pi(a_t | x_t) - log mu(a_t | x_t), shape (T,).
        rho_bar: clip for rho_t, which scales the TD error and sets the fixed point.
        c_bar: clip for c_t, which scales how far a correction travels back.

    Returns:
        (rho, c), each of shape (T,): rho_t = min(rho_bar, exp(log_rho_t)) and
        c_t = min(c_bar, exp(log_rho_t)).
    """
    ratios = np.exp(np.asarray(log_rhos, dtype=float))
    return np.minimum(rho_bar, ratios), np.minimum(c_bar, ratios)


def vtrace_targets(rewards, values, bootstrap_value, terminated, log_rhos,
                   gamma, rho_bar=1.0, c_bar=1.0):
    """V-trace targets v_t by a backward recursion.

    With d_t = gamma * (1 - terminated_t), delta_t = rho_t * (r_t + d_t * V(x_{t+1}) - V(x_t)),
    the targets satisfy v_t - V(x_t) = delta_t + d_t * c_t * (v_{t+1} - V(x_{t+1})),
    and v_T = V(x_T) so that the recursion starts with no correction.

    Args:
        rewards: r_t, shape (T,).
        values: V(x_t), shape (T,).
        bootstrap_value: V(x_T), a float.
        terminated: booleans, shape (T,).
        log_rhos: log pi - log mu at the taken actions, shape (T,).
        gamma: the discount factor.
        rho_bar, c_bar: the two clips.

    Returns:
        v, shape (T,).
    """
    rewards = np.asarray(rewards, dtype=float)
    values = np.asarray(values, dtype=float)
    T = len(rewards)
    rho, c = clipped_ratios(log_rhos, rho_bar, c_bar)
    disc = gamma * (1.0 - np.asarray(terminated, dtype=float))
    next_values = np.append(values[1:], bootstrap_value)
    delta = rho * (rewards + disc * next_values - values)
    acc = 0.0
    out = np.zeros(T)
    for t in range(T - 1, -1, -1):
        acc = delta[t] + disc[t] * c[t] * acc
        out[t] = values[t] + acc
    return out


def vtrace_advantages(rewards, values, vs, bootstrap_value, terminated, log_rhos,
                      gamma, rho_bar=1.0):
    """Policy-gradient advantages, rho_t * (r_t + d_t * v_{t+1} - V(x_t)).

    The next target v_{t+1} replaces V(x_{t+1}), and v_T is bootstrap_value.
    Treat the result as a constant when you differentiate the loss.

    Args:
        rewards, values, terminated, log_rhos, gamma, rho_bar: as above.
        vs: the targets from vtrace_targets, shape (T,).
        bootstrap_value: V(x_T), a float.

    Returns:
        Advantages, shape (T,).
    """
    rewards = np.asarray(rewards, dtype=float)
    values = np.asarray(values, dtype=float)
    rho, _ = clipped_ratios(log_rhos, rho_bar, np.inf)
    disc = gamma * (1.0 - np.asarray(terminated, dtype=float))
    next_vs = np.append(np.asarray(vs, dtype=float)[1:], bootstrap_value)
    return rho * (rewards + disc * next_vs - values)


def interpolated_policy(pi, mu, rho_bar):
    """The policy whose value the V-trace operator converges to.

    pi_rho_bar(a | s) is proportional to min(rho_bar * mu(a | s), pi(a | s)).

    Args:
        pi, mu: policies, shape (S, A), rows summing to one.
        rho_bar: the rho clip.

    Returns:
        A policy of shape (S, A) whose rows sum to one.
    """
    w = np.minimum(rho_bar * mu, pi)
    return w / w.sum(axis=1, keepdims=True)


def vtrace_operator(mdp, pi, mu, V, rho_bar=1.0, c_bar=1.0):
    """Apply the V-trace operator to V, exactly, in expectation over mu.

    With rho(s, a) = min(rho_bar, pi / mu) and c(s, a) = min(c_bar, pi / mu):

        g(s)     = sum_a mu(a|s) rho(s,a) (R(s,a) + gamma sum_s' P(s'|s,a) V(s') - V(s))
        M(s, s') = sum_a mu(a|s) c(s,a) P(s'|s,a)
        (R V)    = V + (I - gamma M)^{-1} g

    The MDP has no terminal states.

    Args:
        mdp: a TabularMDP with P (S, A, S), R (S, A), gamma.
        pi, mu: policies, shape (S, A). mu is positive wherever pi is.
        V: current values, shape (S,).
        rho_bar, c_bar: the clips.

    Returns:
        The updated values, shape (S,).
    """
    ratio = pi / mu
    rho = np.minimum(rho_bar, ratio)
    c = np.minimum(c_bar, ratio)
    nxt = np.einsum("sat,t->sa", mdp.P, V)
    g = (mu * rho * (mdp.R + mdp.gamma * nxt - V[:, None])).sum(axis=1)
    M = np.einsum("sa,sa,sat->st", mu, c, mdp.P)
    S = len(V)
    return V + np.linalg.solve(np.eye(S) - mdp.gamma * M, g)


if __name__ == "__main__":
    rewards = np.array([1.0, 0.0, 1.0])
    values = np.array([0.5, 0.4, 0.3])
    log_rhos = np.log(np.array([0.5, 1.5, 2.0]))
    no = np.zeros(3, dtype=bool)
    vs = vtrace_targets(rewards, values, 0.2, no, log_rhos, gamma=0.9)
    adv = vtrace_advantages(rewards, values, vs, 0.2, no, log_rhos, gamma=0.9)
    print("targets   :", np.round(vs, 4))
    print("advantages:", np.round(adv, 4))

    mdp = rlenvs.random_mdp(n_states=5, n_actions=3, gamma=0.9, seed=0)
    rng = np.random.default_rng(1)
    pi = rng.dirichlet(np.ones(3), size=5)
    mu = rng.dirichlet(np.ones(3), size=5)
    V = np.zeros(5)
    for _ in range(300):
        V = vtrace_operator(mdp, pi, mu, V, rho_bar=1.0, c_bar=1.0)
    target = rlenvs.policy_evaluation(mdp, interpolated_policy(pi, mu, 1.0))
    print("fixed point  :", np.round(V, 4))
    print("V of pi_rho  :", np.round(target, 4))
    print("V of pi      :", np.round(rlenvs.policy_evaluation(mdp, pi), 4))
