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
    raise NotImplementedError("clipped_ratios")


def vtrace_targets(rewards, values, bootstrap_value, terminated, log_rhos,
                   gamma, rho_bar=1.0, c_bar=1.0):
    """V-trace targets v_t, computed from the end of the rollout back to the start.

    An episode end must stop the correction from reaching earlier steps, and
    the targets must not depend on the next episode's values.

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
    raise NotImplementedError("vtrace_targets")


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
    raise NotImplementedError("vtrace_advantages")


def interpolated_policy(pi, mu, rho_bar):
    """The policy whose value the V-trace operator converges to.

    pi_rho_bar(a | s) is proportional to min(rho_bar * mu(a | s), pi(a | s)).

    Args:
        pi, mu: policies, shape (S, A), rows summing to one.
        rho_bar: the rho clip.

    Returns:
        A policy of shape (S, A) whose rows sum to one.
    """
    raise NotImplementedError("interpolated_policy")


def vtrace_operator(mdp, pi, mu, V, rho_bar=1.0, c_bar=1.0):
    """Apply the V-trace operator to V once, exactly, in expectation over mu.

    Use the model: mdp.P (S, A, S), mdp.R (S, A), mdp.gamma. Each step's
    expected correction is weighted by mu, rho, and c, and corrections
    accumulate along the trajectory. Solve a linear system instead of
    sampling. The MDP has no terminal states.

    Args:
        mdp: a TabularMDP with P (S, A, S), R (S, A), gamma.
        pi, mu: policies, shape (S, A). mu is positive wherever pi is.
        V: current values, shape (S,).
        rho_bar, c_bar: the clips.

    Returns:
        The updated values, shape (S,).
    """
    raise NotImplementedError("vtrace_operator")


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
