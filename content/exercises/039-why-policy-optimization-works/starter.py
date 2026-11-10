"""Why policy optimisation works: the theory, checked exactly.

Fill in the functions below. A policy for a tabular MDP is an array `pi` of
shape (S, A) whose rows sum to one, and J(pi) = sum_s mu0(s) V^pi(s) is the
expected discounted return from the start distribution. You verify the
performance difference lemma, measure the first-order error of the surrogate
that trust-region methods optimise, watch softmax policy gradient crawl at a
1/t rate on a bandit, and build the natural policy gradient step for a
softmax policy.

`rlenvs` gives you exact tools: `policy_evaluation`, `q_from_v`,
`state_distribution` (the normalised discounted visitation d^pi, summing to
one), `value_iteration`, `greedy`, and `uniform_policy`.
"""

import numpy as np

import rlenvs


def advantage(mdp, pi):
    """The advantage A^pi(s, a) = Q^pi(s, a) - V^pi(s), computed exactly.

    Args:
        mdp: a `rlenvs.TabularMDP`.
        pi: the policy, shape (S, A).

    Returns:
        A, shape (S, A).
    """
    raise NotImplementedError("advantage")


def performance_difference(mdp, pi_old, pi_new):
    """The right-hand side of the performance difference lemma.

    (1 / (1 - gamma)) * sum_s d^{pi_new}(s) * sum_a pi_new(a | s) * A^{pi_old}(s, a),
    with d the normalised discounted state distribution of the NEW policy.

    Args:
        mdp: a `rlenvs.TabularMDP` with gamma < 1.
        pi_old: the policy whose advantage scores the actions, shape (S, A).
        pi_new: the policy whose states and actions are scored, shape (S, A).

    Returns:
        A float, which equals J(pi_new) - J(pi_old).
    """
    raise NotImplementedError("performance_difference")


def surrogate_gain(mdp, pi_old, pi_new):
    """The surrogate for J(pi_new) - J(pi_old) that trust-region methods use.

    The same sum as `performance_difference`, but weighted by d^{pi_old}, the
    state distribution of the OLD policy, which you can sample from.

    Args:
        mdp: a `rlenvs.TabularMDP` with gamma < 1.
        pi_old: shape (S, A).
        pi_new: shape (S, A).

    Returns:
        A float.
    """
    raise NotImplementedError("surrogate_gain")


def bandit_gradient(r, theta):
    """The gradient of the expected reward pi_theta^T r for a softmax bandit policy.

    d(pi^T r) / d theta(a) = pi(a) * (r(a) - pi^T r).

    Args:
        r: the mean reward of each arm, shape (K,).
        theta: the logits, shape (K,). pi_theta is their softmax.

    Returns:
        The gradient, shape (K,).
    """
    raise NotImplementedError("bandit_gradient")


def softmax_pg_bandit(r, theta, eta, steps):
    """Run exact softmax policy gradient ascent on a bandit.

    theta_{t+1} = theta_t + eta * bandit_gradient(r, theta_t).

    Args:
        r: the mean reward of each arm, shape (K,).
        theta: the starting logits theta_1, shape (K,). Don't modify it.
        eta: the step size.
        steps: the number of policies to record, T.

    Returns:
        An array of shape (T, K). Row t - 1 is pi_{theta_t}, so row 0 is the
        softmax of the starting logits and row 1 is the policy after one update.
    """
    raise NotImplementedError("softmax_pg_bandit")


def npg_step(mdp, pi, eta):
    """One natural policy gradient step for a tabular softmax policy.

    pi_new(a | s) is proportional to pi(a | s) * exp(eta * A^pi(s, a) / (1 - gamma)).

    Args:
        mdp: a `rlenvs.TabularMDP` with gamma < 1.
        pi: the current policy, shape (S, A).
        eta: the step size. It can be as large as 10^4, so subtract each
            row's largest exponent before calling `np.exp`.

    Returns:
        The new policy, shape (S, A), rows summing to one.
    """
    raise NotImplementedError("npg_step")


def run_npg(mdp, eta, iters):
    """Run `iters` NPG steps from the uniform policy and record J after each.

    Args:
        mdp: a `rlenvs.TabularMDP` with gamma < 1.
        eta: the step size.
        iters: the number of steps, T.

    Returns:
        An array of shape (T + 1,). Entry 0 is J of the uniform policy and
        entry t is J after t steps.
    """
    raise NotImplementedError("run_npg")


if __name__ == "__main__":
    mdp = rlenvs.random_mdp(n_states=4, n_actions=3, gamma=0.9, seed=0)
    rng = np.random.default_rng(1)
    pi_old = rng.dirichlet(np.ones(3), size=4)
    pi_new = rng.dirichlet(np.ones(3), size=4)
    j_old = mdp.mu0 @ rlenvs.policy_evaluation(mdp, pi_old)
    j_new = mdp.mu0 @ rlenvs.policy_evaluation(mdp, pi_new)
    print("Performance difference lemma on a random 4-state MDP")
    print(f"  J(new) - J(old)        = {j_new - j_old:+.12f}")
    print(f"  lemma, d of new policy = {performance_difference(mdp, pi_old, pi_new):+.12f}")
    print(f"  surrogate, d of old    = {surrogate_gain(mdp, pi_old, pi_new):+.12f}")

    r = np.array([1.0, 0.9, 0.1])
    for name, theta0 in [("uniform start", np.zeros(3)), ("bad start", np.array([0.0, 5.0, 0.0]))]:
        pis = softmax_pg_bandit(r, theta0, eta=0.4, steps=10_000)
        gap = r.max() - pis @ r
        print(f"\nSoftmax policy gradient, {name}: gap (pi* - pi_t)^T r")
        for t in [1, 10, 100, 1000, 10_000]:
            print(f"  t = {t:>6}: gap = {gap[t - 1]:.6f}, t * gap = {t * gap[t - 1]:.3f}")

    print("\nNatural policy gradient, eta = 1: J at each iterate")
    values = run_npg(mdp, eta=1.0, iters=8)
    j_star = mdp.mu0 @ rlenvs.value_iteration(mdp)
    for t, v in enumerate(values):
        print(f"  t = {t}: J = {v:.6f}, gap to optimal = {j_star - v:.2e}")
