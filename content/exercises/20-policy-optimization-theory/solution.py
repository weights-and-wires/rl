"""Reference solution for the policy optimisation theory exercise."""

import numpy as np

import rlenvs


def advantage(mdp, pi):
    v = rlenvs.policy_evaluation(mdp, pi)
    q = rlenvs.q_from_v(mdp, v)
    return q - v[:, None]


def performance_difference(mdp, pi_old, pi_new):
    d_new = rlenvs.state_distribution(mdp, pi_new)
    gain_per_state = (pi_new * advantage(mdp, pi_old)).sum(axis=1)
    return float(d_new @ gain_per_state / (1.0 - mdp.gamma))


def surrogate_gain(mdp, pi_old, pi_new):
    d_old = rlenvs.state_distribution(mdp, pi_old)
    gain_per_state = (pi_new * advantage(mdp, pi_old)).sum(axis=1)
    return float(d_old @ gain_per_state / (1.0 - mdp.gamma))


def _softmax(theta):
    z = np.exp(theta - theta.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def bandit_gradient(r, theta):
    r = np.asarray(r, dtype=float)
    pi = _softmax(np.asarray(theta, dtype=float))
    return pi * (r - pi @ r)


def softmax_pg_bandit(r, theta, eta, steps):
    theta = np.array(theta, dtype=float)
    history = np.zeros((steps, theta.shape[0]))
    for t in range(steps):
        history[t] = _softmax(theta)
        theta = theta + eta * bandit_gradient(r, theta)
    return history


def npg_step(mdp, pi, eta):
    adv = advantage(mdp, pi)
    # Subtracting each row's largest advantage changes nothing after
    # normalising, and keeps exp from overflowing when eta is large.
    logits = eta * adv / (1.0 - mdp.gamma)
    weights = pi * np.exp(logits - logits.max(axis=1, keepdims=True))
    return weights / weights.sum(axis=1, keepdims=True)


def run_npg(mdp, eta, iters):
    pi = rlenvs.uniform_policy(mdp)
    values = np.zeros(iters + 1)
    values[0] = mdp.mu0 @ rlenvs.policy_evaluation(mdp, pi)
    for t in range(iters):
        pi = npg_step(mdp, pi, eta)
        values[t + 1] = mdp.mu0 @ rlenvs.policy_evaluation(mdp, pi)
    return values


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
