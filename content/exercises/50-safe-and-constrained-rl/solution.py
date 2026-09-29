"""Reference solution for the constrained reinforcement learning exercise."""

import numpy as np

import rlenvs


def make_cmdp(seed=0, n_states=4, n_actions=3, gamma=0.9):
    """A random MDP with a random cost and a budget that the reward-optimal policy breaks.

    The budget sits halfway between the cost of the reward-optimal policy and
    the lowest cost any policy can reach, so the constraint always binds.

    Returns:
        mdp: an `rlenvs.TabularMDP` with no terminal states.
        cost: shape (S, A), the cost of taking a in s, in [0, 1].
        budget: a float, the largest allowed discounted cost from mu0.
    """
    mdp = rlenvs.random_mdp(n_states, n_actions, gamma, seed)
    cost = np.random.default_rng(100 + seed).uniform(0.0, 1.0, size=(n_states, n_actions))
    cost_mdp = rlenvs.TabularMDP(P=mdp.P, R=cost, gamma=gamma, mu0=mdp.mu0)
    best = rlenvs.greedy(rlenvs.q_from_v(mdp, rlenvs.value_iteration(mdp)))
    cost_of_best = rlenvs.policy_evaluation(cost_mdp, best) @ mdp.mu0
    neg = rlenvs.TabularMDP(P=mdp.P, R=-cost, gamma=gamma, mu0=mdp.mu0)
    lowest_cost = -(rlenvs.value_iteration(neg) @ mdp.mu0)
    return mdp, cost, float(0.5 * (cost_of_best + lowest_cost))


def softmax_policy(theta):
    """Row-wise softmax of logits, shape (S, A)."""
    z = theta - theta.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def occupancy(mdp, pi):
    """rho[s, a] = sum_t gamma^t Pr(s_t = s, a_t = a) from mu0, shape (S, A)."""
    P_pi = np.einsum("sa,sat->st", pi, mdp.P)
    state = np.linalg.solve(np.eye(mdp.n_states) - mdp.gamma * P_pi.T, mdp.mu0)
    return state[:, None] * pi


def cmdp_lp(mdp, cost, budget):
    """The occupancy-measure linear program in `scipy.optimize.linprog` form."""
    S, A = mdp.n_states, mdp.n_actions
    c = -mdp.R.reshape(-1)
    A_ub = cost.reshape(1, -1).astype(float)
    b_ub = np.array([float(budget)])
    # Flow: sum_a rho(s2, a) - gamma * sum_{s, a} P(s2 | s, a) rho(s, a) = mu0(s2).
    A_eq = np.zeros((S, S * A))
    for s in range(S):
        for a in range(A):
            A_eq[:, s * A + a] -= mdp.gamma * mdp.P[s, a]
            A_eq[s, s * A + a] += 1.0
    return c, A_ub, b_ub, A_eq, mdp.mu0.astype(float)


def policy_from_occupancy(rho):
    """pi(a | s) = rho(s, a) / sum_a rho(s, a), uniform where a state has no mass."""
    rho = np.asarray(rho, dtype=float)
    mass = rho.sum(axis=1, keepdims=True)
    uniform = np.full_like(rho, 1.0 / rho.shape[1])
    return np.where(mass > 0, rho / np.where(mass > 0, mass, 1.0), uniform)


def dual_function(mdp, cost, budget, lam):
    """g(lam) = max_pi [J_r(pi) - lam * (J_c(pi) - budget)], from mu0."""
    shifted = rlenvs.TabularMDP(P=mdp.P, R=mdp.R - lam * cost, gamma=mdp.gamma, mu0=mdp.mu0)
    return float(rlenvs.value_iteration(shifted) @ mdp.mu0 + lam * budget)


def lagrangian_grad(mdp, cost, budget, theta, lam):
    """Exact gradient of the Lagrangian in theta, and the constraint violation."""
    pi = softmax_policy(theta)
    rho = occupancy(mdp, pi)
    shifted = rlenvs.TabularMDP(P=mdp.P, R=mdp.R - lam * cost, gamma=mdp.gamma, mu0=mdp.mu0)
    v = rlenvs.policy_evaluation(shifted, pi)
    advantage = rlenvs.q_from_v(shifted, v) - v[:, None]
    # For a tabular softmax, dJ/dtheta[s, a] = rho(s) * pi(a | s) * A(s, a) = rho(s, a) * A(s, a).
    grad = rho * advantage
    violation = float((rho * cost).sum() - budget)
    return grad, violation


def primal_dual(mdp, cost, budget, steps=4000, lr_theta=1.0, lr_lam=0.1):
    """Gradient ascent on theta, projected gradient descent on lambda."""
    theta = np.zeros((mdp.n_states, mdp.n_actions))
    lam = 0.0
    lams = np.zeros(steps)
    avg = np.zeros_like(theta)
    start = steps // 2
    for k in range(steps):
        grad, violation = lagrangian_grad(mdp, cost, budget, theta, lam)
        theta = theta + lr_theta * grad
        lam = max(0.0, lam + lr_lam * violation)
        lams[k] = lam
        if k >= start:
            avg += occupancy(mdp, softmax_policy(theta))
    return {"lam": lams, "avg_occupancy": avg / (steps - start), "theta": theta}


def _log_sigmoid(z):
    """log(sigmoid(z)), stable for large |z|."""
    return -np.logaddexp(0.0, -z)


def cost_model_loss(cost_w, cost_l, sign_w, sign_l):
    """The Safe RLHF cost-model loss, averaged over pairs."""
    cost_w = np.asarray(cost_w, dtype=float)
    cost_l = np.asarray(cost_l, dtype=float)
    ranking = -_log_sigmoid(cost_w - cost_l)
    labels = -_log_sigmoid(sign_w * cost_w) - _log_sigmoid(sign_l * cost_l)
    return float(np.mean(ranking + labels))


def cost_model_grad(cost_w, cost_l, sign_w, sign_l):
    """Gradient of cost_model_loss with respect to cost_w and cost_l."""
    cost_w = np.asarray(cost_w, dtype=float)
    cost_l = np.asarray(cost_l, dtype=float)
    n = cost_w.shape[0]

    def sigmoid(z):
        return np.exp(_log_sigmoid(z))

    # d/dz [-log sigmoid(z)] = -sigmoid(-z).
    rank = -sigmoid(-(cost_w - cost_l))
    grad_w = rank - sign_w * sigmoid(-sign_w * cost_w)
    grad_l = -rank - sign_l * sigmoid(-sign_l * cost_l)
    return grad_w / n, grad_l / n


if __name__ == "__main__":
    mdp, cost, budget = make_cmdp(seed=0)
    unconstrained = rlenvs.greedy(rlenvs.q_from_v(mdp, rlenvs.value_iteration(mdp)))
    rho = occupancy(mdp, unconstrained)
    print(f"budget {budget:.3f}; the reward-optimal policy earns "
          f"{(rho * mdp.R).sum():.3f} at cost {(rho * cost).sum():.3f}")

    print("dual function g(lambda), an upper bound on the constrained optimum:")
    for lam in (0.0, 0.5, 1.0, 2.0):
        print(f"  g({lam:.1f}) = {dual_function(mdp, cost, budget, lam):.4f}")

    out = primal_dual(mdp, cost, budget)
    avg = out["avg_occupancy"]
    print(f"primal-dual: average lambda over the second half {out['lam'][2000:].mean():.3f}")
    print(f"  averaged policy earns {(avg * mdp.R).sum():.3f} at cost {(avg * cost).sum():.3f}")
    print("  policy from the averaged occupancy:")
    print(np.round(policy_from_occupancy(avg), 3))
