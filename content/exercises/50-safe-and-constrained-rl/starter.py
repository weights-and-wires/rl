"""Constrained MDPs: the occupancy LP, the Lagrangian, and a Safe RLHF cost model.

A constrained MDP adds a cost table `cost` of shape (S, A) and a budget to an
ordinary MDP. The goal is the policy with the highest discounted reward from
mu0 whose discounted cost from mu0 stays within the budget. You write it as a
linear program over occupancy measures, check strong duality with the dual
function, run the Lagrangian primal-dual method, and build the cost-model
loss from Safe RLHF.

Every MDP here has no terminal states and gamma < 1. A policy is an (S, A)
array whose rows sum to one.
"""

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

    This helper is given. You don't need to change it.
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
    """Row-wise softmax of logits, shape (S, A). This helper is given."""
    z = theta - theta.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def occupancy(mdp, pi):
    """The unnormalised discounted occupancy of a policy.

    rho[s, a] = sum_t gamma^t Pr(s_t = s, a_t = a), starting from mdp.mu0.
    Its entries sum to 1 / (1 - gamma), and sum(rho * mdp.R) is the policy's
    expected discounted return.

    Args:
        mdp: an `rlenvs.TabularMDP` with no terminal states and gamma < 1.
        pi: a policy, shape (S, A).

    Returns:
        rho, shape (S, A).
    """
    raise NotImplementedError("occupancy")


def cmdp_lp(mdp, cost, budget):
    """The constrained MDP as a linear program over occupancy measures.

    The variable x is rho flattened in row-major order, so x[s * A + a] is
    rho[s, a]. The program is: minimise c @ x subject to A_ub @ x <= b_ub,
    A_eq @ x == b_eq, and x >= 0. Maximising reward means c = -R flattened.
    The one inequality row caps the discounted cost at the budget. The S
    equality rows are the flow constraints: for every state s2,
    sum_a rho[s2, a] - gamma * sum_{s, a} P[s, a, s2] * rho[s, a] = mu0[s2].

    Args:
        mdp: an `rlenvs.TabularMDP`.
        cost: shape (S, A).
        budget: a float.

    Returns:
        A tuple (c, A_ub, b_ub, A_eq, b_eq) with shapes (S*A,), (1, S*A),
        (1,), (S, S*A), and (S,), ready for scipy.optimize.linprog.
    """
    raise NotImplementedError("cmdp_lp")


def policy_from_occupancy(rho):
    """Recover the policy an occupancy measure came from.

    pi[s, a] = rho[s, a] / sum_a rho[s, a]. A state with no occupancy gets the
    uniform distribution.

    Args:
        rho: shape (S, A), non-negative.

    Returns:
        pi, shape (S, A), rows summing to one.
    """
    raise NotImplementedError("policy_from_occupancy")


def dual_function(mdp, cost, budget, lam):
    """The Lagrangian dual function of the constrained MDP.

    g(lam) = max over policies of J_r(pi) - lam * (J_c(pi) - budget), where
    J_r and J_c are the discounted reward and cost from mdp.mu0. For fixed
    lam, the maximum is an ordinary MDP with reward R - lam * cost.

    Args:
        mdp: an `rlenvs.TabularMDP`.
        cost: shape (S, A).
        budget: a float.
        lam: a non-negative float.

    Returns:
        g(lam), a float.
    """
    raise NotImplementedError("dual_function")


def lagrangian_grad(mdp, cost, budget, theta, lam):
    """The exact gradient of the Lagrangian for a tabular softmax policy.

    The Lagrangian is L(theta, lam) = J_r(pi) - lam * (J_c(pi) - budget) with
    pi = softmax_policy(theta).

    Args:
        mdp: an `rlenvs.TabularMDP`.
        cost: shape (S, A).
        budget: a float.
        theta: logits, shape (S, A).
        lam: a non-negative float.

    Returns:
        A tuple (grad, violation). grad has shape (S, A) and is dL/dtheta.
        violation is the float J_c(pi) - budget.
    """
    raise NotImplementedError("lagrangian_grad")


def primal_dual(mdp, cost, budget, steps=4000, lr_theta=1.0, lr_lam=0.1):
    """Run gradient descent-ascent on the Lagrangian.

    Start from theta = 0 and lam = 0. Each step, compute lagrangian_grad at the
    current (theta, lam), then set theta += lr_theta * grad and
    lam = max(0, lam + lr_lam * violation).

    Args:
        mdp, cost, budget: the constrained MDP.
        steps: the number of steps.
        lr_theta, lr_lam: the two step sizes.

    Returns:
        A dict with keys:
            "lam": shape (steps,), lambda after each step.
            "avg_occupancy": shape (S, A), the mean of occupancy(mdp, pi) over
                the policies after steps steps // 2 to steps - 1.
            "theta": shape (S, A), the final logits.
    """
    raise NotImplementedError("primal_dual")


def cost_model_loss(cost_w, cost_l, sign_w, sign_l):
    """The Safe RLHF cost-model loss, averaged over pairs.

    For each pair, y_w is the more harmful response and y_l the less harmful
    one. sign is +1 for a response labelled harmful and -1 for one labelled
    safe. The per-pair loss is
        -log sigmoid(cost_w - cost_l)
        - log sigmoid(sign_w * cost_w) - log sigmoid(sign_l * cost_l).
    Compute log sigmoid stably: it must not overflow for |z| in the hundreds.

    Args:
        cost_w, cost_l: the cost model's scores, shape (n,).
        sign_w, sign_l: the harm labels, shape (n,), each entry +1 or -1.

    Returns:
        The mean loss, a float.
    """
    raise NotImplementedError("cost_model_loss")


def cost_model_grad(cost_w, cost_l, sign_w, sign_l):
    """Gradient of cost_model_loss with respect to cost_w and cost_l.

    Args:
        cost_w, cost_l, sign_w, sign_l: as in cost_model_loss.

    Returns:
        A tuple (grad_w, grad_l), each shape (n,).
    """
    raise NotImplementedError("cost_model_grad")


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
