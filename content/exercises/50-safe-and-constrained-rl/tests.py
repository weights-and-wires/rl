"""Checks for the constrained reinforcement learning exercise."""

import itertools

import numpy as np
from scipy.optimize import linprog

import rlenvs
from rlcheck import almost, check, close, fail, finite_difference, has, shape


def _bandit():
    """The lesson's one-state example: arm 0 pays 1 and costs 1, arm 1 pays and costs 0."""
    P = np.ones((1, 2, 1))
    R = np.array([[1.0, 0.0]])
    cost = np.array([[1.0, 0.0]])
    return rlenvs.TabularMDP(P=P, R=R, gamma=0.0, mu0=np.array([1.0])), cost, 0.3


def _solve(sub, mdp, cost, budget):
    cmdp_lp = has(sub, "cmdp_lp")
    c, A_ub, b_ub, A_eq, b_eq = cmdp_lp(mdp, cost, budget)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None),
                  method="highs")
    if res.status != 0:
        fail(f"linprog could not solve the program your cmdp_lp built: {res.message}")
    rho = res.x.reshape(mdp.n_states, mdp.n_actions)
    return rho, -res.fun, -res.ineqlin.marginals[0]


@check("occupancy sums to 1 / (1 - gamma), and sum rho * R equals the policy's value")
def test_occupancy(sub):
    occupancy = has(sub, "occupancy")
    mdp = rlenvs.random_mdp(4, 3, 0.9, seed=3)
    pi = np.random.default_rng(0).dirichlet(np.ones(3), size=4)
    rho = occupancy(mdp, pi)
    shape(rho, (4, 3), label="occupancy")
    almost(rho.sum(), 10.0, 1e-9, label="total occupancy at gamma = 0.9")
    value = rlenvs.policy_evaluation(mdp, pi) @ mdp.mu0
    almost((rho * mdp.R).sum(), value, 1e-9, label="sum of rho * R against policy_evaluation")
    close(rho.sum(axis=1), rlenvs.state_distribution(mdp, pi) / 0.1, atol=1e-9,
          label="state marginal against rlenvs.state_distribution / (1 - gamma)")


@check("policy_from_occupancy normalises each row and falls back to uniform on empty rows")
def test_policy_from_occupancy(sub):
    policy_from_occupancy = has(sub, "policy_from_occupancy")
    rho = np.array([[0.3, 0.7, 0.0], [0.0, 0.0, 0.0], [2.0, 1.0, 1.0]])
    want = np.array([[0.3, 0.7, 0.0], [1 / 3, 1 / 3, 1 / 3], [0.5, 0.25, 0.25]])
    close(policy_from_occupancy(rho), want, atol=1e-12, label="policy_from_occupancy")


@check("On the one-state example the LP plays the costly arm with probability 0.3, with multiplier 1")
def test_lp_bandit(sub):
    policy_from_occupancy = has(sub, "policy_from_occupancy")
    mdp, cost, budget = _bandit()
    rho, value, lam = _solve(sub, mdp, cost, budget)
    close(policy_from_occupancy(rho), np.array([[0.3, 0.7]]), atol=1e-8, label="LP policy")
    almost(value, 0.3, 1e-8, label="LP value")
    almost(lam, 1.0, 1e-8, label="the LP's multiplier for the cost constraint")


@check("On a 4-state CMDP the LP optimum is feasible, stochastic, and beats every deterministic policy")
def test_lp_random(sub):
    occupancy = has(sub, "occupancy")
    policy_from_occupancy = has(sub, "policy_from_occupancy")
    make_cmdp = has(sub, "make_cmdp")
    mdp, cost, budget = make_cmdp(seed=0)
    rho, value, _ = _solve(sub, mdp, cost, budget)
    pi = policy_from_occupancy(rho)
    close(occupancy(mdp, pi), rho, atol=1e-7,
          label="the occupancy of the LP's policy against the LP's own variables")
    if (rho * cost).sum() > budget + 1e-7:
        fail(f"the LP solution costs {(rho * cost).sum():.6f}, over the budget {budget:.6f}")
    if not np.any(pi.max(axis=1) < 1 - 1e-6):
        fail(f"expected at least one state to randomise, but the LP policy is deterministic:\n{pi}")
    best = -np.inf
    for actions in itertools.product(range(3), repeat=4):
        det = np.eye(3)[list(actions)]
        occ = occupancy(mdp, det)
        if (occ * cost).sum() <= budget:
            best = max(best, (occ * mdp.R).sum())
    if value < best + 0.5:
        fail(f"the LP value {value:.4f} should beat the best feasible deterministic policy, "
             f"{best:.4f}, by at least 0.5")


@check("dual_function matches hand values and touches the LP value exactly at the LP's multiplier")
def test_dual(sub):
    dual_function = has(sub, "dual_function")
    make_cmdp = has(sub, "make_cmdp")
    mdp, cost, budget = _bandit()
    got = [dual_function(mdp, cost, budget, lam) for lam in (0.0, 0.5, 1.0, 2.0)]
    close(np.array(got), np.array([1.0, 0.65, 0.3, 0.6]), atol=1e-9,
          label="g(lambda) at 0, 0.5, 1, 2 on the one-state example")

    mdp, cost, budget = make_cmdp(seed=0)
    _, value, lam_star = _solve(sub, mdp, cost, budget)
    for lam in np.linspace(0.0, 3.0, 13):
        g = dual_function(mdp, cost, budget, lam)
        if g < value - 1e-8:
            fail(f"weak duality fails: g({lam:.2f}) = {g:.6f} is below the LP value {value:.6f}")
    almost(dual_function(mdp, cost, budget, lam_star), value, 1e-7,
           label="g at the LP multiplier against the LP value (zero duality gap)")


@check("lagrangian_grad matches finite differences of J_r - lambda * (J_c - budget)")
def test_lagrangian_grad(sub):
    lagrangian_grad = has(sub, "lagrangian_grad")
    softmax_policy = has(sub, "softmax_policy")
    make_cmdp = has(sub, "make_cmdp")
    mdp, cost, budget = make_cmdp(seed=1)
    lam = 0.7
    theta0 = np.random.default_rng(2).normal(size=(4, 3))
    cost_mdp = rlenvs.TabularMDP(P=mdp.P, R=cost, gamma=mdp.gamma, mu0=mdp.mu0)

    def lagrangian(theta):
        pi = softmax_policy(theta)
        j_r = rlenvs.policy_evaluation(mdp, pi) @ mdp.mu0
        j_c = rlenvs.policy_evaluation(cost_mdp, pi) @ mdp.mu0
        return j_r - lam * (j_c - budget)

    grad, violation = lagrangian_grad(mdp, cost, budget, theta0, lam)
    shape(grad, (4, 3), label="gradient")
    close(grad, finite_difference(lagrangian, theta0.copy()), rtol=1e-5, atol=1e-7,
          label="lagrangian_grad against finite differences")
    j_c = rlenvs.policy_evaluation(cost_mdp, softmax_policy(theta0)) @ mdp.mu0
    almost(violation, j_c - budget, 1e-9, label="the constraint violation J_c - budget")


@check("primal_dual: the averaged occupancy and lambda land near the LP solution")
def test_primal_dual(sub):
    primal_dual = has(sub, "primal_dual")
    make_cmdp = has(sub, "make_cmdp")
    mdp, cost, budget = make_cmdp(seed=0)
    _, value, lam_star = _solve(sub, mdp, cost, budget)
    out = primal_dual(mdp, cost, budget, steps=4000, lr_theta=1.0, lr_lam=0.1)
    lams = np.asarray(out["lam"])
    shape(lams, (4000,), label="lambda history")
    if lams.min() < 0:
        fail(f"lambda must never go negative; its smallest value was {lams.min():.4f}")
    avg = out["avg_occupancy"]
    lam_avg = lams[2000:].mean()
    if abs(lam_avg / lam_star - 1) > 0.25:
        fail(f"the average lambda over the second half, {lam_avg:.4f}, should be within 25% "
             f"of the LP multiplier {lam_star:.4f}")
    earned, spent = (avg * mdp.R).sum(), (avg * cost).sum()
    if abs(earned / value - 1) > 0.12:
        fail(f"the averaged policy earns {earned:.4f}; expected within 12% of the LP value {value:.4f}")
    if abs(spent / budget - 1) > 0.10:
        fail(f"the averaged policy costs {spent:.4f}; expected within 10% of the budget {budget:.4f}")


@check("cost_model_loss matches the hand value -log s(2) - 2 log s(1) = 0.75345 and is stable")
def test_cost_model_loss(sub):
    cost_model_loss = has(sub, "cost_model_loss")
    got = cost_model_loss(np.array([1.0]), np.array([-1.0]), np.array([1.0]), np.array([-1.0]))
    almost(got, 0.7534513860794183, 1e-10, label="one pair, costs 1 and -1")
    tied = cost_model_loss(np.zeros(3), np.zeros(3), np.ones(3), -np.ones(3))
    almost(tied, 3 * np.log(2.0), 1e-10, label="all costs zero")
    big = cost_model_loss(np.array([-800.0]), np.array([800.0]), np.array([1.0]), np.array([-1.0]))
    almost(big, 3200.0, 1e-6, label="a confidently wrong pair, costs -800 and 800")


@check("cost_model_grad matches finite differences")
def test_cost_model_grad(sub):
    cost_model_loss = has(sub, "cost_model_loss")
    cost_model_grad = has(sub, "cost_model_grad")
    rng = np.random.default_rng(4)
    n = 5
    cw, cl = rng.normal(size=n), rng.normal(size=n)
    sw = np.array([1.0, 1.0, -1.0, 1.0, -1.0])
    sl = np.array([-1.0, 1.0, -1.0, -1.0, -1.0])
    x0 = np.concatenate([cw, cl])
    fd = finite_difference(lambda x: cost_model_loss(x[:n], x[n:], sw, sl), x0)
    gw, gl = cost_model_grad(cw, cl, sw, sl)
    shape(gw, (n,), label="grad for cost_w")
    shape(gl, (n,), label="grad for cost_l")
    close(np.concatenate([gw, gl]), fd, rtol=1e-5, atol=1e-8,
          label="cost_model_grad against finite differences")
