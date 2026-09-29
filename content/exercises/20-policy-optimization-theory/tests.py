"""Checks for the policy optimisation theory exercise."""

import numpy as np

import rlenvs
from rlcheck import almost, check, close, fail, finite_difference, has, shape


def _one_state_mdp():
    """The lesson's worked example: one state, two actions paying 1 and 0."""
    P = np.ones((1, 2, 1))
    R = np.array([[1.0, 0.0]])
    return rlenvs.TabularMDP(P=P, R=R, gamma=0.5, mu0=np.ones(1))


def _unit_reward_mdp(seed):
    """A random MDP whose rewards lie in [0, 1], as the NPG bound assumes."""
    base = rlenvs.random_mdp(n_states=5, n_actions=3, gamma=0.9, seed=seed)
    R = np.random.default_rng(seed).random((5, 3))
    return rlenvs.TabularMDP(P=base.P, R=R, gamma=0.9, mu0=base.mu0)


def _J(mdp, pi):
    return float(mdp.mu0 @ rlenvs.policy_evaluation(mdp, pi))


def _softmax(theta):
    z = np.exp(theta - theta.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


@check("advantage gives A = Q - V, which averages to zero under the policy")
def test_advantage(sub):
    advantage = has(sub, "advantage")
    got = advantage(_one_state_mdp(), np.array([[0.5, 0.5]]))
    shape(got, (1, 2), label="advantage on the one-state example")
    close(got, np.array([[0.5, -0.5]]), atol=1e-12, label="A for pi = (0.5, 0.5), r = (1, 0), gamma = 0.5")

    mdp = rlenvs.random_mdp(n_states=4, n_actions=3, gamma=0.9, seed=2)
    pi = np.random.default_rng(2).dirichlet(np.ones(3), size=4)
    A = advantage(mdp, pi)
    shape(A, (4, 3), label="advantage on a random MDP")
    close((pi * A).sum(axis=1), np.zeros(4), atol=1e-10,
          label="sum over a of pi(a | s) A(s, a), which must be zero in every state")
    P_pi = np.einsum("sa,sat->st", pi, mdp.P)
    v = np.linalg.solve(np.eye(4) - 0.9 * P_pi, (pi * mdp.R).sum(axis=1))
    q = mdp.R + 0.9 * mdp.P @ v
    close(A, q - v[:, None], atol=1e-10, label="A against Q - V from a direct linear solve")


@check("performance_difference equals J(new) - J(old) exactly, on 100 random pairs of policies")
def test_performance_difference(sub):
    performance_difference = has(sub, "performance_difference")
    got = performance_difference(_one_state_mdp(), np.array([[0.5, 0.5]]), np.array([[1.0, 0.0]]))
    almost(got, 1.0, tol=1e-12, label="the one-state example, which the lesson works to 2 - 1 = 1")

    mdp = rlenvs.random_mdp(n_states=4, n_actions=3, gamma=0.9, seed=0)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(100):
        old = rng.dirichlet(np.ones(3), size=4)
        new = rng.dirichlet(np.ones(3), size=4)
        worst = max(worst, abs(performance_difference(mdp, old, new) - (_J(mdp, new) - _J(mdp, old))))
    if worst > 1e-10:
        fail(
            "the lemma is an identity, so both sides must agree to rounding error. The largest "
            f"gap over 100 pairs was {worst:.3g}. Weight by the new policy's state distribution "
            "and divide by 1 - gamma."
        )


@check("surrogate_gain matches the true gain to first order: its error shrinks like alpha squared")
def test_surrogate_gain(sub):
    surrogate_gain = has(sub, "surrogate_gain")
    mdp = rlenvs.random_mdp(n_states=4, n_actions=3, gamma=0.9, seed=0)
    rng = np.random.default_rng(7)
    old = rng.dirichlet(np.ones(3), size=4)
    target = rng.dirichlet(np.ones(3), size=4)

    far = abs(_J(mdp, target) - _J(mdp, old) - surrogate_gain(mdp, old, target))
    if far < 1e-4:
        fail(
            "for a big policy change the surrogate should miss the true gain, because it uses the "
            f"old policy's state distribution. The miss was only {far:.3g}. Use d of pi_old, "
            "not pi_new."
        )
    errors = []
    for alpha in (0.02, 0.01):
        mixed = (1 - alpha) * old + alpha * target
        errors.append(_J(mdp, mixed) - _J(mdp, old) - surrogate_gain(mdp, old, mixed))
    ratio = errors[0] / errors[1]
    if not 3.6 < ratio < 4.4:
        fail(
            "halving the step alpha should cut the surrogate's error by about 4, because the "
            f"error is second order. Errors were {errors[0]:.3g} and {errors[1]:.3g}, ratio {ratio:.3g}."
        )


@check("bandit_gradient matches finite differences of the expected reward pi^T r")
def test_bandit_gradient(sub):
    bandit_gradient = has(sub, "bandit_gradient")
    r = np.array([1.0, 0.9, 0.1])
    got = bandit_gradient(r, np.zeros(3))
    shape(got, (3,), label="bandit_gradient output")
    close(got, (r - 2.0 / 3.0) / 3.0, atol=1e-12,
          label="the gradient at the uniform policy, (1/3) * (r - 2/3)")
    rng = np.random.default_rng(3)
    for _ in range(3):
        theta = rng.normal(size=5) * 2
        r = rng.random(5)
        fd = finite_difference(lambda th: _softmax(th) @ r, theta.copy())
        close(bandit_gradient(r, theta), fd, atol=1e-8, label="gradient against finite differences")


@check("softmax_pg_bandit stays under Mei et al.'s bound 5 / (t c_t^2) and decays like 1/t")
def test_softmax_pg_rate(sub):
    softmax_pg_bandit = has(sub, "softmax_pg_bandit")
    r = np.array([1.0, 0.9, 0.1])
    pis = softmax_pg_bandit(r, np.zeros(3), 0.4, 20_000)
    shape(pis, (20_000, 3), label="softmax_pg_bandit output")
    close(pis[0], np.ones(3) / 3, atol=1e-12, label="row 0, the starting policy pi_1")
    second = _softmax(0.4 * (np.ones(3) / 3) * (r - 2 / 3))
    close(pis[1], second, atol=1e-12, label="row 1, the policy after one update")

    t = np.arange(1, 20_001)
    gap = r.max() - pis @ r
    c = np.minimum.accumulate(pis[:, 0])
    bound = 5.0 / (t * c**2)
    bad = np.flatnonzero(gap > bound)
    if bad.size:
        k = bad[0]
        fail(f"at t = {k + 1} the gap {gap[k]:.4g} exceeds the bound {bound[k]:.4g}")
    window = slice(999, 10_000)
    slope = np.polyfit(np.log(t[window]), np.log(gap[window]), 1)[0]
    if not -1.1 < slope < -0.9:
        fail(f"the log-log slope of the gap over t in [1000, 10000] should be about -1, got {slope:.3f}")


@check("softmax_pg_bandit from a bad start takes at least 10 times longer to reach a gap of 0.01")
def test_softmax_pg_bad_start(sub):
    softmax_pg_bandit = has(sub, "softmax_pg_bandit")
    r = np.array([1.0, 0.9, 0.1])
    steps = 20_000
    good = r.max() - softmax_pg_bandit(r, np.zeros(3), 0.4, steps) @ r
    bad = r.max() - softmax_pg_bandit(r, np.array([0.0, 5.0, 0.0]), 0.4, steps) @ r
    if not (good < 0.01).any() or not (bad < 0.01).any():
        fail("both runs should reach a gap below 0.01 within 20,000 steps")
    t_good = int(np.argmax(good < 0.01)) + 1
    t_bad = int(np.argmax(bad < 0.01)) + 1
    if t_bad < 10 * t_good:
        fail(
            f"from uniform the gap fell below 0.01 at t = {t_good}, and from theta = (0, 5, 0) at "
            f"t = {t_bad}. The bad start should be at least 10 times slower."
        )


@check("npg_step equals theta + eta F^+ grad J with the Fisher matrix built explicitly")
def test_npg_matches_fisher(sub):
    npg_step = has(sub, "npg_step")
    mdp = rlenvs.random_mdp(n_states=4, n_actions=3, gamma=0.9, seed=1)
    S, A = 4, 3
    theta = np.random.default_rng(0).normal(size=(S, A))
    pi = _softmax(theta)
    d = rlenvs.state_distribution(mdp, pi)
    F = np.zeros((S * A, S * A))
    for s in range(S):
        for a in range(A):
            score = np.zeros((S, A))
            score[s] = -pi[s]
            score[s, a] += 1.0
            F += d[s] * pi[s, a] * np.outer(score.ravel(), score.ravel())
    grad = finite_difference(lambda th: _J(mdp, _softmax(th.reshape(S, A))), theta.ravel().copy())
    eta = 0.7
    want = _softmax(theta + eta * (np.linalg.pinv(F) @ grad).reshape(S, A))
    got = npg_step(mdp, pi, eta)
    shape(got, (S, A), label="npg_step output")
    close(got, want, atol=1e-7, label="npg_step against the explicit natural gradient step")

    v = rlenvs.policy_evaluation(mdp, pi)
    greedy = rlenvs.greedy(rlenvs.q_from_v(mdp, v))
    close(npg_step(mdp, pi, 1e4), greedy, atol=1e-8,
          label="npg_step with eta = 10^4, which should be policy iteration's greedy step")


@check("npg_step solves the per-state mirror descent problem: max eta <A, p> / (1 - gamma) - KL(p || pi)")
def test_npg_mirror_descent(sub):
    npg_step = has(sub, "npg_step")
    mdp = rlenvs.random_mdp(n_states=3, n_actions=4, gamma=0.8, seed=4)
    rng = np.random.default_rng(4)
    pi = rng.dirichlet(np.ones(4), size=3)
    v = rlenvs.policy_evaluation(mdp, pi)
    adv = rlenvs.q_from_v(mdp, v) - v[:, None]
    eta = 0.5
    new = npg_step(mdp, pi, eta)

    def objective(p, s):
        return eta * (p @ adv[s]) / (1 - mdp.gamma) - np.sum(p * np.log(p / pi[s]))

    for s in range(3):
        best = objective(new[s], s)
        rivals = rng.dirichlet(np.ones(4), size=2000)
        rivals = np.vstack([rivals, 0.99 * new[s] + 0.01 * rivals[:20]])
        beaten = max(objective(p, s) for p in rivals)
        if beaten > best + 1e-12:
            fail(f"in state {s} a rival distribution scores {beaten:.6f}, above npg_step's {best:.6f}")


@check("run_npg improves every iteration and stays within Agarwal et al.'s bound log A / (eta T) + 1 / ((1 - gamma)^2 T)")
def test_run_npg(sub):
    run_npg = has(sub, "run_npg")
    mdp = _unit_reward_mdp(5)
    j_star = float(mdp.mu0 @ rlenvs.value_iteration(mdp))
    T = np.arange(1, 201)
    for eta in (0.1, 1.0, 10.0):
        values = run_npg(mdp, eta, 200)
        shape(values, (201,), label=f"run_npg output at eta = {eta}")
        almost(values[0], _J(mdp, rlenvs.uniform_policy(mdp)), tol=1e-10,
               label="values[0], the uniform policy's J")
        if np.any(np.diff(values) < -1e-10):
            k = int(np.argmax(np.diff(values) < -1e-10))
            fail(f"at eta = {eta}, J fell from {values[k]:.8f} to {values[k + 1]:.8f} at iteration {k + 1}")
        gap = j_star - values[1:]
        bound = np.log(3) / (eta * T) + 1.0 / ((1 - mdp.gamma) ** 2 * T)
        if np.any(gap > bound):
            k = int(np.argmax(gap > bound))
            fail(f"at eta = {eta} and T = {k + 1}, the gap {gap[k]:.4g} exceeds the bound {bound[k]:.4g}")
        if gap[-1] > 1e-2:
            fail(f"at eta = {eta}, NPG should be within 0.01 of optimal after 200 steps, got {gap[-1]:.4g}")
