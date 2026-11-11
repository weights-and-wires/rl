"""Checks for the meta-RL exercise."""

import itertools

import numpy as np

from rlcheck import almost, check, close, fail, finite_difference, has, shape

FAMILY = np.array([[0.8, 0.2], [0.4, 0.6], [0.2, 0.8]])
PRIOR = np.array([0.5, 0.2, 0.3])
SYM = np.array([[0.8, 0.2], [0.2, 0.8]])
SYM_PRIOR = np.array([0.5, 0.5])

SIGMA = 0.5
TASKS = [
    (np.array([1.2, 0.0]), 0.5),
    (np.array([-0.6, 1.0]), 0.8),
    (np.array([-0.6, -1.0]), 1.2),
]
ALPHA = 4.0


@check("rl2_inputs packs one-hot obs, one-hot previous action, previous reward, and done")
def test_rl2_inputs(sub):
    rl2_inputs = has(sub, "rl2_inputs")
    x = rl2_inputs(2, 1, 0.5, True, 3, 2)
    shape(np.asarray(x), (7,), label="rl2_inputs(...) with n_obs=3, n_actions=2")
    close(x, [0, 0, 1, 0, 1, 0.5, 1], atol=1e-12, label="[obs one-hot | action one-hot | reward | done]")
    first = rl2_inputs(0, None, 0.0, False, 3, 2)
    close(first, [1, 0, 0, 0, 0, 0, 0], atol=1e-12, label="first step of a trial has no previous action")


@check("rl2_trial_inputs feeds step t the action, reward, and done of step t - 1, and starts each trial blank")
def test_rl2_trial_inputs(sub):
    rl2_trial_inputs = has(sub, "rl2_trial_inputs")
    obs, actions, rewards, dones = [0, 1, 2, 0], [1, 0, 1, 0], [1.0, 0.0, 2.0, 3.0], [False, True, False, True]
    got = np.asarray(rl2_trial_inputs(obs, actions, rewards, dones, 3, 2), dtype=float)
    # Row t: one-hot(obs[t]), one-hot(actions[t-1]), rewards[t-1], dones[t-1].
    want = np.array([
        [1, 0, 0, 0, 0, 0.0, 0],
        [0, 1, 0, 0, 1, 1.0, 0],
        [0, 0, 1, 1, 0, 0.0, 1],
        [1, 0, 0, 0, 1, 2.0, 0],
    ])
    close(got, want, atol=1e-12, label="trial inputs, rows t = 0..3")
    again = np.asarray(rl2_trial_inputs([2, 2], [0, 1], [5.0, 5.0], [True, True], 3, 2), dtype=float)
    close(again[0], [0, 0, 1, 0, 0, 0, 0], atol=1e-12, label="the next trial's first row carries nothing over")


@check("posterior_update and task_posterior apply Bayes' rule with the prior: (0.7407, 0.1481, 0.1111) after one win on arm 0")
def test_posterior(sub):
    posterior_update = has(sub, "posterior_update")
    task_posterior = has(sub, "task_posterior")
    # Named wrong implementation: a posterior that normalizes the likelihood
    # alone and ignores the prior. It returns (0.571, 0.286, 0.143) here.
    post = posterior_update(PRIOR, FAMILY, 0, 1)
    close(post, [0.4 / 0.54, 0.08 / 0.54, 0.06 / 0.54], atol=1e-12, label="posterior after (arm 0, reward 1)")
    got = task_posterior(PRIOR, FAMILY, [(0, 1), (0, 1), (0, 0)])
    close(got, np.array([0.064, 0.0192, 0.0096]) / 0.0928, atol=1e-12, label="posterior after (0,1), (0,1), (0,0)")
    close(task_posterior(PRIOR, FAMILY, []), PRIOR, atol=1e-12, label="empty history returns the prior")
    order = task_posterior(PRIOR, FAMILY, [(1, 0), (0, 1)])
    close(order, task_posterior(PRIOR, FAMILY, [(0, 1), (1, 0)]), atol=1e-12, label="the order of the history does not matter")


def _brute_force_h2(prior, thetas):
    """Best of all 32 history-dependent policies for H = 2, by enumeration."""
    histories = [(a, r) for a in (0, 1) for r in (0, 1)]
    best = -1.0
    for first, *second in itertools.product((0, 1), repeat=5):
        reply = dict(zip(histories, second))
        total = 0.0
        for k, w in enumerate(prior):
            for r1 in (0, 1):
                p1 = thetas[k, first] if r1 else 1 - thetas[k, first]
                a2 = reply[(first, r1)]
                total += w * p1 * (r1 + thetas[k, a2])
        best = max(best, total)
    return best


def _unnormalized(weights, thetas, H):
    """Bayes-optimal value with unnormalized weights, so no posterior is ever formed."""
    if H == 0:
        return 0.0
    out = -1.0
    for a in range(thetas.shape[1]):
        w1, w0 = weights * thetas[:, a], weights * (1 - thetas[:, a])
        out = max(out, w1.sum() + _unnormalized(w1, thetas, H - 1) + _unnormalized(w0, thetas, H - 1))
    return out


@check("bayes_optimal_value matches brute force over all policies at H=2 and an unnormalized recursion at H=3, 4")
def test_bayes_optimal(sub):
    bayes_optimal_value = has(sub, "bayes_optimal_value")
    for name, prior, thetas in (("family", PRIOR, FAMILY), ("symmetric", SYM_PRIOR, SYM)):
        almost(bayes_optimal_value(prior, thetas, 1), float((prior @ thetas).max()), 1e-12, label=f"{name}, H=1")
        almost(bayes_optimal_value(prior, thetas, 2), _brute_force_h2(prior, thetas), 1e-10, label=f"{name}, H=2 against 32 policies")
        for H in (3, 4):
            almost(bayes_optimal_value(prior, thetas, H), _unnormalized(prior.copy(), thetas, H), 1e-10, label=f"{name}, H={H}")
    almost(bayes_optimal_value(SYM_PRIOR, SYM, 3), 1.86, 1e-10, label="symmetric family, H=3")


def _thompson_monte_carlo(prior, thetas, H, n, seed):
    rng = np.random.default_rng(seed)
    best = thetas.argmax(axis=1)
    totals = np.zeros(n)
    for i in range(n):
        k = rng.choice(len(prior), p=prior)           # the true task
        post = prior.copy()
        for _ in range(H):
            guess = rng.choice(len(prior), p=post)     # one posterior draw
            arm = best[guess]
            r = float(rng.random() < thetas[k, arm])
            like = thetas[:, arm] if r else 1 - thetas[:, arm]
            post = post * like / (post * like).sum()
            totals[i] += r
    return totals.mean(), totals.std() / np.sqrt(n)


@check("thompson_value is the exact expected return of Thompson sampling, and agrees with simulation")
def test_thompson(sub):
    thompson_value = has(sub, "thompson_value")
    almost(thompson_value(PRIOR, FAMILY, 1), 0.5, 1e-12, label="H=1: arm 0 and arm 1 each have posterior mass 0.5 of being best")
    almost(thompson_value(SYM_PRIOR, SYM, 3), 1.7668235294117651, 1e-10, label="symmetric family, H=3")
    mean, se = _thompson_monte_carlo(PRIOR, FAMILY, 3, 20000, seed=0)
    exact = thompson_value(PRIOR, FAMILY, 3)
    if abs(mean - exact) > 4 * se:
        fail(f"simulating Thompson sampling 20000 times at H=3 gave {mean:.4f} +/- {se:.4f}; thompson_value returned {exact:.4f}")


@check("adaptive beats Thompson sampling beats a fixed arm, and none beats knowing the task")
def test_ordering(sub):
    bayes = has(sub, "bayes_optimal_value")
    thompson = has(sub, "thompson_value")
    fixed = has(sub, "fixed_policy_value")
    oracle = has(sub, "oracle_value")
    almost(fixed(SYM_PRIOR, SYM, 3), 1.5, 1e-12, label="fixed policy, symmetric, H=3")
    almost(oracle(SYM_PRIOR, SYM, 3), 2.4, 1e-12, label="oracle, symmetric, H=3")
    for name, prior, thetas in (("family", PRIOR, FAMILY), ("symmetric", SYM_PRIOR, SYM)):
        for H in (2, 3, 4):
            b, t, f, o = (g(prior, thetas, H) for g in (bayes, thompson, fixed, oracle))
            if not (t < b - 1e-6 and f < b - 1e-6 and b < o - 1e-6):
                fail(f"{name}, H={H}: expected fixed {f:.4f} and Thompson {t:.4f} below Bayes-optimal {b:.4f}, below oracle {o:.4f}")
    t, f = thompson(SYM_PRIOR, SYM, 3), fixed(SYM_PRIOR, SYM, 3)
    if not t > f + 0.2:
        fail(f"symmetric family, H=3: Thompson sampling should beat the fixed policy by more than 0.2, got {t:.4f} against {f:.4f}")


def _monte_carlo_objective(theta, task, n=400_000, seed=0):
    c, w = task
    rng = np.random.default_rng(seed)
    a = theta + SIGMA * rng.standard_normal((n, len(theta)))
    return float(np.mean(np.exp(-np.sum((a - c) ** 2, axis=1) / (2 * w * w))))


@check("task_objective, task_grad, and task_hessian agree with simulation and finite differences")
def test_task_derivatives(sub):
    task_objective = has(sub, "task_objective")
    task_grad = has(sub, "task_grad")
    task_hessian = has(sub, "task_hessian")
    theta = np.array([0.3, -0.4])
    for task in TASKS:
        got, want = task_objective(theta.copy(), task), _monte_carlo_objective(theta, task)
        if abs(got - want) > 0.005:
            fail(f"task (c={task[0]}, w={task[1]}): task_objective is {got:.4f}, but averaging 400000 sampled actions gives {want:.4f}")
        g = np.asarray(task_grad(theta.copy(), task))
        shape(g, (2,), label="task_grad")
        close(g, finite_difference(lambda x: task_objective(x, task), theta), rtol=1e-6, atol=1e-8, label="task_grad against finite differences")
        h = np.asarray(task_hessian(theta.copy(), task))
        shape(h, (2, 2), label="task_hessian")
        fd = np.stack([finite_difference(lambda x, i=i: task_grad(x, task)[i], theta) for i in range(2)])
        close(h, fd, rtol=1e-5, atol=1e-8, label="task_hessian against finite differences of task_grad")


@check("adapt and maml_objective are right, and maml_grad matches finite differences of the post-adaptation objective")
def test_maml_grad(sub):
    maml_grad = has(sub, "maml_grad")
    maml_objective = has(sub, "maml_objective")
    adapt = has(sub, "adapt")
    task_grad = has(sub, "task_grad")
    task_objective = has(sub, "task_objective")
    theta0 = np.array([0.1, 0.2])
    close(adapt(theta0.copy(), TASKS[1], ALPHA), theta0 + ALPHA * task_grad(theta0.copy(), TASKS[1]), atol=1e-12, label="adapt takes one ascent step")
    want = np.mean([task_objective(adapt(theta0.copy(), t, ALPHA), t) for t in TASKS])
    almost(maml_objective(theta0.copy(), TASKS, ALPHA), float(want), 1e-12, label="maml_objective averages the adapted objectives")
    # Named wrong implementation: first-order MAML, which returns the mean of
    # grad J_i at the adapted parameters and drops the (I + alpha * Hessian)
    # factor. It misses the finite-difference gradient by 50 percent or more here.
    for theta in (np.zeros(2), np.array([0.5, -0.3]), np.array([-1.0, 0.8])):
        got = np.asarray(maml_grad(theta.copy(), TASKS, ALPHA))
        shape(got, (2,), label="maml_grad")
        fd = finite_difference(lambda x: maml_objective(x, TASKS, ALPHA), theta)
        close(got, fd, rtol=1e-6, atol=1e-8, label=f"maml_grad at theta={theta}")


@check("the first-order gradient is off by exactly alpha times the mean of H_i(theta) grad J_i(adapted)")
def test_second_order_gap(sub):
    maml_objective = has(sub, "maml_objective")
    fomaml_grad = has(sub, "fomaml_grad")
    task_hessian = has(sub, "task_hessian")
    task_grad = has(sub, "task_grad")
    adapt = has(sub, "adapt")
    theta = np.array([0.5, -0.3])
    full = finite_difference(lambda x: maml_objective(x, TASKS, ALPHA), theta)
    first = np.asarray(fomaml_grad(theta.copy(), TASKS, ALPHA))
    gap = np.mean([ALPHA * task_hessian(theta.copy(), t) @ task_grad(adapt(theta.copy(), t, ALPHA), t) for t in TASKS], axis=0)
    if np.linalg.norm(full - first) < 0.05:
        fail(f"the first-order gradient {first} should differ visibly from the full gradient {full}; fomaml_grad may be returning the full one")
    close(full - first, gap, rtol=1e-5, atol=1e-8, label="full minus first-order gradient")


@check("maml_train finds an initialization that adapts better than the multi-task one")
def test_maml_train(sub):
    maml_step = has(sub, "maml_step")
    maml_train = has(sub, "maml_train")
    maml_objective = has(sub, "maml_objective")
    maml_grad = has(sub, "maml_grad")
    task_grad = has(sub, "task_grad")
    theta = np.array([0.2, 0.1])
    close(maml_step(theta.copy(), TASKS, ALPHA, 0.3), theta + 0.3 * maml_grad(theta.copy(), TASKS, ALPHA), atol=1e-12, label="maml_step")
    mt = np.zeros(2)
    for _ in range(600):
        mt = mt + 0.5 * np.mean([task_grad(mt.copy(), t) for t in TASKS], axis=0)
    base = maml_objective(mt.copy(), TASKS, ALPHA)
    meta = maml_train(mt.copy(), TASKS, ALPHA, 1.0, 150)
    after = maml_objective(meta.copy(), TASKS, ALPHA)
    if not after > base + 0.1:
        fail(f"after one adaptation step, the multi-task start scores {base:.4f}; MAML training should lift that by more than 0.1, got {after:.4f}")
    if not after > 0.55:
        fail(f"the post-adaptation objective after MAML training should exceed 0.55, got {after:.4f}")
