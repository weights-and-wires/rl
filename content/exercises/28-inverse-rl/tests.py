"""Checks for the inverse reinforcement learning exercise."""

import itertools

import numpy as np

import rlenvs
from rlcheck import almost as _almost, check, close, fail, finite, finite_difference, has, shape


def almost(got, want, tol, label=None):
    """rlcheck.almost, except that NaN or infinity fails instead of slipping past the comparison."""
    finite(got, label=label)
    _almost(got, want, tol, label=label)


def _corridor():
    """The lesson's worked example: two cells side by side, start on the left."""
    return rlenvs.gridworld(["S."])


def _small_grid():
    return rlenvs.gridworld(["S...", ".#..", "...."])


def _log_sigmoid(z):
    return -np.logaddexp(0.0, -z)


def _sample_trajectories(P, pi, mu0, n, seed):
    """Sample n state and action sequences from a time-indexed policy."""
    rng = np.random.default_rng(seed)
    horizon, S, A = pi.shape
    states = np.zeros((n, horizon), dtype=int)
    actions = np.zeros((n, horizon), dtype=int)
    for i in range(n):
        s = int(rng.choice(S, p=mu0))
        for t in range(horizon):
            a = int(rng.choice(A, p=pi[t, s]))
            states[i, t], actions[i, t] = s, a
            s = int(rng.choice(S, p=P[s, a]))
    return states, actions


@check("soft_value_iteration reproduces the corridor example: pi_0(right | left) = e / (e + 3)")
def test_soft_vi_corridor(sub):
    soft_value_iteration = has(sub, "soft_value_iteration")
    mdp = _corridor()
    V, pi = soft_value_iteration(mdp.P, np.array([0.0, 1.0]), 2)
    shape(V, (3, 2), label="V")
    shape(pi, (2, 2, 4), label="pi")
    close(pi.sum(axis=2), np.ones((2, 2)), atol=1e-12, label="rows of pi")
    close(V[2], np.zeros(2), atol=1e-12, label="V at the horizon")
    # From the left cell, 4 of the 16 action sequences step right first and earn 1.
    almost(pi[0, 0, rlenvs.RIGHT], np.e / (np.e + 3), tol=1e-12, label="pi_0(right | left cell)")
    almost(V[0, 0], np.log(4 * np.e + 12), tol=1e-12, label="V_0(left cell) = log(4e + 12)")
    close(V[1], np.array([0.0, 1.0]) + np.log(4.0), atol=1e-12, label="V_1 = r + log 4")


@check("soft_value_iteration gives every action sequence probability proportional to exp(its return)")
def test_soft_vi_enumeration(sub):
    soft_value_iteration = has(sub, "soft_value_iteration")
    mdp = rlenvs.gridworld(["S.", ".."])
    r = np.array([0.3, -0.5, 1.2, 0.1])
    horizon = 3
    V, pi = soft_value_iteration(mdp.P, r, horizon)
    start = mdp.names["start"]
    returns, probs = [], []
    for seq in itertools.product(range(4), repeat=horizon):
        s, ret, p = start, 0.0, 1.0
        for t, a in enumerate(seq):
            ret += r[s]
            p *= pi[t, s, a]
            s = int(np.argmax(mdp.P[s, a]))
        returns.append(ret)
        probs.append(p)
    returns, probs = np.array(returns), np.array(probs)
    want = np.exp(returns) / np.exp(returns).sum()
    close(probs, want, atol=1e-12, label="probability of each of the 64 action sequences")
    almost(V[0, start], np.log(np.exp(returns).sum()), tol=1e-10,
           label="V_0(start) against log sum exp(return) over all sequences")


@check("state_visitation propagates the corridor example and sums to one at every step")
def test_state_visitation(sub):
    soft_value_iteration = has(sub, "soft_value_iteration")
    state_visitation = has(sub, "state_visitation")
    mdp = _corridor()
    _, pi = soft_value_iteration(mdp.P, np.array([0.0, 1.0]), 2)
    D = state_visitation(mdp.P, pi, mdp.mu0)
    shape(D, (2, 2), label="D")
    p = np.e / (np.e + 3)
    close(D, np.array([[1.0, 0.0], [1 - p, p]]), atol=1e-12, label="D on the corridor")

    grid = _small_grid()
    rng = np.random.default_rng(0)
    _, pi = soft_value_iteration(grid.P, rng.normal(size=grid.n_states), 6)
    D = state_visitation(grid.P, pi, grid.mu0)
    close(D.sum(axis=1), np.ones(6), atol=1e-12, label="each row of D")
    states, _ = _sample_trajectories(grid.P, pi, grid.mu0, 4000, seed=1)
    freq = np.stack([np.bincount(states[:, t], minlength=grid.n_states) / 4000 for t in range(6)])
    if not np.abs(freq - D).max() <= 0.04:
        fail(f"D disagrees with 4000 sampled trajectories by {np.abs(freq - D).max():.3f}, expected under 0.04")


@check("feature_expectations and empirical_feature_counts match hand counts")
def test_feature_counts(sub):
    feature_expectations = has(sub, "feature_expectations")
    empirical_feature_counts = has(sub, "empirical_feature_counts")
    p = np.e / (np.e + 3)
    D = np.array([[1.0, 0.0], [1 - p, p]])
    close(feature_expectations(D, np.eye(2)), np.array([2 - p, p]), atol=1e-12,
          label="expected one-hot counts on the corridor")
    features = np.array([[1.0, 0.0], [1.0, 2.0], [0.0, 1.0]])
    states = np.array([[0, 1, 1], [0, 2, 2]])
    # Trajectory one: [1,0] + [1,2] + [1,2] = [3,4]. Trajectory two: [1,0] + [0,1] + [0,1] = [1,2].
    close(empirical_feature_counts(states, features), np.array([2.0, 3.0]), atol=1e-12,
          label="mean summed features over two trajectories")


@check("maxent_irl_gradient equals the finite-difference gradient of the demonstrations' log-likelihood")
def test_irl_gradient(sub):
    soft_value_iteration = has(sub, "soft_value_iteration")
    empirical_feature_counts = has(sub, "empirical_feature_counts")
    maxent_irl_gradient = has(sub, "maxent_irl_gradient")
    mdp = _small_grid()
    S, horizon = mdp.n_states, 5
    rng = np.random.default_rng(2)
    features = rng.normal(size=(S, 3))
    theta_expert = np.array([1.0, -0.5, 0.8])
    _, pi_expert = soft_value_iteration(mdp.P, features @ theta_expert, horizon)
    states, actions = _sample_trajectories(mdp.P, pi_expert, mdp.mu0, 30, seed=3)
    f_expert = empirical_feature_counts(states, features)

    def log_likelihood(theta):
        _, pi = soft_value_iteration(mdp.P, features @ theta, horizon)
        t = np.arange(horizon)[None, :]
        return float(np.mean(np.log(pi[t, states, actions]).sum(axis=1)))

    theta = np.array([0.2, 0.1, -0.3])
    fd = finite_difference(log_likelihood, theta.copy(), eps=1e-5)
    got = maxent_irl_gradient(mdp.P, features, mdp.mu0, horizon, theta, f_expert)
    shape(got, (3,), label="gradient")
    close(got, fd, rtol=1e-5, atol=1e-6, label="gradient against finite differences of the log-likelihood")


@check("maxent_irl_gradient is zero at the true reward when the expert counts are exact")
def test_irl_gradient_zero(sub):
    soft_value_iteration = has(sub, "soft_value_iteration")
    state_visitation = has(sub, "state_visitation")
    feature_expectations = has(sub, "feature_expectations")
    maxent_irl_gradient = has(sub, "maxent_irl_gradient")
    mdp = _small_grid()
    features = np.eye(mdp.n_states)
    theta = np.random.default_rng(4).normal(size=mdp.n_states)
    _, pi = soft_value_iteration(mdp.P, features @ theta, 6)
    f_exact = feature_expectations(state_visitation(mdp.P, pi, mdp.mu0), features)
    got = maxent_irl_gradient(mdp.P, features, mdp.mu0, 6, theta, f_exact)
    close(got, np.zeros(mdp.n_states), atol=1e-12, label="gradient at the reward that made the data")


@check("maxent_irl matches the expert's feature counts and behaviour, though not its reward")
def test_maxent_irl(sub):
    soft_value_iteration = has(sub, "soft_value_iteration")
    state_visitation = has(sub, "state_visitation")
    feature_expectations = has(sub, "feature_expectations")
    maxent_irl = has(sub, "maxent_irl")
    mdp = _small_grid()
    S, horizon = mdp.n_states, 8
    features = np.eye(S)
    theta_true = np.random.default_rng(0).normal(size=S)
    _, pi_true = soft_value_iteration(mdp.P, features @ theta_true, horizon)
    D_true = state_visitation(mdp.P, pi_true, mdp.mu0)
    f_expert = feature_expectations(D_true, features)

    theta = np.asarray(maxent_irl(mdp.P, features, mdp.mu0, horizon, f_expert, lr=0.2, iters=1000))
    shape(theta, (S,), label="theta")
    _, pi = soft_value_iteration(mdp.P, features @ theta, horizon)
    f_learned = feature_expectations(state_visitation(mdp.P, pi, mdp.mu0), features)
    gap = np.abs(f_learned - f_expert).max()
    if not gap <= 0.02:
        fail(f"learned feature counts differ from the expert's by {gap:.4f}, expected under 0.02")
    # Compare policies only where the expert actually goes.
    diff = np.abs((pi - pi_true) * D_true[:, :, None]).max()
    if not diff <= 0.005:
        fail(f"visitation-weighted policy gap is {diff:.4f}, expected under 0.005")


@check("discriminator_grad matches finite differences of discriminator_loss")
def test_discriminator_grad(sub):
    discriminator_loss = has(sub, "discriminator_loss")
    discriminator_grad = has(sub, "discriminator_grad")
    rng = np.random.default_rng(5)
    rho_e = rng.dirichlet(np.ones(6)).reshape(2, 3)
    rho_p = rng.dirichlet(np.ones(6)).reshape(2, 3)
    logits = rng.normal(size=(2, 3))
    almost(discriminator_loss(np.zeros((2, 3)), rho_e, rho_p), 2 * np.log(2.0), tol=1e-12,
           label="loss at D = 1/2 everywhere")
    want = -np.sum(rho_e * _log_sigmoid(logits)) - np.sum(rho_p * _log_sigmoid(-logits))
    almost(discriminator_loss(logits, rho_e, rho_p), want, tol=1e-12, label="loss at random logits")
    fd = finite_difference(lambda l: discriminator_loss(l, rho_e, rho_p), logits.copy())
    close(discriminator_grad(logits, rho_e, rho_p), fd, rtol=1e-6, atol=1e-8,
          label="gradient against finite differences")


@check("the best discriminator has logit log(rho_E / rho_pi), and its loss is 2 log 2 - 2 JS")
def test_optimal_discriminator(sub):
    discriminator_loss = has(sub, "discriminator_loss")
    discriminator_grad = has(sub, "discriminator_grad")
    rng = np.random.default_rng(6)
    rho_e = rng.dirichlet(np.ones(8))
    rho_p = rng.dirichlet(np.ones(8))
    best = np.log(rho_e / rho_p)
    close(discriminator_grad(best, rho_e, rho_p), np.zeros(8), atol=1e-12,
          label="gradient at logits log(rho_E / rho_pi)")
    m = 0.5 * (rho_e + rho_p)
    js = 0.5 * np.sum(rho_e * np.log(rho_e / m)) + 0.5 * np.sum(rho_p * np.log(rho_p / m))
    almost(discriminator_loss(best, rho_e, rho_p), 2 * np.log(2.0) - 2 * js, tol=1e-10,
           label="loss of the best discriminator")
    for _ in range(20):
        other = best + rng.normal(size=8)
        if discriminator_loss(other, rho_e, rho_p) < discriminator_loss(best, rho_e, rho_p) - 1e-12:
            fail("a perturbed discriminator scored a lower loss than the optimum")


@check("gail_reward is -log(1 - D), stable at large logits, and gail_reward(l) - gail_reward(-l) = l")
def test_gail_reward(sub):
    gail_reward = has(sub, "gail_reward")
    logits = np.array([-2.0, 0.0, 1.5])
    D = 1 / (1 + np.exp(-logits))
    close(gail_reward(logits), -np.log(1 - D), atol=1e-12, label="-log(1 - D)")
    big = gail_reward(np.array([-800.0, 800.0]))
    close(big, np.array([0.0, 800.0]), atol=1e-9, label="gail_reward at logits of -800 and 800")
    l = np.linspace(-30, 30, 13)
    close(gail_reward(l) - gail_reward(-l), l, atol=1e-9,
          label="-log(1 - D) + log D, which should equal the logit")
