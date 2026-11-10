"""Checks for the advantage-weighted policy improvement exercise."""

import numpy as np

import rlenvs
from rlcheck import almost as _almost, check, close, fail, finite, finite_difference, has, shape


def almost(got, want, tol, label=None):
    """rlcheck.almost, except that NaN or infinity fails instead of slipping past the comparison."""
    finite(got, label=label)
    _almost(got, want, tol, label=label)


def make_dataset(mdp, behavior, n_episodes, seed, max_steps=500):
    """The same helper the starter provides, so the checks don't depend on the reader's copy."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_episodes):
        s = int(rng.choice(mdp.n_states, p=mdp.mu0))
        for _ in range(max_steps):
            a = int(rng.choice(mdp.n_actions, p=behavior[s]))
            s2, r = mdp.sample(s, a, rng)
            done = bool(mdp.terminal[s2])
            rows.append((s, a, r, s2, float(done)))
            if done:
                break
            s = s2
    s, a, r, s2, done = (np.array(col) for col in zip(*rows))
    return {"s": s.astype(int), "a": a.astype(int), "r": r.astype(float), "s2": s2.astype(int), "done": done}


def cautious_behavior(mdp):
    """Heads for the goal, but never steps toward the cliff: no down in row 2, no right at the start."""
    S, A = mdp.n_states, mdp.n_actions
    allowed = np.ones((S, A), dtype=bool)
    for k, (i, j) in enumerate(mdp.names["cells"]):
        if i == 2 and 1 <= j <= 10:
            allowed[k, rlenvs.DOWN] = False
    allowed[mdp.names["start"], rlenvs.RIGHT] = False
    q = rlenvs.q_from_v(mdp, rlenvs.value_iteration(mdp))
    best = np.where(allowed, q, -np.inf).argmax(axis=1)
    pi = 0.4 * allowed / allowed.sum(axis=1, keepdims=True)
    pi[np.arange(S), best] += 0.6
    return pi


def _kl_to_uniform(w):
    w = np.asarray(w, dtype=float)
    pos = w > 0
    return float(np.sum(w[pos] * np.log(w.size * w[pos])))


PI_OLD = np.array([0.5, 0.3, 0.2])
ADV = np.array([1.0, 0.0, 2.0])
# pi_old * exp(adv) = (0.5 e, 0.3, 0.2 e^2) = (1.3591, 0.3, 1.4778), Z = 3.1370.
PI_STAR = np.array([0.43327, 0.09564, 0.47109])


@check("awr_weights is exp(adv / beta), capped at clip, with no overflow")
def test_awr_weights(sub):
    awr_weights = has(sub, "awr_weights")
    # Named wrong implementation: exp(adv * beta) instead of exp(adv / beta).
    # At beta = 0.5 it returns e^0.5 = 1.6487 for adv = 1, not e^2 = 7.3891.
    adv = np.array([-1.0, 0.0, 1.0, 2.0, 100.0])
    got = np.asarray(awr_weights(adv, 1.0, 20.0), dtype=float)
    shape(got, (5,))
    close(got, np.array([np.exp(-1.0), 1.0, np.e, np.e**2, 20.0]), atol=1e-9, label="beta=1, clip=20")
    adv = np.array([-1.0, 0.0, 1.0, 2.0])
    got = np.asarray(awr_weights(adv, 0.5, 20.0), dtype=float)
    close(got, np.array([np.exp(-2.0), 1.0, np.exp(2.0), 20.0]), atol=1e-9, label="beta=0.5, clip=20")
    with np.errstate(over="raise", invalid="raise"):
        got = np.asarray(awr_weights(np.array([1e4, -1e4]), 1.0, 20.0), dtype=float)
    close(got, np.array([20.0, 0.0]), atol=1e-12, label="an advantage of +/-1e4")


@check("kl_improved_policy returns pi_old * exp(adv / beta) / Z, and a shift in adv changes nothing")
def test_kl_improved_policy(sub):
    kl_improved_policy = has(sub, "kl_improved_policy")
    got = np.asarray(kl_improved_policy(PI_OLD, ADV, 1.0), dtype=float)
    shape(got, (3,))
    close(got, PI_STAR, atol=1e-4, label="the worked example at beta=1")
    almost(got.sum(), 1.0, 1e-12, label="sum of the improved policy")
    # The state value V(s) cancels in the normalization, so Q and A give the same policy.
    shifted = np.asarray(kl_improved_policy(PI_OLD, ADV + 1000.0, 1.0), dtype=float)
    close(shifted, got, atol=1e-9, label="adv + 1000")
    far = np.asarray(kl_improved_policy(PI_OLD, ADV, 1e6), dtype=float)
    close(far, PI_OLD, atol=1e-5, label="a huge beta leaves pi_old alone")
    sharp = np.asarray(kl_improved_policy(PI_OLD, ADV, 1e-3), dtype=float)
    close(sharp, np.array([0.0, 0.0, 1.0]), atol=1e-9, label="a tiny beta picks the best action")


@check("improvement_objective is E[adv] - beta * KL, and the improved policy beats 3000 random rivals")
def test_improvement_objective(sub):
    improvement_objective = has(sub, "improvement_objective")
    kl_improved_policy = has(sub, "kl_improved_policy")
    almost(improvement_objective(PI_OLD, PI_OLD, ADV, 1.0), 0.9, 1e-12, label="pi_old scores E[adv] with zero KL")
    almost(improvement_objective(np.array([0.0, 0.0, 1.0]), PI_OLD, ADV, 1.0), 2.0 - np.log(5.0), 1e-12,
           label="all mass on action 2 (zero entries must not give nan)")
    star = np.asarray(kl_improved_policy(PI_OLD, ADV, 1.0), dtype=float)
    z = float(np.sum(PI_OLD * np.exp(ADV)))
    best = improvement_objective(star, PI_OLD, ADV, 1.0)
    almost(best, np.log(z), 1e-9, label="the optimum equals beta * log Z")
    rng = np.random.default_rng(0)
    for rival in rng.dirichlet(np.ones(3), size=3000):
        val = improvement_objective(rival, PI_OLD, ADV, 1.0)
        if val > best + 1e-12:
            fail(f"policy {np.round(rival, 4)} scores {val:.6f}, above the improved policy's {best:.6f}")
    rng = np.random.default_rng(1)
    pi_old = rng.dirichlet(np.ones(5))
    adv = rng.normal(size=5)
    star = np.asarray(kl_improved_policy(pi_old, adv, 0.7), dtype=float)
    best = improvement_objective(star, pi_old, adv, 0.7)
    for rival in rng.dirichlet(np.ones(5), size=3000):
        if improvement_objective(rival, pi_old, adv, 0.7) > best + 1e-12:
            fail("a random policy beat the improved policy on five actions at beta=0.7")


@check("mpo_weights is a softmax of q / eta over the batch, stable for q near 1000")
def test_mpo_weights(sub):
    mpo_weights = has(sub, "mpo_weights")
    # Named wrong implementation: weights without the batch normalization, exp(q / eta) alone.
    q = np.array([0.0, 1.0, 2.0])
    got = np.asarray(mpo_weights(q, 1.0), dtype=float)
    shape(got, (3,))
    close(got, np.array([0.09003, 0.24473, 0.66524]), atol=1e-4, label="softmax of (0, 1, 2)")
    almost(got.sum(), 1.0, 1e-12, label="sum of the weights")
    with np.errstate(over="raise", invalid="raise"):
        big = np.asarray(mpo_weights(np.array([1000.0, 999.0, 998.0]), 0.5), dtype=float)
    close(big, np.asarray(mpo_weights(np.array([2.0, 1.0, 0.0]), 0.5), dtype=float), atol=1e-9,
          label="q shifted by 998")


@check("mpo_dual has derivative eps - KL(w || uniform) in eta, by finite differences")
def test_mpo_dual(sub):
    mpo_dual = has(sub, "mpo_dual")
    mpo_weights = has(sub, "mpo_weights")
    rng = np.random.default_rng(0)
    q = rng.normal(size=32) * 2.0
    eps = 0.3
    for eta in (0.4, 1.0, 3.0):
        fd = finite_difference(lambda e: mpo_dual(q, float(e), eps), np.array(eta), eps=1e-6)
        want = eps - _kl_to_uniform(mpo_weights(q, eta))
        almost(fd, want, 1e-5, label=f"d dual / d eta at eta={eta}")
    almost(mpo_dual(np.array([0.0, 0.0]), 2.0, 0.1), 0.2, 1e-12, label="equal q's give eta * eps")
    almost(mpo_dual(np.array([1000.0, 1000.0]), 2.0, 0.1), 1000.2, 1e-9, label="q near 1000 must not overflow")


@check("mpo_temperature puts the KL of the reweighted samples at eps, and eta falls as eps grows")
def test_mpo_temperature(sub):
    mpo_temperature = has(sub, "mpo_temperature")
    mpo_weights = has(sub, "mpo_weights")
    mpo_dual = has(sub, "mpo_dual")
    rng = np.random.default_rng(0)
    q = rng.normal(size=64) * 3.0
    etas = []
    for eps in (0.01, 0.1, 0.5):
        eta = float(mpo_temperature(q, eps))
        kl = _kl_to_uniform(mpo_weights(q, eta))
        almost(kl, eps, 1e-5, label=f"KL of the weights at eps={eps} (eta={eta:.4f})")
        etas.append(eta)
        for factor in (0.8, 1.25):
            if mpo_dual(q, eta * factor, eps) < mpo_dual(q, eta, eps) - 1e-12:
                fail(f"the dual is lower at eta*{factor} than at the returned eta={eta:.4f} (eps={eps})")
    if not (etas[0] > etas[1] > etas[2]):
        fail(f"eta should fall as eps grows, got {np.round(etas, 4)} for eps = 0.01, 0.1, 0.5")
    # Multiplying every q by 10 multiplies the temperature by 10 and leaves the weights alone.
    almost(mpo_temperature(10.0 * q, 0.1), 10.0 * etas[1], 1e-3 * etas[1] * 10.0, label="eta for 10 * q")


@check("awr_loss and its logit gradient match the weighted log-likelihood, by finite differences")
def test_awr_loss(sub):
    awr_loss = has(sub, "awr_loss")
    awr_loss_grad = has(sub, "awr_loss_grad")
    almost(awr_loss(np.zeros((1, 2)), np.array([0]), np.array([2.0])), 2.0 * np.log(2.0), 1e-12,
           label="one sample with weight 2 and logits (0, 0)")
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(6, 4))
    actions = rng.integers(0, 4, size=6)
    weights = rng.uniform(0.0, 5.0, size=6)
    weights[2] = 0.0
    grad = np.asarray(awr_loss_grad(logits, actions, weights), dtype=float)
    shape(grad, (6, 4))
    fd = finite_difference(lambda x: awr_loss(x.reshape(6, 4), actions, weights), logits.copy())
    close(grad, fd, atol=1e-6, label="gradient with respect to the logits")
    close(grad[2], np.zeros(4), atol=1e-15, label="a sample with weight 0 pulls on nothing")
    almost(awr_loss(np.array([[1000.0, 0.0]]), np.array([0]), np.array([1.0])), 0.0, 1e-9,
           label="logits near 1000 must not overflow")


@check("weighted_gaussian_mle is the weighted mean and spread, and the weighted log-likelihood is flat there")
def test_weighted_gaussian_mle(sub):
    weighted_gaussian_mle = has(sub, "weighted_gaussian_mle")
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 2.0, size=50)
    mu, sigma = weighted_gaussian_mle(a, np.ones(50))
    almost(mu, a.mean(), 1e-10, label="mu with equal weights")
    almost(sigma, a.std(), 1e-10, label="sigma with equal weights")
    w = rng.uniform(0.1, 3.0, size=50)
    mu, sigma = weighted_gaussian_mle(a, w)
    mu2, sigma2 = weighted_gaussian_mle(a, 7.0 * w)
    almost(mu2, mu, 1e-10, label="mu after scaling the weights by 7")
    almost(sigma2, sigma, 1e-10, label="sigma after scaling the weights by 7")

    def neg_ll(p):
        m, log_s = p
        s = np.exp(log_s)
        return float(np.sum(w * (0.5 * ((a - m) / s) ** 2 + log_s + 0.5 * np.log(2 * np.pi))))

    grad = finite_difference(neg_ll, np.array([mu, np.log(sigma)]))
    close(grad, np.zeros(2), atol=1e-5, label="gradient of the weighted NLL at the returned (mu, log sigma)")
    if neg_ll(np.array([mu + 0.1, np.log(sigma)])) <= neg_ll(np.array([mu, np.log(sigma)])):
        fail("moving mu by 0.1 lowered the weighted NLL, so the returned mu is not the maximizer")


@check("mpo_gaussian_step on 20000 samples of N(0, 1) with q = -(a - 1)^2 lands on the reweighted Gaussian")
def test_mpo_gaussian_step(sub):
    mpo_gaussian_step = has(sub, "mpo_gaussian_step")
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, size=20000)
    q = -((a - 1.0) ** 2)
    mu, sigma, eta = mpo_gaussian_step(a, q, 0.1)
    # N(0, 1) * exp(-(a - 1)^2 / eta) is Gaussian with precision 1 + 2/eta and mean (2/eta) / (1 + 2/eta).
    prec = 1.0 + 2.0 / eta
    almost(mu, (2.0 / eta) / prec, 0.03, label=f"new mean at eta={eta:.4f}")
    almost(sigma, prec**-0.5, 0.03, label=f"new sigma at eta={eta:.4f}")
    if not mu > 0.1:
        fail(f"the mean should move from 0 toward the best action 1, got {mu:.4f}")
    if not sigma < 1.0:
        fail(f"the spread should shrink from 1 as the weights concentrate, got {sigma:.4f}")


@check("awac_tabular beats behavior cloning on the cliff by a wide margin, and a huge beta changes nothing")
def test_awac_tabular(sub):
    awac_tabular = has(sub, "awac_tabular")
    mdp = rlenvs.cliff_walking(gamma=0.9)
    S, A = mdp.n_states, mdp.n_actions
    start = mdp.names["start"]
    data = make_dataset(mdp, cautious_behavior(mdp), 100, seed=0)
    counts = np.zeros((S, A))
    np.add.at(counts, (data["s"], data["a"]), 1.0)
    totals = counts.sum(axis=1, keepdims=True)
    bc = np.where(totals > 0, counts / np.maximum(totals, 1.0), 1.0 / A)
    bc_value = rlenvs.policy_evaluation(mdp, bc)[start]
    pi = np.asarray(awac_tabular(data, S, A, mdp.gamma, beta=0.3), dtype=float)
    shape(pi, (S, A))
    close(pi.sum(axis=1), np.ones(S), atol=1e-9, label="every row of the policy sums to 1")
    unseen_at_covered = (counts == 0) & (totals > 0)
    if np.any(pi[unseen_at_covered] > 1e-12):
        fail("AWAC put probability on an action the data never took at a state it covers")
    value = rlenvs.policy_evaluation(mdp, pi)[start]
    # Behavior cloning is worth about -9.02 at the start, AWAC at beta=0.3 about -7.63, over seeds 0 to 4.
    if not value > bc_value + 1.0:
        fail(f"AWAC value at the start is {value:.3f}; expected above behavior cloning's {bc_value:.3f} by at least 1.0")
    flat = np.asarray(awac_tabular(data, S, A, mdp.gamma, beta=1e6), dtype=float)
    close(flat, bc, atol=1e-4, label="beta = 1e6 should leave the cloned policy alone")
