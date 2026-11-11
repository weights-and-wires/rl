"""Checks for the off-policy evaluation exercise."""

import numpy as np

import rlenvs
from rlcheck import almost, check, close, fail, has, shape

# ---------------------------------------------------------------------------
# Test fixtures. Nothing here imports the reader's code.
# ---------------------------------------------------------------------------

P_X = np.array([0.5, 0.3, 0.2])
B_BANDIT = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]])
PI_BANDIT = np.array([[0.1, 0.2, 0.7], [0.7, 0.1, 0.2], [0.2, 0.7, 0.1]])


def _bandit_world(seed):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.0, 1.0, size=(3, 3))
    truth = float(P_X @ (PI_BANDIT * mu).sum(axis=1))
    return rng, mu, truth


def _log_batch(rng, mu, n, noise):
    x = rng.choice(3, size=n, p=P_X)
    u = rng.random(n)
    a = np.minimum((u[:, None] > np.cumsum(B_BANDIT[x], axis=1)).sum(axis=1), 2)
    r = mu[x, a] + rng.normal(0.0, noise, size=n)
    return x, a, r


def _batches(fn, seed, reps, n, noise, q_noise=None):
    """Run an estimator on many independent logs and return its estimates and the truth."""
    rng, mu, truth = _bandit_world(seed)
    q_hat = mu + rng.normal(0.0, q_noise, size=mu.shape) if q_noise is not None else None
    out = np.zeros(reps)
    for i in range(reps):
        x, a, r = _log_batch(rng, mu, n, noise)
        out[i] = fn(x, a, r, q_hat)
    return out, truth


def _finite_horizon_q(mdp, pi, horizon):
    q = np.zeros((horizon, mdp.n_states, mdp.n_actions))
    q[horizon - 1] = mdp.R
    for t in range(horizon - 2, -1, -1):
        q[t] = mdp.R + mdp.gamma * mdp.P @ (pi * q[t + 1]).sum(axis=1)
    return q


def _episodes(mdp, b, n, horizon, rng):
    def draw(p):
        u = rng.random(len(p))
        return np.minimum((u[:, None] > np.cumsum(p, axis=1)).sum(axis=1), p.shape[1] - 1)

    S = np.zeros((n, horizon), dtype=int)
    A = np.zeros((n, horizon), dtype=int)
    R = np.zeros((n, horizon))
    s = rng.choice(mdp.n_states, size=n, p=mdp.mu0)
    for t in range(horizon):
        a = draw(b[s])
        S[:, t], A[:, t], R[:, t] = s, a, mdp.R[s, a]
        s = draw(mdp.P[s, a])
    return S, A, R


def _mdp_world():
    mdp = rlenvs.random_mdp(4, 2, gamma=0.9, seed=1)
    b = np.full((4, 2), 0.5)
    pi = np.array([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]])
    horizon = 5
    q = _finite_horizon_q(mdp, pi, horizon)
    exact = float(mdp.mu0 @ (pi * q[0]).sum(axis=1))
    return mdp, b, pi, horizon, q, exact


# A four-sample log used by the hand-worked checks.
HX = np.array([0, 0, 1, 1])
HA = np.array([0, 0, 1, 0])
HR = np.array([1.0, 0.8, 0.5, 0.2])
H_PI = np.array([[0.8, 0.2], [0.1, 0.9]])
H_B = np.full((2, 2), 0.5)
H_Q = np.array([[0.6, 0.1], [0.3, 0.4]])


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


@check("effective_sample_size is N for equal weights, 1 for one nonzero weight, and 1.6 for (3, 1)")
def test_ess(sub):
    f = has(sub, "effective_sample_size")
    almost(f(np.ones(7)), 7.0, 1e-12, label="ESS of 7 equal weights")
    almost(f(np.array([1.0, 0.0, 0.0, 0.0])), 1.0, 1e-12, label="ESS with one nonzero weight")
    # (3 + 1)^2 / (9 + 1) = 1.6
    almost(f(np.array([3.0, 1.0])), 1.6, 1e-12, label="ESS of (3, 1)")
    almost(f(np.zeros(3)), 0.0, 1e-12, label="ESS when every weight is zero")


@check("ips averages r * pi / b over the log and snips divides by the total weight instead")
def test_ips_snips(sub):
    ips = has(sub, "ips")
    snips = has(sub, "snips")
    # Weights 1.6, 1.6, 1.8, 0.2. Sum of w * r = 3.82, sum of w = 5.2.
    almost(ips(HX, HA, HR, H_PI, H_B), 3.82 / 4, 1e-12, label="ips on the hand log (3.82 / 4)")
    almost(snips(HX, HA, HR, H_PI, H_B), 3.82 / 5.2, 1e-12, label="snips on the hand log (3.82 / 5.2)")
    zero_pi = np.array([[0.0, 1.0], [0.5, 0.5]])
    almost(snips(HX[:2], HA[:2], HR[:2], zero_pi, H_B), 0.0, 1e-12, label="snips when every weight is 0")


@check("doubly_robust equals the model baseline plus a weighted residual, and ips when q_hat is zero")
def test_dr_hand(sub):
    dr = has(sub, "doubly_robust")
    # V_hat = (0.5, 0.39). Samples: 0.5 + 1.6 * 0.4, 0.5 + 1.6 * 0.2,
    # 0.39 + 1.8 * 0.1, 0.39 + 0.2 * (-0.1). The mean is 2.9 / 4.
    almost(dr(HX, HA, HR, H_PI, H_B, H_Q), 0.725, 1e-12, label="doubly_robust on the hand log")
    ips_value = dr(HX, HA, HR, H_PI, H_B, np.zeros((2, 2)))
    almost(ips_value, 3.82 / 4, 1e-12, label="doubly_robust with q_hat = 0 (it must equal ips)")


@check("ips and doubly_robust are unbiased over 4000 logs, even when q_hat is wrong; snips is biased")
def test_bandit_unbiased(sub):
    # Wrong implementation caught here: a self-normalized ips (dividing by the
    # sum of the weights). With n = 10 its bias is many standard errors.
    ips = has(sub, "ips")
    dr = has(sub, "doubly_robust")
    for name, fn, q_noise in [
        ("ips", lambda x, a, r, q: ips(x, a, r, PI_BANDIT, B_BANDIT), None),
        ("doubly_robust with a noisy q_hat", lambda x, a, r, q: dr(x, a, r, PI_BANDIT, B_BANDIT, q), 0.3),
    ]:
        est, truth = _batches(fn, seed=0, reps=4000, n=10, noise=0.3, q_noise=q_noise)
        se = est.std(ddof=1) / np.sqrt(len(est))
        if abs(est.mean() - truth) > 4 * se:
            fail(
                f"{name}: the mean over 4000 logs should be within 4 standard errors "
                f"(4 * {se:.4f}) of the true value {truth:.4f}, got {est.mean():.4f}"
            )


@check("doubly_robust has far less variance than ips with a decent q_hat, and none with the exact q on one context")
def test_dr_variance(sub):
    ips = has(sub, "ips")
    dr = has(sub, "doubly_robust")
    v_ips, _ = _batches(lambda x, a, r, q: ips(x, a, r, PI_BANDIT, B_BANDIT), seed=0, reps=1500, n=50, noise=0.1)
    v_dr, _ = _batches(
        lambda x, a, r, q: dr(x, a, r, PI_BANDIT, B_BANDIT, q), seed=0, reps=1500, n=50, noise=0.1, q_noise=0.05
    )
    if not v_dr.var() < 0.5 * v_ips.var():
        fail(f"doubly_robust variance should be under half of ips variance {v_ips.var():.5f}, got {v_dr.var():.5f}")
    # One context, deterministic rewards, exact q_hat: every sample gives the exact value.
    mu = np.array([[0.2, 0.9, 0.4]])
    pi = np.array([[0.1, 0.6, 0.3]])
    b = np.array([[0.5, 0.25, 0.25]])
    rng = np.random.default_rng(3)
    a = rng.choice(3, size=30, p=b[0])
    x = np.zeros(30, dtype=int)
    got = dr(x, a, mu[x, a], pi, b, mu)
    almost(got, float((pi * mu).sum()), 1e-12, label="doubly_robust with the exact q_hat, deterministic rewards")
    ips_got = ips(x, a, mu[x, a], pi, b)
    if abs(ips_got - float((pi * mu).sum())) < 1e-6:
        fail("this log was meant to show ips off by sampling noise, so the test log is wrong")


@check("per_decision_is weights r_t by the ratios up to step t only, and is unbiased on a random MDP")
def test_pdis(sub):
    pdis = has(sub, "per_decision_is")
    # One 2-step trajectory with ratios 1.6 and 1.8: 1.0 * 1.6 + 0.9 * 2.0 * (1.6 * 1.8) = 6.784.
    # Using the whole-trajectory ratio on every reward would give 8.064 instead.
    S = np.array([[0, 1]])
    A = np.array([[0, 1]])
    R = np.array([[1.0, 2.0]])
    got = pdis(S, A, R, H_PI, H_B, 0.9)
    shape(got, (1,), label="per_decision_is output")
    close(got, np.array([6.784]), atol=1e-12, label="per_decision_is on the 2-step trajectory")
    mdp, b, pi, horizon, _, exact = _mdp_world()
    S, A, R = _episodes(mdp, b, 20000, horizon, np.random.default_rng(0))
    est = pdis(S, A, R, pi, b, mdp.gamma)
    se = est.std(ddof=1) / np.sqrt(len(est))
    if abs(est.mean() - exact) > 4 * se:
        fail(f"mean over 20000 trajectories should be within 4 * {se:.4f} of {exact:.4f}, got {est.mean():.4f}")


@check("doubly_robust_trajectory follows the backward recursion on a hand-worked trajectory")
def test_dr_traj_hand(sub):
    f = has(sub, "doubly_robust_trajectory")
    S = np.array([[0, 1]])
    A = np.array([[0, 1]])
    R = np.array([[1.0, 2.0]])
    q = np.array([[[0.5, 0.2], [0.3, 1.0]], [[0.4, 0.1], [0.5, 1.8]]])
    # t = 1: V_hat = 0.1 * 0.5 + 0.9 * 1.8 = 1.67, est = 1.67 + 1.8 * (2.0 - 1.8) = 2.03.
    # t = 0: V_hat = 0.8 * 0.5 + 0.2 * 0.2 = 0.44, est = 0.44 + 1.6 * (1.0 + 0.9 * 2.03 - 0.5) = 4.1632.
    got = f(S, A, R, H_PI, H_B, 0.9, q)
    shape(got, (1,), label="doubly_robust_trajectory output")
    # Wrong implementation caught here: weighting the correction at step t by the
    # ratio of the step before (a cumulative product off by one) gives another number.
    close(got, np.array([4.1632]), atol=1e-12, label="doubly_robust_trajectory on the 2-step trajectory")


@check("doubly_robust_trajectory is unbiased on a random MDP and beats per_decision_is with the exact q")
def test_dr_traj_stats(sub):
    pdis = has(sub, "per_decision_is")
    f = has(sub, "doubly_robust_trajectory")
    mdp, b, pi, horizon, q, exact = _mdp_world()
    S, A, R = _episodes(mdp, b, 20000, horizon, np.random.default_rng(0))
    est = f(S, A, R, pi, b, mdp.gamma, q)
    se = est.std(ddof=1) / np.sqrt(len(est))
    if abs(est.mean() - exact) > 4 * se:
        fail(f"exact-q estimate: the mean should be within 4 * {se:.4f} of {exact:.4f}, got {est.mean():.4f}")
    # A badly wrong model is still unbiased, because the correction removes its error.
    junk = np.random.default_rng(5).normal(size=q.shape)
    est_junk = f(S, A, R, pi, b, mdp.gamma, junk)
    se_junk = est_junk.std(ddof=1) / np.sqrt(len(est_junk))
    if abs(est_junk.mean() - exact) > 4 * se_junk:
        fail(f"random-q estimate: the mean should be within 4 * {se_junk:.4f} of {exact:.4f}, got {est_junk.mean():.4f}")
    v_pd = pdis(S, A, R, pi, b, mdp.gamma).var()
    if not est.var() < 0.8 * v_pd:
        fail(f"variance with the exact q should be under 0.8 * {v_pd:.3f} (per-decision IS), got {est.var():.3f}")


@check("fqe recovers the exact policy value from a log that holds each transition in proportion to P")
def test_fqe(sub):
    f = has(sub, "fqe")
    # Three states, two actions, probabilities that are multiples of 0.25, so a log of
    # 4 * P[s, a, s'] copies of each transition has the model's exact frequencies.
    P = np.zeros((3, 2, 3))
    P[0, 0] = [0.5, 0.5, 0.0]
    P[0, 1] = [0.0, 0.25, 0.75]
    P[1, 0] = [0.25, 0.0, 0.75]
    P[1, 1] = [0.5, 0.25, 0.25]
    P[2, 0] = [0.0, 0.0, 1.0]
    P[2, 1] = [0.75, 0.25, 0.0]
    R = np.array([[1.0, 0.0], [0.5, -1.0], [2.0, 0.0]])
    mdp = rlenvs.TabularMDP(P=P, R=R, gamma=0.9, mu0=np.ones(3) / 3, terminal=np.zeros(3, dtype=bool))
    pi = np.array([[0.7, 0.3], [0.2, 0.8], [0.5, 0.5]])
    s, a, s2 = [], [], []
    for i in range(3):
        for j in range(2):
            for k in range(3):
                n = int(round(4 * P[i, j, k]))
                s += [i] * n
                a += [j] * n
                s2 += [k] * n
    s, a, s2 = np.array(s), np.array(a), np.array(s2)
    r = R[s, a]
    q = f(s, a, r, s2, pi, 0.9, 3, 2)
    shape(q, (3, 2), label="fqe output")
    want = rlenvs.q_from_v(mdp, rlenvs.policy_evaluation(mdp, pi))
    close(q, want, atol=1e-6, label="fqe against q_from_v(policy_evaluation)")
    # Unlogged pairs stay at zero.
    keep = ~((s == 2) & (a == 1))
    q2 = f(s[keep], a[keep], r[keep], s2[keep], pi, 0.9, 3, 2)
    almost(q2[2, 1], 0.0, 1e-12, label="Q for a pair the log never took")
