"""Checks for the monitors-in-the-reward exercise."""

import numpy as np

from rlcheck import almost, check, close, fail, finite_difference, has, shape

P, EPS, R_V, R_O = 0.3, 0.05, 0.95, 0.6
REF = np.array([0.55, 0.25, 0.18, 0.02])
BETA = 0.05
HONEST, ADMIT, VERBAL, OBFUS = 0, 1, 2, 3


def _softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()


@check("expected_rewards gives [0.3, 0, 1, 0.95] for pass only and [0.3, 0, 0.525, 0.65] at lam = 0.5")
def test_expected_rewards(sub):
    expected_rewards = has(sub, "expected_rewards")
    got = np.asarray(expected_rewards(P, EPS, R_V, R_O), dtype=float)
    shape(got, (4,), label="expected_rewards output")
    close(got, np.array([0.3, 0.0, 1.0, 0.95]), atol=1e-12, label="pass only")
    got = np.asarray(expected_rewards(P, EPS, R_V, R_O, lam=0.5), dtype=float)
    close(got, np.array([0.3, 0.0, 0.525, 0.65]), atol=1e-12, label="lam = 0.5")
    got = np.asarray(expected_rewards(P, EPS, R_V, R_O, q=0.2, c=5.0), dtype=float)
    close(got, np.array([0.3, 0.0, 0.0, -0.05]), atol=1e-12,
          label="an audit with q = 0.2 and c = 5 charges both hacks 1.0 and honest solving nothing")


@check("detection_rates splits hacks into flagged and undetected: the reference policy gives (0.183, 0.017)")
def test_detection_rates(sub):
    detection_rates = has(sub, "detection_rates")
    flagged, undetected = detection_rates(REF, R_V, R_O)
    almost(flagged, 0.183, 1e-12, label="flagged hacks under the reference")
    almost(undetected, 0.017, 1e-12, label="undetected hacks under the reference")
    pi = np.array([0.1, 0.2, 0.4, 0.3])
    flagged, undetected = detection_rates(pi, R_V, R_O)
    almost(flagged, 0.56, 1e-12, label="flagged hacks, second policy")
    almost(undetected, 0.14, 1e-12, label="undetected hacks, second policy")


@check("kl_optimum is the softmax of log(ref) + R / beta: 0.9607 verbal and 0.0393 obfus under pass only")
def test_kl_optimum_values(sub):
    kl_optimum = has(sub, "kl_optimum")
    rewards = np.array([0.3, 0.0, 1.0, 0.95])
    got = np.asarray(kl_optimum(REF, rewards, BETA), dtype=float)
    shape(got, (4,), label="kl_optimum output")
    close(got.sum(), 1.0, atol=1e-12, label="probabilities sum")
    # Relative to verbal: obfus / verbal = (0.02 / 0.18) * exp(-1) = 0.04088.
    ratio = 0.02 / 0.18 * np.exp(-1.0)
    want_v = 1.0 / (1.0 + ratio + 0.55 * np.exp(6.0 - 20.0) / 0.18 + 0.25 * np.exp(-20.0) / 0.18)
    close(got[VERBAL], want_v, atol=1e-12, label="verbal probability")
    close(got[VERBAL], 0.9607, atol=1e-4, label="verbal probability, rounded")
    close(got[OBFUS], 0.0393, atol=1e-4, label="obfus probability, rounded")
    big = np.asarray(kl_optimum(REF, rewards, 1e6), dtype=float)
    close(big, REF, atol=1e-5, label="a huge beta leaves the reference policy unchanged")


@check("kl_optimum maximizes E[reward] - beta * KL against 3000 random policies and reaches beta * log Z")
def test_kl_optimum_is_maximum(sub):
    kl_optimum = has(sub, "kl_optimum")
    rewards = np.array([0.3, 0.0, 0.525, 0.65])  # lam = 0.5

    def objective(pi):
        pi = np.clip(pi, 1e-300, None)
        return float(pi @ rewards - BETA * np.sum(pi * np.log(pi / REF)))

    star = np.asarray(kl_optimum(REF, rewards, BETA), dtype=float)
    best = objective(star)
    log_z = BETA * np.log(np.sum(REF * np.exp(rewards / BETA)))
    almost(best, log_z, 1e-9, label="J(pi*) should equal beta * log Z")
    rng = np.random.default_rng(0)
    for alpha in (0.2, 1.0, 5.0):
        for pi in rng.dirichlet(alpha * np.ones(4), size=1000):
            if objective(pi) > best + 1e-12:
                fail(f"a random policy {np.round(pi, 3)} scores {objective(pi):.6f}, "
                     f"above your optimum's {best:.6f}")
    for nudge in (np.array([0.005, 0.0, -0.005, 0.0]), np.array([0.0, 0.0, 0.01, -0.01]),
                  np.array([-0.005, 0.0, 0.0, 0.005])):
        if objective(star + nudge) > best + 1e-12:
            fail(f"nudging your optimum by {nudge} raises the objective, so it is not the maximum")


@check("lambda_star is 0.7368 against the verbalized hack and 1.0833 against the obfuscated one")
def test_lambda_star(sub):
    lambda_star = has(sub, "lambda_star")
    almost(lambda_star(P, 0.0, R_V), 0.7 / 0.95, 1e-12, label="lambda* against the verbalized hack")
    almost(lambda_star(P, EPS, R_O), 0.65 / 0.6, 1e-12, label="lambda* against the obfuscated hack")
    lam = lambda_star(P, EPS, R_O)
    almost(1.0 - EPS - lam * R_O, P, 1e-12, label="at lambda*, the hack earns exactly p")


@check("brute force over lambda: honest solving first wins at the larger lambda*, and obfuscation wins between the two")
def test_lambda_window(sub):
    expected_rewards = has(sub, "expected_rewards")
    lambda_star = has(sub, "lambda_star")
    lams = np.arange(0.0, 2.0, 0.001)
    first_honest = None
    for lam in lams:
        rew = np.asarray(expected_rewards(P, EPS, R_V, R_O, lam=lam), dtype=float)
        if rew.argmax() == HONEST:
            first_honest = lam
            break
    if first_honest is None:
        fail("honest solving never becomes the best action for lam below 2")
    want = max(lambda_star(P, 0.0, R_V), lambda_star(P, EPS, R_O))
    almost(first_honest, want, 0.0015, label="smallest lam at which honest solving is best")
    for lam in (0.8, 0.9, 1.0, 1.07):
        rew = np.asarray(expected_rewards(P, EPS, R_V, R_O, lam=lam), dtype=float)
        if rew.argmax() != OBFUS:
            fail(f"at lam = {lam} the obfuscated hack should be the best action, got {rew.argmax()}")


@check("audit_threshold is 0.14 against the verbalized hack and 0.13 against the obfuscated one at c = 5")
def test_audit_threshold(sub):
    audit_threshold = has(sub, "audit_threshold")
    expected_rewards = has(sub, "expected_rewards")
    almost(audit_threshold(P, 0.0, 5.0), 0.14, 1e-12, label="q* against the verbalized hack")
    almost(audit_threshold(P, EPS, 5.0), 0.13, 1e-12, label="q* against the obfuscated hack")
    almost(audit_threshold(P, 0.0, 10.0), 0.07, 1e-12, label="doubling c halves q*")
    qs = np.arange(0.0, 0.5, 0.0005)
    for q in qs:
        rew = np.asarray(expected_rewards(P, EPS, R_V, R_O, q=q, c=5.0), dtype=float)
        if rew.argmax() == HONEST:
            almost(q, 0.14, 0.001, label="smallest q at which honest solving is best (brute force)")
            return
    fail("honest solving is never the best action for q below 0.5")


@check("policy_gradient matches finite differences, sums to zero, and vanishes at a deterministic policy")
def test_policy_gradient(sub):
    policy_gradient = has(sub, "policy_gradient")
    rng = np.random.default_rng(0)
    rewards = np.array([0.3, 0.0, 0.525, 0.65])

    def J(theta):
        return float(_softmax(theta) @ rewards)

    for _ in range(5):
        theta = rng.normal(size=4)
        got = np.asarray(policy_gradient(theta.copy(), rewards), dtype=float)
        shape(got, (4,), label="policy_gradient output")
        close(got, finite_difference(J, theta), atol=1e-6, label="gradient vs finite differences")
        almost(float(got.sum()), 0.0, 1e-12, label="logit gradients sum to zero")
    theta = np.array([40.0, 0.0, 0.0, 0.0])
    close(np.asarray(policy_gradient(theta, rewards), dtype=float), np.zeros(4), atol=1e-9,
          label="gradient at a deterministic policy")


@check("training on pass only hacks in the open; training with a flag penalty moves the hack out of sight")
def test_train_obfuscation(sub):
    train = has(sub, "train")
    expected_rewards = has(sub, "expected_rewards")
    detection_rates = has(sub, "detection_rates")
    pi_a = np.asarray(train(REF, expected_rewards(P, EPS, R_V, R_O)), dtype=float)
    pi_b = np.asarray(train(REF, expected_rewards(P, EPS, R_V, R_O, lam=0.5)), dtype=float)
    shape(pi_a, (4,), label="train output")
    close(pi_a.sum(), 1.0, atol=1e-9, label="trained policy sums to one")
    if pi_a[VERBAL] < 0.95:
        fail(f"with pass-only reward the verbalized hack should dominate, got {np.round(pi_a, 3)}")
    if pi_b[OBFUS] < 0.95:
        fail(f"with lam = 0.5 the obfuscated hack should dominate, got {np.round(pi_b, 3)}")
    flag_a, miss_a = detection_rates(pi_a, R_V, R_O)
    flag_b, miss_b = detection_rates(pi_b, R_V, R_O)
    if not flag_b < flag_a:
        fail(f"flagged hacks should fall under the penalty: {flag_a:.3f} then {flag_b:.3f}")
    if not miss_b > 4 * miss_a:
        fail(f"undetected hacks should at least quadruple: {miss_a:.3f} then {miss_b:.3f}")
    recall_a, recall_b = flag_a / (flag_a + miss_a), flag_b / (flag_b + miss_b)
    almost(recall_a, 0.95, 0.01, label="monitor recall under pass only")
    almost(recall_b, 0.60, 0.01, label="monitor recall under the penalty")


@check("with random audits at q = 0.2, c = 5, training ends honest; at q = 0.1, below q*, the hack stays")
def test_train_audits(sub):
    train = has(sub, "train")
    expected_rewards = has(sub, "expected_rewards")
    above = np.asarray(train(REF, expected_rewards(P, EPS, R_V, R_O, q=0.2, c=5.0)), dtype=float)
    below = np.asarray(train(REF, expected_rewards(P, EPS, R_V, R_O, q=0.1, c=5.0)), dtype=float)
    if above[HONEST] < 0.9:
        fail(f"above q* honest solving should dominate, got {np.round(above, 3)}")
    if below[VERBAL] + below[OBFUS] < 0.9:
        fail(f"below q* the hacks should still dominate, got {np.round(below, 3)}")
