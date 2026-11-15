"""Checks for the safety-training-against-a-spec exercise."""

import math

import numpy as np

from rlcheck import almost, check, close, fail, has, shape

REFUSE, COMPLY = 0, 1

# Prompt rows are (harmful, looks fishy, encoded).
TRAIN = np.array([[1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
TRAIN_P = np.array([0.27, 0.03, 0.14, 0.56])
OOD = np.array([[1, 0, 1], [0, 0, 1]], dtype=float)
OOD_P = np.array([0.5, 0.5])
LN9 = math.log(9.0)


def _ref(n):
    return np.tile([0.2, 0.8], (n, 1))


def _rm_feats(X, action):
    if action == REFUSE:
        return np.stack([np.ones(len(X)), X[:, 1]], axis=1)
    return np.zeros((len(X), 2))


def _fit_both(sub):
    """Fit the examples-only and the spec weights with the reader's functions."""
    q = has(sub, "refuse_preference")(TRAIN, 0.1)
    rule_features = has(sub, "rule_features")
    fit_bt = has(sub, "fit_bt")
    theta_rm = fit_bt(_rm_feats(TRAIN, REFUSE) - _rm_feats(TRAIN, COMPLY), q, TRAIN_P)
    margin_spec = rule_features(TRAIN, REFUSE) - rule_features(TRAIN, COMPLY)
    theta_spec = fit_bt(margin_spec, q, TRAIN_P)
    return theta_rm, theta_spec


def _rewards(sub, theta_rm, theta_spec, X):
    reward_table = has(sub, "reward_table")
    rule_features = has(sub, "rule_features")
    R_rm = reward_table(theta_rm, _rm_feats(X, REFUSE), _rm_feats(X, COMPLY))
    R_spec = reward_table(theta_spec, rule_features(X, REFUSE), rule_features(X, COMPLY))
    return R_rm, R_spec


@check("refuse_preference is 1 - noise on harmful prompts and noise on benign ones")
def test_refuse_preference(sub):
    q = has(sub, "refuse_preference")(TRAIN, 0.1)
    shape(q, (4,))
    close(q, np.array([0.9, 0.9, 0.1, 0.1]), atol=1e-12)
    close(has(sub, "refuse_preference")(OOD, 0.25), np.array([0.75, 0.25]), atol=1e-12)


@check("rule_features counts complying with harm in column 0 and refusing the benign in column 1")
def test_rule_features(sub):
    rule_features = has(sub, "rule_features")
    refuse = rule_features(TRAIN, REFUSE)
    comply = rule_features(TRAIN, COMPLY)
    shape(refuse, (4, 2))
    # Rows: harmful-fishy, harmful-plain, benign-fishy, benign-plain.
    close(refuse, np.array([[0, 0], [0, 0], [0, 1], [0, 1]], dtype=float), atol=1e-12,
          label="features of refusing")
    close(comply, np.array([[1, 0], [1, 0], [0, 0], [0, 0]], dtype=float), atol=1e-12,
          label="features of complying")


@check("kl_optimal reweights the reference by exp(R / beta): 9/13 on the lesson's example")
def test_kl_optimal(sub):
    kl_optimal = has(sub, "kl_optimal")
    ref = _ref(2)
    # Harmful prompt: complying breaks a rule worth ln 9. Refusing costs nothing.
    R = np.array([[0.0, -LN9], [-LN9, 0.0]])
    pi = kl_optimal(ref, R, 1.0)
    shape(pi, (2, 2))
    # Refuse: 0.2 * 1 = 0.2. Comply: 0.8 / 9 = 0.0889. Z = 0.2889, so refuse = 9/13.
    close(pi[0], np.array([9 / 13, 4 / 13]), atol=1e-12, label="harmful row")
    # Benign: refuse = 0.2 / 9 = 0.0222, comply = 0.8. Z = 0.8222, so refuse = 1/37.
    close(pi[1], np.array([1 / 37, 36 / 37]), atol=1e-12, label="benign row")
    close(kl_optimal(ref, R, 1e9), ref, atol=1e-6, label="huge beta returns the reference")
    tiny = kl_optimal(ref, R, 1e-4)
    if not np.all(np.isfinite(tiny)):
        fail("a small beta overflowed; subtract the row maximum before exponentiating")
    close(tiny, np.array([[1.0, 0.0], [0.0, 1.0]]), atol=1e-9, label="tiny beta is greedy")


@check("error_rates gives the exact over- and under-refusal of a policy")
def test_error_rates(sub):
    error_rates = has(sub, "error_rates")
    pi = np.array([[0.5, 0.5], [0.2, 0.8], [0.4, 0.6], [0.1, 0.9]])
    over, under = error_rates(TRAIN, TRAIN_P, pi)
    # Benign mass 0.70: (0.14 * 0.4 + 0.56 * 0.1) / 0.70 = 0.16.
    almost(over, 0.16, 1e-12, label="over-refusal")
    # Harmful mass 0.30: (0.27 * 0.5 + 0.03 * 0.8) / 0.30 = 0.53.
    almost(under, 0.53, 1e-12, label="under-refusal")
    over0, under0 = error_rates(TRAIN, TRAIN_P, _ref(4))
    almost(over0, 0.2, 1e-12, label="reference over-refusal")
    almost(under0, 0.8, 1e-12, label="reference under-refusal")


@check("The examples-only reward model puts a positive weight on the spurious cue")
def test_rm_weights(sub):
    theta_rm, _ = _fit_both(sub)
    shape(theta_rm, (2,))
    # With annotator noise 0.1, P(prefer refuse | fishy) = 25.7 / 41 and
    # P(prefer refuse | plain) = 8.3 / 59. The fit matches both log-odds.
    plain = math.log(8.3 / 50.7)
    fishy = math.log(25.7 / 15.3)
    close(theta_rm, np.array([plain, fishy - plain]), atol=1e-6, label="(bias, fishy weight)")
    if not theta_rm[1] > 0:
        fail(f"the fishy weight should be positive, got {theta_rm[1]:.3f}")
    # Break the correlation: fishy independent of harm. The weight must vanish.
    p_indep = np.array([0.3 * 0.5, 0.3 * 0.5, 0.7 * 0.5, 0.7 * 0.5])
    q = has(sub, "refuse_preference")(TRAIN, 0.1)
    M = _rm_feats(TRAIN, REFUSE) - _rm_feats(TRAIN, COMPLY)
    theta_indep = has(sub, "fit_bt")(M, q, p_indep)
    almost(theta_indep[1], 0.0, 1e-6, label="fishy weight when the cue is uninformative")


@check("The spec reward's weights are fit from rule features and equal -ln 9 each")
def test_spec_weights(sub):
    _, theta_spec = _fit_both(sub)
    shape(theta_spec, (2,))
    close(theta_spec, np.array([-LN9, -LN9]), atol=1e-6, label="spec weights")
    R_spec = has(sub, "reward_table")(
        theta_spec, has(sub, "rule_features")(OOD, REFUSE), has(sub, "rule_features")(OOD, COMPLY))
    close(R_spec, np.array([[0.0, -LN9], [-LN9, 0.0]]), atol=1e-6, label="spec rewards on encoded prompts")


@check("frontier sweeps beta from the reference's rates to each reward's limit")
def test_frontier_limits(sub):
    frontier = has(sub, "frontier")
    theta_rm, theta_spec = _fit_both(sub)
    R_rm, R_spec = _rewards(sub, theta_rm, theta_spec, TRAIN)
    betas = np.array([1e6, 1.0, 1e-3])
    over, under = frontier(TRAIN, TRAIN_P, R_spec, _ref(4), betas)
    shape(over, (3,))
    # beta huge: the reference. beta tiny: the spec's rules are followed exactly.
    close([over[0], under[0]], [0.2, 0.8], atol=1e-4, label="spec at huge beta")
    close([over[2], under[2]], [0.0, 0.0], atol=1e-9, label="spec at tiny beta")
    # beta = 1: refuse harmful w.p. 9/13, refuse benign w.p. 1/37.
    close([over[1], under[1]], [1 / 37, 4 / 13], atol=1e-6, label="spec at beta 1")
    over, under = frontier(TRAIN, TRAIN_P, R_rm, _ref(4), betas)
    # The cue catches 90% of harm and flags 20% of benign prompts, so the
    # examples-only policy cannot do better than (0.2, 0.1).
    close([over[2], under[2]], [0.2, 0.1], atol=1e-6, label="examples-only at tiny beta")


@check("weakly_dominates holds when every point has a no-worse point, and fails otherwise")
def test_weakly_dominates(sub):
    dom = has(sub, "weakly_dominates")
    a = (np.array([0.0, 0.1]), np.array([0.5, 0.2]))
    b = (np.array([0.1, 0.3]), np.array([0.5, 0.6]))
    if not dom(a, b):
        fail("a has (0.0, 0.5) and (0.1, 0.2), which are no worse than every point of b")
    if dom(b, a):
        fail("b does not reach a's point (0.0, 0.5): its over-refusal never gets below 0.1")
    if not dom(a, a):
        fail("a frontier weakly dominates itself")


@check("The spec frontier weakly dominates on training and on encoded prompts; the examples-only one does not")
def test_headline(sub):
    frontier = has(sub, "frontier")
    dom = has(sub, "weakly_dominates")
    theta_rm, theta_spec = _fit_both(sub)
    betas = np.geomspace(0.05, 20.0, 40)
    fronts = {}
    for name, X, p in [("train", TRAIN, TRAIN_P), ("encoded", OOD, OOD_P)]:
        R_rm, R_spec = _rewards(sub, theta_rm, theta_spec, X)
        fronts[name] = (frontier(X, p, R_rm, _ref(len(X)), betas),
                        frontier(X, p, R_spec, _ref(len(X)), betas))
        rm_front, spec_front = fronts[name]
        if not dom(spec_front, rm_front):
            fail(f"the spec frontier should weakly dominate the examples-only one on {name} prompts")
    # Encoded prompts are never fishy, so the learned reward refuses less than the base model.
    ood_rm, ood_spec = fronts["encoded"]
    if not dom(ood_spec, ood_rm) or dom(ood_rm, ood_spec):
        fail("on encoded prompts only the spec frontier should dominate")
    # At beta = 1 the examples-only policy refuses an encoded harmful prompt
    # 0.2 * e^-1.81 / (0.8 + 0.2 * e^-1.81) = 3.93% of the time, against the
    # base model's 20%, so it complies 96.07% of the time.
    R_rm_ood = _rewards(sub, theta_rm, theta_spec, OOD)[0]
    _, under = frontier(OOD, OOD_P, R_rm_ood, _ref(2), np.array([1.0]))
    almost(under[0], 0.96068214, 1e-6, label="examples-only under-refusal on encoded prompts at beta 1")
    if not under[0] > 0.8:
        fail("the examples-only policy should under-refuse encoded harm more than the base model's 0.8")
