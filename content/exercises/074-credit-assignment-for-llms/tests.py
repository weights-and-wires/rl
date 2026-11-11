"""Checks for the credit-assignment exercise."""

import itertools

import numpy as np

from rlcheck import almost, check, close, equal, fail, has, shape

P = np.array([0.9, 0.8, 0.7, 0.6])
TT = len(P)


def _brute_value(prefix):
    """Value of a prefix by summing over every completion, no shortcut."""
    k = len(prefix)
    total = 0.0
    for rest in itertools.product([0, 1], repeat=TT - k):
        prob = np.prod([P[k + j] if o else 1 - P[k + j] for j, o in enumerate(rest)])
        if all(prefix) and all(rest):
            total += prob
    return total


@check("exact_value is 0.3024 at the root and 0 after any wrong step")
def test_exact_value(sub):
    exact_value = has(sub, "exact_value")
    almost(exact_value(np.array([], dtype=int), P), 0.9 * 0.8 * 0.7 * 0.6, 1e-12, label="root")
    almost(exact_value(np.array([1, 1]), P), 0.7 * 0.6, 1e-12, label="after two correct steps")
    almost(exact_value(np.array([1, 0]), P), 0.0, 1e-12, label="after a wrong step")
    almost(exact_value(np.array([1, 1, 1, 1]), P), 1.0, 1e-12, label="a finished correct response")
    for prefix in ([], [1], [1, 1, 1], [0, 1]):
        almost(exact_value(np.array(prefix, dtype=int), P), _brute_value(prefix), 1e-12,
               label=f"prefix {prefix} against brute force")


@check("mc_value is unbiased: the mean over 400 seeds matches the exact value")
def test_mc_unbiased(sub):
    mc_value = has(sub, "mc_value")
    K = 50
    for prefix, want in ((np.array([1]), 0.336), (np.array([1, 1]), 0.42)):
        ests = np.array([mc_value(prefix, K, np.random.default_rng(s), P)[0] for s in range(400)])
        se_of_mean = np.sqrt(want * (1 - want) / K / 400)
        if abs(ests.mean() - want) > 4 * se_of_mean:
            fail(f"prefix {prefix}: mean estimate {ests.mean():.4f}, exact {want}, "
                 f"allowed gap {4 * se_of_mean:.4f}")


@check("mc_value's standard error matches sqrt(v(1-v)/K) and covers the exact value")
def test_mc_stderr(sub):
    mc_value = has(sub, "mc_value")
    K, v = 100, 0.42
    out = [mc_value(np.array([1, 1]), K, np.random.default_rng(s), P) for s in range(400)]
    est = np.array([o[0] for o in out])
    se = np.array([o[1] for o in out])
    want = np.sqrt(v * (1 - v) / K)
    almost(se.mean(), want, 0.1 * want, label="mean reported standard error")
    almost(est.std(), want, 0.15 * want, label="spread of the estimates across seeds")
    cover = np.mean(np.abs(est - v) <= 2 * se)
    if not 0.90 <= cover <= 0.99:
        fail(f"a 2-standard-error interval should cover the exact value about 95% of the time, got {cover:.3f}")
    est0, se0 = mc_value(np.array([1, 0]), 20, np.random.default_rng(0), P)
    almost(est0, 0.0, 0.0, label="a prefix with a wrong step is worth 0")


@check("vine_advantages sum telescopically to R minus the root estimate")
def test_vine_telescopes(sub):
    vine = has(sub, "vine_advantages")
    for s, outcomes in enumerate(([1, 1, 1, 1], [1, 1, 0, 1], [0, 1, 1, 1], [1, 1, 1, 0])):
        adv, v_hat = vine(np.array(outcomes), 32, np.random.default_rng(s), P)
        shape(adv, (TT,), label="advantages")
        shape(v_hat, (TT + 1,), label="v_hat")
        almost(v_hat[-1], float(all(outcomes)), 0.0, label=f"v_hat[T] for {outcomes} is the reward")
        almost(adv.sum(), float(all(outcomes)) - v_hat[0], 1e-12, label=f"sum for {outcomes}")
        close(adv, np.diff(v_hat), atol=1e-12, label="advantages are v_hat differences")


@check("vine_advantages recover the exact step advantages as K grows, within the standard error")
def test_vine_converges(sub):
    # Wrong implementation caught here: advantage = V(prefix) - V(next), a sign flip.
    vine = has(sub, "vine_advantages")
    outcomes = np.array([1, 1, 0, 1])
    exact_v = np.array([_brute_value(outcomes[:k]) for k in range(TT + 1)])
    exact_adv = np.diff(exact_v)
    # exact_adv is [0.0336, 0.084, -0.42, 0]: positive for a correct step, negative for a wrong one.
    errs = {}
    for K in (16, 256, 4096):
        runs = np.array([vine(outcomes, K, np.random.default_rng(s), P)[0] for s in range(30)])
        errs[K] = np.abs(runs.mean(axis=0) - exact_adv).max()
        # Each advantage is a difference of two estimates, each with variance
        # at most 0.25 / K, and the 30 runs are averaged: 5 standard errors.
        tol = 5 * np.sqrt(0.5 / K / 30)
        if errs[K] > tol:
            fail(f"K={K}: mean advantages {np.round(runs.mean(axis=0), 4)}, exact "
                 f"{np.round(exact_adv, 4)}, allowed gap {tol:.4f} per step")
    if not errs[4096] < errs[16] / 3:
        fail(f"error should shrink as K grows, got {errs[16]:.4f} at K=16 and {errs[4096]:.4f} at K=4096")
    adv, _ = vine(outcomes, 4096, np.random.default_rng(1), P)
    if not (adv[0] > 0 and adv[1] > 0 and adv[2] < -0.3):
        fail(f"a correct step must score above 0 and the wrong step far below; got {np.round(adv, 3)}")


@check("math_shepherd_labels: hard is 1 if any rollout is correct, soft is the fraction")
def test_math_shepherd(sub):
    # Wrong implementation caught here: a soft label rounded to 0 or 1.
    labels = has(sub, "math_shepherd_labels")
    rollouts = np.array([[0, 0, 0, 0],
                         [1, 0, 0, 0],
                         [1, 1, 0, 1],
                         [1, 1, 1, 1]])
    close(labels(rollouts, "hard"), np.array([0.0, 1.0, 1.0, 1.0]), label="hard labels")
    close(labels(rollouts, "soft"), np.array([0.0, 0.25, 0.75, 1.0]), label="soft labels")


@check("hard labels average 1-(1-v)^K and soft labels average v, exactly over all rollout sets")
def test_label_expectations(sub):
    labels = has(sub, "math_shepherd_labels")
    K = 4
    for v in (0.3024, 0.42, 0.6):
        hard_mean = soft_mean = 0.0
        for pattern in itertools.product([0, 1], repeat=K):
            prob = np.prod([v if o else 1 - v for o in pattern])
            row = np.array([pattern])
            hard_mean += prob * labels(row, "hard")[0]
            soft_mean += prob * labels(row, "soft")[0]
        almost(soft_mean, v, 1e-12, label=f"soft label mean at v={v}")
        almost(hard_mean, 1 - (1 - v) ** K, 1e-12, label=f"hard label mean at v={v}")


@check("implicit_prm_rewards sums log ratios per step and the steps sum to the sequence beta log ratio")
def test_implicit_prm(sub):
    implicit = has(sub, "implicit_prm_rewards")
    logp = np.array([-1.0, -0.5, -2.0, -0.1, -0.3])
    ref = np.array([-1.5, -0.5, -1.0, -0.4, -0.8])
    steps = np.array([0, 0, 1, 2, 2])
    got = implicit(logp, ref, steps, 0.5, 3)
    # Log ratios 0.5, 0, -1, 0.3, 0.5 summed per step: 0.5, -1, 0.8, times beta 0.5.
    close(got, np.array([0.25, -0.5, 0.4]), atol=1e-12, label="step rewards")
    rng = np.random.default_rng(0)
    for beta in (0.1, 1.0):
        lp = rng.normal(-1, 0.5, size=40)
        lr = rng.normal(-1, 0.5, size=40)
        ids = np.sort(rng.integers(0, 6, size=40))
        r = implicit(lp, lr, ids, beta, 6)
        shape(r, (6,))
        almost(r.sum(), beta * (lp.sum() - lr.sum()), 1e-10, label=f"sum of step rewards, beta {beta}")
    empty = implicit(logp, ref, np.array([0, 0, 0, 0, 0]), 1.0, 3)
    close(empty, np.array([(logp - ref).sum(), 0.0, 0.0]), atol=1e-12,
          label="steps with no tokens have reward 0")


@check("per-step vine credit has far lower squared error than the uniform advantage")
def test_credit_comparison(sub):
    errors = has(sub, "credit_errors")
    uniform, vine = errors(P, 16, 1500, 0)
    shape(uniform, (TT,))
    shape(vine, (TT,))
    if not np.all(vine < 0.25 * uniform):
        fail(f"vine (K=16) should be under a quarter of the uniform error at every step; "
             f"uniform {np.round(uniform, 3)}, vine {np.round(vine, 3)}")
    # Averaged over steps the uniform error is about 0.16, and vine's is about 0.02.
    almost(uniform.mean(), 0.16, 0.03, label="mean uniform error")
    big = errors(P, 256, 400, 0)[1]
    if not big.mean() < vine.mean() / 4:
        fail(f"more rollouts should cut the vine error; K=16 gave {vine.mean():.4f}, "
             f"K=256 gave {big.mean():.4f}")
