"""Checks for the watched-and-unwatched exercise."""

import itertools

import numpy as np

from rlcheck import almost, check, close, equal, fail, has, shape

A, H, LAM, PRIOR = 0.6, 1.0, 1.0, 0.5
EVAL_CUE, DEPLOY_CUE, NO_CUE = 0, 1, 2


def _joint(prior, q):
    """The joint of (observation, context), built here so the checks do not lean on the reader's."""
    return np.array([[q * prior, 0.0], [0.0, q * (1 - prior)], [(1 - q) * prior, (1 - q) * (1 - prior)]])


def _policy(sub, q, prior=PRIOR, a=A, h=H, lam=LAM):
    joint = _joint(prior, q)
    w = has(sub, "posterior_eval")(joint, prior)
    R = has(sub, "action_rewards")(w, a, h, lam)
    return joint, R


@check("joint_obs gives the (observation, context) table: 0.3, 0.3, 0.2, 0.2 at q = 0.6")
def test_joint_obs(sub):
    joint_obs = has(sub, "joint_obs")
    got = np.asarray(joint_obs(0.5, 0.6), dtype=float)
    shape(got, (3, 2), label="joint_obs output")
    close(got, np.array([[0.3, 0.0], [0.0, 0.3], [0.2, 0.2]]), atol=1e-12, label="q = 0.6")
    close(np.asarray(joint_obs(0.3, 0.0)), np.array([[0, 0], [0, 0], [0.3, 0.7]]), atol=1e-12,
          label="q = 0 leaves every rollout without a cue")
    close(np.asarray(joint_obs(0.3, 0.4)).sum(), 1.0, atol=1e-12, label="the table sums to 1")


@check("posterior_eval is 1 after an eval cue, 0 after a deploy cue, and the prior with no cue")
def test_posterior(sub):
    posterior_eval = has(sub, "posterior_eval")
    got = np.asarray(posterior_eval(_joint(0.3, 0.6), 0.3), dtype=float)
    close(got, np.array([1.0, 0.0, 0.3]), atol=1e-12, label="prior 0.3")
    got = np.asarray(posterior_eval(_joint(0.5, 1.0), 0.5), dtype=float)
    finite = np.all(np.isfinite(got))
    if not finite:
        fail(f"at q = 1 the no-cue row has probability 0, so use the prior there; got {got}")
    almost(got[NO_CUE], 0.5, 1e-12, label="no-cue posterior when that row is empty")


@check("action_rewards gives comply 0.6 and hack 0, 1, 0.5 for the eval cue, the deploy cue, and no cue")
def test_action_rewards(sub):
    action_rewards = has(sub, "action_rewards")
    got = np.asarray(action_rewards(np.array([1.0, 0.0, 0.5]), A, H, LAM), dtype=float)
    shape(got, (3, 2), label="action_rewards output")
    close(got, np.array([[0.6, 0.0], [0.6, 1.0], [0.6, 0.5]]), atol=1e-12)


@check("optimal_hack picks the best action per cue and matches a brute-force search over all 8 policies")
def test_optimal_hack(sub):
    optimal_hack = has(sub, "optimal_hack")
    joint, R = _policy(sub, 0.6)
    got = np.asarray(optimal_hack(R), dtype=float)
    equal(got, np.array([0.0, 1.0, 0.0]), label="hack indicator at q = 0.6")
    for prior, lam, q in [(0.5, 1.0, 0.6), (0.3, 1.0, 0.6), (0.5, 0.3, 0.6), (0.8, 2.0, 0.25), (0.5, 1.0, 0.0)]:
        joint, R = _policy(sub, q, prior=prior, lam=lam)
        got = np.asarray(optimal_hack(R), dtype=float)
        best, best_value = None, -1.0
        for acts in itertools.product([0.0, 1.0], repeat=3):
            # Value of the policy, computed from the joint without the reader's code.
            value = 0.0
            for o in range(3):
                p_eval, p_deploy = joint[o]
                comply_part = (1 - acts[o]) * A * (p_eval + p_deploy)
                hack_part = acts[o] * (H * (p_eval + p_deploy) - lam * p_eval)
                value += comply_part + hack_part
            if value > best_value + 1e-12:
                best, best_value = acts, value
        # Compare values, so a tie (a cue with probability 0) cannot fail the check.
        got_value = sum(
            (1 - got[o]) * A * joint[o].sum() + got[o] * (H * joint[o].sum() - lam * joint[o, 0]) for o in range(3)
        )
        if got_value < best_value - 1e-9:
            fail(f"prior {prior}, lam {lam}, q {q}: your policy {got} is worth {got_value:.4f}, "
                 f"but {best} is worth {best_value:.4f}")


@check("kl_hack gives 0.0474, 0.8808, 0.3775 at beta = 0.2 and maximizes reward minus beta * KL")
def test_kl_hack(sub):
    kl_hack = has(sub, "kl_hack")
    joint, R = _policy(sub, 0.6)
    got = np.asarray(kl_hack(R, 0.5, 0.2), dtype=float)
    shape(got, (3,), label="kl_hack output")
    close(got, np.array([0.04743, 0.88080, 0.37754]), atol=1e-4)
    # Brute force: scan P(hack) for one cue, with a reference of 0.3.
    ref, beta = 0.3, 0.2
    got = np.asarray(kl_hack(R, ref, beta), dtype=float)
    grid = np.linspace(1e-6, 1 - 1e-6, 200001)
    for o in range(3):
        obj = (grid * R[o, 1] + (1 - grid) * R[o, 0]
               - beta * (grid * np.log(grid / ref) + (1 - grid) * np.log((1 - grid) / (1 - ref))))
        close(got[o], grid[obj.argmax()], atol=1e-4, label=f"brute-force optimum for cue {o}")


@check("the compliance gap is 0 at q = 0, equals q * (deploy-cue hack - eval-cue hack), and grows with q")
def test_gap(sub):
    joint_obs, hack_rates = has(sub, "joint_obs"), has(sub, "hack_rates")
    gap_closed_form = has(sub, "gap_closed_form")
    optimal_hack, kl_hack = has(sub, "optimal_hack"), has(sub, "kl_hack")
    for name, make in [("optimal", lambda R: np.asarray(optimal_hack(R), dtype=float)),
                       ("KL", lambda R: np.asarray(kl_hack(R, 0.5, 0.2), dtype=float))]:
        gaps = []
        for q in np.linspace(0.0, 1.0, 11):
            joint, R = _policy(sub, q)
            pol = make(R)
            e, d = hack_rates(np.asarray(joint_obs(PRIOR, q)), pol)
            gaps.append(d - e)
            almost(gap_closed_form(q, pol), d - e, 1e-9, label=f"{name}: closed form at q = {q:.1f}")
        almost(gaps[0], 0.0, 1e-12, label=f"{name}: gap at q = 0")
        if np.any(np.diff(gaps) <= 0):
            fail(f"{name}: the gap should rise with q, got {np.round(gaps, 4).tolist()}")
    joint, R = _policy(sub, 0.6)
    e, d = hack_rates(_joint(PRIOR, 0.6), np.asarray(optimal_hack(R), dtype=float))
    almost(e, 0.0, 1e-12, label="eval hack rate at q = 0.6")
    almost(d, 0.6, 1e-12, label="deploy hack rate at q = 0.6")


@check("simulated rollouts match the gap formula within four standard errors (KL policy, q = 0.6)")
def test_gap_simulation(sub):
    kl_hack = has(sub, "kl_hack")
    joint, R = _policy(sub, 0.6)
    pol = np.asarray(kl_hack(R, 0.5, 0.2), dtype=float)
    want = has(sub, "gap_closed_form")(0.6, pol)
    rng = np.random.default_rng(0)
    n = 100_000
    context_eval = rng.random(n) < PRIOR
    revealed = rng.random(n) < 0.6
    obs = np.where(revealed, np.where(context_eval, EVAL_CUE, DEPLOY_CUE), NO_CUE)
    hacked = rng.random(n) < pol[obs]
    r_eval, r_deploy = hacked[context_eval], hacked[~context_eval]
    gap = r_deploy.mean() - r_eval.mean()
    se = np.sqrt(r_eval.var() / r_eval.size + r_deploy.var() / r_deploy.size)
    if abs(gap - want) > 4 * se:
        fail(f"simulated gap {gap:.4f} differs from the formula {want:.4f} by more than 4 SE = {4 * se:.4f}")


@check("bootstrap_diff_replicates redraws each group alone: variance matches var_a/n_a + var_b/n_b")
def test_bootstrap(sub):
    reps_fn, ci_fn = has(sub, "bootstrap_diff_replicates"), has(sub, "percentile_ci")
    a = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    reps = np.asarray(reps_fn(a, b, 20000, 0), dtype=float)
    shape(reps, (20000,), label="replicates")
    want = a.var() / a.size + b.var() / b.size
    if abs(reps.var() / want - 1) > 0.05:
        fail(f"replicate variance {reps.var():.5f}, expected {want:.5f} within 5%")
    almost(reps.mean(), a.mean() - b.mean(), 0.01, label="mean of replicates")
    equal(np.asarray(reps_fn(a, b, 50, 3)), np.asarray(reps_fn(a, b, 50, 3)), label="same seed")
    if np.array_equal(reps_fn(a, b, 50, 3), reps_fn(a, b, 50, 4)):
        fail("different seeds gave identical replicates")
    flat = np.asarray(reps_fn(np.ones(5), np.zeros(7), 100, 0))
    close(flat, np.ones(100), atol=1e-12, label="constant groups give a point")
    lo, hi = ci_fn(reps, 0.95)
    close([lo, hi], np.percentile(reps, [2.5, 97.5]), atol=1e-12, label="95% percentile interval")
    lo, hi = ci_fn(reps, 0.8)
    close([lo, hi], np.percentile(reps, [10, 90]), atol=1e-12, label="80% percentile interval")


@check("runs_needed gives 59 runs per condition for 0.30 vs 0.10, and simulation agrees; 12 runs have little power")
def test_runs_needed(sub):
    runs_needed = has(sub, "runs_needed")
    reps_fn, ci_fn = has(sub, "bootstrap_diff_replicates"), has(sub, "percentile_ci")
    equal(runs_needed(0.3, 0.1), 59, label="runs for 0.30 vs 0.10")
    equal(runs_needed(0.5, 0.2), 36, label="runs for 0.50 vs 0.20")
    if runs_needed(0.3, 0.2) <= runs_needed(0.3, 0.1):
        fail("a smaller difference needs more runs")
    rng = np.random.default_rng(0)

    def power(n, sims=800):
        wins = 0
        for i in range(sims):
            a = (rng.random(n) < 0.3).astype(float)
            b = (rng.random(n) < 0.1).astype(float)
            lo, _ = ci_fn(reps_fn(a, b, 400, i))
            wins += lo > 0
        return wins / sims

    p12, p59, p120 = power(12), power(runs_needed(0.3, 0.1)), power(120)
    if p12 > 0.35:
        fail(f"with 12 runs per condition the interval should exclude 0 rarely, got {p12:.2f}")
    if not 0.65 <= p59 <= 0.9:
        fail(f"with 59 runs the interval should exclude 0 about 80% of the time, got {p59:.2f}")
    if p120 < 0.9:
        fail(f"with 120 runs the interval should exclude 0 more than 90% of the time, got {p120:.2f}")


@check("sandbag_gap finds a 0.3 score drop under a capability-eval cue and no drop for an honest model")
def test_sandbag(sub):
    sandbag_gap = has(sub, "sandbag_gap")
    rng = np.random.default_rng(0)
    nocue = np.clip(rng.normal(0.8, 0.1, 40), 0, 1)
    cue_sandbag = np.clip(rng.normal(0.5, 0.1, 40), 0, 1)
    cue_honest = np.clip(rng.normal(0.8, 0.1, 40), 0, 1)
    gap, lo, hi = sandbag_gap(cue_sandbag, nocue)
    almost(gap, nocue.mean() - cue_sandbag.mean(), 1e-12, label="gap is no-cue mean minus cue mean")
    if not (lo > 0 and lo <= 0.3 <= hi):
        fail(f"sandbagger: interval ({lo:.3f}, {hi:.3f}) should exclude 0 and contain 0.3")
    gap, lo, hi = sandbag_gap(cue_honest, nocue)
    if not lo < 0 < hi:
        fail(f"honest model: interval ({lo:.3f}, {hi:.3f}) should contain 0")
