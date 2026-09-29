"""Checks for the speech and audio exercise."""

import numpy as np
import torch

from rlcheck import almost, check, close, equal, fail, finite_difference, has, shape

torch.set_num_threads(1)

WORKED_LOGSCORES = np.array([2.0, 1.0, 0.5, -1.0])
WORKED_ERRORS = np.array([0.0, 1.0, 2.0, 3.0])


def _p_hat(z):
    p = np.exp(z - z.max())
    return p / p.sum()


@check("edit_distance counts substitutions, insertions, and deletions, and is symmetric")
def test_edit_distance(sub):
    edit_distance = has(sub, "edit_distance")
    cases = [
        ("the cat sat".split(), "the cat sit".split(), 1),
        ("the cat sat".split(), "the cat sat".split(), 0),
        ("the cat sat".split(), "the sat".split(), 1),
        ("the cat sat".split(), "the big cat sat down".split(), 2),
        ([], "a b c".split(), 3),
        ("a b c".split(), [], 3),
        ([], [], 0),
        (list("kitten"), list("sitting"), 3),
        ("a b c d".split(), "b c d a".split(), 2),
    ]
    for ref, hyp, want in cases:
        got = edit_distance(ref, hyp)
        if got != want:
            fail(f"edit_distance({ref}, {hyp}) should be {want}, got {got}")
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = list(rng.choice(list("abcd"), size=rng.integers(0, 7)))
        b = list(rng.choice(list("abcd"), size=rng.integers(0, 7)))
        if edit_distance(a, b) != edit_distance(b, a):
            fail(f"edit_distance should be symmetric, but differs for {a} and {b}")


@check("wer divides word edits by the reference length: 1/3 for 'the cat sit', and above 1 with insertions")
def test_wer(sub):
    wer = has(sub, "wer")
    almost(wer("the cat sat", "the cat sit"), 1 / 3, 1e-12, label="'the cat sat' against 'the cat sit'")
    almost(wer("the cat sat", "the cat sat"), 0.0, 1e-12, label="a perfect transcript")
    almost(wer("turn it off", "turn it of"), 1 / 3, 1e-12, label="one misspelt word counts as one word error")
    almost(wer("a b", "x y z w"), 2.0, 1e-12, label="two reference words, four wrong hypothesis words")


@check("mwer_loss gives -0.9128 on the worked example, the expected errors minus the list's mean errors")
def test_mwer_loss(sub):
    mwer_loss = has(sub, "mwer_loss")
    almost(mwer_loss(WORKED_LOGSCORES, WORKED_ERRORS), 0.5872153 - 1.5, 1e-6, label="worked example")
    rng = np.random.default_rng(1)
    z = rng.normal(size=6)
    W = rng.integers(0, 5, size=6).astype(float)
    p = _p_hat(z)
    almost(mwer_loss(z, W), np.sum(p * W) - W.mean(), 1e-10, label="random 6-best list")


@check("mwer_grad is P_hat * (W - expected W): it matches the worked example and finite differences of mwer_loss")
def test_mwer_grad(sub):
    mwer_loss = has(sub, "mwer_loss")
    mwer_grad = has(sub, "mwer_grad")
    got = np.asarray(mwer_grad(WORKED_LOGSCORES, WORKED_ERRORS))
    shape(got, (4,), label="gradient for four hypotheses")
    close(got, np.array([-0.3578843, 0.0925496, 0.1921231, 0.0732117]), atol=1e-6, label="worked example")
    almost(got.sum(), 0.0, 1e-12, label="sum of the gradient over the list")
    rng = np.random.default_rng(2)
    for _ in range(3):
        z = rng.normal(size=5) * 2
        W = rng.integers(0, 6, size=5).astype(float)
        fd = finite_difference(lambda x: mwer_loss(x, W), z.copy())
        close(np.asarray(mwer_grad(z, W)), fd, atol=1e-8, rtol=1e-6, label="mwer_grad against finite differences")


@check("mwer_grad ignores a constant added to every log-score and a constant added to every error count")
def test_mwer_invariances(sub):
    mwer_grad = has(sub, "mwer_grad")
    rng = np.random.default_rng(3)
    z = rng.normal(size=5)
    W = rng.integers(0, 4, size=5).astype(float)
    base = np.asarray(mwer_grad(z, W))
    close(np.asarray(mwer_grad(z + 7.5, W)), base, atol=1e-12, label="log-scores shifted by 7.5")
    close(np.asarray(mwer_grad(z, W + 3.0)), base, atol=1e-12, label="every hypothesis given 3 more errors")
    # Large log-scores must not overflow.
    big = np.asarray(mwer_grad(z * 400.0, W))
    if not np.all(np.isfinite(big)):
        fail(f"mwer_grad overflowed for log-scores in the hundreds: {big}")


@check("reinforce_mwer_samples is unbiased with any constant baseline, and the baseline E[W] cuts its variance")
def test_reinforce_samples(sub):
    mwer_grad = has(sub, "mwer_grad")
    samples = has(sub, "reinforce_mwer_samples")
    exact = np.asarray(mwer_grad(WORKED_LOGSCORES, WORKED_ERRORS))
    expected_w = float(np.sum(_p_hat(WORKED_LOGSCORES) * WORKED_ERRORS))
    with_b = np.asarray(samples(WORKED_LOGSCORES, WORKED_ERRORS, 50_000, expected_w, 0))
    without_b = np.asarray(samples(WORKED_LOGSCORES, WORKED_ERRORS, 50_000, 0.0, 0))
    shape(with_b, (50_000, 4), label="one gradient estimate per sample")
    close(with_b.mean(0), exact, atol=0.01, label="mean estimate with baseline E[W]")
    close(without_b.mean(0), exact, atol=0.015, label="mean estimate with no baseline")
    var_with = with_b.var(0).sum()
    var_without = without_b.var(0).sum()
    if var_without < 1.8 * var_with:
        fail(
            f"the baseline should cut the total variance by at least 1.8x on this example "
            f"(exactly 2.26x): {var_without:.4f} without, {var_with:.4f} with"
        )


@check("tts_reward adds intelligibility, 1 - WER, to a weighted naturalness score, UTMOS / 5")
def test_tts_reward(sub):
    tts_reward = has(sub, "tts_reward")
    rates = has(sub, "SPEAKING_RATES")
    wer_fn, mos_fn = has(sub, "asr_wer"), has(sub, "utmos")
    got = np.asarray(tts_reward(rates, 1.0, 0.5))
    shape(got, (11,), label="one reward per speaking rate")
    close(got, (1 - wer_fn(rates)) + 0.5 * mos_fn(rates) / 5, atol=1e-12, label="reward on the rate grid")
    almost(float(tts_reward(1.0, 1.0, 1.0)), 1 - 0.02 - 0.3 / (1 + np.exp(0.8)) + 0.9, 1e-12,
           label="reward at normal speed")


@check("kl_optimal_policy is softmax(ref + R / beta): the reference for huge beta and the argmax for tiny beta")
def test_kl_optimal_policy(sub):
    kl_optimal_policy = has(sub, "kl_optimal_policy")
    rng = np.random.default_rng(4)
    ref = np.log(rng.dirichlet(np.ones(6)))
    R = rng.normal(size=6)
    pi = np.asarray(kl_optimal_policy(ref, R, 0.5))
    want = np.exp(ref) * np.exp(R / 0.5)
    close(pi, want / want.sum(), atol=1e-12, label="closed form at beta = 0.5")
    close(np.asarray(kl_optimal_policy(ref, R, 1e6)), np.exp(ref), atol=1e-5, label="beta = 1e6")
    onehot = np.eye(6)[np.argmax(R)]
    close(np.asarray(kl_optimal_policy(ref, R, 1e-4)), onehot, atol=1e-6, label="beta = 1e-4")


@check("A WER-only reward drives speech slow and unnatural; adding UTMOS puts the optimum at normal speed")
def test_tts_reward_hacking(sub):
    kl_optimal_policy = has(sub, "kl_optimal_policy")
    tts_reward = has(sub, "tts_reward")
    rates, ref_fn = has(sub, "SPEAKING_RATES"), has(sub, "reference_logprobs")
    mos_fn = has(sub, "utmos")
    ref = ref_fn()
    ref_mos = float(np.sum(np.exp(ref) * mos_fn(rates)))
    hacked = np.asarray(kl_optimal_policy(ref, tts_reward(rates, 1.0, 0.0), 0.01))
    mean_rate = float(np.sum(hacked * rates))
    hacked_mos = float(np.sum(hacked * mos_fn(rates)))
    if mean_rate > 0.7:
        fail(f"with a WER-only reward the optimal mean speaking rate should fall below 0.7, got {mean_rate:.3f}")
    if hacked_mos > ref_mos - 1.0:
        fail(f"the WER-only optimum should lose over a point of naturalness against the reference "
             f"({ref_mos:.3f}), got {hacked_mos:.3f}")
    fixed = np.asarray(kl_optimal_policy(ref, tts_reward(rates, 1.0, 1.0), 0.01))
    almost(float(np.sum(fixed * rates)), 1.0, 0.01, label="mean speaking rate with the combined reward")


@check("dpo_population_loss has zero gradient at pi_ref * exp(R / beta), and minimising it lands there")
def test_dpo_population(sub):
    dpo_loss = has(sub, "dpo_population_loss")
    kl_optimal_policy = has(sub, "kl_optimal_policy")
    tts_reward = has(sub, "tts_reward")
    rates, ref_fn = has(sub, "SPEAKING_RATES"), has(sub, "reference_logprobs")
    ref = ref_fn()
    beta = 0.01
    for w_nat in (0.0, 1.0):
        R = tts_reward(rates, 1.0, w_nat)
        target = kl_optimal_policy(ref, R, beta)
        at_opt = torch.tensor(np.log(np.clip(target, 1e-300, None)), dtype=torch.float64, requires_grad=True)
        loss_opt = dpo_loss(at_opt, ref, R, beta)
        loss_opt.backward()
        g = at_opt.grad.numpy()
        if np.max(np.abs(g * target)) > 1e-9 or not np.all(np.isfinite(g)):
            fail(f"at the closed-form optimum the DPO gradient should vanish, got {g}")
        at_ref = torch.tensor(ref, dtype=torch.float64)
        if float(dpo_loss(at_ref, ref, R, beta)) <= float(loss_opt) + 1e-6:
            fail("the population DPO loss should be higher at the reference than at the KL optimum")

        logits = torch.tensor(ref, dtype=torch.float64).clone().requires_grad_(True)
        opt = torch.optim.LBFGS([logits], lr=1, max_iter=500, tolerance_grad=1e-12,
                                tolerance_change=1e-15, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            value = dpo_loss(logits, ref, R, beta)
            value.backward()
            return value

        opt.step(closure)
        got = torch.softmax(logits, -1).detach().numpy()
        close(got, target, atol=1e-3, label=f"DPO fit against the closed form (w_nat = {w_nat})")
