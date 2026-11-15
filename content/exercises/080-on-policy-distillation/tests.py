"""Checks for the on-policy distillation exercise."""

import numpy as np
import torch

from rlcheck import almost, check, close, fail, finite_difference, has, shape

torch.set_num_threads(1)
D = torch.float64


def _kl(p, q):
    """KL(p || q) for probability vectors, in numpy."""
    return float(np.sum(p * np.log(p / q)))


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


@check("token_kl_losses matches the closed forms on one token: forward 0.2263, reverse 0.3112")
def test_hand_example(sub):
    token_kl_losses = has(sub, "token_kl_losses")
    p_t = np.array([0.9, 0.1])
    p_s = np.array([0.6, 0.4])
    s = torch.tensor(np.log(p_s), dtype=D).reshape(1, 1, 2)
    t = torch.tensor(np.log(p_t), dtype=D).reshape(1, 1, 2)
    fwd, rev = token_kl_losses(s, t, torch.ones(1, 1, dtype=D))
    # Forward KL(pi_T || pi_S) = 0.9 ln(0.9/0.6) + 0.1 ln(0.1/0.4) = 0.2263.
    almost(float(fwd), _kl(p_t, p_s), 1e-9, label="forward KL(pi_T || pi_S)")
    # Reverse KL(pi_S || pi_T) = 0.6 ln(0.6/0.9) + 0.4 ln(0.4/0.1) = 0.3112.
    almost(float(rev), _kl(p_s, p_t), 1e-9, label="reverse KL(pi_S || pi_T)")


@check("the two KLs are zero for equal logits, ignore a logit shift, and differ for unequal ones")
def test_kl_invariants(sub):
    token_kl_losses = has(sub, "token_kl_losses")
    rng = np.random.default_rng(1)
    z = torch.tensor(rng.normal(size=(2, 3, 5)), dtype=D)
    ones = torch.ones(2, 3, dtype=D)
    fwd, rev = token_kl_losses(z, z + 7.0, ones)
    almost(float(fwd), 0.0, 1e-9, label="forward KL of equal distributions")
    almost(float(rev), 0.0, 1e-9, label="reverse KL of equal distributions")
    w = torch.tensor(rng.normal(size=(2, 3, 5)), dtype=D)
    fwd, rev = token_kl_losses(z, w, ones)
    if float(fwd) <= 0 or float(rev) <= 0 or abs(float(fwd) - float(rev)) < 1e-3:
        fail(f"for different distributions expected two positive, different KLs, got forward={float(fwd):.5f}, reverse={float(rev):.5f}")


@check("token_kl_losses averages over real tokens only, so padding changes nothing")
def test_mask(sub):
    # Named wrong implementation: dividing by B * T instead of mask.sum(), or
    # leaving padded positions in the average.
    token_kl_losses = has(sub, "token_kl_losses")
    rng = np.random.default_rng(2)
    s = torch.tensor(rng.normal(size=(2, 4, 6)), dtype=D)
    t = torch.tensor(rng.normal(size=(2, 4, 6)), dtype=D)
    mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]], dtype=D)
    s_bad, t_bad = s.clone(), t.clone()
    s_bad[mask == 0] = 50.0 * torch.tensor(rng.normal(size=6), dtype=D)   # garbage on padding
    t_bad[mask == 0] = -30.0
    real = [(0, 0), (0, 1), (0, 2), (1, 0)]
    want_f = np.mean([_kl(_softmax(t[b, i].numpy()), _softmax(s[b, i].numpy())) for b, i in real])
    want_r = np.mean([_kl(_softmax(s[b, i].numpy()), _softmax(t[b, i].numpy())) for b, i in real])
    fwd, rev = token_kl_losses(s_bad, t_bad, mask)
    almost(float(fwd), want_f, 1e-9, label="forward KL averaged over the 4 real tokens")
    almost(float(rev), want_r, 1e-9, label="reverse KL averaged over the 4 real tokens")


@check("the mean of 20000 sampled reverse-KL estimates matches the exact value within 4 standard errors")
def test_sampled_estimator(sub):
    sampled_reverse_kl = has(sub, "sampled_reverse_kl")
    rng = np.random.default_rng(3)
    s_logits = rng.normal(size=5)
    t_logits = rng.normal(size=5)
    p_s = _softmax(s_logits)
    exact = _kl(p_s, _softmax(t_logits))
    n = 20000
    tokens = rng.choice(5, size=n, p=p_s)
    est = sampled_reverse_kl(
        torch.tensor(s_logits, dtype=D).expand(n, 5),
        torch.tensor(t_logits, dtype=D).expand(n, 5),
        torch.tensor(tokens),
    ).numpy()
    shape(est, (n,), label="one estimate per sample")
    se = est.std(ddof=1) / np.sqrt(n)
    if abs(est.mean() - exact) > 4 * se:
        fail(f"mean of {n} estimates is {est.mean():.5f}, exact reverse KL is {exact:.5f}, "
             f"standard error {se:.5f}")


@check("distill_reward is log pi_T - log pi_S at the sampled token, and its mean is minus the reverse KL")
def test_reward(sub):
    distill_reward = has(sub, "distill_reward")
    p_t = np.array([0.9, 0.1])
    p_s = np.array([0.6, 0.4])
    s = torch.tensor(np.log(p_s), dtype=D).expand(2, 2)
    t = torch.tensor(np.log(p_t), dtype=D).expand(2, 2)
    r = distill_reward(s, t, torch.tensor([0, 1])).numpy()
    # Token 0: ln(0.9 / 0.6) = 0.4055. Token 1: ln(0.1 / 0.4) = -1.3863.
    close(r, np.log(p_t / p_s), atol=1e-9, label="reward at tokens 0 and 1")
    # Averaged under the student's own distribution, the reward is -KL(pi_S || pi_T).
    almost(float(p_s @ r), -_kl(p_s, p_t), 1e-9, label="student-weighted mean reward")


@check("on_policy_distill_loss has the gradient -reward * (onehot - pi_S) / n on real tokens and none on padding")
def test_loss_gradient_closed_form(sub):
    on_policy_distill_loss = has(sub, "on_policy_distill_loss")
    rng = np.random.default_rng(4)
    s = torch.tensor(rng.normal(size=(2, 3, 4)), dtype=D, requires_grad=True)
    t = torch.tensor(rng.normal(size=(2, 3, 4)), dtype=D)
    tokens = torch.tensor(rng.integers(0, 4, size=(2, 3)))
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=D)
    loss = on_policy_distill_loss(s, t, tokens, mask)
    loss.backward()
    p_s = torch.softmax(s.detach(), -1).numpy()
    p_t = torch.softmax(t, -1).numpy()
    want = np.zeros((2, 3, 4))
    for b in range(2):
        for i in range(3):
            if mask[b, i] == 0:
                continue
            k = int(tokens[b, i])
            reward = np.log(p_t[b, i, k]) - np.log(p_s[b, i, k])
            onehot = np.eye(4)[k]
            want[b, i] = -reward * (onehot - p_s[b, i]) / 5.0     # 5 real tokens
    # A reward that carries its own gradient adds a second term here and fails.
    close(s.grad.numpy(), want, atol=1e-9, label="gradient of the loss with respect to the student logits")


@check("on_policy_distill_loss's gradient averaged over 20000 samples matches the finite-difference gradient of the exact KL")
def test_expected_gradient(sub):
    on_policy_distill_loss = has(sub, "on_policy_distill_loss")
    rng = np.random.default_rng(5)
    theta = rng.normal(size=4)
    t_logits = rng.normal(size=4)
    p_t = _softmax(t_logits)

    def exact_kl(z):
        return _kl(_softmax(z), p_t)

    want = finite_difference(exact_kl, theta)
    n = 20000
    tokens = torch.tensor(rng.choice(4, size=(n, 1), p=_softmax(theta)))
    th = torch.tensor(theta, dtype=D, requires_grad=True)
    loss = on_policy_distill_loss(
        th.expand(n, 1, 4), torch.tensor(t_logits, dtype=D).expand(n, 1, 4), tokens, torch.ones(n, 1, dtype=D)
    )
    loss.backward()
    got = th.grad.numpy()
    if np.abs(got - want).max() > 0.02:
        fail(f"expected the gradient of the exact reverse KL {np.round(want, 4)}, got the averaged surrogate gradient {np.round(got, 4)}")


@check("a bimodal teacher: the best reverse-KL student sits on one mode, the best forward-KL student spreads over both")
def test_bimodal(sub):
    fit_unimodal_student = has(sub, "fit_unimodal_student")
    grid = np.linspace(-9, 9, 73)

    def bump(m, s):
        d = np.exp(-((grid - m) ** 2) / (2 * s * s))
        return d / d.sum()

    p = 0.5 * bump(-3, 0.7) + 0.5 * bump(3, 0.7)
    mu_r, sig_r = fit_unimodal_student(np.log(p), grid, "reverse")
    mu_f, sig_f = fit_unimodal_student(np.log(p), grid, "forward")
    if abs(abs(mu_r) - 3.0) > 0.3 or sig_r > 1.2:
        fail(f"reverse KL: expected |mu| near 3 and sigma below 1.2 (one mode), got mu={mu_r:.2f}, sigma={sig_r:.2f}")
    if abs(mu_f) > 0.3 or sig_f < 2.5:
        fail(f"forward KL: expected mu near 0 and sigma above 2.5 (both modes), got mu={mu_f:.2f}, sigma={sig_f:.2f}")


@check("on-policy distillation drives the exact sequence-level reverse KL below 2% of its start, on 3 teachers")
def test_training(sub):
    train = has(sub, "train_on_policy_distill")
    for seed in (0, 1, 2):
        teacher = sub.make_teacher(seed)
        start = sub.exact_sequence_kl(torch.zeros_like(teacher), teacher)
        student = train(teacher, seed=seed + 10)
        end = sub.exact_sequence_kl(student, teacher)
        if not end < 0.02 * start:
            fail(f"teacher {seed}: expected the reverse KL to fall below {0.02 * start:.4f} (2% of {start:.4f}), got {end:.4f}")
