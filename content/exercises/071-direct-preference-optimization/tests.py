"""Checks for the direct preference optimisation exercise."""

import math

import numpy as np

from rlcheck import almost, check, close, fail, finite, finite_difference, has, shape


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=float)))


def _log_softmax(z):
    z = np.asarray(z, dtype=float)
    shifted = z - z.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def _unit_pair_embeddings(n, dim, cosine, seed):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(n, dim))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    a = rng.normal(size=dim)
    a /= np.linalg.norm(a)
    b = rng.normal(size=dim)
    b -= (b @ a) * a
    b /= np.linalg.norm(b)
    E[0] = a
    E[1] = cosine * a + np.sqrt(1.0 - cosine**2) * b
    return E


@check("sequence_logprob sums per-token log-probabilities over the response mask")
def test_sequence_logprob(sub):
    seq = has(sub, "sequence_logprob")
    token_logps = np.array([[-0.5, -1.0, -2.0, -np.inf], [-0.1, -0.2, -7.0, np.nan]])
    mask = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=float)
    got = seq(token_logps, mask)
    shape(got, (2,), label="sequence_logprob output")
    # -0.5 - 1 - 2 = -3.5, and -0.1 - 0.2 = -0.3. Padding never counts.
    close(got, np.array([-3.5, -0.3]), atol=1e-12, label="masked sums")


@check("dpo_loss matches the lesson's worked example and ignores a shared shift of log Z")
def test_dpo_loss(sub):
    dpo = has(sub, "dpo_loss")
    beta = 0.1
    # Margin h = (-10 + 11) - (-12 + 11.5) = 1.5, so the loss is -log sigmoid(0.15).
    got = dpo(np.array([-10.0]), np.array([-12.0]), np.array([-11.0]), np.array([-11.5]), beta)
    almost(got, math.log1p(math.exp(-0.15)), tol=1e-10, label="worked example")
    almost(got, 0.62095, tol=1e-4, label="worked example to four places")

    tied = dpo(np.full(4, -3.0), np.full(4, -5.0), np.full(4, -3.0), np.full(4, -5.0), 0.7)
    almost(tied, math.log(2.0), tol=1e-12, label="policy equal to the reference")

    rng = np.random.default_rng(1)
    pw, pl, rw, rl = rng.uniform(-6, -1, size=(4, 8))
    base = dpo(pw, pl, rw, rl, 0.3)
    shift = rng.uniform(-4, 4, size=8)
    # Shifting both policy log-probs for a prompt by the same amount is a change of
    # log Z(x); the loss must not move.
    almost(dpo(pw + shift, pl + shift, rw, rl, 0.3), base, tol=1e-10,
           label="dpo_loss after a per-prompt shift of the policy")

    w = rng.uniform(0.1, 1.0, size=8)
    w /= w.sum()
    want = float(np.sum(w * np.logaddexp(0.0, -0.3 * ((pw - rw) - (pl - rl)))))
    almost(dpo(pw, pl, rw, rl, 0.3, weights=w), want, tol=1e-10, label="weighted dpo_loss")


@check("dpo_grad matches finite differences, with and without weights")
def test_dpo_grad(sub):
    dpo = has(sub, "dpo_loss")
    grad = has(sub, "dpo_grad")
    rng = np.random.default_rng(2)
    n, beta = 6, 0.4
    pw, pl, rw, rl = rng.uniform(-8, -1, size=(4, n))
    for weights in (None, rng.dirichlet(np.ones(n))):
        g_w, g_l = grad(pw, pl, rw, rl, beta, weights=weights)
        shape(g_w, (n,), label="grad_chosen")
        shape(g_l, (n,), label="grad_rejected")
        fd = finite_difference(lambda x: dpo(x[:n], x[n:], rw, rl, beta, weights=weights),
                               np.concatenate([pw, pl]))
        label = "unweighted" if weights is None else "weighted"
        close(np.concatenate([g_w, g_l]), fd, rtol=1e-5, atol=1e-8,
              label=f"{label} dpo_grad against finite differences")
    # The worked example: -beta * sigmoid(-0.15) = -0.046257.
    g_w, _ = grad(np.array([-10.0]), np.array([-12.0]), np.array([-11.0]), np.array([-11.5]), 0.1)
    almost(g_w[0], -0.1 * _sigmoid(-0.15), tol=1e-10, label="worked-example gradient")


@check("ipo_loss is (h - 1/(2 tau))^2 and ipo_grad matches finite differences")
def test_ipo(sub):
    ipo = has(sub, "ipo_loss")
    grad = has(sub, "ipo_grad")
    # The DPO example has h = 1.5. With tau = 0.1 the target is 5: (1.5 - 5)^2 = 12.25.
    almost(ipo(np.array([-10.0]), np.array([-12.0]), np.array([-11.0]), np.array([-11.5]), 0.1),
           12.25, tol=1e-10, label="worked example")
    rng = np.random.default_rng(3)
    n, tau = 5, 0.25
    pw, pl, rw, rl = rng.uniform(-8, -1, size=(4, n))
    g_w, g_l = grad(pw, pl, rw, rl, tau)
    fd = finite_difference(lambda x: ipo(x[:n], x[n:], rw, rl, tau), np.concatenate([pw, pl]))
    close(np.concatenate([g_w, g_l]), fd, rtol=1e-5, atol=1e-7, label="ipo_grad against finite differences")


@check("With a deterministic preference, IPO's margin settles at 1/(2 tau) while DPO's keeps growing")
def test_ipo_versus_dpo_margin(sub):
    dpo_grad = has(sub, "dpo_grad")
    ipo_grad = has(sub, "ipo_grad")
    rng = np.random.default_rng(3)
    ref = _log_softmax(rng.normal(size=4))
    beta, tau, lr = 0.5, 0.5, 0.1

    def run(kind, steps):
        z = ref.copy()
        margins = []
        for _ in range(steps):
            lp = _log_softmax(z)
            args = (lp[[0]], lp[[1]], ref[[0]], ref[[1]])
            g_w, g_l = dpo_grad(*args, beta) if kind == "dpo" else ipo_grad(*args, tau)
            g = np.zeros(4)
            g[0], g[1] = g_w[0], g_l[0]
            z = z - lr * (g - np.exp(lp) * g.sum())
            lp = _log_softmax(z)
            margins.append((lp[0] - ref[0]) - (lp[1] - ref[1]))
        return np.array(margins)

    ipo_m = run("ipo", 2000)
    dpo_m = run("dpo", 2000)
    finite(ipo_m, label="IPO margins")
    almost(ipo_m[-1], 1.0 / (2.0 * tau), tol=1e-6, label="IPO's final margin")
    if not dpo_m[1999] > dpo_m[999] + 1.0:
        fail(
            "DPO's margin should keep growing on a preference that is always the same: "
            f"it was {dpo_m[999]:.3f} after 1000 steps and {dpo_m[1999]:.3f} after 2000"
        )


@check("train_dpo_linear shows likelihood displacement: both log-probs fall and the mass goes elsewhere")
def test_likelihood_displacement(sub):
    train = has(sub, "train_dpo_linear")
    dpo = has(sub, "dpo_loss")
    E = _unit_pair_embeddings(32, 4, cosine=0.95, seed=0)
    hist = np.asarray(train(E, np.zeros(4), 0, 1, beta=0.1, lr=1.0, steps=500))
    shape(hist, (501, 32), label="history of log-probabilities")
    close(np.exp(hist).sum(axis=1), np.ones(501), atol=1e-9, label="each row of exp(history)")

    ref = hist[0]
    before = dpo(ref[[0]], ref[[1]], ref[[0]], ref[[1]], 0.1)
    after = dpo(hist[-1][[0]], hist[-1][[1]], ref[[0]], ref[[1]], 0.1)
    if not after < before - 0.05:
        fail(f"the DPO loss should fall: {before:.4f} before and {after:.4f} after 500 steps")
    if not (hist[-1, 0] < ref[0] - 0.5 and hist[-1, 1] < ref[1] - 0.5):
        fail(
            "both log-probabilities should fall by more than 0.5: chosen went "
            f"{ref[0]:.3f} -> {hist[-1, 0]:.3f}, rejected {ref[1]:.3f} -> {hist[-1, 1]:.3f}"
        )
    gain = np.exp(hist[-1]) - np.exp(ref)
    aligned = int(np.argmax(E @ (E[0] - E[1])))
    top = int(np.argmax(gain))
    if top != aligned:
        fail(
            f"the response that gains the most probability should be {aligned}, the one whose "
            f"embedding points most along e_w - e_l; it was {top}"
        )


@check("Displacement grows with similarity: the chosen log-prob falls further at cosine 0.95 than at 0")
def test_displacement_needs_similarity(sub):
    train = has(sub, "train_dpo_linear")
    for seed in (0, 1, 2):
        drops = []
        for cosine in (0.95, 0.0):
            E = _unit_pair_embeddings(32, 4, cosine=cosine, seed=seed)
            hist = np.asarray(train(E, np.zeros(4), 0, 1, beta=0.1, lr=1.0, steps=500))
            drops.append(hist[-1, 0] - hist[0, 0])
        if not drops[0] < drops[1] - 1.0:
            fail(
                f"seed {seed}: the chosen log-prob changed by {drops[0]:.3f} at cosine 0.95 and "
                f"{drops[1]:.3f} at cosine 0; the similar pair should lose more"
            )


@check("simpo_loss averages log-probs per token: equal per-token log-probs give softplus(gamma) at any length")
def test_simpo(sub):
    simpo = has(sub, "simpo_loss")
    # Worked example: the chosen response has 10 tokens and a sum of -20 (average -2); the
    # rejected one has 12 tokens and a sum of -36 (average -3). With beta = 2 and gamma = 0.5,
    # the argument is 2 * (-2) - 2 * (-3) - 0.5 = 1.5, so the loss is log(1 + e^-1.5) = 0.2014.
    got = simpo(np.array([-20.0]), np.array([-36.0]), np.array([10]), np.array([12]), 2.0, 0.5)
    almost(got, math.log1p(math.exp(-1.5)), tol=1e-10, label="worked example")

    # Equal per-token log-probs (-2 on both sides) give a reward gap of 0 whatever the lengths,
    # so the loss is -log sigmoid(-gamma) = log(1 + e^gamma).
    # A plausible wrong implementation sums the log-probs without dividing by length. Its
    # reward gap then depends on the lengths (here 2 * (-20) - 2 * (-80) = 120), and it fails.
    for len_w, len_l in ((10, 10), (10, 40), (3, 25)):
        got = simpo(np.array([-2.0 * len_w]), np.array([-2.0 * len_l]),
                    np.array([len_w]), np.array([len_l]), 2.0, 0.5)
        almost(got, math.log1p(math.exp(0.5)), tol=1e-10,
               label=f"loss at equal per-token log-probs, lengths {len_w} and {len_l}")

    # A larger target margin gamma raises the loss on the same batch.
    args = (np.array([-20.0, -9.0]), np.array([-36.0, -14.0]), np.array([10, 6]), np.array([12, 7]), 2.0)
    low, high = simpo(*args, 0.0), simpo(*args, 1.0)
    if not high > low:
        fail(f"a larger gamma should raise the loss: gamma 0 gave {low:.4f} and gamma 1 gave {high:.4f}")


@check("kto_loss falls as a desirable example's log-ratio rises above z_ref, and z_ref is clamped at zero")
def test_kto(sub):
    ref_point = has(sub, "kto_ref_point")
    kto = has(sub, "kto_loss")
    # z_ref is the mean log-ratio of mismatched outputs, clamped at zero.
    # Log-ratios 0.5 and -0.1 average 0.2.
    almost(ref_point(np.array([-9.5, -11.1]), np.array([-10.0, -11.0])), 0.2, tol=1e-10,
           label="z_ref for a positive mean log-ratio")
    # A plausible wrong implementation leaves z_ref unclamped. Log-ratios -0.4 and -0.2 average
    # -0.3, and the paper clamps that to 0.
    almost(ref_point(np.array([-10.4, -11.2]), np.array([-10.0, -11.0])), 0.0, tol=1e-12,
           label="z_ref for a negative mean log-ratio")

    # Worked example: a desirable example with log-ratio +1 and an undesirable one with -0.5.
    # With beta = 1, lambda_D = 1, lambda_U = 1.5, z_ref = 0.2:
    #   desirable:   1.0 * (1 - sigmoid(1 * (1 - 0.2)))      = 0.31003
    #   undesirable: 1.5 * (1 - sigmoid(1 * (0.2 + 0.5)))    = 0.49773
    # and the mean is 0.40387. A flipped sign on the undesirable term gives a different value.
    pol = np.array([-9.0, -12.0])
    ref = np.array([-10.0, -11.5])
    good = np.array([True, False])
    got = kto(pol, ref, good, 1.0, 1.0, 1.5, 0.2)
    want = 0.5 * (1.0 * (1.0 - _sigmoid(0.8)) + 1.5 * (1.0 - _sigmoid(0.7)))
    almost(got, want, tol=1e-10, label="worked example")
    almost(got, 0.40387, tol=1e-5, label="worked example to five places")

    # Raising a desirable example's log-ratio lowers the loss; raising an undesirable one's raises it.
    lows = [kto(np.array([-10.0 + r]), np.array([-10.0]), np.array([True]), 1.0, 1.0, 1.5, 0.2)
            for r in (-1.0, 0.0, 1.0, 2.0)]
    if not all(a > b for a, b in zip(lows, lows[1:])):
        fail(f"the loss on a desirable example should fall as its log-ratio rises: got {lows}")
    highs = [kto(np.array([-10.0 + r]), np.array([-10.0]), np.array([False]), 1.0, 1.0, 1.5, 0.2)
             for r in (-1.0, 0.0, 1.0, 2.0)]
    if not all(a < b for a, b in zip(highs, highs[1:])):
        fail(f"the loss on an undesirable example should rise as its log-ratio rises: got {highs}")
