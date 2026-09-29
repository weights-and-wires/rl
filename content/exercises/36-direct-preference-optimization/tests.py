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


@check("kto_reference_point clamps at zero, and kto_loss matches hand-computed values")
def test_kto(sub):
    ref_point = has(sub, "kto_reference_point")
    kto = has(sub, "kto_loss")
    almost(ref_point(np.array([-3.0, -2.0]), np.array([-3.5, -2.1])), 0.3, tol=1e-12,
           label="mean mismatched log-ratio (0.5 + 0.1) / 2")
    almost(ref_point(np.array([-4.0, -2.0]), np.array([-3.5, -2.1])), 0.0, tol=1e-12,
           label="a negative mean clamps to 0")

    # beta 0.1, z0 0.2. Desirable with r = 1: 1 - sigmoid(0.08) = 0.480006.
    # Undesirable with r = -0.5: 1 - sigmoid(0.07) = 0.482507.
    policy = np.array([-4.0, -6.5])
    ref = np.array([-5.0, -6.0])
    desirable = np.array([True, False])
    want = 0.5 * ((1 - _sigmoid(0.08)) + (1 - _sigmoid(0.07)))
    almost(kto(policy, ref, desirable, 0.2, 0.1), want, tol=1e-10, label="balanced KTO loss")

    # With lambda_d = 1.33 and lambda_u = 1, each term is lambda_y - lambda_y * sigmoid(...).
    want_w = 0.5 * (1.33 * (1 - _sigmoid(0.08)) + (1 - _sigmoid(0.07)))
    almost(kto(policy, ref, desirable, 0.2, 0.1, lambda_d=1.33, lambda_u=1.0), want_w, tol=1e-10,
           label="KTO loss with lambda_d = 1.33")


@check("simpo_loss uses length-normalised log-probs: log 2 on the worked example, blind to repeated tokens")
def test_simpo(sub):
    simpo = has(sub, "simpo_loss")
    T = 12
    tw = np.zeros((1, T))
    tw[0, :10] = -1.0          # 10 tokens, total -10, average -1
    mw = np.zeros((1, T))
    mw[0, :10] = 1
    tl = np.zeros((1, T))
    tl[0, :8] = -1.5           # 8 tokens, total -12, average -1.5
    ml = np.zeros((1, T))
    ml[0, :8] = 1
    # beta 2, gamma 1: 2(-1) - 2(-1.5) - 1 = 0, so the loss is log 2.
    almost(simpo(tw, mw, tl, ml, 2.0, 1.0), math.log(2.0), tol=1e-12, label="worked example")

    # Repeat every token of the rejected response: its average is unchanged.
    rng = np.random.default_rng(4)
    tw2 = rng.uniform(-3, -0.1, size=(3, 4))
    tl2 = rng.uniform(-3, -0.1, size=(3, 4))
    ones = np.ones((3, 4))
    doubled = np.concatenate([tl2, tl2], axis=1)
    base = simpo(tw2, ones, tl2, ones, 2.5, 1.0)
    almost(simpo(tw2, ones, doubled, np.ones((3, 8)), 2.5, 1.0), base, tol=1e-12,
           label="SimPO after doubling the rejected response")

    # One response per row, with a padded tail that must be ignored.
    padded_w = np.concatenate([tw2, np.full((3, 2), -50.0)], axis=1)
    mask_w = np.concatenate([ones, np.zeros((3, 2))], axis=1)
    almost(simpo(padded_w, mask_w, tl2, ones, 2.5, 1.0), base, tol=1e-12,
           label="SimPO with padding on the chosen response")


@check("fit_dpo_exact recovers the hidden reward up to -beta log Z(x), a different constant per prompt")
def test_fit_dpo_exact(sub):
    fit = has(sub, "fit_dpo_exact")
    rng = np.random.default_rng(0)
    ref = _log_softmax(rng.normal(size=(2, 9)))
    reward = rng.normal(size=(2, 9))
    beta = 0.5
    fitted = np.asarray(fit(ref, reward, beta))
    shape(fitted, (2, 9), label="fit_dpo_exact output")
    close(np.exp(fitted).sum(axis=1), np.ones(2), atol=1e-9, label="rows of exp(fitted)")

    gap = beta * (fitted - ref) - reward
    close(gap - gap.mean(axis=1, keepdims=True), np.zeros((2, 9)), atol=1e-5,
          label="implicit reward minus true reward, minus its per-prompt mean")
    log_z = np.log(np.sum(np.exp(ref + reward / beta), axis=1))
    close(gap.mean(axis=1), -beta * log_z, atol=1e-5, label="the per-prompt constant, against -beta log Z(x)")
    gibbs = np.exp(ref + reward / beta - log_z[:, None])
    close(np.exp(fitted), gibbs, atol=1e-5, label="fitted policy against pi_ref exp(r / beta) / Z")


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
