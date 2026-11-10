"""Checks for the exercise on turning a task metric into a reward."""

import numpy as np

from rlcheck import almost, check, close, fail, finite_difference, has, shape

def _iou_reference(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    wa = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    wb = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    iw = max(min(a[2], b[2]) - max(a[0], b[0]), 0.0)
    ih = max(min(a[3], b[3]) - max(a[1], b[1]), 0.0)
    inter = iw * ih
    union = wa + wb - inter
    return inter / union if union > 0 else 0.0

@check("iou gives 1, 0, 1/7 and 0.5 on known boxes, broadcasts over batches, is symmetric, and returns 0 for degenerate boxes")
def test_iou(sub):
    iou = has(sub, "iou")
    almost(iou([0, 0, 2, 2], [0, 0, 2, 2]), 1.0, 1e-12, label="identical boxes")
    almost(iou([0, 0, 1, 1], [2, 2, 3, 3]), 0.0, 1e-12, label="disjoint boxes")
    almost(iou([0, 0, 1, 1], [1, 0, 2, 1]), 0.0, 1e-12, label="boxes that share only an edge")
    almost(iou([0, 0, 2, 2], [1, 1, 3, 3]), 1 / 7, 1e-12, label="(0,0,2,2) against (1,1,3,3)")
    almost(iou([0, 0, 4, 2], [1, 0, 3, 2]), 0.5, 1e-12, label="a box inside a box twice its area")
    rng = np.random.default_rng(0)
    corners = rng.uniform(0, 1, size=(6, 5, 2, 2))
    a = np.concatenate([corners[..., 0, :].min(-1, keepdims=True), corners[..., 1, :].min(-1, keepdims=True),
                        corners[..., 0, :].max(-1, keepdims=True), corners[..., 1, :].max(-1, keepdims=True)], -1)
    b = rng.uniform(0, 1, size=(5, 4))
    b = np.concatenate([np.minimum(b[:, :2], b[:, 2:]), np.maximum(b[:, :2], b[:, 2:])], -1)
    got = np.asarray(iou(a, b))
    shape(got, (6, 5), label="iou of (6, 5, 4) boxes against (5, 4) boxes")
    want = np.array([[_iou_reference(a[i, j], b[j]) for j in range(5)] for i in range(6)])
    close(got, want, atol=1e-12, label="batched iou")
    close(np.asarray(iou(b, a)), want, atol=1e-12, label="iou with the arguments swapped")

    degenerate = np.array([
        [0.5, 0.5, 0.5, 0.5],  # a point
        [0.2, 0.2, 0.2, 0.9],  # zero width
        [0.8, 0.8, 0.2, 0.2],  # corners swapped: far edges before near edges
    ])
    target = np.array([0.1, 0.1, 0.9, 0.9])
    got = np.asarray(iou(degenerate, target))
    if not np.all(np.isfinite(got)):
        fail(f"iou returned NaN or infinity for degenerate boxes: {got}")
    close(got, np.zeros(3), atol=1e-12, label="iou of degenerate boxes against a real box")
    almost(float(iou([0.3, 0.3, 0.3, 0.3], [0.3, 0.3, 0.3, 0.3])), 0.0, 1e-12,
           label="two identical points, whose union has zero area")

@check("detection_reward is hacked by extra boxes without the length penalty and repaired with it")
def test_detection_reward(sub):
    detection_reward = has(sub, "detection_reward")
    gt = np.array([[0.1, 0.1, 0.4, 0.4]])
    junk = np.array([[0.5, 0.5, 0.6, 0.6], [0.7, 0.1, 0.9, 0.2], [0.0, 0.8, 0.2, 1.0]])
    exact = gt.copy()
    spray = np.concatenate([exact, junk])
    almost(detection_reward(exact, gt, length_penalty=True), 1.0, 1e-12, label="one exact box")
    almost(detection_reward(spray, gt, length_penalty=False), 1.0, 1e-12,
           label="exact box plus three junk boxes, no penalty")
    almost(detection_reward(spray, gt, length_penalty=True), 0.25, 1e-12,
           label="exact box plus three junk boxes, with min(1, 1/4)")
    # Two targets, three predictions: best matches 1 and 1/7, penalty 2/3.
    gt2 = np.array([[0, 0, 2, 2], [10, 10, 12, 12]], float)
    pred = np.array([[0, 0, 2, 2], [11, 11, 13, 13], [5, 5, 6, 6]], float)
    almost(detection_reward(pred, gt2, length_penalty=False), (1 + 1 / 7) / 2, 1e-12,
           label="two targets, recall")
    almost(detection_reward(pred, gt2, length_penalty=True), (2 / 3) * (1 + 1 / 7) / 2, 1e-12,
           label="two targets, penalised")
    almost(detection_reward(np.zeros((0, 4)), gt2), 0.0, 1e-12, label="no predictions at all")

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

@check("residual_action returns the base action at raw 0 and never moves further than bound from it")
def test_residual_action(sub):
    residual_action = has(sub, "residual_action")
    base = np.array([-1.2, 0.0, 0.7, 1.9])
    close(np.asarray(residual_action(base, np.zeros(4), 0.5)), base, atol=1e-15, label="raw = 0")
    raw = np.array([-50.0, -0.3, 0.4, 50.0])
    got = np.asarray(residual_action(base, raw, 0.5))
    close(got, base + 0.5 * np.tanh(raw), atol=1e-12, label="base + bound * tanh(raw)")
    if np.max(np.abs(got - base)) > 0.5 + 1e-12:
        fail(f"the residual moved the action by {np.max(np.abs(got - base)):.4f}, more than the bound 0.5")

@check("reinforce_gradient matches a hand computation with reward-to-go and a leave-one-out baseline")
def test_reinforce_gradient_hand(sub):
    reinforce_gradient = has(sub, "reinforce_gradient")
    rng = np.random.default_rng(2)
    n, horizon, sigma = 3, 2, 0.5
    rewards = rng.normal(size=(n, horizon))
    features = rng.normal(size=(n, horizon, 3))
    eps = rng.normal(size=(n, horizon))
    want = np.zeros(3)
    for i in range(n):
        for t in range(horizon):
            g_it = rewards[i, t:].sum()
            b_it = np.mean([rewards[j, t:].sum() for j in range(n) if j != i])
            want += (g_it - b_it) * eps[i, t] / sigma * features[i, t]
    want /= n
    got = np.asarray(reinforce_gradient(rewards, features, eps, sigma))
    shape(got, (3,), label="gradient")
    close(got, want, atol=1e-12, label="gradient on a 3-episode, 2-step batch")
    shifted = np.asarray(reinforce_gradient(rewards + 4.0, features, eps, sigma))
    close(shifted, got, atol=1e-10, label="gradient after adding 4 to every reward")

@check("reinforce_gradient recovers the worked example: base action 0.8, target 1, gradient 0.4")
def test_reinforce_gradient_worked(sub):
    reinforce_gradient = has(sub, "reinforce_gradient")
    rng = np.random.default_rng(3)
    n, sigma = 100_000, 0.1
    eps = rng.standard_normal((n, 1))
    action = 0.8 + 0.0 + sigma * eps[:, 0]  # base action plus a residual w = 0, plus noise
    rewards = -((action - 1.0) ** 2)[:, None]
    got = np.asarray(reinforce_gradient(rewards, np.ones((n, 1, 1)), eps, sigma))
    almost(got[0], 0.4, 0.02, label="d E[r] / d w at w = 0, exactly -2 (0.8 - 1)")

# Plausible wrong implementations this check catches:
#   - decoding to the bin's left edge instead of its centre: the round-trip error reaches a
#     full bin width, so the half-bin bound fails;
#   - encoding with round() on (a - low) / width, which makes n_bins + 1 bins: the top end
#     maps to token n_bins, and every hand value near a bin edge lands one token off.
@check("action_to_tokens and tokens_to_action round-trip within half a bin, map the ends to 0 and n_bins - 1, and clip")
def test_action_tokens(sub):
    action_to_tokens = has(sub, "action_to_tokens")
    tokens_to_action = has(sub, "tokens_to_action")
    # Hand values: 8 bins on [-1, 1] have width 0.25 and centres -0.875, -0.625, ..., 0.875.
    hand = [(0.30, 5, 0.375), (0.5, 6, 0.625), (-0.9, 0, -0.875), (0.0, 4, 0.125), (-0.24, 3, -0.125), (0.40, 5, 0.375)]
    for value, token, centre in hand:
        got = action_to_tokens(np.array([value]), -1.0, 1.0, 8)
        if int(np.asarray(got)[0]) != token:
            fail(f"action_to_tokens({value}, low=-1, high=1, n_bins=8) should be token {token}, got {np.asarray(got)[0]}")
        almost(float(np.asarray(tokens_to_action(np.array([token]), -1.0, 1.0, 8))[0]), centre, 1e-12,
               label=f"decoded centre of token {token}")
    # The ends of the range land in the first and last bins, not in a bin n_bins.
    ends = np.asarray(action_to_tokens(np.array([-1.0, 1.0]), -1.0, 1.0, 8))
    if not np.array_equal(ends, [0, 7]):
        fail(f"the endpoints -1 and 1 should map to tokens [0, 7], got {ends}")
    # Out-of-range actions clip to the end bins.
    clipped = np.asarray(action_to_tokens(np.array([-5.0, 3.0]), -1.0, 1.0, 8))
    if not np.array_equal(clipped, [0, 7]):
        fail(f"actions -5 and 3 are outside [-1, 1] and should clip to tokens [0, 7], got {clipped}")
    # Random actions: a vectorised call, per-dimension ranges, and the half-bin round-trip bound.
    rng = np.random.default_rng(0)
    low = np.array([-0.02, 0.0, -3.0])
    high = np.array([0.02, 1.0, 1.0])
    for n_bins in (8, 256):
        actions = rng.uniform(low, high, size=(500, 3))
        tokens = np.asarray(action_to_tokens(actions, low, high, n_bins))
        shape(tokens, (500, 3), label="tokens for 500 actions of 3 dimensions")
        if tokens.min() < 0 or tokens.max() > n_bins - 1:
            fail(f"tokens should lie in 0 .. {n_bins - 1}, got range {tokens.min()} .. {tokens.max()}")
        back = np.asarray(tokens_to_action(tokens, low, high, n_bins))
        half_bin = (high - low) / n_bins / 2
        worst = np.max(np.abs(back - actions) / half_bin)
        if worst > 1 + 1e-9:
            fail(f"with {n_bins} bins the round-trip error should be at most half a bin width, "
                 f"but the worst error was {worst:.3f} half-bins")
        if n_bins == 8 and worst < 0.8:
            fail(f"the worst round-trip error over 1500 random values should come close to half a bin, got {worst:.3f} half-bins")
