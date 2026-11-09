"""Checks for the supervised fine-tuning exercise.

Wrong implementations this suite rejects:

- sft_loss_without_prompt_mask adds the prompt target's NLL. On the
  three-token example it returns about 3.3345 instead of 0.2395, and the
  finite-difference gradient on the logits that predict the prompt token is
  not zero.
- sft_loss_without_label_shift scores tokens[t] from logits[t]. The response
  gradient then lands on the last position, which the shift drops.
- sft_loss_masking_logit_positions zeros every logit that sits on a prompt
  token, including the last prompt token, which predicts the first response
  token. The finite-difference check expects that position's softmax gradient.
- pack_sequences_full_causal builds one lower-triangular mask over the packed
  row, so the second example attends into the first. The packed loss then
  disagrees with the sum of the unpacked losses.
"""

import numpy as np

from rlcheck import almost, check, close, equal, fail, finite_difference, has


def _toy_logits(tokens, position_ids, attn_mask, seed=0):
    """The attention layer behind Figure 2. seed 0 is the lesson's model."""
    rng = np.random.default_rng(seed)
    width = 4
    vocab = 8
    token_table = rng.normal(size=(16, width))
    position_table = rng.normal(size=(16, width))
    query_w = rng.normal(size=(width, width)) / np.sqrt(width)
    key_w = rng.normal(size=(width, width)) / np.sqrt(width)
    value_w = rng.normal(size=(width, width)) / np.sqrt(width)
    out_w = rng.normal(size=(width, vocab)) / np.sqrt(width)
    hidden = token_table[tokens] + position_table[position_ids]
    scores = (hidden @ query_w) @ (hidden @ key_w).T / np.sqrt(width)
    scores = np.where(attn_mask > 0, scores, -1e9)
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return (weights @ (hidden @ value_w)) @ out_w


def _response_gradient(logits_row, target):
    """Softmax minus one-hot, the gradient of one kept NLL term."""
    shifted = logits_row - np.max(logits_row)
    probs = np.exp(shifted)
    probs = probs / probs.sum()
    grad = probs.copy()
    grad[target] -= 1.0
    return grad


@check("sft_loss matches the three-token example, 0.2395")
def test_sft_loss_example(sub):
    sft_loss = has(sub, "sft_loss")
    logits = np.array([[[3.0, 0.0, 0.0], [0.0, 0.0, 2.0], [1.0, 1.0, 1.0]]])
    tokens = np.array([[0, 1, 2]])
    prompt_mask = np.array([[1.0, 1.0, 0.0]])
    got = float(sft_loss(logits, tokens, prompt_mask))
    # log(2 + e^2) - 2. The prompt target, log(e^3 + 2) = 3.0949, is dropped.
    almost(got, 0.2395447662218846, tol=1e-8, label="three-token NLL")


@check("sft_loss sums two response tokens instead of averaging them")
def test_sft_loss_sums(sub):
    sft_loss = has(sub, "sft_loss")
    logits = np.array(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [1.0, 1.0, 1.0]]]
    )
    tokens = np.array([[0, 1, 2, 2]])
    prompt_mask = np.array([[1.0, 1.0, 0.0, 0.0]])
    got = float(sft_loss(logits, tokens, prompt_mask))
    want = 0.4790895324437692
    if abs(got - want / 2) <= 1e-6 and abs(got - want) > 1e-6:
        fail(
            f"got {got:.6g}, the mean of the two response losses. "
            f"Expected the sum {want:.6g}."
        )
    almost(got, want, tol=1e-8, label="sum of two response NLLs")


@check("prompt targets and the dropped last position have gradient 0")
def test_sft_loss_gradient(sub):
    sft_loss = has(sub, "sft_loss")
    logits = np.array(
        [[[3.0, 0.0, 0.0], [0.0, 0.0, 2.0], [1.0, 1.0, 1.0]]], dtype=np.float64
    )
    tokens = np.array([[0, 1, 2]])
    prompt_mask = np.array([[1.0, 1.0, 0.0]])

    def scalar(flat):
        return float(sft_loss(flat.reshape(logits.shape), tokens, prompt_mask))

    grad = finite_difference(scalar, logits.ravel().copy(), eps=1e-5).reshape(logits.shape)
    prompt_grad = grad[0, 0]
    if np.max(np.abs(prompt_grad)) > 1e-7:
        fail(
            "expected gradient 0 on logits that predict a prompt token, "
            f"got {prompt_grad}."
        )
    tail_grad = grad[0, 2]
    if np.max(np.abs(tail_grad)) > 1e-7:
        fail(
            "expected gradient 0 on the last position, which has no label. "
            f"Got {tail_grad}. A shift that scores tokens[t] from logits[t] "
            "puts the response gradient here."
        )
    # Position 1 is a prompt token, and it predicts the response.
    want = _response_gradient(logits[0, 1], target=2)
    close(grad[0, 1], want, atol=1e-6, label="gradient on the logits that predict the reply")


@check("pack_sequences returns tokens, restarted positions, and a block-diagonal mask")
def test_pack_layout(sub):
    pack_sequences = has(sub, "pack_sequences")
    got = pack_sequences(
        [np.array([1, 2, 3]), np.array([4, 5])],
        [np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0])],
    )
    if not isinstance(got, tuple) or len(got) != 4:
        fail(
            "expected a tuple (tokens, position_ids, attn_mask, prompt_mask), "
            f"got {type(got).__name__}"
            + (f" of length {len(got)}" if isinstance(got, tuple) else "")
        )
    tokens, position_ids, attn_mask, prompt_mask = got
    equal(np.asarray(tokens), np.array([1, 2, 3, 4, 5]), label="packed tokens")
    equal(np.asarray(position_ids), np.array([0, 1, 2, 0, 1]), label="position ids")
    equal(
        np.asarray(prompt_mask, dtype=float),
        np.array([1.0, 0.0, 0.0, 1.0, 0.0]),
        label="packed prompt mask",
    )
    want = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
        ]
    )
    close(np.asarray(attn_mask, dtype=float), want, atol=0.0, label="block-diagonal mask")


@check("packed loss equals the sum of the unpacked losses to 1e-10")
def test_packed_loss_matches(sub):
    sft_loss = has(sub, "sft_loss")
    pack_sequences = has(sub, "pack_sequences")
    examples = [
        (np.array([1, 2, 3]), np.array([1.0, 0.0, 0.0])),
        (np.array([4, 5]), np.array([1.0, 0.0])),
    ]
    unpacked = 0.0
    for tokens, prompt in examples:
        length = len(tokens)
        logits = _toy_logits(tokens, np.arange(length), np.tril(np.ones((length, length))))
        unpacked += float(sft_loss(logits[None], tokens[None], prompt[None]))
    packed = pack_sequences(
        [tokens for tokens, _ in examples],
        [prompt for _, prompt in examples],
    )
    if not isinstance(packed, tuple) or len(packed) != 4:
        fail(
            "expected a tuple (tokens, position_ids, attn_mask, prompt_mask), "
            f"got {type(packed).__name__}"
        )
    tokens, position_ids, attn_mask, prompt_mask = packed
    logits = _toy_logits(
        np.asarray(tokens), np.asarray(position_ids), np.asarray(attn_mask, dtype=float)
    )
    got = float(
        sft_loss(logits[None], np.asarray(tokens)[None], np.asarray(prompt_mask)[None])
    )
    # pack_sequences_full_causal: one tril over the packed row.
    length = len(np.asarray(tokens))
    crossed_logits = _toy_logits(
        np.asarray(tokens),
        np.asarray(position_ids),
        np.tril(np.ones((length, length))),
    )
    crossed = float(
        sft_loss(
            crossed_logits[None],
            np.asarray(tokens)[None],
            np.asarray(prompt_mask)[None],
        )
    )
    if abs(got - unpacked) > 1e-10:
        fail(
            "packed loss does not equal the sum of the unpacked losses.\n"
            f"  got: {got:.12g}\n"
            f"  expected: {unpacked:.12g}\n"
            "  pack_sequences_full_causal, which lets the second example "
            f"attend into the first, gives {crossed:.12g}."
        )


@check("rejection_sample_filter keeps the top k of each prompt, smallest index on a tie")
def test_rejection_filter(sub):
    rejection_sample_filter = has(sub, "rejection_sample_filter")
    prompt_ids = np.array([0, 0, 0, 1, 1])
    rewards = np.array([0.2, 0.9, 0.9, 0.1, 0.4])
    got = np.asarray(
        rejection_sample_filter(prompt_ids, rewards, k=1), dtype=bool
    )
    # Index 1 and index 2 tie at 0.9. The smaller index wins.
    equal(got, np.array([False, True, False, False, True]), label="k = 1, with a tie")
    got_two = np.asarray(
        rejection_sample_filter(prompt_ids, rewards, k=2), dtype=bool
    )
    equal(
        got_two,
        np.array([False, True, True, True, True]),
        label="k = 2",
    )
    scattered = np.array([1, 0, 1, 0, 2])
    scattered_rewards = np.array([0.4, 0.2, 0.1, 0.9, -3.0])
    got_scattered = np.asarray(
        rejection_sample_filter(scattered, scattered_rewards, k=1), dtype=bool
    )
    equal(
        got_scattered,
        np.array([True, False, False, True, True]),
        label="prompts that are not stored contiguously",
    )


@check("the kept rows match reward-weighted regression with weights of 0 or 1")
def test_rejection_matches_rwr(sub):
    sft_loss = has(sub, "sft_loss")
    rejection_sample_filter = has(sub, "rejection_sample_filter")
    logits = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [1.0, 1.0, 1.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 1.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        ]
    )
    tokens = np.array(
        [
            [0, 1, 2],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 0],
        ]
    )
    prompt_mask = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    prompt_ids = np.array([0, 0, 1, 1])
    rewards = np.array([0.2, 1.5, 0.4, 3.0])
    keep = np.asarray(
        rejection_sample_filter(prompt_ids, rewards, k=1), dtype=bool
    )
    per_row = np.array(
        [
            float(sft_loss(logits[i : i + 1], tokens[i : i + 1], prompt_mask[i : i + 1]))
            for i in range(len(rewards))
        ]
    )
    # Independent of the student's filter: top 1 is index 1 (reward 1.5) and
    # index 3 (reward 3.0).
    weights = np.array([0.0, 1.0, 0.0, 1.0])
    rwr = float(weights @ per_row)
    raw = float(rewards @ per_row)
    if not np.any(keep):
        fail(f"the filter kept no rows. Expected weights {weights.astype(int)}.")
    got = float(sft_loss(logits[keep], tokens[keep], prompt_mask[keep]))
    if abs(got - raw) <= 1e-8:
        fail(
            "the loss matches reward-weighted regression with the raw rewards "
            f"as weights ({raw:.6g}). Expected weights of 0 or 1, which give {rwr:.6g}."
        )
    almost(got, rwr, tol=1e-10, label="weights of 0 or 1")
