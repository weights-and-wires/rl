"""Supervised fine-tuning: masked next-token loss, packing, and rejection sampling."""

import numpy as np


def sft_loss(logits, tokens, prompt_mask):
    """Sum of next-token NLLs on response tokens, after the one-token shift.

    Args:
        logits: float array, shape (B, T, V). Row t scores the token at t + 1.
        tokens: int array, shape (B, T).
        prompt_mask: array, shape (B, T). 1 on prompt tokens, 0 on response
            tokens.

    Returns:
        A float. Prompt targets and the last position contribute nothing.
        Two rows add.
    """
    logits = np.asarray(logits, dtype=np.float64)
    tokens = np.asarray(tokens)
    prompt_mask = np.asarray(prompt_mask)
    pred = logits[:, :-1, :]
    target = tokens[:, 1:]
    # The mask follows the token being predicted, one step to the right.
    supervise = np.asarray(prompt_mask[:, 1:] == 0)
    shifted = pred - pred.max(axis=-1, keepdims=True)
    log_z = pred.max(axis=-1) + np.log(np.exp(shifted).sum(axis=-1))
    chosen = np.take_along_axis(pred, target[..., None], axis=-1)[..., 0]
    nll = log_z - chosen
    return float(np.where(supervise, nll, 0.0).sum())


def pack_sequences(token_rows, prompt_rows):
    """Concatenate examples so attention cannot cross the boundary.

    Args:
        token_rows: list of 1D int arrays, one example each.
        prompt_rows: list of 1D arrays, 1 on prompt tokens. Row i has the
            same length as token_rows[i].

    Returns:
        tokens: int array, shape (L,), the examples in order.
        position_ids: int array, shape (L,), restarting at 0 each example.
        attn_mask: float array, shape (L, L). attn_mask[i, j] is 1 when
            token i may attend to token j (same example, and j <= i).
        prompt_mask: float array, shape (L,), the prompt rows concatenated.
    """
    if len(token_rows) != len(prompt_rows):
        raise ValueError("token_rows and prompt_rows must have the same length")
    tokens = []
    positions = []
    prompts = []
    for token_row, prompt_row in zip(token_rows, prompt_rows):
        token_row = np.asarray(token_row, dtype=np.int64)
        prompt_row = np.asarray(prompt_row, dtype=np.float64)
        if token_row.shape != prompt_row.shape:
            raise ValueError("each prompt mask must match its token row")
        tokens.append(token_row)
        prompts.append(prompt_row)
        positions.append(np.arange(token_row.shape[0], dtype=np.int64))
    packed_tokens = np.concatenate(tokens)
    packed_prompt = np.concatenate(prompts)
    position_ids = np.concatenate(positions)
    length = int(packed_tokens.shape[0])
    attn_mask = np.zeros((length, length), dtype=np.float64)
    start = 0
    for token_row in tokens:
        stop = start + int(token_row.shape[0])
        block = stop - start
        attn_mask[start:stop, start:stop] = np.tril(np.ones((block, block)))
        start = stop
    return packed_tokens, position_ids, attn_mask, packed_prompt


def rejection_sample_filter(prompt_ids, rewards, k):
    """Keep the k highest-reward samples of each prompt.

    Args:
        prompt_ids: int array, shape (N,). Equal ids are one prompt.
        rewards: float array, shape (N,).
        k: how many samples to keep for each prompt. A prompt with fewer
            than k samples keeps all of them.

    Returns:
        Boolean array, shape (N,). Ties break toward the smaller index.
        The kept rows are the ones a later `sft_loss` trains on. Weighting
        each row's NLL by 1 on these rows and 0 elsewhere equals that loss.
    """
    prompt_ids = np.asarray(prompt_ids)
    rewards = np.asarray(rewards, dtype=np.float64)
    if prompt_ids.shape != rewards.shape:
        raise ValueError("prompt_ids and rewards must have the same shape")
    if k < 1:
        raise ValueError("k must be at least 1")
    keep = np.zeros(prompt_ids.shape, dtype=bool)
    for prompt in np.unique(prompt_ids):
        idx = np.flatnonzero(prompt_ids == prompt)
        # Last key is the primary key: higher reward first, then smaller index.
        order = np.lexsort((idx, -rewards[idx]))
        keep[idx[order[:k]]] = True
    return keep


if __name__ == "__main__":
    logits = np.array([[[3.0, 0.0, 0.0], [0.0, 0.0, 2.0], [1.0, 1.0, 1.0]]])
    tokens = np.array([[0, 1, 2]])
    prompt_mask = np.array([[1.0, 1.0, 0.0]])
    print("sft loss", sft_loss(logits, tokens, prompt_mask))
    packed = pack_sequences(
        [np.array([1, 2, 3]), np.array([4, 5])],
        [np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0])],
    )
    print("packed length", len(packed[0]))
    kept = rejection_sample_filter(
        np.array([0, 0, 1]), np.array([0.2, 0.9, 0.4]), k=1
    )
    print("kept", kept)
