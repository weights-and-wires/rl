"""Supervised fine-tuning: masked next-token loss, packing, and rejection sampling.

Fill in the three functions that raise NotImplementedError, in order.

`prompt_mask` uses 1 for a prompt token and 0 for a response token. Logits at
position t score the token at position t + 1. `sft_loss` returns the sum of
the kept negative log-probabilities, not the mean.
"""

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
    raise NotImplementedError("sft_loss")


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
    raise NotImplementedError("pack_sequences")


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
    raise NotImplementedError("rejection_sample_filter")


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
