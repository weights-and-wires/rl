#!/usr/bin/env python3
"""Numbers and SVG for the supervised fine-tuning lesson.

The bars in Figure 1 are the per-position negative log-likelihoods of the
three-token example, with and without the prompt mask. The cells in Figure 2
are the block-diagonal mask from the packing example, and the caption numbers
are the packed and unpacked losses of the toy attention model below.

Run from the repo root:

    python3 scripts/figures/048-sft-and-the-post-training-pipeline.py
"""

from __future__ import annotations

import numpy as np


# The lesson's three-token example. Position 0 predicts a prompt token,
# position 1 predicts the response, and position 2 has no label.
EXAMPLE_LOGITS = np.array(
    [
        [3.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)
EXAMPLE_TOKENS = np.array([0, 1, 2])
EXAMPLE_PROMPT = np.array([1.0, 1.0, 0.0])


def token_nll(logits_t: np.ndarray, token: int) -> float:
    """Negative log-probability of one token under a row of logits."""
    shifted = logits_t - np.max(logits_t)
    log_z = np.max(logits_t) + np.log(np.exp(shifted).sum())
    return float(log_z - logits_t[token])


def sft_loss(logits: np.ndarray, tokens: np.ndarray, prompt_mask: np.ndarray) -> float:
    """Sum of response-token NLLs after the one-token label shift."""
    logits = np.asarray(logits, dtype=np.float64)
    tokens = np.asarray(tokens)
    prompt_mask = np.asarray(prompt_mask, dtype=np.float64)
    pred = logits[:, :-1, :]
    target = tokens[:, 1:]
    supervise = prompt_mask[:, 1:] == 0
    shifted = pred - pred.max(axis=-1, keepdims=True)
    log_z = pred.max(axis=-1) + np.log(np.exp(shifted).sum(axis=-1))
    chosen = np.take_along_axis(pred, target[..., None], axis=-1)[..., 0]
    nll = log_z - chosen
    return float(np.where(supervise, nll, 0.0).sum())


def toy_logits(tokens: np.ndarray, position_ids: np.ndarray, attn_mask: np.ndarray, seed: int = 0) -> np.ndarray:
    """One attention layer. attn_mask is 1 where a query may attend to a key."""
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


def example_contributions() -> tuple[np.ndarray, np.ndarray]:
    """Per-position NLL with the mask, and the same terms with the mask removed."""
    masked = np.zeros(3)
    bare = np.zeros(3)
    # Position t scores token t + 1. Position 2 has no label.
    bare[0] = token_nll(EXAMPLE_LOGITS[0], int(EXAMPLE_TOKENS[1]))
    bare[1] = token_nll(EXAMPLE_LOGITS[1], int(EXAMPLE_TOKENS[2]))
    masked[0] = 0.0 if EXAMPLE_PROMPT[1] == 1.0 else bare[0]
    masked[1] = 0.0 if EXAMPLE_PROMPT[2] == 1.0 else bare[1]
    return masked, bare


def packing_losses() -> dict[str, float]:
    """Separate losses, the packed loss, and the loss of a mask that crosses."""
    examples = [
        (np.array([1, 2, 3]), np.array([1.0, 0.0, 0.0])),
        (np.array([4, 5]), np.array([1.0, 0.0])),
    ]
    separate = []
    for tokens, prompt in examples:
        length = len(tokens)
        logits = toy_logits(tokens, np.arange(length), np.tril(np.ones((length, length))))
        separate.append(sft_loss(logits[None], tokens[None], prompt[None]))
    packed_tokens = np.concatenate([tokens for tokens, _ in examples])
    packed_prompt = np.concatenate([prompt for _, prompt in examples])
    packed_pos = np.concatenate([np.arange(len(tokens)) for tokens, _ in examples])
    length = len(packed_tokens)
    block = np.zeros((length, length))
    start = 0
    for tokens, _ in examples:
        stop = start + len(tokens)
        block[start:stop, start:stop] = np.tril(np.ones((stop - start, stop - start)))
        start = stop
    packed = sft_loss(
        toy_logits(packed_tokens, packed_pos, block)[None],
        packed_tokens[None],
        packed_prompt[None],
    )
    crossed_mask = np.tril(np.ones((length, length)))
    crossed = sft_loss(
        toy_logits(packed_tokens, packed_pos, crossed_mask)[None],
        packed_tokens[None],
        packed_prompt[None],
    )
    return {
        "first": separate[0],
        "second": separate[1],
        "packed": packed,
        "crossed": crossed,
    }


def _y(value: float, base: float = 220.0, top: float = 36.0, vmax: float = 3.5) -> float:
    return base - (value / vmax) * (base - top)


def figure_loss_bars(masked: np.ndarray, bare: np.ndarray) -> str:
    """Figure 1. Grouped bars of the three-token loss, masked and unmasked."""
    centers = (168.0, 348.0, 528.0)
    labels = (
        ("logits at 0", "prompt target"),
        ("logits at 1", "reply target"),
        ("logits at 2", "no label"),
    )
    parts = [
        '<svg viewBox="0 0 640 310" role="img" '
        'aria-label="Per-position loss on a 3-token example. '
        "Masking drops a prompt-target loss of 3.0949 to zero. "
        'The reply contributes 0.2395. The last position contributes nothing.">',
    ]
    for tick in (0, 1, 2, 3):
        y = _y(float(tick))
        parts.append(
            f'<line x1="72.0" y1="{y:.1f}" x2="600.0" y2="{y:.1f}" class="stroke-grid"/>'
        )
        parts.append(
            f'<text x="64.0" y="{y + 4.0:.1f}" class="fig-small" text-anchor="end">{tick}</text>'
        )
    parts.append(
        '<text x="16.0" y="128.0" class="fig-small" text-anchor="middle" '
        'transform="rotate(-90 16 128)">loss</text>'
    )
    for center, label, kept, forgotten in zip(centers, labels, masked, bare):
        forgotten_y = _y(float(forgotten))
        kept_y = _y(float(kept))
        forgotten_h = _y(0.0) - forgotten_y
        kept_h = _y(0.0) - kept_y
        parts.append(
            f'<rect x="{center - 38.0:.1f}" y="{forgotten_y:.1f}" width="32.0" '
            f'height="{max(forgotten_h, 0.0):.1f}" class="fill-b"/>'
        )
        parts.append(
            f'<rect x="{center + 6.0:.1f}" y="{kept_y:.1f}" width="32.0" '
            f'height="{max(kept_h, 0.0):.1f}" class="fill-a"/>'
        )
        if abs(float(forgotten) - float(kept)) < 1e-12:
            parts.append(
                f'<text x="{center:.1f}" y="{kept_y - 8.0:.1f}" '
                f'class="fig-small" text-anchor="middle">{kept:.4f}</text>'
            )
        else:
            parts.append(
                f'<text x="{center - 22.0:.1f}" y="{forgotten_y - 6.0:.1f}" '
                f'class="fig-small" text-anchor="middle">{forgotten:.4f}</text>'
            )
            parts.append(
                f'<text x="{center + 22.0:.1f}" y="{max(kept_y - 6.0, forgotten_y + 14.0):.1f}" '
                f'class="fig-small" text-anchor="middle">{kept:.4f}</text>'
            )
        parts.append(
            f'<text x="{center:.1f}" y="242.0" class="fig-small" text-anchor="middle">{label[0]}</text>'
        )
        parts.append(
            f'<text x="{center:.1f}" y="258.0" class="fig-small" text-anchor="middle">{label[1]}</text>'
        )
    parts.append('<rect x="150.0" y="274.0" width="14.0" height="14.0" class="fill-b"/>')
    parts.append(
        '<text x="170.0" y="286.0" class="fig-small" text-anchor="start">without the prompt mask</text>'
    )
    parts.append('<rect x="360.0" y="274.0" width="14.0" height="14.0" class="fill-a"/>')
    parts.append(
        '<text x="380.0" y="286.0" class="fig-small" text-anchor="start">with the prompt mask</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def figure_mask() -> str:
    """Figure 2. The 5 by 5 block-diagonal causal mask, cell by cell."""
    labels = ("A0", "A1", "A2", "B0", "B1")
    length = 5
    # Examples occupy [0, 3) and [3, 5). 1 means the query may attend to the key.
    mask = np.zeros((length, length))
    mask[:3, :3] = np.tril(np.ones((3, 3)))
    mask[3:, 3:] = np.tril(np.ones((2, 2)))
    cell = 48.0
    origin_x = 108.0
    origin_y = 36.0
    parts = [
        '<svg viewBox="0 0 640 330" role="img" '
        'aria-label="A 5 by 5 attention mask for two packed examples. '
        "Ones form a causal triangle inside each example. "
        'Zeros fill every cell that would look across the boundary.">',
    ]
    parts.append(
        '<text x="228.0" y="22.0" class="fig-small" text-anchor="middle">key</text>'
    )
    for index, label in enumerate(labels):
        x = origin_x + index * cell + cell / 2
        parts.append(
            f'<text x="{x:.1f}" y="{origin_y - 8.0:.1f}" class="fig-small fig-bold" '
            f'text-anchor="middle">{label}</text>'
        )
        y = origin_y + index * cell + cell / 2 + 4
        parts.append(
            f'<text x="{origin_x - 16.0:.1f}" y="{y:.1f}" class="fig-small fig-bold" '
            f'text-anchor="end">{label}</text>'
        )
    for row in range(length):
        for col in range(length):
            allowed = mask[row, col] == 1.0
            if not allowed:
                fill = "fill-sunk"
            elif row < 3:
                fill = "fill-soft-a"
            else:
                fill = "fill-soft-b"
            x = origin_x + col * cell
            y = origin_y + row * cell
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell:.1f}" height="{cell:.1f}" '
                f'class="{fill} stroke-muted"/>'
            )
            parts.append(
                f'<text x="{x + cell / 2:.1f}" y="{y + cell / 2 + 4:.1f}" '
                f'class="fig-small fig-mono" text-anchor="middle">{int(mask[row, col])}</text>'
            )
    parts.append(
        f'<text x="{origin_x - 28.0:.1f}" y="150.0" class="fig-small" text-anchor="middle" '
        'transform="rotate(-90 80 150)">query</text>'
    )
    parts.append('<rect x="380.0" y="80.0" width="16.0" height="16.0" class="fill-soft-a stroke-muted"/>')
    parts.append(
        '<text x="404.0" y="93.0" class="fig-small" text-anchor="start">example A may attend</text>'
    )
    parts.append('<rect x="380.0" y="112.0" width="16.0" height="16.0" class="fill-soft-b stroke-muted"/>')
    parts.append(
        '<text x="404.0" y="125.0" class="fig-small" text-anchor="start">example B may attend</text>'
    )
    parts.append('<rect x="380.0" y="144.0" width="16.0" height="16.0" class="fill-sunk stroke-muted"/>')
    parts.append(
        '<text x="404.0" y="157.0" class="fig-small" text-anchor="start">blocked</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    masked, bare = example_contributions()
    losses = packing_losses()
    print("masked contributions", masked)
    print("unmasked contributions", bare)
    print("packing", {key: round(value, 10) for key, value in losses.items()})
    if abs(losses["packed"] - (losses["first"] + losses["second"])) > 1e-12:
        raise SystemExit("packed loss does not match the sum of the separate losses")
    if abs(losses["crossed"] - losses["packed"]) < 1e-6:
        raise SystemExit("the cross-boundary mask did not change the loss")
    print(figure_loss_bars(masked, bare))
    print(figure_mask())


if __name__ == "__main__":
    main()
