#!/usr/bin/env python3
"""Numbers and SVG for the lesson on RL for vision, speech, and robots.

Figures 1 and 2 (IoU boxes and the MWER beam) use the hand-worked numbers in
the lesson and exercise tests. Figure 3 is the action-token round trip: one
action dimension on [-1, 1] split into 8 equal bins, and the decoding error
(bin centre minus the action) swept across the range. Every plotted
coordinate is computed below.

Run from the repo root:

    python3 scripts/figures/063-rl-for-vision-speech-and-robots.py
"""

from __future__ import annotations

import numpy as np

LOW, HIGH, N_BINS = -1.0, 1.0, 8
EXAMPLE = 0.30


def action_to_tokens(action, low, high, n_bins):
    a = np.clip(np.asarray(action, dtype=float), low, high)
    tokens = np.floor((a - low) / ((high - low) / n_bins)).astype(int)
    return np.clip(tokens, 0, n_bins - 1)


def tokens_to_action(tokens, low, high, n_bins):
    return low + (np.asarray(tokens, dtype=float) + 0.5) * (high - low) / n_bins


def round_trip_error(action, n_bins=N_BINS):
    """Decoded bin centre minus the action."""
    tokens = action_to_tokens(action, LOW, HIGH, n_bins)
    return tokens_to_action(tokens, LOW, HIGH, n_bins) - action


# Plot frame: action on x, signed error on y.
X0, X1 = 60.0, 570.0
Y_MID, Y_SCALE = 150.0, 800.0  # 0.125 of error is 100 units


def x_of(a):
    return X0 + (a - LOW) / (HIGH - LOW) * (X1 - X0)


def y_of(err):
    return Y_MID - err * Y_SCALE


def figure_round_trip():
    width = (HIGH - LOW) / N_BINS
    half = width / 2
    parts = [
        '<svg viewBox="0 0 600 300" role="img" aria-label="Round-trip error of an 8-bin action token on the range -1 to 1. '
        'The error falls from plus 0.125 to minus 0.125 inside each bin and never leaves the half-bin bound.">',
        f'<line x1="{X0:.1f}" y1="{y_of(0):.1f}" x2="{X1:.1f}" y2="{y_of(0):.1f}" class="stroke-muted"/>',
        f'<line x1="{X0:.1f}" y1="{y_of(half):.1f}" x2="{X1:.1f}" y2="{y_of(half):.1f}" class="stroke-bad dashed"/>',
        f'<line x1="{X0:.1f}" y1="{y_of(-half):.1f}" x2="{X1:.1f}" y2="{y_of(-half):.1f}" class="stroke-bad dashed"/>',
        f'<text x="{X0 + 4:.1f}" y="{y_of(half) - 6:.1f}" class="fig-small">+half a bin = +{half:.3f}</text>',
        f'<text x="{X0 + 4:.1f}" y="{y_of(-half) + 16:.1f}" class="fig-small">-half a bin = -{half:.3f}</text>',
    ]
    for k in range(N_BINS):
        left = LOW + k * width
        # Inside bin k the decoded value is fixed, so the error falls with slope -1:
        # +half at the left edge, and -half in the limit at the right edge.
        e_left = float(tokens_to_action(k, LOW, HIGH, N_BINS) - left)
        e_right = float(tokens_to_action(k, LOW, HIGH, N_BINS) - (left + width))
        parts.append(
            f'<line x1="{x_of(left):.1f}" y1="{y_of(e_left):.1f}" x2="{x_of(left + width):.1f}" '
            f'y2="{y_of(e_right):.1f}" class="stroke-a"/>'
        )
    token = int(action_to_tokens(EXAMPLE, LOW, HIGH, N_BINS))
    err = float(round_trip_error(EXAMPLE))
    parts.append(f'<circle cx="{x_of(EXAMPLE):.1f}" cy="{y_of(err):.1f}" r="5" class="fill-b"/>')
    parts.append(
        f'<text x="{x_of(EXAMPLE) + 10:.1f}" y="{y_of(err) - 8:.1f}" class="fig-small">'
        f'a = {EXAMPLE}: token {token}, error {err:+.3f}</text>'
    )
    parts.append(f'<text x="{X0:.1f}" y="280" class="fig-small">-1</text>')
    parts.append(f'<text x="{x_of(0):.1f}" y="280" text-anchor="middle" class="fig-small">0</text>')
    parts.append(f'<text x="{X1:.1f}" y="280" text-anchor="end" class="fig-small">1  (action a)</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main():
    width = (HIGH - LOW) / N_BINS
    print(f"range [{LOW}, {HIGH}], {N_BINS} bins, width {width}, half-width {width / 2}")
    token = int(action_to_tokens(EXAMPLE, LOW, HIGH, N_BINS))
    decoded = float(tokens_to_action(token, LOW, HIGH, N_BINS))
    print(f"a = {EXAMPLE}: (a - low) / width = {(EXAMPLE - LOW) / width:.2f}, token {token}, "
          f"decoded {decoded}, error {decoded - EXAMPLE:+.3f}")
    sweep = np.linspace(LOW, HIGH, 2001)
    errs = np.abs(round_trip_error(sweep))
    print(f"sweep of {len(sweep)} actions: max |error| = {errs.max():.6f}, bound {width / 2}")
    print("endpoints -> tokens", action_to_tokens(np.array([LOW, HIGH]), LOW, HIGH, N_BINS))
    print("clip -5, 3 -> tokens", action_to_tokens(np.array([-5.0, 3.0]), LOW, HIGH, N_BINS))
    left_edge = tokens_to_action(action_to_tokens(sweep, LOW, HIGH, N_BINS), LOW, HIGH, N_BINS) - width / 2
    print(f"decoding to the left edge instead: max |error| = {np.abs(left_edge - sweep).max():.6f} (a full bin)")
    print(f"256 bins on [-1, 1]: width {2 / 256:.7f}, half-width {1 / 256:.7f}")
    print("---FIGURE3---")
    print(figure_round_trip())


if __name__ == "__main__":
    main()
