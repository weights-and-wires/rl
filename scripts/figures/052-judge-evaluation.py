#!/usr/bin/env python3
"""Numbers and SVG for the judge-evaluation lesson.

Figure 1 is a reliability diagram for a position-biased judge. The judge
states 0.85 when Mira is shown on the left and 0.15 when Mira is shown on
the right. Mira wins half the time in each group, so both points sit on
the 0.5 line. Expected calibration error with five equal-width bins is 0.35.
A bin-center variant of the same diagram scores 0.40.

Figure 2 is the length-controlled win rate of a length-only judge. The
length gaps are +4, +4, +4, +4, and -2 tokens, and the judge's win
probability is the sigmoid of 1.5 times the gap. The raw mean is about
0.808. Logistic regression with a ridge of 1e-6 on the length coefficient
returns a win rate at a gap of 0 that is within about 1e-6 of 0.5.

Run from the repo root:

    python3 scripts/figures/052-judge-evaluation.py
"""

from __future__ import annotations

import math


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


# Figure 1. Eight stated probabilities and eight outcomes.
PROB = (0.15, 0.15, 0.15, 0.15, 0.85, 0.85, 0.85, 0.85)
OUTCOME = (1, 0, 1, 0, 1, 0, 1, 0)
N_BINS = 5


def expected_calibration_error(
    prob: tuple[float, ...], outcome: tuple[int, ...], n_bins: int, center: bool
) -> float:
    """Equal-width ECE. The last bin includes 1. Empty bins add nothing.

    When `center` is true, confidence is the midpoint of the bin. The lesson
    uses the mean predicted probability instead, and the two disagree here.
    """
    n = len(prob)
    total = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        if b == n_bins - 1:
            chosen = [i for i, p in enumerate(prob) if lo <= p <= hi]
        else:
            chosen = [i for i, p in enumerate(prob) if lo <= p < hi]
        if not chosen:
            continue
        if center:
            conf = 0.5 * (lo + hi)
        else:
            conf = sum(prob[i] for i in chosen) / len(chosen)
        acc = sum(outcome[i] for i in chosen) / len(chosen)
        total += (len(chosen) / n) * abs(acc - conf)
    return total


# Figure 2. A length-only judge, population probabilities, not coin flips.
LENGTH_DIFF = (4.0, 4.0, 4.0, 4.0, -2.0)
LENGTH_COEF = 1.5
RIDGE = 1e-6
NEWTON_STEPS = 20


def length_controlled(length_diff: tuple[float, ...], win: tuple[float, ...]) -> tuple[float, float, float]:
    """Newton fit of logit P = b0 + b1 * length_diff. Ridge on b1 only."""
    b0, b1 = 0.0, 0.0
    for _ in range(NEWTON_STEPS):
        z = [b0 + b1 * x for x in length_diff]
        p = [sigmoid(zi) for zi in z]
        w = [max(pi * (1.0 - pi), 1e-12) for pi in p]
        g0 = sum(y - pi for y, pi in zip(win, p))
        g1 = sum((y - pi) * x for y, pi, x in zip(win, p, length_diff)) - RIDGE * b1
        h00 = -sum(w)
        h01 = -sum(wi * x for wi, x in zip(w, length_diff))
        h11 = -sum(wi * x * x for wi, x in zip(w, length_diff)) - RIDGE
        det = h00 * h11 - h01 * h01
        b0 += (-h11 * g0 + h01 * g1) / det
        b1 += (h01 * g0 - h00 * g1) / det
    return b0, b1, sigmoid(b0)


def reliability_svg() -> str:
    """SVG whose points are the two bins of the worked calibration example."""

    def x_of(p: float) -> float:
        return 90.0 + p * 470.0

    def y_of(a: float) -> float:
        return 210.0 - a * 174.0

    # Both groups have observed frequency 0.5. Predicted values are 0.15 and 0.85.
    points = ((0.15, 0.50), (0.85, 0.50))
    lines = [
        '<svg viewBox="0 0 620 300" role="img" aria-label="Reliability diagram. The judge predicts 0.15 and 0.85, but Mira wins half the time in both groups.">',
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y_of(tick)
        label = f"{tick:.2f}".rstrip("0").rstrip(".") if tick not in (0.0, 1.0) else f"{int(tick)}"
        if tick == 0.5:
            label = "0.5"
        lines.append(f'<line x1="90" y1="{y:.1f}" x2="560" y2="{y:.1f}" class="stroke-grid"/>')
        lines.append(
            f'<text x="82" y="{y + 4:.1f}" class="fig-small" text-anchor="end">{label}</text>'
        )
    for tick, label in ((0.0, "0"), (0.5, "0.5"), (1.0, "1")):
        x = x_of(tick)
        lines.append(f'<line x1="{x:.1f}" y1="210" x2="{x:.1f}" y2="216" class="stroke-muted"/>')
        lines.append(
            f'<text x="{x:.1f}" y="232" class="fig-small" text-anchor="middle">{label}</text>'
        )
    lines.append(
        f'<line x1="{x_of(0):.1f}" y1="{y_of(0):.1f}" x2="{x_of(1):.1f}" y2="{y_of(1):.1f}" class="stroke-muted dashed"/>'
    )
    for pred, obs in points:
        lines.append(
            f'<line x1="{x_of(pred):.1f}" y1="{y_of(pred):.1f}" x2="{x_of(pred):.1f}" y2="{y_of(obs):.1f}" class="stroke-bad"/>'
        )
        lines.append(f'<circle cx="{x_of(pred):.1f}" cy="{y_of(pred):.1f}" r="4" class="fill-a"/>')
        lines.append(f'<circle cx="{x_of(pred):.1f}" cy="{y_of(obs):.1f}" r="5" class="fill-b"/>')
    # Labels sit beside the observed points, clear of the diagonal.
    lines.append(
        f'<text x="{x_of(0.15) + 12:.1f}" y="{y_of(0.50) - 12:.1f}" class="fig-small fig-bold">0.15 predicted, 0.50 won</text>'
    )
    lines.append(
        f'<text x="{x_of(0.85) - 12:.1f}" y="{y_of(0.50) + 22:.1f}" class="fig-small fig-bold" text-anchor="end">0.85 predicted, 0.50 won</text>'
    )
    lines.append('<text x="325" y="258" class="fig-small" text-anchor="middle">stated probability that Mira wins</text>')
    lines.append('<text x="90" y="22" class="fig-small" text-anchor="start">how often Mira won</text>')
    lines.append('<line x1="90" y1="276" x2="112" y2="276" class="stroke-muted dashed"/>')
    lines.append('<text x="118" y="280" class="fig-small">calibrated</text>')
    lines.append('<circle cx="210" cy="276" r="4" class="fill-b"/>')
    lines.append('<text x="220" y="280" class="fig-small">observed</text>')
    lines.append('<circle cx="300" cy="276" r="4" class="fill-a"/>')
    lines.append('<text x="310" y="280" class="fig-small">stated probability</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def length_svg(raw: float, controlled: float) -> str:
    """Two bars: the raw win rate and the win rate at equal length."""

    def height(v: float) -> float:
        return v * 140.0

    baseline = 190.0
    raw_h = height(raw)
    ctl_h = height(controlled)
    half = baseline - height(0.5)
    lines = [
        '<svg viewBox="0 0 560 250" role="img" aria-label="Raw win rate near 0.81 beside the length-controlled win rate of 0.50.">',
        '<line x1="70" y1="50" x2="520" y2="50" class="stroke-grid"/>',
        '<text x="62" y="54" class="fig-small" text-anchor="end">1</text>',
        f'<line x1="70" y1="{half:.1f}" x2="520" y2="{half:.1f}" class="stroke-muted dashed"/>',
        f'<text x="62" y="{half + 4:.1f}" class="fig-small" text-anchor="end">0.5</text>',
        f'<line x1="70" y1="{baseline:.0f}" x2="520" y2="{baseline:.0f}" class="stroke-muted"/>',
        '<text x="62" y="194" class="fig-small" text-anchor="end">0</text>',
        f'<rect x="150" y="{baseline - raw_h:.1f}" width="80" height="{raw_h:.1f}" class="fill-b"/>',
        f'<rect x="330" y="{baseline - ctl_h:.1f}" width="80" height="{ctl_h:.1f}" class="fill-c"/>',
        f'<text x="190" y="{baseline - raw_h - 8:.1f}" class="fig-small fig-bold" text-anchor="middle">{raw:.3f}</text>',
        f'<text x="370" y="{baseline - ctl_h - 8:.1f}" class="fig-small fig-bold" text-anchor="middle">{controlled:.3f}</text>',
        '<text x="190" y="214" class="fig-small" text-anchor="middle">raw win rate</text>',
        '<text x="370" y="214" class="fig-small" text-anchor="middle">equal length</text>',
        '<text x="295" y="240" class="fig-small" text-anchor="middle">chance the judge prefers Mira</text>',
        "</svg>",
    ]
    return "\n".join(lines)


def main() -> None:
    ece = expected_calibration_error(PROB, OUTCOME, N_BINS, center=False)
    ece_center = expected_calibration_error(PROB, OUTCOME, N_BINS, center=True)
    print(f"ece={ece:.10f} ece_bin_center={ece_center:.10f}")

    # Label noise. True gap 2, flip probability 0.1.
    true_gap = 2.0
    eps = 0.1
    p = sigmoid(true_gap)
    p_noisy = (1.0 - 2.0 * eps) * p + eps
    print(f"sigma(2)={p:.6f} p_noisy={p_noisy:.6f} gap_fit={logit(p_noisy):.6f}")

    # Two-player closed form: 3 wins, 1 loss, 1 tie.
    gap = math.log(3.5 / 1.5)
    print(f"bt_gap={gap:.10f} s0={0.5 * gap:.10f} elo={(400 / math.log(10)) * gap:.6f}")
    print(f"tie_as_first_win s0={0.5 * math.log(4.0):.10f}")

    win = tuple(sigmoid(LENGTH_COEF * x) for x in LENGTH_DIFF)
    raw = sum(win) / len(win)
    b0, b1, controlled = length_controlled(LENGTH_DIFF, win)
    print(f"raw={raw:.6f} b0={b0:.6e} b1={b1:.6f} controlled={controlled:.10f}")
    print(f"win_probs={' '.join(f'{p:.6f}' for p in win)}")
    print("--- figure 1 ---")
    print(reliability_svg())
    print("--- figure 2 ---")
    print(length_svg(raw, controlled))


if __name__ == "__main__":
    main()
