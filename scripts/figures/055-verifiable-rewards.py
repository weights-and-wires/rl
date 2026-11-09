#!/usr/bin/env python3
"""Numbers and SVG for the verifiable-rewards lesson.

Figure 1 draws a Wilson score interval for each of several pass counts out of
32 samples. The shaded band is the informative range [0.1, 0.9]. Every
coordinate is computed below.

Run from the repo root:

    python3 scripts/figures/055-verifiable-rewards.py
"""

from __future__ import annotations

import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return the Wilson score interval (low, high) for successes out of n."""
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n))
    return center - margin, center + margin


# Worked example in the lesson: 3 correct samples out of 8.
EXAMPLE_N = 8
EXAMPLE_K = 3
EXAMPLE_Z = 1.96

# Figure 1. Same z, a larger sample, several success counts.
FIG_N = 32
FIG_K = (0, 2, 4, 8, 16, 24, 28, 30, 32)
BAND = (0.1, 0.9)

# Partial-credit group in the lesson. One boxed correct answer, five boxed
# wrong answers, two answers with no box. format_credit is lambda.
GROUP_PATTERN = (1, 1, 1, 1, 1, 1, 0, 0)  # 1 means a box is present
N_CORRECT_BOXED = 1


def centered(rewards: list[float]) -> tuple[float, list[float]]:
    mean = sum(rewards) / len(rewards)
    return mean, [reward - mean for reward in rewards]


def rewards_for(format_credit: float) -> list[float]:
    """Reward 1 + format_credit for the first boxed answer, format_credit for
    the other boxed answers, and 0 when the box is missing.
    """
    out = []
    correct_left = N_CORRECT_BOXED
    for boxed in GROUP_PATTERN:
        if not boxed:
            out.append(0.0)
            continue
        if correct_left:
            out.append(1.0 + format_credit)
            correct_left -= 1
        else:
            out.append(format_credit)
    return out


def main() -> None:
    low, high = wilson_interval(EXAMPLE_K, EXAMPLE_N, EXAMPLE_Z)
    print(f"example n={EXAMPLE_N} k={EXAMPLE_K} z={EXAMPLE_Z}")
    print(f"  phat={EXAMPLE_K / EXAMPLE_N:.6f} low={low:.6f} high={high:.6f}")

    print(f"figure n={FIG_N} band={BAND}")
    rows = []
    for k in FIG_K:
        lo, hi = wilson_interval(k, FIG_N)
        inside = lo >= BAND[0] and hi <= BAND[1]
        rows.append((k, k / FIG_N, lo, hi, inside))
        print(
            f"  k={k:2d} phat={k / FIG_N:.4f} "
            f"low={lo:.4f} high={hi:.4f} inside={inside}"
        )

    for lam in (0.1, 0.9):
        rewards = rewards_for(lam)
        mean, adv = centered(rewards)
        print(f"lambda={lam} rewards={rewards} mean={mean:.4f}")
        print(f"  centered={['%.4f' % a for a in adv]}")

    print()
    print(svg(rows))


def svg(rows: list[tuple[int, float, float, float, bool]]) -> str:
    """One interval per success count. x is the pass rate."""
    left, right = 118.0, 560.0
    top, row_h = 46.0, 26.0
    width = right - left

    def x_of(p: float) -> float:
        return left + p * width

    lines = [
        '<svg viewBox="0 0 620 330" role="img" '
        'aria-label="Wilson intervals for 0 to 32 successes out of 32 samples, '
        'with the band from 0.1 to 0.9 shaded">',
        "",
    ]
    # Shaded informative band.
    band_x = x_of(BAND[0])
    band_w = x_of(BAND[1]) - band_x
    plot_top = top - 8
    plot_bot = top + (len(rows) - 1) * row_h + 14
    lines.append(
        f'<rect x="{band_x:.1f}" y="{plot_top:.1f}" width="{band_w:.1f}" '
        f'height="{plot_bot - plot_top:.1f}" class="fill-soft-c"/>'
    )
    # Axis and ticks at 0, 0.1, 0.5, 0.9, 1.
    axis_y = plot_bot + 8
    lines.append(
        f'<line x1="{left:.1f}" y1="{axis_y:.1f}" x2="{right:.1f}" y2="{axis_y:.1f}" '
        'class="stroke-muted"/>'
    )
    for tick, label in ((0.0, "0"), (0.1, "0.1"), (0.5, "0.5"), (0.9, "0.9"), (1.0, "1")):
        tx = x_of(tick)
        lines.append(
            f'<line x1="{tx:.1f}" y1="{axis_y:.1f}" x2="{tx:.1f}" y2="{axis_y + 5:.1f}" '
            'class="stroke-muted"/>'
        )
        lines.append(
            f'<text x="{tx:.1f}" y="{axis_y + 20:.1f}" text-anchor="middle" '
            f'class="fig-small">{label}</text>'
        )
    lines.append(
        f'<text x="{(left + right) / 2:.1f}" y="{axis_y + 40:.1f}" text-anchor="middle">'
        "estimated pass rate</text>"
    )
    lines.append(
        '<text x="300" y="22" text-anchor="middle" class="fig-bold">'
        "32 samples, Wilson interval, z = 1.96</text>"
    )

    for index, (k, phat, lo, hi, inside) in enumerate(rows):
        y = top + index * row_h
        stroke = "stroke-c" if inside else "stroke-bad"
        fill = "fill-c" if inside else "fill-bad"
        lines.append(
            f'<text x="110" y="{y + 4:.1f}" text-anchor="end" class="fig-small">'
            f"{k}/32</text>"
        )
        lines.append(
            f'<line x1="{x_of(lo):.1f}" y1="{y:.1f}" x2="{x_of(hi):.1f}" y2="{y:.1f}" '
            f'class="{stroke}"/>'
        )
        # End caps.
        for end in (lo, hi):
            lines.append(
                f'<line x1="{x_of(end):.1f}" y1="{y - 5:.1f}" x2="{x_of(end):.1f}" '
                f'y2="{y + 5:.1f}" class="{stroke}"/>'
            )
        lines.append(
            f'<circle cx="{x_of(phat):.1f}" cy="{y:.1f}" r="3.5" class="{fill}"/>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
