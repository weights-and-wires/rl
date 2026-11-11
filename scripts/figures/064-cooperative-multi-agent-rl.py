#!/usr/bin/env python3
"""Numbers and SVG for the cooperative multi-agent lesson.

Figure 1 is the climbing game with each action's average payoff against a
uniform partner. Figure 2 is one independent-learner run on it: agent 1's
three action values over 3000 steps (seed 0, epsilon 0.4, alpha 0.05).

Run from the repo root:

    python3 scripts/figures/064-cooperative-multi-agent-rl.py
"""

from __future__ import annotations

import numpy as np

CLIMBING = np.array([[11.0, -30.0, 0.0], [-30.0, 7.0, 6.0], [0.0, 0.0, 5.0]])
STEPS, ALPHA, EPSILON, SEED = 3000, 0.05, 0.4, 0
SAMPLE_EVERY = 50


def independent_run() -> np.ndarray:
    """Agent 1's Q values, sampled every SAMPLE_EVERY steps. Same loop as the exercise."""
    rng = np.random.default_rng(SEED)
    qs = [np.zeros(3), np.zeros(3)]
    trace = [qs[0].copy()]
    for step in range(1, STEPS + 1):
        acts = []
        for q in qs:
            if rng.random() < EPSILON:
                acts.append(int(rng.integers(len(q))))
            else:
                acts.append(int(np.argmax(q)))
        r = CLIMBING[acts[0], acts[1]]
        for q, a in zip(qs, acts):
            q[a] += ALPHA * (r - q[a])
        if step % SAMPLE_EVERY == 0:
            trace.append(qs[0].copy())
    return np.array(trace)


def figure_game() -> str:
    """Payoff grid on the left, mean payoff per row (uniform partner) as bars on the right."""
    cell, x0, y0 = 70.0, 90.0, 60.0
    means = CLIMBING.mean(axis=1)
    parts = [
        '<svg viewBox="0 0 600 300" role="img" aria-label="The climbing game payoff grid, and each row\'s '
        'mean payoff against a uniform partner: -6.3, -5.7, and 1.7">',
        f'<text x="{x0 + 1.5 * cell:.1f}" y="38" class="fig-small fig-bold" text-anchor="middle">agent 2 plays</text>',
        f'<text x="{x0 - 56:.1f}" y="{y0 + 1.5 * cell:.1f}" class="fig-small fig-bold" text-anchor="middle" '
        f'transform="rotate(-90 {x0 - 56:.1f} {y0 + 1.5 * cell:.1f})">agent 1 plays</text>',
    ]
    for k in range(3):
        parts.append(f'<text x="{x0 + (k + 0.5) * cell:.1f}" y="{y0 - 6:.1f}" class="fig-small" text-anchor="middle">{k}</text>')
        parts.append(f'<text x="{x0 - 10:.1f}" y="{y0 + (k + 0.5) * cell + 4:.1f}" class="fig-small" text-anchor="end">{k}</text>')
    for i in range(3):
        for j in range(3):
            value = CLIMBING[i, j]
            klass = "fill-soft-c" if (i, j) == (0, 0) else ("fill-soft-b" if (i, j) == (2, 2) else "fill-sunk")
            parts.append(
                f'<rect x="{x0 + j * cell:.1f}" y="{y0 + i * cell:.1f}" width="{cell:.1f}" height="{cell:.1f}" '
                f'class="{klass} stroke-muted"/>'
            )
            parts.append(
                f'<text x="{x0 + (j + 0.5) * cell:.1f}" y="{y0 + (i + 0.5) * cell + 5:.1f}" '
                f'class="fig-bold" text-anchor="middle">{value:g}</text>'
            )
    # Bars: mean payoff per row, zero line at bx0, scale 12 units per payoff.
    bx0, scale = 400.0, 12.0
    parts.append(f'<text x="{bx0 + 40:.1f}" y="38" class="fig-small fig-bold" text-anchor="middle">mean payoff, uniform partner</text>')
    parts.append(f'<line x1="{bx0:.1f}" y1="{y0:.1f}" x2="{bx0:.1f}" y2="{y0 + 3 * cell:.1f}" class="stroke-muted"/>')
    for i, m in enumerate(means):
        y = y0 + i * cell + 20
        w = abs(m) * scale
        x = bx0 if m >= 0 else bx0 - w
        klass = "fill-b" if i == 2 else "fill-a"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="30" class="{klass}"/>')
        tx = bx0 + w + 6 if m >= 0 else bx0 + 6
        parts.append(f'<text x="{tx:.1f}" y="{y + 20:.1f}" class="fig-small" text-anchor="start">{m:.2f}</text>')
    parts.append('<text x="300" y="290" class="fig-small" text-anchor="middle">green: the best joint action. orange: the safe one.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def figure_run(trace: np.ndarray) -> str:
    left, right, top, bottom = 70.0, 570.0, 30.0, 220.0
    lo, hi = -8.0, 8.0
    n = len(trace)

    def x_of(i: int) -> float:
        return left + i / (n - 1) * (right - left)

    def y_of(v: float) -> float:
        return bottom - (min(max(v, lo), hi) - lo) / (hi - lo) * (bottom - top)

    parts = [
        '<svg viewBox="0 0 600 280" role="img" aria-label="One independent learner\'s three action values over '
        '3000 steps. The value of action 2 ends highest.">'
    ]
    for tick in (-8, -4, 0, 4, 8):
        y = y_of(tick)
        parts.append(f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{right:.1f}" y2="{y:.1f}" class="{"stroke-muted" if tick == 0 else "stroke-grid"}"/>')
        parts.append(f'<text x="{left - 8:.1f}" y="{y + 4:.1f}" class="fig-small" text-anchor="end">{tick}</text>')
    for step in (0, 1000, 2000, 3000):
        x = x_of(step // SAMPLE_EVERY)
        parts.append(f'<text x="{x:.1f}" y="{bottom + 20:.1f}" class="fig-small" text-anchor="middle">{step}</text>')
    for k, klass in enumerate(("stroke-a", "stroke-d", "stroke-b")):
        pts = " ".join(f"{x_of(i):.1f},{y_of(trace[i, k]):.1f}" for i in range(n))
        parts.append(f'<polyline points="{pts}" class="{klass}" fill="none"/>')
        parts.append(
            f'<text x="{right + 4:.1f}" y="{y_of(trace[-1, k]) + 4:.1f}" class="fig-small" text-anchor="start">action {k}</text>'
        )
    parts.append(f'<text x="{(left + right) / 2:.1f}" y="270" class="fig-small" text-anchor="middle">step</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    print("row means, uniform partner:", CLIMBING.mean(axis=1))
    print("row values, partner always on action 2:", CLIMBING[:, 2])
    trace = independent_run()
    print("final Q1:", trace[-1], "max over run of Q1[0]:", trace[:, 0].max())
    print("---FIGURE1---")
    print(figure_game())
    print("---FIGURE2---")
    print(figure_run(trace))


if __name__ == "__main__":
    main()
