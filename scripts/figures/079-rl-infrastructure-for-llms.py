#!/usr/bin/env python3
"""SVG for Figure 3 of the asynchronous RL lesson.

Four batches, generation time 1, training time 2, run with K = 0 (the
synchronous loop) and K = 1. The box times are recomputed here with the same
rules as the exercise's `async_schedule`, and the script asserts that its
makespan and staleness match the solution's. Paste the printed SVG into the
lesson.

Run from the repo root:

    python3 scripts/figures/079-rl-infrastructure-for-llms.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "content/exercises/079-rl-infrastructure-for-llms")]

import solution as S  # noqa: E402

N, G, T = 4, 1.0, 2.0
X0, SCALE = 70, 42.0


def boxes(k):
    """(sampler start, update start, staleness) per batch, plus the makespan."""
    finish, rows = [], []
    sampler_free = trainer_free = 0.0
    for j in range(N):
        start = sampler_free
        if j - k - 1 >= 0:
            start = max(start, finish[j - k - 1])
        version = sum(1 for f in finish if f <= start)
        sampler_free = start + G
        u_start = max(sampler_free, trainer_free)
        trainer_free = u_start + T
        finish.append(trainer_free)
        rows.append((start, u_start, j - version))
    return rows, trainer_free


def lane(k, top, title):
    rows, end = boxes(k)
    out_ref = S.async_schedule(N, G, T, k)
    assert abs(end - out_ref["makespan"]) < 1e-12
    assert [r[2] for r in rows] == out_ref["staleness"]
    util = out_ref["trainer_utilization"]
    parts = [f'  <text x="20" y="{top}" class="fig-bold">{title}: trainer busy {util:.0%}</text>']
    parts.append(f'  <text x="20" y="{top + 38}" class="fig-small">sampler</text>')
    parts.append(f'  <text x="20" y="{top + 76}" class="fig-small">trainer</text>')
    for j, (s, u, stale) in enumerate(rows):
        x, w = X0 + s * SCALE, G * SCALE
        parts.append(f'  <rect x="{x:.0f}" y="{top + 22}" width="{w:.0f}" height="24" class="fill-a"/>')
        parts.append(f'  <text x="{x + w / 2:.0f}" y="{top + 39}" text-anchor="middle" class="fig-small">b{j}</text>')
        x, w = X0 + u * SCALE, T * SCALE
        parts.append(f'  <rect x="{x:.0f}" y="{top + 56}" width="{w - 2:.0f}" height="24" class="fill-b"/>')
        parts.append(f'  <text x="{x + w / 2:.0f}" y="{top + 73}" text-anchor="middle" class="fig-small">u{j}, stale {stale}</text>')
    return "\n".join(parts), end


def main():
    a, end0 = lane(0, 28, "K = 0, synchronous")
    b, end1 = lane(1, 140, "K = 1, asynchronous")
    ax = 250
    ticks = []
    for t in range(0, int(end0) + 1, 2):
        x = X0 + t * SCALE
        ticks.append(f'  <line x1="{x:.0f}" y1="{ax - 6}" x2="{x:.0f}" y2="{ax}" class="stroke-muted"/>')
        ticks.append(f'  <text x="{x:.0f}" y="{ax + 16}" text-anchor="middle" class="fig-small">{t}</text>')
    svg = [
        '<svg viewBox="0 0 600 285" role="img" aria-label="Timelines of one sampler and one trainer for four batches. With K = 0 the machines take turns and the run ends at time 12. With K = 1 they overlap and the run ends at time 9.">',
        a, b,
        f'  <line x1="{X0}" y1="{ax}" x2="{X0 + end0 * SCALE:.0f}" y2="{ax}" class="stroke-muted"/>',
        *ticks,
        f'  <text x="{X0 + end0 * SCALE:.0f}" y="{ax + 32}" text-anchor="end" class="fig-small">time</text>',
        "</svg>",
    ]
    print("\n".join(svg))


if __name__ == "__main__":
    main()
