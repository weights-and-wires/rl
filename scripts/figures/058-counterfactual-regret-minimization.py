#!/usr/bin/env python3
"""Numbers and SVG for the counterfactual regret minimization lesson.

Figure 1 draws the 12 information sets of Kuhn poker. Each cell shows the
probability that CFR's average strategy bets (or calls) after 10,000
iterations. Figure 2 shows exploitability against iterations for the average
strategy and for the current strategy. Every number comes from the reference
solution of the exercise, so the figures and the exercise agree.

Run from the repo root:

    python3 scripts/figures/058-counterfactual-regret-minimization.py
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOLUTION = ROOT / "content/exercises/058-counterfactual-regret-minimization/solution.py"
spec = importlib.util.spec_from_file_location("kuhn_solution", SOLUTION)
kuhn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kuhn)

CARD_NAMES = ("J", "Q", "K")
ITERS = 10000
CHECKPOINTS = (10, 30, 100, 300, 1000, 3000, 10000)


def figure_infosets(average: dict[str, np.ndarray]) -> str:
    """SVG grid: one row per history, one column per card."""
    rows = [
        ("", "player 1, first move", "bet"),
        ("p", "player 2, after pass", "bet"),
        ("b", "player 2, after bet", "call"),
        ("pb", "player 1, after pass, bet", "call"),
    ]
    left, top, cell_w, cell_h = 190.0, 50.0, 118.0, 56.0
    parts = [
        '<svg viewBox="0 0 580 300" role="img" aria-label="The 12 information sets of Kuhn poker, '
        'with the probability that the average strategy bets or calls in each">',
    ]
    for c, name in enumerate(CARD_NAMES):
        x = left + c * cell_w + cell_w / 2
        parts.append(f'<text x="{x:.1f}" y="34" class="fig-bold" text-anchor="middle">hold {name}</text>')
    for r, (history, label, verb) in enumerate(rows):
        y = top + r * cell_h
        player = kuhn.to_move(history)
        shown = f"history: {history}" if history else "history: (none)"
        parts.append(f'<text x="{left - 10:.1f}" y="{y + 24:.1f}" class="fig-small fig-bold" text-anchor="end">{label}</text>')
        parts.append(f'<text x="{left - 10:.1f}" y="{y + 42:.1f}" class="fig-small" text-anchor="end">{shown}</text>')
        for c in range(3):
            key = kuhn.infoset_key(c, history)
            p = float(average[key][kuhn.BET])
            fill = "fill-soft-a" if player == 0 else "fill-soft-b"
            stroke = "stroke-a" if player == 0 else "stroke-b"
            x = left + c * cell_w
            parts.append(
                f'<rect x="{x + 4:.1f}" y="{y + 4:.1f}" width="{cell_w - 8:.1f}" height="{cell_h - 8:.1f}" '
                f'rx="6" class="{fill} {stroke}"/>'
            )
            parts.append(
                f'<text x="{x + cell_w / 2:.1f}" y="{y + 25:.1f}" class="fig-small" text-anchor="middle">{verb}</text>'
            )
            parts.append(
                f'<text x="{x + cell_w / 2:.1f}" y="{y + 43:.1f}" class="fig-bold" text-anchor="middle">{p:.2f}</text>'
            )
    parts.append("</svg>")
    return "\n".join(parts)


def figure_exploitability(points: list[tuple[int, float, float]]) -> str:
    """SVG of exploitability on a log scale, against iterations on a log scale."""
    left, right, top, bottom = 80.0, 560.0, 30.0, 230.0
    lo_x, hi_x = math.log10(10), math.log10(10000)
    lo_y, hi_y = math.log10(0.001), math.log10(1.0)

    def x_of(n: float) -> float:
        return left + (math.log10(n) - lo_x) / (hi_x - lo_x) * (right - left)

    def y_of(v: float) -> float:
        return bottom - (math.log10(max(v, 0.001)) - lo_y) / (hi_y - lo_y) * (bottom - top)

    parts = [
        '<svg viewBox="0 0 600 280" role="img" aria-label="Exploitability of CFR on Kuhn poker over 10 to 10,000 '
        'iterations: the average strategy falls, the current strategy stays high">',
    ]
    for tick in (0.001, 0.01, 0.1, 1.0):
        y = y_of(tick)
        parts.append(f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{right:.1f}" y2="{y:.1f}" class="stroke-grid"/>')
        parts.append(f'<text x="{left - 8:.1f}" y="{y + 4:.1f}" class="fig-small" text-anchor="end">{tick:g}</text>')
    for tick in (10, 100, 1000, 10000):
        x = x_of(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{bottom:.1f}" x2="{x:.1f}" y2="{bottom + 5:.1f}" class="stroke-muted"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{bottom + 20:.1f}" class="fig-small" text-anchor="middle">{tick:,}</text>'
        )
    parts.append(f'<line x1="{left:.1f}" y1="{bottom:.1f}" x2="{right:.1f}" y2="{bottom:.1f}" class="stroke-muted"/>')
    avg = " ".join(f"{x_of(n):.1f},{y_of(a):.1f}" for n, a, _ in points)
    cur = " ".join(f"{x_of(n):.1f},{y_of(c):.1f}" for n, _, c in points)
    parts.append(f'<polyline points="{avg}" class="stroke-a" fill="none"/>')
    parts.append(f'<polyline points="{cur}" class="stroke-b dashed" fill="none"/>')
    for n, a, c in points:
        parts.append(f'<circle cx="{x_of(n):.1f}" cy="{y_of(a):.1f}" r="4" class="fill-a"/>')
        parts.append(f'<circle cx="{x_of(n):.1f}" cy="{y_of(c):.1f}" r="4" class="fill-b"/>')
    parts.append(f'<text x="{x_of(300):.1f}" y="{y_of(0.03) + 30:.1f}" class="fig-small fig-bold">average strategy</text>')
    parts.append(f'<text x="{x_of(12):.1f}" y="{y_of(0.9) - 4:.1f}" class="fig-small fig-bold">current strategy</text>')
    parts.append(
        f'<text x="{(left + right) / 2:.1f}" y="274" class="fig-small" text-anchor="middle">iterations T (log scale)</text>'
    )
    parts.append(f'<text x="{left:.1f}" y="18" class="fig-small" text-anchor="start">exploitability (log scale)</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    # One-iteration regret at the worked example, player 1 holding the queen after pass, bet.
    keys = kuhn.infosets()
    regret_sum = {k: np.zeros(2) for k in keys}
    strategy_sum = {k: np.zeros(2) for k in keys}
    value = kuhn.cfr_iteration(regret_sum, strategy_sum)
    print("iteration 1 value to player 1:", value)
    print("iteration 1 regret at '2' (king, first move):", regret_sum["2"], "expected [-1/8, 1/8]")
    print("iteration 1 regret at '1pb':", regret_sum["1pb"], "expected [-1/12, 1/12]")

    points = []
    for n in CHECKPOINTS:
        out = kuhn.cfr_kuhn(n)
        points.append((n, kuhn.exploitability(out["average"]), kuhn.exploitability(out["current"])))
        print(f"T={n:6d}  average {points[-1][1]:.5f}  current {points[-1][2]:.5f}")
    out = kuhn.cfr_kuhn(ITERS)
    avg = out["average"]
    print("game value of the average strategy:", kuhn.game_value(avg), "vs -1/18 =", -1 / 18)
    alpha = avg["0"][1]
    print(f"alpha {alpha:.4f}  king bet {avg['2'][1]:.4f}  3*alpha {3 * alpha:.4f}  "
          f"queen call after pb {avg['1pb'][1]:.4f}  alpha+1/3 {alpha + 1 / 3:.4f}")
    print("---FIGURE1---")
    print(figure_infosets(avg))
    print("---FIGURE2---")
    print(figure_exploitability(points))


if __name__ == "__main__":
    main()
