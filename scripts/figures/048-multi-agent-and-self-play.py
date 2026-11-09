#!/usr/bin/env python3
"""Numbers and SVG for the many-agents lesson.

Figure 3 is fictitious play on the one-lane bridge, a 2 by 2 general-sum
game. Both players open on yield. The average weight on yield approaches
1/2, and the action played each round flips between yield and push.

Run from the repo root:

    python3 scripts/figures/048-multi-agent-and-self-play.py
"""

from __future__ import annotations

import numpy as np

# Rows and columns are (yield, push). The column matrix is the transpose of
# the row matrix, and it is not the negative of the row matrix.
BRIDGE_ROW = np.array([[2.0, 1.0], [3.0, 0.0]])
BRIDGE_COL = np.array([[2.0, 3.0], [1.0, 0.0]])
ROUNDS = 16
YIELD, PUSH = 0, 1


def fictitious_play(iters: int) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Empirical mixtures and the action pair played in each round.

    Round 1 plays yield against yield. Each later round, both players
    best-respond to the counts so far, and a tie goes to the lower index.
    """
    count_x = np.zeros(2)
    count_y = np.zeros(2)
    count_x[YIELD] += 1
    count_y[YIELD] += 1
    history = [(YIELD, YIELD)]
    for _ in range(iters - 1):
        row = int(np.argmax(BRIDGE_ROW @ (count_y / count_y.sum())))
        col = int(np.argmax((count_x / count_x.sum()) @ BRIDGE_COL))
        count_x[row] += 1
        count_y[col] += 1
        history.append((row, col))
    return count_x / iters, count_y / iters, history


def yield_weight(round_index: int) -> float:
    """Average weight on yield after `round_index` rounds, counting from 1."""
    if round_index % 2 == 0:
        return 0.5
    return (round_index // 2 + 1) / round_index


def figure_bridge(history: list[tuple[int, int]]) -> str:
    """SVG of the average weight on yield and the action played each round."""
    left, right = 78.0, 560.0
    top, bottom = 48.0, 214.0
    rounds = len(history)

    def x_of(round_index: int) -> float:
        return left + (round_index - 1) / (rounds - 1) * (right - left)

    def y_of(probability: float) -> float:
        return top + (1.0 - probability) * (bottom - top)

    def series(values: list[float]) -> str:
        return " ".join(f"{x_of(i):.1f},{y_of(p):.1f}" for i, p in enumerate(values, 1))

    yields = 0
    average = []
    for round_index, (row, _) in enumerate(history, 1):
        yields += int(row == YIELD)
        average.append(yields / round_index)
    # 1 when that round played yield, 0 when it played push.
    instant = [1.0 if pair[0] == YIELD else 0.0 for pair in history]
    parts = [
        '<svg viewBox="0 0 600 286" role="img" '
        'aria-label="Fictitious play on the bridge. The average weight on yield '
        'falls toward 1/2, while the action each round flips between yield and push.">',
        '<line x1="86" y1="24" x2="118" y2="24" class="stroke-a"/>',
        '<text x="124" y="28" class="fig-small">average strategy</text>',
        '<line x1="280" y1="24" x2="312" y2="24" class="stroke-b dashed"/>',
        '<text x="318" y="28" class="fig-small">action this round</text>',
    ]
    for tick, label in ((0.0, "0"), (0.5, "1/2"), (1.0, "1")):
        y = y_of(tick)
        parts.append(
            f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{right:.1f}" y2="{y:.1f}" class="stroke-grid"/>'
        )
        parts.append(
            f'<text x="{left - 8:.1f}" y="{y + 4:.1f}" class="fig-small" text-anchor="end">{label}</text>'
        )
    for tick in (1, 4, 8, 12, 16):
        x = x_of(tick)
        parts.append(
            f'<line x1="{x:.1f}" y1="{bottom:.1f}" x2="{x:.1f}" y2="{bottom + 6:.1f}" class="stroke-muted"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{bottom + 22:.1f}" class="fig-small" text-anchor="middle">{tick}</text>'
        )
    parts.append(
        f'<polyline points="{series(average)}" class="stroke-a" fill="none"/>'
    )
    parts.append(
        f'<polyline points="{series(instant)}" class="stroke-b dashed" fill="none"/>'
    )
    for i, probability in enumerate(average, 1):
        parts.append(
            f'<circle cx="{x_of(i):.1f}" cy="{y_of(probability):.1f}" r="3.5" class="fill-a"/>'
        )
    for i, probability in enumerate(instant, 1):
        parts.append(
            f'<circle cx="{x_of(i):.1f}" cy="{y_of(probability):.1f}" r="3.5" class="fill-b"/>'
        )
    parts.append(
        f'<text x="{(left + right) / 2:.1f}" y="276" class="fig-small" text-anchor="middle">round</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    _, _, history = fictitious_play(ROUNDS)
    print("plays (0 = yield, 1 = push)")
    for round_index, pair in enumerate(history, 1):
        weight = yield_weight(round_index)
        counted = sum(action == YIELD for action, _ in history[:round_index]) / round_index
        print(
            f"  round {round_index:2d}  play={pair}  "
            f"average yield={counted:.6f}  formula={weight:.6f}"
        )
    three = fictitious_play(3)[0]
    print("after 3 rounds", three, "expected", np.array([2 / 3, 1 / 3]))
    print("after 4 rounds", fictitious_play(4)[0])
    print("after 7 rounds", fictitious_play(7)[0], "expected", 4 / 7)
    print("---FIGURE3---")
    print(figure_bridge(history))


if __name__ == "__main__":
    main()
