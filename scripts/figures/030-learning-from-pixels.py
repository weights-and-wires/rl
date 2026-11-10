#!/usr/bin/env python3
"""Numbers and SVG for the learning-from-pixels lesson.

Figure 1 shows random_shift on a 5 x 5 image whose bright cells touch the left
edge. Figure 2 shows one shift applied to a Catch frame without and with a blank
margin. Figure 3 plots the catch rate of the solution's noisy DQN, trained with
and without shifts, over eight seeds. The offsets are the ones random_shift
would draw (rows, then columns) and the scores come from the exercise's
solution.py, so every number is computed.

Run from the repo root:

    python3 scripts/figures/030-learning-from-pixels.py > /tmp/fig029.txt

The output holds the three figure blocks, separated by lines of "====".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "content" / "exercises" / "030-learning-from-pixels"))
import solution as sol  # noqa: E402


def grid_svg(img, x0, y0, cell, ids):
    """Cells of a 2-D array; 1 draws as fill-a, 2 as fill-b, 0 as fill-sunk."""
    out = []
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            cls = {0: "fill-sunk", 1: "fill-a", 2: "fill-b"}[int(img[r, c])]
            out.append(f'<rect x="{x0 + c * cell}" y="{y0 + r * cell}" width="{cell - 1}" '
                       f'height="{cell - 1}" class="{cls}"/>')
    return "\n".join(out)


def crop(img, pad, dy, dx):
    h, w = img.shape
    padded = np.pad(img, pad, mode="edge")
    return padded[dy:dy + h, dx:dx + w]


def figure1():
    img = np.zeros((5, 5), dtype=int)
    img[1:4, 0] = 1
    img[3, 1] = 1
    shifts = [(1, 1), (1, 0), (1, 2), (0, 2)]  # (dy, dx), pad 1 centered at (1, 1)
    # Offsets chosen to show a no-op, a left step that smears, and a right step.
    panels = [("original", img, (1, 1))] + [
        (f"dy={dy - 1:+d}, dx={dx - 1:+d}", crop(img, 1, dy, dx), (dy, dx)) for dy, dx in shifts[1:]
    ]
    cell, parts = 22, []
    for i, (title, im, _) in enumerate(panels):
        x0 = 20 + i * 150
        parts.append(f'<text x="{x0}" y="20" class="fig-small">{title}</text>')
        parts.append(grid_svg(im, x0, 30, cell, i))
    svg = ('<svg viewBox="0 0 620 160" role="img" aria-label="A 5 by 5 image and three shifted '
           'copies, with the left edge column replicated when the window moves left">\n'
           + "\n".join(parts) + "\n</svg>")
    return svg + ("\nFigure 1. random_shift with pad=1 pads the image by copying its edge cells, then crops a "
                  "5 x 5 window. Moving the window left (dx=-1) duplicates the bright left column, and "
                  "moving it right (dx=+1) drops that column and leaves a blank one.")


def catch_frame(margin):
    env = sol.Catch(0)
    env.reset()
    env.ball_r, env.ball_c, env.paddle = 4, 1, 1
    h, w = sol.Catch.H, sol.Catch.W
    img = np.zeros((h + 2 * margin, w + 2 * margin), dtype=int)
    img[margin + env.ball_r, margin + env.ball_c] = 2
    img[margin + h - 1, margin + env.paddle] = 1
    return img


def figure2():
    cell, parts = 14, []
    titles = ["no margin", "no margin, shifted down", "blank margin", "blank margin, shifted down"]
    frames = []
    for margin in (0, 1):
        base = catch_frame(margin)
        frames.append(base)
        # A window offset of 0 with pad 1 moves the content down by 1 row.
        frames.append(crop(base, 1, 0, 1))
    for i, (t, im) in enumerate(zip(titles, frames)):
        x0 = 15 + i * 150
        parts.append(f'<text x="{x0}" y="16" class="fig-small">{t}</text>')
        parts.append(grid_svg(im, x0, 24, cell, i))
    lost = int((frames[1] == 1).sum()) == 0
    kept = int((frames[3] == 1).sum()) == 1
    assert lost and kept
    svg = ('<svg viewBox="0 0 620 190" role="img" aria-label="A Catch frame shifted down one row; '
           'without a blank margin the paddle leaves the image, with a margin it stays">\n'
           + "\n".join(parts) + "\n</svg>")
    return svg + ("\nFigure 2. The ball (orange) and paddle (blue) in a Catch frame. Without a margin, a one-row "
                  "shift down pushes the paddle out of the window. With a one-pixel blank margin, the same "
                  "shift keeps both objects.")


def figure3():
    rates = {}
    for augment in (False, True):
        rates[augment] = [sol.evaluate_catch(sol.train_catch(seed, augment), episodes=300)
                          for seed in range(8)]
    parts = ['<line x1="70" y1="130" x2="590" y2="130" class="stroke-muted"/>']
    for v in (0.0, 0.5, 1.0):
        y = 130 - 100 * v
        parts.append(f'<line x1="70" y1="{y}" x2="590" y2="{y}" class="stroke-grid"/>')
        parts.append(f'<text x="62" y="{y + 4}" text-anchor="end" class="fig-small">{v:.1f}</text>')
    y_rand = 130 - 100 * 0.2
    parts.append(f'<line x1="70" y1="{y_rand}" x2="590" y2="{y_rand}" class="stroke-bad dashed"/>')
    parts.append(f'<text x="590" y="{y_rand - 5}" text-anchor="end" class="fig-small">random policy 0.20</text>')
    for k, (augment, cls, x0) in enumerate(((False, "fill-a", 160), (True, "fill-b", 400))):
        for i, v in enumerate(rates[augment]):
            x = x0 + (i - 3.5) * 22
            parts.append(f'<circle cx="{x}" cy="{130 - 100 * v:.1f}" r="5" class="{cls}"/>')
        mean = float(np.mean(rates[augment]))
        label = "no shifts" if not augment else "random shifts"
        parts.append(f'<text x="{x0}" y="152" text-anchor="middle" class="fig-bold">{label}</text>')
        parts.append(f'<text x="{x0}" y="170" text-anchor="middle" class="fig-small">mean {mean:.2f}</text>')
    svg = ('<svg viewBox="0 0 620 185" role="img" aria-label="Catch rate of eight trained agents with '
           'and without random shifts, against the random policy rate of 0.20">\n'
           + "\n".join(parts) + "\n</svg>")
    print("rates", {k: np.round(v, 2).tolist() for k, v in rates.items()}, file=sys.stderr)
    return svg + ("\nFigure 3. Catch rate over 300 evaluation episodes for eight seeds, after 4000 steps. "
                  "Both versions beat the random policy's 0.20 on every seed. Shifts did not beat plain DQN "
                  "on this toy.")


if __name__ == "__main__":
    print(figure1())
    print("====")
    print(figure2())
    print("====")
    print(figure3())
