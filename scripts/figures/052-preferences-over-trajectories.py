#!/usr/bin/env python3
"""Numbers and SVG for the preferences-over-trajectories lesson.

The computation is the lesson's exercise at full size. A 4x4 grid world with
a wall, a hole, and a goal has the true reward -0.15 per step of distance to
the goal, +1 at the goal, and -1 at the hole. A Boltzmann labeler with
beta = 3 compares random-walk segments of 6 steps, and a one-hot linear reward
is fit with the Bradley-Terry loss.

Figure 1 draws one segment pair from the seed-0 data set: the first pair
whose true returns differ by more than 1. Figures 2 and 3 are as follows.

Figure 2 plots the learned reward of each of the 15 states against its true
reward, for 300 labels and seed 0. Figure 3 plots the policy regret of the
learned reward against the number of labels, averaged over 20 seeds, next to
the same fit with every label flipped.

Run from the repo root:

    python3 scripts/figures/052-preferences-over-trajectories.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "app" / "runner"))
sys.path.insert(0, str(ROOT / "content" / "exercises" / "052-preferences-over-trajectories"))

import solution as ex  # noqa: E402

BETA = 3.0
LENGTH = 6
LABEL_COUNTS = (25, 50, 100, 200, 300)
SEEDS = 20


def dataset(mdp, r_true, n, seed):
    rng = np.random.default_rng(seed)
    a = ex.sample_segments(mdp, n, LENGTH, rng)
    b = ex.sample_segments(mdp, n, LENGTH, rng)
    y = np.array([ex.simulate_preference(r_true, x, z, BETA, rng) for x, z in zip(a, b)])
    return a, b, y


def first_clear_pair(mdp, r_true):
    a, b, y = dataset(mdp, r_true, 300, 0)
    for i in range(len(a)):
        ga, gb = r_true[a[i]].sum(), r_true[b[i]].sum()
        if abs(ga - gb) > 1.0:
            return a[i], b[i], float(ga), float(gb)
    raise RuntimeError("no clear pair")


def pair_svg(mdp, seg_a, seg_b, ga, gb):
    cells = mdp.names["cells"]
    size, ox, oy = 62.0, 30.0, 20.0

    def center(k):
        i, j = cells[k]
        return ox + (j + 0.5) * size, oy + (i + 0.5) * size

    out = ['<svg viewBox="0 0 600 290" role="img" aria-label="A four by four grid with two six-step segments drawn on it, one in blue and one in orange, with their total true rewards">']
    for i in range(4):
        for j in range(4):
            x, y = ox + j * size, oy + i * size
            kind = ex.LAYOUT[i][j]
            cls = "fill-sunk" if kind == "#" else "fill-paper"
            if kind == "G":
                cls = "fill-soft-c"
            if kind == "H":
                cls = "fill-bad"
            out.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{size:.0f}" height="{size:.0f}" class="{cls} stroke-grid"/>')
            if kind in "GH":
                out.append(f'<text x="{x + size / 2:.0f}" y="{y + size / 2 + 5:.0f}" class="fig-bold" text-anchor="middle">{kind}</text>')
    for seg, cls in ((seg_a, "a"), (seg_b, "b")):
        pts = [center(int(k)) for k in seg]
        out.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + f'" class="stroke-{cls}" fill="none"/>')
        for n, (x, y) in enumerate(pts):
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{5 if n else 7}" class="fill-{cls}"/>')
    out.append(f'<text x="320" y="70" class="fig-bold">Segment A (blue)</text>')
    out.append(f'<text x="320" y="92" class="fig-small">total true reward {ga:.2f}</text>')
    out.append(f'<text x="320" y="140" class="fig-bold">Segment B (orange)</text>')
    out.append(f'<text x="320" y="162" class="fig-small">total true reward {gb:.2f}</text>')
    out.append('<text x="320" y="210" class="fig-small">The large dot is the first state entered.</text>')
    out.append('<text x="320" y="232" class="fig-small">The labeler only says which total is larger.</text>')
    out.append("</svg>")
    return "\n".join(out)


def figure_one(mdp, r_true):
    a, b, y = dataset(mdp, r_true, 300, 0)
    w = ex.fit_reward(np.eye(mdp.n_states), a, b, y)
    slope = float(np.polyfit(r_true, w, 1)[0])
    corr = float(np.corrcoef(r_true, w)[0, 1])
    return w, slope, corr


def figure_two(mdp, r_true):
    X = np.eye(mdp.n_states)
    good, flipped = [], []
    for n in LABEL_COUNTS:
        g, f = [], []
        for seed in range(SEEDS):
            a, b, y = dataset(mdp, r_true, n, seed)
            g.append(ex.policy_regret(mdp, r_true, ex.fit_reward(X, a, b, y)))
            f.append(ex.policy_regret(mdp, r_true, ex.fit_reward(X, a, b, 1 - y)))
        good.append(float(np.mean(g)))
        flipped.append(float(np.mean(f)))
    return good, flipped


def scatter_svg(r_true, w, slope, corr):
    x0, x1, y0, y1 = 80.0, 540.0, 250.0, 30.0
    xmin, xmax = -1.2, 1.2
    ymin, ymax = float(np.floor(w.min())), float(np.ceil(w.max()))

    def X(v):
        return x0 + (v - xmin) / (xmax - xmin) * (x1 - x0)

    def Y(v):
        return y0 + (v - ymin) / (ymax - ymin) * (y1 - y0)

    out = ['<svg viewBox="0 0 600 310" role="img" aria-label="Learned reward of each of 15 states plotted against its true reward: the points fall close to a rising line">']
    out.append(f'<line x1="{x0}" y1="{Y(0):.1f}" x2="{x1}" y2="{Y(0):.1f}" class="stroke-grid"/>')
    out.append(f'<line x1="{X(0):.1f}" y1="{y1}" x2="{X(0):.1f}" y2="{y0}" class="stroke-grid"/>')
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" class="stroke-muted"/>')
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" class="stroke-muted"/>')
    for t in (-1.0, -0.5, 0.0, 0.5, 1.0):
        out.append(f'<text x="{X(t):.1f}" y="{y0 + 18}" class="fig-small" text-anchor="middle">{t:g}</text>')
    step = 1 if ymax - ymin <= 6 else 2
    for t in range(int(ymin), int(ymax) + 1, step):
        out.append(f'<text x="{x0 - 8}" y="{Y(t) + 4:.1f}" class="fig-small" text-anchor="end">{t}</text>')
    c = np.polyfit(r_true, w, 1)
    out.append(f'<line x1="{X(xmin):.1f}" y1="{Y(c[0] * xmin + c[1]):.1f}" x2="{X(xmax):.1f}" y2="{Y(c[0] * xmax + c[1]):.1f}" class="stroke-muted dashed"/>')
    for rt, wl in zip(r_true, w):
        out.append(f'<circle cx="{X(rt):.1f}" cy="{Y(wl):.1f}" r="5" class="fill-b"/>')
    out.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{y0 + 40}" class="fig-small" text-anchor="middle">true reward of the state</text>')
    out.append(f'<text x="{x0}" y="20" class="fig-small">learned reward (weight on the state)</text>')
    out.append(f'<text x="{x1}" y="{y1 + 18}" class="fig-small" text-anchor="end">fitted slope {slope:.2f}, correlation {corr:.2f}</text>')
    out.append("</svg>")
    return "\n".join(out)


def regret_svg(good, flipped):
    x0, x1, y0, y1 = 80.0, 540.0, 250.0, 30.0
    ymax = float(np.ceil(max(flipped)))

    def X(i):
        return x0 + 30 + i * (x1 - x0 - 60) / (len(LABEL_COUNTS) - 1)

    def Y(v):
        return y0 + v / ymax * (y1 - y0)

    out = ['<svg viewBox="0 0 600 310" role="img" aria-label="Policy regret against the number of preference labels: it falls to zero for the fitted reward and stays high when every label is flipped">']
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" class="stroke-muted"/>')
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" class="stroke-muted"/>')
    for t in range(0, int(ymax) + 1, 5 if ymax > 10 else 1):
        out.append(f'<line x1="{x0}" y1="{Y(t):.1f}" x2="{x1}" y2="{Y(t):.1f}" class="stroke-grid"/>')
        out.append(f'<text x="{x0 - 8}" y="{Y(t) + 4:.1f}" class="fig-small" text-anchor="end">{t}</text>')
    for i, n in enumerate(LABEL_COUNTS):
        out.append(f'<text x="{X(i):.1f}" y="{y0 + 18}" class="fig-small" text-anchor="middle">{n}</text>')
    for series, cls in ((good, "a"), (flipped, "bad")):
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(series))
        out.append(f'<polyline points="{pts}" class="stroke-{cls}" fill="none"/>')
        for i, v in enumerate(series):
            out.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="4" class="fill-{cls}"/>')
    out.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{y0 + 40}" class="fig-small" text-anchor="middle">number of preference labels</text>')
    out.append(f'<text x="{x0}" y="20" class="fig-small">policy regret under the true reward (mean of {SEEDS} seeds)</text>')
    out.append(f'<text x="{X(2):.1f}" y="{Y(flipped[2]) - 12:.1f}" class="fig-small" text-anchor="middle">every label flipped</text>')
    out.append(f'<text x="{X(2):.1f}" y="{Y(good[2]) - 12:.1f}" class="fig-small" text-anchor="middle">labels as given</text>')
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    mdp = ex.make_world()
    r_true = ex.true_reward(mdp)
    seg_a, seg_b, ga, gb = first_clear_pair(mdp, r_true)
    print(f"figure 1: A {[int(k) for k in seg_a]} returns {ga:.3f}, B {[int(k) for k in seg_b]} returns {gb:.3f}")
    print(pair_svg(mdp, seg_a, seg_b, ga, gb))
    w, slope, corr = figure_one(mdp, r_true)
    print(f"figure 2: slope {slope:.3f}, correlation {corr:.3f}")
    print(scatter_svg(r_true, w, slope, corr))
    good, flipped = figure_two(mdp, r_true)
    print("figure 3: labels", LABEL_COUNTS)
    print("  as given:", [round(v, 3) for v in good])
    print("  flipped: ", [round(v, 3) for v in flipped])
    print(regret_svg(good, flipped))


if __name__ == "__main__":
    main()
