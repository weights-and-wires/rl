#!/usr/bin/env python3
"""Numbers and SVG for the decision transformer lesson.

Figure 1 is the attention mask over the six tokens of a two-step trajectory, built by
the exercise's `build_tokens` and `causal_mask`. Figure 2 is the realized return of a
tiny decision transformer against the target return it is asked for, averaged over five
training seeds. Figure 3 is the stitching example: the best return in the data from the
start state against the value offline Q-iteration finds. Every plotted number is
computed below, and the script prints the numbers the lesson quotes.

Run from the repo root:

    python3 scripts/figures/053-decision-transformer.py
    python3 scripts/figures/053-decision-transformer.py --svg
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "app" / "runner"))
sys.path.insert(0, str(ROOT / "content" / "exercises" / "053-decision-transformer"))

import solution as ex  # noqa: E402

TARGETS = (-8, -10, -12, -14, -16, -18, -20, -22, -24)
SEEDS = range(5)
TRANSITIONS = [(0, 0, 0.0, 1, False), (1, 0, 1.0, 2, True), (3, 0, 0.0, 1, False), (1, 1, 10.0, 4, True)]
TRAJS = [
    {"states": np.array([0, 1]), "actions": np.array([0, 0]), "rewards": np.array([0.0, 1.0])},
    {"states": np.array([3, 1]), "actions": np.array([0, 1]), "rewards": np.array([0.0, 10.0])},
]


def token_numbers():
    rtg = ex.returns_to_go([-1.0, -1.0])
    kinds, steps, values = ex.build_tokens(rtg, [0, 1], [2, 1])
    return rtg, kinds, steps, values, ex.causal_mask(len(kinds))


def sweep_numbers():
    """Mean realized return at each target, for each of five training seeds."""
    mdp = ex.make_world()
    table = np.zeros((len(SEEDS), len(TARGETS)))
    for i, seed in enumerate(SEEDS):
        trajs = ex.collect_mixed(mdp, 200, seed)
        model = ex.train_dt(trajs, mdp.n_states, mdp.n_actions, seed=seed)
        act = ex.dt_policy(model, seed=seed)
        for j, target in enumerate(TARGETS):
            table[i, j] = ex.mean_return(mdp, act, float(target), 20)
    return table


def stitching_numbers():
    Q = ex.offline_q_iteration(TRANSITIONS, 5, 2, 1.0, 5)
    return Q, ex.max_return_from(TRAJS, 0)


def luck_numbers():
    actions = np.array([0] * 100 + [1] * 100)
    returns = np.array([10.0] * 10 + [0.0] * 90 + [2.0] * 100)
    p10 = ex.conditional_action_probs(actions, returns, 10.0, 2)
    return p10, float(p10 @ np.array([1.0, 2.0]))


def print_numbers():
    rtg, kinds, steps, values, mask = token_numbers()
    print("returns_to_go([-1, -1]) =", rtg, " kinds", kinds, " steps", steps, " values", values)
    print("row of s_1 in the mask:", mask[4].astype(int))
    table = sweep_numbers()
    mean, sd = table.mean(axis=0), table.std(axis=0)
    for t, m, s in zip(TARGETS, mean, sd):
        print(f"  target {t:>4}: mean realized {m:.2f} (sd across seeds {s:.2f})")
    gaps = table[:, 0] - table[:, TARGETS.index(-20)]
    print("  gap between target -8 and -20, per seed:", np.round(gaps, 2))
    Q, best = stitching_numbers()
    print("stitching: Q(s0, a0) =", Q[0, 0], " best return from s0 in the data =", best)
    p10, val = luck_numbers()
    print("luck: P(a | R = 10) =", p10, " expected return of that choice =", val)


# ---------------------------------------------------------------- SVG helpers
def scale(v, lo, hi, a, b):
    return a + (v - lo) / (hi - lo) * (b - a)


def fig1():
    _, kinds, steps, _, mask = token_numbers()
    names = ["G", "s", "a"]
    labels = [f"{names[k]}{t}" for k, t in zip(kinds, steps)]
    n = len(labels)
    x0, y0, c = 150.0, 60.0, 44.0
    out = ['<svg viewBox="0 0 600 370" role="img" aria-label="Attention mask over six tokens G0 s0 a0 G1 s1 a1: each token sees itself and the tokens before it; the s1 row sees G0 to s1 but not a1">']
    for j, lab in enumerate(labels):
        out.append(f'<text x="{x0 + j * c + c / 2:.1f}" y="{y0 - 10}" class="fig-small" text-anchor="middle">{lab}</text>')
    for i, lab in enumerate(labels):
        out.append(f'<text x="{x0 - 10}" y="{y0 + i * c + c / 2 + 4:.1f}" class="fig-small" text-anchor="end">{lab}</text>')
        for j in range(n):
            cls = "fill-soft-a" if mask[i, j] else "fill-sunk"
            if kinds[i] == 1 and mask[i, j]:
                cls = "fill-soft-b"
            out.append(f'<rect x="{x0 + j * c}" y="{y0 + i * c}" width="{c}" height="{c}" class="{cls} stroke-grid"/>')
    out.append(f'<text x="{x0 + n * c / 2:.1f}" y="{y0 + n * c + 28}" class="fig-small" text-anchor="middle">columns: the token being read. rows: the token reading. s rows are orange: they predict the action.</text>')
    out.append(f'<text x="30" y="{y0 + 4 * c + c / 2 + 4:.1f}" class="fig-small" text-anchor="start">predicts a1</text>')
    out.append("</svg>")
    return "\n".join(out)


def fig2():
    table = sweep_numbers()
    mean, lo, hi = table.mean(axis=0), table.min(axis=0), table.max(axis=0)
    X0, X1, Y0, Y1 = 80.0, 560.0, 260.0, 40.0
    out = ['<svg viewBox="0 0 600 330" role="img" aria-label="Mean realized return of a tiny decision transformer against the target return it is asked for; realized return rises with the target and follows the diagonal">']
    for t in (-8, -12, -16, -20, -24):
        y = scale(t, -26, -6, Y0, Y1)
        out.append(f'<line x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}" class="stroke-grid"/>')
        out.append(f'<text x="{X0 - 8}" y="{y + 4:.1f}" class="fig-small" text-anchor="end">{t}</text>')
        x = scale(t, -26, -6, X0, X1)
        out.append(f'<text x="{x:.1f}" y="{Y0 + 22}" class="fig-small" text-anchor="middle">{t}</text>')
    pts = [(scale(t, -26, -6, X0, X1), scale(m, -26, -6, Y0, Y1)) for t, m in zip(TARGETS, mean)]
    out.append(f'<line x1="{scale(-24, -26, -6, X0, X1):.1f}" y1="{scale(-24, -26, -6, Y0, Y1):.1f}" x2="{scale(-8, -26, -6, X0, X1):.1f}" y2="{scale(-8, -26, -6, Y0, Y1):.1f}" class="stroke-muted dashed"/>')
    out.append(f'<text x="{scale(-12, -26, -6, X0, X1) + 10:.1f}" y="{scale(-12, -26, -6, Y0, Y1) - 8:.1f}" class="fig-small">realized = target</text>')
    for t, l, h in zip(TARGETS, lo, hi):
        x = scale(t, -26, -6, X0, X1)
        out.append(f'<line x1="{x:.1f}" y1="{scale(l, -26, -6, Y0, Y1):.1f}" x2="{x:.1f}" y2="{scale(h, -26, -6, Y0, Y1):.1f}" class="stroke-b"/>')
    out.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '" class="stroke-a" fill="none"/>')
    for x, y in pts:
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" class="fill-a"/>')
    out.append(f'<text x="{(X0 + X1) / 2:.0f}" y="{Y0 + 46}" class="fig-small" text-anchor="middle">target return asked for</text>')
    out.append(f'<text x="20" y="{(Y0 + Y1) / 2:.0f}" class="fig-small" text-anchor="middle" transform="rotate(-90 20 {(Y0 + Y1) / 2:.0f})">realized return</text>')
    out.append(f'<text x="{X1 - 10}" y="{Y1 - 10}" class="fig-small" text-anchor="end">bars: lowest and highest of five training seeds</text>')
    out.append("</svg>")
    return "\n".join(out)


def fig3():
    Q, best = stitching_numbers()
    out = ['<svg viewBox="0 0 600 260" role="img" aria-label="Two trajectories meet at state s1. Trajectory A starts at s0 and ends with return 1. Trajectory B starts at s3 and ends with return 10. Offline Q-iteration values s0 at 10; the best trajectory from s0 is worth 1">',
           '<defs><marker id="c052-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>']
    nodes = {"s0": (70, 80), "s3": (70, 190), "s1": (260, 135), "endA": (470, 80), "endB": (470, 190)}
    for k, (x, y) in nodes.items():
        cls = "fill-soft-a" if k in ("s0", "s3", "s1") else "fill-sunk"
        label = {"endA": "end", "endB": "end"}.get(k, k)
        out.append(f'<circle cx="{x}" cy="{y}" r="24" class="{cls} stroke-grid"/><text x="{x}" y="{y + 4}" class="fig-small" text-anchor="middle">{label}</text>')
    def arrow(a, b, cls, text, dy=0):
        (x1, y1), (x2, y2) = nodes[a], nodes[b]
        dx, dyy = x2 - x1, y2 - y1
        L = (dx * dx + dyy * dyy) ** 0.5
        sx, sy, ex_, ey = x1 + dx / L * 26, y1 + dyy / L * 26, x2 - dx / L * 28, y2 - dyy / L * 28
        out.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex_:.1f}" y2="{ey:.1f}" class="{cls}" marker-end="url(#c052-arrow)"/>')
        out.append(f'<text x="{(x1 + x2) / 2:.1f}" y="{(y1 + y2) / 2 + dy:.1f}" class="fig-small" text-anchor="middle">{text}</text>')
    arrow("s0", "s1", "stroke-a", "A: reward 0", -10)
    arrow("s3", "s1", "stroke-b", "B: reward 0", 22)
    arrow("s1", "endA", "stroke-a", "A: reward 1", -10)
    arrow("s1", "endB", "stroke-b", "B: reward 10", 22)
    out.append(f'<text x="300" y="240" class="fig-small" text-anchor="middle">best return from s0 in the data: {best:.0f}. Offline Q-iteration, Q(s0, a0): {Q[0, 0]:.0f}.</text>')
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    if "--svg" in sys.argv:
        for name, f in (("FIG1", fig1), ("FIG2", fig2), ("FIG3", fig3)):
            print(f"=== {name}")
            print(f())
    else:
        print_numbers()
