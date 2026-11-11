#!/usr/bin/env python3
"""Numbers and SVG for the options and hierarchical RL lesson.

Figure 1 draws the four-rooms map and the plan that a planner with primitives and
hallway options picks from the start (a greedy walk with no slips). Figure 2 plots
the largest value error per sweep for planning with primitives only and with
primitives plus hallway options. The script also prints every number the lesson quotes.

Run from the repo root:

    python3 scripts/figures/049-options-and-hierarchical-rl.py            # numbers
    python3 scripts/figures/049-options-and-hierarchical-rl.py --svg DIR  # also write fig1.svg, fig2.svg
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend/app/runner"))
sys.path.insert(0, str(ROOT / "content/exercises/049-options-and-hierarchical-rl"))

import solution as sol  # noqa: E402

ROOM_FILL = ("fill-soft-a", "fill-soft-b", "fill-soft-c", "fill-soft-d")


def sweep_errors(env, options, V_star):
    """Largest |V_k - V*| after each sweep, with the same start as smdp_value_iteration."""
    models = [sol.option_model(env, o) for o in options]
    V = np.full(len(env.cells), env.R.min() / (1 - env.gamma))
    V[env.goal] = 0.0
    errs = [float(np.abs(V - V_star).max())]
    for _ in range(60):
        q = np.full((len(options), len(V)), -np.inf)
        for i, (o, (R_o, P_o)) in enumerate(zip(options, models)):
            q[i] = np.where(o.init, R_o + P_o @ V, -np.inf)
        V = q.max(axis=0)
        V[env.goal] = 0.0
        errs.append(float(np.abs(V - V_star).max()))
    return errs


def plan(env, options, V_star):
    """Walk from the start, always taking the longest option that is optimal. No slips."""
    models = [sol.option_model(env, o) for o in options]
    s = env.start
    path = [s]
    marks = []
    while s != env.goal:
        values = [R_o[s] + P_o[s] @ V_star if o.init[s] else -np.inf
                  for o, (R_o, P_o) in zip(options, models)]
        best = max(values)
        i = max(j for j, v in enumerate(values) if v > best - 1e-9)
        o = options[i]
        marks.append((s, o.name))
        while True:
            a = o.pi[s]
            r, c = env.cells[s]
            dr, dc = sol.MOVES[a]
            s = env.index.get((r + dr, c + dc), s)
            path.append(s)
            if s == env.goal or o.beta[s] >= 1.0:
                break
    return path, marks


def fig1(env, path, marks):
    cs, x0, y0 = 30.0, 20.0, 20.0
    s = ['<svg viewBox="0 0 600 410" role="img" aria-label="The four-rooms map with the plan from the top-left corner to the goal in the bottom-right corner: two hallway options with primitive moves between and after them">']
    s.append('<defs><marker id="c048-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>')
    for r, row in enumerate(sol.LAYOUT):
        for c, ch in enumerate(row):
            x, y = x0 + c * cs, y0 + r * cs
            if ch == "#":
                s.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{cs:.0f}" height="{cs:.0f}" class="fill-ink"/>')
            else:
                st = env.index[(r, c)]
                cls = "fill-b" if (r, c) in sol.HALLWAYS else ROOM_FILL[env.room[st]]
                s.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{cs:.0f}" height="{cs:.0f}" class="{cls} stroke-grid"/>')
    gr, gc = env.cells[env.goal]
    s.append(f'<text x="{x0 + gc * cs + cs / 2:.0f}" y="{y0 + gr * cs + cs / 2 + 5:.0f}" class="fig-bold" text-anchor="middle">G</text>')
    pts = " ".join(f"{x0 + env.cells[p][1] * cs + cs / 2:.0f},{y0 + env.cells[p][0] * cs + cs / 2:.0f}" for p in path)
    s.append(f'<polyline points="{pts}" class="stroke-ink" fill="none" marker-end="url(#c048-arrow)"/>')
    for st, name in marks:
        if not name.startswith("room"):
            continue
        r, c = env.cells[st]
        s.append(f'<circle cx="{x0 + c * cs + cs / 2:.0f}" cy="{y0 + r * cs + cs / 2:.0f}" r="5" class="fill-ink"/>')
    r, c = env.cells[env.start]
    s.append(f'<text x="{x0 + c * cs + cs / 2:.0f}" y="{y0 + r * cs + cs / 2 + 4:.0f}" class="fig-small fig-bold" text-anchor="middle">S</text>')
    tx = x0 + 13 * cs + 24
    s.append(f'<text x="{tx:.0f}" y="40" class="fig-bold" text-anchor="start">Plan from S</text>')
    ly = 64
    runs = []
    for st, name in marks:
        if name.startswith("room"):
            runs.append([env.cells[st], name, 0])
        elif runs and runs[-1][1] == "primitive":
            runs[-1][2] += 1
        else:
            runs.append([env.cells[st], "primitive", 1])
    for cell, name, count in runs:
        label = name if count == 0 else f"{count} primitive move{'s' if count > 1 else ''}"
        s.append(f'<text x="{tx:.0f}" y="{ly}" class="fig-small" text-anchor="start">{cell}: {label}</text>')
        ly += 18
    s.append(f'<rect x="{tx:.0f}" y="{ly + 8}" width="14" height="14" class="fill-b stroke-grid"/>')
    s.append(f'<text x="{tx + 22:.0f}" y="{ly + 20}" class="fig-small" text-anchor="start">hallway cell</text>')
    s.append('</svg>')
    return "\n".join(s)


def fig2(e_prim, e_opt):
    x0, x1, y0, y1 = 70.0, 540.0, 270.0, 40.0
    n = 45
    lo, hi = -7.0, 1.0

    def px(k):
        return x0 + k / n * (x1 - x0)

    def py(e):
        return y0 - (math.log10(max(e, 1e-7)) - lo) / (hi - lo) * (y0 - y1)

    s = ['<svg viewBox="0 0 600 340" role="img" aria-label="Largest value error after each sweep on a log scale: planning with hallway options reaches the 1e-6 tolerance in 25 sweeps and primitives alone in 41">']
    for d in range(-7, 2, 2):
        y = py(10.0**d)
        s.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" class="stroke-grid"/>')
        s.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" class="fig-small" text-anchor="end">1e{d}</text>')
    s.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" class="stroke-muted"/>')
    for k in range(0, n + 1, 10):
        s.append(f'<line x1="{px(k):.1f}" y1="{y0}" x2="{px(k):.1f}" y2="{y0 + 5}" class="stroke-muted"/>')
        s.append(f'<text x="{px(k):.1f}" y="{y0 + 20}" class="fig-small" text-anchor="middle">{k}</text>')
    s.append(f'<text x="{(x0 + x1) / 2}" y="318" class="fig-small" text-anchor="middle">sweep</text>')
    s.append(f'<line x1="{x0}" y1="{py(1e-6):.1f}" x2="{x1}" y2="{py(1e-6):.1f}" class="stroke-bad dashed"/>')
    s.append(f'<text x="{x1 - 4}" y="{py(1e-6) - 5:.1f}" class="fig-small" text-anchor="end">tolerance 1e-6</text>')
    for errs, cls, name, ly in ((e_prim, "stroke-a", "primitives only", 50), (e_opt, "stroke-b", "primitives and hallway options", 68)):
        pts = " ".join(f"{px(k):.1f},{py(e):.1f}" for k, e in enumerate(errs[: n + 1]))
        s.append(f'<polyline points="{pts}" class="{cls}" fill="none"/>')
        s.append(f'<line x1="{x0 + 30}" y1="{ly}" x2="{x0 + 60}" y2="{ly}" class="{cls}"/>')
        s.append(f'<text x="{x0 + 66}" y="{ly + 4}" class="fig-small" text-anchor="start">{name}</text>')
    s.append('</svg>')
    return "\n".join(s)


def main() -> None:
    env = sol.four_rooms()
    prims, halls = sol.primitive_options(env), sol.hallway_options(env)
    options = prims + halls
    V_p, n_p = sol.smdp_value_iteration(env, prims)
    V_o, n_o = sol.smdp_value_iteration(env, options)
    V_star, _ = sol.smdp_value_iteration(env, prims, tol=1e-13)
    print(f"states {len(env.cells)}, goal {env.cells[env.goal]}, start {env.cells[env.start]}")
    print(f"V*(start) = {V_star[env.start]:.4f}; sweeps primitives {n_p}, with hallway options {n_o}")
    print(f"max |V_options - V_primitives| at tol 1e-12: "
          f"{np.abs(sol.smdp_value_iteration(env, options, tol=1e-12)[0] - V_star).max():.2e}")
    R_o, P_o = sol.option_model(env, halls[0])
    print(f"{halls[0].name} from start: R_o {R_o[env.start]:.4f}, sum P_o {P_o[env.start].sum():.4f}")
    a = env.index[(3, 5)]
    print(f"{halls[0].name} from (3, 5): R_o {R_o[a]:.6f} (-1/0.82 = {-1 / 0.82:.6f}), "
          f"P_o {P_o[a].sum():.6f} (0.72/0.82 = {0.72 / 0.82:.6f})")
    gam5 = 0.9**5
    ret5 = sol.discounted_return([-1.0] * 5, 0.9)
    print(f"five steps at -1: return {ret5:.4f}, gamma^5 {gam5:.5f}; V(s')=-6: target {ret5 + gam5 * -6:.4f} "
          f"vs gamma-only {ret5 + 0.9 * -6:.4f}")
    path, marks = plan(env, options, V_star)
    print("plan:", [(env.cells[s], n) for s, n in marks], "path length", len(path) - 1)
    print("Q at start per option:", {o.name: round(float(R[env.start] + P[env.start] @ V_star), 4)
                                      for o, (R, P) in zip(options, (sol.option_model(env, o) for o in options)) if o.init[env.start]})
    # Learning comparison, same targets as tests.py.
    target = np.full((len(env.cells), len(options)), np.nan)
    for i, o in enumerate(options):
        R, P = sol.option_model(env, o)
        target[:, i] = np.where(o.init, R + P @ V_star, np.nan)
    for mode in ("smdp", "intra"):
        errs = [np.nanmean(np.abs(sol.learn(env, options, mode, 100000, 0.1, 0.1, seed) - target)) for seed in (0, 1)]
        print(f"learn {mode}: mean error {np.mean(errs):.3f} ({errs[0]:.3f}, {errs[1]:.3f})")
    e_p, e_o = sweep_errors(env, prims, V_star), sweep_errors(env, options, V_star)
    print("error after sweep 10, 20, 25:", [(round(e_p[k], 5), round(e_o[k], 5)) for k in (10, 20, 25)])
    if "--svg" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--svg") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "fig1.svg").write_text(fig1(env, path, marks) + "\n")
        (out / "fig2.svg").write_text(fig2(e_p, e_o) + "\n")


if __name__ == "__main__":
    main()
