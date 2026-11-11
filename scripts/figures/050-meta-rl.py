#!/usr/bin/env python3
"""Numbers and figures for the meta-RL lesson.

Three computations feed the lesson, all from the exercise's reference solution:

* The values of a fixed arm, Thompson sampling, the Bayes-optimal adaptive
  policy, and an oracle on the three-task bandit family, for H = 1 to 4.
* The one-dimensional MAML worked example: the adapted parameter, the full
  gradient with the second-order term, and the first-order approximation,
  checked against a finite difference.
* The post-adaptation objective on each of three tasks, from the multi-task
  initialization and from the MAML initialization.

Run from the repo root. Pass --svg DIR to also write the three figure bodies
as figure1.svg, figure2.svg, and figure3.svg, which the lesson pastes in:

    python3 scripts/figures/050-meta-rl.py
    python3 scripts/figures/050-meta-rl.py --svg /path/to/dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "content" / "exercises" / "050-meta-rl"))

import solution as ex  # noqa: E402

ALPHA = 4.0
HORIZONS = (1, 2, 3, 4)


def bandit_values() -> dict[str, list[float]]:
    p, th = ex.FAMILY_PRIOR, ex.FAMILY
    return {
        "fixed arm": [ex.fixed_policy_value(p, th, H) for H in HORIZONS],
        "Thompson": [ex.thompson_value(p, th, H) for H in HORIZONS],
        "Bayes-optimal": [ex.bayes_optimal_value(p, th, H) for H in HORIZONS],
        "oracle": [ex.oracle_value(p, th, H) for H in HORIZONS],
    }


def worked_example() -> dict[str, float]:
    """One 1-D task (c=1, w=0.5), theta=0, alpha=1, with sigma=0.5 as in the exercise."""
    task = (np.array([1.0]), 0.5)
    theta, alpha = np.array([0.0]), 1.0
    g = ex.task_grad(theta, task)[0]
    h = ex.task_hessian(theta, task)[0, 0]
    adapted = ex.adapt(theta, task, alpha)
    j_adapted = ex.task_objective(adapted, task)
    g_adapted = ex.task_grad(adapted, task)[0]
    full = (1 + alpha * h) * g_adapted
    eps = 1e-6
    fd = (
        ex.task_objective(ex.adapt(theta + eps, task, alpha), task)
        - ex.task_objective(ex.adapt(theta - eps, task, alpha), task)
    ) / (2 * eps)
    return {
        "J(0)": float(ex.task_objective(theta, task)),
        "g": float(g),
        "H": float(h),
        "adapted": float(adapted[0]),
        "J(adapted)": float(j_adapted),
        "g(adapted)": float(g_adapted),
        "full": float(full),
        "first_order": float(g_adapted),
        "finite_difference": float(fd),
    }


def maml_numbers() -> dict[str, object]:
    mt = ex.multitask_init(ex.TASKS, 0.5, 600)
    meta = ex.maml_train(mt, ex.TASKS, ALPHA, 1.0, 150)
    out = {"mt": mt, "meta": meta}
    for name, th in (("mt", mt), ("meta", meta)):
        out[name + "_pre"] = [float(ex.task_objective(th, t)) for t in ex.TASKS]
        out[name + "_post"] = [float(ex.task_objective(ex.adapt(th, t, ALPHA), t)) for t in ex.TASKS]
        out[name + "_mean_post"] = ex.maml_objective(th, ex.TASKS, ALPHA)
        out[name + "_mean_pre"] = float(np.mean(out[name + "_pre"]))
    th = mt.copy()
    for _ in range(150):
        th = th + 1.0 * ex.fomaml_grad(th, ex.TASKS, ALPHA)
    out["fo"] = th
    out["fo_mean_post"] = ex.maml_objective(th, ex.TASKS, ALPHA)
    return out


# --- SVG ---------------------------------------------------------------------

ARROW = (
    '<defs><marker id="c048-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" '
    'markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
    'class="fill-ink"/></marker></defs>'
)


def svg_rl2() -> str:
    out = [
        '<svg viewBox="0 0 600 270" role="img" aria-label="A recurrent policy unrolled over five steps of one trial. Each step reads the observation and the previous action, reward, and done flag. The hidden state passes along every step, across the episode boundary, and starts at zero only at the trial start">',
        ARROW,
    ]
    xs = [40, 150, 260, 370, 480]
    ep_end = 2  # step index whose done flag is 1
    for i, x in enumerate(xs):
        out.append(f'<rect x="{x}" y="100" width="80" height="50" rx="6" class="fill-soft-b stroke-muted"/>')
        out.append(f'<text x="{x + 40}" y="130" class="fig-bold" text-anchor="middle">GRU</text>')
        out.append(f'<line x1="{x + 40}" y1="206" x2="{x + 40}" y2="152" class="stroke-a" marker-end="url(#c048-arrow)"/>')
        out.append(f'<text x="{x + 40}" y="222" class="fig-small" text-anchor="middle">o{i}, a{i - 1 if i else "∅"}</text>')
        out.append(f'<text x="{x + 40}" y="238" class="fig-small" text-anchor="middle">r{i - 1 if i else "0"}, d{i - 1 if i else "0"}</text>')
        out.append(f'<line x1="{x + 40}" y1="98" x2="{x + 40}" y2="62" class="stroke-b" marker-end="url(#c048-arrow)"/>')
        out.append(f'<text x="{x + 40}" y="50" class="fig-small" text-anchor="middle">action a{i}</text>')
        if i < len(xs) - 1:
            out.append(f'<line x1="{x + 82}" y1="125" x2="{x + 108}" y2="125" class="stroke-ink" marker-end="url(#c048-arrow)"/>')
    out.append(f'<line x1="{xs[ep_end] + 95}" y1="90" x2="{xs[ep_end] + 95}" y2="200" class="stroke-bad dashed"/>')
    out.append(f'<text x="{xs[ep_end] + 95}" y="176" class="fig-small" text-anchor="middle">episode</text>')
    out.append(f'<text x="{xs[ep_end] + 95}" y="192" class="fig-small" text-anchor="middle">ends</text>')
    out.append('<text x="40" y="82" class="fig-small">h = 0</text>')
    out.append('<text x="300" y="22" class="fig-bold" text-anchor="middle">one trial: the hidden state is not reset at the episode end</text>')
    out.append("</svg>")
    return "\n".join(out)


def svg_values(vals: dict[str, list[float]]) -> str:
    x0, x1, y0, y1 = 70, 470, 230, 40
    ymax = 3.2
    def X(i): return x0 + (x1 - x0) * i / (len(HORIZONS) - 1)
    def Y(v): return y0 - (y0 - y1) * v / ymax
    out = ['<svg viewBox="0 0 600 280" role="img" aria-label="Expected return against horizon H on the bandit family: the oracle is highest, then the Bayes-optimal adaptive policy, then Thompson sampling, then a fixed arm. At H = 3 they are 2.28, 1.85, 1.69, and 1.62">']
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" class="stroke-muted"/>')
    for v in (0, 1, 2, 3):
        out.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" class="stroke-grid"/>')
        out.append(f'<text x="{x0 - 8}" y="{Y(v) + 4:.1f}" text-anchor="end" class="fig-small">{v}</text>')
    for i, H in enumerate(HORIZONS):
        out.append(f'<text x="{X(i):.1f}" y="{y0 + 18}" text-anchor="middle" class="fig-small">{H}</text>')
    out.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{y0 + 40}" text-anchor="middle" class="fig-small">horizon H (pulls in one trial)</text>')
    styles = {"oracle": "stroke-ink", "Bayes-optimal": "stroke-a", "Thompson": "stroke-b", "fixed arm": "stroke-c"}
    for name, cls in styles.items():
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals[name]))
        out.append(f'<polyline points="{pts}" fill="none" class="{cls}" stroke-width="2.5"/>')
        for i, v in enumerate(vals[name]):
            fill = cls.replace("stroke", "fill")
            out.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3.5" class="{fill}"/>')
    label_y = {"oracle": 0, "Bayes-optimal": 0, "Thompson": 0, "fixed arm": 0}
    ends = {n: vals[n][-1] for n in styles}
    # Offset the two close labels so they do not overlap.
    offs = {"oracle": 4, "Bayes-optimal": 4, "Thompson": 4, "fixed arm": 6}
    for name in styles:
        out.append(f'<text x="{x1 + 10}" y="{Y(ends[name]) + offs[name]:.1f}" class="fig-small">{name} {ends[name]:.2f}</text>')
    out.append("</svg>")
    return "\n".join(out)


def svg_maml(nums: dict[str, object]) -> str:
    x0, y0, y1 = 60, 220, 40
    def Y(v): return y0 - (y0 - y1) * v / 1.0
    out = ['<svg viewBox="0 0 600 280" role="img" aria-label="Reward after one adaptation step on each of three tasks and on average, starting from the multi-task initialization and from the MAML initialization. MAML trades a worse average before adapting for a better average after">']
    out.append(f'<line x1="{x0}" y1="{y0}" x2="560" y2="{y0}" class="stroke-muted"/>')
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="560" y2="{Y(v):.1f}" class="stroke-grid"/>')
        out.append(f'<text x="{x0 - 8}" y="{Y(v) + 4:.1f}" text-anchor="end" class="fig-small">{v:g}</text>')
    groups = [f"task {i + 1}" for i in range(3)] + ["mean"]
    mt = list(nums["mt_post"]) + [nums["mt_mean_post"]]
    me = list(nums["meta_post"]) + [nums["meta_mean_post"]]
    for g, name in enumerate(groups):
        gx = x0 + 20 + g * 120
        for k, (v, cls) in enumerate(((mt[g], "fill-b"), (me[g], "fill-a"))):
            bx = gx + k * 42
            out.append(f'<rect x="{bx}" y="{Y(v):.1f}" width="36" height="{y0 - Y(v):.1f}" class="{cls}"/>')
            out.append(f'<text x="{bx + 18}" y="{Y(v) - 5:.1f}" text-anchor="middle" class="fig-small">{v:.2f}</text>')
        out.append(f'<text x="{gx + 39}" y="{y0 + 18}" text-anchor="middle" class="fig-small">{name}</text>')
    out.append('<rect x="70" y="258" width="12" height="12" class="fill-b"/><text x="88" y="268" class="fig-small">multi-task initialization</text>')
    out.append('<rect x="290" y="258" width="12" height="12" class="fill-a"/><text x="308" y="268" class="fig-small">MAML initialization</text>')
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--svg", type=Path)
    args = ap.parse_args()
    vals = bandit_values()
    print("bandit values, H = 1..4")
    for k, v in vals.items():
        print(f"  {k:14s}", " ".join(f"{x:.4f}" for x in v))
    print("worked example:", {k: round(v, 4) for k, v in worked_example().items()})
    nums = maml_numbers()
    for k, v in nums.items():
        print(" ", k, np.round(v, 4))
    if args.svg:
        args.svg.mkdir(parents=True, exist_ok=True)
        (args.svg / "figure1.svg").write_text(svg_rl2() + "\n")
        (args.svg / "figure2.svg").write_text(svg_values(vals) + "\n")
        (args.svg / "figure3.svg").write_text(svg_maml(nums) + "\n")


if __name__ == "__main__":
    main()
