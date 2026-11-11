#!/usr/bin/env python3
"""Numbers and figures for the sim-to-real lesson.

Three computations feed the lesson, all from the exercise's reference solution:

* The worked sysid example: two transitions of a mass-2, friction-0.5 robot.
* Cost against mass at friction 0.5 for the gain tuned on the simulator, the
  gain tuned on randomized physics, and the adaptive controller, plus the
  mean, worst-case, and simulator cost of each on the held-out real set.
* A mass estimate from the last 10 transitions while the mass drops from 1.0
  to 0.3 at step 50, as an explicit stand-in for what RMA learns.

Run from the repo root. Pass --svg DIR to also write the two figure bodies as
figure1.svg and figure2.svg, which the lesson pastes in:

    python3 scripts/figures/082-sim-to-real.py
    python3 scripts/figures/082-sim-to-real.py --svg /path/to/dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "content" / "exercises" / "082-sim-to-real"))

import solution as ex  # noqa: E402

MASSES = np.exp(np.linspace(np.log(0.1), np.log(2.0), 40))


def worked_example() -> dict:
    p = np.array([2.0, 0.5])
    S = np.array([[0.0, 1.0], [0.0, 0.0]])
    U = np.array([4.0, 2.0])
    S2 = ex.step(S, U, p)
    mass, friction = ex.sysid_fit(S, U, S2)
    return {"v_next": S2[:, 1].tolist(), "accel": ((S2[:, 1] - S[:, 1]) / ex.DT).tolist(),
            "mass": mass, "friction": friction}


def curves(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    g_nom = ex.tune_gain(ex.NOMINAL[None], seed)
    g_rob = ex.tune_gain(ex.randomized_params(rng, ex.RANGES, 128), seed)
    line = np.stack([MASSES, np.full_like(MASSES, 0.5)], axis=1)
    return {
        "g_nom": g_nom, "g_rob": g_rob, "masses": MASSES,
        "nominal": ex.batch_cost(g_nom[None], line)[0],
        "robust": ex.batch_cost(g_rob[None], line)[0],
        "adaptive": ex.adaptive_cost(line, line),
        "report": ex.report(seed),
        # Velocity multiplier per step in the velocity loop: 1 - dt * (kv + b) / m.
        "mult_nom_light": 1 - ex.DT * (g_nom[1] + 0.5) / 0.15,
        "mult_nom_sim": 1 - ex.DT * (g_nom[1] + 0.5) / 1.0,
        "mult_rob_light": 1 - ex.DT * (g_rob[1] + 0.5) / 0.15,
    }


def window_trace(window: int = 10, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    prior = np.array([1.0, 0.5])
    s = np.array([1.0, 0.0])
    S, U, S2, est, true_m = [], [], [], [], []
    for t in range(100):
        true = np.array([1.0 if t < 50 else 0.3, 0.5])
        est_now = ex.history_estimate(np.array(S), np.array(U), np.array(S2), window, prior) if S else prior
        u = float(np.clip(ex.adaptive_action(s, est_now, ex.ADAPT_GAINS) + rng.normal(0, 1.0), -ex.U_MAX, ex.U_MAX))
        s2 = ex.step(s, u, true)
        S.append(s); U.append(u); S2.append(s2)
        est.append(float(est_now[0])); true_m.append(float(true[0]))
        s = s2
    return {"est": np.array(est), "true": np.array(true_m)}


def fig1(c: dict) -> str:
    x0, x1, y0, y1 = 70, 570, 30, 210
    ylo, yhi = np.log10(0.05), np.log10(5.0)
    xlo, xhi = np.log10(0.1), np.log10(2.0)
    px = lambda m: x0 + (np.log10(m) - xlo) / (xhi - xlo) * (x1 - x0)
    py = lambda v: y1 - (np.log10(np.clip(v, 0.05, 5.0)) - ylo) / (yhi - ylo) * (y1 - y0)
    out = ['<svg viewBox="0 0 600 270" role="img" aria-label="Cost against mass for three controllers: the gain tuned on the simulator blows up for light robots, the randomized gain stays flat, and the adaptive controller is lowest">']
    for v in (0.1, 1.0):
        out.append(f'<line x1="{x0}" y1="{py(v):.1f}" x2="{x1}" y2="{py(v):.1f}" class="stroke-grid"/>')
        out.append(f'<text x="{x0-8}" y="{py(v)+4:.1f}" class="fig-small" text-anchor="end">{v:g}</text>')
    out.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" class="stroke-muted"/>')
    for m in (0.1, 0.2, 0.5, 1.0, 2.0):
        out.append(f'<text x="{px(m):.1f}" y="{y1+18}" class="fig-small" text-anchor="middle">{m:g}</text>')
    out.append(f'<text x="{(x0+x1)//2}" y="{y1+38}" class="fig-small" text-anchor="middle">true mass (log scale), friction 0.5</text>')
    out.append(f'<text x="{x0-8}" y="{y0-10}" class="fig-small" text-anchor="end">cost (log)</text>')
    for key, cls in (("nominal", "stroke-bad"), ("robust", "stroke-b"), ("adaptive", "stroke-c")):
        pts = " ".join(f"{px(m):.1f},{py(v):.1f}" for m, v in zip(c["masses"], c[key]))
        out.append(f'<polyline points="{pts}" fill="none" class="{cls}" stroke-width="2.5"/>')
    out.append(f'<line x1="{px(1.0):.1f}" y1="{y0}" x2="{px(1.0):.1f}" y2="{y1}" class="stroke-muted dashed"/>')
    out.append(f'<text x="{px(1.0)+6:.1f}" y="{y0+10}" class="fig-small">simulator mass</text>')
    lx = x0 + 10
    for i, (label, cls) in enumerate((("tuned on the simulator", "fill-bad"), ("tuned on randomized physics", "fill-b"), ("adaptive, identified physics", "fill-c"))):
        out.append(f'<rect x="{lx}" y="{y0+24+i*16}" width="10" height="10" class="{cls}"/>')
        out.append(f'<text x="{lx+16}" y="{y0+33+i*16}" class="fig-small">{label}</text>')
    out.append("</svg>")
    return "\n".join(out)


def fig2(tr: dict) -> str:
    x0, x1, y0, y1 = 60, 570, 30, 190
    px = lambda t: x0 + t / 99 * (x1 - x0)
    py = lambda v: y1 - (v - 0.0) / 1.2 * (y1 - y0)
    out = ['<svg viewBox="0 0 600 250" role="img" aria-label="Mass estimate from the last ten transitions while the true mass drops from 1.0 to 0.3 at step 50">']
    for v in (0.0, 0.5, 1.0):
        out.append(f'<line x1="{x0}" y1="{py(v):.1f}" x2="{x1}" y2="{py(v):.1f}" class="stroke-grid"/>')
        out.append(f'<text x="{x0-8}" y="{py(v)+4:.1f}" class="fig-small" text-anchor="end">{v:g}</text>')
    out.append(f'<text x="{(x0+x1)//2}" y="{y1+34}" class="fig-small" text-anchor="middle">time step</text>')
    for t in (0, 25, 50, 75, 99):
        out.append(f'<text x="{px(t):.1f}" y="{y1+16}" class="fig-small" text-anchor="middle">{t}</text>')
    pts = " ".join(f"{px(t):.1f},{py(v):.1f}" for t, v in enumerate(tr["true"]))
    out.append(f'<polyline points="{pts}" fill="none" class="stroke-a dashed" stroke-width="2"/>')
    pts = " ".join(f"{px(t):.1f},{py(v):.1f}" for t, v in enumerate(tr["est"]))
    out.append(f'<polyline points="{pts}" fill="none" class="stroke-b" stroke-width="2.5"/>')
    out.append(f'<text x="{px(55):.1f}" y="{py(0.3)+22:.1f}" class="fig-small">true mass</text>')
    out.append(f'<text x="{px(5):.1f}" y="{py(1.0)-12:.1f}" class="fig-small">estimate from the last 10 steps</text>')
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--svg")
    args = ap.parse_args()
    w = worked_example()
    print("worked example:", {k: np.round(v, 6) for k, v in w.items()})
    c = curves()
    print("gains: nominal", c["g_nom"].round(2), "randomized", c["g_rob"].round(2))
    print("velocity multiplier at m=0.15: nominal-tuned", round(c["mult_nom_light"], 3),
          "randomized", round(c["mult_rob_light"], 3), "; nominal-tuned at the simulator mass", round(c["mult_nom_sim"], 3))
    for name, row in c["report"].items():
        print(f"{name:14s} sim {row['sim']:.4f}  real mean {row['real_mean']:.4f}  real worst {row['real_worst']:.4f}")
    tr = window_trace()
    print("window estimate at steps 45, 49, 52, 60, 99:", tr["est"][[45, 49, 52, 60, 99]].round(3))
    if args.svg:
        out = Path(args.svg)
        out.mkdir(parents=True, exist_ok=True)
        (out / "figure1.svg").write_text(fig1(c) + "\n")
        (out / "figure2.svg").write_text(fig2(tr) + "\n")


if __name__ == "__main__":
    main()
