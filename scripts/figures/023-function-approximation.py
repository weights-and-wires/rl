#!/usr/bin/env python3
"""Numbers and SVG for the control section of the function-approximation lesson.

It prints the worked SARSA update from the lesson and the learning curve of
episodic semi-gradient SARSA on Mountain Car. With `--svg` it also prints the
figure that the lesson embeds. It uses the reference solution, so the numbers
match the exercise's checks.

Run from the repo root:

    python3 scripts/figures/023-function-approximation.py [--svg]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "app" / "runner"))
sys.path.insert(0, str(ROOT / "content" / "exercises" / "023-function-approximation"))

import solution as sol  # noqa: E402

N_EPISODES = 80
SEEDS = range(5)
CAP = 1000


def worked_update() -> None:
    """The tiny update from the lesson: one transition, 8 tilings, alpha = 0.05."""
    s, a, s2, a2 = (-0.5, 0.0), 2, (-0.45, 0.01), 0
    x = sol.state_action_features(s[0], s[1], a)
    x2 = sol.state_action_features(s2[0], s2[1], a2)
    w = 0.25 * x + 0.5 * x2
    r, gamma, alpha = -1.0, 0.9, 0.05
    q, q2 = float(w @ x), float(w @ x2)
    print(f"active weights per pair: {int(x.sum())}, each 0.25 for (s, a) and 0.5 for (s2, a2)")
    print(f"q(s, a) = {q:.2f}, q(s2, a2) = {q2:.2f}")
    target = r + gamma * q2
    print(f"target = {target:.2f}, delta = {target - q:.2f}, each active weight moves {alpha * (target - q):.3f}")
    new = sol.semi_gradient_sarsa_update(w, s, a, r, s2, a2, False, alpha, gamma)
    print(f"q(s, a) after the update: {float(new @ x):.2f}; entries changed: {int(np.sum(new != w))}")
    done = sol.semi_gradient_sarsa_update(w, s, a, r, s2, a2, True, alpha, gamma)
    print(f"terminal: delta = {r - q:.2f}, q(s, a) after the update: {float(done @ x):.2f}")


def learning_curves() -> np.ndarray:
    curves = np.array([sol.semi_gradient_sarsa(N_EPISODES, seed=s, max_steps=CAP)[1] for s in SEEDS])
    for s, c in zip(SEEDS, curves):
        print(f"seed {s}: first 3 episodes {c[:3].tolist()}, mean of the last 10 {c[-10:].mean():.1f}")
    mean = curves.mean(axis=0)
    print(f"mean over seeds: episode 1 {mean[0]:.0f}, episode 10 {mean[9]:.0f}, "
          f"episode 40 {mean[39]:.0f}, episode 80 {mean[79]:.0f}")
    return curves


def svg(curves: np.ndarray) -> str:
    x0, x1, y0, y1 = 70.0, 570.0, 220.0, 40.0

    def px(i: int) -> float:
        return x0 + (x1 - x0) * i / (N_EPISODES - 1)

    def py(v: float) -> float:
        return y0 + (y1 - y0) * v / CAP

    def line(values, cls: str) -> str:
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(values))
        return f'<polyline points="{pts}" class="{cls}" fill="none"/>'

    out = [
        '<svg viewBox="0 0 600 270" role="img" aria-label="Steps per episode of semi-gradient SARSA on Mountain Car falling from 1000 in the first episode to about 130 by episode 80">',
    ]
    for v in (0, 500, 1000):
        out.append(f'<line x1="{x0}" y1="{py(v):.1f}" x2="{x1}" y2="{py(v):.1f}" class="stroke-grid"/>')
        out.append(f'<text x="{x0 - 8}" y="{py(v) + 4:.1f}" class="fig-small" text-anchor="end">{v}</text>')
    for e in (1, 20, 40, 60, 80):
        out.append(f'<text x="{px(e - 1):.1f}" y="238.0" class="fig-small" text-anchor="middle">{e}</text>')
    out.append('<text x="62.0" y="22.0" class="fig-small" text-anchor="start">steps to reach the flag (cap 1000)</text>')
    out.append('<text x="320.0" y="262.0" text-anchor="middle">episode</text>')
    out.append(line(curves[0], "stroke-a"))
    out.append(line(curves.mean(axis=0), "stroke-b"))
    out.append('<line x1="400.0" y1="52.0" x2="430.0" y2="52.0" class="stroke-a"/>')
    out.append('<text x="438.0" y="56.0" class="fig-small" text-anchor="start">seed 0</text>')
    out.append('<line x1="400.0" y1="70.0" x2="430.0" y2="70.0" class="stroke-b"/>')
    out.append('<text x="438.0" y="74.0" class="fig-small" text-anchor="start">mean of 5 seeds</text>')
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    worked_update()
    curves = learning_curves()
    if "--svg" in sys.argv:
        print(svg(curves))
