#!/usr/bin/env python3
"""Numbers and SVG for the debugging-RL-runs lesson.

Figure 1 traces the relative entropy of a softmax policy on a four-armed
Bernoulli bandit (means 0.2, 0.4, 0.6, 0.5) under REINFORCE with no baseline.
Rewards are 0 or 1, so every sampled action is pushed up, and a large learning
rate (3.0) sharpens the policy onto whichever arm it tried first before it has
learned which arm is best. A small learning rate (0.1) keeps exploring.

Figure 2 is the norm of the policy-gradient loss's gradient on one fixed batch
as the advantages are multiplied by a reward scale from 0.01 to 1000, with and
without advantage normalization.

The functions come from the exercise's solution file, so the lesson's numbers
and the checks agree. Run from the repo root:

    python3 scripts/figures/053-debugging-rl-runs.py
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "solution", ROOT / "content/exercises/053-debugging-rl-runs/solution.py"
)
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)

MEANS = np.array([0.2, 0.4, 0.6, 0.5])
UPDATES = 300


def run_bandit(lr: float, seed: int):
    rng = np.random.default_rng(seed)
    logits = np.zeros(4)
    trace, favorite = [], []
    for _ in range(UPDATES):
        p = np.exp(logits - logits.max())
        p /= p.sum()
        trace.append(sol.relative_entropy(p[None, :]))
        a = rng.choice(4, p=p)
        r = float(rng.random() < MEANS[a])
        onehot = np.eye(4)[a]
        logits = logits + lr * r * (onehot - p)
        favorite.append(int(np.argmax(p)))
    return np.array(trace), favorite[-1]


def grad_norms():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(64, 3))
    actions = rng.integers(0, 3, size=64)
    adv = rng.normal(size=64)
    scales = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    raw = [sol.grad_norm([sol.policy_loss_grad(logits, actions, s * adv, False)]) for s in scales]
    norm = [sol.grad_norm([sol.policy_loss_grad(logits, actions, s * adv, True)]) for s in scales]
    return scales, raw, norm


def figure_1(seed: int = 2):
    lo, fav_lo = run_bandit(0.1, seed)
    hi, fav_hi = run_bandit(3.0, seed)
    x0, x1, y0, y1 = 70.0, 570.0, 30.0, 190.0
    sx = lambda i: x0 + (x1 - x0) * i / (UPDATES - 1)  # noqa: E731
    sy = lambda v: y1 - (y1 - y0) * v  # noqa: E731
    out = ['<svg viewBox="0 0 600 250" role="img" aria-label="Relative entropy of a bandit policy over 300 updates with a small and a large learning rate">']
    for v in (0.0, 0.5, 1.0):
        out.append(f'<line x1="{x0}" y1="{sy(v):.1f}" x2="{x1}" y2="{sy(v):.1f}" class="stroke-grid"/>')
        out.append(f'<text x="{x0 - 8}" y="{sy(v) + 4:.1f}" class="fig-small" text-anchor="end">{v:g}</text>')
    thr = sy(0.05)
    out.append(f'<line x1="{x0}" y1="{thr:.1f}" x2="{x1}" y2="{thr:.1f}" class="stroke-bad dashed"/>')
    out.append(f'<text x="{x1 - 4}" y="{thr - 6:.1f}" class="fig-small" text-anchor="end">collapse threshold 0.05</text>')
    for i in (0, 100, 200, 299):
        out.append(f'<text x="{sx(i):.1f}" y="{y1 + 18}" class="fig-small" text-anchor="middle">{i}</text>')
    for series, cls in ((lo, "stroke-a"), (hi, "stroke-b")):
        pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(series))
        out.append(f'<polyline points="{pts}" class="{cls}" fill="none"/>')
    out.append(f'<text x="{sx(150):.1f}" y="{sy(lo[150]) - 8:.1f}" class="fig-small fig-bold" text-anchor="middle">learning rate 0.1</text>')
    out.append(f'<text x="{sx(40):.1f}" y="{sy(0.18):.1f}" class="fig-small fig-bold" text-anchor="start">learning rate 3.0</text>')
    out.append(f'<text x="{(x0 + x1) / 2}" y="{y1 + 40}" class="fig-small" text-anchor="middle">policy update</text>')
    out.append(f'<text x="{x0}" y="20" class="fig-small" text-anchor="start">relative entropy of the policy</text>')
    out.append("</svg>")
    return "\n".join(out), lo, hi, fav_lo, fav_hi


def figure_2():
    scales, raw, norm = grad_norms()
    x0, x1, y0, y1 = 80.0, 560.0, 30.0, 200.0
    lg = lambda v: math.log10(v)  # noqa: E731
    sx = lambda s: x0 + (x1 - x0) * (lg(s) - lg(0.01)) / (lg(1000) - lg(0.01))  # noqa: E731
    lo, hi = -3.0, 3.0
    sy = lambda g: y1 - (y1 - y0) * (lg(g) - lo) / (hi - lo)  # noqa: E731
    out = ['<svg viewBox="0 0 600 270" role="img" aria-label="Gradient norm against reward scale, with and without advantage normalization">']
    for e in range(-3, 4):
        out.append(f'<line x1="{x0}" y1="{sy(10.0 ** e):.1f}" x2="{x1}" y2="{sy(10.0 ** e):.1f}" class="stroke-grid"/>')
        out.append(f'<text x="{x0 - 8}" y="{sy(10.0 ** e) + 4:.1f}" class="fig-small" text-anchor="end">{10.0 ** e:g}</text>')
    for s in scales:
        out.append(f'<text x="{sx(s):.1f}" y="{y1 + 18}" class="fig-small" text-anchor="middle">{s:g}</text>')
    for series, cls in ((raw, "stroke-b"), (norm, "stroke-a")):
        pts = " ".join(f"{sx(s):.1f},{sy(g):.1f}" for s, g in zip(scales, series))
        out.append(f'<polyline points="{pts}" class="{cls}" fill="none"/>')
        for s, g in zip(scales, series):
            fill = "fill-b" if cls == "stroke-b" else "fill-a"
            out.append(f'<circle cx="{sx(s):.1f}" cy="{sy(g):.1f}" r="4" class="{fill}"/>')
    out.append(f'<text x="{sx(0.05):.1f}" y="{sy(raw[1]) - 12:.1f}" class="fig-small fig-bold" text-anchor="start">raw advantages</text>')
    out.append(f'<text x="{sx(0.05):.1f}" y="{sy(norm[1]) - 10:.1f}" class="fig-small fig-bold" text-anchor="start">normalized advantages</text>')
    out.append(f'<text x="{(x0 + x1) / 2}" y="{y1 + 42}" class="fig-small" text-anchor="middle">reward scale (advantages multiplied by this)</text>')
    out.append(f'<text x="{x0}" y="20" class="fig-small" text-anchor="start">gradient norm of the policy loss</text>')
    out.append("</svg>")
    return "\n".join(out), scales, raw, norm


if __name__ == "__main__":
    svg1, lo, hi, fav_lo, fav_hi = figure_1()
    print(f"lr 0.1 : relative entropy at update 0/100/200/299 = {lo[0]:.3f} {lo[100]:.3f} {lo[200]:.3f} {lo[299]:.3f}; favorite arm {fav_lo}")
    print(f"lr 3.0 : relative entropy at update 0/10/50/299 = {hi[0]:.3f} {hi[10]:.3f} {hi[50]:.3f} {hi[299]:.3f}; favorite arm {fav_hi}")
    below = int(np.argmax(hi < 0.05)) if (hi < 0.05).any() else None
    print("lr 3.0 first update below 0.05:", below)
    print(svg1)
    svg2, scales, raw, norm = figure_2()
    for s, r, n in zip(scales, raw, norm):
        print(f"scale {s:7g}: raw {r:.4f}  normalized {n:.4f}")
    print(svg2)
