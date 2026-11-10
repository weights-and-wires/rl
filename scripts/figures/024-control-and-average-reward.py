#!/usr/bin/env python3
"""Numbers and SVG for the control-and-average-reward lesson.

It prints the worked examples from the lesson and the data behind its three
figures. With `--svg` it also prints each figure's SVG, which the lesson embeds.
It uses the reference solution, so the numbers match the exercise's checks.

Run from the repo root:

    python3 scripts/figures/024-control-and-average-reward.py [--svg]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "app" / "runner"))
sys.path.insert(0, str(ROOT / "content" / "exercises" / "024-control-and-average-reward"))

import solution as sol  # noqa: E402

N_EPISODES = 30
SEEDS = range(5)
CAP = 1000
RING_STEPS = 20000


def ring():
    stay = np.array([0.5, 0.2, 0.2, 0.2])
    P = np.zeros((4, 4))
    for i in range(4):
        P[i, i] = stay[i]
        P[i, (i + 1) % 4] = 1.0 - stay[i]
    return P, np.array([0.0, 0.0, 0.0, 4.0])


def worked_examples() -> None:
    print("Fourier order 2 at x = 0.25:", np.round(sol.fourier_features([0.25], 2), 4))
    print("Fourier order 1 at (0.25, 0.5):", np.round(sol.fourier_features([0.25, 0.5], 1), 4))
    print("RBF at 0.5, centers 0, 0.5, 1, sigma 0.5:",
          np.round(sol.rbf_features([0.5], np.array([[0.0], [0.5], [1.0]]), 0.5), 4))
    g = sol.n_step_return([-1.0, -1.0, -1.0], -10.0, 0.9, False)
    print(f"3-step return, rewards -1, q = -10, gamma = 0.9: {g:.4f}")
    print(f"terminal 2-step window [-1, -1], gamma = 0.9: {sol.n_step_return([-1.0, -1.0], -10.0, 0.9, True):.4f}")
    w, rb = sol.differential_td_update(
        np.array([1.0, 2.0, 0.0]), 1.5, np.array([1.0, 0.0, 1.0]), 3.0, np.array([0.0, 1.0, 1.0]), 0.1, 0.02
    )
    print(f"differential update: w = {w.tolist()}, r_bar = {rb:.4f}")
    P, r = ring()
    mu = sol.stationary_distribution(P)
    rbar = float(mu @ r)
    print(f"ring: mu = {np.round(mu, 4).tolist()}, average reward = {rbar:.4f}")
    for gamma in (0.9, 0.99):
        v = np.linalg.solve(np.eye(4) - gamma * P, r)
        print(f"gamma = {gamma}: mu @ v = {mu @ v:.4f}, r_bar / (1 - gamma) = {rbar / (1 - gamma):.4f}, "
              f"(1 - gamma) * (mu @ v) = {(1 - gamma) * (mu @ v):.4f}")
    h = np.linalg.solve(np.eye(4) - P + np.outer(np.ones(4), mu), r - rbar)
    print("ring differential values minus h[0]:", np.round(h - h[0], 4).tolist())
    P4 = np.roll(np.eye(4), 1, axis=1)
    r4 = np.array([0.0, 0.0, 0.0, 4.0])
    w, rb = sol.differential_td(P4, r4, np.eye(4), 4000, 0.1, 0.01)
    print(f"cycle: r_bar = {rb:.6f}, values minus v[0] = {np.round(w - w[0], 4).tolist()}")


def learning_curves() -> tuple[np.ndarray, np.ndarray]:
    one = np.array([sol.n_step_sarsa(1, N_EPISODES, seed=s)[1] for s in SEEDS])
    eight = np.array([sol.n_step_sarsa(8, N_EPISODES, seed=s)[1] for s in SEEDS])
    for s in SEEDS:
        print(f"seed {s}: n = 1 mean {one[s].mean():.1f}, n = 8 mean {eight[s].mean():.1f}; "
              f"last 5: {one[s][-5:].mean():.1f} and {eight[s][-5:].mean():.1f}")
    print(f"mean over seeds, episode 1/10/30: n = 1 {one.mean(0)[[0, 9, 29]].round(0).tolist()}, "
          f"n = 8 {eight.mean(0)[[0, 9, 29]].round(0).tolist()}")
    return one, eight


def ring_trace() -> tuple[np.ndarray, float]:
    P, r = ring()
    exact = float(sol.stationary_distribution(P) @ r)
    rng = np.random.default_rng(0)
    w, rb, s = np.zeros(4), 0.0, 0
    trace = [rb]
    for _ in range(RING_STEPS):
        s2 = int(rng.choice(4, p=P[s]))
        w, rb = sol.differential_td_update(w, rb, np.eye(4)[s], float(r[s]), np.eye(4)[s2], 0.05, 0.002)
        s = s2
        trace.append(rb)
    print(f"r_bar after {RING_STEPS} steps: {trace[-1]:.4f}, exact {exact:.4f}")
    return np.array(trace), exact


def svg_fourier() -> str:
    x0, x1, y0, y1 = 70.0, 570.0, 210.0, 40.0
    xs = np.linspace(0, 1, 101)

    def px(x):
        return x0 + (x1 - x0) * x

    def py(v):
        return y0 + (y1 - y0) * (v + 1) / 2

    out = ['<svg viewBox="0 0 600 270" role="img" aria-label="Four Fourier basis functions on the interval from 0 to 1: the constant 1 and cosines with 1, 2, and 3 half-waves">']
    for v in (-1, 0, 1):
        out.append(f'<line x1="{x0}" y1="{py(v):.1f}" x2="{x1}" y2="{py(v):.1f}" class="stroke-grid"/>')
        out.append(f'<text x="{x0 - 8}" y="{py(v) + 4:.1f}" class="fig-small" text-anchor="end">{v}</text>')
    for x in (0, 0.25, 0.5, 0.75, 1):
        out.append(f'<text x="{px(x):.1f}" y="228.0" class="fig-small" text-anchor="middle">{x:g}</text>')
    out.append('<text x="320.0" y="258.0" text-anchor="middle">x</text>')
    for c, cls, dash in ((0, "stroke-muted", " dashed"), (1, "stroke-a", ""), (2, "stroke-b", ""), (3, "stroke-c", "")):
        pts = " ".join(f"{px(x):.1f},{py(math.cos(math.pi * c * x)):.1f}" for x in xs)
        out.append(f'<polyline points="{pts}" class="{cls}{dash}" fill="none"/>')
    for i, (c, cls, dash) in enumerate(((0, "stroke-muted", " dashed"), (1, "stroke-a", ""), (2, "stroke-b", ""), (3, "stroke-c", ""))):
        y = 20.0
        x = 150.0 + 100.0 * i
        out.append(f'<line x1="{x}" y1="{y - 4}" x2="{x + 24}" y2="{y - 4}" class="{cls}{dash}"/>')
        out.append(f'<text x="{x + 30}" y="{y}" class="fig-small" text-anchor="start">c = {c}</text>')
    out.append("</svg>")
    return "\n".join(out)


def svg_curves(one: np.ndarray, eight: np.ndarray) -> str:
    x0, x1, y0, y1 = 70.0, 570.0, 220.0, 40.0

    def px(i):
        return x0 + (x1 - x0) * i / (N_EPISODES - 1)

    def py(v):
        return y0 + (y1 - y0) * v / CAP

    def line(values, cls):
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(values))
        return f'<polyline points="{pts}" class="{cls}" fill="none"/>'

    out = ['<svg viewBox="0 0 600 270" role="img" aria-label="Mean steps per episode on Mountain Car over 5 seeds for one-step and 8-step semi-gradient SARSA; the 8-step curve sits below the one-step curve">']
    for v in (0, 500, 1000):
        out.append(f'<line x1="{x0}" y1="{py(v):.1f}" x2="{x1}" y2="{py(v):.1f}" class="stroke-grid"/>')
        out.append(f'<text x="{x0 - 8}" y="{py(v) + 4:.1f}" class="fig-small" text-anchor="end">{v}</text>')
    for e in (1, 10, 20, 30):
        out.append(f'<text x="{px(e - 1):.1f}" y="238.0" class="fig-small" text-anchor="middle">{e}</text>')
    out.append('<text x="62.0" y="22.0" class="fig-small" text-anchor="start">steps to reach the flag (cap 1000)</text>')
    out.append('<text x="320.0" y="262.0" text-anchor="middle">episode</text>')
    out.append(line(one.mean(axis=0), "stroke-a"))
    out.append(line(eight.mean(axis=0), "stroke-b"))
    out.append('<line x1="400.0" y1="52.0" x2="430.0" y2="52.0" class="stroke-a"/>')
    out.append('<text x="438.0" y="56.0" class="fig-small" text-anchor="start">n = 1</text>')
    out.append('<line x1="400.0" y1="70.0" x2="430.0" y2="70.0" class="stroke-b"/>')
    out.append('<text x="438.0" y="74.0" class="fig-small" text-anchor="start">n = 8</text>')
    out.append("</svg>")
    return "\n".join(out)


def svg_ring(trace: np.ndarray, exact: float) -> str:
    x0, x1, y0, y1 = 70.0, 570.0, 220.0, 40.0
    top = 1.2

    def px(i):
        return x0 + (x1 - x0) * i / RING_STEPS

    def py(v):
        return y0 + (y1 - y0) * v / top

    pts = " ".join(f"{px(i):.1f},{py(trace[i]):.1f}" for i in range(0, RING_STEPS + 1, 100))
    out = ['<svg viewBox="0 0 600 270" role="img" aria-label="The learned average reward on a four-state ring rises from 0 toward the exact value 0.87 and then wobbles around it">']
    for v in (0, 0.4, 0.8, 1.2):
        out.append(f'<line x1="{x0}" y1="{py(v):.1f}" x2="{x1}" y2="{py(v):.1f}" class="stroke-grid"/>')
        out.append(f'<text x="{x0 - 8}" y="{py(v) + 4:.1f}" class="fig-small" text-anchor="end">{v:g}</text>')
    for k in (0, 5000, 10000, 15000, 20000):
        out.append(f'<text x="{px(k):.1f}" y="238.0" class="fig-small" text-anchor="middle">{k}</text>')
    out.append('<text x="62.0" y="22.0" class="fig-small" text-anchor="start">estimate r_bar</text>')
    out.append('<text x="320.0" y="262.0" text-anchor="middle">step</text>')
    out.append(f'<line x1="{x0}" y1="{py(exact):.1f}" x2="{x1}" y2="{py(exact):.1f}" class="stroke-c dashed"/>')
    out.append(f'<text x="{x1}" y="{py(exact) - 6:.1f}" class="fig-small" text-anchor="end">exact {exact:.4f}</text>')
    out.append(f'<polyline points="{pts}" class="stroke-b" fill="none"/>')
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    worked_examples()
    one, eight = learning_curves()
    trace, exact = ring_trace()
    if "--svg" in sys.argv:
        print("---FIGURE fourier---")
        print(svg_fourier())
        print("---FIGURE curves---")
        print(svg_curves(one, eight))
        print("---FIGURE ring---")
        print(svg_ring(trace, exact))
