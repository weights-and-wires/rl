#!/usr/bin/env python3
"""Numbers and figures for the goal-conditioned RL and HER lesson.

Three computations feed the lesson:

* The exact chance that a uniformly random policy reaches a random goal within n
  steps on n-bit bit-flipping, from a Markov chain on the Hamming distance.
* The optimal goal-conditioned value Q*(s, a, g) at distance d with gamma = 0.9.
* Greedy success rate during training on 12 bits for four relabeling strategies,
  averaged over three seeds, using the exercise's reference solution.

Run from the repo root. Pass --svg DIR to also write figure1.svg and figure2.svg,
which the lesson pastes in:

    python3 scripts/figures/048-goal-conditioned-rl-and-her.py
    python3 scripts/figures/048-goal-conditioned-rl-and-her.py --svg /path/to/dir
"""

from __future__ import annotations

import argparse
import sys
from math import comb
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "content" / "exercises" / "048-goal-conditioned-rl-and-her"))

import solution as ex  # noqa: E402

N_VALUES = (4, 6, 8, 10, 12, 16, 20)
STRATEGIES = ("none", "final", "episode", "future")
SEEDS = (0, 1, 2)
EPISODES, EVERY = 500, 50


def random_success(n: int) -> float:
    """P(a uniformly random policy reaches a random goal within n steps)."""
    p = np.array([comb(n, d) / 2**n for d in range(n + 1)])  # distance of a random start
    reached = 0.0
    for _ in range(n):
        q = np.zeros(n + 1)
        for d in range(n + 1):
            if d > 0:
                q[d - 1] += p[d] * d / n
            if d < n:
                q[d + 1] += p[d] * (n - d) / n
        reached += q[0]
        q[0] = 0.0  # absorbed on success
        p = q
    return float(reached)


def q_star(d: int, gamma: float = 0.9) -> float:
    return -(1 - gamma ** (d - 1)) / (1 - gamma)


def curve(strategy: str, seed: int, n: int = 12) -> list[float]:
    rng = np.random.default_rng(seed)
    env = ex.BitFlipEnv(n, seed=seed)
    q_net, target_net = ex.make_qnet(n, seed=seed), ex.make_qnet(n, seed=seed)
    target_net.load_state_dict(q_net.state_dict())
    opt = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    buffer, updates, out = [], 0, [ex.success_rate(env, q_net)]
    for e in range(1, EPISODES + 1):
        buffer.extend(ex.relabel(ex.collect_episode(env, q_net, 0.2, rng), strategy, 4, rng))
        for _ in range(n):
            batch = ex.to_batch(buffer, rng.integers(0, len(buffer), 64))
            loss = ex.dqn_loss(q_net, target_net, batch, 0.9)
            opt.zero_grad()
            loss.backward()
            opt.step()
            updates += 1
            if updates % 50 == 0:
                target_net.load_state_dict(q_net.state_dict())
        if e % EVERY == 0:
            out.append(ex.success_rate(env, q_net))
    return out


def figure1(vals: list[float]) -> str:
    x0, x1, y0, y1 = 70.0, 540.0, 40.0, 250.0
    lo, hi = -7.0, 0.0  # log10 axis

    def px(i):
        return x0 + (x1 - x0) * i / (len(N_VALUES) - 1)

    def py(v):
        return y1 - (y1 - y0) * (np.log10(v) - lo) / (hi - lo)

    o = [f'<svg viewBox="0 0 600 310" role="img" aria-label="Chance that a random policy reaches a random goal, falling from {vals[0]:.2f} at 4 bits to {vals[-1]:.1e} at 20 bits">']
    for e in range(-6, 1, 2):
        y = py(10.0**e)
        o.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" class="stroke-grid"/>')
        label = "1" if e == 0 else f"1e{e}"
        o.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" class="fig-small" text-anchor="end">{label}</text>')
    pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(vals))
    o.append(f'<polyline points="{pts}" class="stroke-b" fill="none"/>')
    for i, (n, v) in enumerate(zip(N_VALUES, vals)):
        o.append(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="4" class="fill-b"/>')
        o.append(f'<text x="{px(i):.1f}" y="{y1 + 20}" class="fig-small" text-anchor="middle">{n}</text>')
        anchor = "start" if i == 0 else "middle"
        dx = 8 if i == 0 else 0
        o.append(f'<text x="{px(i) + dx:.1f}" y="{py(v) - 10:.1f}" class="fig-small" text-anchor="{anchor}">{v:.2g}</text>')
    o.append(f'<text x="{(x0 + x1) / 2}" y="{y1 + 42}" class="fig-small" text-anchor="middle">number of bits n (episode length n)</text>')
    o.append("</svg>")
    return "\n".join(o)


def figure2(curves: dict[str, np.ndarray]) -> str:
    x0, x1, y0, y1 = 70.0, 540.0, 40.0, 250.0
    xs = np.arange(0, EPISODES + 1, EVERY)
    cls = {"none": "stroke-muted", "final": "stroke-a", "episode": "stroke-c", "future": "stroke-b"}
    o = ['<svg viewBox="0 0 600 330" role="img" aria-label="Greedy success rate on 12-bit bit-flipping against training episodes for four relabeling strategies">']
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y1 - (y1 - y0) * v
        o.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" class="stroke-grid"/>')
        o.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" class="fig-small" text-anchor="end">{v:g}</text>')
    for e in xs[::2]:
        x = x0 + (x1 - x0) * e / EPISODES
        o.append(f'<text x="{x:.1f}" y="{y1 + 20}" class="fig-small" text-anchor="middle">{e}</text>')
    o.append(f'<text x="{(x0 + x1) / 2}" y="{y1 + 42}" class="fig-small" text-anchor="middle">training episodes</text>')
    for k, strat in enumerate(STRATEGIES):
        c = curves[strat]
        pts = " ".join(f"{x0 + (x1 - x0) * e / EPISODES:.1f},{y1 - (y1 - y0) * v:.1f}" for e, v in zip(xs, c))
        o.append(f'<polyline points="{pts}" class="{cls[strat]}" fill="none"/>')
        ly = 56 + 18 * k
        o.append(f'<line x1="330" y1="{ly}" x2="360" y2="{ly}" class="{cls[strat]}"/>')
        label = "no relabeling" if strat == "none" else f"{strat}"
        o.append(f'<text x="366" y="{ly + 4}" class="fig-small" text-anchor="start">{label}: {c[-1]:.2f} at episode {EPISODES}</text>')
    o.append("</svg>")
    return "\n".join(o)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--svg", type=Path)
    args = parser.parse_args()

    vals = [random_success(n) for n in N_VALUES]
    for n, v in zip(N_VALUES, vals):
        print(f"random policy, n = {n:2d}: success {v:.3g}")
    for d in range(1, 6):
        print(f"Q*(d = {d}) = {q_star(d) + 0.0:.3f}")
    curves = {}
    for strat in STRATEGIES:
        c = np.mean([curve(strat, s) for s in SEEDS], axis=0)
        curves[strat] = c
        print(f"{strat:8s}", " ".join(f"{v:.2f}" for v in c))
    if args.svg:
        args.svg.mkdir(parents=True, exist_ok=True)
        (args.svg / "figure1.svg").write_text(figure1(vals) + "\n")
        (args.svg / "figure2.svg").write_text(figure2(curves) + "\n")


if __name__ == "__main__":
    main()
