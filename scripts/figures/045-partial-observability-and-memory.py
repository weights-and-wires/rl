#!/usr/bin/env python3
"""Numbers and figures for the partial-observability lesson.

Three computations feed the lesson:

* The tiger belief after k listens that all hear the tiger on the left, from
  belief_update in the exercise's reference solution.
* The exact value of the best policy that sees only the current observation on
  the T-maze, against the best policy that sees the full state, and against
  policies that see a window of the last k observations (frame stacking).
* Greedy value during policy-gradient training, with and without memory,
  for seed 0.

Run from the repo root. Pass --svg DIR to also write the three figure bodies
as figure1.svg, figure2.svg, and figure3.svg, which the lesson pastes in:

    python3 scripts/figures/045-partial-observability-and-memory.py
    python3 scripts/figures/045-partial-observability-and-memory.py --svg /path/to/dir
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "app" / "runner"))
sys.path.insert(0, str(ROOT / "content" / "exercises" / "045-partial-observability-and-memory"))

import rlenvs  # noqa: E402
import solution as ex  # noqa: E402

T_TIGER = np.stack([np.eye(2), np.full((2, 2), 0.5), np.full((2, 2), 0.5)])
O_TIGER = np.stack([
    [[0.85, 0.15], [0.15, 0.85]],
    np.full((2, 2), 0.5),
    np.full((2, 2), 0.5),
])


def tiger_beliefs(n: int = 6) -> list[float]:
    b = np.array([0.5, 0.5])
    out = [float(b[0])]
    for _ in range(n):
        b = ex.belief_update(b, 0, 0, T_TIGER, O_TIGER)
        out.append(float(b[0]))
    return out


def window_obs(obs_of_state: np.ndarray, k: int, length: int = ex.LENGTH) -> np.ndarray:
    """Observation ids when the agent sees its last k raw observations."""
    n_pos = length + 1
    ids: dict[tuple, int] = {}
    out = np.zeros_like(obs_of_state)
    for s in range(len(obs_of_state)):
        if s == len(obs_of_state) - 1:
            key = ("end",)
        else:
            sig, pos = divmod(s, n_pos)
            seen = [obs_of_state[sig * n_pos + p] for p in range(pos + 1)]
            key = tuple(seen[-k:])
        out[s] = ids.setdefault(key, len(ids))
    return out


def best_over_windows(mdp, obs_of_state: np.ndarray) -> float:
    """Best deterministic policy over arbitrary observation ids (the ids need not be 0..4)."""
    ids = np.unique(obs_of_state[~mdp.terminal])
    n_obs = int(obs_of_state.max()) + 1
    best = -np.inf
    for choice in itertools.product(range(mdp.n_actions), repeat=len(ids)):
        pi_obs = np.zeros((n_obs, mdp.n_actions))
        pi_obs[:, 0] = 1.0
        pi_obs[ids] = np.eye(mdp.n_actions)[list(choice)]
        best = max(best, ex.memoryless_value(mdp, obs_of_state, pi_obs))
    return best


def learning_curve(memory: bool, seed: int = 0, iters: int = 150, every: int = 5) -> list[tuple[int, float]]:
    """The reference train_policy loop, recording the exact greedy value."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = ex.PolicyNet(16)
    opt = torch.optim.Adam(net.parameters(), lr=0.02)
    points = [(0, ex.greedy_value(net, memory))]
    for it in range(1, iters + 1):
        data = ex.collect(net, 64, rng, memory)
        G = ex.discounted_returns(data["rewards"], ex.GAMMA)
        mask = data["mask"]
        base = (G * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        first = data["first"] if memory else torch.ones_like(data["first"])
        loss = ex.sequence_loss(net.cell, net.head, data["obs"], data["actions"], G - base, mask, first)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % every == 0:
            points.append((it, ex.greedy_value(net, memory)))
    return points


# --- SVG ---------------------------------------------------------------------

ARROW = (
    '<defs><marker id="c042-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" '
    'markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
    'class="fill-ink"/></marker></defs>'
)


def svg_maze(full: float, memoryless: float) -> str:
    g = ex.GAMMA ** ex.LENGTH
    out = [
        '<svg viewBox="0 0 600 260" role="img" aria-label="A T-maze: a signal at the start, four identical corridor steps, and a junction with two doors">',
        ARROW,
    ]
    names = ["start", "step 1", "step 2", "step 3", "junction"]
    for i, name in enumerate(names):
        x = 30 + i * 96
        cls = "fill-soft-b" if i == 4 else ("fill-soft-a" if i == 0 else "fill-sunk")
        out.append(f'<rect x="{x}" y="110" width="76" height="46" rx="6" class="{cls} stroke-muted"/>')
        out.append(f'<text x="{x + 38}" y="138" class="fig-bold" text-anchor="middle">{name}</text>')
        if i < 4:
            out.append(f'<line x1="{x + 78}" y1="133" x2="{x + 94}" y2="133" class="stroke-ink" marker-end="url(#c042-arrow)"/>')
    out.append('<text x="68" y="76" class="fig-small" text-anchor="middle">sees the signal</text>')
    out.append('<text x="68" y="92" class="fig-small" text-anchor="middle">left or right, 50/50</text>')
    out.append('<line x1="68" y1="98" x2="68" y2="108" class="stroke-a" marker-end="url(#c042-arrow)"/>')
    out.append('<text x="260" y="196" class="fig-small" text-anchor="middle">steps 1 to 3 look identical: the signal is gone</text>')
    out.append('<text x="260" y="214" class="fig-small" text-anchor="middle">so the agent must carry it in memory</text>')
    jx = 30 + 4 * 96
    out.append(f'<line x1="{jx + 38}" y1="108" x2="{jx + 38}" y2="66" class="stroke-b" marker-end="url(#c042-arrow)"/>')
    out.append(f'<line x1="{jx + 38}" y1="158" x2="{jx + 38}" y2="200" class="stroke-b" marker-end="url(#c042-arrow)"/>')
    out.append(f'<text x="{jx + 38}" y="54" class="fig-small" text-anchor="middle">left door: +1 or -1</text>')
    out.append(f'<text x="{jx + 38}" y="222" class="fig-small" text-anchor="middle">right door: +1 or -1</text>')
    out.append(f'<text x="540" y="246" class="fig-small" text-anchor="end">full state: {full:.4f}   memoryless: {memoryless:.4f}   (0.95^4 = {g:.4f})</text>')
    out.append("</svg>")
    return "\n".join(out)


def scale(v: float, lo: float, hi: float, a: float, b: float) -> float:
    return a + (v - lo) / (hi - lo) * (b - a)


def svg_belief(values: list[float]) -> str:
    x0, x1, y0, y1 = 70.0, 540.0, 250.0, 40.0
    out = [
        '<svg viewBox="0 0 600 310" role="img" aria-label="Belief that the tiger is on the left after k listens that all hear left, rising from 0.5 toward 1">',
        ARROW,
    ]
    for tick in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        y = scale(tick, 0.5, 1.0, y0, y1)
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" class="stroke-grid"/>')
        out.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" class="fig-small" text-anchor="end">{tick:.1f}</text>')
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" class="stroke-muted"/>')
    pts = []
    for k, v in enumerate(values):
        x = scale(k, 0, len(values) - 1, x0, x1)
        y = scale(v, 0.5, 1.0, y0, y1)
        pts.append((x, y, k, v))
        out.append(f'<text x="{x:.1f}" y="{y0 + 18}" class="fig-small" text-anchor="middle">{k}</text>')
    out.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in pts) + '" class="stroke-a" fill="none"/>')
    for x, y, k, v in pts:
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" class="fill-a"/>')
        if k in (0, 1, 2, 3):
            out.append(f'<text x="{x + 8:.1f}" y="{y + 18:.1f}" class="fig-small">{v:.4f}</text>')
    out.append(f'<text x="{(x0 + x1) / 2}" y="292" class="fig-small" text-anchor="middle">listens, each one hearing the tiger on the left</text>')
    out.append(f'<text x="22" y="{(y0 + y1) / 2}" class="fig-small" text-anchor="middle" transform="rotate(-90 22 {(y0 + y1) / 2})">belief: tiger on the left</text>')
    out.append("</svg>")
    return "\n".join(out)


def svg_curves(with_mem, no_mem, full: float) -> str:
    x0, x1, y0, y1 = 70.0, 540.0, 250.0, 40.0
    lo, hi = -1.1, 0.9
    xmax = max(p[0] for p in with_mem)
    out = [
        '<svg viewBox="0 0 600 310" role="img" aria-label="Greedy value during training: the recurrent policy rises to 0.8145 while the memoryless policy stays at 0">',
        ARROW,
    ]
    for tick in (-1.0, -0.5, 0.0, 0.5):
        y = scale(tick, lo, hi, y0, y1)
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" class="stroke-grid"/>')
        out.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" class="fig-small" text-anchor="end">{tick:.1f}</text>')
    out.append(f'<text x="{(x0 + x1) / 2}" y="{y0 + 40}" class="fig-small" text-anchor="middle">policy-gradient iteration</text>')
    for tick in range(0, xmax + 1, 50):
        x = scale(tick, 0, xmax, x0, x1)
        out.append(f'<text x="{x:.1f}" y="{y0 + 22}" class="fig-small" text-anchor="middle">{tick}</text>')
    yf = scale(full, lo, hi, y0, y1)
    out.append(f'<line x1="{x0}" y1="{yf:.1f}" x2="{x1}" y2="{yf:.1f}" class="stroke-muted dashed"/>')
    out.append(f'<text x="{x1}" y="{yf - 6:.1f}" class="fig-small" text-anchor="end">full-state optimum {full:.4f}</text>')

    def poly(points, cls):
        return '<polyline points="' + " ".join(
            f"{scale(i, 0, xmax, x0, x1):.1f},{scale(v, lo, hi, y0, y1):.1f}" for i, v in points) + f'" class="{cls}" fill="none"/>'

    out.append(poly(no_mem, "stroke-a"))
    out.append(poly(with_mem, "stroke-b"))
    out.append('<line x1="300" y1="190" x2="330" y2="190" class="stroke-b"/>')
    out.append(f'<text x="336" y="194" class="fig-small">with memory: {with_mem[-1][1]:.4f} at iteration {xmax}</text>')
    out.append('<line x1="300" y1="210" x2="330" y2="210" class="stroke-a"/>')
    out.append(f'<text x="336" y="214" class="fig-small">no memory: {no_mem[-1][1]:.4f} at iteration {xmax}</text>')
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--svg", type=Path, default=None)
    args = ap.parse_args()

    beliefs = tiger_beliefs()
    print("tiger belief after k listens (all hear left):", np.round(beliefs, 4).tolist())
    # Worked example in the lesson.
    b1 = ex.belief_update(np.array([0.5, 0.5]), 0, 0, T_TIGER, O_TIGER)
    b2 = ex.belief_update(b1, 0, 0, T_TIGER, O_TIGER)
    b3 = ex.belief_update(b2, 0, 1, T_TIGER, O_TIGER)
    b_open = ex.belief_update(np.array([0.97, 0.03]), 1, 0, T_TIGER, O_TIGER)
    print("after one listen:", np.round(b1, 4), "two:", np.round(b2, 4), "then a right:", np.round(b3, 4))
    print("after opening a door from [0.97, 0.03]:", b_open)
    assert np.allclose(b1, [0.85, 0.15])
    assert np.allclose(b2, [0.7225 / 0.745, 0.0225 / 0.745])
    assert np.allclose(b3, [0.85, 0.15])  # one hear-right cancels one hear-left
    assert np.allclose(b_open, [0.5, 0.5])

    mdp, obs_of_state = ex.tmaze_mdp()
    full = float(mdp.mu0 @ rlenvs.value_iteration(mdp))
    memoryless, _ = ex.best_memoryless_value(mdp, obs_of_state)
    print(f"full state: {full:.6f}   best memoryless: {memoryless:.6f}   gamma^4: {ex.GAMMA ** 4:.6f}")
    assert abs(full - ex.GAMMA ** 4) < 1e-12 and abs(memoryless) < 1e-12

    print("best deterministic policy that sees only the last k observations:")
    for k in (1, 2, 3, 4, 5):
        v = best_over_windows(mdp, window_obs(obs_of_state, k))
        print(f"  k = {k}: {v:.6f}")

    with_mem = learning_curve(True)
    no_mem = learning_curve(False)
    print("with memory:", [(i, round(v, 3)) for i, v in with_mem[::6]])
    print("no memory:  ", [(i, round(v, 3)) for i, v in no_mem[::6]])
    assert with_mem[-1][1] > 0.8 and abs(no_mem[-1][1]) < 1e-9

    if args.svg:
        args.svg.mkdir(parents=True, exist_ok=True)
        (args.svg / "figure1.svg").write_text(svg_maze(full, memoryless) + "\n")
        (args.svg / "figure2.svg").write_text(svg_belief(beliefs) + "\n")
        (args.svg / "figure3.svg").write_text(svg_curves(with_mem, no_mem, full) + "\n")
        print(f"wrote figure1.svg, figure2.svg, figure3.svg to {args.svg}")


if __name__ == "__main__":
    main()
