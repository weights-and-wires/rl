#!/usr/bin/env python3
"""Numbers and SVG for the sample-complexity and regret lesson.

Figure 1 plots the true gap max |Q^pi - Q_hat^pi| against the simulation-lemma
bound on the 200 random MDP pairs the exercise checks. Figure 2 plots the mean
cumulative regret over five seeds on RiverSwim (horizon 20, 300 episodes) for
epsilon-greedy and UCB-VI at two bonus scales. The script also prints every
number the lesson quotes.

Run from the repo root:

    python3 scripts/figures/047-sample-complexity-and-regret.py            # numbers
    python3 scripts/figures/047-sample-complexity-and-regret.py --svg DIR  # also write fig1.svg, fig2.svg
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend/app/runner"))
sys.path.insert(0, str(ROOT / "content/exercises/047-sample-complexity-and-regret"))

import rlenvs  # noqa: E402
import solution as sol  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)


def lemma_pairs():
    """The same 200 pairs as tests.py: (bound, gap) for each."""
    out = []
    for i in range(200):
        rng = np.random.default_rng(i)
        mdp = rlenvs.random_mdp(5, 3, 0.9, seed=i)
        other = rlenvs.random_mdp(5, 3, 0.9, seed=1000 + i)
        w = rng.uniform(0.02, 0.5)
        P_hat = (1 - w) * mdp.P + w * other.P
        pi = rng.dirichlet(np.ones(3), size=5)
        gap = np.abs(sol.policy_q(mdp.P, mdp.R, 0.9, pi) - sol.policy_q(P_hat, mdp.R, 0.9, pi)).max()
        out.append((sol.simulation_lemma_bound(P_hat, mdp.P, mdp.R, 0.9), float(gap)))
    return np.array(out)


def regret_curves():
    mdp = rlenvs.river_swim()
    runs = {"epsilon-greedy": (0.0, 0.1), "UCB-VI, bonus scale 1": (1.0, 0.0), "UCB-VI, bonus scale 0.1": (0.1, 0.0)}
    curves = {}
    for name, (scale, eps) in runs.items():
        cum = []
        for seed in SEEDS:
            _, _, reg = sol.ucbvi(mdp, 300, 20, 0.1, seed, bonus_scale=scale, epsilon=eps)
            cum.append(np.cumsum(reg))
        curves[name] = np.mean(cum, axis=0)
    return curves


def lx(v, lo, hi, x0, x1):
    return x0 + (math.log10(v) - lo) / (hi - lo) * (x1 - x0)


def fig1(pairs):
    xlo, xhi = 0.0, 3.0  # bound axis: 1 to 1000
    ylo, yhi = -2.0, 1.0  # gap axis: 0.01 to 10
    x0, x1, y0, y1 = 70.0, 540.0, 270.0, 40.0
    s = ['<svg viewBox="0 0 600 330" role="img" aria-label="Scatter plot on log axes: the true Q error of 200 random MDP pairs against the simulation-lemma bound; every point lies far below the diagonal where the two are equal">']
    for d in range(0, 4):
        x = lx(10.0**d, xlo, xhi, x0, x1)
        s.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" class="stroke-grid"/>')
        s.append(f'<text x="{x:.1f}" y="{y0 + 18}" class="fig-small" text-anchor="middle">1e{d}</text>')
    for d in range(-2, 2):
        y = lx(10.0**d, ylo, yhi, y0, y1)
        s.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" class="stroke-grid"/>')
        s.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" class="fig-small" text-anchor="end">1e{d}</text>')
    s.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" class="stroke-muted"/>')
    s.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" class="stroke-muted"/>')
    s.append(f'<line x1="{lx(1.0, xlo, xhi, x0, x1):.1f}" y1="{lx(1.0, ylo, yhi, y0, y1):.1f}" x2="{lx(10.0, xlo, xhi, x0, x1):.1f}" y2="{lx(10.0, ylo, yhi, y0, y1):.1f}" class="stroke-bad dashed"/>')
    for b, g in pairs:
        s.append(f'<circle cx="{lx(b, xlo, xhi, x0, x1):.1f}" cy="{lx(g, ylo, yhi, y0, y1):.1f}" r="2.5" class="fill-a"/>')
    s.append(f'<text x="{lx(10.0, xlo, xhi, x0, x1) + 8:.1f}" y="{lx(10.0, ylo, yhi, y0, y1) + 12:.1f}" class="fig-small" text-anchor="start">gap = bound</text>')
    s.append(f'<text x="{(x0 + x1) / 2}" y="314" class="fig-small" text-anchor="middle">simulation-lemma bound</text>')
    s.append(f'<text x="14" y="{(y0 + y1) / 2}" class="fig-small" text-anchor="middle" transform="rotate(-90 14 {(y0 + y1) / 2})">true max |Q error|</text>')
    s.append('</svg>')
    return "\n".join(s)


def fig2(curves):
    x0, x1, y0, y1 = 70.0, 540.0, 270.0, 40.0
    ymax = 2000.0
    s = ['<svg viewBox="0 0 600 340" role="img" aria-label="Cumulative regret over 300 episodes on RiverSwim: epsilon-greedy and UCB-VI with the full bonus grow linearly, UCB-VI with a tenth of the bonus flattens">']
    for v in (0, 500, 1000, 1500, 2000):
        y = y0 - v / ymax * (y0 - y1)
        s.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" class="stroke-grid"/>')
        s.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" class="fig-small" text-anchor="end">{v}</text>')
    s.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" class="stroke-muted"/>')
    for e in (0, 100, 200, 300):
        x = x0 + e / 300 * (x1 - x0)
        s.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0 + 5}" class="stroke-muted"/>')
        s.append(f'<text x="{x:.1f}" y="{y0 + 20}" class="fig-small" text-anchor="middle">{e}</text>')
    s.append(f'<text x="{(x0 + x1) / 2}" y="318" class="fig-small" text-anchor="middle">episode</text>')
    classes = {"epsilon-greedy": "stroke-a", "UCB-VI, bonus scale 1": "stroke-b", "UCB-VI, bonus scale 0.1": "stroke-c"}
    for ly, (name, cum) in zip((50, 68, 86), curves.items()):
        pts = " ".join(f"{x0 + (k + 1) / 300 * (x1 - x0):.1f},{y0 - cum[k] / ymax * (y0 - y1):.1f}" for k in range(0, 300, 5))
        s.append(f'<polyline points="{pts}" class="{classes[name]}" fill="none"/>')
        s.append(f'<line x1="{x0 + 30}" y1="{ly}" x2="{x0 + 60}" y2="{ly}" class="{classes[name]}"/>')
        s.append(f'<text x="{x0 + 66}" y="{ly + 4}" class="fig-small" text-anchor="start">{name}: {cum[-1]:.0f} at episode 300</text>')
    s.append('</svg>')
    return "\n".join(s)


def main() -> None:
    pairs = lemma_pairs()
    ratio = pairs[:, 1] / pairs[:, 0]
    print(f"pairs: {len(pairs)}, every gap below bound: {bool((pairs[:, 1] <= pairs[:, 0]).all())}")
    print(f"gap / bound: min {ratio.min():.5f}, median {np.median(ratio):.5f}, max {ratio.max():.5f}")
    print(f"bound range {pairs[:, 0].min():.3f} to {pairs[:, 0].max():.3f}; gap range {pairs[:, 1].min():.5f} to {pairs[:, 1].max():.5f}")

    P = np.array([[[0.8, 0.2]], [[0.0, 1.0]]])
    P_hat = np.array([[[0.6, 0.4]], [[0.0, 1.0]]])
    R = np.array([[1.0], [0.0]])
    pi = np.ones((2, 1))
    print("two-state: Q =", sol.policy_q(P, R, 0.5, pi).ravel(), "Q_hat =", sol.policy_q(P_hat, R, 0.5, pi).ravel())
    print("two-state: gap =", sol.simulation_gap(P_hat, P, R, 0.5, pi).ravel(), "bound =", sol.simulation_lemma_bound(P_hat, P, R, 0.5))

    print("hoeffding_bonus [[0,1],[4,100]], H=10, delta=0.1:")
    print(np.round(sol.hoeffding_bonus(np.array([[0, 1], [4, 100]]), 10, 0.1), 4))
    mdp = rlenvs.river_swim()
    for H in (20, 5):
        vstar = float(mdp.mu0 @ sol.backward_induction(mdp.P, mdp.R, H).max(axis=2)[0])
        print(f"river_swim horizon {H}: V*_0 = {vstar:.4f}")
    for n in (1, 10, 100, 1000, 10000):
        print(f"RiverSwim N={n}: bonus at H=20 {float(sol.hoeffding_bonus(np.array([n]), 20, 0.1)[0] * 1):.3f}")
    print("bonus uses N.size=1 above; the exercise uses S*A = 12 pairs")
    N12 = np.full((6, 2), 1000)
    print("H=20, N=1000 per pair, S=6 A=2:", float(sol.hoeffding_bonus(N12, 20, 0.1)[0, 0]))

    n = np.arange(1, 1001)
    print("1/n sums", sol.step_size_sums(1.0 / n), "var", sol.estimator_variance(1.0 / n, 1.0))
    print("0.1 sums", sol.step_size_sums(np.full(1000, 0.1)), "var", sol.estimator_variance(np.full(1000, 0.1), 1.0))

    curves = regret_curves()
    for name, cum in curves.items():
        print(f"{name}: mean cumulative regret {cum[49]:.1f} at 50, {cum[149]:.1f} at 150, {cum[-1]:.1f} at 300")
    if "--svg" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--svg") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "fig1.svg").write_text(fig1(pairs) + "\n")
        (out / "fig2.svg").write_text(fig2(curves) + "\n")


if __name__ == "__main__":
    main()
