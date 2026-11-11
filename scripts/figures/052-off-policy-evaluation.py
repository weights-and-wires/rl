#!/usr/bin/env python3
"""Numbers and SVG for the off-policy evaluation lesson.

Figure 1 plots the effective sample size of whole-trajectory importance
weights against the horizon on a random MDP. Figure 2 plots the mean and
spread of IPS, SNIPS, and doubly robust over many logged bandit batches.
The lesson's worked examples are recomputed and asserted at the end.

Run from the repo root:

    python3 scripts/figures/052-off-policy-evaluation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend" / "app" / "runner"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "content" / "exercises" / "052-off-policy-evaluation"))

import solution as sol  # noqa: E402
import rlenvs  # noqa: E402

P_X = np.array([0.5, 0.3, 0.2])
B = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]])
PI = np.array([[0.1, 0.2, 0.7], [0.7, 0.1, 0.2], [0.2, 0.7, 0.1]])


def ess_by_horizon(max_h=20, n=5000):
    mdp = rlenvs.random_mdp(4, 2, gamma=0.9, seed=1)
    b = np.full((4, 2), 0.5)
    pi = np.array([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]])
    S, A, R = sol.sample_episodes(mdp, b, n, max_h, np.random.default_rng(0))
    ratio = pi[S, A] / b[S, A]
    rho = np.cumprod(ratio, axis=1)
    return [(h, sol.effective_sample_size(rho[:, h - 1]) / n) for h in range(1, max_h + 1)]


def bandit_spread(reps=4000, n=20):
    rng = np.random.default_rng(0)
    mu = rng.uniform(0, 1, size=(3, 3))
    truth = float(P_X @ (PI * mu).sum(axis=1))
    q_hat = mu + rng.normal(0, 0.05, size=mu.shape)
    out = {"IPS": [], "SNIPS": [], "DR": []}
    for _ in range(reps):
        x = rng.choice(3, size=n, p=P_X)
        u = rng.random(n)
        a = np.minimum((u[:, None] > np.cumsum(B[x], axis=1)).sum(axis=1), 2)
        r = mu[x, a] + rng.normal(0, 0.1, size=n)
        out["IPS"].append(sol.ips(x, a, r, PI, B))
        out["SNIPS"].append(sol.snips(x, a, r, PI, B))
        out["DR"].append(sol.doubly_robust(x, a, r, PI, B, q_hat))
    return truth, {k: (float(np.mean(v)), float(np.std(v))) for k, v in out.items()}


def svg_ess(points):
    x0, x1, y0, y1 = 70, 540, 210, 30
    pts = " ".join(f"{x0 + (h - 1) / 19 * (x1 - x0):.1f},{y0 - f * (y0 - y1):.1f}" for h, f in points)
    lines = [
        '<svg viewBox="0 0 570 270" role="img" aria-label="Effective sample size as a fraction of 5000 trajectories against horizon: it falls from 1 at horizon 1 to near 0.02 at horizon 20">'
    ]
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        y = y0 - v * (y0 - y1)
        lines.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" class="stroke-grid"/>')
        lines.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" class="fig-small" text-anchor="end">{v:g}</text>')
    for h in (1, 5, 10, 15, 20):
        x = x0 + (h - 1) / 19 * (x1 - x0)
        lines.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0 + 6}" class="stroke-muted"/>')
        lines.append(f'<text x="{x:.1f}" y="{y0 + 22}" class="fig-small" text-anchor="middle">{h}</text>')
    lines.append(f'<polyline points="{pts}" class="stroke-b" fill="none" stroke-width="2.5"/>')
    for h, f in points:
        if h in (1, 5, 10, 20):
            x = x0 + (h - 1) / 19 * (x1 - x0)
            y = y0 - f * (y0 - y1)
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" class="fill-b"/>')
            lines.append(f'<text x="{x + 8:.1f}" y="{y - 8:.1f}" class="fig-small">{f:.3f}</text>')
    lines.append(f'<text x="{(x0 + x1) / 2:.0f}" y="262" class="fig-small" text-anchor="middle">horizon H (steps in each logged trajectory)</text>')
    lines.append(f'<text x="14" y="22" class="fig-small">ESS / N</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def svg_spread(truth, stats):
    lo, hi = 0.0, 0.7
    x0, x1 = 90, 540

    def X(v):
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)

    lines = [
        '<svg viewBox="0 0 570 250" role="img" aria-label="Mean and one standard deviation of IPS, SNIPS, and doubly robust over 4000 logs of 20 samples, against the true value">'
    ]
    for v in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        lines.append(f'<line x1="{X(v):.1f}" y1="196" x2="{X(v):.1f}" y2="202" class="stroke-muted"/>')
        lines.append(f'<text x="{X(v):.1f}" y="218" class="fig-small" text-anchor="middle">{v:.1f}</text>')
    lines.append(f'<line x1="{x0}" y1="196" x2="{x1}" y2="196" class="stroke-muted"/>')
    lines.append(f'<line x1="{X(truth):.1f}" y1="30" x2="{X(truth):.1f}" y2="196" class="stroke-c dashed"/>')
    lines.append(f'<text x="{X(truth) + 6:.1f}" y="24" class="fig-small">true value {truth:.3f}</text>')
    for i, (name, cls) in enumerate([("IPS", "b"), ("SNIPS", "d"), ("DR", "a")]):
        m, s = stats[name]
        y = 70 + i * 55
        lines.append(f'<text x="14" y="{y + 4}" class="fig-bold">{name}</text>')
        lines.append(f'<line x1="{X(m - s):.1f}" y1="{y}" x2="{X(m + s):.1f}" y2="{y}" class="stroke-{cls}" stroke-width="3"/>')
        lines.append(f'<circle cx="{X(m):.1f}" cy="{y}" r="6" class="fill-{cls}"/>')
        lines.append(f'<text x="{X(m + s) + 8:.1f}" y="{y + 4}" class="fig-small">mean {m:.3f}, std {s:.3f}</text>')
    lines.append(f'<text x="{(x0 + x1) / 2:.0f}" y="240" class="fig-small" text-anchor="middle">estimate of the target policy value</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def worked_examples():
    pi = np.array([[0.8, 0.2], [0.1, 0.9]])
    b = np.full((2, 2), 0.5)
    q = np.array([[0.6, 0.1], [0.3, 0.4]])
    x, a, r = np.array([0, 0, 1, 1]), np.array([0, 0, 1, 0]), np.array([1.0, 0.8, 0.5, 0.2])
    got = (sol.ips(x, a, r, pi, b), sol.snips(x, a, r, pi, b), sol.doubly_robust(x, a, r, pi, b, q))
    assert np.allclose(got, (0.955, 3.82 / 5.2, 0.725)), got
    w = pi[x, a] / b[x, a]
    assert np.isclose(sol.effective_sample_size(w), 5.2**2 / (1.6**2 + 1.6**2 + 1.8**2 + 0.2**2))
    S, A, R = np.array([[0, 1]]), np.array([[0, 1]]), np.array([[1.0, 2.0]])
    assert np.isclose(sol.per_decision_is(S, A, R, pi, b, 0.9)[0], 6.784)
    qq = np.array([[[0.5, 0.2], [0.3, 1.0]], [[0.4, 0.1], [0.5, 1.8]]])
    assert np.isclose(sol.doubly_robust_trajectory(S, A, R, pi, b, 0.9, qq)[0], 4.1632)
    return got, sol.effective_sample_size(w)


def main() -> None:
    got, ess = worked_examples()
    print("hand log: ips %.4f snips %.4f dr %.4f ess %.4f" % (*got, ess))
    pts = ess_by_horizon()
    print("ESS / N:", ", ".join(f"H={h}: {f:.3f}" for h, f in pts if h in (1, 2, 5, 10, 20)))
    truth, stats = bandit_spread()
    print(f"truth {truth:.4f}", {k: tuple(round(v, 4) for v in s) for k, s in stats.items()})
    out = Path(__file__).with_suffix(".svg.txt")
    out.write_text(svg_ess(pts) + "\n\n" + svg_spread(truth, stats) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
