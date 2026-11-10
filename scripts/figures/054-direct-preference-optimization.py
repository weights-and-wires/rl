#!/usr/bin/env python3
"""Numbers and SVG for the SimPO and KTO figure in the DPO lesson.

The figure plots three losses against one scaled-reward axis u:

- DPO: -log sigmoid(u), where u = beta * h is the scaled margin.
- SimPO: -log sigmoid(u - gamma), where u is the scaled gap of average
  per-token log-probabilities and gamma = 0.5 is the target margin.
- KTO, desirable example: 1 - sigmoid(u), where u = beta * (r - z_ref).

The script also reproduces the lesson's worked examples: the SimPO pair with
a sum of -20 over 10 tokens against -36 over 12 tokens, and the KTO pair with
log-ratios +1 and -0.5. Every plotted coordinate is computed below.

Run from the repo root:

    python3 scripts/figures/054-direct-preference-optimization.py
"""

from __future__ import annotations

import math

import numpy as np

GAMMA = 0.5
U_MIN, U_MAX = -3.0, 4.0
LOSS_MAX = 3.5


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def softplus(z):
    return np.logaddexp(0.0, np.asarray(z, dtype=float))


def dpo(u):
    return softplus(-u)


def simpo(u, gamma=GAMMA):
    return softplus(-(u - gamma))


def kto_desirable(u):
    return 1.0 - sigmoid(u)


def simpo_example(beta=2.0, gamma=0.5):
    """Sum -20 over 10 tokens against -36 over 12 tokens."""
    gap = beta * (-20.0 / 10.0) - beta * (-36.0 / 12.0)
    arg = gap - gamma
    summed = beta * -20.0 - beta * -36.0 - gamma
    return gap, arg, float(softplus(-arg)), float(softplus(-summed))


def kto_example(beta=1.0, lambda_d=1.0, lambda_u=1.5):
    """Log-ratios +1 (desirable) and -0.5 (undesirable); mismatched ratios 0.5 and -0.1."""
    z_ref = max(0.0, (0.5 + -0.1) / 2.0)
    cost_d = lambda_d * (1.0 - sigmoid(beta * (1.0 - z_ref)))
    cost_u = lambda_u * (1.0 - sigmoid(beta * (z_ref - -0.5)))
    return z_ref, float(cost_d), float(cost_u), float((cost_d + cost_u) / 2.0)


def _x(u):
    return 80.0 + (u - U_MIN) / (U_MAX - U_MIN) * 480.0


def _y(loss):
    return 250.0 - loss / LOSS_MAX * 214.0


def figure_losses():
    us = np.linspace(U_MIN, U_MAX, 71)

    def pts(f):
        return " ".join(f"{_x(u):.1f},{_y(f(u)):.1f}" for u in us)

    grid = []
    for tick in (0, 1, 2, 3):
        y = _y(tick)
        grid.append(
            f'<line x1="80" y1="{y:.1f}" x2="560" y2="{y:.1f}" class="stroke-grid"/>'
            f'<text x="72" y="{y + 4:.1f}" class="fig-small" text-anchor="end">{tick}</text>'
        )
    xticks = []
    for tick in (-2, 0, 2, 4):
        x = _x(tick)
        xticks.append(
            f'<line x1="{x:.1f}" y1="250" x2="{x:.1f}" y2="256" class="stroke-muted"/>'
            f'<text x="{x:.1f}" y="272" class="fig-small" text-anchor="middle">{tick}</text>'
        )
    return f'''<svg viewBox="0 0 600 320" role="img" aria-label="DPO, SimPO, and KTO losses against the scaled reward u. At u = 0 DPO is 0.693, SimPO with gamma 0.5 is 0.974, and KTO is 0.5. KTO stays below 1 for every u.">
<line x1="80" y1="250" x2="560" y2="250" class="stroke-muted"/>
{"".join(grid)}
{"".join(xticks)}
<polyline points="{pts(simpo)}" class="stroke-c" fill="none"/>
<polyline points="{pts(dpo)}" class="stroke-b" fill="none"/>
<polyline points="{pts(kto_desirable)}" class="stroke-a dashed" fill="none"/>
<circle cx="{_x(0):.1f}" cy="{_y(float(dpo(0.0))):.1f}" r="4.5" class="fill-b"/>
<circle cx="{_x(0):.1f}" cy="{_y(float(simpo(0.0))):.1f}" r="4.5" class="fill-c"/>
<circle cx="{_x(0):.1f}" cy="{_y(float(kto_desirable(0.0))):.1f}" r="4.5" class="fill-a"/>
<line x1="380" y1="60" x2="412" y2="60" class="stroke-c"/>
<text x="418" y="64" class="fig-small">SimPO, γ = 0.5</text>
<line x1="380" y1="80" x2="412" y2="80" class="stroke-b"/>
<text x="418" y="84" class="fig-small">DPO</text>
<line x1="380" y1="100" x2="412" y2="100" class="stroke-a dashed"/>
<text x="418" y="104" class="fig-small">KTO, desirable</text>
<text x="320" y="300" text-anchor="middle">scaled reward u</text>
<text x="78" y="24" class="fig-small" text-anchor="end">loss</text>
</svg>'''


def main():
    for u in (0.0, 3.0):
        print(f"u={u:g}: DPO {float(dpo(u)):.4f}, SimPO {float(simpo(u)):.4f}, KTO {float(kto_desirable(u)):.4f}")
    print(f"SimPO at u=0 exceeds DPO by {float(simpo(0.0) - dpo(0.0)):.4f}; at u=-3 KTO is {float(kto_desirable(-3.0)):.4f}, DPO {float(dpo(-3.0)):.4f}")
    gap, arg, loss, summed = simpo_example()
    print(f"SimPO example: gap {gap:.1f}, argument {arg:.1f}, loss {loss:.4f}; summed log-probs would give {summed:.2e}")
    z_ref, cost_d, cost_u, loss_k = kto_example()
    print(f"KTO example: z_ref {z_ref:.1f}, desirable {cost_d:.4f}, undesirable {cost_u:.4f}, loss {loss_k:.4f}")
    print("---FIGURE---")
    print(figure_losses())


if __name__ == "__main__":
    main()
