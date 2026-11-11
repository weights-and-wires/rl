#!/usr/bin/env python3
"""Figure 1 of the credit-assignment lesson, and the numbers quoted in its text.

One sampled response on the four-step toy, with outcomes [1, 1, 0, 1]: step 3 is
wrong. The figure compares, step by step, the exact advantage, the uniform
sequence-level advantage R - V(root), and the VinePPO estimate with K = 64
rollouts per boundary. The script prints the SVG block that the lesson embeds,
then the quoted numbers.

Run from the repo root:

    python3 scripts/figures/074-credit-assignment-for-llms.py
"""

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "sol", ROOT / "content/exercises/074-credit-assignment-for-llms/solution.py")
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)


def build():
    outcomes = np.array([1, 1, 0, 1])
    values = np.array([sol.exact_value(outcomes[:k]) for k in range(5)])
    exact = np.diff(values)
    uniform = np.full(4, float(outcomes.all()) - values[0])
    vine, v_hat = sol.vine_advantages(outcomes, 64, np.random.default_rng(0))
    return values, exact, uniform, vine, v_hat


def svg(exact, uniform, vine):
    y0, scale = 80.0, 220.0
    parts = [
        '<svg viewBox="0 0 600 250" role="img" aria-label="Per-step advantages for a four-step '
        'response whose third step is wrong: exact, uniform, and Monte Carlo estimates">',
        f'  <line x1="50" y1="{y0:.0f}" x2="580" y2="{y0:.0f}" class="stroke-muted"/>',
        '  <text x="8" y="84" class="fig-small">0</text>',
    ]
    series = [(exact, "fill-c"), (uniform, "fill-muted"), (vine, "fill-b")]
    for step in range(4):
        gx = 70 + step * 130
        for j, (vals, cls) in enumerate(series):
            v = vals[step]
            h = abs(v) * scale
            y = y0 - h if v >= 0 else y0
            parts.append(f'  <rect x="{gx + j * 30}" y="{y:.1f}" width="26" height="{max(h, 0.8):.1f}" class="{cls}"/>')
        parts.append(f'  <text x="{gx + 8}" y="226" class="fig-small">step {step + 1}'
                     f'{" (wrong)" if step == 2 else ""}</text>')
    lx = 330
    for i, (name, cls) in enumerate([("exact", "fill-c"), ("uniform R - V(root)", "fill-muted"),
                                     ("vine, K = 64", "fill-b")]):
        parts.append(f'  <rect x="{lx}" y="{14 + i * 18}" width="12" height="12" class="{cls}"/>')
        parts.append(f'  <text x="{lx + 18}" y="{25 + i * 18}" class="fig-small">{name}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    values, exact, uniform, vine, v_hat = build()
    print(svg(exact, uniform, vine))
    print("\nexact prefix values:", np.round(values, 4))
    print("exact advantages:   ", np.round(exact, 4))
    print("uniform advantage:  ", np.round(uniform, 4))
    print("vine K=64 v_hat:    ", np.round(v_hat, 4))
    print("vine K=64 advantages:", np.round(vine, 4), "sum", round(vine.sum(), 4))
    u, v = sol.credit_errors(sol.STEP_P, 16, 2000, 0)
    print("mse uniform:", np.round(u, 3), "mean", round(u.mean(), 3))
    print("mse vine K=16:", np.round(v, 3), "mean", round(v.mean(), 3))
    for K in (4, 16, 64, 256):
        _, vk = sol.credit_errors(sol.STEP_P, K, 1000, 0)
        print(f"K={K}: mean vine mse {vk.mean():.4f}")
