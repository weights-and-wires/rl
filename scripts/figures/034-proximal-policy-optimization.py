#!/usr/bin/env python3
"""Numbers and SVG for the proximal policy optimization lesson.

Figure 2 is the full-batch approx_kl after each of 10 epochs on one seeded
CartPole batch (learning rate 0.05, 4 minibatches per epoch), against the
stopping target 0.015. The same batch also gives the clip fraction per epoch.
Every plotted coordinate is computed below, using the exercise's solution.

Run from the repo root:

    python3 scripts/figures/034-proximal-policy-optimization.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "backend/app/runner"), str(ROOT / "content/exercises/034-proximal-policy-optimization")]

import torch  # noqa: E402

import solution as S  # noqa: E402

EPOCHS = 10
LR = 0.05
TARGET_KL = 0.015
CLIP_EPS = 0.2


def run(target_kl):
    """Per-epoch full-batch (approx_kl, clip_fraction) on the fixed batch."""
    torch.manual_seed(0)
    model = S.ActorCritic()
    batch = S.collect_rollout(S.VecCartPole(8, seed=0), model, 128)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-5)
    torch.manual_seed(1)
    kls, clips = [], []
    for _ in range(EPOCHS):
        kl, _ = S.ppo_epochs(model, optimizer, batch, epochs=1, minibatches=4)
        with torch.no_grad():
            logits, _ = model(batch["obs"])
            logp = torch.log_softmax(logits, -1).gather(-1, batch["actions"][:, None]).squeeze(-1)
            clips.append(float(S.clip_fraction(logp, batch["logp"], CLIP_EPS)))
        kls.append(kl[0])
        if target_kl is not None and kl[0] > target_kl:
            break
    return kls, clips


def figure(kls, stop):
    x0, x1, y0, y1 = 70.0, 560.0, 250.0, 40.0
    ymax = 0.05

    def px(epoch):
        return x0 + (epoch - 1) * (x1 - x0) / (EPOCHS - 1)

    def py(v):
        return y0 - v / ymax * (y0 - y1)

    p = ['<svg viewBox="0 0 600 320" role="img" aria-label="Approximate KL after each of 10 epochs on one batch, against a stopping target of 0.015: the KL first exceeds the target at epoch 3, where the early stop ends the update">']
    for v in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05):
        cls = "stroke-muted" if v == 0 else "stroke-grid"
        p.append(f'<line x1="70" y1="{py(v):.1f}" x2="560" y2="{py(v):.1f}" class="{cls}"/><text x="62" y="{py(v) + 4:.1f}" class="fig-small" text-anchor="end">{v:.2f}</text>')
    for e in range(1, EPOCHS + 1):
        p.append(f'<text x="{px(e):.1f}" y="270" class="fig-small" text-anchor="middle">{e}</text>')
    p.append(f'<line x1="70" y1="{py(TARGET_KL):.1f}" x2="560" y2="{py(TARGET_KL):.1f}" class="stroke-bad dashed"/>')
    p.append(f'<text x="556" y="{py(TARGET_KL) - 6:.1f}" class="fig-small" text-anchor="end">target_kl = {TARGET_KL}</text>')
    pts = " ".join(f"{px(e + 1):.1f},{py(k):.1f}" for e, k in enumerate(kls))
    p.append(f'<polyline points="{pts}" class="stroke-muted dashed" fill="none"/>')
    kept = " ".join(f"{px(e + 1):.1f},{py(k):.1f}" for e, k in enumerate(kls[:stop]))
    p.append(f'<polyline points="{kept}" class="stroke-b" fill="none"/>')
    for e, k in enumerate(kls):
        cls = "fill-b" if e < stop else "fill-muted"
        p.append(f'<circle cx="{px(e + 1):.1f}" cy="{py(k):.1f}" r="4.5" class="{cls}"/>')
    p.append(f'<text x="{px(stop) - 8:.1f}" y="{py(kls[stop - 1]) - 8:.1f}" class="fig-small" text-anchor="end">stop: {kls[stop - 1]:.4f}</text>')
    p.append('<text x="315" y="294" text-anchor="middle">epoch on one batch (learning rate 0.05)</text><text x="70" y="24" class="fig-small">full-batch approx_kl after the epoch</text>')
    p.append("</svg>")
    return "\n".join(p)


def main():
    free_kl, free_clip = run(None)
    stop_kl, stop_clip = run(TARGET_KL)
    stop = len(stop_kl)
    print("approx_kl per epoch, no stop:", [round(k, 4) for k in free_kl])
    print("clip fraction per epoch, no stop:", [round(c, 3) for c in free_clip])
    print(f"stopped at target_kl={TARGET_KL}: epochs run {stop}, kls {[round(k, 4) for k in stop_kl]}")
    print("clip fraction at the stop:", round(stop_clip[-1], 3))
    print("epochs saved:", EPOCHS - stop, "max kl without stop:", round(max(free_kl), 4), "at epoch", free_kl.index(max(free_kl)) + 1)
    print("---FIGURE2---")
    print(figure(free_kl, stop))


if __name__ == "__main__":
    main()
