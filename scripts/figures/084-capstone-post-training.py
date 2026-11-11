#!/usr/bin/env python3
"""Numbers and SVG for the toy-pipeline figure in the capstone lesson.

The figure plots the toy pipeline of the exercise for seed 0, with the
exercise's solution functions: the SFT loss on demonstrations over 400 steps,
then, over 40 DPO steps, the held-out chosen-over-rejected margin and the exact
KL to the SFT policy. The script repeats the body of `run_toy_pipeline`, records
a point every few steps, and asserts that its end points equal the pipeline's.

Run from the repo root:

    python3 scripts/figures/084-capstone-post-training.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "capstone_solution", ROOT / "content/exercises/084-capstone-post-training/solution.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

SEED, R, ALPHA, BETA = 0, 2, 4.0, 0.5
SFT_STEPS, DPO_STEPS, LR_SFT, LR_DPO = 400, 40, 0.5, 0.02


def history():
    D, V, L = S.D, S.V, S.L
    g = torch.Generator().manual_seed(SEED)
    W = torch.randn(V, V, generator=g, dtype=D) * 0.5
    At = torch.randn(R, V, generator=g, dtype=D)
    Bt = torch.randn(V, R, generator=g, dtype=D)
    teacher = S.table_logits(W, At, Bt, ALPHA, R)
    prompts = torch.randint(0, V, (96,), generator=g)
    demos = S.sample_responses(teacher, prompts, g)
    ones = torch.ones(len(demos), L, dtype=D)
    A, B = S.init_lora(V, V, R, SEED)

    def nll(t):
        return S.sft_loss(S.token_logps(t, demos), ones)

    sft = [(0, float(nll(S.table_logits(W, A, B, ALPHA, R))))]
    for step in range(1, SFT_STEPS + 1):
        A, B = S.train_adapter(nll, W, A, B, ALPHA, R, 1, LR_SFT)
        if step % 20 == 0:
            sft.append((step, float(nll(S.table_logits(W, A, B, ALPHA, R)))))
    sft_table = S.table_logits(W, A, B, ALPHA, R).detach()

    def make_pairs(n):
        p = torch.randint(0, V, (n,), generator=g)
        s1, s2 = S.sample_responses(sft_table, p, g), S.sample_responses(sft_table, p, g)
        r1, r2 = S.toy_reward(s1), S.toy_reward(s2)
        keep = r1 != r2
        s1, s2, r1, r2 = s1[keep], s2[keep], r1[keep], r2[keep]
        win = (r1 > r2)[:, None]
        return torch.where(win, s1, s2), torch.where(win, s2, s1)

    ch, rj = make_pairs(400)
    hc, hr = make_pairs(400)
    ref_c = S.token_logps(sft_table, ch).sum(dim=1)
    ref_r = S.token_logps(sft_table, rj).sum(dim=1)

    def dpo(t):
        return S.dpo_loss(S.token_logps(t, ch).sum(dim=1), S.token_logps(t, rj).sum(dim=1),
                          ref_c, ref_r, BETA)

    def margin(t):
        return float((S.token_logps(t, hc).sum(dim=1) - S.token_logps(t, hr).sum(dim=1)).mean())

    lp_sft = S.response_logps(sft_table)

    def point(step):
        t = S.table_logits(W, A, B, ALPHA, R).detach()
        return (step, margin(t), S.sequence_kl(S.response_logps(t), lp_sft))

    dp = [point(0)]
    for step in range(1, DPO_STEPS + 1):
        A, B = S.train_adapter(dpo, W, A, B, ALPHA, R, 1, LR_DPO)
        if step % 4 == 0:
            dp.append(point(step))
    return sft, dp


def panel(x0, w, pts, ymin, ymax, xmax, cls, title, fmt):
    top, bot = 60.0, 190.0
    px = lambda x: x0 + x / xmax * w
    py = lambda y: bot - (y - ymin) / (ymax - ymin) * (bot - top)
    path = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)
    out = [f'<line x1="{x0}" y1="{bot}" x2="{x0 + w}" y2="{bot}" class="stroke-muted"/>',
           f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{bot}" class="stroke-muted"/>',
           f'<polyline points="{path}" class="{cls}" fill="none" stroke-width="2.5"/>',
           f'<text x="{x0}" y="{top - 22}" class="fig-bold">{title}</text>',
           f'<text x="{x0 - 4}" y="{top + 4}" class="fig-small" text-anchor="end">{fmt % ymax}</text>',
           f'<text x="{x0 - 4}" y="{bot + 4}" class="fig-small" text-anchor="end">{fmt % ymin}</text>',
           f'<text x="{x0 + w}" y="{bot + 18}" class="fig-small" text-anchor="end">{xmax} steps</text>']
    return "\n  ".join(out)


def main():
    sft, dp = history()
    ref = S.run_toy_pipeline(SEED)
    assert abs(sft[-1][1] - ref["sft_nll_after"]) < 1e-9
    assert abs(dp[-1][1] - ref["margin_after"]) < 1e-6 and abs(dp[-1][2] - ref["kl_to_sft"]) < 1e-6
    print("SFT NLL", [(s, round(v, 3)) for s, v in sft[::5]])
    print("DPO margin and KL", [(s, round(m, 3), round(k, 3)) for s, m, k in dp])
    ymax = lambda v: float(f"{v:.1f}")
    parts = [
        panel(60, 130, sft, 0.0, 1.8, SFT_STEPS, "stroke-a", "1. SFT: demo NLL", "%.1f"),
        panel(270, 130, [(s, m) for s, m, _ in dp], min(m for _, m, _ in dp) - 0.1,
              max(m for _, m, _ in dp) + 0.1, DPO_STEPS, "stroke-b", "2. DPO: margin", "%.1f"),
        panel(470, 100, [(s, k) for s, _, k in dp], 0.0, max(k for _, _, k in dp) + 0.05,
              DPO_STEPS, "stroke-d", "KL to SFT (nats)", "%.2f"),
    ]
    print('<svg viewBox="0 0 600 230" role="img" aria-label="Three small line charts from one toy run: SFT demonstration loss falls, then the DPO margin and the KL to the SFT policy both rise">')
    print("  " + "\n  ".join(parts))
    print("</svg>")


if __name__ == "__main__":
    main()
