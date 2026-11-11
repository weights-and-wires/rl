#!/usr/bin/env python3
"""Numbers and SVG for the AlphaZero self-play lesson.

It prints (1) the worked numbers for the visit-count policy, the root noise, and
the loss, (2) the visit counts that a trained network's search gives on the empty
board, and (3) playing strength against training iterations, averaged over three
seeds. With `--svg` it also prints the figures that the lesson embeds. Everything
comes from the reference solution, so it matches the exercise's checks.

Run from the repo root:

    python3 scripts/figures/057-alphazero-self-play.py [--svg]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "content" / "exercises" / "057-alphazero-self-play"))

import solution as sol  # noqa: E402

SEEDS = (0, 1, 2)
CHECKPOINTS = (0, 1, 2, 4, 8, 12, 20)


def worked_numbers() -> dict:
    N = np.array([12.0, 6.0, 2.0])
    print("N =", N)
    for tau in (1.0, 0.5, 0.25, 0.0):
        print(f"tau = {tau}:", np.round(sol.visit_count_policy(N, tau), 4))
    prior = np.array([0.5, 0.3, 0.2])
    legal = np.ones(3, dtype=bool)
    rng = np.random.default_rng(0)
    noise = rng.dirichlet(0.3 * np.ones(3))
    mixed = sol.add_root_noise(prior, legal, 0.3, 0.25, np.random.default_rng(0))
    print("noise draw (alpha 0.3, seed 0) =", np.round(noise, 4), " mixed =", np.round(mixed, 4))
    f = torch.float64
    loss = sol.az_loss(torch.tensor([[2.0, 0.0, 0.0]], dtype=f), torch.tensor([0.5], dtype=f),
                       torch.tensor([[1.0, 0.0, 0.0]], dtype=f), torch.tensor([1.0], dtype=f), [], 0.0)
    print("loss, logits (2, 0, 0), pi (1, 0, 0), v 0.5, z 1:", round(float(loss), 4))
    return {"N": N}


def minimax_player():
    cache = {}

    def value(b):
        k = b.tobytes()
        if k not in cache:
            tv = sol.terminal_value(b)
            cache[k] = tv if tv is not None else max(-value(sol.play(b, a)) for a in sol.legal_moves(b))
        return cache[k]

    def move(b, rng):
        legal = sol.legal_moves(b)
        vals = np.array([-value(sol.play(b, a)) for a in legal])
        return int(rng.choice(legal[vals == vals.max()]))

    return move


def random_move(b, rng):
    return int(rng.choice(sol.legal_moves(b)))


def match(params, opponent, games, rng, n_sim=100):
    wins = losses = draws = 0
    for g in range(games):
        board, me, turn = np.zeros(9, dtype=np.int8), g % 2, 0
        while sol.terminal_value(board) is None:
            a = sol.az_move(params, board, n_sim) if turn % 2 == me else opponent(board, rng)
            board, turn = sol.play(board, a), turn + 1
        r = sol.terminal_value(board)
        mover_me = turn % 2 == me
        if r == 0:
            draws += 1
        elif (r == -1) == mover_me:
            losses += 1
        else:
            wins += 1
    return wins, losses, draws


def strength_curve() -> list[tuple[int, float, float]]:
    """(iterations, mean draws out of 20 against minimax, mean losses out of 100 against random)."""
    mm = minimax_player()
    rows = []
    for k in CHECKPOINTS:
        draws, lost = [], []
        for seed in SEEDS:
            p = sol.train(n_iters=k, seed=seed) if k > 0 else sol.init_params(seed)
            draws.append(match(p, mm, 20, np.random.default_rng(200 + seed))[2])
            lost.append(match(p, random_move, 100, np.random.default_rng(100 + seed))[1])
        rows.append((k, float(np.mean(draws)), float(np.mean(lost))))
        print(f"iterations {k:2d}: minimax draws {rows[-1][1]:.1f}/20, losses to random {rows[-1][2]:.1f}/100")
    return rows


def opening_counts():
    p = sol.train(seed=0)
    board = np.zeros(9, dtype=np.int8)
    N = sol.search(board, lambda b: sol.net_evaluate(p, b), 100, 6.0)
    print("opening visit counts (100 simulations):", N.astype(int).reshape(3, 3).tolist())
    for tau in (1.0, 0.5, 0.25):
        print(f"tau = {tau}:", np.round(sol.visit_count_policy(N, tau), 3).reshape(3, 3).tolist())
    return N


def game_stats():
    rng = np.random.default_rng(0)
    p = sol.init_params(0)
    lens = [len(sol.self_play_game(lambda b: sol.net_evaluate(p, b), 50, 6.0, 3, 0.5, 0.25, rng)) for _ in range(40)]
    print("mean positions per self-play game:", np.mean(lens))
    return float(np.mean(lens))


def svg_loop(positions: float) -> str:
    per_iter = 12 * positions
    return f"""<svg viewBox="0 0 600 250" role="img" aria-label="The AlphaZero loop: the network guides search, search plays games, games become training examples, and the examples train the network">
<defs><marker id="c050-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<rect x="20" y="30" width="150" height="70" rx="8" class="fill-soft-b stroke-b"/>
<text x="95" y="56" text-anchor="middle" class="fig-bold">network f</text>
<text x="95" y="76" text-anchor="middle" class="fig-small">prior p, value v</text>
<rect x="225" y="30" width="150" height="70" rx="8" class="fill-soft-a stroke-a"/>
<text x="300" y="56" text-anchor="middle" class="fig-bold">search</text>
<text x="300" y="76" text-anchor="middle" class="fig-small">50 simulations a move</text>
<rect x="430" y="30" width="150" height="70" rx="8" class="fill-soft-c stroke-c"/>
<text x="505" y="56" text-anchor="middle" class="fig-bold">self-play games</text>
<text x="505" y="76" text-anchor="middle" class="fig-small">12 games, {per_iter:.0f} positions</text>
<rect x="430" y="150" width="150" height="70" rx="8" class="fill-soft-d stroke-d"/>
<text x="505" y="176" text-anchor="middle" class="fig-bold">replay buffer</text>
<text x="505" y="196" text-anchor="middle" class="fig-small">(board, pi, z) x 8 flips</text>
<rect x="225" y="150" width="150" height="70" rx="8" class="fill-soft-b stroke-b"/>
<text x="300" y="176" text-anchor="middle" class="fig-bold">train</text>
<text x="300" y="196" text-anchor="middle" class="fig-small">900 steps of az_loss</text>
<line x1="170" y1="65" x2="222" y2="65" class="stroke-ink" marker-end="url(#c050-arrow)"/>
<line x1="375" y1="65" x2="427" y2="65" class="stroke-ink" marker-end="url(#c050-arrow)"/>
<line x1="505" y1="100" x2="505" y2="147" class="stroke-ink" marker-end="url(#c050-arrow)"/>
<line x1="430" y1="185" x2="378" y2="185" class="stroke-ink" marker-end="url(#c050-arrow)"/>
<path d="M225 185 L95 185 L95 103" fill="none" class="stroke-ink" marker-end="url(#c050-arrow)"/>
<text x="300" y="20" text-anchor="middle" class="fig-small">guides</text>
<text x="400" y="240" text-anchor="middle" class="fig-small">the new network plays the next round of games</text>
</svg>"""


def svg_counts(N: np.ndarray) -> str:
    taus = [(1.0, "tau = 1"), (0.5, "tau = 0.5"), (0.25, "tau = 0.25")]
    cells = list(range(9))
    x0, w = 90, 440
    out = ['<svg viewBox="0 0 600 330" role="img" aria-label="Visit counts of the opening search, and the policies they give at three temperatures">']
    rows = [("visits N", N / N.sum())] + [(label, sol.visit_count_policy(N, t)) for t, label in taus]
    ymax = max(r[1].max() for r in rows)
    bw = 7
    for r, (label, vals) in enumerate(rows):
        y_base = 70 + r * 78
        out.append(f'<line x1="{x0}" y1="{y_base}" x2="{x0 + w}" y2="{y_base}" class="stroke-muted"/>')
        out.append(f'<text x="{x0 - 8}" y="{y_base - 4}" text-anchor="end" class="fig-small fig-bold">{label}</text>')
        for c in cells:
            cx = x0 + 20 + c * 48
            h = vals[c] / ymax * 52
            cls = "fill-a" if r == 0 else "fill-b"
            out.append(f'<rect x="{cx - bw}" y="{y_base - h:.1f}" width="{2 * bw}" height="{h:.1f}" class="{cls}"/>')
            if r == 0:
                out.append(f'<text x="{cx}" y="{y_base + 14}" text-anchor="middle" class="fig-small">{int(N[c])}</text>')
            elif vals[c] > 0.12:
                out.append(f'<text x="{cx}" y="{y_base - h - 3:.1f}" text-anchor="middle" class="fig-small">{vals[c]:.2f}</text>')
    out.append(f'<text x="{x0 + 20}" y="318" class="fig-small">cell 0</text>')
    out.append(f'<text x="{x0 + 20 + 8 * 48}" y="318" text-anchor="end" class="fig-small">cell 8</text>')
    out.append('<text x="300" y="18" text-anchor="middle" class="fig-small">bar height is the probability pi (top row: visits divided by the total); numbers under the top row are raw counts</text>')
    out.append("</svg>")
    return "\n".join(out)


def svg_strength(rows) -> str:
    x0, x1, y0, y1 = 80, 560, 220, 40
    kmax = max(r[0] for r in rows)

    def X(k):
        return x0 + (x1 - x0) * k / kmax

    def Y(v, top):
        return y0 - (y0 - y1) * v / top

    out = ['<svg viewBox="0 0 600 290" role="img" aria-label="Playing strength against training iterations: draws against minimax rise from the untrained level, and losses to a random player fall">']
    for v in (0, 5, 10, 15, 20):
        out.append(f'<line x1="{x0}" y1="{Y(v, 20):.1f}" x2="{x1}" y2="{Y(v, 20):.1f}" class="stroke-grid"/>')
        out.append(f'<text x="{x0 - 8}" y="{Y(v, 20) + 4:.1f}" text-anchor="end" class="fig-small">{v}</text>')
    for k in (0, 5, 10, 15, 20):
        out.append(f'<text x="{X(k):.1f}" y="{y0 + 18}" text-anchor="middle" class="fig-small">{k}</text>')
    pts_d = " ".join(f"{X(k):.1f},{Y(d, 20):.1f}" for k, d, _ in rows)
    pts_l = " ".join(f"{X(k):.1f},{Y(l, 20):.1f}" for k, _, l in rows)
    out.append(f'<polyline points="{pts_d}" class="stroke-a" fill="none"/>')
    out.append(f'<polyline points="{pts_l}" class="stroke-b dashed" fill="none"/>')
    for k, d, l in rows:
        out.append(f'<circle cx="{X(k):.1f}" cy="{Y(d, 20):.1f}" r="4" class="fill-a"/>')
        out.append(f'<circle cx="{X(k):.1f}" cy="{Y(l, 20):.1f}" r="4" class="fill-b"/>')
    out.append(f'<text x="{x0 + 10}" y="{Y(20, 20) - 6:.1f}" class="fig-small fig-bold">draws against minimax, out of 20 games (solid)</text>')
    out.append(f'<text x="{x0 + 160}" y="{Y(10, 20):.1f}" class="fig-small fig-bold">losses to random, out of 100 games (dashed)</text>')
    out.append(f'<text x="320" y="{y0 + 38}" text-anchor="middle" class="fig-small">training iterations, each 12 self-play games and 900 steps</text>')
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    worked_numbers()
    positions = game_stats()
    N = opening_counts()
    rows = strength_curve()
    if "--svg" in sys.argv:
        print("\n--- loop figure ---\n" + svg_loop(positions))
        print("\n--- counts figure ---\n" + svg_counts(N))
        print("\n--- strength figure ---\n" + svg_strength(rows))


if __name__ == "__main__":
    main()
