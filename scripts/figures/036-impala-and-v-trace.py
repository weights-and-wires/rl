#!/usr/bin/env python3
"""Numbers and SVG for the IMPALA and V-trace lesson.

It prints (1) the worked V-trace example, (2) an exact tick-by-tick queue
simulation that compares synchronous A2C with asynchronous actors, and (3) the
fixed point of the V-trace operator on a tabular MDP as rho_bar grows. With
`--svg` it also prints the two figures that the lesson embeds. The numbers come
from the reference solution, so they match the exercise's checks.

Run from the repo root:

    python3 scripts/figures/036-impala-and-v-trace.py [--svg]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "app" / "runner"))
sys.path.insert(0, str(ROOT / "content" / "exercises" / "036-impala-and-v-trace"))

import rlenvs  # noqa: E402
import solution as sol  # noqa: E402

ACT_TICKS = 8  # one actor needs 8 ticks to produce an unroll
LEARN_TICKS = 1  # one learner update on one unroll needs 1 tick
QUEUE = 4  # IMPALA's queue is bounded; a full queue blocks the actors
UPDATES = 400
ACTORS = [1, 2, 4, 8, 16, 32]


def worked_example() -> None:
    rewards = np.array([1.0, 0.0, 1.0])
    values = np.array([0.5, 0.4, 0.3])
    ratios = np.array([0.5, 1.5, 2.0])
    no = np.zeros(3, dtype=bool)
    vs = sol.vtrace_targets(rewards, values, 0.2, no, np.log(ratios), 0.9)
    adv = sol.vtrace_advantages(rewards, values, vs, 0.2, no, np.log(ratios), 0.9)
    rho, c = sol.clipped_ratios(np.log(ratios), 1.0, 1.0)
    print("worked example: rho", rho, "c", c)
    print("  targets", np.round(vs, 4), "advantages", np.round(adv, 4))
    td0 = sol.vtrace_targets(rewards, values, 0.2, no, np.log(ratios), 0.9, 1.0, 0.0)
    print("  c_bar = 0 targets", np.round(td0, 4))
    ones = np.zeros(3)
    on = sol.vtrace_targets(rewards, values, 0.2, no, ones, 0.9)
    print("  on-policy targets", np.round(on, 4), "(n-step: 1 + 0.9*0 + 0.81*1 + 0.729*0.2 =",
          round(1 + 0.81 + 0.729 * 0.2, 4), ")")


def sync_stats(k: int) -> tuple[float, float]:
    """Synchronous A2C: all k actors act, then the learner updates on the batch."""
    round_ticks = ACT_TICKS + k * LEARN_TICKS
    return k * LEARN_TICKS / round_ticks, 0.0


def async_stats(k: int) -> tuple[float, float]:
    """Asynchronous actors with a bounded FIFO queue, simulated one tick at a time.

    An actor fetches the newest parameter version when it starts an unroll. The
    learner takes the oldest unroll when it is idle, and each finished update
    raises the version by one. Staleness is the version when the learner takes
    an unroll minus the version the actor used.
    """
    version = 0
    queue: list[int] = []
    left = [ACT_TICKS] * k  # ticks until each actor finishes, 0 means blocked
    used = [0] * k
    blocked: list[int] = []
    busy_left = 0
    busy_ticks = 0
    done = 0
    stale: list[int] = []
    t = 0
    while done < UPDATES:
        t += 1
        for i in range(k):
            if left[i] > 0:
                left[i] -= 1
                if left[i] == 0:
                    blocked.append(i)
        for i in list(blocked):
            if len(queue) < QUEUE:
                queue.append(used[i])
                blocked.remove(i)
                used[i] = version
                left[i] = ACT_TICKS
        if busy_left == 0 and queue:
            stale.append(version - queue.pop(0))
            busy_left = LEARN_TICKS
        if busy_left > 0:
            busy_ticks += 1
            busy_left -= 1
            if busy_left == 0:
                version += 1
                done += 1
    return busy_ticks / t, float(np.mean(stale))


def lag_table() -> list[tuple[int, float, float, float, float]]:
    rows = []
    for k in ACTORS:
        su, ss = sync_stats(k)
        au, ast = async_stats(k)
        rows.append((k, su, ss, au, ast))
    return rows


def fixed_point_curve():
    mdp = rlenvs.random_mdp(n_states=5, n_actions=3, gamma=0.9, seed=0)
    rng = np.random.default_rng(100)
    pi = rng.dirichlet(np.ones(3), size=5)
    mu = rng.dirichlet(np.ones(3), size=5)
    bars = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0]
    state = 0
    pts = []
    for rb in bars:
        V = np.zeros(5)
        for _ in range(3000):
            V = sol.vtrace_operator(mdp, pi, mu, V, rb, 1.0)
        want = rlenvs.policy_evaluation(mdp, sol.interpolated_policy(pi, mu, rb))
        assert np.allclose(V, want, atol=1e-6), (rb, V, want)
        pts.append((rb, V[state]))
    return pts, rlenvs.policy_evaluation(mdp, mu)[state], rlenvs.policy_evaluation(mdp, pi)[state]


def svg_lag(rows) -> str:
    x0, x1, y0, y1 = 70, 560, 250, 40
    def X(i):
        return x0 + (x1 - x0) * i / (len(rows) - 1)
    def Y(u):
        return y0 + (y1 - y0) * u
    out = ['<svg viewBox="0 0 600 330" role="img" aria-label="Learner utilization against the number of actors for synchronous A2C and asynchronous actors, with the mean staleness of the asynchronous batches">']
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" class="stroke-muted"/>')
    for u in (0, 0.5, 1.0):
        out.append(f'<line x1="{x0}" y1="{Y(u):.1f}" x2="{x1}" y2="{Y(u):.1f}" class="stroke-grid"/><text x="{x0 - 8}" y="{Y(u) + 4:.1f}" class="fig-small" text-anchor="end">{u:g}</text>')
    for i, r in enumerate(rows):
        out.append(f'<text x="{X(i):.1f}" y="{y0 + 20}" class="fig-small" text-anchor="middle">{r[0]}</text>')
    for idx, cls, dash in ((1, "stroke-a", ""), (3, "stroke-b", "")):
        pts = " ".join(f"{X(i):.1f},{Y(r[idx]):.1f}" for i, r in enumerate(rows))
        out.append(f'<polyline points="{pts}" class="{cls}" fill="none"/>')
        fill = "fill-a" if idx == 1 else "fill-b"
        for i, r in enumerate(rows):
            out.append(f'<circle cx="{X(i):.1f}" cy="{Y(r[idx]):.1f}" r="4" class="{fill}"/>')
    for i, r in enumerate(rows):
        out.append(f'<text x="{X(i):.1f}" y="{y0 + 40}" class="fig-small fig-bold" text-anchor="middle">{r[4]:.1f}</text>')
    out.append(f'<text x="{x0 - 8}" y="{y0 + 40}" class="fig-small" text-anchor="end">lag</text>')
    out.append(f'<text x="{(x0 + x1) / 2}" y="{y0 + 64}" text-anchor="middle">number of actors</text>')
    out.append(f'<text x="{x0 - 8}" y="28" class="fig-small" text-anchor="end">learner busy</text>')
    out.append('<line x1="380" y1="62" x2="408" y2="62" class="stroke-b"/><text x="414" y="66" class="fig-small">asynchronous</text>')
    out.append('<line x1="380" y1="82" x2="408" y2="82" class="stroke-a"/><text x="414" y="86" class="fig-small">synchronous A2C</text>')
    out.append('</svg>')
    return "\n".join(out)


def svg_fixed(pts, v_mu, v_pi) -> str:
    x0, x1, y0, y1 = 80, 560, 250, 40
    lo = min(v_mu, min(p[1] for p in pts)) - 0.2
    hi = max(v_pi, max(p[1] for p in pts)) + 0.2
    import math
    lg0, lg1 = math.log10(pts[0][0]), math.log10(pts[-1][0])
    def X(rb):
        return x0 + (x1 - x0) * (math.log10(rb) - lg0) / (lg1 - lg0)
    def Y(v):
        return y0 + (y1 - y0) * (v - lo) / (hi - lo)
    out = ['<svg viewBox="0 0 600 320" role="img" aria-label="Value of the V-trace fixed point at one state as rho-bar grows, moving from the behavior policy value toward the target policy value">']
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" class="stroke-muted"/>')
    out.append(f'<line x1="{x0}" y1="{Y(v_mu):.1f}" x2="{x1}" y2="{Y(v_mu):.1f}" class="stroke-a dashed"/><text x="{x1}" y="{Y(v_mu) - 6:.1f}" class="fig-small" text-anchor="end">V of mu = {v_mu:.2f}</text>')
    out.append(f'<line x1="{x0}" y1="{Y(v_pi):.1f}" x2="{x1}" y2="{Y(v_pi):.1f}" class="stroke-c dashed"/><text x="{x0 + 6}" y="{Y(v_pi) - 6:.1f}" class="fig-small">V of pi = {v_pi:.2f}</text>')
    poly = " ".join(f"{X(rb):.1f},{Y(v):.1f}" for rb, v in pts)
    out.append(f'<polyline points="{poly}" class="stroke-b" fill="none"/>')
    for rb, v in pts:
        out.append(f'<circle cx="{X(rb):.1f}" cy="{Y(v):.1f}" r="4" class="fill-b"/>')
    for rb in (0.1, 0.5, 1.0, 3.0, 10.0, 100.0):
        out.append(f'<line x1="{X(rb):.1f}" y1="{y0}" x2="{X(rb):.1f}" y2="{y0 + 5}" class="stroke-muted"/><text x="{X(rb):.1f}" y="{y0 + 20}" class="fig-small" text-anchor="middle">{rb:g}</text>')
    out.append(f'<text x="{(x0 + x1) / 2}" y="{y0 + 44}" text-anchor="middle">rho-bar (log scale)</text>')
    out.append(f'<text x="{x0 - 8}" y="28" class="fig-small" text-anchor="end">fixed-point value at state 0</text>')
    out.append('</svg>')
    return "\n".join(out)


def main() -> None:
    worked_example()
    rows = lag_table()
    print("actors | sync busy, lag | async busy, mean lag")
    for k, su, ss, au, ast in rows:
        print(f"{k:6d} | {su:.3f}, {ss:.1f} | {au:.3f}, {ast:.2f}")
    pts, v_mu, v_pi = fixed_point_curve()
    print("fixed point at state 0 (c_bar = 1):")
    for rb, v in pts:
        print(f"  rho_bar {rb:6.2f}: {v:.4f}")
    print(f"  V of mu {v_mu:.4f}, V of pi {v_pi:.4f}")
    if "--svg" in sys.argv:
        print(svg_lag(rows))
        print(svg_fixed(pts, v_mu, v_pi))


if __name__ == "__main__":
    main()
