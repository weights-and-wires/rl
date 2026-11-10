#!/usr/bin/env python3
"""Numbers and SVG for the advantage-weighted policy improvement lesson.

Figure 1 is the improved policy of a three-action state at two temperatures.
Figure 2 is MPO's temperature dual for four sampled values, with the KL of the
reweighted samples underneath. Figure 3 is the value of tabular AWAC on the
offline cliff dataset against its temperature. Every plotted coordinate is
computed below, and the script prints the numbers the lesson quotes.

Run from the repo root:

    python3 scripts/figures/051-advantage-weighted-policy-improvement.py
    python3 scripts/figures/051-advantage-weighted-policy-improvement.py --svg
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "app" / "runner"))
sys.path.insert(0, str(ROOT / "content" / "exercises" / "051-advantage-weighted-policy-improvement"))

import rlenvs  # noqa: E402
import solution as ex  # noqa: E402

PI_OLD = np.array([0.5, 0.3, 0.2])
ADV = np.array([1.0, 0.0, 2.0])
MPO_Q = np.array([0.0, 1.0, 2.0, 3.0])
MPO_EPS = 0.2
BETAS = (3.0, 1.0, 0.3, 0.1)


def kl(p, q):
    return float(np.sum(p * (np.log(p) - np.log(q))))


def policy_numbers():
    rows = {}
    for beta in (1.0, 0.5):
        pi = ex.kl_improved_policy(PI_OLD, ADV, beta)
        z = float(np.sum(PI_OLD * np.exp(ADV / beta)))
        rows[beta] = dict(pi=pi, z=z, adv=float(np.sum(pi * ADV)), kl=kl(pi, PI_OLD),
                          obj=ex.improvement_objective(pi, PI_OLD, ADV, beta))
        assert abs(rows[beta]["obj"] - beta * math.log(z)) < 1e-12
    return rows


def mpo_numbers():
    etas = np.exp(np.linspace(math.log(0.1), math.log(10.0), 60))
    dual = np.array([ex.mpo_dual(MPO_Q, e, MPO_EPS) for e in etas])
    klw = np.array([ex.weight_kl(ex.mpo_weights(MPO_Q, e)) for e in etas])
    eta_star = ex.mpo_temperature(MPO_Q, MPO_EPS)
    w = ex.mpo_weights(MPO_Q, eta_star)
    return etas, dual, klw, eta_star, w


def awac_numbers():
    mdp = rlenvs.cliff_walking(gamma=0.9)
    S, A = mdp.n_states, mdp.n_actions
    start = mdp.names["start"]
    data = ex.make_dataset(mdp, ex.cautious_behavior(mdp), 100, seed=0)
    counts = ex.action_counts(data, S, A)
    allowed = counts > 0
    bc = ex.behavior_cloning(data, S, A)
    bc_value = float(rlenvs.policy_evaluation(mdp, bc)[start])
    # The best value any policy limited to the data's actions can reach.
    has_action = allowed.any(axis=1)
    v = np.zeros(S)
    for _ in range(3000):
        q = np.where(allowed, rlenvs.q_from_v(mdp, v), -np.inf)
        v = np.where(has_action & ~mdp.terminal, q.max(axis=1), 0.0)
    best = float(v[start])
    values = []
    for beta in BETAS:
        pi = ex.awac_tabular(data, S, A, mdp.gamma, beta)
        values.append(float(rlenvs.policy_evaluation(mdp, pi)[start]))
    return len(data["s"]), bc_value, best, values


def print_numbers():
    rows = policy_numbers()
    for beta, r in rows.items():
        print(f"beta={beta}: Z={r['z']:.4f} pi*={np.round(r['pi'], 4)} E[A]={r['adv']:.4f} "
              f"KL={r['kl']:.4f} objective={r['obj']:.4f}")
    print("weights at beta=1:", np.round(PI_OLD * np.exp(ADV), 4), " at beta=0.5:", np.round(PI_OLD * np.exp(2 * ADV), 4))
    print("AWR weights, adv=(-1,0,1,2), beta=0.5, clip=20:", np.round(ex.awr_weights(np.array([-1.0, 0, 1, 2]), 0.5, 20.0), 4))
    etas, dual, klw, eta_star, w = mpo_numbers()
    print(f"MPO q={MPO_Q} eps={MPO_EPS}: eta*={eta_star:.4f} weights={np.round(w, 4)} "
          f"KL={ex.weight_kl(w):.6f} dual={ex.mpo_dual(MPO_Q, eta_star, MPO_EPS):.4f}")
    for eta in (0.5, 1.0, 2.0):
        ww = ex.mpo_weights(MPO_Q, eta)
        print(f"  eta={eta}: weights={np.round(ww, 4)} KL={ex.weight_kl(ww):.4f} dual={ex.mpo_dual(MPO_Q, eta, MPO_EPS):.4f}")
    n, bc, best, values = awac_numbers()
    print(f"cliff: N={n}, BC value {bc:.3f}, best in-data {best:.3f}")
    for beta, v in zip(BETAS, values):
        print(f"  AWAC beta={beta}: {v:.3f}")


# ---------------------------------------------------------------- SVG helpers
def scale(v, lo, hi, a, b):
    return a + (v - lo) / (hi - lo) * (b - a)


def fig1():
    rows = policy_numbers()
    series = [(PI_OLD, "fill-sunk", "pi_old"), (rows[1.0]["pi"], "fill-a", "beta = 1"), (rows[0.5]["pi"], "fill-b", "beta = 0.5")]
    x0, y0, y1, ymax = 70.0, 230.0, 60.0, 0.8
    out = ['<svg viewBox="0 0 600 300" role="img" aria-label="Probabilities of three actions under the old policy and under the improved policy at beta 1 and beta 0.5">']
    for t in (0.0, 0.2, 0.4, 0.6, 0.8):
        y = scale(t, 0, ymax, y0, y1)
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="560" y2="{y:.1f}" class="stroke-grid"/>')
        out.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" class="fig-small" text-anchor="end">{t:.1f}</text>')
    labels = ("action 0, A = 1", "action 1, A = 0", "action 2, A = 2")
    for g in range(3):
        gx = 100 + g * 160
        for k, (vals, cls, _) in enumerate(series):
            x = gx + k * 36
            y = scale(vals[g], 0, ymax, y0, y1)
            out.append(f'<rect x="{x}" y="{y:.1f}" width="30" height="{y0 - y:.1f}" class="{cls}"/>')
            out.append(f'<text x="{x + 15}" y="{y - 6:.1f}" class="fig-small" text-anchor="middle">{vals[g]:.3f}</text>')
        out.append(f'<text x="{gx + 54}" y="256" class="fig-small" text-anchor="middle">{labels[g]}</text>')
    for k, (_, cls, name) in enumerate(series):
        lx = 150 + k * 130
        out.append(f'<rect x="{lx}" y="22" width="12" height="12" class="{cls}"/>')
        out.append(f'<text x="{lx + 18}" y="33" class="fig-small">{name}</text>')
    out.append("</svg>")
    return "\n".join(out)


def fig2():
    etas, dual, klw, eta_star, w = mpo_numbers()
    lx0, lx1 = math.log(0.1), math.log(10.0)
    X0, X1 = 80.0, 540.0
    dlo, dhi = 2.0, 3.6
    dual_clip = np.clip(dual, dlo, dhi)
    out = ['<svg viewBox="0 0 600 400" role="img" aria-label="MPO temperature dual and the KL of the reweighted samples against eta; the dual is lowest where the KL equals epsilon">']

    def px(e):
        return scale(math.log(e), lx0, lx1, X0, X1)

    # Top panel: dual.
    ty0, ty1 = 160.0, 40.0
    for t in (2.2, 2.6, 3.0, 3.4):
        y = scale(t, dlo, dhi, ty0, ty1)
        out.append(f'<line x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}" class="stroke-grid"/>')
        out.append(f'<text x="{X0 - 8}" y="{y + 4:.1f}" class="fig-small" text-anchor="end">{t:.1f}</text>')
    pts = " ".join(f"{px(e):.1f},{scale(d, dlo, dhi, ty0, ty1):.1f}" for e, d in zip(etas, dual_clip))
    out.append(f'<polyline points="{pts}" class="stroke-b" fill="none"/>')
    ds = ex.mpo_dual(MPO_Q, eta_star, MPO_EPS)
    out.append(f'<circle cx="{px(eta_star):.1f}" cy="{scale(ds, dlo, dhi, ty0, ty1):.1f}" r="5" class="fill-b"/>')
    out.append(f'<text x="{px(eta_star) + 10:.1f}" y="{scale(ds, dlo, dhi, ty0, ty1) + 24:.1f}" class="fig-small">lowest at eta = {eta_star:.2f}</text>')
    out.append(f'<text x="{X0}" y="26" class="fig-small fig-bold">dual g(eta)</text>')
    # Link line.
    out.append(f'<line x1="{px(eta_star):.1f}" y1="{ty0}" x2="{px(eta_star):.1f}" y2="340" class="stroke-muted dashed"/>')
    # Bottom panel: KL.
    by0, by1, kmax = 340.0, 215.0, 1.2
    for t in (0.0, 0.4, 0.8, 1.2):
        y = scale(t, 0, kmax, by0, by1)
        out.append(f'<line x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}" class="stroke-grid"/>')
        out.append(f'<text x="{X0 - 8}" y="{y + 4:.1f}" class="fig-small" text-anchor="end">{t:.1f}</text>')
    pts = " ".join(f"{px(e):.1f},{scale(k, 0, kmax, by0, by1):.1f}" for e, k in zip(etas, klw))
    out.append(f'<polyline points="{pts}" class="stroke-a" fill="none"/>')
    ye = scale(MPO_EPS, 0, kmax, by0, by1)
    out.append(f'<line x1="{X0}" y1="{ye:.1f}" x2="{X1}" y2="{ye:.1f}" class="stroke-c dashed"/>')
    out.append(f'<text x="{X1}" y="{ye - 6:.1f}" class="fig-small" text-anchor="end">budget epsilon = {MPO_EPS}</text>')
    out.append(f'<circle cx="{px(eta_star):.1f}" cy="{ye:.1f}" r="5" class="fill-a"/>')
    out.append(f'<text x="{X0}" y="205" class="fig-small fig-bold">KL of the weights from uniform</text>')
    for t in (0.1, 1.0, 10.0):
        out.append(f'<line x1="{px(t):.1f}" y1="{by0}" x2="{px(t):.1f}" y2="{by0 + 5}" class="stroke-muted"/>')
        out.append(f'<text x="{px(t):.1f}" y="{by0 + 20}" class="fig-small" text-anchor="middle">{t:g}</text>')
    out.append(f'<text x="{(X0 + X1) / 2}" y="388" class="fig-small" text-anchor="middle">eta (log scale)</text>')
    out.append("</svg>")
    return "\n".join(out)


def fig3():
    n, bc, best, values = awac_numbers()
    out = ['<svg viewBox="0 0 600 320" role="img" aria-label="Value at the start of the cliff world for behavior cloning and for AWAC at four temperatures, between behavior cloning and the best in-data value">']
    X0, X1, y0, y1 = 80.0, 540.0, 250.0, 50.0
    vlo, vhi = -9.5, -7.0
    for t in (-9.0, -8.5, -8.0, -7.5):
        y = scale(t, vlo, vhi, y0, y1)
        out.append(f'<line x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}" class="stroke-grid"/>')
        out.append(f'<text x="{X0 - 8}" y="{y + 4:.1f}" class="fig-small" text-anchor="end">{t:g}</text>')
    yb = scale(best, vlo, vhi, y0, y1)
    out.append(f'<line x1="{X0}" y1="{yb:.1f}" x2="{X1}" y2="{yb:.1f}" class="stroke-c dashed"/>')
    out.append(f'<text x="{X1}" y="{yb - 6:.1f}" class="fig-small" text-anchor="end">best the data supports: {best:.3f}</text>')
    yc = scale(bc, vlo, vhi, y0, y1)
    out.append(f'<line x1="{X0}" y1="{yc:.1f}" x2="{X1}" y2="{yc:.1f}" class="stroke-muted dashed"/>')
    out.append(f'<text x="{X1}" y="{yc + 16:.1f}" class="fig-small" text-anchor="end">behavior cloning: {bc:.3f}</text>')
    lo, hi = math.log10(0.1), math.log10(3.0)
    pts = []
    for beta, v in zip(BETAS, values):
        x = scale(math.log10(beta), lo, hi, X1 - 40, X0 + 40)
        y = scale(v, vlo, vhi, y0, y1)
        pts.append((x, y, beta, v))
    out.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in pts) + '" class="stroke-b" fill="none"/>')
    for x, y, beta, v in pts:
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" class="fill-b"/>')
        out.append(f'<text x="{x:.1f}" y="{y - 10:.1f}" class="fig-small" text-anchor="middle">{v:.3f}</text>')
        out.append(f'<text x="{x:.1f}" y="{y0 + 20}" class="fig-small" text-anchor="middle">{beta:g}</text>')
    out.append(f'<text x="{(X0 + X1) / 2}" y="{y0 + 44}" class="fig-small" text-anchor="middle">beta (large on the left, small on the right)</text>')
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    if "--svg" in sys.argv:
        for name, f in (("FIG1", fig1), ("FIG2", fig2), ("FIG3", fig3)):
            print(f"=== {name}")
            print(f())
    else:
        print_numbers()
