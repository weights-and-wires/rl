#!/usr/bin/env python3
"""Numbers and SVG for the bimodal-teacher figure in the on-policy distillation lesson.

A teacher puts half its probability on each of two modes (at -3 and +3, standard
deviation 0.7) over a 1-D grid of 73 points from -9 to 9. The student is a
discretized Gaussian: one mode. The script searches mu in linspace(-5, 5, 101) and
sigma in linspace(0.3, 6, 58) for the student that minimizes the forward KL and
for the one that minimizes the reverse KL, then prints the KL values the lesson
quotes and the SVG for the figure. Every plotted coordinate is computed here.

It also prints the lesson's one-token example: teacher [0.9, 0.1] against
student [0.6, 0.4].

Run from the repo root:

    python3 scripts/figures/075-on-policy-distillation.py
"""

from __future__ import annotations

import numpy as np

GRID = np.linspace(-9, 9, 73)


def log_bump(mu, sigma):
    """Normalized log-probabilities of a discretized Gaussian, via log-softmax."""
    z = -((GRID - mu) ** 2) / (2 * sigma**2)
    return z - np.logaddexp.reduce(z)


def bump(mu, sigma):
    return np.exp(log_bump(mu, sigma))


def kl(p, q):
    """KL(p || q) from probabilities p and q, safe when q underflows to 0 in the tails."""
    return kl_log(np.log(p), np.log(q))


def kl_log(logp, logq):
    p = np.exp(logp)
    return float(np.sum(p * (logp - logq)))


def fit(p_teacher, direction):
    best, best_loss = None, np.inf
    for mu in np.linspace(-5, 5, 101):
        for sigma in np.linspace(0.3, 6, 58):
            lq = log_bump(mu, sigma)
            lt = np.log(p_teacher)
            loss = kl_log(lt, lq) if direction == "forward" else kl_log(lq, lt)
            if loss < best_loss:
                best, best_loss = (float(mu), float(sigma)), loss
    return best


def main():
    teacher = 0.5 * bump(-3, 0.7) + 0.5 * bump(3, 0.7)
    mu_f, sig_f = fit(teacher, "forward")
    mu_r, sig_r = fit(teacher, "reverse")
    lq_f, lq_r = log_bump(mu_f, sig_f), log_bump(mu_r, sig_r)
    q_f, q_r = np.exp(lq_f), np.exp(lq_r)
    lt = np.log(teacher)
    print(f"forward-KL student: mu={mu_f:.1f} sigma={sig_f:.1f}")
    print(f"reverse-KL student: mu={mu_r:.1f} sigma={sig_r:.1f}")
    print(f"forward student: KL(T||S)={kl_log(lt, lq_f):.3f}  KL(S||T)={kl_log(lq_f, lt):.3f}")
    print(f"reverse student: KL(T||S)={kl_log(lt, lq_r):.3f}  KL(S||T)={kl_log(lq_r, lt):.3f}")
    pt, ps = np.array([0.9, 0.1]), np.array([0.6, 0.4])
    print(f"one token: forward={kl(pt, ps):.4f} reverse={kl(ps, pt):.4f}")

    x0, x1, y0, y1 = 60.0, 580.0, 230.0, 30.0
    ymax = max(teacher.max(), q_f.max(), q_r.max()) * 1.1
    px = lambda g: x0 + (g + 9) / 18 * (x1 - x0)
    py = lambda v: y0 + v / ymax * (y1 - y0)
    out = [f'<svg viewBox="0 0 600 290" role="img" aria-label="A bimodal teacher with two modes at -3 and 3, the reverse-KL student sitting on one mode, and the forward-KL student spread across both">']
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" class="stroke-muted"/>')
    for g in (-9, -6, -3, 0, 3, 6, 9):
        out.append(f'<line x1="{px(g):.1f}" y1="{y0}" x2="{px(g):.1f}" y2="{y0 + 5}" class="stroke-muted"/>')
        out.append(f'<text x="{px(g):.1f}" y="{y0 + 20}" class="fig-small" text-anchor="middle">{g}</text>')
    bw = (x1 - x0) / 72 * 0.8
    for g, v in zip(GRID, teacher):
        out.append(f'<rect x="{px(g) - bw / 2:.1f}" y="{py(v):.1f}" width="{bw:.1f}" height="{y0 - py(v):.1f}" class="fill-soft-a"/>')
    pts = lambda q: " ".join(f"{px(g):.1f},{py(v):.1f}" for g, v in zip(GRID, q))
    out.append(f'<polyline points="{pts(q_r)}" class="stroke-b" fill="none"/>')
    out.append(f'<polyline points="{pts(q_f)}" class="stroke-d dashed" fill="none"/>')
    out.append(f'<text x="{px(0):.1f}" y="{y0 + 44}" text-anchor="middle">position on the grid</text>')
    out.append(f'<rect x="64" y="8" width="14" height="10" class="fill-soft-a"/><text x="84" y="17" class="fig-small">teacher</text>')
    out.append(f'<line x1="164" y1="13" x2="190" y2="13" class="stroke-b"/><text x="196" y="17" class="fig-small">reverse-KL student</text>')
    out.append(f'<line x1="344" y1="13" x2="370" y2="13" class="stroke-d dashed"/><text x="376" y="17" class="fig-small">forward-KL student</text>')
    out.append("</svg>")
    print("\n".join(out))


if __name__ == "__main__":
    main()
