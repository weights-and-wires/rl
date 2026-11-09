#!/usr/bin/env python3
"""Numbers for the TD3+BC actor loss in the offline RL lesson.

The batch is the one in "Where this breaks". Fujimoto and Gu (2021) minimize
the mean squared action gap minus λ times the mean of Q(s, π(s)), with

    λ = α / mean(|Q(s, a)|)

held constant. α = 2.5. Q at the policy actions is 4π on this batch.

Run from the repo root:

    python3 scripts/figures/045-offline-rl.py
"""

from __future__ import annotations

import numpy as np

# Logged actions and their Q values. Mean |Q| is 20, so λ = 2.5 / 20.
LOGGED = np.array([0.0, 0.2, -0.4])
Q_DATA = np.array([10.0, -20.0, 30.0])
PI = np.array([0.5, 0.0, 0.1])
ALPHA = 2.5


def actor_loss(pi, logged, q_pi, q_data, alpha):
    lam = float(alpha) / float(np.mean(np.abs(q_data)))
    return float(np.mean((pi - logged) ** 2) - lam * np.mean(q_pi))


def action_gradient(pi, logged, q_data, alpha, slope):
    """Derivative of the loss when Q(s, π) = slope * π, with λ fixed."""
    lam = float(alpha) / float(np.mean(np.abs(q_data)))
    n = pi.size
    return 2.0 * (pi - logged) / n - lam * slope / n


def main() -> None:
    q_pi = 4.0 * PI
    lam = ALPHA / float(np.mean(np.abs(Q_DATA)))
    gap = float(np.mean((PI - LOGGED) ** 2))
    loss = actor_loss(PI, LOGGED, q_pi, Q_DATA, ALPHA)
    loss_alpha_1 = actor_loss(PI, LOGGED, q_pi, Q_DATA, 1.0)
    only_bc = gap
    no_division = gap - ALPHA * float(np.mean(q_pi))
    grad = action_gradient(PI, LOGGED, Q_DATA, ALPHA, slope=4.0)

    print(f"lambda = {lam:.6f}")
    print(f"mean squared gap = {gap:.6f}")
    print(f"mean Q(s, pi) = {float(np.mean(q_pi)):.6f}")
    print(f"loss at alpha 2.5 = {loss:.6f}")
    print(f"loss at alpha 1 = {loss_alpha_1:.6f}")
    print(f"squared gap only = {only_bc:.6f}")
    print(f"alpha without the division = {no_division:.6f}")
    print(f"gradient = {np.array2string(grad, precision=6)}")

    assert abs(lam - 0.125) < 1e-12
    assert abs(gap - 0.18) < 1e-12
    assert abs(loss - 0.08) < 1e-12
    assert abs(loss_alpha_1 - 0.14) < 1e-12
    assert abs(only_bc - 0.18) < 1e-12
    assert abs(no_division - (-1.82)) < 1e-12
    assert np.allclose(grad, [1.0 / 6.0, -0.3, 1.0 / 6.0], atol=1e-12)


if __name__ == "__main__":
    main()
