#!/usr/bin/env python3
"""Numbers and SVG for the PPO-in-practice lesson.

Figure 1 is the log-probability of a diagonal Gaussian and the same curve
with the constant term removed. Figure 2 is one minibatch of advantages
normalized two ways. Every plotted coordinate is computed below.

Run from the repo root:

    python3 scripts/figures/035-ppo-in-practice.py
"""

from __future__ import annotations

import math

import numpy as np

# Figure 1. One-dimensional Gaussian used as the picture.
FIG_MEAN = 0.5
FIG_SIGMA = 0.5
FIG_ACTION = 1.5  # marked point, z = 2

# Lesson table. Two action dimensions.
MEAN = np.array([0.0, 1.0])
SIGMA = np.array([1.0, 0.5])
ACTION = np.array([0.0, 0.0])
OTHER_MEAN = np.array([0.5, 1.0])
OTHER_SIGMA = np.array([1.0, 1.0])

# Running moments. Two chunks of a four-number batch.
CHUNK_A = np.array([1.0, 3.0])
CHUNK_B = np.array([2.0, 6.0])

# Advantages versus returns. The return std is exactly three times the
# advantage std, so the wrong normalization is one third as large.
ADVANTAGES = np.array([1.0, 3.0, 5.0])
RETURNS = np.array([0.0, 6.0, 12.0])

# Reward stream. gamma = 0.5, epsilon = 1e-8, no episode boundary.
REWARDS = np.array([2.0, 0.0, 4.0])
GAMMA = 0.5
EPSILON = 1e-8

# Learning-rate schedule and the value-clip rows.
INITIAL_LR = 1e-3
TOTAL_UPDATES = 4
THETA = 0.4
GRAD = 2.0
VALUE_ROWS = (
    # value, old value, target, epsilon
    (1.5, 1.0, 2.0, 0.2),
    (1.1, 1.0, 2.0, 0.2),
)


def log_prob_1d(action, mean, sigma):
    """Scalar Gaussian log-density, including -0.5 log(2π)."""
    z = (action - mean) / sigma
    return -0.5 * math.log(2.0 * math.pi) - math.log(sigma) - 0.5 * z * z


def diagonal_stats(mean, sigma, action, other_mean, other_sigma):
    """Per-dimension log-prob and entropy of `(mean, sigma)`, plus KL(first || second).

    `other_*` is the second Gaussian. The lesson's KL(old || current) uses the
    opposite argument order, and the figures do not plot this KL.
    """
    log_std = np.log(sigma)
    other_log_std = np.log(other_sigma)
    var = sigma**2
    other_var = other_sigma**2
    logp = -0.5 * np.log(2.0 * np.pi) - log_std - 0.5 * (action - mean) ** 2 / var
    entropy = 0.5 * (np.log(2.0 * np.pi) + 1.0) + log_std
    # KL(first || second): first is (mean, sigma), second is other_*.
    kl = (
        other_log_std
        - log_std
        + (var + (mean - other_mean) ** 2) / (2.0 * other_var)
        - 0.5
    )
    grad_mean = (action - mean) / var
    return logp, entropy, kl, grad_mean


def chan_merge(a, b):
    """Population mean and variance of the concatenation, via Chan's merge."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n_a, n_b = a.size, b.size
    mean_a, mean_b = a.mean(), b.mean()
    m2_a = ((a - mean_a) ** 2).sum()
    m2_b = ((b - mean_b) ** 2).sum()
    delta = mean_b - mean_a
    n = n_a + n_b
    mean = mean_a + delta * n_b / n
    m2 = m2_a + m2_b + delta**2 * n_a * n_b / n
    return mean, m2 / n


def scale_rewards(rewards, gamma, epsilon):
    """Online reward scaling. Variance is the population variance."""
    ret = 0.0
    seen = []
    scaled = []
    for reward in rewards:
        ret = gamma * ret + float(reward)
        seen.append(ret)
        var = float(np.var(seen))
        scaled.append(float(reward) / math.sqrt(var + epsilon))
    return np.array(seen), np.array(scaled)


def clipped_value(value, old, target, eps):
    clipped = old + np.clip(value - old, -eps, eps)
    unclipped_sq = (value - target) ** 2
    clipped_sq = (clipped - target) ** 2
    return float(clipped), float(unclipped_sq), float(clipped_sq), float(max(unclipped_sq, clipped_sq))


def _x_map(action):
    # Actions from -0.25 to 1.75 land on x = 80 to 560.
    return 80.0 + (action - (-0.25)) / 2.0 * 480.0


def _y_map(logp):
    # Log-probabilities from -3.6 to 1.0 land on y = 250 to 36.
    return 250.0 - (logp - (-3.6)) / 4.6 * 214.0


def figure_gaussian():
    actions = np.linspace(-0.25, 1.75, 61)
    full = np.array([log_prob_1d(a, FIG_MEAN, FIG_SIGMA) for a in actions])
    # Dropping -0.5 log(2π) shifts the curve up by that constant.
    missing = full + 0.5 * math.log(2.0 * math.pi)
    full_pts = " ".join(f"{_x_map(a):.1f},{_y_map(y):.1f}" for a, y in zip(actions, full))
    miss_pts = " ".join(f"{_x_map(a):.1f},{_y_map(y):.1f}" for a, y in zip(actions, missing))
    mark = FIG_ACTION
    y_full = log_prob_1d(mark, FIG_MEAN, FIG_SIGMA)
    y_miss = y_full + 0.5 * math.log(2.0 * math.pi)
    grid = []
    for tick in (-3, -2, -1, 0, 1):
        y = _y_map(tick)
        grid.append(
            f'<line x1="80" y1="{y:.1f}" x2="560" y2="{y:.1f}" class="stroke-grid"/>'
            f'<text x="72" y="{y + 4:.1f}" class="fig-small" text-anchor="end">{tick}</text>'
        )
    xticks = []
    for tick in (0, 0.5, 1, 1.5):
        x = _x_map(tick)
        xticks.append(
            f'<line x1="{x:.1f}" y1="250" x2="{x:.1f}" y2="256" class="stroke-muted"/>'
            f'<text x="{x:.1f}" y="272" class="fig-small" text-anchor="middle">{tick:g}</text>'
        )
    return f'''<svg viewBox="0 0 600 320" role="img" aria-label="Log-probability of a Gaussian with mean 0.5 and standard deviation 0.5. Removing the constant lifts the curve by 0.919 at every action.">
<line x1="80" y1="250" x2="560" y2="250" class="stroke-muted"/>
{"".join(grid)}
{"".join(xticks)}
<polyline points="{miss_pts}" class="stroke-a dashed" fill="none"/>
<polyline points="{full_pts}" class="stroke-b" fill="none"/>
<line x1="{_x_map(mark):.1f}" y1="{_y_map(y_miss):.1f}" x2="{_x_map(mark):.1f}" y2="{_y_map(y_full):.1f}" class="stroke-ink"/>
<circle cx="{_x_map(mark):.1f}" cy="{_y_map(y_miss):.1f}" r="4.5" class="fill-a"/>
<circle cx="{_x_map(mark):.1f}" cy="{_y_map(y_full):.1f}" r="4.5" class="fill-b"/>
<line x1="96" y1="200" x2="128" y2="200" class="stroke-b"/>
<text x="134" y="204" class="fig-small">log-probability</text>
<line x1="96" y1="220" x2="128" y2="220" class="stroke-a dashed"/>
<text x="134" y="224" class="fig-small">constant removed</text>
<text x="320" y="300" text-anchor="middle">action a</text>
<text x="78" y="24" class="fig-small" text-anchor="end">log π(a)</text>
</svg>'''


def figure_advantages():
    correct = (ADVANTAGES - ADVANTAGES.mean()) / ADVANTAGES.std()
    wrong = (ADVANTAGES - ADVANTAGES.mean()) / RETURNS.std()
    # y: value -1.5 to 1.5 maps to pixel 230 to 40. Zero is at 135.
    def y_of(value):
        return 230.0 - (value - (-1.5)) / 3.0 * 190.0

    zero = y_of(0.0)
    parts = [
        '<svg viewBox="0 0 600 300" role="img" aria-label="Three advantages after two normalizations. Dividing by the advantage standard deviation reaches plus or minus 1.225. Dividing by the return standard deviation reaches plus or minus 0.408.">',
        f'<line x1="70" y1="{zero:.1f}" x2="560" y2="{zero:.1f}" class="stroke-muted"/>',
    ]
    for tick in (-1, 0, 1):
        y = y_of(tick)
        parts.append(
            f'<text x="60" y="{y + 4:.1f}" class="fig-small" text-anchor="end">{tick}</text>'
        )
    labels = ("sample 1", "sample 2", "sample 3")
    for i, (c, w) in enumerate(zip(correct, wrong)):
        cx = 150 + i * 160

        def bar(value, x, css):
            if abs(value) < 1e-12:
                return ""
            y0, y1 = y_of(0.0), y_of(value)
            top, height = (y1, y0 - y1) if value >= 0 else (y0, y1 - y0)
            return (
                f'<rect x="{x:.1f}" y="{top:.1f}" width="36" height="{height:.1f}" class="{css}"/>'
            )

        def label(value, x):
            if abs(value) < 1e-12:
                return ""
            y = y_of(value) - 8 if value > 0 else y_of(value) + 14
            return (
                f'<text x="{x:.1f}" y="{y:.1f}" class="fig-small" text-anchor="middle">{value:.3f}</text>'
            )

        parts.append(bar(w, cx - 42, "fill-a"))
        parts.append(bar(c, cx + 6, "fill-b"))
        parts.append(
            f'<text x="{cx:.1f}" y="262" class="fig-small" text-anchor="middle">{labels[i]}</text>'
        )
        parts.append(label(w, cx - 24))
        parts.append(label(c, cx + 24))
        if abs(c) < 1e-12 and abs(w) < 1e-12:
            parts.append(
                f'<text x="{cx:.1f}" y="{y_of(0) - 8:.1f}" class="fig-small" text-anchor="middle">0</text>'
            )
    parts.append('<rect x="78" y="28" width="16" height="12" class="fill-a"/>')
    parts.append('<text x="100" y="39" class="fig-small">divided by std of returns</text>')
    parts.append('<rect x="78" y="48" width="16" height="12" class="fill-b"/>')
    parts.append('<text x="100" y="59" class="fig-small">divided by std of advantages</text>')
    parts.append('<text x="315" y="288" text-anchor="middle">one minibatch, mean removed</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main():
    logp, entropy, kl, grad = diagonal_stats(MEAN, SIGMA, ACTION, OTHER_MEAN, OTHER_SIGMA)
    mean_m, var_m = chan_merge(CHUNK_A, CHUNK_B)
    seen, scaled = scale_rewards(REWARDS, GAMMA, EPSILON)
    correct = (ADVANTAGES - ADVANTAGES.mean()) / ADVANTAGES.std()
    wrong = (ADVANTAGES - ADVANTAGES.mean()) / RETURNS.std()
    print("gaussian logp", logp, "sum", logp.sum())
    print("entropy", entropy, "sum", entropy.sum())
    print("kl first||second", kl, "sum", kl.sum())
    print("d logp / d mean", grad)
    print("constant per dim", 0.5 * math.log(2 * math.pi))
    print("merge mean", mean_m, "var", var_m)
    print("returns", seen, "scaled", scaled)
    print("adv std", ADVANTAGES.std(), "ret std", RETURNS.std())
    print("correct", correct)
    print("wrong", wrong)
    print("fig1 at 1.5", log_prob_1d(FIG_ACTION, FIG_MEAN, FIG_SIGMA))
    for update in range(TOTAL_UPDATES + 1):
        lr = INITIAL_LR * (1.0 - update / TOTAL_UPDATES)
        print(f"lr update {update}", lr, "theta", THETA - lr * GRAD)
    for row in VALUE_ROWS:
        print("value", row, clipped_value(*row))
    print("---FIGURE1---")
    print(figure_gaussian())
    print("---FIGURE2---")
    print(figure_advantages())


if __name__ == "__main__":
    main()
