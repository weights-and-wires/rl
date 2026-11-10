"""Worked measurements for a reward model or a judge."""

import numpy as np


def _sigmoid(z):
    array = np.atleast_1d(np.asarray(z, dtype=float))
    out = np.empty(array.shape, dtype=float)
    positive = array >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    ez = np.exp(array[~positive])
    out[~positive] = ez / (1.0 + ez)
    if np.ndim(z) == 0:
        return float(out[0])
    return out


def pairwise_accuracy(score_a, score_b, label):
    """Fraction of decisive labels that the scores rank correctly.

    Args:
        score_a: scores for answer A, shape `(n,)`.
        score_b: scores for answer B, shape `(n,)`.
        label: `+1` if A is preferred, `-1` if B is preferred, `0` for a
            human tie, shape `(n,)`.

    Returns:
        A Python float. Human ties are excluded. Equal scores on a decisive
        label count as incorrect.
    """
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    label = np.asarray(label, dtype=float)
    decisive = label != 0
    prediction = np.sign(score_a - score_b)
    return float(np.mean(prediction[decisive] == label[decisive]))


def expected_calibration_error(prob, outcome, n_bins):
    """Expected calibration error on equal-width bins.

    Args:
        prob: predicted probabilities in `[0, 1]`, shape `(n,)`.
        outcome: `1` when the event happened and `0` otherwise, shape `(n,)`.
        n_bins: number of equal-width bins on `[0, 1]`. Bin `b` covers
            `[b/n_bins, (b+1)/n_bins)`, and the last bin also includes `1`.

    Returns:
        A Python float. Empty bins contribute nothing. A bin's confidence
        is the mean of `prob` inside it, not the middle of the bin.
    """
    prob = np.asarray(prob, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    n = prob.shape[0]
    error = 0.0
    bins = int(n_bins)
    for b in range(bins):
        low = b / bins
        high = (b + 1) / bins
        if b == bins - 1:
            mask = (prob >= low) & (prob <= high)
        else:
            mask = (prob >= low) & (prob < high)
        count = int(mask.sum())
        if count == 0:
            continue
        frequency = float(outcome[mask].mean())
        confidence = float(prob[mask].mean())
        error += (count / n) * abs(frequency - confidence)
    return float(error)


def human_agreement(score_a, score_b, label):
    """Fraction of rows whose score relation matches the label, including ties.

    Args:
        score_a: scores for answer A, shape `(n,)`.
        score_b: scores for answer B, shape `(n,)`.
        label: `+1`, `-1`, or `0`, shape `(n,)`.

    Returns:
        A Python float. A label of `0` agrees only with equal scores. Any
        other label agrees only with a strict score gap of the same sign.
    """
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    label = np.asarray(label, dtype=float)
    prediction = np.sign(score_a - score_b)
    return float(np.mean(prediction == label))


def swap_consistency(judge, pairs):
    """Fraction of pairs whose preference flips when the order flips.

    Args:
        judge: callable. `judge(left, right)` returns `+1` if `left` wins,
            `-1` if `right` wins, and `0` for a tie.
        pairs: sequence of `(left, right)`.

    Returns:
        A Python float in `[0, 1]`. A pair is consistent when
        `judge(right, left) == -judge(left, right)`.
    """
    consistent = 0
    total = 0
    for left, right in pairs:
        first = judge(left, right)
        second = judge(right, left)
        total += 1
        if second == -first:
            consistent += 1
    return consistent / total


def bt_fit(wins, iters=32):
    """Bradley-Terry strengths from pairwise results, centered at mean zero.

    Args:
        wins: sequence of `(i, j, result)`. `result` is the share credited
            to player `i`, in `[0, 1]`: `1` for a win, `0` for a loss, and
            `0.5` for a tie. Player indices are integers starting at 0.
        iters: minorization-maximization steps, starting from equal weights.

    Returns:
        Strengths, shape `(n,)`, with mean 0. `n` is one more than the
        largest player index.
    """
    records = [(int(i), int(j), float(result)) for i, j, result in wins]
    n = 1 + max(max(i, j) for i, j, _ in records)
    credited = np.zeros((n, n), dtype=float)
    for i, j, result in records:
        credited[i, j] += result
        credited[j, i] += 1.0 - result
    totals = credited.sum(axis=1)
    games = credited + credited.T
    weights = np.ones(n)
    for _ in range(iters):
        updated = np.empty(n)
        for i in range(n):
            denominator = 0.0
            for j in range(n):
                if i == j:
                    continue
                denominator += games[i, j] / (weights[i] + weights[j])
            updated[i] = totals[i] / denominator
        weights = updated
    strength = np.log(weights)
    strength -= strength.mean()
    return strength


def length_controlled_winrate(length_diff, win):
    """Win rate at equal length, from a two-coefficient logistic regression.

    Args:
        length_diff: length of the candidate minus length of the baseline,
            shape `(n,)`.
        win: `1` if the candidate wins, `0` if it loses, or a probability
            in between, shape `(n,)`.

    Returns:
        A Python float, the sigmoid of the intercept. Fit
        `b0 + b1 * length_diff` by Newton for 20 steps from zeros. Ridge
        `b1` by `1e-6` and do not ridge `b0`. Clip the weight `p * (1 - p)`
        at `1e-12`.
    """
    gap = np.asarray(length_diff, dtype=float).reshape(-1)
    target = np.asarray(win, dtype=float).reshape(-1)
    intercept = 0.0
    slope = 0.0
    ridge = 1e-6
    for _ in range(20):
        linear = intercept + slope * gap
        probability = _sigmoid(linear)
        weight = np.maximum(probability * (1.0 - probability), 1e-12)
        grad_intercept = np.sum(target - probability)
        grad_slope = np.sum((target - probability) * gap) - ridge * slope
        hess_intercept = -np.sum(weight)
        hess_cross = -np.sum(weight * gap)
        hess_slope = -np.sum(weight * gap * gap) - ridge
        determinant = hess_intercept * hess_slope - hess_cross * hess_cross
        intercept += (-hess_slope * grad_intercept + hess_cross * grad_slope) / determinant
        slope += (hess_cross * grad_intercept - hess_intercept * grad_slope) / determinant
    return float(_sigmoid(intercept))


if __name__ == "__main__":
    scores_a = np.array([0.2, 1.0, 0.0, 0.5])
    scores_b = np.array([0.0, 0.4, 0.0, 0.8])
    labels = np.array([1, -1, 0, 1])
    print("accuracy", pairwise_accuracy(scores_a, scores_b, labels))
    print("agreement", human_agreement(scores_a, scores_b, labels))
    prob = np.array([0.15] * 4 + [0.85] * 4)
    outcome = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    print("ece", expected_calibration_error(prob, outcome, 5))
    wins = [(0, 1, 1), (0, 1, 1), (0, 1, 1), (0, 1, 0), (0, 1, 0.5)]
    print("strengths", bt_fit(wins))
