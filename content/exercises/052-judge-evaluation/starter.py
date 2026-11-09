"""Evaluate a reward model or a judge you already have.

Fill in every function that raises NotImplementedError. Nothing here trains
a reward model. Scores, probabilities, and labels are one-dimensional arrays.
"""

import numpy as np


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
    raise NotImplementedError("pairwise_accuracy")


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
    raise NotImplementedError("expected_calibration_error")


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
    raise NotImplementedError("human_agreement")


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
    raise NotImplementedError("swap_consistency")


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
    raise NotImplementedError("bt_fit")


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
    raise NotImplementedError("length_controlled_winrate")


if __name__ == "__main__":
    scores_a = np.array([0.2, 1.0, 0.0, 0.5])
    scores_b = np.array([0.0, 0.4, 0.0, 0.8])
    labels = np.array([1, -1, 0, 1])
    print(pairwise_accuracy(scores_a, scores_b, labels))
