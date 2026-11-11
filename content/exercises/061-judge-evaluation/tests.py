"""Checks for reward-model and judge evaluation."""

import math

import numpy as np

from rlcheck import almost, check, close, has


def _sigmoid(z):
    z = np.asarray(z, dtype=float)
    positive = z >= 0
    out = np.empty(np.shape(z), dtype=float)
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    ez = np.exp(z[~positive])
    out[~positive] = ez / (1.0 + ez)
    return out


def _position_judge(biased_ids):
    """Prefer the presented-first answer on `biased_ids`, else follow quality.

    Each answer is `(pair_id, quality)`. Both sides of a pair carry the same
    pair id, so the bias stays attached to the pair after a swap.
    """

    def judge(left, right):
        pair_id, quality_left = left
        _, quality_right = right
        if pair_id in biased_ids:
            return 1
        if quality_left == quality_right:
            return 0
        return 1 if quality_left > quality_right else -1

    return judge


# The four-prompt table in the lesson.
MIRA = np.array([0.2, 1.0, 0.0, 0.5])
NOOR = np.array([0.0, 0.4, 0.0, 0.8])
LABEL = np.array([1, -1, 0, 1])


@check("pairwise_accuracy on the four-prompt table is 1/3")
def test_pairwise_accuracy(sub):
    pairwise_accuracy = has(sub, "pairwise_accuracy")
    got = pairwise_accuracy(MIRA, NOOR, LABEL)
    almost(got, 1.0 / 3.0, 1e-12, label="pairwise accuracy")


@check("a score tie on a decisive label counts as a miss, not as half a point")
def test_score_tie_is_a_miss(sub):
    pairwise_accuracy = has(sub, "pairwise_accuracy")
    got = pairwise_accuracy(
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([1, -1]),
    )
    almost(got, 0.0, 1e-12, label="accuracy on two score ties")


@check("human_agreement keeps the mutual tie and is 1/2 on the four-prompt table")
def test_human_agreement(sub):
    human_agreement = has(sub, "human_agreement")
    got = human_agreement(MIRA, NOOR, LABEL)
    almost(got, 0.5, 1e-12, label="agreement on the four-prompt table")
    ties = human_agreement(
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([1, -1]),
    )
    almost(ties, 0.0, 1e-12, label="agreement when decisive labels meet equal scores")


@check("expected calibration error of the position-biased judge is 0.35")
def test_calibration(sub):
    # The bin centers are 0.10 and 0.90. Using those centers instead of the
    # mean stated probability scores this judge at 0.40, not 0.35.
    expected_calibration_error = has(sub, "expected_calibration_error")
    prob = np.array([0.15, 0.15, 0.15, 0.15, 0.85, 0.85, 0.85, 0.85])
    outcome = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    got = expected_calibration_error(prob, outcome, 5)
    almost(got, 0.35, 1e-12, label="ECE with five bins")


@check("swap_consistency recovers a planted bias and scores an always-first judge at 0")
def test_swap_consistency(sub):
    # Wrong implementation this check rejects: a swap-consistency score that
    # ignores the swapped order. It calls judge(left, right) twice and treats
    # agreement of those two calls as consistency. That score is 1 on every
    # deterministic judge below, so an always-first judge looks fully
    # consistent and a planted bias of 0.25 is reported as 0.
    swap_consistency = has(sub, "swap_consistency")
    lesson_pairs = [
        ((0, 3), (0, 1)),
        ((1, 1), (1, 4)),
        ((2, 5), (2, 2)),
        ((3, 2), (3, 2)),
    ]
    lesson = swap_consistency(_position_judge({0, 1}), lesson_pairs)
    almost(1.0 - lesson, 0.5, 1e-12, label="bias rate on the four-pair judge")

    eight = [
        ((i, left), (i, right))
        for i, (left, right) in enumerate(
            [(3, 1), (1, 4), (5, 2), (2, 2), (0, 3), (4, 1), (2, 6), (7, 3)]
        )
    ]
    mixed = swap_consistency(_position_judge({0, 1}), eight)
    almost(1.0 - mixed, 0.25, 1e-12, label="bias rate on 2 biased pairs out of 8")

    always = swap_consistency(_position_judge({0, 1, 2, 3}), lesson_pairs)
    almost(always, 0.0, 1e-12, label="consistency of a judge that always returns +1")


@check("bt_fit matches the two-player closed form when one comparison is a tie")
def test_bt_two_players(sub):
    # Wrong implementation this check rejects: an Elo fit that treats a tie
    # as a win for the first player. On (0, 1, 0.5) it credits a full win to
    # player 0. The totals become 4 and 1, and the centered strength is
    # 0.5 * log(4) = 0.693147. The half-win closed form is 0.5 * log(7/3)
    # = 0.423649.
    bt_fit = has(sub, "bt_fit")
    wins = [(0, 1, 1), (0, 1, 1), (0, 1, 1), (0, 1, 0), (0, 1, 0.5)]
    half_gap = 0.5 * math.log(3.5 / 1.5)
    got = bt_fit(wins)
    close(got, np.array([half_gap, -half_gap]), rtol=0.0, atol=1e-8, label="two-player strengths")


@check("bt_fit recovers three known strengths from their Bradley-Terry probabilities")
def test_bt_three_players(sub):
    # Population wins make the true strengths the maximizer, where the
    # log-likelihood gradient is 0. Thirty-two MM steps from equal weights
    # leave an error below 1e-7 on this instance. The tolerance 1e-5 sits
    # above that leftover and below the 0.27 error of counting a tie as a
    # full win for the first player.
    bt_fit = has(sub, "bt_fit")
    truth = np.array([1.0, 0.0, -1.0])
    wins = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        wins.append((i, j, float(_sigmoid(truth[i] - truth[j]))))
    got = bt_fit(wins)
    close(got, truth, rtol=0.0, atol=1e-5, label="three-player strengths")


@check("a length-only judge scores 50 percent after the length control")
def test_length_control(sub):
    length_controlled_winrate = has(sub, "length_controlled_winrate")
    gap = np.array([4.0, 4.0, 4.0, 4.0, -2.0])
    # Population win rate of a judge with no intercept, sigmoid(1.5 * gap).
    # The raw mean is 0.8075. The controlled rate is 0.5, shifted by about
    # 1e-6 by the ridge of 1e-6 on the length coefficient. The tolerance
    # 1e-5 is above that shift and far below the raw mean.
    win = _sigmoid(1.5 * gap)
    got = length_controlled_winrate(gap, win)
    almost(got, 0.5, 1e-5, label="length-controlled win rate")


@check("with no length gap, the controlled win rate is the raw rate of 0.75")
def test_length_base_rate(sub):
    length_controlled_winrate = has(sub, "length_controlled_winrate")
    got = length_controlled_winrate(np.zeros(4), np.array([1.0, 1.0, 1.0, 0.0]))
    almost(got, 0.75, 1e-8, label="win rate at a length gap of 0")
