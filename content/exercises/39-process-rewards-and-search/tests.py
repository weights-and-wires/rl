"""Checks for the process-rewards-and-search exercise."""

import itertools
import math

import numpy as np

from rlcheck import almost, check, close, equal, fail, has, shape


def _toy(sub):
    return has(sub, "toy_policy"), has(sub, "V"), has(sub, "T"), has(sub, "S")


def _brute_force_bon(probs, rewards, n):
    """E[max of n draws] by summing over every ordered n-tuple."""
    total = 0.0
    for idx in itertools.product(range(len(probs)), repeat=n):
        total += np.prod(probs[list(idx)]) * rewards[list(idx)].max()
    return total


def _compositions(n, k):
    """Every way to split n samples among k answers, as count tuples."""
    if k == 1:
        return [(n,)]
    return [(c,) + rest for c in range(n + 1) for rest in _compositions(n - c, k - 1)]


def _vote_accuracy(vote, answer_probs, correct, n, weight_of=None):
    """Exact probability that a vote over n i.i.d. samples returns `correct`.

    A vote depends only on how many samples land on each answer, so one
    representative row per composition, weighted by its multinomial
    probability, gives the exact accuracy.
    """
    k = len(answer_probs)
    comps = _compositions(n, k)
    rows = np.array([sum(([j] * c for j, c in enumerate(comp)), []) for comp in comps])
    log_p = np.array([
        math.lgamma(n + 1) - sum(math.lgamma(c + 1) for c in comp)
        + sum(c * math.log(q) for c, q in zip(comp, answer_probs))
        for comp in comps
    ])
    if weight_of is None:
        winners = vote(rows, k)
    else:
        winners = vote(rows, weight_of[rows], k)
    return float(np.sum(np.exp(log_p) * (np.asarray(winners) == correct)))


@check("best_of_n_expected_reward gives 0.555 on the lesson's example and matches brute force")
def test_best_of_n(sub):
    bon = has(sub, "best_of_n_expected_reward")
    probs = np.array([0.5, 0.3, 0.2])
    rewards = np.array([0.0, 0.5, 1.0])
    almost(bon(probs, rewards, 1), 0.35, 1e-12, label="n = 1 is the plain mean")
    almost(bon(probs, rewards, 2), 0.555, 1e-12, label="the lesson's n = 2 example")

    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(5))
    r = rng.normal(size=5)
    for n in (1, 2, 3, 4):
        almost(bon(p, r, n), _brute_force_bon(p, r, n), 1e-10,
               label=f"order-statistic formula against every {n}-tuple")

    # Ties: two outcomes share a reward. The formula must still be exact.
    r_tied = np.array([1.0, 0.0, 1.0, 2.0, 0.0])
    almost(bon(p, r_tied, 3), _brute_force_bon(p, r_tied, 3), 1e-10,
           label="rewards with ties")

    # A 0/1 reward: best-of-n succeeds unless all n samples fail.
    q = 0.3
    for n in (1, 5, 20):
        almost(bon(np.array([1 - q, q]), np.array([0.0, 1.0]), n), 1 - (1 - q) ** n,
               1e-12, label=f"0/1 reward, n = {n}, against 1 - (1 - p)^n")


@check("majority_vote picks the most frequent answer and breaks ties toward the smallest")
def test_majority_vote(sub):
    vote = has(sub, "majority_vote")
    answers = np.array([
        [2, 2, 1, 0, 2],
        [0, 1, 1, 0, 3],   # 0 and 1 tie with two votes each
        [3, 3, 3, 3, 3],
    ])
    got = np.asarray(vote(answers, 4))
    shape(got, (3,), label="majority_vote output")
    equal(got, np.array([2, 0, 3]), label="winners")


@check("Self-consistency accuracy is 0.68256 for p = 0.6 and n = 5, and falls when the mode is wrong")
def test_self_consistency_accuracy(sub):
    vote = has(sub, "majority_vote")
    acc = _vote_accuracy(vote, np.array([0.6, 0.4]), 0, 5)
    almost(acc, 0.68256, 1e-10, label="P(at least 3 of 5 correct) at p = 0.6")

    # The correct answer, index 2, is not the most likely one.
    probs = np.array([0.55, 0.10, 0.35])
    almost(_vote_accuracy(vote, probs, 2, 1), 0.35, 1e-12, label="one sample")
    acc = _vote_accuracy(vote, probs, 2, 51)
    if acc > 0.1:
        fail("when a wrong answer is the mode, majority-vote accuracy should fall "
             f"toward zero: with 51 samples, expected below 0.1, got {acc:.4f}")


@check("weighted_majority_vote equals majority_vote at unit weights and recovers a wrong mode")
def test_weighted_vote(sub):
    vote = has(sub, "majority_vote")
    wvote = has(sub, "weighted_majority_vote")
    rng = np.random.default_rng(1)
    answers = rng.integers(0, 5, size=(200, 7))
    equal(np.asarray(wvote(answers, np.ones(answers.shape), 5)), np.asarray(vote(answers, 5)),
          label="unit weights against plain majority voting")

    got = np.asarray(wvote(np.array([[1, 1, 0]]), np.array([[0.2, 0.3, 0.9]]), 2))
    equal(got, np.array([0]), label="one confident vote for 0 outweighs two weak votes for 1")

    # A verifier that scores correct solutions 0.9 and wrong ones 0.3.
    probs = np.array([0.55, 0.10, 0.35])
    weight_of = np.array([0.3, 0.3, 0.9])
    acc = _vote_accuracy(wvote, probs, 2, 51, weight_of)
    if acc < 0.95:
        fail("with verifier weights 0.9 for correct and 0.3 for wrong, weighted "
             f"voting over 51 samples should be right at least 0.95 of the time; got {acc:.4f}")


@check("step_values equals the success probability found by enumerating all 4096 solutions")
def test_step_values(sub):
    toy_policy, V, T, S = _toy(sub)
    step_values = has(sub, "step_values")
    p = toy_policy(3)
    seqs = np.array(list(itertools.product(range(V), repeat=T)))
    seq_probs = np.prod(p[np.arange(T), seqs], axis=1)
    for target in (0, 9, 14, 21, 28):
        values = np.asarray(step_values(p, target))
        shape(values, (T + 1, S), label="step_values output")
        success = seqs.sum(axis=1) == target
        almost(values[0, 0], seq_probs[success].sum(), 1e-12,
               label=f"value of the empty prefix, target {target}")
        # Every length-2 prefix: condition the enumeration on the first two digits.
        for a in range(V):
            for b in range(V):
                rows = (seqs[:, 0] == a) & (seqs[:, 1] == b)
                want = seq_probs[rows & success].sum() / seq_probs[rows].sum()
                if abs(values[2, a + b] - want) > 1e-12:
                    fail(f"values[2, {a + b}] for prefix ({a}, {b}) and target {target} "
                         f"should be {want:.6f}, got {values[2, a + b]:.6f}")
        close(values[T], (np.arange(S) == target).astype(float), atol=0,
              label="values[T] is the outcome reward")


@check("prm_beam_search keeps the highest scores and breaks ties toward earlier candidates")
def test_beam_search_mechanics(sub):
    _, V, T, _ = _toy(sub)
    search = has(sub, "prm_beam_search")
    beams, scores = search(lambda c: c.sum(axis=1).astype(float), 1)
    shape(beams, (1, T), label="beams at width 1")
    equal(np.asarray(beams)[0], np.full(T, V - 1), label="a PRM that likes big digits")
    almost(float(np.asarray(scores)[0]), float((V - 1) * T), 1e-12, label="its score")

    beams, _ = search(lambda c: np.zeros(len(c)), 3)
    equal(np.asarray(beams), np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]),
          label="a constant PRM keeps the first candidates")

    beams, scores = search(lambda c: -np.abs(c.sum(axis=1) - 2 * c.shape[1]), 4)
    shape(beams, (4, T), label="beams at width 4")
    if not np.all(np.asarray(beams).sum(axis=1) == 2 * T):
        fail(f"a PRM that wants every prefix to average 2 per digit should keep "
             f"sums of {2 * T}; got sums {np.asarray(beams).sum(axis=1)}")


@check("Beam search guided by the exact step value solves every reachable target at width 1")
def test_exact_value_greedy(sub):
    toy_policy, V, T, S = _toy(sub)
    step_values = has(sub, "step_values")
    search = has(sub, "prm_beam_search")
    p = toy_policy(0)
    for target in range(S):
        values = step_values(p, target)
        beams, _ = search(lambda c, v=values: v[c.shape[1], c.sum(axis=1)], 1)
        got = int(np.asarray(beams)[0].sum())
        if got != target:
            fail(f"target {target}: greedy search on the true value should end on "
                 f"the target, ended on sum {got}. The value of the empty prefix "
                 f"is {values[0, 0]:.4f}, so a correct solution exists.")


@check("With a noisy PRM, width 8 beats width 1 by 0.08 and both beat sampling by far")
def test_noisy_prm_width(sub):
    toy_policy, _, _, _ = _toy(sub)
    step_values = has(sub, "step_values")
    search = has(sub, "prm_beam_search")
    p = toy_policy(0)
    targets = range(4, 25)
    base = np.mean([step_values(p, t)[0, 0] for t in targets])
    rates = {}
    for width in (1, 8):
        wins = []
        for seed in range(20):
            rng = np.random.default_rng(100 + seed)
            for target in targets:
                values = step_values(p, target)
                noisy = values + rng.normal(scale=0.1, size=values.shape)
                beams, _ = search(lambda c, v=noisy: v[c.shape[1], c.sum(axis=1)], width)
                wins.append(np.asarray(beams)[0].sum() == target)
        rates[width] = float(np.mean(wins))
    if rates[1] < 0.6 or rates[8] < 0.8 or rates[8] - rates[1] < 0.08:
        fail(f"expected success of at least 0.6 at width 1 and 0.8 at width 8, "
             f"with a gap of 0.08; got {rates[1]:.3f} and {rates[8]:.3f}. "
             f"Sampling from the base policy succeeds {base:.3f} of the time.")
