"""Process rewards and test-time search: reference solution.

The toy reasoning task has a known step-level value. A "solution" is a
sequence of T = 4 digits from 0 to 7, and it is correct when the digits sum
to a target. The base policy picks each digit independently from a fixed
per-position distribution `step_probs` of shape (T, V). Because the reward
depends only on the running sum, the exact value of a prefix is a function
of (step t, partial sum s), and a backward pass computes all of them.
"""

import itertools

import numpy as np

V = 8                    # vocabulary: the digits 0 to 7
T = 4                    # steps per solution
S = (V - 1) * T + 1      # partial sums run from 0 to 28


def toy_policy(seed: int = 0) -> np.ndarray:
    """A fixed base policy: one softmax over the V digits per step, shape (T, V)."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(T, V))
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def all_sequences() -> np.ndarray:
    """Every one of the V**T solutions, shape (V**T, T), in lexicographic order."""
    return np.array(list(itertools.product(range(V), repeat=T)))


def best_of_n_expected_reward(probs: np.ndarray, rewards: np.ndarray, n: int) -> float:
    """Exact expected reward of best-of-n with a perfect scorer.

    Args:
        probs: probability of each distinct outcome, shape (M,), summing to one.
        rewards: the reward of each outcome, shape (M,).
        n: how many independent samples best-of-n draws.

    Returns:
        E[max of n i.i.d. rewards], from the order-statistic formula
        sum_i r_(i) * (F_i**n - F_(i-1)**n) over outcomes sorted by reward.
    """
    order = np.argsort(rewards, kind="stable")
    r = np.asarray(rewards, dtype=float)[order]
    F = np.cumsum(np.asarray(probs, dtype=float)[order])
    F_prev = np.concatenate([[0.0], F[:-1]])
    return float(np.sum(r * (F**n - F_prev**n)))


def majority_vote(answers: np.ndarray, n_answers: int) -> np.ndarray:
    """Self-consistency: the most frequent answer in each row.

    Args:
        answers: integer final answers, shape (B, n), each in [0, n_answers).
        n_answers: the number of distinct answers.

    Returns:
        The winning answer per row, shape (B,). Ties go to the smallest answer.
    """
    return weighted_majority_vote(answers, np.ones(answers.shape), n_answers)


def weighted_majority_vote(
    answers: np.ndarray, weights: np.ndarray, n_answers: int
) -> np.ndarray:
    """Weighted majority voting: the answer with the largest total weight.

    Args:
        answers: integer final answers, shape (B, n), each in [0, n_answers).
        weights: a verifier's score for each sampled solution, shape (B, n).
        n_answers: the number of distinct answers.

    Returns:
        The answer whose samples carry the most total weight, shape (B,).
        Ties go to the smallest answer.
    """
    onehot = np.eye(n_answers)[answers]                  # (B, n, n_answers)
    totals = np.einsum("bn,bnk->bk", weights, onehot)    # (B, n_answers)
    return np.argmax(totals, axis=1)


def step_values(step_probs: np.ndarray, target: int) -> np.ndarray:
    """Exact value of every (step, partial sum) under the base policy.

    Args:
        step_probs: the base policy, shape (T, V).
        target: the sum that counts as a correct solution.

    Returns:
        values, shape (T + 1, S). values[t, s] is the probability that the
        base policy finishes with the target sum, given that the first t
        digits sum to s. values[T, s] is 1 when s == target and 0 otherwise.
    """
    n_steps, n_tok = step_probs.shape
    values = np.zeros((n_steps + 1, S))
    values[n_steps, target] = 1.0
    for t in reversed(range(n_steps)):
        # Pad so that a partial sum past the table's end has value zero.
        nxt = np.concatenate([values[t + 1], np.zeros(n_tok)])
        windows = np.stack([nxt[s : s + n_tok] for s in range(S)])   # (S, V)
        values[t] = windows @ step_probs[t]
    return values


def prm_beam_search(prm, beam_width: int, n_tokens: int = V, horizon: int = T):
    """Beam search that expands every beam by every token and keeps the top scores.

    Args:
        prm: a function from a batch of prefixes, an int array of shape (K, t),
            to one score per prefix, shape (K,). Larger is better.
        beam_width: how many prefixes survive each step.
        n_tokens: the vocabulary size.
        horizon: the number of steps.

    Returns:
        (beams, scores): the surviving complete solutions, shape
        (beam_width, horizon), best first, and their PRM scores, shape
        (beam_width,). Ties keep the earlier candidate, where candidates are
        ordered beam by beam and token by token within a beam.
    """
    beams = np.zeros((1, 0), dtype=int)
    scores = np.zeros(1)
    for _ in range(horizon):
        parents = np.repeat(beams, n_tokens, axis=0)
        tokens = np.tile(np.arange(n_tokens), len(beams))[:, None]
        candidates = np.concatenate([parents, tokens], axis=1)
        cand_scores = np.asarray(prm(candidates), dtype=float)
        keep = np.argsort(-cand_scores, kind="stable")[:beam_width]
        beams, scores = candidates[keep], cand_scores[keep]
    return beams, scores


if __name__ == "__main__":
    p = toy_policy(0)
    target = 14
    seqs = all_sequences()
    seq_probs = np.prod(p[np.arange(T), seqs], axis=1)
    correct = (seqs.sum(axis=1) == target).astype(float)

    values = step_values(p, target)
    print(f"base policy solves target {target} with probability {values[0, 0]:.4f}")

    for n in (1, 4, 16, 64):
        bon = best_of_n_expected_reward(seq_probs, correct, n)
        print(f"best-of-{n:<2d} with a perfect verifier: success {bon:.4f}")

    rng = np.random.default_rng(0)
    noise = rng.normal(scale=0.1, size=values.shape)
    noisy = values + noise

    def prm(prefixes):
        return noisy[prefixes.shape[1], prefixes.sum(axis=1)]

    for width in (1, 2, 4, 8):
        beams, _ = prm_beam_search(prm, width)
        print(f"beam width {width}: top solution {beams[0]}, sum {beams[0].sum()}")
