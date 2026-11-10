"""Language models as policies: sequence log-probabilities, the KL-regularised
optimum, and best-of-n.

Fill in every function that raises NotImplementedError, in order.

Shapes follow the convention used through Part 9:

- per-token log-probabilities and response masks are `(B, T)`,
- sequence rewards and sequence log-probabilities are `(B,)`.
"""

import itertools

import numpy as np


# The toy language model ---------------------------------------------------


class ToyLM:
    """A tiny autoregressive policy over `vocab` tokens and `length` positions.

    The next-token logits depend on the position `t` and the previous token.
    At `t = 0` there is no previous token, so the model reads a start symbol,
    stored as index `vocab`. The whole model is one table of logits with shape
    `(length, vocab + 1, vocab)`.
    """

    def __init__(self, vocab=4, length=3, seed=0, scale=1.0):
        rng = np.random.default_rng(seed)
        self.vocab = vocab
        self.length = length
        self.table = scale * rng.normal(size=(length, vocab + 1, vocab))

    def logits(self, tokens):
        """Next-token logits for every position of `tokens`, shape `(B, T, vocab)`."""
        tokens = np.asarray(tokens)
        batch, steps = tokens.shape
        start = np.full((batch, 1), self.vocab)
        prev = np.concatenate([start, tokens[:, :-1]], axis=1)
        return self.table[np.arange(steps)[None, :], prev]




def all_sequences(vocab, length):
    """Every sequence of `length` tokens, shape `(vocab ** length, length)`.

    The order is lexicographic, so row `i` matches index `i` of an array of
    shape `(vocab,) * length` flattened in C order.
    """
    return np.array(list(itertools.product(range(vocab), repeat=length)), dtype=int)


def _logsumexp(z, axis=-1):
    top = np.max(z, axis=axis, keepdims=True)
    return np.squeeze(top, axis=axis) + np.log(np.sum(np.exp(z - top), axis=axis))


# Your functions -------------------------------------------------------------


def token_logprobs(logits, tokens):
    """Log-probability of each chosen token under the model.

    Args:
        logits: next-token logits, shape `(B, T, V)`. Row `[b, t]` is the
            model's score for every token at position `t` of sequence `b`.
        tokens: the tokens actually chosen, integers, shape `(B, T)`.

    Returns:
        `log pi(tokens[b, t] | prefix)`, shape `(B, T)`: a log-softmax over the
        last axis of `logits`, read off at the chosen token.
    """
    raise NotImplementedError("token_logprobs")


def sequence_logprob(token_logps, mask):
    """Log-probability of each whole response: the masked sum of token log-probs.

    Args:
        token_logps: per-token log-probabilities, shape `(B, T)`. Positions
            outside the response can hold anything, including `-inf` or `nan`.
        mask: response mask, shape `(B, T)`, 1 (or True) for tokens that belong
            to the response and 0 for padding.

    Returns:
        `log pi(y | x)` for each sequence, shape `(B,)`.
    """
    raise NotImplementedError("sequence_logprob")


def kl_optimal_policy(ref_logps, rewards, beta):
    """The optimum of E[r] - beta * KL(pi || pi_ref) over a finite set of responses.

    Args:
        ref_logps: `log pi_ref(y)` for every response, shape `(N,)`.
        rewards: `r(y)` for the same responses, shape `(N,)`.
        beta: the KL coefficient, a positive float.

    Returns:
        `pi*(y) = pi_ref(y) exp(r(y) / beta) / Z` as probabilities, shape `(N,)`.
    """
    raise NotImplementedError("kl_optimal_policy")


def kl_regularised_objective(probs, ref_logps, rewards, beta):
    """J(pi) = E_pi[r] - beta * KL(pi || pi_ref) for a distribution over responses.

    Args:
        probs: `pi(y)` for every response, shape `(N,)`, summing to 1. Entries
            can be exactly 0.
        ref_logps: `log pi_ref(y)`, shape `(N,)`.
        rewards: `r(y)`, shape `(N,)`.
        beta: the KL coefficient.

    Returns:
        The scalar objective. A response with `pi(y) = 0` adds nothing to the KL.
    """
    raise NotImplementedError("kl_regularised_objective")





def best_of_n_probs(ref_probs, rewards, n):
    """The exact distribution of best-of-n sampling over a finite set.

    Draw `n` responses independently from `pi_ref` and keep the one with the
    highest reward. Rewards are distinct, so there are no ties.

    Args:
        ref_probs: `pi_ref(y)`, shape `(N,)`, summing to 1.
        rewards: `r(y)`, shape `(N,)`, all distinct.
        n: the number of draws, a positive integer.

    Returns:
        `pi_BoN(y)`, shape `(N,)`, in the same order as the inputs.
    """
    raise NotImplementedError("best_of_n_probs")


def best_of_n_kl(ref_probs, rewards, n):
    """Exact KL(pi_BoN || pi_ref) for best-of-n over a finite set.

    Args:
        ref_probs: `pi_ref(y)`, shape `(N,)`, summing to 1.
        rewards: `r(y)`, shape `(N,)`, all distinct.
        n: the number of draws.

    Returns:
        The KL in nats, a float.
    """
    raise NotImplementedError("best_of_n_kl")


if __name__ == "__main__":
    vocab, length, beta = 4, 3, 0.5
    lm = ToyLM(vocab=vocab, length=length, seed=0)
    seqs = all_sequences(vocab, length)
    mask = np.ones(seqs.shape, dtype=bool)
    ref_logps = sequence_logprob(token_logprobs(lm.logits(seqs), seqs), mask)
    print(f"{len(seqs)} sequences, total probability {np.exp(ref_logps).sum():.6f}")

    # Reward: 1 for a strictly increasing sequence, else 0.
    rewards = np.all(np.diff(seqs, axis=1) > 0, axis=1).astype(float)
    pi_star = kl_optimal_policy(ref_logps, rewards, beta)
    log_z = np.log(np.sum(np.exp(ref_logps + rewards / beta)))
    print(f"E_ref[r] = {np.exp(ref_logps) @ rewards:.4f}   E_pi*[r] = {pi_star @ rewards:.4f}")
    print(f"J(pi*) = {kl_regularised_objective(pi_star, ref_logps, rewards, beta):.6f}"
          f"   beta log Z = {beta * log_z:.6f}")

    print("\n  n   exact BoN KL   log n - (n-1)/n")
    ref_probs = np.exp(ref_logps)
    scores = np.random.default_rng(1).normal(size=len(seqs))
    for n in (1, 2, 4, 8, 16, 64):
        bound = np.log(n) - (n - 1) / n
        print(f"{n:3d}   {max(best_of_n_kl(ref_probs, scores, n), 0.0):12.4f}   {bound:15.4f}")
