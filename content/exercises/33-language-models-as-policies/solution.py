"""Language models as policies: sequence log-probabilities, the KL-regularised
optimum, and best-of-n.

This is the reference solution. It has the same public surface as the starter.

Shapes follow the convention used through Part 8:

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

    def sample(self, n, rng):
        """Draw `n` sequences, shape `(n, length)`, by sampling token by token."""
        tokens = np.zeros((n, self.length), dtype=int)
        prev = np.full(n, self.vocab)
        for t in range(self.length):
            logits = self.table[t, prev]
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            u = rng.random((n, 1))
            tokens[:, t] = np.minimum((probs.cumsum(axis=1) < u).sum(axis=1), self.vocab - 1)
            prev = tokens[:, t]
        return tokens

    def conditional_tree(self):
        """Next-token log-probabilities for every prefix, as a list of arrays.

        Entry `t` has shape `(vocab,) * t + (vocab,)`. Its leading `t` axes
        index the prefix `y_0 .. y_{t-1}`, and its last axis indexes the next
        token `y_t`.
        """
        v = self.vocab
        tree = []
        for t in range(self.length):
            table = self.table[t] - self.table[t].max(axis=1, keepdims=True)
            logp = table - np.log(np.exp(table).sum(axis=1, keepdims=True))
            if t == 0:
                tree.append(logp[v].copy())
            else:
                shape = (1,) * (t - 1) + (v, v)
                tree.append(np.broadcast_to(logp[:v].reshape(shape), (v,) * t + (v,)).copy())
        return tree


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
    logits = np.asarray(logits, dtype=float)
    log_norm = _logsumexp(logits, axis=-1)
    chosen = np.take_along_axis(logits, np.asarray(tokens)[..., None], axis=-1)[..., 0]
    return chosen - log_norm


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
    mask = np.asarray(mask).astype(bool)
    return np.where(mask, token_logps, 0.0).sum(axis=-1)


def kl_optimal_policy(ref_logps, rewards, beta):
    """The optimum of E[r] - beta * KL(pi || pi_ref) over a finite set of responses.

    Args:
        ref_logps: `log pi_ref(y)` for every response, shape `(N,)`.
        rewards: `r(y)` for the same responses, shape `(N,)`.
        beta: the KL coefficient, a positive float.

    Returns:
        `pi*(y) = pi_ref(y) exp(r(y) / beta) / Z` as probabilities, shape `(N,)`.
    """
    z = np.asarray(ref_logps, dtype=float) + np.asarray(rewards, dtype=float) / beta
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


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
    probs = np.asarray(probs, dtype=float)
    safe = np.where(probs > 0, probs, 1.0)
    kl = np.sum(np.where(probs > 0, probs * (np.log(safe) - ref_logps), 0.0))
    return float(np.sum(probs * rewards) - beta * kl)


def soft_token_policy(ref_cond_logps, rewards, beta):
    """Solve the token-level MDP by soft backward induction.

    The state is the prefix, the action is the next token, and the only reward
    is `r(y)`, paid on the last token. Apply the lesson's soft Bellman backup
    from the last position to the first.

    Args:
        ref_cond_logps: list of `T` arrays. Entry `t` has shape
            `(V,) * t + (V,)` and holds `log pi_ref(y_t | y_0 .. y_{t-1})`.
        rewards: `r(y)` for every full sequence, shape `(V,) * T`.
        beta: the KL coefficient.

    Returns:
        A pair `(policy_cond_logps, v0)`. `policy_cond_logps` is a list shaped
        like `ref_cond_logps` holding `log pi*(y_t | prefix)`, and `v0` is the
        float `V(s_0)`, the soft value of the empty prefix.
    """
    v_next = np.asarray(rewards, dtype=float)
    policy = [None] * len(ref_cond_logps)
    for t in reversed(range(len(ref_cond_logps))):
        z = ref_cond_logps[t] + v_next / beta
        v = beta * _logsumexp(z, axis=-1)
        policy[t] = z - v[..., None] / beta
        v_next = v
    return policy, float(v_next)


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
    ref_probs = np.asarray(ref_probs, dtype=float)
    order = np.argsort(rewards, kind="stable")
    cdf = np.minimum(np.cumsum(ref_probs[order]), 1.0)
    below = np.concatenate([[0.0], cdf[:-1]])
    out = np.empty_like(ref_probs)
    out[order] = cdf**n - below**n
    return out


def best_of_n_kl(ref_probs, rewards, n):
    """Exact KL(pi_BoN || pi_ref) for best-of-n over a finite set.

    Args:
        ref_probs: `pi_ref(y)`, shape `(N,)`, summing to 1.
        rewards: `r(y)`, shape `(N,)`, all distinct.
        n: the number of draws.

    Returns:
        The KL in nats, a float.
    """
    ref_probs = np.asarray(ref_probs, dtype=float)
    p = best_of_n_probs(ref_probs, rewards, n)
    keep = p > 0
    return float(np.sum(p[keep] * np.log(p[keep] / ref_probs[keep])))


if __name__ == "__main__":
    vocab, length, beta = 4, 3, 0.5
    lm = ToyLM(vocab=vocab, length=length, seed=0)
    seqs = all_sequences(vocab, length)
    mask = np.ones(seqs.shape, dtype=bool)
    ref_logps = sequence_logprob(token_logprobs(lm.logits(seqs), seqs), mask)
    print(f"{len(seqs)} sequences, total probability {np.exp(ref_logps).sum():.6f}")

    # Reward: 1 for a strictly increasing sequence, else 0. It isn't a sum of
    # per-token scores, so the optimal token policy must read the whole prefix.
    rewards = np.all(np.diff(seqs, axis=1) > 0, axis=1).astype(float)
    pi_star = kl_optimal_policy(ref_logps, rewards, beta)
    log_z = np.log(np.sum(np.exp(ref_logps + rewards / beta)))
    print(f"E_ref[r] = {np.exp(ref_logps) @ rewards:.4f}   E_pi*[r] = {pi_star @ rewards:.4f}")
    print(f"J(pi*) = {kl_regularised_objective(pi_star, ref_logps, rewards, beta):.6f}"
          f"   beta log Z = {beta * log_z:.6f}")

    policy, v0 = soft_token_policy(lm.conditional_tree(), rewards.reshape((vocab,) * length), beta)
    token_path = sum(policy[t][tuple(seqs[:, : t + 1].T)] for t in range(length))
    print(f"soft backward induction: V(s_0) = {v0:.6f}, "
          f"max |pi_token - pi*| = {np.abs(np.exp(token_path) - pi_star).max():.2e}")

    print("\n  n   exact BoN KL   log n - (n-1)/n")
    ref_probs = np.exp(ref_logps)
    scores = np.random.default_rng(1).normal(size=len(seqs))
    for n in (1, 2, 4, 8, 16, 64):
        bound = np.log(n) - (n - 1) / n
        print(f"{n:3d}   {max(best_of_n_kl(ref_probs, scores, n), 0.0):12.4f}   {bound:15.4f}")
