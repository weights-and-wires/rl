"""Reward models: Bradley-Terry training, ensembles, and outcome against process
rewards.

Fill in every function that raises NotImplementedError, in order.

The toy language model and the prefix-tree layout are the same as in the
previous lesson. A tree of conditional log-probabilities is a list whose entry
`t` has shape `(V,) * t + (V,)`: the leading axes index the prefix and the last
axis indexes the next token.
"""

import itertools

import numpy as np
import torch

torch.set_num_threads(1)


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


def position_token_features(tokens, vocab):
    """One-hot features for every (position, token) pair, shape `(B, T * vocab)`.

    A linear reward on these features gives each token a score that depends on
    where it appears, and adds the scores up.
    """
    tokens = np.asarray(tokens)
    batch, steps = tokens.shape
    feats = np.zeros((batch, steps * vocab))
    feats[np.arange(batch)[:, None], np.arange(steps) * vocab + tokens] = 1.0
    return feats


# Your functions -------------------------------------------------------------


def bradley_terry_loss(reward_chosen, reward_rejected, weights=None):
    """Mean negative log-likelihood of the preferences under Bradley-Terry.

    Args:
        reward_chosen: torch tensor `(n,)`, the reward of the preferred response
            in each pair.
        reward_rejected: torch tensor `(n,)`, the reward of the other response.
        weights: optional torch tensor `(n,)` of non-negative pair weights. When
            given, return the weighted mean, `sum(w * loss) / sum(w)`.

    Returns:
        A scalar torch tensor, the mean of `-log sigmoid(chosen - rejected)`.
        It must stay finite for reward gaps in the hundreds.
    """
    raise NotImplementedError("bradley_terry_loss")


def fit_reward_model(x_chosen, x_rejected, steps=500, lr=0.05, weight_decay=0.0, weights=None):
    """Fit a linear reward `r(y) = x(y) @ phi` by full-batch Adam on the BT loss.

    Args:
        x_chosen: features of the preferred responses, numpy `(n, D)`.
        x_rejected: features of the rejected responses, numpy `(n, D)`.
        steps: number of Adam steps.
        lr: Adam learning rate.
        weight_decay: coefficient `c` of an added penalty `c * sum(phi ** 2)`.
        weights: optional numpy `(n,)` pair weights, passed to the loss.

    Returns:
        The fitted parameters `phi`, a numpy array `(D,)`. Start from zeros, in
        float64.
    """
    raise NotImplementedError("fit_reward_model")


def ensemble_reward(rewards, mode="mean", lam=1.0):
    """Combine the scores of K reward models into one reward per response.

    Args:
        rewards: numpy `(K, N)`, the score of each of `K` models for each of
            `N` responses.
        mode: `"mean"` for the average, `"min"` for worst-case optimisation
            (WCO), or `"uwo"` for uncertainty-weighted optimisation.
        lam: the UWO penalty weight.

    Returns:
        numpy `(N,)`. UWO is the mean minus `lam` times the variance across the
        K models, with the variance taken over K (not K - 1).
    """
    raise NotImplementedError("ensemble_reward")


def prefix_values(cond_logps, rewards):
    """Exact value of every prefix under a policy, by backward induction.

    Args:
        cond_logps: the policy's prefix tree, a list of `T` arrays; entry `t`
            has shape `(V,) * t + (V,)`.
        rewards: outcome reward of every full response, shape `(V,) * T`.

    Returns:
        A list of `T + 1` arrays. Entry `t` has shape `(V,) * t` and holds
        `V(y_0 .. y_{t-1})`, the expected outcome reward from that prefix on.
        Entry `T` is `rewards`, and entry 0 is a 0-d array, the expected reward
        of the whole policy.
    """
    raise NotImplementedError("prefix_values")


def math_shepherd_labels(rng, cond_logps, rewards, prefix, n_rollouts):
    """Monte Carlo step labels for one prefix, as in Math-Shepherd.

    Sample `n_rollouts` completions of `prefix` from the policy and score each
    finished response with the 0-or-1 outcome reward.

    Args:
        rng: a `numpy.random.Generator`.
        cond_logps: the policy's prefix tree, as in `prefix_values`.
        rewards: 0-or-1 outcome reward of every full response, `(V,) * T`.
        prefix: the tokens so far, a sequence of `t` integers with `t < T`.
        n_rollouts: number of completions to sample.

    Returns:
        A pair of floats `(soft, hard)`: the fraction of completions that
        succeed, and 1.0 if any completion succeeds, else 0.0.
    """
    raise NotImplementedError("math_shepherd_labels")


def process_rewards(tokens, values):
    """Per-step process rewards: the change in value that each token causes.

    Args:
        tokens: responses, integers `(B, T)`.
        values: the list from `prefix_values`.

    Returns:
        numpy `(B, T)`. Entry `[b, t]` is `V(y_0 .. y_t) - V(y_0 .. y_{t-1})`
        along response `b`. Each row sums to `r(y) - V(empty prefix)`.
    """
    raise NotImplementedError("process_rewards")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    vocab, length = 4, 3
    lm = ToyLM(vocab=vocab, length=length, seed=0)
    seqs = all_sequences(vocab, length)
    feats = position_token_features(seqs, vocab)
    true_reward = feats @ rng.normal(size=feats.shape[1])

    # Label 3000 pairs of sampled responses with Bradley-Terry probabilities.
    a, b = lm.sample(3000, rng), lm.sample(3000, rng)
    ia, ib = [np.ravel_multi_index(s.T, (vocab,) * length) for s in (a, b)]
    a_wins = rng.random(3000) < 1 / (1 + np.exp(-(true_reward[ia] - true_reward[ib])))
    chosen, rejected = np.where(a_wins, ia, ib), np.where(a_wins, ib, ia)
    phi = fit_reward_model(feats[chosen], feats[rejected], steps=300)
    fitted = feats @ phi
    shift = np.mean(true_reward - fitted)
    print("response  true reward  fitted + constant")
    for i in range(0, len(seqs), 9):
        print(f"{str(seqs[i]):>8}  {true_reward[i]:11.3f}  {fitted[i] + shift:17.3f}")
    print(f"correlation over all {len(seqs)} responses: {np.corrcoef(fitted, true_reward)[0, 1]:.4f}")

    # Outcome against process rewards on a digit-sum task.
    lm4 = ToyLM(vocab=4, length=4, seed=1)
    tree = lm4.conditional_tree()
    all4 = all_sequences(4, 4)
    outcome = (all4.sum(axis=1) == 6).astype(float).reshape((4,) * 4)
    values = prefix_values(tree, outcome)
    sample = lm4.sample(3, rng)
    print(f"\nsuccess probability of the policy: {float(values[0]):.4f}")
    for row, steps in zip(sample, process_rewards(sample, values)):
        print(f"{row}  outcome {outcome[tuple(row)]:.0f}  process rewards {np.round(steps, 3)}")
    soft, hard = math_shepherd_labels(rng, tree, outcome, sample[0][:2], 16)
    print(f"Math-Shepherd labels for prefix {sample[0][:2]}: soft {soft:.3f}, hard {hard:.0f}, "
          f"exact value {values[2][tuple(sample[0][:2])]:.3f}")
