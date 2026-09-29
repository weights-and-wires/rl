"""Checks for the language-models-as-policies exercise."""

import itertools

import numpy as np

from rlcheck import almost, check, close, fail, has, shape


def _toy_table(vocab, length, seed):
    """The same logit table ToyLM(vocab, length, seed) builds."""
    return np.random.default_rng(seed).normal(size=(length, vocab + 1, vocab))


def _log_softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def _toy_logits(table, tokens):
    batch, steps = tokens.shape
    vocab = table.shape[-1]
    prev = np.concatenate([np.full((batch, 1), vocab), tokens[:, :-1]], axis=1)
    return table[np.arange(steps)[None, :], prev]


def _toy_tree(table):
    """Prefix-indexed conditional log-probs, built independently of the starter."""
    length, _, vocab = table.shape
    logp = _log_softmax(table)
    tree = []
    for t in range(length):
        arr = np.zeros((vocab,) * t + (vocab,))
        for prefix in itertools.product(range(vocab), repeat=t):
            prev = prefix[-1] if t else vocab
            arr[prefix] = logp[t, prev]
        tree.append(arr)
    return tree


def _enumerate(vocab, length):
    return np.array(list(itertools.product(range(vocab), repeat=length)), dtype=int)


def _toy_sequence_logps(table, seqs):
    logp = _log_softmax(table)
    total = np.zeros(len(seqs))
    prev = np.full(len(seqs), table.shape[-1])
    for t in range(seqs.shape[1]):
        total += logp[t, prev, seqs[:, t]]
        prev = seqs[:, t]
    return total


@check("token_logprobs reads a log-softmax off at the chosen tokens")
def test_token_logprobs(sub):
    token_logprobs = has(sub, "token_logprobs")
    logits = np.array([[[2.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 3.0, 0.0], [5.0, 5.0, 5.0]]])
    tokens = np.array([[0, 2, 1, 0]])
    got = np.asarray(token_logprobs(logits, tokens), dtype=float)
    shape(got, (1, 4), label="token_logprobs")
    # Hand values: log(e^2 / (e^2 + e + 1)), log(1/3), log(1 / (e^-2 + 1 + e^-3)), log(1/3).
    want = np.array([[-0.40760596, -1.09861229, -0.16984602, -1.09861229]])
    close(got, want, atol=1e-7, label="log-probabilities of the lesson's example")

    # Shifting every logit in a row by a constant changes nothing, even a huge one.
    shifted = np.asarray(token_logprobs(logits + 1000.0, tokens), dtype=float)
    close(shifted, want, atol=1e-7, label="logits shifted by +1000 (overflow check)")


@check("sequence_logprob sums token log-probs under the mask and ignores padding")
def test_sequence_logprob(sub):
    sequence_logprob = has(sub, "sequence_logprob")
    token_logps = np.array([
        [-0.4076, -1.0986, -0.1698, -np.inf],
        [-0.5, -0.25, np.nan, np.nan],
        [-1.0, -2.0, -3.0, -4.0],
    ])
    mask = np.array([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]])
    got = np.asarray(sequence_logprob(token_logps, mask), dtype=float)
    shape(got, (3,), label="sequence_logprob")
    if not np.all(np.isfinite(got)):
        fail(
            "a padded position holding -inf or nan leaked into the sum. Multiplying "
            "by the mask is not enough, because 0 * -inf is nan; select with np.where."
        )
    close(got, np.array([-1.676, -0.75, -10.0]), atol=1e-9, label="masked sums")

    bool_mask = mask.astype(bool)
    close(sequence_logprob(token_logps, bool_mask), got, atol=1e-12, label="boolean mask")


@check("On the toy LM, sequence probabilities over all 4^3 responses sum to 1")
def test_enumerated_sequences(sub):
    token_logprobs = has(sub, "token_logprobs")
    sequence_logprob = has(sub, "sequence_logprob")
    table = _toy_table(4, 3, seed=3)
    seqs = _enumerate(4, 3)
    logits = _toy_logits(table, seqs)
    got = sequence_logprob(token_logprobs(logits, seqs), np.ones(seqs.shape))
    shape(got, (64,), label="sequence log-probs")
    almost(np.exp(got).sum(), 1.0, tol=1e-10, label="total probability of every response")
    close(got, _toy_sequence_logps(table, seqs), atol=1e-10,
          label="log-probs against the chain rule of probability")


@check("kl_optimal_policy matches the lesson's worked example and both limits of beta")
def test_kl_optimal_policy(sub):
    kl_optimal_policy = has(sub, "kl_optimal_policy")
    ref = np.log(np.array([0.5, 0.3, 0.2]))
    r = np.array([1.0, 0.0, 2.0])
    pi = np.asarray(kl_optimal_policy(ref, r, 1.0), dtype=float)
    shape(pi, (3,), label="kl_optimal_policy")
    close(pi, np.array([0.43325, 0.09563, 0.47111]), atol=5e-5, label="beta = 1")
    close(kl_optimal_policy(ref, r, 0.5), np.array([0.24772, 0.02012, 0.73217]), atol=5e-5,
          label="beta = 0.5")

    close(kl_optimal_policy(ref, r, 1e6), np.exp(ref), atol=1e-5,
          label="a huge beta should give back pi_ref")
    close(kl_optimal_policy(ref, r, 1e-3), np.array([0.0, 0.0, 1.0]), atol=1e-6,
          label="a tiny beta should put all mass on the best response")

    # The defining property: log pi* - log pi_ref - r / beta is one constant, -log Z.
    rng = np.random.default_rng(0)
    ref = _log_softmax(rng.normal(size=50))
    r = rng.normal(size=50) * 3.0
    pi = np.asarray(kl_optimal_policy(ref, r, 0.3), dtype=float)
    gap = np.log(pi) - ref - r / 0.3
    if np.ptp(gap) > 1e-8:
        fail(f"log pi* - log pi_ref - r / beta should be constant, but it spans {np.ptp(gap):.3g}")
    log_z = np.log(np.sum(np.exp(ref + r / 0.3)))
    almost(gap[0], -log_z, tol=1e-8, label="the constant should be -log Z")


@check("kl_regularised_objective peaks at pi* with value beta log Z on the toy LM")
def test_objective(sub):
    token_logprobs = has(sub, "token_logprobs")
    sequence_logprob = has(sub, "sequence_logprob")
    kl_optimal_policy = has(sub, "kl_optimal_policy")
    objective = has(sub, "kl_regularised_objective")

    ref = np.log(np.array([0.5, 0.3, 0.2]))
    r = np.array([1.0, 0.0, 2.0])
    almost(objective(np.exp(ref), ref, r, 1.0), 0.9, tol=1e-10,
           label="J(pi_ref) = E_ref[r], with zero KL")
    almost(objective(np.array([0.0, 0.0, 1.0]), ref, r, 1.0), 2.0 - np.log(5.0), tol=1e-10,
           label="J of a point mass on the third response (it has a zero entry)")

    table = _toy_table(4, 3, seed=5)
    seqs = _enumerate(4, 3)
    ref = sequence_logprob(token_logprobs(_toy_logits(table, seqs), seqs), np.ones(seqs.shape))
    rewards = np.all(np.diff(seqs, axis=1) > 0, axis=1) + 0.25 * (seqs == 2).sum(axis=1)
    rewards = rewards.astype(float)
    beta = 0.4
    pi = kl_optimal_policy(ref, rewards, beta)
    best = objective(pi, ref, rewards, beta)
    log_z = np.log(np.sum(np.exp(ref + rewards / beta)))
    almost(best, beta * log_z, tol=1e-10, label="J(pi*) against beta log Z")

    rng = np.random.default_rng(1)
    for i in range(20):
        other = np.exp(_log_softmax(np.log(pi) + 0.3 * rng.normal(size=pi.shape)))
        value = objective(other, ref, rewards, beta)
        if value > best - 1e-9:
            fail(f"a perturbed policy (try {i}) scored {value:.6f}, not below J(pi*) = {best:.6f}")


@check("soft_token_policy rebuilds pi* token by token, and V(s_0) = beta log Z")
def test_soft_token_policy(sub):
    soft_token_policy = has(sub, "soft_token_policy")

    # The lesson's two-token example.
    tree = [np.log(np.array([0.5, 0.5])), np.log(np.array([[0.8, 0.2], [0.4, 0.6]]))]
    rewards = np.array([[0.0, 1.0], [0.0, 0.0]])
    policy, v0 = soft_token_policy(tree, rewards, 1.0)
    almost(v0, 0.15856508, tol=1e-7, label="V(s_0) in the lesson's example")
    close(np.exp(policy[0]), np.array([0.57331629, 0.42668371]), atol=1e-7, label="pi*(y_0)")
    close(np.exp(policy[1]), np.array([[0.59539032, 0.40460968], [0.4, 0.6]]), atol=1e-7,
          label="pi*(y_1 | y_0)")

    # A non-additive reward on the toy LM, so the best policy must read the whole prefix.
    vocab, length, beta = 4, 3, 0.5
    table = _toy_table(vocab, length, seed=7)
    tree = _toy_tree(table)
    seqs = _enumerate(vocab, length)
    ref = _toy_sequence_logps(table, seqs)
    r = np.all(np.diff(seqs, axis=1) > 0, axis=1).astype(float) - 0.5 * (seqs[:, 0] == seqs[:, 2])
    policy, v0 = soft_token_policy([a.copy() for a in tree], r.reshape((vocab,) * length), beta)
    if len(policy) != length:
        fail(f"expected {length} arrays of conditional log-probs, got {len(policy)}")
    for t in range(length):
        shape(policy[t], (vocab,) * (t + 1), label=f"policy[{t}]")
        close(np.exp(policy[t]).sum(axis=-1), np.ones((vocab,) * t), atol=1e-10,
              label=f"each conditional at position {t} sums to 1")
    path = sum(policy[t][tuple(seqs[:, : t + 1].T)] for t in range(length))
    z = ref + r / beta
    pi_star = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    close(np.exp(path), pi_star, atol=1e-10,
          label="product of token-level pi* against the closed-form sequence pi*")
    almost(v0, beta * np.log(np.sum(np.exp(z))), tol=1e-10, label="V(s_0) against beta log Z")


@check("best_of_n_probs gives the exact best-of-n distribution, checked by sampling")
def test_best_of_n_probs(sub):
    best_of_n_probs = has(sub, "best_of_n_probs")
    ref = np.array([0.1, 0.4, 0.2, 0.3])
    rewards = np.array([2.0, -1.0, 0.5, 1.0])
    close(best_of_n_probs(ref, rewards, 1), ref, atol=1e-12, label="n = 1 is pi_ref")
    # Sorted by reward the order is 1, 2, 3, 0 with cumulative mass 0.4, 0.6, 0.9, 1.0.
    want = np.array([1 - 0.9**2, 0.4**2, 0.6**2 - 0.4**2, 0.9**2 - 0.6**2])
    close(best_of_n_probs(ref, rewards, 2), want, atol=1e-12, label="n = 2, by hand")

    rng = np.random.default_rng(0)
    m, n, draws = 12, 5, 100_000
    ref = np.exp(_log_softmax(rng.normal(size=m)))
    rewards = rng.normal(size=m)
    p = np.asarray(best_of_n_probs(ref, rewards, n), dtype=float)
    shape(p, (m,), label="best_of_n_probs")
    almost(p.sum(), 1.0, tol=1e-10, label="total BoN probability")
    samples = rng.choice(m, size=(draws, n), p=ref)
    picked = samples[np.arange(draws), np.argmax(rewards[samples], axis=1)]
    hist = np.bincount(picked, minlength=m) / draws
    tv = 0.5 * np.abs(hist - p).sum()
    if tv > 0.01:
        fail(f"total variation between your distribution and 100,000 BoN draws is {tv:.4f}, "
             "above 0.01")


@check("best_of_n_kl never exceeds log n - (n-1)/n, and meets it only for a spread reference")
def test_best_of_n_kl(sub):
    best_of_n_kl = has(sub, "best_of_n_kl")
    rng = np.random.default_rng(2)
    for trial in range(10):
        m = int(rng.integers(3, 40))
        ref = np.exp(_log_softmax(2.0 * rng.normal(size=m)))
        rewards = rng.normal(size=m)
        for n in range(1, 65):
            kl = best_of_n_kl(ref, rewards, n)
            bound = np.log(n) - (n - 1) / n
            if kl > bound + 1e-10:
                fail(f"n = {n}: KL {kl:.6f} is above the bound {bound:.6f} (trial {trial})")
            if kl < -1e-10:
                fail(f"n = {n}: a KL divergence can't be negative, got {kl:.3g}")

    almost(best_of_n_kl(np.ones(4) / 4, np.arange(4.0), 4), 0.590310, tol=1e-6,
           label="uniform over 4 responses, n = 4")
    uniform = np.ones(10_000) / 10_000
    gap = (np.log(16) - 15 / 16) - best_of_n_kl(uniform, np.arange(10_000.0), 16)
    if not 0 <= gap < 1e-3:
        fail(f"uniform over 10,000 responses should sit within 1e-3 of the bound; gap {gap:.3g}")

    # Half the reference mass on the single best response: the bound is far too high.
    peaked = np.array([0.5] + [0.5 / 99] * 99)
    kl = best_of_n_kl(peaked, -np.arange(100.0), 16)
    almost(kl, 0.692991, tol=1e-5, label="half the mass on the best response, n = 16")
