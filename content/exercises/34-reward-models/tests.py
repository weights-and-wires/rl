"""Checks for the reward-models exercise."""

import itertools

import numpy as np
import torch

from rlcheck import almost, check, close, fail, finite_difference, has, shape

torch.set_num_threads(1)


def _log_softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def _enumerate(vocab, length):
    return np.array(list(itertools.product(range(vocab), repeat=length)), dtype=int)


def _toy_table(vocab, length, seed, scale=1.0):
    return scale * np.random.default_rng(seed).normal(size=(length, vocab + 1, vocab))


def _sequence_probs(table, seqs):
    logp = _log_softmax(table)
    total = np.zeros(len(seqs))
    prev = np.full(len(seqs), table.shape[-1])
    for t in range(seqs.shape[1]):
        total += logp[t, prev, seqs[:, t]]
        prev = seqs[:, t]
    return np.exp(total)


def _toy_tree(table):
    length, _, vocab = table.shape
    logp = _log_softmax(table)
    tree = []
    for t in range(length):
        arr = np.zeros((vocab,) * t + (vocab,))
        for prefix in itertools.product(range(vocab), repeat=t):
            arr[prefix] = logp[t, prefix[-1] if t else vocab]
        tree.append(arr)
    return tree


def _features(seqs, vocab):
    batch, steps = seqs.shape
    feats = np.zeros((batch, steps * vocab))
    feats[np.arange(batch)[:, None], np.arange(steps) * vocab + seqs] = 1.0
    return feats


def _sample_pairs(rng, probs, reward, n):
    a = rng.choice(len(probs), n, p=probs)
    b = rng.choice(len(probs), n, p=probs)
    a_wins = rng.random(n) < 1.0 / (1.0 + np.exp(-(reward[a] - reward[b])))
    return np.where(a_wins, a, b), np.where(a_wins, b, a)


def _loss_value(loss_fn, rc, rr, weights=None):
    rc = torch.as_tensor(rc, dtype=torch.float64)
    rr = torch.as_tensor(rr, dtype=torch.float64)
    w = None if weights is None else torch.as_tensor(weights, dtype=torch.float64)
    value = loss_fn(rc, rr) if w is None else loss_fn(rc, rr, w)
    return float(value)


@check("bradley_terry_loss gives log 2 for ties, 0.3133 for a gap of 1, and stays finite")
def test_bt_loss_values(sub):
    loss = has(sub, "bradley_terry_loss")
    almost(_loss_value(loss, np.zeros(5), np.zeros(5)), np.log(2.0), tol=1e-10, label="tied rewards")
    almost(_loss_value(loss, [2.0], [1.0]), 0.31326169, tol=1e-7, label="chosen 2, rejected 1")
    almost(_loss_value(loss, [0.0, 0.0], [800.0, -800.0]), 400.0, tol=1e-6,
           label="gaps of -800 and +800 (log(sigmoid) underflows to -inf here)")
    base = _loss_value(loss, [0.3, -1.0, 2.0], [0.1, 0.5, -0.7])
    almost(_loss_value(loss, [100.3, 99.0, 102.0], [100.1, 100.5, 99.3]), base, tol=1e-10,
           label="adding 100 to every reward must change nothing")
    almost(_loss_value(loss, [2.0, 0.0], [1.0, 1.0], weights=[3.0, 1.0]),
           (3 * 0.31326169 + 1.31326169) / 4, tol=1e-7, label="weighted mean with weights [3, 1]")


@check("The BT gradient is -(1 - sigmoid(gap)) / n, checked by finite differences")
def test_bt_gradient(sub):
    loss = has(sub, "bradley_terry_loss")
    rng = np.random.default_rng(0)
    n = 6
    x0 = rng.normal(size=2 * n) * 2.0

    def f(x):
        return _loss_value(loss, x[:n], x[n:])

    fd = finite_difference(f, x0.copy())
    gap = x0[:n] - x0[n:]
    want = -(1.0 - 1.0 / (1.0 + np.exp(-gap))) / n
    close(fd[:n], want, atol=1e-6, label="d loss / d reward_chosen")
    close(fd[n:], -want, atol=1e-6, label="d loss / d reward_rejected")


@check("fit_reward_model on every weighted pair recovers the true reward up to a constant")
def test_fit_population(sub):
    fit = has(sub, "fit_reward_model")
    vocab, length = 4, 3
    seqs = _enumerate(vocab, length)
    feats = _features(seqs, vocab)
    for seed in (0, 1):
        rng = np.random.default_rng(seed)
        probs = _sequence_probs(_toy_table(vocab, length, seed), seqs)
        true = feats @ rng.normal(size=feats.shape[1])
        a, b = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
        a, b = a.ravel(), b.ravel()
        # Each ordered pair (a over b) weighted by how often it would be observed.
        weights = probs[a] * probs[b] / (1.0 + np.exp(-(true[a] - true[b])))
        phi = np.asarray(fit(feats[a], feats[b], steps=600, lr=0.05, weights=weights), dtype=float)
        shape(phi, (feats.shape[1],), label="phi")
        err = feats @ phi - true
        err -= err.mean()
        if np.abs(err).max() > 1e-3:
            fail(f"seed {seed}: with infinite data the fit should equal the true reward plus a "
                 f"constant, but after removing the mean the largest error is {np.abs(err).max():.3g}")


@check("fit_reward_model on 4000 sampled preferences tracks the true reward")
def test_fit_samples(sub):
    fit = has(sub, "fit_reward_model")
    vocab, length = 4, 3
    seqs = _enumerate(vocab, length)
    feats = _features(seqs, vocab)
    rng = np.random.default_rng(0)
    probs = _sequence_probs(_toy_table(vocab, length, 0), seqs)
    true = feats @ rng.normal(size=feats.shape[1])
    chosen, rejected = _sample_pairs(rng, probs, true, 4000)
    phi = np.asarray(fit(feats[chosen], feats[rejected], steps=300, lr=0.05), dtype=float)
    fitted = feats @ phi
    corr = np.corrcoef(fitted, true)[0, 1]
    if corr < 0.98:
        fail(f"correlation between fitted and true rewards is {corr:.4f}, expected above 0.98")
    err = fitted - true
    err -= probs @ err
    rms = np.sqrt(probs @ err**2)
    if rms > 0.2:
        fail(f"reference-weighted RMS error after removing a constant is {rms:.3f}, above 0.2")

    held_c, held_r = _sample_pairs(np.random.default_rng(1), probs, true, 20000)
    fitted_loss = np.mean(np.logaddexp(0.0, -(fitted[held_c] - fitted[held_r])))
    floor = np.mean(np.logaddexp(0.0, -(true[held_c] - true[held_r])))
    if fitted_loss - floor > 0.01:
        fail(f"held-out loss {fitted_loss:.4f} is more than 0.01 above the true reward's "
             f"{floor:.4f}")


@check("ensemble_reward computes the mean, the worst case, and mean minus lam * variance")
def test_ensemble_values(sub):
    ensemble_reward = has(sub, "ensemble_reward")
    scores = np.array([[1.0, 2.0, 0.5], [3.0, 2.0, 0.5], [2.0, 2.0, 2.0]])
    close(ensemble_reward(scores, "mean"), [2.0, 2.0, 1.0], atol=1e-12, label="mean")
    close(ensemble_reward(scores, "min"), [1.0, 2.0, 0.5], atol=1e-12, label="min (WCO)")
    close(ensemble_reward(scores, "uwo", lam=0.5), [2.0 - 1 / 3, 2.0, 1.0 - 0.25], atol=1e-12,
          label="UWO with lam = 0.5 (variance over K, not K - 1)")


def _fit_many(chosen, rejected, n_items, steps=200, lr=0.05, weight_decay=1e-4):
    """Fit many tabular BT models at once: chosen and rejected are (M, n) index arrays."""
    c = torch.as_tensor(chosen)
    r = torch.as_tensor(rejected)
    w = torch.zeros(c.shape[0], n_items, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        gap = torch.gather(w, 1, c) - torch.gather(w, 1, r)
        loss = torch.nn.functional.softplus(-gap).mean(dim=1).sum() + weight_decay * (w**2).sum()
        loss.backward()
        opt.step()
    return w.detach().numpy()


@check("A bootstrap ensemble disagrees most where the proxy looks best, and WCO overrates less")
def test_ensemble_overoptimisation(sub):
    ensemble_reward = has(sub, "ensemble_reward")
    vocab, length, prompts, pairs, members = 4, 3, 40, 300, 5
    seqs = _enumerate(vocab, length)
    n_items = len(seqs)
    rng = np.random.default_rng(1)
    probs, gold, chosen, rejected = [], [], [], []
    for _ in range(prompts):
        p = _sequence_probs(3.0 * rng.normal(size=(length, vocab + 1, vocab)), seqs)
        g = rng.normal(size=n_items)
        c, r = _sample_pairs(rng, p, g, pairs)
        probs.append(p)
        gold.append(g)
        chosen.append(c)
        rejected.append(r)
        for _ in range(members):
            idx = rng.integers(0, pairs, pairs)
            chosen.append(c[idx])
            rejected.append(r[idx])
    fits = _fit_many(np.array(chosen), np.array(rejected), n_items).reshape(prompts, members + 1, n_items)
    single_over, wco_over, top_sd, typical_sd = [], [], [], []
    for i in range(prompts):
        p, g = probs[i], gold[i] - probs[i] @ gold[i]
        single, ens = fits[i, 0], fits[i, 1:]
        wco = np.asarray(ensemble_reward(ens, "min"), dtype=float)
        mean = np.asarray(ensemble_reward(ens, "mean"), dtype=float)
        shape(wco, (n_items,), label="ensemble_reward output")
        for proxy, store in ((single, single_over), (wco, wco_over)):
            j = int(np.argmax(proxy))
            store.append((proxy[j] - p @ proxy) - g[j])
        sd = ens.std(axis=0)
        top_sd.append(sd[np.argsort(-mean)[:3]].mean())
        typical_sd.append(p @ sd)
    if np.mean(top_sd) < 1.5 * np.mean(typical_sd):
        fail(f"members' disagreement on the ensemble's top 3 responses ({np.mean(top_sd):.3f}) "
             f"should be well above the typical disagreement ({np.mean(typical_sd):.3f})")
    s, w = np.mean(single_over), np.mean(wco_over)
    if not w < s - 0.1:
        fail(f"at its own favourite response, the single model overrates the gold reward by "
             f"{s:.3f} on average; WCO should overrate by at least 0.1 less, got {w:.3f}")


@check("prefix_values matches the lesson's example and the policy's exact expected reward")
def test_prefix_values(sub):
    prefix_values = has(sub, "prefix_values")
    tree = [np.log(np.array([0.5, 0.5])), np.log(np.array([[0.8, 0.2], [0.4, 0.6]]))]
    rewards = np.array([[0.0, 1.0], [1.0, 0.0]])
    values = prefix_values(tree, rewards)
    if len(values) != 3:
        fail(f"expected T + 1 = 3 arrays of values, got {len(values)}")
    almost(values[0], 0.3, tol=1e-12, label="V(empty prefix)")
    close(values[1], [0.2, 0.4], atol=1e-12, label="V(y_0)")
    close(values[2], rewards, atol=1e-12, label="the last entry is the outcome reward")

    table = _toy_table(4, 4, seed=2)
    seqs = _enumerate(4, 4)
    outcome = (seqs.sum(axis=1) == 6).astype(float)
    values = prefix_values(_toy_tree(table), outcome.reshape((4,) * 4))
    almost(values[0], _sequence_probs(table, seqs) @ outcome, tol=1e-12,
           label="V(empty prefix) against the enumerated success probability")
    for t in range(4):
        shape(values[t], (4,) * t, label=f"values[{t}]")


@check("math_shepherd_labels: the soft label averages to V, the hard one to 1 - (1 - V)^n")
def test_math_shepherd(sub):
    prefix_values = has(sub, "prefix_values")
    labels = has(sub, "math_shepherd_labels")
    table = _toy_table(4, 4, seed=2)
    tree = _toy_tree(table)
    outcome = (_enumerate(4, 4).sum(axis=1) == 6).astype(float).reshape((4,) * 4)
    values = prefix_values(tree, outcome)
    rng = np.random.default_rng(0)
    repeats, n = 600, 4
    for prefix in ((1,), (2, 1)):
        v = float(values[len(prefix)][prefix])
        draws = np.array([labels(rng, tree, outcome, prefix, n) for _ in range(repeats)])
        shape(draws, (repeats, 2), label="(soft, hard) pairs")
        for col, want, name in ((0, v, "soft"), (1, 1 - (1 - v) ** n, "hard")):
            mean = draws[:, col].mean()
            se = max(draws[:, col].std(), 1e-3) / np.sqrt(repeats)
            if abs(mean - want) > 4 * se:
                fail(f"prefix {prefix}: the mean {name} label over {repeats} repeats is "
                     f"{mean:.4f}, expected {want:.4f} within 4 standard errors ({4 * se:.4f})")


@check("process_rewards telescopes to r(y) - V(empty prefix) along every response")
def test_process_rewards(sub):
    prefix_values = has(sub, "prefix_values")
    process_rewards = has(sub, "process_rewards")
    tree = [np.log(np.array([0.5, 0.5])), np.log(np.array([[0.8, 0.2], [0.4, 0.6]]))]
    rewards = np.array([[0.0, 1.0], [1.0, 0.0]])
    got = np.asarray(process_rewards(np.array([[0, 1], [1, 0], [1, 1]]), prefix_values(tree, rewards)))
    close(got, [[-0.1, 0.8], [0.1, 0.6], [0.1, -0.4]], atol=1e-12, label="the lesson's example")

    table = _toy_table(4, 4, seed=2)
    seqs = _enumerate(4, 4)
    outcome = (seqs.sum(axis=1) == 6).astype(float)
    values = prefix_values(_toy_tree(table), outcome.reshape((4,) * 4))
    steps = np.asarray(process_rewards(seqs, values), dtype=float)
    shape(steps, seqs.shape, label="process_rewards")
    close(steps.sum(axis=1), outcome - float(values[0]), atol=1e-12,
          label="row sums against r(y) - V(empty prefix)")


@check("Process rewards give the same policy gradient as the outcome reward, with less variance")
def test_gradient_variance(sub):
    prefix_values = has(sub, "prefix_values")
    process_rewards = has(sub, "process_rewards")
    vocab, length = 4, 4
    table = _toy_table(vocab, length, seed=2)
    seqs = _enumerate(vocab, length)
    outcome = (seqs.sum(axis=1) == 6).astype(float)
    probs = _sequence_probs(table, seqs)
    values = prefix_values(_toy_tree(table), outcome.reshape((vocab,) * length))
    steps = np.asarray(process_rewards(seqs, values), dtype=float)
    to_go = np.cumsum(steps[:, ::-1], axis=1)[:, ::-1]   # r(y) - V(y_<t) at each t

    # Score function of the tabular policy: d log pi(y_t | t, prev) / d table[t, prev, :].
    pi = np.exp(_log_softmax(table))
    prev = np.concatenate([np.full((len(seqs), 1), vocab), seqs[:, :-1]], axis=1)
    score = np.zeros((len(seqs), length) + table.shape)
    for t in range(length):
        rows = np.arange(len(seqs))
        score[rows, t, t, prev[:, t]] = -pi[t, prev[:, t]]
        score[rows, t, t, prev[:, t], seqs[:, t]] += 1.0
    outcome_g = (outcome[:, None, None, None, None] * score).sum(axis=1)
    process_g = (to_go[:, :, None, None, None] * score).sum(axis=1)

    def success(flat):
        return _sequence_probs(flat.reshape(table.shape), seqs) @ outcome

    true_grad = finite_difference(success, table.ravel().copy()).reshape(table.shape)
    for name, g in (("outcome", outcome_g), ("process", process_g)):
        mean = np.tensordot(probs, g, axes=1)
        close(mean, true_grad, atol=1e-7, label=f"expected {name}-reward gradient")
    def total_var(g):
        mean = np.tensordot(probs, g, axes=1)
        return float(np.tensordot(probs, ((g - mean) ** 2).reshape(len(seqs), -1).sum(axis=1), axes=1))
    v_out, v_proc = total_var(outcome_g), total_var(process_g)
    if not v_proc < 0.9 * v_out:
        fail(f"process-reward gradient variance {v_proc:.4f} should be well below the outcome "
             f"reward's {v_out:.4f}")
