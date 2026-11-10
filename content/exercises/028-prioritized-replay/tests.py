"""Checks for the prioritized replay exercise."""

import numpy as np

from rlcheck import almost, check, close, equal, fail, has


@check("SumTree's total equals the sum of its leaves after arbitrary updates and overwrites")
def test_sumtree_total(sub):
    SumTree = has(sub, "SumTree")
    rng = np.random.default_rng(0)
    for capacity in (1, 5, 8, 13):
        tree = SumTree(capacity)
        leaves = np.zeros(capacity)
        for _ in range(200):
            i = int(rng.integers(capacity))
            p = float(rng.uniform(0, 3))
            tree.update(i, p)
            leaves[i] = p
        almost(tree.total(), leaves.sum(), 1e-9, label=f"total() with capacity {capacity}")


@check("SumTree.find maps exactly p_i units of mass to leaf i, so sampling follows the priorities")
def test_sumtree_find(sub):
    SumTree = has(sub, "SumTree")
    for capacity, priorities in ((6, [3, 1, 0, 4, 2, 5]), (8, [1, 2, 3, 4, 0, 0, 7, 1]), (5, [2, 2, 1, 0, 6])):
        tree = SumTree(capacity)
        for i, p in enumerate(priorities):
            tree.update(i, float(p))
        total = int(sum(priorities))
        # Integer priorities put every leaf boundary on an integer, so the
        # midpoint of each unit interval lies inside exactly one leaf.
        found = [int(tree.find(k + 0.5)) for k in range(total)]
        counts = np.bincount(found, minlength=capacity)[:capacity]
        equal(counts, np.array(priorities), label=f"mass units per leaf for priorities {priorities}")


@check("per_probabilities and importance_weights match the formulas and the worked example")
def test_per_formulas(sub):
    per_probabilities = has(sub, "per_probabilities")
    importance_weights = has(sub, "importance_weights")
    p = np.array([2.0, 1.0, 0.5, 0.5])
    probs = np.asarray(per_probabilities(p, 0.6), dtype=float)
    want = p**0.6 / np.sum(p**0.6)
    close(probs, want, atol=1e-12, label="P(i) = p_i^alpha / sum p^alpha")
    close(probs, [0.395209, 0.260741, 0.172025, 0.172025], atol=1e-6, label="worked example P(i)")
    w = np.asarray(importance_weights(probs, 0.4), dtype=float)
    close(w, [0.716978, 0.846745, 1.0, 1.0], atol=1e-6, label="worked example weights")
    close(np.asarray(per_probabilities(p, 0.0)), np.full(4, 0.25), atol=1e-12, label="alpha = 0 is uniform")


@check("importance_weights with beta = 1 undo the prioritisation exactly")
def test_per_unbiased(sub):
    per_probabilities = has(sub, "per_probabilities")
    importance_weights = has(sub, "importance_weights")

    # With beta = 1 the weights undo the prioritisation exactly.
    rng = np.random.default_rng(0)
    probs = np.asarray(per_probabilities(rng.uniform(0.1, 5, size=20), 0.7), dtype=float)
    w = np.asarray(importance_weights(probs, 1.0), dtype=float)
    almost(w.max(), 1.0, 1e-12, label="largest weight")
    f = rng.normal(size=20)
    almost(np.sum(probs * w * f) / np.sum(probs * w), f.mean(), 1e-12,
           label="weighted mean of f under P, against the uniform mean of f")


@check("PrioritizedReplay: new items get the max priority, and the tree holds (|delta| + eps)^alpha")
def test_per_buffer_priorities(sub):
    PrioritizedReplay = has(sub, "PrioritizedReplay")
    alpha, eps = 0.6, 0.01
    replay = PrioritizedReplay(capacity=8, alpha=alpha, eps=eps, seed=0)
    for _ in range(5):
        replay.add()
    equal(len(replay), 5, label="len after 5 adds")
    almost(replay.tree.total(), 5.0, 1e-12, label="total before any update (priority 1 each)")
    errors = np.array([0.5, -2.0, 0.0, 3.0, 1.0])
    replay.update_priorities(np.arange(5), errors)
    scaled = (np.abs(errors) + eps) ** alpha
    almost(replay.tree.total(), scaled.sum(), 1e-9, label="total after update_priorities")
    index = replay.add()
    equal(int(index), 5, label="index returned by the sixth add")
    almost(replay.tree.total(), scaled.sum() + (3.0 + eps) ** alpha, 1e-9,
           label="total after adding an item, which should get the max priority 3.01")


@check("PrioritizedReplay samples in proportion to p^alpha and returns the paper's weights")
def test_per_buffer_sampling(sub):
    PrioritizedReplay = has(sub, "PrioritizedReplay")
    alpha, beta, eps = 0.6, 0.4, 0.01
    replay = PrioritizedReplay(capacity=6, alpha=alpha, eps=eps, seed=0)
    for _ in range(6):
        replay.add()
    errors = np.array([2.0, 1.0, 0.5, 0.5, 4.0, 0.0])
    replay.update_priorities(np.arange(6), errors)
    scaled = (np.abs(errors) + eps) ** alpha
    probs = scaled / scaled.sum()
    counts = np.zeros(6)
    for _ in range(2000):
        idx, w = replay.sample(8, beta)
        idx = np.asarray(idx).astype(int)
        counts += np.bincount(idx, minlength=6)[:6]
        want_w = (6 * probs[idx]) ** (-beta) / (6 * probs.min()) ** (-beta)
        if not np.allclose(np.asarray(w, dtype=float), want_w, atol=1e-9):
            fail(f"weights for indices {idx.tolist()} should be (N P(i))^-beta / max_j (N P(j))^-beta "
                 f"= {np.round(want_w, 4).tolist()}; got {np.round(np.asarray(w, dtype=float), 4).tolist()}")
    expected = probs * counts.sum()
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    # Five degrees of freedom; stratified sampling only lowers the statistic.
    if chi2 > 30:
        fail(f"sampled frequencies {np.round(counts / counts.sum(), 4).tolist()} should match "
             f"P(i) = {np.round(probs, 4).tolist()}; chi-squared {chi2:.1f} against a limit of 30")
