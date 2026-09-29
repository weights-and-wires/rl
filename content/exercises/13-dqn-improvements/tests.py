"""Checks for the DQN improvements exercise."""

import numpy as np
import torch

from rlcheck import almost, check, close, equal, fail, has, shape

torch.set_num_threads(1)


def _as_np(x):
    return torch.as_tensor(x).detach().numpy().astype(float)


@check("double_dqn_targets picks with the online net and scores with the target net")
def test_double_targets(sub):
    double_dqn_targets = has(sub, "double_dqn_targets")
    online = torch.tensor([[1.0, 3.0, 2.0], [1.0, 3.0, 2.0], [0.0, -1.0, 4.0]], dtype=torch.float64)
    target = torch.tensor([[2.5, 1.5, 2.0], [2.5, 1.5, 2.0], [1.0, 7.0, -2.0]], dtype=torch.float64)
    y = double_dqn_targets(torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64), online, target,
                           torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), 0.9)
    # Row 0: online argmax is action 1, target scores it 1.5: 1 + 0.9 * 1.5 = 2.35.
    # Row 1 terminates: 1. Row 2: online picks action 2, target says -2: 0.9 * -2 = -1.8.
    close(_as_np(y), np.array([2.35, 1.0, -1.8]), atol=1e-12, label="Double DQN targets")


@check("Double DQN removes the max bias: 20,000 noisy estimates of ten zero-valued actions")
def test_double_bias(sub):
    double_dqn_targets = has(sub, "double_dqn_targets")
    torch.manual_seed(0)
    n, m = 20000, 10
    online = torch.randn(n, m, dtype=torch.float64)
    target = torch.randn(n, m, dtype=torch.float64)
    zeros = torch.zeros(n, dtype=torch.float64)
    y = _as_np(double_dqn_targets(zeros, online, target, zeros, 1.0))
    plain = target.max(dim=1).values.numpy()
    # E[max of 10 standard normals] = 1.53875 (numerical integration).
    se = plain.std() / np.sqrt(n)
    if abs(plain.mean() - 1.53875) > 4 * se:
        fail("internal: the plain max estimate is off; this check is broken")
    se_double = y.std() / np.sqrt(n)
    if abs(y.mean()) > 4 * se_double:
        fail(f"with independent online and target noise the Double DQN target should average 0 "
             f"within {4 * se_double:.3f}; it averaged {y.mean():.3f} (plain DQN: {plain.mean():.3f})")


@check("dueling_combine gives Q = V + A - mean(A): hand example, mean(Q) = V, and shift invariance")
def test_dueling(sub):
    dueling_combine = has(sub, "dueling_combine")
    q = dueling_combine(torch.tensor([2.0], dtype=torch.float64),
                        torch.tensor([[1.0, -1.0, 3.0]], dtype=torch.float64))
    close(_as_np(q), np.array([[2.0, 0.0, 4.0]]), atol=1e-12, label="V = 2, A = [1, -1, 3]")
    torch.manual_seed(1)
    v = torch.randn(5, 1, dtype=torch.float64)
    a = torch.randn(5, 4, dtype=torch.float64)
    q = dueling_combine(v, a)
    shape(q, (5, 4), label="dueling_combine output for V (5, 1), A (5, 4)")
    close(_as_np(q).mean(axis=1), v.numpy()[:, 0], atol=1e-12, label="mean over actions of Q")
    shifted = dueling_combine(v, a + torch.randn(5, 1, dtype=torch.float64))
    close(_as_np(shifted), _as_np(q), atol=1e-12, label="Q after adding a per-state constant to A")
    q_flat = dueling_combine(v[:, 0], a)
    close(_as_np(q_flat), _as_np(q), atol=1e-12, label="V given with shape (5,) instead of (5, 1)")


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


@check("n_step_targets matches the hand example, reduces to one step at n = 1, and to the return at large n")
def test_n_step(sub):
    n_step_targets = has(sub, "n_step_targets")
    got = np.asarray(n_step_targets([1.0, 0.0, 2.0, 1.0], [False, False, False, True],
                                    [5.0, 5.0, 5.0, 5.0], 0.9, 3), dtype=float)
    close(got, [6.265, 2.61, 2.9, 1.0], atol=1e-12, label="3-step targets, episode ends after step 3")

    rng = np.random.default_rng(0)
    T, gamma = 12, 0.8
    r = rng.normal(size=T)
    boot = rng.normal(size=T)
    term = np.zeros(T, dtype=bool)
    term[[4, 11]] = True
    one = np.asarray(n_step_targets(r, term, boot, gamma, 1), dtype=float)
    close(one, r + gamma * (~term) * boot, atol=1e-12, label="n = 1 against the one-step target")

    big = np.asarray(n_step_targets(r, term, boot, gamma, 50), dtype=float)
    want = np.zeros(T)
    for t in range(T):
        end = 4 if t <= 4 else 11
        want[t] = sum(gamma ** (k - t) * r[k] for k in range(t, end + 1))
    close(big, want, atol=1e-12, label="n = 50 against the discounted return to the episode's end")

    # n = 3 with an unfinished trajectory: bootstrap from the last value available.
    term = np.zeros(T, dtype=bool)
    term[4] = True
    got = np.asarray(n_step_targets(r, term, boot, gamma, 3), dtype=float)
    want = np.zeros(T)
    for t in range(T):
        total = 0.0
        for k in range(3):
            j = t + k
            total += gamma**k * r[j]
            if term[j]:
                break
            if k == 2 or j == T - 1:
                total += gamma ** (k + 1) * boot[j]
                break
        want[t] = total
    close(got, want, atol=1e-12, label="n = 3 with an episode boundary and an unfinished tail")
