"""Checks for the Double and dueling DQN exercise."""

import numpy as np
import torch

from rlcheck import almost, check, close, fail, has, shape

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


@check("dueling_combine gives Q = V + A - mean(A) on the hand example")
def test_dueling_hand(sub):
    dueling_combine = has(sub, "dueling_combine")
    q = dueling_combine(torch.tensor([2.0], dtype=torch.float64),
                        torch.tensor([[1.0, -1.0, 3.0]], dtype=torch.float64))
    close(_as_np(q), np.array([[2.0, 0.0, 4.0]]), atol=1e-12, label="V = 2, A = [1, -1, 3]")


@check("dueling_combine's Q averages to V and ignores a shift in the advantages")
def test_dueling_invariance(sub):
    dueling_combine = has(sub, "dueling_combine")
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


@check("n_step_targets matches the hand example, where the episode ends before the bootstrap")
def test_n_step_hand(sub):
    n_step_targets = has(sub, "n_step_targets")
    got = np.asarray(n_step_targets([1.0, 0.0, 2.0, 1.0], [False, False, False, True],
                                    [5.0, 5.0, 5.0, 5.0], 0.9, 3), dtype=float)
    close(got, [6.265, 2.61, 2.9, 1.0], atol=1e-12, label="3-step targets, episode ends after step 3")


@check("n_step_targets reduces to one step at n = 1, and to the discounted return at large n")
def test_n_step_limits(sub):
    n_step_targets = has(sub, "n_step_targets")
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
