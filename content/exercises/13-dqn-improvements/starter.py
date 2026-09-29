"""Improving DQN: Double DQN, dueling heads, prioritised replay, and n-step targets.

Fill in the functions and classes below, in order. Each one is a drop-in
change to the DQN you built in the previous lesson: a different target, a
different output head, a different way to sample the buffer, and a longer
reward sum.
"""

import numpy as np
import torch

torch.set_num_threads(1)


def double_dqn_targets(rewards, next_q_online, next_q_target, terminated, gamma: float) -> torch.Tensor:
    """The Double DQN targets.

    Args:
        rewards: r_t, shape (B,).
        next_q_online: the online network's Q-values at s_{t+1}, shape (B, A).
        next_q_target: the target network's Q-values at s_{t+1}, shape (B, A).
        terminated: 1.0 where s_{t+1} is terminal, else 0.0, shape (B,).
        gamma: discount.

    Returns:
        y = r + gamma * (1 - terminated) * next_q_target[a*], where a* is the
        argmax of next_q_online in each row. Shape (B,), no gradient.
    """
    raise NotImplementedError("double_dqn_targets")


def dueling_combine(value: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    """Combine a dueling network's two streams into Q-values.

    Args:
        value: V(s), shape (B,) or (B, 1).
        advantages: A(s, a), shape (B, A).

    Returns:
        Q = V + A - mean over actions of A, shape (B, A).
    """
    raise NotImplementedError("dueling_combine")


class SumTree:
    """A binary tree whose leaves hold priorities and whose inner nodes hold sums.

    Args:
        capacity: the number of leaves. Leaf indices run from 0 to
            capacity - 1. Any positive capacity must work.

    A common layout stores the tree in one array of length 2 * capacity:
    node 1 is the root, node i has children 2i and 2i + 1, and leaf j is
    node capacity + j.
    """

    def __init__(self, capacity: int) -> None:
        raise NotImplementedError("SumTree.__init__")

    def update(self, index: int, priority: float) -> None:
        """Set leaf `index` to `priority` and fix every sum above it. O(log capacity)."""
        raise NotImplementedError("SumTree.update")

    def total(self) -> float:
        """The sum of all leaves, read from the root."""
        raise NotImplementedError("SumTree.total")

    def leaf(self, index: int) -> float:
        """The priority stored at leaf `index`."""
        raise NotImplementedError("SumTree.leaf")

    def find(self, mass: float) -> int:
        """The leaf where the running sum of priorities first exceeds `mass`.

        Walk down from the root. At each node, go left if mass is below the
        left child's sum. Otherwise subtract the left child's sum and go
        right. For mass drawn uniformly from [0, total()), leaf i comes up
        with probability priority_i / total().
        """
        raise NotImplementedError("SumTree.find")


def per_probabilities(priorities: np.ndarray, alpha: float) -> np.ndarray:
    """Prioritised replay's sampling distribution, P(i) = p_i^alpha / sum_k p_k^alpha.

    Args:
        priorities: p_i for every stored item, shape (N,), all positive.
        alpha: how strongly to prioritise. 0 is uniform.

    Returns:
        P, shape (N,), summing to 1.
    """
    raise NotImplementedError("per_probabilities")


def importance_weights(probs: np.ndarray, beta: float) -> np.ndarray:
    """Importance-sampling weights w_i = (N P(i))^-beta / max_j (N P(j))^-beta.

    Args:
        probs: P(i) for every stored item, shape (N,).
        beta: how much of the bias to correct. 1 corrects it fully.

    Returns:
        w, shape (N,), with largest value exactly 1.
    """
    raise NotImplementedError("importance_weights")


class PrioritizedReplay:
    """Proportional prioritised replay over item indices 0, ..., capacity - 1.

    It stores only priorities. Keep the transitions themselves in parallel
    arrays indexed the same way, as in the previous lesson's ReplayBuffer.

    Args:
        capacity: the most items it holds; later adds overwrite the oldest.
        alpha: the prioritisation exponent.
        eps: added to |TD error| so that no priority is zero.
        seed: seeds the buffer's own numpy generator.

    It must expose `self.tree`, a SumTree whose leaf i holds p_i ** alpha.
    """

    def __init__(self, capacity: int, alpha: float = 0.6, eps: float = 0.01, seed: int = 0) -> None:
        raise NotImplementedError("PrioritizedReplay.__init__")

    def __len__(self) -> int:
        """The number of items stored, at most capacity."""
        raise NotImplementedError("PrioritizedReplay.__len__")

    def add(self) -> int:
        """Claim the next slot and return its index.

        The item gets the largest raw priority seen so far, starting at 1.0,
        so the tree leaf becomes max_priority ** alpha.
        """
        raise NotImplementedError("PrioritizedReplay.add")

    def sample(self, batch_size: int, beta: float):
        """Draw a batch by stratified sampling, with importance weights.

        Split [0, total) into batch_size equal slices, draw one mass
        uniformly inside each slice, and find its leaf.

        Returns:
            indices: shape (batch_size,), ints.
            weights: shape (batch_size,), (N P(i))^-beta divided by the
                largest such weight over all N stored items, where N = len(self).
        """
        raise NotImplementedError("PrioritizedReplay.sample")

    def update_priorities(self, indices, td_errors) -> None:
        """Set p_i = |delta_i| + eps for each index, and track the largest p_i seen."""
        raise NotImplementedError("PrioritizedReplay.update_priorities")


def n_step_targets(rewards, terminated, bootstrap, gamma: float, n: int) -> np.ndarray:
    """n-step Q-learning targets along one stretch of experience.

    Args:
        rewards: r_t for t = 0, ..., T - 1, shape (T,).
        terminated: shape (T,), true where the step from s_t ended the episode.
        bootstrap: shape (T,). bootstrap[t] is max_a Q_target(s_{t+1}, a), the
            value of the state reached after step t.
        gamma: discount.
        n: how many rewards to sum before bootstrapping.

    Returns:
        y, shape (T,). y[t] sums gamma^k r_{t+k} for k = 0, ..., m - 1 and
        adds gamma^m bootstrap[t + m - 1], where m = min(n, T - t). If the
        episode terminates at step t + j with j < m, the sum stops there and
        nothing is bootstrapped.
    """
    raise NotImplementedError("n_step_targets")


if __name__ == "__main__":
    # 1. Double DQN against DQN on the worked example.
    q_online = torch.tensor([[1.0, 3.0, 2.0]])
    q_target = torch.tensor([[2.5, 1.5, 2.0]])
    y = double_dqn_targets(torch.tensor([1.0]), q_online, q_target, torch.tensor([0.0]), 0.9)
    print(f"Double DQN target {y.item():.2f}; plain DQN would give {1 + 0.9 * 2.5:.2f}")

    # 2. A dueling head's Q-values average to V.
    q = dueling_combine(torch.tensor([2.0]), torch.tensor([[1.0, -1.0, 3.0]]))
    print(f"dueling Q = {q.tolist()[0]}, mean {q.mean().item():.1f}")

    # 3. Prioritised replay: probabilities and weights.
    p = np.array([2.0, 1.0, 0.5, 0.5])
    probs = per_probabilities(p, 0.6)
    print(f"P(i) = {np.round(probs, 4)}, w = {np.round(importance_weights(probs, 0.4), 4)}")
    replay = PrioritizedReplay(capacity=4, alpha=0.6, eps=0.0, seed=0)
    for _ in range(4):
        replay.add()
    replay.update_priorities(np.arange(4), p)
    counts = np.zeros(4)
    for _ in range(2500):
        counts += np.bincount(replay.sample(4, 0.4)[0], minlength=4)
    print(f"sampled frequencies {np.round(counts / counts.sum(), 4)}")

    # 4. n-step targets on a short episode.
    print("3-step targets:", n_step_targets([1.0, 0.0, 2.0, 1.0], [False, False, False, True],
                                             [5.0, 5.0, 5.0, 5.0], 0.9, 3))
