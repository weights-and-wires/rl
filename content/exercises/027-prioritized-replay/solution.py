"""Reference solution for the prioritized replay exercise."""

import numpy as np


class SumTree:
    """A binary tree whose leaves hold priorities and whose nodes hold sums.

    The tree lives in one array of length 2 * capacity. Node 1 is the root,
    node i has children 2i and 2i + 1, and leaf j sits at node capacity + j.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.nodes = np.zeros(2 * capacity)

    def update(self, index: int, priority: float) -> None:
        node = self.capacity + index
        self.nodes[node] = priority
        node //= 2
        while node >= 1:
            self.nodes[node] = self.nodes[2 * node] + self.nodes[2 * node + 1]
            node //= 2

    def total(self) -> float:
        return float(self.nodes[1])

    def leaf(self, index: int) -> float:
        return float(self.nodes[self.capacity + index])

    def find(self, mass: float) -> int:
        node = 1
        while node < self.capacity:
            left = 2 * node
            # Rounding can leave mass a hair above the total; never step into an empty subtree.
            if mass < self.nodes[left] or self.nodes[left + 1] <= 0.0:
                node = left
            else:
                mass -= self.nodes[left]
                node = left + 1
        return node - self.capacity


def per_probabilities(priorities: np.ndarray, alpha: float) -> np.ndarray:
    scaled = np.asarray(priorities, dtype=float) ** alpha
    return scaled / scaled.sum()


def importance_weights(probs: np.ndarray, beta: float) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    raw = (len(probs) * probs) ** (-beta)
    return raw / raw.max()


class PrioritizedReplay:
    """Proportional prioritised replay over item indices 0, ..., capacity - 1."""

    def __init__(self, capacity: int, alpha: float = 0.6, eps: float = 0.01, seed: int = 0) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.tree = SumTree(capacity)
        self.scaled = np.zeros(capacity)  # p_i ** alpha, mirrored for the minimum
        self.max_priority = 1.0
        self.next = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def _set(self, index: int, priority: float) -> None:
        scaled = priority**self.alpha
        self.scaled[index] = scaled
        self.tree.update(index, scaled)

    def add(self) -> int:
        index = self.next
        # A new transition gets the largest priority so far, so it's replayed soon.
        self._set(index, self.max_priority)
        self.next = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return index

    def sample(self, batch_size: int, beta: float):
        total = self.tree.total()
        segment = total / batch_size
        # Stratified: one draw from each of batch_size equal slices of the mass.
        masses = (np.arange(batch_size) + self.rng.random(batch_size)) * segment
        indices = np.array([self.tree.find(m) for m in masses])
        probs = self.scaled[indices] / total
        min_prob = self.scaled[: self.size].min() / total
        weights = (self.size * probs) ** (-beta) / (self.size * min_prob) ** (-beta)
        return indices, weights

    def update_priorities(self, indices, td_errors) -> None:
        for index, error in zip(indices, td_errors):
            priority = abs(float(error)) + self.eps
            self._set(int(index), priority)
            self.max_priority = max(self.max_priority, priority)


if __name__ == "__main__":
    # 1. Prioritised replay: probabilities and weights.
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
