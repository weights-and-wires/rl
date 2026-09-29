"""Reference solution for the DQN improvements exercise."""

import numpy as np
import torch

torch.set_num_threads(1)


def double_dqn_targets(rewards, next_q_online, next_q_target, terminated, gamma: float) -> torch.Tensor:
    rewards = torch.as_tensor(rewards, dtype=next_q_target.dtype)
    terminated = torch.as_tensor(terminated, dtype=next_q_target.dtype)
    with torch.no_grad():
        # The online network picks the action; the target network scores it.
        best = next_q_online.argmax(dim=1, keepdim=True)
        value = next_q_target.gather(1, best).squeeze(1)
        return rewards + gamma * (1.0 - terminated) * value


def dueling_combine(value: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    return value.reshape(-1, 1) + advantages - advantages.mean(dim=1, keepdim=True)


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


def n_step_targets(rewards, terminated, bootstrap, gamma: float, n: int) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=float)
    terminated = np.asarray(terminated, dtype=bool)
    bootstrap = np.asarray(bootstrap, dtype=float)
    T = len(rewards)
    targets = np.zeros(T)
    for t in range(T):
        total, discount = 0.0, 1.0
        for k in range(n):
            j = t + k
            total += discount * rewards[j]
            discount *= gamma
            if terminated[j]:
                break  # the episode ended: nothing to bootstrap from
            if k == n - 1 or j == T - 1:
                total += discount * bootstrap[j]
                break
        targets[t] = total
    return targets


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
