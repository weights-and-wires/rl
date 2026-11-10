"""Reference solution for the Double and dueling DQN exercise."""

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

    # 3. n-step targets on a short episode.
    print("3-step targets:", n_step_targets([1.0, 0.0, 2.0, 1.0], [False, False, False, True],
                                             [5.0, 5.0, 5.0, 5.0], 0.9, 3))
