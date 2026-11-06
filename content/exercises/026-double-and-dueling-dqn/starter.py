"""Double DQN, dueling heads, and n-step targets.

Fill in the three functions below, in order. Each one is a drop-in change to
the DQN you built in the previous chapter: a different target, a different
output head, and a longer reward sum.
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

    # 3. n-step targets on a short episode.
    print("3-step targets:", n_step_targets([1.0, 0.0, 2.0, 1.0], [False, False, False, True],
                                             [5.0, 5.0, 5.0, 5.0], 0.9, 3))
