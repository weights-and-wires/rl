"""Reference solution for the deep Q-network exercise."""

import numpy as np
import torch
import torch.nn as nn

import rlenvs

torch.set_num_threads(1)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, seed: int = 0) -> None:
        self.capacity = capacity
        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.term = np.zeros(capacity, dtype=np.float32)
        self.next = 0  # the slot the next transition goes into
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, s, a, r, s2, terminated) -> None:
        i = self.next
        self.s[i], self.a[i], self.r[i] = s, a, r
        self.s2[i], self.term[i] = s2, float(terminated)
        self.next = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.term[idx]


def make_q_network(obs_dim: int, n_actions: int, hidden: int = 96) -> nn.Module:
    return nn.Sequential(
        nn.Linear(obs_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_actions),
    )


def dqn_targets(rewards, next_q, terminated, gamma: float) -> torch.Tensor:
    rewards = torch.as_tensor(rewards, dtype=next_q.dtype)
    terminated = torch.as_tensor(terminated, dtype=next_q.dtype)
    with torch.no_grad():
        return rewards + gamma * (1.0 - terminated) * next_q.max(dim=1).values


def huber_loss(td_errors: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    abs_u = td_errors.abs()
    quadratic = 0.5 * td_errors**2
    linear = kappa * (abs_u - 0.5 * kappa)
    return torch.where(abs_u <= kappa, quadratic, linear).mean()


def dqn_loss(q_net, target_net, batch, gamma: float, kappa: float = 1.0) -> torch.Tensor:
    s, a, r, s2, term = batch
    s = torch.as_tensor(s, dtype=torch.float32)
    s2 = torch.as_tensor(s2, dtype=torch.float32)
    a = torch.as_tensor(a, dtype=torch.int64)
    with torch.no_grad():
        y = dqn_targets(r, target_net(s2), term, gamma)
    q_sa = q_net(s).gather(1, a[:, None]).squeeze(1)
    return huber_loss(y - q_sa, kappa)


def soft_update(target_net, q_net, tau: float) -> None:
    with torch.no_grad():
        for p_target, p in zip(target_net.parameters(), q_net.parameters()):
            p_target.mul_(1.0 - tau).add_(tau * p)


def epsilon_at(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    fraction = min(step / decay_steps, 1.0)
    return eps_start + fraction * (eps_end - eps_start)


def select_action(q_net, obs, epsilon: float, rng: np.random.Generator) -> int:
    with torch.no_grad():
        q = q_net(torch.as_tensor(obs, dtype=torch.float32))
    if rng.random() < epsilon:
        return int(rng.integers(q.shape[-1]))
    return int(q.argmax())


def train_dqn(
    env,
    total_steps: int = 30_000,
    seed: int = 0,
    gamma: float = 0.98,
    lr: float = 5e-4,
    batch_size: int = 256,
    buffer_size: int = 30_000,
    learning_starts: int = 1_000,
    train_every: int = 4,
    tau: float = 0.005,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_steps: int = 6_000,
    hidden: int = 96,
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    q_net = make_q_network(env.obs_dim, env.n_actions, hidden)
    target_net = make_q_network(env.obs_dim, env.n_actions, hidden)
    target_net.load_state_dict(q_net.state_dict())
    optimiser = torch.optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_size, env.obs_dim, seed=seed)

    obs = env.reset(seed=seed)
    for step in range(total_steps):
        for group in optimiser.param_groups:
            group["lr"] = lr * (1.0 - step / total_steps)
        epsilon = epsilon_at(step, eps_start, eps_end, eps_decay_steps)
        action = select_action(q_net, obs, epsilon, rng)
        obs2, reward, terminated, truncated = env.step(action)
        # Store `terminated`, not `truncated`: a time limit is not a real end.
        buffer.add(obs, action, reward, obs2, terminated)
        obs = env.reset() if (terminated or truncated) else obs2

        if step >= learning_starts and step % train_every == 0:
            loss = dqn_loss(q_net, target_net, buffer.sample(batch_size), gamma)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            soft_update(target_net, q_net, tau)
    return q_net


def greedy_return(q_net, env, episodes: int = 10, seed: int = 1000) -> float:
    """Mean return of the greedy policy over `episodes` episodes. Already written."""
    returns = []
    for episode in range(episodes):
        obs, total, done = env.reset(seed=seed + episode), 0.0, False
        while not done:
            with torch.no_grad():
                action = int(q_net(torch.as_tensor(obs, dtype=torch.float32)).argmax())
            obs, reward, terminated, truncated = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
    return float(np.mean(returns))


if __name__ == "__main__":
    # The pieces on a tiny batch, then a full training run.
    q_net = make_q_network(4, 2)
    print(f"Q-network parameters: {sum(p.numel() for p in q_net.parameters())}")
    y = dqn_targets(torch.tensor([1.0, 1.0]), torch.tensor([[2.5, 1.5], [2.5, 1.5]]),
                    torch.tensor([0.0, 1.0]), gamma=0.9)
    print(f"targets for a live and a terminal step: {y.tolist()}")
    print(f"Huber loss of errors [0.5, 3]: {huber_loss(torch.tensor([0.5, 3.0])).item():.4f}")

    env = rlenvs.CartPole(max_steps=500)
    untrained = greedy_return(make_q_network(4, 2), env)
    print(f"greedy return before training: {untrained:.1f}")
    q_net = train_dqn(env, seed=0)
    print(f"greedy return after 30,000 steps: {greedy_return(q_net, env):.1f}")
