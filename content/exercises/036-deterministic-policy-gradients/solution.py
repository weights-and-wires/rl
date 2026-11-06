"""Reference solution for the deterministic policy gradients exercise."""

import copy
from dataclasses import dataclass

import numpy as np
import torch

import rlenvs

torch.set_num_threads(1)


# Given: a scalar linear-quadratic problem -----------------------------------


@dataclass
class ScalarLQR:
    """s_{t+1} = c * s_t + b * a_t, with reward r_t = -(s_t**2 + lam * a_t**2)."""

    c: float = 1.0
    b: float = 1.0
    lam: float = 0.5
    gamma: float = 0.9

    def step(self, s, a):
        return self.c * s + self.b * a, -(s**2 + self.lam * a**2)


# Given: networks, the agent, a replay buffer, and the training loop -----------


def mlp(n_in, n_out, hidden):
    return torch.nn.Sequential(
        torch.nn.Linear(n_in, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, n_out),
    )


class Actor(torch.nn.Module):
    """The deterministic policy mu_theta(s) = max_action * tanh(net(s)), shape (N, act_dim)."""

    def __init__(self, obs_dim, act_dim, max_action, hidden=64):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, hidden)
        self.max_action = max_action

    def forward(self, s):
        return self.max_action * torch.tanh(self.net(s))


class Critic(torch.nn.Module):
    """Q_phi(s, a), returned with shape (N,)."""

    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


class TD3Agent:
    """An actor, two critics, a target copy of each, and their optimisers."""

    def __init__(self, obs_dim, act_dim, max_action, hidden=128, actor_lr=3e-4, critic_lr=1e-3,
                 gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_delay=2, seed=0):
        torch.manual_seed(seed)
        self.actor = Actor(obs_dim, act_dim, max_action, hidden)
        self.critic1 = Critic(obs_dim, act_dim, hidden)
        self.critic2 = Critic(obs_dim, act_dim, hidden)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=critic_lr
        )
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        # Target smoothing noise and its clip, in action units.
        self.policy_noise = policy_noise * max_action
        self.noise_clip = noise_clip * max_action
        self.policy_delay = policy_delay


class ReplayBuffer:
    """A fixed-size store of transitions, sampled uniformly."""

    def __init__(self, obs_dim, act_dim, size):
        self.s = np.zeros((size, obs_dim), dtype=np.float32)
        self.a = np.zeros((size, act_dim), dtype=np.float32)
        self.r = np.zeros(size, dtype=np.float32)
        self.s2 = np.zeros((size, obs_dim), dtype=np.float32)
        self.terminated = np.zeros(size, dtype=np.float32)
        self.n = 0

    def add(self, s, a, r, s2, terminated):
        i = self.n % len(self.r)
        self.s[i], self.a[i], self.r[i], self.s2[i], self.terminated[i] = s, a, r, s2, terminated
        self.n += 1

    def sample(self, batch_size, rng):
        idx = rng.integers(0, min(self.n, len(self.r)), size=batch_size)
        return {
            "s": torch.as_tensor(self.s[idx]),
            "a": torch.as_tensor(self.a[idx]),
            "r": torch.as_tensor(self.r[idx]),
            "s2": torch.as_tensor(self.s2[idx]),
            "terminated": torch.as_tensor(self.terminated[idx]),
        }


def evaluate(actor, episodes=10, seed=1000):
    """Mean return of the noise-free actor over Pendulum episodes seeded seed, seed + 1, and so on."""
    returns = []
    for k in range(episodes):
        env = rlenvs.Pendulum(max_steps=200, seed=seed + k)
        s, total, done = env.reset(), 0.0, False
        while not done:
            with torch.no_grad():
                a = actor(torch.as_tensor(s, dtype=torch.float32)).numpy()
            s, r, terminated, truncated = env.step(a)
            total += r
            done = terminated or truncated
        returns.append(total)
    return float(np.mean(returns))


def train_td3(steps=8000, seed=0, start_steps=1000, batch_size=128):
    """Train TD3 on Pendulum with your update. Returns the agent and the training episode returns."""
    rng = np.random.default_rng(seed)
    env = rlenvs.Pendulum(max_steps=200, seed=seed)
    agent = TD3Agent(env.obs_dim, env.act_dim, env.max_torque, seed=seed)
    buffer = ReplayBuffer(env.obs_dim, env.act_dim, steps)
    s, episode_return, returns, updates = env.reset(), 0.0, [], 0
    for t in range(steps):
        if t < start_steps:
            a = rng.uniform(-env.max_torque, env.max_torque, size=env.act_dim)
        else:
            with torch.no_grad():
                a = agent.actor(torch.as_tensor(s, dtype=torch.float32)).numpy()
            a = np.clip(a + rng.normal(0.0, 0.1 * env.max_torque, size=env.act_dim),
                        -env.max_torque, env.max_torque)
        s2, r, terminated, truncated = env.step(a)
        buffer.add(s, a, r, s2, float(terminated))
        episode_return += r
        s = s2
        if terminated or truncated:
            returns.append(episode_return)
            s, episode_return = env.reset(), 0.0
        if t >= start_steps:
            updates += 1
            batch = buffer.sample(batch_size, rng)
            eps = torch.as_tensor(rng.standard_normal(batch["a"].shape), dtype=torch.float32)
            td3_update(agent, batch, eps, updates)
    return agent, returns


# Given: the value and action value of a linear policy ------------------------


def lqr_value_coef(theta, lqr):
    m = lqr.c + lqr.b * theta
    return (1.0 + lqr.lam * theta**2) / (1.0 - lqr.gamma * m**2)


def lqr_q(s, a, theta, lqr):
    P = lqr_value_coef(theta, lqr)
    s2 = lqr.c * s + lqr.b * a
    return -(s**2 + lqr.lam * a**2) - lqr.gamma * P * s2**2


# The exercise -----------------------------------------------------------------


def lqr_dpg(theta, lqr, s0, horizon=2000):
    P = lqr_value_coef(theta, lqr)
    grad, s, discount = 0.0, float(s0), 1.0
    for _ in range(horizon):
        a = theta * s
        # dQ/da at a = mu(s), from lqr_q: -2 lam a - 2 gamma P b (c s + b a).
        dq_da = -2.0 * lqr.lam * a - 2.0 * lqr.gamma * P * lqr.b * (lqr.c * s + lqr.b * a)
        dmu_dtheta = s
        grad += discount * dmu_dtheta * dq_da
        s, discount = lqr.c * s + lqr.b * a, discount * lqr.gamma
    return grad


def smoothed_target_action(mu_next, eps, sigma, noise_clip, max_action):
    noise = (sigma * eps).clamp(-noise_clip, noise_clip)
    return (mu_next + noise).clamp(-max_action, max_action)


def td3_target(r, terminated, q1_next, q2_next, gamma):
    return r + gamma * (1.0 - terminated) * torch.minimum(q1_next, q2_next)


def critic_loss(agent, batch, eps):
    with torch.no_grad():
        a2 = smoothed_target_action(
            agent.actor_target(batch["s2"]), eps, agent.policy_noise, agent.noise_clip, agent.max_action
        )
        y = td3_target(
            batch["r"],
            batch["terminated"],
            agent.critic1_target(batch["s2"], a2),
            agent.critic2_target(batch["s2"], a2),
            agent.gamma,
        )
    q1 = agent.critic1(batch["s"], batch["a"])
    q2 = agent.critic2(batch["s"], batch["a"])
    return ((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean()


def actor_loss(agent, s):
    return -agent.critic1(s, agent.actor(s)).mean()


def soft_update(target, source, tau):
    with torch.no_grad():
        for t, p in zip(target.parameters(), source.parameters()):
            t.mul_(1.0 - tau).add_(tau * p)


def td3_update(agent, batch, eps, step):
    loss = critic_loss(agent, batch, eps)
    agent.critic_opt.zero_grad()
    loss.backward()
    agent.critic_opt.step()
    if step % agent.policy_delay != 0:
        return False
    loss = actor_loss(agent, batch["s"])
    agent.actor_opt.zero_grad()
    loss.backward()
    agent.actor_opt.step()
    soft_update(agent.actor_target, agent.actor, agent.tau)
    soft_update(agent.critic1_target, agent.critic1, agent.tau)
    soft_update(agent.critic2_target, agent.critic2, agent.tau)
    return True


if __name__ == "__main__":
    lqr = ScalarLQR()
    theta = -0.5
    fd = (-lqr_value_coef(theta + 1e-6, lqr) + lqr_value_coef(theta - 1e-6, lqr)) / 2e-6
    print("Scalar LQR, policy a = theta * s with theta = -0.5, start s0 = 1")
    print(f"  V(s0) = -P = {-lqr_value_coef(theta, lqr):.6f}")
    print(f"  deterministic policy gradient = {lqr_dpg(theta, lqr, 1.0):.6f}")
    print(f"  finite difference of J        = {fd:.6f}")

    print("\nTD3 on Pendulum, 8000 steps (a few seconds)")
    agent, returns = train_td3(steps=8000, seed=0)
    for k in range(0, len(returns), 8):
        print(f"  episode {k + 1:>2}: return {returns[k]:8.1f}")
    print(f"  noise-free actor, mean over 10 episodes: {evaluate(agent.actor):.1f}")
