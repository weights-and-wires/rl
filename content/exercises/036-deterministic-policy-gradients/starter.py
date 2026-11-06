"""Deterministic policy gradients: DDPG's pieces, TD3's fixes, and Pendulum.

Fill in the functions under "The exercise". The first, `lqr_dpg`, works on a
scalar linear-quadratic problem, where every quantity has a closed form, so you
can check the deterministic policy gradient exactly. The rest are the
pieces of a TD3 update in PyTorch: the smoothed target action, the clipped
double-Q target, the critic and actor losses, Polyak averaging, and the
delayed update that ties them together.

Everything above "The exercise" is given. `train_td3` runs your `td3_update`
on Pendulum. Shapes: states (N, obs_dim), actions (N, act_dim), and rewards,
termination flags, and critic outputs (N,).
"""

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
    """The coefficient P(theta) with V(s) = -P * s**2 for the policy a = theta * s.

    Given in full, because the lesson works it out by hand. Assume
    gamma * (c + b * theta)**2 < 1.
    """
    m = lqr.c + lqr.b * theta
    return (1.0 + lqr.lam * theta**2) / (1.0 - lqr.gamma * m**2)


def lqr_q(s, a, theta, lqr):
    """Q(s, a) for the policy a = theta * s: one step with action a, then follow the policy.

    Given in full. It is -(s**2 + lam * a**2) - gamma * P * (c * s + b * a)**2.
    """
    P = lqr_value_coef(theta, lqr)
    s2 = lqr.c * s + lqr.b * a
    return -(s**2 + lqr.lam * a**2) - lqr.gamma * P * s2**2


# The exercise -----------------------------------------------------------------


def lqr_dpg(theta, lqr, s0, horizon=2000):
    """The deterministic policy gradient dJ/dtheta, with J(theta) = V(s0).

    Sum over the policy's own trajectory from s0, for t = 0 .. horizon - 1:
    gamma**t * (d mu / d theta at s_t) * (dQ/da at s_t, a = mu(s_t)).

    Args:
        theta: the policy's gain.
        lqr: a `ScalarLQR`.
        s0: the start state, a float.
        horizon: the number of steps to sum.

    Returns:
        The gradient, a float.
    """
    raise NotImplementedError("lqr_dpg")


def smoothed_target_action(mu_next, eps, sigma, noise_clip, max_action):
    """TD3's target policy smoothing: clip(mu + clip(sigma * eps, -c, c), -max_action, max_action).

    Args:
        mu_next: the target actor's actions at the next states, shape (N, act_dim).
        eps: standard normal draws, the same shape.
        sigma: the noise's standard deviation, in action units.
        noise_clip: c, the largest noise allowed, in action units.
        max_action: the action bound.

    Returns:
        The smoothed actions, shape (N, act_dim).
    """
    raise NotImplementedError("smoothed_target_action")


def td3_target(r, terminated, q1_next, q2_next, gamma):
    """The clipped double-Q target y = r + gamma * (1 - terminated) * min(q1_next, q2_next).

    Args:
        r, terminated, q1_next, q2_next: tensors of shape (N,).
        gamma: the discount factor.

    Returns:
        y, shape (N,).
    """
    raise NotImplementedError("td3_target")


def critic_loss(agent, batch, eps):
    """The TD3 critic loss: both critics' mean squared errors against one shared target.

    Build the target from the TARGET actor and the two TARGET critics, with
    `smoothed_target_action` and `td3_target`, and keep it out of the graph.

    Args:
        agent: a `TD3Agent`.
        batch: a dict of tensors "s", "a", "r", "s2", "terminated".
        eps: standard normal draws for the smoothing noise, shape (N, act_dim).

    Returns:
        A scalar tensor: mean((Q1(s, a) - y)**2) + mean((Q2(s, a) - y)**2).
    """
    raise NotImplementedError("critic_loss")


def actor_loss(agent, s):
    """The DDPG and TD3 actor loss: -mean over the batch of Q1(s, mu_theta(s)).

    Args:
        agent: a `TD3Agent`.
        s: states, shape (N, obs_dim).

    Returns:
        A scalar tensor.
    """
    raise NotImplementedError("actor_loss")


def soft_update(target, source, tau):
    """Polyak averaging, in place: target <- tau * source + (1 - tau) * target.

    Args:
        target: a `torch.nn.Module` to update.
        source: a module with the same architecture. Leave it unchanged.
        tau: the fraction to move, in [0, 1].
    """
    raise NotImplementedError("soft_update")


def td3_update(agent, batch, eps, step):
    """One TD3 update.

    Always take one critic step on `critic_loss`. When step % agent.policy_delay
    == 0, also take one actor step on `actor_loss(agent, batch["s"])`, then
    soft-update all three target networks with agent.tau.

    Args:
        agent: a `TD3Agent`.
        batch: a dict of tensors from `ReplayBuffer.sample`.
        eps: standard normal draws for the smoothing noise.
        step: the update count, starting at 1.

    Returns:
        True if the actor was updated, else False.
    """
    raise NotImplementedError("td3_update")


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
