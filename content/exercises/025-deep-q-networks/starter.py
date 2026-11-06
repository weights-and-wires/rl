"""Deep Q-networks: a replay buffer, a target network, the Huber loss, and CartPole.

Fill in the functions below, in order. The last one, `train_dqn`, wires the
others into the DQN training loop and should balance CartPole in about
30,000 environment steps.

`greedy_return` is already written for you. It scores a trained network.
"""

import numpy as np
import torch
import torch.nn as nn

import rlenvs

torch.set_num_threads(1)


class ReplayBuffer:
    """A fixed-size ring buffer of transitions, sampled uniformly.

    Args:
        capacity: the most transitions it holds. Once full, each add
            overwrites the oldest transition.
        obs_dim: the length of an observation vector.
        seed: seeds the buffer's own numpy generator for sampling.
    """

    def __init__(self, capacity: int, obs_dim: int, seed: int = 0) -> None:
        raise NotImplementedError("ReplayBuffer.__init__")

    def __len__(self) -> int:
        """The number of transitions stored, at most `capacity`."""
        raise NotImplementedError("ReplayBuffer.__len__")

    def add(self, s, a, r, s2, terminated) -> None:
        """Store one transition: state, action, reward r_t, next state, and termination flag."""
        raise NotImplementedError("ReplayBuffer.add")

    def sample(self, batch_size: int):
        """Draw `batch_size` stored transitions uniformly at random, with replacement.

        Returns:
            A tuple (s, a, r, s2, terminated) of numpy arrays with shapes
            (B, obs_dim) float32, (B,) int64, (B,) float32, (B, obs_dim)
            float32, and (B,) float32 holding 0.0 or 1.0.
        """
        raise NotImplementedError("ReplayBuffer.sample")


def make_q_network(obs_dim: int, n_actions: int, hidden: int = 96) -> nn.Module:
    """An MLP from observations to one Q-value per action.

    Two hidden layers of width `hidden` with ReLU activations:
    Linear(obs_dim, hidden), ReLU, Linear(hidden, hidden), ReLU,
    Linear(hidden, n_actions). With the defaults on CartPole that's 9,986
    parameters.
    """
    raise NotImplementedError("make_q_network")


def dqn_targets(rewards, next_q, terminated, gamma: float) -> torch.Tensor:
    """The one-step Q-learning targets.

    Args:
        rewards: r_t, shape (B,).
        next_q: the target network's Q-values at s_{t+1}, shape (B, A).
        terminated: 1.0 where s_{t+1} is terminal, else 0.0, shape (B,).
        gamma: discount.

    Returns:
        y = r + gamma * (1 - terminated) * max_a next_q, shape (B,), with no
        gradient attached.
    """
    raise NotImplementedError("dqn_targets")


def huber_loss(td_errors: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """The mean Huber loss of a batch of TD errors u.

    Each error costs u**2 / 2 when |u| <= kappa, and kappa * (|u| - kappa / 2)
    otherwise.

    Args:
        td_errors: shape (B,).
        kappa: where the loss switches from quadratic to linear.

    Returns:
        A scalar tensor, the mean over the batch.
    """
    raise NotImplementedError("huber_loss")


def dqn_loss(q_net, target_net, batch, gamma: float, kappa: float = 1.0) -> torch.Tensor:
    """The DQN loss on one minibatch.

    Args:
        q_net: the online network, which gets the gradient.
        target_net: the target network, used only to build targets.
        batch: the tuple (s, a, r, s2, terminated) from ReplayBuffer.sample.
        gamma: discount.
        kappa: the Huber threshold.

    Returns:
        huber_loss(y - Q(s, a)), a scalar tensor. Gradients must flow into
        q_net only.
    """
    raise NotImplementedError("dqn_loss")


def soft_update(target_net, q_net, tau: float) -> None:
    """Polyak averaging, in place: target <- (1 - tau) * target + tau * online.

    tau = 1 copies the online weights. The target must stay a separate copy.
    """
    raise NotImplementedError("soft_update")


def epsilon_at(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    """Linear decay from eps_start at step 0 to eps_end at decay_steps, then flat."""
    raise NotImplementedError("epsilon_at")


def select_action(q_net, obs, epsilon: float, rng: np.random.Generator) -> int:
    """Epsilon-greedy: a uniformly random action with probability epsilon, else argmax Q.

    Compute the Q-values first. Then draw rng.random(), and if it's below
    epsilon, return rng.integers(n_actions). Otherwise return the argmax.
    """
    raise NotImplementedError("select_action")


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
    """Train a DQN on `env` and return the online Q-network.

    The defaults are tuned for rlenvs.CartPole. Follow this recipe:

    - Seed torch with torch.manual_seed(seed), make a numpy generator with
      np.random.default_rng(seed) for exploration, and give the buffer the
      same seed. Start with obs = env.reset(seed=seed).
    - Build the online and target networks with make_q_network, copy the
      online weights into the target, and use torch.optim.Adam.
    - At every step: set the learning rate to lr * (1 - step / total_steps),
      choose an action with select_action at epsilon_at(step, ...), step the
      environment, and store the transition with its `terminated` flag
      (never `truncated`). Reset the environment when either flag is true.
    - Once step >= learning_starts, and when step % train_every == 0, take
      one Adam step on dqn_loss of a sampled batch, then soft_update the
      target network.
    """
    raise NotImplementedError("train_dqn")


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
