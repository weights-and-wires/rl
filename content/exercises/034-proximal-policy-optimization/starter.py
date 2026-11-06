"""Proximal policy optimisation on CartPole.

Fill in the functions below. The network, the vectorised CartPole, the rollout
loop, GAE from the previous chapter, and the greedy evaluation are given. You
write PPO's losses and diagnostics, then the training loop that uses them.

Log-probabilities, advantages, values, and returns are 1-D tensors with one
entry per sample.
"""

import numpy as np
import torch
from torch import nn

import rlenvs

torch.set_num_threads(1)


# Given: the network, the vectorised CartPole rollout, and GAE --------------


class ActorCritic(nn.Module):
    """Separate policy and value MLPs, 4 -> 32 -> 32 -> out, with tanh. Given."""

    def __init__(self, obs_dim=4, n_actions=2, hidden=32):
        super().__init__()

        def mlp(out, gain):
            layers = [nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out)]
            for layer in layers:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.zeros_(layer.bias)
            nn.init.orthogonal_(layers[-1].weight, gain=gain)
            return nn.Sequential(*layers)

        self.policy = mlp(n_actions, 0.01)
        self.value = mlp(1, 1.0)

    def forward(self, obs):
        """Returns (logits, values) with shapes (N, A) and (N,)."""
        return self.policy(obs), self.value(obs).squeeze(-1)


def gae(rewards, values, next_values, terminated, truncated, gamma, lam):
    """Generalised advantage estimates over axis 0, as in the previous chapter. Given."""
    delta = rewards + gamma * (1.0 - terminated) * next_values - values
    ended = np.maximum(terminated, truncated)
    adv = np.zeros_like(delta)
    running = np.zeros(delta.shape[1:])
    for t in range(delta.shape[0] - 1, -1, -1):
        running = delta[t] + gamma * lam * (1.0 - ended[t]) * running
        adv[t] = running
    return adv


class VecCartPole:
    """N CartPoles stepped together, resetting each one when its episode ends. Given."""

    def __init__(self, n_envs, max_steps=500, seed=0):
        self.n, self.max_steps = n_envs, max_steps
        self.rng = np.random.default_rng(seed)
        self.state = self.rng.uniform(-0.05, 0.05, size=(n_envs, 4))
        self.t = np.zeros(n_envs, dtype=int)
        self.ep_return = np.zeros(n_envs)
        self.finished = []  # returns of completed episodes

    def step(self, actions):
        """Returns (next_obs, final_obs, rewards, terminated, truncated).

        final_obs is the state each step reached. next_obs is the same, except
        that environments whose episode ended have been reset.
        """
        final = rlenvs.CartPole.dynamics(self.state, actions)
        self.t += 1
        terminated = rlenvs.CartPole.failed(final)
        truncated = ~terminated & (self.t >= self.max_steps)
        rewards = np.ones(self.n)
        self.ep_return += rewards
        done = terminated | truncated
        nxt = final.copy()
        if done.any():
            self.finished.extend(self.ep_return[done].tolist())
            nxt[done] = self.rng.uniform(-0.05, 0.05, size=(int(done.sum()), 4))
            self.t[done] = 0
            self.ep_return[done] = 0.0
        self.state = nxt
        return nxt, final, rewards, terminated, truncated


def collect_rollout(env, model, n_steps, gamma=0.99, lam=0.95):
    """Run the policy for n_steps in every environment and compute GAE. Given.

    Returns a dict of flat tensors, each with n_steps * n_envs rows: obs,
    actions, logp (log-probabilities under the policy that acted), values,
    advantages, and returns (the lambda-returns, advantages + values).
    """
    N = env.n
    obs_buf = np.zeros((n_steps, N, 4))
    act_buf = np.zeros((n_steps, N), dtype=int)
    logp_buf, val_buf, next_val_buf = np.zeros((n_steps, N)), np.zeros((n_steps, N)), np.zeros((n_steps, N))
    rew_buf, term_buf, trunc_buf = np.zeros((n_steps, N)), np.zeros((n_steps, N)), np.zeros((n_steps, N))
    obs = env.state.copy()
    with torch.no_grad():
        for t in range(n_steps):
            logits, values = model(torch.as_tensor(obs, dtype=torch.float32))
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            nxt, final, rewards, terminated, truncated = env.step(actions.numpy())
            _, final_values = model(torch.as_tensor(final, dtype=torch.float32))
            obs_buf[t], act_buf[t] = obs, actions.numpy()
            logp_buf[t], val_buf[t] = dist.log_prob(actions).numpy(), values.numpy()
            next_val_buf[t], rew_buf[t] = final_values.numpy(), rewards
            term_buf[t], trunc_buf[t] = terminated, truncated
            obs = nxt
    # The rollout's end is a truncation too: bootstrap from V of the last state.
    trunc_buf[-1] = 1.0
    adv = gae(rew_buf, val_buf, next_val_buf, term_buf, trunc_buf, gamma, lam)
    flat = lambda x: torch.as_tensor(x.reshape(n_steps * N, *x.shape[2:]), dtype=torch.float32)
    return {
        "obs": flat(obs_buf),
        "actions": torch.as_tensor(act_buf.reshape(-1)),
        "logp": flat(logp_buf),
        "values": flat(val_buf),
        "advantages": flat(adv),
        "returns": flat(adv + val_buf),
    }


def evaluate(model, episodes=10, max_steps=500, seed=123):
    """Mean return of the greedy policy over `episodes` CartPole episodes. Given."""
    rng = np.random.default_rng(seed)
    state = rng.uniform(-0.05, 0.05, size=(episodes, 4))
    alive = np.ones(episodes, dtype=bool)
    total = np.zeros(episodes)
    with torch.no_grad():
        for _ in range(max_steps):
            logits, _ = model(torch.as_tensor(state, dtype=torch.float32))
            state = rlenvs.CartPole.dynamics(state, logits.argmax(-1).numpy())
            total += alive
            alive &= ~rlenvs.CartPole.failed(state)
            if not alive.any():
                break
    return float(total.mean())


# The exercise --------------------------------------------------------------


def clipped_surrogate(logp_new, logp_old, advantages, clip_eps):
    """PPO's clipped surrogate, one value per sample (to maximise).

    min(r * A, clip(r, 1 - clip_eps, 1 + clip_eps) * A), with the ratio
    r = exp(logp_new - logp_old).

    Args:
        logp_new: log pi_theta(a | s), shape (N,), differentiable.
        logp_old: log pi_old(a | s) from when the action was sampled, shape (N,).
        advantages: shape (N,).
        clip_eps: the clipping range epsilon, such as 0.2.

    Returns:
        A tensor of shape (N,). Don't average it.
    """
    raise NotImplementedError("clipped_surrogate")




def categorical_entropy(logits):
    """The entropy of each row's categorical distribution.

    Args:
        logits: shape (N, A).

    Returns:
        A tensor of shape (N,): -sum_a pi(a) log pi(a) for each row.
    """
    raise NotImplementedError("categorical_entropy")


def approx_kl(logp_new, logp_old):
    """The k3 estimate of KL(pi_old || pi_new) from samples drawn under pi_old.

    The mean of (r - 1) - log r, with r = exp(logp_new - logp_old).

    Args:
        logp_new: shape (N,).
        logp_old: shape (N,).

    Returns:
        A scalar tensor.
    """
    raise NotImplementedError("approx_kl")


def clip_fraction(logp_new, logp_old, clip_eps):
    """The fraction of samples whose ratio lies outside [1 - clip_eps, 1 + clip_eps].

    Args:
        logp_new: shape (N,).
        logp_old: shape (N,).
        clip_eps: the clipping range.

    Returns:
        A scalar tensor.
    """
    raise NotImplementedError("clip_fraction")


def ppo_loss(model, batch, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    """The PPO loss to minimise on one minibatch.

    policy_loss + vf_coef * value_loss - ent_coef * mean entropy, where:

    - policy_loss is minus the mean of clipped_surrogate, using the minibatch's
      advantages normalised to mean 0 and standard deviation 1:
      (adv - adv.mean()) / (adv.std() + 1e-8).
    - value_loss is the mean squared error between the model's values and
      batch["returns"].

    Args:
        model: maps obs, shape (N, 4), to (logits, values), shapes (N, 2) and (N,).
        batch: a dict of tensors with N rows each: "obs", "actions", "logp",
            "values", "advantages", and "returns", as collect_rollout makes.
        clip_eps: the policy clipping range.
        vf_coef: the value-loss coefficient.
        ent_coef: the entropy-bonus coefficient.

    Returns:
        A scalar tensor that you can call .backward() on.
    """
    raise NotImplementedError("ppo_loss")


def train_ppo(seed=0, iterations=20, n_envs=8, n_steps=128, epochs=4, minibatches=4, lr=2.5e-3):
    """Train an ActorCritic on CartPole with PPO.

    Call torch.manual_seed(seed), build VecCartPole(n_envs, seed=seed),
    ActorCritic(), and Adam with lr and eps=1e-5. Then, for each iteration:

    1. Anneal the learning rate linearly: lr * (1 - iteration / iterations).
    2. Collect a batch with collect_rollout(env, model, n_steps).
    3. For `epochs` passes, shuffle the n_envs * n_steps samples and split
       them into `minibatches` equal minibatches. For each one, compute
       ppo_loss with its defaults, backpropagate, clip the gradient norm to
       0.5 with torch.nn.utils.clip_grad_norm_, and step the optimiser.
    4. Append the mean of the last 10 finished episode returns in
       env.finished to the history, or 0.0 if none have finished.

    Returns:
        (model, history), where history has one float per iteration.
    """
    raise NotImplementedError("train_ppo")


if __name__ == "__main__":
    import time

    start = time.perf_counter()
    model, history = train_ppo(seed=0)
    for it in range(0, len(history), 4):
        print(f"iteration {it:3d}: mean return of the last 10 episodes {history[it]:6.1f}")
    print(f"greedy policy over 10 episodes: {evaluate(model):.1f} (500 is the limit)")
    print(f"trained in {time.perf_counter() - start:.1f} s")
