"""Reference solution for the proximal policy optimisation exercise."""

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
    ratio = torch.exp(logp_new - logp_old)
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    return torch.min(ratio * advantages, clipped * advantages)




def categorical_entropy(logits):
    logp = torch.log_softmax(logits, dim=-1)
    return -(logp.exp() * logp).sum(dim=-1)


def approx_kl(logp_new, logp_old):
    log_ratio = logp_new - logp_old
    return (torch.exp(log_ratio) - 1.0 - log_ratio).mean()


def clip_fraction(logp_new, logp_old, clip_eps):
    ratio = torch.exp(logp_new - logp_old)
    return ((ratio - 1.0).abs() > clip_eps).float().mean()


def ppo_loss(model, batch, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    logits, values = model(batch["obs"])
    logp = torch.log_softmax(logits, dim=-1).gather(-1, batch["actions"][:, None]).squeeze(-1)
    adv = batch["advantages"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    policy_loss = -clipped_surrogate(logp, batch["logp"], adv, clip_eps).mean()
    value_loss = ((values - batch["returns"]) ** 2).mean()
    entropy = categorical_entropy(logits).mean()
    return policy_loss + vf_coef * value_loss - ent_coef * entropy


def ppo_epochs(model, optimizer, batch, epochs=4, minibatches=4, clip_eps=0.2, target_kl=None):
    n = batch["obs"].shape[0]
    mb_size = n // minibatches
    kls = []
    for _ in range(epochs):
        order = torch.randperm(n)
        for start in range(0, n, mb_size):
            idx = order[start:start + mb_size]
            loss = ppo_loss(model, {k: v[idx] for k, v in batch.items()}, clip_eps=clip_eps)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        with torch.no_grad():
            logits, _ = model(batch["obs"])
            logp = torch.log_softmax(logits, dim=-1).gather(-1, batch["actions"][:, None]).squeeze(-1)
            kls.append(float(approx_kl(logp, batch["logp"])))
        if target_kl is not None and kls[-1] > target_kl:
            break
    return kls, len(kls)


def train_ppo(seed=0, iterations=20, n_envs=8, n_steps=128, epochs=4, minibatches=4, lr=2.5e-3, target_kl=None):
    torch.manual_seed(seed)
    env = VecCartPole(n_envs, seed=seed)
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    history = []
    for it in range(iterations):
        for group in optimizer.param_groups:
            group["lr"] = lr * (1.0 - it / iterations)
        batch = collect_rollout(env, model, n_steps)
        ppo_epochs(model, optimizer, batch, epochs, minibatches, target_kl=target_kl)
        recent = env.finished[-10:]
        history.append(float(np.mean(recent)) if recent else 0.0)
    return model, history


if __name__ == "__main__":
    import time

    start = time.perf_counter()
    model, history = train_ppo(seed=0)
    for it in range(0, len(history), 4):
        print(f"iteration {it:3d}: mean return of the last 10 episodes {history[it]:6.1f}")
    print(f"greedy policy over 10 episodes: {evaluate(model):.1f} (500 is the limit)")
    print(f"trained in {time.perf_counter() - start:.1f} s")
