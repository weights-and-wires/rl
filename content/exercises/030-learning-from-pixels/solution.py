"""Learning from pixels: frame preprocessing, noisy layers, and augmented DQN.

Fill in the functions marked in the starter, in order. The Catch game and the
training loop at the bottom are given: they call your functions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------

def preprocess(frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
    """Merge two consecutive RGB frames into one grayscale frame in [0, 1]."""
    merged = np.maximum(frame_a, frame_b).astype(np.float64)
    gray = merged @ np.array([0.299, 0.587, 0.114])
    return gray / 255.0


def frame_stack(frames, k: int) -> np.ndarray:
    """Stack the last k frames; repeat the oldest frame when fewer than k exist."""
    recent = list(frames[-k:])
    recent = [recent[0]] * (k - len(recent)) + recent
    return np.stack(recent, axis=0)


# --------------------------------------------------------------------------
# Noisy layers
# --------------------------------------------------------------------------

def noise_f(x: np.ndarray) -> np.ndarray:
    """The Fortunato et al. transform f(x) = sign(x) * sqrt(|x|)."""
    return np.sign(x) * np.sqrt(np.abs(x))


def factorized_noise(n_in: int, n_out: int, rng: np.random.Generator):
    """Factorized Gaussian noise: draw n_in values, then n_out values."""
    f_in = noise_f(rng.standard_normal(n_in))
    f_out = noise_f(rng.standard_normal(n_out))
    return np.outer(f_out, f_in), f_out


class NoisyLinear(nn.Module):
    """A linear layer whose weights are mu + sigma * eps, with factorized eps."""

    def __init__(self, n_in: int, n_out: int, sigma0: float = 0.5, seed: int = 0):
        super().__init__()
        rng = np.random.default_rng(seed)
        bound = 1.0 / np.sqrt(n_in)
        self.weight_mu = nn.Parameter(
            torch.as_tensor(rng.uniform(-bound, bound, (n_out, n_in)), dtype=torch.float32))
        self.bias_mu = nn.Parameter(
            torch.as_tensor(rng.uniform(-bound, bound, n_out), dtype=torch.float32))
        self.weight_sigma = nn.Parameter(torch.full((n_out, n_in), sigma0 / np.sqrt(n_in)))
        self.bias_sigma = nn.Parameter(torch.full((n_out,), sigma0 / np.sqrt(n_in)))
        self.register_buffer("eps_w", torch.zeros(n_out, n_in))
        self.register_buffer("eps_b", torch.zeros(n_out))
        self.noisy = True
        self.reset_noise(rng)

    def reset_noise(self, rng: np.random.Generator) -> None:
        eps_w, eps_b = factorized_noise(self.weight_mu.shape[1], self.weight_mu.shape[0], rng)
        self.eps_w.copy_(torch.as_tensor(eps_w, dtype=self.eps_w.dtype))
        self.eps_b.copy_(torch.as_tensor(eps_b, dtype=self.eps_b.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.noisy:
            return F.linear(x, self.weight_mu, self.bias_mu)
        weight = self.weight_mu + self.weight_sigma * self.eps_w
        bias = self.bias_mu + self.bias_sigma * self.eps_b
        return F.linear(x, weight, bias)


# --------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------

def random_shift(imgs: np.ndarray, pad: int, rng: np.random.Generator) -> np.ndarray:
    """Pad each image by pad pixels (replicating edges), then crop a random window."""
    b, _, h, w = imgs.shape
    padded = np.pad(imgs, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="edge")
    shifts = rng.integers(0, 2 * pad + 1, size=(b, 2))
    out = np.empty_like(imgs)
    for i, (dy, dx) in enumerate(shifts):
        out[i] = padded[i, :, dy:dy + h, dx:dx + w]
    return out


def augmented_td_loss(q_fn, target_fn, batch: dict, gamma: float, pad: int,
                      rng: np.random.Generator, k_target: int = 1, m_q: int = 1):
    """DrQ-style TD loss: average the target over k shifts and Q over m shifts."""
    obs, next_obs = batch["obs"], batch["next_obs"]
    actions = torch.as_tensor(batch["actions"], dtype=torch.int64)
    rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
    not_done = 1.0 - torch.as_tensor(batch["terminated"], dtype=torch.float32)
    with torch.no_grad():
        best = []
        for _ in range(k_target):
            shifted = torch.as_tensor(random_shift(next_obs, pad, rng), dtype=torch.float32)
            best.append(target_fn(shifted).max(dim=1).values)
        y = rewards + gamma * not_done * torch.stack(best).mean(dim=0)
    losses = []
    for _ in range(m_q):
        shifted = torch.as_tensor(random_shift(obs, pad, rng), dtype=torch.float32)
        q = q_fn(shifted).gather(1, actions[:, None])[:, 0]
        losses.append(F.smooth_l1_loss(q, y))
    return torch.stack(losses).mean()


# --------------------------------------------------------------------------
# Given: Catch and the training loop
# --------------------------------------------------------------------------

class Catch:
    """A ball falls down a grid with 10 rows and 5 columns; move the paddle under it.

    The observation is a float32 image of shape (1, 10, 5) with the ball and the
    paddle drawn as 1.0. Actions: 0 left, 1 stay, 2 right. The ball starts in a
    random column of row 0 and falls one row per step. The paddle sits in row 9
    and starts in column 2. The episode ends when the ball reaches row 9, with
    reward 1.0 if it lands on the paddle and 0.0 otherwise.
    """

    H, W, N_ACTIONS = 10, 5, 3

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.ball_r, self.ball_c = 0, int(self.rng.integers(self.W))
        self.paddle = self.W // 2
        return self._obs()

    def step(self, action: int):
        self.paddle = int(np.clip(self.paddle + action - 1, 0, self.W - 1))
        self.ball_r += 1
        done = self.ball_r == self.H - 1
        reward = float(done and self.ball_c == self.paddle)
        return self._obs(), reward, done

    def _obs(self):
        img = np.zeros((1, self.H + 2, self.W + 2), dtype=np.float32)
        img[0, 1 + self.ball_r, 1 + self.ball_c] = 1.0
        img[0, self.H, 1 + self.paddle] = 1.0
        return img


def make_q_net(seed: int = 0) -> nn.Sequential:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Flatten(), nn.Linear((Catch.H + 2) * (Catch.W + 2), 64), nn.ReLU(),
                         NoisyLinear(64, Catch.N_ACTIONS, seed=seed))


def _set_noise(net, on: bool, rng=None):
    for m in net.modules():
        if isinstance(m, NoisyLinear):
            m.noisy = on
            if on and rng is not None:
                m.reset_noise(rng)


def evaluate_catch(net, episodes: int = 200, seed: int = 123) -> float:
    """Fraction of episodes the greedy, noise-free agent catches the ball."""
    env, caught = Catch(seed), 0.0
    _set_noise(net, False)
    for _ in range(episodes):
        obs, done = env.reset(), False
        while not done:
            with torch.no_grad():
                a = int(net(torch.as_tensor(obs[None])).argmax())
            obs, r, done = env.step(a)
        caught += r
    return caught / episodes


def train_catch(seed: int, augment: bool, steps: int = 4000, pad: int = 1,
                batch_size: int = 32, lr: float = 2e-3, gamma: float = 0.99):
    """Train a noisy DQN on Catch for a fixed number of environment steps."""
    rng = np.random.default_rng(seed)
    env = Catch(seed)
    net, target = make_q_net(seed), make_q_net(seed)
    target.load_state_dict(net.state_dict())
    _set_noise(target, False)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    buf = {k: [] for k in ("obs", "actions", "rewards", "next_obs", "terminated")}
    obs = env.reset()
    for t in range(steps):
        _set_noise(net, True, rng)
        with torch.no_grad():
            a = int(net(torch.as_tensor(obs[None])).argmax())
        obs2, r, done = env.step(a)
        for k, v in zip(buf, (obs, a, r, obs2, float(done))):
            buf[k].append(v)
        obs = env.reset() if done else obs2
        if t >= 200 and t % 2 == 0:
            idx = rng.integers(len(buf["obs"]), size=batch_size)
            batch = {k: np.asarray([v[i] for i in idx]) for k, v in buf.items()}
            _set_noise(net, True, rng)
            loss = augmented_td_loss(net, target, batch, gamma, pad if augment else 0, rng)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if t % 100 == 0:
            target.load_state_dict(net.state_dict())
            _set_noise(target, False)
    return net


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    red, green = np.zeros((2, 2, 3)), np.zeros((2, 2, 3))
    red[..., 0], green[..., 1] = 255, 255
    print("preprocess of red and green frames:", np.round(preprocess(red, green)[0, 0], 4))
    img = np.arange(25, dtype=np.float32).reshape(1, 1, 5, 5)
    print("a shifted 5x5 image:\n", random_shift(img, 1, rng)[0, 0])
    for augment in (False, True):
        net = train_catch(seed=0, augment=augment)
        print(f"augment={augment}: catch rate {evaluate_catch(net):.2f}")
