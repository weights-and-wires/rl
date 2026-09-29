"""Checks for the deep Q-network exercise."""

import numpy as np
import torch
import torch.nn as nn

import rlenvs
from rlcheck import almost, check, close, equal, fail, finite_difference, has, shape, timed

torch.set_num_threads(1)


def _fill(buffer, start, stop, obs_dim=3):
    """Add transitions start, ..., stop - 1, whose every field encodes the index i."""
    for i in range(start, stop):
        buffer.add(np.full(obs_dim, float(i)), i % 2, float(i), np.full(obs_dim, i + 0.5), i % 3 == 0)


def _net(seed, obs_dim=3, n_actions=2):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(obs_dim, 8), nn.ReLU(), nn.Linear(8, n_actions))


@check("ReplayBuffer keeps the most recent `capacity` transitions, each stored whole")
def test_buffer_ring(sub):
    ReplayBuffer = has(sub, "ReplayBuffer")
    buffer = ReplayBuffer(capacity=5, obs_dim=3, seed=0)
    _fill(buffer, 0, 3)
    equal(len(buffer), 3, label="len after 3 adds")
    _fill(buffer, 3, 7)
    equal(len(buffer), 5, label="len after 7 adds into capacity 5")
    s, a, r, s2, term = buffer.sample(2000)
    shape(s, (2000, 3), label="sampled states")
    shape(a, (2000,), label="sampled actions")
    seen = sorted(set(np.asarray(r, dtype=float).astype(int).tolist()))
    if seen != [2, 3, 4, 5, 6]:
        fail(f"after 7 adds into capacity 5 the buffer should hold transitions 2 to 6 "
             f"(the two oldest overwritten); sampled rewards were {seen}")
    r = np.asarray(r, dtype=float)
    close(np.asarray(s, dtype=float)[:, 0], r, label="state matches its own reward")
    close(np.asarray(s2, dtype=float)[:, 0], r + 0.5, label="next state matches its own reward")
    equal(np.asarray(a).astype(int), r.astype(int) % 2, label="action matches its own reward")
    close(np.asarray(term, dtype=float), (r.astype(int) % 3 == 0).astype(float),
          label="terminated flag matches its own reward")


@check("ReplayBuffer samples uniformly: a chi-squared test over 20,000 draws")
def test_buffer_uniform(sub):
    ReplayBuffer = has(sub, "ReplayBuffer")
    buffer = ReplayBuffer(capacity=10, obs_dim=3, seed=0)
    _fill(buffer, 0, 10)
    counts = np.zeros(10)
    for _ in range(20):
        r = np.asarray(buffer.sample(1000)[2], dtype=float).astype(int)
        counts += np.bincount(r, minlength=10)[:10]
    expected = 2000.0
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    # Nine degrees of freedom: mean 9, standard deviation about 4.2.
    if chi2 > 40:
        fail(f"sample counts per slot should be near 2000 each; got {counts.astype(int).tolist()}, "
             f"chi-squared {chi2:.1f} against a limit of 40")


@check("dqn_targets gives r + gamma * max Q_target(s2) and drops the bootstrap when terminated")
def test_dqn_targets(sub):
    dqn_targets = has(sub, "dqn_targets")
    next_q = torch.tensor([[2.5, 1.5, 2.0], [2.5, 1.5, 2.0], [-1.0, -3.0, -2.0]], dtype=torch.float64)
    y = dqn_targets(torch.tensor([1.0, 1.0, 0.5], dtype=torch.float64), next_q,
                    torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), 0.9)
    # 1 + 0.9 * 2.5 = 3.25; terminal gives r = 1; 0.5 + 0.9 * (-1) = -0.4.
    close(torch.as_tensor(y).detach().numpy(), np.array([3.25, 1.0, -0.4]), atol=1e-12,
          label="targets")


@check("huber_loss is quadratic inside kappa, linear outside, and its gradient is clip(u)/n")
def test_huber(sub):
    huber_loss = has(sub, "huber_loss")
    u = torch.tensor([0.5, -3.0, 1.0, 2.0], dtype=torch.float64)
    # 0.125, 2.5, 0.5, and 1.5 average to 1.15625 with kappa = 1.
    almost(float(huber_loss(u, 1.0)), 1.15625, 1e-12, label="huber_loss([0.5, -3, 1, 2], kappa=1)")
    # With kappa = 2: 0.125, 2 * (3 - 1) = 4, 0.5, 2, mean 1.65625.
    almost(float(huber_loss(u, 2.0)), 1.65625, 1e-12, label="huber_loss(..., kappa=2)")

    rng = np.random.default_rng(0)
    x0 = rng.normal(scale=2.0, size=8)
    x0 = x0[np.abs(np.abs(x0) - 1.0) > 0.05]  # stay off the kink at |u| = kappa

    def f(x):
        return float(huber_loss(torch.tensor(x, dtype=torch.float64), 1.0))

    t = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    huber_loss(t, 1.0).backward()
    close(t.grad.numpy(), np.clip(x0, -1, 1) / len(x0), atol=1e-12, label="gradient vs clip(u)/n")
    close(t.grad.numpy(), finite_difference(f, x0), atol=1e-7, label="gradient vs finite differences")


@check("dqn_loss is the Huber loss of y - Q(s, a), and no gradient reaches the target network")
def test_dqn_loss(sub):
    dqn_loss = has(sub, "dqn_loss")
    q_net, target_net = _net(1), _net(2)
    rng = np.random.default_rng(3)
    s = rng.normal(size=(6, 3)).astype(np.float32)
    s2 = rng.normal(size=(6, 3)).astype(np.float32)
    a = rng.integers(0, 2, size=6)
    r = rng.normal(size=6).astype(np.float32)
    term = np.array([0, 1, 0, 0, 1, 0], dtype=np.float32)
    loss = dqn_loss(q_net, target_net, (s, a, r, s2, term), 0.9)
    with torch.no_grad():
        y = torch.tensor(r) + 0.9 * (1 - torch.tensor(term)) * target_net(torch.tensor(s2)).max(1).values
        q_sa = q_net(torch.tensor(s))[torch.arange(6), torch.tensor(a)]
        u = y - q_sa
        want = torch.where(u.abs() <= 1, 0.5 * u**2, u.abs() - 0.5).mean()
    almost(float(loss), float(want), 1e-6, label="dqn_loss against the hand-built loss")
    loss.backward()
    if any(p.grad is not None and p.grad.abs().sum() > 0 for p in target_net.parameters()):
        fail("the target network received a gradient; compute the target under torch.no_grad()")
    if all(p.grad is None for p in q_net.parameters()):
        fail("the online network received no gradient from dqn_loss")


@check("soft_update moves every target parameter a fraction tau toward the online network")
def test_soft_update(sub):
    soft_update = has(sub, "soft_update")
    q_net, target_net = _net(1), _net(2)
    before = [p.detach().clone() for p in target_net.parameters()]
    online = [p.detach().clone() for p in q_net.parameters()]
    soft_update(target_net, q_net, 0.25)
    for b, o, p in zip(before, online, target_net.parameters()):
        close(p.detach().numpy(), (0.75 * b + 0.25 * o).numpy(), atol=1e-6, label="tau = 0.25")
    soft_update(target_net, q_net, 1.0)
    for o, p in zip(online, target_net.parameters()):
        close(p.detach().numpy(), o.numpy(), atol=1e-7, label="tau = 1 copies")
    with torch.no_grad():
        for p in q_net.parameters():
            p.add_(1.0)
    for o, p in zip(online, target_net.parameters()):
        close(p.detach().numpy(), o.numpy(), atol=1e-7,
              label="target after the online net changes (it must be a copy, not shared)")


@check("epsilon_at decays linearly and select_action is greedy at 0 and uniform at 1")
def test_exploration(sub):
    epsilon_at = has(sub, "epsilon_at")
    select_action = has(sub, "select_action")
    make_q_network = has(sub, "make_q_network")
    got = [epsilon_at(t, 1.0, 0.05, 1000) for t in (0, 500, 1000, 5000)]
    close(got, [1.0, 0.525, 0.05, 0.05], atol=1e-12, label="epsilon at steps 0, 500, 1000, 5000")

    torch.manual_seed(0)
    q_net = make_q_network(4, 3, 16)
    n_params = sum(p.numel() for p in q_net.parameters())
    equal(n_params, 4 * 16 + 16 + 16 * 16 + 16 + 16 * 3 + 3,
          label="parameters of make_q_network(4, 3, 16), two hidden layers of 16")
    obs = np.array([0.1, -0.2, 0.3, 0.05], dtype=np.float32)
    with torch.no_grad():
        best = int(q_net(torch.tensor(obs)).argmax())
    rng = np.random.default_rng(0)
    greedy = {select_action(q_net, obs, 0.0, rng) for _ in range(20)}
    equal(sorted(greedy), [best], label="actions chosen at epsilon = 0")
    counts = np.bincount([select_action(q_net, obs, 1.0, rng) for _ in range(3000)], minlength=3)
    if np.any(np.abs(counts - 1000) > 150):
        fail(f"at epsilon = 1 each of 3 actions should come up about 1000 times in 3000; got {counts.tolist()}")


@check("train_dqn balances CartPole: greedy mean return above 100 over 10 episodes")
def test_train_dqn(sub):
    train_dqn = has(sub, "train_dqn")
    env = rlenvs.CartPole(max_steps=500)
    with timed("train_dqn, 30,000 steps"):
        q_net = train_dqn(env, seed=0)
    eval_env = rlenvs.CartPole(max_steps=500)
    returns = []
    for episode in range(10):
        obs, total, done = eval_env.reset(seed=1000 + episode), 0.0, False
        while not done:
            with torch.no_grad():
                action = int(q_net(torch.as_tensor(obs, dtype=torch.float32)).argmax())
            obs, reward, terminated, truncated = eval_env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
    mean = float(np.mean(returns))
    # A random policy scores about 22. The reference solution scores 283 on
    # this seed, and at least 104 on each of 36 seeds.
    if mean < 100:
        fail(f"the greedy policy should average above 100 over 10 episodes; it averaged "
             f"{mean:.1f}, with returns {[int(x) for x in returns]}")
