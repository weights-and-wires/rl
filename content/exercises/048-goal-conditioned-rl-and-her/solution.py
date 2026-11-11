"""Goal-conditioned RL: sparse goal rewards, hindsight relabeling, and GCSL targets.

Fill in the functions below, in order. You work on the bit-flipping task. The
state is a string of n bits, the goal is another string of n bits, and an action
flips one bit. The reward is 0 when the state equals the goal and -1 otherwise.
You write the reward function, the hindsight relabeling that turns failed
episodes into successes, the goal-conditioned DQN loss, and a training loop that
runs with and without relabeling. Last, you build the targets for goal-conditioned
supervised learning (GCSL).

An episode is a dict: "states" has shape (T + 1, n) and holds s_0 .. s_T, "actions"
has shape (T,), and "goal" has shape (n,). In bit-flipping, the goal a state
achieves is the state itself, so the goal that step t reached is states[t + 1].
"""

import numpy as np
import torch

torch.set_num_threads(1)


class BitFlipEnv:
    """n bits, n actions (action i flips bit i), at most n steps per episode.

    reset() draws a random start and a random goal and returns (state, goal).
    step(a) returns (state, reward, done). The reward is 0 on reaching the goal
    and -1 otherwise, and done is True on success or after n steps.
    """

    def __init__(self, n, seed=0):
        self.n = n
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.t = 0
        self.state = self.rng.integers(0, 2, self.n).astype(np.float32)
        self.goal = self.rng.integers(0, 2, self.n).astype(np.float32)
        return self.state.copy(), self.goal.copy()

    def step(self, a):
        self.state[a] = 1.0 - self.state[a]
        self.t += 1
        success = bool(np.array_equal(self.state, self.goal))
        return self.state.copy(), (0.0 if success else -1.0), success or self.t >= self.n


def make_qnet(n, seed=0, hidden=64):
    """The Q network Q(s, a, g): input is the state and goal joined, output is n action values."""
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(2 * n, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, n),
    )


def goal_reward(achieved, goal):
    achieved = np.asarray(achieved)
    goal = np.asarray(goal)
    return np.where(np.all(achieved == goal, axis=-1), 0.0, -1.0)


def relabel(episode, strategy, k, rng):
    states, actions, goal = episode["states"], episode["actions"], episode["goal"]
    T = len(actions)

    def make(t, g):
        s2 = states[t + 1]
        r = float(goal_reward(s2, g))
        return (states[t], int(actions[t]), r, s2, g, r == 0.0)

    out = [make(t, goal) for t in range(T)]
    if strategy == "none":
        return out
    for t in range(T):
        if strategy == "final":
            out.append(make(t, states[T]))
        elif strategy == "future":
            for j in rng.integers(t + 1, T + 1, size=k):
                out.append(make(t, states[j]))
        elif strategy == "episode":
            for j in rng.integers(1, T + 1, size=k):
                out.append(make(t, states[j]))
        else:
            raise ValueError(f"unknown strategy {strategy!r}")
    return out


def dqn_loss(q_net, target_net, batch, gamma):
    s, a, r, s2, g, done = (batch[key] for key in ("s", "a", "r", "s2", "g", "done"))
    q = q_net(torch.cat([s, g], dim=1)).gather(1, a[:, None])[:, 0]
    with torch.no_grad():
        best = target_net(torch.cat([s2, g], dim=1)).max(dim=1).values
        y = r + gamma * (1.0 - done) * best
    return ((q - y) ** 2).mean()


def to_batch(buffer, idx):
    """Stack the transitions buffer[i] for i in idx into a batch dict of float32 tensors (a is int64)."""
    rows = [buffer[i] for i in idx]
    f = lambda j: torch.tensor(np.array([row[j] for row in rows]), dtype=torch.float32)
    return {"s": f(0), "a": torch.tensor([row[1] for row in rows]), "r": f(2),
            "s2": f(3), "g": f(4), "done": f(5)}


def greedy_action(q_net, s, g):
    """The action with the highest Q(s, a, g)."""
    with torch.no_grad():
        return int(q_net(torch.tensor(np.concatenate([s, g]))[None])[0].argmax())


def collect_episode(env, q_net, epsilon, rng):
    """Run one epsilon-greedy episode on a fresh start and goal. Returns an episode dict."""
    s, g = env.reset()
    states, actions = [s], []
    done = False
    while not done:
        a = int(rng.integers(env.n)) if rng.random() < epsilon else greedy_action(q_net, s, g)
        s, _, done = env.step(a)
        states.append(s)
        actions.append(a)
    return {"states": np.array(states), "actions": np.array(actions), "goal": g}


def success_rate(env, q_net, episodes=100):
    """The fraction of greedy episodes, each on a fresh random goal, that reach the goal."""
    rng = np.random.default_rng(12345)
    wins = 0
    for _ in range(episodes):
        ep = collect_episode(env, q_net, 0.0, rng)
        wins += bool(np.array_equal(ep["states"][-1], ep["goal"]))
    return wins / episodes


def train_gcdqn(n, strategy, seed, episodes=500, k=4, gamma=0.9, lr=1e-3, batch_size=64,
                epsilon=0.2, sync_every=50):
    rng = np.random.default_rng(seed)
    env = BitFlipEnv(n, seed=seed)
    q_net = make_qnet(n, seed=seed)
    target_net = make_qnet(n, seed=seed)
    target_net.load_state_dict(q_net.state_dict())
    opt = torch.optim.Adam(q_net.parameters(), lr=lr)
    buffer, updates = [], 0
    for _ in range(episodes):
        ep = collect_episode(env, q_net, epsilon, rng)
        buffer.extend(relabel(ep, strategy, k, rng))
        for _ in range(n):
            batch = to_batch(buffer, rng.integers(0, len(buffer), batch_size))
            loss = dqn_loss(q_net, target_net, batch, gamma)
            opt.zero_grad()
            loss.backward()
            opt.step()
            updates += 1
            if updates % sync_every == 0:
                target_net.load_state_dict(q_net.state_dict())
    return success_rate(env, q_net), q_net


def gcsl_targets(episode, rng):
    states, actions = episode["states"], episode["actions"]
    T = len(actions)
    j = np.array([rng.integers(t + 1, T + 1) for t in range(T)])
    return states[:T], states[j], actions


if __name__ == "__main__":
    n = 12
    for strategy in ("none", "future"):
        wins, _ = train_gcdqn(n, strategy, seed=0)
        print(f"n = {n}, relabeling {strategy:6s}: success rate {wins:.2f}")
