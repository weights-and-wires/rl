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
    """The sparse goal reward: 0 where the achieved goal equals the goal, -1 where it does not.

    Args:
        achieved: achieved goals, shape (..., n), or a single goal of shape (n,).
        goal: desired goals, the same shape as achieved.

    Returns:
        float array of shape (...), or a scalar for a single goal: 0.0 where all
        n bits match, -1.0 otherwise.
    """
    raise NotImplementedError("goal_reward")


def relabel(episode, strategy, k, rng):
    """Turn one episode into goal-conditioned transitions, with hindsight relabeling.

    Args:
        episode: dict with "states" (T + 1, n), "actions" (T,), and "goal" (n,).
            The goal that step t achieved is states[t + 1].
        strategy: "none", "final", "future", or "episode".
        k: relabeled goals per step, for "future" and "episode". "final" gives
            one per step and ignores k.
        rng: a numpy Generator.

    Returns:
        A list of transitions (s, a, r, s2, g, done), with s = states[t],
        a = actions[t], s2 = states[t + 1]. The first T are the original
        transitions with the original goal, in step order. For "none" that is
        the whole list. Otherwise relabeled copies follow, step by step: all of
        step t's copies come before step t + 1's. r = goal_reward(s2, g) and
        done is True exactly when r == 0.
        "final": g = states[T]. "future": k goals per step, each states[j] with
        j drawn uniformly from t + 1 .. T. "episode": k goals per step, each
        states[j] with j drawn uniformly from 1 .. T.
    """
    raise NotImplementedError("relabel")


def dqn_loss(q_net, target_net, batch, gamma):
    """The goal-conditioned DQN loss on one batch.

    Args:
        q_net, target_net: networks that map the joined [state, goal] input of
            shape (B, 2n) to action values of shape (B, n).
        batch: dict of tensors s, g, s2 of shape (B, n); a (B,) of int64; r and
            done of shape (B,).
        gamma: the discount.

    Returns:
        The mean over the batch of (Q(s, a, g) - y)^2, with
        y = r + gamma * (1 - done) * max over a2 of target_net([s2, g])[a2].
        The target y carries no gradient.
    """
    raise NotImplementedError("dqn_loss")


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
    """Train a goal-conditioned DQN on n-bit bit-flipping, with a relabeling strategy.

    Args:
        n: number of bits.
        strategy: passed to relabel: "none" (plain goal-conditioned DQN),
            "final", "future", or "episode".
        seed: seeds the environment, the network, and the sampling.

    Each episode: collect one with collect_episode (epsilon-greedy), add
    relabel(...) to a replay buffer (a list of transitions), then do n gradient
    steps, each on a batch of batch_size transitions drawn uniformly from the
    buffer with to_batch and dqn_loss. Copy q_net into target_net every
    sync_every gradient steps.

    Returns:
        (success_rate(env, q_net), q_net).
    """
    raise NotImplementedError("train_gcdqn")


def gcsl_targets(episode, rng):
    """Supervised targets for goal-conditioned supervised learning (GCSL) from one episode.

    Args:
        episode: dict with "states" (T + 1, n), "actions" (T,), and "goal" (n,).
        rng: a numpy Generator.

    Returns:
        (obs, goals, actions): obs = states[:T] of shape (T, n); goals of shape
        (T, n) where goals[t] = states[j] with j drawn uniformly from
        t + 1 .. T; and actions = actions of shape (T,). A policy trained to
        output actions[t] from (obs[t], goals[t]) imitates an episode that did
        reach that goal.
    """
    raise NotImplementedError("gcsl_targets")


if __name__ == "__main__":
    n = 12
    for strategy in ("none", "future"):
        wins, _ = train_gcdqn(n, strategy, seed=0)
        print(f"n = {n}, relabeling {strategy:6s}: success rate {wins:.2f}")
