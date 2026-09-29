"""Reference solution for the planning, learning, and search exercise."""

import heapq
import math

import numpy as np

import rlenvs

# Sutton and Barto's Dyna maze (Figure 8.2): 6 by 9, start on the left, goal
# in the top-right corner, reward 1 on reaching the goal and 0 otherwise.
DYNA_MAZE = [
    ".......#G",
    "..#....#.",
    "S.#....#.",
    "..#......",
    ".....#...",
    ".........",
]


def dyna_maze(gamma: float = 0.95) -> rlenvs.TabularMDP:
    """The Dyna maze as a deterministic `TabularMDP`."""
    return rlenvs.gridworld(DYNA_MAZE, gamma=gamma, step_reward=0.0, goal_reward=1.0, max_steps=2000)


class Nim:
    """Take-away Nim: a pile of stones, and each turn a player removes 1 to 3.

    The player who takes the last stone wins. A state is the number of stones
    left, and it's always the turn of "the player to move". Values are from
    that player's point of view: +1 for a win and -1 for a loss.
    """

    def __init__(self, max_take: int = 3):
        self.max_take = max_take

    def actions(self, stones: int) -> list[int]:
        """The legal moves: how many stones to take."""
        return list(range(1, min(self.max_take, stones) + 1))

    def next_state(self, stones: int, take: int) -> int:
        return stones - take

    def is_terminal(self, stones: int) -> bool:
        return stones == 0

    def terminal_value(self, stones: int) -> float:
        """At an empty pile the opponent took the last stone, so the player to move has lost."""
        return -1.0

    def rollout(self, stones: int, rng: np.random.Generator) -> float:
        """Play uniformly random moves to the end. Returns the result for the player to move now."""
        sign = 1.0
        while stones > 0:
            stones -= int(rng.integers(1, min(self.max_take, stones) + 1))
            sign = -sign
        # The loop flips the sign once per move. The player who just moved took
        # the last stone and won; that player is "to move now" if sign is -1.
        return -sign


def q_update(Q, s, a, r, s2, terminated, alpha, gamma):
    target = r + gamma * Q[s2].max() * (not terminated)
    Q[s, a] += alpha * (target - Q[s, a])
    return Q


def plan(Q, model, n_steps, alpha, gamma, rng):
    pairs = list(model)
    for _ in range(n_steps):
        s, a = pairs[int(rng.integers(len(pairs)))]
        r, s2, terminated = model[(s, a)]
        q_update(Q, s, a, r, s2, terminated, alpha, gamma)
    return Q


def dyna_q(mdp, n_episodes, n_planning, alpha, epsilon, seed):
    rng = np.random.default_rng(seed)
    Q = np.zeros((mdp.n_states, mdp.n_actions))
    model = {}
    steps = np.zeros(n_episodes, dtype=int)
    for i in range(n_episodes):
        s = int(rng.choice(mdp.n_states, p=mdp.mu0))
        while steps[i] < mdp.max_steps:
            if rng.random() < epsilon:
                a = int(rng.integers(mdp.n_actions))
            else:
                a = int(rng.choice(np.flatnonzero(Q[s] == Q[s].max())))
            s2, r = mdp.sample(s, a, rng)
            done = bool(mdp.terminal[s2])
            steps[i] += 1
            q_update(Q, s, a, r, s2, done, alpha, mdp.gamma)
            model[(s, a)] = (r, s2, done)
            plan(Q, model, n_planning, alpha, mdp.gamma, rng)
            s = s2
            if done:
                break
    return Q, steps


def prioritized_sweeping(model, n_states, n_actions, gamma, theta, max_updates=1_000_000):
    Q = np.zeros((n_states, n_actions))
    predecessors = {}
    for (s, a), (_, s2, _) in model.items():
        predecessors.setdefault(s2, set()).add((s, a))

    def priority(s, a):
        r, s2, terminated = model[(s, a)]
        return abs(r + gamma * Q[s2].max() * (not terminated) - Q[s, a])

    # A max-heap by negated priority. `best` holds each queued pair's current
    # priority, so a stale, lower entry can be skipped when it surfaces.
    heap, best = [], {}

    def push(s, a):
        p = priority(s, a)
        if p > theta and p > best.get((s, a), 0.0):
            best[(s, a)] = p
            heapq.heappush(heap, (-p, s, a))

    for s, a in sorted(model):
        push(s, a)
    updates = 0
    while heap and updates < max_updates:
        neg_p, s, a = heapq.heappop(heap)
        if best.get((s, a)) != -neg_p:
            continue
        del best[(s, a)]
        q_update(Q, s, a, *model[(s, a)], alpha=1.0, gamma=gamma)
        updates += 1
        for ps, pa in sorted(predecessors.get(s, ())):
            push(ps, pa)
    return Q, updates


def uct_scores(q, n, n_parent, c):
    q = np.asarray(q, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        bonus = c * np.sqrt(math.log(n_parent) / n) if n_parent > 0 else np.zeros_like(q)
        scores = q + bonus
    return np.where(n == 0, np.inf, scores)


def mcts(game, root, n_simulations, c, seed):
    rng = np.random.default_rng(seed)
    # stats[state] = (visit counts, total values), one entry per legal action,
    # with values from the point of view of the player to move in `state`.
    stats = {}
    for _ in range(n_simulations):
        s, path = root, []
        # Selection: descend while the state is already in the tree.
        while s in stats and not game.is_terminal(s):
            N, W = stats[s]
            q = np.divide(W, N, out=np.zeros_like(W), where=N > 0)
            scores = uct_scores(q, N, N.sum(), c)
            i = int(rng.choice(np.flatnonzero(scores == scores.max())))
            path.append((s, i))
            s = game.next_state(s, game.actions(s)[i])
        # Expansion and simulation.
        if game.is_terminal(s):
            value = game.terminal_value(s)
        else:
            k = len(game.actions(s))
            stats[s] = (np.zeros(k), np.zeros(k))
            value = game.rollout(s, rng)
        # Backup: each step up the path switches to the other player.
        for state, i in reversed(path):
            value = -value
            N, W = stats[state]
            N[i] += 1
            W[i] += value
    return stats[root][0] if root in stats else np.zeros(len(game.actions(root)))


if __name__ == "__main__":
    maze = dyna_maze()
    for n in (0, 5, 50):
        steps = np.mean([dyna_q(maze, 20, n, 0.1, 0.1, seed)[1] for seed in range(3)], axis=0)
        print(f"Dyna-Q, n = {n:2d}: steps per episode {np.round(steps[[0, 1, 2, 4, 9, 19]]).astype(int)} (episodes 1, 2, 3, 5, 10, 20)")

    # Record every transition of the maze, then plan with the full model.
    model = {}
    for s in range(maze.n_states):
        if maze.terminal[s]:
            continue
        for a in range(maze.n_actions):
            s2, r = maze.sample(s, a, np.random.default_rng(0))
            model[(s, a)] = (r, s2, bool(maze.terminal[s2]))
    Q, updates = prioritized_sweeping(model, maze.n_states, maze.n_actions, maze.gamma, theta=1e-5)
    q_star = rlenvs.q_from_v(maze, rlenvs.value_iteration(maze))
    print(f"prioritized sweeping: {updates} updates, max error {np.abs(Q - q_star).max():.2e}")

    game = Nim()
    for stones in (5, 6, 7, 10):
        visits = mcts(game, stones, 2000, c=1.4, seed=0)
        print(f"UCT on Nim with {stones:2d} stones: visits for taking 1, 2, 3 = {visits.astype(int)}")
