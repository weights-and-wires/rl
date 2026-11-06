"""Reference solution for the Dyna and prioritized sweeping exercise."""

import heapq

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
