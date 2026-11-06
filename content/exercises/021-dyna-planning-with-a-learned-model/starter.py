"""Planning with a learned model: Dyna-Q and prioritized sweeping.

Fill in the functions below, in order. `dyna_maze` builds Sutton and Barto's
Dyna maze and is ready to use.

A model is a dict that maps a state-action pair (s, a) to what happened the
last time you tried it: (r, s2, terminated). Every world here is
deterministic, so one entry per pair is the whole truth.
"""

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
    """One Q-learning update, in place: move Q[s, a] toward r + gamma * max Q[s2].

    The target is r alone if terminated. Returns Q.
    """
    raise NotImplementedError("q_update")


def plan(Q, model, n_steps, alpha, gamma, rng):
    """Dyna's planning loop: replay remembered transitions as Q-learning updates.

    Args:
        Q: action values, shape (S, A), updated in place.
        model: a dict mapping (s, a) to (r, s2, terminated), one entry for
            each pair seen so far.
        n_steps: how many planning updates to make.
        alpha: the step size.
        gamma: the discount factor.
        rng: a np.random.Generator. Pick each pair uniformly from the keys
            of model, with pairs[int(rng.integers(len(pairs)))].

    Returns:
        Q.
    """
    raise NotImplementedError("plan")


def dyna_q(mdp, n_episodes, n_planning, alpha, epsilon, seed):
    """Tabular Dyna-Q on a deterministic TabularMDP, from Q = 0 and an empty model.

    After every real step: apply the Q-learning update, store
    model[(s, a)] = (r, s2, terminated), and call plan for n_planning steps.
    Act epsilon-greedily, breaking ties at random. Use one
    np.random.default_rng(seed) for acting, stepping with mdp.sample, and
    planning.

    Returns:
        (Q, steps): Q has shape (S, A), and steps has shape (n_episodes,),
        the number of real steps each episode took. Cap each episode at
        mdp.max_steps steps.
    """
    raise NotImplementedError("dyna_q")


def prioritized_sweeping(model, n_states, n_actions, gamma, theta, max_updates=1_000_000):
    """Plan with a known deterministic model, largest expected change first.

    Start from Q = 0. The priority of a pair is the size of the change its
    update would make, |r + gamma * max Q[s2] - Q[s, a]| (without the max
    term if terminated). Queue every pair whose priority exceeds theta. Then
    repeatedly pop the highest-priority pair, update it with alpha = 1, and
    recompute and queue each predecessor pair (every (ps, pa) whose model
    leads to s) if its priority exceeds theta. Stop when the queue is empty
    or after max_updates updates.

    Args:
        model: a dict mapping (s, a) to (r, s2, terminated).
        n_states: the number of states S.
        n_actions: the number of actions A.
        gamma: the discount factor.
        theta: the priority threshold.
        max_updates: a cap on the number of updates.

    Returns:
        (Q, updates): Q has shape (S, A), and updates is the number of Q
        updates made.
    """
    raise NotImplementedError("prioritized_sweeping")


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
