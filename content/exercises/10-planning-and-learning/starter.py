"""Dyna-Q, prioritized sweeping, and Monte Carlo tree search with UCT.

Fill in the functions below, in order. `dyna_maze` builds Sutton and Barto's
Dyna maze, and `Nim` is a tiny two-player game for tree search. Both are
ready to use.

A model is a dict that maps a state-action pair (s, a) to what happened the
last time you tried it: (r, s2, terminated). Every world here is
deterministic, so one entry per pair is the whole truth.
"""

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


def uct_scores(q, n, n_parent, c):
    """The UCT score of every action at one tree node.

    Args:
        q: shape (A,), the mean value of each action so far.
        n: shape (A,), how many times each action has been tried.
        n_parent: the node's total visit count, N(s).
        c: the exploration constant.

    Returns:
        Shape (A,): q + c * sqrt(ln(n_parent) / n), and +inf for any action
        with n == 0.
    """
    raise NotImplementedError("uct_scores")


def mcts(game, root, n_simulations, c, seed):
    """Monte Carlo tree search with UCT for a two-player game such as Nim.

    Keep a table from state to (N, W): two arrays with one entry per legal
    action, the visit counts and total values from the point of view of the
    player to move in that state. Each simulation:

    1. Selection: from the root, while the state is in the table and not
       terminal, pick the action with the highest uct_scores (ties at
       random) and move to the next state.
    2. Expansion: if the state is not terminal, add it to the table with
       zero counts, and get a value from game.rollout(state, rng). If it is
       terminal, the value is game.terminal_value(state).
    3. Backup: walk the path from the leaf back to the root. At each step,
       negate the value first, because the parent's player is the opponent,
       then add 1 to N and the value to W for the action taken.

    Args:
        game: a Nim instance.
        root: the starting state, not terminal.
        n_simulations: how many simulations to run.
        c: the exploration constant.
        seed: seed for np.random.default_rng, the only source of randomness.

    Returns:
        The visit counts at the root, shape (len(game.actions(root)),).
    """
    raise NotImplementedError("mcts")


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
