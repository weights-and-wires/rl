"""Monte Carlo tree search with UCT, on the game of Nim.

Fill in `uct_scores` and `mcts`. `Nim` is a tiny two-player game and is ready
to use. A state is the number of stones left, and it is always the turn of
"the player to move".
"""

import math

import numpy as np


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
    game = Nim()
    for stones in (5, 6, 7, 10):
        visits = mcts(game, stones, 2000, c=1.4, seed=0)
        print(f"UCT on Nim with {stones:2d} stones: visits for taking 1, 2, 3 = {visits.astype(int)}")
