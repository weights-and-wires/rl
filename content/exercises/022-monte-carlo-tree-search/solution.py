"""Reference solution for the Monte Carlo tree search exercise."""

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
    game = Nim()
    for stones in (5, 6, 7, 10):
        visits = mcts(game, stones, 2000, c=1.4, seed=0)
        print(f"UCT on Nim with {stones:2d} stones: visits for taking 1, 2, 3 = {visits.astype(int)}")
