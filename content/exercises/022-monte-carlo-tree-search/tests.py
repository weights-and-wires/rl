"""Checks for the Monte Carlo tree search exercise."""

import math

import numpy as np

from rlcheck import check, close, equal, fail, has, shape


@check("uct_scores matches the worked example and picks b, whose bonus outweighs its lower mean")
def test_uct_worked(sub):
    f = has(sub, "uct_scores")
    got = np.asarray(f(np.array([0.5, 0.4]), np.array([6, 4]), 10, math.sqrt(2)), dtype=float)
    shape(got, (2,), label="uct_scores output")
    want = np.array([0.5 + math.sqrt(2) * math.sqrt(math.log(10) / 6),
                     0.4 + math.sqrt(2) * math.sqrt(math.log(10) / 4)])
    close(got, want, atol=1e-12, label="scores for a and b")
    if int(np.argmax(got)) != 1:
        fail("UCT should select b, whose larger bonus outweighs its lower mean")


@check("uct_scores gives +inf to an untried action, and a bonus that grows with the parent's visits")
def test_uct_untried(sub):
    f = has(sub, "uct_scores")
    got = np.asarray(f(np.array([0.9, 0.0]), np.array([3, 0]), 3, 1.0), dtype=float)
    if not (np.isinf(got[1]) and got[1] > 0):
        fail(f"an unvisited action must score +inf so it gets tried first; got {got[1]}")
    small = float(f(np.array([0.5]), np.array([4]), 10, 1.0)[0])
    large = float(f(np.array([0.5]), np.array([4]), 1000, 1.0)[0])
    if not large > small:
        fail(f"with the same n(s, a) = 4, a parent with 1000 visits should give a larger "
             f"score than one with 10; got {large:.4f} and {small:.4f}")


@check("mcts finds the only winning move in the two smallest games, taking everything")
def test_mcts_tiny(sub):
    f = has(sub, "mcts")
    game = sub.Nim()
    for stones in (2, 3):
        visits = np.asarray(f(game, stones, 300, 1.4, seed=0), dtype=float)
        shape(visits, (stones,), label=f"visit counts with {stones} stones")
        if int(np.argmax(visits)) != stones - 1:
            fail(f"with {stones} stones, taking all {stones} wins at once. Expected most "
                 f"visits on the last action; got {visits.astype(int)}")


@check("mcts with UCT finds the winning Nim move, stones mod 4, from six starting piles")
def test_mcts_nim(sub):
    f = has(sub, "mcts")
    if not hasattr(sub, "Nim"):
        fail("your module defines no `Nim`; keep the class from the starter")
    game = sub.Nim()
    for stones in (5, 6, 7, 9, 10, 11):
        visits = np.asarray(f(game, stones, 1000, 1.4, seed=0), dtype=float)
        shape(visits, (3,), label=f"visit counts with {stones} stones")
        best = stones % 4
        share = visits[best - 1] / visits.sum()
        if int(np.argmax(visits)) != best - 1 or share < 0.5:
            fail(
                f"with {stones} stones the winning move takes {best}, leaving a multiple of 4. "
                f"Expected most visits there; got visits {visits.astype(int)} for taking 1, 2, 3. "
                "A search that forgets to flip the sign for the other player picks the wrong move."
            )
    again = np.asarray(f(game, 10, 1000, 1.4, seed=0), dtype=float)
    equal(again, np.asarray(f(game, 10, 1000, 1.4, seed=0), dtype=float), label="two runs with seed 0")
