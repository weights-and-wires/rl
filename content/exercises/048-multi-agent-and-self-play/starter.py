"""Multi-agent learning: regret matching, fictitious play, and self-play.

Rock-paper-scissors and the worked 2 by 2 matrix are zero-sum: the row
player wins x @ A @ y, and the column player loses that amount. The bridge
is general-sum. BRIDGE_ROW[i, j] is the row player's payoff and
BRIDGE_COL[i, j] is the column player's. Action 0 is yield and action 1
is push.

`ROCK_PAPER_SCISSORS`, `BRIDGE_ROW`, `BRIDGE_COL`, and `nim_legal` are
written for you. Fill in the rest.
"""

import numpy as np

ROCK_PAPER_SCISSORS = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])

# One-lane bridge. Action 0 is yield and action 1 is push. BRIDGE_ROW[i, j]
# is the row player's payoff, and BRIDGE_COL[i, j] is the column player's.
BRIDGE_ROW = np.array([[2.0, 1.0], [3.0, 0.0]])
BRIDGE_COL = np.array([[2.0, 3.0], [1.0, 0.0]])


def exploitability(A: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """How much the two players could gain, together, by best-responding.

    max_i (A @ y)_i - min_j (x @ A)_j. It is zero exactly at a Nash
    equilibrium and positive everywhere else.
    """
    raise NotImplementedError("exploitability")


def regret_matching_strategy(regrets: np.ndarray) -> np.ndarray:
    """Play each action in proportion to its positive cumulative regret.

    Returns max(regrets, 0) normalised to sum to one, or the uniform strategy
    when no regret is positive.
    """
    raise NotImplementedError("regret_matching_strategy")


def regret_matching(A: np.ndarray, iters: int, x0: np.ndarray, y0: np.ndarray) -> dict:
    """Both players run regret matching against each other for `iters` rounds.

    Round 0 plays x0 and y0. Every later round plays
    regret_matching_strategy of each player's cumulative regrets. After each
    round, with u_x = A @ y the row player's payoff for each row and
    u_y = -(x @ A) the column player's payoff for each column:
        regret_x += u_x - x @ u_x
        regret_y += u_y - y @ u_y

    Returns:
        A dict with "x_avg" and "y_avg" (the averages of the strategies played
        over all rounds), "x_last" and "y_last" (the last round's strategies),
        and "regret_x" and "regret_y" (the cumulative regrets).
    """
    raise NotImplementedError("regret_matching")


def fictitious_play(A: np.ndarray, iters: int, first_x: int = 0, first_y: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Each round, both players best-respond to the other's empirical mixture so far.

    Round 0 plays the pure actions first_x and first_y. In each later round,
    the row player plays argmax of A @ (column counts / total), and the column
    player plays argmin of (row counts / total) @ A, both computed from the
    counts before this round. Break ties by the lowest index (np.argmax and
    np.argmin do this).

    Returns:
        (x, y): each player's action counts over the `iters` rounds, divided by iters.
    """
    raise NotImplementedError("fictitious_play")


def bimatrix_best_responses(
    A: np.ndarray, B: np.ndarray, y: np.ndarray, x: np.ndarray
) -> tuple[int, int]:
    """Pure best responses for a general-sum game.

    A and B have shape (n, m). A is the row player's payoff and B is the
    column player's. y has shape (m,) and x has shape (n,). A vector of
    counts works in place of probabilities.

    Returns:
        (i, j). i maximizes A @ y and j maximizes x @ B. A tie returns the
        lowest index.
    """
    raise NotImplementedError("bimatrix_best_responses")


def general_sum_fictitious_play(
    A: np.ndarray,
    B: np.ndarray,
    iters: int,
    first_x: int = 0,
    first_y: int = 0,
) -> dict:
    """Fictitious play when each player has a payoff table.

    Plays `iters` rounds. Round 0 plays the pure actions first_x and
    first_y. Each later round, both players best-respond to the opponent's
    counts so far, then both counts increase. Ties break to the lowest index.

    Returns:
        A dict with "x_avg" and "y_avg" (the counts divided by iters, shapes
        (n,) and (m,)) and "last_x" and "last_y" (the actions in the final
        round).
    """
    raise NotImplementedError("general_sum_fictitious_play")


def nim_legal(n_max: int, max_take: int = 3) -> np.ndarray:
    """legal[n, k - 1] is True when you may take k sticks from a pile of n."""
    n = np.arange(n_max + 1)[:, None]
    k = np.arange(1, max_take + 1)[None, :]
    return k <= n


def nim_self_play(
    n_max: int,
    episodes: int,
    alpha: float,
    epsilon: float,
    seed: int,
    max_take: int = 3,
) -> np.ndarray:
    """Learn Nim by playing against yourself with one shared Q table.

    The game: a pile of n sticks, players alternate taking 1 to max_take
    sticks, and whoever takes the last stick wins. Q[n, k - 1] is the value
    of taking k sticks from a pile of n, for the player about to move, with
    +1 for a win and -1 for a loss.

    Make rng = np.random.default_rng(seed). Each episode starts from a pile
    drawn with rng.integers(1, n_max + 1). At each move, with probability
    epsilon take a random legal move; otherwise a greedy legal move, breaking
    ties at random. Update with the negamax target:
        +1 if the move empties the pile, else -max over legal k2 of Q[n - k, k2 - 1],
    since the opponent moves next. Then Q[n, k - 1] moves alpha of the way
    toward the target.

    Returns:
        Q, shape (n_max + 1, max_take). Row 0 and illegal entries stay 0.
    """
    raise NotImplementedError("nim_self_play")


if __name__ == "__main__":
    A = ROCK_PAPER_SCISSORS
    rock = np.array([1.0, 0.0, 0.0])
    print("regret matching on rock-paper-scissors, starting from rock against rock")
    for T in (10, 100, 1000, 10000):
        out = regret_matching(A, T, rock, rock)
        print(f"  T={T:5d}  average x={np.round(out['x_avg'], 3)}  "
              f"exploitability of averages={exploitability(A, out['x_avg'], out['y_avg']):.4f}  "
              f"of last strategies={exploitability(A, out['x_last'], out['y_last']):.4f}")

    G = np.array([[3.0, -1.0], [-2.0, 1.0]])
    out = regret_matching(G, 20000, np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    print("\n2x2 game: average x", np.round(out["x_avg"], 4), "(Nash 3/7, 4/7);",
          "average y", np.round(out["y_avg"], 4), "(Nash 2/7, 5/7)")
    x, y = fictitious_play(A, 5000)
    print("fictitious play on RPS after 5000 rounds:", np.round(x, 3), np.round(y, 3))

    print("\nbridge, both players open on yield (mixed Nash puts weight 1/2 on yield)")
    for rounds in (1, 2, 3, 4, 8):
        out = general_sum_fictitious_play(BRIDGE_ROW, BRIDGE_COL, rounds, 0, 0)
        print(f"  rounds={rounds}  average x={np.round(out['x_avg'], 3)}  "
              f"last=({out['last_x']}, {out['last_y']})")

    Q = nim_self_play(n_max=15, episodes=3000, alpha=0.5, epsilon=0.2, seed=0)
    legal = nim_legal(15)
    print("\nNim self-play: greedy move for each pile (the theory says take n mod 4)")
    for n in range(1, 16):
        k = int(np.argmax(np.where(legal[n], Q[n], -np.inf))) + 1
        print(f"  pile {n:2d}: take {k}   value {Q[n, legal[n]].max():+.2f}   theory {n % 4 or '-'}")
