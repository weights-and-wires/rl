"""Reference solution for the multi-agent learning and self-play exercise."""

import numpy as np

ROCK_PAPER_SCISSORS = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
BRIDGE_ROW = np.array([[2.0, 1.0], [3.0, 0.0]])
BRIDGE_COL = np.array([[2.0, 3.0], [1.0, 0.0]])


def exploitability(A: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    A = np.asarray(A, dtype=float)
    return float(np.max(A @ y) - np.min(x @ A))


def regret_matching_strategy(regrets: np.ndarray) -> np.ndarray:
    positive = np.maximum(regrets, 0.0)
    total = positive.sum()
    if total > 0:
        return positive / total
    return np.full(len(regrets), 1.0 / len(regrets))


def regret_matching(A: np.ndarray, iters: int, x0: np.ndarray, y0: np.ndarray) -> dict:
    A = np.asarray(A, dtype=float)
    n, m = A.shape
    regret_x, regret_y = np.zeros(n), np.zeros(m)
    sum_x, sum_y = np.zeros(n), np.zeros(m)
    x, y = np.asarray(x0, dtype=float), np.asarray(y0, dtype=float)
    for t in range(iters):
        if t > 0:
            x, y = regret_matching_strategy(regret_x), regret_matching_strategy(regret_y)
        sum_x += x
        sum_y += y
        u_x = A @ y        # the row player's payoff for each pure action
        u_y = -(x @ A)     # the column player's payoff: minus what the row player wins
        regret_x += u_x - x @ u_x
        regret_y += u_y - y @ u_y
    return {
        "x_avg": sum_x / iters,
        "y_avg": sum_y / iters,
        "x_last": x,
        "y_last": y,
        "regret_x": regret_x,
        "regret_y": regret_y,
    }


def fictitious_play(A: np.ndarray, iters: int, first_x: int = 0, first_y: int = 0) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=float)
    n, m = A.shape
    count_x, count_y = np.zeros(n), np.zeros(m)
    count_x[first_x] += 1
    count_y[first_y] += 1
    for _ in range(iters - 1):
        # Both players best-respond to the other's empirical mixture so far.
        a = int(np.argmax(A @ (count_y / count_y.sum())))
        b = int(np.argmin((count_x / count_x.sum()) @ A))
        count_x[a] += 1
        count_y[b] += 1
    return count_x / iters, count_y / iters


def bimatrix_best_responses(
    A: np.ndarray, B: np.ndarray, y: np.ndarray, x: np.ndarray
) -> tuple[int, int]:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    return int(np.argmax(A @ y)), int(np.argmax(np.asarray(x, dtype=float) @ B))


def general_sum_fictitious_play(
    A: np.ndarray,
    B: np.ndarray,
    iters: int,
    first_x: int = 0,
    first_y: int = 0,
) -> dict:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n, m = A.shape
    count_x, count_y = np.zeros(n), np.zeros(m)
    count_x[first_x] += 1
    count_y[first_y] += 1
    last_x, last_y = int(first_x), int(first_y)
    for _ in range(iters - 1):
        # Both players answer the counts so far, then the counts record this round.
        last_x, last_y = bimatrix_best_responses(
            A, B, count_y / count_y.sum(), count_x / count_x.sum()
        )
        count_x[last_x] += 1
        count_y[last_y] += 1
    return {
        "x_avg": count_x / iters,
        "y_avg": count_y / iters,
        "last_x": last_x,
        "last_y": last_y,
    }


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
    rng = np.random.default_rng(seed)
    legal = nim_legal(n_max, max_take)
    Q = np.zeros((n_max + 1, max_take))
    for _ in range(episodes):
        n = int(rng.integers(1, n_max + 1))
        while n > 0:
            moves = np.flatnonzero(legal[n])
            if rng.random() < epsilon:
                k = int(rng.choice(moves)) + 1
            else:
                best = moves[Q[n, moves] == Q[n, moves].max()]
                k = int(rng.choice(best)) + 1
            left = n - k
            # The same table plays both sides. Taking the last stick wins; otherwise
            # the opponent moves next, and their best value is our loss.
            target = 1.0 if left == 0 else -Q[left, legal[left]].max()
            Q[n, k - 1] += alpha * (target - Q[n, k - 1])
            n = left
    return Q


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
