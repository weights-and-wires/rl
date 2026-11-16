"""Attack and defense as a repeated zero-sum game.

An attacker picks one of three attack families and a defender picks one of
three defenses. `SUCCESS[i, j]` is the chance that attack i gets through
defense j. `OVER_REFUSAL[j]` is the share of harmless requests that defense j
refuses. The defender pays a loss of SUCCESS[i, j] + lam * OVER_REFUSAL[j].
The attacker's payoff is that loss minus the lam term, which depends only on
the defense, so both players choose as in a zero-sum game on the loss.

`SUCCESS`, `OVER_REFUSAL`, and `softmax` are written for you. Fill in the rest.
"""

import itertools

import numpy as np

# Rows: 0 role-play wrapper, 1 encoded request, 2 many-turn drift.
# Columns: 0 wrapper filter, 1 decoding patch, 2 conversation classifier.
SUCCESS = np.array([[0.10, 0.70, 0.50], [0.60, 0.10, 0.70], [0.80, 0.50, 0.10]])
OVER_REFUSAL = np.array([0.05, 0.10, 0.25])


def softmax(z: np.ndarray) -> np.ndarray:
    """Return exp(z) normalised to sum to one, computed without overflow."""
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()


def loss_matrix(success: np.ndarray, over_refusal: np.ndarray, lam: float) -> np.ndarray:
    """Return L with L[i, j] = success[i, j] + lam * over_refusal[j].

    Args:
        success: (attacks, defenses) attack-success rates in [0, 1].
        over_refusal: (defenses,) share of harmless requests each defense refuses.
        lam: weight on over-refusal, at least 0.

    Returns:
        The (attacks, defenses) loss the defender pays. The attacker's row
        player maximizes it and the defender's column player minimizes it.
    """
    return success + lam * over_refusal[None, :]


def exploitability(L: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """How much the two players could gain, together, by best-responding.

    max_i (L @ y)_i - min_j (x @ L)_j, as in lesson 061. It is zero exactly at
    an equilibrium and positive everywhere else.
    """
    return float(np.max(L @ y) - np.min(x @ L))


def best_response_dynamics(L: np.ndarray, first_defense: int, rounds: int) -> list[tuple[int, int]]:
    """Patch the most recent attack, then find a new attack, for `rounds` rounds.

    Each round, the attacker picks argmax_i L[i, d] against the current
    defense d, then the defender switches to argmin_j L[a, j] against that
    attack a. Break ties at the lowest index.

    Returns:
        A list of `rounds` pairs (attack, defense), one per round, in order.
    """
    states = []
    d = first_defense
    for _ in range(rounds):
        a = int(np.argmax(L[:, d]))
        d = int(np.argmin(L[a]))
        states.append((a, d))
    return states


def exp3_strategy(log_w: np.ndarray, gamma: float) -> np.ndarray:
    """EXP3's probabilities: (1 - gamma) * softmax(log_w) + gamma / K."""
    k = len(log_w)
    return (1.0 - gamma) * softmax(log_w) + gamma / k


def exp3_update(log_w: np.ndarray, arm: int, reward: float, gamma: float) -> np.ndarray:
    """One EXP3 step after seeing the reward of the arm that was played.

    With p = exp3_strategy(log_w, gamma), the importance-weighted estimate is
    reward / p[arm] for the played arm and 0 for every other arm. Add
    gamma * estimate / K to that arm's log weight.

    Returns:
        A new array of log weights. The input is not modified.
    """
    k = len(log_w)
    p = exp3_strategy(log_w, gamma)
    estimate = np.zeros(k)
    estimate[arm] = reward / p[arm]
    return log_w + gamma * estimate / k


def hedge_update(log_w: np.ndarray, losses: np.ndarray, eta: float) -> np.ndarray:
    """One Hedge step: subtract eta * losses from the log weights.

    Returns:
        A new array of log weights. The input is not modified.
    """
    return log_w - eta * losses


def train_no_regret(
    success: np.ndarray,
    over_refusal: np.ndarray,
    lam: float,
    rounds: int,
    gamma: float,
    eta: float,
    seed: int,
) -> dict:
    """Run EXP3 (attacker) against Hedge (defender) for `rounds` rounds.

    Each round, both players draw an action from their current strategy with
    one numpy Generator seeded by `seed`: attacker first, then defender. The
    attacker sees only reward = success[i, j] for the pair played. The defender
    sees the loss of every defense against the attack that was drawn,
    L[i, :] with L = loss_matrix(success, over_refusal, lam).

    Returns:
        A dict with "x_avg" and "y_avg" (the averages of the strategies the
        players held each round, not of the actions drawn) and "x_last" and
        "y_last" (the strategies after the final update).
    """
    L = loss_matrix(success, over_refusal, lam)
    rng = np.random.default_rng(seed)
    m, n = L.shape
    log_wx, log_wy = np.zeros(m), np.zeros(n)
    x_sum, y_sum = np.zeros(m), np.zeros(n)
    for _ in range(rounds):
        x = exp3_strategy(log_wx, gamma)
        y = softmax(log_wy)
        x_sum += x
        y_sum += y
        i = int(rng.choice(m, p=x))
        j = int(rng.choice(n, p=y))
        log_wx = exp3_update(log_wx, i, success[i, j], gamma)
        log_wy = hedge_update(log_wy, L[i], eta)
    return {
        "x_avg": x_sum / rounds,
        "y_avg": y_sum / rounds,
        "x_last": exp3_strategy(log_wx, gamma),
        "y_last": softmax(log_wy),
    }


def solve_game(L: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Solve a zero-sum matrix game by support enumeration.

    The row player maximizes x @ L @ y and the column player minimizes it. For
    every pair of row and column supports of equal size k, solve for the
    column mix that makes every supported row pay v, and for the row mix that
    makes every supported column pay v. Accept the first pair whose mixes are
    non-negative and where no unsupported row pays more than v or unsupported
    column pays less than v.

    Returns:
        (value, x, y) with x of shape (rows,) and y of shape (columns,).
    """
    m, n = L.shape
    for k in range(1, min(m, n) + 1):
        for rows in itertools.combinations(range(m), k):
            for cols in itertools.combinations(range(n), k):
                sub = L[np.ix_(rows, cols)]
                # Unknowns (y_J, v): sub @ y - v = 0 and sum(y) = 1.
                a_y = np.zeros((k + 1, k + 1))
                a_y[:k, :k], a_y[:k, k], a_y[k, :k] = sub, -1.0, 1.0
                a_x = np.zeros((k + 1, k + 1))
                a_x[:k, :k], a_x[:k, k], a_x[k, :k] = sub.T, -1.0, 1.0
                rhs = np.r_[np.zeros(k), 1.0]
                try:
                    sol_y = np.linalg.solve(a_y, rhs)
                    sol_x = np.linalg.solve(a_x, rhs)
                except np.linalg.LinAlgError:
                    continue
                y_s, v = sol_y[:k], sol_y[k]
                x_s, v_x = sol_x[:k], sol_x[k]
                if y_s.min() < -1e-12 or x_s.min() < -1e-12 or abs(v - v_x) > 1e-9:
                    continue
                x, y = np.zeros(m), np.zeros(n)
                x[list(rows)], y[list(cols)] = x_s, y_s
                if np.max(L @ y) <= v + 1e-9 and np.min(x @ L) >= v - 1e-9:
                    return float(v), x, y
    raise ValueError("no equilibrium found")


def frontier(success: np.ndarray, over_refusal: np.ndarray, lams) -> tuple[np.ndarray, np.ndarray]:
    """The defender's trade-off as the over-refusal weight changes.

    For each lam, solve the game on loss_matrix(success, over_refusal, lam)
    and take the defender's equilibrium mix y. Report its over-refusal rate
    over_refusal @ y and its worst-case attack success max_i (success @ y)_i.

    Returns:
        (over, attack) arrays with one entry per lam, in the order given.
    """
    over, attack = [], []
    for lam in lams:
        _, _, y = solve_game(loss_matrix(success, over_refusal, lam))
        over.append(float(over_refusal @ y))
        attack.append(float(np.max(success @ y)))
    return np.array(over), np.array(attack)


if __name__ == "__main__":
    L = loss_matrix(SUCCESS, OVER_REFUSAL, 1.0)
    print("Patching the most recent attack, as (attack, defense) per round:")
    print(best_response_dynamics(L, 0, 7))
    value, x, y = solve_game(L)
    print(f"game value {value:.4f}, attacker mix {np.round(x, 3)}, defender mix {np.round(y, 3)}")
    out = train_no_regret(SUCCESS, OVER_REFUSAL, 1.0, 6000, gamma=0.03, eta=0.02, seed=0)
    print("average play exploitability:", round(exploitability(L, out["x_avg"], out["y_avg"]), 4))
    print("last play exploitability:   ", round(exploitability(L, out["x_last"], out["y_last"]), 4))
    over, attack = frontier(SUCCESS, OVER_REFUSAL, [0, 1, 2, 4, 8])
    print("over-refusal:", np.round(over, 3))
    print("attack success:", np.round(attack, 3))
