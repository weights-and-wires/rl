"""Checks for the multi-agent learning and self-play exercise."""

import numpy as np

from rlcheck import almost as _almost, check, close, equal, fail, finite, has, shape


def almost(got, want, tol, label=None):
    """rlcheck.almost, except that NaN or infinity fails instead of slipping past the comparison."""
    finite(got, label=label)
    _almost(got, want, tol, label=label)


RPS = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
GAME = np.array([[3.0, -1.0], [-2.0, 1.0]])  # Nash: x = (3/7, 4/7), y = (2/7, 5/7), value 1/7
# Bridge: action 0 is yield, action 1 is push. Mixed Nash is (1/2, 1/2).
BRIDGE_ROW = np.array([[2.0, 1.0], [3.0, 0.0]])
BRIDGE_COL = np.array([[2.0, 3.0], [1.0, 0.0]])
X_NASH, Y_NASH = np.array([3 / 7, 4 / 7]), np.array([2 / 7, 5 / 7])
ROCK = np.array([1.0, 0.0, 0.0])


@check("regret_matching_strategy plays in proportion to positive regret, uniform when none is positive")
def test_rm_strategy(sub):
    rm = has(sub, "regret_matching_strategy")
    close(rm(np.array([2.0, -1.0, 1.0])), np.array([2 / 3, 0.0, 1 / 3]), atol=1e-12, label="regrets [2, -1, 1]")
    close(rm(np.array([-2.0, -1.0, -0.5])), np.full(3, 1 / 3), atol=1e-12, label="all regrets negative")
    close(rm(np.zeros(4)), np.full(4, 0.25), atol=1e-12, label="all regrets zero")


@check("exploitability is max(A y) - min(x A): 0 at a Nash equilibrium, 2 for rock against rock")
def test_exploitability(sub):
    exploitability = has(sub, "exploitability")
    almost(exploitability(RPS, np.full(3, 1 / 3), np.full(3, 1 / 3)), 0.0, tol=1e-12, label="RPS, uniform play")
    almost(exploitability(RPS, ROCK, ROCK), 2.0, tol=1e-12, label="RPS, rock against rock")
    almost(exploitability(GAME, X_NASH, Y_NASH), 0.0, tol=1e-12, label="the 2x2 game at (3/7, 4/7), (2/7, 5/7)")
    almost(exploitability(GAME, np.array([1.0, 0.0]), np.array([1.0, 0.0])), 4.0, tol=1e-12,
           label="the 2x2 game, both players on their first action")
    rng = np.random.default_rng(0)
    for _ in range(20):
        A = rng.normal(size=(3, 4))
        x, y = rng.dirichlet(np.ones(3)), rng.dirichlet(np.ones(4))
        if not exploitability(A, x, y) >= -1e-12:
            fail("exploitability came out negative; it is a sum of two best-response gains")


@check("regret_matching's first two rounds of rock-paper-scissors match the hand computation")
def test_rm_first_steps(sub):
    regret_matching = has(sub, "regret_matching")
    out = regret_matching(RPS, 1, ROCK, ROCK)
    close(out["regret_x"], np.array([0.0, 1.0, -1.0]), atol=1e-12, label="row regrets after rock against rock")
    close(out["regret_y"], np.array([0.0, 1.0, -1.0]), atol=1e-12, label="column regrets after rock against rock")
    out = regret_matching(RPS, 2, ROCK, ROCK)
    close(out["x_last"], np.array([0.0, 1.0, 0.0]), atol=1e-12, label="round-2 strategy: all positive regret on paper")
    close(out["x_avg"], np.array([0.5, 0.5, 0.0]), atol=1e-12, label="average of rock and paper")
    close(out["regret_x"], np.array([-1.0, 1.0, 0.0]), atol=1e-12, label="row regrets after paper against paper")


@check("regret matching's average strategies converge to Nash in RPS, with exploitability equal to total regret / T")
def test_rm_rps(sub):
    regret_matching = has(sub, "regret_matching")
    exploitability = has(sub, "exploitability")
    T = 10000
    out = regret_matching(RPS, T, ROCK, ROCK)
    e = exploitability(RPS, out["x_avg"], out["y_avg"])
    almost(e, (out["regret_x"].max() + out["regret_y"].max()) / T, tol=1e-9,
           label="exploitability of the averages against (max row regret + max column regret) / T")
    bound = 2.0 * np.sqrt(3 * T)  # payoff range 2, three actions
    for who in ("regret_x", "regret_y"):
        if not out[who].max() <= bound:
            fail(f"{who} reached {out[who].max():.1f}, above the regret-matching bound {bound:.1f}")
    if not e <= 0.04:
        fail(f"after {T} rounds the average strategies should be exploitable by under 0.04, got {e:.4f}")
    close(out["x_avg"], np.full(3, 1 / 3), atol=0.02, label="the row player's average strategy")


@check("regret matching's last strategies keep cycling in RPS: only the average converges")
def test_rm_last_iterate(sub):
    regret_matching = has(sub, "regret_matching")
    exploitability = has(sub, "exploitability")
    for T in (9000, 10000):
        out = regret_matching(RPS, T, ROCK, ROCK)
        e = exploitability(RPS, out["x_last"], out["y_last"])
        if not e >= 0.3:
            fail(f"at T = {T} the last strategies are exploitable by only {e:.3f}; expected them to keep cycling")


@check("regret matching finds the 2x2 game's mixed equilibrium (3/7, 4/7) and (2/7, 5/7)")
def test_rm_2x2(sub):
    regret_matching = has(sub, "regret_matching")
    exploitability = has(sub, "exploitability")
    out = regret_matching(GAME, 20000, np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    close(out["x_avg"], X_NASH, atol=0.02, label="row player's average against 3/7, 4/7")
    close(out["y_avg"], Y_NASH, atol=0.02, label="column player's average against 2/7, 5/7")
    e = exploitability(GAME, out["x_avg"], out["y_avg"])
    if not e <= 0.05:
        fail(f"the averages should be exploitable by under 0.05, got {e:.4f}")
    value = out["x_avg"] @ GAME @ out["y_avg"]
    almost(value, 1 / 7, tol=0.01, label="the value of the average play")


@check("fictitious play's empirical mixtures converge to Nash in RPS and in the 2x2 game")
def test_fictitious_play(sub):
    fictitious_play = has(sub, "fictitious_play")
    exploitability = has(sub, "exploitability")
    x, y = fictitious_play(RPS, 5000, 0, 0)
    shape(x, (3,), label="x")
    close(x.sum(), 1.0, atol=1e-12, label="sum of x")
    close(x, np.full(3, 1 / 3), atol=0.02, label="row mixture in RPS")
    close(y, np.full(3, 1 / 3), atol=0.02, label="column mixture in RPS")
    x, y = fictitious_play(GAME, 5000, 0, 0)
    close(x, X_NASH, atol=0.02, label="row mixture in the 2x2 game")
    close(y, Y_NASH, atol=0.02, label="column mixture in the 2x2 game")
    x, y = fictitious_play(RPS, 2, 0, 0)
    close(x, np.array([0.5, 0.5, 0.0]), atol=1e-12, label="after rock, the best response to rock is paper")


@check("bimatrix_best_responses uses each player's own table, and a tie returns the lowest index")
def test_bimatrix_best_responses(sub):
    # Wrong implementation `column_br_from_row_payoff` ignores B and returns
    # argmin(x @ A). When the row player pushes, x @ B is [1, 0], so the
    # column player yields (action 0). argmin([3, 0]) returns 1.
    # argmax(x @ A) fails the third call: at x = [1, 0] it returns 0, and
    # the column table pays 3 for push against 2 for yield.
    best = has(sub, "bimatrix_best_responses")
    yield_pure = np.array([1.0, 0.0])
    push_pure = np.array([0.0, 1.0])
    half = np.array([0.5, 0.5])
    row, col = best(BRIDGE_ROW, BRIDGE_COL, yield_pure, push_pure)
    equal(np.array([row, col]), np.array([1, 0]),
          label="row faces yield (expected push, 1) and column faces push (expected yield, 0)")
    row, col = best(BRIDGE_ROW, BRIDGE_COL, half, half)
    equal(np.array([row, col]), np.array([0, 0]),
          label="a 3/2 tie for both players (expected the lowest index, 0)")
    row, col = best(BRIDGE_ROW, BRIDGE_COL, push_pure, yield_pure)
    equal(np.array([row, col]), np.array([0, 1]),
          label="row faces push (expected yield, 0) and column faces yield (expected push, 1)")


@check("general-sum fictitious play counts the latest action, and that action keeps flipping")
def test_general_sum_fictitious_play(sub):
    # Wrong implementation `fictitious_play_average_forgets_latest` averages
    # the first iters - 1 actions. After yield, push, yield that average is
    # [1/2, 1/2]. Counting the latest yield gives [2/3, 1/3].
    play = has(sub, "general_sum_fictitious_play")
    out = play(BRIDGE_ROW, BRIDGE_COL, 3, 0, 0)
    close(out["x_avg"], np.array([2 / 3, 1 / 3]), atol=1e-12,
          label="row average after yield, push, yield (expected 2/3, 1/3)")
    close(out["y_avg"], np.array([2 / 3, 1 / 3]), atol=1e-12,
          label="column average after yield, push, yield (expected 2/3, 1/3)")
    equal(np.array([out["last_x"], out["last_y"]]), np.array([0, 0]),
          label="round 3 plays yield against yield (expected 0, 0)")
    out = play(BRIDGE_ROW, BRIDGE_COL, 4, 0, 0)
    close(out["x_avg"], np.array([0.5, 0.5]), atol=1e-12,
          label="row average after four rounds (expected 1/2, 1/2)")
    equal(np.array([out["last_x"], out["last_y"]]), np.array([1, 1]),
          label="round 4 plays push against push (expected 1, 1)")
    out = play(BRIDGE_ROW, BRIDGE_COL, 7, 0, 0)
    close(out["x_avg"], np.array([4 / 7, 3 / 7]), atol=1e-12,
          label="row average after seven rounds (expected 4/7, 3/7)")
    equal(out["last_x"], 0, label="round 7 plays yield (expected 0, not a settled push)")
    out = play(BRIDGE_ROW, BRIDGE_COL, 8, 0, 0)
    equal(out["last_x"], 1, label="round 8 plays push (expected 1, so the latest action flipped)")


@check("nim_self_play learns the winning rule, take n mod 4, from playing itself")
def test_nim(sub):
    nim_self_play = has(sub, "nim_self_play")
    nim_legal = has(sub, "nim_legal")
    Q = nim_self_play(15, 3000, 0.5, 0.2, seed=0)
    shape(Q, (16, 3), label="Q")
    legal = nim_legal(15)
    wrong = []
    for n in range(1, 16):
        value = Q[n, legal[n]].max()
        if (value > 0) != (n % 4 != 0):
            wrong.append(f"pile {n} has value {value:+.2f}")
        if n % 4:
            k = int(np.argmax(np.where(legal[n], Q[n], -np.inf))) + 1
            if k != n % 4:
                wrong.append(f"from pile {n} it takes {k}, not {n % 4}")
    if wrong:
        fail("self-play didn't find optimal play: " + "; ".join(wrong[:4]))
