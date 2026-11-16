"""Checks for the attack-and-defense game exercise."""

import numpy as np
from scipy.optimize import linprog

from rlcheck import almost as _almost, check, close, fail, finite, has, shape


def almost(got, want, tol, label=None):
    """rlcheck.almost, except that NaN or infinity fails instead of slipping past the comparison."""
    finite(got, label=label)
    _almost(got, want, tol, label=label)


SUCCESS = np.array([[0.10, 0.70, 0.50], [0.60, 0.10, 0.70], [0.80, 0.50, 0.10]])
OVER_REFUSAL = np.array([0.05, 0.10, 0.25])
# The loss at lam = 1: 0.10 + 0.05, 0.70 + 0.10, 0.50 + 0.25, and so on.
L1 = np.array([[0.15, 0.80, 0.75], [0.65, 0.20, 0.95], [0.85, 0.60, 0.35]])
# The 2x2 game of lesson 061: Nash x = (3/7, 4/7), y = (2/7, 5/7), value 1/7.
GAME = np.array([[3.0, -1.0], [-2.0, 1.0]])
# At lam = 1 every attack costs the defender 27.7 / 47 against y = (14, 17, 16) / 47.
Y1 = np.array([14.0, 17.0, 16.0]) / 47.0
VALUE1 = 27.7 / 47.0


def lp_value(L):
    """Independent reference: the value of the game that maximizes over rows, minimizes over columns."""
    m, n = L.shape
    cost = np.r_[np.zeros(n), 1.0]
    res = linprog(
        cost,
        A_ub=np.c_[L, -np.ones(m)],
        b_ub=np.zeros(m),
        A_eq=[np.r_[np.ones(n), 0.0]],
        b_eq=[1.0],
        bounds=[(0, None)] * n + [(None, None)],
        method="highs",
    )
    return res.x[-1], res.x[:n]


@check("loss_matrix adds lam times each defense's over-refusal to its whole column")
def test_loss_matrix(sub):
    loss_matrix = has(sub, "loss_matrix")
    L = loss_matrix(SUCCESS, OVER_REFUSAL, 1.0)
    shape(L, (3, 3), label="loss matrix")
    close(L, L1, atol=1e-12, label="loss at lam = 1")
    close(loss_matrix(SUCCESS, OVER_REFUSAL, 0.0), SUCCESS, atol=1e-12, label="loss at lam = 0")
    close(loss_matrix(SUCCESS, OVER_REFUSAL, 2.0)[:, 2], SUCCESS[:, 2] + 0.5, atol=1e-12,
          label="the classifier column at lam = 2 (over-refusal 0.25, so 0.5 added)")


@check("exploitability is max(L y) - min(x L): 0 at Nash, 4 for the 2x2 game's first actions, 0.7 for one pure pair")
def test_exploitability(sub):
    exploitability = has(sub, "exploitability")
    almost(exploitability(GAME, np.array([3 / 7, 4 / 7]), np.array([2 / 7, 5 / 7])), 0.0, tol=1e-12,
           label="the 2x2 game at its equilibrium")
    almost(exploitability(GAME, np.array([1.0, 0.0]), np.array([1.0, 0.0])), 4.0, tol=1e-12,
           label="the 2x2 game, both players on their first action")
    # Attack 0 against defense 0: the best attack against defense 0 pays 0.85 and the best defense against attack 0 pays 0.15.
    almost(exploitability(L1, np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])), 0.7, tol=1e-12,
           label="attack 0 against defense 0 at lam = 1")
    rng = np.random.default_rng(0)
    for _ in range(20):
        A = rng.normal(size=(3, 4))
        x, y = rng.dirichlet(np.ones(3)), rng.dirichlet(np.ones(4))
        if not exploitability(A, x, y) >= -1e-12:
            fail("exploitability came out negative; it is a sum of two best-response gains")


@check("patching the most recent attack cycles: the same (attack, defense) pairs return every 3 rounds")
def test_best_response_cycle(sub):
    best_response_dynamics = has(sub, "best_response_dynamics")
    exploitability = has(sub, "exploitability")
    states = [tuple(s) for s in best_response_dynamics(L1, 0, 9)]
    want = [(2, 2), (1, 1), (0, 0)] * 3
    if states != want:
        fail(f"from defense 0 the rounds should be {want}, got {states}")
    if len(set(states)) >= len(states):
        fail("no state repeated, but best-response dynamics on this game must cycle")
    for a, d in states[:3]:
        x, y = np.eye(3)[a], np.eye(3)[d]
        e = exploitability(L1, x, y)
        if not e >= 0.5:
            fail(f"the pair (attack {a}, defense {d}) is exploitable by only {e:.3f}; patching the last attack should leave a large gap")
    # The same dynamics from another start fall into the same cycle.
    other = [tuple(s) for s in best_response_dynamics(L1, 1, 6)]
    if set(other) != {(2, 2), (1, 1), (0, 0)}:
        fail(f"starting from defense 1 the dynamics should visit the same three pairs, got {other}")


@check("EXP3 matches the hand computation, including importance weighting by 1 / p")
def test_exp3(sub):
    exp3_strategy = has(sub, "exp3_strategy")
    exp3_update = has(sub, "exp3_update")
    close(exp3_strategy(np.zeros(3), 0.3), np.full(3, 1 / 3), atol=1e-12, label="zero log weights")
    # Uniform p = 1/3: estimate 0.8 / (1/3) = 2.4, and the log weight grows by 0.3 * 2.4 / 3 = 0.24.
    new = exp3_update(np.zeros(3), 2, 0.8, 0.3)
    close(new, np.array([0.0, 0.0, 0.24]), atol=1e-12, label="uniform start, arm 2, reward 0.8")
    # log_w = (ln 2, 0, 0), gamma 0.3: softmax (0.5, 0.25, 0.25), so p = (0.45, 0.275, 0.275).
    # Arm 1 with reward 0.6: estimate 0.6 / 0.275, and the log weight grows by 0.3 * estimate / 3.
    log_w = np.array([np.log(2.0), 0.0, 0.0])
    close(exp3_strategy(log_w, 0.3), np.array([0.45, 0.275, 0.275]), atol=1e-12, label="p for log weights (ln 2, 0, 0)")
    before = log_w.copy()
    new = exp3_update(log_w, 1, 0.6, 0.3)
    close(new, before + np.array([0.0, 0.3 * (0.6 / 0.275) / 3, 0.0]), atol=1e-12, label="update with p[arm] = 0.275")
    close(log_w, before, atol=0, label="the input log weights (they must not change)")


@check("EXP3's importance-weighted estimate is unbiased: the expected update on every arm is gamma * reward / K")
def test_exp3_unbiased(sub):
    exp3_strategy = has(sub, "exp3_strategy")
    exp3_update = has(sub, "exp3_update")
    gamma = 0.2
    log_w = np.array([0.3, -0.2, 0.1])
    rewards = np.array([0.9, 0.2, 0.5])
    p = exp3_strategy(log_w, gamma)
    expected = np.zeros(3)
    for arm in range(3):
        expected += p[arm] * (exp3_update(log_w, arm, rewards[arm], gamma) - log_w)
    close(expected, gamma * rewards / 3, atol=1e-12,
          label="expected change in each log weight (the estimate must be reward / p[arm], so that p[arm] cancels)")


@check("hedge_update subtracts eta times the loss from each log weight")
def test_hedge(sub):
    hedge_update = has(sub, "hedge_update")
    log_w = np.array([0.0, 0.1, -0.2])
    before = log_w.copy()
    new = hedge_update(log_w, np.array([0.5, 1.0, 0.2]), 0.1)
    close(new, np.array([-0.05, 0.0, -0.22]), atol=1e-12, label="updated log weights")
    close(log_w, before, atol=0, label="the input log weights (they must not change)")


@check("solve_game finds the 2x2 equilibrium, the hand-checked 3x3 value 27.7 / 47, and agrees with an LP on random games")
def test_solve_game(sub):
    solve_game = has(sub, "solve_game")
    value, x, y = solve_game(GAME)
    almost(value, 1 / 7, tol=1e-9, label="value of the 2x2 game")
    close(x, np.array([3 / 7, 4 / 7]), atol=1e-9, label="row mix of the 2x2 game")
    close(y, np.array([2 / 7, 5 / 7]), atol=1e-9, label="column mix of the 2x2 game")
    value, x, y = solve_game(L1)
    almost(value, VALUE1, tol=1e-9, label="value at lam = 1")
    close(y, Y1, atol=1e-9, label="defender's mix at lam = 1")
    close(L1 @ y, np.full(3, VALUE1), atol=1e-9, label="what each attack pays against the defender's mix")
    close(x @ L1, np.full(3, VALUE1), atol=1e-9, label="what each defense costs against the attacker's mix")
    rng = np.random.default_rng(0)
    for k in range(20):
        L = rng.normal(size=(3, 4))
        value, x, y = solve_game(L)
        want, _ = lp_value(L)
        almost(value, want, tol=1e-7, label=f"random 3x4 game {k}, against the LP value")
        almost(x @ L @ y, value, tol=1e-7, label=f"payoff of the returned mixes in random game {k}")


@check("EXP3 against Hedge: the average strategies are nearly unexploitable while patch-the-last-attack stays far off")
def test_no_regret(sub):
    train_no_regret = has(sub, "train_no_regret")
    exploitability = has(sub, "exploitability")
    solve_game = has(sub, "solve_game")
    _, x_star, y_star = solve_game(L1)
    for seed in (0, 1, 2):
        out = train_no_regret(SUCCESS, OVER_REFUSAL, 1.0, 10000, gamma=0.03, eta=0.02, seed=seed)
        for name in ("x_avg", "y_avg", "x_last", "y_last"):
            finite(out[name], label=name)
            almost(out[name].sum(), 1.0, tol=1e-9, label=f"sum of {name}")
        e = exploitability(L1, out["x_avg"], out["y_avg"])
        if not e <= 0.04:
            fail(f"seed {seed}: after 10000 rounds the average strategies should be exploitable by under 0.04, got {e:.4f}")
        close(out["y_avg"], y_star, atol=0.08, label=f"seed {seed}: defender's average mix against the equilibrium")
        close(out["x_avg"], x_star, atol=0.12, label=f"seed {seed}: attacker's average mix against the equilibrium")


@check("the frontier is monotone: a larger over-refusal weight never raises over-refusal or lowers attack success")
def test_frontier(sub):
    frontier = has(sub, "frontier")
    over, attack = frontier(SUCCESS, OVER_REFUSAL, [0, 1, 2, 4, 8])
    shape(over, (5,), label="over-refusal")
    shape(attack, (5,), label="attack success")
    close(over[1], 6.4 / 47, atol=1e-9, label="over-refusal at lam = 1 (6.4 / 47)")
    close(attack[1], 21.3 / 47, atol=1e-9, label="worst-case attack success at lam = 1 (21.3 / 47)")
    close(over[1] + attack[1], VALUE1, atol=1e-9, label="over-refusal plus attack success at lam = 1 (the game value)")
    close(over[4], 0.05, atol=1e-9, label="at lam = 8 the defender plays the cheapest defense alone")
    close(attack[4], 0.80, atol=1e-9, label="and the worst attack then succeeds 80% of the time")
    lams = np.arange(0, 10.01, 0.25)
    over, attack = frontier(SUCCESS, OVER_REFUSAL, lams)
    if np.any(np.diff(over) > 1e-9):
        fail("over-refusal went up as the over-refusal weight grew")
    if np.any(np.diff(attack) < -1e-9):
        fail("worst-case attack success went down as the over-refusal weight grew")
    if not over[0] > over[-1] + 0.05:
        fail(f"over-refusal should fall by more than 0.05 between lam = 0 and lam = 10, got {over[0]:.3f} to {over[-1]:.3f}")
    if len(set(np.round(over, 6))) < 3:
        fail("the sweep from lam = 0 to 10 should show at least three different frontier points")
