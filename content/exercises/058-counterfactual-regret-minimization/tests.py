"""Checks for the counterfactual regret minimization exercise."""

import itertools

import numpy as np

from rlcheck import almost as _almost, check, close, equal, fail, finite, has

LETTERS = "pb"
CARDS = (0, 1, 2)
DEALS = [(a, b) for a in CARDS for b in CARDS if a != b]
TERMINAL = ("pp", "bp", "bb", "pbp", "pbb")
P1_KEYS = [f"{c}{h}" for h in ("", "pb") for c in CARDS]
P2_KEYS = [f"{c}{h}" for h in ("p", "b") for c in CARDS]
VALUE = -1.0 / 18.0


def almost(got, want, tol, label=None):
    """rlcheck.almost, except that NaN or infinity fails instead of slipping past the comparison."""
    finite(got, label=label)
    _almost(got, want, tol, label=label)


def _payoff(cards, h):
    if h == "bp":
        return 1.0
    if h == "pbp":
        return -1.0
    stake = 1.0 if h == "pp" else 2.0
    return stake if cards[0] > cards[1] else -stake


def _value(strategy):
    """Player 1's expected payoff, summed over deals and terminal histories with no recursion."""
    total = 0.0
    for cards in DEALS:
        for h in TERMINAL:
            prob = 1.0 / len(DEALS)
            for i, letter in enumerate(h):
                prob *= strategy[f"{cards[i % 2]}{h[:i]}"][LETTERS.index(letter)]
            total += prob * _payoff(cards, h)
    return total


def _best_response(strategy, player):
    """Brute force: try all 64 pure strategies for `player`, keep the best."""
    keys = P1_KEYS if player == 0 else P2_KEYS
    sign = 1.0 if player == 0 else -1.0
    best = -np.inf
    for pure in itertools.product((0, 1), repeat=len(keys)):
        s = dict(strategy)
        for k, a in zip(keys, pure):
            s[k] = np.eye(2)[a]
        best = max(best, sign * _value(s))
    return best


def equilibrium(alpha):
    """A member of the known equilibrium family, with alpha in [0, 1/3]."""
    s = {
        "0": [1 - alpha, alpha], "1": [1.0, 0.0], "2": [1 - 3 * alpha, 3 * alpha],
        "0pb": [1.0, 0.0], "1pb": [2 / 3 - alpha, alpha + 1 / 3], "2pb": [0.0, 1.0],
        "0b": [1.0, 0.0], "1b": [2 / 3, 1 / 3], "2b": [0.0, 1.0],
        "0p": [2 / 3, 1 / 3], "1p": [1.0, 0.0], "2p": [0.0, 1.0],
    }
    return {k: np.array(v, dtype=float) for k, v in s.items()}


def constant(action_p1, action_p2):
    """Every player 1 information set plays action_p1, and every player 2 one plays action_p2."""
    s = {k: np.eye(2)[action_p1] for k in P1_KEYS}
    s.update({k: np.eye(2)[action_p2] for k in P2_KEYS})
    return s


def uniform():
    return {k: np.array([0.5, 0.5]) for k in P1_KEYS + P2_KEYS}


def random_strategy(seed):
    rng = np.random.default_rng(seed)
    return {k: rng.dirichlet([1.0, 1.0]) for k in P1_KEYS + P2_KEYS}


@check("regret_matching plays in proportion to positive regret, uniform when none is positive")
def test_regret_matching(sub):
    regret_matching = has(sub, "regret_matching")
    close(regret_matching(np.array([3.0, -1.0])), np.array([1.0, 0.0]), atol=1e-12, label="regrets [3, -1]")
    close(regret_matching(np.array([1.0, 3.0])), np.array([0.25, 0.75]), atol=1e-12, label="regrets [1, 3]")
    close(regret_matching(np.array([-2.0, -1.0])), np.array([0.5, 0.5]), atol=1e-12, label="all regrets negative")
    close(regret_matching(np.zeros(2)), np.array([0.5, 0.5]), atol=1e-12, label="all regrets zero")


@check("infosets finds the 12 information sets: six per player, keyed by card and history")
def test_infosets(sub):
    infosets = has(sub, "infosets")
    got = infosets()
    want = {k: 0 for k in P1_KEYS} | {k: 1 for k in P2_KEYS}
    if len(got) != 12:
        fail(f"expected 12 information sets, got {len(got)}: {sorted(got)}")
    if dict(got) != want:
        fail(f"expected player 1 to own {P1_KEYS} and player 2 to own {P2_KEYS}, got {dict(sorted(got.items()))}")


@check("game_value matches hand values: -1/18 at equilibrium, 0 for always-bet, -1 for pass against bet")
def test_game_value(sub):
    game_value = has(sub, "game_value")
    almost(game_value(equilibrium(0.0)), VALUE, tol=1e-12, label="equilibrium with alpha = 0")
    almost(game_value(equilibrium(1 / 3)), VALUE, tol=1e-12, label="equilibrium with alpha = 1/3")
    almost(game_value(constant(1, 1)), 0.0, tol=1e-12, label="both players always bet")
    almost(game_value(constant(0, 1)), -1.0, tol=1e-12,
           label="player 1 always passes and player 2 always bets (player 1 folds to every bet)")
    for seed in range(3):
        s = random_strategy(seed)
        almost(game_value(s), _value(s), tol=1e-12, label=f"random strategy, seed {seed}")


@check("best_response_value is exact, and cannot see the opponent's card")
def test_best_response(sub):
    best_response_value = has(sub, "best_response_value")
    # Against a player 2 who folds to every bet, player 1 bets with every card and wins 1 each hand.
    almost(best_response_value(constant(0, 0), 0), 1.0, tol=1e-12, label="player 1 against a player 2 who always folds")
    # Against a player 1 who always bets, player 2 calls with the king (+2), is
    # indifferent with the queen (0), and folds the jack (-1): (2 + 0 - 1) / 3.
    almost(best_response_value(constant(1, 0), 1), 1 / 3, tol=1e-12, label="player 2 against a player 1 who always bets")
    for player in (0, 1):
        almost(best_response_value(equilibrium(0.1), player), VALUE if player == 0 else -VALUE, tol=1e-12,
               label=f"player {player + 1} against the equilibrium strategy")
    # Wrong implementation: maximizing the action inside each deal, which lets
    # the best response read the opponent's card. It beats the brute-force
    # value over pure strategies, which pick one action per information set.
    for seed in range(4):
        s = random_strategy(seed)
        for player in (0, 1):
            almost(best_response_value(s, player), _best_response(s, player), tol=1e-12,
                   label=f"player {player + 1}, random opponent, seed {seed}")


@check("exploitability is the sum of both best-response gains: 0 on the equilibrium family, positive elsewhere")
def test_exploitability(sub):
    exploitability = has(sub, "exploitability")
    for alpha in (0.0, 0.2, 1 / 3):
        almost(exploitability(equilibrium(alpha)), 0.0, tol=1e-12, label=f"equilibrium with alpha = {alpha:.2f}")
    for name, s in [("uniform", uniform()), ("always bet", constant(1, 1)), ("random 5", random_strategy(5))]:
        v = _value(s)
        want = (_best_response(s, 0) - v) + (_best_response(s, 1) + v)
        almost(exploitability(s), want, tol=1e-12, label=f"{name}: sum of the two best-response gains")
        if not want > 1e-6:
            fail(f"test setup: {name} should be exploitable, got {want}")


@check("cfr_iteration weights regret by the opponent's reach, and the average by the player's own")
def test_cfr_iteration(sub):
    cfr_iteration = has(sub, "cfr_iteration")
    keys = P1_KEYS + P2_KEYS
    # From zero regrets both players are uniform. Each information set is visited
    # twice, once per card the opponent can hold. Player 1's second decision
    # ("pb") is reached with own probability 1/2, every other one with 1.
    regret_sum = {k: np.zeros(2) for k in keys}
    strategy_sum = {k: np.zeros(2) for k in keys}
    v = cfr_iteration(regret_sum, strategy_sum)
    almost(v, _value(uniform()), tol=1e-12, label="returned payoff to player 1 under uniform play")
    for k in keys:
        want = np.array([0.5, 0.5]) if k.endswith("pb") else np.array([1.0, 1.0])
        close(strategy_sum[k], want, atol=1e-12, label=f"strategy_sum[{k!r}] after one uniform iteration")
    # Away from uniform, the regret at a set must match a check that never
    # traverses a tree: forcing action a at the set changes the value by
    # (own reach of the set) * (counterfactual regret of a).
    for seed in (0, 1):
        sigma = random_strategy(seed)
        regret_sum = {k: 3.0 * sigma[k] for k in keys}  # regret matching turns this back into sigma
        strategy_sum = {k: np.zeros(2) for k in keys}
        cfr_iteration(regret_sum, strategy_sum)
        base = _value(sigma)
        for k in keys:
            player = 0 if k in P1_KEYS else 1
            sign = 1.0 if player == 0 else -1.0
            own = 1.0
            if k.endswith("pb"):
                own = sigma[f"{k[0]}"][0]  # player 1 passed at the root with this card
            want = np.zeros(2)
            for a in (0, 1):
                forced = dict(sigma)
                forced[k] = np.eye(2)[a]
                want[a] = sign * (_value(forced) - base) / own
            # Wrong implementation: weighting by the player's own reach instead of the opponent's.
            close(regret_sum[k] - 3.0 * sigma[k], want, atol=1e-12,
                  label=f"regret added at {k!r}, seed {seed}")


@check("cfr_kuhn's average strategy is nearly unexploitable, and the current strategy is not")
def test_cfr_kuhn(sub):
    cfr_kuhn = has(sub, "cfr_kuhn")
    game_value = has(sub, "game_value")
    exploitability = has(sub, "exploitability")
    out = cfr_kuhn(10000)
    avg = out["average"]
    if set(avg) != set(P1_KEYS + P2_KEYS):
        fail(f"expected the 12 information-set keys, got {sorted(avg)}")
    for k, p in avg.items():
        if not np.isclose(np.sum(p), 1.0) or np.min(p) < 0:
            fail(f"average strategy at {k!r} is not a probability vector: {p}")
    # Wrong implementation: returning the current regret-matching strategy.
    # It keeps jumping between near-pure strategies and is far from equilibrium.
    expl = exploitability(avg)
    if not expl < 0.01:
        fail(f"after 10000 iterations the average strategy's exploitability should be below 0.01, got {expl:.4f}")
    almost(game_value(avg), VALUE, tol=1e-3, label="game value of the average strategy (-1/18)")
    current = exploitability(out["current"])
    if not current > 5 * expl:
        fail(f"expected the current strategy to be far more exploitable than the average ({expl:.4f}), "
             f"got {current:.4f}; the current strategy keeps moving and only the average converges")


@check("the average strategy lands in the equilibrium family: king bets 3x the jack, and player 2 mixes at 1/3")
def test_equilibrium_family(sub):
    cfr_kuhn = has(sub, "cfr_kuhn")
    avg = cfr_kuhn(10000)["average"]
    alpha = avg["0"][1]
    if not 0.0 <= alpha <= 1 / 3 + 0.03:
        fail(f"player 1's jack bluff frequency alpha should fall in [0, 1/3], got {alpha:.3f}")
    almost(avg["2"][1], 3 * alpha, tol=0.06, label="player 1 bets the king 3 * alpha")
    almost(avg["1"][1], 0.0, tol=0.03, label="player 1 passes the queen at first")
    almost(avg["1pb"][1], alpha + 1 / 3, tol=0.06, label="player 1 calls with the queen after pass, bet: alpha + 1/3")
    almost(avg["2pb"][1], 1.0, tol=0.03, label="player 1 calls with the king after pass, bet")
    almost(avg["0pb"][1], 0.0, tol=0.03, label="player 1 folds the jack after pass, bet")
    almost(avg["2b"][1], 1.0, tol=0.03, label="player 2 calls a bet with the king")
    almost(avg["0b"][1], 0.0, tol=0.03, label="player 2 folds the jack to a bet")
    almost(avg["1b"][1], 1 / 3, tol=0.05, label="player 2 calls a bet with the queen 1/3 of the time")
    almost(avg["0p"][1], 1 / 3, tol=0.05, label="player 2 bluffs the jack after a pass 1/3 of the time")
    almost(avg["2p"][1], 1.0, tol=0.03, label="player 2 bets the king after a pass")
    almost(avg["1p"][1], 0.0, tol=0.03, label="player 2 passes the queen after a pass")
