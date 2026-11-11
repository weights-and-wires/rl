"""Counterfactual regret minimization on Kuhn poker.

Kuhn poker has three cards: 0 (jack), 1 (queen), and 2 (king). Each player
antes 1 and gets one card. Player 1 acts first and can pass (action 0) or bet 1
(action 1). The game ends on a pass after a bet (a fold), on two passes, or on
two bets, and the higher card wins at a showdown. A history is the string of
actions so far, such as "pb" for pass then bet.
"""

import numpy as np

PASS, BET = 0, 1
LETTERS = "pb"
CARDS = (0, 1, 2)
DEALS = [(a, b) for a in CARDS for b in CARDS if a != b]  # six deals, each with probability 1/6
TERMINAL = ("pp", "bp", "bb", "pbp", "pbb")


def is_terminal(history):
    """Return True when no player acts after this history."""
    return history in TERMINAL


def to_move(history):
    """Return 0 when player 1 acts after this history, and 1 when player 2 does."""
    return len(history) % 2


def payoff(cards, history):
    """Player 1's payoff at a terminal history. Player 2 receives the negative."""
    if history == "bp":
        return 1.0
    if history == "pbp":
        return -1.0
    stake = 1.0 if history == "pp" else 2.0
    return stake if cards[0] > cards[1] else -stake


def infoset_key(card, history):
    """Name the information set of a player who holds `card` after `history`, such as "2pb"."""
    return f"{card}{history}"


def regret_matching(regrets):
    """Play each action in proportion to its positive regret, uniform when none is positive."""
    positive = np.maximum(regrets, 0.0)
    total = positive.sum()
    if total <= 0.0:
        return np.full(len(regrets), 1.0 / len(regrets))
    return positive / total


def infosets():
    """Return {key: player} for every information set, found by walking the game tree."""
    found = {}

    def walk(history):
        if is_terminal(history):
            return
        for card in CARDS:
            found[infoset_key(card, history)] = to_move(history)
        for letter in LETTERS:
            walk(history + letter)

    walk("")
    return found


def game_value(strategy):
    """Expected payoff to player 1 when both players follow `strategy` ({key: array of 2})."""

    def walk(cards, history):
        if is_terminal(history):
            return payoff(cards, history)
        p = to_move(history)
        probs = strategy[infoset_key(cards[p], history)]
        return sum(probs[a] * walk(cards, history + LETTERS[a]) for a in (PASS, BET))

    return sum(walk(cards, "") for cards in DEALS) / len(DEALS)


def best_response_value(strategy, player):
    """Expected payoff to `player` (0 or 1) who best-responds to the other player's `strategy`.

    The best response picks one action per information set, so it cannot
    depend on the opponent's card.
    """
    sign = 1.0 if player == 0 else -1.0

    def walk(card, history, weights):
        # weights[o] is the chance, times the opponent's own action probabilities,
        # of reaching `history` when the opponent holds card o.
        if is_terminal(history):
            total = 0.0
            for o in CARDS:
                if o != card:
                    cards = (card, o) if player == 0 else (o, card)
                    total += weights[o] * sign * payoff(cards, history)
            return total
        if to_move(history) == player:
            return max(walk(card, history + LETTERS[a], weights) for a in (PASS, BET))
        total = 0.0
        for a in (PASS, BET):
            w = weights.copy()
            for o in CARDS:
                w[o] *= strategy[infoset_key(o, history)][a]
            total += walk(card, history + LETTERS[a], w)
        return total

    start = np.full(len(CARDS), 1.0 / len(DEALS))
    return sum(walk(card, "", start) for card in CARDS)


def exploitability(strategy):
    """Sum of both players' best-response gains over their value under `strategy`."""
    v1 = game_value(strategy)
    gain1 = best_response_value(strategy, 0) - v1
    gain2 = best_response_value(strategy, 1) + v1
    return gain1 + gain2


def cfr_iteration(regret_sum, strategy_sum):
    """Run one full-tree iteration of CFR, updating both dicts in place.

    Both players play regret matching on `regret_sum`. Each visit to an
    information set adds the counterfactual regret of each action to
    `regret_sum`, and adds the acting player's own reach probability times the
    strategy to `strategy_sum`. Every deal uses the strategies from the start of
    the iteration. Returns the expected payoff to player 1.
    """
    current = {k: regret_matching(r) for k, r in regret_sum.items()}

    def walk(cards, history, reach):
        if is_terminal(history):
            return payoff(cards, history)
        p = to_move(history)
        key = infoset_key(cards[p], history)
        sigma = current[key]
        values = np.zeros(2)
        for a in (PASS, BET):
            r = reach.copy()
            r[p] *= sigma[a]
            values[a] = walk(cards, history + LETTERS[a], r)
        node = sigma @ values
        sign = 1.0 if p == 0 else -1.0
        regret_sum[key] += sign * reach[1 - p] * (values - node) / len(DEALS)
        strategy_sum[key] += reach[p] * sigma
        return node

    return sum(walk(cards, "", np.ones(2)) for cards in DEALS) / len(DEALS)


def cfr_kuhn(iters):
    """Run `iters` iterations of CFR. Return {"average": ..., "current": ...} strategy dicts."""
    keys = infosets()
    regret_sum = {k: np.zeros(2) for k in keys}
    strategy_sum = {k: np.zeros(2) for k in keys}
    for _ in range(iters):
        cfr_iteration(regret_sum, strategy_sum)
    average = {k: s / s.sum() for k, s in strategy_sum.items()}
    current = {k: regret_matching(r) for k, r in regret_sum.items()}
    return {"average": average, "current": current}


if __name__ == "__main__":
    out = cfr_kuhn(2000)
    avg = out["average"]
    print("game value of the average strategy:", round(game_value(avg), 5), "(exact: -1/18 = -0.05556)")
    print("exploitability, average strategy:", round(exploitability(avg), 5))
    print("exploitability, current strategy:", round(exploitability(out["current"]), 5))
    print("player 1 bets the jack with probability", round(float(avg["0"][BET]), 3),
          "and the king with", round(float(avg["2"][BET]), 3), "(the king should bet about 3x as often)")
