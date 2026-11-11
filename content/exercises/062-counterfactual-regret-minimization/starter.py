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
    """Return the strategy that plays each action in proportion to its positive regret.

    `regrets` is a 1-D array. Return a probability vector of the same length,
    uniform when no regret is positive.
    """
    raise NotImplementedError("regret_matching: clip at zero, normalize, fall back to uniform")


def infosets():
    """Return {key: player} for all 12 information sets.

    A key comes from `infoset_key(card, history)`, and the player from
    `to_move(history)`. Walk the histories from "" and skip terminal ones.
    """
    raise NotImplementedError("infosets: walk the histories from the empty one")


def game_value(strategy):
    """Return the expected payoff to player 1 when both players follow `strategy`.

    `strategy` maps each information-set key to an array of two probabilities
    (pass, bet). Average over the six deals.
    """
    raise NotImplementedError("game_value: recurse over the history for each deal")


def best_response_value(strategy, player):
    """Return the expected payoff to `player` (0 or 1) who best-responds to the other player's `strategy`.

    The best response picks one action per information set, so it cannot
    depend on the opponent's card. Fix the best-responder's card, carry one
    weight per opponent card, and take the max over actions outside the sum
    over opponent cards. Average over the best-responder's three cards.
    """
    raise NotImplementedError("best_response_value: max at your own sets, sum at the opponent's")


def exploitability(strategy):
    """Return the sum of both players' best-response gains under `strategy`.

    A gain is the best-response value minus the value of `strategy` to that player.
    """
    raise NotImplementedError("exploitability: add the two gains")


def cfr_iteration(regret_sum, strategy_sum):
    """Run one full-tree iteration of CFR, updating both dicts in place.

    `regret_sum` and `strategy_sum` map each key to an array of two numbers.
    Both players play regret matching on `regret_sum`, using the values from the
    start of the iteration for all six deals. At each visit to an information
    set, add the counterfactual regret of each action to `regret_sum` (the
    opponent's reach, deal included, times the action's value minus the node's
    value, with the sign flipped for player 2), and add the acting player's own
    reach times the strategy to `strategy_sum`. Return player 1's expected payoff.
    """
    raise NotImplementedError("cfr_iteration: one walk per deal, then update both dicts")


def cfr_kuhn(iters):
    """Run `iters` iterations of CFR from zero regrets.

    Return {"average": strategy, "current": strategy}. The average divides each
    `strategy_sum` row by its total. The current one is regret matching on the
    final `regret_sum`.
    """
    raise NotImplementedError("cfr_kuhn: call cfr_iteration in a loop, then normalize")


if __name__ == "__main__":
    out = cfr_kuhn(2000)
    avg = out["average"]
    print("game value of the average strategy:", round(game_value(avg), 5), "(exact: -1/18 = -0.05556)")
    print("exploitability, average strategy:", round(exploitability(avg), 5))
    print("exploitability, current strategy:", round(exploitability(out["current"]), 5))
    print("player 1 bets the jack with probability", round(float(avg["0"][BET]), 3),
          "and the king with", round(float(avg["2"][BET]), 3), "(the king should bet about 3x as often)")
