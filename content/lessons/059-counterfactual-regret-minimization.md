---
title: Counterfactual regret minimization
slug: 059-counterfactual-regret-minimization
part: "Part 8 — Rewards, evaluation, and many agents"
summary: Solve Kuhn poker with regret matching at every information set, and see why only the average strategy converges.
minutes: 50
gpu: false
objectives:
  - Define an information set, and say why a strategy in a hidden-card game maps information sets to actions, not histories to actions.
  - Compute counterfactual regret from the opponent's reach probability, and say why the player's own reach does not belong in it.
  - Implement CFR with regret matching at each information set, and report the average strategy.
  - Measure exploitability with exact best responses, and check the strategy against Kuhn's equilibrium family and the game value of -1/18.
  - Explain why self-play with a changing current policy can cycle in imperfect-information games.
prereqs:
  - 057-multi-agent-and-self-play
exercise: 059-counterfactual-regret-minimization
---

# Counterfactual regret minimization

You hold a king in a card game and bet. Your opponent holds a card you cannot see, and you do not know whether they will call. Should you bet with a jack too, as a bluff? If you never bluff, a smart opponent folds every time you bet, so your bets stop earning money. If you always bluff, they call every time. The best answer is a *mix*, and the right mix depends on how the opponent mixes.

Chess and Go have no such problem, because both players see the whole board. In *imperfect-information* games such as poker, each player sees only part of the state. A player must pick one action for every situation that looks the same from where they sit.

This chapter builds *counterfactual regret minimization* (CFR), the algorithm behind superhuman poker programs. It runs the regret matching of the previous chapter once per situation, and it weights each regret by how likely the *opponent* made that situation happen. You run it on Kuhn poker, the smallest poker game that still needs bluffing, and check the answer against a game value that is known exactly.

## What you already know

- Regret for an action is how much better you would have done by always playing it. Regret matching plays each action in proportion to its positive regret ([Many agents and self-play](/lessons/057-multi-agent-and-self-play)).
- In a two-player zero-sum game, the *average* of the strategies converges to a Nash equilibrium while the latest strategy keeps moving.
- Exploitability adds the two players' best-response gains, and it is zero only at a Nash equilibrium.
- The perfect-information case, such as [AlphaZero's](/lessons/058-alphazero-self-play) board games, lets both players see everything. This chapter removes that.

## Kuhn poker and information sets

Kuhn poker (Kuhn, 1950) uses three cards: jack (J, written 0), queen (Q, 1), and king (K, 2). Each player antes 1 chip and gets one card. Player 1 acts first and can *pass* or *bet* 1 chip. If player 1 passes, player 2 can pass (a showdown for the antes) or bet. If anyone bets, the other player can *fold*, which gives up the pot, or *call* by betting 1 too, which leads to a showdown for 2 chips each. The higher card wins at a showdown.

A *history* is the sequence of actions so far, written with `p` for pass and `b` for bet. The five terminal histories are `pp`, `bp`, `bb`, `pbp`, and `pbb`. Only the hands are hidden, so what player 2 knows after player 1 bets is their own card and the history `b`.

An *information set* is a group of situations that one player cannot tell apart. Player 2 holding a queen after history `b` is in one information set, whichever card player 1 holds. A *strategy* gives, for each information set, a probability for each action. It has to, because a player cannot act differently in two situations they cannot tell apart.

Player 1 acts at the first move and after `pb`, and player 2 acts after `p` and after `b`. That is 4 histories times 3 cards, which is 12 information sets, six per player.

```figure
<svg viewBox="0 0 580 300" role="img" aria-label="The 12 information sets of Kuhn poker, with the probability that the average strategy bets or calls in each">
<text x="249.0" y="34" class="fig-bold" text-anchor="middle">hold J</text>
<text x="367.0" y="34" class="fig-bold" text-anchor="middle">hold Q</text>
<text x="485.0" y="34" class="fig-bold" text-anchor="middle">hold K</text>
<text x="180.0" y="74.0" class="fig-small fig-bold" text-anchor="end">player 1, first move</text>
<text x="180.0" y="92.0" class="fig-small" text-anchor="end">history: (none)</text>
<rect x="194.0" y="54.0" width="110.0" height="48.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="249.0" y="75.0" class="fig-small" text-anchor="middle">bet</text>
<text x="249.0" y="93.0" class="fig-bold" text-anchor="middle">0.21</text>
<rect x="312.0" y="54.0" width="110.0" height="48.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="367.0" y="75.0" class="fig-small" text-anchor="middle">bet</text>
<text x="367.0" y="93.0" class="fig-bold" text-anchor="middle">0.00</text>
<rect x="430.0" y="54.0" width="110.0" height="48.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="485.0" y="75.0" class="fig-small" text-anchor="middle">bet</text>
<text x="485.0" y="93.0" class="fig-bold" text-anchor="middle">0.61</text>
<text x="180.0" y="130.0" class="fig-small fig-bold" text-anchor="end">player 2, after pass</text>
<text x="180.0" y="148.0" class="fig-small" text-anchor="end">history: p</text>
<rect x="194.0" y="110.0" width="110.0" height="48.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="249.0" y="131.0" class="fig-small" text-anchor="middle">bet</text>
<text x="249.0" y="149.0" class="fig-bold" text-anchor="middle">0.33</text>
<rect x="312.0" y="110.0" width="110.0" height="48.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="367.0" y="131.0" class="fig-small" text-anchor="middle">bet</text>
<text x="367.0" y="149.0" class="fig-bold" text-anchor="middle">0.00</text>
<rect x="430.0" y="110.0" width="110.0" height="48.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="485.0" y="131.0" class="fig-small" text-anchor="middle">bet</text>
<text x="485.0" y="149.0" class="fig-bold" text-anchor="middle">1.00</text>
<text x="180.0" y="186.0" class="fig-small fig-bold" text-anchor="end">player 2, after bet</text>
<text x="180.0" y="204.0" class="fig-small" text-anchor="end">history: b</text>
<rect x="194.0" y="166.0" width="110.0" height="48.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="249.0" y="187.0" class="fig-small" text-anchor="middle">call</text>
<text x="249.0" y="205.0" class="fig-bold" text-anchor="middle">0.00</text>
<rect x="312.0" y="166.0" width="110.0" height="48.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="367.0" y="187.0" class="fig-small" text-anchor="middle">call</text>
<text x="367.0" y="205.0" class="fig-bold" text-anchor="middle">0.34</text>
<rect x="430.0" y="166.0" width="110.0" height="48.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="485.0" y="187.0" class="fig-small" text-anchor="middle">call</text>
<text x="485.0" y="205.0" class="fig-bold" text-anchor="middle">1.00</text>
<text x="180.0" y="242.0" class="fig-small fig-bold" text-anchor="end">player 1, after pass, bet</text>
<text x="180.0" y="260.0" class="fig-small" text-anchor="end">history: pb</text>
<rect x="194.0" y="222.0" width="110.0" height="48.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="249.0" y="243.0" class="fig-small" text-anchor="middle">call</text>
<text x="249.0" y="261.0" class="fig-bold" text-anchor="middle">0.00</text>
<rect x="312.0" y="222.0" width="110.0" height="48.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="367.0" y="243.0" class="fig-small" text-anchor="middle">call</text>
<text x="367.0" y="261.0" class="fig-bold" text-anchor="middle">0.55</text>
<rect x="430.0" y="222.0" width="110.0" height="48.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="485.0" y="243.0" class="fig-small" text-anchor="middle">call</text>
<text x="485.0" y="261.0" class="fig-bold" text-anchor="middle">1.00</text>
</svg>
Figure 1. The 12 information sets of Kuhn poker. Each cell holds the probability that the average strategy from the exercise bets (or calls, after a bet) after 10,000 iterations. Notice that the cells that hold a fraction are the ones the equilibrium table lists as fractions, and that the king bets or calls almost always while the jack folds to a bet.
```

The solution to Kuhn poker is known. The value of the game is $-1/18$ for player 1, which is player 1's expected payoff per hand under optimal play. Player 1 has a family of equilibrium strategies, indexed by a bluff frequency $\alpha$ between 0 and $1/3$:

| Information set | Equilibrium probability |
|---|---|
| Player 1, jack, first move | bet with $\alpha$ |
| Player 1, queen, first move | pass |
| Player 1, king, first move | bet with $3\alpha$ |
| Player 1, jack after `pb` | fold |
| Player 1, queen after `pb` | call with $\alpha + 1/3$ |
| Player 1, king after `pb` | call |
| Player 2, king (either history) | bet or call |
| Player 2, queen after `b` | call with $1/3$ |
| Player 2, queen after `p` | pass |
| Player 2, jack after `b` | fold |
| Player 2, jack after `p` | bet with $1/3$ |

The strategy in Figure 1 has $\alpha = 0.21$ and lands on this family.

## Why not treat a whole strategy as one action

The previous chapter's regret matching would treat each pure strategy as one action. A pure strategy here is a choice at each of a player's six information sets, so each player has $2^6 = 64$ of them, and the count doubles with every information set a game adds. CFR instead keeps one small regret vector per information set. That raises a question: when you compare two actions at one information set, what is the right measure of "how much better"?

## Counterfactual regret

Take one information set $I$ of player $i$ and one action $a$ there. Compare two worlds. In the first, player $i$ follows their strategy $\sigma$. In the second, they play $a$ at $I$ and follow $\sigma$ everywhere else. The *counterfactual value* $v_i(I, a)$ is player $i$'s expected payoff in the second world, counting only the part of the game that passes through $I$, weighted by the *opponent's reach*: the chance that the opponent's own choices, and the deal, bring the game to $I$. The player's own actions to reach $I$ are *not* in the weight, because they pretend the player steered there on purpose.

Why drop the player's own reach? You are judging the decision at $I$ as if you were sure to arrive. Weighting by your own reach would shrink the lesson at $I$ whenever your own strategy rarely goes there.

The counterfactual regret of $a$ at $I$ on iteration $t$ is the gap between $a$ and what $\sigma$ earns:

$$
r^t(I, a) = v_i^t(I, a) - \sum_{a'} \sigma^t(a' \mid I)\, v_i^t(I, a')
$$

Read it as "how much more I would have earned at this information set by playing $a$ for sure", weighted by how often the opponent lets us get here. Zinkevich and colleagues (2007) sum these over iterations, play regret matching on the sum, and prove that the average strategy approaches equilibrium.

### One information set by hand

Both players start uniform. Player 1 holds the queen, player 2 passed, then player 1 passes, then player 2 bets: the information set `1pb`. Player 1 can fold (pass) or call (bet).

Player 2 holds the jack or the king, each with chance $1/6$ as a deal, and bets after a pass with probability $1/2$. So the opponent's reach, deal included, is $\tfrac16 \cdot \tfrac12 = \tfrac1{12}$ for each card. Folding loses 1 against either card. Calling wins 2 against the jack and loses 2 against the king.

$$
v(\text{fold}) = \tfrac1{12}(-1) + \tfrac1{12}(-1) = -\tfrac16, \qquad
v(\text{call}) = \tfrac1{12}(+2) + \tfrac1{12}(-2) = 0
$$

Under the uniform strategy, the value at the set is $\tfrac12(-\tfrac16) + \tfrac12(0) = -\tfrac1{12}$. So the regrets are

$$
r(\text{fold}) = -\tfrac16 + \tfrac1{12} = -\tfrac1{12}, \qquad
r(\text{call}) = 0 + \tfrac1{12} = +\tfrac1{12}
$$

After one iteration, regret matching calls with probability 1. That is too eager, and later iterations correct it as the opponent's strategy changes. The exercise checks this number: after one iteration from uniform play, the regret at `1pb` is $(-1/12, +1/12)$.

### The two ingredients: strategy and average

Each iteration does a full walk of the game tree for every deal. At each information set, the walk computes the current strategy from the regret sums by regret matching, recurses on both actions, and adds each action's counterfactual regret to a running sum. It also adds the acting player's *own* reach times the current strategy to a second running sum. That second sum is where the output comes from:

$$
\bar\sigma(a \mid I) = \frac{\sum_t \pi_i^t(I)\, \sigma^t(a \mid I)}{\sum_t \pi_i^t(I)}
$$

Here $\pi_i^t(I)$ is player $i$'s own reach probability of $I$ on iteration $t$. Weight each iteration's strategy by how likely the player was to get there, so an iteration that barely reaches $I$ counts for little at $I$. The reports in this chapter always use $\bar\sigma$.

```python
# one step of the walk, at an information set with strategy sigma
values = np.array([walk(child_a) for child_a in children])   # payoff to player 1
node = sigma @ values
regret_sum[key] += sign * opponent_reach * (values - node) / 6
strategy_sum[key] += own_reach * sigma
```

Two details matter. All six deals use the strategies from the *start* of the iteration, so compute `sigma` for every set before you walk. And `sign` is $+1$ for player 1 and $-1$ for player 2, because the walk returns player 1's payoff.

## Exploitability by exact best response

You cannot read quality off a strategy table, so measure it. A *best response* to a strategy is the strategy that earns the most against it. In an imperfect-information game, a best response still picks one action per information set, so it cannot peek at the opponent's card. To compute it exactly, visit the tree once for each card the best-responder holds. Carry a weight for each card the opponent might hold, which starts at $1/6$ and gets multiplied by the opponent's action probabilities. At a terminal, add the weighted payoffs. At the best-responder's own decision, take the *max* over actions, outside the sum over the opponent's cards. At the opponent's decision, sum over their actions.

The *gain* of a player is their best-response value minus what they earn under the strategy. Because the game is zero-sum, the two gains sum to exploitability:

$$
\operatorname{expl}(\sigma) = \big(\mathrm{BR}_1(\sigma_2) - v_1(\sigma)\big) + \big(\mathrm{BR}_2(\sigma_1) - v_2(\sigma)\big)
$$

At a Nash equilibrium, each player already plays a best response, so both gains are 0. Figure 2 shows what CFR does to it.

```figure
<svg viewBox="0 0 600 280" role="img" aria-label="Exploitability of CFR on Kuhn poker over 10 to 10,000 iterations: the average strategy falls, the current strategy stays high">
<line x1="80.0" y1="230.0" x2="560.0" y2="230.0" class="stroke-grid"/>
<text x="72.0" y="234.0" class="fig-small" text-anchor="end">0.001</text>
<line x1="80.0" y1="163.3" x2="560.0" y2="163.3" class="stroke-grid"/>
<text x="72.0" y="167.3" class="fig-small" text-anchor="end">0.01</text>
<line x1="80.0" y1="96.7" x2="560.0" y2="96.7" class="stroke-grid"/>
<text x="72.0" y="100.7" class="fig-small" text-anchor="end">0.1</text>
<line x1="80.0" y1="30.0" x2="560.0" y2="30.0" class="stroke-grid"/>
<text x="72.0" y="34.0" class="fig-small" text-anchor="end">1</text>
<line x1="80.0" y1="230.0" x2="80.0" y2="235.0" class="stroke-muted"/>
<text x="80.0" y="250.0" class="fig-small" text-anchor="middle">10</text>
<line x1="240.0" y1="230.0" x2="240.0" y2="235.0" class="stroke-muted"/>
<text x="240.0" y="250.0" class="fig-small" text-anchor="middle">100</text>
<line x1="400.0" y1="230.0" x2="400.0" y2="235.0" class="stroke-muted"/>
<text x="400.0" y="250.0" class="fig-small" text-anchor="middle">1,000</text>
<line x1="560.0" y1="230.0" x2="560.0" y2="235.0" class="stroke-muted"/>
<text x="560.0" y="250.0" class="fig-small" text-anchor="middle">10,000</text>
<line x1="80.0" y1="230.0" x2="560.0" y2="230.0" class="stroke-muted"/>
<polyline points="80.0,77.7 156.3,94.6 240.0,116.0 316.3,134.5 400.0,152.5 476.3,168.4 560.0,185.6" class="stroke-a" fill="none"/>
<polyline points="80.0,57.9 156.3,42.0 240.0,52.7 316.3,54.9 400.0,54.4 476.3,53.2 560.0,65.0" class="stroke-b dashed" fill="none"/>
<circle cx="80.0" cy="77.7" r="4" class="fill-a"/>
<circle cx="80.0" cy="57.9" r="4" class="fill-b"/>
<circle cx="156.3" cy="94.6" r="4" class="fill-a"/>
<circle cx="156.3" cy="42.0" r="4" class="fill-b"/>
<circle cx="240.0" cy="116.0" r="4" class="fill-a"/>
<circle cx="240.0" cy="52.7" r="4" class="fill-b"/>
<circle cx="316.3" cy="134.5" r="4" class="fill-a"/>
<circle cx="316.3" cy="54.9" r="4" class="fill-b"/>
<circle cx="400.0" cy="152.5" r="4" class="fill-a"/>
<circle cx="400.0" cy="54.4" r="4" class="fill-b"/>
<circle cx="476.3" cy="168.4" r="4" class="fill-a"/>
<circle cx="476.3" cy="53.2" r="4" class="fill-b"/>
<circle cx="560.0" cy="185.6" r="4" class="fill-a"/>
<circle cx="560.0" cy="65.0" r="4" class="fill-b"/>
<text x="316.3" y="161.5" class="fig-small fig-bold">average strategy</text>
<text x="92.7" y="29.1" class="fig-small fig-bold">current strategy</text>
<text x="320.0" y="274" class="fig-small" text-anchor="middle">iterations T (log scale)</text>
<text x="80.0" y="18" class="fig-small" text-anchor="start">exploitability (log scale)</text>
</svg>
Figure 2. Exploitability on Kuhn poker. The average strategy (solid) falls from 0.19 at 10 iterations to 0.0046 at 10,000, and its game value lands within 0.00001 of -1/18. Notice that the current strategy (dashed) stays above 0.29 the whole time, and never settles.
```

> [!KEY]
> CFR runs regret matching at every information set, with each regret weighted by the opponent's reach. The average strategy converges to equilibrium. The current strategy does not.

## Why the current strategy keeps cycling

The dashed line in Figure 2 is the current strategy jumping between near-pure strategies, as the latest action in rock-paper-scissors did in the previous chapter. Suppose player 1 learns that player 2 never bluffs, so player 1 stops calling with the queen. Then player 2 gains from bluffing, which makes player 1 want to call again. A learner that chases the opponent's latest behavior circles forever, because the equilibrium itself is a mix.

Perfect-information games such as chess have no such trap. There, a deterministic best strategy exists, and self-play can settle on it, and the last policy is enough. With hidden cards, self-play with a current policy can cycle, and the equilibrium shows up only in the average. That is why systems like neural fictitious self-play (Heinrich and Silver, 2016) train a second network to imitate the average policy, and why Deep CFR (Brown and colleagues, 2019) trains one network for the regrets and another for the average strategy.

> [!RECAP]
> - An information set groups the situations a player cannot tell apart. A strategy gives one action distribution per set. Kuhn poker has 12.
> - Counterfactual regret weights each action's advantage by the opponent's reach, which includes the deal, and never by the player's own.
> - CFR adds regrets over iterations, plays regret matching on them, and reports the average strategy, weighted by the player's own reach.
> - Exploitability is the sum of both exact best-response gains, and it is 0 only at an equilibrium. A best response picks one action per information set.
> - Kuhn poker's value is $-1/18$ for player 1, and its equilibria form a one-parameter family.

## Build it step by step

The starter writes the game for you: `is_terminal`, `to_move`, `payoff`, and `infoset_key`. A strategy is a dict that maps a key such as `"1pb"` to an array of two probabilities, pass first. The checks test the functions in this order.

1. **`regret_matching(regrets)`.** Clip at zero, normalize, and return the uniform vector when the sum is zero. The common mistake is normalizing the raw regrets, which gives negative probabilities.
2. **`infosets()`.** Walk the histories from `""`, and for each non-terminal history and each card, record `infoset_key(card, history)` with `to_move(history)`. Return a dict from key to player.
3. **`game_value(strategy)`.** Return player 1's expected payoff, averaged over the six deals.
4. **`best_response_value(strategy, player)`.** Fix the best-responder's card, carry one weight per opponent card, and take the max at the best-responder's decisions. The common mistake is maximizing inside each deal, which lets the best response see the opponent's card and overstates its value.
5. **`exploitability(strategy)`.** Add the two gains. Player 2's value is minus `game_value`.
6. **`cfr_iteration(regret_sum, strategy_sum)`.** One full walk over all six deals, updating both dicts in place and returning player 1's expected payoff. The common mistake is weighting regret by the player's own reach, which looks plausible and passes the first iteration from uniform play, but not the check that starts from an uneven strategy.
7. **`cfr_kuhn(iters)`.** Call `cfr_iteration` `iters` times, and return both the average strategy and the current one in a dict. The common mistake is returning the current one as the answer.

## Cost and scaling

One iteration visits every node of the game tree. Kuhn poker has $6 \times 9 = 54$ visits per iteration, so 10,000 iterations take about a second in plain Python. Heads-up no-limit Texas hold'em has about $10^{161}$ decision points, so no one can walk it, and practical solvers group similar hands and sample branches instead.

## Where this breaks

**Large games.** Libratus (Brown and Sandholm, 2018) solved an abstracted game in advance, then re-solved subgames during play.

**Convergence speed.** The average's exploitability falls roughly like $1/\sqrt{T}$, so each extra digit costs about 100 times more iterations.

**More than two players, or general-sum payoffs.** The guarantee is for two-player zero-sum games. With three players, CFR still runs, but its average does not have to be a Nash equilibrium.

## Further reading

- [Neller and Lanctot (2013), An Introduction to Counterfactual Regret Minimization](http://modelai.gettysburg.edu/2013/cfr/index.html). A teaching introduction that works CFR on Kuhn poker.
- [Zinkevich, Johanson, Bowling, and Piccione (2007), Regret Minimization in Games with Incomplete Information](https://proceedings.neurips.cc/paper_files/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf). The CFR paper, with the convergence proof.
- [Kuhn (1950), A Simplified Two-Person Poker](https://sites.math.rutgers.edu/~zeilberg/akherim/PokerPapers/Kuhn1951.pdf). The game, and its equilibrium family and value of $-1/18$.
- [Silver, UCL course on reinforcement learning (2015), Lecture 10: Case Study: RL in Classic Games](https://davidstarsilver.wordpress.com/teaching/). Search and self-play in board and card games, the lecture after exploration and exploitation.
- [Heinrich and Silver (2016), Deep Reinforcement Learning from Self-Play in Imperfect-Information Games](https://arxiv.org/abs/1603.01121). Neural fictitious self-play, which learns an average policy with a network.
- [Brown, Lerer, Gross, and Sandholm (2019), Deep Counterfactual Regret Minimization](https://arxiv.org/abs/1811.00164). CFR with neural networks in place of tables.
- [Brown and Sandholm (2018), Superhuman AI for heads-up no-limit poker: Libratus beats top professionals](https://www.science.org/doi/10.1126/science.aao1733). CFR at the scale of a real poker game.

## The exercise

You build the whole pipeline for Kuhn poker. Write `regret_matching` and `infosets`, then `game_value`, `best_response_value`, and `exploitability`. Finish with `cfr_iteration` and `cfr_kuhn`, which returns the average and the current strategies.

The checks confirm 12 information sets, a game value of $-1/18$ on the equilibrium family, and a `best_response_value` that agrees with a brute-force search over a player's 64 pure strategies. After one iteration from an uneven strategy, each set's regret must match an independent calculation, which catches regret weighted by the wrong reach. After 10,000 iterations, the average strategy must have exploitability below 0.01, a game value within $10^{-3}$ of $-1/18$, and the family's shape, such as the king betting about three times as often as the jack.
