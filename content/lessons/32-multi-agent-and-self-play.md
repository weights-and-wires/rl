---
title: Multi-agent learning and self-play
slug: 32-multi-agent-and-self-play
part: "Part 7 — Rewards, evaluation, and many agents"
summary: Drive matrix games to Nash equilibrium with regret matching and fictitious play, measure exploitability, and learn Nim by self-play.
minutes: 55
gpu: false
objectives:
  - Define a Nash equilibrium of a two-player zero-sum game, and measure distance from one with exploitability.
  - Implement regret matching, and prove that its average play's exploitability equals total regret divided by rounds.
  - Implement fictitious play, and explain why averages converge while the latest strategies keep cycling.
  - Train one Q table to play Nim against itself with a negamax target, and check it against the known winning rule.
prereqs:
  - 02-multi-armed-bandits
  - 07-temporal-difference-learning
exercise: 32-multi-agent-and-self-play
---

# Multi-agent learning and self-play

Every lesson so far had one learner in a world that holds still. The dynamics
$P$ never changed while the agent learned. Put two learners in the same world,
and that stops being true. Each one is part of the other's environment, so
when one improves, the world the other faces changes under its feet.

This is how the most famous RL systems were trained. TD-Gammon, AlphaGo
Zero, and AlphaZero learned by playing against copies of themselves, and
AlphaStar trained a whole league. Self-play gives an agent an opponent that is
always exactly as strong as it is.

It also brings a new failure: learners that chase each other in circles
forever. You build two classic algorithms that provably stop the chase,
regret matching and fictitious play, and a yardstick, exploitability, that
says how close you are. Then one Q table learns Nim by playing itself.

## What you already know

- A policy can be stochastic: it picks actions with probabilities, not
  always the same one.
- A bandit's regret is how much reward you lost compared with always playing
  the best arm (lesson 2).
- Q-learning bootstraps from the best next value (lesson 7).
- Epsilon-greedy exploration takes a random action with probability
  $\epsilon$.
- A dot product $x^\top A y$ with a matrix in the middle weighs every entry
  $A_{ij}$ by $x_i y_j$.

## Many learners make the world move

When several agents learn at once, three patterns dominate practice:

- **Self-play** trains an agent against copies or past versions of itself.
- **Centralised training with decentralised execution** gives critics
  information about every agent during training, as in MADDPG and QMIX.
- **Population training** keeps a league of opponents, so the agent can't
  overfit to one.

All three fight the same moving target. The cleanest place to see it is a
matrix game.

## Matrix games and Nash equilibria

In a two-player *zero-sum* matrix game, the row player picks a row $i$, and
the column player picks a column $j$, at the same time. The row player wins
$A_{ij}$, and the column player loses the same amount. A *mixed strategy* is a
probability vector: $x$ over rows and $y$ over columns. The row player's
expected winnings are

$$
u(x, y) = x^\top A y = \sum_{i, j} x_i A_{ij} y_j
$$

Read it aloud: average the payoff table, weighting each cell by the chance
that both players land on it. The row player wants it high and the column
player wants it low. `x @ A @ y` computes it.

In rock-paper-scissors (RPS), the rows and columns are rock, paper, and
scissors. A win pays 1, a loss pays $-1$, and a tie pays 0.

A *best response* to $y$ is a strategy that maximises $x^\top A y$. The pure
row with the largest entry of $A y$ is always one. A *Nash equilibrium* is a
pair $(x^*, y^*)$ in which each strategy is a best response to the other, so
neither player gains by changing alone. Von Neumann's minimax theorem says
every zero-sum matrix game has one, and all its equilibria give the same
payoff, called the *value* of the game. In RPS, the equilibrium is uniform play
for both players, and the value is 0.

### Worked example: a 2-by-2 game with a mixed equilibrium

Take $A = \begin{bmatrix} 3 & -1 \\ -2 & 1 \end{bmatrix}$. No pure pair is
stable: whatever cell you start in, one player wants to switch. At a mixed
equilibrium, each player mixes so that the *other* is indifferent. Suppose the
row player plays row 1 with probability $p$. The column player is
indifferent when both columns cost the same:

$$
3p - 2(1 - p) = -p + (1 - p) \quad\Rightarrow\quad 5p - 2 = 1 - 2p \quad\Rightarrow\quad p = 3/7
$$

The same argument for the column player's $q$ gives $3q - (1 - q) = -2q +
(1 - q)$, so $q = 2/7$. The equilibrium is $x^* = (3/7, 4/7)$ and $y^* =
(2/7, 5/7)$, with value $4q - 1 = 1/7$.

## Measure the distance: exploitability

You'll want a single number that says how far a pair of strategies is from
equilibrium. Ask how much each player could gain by switching to a best
response, and add the two gains:

$$
\operatorname{expl}(x, y) = \max_i\, (A y)_i - \min_j\, (x^\top A)_j
$$

Read it aloud: the first term is the most the row player could win against
$y$. The second is the least the column player could concede against $x$.
Their difference is the sum of both players' possible gains, since the
current payoff $x^\top A y$ cancels. It is zero exactly at a Nash equilibrium
and positive everywhere else.

```python
exploitability = np.max(A @ y) - np.min(x @ A)
```

For rock against rock, $A y$ is the rock column $(0, 1, -1)$, whose max is 1.
$x^\top A$ is the rock row $(0, -1, 1)$, whose min is $-1$. So the
exploitability is 2. This sum is often called *NashConv*. Some libraries, such
as OpenSpiel, divide it by the number of players and call that exploitability.

## Why naive self-play goes in circles

The simplest self-play rule is to best-respond to the opponent's latest
strategy. In RPS, that cycles forever. If your opponent last played rock, you
play paper. They respond with scissors, you respond with rock, and so on.
Every strategy in the cycle has exploitability 2, and nothing settles.
Simultaneous gradient steps for both players on $x^\top A y$ cycle too,
spiralling around the equilibrium instead of reaching it.

The two algorithms below fix this the same way: they make the *average* of
play converge, even though the latest strategy may keep moving.

> [!KEY]
> In games, the latest strategy often cycles. Regret matching and fictitious
> play make the average strategy converge to equilibrium instead.

## Regret matching

A player's *regret* for an action is how much better they'd have done by
always playing it, compared with what they actually played. For the row
player after $T$ rounds, it is a vector over rows:

$$
R^x_T(i) = \sum_{t=1}^{T} \Big[ (A y_t)_i - x_t^\top A y_t \Big]
$$

Read it aloud: for each row $i$, add up, over every round, what row $i$ would
have won against that round's opponent minus what you won. The column player
keeps the same sum with its own payoffs, $u^y_t = -x_t^\top A$, since it
loses what the row player wins.

*Regret matching* (Hart and Mas-Colell, 2000) plays each action in proportion
to its positive regret:

$$
x_{T+1}(i) = \frac{\max(R^x_T(i), 0)}{\sum_k \max(R^x_T(k), 0)}
$$

When no regret is positive, it plays uniformly. Read it aloud: favour the
actions you most wish you had played, and ignore the ones you're glad you
didn't. In numpy, one round for both players is the following:

```python
u_x, u_y = A @ y, -(x @ A)
regret_x += u_x - x @ u_x
regret_y += u_y - y @ u_y
x, y = rm_strategy(regret_x), rm_strategy(regret_y)
```

Regret matching guarantees that no action's regret grows faster than
$\Delta \sqrt{|\mathcal{A}| T}$. Here $\Delta$ is the range of payoffs, 2 in
RPS, and $|\mathcal{A}|$ is the number of actions. So regret per round shrinks
like $1 / \sqrt{T}$, whatever the opponent does.

### Why low regret means near-equilibrium

For zero-sum games, there is an exact link. Let $\bar x$ and $\bar y$ be the
averages of the strategies played over $T$ rounds. Then

$$
\operatorname{expl}(\bar x, \bar y) = \frac{\max_i R^x_T(i) + \max_j R^y_T(j)}{T}
$$

The proof is three lines. $A \bar y$ is the average of $A y_t$, so $\max_i (A
\bar y)_i$ is $1/T$ times the best row's total payoff. The same goes for
$\min_j (\bar x^\top A)_j$. Add and subtract the total payoff actually played,
$\sum_t x_t^\top A y_t$, and the two halves are exactly the two players'
largest regrets. Since both regrets grow at most like $\sqrt{T}$, the average
strategies' exploitability falls like $1 / \sqrt{T}$.

### Worked example: the first two rounds

Start both players on rock in RPS.

- Round 1 pays 0. The row player's regret is the rock column, $(0, 1, -1)$,
  minus 0: paper would have won, scissors would have lost. The column
  player's regret is the same by symmetry.
- Round 2: the only positive regret is on paper, so both play paper. Against
  paper, the rows pay $(-1, 0, 1)$, and the round pays 0. The regret becomes
  $(-1, 1, 0)$.
- The average strategy after two rounds is $(0.5, 0.5, 0)$.

Run it longer, and the averages close in on uniform play while the latest
strategies keep wandering:

| Rounds $T$ | Average $x$ | Expl. of averages | Expl. of latest |
|---|---|---|---|
| 10 | (0.231, 0.275, 0.494) | 0.439 | 0.741 |
| 100 | (0.382, 0.293, 0.325) | 0.114 | 0.834 |
| 1,000 | (0.325, 0.346, 0.329) | 0.041 | 2.000 |
| 10,000 | (0.338, 0.329, 0.333) | 0.010 | 0.854 |

On the 2-by-2 game, 20,000 rounds give averages of $(0.427, 0.573)$ and
$(0.288, 0.712)$, against the equilibrium's $(0.4286, 0.5714)$ and
$(0.2857, 0.7143)$.

> [!TIP]
> Test your implementation with the identity. If exploitability of the
> averages doesn't equal (max row regret + max column regret) / $T$ to
> rounding error, one of the payoff vectors has the wrong sign.

## Fictitious play

*Fictitious play* (Brown, 1951) is even simpler. Each round, each player
best-responds to the *empirical mixture* of everything the opponent has played
so far:

$$
i_{t+1} = \operatorname*{arg\,max}_i\, (A \bar y_t)_i, \qquad j_{t+1} = \operatorname*{arg\,min}_j\, (\bar x_t^\top A)_j
$$

Read it aloud: the row player treats the column player's history as a mixed
strategy and plays the best pure reply. The column player does the same from
its side. Robinson (1951) proved that in two-player zero-sum games, the
empirical mixtures approach equilibrium. The convergence can be slow, and the
pure actions played each round keep cycling.

In RPS from rock against rock, round 2 is paper against paper, because paper
beats the rock history. After 5,000 rounds, each player's mixture is within
0.004 of uniform. Fictitious play is the ancestor of neural fictitious
self-play (NFSP) and of the league training used for AlphaStar, where the
"empirical mixture" becomes a pool of past agents.

## Self-play with a value function: Nim

Most games have many moves in a row, and there you learn a value function by
self-play. Nim is small enough to check exactly.

A pile holds $n$ sticks. Players take turns removing 1, 2, or 3 sticks, and
whoever takes the last stick wins. The theory is known: piles that are a
multiple of 4 are lost for the player to move, and from any other pile you win
by taking $n \bmod 4$ sticks. That leaves your opponent a multiple of 4.

Use one table $Q(n, k)$ for both players. It holds the value of taking $k$
sticks from a pile of $n$, for whoever is about to move, with $+1$ for a win.
After your move, the opponent moves from $n - k$ using the same table. Their
best value is your loss, so the target is the *negamax* target:

$$
y = \begin{cases} +1 & \text{if } n - k = 0 \\ -\max_{k'} Q(n - k, k') & \text{otherwise} \end{cases}
$$

Read it aloud: emptying the pile wins outright. Otherwise, your value is
minus the best value your opponent can get from what you leave. That minus sign
is what turns single-agent Q-learning into two-player self-play in a zero-sum
game. With $\epsilon = 0.2$ exploration and $\alpha = 0.5$, 3,000 episodes
from random piles up to 15 are enough. The greedy move is $n \bmod 4$ at every
winning pile, and the values are exactly $+1$ and $-1$.

```python
left = n - k
target = 1.0 if left == 0 else -Q[left, legal[left]].max()
Q[n, k - 1] += alpha * (target - Q[n, k - 1])
```

> [!KEY]
> In a two-player zero-sum game, self-play Q-learning uses one table for both
> sides, and the opponent's best value enters the target with a minus sign.

> [!RECAP]
> - Exploitability is $\max_i (Ay)_i - \min_j (x^\top A)_j$. It is zero only
>   at a Nash equilibrium.
> - Regret matching plays each action in proportion to its positive
>   cumulative regret, uniform when none is positive.
> - The column player's payoff vector is $-x^\top A$. The average strategies,
>   not the latest ones, converge.
> - Fictitious play best-responds to the opponent's empirical mixture.
> - Self-play Nim uses the negamax target, $+1$ for emptying the pile or
>   minus the opponent's best value.

## Build it step by step

The checks test the functions in this order. `ROCK_PAPER_SCISSORS` and
`nim_legal` are written for you.

1. **`regret_matching_strategy(regrets)`.** Clip at zero with `np.maximum`,
   and normalise. Return the uniform vector when the sum is zero. Common
   mistake: normalising the raw regrets, which gives negative probabilities.
2. **`exploitability(A, x, y)`.** One line. The check expects 2 for rock
   against rock and 0 at both equilibria.
3. **`regret_matching(A, iters, x0, y0)`.** Play `x0` and `y0` in round 0.
   Add each round's strategies to running sums *before* updating regrets, and
   return the averages, the last strategies, and the regrets in a dict. Common
   mistake: $u^y = +x^\top A$, which makes the column player chase its own
   losses. The identity check catches it at once.
4. **`fictitious_play(A, iters, first_x, first_y)`.** Keep action counts.
   Compute both best responses from the counts *before* this round, then add
   both. Common mistake: updating the row player's counts first and letting
   the column player respond to them, which is a different algorithm.
5. **`nim_self_play(n_max, episodes, alpha, epsilon, seed)`.** Loop until
   the pile is empty, choosing among legal moves only, with random greedy
   tie-breaking. Common mistake: taking the max over illegal moves from a small
   pile, which reads entries that are never updated.

## Cost and scaling

The exercise's algorithms are cheap per round. The following table shows how
they grow.

| Method | Work per round | Rounds to exploitability $\epsilon$ |
|---|---|---|
| Regret matching | $O(nm)$ for an $n \times m$ game | $O(1/\epsilon^2)$ |
| Fictitious play | $O(nm)$ | Often faster in practice, slow in the worst case |
| Linear programming | One solve, polynomial in $n + m$ | Exact |
| Tabular self-play Nim | $O(1)$ per move | Grows with the number of positions |

For a single matrix game, linear programming finds the exact equilibrium.
Learning pays off when the game is too large to write down. Counterfactual
regret minimisation (CFR) runs regret matching at every decision point of a
poker game, and AlphaZero replaces the Q table with a network and search.

## Where this breaks

**General-sum games.** When one player's gain isn't the other's loss, regret
matching only drives the empirical distribution of joint play toward a *coarse
correlated equilibrium*, a weaker notion than Nash. Fictitious play can fail
to converge at all. Computing a
Nash equilibrium of a general game is believed to be computationally hard.

**Non-transitive strategy spaces.** In games like RPS, "better" isn't a line.
Self-play against only the latest copy can forget how to beat old strategies.
Leagues and population methods such as PSRO keep old opponents around for this
reason.

**Evaluating against yourself.** An agent that beats its own copies can still
lose to a fixed strategy it never met. Evaluate against fixed baselines too.

**Teams and credit assignment.** In cooperative games, it's hard to tell
which agent earned a shared reward. Methods such as QMIX factor the team's
value, but can't represent every team optimum.

## Further reading

- [Hart and Mas-Colell (2000), A Simple Adaptive Procedure Leading to
  Correlated Equilibrium](https://doi.org/10.1111/1468-0262.00153). Introduces
  regret matching and proves its convergence properties.
- [Robinson (1951), An Iterative Method of Solving a
  Game](https://doi.org/10.2307/1969530). The proof that fictitious play
  converges in two-player zero-sum games.
- [Zinkevich, Johanson, Bowling, and Piccione (2007), Regret Minimization in
  Games with Incomplete
  Information](https://proceedings.neurips.cc/paper_files/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf).
  CFR, which runs regret matching at every decision point of a poker game.
- [Heinrich and Silver (2016), Deep Reinforcement Learning from Self-Play in
  Imperfect-Information Games](https://arxiv.org/abs/1603.01121). Neural
  fictitious self-play, fictitious play with networks.
- [Lanctot and colleagues (2017), A Unified Game-Theoretic Approach to
  Multiagent Reinforcement Learning](https://arxiv.org/abs/1711.00832). PSRO,
  which grows a population of policies and mixes them by solving a meta-game.
- [Silver and colleagues (2017), Mastering Chess and Shogi by Self-Play with a
  General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815).
  AlphaZero, which learns from self-play alone with a network and tree search.
- [Lowe and colleagues (2017), Multi-Agent Actor-Critic for Mixed
  Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275).
  MADDPG, the standard example of centralised critics with decentralised
  actors.

Open questions a researcher might work on:

- Which learning dynamics make the *last* strategy converge in games, and
  not only the average, when the game is too large for exact methods?
- How should you rank agents in non-transitive games, where no single
  number can order them?
- How far do self-play methods transfer to settings with many learning
  agents that are neither fully cooperative nor fully competitive, such as
  language-model agents negotiating?

## The exercise

You build the tools for two-player zero-sum games. Write `exploitability`
and `regret_matching_strategy`, then `regret_matching` for both players at
once, and `fictitious_play`. Finish with `nim_self_play`, one Q table that
learns Nim by playing itself.

The checks confirm the hand computations: the regret-matching strategy, an
exploitability of 2 for rock against rock, and regret matching's first two
rounds. After 10,000 rounds of RPS, the average play must be exploitable by
under 0.04, equal to total regret over $T$ to $10^{-9}$, with each regret
under the $\Delta \sqrt{|\mathcal{A}| T}$ bound. The last strategies must
still be exploitable by more than 0.3. On the 2-by-2 game, regret matching and
fictitious play must both land within 0.02 of $(3/7, 4/7)$ and $(2/7, 5/7)$.
Nim self-play must take $n \bmod 4$ from every winning pile.
