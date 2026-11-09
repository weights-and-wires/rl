---
title: Many agents and self-play
slug: 048-multi-agent-and-self-play
part: "Part 8 — Rewards, evaluation, and many agents"
summary: Drive matrix games to Nash equilibrium with regret matching and fictitious play, then learn Nim by playing against yourself.
minutes: 40
gpu: false
objectives:
  - Say why two learners that each best-respond to the other's most recent move cycle forever.
  - Measure distance from a Nash equilibrium with exploitability.
  - Implement regret matching and fictitious play, and explain why their averages converge while their last strategies keep moving.
  - Train one Q table to play Nim against itself with a negamax target, and check it against the known winning rule.
prereqs:
  - 003-explore-or-exploit
  - 017-sarsa-and-q-learning
exercise: 048-multi-agent-and-self-play
---

# Many agents and self-play

Two players play rock-paper-scissors, and each one learns by answering the other's last move. You played rock, so I play paper. I played paper, so you play scissors. Where does this end?

It never ends. Every chapter so far had one learner in a world that holds still. Put two learners in the same world and each becomes part of the other's environment, so when one improves, the world the other faces changes. AlphaGo Zero and AlphaZero learned by playing copies of themselves, and AlphaStar trained a whole league, because self-play gives an agent an opponent that is always as strong as it is.

This chapter shows why chasing the most recent move fails, and the fix: make the *average* of play settle down even though the most recent move keeps moving. You also get a yardstick, exploitability, and then one Q table learns Nim by playing itself.

## What you already know

- A bandit's regret is how much reward you lost compared with always pulling the best arm.
- Q-learning bootstraps from the best next value.
- A dot product $x^\top A y$ with a matrix in the middle weighs every entry $A_{ij}$ by $x_i y_j$.

## Chasing the most recent move goes in circles

In a two-player *zero-sum* matrix game, the row player picks a row $i$ and the column player picks a column $j$ at the same time. The row player wins $A_{ij}$, and the column player loses the same amount. In rock-paper-scissors (RPS), a win pays 1, a loss pays $-1$, and a tie pays 0.

The simplest self-play rule is to best-respond to the opponent's most recent move. The opponent last played rock, so you play paper. They answer scissors, you answer rock, and the cycle of three repeats forever. Every pure move can be beaten for a full point by a best reply, so nothing settles.

The fix is to let a player mix. A *mixed strategy* is a probability vector: $x$ over rows and $y$ over columns. The row player's expected winnings are

$$
u(x, y) = x^\top A y = \sum_{i, j} x_i A_{ij} y_j
$$

Read it aloud: average the payoff table, weighting each cell by the chance that both players land on it. The row player wants it high and the column player wants it low. In numpy, that is `x @ A @ y`.

A *best response* to $y$ is a strategy that maximizes $x^\top A y$. A *Nash equilibrium* is a pair $(x^*, y^*)$ in which each strategy is a best response to the other, so neither player gains by changing alone. Every zero-sum matrix game has one, and all its equilibria pay the same, the *value* of the game. In RPS the equilibrium is uniform play for both players, with value 0.

### Find an equilibrium by hand

Take $A = \begin{bmatrix} 3 & -1 \\ -2 & 1 \end{bmatrix}$. No pure pair is stable, because from any cell one player wants to switch. At a mixed equilibrium, each player mixes so that the *other* is indifferent between their options. Let the column player play column 1 with probability $q$. The row player's two rows then pay $4q - 1$ and $1 - 3q$. They are equal at $q = 2/7$, and Figure 1 shows the crossing.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Row player's payoff for each row of the 2 by 2 game against the column player's probability q of column 1; the lines cross at q = 2/7 with payoff 1/7">
<line x1="80.0" y1="230.0" x2="560.0" y2="230.0" class="stroke-grid"/>
<text x="72.0" y="234.0" class="fig-small" text-anchor="end">-2</text>
<line x1="80.0" y1="190.0" x2="560.0" y2="190.0" class="stroke-grid"/>
<text x="72.0" y="194.0" class="fig-small" text-anchor="end">-1</text>
<line x1="80.0" y1="150.0" x2="560.0" y2="150.0" class="stroke-muted"/>
<text x="72.0" y="154.0" class="fig-small" text-anchor="end">0</text>
<line x1="80.0" y1="110.0" x2="560.0" y2="110.0" class="stroke-grid"/>
<text x="72.0" y="114.0" class="fig-small" text-anchor="end">1</text>
<line x1="80.0" y1="70.0" x2="560.0" y2="70.0" class="stroke-grid"/>
<text x="72.0" y="74.0" class="fig-small" text-anchor="end">2</text>
<line x1="80.0" y1="30.0" x2="560.0" y2="30.0" class="stroke-grid"/>
<text x="72.0" y="34.0" class="fig-small" text-anchor="end">3</text>
<line x1="80.0" y1="230.0" x2="80.0" y2="235.0" class="stroke-muted"/>
<text x="80.0" y="250.0" class="fig-small" text-anchor="middle">0</text>
<line x1="200.0" y1="230.0" x2="200.0" y2="235.0" class="stroke-muted"/>
<text x="200.0" y="250.0" class="fig-small" text-anchor="middle">0.25</text>
<line x1="320.0" y1="230.0" x2="320.0" y2="235.0" class="stroke-muted"/>
<text x="320.0" y="250.0" class="fig-small" text-anchor="middle">0.5</text>
<line x1="440.0" y1="230.0" x2="440.0" y2="235.0" class="stroke-muted"/>
<text x="440.0" y="250.0" class="fig-small" text-anchor="middle">0.75</text>
<line x1="560.0" y1="230.0" x2="560.0" y2="235.0" class="stroke-muted"/>
<text x="560.0" y="250.0" class="fig-small" text-anchor="middle">1</text>
<line x1="80.0" y1="190.0" x2="560.0" y2="30.0" class="stroke-a"/>
<line x1="80.0" y1="110.0" x2="560.0" y2="230.0" class="stroke-b"/>
<text x="99.2" y="206.0" class="fig-small fig-bold" text-anchor="start">row 1: 4q − 1</text>
<text x="99.2" y="90.0" class="fig-small fig-bold" text-anchor="start">row 2: 1 − 3q</text>
<circle cx="217.1" cy="144.3" r="6" class="fill-c"/>
<line x1="217.1" y1="144.3" x2="217.1" y2="230.0" class="stroke-c dashed"/>
<text x="227.1" y="190.0" class="fig-small fig-bold" text-anchor="start">q = 2/7, value 1/7</text>
<text x="320.0" y="266.0" class="fig-small" text-anchor="middle">q, the column player's chance of playing column 1</text>
</svg>
Figure 1. The row player's payoff from each pure row, as the column player's mix q changes. Notice that the lines cross at q = 2/7: only there is the row player indifferent between rows, so only there does the column player leave nothing to exploit.
```

The same argument for the row player's $p$ gives $p = 3/7$. The equilibrium is $x^* = (3/7, 4/7)$ and $y^* = (2/7, 5/7)$, with value $1/7$.

## Measure the distance: exploitability

You want one number for how far a pair of strategies is from equilibrium. Ask how much each player could gain by switching to a best response, and add the two gains. The row player's gain is $\max_i (Ay)_i - x^\top A y$ and the column player's is $x^\top A y - \min_j (x^\top A)_j$, so the payoff $x^\top A y$ cancels in the sum:

$$
\operatorname{expl}(x, y) = \max_i\, (A y)_i - \min_j\, (x^\top A)_j
$$

Read it aloud: the most the row player could win against $y$, minus the least the column player could concede against $x$. It is zero exactly at a Nash equilibrium and positive everywhere else.

```python
exploitability = np.max(A @ y) - np.min(x @ A)
```

For rock against rock, $A y$ is the rock column $(0, 1, -1)$, whose max is 1, and $x^\top A$ is the rock row $(0, -1, 1)$, whose min is $-1$. So the exploitability is 2.

## Regret matching

*Regret* for an action is how much better you would have done by always playing it, compared with what you played. For the row player after $T$ rounds, it is a vector over rows:

$$
R^x_T(i) = \sum_{t=1}^{T} \Big[ (A y_t)_i - x_t^\top A y_t \Big]
$$

Read it aloud: for each row $i$, add up what row $i$ would have won against each round's opponent, minus what you actually won. The column player keeps the same sum with its own payoffs, $u^y_t = -x_t^\top A$, since it loses what the row player wins.

*Regret matching* (Hart and Mas-Colell, 2000) plays each action in proportion to its positive regret, and plays uniformly when no regret is positive:

$$
x_{T+1}(i) = \frac{\max(R^x_T(i), 0)}{\sum_k \max(R^x_T(k), 0)}
$$

Read it aloud: favor the actions you most wish you had played, and ignore the ones you are glad you skipped. One round for both players is four lines of numpy:

```python
u_x, u_y = A @ y, -(x @ A)
regret_x += u_x - x @ u_x
regret_y += u_y - y @ u_y
x, y = rm_strategy(regret_x), rm_strategy(regret_y)
```

### The first two rounds

Start both players on rock in RPS. Round 1 pays 0, and the row player's regret is the rock column $(0, 1, -1)$, because paper would have won and scissors would have lost. In round 2 the only positive regret is on paper, so both play paper. Against paper the rows pay $(-1, 0, 1)$ and the round pays 0, so the regret becomes $(-1, 1, 0)$, and the average strategy is $(0.5, 0.5, 0)$.

### Why the average converges

Let $\bar x$ and $\bar y$ be the averages of the strategies played over $T$ rounds. For zero-sum games there is an exact link:

$$
\operatorname{expl}(\bar x, \bar y) = \frac{\max_i R^x_T(i) + \max_j R^y_T(j)}{T}
$$

The proof is short. $A \bar y$ is the average of $A y_t$, so $\max_i (A \bar y)_i$ is $1/T$ times the best row's total payoff, and likewise for $\min_j (\bar x^\top A)_j$. Add and subtract the total payoff actually played, and the two halves are the two players' largest regrets. Regret matching keeps every regret below $\Delta \sqrt{|\mathcal{A}| T}$, where $\Delta$ is the payoff range, so the averages' exploitability falls like $1/\sqrt{T}$. The last strategies carry no such guarantee, and Figure 2 shows both.

```figure
<svg viewBox="0 0 600 280" role="img" aria-label="Exploitability of regret matching's average strategies, falling, against its last strategies, which stay high, over 10 to 10000 rounds">
<line x1="80.0" y1="230.0" x2="560.0" y2="230.0" class="stroke-grid"/>
<text x="72.0" y="234.0" class="fig-small" text-anchor="end">0</text>
<line x1="80.0" y1="184.5" x2="560.0" y2="184.5" class="stroke-grid"/>
<text x="72.0" y="188.5" class="fig-small" text-anchor="end">0.5</text>
<line x1="80.0" y1="139.1" x2="560.0" y2="139.1" class="stroke-grid"/>
<text x="72.0" y="143.1" class="fig-small" text-anchor="end">1</text>
<line x1="80.0" y1="93.6" x2="560.0" y2="93.6" class="stroke-grid"/>
<text x="72.0" y="97.6" class="fig-small" text-anchor="end">1.5</text>
<line x1="80.0" y1="48.2" x2="560.0" y2="48.2" class="stroke-grid"/>
<text x="72.0" y="52.2" class="fig-small" text-anchor="end">2</text>
<line x1="80.0" y1="230.0" x2="80.0" y2="235.0" class="stroke-muted"/>
<text x="80.0" y="250.0" class="fig-small" text-anchor="middle">10</text>
<line x1="240.0" y1="230.0" x2="240.0" y2="235.0" class="stroke-muted"/>
<text x="240.0" y="250.0" class="fig-small" text-anchor="middle">100</text>
<line x1="400.0" y1="230.0" x2="400.0" y2="235.0" class="stroke-muted"/>
<text x="400.0" y="250.0" class="fig-small" text-anchor="middle">1,000</text>
<line x1="560.0" y1="230.0" x2="560.0" y2="235.0" class="stroke-muted"/>
<text x="560.0" y="250.0" class="fig-small" text-anchor="middle">10,000</text>
<line x1="80.0" y1="230.0" x2="560.0" y2="230.0" class="stroke-muted"/>
<polyline points="80.0,190.1 128.2,202.4 191.8,215.0 240.0,219.7 288.2,221.6 351.8,225.3 400.0,226.3 448.2,227.9 511.8,228.4 560.0,229.1" class="stroke-a" fill="none"/>
<polyline points="80.0,162.6 128.2,48.2 191.8,152.1 240.0,154.2 288.2,48.2 351.8,103.8 400.0,48.2 448.2,123.9 511.8,91.9 560.0,152.4" class="stroke-b dashed" fill="none"/>
<circle cx="80.0" cy="190.1" r="4" class="fill-a"/>
<circle cx="128.2" cy="202.4" r="4" class="fill-a"/>
<circle cx="191.8" cy="215.0" r="4" class="fill-a"/>
<circle cx="240.0" cy="219.7" r="4" class="fill-a"/>
<circle cx="288.2" cy="221.6" r="4" class="fill-a"/>
<circle cx="351.8" cy="225.3" r="4" class="fill-a"/>
<circle cx="400.0" cy="226.3" r="4" class="fill-a"/>
<circle cx="448.2" cy="227.9" r="4" class="fill-a"/>
<circle cx="511.8" cy="228.4" r="4" class="fill-a"/>
<circle cx="560.0" cy="229.1" r="4" class="fill-a"/>
<circle cx="80.0" cy="162.6" r="4" class="fill-b"/>
<circle cx="128.2" cy="48.2" r="4" class="fill-b"/>
<circle cx="191.8" cy="152.1" r="4" class="fill-b"/>
<circle cx="240.0" cy="154.2" r="4" class="fill-b"/>
<circle cx="288.2" cy="48.2" r="4" class="fill-b"/>
<circle cx="351.8" cy="103.8" r="4" class="fill-b"/>
<circle cx="400.0" cy="48.2" r="4" class="fill-b"/>
<circle cx="448.2" cy="123.9" r="4" class="fill-b"/>
<circle cx="511.8" cy="91.9" r="4" class="fill-b"/>
<circle cx="560.0" cy="152.4" r="4" class="fill-b"/>
<text x="360.0" y="202.7" class="fig-small fig-bold" text-anchor="start">average strategies</text>
<text x="134.8" y="175.5" class="fig-small fig-bold" text-anchor="start">last strategies</text>
<text x="320.0" y="274.0" class="fig-small" text-anchor="middle">rounds T (log scale)</text>
<text x="80.0" y="20.0" class="fig-small" text-anchor="start">exploitability</text>
</svg>
Figure 2. Regret matching on rock-paper-scissors, starting from rock against rock. Notice that the average strategies' exploitability falls steadily toward 0 (0.010 at 10,000 rounds), while the last strategies keep bouncing and reach 2.0 at 1,000 rounds.
```

> [!KEY]
> In games, the strategy played each round often cycles. Regret matching makes the average strategy converge to equilibrium, because the average's exploitability equals the players' total regret divided by $T$.

## Fictitious play

*Fictitious play* (Brown, 1951) is simpler still. Each round, each player best-responds to the *empirical mixture* of everything the opponent has played so far:

$$
i_{t+1} = \operatorname*{arg\,max}_i\, (A \bar y_t)_i, \qquad j_{t+1} = \operatorname*{arg\,min}_j\, (\bar x_t^\top A)_j
$$

Read it aloud: treat the opponent's history as a mixed strategy, and play the best pure reply to it. Robinson (1951) proved that in two-player zero-sum games the empirical mixtures approach equilibrium, while the pure moves keep cycling. AlphaStar's league training is a descendant, with the empirical mixture replaced by a pool of past agents.

## Self-play with a value table: Nim

Most games have many moves in a row, and there you learn a value function by self-play. Nim is small enough to check exactly: a pile holds $n$ sticks, players alternate taking 1, 2, or 3, and whoever takes the last stick wins. Piles that are a multiple of 4 are lost for the player to move, and from any other pile you win by taking $n \bmod 4$.

Use one table $Q(n, k)$ for both players: the value of taking $k$ sticks from a pile of $n$ for whoever is about to move, with $+1$ for a win. After your move the opponent moves from $n - k$ with the same table, and their best value is your loss. That gives the *negamax* target:

$$
y = \begin{cases} +1 & \text{if } n - k = 0 \\ -\max_{k'} Q(n - k, k') & \text{otherwise} \end{cases}
$$

Read it aloud: emptying the pile wins outright, and otherwise your value is minus the best value your opponent can get from what you leave. With $\epsilon = 0.2$ and $\alpha = 0.5$, 3,000 episodes from random piles up to 15 find the greedy move $n \bmod 4$ at every winning pile.

```python
left = n - k
target = 1.0 if left == 0 else -Q[left, legal[left]].max()
Q[n, k - 1] += alpha * (target - Q[n, k - 1])
```

> [!KEY]
> In a two-player zero-sum game, self-play Q-learning uses one table for both sides, and the opponent's best value enters the target with a minus sign.

> [!RECAP]
> - Exploitability is $\max_i (Ay)_i - \min_j (x^\top A)_j$. It is zero only at a Nash equilibrium.
> - Regret matching plays each action in proportion to its positive cumulative regret, uniform when none is positive.
> - The column player's payoff vector is $-x^\top A$. The average strategies, not the last ones, converge.
> - Fictitious play best-responds to the opponent's empirical mixture.
> - Self-play Nim uses the negamax target: $+1$ for emptying the pile, otherwise minus the opponent's best value.

## Build it step by step

The checks test the functions in this order. `ROCK_PAPER_SCISSORS` and `nim_legal` are written for you.

1. **`regret_matching_strategy(regrets)`.** Clip at zero with `np.maximum`, and normalize, returning the uniform vector when the sum is zero. The common mistake is normalizing the raw regrets, which gives negative probabilities.
2. **`exploitability(A, x, y)`.** One line. The check expects 2 for rock against rock and 0 at both equilibria.
3. **`regret_matching(A, iters, x0, y0)`.** Play `x0` and `y0` in round 0. Add each round's strategies to running sums *before* updating regrets, and return the averages, the last strategies, and the regrets in a dict. The common mistake is $u^y = +x^\top A$, which the identity check catches at once.
4. **`fictitious_play(A, iters, first_x, first_y)`.** Keep action counts, compute both best responses from the counts *before* this round, then add both. The common mistake is letting the column player respond to the row player's updated counts.
5. **`nim_self_play(n_max, episodes, alpha, epsilon, seed)`.** Loop until the pile is empty, choosing among legal moves only. The common mistake is taking the max over illegal moves, which reads entries that are never updated.

## Where this breaks

**General-sum games.** When one player's gain is not the other's loss, regret matching only drives the empirical distribution of joint play toward a *coarse correlated equilibrium*, a weaker notion than Nash, and fictitious play can fail to converge at all.

**Non-transitive strategy spaces.** In RPS, "better" is not a line. Self-play against only the last copy can forget how to beat old strategies, which is why leagues keep old opponents around.

## Further reading

- [Hart and Mas-Colell (2000), A Simple Adaptive Procedure Leading to Correlated Equilibrium](https://doi.org/10.1111/1468-0262.00153). Introduces regret matching and proves its convergence properties.
- [Robinson (1951), An Iterative Method of Solving a Game](https://doi.org/10.2307/1969530). The proof that fictitious play converges in two-player zero-sum games.
- [Zinkevich, Johanson, Bowling, and Piccione (2007), Regret Minimization in Games with Incomplete Information](https://proceedings.neurips.cc/paper_files/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf). CFR, which runs regret matching in poker.
- [Lanctot and colleagues (2017), A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning](https://arxiv.org/abs/1711.00832). PSRO, which grows a population of policies.
- [Silver and colleagues (2017), Mastering Chess and Shogi by Self-Play](https://arxiv.org/abs/1712.01815). AlphaZero, which learns from self-play alone with a network and tree search.

## The exercise

You build the tools for two-player zero-sum games. Write `exploitability` and `regret_matching_strategy`, then `regret_matching` for both players at once, and `fictitious_play`. Finish with `nim_self_play`, one Q table that learns Nim by playing itself.

The checks confirm the hand computations: the strategy, an exploitability of 2 for rock against rock, and regret matching's first two rounds. After 10,000 rounds of RPS, the average play must be exploitable by under 0.04 and equal to total regret over $T$, while the last strategies must still be exploitable by more than 0.3. On the 2-by-2 game, both algorithms must land within 0.02 of the equilibrium. Nim self-play must take $n \bmod 4$ from every winning pile.
