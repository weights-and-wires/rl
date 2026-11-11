---
title: "AlphaZero: learning from self-play"
slug: 060-alphazero-self-play
part: "Part 8 — Rewards, evaluation, and many agents"
summary: Train a policy-value network on search results from self-play, with visit-count targets, temperature, root noise, and symmetry augmentation.
minutes: 50
gpu: false
objectives:
  - Turn root visit counts into a policy target with a temperature, and say why the target comes from visits and not from the network's prior.
  - Write the policy-value loss and check its gradient against finite differences.
  - Add Dirichlet noise at the root so that self-play keeps trying moves the network dislikes.
  - Use the eight symmetries of the board as data augmentation, moving the board and the policy together.
  - Train a small network by self-play on tic-tac-toe and measure it against random play and a perfect minimax player.
prereqs:
  - 022-monte-carlo-tree-search
  - 044-world-models-and-muzero
  - 059-multi-agent-and-self-play
exercise: 060-alphazero-self-play
---

# AlphaZero: learning from self-play

A chess engine spends its time on one question: of the moves it could play, which is best? Search answers it by looking ahead, which is slow, so the engine can look at only a few moves deeply. A network answers it by recognizing the position, which is fast but only as good as what it has seen. Each helps the other. If the network says which moves look promising, search spends its time there. If search finds a better move than the network liked, that result is a lesson the network can learn.

AlphaGo Zero and AlphaZero turned that loop into a training method. The agent plays against itself, and a network guides its search at every move. The search results become the training data for the next network, with no human games and no handwritten evaluation. The network gets better, so the search gets better, so the next games and their lessons get better.

This chapter builds that loop on tic-tac-toe, small enough to check against a perfect player. Four details make it work: the target the network learns from, a temperature that sets how greedy the games are, noise that keeps them varied, and symmetry that multiplies the data by eight.

## What you already know

- [Monte Carlo tree search](/lessons/022-monte-carlo-tree-search) grows a tree of simulations and counts how often each move is tried.
- [pUCT](/lessons/044-world-models-and-muzero) scores a move by its average value plus a bonus proportional to a prior and to $\sqrt{\sum N}/(1+N)$.
- [Self-play on Nim](/lessons/059-multi-agent-and-self-play) learns from games against a copy of yourself, with the opponent's value entering with a minus sign.
- A softmax turns logits into probabilities, and cross-entropy compares two distributions.

## The loop

Figure 1 shows one round. The network $f_\theta$ looks at a board and returns two things: a *prior* $p$, a probability for each move, and a *value* $v$, a number in $[-1, 1]$ for how the player to move is doing. Search uses both. It follows pUCT through the tree, and at a new leaf it asks the network for a prior and a value instead of playing a random game to the end.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="The AlphaZero loop: the network guides search, search plays games, games become training examples, and the examples train the network">
<defs><marker id="c050-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<rect x="20" y="30" width="150" height="70" rx="8" class="fill-soft-b stroke-b"/>
<text x="95" y="56" text-anchor="middle" class="fig-bold">network f</text>
<text x="95" y="76" text-anchor="middle" class="fig-small">prior p, value v</text>
<rect x="225" y="30" width="150" height="70" rx="8" class="fill-soft-a stroke-a"/>
<text x="300" y="56" text-anchor="middle" class="fig-bold">search</text>
<text x="300" y="76" text-anchor="middle" class="fig-small">50 simulations a move</text>
<rect x="430" y="30" width="150" height="70" rx="8" class="fill-soft-c stroke-c"/>
<text x="505" y="56" text-anchor="middle" class="fig-bold">self-play games</text>
<text x="505" y="76" text-anchor="middle" class="fig-small">12 games, 85 positions</text>
<rect x="430" y="150" width="150" height="70" rx="8" class="fill-soft-d stroke-d"/>
<text x="505" y="176" text-anchor="middle" class="fig-bold">replay buffer</text>
<text x="505" y="196" text-anchor="middle" class="fig-small">(board, pi, z) x 8 flips</text>
<rect x="225" y="150" width="150" height="70" rx="8" class="fill-soft-b stroke-b"/>
<text x="300" y="176" text-anchor="middle" class="fig-bold">train</text>
<text x="300" y="196" text-anchor="middle" class="fig-small">900 steps of az_loss</text>
<line x1="170" y1="65" x2="222" y2="65" class="stroke-ink" marker-end="url(#c050-arrow)"/>
<line x1="375" y1="65" x2="427" y2="65" class="stroke-ink" marker-end="url(#c050-arrow)"/>
<line x1="505" y1="100" x2="505" y2="147" class="stroke-ink" marker-end="url(#c050-arrow)"/>
<line x1="430" y1="185" x2="378" y2="185" class="stroke-ink" marker-end="url(#c050-arrow)"/>
<path d="M225 185 L95 185 L95 103" fill="none" class="stroke-ink" marker-end="url(#c050-arrow)"/>
<text x="300" y="20" text-anchor="middle" class="fig-small">guides</text>
<text x="400" y="240" text-anchor="middle" class="fig-small">the new network plays the next round of games</text>
</svg>
Figure 1. One round of the loop. Notice that the network appears twice: it guides the search that plays the games, and it is the thing the games train. A round of 12 games gives about 85 positions, and the 8 symmetries turn them into about 680 examples.
```

Each finished game gives one training example per position: the board, a policy target $\pi$ that the search produced at that position, and the final result $z$ from the point of view of the player to move there. The network trains to predict both. Then the improved network plays the next round of games.

> [!KEY]
> Search is a policy improvement step and the network is the memory. Search takes the network's prior and returns something stronger, the visit distribution. Training moves the network toward that stronger answer, much as policy iteration alternates evaluation and improvement in [policy iteration](/lessons/011-policy-iteration).

## The target is the visit distribution

After $n$ simulations from a board, the root holds a visit count $N(a)$ for every move $a$. Visits are not spread by the prior alone. pUCT sends a simulation down a move when the prior likes it or when the values found below it look good, so moves that search discovers to be strong collect visits the prior did not predict. The counts therefore hold what search learned beyond the network.

Why not train on the prior itself? The prior is what the network already says, so the loss would have nothing to correct and the network would never move. Why not train on the single most visited move? A one-hot target throws away the runner-up moves, and a position with two equally good moves would teach the network to favor one of them. The visit distribution keeps both.

The *temperature* $\tau$ shapes the counts into probabilities:

$$
\pi_\tau(a) = \frac{N(a)^{1/\tau}}{\sum_b N(b)^{1/\tau}}
$$

Raise each count to the power $1/\tau$, then divide by the total. At $\tau = 1$ this is the counts divided by their sum. As $\tau$ falls, the largest count dominates, and in the limit $\tau \to 0$ all the mass sits on the most visited move, shared equally among ties. At large $\tau$ it flattens toward uniform.

### A worked example

Suppose three moves have visits $N = (12, 6, 2)$ out of 20 simulations.

| $\tau$ | probabilities | what it does |
|---|---|---|
| 1 | $(0.600, 0.300, 0.100)$ | counts divided by 20 |
| 0.5 | $(0.783, 0.196, 0.022)$ | squares $(144, 36, 4)$ divided by 184 |
| 0.25 | $(0.941, 0.059, 0.001)$ | fourth powers $(20736, 1296, 16)$ divided by 22048 |
| $\to 0$ | $(1, 0, 0)$ | all mass on the most visited move |

Figure 2 plots the same effect on a real search, the 100 simulations a trained network runs from the empty board.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Visit counts of the opening search, and the policies they give at three temperatures">
<line x1="90" y1="70" x2="530" y2="70" class="stroke-muted"/>
<text x="82" y="66" text-anchor="end" class="fig-small fig-bold">visits N</text>
<rect x="103" y="58.8" width="14" height="11.2" class="fill-a"/>
<text x="110" y="84" text-anchor="middle" class="fig-small">17</text>
<rect x="151" y="68.7" width="14" height="1.3" class="fill-a"/>
<text x="158" y="84" text-anchor="middle" class="fig-small">2</text>
<rect x="199" y="58.8" width="14" height="11.2" class="fill-a"/>
<text x="206" y="84" text-anchor="middle" class="fig-small">17</text>
<rect x="247" y="68.7" width="14" height="1.3" class="fill-a"/>
<text x="254" y="84" text-anchor="middle" class="fig-small">2</text>
<rect x="295" y="62.7" width="14" height="7.3" class="fill-a"/>
<text x="302" y="84" text-anchor="middle" class="fig-small">11</text>
<rect x="343" y="68.7" width="14" height="1.3" class="fill-a"/>
<text x="350" y="84" text-anchor="middle" class="fig-small">2</text>
<rect x="391" y="59.5" width="14" height="10.5" class="fill-a"/>
<text x="398" y="84" text-anchor="middle" class="fig-small">16</text>
<rect x="439" y="68.7" width="14" height="1.3" class="fill-a"/>
<text x="446" y="84" text-anchor="middle" class="fig-small">2</text>
<rect x="487" y="49.6" width="14" height="20.4" class="fill-a"/>
<text x="494" y="84" text-anchor="middle" class="fig-small">31</text>
<line x1="90" y1="148" x2="530" y2="148" class="stroke-muted"/>
<text x="82" y="144" text-anchor="end" class="fig-small fig-bold">tau = 1</text>
<rect x="103" y="136.8" width="14" height="11.2" class="fill-b"/>
<text x="110" y="133.8" text-anchor="middle" class="fig-small">0.17</text>
<rect x="151" y="146.7" width="14" height="1.3" class="fill-b"/>
<rect x="199" y="136.8" width="14" height="11.2" class="fill-b"/>
<text x="206" y="133.8" text-anchor="middle" class="fig-small">0.17</text>
<rect x="247" y="146.7" width="14" height="1.3" class="fill-b"/>
<rect x="295" y="140.7" width="14" height="7.3" class="fill-b"/>
<rect x="343" y="146.7" width="14" height="1.3" class="fill-b"/>
<rect x="391" y="137.5" width="14" height="10.5" class="fill-b"/>
<text x="398" y="134.5" text-anchor="middle" class="fig-small">0.16</text>
<rect x="439" y="146.7" width="14" height="1.3" class="fill-b"/>
<rect x="487" y="127.6" width="14" height="20.4" class="fill-b"/>
<text x="494" y="124.6" text-anchor="middle" class="fig-small">0.31</text>
<line x1="90" y1="226" x2="530" y2="226" class="stroke-muted"/>
<text x="82" y="222" text-anchor="end" class="fig-small fig-bold">tau = 0.5</text>
<rect x="103" y="216.1" width="14" height="9.9" class="fill-b"/>
<text x="110" y="213.1" text-anchor="middle" class="fig-small">0.15</text>
<rect x="151" y="225.9" width="14" height="0.1" class="fill-b"/>
<rect x="199" y="216.1" width="14" height="9.9" class="fill-b"/>
<text x="206" y="213.1" text-anchor="middle" class="fig-small">0.15</text>
<rect x="247" y="225.9" width="14" height="0.1" class="fill-b"/>
<rect x="295" y="221.9" width="14" height="4.1" class="fill-b"/>
<rect x="343" y="225.9" width="14" height="0.1" class="fill-b"/>
<rect x="391" y="217.3" width="14" height="8.7" class="fill-b"/>
<text x="398" y="214.3" text-anchor="middle" class="fig-small">0.13</text>
<rect x="439" y="225.9" width="14" height="0.1" class="fill-b"/>
<rect x="487" y="193.2" width="14" height="32.8" class="fill-b"/>
<text x="494" y="190.2" text-anchor="middle" class="fig-small">0.50</text>
<line x1="90" y1="304" x2="530" y2="304" class="stroke-muted"/>
<text x="82" y="300" text-anchor="end" class="fig-small fig-bold">tau = 0.25</text>
<rect x="103" y="299.3" width="14" height="4.7" class="fill-b"/>
<rect x="151" y="304.0" width="14" height="0.0" class="fill-b"/>
<rect x="199" y="299.3" width="14" height="4.7" class="fill-b"/>
<rect x="247" y="304.0" width="14" height="0.0" class="fill-b"/>
<rect x="295" y="303.2" width="14" height="0.8" class="fill-b"/>
<rect x="343" y="304.0" width="14" height="0.0" class="fill-b"/>
<rect x="391" y="300.3" width="14" height="3.7" class="fill-b"/>
<rect x="439" y="304.0" width="14" height="0.0" class="fill-b"/>
<rect x="487" y="252.0" width="14" height="52.0" class="fill-b"/>
<text x="494" y="249.0" text-anchor="middle" class="fig-small">0.79</text>
<text x="110" y="318" class="fig-small">cell 0</text>
<text x="494" y="318" text-anchor="end" class="fig-small">cell 8</text>
<text x="300" y="18" text-anchor="middle" class="fig-small">bar height is the probability pi (top row: visits divided by the total); numbers under the top row are raw counts</text>
</svg>
Figure 2. Visit counts from the empty board, and the policy at three temperatures. Notice that the counts put 31 of 100 visits on cell 8, and that tau = 0.25 raises that move's share from 0.31 to 0.79.
```

Self-play uses the temperature to choose moves. Early moves use $\tau = 1$, so the same opening does not repeat in every game and the data covers many positions. Later moves use $\tau \to 0$, so the game is played well and its result $z$ is informative. The stored target is a separate choice. The papers write the move-selection distribution and the stored search probabilities with the same $\pi$, and a target at $\tau \to 0$ would be one-hot, with the problem above. This exercise always stores the $\tau = 1$ distribution, as many open implementations do.

```python
w = (N / N.max()) ** (1.0 / tau)   # divide by the max first: N ** 100 overflows
pi = w / w.sum()
```

At $\tau = 0$ the power is undefined, so handle it as its own case.

## Noise keeps the root honest

Search follows the prior. If the network gives a move a prior near zero, pUCT visits it rarely, the counts say it is bad, and the target tells the network it is bad. The mistake feeds itself. The fix is to add noise at the root, in self-play only. With $p$ the prior over the legal moves and $\eta$ a draw from a Dirichlet distribution with concentration $\alpha$:

$$
P(a) = (1 - \epsilon)\, p(a) + \epsilon\, \eta(a), \qquad \eta \sim \operatorname{Dir}(\alpha, \dots, \alpha)
$$

A Dirichlet draw is a random probability vector. A small $\alpha$ makes it lumpy, so each game boosts a different few moves. Take $p = (0.5, 0.3, 0.2)$, $\alpha = 0.3$, and $\epsilon = 0.25$. With seed 0 the draw is $\eta = (0.283, 0.000, 0.717)$, and the mixed prior is $(0.446, 0.225, 0.329)$. The third move jumped from 0.2 to 0.329 because this game's noise liked it.

AlphaGo Zero used $\epsilon = 0.25$ and $\alpha = 0.03$ on Go, and the AlphaZero paper scales $\alpha$ in inverse proportion to the number of legal moves, with 0.3 for chess, 0.15 for shogi, and 0.03 for Go. The noise applies only at the root and stays off when you measure strength.

```python
noise = np.zeros(9)
noise[legal] = rng.dirichlet(alpha * np.ones(legal.sum()))
mixed = (1 - eps) * prior + eps * noise
```

## The loss

The network has two heads, so the loss has two terms and a regularizer. For a batch of $B$ examples with results $z_i$, search targets $\pi_i$, and predictions $(p_i, v_i)$, where $p_i = \operatorname{softmax}(\text{logits}_i)$:

$$
\mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} \Big[ (z_i - v_i)^2 - \pi_i^\top \log p_i \Big] + c\, \lVert \theta \rVert^2
$$

The first term is a regression: the predicted value should match the game's result. The second is cross-entropy: the predicted move probabilities should match the search's. The last term pulls the weights toward zero with strength $c$.

Read the gradient off the logits. For one example, the cross-entropy has derivative $p - \pi$ in the logits, so the network is pushed until its probabilities equal the target's. Take logits $(2, 0, 0)$, a target $\pi = (1, 0, 0)$, a value $v = 0.5$, and a result $z = 1$. The softmax is $(0.787, 0.107, 0.107)$, so the cross-entropy is $-\ln 0.787 = 0.2395$. The value term is $(1 - 0.5)^2 = 0.25$. The loss is $0.4895$ before the L2 term, and the logit gradient is $(-0.213, 0.107, 0.107)$.

```python
value_loss = ((z - v) ** 2).mean()
policy_loss = -(pi * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
loss = value_loss + policy_loss + c * sum((p ** 2).sum() for p in params)
```

## Symmetry multiplies the data

A tic-tac-toe board has eight symmetries: four rotations, each with and without a mirror image. A position and its rotation are the same game. If the winning move is the top-right cell, then in the rotated board it is the cell the top-right moved to, and $z$ is unchanged. So one example gives eight.

The trap is to transform the board and forget the policy. Both must go through the same permutation of cells, so reshape them to $3 \times 3$ and apply the same `np.rot90` and `np.fliplr`. AlphaGo Zero augmented its Go data this way. Chess and shogi have no such symmetry, so AlphaZero does not augment.

## What the loop buys on tic-tac-toe

The exercise trains a network with one hidden layer of 64 units, using 50 simulations per move in self-play and 100 when playing. Figure 3 plots its strength after $k$ training iterations, each of 12 games and 900 gradient steps, averaged over three seeds. The opponents are a random player and a minimax player that plays perfectly and picks at random among its best moves.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Playing strength against training iterations: draws against minimax rise from the untrained level, and losses to a random player fall">
<line x1="80" y1="220.0" x2="560" y2="220.0" class="stroke-grid"/>
<text x="72" y="224.0" text-anchor="end" class="fig-small">0</text>
<line x1="80" y1="175.0" x2="560" y2="175.0" class="stroke-grid"/>
<text x="72" y="179.0" text-anchor="end" class="fig-small">5</text>
<line x1="80" y1="130.0" x2="560" y2="130.0" class="stroke-grid"/>
<text x="72" y="134.0" text-anchor="end" class="fig-small">10</text>
<line x1="80" y1="85.0" x2="560" y2="85.0" class="stroke-grid"/>
<text x="72" y="89.0" text-anchor="end" class="fig-small">15</text>
<line x1="80" y1="40.0" x2="560" y2="40.0" class="stroke-grid"/>
<text x="72" y="44.0" text-anchor="end" class="fig-small">20</text>
<text x="80.0" y="238" text-anchor="middle" class="fig-small">0</text>
<text x="200.0" y="238" text-anchor="middle" class="fig-small">5</text>
<text x="320.0" y="238" text-anchor="middle" class="fig-small">10</text>
<text x="440.0" y="238" text-anchor="middle" class="fig-small">15</text>
<text x="560.0" y="238" text-anchor="middle" class="fig-small">20</text>
<polyline points="80.0,109.0 104.0,112.0 128.0,76.0 176.0,76.0 272.0,58.0 368.0,46.0 560.0,40.0" class="stroke-a" fill="none"/>
<polyline points="80.0,196.0 104.0,193.0 128.0,199.0 176.0,202.0 272.0,208.0 368.0,217.0 560.0,220.0" class="stroke-b dashed" fill="none"/>
<circle cx="80.0" cy="109.0" r="4" class="fill-a"/>
<circle cx="80.0" cy="196.0" r="4" class="fill-b"/>
<circle cx="104.0" cy="112.0" r="4" class="fill-a"/>
<circle cx="104.0" cy="193.0" r="4" class="fill-b"/>
<circle cx="128.0" cy="76.0" r="4" class="fill-a"/>
<circle cx="128.0" cy="199.0" r="4" class="fill-b"/>
<circle cx="176.0" cy="76.0" r="4" class="fill-a"/>
<circle cx="176.0" cy="202.0" r="4" class="fill-b"/>
<circle cx="272.0" cy="58.0" r="4" class="fill-a"/>
<circle cx="272.0" cy="208.0" r="4" class="fill-b"/>
<circle cx="368.0" cy="46.0" r="4" class="fill-a"/>
<circle cx="368.0" cy="217.0" r="4" class="fill-b"/>
<circle cx="560.0" cy="40.0" r="4" class="fill-a"/>
<circle cx="560.0" cy="220.0" r="4" class="fill-b"/>
<text x="90" y="34.0" class="fig-small fig-bold">draws against minimax, out of 20 games (solid)</text>
<text x="240" y="130.0" class="fig-small fig-bold">losses to random, out of 100 games (dashed)</text>
<text x="320" y="258" text-anchor="middle" class="fig-small">training iterations, each 12 self-play games and 900 steps</text>
</svg>
Figure 3. Strength against training iterations, averaged over three seeds. Notice that the untrained network already draws 12.3 of 20 games against minimax because of search, and training lifts that to 20 of 20 while losses to random play fall to 0.
```

With $k = 0$ the network is untrained, yet it still draws 12.3 of 20 games against minimax, because 100 simulations see a few moves ahead on their own. Training lifts that to 20 of 20 by 20 iterations, and losses to random play fall from 2.7 per 100 to 0. Tic-tac-toe is easy for search. In Go the network must carry nearly all the knowledge, and the same loop builds it.

> [!RECAP]
> - The policy target is the visit distribution $\pi_\tau \propto N^{1/\tau}$. At $\tau = 1$ it is the counts over their sum, and at $\tau = 0$ it is one-hot with ties split.
> - Move selection uses a temperature, and the stored target stays at $\tau = 1$.
> - Root noise is $(1 - \epsilon)p + \epsilon\,\eta$ with $\eta$ Dirichlet over the legal moves only.
> - The loss is value MSE plus policy cross-entropy plus L2. The logit gradient is $(p - \pi)/B$.
> - The eight symmetries must move the board and the policy together. The result $z$ is labeled from the view of the player to move.

## Build it step by step

The game, the network, and the search are written for you. You write six functions, in the order the checks call them.

1. **`visit_count_policy(N, tau)`.** Handle $\tau = 0$ first, by marking the counts equal to the maximum. Otherwise divide by `N.max()`, raise to `1 / tau`, and normalize. The common mistake is to normalize raw counts and ignore `tau`, which passes at $\tau = 1$ and fails everywhere else.
2. **`add_root_noise(prior, legal, alpha, eps, rng)`.** Draw one Dirichlet sample over the `legal.sum()` legal moves, place it in a zero vector, and mix. The common mistake is to draw 9 values, which gives the illegal cells probability.
3. **`az_loss(logits, v, pi, z, params, c)`.** Use `torch.log_softmax` for the cross-entropy and take batch means of both terms. The checks compare your gradient with finite differences, so keep every step in torch. The common mistake is to sum over the batch, which makes the loss grow with the batch size.
4. **`dihedral_augment(board, pi)`.** Loop over four rotations and an optional mirror, applying each to both arrays. The first pair is the identity. The common mistake is to transform only the board.
5. **`self_play_game(evaluate, n_sim, c_puct, tau_moves, alpha, eps, rng, board)`.** At each move, run `search` with noise, store the board and `visit_count_policy(N, 1.0)`, then sample the move from the $\tau = 1$ policy for the first `tau_moves` moves and from $\tau = 0$ afterward. After the game ends, label each position with the result for the player to move there, which alternates in sign. The common mistake is to store the network's prior in place of the visits, which the target check catches.
6. **`train(...)`.** Each iteration plays games, adds all eight symmetries of every example to a buffer, keeps the newest 20,000, and calls `sgd_step` on random minibatches. The checks train once with the defaults and then play 200 games against a random player and 20 against minimax.

## Where this breaks

**Cost.** AlphaGo Zero ran 1,600 simulations per move and AlphaZero 800, across thousands of machines. Tic-tac-toe needs 50 and a few seconds.

**Self-play can chase its own tail.** A network that only plays its most recent self can forget how to beat old strategies, as in [rock-paper-scissors](/lessons/059-multi-agent-and-self-play). AlphaGo Zero kept the best network and promoted a challenger only after it won 55 percent of evaluation games. AlphaZero dropped that gate and kept one continuously updated network.

**The rules must be known.** Search needs to play moves forward, so the game must come with a simulator. [MuZero](/lessons/044-world-models-and-muzero) learns that model too. For language models, the same idea of using search to improve a policy and then distilling the result appears in [process rewards and search](/lessons/071-process-rewards-and-search).

## Further reading

- [Silver and colleagues (2017), Mastering the game of Go without human knowledge](https://doi.org/10.1038/nature24270). AlphaGo Zero: the policy-value network, visit-count targets, the loss, Dirichlet noise, and symmetry augmentation.
- [Silver and colleagues (2018), A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://doi.org/10.1126/science.aar6404). The Science paper on AlphaZero, with the same loop applied to three games. The [arXiv preprint (2017)](https://arxiv.org/abs/1712.01815) is Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm.
- [Silver, Lecture 10: Case Study: RL in Classic Games](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-10-case-study-rl-in-classic-games.pdf). The UCL reinforcement learning course lecture on game-playing programs.
- [CMU 10-703 Deep Reinforcement Learning, Fall 2024](https://cmudeeprl.github.io/703website_f24/lectures/). Lecture 16 is titled AlphaGo, AlphaGoZero, AlphaZero.
- [Berkeley CS285 Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/). The schedule lists Lecture 15, Model-Based RL Part 1, and Lecture 16, Model-Based RL Part 2.
- [Anthony, Tian, and Barber (2017), Thinking fast and slow with deep learning and tree search](https://arxiv.org/abs/1705.08439). Expert iteration, the same idea of search as the expert and the network as the apprentice.
- [Schrittwieser and colleagues (2020), Mastering Atari, Go, chess and shogi by planning with a learned model](https://arxiv.org/abs/1911.08265). MuZero, which drops the known rules.

## The exercise

You build the training loop of AlphaZero on tic-tac-toe. Write `visit_count_policy`, `add_root_noise`, `az_loss`, `dihedral_augment`, `self_play_game`, and `train`. The board, `search`, `net_evaluate`, `az_move`, and `sgd_step` are provided.

The checks verify each piece. `visit_count_policy` matches the table above and stays finite for large counts. `add_root_noise` equals the manual mix under a seed. `az_loss` matches hand values and finite differences. `dihedral_augment` keeps a winning move winning in all eight views. `self_play_game` labels results from the correct side and stores the visit distribution. The last two checks train the network and require at most 3 losses in 200 games against a random player, and at least 13 draws in 20 games against minimax.
