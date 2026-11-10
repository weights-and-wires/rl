---
title: Monte Carlo tree search
slug: 022-monte-carlo-tree-search
part: "Part 3 — Learning from experience"
summary: Plan at decision time by growing a search tree with UCB at every node, backing results up with the sign flipped for the opponent.
minutes: 35
gpu: false
objectives:
  - Explain why a planner that spends effort only on the current state can beat one that values every state.
  - Compute UCT scores by hand and say why an untried action scores infinity.
  - Run the four phases of one MCTS simulation, and say why the backup negates the value at each ply.
  - Implement UCT search on Nim, and check that most visits land on the winning move.
prereqs:
  - 004-optimism-and-ucb
  - 021-dyna-planning-with-a-learned-model
exercise: 022-monte-carlo-tree-search
---

# Monte Carlo tree search

A chess player does not memorize the value of every board. Before each move, the player looks ahead from the position on the table: if I go here, they go there, and so on. Most branches get a glance, and a few promising ones get a long look. After the move, the player throws the analysis away and starts from the next position.

The planners in [the previous chapter](/lessons/021-dyna-planning-with-a-learned-model) work differently. They compute a value for every state, whether or not the agent ever visits it. That works for a small maze, and it fails for a game as large as Go. This chapter builds a planner that spends all its effort on the state in front of it: Monte Carlo tree search, or MCTS.

You will use the simplest game that has the structure, a pile of stones.

## What you already know

- A UCB rule picks the arm with the highest mean plus a bonus that shrinks as the arm is tried, as in [Optimism in the face of uncertainty: UCB](/lessons/004-optimism-and-ucb).
- A model answers "what happens if I do this here?", as in [Planning with a learned model: Dyna](/lessons/021-dyna-planning-with-a-learned-model).
- A *rollout* is one run of an episode from a given state with a simple policy, such as uniformly random moves.
- In a two-player game with one winner, a good result for one player is a bad result for the other.
- `numpy` arrays can be indexed by a boolean mask, as in `scores[n == 0]`.

## The game: Nim

A pile holds some stones. On each turn a player takes 1, 2, or 3 stones. Whoever takes the last stone wins. A *state* is the number of stones left, and it is always the turn of "the player to move".

You can solve this game by hand. A pile that is a multiple of 4 loses for the player to move: whatever that player takes, the opponent takes enough to leave another multiple of 4, and the pile counts down to 0 on the opponent's last move. So the winning move from $k$ stones is to take $k \bmod 4$, which leaves a multiple of 4. From 10 stones, take 2.

A search that finds this without being told the rule is a fair test, because you know the right answer.

## The obvious approach: flat Monte Carlo

Start with the simplest idea. For each first move, play 1,000 games from the resulting position with random moves for both players, and average the results. Pick the move with the best average.

On 10 stones this gives win rates of 0.522, 0.556, and 0.455 for taking 1, 2, and 3. Take 2 is first, which is correct, but the lead over take 1 is only 0.034. On 11 stones, where the winning move is take 3, the rates are 0.493, 0.516, and 0.534, and the winner leads by 0.018. Two things go wrong here:

- The estimate treats the opponent as a fool who plays randomly. A move that wins against perfect replies and a move that wins against random replies get nearly the same score, because a random opponent stumbles into a loss often enough.
- Every first move gets the same number of games, even when one is clearly worse after 100 games.

The fix for the second problem is one you already have. Choosing which move to spend the next game on is a bandit problem. The fix for the first is to make the replies matter, which means searching deeper than one move.

## Grow a tree of what you try

MCTS grows a tree of states reachable from the current one. Each *simulation* runs four phases:

1. **Selection.** From the root, walk down the tree. At each node, treat the legal moves as bandit arms and pick one with a UCB rule.
2. **Expansion.** When you reach a state that is not in the tree yet, add it.
3. **Simulation.** From the added state, play random moves to the end of the game. This rollout gives a rough value.
4. **Backup.** Walk back up the path and add the result to the count and the total of every move along it.

After the budget runs out, play the root move with the most visits. Visit counts pile up on the promising moves because UCB keeps choosing them, so the tree is deep where the game is sharp and shallow where it is not.

```figure
<svg viewBox="0 0 640 350" role="img" aria-label="A Nim search tree from 5 stones with the four phases of one simulation: selection, expansion, simulation, backup">
<defs><marker id="c022-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" class="fill-muted"/></marker></defs>
<text x="20.0" y="20.0" class="fig-small" text-anchor="start">Nim, 5 stones at the root. Nodes show stones left.</text>
<line x1="300.0" y1="77.0" x2="170.0" y2="131.0" class="stroke-grid"/>
<text x="201.0" y="108.0" class="fig-small" text-anchor="middle">take 1</text>
<line x1="300.0" y1="77.0" x2="300.0" y2="131.0" class="stroke-b" marker-end="url(#c022-arrow)"/>
<text x="330.0" y="108.0" class="fig-small" text-anchor="middle">take 2</text>
<line x1="300.0" y1="77.0" x2="430.0" y2="131.0" class="stroke-grid"/>
<text x="399.0" y="108.0" class="fig-small" text-anchor="middle">take 3</text>
<line x1="300.0" y1="167.0" x2="300.0" y2="221.0" class="stroke-b" marker-end="url(#c022-arrow)"/>
<text x="330.0" y="198.0" class="fig-small" text-anchor="middle">take 1</text>
<rect x="274.0" y="43.0" width="52.0" height="34.0" class="fill-soft-a stroke-a" rx="17"/>
<text x="300.0" y="65.0" class="fig-bold" text-anchor="middle">5</text>
<rect x="144.0" y="133.0" width="52.0" height="34.0" class="fill-sunk stroke-muted" rx="17"/>
<text x="170.0" y="155.0" class="fig-bold" text-anchor="middle">4</text>
<rect x="274.0" y="133.0" width="52.0" height="34.0" class="fill-soft-a stroke-a" rx="17"/>
<text x="300.0" y="155.0" class="fig-bold" text-anchor="middle">3</text>
<rect x="404.0" y="133.0" width="52.0" height="34.0" class="fill-sunk stroke-muted" rx="17"/>
<text x="430.0" y="155.0" class="fig-bold" text-anchor="middle">2</text>
<rect x="274.0" y="223.0" width="52.0" height="34.0" class="fill-soft-b stroke-b" rx="17"/>
<text x="300.0" y="245.0" class="fig-bold" text-anchor="middle">2</text>
<polyline points="300.0,258.0 318.0,276.0 296.0,292.0 316.0,308.0" class="stroke-muted dashed" fill="none"/>
<circle cx="316" cy="308" r="4" class="fill-muted"/>
<text x="460.0" y="60.0" class="fig-small" text-anchor="start">1 Selection: follow UCT</text>
<text x="460.0" y="76.0" class="fig-small" text-anchor="start">scores down the tree</text>
<text x="460.0" y="246.0" class="fig-small" text-anchor="start">2 Expansion: add the</text>
<text x="460.0" y="262.0" class="fig-small" text-anchor="start">state to the tree</text>
<text x="460.0" y="300.0" class="fig-small" text-anchor="start">3 Simulation: random moves</text>
<text x="460.0" y="316.0" class="fig-small" text-anchor="start">to the end of the game</text>
<text x="20.0" y="338.0" class="fig-small" text-anchor="start">4 Backup: send the result up the same path, flipping its sign at each ply</text>
<path d="M 272 248 C 200 215 200 110 274 66" class="stroke-c" fill="none" marker-end="url(#c022-arrow)"/>
</svg>
Figure 1. One simulation from 5 stones. The path follows take 2 to 3 stones, then take 1 adds a node holding 2 stones. A random rollout plays from there, and the result travels back up the same path. Notice that only the gray nodes are untouched: the search spends effort where it already looks promising.
```

## Pick moves with UCT

*UCT* (Kocsis and Szepesvári, 2006) is MCTS with the UCB1 rule at every node:

$$
a = \operatorname*{arg\,max}_a \left[ Q(s, a) + c \sqrt{\frac{\ln N(s)}{N(s, a)}} \right]
$$

Read it aloud: $Q(s, a)$ is the mean result of the simulations that took move $a$ in state $s$, and $N(s, a)$ is how many there were. $N(s)$ is the node's total visits. The square-root term is the bonus. It grows slowly as the node is visited, because $\ln N(s)$ rises, and it shrinks as $a$ is tried, because $N(s, a)$ sits in the denominator. An untried move has $N(s, a) = 0$, so its score is infinite and every move is tried once before any is repeated. The constant $c$ trades exploring against exploiting.

In numpy:

```python
scores = q + c * np.sqrt(np.log(n_parent) / n)
scores = np.where(n == 0, np.inf, scores)
```

Here is a worked example. A node has $N(s) = 10$. Move $a$ has $N = 6$ and $Q = 0.5$. Move $b$ has $N = 4$ and $Q = 0.4$. With $c = \sqrt 2$ and $\ln 10 = 2.3026$:

- $a$ scores $0.5 + 1.4142 \sqrt{2.3026 / 6} = 0.5 + 0.8761 = 1.3761$.
- $b$ scores $0.4 + 1.4142 \sqrt{2.3026 / 4} = 0.4 + 1.0730 = 1.4730$.

UCT picks $b$. Its lower mean is outweighed by how little it has been tried. Kocsis and Szepesvári showed that, with a suitable $c$, the probability of choosing a suboptimal root move goes to zero as the simulations grow.

## Flip the sign for the other player

The backup holds the one subtle step. A rollout ends with a result for the player to move in the leaf. But the leaf's parent is a decision by the *other* player, and a good leaf for one player is a bad leaf for the other. If you add the raw result to the parent's totals, the parent's player learns to prefer moves that help the opponent.

The fix is to store each node's values from the point of view of the player to move there, and to negate the value at each step up the tree:

```python
for state, i in reversed(path):
    value = -value
    N[state][i] += 1
    W[state][i] += value
```

Work through one simulation from 6 stones. Player A moves at the root, takes 2, and leaves 4. Player B takes 1 and leaves 3, a state not yet in the tree. Suppose the rollout from 3 stones, with A to move, ends with A winning, so its value is $+1$ for A.

- Negate once: $-1$. This is stored on B's move, take 1 from 4 stones. It is bad for B, which is correct, because B took 1 and let A win.
- Negate again: $+1$. This is stored on A's move, take 2 from 6 stones. It is good for A, which is correct.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="The value from a rollout at 3 stones flips sign at each step up: it is +1 at the leaf, -1 on the opponent's move, and +1 on the root move">
<defs><marker id="c022-arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" class="fill-muted"/></marker></defs>
<text x="20.0" y="22.0" class="fig-small" text-anchor="start">One simulation from 6 stones. Player A moves at the root.</text>
<rect x="35.0" y="50.0" width="130.0" height="50.0" class="fill-soft-a stroke-a" rx="6"/>
<text x="100.0" y="72.0" class="fig-bold" text-anchor="middle">6 stones</text>
<text x="100.0" y="90.0" class="fig-small" text-anchor="middle">A to move</text>
<rect x="235.0" y="50.0" width="130.0" height="50.0" class="fill-soft-a stroke-a" rx="6"/>
<text x="300.0" y="72.0" class="fig-bold" text-anchor="middle">4 stones</text>
<text x="300.0" y="90.0" class="fig-small" text-anchor="middle">B to move</text>
<rect x="435.0" y="50.0" width="130.0" height="50.0" class="fill-soft-a stroke-a" rx="6"/>
<text x="500.0" y="72.0" class="fig-bold" text-anchor="middle">3 stones</text>
<text x="500.0" y="90.0" class="fig-small" text-anchor="middle">A to move</text>
<line x1="165.0" y1="75.0" x2="235.0" y2="75.0" class="stroke-muted" marker-end="url(#c022-arrow2)"/>
<text x="200.0" y="66.0" class="fig-small" text-anchor="middle">A takes 2</text>
<line x1="365.0" y1="75.0" x2="435.0" y2="75.0" class="stroke-muted" marker-end="url(#c022-arrow2)"/>
<text x="400.0" y="66.0" class="fig-small" text-anchor="middle">B takes 1</text>
<text x="500.0" y="124.0" class="fig-bold" text-anchor="middle">rollout says +1</text>
<text x="500.0" y="142.0" class="fig-small" text-anchor="middle">(A wins from here)</text>
<text x="300.0" y="190.0" class="fig-bold" text-anchor="middle">Backup, walking up the path:</text>
<text x="300.0" y="222.0" class="fig-small" text-anchor="middle">negate once: -1, stored on B's move (take 1). Bad for B.</text>
<text x="300.0" y="250.0" class="fig-small" text-anchor="middle">negate again: +1, stored on A's move (take 2). Good for A.</text>
<line x1="500.0" y1="150.0" x2="420.0" y2="205.0" class="stroke-c" marker-end="url(#c022-arrow2)"/>
<line x1="200.0" y1="232.0" x2="120.0" y2="104.0" class="stroke-c" marker-end="url(#c022-arrow2)"/>
</svg>
Figure 2. The backup for one simulation from 6 stones. The leaf's value is +1 for A. It becomes -1 on B's move and +1 on A's move, so each move is scored from the view of the player who made it. Notice that one result updates two moves with opposite signs.
```

> [!KEY]
> MCTS spends its effort on the state in front of it. UCT chooses which move to simulate next, and the backup flips the value's sign at every ply, so each move is scored from the viewpoint of the player who made it.

## Watch the search find the move

With 1,000 simulations and $c = 1.4$, UCT puts most of its root visits on the winning move for every pile from 5 to 15 that is not a multiple of 4. For example, from 5, 6, 7, 9, 10, and 11 stones, the winning move gets 991, 985, 986, 990, 980, and 978 of about 1,000 visits. The search also sharpens as the budget grows. From 10 stones, the share of visits on take 2 rises from 0.69 at 50 simulations to 0.99 at 2,000.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Stacked bars of visit counts for taking 1, 2, or 3 stones from 10, after 50, 200, 1000, and 2000 simulations; take 2 dominates">
<text x="20.0" y="22.0" class="fig-bold" text-anchor="start">Share of root visits on each move, 10 stones (take 2 wins)</text>
<text x="20.0" y="60.0" class="fig-small" text-anchor="start">50 sims</text>
<rect x="100.0" y="40.0" width="122.0" height="30.0" class="fill-muted"/>
<text x="161.0" y="60.0" class="fig-small fill-paper" text-anchor="middle">13</text>
<rect x="222.0" y="40.0" width="319.2" height="30.0" class="fill-b"/>
<text x="381.6" y="60.0" class="fig-small fill-paper" text-anchor="middle">34</text>
<rect x="541.2" y="40.0" width="18.8" height="30.0" class="fill-a"/>
<text x="20.0" y="108.0" class="fig-small" text-anchor="start">200 sims</text>
<rect x="100.0" y="88.0" width="32.4" height="30.0" class="fill-muted"/>
<text x="116.2" y="108.0" class="fig-small fill-paper" text-anchor="middle">14</text>
<rect x="132.4" y="88.0" width="420.7" height="30.0" class="fill-b"/>
<text x="342.7" y="108.0" class="fig-small fill-paper" text-anchor="middle">182</text>
<rect x="553.1" y="88.0" width="6.9" height="30.0" class="fill-a"/>
<text x="20.0" y="156.0" class="fig-small" text-anchor="start">1000 sims</text>
<rect x="100.0" y="136.0" width="6.9" height="30.0" class="fill-muted"/>
<rect x="106.9" y="136.0" width="451.3" height="30.0" class="fill-b"/>
<text x="332.5" y="156.0" class="fig-small fill-paper" text-anchor="middle">980</text>
<rect x="558.2" y="136.0" width="1.8" height="30.0" class="fill-a"/>
<text x="20.0" y="204.0" class="fig-small" text-anchor="start">2000 sims</text>
<rect x="100.0" y="184.0" width="3.7" height="30.0" class="fill-muted"/>
<rect x="103.7" y="184.0" width="455.4" height="30.0" class="fill-b"/>
<text x="331.4" y="204.0" class="fig-small fill-paper" text-anchor="middle">1979</text>
<rect x="559.1" y="184.0" width="0.9" height="30.0" class="fill-a"/>
<rect x="100.0" y="242.0" width="14.0" height="14.0" class="fill-muted"/>
<text x="120.0" y="254.0" class="fig-small" text-anchor="start">take 1</text>
<rect x="200.0" y="242.0" width="14.0" height="14.0" class="fill-b"/>
<text x="220.0" y="254.0" class="fig-small" text-anchor="start">take 2</text>
<rect x="300.0" y="242.0" width="14.0" height="14.0" class="fill-a"/>
<text x="320.0" y="254.0" class="fig-small" text-anchor="start">take 3</text>
</svg>
Figure 3. Root visits for taking 1, 2, or 3 stones from a pile of 10, after 50 to 2,000 simulations. Notice that the winning move already leads at 50 simulations and then takes almost every visit, which is what the UCB bonus is for: the other moves still get a few visits, so a surprise would be caught.
```

> [!TIP]
> Test a two-player search on a game whose answer you can compute by hand, such as Nim. A missing sign flip makes the search prefer the opponent's best moves, and on Nim the visits spread out almost evenly.

> [!RECAP]
> - UCT adds $c\sqrt{\ln N(s) / N(s, a)}$ to each mean and scores untried moves as $+\infty$.
> - One simulation runs selection, expansion, a random rollout, and a backup.
> - The backup negates the value at each ply, because the parent's player is the opponent.
> - After the budget is spent, play the root move with the most visits.

## Build it step by step

The checks test the functions in this order. `Nim` is already written.

1. **`uct_scores(q, n, n_parent, c)`.** Returns an array of shape `(A,)`. Silence the division warning with `np.errstate`, then replace untried moves with `np.inf`. Common mistake: forgetting that `n_parent` is the node's total visits, not the number of moves.
2. **`mcts(game, root, n_simulations, c, seed)`.** Returns the root's visit counts. Keep a dict from state to two arrays, `N` and `W`, one entry per move in `game.actions(state)`. The mean is `W / N` where `N > 0`. Break ties at random with the generator. Common mistake: forgetting to negate during the backup.

## Cost and scaling

The following table compares the work of MCTS with the table-based planners from the previous chapter:

| Method | Work per decision | Memory |
|---|---|---|
| Dyna-Q | $n$ updates per real step | One entry per state-action pair seen |
| MCTS with UCT | Many simulations, each as long as the game | One node per expanded state |

MCTS never builds a table of the whole world. Its cost depends on the search budget, not on the number of states, which is why the idea scales to Go. It needs a model it can query at every decision, and a rollout policy that is cheap to run.

## Where this breaks

**Weak rollouts.** A uniformly random rollout is a poor estimate in games where random play is far from good play. AlphaGo mixed rollouts with a learned value function, and AlphaZero dropped rollouts and used a learned value function and policy to guide the search. [World models and MuZero](/lessons/044-world-models-and-muzero) builds that version.

**Tuning $c$.** UCT's constant depends on the scale of the returns. The textbook $c = \sqrt 2$ assumes results in $[0, 1]$. With other scales, tune $c$ or rescale the values.

**Two-player only as written.** The sign flip assumes strict alternation and a zero-sum result. A game with more players or shared rewards needs a different backup.

## Further reading

- [Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd edition (2018), Chapter 8](http://incompleteideas.net/book/the-book-2nd.html). Decision-time planning and MCTS.
- [Kocsis and Szepesvári, "Bandit based Monte-Carlo planning" (2006)](https://doi.org/10.1007/11871842_29). UCT and its consistency result.
- [Browne and colleagues, "A survey of Monte Carlo tree search methods" (2012)](https://doi.org/10.1109/TCIAIG.2012.2186810). The many variants of MCTS, and where each one works.

## The exercise

You build a tree search for Nim. Write `uct_scores`, then `mcts`.

The checks verify that `uct_scores` matches the worked example and picks $b$, scores an untried move as infinity, and gives a larger bonus to a parent with more visits. They verify that `mcts` takes everything when 2 or 3 stones are left, and that it puts most of its visits on the winning move for six piles, the same way on every run with the same seed.
