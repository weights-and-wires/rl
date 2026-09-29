---
title: Planning, learning, and search
slug: 10-planning-and-learning
part: "Part 2 — Learning from experience"
summary: Reuse real experience through a learned model with Dyna-Q, order the updates with prioritized sweeping, and search ahead with MCTS.
minutes: 60
gpu: false
objectives:
  - Implement Dyna-Q, and measure how planning steps cut the real experience needed to find the goal.
  - Implement prioritized sweeping on a known model, and explain why it needs far fewer updates than random replay.
  - Compute UCT scores and run Monte Carlo tree search on a two-player game, flipping the value's sign at each ply.
prereqs:
  - 02-multi-armed-bandits
  - 05-dynamic-programming
  - 07-temporal-difference-learning
exercise: 10-planning-and-learning
---

# Planning, learning, and search

Q-learning uses each real step once and then throws it away. That's wasteful
when real steps are expensive, such as a robot's arm movements or a trade in
a live market. Every step tells you something lasting about the world: taking
this action here led there and paid this much. If you remember that fact, you
can learn from it again and again without taking another real step.

That's the idea behind *planning* in reinforcement learning. Planning means
improving a policy by computing with a model of the world, instead of by
acting in it. Dynamic programming was planning with a perfect model handed to
you. This lesson builds the model from experience, and then plans with it in
three ways.

Dyna-Q replays remembered transitions at random between real steps.
Prioritized sweeping replays them in the order that changes the values most.
Monte Carlo tree search plans only at the moment of a decision, looking ahead
from the current state as a chess player does before each move.

## What you already know

- The Q-learning update moves $Q(s, a)$ toward $r + \gamma \max_{a'} Q(s',
  a')$, as in [Temporal-difference learning](07-temporal-difference-learning).
- Value iteration applies that same backup to every state with a known model,
  and converges to $Q^*$, as in [Dynamic programming](05-dynamic-programming).
- UCB1 from [Multi-armed bandits](02-multi-armed-bandits) picks the arm with
  the highest mean plus a bonus that shrinks as the arm is pulled more.
- A *deterministic* world always gives the same next state and reward for the
  same state and action.
- `rlenvs.gridworld` builds a maze from rows of characters.

## Learn a model and plan with it

A *model* here is a table. For each state-action pair you've tried, it
remembers what happened: the reward $r$, the next state $s'$, and whether the
episode ended. In a deterministic world, one entry per pair is the whole
truth. In Python, a dict does the job:

```python
model[(s, a)] = (r, s2, terminated)
```

Dyna-Q (Sutton, 1990) interleaves three things after every real step:

1. **Learn directly.** Apply the ordinary Q-learning update to the real
   transition.
2. **Learn the model.** Store the transition in the table.
3. **Plan.** Repeat $n$ times: pick a pair you've seen, uniformly at random,
   read its outcome from the model, and apply the same Q-learning update as
   if it had happened again.

With $n = 0$, Dyna-Q is Q-learning. Each planning update is the Q-learning
update you know, applied to a remembered transition:

$$
Q(\tilde s, \tilde a) \leftarrow Q(\tilde s, \tilde a) + \alpha \Bigl[\tilde r + \gamma \max_{a'} Q(\tilde s', a') - Q(\tilde s, \tilde a)\Bigr]
$$

Read it as: $(\tilde s, \tilde a)$ is a pair drawn from the model, and
$(\tilde r, \tilde s')$ is what the model says it does. The update pulls the
remembered pair's value toward the reward plus the best value of where it
leads, with no further experience. In numpy the loop is short:

```python
pairs = list(model)
for _ in range(n_steps):
    s, a = pairs[int(rng.integers(len(pairs)))]
    r, s2, terminated = model[(s, a)]
    Q[s, a] += alpha * (r + gamma * Q[s2].max() * (not terminated) - Q[s, a])
```

Here's a worked example. A chain $s_0 \to s_1 \to s_2$ has one action,
*right*, and reaching the terminal $s_2$ pays 1. Use $\gamma = 0.9$, $\alpha =
1$, and $Q = 0$ at the start:

- The first real episode updates $Q(s_0) = 0 + 0.9 \times 0 = 0$, because
  $Q(s_1)$ is still 0. Then it updates $Q(s_1) = 1$.
- One planning update on $(s_0, \text{right})$ gives $Q(s_0) = 0 + 0.9 \times
  1 = 0.9$, with no extra real experience.
- Without planning, $Q(s_0)$ reaches 0.9 only during the second real episode.

### Measure the payoff

Sutton and Barto's Dyna maze is a 6 by 9 grid with a few walls. Reaching the
goal in the top-right corner pays 1, every other step pays 0, and $\gamma =
0.95$. The shortest path takes 14 steps, so the start's optimal value is
$0.95^{13} = 0.513$. The following table shows the mean number of real steps
per episode, over 10 seeded runs with $\alpha = 0.1$ and $\varepsilon = 0.1$:

| Planning steps $n$ | Episode 1 | Episode 2 | Episode 3 | Episode 10 | Episodes 2 to 10, total |
|---|---|---|---|---|---|
| 0 | 1049 | 752 | 684 | 154 | 3159 |
| 5 | 1164 | 200 | 84 | 17 | 441 |
| 50 | 677 | 39 | 18 | 17 | 173 |

The first episode is a long random walk for everyone, because $Q$ is zero
until the goal is found. After that, planning spreads the goal's value back
through the whole remembered maze. With $n = 50$, the agent walks a
near-shortest path from episode 3 on. Without planning, it's still wandering
at episode 10.

> [!KEY]
> Dyna-Q replays remembered transitions as extra Q-learning updates. Each real
> step buys $n$ imagined ones, so the agent needs far less real experience.

> [!WARNING]
> Tabular Dyna-Q trusts its model completely. If the world changes, such as a
> shortcut opening in the maze, the stored transitions are wrong and planning
> spreads the error. Dyna-Q+ adds a bonus $\kappa \sqrt{\tau}$ to a modelled
> reward, where $\tau$ is the number of steps since the pair was last tried,
> so the agent goes back to check stale entries.

## Update where the news is

Random replay wastes most of its updates. Early in the maze, only the pairs
next to the goal have anything to learn. Updating a pair whose successor
still has value 0 changes nothing. Prioritized sweeping (Moore and Atkeson,
1993) keeps a priority queue of pairs, ordered by how much their value would
change.

For a deterministic model with outcome $(r, s')$, the priority of a pair is
the size of the update it would make:

$$
p(s, a) = \Bigl| r + \gamma \max_{a'} Q(s', a') - Q(s, a) \Bigr|
$$

Read it as: how far the pair's value is from its one-step target. A pair
enters the queue only if $p > \theta$, a small threshold. After you update a
pair $(s, a)$, the value of $s$ may have changed, so every *predecessor*
$(\bar s, \bar a)$, a pair whose model leads to $s$, gets its priority
recomputed and joins the queue if it clears $\theta$. News flows backward from
where it arrived. In the chain, after $Q(s_1)$ changes to 1, the predecessor
$(s_0, \text{right})$ gets priority $|0 + 0.9 \times 1 - 0| = 0.9$.

Python's `heapq` is a min-heap, so push negated priorities:

```python
heapq.heappush(heap, (-p, s, a))
neg_p, s, a = heapq.heappop(heap)       # the highest priority comes out first
```

The exercise starts prioritized sweeping from $Q = 0$ with the maze's full
model and $\alpha = 1$. When the queue empties, every pair's priority is at
most $\theta$. The backup is a $\gamma$-contraction, so the error is then at
most $\theta / (1 - \gamma)$. On the maze, prioritized sweeping reaches $Q^*$
in 184 updates, exactly one per non-terminal pair. The priorities fall off
as powers of $\gamma$ with distance from the goal, so the queue works outward
from the goal, and here each pair's first update already uses its
successor's final value. Uniform random planning needs about 3,000 updates to get within
$10^{-3}$.

> [!KEY]
> Prioritized sweeping updates the pairs whose values would change most, and
> then their predecessors. Changes propagate backward from the goal instead of
> being found by chance.

## Search at decision time with MCTS

Dyna and prioritized sweeping compute values for every state, whether or not
you'll ever visit it. *Decision-time planning* spends its effort on the state
you're in. Before each move, it looks ahead from that state, picks the best
first action, and throws most of the search away.

Monte Carlo tree search (MCTS) grows a tree of states reachable from the
current one. Each *simulation* runs four phases:

1. **Selection.** From the root, walk down the tree. At each node, treat the
   actions as bandit arms and pick one with a UCB rule.
2. **Expansion.** When you reach a state that isn't in the tree yet, add it.
3. **Simulation.** From the added state, play random moves to the end of the
   game. This *rollout* gives a rough value.
4. **Backup.** Walk back up the path, and add the result to the count and
   total of every action along it.

Visit counts pile up on the promising moves, because UCB keeps choosing them.
After the budget runs out, play the root action with the most visits.

### Pick actions with UCT

UCT (Kocsis and Szepesvári, 2006) is MCTS with the UCB1 rule at every node:

$$
a = \operatorname*{arg\,max}_a \left[ Q(s, a) + c \sqrt{\frac{\ln N(s)}{N(s, a)}} \right]
$$

Read it as: $Q(s, a)$ is the mean result of the simulations that took $a$ in
$s$, $N(s, a)$ is how many there were, and $N(s)$ is the node's total. The
bonus grows slowly with the node's visits and shrinks as $a$ is tried. An
action with $N(s, a) = 0$ has an infinite score, so every action is tried
once first. The constant $c$ trades exploring against exploiting. Kocsis and
Szepesvári showed that, with a suitable $c$, the probability of choosing a
suboptimal root action goes to zero as the simulations grow.

```python
scores = q + c * np.sqrt(np.log(n_parent) / n)
scores = np.where(n == 0, np.inf, scores)
```

Here's a worked example. A root has $N = 10$. Action $a$ has $N = 6$ and $Q =
0.5$. Action $b$ has $N = 4$ and $Q = 0.4$. With $c = \sqrt 2$ and $\ln 10 =
2.3026$:

- $a$ scores $0.5 + 1.4142 \sqrt{2.3026 / 6} = 0.5 + 0.8761 = 1.3761$.
- $b$ scores $0.4 + 1.4142 \sqrt{2.3026 / 4} = 0.4 + 1.0730 = 1.4730$.

UCT picks $b$. Its lower mean is outweighed by how little it has been tried.

### Flip the sign for the other player

In a two-player game, a good result for one player is a bad one for the
other. Store each node's values from the point of view of the player to move
there. Then, during the backup, negate the value at each step up the tree,
because the parent's player is the opponent:

```python
for state, i in reversed(path):
    value = -value
    N[state][i] += 1
    W[state][i] += value
```

The exercise's game is *Nim*. A pile holds some stones, and each turn a
player takes 1, 2, or 3. Whoever takes the last stone wins. A pile that's a
multiple of 4 loses for the player to move: whatever you take, your opponent
takes the rest of 4. So the winning move from $k$ stones is to take $k \bmod
4$. With 1,000 simulations and $c = 1.4$, UCT puts most of its root visits
on that move for every pile from 5 to 15 that isn't a multiple of 4. From 10
stones with 2,000 simulations, the visits for taking 1, 2, and 3 are 16,
1,979, and 4.

> [!TIP]
> Test a two-player search on a game whose answer you can compute by hand,
> such as Nim. A missing sign flip makes it prefer the opponent's best moves,
> and on Nim the visits spread out almost evenly.

> [!RECAP]
> - A model is a dict from `(s, a)` to `(r, s2, terminated)`. Plan by drawing
>   keys uniformly and applying the Q-learning update.
> - Dyna-Q calls the planning loop $n$ times after every real step.
> - Prioritized sweeping pops the largest $|r + \gamma \max Q(s') - Q(s,
>   a)|$, updates it, and queues predecessors above $\theta$.
> - UCT adds $c\sqrt{\ln N(s) / N(s, a)}$ to each mean and scores untried
>   actions as $+\infty$.
> - MCTS backs up rollout results along the path, negating at each ply.

## Build it step by step

The checks test the functions in this order. `dyna_maze` and `Nim` are
already written.

1. **`q_update(Q, s, a, r, s2, terminated, alpha, gamma)`.** The Q-learning
   update from the previous lessons, in place.
2. **`plan(Q, model, n_steps, alpha, gamma, rng)`.** Returns `Q`. Turn the
   dict's keys into a list once, then draw indices with `rng.integers`.
   Common mistake: planning from pairs you never tried, which reads outcomes
   that don't exist.
3. **`dyna_q(mdp, n_episodes, n_planning, alpha, epsilon, seed)`.** Returns
   `(Q, steps)`. After each real step, call `q_update`, store the model entry,
   then call `plan`. Use one generator for acting, stepping, and planning.
4. **`prioritized_sweeping(model, n_states, n_actions, gamma, theta)`.**
   Returns `(Q, updates)`. Build a predecessor map from `s2` to the set of
   pairs that lead to it, queue every pair above `theta`, and loop. Common
   mistake: leaving an old, lower copy of a pair in the heap and updating it
   twice. Keep a dict of each queued pair's current priority, and skip stale
   entries when they surface.
5. **`uct_scores(q, n, n_parent, c)`.** Returns `(A,)`. Silence the division
   warning with `np.errstate`, then replace untried actions with `np.inf`.
6. **`mcts(game, root, n_simulations, c, seed)`.** Returns the root's visit
   counts. Keep a dict from state to two arrays, `N` and `W`, one entry per
   action in `game.actions(state)`. The mean is `W / N` where `N > 0`. Common
   mistake: forgetting to negate during the backup.

## Cost and scaling

The following table compares the methods in this lesson:

| Method | Work per real step | Memory | Scales with |
|---|---|---|---|
| Dyna-Q | $1 + n$ updates of $O(A)$ each | Q table plus one model entry per pair | The planning budget $n$ |
| Prioritized sweeping | $O(\log M)$ per queue operation | Queue, model, and predecessor map | The number of predecessors per state |
| MCTS with UCT | Many simulations per decision | One node per expanded state | Simulations times game length |

Here $M$ is the queue's size. Dyna-Q and prioritized sweeping need memory for
every pair they've seen, so like all tabular methods they fail on huge state
spaces. MCTS never builds a table of the whole world. Its cost depends on the
search budget, not the number of states, which is why it works for games like
Go. But it needs a model it can query at every decision, and a rollout policy
that's cheap to run.

## Where this breaks

**Stochastic worlds.** The model here stores only the last outcome of each
pair. When outcomes are random, that single sample is treated as certain.
Store counts of outcomes and sample from them, or learn a model of the
distribution.

**Wrong models.** A model that's wrong somewhere misleads every planning step
that touches it. Dyna-Q+ re-explores stale pairs. Model-based deep methods,
covered in [Model-based reinforcement learning](24-model-based-rl), keep
rollouts short so errors can't compound for long.

**Weak rollouts.** A uniformly random rollout is a poor estimate in games
where random play is far from good play. AlphaGo mixed rollouts with a learned
value function, and AlphaZero dropped rollouts entirely, using a learned value
function and a learned policy to guide the search. [World models and MuZero](25-world-models-and-muzero) builds that
version.

**Tuning $c$.** UCT's constant depends on the scale of the returns. The
textbook $c = \sqrt 2$ assumes results in $[0, 1]$. With other scales, tune
$c$, or rescale the values.

## Further reading

- [Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd edition
  (2018), Chapter 8](http://incompleteideas.net/book/the-book-2nd.html).
  Dyna, the Dyna maze, prioritized sweeping, and MCTS.
- [Moore and Atkeson, "Prioritized sweeping: Reinforcement learning with less
  data and less time" (1993)](https://dl.acm.org/doi/abs/10.1023/A:1022635613229).
  The original prioritized sweeping paper.
- [Kocsis and Szepesvári, "Bandit based Monte-Carlo planning"
  (2006)](https://doi.org/10.1007/11871842_29). UCT and its consistency
  result.
- [Browne and colleagues, "A survey of Monte Carlo tree search methods"
  (2012)](https://doi.org/10.1109/TCIAIG.2012.2186810). The many variants of
  MCTS, and where each one works.

## The exercise

You build a learned-model planner, a prioritized planner, and a tree search.
Write `q_update` and `plan`, then `dyna_q` for the maze, and
`prioritized_sweeping` for a known model. Then write `uct_scores` and `mcts`
for Nim.

The checks verify the chain example's 0.9 and the terminal case by hand, and
that `plan` with the maze's full model reaches $Q^*$ from value iteration to
$10^{-6}$. With 50 planning steps, `dyna_q` must need several times fewer
real steps in episodes 2 to 10 than with none. `prioritized_sweeping` must
reach $Q^*$ within $10^{-3}$ in under 1,000 updates, at least five times fewer
than random planning. `uct_scores` must match the worked example, and `mcts`
must put most of its visits on the winning move for six Nim piles, the same
way on every run with the same seed.
