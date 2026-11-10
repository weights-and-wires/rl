---
title: "Planning with a learned model: Dyna"
slug: 021-dyna-planning-with-a-learned-model
part: "Part 3 — Learning from experience"
summary: Remember each real step in a model, replay it as extra Q-learning updates, and order the replays by how much they matter.
minutes: 35
gpu: false
objectives:
  - Explain why a remembered transition can be learned from again without another real step.
  - Implement Dyna-Q, and measure how planning steps cut the real experience needed to find a goal.
  - Say why random replay wastes most of its updates early in learning.
  - Implement prioritized sweeping on a known model, and count its updates against random replay.
prereqs:
  - 012-value-iteration
  - 017-sarsa-and-q-learning
exercise: 021-dyna-planning-with-a-learned-model
---

# Planning with a learned model: Dyna

A robot arm takes about a second per move, and every move wears the motors a little. Q-learning uses each of those moves once. It nudges one table entry and moves on, and the experience is gone. Yet every step taught the robot something lasting about its world: from this state, this action led there and paid that.

This chapter asks a simple question: can the agent learn from a step again, without taking another real one? The answer is yes, if the agent keeps a *model*, a record of what happened. Learning from the record, instead of from the world, is called *planning*.

You will see how much planning buys on a small maze, and then why replaying the record in random order is wasteful and how to fix it.

## What you already know

- The Q-learning update moves $Q(s, a)$ toward $r + \gamma \max_{a'} Q(s', a')$, as in [SARSA and Q-learning](/lessons/017-sarsa-and-q-learning).
- Value iteration applies the same backup to every state when the model is known, as in [Value iteration and contraction](/lessons/012-value-iteration).
- A *deterministic* world gives the same next state and reward every time for the same state and action.
- A step size $\alpha$ sets what fraction of the gap to the target one update closes.
- `rlenvs.gridworld` builds a maze from rows of characters.

## Why one update per step is slow

Take the smallest world that shows the problem. A chain $s_0 \to s_1 \to s_2$ has one action, *right*, and stepping into the terminal $s_2$ pays 1. Use $\gamma = 0.9$, $\alpha = 1$, and $Q = 0$ everywhere.

On the first real trip, the update at $s_0$ runs before the agent has learned anything about $s_1$, so it sees $0 + 0.9 \times 0 = 0$. Then the update at $s_1$ sees the reward and sets $Q(s_1) = 1$. The news about the goal has moved back one state. It takes a second trip for $Q(s_0)$ to become $0.9 \times 1 = 0.9$.

In a chain of 14 states, the goal's value needs 14 trips to reach the start. Each trip repeats long walks that tell the agent nothing it did not already know. The agent already *knows*, after the first trip, that $s_0$ leads to $s_1$. The only missing step is to use that fact.

```figure
<svg viewBox="0 0 560 250" role="img" aria-label="A three-state chain: planning on the remembered step from s0 to s1 raises Q of s0 from 0 to 0.9">
<defs><marker id="c021-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" class="fill-muted"/></marker></defs>
<text x="20.0" y="32.0" class="fig-bold" text-anchor="start">After the first real episode</text>
<rect x="60.0" y="40.0" width="100.0" height="44.0" class="fill-soft-a stroke-a" rx="6"/>
<text x="110.0" y="58.0" class="fig-small" text-anchor="middle">s0</text>
<text x="110.0" y="76.0" text-anchor="middle">Q = 0</text>
<rect x="230.0" y="40.0" width="100.0" height="44.0" class="fill-soft-a stroke-a" rx="6"/>
<text x="280.0" y="58.0" class="fig-small" text-anchor="middle">s1</text>
<text x="280.0" y="76.0" text-anchor="middle">Q = 1</text>
<rect x="400.0" y="40.0" width="100.0" height="44.0" class="fill-sunk stroke-muted" rx="6"/>
<text x="450.0" y="58.0" class="fig-small" text-anchor="middle">s2 (terminal)</text>
<line x1="162.0" y1="62.0" x2="228.0" y2="62.0" class="stroke-muted" marker-end="url(#c021-arrow)"/>
<line x1="332.0" y1="62.0" x2="398.0" y2="62.0" class="stroke-muted" marker-end="url(#c021-arrow)"/>
<text x="365.0" y="54.0" class="fig-small" text-anchor="middle">reward 1</text>
<text x="20.0" y="142.0" class="fig-bold" text-anchor="start">After one planning update on (s0, right)</text>
<rect x="60.0" y="150.0" width="100.0" height="44.0" class="fill-soft-a stroke-a" rx="6"/>
<text x="110.0" y="168.0" class="fig-small" text-anchor="middle">s0</text>
<text x="110.0" y="186.0" class="fig-bold" text-anchor="middle">Q = 0.9</text>
<rect x="230.0" y="150.0" width="100.0" height="44.0" class="fill-soft-a stroke-a" rx="6"/>
<text x="280.0" y="168.0" class="fig-small" text-anchor="middle">s1</text>
<text x="280.0" y="186.0" text-anchor="middle">Q = 1</text>
<rect x="400.0" y="150.0" width="100.0" height="44.0" class="fill-sunk stroke-muted" rx="6"/>
<text x="450.0" y="168.0" class="fig-small" text-anchor="middle">s2 (terminal)</text>
<line x1="162.0" y1="172.0" x2="228.0" y2="172.0" class="stroke-muted" marker-end="url(#c021-arrow)"/>
<line x1="332.0" y1="172.0" x2="398.0" y2="172.0" class="stroke-muted" marker-end="url(#c021-arrow)"/>
<text x="365.0" y="164.0" class="fig-small" text-anchor="middle">reward 1</text>
<text x="280.0" y="235.0" class="fig-small" text-anchor="middle">alpha = 1, gamma = 0.9. Planning gave s0 its value with no real step.</text>
</svg>
Figure 1. After one real trip, only Q(s1) has the goal's news. Replaying the remembered step from s0 pushes the news back one state with no real step. Notice that the planning update is the same Q-learning update, applied to a stored transition.
```

## Learn a model, then plan with it

A *model* here is a table. For each state-action pair the agent has tried, it stores what happened: the reward $r$, the next state $s'$, and whether the episode ended. A world that is deterministic needs only one entry per pair. In Python, a dict does the job:

```python
model[(s, a)] = (r, s2, terminated)
```

*Dyna-Q* (Sutton, 1990) does three things after every real step:

1. **Learn directly.** Apply the ordinary Q-learning update to the real transition.
2. **Learn the model.** Store the transition in the table.
3. **Plan.** Repeat $n$ times: pick a pair you have seen, uniformly at random, read its outcome from the model, and apply the Q-learning update as if it had happened again.

With $n = 0$, Dyna-Q is plain Q-learning. A planning update is the Q-learning update you already know, with a remembered pair $(\tilde s, \tilde a)$ in place of the real one:

$$
Q(\tilde s, \tilde a) \leftarrow Q(\tilde s, \tilde a) + \alpha \Bigl[\tilde r + \gamma \max_{a'} Q(\tilde s', a') - Q(\tilde s, \tilde a)\Bigr]
$$

Read it aloud: take a pair from the table, look up what it did, and pull its value toward the reward plus the best value of where it led. The agent spends computation instead of real steps. In numpy the planning loop is short:

```python
pairs = list(model)
for _ in range(n_steps):
    s, a = pairs[int(rng.integers(len(pairs)))]
    r, s2, terminated = model[(s, a)]
    Q[s, a] += alpha * (r + gamma * Q[s2].max() * (not terminated) - Q[s, a])
```

Return to the chain. After the first real trip, the model holds $(s_0, \text{right}) \to (0, s_1)$ and $(s_1, \text{right}) \to (1, s_2, \text{terminal})$. One planning update on $(s_0, \text{right})$ gives $Q(s_0) = 0 + 0.9 \times 1 = 0.9$. That value took the second real trip in plain Q-learning, and here it cost one table lookup.

> [!WARNING]
> Plan only from pairs in the model. A pair the agent never tried has no stored outcome, and reading one invents a transition that never happened.

## Measure the payoff

Sutton and Barto's Dyna maze is a 6 by 9 grid with a few walls. Reaching the goal in the top-right corner pays 1, every other step pays 0, and $\gamma = 0.95$. The shortest path takes 14 steps, so the start's optimal value is $0.95^{13} = 0.513$. The following table shows the mean number of real steps per episode over 10 seeded runs, with $\alpha = 0.1$ and $\varepsilon = 0.1$:

| Planning steps $n$ | Episode 1 | Episode 2 | Episode 3 | Episode 10 | Episodes 2 to 10, total |
|---|---|---|---|---|---|
| 0 | 1049 | 752 | 684 | 154 | 3159 |
| 5 | 1164 | 200 | 84 | 17 | 441 |
| 50 | 677 | 39 | 18 | 17 | 173 |

The first episode is a long random walk for everyone, because $Q$ is zero until the agent bumps into the goal by chance. After that, planning spreads the goal's value back through the whole remembered maze. With $n = 50$, the agent walks a near-shortest path from episode 3 on. Without planning, it is still wandering at episode 10.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Real steps per episode on the Dyna maze for 0, 5, and 50 planning steps: more planning reaches about 17 steps by episode 3">
<line x1="70.0" y1="250.0" x2="560.0" y2="250.0" class="stroke-grid"/>
<text x="62.0" y="254.0" class="fig-small" text-anchor="end">10</text>
<line x1="70.0" y1="154.4" x2="560.0" y2="154.4" class="stroke-grid"/>
<text x="62.0" y="158.4" class="fig-small" text-anchor="end">100</text>
<line x1="70.0" y1="58.8" x2="560.0" y2="58.8" class="stroke-grid"/>
<text x="62.0" y="62.8" class="fig-small" text-anchor="end">1000</text>
<text x="70.0" y="268.0" class="fig-small" text-anchor="middle">1</text>
<text x="124.4" y="268.0" class="fig-small" text-anchor="middle">2</text>
<text x="178.9" y="268.0" class="fig-small" text-anchor="middle">3</text>
<text x="233.3" y="268.0" class="fig-small" text-anchor="middle">4</text>
<text x="287.8" y="268.0" class="fig-small" text-anchor="middle">5</text>
<text x="342.2" y="268.0" class="fig-small" text-anchor="middle">6</text>
<text x="396.7" y="268.0" class="fig-small" text-anchor="middle">7</text>
<text x="451.1" y="268.0" class="fig-small" text-anchor="middle">8</text>
<text x="505.6" y="268.0" class="fig-small" text-anchor="middle">9</text>
<text x="560.0" y="268.0" class="fig-small" text-anchor="middle">10</text>
<line x1="70.0" y1="250.0" x2="560.0" y2="250.0" class="stroke-muted"/>
<text x="315.0" y="290.0" text-anchor="middle">episode</text>
<text x="62.0" y="18.0" class="fig-small" text-anchor="start">real steps per episode (log scale)</text>
<polyline points="70.0,56.8 124.4,70.6 178.9,74.6 233.3,98.3 287.8,123.4 342.2,115.8 396.7,101.1 451.1,120.3 505.6,143.6 560.0,136.4" class="stroke-a" fill="none"/>
<circle cx="70.0" cy="56.8" r="3" class="fill-a"/>
<circle cx="124.4" cy="70.6" r="3" class="fill-a"/>
<circle cx="178.9" cy="74.6" r="3" class="fill-a"/>
<circle cx="233.3" cy="98.3" r="3" class="fill-a"/>
<circle cx="287.8" cy="123.4" r="3" class="fill-a"/>
<circle cx="342.2" cy="115.8" r="3" class="fill-a"/>
<circle cx="396.7" cy="101.1" r="3" class="fill-a"/>
<circle cx="451.1" cy="120.3" r="3" class="fill-a"/>
<circle cx="505.6" cy="143.6" r="3" class="fill-a"/>
<circle cx="560.0" cy="136.4" r="3" class="fill-a"/>
<line x1="380.0" y1="62.0" x2="410.0" y2="62.0" class="stroke-a"/>
<text x="418.0" y="66.0" class="fig-small" text-anchor="start">n = 0 (Q-learning)</text>
<polyline points="70.0,52.5 124.4,125.5 178.9,161.4 233.3,192.2 287.8,214.0 342.2,217.5 396.7,222.9 451.1,226.5 505.6,228.2 560.0,229.0" class="stroke-c" fill="none"/>
<circle cx="70.0" cy="52.5" r="3" class="fill-c"/>
<circle cx="124.4" cy="125.5" r="3" class="fill-c"/>
<circle cx="178.9" cy="161.4" r="3" class="fill-c"/>
<circle cx="233.3" cy="192.2" r="3" class="fill-c"/>
<circle cx="287.8" cy="214.0" r="3" class="fill-c"/>
<circle cx="342.2" cy="217.5" r="3" class="fill-c"/>
<circle cx="396.7" cy="222.9" r="3" class="fill-c"/>
<circle cx="451.1" cy="226.5" r="3" class="fill-c"/>
<circle cx="505.6" cy="228.2" r="3" class="fill-c"/>
<circle cx="560.0" cy="229.0" r="3" class="fill-c"/>
<line x1="380.0" y1="80.0" x2="410.0" y2="80.0" class="stroke-c"/>
<text x="418.0" y="84.0" class="fig-small" text-anchor="start">n = 5</text>
<polyline points="70.0,75.0 124.4,193.9 178.9,226.5 233.3,228.2 287.8,229.5 342.2,229.0 396.7,231.0 451.1,227.2 505.6,229.5 560.0,228.2" class="stroke-b" fill="none"/>
<circle cx="70.0" cy="75.0" r="3" class="fill-b"/>
<circle cx="124.4" cy="193.9" r="3" class="fill-b"/>
<circle cx="178.9" cy="226.5" r="3" class="fill-b"/>
<circle cx="233.3" cy="228.2" r="3" class="fill-b"/>
<circle cx="287.8" cy="229.5" r="3" class="fill-b"/>
<circle cx="342.2" cy="229.0" r="3" class="fill-b"/>
<circle cx="396.7" cy="231.0" r="3" class="fill-b"/>
<circle cx="451.1" cy="227.2" r="3" class="fill-b"/>
<circle cx="505.6" cy="229.5" r="3" class="fill-b"/>
<circle cx="560.0" cy="228.2" r="3" class="fill-b"/>
<line x1="380.0" y1="98.0" x2="410.0" y2="98.0" class="stroke-b"/>
<text x="418.0" y="102.0" class="fig-small" text-anchor="start">n = 50</text>
</svg>
Figure 2. Real steps per episode, averaged over 10 seeds, on a log scale. Notice that the three curves start together and then separate: the planning agent turns one real step into many updates, so it needs far less real experience.
```

> [!KEY]
> Dyna-Q replays remembered transitions as extra Q-learning updates. Each real step buys $n$ imagined ones, so the agent needs far less real experience.

> [!WARNING]
> Tabular Dyna-Q trusts its model completely. If the world changes, such as a shortcut opening in the maze, the stored transitions are wrong and planning spreads the error. Dyna-Q+ adds a bonus $\kappa \sqrt{\tau}$ to a modeled reward, where $\tau$ is the number of steps since the pair was last tried, so the agent goes back to check stale entries.

## Update where the news is

Dyna-Q picks pairs at random, and that is wasteful. Early in the maze, only the pairs next to the goal have anything to learn. Updating a pair whose successor still has value 0 changes nothing, because the target is $0 + \gamma \times 0$, the same value the pair already holds. Most random picks land on such pairs.

A better rule updates the pairs whose values would change most, and does so first. This is *prioritized sweeping* (Moore and Atkeson, 1993). Define the *priority* of a pair as the size of the update it would make:

$$
p(s, a) = \Bigl| r + \gamma \max_{a'} Q(s', a') - Q(s, a) \Bigr|
$$

Read it aloud: how far the pair's value sits from its one-step target. Keep a priority queue and put a pair in it only if $p > \theta$, where $\theta$ is a small threshold. Always pop the largest priority, update that pair, and then ask who else cares. When $Q(s, \cdot)$ changes, the value of $s$ may have changed, so every *predecessor* $(\bar s, \bar a)$ whose model leads to $s$ has an updated priority. Recompute it, and queue it if it clears $\theta$. News flows backward from where it arrived.

In the chain, once $Q(s_1)$ becomes 1, the predecessor $(s_0, \text{right})$ gets priority $|0 + 0.9 \times 1 - 0| = 0.9$, enters the queue, and is the next pair updated.

Python's `heapq` is a min-heap, so push the negated priority:

```python
heapq.heappush(heap, (-p, s, a))
neg_p, s, a = heapq.heappop(heap)     # the highest priority comes out first
```

The exercise starts prioritized sweeping from $Q = 0$ with the maze's full model and $\alpha = 1$. When the queue empties, every pair's priority is at most $\theta$. The backup is a $\gamma$-contraction, so the error is then at most $\theta / (1 - \gamma)$. On the maze, prioritized sweeping reaches $Q^*$ in 184 updates, one per non-terminal pair. The priorities fall off as powers of $\gamma$ with distance from the goal, so the queue works outward from the goal, and each pair's first update already uses its successor's final value. Random planning needs about 3,000 updates to get within $10^{-3}$.

```figure
<svg viewBox="0 0 600 200" role="img" aria-label="Bar chart: prioritized sweeping needs 184 updates and random planning needs about 3,100">
<text x="20.0" y="24.0" class="fig-bold" text-anchor="start">Updates until Q is within 0.001 of Q*, Dyna maze with a full model</text>
<text x="20.0" y="62.0" class="fig-small" text-anchor="start">prioritized sweeping</text>
<rect x="20.0" y="70.0" width="24.5" height="26.0" class="fill-b"/>
<text x="52.5" y="89.0" class="fig-bold" text-anchor="start">184</text>
<text x="20.0" y="132.0" class="fig-small" text-anchor="start">random planning (median of 5 seeds)</text>
<rect x="20.0" y="140.0" width="413.3" height="26.0" class="fill-a"/>
<text x="441.3" y="159.0" class="fig-bold" text-anchor="start">3,100</text>
</svg>
Figure 3. Updates needed to reach Q* within 0.001 on the Dyna maze with a full model. Notice that prioritized sweeping spends 184 updates, each on a pair that has news, while random planning spends about 17 times as many.
```

> [!KEY]
> Prioritized sweeping updates the pairs whose values would change most, and then their predecessors. Changes propagate backward from the goal instead of being found by chance.

> [!RECAP]
> - A model is a dict from `(s, a)` to `(r, s2, terminated)`. Plan by drawing keys uniformly and applying the Q-learning update.
> - Dyna-Q calls the planning loop $n$ times after every real step.
> - Prioritized sweeping pops the largest $|r + \gamma \max Q(s') - Q(s, a)|$, updates it, and queues predecessors above $\theta$.
> - Both methods trust the model. They fail when the model is wrong.

## Build it step by step

The checks test the functions in this order. `dyna_maze` is already written.

1. **`q_update(Q, s, a, r, s2, terminated, alpha, gamma)`.** The Q-learning update from the previous chapters, in place. Common mistake: bootstrapping on a terminal step. The target is `r` alone when `terminated` is true.
2. **`plan(Q, model, n_steps, alpha, gamma, rng)`.** Returns `Q`. Turn the dict's keys into a list once, then draw indices with `rng.integers`. Common mistake: planning from pairs you never tried.
3. **`dyna_q(mdp, n_episodes, n_planning, alpha, epsilon, seed)`.** Returns `(Q, steps)`. After each real step, call `q_update`, store the model entry, then call `plan`. Use one generator for acting, stepping, and planning.
4. **`prioritized_sweeping(model, n_states, n_actions, gamma, theta)`.** Returns `(Q, updates)`. Build a predecessor map from `s2` to the set of pairs that lead to it, queue every pair above `theta`, and loop. Common mistake: leaving an old, lower copy of a pair in the heap and updating it twice. Keep a dict of each queued pair's current priority, and skip stale entries when they surface.

## Where this breaks

**Stochastic worlds.** The model here stores only the last outcome of each pair. When outcomes are random, that single sample is treated as certain. Store counts of outcomes and sample from them, or learn a model of the distribution.

**Wrong models.** A model that is wrong somewhere misleads every planning step that touches it. Dyna-Q+ re-explores stale pairs. [Learning a model and planning with it](/lessons/043-model-based-rl) shows how deep model-based methods keep rollouts short so that errors cannot compound for long.

**Huge state spaces.** Both methods keep a table entry per pair. The next chapter, [Monte Carlo tree search](/lessons/022-monte-carlo-tree-search), plans without a table of the whole world.

## Further reading

- [Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd edition (2018), Chapter 8](http://incompleteideas.net/book/the-book-2nd.html). Dyna, the Dyna maze, and prioritized sweeping.
- [Sutton, "Integrated architectures for learning, planning, and reacting based on approximating dynamic programming" (1990)](https://doi.org/10.1016/B978-1-55860-141-3.50030-4). The original Dyna paper.
- [Moore and Atkeson, "Prioritized sweeping: Reinforcement learning with less data and less time" (1993)](https://dl.acm.org/doi/abs/10.1023/A:1022635613229). The original prioritized sweeping paper.

## The exercise

You build a learned-model planner and a prioritized planner. Write `q_update` and `plan`, then `dyna_q` for the maze, and `prioritized_sweeping` for a known model.

The checks verify the Q-learning update and its terminal case by hand, and the chain example's 0.9. They confirm that `plan` with the maze's full model reaches $Q^*$ from value iteration to $10^{-6}$. With 50 planning steps, `dyna_q` must need several times fewer real steps in episodes 2 to 10 than with none. `prioritized_sweeping` must reach $Q^*$ within $10^{-3}$ in under 1,000 updates, at least five times fewer than random planning.
