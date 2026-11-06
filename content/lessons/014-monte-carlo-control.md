---
title: Monte Carlo control
slug: 014-monte-carlo-control
part: "Part 3 — Learning from experience"
summary: Learn action values from your own epsilon-greedy episodes, and see why exploring leads to the best epsilon-soft policy and not the optimal one.
minutes: 40
gpu: false
objectives:
  - Explain why control without a model needs action values, not state values.
  - Build an epsilon-greedy policy from a table of Q values, and say why every action keeps a floor of epsilon over A.
  - Run on-policy Monte Carlo control with a constant step size.
  - Say which policy the loop converges toward, and compute its value from a modified world.
prereqs:
  - 003-explore-or-exploit
  - 012-value-iteration
  - 013-monte-carlo-prediction
exercise: 014-monte-carlo-control
---

# Monte Carlo control

You can now estimate how good a fixed policy is from episodes alone. But the real goal is a good policy, and a robot with no model has no way to look one step ahead. With a model, you improve a policy by comparing the actions' one-step lookaheads, which needs the transition probabilities. Without one, you have to learn the comparison itself from experience.

Two things stand in the way. First, state values $V$ aren't enough: to pick between "left" and "right" you need the value of each action, and you can't get that from $V$ without the model. Second, if you always act on your current best guess, you only ever try what already looks best. An action that looked bad because of one unlucky return never gets a second chance, and its estimate never improves. This chapter fixes both: it learns action values, and it keeps exploring on purpose.

## What you already know

- $Q^\pi(s, a)$ is the expected return when you take $a$ in $s$ and then follow $\pi$.
- Monte Carlo prediction averages the returns that followed each visit, and `discounted_returns` computes them with one backward pass.
- Policy iteration alternates evaluating a policy and acting greedily on its values.
- Epsilon-greedy picks a random action with probability $\varepsilon$ and the best-looking one otherwise (see [Explore or exploit](/lessons/003-explore-or-exploit)).
- Value iteration solves a known MDP exactly, so you can use it to check what you learn.

## Learn action values, then act on them

Without a model, store a table `Q` of shape `(S, A)` and estimate each entry as the average return after taking that action in that state. The greedy action in state $s$ is `Q[s].argmax()`. The loop is policy iteration with samples in place of the model:

1. Play one episode with the current policy.
2. Move each visited $Q(s, a)$ toward the return that followed it.
3. Make the policy greedier with respect to the updated $Q$, and repeat.

Mixing a little evaluation with a little improvement like this is called *generalized policy iteration*. Step 3 is where the trouble sits. If the policy is fully greedy, it plays one action per state, so only that action's $Q$ ever changes. The other actions keep their starting estimates forever, however good they really are.

## Keep a floor under every action

The fix is to never let any action's probability reach zero. An *$\varepsilon$-greedy* policy picks the greedy action most of the time and a uniformly random action with probability $\varepsilon$. With $A$ actions:

$$
\pi(a \mid s) =
\begin{cases}
1 - \varepsilon + \dfrac{\varepsilon}{A} & \text{if } a = \operatorname*{arg\,max}_{a'} Q(s, a') \\[4pt]
\dfrac{\varepsilon}{A} & \text{otherwise}
\end{cases}
$$

Read it as: every action gets a floor of $\varepsilon / A$, its share of the random pick, and the greedy action also gets the remaining $1 - \varepsilon$. A policy that gives every action at least $\varepsilon / A$ is called *$\varepsilon$-soft*. For $Q(s, \cdot) = (0.5, 2.0, 1.0, -1.0)$ and $\varepsilon = 0.2$, the floor is $0.2 / 4 = 0.05$, and the greedy action 1 gets $0.8 + 0.05 = 0.85$, so the row is $(0.05, 0.85, 0.05, 0.05)$.

```figure
<svg viewBox="0 0 570 240" role="img" aria-label="Action values 0.5, 2, 1, minus 1 and the epsilon-greedy policy they produce: 0.05, 0.85, 0.05, 0.05">
<text x="14.0" y="24.0" class="fig-bold" text-anchor="start">Q(s, a)</text>
<text x="330.0" y="24.0" class="fig-bold" text-anchor="start">policy, epsilon = 0.2</text>
<rect x="80.0" y="139.0" width="32.0" height="11.0" rx="2" class="fill-soft-a stroke-a"/>
<text x="96.0" y="133.0" class="fig-small" text-anchor="middle">0.5</text>
<text x="96.0" y="206.0" class="fig-small" text-anchor="middle">a0</text>
<rect x="344.0" y="144.5" width="32.0" height="5.5" rx="2" class="fill-sunk stroke-muted"/>
<text x="360.0" y="138.5" class="fig-small" text-anchor="middle">0.05</text>
<text x="360.0" y="206.0" class="fig-small" text-anchor="middle">a0</text>
<rect x="130.0" y="106.0" width="32.0" height="44.0" rx="2" class="fill-soft-a stroke-a"/>
<text x="146.0" y="100.0" class="fig-small" text-anchor="middle">2</text>
<text x="146.0" y="206.0" class="fig-small" text-anchor="middle">a1</text>
<rect x="394.0" y="56.5" width="32.0" height="93.5" rx="2" class="fill-soft-b stroke-b"/>
<text x="410.0" y="50.5" class="fig-small" text-anchor="middle">0.85</text>
<text x="410.0" y="206.0" class="fig-small" text-anchor="middle">a1</text>
<rect x="180.0" y="128.0" width="32.0" height="22.0" rx="2" class="fill-soft-a stroke-a"/>
<text x="196.0" y="122.0" class="fig-small" text-anchor="middle">1</text>
<text x="196.0" y="206.0" class="fig-small" text-anchor="middle">a2</text>
<rect x="444.0" y="144.5" width="32.0" height="5.5" rx="2" class="fill-sunk stroke-muted"/>
<text x="460.0" y="138.5" class="fig-small" text-anchor="middle">0.05</text>
<text x="460.0" y="206.0" class="fig-small" text-anchor="middle">a2</text>
<rect x="230.0" y="150.0" width="32.0" height="22.0" rx="2" class="fill-soft-a stroke-a"/>
<text x="246.0" y="186.0" class="fig-small" text-anchor="middle">-1</text>
<text x="246.0" y="206.0" class="fig-small" text-anchor="middle">a3</text>
<rect x="494.0" y="144.5" width="32.0" height="5.5" rx="2" class="fill-sunk stroke-muted"/>
<text x="510.0" y="138.5" class="fig-small" text-anchor="middle">0.05</text>
<text x="510.0" y="206.0" class="fig-small" text-anchor="middle">a3</text>
<line x1="60.0" y1="150.0" x2="250.0" y2="150.0" class="stroke-muted"/>
<line x1="335.0" y1="150.0" x2="550.0" y2="150.0" class="stroke-muted"/>
<text x="285.0" y="228.0" class="fig-small" text-anchor="middle">every action gets 0.2 / 4 = 0.05, and the greedy action gets 0.8 more</text>
</svg>
Figure 1. One row of Q, and the epsilon-greedy row built from it. Notice that the three non-greedy actions are not ignored: each keeps a floor of 0.05, so each still gets tried.
```

```python
pi = np.full((S, A), epsilon / A)
pi[np.arange(S), Q.argmax(axis=1)] += 1.0 - epsilon
```

## Move Q toward each return

After an episode, find each state-action pair's first visit, take the return that followed it, and move the estimate a fraction $\alpha$ of the way there:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \bigl(G_t - Q(s_t, a_t)\bigr)
$$

Read it as: compute the gap between the return you saw and your current estimate, and close a fraction $\alpha$ of it. With $Q = 0.5$, $G = -1$, and $\alpha = 0.1$, the gap is $-1.5$ and the estimate moves to $0.5 - 0.15 = 0.35$.

You could use $\alpha = 1/N$, where $N$ counts visits, which turns the update into a running mean. This chapter uses a constant $\alpha$ instead. The policy keeps changing, so the early returns came from a worse policy, and a running mean would weight those stale returns forever. A constant $\alpha$ shrinks an old return's weight by $1 - \alpha$ with every update.

Acting epsilon-greedily has one more trap. When all estimates are tied, such as $Q = 0$ at the start, `argmax` always returns the first action. In the exercise's grid that action is "up", and the agent walks into the top wall for hundreds of steps. So break ties between best actions at random:

```python
if rng.random() < epsilon:
    a = rng.integers(A)
else:
    a = rng.choice(np.flatnonzero(Q[s] == Q[s].max()))
```

## What the loop converges to

Because the policy explores forever, it can't converge to the optimal policy, which never takes a random action. What does it converge to? Sutton and Barto answer with a modified world. In it, every action you choose is replaced, with probability $\varepsilon$, by a uniformly random one. Playing greedily in that world is the same as playing $\varepsilon$-greedily in the real one, so the best $\varepsilon$-soft policy is worth exactly what the optimal policy is worth in the modified world. Its model takes two lines:

```python
P_soft = (1 - eps) * mdp.P + eps * mdp.P.mean(axis=1, keepdims=True)
R_soft = (1 - eps) * mdp.R + eps * mdp.R.mean(axis=1, keepdims=True)
```

Solve that world with value iteration and you have the target. On the exercise's grid, three rows of four cells with one wall and a $-1$ forevery step, the optimal policy reaches the goal in 3 steps from the start, worth $-3$. The best $\varepsilon$-soft policy with $\varepsilon = 0.2$ is worth $-3.818$.

```figure
<svg viewBox="0 0 570 250" role="img" aria-label="A three by four grid with one wall: value of each cell under the best epsilon-soft policy with epsilon 0.2, the optimal value in brackets, and the greedy action as an arrow">
<defs><marker id="c014-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<rect x="62.0" y="22.0" width="106.0" height="58.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="103.0" y="53.0" class="fig-bold" text-anchor="middle">-6.04</text>
<text x="103.0" y="70.0" class="fig-small" text-anchor="middle">(-5)</text>
<line x1="137.0" y1="51.0" x2="155.0" y2="51.0" class="stroke-ink" marker-end="url(#c014-arrow)"/>
<rect x="172.0" y="22.0" width="106.0" height="58.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="213.0" y="53.0" class="fig-bold" text-anchor="middle">-4.92</text>
<text x="213.0" y="70.0" class="fig-small" text-anchor="middle">(-4)</text>
<line x1="247.0" y1="51.0" x2="265.0" y2="51.0" class="stroke-ink" marker-end="url(#c014-arrow)"/>
<rect x="282.0" y="22.0" width="106.0" height="58.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="323.0" y="53.0" class="fig-bold" text-anchor="middle">-3.68</text>
<text x="323.0" y="70.0" class="fig-small" text-anchor="middle">(-3)</text>
<line x1="366.0" y1="42.0" x2="366.0" y2="60.0" class="stroke-ink" marker-end="url(#c014-arrow)"/>
<rect x="392.0" y="22.0" width="106.0" height="58.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="433.0" y="53.0" class="fig-bold" text-anchor="middle">-2.56</text>
<text x="433.0" y="70.0" class="fig-small" text-anchor="middle">(-2)</text>
<line x1="476.0" y1="42.0" x2="476.0" y2="60.0" class="stroke-ink" marker-end="url(#c014-arrow)"/>
<rect x="62.0" y="84.0" width="106.0" height="58.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="103.0" y="115.0" class="fig-bold" text-anchor="middle">-5.05</text>
<text x="103.0" y="132.0" class="fig-small" text-anchor="middle">(-4)</text>
<line x1="146.0" y1="104.0" x2="146.0" y2="122.0" class="stroke-ink" marker-end="url(#c014-arrow)"/>
<rect x="282.0" y="84.0" width="106.0" height="58.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="323.0" y="115.0" class="fig-bold" text-anchor="middle">-2.50</text>
<text x="323.0" y="132.0" class="fig-small" text-anchor="middle">(-2)</text>
<line x1="357.0" y1="113.0" x2="375.0" y2="113.0" class="stroke-ink" marker-end="url(#c014-arrow)"/>
<rect x="392.0" y="84.0" width="106.0" height="58.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="433.0" y="115.0" class="fig-bold" text-anchor="middle">-1.32</text>
<text x="433.0" y="132.0" class="fig-small" text-anchor="middle">(-1)</text>
<line x1="476.0" y1="104.0" x2="476.0" y2="122.0" class="stroke-ink" marker-end="url(#c014-arrow)"/>
<rect x="62.0" y="146.0" width="106.0" height="58.0" rx="4" class="fill-soft-b stroke-b"/>
<text x="103.0" y="177.0" class="fig-bold" text-anchor="middle">-3.82</text>
<text x="103.0" y="194.0" class="fig-small" text-anchor="middle">(-3)</text>
<line x1="137.0" y1="175.0" x2="155.0" y2="175.0" class="stroke-ink" marker-end="url(#c014-arrow)"/>
<text x="66.0" y="159.0" class="fig-small" text-anchor="start">start</text>
<rect x="172.0" y="146.0" width="106.0" height="58.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="213.0" y="177.0" class="fig-bold" text-anchor="middle">-2.57</text>
<text x="213.0" y="194.0" class="fig-small" text-anchor="middle">(-2)</text>
<line x1="247.0" y1="175.0" x2="265.0" y2="175.0" class="stroke-ink" marker-end="url(#c014-arrow)"/>
<rect x="282.0" y="146.0" width="106.0" height="58.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="323.0" y="177.0" class="fig-bold" text-anchor="middle">-1.32</text>
<text x="323.0" y="194.0" class="fig-small" text-anchor="middle">(-1)</text>
<line x1="357.0" y1="175.0" x2="375.0" y2="175.0" class="stroke-ink" marker-end="url(#c014-arrow)"/>
<rect x="392.0" y="146.0" width="106.0" height="58.0" rx="4" class="fill-soft-c stroke-c"/>
<text x="445.0" y="179.0" class="fig-bold" text-anchor="middle">goal</text>
<rect x="172.0" y="84.0" width="106.0" height="58.0" rx="4" class="fill-muted"/>
<text x="225.0" y="117.0" class="fig-bold fig-small" text-anchor="middle">wall</text>
<text x="14.0" y="222.0" class="fig-small" text-anchor="start">Each cell shows the value under the best epsilon-soft policy, then the optimal value in brackets.</text>
<text x="14.0" y="238.0" class="fig-small" text-anchor="start">The arrow is the best epsilon-soft policy's greedy action.</text>
</svg>
Figure 2. The grid from the exercise, with every step paying minus 1. Notice that the start is worth -3.818 under the best epsilon-soft policy and -3 under the optimal one, even though both walk the same three steps along the bottom row: the gap is the cost of random moves.
```

Both policies take the same route along the bottom row, so the gap comes from the random moves: a stumble costs steps. On a grid where the short route is dangerous, exploring also changes the route, as the [SARSA and Q-learning chapter](/lessons/017-sarsa-and-q-learning) shows on the cliff.

```figure
<svg viewBox="0 0 570 285" role="img" aria-label="Value of the epsilon-soft policy learned by Monte Carlo control against the number of episodes, settling at the best epsilon-soft value of minus 3.818 and not at the optimal minus 3">
<line x1="80.0" y1="214.2" x2="540.0" y2="214.2" class="stroke-grid"/>
<text x="72.0" y="218.2" class="fig-small" text-anchor="end">-4.0</text>
<line x1="80.0" y1="182.5" x2="540.0" y2="182.5" class="stroke-grid"/>
<text x="72.0" y="186.5" class="fig-small" text-anchor="end">-3.8</text>
<line x1="80.0" y1="150.8" x2="540.0" y2="150.8" class="stroke-grid"/>
<text x="72.0" y="154.8" class="fig-small" text-anchor="end">-3.6</text>
<line x1="80.0" y1="119.2" x2="540.0" y2="119.2" class="stroke-grid"/>
<text x="72.0" y="123.2" class="fig-small" text-anchor="end">-3.4</text>
<line x1="80.0" y1="87.5" x2="540.0" y2="87.5" class="stroke-grid"/>
<text x="72.0" y="91.5" class="fig-small" text-anchor="end">-3.2</text>
<line x1="80.0" y1="55.8" x2="540.0" y2="55.8" class="stroke-grid"/>
<text x="72.0" y="59.8" class="fig-small" text-anchor="end">-3.0</text>
<line x1="80.0" y1="230.0" x2="80.0" y2="234.0" class="stroke-muted"/>
<text x="80.0" y="248.0" class="fig-small" text-anchor="middle">20</text>
<line x1="240.8" y1="230.0" x2="240.8" y2="234.0" class="stroke-muted"/>
<text x="240.8" y="248.0" class="fig-small" text-anchor="middle">100</text>
<line x1="470.8" y1="230.0" x2="470.8" y2="234.0" class="stroke-muted"/>
<text x="470.8" y="248.0" class="fig-small" text-anchor="middle">1000</text>
<text x="310.0" y="266.0" text-anchor="middle">episodes of experience</text>
<line x1="80.0" y1="55.8" x2="540.0" y2="55.8" class="stroke-c dashed"/>
<text x="540.0" y="49.8" class="fig-small" text-anchor="end">optimal policy: -3</text>
<line x1="80.0" y1="185.4" x2="540.0" y2="185.4" class="stroke-a dashed"/>
<text x="88.0" y="179.4" class="fig-small" text-anchor="start">best epsilon-soft policy: -3.818</text>
<polyline points="80.0,206.1 171.5,198.5 240.8,190.7 310.0,190.6 401.5,205.0 470.8,185.4 540.0,185.4" class="stroke-b"/>
<circle cx="80.0" cy="206.1" r="4" class="fill-b"/>
<circle cx="171.5" cy="198.5" r="4" class="fill-b"/>
<circle cx="240.8" cy="190.7" r="4" class="fill-b"/>
<circle cx="310.0" cy="190.6" r="4" class="fill-b"/>
<circle cx="401.5" cy="205.0" r="4" class="fill-b"/>
<circle cx="470.8" cy="185.4" r="4" class="fill-b"/>
<circle cx="540.0" cy="185.4" r="4" class="fill-b"/>
<text x="14.0" y="24.0" class="fig-small" text-anchor="start">value of the learned epsilon-soft policy at the start (median of 5 seeds)</text>
</svg>
Figure 3. Monte Carlo control with epsilon 0.2 and step size 0.1, median of five seeds at each size. Notice that the curve settles on the lower dashed line, the best epsilon-soft value, and never reaches the optimal value above it.
```

> [!KEY]
> On-policy control with $\varepsilon$-greedy exploration converges toward the best $\varepsilon$-soft policy, not the optimal one. The price of exploring is built into the values it learns, because those values describe the policy that explores.

> [!WARNING]
> No theorem guarantees that this loop settles with a constant $\alpha$. The estimates keep fluctuating, and an unlucky run can lock in a bad greedy action for a while. At 200 episodes, the worst of the five seeds behind Figure 3 was worth $-25.0$, far below the median. The exercise's check takes the median of five seeded runs for that reason.

> [!RECAP]
> - Without a model you need $Q(s, a)$, and the greedy action is `Q[s].argmax()`.
> - An $\varepsilon$-greedy row is a floor of $\varepsilon / A$ plus $1 - \varepsilon$ on the first greedy action.
> - Control plays an episode, then moves each first-visited $Q(s, a)$ a fraction $\alpha$ toward its return.
> - Break ties between equally good actions at random, or the agent repeats one action while $Q$ is still all zeros.
> - The loop approaches the best $\varepsilon$-soft policy, whose value you can compute from the modified world.

## Build it step by step

The checks test the functions in this order. `discounted_returns` is given.

1. **`epsilon_soft(Q, epsilon)`.** Takes `Q` of shape `(S, A)` and returns a policy of shape `(S, A)`. Fill with `epsilon / A`, then add `1 - epsilon` at `Q.argmax(axis=1)`. Common mistake: giving the greedy action $1 - \varepsilon$ and forgetting its share of the random pick, so the rows sum to less than one.
2. **`epsilon_soft_world(mdp, epsilon)`.** Returns a new `rlenvs.TabularMDP` with the same `gamma`, `mu0`, and `terminal`, and the mixed `P` and `R`. Average over the action axis, which is axis 1, and keep the dimension. Common mistake: averaging over the wrong axis. The checks confirm that the optimal value of this world from the start is $-3.818$.
3. **`mc_control(mdp, n_episodes, epsilon, alpha, seed)`.** Returns `Q`. Make one `np.random.default_rng(seed)`. For each episode, draw the start from `mdp.mu0`, act as in the code above, and step with `mdp.sample(s, a, rng)` until `mdp.terminal[s2]` or `mdp.max_steps` steps. Then update each first-visited `(s, a)` pair with its return. Common mistake: always taking `argmax` among ties.

## Cost and scaling

Time grows with the steps you play, $O(T)$ per episode, and memory with the table, $S \times A$ values. The hidden cost is how many episodes you need. Every action in every state needs enough visits to average out its noise, and the floor of $\varepsilon / A$ means a rarely chosen action gets few of them.

## Where this breaks

**Episodes that never end.** A policy that never reaches the goal never produces a return. Cap the episode length with `mdp.max_steps`, as `mc_control` does.

**Stuck greedy actions.** An action with an unlucky early estimate is tried only $\varepsilon / A$ of the time, so it recovers slowly. A larger $\varepsilon$, a constant step size, or *exploring starts*, which begin each episode from a random state and action, all help.

**A fixed $\varepsilon$ costs value forever.** The learned policy can never beat the best $\varepsilon$-soft policy. To approach the optimal policy, shrink $\varepsilon$ toward zero as learning goes on, or learn about a different policy than the one you play, which is the subject of [the off-policy chapter](/lessons/015-off-policy-importance-sampling).

## Further reading

- [Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd edition (2018), Chapter 5](http://incompleteideas.net/book/the-book-2nd.html) is the source of the modified-world argument for $\varepsilon$-soft policies.
- [Tsitsiklis, "On the convergence of optimistic policy iteration" (2002)](https://www.jmlr.org/papers/v3/tsitsiklis02a.html) proves convergence for a form of optimistic policy iteration that estimates $Q$ values by Monte Carlo.
- [Singh and Sutton, "Reinforcement learning with replacing eligibility traces" (1996)](https://doi.org/10.1007/BF00114726) analyzes first-visit against every-visit Monte Carlo.

## The exercise

You build on-policy Monte Carlo control. Write `epsilon_soft(Q, epsilon)`, `epsilon_soft_world(mdp, epsilon)`, and `mc_control(mdp, n_episodes, epsilon, alpha, seed)`. The starter gives `discounted_returns`.

The checks verify the table $(0.05, 0.85, 0.05, 0.05)$, that $\varepsilon = 0$ gives the greedy policy and $\varepsilon = 1$ the uniform one, and that every row sums to one. They check that your modified world has valid rows and that the best $\varepsilon$-soft value from the start is $-3.818$, equal to playing $\varepsilon$-greedily in the real grid. Last, they check that `mc_control` learns $Q = (1, -1)$ on a two-armed one-step problem with $\alpha = 1$, and that the median of five seeded runs on the grid comes within 0.5 of the best $\varepsilon$-soft value.
