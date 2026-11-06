---
title: Monte Carlo prediction
slug: 013-monte-carlo-prediction
part: "Part 3 — Learning from experience"
summary: Estimate a policy's state values by averaging the returns of complete episodes, with no model of the world.
minutes: 35
gpu: false
objectives:
  - Explain why the average of the returns that followed a state estimates its value.
  - Compute the return after every step of an episode by hand, and by one backward pass.
  - Compute first-visit and every-visit estimates from episodes, and say why they differ.
  - Say how fast the estimate improves as you average more episodes.
prereqs:
  - 002-returns-and-discounting
  - 010-policy-evaluation
exercise: 013-monte-carlo-prediction
---

# Monte Carlo prediction

The last three chapters found exact values, but only because you handed the algorithm the whole model: every transition probability and every reward. Most real problems don't come with that table. A robot doesn't know its gripper's physics to four decimal places, and a game doesn't announce the odds of every position. What the robot can do is try things and see what happens.

That is enough to estimate a value. Think of estimating the length of your commute: you don't model the traffic lights, you drive the route twenty times and take the mean. This chapter turns that into a method. Play an episode to the end, record what you collected after each state, and average over many episodes. It is called Monte Carlo prediction, "prediction" because it predicts the value of a fixed policy, and "Monte Carlo" because it averages random samples.

## What you already know

- A policy $\pi(a \mid s)$ gives the probability of each action in each state.
- The return $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$ is the discounted sum of rewards from step $t$ to the end of the episode.
- The value $V^\pi(s)$ is the expected return when you start in $s$ and follow $\pi$.
- An episode ends when the agent reaches a terminal state, a state with no future.
- A sample mean of independent draws estimates their expected value, and it gets more accurate as you add draws.

## A value is an average, so average

Dynamic programming computes $V^\pi(s) = \mathbb{E}[G_t \mid s_t = s]$ by using the model to enumerate everything that could happen next. Without a model you can't enumerate, but you can sample: run $\pi$, and each time the agent is in $s$, write down the return that followed. The average of those numbers is an estimate of the expectation, and it needs nothing but the episodes themselves.

You'll test this on Sutton and Barto's random walk: five states in a row, numbered 1 to 5, with an exit at each end. At every step the walker moves left or right with probability one half. Leaving on the right pays 1, leaving on the left pays 0, and every other step pays 0. Episodes start in state 3, and $\gamma = 1$. The value of a state is the chance of leaving on the right. Because a state's value is the average of its two neighbors', the values lie on a straight line from 0 at the left exit to 1 at the right exit, counting the payment as the exit's value, so $V(k) = k/6$. You know the answer, which lets you check the method.

```figure
<svg viewBox="0 0 570 300" role="img" aria-label="One random walk episode: start in state 3, go to 2, back to 3, then 4, 5, and out through the right exit, so the return from every step is 1">
<defs><marker id="c013-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<text x="14.0" y="22.0" class="fig-small" text-anchor="start">state</text>
<rect x="34.0" y="32.0" width="52.0" height="30.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="60.0" y="52.0" class="fig-bold" text-anchor="middle">exit</text>
<rect x="109.0" y="32.0" width="52.0" height="30.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="135.0" y="52.0" class="fig-bold" text-anchor="middle">1</text>
<rect x="184.0" y="32.0" width="52.0" height="30.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="210.0" y="52.0" class="fig-bold" text-anchor="middle">2</text>
<rect x="259.0" y="32.0" width="52.0" height="30.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="285.0" y="52.0" class="fig-bold" text-anchor="middle">3</text>
<rect x="334.0" y="32.0" width="52.0" height="30.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="360.0" y="52.0" class="fig-bold" text-anchor="middle">4</text>
<rect x="409.0" y="32.0" width="52.0" height="30.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="435.0" y="52.0" class="fig-bold" text-anchor="middle">5</text>
<rect x="484.0" y="32.0" width="52.0" height="30.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="510.0" y="52.0" class="fig-bold" text-anchor="middle">exit</text>
<text x="60.0" y="78.0" class="fig-small" text-anchor="middle">pays 0</text>
<text x="510.0" y="78.0" class="fig-small" text-anchor="middle">pays 1</text>
<text x="14.0" y="94.0" class="fig-small" text-anchor="start">time</text>
<text x="24.0" y="122.0" class="fig-small" text-anchor="middle">t = 0</text>
<text x="24.0" y="152.0" class="fig-small" text-anchor="middle">t = 1</text>
<text x="24.0" y="182.0" class="fig-small" text-anchor="middle">t = 2</text>
<text x="24.0" y="212.0" class="fig-small" text-anchor="middle">t = 3</text>
<text x="24.0" y="242.0" class="fig-small" text-anchor="middle">t = 4</text>
<text x="24.0" y="272.0" class="fig-small" text-anchor="middle">t = 5</text>
<line x1="274.0" y1="126.0" x2="221.0" y2="140.0" class="stroke-muted" marker-end="url(#c013-arrow)"/>
<line x1="221.0" y1="156.0" x2="274.0" y2="170.0" class="stroke-muted" marker-end="url(#c013-arrow)"/>
<line x1="296.0" y1="186.0" x2="349.0" y2="200.0" class="stroke-muted" marker-end="url(#c013-arrow)"/>
<line x1="371.0" y1="216.0" x2="424.0" y2="230.0" class="stroke-muted" marker-end="url(#c013-arrow)"/>
<line x1="446.0" y1="246.0" x2="499.0" y2="260.0" class="stroke-muted" marker-end="url(#c013-arrow)"/>
<circle cx="285.0" cy="118.0" r="8" class="fill-b"/>
<circle cx="210.0" cy="148.0" r="8" class="fill-a"/>
<circle cx="285.0" cy="178.0" r="8" class="fill-soft-b stroke-b"/>
<circle cx="360.0" cy="208.0" r="8" class="fill-a"/>
<circle cx="435.0" cy="238.0" r="8" class="fill-a"/>
<circle cx="510.0" cy="268.0" r="8" class="fill-sunk stroke-muted"/>
<text x="560.0" y="122.0" class="fig-small" text-anchor="end">G = 1</text>
<text x="560.0" y="152.0" class="fig-small" text-anchor="end">G = 1</text>
<text x="560.0" y="182.0" class="fig-small" text-anchor="end">G = 1</text>
<text x="560.0" y="212.0" class="fig-small" text-anchor="end">G = 1</text>
<text x="560.0" y="242.0" class="fig-small" text-anchor="end">G = 1</text>
<text x="267.0" y="122.0" class="fig-small" text-anchor="end">first visit to state 3</text>
<text x="267.0" y="182.0" class="fig-small" text-anchor="end">second visit</text>
</svg>
Figure 1. One episode: states 3, 2, 3, 4, 5, then the right exit. Notice that every step has return 1, because the only reward comes at the end and gamma is 1, and that state 3 appears twice, so first-visit and every-visit Monte Carlo disagree about how many samples it gives.
```

One episode is three arrays of length $T$: the states $s_0, \dots, s_{T-1}$, the actions, and the rewards $r_0, \dots, r_{T-1}$. The terminal state isn't stored, because its value is zero. The episode in Figure 1 has states $(3, 2, 3, 4, 5)$ and rewards $(0, 0, 0, 0, 1)$.

## Compute every return in one pass

Each step of the episode needs its own return. Adding up the future rewards separately for each step costs $T^2$ additions. The return has a recursive shape that makes it cost $T$:

$$
G_t = r_t + \gamma\, G_{t+1}, \qquad G_T = 0
$$

Read it as: the return from step $t$ is the reward you get right away, plus the discounted return from the next step. Start at the end, where nothing is left, and walk backward. For the episode in Figure 1, $G_4 = 1$, then $G_3 = 0 + 1 = 1$, and so on, so every step has return 1. In code:

```python
running = 0.0
for t in reversed(range(T)):
    running = rewards[t] + gamma * running
    G[t] = running
```

This is the function `discounted_returns` from the [returns chapter](/lessons/002-returns-and-discounting), and it is given to you in the exercise.

## First visit or every visit

A state can appear more than once in an episode. In Figure 1, state 3 appears at $t = 0$ and $t = 2$. Which of its returns do you average?

Take a smaller episode to see both choices. It visits states $A, B, A$ and ends, with rewards $(1, 0, 2)$ and $\gamma = 1$. The backward pass gives $G_2 = 2$, $G_1 = 0 + 2 = 2$, and $G_0 = 1 + 2 = 3$.

- *First-visit* Monte Carlo uses only the return after the first visit to a state in each episode. It gives $V(A) = G_0 = 3$ and $V(B) = G_1 = 2$.
- *Every-visit* Monte Carlo uses the return after each visit. It averages both visits to $A$, so $V(A) = (3 + 2) / 2 = 2.5$, and still $V(B) = 2$.

Both converge to the true value as you add episodes. They differ in what they are made of. The two returns for $A$ in one episode overlap, because the return at $t = 0$ contains the reward at $t = 2$, so they are not independent samples. First-visit takes at most one return per episode, so its samples are independent, and the estimate is unbiased: averaged over many repeats of the experiment, it hits the true value, however few episodes each repeat uses. Every-visit uses more samples per episode, but it is biased for a finite number of episodes, and the bias fades as episodes accumulate.

To average across episodes, keep a running total and a count for each state, and a `set` of the states already seen in the current episode:

```python
seen = set()
for t, s in enumerate(states):
    if first_visit and s in seen:
        continue
    seen.add(s)
    total[s] += G[t]
    count[s] += 1
```

After all episodes, $V(s)$ is `total[s] / count[s]`, and a state that no episode visited stays at 0. One mistake is common here: creating the `seen` set once for all episodes, when it must start empty in each episode.

> [!KEY]
> Monte Carlo prediction needs no model. It averages the returns that actually followed each state. First-visit takes one return per episode and is unbiased, and every-visit takes all of them and is biased only for a finite number of episodes.

## Check it on the random walk

With 2,000 episodes of the random walk, first-visit Monte Carlo gives

| State | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Estimate | 0.170 | 0.339 | 0.506 | 0.681 | 0.837 |
| Truth $k/6$ | 0.167 | 0.333 | 0.500 | 0.667 | 0.833 |

```figure
<svg viewBox="0 0 570 270" role="img" aria-label="First-visit Monte Carlo estimates of the five random walk states after 2000 episodes, each close to the true value k over 6">
<line x1="70.0" y1="240.0" x2="540.0" y2="240.0" class="stroke-grid"/>
<text x="62.0" y="244.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="190.0" x2="540.0" y2="190.0" class="stroke-grid"/>
<text x="62.0" y="194.0" class="fig-small" text-anchor="end">0.25</text>
<line x1="70.0" y1="140.0" x2="540.0" y2="140.0" class="stroke-grid"/>
<text x="62.0" y="144.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="70.0" y1="90.0" x2="540.0" y2="90.0" class="stroke-grid"/>
<text x="62.0" y="94.0" class="fig-small" text-anchor="end">0.75</text>
<line x1="70.0" y1="40.0" x2="540.0" y2="40.0" class="stroke-grid"/>
<text x="62.0" y="44.0" class="fig-small" text-anchor="end">1</text>
<rect x="88.0" y="206.7" width="44.0" height="33.3" rx="2" class="fill-soft-a"/>
<circle cx="110.0" cy="206.0" r="5" class="fill-b"/>
<text x="110.0" y="258.0" class="fig-small" text-anchor="middle">state 1</text>
<text x="138.0" y="210.0" class="fig-small" text-anchor="start">0.170</text>
<rect x="178.0" y="173.3" width="44.0" height="66.7" rx="2" class="fill-soft-a"/>
<circle cx="200.0" cy="172.2" r="5" class="fill-b"/>
<text x="200.0" y="258.0" class="fig-small" text-anchor="middle">state 2</text>
<text x="228.0" y="176.2" class="fig-small" text-anchor="start">0.339</text>
<rect x="268.0" y="140.0" width="44.0" height="100.0" rx="2" class="fill-soft-a"/>
<circle cx="290.0" cy="138.7" r="5" class="fill-b"/>
<text x="290.0" y="258.0" class="fig-small" text-anchor="middle">state 3</text>
<text x="318.0" y="142.7" class="fig-small" text-anchor="start">0.506</text>
<rect x="358.0" y="106.7" width="44.0" height="133.3" rx="2" class="fill-soft-a"/>
<circle cx="380.0" cy="103.8" r="5" class="fill-b"/>
<text x="380.0" y="258.0" class="fig-small" text-anchor="middle">state 4</text>
<text x="408.0" y="107.8" class="fig-small" text-anchor="start">0.681</text>
<rect x="448.0" y="73.3" width="44.0" height="166.7" rx="2" class="fill-soft-a"/>
<circle cx="470.0" cy="72.6" r="5" class="fill-b"/>
<text x="470.0" y="258.0" class="fig-small" text-anchor="middle">state 5</text>
<text x="498.0" y="76.6" class="fig-small" text-anchor="start">0.837</text>
<rect x="70.0" y="46.0" width="14.0" height="10.0" rx="2" class="fill-soft-a"/>
<text x="90.0" y="55.0" class="fig-small" text-anchor="start">true value k/6</text>
<circle cx="77.0" cy="72.0" r="5" class="fill-b"/>
<text x="90.0" y="76.0" class="fig-small" text-anchor="start">estimate after 2,000 episodes</text>
</svg>
Figure 2. First-visit Monte Carlo after 2,000 episodes, from one seeded run. Notice that every estimate sits within 0.015 of the true value k/6, about one standard error, and that no model of the walk was used.
```

The estimates are close but not exact, and the size of the miss is predictable. Every return here is 0 or 1, so a first-visit average over $n$ visits is a fraction of successes with standard error $\sqrt{V(1 - V)/n}$. The middle state has 2,000 visits, so its standard error is $\sqrt{0.25/2000} = 0.011$, and its estimate 0.506 is half a standard error from 0.5.

The $\sqrt n$ in that formula is the cost of this method. To halve the error you need four times as many episodes.

```figure
<svg viewBox="0 0 570 300" role="img" aria-label="Mean error of the Monte Carlo estimate of the middle state against the number of episodes on log-log axes, falling with slope minus one half">
<line x1="80.0" y1="240.0" x2="540.0" y2="240.0" class="stroke-grid"/>
<text x="72.0" y="244.0" class="fig-small" text-anchor="end">0.005</text>
<line x1="80.0" y1="202.4" x2="540.0" y2="202.4" class="stroke-grid"/>
<text x="72.0" y="206.4" class="fig-small" text-anchor="end">0.01</text>
<line x1="80.0" y1="164.8" x2="540.0" y2="164.8" class="stroke-grid"/>
<text x="72.0" y="168.8" class="fig-small" text-anchor="end">0.02</text>
<line x1="80.0" y1="115.2" x2="540.0" y2="115.2" class="stroke-grid"/>
<text x="72.0" y="119.2" class="fig-small" text-anchor="end">0.05</text>
<line x1="80.0" y1="77.6" x2="540.0" y2="77.6" class="stroke-grid"/>
<text x="72.0" y="81.6" class="fig-small" text-anchor="end">0.1</text>
<line x1="80.0" y1="40.0" x2="540.0" y2="40.0" class="stroke-grid"/>
<text x="72.0" y="44.0" class="fig-small" text-anchor="end">0.2</text>
<line x1="80.0" y1="240.0" x2="80.0" y2="244.0" class="stroke-muted"/>
<text x="80.0" y="258.0" class="fig-small" text-anchor="middle">10</text>
<line x1="256.9" y1="240.0" x2="256.9" y2="244.0" class="stroke-muted"/>
<text x="256.9" y="258.0" class="fig-small" text-anchor="middle">100</text>
<line x1="433.8" y1="240.0" x2="433.8" y2="244.0" class="stroke-muted"/>
<text x="433.8" y="258.0" class="fig-small" text-anchor="middle">1000</text>
<text x="310.0" y="276.0" text-anchor="middle">episodes averaged, n</text>
<polyline points="80.0,65.0 540.0,227.3" class="stroke-muted dashed"/>
<polyline points="80.0,66.6 164.4,92.6 256.9,119.7 341.3,151.7 433.8,186.5 518.3,217.3" class="stroke-b"/>
<circle cx="80.0" cy="66.6" r="4" class="fill-b"/>
<circle cx="164.4" cy="92.6" r="4" class="fill-b"/>
<circle cx="256.9" cy="119.7" r="4" class="fill-b"/>
<circle cx="341.3" cy="151.7" r="4" class="fill-b"/>
<circle cx="433.8" cy="186.5" r="4" class="fill-b"/>
<circle cx="518.3" cy="217.3" r="4" class="fill-b"/>
<text x="14.0" y="24.0" class="fig-small" text-anchor="start">mean error of V(3), 40 runs</text>
<line x1="330.0" y1="52.0" x2="360.0" y2="52.0" class="stroke-b"/>
<text x="366.0" y="56.0" class="fig-small" text-anchor="start">measured</text>
<line x1="330.0" y1="70.0" x2="360.0" y2="70.0" class="stroke-muted dashed"/>
<text x="366.0" y="74.0" class="fig-small" text-anchor="start">0.399 / sqrt(n)</text>
</svg>
Figure 3. Mean absolute error of the estimate of V(3) over 40 seeded runs of n episodes. Notice the slope of minus one half: ten times more episodes cut the error by only a factor of about three.
```

> [!WARNING]
> Monte Carlo waits for the episode to end before it learns anything. A policy that walks into a wall forever never produces a return. Cap the episode length, or use the methods of the [temporal-difference chapter](/lessons/016-temporal-difference-learning), which learn from partial episodes.

> [!RECAP]
> - $V^\pi(s)$ is an expected return, so the average of the returns that followed $s$ estimates it.
> - One backward pass, `running = r[t] + gamma * running`, gives every return in the episode.
> - First-visit averages one return per state per episode, and every-visit averages all of them.
> - Keep `total[s]` and `count[s]`, and divide with `np.divide(..., where=count > 0)`.
> - The error shrinks like $1/\sqrt n$, so four times the episodes halves it.

## Build it step by step

The checks test `mc_prediction`. `random_walk`, `generate_episode`, and `discounted_returns` are given.

1. **Set up the sums.** Make `total` and `count`, both of shape `(S,)`, filled with zeros. Loop over the episodes, and call `discounted_returns(rewards, gamma)` once per episode.
2. **Add each counted visit.** Start a fresh `seen = set()` in each episode. For each `(t, s)`, skip the visit when `first_visit` is true and `s` is already in `seen`. Otherwise add `G[t]` to `total[s]` and 1 to `count[s]`. Common mistake: resetting `seen` once for all episodes, so a state counts only in the first episode that visits it.
3. **Divide.** Return `np.divide(total, count, out=np.zeros(n_states), where=count > 0)`, so that unvisited states stay 0. Common mistake: a plain `total / count`, which gives `nan` for unvisited states.

## Cost and scaling

Time grows with the total number of steps you play, $O(T)$ per episode, and memory grows with the number of states, one total and one count each. The hidden cost is variance. A return sums $T$ random rewards, so long and noisy episodes need more of them to average out, and the error falls only as $1/\sqrt n$.

## Where this breaks

**Episodes that never end.** Monte Carlo needs a complete return. Cap the episode length, or switch to a method that learns from partial episodes.

**Noisy returns on long episodes.** Every return carries the noise of every later step. People average more episodes, or move to bootstrapped methods, which you meet in the temporal-difference chapter.

**Rarely visited states.** A state that the policy seldom reaches gets few samples and a poor estimate. In the random walk, the end states have fewer visits (1,189 and 1,210) than the middle one (2,000).

## Further reading

- [Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd edition (2018), Chapter 5](http://incompleteideas.net/book/the-book-2nd.html) is the source of the random walk and of Monte Carlo prediction.
- [Singh and Sutton, "Reinforcement learning with replacing eligibility traces" (1996)](https://doi.org/10.1007/BF00114726) analyzes first-visit against every-visit Monte Carlo, including the bias of every-visit.
- [Precup, Sutton, and Singh, "Eligibility traces for off-policy policy evaluation" (2000)](https://dl.acm.org/doi/10.5555/645529.658134) treats the off-policy version that [a later chapter](/lessons/015-off-policy-importance-sampling) builds.

## The exercise

You build the Monte Carlo estimator. Write `mc_prediction(episodes, n_states, gamma, first_visit=True)`, which takes a list of `(states, actions, rewards)` tuples and returns `V` of shape `(S,)`. The starter gives `random_walk`, `generate_episode`, and `discounted_returns`.

The checks verify the $A, B, A$ example by hand: 3 and 2 for first-visit, 2.5 and 2 forevery-visit. They average across three episodes with an unvisited state, and check the discounted case with $\gamma = 0.9$. On the random walk with 2,000 episodes, they check that first-visit estimates land within four standard errors of $k/6$ and that every-visit estimates land near it too. Last, they check that the error of the middle state falls between 10 and 1,000 episodes.
