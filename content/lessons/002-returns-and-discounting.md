---
title: Returns and discounting
slug: 002-returns-and-discounting
part: "Part 1 — Learning from reward"
summary: Turn a stream of rewards into one score with a discount, compute every return in one backward pass, and estimate a policy's value by averaging.
minutes: 35
gpu: false
objectives:
  - Explain why adding up rewards fails in a task that never ends, and how a discount fixes it.
  - Compute the return from every step of an episode by hand and with one backward loop.
  - Define the value of a state as an expected return, and estimate it by averaging episodes.
  - Report a Monte Carlo estimate with its standard error, and say how fast the error shrinks.
prereqs:
  - 001-learning-by-trial-and-error
exercise: 002-returns-and-discounting
---

# Returns and discounting

In the room from the last chapter, one agent walks to the goal in four steps and another wanders for thirty-five. Both finish, and both collect $-1$ per step, so the first earns $-4$ in total and the second $-35$. Adding up the rewards seems to answer the question.

Suppose the task never ends. A robot earns one point every step, forever. Its total is infinite, and so is the total of a clumsier robot that earns a tenth of a point per step. Adding up cannot tell them apart, yet nobody would call them equally good. An agent that learns from reward needs a single score for "everything that follows from here" that stays finite and that prefers the better robot.

This chapter builds that score, called the return. You compute it by hand, then with one backward loop, and then use it to measure how good a policy is by averaging over many episodes.

## What you already know

- The agent-environment loop produces a log of states, actions, and rewards, one row per step.
- A policy is an array `pi` whose row `pi[s]` holds the probabilities of each action.
- An expected value is the long-run average of a random quantity.
- The mean of $n$ samples has a standard error of about the sample standard deviation divided by $\sqrt{n}$.

## Why adding up rewards breaks

Try the obvious score first: the plain sum of the rewards from step $t$ onward. In an episode that ends, the sum is finite and works. The trouble starts when the episode can be arbitrarily long. The next table sums two robots over longer and longer runs.

| Steps | Robot A, 1 per step | Robot B, 0.1 per step |
|---|---|---|
| 10 | 10 | 1 |
| 100 | 100 | 10 |
| 1000 | 1000 | 100 |

Both sums grow without limit. Robot A is plainly better, but "infinity" and "infinity" cannot be compared. The sum has a second flaw: it counts a reward a thousand steps away exactly as much as a reward at this step, even though the far future is harder to predict and a reward in hand is worth more than one promised.

## Count later rewards for less

The fix is to shrink each reward by a fixed factor for every step it lies ahead. The factor is the *discount factor* $\gamma$ (gamma), a number from 0 to 1. A reward one step away counts $\gamma$ times as much as a reward at this step, a reward two steps away counts $\gamma^2$ times as much, and a reward $k$ steps away counts $\gamma^k$ times as much. Figure 1 shows those weights.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Bar chart of the weight gamma to the power k on a reward k steps ahead, for gamma 0.5 and gamma 0.9, k from 0 to 10">
<line x1="50" y1="210" x2="580" y2="210" class="stroke-muted"/>
<line x1="50" y1="210.0" x2="580" y2="210.0" class="stroke-grid"/>
<text x="42" y="214.0" text-anchor="end" class="fig-small">0</text>
<line x1="50" y1="120.0" x2="580" y2="120.0" class="stroke-grid"/>
<text x="42" y="124.0" text-anchor="end" class="fig-small">0.5</text>
<line x1="50" y1="30.0" x2="580" y2="30.0" class="stroke-grid"/>
<text x="42" y="34.0" text-anchor="end" class="fig-small">1</text>
<rect x="62" y="30.0" width="14" height="180.0" class="fill-a"/>
<rect x="77" y="30.0" width="14" height="180.0" class="fill-b"/>
<text x="77" y="228" text-anchor="middle" class="fig-small">0</text>
<rect x="108" y="120.0" width="14" height="90.0" class="fill-a"/>
<rect x="123" y="48.0" width="14" height="162.0" class="fill-b"/>
<text x="123" y="228" text-anchor="middle" class="fig-small">1</text>
<rect x="154" y="165.0" width="14" height="45.0" class="fill-a"/>
<rect x="169" y="64.2" width="14" height="145.8" class="fill-b"/>
<text x="169" y="228" text-anchor="middle" class="fig-small">2</text>
<rect x="200" y="187.5" width="14" height="22.5" class="fill-a"/>
<rect x="215" y="78.8" width="14" height="131.2" class="fill-b"/>
<text x="215" y="228" text-anchor="middle" class="fig-small">3</text>
<rect x="246" y="198.8" width="14" height="11.2" class="fill-a"/>
<rect x="261" y="91.9" width="14" height="118.1" class="fill-b"/>
<text x="261" y="228" text-anchor="middle" class="fig-small">4</text>
<rect x="292" y="204.4" width="14" height="5.6" class="fill-a"/>
<rect x="307" y="103.7" width="14" height="106.3" class="fill-b"/>
<text x="307" y="228" text-anchor="middle" class="fig-small">5</text>
<rect x="338" y="207.2" width="14" height="2.8" class="fill-a"/>
<rect x="353" y="114.3" width="14" height="95.7" class="fill-b"/>
<text x="353" y="228" text-anchor="middle" class="fig-small">6</text>
<rect x="384" y="208.6" width="14" height="1.4" class="fill-a"/>
<rect x="399" y="123.9" width="14" height="86.1" class="fill-b"/>
<text x="399" y="228" text-anchor="middle" class="fig-small">7</text>
<rect x="430" y="209.3" width="14" height="0.7" class="fill-a"/>
<rect x="445" y="132.5" width="14" height="77.5" class="fill-b"/>
<text x="445" y="228" text-anchor="middle" class="fig-small">8</text>
<rect x="476" y="209.6" width="14" height="0.4" class="fill-a"/>
<rect x="491" y="140.3" width="14" height="69.7" class="fill-b"/>
<text x="491" y="228" text-anchor="middle" class="fig-small">9</text>
<rect x="522" y="209.8" width="14" height="0.2" class="fill-a"/>
<rect x="537" y="147.2" width="14" height="62.8" class="fill-b"/>
<text x="537" y="228" text-anchor="middle" class="fig-small">10</text>
<text x="315.0" y="250" text-anchor="middle" class="fig-small">steps ahead, k</text>
<rect x="400" y="30" width="12" height="12" class="fill-a"/><text x="418" y="41" class="fig-small">γ = 0.5</text>
<rect x="400" y="50" width="12" height="12" class="fill-b"/><text x="418" y="61" class="fig-small">γ = 0.9 (0.9^10 = 0.35)</text>
</svg>
Figure 1. The weight gamma^k on a reward k steps ahead. With gamma = 0.5 the weight is almost gone after five steps. With gamma = 0.9 a reward ten steps away still counts 35 percent.
```

The two ends of the range show what $\gamma$ means. With $\gamma = 0$ only the next reward counts, so the agent is short-sighted. With $\gamma = 1$ every reward counts equally, which is the plain sum. Between them, $1 / (1 - \gamma)$ is a rough planning horizon: about 10 steps for $\gamma = 0.9$ and about 100 for $\gamma = 0.99$.

Discounting fixes the infinity. A reward of 1 on every step is worth $1 + \gamma + \gamma^2 + \cdots = 1/(1-\gamma)$, which is 10 at $\gamma = 0.9$. Robot B's tenth-of-a-point stream is worth 1. Both numbers are finite, and robot A is the better one, as it should be.

> [!KEY]
> A discount makes the score finite in a task that never ends, and it says how far ahead the agent should care. The price is that $\gamma$ changes what you are measuring, so you choose it as part of the problem.

## The return

The discounted total from step $t$ onward is the *return*, written $G_t$:

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots + \gamma^{T-1-t} r_{T-1}
$$

Read it as: this step's reward at full weight, the next reward at weight $\gamma$, the one after at $\gamma^2$, and so on to the end of the episode. In numpy, the weights are one line:

```python
weights = gamma ** np.arange(n)      # 1, gamma, gamma^2, ...
```

Work one example by hand. Take rewards $r_0 = 1$, $r_1 = 0$, and $r_2 = 2$, then termination, with $\gamma = 0.9$. The return from step 0 is $1 + 0.9 \times 0 + 0.81 \times 2 = 2.62$. The return from step 1 is $0 + 0.9 \times 2 = 1.8$, and from step 2 it is $2$.

Computing each of those sums separately repeats work. The return from step 0 contains the return from step 1 inside it, and that contains the return from step 2.

## Every return in one backward pass

Pull the first reward out of the sum, and what remains is $\gamma$ times the next return:

$$
G_t = r_t + \gamma G_{t+1}, \qquad G_T = 0
$$

Read it as: the return from this step is this reward plus the discounted return from the next step. After the last step there are no rewards, so the return there is 0. That lets you start at the end, where $G$ is easy, and work back. Figure 2 does the example this way.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="The backward pass for rewards 1, 0, 2 with gamma 0.9: G at step 2 is 2, at step 1 is 1.8, at step 0 is 2.62">
<defs><marker id="c002-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<rect x="40" y="40" width="110" height="110" rx="8" class="fill-soft-a stroke-a"/>
<text x="95" y="64" text-anchor="middle" class="fig-bold">step 0</text>
<text x="95" y="92" text-anchor="middle">r = 1</text>
<text x="95" y="124" text-anchor="middle" class="fig-bold">G = 2.62</text>
<rect x="190" y="40" width="110" height="110" rx="8" class="fill-soft-a stroke-a"/>
<text x="245" y="64" text-anchor="middle" class="fig-bold">step 1</text>
<text x="245" y="92" text-anchor="middle">r = 0</text>
<text x="245" y="124" text-anchor="middle" class="fig-bold">G = 1.8</text>
<rect x="340" y="40" width="110" height="110" rx="8" class="fill-soft-a stroke-a"/>
<text x="395" y="64" text-anchor="middle" class="fig-bold">step 2</text>
<text x="395" y="92" text-anchor="middle">r = 2</text>
<text x="395" y="124" text-anchor="middle" class="fig-bold">G = 2</text>
<rect x="490" y="40" width="80" height="110" rx="8" class="fill-sunk stroke-muted"/>
<text x="530" y="88" text-anchor="middle">episode</text><text x="530" y="106" text-anchor="middle">ends</text><text x="530" y="128" text-anchor="middle" class="fig-bold">G = 0</text>
<line x1="488" y1="190" x2="452" y2="190" class="stroke-b" stroke-width="2" marker-end="url(#c002-arrow)"/>
<line x1="338" y1="190" x2="302" y2="190" class="stroke-b" stroke-width="2" marker-end="url(#c002-arrow)"/>
<line x1="188" y1="190" x2="152" y2="190" class="stroke-b" stroke-width="2" marker-end="url(#c002-arrow)"/>
<text x="470" y="215" text-anchor="middle" class="fig-small">0.9 × 0 + 2</text>
<text x="320" y="215" text-anchor="middle" class="fig-small">0.9 × 2 + 0</text>
<text x="170" y="215" text-anchor="middle" class="fig-small">0.9 × 1.8 + 1</text>
<text x="300" y="20" text-anchor="middle" class="fig-small">work from the last step back to the first</text>
</svg>
Figure 2. The backward pass for rewards 1, 0, 2 at gamma = 0.9. Each return takes the one to its right, discounts it, and adds the reward in its own box.
```

The table repeats the arithmetic.

| Step $t$ | Reward $r_t$ | $G_t = r_t + 0.9\, G_{t+1}$ |
|---|---|---|
| 2 | 2 | $2 + 0.9 \times 0 = 2$ |
| 1 | 0 | $0 + 0.9 \times 2 = 1.8$ |
| 0 | 1 | $1 + 0.9 \times 1.8 = 2.62$ |

These agree with the direct sums. The backward form is also cheaper: summing from scratch at every step costs time proportional to $T^2$, and the loop costs time proportional to $T$.

```python
running = 0.0
for t in reversed(range(len(rewards))):
    running = rewards[t] + gamma * running
    returns[t] = running
```

In the room with $\gamma = 0.9$, the four-step route has return $-(1 + 0.9 + 0.81 + 0.729) = -3.439$ from the start.

## From one return to a value

A single return depends on luck: which actions the policy happened to pick, and where the environment sent the agent. The random agent's first five episodes from the start had lengths 35, 30, 16, 49, and 148, and returns of $-9.75$, $-9.58$, $-8.15$, $-9.94$, and $-10.00$. They differ, so no single one is the answer to "how good is this policy from here?"

Notice the discount at work. A 148-step wander scores only slightly worse than a 35-step one, because when each step costs 1, no return can fall below $-1/(1-\gamma) = -10$. At $\gamma = 0.9$ the agent barely looks past ten steps.

The fix is to average the luck away. The *value* of a state $s$ under a policy $\pi$ is the expected return when you start in $s$ and follow $\pi$:

$$
V^\pi(s) = \mathbb{E}_\pi\left[\, G_t \mid s_t = s \,\right]
$$

Read it as: over all the episodes that could unfold from $s$ when the agent acts by $\pi$, the average return. Almost every method in this course estimates values and then uses them to act better.

The simplest estimate follows from the definition. Run $n$ episodes from the start, compute each episode's return $G_0$, and average them. This is a *Monte Carlo* estimate, named after the casino, because it works by random sampling:

$$
\hat{V} = \frac{1}{n} \sum_{i=1}^{n} G_0^{(i)}, \qquad
\text{SE} = \frac{\text{sd}\left(G_0^{(1)}, \dots, G_0^{(n)}\right)}{\sqrt{n}}
$$

Read it as: the estimate is the mean of the $n$ episode returns, and its *standard error* SE, a measure of how far the mean is likely to sit from the true value, is their sample standard deviation divided by $\sqrt{n}$. The true value lies within two standard errors of the estimate about 95 percent of the time.

```python
g0 = np.array([discounted_returns(run_episode(mdp, pi, rng)[2], gamma)[0]
               for _ in range(n)])
estimate, se = g0.mean(), g0.std(ddof=1) / np.sqrt(n)
```

For the random policy in the room, the exact value at the start is $V^\pi(\text{start}) = -8.551$, which a later chapter, [The Bellman equation](/lessons/009-the-bellman-equation), computes without sampling. Here the exact number lets you check the estimate. Averaging more episodes gives:

| Episodes | Estimate | Standard error |
|---|---|---|
| 100 | $-8.631$ | 0.174 |
| 400 | $-8.610$ | 0.082 |
| 1600 | $-8.551$ | 0.040 |

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Monte Carlo estimates of the random policy's value with two-standard-error bars for 100, 400, and 1600 episodes, closing in on the exact value of minus 8.551">
<line x1="70" y1="220" x2="570" y2="220" class="stroke-muted"/>
<line x1="70" y1="101.5" x2="570" y2="101.5" class="stroke-c dashed" stroke-width="2"/>
<text x="76" y="93.5" class="fig-small">exact value, −8.551</text>
<text x="570" y="20" text-anchor="end" class="fig-small">bars show the estimate ± 2 standard errors</text>
<line x1="66" y1="208.1" x2="70" y2="208.1" class="stroke-muted"/>
<text x="60" y="212.1" text-anchor="end" class="fig-small">-9.0</text>
<line x1="66" y1="160.6" x2="70" y2="160.6" class="stroke-muted"/>
<text x="60" y="164.6" text-anchor="end" class="fig-small">-8.8</text>
<line x1="66" y1="113.1" x2="70" y2="113.1" class="stroke-muted"/>
<text x="60" y="117.1" text-anchor="end" class="fig-small">-8.6</text>
<line x1="66" y1="65.6" x2="70" y2="65.6" class="stroke-muted"/>
<text x="60" y="69.6" text-anchor="end" class="fig-small">-8.4</text>
<line x1="170" y1="37.8" x2="170" y2="203.1" class="stroke-b" stroke-width="3"/>
<circle cx="170" cy="120.5" r="6" class="fill-b"/>
<text x="170" y="240" text-anchor="middle">100 episodes</text>
<text x="170" y="258" text-anchor="middle" class="fig-small">-8.631 ± 0.348</text>
<line x1="320" y1="76.5" x2="320" y2="154.4" class="stroke-b" stroke-width="3"/>
<circle cx="320" cy="115.5" r="6" class="fill-b"/>
<text x="320" y="240" text-anchor="middle">400 episodes</text>
<text x="320" y="258" text-anchor="middle" class="fig-small">-8.610 ± 0.164</text>
<line x1="470" y1="82.5" x2="470" y2="120.5" class="stroke-b" stroke-width="3"/>
<circle cx="470" cy="101.5" r="6" class="fill-b"/>
<text x="470" y="240" text-anchor="middle">1600 episodes</text>
<text x="470" y="258" text-anchor="middle" class="fig-small">-8.551 ± 0.080</text>
</svg>
Figure 3. Monte Carlo estimates with two-standard-error bars against the exact value. Each estimate is within its bar, and the bar halves each time the episodes quadruple.
```

The error shrinks like $1/\sqrt{n}$, which is slow: one more digit of accuracy costs a hundred times the episodes.

> [!KEY]
> The return $G_t = r_t + \gamma G_{t+1}$ scores everything from step $t$ on, and the value $V^\pi(s)$ is its average over luck. Averaging $n$ episodes estimates the value, with an error that shrinks like $1/\sqrt{n}$.

> [!RECAP]
> - A discount $\gamma$ weights a reward $k$ steps ahead by $\gamma^k$, so the sum stays finite.
> - $G_t = r_t + \gamma G_{t+1}$ with $G_T = 0$ gives every return in one backward pass.
> - $V^\pi(s)$ is the expected return from $s$, and the mean of many $G_0$ values estimates it.
> - The standard error is the sample standard deviation (with `ddof=1`) over $\sqrt{n}$.

## Build it step by step

`run_episode` is given in full, because you wrote it in the previous chapter. The checks test the three functions in this order.

1. **`discount_weights(gamma, n)`.** Returns `gamma ** np.arange(n)`, an array of shape `(n,)`. The common mistake is an off-by-one that starts the weights at $\gamma$ instead of 1.
2. **`discounted_returns(rewards, gamma)`.** Loop backward with `running = rewards[t] + gamma * running`, and store `running` at index `t`. Return an array of shape `(T,)`. The common mistake is looping forward, which gives each step the wrong future. The checks compare against the worked example, the direct sum at four values of $\gamma$, and the recursion itself.
3. **`mc_value_estimate(mdp, pi, n_episodes, seed)`.** Make one generator from `seed` and reuse it for every episode. Return the mean of the $G_0$ values and their standard error, using `ddof=1`. The common mistake is a fresh generator with the same seed for every episode, which repeats one episode and reports a standard error of zero.

## Where this breaks

**Episodes that never end.** A Monte Carlo estimate waits for the episode to finish, so a policy that walks into a wall forever never reports. The time limit fixes that, but it leaves out the rewards after the cutoff. People set the limit far past a typical episode, or add an estimate of the remaining value.

**High variance.** Long episodes with random rewards make $G_0$ swing widely, so the standard error stays large. A smaller $\gamma$ shortens the horizon and tames the swing, at the cost of changing what you measure.

**Choosing $\gamma$.** A $\gamma$ of 0.9 makes a task that needs a hundred-step plan look hopeless, because the payoff weighs almost nothing. The discount is part of the problem you pose, and not a free tuning knob.

## The exercise

You build the three pieces of the return. Write `discount_weights(gamma, n)`, `discounted_returns(rewards, gamma)` for the backward recursion, and `mc_value_estimate(mdp, pi, n_episodes, seed)` for the Monte Carlo estimate and its standard error. The environment is the 3 by 3 room from the last chapter, built with `rlenvs.gridworld`.

The checks verify the weights and the worked returns $(2.62, 1.8, 2)$, and compare the direct sum at four discount factors and the recursion $G_t = r_t + \gamma G_{t+1}$. They check that a constant reward of 1 approaches $1/(1-\gamma)$. Then they check that `mc_value_estimate` returns exactly $-3.439$ with zero error on the shortest route, lands within four standard errors of the random policy's exact value $-8.551$, and that its standard error halves when the episode count quadruples.
