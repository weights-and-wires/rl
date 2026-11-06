---
title: Generalized advantage estimation
slug: 032-generalized-advantage-estimation
part: "Part 5 — Policy gradients"
summary: Blend one-step and Monte Carlo advantages with one dial, lambda, and bootstrap correctly when a time limit cuts an episode off.
minutes: 40
gpu: false
objectives:
  - Write the k-step advantage as a discounted sum of TD errors, and GAE as their exponentially weighted average.
  - Compute GAE by backward recursion over a rollout that spans several episodes.
  - Show that lambda = 0 gives the TD error and lambda = 1 gives the Monte Carlo return minus V.
  - "Treat termination and truncation differently: stop bootstrapping at a terminal state, but bootstrap from V at a time limit."
prereqs:
  - 020-eligibility-traces
  - 031-baselines-and-actor-critic
exercise: 032-generalized-advantage-estimation
---

# Generalized advantage estimation

The actor needs to know, for each action it took, how much better that action was than usual. The previous chapter gave two answers at opposite ends. The Monte Carlo answer, the whole return minus a baseline, is correct on average but noisy, because it adds up every later reward. The one-step answer, the critic's TD error, is steady but only as right as the critic.

Which should you use? Neither end is best. Early in training the critic is poor, so you want to lean on real rewards. Late in training the critic is good, and long sums of rewards only add noise. You want one dial that slides between the two.

*Generalized advantage estimation* (GAE), from Schulman, Moritz, Levine, Jordan, and Abbeel (2016), is that dial. It costs one backward pass over the rollout, which is why PPO and most policy gradient methods in wide use compute advantages this way. The formula is short and the bugs are in the bookkeeping, because a rollout usually crosses several episode boundaries, and an episode that ends because the world ended it must be treated differently from one that a clock cut off.

## What you already know

- The TD error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is a one-step estimate of the advantage.
- The return minus a baseline, $G_t - V(s_t)$, is correct on average whatever $V$ is, but its variance grows with the episode length.
- The $n$-step return and the $\lambda$-return from [Eligibility traces and TD(lambda)](/lessons/020-eligibility-traces) blend short and long lookaheads.
- `rlenvs` steps return `(s2, r, terminated, truncated)`. `terminated` means the episode reached a terminal state. `truncated` means a time limit stopped it.

## Look k steps ahead

Between the two ends sit the $k$-step estimates. Take $k$ real rewards, then let the critic guess the rest:

$$
\hat A^{(k)}_t = r_t + \gamma r_{t+1} + \cdots + \gamma^{k-1} r_{t+k-1} + \gamma^k V(s_{t+k}) - V(s_t)
$$

Read it as: $k$ steps of what happened, then the critic's value of where you ended up, minus its value of where you started. With $k = 1$ it is the TD error. As $k$ grows, the critic matters less and the sampled rewards matter more.

Work one rollout by hand and reuse it all chapter. Three steps have rewards $r = [1, 0, 2]$, critic values $V = [0.5, 1.0, 1.5]$, and $\gamma = 0.9$. The world ends the episode after the third step. The TD errors are

- $\delta_0 = 1 + 0.9 \times 1.0 - 0.5 = 1.4$
- $\delta_1 = 0 + 0.9 \times 1.5 - 1.0 = 0.35$
- $\delta_2 = 2 + 0 - 1.5 = 0.5$, with no bootstrap after the end

and the $k$-step advantages at $t = 0$ are $\hat A^{(1)}_0 = 1.4$, $\hat A^{(2)}_0 = 1 + 0 + 0.81 \times 1.5 - 0.5 = 1.715$, and $\hat A^{(3)}_0 = 1 + 0 + 0.81 \times 2 - 0.5 = 2.12$.

Look at those numbers next to the TD errors. $1.715 = 1.4 + 0.9 \times 0.35$, and $2.12 = 1.715 + 0.81 \times 0.5$. That is no accident. Every $k$-step advantage is a sum of TD errors:

$$
\hat A^{(k)}_t = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}
$$

Read it as: add the next $k$ TD errors, each discounted by how far ahead it is. Write out the sum and the middle values cancel in pairs, which is called a *telescoping sum*: the $+\gamma V(s_{t+1})$ inside $\delta_t$ cancels the $-\gamma V(s_{t+1})$ inside $\gamma \delta_{t+1}$, and so on. Only the first $-V(s_t)$ and the last $+\gamma^k V(s_{t+k})$ survive.

## Average them with one dial

Instead of picking one $k$, average them all. Give $\hat A^{(1)}$ weight $1 - \lambda$, $\hat A^{(2)}$ weight $(1 - \lambda)\lambda$, and so on, with weights that shrink geometrically and sum to one. Substitute the TD-error form and collect terms, and the average collapses:

$$
\hat A^{\text{GAE}}_t = (1 - \lambda) \sum_{k \ge 1} \lambda^{k-1} \hat A^{(k)}_t = \sum_{l \ge 0} (\gamma \lambda)^l\, \delta_{t+l}
$$

Read it as: a discounted sum of future TD errors, where the discount per step is $\gamma\lambda$ instead of $\gamma$. A TD error $l$ steps ahead counts $(\gamma\lambda)^l$ as much as the one at step $t$. Figure 1 shows those weights for four values of $\lambda$.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Weight on the TD error l steps ahead for four values of lambda with gamma 0.9">
<line x1="70" y1="230.0" x2="560" y2="230.0" class="stroke-grid"/><text x="62" y="234.0" class="fig-small" text-anchor="end">0</text>
<line x1="70" y1="135.0" x2="560" y2="135.0" class="stroke-grid"/><text x="62" y="139.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="70" y1="40.0" x2="560" y2="40.0" class="stroke-grid"/><text x="62" y="44.0" class="fig-small" text-anchor="end">1</text>
<text x="100.0" y="250" class="fig-small" text-anchor="middle">0</text>
<text x="186.0" y="250" class="fig-small" text-anchor="middle">1</text>
<text x="272.0" y="250" class="fig-small" text-anchor="middle">2</text>
<text x="358.0" y="250" class="fig-small" text-anchor="middle">3</text>
<text x="444.0" y="250" class="fig-small" text-anchor="middle">4</text>
<text x="530.0" y="250" class="fig-small" text-anchor="middle">5</text>
<text x="315" y="272" text-anchor="middle">steps ahead, l</text><text x="70" y="24" class="fig-small">weight (γλ)^l on the TD error</text>
<polyline points="100.0,40.0 186.0,230.0 272.0,230.0 358.0,230.0 444.0,230.0 530.0,230.0" class="stroke-a" fill="none"/>
<circle cx="100.0" cy="40.0" r="3.5" class="fill-a"/>
<circle cx="186.0" cy="230.0" r="3.5" class="fill-a"/>
<circle cx="272.0" cy="230.0" r="3.5" class="fill-a"/>
<circle cx="358.0" cy="230.0" r="3.5" class="fill-a"/>
<circle cx="444.0" cy="230.0" r="3.5" class="fill-a"/>
<circle cx="530.0" cy="230.0" r="3.5" class="fill-a"/>
<line x1="440" y1="60" x2="470" y2="60" class="stroke-a"/><text x="478" y="64" class="fig-small">λ = 0</text>
<polyline points="100.0,40.0 186.0,144.5 272.0,191.5 358.0,212.7 444.0,222.2 530.0,226.5" class="stroke-c" fill="none"/>
<circle cx="100.0" cy="40.0" r="3.5" class="fill-c"/>
<circle cx="186.0" cy="144.5" r="3.5" class="fill-c"/>
<circle cx="272.0" cy="191.5" r="3.5" class="fill-c"/>
<circle cx="358.0" cy="212.7" r="3.5" class="fill-c"/>
<circle cx="444.0" cy="222.2" r="3.5" class="fill-c"/>
<circle cx="530.0" cy="226.5" r="3.5" class="fill-c"/>
<line x1="440" y1="80" x2="470" y2="80" class="stroke-c"/><text x="478" y="84" class="fig-small">λ = 0.5</text>
<polyline points="100.0,40.0 186.0,93.2 272.0,131.5 358.0,159.1 444.0,178.9 530.0,193.2" class="stroke-b" fill="none"/>
<circle cx="100.0" cy="40.0" r="3.5" class="fill-b"/>
<circle cx="186.0" cy="93.2" r="3.5" class="fill-b"/>
<circle cx="272.0" cy="131.5" r="3.5" class="fill-b"/>
<circle cx="358.0" cy="159.1" r="3.5" class="fill-b"/>
<circle cx="444.0" cy="178.9" r="3.5" class="fill-b"/>
<circle cx="530.0" cy="193.2" r="3.5" class="fill-b"/>
<line x1="440" y1="100" x2="470" y2="100" class="stroke-b"/><text x="478" y="104" class="fig-small">λ = 0.8</text>
<polyline points="100.0,40.0 186.0,59.0 272.0,76.1 358.0,91.5 444.0,105.3 530.0,117.8" class="stroke-d" fill="none"/>
<circle cx="100.0" cy="40.0" r="3.5" class="fill-d"/>
<circle cx="186.0" cy="59.0" r="3.5" class="fill-d"/>
<circle cx="272.0" cy="76.1" r="3.5" class="fill-d"/>
<circle cx="358.0" cy="91.5" r="3.5" class="fill-d"/>
<circle cx="444.0" cy="105.3" r="3.5" class="fill-d"/>
<circle cx="530.0" cy="117.8" r="3.5" class="fill-d"/>
<line x1="440" y1="120" x2="470" y2="120" class="stroke-d"/><text x="478" y="124" class="fig-small">λ = 1</text>
</svg>
Figure 1. The weight GAE puts on the TD error l steps ahead, for γ = 0.9. Notice that λ = 0 looks only at the current step, λ = 1 weights later errors by γ^l alone, and values in between fade out faster.
```

When the episode ends $n$ steps after $t$, only $n$ distinct estimates exist. The last one, the full return minus $V(s_t)$, takes all the leftover weight $\lambda^{n-1}$. In the example, with $\lambda = 0.8$, the weights are $0.2$, $0.16$, and $0.64$, and $0.2 \times 1.4 + 0.16 \times 1.715 + 0.64 \times 2.12 = 1.9112$.

The sum has a backward recursion, the same trick as for returns:

$$
\hat A_t = \delta_t + \gamma \lambda\, \hat A_{t+1}
$$

Read it as: this step's advantage is its own TD error plus a shrunken copy of the next step's advantage. Start from zero past the end and walk backward. For the example, $\hat A_2 = 0.5$, then $\hat A_1 = 0.35 + 0.72 \times 0.5 = 0.71$, then $\hat A_0 = 1.4 + 0.72 \times 0.71 = 1.9112$, which matches the weighted average.

```python
adv = np.zeros_like(delta)
running = np.zeros(delta.shape[1:])          # works for (T,) and (T, N)
for t in reversed(range(T)):
    running = delta[t] + gamma * lam * (1.0 - ended[t]) * running
    adv[t] = running
```

The factor `1.0 - ended[t]` resets the running sum wherever an episode ended at step $t$. Without it, advantages from the next episode leak backward into this one.

### The two ends of the dial

Set $\lambda = 0$ and only the $l = 0$ term survives, so $\hat A_t = \delta_t$, the one-step TD error. Set $\lambda = 1$ and the sum telescopes all the way to the end of the episode, so $\hat A_t = \sum_{l \ge 0} \gamma^l r_{t+l} - V(s_t)$, the Monte Carlo return minus the baseline. In the example, $\lambda = 1$ gives $[2.12, 0.8, 0.5]$, and the returns are $[2.62, 1.8, 2]$, so $G - V = [2.12, 0.8, 0.5]$, as promised.

> [!KEY]
> GAE is $\sum_l (\gamma\lambda)^l \delta_{t+l}$, computed backward as $\hat A_t = \delta_t + \gamma\lambda \hat A_{t+1}$. $\lambda = 0$ gives the TD error, and $\lambda = 1$ gives the return minus $V$.

The two dials do different jobs. $\lambda$ is only an estimation choice: with an exact critic, every $\lambda$ gives an estimate that is correct on average, and only the variance changes. $\gamma$ defines the objective, so lowering it changes which policy is best. Typical values are $\gamma = 0.99$ and $\lambda = 0.95$. The value targets that go with GAE are $\hat A_t + V(s_t)$, which are the $\lambda$-returns.

## Separate termination from truncation

An episode can end in two ways, and they mean opposite things for the future.

A *termination* is part of the world: the pole fell, or the agent reached the goal. Nothing comes after it, so the value of a terminal state is zero, and the TD error drops the bootstrap: $\delta_t = r_t - V(s_t)$.

A *truncation* is part of your training setup: a 500-step limit, or the end of a rollout buffer. The world would have gone on. If you drop the bootstrap here, you tell the critic that the state before the limit is worth only its last reward, which is false. The error spreads backward through every advantage in the episode (Pardo, Tavakoli, Levdik, and Kormushev, 2018).

So the two flags do different jobs in the code. Only `terminated` removes the bootstrap from $\delta_t$. Both flags stop the backward recursion, because the next array entry belongs to a different episode.

```python
delta = rewards + gamma * (1.0 - terminated) * next_values - values
ended = terminated | truncated
```

Rerun the example, but say a time limit cut the episode off after the third step, at a state the critic values at 1.0. Then $\delta_2 = 2 + 0.9 \times 1.0 - 1.5 = 1.4$ instead of 0.5, and that one number moves every advantage, as Figure 2 shows.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Advantages at three steps when the episode terminates and when it is truncated: truncation raises every advantage">
<line x1="70" y1="250" x2="570" y2="250" class="stroke-muted"/><text x="64" y="254" class="fig-small" text-anchor="end">0</text>
<line x1="70" y1="165" x2="570" y2="165" class="stroke-grid"/><text x="64" y="169" class="fig-small" text-anchor="end">1</text>
<line x1="70" y1="80" x2="570" y2="80" class="stroke-grid"/><text x="64" y="84" class="fig-small" text-anchor="end">2</text>
<rect x="120" y="87.5" width="50" height="162.5" class="fill-a"/><text x="145" y="81.5" class="fig-small" text-anchor="middle">1.9112</text>
<rect x="176" y="47.9" width="50" height="202.1" class="fill-b"/><text x="201" y="41.9" class="fig-small" text-anchor="middle">2.37776</text>
<text x="173" y="272" text-anchor="middle">advantage at t = 0</text>
<rect x="280" y="189.7" width="50" height="60.3" class="fill-a"/><text x="305" y="183.7" class="fig-small" text-anchor="middle">0.71</text>
<rect x="336" y="134.6" width="50" height="115.4" class="fill-b"/><text x="361" y="128.6" class="fig-small" text-anchor="middle">1.358</text>
<text x="333" y="272" text-anchor="middle">advantage at t = 1</text>
<rect x="440" y="207.5" width="50" height="42.5" class="fill-a"/><text x="465" y="201.5" class="fig-small" text-anchor="middle">0.5</text>
<rect x="496" y="131.0" width="50" height="119.0" class="fill-b"/><text x="521" y="125.0" class="fig-small" text-anchor="middle">1.4</text>
<text x="493" y="272" text-anchor="middle">advantage at t = 2</text>
<rect x="400" y="14" width="14" height="14" class="fill-a"/><text x="420" y="26" class="fig-small">terminated</text>
<rect x="400" y="36" width="14" height="14" class="fill-b"/><text x="420" y="48" class="fig-small">truncated, V(s3) = 1.0</text>
</svg>
Figure 2. GAE advantages for the worked rollout (γ = 0.9, λ = 0.8) when the last step terminates and when a time limit truncates it. Notice that one changed TD error at the last step lifts the advantages at every earlier step too.
```

> [!WARNING]
> Environments that reset themselves hand you the next episode's first observation after a step that ends an episode. Its value belongs to a different episode. For a truncated step, bootstrap from the value of the final observation the step reached, which the environment reports separately.

The function `next_state_values` builds the `next_values` array this way. Shift the values left by one, put the value of the observation after the rollout at the end, and overwrite every step that ended an episode with its final observation's value.

> [!KEY]
> Termination means there is no future, so do not bootstrap. Truncation means the future was cut off, so bootstrap from $V$ of the state you reached. Both stop the advantage recursion.

> [!RECAP]
> - $\delta_t = r_t + \gamma (1 - \text{terminated}_t) V(s_{t+1}) - V(s_t)$.
> - $\hat A_t = \delta_t + \gamma\lambda (1 - \text{ended}_t) \hat A_{t+1}$, walking backward from zero.
> - $\lambda = 0$ gives $\delta_t$, and $\lambda = 1$ gives $G_t - V(s_t)$.
> - The value targets are $\hat A_t + V(s_t)$.
> - After an episode ends, `next_values[t]` must be the final observation's value, not the reset state's.

## Build it step by step

The checks test the functions in this order. Every array has time on axis 0.

1. **`td_errors(rewards, values, next_values, terminated, gamma)`.** Returns an array shaped like `rewards`. Convert `terminated` to floats before subtracting it from one. The common mistake is to mask with `terminated | truncated`, which drops the bootstrap at time limits.
2. **`next_state_values(values, last_value, done, final_values)`.** Returns `(T,)`. Build the shifted array, then use `np.where(done, final_values, shifted)`. The common mistake is to use `values[t + 1]` after a reset.
3. **`gae(rewards, values, next_values, terminated, truncated, gamma, lam)`.** Call `td_errors`, then walk backward. Start the running sum at `np.zeros(delta.shape[1:])` so one piece of code handles one environment or many. The common mistake is to use `ended[t + 1]` instead of `ended[t]`: the flag at index $t$ describes the step from $s_t$.
4. **`lambda_returns(...)`.** Returns `gae(...) + values`.

## Where this breaks

**Off-by-one done flags.** Codebases disagree on whether `done[t]` describes the step into $s_t$ or the step out of it. Mixing conventions shifts every episode boundary by one and bootstraps across episodes. Write the convention down and test it on a two-episode rollout.

**Truncation treated as termination.** This is the most common GAE bug. In a task with positive rewards, such as CartPole, the symptom is advantages that are too low near the end of every episode.

**Time limits that are part of the task.** If the agent really has only 500 steps, the time left matters, and Pardo et al. (2018) recommend adding it to the observation.

**Normalizing too early.** PPO implementations normalize advantages to zero mean and unit variance. Do it after you compute the value targets $\hat A_t + V(s_t)$, or the critic learns from corrupted targets.

## Further reading

- [Schulman, Moritz, Levine, Jordan, and Abbeel (2016), High-dimensional continuous control using generalized advantage estimation](https://arxiv.org/abs/1506.02438) introduces GAE and discusses $\gamma$ and $\lambda$ as bias-variance knobs.
- [Pardo, Tavakoli, Levdik, and Kormushev (2018), Time limits in reinforcement learning](https://arxiv.org/abs/1712.00378) shows why you bootstrap at training-only time limits.
- [Sutton and Barto (2018), Reinforcement Learning: An Introduction, chapter 12](http://incompleteideas.net/book/the-book-2nd.html) covers the $\lambda$-return that GAE's value targets reproduce.

## The exercise

You build the advantage pipeline that PPO uses in the next chapters. Write `td_errors` with termination masking, `next_state_values` for environments that reset themselves, `gae` by backward recursion, and `lambda_returns` for the value targets.

The checks confirm the TD errors and GAE on the worked example, terminated and truncated, and the `next_state_values` bookkeeping around resets. They check that `lam=0` equals `td_errors` and `lam=1` equals the Monte Carlo return minus $V$ across a two-episode rollout, that advantages never cross an episode boundary, and that intermediate $\lambda$ matches the explicit weighted average of $k$-step advantages. They check that a `(T, N)` batch matches column-by-column calls. Finally, with the exact $V^\pi$ as critic, the mean GAE from a fixed state and action must match $A^\pi(s, a)$ for three values of $\lambda$ on truncated rollouts.
