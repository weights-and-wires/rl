---
title: Generalised advantage estimation
slug: 17-generalized-advantage-estimation
part: "Part 4 — Policy optimisation"
summary: Blend one-step and Monte Carlo advantages with one dial, lambda, and bootstrap correctly when a time limit cuts an episode off.
minutes: 50
gpu: false
objectives:
  - Write the k-step advantage as a discounted sum of TD errors, and GAE as their exponentially weighted average.
  - Compute GAE by backward recursion over a rollout that spans several episodes.
  - Show that lambda = 0 gives the TD error and lambda = 1 gives the Monte Carlo return minus V.
  - "Treat termination and truncation differently: stop bootstrapping at a terminal state, but bootstrap from V at a time limit."
prereqs:
  - 09-n-step-and-eligibility-traces
  - 16-actor-critic
exercise: 17-generalized-advantage-estimation
---

# Generalised advantage estimation

The actor needs to know, for each action it took, how much better that action
was than usual. The previous lesson gave two answers at opposite extremes. The
Monte Carlo answer, the whole return minus a baseline, is unbiased but noisy,
because it adds up every later reward. The one-step answer, the critic's TD
error, is steady but only as right as the critic.

Generalised advantage estimation, from Schulman, Moritz, Levine, Jordan, and
Abbeel (2016), puts both answers on one dial. Turn the dial, called
$\lambda$, and you slide smoothly from the TD error to the Monte Carlo
advantage. It costs one backward pass over the rollout. That's why most policy
gradient methods in wide use, including PPO, compute their advantages this
way.

The formula is short, and the bugs are in the bookkeeping. A rollout of a few
hundred steps usually crosses several episode boundaries. Some episodes end
because the world ended them. Others end because a clock ran out. Those two
cases need opposite treatment, and mixing them up quietly biases every
advantage near the end of an episode. This lesson gets both right.

## What you already know

- The TD error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is a one-step
  estimate of the advantage. With the true $V^\pi$ it averages to exactly
  $A^\pi(s_t, a_t)$.
- The return minus a baseline, $G_t - V(s_t)$, is an unbiased estimate of the
  advantage whatever $V$ is, but its variance grows with the episode length.
- The $n$-step return and the $\lambda$-return from
  *n-step returns and eligibility traces* blend short and long lookaheads.
- `rlenvs` steps return `(s2, r, terminated, truncated)`. `terminated` means
  the episode reached a terminal state. `truncated` means a time limit
  stopped it.

## Build k-step advantages from TD errors

Between the two extremes sit the $k$-step estimates. Take $k$ real rewards,
then let the critic guess the rest:

$$
\hat A^{(k)}_t = r_t + \gamma r_{t+1} + \cdots + \gamma^{k-1} r_{t+k-1} +
\gamma^k V(s_{t+k}) - V(s_t)
$$

Read it as: $k$ steps of what actually happened, then the critic's value of
where you ended up, minus the critic's value of where you started. With
$k = 1$ it's the TD error. As $k$ grows, the critic matters less and the
sampled rewards matter more.

Here's the identity that makes GAE cheap. Every $k$-step advantage is a sum
of TD errors:

$$
\hat A^{(k)}_t = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}
$$

Read it as: add the next $k$ TD errors, each discounted by how far ahead it
is. Write out the sum and the middle values cancel in pairs, a *telescoping
sum*. The $+\gamma V(s_{t+1})$ inside $\delta_t$ cancels the
$-\gamma V(s_{t+1})$ inside $\gamma \delta_{t+1}$, and so on down the line.
Only the first $-V(s_t)$ and the last $+\gamma^k V(s_{t+k})$ survive.

## Average them with one dial

Instead of picking one $k$, GAE averages all of them. It gives $\hat A^{(1)}$
weight $1 - \lambda$, $\hat A^{(2)}$ weight $(1 - \lambda)\lambda$, and so on,
with weights shrinking geometrically and summing to one. Substitute the
TD-error form and collect terms, and the average collapses:

$$
\hat A^{\text{GAE}}_t = (1 - \lambda) \sum_{k \ge 1} \lambda^{k-1} \hat
A^{(k)}_t = \sum_{l \ge 0} (\gamma \lambda)^l\, \delta_{t+l}
$$

Read it as: a discounted sum of future TD errors, where the discount per step
is $\gamma\lambda$ instead of $\gamma$. A TD error $l$ steps ahead counts
$(\gamma\lambda)^l$ as much as the one at step $t$.

When the episode ends $n$ steps after $t$, there are only $n$ distinct
estimates. The last one, $\hat A^{(n)}_t$, is the full return minus
$V(s_t)$. It collects all the weight that would have gone to longer
estimators, $\lambda^{n-1}$ in total. The sum of TD errors stops at the end
of the episode in the same way, so the two forms still agree.

The sum has a backward recursion, the same trick as for returns:

$$
\hat A_t = \delta_t + \gamma \lambda\, \hat A_{t+1}
$$

Read it as: this step's advantage is its own TD error plus a shrunken copy of
the next step's advantage. Start from zero past the end and walk backwards.

```python
adv = np.zeros_like(delta)
running = np.zeros(delta.shape[1:])          # works for (T,) and (T, N)
for t in reversed(range(T)):
    running = delta[t] + gamma * lam * (1.0 - ended[t]) * running
    adv[t] = running
```

The factor `1.0 - ended[t]` resets the running sum wherever an episode ended
at step $t$. Without it, advantages from the next episode leak backwards into
this one.

### The two ends of the dial

Set $\lambda = 0$ and only the $l = 0$ term survives: $\hat A_t = \delta_t$,
the one-step TD error. Set $\lambda = 1$ and the sum is the telescoping sum of
all TD errors to the end of the episode:

$$
\hat A^{(\lambda = 1)}_t = \sum_{l \ge 0} \gamma^l r_{t+l} - V(s_t)
$$

Read it as: the Monte Carlo return minus the baseline, the REINFORCE with
baseline estimate from the previous lesson.

> [!KEY]
> GAE is $\sum_l (\gamma\lambda)^l \delta_{t+l}$, computed backwards as
> $\hat A_t = \delta_t + \gamma\lambda \hat A_{t+1}$. $\lambda = 0$ gives the
> TD error, and $\lambda = 1$ gives the return minus $V$.

### What $\lambda$ trades, and what $\gamma$ trades

The two discount-like numbers do different jobs. $\lambda$ is purely an
estimation choice. If the critic is exact, $V = V^\pi$, then every future TD
error averages to zero given the present, so every $\lambda$ gives an
unbiased estimate of $A^\pi(s_t, a_t)$. Only the variance changes. A smaller
$\lambda$ adds bias only when $V$ is wrong.

$\gamma$ is different. It defines the objective. A smaller $\gamma$ also cuts
variance, by ignoring distant rewards, but it changes which policy is best,
even with a perfect critic. Schulman et al. discuss both as variance-reduction
knobs, and in practice people treat both as hyperparameters. Typical values
are $\gamma = 0.99$ and $\lambda = 0.95$.

The value targets that go with GAE are $\hat A_t + V(s_t)$. That's exactly
the $\lambda$-return from Part 2, so the critic learns TD($\lambda$)-style
targets for free.

### Worked example

Take three steps with rewards $r = [1, 0, 2]$, critic values
$V = [0.5, 1.0, 1.5]$, $\gamma = 0.9$, and $\lambda = 0.8$, so
$\gamma\lambda = 0.72$. The episode terminates after the third step. The TD
errors are:

- $\delta_0 = 1 + 0.9 \times 1.0 - 0.5 = 1.4$
- $\delta_1 = 0 + 0.9 \times 1.5 - 1.0 = 0.35$
- $\delta_2 = 2 + 0 - 1.5 = 0.5$, with no bootstrap after the terminal step

Walking backwards:

- $\hat A_2 = 0.5$
- $\hat A_1 = 0.35 + 0.72 \times 0.5 = 0.71$
- $\hat A_0 = 1.4 + 0.72 \times 0.71 = 1.9112$

With $\lambda = 1$, the same recursion with factor $0.9$ gives
$[2.12, 0.8, 0.5]$. The returns are $G = [1 + 0 + 0.81 \times 2, 0 + 0.9
\times 2, 2] = [2.62, 1.8, 2]$, and $G - V = [2.12, 0.8, 0.5]$, as promised.

## Separate termination from truncation

An episode can end in two ways, and they mean opposite things for the
future.

A *termination* is part of the world. The pole fell, the agent reached the
goal, or it fell off the cliff. Nothing comes after it. The value of a
terminal state is zero by definition, so the TD error drops the bootstrap:
$\delta_t = r_t - V(s_t)$.

A *truncation* is part of your training setup. A time limit of 500 steps, or
the end of a rollout buffer, stopped the episode. The world would have gone
on. If you drop the bootstrap here, you tell the critic that the state just
before the time limit is worth only its last reward. That's false, and the
error spreads backwards through every advantage in the episode. Pardo,
Tavakoli, Levdik, and Kormushev (2018) showed that bootstrapping from the
value of the final state fixes this when the time limit is only a training
convenience.

So the two flags do different jobs in the code:

- Only `terminated` removes the bootstrap from $\delta_t$.
- Both flags stop the backward recursion, because the next array entry
  belongs to a different episode.

```python
delta = rewards + gamma * (1.0 - terminated) * next_values - values
ended = terminated | truncated
```

### Worked example

Rerun the same three steps, but say a time limit cut the episode off after
the third step, at a state the critic values at 1.0. The only change is
$\delta_2 = 2 + 0.9 \times 1.0 - 1.5 = 1.4$ instead of 0.5. The following
table shows how that one number moves every advantage:

| Episode end | $\delta_2$ | $\hat A_2$ | $\hat A_1$ | $\hat A_0$ |
|---|---|---|---|---|
| Terminated | 0.5 | 0.5 | 0.71 | 1.9112 |
| Truncated, $V(s_3) = 1.0$ | 1.4 | 1.4 | 1.358 | 2.37776 |

Treating this truncation as a termination would change every advantage in
the episode. Here, it would lower each of them.

> [!WARNING]
> Environments that reset themselves hand you the *next episode's* first
> observation after a step that ends an episode. Its value belongs to a
> different episode. For a truncated step, bootstrap from the value of the
> final observation the step reached, which the environment reports
> separately.

The function `next_state_values` in the exercise builds the `next_values`
array this way. Shift the values left by one, put the value of the
observation after the rollout at the end, and overwrite every step that ended
an episode with its final observation's value.

> [!KEY]
> Termination means there's no future, so don't bootstrap. Truncation means
> the future was cut off, so bootstrap from $V$ of the state you reached. Both
> stop the advantage recursion.

> [!RECAP]
> - $\delta_t = r_t + \gamma (1 - \text{terminated}_t) V(s_{t+1}) - V(s_t)$.
> - $\hat A_t = \delta_t + \gamma\lambda (1 - \text{ended}_t) \hat A_{t+1}$,
>   walking backwards from zero.
> - $\lambda = 0$ gives $\delta_t$, and $\lambda = 1$ gives $G_t - V(s_t)$.
> - The value targets are $\hat A_t + V(s_t)$, the $\lambda$-returns.
> - After an episode ends, `next_values[t]` must be the final observation's
>   value, not the reset state's.

## Build it step by step

The checks test the functions in this order. Every array has time on axis 0.

1. **`td_errors(rewards, values, next_values, terminated, gamma)`.** Returns
   an array the same shape as `rewards`. Convert `terminated` to floats before
   subtracting it from one. Common mistake: masking with
   `terminated | truncated`, which drops the bootstrap at time limits.
2. **`next_state_values(values, last_value, done, final_values)`.** Returns
   `(T,)`. Build the shifted array, then use `np.where(done, final_values,
   shifted)`. Common mistake: using `values[t + 1]` after a reset.
3. **`gae(rewards, values, next_values, terminated, truncated, gamma, lam)`.**
   Call `td_errors`, then walk backwards. Initialise the running sum with
   `np.zeros(delta.shape[1:])`, so the same code handles one environment or
   many. Common mistake: using `ended[t + 1]` instead of `ended[t]`. In this
   course's convention, the flag at index $t$ describes the step from $s_t$.
4. The next four checks test `gae` against the $\lambda$ limits, episode
   boundaries, the explicit weighted average of $k$-step advantages, and a
   `(T, N)` batch. They need nothing new.
5. **`lambda_returns(...)`.** Returns `gae(...) + values`.
6. The last check samples 20,000 short rollouts from a fixed state and action
   on a random MDP, uses the exact $V^\pi$ as the critic, and truncates each
   rollout after four steps. For $\lambda = 0$, $0.5$, and $1$, the mean
   advantage at the first step must be within four standard errors of
   $A^\pi(s, a)$. A missing bootstrap at the truncation fails it.

## Cost and scaling

GAE is one elementwise pass for the TD errors and one backward pass for the
recursion. The following table gives the costs for a rollout of $T$ steps in
$N$ environments:

| Piece | Time | Memory |
|---|---|---|
| TD errors | $O(TN)$, fully vectorised | $O(TN)$ |
| Backward recursion | $T$ sequential steps of $O(N)$ | $O(TN)$ for the output |
| Critic values | One forward pass over $T \times N$ states | $O(TN)$ |

The recursion is sequential in time, but each step is a vector operation
over environments, so it's never the bottleneck. The expensive part is
running the critic on every state. Compute all the values once, in one
batch, before the backward pass.

## Where this breaks

**Off-by-one done flags.** Codebases disagree on whether `done[t]` describes
the step into $s_t$ or the step out of it. Mixing conventions shifts every
episode boundary by one, which bootstraps across episodes. Write down the
convention and test it with a two-episode rollout, as the checks do.

**Truncation treated as termination.** This is the most common GAE bug. It
makes the critic believe states near the time limit are worth only their last
reward. In a task with positive rewards, such as CartPole, the symptom is
advantages that are systematically low near the end of every episode.

**Time limits that are part of the task.** If the agent really does have only
500 steps, then the time left matters, and the state should include it.
Pardo et al. (2018) recommend adding the remaining time to the observation in
that case.

**Normalising too early.** PPO implementations normalise advantages to zero
mean and unit variance. Do it after you compute the value targets
$\hat A_t + V(s_t)$, or the critic learns from corrupted targets.

**A poor critic with small $\lambda$.** At small $\lambda$, the advantage
leans on $V$. Early in training, $V$ is poor, and the actor follows its
errors. A larger $\lambda$ early on, or a critic warmed up first, helps.

## Further reading

- [Schulman, Moritz, Levine, Jordan, and Abbeel (2016), High-dimensional continuous control using generalized advantage estimation](https://arxiv.org/abs/1506.02438)
  introduces GAE, derives its $\lambda$ limits, and discusses $\gamma$ and
  $\lambda$ as bias-variance knobs.
- [Pardo, Tavakoli, Levdik, and Kormushev (2018), Time limits in reinforcement learning](https://arxiv.org/abs/1712.00378)
  shows why you bootstrap at time limits that are only for training, and what
  to do when the limit is part of the task.
- [Sutton and Barto (2018), Reinforcement Learning: An Introduction, chapter 12](http://incompleteideas.net/book/the-book-2nd.html)
  covers the $\lambda$-return that GAE's value targets reproduce.
- [Andrychowicz et al. (2021), What matters in on-policy reinforcement learning? A large-scale empirical study](https://arxiv.org/abs/2006.05990)
  compares advantage estimators and their settings across a large number of
  training runs.

## The exercise

You build the advantage pipeline that PPO uses in the next two lessons. Write
`td_errors` with termination masking, `next_state_values` for environments
that reset themselves, `gae` by backward recursion, and `lambda_returns` for
the value targets.

The checks confirm the TD errors and GAE on the lesson's worked example, both
terminated and truncated, and the `next_state_values` bookkeeping around
resets. They check that `lam=0` equals `td_errors` and `lam=1` equals the
Monte Carlo return minus $V$ across a two-episode rollout. They check that
advantages never cross an episode boundary, that intermediate $\lambda$
matches the explicit weighted average of $k$-step advantages, and that a
`(T, N)` batch matches column-by-column calls. Finally, with the exact $V^\pi$
as critic, the mean GAE from a fixed state and action must match
$A^\pi(s, a)$ for three values of $\lambda$ on truncated rollouts.
