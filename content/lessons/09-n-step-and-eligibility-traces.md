---
title: n-step returns and eligibility traces
slug: 09-n-step-and-eligibility-traces
part: "Part 2 — Learning from experience"
summary: Bootstrap after n steps instead of one, average every n into the lambda-return, and compute it backward with eligibility traces.
minutes: 60
gpu: false
objectives:
  - Compute n-step returns and the lambda-return from one episode, including their Monte Carlo and one-step limits.
  - Show that an n-step return shrinks the error of the values it bootstraps from by a factor of gamma to the n.
  - Implement TD(lambda) with accumulating traces and verify that, offline, it equals the lambda-return update exactly.
  - Explain why a middle value of n or lambda learns fastest, and why large step sizes make traces unstable.
prereqs:
  - 06-monte-carlo-methods
  - 07-temporal-difference-learning
exercise: 09-n-step-and-eligibility-traces
---

# n-step returns and eligibility traces

TD(0) and Monte Carlo sit at two ends of a scale. TD(0) looks one step ahead
and then trusts its own estimate. Monte Carlo looks all the way to the end
and trusts nothing but the rewards. TD(0) has low variance but learns slowly,
because news travels back one state per update. Monte Carlo sends news all
the way back at once, but every return carries the noise of the whole
episode.

This lesson fills in the scale. An *n-step return* adds up $n$ real rewards
and then bootstraps. The *$\lambda$-return* averages every $n$ at once, with
weights that fade geometrically. And *eligibility traces* compute that
average without looking into the future, by keeping a fading memory of
the states visited in the last few steps and crediting each TD error back to them.

It's like tracing a bad restaurant meal back to its causes. The last dish you
ate gets most of the blame, but the dishes before it get some too, less the
further back they were. On the 19-state random walk, a middle setting cuts
the error after 10 episodes by more than half compared with either end.

## What you already know

- The TD(0) target is $r_t + \gamma V(s_{t+1})$, and the TD error is
  $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.
- The Monte Carlo target is the full return $G_t$.
- If $V$ doesn't change during an episode, $G_t - V(s_t) = \sum_{k \ge t}
  \gamma^{k-t} \delta_k$, as you saw in
  [Temporal-difference learning](07-temporal-difference-learning).
- The random walk moves left or right with probability one half until it
  exits at one end.
- $P_\pi$ is the state-to-state transition matrix under a policy, from
  [Markov decision processes](04-markov-decision-processes).

## Look n steps ahead

The n-step return takes $n$ real rewards and then bootstraps from the value
of the state it reaches:

$$
G_{t:t+n} = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})
$$

Read it as: the discounted sum of the next $n$ rewards, plus the discounted
estimate of everything after. If the episode ends first, when $t + n \ge T$,
there's nothing to bootstrap from, and $G_{t:t+n}$ is the full return $G_t$.
With $n = 1$ it's the TD(0) target. With $n \ge T$ it's the Monte Carlo
return. In numpy, sum the rewards in the window and add the tail if there is
one:

```python
end = min(t + n, T)
G = sum(gamma ** (k - t) * rewards[k] for k in range(t, end))
if t + n < T:
    G += gamma ** n * values[t + n]
```

Here's a worked example. An episode visits $s_0, s_1, s_2$ and then ends,
with rewards $r = (1, 0, 2)$, estimates $V = (0.5, 1.0, 0.8)$, and $\gamma =
0.9$. From $t = 0$:

- $G_{0:1} = 1 + 0.9 \times 1.0 = 1.9$.
- $G_{0:2} = 1 + 0 + 0.81 \times 0.8 = 1.648$.
- $G_{0:3} = 1 + 0 + 0.81 \times 2 = 2.62$, the full return.

### Why bootstrapping later helps

The n-step return has an *error-reduction property*. Its expected value is
closer to the true $V^\pi$ than $V$ is, by a factor of $\gamma^n$ in the
worst state:

$$
\max_s \Bigl| \mathbb{E}_\pi[G_{t:t+n} \mid s_t = s] - V^\pi(s) \Bigr|
\le \gamma^n \max_s \bigl| V(s) - V^\pi(s) \bigr|
$$

Read it as: whatever is wrong with your estimates, an n-step target is wrong
by at most $\gamma^n$ times as much, on average. The proof idea is short. The
expected n-step return is $\sum_{k<n} \gamma^k P_\pi^k r_\pi + \gamma^n
P_\pi^n V$, and the true value satisfies the same equation with $V^\pi$ in
place of $V$. Subtract them and the error is $\gamma^n P_\pi^n (V - V^\pi)$.
Each row of $P_\pi^n$ is a probability distribution, so it averages the error
without growing it. The exercise builds the expected return with matrix
powers and checks this bound on a random MDP:

```python
out = sum(gamma**k * np.linalg.matrix_power(P_pi, k) @ r_pi for k in range(n))
out += gamma**n * np.linalg.matrix_power(P_pi, n) @ V
```

### n-step TD

n-step TD moves $V(s_t)$ toward $G_{t:t+n}$. Online, the update for $s_t$
waits until $r_{t+n-1}$ arrives, $n - 1$ steps later. For stored episodes,
updating $s_0, s_1, \dots$ in order, each with the current $V$, gives exactly
the same result:

```python
for tau in range(T):
    G = ...                               # G_{tau:tau+n} from the current V
    V[states[tau]] += alpha * (G - V[states[tau]])
```

### Pick n

Sutton and Barto's 19-state random walk exits on the left with reward $-1$
and on the right with reward $+1$. Every other step pays 0 and $\gamma = 1$,
so the true value of state $k$ is $(k - 10) / 10$. Start every estimate at 0
and run 10 episodes with $\alpha = 0.2$. Averaged over 30 seeded runs, the RMS
error across the 19 states is:

| n | 1 | 4 | 64 |
|---|---|---|---|
| RMS error after 10 episodes | 0.41 | 0.18 | 0.51 |

One-step TD moves news back too slowly. With $n = 64$ the target is almost
always the full return, and it's too noisy for $\alpha = 0.2$. A middle $n$
beats both ends.

> [!KEY]
> An n-step return adds $n$ real rewards before it bootstraps. Its expected
> error is at most $\gamma^n$ times the error of the values it uses, and a
> middle $n$ usually learns fastest.

## Average all the n-step returns

Choosing one $n$ is awkward. The $\lambda$-return uses all of them at once, in
a weighted average whose weights shrink by a factor $\lambda$ per extra step.
For an episode that ends at $T$:

$$
G_t^\lambda = (1 - \lambda) \sum_{n=1}^{T-t-1} \lambda^{n-1} G_{t:t+n} + \lambda^{T-t-1} G_t
$$

Read it as: the one-step return gets weight $1 - \lambda$, the two-step
return $(1 - \lambda)\lambda$, and so on. Every n-step return that would run
past the end equals $G_t$, so all their weight lumps onto the full return as
$\lambda^{T-t-1}$. The weights sum to one. With $\lambda = 0$ it's the TD(0)
target. With $\lambda = 1$ it's the Monte Carlo return.

The same quantity has a backward recursion that costs $O(T)$ instead of
$O(T^2)$:

$$
G_t^\lambda = r_t + \gamma\bigl[(1 - \lambda) V(s_{t+1}) + \lambda\, G_{t+1}^\lambda\bigr]
$$

Read it as: take the reward, then blend two views of what comes next. With
weight $1 - \lambda$, trust the estimate $V(s_{t+1})$. With weight
$\lambda$, keep going with the next step's $\lambda$-return. At the last step
the next state is terminal, so both are 0 and $G_{T-1}^\lambda = r_{T-1}$.

```python
next_value, next_G = 0.0, 0.0
for t in reversed(range(T)):
    G[t] = rewards[t] + gamma * ((1 - lam) * next_value + lam * next_G)
    next_value, next_G = values[t], G[t]
```

For the worked episode with $\lambda = 0.5$:

$$
G_0^\lambda = 0.5 \times 1.9 + 0.5 \times 0.5 \times 1.648 + 0.5^2 \times 2.62 = 0.95 + 0.412 + 0.655 = 2.017
$$

> [!WARNING]
> Leave out the tail weight $\lambda^{T-t-1}$ on the full return, and the
> weights no longer sum to one. The target then shrinks toward zero on short
> episodes, and the checks at $\lambda = 1$ catch it.

## Compute it backward with traces

The $\lambda$-return looks into the future, so an update that uses it must
wait for the episode to end. Eligibility traces turn it around. They look
back instead of forward, and they update after every step.

An *eligibility trace* $z$ holds one number per state, starting at zero. It
records how often each state was visited, and how many steps ago. At each step it
fades by $\gamma\lambda$ and the current state gets a bump of 1. Then the TD
error $\delta_t$ is credited to every state in proportion to its trace:

$$
z_t = \gamma\lambda\, z_{t-1} + \mathbf{e}_{s_t}, \qquad
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad
V \leftarrow V + \alpha\, \delta_t\, z_t
$$

Read it as follows. $\mathbf{e}_{s_t}$ is a vector with a 1 at $s_t$ and 0
elsewhere. The state you're in is fully eligible for credit. A state you
left $k$ steps ago keeps a share $(\gamma\lambda)^k$. A surprise, positive or
negative, updates every recent state at once. This is *TD($\lambda$)*, and
this trace, which adds 1 on every visit, is an *accumulating* trace.

```python
z *= gamma * lam
z[s] += 1.0
delta = r + gamma * V[s2] * (not terminated) - V[s]
V += alpha * delta * z
```

For the worked episode, $\gamma\lambda = 0.45$. The TD errors are $\delta_0 =
1 + 0.9 - 0.5 = 1.4$, $\delta_1 = 0 + 0.72 - 1.0 = -0.28$, and $\delta_2 = 2 -
0.8 = 1.2$. The trace on $s_0$ is 1, then 0.45, then 0.2025. So with $V$ held
fixed and $\alpha = 1$, the total credit to $s_0$ is $1.4 - 0.45 \times 0.28 +
0.2025 \times 1.2 = 1.517$. That's exactly $G_0^\lambda - V(s_0) = 2.017 -
0.5$.

### The offline equivalence

That match isn't luck. Hold $V$ fixed for a whole episode, add up the trace
updates, and apply them at the end. The total is exactly the update toward
the $\lambda$-return:

$$
\sum_{t=0}^{T-1} \alpha\, \delta_t\, z_t = \sum_{t=0}^{T-1} \alpha\bigl(G_t^\lambda - V(s_t)\bigr)\, \mathbf{e}_{s_t}
$$

Read it as: the backward view, with traces, and the forward view, with
$\lambda$-returns, move every state by the same amount. Here's the proof
idea. With $V$ fixed, the $\lambda$-return error telescopes into TD errors,
$G_t^\lambda - V(s_t) = \sum_{k \ge t} (\gamma\lambda)^{k-t} \delta_k$, the
same identity you met for Monte Carlo, with $\gamma\lambda$ in place of
$\gamma$. Swap the order of the double sum, and the coefficient on
$\delta_k$ is $\sum_{t \le k} (\gamma\lambda)^{k-t} \mathbf{e}_{s_t}$, which
is the trace $z_k$. On a random 12-step episode over 5 states, the two sides
agree to about $10^{-15}$.

> [!KEY]
> Offline, TD($\lambda$) with accumulating traces makes exactly the
> $\lambda$-return update. Traces compute a forward-looking target by
> crediting each TD error backward.

### Online, and its hazard

Online TD($\lambda$) updates $V$ at every step, so later TD errors use values
that earlier steps already changed. It no longer matches the
$\lambda$-return exactly, but with a small $\alpha$ it's close. On the
19-state walk with $\alpha = 0.2$, online TD($\lambda$) reaches an RMS error
of about 0.16 at $\lambda = 0.8$, against 0.41 at $\lambda = 0$.

Large $\lambda$ with a large $\alpha$ is dangerous. On the walk, a state can
be visited ten or more times in one episode, and an accumulating trace can
build well above 1. Then $\alpha z$ exceeds 1, each update overshoots, and
the values blow up. At $\lambda = 1$ and $\alpha = 0.2$, the estimates on the
walk grow to astronomically large numbers. *Replacing traces* set the
current state's trace to 1 instead of adding 1, which caps it. *True online
TD($\lambda$)* (van Seijen and Sutton, 2014) uses a different trace that
matches an online $\lambda$-return exactly for linear value functions.

> [!RECAP]
> - `n_step_return` sums rewards in the window $[t, \min(t + n, T))$ and adds
>   $\gamma^n V(s_{t+n})$ only if $t + n < T$.
> - `lambda_return` runs the backward recursion
>   `G[t] = r[t] + gamma * ((1 - lam) * V_next + lam * G_next)`.
> - The expected n-step return is a sum of matrix powers, and its error is
>   $\gamma^n P_\pi^n (V - V^\pi)$.
> - A trace decays by `gamma * lam`, then adds 1 at the current state. Reset
>   it each episode.
> - Offline, add `alpha * delta * z` into a separate total with $V$ fixed.
>   Online, add it straight into `V`.

## Build it step by step

The checks test the functions in this order. `random_walk` and
`generate_episode` are already written. Episodes are `(states, rewards)`
pairs, with `states` one longer than `rewards` because it ends at the
terminal state. The return functions take `values`, the estimates along the
episode, without the terminal state.

1. **`n_step_return(rewards, values, t, n, gamma)`.** Returns a float. Common
   mistake: adding the bootstrap term when $t + n = T$. There's no state
   after the terminal one.
2. **`lambda_return(rewards, values, gamma, lam)`.** Returns a `(T,)` array.
   Use the backward recursion. Common mistake: using `values[t]` instead of
   the next state's value inside step $t$.
3. **`expected_n_step_return(P_pi, r_pi, V, n, gamma)`.** Returns `(S,)`.
   Keep a running matrix power, `step = step @ P_pi`, instead of calling
   `matrix_power` in the loop.
4. **`td_lambda_offline(states, rewards, V, alpha, gamma, lam)`.** Returns the
   total increment, shape `(S,)`, and leaves `V` alone. Bootstrap from
   `V[states[t + 1]]` except on the last step. Common mistake: adding the
   bump before decaying, which multiplies the current state's credit by
   $\gamma\lambda$.
5. **`n_step_td(episodes, n_states, n, alpha, gamma, v_init=0.0)`.** Returns
   `V`. Loop `tau` over each episode in order and update with the current
   `V`. Set each episode's final state to 0 before you start.
6. **`td_lambda(episodes, n_states, alpha, gamma, lam, v_init=0.0)`.** Returns
   `V`. The same loop as `td_lambda_offline`, but add into `V` at every step.
   Common mistake: carrying the trace from one episode into the next.

## Cost and scaling

The following table compares the cost of each method per episode of length
$T$:

| Method | Time per episode | Extra memory |
|---|---|---|
| TD(0) | $O(T)$ | None |
| n-step TD | $O(nT)$, or $O(T)$ with a running sum | The last $n$ rewards |
| $\lambda$-return, backward recursion | $O(T)$ | The whole episode |
| TD($\lambda$) with a trace vector | $O(ST)$ | One trace per state |

Tabular TD($\lambda$) updates every state at every step, so it costs $S$ per
step. In practice, you keep only the states whose trace is above a small
threshold, since $(\gamma\lambda)^k$ shrinks fast. With function
approximation, the trace has one entry per parameter instead, so it costs
the same as a gradient step.

## Where this breaks

**Unstable traces.** With accumulating traces, a large $\alpha$, and $\lambda$
near 1, revisited states build large traces and the values diverge. Lower
$\alpha$, use replacing traces, or use true online TD($\lambda$).

**Tuning two knobs.** The best $\alpha$ depends on $n$ or $\lambda$. A larger
$n$ usually needs a smaller $\alpha$. Tune them together, not one at a time.

**Delayed updates.** An n-step update waits $n$ steps, and a forward-view
$\lambda$-return update waits for the episode to end. For continuing tasks
with no end, only the trace form works online.

**Off-policy data.** Traces follow the actions the agent took. To learn about
a different policy, each step's credit needs an importance ratio, and the
product of ratios brings back the variance problems of
[Monte Carlo methods](06-monte-carlo-methods). Methods such as tree backup
avoid the ratios.

## Further reading

- [Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd edition
  (2018), Chapters 7 and 12](http://incompleteideas.net/book/the-book-2nd.html).
  n-step methods, the 19-state walk, the $\lambda$-return, and traces.
- [Sutton, "Learning to predict by the methods of temporal differences"
  (1988)](https://doi.org/10.1007/BF00115009). The paper that introduced
  TD($\lambda$).
- [Singh and Sutton, "Reinforcement learning with replacing eligibility
  traces" (1996)](https://doi.org/10.1007/BF00114726). Replacing traces, and
  why they behave better than accumulating ones.
- [van Seijen and Sutton, "True online TD($\lambda$)"
  (2014)](https://proceedings.mlr.press/v32/seijen14.html). A trace that makes
  the online algorithm match an online $\lambda$-return exactly.
- [Schulman, Moritz, Levine, Jordan, and Abbeel, "High-dimensional continuous
  control using generalized advantage estimation"
  (2016)](https://arxiv.org/abs/1506.02438). The same weighting applied to
  advantages, which lesson 17 builds.

## The exercise

You build the returns, the traces, and two learners. Write `n_step_return`,
`lambda_return`, and `expected_n_step_return`. Then write `td_lambda_offline`
for the trace update with $V$ fixed, and the online learners `n_step_td` and
`td_lambda`.

The checks verify the worked values 1.9, 1.648, 2.62, and 2.017, the
$\lambda = 0$ and $\lambda = 1$ limits, and that `lambda_return` equals the
explicit weighted sum on a random episode. They check the expected n-step
return against matrix powers, and the $\gamma^n$ error bound for $n = 1$ to
6 on a random MDP. `td_lambda_offline` must equal the forward-view update to
$10^{-12}$ on random episodes, and reduce to TD(0) and Monte Carlo at the
limits. On the 19-state walk, $n = 4$ must beat $n = 1$ and $n = 64$, and
online TD($\lambda$) at $\lambda = 0.8$ must beat $\lambda = 0$.
