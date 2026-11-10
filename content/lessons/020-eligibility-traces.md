---
title: Eligibility traces and TD(lambda)
slug: 020-eligibility-traces
part: "Part 3 — Learning from experience"
summary: Average every n-step return into the lambda-return, then compute it step by step with a fading memory of recent states.
minutes: 40
gpu: false
objectives:
  - Compute the lambda-return from one episode, including its TD(0) and Monte Carlo limits.
  - Implement TD(lambda) with accumulating traces.
  - Verify that, offline, TD(lambda) makes exactly the lambda-return update.
  - Explain why a middle lambda learns fastest, and why a large lambda with a large step size is unstable.
prereqs:
  - 019-n-step-returns
exercise: 020-eligibility-traces
---

# Eligibility traces and TD(lambda)

You get food poisoning after a long dinner. Which dish was to blame? The last one you ate is the most likely suspect, but the one before it is a candidate too, and so is the one before that, with less suspicion the further back you go.

The n-step return forces a hard choice about this. It takes $n$ real rewards and then trusts a guess, and you have to pick one $n$ for the whole problem. Two questions follow. Instead of picking, can you average every $n$ at once? And can you do it without waiting for the episode to end, which the n-step return has to do?

This chapter answers yes to both. The *$\lambda$-return* averages all the n-step returns with weights that fade geometrically. *Eligibility traces* compute the same update by looking backward: every TD error is credited to the recent states, in proportion to how recently each was visited. You will check by hand that the two views agree, and then build them.

## What you already know

- The n-step return $G_{t:t+n}$ takes $n$ real rewards and then bootstraps. With $n = 1$ it is the TD(0) target, and with $n \ge T$ it is the Monte Carlo return (see [n-step returns](/lessons/019-n-step-returns)).
- The TD error is $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ (see [Temporal-difference learning](/lessons/016-temporal-difference-learning)).
- If $V$ does not change during an episode, the Monte Carlo error is the discounted sum of the TD errors: $G_t - V(s_t) = \sum_{k \ge t} \gamma^{k-t} \delta_k$.
- A middle $n$ learns fastest on the 19-state random walk.
- A terminal state has value zero.

## Average every n at once

Choosing one $n$ is awkward, so use all of them. Give the one-step return weight $1 - \lambda$, the two-step return weight $(1 - \lambda)\lambda$, and so on, with each extra step shrinking the weight by a factor $\lambda$ between 0 and 1. For an episode that ends at time $T$, the n-step returns that would run past the end all equal the full return $G_t$, so their weight lumps onto it:

$$
G_t^\lambda = (1 - \lambda) \sum_{n=1}^{T-t-1} \lambda^{n-1} G_{t:t+n} + \lambda^{T-t-1} G_t
$$

Read it aloud: a weighted average of the n-step returns, where each longer return counts $\lambda$ times as much as the one before, and the full return takes whatever weight is left. The weights add up to 1. With $\lambda = 0$ it is the TD(0) target, and with $\lambda = 1$ it is the Monte Carlo return, so $\lambda$ is a dial between the same two ends as $n$.

Take the episode from the last chapter: rewards $(1, 0, 2)$, estimates $V = (0.5, 1.0, 0.8)$, and $\gamma = 0.9$. Its n-step returns from $t = 0$ were 1.9, 1.648, and 2.62. With $\lambda = 0.5$:

$$
G_0^\lambda = 0.5 \times 1.9 + 0.5 \times 0.5 \times 1.648 + 0.5^2 \times 2.62 = 0.95 + 0.412 + 0.655 = 2.017
$$

```figure
<svg viewBox="0 0 580 255" role="img" aria-label="Weights of the lambda-return with lambda 0.5 on the three n-step returns of the worked episode, which add up to 2.017">
<line x1="70.0" y1="200.0" x2="400.0" y2="200.0" class="stroke-grid"/>
<text x="62.0" y="204.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="129.2" x2="400.0" y2="129.2" class="stroke-grid"/>
<text x="62.0" y="133.2" class="fig-small" text-anchor="end">0.25</text>
<line x1="70.0" y1="58.3" x2="400.0" y2="58.3" class="stroke-grid"/>
<text x="62.0" y="62.3" class="fig-small" text-anchor="end">0.5</text>
<line x1="70.0" y1="200.0" x2="400.0" y2="200.0" class="stroke-muted"/>
<rect x="89.0" y="58.3" width="72.0" height="141.7" class="fill-b"/>
<text x="125.0" y="52.3" class="fig-small" text-anchor="middle">weight 0.5</text>
<text x="125.0" y="218.0" class="fig-small" text-anchor="middle">1-step return</text>
<text x="125.0" y="234.0" class="fig-small" text-anchor="middle">G = 1.9</text>
<rect x="199.0" y="129.2" width="72.0" height="70.8" class="fill-b"/>
<text x="235.0" y="123.2" class="fig-small" text-anchor="middle">weight 0.25</text>
<text x="235.0" y="218.0" class="fig-small" text-anchor="middle">2-step return</text>
<text x="235.0" y="234.0" class="fig-small" text-anchor="middle">G = 1.648</text>
<rect x="309.0" y="129.2" width="72.0" height="70.8" class="fill-a"/>
<text x="345.0" y="123.2" class="fig-small" text-anchor="middle">weight 0.25</text>
<text x="345.0" y="218.0" class="fig-small" text-anchor="middle">full return</text>
<text x="345.0" y="234.0" class="fig-small" text-anchor="middle">G = 2.62</text>
<text x="425.0" y="70.0" class="fig-small" text-anchor="start">0.5 × 1.9</text>
<text x="425.0" y="90.0" class="fig-small" text-anchor="start">+ 0.25 × 1.648</text>
<text x="425.0" y="110.0" class="fig-small" text-anchor="start">+ 0.25 × 2.62</text>
<line x1="425.0" y1="120.0" x2="560.0" y2="120.0" class="stroke-ink"/>
<text x="425.0" y="140.0" class="fig-bold" text-anchor="start">= 2.017</text>
<text x="425.0" y="164.0" class="fig-small" text-anchor="start">the lambda-return</text>
<text x="425.0" y="180.0" class="fig-small" text-anchor="start">from state s0</text>
</svg>
Figure 1. The λ-return with λ = 0.5 for the worked episode. Notice that the weights fall by half per step, and the full return gets the leftover weight, so the three weights add to 1.
```

Computing the sum directly costs $O(T^2)$ per episode. A backward recursion gives the same numbers in $O(T)$:

$$
G_t^\lambda = r_t + \gamma\bigl[(1 - \lambda) V(s_{t+1}) + \lambda\, G_{t+1}^\lambda\bigr]
$$

Read it aloud: take the reward, then blend two views of what comes next. With weight $1 - \lambda$, trust the estimate $V(s_{t+1})$. With weight $\lambda$, keep going with the next step's $\lambda$-return. At the last step the next state is terminal, so both are 0 and $G_{T-1}^\lambda = r_{T-1}$.

```python
next_value, next_G = 0.0, 0.0
for t in reversed(range(T)):
    G[t] = rewards[t] + gamma * ((1 - lam) * next_value + lam * next_G)
    next_value, next_G = values[t], G[t]
```

> [!WARNING]
> Leave out the tail weight $\lambda^{T-t-1}$ on the full return, and the weights no longer sum to one. The target then shrinks toward zero on short episodes. The checks at $\lambda = 1$ catch it.

## Look back instead of forward

The $\lambda$-return still looks into the future, so an update that uses it must wait for the episode to end. Eligibility traces turn the idea around. Rather than asking "what happens next?", they ask "who deserves credit for this surprise?", and answer with the states visited recently.

An *eligibility trace* $z$ holds one number per state, starting at zero. At each step it fades by a factor $\gamma\lambda$, and the current state gets a bump of 1. Then the TD error $\delta_t$ is credited to every state in proportion to its trace:

$$
z_t = \gamma\lambda\, z_{t-1} + \mathbf{e}_{s_t}, \qquad
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad
V \leftarrow V + \alpha\, \delta_t\, z_t
$$

Read it aloud. $\mathbf{e}_{s_t}$ is a vector with a 1 at $s_t$ and 0 elsewhere, so the state you are in is fully eligible for credit. A state you left $k$ steps ago keeps a share $(\gamma\lambda)^k$. A surprise, positive or negative, updates every recent state at once. This is *TD($\lambda$)*, and a trace that adds 1 on every visit is an *accumulating* trace.

```python
z *= gamma * lam
z[s] += 1.0
delta = r + gamma * V[s2] * (not terminated) - V[s]
V += alpha * delta * z
```

Check it on the worked episode. Here $\gamma\lambda = 0.45$, and the TD errors are $\delta_0 = 1 + 0.9 \times 1.0 - 0.5 = 1.4$, $\delta_1 = 0 + 0.9 \times 0.8 - 1.0 = -0.28$, and $\delta_2 = 2 - 0.8 = 1.2$. State $s_0$ is visited only at step 0, so its trace is 1, then 0.45, then 0.2025.

```figure
<svg viewBox="0 0 580 225" role="img" aria-label="The eligibility trace of the first state fading over three steps, and the credit from each TD error adding to the lambda-return error">
<text x="20.0" y="24.0" class="fig-bold" text-anchor="start">State s0 is visited at step 0 only. Its trace fades by γλ = 0.45 each step.</text>
<text x="60.0" y="60.0" class="fig-bold" text-anchor="start">step 0</text>
<text x="60.0" y="80.0" class="fig-small" text-anchor="start">δ0 = 1.40</text>
<rect x="60.0" y="92.0" width="90.0" height="16.0" class="fill-sunk"/>
<rect x="60.0" y="92.0" width="90.0" height="16.0" class="fill-soft-b"/>
<text x="156.0" y="105.0" class="fig-small" text-anchor="start">z = 1</text>
<text x="60.0" y="130.0" class="fig-small" text-anchor="start">credit to s0: 1 × 1.4</text>
<text x="60.0" y="148.0" class="fig-small fig-bold" text-anchor="start">= +1.4000</text>
<text x="230.0" y="60.0" class="fig-bold" text-anchor="start">step 1</text>
<text x="230.0" y="80.0" class="fig-small" text-anchor="start">δ1 = −0.28</text>
<rect x="230.0" y="92.0" width="90.0" height="16.0" class="fill-sunk"/>
<rect x="230.0" y="92.0" width="40.5" height="16.0" class="fill-soft-b"/>
<text x="326.0" y="105.0" class="fig-small" text-anchor="start">z = 0.45</text>
<text x="230.0" y="130.0" class="fig-small" text-anchor="start">credit to s0: 0.45 × -0.28</text>
<text x="230.0" y="148.0" class="fig-small fig-bold" text-anchor="start">= -0.1260</text>
<text x="400.0" y="60.0" class="fig-bold" text-anchor="start">step 2</text>
<text x="400.0" y="80.0" class="fig-small" text-anchor="start">δ2 = 1.20</text>
<rect x="400.0" y="92.0" width="90.0" height="16.0" class="fill-sunk"/>
<rect x="400.0" y="92.0" width="18.2" height="16.0" class="fill-soft-b"/>
<text x="496.0" y="105.0" class="fig-small" text-anchor="start">z = 0.2025</text>
<text x="400.0" y="130.0" class="fig-small" text-anchor="start">credit to s0: 0.2025 × 1.2</text>
<text x="400.0" y="148.0" class="fig-small fig-bold" text-anchor="start">= +0.2430</text>
<line x1="60.0" y1="164.0" x2="560.0" y2="164.0" class="stroke-grid"/>
<text x="60.0" y="188.0" class="fig-bold" text-anchor="start">Total credit to s0 = 1.517</text>
<text x="60.0" y="208.0" class="fig-bold" text-anchor="start">The λ-return error: G0(λ) − V(s0) = 2.017 − 0.5 = 1.517</text>
</svg>
Figure 2. Backward view for the worked episode with α = 1 and V held fixed. Notice that the three pieces of credit add up to the same 1.517 as the forward view.
```

With $V$ held fixed and $\alpha = 1$, the total credit to $s_0$ is $1.4 - 0.45 \times 0.28 + 0.2025 \times 1.2 = 1.517$. That is exactly $G_0^\lambda - V(s_0) = 2.017 - 0.5$. The backward view, with no look into the future, produced the forward view's number.

## Why the two views agree

That match is not luck. Hold $V$ fixed for a whole episode, add up the trace updates, and apply them at the end. The total is exactly the update toward the $\lambda$-return:

$$
\sum_{t=0}^{T-1} \alpha\, \delta_t\, z_t = \sum_{t=0}^{T-1} \alpha\bigl(G_t^\lambda - V(s_t)\bigr)\, \mathbf{e}_{s_t}
$$

Read it aloud: the backward view, with traces, and the forward view, with $\lambda$-returns, move every state by the same amount. The proof idea takes three steps. With $V$ fixed, the $\lambda$-return error telescopes into TD errors, $G_t^\lambda - V(s_t) = \sum_{k \ge t} (\gamma\lambda)^{k-t} \delta_k$, which is the identity from Monte Carlo with $\gamma\lambda$ in place of $\gamma$. Swap the order of the double sum. The coefficient on $\delta_k$ is then $\sum_{t \le k} (\gamma\lambda)^{k-t} \mathbf{e}_{s_t}$, which is the trace $z_k$. On a random 12-step episode over 5 states, the two sides agree to about $10^{-15}$.

> [!KEY]
> Offline, TD($\lambda$) with accumulating traces makes exactly the $\lambda$-return update. Traces let you compute a forward-looking target after every step, by crediting each TD error backward to the states that led to it. The restaurant works the same way: the blame fades with each dish you go back.

## Run it online

Online TD($\lambda$) updates $V$ at every step, so later TD errors use values that earlier steps already changed. It no longer matches the $\lambda$-return exactly, but with a small $\alpha$ it comes close. On the 19-state walk with $\alpha = 0.2$, the error depends on $\lambda$ like this:

```figure
<svg viewBox="0 0 580 285" role="img" aria-label="RMS error of online TD(lambda) after ten episodes on the 19-state walk for lambda from 0 to 0.95, lowest at a middle lambda">
<line x1="70.0" y1="220.0" x2="540.0" y2="220.0" class="stroke-grid"/>
<text x="62.0" y="224.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="182.0" x2="540.0" y2="182.0" class="stroke-grid"/>
<text x="62.0" y="186.0" class="fig-small" text-anchor="end">0.1</text>
<line x1="70.0" y1="144.0" x2="540.0" y2="144.0" class="stroke-grid"/>
<text x="62.0" y="148.0" class="fig-small" text-anchor="end">0.2</text>
<line x1="70.0" y1="106.0" x2="540.0" y2="106.0" class="stroke-grid"/>
<text x="62.0" y="110.0" class="fig-small" text-anchor="end">0.3</text>
<line x1="70.0" y1="68.0" x2="540.0" y2="68.0" class="stroke-grid"/>
<text x="62.0" y="72.0" class="fig-small" text-anchor="end">0.4</text>
<line x1="70.0" y1="30.0" x2="540.0" y2="30.0" class="stroke-grid"/>
<text x="62.0" y="34.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="70.0" y1="220.0" x2="540.0" y2="220.0" class="stroke-muted"/>
<line x1="70.0" y1="220.0" x2="70.0" y2="225.0" class="stroke-muted"/>
<text x="70.0" y="240.0" class="fig-small" text-anchor="middle">0</text>
<line x1="168.9" y1="220.0" x2="168.9" y2="225.0" class="stroke-muted"/>
<text x="168.9" y="240.0" class="fig-small" text-anchor="middle">0.2</text>
<line x1="267.9" y1="220.0" x2="267.9" y2="225.0" class="stroke-muted"/>
<text x="267.9" y="240.0" class="fig-small" text-anchor="middle">0.4</text>
<line x1="366.8" y1="220.0" x2="366.8" y2="225.0" class="stroke-muted"/>
<text x="366.8" y="240.0" class="fig-small" text-anchor="middle">0.6</text>
<line x1="465.8" y1="220.0" x2="465.8" y2="225.0" class="stroke-muted"/>
<text x="465.8" y="240.0" class="fig-small" text-anchor="middle">0.8</text>
<line x1="515.3" y1="220.0" x2="515.3" y2="225.0" class="stroke-muted"/>
<text x="515.3" y="240.0" class="fig-small" text-anchor="middle">0.9</text>
<text x="305.0" y="262.0" text-anchor="middle">λ</text>
<text x="62.0" y="18.0" text-anchor="end">RMS error</text>
<polyline points="70.0,63.0 168.9,75.0 267.9,92.8 366.8,121.0 465.8,158.6 515.3,142.1 540.0,92.8" class="stroke-b" fill="none"/>
<circle cx="70.0" cy="63.0" r="5" class="fill-b"/>
<text x="70.0" y="53.0" class="fig-small" text-anchor="middle">0.41</text>
<circle cx="168.9" cy="75.0" r="5" class="fill-b"/>
<text x="168.9" y="65.0" class="fig-small" text-anchor="middle">0.38</text>
<circle cx="267.9" cy="92.8" r="5" class="fill-b"/>
<text x="267.9" y="82.8" class="fig-small" text-anchor="middle">0.33</text>
<circle cx="366.8" cy="121.0" r="5" class="fill-b"/>
<text x="366.8" y="111.0" class="fig-small" text-anchor="middle">0.26</text>
<circle cx="465.8" cy="158.6" r="5" class="fill-b"/>
<text x="465.8" y="148.6" class="fig-small" text-anchor="middle">0.16</text>
<circle cx="515.3" cy="142.1" r="5" class="fill-b"/>
<text x="515.3" y="132.1" class="fig-small" text-anchor="middle">0.21</text>
<circle cx="540.0" cy="92.8" r="5" class="fill-b"/>
<text x="540.0" y="82.8" class="fig-small" text-anchor="middle">0.33</text>
</svg>
Figure 3. RMS error after 10 episodes with α = 0.2, averaged over 30 runs. Notice that the error falls from λ = 0 to a minimum near 0.8, then rises again. At λ = 1 the error explodes past 1e20 and is not plotted.
```

Online TD($\lambda$) reaches an RMS error of about 0.16 at $\lambda = 0.8$, against 0.41 at $\lambda = 0$, and it matches TD(0) exactly at $\lambda = 0$. The U shape has the same cause as for $n$: small $\lambda$ passes news back slowly, and large $\lambda$ carries too much noise.

Large $\lambda$ with a large $\alpha$ is dangerous. On the walk, a state can be visited ten or more times in one episode, and an accumulating trace then builds well above 1. Then $\alpha z$ exceeds 1, each update overshoots the target, and the values blow up. *Replacing traces* set the current state's trace to 1 instead of adding 1, which caps it. *True online TD($\lambda$)* (van Seijen and Sutton, 2014) uses a different trace that matches an online $\lambda$-return exactly for linear value functions.

> [!RECAP]
> - `lambda_return` runs the backward recursion `G[t] = r[t] + gamma * ((1 - lam) * V_next + lam * G_next)`.
> - A trace decays by `gamma * lam`, then adds 1 at the current state. Reset it each episode.
> - Offline, add `alpha * delta * z` into a separate total with $V$ fixed. The total equals the $\lambda$-return update.
> - Online, add `alpha * delta * z` straight into `V`.
> - $\lambda = 0$ is TD(0), and $\lambda = 1$ is Monte Carlo. A middle $\lambda$ learns fastest.

## Build it step by step

The checks test the functions in this order. `random_walk` and `generate_episode` are already written. Episodes are `(states, rewards)` pairs, with `states` one longer than `rewards` because it ends at the terminal state. `lambda_return` takes `values`, the estimates along the episode, without the terminal state.

1. **`lambda_return(rewards, values, gamma, lam)`.** Returns a `(T,)` array. Use the backward recursion. The common mistake is using `values[t]` in place of the next state's value inside step $t$.
2. **`td_lambda_offline(states, rewards, V, alpha, gamma, lam)`.** Returns the total increment, shape `(S,)`, and leaves `V` alone. Bootstrap from `V[states[t + 1]]` except on the last step. The common mistake is adding the bump before decaying, which multiplies the current state's credit by $\gamma\lambda$.
3. **`td_lambda(episodes, n_states, alpha, gamma, lam, v_init=0.0)`.** Returns `V`. The same loop as `td_lambda_offline`, but add into `V` at every step. The common mistake is carrying the trace from one episode into the next.

## Cost and scaling

The following table compares the cost of each method per episode of length $T$:

| Method | Time per episode | Extra memory |
|---|---|---|
| TD(0) | $O(T)$ | None |
| $\lambda$-return, backward recursion | $O(T)$ | The whole episode |
| TD($\lambda$) with a trace vector | $O(ST)$ | One trace per state |

Tabular TD($\lambda$) updates every state at every step, so it costs $S$ per step. In practice you keep only the states whose trace is above a small threshold, because $(\gamma\lambda)^k$ shrinks fast. With function approximation, the trace has one entry per parameter, so it costs the same as a gradient step.

## Where this breaks

- **Unstable traces.** With accumulating traces, a large $\alpha$, and $\lambda$ near 1, revisited states build large traces and the values diverge. Lower $\alpha$, use replacing traces, or use true online TD($\lambda$).
- **Tuning two knobs.** The best $\alpha$ depends on $\lambda$. Tune them together, not one at a time.
- **Delayed updates.** A forward-view $\lambda$-return update waits for the episode to end. For continuing tasks with no end, only the trace form works online.
- **Off-policy data.** Traces follow the actions the agent took. To learn about a different policy, each step's credit needs an importance ratio, and the product of ratios brings back the variance problems of [importance sampling](/lessons/015-off-policy-importance-sampling). Methods such as tree backup avoid the ratios.

## Further reading

- [Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd edition (2018), Chapter 12](http://incompleteideas.net/book/the-book-2nd.html). The $\lambda$-return, traces, and the offline equivalence.
- [Sutton, "Learning to predict by the methods of temporal differences" (1988)](https://doi.org/10.1007/BF00115009). The paper that introduced TD($\lambda$).
- [Singh and Sutton, "Reinforcement learning with replacing eligibility traces" (1996)](https://doi.org/10.1007/BF00114726). Replacing traces, and why they behave better than accumulating ones.
- [van Seijen and Sutton, "True online TD($\lambda$)" (2014)](https://proceedings.mlr.press/v32/seijen14.html). A trace that makes the online algorithm match an online $\lambda$-return exactly.
- [Schulman, Moritz, Levine, Jordan, and Abbeel, "High-dimensional continuous control using generalized advantage estimation" (2016)](https://arxiv.org/abs/1506.02438). The same weighting applied to advantages, which [Generalized advantage estimation](/lessons/034-generalized-advantage-estimation) builds.

## The exercise

You write `lambda_return`, `td_lambda_offline`, and `td_lambda`. The checks verify the worked value 2.017, the $\lambda = 0$ and $\lambda = 1$ limits, and that `lambda_return` equals the explicit weighted sum of n-step returns on a random episode. `td_lambda_offline` must equal the forward-view update to $10^{-12}$ on random episodes, reduce to TD(0) and Monte Carlo at the limits, and leave `V` unchanged. Online `td_lambda` must match plain TD(0) at $\lambda = 0$, match a hand-written trace loop at $\lambda = 0.8$, and beat $\lambda = 0$ on the 19-state walk.
