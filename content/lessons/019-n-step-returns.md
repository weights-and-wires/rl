---
title: n-step returns
slug: 019-n-step-returns
part: "Part 3 — Learning from experience"
summary: Collect n real rewards before you trust your own guess, and see why a middle n learns faster than TD(0) or Monte Carlo.
minutes: 35
gpu: false
objectives:
  - Compute n-step returns from one episode, including the one-step and Monte Carlo limits.
  - Show that an n-step return shrinks the error of the values it bootstraps from by a factor of gamma to the n.
  - Implement n-step TD on stored episodes.
  - Explain why a middle n learns fastest.
prereqs:
  - 016-temporal-difference-learning
exercise: 019-n-step-returns
---

# n-step returns

A robot walks down a corridor of 19 cells. Leaving the left end pays $-1$, leaving the right end pays $+1$, and everything between pays 0. Each step goes left or right at random. After the robot has crossed the corridor once, how fast can the news reach cell 10, in the middle?

TD(0) looks one step ahead and then trusts its own guess, so the news from the exit moves back one cell per episode. Monte Carlo looks all the way to the end, so the news arrives at every visited cell at once, but it arrives wrapped in the luck of every random step on the way. Each is a reasonable answer, and each is an end of the same dial.

This chapter turns the dial. An *n-step return* collects $n$ real rewards and only then trusts a guess. You will see what moving the dial does to the target, why a later guess is a better guess, and which setting learns fastest.

## What you already know

- The TD(0) target is $r_t + \gamma V(s_{t+1})$, and it moves $V(s_t)$ a fraction $\alpha$ toward itself (see [Temporal-difference learning](/lessons/016-temporal-difference-learning)).
- The Monte Carlo target is the full return $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$.
- TD(0) bootstraps, so it has low variance and some bias. Monte Carlo does not, so it has no bias and high variance.
- A terminal state has value zero.
- In a random walk, each step goes left or right with probability one half until the walk exits at an end.

## Turn the dial between guessing and waiting

Look at what the two ends are made of. The TD(0) target is one real reward plus a discounted guess for everything after it. The Monte Carlo return is all real rewards and no guess. The middle is the obvious thing to try: take $n$ real rewards, and then use a guess for the rest. That is the *n-step return*:

$$
G_{t:t+n} = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})
$$

Read it aloud: the discounted sum of the next $n$ rewards, plus the discounted estimate of everything after. If the episode ends first, when $t + n \ge T$, there is nothing to bootstrap from, so $G_{t:t+n}$ is the full return $G_t$. With $n = 1$ it is the TD(0) target. With $n \ge T$ it is the Monte Carlo return.

Work it by hand. An episode visits $s_0, s_1, s_2$ and then ends, with rewards $r = (1, 0, 2)$, current estimates $V = (0.5, 1.0, 0.8)$, and $\gamma = 0.9$. From $t = 0$:

- $G_{0:1} = 1 + 0.9 \times 1.0 = 1.9$.
- $G_{0:2} = 1 + 0.9 \times 0 + 0.81 \times 0.8 = 1.648$.
- $G_{0:3} = 1 + 0 + 0.81 \times 2 = 2.62$, the full return.

```figure
<svg viewBox="0 0 580 250" role="img" aria-label="Stacked bars for the one-, two-, and three-step returns of the worked episode, split into real rewards and a bootstrapped guess">
<line x1="70.0" y1="190.0" x2="540.0" y2="190.0" class="stroke-muted"/>
<line x1="70.0" y1="190.0" x2="540.0" y2="190.0" class="stroke-grid"/>
<text x="62.0" y="194.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="163.3" x2="540.0" y2="163.3" class="stroke-grid"/>
<text x="62.0" y="167.3" class="fig-small" text-anchor="end">0.5</text>
<line x1="70.0" y1="136.7" x2="540.0" y2="136.7" class="stroke-grid"/>
<text x="62.0" y="140.7" class="fig-small" text-anchor="end">1</text>
<line x1="70.0" y1="110.0" x2="540.0" y2="110.0" class="stroke-grid"/>
<text x="62.0" y="114.0" class="fig-small" text-anchor="end">1.5</text>
<line x1="70.0" y1="83.3" x2="540.0" y2="83.3" class="stroke-grid"/>
<text x="62.0" y="87.3" class="fig-small" text-anchor="end">2</text>
<line x1="70.0" y1="56.7" x2="540.0" y2="56.7" class="stroke-grid"/>
<text x="62.0" y="60.7" class="fig-small" text-anchor="end">2.5</text>
<line x1="70.0" y1="30.0" x2="540.0" y2="30.0" class="stroke-grid"/>
<text x="62.0" y="34.0" class="fig-small" text-anchor="end">3</text>
<rect x="118.1" y="136.7" width="80.0" height="53.3" class="fill-a"/>
<rect x="118.1" y="88.7" width="80.0" height="48.0" class="fill-b"/>
<text x="158.1" y="116.7" class="fig-small" text-anchor="middle">guess 0.9</text>
<text x="158.1" y="167.3" class="fig-small" text-anchor="middle">rewards 1</text>
<text x="158.1" y="80.7" class="fig-small fig-bold" text-anchor="middle">G = 1.9</text>
<text x="158.1" y="208.0" class="fig-small fig-bold" text-anchor="middle">n = 1</text>
<text x="158.1" y="224.0" class="fig-small" text-anchor="middle">TD(0) target</text>
<rect x="265.0" y="136.7" width="80.0" height="53.3" class="fill-a"/>
<rect x="265.0" y="102.1" width="80.0" height="34.6" class="fill-b"/>
<text x="305.0" y="123.4" class="fig-small" text-anchor="middle">guess 0.648</text>
<text x="305.0" y="167.3" class="fig-small" text-anchor="middle">rewards 1</text>
<text x="305.0" y="94.1" class="fig-small fig-bold" text-anchor="middle">G = 1.648</text>
<text x="305.0" y="208.0" class="fig-small fig-bold" text-anchor="middle">n = 2</text>
<text x="305.0" y="224.0" class="fig-small" text-anchor="middle">in between</text>
<rect x="411.9" y="50.3" width="80.0" height="139.7" class="fill-a"/>
<text x="451.9" y="124.1" class="fig-small" text-anchor="middle">rewards 2.62</text>
<text x="451.9" y="42.3" class="fig-small fig-bold" text-anchor="middle">G = 2.62</text>
<text x="451.9" y="208.0" class="fig-small fig-bold" text-anchor="middle">n = 3</text>
<text x="451.9" y="224.0" class="fig-small" text-anchor="middle">Monte Carlo return</text>
</svg>
Figure 1. The three n-step returns from state s0 of the worked episode, whose current estimate is V(s0) = 0.5. Blue is real reward, orange is a discounted guess. Notice that each extra step swaps some guess for real reward until the guess is gone.
```

Each extra step swaps a discounted guess for a real reward. The first target is nearly half guess. The last has no guess at all. In numpy, sum the rewards in the window and add the tail only if the episode has not ended:

```python
end = min(t + n, T)
G = sum(gamma ** (k - t) * rewards[k] for k in range(t, end))
if t + n < T:
    G += gamma ** n * values[t + n]
```

## Why a later guess is a better guess

Why should waiting help? Because every step you wait multiplies the guess's mistakes by $\gamma$. The target $G_{t:t+n}$ trusts $V$ only for the part of the future beyond $n$ steps, and that part is discounted by $\gamma^n$. So whatever is wrong with $V$ matters less.

This has a precise form, the *error-reduction property*. On average, an n-step target is closer to the truth $V^\pi$ than $V$ is, by a factor of $\gamma^n$ in the worst state:

$$
\max_s \Bigl| \mathbb{E}_\pi[G_{t:t+n} \mid s_t = s] - V^\pi(s) \Bigr|
\le \gamma^n \max_s \bigl| V(s) - V^\pi(s) \bigr|
$$

Read it aloud: whatever your estimates get wrong, an n-step target is wrong by at most $\gamma^n$ times as much, on average. The proof idea takes three sentences. Let $P_\pi$ be the matrix of state-to-state probabilities under the policy, and $r_\pi$ the expected rewards. The expected n-step return is $\sum_{k<n} \gamma^k P_\pi^k r_\pi + \gamma^n P_\pi^n V$, and the true value satisfies the same equation with $V^\pi$ in place of $V$. Subtract them, and the error is $\gamma^n P_\pi^n (V - V^\pi)$. Each row of $P_\pi^n$ is a probability distribution, so it averages the error without growing it. In numpy, the expected return is a sum of matrix powers:

```python
out = sum(gamma**k * np.linalg.matrix_power(P_pi, k) @ r_pi for k in range(n))
out += gamma**n * np.linalg.matrix_power(P_pi, n) @ V
```

```figure
<svg viewBox="0 0 580 285" role="img" aria-label="Worst-case error of the expected n-step return on a random six-state MDP, below the gamma-to-the-n bound at every n">
<line x1="70.0" y1="220.0" x2="540.0" y2="220.0" class="stroke-grid"/>
<text x="62.0" y="224.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="156.7" x2="540.0" y2="156.7" class="stroke-grid"/>
<text x="62.0" y="160.7" class="fig-small" text-anchor="end">0.5</text>
<line x1="70.0" y1="93.3" x2="540.0" y2="93.3" class="stroke-grid"/>
<text x="62.0" y="97.3" class="fig-small" text-anchor="end">1</text>
<line x1="70.0" y1="220.0" x2="540.0" y2="220.0" class="stroke-muted"/>
<line x1="70.0" y1="220.0" x2="70.0" y2="225.0" class="stroke-muted"/>
<text x="70.0" y="240.0" class="fig-small" text-anchor="middle">0</text>
<line x1="164.0" y1="220.0" x2="164.0" y2="225.0" class="stroke-muted"/>
<text x="164.0" y="240.0" class="fig-small" text-anchor="middle">2</text>
<line x1="258.0" y1="220.0" x2="258.0" y2="225.0" class="stroke-muted"/>
<text x="258.0" y="240.0" class="fig-small" text-anchor="middle">4</text>
<line x1="352.0" y1="220.0" x2="352.0" y2="225.0" class="stroke-muted"/>
<text x="352.0" y="240.0" class="fig-small" text-anchor="middle">6</text>
<line x1="446.0" y1="220.0" x2="446.0" y2="225.0" class="stroke-muted"/>
<text x="446.0" y="240.0" class="fig-small" text-anchor="middle">8</text>
<line x1="540.0" y1="220.0" x2="540.0" y2="225.0" class="stroke-muted"/>
<text x="540.0" y="240.0" class="fig-small" text-anchor="middle">10</text>
<text x="305.0" y="262.0" text-anchor="middle">n</text>
<text x="62.0" y="18.0" text-anchor="end">error</text>
<polyline points="70.0,57.8 117.0,74.0 164.0,88.6 211.0,101.7 258.0,113.6 305.0,124.2 352.0,133.8 399.0,142.4 446.0,150.2 493.0,157.1 540.0,163.4" class="stroke-muted dashed" fill="none"/>
<polyline points="70.0,57.8 117.0,187.6 164.0,206.4 211.0,211.0 258.0,212.5 305.0,213.3 352.0,214.0 399.0,214.6 446.0,215.2 493.0,215.7 540.0,216.1" class="stroke-b" fill="none"/>
<circle cx="70.0" cy="57.8" r="4" class="fill-b"/>
<circle cx="117.0" cy="187.6" r="4" class="fill-b"/>
<circle cx="164.0" cy="206.4" r="4" class="fill-b"/>
<circle cx="211.0" cy="211.0" r="4" class="fill-b"/>
<circle cx="258.0" cy="212.5" r="4" class="fill-b"/>
<circle cx="305.0" cy="213.3" r="4" class="fill-b"/>
<circle cx="352.0" cy="214.0" r="4" class="fill-b"/>
<circle cx="399.0" cy="214.6" r="4" class="fill-b"/>
<circle cx="446.0" cy="215.2" r="4" class="fill-b"/>
<circle cx="493.0" cy="215.7" r="4" class="fill-b"/>
<circle cx="540.0" cy="216.1" r="4" class="fill-b"/>
<line x1="330.0" y1="50.0" x2="360.0" y2="50.0" class="stroke-muted dashed"/>
<text x="366.0" y="54.0" class="fig-small" text-anchor="start">bound: γⁿ × starting error</text>
<line x1="330.0" y1="72.0" x2="360.0" y2="72.0" class="stroke-b"/>
<text x="366.0" y="76.0" class="fig-small" text-anchor="start">actual error of the n-step return</text>
</svg>
Figure 2. Worst-state error of the expected n-step return on a random six-state MDP with γ = 0.9, starting from estimates that are off by 1.28. Notice that the actual error stays under the bound and shrinks as n grows.
```

On a random six-state MDP with $\gamma = 0.9$, the starting estimates are off by 1.28 in the worst state. The actual error of the expected n-step return falls well below the bound at every $n$. The bound says how much waiting helps at most, and it does not say waiting is free. The price of a longer wait is variance, which the next section measures.

## Update toward the n-step return

n-step TD moves $V(s_t)$ a fraction $\alpha$ toward $G_{t:t+n}$. Done online, the update for $s_t$ has to wait until $r_{t+n-1}$ arrives, which is $n - 1$ steps later. For stored episodes, you can update $s_0, s_1, \dots$ in order, each with the current $V$, which gives exactly the same result:

```python
for tau in range(T):
    G = ...                               # G_{tau:tau+n} from the current V
    V[states[tau]] += alpha * (G - V[states[tau]])
```

Check this by hand on the episode from before. The states are $0, 1, 2$ and then the end, the rewards are $(1, 0, 2)$, $\alpha = 1$, $\gamma = 0.9$, and every value starts at 0.5.

- With $n = 1$, $V(0) = 1 + 0.9 \times 0.5 = 1.45$, $V(1) = 0 + 0.9 \times 0.5 = 0.45$, and $V(2) = 2$.
- With $n = 2$, $V(0) = 1 + 0 + 0.81 \times 0.5 = 1.405$. The windows from states 1 and 2 reach the end, so $V(1) = 0 + 0.9 \times 2 = 1.8$ and $V(2) = 2$.
- With $n = 3$, every window reaches the end, so the values are the Monte Carlo returns $2.62$, $1.8$, and $2$.

## Pick n

Run the corridor from the opening: 19 states, exits that pay $-1$ on the left and $+1$ on the right, and $\gamma = 1$, so the true value of state $k$ is $(k - 10)/10$. Start every estimate at 0 and run 10 episodes with $\alpha = 0.2$. The RMS error across the 19 states, averaged over 30 seeded runs, depends on $n$ like this:

```figure
<svg viewBox="0 0 580 285" role="img" aria-label="RMS error after ten episodes on the 19-state random walk for n from 1 to 64, lowest at a middle n">
<line x1="70.0" y1="220.0" x2="540.0" y2="220.0" class="stroke-grid"/>
<text x="62.0" y="224.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="156.7" x2="540.0" y2="156.7" class="stroke-grid"/>
<text x="62.0" y="160.7" class="fig-small" text-anchor="end">0.2</text>
<line x1="70.0" y1="93.3" x2="540.0" y2="93.3" class="stroke-grid"/>
<text x="62.0" y="97.3" class="fig-small" text-anchor="end">0.4</text>
<line x1="70.0" y1="30.0" x2="540.0" y2="30.0" class="stroke-grid"/>
<text x="62.0" y="34.0" class="fig-small" text-anchor="end">0.6</text>
<line x1="70.0" y1="220.0" x2="540.0" y2="220.0" class="stroke-muted"/>
<text x="62.0" y="18.0" text-anchor="end">RMS error</text>
<polyline points="70.0,89.2 148.3,123.7 226.7,161.8 305.0,159.6 383.3,127.5 461.7,90.2 540.0,58.1" class="stroke-b" fill="none"/>
<circle cx="70.0" cy="89.2" r="5" class="fill-b"/>
<text x="70.0" y="79.2" class="fig-small" text-anchor="middle">0.41</text>
<text x="70.0" y="238.0" class="fig-small" text-anchor="middle">n = 1</text>
<circle cx="148.3" cy="123.7" r="5" class="fill-b"/>
<text x="148.3" y="113.7" class="fig-small" text-anchor="middle">0.30</text>
<text x="148.3" y="238.0" class="fig-small" text-anchor="middle">n = 2</text>
<circle cx="226.7" cy="161.8" r="5" class="fill-b"/>
<text x="226.7" y="151.8" class="fig-small" text-anchor="middle">0.18</text>
<text x="226.7" y="238.0" class="fig-small" text-anchor="middle">n = 4</text>
<circle cx="305.0" cy="159.6" r="5" class="fill-b"/>
<text x="305.0" y="149.6" class="fig-small" text-anchor="middle">0.19</text>
<text x="305.0" y="238.0" class="fig-small" text-anchor="middle">n = 8</text>
<circle cx="383.3" cy="127.5" r="5" class="fill-b"/>
<text x="383.3" y="117.5" class="fig-small" text-anchor="middle">0.29</text>
<text x="383.3" y="238.0" class="fig-small" text-anchor="middle">n = 16</text>
<circle cx="461.7" cy="90.2" r="5" class="fill-b"/>
<text x="461.7" y="80.2" class="fig-small" text-anchor="middle">0.41</text>
<text x="461.7" y="238.0" class="fig-small" text-anchor="middle">n = 32</text>
<circle cx="540.0" cy="58.1" r="5" class="fill-b"/>
<text x="540.0" y="48.1" class="fig-small" text-anchor="middle">0.51</text>
<text x="540.0" y="238.0" class="fig-small" text-anchor="middle">n = 64</text>
<text x="305.0" y="262.0" text-anchor="middle">steps of real reward before the guess</text>
</svg>
Figure 3. RMS error across the 19 states after 10 episodes with α = 0.2, averaged over 30 runs. Notice the U shape: n = 1 spreads news too slowly, and n = 64 is nearly Monte Carlo and too noisy.
```

The reason for the U shape is the two costs you have already met. At $n = 1$, news moves one cell per episode, so after 10 episodes most cells have not heard from the exits. At $n = 64$, nearly every target is the full return, which carries the luck of the whole walk, and a step size of 0.2 is too big for that much noise. A middle $n$ gets news back quickly and keeps the noise small.

> [!KEY]
> An n-step return adds $n$ real rewards before it bootstraps. Its expected error is at most $\gamma^n$ times the error of the values it uses, and a middle $n$ usually learns fastest, because it balances slow news against noisy news. In the corridor, $n = 4$ cuts the error from 0.41 to 0.18.

> [!RECAP]
> - `n_step_return` sums rewards in the window $[t, \min(t + n, T))$ and adds $\gamma^n V(s_{t+n})$ only if $t + n < T$.
> - The expected n-step return is a sum of matrix powers, and its error is $\gamma^n P_\pi^n (V - V^\pi)$.
> - n-step TD moves `V[states[tau]]` toward the n-step return, updating each step of a stored episode in order.
> - $n = 1$ is TD(0), and $n \ge T$ is Monte Carlo.
> - The best $n$ sits between the two ends.

## Build it step by step

The checks test the functions in this order. `random_walk` and `generate_episode` are already written. Episodes are `(states, rewards)` pairs, with `states` one longer than `rewards` because it ends at the terminal state. The return functions take `values`, the estimates along the episode, without the terminal state.

1. **`n_step_return(rewards, values, t, n, gamma)`.** Returns a float. The common mistake is adding the bootstrap term when $t + n = T$. There is no state after the terminal one.
2. **`expected_n_step_return(P_pi, r_pi, V, n, gamma)`.** Returns `(S,)`. Keep a running matrix power, `step = step @ P_pi`, instead of calling `matrix_power` in the loop.
3. **`n_step_td(episodes, n_states, n, alpha, gamma, v_init=0.0)`.** Returns `V`. Loop `tau` over each episode in order and update with the current `V`. The common mistake is forgetting to set each episode's final state to 0 before you start.

## Cost and scaling

Each n-step update sums up to $n$ rewards, so one episode of length $T$ costs $O(nT)$, or $O(T)$ if you keep a running sum. You also store the last $n$ rewards and states, because an update has to wait for them. Larger $n$ therefore costs more time and memory per step, and it delays every update by $n - 1$ steps.

## Where this breaks

- **Tuning two knobs.** The best $\alpha$ depends on $n$. A larger $n$ usually needs a smaller $\alpha$, so tune them together, not one at a time.
- **Delayed updates.** An n-step update waits $n$ steps, and for an episode that never ends, you cannot wait for the full return.
- **Choosing one n.** You pick a single $n$ for the whole problem, but the best $n$ can differ between states. [Eligibility traces](/lessons/020-eligibility-traces) average over every $n$ at once.
- **Off-policy data.** The n-step return follows the actions the agent took. To learn about another policy, each step needs an importance ratio, and the product of ratios brings back the variance problems of [importance sampling](/lessons/015-off-policy-importance-sampling).

## Further reading

- [Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd edition (2018), Chapter 7](http://incompleteideas.net/book/the-book-2nd.html). n-step methods, the error-reduction property, and the 19-state walk.
- [Sutton, "Learning to predict by the methods of temporal differences" (1988)](https://doi.org/10.1007/BF00115009). The paper that introduced TD learning.
- [Hessel, Modayil, van Hasselt, and others, "Rainbow: combining improvements in deep reinforcement learning" (2018)](https://arxiv.org/abs/1710.02298). n-step targets inside a deep Q-network.

## The exercise

You write `n_step_return`, `expected_n_step_return`, and `n_step_td`. The checks verify the worked values 1.9, 1.648, and 2.62, and that the return equals the full return once $t + n$ reaches $T$. They check the expected n-step return against matrix powers, and the $\gamma^n$ error bound for $n = 1$ to 6 on a random MDP. `n_step_td` must give the hand-computed values for $n = 1$, 2, and 3 on the worked episode. On the 19-state walk, $n = 4$ must beat both $n = 1$ and $n = 64$.
