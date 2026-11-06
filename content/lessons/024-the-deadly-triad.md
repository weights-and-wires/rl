---
title: The deadly triad
slug: 024-the-deadly-triad
part: "Part 4 — Deep value methods"
summary: Find the point linear TD settles at, estimate it from data with LSTD, and watch Baird's counterexample diverge off-policy.
minutes: 40
gpu: false
objectives:
  - Name the three ingredients of the deadly triad, and say why any two alone can be made stable.
  - Compute the TD fixed point as a linear solve, and say why it differs from the best linear fit.
  - Estimate the fixed point from a batch of transitions with LSTD.
  - Run Baird's counterexample, and say why uniform weighting diverges and on-policy weighting does not.
prereqs:
  - 009-the-bellman-equation
  - 023-function-approximation
exercise: 024-the-deadly-triad
---

# The deadly triad

Q-learning learns about the best policy while following an exploring one. In a table that is fine, because each entry has its own number, and no update can disturb another entry. Swap the table for the weighted sum from the last chapter. An update at one state moves the estimates at others, and the data no longer comes from the policy being evaluated.

This chapter asks what can go wrong, and the answer is startling. There is a seven-state problem where every true value is 0, the weights $w = 0$ fit it perfectly, and a standard TD update, started near zero, drives the weights to infinity.

You will first learn where TD settles when it behaves, so that you can see exactly what changes when it does not.

## What you already know

- Linear semi-gradient TD moves $w$ by $\alpha \delta_t x(s_t)$, as in [Value-function approximation](/lessons/023-function-approximation).
- $\overline{\mathrm{VE}}(w)$ is the squared gap to the true values, weighted by how often the policy visits each state.
- The Bellman equation $V = r_\pi + \gamma P_\pi V$ is a linear system, as in [The Bellman equation](/lessons/009-the-bellman-equation).
- Off-policy learning evaluates one policy, the *target policy*, from data that another policy, the *behavior policy*, collected.
- A matrix whose largest eigenvalue magnitude exceeds 1 makes repeated multiplication grow without bound.

## Three ingredients

Sutton and Barto call the following three ingredients the *deadly triad*:

- **Function approximation**, so updating one state changes others.
- **Bootstrapping**, so targets depend on the current estimates.
- **Off-policy training**, so the states you update are not distributed the way the target policy visits them.

Any two are safe to combine. Tabular Q-learning bootstraps off-policy and converges. Monte Carlo with function approximation never bootstraps and is stable. The linear TD of the last chapter approximates and bootstraps, and it converges when the data is on-policy. Add the third, and instability can appear even in the simplest setting.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="A triangle with function approximation, bootstrapping, and off-policy data at the corners; each edge is a safe pair and the center is the unstable combination">
<polyline points="300.0,50.0 110.0,240.0 490.0,240.0 300.0,50.0" class="stroke-muted" fill="none"/>
<rect x="205.0" y="30.0" width="190.0" height="40.0" class="fill-soft-a stroke-a" rx="6"/>
<text x="300.0" y="55.0" class="fig-bold" text-anchor="middle">function approximation</text>
<rect x="15.0" y="240.0" width="190.0" height="40.0" class="fill-soft-b stroke-b" rx="6"/>
<text x="110.0" y="265.0" class="fig-bold" text-anchor="middle">bootstrapping</text>
<rect x="395.0" y="240.0" width="190.0" height="40.0" class="fill-soft-c stroke-c" rx="6"/>
<text x="490.0" y="265.0" class="fig-bold" text-anchor="middle">off-policy data</text>
<text x="300.0" y="150.0" class="fig-bold" text-anchor="middle">all three:</text>
<text x="300.0" y="170.0" class="fig-bold" text-anchor="middle">can diverge</text>
<text x="175.0" y="135.0" class="fig-small" text-anchor="end">bootstrapping + approximation,</text>
<text x="175.0" y="151.0" class="fig-small" text-anchor="end">on-policy: converges</text>
<text x="425.0" y="135.0" class="fig-small" text-anchor="start">approximation + off-policy,</text>
<text x="425.0" y="151.0" class="fig-small" text-anchor="start">no bootstrapping: stable</text>
<text x="300.0" y="305.0" class="fig-small" text-anchor="middle">bootstrapping + off-policy, no approximation (tabular Q-learning): converges</text>
</svg>
Figure 1. Each edge is a pair of ingredients that has a known safe case. The center is all three together, which can diverge. Notice that the danger comes from the combination, not from any one ingredient.
```

## Where on-policy TD settles

Start with the safe case. You might expect semi-gradient TD to find the best linear fit, the weights that minimize $\overline{\mathrm{VE}}$. It does not, and the reason is the "semi". The update ignores that the target depends on $w$, so it is not following the gradient of $\overline{\mathrm{VE}}$. It follows something else, and it rests where that something else is zero.

Average the update over the on-policy distribution. The expected change is $\alpha(b - Aw)$, so the resting point solves a linear system:

$$
A = \Phi^\top D (I - \gamma P_\pi) \Phi, \qquad b = \Phi^\top D \, r_\pi, \qquad w_{\text{TD}} = A^{-1} b
$$

Read it aloud: $\Phi$ is the `(S, d)` feature matrix, $D$ is a diagonal matrix of the visit fractions $\mu(s)$, $P_\pi$ is the `(S, S)` matrix of transition probabilities under the policy, and $r_\pi$ is the expected reward in each state. The solution $w_{\text{TD}}$ is the *TD fixed point*, where the expected update is zero. In numpy:

```python
D = np.diag(d)
A = Phi.T @ D @ (np.eye(S) - gamma * P_pi) @ Phi
w_td = np.linalg.solve(A, Phi.T @ D @ r_pi)
```

With one-hot features, $\Phi = I$ and the system becomes the Bellman equation, so $w_{\text{TD}} = V^\pi$. With fewer features, the fixed point can be worse than the best fit, but not by much. Tsitsiklis and Van Roy (1997) proved a bound that implies the following, for the continuing case with $\mu$ the on-policy distribution:

$$
\overline{\mathrm{VE}}(w_{\text{TD}}) \le \frac{1}{1 - \gamma} \min_w \overline{\mathrm{VE}}(w)
$$

Read it aloud: the error at the TD fixed point is at most $1/(1 - \gamma)$ times the smallest error any weights could reach. With $\gamma = 0.9$ that is a factor of 10 in the worst case. In practice it is usually far smaller.

Here is a worked example. Two states swap back and forth forever. State 0 pays $r = 1$ and state 1 pays $r = 0$, with $\gamma = 0.5$. The true values solve $V_0 = 1 + 0.5 V_1$ and $V_1 = 0.5 V_0$, so $V^\pi = [4/3, 2/3]$. The chain spends half its time in each state, so $\mu = [0.5, 0.5]$. Use one feature, $x = [1, 2]$.

- Best fit: $w = \sum \mu x V / \sum \mu x^2 = (8/3)/5 = 0.5333$, giving estimates $[0.533, 1.067]$ and $\overline{\mathrm{VE}} = 0.4$.
- TD fixed point: $A = 0.5 \times 1 \times (1 - 0.5 \times 2) + 0.5 \times 2 \times (2 - 0.5 \times 1) = 1.5$ and $b = 0.5 \times 1 \times 1 = 0.5$, so $w_{\text{TD}} = 1/3$. The estimates are $[0.333, 0.667]$ and $\overline{\mathrm{VE}} = 0.5$.

TD's error is 1.25 times the best, inside the bound of $1/(1 - 0.5) = 2$. It fits the Bellman equation in feature space, not the values themselves.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Estimates in two states: the true values, the best linear fit, and the TD fixed point, which fits the Bellman equation rather than the values">
<text x="20.0" y="22.0" class="fig-bold" text-anchor="start">Two-state chain, gamma = 0.5, one feature x = [1, 2]</text>
<rect x="90.0" y="80.0" width="44.0" height="120.0" class="fill-ink"/>
<text x="112.0" y="74.0" class="fig-small" text-anchor="middle">1.333</text>
<rect x="142.0" y="152.0" width="44.0" height="48.0" class="fill-a"/>
<text x="164.0" y="146.0" class="fig-small" text-anchor="middle">0.533</text>
<rect x="194.0" y="170.0" width="44.0" height="30.0" class="fill-b"/>
<text x="216.0" y="164.0" class="fig-small" text-anchor="middle">0.333</text>
<text x="164.0" y="220.0" class="fig-small" text-anchor="middle">state 0</text>
<rect x="350.0" y="140.0" width="44.0" height="60.0" class="fill-ink"/>
<text x="372.0" y="134.0" class="fig-small" text-anchor="middle">0.667</text>
<rect x="402.0" y="104.0" width="44.0" height="96.0" class="fill-a"/>
<text x="424.0" y="98.0" class="fig-small" text-anchor="middle">1.067</text>
<rect x="454.0" y="140.0" width="44.0" height="60.0" class="fill-b"/>
<text x="476.0" y="134.0" class="fig-small" text-anchor="middle">0.667</text>
<text x="424.0" y="220.0" class="fig-small" text-anchor="middle">state 1</text>
<line x1="70.0" y1="200.0" x2="540.0" y2="200.0" class="stroke-muted"/>
<rect x="40.0" y="245.0" width="14.0" height="14.0" class="fill-ink"/>
<text x="60.0" y="257.0" class="fig-small" text-anchor="start">true V</text>
<rect x="230.0" y="245.0" width="14.0" height="14.0" class="fill-a"/>
<text x="250.0" y="257.0" class="fig-small" text-anchor="start">best fit, VE 0.4</text>
<rect x="420.0" y="245.0" width="14.0" height="14.0" class="fill-b"/>
<text x="440.0" y="257.0" class="fig-small" text-anchor="start">TD fixed point, VE 0.5</text>
</svg>
Figure 2. True values, the best linear fit, and the TD fixed point for the two-state chain. Notice that the fixed point is exact in state 1 and off in state 0, while the best fit spreads its error across both: TD trades a little accuracy for a self-consistent answer.
```

> [!KEY]
> Linear TD does not find the best fit. It finds the TD fixed point $A^{-1} b$, whose error on-policy is at most $1/(1 - \gamma)$ times the best fit's.

## Estimate the fixed point from data

Semi-gradient TD reaches $w_{\text{TD}}$ slowly, one small step at a time, and its answer depends on the step size schedule. *Least-squares TD*, or LSTD (Bradtke and Barto, 1996), skips the steps. It builds $A$ and $b$ from a batch of transitions and solves once:

$$
\hat A = \sum_{t} x_t \big(x_t - \gamma (1 - \text{term}_t) \, x_{t+1}\big)^\top + \epsilon I, \qquad \hat b = \sum_t r_t \, x_t
$$

Read it aloud: the same $A$ and $b$, with expectations replaced by sums over transitions. The next features drop out when the step terminated, and a small ridge $\epsilon$ on the diagonal keeps the solve from failing. Then solve $\hat A w = \hat b$.

```python
live = 1.0 - terminated
A = X.T @ (X - gamma * live[:, None] * X2) + reg * np.eye(d)
w = np.linalg.solve(A, X.T @ rewards)
```

With one-hot features, LSTD returns the exact value of the *empirical model*, the MDP whose transition probabilities and rewards are the observed frequencies and means.

## Watch it diverge: Baird's counterexample

Baird (1995) built the standard example. There are seven states. Six upper states have value $2w_i + w_8$, and the lower state has value $w_7 + 2w_8$. The target policy always takes the *solid* action, which moves every state to the lower state with reward 0. So the true value of every state is 0, and $w = 0$ fits it exactly. In code, $w_1$ to $w_8$ are `w[0]` to `w[7]`, and the lower state is state 6.

The behavior policy picks the solid action with probability 1/7 and the *dashed* action, which jumps to a random upper state, with probability 6/7. Its data covers all seven states evenly, although the target policy visits only the lower state. Run the expected, synchronous update, with each state weighted by $\mu(s) = 1/7$:

$$
w \leftarrow w + \alpha \sum_{s} \mu(s) \big(\gamma \hat V_w(s_{\text{low}}) - \hat V_w(s)\big) \, x(s)
$$

Read it aloud: every state's target is $\gamma$ times the lower state's estimate, because the solid action always lands there with no reward. Move each state's estimate toward its target, weighted by $\mu(s)$.

Start from $w = (1, 1, 1, 1, 1, 1, 10, 1)$ with $\alpha = 0.01$ and $\gamma = 0.99$. The upper states are worth $2 + 1 = 3$ and the lower state is worth $10 + 2 = 12$.

- Every target is $0.99 \times 12 = 11.88$.
- The upper states have TD error $11.88 - 3 = 8.88$. They all want to rise.
- The lower state has TD error $11.88 - 12 = -0.12$. It wants to fall a little.
- All seven states share $w_8$. Six large positive errors outvote one small negative one, so $w_8$ rises from 1 to 1.0758.
- After the sweep, the upper states are worth 3.127 and the lower state is worth 12.151. The lower state went *up*, and its target went up with it.

That repeats every sweep. The iteration is linear, and its matrix has an eigenvalue of about $1.0024$, so the growth is exponential. After 1,000 sweeps the norm of $w$ has grown from 10.3 to 437. Sutton and Barto show that it diverges for any positive step size, because the problem is the update's direction, not its size.

Change only the weighting to the on-policy distribution, with all mass on the lower state. Only the lower state updates, and its value shrinks by a factor $1 - \alpha(1 - \gamma) \lVert x_{\text{low}} \rVert^2 = 0.9995$ each sweep, from 12 to $12 \times 0.9995^{1000} = 7.28$.

```figure
<svg viewBox="0 0 600 275" role="img" aria-label="Norm of the weight vector over 1000 sweeps of Baird's counterexample: it grows from 10.3 to 437 under uniform weights and stays near 10 under on-policy weights">
<line x1="70.0" y1="202.5" x2="560.0" y2="202.5" class="stroke-grid"/>
<text x="62.0" y="206.5" class="fig-small" text-anchor="end">10</text>
<line x1="70.0" y1="111.1" x2="560.0" y2="111.1" class="stroke-grid"/>
<text x="62.0" y="115.1" class="fig-small" text-anchor="end">100</text>
<text x="70.0" y="248.0" class="fig-small" text-anchor="middle">0</text>
<text x="192.5" y="248.0" class="fig-small" text-anchor="middle">250</text>
<text x="315.0" y="248.0" class="fig-small" text-anchor="middle">500</text>
<text x="437.5" y="248.0" class="fig-small" text-anchor="middle">750</text>
<text x="560.0" y="248.0" class="fig-small" text-anchor="middle">1000</text>
<text x="300.0" y="270.0" text-anchor="middle">sweeps</text>
<text x="62.0" y="22.0" class="fig-small" text-anchor="start">norm of w (log scale)</text>
<polyline points="70.0,201.1 94.5,193.4 119.0,181.8 143.5,170.1 168.0,159.4 192.5,149.6 217.0,140.6 241.5,132.4 266.0,124.7 290.5,117.5 315.0,110.6 339.5,104.1 364.0,97.7 388.5,91.6 413.0,85.7 437.5,79.9 462.0,74.3 486.5,68.7 511.0,63.3 535.5,57.9 560.0,52.6" class="stroke-b" fill="none"/>
<polyline points="70.0,201.1 94.5,201.4 119.0,201.7 143.5,201.9 168.0,202.1 192.5,202.4 217.0,202.6 241.5,202.8 266.0,203.0 290.5,203.2 315.0,203.4 339.5,203.5 364.0,203.7 388.5,203.9 413.0,204.0 437.5,204.2 462.0,204.3 486.5,204.5 511.0,204.6 535.5,204.7 560.0,204.9" class="stroke-a" fill="none"/>
<text x="554.0" y="44.6" class="fig-small" text-anchor="end">437</text>
<text x="554.0" y="196.9" class="fig-small" text-anchor="end">9.4</text>
<line x1="110.0" y1="52.0" x2="140.0" y2="52.0" class="stroke-b"/>
<text x="148.0" y="56.0" class="fig-small" text-anchor="start">uniform weights (off-policy)</text>
<line x1="110.0" y1="70.0" x2="140.0" y2="70.0" class="stroke-a"/>
<text x="148.0" y="74.0" class="fig-small" text-anchor="start">on-policy weights</text>
</svg>
Figure 3. The norm of w over 1,000 sweeps of the same update with two weightings of the states. Notice that the only difference is which states count: with the behavior policy's uniform weights the norm explodes, and with the target policy's own visit distribution it stays near 10.
```

> [!KEY]
> Function approximation, bootstrapping, and off-policy updates together can diverge, even with expected updates and a tiny step size. In Baird's example, on-policy weighting alone restores stability.

> [!RECAP]
> - The TD fixed point solves $\Phi^\top D (I - \gamma P_\pi) \Phi w = \Phi^\top D r_\pi$. One-hot features recover $V^\pi$.
> - On-policy, its error is at most $1/(1 - \gamma)$ times the best fit's.
> - LSTD builds the same $A$ and $b$ from samples and solves once.
> - Baird's expected update diverges under uniform weights and decays under on-policy weights.

## Build it step by step

The checks test the functions in this order. `stationary_distribution` and `value_error` are already written, because you wrote `value_error` in the previous chapter.

1. **`td_fixed_point(Phi, P_pi, r_pi, d, gamma)`.** Build $A$ and $b$ with `np.diag` and `@`, then call `np.linalg.solve`. Common mistake: writing `(I - gamma P) @ D` instead of `D @ (I - gamma P)`. The residual check and the bound check on 40 random problems catch it.
2. **`lstd(X, rewards, X2, terminated, gamma, reg)`.** One matrix product for $\hat A$ and one for $\hat b$. Common mistake: forgetting to mask terminal rows.
3. **`baird_features()`.** Returns the `(7, 8)` matrix. Upper state $i$ has a 2 in column $i$ and a 1 in column 7. The lower state has a 1 in column 6 and a 2 in column 7.
4. **`baird_expected_update(w, alpha, gamma, weights)`.** Compute `v = Phi @ w`, the errors `gamma * v[6] - v`, and step by `alpha * Phi.T @ (weights * errors)`. Common mistake: using `v` instead of `v[6]` in the target.

## Where this breaks

**Off-policy divergence.** Baird's example is the clean version of a real problem. Gradient-TD methods such as TDC follow the true gradient of a projected Bellman error, and emphatic TD reweights each update so that the expected update is stable. Deep Q-learning leans on replay buffers and target networks instead, which [the next chapter](/lessons/025-deep-q-networks) builds.

**Ill-conditioned LSTD.** With few samples or rare features, $\hat A$ is nearly singular and the solve amplifies noise. The ridge term trades a small bias for stability. LSTD also needs $O(d^2)$ memory, which rules it out beyond a few thousand features.

**Nonlinear approximators.** The fixed-point analysis is for linear functions. With neural networks, even on-policy TD can diverge.

## Further reading

- [Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd edition (2018)](http://incompleteideas.net/book/the-book-2nd.html). Chapter 9 covers linear methods and LSTD. Chapter 11 covers Baird's counterexample and the deadly triad.
- [Tsitsiklis and Van Roy, "An analysis of temporal-difference learning with function approximation" (1997)](https://doi.org/10.1109/9.580874). Proves linear TD converges to the fixed point and bounds its error.
- [Bradtke and Barto, "Linear least-squares algorithms for temporal difference learning" (1996)](https://doi.org/10.1007/BF00114723). Introduces LSTD.
- [Baird, "Residual algorithms: Reinforcement learning with function approximation" (1995)](https://doi.org/10.1016/B978-1-55860-377-6.50013-X). The counterexample, and residual-gradient updates as one fix.
- [Sutton, Maei, and Precup, "Fast gradient-descent methods for temporal-difference learning with linear function approximation" (2009)](https://doi.org/10.1145/1553374.1553501). The TDC and GTD2 algorithms, which converge off-policy.
- [van Hasselt and colleagues, "Deep reinforcement learning and the deadly triad" (2018)](https://arxiv.org/abs/1812.02648). Measures how often the triad causes divergence in deep Q-learning, and which design choices make it rarer.

## The exercise

You write `td_fixed_point`, `lstd`, `baird_features`, and `baird_expected_update`.

The checks confirm that `td_fixed_point` recovers $V^\pi$ with one-hot features, zeroes the projected Bellman residual, and meets the $1/(1 - \gamma)$ bound on 40 random problems. They check that LSTD matches the empirical model exactly and lands near the fixed point on a long trajectory. Finally, Baird's feature matrix must have the right shape and entries, and its weights must blow up under uniform weighting and decay by the exact factor under on-policy weighting.
