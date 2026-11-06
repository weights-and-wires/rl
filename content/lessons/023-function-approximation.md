---
title: Value-function approximation
slug: 023-function-approximation
part: "Part 4 — Deep value methods"
summary: Replace the value table with a weighted sum of features, build tile-coding features, and learn the weights with semi-gradient TD.
minutes: 35
gpu: false
objectives:
  - Explain why a table cannot learn about states it has not visited, and how a feature vector fixes that.
  - Build tile-coding features, and say how the number of shared tiles sets how far a lesson spreads.
  - Apply one semi-gradient TD update by hand, including the terminal case.
  - Say why semi-gradient TD is not gradient descent on any loss.
prereqs:
  - 010-policy-evaluation
  - 016-temporal-difference-learning
exercise: 023-function-approximation
---

# Value-function approximation

A cart slides along a track, and the agent sees its position and speed as two real numbers. The position 0.3141 and the position 0.3142 are, for every practical purpose, the same place. But to a table they are two different rows, and the agent will probably never see the same row twice. Whatever it learns at 0.3141 tells it nothing about 0.3142.

Every method so far kept one number per state. This chapter replaces the table with a function that has a few adjustable weights. Learning at one state then moves the estimate at similar states too. That sharing is the point, and it is also the danger: an update that fixes one state can disturb another.

You will build the features, write the update, and see exactly how far a single lesson spreads.

## What you already know

- $V^\pi(s)$ is the expected discounted return from state $s$ under policy $\pi$, and `rlenvs.policy_evaluation` solves for it exactly.
- TD(0) nudges $V(s_t)$ toward the target $r_t + \gamma V(s_{t+1})$ after every step, as in [Temporal-difference learning](/lessons/016-temporal-difference-learning).
- A target that uses the current estimate is called *bootstrapping*.
- Gradient descent moves weights a small step against the gradient of a loss.
- A dot product `w @ x` multiplies matching entries and adds them up.

## Why a table is not enough

A table fails in two ways at once. A real state, such as a position and a speed, has far too many values to list. And a table has no notion of "nearby", so a lesson learned in one row never helps a row beside it.

The obvious fix is to chop the input into bins and keep one number per bin. A bin is a table row that covers a range. This helps, but it forces a choice. With wide bins, two states that need different values share one number, and the estimate is too coarse. With narrow bins, you are back to a table where each row is visited rarely, and a lesson spreads nowhere. You want fine resolution *and* wide sharing, and a single grid cannot give both.

Think of a weather map. A few stations report, and the map fills in the rest by assuming that nearby places are alike. A value function approximator does the same for states.

## Replace the table with a weighted sum

In the linear case, each state $s$ gets a *feature vector* $x(s)$ of length $d$, a short description of the state. The estimate is a weighted sum of the features:

$$
\hat V_w(s) = w^\top x(s) = \sum_{i=1}^{d} w_i \, x_i(s)
$$

Read it aloud: multiply each feature by its weight and add them up. There are $d$ weights to learn instead of one value per state. In numpy, with the feature vectors of all states stacked as rows of a matrix `Phi` of shape `(S, d)`, all the estimates at once are `Phi @ w`. A table is the special case where $x(s)$ is a one-hot vector, so each weight is one state's value.

With far fewer weights than states, no $w$ fits every state exactly, so you must say which errors matter most. The standard choice weights each state by how often the policy visits it. The *mean squared value error* is

$$
\overline{\mathrm{VE}}(w) = \sum_{s} \mu(s) \big(V^\pi(s) - \hat V_w(s)\big)^2
$$

Read it aloud: for each state, square the gap between the true value and the estimate, weight it by $\mu(s)$, and add. Here $\mu$ is the *on-policy distribution*, the long-run fraction of time the policy spends in each state. In code it is `d @ (Phi @ w - v) ** 2`. You cannot compute this during learning, because you do not know $V^\pi$. It is the yardstick that checks measure against.

## Build features with tile coding

The features decide what generalizes to what. *Tile coding* is a simple, fast choice for low-dimensional inputs. Lay a grid over the input range, and turn on the one tile the input falls in. Then lay several copies of the grid, each shifted a little, and turn on one tile in each copy. The copies are called *tilings*.

For a scalar $x$ in $[\text{low}, \text{high}]$, tiles of width $h$, and $n$ tilings, tiling $k$ is shifted left by $k/n$ of a tile. The input lands in tile

$$
\text{tile}_k(x) = \left\lfloor \frac{x - \text{low}}{h} + \frac{k}{n} \right\rfloor, \qquad k = 0, 1, \ldots, n - 1
$$

Read it aloud: measure how many tile widths $x$ sits above the bottom of the range, add the tiling's shift, and round down. Each tiling needs one extra tile to cover the top of the range after the shift. The feature vector is all zeros except for exactly $n$ ones, one per tiling:

```python
tile = int(np.floor((x - low) / width + k / n_tilings))
features[k * (n_tiles + 1) + tile] = 1.0
```

Take the range $[0, 1]$ with 4 tiles of width 0.25 and 2 tilings. Each tiling has 5 tiles, so there are 10 features.

| Input | Tiling 0 | Tiling 1 | Active features |
|---|---|---|---|
| $x = 0.3$ | $\lfloor 1.2 \rfloor = 1$ | $\lfloor 1.2 + 0.5 \rfloor = 1$ | 1 and $5 + 1 = 6$ |
| $x = 0.4$ | $\lfloor 1.6 \rfloor = 1$ | $\lfloor 1.6 + 0.5 \rfloor = 2$ | 1 and $5 + 2 = 7$ |
| $x = 0.9$ | $\lfloor 3.6 \rfloor = 3$ | $\lfloor 3.6 + 0.5 \rfloor = 4$ | 3 and 9 |

```figure
<svg viewBox="0 0 600 235" role="img" aria-label="Two shifted tilings of the interval from 0 to 1 with the active tiles of x = 0.3, 0.4, and 0.9 marked">
<text x="20.0" y="20.0" class="fig-small" text-anchor="start">Two tilings of [0, 1], four tiles each. Tiling 1 is shifted left by half a tile.</text>
<rect x="80.0" y="40.0" width="115.0" height="40.0" class="fill-soft-a stroke-muted"/>
<text x="137.5" y="65.0" class="fig-small" text-anchor="middle">0</text>
<rect x="195.0" y="40.0" width="115.0" height="40.0" class="fill-soft-a stroke-muted"/>
<text x="252.5" y="65.0" class="fig-small" text-anchor="middle">1</text>
<rect x="310.0" y="40.0" width="115.0" height="40.0" class="fill-soft-a stroke-muted"/>
<text x="367.5" y="65.0" class="fig-small" text-anchor="middle">2</text>
<rect x="425.0" y="40.0" width="115.0" height="40.0" class="fill-soft-a stroke-muted"/>
<text x="482.5" y="65.0" class="fig-small" text-anchor="middle">3</text>
<text x="70.0" y="65.0" class="fig-small" text-anchor="end">tiling 0</text>
<rect x="80.0" y="100.0" width="57.5" height="40.0" class="fill-soft-a stroke-muted"/>
<text x="108.8" y="125.0" class="fig-small" text-anchor="middle">5</text>
<rect x="137.5" y="100.0" width="115.0" height="40.0" class="fill-soft-a stroke-muted"/>
<text x="195.0" y="125.0" class="fig-small" text-anchor="middle">6</text>
<rect x="252.5" y="100.0" width="115.0" height="40.0" class="fill-soft-a stroke-muted"/>
<text x="310.0" y="125.0" class="fig-small" text-anchor="middle">7</text>
<rect x="367.5" y="100.0" width="115.0" height="40.0" class="fill-soft-a stroke-muted"/>
<text x="425.0" y="125.0" class="fig-small" text-anchor="middle">8</text>
<rect x="482.5" y="100.0" width="57.5" height="40.0" class="fill-soft-a stroke-muted"/>
<text x="511.2" y="125.0" class="fig-small" text-anchor="middle">9</text>
<text x="70.0" y="125.0" class="fig-small" text-anchor="end">tiling 1</text>
<line x1="218.0" y1="36.0" x2="218.0" y2="150.0" class="stroke-b" stroke-width="2"/>
<text x="218.0" y="168.0" class="fig-small" text-anchor="middle">x = 0.3</text>
<line x1="264.0" y1="36.0" x2="264.0" y2="150.0" class="stroke-c" stroke-width="2"/>
<text x="264.0" y="168.0" class="fig-small" text-anchor="middle">x = 0.4</text>
<line x1="494.0" y1="36.0" x2="494.0" y2="150.0" class="stroke-d" stroke-width="2"/>
<text x="494.0" y="168.0" class="fig-small" text-anchor="middle">x = 0.9</text>
<text x="300.0" y="200.0" class="fig-small" text-anchor="middle">0.3 activates tiles 1 and 6. 0.4 activates 1 and 7. 0.9 activates 3 and 9.</text>
<text x="300.0" y="220.0" class="fig-small" text-anchor="middle">0.3 and 0.4 share tile 1. 0.9 shares nothing with 0.3.</text>
</svg>
Figure 1. Two tilings of [0, 1]. The inputs 0.3 and 0.4 share tile 1 and differ in the shifted tiling, so they look similar to the learner. The input 0.9 shares no tile with 0.3. Notice that the shift is what lets nearby inputs share some tiles and not others.
```

The count of shared tiles decides how far a lesson spreads, which the next section turns into a number.

## Update the weights with semi-gradient TD

Tabular TD moves one entry toward its target. Here the target is the same, a bootstrapped number, but there is no entry to move. For a transition from $s_t$ with reward $r_t$ to $s_{t+1}$, the *TD error* is

$$
\delta_t = r_t + \gamma \, \hat V_w(s_{t+1}) - \hat V_w(s_t)
$$

Read it aloud: the target, which is the reward plus the discounted estimate at the next state, minus the current estimate. When $s_{t+1}$ is terminal, its value is zero and the target is $r_t$ alone. The weights then move along the features of the state you are correcting:

$$
w \leftarrow w + \alpha \, \delta_t \, x(s_t)
$$

Read it aloud: step by $\alpha$ times the error, in the direction that changes $\hat V_w(s_t)$ fastest. For a linear function, that direction is the feature vector. In numpy:

```python
target = r + gamma * (0.0 if terminated else w @ x2)
w = w + alpha * (target - w @ x) * x
```

Here is a worked example. Take $w = [1, 2, 0]$, $x(s_t) = [1, 0, 1]$, $x(s_{t+1}) = [0, 1, 1]$, $r_t = 0.5$, $\gamma = 0.9$, and $\alpha = 0.1$.

- Estimates: $\hat V(s_t) = 1$ and $\hat V(s_{t+1}) = 2$.
- Target: $0.5 + 0.9 \times 2 = 2.3$, so $\delta_t = 1.3$.
- Update: $w = [1, 2, 0] + 0.13 \times [1, 0, 1] = [1.13, 2, 0.13]$.

If the step had terminated, the target would be 0.5, $\delta_t = -0.5$, and $w = [0.95, 2, -0.05]$.

```figure
<svg viewBox="0 0 600 280" role="img" aria-label="Bar chart of the weights before and after one semi-gradient TD step: w1 rises from 1 to 1.13, w2 stays at 2, and w3 rises from 0 to 0.13">
<text x="20.0" y="22.0" class="fig-bold" text-anchor="start">One semi-gradient step: w moves along x(s_t) by alpha times delta</text>
<rect x="110.0" y="150.0" width="40.0" height="40.0" class="fill-a"/>
<rect x="154.0" y="144.8" width="40.0" height="45.2" class="fill-b"/>
<text x="130.0" y="144.0" class="fig-small" text-anchor="middle">1</text>
<text x="174.0" y="138.8" class="fig-small" text-anchor="middle">1.13</text>
<text x="152.0" y="208.0" class="fig-small" text-anchor="middle">w1</text>
<rect x="240.0" y="110.0" width="40.0" height="80.0" class="fill-a"/>
<rect x="284.0" y="110.0" width="40.0" height="80.0" class="fill-b"/>
<text x="260.0" y="104.0" class="fig-small" text-anchor="middle">2</text>
<text x="304.0" y="104.0" class="fig-small" text-anchor="middle">2</text>
<text x="282.0" y="208.0" class="fig-small" text-anchor="middle">w2</text>
<rect x="370.0" y="190.0" width="40.0" height="1.0" class="fill-a"/>
<rect x="414.0" y="184.8" width="40.0" height="5.2" class="fill-b"/>
<text x="390.0" y="184.0" class="fig-small" text-anchor="middle">0</text>
<text x="434.0" y="178.8" class="fig-small" text-anchor="middle">0.13</text>
<text x="412.0" y="208.0" class="fig-small" text-anchor="middle">w3</text>
<line x1="90.0" y1="190.0" x2="520.0" y2="190.0" class="stroke-muted"/>
<rect x="110.0" y="250.0" width="14.0" height="14.0" class="fill-a"/>
<text x="130.0" y="262.0" class="fig-small" text-anchor="start">before</text>
<rect x="210.0" y="250.0" width="14.0" height="14.0" class="fill-b"/>
<text x="230.0" y="262.0" class="fig-small" text-anchor="start">after</text>
<text x="320.0" y="262.0" class="fig-small" text-anchor="start">x(s_t) = [1, 0, 1], so w2 stays put</text>
</svg>
Figure 2. The weights before and after the update. Only the weights whose features were on in x(s_t) move, and w2 stays at 2 because its feature was off. Notice that each moved weight changes by the same amount, alpha times delta.
```

### How far does a lesson spread?

After this update, the estimate at any other state $s'$ changes by $\alpha \, \delta_t \, x(s_t)^\top x(s')$. The dot product counts the tiles that $s_t$ and $s'$ share. In the tiling example, 0.4 shares one of two tiles with 0.3, so it moves half as far, and 0.9 does not move at all.

With $\alpha = 1/n$ and $n$ tilings, the trained state moves exactly to its target in one update, because $x(s_t)^\top x(s_t) = n$ and so its change is $\delta_t$. Sutton and Barto suggest $\alpha = 1/(10n)$ to move a tenth of the way.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="After one update at x = 0.3, the value at nearby x changes by a fraction of the error that falls off with distance; 8 tilings spread it wider and more smoothly than 2">
<line x1="70.0" y1="220.0" x2="570.0" y2="220.0" class="stroke-grid"/>
<text x="62.0" y="224.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="130.0" x2="570.0" y2="130.0" class="stroke-grid"/>
<text x="62.0" y="134.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="70.0" y1="40.0" x2="570.0" y2="40.0" class="stroke-grid"/>
<text x="62.0" y="44.0" class="fig-small" text-anchor="end">1</text>
<text x="70.0" y="238.0" class="fig-small" text-anchor="middle">0</text>
<text x="195.0" y="238.0" class="fig-small" text-anchor="middle">0.25</text>
<text x="320.0" y="238.0" class="fig-small" text-anchor="middle">0.5</text>
<text x="445.0" y="238.0" class="fig-small" text-anchor="middle">0.75</text>
<text x="570.0" y="238.0" class="fig-small" text-anchor="middle">1</text>
<text x="62.0" y="22.0" class="fig-small" text-anchor="start">fraction of the error that reaches x</text>
<text x="300.0" y="262.0" text-anchor="middle">x (training happened at x = 0.3)</text>
<polyline points="70.0,220.0 75.0,220.0 80.0,220.0 85.0,220.0 90.0,220.0 95.0,220.0 100.0,220.0 105.0,220.0 110.0,220.0 115.0,220.0 120.0,220.0 125.0,220.0 130.0,220.0 135.0,130.0 140.0,130.0 145.0,130.0 150.0,130.0 155.0,130.0 160.0,130.0 165.0,130.0 170.0,130.0 175.0,130.0 180.0,130.0 185.0,130.0 190.0,130.0 195.0,40.0 200.0,40.0 205.0,40.0 210.0,40.0 215.0,40.0 220.0,40.0 225.0,40.0 230.0,40.0 235.0,40.0 240.0,40.0 245.0,40.0 250.0,40.0 255.0,40.0 260.0,130.0 265.0,130.0 270.0,130.0 275.0,130.0 280.0,130.0 285.0,130.0 290.0,130.0 295.0,130.0 300.0,130.0 305.0,130.0 310.0,130.0 315.0,130.0 320.0,220.0 325.0,220.0 330.0,220.0 335.0,220.0 340.0,220.0 345.0,220.0 350.0,220.0 355.0,220.0 360.0,220.0 365.0,220.0 370.0,220.0 375.0,220.0 380.0,220.0 385.0,220.0 390.0,220.0 395.0,220.0 400.0,220.0 405.0,220.0 410.0,220.0 415.0,220.0 420.0,220.0 425.0,220.0 430.0,220.0 435.0,220.0 440.0,220.0 445.0,220.0 450.0,220.0 455.0,220.0 460.0,220.0 465.0,220.0 470.0,220.0 475.0,220.0 480.0,220.0 485.0,220.0 490.0,220.0 495.0,220.0 500.0,220.0 505.0,220.0 510.0,220.0 515.0,220.0 520.0,220.0 525.0,220.0 530.0,220.0 535.0,220.0 540.0,220.0 545.0,220.0 550.0,220.0 555.0,220.0 560.0,220.0 565.0,220.0 570.0,220.0" class="stroke-a" fill="none"/>
<polyline points="70.0,220.0 75.0,220.0 80.0,220.0 85.0,220.0 90.0,220.0 95.0,220.0 100.0,220.0 105.0,197.5 110.0,197.5 115.0,197.5 120.0,175.0 125.0,175.0 130.0,175.0 135.0,152.5 140.0,152.5 145.0,152.5 150.0,130.0 155.0,130.0 160.0,130.0 165.0,107.5 170.0,107.5 175.0,107.5 180.0,85.0 185.0,85.0 190.0,85.0 195.0,62.5 200.0,62.5 205.0,62.5 210.0,62.5 215.0,40.0 220.0,40.0 225.0,40.0 230.0,62.5 235.0,62.5 240.0,62.5 245.0,85.0 250.0,85.0 255.0,85.0 260.0,107.5 265.0,107.5 270.0,107.5 275.0,130.0 280.0,130.0 285.0,130.0 290.0,152.5 295.0,152.5 300.0,152.5 305.0,175.0 310.0,175.0 315.0,175.0 320.0,197.5 325.0,197.5 330.0,197.5 335.0,197.5 340.0,220.0 345.0,220.0 350.0,220.0 355.0,220.0 360.0,220.0 365.0,220.0 370.0,220.0 375.0,220.0 380.0,220.0 385.0,220.0 390.0,220.0 395.0,220.0 400.0,220.0 405.0,220.0 410.0,220.0 415.0,220.0 420.0,220.0 425.0,220.0 430.0,220.0 435.0,220.0 440.0,220.0 445.0,220.0 450.0,220.0 455.0,220.0 460.0,220.0 465.0,220.0 470.0,220.0 475.0,220.0 480.0,220.0 485.0,220.0 490.0,220.0 495.0,220.0 500.0,220.0 505.0,220.0 510.0,220.0 515.0,220.0 520.0,220.0 525.0,220.0 530.0,220.0 535.0,220.0 540.0,220.0 545.0,220.0 550.0,220.0 555.0,220.0 560.0,220.0 565.0,220.0 570.0,220.0" class="stroke-b" fill="none"/>
<line x1="400.0" y1="52.0" x2="430.0" y2="52.0" class="stroke-a"/>
<text x="438.0" y="56.0" class="fig-small" text-anchor="start">2 tilings</text>
<line x1="400.0" y1="70.0" x2="430.0" y2="70.0" class="stroke-b"/>
<text x="438.0" y="74.0" class="fig-small" text-anchor="start">8 tilings</text>
</svg>
Figure 3. Change at each input after one update at 0.3, as a fraction of the error, with 4 tiles per tiling. Notice that more tilings make the bump smoother and wider while keeping its peak at 1, so tilings set how far a lesson spreads and the tile width sets the resolution.
```

> [!KEY]
> A function approximator lets one update teach many states. In tile coding, the number of shared tiles decides how much each state learns from another.

### Why "semi"

This is called *semi-gradient* TD. It is the gradient of $\tfrac12 (\text{target} - \hat V_w(s_t))^2$ with the target held fixed. But the target depends on $w$ too, through $\hat V_w(s_{t+1})$, and the update ignores that dependence. So semi-gradient TD is not gradient descent on any loss. That detail is harmless on-policy with a shrinking step size, and decisive off-policy, as [the next chapter](/lessons/024-the-deadly-triad) shows.

> [!TIP]
> Computing $\overline{\mathrm{VE}}$ needs $V^\pi$, so use it only on problems where you can solve for $V^\pi$ exactly, such as the exercise's random MDP. In a real task, you track return instead.

> [!RECAP]
> - A linear estimate is $\hat V_w(s) = w^\top x(s)$, and a table is the case of one-hot features.
> - Tile coding turns on one tile per tiling. Shared tiles decide how far an update spreads.
> - Semi-gradient TD moves $w$ by $\alpha \delta_t x(s_t)$ and treats the target as a constant. A terminal step drops the bootstrap term.
> - $\overline{\mathrm{VE}}$ weights each state's squared error by how often the policy visits it.

## Build it step by step

The checks test the functions in this order. `stationary_distribution` is already written.

1. **`tile_features(x, low, high, n_tiles, n_tilings)`.** Returns a float vector of length `n_tilings * (n_tiles + 1)`. Clip `x` to the range, compute each tiling's tile with `np.floor`, and set one entry per tiling. Common mistake: shifting the tilings right instead of left. The hand example catches it.
2. **`semi_gradient_td_update(w, x, r, x2, terminated, alpha, gamma)`.** Returns the updated weights: two dot products and one scaled vector add. Common mistake: bootstrapping from `x2` on a terminal step. A later check combines this function with your tiles and expects one-trial learning at $\alpha = 1/n$.
3. **`value_error(Phi, w, v, d)`.** One line, `d @ (Phi @ w - v) ** 2`. Common mistake: squaring after the weighted sum instead of before.

## Where this breaks

**Bad features.** Two states with the same features always get the same value, whatever the update rule. If the features cannot tell two states apart, no learning rule can.

**Dimension.** The number of tiles grows exponentially with the input dimension. That is fine for two or three inputs and hopeless for pixels. [Deep Q-networks](/lessons/025-deep-q-networks) learn the features with a neural network instead.

**What the weights converge to.** On-policy, linear semi-gradient TD settles at a specific point that is not the best fit. [The deadly triad](/lessons/024-the-deadly-triad) finds that point and shows what happens when the data comes from a different policy.

## Further reading

- [Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd edition (2018), Chapter 9](http://incompleteideas.net/book/the-book-2nd.html). Linear methods, tile coding, and the step size rule $\alpha = 1/(10n)$.
- [Sutton, "Learning to predict by the methods of temporal differences" (1988)](https://doi.org/10.1007/BF00115009). The paper that introduced TD learning.
- [Tsitsiklis and Van Roy, "An analysis of temporal-difference learning with function approximation" (1997)](https://doi.org/10.1109/9.580874). Proves that linear TD converges on-policy.

## The exercise

You build the pieces of a linear TD learner. Write `tile_features`, then `semi_gradient_td_update`, then `value_error`.

The checks verify the tiles against the hand example, including the clipping of an input below the range. They test the update's arithmetic and terminal masking against the worked example. They confirm one-trial learning at $\alpha = 1/n$ with your tiles, where 0.4 moves half as far as 0.3 and 0.9 does not move. They check `value_error` by hand. Finally, semi-gradient TD with a shrinking step size must bring the values on a random MDP within a set distance of the point it converges to.
