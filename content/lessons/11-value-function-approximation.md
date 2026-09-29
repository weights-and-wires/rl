---
title: Value-function approximation
slug: 11-value-function-approximation
part: "Part 3 — Approximation and deep value methods"
summary: Replace the value table with a linear function, find where semi-gradient TD lands, and watch it diverge off-policy.
minutes: 60
gpu: false
objectives:
  - Build tile-coding features and explain how the number of shared tiles sets generalisation.
  - Implement linear semi-gradient TD(0), and say why it is not true gradient descent.
  - Solve for the TD fixed point exactly, estimate it from data with LSTD, and check the 1/(1 - gamma) error bound.
  - Reproduce Baird's counterexample and name the three ingredients of the deadly triad.
prereqs:
  - 05-dynamic-programming
  - 07-temporal-difference-learning
exercise: 11-value-function-approximation
---

# Value-function approximation

Every method so far kept one number per state in a table. A cart's position
and speed, or a screen of pixels, have far too many states to list, and most
never repeat. A table also learns nothing about a state it hasn't visited,
even when that state is almost identical to one it has.

This lesson swaps the table for a function with a few weights. Learning one
state then moves the estimate for similar states too. That sharing is the
point, and it's also the danger. An update that fixes one state can break
another, and the guarantees you relied on in Part 2 no longer hold
automatically.

You build the pieces in the linear case, where everything can be checked
exactly: tile-coding features, semi-gradient TD, the point it converges to,
and LSTD, which estimates that point from data. Then you run Baird's
counterexample, where the same update, fed the wrong mix of states, sends the
weights to infinity.

## What you already know

- $V^\pi(s)$ is the expected discounted return from state $s$ under policy
  $\pi$. `rlenvs.policy_evaluation` solves for it exactly.
- TD(0) nudges $V(s_t)$ toward the target $r_t + \gamma V(s_{t+1})$ after
  every step, as in *Temporal-difference learning*.
- A target that uses the current estimate is called *bootstrapping*.
- Off-policy learning evaluates one policy, the *target policy*, from data
  that another policy, the *behaviour policy*, collected.
- Gradient descent moves weights a small step against the gradient of a loss.

## Replace the table with a function

Think of a weather map. A few stations report, and the map fills in the rest
by assuming nearby places are alike. A value function approximator does the
same for states.

In the linear case, each state $s$ gets a *feature vector* $x(s)$ of length
$d$, a short description of the state. The estimate is a weighted sum of the
features:

$$
\hat V_w(s) = w^\top x(s) = \sum_{i=1}^{d} w_i \, x_i(s)
$$

Read it as: multiply each feature by its weight and add them up. There are
$d$ weights to learn instead of one value per state. In numpy, with the
features of every state stacked as rows of a matrix `Phi` of shape `(S, d)`,
all the estimates at once are `Phi @ w`.

With far fewer weights than states, no $w$ fits every state exactly. So you
need to say which errors matter most. The standard choice weights each state
by how often the policy visits it. The *mean squared value error* is

$$
\overline{\mathrm{VE}}(w) = \sum_{s} \mu(s) \big(V^\pi(s) - \hat V_w(s)\big)^2
$$

Read it as: for each state, square the gap between the true value and the
estimate, weight it by $\mu(s)$, and add. Here $\mu$ is the *on-policy
distribution*, the long-run fraction of time the policy spends in each state.
In code it's `d @ (Phi @ w - v) ** 2`. The weights that minimise it are a
weighted least-squares fit, the best any linear function can do.

## Build features with tile coding

The features decide what generalises to what. *Tile coding* is a simple,
fast choice for low-dimensional inputs. Lay a grid over the input range, and
turn on the one tile the input falls in. Then lay several copies of the grid,
each shifted a little, and turn on one tile in each copy. The copies are
called *tilings*.

For a scalar $x$ in $[\text{low}, \text{high}]$ with tiles of width
$h$ and $n$ tilings, tiling $k$ is shifted left by $k/n$ of a
tile. The input lands in tile

$$
\text{tile}_k(x) = \left\lfloor \frac{x - \text{low}}{h} + \frac{k}{n} \right\rfloor, \qquad k = 0, 1, \ldots, n - 1
$$

Read it as: measure how many tile widths $x$ sits above the bottom of the
range, add the tiling's shift, and round down. Each tiling needs one extra
tile to cover the top of the range after the shift. The feature vector is all
zeros except for exactly $n$ ones, one per tiling:

```python
tile = int(np.floor((x - low) / width + k / n_tilings))
features[k * (n_tiles + 1) + tile] = 1.0
```

### Worked example

Take the range $[0, 1]$ with 4 tiles of width 0.25, and 2 tilings, so each
tiling has 5 tiles and there are 10 features.

| Input | Tiling 0 | Tiling 1 | Active features |
|---|---|---|---|
| $x = 0.3$ | $\lfloor 1.2 \rfloor = 1$ | $\lfloor 1.2 + 0.5 \rfloor = 1$ | 1 and $5 + 1 = 6$ |
| $x = 0.4$ | $\lfloor 1.6 \rfloor = 1$ | $\lfloor 1.6 + 0.5 \rfloor = 2$ | 1 and $5 + 2 = 7$ |
| $x = 0.9$ | $\lfloor 3.6 \rfloor = 3$ | $\lfloor 3.6 + 0.5 \rfloor = 4$ | 3 and 9 |

The inputs 0.3 and 0.4 share one tile of two. The input 0.9 shares none with
0.3. How far a lesson learned at 0.3 spreads depends on exactly that count, as
the next section shows.

> [!TIP]
> Because exactly $n$ features are on, a step size of $\alpha = 1/n$ moves the
> trained state all the way to its target in one update. Sutton and Barto
> suggest $\alpha = 1/(10n)$ to move a tenth of the way.

## Update the weights with semi-gradient TD

You can't compute $\overline{\mathrm{VE}}$, because you don't know $V^\pi$. TD
learning replaces the unknown true value with its bootstrapped target, the
same one you used in the table. For a transition from $s_t$ with reward $r_t$
to $s_{t+1}$, the *TD error* is

$$
\delta_t = r_t + \gamma \, \hat V_w(s_{t+1}) - \hat V_w(s_t)
$$

Read it as: the target, which is the reward plus the discounted estimate at
the next state, minus the current estimate. When $s_{t+1}$ is terminal, its
value is zero and the target is $r_t$ alone. The update moves the weights
along the features of the state you're correcting:

$$
w \leftarrow w + \alpha \, \delta_t \, x(s_t)
$$

Read it as: step by $\alpha$ times the error, along the direction that
changes $\hat V_w(s_t)$ fastest. For a linear function, that direction is
the feature vector.

```python
target = r + gamma * (0.0 if terminated else w @ x2)
w = w + alpha * (target - w @ x) * x
```

This is called *semi-gradient* TD. It's the gradient of
$\tfrac12 (\text{target} - \hat V_w(s_t))^2$ with the target held fixed. The
target depends on $w$ too, through $\hat V_w(s_{t+1})$, and the update ignores
that dependence, so semi-gradient TD isn't gradient descent on any loss. For
linear functions, that detail is harmless on-policy and decisive off-policy.

With tile features, the change at another state $s'$ after one update is
$\alpha \, \delta_t \, x(s_t)^\top x(s')$, and the dot product counts the
shared tiles. In the worked example, 0.4 moves half as far as 0.3, and 0.9
doesn't move at all.

### Worked example

Take $w = [1, 2, 0]$, $x(s_t) = [1, 0, 1]$, $x(s_{t+1}) = [0, 1, 1]$,
$r_t = 0.5$, $\gamma = 0.9$, and $\alpha = 0.1$.

- Estimates: $\hat V(s_t) = 1$ and $\hat V(s_{t+1}) = 2$.
- Target: $0.5 + 0.9 \times 2 = 2.3$, so $\delta_t = 1.3$.
- Update: $w = [1, 2, 0] + 0.13 \times [1, 0, 1] = [1.13, 2, 0.13]$.

If the step had terminated, the target would be 0.5, $\delta_t = -0.5$, and
$w = [0.95, 2, -0.05]$.

## Find where TD converges

On-policy, linear semi-gradient TD converges, provided the step size shrinks
at the right rate and the features are linearly independent. It doesn't
converge to the best fit, though. It converges to the point where the TD
error averages to zero along every feature. Averaging the update
over the on-policy distribution gives $\mathbb{E}[\Delta w] = \alpha(b - Aw)$,
so the resting point solves a linear system:

$$
A = \Phi^\top D (I - \gamma P_\pi) \Phi, \qquad b = \Phi^\top D \, r_\pi, \qquad w_{\text{TD}} = A^{-1} b
$$

Read it as: $\Phi$ is the `(S, d)` feature matrix, $D$ is a diagonal matrix
of the $\mu(s)$, $P_\pi$ is the `(S, S)` matrix of transition probabilities
under the policy, and $r_\pi$ is the expected reward in each state. The
solution $w_{\text{TD}}$ is the *TD fixed point*, where the expected update is
zero.

```python
D = np.diag(d)
A = Phi.T @ D @ (np.eye(S) - gamma * P_pi) @ Phi
w_td = np.linalg.solve(A, Phi.T @ D @ r_pi)
```

With one-hot features, $\Phi = I$, and the system becomes the Bellman
equation, so $w_{\text{TD}} = V^\pi$. With fewer features, the TD fixed point
can be worse than the best fit, but not by much. Tsitsiklis and Van Roy
(1997) proved a bound that implies the following one, for the continuing
case with $\mu$ the on-policy distribution:

$$
\overline{\mathrm{VE}}(w_{\text{TD}}) \le \frac{1}{1 - \gamma} \min_w \overline{\mathrm{VE}}(w)
$$

Read it as: the error at the TD fixed point is at most $1/(1 - \gamma)$ times
the smallest error any weights could reach. With $\gamma = 0.9$ that's a
factor of 10 in the worst case. In practice it's usually far smaller.

### Worked example

Two states swap back and forth forever. State 0 pays $r = 1$ and state 1
pays $r = 0$, with $\gamma = 0.5$. The true values solve $V_0 = 1 + 0.5 V_1$
and $V_1 = 0.5 V_0$, so $V^\pi = [4/3, 2/3]$. The chain spends half its time
in each state, so $\mu = [0.5, 0.5]$. Use one feature, $x = [1, 2]$.

- Best fit: $w = \sum \mu x V / \sum \mu x^2 = (8/3)/5 = 0.5333$, giving
  estimates $[0.533, 1.067]$ and $\overline{\mathrm{VE}} = 0.4$.
- TD fixed point: $A = 0.5 \times 1 \times (1 - 0.5 \times 2) + 0.5 \times 2
  \times (2 - 0.5 \times 1) = 1.5$ and $b = 0.5 \times 1 \times 1 = 0.5$, so
  $w_{\text{TD}} = 1/3$. The estimates are $[0.333, 0.667]$ and
  $\overline{\mathrm{VE}} = 0.5$.

TD's error is 1.25 times the best, inside the bound of $1/(1 - 0.5) = 2$. It
fits the Bellman equation in feature space, not the values themselves.

> [!KEY]
> Linear TD doesn't find the best fit. It finds the TD fixed point
> $A^{-1} b$, whose error on-policy is at most $1/(1 - \gamma)$ times the
> best fit's.

## Solve for the fixed point from data with LSTD

Semi-gradient TD reaches $w_{\text{TD}}$ slowly, one small step at a time,
and its answer depends on the step size schedule. *Least-squares TD*, or LSTD
(Bradtke and Barto, 1996), skips the steps. It estimates $A$ and $b$ from a
batch of transitions and solves once:

$$
\hat A = \sum_{t} x_t \big(x_t - \gamma (1 - \text{term}_t) \, x_{t+1}\big)^\top + \epsilon I, \qquad \hat b = \sum_t r_t \, x_t
$$

Read it as: the same $A$ and $b$, with expectations replaced by sums over
transitions. The next features drop out when the step terminated, and a small
ridge $\epsilon$ on the diagonal keeps the solve from failing. Then solve
$\hat A w = \hat b$.

```python
live = 1.0 - terminated
A = X.T @ (X - gamma * live[:, None] * X2) + reg * np.eye(d)
w = np.linalg.solve(A, X.T @ rewards)
```

With one-hot features, LSTD returns the exact value of the *empirical model*,
the MDP whose transition probabilities and rewards are the observed
frequencies and means. The exercise checks that exactly.

## Watch it diverge: Baird's counterexample

Three ingredients together can make value estimates diverge. Sutton and
Barto call them the *deadly triad*:

- Function approximation, so updating one state changes others.
- Bootstrapping, so targets depend on the current estimates.
- Off-policy training, so the states you update aren't distributed the way
  the target policy visits them.

With any two of the three, instability can be avoided. With all three, it can
appear even in the simplest setting. Baird (1995) built the standard example.
There are seven states. Six upper states
have value $2w_i + w_8$, and the lower state has value $w_7 + 2w_8$. The
target policy always takes the *solid* action, which moves every state to the
lower state with reward 0. So the true value of every state is 0, which
$w = 0$ fits exactly. In code, $w_1$ to $w_8$ are `w[0]` to `w[7]`, and the
lower state is state 6.

The update is the expected, synchronous version of semi-gradient TD, like a
dynamic programming sweep. Each state $s$ is weighted by $\mu(s)$:

$$
w \leftarrow w + \alpha \sum_{s} \mu(s) \big(\gamma \hat V_w(s_{\text{low}}) - \hat V_w(s)\big) \, x(s)
$$

Read it as: every state's target is $\gamma$ times the lower state's
estimate, because the solid action always lands there with no reward. Move
each state's estimate toward its target, weighted by $\mu(s)$.

The behaviour policy picks the solid action with probability 1/7 and the
*dashed* action, which jumps to a random upper state, with probability 6/7.
So its data covers all seven states evenly, and the update weights them
uniformly, $\mu(s) = 1/7$, although the target policy visits only the lower
state.

### Worked example

Start from $w = (1, 1, 1, 1, 1, 1, 10, 1)$ with $\alpha = 0.01$ and
$\gamma = 0.99$. The upper states are worth 3 and the lower state is worth 12.

- Every target is $0.99 \times 12 = 11.88$.
- The upper states have TD error $11.88 - 3 = 8.88$. They all want to rise.
- The lower state has TD error $11.88 - 12 = -0.12$. It wants to fall a
  little.
- All seven share $w_8$. Six large positive errors outvote one small negative
  one, so $w_8$ rises from 1 to 1.0758.
- After the sweep, the upper states are worth 3.127 and the lower state is
  worth 12.151. The lower state went *up*, and its target went up with it.

That repeats every sweep. After 1000 sweeps $\lVert w \rVert$ has grown from
10.3 to 437. The iteration is linear, and its matrix has an eigenvalue of
about $1.0024$, so the growth is exponential. Sutton and Barto show that it
diverges for any positive step size, because the problem is the update's
direction, not its size.

Change only the weighting to the on-policy distribution, all mass on the
lower state, and it's stable. Only the lower state updates. Its value shrinks
by a factor $1 - \alpha(1 - \gamma) \lVert x_{\text{low}} \rVert^2 = 0.9995$
each sweep, from 12 to $12 \times 0.9995^{1000} = 7.28$ after 1000 sweeps.

> [!KEY]
> Function approximation, bootstrapping, and off-policy updates together can
> diverge, even with expected updates and a tiny step size. On-policy
> weighting, in Baird's example, is enough to restore stability.

> [!RECAP]
> - Tile coding turns on one tile per tiling. Shared tiles decide how far an
>   update spreads.
> - Semi-gradient TD moves $w$ by $\alpha \delta_t x(s_t)$ and treats the
>   target as a constant. A terminal step drops the bootstrap term.
> - The TD fixed point solves $\Phi^\top D (I - \gamma P_\pi) \Phi w =
>   \Phi^\top D r_\pi$. One-hot features recover $V^\pi$.
> - LSTD builds the same $A$ and $b$ from samples and solves once.
> - Baird's expected update diverges under uniform weights and decays
>   under on-policy weights.

## Build it step by step

The checks test the functions in this order. `stationary_distribution` is
already written for you.

1. **`tile_features(x, low, high, n_tiles, n_tilings)`.** Returns a float
   vector of length `n_tilings * (n_tiles + 1)`. Clip `x`, compute each
   tiling's tile with `np.floor`, and set one entry per tiling. Common
   mistake: shifting the tilings right instead of left. The hand example
   catches it.
2. **`semi_gradient_td_update(w, x, r, x2, terminated, alpha, gamma)`.**
   Returns the updated weights: two dot products and one scaled vector add.
   Common mistake: bootstrapping from `x2` on a terminal step. A later check
   combines this with your tiles and expects one-trial learning at
   $\alpha = 1/n$.
3. **`td_fixed_point(Phi, P_pi, r_pi, d, gamma)`.** Build $A$ and $b$ with
   `np.diag(d)` and `@`, then call `np.linalg.solve`. Common mistake: writing
   `(I - gamma P) @ D` instead of `D @ (I - gamma P)`. The residual check and
   the bound check on 40 random problems catch it.
4. **`lstd(X, rewards, X2, terminated, gamma, reg)`.** One matrix product
   for $\hat A$ and one for $\hat b$. Common mistake: forgetting to mask
   terminal rows. A later check also runs your TD update over the same data
   and expects it near your fixed point.
5. **`baird_features()`.** Returns the `(7, 8)` matrix. Upper state $i$ has a
   2 in column $i$ and a 1 in column 7. The lower state has a 1 in column 6
   and a 2 in column 7.
6. **`baird_expected_update(w, alpha, gamma, weights)`.** Compute `v = Phi @
   w`, the errors `gamma * v[6] - v`, and step by `alpha * Phi.T @ (weights *
   errors)`. Common mistake: using `v` instead of `v[6]` in the target.

## Cost and scaling

Linear methods scale with the number of features $d$ and transitions $T$, not
with the number of states. The following table compares them:

| Method | Time | Memory | Step size |
|---|---|---|---|
| Tile coding | $O(n)$ per input for $n$ tilings | $n$ active entries | None |
| Semi-gradient TD | $O(d)$ per transition, $O(n)$ with sparse tiles | $O(d)$ | Needed |
| LSTD | $O(T d^2)$ to build $\hat A$, $O(d^3)$ to solve | $O(d^2)$ | None |

LSTD squeezes more out of each sample and has no step size to tune, but its
$d^2$ memory rules it out beyond a few thousand features. The number of tiles
grows exponentially with the input dimension, which is fine for two or three
inputs and hopeless for pixels. The next lesson learns the features instead,
with a neural network.

## Where this breaks

**Off-policy divergence.** Baird's example is the clean version of a real
problem. Gradient-TD methods such as TDC follow the true gradient of a
projected Bellman error, and emphatic TD reweights each update so that the
expected update is stable. Deep Q-learning leans on replay buffers and target
networks instead, which the next lesson builds.

**A loose bound.** With $\gamma = 0.99$ the bound allows 100 times the best
error. With poor features, TD can be much worse than a Monte Carlo fit.

**Bad features.** Two states with the same features always get the same
value, whatever the update rule.

**Ill-conditioned LSTD.** With few samples or rare features, $\hat A$ is
nearly singular and the solve amplifies noise. The ridge term trades a small
bias for stability.

**Non-linear approximators.** The fixed-point analysis is for linear
functions. With neural networks, even on-policy TD can diverge.

## Further reading

- [Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd edition
  (2018)](http://incompleteideas.net/book/the-book-2nd.html). Chapter 9
  covers linear methods, tile coding, and LSTD. Chapter 11 covers Baird's
  counterexample and the deadly triad.
- [Tsitsiklis and Van Roy, *An analysis of temporal-difference learning with
  function approximation* (1997)](https://doi.org/10.1109/9.580874). Proves
  linear TD converges to the fixed point and bounds its error.
- [Bradtke and Barto, *Linear least-squares algorithms for temporal
  difference learning* (1996)](https://doi.org/10.1007/BF00114723).
  Introduces LSTD.
- [Baird, *Residual algorithms: reinforcement learning with function
  approximation* (1995)](https://doi.org/10.1016/B978-1-55860-377-6.50013-X).
  The counterexample, and residual-gradient updates as one fix.
- [Sutton, Maei, and Precup, *Fast gradient-descent methods for
  temporal-difference learning with linear function approximation*
  (2009)](https://doi.org/10.1145/1553374.1553501). The TDC and GTD2
  algorithms, which converge off-policy.
- [van Hasselt et al., *Deep reinforcement learning and the deadly triad*
  (2018)](https://arxiv.org/abs/1812.02648). Measures how often the triad
  causes divergence in deep Q-learning, and which design choices make it rarer.

## The exercise

You write `tile_features`, `semi_gradient_td_update`, `td_fixed_point`,
`lstd`, `baird_features`, and `baird_expected_update`.

The checks verify the tiles against the hand example and one-trial learning
at $\alpha = 1/n$. They test the TD update's arithmetic and terminal masking.
They confirm that `td_fixed_point` recovers $V^\pi$ with one-hot features,
zeroes the projected Bellman residual, and meets the $1/(1 - \gamma)$ bound on
40 random problems. They check that LSTD matches the empirical model exactly
and lands near the fixed point on a long trajectory, as does semi-gradient
TD. Finally, Baird's weights must blow up under uniform weighting and decay
by the exact factor under on-policy weighting.
