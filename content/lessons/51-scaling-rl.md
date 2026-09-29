---
title: Scaling reinforcement learning
slug: 51-scaling-rl
part: "Part 10 — Safety and the frontier"
summary: Fit ScaleRL's sigmoid to compute curves, then measure a network's loss of plasticity and restore it with resets.
minutes: 60
gpu: false
objectives:
  - State ScaleRL's sigmoidal compute-performance curve and say what its asymptote, midpoint, and exponent each control.
  - Fit the curve with numpy by profiling over the asymptote, and show why runs that stop before the midpoint can't pin the asymptote down.
  - Measure plasticity loss as a rising fraction of dead units, a falling effective rank, and a rising loss on new targets.
  - Restore plasticity with ReDo resets that keep the network's outputs unchanged.
prereqs:
  - 12-deep-q-networks
  - 31-evaluating-rl
  - 37-group-relative-policy-optimization
exercise: 51-scaling-rl
---

# Scaling reinforcement learning

A large training run can cost more than everything that came before it. You
can't afford to find out at the end that the recipe stalls. In supervised
learning, loss falls along a smooth power law in compute, so people plan big
runs from small ones. Reinforcement learning long looked too noisy for that.
Returns jump around, and changing a hyperparameter can change everything.

That picture changed between 2023 and 2025. With a stable recipe, RL
performance also follows smooth curves in compute, so small runs can predict
large ones. But RL has a problem supervised learning mostly doesn't. The data
changes as the policy improves, so the network must keep learning new
targets. Networks trained this way slowly lose the ability to learn at all.

This lesson covers both halves. You fit the sigmoidal compute curve from
ScaleRL, and you see when extrapolating it is safe and when it isn't. Then
you train a small network on a sequence of targets, watch its *plasticity*
drain away, and restore it with resets.

## What you already know

- Least squares fits a straight line with `np.linalg.lstsq`. A curve that
  becomes a line after a change of variables can be fit the same way.
- A ReLU unit outputs $\max(0, z)$. If $z < 0$ for every input, the unit
  outputs zero everywhere, and no gradient reaches its incoming weights.
- The singular values of a matrix measure how many independent directions
  its rows span.
- DQN's replay buffer and target network exist because the regression
  targets keep moving. You built them in *Deep Q-networks*.
- A single run proves little. *Evaluating reinforcement learning* showed how
  to report results across seeds.

## Fit a sigmoid to compute

Khatri, Madaan, and others (2025) ran more than 400,000 GPU-hours of RL on
language models and tracked the pass rate on held-out validation problems against
training compute. They found that the curves have an S shape in log compute:
a slow start, steady gains, and a ceiling. Their fit, which they call
ScaleRL's compute-performance law, is

$$
R_C - R_0 = \frac{A - R_0}{1 + (C_{\text{mid}} / C)^{B}} .
$$

Read it piece by piece:

- $R_0$ is the reward before training, and $R_C$ is the reward after
  compute $C$.
- $A$ is the *asymptote*, the reward the recipe approaches as compute grows
  without limit.
- $C_{\text{mid}}$ is the compute at which half of the total gain $A - R_0$
  is reached.
- $B > 0$ is an efficiency exponent. A larger $B$ makes the climb steeper
  around $C_{\text{mid}}$.

In numpy it's one line:
`R = R0 + (A - R0) / (1 + (C_mid / C) ** B)`.

Their main finding is about which knobs move which parameter. A few choices
moved the asymptote $A$. The loss type did, and computing the final logits in
32-bit floating point raised $A$ from 0.52 to 0.61. Many other details, such
as loss aggregation, advantage normalisation, the curriculum, and the
off-policy algorithm, mostly changed efficiency, $B$ and $C_{\text{mid}}$. A stable recipe fitted on
smaller runs predicted a single run scaled to 100,000 GPU-hours.

### Worked example

Take $R_0 = 0.1$, $A = 0.6$, $B = 1.5$, and $C_{\text{mid}} = 1{,}000$
GPU-hours. At $C = 4{,}000$, the ratio is $(1000/4000)^{1.5} = 0.125$, so
$R = 0.1 + 0.5 / 1.125 = 0.544$. Doubling to $C = 8{,}000$ gives
$0.125^{1.5} = 0.0442$ and $R = 0.579$. Ten times more again, $C = 80{,}000$,
gives $R = 0.599$. The last 0.02 of the gain costs ten times the compute of
everything before it.

## Fit it with numpy

The curve has three unknowns, $A$, $B$, and $C_{\text{mid}}$, with $R_0$
known from the starting checkpoint. It isn't linear in them, but a change of
variables makes it nearly so. Fix $A$ and rearrange:

$$
\log\!\left(\frac{A - R_0}{R_C - R_0} - 1\right) = B \log C_{\text{mid}} - B \log C .
$$

Read it as: for the right $A$, this transformed reward is a straight line in
$\log C$, with slope $-B$ and intercept $B \log C_{\text{mid}}$. In the worked
example at $C = 4{,}000$, the left side is $\log(0.5 / 0.444 - 1) = \log
0.125 = -2.079$. The right side is $1.5 \times (\log 1000 - \log 4000) =
-2.079$.

So you *profile* over $A$. For each candidate $A$ on a grid, fit the line,
recover $B$ and $C_{\text{mid}}$, and score the resulting curve by its
squared error on the raw rewards. Keep the best:

```python
for A in A_grid:
    keep = (R > R0) & (R < A)
    y = np.log((A - R0) / (R[keep] - R0) - 1)
    X = np.stack([np.ones(keep.sum()), np.log(C[keep])], axis=1)
    (intercept, slope), *_ = np.linalg.lstsq(X, y, rcond=None)
    # B = -slope, C_mid = exp(intercept / B), then score on all runs
```

The mask matters. The logarithm needs $R_0 < R < A$, and noisy early runs can
dip below $R_0$. Score on *all* runs, though, so that candidates can't win by
throwing data away.

On 16 runs from $C_{\text{mid}}/8$ to $8\,C_{\text{mid}}$, with noise of
standard deviation 0.01, the fit recovers $A$ within 0.02 and
$C_{\text{mid}}$ within 9 percent across 20 noise draws. $B$ is less
reliable, off by up to 50 percent, because the logarithm magnifies noise on
the flat early runs.

> [!WARNING]
> Don't trust an asymptote fitted to runs that haven't passed the midpoint.
> With only the runs up to $C_{\text{mid}}/2$, the same fit returns $A$
> anywhere on its grid, from 0.3 to 1.0, depending on the noise draw.

The reason is visible in the curve's shape. Before the midpoint, the sigmoid
looks like a power law, $R_C - R_0 \approx (A - R_0)(C / C_{\text{mid}})^B$.
A larger $A$ with a later midpoint traces almost the same early points.
Only the bend tells them apart.

## Know what else scales

The sigmoid is one of several scaling results for RL. Hilton, Tang, and
Schulman (2023) found power laws in model size and environment interactions
for single-agent RL, with exponents that vary widely between environments.
Rybkin et al. (2025) found that value-based methods such as SAC scale
predictably once you treat the *updates-to-data ratio* as a knob. That ratio
counts gradient steps per environment step, and the best batch size and
learning rate both fall as it rises. In every case the prediction needs a
stable recipe first. An unstable run doesn't follow any curve.

> [!KEY]
> A stable recipe's compute curve is predictable, but its ceiling is only
> visible after the bend. Most recipe changes move the efficiency, and only
> some move the ceiling.

## Measure plasticity loss

Why would a network stop learning? An RL agent's regression targets change
constantly: new data arrives, the policy improves, and the value targets
move. Train a network on one target after another, and it gets steadily
worse at fitting the next one. This is *loss of plasticity*. Dohare et al.
(2024) showed it in standard deep learning on long sequences of tasks.
Nikishin et al. (2022) showed a related *primacy bias* in RL: agents overfit
to early experience and fit later data badly.

The exercise makes this concrete. A ReLU network with two hidden layers of
64 units learns random targets for 512 fixed inputs. Each task gets fresh
targets and 150 Adam steps. You measure plasticity three ways:

- **Loss on the new target.** The final mean squared error divided by the
  target's variance. This is the direct measure: can the network still fit
  what you give it?
- **Dead units.** A ReLU unit is *dead* when it outputs zero on every input.
  Sokar et al. (2023) generalise this to *dormant* units, whose average
  activation is small relative to the layer's.
- **Effective rank.** How many directions the hidden features really use.

Sokar et al. score unit $i$ in a layer of $H$ units as

$$
s_i = \frac{\mathbb{E}_x \lvert h_i(x) \rvert}{\frac{1}{H} \sum_{k=1}^{H} \mathbb{E}_x \lvert h_k(x) \rvert} ,
$$

and call it $\tau$-dormant when $s_i \le \tau$. Read it as: the unit's mean
absolute activation, divided by the layer's average of the same quantity.
With $\tau = 0$ it counts dead units. In numpy, `a = np.abs(h).mean(axis=0)`
and then `np.mean(a / a.mean() <= tau)`. For four units with mean absolute
activations $[0, 1, 2, 5]$, the layer average is 2, the scores are $[0, 0.5,
1, 2.5]$, and $\tau = 0.5$ marks half the units dormant.

Kumar et al. (2020) define the effective rank of a feature matrix $\Phi$,
one row per input, from its singular values $\sigma_1 \ge \sigma_2 \ge
\cdots$:

$$
\operatorname{srank}_\delta(\Phi) = \min \Big\{ k : \frac{\sum_{i=1}^{k} \sigma_i}{\sum_{i} \sigma_i} \ge 1 - \delta \Big\} .
$$

Read it as: the fewest singular values that hold all but a $\delta$ share of
their total. With singular values $[4, 3, 2, 1]$, the cumulative shares are
$[0.4, 0.7, 0.9, 1.0]$. So $\delta = 0.01$ gives rank 4, and $\delta = 0.15$
gives rank 3. In numpy, take `np.linalg.svd(F, compute_uv=False)`, form the
normalised cumulative sum, and use `np.searchsorted`.

The following table shows one run with seed 0, sampled every third task:

| Task | Loss on new target | Dead fraction | Effective rank |
|---|---|---|---|
| 1 | 0.023 | 0.22 | 49 |
| 7 | 0.014 | 0.36 | 39 |
| 13 | 0.427 | 0.55 | 28 |
| 19 | 0.918 | 0.89 | 7 |

By task 19 the network can barely fit a new target: it explains less than 10
percent of the variance. Nine units in ten of the second layer are dead, and
the features span 7 directions out of 64. Lyle et al. (2023) found that dead
units aren't the whole story, since plasticity can drop even with no
saturated units. In this small network, though, the two track each other
closely.

## Restore it with resets

The oldest fix is to reset. Nikishin et al. reset an agent's last layers
periodically and kept the replay buffer, so the fresh layers relearn from
all the data. A full reset throws away everything the network knew, so
cheaper, targeted versions exist:

- **ReDo** (Sokar et al., 2023) reinitialises only the dormant units.
- **Shrink and perturb** (Ash and Adams, 2020) scales every weight toward
  zero and adds a little noise.
- **Continual backpropagation** (Dohare et al., 2024) keeps reinitialising a
  small fraction of the least useful units.

ReDo's trick is to change nothing the network computes. For each dead unit,
it draws fresh incoming weights and bias from the initial distribution, and
sets the unit's outgoing weights to zero. The unit now responds to inputs
again, but it contributes nothing to the next layer until training grows
its outgoing weights. In PyTorch, with `dead` a boolean mask over units:

```python
with torch.no_grad():
    net[0].weight[dead] = fresh[0].weight[dead]
    net[0].bias[dead] = fresh[0].bias[dead]
    net[2].weight[:, dead] = 0.0
```

With ReDo between tasks, the same seed-0 run fits task 19 to a normalised
loss of 0.013, and no task does worse than 0.07. The dead fraction stays at
or below 0.22, and the effective rank stays between 46 and 57.

> [!TIP]
> Resetting only the output layer doesn't restore plasticity here: the loss
> on task 19 is still 0.69. The dead units sit in the hidden layers, and a
> fresh output layer still reads from them.

> [!KEY]
> Plasticity loss shows up as dead units and collapsing feature rank, and as
> a rising loss on each new target. Reinitialising the dead units restores
> the ability to fit.

> [!RECAP]
> - ScaleRL's curve is $R_0 + (A - R_0) / (1 + (C_{\text{mid}}/C)^B)$.
> - For a fixed $A$, $\log((A - R_0)/(R - R_0) - 1)$ is linear in $\log C$.
>   Profile over $A$ and score every candidate on the raw rewards.
> - Runs that stop before $C_{\text{mid}}$ leave $A$ undetermined.
> - Measure plasticity with the loss on new targets, the dormant fraction
>   $s_i \le \tau$, and $\operatorname{srank}_\delta$.
> - ReDo reinitialises dead units' incoming weights and zeroes their
>   outgoing weights, so the outputs don't change.

## Build it step by step

The checks test the functions in this order. `make_mlp` is given.

1. **`scalerl_curve(C, R0, A, B, C_mid)`.** One line of numpy. Common
   mistake: writing $(C / C_{\text{mid}})^B$, which flips the curve.
2. **`fit_scalerl(C, R, R0, A_grid)`.** Returns `(A, B, C_mid)`. Loop over
   the grid, mask, fit the line with `np.linalg.lstsq`, and score with
   `scalerl_curve`. Common mistake: scoring only the kept runs, which rewards
   candidates that discard data.
3. **`effective_rank(features, delta)`.** Singular values, normalised
   cumulative sum, and `np.searchsorted(share, 1 - delta) + 1`. Common
   mistake: an off-by-one from `searchsorted`'s zero-based index.
4. **`dormant_fraction(activations, tau)`.** Mean absolute activation per
   column, divided by its mean. Common mistake: averaging over units instead
   of inputs. Check your axis.
5. **`hidden_features(net, x)`.** Run the first four modules of the network
   under `torch.no_grad()`, keeping the output after each ReLU.
6. **`train_task(net, x, y, steps, lr)`.** A fresh `torch.optim.Adam` and a
   full-batch loop. Return the final loss as a float.
7. **`redo_reset(net, x, seed)`.** Find the dead units of both layers first,
   then copy fresh incoming weights and biases from `make_mlp(seed)`. Common
   mistake: forgetting to zero the outgoing columns, which changes the
   network's outputs. The check compares the outputs before and after.
8. **`continual_run(seed, ...)`.** Draw `x` and each `y` from one
   generator in the documented order, reset before each task after the
   first, and record the three measures.

## Cost and scaling

The sigmoid fit is trivial. Each candidate $A$ is one tiny least-squares
solve, so a grid of $g$ values over $m$ runs costs $O(gm)$. The expensive
part is producing the runs, and the fit is only worth doing on runs that
passed the midpoint.

The plasticity experiment costs one training run. The following table
summarises what each measure costs on a network of width $H$ and $n$ inputs:

| Measure | Time | Notes |
|---|---|---|
| Dormant fraction | $O(nH)$ | One forward pass |
| Effective rank | $O(nH^2)$ | One SVD of an $n \times H$ matrix |
| Loss on a new target | One full training phase | The measure that matters |
| ReDo reset | $O(H \cdot \text{fan-in})$ | Rewrites only the dead units' weights |

In large RL systems, people track the dormant fraction and effective rank
during training because they're cheap, and treat them as early warnings.
Whether they predict trouble in LLM-scale RL is an open question.

## Where this breaks

**Extrapolation from early runs.** As the warning showed, the asymptote is
unconstrained before the bend. Report fitted ceilings with bootstrap
intervals over seeds, as in *Evaluating reinforcement learning*, and only
from runs past $C_{\text{mid}}$.

**Recipe changes that look like ceiling changes.** A change that speeds up
early learning can look like a higher ceiling at a fixed budget. Compare
fitted $A$ values, not rewards at one compute level.

**Hyperparameters tied to scale.** Rybkin et al. found that the best batch
size and learning rate depend on the updates-to-data ratio. Comparing model
sizes without retuning them compares recipes, not sizes.

**Resets that discard the data.** A reset recovers because the agent relearns
from its replay buffer. Resetting the buffer too throws away the evidence
the fresh weights need.

**Measures that miss the problem.** Dead units are easy to count, but
plasticity can fall without them. The loss on a new target is the measure
to trust.

## Further reading

- [The art of scaling reinforcement learning compute for LLMs](https://arxiv.org/abs/2510.13786)
  (Khatri, Madaan, et al., 2025). The sigmoidal fit, the ablations that
  separate the asymptote from efficiency, and the ScaleRL recipe.
- [Value-based deep RL scales predictably](https://arxiv.org/abs/2502.04327)
  (Rybkin et al., 2025). Data and compute scaling in the updates-to-data
  ratio for SAC-style methods.
- [Scaling laws for single-agent reinforcement learning](https://arxiv.org/abs/2301.13442)
  (Hilton, Tang, and Schulman, 2023). Power laws in model size and
  interactions, through an "intrinsic performance" measure.
- [The primacy bias in deep reinforcement learning](https://arxiv.org/abs/2205.07802)
  (Nikishin et al., 2022). Periodic resets of part of the agent, with the
  replay buffer kept.
- [The dormant neuron phenomenon in deep reinforcement learning](https://arxiv.org/abs/2302.12902)
  (Sokar et al., 2023). The dormant score and the ReDo reset.
- [Loss of plasticity in deep continual learning](https://doi.org/10.1038/s41586-024-07711-7)
  (Dohare et al., 2024). Plasticity loss on long task sequences, and
  continual backpropagation.
- [Implicit under-parameterization inhibits data-efficient deep reinforcement learning](https://arxiv.org/abs/2010.14498)
  (Kumar et al., 2020). The effective rank and its collapse under
  bootstrapping.
- [Understanding plasticity in neural networks](https://arxiv.org/abs/2303.01486)
  (Lyle et al., 2023). Plasticity loss tied to loss-landscape curvature,
  even without saturated units.

Researchers are working on open questions such as the following:

- What sets the asymptote $A$: the base model, the diversity of the data,
  the verifier, or the context length? Can a cheap probe predict it?
- Does plasticity loss happen in long LLM RL runs, and is entropy collapse
  partly a plasticity effect?
- Does environment diversity follow its own scaling law in agentic RL?

## The exercise

You fit compute curves and then measure and repair plasticity. Write
`scalerl_curve` for the sigmoid and `fit_scalerl` to fit it by profiling
over $A$. Then write `effective_rank` and `dormant_fraction` for the two
feature measures, `hidden_features` and `train_task` for the network, and
`redo_reset` for the ReDo reset. Finish with `continual_run`, which trains
one network on 20 tasks with or without resets and records the loss, dead
fraction, and effective rank after each.

The checks test `scalerl_curve` at hand values and `fit_scalerl` on
noise-free runs, where it must recover all three parameters exactly. On 10
noise draws, fits over all 16 runs must agree on $A$, while fits on runs up to
$C_{\text{mid}}/2$ must scatter by at least 0.3. The feature measures must
match hand-built matrices, and `redo_reset` must revive dead units without
changing the network's outputs. Without resets, the late-task loss must be
at least three times the early one, with dead units rising by 0.3 and rank
falling by 20. With resets, every task must fit to a normalised loss under
0.15.
