---
title: Flow models and GRPO
slug: 45-flow-models-and-grpo
part: "Part 9 — Beyond text"
summary: Turn a flow model's deterministic sampler into an SDE with the same marginals, then fine-tune a 2-D flow with Flow-GRPO.
minutes: 60
gpu: false
objectives:
  - Recover the score of a rectified flow from its velocity, and use it to build an SDE whose marginals match the ODE's at every time.
  - Verify by simulation that the SDE keeps the data distribution for every noise level, and measure how much the sample remembers its starting noise.
  - Implement Flow-GRPO's per-step Gaussian policy, its closed-form KL, group-normalised advantages, and clipped loss.
  - "Explain the two main failure modes: an uncapped noise level near t = 1 and the illusory advantage of near-equal rewards."
prereqs:
  - 37-group-relative-policy-optimization
  - 43-diffusion-models-as-policies
  - 44-preference-tuning-for-diffusion
exercise: 45-flow-models-and-grpo
---

# Flow models and GRPO

Many image and video generators, such as Stable Diffusion 3 and FLUX, are
*flow models*. They sample by following a velocity field from noise to data,
a deterministic path with no randomness after the first draw. That is a
problem for reinforcement learning. A policy gradient needs the probability
of each action, and a deterministic step has none.

The fix is surprising. You can add noise to every step and correct the drift
so that the distribution of samples at every time stays exactly the same.
The sampler still produces the same images on average, but now each step is a
Gaussian with a known density. Flow-GRPO (Liu et al., 2025) uses this to run
the GRPO recipe from Part 8 on flow models: a group of samples per prompt,
rewards normalised within the group, and a clipped per-step ratio.

You build that conversion, check it on an exact Gaussian flow, and then
fine-tune a two-dimensional flow with Flow-GRPO.

## What you already know

- GRPO scores a group of responses to one prompt and uses each reward's
  distance from the group mean, divided by the group's standard deviation,
  as its advantage.
- A diffusion sampler is a policy whose actions are Gaussian steps, and the
  log-probability of a path is a sum of per-step Gaussian log-densities.
- The path KL between two samplers is a sum of per-step KLs, and for
  Gaussians with a shared variance each one is $\lVert \Delta\mu \rVert^2 / (2\sigma^2)$.
- The *score* of a distribution is $\nabla_x \log p(x)$, the direction in
  which the density rises fastest.

## Rectified flow in one screen

A rectified flow blends each data point with noise along a straight line.
Time is continuous, and as in the diffusion lessons a larger $t$ means more
noise: $t = 1$ is pure noise and $t = 0$ is data.

$$
x_t = (1 - t)\, x_0 + t\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
$$

Read it as: at time $t$, a point is a fraction $1 - t$ of the data and a
fraction $t$ of the noise. The model learns a *velocity* $v(x, t)$, the
average of $\epsilon - x_0$ over all the lines that pass through $x$ at time
$t$. Sampling starts at $x_1 \sim \mathcal{N}(0, I)$ and follows
$\mathrm{d}x/\mathrm{d}t = v$ down to $t = 0$. With a positive step size $h$,
the Euler step is $x_{t-h} = x_t - h\, v(x_t, t)$.

When the data is one Gaussian, $\mathcal{N}(m, s^2)$, the velocity has a
closed form. The pair $(x_0, \epsilon)$ and $x_t$ are jointly Gaussian, so
the average is linear in $x$:

$$
v(x, t) = \frac{t - (1 - t) s^2}{(1 - t)^2 s^2 + t^2}\, \big(x - (1 - t) m\big) - m
$$

Read it as: measure how far $x$ sits from the centre of $x_t$'s distribution,
scale that by a time-dependent factor, and subtract the data mean. At $t = 0$
it gives $v = -x$, and at $t = 1$ it gives $v = x - m$. The exercise uses $m = 2$ and $s = 0.5$, so every simulation has an exact answer to hit.

## The score is hiding in the velocity

To add noise without changing the distribution, you need the score of $x_t$.
A rectified flow gives it to you for free. Take the average of the blending
equation over the lines through $x$. Because $v = \mathbb{E}[\epsilon - x_0 \mid x_t]$, two lines of algebra give $\mathbb{E}[\epsilon \mid x_t] = x_t + (1 - t) v$. For Gaussian noise of size $t$, the score is minus the average
noise divided by $t$:

$$
\nabla_x \log p_t(x) = -\frac{x + (1 - t)\, v(x, t)}{t}
$$

Read it as: estimate the noise inside $x$, then point the other way. The
larger the noise, the stronger the pull back toward the data.

### Worked example: the score from the velocity

Take $m = 2$, $s = 0.5$, $t = 0.5$, and $x = 1.5$. Then:

- The factor is $(0.5 - 0.5 \times 0.25) / (0.25 \times 0.25 + 0.25) = 0.375 / 0.3125 = 1.2$.
- The velocity is $1.2 \times (1.5 - 1) - 2 = -1.4$.
- The average noise is $1.5 + 0.5 \times (-1.4) = 0.8$, so the score is
  $-0.8 / 0.5 = -1.6$.

Check it directly. At $t = 0.5$, $x_t$ is Gaussian with mean $0.5 \times 2 = 1$
and variance $0.3125$, so its score is $-(1.5 - 1) / 0.3125 = -1.6$. The two
agree.

## Trade the ODE for an SDE with the same marginals

Adding noise at each step spreads the samples out. To keep the distribution
the same, you also push each sample a little further up the density, toward
where samples are already dense. The score says which way that is. Flow-GRPO
writes the resulting *stochastic differential equation* (SDE), a
differential equation with a noise term, as:

$$
\mathrm{d}x_t = \Big[v(x_t, t) + \frac{\sigma_t^2}{2t}\big(x_t + (1 - t)\, v(x_t, t)\big)\Big]\mathrm{d}t + \sigma_t\, \mathrm{d}w
$$

Read it as: the ODE's drift $v$, plus a score correction of size
$\sigma_t^2 / 2$, plus fresh noise of size $\sigma_t$. Time decreases, so
$\mathrm{d}t$ is negative. That flips the sign of the correction: it becomes a
pull toward high density. The Fokker-Planck equation, the rule for how a
density evolves under drift and noise, shows that the correction cancels the
spreading exactly, so every $\sigma_t$ gives the ODE's marginal $p_t$ at every
time. Song et al. (2021) used the same argument to derive the ODE from a
diffusion SDE.

Discretise it with a positive step $h$ from $t$ to $t - h$. This is the
Euler-Maruyama step, the paper's Equation 9 rewritten for $h > 0$:

$$
x_{t-h} = x_t - h\Big[v + \frac{\sigma_t^2}{2t}\big(x_t + (1 - t) v\big)\Big] + \sigma_t \sqrt{h}\, z, \qquad \sigma_t = a \sqrt{\frac{t}{1 - t}}
$$

Read it as: take a slightly bent Euler step, then add Gaussian noise with
standard deviation $\sigma_t \sqrt{h}$. The multiplier $a$ sets the noise
level, and $a = 0$ gives back the Euler ODE step. The Flow-GRPO paper uses $a = 0.7$. Each step is a Gaussian policy, $\pi_\theta(x_{t-h} \mid x_t) = \mathcal{N}(\mu_\theta, \sigma_t^2 h\, I)$, so the log-probability tools from
the diffusion lessons apply unchanged.

```python
sigma = a * math.sqrt(t / (1 - min(t, t_cap)))
mean = x - h * (v + sigma**2 / (2 * t) * (x + (1 - t) * v))
std = sigma * math.sqrt(h)
```

### Worked example: one SDE step

Take $t = 0.6$, $h = 0.1$, $a = 0.7$, and $x = 1.0$ on the Gaussian flow.
Then:

- The velocity is $1.25 \times (1.0 - 0.8) - 2 = -1.75$.
- The noise level is $\sigma_t = 0.7 \sqrt{1.5} = 0.8573$, so $\sigma_t^2 / (2t) = 0.6125$.
- The bracket is $-1.75 + 0.6125 \times (1.0 + 0.4 \times (-1.75)) = -1.75 + 0.6125 \times 0.3 = -1.5663$.
- The mean is $1.0 + 0.1 \times 1.5663 = 1.1566$, and the standard deviation
  is $0.8573 \sqrt{0.1} = 0.2711$.

The plain Euler step would land at $1.175$. The SDE step lands a little short
of it on average and adds noise to make up the spread.

> [!WARNING]
> $\sigma_t$ is infinite at $t = 1$, where every sampler starts. The Flow-GRPO
> code replaces the $t$ in the denominator by the grid's second time, and the
> exercise does the same with `t_cap = grid[1]`. A cap much closer to 1 makes
> the first step's variance huge and the Euler-Maruyama error large. On a
> 250-step grid, capping at $1 - 10^{-4}$ instead of $0.996$ gives a final
> standard deviation of 1.007 for $a = 0.7$, not 0.5. Correct formulas
> fail the check because of the step size alone.

## What the noise level changes

Run the sampler on the Gaussian flow with the same 40,000 starting draws, a
250-step grid, and four noise levels. The results show what $a$ does and
doesn't change:

| $a$ | Final mean | Final std | Correlation with the starting noise |
|---|---|---|---|
| 0 | 1.996 | 0.500 | 1.000 |
| 0.3 | 2.000 | 0.498 | 0.701 |
| 0.7 | 1.998 | 0.497 | 0.141 |
| $\sqrt{2}$ | 1.998 | 0.502 | −0.002 |

Every row hits the data distribution $\mathcal{N}(2, 0.5^2)$ within the
simulation's error. What changes is *memory*: with $a = 0$ the sample is a
fixed function of its starting noise, and as $a$ grows it forgets where it
started.

> [!KEY]
> Every noise level $a$ produces the same distribution of samples. Only the
> coupling between starting noise and final sample changes, and only $a > 0$
> gives each step a density.

Memory matters for fine-tuning with a KL penalty. Adjoint Matching
(Domingo-Enrich et al., 2024) shows that KL-regularised fine-tuning over
paths reaches the intended target, $p^\star(x) \propto p_{\text{ref}}(x) \exp(r(x))$, only when the reference process is *memoryless*: the final
sample independent of the starting noise. Otherwise the target is tilted by
an extra factor that depends on the starting noise. For rectified flow their
memoryless schedule is $\sigma(t) = \sqrt{2(1 - t)/t}$ in their time
convention, where noise sits at $t = 0$. Flip the time to Flow-GRPO's
convention and it becomes $\sqrt{2} \sqrt{t / (1 - t)}$, which is Flow-GRPO's
$\sigma_t$ with $a = \sqrt{2}$.

The paper's Proposition 1 is broader than that one schedule. Its condition
holds for any $a > 0$ here, because $\sigma_t$ grows without bound near $t = 1$ and erases the starting noise there. The erasure is slow, though. For
Gaussian data the leftover correlation shrinks roughly like $(1 - t_{\text{cap}})^{a^2/2}$. Computed exactly rather than by simulation, at $a = 0.7$ it is 0.23 on a 10-step grid, 0.13 on 250 steps, and still 0.03 on
100,000 steps. At $a = \sqrt{2}$ it is gone on
any grid. How much a coarse grid's leftover memory biases the fine-tuned
target in practice is not settled.

## Flow-GRPO

With a Gaussian policy at every step, Flow-GRPO follows GRPO closely. For
each prompt it samples a group of $G$ images, computes group-normalised
advantages, and maximises:

$$
\mathcal{J}(\theta) = \mathbb{E}\Big[\frac{1}{G}\sum_{i=1}^{G} \frac{1}{T}\sum_{t} \Big(\min\big(\rho_t^i \hat A^i,\ \operatorname{clip}(\rho_t^i, 1 - \varepsilon, 1 + \varepsilon)\, \hat A^i\big) - \beta\, D_{\text{KL}, t}^i\Big)\Big], \qquad \hat A^i = \frac{R^i - \operatorname{mean}(R)}{\operatorname{std}(R)}
$$

Read it as the DDPO loss with two changes: each image is compared with the
others for its prompt, and a KL penalty to the reference is subtracted at
every step. Here $\rho_t^i$ is the per-step probability ratio,
$D_{\text{KL}, t}^i$ is the KL between the policy's and the reference's step
at image $i$'s state, and every step of image $i$ shares the advantage
$\hat A^i$.

Both steps share $x_t$ and variance, so the KL has a closed form. Their means
differ by $h\big(1 + \frac{\sigma_t^2 (1 - t)}{2t}\big) \Delta v$, where $\Delta v = v_\theta - v_{\text{ref}}$. Dividing its square by $2\sigma_t^2 h$ gives the paper's
formula:

$$
D_{\text{KL}} = \frac{h}{2}\Big(\frac{\sigma_t (1 - t)}{2t} + \frac{1}{\sigma_t}\Big)^2 \lVert \Delta v \rVert^2
$$

Read it as: a step's KL is its squared velocity gap times a weight that
depends only on the time, the step size, and the noise level. At $t = 0.6$,
$h = 0.1$, and $a = 0.7$, the bracket is $0.8573 \times 0.4 / 1.2 + 1 / 0.8573 = 1.4522$. With $\lVert \Delta v \rVert = 0.2$, the KL is $0.05 \times 2.1089 \times 0.04 = 0.004218$.

```python
coef = h / 2 * (sigma * (1 - t) / (2 * t) + 1 / sigma) ** 2
kl = coef * ((v_theta - v_ref) ** 2).sum(-1)
```

### Worked example: group advantages

A group of four images scores $(0.2, 0.5, 0.9, 0.4)$. The mean is 0.5 and
the population standard deviation is 0.2550, so the advantages are
$(-1.177, 0, 1.569, -0.392)$. Now take $(0.801, 0.802, 0.800, 0.803)$, a group
where the reward model can barely tell the images apart. The standard
deviation is 0.00112, and the advantages are $(-0.447, 0.447, -1.342, 1.342)$: as large as for a group that really differed. Pref-GRPO (Wang et
al., 2025) calls this the *illusory advantage*. Tiny, meaningless score gaps
get amplified into full-sized training signals. Adding a floor of 0.01 to the
standard deviation shrinks them below 0.14.

The paper reports these settings and results:

- Noise level $a = 0.7$ and group size $G = 24$.
- Training rollouts with only 10 denoising steps, while evaluation uses 40,
  because rollouts only have to rank images within a group.
- $\beta = 0.04$ for GenEval and text rendering, and 0.01 for PickScore.
- On SD3.5-M, GenEval accuracy rises from 63% to 95%, and visual text
  rendering accuracy from 59% to 92%.

Flow-GRPO starts each rollout in a group from different noise. DanceGRPO
(Xue et al., 2025) makes the opposite choice. It shares the starting noise
within a group, which it reports prevents reward hacking and instability in
video, and it trains on a random subset of the steps. MixGRPO (Li et al., 2025) makes only a
sliding window of steps stochastic and trained, and samples the rest with the
ODE, for about half of DanceGRPO's training time.

On the exercise's two-dimensional toy, four prompts each ask for one of four
bumps. The reward is $\exp(-\lVert x_0 - c \rVert^2)$ for the asked-for bump
$c$. With $\beta = 0.01$, 16 samples per prompt, and 10 steps, the mean reward
climbs from about 0.2 to about 0.98 in 80 iterations, while the path KL grows
to about 10. With $\beta = 0.3$ the path KL stays below 0.6 and the reward
barely moves. The penalty is doing its job, and its weight is the knob.

> [!RECAP]
> - The score of a rectified flow is $-(x + (1 - t) v) / t$.
> - The SDE step's mean is $x - h[v + \sigma_t^2 (x + (1 - t) v) / (2t)]$
>   and its standard deviation is $\sigma_t \sqrt{h}$, with $\sigma_t = a \sqrt{t / (1 - \min(t, t_{\text{cap}}))}$.
> - Every $a$ keeps the marginals. Larger $a$ forgets the starting noise
>   faster.
> - Flow-GRPO is GRPO over denoising steps, with one group-normalised
>   advantage per image and a closed-form per-step KL.

## Build it step by step

The checks test these functions in order.

1. **`score_from_velocity(x, t, v)`.** One line. The check feeds it the exact
   velocity and compares with the exact Gaussian score.
2. **`gaussian_velocity(x, t, m, s)`.** The closed form. The check confirms
   the limits at $t = 0$ and $t = 1$, and the *continuity equation*
   $\partial_t p + \partial_x (p v) = 0$, which says probability moves with the
   flow and is never created. Common mistake: $(1 - t)^2$ written as $(1 - t^2)$.
3. **`sde_step(x, t, h, v, a, t_cap)`.** Return the mean and the standard
   deviation as a float. Put `min(t, t_cap)` only in the denominator of
   $\sigma_t$. Common mistake: capping the numerator too.
4. **`flow_step_kl(v_theta, v_ref, t, h, a, t_cap)`.** The closed form, with
   the squared norm summed over the last axis. The check compares it with
   $\lVert \Delta\mu \rVert^2 / (2\,\mathrm{std}^2)$ from your `sde_step`.
5. **`sample_sde(velocity_fn, x1, grid, a, generator, return_path=False)`.**
   Loop over consecutive grid times with `h = t - t_next` and `t_cap =
   grid[1]`. Draw noise with `torch.randn(x.shape, generator=generator)`.
   Common mistake: `h = t_next - t`, which is negative and makes
   `math.sqrt(h)` fail.
6. **`group_advantages(rewards, eps)`.** Normalise along the last axis with
   `std(unbiased=False)`. Common mistake: the sample standard deviation,
   which gives $(-1.019, 0, 1.359, -0.340)$ on the first worked group.
7. **`flow_logprob(policy, xs, c, grid, a)`.** Detach `xs`, recompute each
   step's mean from `policy`, sum the log-density over both coordinates, and
   compute the KL against `reference_velocity` at the same state.
8. **`flow_grpo_loss(...)`.** The clipped surrogate from the diffusion
   lessons, plus `beta * kl.mean()`.
9. **`flow_grpo_train(...)`.** Build prompts with
   `torch.arange(N_PROMPTS).repeat_interleave(group_size)`, sample paths
   under `torch.no_grad()`, reshape rewards to `(N_PROMPTS, group_size)` for
   the advantages, and take `epochs` Adam steps.

## Cost and scaling

Flow-GRPO's cost is dominated by sampling. The main terms, for a model with
$T$ steps and groups of $G$:

| Cost | Grows with | Notes |
|---|---|---|
| Rollouts | $G \times T$ network calls per prompt | Flow-GRPO cuts $T$ to 10 for training. |
| Old log-probabilities | Free | Recorded during sampling. |
| Update | $T$ calls of policy and reference per image, per epoch | MixGRPO trains only a window of steps. |
| Memory | As little as one step's activations | Each step's loss term can be backpropagated on its own. |

The SDE step costs the same network call as an ODE step, plus a few
elementwise operations.

## Where this breaks

**The noise level near t = 1.** An uncapped $\sigma_t$ divides by zero at
the first step. A cap too close to 1 leaves a correct formula with a wrong
answer, because one step with a huge variance has a large discretisation
error. Fix the grid, and cap at its second time.

**Noise artifacts.** SDE sampling can leave noise artifacts in the images,
which hurt reward learning. Coefficients-Preserving Sampling (Wang and Yu,
2025) traces them to excess injected noise and proposes a stochastic step that
removes them.

**The illusory advantage.** Group normalisation turns tiny, noisy reward gaps
into full-sized advantages. A floor on the standard deviation or pairwise win
rates in place of raw scores both help.

**Mismatched settings.** The ratio at the first update must be exactly 1. If
the log-probabilities are recomputed with a different $a$, a different grid,
or a different guidance scale than the rollout used, it isn't, and the clip
acts on noise.

## Further reading

The following papers cover the SDE conversion, Flow-GRPO, and its
successors:

- [Liu et al., *Flow-GRPO* (2025)](https://arxiv.org/abs/2505.05470) converts
  the flow ODE to an SDE and runs GRPO with a closed-form KL.
- [Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations* (2021)](https://arxiv.org/abs/2011.13456)
  shows that the ODE and a family of SDEs share marginals.
- [Domingo-Enrich et al., *Adjoint Matching* (2024)](https://arxiv.org/abs/2409.08861)
  defines memoryless noise schedules and explains why fine-tuning needs them.
- [Xue et al., *DanceGRPO* (2025)](https://arxiv.org/abs/2505.07818) applies
  the recipe across image and video models, with shared starting noise.
- [Li et al., *MixGRPO* (2025)](https://arxiv.org/abs/2507.21802) mixes ODE and
  SDE steps with a sliding window.
- [Wang et al., *Pref-GRPO* (2025)](https://arxiv.org/abs/2508.20751)
  identifies the illusory advantage and uses pairwise win rates.
- [Wang and Yu, *Coefficients-Preserving Sampling* (2025)](https://arxiv.org/abs/2509.05952)
  traces image artifacts to the SDE's excess noise.

Open questions that researchers are working on:

- How much does leftover memory of the starting noise, on a coarse grid,
  bias KL-regularised fine-tuning?
- Which denoising steps deserve credit for a final reward, and can branching
  rollouts estimate it cheaply?
- Can flow models be fine-tuned without any sampler densities, from the
  forward process alone?

## The exercise

You build the SDE conversion and Flow-GRPO. Write `score_from_velocity` and
`gaussian_velocity` for the one-dimensional Gaussian flow, `sde_step` and
`flow_step_kl` for one Gaussian step and its KL, and `sample_sde` for the
sampler. Then write `group_advantages`, `flow_logprob`, `flow_grpo_loss`, and
`flow_grpo_train` to train the two-dimensional prompted flow.

The checks compare each piece with an exact answer: the Gaussian score, the
continuity equation, the worked step and KL, and both worked groups. The
sampler must hit $\mathcal{N}(2, 0.5^2)$ for all four noise levels, with
memory falling as $a$ grows. Flow-GRPO must lift the mean reward above 0.85,
while $\beta = 0.3$ holds the path KL below 1.5.
