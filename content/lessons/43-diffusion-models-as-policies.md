---
title: Diffusion models as policies
slug: 43-diffusion-models-as-policies
part: "Part 9 — Beyond text"
summary: Treat each denoising step as an action, then fine-tune a diffusion model on a reward with DDPO and with reward backpropagation.
minutes: 60
gpu: false
objectives:
  - Write the denoising MDP, with its states, actions, transitions, and a reward that arrives only at the last step.
  - Compute the Gaussian log-probability of each denoising step, and explain why the whole trajectory has a density when the final sample does not.
  - Implement DDPO's clipped per-step surrogate and train a toy diffusion model toward a reward.
  - Backpropagate a differentiable reward through the sampler, truncate it DRaFT-style, and compare its gradient with DDPO's.
prereqs:
  - 15-policy-gradient-theorem
  - 19-proximal-policy-optimization
  - 33-language-models-as-policies
exercise: 43-diffusion-models-as-policies
---

# Diffusion models as policies

A text-to-image model is trained to copy the pictures in its training set.
Nothing in that training says which pictures people like or which ones match
the prompt. When you care about a score like that, you fine-tune toward it,
the way Part 8 fine-tuned language models toward a reward.

A diffusion model starts from pure noise and cleans it up over many small
steps. This lesson shows that those steps are actions. Each one is a random
draw from a Gaussian whose mean the network chooses. Once sampling is a
sequence of actions, the tools from Part 4 apply: REINFORCE, baselines, and
PPO's clipped ratio.

You train a one-dimensional diffusion model two ways. DDPO uses the policy
gradient and needs only a reward score. Reward backpropagation differentiates
the reward through the whole sampler. The two gradients agree on average but
fail in different places.

## What you already know

- The policy gradient theorem says the gradient of expected reward is
  $\mathbb{E}[\nabla_\theta \log \pi_\theta(a \mid s)\, A]$, where the
  *advantage* $A$ says how much better an outcome was than a baseline.
- PPO reuses a batch of samples for a few gradient steps and clips the
  probability ratio $\pi_\theta / \pi_{\theta_\text{old}}$ to stay close to
  the policy that sampled.
- A language model is a policy over tokens, and a response's probability is
  the product of its token probabilities.
- The log-density of a Gaussian is
  $-\tfrac12 ((x - \mu)/\sigma)^2 - \log \sigma - \tfrac12 \log 2\pi$. It can
  be positive, because a density can exceed one.
- Autograd differentiates any chain of tensor operations, even a sampling
  loop.

## How a diffusion model draws a sample

A diffusion model learns to undo noise. During training you take a clean
example $x_0$, pick a noise level $t$ between 1 and $T$, and blend the
example with Gaussian noise $\epsilon$:

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon
$$

Read it as: keep a fraction $\sqrt{\bar\alpha_t}$ of the signal and fill the
rest with noise. The numbers $\bar\alpha_t$ fall from near 1 at $t = 1$ to
near 0 at $t = T$, so $x_T$ is almost pure noise. They come from a *noise
schedule* $\beta_1, \dots, \beta_T$ through $\alpha_t = 1 - \beta_t$ and
$\bar\alpha_t = \alpha_1 \alpha_2 \cdots \alpha_t$. A network
$\epsilon_\theta(x_t, t)$ learns to predict the noise $\epsilon$ that was
mixed in.

To sample, you run the process backward. Start from $x_T \sim \mathcal{N}(0, 1)$ and take $T$ steps, each from $x_t$ to $x_{t-1}$. The DDPM
sampler of Ho, Jain, and Abbeel (2020) makes each step a Gaussian:

$$
x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z, \qquad
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\, \epsilon_\theta(x_t, t)\Big)
$$

Read it as: subtract a scaled copy of the predicted noise, rescale, then add
fresh noise $z \sim \mathcal{N}(0, 1)$ of size $\sigma_t$. The exercise uses
$\sigma_t^2 = \beta_t$, one of the two choices in the DDPM paper. In code, one
step is two lines:

```python
mean = (x - BETAS[t - 1] / torch.sqrt(1 - ALPHA_BARS[t - 1]) * eps) / torch.sqrt(ALPHAS[t - 1])
x = mean + SIGMAS[t - 1] * noise[k]
```

The exercise's toy has $T = 10$ steps, with $\beta_t$ spaced evenly from
0.02 to 0.6. Its data is two bumps, at $-1.5$ and $+1.5$, with standard
deviation 0.3. For data this simple the best noise prediction has a closed
form, so the "pretrained" model is exact. The trainable part is a correction
$w_t x + b_t$ for each step, which starts at zero.

### Worked example: one denoising step

Take the first sampling step, $t = 10$, from $x_{10} = 0.8$. The schedule
gives $\beta_{10} = 0.6$, $\alpha_{10} = 0.4$, and $\bar\alpha_{10} = 0.01664$. The pretrained model predicts $\epsilon = 0.7751$. Then:

- The noise coefficient is $\beta_t / \sqrt{1 - \bar\alpha_t} = 0.6 / 0.99165 = 0.6051$.
- The mean is $(0.8 - 0.6051 \times 0.7751) / \sqrt{0.4} = 0.3310 \times 1.5811 = 0.5233$.
- The step's standard deviation is $\sigma_{10} = \sqrt{0.6} = 0.7746$.
- With the draw $z = 0.5$, the next state is $x_9 = 0.5233 + 0.7746 \times 0.5 = 0.9106$.

That draw has log-probability $-\tfrac12 (0.5)^2 - \log 0.7746 - 0.9189 = -0.7885$ under the step's Gaussian.

## Denoising as a Markov decision process

Black, Janner, Du, Kostrikov, and Levine (2023) wrote the sampler as a
*Markov decision process* (MDP), a world of states, actions, and rewards
where the next state depends only on the current one. Their DDPO paper
defines it as follows:

- **State** $s = (c, t, x_t)$: the prompt $c$, the step, and the current
  noisy sample.
- **Action** $a = x_{t-1}$: the next, slightly cleaner sample.
- **Policy** $\pi_\theta(a \mid s) = p_\theta(x_{t-1} \mid x_t, c)$: the
  Gaussian denoising step.
- **Transition**: deterministic. The action becomes the next state's sample,
  and the step counter drops by one.
- **Start**: a prompt from the data, $t = T$, and $x_T \sim \mathcal{N}(0, I)$.
- **Reward**: $r(x_0, c)$ after the last step, and 0 after every other step.

Diffusion time runs backward, so the MDP's first action is the step from
$x_T$, and every episode lasts exactly $T$ actions.

> [!KEY]
> Every denoising step is an action drawn from a Gaussian policy. The reward
> scores only the final sample, so every step in a trajectory shares one
> terminal reward.

Why not raise $\log p_\theta(x_0)$ for good samples directly? Because the
final sample's density is intractable: an integral over every path that could
have led to $x_0$. The density of a whole trajectory is easy, the start
density times $T$ Gaussians:

$$
\log p_\theta(x_{0:T}) = \log p(x_T) + \sum_{t=1}^{T} \log \mathcal{N}\big(x_{t-1};\ \mu_\theta(x_t, t),\ \sigma_t^2\big)
$$

Read it as: the path's log-probability is that of the starting noise, which
doesn't depend on $\theta$, plus one Gaussian log-density per step. It is the
same move as scoring a response token by token.

For an image latent, each step's log-density sums over every coordinate.

> [!WARNING]
> The sampler must be stochastic. The deterministic DDIM sampler, with its
> noise parameter $\eta = 0$, maps $x_t$ to $x_{t-1}$ with no randomness, so
> a step has no density and no log-probability. The public DDPO code samples
> with DDIM at $\eta = 1$, which is fully stochastic.

A useful identity follows from the step formula. If you sample $x_{t-1}$ with
noise $z$ and then evaluate its log-probability under the *same* parameters,
$(x_{t-1} - \mu_\theta)/\sigma_t$ is exactly $z$. The log-probability is
$-\tfrac12 z^2 - \log \sigma_t - \tfrac12 \log 2\pi$, whatever the network
predicted. The checks use this identity to test your code.

## DDPO: the policy gradient over denoising steps

With the MDP in place, REINFORCE applies unchanged. The gradient of the
expected reward is the sum of per-step score terms, each weighted by the
trajectory's reward. DDPO's score-function estimator is:

$$
\nabla_\theta \mathbb{E}[r] = \mathbb{E}\Big[\sum_{t=1}^{T} \nabla_\theta \log p_\theta(x_{t-1} \mid x_t, c)\; r(x_0, c)\Big]
$$

Read it as: raise the log-probability of every step in trajectories that
scored well, and lower it in trajectories that scored badly. In practice you
replace $r$ with an advantage. DDPO normalises rewards per prompt with a
running mean and standard deviation, so an easy prompt's high scores don't
drown out a hard prompt's. The toy exercise has one prompt, so it normalises
over the batch:

```python
adv = (r - r.mean()) / (r.std() + 1e-8)          # one number per trajectory
surrogate = (logp.sum(0) * adv).mean()           # logp has shape (T, n)
```

Sampling a batch of images is expensive, so DDPO also reuses each batch for
several gradient steps. That needs importance weights and PPO's clip. For
each step of each trajectory, the ratio is $\rho = p_\theta(x_{t-1} \mid x_t) / p_{\theta_\text{old}}(x_{t-1} \mid x_t)$. The loss to minimise is:

$$
L(\theta) = -\frac{1}{Tn} \sum_{i=1}^{n} \sum_{t=1}^{T} \min\big(\rho_t^i A^i,\ \operatorname{clip}(\rho_t^i, 1 - \varepsilon, 1 + \varepsilon)\, A^i\big)
$$

Read it as PPO's objective, averaged over every denoising step of every
trajectory, with each trajectory's single advantage $A^i$ reused at all its
steps. At the first gradient step $\rho = 1$, and the gradient equals the
score-function estimator averaged over steps.

```python
ratio = torch.exp(logp - logp_old)                      # (T, n)
obj = torch.minimum(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)
loss = -obj.mean()
```

### Worked example: three clipped ratios

Suppose one step had log-probability 0.4093 under the old policy and 0.1905
under the current one. The ratio is $e^{0.1905 - 0.4093} = 0.8035$. With
$\varepsilon = 0.2$ it sits inside $[0.8, 1.2]$, so it counts in full. Two
more cases show the clip:

| Ratio | Advantage | Unclipped | Clipped | Counted |
|---|---|---|---|---|
| 0.8035 | +1 | 0.8035 | 0.8035 | 0.8035 |
| 1.5 | +1 | 1.5 | 1.2 | 1.2 |
| 1.5 | −1 | −1.5 | −1.2 | −1.5 |

The minimum keeps the pessimistic value: no extra credit for moving far, full
blame for moving the wrong way.

The DDPO paper uses a clip range of $10^{-4}$, far smaller than the 0.2
common for language models. A step's ratio is a product over every
coordinate of the latent, so it leaves $[0.8, 1.2]$ after very little
movement. The one-dimensional toy uses 0.2.

## Reward backpropagation

Many image rewards are neural networks that you can differentiate with
respect to the pixels. Then you don't need a policy gradient. Fix the
starting noise $x_T$ and every step's noise $z$. The sample $x_0$ becomes an
ordinary differentiable function of $\theta$, a chain of $T$ network calls,
and you can backpropagate:

$$
\nabla_\theta\, r(x_0) = \frac{\partial r}{\partial x_0}\, \frac{\partial x_0}{\partial \theta}
$$

Read it as the chain rule: how the reward changes with the sample, times how
the sample changes with the parameters through all $T$ steps. Averaged over
random noise, this *pathwise* gradient is an unbiased estimate of the same
$\nabla_\theta \mathbb{E}[r]$ that DDPO estimates. In PyTorch, the whole
method is:

```python
xs = rollout(model, x_T, noise)       # keep the graph
reward(xs[-1]).mean().backward()      # gradients land in model.parameters()
```

DRaFT (Clark, Vicol, Swersky, and Fleet, 2023) does this through a 50-step
DDIM sampler with LoRA and gradient checkpointing to fit in memory. Its
DRaFT-K variant stops the gradient before the last $K$ steps and
differentiates only those. The paper finds $K = 1$ works best, and that
gradient norms explode as $K$ grows. In the toy each step has its own
parameters, so DRaFT-K gives exact gradients for the last $K$ steps'
parameters and zero for the rest. A real network shares its parameters across
steps, so truncation drops part of every parameter's gradient. ReFL, from the
ImageReward paper (Xu et al., 2023), scores a one-step prediction of $x_0$
made at a random late step.

Two gradient estimators for the same quantity invite a comparison. In the
toy, with 2,000 trajectories per estimate, the variance of each estimator for
the per-step biases $b_t$ comes out as follows:

- For the last step, $b_1$, the score-function estimate's variance is about
  90 times the pathwise one. The pathwise gradient knows exactly how $x_0$
  moves. The score function has to infer it from the correlation between
  noise and reward.
- For the high-noise steps, $b_6$ to $b_9$, the pathwise estimate's variance
  is 4 to 6 times *larger*. Its gradient passes through every later step.
  A sample that sits between the two bumps at step 7 can land in either one,
  so tiny changes to it move $x_0$ a lot. Its derivative
  $\partial x_0 / \partial x_7$ averages about three times that of a sample
  already inside a bump, and reaches 245 for the worst one.

> [!KEY]
> DDPO and reward backpropagation estimate the same gradient. DDPO needs only
> reward values. Reward backpropagation needs a differentiable reward and
> memory for the whole chain, and its gradient can blow up through long
> chains.

> [!RECAP]
> - Each step is $x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z$, a Gaussian
>   action with a closed-form log-probability.
> - A trajectory is `(T + 1, n)`: row 0 is $x_T$, row $T$ is $x_0$, and step
>   $k$ is diffusion step $t = T - k$.
> - DDPO's loss is PPO's clipped objective over every step, with one
>   advantage per trajectory.
> - Reward backpropagation differentiates `reward(xs[-1])` through a rollout
>   with fixed noise. DRaFT-K cuts the graph before the last $K$ steps.

## Build it step by step

The checks test these functions in order.

1. **`gaussian_logprob(x, mean, std)`.** Elementwise log-density, any shape.
   Convert `std` with `torch.as_tensor` so a float works. Common mistake:
   writing $-\log \sigma^2$ instead of $-\log \sigma$.
2. **`ddpm_mean(x_t, eps, t)`.** The step mean, with the schedule indexed at
   `t - 1`. The check compares it with the posterior mean of
   $q(x_{t-1} \mid x_t, x_0)$ at every $t$. Common mistake: using
   `ALPHA_BARS` where the formula wants `ALPHAS`.
3. **`rollout(model, x_T, noise, grad_steps=None)`.** Loop `k` from 0 to
   `T - 1` with `t = T - k`, append each state, and return
   `torch.stack(xs)`. For `grad_steps=K`, run the first `T - K` steps inside
   `torch.no_grad()`, then call `.detach()` on the state. Common mistake:
   detaching the state but still building graphs for the early steps, which
   leaves nonzero gradients on their parameters.
4. **`trajectory_logprob(model, xs)`.** Detach `xs` first, then for each
   step recompute the mean from `model.eps` and call your
   `gaussian_logprob`. Return shape `(T, n)`. Common mistake: skipping the
   detach, so a trajectory that still carries its graph leaks pathwise terms
   into the DDPO gradient.
5. **`ddpo_loss(logp, logp_old, advantages, clip_eps)`.** Broadcast
   `advantages[None, :]` across steps, take `torch.minimum` of the unclipped
   and clipped terms, and negate the mean. Common mistake: `torch.max` in
   place of `torch.minimum`.
6. **`ddpo_train(...)`.** Each round, sample under `torch.no_grad()`, record
   `logp_old` and batch-normalised advantages, then take `epochs` Adam steps
   on `ddpo_loss`. Common mistake: recomputing `logp_old` inside the epoch loop,
   which pins the ratio at 1 and turns off the clip.

## Cost and scaling

The two methods spend their budget in different places. For an image model
with $T$ sampling steps, the costs compare as follows:

| | DDPO | Reward backpropagation |
|---|---|---|
| Reward requirement | Any score, even a black box | Differentiable in the sample |
| Forward passes per trajectory | $T$ to sample, $T$ per update epoch | $T$ |
| Memory | One step's activations at a time | All $T$ steps', unless checkpointed or truncated |
| Gradient variance | High, from the score function | Low for late steps, but can explode through long chains |

DDPO's time grows linearly in $T$ and in the number of epochs, and its
memory stays flat. Full reward backpropagation's memory grows linearly in
$T$. Checkpointing trades that memory for a second forward pass, and DRaFT-K
cuts it to $K$ steps.

## Where this breaks

**Reward hacking.** The policy finds whatever the reward rewards. DDPO's
authors report that optimising for incompressibility eventually removes all
recognisable content from the images. Their vision-language reward, which
asks LLaVA to describe the picture, was fooled by images that write the
prompt's words as text. DRaFT reports collapse to nearly identical
high-reward images. People stop early, add a KL penalty to the pretrained
model, and evaluate with a reward that wasn't used in training. The next
lesson measures that KL.

**No density.** A deterministic sampler gives no log-probability. The old
and new log-probabilities must also use the same sampler settings, or the
first ratio isn't 1.

**Loose settings copied from text.** Raw rewards across prompts of different
difficulty tell the model which prompt it got, not how well it did. A clip
range copied from language-model PPO lets an image policy move far on one
batch.

**Exploding chains.** Reward backpropagation multiplies $T$ Jacobians, and
the gradient can explode, as DRaFT measured. Truncating to the last few steps
keeps it in check.

## Further reading

The following papers set up the denoising MDP and the two families of
fine-tuning methods:

- [Black et al., *Training Diffusion Models with Reinforcement Learning* (2023)](https://arxiv.org/abs/2305.13301)
  defines the denoising MDP and DDPO, and documents its reward hacks.
- [Fan et al., *DPOK* (2023)](https://arxiv.org/abs/2305.16381) adds a KL
  penalty to the pretrained model and a learned value baseline.
- [Clark et al., *Directly Fine-Tuning Diffusion Models on Differentiable Rewards* (2023)](https://arxiv.org/abs/2309.17400)
  introduces DRaFT, DRaFT-K, and DRaFT-LV.
- [Xu et al., *ImageReward* (2023)](https://arxiv.org/abs/2304.05977) trains a
  human-preference reward for images and fine-tunes with ReFL.
- [Ho, Jain, and Abbeel, *Denoising Diffusion Probabilistic Models* (2020)](https://arxiv.org/abs/2006.11239)
  gives the sampler and the step variances used here.
- [Song, Meng, and Ermon, *Denoising Diffusion Implicit Models* (2020)](https://arxiv.org/abs/2010.02502)
  gives DDIM and its noise parameter $\eta$.

Open questions that researchers are working on:

- How should credit be split across denoising steps, when early steps fix
  the layout and late steps fix the texture?
- Can a reward model be made robust enough that fine-tuning against it
  doesn't find its blind spots?
- When is a pathwise gradient worth its memory, and when does a
  score-function method with more samples win?

## The exercise

You build the denoising MDP for a 10-step, one-dimensional diffusion model
whose pretrained noise prediction is exact for two-bump data. Write
`gaussian_logprob` and `ddpm_mean` for one step, `rollout` to run the sampler
with given noise and optional DRaFT-K truncation, and `trajectory_logprob` to
score a stored trajectory. Then write `ddpo_loss` for the clipped per-step
surrogate and `ddpo_train` to run DDPO against the reward
$\sigma(4(x_0 - 1))$, which favours samples to the right of 1.

The checks compare each piece with an independent computation: the worked
values, the two-coefficient posterior mean, and the hand-clipped table. They
test that DDPO's score-function gradient agrees with the pathwise gradient
within four standard errors, that the pathwise gradient matches finite
differences, and that DRaFT-K zeroes the early steps. The last check expects
DDPO to lift $P(x_0 > 1)$ from about 0.44 to above 0.9.
