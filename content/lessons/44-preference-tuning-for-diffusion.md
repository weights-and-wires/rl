---
title: Preference tuning for diffusion
slug: 44-preference-tuning-for-diffusion
part: "Part 9 — Beyond text"
summary: Measure how far a fine-tuned sampler has drifted with the path KL, train from preference pairs with Diffusion-DPO, and watch a reward get hacked.
minutes: 60
gpu: false
objectives:
  - Show that the KL between two samplers' paths is a sum of per-step Gaussian KLs, and compute it exactly on a linear chain.
  - Explain why the path KL can be far larger than the KL between final samples, with a worked example where it is 21 times larger.
  - Derive the Diffusion-DPO loss from DPO in four steps and implement it from denoising errors.
  - Reproduce reward hacking on a toy model and show that a path-KL penalty holds it back.
prereqs:
  - 36-direct-preference-optimization
  - 43-diffusion-models-as-policies
exercise: 44-preference-tuning-for-diffusion
---

# Preference tuning for diffusion

The previous lesson fine-tuned a diffusion model toward a reward. It worked,
and it also showed the usual danger: a policy pushed hard toward a score
finds whatever the score gets wrong. In language models the standard guard is
a KL penalty that keeps the policy near a reference model. For a diffusion
model that raises a question. The KL between two models' *images* is
intractable, so what exactly do you penalise?

The answer is the KL between whole denoising paths, which splits into one
closed-form Gaussian KL per step. It is cheap and exact, but it measures
something different from the KL between final images, and the difference can
be large.

The same path view makes preference learning possible. DPO needs a
response's log-probability, which an image doesn't have. Diffusion-DPO works
around that with the path, and ends with a loss built from denoising errors.
Then you watch a proxy reward get hacked and a path-KL penalty stop it.

## What you already know

- A diffusion sampler is a policy whose actions are Gaussian denoising steps,
  $x_{t-1} \sim \mathcal{N}(\mu_\theta(x_t, t), \sigma_t^2)$.
- The *KL divergence* $\mathrm{KL}(p \,\|\, q) = \mathbb{E}_p[\log p - \log q]$ is zero when two distributions match and positive otherwise.
- DPO trains on preference pairs with the loss $-\log \sigma(\beta \log \tfrac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \log \tfrac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)})$, where $\sigma$ is the
  sigmoid, $y_w$ won, and $y_l$ lost.
- A diffusion model is trained to predict the noise $\epsilon$ in $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$.
- Reward backpropagation differentiates a reward through the sampler with
  the noise held fixed.

## The KL between two sampling paths

Two samplers, the fine-tuned $p_\theta$ and the reference $p_{\text{ref}}$,
start from the same noise distribution and differ only in their steps. The
probability of a whole path is the start density times one Gaussian per
step. Divide the two path probabilities and the start densities cancel:

$$
\log \frac{p_\theta(x_{0:T})}{p_{\text{ref}}(x_{0:T})} = \sum_{t=1}^{T} \log \frac{p_\theta(x_{t-1} \mid x_t)}{p_{\text{ref}}(x_{t-1} \mid x_t)}
$$

Read it as: the path log-ratio is the sum of the per-step log-ratios. Take the
expectation under $p_\theta$. For each step, first average over the next
state given the current one. That inner average is the KL between the two
steps, and what's left is an average over where $p_\theta$ puts $x_t$:

$$
\mathrm{KL}\big(p_\theta(x_{0:T}) \,\|\, p_{\text{ref}}(x_{0:T})\big) = \sum_{t=1}^{T} \mathbb{E}_{x_t \sim p_\theta}\Big[\mathrm{KL}\big(p_\theta(\cdot \mid x_t) \,\|\, p_{\text{ref}}(\cdot \mid x_t)\big)\Big]
$$

This is the chain rule for KL. Each inner KL is between two Gaussians, which
has a closed form:

$$
\mathrm{KL}\big(\mathcal{N}(\mu_1, \sigma_1^2) \,\|\, \mathcal{N}(\mu_2, \sigma_2^2)\big) = \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac12
$$

Read it as: a penalty for the variance ratio plus a penalty for the squared
gap between means, measured in units of the second variance. For
$\mathcal{N}(0.5, 0.2^2)$ against $\mathcal{N}(0.4, 0.25^2)$ it gives
$\log 1.25 + 0.05 / 0.125 - 0.5 = 0.2231 + 0.4 - 0.5 = 0.1231$. When both
steps share $\sigma_t$, only the mean gap survives:
$(\mu_\theta - \mu_{\text{ref}})^2 / (2\sigma_t^2)$.

```python
kl = torch.log(s2 / s1) + (s1**2 + (m1 - m2)**2) / (2 * s2**2) - 0.5
```

DPOK (Fan et al., 2023) penalises exactly this sum. Its Lemma 4.2 bounds the
final-sample KL by the path KL, so penalising the path also limits how far the
images drift.

> [!KEY]
> The path KL is a sum of per-step Gaussian KLs, each evaluated at states the
> fine-tuned sampler visits. It is exact and cheap. The KL between final
> samples is intractable.

For the toy diffusion model, both steps share $\sigma_t$ and differ only in
their noise prediction. Put the step mean from the previous lesson into the
shared-variance KL:

$$
\mathrm{KL}_t = \frac{(\mu_\theta - \mu_{\text{ref}})^2}{2\sigma_t^2} = \frac{\beta_t^2}{2 \sigma_t^2\, \alpha_t (1 - \bar\alpha_t)}\, (\epsilon_\theta - \epsilon_{\text{ref}})^2
$$

Read it as: a step's KL is its squared noise-prediction gap times a
step-dependent weight. With $\sigma_t^2 = \beta_t$ the weight is $\beta_t / (2\alpha_t(1 - \bar\alpha_t))$, which runs from 0.34 to 0.76 across the
toy's ten steps.

### Worked example: a four-step chain

A linear-Gaussian chain has an exact answer for everything, so it's the
place to see the path KL and the final-sample KL side by side. Start from $x \sim \mathcal{N}(0, 1)$ and take four steps, $x \leftarrow a x + b + s z$,
with these coefficients in sampling order:

| | Step 1 | Step 2 | Step 3 | Step 4 |
|---|---|---|---|---|
| $a_\theta$ | 0.90 | 0.80 | 0.85 | 0.95 |
| $b_\theta$ | 0.10 | 0.20 | 0.00 | 0.30 |
| $a_{\text{ref}}$ | 0.90 | 0.70 | 0.85 | 1.00 |
| $b_{\text{ref}}$ | 0.00 | 0.20 | 0.10 | 0.00 |
| $s$ | 0.50 | 0.40 | 0.30 | 0.20 |

The two steps' means differ by $\Delta a\, x + \Delta b$, so each step's
expected KL is $\mathbb{E}[(\Delta a\, x + \Delta b)^2] / (2s^2)$, with $x$
drawn from the fine-tuned chain. You need the mean $m$ and variance $v$ of
$x$ before each step. They update as $m \leftarrow a m + b$ and $v \leftarrow a^2 v + s^2$, and $\mathbb{E}[(\Delta a\, x + \Delta b)^2] = \Delta a^2 (v + m^2) + 2 \Delta a\, \Delta b\, m + \Delta b^2$.

```python
m, v, kl = 0.0, 1.0, 0.0
for k in range(4):
    da, db = a_th[k] - a_ref[k], b_th[k] - b_ref[k]
    kl += (da**2 * (v + m**2) + 2 * da * db * m + db**2) / (2 * s[k]**2)
    m, v = a_th[k] * m + b_th[k], a_th[k]**2 * v + s[k]**2
```

The four terms are 0.02, 0.03344, 0.05556, and 1.05926, which add to a path KL
of 1.16826. The last step contributes most, because its offset gap of 0.3 is
measured against the smallest noise, $s = 0.2$.

Now compare the final samples. The fine-tuned chain ends at
$\mathcal{N}(0.5261, 0.6679)$ and the reference at $\mathcal{N}(0.27, 0.6209)$, written as mean and variance. The Gaussian KL between them is
0.05419, about $1/21$ of the path KL. The two chains take different routes to
nearly the same place, and the path KL charges for the route.

The gap always points one way. Throwing away the intermediate states can
only lose information, so the final-sample KL never exceeds the path KL. The
path penalty is an upper bound, and it can be loose: it charges for changes
that don't reach the output.

> [!WARNING]
> Evaluate the per-step KL at states from the fine-tuned model's own
> rollouts. The expectation is over $x_t \sim p_\theta$. Averaging over
> reference rollouts computes a different number.

## Diffusion-DPO

DPO needs $\log \pi_\theta(y)$ for the winning and losing responses. For an
image, $\log p_\theta(x_0)$ is intractable. Wallace et al. (2023) get around
this in four steps:

1. **Apply DPO to paths.** Put a reward on whole denoising paths and write
   DPO's objective with path log-ratios in place of response log-ratios. By
   the chain rule of the previous section, each path log-ratio is a sum of
   per-step log-ratios.
2. **Swap the unknown path for the noising path.** The loss needs paths that
   end at the given winner $x_0^w$, but sampling $x_{1:T}$ from $p_\theta(x_{1:T} \mid x_0)$ is intractable. The paper replaces it with the forward noising
   process $q(x_{1:T} \mid x_0)$, which you can sample in one line.
3. **Move the average outside.** $-\log \sigma$ is convex, so by Jensen's
   inequality the loss with the average over steps outside the sigmoid is an
   upper bound on the loss with it inside. The bound is what gets trained.
4. **Turn Gaussian KLs into noise errors.** Each remaining term compares a
   forward-process step with a model step. Both are Gaussians with the same
   variance, so their KL is a weighted squared error between the true noise
   and the predicted noise.

The result, the paper's Equation 14, is:

$$
L(\theta) = -\mathbb{E}\, \log \sigma\Big(-\beta T \omega(\lambda_t) \big[(e_\theta^w - e_{\text{ref}}^w) - (e_\theta^l - e_{\text{ref}}^l)\big]\Big), \qquad e_\theta^w = \lVert \epsilon^w - \epsilon_\theta(x_t^w, t)\rVert^2
$$

Read it as: noise the winner and the loser to the same step $t$. For each,
ask how much better the model denoises it than the reference does, as a
change in squared error. The loss falls when the model improves on the winner
by more than on the loser. Here $x_t^w = \sqrt{\bar\alpha_t}\, x_0^w + \sqrt{1 - \bar\alpha_t}\, \epsilon^w$, $t$ is drawn uniformly, and $\omega(\lambda_t)$
is a weight that the paper keeps constant. The exercise folds $\beta T \omega$
into one number, `beta`. In PyTorch:

```python
inside = -beta * ((err_w - err_w_ref) - (err_l - err_l_ref))
loss = -F.logsigmoid(inside).mean()
```

The model and the reference must see the same $x_t$ for each image, with the
same $t$ and the same $\epsilon$. Otherwise the error differences are mostly
noise.

### Worked example: one pair

Take $\beta T \omega = 1$ and these squared errors:

| | Model | Reference |
|---|---|---|
| Winner | 0.30 | 0.50 |
| Loser | 0.60 | 0.40 |

The model improved on the winner by $-0.20$ and got worse on the loser by
$+0.20$, so the bracket is $-0.20 - 0.20 = -0.40$. The loss is $-\log \sigma(0.40) = 0.5130$. At the start, when the model equals the reference,
the bracket is 0 and the loss is $\log 2 = 0.6931$.

The paper trains on Pick-a-Pic v2, 851,293 pairs after removing ties, with
$\beta = 2000$ for Stable Diffusion 1.5 and $\beta = 5000$ for SDXL, far from
the $\beta \approx 0.1$ common in text DPO. The right scale depends on how an
implementation normalises the squared error and folds in $T$, so copy a
$\beta$ only together with the code it came from.

> [!KEY]
> Diffusion-DPO is DPO on denoising paths, made tractable by noising the
> preferred and rejected images and comparing the model's denoising errors
> with the reference's on the same noisy inputs.

The loss sees only *relative* improvement, so both errors can rise while the
loss falls. The toy shows it clearly. After 150 steps of Diffusion-DPO its
samples are much better, with mean true reward rising from $-6.1$ to $-0.3$.
Yet its squared error on the winning images rose from 0.43 to 1.71, four
times the reference's. The error on the losers rose further, to 2.45. Text
DPO has the same failure, where the chosen response's likelihood falls.

Two relatives take other routes. D3PO (Yang et al., 2023) applies the DPO
loss at every step of stored sampling trajectories, so it needs rollouts
rather than a fixed image dataset. Diffusion-KTO (Li et al., 2024) learns from
single images labelled good or bad, rather than from pairs.

## Reward hacking

A learned reward is right where it was trained and can be wrong everywhere
else. Fine-tuning goes looking for "everywhere else." Documented cases
include DDPO's incompressibility runs, which lost all recognisable content,
and its LLaVA-based alignment reward, fooled by images that write the prompt
as text. DRaFT reports collapse to nearly identical images. Flow-GRPO reports
that its KL penalty prevents this kind of hacking while keeping the task
gains.

The toy reproduces the pattern with nothing hidden. The true reward is
$r^\star(x_0) = -(x_0 - 2)^2$: land near 2. The proxy reward is $\hat r(x_0) = x_0$, a reward model that learned "further right is better." On pairs of
reference samples the proxy ranks 99.4% of pairs the same way as the true
reward, because only 4% of reference samples lie beyond 2. It looks like an
excellent reward model.

Now maximise the proxy by reward backpropagation, with and without a
path-KL penalty, $\max_\theta \mathbb{E}[\hat r(x_0)] - \lambda\, \mathrm{KL}_{\text{path}}$. Averaged over each batch of 256 samples, a run
looks like this:

| Step | Proxy, $\lambda = 0$ | True, $\lambda = 0$ | Path KL, $\lambda = 0$ | True, $\lambda = 0.3$ | Path KL, $\lambda = 0.3$ |
|---|---|---|---|---|---|
| 0 | 0.00 | −6.30 | 0.00 | −6.58 | 0.00 |
| 20 | 1.51 | −2.14 | 1.63 | −2.60 | 0.91 |
| 30 | 2.53 | −1.56 | 6.67 | −1.13 | 1.18 |
| 40 | 3.65 | −4.70 | 23.5 | −1.22 | 1.34 |
| 50 | 6.13 | −20.3 | 95.4 | −0.65 | 1.51 |

Without the penalty the proxy climbs without limit. The true reward improves
until the samples reach 2, around step 30, and then collapses as they keep
going. The path KL explodes along with it. With $\lambda = 0.3$ the path KL
levels off near 1.5, the samples settle near 1.8, and the true reward keeps
improving.

> [!RECAP]
> - The path KL is the sum over steps of $\mathbb{E}_{x_t \sim p_\theta}[\mathrm{KL}_t]$, and for shared variance $\mathrm{KL}_t = (\mu_\theta - \mu_{\text{ref}})^2 / (2\sigma_t^2)$.
> - On the lesson's chain the path KL is 1.16826 and the final-sample KL is
>   0.05419.
> - Diffusion-DPO is $-\log \sigma(-\beta[(e^w_\theta - e^w_{\text{ref}}) - (e^l_\theta - e^l_{\text{ref}})])$, with one $t$ per pair and shared
>   noise for model and reference.
> - A proxy that ranks reference samples well can still be hacked. A
>   path-KL penalty limits how far fine-tuning can chase it.

## Build it step by step

The checks test these functions in order.

1. **`gaussian_kl(mu1, std1, mu2, std2)`.** One line of the closed form,
   elementwise. Convert inputs with `torch.as_tensor`. Common mistake: using
   variances where the formula has standard deviations in the log.
2. **`chain_path_kl(...)`.** Loop over the steps, carrying the mean and
   variance of $x$ under the fine-tuned chain. Add each step's term *before*
   updating the moments. Common mistake: updating first, which scores each
   step at the next step's state.
3. **`chain_marginal_kl(...)`.** Propagate both chains' moments to the end
   and call `gaussian_kl` with standard deviations `torch.sqrt(v)`.
4. **`chain_path_kl_mc(...)`.** Draw the start, then for each step draw the
   next state from the fine-tuned step and add $[(x' - \mu_{\text{ref}})^2 - (x' - \mu_\theta)^2] / (2s^2)$. Return the mean and `std / sqrt(n)`.
   Common mistake: drawing a fresh $x'$ for the reference term.
5. **`ddpm_step_kl(model, xs)`.** For each row, compute both step means with
   `ddpm_mean` and call `gaussian_kl` with the shared `SIGMAS[t - 1]`. Keep
   the graph, because training differentiates it.
6. **`denoising_errors(model, x0, t, eps)`.** Noise once, then score
   `model.eps` and `eps_pretrained` on the same `x_t`. Common mistake:
   noising twice with different noise.
7. **`diffusion_dpo_loss(...)`.** Use `F.logsigmoid` for stability. The
   checks expect $\log 2$ at the reference and 0.5130 on the worked example.
8. **`diffusion_dpo_train(...)`.** Each step, draw pair indices, one `t` per
   pair with `torch.randint(1, T + 1, ...)`, and separate noise for winner and
   loser. Record the loss before stepping.
9. **`reward_backprop_train(...)`.** Roll out with `rollout`, compute
   `-reward_fn(xs[-1]).mean() + kl_coef * ddpm_step_kl(model, xs).sum(0).mean()`,
   and step. Common mistake: detaching `xs` before the KL, which removes the
   KL's gradient through the visited states.

## Cost and scaling

Each quantity costs a different amount as the model grows. For a model with
$T$ steps and $d$ latent coordinates:

| Quantity | Extra cost | Notes |
|---|---|---|
| Path KL on rollouts | One reference forward pass per step | Exact given the states. Memory for both networks. |
| Final-sample KL | Intractable | Only bounds or rough density estimates. |
| Diffusion-DPO step | Two forward passes of model and reference per pair | One random $t$, no sampling chain. |
| Reward backprop with KL | $T$ steps of both networks, kept for backward | Memory grows linearly in $T$. |

Diffusion-DPO never runs the sampler, which is why it scales to hundreds of
thousands of pairs.

## Where this breaks

**A loose KL bound.** The path KL can be many times the final-sample KL, as
the chain shows. A penalty tuned on the path can hold the model back more
than intended, or less, depending on the route. Its weight has to be tuned on
held-out image quality.

**Displaced likelihoods.** Diffusion-DPO can raise the model's denoising
error on the very images it prefers. A supervised denoising term on the
winners, a remedy borrowed from text DPO, is one fix. Stopping early is
another.

**Reward hacking.** A reward model is trustworthy only near its training
data. People mix several rewards, use pairwise rewards, keep a KL penalty, and
judge progress with a reward model or human raters not used in training.

## Further reading

The following papers cover the KL view and preference learning for
diffusion models:

- [Fan et al., *DPOK* (2023)](https://arxiv.org/abs/2305.16381) penalises the
  path KL and proves it bounds the final-sample KL.
- [Wallace et al., *Diffusion Model Alignment Using Direct Preference Optimization* (2023)](https://arxiv.org/abs/2311.12908)
  derives Diffusion-DPO and trains SDXL on Pick-a-Pic v2.
- [Yang et al., *Using Human Feedback to Fine-tune Diffusion Models without Any Reward Model* (2023)](https://arxiv.org/abs/2311.13231)
  gives D3PO, per-step DPO on sampled trajectories.
- [Li et al., *Aligning Diffusion Models by Optimizing Human Utility* (2024)](https://arxiv.org/abs/2404.04465)
  gives Diffusion-KTO for single-image labels.
- [Kirstain et al., *Pick-a-Pic* (2023)](https://arxiv.org/abs/2305.01569)
  collects the preference dataset and trains PickScore.
- [Rafailov et al., *Direct Preference Optimization* (2023)](https://arxiv.org/abs/2305.18290)
  derives the text version that Diffusion-DPO extends.

Open questions that researchers are working on:

- Can a regulariser penalise drift in the images rather than drift in the
  route, without an intractable density?
- Should a single image-level preference apply equally to every denoising
  step, or should early and late steps get different credit?
- How can you detect reward hacking automatically, before a person looks at
  the samples?

## The exercise

You measure KL exactly on the four-step chain with `gaussian_kl`,
`chain_path_kl`, `chain_marginal_kl`, and the Monte Carlo
`chain_path_kl_mc`. On the toy diffusion model you write `ddpm_step_kl`, then
`denoising_errors`, `diffusion_dpo_loss`, and `diffusion_dpo_train` for
Diffusion-DPO, and `reward_backprop_train` for reward backpropagation with a
path-KL penalty.

The checks confirm the worked values 0.1231, 1.16826, and 0.05419, and that
the Monte Carlo estimate lands within three standard errors. They test that
summed step KLs and summed log-ratios agree on the toy, that the Diffusion-DPO
loss is $\log 2$ at the reference and 0.5130 on the worked example, and that
its gradient matches finite differences. Training on reference pairs must
lift the true reward above $-1$. Finally, reward backpropagation on the
proxy must hack it without a penalty and stay safe with $\lambda = 0.3$.
