---
title: Diffusion and flow models as policies
slug: 068-diffusion-and-flow-policies
part: "Part 10 — Beyond text, and what can go wrong"
summary: See a denoising chain as an MDP whose steps are Gaussian actions, then fine-tune a diffusion model with a policy gradient (DDPO) or with preference pairs.
minutes: 45
gpu: false
objectives:
  - Write the denoising MDP, with its states, actions, transitions, and a reward that arrives only at the last step.
  - Compute the Gaussian log-probability of each denoising step, and say why the whole path has a density when the final image does not.
  - Implement DDPO's clipped per-step surrogate and train a toy diffusion model toward a reward.
  - Compute the Diffusion-DPO loss from denoising errors, and say how Flow-GRPO makes a flow model stochastic.
prereqs:
  - 031-the-policy-gradient
  - 036-proximal-policy-optimization
  - 056-language-models-as-policies
  - 060-direct-preference-optimization
exercise: 068-diffusion-and-flow-policies
---

# Diffusion and flow models as policies

A text-to-image model is trained to copy the pictures in its training set. Nothing in that training says which pictures people like or which ones match the prompt. When you care about a score like that, you fine-tune toward it, the way you fine-tuned a language model toward a reward.

But a language model gives you something a diffusion model does not: the probability of what it wrote. A policy gradient needs that probability, and the probability of a final image is an integral over every path that could have led to it. Nobody can compute it. So how do you apply a policy gradient?

You change what counts as the action. The chain of cleaning steps is a sequence of small random choices, each with a simple probability. This chapter shows that the chain is an MDP, so REINFORCE and PPO's clip apply to it unchanged.

## What you already know

- The policy gradient raises the log-probability of actions in trajectories that scored well (see [The policy gradient and REINFORCE](/lessons/031-the-policy-gradient)).
- PPO clips the ratio $\pi_\theta/\pi_{\text{old}}$ so a batch can be reused for a few updates (see [Proximal policy optimization](/lessons/036-proximal-policy-optimization)).
- A response's probability is the product of its token probabilities (see [Language models as policies](/lessons/056-language-models-as-policies)).
- DPO trains on a winner and a loser with a loss in log-probability ratios (see [Direct preference optimization](/lessons/060-direct-preference-optimization)).
- The log-density of a Gaussian is $-\tfrac12((x-\mu)/\sigma)^2 - \log\sigma - \tfrac12\log 2\pi$.

## A sample is a chain of small choices

A diffusion model learns to undo noise. In training you blend a clean example $x_0$ with Gaussian noise $\epsilon$, keeping a fraction $\sqrt{\bar\alpha_t}$ of the signal at noise level $t$, and a network $\epsilon_\theta(x_t, t)$ learns to predict $\epsilon$. The numbers $\bar\alpha_t$ fall from near 1 at $t = 1$ to near 0 at $t = T$, so $x_T$ is almost pure noise.

To sample, you run the process backward. Start from $x_T \sim \mathcal{N}(0, 1)$ and take $T$ steps, each from $x_t$ to $x_{t-1}$. The DDPM sampler (Ho, Jain, and Abbeel, 2020) makes each step a Gaussian draw:

$$
x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z, \qquad
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\, \epsilon_\theta(x_t, t)\Big)
$$

Read it as: subtract a scaled copy of the predicted noise, rescale, then add fresh noise $z \sim \mathcal{N}(0, 1)$ of size $\sigma_t$. Here $\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \alpha_1 \cdots \alpha_t$ come from a noise schedule $\beta_t$, and the exercise uses $\sigma_t^2 = \beta_t$. In code, one step is two lines:

```python
mean = (x - BETAS[t - 1] / torch.sqrt(1 - ALPHA_BARS[t - 1]) * eps) / torch.sqrt(ALPHAS[t - 1])
x = mean + SIGMAS[t - 1] * noise[k]
```

The exercise's toy has $T = 10$ steps, with $\beta_t$ spaced evenly from 0.02 to 0.6, and data that is two bumps at $-1.5$ and $+1.5$. Take the first sampling step, $t = 10$, from $x_{10} = 0.8$. The schedule gives $\beta_{10} = 0.6$, $\alpha_{10} = 0.4$, and $\bar\alpha_{10} = 0.01664$, and the pretrained network predicts $\epsilon = 0.7751$.

- The mean is $(0.8 - 0.6051 \times 0.7751)/\sqrt{0.4} = 0.5233$, where $0.6051 = 0.6/\sqrt{1 - 0.01664}$.
- The standard deviation is $\sigma_{10} = \sqrt{0.6} = 0.7746$.
- With the draw $z = 0.5$, the next state is $x_9 = 0.5233 + 0.7746 \times 0.5 = 0.9106$.

That draw has log-probability $-\tfrac12(0.5)^2 - \log 0.7746 - 0.9189 = -0.7885$ under the step's Gaussian. The network chose the mean, and the sampler rolled the dice for the rest. That is exactly what a policy does.

## The denoising chain is an MDP

Black, Janner, Du, Kostrikov, and Levine (2023) wrote the sampler as a Markov decision process, which DDPO then trains with a policy gradient. Figure 1 shows it with the numbers above.

```figure
<svg viewBox="0 0 600 230" role="img" aria-label="A denoising chain from x10 equal 0.80 to x9 equal 0.91, with each arrow a Gaussian action and the reward only after the last step">
  <defs>
    <marker id="c058-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" class="fill-b"/>
    </marker>
  </defs>
  <circle cx="70" cy="90" r="34" class="fill-soft-a stroke-a"/>
  <circle cx="230" cy="90" r="34" class="fill-soft-a stroke-a"/>
  <circle cx="380" cy="90" r="34" class="fill-soft-a stroke-a dashed"/>
  <circle cx="530" cy="90" r="34" class="fill-soft-c stroke-c"/>
  <text x="70" y="86" text-anchor="middle" class="fig-bold">x10</text>
  <text x="70" y="104" text-anchor="middle" class="fig-small">0.80</text>
  <text x="230" y="86" text-anchor="middle" class="fig-bold">x9</text>
  <text x="230" y="104" text-anchor="middle" class="fig-small">0.91</text>
  <text x="380" y="94" text-anchor="middle" class="fig-bold">...</text>
  <text x="530" y="86" text-anchor="middle" class="fig-bold">x0</text>
  <text x="530" y="104" text-anchor="middle" class="fig-small">sample</text>
  <line x1="106" y1="90" x2="192" y2="90" class="stroke-b" marker-end="url(#c058-arrow)"/>
  <line x1="268" y1="90" x2="342" y2="90" class="stroke-b" marker-end="url(#c058-arrow)"/>
  <line x1="418" y1="90" x2="492" y2="90" class="stroke-b" marker-end="url(#c058-arrow)"/>
  <text x="150" y="60" text-anchor="middle" class="fig-small">action, z = 0.5</text>
  <text x="150" y="136" text-anchor="middle" class="fig-small">N(0.523, 0.775²)</text>
  <text x="150" y="154" text-anchor="middle" class="fig-small">log prob -0.789</text>
  <text x="305" y="60" text-anchor="middle" class="fig-small">reward 0</text>
  <text x="455" y="60" text-anchor="middle" class="fig-small">reward 0</text>
  <rect x="470" y="150" width="120" height="44" class="fill-sunk stroke-muted"/>
  <text x="530" y="170" text-anchor="middle" class="fig-small">r(x0) arrives</text>
  <text x="530" y="186" text-anchor="middle" class="fig-small">only here</text>
  <line x1="530" y1="124" x2="530" y2="150" class="stroke-muted"/>
</svg>
Figure 1. Each arrow is a Gaussian action chosen by the network. The first step uses the worked example's numbers. The reward scores only the final sample, so every step in a trajectory shares one terminal reward.
```

- **State** $s = (c, t, x_t)$: the prompt $c$, the step, and the current noisy sample.
- **Action** $a = x_{t-1}$: the next, slightly cleaner sample.
- **Policy** $\pi_\theta(a \mid s) = p_\theta(x_{t-1} \mid x_t, c)$: the Gaussian denoising step.
- **Transition**: deterministic. The action becomes the next state's sample, and the step counter drops by one.
- **Reward**: $r(x_0, c)$ after the last step, and 0 after every other step.

Why not raise $\log p_\theta(x_0)$ for good samples directly? Because that density is intractable. But the density of the whole *path* is easy: the start density times $T$ Gaussians,

$$
\log p_\theta(x_{0:T}) = \log p(x_T) + \sum_{t=1}^{T} \log \mathcal{N}\big(x_{t-1};\ \mu_\theta(x_t, t),\ \sigma_t^2\big)
$$

Read it as: the path's log-probability is the starting noise's, which does not depend on $\theta$, plus one Gaussian log-density per step. It is the same move as scoring a response token by token.

> [!KEY]
> Every denoising step is an action drawn from a Gaussian policy with a closed-form log-probability. The reward scores only the final sample, so every step shares one terminal reward.

> [!WARNING]
> The sampler must be stochastic. The deterministic DDIM sampler with noise parameter $\eta = 0$ maps $x_t$ to $x_{t-1}$ with no randomness, so a step has no density and no log-probability. The public DDPO code samples with $\eta = 1$.

## DDPO: the policy gradient over denoising steps

With the MDP in place, REINFORCE applies unchanged. Raise the log-probability of every step in trajectories that scored well, and lower it in the others. In practice you replace the reward with an advantage, $A = (r - \text{mean})/\text{std}$ over the batch. Sampling images is expensive, so you reuse each batch for several gradient steps, which needs PPO's clip on the per-step ratio $\rho = p_\theta(x_{t-1} \mid x_t)/p_{\theta_{\text{old}}}(x_{t-1} \mid x_t)$. DDPO's loss is

$$
L(\theta) = -\frac{1}{Tn} \sum_{i=1}^{n} \sum_{t=1}^{T} \min\big(\rho_t^i A^i,\ \operatorname{clip}(\rho_t^i, 1 - \varepsilon, 1 + \varepsilon)\, A^i\big)
$$

Read it as: PPO's objective averaged over every denoising step of every trajectory, with each trajectory's single advantage $A^i$ reused at all its steps. In code:

```python
ratio = torch.exp(logp - logp_old)                      # (T, n)
obj = torch.minimum(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)
loss = -obj.mean()
```

Work three cases with $\varepsilon = 0.2$. A step with log-probability 0.4093 under the old policy and 0.1905 under the current one has ratio $e^{-0.2188} = 0.8035$, inside $[0.8, 1.2]$, so it counts in full. Two more cases show the clip:

| Ratio | Advantage | Unclipped | Clipped | Counted |
|---|---|---|---|---|
| 0.8035 | +1 | 0.8035 | 0.8035 | 0.8035 |
| 1.5 | +1 | 1.5 | 1.2 | 1.2 |
| 1.5 | -1 | -1.5 | -1.2 | -1.5 |

The minimum keeps the pessimistic value: no extra credit for moving far, full blame for moving the wrong way. The DDPO paper uses a clip range of $10^{-4}$, because an image step's ratio is a product over every coordinate and leaves $[0.8, 1.2]$ after very little movement. The one-dimensional toy uses 0.2.

The exercise trains the toy against the reward $\sigma(4(x_0 - 1))$, which favors samples to the right of 1. The pretrained model puts 43% of its samples there. Figure 2 shows 20 DDPO iterations lifting the batch's mean reward from 0.38 to 1.00, and afterward 99.9% of samples land right of 1.

```figure
<svg viewBox="0 0 600 260" role="img" aria-label="Mean reward of each DDPO batch rising from 0.38 at iteration 0 to 1.00 at iteration 19">
  <line x1="60" y1="190" x2="560" y2="190" class="stroke-muted"/>
  <line x1="60" y1="40" x2="560" y2="40" class="stroke-grid"/>
  <line x1="60" y1="115" x2="560" y2="115" class="stroke-grid"/>
  <text x="52" y="194" text-anchor="end" class="fig-small">0</text>
  <text x="52" y="119" text-anchor="end" class="fig-small">0.5</text>
  <text x="52" y="44" text-anchor="end" class="fig-small">1</text>
  <polyline points="60.0,132.7 86.3,115.0 112.6,103.2 138.9,92.8 165.3,79.8 191.6,68.3 217.9,63.6 244.2,53.0 270.5,49.9 296.8,48.6 323.2,46.2 349.5,44.7 375.8,43.6 402.1,43.6 428.4,43.8 454.7,40.6 481.1,41.2 507.4,40.0 533.7,41.8 560.0,40.0" fill="none" class="stroke-b"/>
  <circle cx="60" cy="132.7" r="4" class="fill-b"/>
  <circle cx="560" cy="40" r="4" class="fill-b"/>
  <text x="70" y="154" class="fig-small">0.38</text>
  <text x="500" y="30" class="fig-small">1.00</text>
  <text x="60" y="212" class="fig-small">0</text>
  <text x="310" y="212" text-anchor="middle" class="fig-small">iteration</text>
  <text x="560" y="212" text-anchor="end" class="fig-small">19</text>
  <text x="20" y="20" class="fig-bold">Mean reward of the batch</text>
</svg>
Figure 2. DDPO needs only reward values. The curve rises quickly and flattens near 1, because once nearly every trajectory scores high, the normalized advantages carry little signal.
```

## Learn from preferences instead of a reward

Often you have pairs of images, a winner $x_0^w$ and a loser $x_0^l$, and no reward. DPO needs the probability of each response, which an image does not have. Wallace and colleagues (2023) get around this by working with paths, replacing the intractable path ending at $x_0$ with the forward noising process, which you can sample in one line. The result is a loss built from denoising errors:

$$
L(\theta) = -\mathbb{E}\, \log \sigma\Big(-\beta \big[(e_\theta^w - e_{\text{ref}}^w) - (e_\theta^l - e_{\text{ref}}^l)\big]\Big), \qquad e_\theta^w = \lVert \epsilon^w - \epsilon_\theta(x_t^w, t)\rVert^2
$$

Read it as: noise the winner and the loser to the same step $t$. For each, ask how much better the model denoises it than the reference does, as a change in squared error. The loss falls when the model improves on the winner by more than on the loser. The exercise folds the paper's $\beta T \omega$ into one number, `beta`.

Take $\beta = 1$ and these squared errors:

| | Model | Reference |
|---|---|---|
| Winner | 0.30 | 0.50 |
| Loser | 0.60 | 0.40 |

The model improved on the winner by $-0.20$ and got worse on the loser by $+0.20$, so the bracket is $-0.20 - 0.20 = -0.40$. The loss is $-\log\sigma(0.40) = 0.5130$. When the model equals the reference, the bracket is 0 and the loss is $\log 2 = 0.6931$. In code:

```python
inside = -beta * ((err_w - err_w_ref) - (err_l - err_l_ref))
loss = -F.logsigmoid(inside).mean()
```

## Flow models and other routes

Many image generators, such as Stable Diffusion 3 and FLUX, are flow models. They follow a velocity field from noise to data with no randomness after the first draw, so each step has no density. Flow-GRPO (Liu and colleagues, 2025) fixes this by converting the deterministic ODE into an SDE that adds noise to each step and corrects the drift so that the distribution of samples at every time stays the same. Each step becomes a Gaussian, the GRPO recipe applies, and on SD3.5-M it raises GenEval accuracy from 63% to 95% and visual text rendering from 59% to 92%. DanceGRPO (Xue and colleagues, 2025) and MixGRPO (Li and colleagues, 2025) are variants of the same recipe.

Two other routes exist. DRaFT (Clark and colleagues, 2023) backpropagates a differentiable reward through the sampler, and DPOK (Fan and colleagues, 2023) adds a KL penalty between the fine-tuned and pretrained paths.

> [!RECAP]
> - Each step is $x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z$, a Gaussian action with a closed-form log-probability.
> - A trajectory has shape `(T + 1, n)`. Row 0 is $x_T$, row $T$ is $x_0$, and row $k$ to $k+1$ is step $t = T - k$.
> - DDPO's loss is PPO's clipped objective over every step, with one advantage per trajectory.
> - Diffusion-DPO compares the model's and the reference's denoising errors on the same noisy inputs.

## Build it step by step

The checks test these functions in order.

1. **`gaussian_logprob(x, mean, std)`.** Elementwise log-density, any shape. Convert `std` with `torch.as_tensor` so a float works. Common mistake: writing $-\log\sigma^2$ instead of $-\log\sigma$.
2. **`ddpm_mean(x_t, eps, t)`.** The step mean, with the schedule indexed at `t - 1`. Common mistake: using `ALPHA_BARS` where the formula wants `ALPHAS`.
3. **`rollout(model, x_T, noise)`.** Loop `k` from 0 to `T - 1` with `t = T - k`, append each state, and return `torch.stack(xs)`.
4. **`trajectory_logprob(model, xs)`.** Detach `xs`, recompute each step's mean from `model.eps`, and call `gaussian_logprob`. Return shape `(T, n)`.
5. **`ddpo_loss(logp, logp_old, advantages, clip_eps)`.** Broadcast `advantages[None, :]`, take `torch.minimum`, and negate the mean. Common mistake: `torch.max` in place of `torch.minimum`.
6. **`diffusion_dpo_loss(err_w, err_w_ref, err_l, err_l_ref, beta)`.** Use `F.logsigmoid` so large margins stay finite.
7. **`ddpo_train(...)`.** Each round, sample under `torch.no_grad()`, record `logp_old` and batch-normalized advantages, then take `epochs` Adam steps on `ddpo_loss`. Common mistake: recomputing `logp_old` inside the epoch loop, which pins the ratio at 1 and turns off the clip.

## Where this breaks

**Reward hacking.** The policy finds whatever the reward rewards. DDPO's authors report that optimizing for incompressibility eventually removes all recognizable content from the images, and that a vision-language reward was fooled by images that write the prompt's words as text. People stop early, add a KL penalty to the pretrained model, and evaluate with a reward that was not used in training. [Reward hacking, safety, and the frontier](/lessons/070-reward-hacking-and-safe-rl) returns to this.

## Further reading

- [Black and colleagues (2023), Training Diffusion Models with Reinforcement Learning](https://arxiv.org/abs/2305.13301). Defines the denoising MDP and DDPO.
- [Wallace and colleagues (2023), Diffusion Model Alignment Using Direct Preference Optimization](https://arxiv.org/abs/2311.12908). Diffusion-DPO and its loss.
- [Liu and colleagues (2025), Flow-GRPO](https://arxiv.org/abs/2505.05470). Converts a flow's ODE to an SDE so GRPO applies.
- [Clark and colleagues (2023), Directly Fine-Tuning Diffusion Models on Differentiable Rewards](https://arxiv.org/abs/2309.17400). Reward backpropagation through the sampler.
- [Fan and colleagues (2023), DPOK](https://arxiv.org/abs/2305.16381). A path-KL penalty to the pretrained model.

Researchers are still working on these open questions:

- How should credit be split across denoising steps, when early steps fix the layout and late steps fix the texture?

## The exercise

You build the denoising MDP for a 10-step, one-dimensional diffusion model whose pretrained noise prediction is exact for two-bump data. Write `gaussian_logprob` and `ddpm_mean` for one step, `rollout` to run the sampler with given noise, and `trajectory_logprob` to score a stored trajectory. Then write `ddpo_loss`, `diffusion_dpo_loss`, and `ddpo_train`.

The checks compare each piece with an independent computation: the worked values, the posterior mean of the noising process at every step, and the hand-clipped table. They check that `diffusion_dpo_loss` gives $\log 2$ at the reference and 0.5130 on the worked pair, with a gradient that matches finite differences. The last check expects DDPO to lift $P(x_0 > 1)$ from about 0.44 to above 0.9 in 20 iterations.

> [!KEY]
> A denoising chain is an MDP: each step is a Gaussian action with a closed-form log-probability, and one reward scores the end. That answers the opening question. You never need the probability of the image, only of the path, so policy gradients and PPO's clip apply to a diffusion model.
