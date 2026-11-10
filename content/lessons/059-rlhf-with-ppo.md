---
title: RLHF with PPO
slug: 059-rlhf-with-ppo
part: "Part 9 — RL for language models"
summary: Pay a KL penalty token by token, estimate that KL from samples, and run PPO on a language model, from a toy policy up to Qwen3-0.6B.
minutes: 45
gpu: true
objectives:
  - Spread the KL penalty over tokens so the task reward lands on the last token and the return equals reward minus beta times the log-ratio.
  - Compare the k1, k2, and k3 KL estimators, and say which KL each one's gradient follows when you use it as a loss.
  - "Name the pieces PPO adds for text: a value head, a response mask, and per-token means."
  - Run PPO on a toy policy to within a few percent of the exact KL-regularized optimum, and on Qwen3-0.6B against a rule-based reward.
prereqs:
  - 034-generalized-advantage-estimation
  - 036-proximal-policy-optimization
  - 055-sft-and-the-post-training-pipeline
  - 056-language-models-as-policies
  - 057-reward-models
exercise: 059-rlhf-with-ppo
---

# RLHF with PPO

Reinforcement learning from human feedback (RLHF) made chat models follow instructions. The recipe has three stages: fine-tune on example answers, fit a reward model on comparisons, then run PPO against that reward while a penalty keeps the model near where it started. This chapter builds the third stage.

You already know PPO and the leashed objective. Putting them together raises two questions specific to text. The reward arrives only after the last token, so where does the leash's price get paid? And the exact KL sums over every possible response, so how do you estimate it from the tokens you sampled?

You build the answers as functions on plain tensors. The checks run them on a toy policy, where PPO gets within a few percent of the exact optimum. On the GPU worker, the same functions train Qwen3-0.6B to describe a topic in exactly ten words.

## What you already know

- The objective $\mathbb{E}[r] - \beta\, \mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})$ has the exact optimum $\pi^* \propto \pi_{\text{ref}} \exp(r / \beta)$, with value $\beta \log Z$.
- A reward model scores a finished response with one number, $r_\phi(x, y)$.
- PPO maximizes a clipped surrogate of the probability ratio times the advantage, and GAE computes the advantages backwards from TD errors.
- A response's tokens, log-probabilities, and mask are `(B, T)` tensors.

## Put the KL penalty in the reward

The KL between whole responses splits into a sum of per-token log-ratios, because both policies multiply token probabilities. So you can charge the penalty one token at a time, and pay the reward model's score on the last token. Ziegler and colleagues (2019), Stiennon and colleagues (2020), and InstructGPT all do this:

$$
r_t = -\beta\, \big(\log \pi_{\text{old}}(y_t \mid s_t) - \log \pi_{\text{ref}}(y_t \mid s_t)\big) + \mathbf{1}[t = T - 1]\; r_\phi(x, y)
$$

Read it aloud: every token pays $\beta$ times how much more likely the current policy made it than the reference did, and the last token, $t = T - 1$, also collects the task reward. Here $\pi_{\text{old}}$ is the policy that sampled the batch. Add the $r_t$ up and you get $r_\phi(x, y) - \beta \log(\pi_{\text{old}}(y) / \pi_{\text{ref}}(y))$, whose average is the KL-regularized objective. In torch:

```python
rewards = -beta * (logp - ref_logp) * mask
last = mask.sum(dim=1).long() - 1
rewards[torch.arange(B), last] += seq_rewards
```

A response has three tokens, padded to four. The policy's log-probabilities are $(-0.5, -1.0, -0.2)$, the reference's are $(-0.7, -0.9, -0.6)$, the reward model gives 1, and $\beta = 0.1$. The log-ratios are $(0.2, -0.1, 0.4)$, so the KL terms are $(-0.02, +0.01, -0.04)$. The shaped rewards are $(-0.02, 0.01, 0.96, 0)$, with the task reward on the third token and not on the padding. They sum to $0.95 = 1 - 0.1 \times 0.5$. Figure 1 draws them.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Shaped rewards for a three-token response padded to four tokens: small KL terms on each token and a task reward of 1 added to the third token">
<line x1="90.0" y1="182.3" x2="570.0" y2="182.3" class="stroke-muted"/>
<text x="82.0" y="186.3" class="fig-small" text-anchor="end">0</text>
<line x1="90.0" y1="113.1" x2="570.0" y2="113.1" class="stroke-grid"/>
<text x="82.0" y="117.1" class="fig-small" text-anchor="end">0.5</text>
<line x1="90.0" y1="43.8" x2="570.0" y2="43.8" class="stroke-grid"/>
<text x="82.0" y="47.8" class="fig-small" text-anchor="end">1</text>
<rect x="116.0" y="182.3" width="68.0" height="2.8" class="fill-b"/>
<text x="150.0" y="202.3" class="fig-small" text-anchor="middle"></text>
<text x="150.0" y="172.3" class="fig-small fig-bold" text-anchor="middle">-0.02</text>
<text x="150.0" y="236.0" class="fig-small" text-anchor="middle">token 1</text>
<text x="150.0" y="252.0" class="fig-small" text-anchor="middle">KL term -0.02</text>
<rect x="236.0" y="180.9" width="68.0" height="2.0" class="fill-b"/>
<text x="270.0" y="174.3" class="fig-small" text-anchor="middle"></text>
<text x="270.0" y="172.3" class="fig-small fig-bold" text-anchor="middle">0.01</text>
<text x="270.0" y="236.0" class="fig-small" text-anchor="middle">token 2</text>
<text x="270.0" y="252.0" class="fig-small" text-anchor="middle">KL term +0.01</text>
<rect x="356.0" y="49.4" width="68.0" height="132.9" class="fill-c"/>
<rect x="356.0" y="182.3" width="68.0" height="2.0" class="fill-b"/>
<text x="390.0" y="174.3" class="fig-small" text-anchor="middle"></text>
<text x="390.0" y="41.4" class="fig-small fig-bold" text-anchor="middle">0.96</text>
<text x="390.0" y="236.0" class="fig-small" text-anchor="middle">token 3 (task reward 1 added)</text>
<text x="390.0" y="252.0" class="fig-small" text-anchor="middle">KL term -0.04</text>
<text x="510.0" y="172.3" class="fig-small" text-anchor="middle">0 (padding)</text>
<text x="510.0" y="236.0" class="fig-small" text-anchor="middle">padding</text>
</svg>
Figure 1. The shaped rewards of the worked example. Notice that each token's KL term is tiny next to the task reward, which lands on the last response token (0.96 after its own KL term of −0.04) and not on the padding.
```

The second token earns a small bonus, because the policy made it *less* likely than the reference did. One token's log-ratio can have either sign, but its average over the policy's samples is a KL, which is never negative.

## Estimate the KL from samples

The per-token log-ratio is one of three single-sample estimators that Schulman (2020) compares. Let $q = \pi_\theta$ be the policy you sample from and $p = \pi_{\text{ref}}$. For a sampled token, let the ratio be $\rho = p(y_t) / q(y_t)$. Then

$$
k_1 = -\log \rho,
\qquad
k_2 = \tfrac{1}{2} (\log \rho)^2,
\qquad
k_3 = (\rho - 1) - \log \rho
$$

Read it aloud: $k_1$ is the log-ratio itself, $k_2$ is half its square, and $k_3$ adds $\rho - 1$, a term whose average under $q$ is exactly zero. So $k_1$ and $k_3$ are both unbiased for $\mathrm{KL}(q \,\|\, p)$. $k_3$ is never negative, because $\log \rho \le \rho - 1$, and it usually has much lower variance when $q$ and $p$ are close. $k_2$ is biased, but it matches the KL to second order when $q \approx p$.

```python
log_r = ref_logp - logp
k1, k2, k3 = -log_r, 0.5 * log_r**2, torch.expm1(log_r) - log_r
```

Take three tokens with $q = (0.5, 0.3, 0.2)$ and $p = (0.3, 0.3, 0.4)$. The exact $\mathrm{KL}(q \,\|\, p) = 0.1168$.

| Estimator | Token 1 | Token 2 | Token 3 | Mean under $q$ | Variance under $q$ |
|---|---|---|---|---|---|
| $k_1$ | 0.5108 | 0 | $-0.6931$ | 0.1168 | 0.2129 |
| $k_2$ | 0.1305 | 0 | 0.2402 | 0.1133 | 0.0072 |
| $k_3$ | 0.1108 | 0 | 0.3069 | 0.1168 | 0.0113 |

$k_3$ has a variance 19 times smaller than $k_1$'s, with no bias. Here $k_2$'s variance is lower still, but it pays with bias. Figure 2 shows the per-token values.

```figure
<svg viewBox="0 0 600 280" role="img" aria-label="Values of the k1, k2, and k3 KL estimators on three tokens, with k1 negative on the third token and k3 never negative">
<line x1="80.0" y1="210.0" x2="570.0" y2="210.0" class="stroke-grid"/>
<text x="72.0" y="214.0" class="fig-small" text-anchor="end">-0.8</text>
<line x1="80.0" y1="158.6" x2="570.0" y2="158.6" class="stroke-grid"/>
<text x="72.0" y="162.6" class="fig-small" text-anchor="end">-0.4</text>
<line x1="80.0" y1="107.1" x2="570.0" y2="107.1" class="stroke-muted"/>
<text x="72.0" y="111.1" class="fig-small" text-anchor="end">0</text>
<line x1="80.0" y1="55.7" x2="570.0" y2="55.7" class="stroke-grid"/>
<text x="72.0" y="59.7" class="fig-small" text-anchor="end">0.4</text>
<rect x="95.7" y="41.5" width="40.0" height="65.7" class="fill-a"/>
<text x="115.7" y="36.5" class="fig-small" text-anchor="middle">0.511</text>
<rect x="141.7" y="90.4" width="40.0" height="16.8" class="fill-b"/>
<text x="161.7" y="85.4" class="fig-small" text-anchor="middle">0.130</text>
<rect x="187.7" y="92.9" width="40.0" height="14.2" class="fill-c"/>
<text x="207.7" y="87.9" class="fig-small" text-anchor="middle">0.111</text>
<text x="161.7" y="250.0" class="fig-small" text-anchor="middle">token 1: q = 0.5, p = 0.3</text>
<rect x="259.0" y="107.1" width="40.0" height="1.5" class="fill-a"/>
<text x="279.0" y="102.1" class="fig-small" text-anchor="middle">0</text>
<rect x="305.0" y="107.1" width="40.0" height="1.5" class="fill-b"/>
<text x="325.0" y="102.1" class="fig-small" text-anchor="middle">0</text>
<rect x="351.0" y="107.1" width="40.0" height="1.5" class="fill-c"/>
<text x="371.0" y="102.1" class="fig-small" text-anchor="middle">0</text>
<text x="325.0" y="250.0" class="fig-small" text-anchor="middle">token 2: q = 0.3, p = 0.3</text>
<rect x="422.2" y="107.1" width="40.0" height="89.1" class="fill-a"/>
<text x="442.2" y="210.3" class="fig-small" text-anchor="middle">-0.693</text>
<rect x="468.2" y="76.3" width="40.0" height="30.9" class="fill-b"/>
<text x="488.2" y="71.3" class="fig-small" text-anchor="middle">0.240</text>
<rect x="514.2" y="67.7" width="40.0" height="39.5" class="fill-c"/>
<text x="534.2" y="62.7" class="fig-small" text-anchor="middle">0.307</text>
<text x="488.2" y="250.0" class="fig-small" text-anchor="middle">token 3: q = 0.2, p = 0.4</text>
<rect x="160.0" y="262.0" width="12.0" height="12.0" class="fill-a"/>
<text x="178.0" y="273.0" class="fig-small" text-anchor="start">k1</text>
<rect x="250.0" y="262.0" width="12.0" height="12.0" class="fill-b"/>
<text x="268.0" y="273.0" class="fig-small" text-anchor="start">k2</text>
<rect x="340.0" y="262.0" width="12.0" height="12.0" class="fill-c"/>
<text x="358.0" y="273.0" class="fig-small" text-anchor="start">k3</text>
</svg>
Figure 2. The three estimators on each token. Notice that k1 goes negative on token 3, where the policy gave the token less probability than the reference did, while k3 stays at or above 0 and k2 is half a squared log-ratio.
```

### Estimators as losses

If you put an estimator in the loss and backpropagate, autodiff differentiates it at the sampled tokens and does not differentiate through the sampling. Work out the expected gradient under $q_\theta$:

- $k_1$: $\mathbb{E}_q[\nabla \log q] = 0$. As a loss it does nothing on average.
- $k_2$: $\mathbb{E}_q[\log(q / p)\, \nabla \log q] = \nabla\, \mathrm{KL}(q \,\|\, p)$, the gradient of the reverse KL the objective asks for.
- $k_3$: $\mathbb{E}_q[(1 - p / q)\, \nabla \log q] = \nabla\, \mathrm{KL}(p \,\|\, q)$, the gradient of the *forward* KL.

So an estimator can be unbiased for the KL's value and still follow the gradient of a different quantity. Using $k_1$ in the *reward*, as this chapter does, is correct: the policy gradient sends it through the score function, which gives the reverse-KL gradient. Shah and colleagues (2025) study how these choices change training.

> [!KEY]
> $k_1$ and $k_3$ are unbiased for the KL's value. As loss terms, $k_1$ has zero expected gradient, $k_2$ follows the reverse KL, and $k_3$ follows the forward KL. Put $k_1$ in the reward, or $k_2$ in the loss.

## The rest is PPO on masked tokens

The starter gives you the other pieces in full. The reward model speaks once at the end, but PPO needs an advantage for every token, so a *value head* (a linear layer on the language model's last hidden state) predicts $V(s_t)$ at each position. GAE runs over the response tokens with $\gamma = 1$ and $\lambda = 0.95$, and a mask turns off the bootstrap after the last response token. The clipped policy loss and value loss are per-token means under the mask, so longer responses get more weight. `ppo_update` shuffles the rows, recomputes log-probabilities and values per minibatch, adds the policy loss to `vf_coef` times the value loss, clips the gradient norm, and steps.

## Run it on Qwen3-0.6B

On the GPU worker, **Run** trains Qwen3-0.6B with your functions. The prompt asks the model to describe a topic in exactly ten words. A Python rule replaces the reward model: a response of exactly ten words scores 1, and each word too many or too few costs 0.1. Qwen3-0.6B, in non-thinking mode, writes sentences of anywhere from 3 to 20 words, so there is room to improve.

The run uses the following settings, each for a reason:

- **Sampling at temperature 1 with `top_p=1.0` and `top_k=0`.** Qwen3's default generation settings truncate the distribution. The log-probabilities in the loss then describe a different policy from the one that sampled.
- **A mask up to the first end-of-sequence token.** Everything after it is padding. The attention mask and the position IDs come from the same mask, so the training forward pass sees exactly what generation saw.
- **Float32 master weights, bfloat16 compute.** bfloat16 keeps about three significant digits, so most weights cannot represent an update of $10^{-6}$ and would not change. The frozen reference stays in bfloat16.
- **A value head on detached hidden states.** The value loss then trains only the head, and cannot pull the language model's representation around.
- **A small learning rate, $10^{-6}$.** Adam's first steps move every weight by about the learning rate, whatever its gradient. Across 600 million weights that adds up: in a test with four responses per batch, one step moved some token ratios past 2. Larger batches average out that noise.

The run prints mean reward, the fraction of exactly-ten-word responses, the mean word count, the KL, and the clip fraction every few steps. Expect the mean reward to start near 0.7 and climb within the first dozen steps, while the KL grows to a few nats.

> [!WARNING]
> If the clip fraction at the first minibatch of the first step is not 0, your old log-probabilities do not match your training forward pass. Look for a sampling setting, a dtype, dropout, or a position-ID difference.

> [!RECAP]
> - Shaped rewards are $-\beta$ times the log-ratio on every response token, plus the task reward on the last one.
> - $k_1$ and $k_3$ are unbiased for the KL. As losses, $k_1$'s expected gradient is 0, $k_2$'s is the reverse-KL gradient, and $k_3$'s is the forward-KL one.
> - The value head, masked GAE, and per-token means turn ordinary PPO into PPO for text.

## Build it step by step

The checks test your two functions, and then run the whole PPO loop on the toy policy with the given pieces. Every function takes and returns torch tensors. The starter provides `ToyPolicy`, `rising_reward`, `train_toy_ppo`, and `real_run`, which call your functions.

1. **`kl_shaped_rewards(logp, ref_logp, seq_rewards, mask, beta)`.** Returns `(B, T)`. Find each row's last response index with `mask.sum(dim=1) - 1`. The common mistake is adding the task reward at index `T - 1`, which is padding for short responses.
2. **`kl_estimators(logp, ref_logp)`.** Returns `(k1, k2, k3)`. Use `torch.expm1` for $\rho - 1$. The common mistake is flipping the ratio, so $k_1$ comes out as the negative log-ratio. One check puts each estimator in a loss and compares exact expected gradients.
3. The last check runs 40 PPO iterations on the toy policy and expects PPO to close at least 85 percent of the gap to the exact optimum $\beta \log Z$.

## Where this breaks

**Reward hacking.** A learned reward has errors, and PPO finds them: longer answers, flattery, and formatting tricks. The KL penalty slows it. Gao and colleagues (2022) found that in their setup the penalty acted mostly like early stopping.

**Mismatched log-probabilities.** Sampling with one temperature and scoring with another, dropout in training mode, a bfloat16 engine against a float32 trainer, or wrong position IDs all make $\rho \ne 1$ before any update. The first minibatch's clip fraction is the test.

**A critic that lags.** Early in training the value head knows nothing, so GAE with $\lambda < 1$ leans on bad values. Warm up the value head first, or start from the reward model's weights.

**Too large a step.** Adam normalizes each weight's update, so a large model moves a lot even at a tiny learning rate. Watch the clip fraction and the KL.

## Further reading

- [Schulman et al. (2017), Proximal policy optimization algorithms](https://arxiv.org/abs/1707.06347). Introduces the clipped surrogate.
- [Ziegler et al. (2019), Fine-tuning language models from human preferences](https://arxiv.org/abs/1909.08593). Puts the KL penalty in the reward and adapts $\beta$ toward a target KL.
- [Ouyang et al. (2022), Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155). InstructGPT, which adds a pretraining loss to limit regressions.
- [Huang et al. (2024), The N+ implementation details of RLHF with PPO](https://arxiv.org/abs/2403.17031). Lists the details that decide whether a reproduction works.
- [Schulman (2020), Approximating KL divergence](http://joschu.net/blog/kl-approx.html). Defines $k_1$, $k_2$, and $k_3$ and compares their bias and variance.
- [Shah et al. (2025), A comedy of estimators](https://arxiv.org/abs/2512.21852). Shows that common KL setups do not give the gradient of the stated objective.

Open questions researchers are working on:

- Is the critic worth its memory for language models, or do group baselines do as well?
- How should the loss aggregate over tokens so that length does not leak into the objective?

## The exercise

You build the RLHF-specific pieces: `kl_shaped_rewards` and `kl_estimators`. They work on `(B, T)` tensors, and the given `gae`, `masked_whiten`, `ppo_policy_loss`, `value_loss`, and `ppo_update` use them unchanged, in the starter's `train_toy_ppo` and `real_run`.

The checks confirm the shaped rewards on the worked example and on random masks, each estimator's value and bias exactly, that $k_3 \ge 0$ with far lower variance than $k_1$, and the exact expected gradient of each estimator as a loss. Finally, PPO from your functions must close 85 percent of the gap to $\beta \log Z$ on the toy policy in 40 iterations.
