---
title: Stale samples and importance weights
slug: 072-rl-infrastructure-for-llms
part: "Part 9 — RL for language models"
summary: Correct a policy gradient for samples that came from a stale or slightly different policy, and see what truncating the importance weights buys.
minutes: 40
gpu: false
objectives:
  - Explain why a fast sampling engine makes training off-policy, even when you intend it to be on-policy.
  - Compute the bias and variance of importance sampling, truncated importance sampling, masked importance sampling, and no correction on a three-token example.
  - Explain why a sequence-level ratio is unbiased with exploding variance, and why a token-level ratio is stable but biased.
  - Write an importance-corrected policy-gradient loss with detached weights.
prereqs:
  - 015-off-policy-importance-sampling
  - 036-proximal-policy-optimization
  - 066-group-relative-policy-optimization
exercise: 072-rl-infrastructure-for-llms
---

# Stale samples and importance weights

A fast RL system for language models has two programs. An inference engine writes thousands of responses quickly. A trainer reads those responses and updates the weights. The gradient formula you have used since the policy gradient theorem assumes the responses came from the policy being trained. Do they?

Often they do not. The engine may hold weights from a few updates ago, because training and generation overlap. Even with identical weights, the two programs use different kernels and number formats, so their probabilities differ in the last digits. The gradient then points a little in the wrong direction.

This chapter measures that error on a three-token example, then fixes it with importance weights. You will see why an exact fix explodes in variance, and why people cap the weights and accept a small bias. This chapter corrects stale and mismatched samples, and asynchronous layouts, sequence packing, and KV-cache reuse are outside it.

## What you already know

- Importance sampling estimates an average under $\pi$ from samples of $\mu$ by weighting each sample by $\pi/\mu$ (see [Learning about one policy from another](/lessons/015-off-policy-importance-sampling)).
- PPO reuses a batch for several updates and clips the ratio $\pi_\theta/\pi_{\text{old}}$ (see [Proximal policy optimization](/lessons/036-proximal-policy-optimization)).
- GRPO computes a loss over per-token log-probabilities of shape `(B, T)` (see [Group-relative policy optimization](/lessons/066-group-relative-policy-optimization)).
- The mean squared error of an estimate is its bias squared plus its variance.

## Where the gap comes from

Two separate effects make the sampling policy differ from the trained one.

**Staleness.** A training step can split one batch of rollouts into several minibatch updates, so later minibatches train on samples from a policy that has since moved. Asynchronous systems keep generating while training runs, so a sample can come from a policy several updates old. AReaL (Fu and colleagues, 2025) bounds how many versions old a sample may be, and reports up to 2.77 times faster training than a synchronous system with the same GPUs.

**Engine mismatch.** The engine and the trainer compute $\log \pi(a \mid s)$ with different kernels, reduction orders, and numerical precision. Yao and colleagues (2025) showed that, in common RL frameworks, this makes nominally on-policy training off-policy.

Call the engine's distribution $\mu$, the *behavior policy*, and the trainer's distribution $\pi$, the *target policy*. Everything below is about the gap between them.

> [!WARNING]
> The behavior log-probabilities must come from the engine that sampled. If you recompute them with the trainer, every ratio is exactly 1, and the correction silently does nothing.

## One average, two answers

Take three tokens. The engine gives them probabilities $\mu = [0.1, 0.1, 0.8]$, and the trainer gives $\pi = [0.1, 0.3, 0.6]$. Suppose a quantity $f = [1, 1, 0]$ is 1 on the first two tokens. In a policy gradient, $f$ stands for an advantage times a score. The average you want is under the trainer:

$$
\mathbb{E}_\pi[f] = 0.1 + 0.3 = 0.4
$$

But your samples come from the engine, and the plain average of $f$ over them is $\mathbb{E}_\mu[f] = 0.1 + 0.1 = 0.2$. The estimate is half the truth, and no number of samples fixes that. Figure 1 shows the two distributions.

```figure
<svg viewBox="0 0 600 260" role="img" aria-label="Bars for the engine distribution mu and the trainer distribution pi over three tokens, with ratios 1, 3, and 0.75">
  <line x1="40" y1="190" x2="560" y2="190" class="stroke-muted"/>
  <rect x="65" y="170" width="50" height="20" class="fill-a"/>
  <rect x="120" y="170" width="50" height="20" class="fill-b"/>
  <rect x="245" y="170" width="50" height="20" class="fill-a"/>
  <rect x="300" y="130" width="50" height="60" class="fill-b"/>
  <rect x="425" y="30" width="50" height="160" class="fill-a"/>
  <rect x="480" y="70" width="50" height="120" class="fill-b"/>
  <text x="90" y="164" text-anchor="middle" class="fig-small">0.1</text>
  <text x="145" y="164" text-anchor="middle" class="fig-small">0.1</text>
  <text x="270" y="164" text-anchor="middle" class="fig-small">0.1</text>
  <text x="325" y="124" text-anchor="middle" class="fig-small">0.3</text>
  <text x="450" y="24" text-anchor="middle" class="fig-small">0.8</text>
  <text x="505" y="64" text-anchor="middle" class="fig-small">0.6</text>
  <text x="118" y="212" text-anchor="middle" class="fig-bold">token 1</text>
  <text x="298" y="212" text-anchor="middle" class="fig-bold">token 2</text>
  <text x="478" y="212" text-anchor="middle" class="fig-bold">token 3</text>
  <text x="118" y="234" text-anchor="middle" class="fig-small">ratio 1, f = 1</text>
  <text x="298" y="234" text-anchor="middle" class="fig-small">ratio 3, f = 1</text>
  <text x="478" y="234" text-anchor="middle" class="fig-small">ratio 0.75, f = 0</text>
  <rect x="60" y="8" width="14" height="14" class="fill-a"/>
  <text x="80" y="20" class="fig-small">engine mu</text>
  <rect x="160" y="8" width="14" height="14" class="fill-b"/>
  <text x="180" y="20" class="fig-small">trainer pi</text>
</svg>
Figure 1. The engine under-samples token 2 by a factor of 3. Because token 2 is one of the two tokens with $f = 1$, the engine's plain average of $f$ is 0.2, not 0.4.
```

The fix is to count each sample by how much more the trainer likes it than the engine does. Token 2 is three times as likely under $\pi$ as under $\mu$, so each time it appears, count it three times. This weight, $\rho = \pi/\mu$, is the *importance ratio*. Here $\rho = [1, 3, 0.75]$. Weighting the sample average by $\rho$ gives

$$
\mathbb{E}_\mu[\rho f] = \sum_y \mu(y)\,\frac{\pi(y)}{\mu(y)}\,f(y) = \mathbb{E}_\pi[f]
$$

Read it as: the $\mu(y)$ in the sampling probability cancels the $\mu(y)$ in the denominator of the weight, leaving the average under $\pi$. Numerically, $0.1 \times 1 + 0.1 \times 3 = 0.4$. This is *importance sampling* (IS), and it is unbiased.

## Why not just use the exact weights?

An unbiased estimate can still be a bad one. The *variance* says how much one sample scatters around the estimator's own average. With weights $w$ applied to $f$, the bias and variance, taking every expectation under $\mu$, are

$$
\text{bias} = \mathbb{E}_{\mu}[w f] - \mathbb{E}_{\pi}[f], \qquad \text{variance} = \mathbb{E}_{\mu}[w^2 f^2] - \big(\mathbb{E}_{\mu}[w f]\big)^2
$$

With $n$ samples, the mean squared error is $\text{bias}^2 + \text{variance}/n$. Every expectation here is a sum weighted by $\mu$, because that is where samples come from:

```python
x = weights * f
mean = (mu * x).sum()
bias = mean - (pi * f).sum()
variance = (mu * x**2).sum() - mean**2
```

For IS, the second moment is $0.1 \times 1 + 0.1 \times 9 = 1.0$, and the variance is $1.0 - 0.4^2 = 0.84$. That is five times the variance of the uncorrected estimate, which is $0.2 - 0.04 = 0.16$. The big weight on token 2 makes each sample that lands there count for a lot, so the estimate jumps.

The remedy is to cap the weight. *Truncated importance sampling* (TIS) uses $w = \min(\rho, C)$. Take $C = 2$, so the weights are $[1, 2, 0.75]$ and $wf = [1, 2, 0]$. The mean is $0.1 \times 1 + 0.1 \times 2 = 0.3$, so the bias is $0.3 - 0.4 = -0.1$. The second moment is $0.1 \times 1 + 0.1 \times 4 = 0.5$, so the variance is $0.5 - 0.09 = 0.41$. *Masked importance sampling* (MIS) instead zeroes any weight outside a band, here $[0.5, 2]$, which drops token 2 entirely. Figure 2 and the table below give all four. IMPALA caps the weights in its V-trace targets the same way, as [IMPALA and V-trace](/lessons/038-impala-and-v-trace) shows.

| Method | Weight $w$ | Bias | Variance | MSE at $n = 16$ | MSE at $n = 1000$ |
|---|---|---|---|---|---|
| No correction | 1 | $-0.2$ | 0.16 | 0.0500 | 0.0402 |
| IS | $\rho$ | 0 | 0.84 | 0.0525 | 0.00084 |
| TIS, $C = 2$ | $\min(\rho, C)$ | $-0.1$ | 0.41 | 0.0356 | 0.0104 |
| MIS, $[0.5, 2]$ | $\rho$ inside the band, else 0 | $-0.3$ | 0.09 | 0.0956 | 0.0901 |

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Mean squared error of no correction, IS, TIS, and MIS at n equal 16 and n equal 1000; TIS is lowest at 16 and IS is lowest at 1000">
  <text x="20" y="22" class="fig-bold">Mean squared error</text>
  <line x1="40" y1="200" x2="270" y2="200" class="stroke-muted"/>
  <line x1="330" y1="200" x2="560" y2="200" class="stroke-muted"/>
  <rect x="55" y="135" width="40" height="65" class="fill-muted"/>
  <rect x="105" y="131.75" width="40" height="68.25" class="fill-a"/>
  <rect x="155" y="153.7" width="40" height="46.3" class="fill-b"/>
  <rect x="205" y="75.7" width="40" height="124.3" class="fill-d"/>
  <rect x="345" y="147.8" width="40" height="52.2" class="fill-muted"/>
  <rect x="395" y="198.9" width="40" height="1.1" class="fill-a"/>
  <rect x="445" y="186.5" width="40" height="13.5" class="fill-b"/>
  <rect x="495" y="82.9" width="40" height="117.1" class="fill-d"/>
  <text x="75" y="128" text-anchor="middle" class="fig-small">.0500</text>
  <text x="125" y="124" text-anchor="middle" class="fig-small">.0525</text>
  <text x="175" y="146" text-anchor="middle" class="fig-small">.0356</text>
  <text x="225" y="68" text-anchor="middle" class="fig-small">.0956</text>
  <text x="365" y="141" text-anchor="middle" class="fig-small">.0402</text>
  <text x="415" y="191" text-anchor="middle" class="fig-small">.0008</text>
  <text x="465" y="179" text-anchor="middle" class="fig-small">.0104</text>
  <text x="515" y="76" text-anchor="middle" class="fig-small">.0901</text>
  <text x="155" y="222" text-anchor="middle" class="fig-bold">n = 16</text>
  <text x="445" y="222" text-anchor="middle" class="fig-bold">n = 1000</text>
  <rect x="60" y="244" width="12" height="12" class="fill-muted"/>
  <text x="78" y="255" class="fig-small">none</text>
  <rect x="150" y="244" width="12" height="12" class="fill-a"/>
  <text x="168" y="255" class="fig-small">IS</text>
  <rect x="220" y="244" width="12" height="12" class="fill-b"/>
  <text x="238" y="255" class="fig-small">TIS</text>
  <rect x="300" y="244" width="12" height="12" class="fill-d"/>
  <text x="318" y="255" class="fig-small">MIS</text>
</svg>
Figure 2. The winner changes with the sample size. At $n = 16$, TIS has the lowest error because IS's variance dominates. At $n = 1000$, the variance term is small and unbiased IS wins.
```

TIS's bias has a closed form: it loses exactly the part of each weight above $C$, so $\text{bias} = -\mathbb{E}_\mu[(\rho - C)^+ f]$, where $(z)^+ = \max(z, 0)$. Check it: $-0.1 \times (3 - 2) \times 1 = -0.1$. MIS does not shrink a large weight. It drops the sample, so its bias is the part of $\mathbb{E}_\pi[f]$ that lives outside the band.

> [!KEY]
> Truncation and masking buy lower variance at the price of bias. Whether that trade pays depends on how many samples you average.

TIS for the engine gap (Yao and colleagues, 2025) multiplies each token's PPO term by $\min(\pi_{\text{old}}/\mu, C)$. IcePop (Ling Team, 2025) masks instead: it keeps a token's ratio inside $[0.5, 5]$ and gives weight 0 otherwise.

## Tokens or whole sequences?

A response is a sequence of $T$ tokens, so the exact correction is the ratio of whole-sequence probabilities:

$$
\rho(y) = \prod_{t=1}^{T} \frac{\pi(y_t \mid s_t)}{\mu(y_t \mid s_t)}
$$

Read it as: multiply every token's ratio together. Weighting the REINFORCE gradient by this product gives exactly the on-policy gradient. But the second moment of a product multiplies too. If each token has a second moment of 1.001, a sequence of 10,000 tokens has $1.001^{10000} \approx 21{,}900$. In code, sum log-ratios and exponentiate once, so the product does not overflow:

```python
seq_ratio = torch.exp(((logp_target - logp_behavior) * mask).sum(dim=1))
```

Token-level weights avoid the explosion by weighting token $t$ by its own ratio alone. But that ignores that the earlier tokens came from $\mu$ too, so a token-level weight is biased even without any truncation. The exercise checks both claims on a four-token, length-3 toy where all 64 sequences can be enumerated. With the sequence ratio, the expected gradient from $\mu$'s samples equals the true gradient to $10^{-10}$. With token ratios, it is off by 28%.

The same toy shows what staleness does without any correction. Make $\mu$ an older copy of the policy and compare the stale gradient with the true one by cosine similarity. It is 1 at lag 0, 0.88 at lag 2, 0.13 at lag 4, and $-0.67$ at lag 8. At lag 8, the update points partly backward.

> [!KEY]
> Sequence-level IS is unbiased, and its variance grows geometrically with length. Token-level IS keeps variance in check and is biased because it ignores the prefix.

The loss that uses these weights multiplies each sampled token's log-probability by a constant weight and its sequence's advantage:

$$
\mathcal{L}(\theta) = -\frac{1}{B} \sum_{i=1}^{B} \sum_{t=1}^{T} m_{i,t}\, w_{i,t}\, \hat A_i \log \pi_\theta(y_{i,t} \mid s_{i,t})
$$

Read it as: for each sequence, sum over its response tokens the weight times the advantage times the log-probability, average over sequences, and flip the sign. At token level, $w_{i,t} = \min(\rho_{i,t}, C)$. At sequence level, $w_{i,t} = \min(\rho_i, C)$ for every token of sequence $i$. The weights come from detached log-probabilities:

```python
log_rho = (logp.detach() - logp_behavior) * mask
w = torch.clamp(torch.exp(log_rho), max=cap)                  # token level
loss = -(mask * w * adv[:, None] * logp).sum(dim=1).mean()
```

> [!RECAP]
> - Bias is $\mathbb{E}_\mu[wf] - \mathbb{E}_\pi[f]$ and variance is $\mathbb{E}_\mu[w^2f^2] - (\mathbb{E}_\mu[wf])^2$, both weighted by $\mu$.
> - IS uses $w = \rho$, TIS uses $\min(\rho, C)$, and MIS keeps $\rho$ inside a band and zeroes it outside.
> - The sequence ratio is $\exp$ of the masked sum of log-ratios.
> - The corrected loss detaches the weights, sums over tokens, and averages over sequences.

## Build it step by step

The checks test the functions in this order.

1. **`tis_weights(ratio, cap)`** and **`mis_weights(ratio, low, high)`.** Elementwise, any shape. Use `torch.clamp` and `torch.where`. Common mistake: clamping from below too.
2. **`estimator_bias_variance(mu, pi, f, weights)`.** Four `(M,)` tensors in, two floats out. Every expectation is weighted by `mu`. Common mistake: computing the variance with `pi` as the weights.
3. **`sequence_ratio(logp_target, logp_behavior, mask)`.** `(B, T)` in, `(B,)` out. Sum masked log-ratios, then exponentiate. Common mistake: forgetting the mask, so padding tokens count.
4. **`is_pg_loss(logp, logp_behavior, mask, advantages, level, cap)`.** Compute ratios from `logp.detach()`. For sequence level, sum the masked log-ratios with `keepdim=True`, exponentiate, clamp, and expand to `(B, T)`. Common mistake: letting gradient flow through the weights, which adds a second term to the gradient and breaks the unbiasedness check.

## Where this breaks

**Truncation bias adds up.** Each truncated token loses a little weight, and long responses have many tokens. A run can look stable and still drift toward the engine's policy. Log the mean $k_3$ estimator, $\rho - 1 - \log\rho$, which averages to the KL between the engines, and log the fraction of truncated tokens.

**Reduce the gap at the source.** Qi and colleagues (2025) switch the trainer from BF16 to FP16. FP16 keeps 10 mantissa bits to BF16's 7, so its rounding errors are 8 times smaller. A KL grows with the square of a small perturbation, so the mismatch shrinks about 64 times. They report that this makes the plain sequence-level correction stable.

## Further reading

- [Yao and colleagues (2025), Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://fengyao.notion.site/off-policy-rl). Measures the engine gap and proposes TIS.
- [Liu, Li, and colleagues (2025), When Speed Kills Stability](https://richardli.xyz/rl-collapse). Why token-level corrections are biased, and sequence-level TIS and MIS.
- [Ling Team (2025), Every Step Evolves](https://arxiv.org/abs/2510.18855). The Ring-1T report, which introduces IcePop.
- [Qi and colleagues (2025), Defeating the Training-Inference Mismatch via FP16](https://arxiv.org/abs/2510.26788). Switching precision instead of correcting the estimator.
- [Fu and colleagues (2025), AReaL](https://arxiv.org/abs/2505.24298). A fully asynchronous RL system with bounded staleness.

Researchers are still working on these open questions:

- Is there a correction with bounded variance whose bias does not grow with the length of the response?
- How much of the engine gap can deterministic, batch-invariant kernels remove, and at what cost in throughput?

## The exercise

You build the corrections and measure them. Write `tis_weights`, `mis_weights`, `estimator_bias_variance`, `sequence_ratio`, and `is_pg_loss`.

The checks confirm the weights on the engine example and the bias and variance table. On random distributions, they confirm that IS has zero bias, that TIS's bias is $-\mathbb{E}_\mu[(\rho - C)^+ f]$, and that TIS wins at $n = 16$ while IS wins at $n = 1000$. They check that the sequence ratio averages 1 with a second moment of $(\mathbb{E}_\mu[\rho^2])^T$. By enumerating all 64 toy sequences, they check that sequence-level IS recovers the on-policy gradient while token-level IS is off by more than 5%, and that the uncorrected stale gradient turns away as the lag grows.

> [!KEY]
> Samples from a fast engine come from a policy that is not quite the one you train. Importance weights repair the average exactly but can have huge variance, so systems cap them and accept a small bias. That answers the opening question: the responses are off-policy, and this is how you correct for it.
