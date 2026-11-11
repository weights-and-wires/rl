---
title: Asynchronous RL for language models
slug: 079-rl-infrastructure-for-llms
part: "Part 9 — RL for language models"
summary: Overlap sampling and training with a staleness cap, then correct the samples that come back stale, mismatched by bf16, or packed together.
minutes: 40
gpu: false
objectives:
  - Simulate an asynchronous sampler and trainer with a maximum-staleness cap K, and compute utilization exactly against a synchronous loop.
  - Measure the log-probability gap between a bf16 sampler and an fp32 trainer, and correct it with truncated importance weights.
  - Compute the bias and variance of importance sampling, truncated importance sampling, and masked importance sampling.
  - Pack variable-length rollouts into one row so the loss equals the unpacked sum to 1e-10.
prereqs:
  - 015-off-policy-importance-sampling
  - 038-impala-and-v-trace
  - 065-sft-and-the-post-training-pipeline
  - 071-group-relative-policy-optimization
exercise: 079-rl-infrastructure-for-llms
---

# Asynchronous RL for language models

Picture a cluster training a reasoning model. One step has two phases. An inference engine writes thousands of long responses, then a trainer reads them and updates the weights. If the phases take turns, one group of GPUs always waits for the other.

The fix is to let the engine keep generating while the trainer trains. But the engine then writes with weights a version or two old, and the policy gradient formula assumes the responses came from the policy being trained. Even with identical weights, the engine and the trainer use different kernels and number formats, so their probabilities differ in the last digits.

This chapter builds the overlap, then pays for it. You simulate the queue with a cap on staleness, measure the bf16 mismatch, repair both with importance weights, and pack the responses so the trainer wastes no work.

## What you already know

- Importance sampling weights samples from $\mu$ by $\pi/\mu$ to estimate an average under $\pi$ (see [Learning about one policy from another](/lessons/015-off-policy-importance-sampling)).
- IMPALA runs many actors against one learner and clips ratios in its V-trace targets (see [IMPALA and V-trace](/lessons/038-impala-and-v-trace)).
- Packing concatenates examples with a block-diagonal mask (see [Supervised fine-tuning and the post-training pipeline](/lessons/065-sft-and-the-post-training-pipeline)).
- GRPO's loss uses per-token log-probabilities of shape `(B, T)` and one advantage per response (see [Group-relative policy optimization](/lessons/071-group-relative-policy-optimization)).

## A sampler and a trainer with a queue

Call the policy after $v$ updates *version* $v$. Update $j$ starts from version $j$ and produces version $j+1$. It consumes batch $j$. If the sampler generated batch $j$ with version $v_j$, the batch is $j - v_j$ versions *stale* when the trainer uses it.

Take four batches, a generation time of 1 per batch, and a training time of 2 per update. The synchronous loop generates batch $j$, trains on it, and only then generates batch $j+1$. The run takes $4 \times (1 + 2) = 12$ time units, and the trainer is busy for 8 of them, which is 67%.

The asynchronous loop lets the sampler start batch $j+1$ while update $j$ runs. The sampler uses the newest version finished by its start time, so batch 1 starts at time 1 with version 0, and update 1 consumes it at version 1: it is 1 version stale. Without a cap, the trainer is slower than the sampler, so the sampler gets further ahead and the staleness keeps growing: it is $[0, 1, 2, 2]$ after four batches.

The cap fixes that. With a *maximum staleness* $K$, the sampler may not start batch $j$ until version $j - K$ exists, which means update $j - K - 1$ has finished. With $K = 1$ the schedule is:

| Batch | Sampler starts | Version used | Update starts | Staleness |
|---|---|---|---|---|
| 0 | 0 | 0 | 1 | 0 |
| 1 | 1 | 0 | 3 | 1 |
| 2 | 3 | 1 | 5 | 1 |
| 3 | 5 | 2 | 7 | 1 |

The last update finishes at time 9, so the trainer is busy $8/9 = 89\%$ of the run. Capping at $K = 1$ costs nothing here, because the trainer never waits after its first batch. Figure 1 draws both schedules.

```figure
<svg viewBox="0 0 600 285" role="img" aria-label="Timelines of one sampler and one trainer for four batches. With K = 0 the machines take turns and the run ends at time 12. With K = 1 they overlap and the run ends at time 9.">
  <text x="20" y="28" class="fig-bold">K = 0, synchronous: trainer busy 67%</text>
  <text x="20" y="66" class="fig-small">sampler</text>
  <text x="20" y="104" class="fig-small">trainer</text>
  <rect x="70" y="50" width="42" height="24" class="fill-a"/>
  <text x="91" y="67" text-anchor="middle" class="fig-small">b0</text>
  <rect x="112" y="84" width="82" height="24" class="fill-b"/>
  <text x="154" y="101" text-anchor="middle" class="fig-small">u0, stale 0</text>
  <rect x="196" y="50" width="42" height="24" class="fill-a"/>
  <text x="217" y="67" text-anchor="middle" class="fig-small">b1</text>
  <rect x="238" y="84" width="82" height="24" class="fill-b"/>
  <text x="280" y="101" text-anchor="middle" class="fig-small">u1, stale 0</text>
  <rect x="322" y="50" width="42" height="24" class="fill-a"/>
  <text x="343" y="67" text-anchor="middle" class="fig-small">b2</text>
  <rect x="364" y="84" width="82" height="24" class="fill-b"/>
  <text x="406" y="101" text-anchor="middle" class="fig-small">u2, stale 0</text>
  <rect x="448" y="50" width="42" height="24" class="fill-a"/>
  <text x="469" y="67" text-anchor="middle" class="fig-small">b3</text>
  <rect x="490" y="84" width="82" height="24" class="fill-b"/>
  <text x="532" y="101" text-anchor="middle" class="fig-small">u3, stale 0</text>
  <text x="20" y="140" class="fig-bold">K = 1, asynchronous: trainer busy 89%</text>
  <text x="20" y="178" class="fig-small">sampler</text>
  <text x="20" y="216" class="fig-small">trainer</text>
  <rect x="70" y="162" width="42" height="24" class="fill-a"/>
  <text x="91" y="179" text-anchor="middle" class="fig-small">b0</text>
  <rect x="112" y="196" width="82" height="24" class="fill-b"/>
  <text x="154" y="213" text-anchor="middle" class="fig-small">u0, stale 0</text>
  <rect x="112" y="162" width="42" height="24" class="fill-a"/>
  <text x="133" y="179" text-anchor="middle" class="fig-small">b1</text>
  <rect x="196" y="196" width="82" height="24" class="fill-b"/>
  <text x="238" y="213" text-anchor="middle" class="fig-small">u1, stale 1</text>
  <rect x="196" y="162" width="42" height="24" class="fill-a"/>
  <text x="217" y="179" text-anchor="middle" class="fig-small">b2</text>
  <rect x="280" y="196" width="82" height="24" class="fill-b"/>
  <text x="322" y="213" text-anchor="middle" class="fig-small">u2, stale 1</text>
  <rect x="280" y="162" width="42" height="24" class="fill-a"/>
  <text x="301" y="179" text-anchor="middle" class="fig-small">b3</text>
  <rect x="364" y="196" width="82" height="24" class="fill-b"/>
  <text x="406" y="213" text-anchor="middle" class="fig-small">u3, stale 1</text>
  <line x1="70" y1="250" x2="574" y2="250" class="stroke-muted"/>
  <line x1="70" y1="244" x2="70" y2="250" class="stroke-muted"/>
  <text x="70" y="266" text-anchor="middle" class="fig-small">0</text>
  <line x1="154" y1="244" x2="154" y2="250" class="stroke-muted"/>
  <text x="154" y="266" text-anchor="middle" class="fig-small">2</text>
  <line x1="238" y1="244" x2="238" y2="250" class="stroke-muted"/>
  <text x="238" y="266" text-anchor="middle" class="fig-small">4</text>
  <line x1="322" y1="244" x2="322" y2="250" class="stroke-muted"/>
  <text x="322" y="266" text-anchor="middle" class="fig-small">6</text>
  <line x1="406" y1="244" x2="406" y2="250" class="stroke-muted"/>
  <text x="406" y="266" text-anchor="middle" class="fig-small">8</text>
  <line x1="490" y1="244" x2="490" y2="250" class="stroke-muted"/>
  <text x="490" y="266" text-anchor="middle" class="fig-small">10</text>
  <line x1="574" y1="244" x2="574" y2="250" class="stroke-muted"/>
  <text x="574" y="266" text-anchor="middle" class="fig-small">12</text>
  <text x="574" y="282" text-anchor="end" class="fig-small">time</text>
</svg>
Figure 1. The same four batches with K = 0 and K = 1. The boxes on the K = 1 timeline overlap, so the run ends at time 9 instead of 12. Notice that every batch after the first is 1 version stale, and the cap never lets it reach 2.
```

When generation is the slow phase instead (3 per batch, 2 per update), the run takes 14 units instead of 20, and the sampler is busy 86% of it. Any overlap costs one version of staleness.

> [!KEY]
> A staleness cap $K$ turns the question "how off-policy is my data?" into a number you choose. $K = 0$ is the synchronous loop. A larger $K$ buys utilization until one machine is the bottleneck, and no more.

AReaL (Fu and colleagues, 2025) runs this design with a staleness bound and reports up to 2.77 times faster training than a synchronous system with the same GPUs. Noukhovitch and colleagues (2024) find that online DPO tolerates the most stale data of the algorithms they test. PipelineRL (Piché and colleagues, 2025) updates weights in the middle of a generation. Such a *partial rollout* has tokens from two versions, so each token's behavior log-probability must be saved when it is sampled.

## One average, two answers

The correction needs a clean picture first. Take three tokens. The sampler gives them probabilities $\mu = [0.1, 0.1, 0.8]$, the trainer gives $\pi = [0.1, 0.3, 0.6]$, and a quantity $f = [1, 1, 0]$ stands in for an advantage times a score. The average you want is $\mathbb{E}_\pi[f] = 0.1 + 0.3 = 0.4$. Your samples come from the sampler, so their plain average is $\mathbb{E}_\mu[f] = 0.2$, half the truth, and no number of samples fixes that. Figure 2 shows both.

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
  <text x="80" y="20" class="fig-small">sampler mu</text>
  <rect x="160" y="8" width="14" height="14" class="fill-b"/>
  <text x="180" y="20" class="fig-small">trainer pi</text>
</svg>
Figure 2. The sampler under-samples token 2 by a factor of 3. Because token 2 is one of the two tokens with $f = 1$, the sampler's plain average of $f$ is 0.2, not 0.4.
```
Count each sample by how much more the trainer likes it than the sampler does. Token 2 is three times as likely under $\pi$, so count it three times. The weight $\rho = \pi/\mu = [1, 3, 0.75]$ is the *importance ratio*, and

$$
\mathbb{E}_\mu[\rho f] = \sum_y \mu(y)\,\frac{\pi(y)}{\mu(y)}\,f(y) = \mathbb{E}_\pi[f]
$$

The $\mu(y)$ in the sampling probability cancels the one in the weight, and numerically $0.1 \times 1 + 0.1 \times 3 = 0.4$. This is *importance sampling* (IS), and it is unbiased.

An unbiased estimate can still be a bad one. With weights $w$, and every expectation under $\mu$,

$$
\text{bias} = \mathbb{E}_{\mu}[w f] - \mathbb{E}_{\pi}[f], \qquad \text{variance} = \mathbb{E}_{\mu}[w^2 f^2] - \big(\mathbb{E}_{\mu}[w f]\big)^2
$$

and with $n$ samples the mean squared error is $\text{bias}^2 + \text{variance}/n$. For IS the second moment is $0.1 \times 1 + 0.1 \times 9 = 1.0$, so the variance is $1.0 - 0.4^2 = 0.84$, five times the uncorrected 0.16.

The remedy is to cap the weight. *Truncated importance sampling* (TIS) uses $w = \min(\rho, C)$. With $C = 2$, the weights are $[1, 2, 0.75]$, the mean is 0.3, the bias is $-0.1$, and the variance is $0.5 - 0.09 = 0.41$. *Masked importance sampling* (MIS) zeroes any weight outside a band, here $[0.5, 2]$.

| Method | Weight $w$ | Bias | Variance | MSE at $n = 16$ | MSE at $n = 1000$ |
|---|---|---|---|---|---|
| No correction | 1 | $-0.2$ | 0.16 | 0.0500 | 0.0402 |
| IS | $\rho$ | 0 | 0.84 | 0.0525 | 0.00084 |
| TIS, $C = 2$ | $\min(\rho, C)$ | $-0.1$ | 0.41 | 0.0356 | 0.0104 |
| MIS, $[0.5, 2]$ | $\rho$ inside the band, else 0 | $-0.3$ | 0.09 | 0.0956 | 0.0901 |

Truncation buys lower variance for bias: TIS wins at 16 samples and IS wins at 1000. TIS's bias is $-\mathbb{E}_\mu[(\rho - C)^+ f]$, the weight it cuts off.

### Tokens or whole sequences

A response has $T$ tokens, so the exact correction is the product of the token ratios. It recovers the on-policy gradient exactly, but its second moment multiplies: a per-token second moment of 1.001 gives $1.001^{10000} \approx 21{,}900$ at 10,000 tokens. A token-level weight avoids the explosion, but it ignores that the earlier tokens came from $\mu$ too, so it is biased. On a toy with all 64 sequences enumerated, the sequence ratio matches the true gradient to $10^{-10}$ and token ratios are off by 28%. Uncorrected stale gradients drift: their cosine with the true gradient is 0.88 at lag 2 and $-0.67$ at lag 8.

The corrected loss multiplies each sampled token's log-probability by a constant weight and its sequence's advantage. The weights come from detached log-probabilities, so no gradient flows through them:

```python
log_rho = (logp.detach() - logp_behavior) * mask
w = torch.clamp(torch.exp(log_rho), max=cap)                  # token level
loss = -(mask * w * adv[:, None] * logp).sum(dim=1).mean()
```

At sequence level, sum `log_rho` over tokens first. Yao and colleagues (2025) truncate each token's ratio this way, as IMPALA does.

> [!WARNING]
> The behavior log-probabilities must come from the engine that sampled. If you recompute them with the trainer, every ratio is exactly 1, and the correction silently does nothing.

## The same gap at equal weights

Staleness is not the only source of $\mu \ne \pi$. With identical weights, the sampler still runs fused kernels with a different reduction order, and in bf16, which keeps 8 significant bits. A logit of size 3 rounds with an error of about 0.006, small but not random.

Measure the gap as the KL divergence from the sampler to the trainer at each position, $\sum_v \mu(v)(\log\mu(v) - \log\pi(v))$, and as the largest log-probability difference. To emulate the sampler on a CPU, round the logits through bf16:

```python
low = logits.to(torch.bfloat16).to(torch.float64)
kl, max_gap = logprob_mismatch(logits, low)
```

On 50 positions with 32 tokens and logits of scale 3, the mean KL is $2.0 \times 10^{-5}$ and the largest gap is 0.031.

Does the gap matter? Estimate $\mathbb{E}_\pi[f]$ for random $f$. Without correction, the mean absolute bias is $3.1 \times 10^{-3}$. IS has zero bias. TIS with $C = 1.02$ cuts it to $2.3 \times 10^{-7}$, and costs almost no variance, because the ratios stay near 1 and truncation rarely fires.

Rounding the logits is a lower bound, because a real engine also rounds activations and partial sums.

## Packing rollouts

Responses in a group vary in length, so padding every row to the longest wastes trainer time. *Packing* puts several rollouts in one row, as in [Supervised fine-tuning and the post-training pipeline](/lessons/065-sft-and-the-post-training-pipeline): position ids restart at 0 for each rollout, and the attention mask is a causal block per rollout.

RL adds a requirement. Each rollout keeps its own advantage and normalization. Position $i$ of rollout $k$ predicts the next token with weight

$$
\text{weight}_i = \frac{A_k \, m_{i+1}}{\sum_t m_{k,t}}
$$

where $m$ is the response mask. Prompt targets and the last position of a rollout get weight 0. Dividing by the rollout's own response length is GRPO's per-sequence mean. Dividing by the whole row's length lets a long neighbor change a short rollout's weight.

Take $(1, 2, 3, 4)$ with a two-token response and advantage 1, and $(5, 6, 7)$ with a two-token response and advantage $-0.5$. Their separate losses on a small attention model are 3.9424 and $-1.2439$, which sum to 2.6986. Packed into one row, the loss is 2.6986 to $10^{-10}$. One causal mask over the row gives 2.9648, and position ids that do not restart give 2.6326. Neither raises an error.

## Where the programs run

In a *colocated* layout, the sampler and the trainer share GPUs and take turns, swapping weights or optimizer state out of the way. In a *disaggregated* layout, each has its own GPUs and they run at once. verl and OpenRLHF both offer a hybrid colocated engine, and both offer asynchronous trainers.

A worked number shows the trade. Take 8 GPUs, 48 GPU-seconds of generation and 24 GPU-seconds of training per step, and assume linear scaling. Colocated, the step takes $48/8 + 24/8 = 9$ seconds plus 1 second to swap, which is 10. Disaggregated with 5 sampler GPUs and 3 trainer GPUs, generation takes $48/5 = 9.6$ seconds and training takes $24/3 = 8$, so the step takes 9.6 seconds, the trainer waits 1.6, and the data is 1 version stale. Disaggregation pays when the swap is expensive.

*KV-cache reuse* is the other saving. A group of 8 responses shares one prompt. With a 1000-token prompt, prefix caching prefills it once instead of 8 times and saves 7000 tokens per group. After a mid-generation weight update, the cache holds old-weight keys and values, and systems differ on whether to keep or recompute it.

> [!RECAP]
> - Update $j$ consumes batch $j$ and a batch is $j - v_j$ versions stale. A cap $K$ makes the sampler wait for version $j - K$.
> - Utilization is busy time divided by the makespan.
> - Bias and variance of a weighted estimate are both sums weighted by $\mu$.
> - The mismatch between a bf16 sampler and an fp32 trainer is a small KL. Truncated weights near 1 remove its bias.
> - Packing needs restarted positions, a block-diagonal mask, and per-rollout weights.

## Build it step by step

The checks test the functions in this order.

1. **`tis_weights(ratio, cap)`** and **`mis_weights(ratio, low, high)`.** Elementwise. Use `torch.clamp` and `torch.where`. Common mistake: clamping from below too.
2. **`estimator_bias_variance(mu, pi, f, weights)`.** Every expectation is weighted by `mu`. Common mistake: using `pi` for the variance.
3. **`sequence_ratio(logp_target, logp_behavior, mask)`.** Sum masked log-ratios, then exponentiate. Common mistake: forgetting the mask.
4. **`is_pg_loss(logp, logp_behavior, mask, advantages, level, cap)`.** Take ratios from `logp.detach()`. Common mistake: letting gradient flow through the weights.
5. **`async_schedule(n_updates, gen_time, train_time, max_staleness)`.** Loop over batches. Start batch $j$ at the sampler's free time or at the finish of update $j - K - 1$, whichever is later, and count finishes at or before that start to get the version. Return a dict with `versions`, `staleness`, `makespan`, `trainer_utilization`, and `sampler_utilization`. Common mistake: gating on the wrong index.
6. **`logprob_mismatch(logits_fp32, logits_lowprec)`.** Return the per-position KL, shape `(N,)`, and the largest absolute gap as a float. Use `log_softmax`.
7. **`pack_rollouts(token_rows, response_rows, advantages)`** and **`rollout_loss(logits, targets, weights)`.** Return tokens, restarted positions, the block-diagonal mask, shifted targets, and weights. The loss is a weighted sum of $-\log\pi$ at the targets. The model `toy_logits` is given. Common mistake: one causal mask over the whole row.

## Where this breaks

**Truncation bias adds up.** Log the fraction of truncated tokens, because a long response has many chances to lose weight.

**Reduce the gap at the source.** Qi and colleagues (2025) train in FP16, whose rounding errors are 8 times smaller than BF16's, so the KL is about 64 times smaller.

## Further reading

- [Lambert, RLHF Book, the reinforcement learning chapter](https://rlhfbook.com/c/06-policy-gradients). Its section "Asynchronous RL Systems" covers learner and actor nodes and packing. I confirmed the section title, not the chapter number.
- [Stanford CS336, Spring 2025](http://cs336.stanford.edu/spring2025/). Lectures 15 to 17 are "Alignment - SFT/RLHF" and two lectures "Alignment - RL". None is titled "RL systems".
- [Espeholt and colleagues (2018), IMPALA](https://arxiv.org/abs/1802.01561).
- [Noukhovitch and colleagues (2024), Asynchronous RLHF](https://arxiv.org/abs/2410.18252).
- [Piché and colleagues (2025), PipelineRL](https://arxiv.org/abs/2509.19128).
- [Fu and colleagues (2025), AReaL](https://arxiv.org/abs/2505.24298).
- [Yao and colleagues (2025), Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://fengyao.notion.site/off-policy-rl). Proposes TIS for the engine gap. I could not load this page, so I did not recheck it.
- [verl](https://verl.readthedocs.io/en/latest/) and [OpenRLHF](https://openrlhf.readthedocs.io/en/latest/).

Researchers are still working on these open questions:

- How stale can a sample be before correcting it costs more than the throughput it buys?
- Is there a correction with bounded variance whose bias does not grow with response length?

## The exercise

You build the queue, the corrections, and the packing: `tis_weights`, `mis_weights`, `estimator_bias_variance`, `sequence_ratio`, `is_pg_loss`, `async_schedule`, `logprob_mismatch`, `pack_rollouts`, and `rollout_loss`.

The checks match the bias and variance table, the staleness $[0, 1, 1, 1]$, and exact utilization. They confirm that sequence-level IS recovers the on-policy gradient, that truncation removes the bf16 bias, and that packed and unpacked losses agree to $10^{-10}$.

> [!KEY]
> Overlapping the sampler and the trainer buys utilization, and a staleness cap $K$ sets what it costs. The samples that come back are off-policy through staleness and through bf16 rounding, and importance weights repair both, with a cap that trades a little bias for bounded variance. That answers the opening question: the responses are off-policy, and this is how you correct for it and keep the GPUs busy.
