---
title: Infrastructure for LLM reinforcement learning
slug: 42-rl-infrastructure-for-llms
part: "Part 8 — Reinforcement learning for language models"
summary: See why fast RL systems train off-policy, and correct for stale rollouts and engine mismatch with truncated importance sampling.
minutes: 55
gpu: false
objectives:
  - Explain how asynchronous rollouts and separate inference engines make nominally on-policy training off-policy.
  - Compute the exact bias and variance of IS, truncated IS, masked IS, and no correction on a toy.
  - Show that sequence-level ratios are unbiased but explode in variance, while token-level ratios are biased.
  - Describe TIS, IcePop, masked importance sampling, and the FP16 finding, and what each trades.
prereqs:
  - 06-monte-carlo-methods
  - 19-proximal-policy-optimization
  - 37-group-relative-policy-optimization
exercise: 42-rl-infrastructure-for-llms
---

# Infrastructure for LLM reinforcement learning

In RL for language models, most of the wall-clock time goes to generating
text, not to computing gradients. So systems split the work between two
programs. A fast inference engine, such as vLLM or SGLang, samples the
responses. A training engine, such as FSDP or Megatron, computes the loss and
updates the weights. The two run on different code, often on different GPUs,
and sometimes at the same time.

That split quietly breaks an assumption behind every policy-gradient method in
this part. The algorithms assume that the samples came from the policy being
trained. In a real system, they came from a slightly different policy. The
engine may hold weights a few updates old. Even with identical weights, it
computes probabilities with different kernels, precision, and summation order,
so its numbers don't match the trainer's.

This lesson measures what that costs and how people correct for it. You
compute, exactly on a toy, the bias and variance of importance sampling,
truncated importance sampling, masking, and no correction at all. You see why
correcting per token is biased and correcting per sequence explodes. You also
see why switching from BF16 to FP16 shrinks the problem at its source.

## What you already know

- Importance sampling estimates an average under $\pi$ from samples of
  $\mu$ by weighting each sample by $\pi / \mu$. Ordinary IS is unbiased but
  can have huge variance (`06-monte-carlo-methods`).
- PPO reuses a batch for several updates and clips the ratio
  $\pi_\theta / \pi_{\text{old}}$ (`19-proximal-policy-optimization`).
- GRPO computes a loss over per-token log-probabilities of shape `(B, T)`, with
  per-token importance ratios (`37-group-relative-policy-optimization`).
- The $k_3$ estimator $\rho - 1 - \log \rho$ estimates a KL divergence from
  samples (`35-rlhf-with-ppo`).
- A bias-variance trade-off is judged by the mean squared error, bias squared
  plus variance over the number of samples.

## Where the off-policyness comes from

Two separate effects make the sampling policy differ from the trained one.

**Staleness.** A training step can split one large batch of rollouts into
several minibatch updates, so later minibatches train on samples from a policy
that has since moved. Asynchronous systems go further. They keep generating
while training runs, so a sample can come from a policy several updates old.
AReaL, from Fu and colleagues (2025), bounds how many versions old a sample
may be and reports up to 2.77 times faster training than a synchronous system
with the same GPUs. Noukhovitch and colleagues (2024) found online DPO the
most robust to such stale data, and trained about 40% faster on a chatbot task
while matching final performance.

**Training-inference mismatch.** The inference engine and the trainer compute
$\log \pi(a \mid s)$ with different kernels, reduction orders, and numerical
precision. Mixture-of-experts models add another source, because a tiny
numerical difference can route a token to different experts. Yao and
colleagues (2025) showed that, in common RL frameworks, this makes nominally
on-policy training off-policy.

The result is a three-policy picture:

- The behaviour policy $\mu$ is the engine that sampled the tokens.
- The proximal policy $\pi_{\text{old}}$ is the trainer's copy at the start of
  the update. PPO's clip is centred on it.
- The current policy $\pi_\theta$ is the one taking gradient steps.

The ratio $\pi_\theta / \pi_{\text{old}}$ is PPO's business. This lesson is
about $\pi_{\text{old}} / \mu$, the gap between the trainer and the engine.
Hilton and colleagues (2021) introduced the decoupled objective that keeps the
two apart, and AReaL uses it.

> [!WARNING]
> The behaviour log-probabilities must come from the engine that sampled. If
> you recompute them with the trainer, every ratio is exactly 1 and the
> correction does nothing.

## Weight, truncate, or mask

Say you want the average of some quantity $f(y)$ under the trainer's
distribution $\pi$, but your samples come from $\mu$. In a policy gradient,
$f$ is an advantage times a score. Every correction is a weight $w(y)$ that
multiplies each sample, and the estimate is the average of $w(y) f(y)$ over
samples $y \sim \mu$. Its bias and variance are

$$
\text{bias} = \mathbb{E}_{\mu}[w f] - \mathbb{E}_{\pi}[f], \qquad \text{variance} = \mathbb{E}_{\mu}[w^2 f^2] - \big(\mathbb{E}_{\mu}[w f]\big)^2
$$

Read it as: the bias is how far the estimator's average sits from the true
average under $\pi$. The variance is how much one sample scatters around the
estimator's own average. With $n$ samples, the mean squared error is
$\text{bias}^2 + \text{variance} / n$. On a finite toy, every expectation is a
sum weighted by $\mu$:

```python
x = weights * f
mean = (mu * x).sum()
bias = mean - (pi * f).sum()
variance = (mu * x**2).sum() - mean**2
```

The weights differ by method. The ratio is $\rho(y) = \pi(y) / \mu(y)$:

| Method | Weight $w(y)$ | Bias |
|---|---|---|
| No correction | 1 | $\mathbb{E}_\mu[f] - \mathbb{E}_\pi[f]$ |
| Importance sampling (IS) | $\rho$ | 0 |
| Truncated IS (TIS) | $\min(\rho, C)$ | $-\mathbb{E}_\mu[(\rho - C)^+ f]$ |
| Masked IS (MIS) | $\rho$ if $\alpha \le \rho \le \beta$, else 0 | $-\mathbb{E}_\pi[f\, \mathbf{1}(\rho \notin [\alpha, \beta])]$ |

Here $(z)^+ = \max(z, 0)$. IS is unbiased because $\mathbb{E}_\mu[\rho f] = \sum_y \mu(y) \frac{\pi(y)}{\mu(y)} f(y) = \mathbb{E}_\pi[f]$.
TIS caps each weight at a constant $C$, so it loses exactly the part of each
weight above $C$. That's where its bias formula comes from. MIS doesn't shrink
a large weight. It drops the sample.

### Worked example

Take three tokens. The engine gives them $\mu = [0.1, 0.1, 0.8]$ and the
trainer gives them $\pi = [0.1, 0.3, 0.6]$, so the ratios are
$\rho = [1, 3, 0.75]$. Let $f = [1, 1, 0]$, so the true average is
$\mathbb{E}_\pi[f] = 0.1 + 0.3 = 0.4$. The following table gives the exact
numbers:

| Method | $\mathbb{E}_\mu[w f]$ | Bias | Variance | MSE, $n = 16$ | MSE, $n = 1000$ |
|---|---|---|---|---|---|
| No correction | 0.2 | $-0.2$ | 0.16 | 0.0500 | 0.0402 |
| IS | 0.4 | 0 | 0.84 | 0.0525 | 0.00084 |
| TIS, $C = 2$ | 0.3 | $-0.1$ | 0.41 | 0.0356 | 0.0104 |
| MIS, $[0.5, 2]$ | 0.1 | $-0.3$ | 0.09 | 0.0956 | 0.0901 |

Work through TIS by hand. The weights are $[1, 2, 0.75]$, so $w f = [1, 2, 0]$.
The mean is $0.1 \times 1 + 0.1 \times 2 = 0.3$, and the bias is
$0.3 - 0.4 = -0.1$. That matches $-\mathbb{E}_\mu[(\rho - C)^+ f] = -0.1 \times (3 - 2) \times 1 = -0.1$.
The second moment is $0.1 \times 1 + 0.1 \times 4 = 0.5$, so the variance is
$0.5 - 0.09 = 0.41$.

The best method depends on how many samples you average. At $n = 16$, TIS has
the lowest error, because IS's variance dominates. At $n = 1000$, the
variance term is small and IS wins, because its bias is zero. A tight mask
that drops the ratio of 3 has the least variance but the most bias. IcePop's
band of $[0.5, 5]$ would keep that ratio, and here it matches IS exactly.

> [!KEY]
> Truncation and masking buy lower variance with bias. TIS's bias is exactly
> $-\mathbb{E}_\mu[(\rho - C)^+ f]$, the part of each weight it cuts off.

### Methods in use

**TIS.** Yao and colleagues (2025) proposed token-level truncated importance
sampling for the engine gap. In the notation of the verl documentation, with
PPO's ratio $u_t = \pi_\theta(a_t \mid s_t) / \pi_{\text{old}}(a_t \mid s_t)$:

$$
w_t = \min\Big(\frac{\pi_{\text{old}}(a_t \mid s_t)}{\mu(a_t \mid s_t)},\, C\Big), \qquad \mathcal{L}(\theta) = -\mathbb{E}_{\mu}\Big[w_t \min\big(u_t \hat A_t,\; \operatorname{clip}(u_t, 1 - \epsilon, 1 + \epsilon)\, \hat A_t\big)\Big]
$$

Read it as: take PPO's clipped objective, and multiply each token's term by
its trainer-to-engine ratio, capped at $C$. The weight is a constant: no
gradient flows through it. $C$ is a hyperparameter. The verl documentation
suggests values from 1.5 to 5 at the token level.

**IcePop.** The Ling Team's Ring-1T report (2025) masks instead of
truncating. For $k = \pi_{\text{train}} / \pi_{\text{infer}}$, a token keeps
weight $k$ when $0.5 \le k \le 5$ and gets weight 0 otherwise. The report says
this masks about 1 to 2 tokens in a thousand, and it targets the mismatch in
mixture-of-experts models.

**Sequence-level corrections.** Liu, Li, and colleagues (2025) argue that
token-level corrections are biased and can still collapse, and they recommend
sequence-level truncation (Seq-TIS) or masking (Seq-MIS) instead. The next
section shows both halves of that argument on a toy.

Here's a worked example with these rules. A token
has $\mu = 0.02$ in the engine and $\pi_{\text{old}} = 0.08$ in the trainer,
so its ratio is 4. TIS with $C = 2$ gives it weight 2, and IcePop keeps it at
weight 4. If instead $\mu = 0.01$, the ratio is 8. TIS still gives weight 2,
and IcePop gives weight 0.

```python
ratio = torch.exp(logp_train - logp_infer)
tis = torch.clamp(ratio, max=cap)
icepop = torch.where((ratio >= 0.5) & (ratio <= 5.0), ratio, torch.zeros_like(ratio))
```

## Token-level ratios versus sequence-level ratios

A response is a sequence, so the exact correction is the sequence's ratio:

$$
\rho(y) = \prod_{t=1}^{T} \frac{\pi(y_t \mid s_t)}{\mu(y_t \mid s_t)}, \qquad \mathbb{E}_\mu\big[\rho(y)^2\big] = \big(\mathbb{E}_\mu[\rho_t^2]\big)^T \text{ for i.i.d. tokens}
$$

Read it as: multiply every token's ratio together. Weighting the REINFORCE
gradient by this product gives exactly the on-policy gradient. But its second
moment is a product too. When tokens are independent, each with the same
per-token second moment $\mathbb{E}_\mu[\rho_t^2] = 1 + \chi^2(\pi \,\|\, \mu) \ge 1$,
the second moment grows geometrically in the length. A tiny per-token value
of 1.001 over 10,000 tokens gives $1.001^{10000} \approx 22{,}000$. Sum log-ratios
rather than multiplying ratios, so the product doesn't overflow:

```python
seq_ratio = torch.exp(((logp_target - logp_behavior) * mask).sum(dim=1))
```

Token-level weights avoid the explosion by weighting token $t$ by its own
ratio $\rho_t$ alone. That ignores how the earlier tokens were distributed:
the prefix came from $\mu$, not $\pi$. So a token-level weight is biased even
with no truncation at all. The verl documentation puts this bias at
$O(T^2 \Delta_{\max})$ for sequences of length $T$ and a per-token divergence
of at most $\Delta_{\max}$.

The exercise checks both claims on a bigram toy with 4 tokens and length 3,
where all 64 sequences can be enumerated. The reward counts the 3s in a
sequence, plus 1 if its first and last tokens match. With samples from a
perturbed policy $\mu$, the expected gradient with the full sequence ratio
equals the true gradient to $10^{-10}$. With token-level ratios, it's off by
28%. As a length-invariant alternative, the verl documentation also offers the
geometric mean of the ratios, $\rho^{1/T}$, as a filter.

### The loss

The loss that produces these gradients multiplies each sampled token's
log-probability by a constant weight and the sequence's advantage:

$$
\mathcal{L}(\theta) = -\frac{1}{B} \sum_{i=1}^{B} \sum_{t=1}^{T} m_{i,t}\, w_{i,t}\, \hat A_i \log \pi_\theta(y_{i,t} \mid s_{i,t})
$$

Read it as: for each sequence, sum over its response tokens the weight times
the advantage times the log-probability. Average over sequences, and flip the
sign. This is a sum over tokens and a mean over sequences, the REINFORCE
aggregation. With token level, $w_{i,t} = \min(\rho_{i,t}, C)$. With sequence
level, $w_{i,t} = \min(\rho_i, C)$ for every token of sequence $i$. The weights
come from detached log-probabilities.

```python
log_rho = (logp.detach() - logp_behavior) * mask
w = torch.clamp(torch.exp(log_rho), max=cap)                  # token level
loss = -(mask * w * adv[:, None] * logp).sum(dim=1).mean()
```

### Staleness bends the gradient

Next, make $\mu$ an older copy of the policy. The exercise takes eight exact
gradient-ascent steps on the toy, then estimates the gradient at the last
policy from samples of an earlier one, with no correction. The cosine
similarity between the stale estimate and the true gradient is 1 with no lag,
0.88 at a lag of 2, 0.13 at a lag of 4, and $-0.67$ at a lag of 8. At that
point, the uncorrected update points partly backwards. Sequence-level IS from
the policy eight steps old still recovers the true gradient exactly. That
exactness holds only in expectation. With finite samples, its variance is the
price.

> [!KEY]
> Sequence-level IS is unbiased, and its variance grows geometrically with
> length. Token-level IS keeps variance in check, and it's biased because it
> ignores the prefix.

## Shrink the mismatch at the source

Qi and colleagues (2025) attacked the engine gap from the other side. BF16,
the usual training format, has a wide range but only 7 mantissa bits. FP16
has 10. Its rounding errors are 8 times smaller, at the price of a narrower
range that needs loss scaling. They report that FP16 shrinks the mismatch
enough that the plain, unbiased policy gradient with sequence-level IS becomes
stable, and it outperformed the algorithmic corrections run in BF16.

The exercise's stand-in engine rounds the trainer's logits to a lower
precision before the softmax. Real engines differ in more ways, such as kernel
choice and reduction order, but the rounding shows the scale. With 2,000 rows
of 16 logits, rounding to BF16 gives a mean KL from engine to trainer of
$1.2 \times 10^{-5}$ per token. FP16 gives $1.9 \times 10^{-7}$, 64 times
smaller. That factor is $8^2$: a KL grows with the square of a small
perturbation.

To watch the gap during training, log the mean $k_3$ per sequence, with
$\rho_t = \pi_{\text{train}} / \mu_{\text{infer}}$ at the sampled tokens:

$$
\hat{k}_3 = \frac{1}{T} \sum_{t=1}^{T} \big(\rho_t - 1 - \log \rho_t\big)
$$

Read it as: average $\rho - 1 - \log \rho$ over the tokens. Each term is zero
when the engines agree and positive otherwise. Under the engine's samples,
$\mathbb{E}_\mu[\rho_t] = 1$, so the average is an unbiased estimate of
$\mathrm{KL}(\mu \,\|\, \pi)$ per token.

> [!RECAP]
> - Bias is $\mathbb{E}_\mu[w f] - \mathbb{E}_\pi[f]$ and variance is
>   $\mathbb{E}_\mu[w^2 f^2] - (\mathbb{E}_\mu[w f])^2$, both sums weighted by
>   $\mu$.
> - TIS is $\min(\rho, C)$, and MIS keeps $\rho$ inside a band and zeroes it
>   outside.
> - The sequence ratio is $\exp$ of the masked sum of log-ratios.
> - $k_3 = \rho - 1 - \log \rho$ averages to $\mathrm{KL}(\mu \,\|\, \pi)$.
> - The corrected loss detaches the weights, sums over tokens, and averages
>   over sequences.

## Build it step by step

The checks test the functions in this order.

1. **`tis_weights(ratio, cap)`** and **`mis_weights(ratio, low, high)`.**
   Elementwise, any shape. Use `torch.clamp` and `torch.where`. Common
   mistake: clamping from below too, which changes the weights of tokens the
   engine overrates.
2. **`estimator_bias_variance(mu, pi, f, weights)`.** Four `(M,)` tensors,
   returns two floats. Every expectation is weighted by `mu`, because that's
   where samples come from. Common mistake: computing the variance with `pi`
   as the weights.
3. **`sequence_ratio(logp_target, logp_behavior, mask)`.** `(B, T)` in,
   `(B,)` out. Sum masked log-ratios, then exponentiate. Common mistake:
   forgetting the mask, so padding tokens count.
4. **`k3_mismatch(logp_train, logp_infer, mask)`.** `(B, T)` in, `(B,)` out,
   a masked mean per sequence. Common mistake: using
   $\rho = \mu / \pi$, which estimates the reverse KL.
5. **`is_pg_loss(logp, logp_behavior, mask, advantages, level, cap)`.**
   Compute ratios from `logp.detach()`. For sequence level, sum the masked
   log-ratios with `keepdim=True`, exponentiate, clamp, and expand to
   `(B, T)`. Common mistake: letting gradient flow through the weights, which
   adds a second term to the gradient and breaks the unbiasedness check.
6. **`engine_logprobs(logits, dtype)`.** Round with `logits.to(dtype)`,
   convert back to `torch.float64`, and take `log_softmax`. With
   `torch.float64`, the result must equal the trainer's log-probabilities
   exactly.

## Cost and scaling

The corrections themselves are cheap. The cost is in producing the numbers
they need. The following table summarizes the cost of each piece:

| Piece | Extra compute | Extra memory |
|---|---|---|
| Behaviour log-probs $\mu$ | None, the engine returns them while sampling | One float per token |
| Proximal log-probs $\pi_{\text{old}}$ | One trainer forward pass over the batch | One float per token |
| TIS, MIS, IcePop | Elementwise, $O(BT)$ | None |
| Sequence ratios | A sum over $T$ per sequence | None |
| FP16 training | Loss scaling, a few lines of configuration | None |

Asynchronous training is where the savings are. Generation and training
overlap, so GPUs don't sit idle waiting for the slowest response. AReaL's
2.77 times speedup came from exactly that. The price is more staleness, and
that makes the corrections in this lesson necessary. The variance of
sequence-level weights grows exponentially with length, which is why
long-reasoning runs use token-level or geometric-mean filters despite their
bias. The exercise's checks enumerate 64 sequences and cost a fraction of a
second.

## Where this breaks

**Recomputed behaviour probabilities.** If the trainer recomputes $\mu$, the
ratio is 1 and the correction silently does nothing. Log the engine's own
log-probabilities during sampling.

**Different action spaces.** An engine that samples with top-$p$ or top-$k$
never proposes tokens outside that set, but the trainer's softmax covers the
whole vocabulary. DeepSeek-V3.2 reuses the engine's truncation mask in
training, so both policies share an action space.

**Mixture-of-experts routing.** A small numerical difference can send a token
to different experts in the two engines, and the ratio can then be far from
1. Routing replay records the experts chosen during sampling and reuses them
in training.

**Truncation bias adds up.** Each truncated token loses a little weight, and
long responses have many tokens. A run can look stable and still drift toward
the engine's policy instead of the trainer's. Monitoring the $k_3$ gap and the
fraction of truncated or masked tokens catches it.

**FP16 has a narrow range.** Its largest value is 65,504, so activations and
gradients can overflow without loss scaling. That's why BF16 became the
default, and why switching back takes care.

## Further reading

- [Yao and colleagues (2025), Your Efficient RL Framework Secretly Brings You
  Off-Policy RL Training](https://fengyao.notion.site/off-policy-rl). The blog
  post that measured the engine gap and proposed TIS.
- [Liu, Li, and colleagues (2025), When Speed Kills Stability: Demystifying RL
  Collapse from the Training-Inference Mismatch](https://richardli.xyz/rl-collapse).
  Why token-level corrections are biased, and sequence-level TIS and MIS.
- [verl documentation, Mathematical Formulations of Rollout Correction
  Methods](https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html).
  The three-policy notation, token and sequence levels, and geometric-mean
  filtering.
- [Ling Team (2025), Every Step Evolves: Scaling Reinforcement Learning for
  Trillion-Scale Thinking Model](https://arxiv.org/abs/2510.18855). The
  Ring-1T report, which introduces IcePop's masked ratio.
- [Qi and colleagues (2025), Defeating the Training-Inference Mismatch via
  FP16](https://arxiv.org/abs/2510.26788). Switching precision instead of
  correcting the estimator.
- [Fu and colleagues (2025), AReaL](https://arxiv.org/abs/2505.24298). A fully
  asynchronous RL system with bounded staleness and a decoupled PPO
  objective.
- [Noukhovitch and colleagues (2024), Asynchronous RLHF](https://arxiv.org/abs/2410.18252).
  How much off-policy data RLHF tolerates, and which algorithms tolerate it
  best.
- [Hilton, Cobbe, and Schulman (2021), Batch Size-Invariance for Policy
  Optimization](https://arxiv.org/abs/2110.00641). The decoupled objective
  that separates the behaviour policy from the proximal policy.

Researchers are still working on these open questions:

- Is there a correction with bounded variance whose bias doesn't grow with
  the length of the response?
- Should an RL algorithm be designed for a lagged inference policy from the
  start, instead of correcting back toward on-policy training?
- How much of the engine gap can deterministic, batch-invariant kernels
  remove, and at what cost in throughput?

## The exercise

You build the corrections and measure them. Write `tis_weights`,
`mis_weights`, `estimator_bias_variance`, `sequence_ratio`, `k3_mismatch`,
`is_pg_loss`, and `engine_logprobs`.

The checks confirm the lesson's weights of 2, 2, 4, and 0, and every number in
the bias and variance table. On random distributions, they confirm that IS has
zero bias and that TIS's bias is $-\mathbb{E}_\mu[(\rho - C)^+ f]$. They
confirm that TIS has the lowest error at $n = 16$ and IS at $n = 1000$, that
the sequence ratio averages 1 with a second moment of $(\mathbb{E}_\mu[\rho^2])^T$,
and that $k_3$ averages to the KL. By enumerating all 64 toy sequences, they
check that sequence-level IS recovers the on-policy gradient while
token-level IS is off by more than 5%, and that the uncorrected stale gradient
turns away as the lag grows. The last check confirms that FP16 rounding
mismatches at least 20 times less than BF16.
