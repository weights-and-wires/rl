---
title: Group-relative policy optimisation
slug: 37-group-relative-policy-optimization
part: "Part 8 — Reinforcement learning for language models"
summary: Replace PPO's critic with a group of samples, then fix GRPO's length and difficulty biases with Dr. GRPO, DAPO, and GSPO.
minutes: 70
gpu: true
objectives:
  - Compute RLOO and GRPO advantages from a group of rewards, and show by exact enumeration which baselines are unbiased.
  - Implement GRPO's clipped, KL-penalised loss with the per-sequence, per-token, and constant aggregations.
  - Explain and reproduce GRPO's length bias, and show that Dr. GRPO's constant normaliser removes it.
  - "Implement DAPO's clip-higher with asymmetric epsilon, and GSPO's length-normalised sequence ratio."
  - Run GRPO on Qwen3-0.6B with and without the Dr. GRPO fixes, and read the response lengths.
prereqs:
  - 16-actor-critic
  - 19-proximal-policy-optimization
  - 35-rlhf-with-ppo
exercise: 37-group-relative-policy-optimization
---

# Group-relative policy optimisation

PPO for language models carries a critic: a second network, as large as the
policy, that guesses how much reward a half-written response earns in the end. It
doubles the memory. It is also hard to train, because the reward arrives only
once, at the end of the response.

There is a cheaper way to tell a good response from a bad one. Ask the model
the same question several times, score every answer, and compare each answer
with its siblings. An answer that beats the others' average was better than
usual. That comparison replaces the critic. REINFORCE leave-one-out (RLOO)
does it one way, and group-relative policy optimisation (GRPO), from
DeepSeekMath (2024), does it another. GRPO trained DeepSeek-R1 and many of the
open reasoning models that followed.

GRPO's details turned out to matter. How you average the loss over tokens
changes which lengths the model learns to prefer. How you scale the
advantages changes which questions count most. This lesson builds GRPO, then
three follow-ups that fix a piece each: Dr. GRPO, DAPO, and GSPO. It ends with
a real run on Qwen3-0.6B.

## What you already know

- REINFORCE estimates the policy gradient as the average of
  $(G_t - b)\, \nabla_\theta \log \pi_\theta(a_t \mid s_t)$, with $G_t$ the
  return. A baseline $b$ that doesn't depend on the action keeps it unbiased
  and lowers its variance. In this lesson a bare $G$ is the group size.
- PPO reuses a batch for several steps and clips the probability ratio
  $\rho = \pi_\theta / \pi_{\text{old}}$ to stay near the policy that sampled.
- A response's log-probability is the sum of its per-token log-probabilities.
  Per-token log-probabilities and masks have shape `(B, T)`.
- The KL penalty keeps the policy near the reference $\pi_{\text{ref}}$. The
  $k_3$ estimator from *RLHF with PPO* is $\frac{\pi_{\text{ref}}}{\pi} - \log \frac{\pi_{\text{ref}}}{\pi} - 1$ per token.
- An *advantage* says how much better an action was than the policy's usual
  choice. Its sign decides whether a log-probability goes up or down.

## Replace the critic with a group

In this lesson a whole response is one action, and its reward $r(x, y)$ arrives
at the end. For each prompt $x$, sample a *group* of $G$ responses
$y_1, \dots, y_G$ and score them all.

### RLOO: leave yourself out

REINFORCE leave-one-out (RLOO), which Ahmadian and colleagues (2024) brought
to language models, uses the other $G - 1$ rewards as the baseline:

$$
\hat A_i = r_i - \frac{1}{G - 1}\sum_{j \ne i} r_j = \frac{G}{G - 1}\Big(r_i - \frac{1}{G}\sum_{j=1}^{G} r_j\Big)
$$

Read it as: each response's reward minus the average of its siblings. The
second form is the same number, computed from the group mean. The baseline
never looks at $y_i$, because the responses are independent. So the gradient
estimate stays unbiased. In PyTorch, with `r = rewards.view(-1, G)`,
`G / (G - 1) * (r - r.mean(dim=1, keepdim=True))`.

### GRPO: standardise within the group

GRPO subtracts the group mean, which includes $r_i$ itself, and divides by the
group's standard deviation:

$$
\hat A_i = \frac{r_i - \operatorname{mean}(r_1, \dots, r_G)}{\operatorname{std}(r_1, \dots, r_G)}
$$

Read it as: a z-score within the group. Every token of response $i$ gets this
same advantage. In PyTorch, `(r - r.mean(1, keepdim=True)) / (r.std(1, unbiased=False, keepdim=True) + eps)`.
The `eps` matters. When all $G$ rewards tie, the numerator and the standard
deviation are both zero, and the group must contribute no policy gradient.

### Worked example

A group of $G = 8$ with rewards $(1, 0, 0, 1, 1, 0, 0, 0)$:

- The mean is $0.375$. The population standard deviation is
  $\sqrt{0.375 \times 0.625} = 0.4841$.
- A correct answer gets $(1 - 0.375) / 0.4841 = +1.291$. A wrong one gets
  $-0.375 / 0.4841 = -0.775$.
- With the Bessel-corrected standard deviation, PyTorch's default, $0.5175$,
  the values are $+1.208$ and $-0.725$. Implementations differ here. The
  exercise uses the population form.
- RLOO on the group $(1, 0, 0, 1)$ with $G = 4$ gives
  $1 - 1/3 = +0.667$ and $0 - 2/3 = -0.667$.

### Which baselines are unbiased

With a tiny policy you can check unbiasedness exactly, with no sampling. Take
a softmax over four responses with probabilities $(0.4, 0.3, 0.2, 0.1)$ and
rewards $(1, 0, 0.5, 2)$. The true gradient of the expected reward with respect
to the logits is $\pi_k (r_k - \mathbb{E}_\pi[r]) = (0.12, -0.21, -0.04, 0.13)$.
Enumerate all $4^3 = 64$ groups of three, weight each by its probability, and
average the estimate $\frac{1}{G}\sum_i \hat A_i \nabla \log \pi(y_i)$:

- RLOO's expected estimate equals the true gradient exactly.
- The mean-centred estimate equals exactly $\frac{G - 1}{G} = \frac{2}{3}$ of
  it. Including $r_i$ in its own baseline shrinks the step by a constant, and
  a constant is harmless.
- The standard-deviation-scaled estimate is a different vector, with cosine
  similarity $0.980$ to the truth. Dividing by a quantity that depends on the
  sampled rewards biases the estimate. It still points roughly uphill.

> [!KEY]
> RLOO is unbiased, and GRPO's mean baseline only rescales it by
> $(G - 1)/G$. GRPO's division by the standard deviation is what makes it a
> biased estimator.

## Build the GRPO loss

GRPO keeps PPO's clipped ratio and adds a KL penalty to the loss. The
DeepSeekMath paper writes the objective, to be maximised, as

$$
\mathcal{J}_{\text{GRPO}} = \mathbb{E}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{\lvert o_i \rvert}\sum_{t=1}^{\lvert o_i \rvert}\Big\{\min\big[\rho_{i,t}\hat A_{i},\ \operatorname{clip}(\rho_{i,t}, 1 - \varepsilon, 1 + \varepsilon)\,\hat A_{i}\big] - \beta\, k_3(i, t)\Big\}\Bigg]
$$

Read it piece by piece:

- $o_i$ is the $i$-th response and $\lvert o_i \rvert$ its number of tokens.
- $\rho_{i,t} = \pi_\theta(o_{i,t} \mid x, o_{i,<t}) / \pi_{\text{old}}(o_{i,t} \mid x, o_{i,<t})$
  is a per-token ratio against the policy that sampled.
- The $\min$ with the clipped term stops the gradient once the ratio has moved
  more than $\varepsilon$ in the direction the advantage wants.
- $k_3(i, t)$ is the $k_3$ KL estimate at that token, with weight $\beta$.
  DeepSeekMath used $\beta = 0.04$ and $G = 64$.
- The $\frac{1}{\lvert o_i \rvert}$ averages each response over its own
  tokens. Then $\frac{1}{G}$ averages over responses.

The DeepSeek-R1 paper (2025) writes the same objective with one ratio per
sequence and no $\frac{1}{\lvert o_i \rvert}$. This course follows
DeepSeekMath and the common open implementations, which use per-token ratios.
GSPO, later in this lesson, makes the sequence-level ratio its main idea.

In code, the loss splits into three parts. A per-token surrogate:

```python
ratio = torch.exp(logp_new - logp_old)
adv = advantages[:, None]                       # (B, 1) broadcasts over T
per_token = torch.minimum(ratio * adv, ratio.clamp(1 - eps_low, 1 + eps_high) * adv)
```

Then the KL term, subtracted per token. Then an *aggregation*, which turns the
`(B, T)` array into one number. The original GRPO aggregation is
`((x * mask).sum(1) / mask.sum(1)).mean()`. With one gradient step per batch,
$\rho = 1$ at the moment you differentiate. Each token then gets gradient
$\hat A_i / (G \lvert o_i \rvert)$, the REINFORCE gradient.

## Find GRPO's two biases

Liu and colleagues (2025) pointed at two divisions in GRPO. Their fix is
called Dr. GRPO.

### Length bias

Look at one response's share of the gradient. Its tokens' score terms add up
to the gradient of the whole response's log-probability, so its term is

$$
\frac{\hat A_i}{\lvert o_i \rvert}\sum_{t} \nabla_\theta \log \pi_\theta(o_{i,t} \mid x, o_{i,<t}) = \frac{\hat A_i}{\lvert o_i \rvert}\, \nabla_\theta \log \pi_\theta(o_i \mid x)
$$

Read it as: the effective advantage of a response is $\hat A_i / \lvert o_i \rvert$.
A long correct answer is rewarded less per token than a short one, so correct
answers drift shorter. A long wrong answer is punished less per token, so wrong
answers drift longer. For example, two wrong answers with $\hat A = -1$ and
lengths 10 and 100 get per-token coefficients $-0.1$ and $-0.01$. The long
one's tokens are pushed down ten times more gently.

The exercise makes the bias visible with exact gradients. The toy writes a
correctness token, frozen at probability $0.5$ each way. Then it writes a
length $L$ from 1 to 8 with learnable logits for each correctness value, then
$L - 1$ filler tokens with log-probability zero. Reward is 1 when right, and
both runs use the mean-centred advantage $r - 0.5 = \pm 0.5$. With $\pi(c)$
frozen at one half, GRPO's division by the standard deviation would only
double every step, so leaving it out isolates the length normaliser. Summing
over all 16 outcomes gives the expected gradient, with no sampling noise. The
following table shows mean lengths after 300 steps of gradient ascent with
step size $0.5$, from a uniform start at $4.5$:

| Aggregation | Mean length if wrong | Mean length if right |
|---|---|---|
| GRPO: $\frac{1}{\lvert o_i \rvert}$ per sequence | $5.39$ | $2.36$ |
| Dr. GRPO: a constant | $4.50$ | $4.50$ |

Under GRPO the update on the length logits is proportional to
$\hat A_c\, \nabla \mathbb{E}[1/(L + 1) \mid c]$.
For wrong answers $\hat A_c < 0$, so ascent lowers $\mathbb{E}[1/(L+1)]$ and
lengthens them. With a constant normaliser the same term is
$\hat A_c\, \nabla \mathbb{E}[\text{const}] = 0$, and nothing moves.

Dr. GRPO drops the per-response division. Its objective sums each response's
tokens, $\frac{1}{G}\sum_i \sum_t$, and its released code divides by a fixed
`MAX_TOKENS`, the generation budget, to keep the scale steady. The Dr. GRPO
paper also sets $\beta = 0$.

> [!KEY]
> Averaging each response over its own length gives it an effective advantage
> of $\hat A_i / \lvert o_i \rvert$. That rewards short right answers and
> spares long wrong ones. A constant normaliser removes the effect.

### Difficulty bias

Dividing by the group's standard deviation changes how much each question
weighs. With binary rewards and success rate $p$, the standard deviation is
$\sqrt{p(1 - p)}$, so the total advantage mass $\sum_i \lvert \hat A_i \rvert$
is multiplied by $1 / \sqrt{p(1 - p)}$. That factor is $2.0$ at $p = 0.5$ and
$3.33$ at $p = 0.1$ or $p = 0.9$. A question the group gets 10% or 90% right
weighs $1.67$ times as much as a coin-flip question, and the factor keeps
growing toward the extremes. Dr. GRPO uses the plain
mean-centred advantage $r_i - \operatorname{mean}(r)$.

## Clip higher, and count every token: DAPO

DAPO, from Yu and colleagues (2025), is an open recipe for long chains of
thought. It changes four things. Two of them are in the loss.

**Clip-higher.** A symmetric clip at $1 + 0.2$ caps how far one update can
raise a rare token. A token at probability $0.01$ can reach only $0.012$. An
exploratory token that turned out well barely moves, and the policy's entropy
collapses. DAPO decouples the two bounds: $\varepsilon_{\text{low}} = 0.2$ and
$\varepsilon_{\text{high}} = 0.28$. The rare token can then rise to $0.0128$.
The lower bound stays, so tokens being pushed down still stop at
$\rho = 0.8$.

**Token-level aggregation.** DAPO divides the sum over every token in the
group by the total number of tokens:

$$
\mathcal{J}_{\text{DAPO}} = \mathbb{E}\Bigg[\frac{1}{\sum_{i=1}^{G}\lvert o_i \rvert}\sum_{i=1}^{G}\sum_{t=1}^{\lvert o_i \rvert}\min\big(\rho_{i,t}\hat A_{i},\ \operatorname{clip}(\rho_{i,t}, 1 - \varepsilon_{\text{low}}, 1 + \varepsilon_{\text{high}})\,\hat A_{i}\big)\Bigg]
$$

Read it as: every token in the batch counts equally, whichever response it
came from. In code, `(x * mask).sum() / mask.sum()`. Take a batch with one
10-token response and one 1000-token response. Per-sequence averaging gives
each response half the weight. Token averaging gives the long one about 99%
of it. That is the point. Repetition and gibberish in long responses get
penalised in full. When all responses have equal length, the two
aggregations agree.

DAPO also drops the KL term and keeps the group-normalised advantage. Its two
other changes act on the data: *dynamic sampling* drops groups whose rewards
all tie and refills the batch, and *overlong shaping* penalises responses
that run past a length budget.

> [!WARNING]
> The clip is one-sided, because of the $\min$. For $\hat A < 0$ and
> $\rho = 1.5$, $\min(-1.5, -1.28) = -1.5$ keeps the unclipped value and its
> full gradient. Clipping never protects a token that has moved the wrong
> way.

## One ratio per sequence: GSPO

The reward belongs to the whole response, but GRPO's ratio belongs to each
token. Token ratios are noisy, and a long response sums a lot of noise. GSPO,
from Zheng and colleagues (2025), uses one length-normalised ratio per
sequence:

$$
s_i(\theta) = \Big(\frac{\pi_\theta(o_i \mid x)}{\pi_{\text{old}}(o_i \mid x)}\Big)^{1/\lvert o_i \rvert} = \exp\Big(\frac{1}{\lvert o_i \rvert}\sum_{t=1}^{\lvert o_i \rvert}\log \frac{\pi_\theta(o_{i,t} \mid x, o_{i,<t})}{\pi_{\text{old}}(o_{i,t} \mid x, o_{i,<t})}\Big)
$$

Read it as: the geometric mean of the token ratios. GSPO clips $s_i$ and
averages over the group:

$$
\mathcal{J}_{\text{GSPO}} = \mathbb{E}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\min\big(s_i \hat A_i,\ \operatorname{clip}(s_i, 1 - \varepsilon, 1 + \varepsilon)\,\hat A_i\big)\Bigg]
$$

Unclipped, the gradient of one response's term is
$s_i \hat A_i \cdot \frac{1}{\lvert o_i \rvert}\sum_t \nabla \log \pi_\theta(o_{i,t} \mid \cdot)$.
Every token of a response shares one weight. Because $s_i$ is an average, it
sits very close to 1, and the paper's clip range is tiny: $3 \times 10^{-4}$
below and $4 \times 10^{-4}$ above. For example, token log-ratios
$(0.01, -0.02, 0.04)$ average $0.01$, so $s = e^{0.01} = 1.01005$. That is past
$1 + 4 \times 10^{-4}$. With $\hat A > 0$ the whole response is clipped. The
unnormalised product of ratios would be $e^{0.03} = 1.0305$.

## Run GRPO on Qwen3-0.6B

The **Run** block trains Qwen3-0.6B with your functions when a GPU is
available. Each step samples 8 two-digit multiplication questions, 8
responses each, with at most 96 new tokens. The reward is a Python rule: 1 when
the last `Answer: N` in the response equals the product, and 0 otherwise.
The block makes one gradient step per batch with AdamW at learning rate
$10^{-6}$, with $\beta = 0.04$ and the $k_3$ KL. It runs twice, once for each
mode:

- `mode="grpo"`: advantages scaled by the standard deviation, and
  `"sequence_mean"` aggregation.
- `mode="dr_grpo"`: mean-centred advantages, and `"constant"` aggregation
  with `max_tokens=96`.

Every few steps it prints the mean reward, the mean length of right and wrong
responses, and the KL. Twenty steps is a short run with noisy batches.
Compare the trend in the two length columns between modes, not single rows.

> [!TIP]
> The run keeps float32 master weights and runs its forward passes in
> bfloat16. A weight stored in bfloat16 has between two and three significant
> digits, so an update of $10^{-6}$ to a weight of $0.02$ rounds to no change
> at all.

> [!RECAP]
> - RLOO subtracts the siblings' mean and is unbiased. GRPO's mean-centring
>   scales it by $(G - 1)/G$, and its standard-deviation division biases it.
> - The GRPO loss is minus the aggregate of the clipped surrogate minus
>   $\beta k_3$.
> - Per-sequence averaging gives each response an effective advantage of
>   $\hat A_i / \lvert o_i \rvert$, which lengthens wrong answers. A constant
>   normaliser removes that.
> - DAPO raises only the upper clip and averages over all tokens. GSPO clips
>   one geometric-mean ratio per response.

## Build it step by step

The checks test the functions in this order.

1. **`rloo_advantages(rewards, group_size)`.** Takes `(B,)`, returns `(B,)`.
   Reshape with `view(-1, group_size)` and apply the $G/(G-1)$ form. Common
   mistake: reshaping as `(G, P)`, which mixes prompts.
2. **`grpo_advantages(rewards, group_size, scale_by_std, eps)`.** Use
   `std(dim=1, unbiased=False)`. Common mistake: the Bessel default, which
   misses the hand values by about 6%.
3. **The exact-expectation check** uses only your two advantage functions.
4. **`k3_kl(logp, ref_logp)`.** Elementwise, with `log_ratio = ref_logp - logp`.
   Common mistake: the ratio upside down, which estimates the other KL.
5. **`clipped_surrogate(logp_new, logp_old, advantages, eps_low, eps_high)`.**
   Returns `(B, T)`. Broadcast with `advantages[:, None]` and take
   `torch.minimum`. Common mistake: clipping the product instead of the ratio.
6. **`aggregate(per_token, mask, mode, max_tokens)`.** Zero the padding with
   `torch.where` first, because padded log-probabilities can be anything.
7. **`grpo_loss(...)`.** Subtract `beta * k3_kl` from the surrogate, aggregate,
   and negate. Common mistake: forgetting the sign, so training runs downhill.
8. **`gspo_ratio` and `gspo_loss`.** The ratio is `exp(masked sum / mask sum)`
   of the log-ratios, and the loss is `-mean(min(s * A, clamp(s) * A))`.
9. **`length_bias_toy(mode, steps, lr)`.** Build the `(16, 9)` batch from
   `torch.log_softmax(logits, dim=1)` on each step. Weight row $o$ by
   $16\, \pi(o)\, \hat A_o$ with $\pi(o)$ detached, call your `aggregate`, and
   step with `torch.autograd.grad`. Common mistake: letting gradient flow
   through $\pi(o)$, which adds a term the policy gradient doesn't have.

## Cost and scaling

Group methods trade the critic's memory for more samples per prompt. The
following table compares a step of each method on $P$ prompts:

| Method | Networks in memory | Responses sampled | Extra cost |
|---|---|---|---|
| PPO | Policy, critic, reference, reward model | $P$ | Critic forward and backward |
| RLOO, GRPO, Dr. GRPO | Policy, reference | $P \times G$ | None beyond sampling |
| DAPO | Policy only | Over $P \times G$, to refill tied groups | Discarded groups |
| GSPO | Policy, plus a reference if you add a KL term | $P \times G$ | None |

Sampling dominates. Generating $G$ responses costs about $G$ times one
response, though batching hides some of it on a GPU. The loss itself is a
few elementwise operations over `(B, T)`, linear in the number of tokens. The
real run's peak memory comes from the logits, of shape `(B, T, V)` with
$V \approx 152{,}000$ for Qwen3. It computes them in micro-batches of eight
responses and accumulates gradients.

## Where this breaks

**Tied groups.** When every response in a group is right, or every one is
wrong, all advantages are zero and the group teaches nothing. Easy and
impossible prompts waste compute. DAPO's dynamic sampling refills the batch.
Curriculum filtering by pass rate does the same job before training.

**Entropy collapse.** Symmetric clipping and positive updates on likely
tokens make the policy deterministic early. Exploration stops, and so does
improvement. Clip-higher is one remedy. *Reasoning with verifiable rewards*
measures the collapse.

**Length drift.** Per-sequence averaging lengthens wrong answers, as the toy
shows. Growing length alone doesn't prove the model reasons more. Check the
length of right and wrong answers separately.

**Off-policy noise.** With several updates per batch, or with an inference
engine whose probabilities differ from the trainer's, token ratios drift from
1. GSPO's averaged ratio is steadier, and *Infrastructure for LLM
reinforcement learning* covers the corrections.

**Changing the scale.** Dropping the standard deviation shrinks the advantages
for easy and hard questions. Adam absorbs much of an overall rescaling, but
not a change in relative weights between questions.

## Further reading

The following papers define the methods in this lesson:

- [Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (2024)](https://arxiv.org/abs/2402.03300): introduces GRPO with per-token ratios and the $k_3$ penalty.
- [Ahmadian et al., "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs" (2024)](https://arxiv.org/abs/2402.14740): RLOO for language models, and why PPO's machinery is often unnecessary.
- [Liu et al., "Understanding R1-Zero-Like Training: A Critical Perspective" (2025)](https://arxiv.org/abs/2503.20783): the length and difficulty biases, and Dr. GRPO.
- [Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale" (2025)](https://arxiv.org/abs/2503.14476): clip-higher, dynamic sampling, token-level loss, and overlong shaping.
- [Zheng et al., "Group Sequence Policy Optimization" (2025)](https://arxiv.org/abs/2507.18071): the length-normalised sequence ratio.
- [DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025)](https://arxiv.org/abs/2501.12948): GRPO at scale, written with sequence-level ratios.

Open questions that researchers are working on:

- Which aggregation gives the lowest-variance unbiased gradient when response lengths vary a lot?
- How large must a group be before rare correct responses show up often enough to learn from?
- When does a learned critic beat a group baseline for very long responses?

## The exercise

You build the group baselines and every loss in this lesson. Write
`rloo_advantages` and `grpo_advantages`, then `k3_kl`, `clipped_surrogate`,
`aggregate`, and `grpo_loss`. Then write `gspo_ratio` and `gspo_loss`, and
`length_bias_toy`, which trains the two-step length toy with exact expected
gradients through your `aggregate`.

The checks verify both advantage functions against the hand-computed groups.
They enumerate every group of three to confirm that RLOO is exactly unbiased
and mean-centring scales it by $(G-1)/G$, and they confirm the difficulty
factor $1/\sqrt{p(1-p)}$. They check $k_3$ against the exact KL, the clip
pattern for clip-higher, all three aggregations on unequal lengths, and the
on-policy GRPO gradient in closed form. They check GSPO's ratio and its shared
per-token weight. The last check reproduces the table: 5.39 and 2.36 under
GRPO, and 4.50 for both under Dr. GRPO.
