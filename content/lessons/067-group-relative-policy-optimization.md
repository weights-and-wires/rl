---
title: Group-relative policy optimization
slug: 067-group-relative-policy-optimization
part: "Part 9 — RL for language models"
summary: Replace PPO's critic with a group of sampled answers, then see how GRPO's averaging biases lengths and what Dr. GRPO, DAPO, and GSPO change.
minutes: 40
gpu: true
objectives:
  - Compute RLOO and GRPO advantages from a group of rewards by hand.
  - Show by exact enumeration which group baselines are unbiased.
  - Implement the clipped GRPO loss with per-sequence, per-token, and constant aggregation, and GSPO's length-normalized sequence ratio.
  - Explain why per-sequence averaging lengthens wrong answers, and why a constant normalizer removes the effect.
  - Run GRPO on Qwen3-0.6B with and without the Dr. GRPO fixes, and read the response lengths.
prereqs:
  - 033-baselines-and-actor-critic
  - 036-proximal-policy-optimization
  - 065-rlhf-with-ppo
exercise: 067-group-relative-policy-optimization
---

# Group-relative policy optimization

PPO for language models carries a critic: a second network, as large as the policy, that guesses how much reward a half-written response will earn. It doubles the memory, and it is hard to train, because the reward arrives only once, at the end of the response.

The critic tells the policy whether a response was better or worse than usual, so that good responses are pushed up and bad ones pushed down. You can get that comparison without a network. Ask the model the same question several times, score every answer, and compare each answer with its siblings. This chapter builds that idea, called *group-relative policy optimization* (GRPO, from DeepSeekMath, 2024), and then shows two places where its averaging quietly misleads the model.

## What you already know

- REINFORCE estimates the policy gradient as the average of $(G_t - b)\,\nabla_\theta \log \pi_\theta(a_t \mid s_t)$. A baseline $b$ that does not depend on the action keeps it unbiased and lowers its variance, from [Baselines and actor-critic](/lessons/033-baselines-and-actor-critic). In this chapter a bare $G$ is the group size.
- PPO reuses a batch and clips the ratio $\rho = \pi_\theta / \pi_{\text{old}}$ to stay near the policy that sampled, from [Proximal policy optimization](/lessons/036-proximal-policy-optimization).
- A response's log-probability is the sum of its per-token log-probabilities. Per-token log-probabilities and masks have shape `(B, T)`.
- The $k_3$ estimator of the KL to the reference, $\frac{\pi_{\text{ref}}}{\pi} - \log\frac{\pi_{\text{ref}}}{\pi} - 1$ per token, comes from [RLHF with PPO](/lessons/065-rlhf-with-ppo).
- An *advantage* says how much better an action was than usual. Its sign decides whether a log-probability goes up or down.

## A group is its own baseline

Here a whole response is one action, and its reward $r(x, y)$ arrives at the end. For each prompt $x$, sample a *group* of $G$ responses $y_1, \dots, y_G$ and score them.

The raw reward is a poor advantage. With rewards of 0 or 1, every correct answer would be pushed up by 1 and every wrong one left alone, so the gradient mostly measures how often the model is right, not which answer to prefer. The group's average reward says what "usual" is for this prompt. Subtract it.

Take a group of $G = 8$ with rewards $(1, 0, 0, 1, 1, 0, 0, 0)$.

- The mean is $0.375$, and the population standard deviation is $\sqrt{0.375 \times 0.625} = 0.4841$.
- GRPO divides the centered reward by that standard deviation, a z-score within the group. A correct answer gets $(1 - 0.375)/0.4841 = +1.291$. A wrong one gets $-0.375/0.4841 = -0.775$.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Advantages of eight responses to one prompt: three correct responses get plus 1.291 and five wrong ones get minus 0.775">

<line x1="60" y1="140" x2="590" y2="140" class="stroke-muted"/>
<text x="56" y="144" class="fig-small" text-anchor="end">0</text>
<line x1="60" y1="200" x2="590" y2="200" class="stroke-grid"/><text x="56" y="204" class="fig-small" text-anchor="end">-1</text>
<line x1="60" y1="80" x2="590" y2="80" class="stroke-grid"/><text x="56" y="84" class="fig-small" text-anchor="end">+1</text>
<rect x="82.5" y="62.5" width="40" height="77.5" class="fill-c"/>
<text x="102.5" y="56.5" class="fig-small" text-anchor="middle">+1.291</text>
<text x="102.5" y="262" class="fig-small" text-anchor="middle">y1</text><text x="102.5" y="278" class="fig-small" text-anchor="middle">r=1</text>
<rect x="147.5" y="140.0" width="40" height="46.5" class="fill-bad"/>
<text x="167.5" y="202.5" class="fig-small" text-anchor="middle">-0.775</text>
<text x="167.5" y="262" class="fig-small" text-anchor="middle">y2</text><text x="167.5" y="278" class="fig-small" text-anchor="middle">r=0</text>
<rect x="212.5" y="140.0" width="40" height="46.5" class="fill-bad"/>
<text x="232.5" y="202.5" class="fig-small" text-anchor="middle">-0.775</text>
<text x="232.5" y="262" class="fig-small" text-anchor="middle">y3</text><text x="232.5" y="278" class="fig-small" text-anchor="middle">r=0</text>
<rect x="277.5" y="62.5" width="40" height="77.5" class="fill-c"/>
<text x="297.5" y="56.5" class="fig-small" text-anchor="middle">+1.291</text>
<text x="297.5" y="262" class="fig-small" text-anchor="middle">y4</text><text x="297.5" y="278" class="fig-small" text-anchor="middle">r=1</text>
<rect x="342.5" y="62.5" width="40" height="77.5" class="fill-c"/>
<text x="362.5" y="56.5" class="fig-small" text-anchor="middle">+1.291</text>
<text x="362.5" y="262" class="fig-small" text-anchor="middle">y5</text><text x="362.5" y="278" class="fig-small" text-anchor="middle">r=1</text>
<rect x="407.5" y="140.0" width="40" height="46.5" class="fill-bad"/>
<text x="427.5" y="202.5" class="fig-small" text-anchor="middle">-0.775</text>
<text x="427.5" y="262" class="fig-small" text-anchor="middle">y6</text><text x="427.5" y="278" class="fig-small" text-anchor="middle">r=0</text>
<rect x="472.5" y="140.0" width="40" height="46.5" class="fill-bad"/>
<text x="492.5" y="202.5" class="fig-small" text-anchor="middle">-0.775</text>
<text x="492.5" y="262" class="fig-small" text-anchor="middle">y7</text><text x="492.5" y="278" class="fig-small" text-anchor="middle">r=0</text>
<rect x="537.5" y="140.0" width="40" height="46.5" class="fill-bad"/>
<text x="557.5" y="202.5" class="fig-small" text-anchor="middle">-0.775</text>
<text x="557.5" y="262" class="fig-small" text-anchor="middle">y8</text><text x="557.5" y="278" class="fig-small" text-anchor="middle">r=0</text>
<text x="300" y="22" text-anchor="middle" class="fig-bold">Group of 8: mean 0.375, std 0.484</text>
</svg>
Figure 1. Three correct answers are pushed up and five wrong ones pushed down. Notice that the correct answers get the larger advantage because they are rarer, and that the advantages sum to zero.
```

Every token of response $i$ receives that response's advantage. In PyTorch, with `r = rewards.view(-1, G)`:

```python
adv = (r - r.mean(1, keepdim=True)) / (r.std(1, unbiased=False, keepdim=True) + eps)
```

Use the population standard deviation, `unbiased=False`. PyTorch's default divides by $G - 1$ and gives $+1.208$ and $-0.725$ here. The `eps` matters: when all $G$ rewards tie, the numerator and the standard deviation are both zero, and the group must add no gradient.

REINFORCE leave-one-out (RLOO, Ahmadian and colleagues, 2024) takes a different baseline: the mean of the *other* $G - 1$ rewards, $\hat A_i = r_i - \frac{1}{G-1}\sum_{j \ne i} r_j = \frac{G}{G-1}(r_i - \bar r)$. For the group $(1, 0, 0, 1)$ it gives $1 - 1/3 = +0.667$ and $0 - 2/3 = -0.667$.

### Which baselines are unbiased

You can check this exactly, with no sampling. Take a softmax over four responses with probabilities $(0.4, 0.3, 0.2, 0.1)$ and rewards $(1, 0, 0.5, 2)$. The true gradient of expected reward with respect to the logits is $\pi_k(r_k - \mathbb{E}_\pi[r]) = (0.12, -0.21, -0.04, 0.13)$. Enumerate all $4^3 = 64$ groups of three, weight each by its probability, and average $\frac{1}{G}\sum_i \hat A_i \nabla \log \pi(y_i)$:

- RLOO's expected estimate equals the true gradient exactly, because its baseline never looks at $y_i$.
- The mean-centered estimate is exactly $\frac{G-1}{G} = \frac{2}{3}$ of it. Its baseline includes $r_i$, which shrinks the step by a constant, and a constant is harmless.
- The standard-deviation-scaled estimate is a different vector, with cosine similarity $0.980$ to the truth. Dividing by a quantity that depends on the sampled rewards biases the estimate, although it still points roughly uphill.

> [!KEY]
> RLOO is unbiased. GRPO's mean baseline only rescales it by $(G-1)/G$, and its division by the standard deviation is what makes it a biased estimator.

## The GRPO loss

GRPO keeps PPO's clipped ratio and adds a $k_3$ penalty. DeepSeekMath writes the objective, to maximize, as

$$
\mathcal{J}_{\text{GRPO}} = \mathbb{E}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{\lvert o_i \rvert}\sum_{t=1}^{\lvert o_i \rvert}\Big\{\min\big[\rho_{i,t}\hat A_{i},\ \operatorname{clip}(\rho_{i,t}, 1 - \varepsilon, 1 + \varepsilon)\,\hat A_{i}\big] - \beta\, k_3(i, t)\Big\}\Bigg]
$$

Read it aloud. $o_i$ is the $i$-th response and $\lvert o_i \rvert$ its length. $\rho_{i,t}$ is the per-token ratio against the policy that sampled. The $\min$ stops the gradient once the ratio has moved more than $\varepsilon$ in the direction the advantage wants. $\beta k_3$ keeps the policy near the reference. The $\frac{1}{\lvert o_i \rvert}$ averages each response over its own tokens, and $\frac{1}{G}$ then averages over responses. DeepSeekMath used $\beta = 0.04$ and $G = 64$. The DeepSeek-R1 paper (2025) writes one ratio per sequence instead, which GSPO, below, turns into a method.

In code, the loss splits into a per-token surrogate and an *aggregation* that turns the `(B, T)` array into one number:

```python
ratio = torch.exp(logp_new - logp_old)
per_token = torch.minimum(ratio * adv[:, None], ratio.clamp(1 - eps_low, 1 + eps_high) * adv[:, None])
loss = -aggregate(per_token - beta * k3, mask, mode)    # mode picks how to average
```

With one gradient step per batch, $\rho = 1$ when you differentiate, and each token of response $i$ gets gradient $\hat A_i / (G \lvert o_i \rvert)$.

## Where the averaging misleads

That $1/\lvert o_i \rvert$ is the first trap. A response's tokens share its advantage, so response $i$ contributes $\frac{\hat A_i}{\lvert o_i \rvert}\nabla \log \pi_\theta(o_i \mid x)$: an effective advantage of $\hat A_i / \lvert o_i \rvert$. Two wrong answers with $\hat A = -1$ and lengths 10 and 100 get per-token coefficients $-0.1$ and $-0.01$. The long wrong answer is punished ten times more gently, so wrong answers drift longer. A long correct answer is rewarded less per token, so correct answers drift shorter. Liu and colleagues (2025) call the fix *Dr. GRPO*: sum over tokens and divide by a constant, such as the generation budget.

The exercise shows this with exact gradients. A response is a correctness token (frozen at probability $0.5$), then a length $L$ from 1 to 8 with learnable logits for each correctness value, then $L - 1$ filler tokens. The reward is 1 when right, both runs use the advantage $r - 0.5$, and summing over all 16 outcomes removes sampling noise.

| Aggregation after 300 steps | Mean length if wrong | Mean length if right |
|---|---|---|
| GRPO: $1/\lvert o_i \rvert$ per sequence | $5.39$ | $2.36$ |
| Dr. GRPO: a constant | $4.50$ | $4.50$ |

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Under GRPO's per-sequence average, wrong answers grow from 4.5 to 5.39 tokens and right answers shrink to 2.36; under a constant normalizer both stay at 4.5">

<line x1="64" y1="252.0" x2="580" y2="252.0" class="stroke-grid"/>
<text x="58" y="256.0" class="fig-small" text-anchor="end">2</text>
<line x1="64" y1="194.0" x2="580" y2="194.0" class="stroke-grid"/>
<text x="58" y="198.0" class="fig-small" text-anchor="end">3</text>
<line x1="64" y1="136.0" x2="580" y2="136.0" class="stroke-grid"/>
<text x="58" y="140.0" class="fig-small" text-anchor="end">4</text>
<line x1="64" y1="78.0" x2="580" y2="78.0" class="stroke-grid"/>
<text x="58" y="82.0" class="fig-small" text-anchor="end">5</text>
<line x1="64" y1="20.0" x2="580" y2="20.0" class="stroke-grid"/>
<text x="58" y="24.0" class="fig-small" text-anchor="end">6</text>
<line x1="64" y1="252" x2="580" y2="252" class="stroke-muted"/>
<line x1="64.0" y1="252" x2="64.0" y2="257" class="stroke-muted"/>
<text x="64.0" y="272" class="fig-small" text-anchor="middle">0</text>
<line x1="236.0" y1="252" x2="236.0" y2="257" class="stroke-muted"/>
<text x="236.0" y="272" class="fig-small" text-anchor="middle">100</text>
<line x1="408.0" y1="252" x2="408.0" y2="257" class="stroke-muted"/>
<text x="408.0" y="272" class="fig-small" text-anchor="middle">200</text>
<line x1="580.0" y1="252" x2="580.0" y2="257" class="stroke-muted"/>
<text x="580.0" y="272" class="fig-small" text-anchor="middle">300</text>
<polyline points="64.0,107.0 65.7,106.8 67.4,106.5 69.2,106.3 70.9,106.1 72.6,105.8 74.3,105.6 76.0,105.4 77.8,105.2 79.5,104.9 81.2,104.7 82.9,104.5 84.6,104.2 86.4,104.0 88.1,103.8 89.8,103.6 91.5,103.4 93.2,103.1 95.0,102.9 96.7,102.7 98.4,102.5 100.1,102.2 101.8,102.0 103.6,101.8 105.3,101.6 107.0,101.4 108.7,101.2 110.4,100.9 112.2,100.7 113.9,100.5 115.6,100.3 117.3,100.1 119.0,99.9 120.8,99.7 122.5,99.4 124.2,99.2 125.9,99.0 127.6,98.8 129.4,98.6 131.1,98.4 132.8,98.2 134.5,98.0 136.2,97.8 138.0,97.6 139.7,97.3 141.4,97.1 143.1,96.9 144.8,96.7 146.6,96.5 148.3,96.3 150.0,96.1 151.7,95.9 153.4,95.7 155.2,95.5 156.9,95.3 158.6,95.1 160.3,94.9 162.0,94.7 163.8,94.5 165.5,94.3 167.2,94.1 168.9,93.9 170.6,93.7 172.4,93.5 174.1,93.3 175.8,93.1 177.5,92.9 179.2,92.7 181.0,92.5 182.7,92.3 184.4,92.1 186.1,91.9 187.8,91.7 189.6,91.6 191.3,91.4 193.0,91.2 194.7,91.0 196.4,90.8 198.2,90.6 199.9,90.4 201.6,90.2 203.3,90.0 205.0,89.8 206.8,89.6 208.5,89.5 210.2,89.3 211.9,89.1 213.6,88.9 215.4,88.7 217.1,88.5 218.8,88.3 220.5,88.2 222.2,88.0 224.0,87.8 225.7,87.6 227.4,87.4 229.1,87.2 230.8,87.0 232.6,86.9 234.3,86.7 236.0,86.5 237.7,86.3 239.4,86.1 241.2,86.0 242.9,85.8 244.6,85.6 246.3,85.4 248.0,85.2 249.8,85.1 251.5,84.9 253.2,84.7 254.9,84.5 256.6,84.3 258.4,84.2 260.1,84.0 261.8,83.8 263.5,83.6 265.2,83.5 267.0,83.3 268.7,83.1 270.4,82.9 272.1,82.8 273.8,82.6 275.6,82.4 277.3,82.2 279.0,82.1 280.7,81.9 282.4,81.7 284.2,81.5 285.9,81.4 287.6,81.2 289.3,81.0 291.0,80.9 292.8,80.7 294.5,80.5 296.2,80.3 297.9,80.2 299.6,80.0 301.4,79.8 303.1,79.7 304.8,79.5 306.5,79.3 308.2,79.2 310.0,79.0 311.7,78.8 313.4,78.7 315.1,78.5 316.8,78.3 318.6,78.2 320.3,78.0 322.0,77.8 323.7,77.7 325.4,77.5 327.2,77.3 328.9,77.2 330.6,77.0 332.3,76.8 334.0,76.7 335.8,76.5 337.5,76.3 339.2,76.2 340.9,76.0 342.6,75.9 344.4,75.7 346.1,75.5 347.8,75.4 349.5,75.2 351.2,75.0 353.0,74.9 354.7,74.7 356.4,74.6 358.1,74.4 359.8,74.2 361.6,74.1 363.3,73.9 365.0,73.8 366.7,73.6 368.4,73.4 370.2,73.3 371.9,73.1 373.6,73.0 375.3,72.8 377.0,72.7 378.8,72.5 380.5,72.3 382.2,72.2 383.9,72.0 385.6,71.9 387.4,71.7 389.1,71.6 390.8,71.4 392.5,71.3 394.2,71.1 396.0,70.9 397.7,70.8 399.4,70.6 401.1,70.5 402.8,70.3 404.6,70.2 406.3,70.0 408.0,69.9 409.7,69.7 411.4,69.6 413.2,69.4 414.9,69.3 416.6,69.1 418.3,69.0 420.0,68.8 421.8,68.6 423.5,68.5 425.2,68.3 426.9,68.2 428.6,68.0 430.4,67.9 432.1,67.7 433.8,67.6 435.5,67.4 437.2,67.3 439.0,67.1 440.7,67.0 442.4,66.9 444.1,66.7 445.8,66.6 447.6,66.4 449.3,66.3 451.0,66.1 452.7,66.0 454.4,65.8 456.2,65.7 457.9,65.5 459.6,65.4 461.3,65.2 463.0,65.1 464.8,64.9 466.5,64.8 468.2,64.6 469.9,64.5 471.6,64.4 473.4,64.2 475.1,64.1 476.8,63.9 478.5,63.8 480.2,63.6 482.0,63.5 483.7,63.3 485.4,63.2 487.1,63.1 488.8,62.9 490.6,62.8 492.3,62.6 494.0,62.5 495.7,62.3 497.4,62.2 499.2,62.1 500.9,61.9 502.6,61.8 504.3,61.6 506.0,61.5 507.8,61.3 509.5,61.2 511.2,61.1 512.9,60.9 514.6,60.8 516.4,60.6 518.1,60.5 519.8,60.4 521.5,60.2 523.2,60.1 525.0,59.9 526.7,59.8 528.4,59.7 530.1,59.5 531.8,59.4 533.6,59.2 535.3,59.1 537.0,59.0 538.7,58.8 540.4,58.7 542.2,58.6 543.9,58.4 545.6,58.3 547.3,58.1 549.0,58.0 550.8,57.9 552.5,57.7 554.2,57.6 555.9,57.5 557.6,57.3 559.4,57.2 561.1,57.1 562.8,56.9 564.5,56.8 566.2,56.6 568.0,56.5 569.7,56.4 571.4,56.2 573.1,56.1 574.8,56.0 576.6,55.8 578.3,55.7 580.0,55.6" class="stroke-bad" fill="none"/>
<polyline points="64.0,107.0 65.7,107.2 67.4,107.5 69.2,107.7 70.9,107.9 72.6,108.2 74.3,108.4 76.0,108.6 77.8,108.9 79.5,109.1 81.2,109.4 82.9,109.6 84.6,109.8 86.4,110.1 88.1,110.3 89.8,110.6 91.5,110.8 93.2,111.1 95.0,111.3 96.7,111.6 98.4,111.8 100.1,112.1 101.8,112.3 103.6,112.6 105.3,112.8 107.0,113.1 108.7,113.3 110.4,113.6 112.2,113.8 113.9,114.1 115.6,114.3 117.3,114.6 119.0,114.8 120.8,115.1 122.5,115.4 124.2,115.6 125.9,115.9 127.6,116.1 129.4,116.4 131.1,116.7 132.8,116.9 134.5,117.2 136.2,117.5 138.0,117.7 139.7,118.0 141.4,118.3 143.1,118.6 144.8,118.8 146.6,119.1 148.3,119.4 150.0,119.6 151.7,119.9 153.4,120.2 155.2,120.5 156.9,120.8 158.6,121.0 160.3,121.3 162.0,121.6 163.8,121.9 165.5,122.2 167.2,122.5 168.9,122.7 170.6,123.0 172.4,123.3 174.1,123.6 175.8,123.9 177.5,124.2 179.2,124.5 181.0,124.8 182.7,125.1 184.4,125.4 186.1,125.7 187.8,126.0 189.6,126.3 191.3,126.6 193.0,126.9 194.7,127.2 196.4,127.5 198.2,127.8 199.9,128.1 201.6,128.4 203.3,128.7 205.0,129.0 206.8,129.3 208.5,129.6 210.2,130.0 211.9,130.3 213.6,130.6 215.4,130.9 217.1,131.2 218.8,131.6 220.5,131.9 222.2,132.2 224.0,132.5 225.7,132.9 227.4,133.2 229.1,133.5 230.8,133.9 232.6,134.2 234.3,134.5 236.0,134.9 237.7,135.2 239.4,135.5 241.2,135.9 242.9,136.2 244.6,136.6 246.3,136.9 248.0,137.2 249.8,137.6 251.5,137.9 253.2,138.3 254.9,138.7 256.6,139.0 258.4,139.4 260.1,139.7 261.8,140.1 263.5,140.4 265.2,140.8 267.0,141.2 268.7,141.5 270.4,141.9 272.1,142.3 273.8,142.6 275.6,143.0 277.3,143.4 279.0,143.8 280.7,144.1 282.4,144.5 284.2,144.9 285.9,145.3 287.6,145.7 289.3,146.1 291.0,146.4 292.8,146.8 294.5,147.2 296.2,147.6 297.9,148.0 299.6,148.4 301.4,148.8 303.1,149.2 304.8,149.6 306.5,150.0 308.2,150.4 310.0,150.8 311.7,151.2 313.4,151.7 315.1,152.1 316.8,152.5 318.6,152.9 320.3,153.3 322.0,153.7 323.7,154.2 325.4,154.6 327.2,155.0 328.9,155.5 330.6,155.9 332.3,156.3 334.0,156.8 335.8,157.2 337.5,157.6 339.2,158.1 340.9,158.5 342.6,159.0 344.4,159.4 346.1,159.9 347.8,160.3 349.5,160.8 351.2,161.2 353.0,161.7 354.7,162.1 356.4,162.6 358.1,163.1 359.8,163.5 361.6,164.0 363.3,164.5 365.0,164.9 366.7,165.4 368.4,165.9 370.2,166.4 371.9,166.8 373.6,167.3 375.3,167.8 377.0,168.3 378.8,168.8 380.5,169.3 382.2,169.7 383.9,170.2 385.6,170.7 387.4,171.2 389.1,171.7 390.8,172.2 392.5,172.7 394.2,173.2 396.0,173.7 397.7,174.2 399.4,174.8 401.1,175.3 402.8,175.8 404.6,176.3 406.3,176.8 408.0,177.3 409.7,177.8 411.4,178.4 413.2,178.9 414.9,179.4 416.6,179.9 418.3,180.5 420.0,181.0 421.8,181.5 423.5,182.0 425.2,182.6 426.9,183.1 428.6,183.6 430.4,184.2 432.1,184.7 433.8,185.3 435.5,185.8 437.2,186.3 439.0,186.9 440.7,187.4 442.4,188.0 444.1,188.5 445.8,189.1 447.6,189.6 449.3,190.2 451.0,190.7 452.7,191.3 454.4,191.8 456.2,192.4 457.9,192.9 459.6,193.5 461.3,194.0 463.0,194.6 464.8,195.1 466.5,195.7 468.2,196.2 469.9,196.8 471.6,197.4 473.4,197.9 475.1,198.5 476.8,199.0 478.5,199.6 480.2,200.1 482.0,200.7 483.7,201.3 485.4,201.8 487.1,202.4 488.8,202.9 490.6,203.5 492.3,204.0 494.0,204.6 495.7,205.2 497.4,205.7 499.2,206.3 500.9,206.8 502.6,207.4 504.3,207.9 506.0,208.5 507.8,209.0 509.5,209.6 511.2,210.2 512.9,210.7 514.6,211.3 516.4,211.8 518.1,212.4 519.8,212.9 521.5,213.4 523.2,214.0 525.0,214.5 526.7,215.1 528.4,215.6 530.1,216.2 531.8,216.7 533.6,217.2 535.3,217.8 537.0,218.3 538.7,218.8 540.4,219.4 542.2,219.9 543.9,220.4 545.6,221.0 547.3,221.5 549.0,222.0 550.8,222.5 552.5,223.0 554.2,223.6 555.9,224.1 557.6,224.6 559.4,225.1 561.1,225.6 562.8,226.1 564.5,226.6 566.2,227.1 568.0,227.6 569.7,228.1 571.4,228.6 573.1,229.1 574.8,229.6 576.6,230.1 578.3,230.6 580.0,231.1" class="stroke-c" fill="none"/>
<polyline points="64.0,107.0 65.7,107.0 67.4,107.0 69.2,107.0 70.9,107.0 72.6,107.0 74.3,107.0 76.0,107.0 77.8,107.0 79.5,107.0 81.2,107.0 82.9,107.0 84.6,107.0 86.4,107.0 88.1,107.0 89.8,107.0 91.5,107.0 93.2,107.0 95.0,107.0 96.7,107.0 98.4,107.0 100.1,107.0 101.8,107.0 103.6,107.0 105.3,107.0 107.0,107.0 108.7,107.0 110.4,107.0 112.2,107.0 113.9,107.0 115.6,107.0 117.3,107.0 119.0,107.0 120.8,107.0 122.5,107.0 124.2,107.0 125.9,107.0 127.6,107.0 129.4,107.0 131.1,107.0 132.8,107.0 134.5,107.0 136.2,107.0 138.0,107.0 139.7,107.0 141.4,107.0 143.1,107.0 144.8,107.0 146.6,107.0 148.3,107.0 150.0,107.0 151.7,107.0 153.4,107.0 155.2,107.0 156.9,107.0 158.6,107.0 160.3,107.0 162.0,107.0 163.8,107.0 165.5,107.0 167.2,107.0 168.9,107.0 170.6,107.0 172.4,107.0 174.1,107.0 175.8,107.0 177.5,107.0 179.2,107.0 181.0,107.0 182.7,107.0 184.4,107.0 186.1,107.0 187.8,107.0 189.6,107.0 191.3,107.0 193.0,107.0 194.7,107.0 196.4,107.0 198.2,107.0 199.9,107.0 201.6,107.0 203.3,107.0 205.0,107.0 206.8,107.0 208.5,107.0 210.2,107.0 211.9,107.0 213.6,107.0 215.4,107.0 217.1,107.0 218.8,107.0 220.5,107.0 222.2,107.0 224.0,107.0 225.7,107.0 227.4,107.0 229.1,107.0 230.8,107.0 232.6,107.0 234.3,107.0 236.0,107.0 237.7,107.0 239.4,107.0 241.2,107.0 242.9,107.0 244.6,107.0 246.3,107.0 248.0,107.0 249.8,107.0 251.5,107.0 253.2,107.0 254.9,107.0 256.6,107.0 258.4,107.0 260.1,107.0 261.8,107.0 263.5,107.0 265.2,107.0 267.0,107.0 268.7,107.0 270.4,107.0 272.1,107.0 273.8,107.0 275.6,107.0 277.3,107.0 279.0,107.0 280.7,107.0 282.4,107.0 284.2,107.0 285.9,107.0 287.6,107.0 289.3,107.0 291.0,107.0 292.8,107.0 294.5,107.0 296.2,107.0 297.9,107.0 299.6,107.0 301.4,107.0 303.1,107.0 304.8,107.0 306.5,107.0 308.2,107.0 310.0,107.0 311.7,107.0 313.4,107.0 315.1,107.0 316.8,107.0 318.6,107.0 320.3,107.0 322.0,107.0 323.7,107.0 325.4,107.0 327.2,107.0 328.9,107.0 330.6,107.0 332.3,107.0 334.0,107.0 335.8,107.0 337.5,107.0 339.2,107.0 340.9,107.0 342.6,107.0 344.4,107.0 346.1,107.0 347.8,107.0 349.5,107.0 351.2,107.0 353.0,107.0 354.7,107.0 356.4,107.0 358.1,107.0 359.8,107.0 361.6,107.0 363.3,107.0 365.0,107.0 366.7,107.0 368.4,107.0 370.2,107.0 371.9,107.0 373.6,107.0 375.3,107.0 377.0,107.0 378.8,107.0 380.5,107.0 382.2,107.0 383.9,107.0 385.6,107.0 387.4,107.0 389.1,107.0 390.8,107.0 392.5,107.0 394.2,107.0 396.0,107.0 397.7,107.0 399.4,107.0 401.1,107.0 402.8,107.0 404.6,107.0 406.3,107.0 408.0,107.0 409.7,107.0 411.4,107.0 413.2,107.0 414.9,107.0 416.6,107.0 418.3,107.0 420.0,107.0 421.8,107.0 423.5,107.0 425.2,107.0 426.9,107.0 428.6,107.0 430.4,107.0 432.1,107.0 433.8,107.0 435.5,107.0 437.2,107.0 439.0,107.0 440.7,107.0 442.4,107.0 444.1,107.0 445.8,107.0 447.6,107.0 449.3,107.0 451.0,107.0 452.7,107.0 454.4,107.0 456.2,107.0 457.9,107.0 459.6,107.0 461.3,107.0 463.0,107.0 464.8,107.0 466.5,107.0 468.2,107.0 469.9,107.0 471.6,107.0 473.4,107.0 475.1,107.0 476.8,107.0 478.5,107.0 480.2,107.0 482.0,107.0 483.7,107.0 485.4,107.0 487.1,107.0 488.8,107.0 490.6,107.0 492.3,107.0 494.0,107.0 495.7,107.0 497.4,107.0 499.2,107.0 500.9,107.0 502.6,107.0 504.3,107.0 506.0,107.0 507.8,107.0 509.5,107.0 511.2,107.0 512.9,107.0 514.6,107.0 516.4,107.0 518.1,107.0 519.8,107.0 521.5,107.0 523.2,107.0 525.0,107.0 526.7,107.0 528.4,107.0 530.1,107.0 531.8,107.0 533.6,107.0 535.3,107.0 537.0,107.0 538.7,107.0 540.4,107.0 542.2,107.0 543.9,107.0 545.6,107.0 547.3,107.0 549.0,107.0 550.8,107.0 552.5,107.0 554.2,107.0 555.9,107.0 557.6,107.0 559.4,107.0 561.1,107.0 562.8,107.0 564.5,107.0 566.2,107.0 568.0,107.0 569.7,107.0 571.4,107.0 573.1,107.0 574.8,107.0 576.6,107.0 578.3,107.0 580.0,107.0" class="stroke-muted dashed" fill="none"/>
<text x="322" y="292" text-anchor="middle">gradient step</text>
<text x="58" y="14" text-anchor="end" class="fig-small">mean length</text>
<line x1="350" y1="28" x2="380" y2="28" class="stroke-bad"/>
<text x="388" y="32" class="fig-small">GRPO, wrong answers</text>
<line x1="350" y1="46" x2="380" y2="46" class="stroke-c"/>
<text x="388" y="50" class="fig-small">GRPO, right answers</text>
<line x1="350" y1="64" x2="380" y2="64" class="stroke-muted dashed"/>
<text x="388" y="68" class="fig-small">Dr. GRPO, both (4.5)</text>
</svg>
Figure 2. Both runs start with a mean length of 4.5. Notice that GRPO's wrong answers lengthen and its right answers shorten at the same time, while the constant normalizer leaves both flat.
```

With a constant normalizer, the update on the length logits is proportional to $\hat A_c \nabla \mathbb{E}[\text{const}] = 0$, so nothing moves. With $1/(L+1)$ it is proportional to $\hat A_c \nabla \mathbb{E}[1/(L+1)]$, and for wrong answers $\hat A_c < 0$, so ascent lowers $\mathbb{E}[1/(L+1)]$, which lengthens them.

> [!KEY]
> Averaging each response over its own length gives it an effective advantage of $\hat A_i / \lvert o_i \rvert$. That rewards short right answers and spares long wrong ones. A constant normalizer removes the effect.

The second trap is the standard deviation. With binary rewards and success rate $p$, it is $\sqrt{p(1-p)}$, so a question's total advantage mass is multiplied by $1/\sqrt{p(1-p)}$: $2.0$ at $p = 0.5$ and $3.33$ at $p = 0.1$ or $0.9$. Questions that are nearly always or nearly never solved count $1.67$ times more. Dr. GRPO uses the plain mean-centered advantage.

Two more fixes change one piece each:

- **DAPO** (Yu and colleagues, 2025) uses *clip-higher*: $\varepsilon_{\text{low}} = 0.2$ but $\varepsilon_{\text{high}} = 0.28$. A symmetric clip lets a token at probability $0.01$ reach only $0.012$, so exploratory tokens barely move and entropy collapses. With $0.28$ it can reach $0.0128$. DAPO also averages over all tokens in the group, `(x * mask).sum() / mask.sum()`, so long responses are penalized in full.
- **GSPO** (Zheng and colleagues, 2025) clips one length-normalized ratio per sequence, $s_i = \exp\big(\frac{1}{\lvert o_i \rvert}\sum_t \log \rho_{i,t}\big)$, because token ratios are noisy. It sits near 1, so the paper's clip range is tiny, $3 \times 10^{-4}$ and $4 \times 10^{-4}$ (the paper's GRPO baseline uses $0.2$ and $0.27$).

  ```python
  s = torch.exp(((logp_new - logp_old) * mask).sum(1) / mask.sum(1))   # (B,)
  ```

  A response with token ratios $(4, 1)$ has $s = \sqrt{4 \times 1} = 2$. One with $(8, 1, 1, 1)$ has $s = 8^{1/4} = 1.68$, not the whole-sequence ratio $8$.

> [!WARNING]
> The clip is one-sided, because of the $\min$. For $\hat A < 0$ and $\rho = 1.5$, $\min(-1.5, -1.28) = -1.5$ keeps the unclipped value and its full gradient. Clipping never protects a token that has moved the wrong way.

## Run GRPO on Qwen3-0.6B

The **Run** block trains Qwen3-0.6B with your functions when a GPU is available. Each step samples 8 two-digit multiplication questions, 8 responses each, with at most 96 new tokens. The reward is a Python rule: 1 when the last `Answer: N` in the response equals the product, and 0 otherwise. The block makes one gradient step per batch with AdamW at learning rate $10^{-6}$, with $\beta = 0.04$ and the $k_3$ KL. It runs twice:

- `mode="grpo"`: advantages scaled by the standard deviation, and `"sequence_mean"` aggregation.
- `mode="dr_grpo"`: mean-centered advantages, and `"constant"` aggregation with `max_tokens=96`.

Every few steps it prints the mean reward, the mean length of right and wrong responses, and the KL. Twenty steps is a short run with noisy batches, so compare the trend in the two length columns between modes, not single rows. Without a GPU, **Run** prints the toy table from Figure 2 instead.

> [!TIP]
> The run keeps float32 master weights and runs its forward passes in bfloat16. A weight stored in bfloat16 has between two and three significant digits, so an update of $10^{-6}$ to a weight of $0.02$ rounds to no change at all.

> [!KEY]
> Sampling a group of answers replaces the critic, because each answer's reward minus its siblings' reward says whether it was better than usual. The price is in the averaging: dividing by length and by the standard deviation changes which answers and questions count, and the length bias is visible in a toy you can compute exactly.

> [!RECAP]
> - RLOO subtracts the siblings' mean and is unbiased. GRPO's mean-centring scales it by $(G-1)/G$, and its standard-deviation division biases it.
> - The GRPO loss is minus the aggregate of the clipped surrogate minus $\beta k_3$.
> - Per-sequence averaging gives each response an effective advantage of $\hat A_i / \lvert o_i \rvert$, which lengthens wrong answers. A constant normalizer removes that.
> - DAPO raises only the upper clip. GSPO clips one geometric-mean ratio per response, which `gspo_ratio` computes.

## Build it step by step

The checks test the functions in this order. The function `k3_kl` is given in full, because you built it in the previous chapter on RLHF.

1. **`rloo_advantages(rewards, group_size)`.** Takes `(B,)` and returns `(B,)`. Reshape with `view(-1, group_size)` and apply the $G/(G-1)$ form. Common mistake: reshaping as `(G, P)`, which mixes prompts.
2. **`grpo_advantages(rewards, group_size, scale_by_std, eps)`.** Use `std(dim=1, unbiased=False)`. Common mistake: the Bessel default, which misses the hand values by about 6%.
3. **The exact-expectation check** uses only your two advantage functions.
4. **`clipped_surrogate(logp_new, logp_old, advantages, eps_low, eps_high)`.** Returns `(B, T)`. Broadcast with `advantages[:, None]` and take `torch.minimum`. Common mistake: clipping the product instead of the ratio.
5. **`aggregate(per_token, mask, mode, max_tokens)`.** Zero the padding with `torch.where` first, because padded log-probabilities can be anything.
6. **`gspo_ratio(logp_new, logp_old, mask)`.** Takes three `(B, T)` arrays and returns `(B,)`. Zero the padded log-ratios, sum each row, divide by the row's mask sum, and exponentiate. Common mistakes: skipping the division, which gives the whole-sequence ratio, and dividing by `T`, which counts padding as tokens.
7. **`grpo_loss(...)`.** Subtract `beta * k3_kl` from the surrogate, aggregate, and negate. Common mistake: forgetting the sign, so training runs downhill.
8. **`length_bias_toy(mode, steps, lr)`.** Build the `(16, 9)` batch from `torch.log_softmax(logits, dim=1)` on each step. Weight row $o$ by $16\,\pi(o)\,\hat A_o$ with $\pi(o)$ detached, call your `aggregate`, and step with `torch.autograd.grad`. Common mistake: letting gradient flow through $\pi(o)$, which adds a term the policy gradient does not have.

## Where this breaks

**Tied groups.** When every response in a group is right, or every one is wrong, all advantages are zero and the group teaches nothing. [Verifiable rewards](/lessons/068-verifiable-rewards) covers DAPO's dynamic sampling, which drops such groups.

**Entropy collapse.** Symmetric clipping and positive updates on likely tokens make the policy deterministic early, and exploration stops. Clip-higher is one remedy, and [Reasoning with verifiable rewards](/lessons/069-reasoning-with-verifiable-rewards) measures the collapse.

**Length drift.** Growing length alone does not prove the model reasons more. Check the length of right and wrong answers separately.

## Further reading

The following papers define the methods in this chapter:

- [Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (2024)](https://arxiv.org/abs/2402.03300): introduces GRPO with per-token ratios and the $k_3$ penalty.
- [Ahmadian et al., "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs" (2024)](https://arxiv.org/abs/2402.14740): RLOO for language models.
- [Liu et al., "Understanding R1-Zero-Like Training: A Critical Perspective" (2025)](https://arxiv.org/abs/2503.20783): the length and difficulty biases, and Dr. GRPO.
- [Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale" (2025)](https://arxiv.org/abs/2503.14476): clip-higher, dynamic sampling, token-level loss, and overlong shaping.
- [Zheng et al., "Group Sequence Policy Optimization" (2025)](https://arxiv.org/abs/2507.18071): the length-normalized sequence ratio, Equation 7.
- [DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025)](https://arxiv.org/abs/2501.12948): GRPO at scale.

For university courses, see Stanford CS336, *Language Modeling from Scratch* (Spring 2026), Lecture 16, "Post-training - RLVR", and Stanford CS 224R, *Deep Reinforcement Learning* (Spring 2026), Lecture 10, "RL for LLMs: Reasoning" (guest lecture), and Lecture 3, "Policy Gradients".

Open questions that researchers are working on:

- Which aggregation gives the lowest-variance unbiased gradient when response lengths vary a lot?
- How large must a group be before rare correct responses show up often enough to learn from?

## The exercise

You build the group baselines, the GRPO loss, and the length-bias experiment, then run them on a real model when a GPU is available. Write `rloo_advantages` and `grpo_advantages`, then `clipped_surrogate`, `aggregate`, `gspo_ratio`, and `grpo_loss`, then `length_bias_toy`.

The checks run on the CPU with toy tensors. They verify the advantages against the hand values and, by exact enumeration, which baseline is unbiased. They verify that the clip stops the gradient on the right side, that the three aggregations match hand values on unequal lengths, that `gspo_ratio` returns the geometric mean of the token ratios and ignores padding, and that the on-policy loss gradient is $-(\hat A - \beta(1 - \pi_{\text{ref}}/\pi))/(B\lvert o_i \rvert)$ per token. Last, they confirm that GRPO lengthens wrong answers and shortens right ones, while Dr. GRPO leaves both at 4.5.
