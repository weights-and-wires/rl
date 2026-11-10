---
title: Reasoning with verifiable rewards
slug: 062-reasoning-with-verifiable-rewards
part: "Part 9 — RL for language models"
summary: Train on answers a program can check, measure pass@k without bias, and see why entropy collapse lets pass@1 rise while pass@k falls.
minutes: 40
gpu: true
objectives:
  - Write a verifiable reward for the R1-Zero answer format that rejects malformed outputs.
  - Compute the unbiased pass@k estimator by hand and in a numerically stable form.
  - Derive the first-order entropy change of a softmax as a covariance, and measure entropy collapse during RLVR.
  - Explain on a toy why pass@1 can rise while pass@k at large k falls.
prereqs:
  - 056-reward-models
  - 060-group-relative-policy-optimization
  - 061-verifiable-rewards
exercise: 062-reasoning-with-verifiable-rewards
---

# Reasoning with verifiable rewards

A learned reward model is a guess about what people want, and a policy that pushes hard enough finds where the guess is wrong. For some tasks you need no guess: a math answer is right or wrong, a program passes its tests or fails them, and a few lines of Python can check each.

Reinforcement learning with verifiable rewards, or RLVR (named in Tülu 3, 2024), trains against such checks. DeepSeek-R1-Zero (2025) trained a base model with GRPO on rule-checked math, using a fixed answer template.

After RLVR the model is right more often on its first try. Did it learn to solve problems it could not solve before, or only get better at picking answers it could already find? This chapter gives you tools to ask, and a toy where you watch the answer form.

## What you already know

- GRPO uses group-normalized advantages, and with one gradient step per batch its loss has the gradient of plain REINFORCE, from [Group-relative policy optimization](/lessons/060-group-relative-policy-optimization).
- A verifier can normalize answers, pay a format bonus, and drop all-equal reward groups, from [Verifiable rewards](/lessons/061-verifiable-rewards).
- A reward model can be hacked: a policy optimizing it finds its mistakes, from [Reward models from preferences](/lessons/056-reward-models).
- The *entropy* of a distribution, $H(\pi) = -\sum_a \pi_a \log \pi_a$, is zero when one outcome is certain and largest when all are equally likely. A softmax policy $\pi = \operatorname{softmax}(z)$ moves when its logits $z$ move.

## Reward what a program can check

A *verifier* is a function from a response to a reward. [Verifiable rewards](/lessons/061-verifiable-rewards) covers how to design one. Here the rule is strict. DeepSeek-R1-Zero asks the model to reason inside `<think>` tags and answer inside `<answer>` tags, and the exercise's `answer_reward(text, target)` pays 1 only when the response closes its reasoning, then holds a closed `<answer>` block with exactly one integer equal to the target. In Python, `re.search(r"</think>\s*<answer>(.*?)</answer>", text, re.DOTALL)` finds the block and `re.fullmatch(r"-?\d+", content.strip())` accepts it.

> [!WARNING]
> A loose verifier is a reward model with a bug: one that accepts any response *containing* the right number pays a model that lists every number from 1 to 1000.

## Measure a model that gets k tries

*pass@k* is the chance that at least one of $k$ independent samples is correct. It asks whether the model *can* solve a problem, given tries, and pass@1 is plain accuracy.

Drawing exactly $k$ samples is noisy. Chen and colleagues (2021) draw $n \ge k$ samples, count the $c$ correct ones, and ask how likely at least one correct sample is among $k$ of the $n$ picked at random without replacement.

$$
\text{pass@}k = \mathbb{E}_{\text{problems}}\left[1 - \frac{\binom{n - c}{k}}{\binom{n}{k}}\right]
$$

Read it aloud: $\binom{n-c}{k}/\binom{n}{k}$ is the fraction of size-$k$ subsets with only wrong samples. With $n = 10$, $c = 3$, and $k = 5$, it is $1 - 21/252 = 0.917$. The shortcut $1 - (1 - c/n)^k = 0.832$ plugs an estimate into a nonlinear formula, which makes it biased.

Binomial coefficients overflow a float for large $n$, so cancel them into a product of $c$ factors:

$$
\frac{\binom{n - c}{k}}{\binom{n}{k}} = \prod_{j = n - c + 1}^{n}\left(1 - \frac{k}{j}\right)
$$

When $n - c < k$ every subset holds a correct sample, and pass@$k$ is exactly 1:

```python
if n - c < k:
    return 1.0
return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
```

## Train a toy arithmetic model

The toy answers 16 addition questions. A response is two tokens: a *method* (a habit such as column sums, drawn from one distribution that ignores the question) and then a digit from a table indexed by question and method. Each question has one *suited* method: ten questions suit method 0, and two each suit methods 1, 2, and 3. The suited method gives the right digit with probability $0.8$, and any other method with $0.002$. In the base model the four methods are equally likely, so each question's success rate is $0.25 \times 0.8 + 0.75 \times 0.002 = 0.2015$.

Training is RLVR as the GPU run does it. For each question, sample a group of 8, reward 1 for the right digit, compute GRPO advantages, and take one SGD step of `pg_loss`. The toy gives each question's exact success probability $p$, so pass@$k$ is $1 - (1 - p)^k$.

| Step | pass@1 | pass@8 | pass@64 | Entropy |
|---|---|---|---|---|
| 0 | $0.201$ | $0.835$ | $1.000$ | $1.574$ |
| 25 | $0.451$ | $0.738$ | $0.979$ | $1.086$ |
| 50 | $0.512$ | $0.668$ | $0.858$ | $0.794$ |
| 200 | $0.584$ | $0.638$ | $0.717$ | $0.571$ |

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Exact pass@k against k for the toy: after RLVR, pass@1 rises from 0.201 to 0.584 but pass@64 falls from 1.000 to 0.717">

<line x1="64" y1="252.0" x2="580" y2="252.0" class="stroke-grid"/>
<text x="58" y="256.0" class="fig-small" text-anchor="end">0</text>
<line x1="64" y1="196.8" x2="580" y2="196.8" class="stroke-grid"/>
<text x="58" y="200.8" class="fig-small" text-anchor="end">0.25</text>
<line x1="64" y1="141.5" x2="580" y2="141.5" class="stroke-grid"/>
<text x="58" y="145.5" class="fig-small" text-anchor="end">0.5</text>
<line x1="64" y1="86.3" x2="580" y2="86.3" class="stroke-grid"/>
<text x="58" y="90.3" class="fig-small" text-anchor="end">0.75</text>
<line x1="64" y1="31.0" x2="580" y2="31.0" class="stroke-grid"/>
<text x="58" y="35.0" class="fig-small" text-anchor="end">1</text>
<line x1="64" y1="252" x2="580" y2="252" class="stroke-muted"/>
<line x1="64.0" y1="252" x2="64.0" y2="257" class="stroke-muted"/>
<text x="64.0" y="272" class="fig-small" text-anchor="middle">1</text>
<line x1="121.3" y1="252" x2="121.3" y2="257" class="stroke-muted"/>
<text x="121.3" y="272" class="fig-small" text-anchor="middle">8</text>
<line x1="186.9" y1="252" x2="186.9" y2="257" class="stroke-muted"/>
<text x="186.9" y="272" class="fig-small" text-anchor="middle">16</text>
<line x1="317.9" y1="252" x2="317.9" y2="257" class="stroke-muted"/>
<text x="317.9" y="272" class="fig-small" text-anchor="middle">32</text>
<line x1="449.0" y1="252" x2="449.0" y2="257" class="stroke-muted"/>
<text x="449.0" y="272" class="fig-small" text-anchor="middle">48</text>
<line x1="580.0" y1="252" x2="580.0" y2="257" class="stroke-muted"/>
<text x="580.0" y="272" class="fig-small" text-anchor="middle">64</text>
<polyline points="64.0,207.5 72.2,171.9 80.4,143.5 88.6,120.9 96.8,102.8 105.0,88.3 113.1,76.8 121.3,67.6 129.5,60.2 137.7,54.3 145.9,49.6 154.1,45.9 162.3,42.9 170.5,40.5 178.7,38.6 186.9,37.1 195.0,35.9 203.2,34.9 211.4,34.1 219.6,33.5 227.8,33.0 236.0,32.6 244.2,32.3 252.4,32.0 260.6,31.8 268.8,31.7 277.0,31.6 285.1,31.5 293.3,31.4 301.5,31.3 309.7,31.3 317.9,31.2 326.1,31.2 334.3,31.2 342.5,31.1 350.7,31.1 358.9,31.1 367.0,31.1 375.2,31.1 383.4,31.1 391.6,31.1 399.8,31.1 408.0,31.1 416.2,31.1 424.4,31.1 432.6,31.1 440.8,31.1 449.0,31.1 457.1,31.1 465.3,31.1 473.5,31.0 481.7,31.0 489.9,31.0 498.1,31.0 506.3,31.0 514.5,31.0 522.7,31.0 530.9,31.0 539.0,31.0 547.2,31.0 555.4,31.0 563.6,31.0 571.8,31.0 580.0,31.0" class="stroke-a" fill="none"/>
<polyline points="64.0,123.0 72.2,113.8 80.4,112.9 88.6,112.5 96.8,112.1 105.0,111.8 113.1,111.4 121.3,111.1 129.5,110.7 137.7,110.4 145.9,110.0 154.1,109.7 162.3,109.3 170.5,109.0 178.7,108.6 186.9,108.3 195.0,108.0 203.2,107.6 211.4,107.3 219.6,107.0 227.8,106.6 236.0,106.3 244.2,106.0 252.4,105.6 260.6,105.3 268.8,105.0 277.0,104.7 285.1,104.3 293.3,104.0 301.5,103.7 309.7,103.4 317.9,103.1 326.1,102.8 334.3,102.4 342.5,102.1 350.7,101.8 358.9,101.5 367.0,101.2 375.2,100.9 383.4,100.6 391.6,100.3 399.8,100.0 408.0,99.7 416.2,99.4 424.4,99.1 432.6,98.8 440.8,98.5 449.0,98.2 457.1,97.9 465.3,97.6 473.5,97.3 481.7,97.0 489.9,96.7 498.1,96.5 506.3,96.2 514.5,95.9 522.7,95.6 530.9,95.3 539.0,95.0 547.2,94.8 555.4,94.5 563.6,94.2 571.8,93.9 580.0,93.7" class="stroke-b" fill="none"/>
<circle cx="64.0" cy="207.5" r="4" class="fill-a"/>
<circle cx="64.0" cy="123.0" r="4" class="fill-b"/>
<circle cx="580.0" cy="31.0" r="4" class="fill-a"/>
<circle cx="580.0" cy="93.7" r="4" class="fill-b"/>
<text x="322" y="292" text-anchor="middle">k, the number of tries</text>
<text x="58" y="14" text-anchor="end" class="fig-small">pass@k</text>
<line x1="350" y1="28" x2="380" y2="28" class="stroke-a"/>
<text x="388" y="32" class="fig-small">base model</text>
<line x1="350" y1="46" x2="380" y2="46" class="stroke-b"/>
<text x="388" y="50" class="fig-small">after 200 RLVR steps</text>
</svg>
Figure 1. Exact pass@k against k, before and after training. Notice that the curves cross: the trained model wins with one try and loses with eight or more.
```

pass@1 nearly triples, and pass@64 falls from 1 to $0.717$. The next two sections explain each half.

## Why entropy collapses

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Over 200 RLVR steps, pass@1 rises while entropy and pass@64 fall">

<line x1="64" y1="252.0" x2="580" y2="252.0" class="stroke-grid"/>
<text x="58" y="256.0" class="fig-small" text-anchor="end">0</text>
<line x1="64" y1="183.8" x2="580" y2="183.8" class="stroke-grid"/>
<text x="58" y="187.8" class="fig-small" text-anchor="end">0.5</text>
<line x1="64" y1="115.5" x2="580" y2="115.5" class="stroke-grid"/>
<text x="58" y="119.5" class="fig-small" text-anchor="end">1</text>
<line x1="64" y1="47.3" x2="580" y2="47.3" class="stroke-grid"/>
<text x="58" y="51.3" class="fig-small" text-anchor="end">1.5</text>
<line x1="64" y1="252" x2="580" y2="252" class="stroke-muted"/>
<line x1="64.0" y1="252" x2="64.0" y2="257" class="stroke-muted"/>
<text x="64.0" y="272" class="fig-small" text-anchor="middle">0</text>
<line x1="193.0" y1="252" x2="193.0" y2="257" class="stroke-muted"/>
<text x="193.0" y="272" class="fig-small" text-anchor="middle">50</text>
<line x1="322.0" y1="252" x2="322.0" y2="257" class="stroke-muted"/>
<text x="322.0" y="272" class="fig-small" text-anchor="middle">100</text>
<line x1="451.0" y1="252" x2="451.0" y2="257" class="stroke-muted"/>
<text x="451.0" y="272" class="fig-small" text-anchor="middle">150</text>
<line x1="580.0" y1="252" x2="580.0" y2="257" class="stroke-muted"/>
<text x="580.0" y="272" class="fig-small" text-anchor="middle">200</text>
<polyline points="64.0,224.5 66.6,223.3 69.2,222.2 71.7,220.7 74.3,218.7 76.9,217.0 79.5,215.2 82.1,213.2 84.6,211.9 87.2,210.4 89.8,208.8 92.4,207.0 95.0,205.1 97.5,203.7 100.1,202.4 102.7,201.0 105.3,199.8 107.9,198.4 110.4,197.1 113.0,196.1 115.6,194.9 118.2,193.9 120.8,193.0 123.3,191.8 125.9,191.3 128.5,190.5 131.1,189.9 133.7,189.6 136.2,189.1 138.8,188.6 141.4,188.1 144.0,187.6 146.6,187.2 149.1,186.6 151.7,186.3 154.3,186.0 156.9,185.6 159.5,185.3 162.0,185.0 164.6,184.7 167.2,184.5 169.8,184.2 172.4,183.9 174.9,183.7 177.5,183.5 180.1,183.2 182.7,183.0 185.3,182.8 187.8,182.5 190.4,182.3 193.0,182.1 195.6,181.9 198.2,181.8 200.7,181.6 203.3,181.4 205.9,181.4 208.5,181.3 211.1,181.2 213.6,181.1 216.2,181.0 218.8,180.8 221.4,180.8 224.0,180.6 226.5,180.4 229.1,180.3 231.7,180.2 234.3,180.0 236.9,179.9 239.4,179.8 242.0,179.7 244.6,179.6 247.2,179.5 249.8,179.4 252.3,179.3 254.9,179.1 257.5,179.0 260.1,179.0 262.7,178.9 265.2,178.8 267.8,178.7 270.4,178.6 273.0,178.5 275.6,178.4 278.1,178.3 280.7,178.2 283.3,178.1 285.9,178.0 288.5,177.9 291.0,177.8 293.6,177.7 296.2,177.6 298.8,177.5 301.4,177.4 303.9,177.3 306.5,177.2 309.1,177.1 311.7,177.1 314.3,177.0 316.8,176.9 319.4,176.8 322.0,176.8 324.6,176.7 327.2,176.6 329.7,176.6 332.3,176.6 334.9,176.5 337.5,176.5 340.1,176.4 342.6,176.3 345.2,176.2 347.8,176.1 350.4,176.1 353.0,176.0 355.5,175.9 358.1,175.8 360.7,175.8 363.3,175.7 365.9,175.6 368.4,175.6 371.0,175.5 373.6,175.5 376.2,175.4 378.8,175.4 381.3,175.4 383.9,175.3 386.5,175.2 389.1,175.2 391.7,175.1 394.2,175.1 396.8,175.0 399.4,175.0 402.0,174.9 404.6,174.9 407.1,174.8 409.7,174.7 412.3,174.7 414.9,174.6 417.5,174.6 420.0,174.5 422.6,174.5 425.2,174.4 427.8,174.4 430.4,174.3 432.9,174.3 435.5,174.2 438.1,174.2 440.7,174.2 443.3,174.1 445.8,174.1 448.4,174.1 451.0,174.0 453.6,174.0 456.2,173.9 458.7,173.9 461.3,173.8 463.9,173.8 466.5,173.8 469.1,173.8 471.6,173.7 474.2,173.7 476.8,173.7 479.4,173.6 482.0,173.6 484.5,173.5 487.1,173.5 489.7,173.5 492.3,173.4 494.9,173.4 497.4,173.4 500.0,173.4 502.6,173.3 505.2,173.3 507.8,173.3 510.3,173.2 512.9,173.2 515.5,173.1 518.1,173.1 520.7,173.1 523.2,173.0 525.8,173.0 528.4,172.9 531.0,172.9 533.6,172.9 536.1,172.8 538.7,172.8 541.3,172.8 543.9,172.8 546.5,172.7 549.0,172.7 551.6,172.7 554.2,172.6 556.8,172.6 559.4,172.6 561.9,172.5 564.5,172.5 567.1,172.5 569.7,172.5 572.3,172.4 574.8,172.4 577.4,172.4 580.0,172.3" class="stroke-c" fill="none"/>
<polyline points="64.0,115.5 66.6,115.5 69.2,115.5 71.7,115.5 74.3,115.5 76.9,115.5 79.5,115.5 82.1,115.5 84.6,115.5 87.2,115.5 89.8,115.5 92.4,115.5 95.0,115.6 97.5,115.6 100.1,115.6 102.7,115.7 105.3,115.7 107.9,115.8 110.4,115.9 113.0,116.1 115.6,116.3 118.2,116.6 120.8,116.9 123.3,117.5 125.9,117.9 128.5,118.4 131.1,118.9 133.7,119.3 136.2,119.7 138.8,120.2 141.4,121.0 144.0,121.7 146.6,122.4 149.1,123.5 151.7,124.2 154.3,124.7 156.9,125.4 159.5,126.1 162.0,126.8 164.6,127.6 167.2,127.9 169.8,128.9 172.4,129.6 174.9,130.2 177.5,130.9 180.1,131.5 182.7,132.3 185.3,132.7 187.8,133.5 190.4,134.3 193.0,134.8 195.6,135.7 198.2,136.1 200.7,136.3 203.3,136.8 205.9,137.0 208.5,137.0 211.1,137.1 213.6,137.1 216.2,137.2 218.8,137.6 221.4,137.4 224.0,137.7 226.5,138.3 229.1,138.7 231.7,139.1 234.3,139.8 236.9,140.2 239.4,140.5 242.0,140.5 244.6,140.6 247.2,141.1 249.8,141.2 252.3,141.3 254.9,141.7 257.5,142.3 260.1,142.0 262.7,142.2 265.2,142.5 267.8,142.6 270.4,142.7 273.0,143.2 275.6,143.6 278.1,144.0 280.7,144.5 283.3,144.8 285.9,145.0 288.5,145.7 291.0,146.0 293.6,146.1 296.2,146.2 298.8,146.6 301.4,146.7 303.9,146.9 306.5,147.0 309.1,147.2 311.7,147.4 314.3,147.7 316.8,148.0 319.4,148.3 322.0,148.3 324.6,148.3 327.2,148.4 329.7,148.2 332.3,148.2 334.9,148.3 337.5,148.4 340.1,148.5 342.6,148.7 345.2,148.7 347.8,149.1 350.4,149.3 353.0,149.6 355.5,149.6 358.1,149.9 360.7,150.0 363.3,150.2 365.9,150.4 368.4,150.6 371.0,150.7 373.6,150.7 376.2,150.8 378.8,150.8 381.3,150.8 383.9,150.9 386.5,150.9 389.1,151.2 391.7,151.1 394.2,151.1 396.8,151.0 399.4,151.0 402.0,151.1 404.6,151.1 407.1,151.2 409.7,151.2 412.3,151.3 414.9,151.3 417.5,151.6 420.0,151.6 422.6,151.8 425.2,151.6 427.8,151.7 430.4,151.7 432.9,151.7 435.5,151.7 438.1,151.7 440.7,151.8 443.3,151.8 445.8,151.8 448.4,151.7 451.0,151.7 453.6,151.9 456.2,152.0 458.7,152.1 461.3,152.1 463.9,152.1 466.5,152.1 469.1,152.2 471.6,152.3 474.2,152.3 476.8,152.3 479.4,152.4 482.0,152.4 484.5,152.4 487.1,152.4 489.7,152.3 492.3,152.4 494.9,152.6 497.4,152.8 500.0,152.8 502.6,152.8 505.2,152.9 507.8,152.9 510.3,152.8 512.9,152.8 515.5,152.9 518.1,153.0 520.7,153.1 523.2,153.2 525.8,153.3 528.4,153.4 531.0,153.4 533.6,153.3 536.1,153.5 538.7,153.7 541.3,153.7 543.9,153.7 546.5,153.7 549.0,153.7 551.6,153.7 554.2,153.9 556.8,154.0 559.4,154.1 561.9,154.1 564.5,154.1 567.1,154.1 569.7,154.1 572.3,154.1 574.8,154.1 577.4,154.1 580.0,154.2" class="stroke-bad" fill="none"/>
<polyline points="64.0,37.2 66.6,37.2 69.2,31.2 71.7,26.1 74.3,32.7 76.9,28.4 79.5,41.8 82.1,38.0 84.6,46.2 87.2,48.5 89.8,50.5 92.4,52.9 95.0,54.4 97.5,63.2 100.1,68.1 102.7,67.2 105.3,75.2 107.9,77.7 110.4,79.8 113.0,89.3 115.6,91.1 118.2,92.4 120.8,98.1 123.3,96.4 125.9,108.5 128.5,103.8 131.1,112.5 133.7,114.4 136.2,114.1 138.8,114.0 141.4,116.5 144.0,119.8 146.6,120.6 149.1,121.5 151.7,127.1 154.3,127.6 156.9,127.9 159.5,129.2 162.0,131.0 164.6,132.6 167.2,133.7 169.8,132.8 172.4,135.5 174.9,136.5 177.5,136.6 180.1,136.5 182.7,138.9 185.3,141.4 187.8,139.7 190.4,140.6 193.0,143.6 195.6,142.8 198.2,145.3 200.7,146.4 203.3,145.9 205.9,148.3 208.5,148.4 211.1,147.8 213.6,149.5 216.2,149.2 218.8,146.8 221.4,151.2 224.0,149.1 226.5,148.0 229.1,147.8 231.7,150.5 234.3,148.7 236.9,150.9 239.4,152.1 242.0,153.9 244.6,153.5 247.2,152.0 249.8,154.1 252.3,155.0 254.9,152.9 257.5,152.6 260.1,157.1 262.7,155.4 265.2,155.0 267.8,156.1 270.4,156.4 273.0,154.0 275.6,156.3 278.1,156.6 280.7,155.2 283.3,156.6 285.9,157.7 288.5,155.5 291.0,158.5 293.6,158.8 296.2,159.9 298.8,158.5 301.4,160.5 303.9,160.8 306.5,160.1 309.1,160.4 311.7,161.5 314.3,160.0 316.8,160.3 319.4,159.8 322.0,163.4 324.6,163.6 327.2,162.9 329.7,164.6 332.3,164.0 334.9,163.2 337.5,163.3 340.1,163.5 342.6,162.8 345.2,164.8 347.8,161.4 350.4,161.7 353.0,163.6 355.5,165.0 358.1,164.2 360.7,165.4 363.3,164.5 365.9,164.8 368.4,165.2 371.0,166.3 373.6,167.3 376.2,166.5 378.8,167.4 381.3,167.4 383.9,166.8 386.5,167.9 389.1,166.2 391.7,168.0 394.2,167.5 396.8,169.2 399.4,168.5 402.0,167.8 404.6,168.9 407.1,168.0 409.7,168.2 412.3,168.3 414.9,169.4 417.5,164.9 420.0,169.6 422.6,167.1 425.2,170.8 427.8,169.2 430.4,170.3 432.9,170.1 435.5,170.5 438.1,170.6 440.7,169.7 443.3,170.8 445.8,170.9 448.4,170.8 451.0,171.0 453.6,169.2 456.2,170.3 458.7,170.5 461.3,171.6 463.9,171.7 466.5,171.8 469.1,169.9 471.6,171.0 474.2,170.8 476.8,171.9 479.4,171.0 482.0,172.8 484.5,172.4 487.1,171.3 489.7,172.5 492.3,171.7 494.9,170.9 497.4,171.0 500.0,172.1 502.6,173.1 505.2,172.3 507.8,173.4 510.3,174.1 512.9,172.4 515.5,172.3 518.1,173.4 520.7,172.6 523.2,172.0 525.8,173.1 528.4,172.3 531.0,174.4 533.6,175.8 536.1,171.5 538.7,172.4 541.3,173.8 543.9,174.8 546.5,174.9 549.0,174.9 551.6,175.0 554.2,172.2 556.8,174.3 559.4,173.4 561.9,175.5 564.5,175.5 567.1,175.6 569.7,175.6 572.3,175.7 574.8,174.8 577.4,175.9 580.0,174.1" class="stroke-d dashed" fill="none"/>
<text x="322" y="292" text-anchor="middle">RLVR step</text>
<text x="58" y="14" text-anchor="end" class="fig-small">pass@k, entropy (nats)</text>
<line x1="350" y1="28" x2="380" y2="28" class="stroke-c"/>
<text x="388" y="32" class="fig-small">pass@1</text>
<line x1="350" y1="46" x2="380" y2="46" class="stroke-bad"/>
<text x="388" y="50" class="fig-small">pass@64</text>
<line x1="350" y1="64" x2="380" y2="64" class="stroke-d dashed"/>
<text x="388" y="68" class="fig-small">entropy</text>
</svg>
Figure 2. Entropy falls from 1.57 to 0.57 nats as pass@1 rises, and pass@64 falls with it. Notice that the entropy drops fastest where training helps most.
```

Most of the entropy drop is the method habit: method 0's probability rises to about $0.99$. Cui and colleagues (2025) found the same on real models: entropy collapses early, and performance levels off.

To see why, differentiate $H$ with respect to one logit:

$$
\frac{\partial H}{\partial z_k} = -\pi_k\big(\log \pi_k - \mathbb{E}_{\pi}[\log \pi]\big)
\quad\Longrightarrow\quad
\Delta H \approx -\operatorname{Cov}_{a \sim \pi}\big(\log \pi_a,\ \Delta z_a\big)
$$

Read it aloud: entropy falls when the logits that rise belong to tokens that are already likely. A positive advantage on a likely token is the common case in RLVR, because likely answers are the ones sampled and rewarded. Take $\pi = (0.7, 0.2, 0.1)$, with entropy $0.802$, and a step $\Delta z = 0.1 A$ with $A = (+1, -1, 0)$.

- $\log \pi = (-0.357, -1.609, -2.303)$, and $\operatorname{Cov}_\pi(\log \pi, A) = 0.473$.
- The estimate is $-0.1 \times 0.473 = -0.0473$. The exact change is $-0.0487$.
- With $A = (-1, +1, 0)$, the signs flip: $+0.0473$ estimated, $+0.0454$ exact.

In PyTorch, `-(p * (logp - (p * logp).sum(-1, keepdim=True)) * (dz - (p * dz).sum(-1, keepdim=True))).sum(-1)`.

> [!KEY]
> A step changes entropy by about minus the covariance between each token's log-probability and its logit change. RLVR rewards likely answers, so it drains entropy unless something pushes back.

Ten of sixteen questions reward method 0, so the shared habit moves toward it. The six other questions get their suited method less and less, and their success rate falls from $0.20$ to about $0.005$. pass@64 counts those questions, so it falls.

## Does RL add capability?

Yue and colleagues (2025) saw the same crossover on real models across math and code: RL models won at small $k$, base models at large $k$. They read it as RLVR sharpening the distribution toward answers the base model could already sample.

The toy does not settle it. Rerun it with Adam at step size $0.1$ and pass@1 reaches $0.998$ while pass@64 stays at $1.0$. Adam scales each table entry's step by its own gradient history, so the rare correct samples on the six minority questions teach the table in full. The outcome depends on whether rare correct paths get sampled and followed.

The evidence on real models is contested. ProRL (Liu and colleagues, 2025) trained far longer, with KL control and diverse tasks, and reports pass@$k$ gains even at large $k$. Wen and colleagues (2025) argue that pass@$k$ on final answers counts lucky guesses, and that a metric that checks the reasoning shows RLVR improving it.

A random reward can still raise accuracy on some models (Shao and colleagues, 2025, on Qwen2.5-Math). On-policy with one step per batch, its expected gradient is exactly zero, because the advantage is independent of the response. Any drift comes from clipping or the optimizer, and it did not carry over to Llama 3 or OLMo 2.

> [!KEY]
> RLVR reliably raises pass@1. Whether it raises pass@$k$ at large $k$ depends on training length, task diversity, exploration, and the metric.

## Count the groups that teach nothing

A prompt with pass rate $p$, sampled $G$ times, gives GRPO no signal when all $G$ rewards match. Every sample is right with probability $p^G$, and every sample is wrong with probability $(1 - p)^G$:

$$
P(\text{zero signal}) = p^{G} + (1 - p)^{G}
$$

For $p = 0.5$ and $G = 8$ that is $2/256 = 0.0078$. For $p = 0.9$ it is $0.9^8 + 0.1^8 = 0.4305$, so an easy prompt wastes nearly half its groups. At the toy's base $p = 0.2015$, $0.165$ of groups are empty. After training, a minority question has $p \approx 0.005$ and $0.961$ of its groups are empty, so its rare correct samples almost never land in a rewarded group. A group of 64 still wastes $0.726$. It is the quantity behind the all-equal filter in [Verifiable rewards](/lessons/061-verifiable-rewards). In code, `zero_signal_probability(p, group_size)` is `p**G + (1 - p)**G` with `p` as an array. Common mistake: `p**G` alone, which gives $0$ at $p = 0$ instead of $1$.

## Run R1-Zero-style RLVR on Qwen3-0.6B-Base

The **Run** block trains Qwen3-0.6B-Base, a model with no instruction tuning, with your functions when a GPU is available. The questions have the form $a \times b + c$, with $a$ from 2 to 9 and $b$ and $c$ from 10 to 99. Each step samples 8 questions and 8 responses each, at temperature 1, stopping at `</answer>` or 160 tokens. The reward is your `answer_reward`, called with `close_tag="</reasoning>"`. The step is your `grpo_advantages` and `pg_loss`, with AdamW at learning rate $10^{-6}$. R1-Zero also used a KL penalty, and this run drops it, as DAPO and Dr. GRPO do.

The run uses `<reasoning>` tags where R1-Zero used `<think>`, because of a trap. Qwen3's tokenizer maps `<think>` and `</think>` to reserved tokens that the chat models learned in post-training. The base model almost never saw them, and in 128 samples at temperatures 0.7 and 1.0 it closed the tag zero times. Every reward was 0, every advantage was 0, and nothing trained. With ordinary-text `<reasoning>`, the same model answers in the right format about 60% of the time and is right about 12% of the time at temperature 1. When every reward is 0, check that the model can produce the format.

Every 10 steps it draws 16 samples for each of 16 held-out questions and prints pass@1, pass@4, and pass@16 from your `pass_at_k`, with the mean reward and response length. Watch whether pass@16 moves as much as pass@1.

> [!KEY]
> After RLVR the model is right more often on the first try, and the toy shows the cost: entropy falls, the rare correct paths are sampled less, and pass@$k$ at large $k$ can fall. A higher pass@1 alone does not show it learned anything new.

> [!RECAP]
> - `answer_reward` pays 1 only for a closed reasoning tag, then a closed `<answer>` block holding exactly the right integer.
> - pass@$k$ is $1 - \prod_{j=n-c+1}^{n}(1 - k/j)$, and 1 when $n - c < k$.
> - $\Delta H \approx -\operatorname{Cov}_\pi(\log \pi, \Delta z)$, so rewarding likely answers drains entropy.
> - $p^G + (1-p)^G$ is the chance that a group has all-equal rewards and adds no gradient, so `zero_signal_probability(p, group_size)` prices too easy or too hard prompts.
- On the toy, RLVR with SGD raises pass@1 and lowers pass@64.

## Build it step by step

The checks test the functions in this order. The function `grpo_advantages` is given in full, because you built it in the previous chapter.

1. **`pass_at_k(n, c, k)`.** Use the product form. Common mistake: `math.comb` and float division, which overflows at $n = 5000$.
2. **`zero_signal_probability(p, group_size)`.** Add `p**group_size` and `(1 - p)**group_size` on `np.asarray(p)`. Common mistake: `1 - p**group_size`, which is the chance that some sample is wrong.
3. **`answer_reward(text, target)`.** Search for the block after `</think>`, strip it, and `fullmatch` an integer. Common mistake: `re.search(r"\d+")` inside the block, which accepts `42 or 43`.
4. **`pg_loss(token_logps, mask, advantages)`.** Zero the padding with `torch.where`, multiply by `advantages[:, None]`, sum, and divide by `mask.sum()`. Common mistake: multiplying `-inf` padding by a zero mask, which gives `nan`.
5. **`token_entropy(logits, mask)`.** `log_softmax`, then `torch.where(p > 0, p * logp, 0)`, summed over the vocabulary and averaged over the mask.
6. **`entropy_change_estimate(logits, delta_logits)`.** Center both quantities under `p` along the last axis, multiply, weight by `p`, sum, and negate.
7. **`train_rlvr_toy(...)`.** Build the batch with `torch.arange(16).repeat_interleave(group_size)`, so groups are contiguous. Record the entropy of the batch's logits, and each pass@$k$ row from `model.success_probs()`. Common mistake: forgetting `opt.zero_grad()`, so gradients pile up.

## Where this breaks

**Verifier loopholes.** The model finds what the check does not check, such as tests it can edit or formats the parser misreads. [Verifiable rewards](/lessons/061-verifiable-rewards) shows how to test a verifier for them. Keep it out of the model's reach, and read samples.

**Truncation.** A response cut off at the length limit gets reward 0, which punishes long, correct reasoning. DAPO masks or softly penalizes overlong responses instead.

**Contaminated baselines.** Base-model accuracy depends heavily on the prompt template, so compare models with the same template and sampling settings.

## Further reading

The following sources cover the recipe and the debates in this chapter:

- [Stanford CS336, Spring 2026, Lecture 16, "Post-training - RLVR"](https://cs336.stanford.edu/): a university lecture on training language models against verifiable rewards.
- [Stanford CS 224R, Spring 2026, "RL for LLMs: Reasoning" (guest lecture by Noam Brown)](https://cs224r.stanford.edu/): RL for reasoning models.
- [DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025)](https://arxiv.org/abs/2501.12948): R1-Zero's template, rule-based rewards, and length growth.
- [Lambert et al., "Tülu 3: Pushing Frontiers in Open Language Model Post-Training" (2024)](https://arxiv.org/abs/2411.15124): names RLVR and uses it alongside preference tuning.
- [Chen et al., "Evaluating Large Language Models Trained on Code" (2021)](https://arxiv.org/abs/2107.03374): the unbiased pass@k estimator and its stable form.
- [Yue et al., "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?" (2025)](https://arxiv.org/abs/2504.13837): the crossover at large k.
- [Liu et al., "ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models" (2025)](https://arxiv.org/abs/2505.24864): longer, more diverse training that reports gains at large k.
- [Wen et al., "Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs" (2025)](https://arxiv.org/abs/2506.14245): a metric that checks the reasoning as well as the answer.
- [Cui et al., "The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models" (2025)](https://arxiv.org/abs/2505.22617): the covariance law and Clip-Cov.
- [Shao et al., "Spurious Rewards: Rethinking Training Signals in RLVR" (2025)](https://arxiv.org/abs/2506.10947): gains from random rewards, and why they are model-specific.

Open questions that researchers are working on:

- Under what training conditions does RLVR add problems to the set a model can solve at any $k$?
- Which entropy controls keep exploration alive without degrading the output?
- Can verifiable rewards extend to tasks without a single checkable answer?

## The exercise

Write `pass_at_k`, `zero_signal_probability`, and `answer_reward`, then `pg_loss`, `token_entropy`, and `entropy_change_estimate`. Last, write `train_rlvr_toy`, which trains `ToyArithmeticLM` with GRPO advantages and records exact pass@$k$ and batch entropy.

The checks verify `pass_at_k` against exact fractions and an $n = 5000$ case. They check `zero_signal_probability` against hand values, a seeded Monte Carlo estimate, and the endpoints $p = 0$ and $p = 1$, which a `p**G`-only version fails. They run `answer_reward` on nine responses, check `pg_loss`'s gradient in closed form, and compare the entropy estimate with the exact change. Then they train the toy twice. With SGD, pass@1 must rise above $0.45$, entropy must fall below 60% of its start, and pass@64 must fall below the base model's. With Adam, pass@1 must pass $0.9$ and pass@64 must not fall.
