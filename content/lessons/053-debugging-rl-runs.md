---
title: Debugging RL runs
slug: 053-debugging-rl-runs
part: "Part 8 — Rewards, evaluation, and many agents"
summary: Read entropy, KL, explained variance, and gradient norm, overfit one batch, and turn the readings into a diagnosis.
minutes: 40
gpu: false
objectives:
  - Compute explained variance, relative entropy, value-target correlation, and gradient norm, and say what a healthy run shows for each.
  - Explain why explained variance catches a critic that correlation calls perfect.
  - Run the overfit-one-batch test to separate a broken pipeline from a hard problem.
  - Show that normalized advantages make the policy gradient independent of reward scale, and classify a training log with stated thresholds.
prereqs:
  - 036-proximal-policy-optimization
  - 037-ppo-in-practice
  - 052-evaluating-rl
exercise: 053-debugging-rl-runs
---

# Debugging RL runs

You train a PPO agent for two hours, and the return curve stays flat. What do you change? Each guess costs another two hours, and a flat curve fits a dozen different causes, so you cannot tell whether any guess could have helped.

RL fails quietly, because the agent produces its own training data. A broken value function, a policy that stopped exploring, and a reward in the wrong units all look the same from the outside.

This chapter teaches you to read the numbers that tell the causes apart. You compute four of them, run one test that separates "the code is broken" from "the problem is hard", and turn the readings into a diagnosis with thresholds you state before you look at a run.

## What you already know

- The policy gradient loss is $-\hat{\mathbb{E}}[A \log \pi(a \mid s)]$, and a softmax policy turns logits into action probabilities.
- PPO monitors each update with an approximate KL and a clip fraction, from [Proximal policy optimization](/lessons/036-proximal-policy-optimization).
- Advantages are normalized and rewards are scaled for different reasons, from [PPO in practice](/lessons/037-ppo-in-practice).
- A critic is trained to predict value targets, which are the returns the agent collected.
- [Evaluating RL honestly](/lessons/052-evaluating-rl) reports a finished run. This chapter fixes one that is still going.

## Suspect a bug before you tune

Spinning Up advises new researchers to look for bugs first when an agent does not learn, because broken RL code often fails silently and hyperparameters are rarely the real problem. Schulman's talk on deep RL experimentation gives the same advice and adds the habit that makes it practical: watch the policy's entropy, the KL of each update, and the explained variance of the value function, so a failure points at a cause. This chapter turns that habit into code.

The next four sections build those numbers, and the one after builds the test you run when they disagree with your intuition.

## Entropy says whether the policy is still exploring

A policy's *entropy* measures how spread out its action probabilities are. For a categorical policy with $A$ actions, the entropy is the maximum, $\log A$, when every action is equally likely, and 0 when one action has probability 1. Dividing by $\log A$ gives the *relative entropy*, a number between 0 and 1 that means the same thing for a 2-action and a 200-action policy:

$$
H_{\text{rel}} = \frac{-\sum_a \pi(a \mid s) \log \pi(a \mid s)}{\log A}
$$

Read it aloud: how much of the available randomness is left. For $\pi = [0.9, 0.1]$ the entropy is $-(0.9 \ln 0.9 + 0.1 \ln 0.1) = 0.325$ and $\log 2 = 0.693$, so the relative entropy is 0.469. In code, mask zero probabilities, because $0 \cdot \log 0$ is 0 and numpy computes it as NaN:

```python
safe = np.where(p > 0, p, 1.0)          # log(1) = 0, so a zero probability adds nothing
rel = (-(p * np.log(safe)).sum(-1)).mean() / np.log(p.shape[-1])
```

A healthy run starts near 1, falls as the policy commits, and levels off above 0. A policy whose relative entropy falls to nearly 0 early has stopped exploring, and it will keep repeating whatever it happened to try first. Figure 1 shows this on a four-armed bandit with payout probabilities 0.2, 0.4, 0.6, and 0.5. With a learning rate of 0.1 the relative entropy is still 0.57 after 300 updates, and the policy favors the best arm. With a learning rate of 3.0 it drops to 0.153 after 10 updates and below 0.05 at update 54, and it ends at 0.028 on arm 3, which pays 0.5 and not 0.6. The policy is confident and wrong, and it never samples the other arms to learn better.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Relative entropy of a bandit policy over 300 updates with a small and a large learning rate">
<line x1="70.0" y1="190.0" x2="570.0" y2="190.0" class="stroke-grid"/>
<text x="62.0" y="194.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="110.0" x2="570.0" y2="110.0" class="stroke-grid"/>
<text x="62.0" y="114.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="70.0" y1="30.0" x2="570.0" y2="30.0" class="stroke-grid"/>
<text x="62.0" y="34.0" class="fig-small" text-anchor="end">1</text>
<line x1="70.0" y1="182.0" x2="570.0" y2="182.0" class="stroke-bad dashed"/>
<text x="566.0" y="176.0" class="fig-small" text-anchor="end">collapse threshold 0.05</text>
<text x="70.0" y="208.0" class="fig-small" text-anchor="middle">0</text>
<text x="237.2" y="208.0" class="fig-small" text-anchor="middle">100</text>
<text x="404.4" y="208.0" class="fig-small" text-anchor="middle">200</text>
<text x="570.0" y="208.0" class="fig-small" text-anchor="middle">299</text>
<polyline points="70.0,30.0 71.7,30.1 73.3,30.1 75.0,30.1 76.7,30.1 78.4,30.1 80.0,30.0 81.7,30.0 83.4,30.0 85.1,30.0 86.7,30.1 88.4,30.1 90.1,30.1 91.7,30.1 93.4,30.1 95.1,30.1 96.8,30.1 98.4,30.1 100.1,30.1 101.8,30.3 103.4,30.1 105.1,30.1 106.8,30.2 108.5,30.4 110.1,30.9 111.8,30.7 113.5,31.3 115.2,31.3 116.8,31.3 118.5,31.3 120.2,30.9 121.8,30.7 123.5,31.3 125.2,31.3 126.9,31.3 128.5,31.3 130.2,31.3 131.9,32.2 133.5,33.3 135.2,32.7 136.9,33.8 138.6,33.8 140.2,33.8 141.9,33.8 143.6,33.8 145.3,33.3 146.9,32.7 148.6,32.0 150.3,31.6 151.9,32.5 153.6,32.5 155.3,33.6 157.0,35.0 158.6,35.0 160.3,35.0 162.0,35.0 163.6,34.2 165.3,34.2 167.0,34.2 168.7,35.6 170.3,35.6 172.0,35.6 173.7,37.1 175.4,37.1 177.0,37.1 178.7,38.9 180.4,38.9 182.0,37.6 183.7,37.6 185.4,39.3 187.1,41.2 188.7,39.8 190.4,39.8 192.1,38.3 193.7,38.3 195.4,38.3 197.1,37.1 198.8,38.8 200.4,40.7 202.1,42.6 203.8,42.6 205.5,41.1 207.1,41.1 208.8,41.1 210.5,41.1 212.1,39.8 213.8,39.8 215.5,39.8 217.2,38.8 218.8,38.8 220.5,38.8 222.2,38.8 223.8,37.5 225.5,37.5 227.2,36.8 228.9,36.8 230.5,36.5 232.2,37.8 233.9,37.8 235.6,37.8 237.2,37.8 238.9,37.5 240.6,38.9 242.2,38.9 243.9,38.9 245.6,40.4 247.3,39.9 248.9,39.9 250.6,38.4 252.3,38.4 253.9,37.1 255.6,38.4 257.3,38.4 259.0,38.1 260.6,39.5 262.3,38.1 264.0,36.9 265.7,35.7 267.3,35.7 269.0,37.0 270.7,38.5 272.3,38.5 274.0,38.5 275.7,38.5 277.4,40.2 279.0,38.8 280.7,37.7 282.4,37.0 284.0,37.0 285.7,37.0 287.4,37.0 289.1,36.2 290.7,37.5 292.4,36.8 294.1,38.2 295.8,39.8 297.4,39.8 299.1,39.8 300.8,39.8 302.4,39.8 304.1,39.8 305.8,39.8 307.5,39.8 309.1,39.8 310.8,39.8 312.5,39.8 314.1,39.8 315.8,41.5 317.5,43.4 319.2,42.0 320.8,42.0 322.5,43.9 324.2,43.9 325.9,43.9 327.5,43.9 329.2,45.8 330.9,45.8 332.5,45.8 334.2,45.8 335.9,47.9 337.6,47.9 339.2,50.0 340.9,52.2 342.6,52.2 344.2,52.2 345.9,52.2 347.6,54.4 349.3,56.6 350.9,53.8 352.6,53.8 354.3,53.8 356.0,53.8 357.6,53.8 359.3,56.1 361.0,58.3 362.6,60.5 364.3,62.8 366.0,62.8 367.7,65.0 369.3,65.0 371.0,67.2 372.7,63.6 374.3,63.6 376.0,63.6 377.7,63.6 379.4,60.4 381.0,60.4 382.7,57.4 384.4,57.4 386.1,57.4 387.7,57.4 389.4,57.4 391.1,54.5 392.7,54.5 394.4,52.0 396.1,54.2 397.8,54.2 399.4,51.7 401.1,49.6 402.8,51.7 404.4,49.7 406.1,49.7 407.8,51.7 409.5,51.7 411.1,53.9 412.8,53.9 414.5,56.0 416.2,56.0 417.8,53.7 419.5,55.8 421.2,58.0 422.8,58.0 424.5,58.0 426.2,60.1 427.9,60.1 429.5,62.3 431.2,62.3 432.9,62.3 434.5,62.3 436.2,62.3 437.9,64.5 439.6,64.5 441.2,61.3 442.9,63.5 444.6,63.5 446.3,63.5 447.9,60.4 449.6,60.4 451.3,62.6 452.9,64.7 454.6,64.7 456.3,64.7 458.0,66.9 459.6,69.1 461.3,71.2 463.0,71.2 464.6,67.5 466.3,67.5 468.0,67.5 469.7,69.7 471.3,71.8 473.0,71.8 474.7,71.8 476.4,71.8 478.0,73.9 479.7,73.9 481.4,73.9 483.0,73.9 484.7,76.0 486.4,76.0 488.1,78.0 489.7,78.0 491.4,78.0 493.1,79.9 494.7,79.9 496.4,75.8 498.1,75.8 499.8,71.7 501.4,67.9 503.1,70.1 504.8,72.2 506.5,74.3 508.1,70.3 509.8,72.5 511.5,72.5 513.1,74.5 514.8,74.5 516.5,74.5 518.2,76.6 519.8,76.6 521.5,76.6 523.2,78.6 524.8,78.6 526.5,80.5 528.2,82.5 529.9,82.5 531.5,78.1 533.2,78.1 534.9,78.1 536.6,80.1 538.2,82.0 539.9,83.9 541.6,83.9 543.2,85.7 544.9,85.7 546.6,87.5 548.3,89.3 549.9,89.3 551.6,89.3 553.3,91.0 554.9,92.6 556.6,92.6 558.3,94.2 560.0,94.2 561.6,94.2 563.3,95.8 565.0,95.8 566.7,97.3 568.3,97.3 570.0,98.8" class="stroke-a" fill="none"/>
<polyline points="70.0,30.0 71.7,128.9 73.3,147.0 75.0,147.0 76.7,156.1 78.4,156.1 80.0,161.7 81.7,161.7 83.4,161.7 85.1,161.7 86.7,165.6 88.4,165.6 90.1,165.6 91.7,165.6 93.4,165.6 95.1,168.4 96.8,168.4 98.4,168.4 100.1,168.4 101.8,168.4 103.4,168.4 105.1,168.4 106.8,168.4 108.5,170.7 110.1,172.4 111.8,173.9 113.5,173.9 115.2,173.9 116.8,173.9 118.5,173.9 120.2,175.1 121.8,176.1 123.5,177.0 125.2,177.0 126.9,177.0 128.5,177.0 130.2,177.0 131.9,177.0 133.5,177.0 135.2,177.7 136.9,177.7 138.6,178.4 140.2,178.4 141.9,179.0 143.6,179.0 145.3,179.6 146.9,179.6 148.6,180.0 150.3,180.5 151.9,180.9 153.6,181.2 155.3,181.6 157.0,181.9 158.6,181.9 160.3,182.2 162.0,182.2 163.6,182.4 165.3,182.4 167.0,182.4 168.7,182.7 170.3,182.7 172.0,182.9 173.7,183.1 175.4,183.1 177.0,183.1 178.7,183.1 180.4,183.1 182.0,183.3 183.7,183.3 185.4,183.5 187.1,183.7 188.7,183.8 190.4,183.8 192.1,184.0 193.7,184.0 195.4,184.0 197.1,184.2 198.8,184.3 200.4,184.4 202.1,184.4 203.8,184.4 205.5,184.6 207.1,184.7 208.8,184.7 210.5,184.7 212.1,184.8 213.8,184.8 215.5,184.8 217.2,184.9 218.8,184.9 220.5,184.9 222.2,184.9 223.8,185.0 225.5,185.0 227.2,185.1 228.9,185.1 230.5,185.2 232.2,185.3 233.9,185.3 235.6,185.3 237.2,185.3 238.9,185.4 240.6,185.5 242.2,185.5 243.9,185.5 245.6,185.6 247.3,185.6 248.9,185.6 250.6,185.7 252.3,185.7 253.9,185.8 255.6,185.8 257.3,185.8 259.0,185.9 260.6,185.9 262.3,185.9 264.0,186.0 265.7,186.1 267.3,186.1 269.0,186.1 270.7,186.1 272.3,186.1 274.0,186.1 275.7,186.1 277.4,186.2 279.0,186.2 280.7,186.3 282.4,186.3 284.0,186.3 285.7,186.3 287.4,186.3 289.1,186.4 290.7,186.5 292.4,186.5 294.1,186.5 295.8,186.6 297.4,186.6 299.1,186.6 300.8,186.6 302.4,186.6 304.1,186.6 305.8,186.6 307.5,186.6 309.1,186.6 310.8,186.6 312.5,186.6 314.1,186.7 315.8,186.7 317.5,186.8 319.2,186.8 320.8,186.8 322.5,186.8 324.2,186.8 325.9,186.8 327.5,186.8 329.2,186.9 330.9,186.9 332.5,186.9 334.2,186.9 335.9,186.9 337.6,186.9 339.2,187.0 340.9,187.0 342.6,187.0 344.2,187.0 345.9,187.0 347.6,187.0 349.3,187.1 350.9,187.1 352.6,187.1 354.3,187.1 356.0,187.1 357.6,187.1 359.3,187.1 361.0,187.2 362.6,187.2 364.3,187.2 366.0,187.2 367.7,187.2 369.3,187.2 371.0,187.3 372.7,187.3 374.3,187.3 376.0,187.3 377.7,187.3 379.4,187.3 381.0,187.3 382.7,187.3 384.4,187.3 386.1,187.3 387.7,187.3 389.4,187.3 391.1,187.4 392.7,187.4 394.4,187.4 396.1,187.4 397.8,187.4 399.4,187.4 401.1,187.5 402.8,187.5 404.4,109.0 406.1,109.0 407.8,152.9 409.5,152.9 411.1,161.7 412.8,166.6 414.5,169.9 416.2,169.9 417.8,169.9 419.5,172.3 421.2,172.3 422.8,172.3 424.5,172.3 426.2,174.1 427.9,174.1 429.5,175.5 431.2,175.5 432.9,175.5 434.5,175.5 436.2,175.5 437.9,175.5 439.6,175.5 441.2,176.7 442.9,177.6 444.6,177.6 446.3,177.6 447.9,127.9 449.6,127.9 451.3,154.5 452.9,162.9 454.6,162.9 456.3,162.9 458.0,167.6 459.6,170.8 461.3,173.0 463.0,173.0 464.6,174.7 466.3,174.7 468.0,174.7 469.7,174.7 471.3,176.1 473.0,176.1 474.7,176.1 476.4,176.1 478.0,176.1 479.7,176.1 481.4,176.1 483.0,176.1 484.7,176.1 486.4,176.1 488.1,177.2 489.7,177.2 491.4,178.1 493.1,178.9 494.7,178.9 496.4,179.6 498.1,180.2 499.8,180.7 501.4,181.2 503.1,181.6 504.8,181.9 506.5,182.3 508.1,182.6 509.8,182.9 511.5,182.9 513.1,183.1 514.8,183.1 516.5,183.1 518.2,183.4 519.8,183.4 521.5,183.4 523.2,183.6 524.8,183.6 526.5,183.6 528.2,183.8 529.9,183.8 531.5,184.0 533.2,184.0 534.9,184.0 536.6,184.1 538.2,184.3 539.9,184.5 541.6,184.5 543.2,184.6 544.9,184.6 546.6,184.8 548.3,184.9 549.9,184.9 551.6,184.9 553.3,185.0 554.9,185.1 556.6,185.1 558.3,185.2 560.0,185.2 561.6,185.2 563.3,185.3 565.0,185.3 566.7,185.4 568.3,185.4 570.0,185.5" class="stroke-b" fill="none"/>
<text x="320.8" y="34.0" class="fig-small fig-bold" text-anchor="middle">learning rate 0.1</text>
<text x="136.9" y="161.2" class="fig-small fig-bold" text-anchor="start">learning rate 3.0</text>
<text x="320.0" y="230.0" class="fig-small" text-anchor="middle">policy update</text>
<text x="70.0" y="20" class="fig-small" text-anchor="start">relative entropy of the policy</text>
</svg>
Figure 1. Relative entropy of a softmax policy on a four-armed bandit. Notice that the large learning rate crosses the 0.05 line at update 54 and settles on the second-best arm, while the small one is still exploring after 300 updates.
```

> [!KEY]
> Falling entropy is normal. Entropy that reaches nearly zero early is a policy that stopped learning.

## KL says how far each update moved

The KL between the policy that collected a batch and the policy after the update measures the step size in the only unit that matters, which is how much the behavior changed. [Proximal policy optimization](/lessons/036-proximal-policy-optimization) builds the k3 estimate and the clip fraction, so this chapter reuses `approx_kl` and does not rederive it.

Over a whole run, a small positive KL means gentle updates, and a KL that grows steadily means stale data or too large a step. A true KL is never negative, so a negative reading means a bug in your log-probabilities.

## Explained variance says whether the critic learned anything

The critic predicts value targets $y$, and its predictions are $\hat y$. A good critic removes most of the variation in $y$. *Explained variance* measures how much:

$$
\text{EV} = 1 - \frac{\operatorname{Var}(y - \hat y)}{\operatorname{Var}(y)}
$$

Read it aloud: take the variance the targets had, subtract the variance left in the errors, and report the share removed. Work it by hand with targets $y = [0, 2, 4, 6]$, whose mean is 3 and whose variance is 5:

| Prediction $\hat y$ | Residual $y - \hat y$ | Residual variance | EV |
|---|---|---|---|
| $[0, 2, 4, 6]$ (perfect) | $[0, 0, 0, 0]$ | 0 | 1 |
| $[3, 3, 3, 3]$ (a constant) | $[-3, -1, 1, 3]$ | 5 | 0 |
| $[1, 2, 3, 4]$ | $[-1, 0, 1, 2]$ | 1.25 | 0.75 |
| $[0, -2, -4, -6]$ (anti-correlated) | $[0, 4, 8, 12]$ | 20 | -3 |

A perfect critic scores 1, and a critic that outputs the same number everywhere scores 0, whatever that number is, because subtracting a constant from the errors does not change their variance. A critic that is worse than a constant scores below 0. When all the targets are equal, $\operatorname{Var}(y) = 0$ and the ratio is undefined, so the function returns NaN, and a caller has to decide what that means.

> [!WARNING]
> The formula $1 - \operatorname{Var}(\hat y)/\operatorname{Var}(y)$ scores a perfect critic 0 and a constant critic 1, which is backwards. Swapping the arguments divides by the wrong variance.

Correlation is the number people reach for first, but it ignores scale and offset. The prediction $3y$ has a correlation of exactly 1 with $y$ and an explained variance of $-3$. The critic's absolute values feed the advantages, so scale matters, and explained variance sees it.

A healthy critic starts near 0 and rises. One that stays near 0 learns nothing.

## Gradient norm says whether the update is the right size

The *global gradient norm* treats every parameter's gradient as one long vector and takes its length, $\sqrt{\sum_i \lVert g_i \rVert^2}$. A norm that sits at the clip limit means the clip decides every step. A spike to ten times the usual value points at a bad batch, and a norm near zero means the gradient is not reaching the parameters.

```python
total = np.sqrt(sum(np.sum(g ** 2) for g in grads))   # one number for the whole network
```

## Overfit one batch, then trust the rest

Instruments show symptoms, and a test separates the two explanations for a flat curve: the code is broken, or the problem is hard. Take one small fixed batch of inputs and targets, and train the network on only that batch until the loss stops moving. A correct network, loss, and optimizer drive the loss close to zero, because a network with thousands of parameters can memorize sixteen examples. If the loss stalls, the bug is in the machinery, such as a detached tensor, a dead activation, a missing optimizer step, or a wrong sign, and no amount of data or tuning will fix it.

In the exercise, a 32-unit network drives the loss from 0.49 to about 0.000001 in 300 Adam steps. A network whose ReLUs are all off can only fit the mean of the targets, so its loss stays at their variance.

## Reward scale changes every number above

Returns are in the environment's reward units. Multiply every reward by 1000 and the value targets grow 1000 times, and the policy gradient does too, though nothing about the task changed.

Advantage normalization removes this dependence for the policy. Subtract the batch mean and divide by the batch standard deviation, and multiplying the advantages by $c > 0$ changes nothing, because $c$ cancels from the numerator and the denominator. The gradient of the loss $L = -\tfrac{1}{N} \sum_i A_i \log \pi(a_i \mid s_i)$ with respect to the logits $z_{ij}$ of a softmax policy is

$$
\frac{\partial L}{\partial z_{ij}} = -\frac{A_i}{N}\big(\mathbb{1}[j = a_i] - \pi_{ij}\big)
$$

Read it aloud: each sample pushes up its chosen action and pushes down the others, in proportion to its advantage. Treat $A_i$ as a number the optimizer cannot change, so the formula is linear in $A$. Without normalization, scaling $A$ by $c$ scales the gradient by $c$. Figure 2 shows both on one fixed batch of 64 samples. The unnormalized gradient norm goes from 0.0013 at scale 0.01 to 125.49 at scale 1000, a factor of $10^5$, and the normalized one stays at 0.1085 at every scale.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Gradient norm against reward scale, with and without advantage normalization">
<line x1="80.0" y1="200.0" x2="560.0" y2="200.0" class="stroke-grid"/>
<text x="72.0" y="204.0" class="fig-small" text-anchor="end">0.001</text>
<line x1="80.0" y1="171.7" x2="560.0" y2="171.7" class="stroke-grid"/>
<text x="72.0" y="175.7" class="fig-small" text-anchor="end">0.01</text>
<line x1="80.0" y1="143.3" x2="560.0" y2="143.3" class="stroke-grid"/>
<text x="72.0" y="147.3" class="fig-small" text-anchor="end">0.1</text>
<line x1="80.0" y1="115.0" x2="560.0" y2="115.0" class="stroke-grid"/>
<text x="72.0" y="119.0" class="fig-small" text-anchor="end">1</text>
<line x1="80.0" y1="86.7" x2="560.0" y2="86.7" class="stroke-grid"/>
<text x="72.0" y="90.7" class="fig-small" text-anchor="end">10</text>
<line x1="80.0" y1="58.3" x2="560.0" y2="58.3" class="stroke-grid"/>
<text x="72.0" y="62.3" class="fig-small" text-anchor="end">100</text>
<line x1="80.0" y1="30.0" x2="560.0" y2="30.0" class="stroke-grid"/>
<text x="72.0" y="34.0" class="fig-small" text-anchor="end">1000</text>
<text x="80.0" y="218.0" class="fig-small" text-anchor="middle">0.01</text>
<text x="176.0" y="218.0" class="fig-small" text-anchor="middle">0.1</text>
<text x="272.0" y="218.0" class="fig-small" text-anchor="middle">1</text>
<text x="368.0" y="218.0" class="fig-small" text-anchor="middle">10</text>
<text x="464.0" y="218.0" class="fig-small" text-anchor="middle">100</text>
<text x="560.0" y="218.0" class="fig-small" text-anchor="middle">1000</text>
<polyline points="80.0,197.2 176.0,168.9 272.0,140.5 368.0,112.2 464.0,83.9 560.0,55.5" class="stroke-b" fill="none"/>
<circle cx="80.0" cy="197.2" r="4" class="fill-b"/>
<circle cx="176.0" cy="168.9" r="4" class="fill-b"/>
<circle cx="272.0" cy="140.5" r="4" class="fill-b"/>
<circle cx="368.0" cy="112.2" r="4" class="fill-b"/>
<circle cx="464.0" cy="83.9" r="4" class="fill-b"/>
<circle cx="560.0" cy="55.5" r="4" class="fill-b"/>
<polyline points="80.0,142.3 176.0,142.3 272.0,142.3 368.0,142.3 464.0,142.3 560.0,142.3" class="stroke-a" fill="none"/>
<circle cx="80.0" cy="142.3" r="4" class="fill-a"/>
<circle cx="176.0" cy="142.3" r="4" class="fill-a"/>
<circle cx="272.0" cy="142.3" r="4" class="fill-a"/>
<circle cx="368.0" cy="142.3" r="4" class="fill-a"/>
<circle cx="464.0" cy="142.3" r="4" class="fill-a"/>
<circle cx="560.0" cy="142.3" r="4" class="fill-a"/>
<text x="147.1" y="156.9" class="fig-small fig-bold" text-anchor="start">raw advantages</text>
<text x="147.1" y="132.3" class="fig-small fig-bold" text-anchor="start">normalized advantages</text>
<text x="320.0" y="242.0" class="fig-small" text-anchor="middle">reward scale (advantages multiplied by this)</text>
<text x="80.0" y="20" class="fig-small" text-anchor="start">gradient norm of the policy loss</text>
</svg>
Figure 2. Gradient norm of the policy loss on one batch as the reward scale grows. Notice that the raw norm rises by a factor of 10^5 across the range while the normalized norm stays at 0.1085.
```

This matters for clipping: with a gradient norm of 125 and a clip at 0.5, the clip rescales every update. A run whose value targets have a standard deviation of 5000 has a reward-scale problem, whatever its entropy and KL say.

> [!KEY]
> Advantage normalization makes the policy gradient independent of the reward's units. The critic's targets are not normalized by it, so the target scale is its own instrument.

## Turn the readings into a diagnosis

Four instruments become signatures. Each rule reads the mean of the last quarter of the log, so an early transient does not trigger it. The rules run in the order below, so a root cause outranks its symptoms: reward scale outranks a dead critic because a critic that cannot fit targets of 5000 also has an explained variance near 0.

| Order | Diagnosis | Fires when the last-quarter mean of | Likely cause |
|---|---|---|---|
| 1 | `exploding_kl` | `approx_kl` is above 0.05 | Learning rate too high, or too many epochs on stale data |
| 2 | `entropy_collapse` | `rel_entropy` is below 0.05 | Learning rate too high, no entropy bonus, or a rewarded accident |
| 3 | `reward_scale` | `target_std` is above 100 or below 0.01 | Rewards in the wrong units, and no scaling |
| 4 | `dead_critic` | `ev` is below 0.05 | Critic learning rate, a bug in the targets, or a detached critic |
| 5 | `healthy` | none of the above | Look at the return curve, then the next experiment |

These thresholds are this chapter's rules of thumb for toy logs, and no paper sets them. Set yours from your own healthy runs, and decide them before you look at the curve, so you do not read a cause into noise.

> [!RECAP]
> - Relative entropy near 0 early means the policy stopped exploring, and the KL tells you the size of each step.
> - Explained variance is $1 - \operatorname{Var}(y - \hat y)/\operatorname{Var}(y)$: 1 for perfect, 0 for any constant, negative for worse than a constant.
> - Correlation ignores scale, so explained variance catches critics that correlation calls perfect.
> - Overfit one fixed batch to separate a broken pipeline from a hard problem.
> - Normalized advantages make the policy gradient independent of reward scale, and unnormalized ones scale with it.

## Build it step by step

The checks test the functions in this order.

1. **`explained_variance(y_pred, y)`.** Return NaN when `np.var(y)` is 0. The common mistake is `1 - np.var(y_pred) / np.var(y)`, which the constant-prediction check catches.
2. **`value_target_correlation(y_pred, y)`.** Use `np.corrcoef`, and return NaN for a constant input.
3. **`relative_entropy(probs)`.** Mask zero probabilities before the logarithm, then divide by `np.log(A)`.
4. **`grad_norm(grads)`.** Sum the squares of every array, then take one square root. The common mistake is the mean of per-array norms.
5. **`policy_loss(...)` and `policy_loss_grad(...)`.** Use the formula above, and do not differentiate through the normalization.
6. **`overfit_one_batch(net, x, y, steps, lr)`.** Run Adam on the one fixed batch and return the loss before and after. The common mistake is drawing a new batch each step.
7. **`diagnose(log)`.** Average the last `max(1, T // 4)` entries of each series and apply the rules in the table's order.

## Where this breaks

**Healthy numbers do not prove a healthy run.** A policy that exploits a flaw in the reward can look fine on every instrument. Watch the agent too, and read [Reward hacking and safe RL](/lessons/070-reward-hacking-and-safe-rl).

**Thresholds depend on the algorithm.** A KL of 0.05 is high for PPO with early stopping. Compute thresholds from healthy runs of the same code.

**Normalization has a floor.** When the advantages' standard deviation falls to the `1e-8` in the denominator, the normalized gradient shrinks with them.

**One run is one sample.** Whether a fix helped needs several seeds and an interval, which [Evaluating RL honestly](/lessons/052-evaluating-rl) builds.

## Further reading

- [Schulman (2017), Nuts and Bolts of Deep RL Experimentation](https://youtu.be/8EcdaCk9KaQ). Core Lecture 6 of the Deep RL Bootcamp, held in Berkeley in August 2017, on what to log and how to debug a policy-gradient run.
- [OpenAI, Spinning Up as a Deep RL Researcher](https://spinningup.openai.com/en/latest/spinningup/spinningup.html). Assume a bug before tuning, start on toy tasks, and log returns, values, losses, and policy entropy.
- [Jones (2021), Debugging RL, Without the Agonizing Pain](https://andyljones.com/posts/rl-debugging.html). The source of relative policy entropy, KL, and residual variance as instruments.
- [Henderson, Islam, Bachman, Pineau, Precup, and Meger (2018), Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560). Shows how seeds, hyperparameters, and implementations change results.
- [UC Berkeley CS285, Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/) has Lecture 5, "Policy Gradients", Lecture 6, "Actor Critic", and the homeworks "HW2: Policy Gradients" and "HW3: Q-Learning and Actor Critic", where you debug these algorithms.

## The exercise

You build a small debugging toolkit: `explained_variance`, `value_target_correlation`, `relative_entropy`, and `grad_norm` for the instruments, `policy_loss` and `policy_loss_grad` for the scale test, `overfit_one_batch` for the pipeline test, and `diagnose` to classify a log.

The checks reproduce the worked table (1, 0, 0.75, and $-3$), and show that the $3y$ prediction has correlation 1 and explained variance $-3$. The gradient matches finite differences, the normalized gradient is unchanged from reward scale 0.01 to $10^4$, and the unnormalized norm grows in proportion. A healthy network overfits one batch to under 1% of its starting loss, and a dead one does not. Last, `diagnose` must name six injected failure logs on five seeds each, honor every threshold from just below to just above, and ignore a critic that was bad early and recovered.
