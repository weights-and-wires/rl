---
title: Direct preference optimization
slug: 059-direct-preference-optimization
part: "Part 9 — RL for language models"
summary: Train a policy directly on preference pairs with no reward model, then see why the loss can fall while both answers get less likely.
minutes: 40
gpu: false
objectives:
  - Explain why raising the chosen response and lowering the rejected one needs a reference model to stay safe.
  - Derive the DPO loss from the KL-regularized optimum, and say why the partition function cancels.
  - Compute the DPO margin, loss, and gradient by hand, and implement them on log-probabilities.
  - Say what IPO changes, and why a falling DPO loss does not mean the chosen response became more likely.
  - Compute the SimPO and KTO losses on a tiny batch, and say what each changes.
prereqs:
  - 054-sft-and-the-post-training-pipeline
  - 055-language-models-as-policies
  - 056-reward-models
  - 058-rlhf-with-ppo
exercise: 059-direct-preference-optimization
---

# Direct preference optimization

You have thousands of pairs of replies to the same prompt, and a person marked one reply in each pair as better. RLHF with PPO turns those pairs into a reward model, then samples replies, scores them, and runs reinforcement learning against the score. The policy, the reference, the reward model, and a value head all sit in memory.

The pairs already say which reply to make more likely. Why not push the policy toward the chosen reply and away from the rejected one directly? This chapter shows the one fix that makes this safe, and the result is *direct preference optimization*, or DPO (Rafailov and colleagues, 2023).

You will also see the price. DPO's loss can fall while the probability of the reply people preferred falls too.

## What you already know

- A language model is a policy $\pi(y \mid x)$ over responses $y$ to a prompt $x$, and a response's log-probability is the sum of its per-token log-probabilities, from [Language models as policies](/lessons/055-language-models-as-policies).
- The KL-regularized objective pays for reward and charges $\beta$ times the KL divergence from a reference policy $\pi_{\text{ref}}$. Its best policy has a closed form.
- A Bradley-Terry reward model says the chance that $y_w$ (the chosen response) beats $y_l$ (the rejected one) is $\sigma(r(x, y_w) - r(x, y_l))$, where $\sigma$ is the sigmoid. It sees only reward differences, from [Reward models from preferences](/lessons/056-reward-models).
- RLHF with PPO optimizes that objective by sampling, from [RLHF with PPO](/lessons/058-rlhf-with-ppo).
- A softmax's probabilities sum to one, so raising one lowers the others.

## The obvious approach, and why it fails

The direct loss is $-(\log \pi_\theta(y_w) - \log \pi_\theta(y_l))$: make the chosen response more likely and the rejected one less likely. It fails because nothing holds the policy back. The cheapest way to shrink this loss is to push $\log \pi_\theta(y_l)$ toward minus infinity, and a model that cannot say that sentence has usually lost much else too. PPO-based RLHF avoids this with a KL leash to the reference model. The loss needs the same leash.

So measure each response by how far the policy has moved it from the reference, not by its raw probability. That quantity is the *log-ratio*, $\log \pi_\theta(y \mid x) - \log \pi_{\text{ref}}(y \mid x)$. Positive means the policy likes the response more than the reference did. The *margin* $h$ of a pair is the chosen log-ratio minus the rejected log-ratio:

$$
h_\theta = \Big(\log \pi_\theta(y_w \mid x) - \log \pi_{\text{ref}}(y_w \mid x)\Big) - \Big(\log \pi_\theta(y_l \mid x) - \log \pi_{\text{ref}}(y_l \mid x)\Big)
$$

Read it aloud: how much the policy raised the chosen response, minus how much it raised the rejected one. Training should make $h$ large. At the start $h = 0$.

## A margin by hand

One pair: the policy gives the chosen response log-probability $-10$ and the rejected one $-12$. The reference gives them $-11$ and $-11.5$.

- Chosen log-ratio: $-10 - (-11) = +1$.
- Rejected log-ratio: $-12 - (-11.5) = -0.5$.
- Margin: $h = 1 - (-0.5) = 1.5$.

```figure
<svg viewBox="0 0 600 210" role="img" aria-label="The policy raised the chosen response's log-probability by 1 and lowered the rejected one's by 0.5, so the margin is 1.5">

<defs><marker id="c051-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<line x1="66.0" y1="150" x2="534.0" y2="150" class="stroke-muted"/>
<line x1="170" y1="145" x2="170" y2="155" class="stroke-muted"/><text x="170" y="172" class="fig-small" text-anchor="middle">-1</text>
<line x1="300" y1="145" x2="300" y2="155" class="stroke-muted"/><text x="300" y="172" class="fig-small" text-anchor="middle">0</text>
<line x1="430" y1="145" x2="430" y2="155" class="stroke-muted"/><text x="430" y="172" class="fig-small" text-anchor="middle">+1</text>
<text x="300" y="196" text-anchor="middle">log-ratio: log π(y) − log π_ref(y)</text>
<line x1="300" y1="95" x2="427" y2="95" class="stroke-c" marker-end="url(#c051-arrow)"/>
<text x="288" y="99" text-anchor="end">chosen y_w</text><text x="440" y="99" class="fig-small">+1.0</text>
<line x1="300" y1="125" x2="238.0" y2="125" class="stroke-bad" marker-end="url(#c051-arrow)"/>
<text x="312" y="129" class="fig-small" text-anchor="start">rejected y_l</text><text x="227.0" y="129" class="fig-small" text-anchor="end">−0.5</text>
<line x1="235.0" y1="55" x2="430" y2="55" class="stroke-b"/><line x1="235.0" y1="49" x2="235.0" y2="61" class="stroke-b"/><line x1="430" y1="49" x2="430" y2="61" class="stroke-b"/>
<text x="332.5" y="42" text-anchor="middle" class="fig-bold">margin h = 1.5</text>
</svg>
Figure 1. The policy has moved the chosen response up by 1 and the rejected one down by 0.5 relative to the reference. Notice that the margin adds the two moves, and that the reference is the zero of the axis.
```

## Why this margin is a reward gap

The margin is secretly a reward gap. The derivation has three short steps.

**Step 1: the best policy and the reward determine each other.** From the previous chapters, the policy that maximizes reward minus $\beta$ times KL to the reference is

$$
\pi^\star(y \mid x) = \frac{\pi_{\text{ref}}(y \mid x)\, e^{r(x, y)/\beta}}{Z(x)}, \qquad Z(x) = \sum_{y} \pi_{\text{ref}}(y \mid x)\, e^{r(x, y)/\beta}
$$

Read it aloud: boost the reference's probability of each response by $e^{r/\beta}$, then divide by the total so it sums to one. $Z(x)$ is the *partition function*, the normalizer.

**Step 2: solve for the reward.** Take logs and rearrange:

$$
r(x, y) = \beta \log \frac{\pi^\star(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

The reward is $\beta$ times the log-ratio, plus a number that depends only on the prompt. You cannot compute $Z(x)$ for a language model, because it sums over every possible response.

**Step 3: plug it into Bradley-Terry.** Bradley-Terry needs only $r(x, y_w) - r(x, y_l)$, and both responses answer the same prompt, so $\beta \log Z(x)$ appears in both and cancels. The normalizer is gone, and the preference probability is $\sigma(\beta h)$ with $\pi^\star$ replaced by the policy $\pi_\theta$ you train.

Maximum likelihood minimizes the negative log of that probability:

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\Big[\log \sigma\big(\beta\, h_\theta\big)\Big]
$$

Read it aloud: the reward model's loss, applied to the *implicit reward* $\hat r_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$. The policy is its own reward model.

> [!KEY]
> DPO is the Bradley-Terry loss on the implicit reward $\beta \log(\pi_\theta / \pi_{\text{ref}})$. The partition function cancels because a preference compares two responses to the same prompt.

## The loss and its gradient on the example

With $\beta = 0.1$ and $h = 1.5$:

- The argument is $\beta h = 0.15$.
- The loss is $-\log \sigma(0.15) = \log(1 + e^{-0.15}) = 0.6210$. At the start, $h = 0$ and the loss is $\log 2 = 0.6931$.
- The weight is $\sigma(-0.15) = 0.4626$. It is $0.5$ at the start.

The derivative of $-\log \sigma(u)$ is $-\sigma(-u)$, and $u = \beta h$ moves by $\beta$ for each unit of $\log \pi_\theta(y_w)$. For a batch of $B$ pairs,

$$
\frac{\partial \mathcal{L}}{\partial \log \pi_\theta(y_w)} = -\frac{\beta}{B}\, \sigma(-\beta h), \qquad \frac{\partial \mathcal{L}}{\partial \log \pi_\theta(y_l)} = +\frac{\beta}{B}\, \sigma(-\beta h)
$$

Read it aloud: raise the chosen log-probability and lower the rejected one, each with weight $\sigma(-\beta h)$. The weight is near one for a pair ranked the wrong way and near zero for a pair ranked right by a wide margin, so DPO spends its effort on the pairs it gets wrong. On the example with $B = 1$, the chosen gradient is $-0.1 \times 0.4626 = -0.04626$, and descent moves it up. 

```python
h = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
loss = np.mean(-log_sigmoid(beta * h))        # log_sigmoid(z) = -np.logaddexp(0, -z)
g_chosen = -beta * sigmoid(-beta * h) / B     # and the rejected gradient is -g_chosen
```

## When the loss falls but the chosen response does not

DPO's loss sees only the margin $h$. It is satisfied when the chosen response rises *relative to* the rejected one, and nothing forces the chosen response to rise at all. Razin and colleagues (2025) call a fall in the chosen log-probability *likelihood displacement*.

Responses share parameters. Take a softmax policy over 32 responses with logits $z_y = \theta^\top e_y$, where each response has a unit-length embedding $e_y \in \mathbb{R}^4$. The gradient of $\log \pi(y)$ with respect to $\theta$ is $e_y - \bar e$, where $\bar e = \sum_{y'} \pi(y') e_{y'}$. The two DPO gradients are equal and opposite, so the $\bar e$ terms cancel and each step moves $\theta$ along $e_w - e_l$.

Now make the chosen and rejected embeddings nearly parallel, with cosine similarity $0.95$. A step along $e_w - e_l$ changes each logit in proportion to $e_y^\top(e_w - e_l)$. For the chosen response that is $1 - 0.95 = 0.05$, and for the rejected one it is $-0.05$. Some other response, whose embedding points along $e_w - e_l$, gets $0.258$ in the seeded toy: five times the chosen one. It takes over the normalizer, and the chosen log-probability falls.

The exercise runs 500 steps at $\beta = 0.1$ and step size 1, from a uniform start.

| Step | $\log \pi(y_w)$ | $\log \pi(y_l)$ | DPO loss |
|---|---|---|---|
| 0 | $-3.466$ | $-3.466$ | $0.693$ |
| 100 | $-3.356$ | $-3.850$ | $0.669$ |
| 500 | $-5.688$ | $-8.039$ | $0.583$ |

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Over 500 DPO steps, the chosen and rejected log-probabilities both end lower than they started, while a third response gains">

<line x1="64" y1="232.7" x2="580" y2="232.7" class="stroke-grid"/>
<text x="58" y="236.7" class="fig-small" text-anchor="end">-8</text>
<line x1="64" y1="194.0" x2="580" y2="194.0" class="stroke-grid"/>
<text x="58" y="198.0" class="fig-small" text-anchor="end">-7</text>
<line x1="64" y1="155.3" x2="580" y2="155.3" class="stroke-grid"/>
<text x="58" y="159.3" class="fig-small" text-anchor="end">-6</text>
<line x1="64" y1="116.7" x2="580" y2="116.7" class="stroke-grid"/>
<text x="58" y="120.7" class="fig-small" text-anchor="end">-5</text>
<line x1="64" y1="78.0" x2="580" y2="78.0" class="stroke-grid"/>
<text x="58" y="82.0" class="fig-small" text-anchor="end">-4</text>
<line x1="64" y1="39.3" x2="580" y2="39.3" class="stroke-grid"/>
<text x="58" y="43.3" class="fig-small" text-anchor="end">-3</text>
<line x1="64" y1="252" x2="580" y2="252" class="stroke-muted"/>
<line x1="64.0" y1="252" x2="64.0" y2="257" class="stroke-muted"/>
<text x="64.0" y="272" class="fig-small" text-anchor="middle">0</text>
<line x1="167.2" y1="252" x2="167.2" y2="257" class="stroke-muted"/>
<text x="167.2" y="272" class="fig-small" text-anchor="middle">100</text>
<line x1="270.4" y1="252" x2="270.4" y2="257" class="stroke-muted"/>
<text x="270.4" y="272" class="fig-small" text-anchor="middle">200</text>
<line x1="373.6" y1="252" x2="373.6" y2="257" class="stroke-muted"/>
<text x="373.6" y="272" class="fig-small" text-anchor="middle">300</text>
<line x1="476.8" y1="252" x2="476.8" y2="257" class="stroke-muted"/>
<text x="476.8" y="272" class="fig-small" text-anchor="middle">400</text>
<line x1="580.0" y1="252" x2="580.0" y2="257" class="stroke-muted"/>
<text x="580.0" y="272" class="fig-small" text-anchor="middle">500</text>
<polyline points="64.0,57.3 65.0,57.2 66.1,57.1 67.1,56.9 68.1,56.8 69.2,56.6 70.2,56.5 71.2,56.4 72.3,56.2 73.3,56.1 74.3,56.0 75.4,55.9 76.4,55.7 77.4,55.6 78.4,55.5 79.5,55.4 80.5,55.3 81.5,55.2 82.6,55.1 83.6,55.0 84.6,54.8 85.7,54.7 86.7,54.6 87.7,54.5 88.8,54.5 89.8,54.4 90.8,54.3 91.9,54.2 92.9,54.1 93.9,54.0 95.0,53.9 96.0,53.8 97.0,53.8 98.1,53.7 99.1,53.6 100.1,53.5 101.2,53.5 102.2,53.4 103.2,53.3 104.2,53.3 105.3,53.2 106.3,53.2 107.3,53.1 108.4,53.0 109.4,53.0 110.4,52.9 111.5,52.9 112.5,52.8 113.5,52.8 114.6,52.7 115.6,52.7 116.6,52.7 117.7,52.6 118.7,52.6 119.7,52.6 120.8,52.5 121.8,52.5 122.8,52.5 123.9,52.5 124.9,52.4 125.9,52.4 127.0,52.4 128.0,52.4 129.0,52.4 130.0,52.3 131.1,52.3 132.1,52.3 133.1,52.3 134.2,52.3 135.2,52.3 136.2,52.3 137.3,52.3 138.3,52.3 139.3,52.3 140.4,52.3 141.4,52.3 142.4,52.3 143.5,52.3 144.5,52.4 145.5,52.4 146.6,52.4 147.6,52.4 148.6,52.4 149.7,52.5 150.7,52.5 151.7,52.5 152.8,52.5 153.8,52.6 154.8,52.6 155.8,52.6 156.9,52.7 157.9,52.7 158.9,52.7 160.0,52.8 161.0,52.8 162.0,52.9 163.1,52.9 164.1,53.0 165.1,53.0 166.2,53.1 167.2,53.1 168.2,53.2 169.3,53.2 170.3,53.3 171.3,53.3 172.4,53.4 173.4,53.5 174.4,53.5 175.5,53.6 176.5,53.7 177.5,53.7 178.6,53.8 179.6,53.9 180.6,53.9 181.6,54.0 182.7,54.1 183.7,54.2 184.7,54.3 185.8,54.3 186.8,54.4 187.8,54.5 188.9,54.6 189.9,54.7 190.9,54.8 192.0,54.9 193.0,55.0 194.0,55.0 195.1,55.1 196.1,55.2 197.1,55.3 198.2,55.4 199.2,55.5 200.2,55.6 201.3,55.7 202.3,55.9 203.3,56.0 204.4,56.1 205.4,56.2 206.4,56.3 207.4,56.4 208.5,56.5 209.5,56.6 210.5,56.8 211.6,56.9 212.6,57.0 213.6,57.1 214.7,57.2 215.7,57.4 216.7,57.5 217.8,57.6 218.8,57.7 219.8,57.9 220.9,58.0 221.9,58.1 222.9,58.3 224.0,58.4 225.0,58.5 226.0,58.7 227.1,58.8 228.1,58.9 229.1,59.1 230.2,59.2 231.2,59.4 232.2,59.5 233.2,59.6 234.3,59.8 235.3,59.9 236.3,60.1 237.4,60.2 238.4,60.4 239.4,60.5 240.5,60.7 241.5,60.9 242.5,61.0 243.6,61.2 244.6,61.3 245.6,61.5 246.7,61.6 247.7,61.8 248.7,62.0 249.8,62.1 250.8,62.3 251.8,62.5 252.9,62.6 253.9,62.8 254.9,63.0 256.0,63.1 257.0,63.3 258.0,63.5 259.0,63.6 260.1,63.8 261.1,64.0 262.1,64.2 263.2,64.3 264.2,64.5 265.2,64.7 266.3,64.9 267.3,65.1 268.3,65.2 269.4,65.4 270.4,65.6 271.4,65.8 272.5,66.0 273.5,66.2 274.5,66.3 275.6,66.5 276.6,66.7 277.6,66.9 278.7,67.1 279.7,67.3 280.7,67.5 281.8,67.7 282.8,67.9 283.8,68.1 284.8,68.3 285.9,68.5 286.9,68.6 287.9,68.8 289.0,69.0 290.0,69.2 291.0,69.4 292.1,69.6 293.1,69.8 294.1,70.1 295.2,70.3 296.2,70.5 297.2,70.7 298.3,70.9 299.3,71.1 300.3,71.3 301.4,71.5 302.4,71.7 303.4,71.9 304.5,72.1 305.5,72.3 306.5,72.5 307.6,72.8 308.6,73.0 309.6,73.2 310.6,73.4 311.7,73.6 312.7,73.8 313.7,74.1 314.8,74.3 315.8,74.5 316.8,74.7 317.9,74.9 318.9,75.1 319.9,75.4 321.0,75.6 322.0,75.8 323.0,76.0 324.1,76.3 325.1,76.5 326.1,76.7 327.2,76.9 328.2,77.2 329.2,77.4 330.3,77.6 331.3,77.8 332.3,78.1 333.4,78.3 334.4,78.5 335.4,78.8 336.4,79.0 337.5,79.2 338.5,79.5 339.5,79.7 340.6,79.9 341.6,80.2 342.6,80.4 343.7,80.6 344.7,80.9 345.7,81.1 346.8,81.3 347.8,81.6 348.8,81.8 349.9,82.0 350.9,82.3 351.9,82.5 353.0,82.8 354.0,83.0 355.0,83.2 356.1,83.5 357.1,83.7 358.1,84.0 359.2,84.2 360.2,84.5 361.2,84.7 362.2,84.9 363.3,85.2 364.3,85.4 365.3,85.7 366.4,85.9 367.4,86.2 368.4,86.4 369.5,86.7 370.5,86.9 371.5,87.2 372.6,87.4 373.6,87.7 374.6,87.9 375.7,88.2 376.7,88.4 377.7,88.7 378.8,88.9 379.8,89.2 380.8,89.4 381.9,89.7 382.9,89.9 383.9,90.2 385.0,90.4 386.0,90.7 387.0,91.0 388.0,91.2 389.1,91.5 390.1,91.7 391.1,92.0 392.2,92.2 393.2,92.5 394.2,92.7 395.3,93.0 396.3,93.3 397.3,93.5 398.4,93.8 399.4,94.0 400.4,94.3 401.5,94.6 402.5,94.8 403.5,95.1 404.6,95.3 405.6,95.6 406.6,95.9 407.7,96.1 408.7,96.4 409.7,96.7 410.8,96.9 411.8,97.2 412.8,97.5 413.8,97.7 414.9,98.0 415.9,98.2 416.9,98.5 418.0,98.8 419.0,99.0 420.0,99.3 421.1,99.6 422.1,99.8 423.1,100.1 424.2,100.4 425.2,100.7 426.2,100.9 427.3,101.2 428.3,101.5 429.3,101.7 430.4,102.0 431.4,102.3 432.4,102.5 433.5,102.8 434.5,103.1 435.5,103.3 436.6,103.6 437.6,103.9 438.6,104.2 439.6,104.4 440.7,104.7 441.7,105.0 442.7,105.3 443.8,105.5 444.8,105.8 445.8,106.1 446.9,106.3 447.9,106.6 448.9,106.9 450.0,107.2 451.0,107.4 452.0,107.7 453.1,108.0 454.1,108.3 455.1,108.6 456.2,108.8 457.2,109.1 458.2,109.4 459.3,109.7 460.3,109.9 461.3,110.2 462.4,110.5 463.4,110.8 464.4,111.0 465.4,111.3 466.5,111.6 467.5,111.9 468.5,112.2 469.6,112.4 470.6,112.7 471.6,113.0 472.7,113.3 473.7,113.6 474.7,113.8 475.8,114.1 476.8,114.4 477.8,114.7 478.9,115.0 479.9,115.2 480.9,115.5 482.0,115.8 483.0,116.1 484.0,116.4 485.1,116.7 486.1,116.9 487.1,117.2 488.2,117.5 489.2,117.8 490.2,118.1 491.2,118.4 492.3,118.6 493.3,118.9 494.3,119.2 495.4,119.5 496.4,119.8 497.4,120.1 498.5,120.3 499.5,120.6 500.5,120.9 501.6,121.2 502.6,121.5 503.6,121.8 504.7,122.1 505.7,122.3 506.7,122.6 507.8,122.9 508.8,123.2 509.8,123.5 510.9,123.8 511.9,124.1 512.9,124.3 514.0,124.6 515.0,124.9 516.0,125.2 517.0,125.5 518.1,125.8 519.1,126.1 520.1,126.4 521.2,126.6 522.2,126.9 523.2,127.2 524.3,127.5 525.3,127.8 526.3,128.1 527.4,128.4 528.4,128.7 529.4,129.0 530.5,129.2 531.5,129.5 532.5,129.8 533.6,130.1 534.6,130.4 535.6,130.7 536.7,131.0 537.7,131.3 538.7,131.6 539.8,131.9 540.8,132.1 541.8,132.4 542.8,132.7 543.9,133.0 544.9,133.3 545.9,133.6 547.0,133.9 548.0,134.2 549.0,134.5 550.1,134.8 551.1,135.1 552.1,135.4 553.2,135.6 554.2,135.9 555.2,136.2 556.3,136.5 557.3,136.8 558.3,137.1 559.4,137.4 560.4,137.7 561.4,138.0 562.5,138.3 563.5,138.6 564.5,138.9 565.6,139.2 566.6,139.5 567.6,139.8 568.6,140.0 569.7,140.3 570.7,140.6 571.7,140.9 572.8,141.2 573.8,141.5 574.8,141.8 575.9,142.1 576.9,142.4 577.9,142.7 579.0,143.0 580.0,143.3" class="stroke-c" fill="none"/>
<polyline points="64.0,57.3 65.0,57.4 66.1,57.4 67.1,57.5 68.1,57.5 69.2,57.6 70.2,57.7 71.2,57.7 72.3,57.8 73.3,57.8 74.3,57.9 75.4,58.0 76.4,58.1 77.4,58.1 78.4,58.2 79.5,58.3 80.5,58.4 81.5,58.4 82.6,58.5 83.6,58.6 84.6,58.7 85.7,58.8 86.7,58.9 87.7,59.0 88.8,59.1 89.8,59.2 90.8,59.3 91.9,59.4 92.9,59.5 93.9,59.6 95.0,59.7 96.0,59.8 97.0,59.9 98.1,60.0 99.1,60.2 100.1,60.3 101.2,60.4 102.2,60.5 103.2,60.6 104.2,60.8 105.3,60.9 106.3,61.0 107.3,61.2 108.4,61.3 109.4,61.4 110.4,61.6 111.5,61.7 112.5,61.9 113.5,62.0 114.6,62.2 115.6,62.3 116.6,62.5 117.7,62.6 118.7,62.8 119.7,62.9 120.8,63.1 121.8,63.3 122.8,63.4 123.9,63.6 124.9,63.8 125.9,63.9 127.0,64.1 128.0,64.3 129.0,64.4 130.0,64.6 131.1,64.8 132.1,65.0 133.1,65.2 134.2,65.3 135.2,65.5 136.2,65.7 137.3,65.9 138.3,66.1 139.3,66.3 140.4,66.5 141.4,66.7 142.4,66.9 143.5,67.1 144.5,67.3 145.5,67.5 146.6,67.7 147.6,67.9 148.6,68.1 149.7,68.3 150.7,68.5 151.7,68.8 152.8,69.0 153.8,69.2 154.8,69.4 155.8,69.6 156.9,69.9 157.9,70.1 158.9,70.3 160.0,70.6 161.0,70.8 162.0,71.0 163.1,71.3 164.1,71.5 165.1,71.7 166.2,72.0 167.2,72.2 168.2,72.4 169.3,72.7 170.3,72.9 171.3,73.2 172.4,73.4 173.4,73.7 174.4,73.9 175.5,74.2 176.5,74.5 177.5,74.7 178.6,75.0 179.6,75.2 180.6,75.5 181.6,75.8 182.7,76.0 183.7,76.3 184.7,76.6 185.8,76.8 186.8,77.1 187.8,77.4 188.9,77.6 189.9,77.9 190.9,78.2 192.0,78.5 193.0,78.8 194.0,79.0 195.1,79.3 196.1,79.6 197.1,79.9 198.2,80.2 199.2,80.5 200.2,80.8 201.3,81.0 202.3,81.3 203.3,81.6 204.4,81.9 205.4,82.2 206.4,82.5 207.4,82.8 208.5,83.1 209.5,83.4 210.5,83.7 211.6,84.0 212.6,84.3 213.6,84.6 214.7,85.0 215.7,85.3 216.7,85.6 217.8,85.9 218.8,86.2 219.8,86.5 220.9,86.8 221.9,87.1 222.9,87.5 224.0,87.8 225.0,88.1 226.0,88.4 227.1,88.8 228.1,89.1 229.1,89.4 230.2,89.7 231.2,90.1 232.2,90.4 233.2,90.7 234.3,91.1 235.3,91.4 236.3,91.7 237.4,92.1 238.4,92.4 239.4,92.7 240.5,93.1 241.5,93.4 242.5,93.7 243.6,94.1 244.6,94.4 245.6,94.8 246.7,95.1 247.7,95.5 248.7,95.8 249.8,96.2 250.8,96.5 251.8,96.9 252.9,97.2 253.9,97.6 254.9,97.9 256.0,98.3 257.0,98.6 258.0,99.0 259.0,99.3 260.1,99.7 261.1,100.0 262.1,100.4 263.2,100.8 264.2,101.1 265.2,101.5 266.3,101.9 267.3,102.2 268.3,102.6 269.4,103.0 270.4,103.3 271.4,103.7 272.5,104.1 273.5,104.4 274.5,104.8 275.6,105.2 276.6,105.5 277.6,105.9 278.7,106.3 279.7,106.7 280.7,107.0 281.8,107.4 282.8,107.8 283.8,108.2 284.8,108.5 285.9,108.9 286.9,109.3 287.9,109.7 289.0,110.1 290.0,110.5 291.0,110.8 292.1,111.2 293.1,111.6 294.1,112.0 295.2,112.4 296.2,112.8 297.2,113.2 298.3,113.5 299.3,113.9 300.3,114.3 301.4,114.7 302.4,115.1 303.4,115.5 304.5,115.9 305.5,116.3 306.5,116.7 307.6,117.1 308.6,117.5 309.6,117.9 310.6,118.3 311.7,118.7 312.7,119.1 313.7,119.5 314.8,119.9 315.8,120.3 316.8,120.7 317.9,121.1 318.9,121.5 319.9,121.9 321.0,122.3 322.0,122.7 323.0,123.1 324.1,123.5 325.1,123.9 326.1,124.3 327.2,124.7 328.2,125.1 329.2,125.5 330.3,125.9 331.3,126.3 332.3,126.7 333.4,127.2 334.4,127.6 335.4,128.0 336.4,128.4 337.5,128.8 338.5,129.2 339.5,129.6 340.6,130.0 341.6,130.5 342.6,130.9 343.7,131.3 344.7,131.7 345.7,132.1 346.8,132.5 347.8,133.0 348.8,133.4 349.9,133.8 350.9,134.2 351.9,134.6 353.0,135.1 354.0,135.5 355.0,135.9 356.1,136.3 357.1,136.7 358.1,137.2 359.2,137.6 360.2,138.0 361.2,138.4 362.2,138.9 363.3,139.3 364.3,139.7 365.3,140.1 366.4,140.6 367.4,141.0 368.4,141.4 369.5,141.8 370.5,142.3 371.5,142.7 372.6,143.1 373.6,143.6 374.6,144.0 375.7,144.4 376.7,144.8 377.7,145.3 378.8,145.7 379.8,146.1 380.8,146.6 381.9,147.0 382.9,147.4 383.9,147.9 385.0,148.3 386.0,148.7 387.0,149.2 388.0,149.6 389.1,150.0 390.1,150.5 391.1,150.9 392.2,151.3 393.2,151.8 394.2,152.2 395.3,152.7 396.3,153.1 397.3,153.5 398.4,154.0 399.4,154.4 400.4,154.8 401.5,155.3 402.5,155.7 403.5,156.2 404.6,156.6 405.6,157.0 406.6,157.5 407.7,157.9 408.7,158.4 409.7,158.8 410.8,159.2 411.8,159.7 412.8,160.1 413.8,160.6 414.9,161.0 415.9,161.5 416.9,161.9 418.0,162.3 419.0,162.8 420.0,163.2 421.1,163.7 422.1,164.1 423.1,164.6 424.2,165.0 425.2,165.5 426.2,165.9 427.3,166.3 428.3,166.8 429.3,167.2 430.4,167.7 431.4,168.1 432.4,168.6 433.5,169.0 434.5,169.5 435.5,169.9 436.6,170.4 437.6,170.8 438.6,171.3 439.6,171.7 440.7,172.2 441.7,172.6 442.7,173.1 443.8,173.5 444.8,174.0 445.8,174.4 446.9,174.9 447.9,175.3 448.9,175.8 450.0,176.2 451.0,176.7 452.0,177.1 453.1,177.6 454.1,178.0 455.1,178.5 456.2,178.9 457.2,179.4 458.2,179.8 459.3,180.3 460.3,180.7 461.3,181.2 462.4,181.6 463.4,182.1 464.4,182.5 465.4,183.0 466.5,183.5 467.5,183.9 468.5,184.4 469.6,184.8 470.6,185.3 471.6,185.7 472.7,186.2 473.7,186.6 474.7,187.1 475.8,187.6 476.8,188.0 477.8,188.5 478.9,188.9 479.9,189.4 480.9,189.8 482.0,190.3 483.0,190.7 484.0,191.2 485.1,191.7 486.1,192.1 487.1,192.6 488.2,193.0 489.2,193.5 490.2,193.9 491.2,194.4 492.3,194.9 493.3,195.3 494.3,195.8 495.4,196.2 496.4,196.7 497.4,197.2 498.5,197.6 499.5,198.1 500.5,198.5 501.6,199.0 502.6,199.5 503.6,199.9 504.7,200.4 505.7,200.8 506.7,201.3 507.8,201.7 508.8,202.2 509.8,202.7 510.9,203.1 511.9,203.6 512.9,204.1 514.0,204.5 515.0,205.0 516.0,205.4 517.0,205.9 518.1,206.4 519.1,206.8 520.1,207.3 521.2,207.7 522.2,208.2 523.2,208.7 524.3,209.1 525.3,209.6 526.3,210.0 527.4,210.5 528.4,211.0 529.4,211.4 530.5,211.9 531.5,212.4 532.5,212.8 533.6,213.3 534.6,213.7 535.6,214.2 536.7,214.7 537.7,215.1 538.7,215.6 539.8,216.1 540.8,216.5 541.8,217.0 542.8,217.5 543.9,217.9 544.9,218.4 545.9,218.8 547.0,219.3 548.0,219.8 549.0,220.2 550.1,220.7 551.1,221.2 552.1,221.6 553.2,222.1 554.2,222.6 555.2,223.0 556.3,223.5 557.3,223.9 558.3,224.4 559.4,224.9 560.4,225.3 561.4,225.8 562.5,226.3 563.5,226.7 564.5,227.2 565.6,227.7 566.6,228.1 567.6,228.6 568.6,229.1 569.7,229.5 570.7,230.0 571.7,230.5 572.8,230.9 573.8,231.4 574.8,231.8 575.9,232.3 576.9,232.8 577.9,233.2 579.0,233.7 580.0,234.2" class="stroke-bad" fill="none"/>
<polyline points="64.0,57.3 65.0,56.8 66.1,56.2 67.1,55.7 68.1,55.2 69.2,54.6 70.2,54.1 71.2,53.6 72.3,53.0 73.3,52.5 74.3,52.0 75.4,51.4 76.4,50.9 77.4,50.4 78.4,49.9 79.5,49.4 80.5,48.9 81.5,48.3 82.6,47.8 83.6,47.3 84.6,46.8 85.7,46.3 86.7,45.8 87.7,45.3 88.8,44.8 89.8,44.3 90.8,43.8 91.9,43.4 92.9,42.9 93.9,42.4 95.0,41.9 96.0,41.4 97.0,40.9 98.1,40.5 99.1,40.0 100.1,39.5 101.2,39.1 102.2,38.6 103.2,38.1 104.2,37.7 105.3,37.2 106.3,36.7 107.3,36.3 108.4,35.8 109.4,35.4 110.4,34.9 111.5,34.5 112.5,34.0 113.5,33.6 114.6,33.2 115.6,32.7 116.6,32.3 117.7,31.8 118.7,31.4 119.7,31.0 120.8,30.6 121.8,30.1 122.8,29.7 123.9,29.3 124.9,28.9 125.9,28.4 127.0,28.0 128.0,27.6 129.0,27.2 130.0,26.8 131.1,26.4 132.1,26.0 133.1,25.6 134.2,25.2 135.2,24.8 136.2,24.4 137.3,24.0 138.3,23.6 139.3,23.2 140.4,22.8 141.4,22.4 142.4,22.0 143.5,21.7 144.5,21.3 145.5,20.9 146.6,20.5 147.6,20.1 148.6,19.8 149.7,19.4 150.7,19.0 151.7,18.7 152.8,18.3 153.8,17.9 154.8,17.6 155.8,17.2 156.9,16.9 157.9,16.5 158.9,16.1 160.0,15.8 161.0,15.4 162.0,15.1 163.1,14.7 164.1,14.4 165.1,14.1 166.2,13.7 167.2,13.4 168.2,13.0 169.3,12.7 170.3,12.4 171.3,12.0 172.4,11.7 173.4,11.4 174.4,11.0 175.5,10.7 176.5,10.4 177.5,10.1 178.6,9.8 179.6,9.4 180.6,9.1 181.6,8.8 182.7,8.5 183.7,8.2 184.7,7.9 185.8,7.6 186.8,7.3 187.8,6.9 188.9,6.6 189.9,6.3 190.9,6.0 192.0,5.7 193.0,5.4 194.0,5.1 195.1,4.9 196.1,4.6 197.1,4.3 198.2,4.0 199.2,3.7 200.2,3.4 201.3,3.1 202.3,2.8 203.3,2.6 204.4,2.3 205.4,2.0 206.4,1.7 207.4,1.4 208.5,1.2 209.5,0.9 210.5,0.6 211.6,0.4 212.6,0.1 213.6,-0.2 214.7,-0.5 215.7,-0.7 216.7,-1.0 217.8,-1.2 218.8,-1.5 219.8,-1.8 220.9,-2.0 221.9,-2.3 222.9,-2.5 224.0,-2.8 225.0,-3.0 226.0,-3.3 227.1,-3.5 228.1,-3.8 229.1,-4.0 230.2,-4.3 231.2,-4.5 232.2,-4.8 233.2,-5.0 234.3,-5.2 235.3,-5.5 236.3,-5.7 237.4,-6.0 238.4,-6.2 239.4,-6.4 240.5,-6.7 241.5,-6.9 242.5,-7.1 243.6,-7.3 244.6,-7.6 245.6,-7.8 246.7,-8.0 247.7,-8.2 248.7,-8.5 249.8,-8.7 250.8,-8.9 251.8,-9.1 252.9,-9.3 253.9,-9.6 254.9,-9.8 256.0,-10.0 257.0,-10.2 258.0,-10.4 259.0,-10.6 260.1,-10.8 261.1,-11.1 262.1,-11.3 263.2,-11.5 264.2,-11.7 265.2,-11.9 266.3,-12.1 267.3,-12.3 268.3,-12.5 269.4,-12.7 270.4,-12.9 271.4,-13.1 272.5,-13.3 273.5,-13.5 274.5,-13.7 275.6,-13.9 276.6,-14.1 277.6,-14.3 278.7,-14.5 279.7,-14.6 280.7,-14.8 281.8,-15.0 282.8,-15.2 283.8,-15.4 284.8,-15.6 285.9,-15.8 286.9,-16.0 287.9,-16.1 289.0,-16.3 290.0,-16.5 291.0,-16.7 292.1,-16.9 293.1,-17.0 294.1,-17.2 295.2,-17.4 296.2,-17.6 297.2,-17.7 298.3,-17.9 299.3,-18.1 300.3,-18.3 301.4,-18.4 302.4,-18.6 303.4,-18.8 304.5,-18.9 305.5,-19.1 306.5,-19.3 307.6,-19.4 308.6,-19.6 309.6,-19.8 310.6,-19.9 311.7,-20.1 312.7,-20.3 313.7,-20.4 314.8,-20.6 315.8,-20.7 316.8,-20.9 317.9,-21.1 318.9,-21.2 319.9,-21.4 321.0,-21.5 322.0,-21.7 323.0,-21.8 324.1,-22.0 325.1,-22.2 326.1,-22.3 327.2,-22.5 328.2,-22.6 329.2,-22.8 330.3,-22.9 331.3,-23.1 332.3,-23.2 333.4,-23.4 334.4,-23.5 335.4,-23.7 336.4,-23.8 337.5,-23.9 338.5,-24.1 339.5,-24.2 340.6,-24.4 341.6,-24.5 342.6,-24.7 343.7,-24.8 344.7,-24.9 345.7,-25.1 346.8,-25.2 347.8,-25.4 348.8,-25.5 349.9,-25.6 350.9,-25.8 351.9,-25.9 353.0,-26.0 354.0,-26.2 355.0,-26.3 356.1,-26.4 357.1,-26.6 358.1,-26.7 359.2,-26.8 360.2,-27.0 361.2,-27.1 362.2,-27.2 363.3,-27.4 364.3,-27.5 365.3,-27.6 366.4,-27.7 367.4,-27.9 368.4,-28.0 369.5,-28.1 370.5,-28.2 371.5,-28.4 372.6,-28.5 373.6,-28.6 374.6,-28.7 375.7,-28.9 376.7,-29.0 377.7,-29.1 378.8,-29.2 379.8,-29.4 380.8,-29.5 381.9,-29.6 382.9,-29.7 383.9,-29.8 385.0,-29.9 386.0,-30.1 387.0,-30.2 388.0,-30.3 389.1,-30.4 390.1,-30.5 391.1,-30.6 392.2,-30.8 393.2,-30.9 394.2,-31.0 395.3,-31.1 396.3,-31.2 397.3,-31.3 398.4,-31.4 399.4,-31.5 400.4,-31.7 401.5,-31.8 402.5,-31.9 403.5,-32.0 404.6,-32.1 405.6,-32.2 406.6,-32.3 407.7,-32.4 408.7,-32.5 409.7,-32.6 410.8,-32.7 411.8,-32.8 412.8,-33.0 413.8,-33.1 414.9,-33.2 415.9,-33.3 416.9,-33.4 418.0,-33.5 419.0,-33.6 420.0,-33.7 421.1,-33.8 422.1,-33.9 423.1,-34.0 424.2,-34.1 425.2,-34.2 426.2,-34.3 427.3,-34.4 428.3,-34.5 429.3,-34.6 430.4,-34.7 431.4,-34.8 432.4,-34.9 433.5,-35.0 434.5,-35.1 435.5,-35.2 436.6,-35.3 437.6,-35.4 438.6,-35.5 439.6,-35.6 440.7,-35.7 441.7,-35.7 442.7,-35.8 443.8,-35.9 444.8,-36.0 445.8,-36.1 446.9,-36.2 447.9,-36.3 448.9,-36.4 450.0,-36.5 451.0,-36.6 452.0,-36.7 453.1,-36.8 454.1,-36.9 455.1,-36.9 456.2,-37.0 457.2,-37.1 458.2,-37.2 459.3,-37.3 460.3,-37.4 461.3,-37.5 462.4,-37.6 463.4,-37.7 464.4,-37.7 465.4,-37.8 466.5,-37.9 467.5,-38.0 468.5,-38.1 469.6,-38.2 470.6,-38.3 471.6,-38.3 472.7,-38.4 473.7,-38.5 474.7,-38.6 475.8,-38.7 476.8,-38.8 477.8,-38.8 478.9,-38.9 479.9,-39.0 480.9,-39.1 482.0,-39.2 483.0,-39.3 484.0,-39.3 485.1,-39.4 486.1,-39.5 487.1,-39.6 488.2,-39.7 489.2,-39.7 490.2,-39.8 491.2,-39.9 492.3,-40.0 493.3,-40.1 494.3,-40.1 495.4,-40.2 496.4,-40.3 497.4,-40.4 498.5,-40.4 499.5,-40.5 500.5,-40.6 501.6,-40.7 502.6,-40.8 503.6,-40.8 504.7,-40.9 505.7,-41.0 506.7,-41.1 507.8,-41.1 508.8,-41.2 509.8,-41.3 510.9,-41.4 511.9,-41.4 512.9,-41.5 514.0,-41.6 515.0,-41.7 516.0,-41.7 517.0,-41.8 518.1,-41.9 519.1,-41.9 520.1,-42.0 521.2,-42.1 522.2,-42.2 523.2,-42.2 524.3,-42.3 525.3,-42.4 526.3,-42.4 527.4,-42.5 528.4,-42.6 529.4,-42.7 530.5,-42.7 531.5,-42.8 532.5,-42.9 533.6,-42.9 534.6,-43.0 535.6,-43.1 536.7,-43.1 537.7,-43.2 538.7,-43.3 539.8,-43.3 540.8,-43.4 541.8,-43.5 542.8,-43.5 543.9,-43.6 544.9,-43.7 545.9,-43.8 547.0,-43.8 548.0,-43.9 549.0,-43.9 550.1,-44.0 551.1,-44.1 552.1,-44.1 553.2,-44.2 554.2,-44.3 555.2,-44.3 556.3,-44.4 557.3,-44.5 558.3,-44.5 559.4,-44.6 560.4,-44.7 561.4,-44.7 562.5,-44.8 563.5,-44.9 564.5,-44.9 565.6,-45.0 566.6,-45.0 567.6,-45.1 568.6,-45.2 569.7,-45.2 570.7,-45.3 571.7,-45.4 572.8,-45.4 573.8,-45.5 574.8,-45.5 575.9,-45.6 576.9,-45.7 577.9,-45.7 579.0,-45.8 580.0,-45.8" class="stroke-b dashed" fill="none"/>
<text x="322" y="292" text-anchor="middle">DPO step</text>
<text x="58" y="14" text-anchor="end" class="fig-small">log-probability</text>
<line x1="350" y1="28" x2="380" y2="28" class="stroke-c"/>
<text x="388" y="32" class="fig-small">chosen y_w</text>
<line x1="350" y1="46" x2="380" y2="46" class="stroke-bad"/>
<text x="388" y="50" class="fig-small">rejected y_l</text>
<line x1="350" y1="64" x2="380" y2="64" class="stroke-b dashed"/>
<text x="388" y="68" class="fig-small">most-aligned response</text>
</svg>
Figure 2. The chosen log-probability rises for about 70 steps, then falls for good, while the rejected one falls faster. Notice the dashed curve: by step 500 the most-aligned response holds 45% of the probability, up from 3%.
```

The loss fell from $0.693$ to $0.583$, so by the loss alone, training worked. With orthogonal embeddings the effect is far weaker, because $e_w^\top(e_w - e_l) = 1$.

> [!WARNING]
> A falling DPO loss says the margin grew. It does not say the chosen response became more likely. Log $\log \pi(y_w)$ and $\log \pi(y_l)$ separately.

## IPO: a finite target for the margin

When a preference is deterministic, so that $y_w$ always wins, Bradley-Terry needs an infinite reward gap to fit it. DPO then pushes $h$ upward forever, and the KL leash stops working. Azar and colleagues (2023) fix this with a squared error toward a finite margin, the *identity preference optimization* loss:

$$
\mathcal{L}_{\text{IPO}} = \mathbb{E}\Big[\Big(h_\theta - \frac{1}{2\tau}\Big)^2\Big]
$$

Read it aloud: push the margin toward $1/(2\tau)$, and push back if it overshoots. $\tau$ plays the role of $\beta$. On the example with $\tau = 0.1$, the target is $5$ and the loss is $(1.5 - 5)^2 = 12.25$. The chosen gradient is `2 * (h - 1 / (2 * tau)) / B`.

## SimPO and KTO: two more relatives

SimPO (Meng, Xia, and Chen, 2024) drops the reference model. It scores each response by its average per-token log-probability and asks for a target margin $\gamma$:

$$
\mathcal{L}_{\text{SimPO}} = -\log \sigma\Big(\frac{\beta}{|y_w|}\log \pi_\theta(y_w \mid x) - \frac{\beta}{|y_l|}\log \pi_\theta(y_l \mid x) - \gamma\Big)
$$

Read it aloud: the chosen response's average log-probability must beat the rejected one's by at least $\gamma$. Take sums of $-20$ over 10 tokens and $-36$ over 12, with $\beta = 2$ and $\gamma = 0.5$. The averages are $-2$ and $-3$, so the argument is $2(-2) - 2(-3) - 0.5 = 1.5$, and the loss is $0.2014$. Sums would give an argument of $31.5$, so the division by length matters.

KTO (Ethayarajh and colleagues, 2024) learns from single thumbs-up or thumbs-down labels. With $r = \log(\pi_\theta / \pi_{\text{ref}})$ and a reference point $z_{\text{ref}}$, the loss is $\mathbb{E}[\lambda_y - v(x, y)]$, where

$$
v(x, y) = \begin{cases} \lambda_D\, \sigma\big(\beta (r - z_{\text{ref}})\big) & y \text{ desirable} \\ \lambda_U\, \sigma\big(\beta (z_{\text{ref}} - r)\big) & y \text{ undesirable} \end{cases}
$$

Read it aloud: a good response costs less the further its log-ratio sits above $z_{\text{ref}}$, and a bad one costs less the further it sits below. The paper sets $z_{\text{ref}}$ to the mean log-ratio of responses paired with other prompts, clamped at zero, with no gradient. Take log-ratios $+1$ (desirable) and $-0.5$ (undesirable), with $\beta = 1$, $\lambda_D = 1$, $\lambda_U = 1.5$, and $z_{\text{ref}} = 0.2$. The costs are $0.3100$ and $0.4977$, so the mean loss is $0.4039$.

```python
simpo = np.mean(-log_sigmoid(beta * logp_w / len_w - beta * logp_l / len_l - gamma))
z_ref = max(0.0, np.mean(policy_mismatched - ref_mismatched))
```

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="DPO, SimPO, and KTO losses against the scaled reward u. At u = 0 DPO is 0.693, SimPO with gamma 0.5 is 0.974, and KTO is 0.5. KTO stays below 1 for every u.">
<line x1="80" y1="250" x2="560" y2="250" class="stroke-muted"/>
<line x1="80" y1="250.0" x2="560" y2="250.0" class="stroke-grid"/><text x="72" y="254.0" class="fig-small" text-anchor="end">0</text><line x1="80" y1="188.9" x2="560" y2="188.9" class="stroke-grid"/><text x="72" y="192.9" class="fig-small" text-anchor="end">1</text><line x1="80" y1="127.7" x2="560" y2="127.7" class="stroke-grid"/><text x="72" y="131.7" class="fig-small" text-anchor="end">2</text><line x1="80" y1="66.6" x2="560" y2="66.6" class="stroke-grid"/><text x="72" y="70.6" class="fig-small" text-anchor="end">3</text>
<line x1="148.6" y1="250" x2="148.6" y2="256" class="stroke-muted"/><text x="148.6" y="272" class="fig-small" text-anchor="middle">-2</text><line x1="285.7" y1="250" x2="285.7" y2="256" class="stroke-muted"/><text x="285.7" y="272" class="fig-small" text-anchor="middle">0</text><line x1="422.9" y1="250" x2="422.9" y2="256" class="stroke-muted"/><text x="422.9" y="272" class="fig-small" text-anchor="middle">2</text><line x1="560.0" y1="250" x2="560.0" y2="256" class="stroke-muted"/><text x="560.0" y="272" class="fig-small" text-anchor="middle">4</text>
<polyline points="80.0,34.2 86.9,40.1 93.7,46.0 100.6,51.9 107.4,57.8 114.3,63.6 121.1,69.4 128.0,75.2 134.9,80.9 141.7,86.6 148.6,92.3 155.4,97.9 162.3,103.5 169.1,109.1 176.0,114.5 182.9,120.0 189.7,125.3 196.6,130.6 203.4,135.8 210.3,140.9 217.1,146.0 224.0,150.9 230.9,155.8 237.7,160.5 244.6,165.2 251.4,169.7 258.3,174.1 265.1,178.4 272.0,182.5 278.9,186.6 285.7,190.4 292.6,194.2 299.4,197.8 306.3,201.2 313.1,204.5 320.0,207.6 326.9,210.6 333.7,213.4 340.6,216.1 347.4,218.6 354.3,221.0 361.1,223.3 368.0,225.3 374.9,227.3 381.7,229.1 388.6,230.8 395.4,232.4 402.3,233.9 409.1,235.3 416.0,236.5 422.9,237.7 429.7,238.8 436.6,239.7 443.4,240.6 450.3,241.5 457.1,242.2 464.0,242.9 470.9,243.6 477.7,244.2 484.6,244.7 491.4,245.2 498.3,245.6 505.1,246.0 512.0,246.4 518.9,246.7 525.7,247.0 532.6,247.3 539.4,247.6 546.3,247.8 553.1,248.0 560.0,248.2" class="stroke-c" fill="none"/>
<polyline points="80.0,63.6 86.9,69.4 93.7,75.2 100.6,80.9 107.4,86.6 114.3,92.3 121.1,97.9 128.0,103.5 134.9,109.1 141.7,114.5 148.6,120.0 155.4,125.3 162.3,130.6 169.1,135.8 176.0,140.9 182.9,146.0 189.7,150.9 196.6,155.8 203.4,160.5 210.3,165.2 217.1,169.7 224.0,174.1 230.9,178.4 237.7,182.5 244.6,186.6 251.4,190.4 258.3,194.2 265.1,197.8 272.0,201.2 278.9,204.5 285.7,207.6 292.6,210.6 299.4,213.4 306.3,216.1 313.1,218.6 320.0,221.0 326.9,223.3 333.7,225.3 340.6,227.3 347.4,229.1 354.3,230.8 361.1,232.4 368.0,233.9 374.9,235.3 381.7,236.5 388.6,237.7 395.4,238.8 402.3,239.7 409.1,240.6 416.0,241.5 422.9,242.2 429.7,242.9 436.6,243.6 443.4,244.2 450.3,244.7 457.1,245.2 464.0,245.6 470.9,246.0 477.7,246.4 484.6,246.7 491.4,247.0 498.3,247.3 505.1,247.6 512.0,247.8 518.9,248.0 525.7,248.2 532.6,248.4 539.4,248.5 546.3,248.6 553.1,248.8 560.0,248.9" class="stroke-b" fill="none"/>
<polyline points="80.0,191.8 86.9,192.0 93.7,192.4 100.6,192.7 107.4,193.1 114.3,193.5 121.1,193.9 128.0,194.4 134.9,195.0 141.7,195.5 148.6,196.1 155.4,196.8 162.3,197.5 169.1,198.3 176.0,199.1 182.9,200.0 189.7,201.0 196.6,202.0 203.4,203.0 210.3,204.1 217.1,205.3 224.0,206.5 230.9,207.8 237.7,209.1 244.6,210.5 251.4,211.9 258.3,213.4 265.1,214.9 272.0,216.4 278.9,217.9 285.7,219.4 292.6,221.0 299.4,222.5 306.3,224.0 313.1,225.5 320.0,226.9 326.9,228.3 333.7,229.7 340.6,231.0 347.4,232.3 354.3,233.6 361.1,234.7 368.0,235.8 374.9,236.9 381.7,237.9 388.6,238.8 395.4,239.7 402.3,240.6 409.1,241.3 416.0,242.0 422.9,242.7 429.7,243.3 436.6,243.9 443.4,244.4 450.3,244.9 457.1,245.4 464.0,245.8 470.9,246.1 477.7,246.5 484.6,246.8 491.4,247.1 498.3,247.4 505.1,247.6 512.0,247.8 518.9,248.0 525.7,248.2 532.6,248.4 539.4,248.5 546.3,248.7 553.1,248.8 560.0,248.9" class="stroke-a dashed" fill="none"/>
<circle cx="285.7" cy="207.6" r="4.5" class="fill-b"/>
<circle cx="285.7" cy="190.4" r="4.5" class="fill-c"/>
<circle cx="285.7" cy="219.4" r="4.5" class="fill-a"/>
<line x1="380" y1="60" x2="412" y2="60" class="stroke-c"/>
<text x="418" y="64" class="fig-small">SimPO, γ = 0.5</text>
<line x1="380" y1="80" x2="412" y2="80" class="stroke-b"/>
<text x="418" y="84" class="fig-small">DPO</text>
<line x1="380" y1="100" x2="412" y2="100" class="stroke-a dashed"/>
<text x="418" y="104" class="fig-small">KTO, desirable</text>
<text x="320" y="300" text-anchor="middle">scaled reward u</text>
<text x="78" y="24" class="fig-small" text-anchor="end">loss</text>
</svg>
Figure 3. Three losses against one scaled-reward axis. Notice that SimPO's curve is DPO's shifted right by $\gamma$, and that KTO's loss stays below 1 while DPO's grows without bound.
```

ORPO (Hong, Lee, and Thorne, 2024) also needs no reference model. It adds an odds-ratio penalty on the disfavored response to the supervised loss, so one stage replaces SFT followed by preference training.

> [!KEY]
> DPO trains on preference pairs with no reward model or sampling loop by treating $\beta \log(\pi_\theta / \pi_{\text{ref}})$ as the reward. The policy is its own reward model and the reference is the leash. Because the loss sees only the margin, a falling loss does not prove the chosen response got more likely.

> [!RECAP]
> - Raising the chosen and lowering the rejected response needs a leash, and the reference log-ratio is the leash.
> - Inverting the KL-regularized optimum gives $r = \beta \log(\pi^\star / \pi_{\text{ref}}) + \beta \log Z(x)$, and $\beta \log Z(x)$ cancels in a comparison.
> - The DPO loss is $-\log \sigma(\beta h)$, and its chosen-gradient is $-\beta\,\sigma(-\beta h)/B$.
> - IPO regresses $h$ onto $1/(2\tau)$, so the margin stops growing.
> - The margin can grow while both log-probabilities fall.
> - SimPO averages log-probabilities per token and adds a margin $\gamma$ with no reference model. KTO scores single labels against a clamped reference point $z_{\text{ref}}$.

## Build it step by step

The checks test the functions in this order.

1. **`sequence_logprob(token_logps, mask)`.** Takes `(B, T)` arrays and returns `(B,)`. Select the masked tokens and sum over the last axis. Common mistake: multiplying by the mask, because `0 * -inf` is `nan`. Use `np.where`.
2. **`dpo_loss(...)`.** Takes four `(B,)` arrays, `beta`, and optional `weights`. Build `h`, then take the weighted sum of `-log_sigmoid(beta * h)`. Common mistake: `-np.log(sigmoid(z))`, which returns `inf` for large negative `z`.
3. **`dpo_grad(...)`.** Returns `(grad_chosen, grad_rejected)`. Multiply `-beta * sigmoid(-beta * h)` by the weights. Common mistake: dropping the factor $\beta$ from the chain rule. The finite-difference check catches it.
4. **`ipo_loss` and `ipo_grad`.** The target is `1 / (2 * tau)`. Common mistake: using `1 / tau`.
5. **`train_dpo_linear(E, theta0, chosen, rejected, beta, lr, steps)`.** Keep the reference log-probabilities fixed at their starting values. Each step, call `dpo_grad` on one pair, and form the $\theta$ gradient with `g_w * (E[w] - e_bar) + g_l * (E[l] - e_bar)`. Return all `steps + 1` rows. Common mistake: updating the reference as you go, which makes the margin zero every step.
6. **`simpo_loss(logp_chosen, logp_rejected, len_chosen, len_rejected, beta, gamma)`.** Takes summed log-probabilities and token counts. Divide each by its length before scaling by `beta`. Common mistake: using the sums, which makes the loss depend on length.
7. **`kto_ref_point(...)` and `kto_loss(...)`.** The reference point is the mean mismatched log-ratio, clamped with `max(0, ...)`. Use `z_ref - r` for undesirable examples. Common mistake: leaving `z_ref` unclamped, or using `r - z_ref` for both labels.

## Where this breaks

**Likelihood displacement.** Razin and colleagues link it to chosen and rejected responses with similar hidden representations, and show that filtering such pairs helps. A supervised loss on the chosen responses also holds them up.

**Offline data.** The pairs usually come from other models, so DPO learns about responses its own policy rarely produces. Online DPO samples fresh pairs from the current policy.

**Length.** Preferred responses are often longer, and DPO's summed log-ratio lets length act as a shortcut, so trained models grow verbose.

## Further reading

The following sources cover the derivation and the relatives:

- [Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023)](https://arxiv.org/abs/2305.18290): the derivation this chapter follows.
- [Azar et al., "A General Theoretical Paradigm to Understand Learning from Human Preferences" (2023)](https://arxiv.org/abs/2310.12036): the general family of objectives and IPO.
- [Ethayarajh et al., "KTO: Model Alignment as Prospect Theoretic Optimization" (2024)](https://arxiv.org/abs/2402.01306): learning from unpaired binary labels.
- [Meng, Xia, and Chen, "SimPO: Simple Preference Optimization with a Reference-Free Reward" (2024)](https://arxiv.org/abs/2405.14734): the length-normalized, reference-free loss.
- [Hong, Lee, and Thorne, "ORPO: Monolithic Preference Optimization without Reference Model" (2024)](https://arxiv.org/abs/2403.07691): an odds-ratio penalty added to supervised fine-tuning.
- [Razin et al., "Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization" (2025)](https://arxiv.org/abs/2410.08847): why both log-probabilities fall, and a filter for the pairs that cause it.
- [Stanford CS336, Spring 2026, Lecture 15, "Mid/post-training (SFT/RLHF)"](https://cs336.stanford.edu/): a university lecture on the pipeline that DPO replaces in part.

Open questions that researchers are working on:

- Which properties of a preference dataset predict displacement before training starts?
- When does offline DPO match online RLHF, given equal labels and compute?

## The exercise

You build the losses and the displacement experiment from this chapter. Write `sequence_logprob`, then `dpo_loss` and `dpo_grad`, then `ipo_loss` and `ipo_grad`, then `train_dpo_linear`, which runs DPO on one pair of a softmax policy with shared embeddings. Last, write `simpo_loss`, `kto_ref_point`, and `kto_loss`.

The checks verify each loss against the worked examples and both gradients against finite differences. They train IPO and DPO on a deterministic pair and confirm that only IPO's margin settles. They confirm that `train_dpo_linear` lowers the loss while both log-probabilities fall, and that dissimilar pairs lose less. Two last checks test SimPO's length invariance and margin $\gamma$, and KTO's clamped $z_{\text{ref}}$ and its fall as a desirable log-ratio rises.
