---
title: Advantage-weighted policy improvement
slug: 054-advantage-weighted-policy-improvement
part: "Part 7 — Models, exploration, and fixed data"
summary: Improve a policy by reweighting its own actions with exp(A / beta), then fit it by weighted likelihood in AWR, MPO, and AWAC.
minutes: 40
gpu: false
objectives:
  - "Derive the KL-penalized improvement step and its solution, pi_old * exp(A / beta) / Z."
  - Compute AWR's clipped weights, and the weighted log-likelihood loss with its gradient, for discrete and continuous actions.
  - Choose MPO's temperature from the dual so the reweighted samples sit a fixed KL from uniform, then fit a Gaussian by weighted maximum likelihood.
  - Run AWAC on a fixed dataset, and see it beat behavior cloning.
  - Connect the weights to the Boltzmann policy, the KL-regularized optimum for language models, and reward-weighted regression.
prereqs:
  - 033-baselines-and-actor-critic
  - 041-maximum-entropy-rl
  - 053-offline-rl
exercise: 054-advantage-weighted-policy-improvement
---

# Advantage-weighted policy improvement

You have a policy and an estimate of how much better or worse each of its actions is than its average. You want the next policy to do the better actions more often. How far should it move? Jumping to the action with the largest estimate bets everything on a noisy number, and moving too little makes learning crawl.

Trust-region methods such as TRPO and PPO cap how far the action probabilities move (see [Trust regions and TRPO](/lessons/035-trust-regions-and-trpo)). This chapter takes a different route. It writes down the best policy within a fixed distance of the current one, and the answer is a short formula: multiply each action's probability by an exponential of its advantage. Then it turns that formula into supervised learning, a weighted copy of the actions you already took.

The same step appears under several names: advantage-weighted regression (AWR), maximum a posteriori policy optimization (MPO), advantage-weighted actor-critic (AWAC), and the policy-extraction step of IQL.

## What you already know

- The advantage $A(s, a) = Q(s, a) - V(s)$ is how much better an action is than the policy's average (see [Baselines and actor-critic](/lessons/033-baselines-and-actor-critic)).
- Maximum-entropy RL has the Boltzmann policy $\pi(a \mid s) \propto \exp(Q(s, a) / \alpha)$ (see [Maximum-entropy reinforcement learning](/lessons/041-maximum-entropy-rl) and [Soft actor-critic](/lessons/042-soft-actor-critic)).
- A language model's KL-regularized optimum is $\pi_{\text{ref}}\, e^{r/\beta} / Z$ (see [Language models as policies](/lessons/066-language-models-as-policies)).
- IQL extracts its policy by weighting the log-likelihood of dataset actions with advantages (see [Offline reinforcement learning](/lessons/053-offline-rl)). That chapter builds the critic, not the extraction.
- Behavior cloning fits logged actions by maximum likelihood (see [Imitation learning](/lessons/051-imitation-learning)).

## Reweight the policy you have

Take one state with three actions, a policy $\pi_{\text{old}} = (0.5, 0.3, 0.2)$, and advantages $A = (1, 0, 2)$. These are the reward example from the language-model chapter, with the advantage as the reward. The greedy move puts all the probability on action 2, a large jump based on one estimate.

The fix is a leash. Score a candidate policy $\pi$ by its expected advantage, and subtract $\beta$ times how far it sits from the old policy:

$$
J(\pi) = \sum_a \pi(a)\, A(a) - \beta\, \mathrm{KL}\big(\pi \,\Vert\, \pi_{\text{old}}\big)
$$

Read it aloud: reward the policy for choosing good actions, and charge it for leaving the old policy. The *temperature* $\beta > 0$ is the price of leaving. A large $\beta$ keeps the policy in place, and a small $\beta$ lets it move. The best policy under this score is

$$
\pi^*(a) = \frac{1}{Z}\, \pi_{\text{old}}(a)\, \exp\!\Big(\frac{A(a)}{\beta}\Big), \qquad Z = \sum_a \pi_{\text{old}}(a)\, \exp\!\Big(\frac{A(a)}{\beta}\Big)
$$

Read it aloud: multiply each action's old probability by $e^{A/\beta}$, then divide by the total so the result sums to one. Here is why it is the best. Expand $\mathrm{KL}(\pi \,\Vert\, \pi^*)$ using this formula and collect terms. You get $J(\pi) = \beta \log Z - \beta\, \mathrm{KL}(\pi \,\Vert\, \pi^*)$. The first term does not depend on $\pi$, and the KL is never negative and is zero only at $\pi = \pi^*$. So $\pi^*$ wins, and its score is $\beta \log Z$.

Work the example with $\beta = 1$. The boosted weights are $0.5\,e^{1} = 1.3591$, $0.3\,e^{0} = 0.3$, and $0.2\,e^{2} = 1.4778$. They sum to $Z = 3.1370$, so $\pi^* = (0.4333, 0.0956, 0.4711)$. The expected advantage rises from $0.9$ to $1.3755$ at a KL of $0.2322$, so the score is $1.1433 = \log 3.1370$. At $\beta = 0.5$, $\pi^* = (0.2477, 0.0201, 0.7322)$. Figure 1 draws both.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Probabilities of three actions under the old policy and under the improved policy at beta 1 and beta 0.5">
<line x1="70.0" y1="230.0" x2="560" y2="230.0" class="stroke-grid"/>
<text x="62.0" y="234.0" class="fig-small" text-anchor="end">0.0</text>
<line x1="70.0" y1="187.5" x2="560" y2="187.5" class="stroke-grid"/>
<text x="62.0" y="191.5" class="fig-small" text-anchor="end">0.2</text>
<line x1="70.0" y1="145.0" x2="560" y2="145.0" class="stroke-grid"/>
<text x="62.0" y="149.0" class="fig-small" text-anchor="end">0.4</text>
<line x1="70.0" y1="102.5" x2="560" y2="102.5" class="stroke-grid"/>
<text x="62.0" y="106.5" class="fig-small" text-anchor="end">0.6</text>
<line x1="70.0" y1="60.0" x2="560" y2="60.0" class="stroke-grid"/>
<text x="62.0" y="64.0" class="fig-small" text-anchor="end">0.8</text>
<rect x="100" y="123.8" width="30" height="106.2" class="fill-sunk"/>
<text x="115" y="117.8" class="fig-small" text-anchor="middle">0.500</text>
<rect x="136" y="137.9" width="30" height="92.1" class="fill-a"/>
<text x="151" y="131.9" class="fig-small" text-anchor="middle">0.433</text>
<rect x="172" y="177.4" width="30" height="52.6" class="fill-b"/>
<text x="187" y="171.4" class="fig-small" text-anchor="middle">0.248</text>
<text x="154" y="256" class="fig-small" text-anchor="middle">action 0, A = 1</text>
<rect x="260" y="166.2" width="30" height="63.8" class="fill-sunk"/>
<text x="275" y="160.2" class="fig-small" text-anchor="middle">0.300</text>
<rect x="296" y="209.7" width="30" height="20.3" class="fill-a"/>
<text x="311" y="203.7" class="fig-small" text-anchor="middle">0.096</text>
<rect x="332" y="225.7" width="30" height="4.3" class="fill-b"/>
<text x="347" y="219.7" class="fig-small" text-anchor="middle">0.020</text>
<text x="314" y="256" class="fig-small" text-anchor="middle">action 1, A = 0</text>
<rect x="420" y="187.5" width="30" height="42.5" class="fill-sunk"/>
<text x="435" y="181.5" class="fig-small" text-anchor="middle">0.200</text>
<rect x="456" y="129.9" width="30" height="100.1" class="fill-a"/>
<text x="471" y="123.9" class="fig-small" text-anchor="middle">0.471</text>
<rect x="492" y="74.4" width="30" height="155.6" class="fill-b"/>
<text x="507" y="68.4" class="fig-small" text-anchor="middle">0.732</text>
<text x="474" y="256" class="fig-small" text-anchor="middle">action 2, A = 2</text>
<rect x="150" y="22" width="12" height="12" class="fill-sunk"/>
<text x="168" y="33" class="fig-small">pi_old</text>
<rect x="280" y="22" width="12" height="12" class="fill-a"/>
<text x="298" y="33" class="fig-small">beta = 1</text>
<rect x="410" y="22" width="12" height="12" class="fill-b"/>
<text x="428" y="33" class="fig-small">beta = 0.5</text>
</svg>
Figure 1. The improved policy of a three-action state at two temperatures. Halving beta moves 26 points of probability onto the best action, and action 1, whose advantage is zero, shrinks both times.
```

In code, add the log of the old probabilities to $A / \beta$ and take a softmax:

```python
z = np.log(pi_old) + adv / beta
pi_star = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
```

Because $A = Q - V$ and $V(s)$ is the same for every action, $e^{-V/\beta}$ cancels in $Z$, so weighting by $Q$ gives the same policy as weighting by $A$. The exercise checks that adding 1000 to every advantage changes nothing.

You can also state the problem as a budget: maximize the expected advantage subject to $\mathrm{KL}(\pi \,\Vert\, \pi_{\text{old}}) \le \epsilon$. The constraint's Lagrange multiplier plays the role of $\beta$, so every budget $\epsilon$ has one matching $\beta$. MPO fixes $\epsilon$ and solves for $\beta$.

> [!KEY]
> Within a KL distance of the current policy, the best policy multiplies each action's probability by $\exp(A / \beta)$ and renormalizes. It is the Boltzmann form with the old policy in place of a uniform prior, and the language-model optimum with the advantage in place of the reward.

## From a formula to a loss: AWR

The formula sums over every action, which is fine for three actions and impossible for a continuous action. So fit a parametric policy $\pi_\theta$ to $\pi^*$ by minimizing $\mathrm{KL}(\pi^* \,\Vert\, \pi_\theta)$. Drop the part that does not depend on $\theta$, and an expectation under $\pi^*$ of $-\log \pi_\theta(a)$ is left. Because $\pi^* \propto \pi_{\text{old}}\, e^{A/\beta}$, it is a weighted average over actions drawn from $\pi_{\text{old}}$, or from any dataset, with weight $e^{A/\beta}$:

$$
\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} w_i \log \pi_\theta(a_i \mid s_i), \qquad w_i = \min\!\Big(\exp\!\Big(\frac{A_i}{\beta}\Big),\, w_{\max}\Big)
$$

Read it aloud: behavior cloning, except that each logged action counts $w_i$ times. This is advantage-weighted regression (Peng and colleagues, 2019). It samples nothing from $\pi_\theta$ and needs no probability ratio, so it works with data from any behavior policy, and the loss is the same for a softmax and for a Gaussian.

The cap $w_{\max}$ stops one huge advantage from swamping the batch. Peng and colleagues set $w_{\max} = 20$ and $\beta = 0.05$, and they leave the weights unnormalized. Cap the exponent before you exponentiate, so that an advantage of $10^4$ does not overflow:

```python
w = np.exp(np.minimum(adv / beta, np.log(clip)))   # (N,), never inf
```

With $\beta = 0.5$ and $w_{\max} = 20$, advantages $(-1, 0, 1, 2)$ give weights $(0.135, 1, 7.389, 20)$. The last one would be $e^4 = 54.6$ without the cap.

For a softmax policy with logits $z_i$, the loss gradient for row $i$ is $w_i (\operatorname{softmax}(z_i) - e_{a_i}) / N$, where $e_{a_i}$ is the indicator vector of the logged action. The exercise checks it by finite differences.

> [!WARNING]
> Papers disagree about what $\beta$ means. Here, and in AWR and MPO, $\beta$ is a temperature and the weight is $\exp(A/\beta)$, so a small $\beta$ is sharp. The IQL code multiplies the advantage by a number it also calls beta, an inverse temperature. Writing $\exp(A \beta)$ when $\beta$ is a temperature flips the effect of every setting.

SAC's actor fits the same Boltzmann target in the other direction, with reparameterized samples and gradients of $Q$ ([Soft actor-critic](/lessons/042-soft-actor-critic)). AWR minimizes the forward KL, which needs only logged actions and weights.

## Choose the temperature from a budget: MPO

A fixed $\beta$ is hard to set, because its right value depends on the scale of the advantages, which shifts as the critic trains. MPO (Abdolmaleki and colleagues, 2018) fixes the KL budget $\epsilon$ instead and solves for the temperature, which it calls $\eta$.

At a state, draw $N$ actions $a_i$ from $\pi_{\text{old}}$ and score each with $q_i = Q(s, a_i)$. The *E-step* weights are a softmax over the batch, $w_i = e^{q_i / \eta} / \sum_j e^{q_j / \eta}$. Because the samples already came from $\pi_{\text{old}}$, the old policy is the uniform distribution over the $N$ samples, and the budget is the KL from uniform, $\sum_i w_i \log(N w_i)$. A large $\eta$ gives nearly uniform weights, and a small one puts nearly all the weight on the best sample.

The temperature minimizes the convex dual of the constrained problem:

$$
g(\eta) = \eta \epsilon + \eta \log \frac{1}{N} \sum_{i=1}^{N} \exp\!\Big(\frac{q_i}{\eta}\Big)
$$

Differentiate it. The KL of the weights equals $\frac{1}{\eta}\sum_i w_i q_i - \log \operatorname{mean}_j e^{q_j/\eta}$, and subtracting it from $\epsilon$ gives $g'(\eta) = \epsilon - \mathrm{KL}(w \,\Vert\, \text{uniform})$. So the dual is lowest where the weights sit exactly $\epsilon$ from uniform. The KL falls as $\eta$ grows, so a bisection finds that point.

Take $q = (0, 1, 2, 3)$ and $\epsilon = 0.2$. At $\eta = 1$ the weights have a KL of $0.439$, over budget. At $\eta = 2$ it is $0.141$, under. The answer is $\eta^* = 1.6432$, with weights $(0.0805, 0.1479, 0.2719, 0.4997)$ and KL exactly $0.2$. The dual is $2.1907$ there, below $2.2539$ at $\eta = 1$ and $2.2021$ at $\eta = 2$. Figure 2 draws the dual and the KL on one axis of $\eta$.

```figure
<svg viewBox="0 0 600 400" role="img" aria-label="MPO temperature dual and the KL of the reweighted samples against eta; the dual is lowest where the KL equals epsilon">
<line x1="80.0" y1="145.0" x2="540.0" y2="145.0" class="stroke-grid"/>
<text x="72.0" y="149.0" class="fig-small" text-anchor="end">2.2</text>
<line x1="80.0" y1="115.0" x2="540.0" y2="115.0" class="stroke-grid"/>
<text x="72.0" y="119.0" class="fig-small" text-anchor="end">2.6</text>
<line x1="80.0" y1="85.0" x2="540.0" y2="85.0" class="stroke-grid"/>
<text x="72.0" y="89.0" class="fig-small" text-anchor="end">3.0</text>
<line x1="80.0" y1="55.0" x2="540.0" y2="55.0" class="stroke-grid"/>
<text x="72.0" y="59.0" class="fig-small" text-anchor="end">3.4</text>
<polyline points="80.0,93.9 87.8,94.6 95.6,95.4 103.4,96.2 111.2,97.2 119.0,98.1 126.8,99.2 134.6,100.3 142.4,101.5 150.2,102.9 158.0,104.3 165.8,105.7 173.6,107.3 181.4,109.0 189.2,110.7 196.9,112.6 204.7,114.5 212.5,116.5 220.3,118.5 228.1,120.6 235.9,122.7 243.7,124.9 251.5,127.0 259.3,129.1 267.1,131.2 274.9,133.2 282.7,135.1 290.5,137.0 298.3,138.7 306.1,140.2 313.9,141.6 321.7,142.9 329.5,143.9 337.3,144.7 345.1,145.3 352.9,145.6 360.7,145.7 368.5,145.5 376.3,145.1 384.1,144.4 391.9,143.3 399.7,142.0 407.5,140.4 415.3,138.4 423.1,136.1 430.8,133.4 438.6,130.3 446.4,126.9 454.2,123.0 462.0,118.6 469.8,113.8 477.6,108.5 485.4,102.6 493.2,96.1 501.0,89.1 508.8,81.3 516.6,72.9 524.4,63.7 532.2,53.7 540.0,42.8" class="stroke-b" fill="none"/>
<circle cx="359.6" cy="145.7" r="5" class="fill-b"/>
<text x="369.6" y="169.7" class="fig-small">lowest at eta = 1.64</text>
<text x="80.0" y="26" class="fig-small fig-bold">dual g(eta)</text>
<line x1="359.6" y1="160.0" x2="359.6" y2="340" class="stroke-muted dashed"/>
<line x1="80.0" y1="340.0" x2="540.0" y2="340.0" class="stroke-grid"/>
<text x="72.0" y="344.0" class="fig-small" text-anchor="end">0.0</text>
<line x1="80.0" y1="298.3" x2="540.0" y2="298.3" class="stroke-grid"/>
<text x="72.0" y="302.3" class="fig-small" text-anchor="end">0.4</text>
<line x1="80.0" y1="256.7" x2="540.0" y2="256.7" class="stroke-grid"/>
<text x="72.0" y="260.7" class="fig-small" text-anchor="end">0.8</text>
<line x1="80.0" y1="215.0" x2="540.0" y2="215.0" class="stroke-grid"/>
<text x="72.0" y="219.0" class="fig-small" text-anchor="end">1.2</text>
<polyline points="80.0,195.6 87.8,195.7 95.6,195.8 103.4,195.9 111.2,196.2 119.0,196.5 126.8,197.0 134.6,197.8 142.4,198.7 150.2,200.0 158.0,201.6 165.8,203.6 173.6,206.0 181.4,208.7 189.2,212.0 196.9,215.6 204.7,219.7 212.5,224.1 220.3,228.9 228.1,234.1 235.9,239.5 243.7,245.2 251.5,251.1 259.3,257.1 267.1,263.1 274.9,269.1 282.7,275.1 290.5,280.9 298.3,286.4 306.1,291.8 313.9,296.8 321.7,301.4 329.5,305.8 337.3,309.7 345.1,313.4 352.9,316.6 360.7,319.5 368.5,322.2 376.3,324.5 384.1,326.5 391.9,328.3 399.7,329.9 407.5,331.3 415.3,332.5 423.1,333.5 430.8,334.4 438.6,335.2 446.4,335.9 454.2,336.5 462.0,337.0 469.8,337.4 477.6,337.8 485.4,338.1 493.2,338.4 501.0,338.6 508.8,338.8 516.6,339.0 524.4,339.1 532.2,339.2 540.0,339.4" class="stroke-a" fill="none"/>
<line x1="80.0" y1="319.2" x2="540.0" y2="319.2" class="stroke-c dashed"/>
<text x="540.0" y="313.2" class="fig-small" text-anchor="end">budget epsilon = 0.2</text>
<circle cx="359.6" cy="319.2" r="5" class="fill-a"/>
<text x="80.0" y="205" class="fig-small fig-bold">KL of the weights from uniform</text>
<line x1="80.0" y1="340.0" x2="80.0" y2="345.0" class="stroke-muted"/>
<text x="80.0" y="360.0" class="fig-small" text-anchor="middle">0.1</text>
<line x1="310.0" y1="340.0" x2="310.0" y2="345.0" class="stroke-muted"/>
<text x="310.0" y="360.0" class="fig-small" text-anchor="middle">1</text>
<line x1="540.0" y1="340.0" x2="540.0" y2="345.0" class="stroke-muted"/>
<text x="540.0" y="360.0" class="fig-small" text-anchor="middle">10</text>
<text x="310.0" y="388" class="fig-small" text-anchor="middle">eta (log scale)</text>
</svg>
Figure 2. The dual, top, is lowest at the temperature where the KL of the reweighted samples, bottom, crosses the budget epsilon. The dashed line connects the two views.
```

The *M-step* fits $\pi_\theta$ to those weighted samples by maximum likelihood, $\max_\theta \sum_i w_i \log \pi_\theta(a_i \mid s)$. This is the same weighted loss as AWR, with weights normalized across the batch and a temperature set by the budget. For a Gaussian policy it has a closed form, the weighted mean and weighted standard deviation:

```python
w = w / w.sum()
mu = np.sum(w * a)
sigma = np.sqrt(np.sum(w * (a - mu) ** 2))
```

MPO also puts a KL trust region on the M-step, which this chapter leaves out. Try actions $a \sim \mathcal{N}(0, 1)$ with $q = -(a - 1)^2$ and $\epsilon = 0.1$. Reweighting a standard normal by $e^{q/\eta}$ gives a Gaussian with precision $1 + 2/\eta$ and mean $(2/\eta)/(1 + 2/\eta)$. The bisection finds $\eta \approx 3.72$, so the improved Gaussian has mean $0.35$ and standard deviation $0.81$: a third of the way toward the best action, and narrower.

The E-step and M-step are the two halves of EM in the control-as-inference view (CS285 lecture 13).

> [!KEY]
> AWR fixes the temperature $\beta$ and leaves the weights unnormalized. MPO fixes the KL budget $\epsilon$, solves its dual for the temperature $\eta$, and normalizes the weights across the batch. Both finish with the same weighted maximum-likelihood fit.

## AWAC: a critic that reuses old data

AWR estimates advantages from the returns in its data, which ties it to recent data. AWAC (Nair and colleagues, 2020) uses an off-policy critic instead. It fits $Q$ by temporal-difference learning with the target $r + \gamma\, \mathbb{E}_{a' \sim \pi}[Q(s', a')]$, computes $A(s, a) = Q(s, a) - \mathbb{E}_{a \sim \pi} Q(s, a)$, and updates the actor by the weighted maximum likelihood of the buffer's actions. IQL has the same actor step and differs only in where $A$ comes from: an expectile $V$ instead of the current policy.

The exercise builds a tabular AWAC on the offline cliff data from the previous chapter: 2,370 transitions from a cautious demonstrator at $\gamma = 0.9$. Start from behavior cloning and repeat four steps: evaluate the critic under the current policy, compute each logged transition's advantage, weight with the clipped exponential, and refit. In a table the weighted maximum-likelihood policy is closed form: $\pi(a \mid s)$ is the sum of the weights of the transitions with that $(s, a)$, divided by the sum over all actions at $s$.

Behavior cloning is worth $-9.018$ at the start. AWAC reaches $-8.701$ at $\beta = 3$, $-8.237$ at $\beta = 1$, $-7.635$ at $\beta = 0.3$, and $-7.471$ at $\beta = 0.1$. The best any policy limited to the data's actions can do is $-7.458$. A huge $\beta$ returns behavior cloning, and a tiny one nears the best in-data policy but cannot pass it, because the weights only reshuffle actions the data contains.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="Value at the start of the cliff world for behavior cloning and for AWAC at four temperatures, between behavior cloning and the best in-data value">
<line x1="80.0" y1="210.0" x2="540.0" y2="210.0" class="stroke-grid"/>
<text x="72.0" y="214.0" class="fig-small" text-anchor="end">-9</text>
<line x1="80.0" y1="170.0" x2="540.0" y2="170.0" class="stroke-grid"/>
<text x="72.0" y="174.0" class="fig-small" text-anchor="end">-8.5</text>
<line x1="80.0" y1="130.0" x2="540.0" y2="130.0" class="stroke-grid"/>
<text x="72.0" y="134.0" class="fig-small" text-anchor="end">-8</text>
<line x1="80.0" y1="90.0" x2="540.0" y2="90.0" class="stroke-grid"/>
<text x="72.0" y="94.0" class="fig-small" text-anchor="end">-7.5</text>
<line x1="80.0" y1="86.7" x2="540.0" y2="86.7" class="stroke-c dashed"/>
<text x="540.0" y="80.7" class="fig-small" text-anchor="end">best the data supports: -7.458</text>
<line x1="80.0" y1="211.4" x2="540.0" y2="211.4" class="stroke-muted dashed"/>
<text x="540.0" y="227.4" class="fig-small" text-anchor="end">behavior cloning: -9.018</text>
<polyline points="120.0,186.1 242.7,149.0 377.3,100.8 500.0,87.7" class="stroke-b" fill="none"/>
<circle cx="120.0" cy="186.1" r="5" class="fill-b"/>
<text x="120.0" y="176.1" class="fig-small" text-anchor="middle">-8.701</text>
<text x="120.0" y="270.0" class="fig-small" text-anchor="middle">3</text>
<circle cx="242.7" cy="149.0" r="5" class="fill-b"/>
<text x="242.7" y="139.0" class="fig-small" text-anchor="middle">-8.237</text>
<text x="242.7" y="270.0" class="fig-small" text-anchor="middle">1</text>
<circle cx="377.3" cy="100.8" r="5" class="fill-b"/>
<text x="377.3" y="90.8" class="fig-small" text-anchor="middle">-7.635</text>
<text x="377.3" y="270.0" class="fig-small" text-anchor="middle">0.3</text>
<circle cx="500.0" cy="87.7" r="5" class="fill-b"/>
<text x="500.0" y="77.7" class="fig-small" text-anchor="middle">-7.471</text>
<text x="500.0" y="270.0" class="fig-small" text-anchor="middle">0.1</text>
<text x="310.0" y="294.0" class="fig-small" text-anchor="middle">beta (large on the left, small on the right)</text>
</svg>
Figure 3. AWAC's value at the start of the cliff world against its temperature, on the offline dataset. A large beta stays at behavior cloning, a small beta nears the best value the data supports, and none passes it.
```

## Four names for one step

The language-model chapter says a real model cannot compute $Z$ in $\pi^* \propto \pi_{\text{ref}}\, e^{r/\beta}$. AWR is the way around it: sample from the reference, weight by $e^{r/\beta}$, and fit by weighted maximum likelihood. Reward-weighted regression (Peters and Schaal, 2007) is that fit with a weight that is any function of the reward, and expert iteration is the case of a 0/1 pass indicator (see [AI feedback and self-improvement](/lessons/076-ai-feedback-and-self-improvement)). The Boltzmann policy is the case where $\pi_{\text{old}}$ is uniform.

> [!RECAP]
> - The best policy within a KL leash of $\pi_{\text{old}}$ is $\pi_{\text{old}}\, e^{A/\beta} / Z$, and its score is $\beta \log Z$.
> - AWR fits it by weighted log-likelihood with weights $\min(e^{A/\beta}, w_{\max})$.
> - MPO picks $\eta$ so the KL of the batch-normalized weights from uniform equals $\epsilon$, then fits by weighted maximum likelihood.
> - AWAC computes the advantages from an off-policy critic, and a large $\beta$ gives back behavior cloning.

## Build it step by step

The checks test the functions in this order. The helpers `make_dataset`, `action_counts`, `cautious_behavior`, `behavior_cloning`, and `weight_kl` are written for you.

1. **`awr_weights(adv, beta, clip)`.** Cap the exponent with `np.minimum(adv / beta, np.log(clip))`, then `np.exp`. The common mistake is `adv * beta`.
2. **`kl_improved_policy(pi_old, adv, beta)`.** A softmax of `np.log(pi_old) + adv / beta`.
3. **`improvement_objective(pi, pi_old, adv, beta)`.** The expected advantage minus `beta` times the KL, summed only where `pi > 0`.
4. **`mpo_weights(q, eta)`.** A softmax of `q / eta` over the batch. Skipping the normalization makes the KL meaningless.
5. **`mpo_dual(q, eta, eps)`.** Use `top / eta + log(mean(exp((q - top) / eta)))` for the log-mean-exp.
6. **`mpo_temperature(q, eps)`.** Bisect on $\log \eta$, since the KL falls as $\eta$ grows.
7. **`awr_loss` and `awr_loss_grad`.** The gradient row is `weights[i] * (softmax - onehot) / N`.
8. **`weighted_gaussian_mle(actions, weights)`.** Normalize the weights first.
9. **`mpo_gaussian_step(actions, q, eps)`.** Chain the temperature, the weights, and the Gaussian fit.
10. **`awac_tabular(data, S, A, gamma, beta)`.** Critic target `r + gamma * (1 - done) * (pi * Q).sum(1)[s2]`, then weights, then a per-row normalized `np.add.at` sum.

## Where this breaks

**The exponential amplifies noise.** An advantage that is off by one gets a weight off by a factor of $e^{1/\beta}$. The cap, a larger $\beta$, and normalizing advantages by their batch standard deviation limit the damage.

**Few samples carry the update.** As $\beta$ falls, a handful of actions hold nearly all the weight. MPO's budget $\epsilon$ controls this directly.

**A value-free alternative.** A [decision transformer](/lessons/056-decision-transformer) conditions a sequence model on the return you want and needs no critic. It cannot stitch trajectories the way a critic can.

**The fit is forward KL.** A Gaussian fitted to a two-peaked target lands between the peaks.

## Further reading

- [Peng, Kumar, Zhang, and Levine (2019), Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning](https://arxiv.org/abs/1910.00177). Derives the weighted-regression update from a KL trust region, with $\beta = 0.05$ and weights clipped at 20.
- [Nair, Gupta, Dalal, and Levine (2020), AWAC: Accelerating Online Reinforcement Learning with Offline Datasets](https://arxiv.org/abs/2006.09359). AWR with an off-policy critic, for offline pretraining and online fine-tuning.
- [Abdolmaleki, Springenberg, Tassa, Munos, Heess, and Riedmiller (2018), Maximum a Posteriori Policy Optimisation](https://arxiv.org/abs/1806.06920). The E-step, the temperature dual, and the M-step with a trust region.
- [Peters and Schaal (2007), Reinforcement Learning by Reward-Weighted Regression for Operational Space Control](https://doi.org/10.1145/1273496.1273590). The EM view of reward-weighted likelihood, from the robotics side.
- [Kostrikov, Nair, and Levine (2022), Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169). Advantage-weighted extraction on top of an expectile critic.
- [CS285, Spring 2026, lectures 11 to 13: "Variational Inference", "VI in RL", and "Control as Inference"](https://rail.eecs.berkeley.edu/deeprlcourse/). The probabilistic view behind the E-step and M-step. Lectures 17 and 18 are "Offline RL Part 1" and "Offline RL Part 2".

## The exercise

You build the improvement step in pieces and finish with AWAC on the cliff data, writing the ten functions above in order, from `awr_weights` to `awac_tabular`.

The checks confirm the weights against hand values and the improved policy $(0.4333, 0.0956, 0.4711)$, which must beat 3,000 random rivals and score $\log Z$. They check the dual's derivative and the loss gradient by finite differences, require the temperature to put the weights' KL at $\epsilon$ within $10^{-5}$, and require the Gaussian fit to match the analytic reweighted Gaussian within $0.03$. AWAC must beat behavior cloning by at least 1.0.
