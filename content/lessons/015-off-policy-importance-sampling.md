---
title: Learning about one policy from another
slug: 015-off-policy-importance-sampling
part: "Part 3 — Learning from experience"
summary: Estimate one policy's value from episodes that a different policy played, by reweighting each return and comparing two estimators.
minutes: 40
gpu: false
objectives:
  - Explain why episodes from one policy can estimate another policy's value once you reweight them.
  - Compute the importance ratio of an episode by hand, and with one reversed cumulative product.
  - Compute the ordinary and weighted importance-sampling estimates, and say why one is unbiased and the other stable.
  - Say when ordinary importance sampling has infinite variance.
prereqs:
  - 013-monte-carlo-prediction
  - 014-monte-carlo-control
exercise: 015-off-policy-importance-sampling
---

# Learning about one policy from another

You have a log of episodes that last week's controller played, and you want to know how a different policy would have done, without running it. Running it might be expensive, or unsafe, or the log is all the data there is. The obvious move is to average the returns in the log. But that average is the value of last week's controller, not the value of the policy you care about.

Monte Carlo prediction assumed that the policy that played the episodes was the policy you were evaluating. This chapter drops that assumption. The idea is to count each episode in the log by how much more, or less, likely your policy would have been to produce it. An episode your policy would produce twice as often as the old one counts twice. Done carefully, the weighted average estimates the value of the policy you care about, and you can see what the data can and can't tell you.

## What you already know

- Monte Carlo prediction averages the returns that followed a state, and a return $G$ is the discounted sum of rewards in an episode.
- A policy is an `(S, A)` array whose rows sum to one, and an episode is three arrays: states, actions, and rewards.
- An average of draws from one distribution estimates the expected value under that distribution, and nothing else.
- Learning about the policy that generated the data is *on-policy*, and the chapters so far did that.
- The *behavior policy* $b$ is the one that generated the data. The *target policy* $\pi$ is the one you want to evaluate. Learning about $\pi$ from $b$'s data is *off-policy*.

## Count each episode by how likely it was

Start with a one-step problem. There are two actions: $L$ pays 1 and $R$ pays 0. The target policy picks $L$ with probability $\pi(L) = 0.9$, so its value is $V^\pi = 0.9$. The behavior policy flips a fair coin, $b(L) = 0.5$, so a plain average of its returns converges to $V^b = 0.5$, which is the wrong answer.

Look at one episode. An $L$ episode is $0.9 / 0.5 = 1.8$ times as likely under $\pi$ as under $b$, so in a world where $\pi$ plays, it would show up 1.8 times as often. An $R$ episode is $0.1 / 0.5 = 0.2$ times as likely. So give each episode a weight equal to that ratio before you average. An $L$ episode counts 1.8 and an $R$ episode counts 0.2. With those weights, the log contains the same mix as $\pi$ would produce.

## Multiply the ratios along the episode

A real episode has many steps. The probability of the whole trajectory is a product: at each step, the policy's probability of the action times the environment's probability of the next state. The environment's factors are identical under $\pi$ and $b$, because the world doesn't care who is acting. So they cancel in the ratio, and the weight needs only the two policies, with no model. The weight of the part of an episode from step $t$ to the end is

$$
\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(a_k \mid s_k)}{b(a_k \mid s_k)}
$$

Read it as: at each step, divide the target policy's probability of the action taken by the behavior policy's, and multiply the ratios from step $t$ onward. The ratio is only defined when $b$ takes every action that $\pi$ might take. This is the *coverage* condition, and without it no amount of data reveals what $\pi$ would do in the missing cases.

Take three steps with ratios $(1.8, 1.6, 0.2)$. The tail products are $1.8 \times 1.6 \times 0.2 = 0.576$ from step 0, $1.6 \times 0.2 = 0.32$ from step 1, and $0.2$ from step 2. All the tails come from one reversed cumulative product:

```python
step = pi[states, actions] / b[states, actions]
rho = np.cumprod(step[::-1])[::-1]       # rho[t] is rho_{t:T-1}
```

Notice that one step with $\pi(a \mid s) = 0$ makes every tail up to that step zero, because a policy that would never take that action would never produce the episode.

## Two ways to average

Given $n$ episodes from the same start state, with returns $G^{(i)}$ and full-episode weights $\rho^{(i)} = \rho^{(i)}_{0:T-1}$, there are two natural estimates of $V^\pi$:

$$
V_{\text{OIS}} = \frac{1}{n} \sum_{i=1}^{n} \rho^{(i)} G^{(i)},
\qquad
V_{\text{WIS}} = \frac{\sum_{i=1}^{n} \rho^{(i)} G^{(i)}}{\sum_{i=1}^{n} \rho^{(i)}}
$$

Read them as follows. *Ordinary importance sampling* (OIS) multiplies each return by its weight and divides by the number of episodes. *Weighted importance sampling* (WIS) divides by the total weight instead, so it is a weighted average of the returns you saw. In code:

```python
ois = np.mean(w * G)
wis = np.sum(w * G) / np.sum(w)
```

Run both on the one-step problem with three behavior episodes, $L$, $R$, $L$. The returns are $(1, 0, 1)$ and the weights are $(1.8, 0.2, 1.8)$.

- OIS gives $(1.8 + 0 + 1.8) / 3 = 1.2$. That is above 1, the largest return the problem can pay.
- WIS gives $3.6 / (1.8 + 0.2 + 1.8) = 0.947$.

```figure
<svg viewBox="0 0 570 270" role="img" aria-label="Three behavior episodes L, R, L with importance weights 1.8, 0.2, 1.8, and the two estimates on a number line: ordinary importance sampling gives 1.2, above every possible return, and weighted gives 0.947, near the truth 0.9">
<text x="14.0" y="22.0" class="fig-bold" text-anchor="start">episode</text>
<text x="110.0" y="22.0" class="fig-bold" text-anchor="start">return G</text>
<text x="200.0" y="22.0" class="fig-bold" text-anchor="start">weight rho = pi / b</text>
<text x="14.0" y="56.0" class="fig-small" text-anchor="start">1. action L</text>
<text x="110.0" y="56.0" class="fig-small" text-anchor="start">1</text>
<rect x="200.0" y="43.0" width="200.0" height="20.0" rx="2" class="fill-soft-b stroke-b"/>
<text x="406.0" y="58.0" class="fig-small" text-anchor="start">1.8</text>
<text x="14.0" y="90.0" class="fig-small" text-anchor="start">2. action R</text>
<text x="110.0" y="90.0" class="fig-small" text-anchor="start">0</text>
<rect x="200.0" y="77.0" width="22.2" height="20.0" rx="2" class="fill-soft-b stroke-b"/>
<text x="228.2" y="92.0" class="fig-small" text-anchor="start">0.2</text>
<text x="14.0" y="124.0" class="fig-small" text-anchor="start">3. action L</text>
<text x="110.0" y="124.0" class="fig-small" text-anchor="start">1</text>
<rect x="200.0" y="111.0" width="200.0" height="20.0" rx="2" class="fill-soft-b stroke-b"/>
<text x="406.0" y="126.0" class="fig-small" text-anchor="start">1.8</text>
<rect x="40.0" y="178.0" width="357.1" height="24.0" rx="2" class="fill-soft-c"/>
<line x1="40.0" y1="190.0" x2="540.0" y2="190.0" class="stroke-muted"/>
<line x1="40.0" y1="202.0" x2="40.0" y2="208.0" class="stroke-muted"/>
<text x="40.0" y="224.0" class="fig-small" text-anchor="middle">0</text>
<line x1="111.4" y1="202.0" x2="111.4" y2="208.0" class="stroke-muted"/>
<text x="111.4" y="224.0" class="fig-small" text-anchor="middle">0.2</text>
<line x1="182.9" y1="202.0" x2="182.9" y2="208.0" class="stroke-muted"/>
<text x="182.9" y="224.0" class="fig-small" text-anchor="middle">0.4</text>
<line x1="254.3" y1="202.0" x2="254.3" y2="208.0" class="stroke-muted"/>
<text x="254.3" y="224.0" class="fig-small" text-anchor="middle">0.6</text>
<line x1="325.7" y1="202.0" x2="325.7" y2="208.0" class="stroke-muted"/>
<text x="325.7" y="224.0" class="fig-small" text-anchor="middle">0.8</text>
<line x1="397.1" y1="202.0" x2="397.1" y2="208.0" class="stroke-muted"/>
<text x="397.1" y="224.0" class="fig-small" text-anchor="middle">1</text>
<line x1="468.6" y1="202.0" x2="468.6" y2="208.0" class="stroke-muted"/>
<text x="468.6" y="224.0" class="fig-small" text-anchor="middle">1.2</text>
<line x1="540.0" y1="202.0" x2="540.0" y2="208.0" class="stroke-muted"/>
<text x="540.0" y="224.0" class="fig-small" text-anchor="middle">1.4</text>
<circle cx="361.4" cy="190.0" r="6" class="fill-c"/>
<text x="353.4" y="176.0" class="fig-small" text-anchor="end">truth 0.9</text>
<circle cx="378.3" cy="190.0" r="6" class="fill-b"/>
<circle cx="468.6" cy="190.0" r="6" class="fill-bad"/>
<text x="468.6" y="176.0" class="fig-small" text-anchor="middle">OIS 1.2</text>
<text x="384.3" y="176.0" class="fig-small" text-anchor="start">WIS 0.947</text>
<text x="218.6" y="252.0" class="fig-small" text-anchor="middle">every return the problem can pay lies in [0, 1] (green)</text>
</svg>
Figure 1. The three-episode example. Notice that the ordinary estimate 1.2 is larger than any return the problem can pay, while the weighted estimate 0.947 stays inside the range of returns and lands close to the true value 0.9.
```

OIS is *unbiased*: its average over many batches is exactly $V^\pi$, because the weights undo the change of policy in expectation. But a single batch can land far from the truth, as 1.2 shows. WIS is *biased*: with one episode it returns that episode's return, which estimates $V^b$, not $V^\pi$. The bias shrinks to zero as $n$ grows. In exchange, WIS can never leave the range of the returns you saw.

## Bias against variance

Which one should you trust? Draw 2,000 batches of 10 behavior episodes from the one-step problem, and compute both estimates for each batch. With $k$ of the 10 episodes being $L$, OIS is exactly $0.18k$ and WIS is $9k / (8k + 10)$. Figure 2 shows the result.

```figure
<svg viewBox="0 0 570 310" role="img" aria-label="Histograms over 2000 batches of 10 episodes of the ordinary and weighted importance sampling estimates: ordinary is spread from 0 to 1.8, weighted is bunched near 0.9 and never above 1">
<line x1="60.0" y1="130.0" x2="540.0" y2="130.0" class="stroke-muted"/>
<text x="14.0" y="29.0" class="fig-bold" text-anchor="start">OIS</text>
<rect x="53.0" y="129.6" width="14.0" height="0.4" rx="1" class="fill-bad"/>
<rect x="98.5" y="123.3" width="14.0" height="6.7" rx="1" class="fill-bad"/>
<rect x="143.9" y="110.7" width="14.0" height="19.3" rx="1" class="fill-bad"/>
<rect x="189.4" y="81.5" width="14.0" height="48.5" rx="1" class="fill-bad"/>
<rect x="234.9" y="39.5" width="14.0" height="90.5" rx="1" class="fill-bad"/>
<rect x="280.4" y="25.0" width="14.0" height="105.0" rx="1" class="fill-bad"/>
<rect x="325.8" y="42.1" width="14.0" height="87.9" rx="1" class="fill-bad"/>
<rect x="371.3" y="78.3" width="14.0" height="51.7" rx="1" class="fill-bad"/>
<rect x="416.8" y="112.9" width="14.0" height="17.1" rx="1" class="fill-bad"/>
<rect x="462.3" y="124.6" width="14.0" height="5.4" rx="1" class="fill-bad"/>
<rect x="507.7" y="129.6" width="14.0" height="0.4" rx="1" class="fill-bad"/>
<line x1="60.0" y1="275.0" x2="540.0" y2="275.0" class="stroke-muted"/>
<text x="14.0" y="174.0" class="fig-bold" text-anchor="start">WIS</text>
<rect x="53.0" y="274.6" width="14.0" height="0.4" rx="1" class="fill-c"/>
<rect x="179.3" y="268.3" width="14.0" height="6.7" rx="1" class="fill-c"/>
<rect x="227.9" y="255.7" width="14.0" height="19.3" rx="1" class="fill-c"/>
<rect x="253.6" y="226.5" width="14.0" height="48.5" rx="1" class="fill-c"/>
<rect x="269.5" y="184.5" width="14.0" height="90.5" rx="1" class="fill-c"/>
<rect x="280.4" y="170.0" width="14.0" height="105.0" rx="1" class="fill-c"/>
<rect x="288.2" y="187.1" width="14.0" height="87.9" rx="1" class="fill-c"/>
<rect x="294.1" y="223.3" width="14.0" height="51.7" rx="1" class="fill-c"/>
<rect x="298.8" y="257.9" width="14.0" height="17.1" rx="1" class="fill-c"/>
<rect x="302.6" y="269.6" width="14.0" height="5.4" rx="1" class="fill-c"/>
<rect x="305.6" y="274.6" width="14.0" height="0.4" rx="1" class="fill-c"/>
<line x1="60.0" y1="275.0" x2="60.0" y2="281.0" class="stroke-muted"/>
<text x="60.0" y="297.0" class="fig-small" text-anchor="middle">0</text>
<line x1="186.3" y1="275.0" x2="186.3" y2="281.0" class="stroke-muted"/>
<text x="186.3" y="297.0" class="fig-small" text-anchor="middle">0.5</text>
<line x1="287.4" y1="275.0" x2="287.4" y2="281.0" class="stroke-muted"/>
<text x="287.4" y="297.0" class="fig-small" text-anchor="middle">0.9</text>
<line x1="312.6" y1="275.0" x2="312.6" y2="281.0" class="stroke-muted"/>
<text x="312.6" y="297.0" class="fig-small" text-anchor="middle">1</text>
<line x1="438.9" y1="275.0" x2="438.9" y2="281.0" class="stroke-muted"/>
<text x="438.9" y="297.0" class="fig-small" text-anchor="middle">1.5</text>
<line x1="287.4" y1="20.0" x2="287.4" y2="275.0" class="stroke-a dashed"/>
<text x="292.4" y="22.0" class="fig-small" text-anchor="start">truth 0.9</text>
<line x1="312.6" y1="150.0" x2="312.6" y2="275.0" class="stroke-ink dashed"/>
<text x="317.6" y="160.0" class="fig-small" text-anchor="start">largest possible return</text>
<text x="540.0" y="120.0" class="fig-small" text-anchor="end">std 0.290</text>
<text x="540.0" y="265.0" class="fig-small" text-anchor="end">std 0.085</text>
</svg>
Figure 2. Two thousand batches of ten episodes from the one-step problem. Notice that the ordinary estimates spread from 0 to 1.8 with a standard deviation of 0.290, and about 38 percent of them exceed 1, while the weighted estimates stay in [0, 1] with a standard deviation of 0.085 and sit slightly low, at an average of 0.880.
```

The averages are 0.897 for OIS and 0.880 for WIS. OIS is within noise of the truth 0.9, because its standard error is $0.290 / \sqrt{2000} = 0.0065$ and it is off by 0.003. WIS is low by 0.020. But the spread tells the other half. OIS has a standard deviation of 0.290 and about 38 percent of its batches exceed 1, a value that no episode can earn. WIS has a standard deviation of 0.085 and never leaves $[0, 1]$. Measured by root-mean-square error, which counts bias and spread together, WIS scores 0.088 against OIS's 0.290, so a single WIS batch is much closer to the truth, even though WIS is biased.

> [!KEY]
> Ordinary importance sampling is unbiased but can have huge variance. Weighted importance sampling is biased for small samples, but its bias vanishes as $n$ grows, it always stays within the range of the returns you saw, and its variance is usually far smaller.

### When the variance is infinite

Sutton and Barto's Example 5.5 shows how bad OIS can get. One state has two actions. One ends the episode with reward 0. The other returns to the state with probability 0.9, and otherwise ends the episode with reward 1. The target policy always takes the looping action, so $V^\pi = 1$. The behavior policy picks each action half the time.

An episode that loops $k$ times and then ends with reward 1 has probability $(0.5 \times 0.9)^k \times (0.5 \times 0.1)$ under $b$, and its weight is $2^{k+1}$, because each of its $k + 1$ steps has ratio $1 / 0.5 = 2$. The expected squared OIS sample adds up probability times squared weight over every episode length:

$$
\mathbb{E}_b\bigl[(\rho G)^2\bigr] = \sum_{k=0}^{\infty} 0.05 \times 0.45^k \times 4^{k+1} = 0.2 \sum_{k=0}^{\infty} 1.8^k
$$

Each extra loop multiplies the probability by 0.45 and the squared weight by 4, so each term is 1.8 times the last and the sum diverges. Its first 10 terms add to 89, and its first 20 to 31,870. OIS still has mean exactly 1, so it converges in the end, but with infinite variance no error bar applies and progress is painfully slow. WIS returns exactly 1 as soon as one episode with nonzero weight arrives, because every such episode returns 1.

> [!RECAP]
> - An off-policy estimate weights each return by $\pi / b$, which needs the two policies and no model.
> - The ratio from step $t$ to the end is `np.cumprod(step[::-1])[::-1]`, where `step = pi[s, a] / b[s, a]`.
> - OIS divides the weighted sum by $n$, and WIS divides it by the total weight.
> - OIS is unbiased with high variance, and WIS is biased for small $n$ with lower variance.
> - You need coverage: $b$ must take every action that $\pi$ might take.

## Build it step by step

The checks test the functions in this order. `discounted_returns` is given.

1. **`importance_ratios(states, actions, pi, b)`.** Takes the first three as arrays of length $T$ and the two policies as `(S, A)` arrays, and returns a `(T,)` array. Index `pi[states, actions]` and `b[states, actions]` to get every step's probability at once, divide, and take the reversed cumulative product. Common mistake: dividing $b$ by $\pi$. The checks use the tails $(0.576, 0.32, 0.2)$, confirm that the ratios are all 1 when $\pi = b$, and check a zero ratio in the middle.
2. **`importance_sampling_estimates(episodes, pi, b, gamma)`.** Returns `(ois, wis)`, two floats. For each episode, take `importance_ratios(...)[0]` and `discounted_returns(...)[0]`. Compute the two averages from the arrays of weights and returns. Common mistake: returning `nan` when every weight is zero, because the total weight is zero. Return `0.0` for WIS in that case.

## Cost and scaling

Every estimate costs $O(T)$ per episode and stores one weight and one return per episode, or two running sums. The hidden cost is variance. A return sums $T$ random rewards, and an importance weight multiplies $T$ ratios, so its variance can grow exponentially with $T$.

## Where this breaks

**Exploding weights.** One rare long episode can carry a weight in the thousands and dominate an OIS average. People use weighted importance sampling, cap the weights, or use *per-decision* methods that weight each reward only by the ratios up to the step that earned it (see [Evaluating a policy from logs](/lessons/052-off-policy-evaluation)).

**Missing coverage.** If the target policy takes an action that the behavior policy never takes, no amount of data reveals its value. Make the behavior policy $\varepsilon$-soft, as in [Monte Carlo control](/lessons/014-monte-carlo-control), so that every action has a chance.

**Policies far apart.** The more $\pi$ and $b$ differ, the more the ratios spread, and the fewer episodes carry most of the weight. A log from a very different policy is a poor guide.

## Further reading

- [Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd edition (2018), Chapter 5](http://incompleteideas.net/book/the-book-2nd.html) covers ordinary and weighted importance sampling, and the infinite-variance example.
- [Precup, Sutton, and Singh, "Eligibility traces for off-policy policy evaluation" (2000)](https://dl.acm.org/doi/10.5555/645529.658134) introduces per-decision importance sampling, which cuts the variance of long-horizon weights.
- [Mahmood, van Hasselt, and Sutton, "Weighted importance sampling for off-policy learning with linear function approximation" (2014)](https://dl.acm.org/doi/10.5555/2969033.2969163) carries the weighted estimator beyond tables.

## The exercise

You build both importance-sampling estimators. Write `importance_ratios(states, actions, pi, b)` for the tail products $\rho_{t:T-1}$, and `importance_sampling_estimates(episodes, pi, b, gamma)` for the ordinary and weighted estimates of a start value. The starter gives `discounted_returns`.

The checks verify the tails $(0.576, 0.32, 0.2)$, that the ratios are all 1 when $\pi = b$, and that a zero step ratio zeroes every tail up to it. They confirm the 1.2 and 0.947 example, the edge cases of one episode and all-zero weights, and a two-step discounted example. Last, they run 2,000 batches and check that OIS is unbiased for $V^\pi = 0.9$, while WIS has lower variance and stays in $[0, 1]$.
