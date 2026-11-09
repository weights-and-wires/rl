---
title: Language models as policies
slug: 049-language-models-as-policies
part: "Part 9 — RL for language models"
summary: Score a response from its token log-probabilities, solve reward minus a KL leash in closed form, and measure how far best-of-n moves.
minutes: 40
gpu: false
objectives:
  - Score a whole response as the masked sum of its token log-probabilities.
  - Explain why maximizing reward alone fails, and write the reward-minus-KL objective that fixes it.
  - Compute the optimum of that objective in closed form, and check it exactly on an enumerable toy.
  - Compute the exact best-of-n distribution and its KL, and say when log n - (n-1)/n is only a bound.
prereqs:
  - 048-sft-and-the-post-training-pipeline
  - 029-the-policy-gradient
  - 037-maximum-entropy-rl
exercise: 049-language-models-as-policies
---

# Language models as policies

A chatbot writes its reply one token at a time, and each token is a choice sampled from a distribution that depends on everything written so far. That is what a policy does, so policy gradients, value functions, and the soft Bellman equations all apply to text.

Post-training changes what the model writes so that people, or a program, score its replies higher. Two questions come first. How do you turn token probabilities into one number for a whole reply? And how hard can you push on a score before the model finds strange replies that score well and read badly?

This chapter answers both. You score whole responses from token probabilities, put the model on a leash and solve the leashed objective exactly, and measure how far best-of-n sampling moves the model.

## What you already know

- A *policy* $\pi(a \mid s)$ gives the probability of each action $a$ in state $s$.
- The policy gradient uses the log-probabilities of the actions a policy took.
- Maximum-entropy RL has a Boltzmann optimal policy, $\pi^*(a \mid s) \propto \exp(Q^*(s, a) / \alpha)$.
- The KL divergence $\mathrm{KL}(p \,\|\, q) = \sum_i p_i \log(p_i / q_i)$ is zero when two distributions match and positive otherwise.

## Score a response one token at a time

A *prompt* $x$ is the text the model is given, and a *response* $y = (y_0, \dots, y_{T-1})$ is the $T$ tokens it writes back. In the *token-level MDP*, the state is the prompt plus everything written so far and the action is the next token. A reward model or a checker scores only the finished response, so the reward is zero at every token except the last.

The model's probability of a whole response is the product of its token probabilities, by the chain rule of probability. So the log-probability is a sum:

$$
\log \pi(y \mid x) = \sum_{t=0}^{T-1} m_t \log \pi(y_t \mid x, y_{<t})
$$

Read it aloud: for each position, take the log-probability the model gave the token it actually wrote, and add them up. The *response mask* $m_t$ is 1 for tokens that belong to the response and 0 for padding, which fills out a batch of responses of different lengths.

Take a vocabulary of three tokens and a response of three tokens, padded to length four. The logits at the four positions are $[2, 1, 0]$, $[0, 0, 0]$, $[1, 3, 0]$, and $[5, 5, 5]$. The response is tokens 0, 2, and 1, with mask $[1, 1, 1, 0]$.

- Position 0: $\pi = e^2 / (e^2 + e + 1) = 0.6652$, log-probability $-0.4076$.
- Position 1: all logits equal, so $\pi = 1/3$ and the log-probability is $-1.0986$.
- Position 2: $\pi = e^3 / (e + e^3 + 1) = 0.8438$, log-probability $-0.1698$.
- Position 3 is padding, so it is left out whatever it holds.

The sum is $-1.6761$, so the whole response has probability $e^{-1.6761} = 0.1871$. Figure 1 shows how each token's probability multiplies in.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Token probabilities 0.665, 0.333, and 0.844 for the three response tokens, a grayed padding position, and their product 0.187">
<line x1="90.0" y1="200.0" x2="570.0" y2="200.0" class="stroke-grid"/>
<text x="82.0" y="204.0" class="fig-small" text-anchor="end">0</text>
<line x1="90.0" y1="160.0" x2="570.0" y2="160.0" class="stroke-grid"/>
<text x="82.0" y="164.0" class="fig-small" text-anchor="end">0.25</text>
<line x1="90.0" y1="120.0" x2="570.0" y2="120.0" class="stroke-grid"/>
<text x="82.0" y="124.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="90.0" y1="80.0" x2="570.0" y2="80.0" class="stroke-grid"/>
<text x="82.0" y="84.0" class="fig-small" text-anchor="end">0.75</text>
<line x1="90.0" y1="40.0" x2="570.0" y2="40.0" class="stroke-grid"/>
<text x="82.0" y="44.0" class="fig-small" text-anchor="end">1</text>
<rect x="116.0" y="93.6" width="68.0" height="106.4" class="fill-a"/>
<text x="150.0" y="75.6" class="fig-small fig-bold" text-anchor="middle">π = 0.6652</text>
<text x="150.0" y="89.6" class="fig-small" text-anchor="middle">log = -0.4076</text>
<text x="150.0" y="218.0" class="fig-small" text-anchor="middle">position 0</text>
<rect x="236.0" y="146.7" width="68.0" height="53.3" class="fill-a"/>
<text x="270.0" y="128.7" class="fig-small fig-bold" text-anchor="middle">π = 0.3333</text>
<text x="270.0" y="142.7" class="fig-small" text-anchor="middle">log = -1.0986</text>
<text x="270.0" y="218.0" class="fig-small" text-anchor="middle">position 1</text>
<rect x="356.0" y="65.0" width="68.0" height="135.0" class="fill-a"/>
<text x="390.0" y="47.0" class="fig-small fig-bold" text-anchor="middle">π = 0.8438</text>
<text x="390.0" y="61.0" class="fig-small" text-anchor="middle">log = -0.1698</text>
<text x="390.0" y="218.0" class="fig-small" text-anchor="middle">position 2</text>
<rect x="476.0" y="120.0" width="68.0" height="80.0" class="fill-sunk"/>
<text x="510.0" y="114.0" class="fig-small" text-anchor="middle">masked out</text>
<text x="510.0" y="218.0" class="fig-small" text-anchor="middle">position 3 (padding)</text>
<text x="330.0" y="250.0" class="fig-small" text-anchor="middle">sum of the three log-probs = -1.6761, so the whole response has probability 0.1871</text>
</svg>
Figure 1. The probability the model gave to each token it wrote. Notice that the padded position is dropped from the sum, and that the response's probability, 0.1871, is the product of the three bars.
```

In numpy, with logits of shape `(B, T, V)` and tokens of shape `(B, T)`:

```python
log_norm = logsumexp(logits, axis=-1)                          # (B, T)
chosen = np.take_along_axis(logits, tokens[..., None], axis=-1)[..., 0]
token_logps = chosen - log_norm                                # (B, T)
seq_logps = np.where(mask, token_logps, 0.0).sum(axis=-1)      # (B,)
```

> [!WARNING]
> Multiplying by the mask does not remove padding. A padded position can hold $-\infty$, and $0 \times (-\infty)$ is `nan` in floating point. Select with `np.where(mask, token_logps, 0.0)` instead.

## Maximize reward alone, and watch it break

Suppose you ask only for the highest expected reward. The best policy then puts all its probability on the single top-scoring response. Take three responses with reference probabilities $\pi_{\text{ref}} = (0.5, 0.3, 0.2)$ and rewards $r = (1, 0, 2)$. The reward-only policy writes response 3 every time, for an expected reward of 2. But the reference writes that response only 20% of the time, so the model has collapsed onto whatever the reward happens to overrate. A learned reward has errors, and this policy finds the largest one.

The standard fix is a leash. The *reference policy* $\pi_{\text{ref}}$ is the model before this round of training, usually after supervised fine-tuning. You raise the expected reward while charging for every step away from it:

$$
J(\pi) = \mathbb{E}_{y \sim \pi}\big[r(x, y)\big] - \beta\, \mathrm{KL}\big(\pi(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\big)
$$

Read it aloud: the average reward of the responses $\pi$ writes, minus $\beta$ times how far $\pi$ has moved from the reference. The *KL coefficient* $\beta > 0$ is the length of the leash. The KL runs from $\pi$ to $\pi_{\text{ref}}$, called the *reverse KL*, so it charges heavily for any response the reference would almost never write.

With $\beta = 1$, the reward-only policy scores $J = 2 - \log 5 = 0.39$ in the example, since its KL is $\log(1/0.2) = 1.61$. The reference itself scores $J = 0.9$, because its KL is 0 and its expected reward is $0.5 + 0 + 0.4$. Chasing reward without the leash loses.

For a small set of responses you can write $J$ directly:

```python
kl = np.sum(np.where(p > 0, p * (np.log(p) - ref_logps), 0.0))
J = np.sum(p * rewards) - beta * kl
```

## Solve the leash in closed form

The objective has an exact solution. The best policy starts from the reference and multiplies each response's probability by a boost that grows with its reward:

$$
\pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x)\, \exp\!\Big(\frac{r(x, y)}{\beta}\Big),
\qquad
Z(x) = \sum_{y} \pi_{\text{ref}}(y \mid x) \exp\!\Big(\frac{r(x, y)}{\beta}\Big)
$$

Read it aloud: multiply each response's reference probability by $\exp(r / \beta)$, and divide by the total $Z(x)$ so the result sums to one. In log space it is a softmax:

```python
z = ref_logps + rewards / beta
pi_star = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
```

Work the example with $\beta = 1$. The boosted weights are $0.5 e^1 = 1.3591$, $0.3 e^0 = 0.3$, and $0.2 e^2 = 1.4778$, so $Z = 3.1370$ and $\pi^* = (0.4333, 0.0956, 0.4711)$. Figure 2 compares it with the reference. Expected reward under $\pi^*$ is 1.3755 and the KL is 0.2322, so $J(\pi^*) = 1.1433$, well above the reference's 0.9 and the reward-only policy's 0.39.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Probabilities of three responses under the reference, the optimal policy at beta 1, and the optimal policy at beta 0.5">
<line x1="80.0" y1="200.0" x2="570.0" y2="200.0" class="stroke-grid"/>
<text x="72.0" y="204.0" class="fig-small" text-anchor="end">0</text>
<line x1="80.0" y1="160.0" x2="570.0" y2="160.0" class="stroke-grid"/>
<text x="72.0" y="164.0" class="fig-small" text-anchor="end">0.2</text>
<line x1="80.0" y1="120.0" x2="570.0" y2="120.0" class="stroke-grid"/>
<text x="72.0" y="124.0" class="fig-small" text-anchor="end">0.4</text>
<line x1="80.0" y1="80.0" x2="570.0" y2="80.0" class="stroke-grid"/>
<text x="72.0" y="84.0" class="fig-small" text-anchor="end">0.6</text>
<line x1="80.0" y1="40.0" x2="570.0" y2="40.0" class="stroke-grid"/>
<text x="72.0" y="44.0" class="fig-small" text-anchor="end">0.8</text>
<rect x="95.7" y="100.0" width="40.0" height="100.0" class="fill-sunk"/>
<rect x="95.7" y="100.0" width="40" height="100.0" class="stroke-muted" fill="none"/>
<text x="115.7" y="95.0" class="fig-small" text-anchor="middle">0.500</text>
<rect x="141.7" y="113.3" width="40.0" height="86.7" class="fill-a"/>
<text x="161.7" y="108.3" class="fig-small" text-anchor="middle">0.433</text>
<rect x="187.7" y="150.5" width="40.0" height="49.5" class="fill-b"/>
<text x="207.7" y="145.5" class="fig-small" text-anchor="middle">0.248</text>
<text x="161.7" y="220.0" class="fig-small" text-anchor="middle">response 1, reward 1</text>
<rect x="259.0" y="140.0" width="40.0" height="60.0" class="fill-sunk"/>
<rect x="259.0" y="140.0" width="40" height="60.0" class="stroke-muted" fill="none"/>
<text x="279.0" y="135.0" class="fig-small" text-anchor="middle">0.300</text>
<rect x="305.0" y="180.9" width="40.0" height="19.1" class="fill-a"/>
<text x="325.0" y="175.9" class="fig-small" text-anchor="middle">0.096</text>
<rect x="351.0" y="196.0" width="40.0" height="4.0" class="fill-b"/>
<text x="371.0" y="191.0" class="fig-small" text-anchor="middle">0.020</text>
<text x="325.0" y="220.0" class="fig-small" text-anchor="middle">response 2, reward 0</text>
<rect x="422.2" y="160.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="422.2" y="160.0" width="40" height="40.0" class="stroke-muted" fill="none"/>
<text x="442.2" y="155.0" class="fig-small" text-anchor="middle">0.200</text>
<rect x="468.2" y="105.8" width="40.0" height="94.2" class="fill-a"/>
<text x="488.2" y="100.8" class="fig-small" text-anchor="middle">0.471</text>
<rect x="514.2" y="53.6" width="40.0" height="146.4" class="fill-b"/>
<text x="534.2" y="48.6" class="fig-small" text-anchor="middle">0.732</text>
<text x="488.2" y="220.0" class="fig-small" text-anchor="middle">response 3, reward 2</text>
<rect x="130.0" y="246.0" width="12.0" height="12.0" class="fill-sunk"/>
<text x="148.0" y="257.0" class="fig-small" text-anchor="start">reference</text>
<rect x="250.0" y="246.0" width="12.0" height="12.0" class="fill-a"/>
<text x="268.0" y="257.0" class="fig-small" text-anchor="start">π* at β = 1</text>
<rect x="400.0" y="246.0" width="12.0" height="12.0" class="fill-b"/>
<text x="418.0" y="257.0" class="fig-small" text-anchor="start">π* at β = 0.5</text>
</svg>
Figure 2. The reference and two optimal policies. Notice that the higher the reward, the more probability the optimum moves onto a response, and that the shorter leash (β = 0.5) moves far more of it.
```

Here is why it is the best. Expand $\mathrm{KL}(\pi \,\|\, \pi^*)$ using the formula for $\pi^*$ and collect terms. You get $J(\pi) = \beta \log Z(x) - \beta\, \mathrm{KL}(\pi \,\|\, \pi^*)$. The first term does not depend on $\pi$, and the KL is zero only at $\pi = \pi^*$. So $\pi^*$ is the unique maximiser, and its value is $J(\pi^*) = \beta \log Z = \log 3.1370 = 1.1433$, matching the number above.

The leash length shows in the limits. When $\beta$ is huge, $r / \beta$ is near zero and $\pi^* = \pi_{\text{ref}}$. When $\beta$ is tiny, the best response's boost swamps every other. At $\beta = 0.5$, the example gives $\pi^* = (0.2477, 0.0201, 0.7322)$ at a KL of 0.7218. This is the maximum-entropy result in a different coat, with $\beta$ as the temperature and $\pi_{\text{ref}}$ as the prior.

> [!KEY]
> The KL-regularized optimum is $\pi^* \propto \pi_{\text{ref}} \exp(r / \beta)$, and its value is $\beta \log Z$. A real model cannot compute $Z$, because it sums over every possible response, so later chapters sample or cancel it.

## Measure best-of-n

*Best-of-n* sampling draws $n$ responses from $\pi_{\text{ref}}$ and returns the one with the highest reward. It needs no training. Sort the responses by reward, and let $F_i$ be the reference probability of the $i$-th response and everything below it. The best of $n$ draws is at or below response $i$ exactly when all $n$ draws are, which happens with probability $F_i^n$. So

$$
\pi_{\text{BoN}}(y_{(i)}) = F_i^{\,n} - F_{i-1}^{\,n}
$$

Read it aloud: the chance the best draw is at or below response $i$, minus the chance it is at or below the one before. Here $y_{(i)}$ is the $i$-th response in reward order, $F_0 = 0$, and rewards are distinct. In numpy:

```python
order = np.argsort(rewards)
F = np.cumsum(ref_probs[order])
out[order] = F**n - np.concatenate([[0.0], F[:-1]])**n
```

Take four responses with $\pi_{\text{ref}} = (0.1, 0.4, 0.2, 0.3)$, rewards $(2, -1, 0.5, 1)$, and $n = 2$. In reward order the cumulative mass is $F = (0.4, 0.6, 0.9, 1.0)$, so the best-of-2 probabilities are $0.16$, $0.20$, $0.45$, and $0.19$. In the original order that is $(0.19, 0.16, 0.20, 0.45)$, and expected reward rises from 0.2 to 0.77. The exact KL from the reference is 0.1578.

A widely quoted formula says the KL of best-of-n is $\log n - (n-1)/n$, which here gives 0.1931. It is exact when the reward is continuous, so no response has probability of its own. On a finite set, Beirami and colleagues (2024) showed it is only an upper bound. Figure 3 shows the gap: tiny when the reference spreads over many responses, large when one top response holds half the probability.

```figure
<svg viewBox="0 0 600 280" role="img" aria-label="Exact best-of-n KL for a spread reference and a peaked reference against the bound log n minus (n minus 1) over n, for n from 1 to 64">
<line x1="80.0" y1="230.0" x2="560.0" y2="230.0" class="stroke-grid"/>
<text x="72.0" y="234.0" class="fig-small" text-anchor="end">0</text>
<line x1="80.0" y1="172.9" x2="560.0" y2="172.9" class="stroke-grid"/>
<text x="72.0" y="176.9" class="fig-small" text-anchor="end">1</text>
<line x1="80.0" y1="115.7" x2="560.0" y2="115.7" class="stroke-grid"/>
<text x="72.0" y="119.7" class="fig-small" text-anchor="end">2</text>
<line x1="80.0" y1="58.6" x2="560.0" y2="58.6" class="stroke-grid"/>
<text x="72.0" y="62.6" class="fig-small" text-anchor="end">3</text>
<line x1="80.0" y1="230.0" x2="80.0" y2="235.0" class="stroke-muted"/>
<text x="80.0" y="250.0" class="fig-small" text-anchor="middle">1</text>
<line x1="194.3" y1="230.0" x2="194.3" y2="235.0" class="stroke-muted"/>
<text x="194.3" y="250.0" class="fig-small" text-anchor="middle">16</text>
<line x1="316.2" y1="230.0" x2="316.2" y2="235.0" class="stroke-muted"/>
<text x="316.2" y="250.0" class="fig-small" text-anchor="middle">32</text>
<line x1="438.1" y1="230.0" x2="438.1" y2="235.0" class="stroke-muted"/>
<text x="438.1" y="250.0" class="fig-small" text-anchor="middle">48</text>
<line x1="560.0" y1="230.0" x2="560.0" y2="235.0" class="stroke-muted"/>
<text x="560.0" y="250.0" class="fig-small" text-anchor="middle">64</text>
<line x1="80.0" y1="230.0" x2="560.0" y2="230.0" class="stroke-muted"/>
<polyline points="80.0,230.0 87.6,219.0 95.2,205.3 102.9,193.6 110.5,183.7 118.1,175.2 125.7,167.8 133.3,161.2 141.0,155.2 148.6,149.9 156.2,144.9 163.8,140.4 171.4,136.2 179.0,132.3 186.7,128.6 194.3,125.1 201.9,121.9 209.5,118.8 217.1,115.9 224.8,113.1 232.4,110.5 240.0,107.9 247.6,105.5 255.2,103.2 262.9,100.9 270.5,98.8 278.1,96.7 285.7,94.7 293.3,92.8 301.0,90.9 308.6,89.1 316.2,87.3 323.8,85.6 331.4,84.0 339.0,82.4 346.7,80.8 354.3,79.3 361.9,77.8 369.5,76.3 377.1,74.9 384.8,73.5 392.4,72.2 400.0,70.9 407.6,69.6 415.2,68.4 422.9,67.1 430.5,65.9 438.1,64.7 445.7,63.6 453.3,62.5 461.0,61.4 468.6,60.3 476.2,59.2 483.8,58.1 491.4,57.1 499.0,56.1 506.7,55.1 514.3,54.1 521.9,53.2 529.5,52.2 537.1,51.3 544.8,50.4 552.4,49.5 560.0,48.6" class="stroke-a" fill="none"/>
<polyline points="80.0,230.0 87.6,219.8 95.2,208.8 102.9,201.5 110.5,196.9 118.1,194.1 125.7,192.5 133.3,191.6 141.0,191.1 148.6,190.8 156.2,190.6 163.8,190.5 171.4,190.4 179.0,190.4 186.7,190.4 194.3,190.4 201.9,190.4 209.5,190.4 217.1,190.4 224.8,190.4 232.4,190.4 240.0,190.4 247.6,190.4 255.2,190.4 262.9,190.4 270.5,190.4 278.1,190.4 285.7,190.4 293.3,190.4 301.0,190.4 308.6,190.4 316.2,190.4 323.8,190.4 331.4,190.4 339.0,190.4 346.7,190.4 354.3,190.4 361.9,190.4 369.5,190.4 377.1,190.4 384.8,190.4 392.4,190.4 400.0,190.4 407.6,190.4 415.2,190.4 422.9,190.4 430.5,190.4 438.1,190.4 445.7,190.4 453.3,190.4 461.0,190.4 468.6,190.4 476.2,190.4 483.8,190.4 491.4,190.4 499.0,190.4 506.7,190.4 514.3,190.4 521.9,190.4 529.5,190.4 537.1,190.4 544.8,190.4 552.4,190.4 560.0,190.4" class="stroke-b" fill="none"/>
<polyline points="80.0,230.0 87.6,219.0 95.2,205.3 102.9,193.6 110.5,183.7 118.1,175.2 125.7,167.8 133.3,161.2 141.0,155.2 148.6,149.9 156.2,144.9 163.8,140.4 171.4,136.2 179.0,132.3 186.7,128.6 194.3,125.1 201.9,121.9 209.5,118.8 217.1,115.9 224.8,113.1 232.4,110.4 240.0,107.9 247.6,105.5 255.2,103.2 262.9,100.9 270.5,98.8 278.1,96.7 285.7,94.7 293.3,92.8 301.0,90.9 308.6,89.1 316.2,87.3 323.8,85.6 331.4,84.0 339.0,82.3 346.7,80.8 354.3,79.3 361.9,77.8 369.5,76.3 377.1,74.9 384.8,73.5 392.4,72.2 400.0,70.9 407.6,69.6 415.2,68.3 422.9,67.1 430.5,65.9 438.1,64.7 445.7,63.6 453.3,62.5 461.0,61.3 468.6,60.3 476.2,59.2 483.8,58.1 491.4,57.1 499.0,56.1 506.7,55.1 514.3,54.1 521.9,53.2 529.5,52.2 537.1,51.3 544.8,50.4 552.4,49.5 560.0,48.6" class="stroke-ink dashed" fill="none"/>
<text x="555.0" y="32.9" class="fig-small fig-bold" text-anchor="end">dashed: the bound log n − (n−1)/n</text>
<text x="555.0" y="48.9" class="fig-small fig-bold" text-anchor="end">blue: 1000 spread-out responses (on the bound)</text>
<text x="270.5" y="158.6" class="fig-small fig-bold" text-anchor="start">half the mass on the best response</text>
<text x="320.0" y="274.0" class="fig-small" text-anchor="middle">n, the number of draws</text>
<text x="80.0" y="20.0" class="fig-small" text-anchor="start">KL in nats</text>
</svg>
Figure 3. Exact best-of-n KL against the bound. Notice that the spread reference hugs the bound, while the peaked one stays far below it: at n = 16 the exact KL is 0.6930 against a bound of 1.8351.
```

> [!KEY]
> $\log n - (n-1)/n$ is the exact best-of-n KL for a continuous reward and an upper bound on any finite set. It is loose when a few likely responses sit near the top of the ranking.

> [!RECAP]
> - A response's log-probability is the sum of its token log-probabilities, selected with `np.where` under the mask.
> - Maximizing reward alone collapses onto the reward's errors, so you subtract $\beta$ times the reverse KL to the reference.
> - $\pi^* = \operatorname{softmax}(\log \pi_{\text{ref}} + r / \beta)$, and $J(\pi^*) = \beta \log Z$.
> - Best-of-n gives response $i$ in reward order $F_i^n - F_{i-1}^n$, and its KL is at most $\log n - (n-1)/n$.

## Build it step by step

The checks test the functions in this order. The starter provides `ToyLM`, a tabular policy whose logits depend on the position and the previous token, and `all_sequences`, which lists every response of a given length.

1. **`token_logprobs(logits, tokens)`.** Takes `(B, T, V)` logits and `(B, T)` tokens, and returns `(B, T)`. Subtract the log-sum-exp, then gather with `np.take_along_axis`. The common mistake is exponentiating without subtracting the maximum. The check adds 1000 to every logit.
2. **`sequence_logprob(token_logps, mask)`.** Returns `(B,)`. Use `np.where`, then sum over the last axis. The common mistake is multiplying by the mask.
3. **`kl_optimal_policy(ref_logps, rewards, beta)`.** Returns `(N,)`, a softmax of `ref_logps + rewards / beta`. The common mistake is skipping the max subtraction, which overflows at small `beta`.
4. **`kl_regularised_objective(probs, ref_logps, rewards, beta)`.** Returns a float. The common mistake is letting a zero probability produce `nan`.
5. **`best_of_n_probs(ref_probs, rewards, n)`.** Returns `(N,)` in the input order, not the sorted order.
6. **`best_of_n_kl(ref_probs, rewards, n)`.** Returns a float. Skip entries with zero best-of-n probability.

## Where this breaks

**A policy that cannot represent $\pi^*$.** The closed form ranges over every distribution on responses. A real network reaches, at best, the nearest policy it can represent.

**A reward that is wrong somewhere.** If the reward overrates an odd response, $\pi^*$ boosts it by $\exp(r / \beta)$, and a small $\beta$ makes that boost enormous. [Reward models from preferences](/lessons/050-reward-models) returns to this.

**Quoting the best-of-n formula as exact.** Language models often repeat the same short answer, and all the repeats act as one response with a large probability. The formula then overstates best-of-n's KL.

## Further reading

- [Ziegler et al. (2019), Fine-tuning language models from human preferences](https://arxiv.org/abs/1909.08593). Sets up the reward-minus-KL objective for a language model.
- [Korbak, Perez, and Buckley (2022), RL with KL penalties is better viewed as Bayesian inference](https://arxiv.org/abs/2205.11275). Reads $\pi^*$ as a posterior.
- [Rafailov et al. (2023), Direct preference optimization](https://arxiv.org/abs/2305.18290). Derives the closed-form optimum, then inverts it to train without sampling.
- [Rafailov, Hejna, Park, and Finn (2024), From r to Q*](https://arxiv.org/abs/2404.12358). Solves the same optimum token by token with the soft Bellman equations.
- [Gao, Schulman, and Hilton (2022), Scaling laws for reward model overoptimization](https://arxiv.org/abs/2210.10760). Measures how a proxy reward and a true reward come apart under best-of-n.
- [Beirami et al. (2024), Theoretical guarantees on the best-of-n alignment policy](https://arxiv.org/abs/2401.01879). Proves that $\log n - (n-1)/n$ is an upper bound.

Open questions researchers are working on:

- Is reverse KL to a fixed reference the right leash, or should the anchor move as training goes on?

## The exercise

You build the scoring and the exact solutions from this chapter: `token_logprobs`, `sequence_logprob`, `kl_optimal_policy`, `kl_regularised_objective`, `best_of_n_probs`, and `best_of_n_kl`.

The checks compare log-probabilities with the worked example, with logits shifted by 1000, and with padding that holds $-\infty$ and `nan`. All 64 toy responses must sum to probability one. $\pi^*$ must match the worked example and both limits of $\beta$, and $J(\pi^*) = \beta \log Z$ must beat 20 perturbed policies. Best-of-n must match 100,000 sampled draws, and its KL must stay below $\log n - (n-1)/n$ for every $n$ up to 64.
