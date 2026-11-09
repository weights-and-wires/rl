---
title: Reward models from preferences
slug: 051-reward-models
part: "Part 9 — RL for language models"
summary: Learn a reward from pairwise preferences with Bradley-Terry, combine several models so they overrate less, and see process rewards as value changes.
minutes: 40
gpu: false
objectives:
  - Fit a Bradley-Terry reward model and show that it recovers a hidden reward up to a constant.
  - Explain why optimizing one learned reward overrates its favorite answer, and how a conservative ensemble reduces that.
  - Compute exact prefix values on a toy task, and turn an outcome reward into per-step process rewards.
  - Show that process rewards telescope to the outcome reward minus a constant.
prereqs:
  - 050-language-models-as-policies
  - 018-maximization-bias
exercise: 051-reward-models
---

# Reward models from preferences

The previous chapter assumed a reward $r(x, y)$ for every response. For helpfulness or tone, nobody can write that function down. People can do something easier: read two answers and say which one is better. A *reward model* turns thousands of such comparisons into a function that scores any answer, so reinforcement learning has something to maximize.

A learned reward is only a guess, and the policy you train against it searches for the answers it overrates. This chapter shows how to fit a reward from comparisons, why one fitted reward is dangerous, and how to push back by training several and trusting only what they agree on. It ends by asking whether to score a whole answer or each reasoning step.

## What you already know

- A response $y$ to a prompt $x$ has log-probability equal to the sum of its token log-probabilities.
- The sigmoid $\sigma(z) = 1 / (1 + e^{-z})$ turns any number into a probability.
- The maximum of several noisy estimates is biased upward.
- The value $V^\pi(s)$ of a state is the expected return from it under $\pi$.

## Learn a reward from comparisons

Think of chess ratings. You never observe a player's skill, only who beat whom. A rating gives each player a number so that the gap between two numbers predicts the winner. The Bradley-Terry model, from 1952, does the same for responses. Each comparison is a *chosen* response $y_w$ and a *rejected* response $y_l$ for one prompt, and the model says the chance a person prefers $y_w$ is

$$
p(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)
$$

Read it aloud: take the reward gap, chosen minus rejected, and squash it into a probability. Figure 1 shows the squash. A gap of 0 gives even odds, a gap of 1 gives 0.73, and a gap of 3 gives 0.95.

```figure
<svg viewBox="0 0 600 260" role="img" aria-label="Sigmoid of the reward gap, rising from 0 to 1, with marked points at gaps 0, 1, and 3 giving probabilities 0.5, 0.73, and 0.95">
<line x1="80.0" y1="210.0" x2="560.0" y2="210.0" class="stroke-grid"/>
<text x="72.0" y="214.0" class="fig-small" text-anchor="end">0</text>
<line x1="80.0" y1="165.0" x2="560.0" y2="165.0" class="stroke-grid"/>
<text x="72.0" y="169.0" class="fig-small" text-anchor="end">0.25</text>
<line x1="80.0" y1="120.0" x2="560.0" y2="120.0" class="stroke-grid"/>
<text x="72.0" y="124.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="80.0" y1="75.0" x2="560.0" y2="75.0" class="stroke-grid"/>
<text x="72.0" y="79.0" class="fig-small" text-anchor="end">0.75</text>
<line x1="80.0" y1="30.0" x2="560.0" y2="30.0" class="stroke-grid"/>
<text x="72.0" y="34.0" class="fig-small" text-anchor="end">1</text>
<line x1="80.0" y1="210.0" x2="80.0" y2="215.0" class="stroke-muted"/>
<text x="80.0" y="230.0" class="fig-small" text-anchor="middle">-6</text>
<line x1="200.0" y1="210.0" x2="200.0" y2="215.0" class="stroke-muted"/>
<text x="200.0" y="230.0" class="fig-small" text-anchor="middle">-3</text>
<line x1="320.0" y1="210.0" x2="320.0" y2="215.0" class="stroke-muted"/>
<text x="320.0" y="230.0" class="fig-small" text-anchor="middle">0</text>
<line x1="440.0" y1="210.0" x2="440.0" y2="215.0" class="stroke-muted"/>
<text x="440.0" y="230.0" class="fig-small" text-anchor="middle">3</text>
<line x1="560.0" y1="210.0" x2="560.0" y2="215.0" class="stroke-muted"/>
<text x="560.0" y="230.0" class="fig-small" text-anchor="middle">6</text>
<line x1="80.0" y1="210.0" x2="560.0" y2="210.0" class="stroke-muted"/>
<polyline points="80.0,209.6 84.0,209.5 88.0,209.5 92.0,209.4 96.0,209.3 100.0,209.3 104.0,209.2 108.0,209.1 112.0,209.0 116.0,208.9 120.0,208.8 124.0,208.7 128.0,208.5 132.0,208.4 136.0,208.2 140.0,208.0 144.0,207.8 148.0,207.6 152.0,207.3 156.0,207.1 160.0,206.8 164.0,206.4 168.0,206.1 172.0,205.7 176.0,205.2 180.0,204.7 184.0,204.2 188.0,203.6 192.0,203.0 196.0,202.2 200.0,201.5 204.0,200.6 208.0,199.7 212.0,198.7 216.0,197.6 220.0,196.3 224.0,195.0 228.0,193.6 232.0,192.0 236.0,190.4 240.0,188.5 244.0,186.6 248.0,184.5 252.0,182.2 256.0,179.8 260.0,177.2 264.0,174.4 268.0,171.5 272.0,168.3 276.0,165.0 280.0,161.6 284.0,158.0 288.0,154.2 292.0,150.3 296.0,146.2 300.0,142.0 304.0,137.8 308.0,133.4 312.0,129.0 316.0,124.5 320.0,120.0 324.0,115.5 328.0,111.0 332.0,106.6 336.0,102.2 340.0,98.0 344.0,93.8 348.0,89.7 352.0,85.8 356.0,82.0 360.0,78.4 364.0,75.0 368.0,71.7 372.0,68.5 376.0,65.6 380.0,62.8 384.0,60.2 388.0,57.8 392.0,55.5 396.0,53.4 400.0,51.5 404.0,49.6 408.0,48.0 412.0,46.4 416.0,45.0 420.0,43.7 424.0,42.4 428.0,41.3 432.0,40.3 436.0,39.4 440.0,38.5 444.0,37.8 448.0,37.0 452.0,36.4 456.0,35.8 460.0,35.3 464.0,34.8 468.0,34.3 472.0,33.9 476.0,33.6 480.0,33.2 484.0,32.9 488.0,32.7 492.0,32.4 496.0,32.2 500.0,32.0 504.0,31.8 508.0,31.6 512.0,31.5 516.0,31.3 520.0,31.2 524.0,31.1 528.0,31.0 532.0,30.9 536.0,30.8 540.0,30.7 544.0,30.7 548.0,30.6 552.0,30.5 556.0,30.5 560.0,30.4" class="stroke-a" fill="none"/>
<circle cx="320.0" cy="120.0" r="5" class="fill-b"/>
<text x="328.0" y="138.0" class="fig-small fig-bold" text-anchor="start">gap 0: 0.500</text>
<circle cx="360.0" cy="78.4" r="5" class="fill-b"/>
<text x="368.0" y="70.4" class="fig-small fig-bold" text-anchor="start">gap 1: 0.731</text>
<circle cx="440.0" cy="38.5" r="5" class="fill-b"/>
<text x="448.0" y="30.5" class="fig-small fig-bold" text-anchor="start">gap 3: 0.953</text>
<text x="320.0" y="252.0" class="fig-small" text-anchor="middle">reward gap: r(chosen) − r(rejected)</text>
<text x="80.0" y="20.0" class="fig-small" text-anchor="start">chance chosen is preferred</text>
</svg>
Figure 1. The Bradley-Terry preference probability as a function of the reward gap. Notice that only gaps matter, and that the curve flattens: a gap of 3 already gives 0.95, so a pair that is ranked right by a wide margin has almost no gradient left.
```

You fit the parameters $\phi$ by maximum likelihood, which means minimizing minus the log-probability the model gave to the choices people made:

$$
\mathcal{L}(\phi) = \frac{1}{n} \sum_{i=1}^{n} -\log \sigma(\Delta_i),
\qquad
\Delta_i = r_\phi(x_i, y_{w,i}) - r_\phi(x_i, y_{l,i})
$$

Read it aloud: for each pair, find the gap, take minus the log of its sigmoid, and average. The function $-\log \sigma(z)$ is *softplus* of $-z$, and torch computes it without overflow:

```python
loss = torch.nn.functional.softplus(-(r_chosen - r_rejected)).mean()
```

The derivative of $-\log \sigma(z)$ is $-(1 - \sigma(z))$, so each pair pushes its chosen reward up and its rejected reward down by $(1 - \sigma(\Delta_i)) / n$. A pair the model already ranks right barely moves anything, and a pair it ranks wrong pushes with nearly full force.

Take one pair with $r_w = 2$ and $r_l = 1$. The gap is 1, so $p = \sigma(1) = 0.7311$ and the loss is $-\log 0.7311 = 0.3133$. The gradient is $-0.2689$ for $r_w$ and $+0.2689$ for $r_l$. Add 100 to both rewards and every one of those numbers stays the same, because the loss sees only gaps. So rewards are fixed only up to a constant per prompt: compare rewards for the same prompt, never across prompts.

> [!WARNING]
> `torch.log(torch.sigmoid(z))` underflows to $-\infty$ once $z$ is below about $-100$ in float32. Use softplus. The checks include a gap of $-800$.

## Fit it and recover the truth

A reward model for a language model is usually the fine-tuned model itself, with its next-token head swapped for a linear layer that outputs one number at the last token. The exercise uses a linear reward on features, $r_\phi(y) = x(y)^\top \phi$, where each (position, token) pair has its own weight. The fitting loop is plain Adam on the loss above:

```python
phi = torch.zeros(D, dtype=torch.float64, requires_grad=True)
opt = torch.optim.Adam([phi], lr=lr)
for _ in range(steps):
    opt.zero_grad()
    bradley_terry_loss(xc @ phi, xr @ phi).backward()
    opt.step()
```

When does this recover the true reward? Suppose preferences really come from Bradley-Terry with a true reward $r^\star$, and your model family contains it. With unlimited data, the expected loss is a log loss on the preference probabilities, and a log loss is smallest when the predicted probabilities equal the true ones. Every gap $r_\phi(y) - r_\phi(y')$ then matches $r^\star(y) - r^\star(y')$, so $r_\phi = r^\star + c$ for one constant $c$. Every response must be linked to every other by some chain of comparisons, or each linked group gets its own constant.

> [!KEY]
> A Bradley-Terry reward model is a classifier on reward gaps. With enough data and a model that can represent the truth, it recovers the true reward up to one constant per prompt.

## Trust several models, not one

Here is the problem with optimizing a learned reward. The fitted reward is the true reward plus an error, and the error is largest on responses that appeared in few comparisons. When you pick the response with the highest fitted reward, you pick, in part, the one with the most positive error. This is maximization bias, and it causes *overoptimization*: the proxy reward keeps rising while the true reward stalls or falls (Gao, Schulman, and Hilton, 2022).

One response is to train $K$ reward models on resampled data and combine them. Where they disagree, the error is probably large. Coste and colleagues (2023) compare two conservative combinations with the mean:

$$
r_{\text{WCO}}(y) = \min_k r_k(y),
\qquad
r_{\text{UWO}}(y) = \bar r(y) - \lambda\, \frac{1}{K} \sum_k \big(r_k(y) - \bar r(y)\big)^2
$$

Read it aloud: *worst-case optimization* (WCO) trusts the most pessimistic model. *Uncertainty-weighted optimization* (UWO) takes the mean $\bar r(y)$ and subtracts $\lambda$ times the variance across models, taken over $K$, not $K - 1$. With scores of shape `(K, N)`, that is `scores.min(axis=0)` and `scores.mean(axis=0) - lam * scores.var(axis=0)`.

Three models score three responses:

| Model | Response 1 | Response 2 | Response 3 |
|---|---|---|---|
| 1 | 1.0 | 2.0 | 0.5 |
| 2 | 3.0 | 2.0 | 0.5 |
| 3 | 2.0 | 2.0 | 2.0 |

The means are $(2, 2, 1)$, so responses 1 and 2 tie. WCO gives $(1, 2, 0.5)$ and picks response 2, which every model agrees on. The variances are $(2/3, 0, 1/2)$, so UWO with $\lambda = 0.5$ gives $(1.667, 2, 0.75)$ and also picks response 2. Figure 2 draws the three rewards.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Scores of three reward models for three responses, with the mean and the worst-case value; responses 1 and 2 tie on the mean but only response 2 keeps a high worst case">
<line x1="80.0" y1="220.0" x2="570.0" y2="220.0" class="stroke-grid"/>
<text x="72.0" y="224.0" class="fig-small" text-anchor="end">0</text>
<line x1="80.0" y1="163.8" x2="570.0" y2="163.8" class="stroke-grid"/>
<text x="72.0" y="167.8" class="fig-small" text-anchor="end">1</text>
<line x1="80.0" y1="107.5" x2="570.0" y2="107.5" class="stroke-grid"/>
<text x="72.0" y="111.5" class="fig-small" text-anchor="end">2</text>
<line x1="80.0" y1="51.2" x2="570.0" y2="51.2" class="stroke-grid"/>
<text x="72.0" y="55.2" class="fig-small" text-anchor="end">3</text>
<rect x="101.7" y="163.8" width="24.0" height="56.2" class="fill-soft-a"/>
<rect x="101.7" y="163.8" width="24" height="56.2" class="stroke-a" fill="none"/>
<rect x="131.7" y="51.2" width="24.0" height="168.8" class="fill-soft-a"/>
<rect x="131.7" y="51.2" width="24" height="168.8" class="stroke-a" fill="none"/>
<rect x="161.7" y="107.5" width="24.0" height="112.5" class="fill-soft-a"/>
<rect x="161.7" y="107.5" width="24" height="112.5" class="stroke-a" fill="none"/>
<line x1="97.7" y1="107.5" x2="195.7" y2="107.5" class="stroke-c"/>
<line x1="97.7" y1="163.8" x2="195.7" y2="163.8" class="stroke-b dashed"/>
<circle cx="201.7" cy="107.5" r="4" class="fill-c"/>
<circle cx="201.7" cy="163.8" r="4" class="fill-b"/>
<text x="161.7" y="238.0" class="fig-small" text-anchor="middle">response 1</text>
<text x="209.7" y="104.5" class="fig-small" text-anchor="start">2</text>
<text x="209.7" y="177.8" class="fig-small" text-anchor="start">1</text>
<rect x="265.0" y="107.5" width="24.0" height="112.5" class="fill-soft-a"/>
<rect x="265.0" y="107.5" width="24" height="112.5" class="stroke-a" fill="none"/>
<rect x="295.0" y="107.5" width="24.0" height="112.5" class="fill-soft-a"/>
<rect x="295.0" y="107.5" width="24" height="112.5" class="stroke-a" fill="none"/>
<rect x="325.0" y="107.5" width="24.0" height="112.5" class="fill-soft-a"/>
<rect x="325.0" y="107.5" width="24" height="112.5" class="stroke-a" fill="none"/>
<line x1="261.0" y1="107.5" x2="359.0" y2="107.5" class="stroke-c"/>
<line x1="261.0" y1="107.5" x2="359.0" y2="107.5" class="stroke-b dashed"/>
<circle cx="365.0" cy="107.5" r="4" class="fill-c"/>
<circle cx="365.0" cy="107.5" r="4" class="fill-b"/>
<text x="325.0" y="238.0" class="fig-small" text-anchor="middle">response 2</text>
<text x="373.0" y="104.5" class="fig-small" text-anchor="start">2</text>
<text x="373.0" y="121.5" class="fig-small" text-anchor="start">2</text>
<rect x="428.2" y="191.9" width="24.0" height="28.1" class="fill-soft-a"/>
<rect x="428.2" y="191.9" width="24" height="28.1" class="stroke-a" fill="none"/>
<rect x="458.2" y="191.9" width="24.0" height="28.1" class="fill-soft-a"/>
<rect x="458.2" y="191.9" width="24" height="28.1" class="stroke-a" fill="none"/>
<rect x="488.2" y="107.5" width="24.0" height="112.5" class="fill-soft-a"/>
<rect x="488.2" y="107.5" width="24" height="112.5" class="stroke-a" fill="none"/>
<line x1="424.2" y1="163.8" x2="522.2" y2="163.8" class="stroke-c"/>
<line x1="424.2" y1="191.9" x2="522.2" y2="191.9" class="stroke-b dashed"/>
<circle cx="528.2" cy="163.8" r="4" class="fill-c"/>
<circle cx="528.2" cy="191.9" r="4" class="fill-b"/>
<text x="488.2" y="238.0" class="fig-small" text-anchor="middle">response 3</text>
<text x="536.2" y="155.8" class="fig-small" text-anchor="start">1</text>
<text x="536.2" y="205.9" class="fig-small" text-anchor="start">0.5</text>
<rect x="110.0" y="258.0" width="12.0" height="12.0" class="fill-soft-a"/>
<text x="128.0" y="269.0" class="fig-small" text-anchor="start">three models</text>
<line x1="240.0" y1="264.0" x2="262.0" y2="264.0" class="stroke-c"/>
<text x="268.0" y="269.0" class="fig-small" text-anchor="start">mean</text>
<line x1="330.0" y1="264.0" x2="352.0" y2="264.0" class="stroke-b dashed"/>
<text x="358.0" y="269.0" class="fig-small" text-anchor="start">minimum (WCO)</text>
</svg>
Figure 2. Three models score three responses. Notice that responses 1 and 2 tie on the mean at 2, but the models disagree about response 1, so its worst case is 1 while response 2 keeps a worst case of 2.
```

The exercise runs this on 40 toy prompts with 5 models each. On the ensemble's top 3 responses, the models disagree more than twice as much as on a typical response, a standard deviation of 0.74 against 0.31. At its own favorite response, a single model overrates the true reward by 1.92 on average, and WCO overrates it by 1.20. Ensembles help less than that suggests, though: Eisenstein and colleagues (2023) found that models fine-tuned from one checkpoint agree exactly where they are all wrong.

## Score the steps, not only the answer

An *outcome reward model* scores only the final answer. A *process reward model* scores every step. Lightman and colleagues (2023) found that process supervision beats outcome supervision for choosing among sampled math solutions, but human step labels are expensive. So ask what a good step reward would be. A step is good when it raises the chance that the answer comes out right, which is the *value* of the prefix.

For a toy policy you can compute that value exactly by walking back up the prefix tree, because a prefix's value is the probability-weighted average of its children's values:

```python
values[T] = rewards                                   # shape (V,) * T
for t in reversed(range(T)):
    values[t] = (np.exp(cond_logps[t]) * values[t + 1]).sum(axis=-1)
```

The step reward is the change in value that a token causes, $r^{\text{proc}}_t = V^\pi(s_{t+1}) - V^\pi(s_t)$. Take a two-token policy where $y_0$ is 0 or 1 with probability 0.5, then $(0.8, 0.2)$ after a 0 and $(0.4, 0.6)$ after a 1. Responses $(0, 1)$ and $(1, 0)$ succeed with reward 1. Then $V(0) = 0.2$, $V(1) = 0.4$, and $V(s_0) = 0.3$. For the response $(0, 1)$, the process rewards are $(0.2 - 0.3, 1 - 0.2) = (-0.1, 0.8)$. For the failure $(1, 1)$ they are $(0.1, -0.4)$.

Each row sums to the outcome reward minus 0.3, because the middle values cancel. In general the process rewards along a response sum to $r(y) - V^\pi(s_0)$: the outcome reward minus a constant. So the two rewards agree on which whole responses are better, and they differ in which tokens get the credit. The first step of $(0, 1)$ gets negative credit even though the response succeeds, because a 0 made success less likely.

> [!KEY]
> A value-based process reward is $V(s_{t+1}) - V(s_t)$. It telescopes to the outcome reward minus $V(s_0)$, so it moves credit between tokens without changing which responses are better. [Process rewards and test-time search](/lessons/057-process-rewards-and-search) estimates these values from sampled completions.

> [!RECAP]
> - The Bradley-Terry loss is `softplus(-(r_chosen - r_rejected)).mean()`, and its gradient for the chosen reward is $-(1 - \sigma(\Delta)) / n$.
> - With enough data, a fit equals the true reward plus a constant.
> - WCO is the minimum over models. UWO is the mean minus $\lambda$ times the variance over $K$.
> - Prefix values come from a backward pass, and process rewards are value differences that sum to $r(y) - V(s_0)$.

## Build it step by step

The checks test the functions in this order. The starter provides `ToyLM`, `all_sequences`, and `position_token_features`.

1. **`bradley_terry_loss(reward_chosen, reward_rejected, weights=None)`.** Takes torch tensors of shape `(n,)` and returns a scalar tensor. With `weights`, return `sum(w * loss) / sum(w)`. The common mistake is `torch.log(torch.sigmoid(...))`, which fails the gap of $-800$.
2. **`fit_reward_model(x_chosen, x_rejected, steps, lr, weight_decay, weights)`.** Takes numpy features of shape `(n, D)` and returns `phi` of shape `(D,)`. Work in float64 and start from zeros. The common mistake is forgetting `opt.zero_grad()`.
3. **`ensemble_reward(rewards, mode, lam)`.** Takes `(K, N)` and returns `(N,)`. The common mistake is using the variance over $K - 1$.
4. **`prefix_values(cond_logps, rewards)`.** Returns a list of $T + 1$ arrays, entry $t$ of shape `(V,) * t`. The common mistake is summing log-probabilities instead of probabilities.
5. **`process_rewards(tokens, values)`.** Takes `(B, T)` tokens and returns `(B, T)`. Gather the value of every prefix along each row, including the empty one and the full response, then `np.diff`.

## Where this breaks

**Reading scores across prompts.** Rewards carry an unknown constant per prompt. Normalize them, for example to mean 0 over the reference's responses, before you feed them to RL.

**Deterministic labels.** If every comparison goes to the truly better response, the maximum-likelihood gaps grow without bound. Noisy labels and weight decay keep the fit finite.

**Errors the ensemble shares.** Models trained on the same data from the same checkpoint make the same mistakes, so their variance is small exactly where it matters.

## Further reading

- [Christiano et al. (2017), Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741). Trains a reward model from pairwise comparisons and a policy against it.
- [Ouyang et al. (2022), Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155). The InstructGPT recipe, with a reward model trained on ranked responses.
- [Coste, Anwar, Kirk, and Krueger (2023), Reward model ensembles help mitigate overoptimization](https://arxiv.org/abs/2310.02743). Defines WCO and UWO and tests them with best-of-n and PPO.
- [Eisenstein et al. (2023), Helping or herding?](https://arxiv.org/abs/2312.09244). Shows that members sharing a pretraining seed share their errors.
- [Lightman et al. (2023), Let's verify step by step](https://arxiv.org/abs/2305.20050). Releases PRM800K and compares process with outcome supervision.
- [Setlur et al. (2024), Rewarding progress](https://arxiv.org/abs/2410.08146). Argues that a step's reward is its change in the chance of success.

Open questions researchers are working on:

- How much preference data does a reward model need before a policy stops finding its errors?

## The exercise

You build a reward-model toolkit. Write `bradley_terry_loss` and `fit_reward_model` to learn a linear reward from comparisons, `ensemble_reward` to combine several, and `prefix_values` and `process_rewards` for outcome and process rewards on a toy digit-sum task.

The checks test the loss at ties, at a gap of 1, at gaps of $\pm 800$, and under a shift of every reward, and check its gradient with finite differences. A fit on every weighted pair must recover the true reward plus a constant. The ensemble formulas are checked by hand, then on 240 fitted tabular models, where disagreement must peak at the top and WCO must overrate less. Finally, the checks verify exact prefix values and that process rewards telescope to $r(y) - V(s_0)$.
