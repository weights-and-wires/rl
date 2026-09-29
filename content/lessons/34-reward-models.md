---
title: Reward models
slug: 34-reward-models
part: "Part 8 — Reinforcement learning for language models"
summary: Learn a reward from pairwise preferences, combine reward models so they overrate less, and score reasoning steps instead of answers.
minutes: 60
gpu: false
objectives:
  - Fit a Bradley-Terry reward model and show that it recovers a hidden reward up to a constant.
  - Explain why optimising against one learned reward overrates its favourite answer, and how a conservative ensemble reduces that.
  - Compute exact prefix values, and the Math-Shepherd soft and hard labels that estimate them.
  - Turn an outcome reward into per-step process rewards that telescope, and show they cut policy-gradient variance.
prereqs:
  - 33-language-models-as-policies
  - 16-actor-critic
exercise: 34-reward-models
---

# Reward models

The previous lesson assumed you had a reward $r(x, y)$ for every response. For
tasks like helpfulness or tone, nobody can write that function down. People can
do something easier: read two answers and say which one is better. A *reward
model* turns thousands of those comparisons into a function that scores any
answer, so reinforcement learning has something to maximise.

A learned reward is only a guess. The policy you train against it searches for
the answers it overrates, and it finds them. This lesson shows why that happens
and one standard way to push back: train several reward models and trust only
what they agree on.

The last part of the lesson asks a different question. When the answer is a
chain of reasoning, should you score the final answer, or every step? You build
exact step values on a toy task and see that per-step rewards are the value
changes that a critic would compute.

## What you already know

- A response $y$ to a prompt $x$ has log-probability equal to the sum of its
  token log-probabilities, from *Language models as policies*.
- The sigmoid $\sigma(z) = 1 / (1 + e^{-z})$ turns any number into a
  probability.
- Subtracting a baseline that doesn't depend on the action leaves the policy
  gradient unbiased and can cut its variance, from *Baselines and
  actor-critic*.
- The maximum of several noisy estimates is biased upward, from *Maximisation
  bias and double learning*.
- The value $V^\pi(s)$ of a state is the expected return from it under $\pi$.

## Learn a reward from comparisons

Think of chess ratings. You never observe a player's skill, only who beat
whom. A rating gives each player a number so that the gap between two numbers
predicts the winner. The Bradley-Terry model, from 1952, does the same for
responses.

Each comparison is a *chosen* response $y_w$ and a *rejected* response $y_l$
for one prompt. The model says the chance a person prefers $y_w$ is

$$
p(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)
$$

Read it as: take the reward gap, chosen minus rejected, and squash it. A gap of
0 gives even odds, and a gap of 3 gives 0.95. The parameters $\phi$ of the
reward model are fitted by maximum likelihood. As a loss, over $n$ pairs:

$$
\mathcal{L}(\phi) = \frac{1}{n} \sum_{i=1}^{n} -\log \sigma(\Delta_i),
\qquad
\Delta_i = r_\phi(x_i, y_{w,i}) - r_\phi(x_i, y_{l,i})
$$

Read it as: for each pair, find the gap and take minus the log of its sigmoid,
then average. The function $-\log \sigma(z)$ is called *softplus* of $-z$, and
torch computes it without overflow:

```python
loss = torch.nn.functional.softplus(-(r_chosen - r_rejected)).mean()
```

The derivative of $-\log \sigma(z)$ is $-(1 - \sigma(z))$. So each pair pushes
its chosen reward up, and its rejected reward down, by
$(1 - \sigma(\Delta_i)) / n$. A pair the model already ranks right by a wide
margin has $\sigma(\Delta) \approx 1$ and barely moves anything. A pair it
ranks wrong pushes with nearly full force.

### Worked example

Take one pair with $r_w = 2$ and $r_l = 1$. The gap is 1, so
$p = \sigma(1) = 0.7311$ and the loss is $-\log 0.7311 = 0.3133$. The gradient
for $r_w$ is $-(1 - 0.7311) = -0.2689$, and for $r_l$ it's $+0.2689$. Add 100 to
both rewards and every one of those numbers stays the same.

That last fact matters. The loss sees only gaps, so rewards are fixed only up
to a constant per prompt. Compare rewards for the same prompt, never across
prompts.

> [!WARNING]
> `torch.log(torch.sigmoid(z))` underflows to $-\infty$ once $z$ is below
> about $-100$ in float32, or about $-745$ in float64. Use softplus. The checks
> include a gap of $-800$.

## Fit a reward model

A reward model for a language model is usually the fine-tuned model itself,
with its next-token head swapped for a linear layer that outputs one number.
The number is read at the last token of the response, so it sees the whole
answer. InstructGPT (Ouyang and colleagues, 2022) trained one on rankings of 4
to 9 responses per prompt, using every pair in each ranking.

The exercise uses a linear reward on features, $r_\phi(y) = x(y)^\top \phi$.
One feature map gives each (position, token) pair its own weight. Another gives
every response its own weight, a table. The fitting loop is the same for both:

```python
phi = torch.zeros(D, dtype=torch.float64, requires_grad=True)
opt = torch.optim.Adam([phi], lr=lr)
for _ in range(steps):
    opt.zero_grad()
    bradley_terry_loss(xc @ phi, xr @ phi).backward()
    opt.step()
```

When does this recover the true reward? Suppose preferences really come from
Bradley-Terry with a true reward $r^\star$, and your model family contains
$r^\star$. With unlimited data, the expected loss is a log loss on the
preference probabilities. A log loss is smallest when the predicted
probabilities equal the true ones. Every gap $r_\phi(y) - r_\phi(y')$ then
matches $r^\star(y) - r^\star(y')$, so $r_\phi = r^\star + c$ for one constant
$c$. You need every response to be linked to every other by some chain of
comparisons, or each linked group gets its own constant.

The exercise checks this twice. With every ordered pair weighted by how often
it would be observed, the fit matches $r^\star$ plus a constant to $10^{-3}$.
With 4,000 sampled pairs it's noisy, but its held-out loss is within 0.01 of
the loss of $r^\star$ itself.

> [!KEY]
> A Bradley-Terry reward model is a classifier on reward gaps. With enough
> data and a model that can represent the truth, it recovers the true reward up
> to one constant per prompt.

## Combine several reward models

Here's the problem with optimising a learned reward. The fitted reward is the
true reward plus an error, and the error is largest on responses that appeared
in few comparisons. When you pick the response with the highest fitted reward,
you pick, in part, the one with the most positive error. This is the
maximisation bias from *Maximisation bias and double learning*. It's one cause
of what RLHF calls *overoptimisation*: the proxy reward keeps rising while the
true reward stalls or falls. Gao, Schulman, and Hilton (2022) measured it at
scale.

One response is to train $K$ reward models, on resampled data or from
different starting points, and combine them. Where the models disagree, the
error is probably large. Coste and colleagues (2023) compare two conservative
combinations with the plain mean:

$$
r_{\text{WCO}}(y) = \min_k r_k(y),
\qquad
r_{\text{UWO}}(y) = \frac{1}{K} \sum_{k} r_k(y) - \lambda\, \frac{1}{K} \sum_k \big(r_k(y) - \bar r(y)\big)^2
$$

Read it as: *worst-case optimisation* (WCO) trusts the most pessimistic model.
*Uncertainty-weighted optimisation* (UWO) takes the mean $\bar r(y)$ and
subtracts $\lambda$ times the variance across models, taken over $K$, not
$K - 1$. In numpy, with scores of shape `(K, N)`:

```python
wco = scores.min(axis=0)
uwo = scores.mean(axis=0) - lam * scores.var(axis=0)
```

### Worked example

Three models score three responses:

| Model | Response 1 | Response 2 | Response 3 |
|---|---|---|---|
| 1 | 1.0 | 2.0 | 0.5 |
| 2 | 3.0 | 2.0 | 0.5 |
| 3 | 2.0 | 2.0 | 2.0 |

The means are $(2, 2, 1)$, so responses 1 and 2 tie. WCO gives $(1, 2, 0.5)$
and prefers response 2, which every model agrees on. The variances are
$(2/3, 0, 1/2)$, so UWO with $\lambda = 0.5$ gives $(1.667, 2, 0.75)$ and also
prefers response 2.

The exercise runs this on 40 toy prompts. Each has 64 responses, a peaked
reference, 300 comparisons, and 5 models fitted to bootstrap resamples. On the
ensemble's top 3 responses, the models disagree more than twice as much as on a
typical response: a standard deviation of 0.74 against 0.31. At its own
favourite response, a single model overrates the true reward by 1.92 on
average, and WCO overrates it by 1.20.

Ensembles help less than those numbers suggest. Eisenstein and colleagues
(2023) found that models fine-tuned from one pretrained checkpoint make similar
mistakes, so they agree exactly where they're all wrong. Ensembles that differ
in their pretraining did better, but still didn't stop reward hacking.

> [!KEY]
> Picking the best response under a noisy reward picks its errors too. WCO
> and UWO penalise responses the ensemble disagrees on, which is where those
> errors concentrate.

## Score the outcome or the process

An *outcome reward model* (ORM) scores only the final answer. A *process reward
model* (PRM) scores every step of a solution. On maths, Lightman and colleagues
(2023) collected PRM800K, about 800,000 human labels on solution steps. They
found that process supervision significantly beats outcome supervision for
choosing among sampled solutions. Their PRM scores a solution as the product of
its step probabilities: steps at $(0.9, 0.8, 0.95)$ score $0.684$. Uesato and
colleagues (2022) found that outcome supervision reaches similar final-answer
accuracy, but that process supervision gives more correct reasoning.

Human step labels are expensive. Math-Shepherd (Wang and colleagues, 2023)
labels a step automatically. From the solution so far, sample $N$ completions
and check their final answers. The *soft label* is the fraction that succeed,
and the *hard label* is 1 if any succeeds:

$$
y^{\text{SE}} = \frac{1}{N} \sum_{j=1}^{N} \mathbf{1}[a_j = a^\star],
\qquad
y^{\text{HE}} = \max_j \mathbf{1}[a_j = a^\star]
$$

Read it as: $a_j$ is the final answer of completion $j$, and $a^\star$ is the
correct one. The soft label is a Monte Carlo estimate of the prefix's value
$V^\pi(s_t)$, the chance that the policy succeeds from here. Its average is
exactly $V^\pi(s_t)$. The hard label's average is $1 - (1 - V^\pi(s_t))^N$,
which climbs toward 1 as $N$ grows, so it's not a calibrated probability.

For a toy policy you can compute $V^\pi$ exactly, by walking back up the prefix
tree from the last lesson:

```python
values[T] = rewards                                   # shape (V,) * T
for t in reversed(range(T)):
    values[t] = (np.exp(cond_logps[t]) * values[t + 1]).sum(axis=-1)
```

### Process rewards are value changes

Given exact values, a natural step reward is how much a token changed the
chance of success:

$$
r^{\text{proc}}_t = V^\pi(s_{t+1}) - V^\pi(s_t)
$$

Read it as: the value after writing $y_t$, minus the value before. Setlur and
colleagues (2024) argue that process rewards should measure this kind of
progress, a step-level advantage under a chosen policy. Sum it over a response and the middle terms
cancel, leaving $r(y) - V^\pi(s_0)$. That's the outcome reward minus a
constant, so the two rewards agree on which whole responses are better. They
differ in which tokens get the credit.

This connects to the critic from Part 4. Use the process rewards from token $t$
onward as the policy-gradient weight for token $t$. That weight is
$r(y) - V^\pi(s_t)$: the return minus a baseline that doesn't depend on
$y_t$. So the expected gradient is unchanged, and the variance usually falls,
by an amount that depends on how well the baseline predicts the outcome. On the
exercise's digit-sum task, the exact gradient variance drops from 0.407 with the
outcome reward to 0.336 with process rewards.

### Worked example

Use the two-token reference from the last lesson: $y_0$ is 0 or 1 with
probability 0.5 each, then $(0.8, 0.2)$ after a 0 and $(0.4, 0.6)$ after a 1.
The responses $(0, 1)$ and $(1, 0)$ succeed, with reward 1.

- $V(0) = 0.2$, $V(1) = 0.4$, and $V(s_0) = 0.5 \times 0.2 + 0.5 \times 0.4 = 0.3$.
- For $(0, 1)$: $r^{\text{proc}} = (0.2 - 0.3, 1 - 0.2) = (-0.1, 0.8)$.
- For $(1, 0)$: $(0.4 - 0.3, 1 - 0.4) = (0.1, 0.6)$.
- For $(1, 1)$, a failure: $(0.1, 0 - 0.4) = (0.1, -0.4)$.

Each row sums to $r(y) - 0.3$. The first step of $(0, 1)$ gets negative credit
even though the response succeeds, because a 0 made success less likely. From
the prefix $(0)$, Math-Shepherd with $N = 4$ gives a soft label averaging 0.2
and a hard label averaging $1 - 0.8^4 = 0.5904$.

> [!KEY]
> A value-based process reward is $V(s_{t+1}) - V(s_t)$. It telescopes to the
> outcome reward minus $V(s_0)$, so it moves credit between tokens without
> changing which responses are better.

> [!RECAP]
> - The BT loss is `softplus(-(r_chosen - r_rejected)).mean()`, and its
>   gradient for the chosen reward is $-(1 - \sigma(\Delta)) / n$.
> - With enough data, a BT fit equals the true reward plus a constant.
> - WCO is the minimum over models. UWO is the mean minus $\lambda$ times the
>   variance over $K$.
> - Prefix values come from a backward pass. The soft Math-Shepherd label
>   averages to $V$, and the hard one to $1 - (1 - V)^N$.
> - Process rewards are value differences along a response, and they sum to
>   $r(y) - V(s_0)$.

## Build it step by step

The checks test the functions in this order. The starter repeats `ToyLM` and
`all_sequences` from the previous lesson, and adds `position_token_features`.

1. **`bradley_terry_loss(reward_chosen, reward_rejected, weights=None)`.** Takes
   torch tensors of shape `(n,)` and returns a scalar tensor. With `weights`,
   return `sum(w * loss) / sum(w)`. Common mistake: `torch.log(torch.sigmoid(...))`,
   which fails the gap of $-800$. The second check compares finite
   differences of your loss with $-(1 - \sigma(\Delta)) / n$.
2. **`fit_reward_model(x_chosen, x_rejected, steps, lr, weight_decay, weights)`.**
   Takes numpy features of shape `(n, D)` and returns `phi` of shape `(D,)`.
   Work in float64 and start from zeros. Common mistake: forgetting
   `opt.zero_grad()`, so gradients pile up across steps.
3. **`ensemble_reward(rewards, mode, lam)`.** Takes `(K, N)` and returns `(N,)`.
   Common mistake: using the variance over $K - 1$.
4. **`prefix_values(cond_logps, rewards)`.** Returns a list of $T + 1$ arrays,
   entry $t$ of shape `(V,) * t`. Common mistake: summing log-probabilities
   instead of probabilities.
5. **`math_shepherd_labels(rng, cond_logps, rewards, prefix, n_rollouts)`.**
   Returns `(soft, hard)` floats. Grow an `(n_rollouts, t)` array of prefixes
   one token at a time, and index `cond_logps[t]` with `tuple(cur.T)`. Common
   mistake: sampling every rollout from the same next-token distribution after
   the first sampled token.
6. **`process_rewards(tokens, values)`.** Takes `(B, T)` tokens and returns
   `(B, T)`. Gather the value of every prefix along each row, including the
   empty one and the full response, then `np.diff`. The last check computes the
   exact policy gradient from your process rewards and from the outcome reward,
   and compares their means and variances.

## Cost and scaling

A reward model costs one forward pass per response. The following table compares
the pieces for a batch of $B$ responses of length $T$:

| Piece | Cost | What grows |
|---|---|---|
| BT reward model | One forward pass per response | Linear in $B$; pairs reuse scores |
| Ensemble of $K$ | $K$ forward passes, $K$ copies in memory | Linear in $K$ |
| PRM scoring | One forward pass, a score at each step | Same as an ORM |
| Math-Shepherd labels | $N$ completions per step | $N$ times the steps times the completion cost |

Math-Shepherd labels are the expensive part: every labelled step costs $N$
generations. That cost is paid once, to build training data for a PRM, which
then scores new solutions in one pass. For ensembles, some implementations share
one trunk and train several small heads, which cuts memory but also cuts how
different the members are.

## Where this breaks

**Reading scores across prompts.** Rewards carry an unknown constant per
prompt. Normalise them, for example to mean 0 over the reference's responses,
before you feed them to RL.

**Deterministic labels.** If every comparison goes to the truly better
response, the maximum-likelihood gaps grow without bound. Real preference
data is noisy, which keeps the fit finite. Weight decay also helps.

**Errors the ensemble shares.** Models trained on the same data from the same
checkpoint make the same mistakes, so their variance is small exactly where it
matters. Vary the pretraining, and keep a held-out gold evaluation.

**Hard labels read as probabilities.** A hard Math-Shepherd label estimates
$1 - (1 - V)^N$, not $V$. Train on soft labels when you want a value.

**A policy that games a PRM.** A PRM trained on one policy's rollouts values
steps under that policy. A stronger policy can complete steps it marks as dead
ends, and a policy can pad a solution with many short, safe-looking steps. The
DeepSeek-R1 report lists PRMs among its unsuccessful attempts, citing reward
hacking and the difficulty of defining and judging a step.

## Further reading

- [Christiano et al. (2017), Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741)
  trains a reward model from pairwise comparisons and a policy against it.
- [Ouyang et al. (2022), Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
  is the InstructGPT recipe, with a reward model trained on ranked responses.
- [Coste, Anwar, Kirk, and Krueger (2023), Reward model ensembles help mitigate overoptimization](https://arxiv.org/abs/2310.02743)
  defines WCO and UWO and tests them with best-of-n and PPO.
- [Eisenstein et al. (2023), Helping or herding? Reward model ensembles mitigate but do not eliminate reward hacking](https://arxiv.org/abs/2312.09244)
  shows that members sharing a pretraining seed share their errors.
- [Lightman et al. (2023), Let's verify step by step](https://arxiv.org/abs/2305.20050)
  releases PRM800K and compares process with outcome supervision.
- [Wang et al. (2023), Math-Shepherd](https://arxiv.org/abs/2312.08935)
  labels steps with Monte Carlo completions instead of people.
- [Setlur et al. (2024), Rewarding progress: scaling automated process verifiers for LLM reasoning](https://arxiv.org/abs/2410.08146)
  argues that a step's reward should be its change in the chance of success.

Open questions researchers are working on:

- How much preference data does a reward model need before a policy stops
  finding its errors?
- Can a process reward model be trained so that a stronger policy can't game it?
- When does a verifier that writes a critique before it scores beat a scalar
  reward head?

## The exercise

You build a reward-model toolkit. Write `bradley_terry_loss` and
`fit_reward_model` to learn a linear reward from comparisons, `ensemble_reward`
to combine several, and `prefix_values`, `math_shepherd_labels`, and
`process_rewards` for outcome and process rewards on a toy digit-sum task.

The checks test the loss at ties, at a gap of 1, at gaps of $\pm 800$, and under
a shift of every reward, and check its gradient with finite differences. They
check that a fit on every weighted pair recovers the true reward plus a
constant, and that a fit on 4,000 sampled pairs comes within 0.01 of the true
reward's held-out loss. They check the ensemble formulas by hand, then fit 240
tabular models to show disagreement peaking at the top and WCO overrating less.
Finally, they check exact prefix values, the averages of both Math-Shepherd
labels over 600 repeats, the telescoping sum, and that process rewards give the
exact policy gradient with at least 10 percent less variance.
