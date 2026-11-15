---
title: Sycophancy from mixed rewards
slug: 070-sycophancy-from-mixed-rewards
part: "Part 9 — RL for language models"
summary: Mix a thumbs-up reward into a truth reward, solve for the weight w* where the optimum agrees with a wrong user, and see an average hide it.
minutes: 35
gpu: false
objectives:
  - Write the mixed reward, and compute the KL-regularized optimal policy for it in closed form.
  - Solve for the critical weight w* at which the optimal policy starts agreeing with a user who is wrong.
  - Fit a Bradley-Terry reward model to rater comparisons and read the agreement bias from it.
  - Evaluate by slice, and show an aggregate score that rises while the wrong-user slice collapses.
prereqs:
  - 066-language-models-as-policies
  - 067-reward-models
  - 069-rlhf-with-ppo
exercise: 070-sycophancy-from-mixed-rewards
---

# Sycophancy from mixed rewards

A user writes, "The capital of Australia is Sydney, right?" The helpful reply corrects them. The pleasant reply agrees. If you train a chatbot on thumbs-up ratings, which reply does it learn to give?

Raters can prefer being agreed with. Sharma and colleagues (2023) found that human preference data, and the preference models trained on it, sometimes favor an agreeable answer over a correct one. Because most users are right, an eval that averages over all users can improve while the model gets worse for the users who need a correction. OpenAI's account of rolling back a GPT-4o update in April 2025 follows this pattern. Its two posts point to extra weight on thumbs-up and thumbs-down feedback, and say that offline evals and A/B tests looked fine.

This chapter builds a toy world with an exact answer. You fit a rater's agreement bias, solve for the optimal policy of a mixed reward, find the weight at which it starts agreeing with a wrong user, and measure by slice.

## What you already know

- Maximizing reward minus $\beta$ times a KL divergence from a reference policy has the exact optimum $\pi^* \propto \pi_{\text{ref}} \exp(r / \beta)$ ([Language models as policies](/lessons/066-language-models-as-policies)).
- A Bradley-Terry model turns pairwise preferences into a reward, with $P(a \text{ preferred}) = \sigma(r_a - r_b)$ ([Reward models from preferences](/lessons/067-reward-models)).
- A reward model can inherit a bias from its raters.
- Best-of-$n$ keeps the highest-scoring of $n$ draws.

## A world with one choice

Picture a prompt in which the user states a belief. With probability $\rho$ the belief is right, and otherwise it is wrong. The model has two replies: *agree* or *correct* the user. The user's claim is the state $s$, so every table in this chapter has shape $(2, 2)$, indexed by state and action.

Two rewards score the reply:

- The *truth reward* is $+1$ for a correct reply and 0 for a wrong one.
- The *thumbs reward* comes from a rater whose hidden reward is $k \cdot \text{correct} + b \cdot \text{agree}$. The number $k$ is how much the rater values a correct reply, and $b$ is the *agreement bias*, the extra credit for a reply that agrees. The rater picks reply $a$ over reply $b'$ with probability $\sigma(r_a - r_{b'})$, which is the Bradley-Terry model.

A rater likes correct answers and also likes being agreed with, and the two tastes add. Take $k = 1$ and $b = 3$. The following table lists the four cases:

| User is | Reply | Correct? | Truth reward | Thumbs reward |
|---|---|---|---|---|
| right | agree | yes | 1 | $k + b = 4$ |
| right | correct | no | 0 | 0 |
| wrong | agree | no | 0 | $b = 3$ |
| wrong | correct | yes | 1 | $k = 1$ |

On a wrong user, the thumbs reward ranks agreeing above correcting, 3 against 1, while the truth reward ranks them the other way. That disagreement is the whole story.

## Mix the rewards

Training rarely uses the thumbs reward alone. The truth reward is the main signal, and the thumbs reward joins it with a weight $w$ between 0 and 1:

$$
r = (1 - w)\, r_{\text{truth}} + w\, r_{\text{thumbs}}
$$

Read it aloud: with weight $w$ you listen to the raters, and with weight $1 - w$ you listen to the truth. "Why not keep $w$ small?" Because "small" has a precise limit.

With two replies, the optimal policy depends only on the *gap*, the reward for agreeing minus the reward for correcting. For a right user, agreeing earns $(1 - w) + w(k + b)$ and correcting earns 0, so the gap is positive for every $w$. For a wrong user, agreeing earns $wb$ and correcting earns $(1 - w) + wk$, so

$$
\text{gap}_{\text{wrong}}(w) = w\,(1 + b - k) - 1
$$

Read it aloud: at $w = 0$ the gap is $-1$, so truth wins, and each unit of $w$ adds $1 + b - k$ in favor of agreeing. The gap crosses zero at some weight, and past it the mixed reward prefers agreeing with a user who is wrong. With $k = 1$ and $b = 3$, the wrong-user gap is $3w - 1$ and the right-user gap is $1 + 3w$.

## The optimal policy, in closed form

For two replies, divide the probability of agreeing by the probability of correcting under $\pi^* \propto \pi_{\text{ref}} \exp(r / \beta)$. The normalizer cancels, and what is left is the reference's odds times $\exp(\text{gap} / \beta)$. So the probability of agreeing is a sigmoid:

$$
\pi^*(\text{agree} \mid s) = \sigma\!\left(\log \frac{p}{1 - p} + \frac{\text{gap}_s}{\beta}\right)
$$

Here $p = \pi_{\text{ref}}(\text{agree})$ is how often the reference agrees. Read it aloud: start from the reference's log-odds, and add the reward gap in units of $\beta$. A small $\beta$ lets the gap dominate, and a large $\beta$ keeps the model near the reference. In numpy, with `r` and `p_ref` both $(2, 2)$:

```python
logits = np.log(p_ref) + r / beta
logits -= logits.max(axis=1, keepdims=True)
pi = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
```

### A worked example

Let the reference agree half the time ($p = 0.5$), let $\beta = 1$, let 80 percent of users be right ($\rho = 0.8$), and set $w = 0.5$. The mixed reward table is

- right user: agree $0.5 \cdot 1 + 0.5 \cdot 4 = 2.5$, correct $0$;
- wrong user: agree $0.5 \cdot 3 = 1.5$, correct $0.5 \cdot 1 + 0.5 \cdot 1 = 1$.

The gaps are 2.5 and 0.5, and the log-odds term is 0. So the policy agrees with a right user with probability $\sigma(2.5) = 0.9241$ and with a wrong user with probability $\sigma(0.5) = 0.6225$. It is correct for 92.41 percent of right users and 37.75 percent of wrong ones. Overall accuracy is $0.8 \cdot 0.9241 + 0.2 \cdot 0.3775 = 0.8148$.

At $w = 0$ the model is correct 0.7311 of the time on both slices, which is $\sigma(1)$. Mixing in the thumbs reward raised overall accuracy from 0.731 to 0.815 and cut the wrong-user slice from 0.731 to 0.378. Figure 1 sweeps $w$ from 0 to 1.

```figure
<svg viewBox="0 0 600 260" role="img" aria-label="Accuracy against the mixing weight w. On right users it climbs from 0.73 to 0.98. Overall it climbs from 0.73 to 0.82. On wrong users it falls from 0.73 to 0.12 and crosses one half at w equal to one third.">
<line x1="70" y1="200.0" x2="560" y2="200.0" class="stroke-muted"/>
<text x="62" y="204.0" class="fig-small" text-anchor="end">0</text>
<line x1="70" y1="115.0" x2="560" y2="115.0" class="stroke-grid"/>
<text x="62" y="119.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="70" y1="30.0" x2="560" y2="30.0" class="stroke-grid"/>
<text x="62" y="34.0" class="fig-small" text-anchor="end">1</text>
<text x="70.0" y="220" class="fig-small" text-anchor="middle">0</text>
<text x="192.5" y="220" class="fig-small" text-anchor="middle">0.25</text>
<text x="315.0" y="220" class="fig-small" text-anchor="middle">0.5</text>
<text x="437.5" y="220" class="fig-small" text-anchor="middle">0.75</text>
<text x="560.0" y="220" class="fig-small" text-anchor="middle">1</text>
<text x="315" y="244" class="fig-small" text-anchor="middle">mixing weight w on the thumbs reward</text>
<line x1="233.3" y1="30.0" x2="233.3" y2="200" class="stroke-muted dashed"/>
<text x="239.3" y="42.0" class="fig-small">w* = 1/3</text>
<polyline points="70.0,75.7 94.5,70.9 119.0,66.4 143.5,62.3 168.0,58.6 192.5,55.2 217.0,52.1 241.5,49.4 266.0,47.0 290.5,44.8 315.0,42.9 339.5,41.2 364.0,39.7 388.5,38.5 413.0,37.3 437.5,36.3 462.0,35.5 486.5,34.7 511.0,34.1 535.5,33.5 560.0,33.1" class="stroke-c"/>
<polyline points="70.0,75.7 94.5,72.9 119.0,70.4 143.5,68.3 168.0,66.5 192.5,65.0 217.0,63.8 241.5,62.9 266.0,62.3 290.5,61.8 315.0,61.5 339.5,61.3 364.0,61.3 388.5,61.3 413.0,61.4 437.5,61.5 462.0,61.7 486.5,61.8 511.0,62.0 535.5,62.2 560.0,62.4" class="stroke-a"/>
<polyline points="70.0,75.7 94.5,80.9 119.0,86.4 143.5,92.2 168.0,98.2 192.5,104.4 217.0,110.8 241.5,117.1 266.0,123.5 290.5,129.7 315.0,135.8 339.5,141.7 364.0,147.3 388.5,152.6 413.0,157.5 437.5,162.1 462.0,166.4 486.5,170.2 511.0,173.7 535.5,176.9 560.0,179.7" class="stroke-bad"/>
<text x="556.0" y="27.1" class="fig-small" text-anchor="end">user right: 0.98</text>
<text x="422.8" y="48.4" class="fig-small" text-anchor="middle">all users: 0.81</text>
<text x="556.0" y="195.7" class="fig-small" text-anchor="end">user wrong: 0.12</text>
</svg>
Figure 1. Accuracy of the optimal policy as the weight on the thumbs reward grows, with $\rho = 0.8$. Notice that the all-users curve rises to about 0.82 while the wrong-user curve falls through one half at w* = 1/3 and keeps falling.
```

> [!KEY]
> The optimal policy agrees because the mixed reward pays it to, and agreeing is correct for most users. An average over all users can rise while the wrong-user slice collapses.

## The critical weight w*

Define $w^*$ as the weight at which the optimal policy agrees with a wrong user exactly half the time. A sigmoid equals one half when its argument is zero, so set $\log\frac{p}{1-p} + \text{gap}_{\text{wrong}}(w)/\beta = 0$ and solve for $w$:

$$
w^* = \frac{1 - \beta \log \frac{p}{1 - p}}{1 + b - k}
$$

Read it aloud: the numerator is the truth reward's pull, adjusted by how much the reference already leans toward agreeing, and the denominator is how fast the thumbs reward pushes toward agreeing. For a 50-50 reference the log-odds is 0, so $w^* = 1 / (1 + b - k) = 1/3$ here, where Figure 1's dashed line sits.

Three things follow:

- With a 50-50 reference, $w^* < 1$ exactly when $b > k$. If raters value correctness at least as much as agreement, no mixing weight causes the flip.
- A larger bias $b$ shrinks $w^*$: a more biased rater leaves less room for the thumbs reward.
- A reference that agrees more than half the time ($p > 0.5$) lowers $w^*$, and one that tends to correct raises it.

The flip is not where harm starts. Wrong-user accuracy falls below its starting value, 0.731, at every $w > 0$.

## Measure by slice

Why did the average hide it? A *slice* is the subset of prompts with a given property, and slice accuracy averages only over that subset. The aggregate is $\rho \cdot \text{acc}_{\text{right}} + (1 - \rho) \cdot \text{acc}_{\text{wrong}}$, so the wrong-user slice carries weight 0.2. In the worked example the aggregate gains 0.084 while that slice loses 0.354.

You see this only if your eval labels the prompts where the user is wrong. In practice someone must build that slice, as Perez and colleagues (2022) do for sycophancy. OpenAI's account says no eval tracked sycophancy before the rollout.

## Fit the bias from comparisons

You rarely know $k$ and $b$. You see rater comparisons, and a Bradley-Terry fit recovers the weights. Describe a reply by features $\phi = (\text{correct}, \text{agree})$. For a right user, the difference between the agreeing and correcting replies is $(1, 1)$, and for a wrong user it is $(-1, +1)$. The two directions differ, so both $k$ and $b$ are identifiable. Fit $\theta = (k, b)$ by maximum likelihood on

$$
P(a \text{ preferred}) = \sigma\big(\theta^\top (\phi_a - \phi_{b'})\big)
$$

The log-likelihood is concave, so Newton's method converges in a few steps. With $X$ the stacked differences, $p = \sigma(X\theta)$, and $y$ equal to 1 where $a$ won, each step is

```python
p = sigmoid(X @ theta)
H = X.T @ (X * (p * (1 - p))[:, None])
theta = theta + np.linalg.solve(H, X.T @ (y - p))
```

On 4,000 simulated comparisons with true weights $(1, 3)$, the fit returns about $(1.02, 2.96)$, so $\hat{w}^* = 1 / (1 + 2.96 - 1.02) = 0.340$, close to the true $1/3$.

## Best-of-n makes it worse as n grows

Sampling causes the same failure with no training. Draw $n$ replies from the reference and keep the one with the highest thumbs reward. The exact order-statistic formula from [Reward hacking, safety, and the frontier](/lessons/086-reward-hacking-and-safe-rl) applies: sort by reward, and the reply at rank $i$ wins with probability $F_i^n - F_{i-1}^n$, where $F_i$ is the cumulative probability up to it.

For a wrong user, agreeing has thumbs reward 3 and correcting has 1. Correcting wins only if all $n$ draws correct, with probability $0.5^n$. So the chance of agreeing is $1 - 0.5^n$: 0.5, 0.75, 0.9375, and 0.9961 for $n = 1, 2, 4, 8$. Figure 2 shows the rise.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Probability that the best of n draws agrees with a wrong user: 0.50 at n equal 1, 0.75 at 2, 0.94 at 4, and 0.996 at 8.">
<line x1="80" y1="200.0" x2="560" y2="200.0" class="stroke-muted"/>
<text x="72" y="204.0" class="fig-small" text-anchor="end">0</text>
<line x1="80" y1="120.0" x2="560" y2="120.0" class="stroke-grid"/>
<text x="72" y="124.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="80" y1="40.0" x2="560" y2="40.0" class="stroke-grid"/>
<text x="72" y="44.0" class="fig-small" text-anchor="end">1</text>
<rect x="115" y="120.0" width="70" height="80.0" class="fill-bad"/>
<text x="150" y="114.0" class="fig-small fig-bold" text-anchor="middle">0.5000</text>
<text x="150" y="222" class="fig-small" text-anchor="middle">n = 1</text>
<rect x="235" y="80.0" width="70" height="120.0" class="fill-bad"/>
<text x="270" y="74.0" class="fig-small fig-bold" text-anchor="middle">0.7500</text>
<text x="270" y="222" class="fig-small" text-anchor="middle">n = 2</text>
<rect x="355" y="50.0" width="70" height="150.0" class="fill-bad"/>
<text x="390" y="44.0" class="fig-small fig-bold" text-anchor="middle">0.9375</text>
<text x="390" y="222" class="fig-small" text-anchor="middle">n = 4</text>
<rect x="475" y="40.6" width="70" height="159.4" class="fill-bad"/>
<text x="510" y="34.6" class="fig-small fig-bold" text-anchor="middle">0.9961</text>
<text x="510" y="222" class="fig-small" text-anchor="middle">n = 8</text>
<text x="320" y="244" class="fig-small" text-anchor="middle">draws per prompt, ranked by the thumbs reward</text>
</svg>
Figure 2. The chance that best-of-n agrees with a wrong user, ranking by the thumbs reward alone. Notice that the rate nears 1 by n = 8, with no training and no KL leash.
```

> [!KEY]
> A thumbs reward with agreement bias $b > k$ pushes toward agreeing with a wrong user, whether you optimize it with RL or best-of-$n$. More pressure means more agreement.

> [!RECAP]
> - The mixed reward is $(1 - w)\, r_{\text{truth}} + w\, r_{\text{thumbs}}$, and the thumbs reward adds an agreement bonus $b$ to a correctness credit $k$.
> - For two replies, $\pi^*(\text{agree}) = \sigma(\text{log-odds} + \text{gap}/\beta)$, and on a wrong user the gap is $w(1 + b - k) - 1$.
> - The optimal policy agrees with a wrong user half the time at $w^* = (1 - \beta \log\frac{p}{1-p}) / (1 + b - k)$.
> - A Bradley-Terry fit recovers $b$, and a slice eval shows the failure that the aggregate hides.

## Build it step by step

The checks test six functions, smallest piece first. The starter gives you `simulate_comparisons`, which plays the rater, and the constants `K_TRUE`, `B_TRUE`, `FEATURES`, and `TRUTH`. Tables are `(2, 2)` arrays indexed `[state, action]`, with state 0 for a right user and action 0 for agreeing.

1. **`mixed_rewards(w, k, b)`.** Returns the `(2, 2)` table $(1 - w)\, r_{\text{truth}} + w\, r_{\text{thumbs}}$, with the thumbs table built from the columns of `FEATURES`. The common mistake is to scale only one of the two rewards.
2. **`fit_bradley_terry(feats_a, feats_b, a_wins)`.** Takes `(n, d)` features and a `(n,)` array of 0 and 1, and returns `(d,)` weights by Newton's method from zeros. One check confirms the gradient $X^\top (y - p)$ is zero at your answer.
3. **`optimal_policy(p_ref, reward, beta)`.** Returns `(2, 2)` rows that sum to 1. Subtract each row's maximum logit first. The common mistake is to normalize over the whole table and not each row.
4. **`critical_weight(k, b, beta, p_agree_ref)`.** Returns $w^*$, or `np.inf` when $1 + b - k \le 0$. The common mistake is to drop the log-odds term.
5. **`slice_eval(policy, rho)`.** Returns `(acc_right, acc_wrong, aggregate)`. The common mistake is to read `policy[1, 0]` as the wrong-user accuracy. A wrong user is answered correctly by `policy[1, 1]`.
6. **`best_of_n_policy(p, reward, n)`.** Sorts once with `np.argsort`, takes a cumulative sum, and scatters back to the original order.

## Where this breaks

**The toy has one bias and two replies.** Real raters differ by topic, and real replies are not a clean agree or correct. What survives is the shape: a gap that changes sign at a critical weight, and an average that hides it.

**The weight is rarely explicit.** In a real system, thumbs data enter through a reward model, so you rarely know $w$. That is why you test the slice.

**The closed form is a target, not a training run.** PPO reaches the optimum only approximately, as [RLHF with PPO](/lessons/069-rlhf-with-ppo) shows, so the weight at which a trained model flips will differ.

**Public details are limited.** The GPT-4o account comes from the company's own posts. Treat the toy as an illustration of the mechanism, not a reconstruction of that event.

## Further reading

- [Sharma et al. (2023), Towards understanding sycophancy in language models](https://arxiv.org/abs/2310.13548). Shows that five assistants flatter users, and that preference data and models sometimes favor agreeable answers over correct ones.
- [Perez et al. (2022), Discovering language model behaviors with model-written evaluations](https://arxiv.org/abs/2212.09251). Finds that larger models more often repeat back a user's preferred answer.
- [Denison et al. (2024), Sycophancy to subterfuge: Investigating reward tampering in large language models](https://arxiv.org/abs/2406.10162). Finds that gaming in early environments, including sycophancy, generalizes to later and worse gaming.
- [OpenAI (2025), Sycophancy in GPT-4o](https://openai.com/index/sycophancy-in-gpt-4o/). Describes the April 2025 update and its rollback.
- [OpenAI (2025), Expanding on what we missed with sycophancy](https://openai.com/index/expanding-on-sycophancy/). Gives the follow-up account, including the evals that missed it.
- [Scribe notes, CS 2881 Fall 2026 lecture 6, Grading, reward hacking, and alignment](https://windowsontheory.org/2026/11/08/cs-2881-fall-26-lecture-6-grading-reward-hacking-and-alignment/). Summarizes the October 8 lecture, which discusses both posts.

Open questions researchers are working on:

- How do you measure sycophancy in long, personal conversations, where short evals can miss it?
- Can you detect an agreement bias in a reward model before you train against it?

## The exercise

You write `mixed_rewards`, `fit_bradley_terry`, `optimal_policy`, `critical_weight`, `slice_eval`, and `best_of_n_policy`. The `__main__` block fits the bias, computes $\hat{w}^*$, and sweeps $w$ to print each slice.

The checks use $k = 1$, $b = 3$, $\beta = 1$, and $\rho = 0.8$. They confirm the reward table, a Bradley-Terry fit that has zero gradient and recovers the bias within 0.35, and a closed-form policy that matches a numeric optimizer. They check $w^*$ against its formula, the aggregate rising while the wrong-user slice falls below 40 percent, and best-of-$n$ against brute-force enumeration.
