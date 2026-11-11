---
title: Credit assignment for language models
slug: 074-credit-assignment-for-llms
part: "Part 9 — RL for language models"
summary: Give each reasoning step its own advantage with Monte Carlo values, automatic step labels, and a log-ratio reward from an outcome-trained model.
minutes: 40
gpu: false
objectives:
  - Explain why one sequence-level advantage gives every token in a response the same credit.
  - Estimate the value of a prefix from K rollouts, and turn the estimates into per-step advantages that sum telescopically.
  - Compute Math-Shepherd hard and soft step labels, and say which one is biased.
  - Read a process reward off an outcome-trained model as beta times a log ratio, summed over a step.
  - Count the extra rollouts each method needs per sampled response.
prereqs:
  - 034-generalized-advantage-estimation
  - 066-reward-models
  - 070-group-relative-policy-optimization
  - 073-process-rewards-and-search
exercise: 074-credit-assignment-for-llms
---

# Credit assignment for language models

A model writes a four-step solution to a math problem. The first two steps are fine, the third step makes an arithmetic slip, and the fourth step carries the slip to a wrong answer. The reward is 0. Which step should the training signal blame?

Only the third step deserves the blame. Yet the methods in the earlier chapters hand all four steps, and every token inside them, the same number. The learner then has to work out, across many responses, that the third step is the culprit. That takes far more samples than reading the answer off one response would. This chapter is about giving each step its own credit, which RL calls *credit assignment*, and it covers three ways to get it: roll the model out many times from each step, label steps from those rollouts ahead of time, or read the credit off a model you have already trained.

## What you already know

- A language model is a policy $\pi(y \mid x)$, and a response is a sequence of tokens, from [Language models as policies](/lessons/065-language-models-as-policies).
- The advantage is a return minus a baseline, and a critic supplies the baseline, from [Generalized advantage estimation](/lessons/034-generalized-advantage-estimation).
- GRPO scores a group of responses and gives every token of a response one shared advantage, from [Group-relative policy optimization](/lessons/070-group-relative-policy-optimization).
- The value of a prefix is the probability that the policy finishes with a correct answer, from [Process rewards and test-time search](/lessons/073-process-rewards-and-search).
- A reward model trained on outcomes scores whole responses, from [Reward models from preferences](/lessons/066-reward-models).

## Why one advantage is too coarse

GRPO and RLOO compute one number per response, $A = R - b$, where $R$ is the final reward and $b$ is the average reward of the group. The loss multiplies that number into the log-probability of every token. A response with $A < 0$ pushes down every token it contains, including the correct first two steps.

This is not wrong on average. Averaged over many samples, the correct steps appear in good and bad responses alike, and their pushes cancel, so the gradient points the right way. It is slow, because the cancellation has to happen through samples. The *per-step advantage* removes the wait. Write $s_k$ for the prefix after $k$ steps, so $s_0$ is the prompt. The advantage of step $k$ is how much that step changed the chance of success:

$$
A_k = V(s_k) - V(s_{k-1})
$$

Read it aloud: a step is good when the prefix after it is worth more than the prefix before it. Add the steps up and the middle values cancel, which leaves a telescoping sum:

$$
\sum_{k=1}^{T} A_k = V(s_T) - V(s_0) = R - V(s_0)
$$

The last prefix is a finished response, so $V(s_T) = R$. The per-step advantages split the sequence-level advantage $R - V(s_0)$ into pieces, and blame lands where the value dropped. Everything now depends on getting $V$ for prefixes, and there are two ways to get it.

## A toy with exact values

The toy has $T = 4$ steps. Step $j$ is correct with probability $0.9, 0.8, 0.7, 0.6$, independently, and the answer is correct only if all four steps are. A prefix is an array of step outcomes. If it holds a wrong step, its value is 0. Otherwise its value is the product of the probabilities of the steps not yet written. So $V(s_0) = 0.9 \times 0.8 \times 0.7 \times 0.6 = 0.3024$.

Take the response from the opening, with outcomes $[1, 1, 0, 1]$. The prefix values are $[0.3024, 0.336, 0.42, 0, 0]$. The exact advantages are the differences:

- Step 1: $0.336 - 0.3024 = +0.0336$. Writing a correct step 1 raised the chance of finishing.
- Step 2: $0.42 - 0.336 = +0.084$.
- Step 3: $0 - 0.42 = -0.42$. The slip destroyed the response.
- Step 4: $0 - 0 = 0$. The response was already lost.

They sum to $-0.3024 = R - V(s_0)$. GRPO, even with a perfect baseline, gives each step $-0.3024$ instead. It blames steps 1 and 2, which did nothing wrong, and it underblames step 3.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Per-step advantages for a four-step response whose third step is wrong: exact, uniform, and Monte Carlo estimates">
  <line x1="50" y1="80" x2="580" y2="80" class="stroke-muted"/>
  <text x="8" y="84" class="fig-small">0</text>
  <rect x="70" y="72.6" width="26" height="7.4" class="fill-c"/>
  <rect x="100" y="80.0" width="26" height="66.5" class="fill-muted"/>
  <rect x="130" y="69.7" width="26" height="10.3" class="fill-b"/>
  <text x="78" y="226" class="fig-small">step 1</text>
  <rect x="200" y="61.5" width="26" height="18.5" class="fill-c"/>
  <rect x="230" y="80.0" width="26" height="66.5" class="fill-muted"/>
  <rect x="260" y="38.8" width="26" height="41.2" class="fill-b"/>
  <text x="208" y="226" class="fig-small">step 2</text>
  <rect x="330" y="80.0" width="26" height="92.4" class="fill-c"/>
  <rect x="360" y="80.0" width="26" height="66.5" class="fill-muted"/>
  <rect x="390" y="80.0" width="26" height="110.0" class="fill-b"/>
  <text x="338" y="226" class="fig-small">step 3 (wrong)</text>
  <rect x="460" y="80.0" width="26" height="0.8" class="fill-c"/>
  <rect x="490" y="80.0" width="26" height="66.5" class="fill-muted"/>
  <rect x="520" y="80.0" width="26" height="0.8" class="fill-b"/>
  <text x="468" y="226" class="fig-small">step 4</text>
  <rect x="330" y="14" width="12" height="12" class="fill-c"/>
  <text x="348" y="25" class="fig-small">exact</text>
  <rect x="330" y="32" width="12" height="12" class="fill-muted"/>
  <text x="348" y="43" class="fig-small">uniform R - V(root)</text>
  <rect x="330" y="50" width="12" height="12" class="fill-b"/>
  <text x="348" y="61" class="fig-small">vine, K = 64</text>
</svg>
Figure 1. One response whose third step is wrong. The uniform advantage (gray) is the same on all four steps. The exact advantage (green) and the Monte Carlo estimate with 64 rollouts per boundary (orange) put the blame on step 3.
```

## Estimate the values by rollout

You can get $V(s_k)$ from a learned critic $V_\phi$, as in [RLHF with PPO](/lessons/068-rlhf-with-ppo). The critic is a network, so it is biased, and it has to read the meaning of a half-written solution. The alternative is to measure the value. A prefix is just text, and a language model can continue any text. So from the prefix $s_k$, sample $K$ completions from the current policy, score each with the reward, and average:

$$
\hat V(s_k) = \frac{1}{K} \sum_{i=1}^{K} R\big(s_k, \text{completion}_i\big)
$$

This is the Monte Carlo value estimate that VinePPO (Kazemnejad and colleagues, 2024) uses in place of the critic. It is unbiased, and its *standard error* is $\sqrt{V(1 - V)/K}$ for a 0/1 reward. At $V = 0.42$ and $K = 100$, that is $0.049$. The advantage of step $k$ is then $\hat V(s_k) - \hat V(s_{k-1})$, the same difference as before, with estimates in place of exact values.

Two details matter. First, estimate each boundary once and reuse it. Boundary $k$ serves as the "next" value of step $k$ and the "previous" value of step $k + 1$. Reusing it keeps the sum telescoping exactly: the advantages add up to $R - \hat V(s_0)$. Second, the last prefix is a finished response, so $\hat V(s_T)$ is the real reward and needs no rollouts. The sign of the difference is next minus previous. Reverse it and a correct step is blamed.

```python
v_hat = np.empty(T + 1)
for k in range(T):
    v_hat[k] = mc_value(outcomes[:k], K, rng)[0]   # K rollouts from boundary k
v_hat[T] = float(outcomes.all())                   # the real reward
advantages = np.diff(v_hat)                        # next minus previous
```

On the response $[1, 1, 0, 1]$ with $K = 64$, the estimated advantages are $[0.047, 0.188, -0.5, 0]$, against the exact $[0.034, 0.084, -0.42, 0]$. They sum to $-0.266 = 0 - \hat V(s_0)$ with $\hat V(s_0) = 0.266$. The estimates are noisy, but they have the right signs and blame step 3. More rollouts help at the rate the standard error predicts. On 1,000 sampled responses, the mean squared error of the per-step advantage is $0.070$ at $K = 4$, $0.020$ at $K = 16$, $0.0047$ at $K = 64$, and $0.0012$ at $K = 256$. The uniform advantage scores about $0.16$, so even $K = 4$ beats it by more than a factor of two.

The price is generation. Estimating $T$ boundaries with $K$ rollouts each costs $T \times K$ extra completions per response, where GRPO spends none. At $T = 4$ and $K = 16$, that is 64 extra completions for each response.

> [!KEY]
> A per-step advantage is the change in the value of the prefix. Estimating that value by rollouts gives an unbiased advantage whose noise shrinks as $1/\sqrt{K}$, at a cost of $T \times K$ completions per response.

## Label steps once, then train a scorer

The same rollouts can be spent ahead of time. Math-Shepherd (Wang and colleagues, 2023) rolls out $K$ completions from each step of a training solution, and uses the results as labels for a process reward model. It defines two rules:

- *Hard estimation* labels a step 1 if at least one completion reaches the correct answer, and 0 otherwise.
- *Soft estimation* labels a step with the fraction of completions that reach the correct answer.

Soft is the Monte Carlo value $\hat V$ again, so it is unbiased. Hard is not. A step with true value $v$ gets a hard label of 1 with probability $1 - (1 - v)^K$. At $K = 4$ and $v = 0.3024$, that is $0.763$, so the average hard label overstates the value by more than a factor of two. The label is optimistic because one lucky completion is enough. The mistake to avoid is turning a soft label into a hard one by rounding. A step with one correct completion out of four has soft label $0.25$, and rounding turns it into 0, which throws away the information the rule exists to keep.

The cost is $T \times K$ completions per labeled solution, spent once and offline. The trained PRM then scores steps in one forward pass, as in [Process rewards and test-time search](/lessons/073-process-rewards-and-search), but it must keep up as the policy changes.

## Read the credit off a trained model

A third route avoids rollouts entirely. Suppose you train an outcome reward model whose reward for a whole response is $r(y) = \beta \log \frac{\pi_\phi(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$, using only outcome labels. This is the parameterization that [Direct preference optimization](/lessons/069-direct-preference-optimization) uses. Yuan and colleagues (2024), in "Free Process Rewards without Process Labels", point out that a log ratio over a whole response is a sum of per-token log ratios. So the model also gives a reward for each token, with no step labels:

$$
r_t = \beta \log \frac{\pi_\phi(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})}
$$

Sum the token rewards inside each step to get a step reward, and add the steps up to get a score for a prefix. The step rewards add to the sequence-level $\beta$ log ratio by construction, the same telescoping as before. Take a response of five tokens, split into steps of 2, 1, and 2 tokens, with $\beta = 0.5$. The token log ratios are $0.5, 0, -1, 0.3, 0.5$, so the step sums are $0.5, -1, 0.8$, and the step rewards are $0.25, -0.5, 0.4$. They add to $0.15$, which is $0.5$ times the sequence log ratio of $0.3$.

```python
per_token = beta * (logp - logp_ref)
step_rewards = np.bincount(step_ids, weights=per_token, minlength=n_steps)
```

This costs one forward pass through the reference model and no rollouts, against $T \times K$ completions for the other two methods. The authors report that it matches or beats rollout-labeled PRMs at a fraction of the cost. Whether a per-token ratio is a trustworthy credit signal is less settled, because the model saw only whole-response outcomes.

> [!RECAP]
> - A per-step advantage is $V(s_k) - V(s_{k-1})$, and the steps sum to $R - V(s_0)$.
> - VinePPO estimates each value from $K$ rollouts, at $T \times K$ completions per response.
> - Math-Shepherd soft labels are unbiased values, and hard labels are optimistic.
> - An implicit PRM sums $\beta \log(\pi_\phi / \pi_{\text{ref}})$ over each step, with no rollouts.

## Build it step by step

Fill in the functions in this order. Each is tested on the toy above.

1. **`exact_value(prefix, step_p)`.** Takes the outcomes of the first $k$ steps and returns the probability of a correct answer. It is 0 if any outcome is 0, and `np.prod(step_p[k:])` otherwise. Common mistake: forgetting that a wrong step zeroes the value.
2. **`mc_value(prefix, K, rng, step_p)`.** Draws `rng.random((K, T - k)) < step_p[k:]`, counts a rollout as a success when it and the prefix are all correct, and returns the mean and the standard error, `std(ddof=1) / sqrt(K)`. Common mistake: using `ddof=0`, which shrinks the standard error at small $K$.
3. **`vine_advantages(outcomes, K, rng, step_p)`.** Estimates $\hat V$ at the $T$ boundaries, sets the last value to the real reward, and returns `np.diff` and the values. Common mistake: flipping the difference, which gives every correct step a negative advantage.
4. **`math_shepherd_labels(rollout_correct, mode)`.** Takes a `(T, K)` array of 0/1 rollout outcomes. Hard is `max(axis=1)`, soft is `mean(axis=1)`. Common mistake: rounding the soft label.
5. **`implicit_prm_rewards(logp, logp_ref, step_ids, beta, n_steps)`.** Sums $\beta(\log \pi - \log \pi_{\text{ref}})$ over the tokens of each step with `np.bincount`. Common mistake: averaging in place of summing, which breaks the sum to the sequence log ratio.
6. **`credit_errors(step_p, K, n_samples, seed)`.** Samples responses, computes the exact advantage of every step, and returns the mean squared error of the uniform and vine estimates, per step.

## Where this breaks

- **Rollouts are expensive on long solutions.** $T \times K$ grows with both the number of steps and the rollouts per step.
- **Step boundaries are a choice.** A real model writes free text, so a boundary is a heuristic, such as a line break.
- **Rare successes break hard labels and small $K$.** A prefix with $V = 0.01$ almost always shows 0 correct rollouts at $K = 16$. The estimate is then 0, and the advantage cannot tell a doomed prefix from a hard one.
- **A learned or implicit PRM drifts.** The policy moves away from the data the scorer saw.

## Further reading

The following sources cover the methods in this chapter:

- [Kazemnejad et al., "VinePPO: Refining Credit Assignment in RL Training of LLMs" (2024)](https://arxiv.org/abs/2410.01679): Monte Carlo value estimates from intermediate states in place of a learned critic.
- [Wang et al., "Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations" (2023)](https://arxiv.org/abs/2312.08935): hard and soft step labels from completions.
- [Yuan et al., "Free Process Rewards without Process Labels" (2024)](https://arxiv.org/abs/2412.01981): process rewards from an outcome model parameterized as a log ratio.
- [Lightman et al., "Let's Verify Step by Step" (2023)](https://arxiv.org/abs/2305.20050): human step labels and the case for process supervision.
- [Stanford CS336, Spring 2026, Lecture 16, "Post-training - RLVR"](https://cs336.stanford.edu/): the RL-with-verifiable-rewards lecture.
- [Stanford CS224R, Spring 2026, "Actor-Critic Methods"](https://cs224r.stanford.edu/): the critic and advantage that this chapter replaces with rollouts.
- [Lambert, *Reinforcement Learning from Human Feedback* (book)](https://rlhfbook.com/): the RL chapters cover the policy gradient and advantage estimation. Chapter numbers have changed between revisions, so find the chapter by title.

Open questions that researchers are working on:

- How many rollouts per boundary does a given task need, and can the budget be spent more at the steps where the value changes most?
- Does the implicit process reward stay a reliable credit signal as the policy moves away from the data its outcome model saw?
- How should a step boundary be chosen for free-form reasoning, where the model does not write clean steps?

## The exercise

You build the credit-assignment pieces on a four-step toy whose prefix values are known exactly. Write `exact_value`, `mc_value` with its standard error, `vine_advantages`, `math_shepherd_labels` in hard and soft modes, `implicit_prm_rewards`, and `credit_errors`.

The checks compare `mc_value` with the exact value over 400 seeds and confirm its standard error. They confirm that `vine_advantages` sum telescopically, recover the exact advantages within a tolerance tied to the standard error as $K$ grows, and have the right sign. They compute the expected hard and soft labels exactly by summing over every rollout pattern. They confirm that the implicit step rewards add to the sequence log ratio, and that the vine error is under a quarter of the uniform error at every step.
