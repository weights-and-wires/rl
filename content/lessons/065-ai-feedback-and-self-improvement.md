---
title: AI feedback and self-improvement
slug: 065-ai-feedback-and-self-improvement
part: "Part 9 — RL for language models"
summary: Train a model on its own filtered samples, see that this is reward-weighted regression, and see why it collapses, then play it against its past self with SPIN.
minutes: 40
gpu: false
objectives:
  - Show that expert iteration (STaR, ReST) is reward-weighted regression with a 0/1 reward, and check it exactly.
  - Compute the exact M-step on a toy, and say why diversity collapses.
  - Implement the SPIN loss as its paper writes it.
  - Show that SPIN's first step in each iteration is scaled supervised fine-tuning.
prereqs:
  - 031-the-policy-gradient
  - 060-direct-preference-optimization
  - 063-reasoning-with-verifiable-rewards
  - 064-process-rewards-and-search
exercise: 065-ai-feedback-and-self-improvement
---

# AI feedback and self-improvement

Human labels are the expensive part of RLHF: every comparison takes a person's time. A language model reads well, so a model can supply labels or checks instead. Constitutional AI and RLAIF (Bai and colleagues, 2022; Lee and colleagues, 2023) let a model compare answers against written principles. Self-rewarding models (Yuan and colleagues, 2024) let the model being trained judge its own answers.

This chapter follows the simplest form of the idea: a model that trains on its own outputs. Suppose a model solves a hard problem 9% of the time. Sample many answers, keep the ones a check accepts, and fine-tune on them. Why might this make the model better, what does it secretly compute, and where does it stop working? The answers are that it is a policy-gradient method in disguise, and that it narrows the model until one solution is left. Then you will see a second loop, SPIN, that needs no check at all.

## What you already know

- The REINFORCE gradient of expected reward is $\mathbb{E}_{y \sim \pi_\theta}[r(y)\nabla_\theta \log \pi_\theta(y)]$, from [The policy gradient and REINFORCE](/lessons/031-the-policy-gradient).
- DPO trains on a chosen and a rejected response with the loss $-\log\sigma\big(\beta\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\big)$, from [Direct preference optimization](/lessons/060-direct-preference-optimization).
- A verifiable reward is a 0/1 check such as "the answer is correct", from [Reasoning with verifiable rewards](/lessons/063-reasoning-with-verifiable-rewards).
- The digit-sum toy from [Process rewards and test-time search](/lessons/064-process-rewards-and-search): four digits from 0 to 7, correct when they sum to a target, with a per-step table of probabilities as the policy.

## Keep what works

The obvious plan is the one above. Generate answers with the current model, keep the ones that pass a check, and fine-tune on them. This is *expert iteration*, named by Anthony and colleagues (2017), where a tree search played the expert. For language models, the check plays that role:

- **STaR** (Zelikman and colleagues, 2022) samples rationales and keeps those that reach the correct answer. For a question it gets wrong, it shows the answer as a hint and asks for a rationale that reaches it, then adds that rationale with the hint removed.
- **ReST** (Gulcehre and colleagues, 2023) alternates a Grow step that samples a dataset and Improve steps that fine-tune on samples whose reward exceeds a rising threshold. **ReST-EM** (Singh and colleagues, 2023) uses a binary reward and frames the loop as expectation-maximization.

The loss is the negative log-likelihood of the kept answers. With reward $r_i \in \{0, 1\}$ for each of $B$ sampled answers:

$$
\mathcal{L}_{\text{filter}}(\theta) = -\frac{1}{\sum_i r_i} \sum_{i=1}^{B} r_i \log \pi_\theta(y_i \mid x_i)
$$

Read it aloud: add up the log-probabilities of the answers that passed, divide by how many passed, and flip the sign. An answer with $r_i = 0$ contributes nothing. Take four answers with sequence log-probabilities $[-2, -3, -1.5, -4]$ and rewards $[1, 0, 1, 0]$. Two pass, so the loss is $(2 + 1.5)/2 = 1.75$.

```python
seq_logps = (token_logps * mask).sum(dim=1)          # (B,)
loss = -(rewards * seq_logps).sum() / rewards.sum().clamp(min=1.0)
```

## Why this is policy gradient

Reward-weighted regression (RWR, Peters and Schaal, 2007) fits a policy to its own samples, weighting each by a non-negative function of its reward:

$$
\mathcal{L}_{\text{RWR}}(\theta) = -\frac{1}{B} \sum_{i=1}^{B} w_i \log \pi_\theta(y_i \mid x_i), \qquad y_i \sim \pi_{\text{old}}
$$

With $w_i = r_i \in \{0, 1\}$, the two losses differ only in what they divide by: $\mathcal{L}_{\text{filter}} = \mathcal{L}_{\text{RWR}} / \bar r$, where $\bar r$ is the pass rate. Dividing by a positive number changes neither the minimizer nor the gradient's direction, so filtering and fine-tuning is RWR with a 0/1 reward. In the four-answer example, $\mathcal{L}_{\text{RWR}} = 3.5/4 = 0.875$ and $\bar r = 0.5$, which gives $1.75$ again.

There is a second connection. At $\theta = \theta_{\text{old}}$, the expected RWR gradient is the REINFORCE gradient of the success rate:

$$
\nabla_\theta\, \mathbb{E}_{y \sim \pi_{\text{old}}}\big[r(y) \log \pi_\theta(y)\big]\Big|_{\theta = \theta_{\text{old}}} = \mathbb{E}_{y \sim \pi_\theta}\big[r(y)\, \nabla_\theta \log \pi_\theta(y)\big] = \nabla_\theta \Pr_{\pi_\theta}(r = 1)
$$

Read it aloud: at the start of each round, fine-tuning on the kept samples pushes the weights in the direction that raises the pass rate fastest, the same direction REINFORCE with no baseline takes. The difference is that expert iteration keeps following a fixed dataset for many steps, while REINFORCE draws fresh samples after every step.

> [!KEY]
> Filter-and-fine-tune is reward-weighted regression with a 0/1 reward. Its first gradient step is the REINFORCE gradient of the pass rate, with no baseline.

## The exact step, and the collapse

What does one round converge to with unlimited samples and a policy that can fit anything? The loss is the cross-entropy between the kept samples and the model, and the kept samples are $\pi_{\text{old}}$ conditioned on success, so the minimizer is that conditional distribution:

$$
\pi_{\text{new}}(y) = \frac{\pi_{\text{old}}(y)\, r(y)}{\sum_{y'} \pi_{\text{old}}(y')\, r(y')}
$$

Read it aloud: keep the old probabilities of the correct answers, zero out the rest, and renormalize. A model that can fit anything reaches 100% success in one step.

The exercise uses a weaker policy, the per-step table. The best table under the loss matches the per-step marginals of the conditional distribution, which is `em_step`. Expectation-maximization guarantees each step never lowers the pass rate, because the pass rate is a likelihood: the probability of the observation "correct". For target 14, the success rate climbs from $0.086$ over seven steps to $0.969$, while the mean per-step entropy falls from $1.89$ to $0.05$ nats.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Exact expectation-maximization steps on the digit-sum toy: success rate rises from 0.086 to 0.969 while mean per-step entropy falls from 1.89 to 0.05 nats">

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
<text x="64.0" y="272" class="fig-small" text-anchor="middle">0</text>
<line x1="137.7" y1="252" x2="137.7" y2="257" class="stroke-muted"/>
<text x="137.7" y="272" class="fig-small" text-anchor="middle">1</text>
<line x1="211.4" y1="252" x2="211.4" y2="257" class="stroke-muted"/>
<text x="211.4" y="272" class="fig-small" text-anchor="middle">2</text>
<line x1="285.1" y1="252" x2="285.1" y2="257" class="stroke-muted"/>
<text x="285.1" y="272" class="fig-small" text-anchor="middle">3</text>
<line x1="358.9" y1="252" x2="358.9" y2="257" class="stroke-muted"/>
<text x="358.9" y="272" class="fig-small" text-anchor="middle">4</text>
<line x1="432.6" y1="252" x2="432.6" y2="257" class="stroke-muted"/>
<text x="432.6" y="272" class="fig-small" text-anchor="middle">5</text>
<line x1="506.3" y1="252" x2="506.3" y2="257" class="stroke-muted"/>
<text x="506.3" y="272" class="fig-small" text-anchor="middle">6</text>
<line x1="580.0" y1="252" x2="580.0" y2="257" class="stroke-muted"/>
<text x="580.0" y="272" class="fig-small" text-anchor="middle">7</text>
<polyline points="64.0,232.9 137.7,231.0 211.4,227.8 285.1,221.8 358.9,208.1 432.6,171.9 506.3,95.1 580.0,38.0" class="stroke-c" fill="none"/>
<polyline points="64.0,31.0 137.7,36.3 211.4,47.1 285.1,65.9 358.9,97.6 432.6,147.2 506.3,207.9 580.0,245.8" class="stroke-bad dashed" fill="none"/>
<circle cx="64.0" cy="232.9" r="4" class="fill-c"/>
<circle cx="137.7" cy="231.0" r="4" class="fill-c"/>
<circle cx="211.4" cy="227.8" r="4" class="fill-c"/>
<circle cx="285.1" cy="221.8" r="4" class="fill-c"/>
<circle cx="358.9" cy="208.1" r="4" class="fill-c"/>
<circle cx="432.6" cy="171.9" r="4" class="fill-c"/>
<circle cx="506.3" cy="95.1" r="4" class="fill-c"/>
<circle cx="580.0" cy="38.0" r="4" class="fill-c"/>
<circle cx="64.0" cy="31.0" r="4" class="fill-bad"/>
<circle cx="137.7" cy="36.3" r="4" class="fill-bad"/>
<circle cx="211.4" cy="47.1" r="4" class="fill-bad"/>
<circle cx="285.1" cy="65.9" r="4" class="fill-bad"/>
<circle cx="358.9" cy="97.6" r="4" class="fill-bad"/>
<circle cx="432.6" cy="147.2" r="4" class="fill-bad"/>
<circle cx="506.3" cy="207.9" r="4" class="fill-bad"/>
<circle cx="580.0" cy="245.8" r="4" class="fill-bad"/>
<text x="322" y="292" text-anchor="middle">expert iteration step</text>
<text x="58" y="14" text-anchor="end" class="fig-small">success rate; entropy as a fraction of its start</text>
<line x1="350" y1="28" x2="380" y2="28" class="stroke-c"/>
<text x="388" y="32" class="fig-small">success rate</text>
<line x1="350" y1="46" x2="380" y2="46" class="stroke-bad dashed"/>
<text x="388" y="50" class="fig-small">mean per-step entropy / 1.888</text>
</svg>
Figure 1. Success rate and entropy over eight expectation-maximization steps. Notice that success barely moves for four steps and then jumps, and that entropy is spent in the same steps: the model ends up writing nearly one solution.
```

Each round keeps only answers that already work, so probability piles onto a few solutions. People restart from the base model each round, as STaR and ReST-EM do, stop after two or three rounds, and sample at a higher temperature.

## SPIN: play against your past self

Expert iteration needs a check. Self-play fine-tuning (SPIN, Chen and colleagues, 2024) needs only a supervised dataset of human answers. At iteration $t$, the model $p_{\theta_t}$ generates its own answer $y'$ to each prompt. The next model is trained to tell the human answer $y$ from $y'$, using its log-probability ratio against $p_{\theta_t}$ as the score. The paper writes the loss as

$$
\mathcal{L}_{\text{SPIN}}(\theta, \theta_t) = \mathbb{E}\Big[\ell\Big(\lambda \log \frac{p_\theta(y \mid x)}{p_{\theta_t}(y \mid x)} - \lambda \log \frac{p_\theta(y' \mid x)}{p_{\theta_t}(y' \mid x)}\Big)\Big], \qquad \ell(z) = \log(1 + e^{-z})
$$

with $y \sim p_{\text{data}}$ and $y' \sim p_{\theta_t}$. Read it aloud: measure how much more likely the model being trained makes each answer than the previous model did, and reward it for raising the human answer more than the model's own.

This is DPO. Set $\beta = \lambda$, chosen $= y$, rejected $= y'$, and reference $= p_{\theta_t}$, and the loss matches term for term, since $\ell(z) = -\log\sigma(z)$. The difference is the data: the rejected answer is always the model's own, and the reference moves forward every iteration.

```python
margin = lam * (logp_real - prev_real) - lam * (logp_synth - prev_synth)
losses = torch.nn.functional.softplus(-margin)       # ell(margin), one per pair
```

For example, with $\lambda = 1$, the human answer has log-probability $-2.0$ under the model being trained and $-2.5$ under the previous one. The model's own answer has $-1.2$ and $-1.0$. The margin is $(-2.0 + 2.5) - (-1.2 + 1.0) = 0.7$, and the loss is $\log(1 + e^{-0.7}) = 0.4032$.

### The first step of each iteration is scaled supervised fine-tuning

Each iteration starts at $\theta = \theta_t$, where both log-ratios are zero, and $\ell'(0) = -\frac{1}{2}$. So the gradient is

$$
\nabla_\theta \mathcal{L}_{\text{SPIN}}\big|_{\theta = \theta_t} = -\frac{\lambda}{2}\Big(\mathbb{E}_{y \sim p_{\text{data}}}\big[\nabla \log p_\theta(y)\big] - \mathbb{E}_{y' \sim p_{\theta_t}}\big[\nabla \log p_\theta(y')\big]\Big)
$$

The second expectation is the average score of a model's own samples, which is always zero. What is left is $\frac{\lambda}{2}$ times the gradient of the ordinary supervised loss. For a softmax over a finite set of answers with logits $\theta$, it is $\frac{\lambda}{2}(p_{\theta_t} - p_{\text{data}})$. That also gives the fixed point Chen and colleagues prove: when $p_{\theta_t} = p_{\text{data}}$, the gradient is zero and SPIN stops moving.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Total variation distance between the model and the data distribution over five SPIN iterations: 0.414, 0.070, 0.022, 0.012, 0.008, 0.0056">

<line x1="70" y1="240" x2="580" y2="240" class="stroke-muted"/>
<line x1="70" y1="240.0" x2="580" y2="240.0" class="stroke-grid"/><text x="64" y="244.0" class="fig-small" text-anchor="end">0</text>
<line x1="70" y1="197.8" x2="580" y2="197.8" class="stroke-grid"/><text x="64" y="201.8" class="fig-small" text-anchor="end">0.1</text>
<line x1="70" y1="155.6" x2="580" y2="155.6" class="stroke-grid"/><text x="64" y="159.6" class="fig-small" text-anchor="end">0.2</text>
<line x1="70" y1="113.3" x2="580" y2="113.3" class="stroke-grid"/><text x="64" y="117.3" class="fig-small" text-anchor="end">0.3</text>
<line x1="70" y1="71.1" x2="580" y2="71.1" class="stroke-grid"/><text x="64" y="75.1" class="fig-small" text-anchor="end">0.4</text>
<rect x="95" y="65.2" width="50" height="174.8" class="fill-d"/>
<text x="120" y="59.2" class="fig-small" text-anchor="middle">0.414</text>
<text x="120" y="258" class="fig-small" text-anchor="middle">start</text>
<rect x="180" y="210.5" width="50" height="29.5" class="fill-d"/>
<text x="205" y="204.5" class="fig-small" text-anchor="middle">0.0699</text>
<text x="205" y="258" class="fig-small" text-anchor="middle">iter 1</text>
<rect x="265" y="230.9" width="50" height="9.1" class="fill-d"/>
<text x="290" y="224.9" class="fig-small" text-anchor="middle">0.0216</text>
<text x="290" y="258" class="fig-small" text-anchor="middle">iter 2</text>
<rect x="350" y="234.8" width="50" height="5.2" class="fill-d"/>
<text x="375" y="228.8" class="fig-small" text-anchor="middle">0.0122</text>
<text x="375" y="258" class="fig-small" text-anchor="middle">iter 3</text>
<rect x="435" y="236.6" width="50" height="3.4" class="fill-d"/>
<text x="460" y="230.6" class="fig-small" text-anchor="middle">0.008</text>
<text x="460" y="258" class="fig-small" text-anchor="middle">iter 4</text>
<rect x="520" y="237.6" width="50" height="2.4" class="fill-d"/>
<text x="545" y="231.6" class="fig-small" text-anchor="middle">0.0056</text>
<text x="545" y="258" class="fig-small" text-anchor="middle">iter 5</text>
<text x="64" y="22" text-anchor="end" class="fig-small">TV distance</text>
<text x="325" y="276" text-anchor="middle">SPIN iteration (100 gradient steps each)</text>
</svg>
Figure 2. The distance between the model and the data distribution, over five SPIN iterations on a model with 16 possible answers. Notice that the first iteration does most of the work, and that each later one gains less.
```

> [!WARNING]
> SPIN is capped at the data. Its fixed point is $p_{\text{data}}$, so it cannot improve beyond the human answers it imitates.

> [!KEY]
> A model can train on its own samples in two ways. Filtering them with a check is reward-weighted regression and a disguised policy gradient, and it collapses onto the solutions that already work. Pitting them against human answers is DPO with the previous iterate as the reference, and it stops at the data distribution.

> [!RECAP]
> - Filter-and-fine-tune is RWR with $w = r \in \{0, 1\}$, divided by the pass rate, and its first gradient is REINFORCE on the pass rate.
> - The exact M-step keeps $\pi_{\text{old}}$ on correct answers and renormalizes. For a per-step policy, it gives the per-step marginals.
> - Repeated steps raise the pass rate and drain entropy.
> - SPIN is $\ell$ of $\lambda$ times the difference of two log-ratios, with $\ell(z) = \log(1 + e^{-z})$. Its fixed point is $p_\theta = p_{\text{data}}$.

## Build it step by step

The checks test the functions in this order. The torch functions take per-token log-probabilities and masks of shape `(B, T)`.

1. **`sequence_logprobs(token_logps, mask)`.** The masked sum over the token axis, shape `(B,)`.
2. **`filtered_sft_loss(token_logps, mask, rewards)`.** Minus the reward-weighted sum of sequence log-probabilities, divided by the number kept. Clamp that count to at least 1. Common mistake: dividing by $B$, which is RWR, not the filtered loss.
3. **`rwr_loss(token_logps, mask, weights)`.** The same sum divided by $B$. The checks confirm the ratio between the two losses and that the RWR gradient at $\theta_{\text{old}}$ is minus the policy gradient.
4. **`em_step(step_probs, target)`.** Enumerate all 4096 sequences with `all_sequences()`, weight each by $\pi_{\text{old}}(y) r(y)$, normalize, and accumulate per-step marginals with `np.add.at`. Common mistake: forgetting to normalize, which leaves rows that sum to the pass rate.
5. **`spin_loss(...)`.** Four `(B,)` tensors of log-probabilities and $\lambda$. Return one loss per pair with `torch.nn.functional.softplus(-margin)`, which computes $\log(1 + e^{-z})$ without overflow. Common mistake: swapping the real and synthetic arguments, which trains the model toward its own outputs.

## Where this breaks

**Hard problems give no signal.** A prompt the model never solves contributes nothing to the filtered loss. STaR's hint-based rationalization is one fix, and a curriculum from easy to hard prompts is another. If the pass rate is 1%, keeping 100 answers takes about 10,000 generations.

**Judges have biases, and the policy learns them.** LLM judges favor whichever answer they see first or second, longer answers, and answers in their own style. Asking in both orders and averaging, $P(y_1 \succ y_2) = \frac{1}{2}(p_{ab} + 1 - p_{ba})$, removes position bias only. When the policy is its own judge, the other biases feed back every round, so people use length control and a fixed held-out human evaluation.

**Right answers can hide wrong reasoning.** A 0/1 check on the final answer keeps a lucky guess with a broken rationale, and expert iteration then trains on it. Process checks from [Process rewards and test-time search](/lessons/064-process-rewards-and-search) help.

## Further reading

The following sources define the methods in this chapter:

- [Bai et al., "Constitutional AI: Harmlessness from AI Feedback" (2022)](https://arxiv.org/abs/2212.08073): the critique-and-revise phase and RL from AI feedback.
- [Lee et al., "RLAIF vs. RLHF" (2023)](https://arxiv.org/abs/2309.00267): RLAIF matches RLHF on three tasks.
- [Zelikman et al., "STaR: Bootstrapping Reasoning With Reasoning" (2022)](https://arxiv.org/abs/2203.14465): filtering rationales and rationalization with a hint.
- [Singh et al., "Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models" (2023)](https://arxiv.org/abs/2312.06585): ReST-EM and the expectation-maximization view.
- [Yuan et al., "Self-Rewarding Language Models" (2024)](https://arxiv.org/abs/2401.10020): the model as its own judge, trained with DPO.
- [Chen et al., "Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models" (2024)](https://arxiv.org/abs/2401.01335): SPIN and its fixed point.

Open questions that researchers are working on:

- How many rounds of self-training can a model run before diversity collapses, and what restores it?
- Can a model judge its own answers without learning the judge's biases?
- Can self-play improve a model beyond the data it started from?

## The exercise

You build the pieces of expert iteration and SPIN. Write `sequence_logprobs`, then `filtered_sft_loss` and `rwr_loss`, then `em_step` for the exact M-step on the digit-sum toy, then `spin_loss`.

The checks verify the filtered loss on the four-answer example, that it equals the RWR loss divided by the pass rate, and that the RWR gradient at $\theta_{\text{old}}$ is minus the policy gradient of the success rate. They verify that `em_step` returns the per-step marginals of the policy conditioned on success, and that iterating it never lowers the success rate and passes $0.99$ in ten steps. For SPIN, they check $\log 2$ at $\theta = \theta_t$, equality with DPO on the pair, finite differences, the first-step gradient $\frac{\lambda}{2}(p_t - p_{\text{data}})$, and that five iterations bring the model within $0.01$ of the data distribution.
