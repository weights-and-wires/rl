---
title: AI feedback and self-improvement
slug: 40-ai-feedback-and-self-improvement
part: "Part 8 — Reinforcement learning for language models"
summary: Replace human labels with AI judges, and let a model improve on its own filtered or self-judged outputs.
minutes: 60
gpu: false
objectives:
  - Describe Constitutional AI and RLAIF, and remove an AI judge's position bias by averaging both orders.
  - Build self-rewarding preference pairs from a model's own judge scores.
  - Show that expert iteration (STaR, ReST) is reward-weighted regression with a 0/1 reward, and check it exactly.
  - Implement the SPIN loss as its paper writes it, and show its first step in each iteration is scaled SFT.
prereqs:
  - 34-reward-models
  - 36-direct-preference-optimization
  - 38-reasoning-with-verifiable-rewards
  - 39-process-rewards-and-search
exercise: 40-ai-feedback-and-self-improvement
---

# AI feedback and self-improvement

Human preference labels are the expensive part of RLHF. Every comparison
takes a person's time, and for harmful content it asks people to read things
nobody should have to read in bulk. Large language models are good at
reading, though. If one can compare two answers well enough, it can supply
the labels instead.

This lesson follows that idea through four methods. Constitutional AI and
RLAIF replace the human labeler with a model that follows written principles.
Self-rewarding models go further and let the model being trained judge its own
answers. Expert iteration, the core of STaR and ReST, keeps only the model's
answers that pass a check and fine-tunes on them. SPIN pits the model against
its own previous version and trains it to tell human answers from its own.

Each method is a loop in which a model produces its own training data. The
loops look different, but most of them reduce to losses you already know. You
show that filtering and fine-tuning is a policy-gradient method in disguise,
and that SPIN is DPO with a moving reference.

## What you already know

- A Bradley-Terry reward model learns from pairwise preferences
  (`34-reward-models`).
- DPO trains a policy on preference pairs directly, with the loss
  $-\log \sigma\big(\beta \log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\big)$
  (`36-direct-preference-optimization`).
- A verifiable reward is a 0/1 check, such as "the answer is correct"
  (`38-reasoning-with-verifiable-rewards`).
- The REINFORCE gradient of expected reward is
  $\mathbb{E}_{y \sim \pi_\theta}[r(y) \nabla_\theta \log \pi_\theta(y)]$
  (`15-policy-gradient-theorem`).
- The digit-sum toy from `39-process-rewards-and-search`: four digits from 0
  to 7, correct when they sum to a target.

## Replace human labels with an AI judge

Constitutional AI, from Bai and colleagues (2022), trains a harmless assistant
without human harmlessness labels. The constitution is a short list of
written principles, such as "choose the response that is less harmful". The
method has two phases:

1. **Supervised phase.** Sample answers to harmful prompts from a
   helpful-only model. Ask the model to critique each answer against a
   principle and then revise it. Fine-tune on the revisions.
2. **RL phase.** Sample pairs of answers, and ask a feedback model which one
   better follows a principle. Train a preference model on these AI labels,
   mixed with human labels for helpfulness, and run RL against it.

The paper calls the second phase reinforcement learning from AI feedback
(RLAIF). It reports an assistant that is harmless but not evasive: it engages
with harmful questions and explains its objections. Lee and colleagues (2023)
compared RLAIF with RLHF head to head. On summarisation, helpful dialogue, and
harmless dialogue, RLAIF performed comparably. It also beat supervised
fine-tuning when the labeler was the same size as the policy, or the very same
checkpoint.

The AI label is usually a probability read off the judge's next token. Show
the judge a prompt, two answers labeled (A) and (B), and a principle $c$, and
normalise its probabilities for the two option tokens:

$$
p_{\text{AI}}(y_1 \succ y_2) = \frac{p_{\text{LM}}(\text{(A)} \mid x, y_1, y_2, c)}{p_{\text{LM}}(\text{(A)} \mid x, y_1, y_2, c) + p_{\text{LM}}(\text{(B)} \mid x, y_1, y_2, c)}
$$

Read it as: of the probability the judge puts on answering (A) or (B), the
share that goes to (A), where (A) is $y_1$. This soft label replaces a human's
0 or 1 in the Bradley-Terry loss.

### Remove the position bias

LLM judges tend to favor whichever answer they see first, or second,
regardless of content. That's position bias. The fix is to ask twice, once in
each order, and average. Let $p_{ab}$ be the judge's probability that the
first answer shown wins when it sees $(y_1, y_2)$, and $p_{ba}$ the same
probability when it sees $(y_2, y_1)$. In the second order, $y_1$ is the
second answer shown, so it wins with probability $1 - p_{ba}$:

$$
P(y_1 \succ y_2) = \tfrac{1}{2}\big(p_{ab} + 1 - p_{ba}\big)
$$

Read it as: two estimates of the chance that $y_1$ wins, one from each order,
averaged.

Here's a worked example. A judge gives $p_{ab} = 0.8$ and $p_{ba} = 0.4$. The
debiased preference is $\frac{1}{2}(0.8 + 1 - 0.4) = 0.7$. The judge's first
position gets an average of $\frac{1}{2}(0.8 + 0.4) = 0.6$, a bias of 0.1 above
the fair 0.5, and the average cancels it. A judge with pure position bias, say
$p_{ab} = p_{ba} = 0.7$, gets exactly 0.5, the right answer for a judge that
knows nothing.

```python
p_y1_wins = 0.5 * (p_first_ab + 1.0 - p_first_ba)
```

## Let the model judge itself

A reward model trained on human labels stays fixed while the policy
improves. Self-rewarding language models, from Yuan and colleagues (2024),
make the policy its own judge, so both can improve together. Starting from
Llama 2 70B fine-tuned on seed instruction and judging data, each iteration
does the following:

1. Sample $N = 4$ answers to each prompt.
2. Ask the same model to score each answer out of 5 with an LLM-as-a-judge
   prompt that adds points for criteria such as relevance and clarity. Average
   three sampled judgements per answer.
3. Keep the highest-scoring answer as chosen and the lowest as rejected. Drop
   the prompt if the two scores tie.
4. Train the next model with DPO on these pairs, starting from the current
   one.

After three iterations, the model beat several strong systems on the
AlpacaEval 2.0 leaderboard. The pair rule is a few lines:

```python
chosen, rejected = scores.argmax(axis=1), scores.argmin(axis=1)
rows = np.arange(len(scores))
keep = scores[rows, chosen] > scores[rows, rejected]
```

The judge and the policy are the same model, so any bias the judge has, such
as a taste for long answers, becomes a preference the policy learns. The
check in the exercise is small, but that loop is the method's main risk.

## Keep what works: expert iteration

Expert iteration alternates two steps. Generate answers with the current
model, keep the ones that pass a check, and fine-tune on them. The name comes
from Anthony and colleagues (2017), where a tree search played the expert.
For language models, the check plays that role:

- **STaR**, from Zelikman and colleagues (2022), samples rationales for
  questions and keeps those that reach the correct answer. For a question it
  gets wrong, it shows the correct answer as a hint and asks for a rationale
  that reaches it. This is rationalisation. It adds that rationale with the
  hint removed, fine-tunes, and repeats. Each round restarts fine-tuning from
  the original pretrained model to avoid overfitting.
- **ReST**, from Gulcehre and colleagues (2023), alternates a Grow step that
  samples a dataset from the policy and Improve steps that fine-tune on the
  samples whose reward exceeds a threshold. The threshold rises from one
  Improve step to the next.
- **ReST-EM**, from Singh and colleagues (2023), uses a binary reward and
  frames the loop as expectation-maximisation. Each round samples and filters
  (the E-step) and then fine-tunes the base model (the M-step).

The loss in every case is the negative log-likelihood of the kept answers.
With reward $r_i \in \{0, 1\}$ for each of $B$ sampled answers:

$$
\mathcal{L}_{\text{filter}}(\theta) = -\frac{1}{\sum_i r_i} \sum_{i=1}^{B} r_i \log \pi_\theta(y_i \mid x_i)
$$

Read it as: add up the log-probabilities of the answers that passed, divide by
how many passed, and flip the sign. An answer with $r_i = 0$ contributes
nothing. Here $\log \pi_\theta(y_i \mid x_i)$ is the masked sum of the answer's
token log-probabilities.

```python
seq_logps = (token_logps * mask).sum(dim=1)          # (B,)
loss = -(rewards * seq_logps).sum() / rewards.sum().clamp(min=1.0)
```

For example, take four answers with sequence log-probabilities
$[-2, -3, -1.5, -4]$ and rewards $[1, 0, 1, 0]$. Two pass, so the loss is
$(2 + 1.5) / 2 = 1.75$.

### Expert iteration is reward-weighted regression

Reward-weighted regression (RWR), from Peters and Schaal (2007), fits a policy
to its own samples, weighting each by a non-negative function of its reward.
ReST-EM writes its M-step in this form:

$$
\mathcal{L}_{\text{RWR}}(\theta) = -\frac{1}{B} \sum_{i=1}^{B} w_i \log \pi_\theta(y_i \mid x_i), \qquad y_i \sim \pi_{\text{old}}
$$

Read it as: a weighted average of sequence log-likelihoods over the samples,
with larger weights on better answers. With $w_i = r_i \in \{0, 1\}$, the two
losses differ only in what they divide by:

$$
\mathcal{L}_{\text{filter}} = \frac{\mathcal{L}_{\text{RWR}}}{\bar r}, \qquad \bar r = \frac{1}{B}\sum_i r_i
$$

Read it as: the filtered loss is the RWR loss divided by the batch's pass
rate. Dividing by a positive number changes neither the minimiser nor the
gradient's direction. So filtering and fine-tuning is RWR with a 0/1 reward.
In the four-answer example, $\mathcal{L}_{\text{RWR}} = 3.5 / 4 = 0.875$ and
$\bar r = 0.5$, which gives 1.75 again.

There's a second connection, which STaR's paper points out. At
$\theta = \theta_{\text{old}}$, the expected RWR gradient is the REINFORCE
gradient of the success rate:

$$
\nabla_\theta\, \mathbb{E}_{y \sim \pi_{\text{old}}}\big[r(y) \log \pi_\theta(y)\big]\Big|_{\theta = \theta_{\text{old}}} = \mathbb{E}_{y \sim \pi_\theta}\big[r(y)\, \nabla_\theta \log \pi_\theta(y)\big] = \nabla_\theta \Pr_{\pi_\theta}(r = 1)
$$

Read it as: at the start of each round, fine-tuning on the kept samples pushes
the weights in the direction that raises the pass rate fastest, the same
direction REINFORCE with no baseline takes. The difference is that expert
iteration keeps following that fixed dataset for many steps, while REINFORCE
draws fresh samples after every step.

> [!KEY]
> Filter-and-fine-tune is reward-weighted regression with a 0/1 reward. Its
> first gradient step is the REINFORCE gradient of the pass rate, with no
> baseline.

### The exact M-step, and the collapse it leads to

What does the M-step converge to with unlimited samples and a policy that can
fit anything? The loss is the cross-entropy between the kept samples and the
model. The kept samples come from $\pi_{\text{old}}$ conditioned on success,
so the minimiser is that conditional distribution:

$$
\pi_{\text{new}}(y) = \frac{\pi_{\text{old}}(y)\, r(y)}{\sum_{y'} \pi_{\text{old}}(y')\, r(y')}
$$

Read it as: keep the old probabilities of the correct answers, zero out the
rest, and renormalise. On the digit-sum toy, one such step would reach 100%
success at once.

The exercise uses a weaker policy class, the per-step table from the previous
lesson. The best table under the loss matches the per-step marginals of that
conditional distribution. That's `em_step`. Expectation-maximisation guarantees
that each step never lowers the pass rate. The pass rate is a likelihood,
the probability of the observation "correct". For target 14, the success rate
climbs from 0.086 to 0.095, 0.110, 0.137, 0.199, 0.362, 0.710, and 0.969 over
seven steps. The mean per-step entropy falls from 1.89 to 0.05 nats. The model
ends up nearly always writing one solution.

## Play against your past self: SPIN

Expert iteration needs a check. Self-play fine-tuning (SPIN), from Chen and
colleagues (2024), needs only a supervised dataset of human answers. At
iteration $t$, the model $p_{\theta_t}$ generates its own answer $y'$ to each
prompt. The next model is trained to tell the human answer $y$ from $y'$,
using its log-probability ratio against $p_{\theta_t}$ as the score. The
paper writes the loss as:

$$
\mathcal{L}_{\text{SPIN}}(\theta, \theta_t) = \mathbb{E}_{x \sim q,\; y \sim p_{\text{data}}(\cdot \mid x),\; y' \sim p_{\theta_t}(\cdot \mid x)}\Big[\ell\Big(\lambda \log \frac{p_\theta(y \mid x)}{p_{\theta_t}(y \mid x)} - \lambda \log \frac{p_\theta(y' \mid x)}{p_{\theta_t}(y' \mid x)}\Big)\Big]
$$

Read it as: for each prompt, take a human answer and one of the previous
model's answers. Measure how much more likely the model being trained makes each one
than the previous model did. Reward it for raising the human
answer more than the previous model's answer. The logistic loss
$\ell(z) = \log(1 + e^{-z})$ turns the gap into a penalty that falls toward
zero as the gap grows, and $\lambda$ sets how sharply.

This is DPO. Set $\beta = \lambda$, chosen $= y$, rejected $= y'$, and
reference $= p_{\theta_t}$, and the DPO loss matches term for term, since
$\ell(z) = -\log \sigma(z)$. The difference is the data. The rejected answer
is always the model's own, and the reference moves forward every iteration.

```python
margin = lam * (logp_real - prev_real) - lam * (logp_synth - prev_synth)
losses = torch.nn.functional.softplus(-margin)       # ell(margin), one per pair
```

Here's a worked example with $\lambda = 1$. The human answer has
log-probability $-2.0$ under the model being trained and $-2.5$ under the previous one.
The synthetic answer has $-1.2$ and $-1.0$. The margin is
$(-2.0 + 2.5) - (-1.2 + 1.0) = 0.7$, and the loss is
$\log(1 + e^{-0.7}) = 0.4032$.

### The first step of each iteration is scaled SFT

Each iteration starts at $\theta = \theta_t$, where both log-ratios are zero.
The derivative of $\ell$ at zero is $-\frac{1}{2}$. So the gradient is

$$
\nabla_\theta \mathcal{L}_{\text{SPIN}}\big|_{\theta = \theta_t} = -\frac{\lambda}{2}\Big(\mathbb{E}_{y \sim p_{\text{data}}}\big[\nabla \log p_\theta(y)\big] - \mathbb{E}_{y' \sim p_{\theta_t}}\big[\nabla \log p_\theta(y')\big]\Big)
$$

Read it as: raise the log-probability of the human answers, and lower that of
the model's own answers, each by half of $\lambda$. At $\theta = \theta_t$, the
second expectation is the average score of a model's own samples, which is
always zero. What's left is $\frac{\lambda}{2}$ times the gradient of the
ordinary supervised loss $-\mathbb{E}_{p_{\text{data}}}[\log p_\theta(y)]$.
For a softmax over a finite set of answers with logits $\theta$, it's
$\frac{\lambda}{2}(p_{\theta_t} - p_{\text{data}})$.

That formula also shows the fixed point Chen and colleagues prove. When
$p_{\theta_t} = p_{\text{data}}$, the gradient is zero, and SPIN stops
moving. In the exercise, a model over 16 answers starts uniform, at a total
variation distance of 0.414 from the data distribution. Five SPIN iterations
of 100 gradient steps each bring it to 0.070, 0.022, 0.012, 0.008, and
0.0056.

> [!KEY]
> SPIN is DPO with the human answer as chosen, the model's own answer as
> rejected, and the previous iterate as the reference. Its fixed point is
> $p_\theta = p_{\text{data}}$.

> [!RECAP]
> - Average an AI judge over both orders:
>   $\frac{1}{2}(p_{ab} + 1 - p_{ba})$.
> - Self-rewarding pairs take the highest and lowest judge score per prompt
>   and drop ties.
> - Filter-and-fine-tune is RWR with $w = r \in \{0, 1\}$, divided by the pass
>   rate, and its first gradient is REINFORCE on the pass rate.
> - The exact M-step keeps $\pi_{\text{old}}$ on correct answers and
>   renormalises. For a per-step policy, it's the per-step marginals.
> - SPIN is $\ell$ of $\lambda$ times the difference of two log-ratios, with
>   $\ell(z) = \log(1 + e^{-z})$.

## Build it step by step

The checks test the functions in this order. The torch functions take
per-token log-probabilities and masks of shape `(B, T)`, like the rest of
Part 8.

1. **`debias_judge(p_first_ab, p_first_ba)`.** Takes two `(B,)` arrays and
   returns `(B,)`. One line. Common mistake: averaging $p_{ab}$ and $p_{ba}$
   directly, which measures the position bias instead of removing it.
2. **`preference_pairs(scores)`.** Takes `(B, N)` scores and returns the
   chosen index, the rejected index, and a `keep` mask. Common mistake:
   keeping rows where every score ties, which pairs an answer with itself.
3. **`sequence_logprobs(token_logps, mask)`.** The masked sum over the token
   axis, shape `(B,)`.
4. **`filtered_sft_loss(token_logps, mask, rewards)`.** Minus the reward-weighted
   sum of sequence log-probabilities, divided by the number kept. Clamp that
   count to at least 1. Common mistake: dividing by $B$, which is RWR, not
   the filtered loss.
5. **`rwr_loss(token_logps, mask, weights)`.** The same sum divided by $B$.
   The checks confirm the ratio between the two losses and that the RWR
   gradient at $\theta_{\text{old}}$ is minus the policy gradient.
6. **`em_step(step_probs, target)`.** Enumerate all 4096 sequences with
   `all_sequences()`, weight each by $\pi_{\text{old}}(y) r(y)$, normalise,
   and accumulate per-step marginals with `np.add.at`. Common mistake:
   forgetting to normalise, which leaves rows that sum to the pass rate.
7. **`spin_loss(...)`.** Four `(B,)` tensors of log-probabilities and
   $\lambda$. Return one loss per pair with `torch.nn.functional.softplus(-margin)`,
   which computes $\log(1 + e^{-z})$ without overflow. Common mistake:
   swapping the real and synthetic arguments, which trains the model toward
   its own outputs.

## Cost and scaling

The methods differ in what they generate and what they score. The following
table compares them per round, for $P$ prompts and $N$ samples per prompt:

| Method | Generations | Judge calls | Training step |
|---|---|---|---|
| RLAIF | RL rollouts, plus $2P$ for labels | 2 per pair, both orders | Reward model, then PPO or GRPO |
| Self-rewarding | $PN$ | $3PN$ in the paper | DPO |
| Expert iteration | $PN$ | $PN$ checks, often a Python rule | SFT on the kept samples |
| SPIN | $P$, one per human answer | 0 | A DPO-style step |

Expert iteration and SPIN separate generation from training. You can sample a
whole dataset with a fast inference engine, then fine-tune offline. That makes
them cheaper to run than online RL, where each update waits for fresh
samples. The price is staleness: within a round, the model trains on samples
from an older version of itself. Expert iteration also wastes most samples on
hard prompts. If the pass rate is 1%, keeping 100 answers takes about 10,000
generations. In the exercise, `em_step` enumerates $8^4 = 4096$ sequences, so
its cost grows as $V^T$. That's fine for a toy and impossible for a real
model.

## Where this breaks

**Diversity collapses.** The exact M-step shows it. Each round keeps only the
answers that already work, so probability piles onto a few solutions. On the
toy, entropy fell from 1.89 to 0.05 nats in seven steps. People restart from
the base model each round, as STaR and ReST-EM do, stop after two or three
rounds, and sample at a higher temperature.

**Hard problems give no signal.** A prompt the model never solves contributes
nothing to the filtered loss. STaR's rationalisation, which gives the answer
as a hint, is one fix. Another is a curriculum that moves from easy prompts to
harder ones.

**Judges have biases, and the policy learns them.** LLM judges favor
positions, longer answers, and answers in their own style. Averaging both
orders removes position bias only. When the policy is its own judge, as in
self-rewarding models, those biases feed back every round. Length control and
a fixed held-out human evaluation catch the drift.

**Right answers can hide wrong reasoning.** A 0/1 check on the final answer
keeps a lucky guess with a broken rationale, and expert iteration then trains
on it. Process checks from `39-process-rewards-and-search` help.

**SPIN is capped at the data.** Its fixed point is $p_{\text{data}}$, so it
can't improve beyond the human answers it imitates. Chen and colleagues also
report that gains shrink after a few iterations.

## Further reading

- [Bai and colleagues (2022), Constitutional AI: Harmlessness from AI
  Feedback](https://arxiv.org/abs/2212.08073). The critique-and-revise phase
  and RL from AI feedback, driven by a written constitution.
- [Lee and colleagues (2023), RLAIF vs. RLHF](https://arxiv.org/abs/2309.00267).
  RLAIF matches RLHF on three tasks, and direct RLAIF skips the reward model.
- [Zelikman and colleagues (2022), STaR: Bootstrapping Reasoning With
  Reasoning](https://arxiv.org/abs/2203.14465). Filtered rationales,
  rationalisation with hints, and the policy-gradient view of filtering.
- [Gulcehre and colleagues (2023), Reinforced Self-Training (ReST) for
  Language Modeling](https://arxiv.org/abs/2308.08998). Grow and Improve steps
  with rising reward thresholds, on machine translation.
- [Singh and colleagues (2023), Beyond Human Data: Scaling Self-Training for
  Problem-Solving with Language Models](https://arxiv.org/abs/2312.06585).
  ReST-EM, the EM view with binary rewards, on math and code.
- [Yuan and colleagues (2024), Self-Rewarding Language
  Models](https://arxiv.org/abs/2401.10020). The model judges its own answers,
  and iterative DPO trains on the pairs.
- [Chen and colleagues (2024), Self-Play Fine-Tuning Converts Weak Language
  Models to Strong Language Models](https://arxiv.org/abs/2401.01335). SPIN,
  its loss, and the proof that its optimum is the data distribution.

Researchers are still working on these open questions:

- How far can a model improve on its own judgements before its errors
  compound, and what external signal is the minimum to stop that?
- Can self-improvement keep the diversity that later RL and test-time search
  depend on, instead of collapsing onto a few solutions?
- When the judge and the policy share weights, how do you measure and remove
  the judge's biases before the policy learns them?

## The exercise

You build the small computations inside each loop. Write `debias_judge` for
RLAIF labels and `preference_pairs` for self-rewarding pairs. Then write
`sequence_logprobs`, `filtered_sft_loss`, and `rwr_loss` in torch, `em_step`
for the exact expert-iteration step on the digit-sum toy, and `spin_loss` as
Chen and colleagues write it.

The checks confirm the worked examples: 0.7 for the judge, 1.75 for the
filtered loss, and 0.4032 for SPIN. They confirm that the filtered loss is the
RWR loss divided by the pass rate, with gradients to match. By enumerating all
4096 sequences, they check that the RWR gradient at $\theta_{\text{old}}$ is
minus the gradient of the success rate. They check that `em_step` returns the
marginals of the success-conditioned policy and never lowers the pass rate.
For SPIN, they check $\log 2$ at $\theta = \theta_t$, equality with DPO,
finite differences, and the first gradient $\frac{\lambda}{2}(p_{\theta_t} - p_{\text{data}})$.
The last check runs five SPIN iterations and expects a total variation below
0.01.
