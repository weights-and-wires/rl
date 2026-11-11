---
title: "Capstone: end-to-end post-training"
slug: 084-capstone-post-training
part: "Part 10 — Beyond text, and what can go wrong"
summary: Run supervised fine-tuning, then DPO, with a hand-written LoRA adapter, and judge the result with a written evaluation, not a loss curve.
minutes: 45
gpu: true
objectives:
  - Write a LoRA layer whose start equals the base model, and count which parameters train.
  - Chain SFT and DPO on one adapter, with the SFT policy as the DPO reference.
  - Compute pass@k, a judge win rate with swap consistency, and the KL to the reference from raw numbers.
  - "Plan a written evaluation: held-out pass@k, a win rate, KL, and a reward-hacking check."
  - Run the pipeline on Qwen3-0.6B and read the table it prints.
prereqs:
  - 065-sft-and-the-post-training-pipeline
  - 068-judge-evaluation
  - 070-direct-preference-optimization
  - 073-reasoning-with-verifiable-rewards
  - 083-reward-hacking-and-safe-rl
exercise: 084-capstone-post-training
---

# Capstone: end-to-end post-training

You have a base language model that answers every question with a ramble. You want a model that multiplies two-digit numbers, shows short working, and ends with a line a program can read. No single chapter in this course builds that model. Each one built a part: a loss, a reward, a way to compare two models. This chapter joins the parts into one project, from the raw model to a written report.

The project has two training stages. Supervised fine-tuning (SFT) teaches the format by imitating worked solutions. Direct preference optimization (DPO) then teaches the model to prefer right answers over wrong ones, using pairs the SFT model wrote for itself. A third piece makes both stages cheap enough for one GPU: *LoRA*, which trains a small add-on to each weight matrix and leaves the model's own weights alone.

The training is the easy half. The hard half is telling the reader whether it worked.

## What you already know

- SFT is the pretraining loss on response tokens only, with the prompt masked, from [Supervised fine-tuning and the post-training pipeline](/lessons/065-sft-and-the-post-training-pipeline).
- The DPO loss is $-\log \sigma(\beta h)$, where $h$ is the chosen log-ratio minus the rejected one, from [Direct preference optimization](/lessons/070-direct-preference-optimization).
- pass@k is the chance that at least one of $k$ samples is right, from [Reasoning with verifiable rewards](/lessons/073-reasoning-with-verifiable-rewards).
- A judge can have position bias, which swap consistency measures, from [Evaluate a reward model and a judge](/lessons/068-judge-evaluation).
- A proxy reward can climb while the real goal falls, from [Reward hacking, safety, and the frontier](/lessons/083-reward-hacking-and-safe-rl).

## The pipeline, stage by stage

The following table lists each stage, what it trains, and the chapter whose ideas it reuses.

| Stage | What changes | Reused from |
|---|---|---|
| 0. Base | Nothing. The adapter's $B$ is zero. | [Language models as policies](/lessons/066-language-models-as-policies) |
| 1. SFT | The adapter, on worked solutions. | [SFT](/lessons/065-sft-and-the-post-training-pipeline) |
| 2. DPO | The same adapter, on pairs from the SFT policy. The SFT policy is the reference. | [DPO](/lessons/070-direct-preference-optimization), [verifiable rewards](/lessons/072-verifiable-rewards) |
| 3. Evaluate | Nothing. | [judges](/lessons/068-judge-evaluation), [pass@k](/lessons/073-reasoning-with-verifiable-rewards), [reward hacking](/lessons/083-reward-hacking-and-safe-rl) |

Why DPO and not GRPO for stage 2? DPO needs one batch of pairs and no sampling inside the training loop, so it is cheap to debug. If you swap in [GRPO](/lessons/071-group-relative-policy-optimization), only the loss changes.

## LoRA: train a small change, not the matrix

A weight matrix $W$ with shape $(\text{out}, \text{in})$ holds millions of numbers. Fine-tuning changes all of them, and the optimizer must store extra numbers for each one. *LoRA* (low-rank adaptation, Hu and colleagues, 2021) bets that the useful change is simple: a *low-rank* matrix, the product of two thin ones. You freeze $W$ and learn $A$ with shape $(r, \text{in})$ and $B$ with shape $(\text{out}, r)$, where the rank $r$ is small, such as 8 or 16. The adapted layer computes

$$
y = x W^\top + \frac{\alpha}{r}\, x A^\top B^\top
$$

Read it aloud: the base layer's output, plus a correction that squeezes $x$ down to $r$ numbers with $A$ and expands it back with $B$. The number $\alpha$ sets how loud the correction is, and dividing by $r$ keeps the loudness the same when you change the rank. The correction equals $x\,(\frac{\alpha}{r} B A)^\top$, so the adapter is the update $\Delta W = \frac{\alpha}{r} B A$. The code never builds that $(\text{out}, \text{in})$ matrix.

### A layer by hand

Take $W = I$ with two inputs, $r = 1$, $\alpha = 2$, so the scale is 2. Let $A = (1, 1)$ and $B = (0.5, 0)^\top$, and feed $x = (1, 2)$.

- Base output: $xW^\top = (1, 2)$.
- Squeeze: $xA^\top = 1 + 2 = 3$.
- Expand: $3 \cdot (0.5, 0) = (1.5, 0)$, then scale by 2 to get $(3, 0)$.
- Output: $(1, 2) + (3, 0) = (4, 2)$.

Check it as an update: $\Delta W = 2BA = \begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix}$, and $\Delta W\,x = (3, 0)$. The two paths agree.

### Start equal to the base, and the trainable set

You want training to begin at the base model, not somewhere near it. So $A$ gets random values and $B$ gets zeros. The product $BA$ is zero, the adapted layer equals the base layer, and the first gradient step has a signal to follow. The gradient to $B$ is nonzero at the start. The gradient to $A$ is exactly zero at the start, because $A$ reaches the output only through $B$, and $B = 0$. After one step $B$ moves, and then $A$ learns too.

If you made both random, the start would differ from the base, and if you swapped the roles, the adapter would train differently. A check catches both.

The trainable set is only $A$ and $B$ in each wrapped layer. A layer of shape $(1024, 1024)$ at rank 8 trains $8 \times (1024 + 1024) = 16{,}384$ numbers, against $1{,}048{,}576$ in the layer: $1/64$. Optimizer state shrinks by the same factor, and that is the reason LoRA fits on one GPU.

```python
def lora_linear(x, W, A, B, alpha, r):
    return x @ W.T + (alpha / r) * ((x @ A.T) @ B.T)
```

## The toy pipeline

The exercise runs the whole pipeline on a language model small enough to check exactly. It has 4 tokens. A sequence is a prompt token followed by 3 response tokens, and each next-token distribution depends on the token before. Its logit table is one adapted layer applied to one-hot inputs, so the adapter is a rank-2 change to a $4 \times 4$ table. With $4^3 = 64$ possible responses per prompt, you can enumerate every response and compute the KL exactly.

Stage 1 trains on 96 demonstrations from a *teacher*, the base plus a rank-2 change, so the adapter can reach it. Stage 2 samples pairs from the SFT policy, calls the response with more zero tokens the chosen one, and trains with DPO at $\beta = 0.5$ against the SFT policy as the reference.

```figure
<svg viewBox="0 0 600 230" role="img" aria-label="Three small line charts from one toy run: SFT demonstration loss falls, then the DPO margin and the KL to the SFT policy both rise">
  <line x1="60" y1="190.0" x2="190" y2="190.0" class="stroke-muted"/>
  <line x1="60" y1="60.0" x2="60" y2="190.0" class="stroke-muted"/>
  <polyline points="60.0,71.0 66.5,134.3 73.0,138.2 79.5,139.5 86.0,140.0 92.5,140.3 99.0,140.4 105.5,140.5 112.0,140.6 118.5,140.6 125.0,140.7 131.5,140.7 138.0,140.7 144.5,140.7 151.0,140.7 157.5,140.8 164.0,140.8 170.5,140.8 177.0,140.8 183.5,140.8 190.0,140.8" class="stroke-a" fill="none" stroke-width="2.5"/>
  <text x="60" y="38.0" class="fig-bold">1. SFT: demo NLL</text>
  <text x="56" y="64.0" class="fig-small" text-anchor="end">1.8</text>
  <text x="56" y="194.0" class="fig-small" text-anchor="end">0.0</text>
  <text x="190" y="208.0" class="fig-small" text-anchor="end">400 steps</text>
  <line x1="270" y1="190.0" x2="400" y2="190.0" class="stroke-muted"/>
  <line x1="270" y1="60.0" x2="270" y2="190.0" class="stroke-muted"/>
  <polyline points="270.0,185.0 283.0,168.7 296.0,153.7 309.0,139.8 322.0,126.9 335.0,114.9 348.0,103.6 361.0,93.1 374.0,83.2 387.0,73.8 400.0,65.0" class="stroke-b" fill="none" stroke-width="2.5"/>
  <text x="270" y="38.0" class="fig-bold">2. DPO: margin</text>
  <text x="266" y="64.0" class="fig-small" text-anchor="end">2.8</text>
  <text x="266" y="194.0" class="fig-small" text-anchor="end">0.2</text>
  <text x="400" y="208.0" class="fig-small" text-anchor="end">40 steps</text>
  <line x1="470" y1="190.0" x2="570" y2="190.0" class="stroke-muted"/>
  <line x1="470" y1="60.0" x2="470" y2="190.0" class="stroke-muted"/>
  <polyline points="470.0,190.0 480.0,188.1 490.0,182.6 500.0,174.0 510.0,162.8 520.0,149.4 530.0,134.3 540.0,118.1 550.0,101.1 560.0,83.7 570.0,66.2" class="stroke-d" fill="none" stroke-width="2.5"/>
  <text x="470" y="38.0" class="fig-bold">KL to SFT (nats)</text>
  <text x="466" y="64.0" class="fig-small" text-anchor="end">1.05</text>
  <text x="466" y="194.0" class="fig-small" text-anchor="end">0.00</text>
  <text x="570" y="208.0" class="fig-small" text-anchor="end">40 steps</text>
</svg>
Figure 1. One toy run (seed 0). SFT lowers the demonstration loss from 1.648 to 0.681. DPO then raises the held-out margin from 0.297 to 2.716 while the exact KL to the SFT policy climbs to 0.996 nats. The margin and the KL rise together, which is why you report both.
```

The margin is the mean of $\log \pi(y_w) - \log \pi(y_l)$ on held-out pairs. The KL is the sum over all 64 responses, averaged over the 4 prompts.

> [!KEY]
> One adapter carries both stages. SFT moves it away from zero, and the SFT policy then becomes the reference for DPO. The reference is a frozen snapshot, not the base model, so the KL you report measures what stage 2 changed.

## The written evaluation

A loss curve tells you the optimizer worked, not that the model got better. Write the evaluation plan before you train, so the numbers cannot choose the story.

| Question | Measurement | The failure it catches |
|---|---|---|
| Does it solve held-out problems? | pass@1 and pass@4 from $n = 8$ samples per question, with the unbiased estimator. | Memorized training questions. |
| Is it better than the SFT model, head to head? | A win rate of DPO over SFT, with ties counting half, in both orders. | A judge that prefers the first response, or the longer one. |
| Is the judge reliable? | Swap consistency: the fraction of pairs whose winner survives swapping the order. | Position bias. A judge that always picks the first slot scores every model 0.5. |
| How far did it move? | The KL to the SFT policy, per token. | A small gain bought with a large drift. |
| Did it hack the reward? | Length, format-only answers, and the proxy against a held-out check. | A reward that rises while the real goal does not. |

Two rules keep the table honest. Compute pass@k with the unbiased estimator $1 - \binom{n-c}{k} / \binom{n}{k}$, not $1 - (1 - c/n)^k$, which is biased for small $n$: with 8 samples and 2 right, pass@4 is $1 - 15/70 = 0.786$. And average a win rate over both orders. For the verdicts `[1, 1, -1, 0]` with A first and `[1, -1, -1, 0]` with B first, two of four pairs flip with the order, so the swap consistency is 0.5 and the win rate is 0.625.

> [!WARNING]
> A win rate from a judge you have not checked for position bias is a measurement of the judge. Report its swap consistency beside the win rate, and if the consistency is low, fix the judge before you trust the rate.

### What to report

The report has six lines, and each one says what it was measured on:

1. The base, SFT, and DPO pass@1 and pass@4 on the same held-out questions.
2. The win rate of DPO over SFT, the judge, and its swap consistency.
3. The KL to the SFT policy, per token.
4. The mean response length at each stage.
5. The number of trainable parameters and the fraction of the model they are.
6. What failed, what you skipped, and the seeds you ran. One seed is an anecdote, and [Evaluating RL honestly](/lessons/059-evaluating-rl) shows how to put an interval on it.

## Run the pipeline on Qwen3-0.6B

On the GPU worker, **Run** post-trains `Qwen/Qwen3-0.6B` on two-digit multiplication, with your functions. The worker has no `peft`, so a `LoRALinear` wrapper calls your `lora_linear` on all seven projections in every layer, at rank 16 and $\alpha = 32$. The base weights stay in bfloat16 and frozen. The adapter keeps float32 master weights and runs under bfloat16 autocast, for the reason in [RLHF with PPO](/lessons/069-rlhf-with-ppo).

The run follows the stages in order:

- **Base:** evaluates the model with the adapter at zero, which is the base model.
- **SFT:** 60 steps of 16 worked solutions at learning rate $2 \times 10^{-4}$, such as `37 * 24 = 37 * 20 + 37 * 4 = 740 + 148 = 888. Answer: 888`, with the loss on the response only.
- **DPO:** samples 8 responses for each of 48 questions from the SFT policy, pairs a right response with a wrong one, and runs 30 steps at $3 \times 10^{-5}$ with $\beta = 0.1$ against the SFT adapter.
- **Evaluation:** for each stage it prints pass@1, pass@4 from 8 samples per held-out question, mean length, and the per-token KL to SFT, then the win rate of DPO over SFT. The judge is the answer-checking rule, which cannot be position-biased, so the swap consistency prints as 1.00.

Without a GPU, **Run** prints the toy pipeline for three seeds instead. The real table comes from one seed and 48 questions, so read the direction, not the third digit.

> [!RECAP]
> - LoRA computes $xW^\top + \frac{\alpha}{r} xA^\top B^\top$, trains only $A$ and $B$, and starts at the base model because $B = 0$.
> - SFT and DPO share one adapter, and the SFT policy is the DPO reference.
> - The evaluation reports held-out pass@k, a win rate with swap consistency, the KL, and a reward-hacking check.
> - The checks run the whole pipeline on a toy model you can enumerate.

## Build it step by step

The checks test the functions in this order. The helpers, the toy pipeline, and the real run are given.

1. **`lora_linear(x, W, A, B, alpha, r)`.** Takes `x` of shape `(..., in)` and returns `(..., out)`. Compute the adapter path as `(x @ A.T) @ B.T`. Common mistake: dropping the `alpha / r` scale, which hides at `alpha == r`.
2. **`init_lora(in_features, out_features, r, seed)`.** Returns `A` as normal draws with standard deviation `1 / sqrt(in)` and `B` as exact zeros. Common mistake: swapping the two, or initializing both at random.
3. **`lora_param_count(shapes, r)`.** Sums `r * (out + in)` over the layers.
4. **`lora_grads(x, W, A, B, alpha, r, y)`.** Returns `(dA, dB)` of half the squared error, with `W` detached. The check compares both with finite differences and confirms `dA` is zero at `B = 0`.
5. **`train_adapter(loss_fn, W, A, B, alpha, r, steps, lr)`.** Gradient descent on copies of `A` and `B`, with the logit table rebuilt on every step. Common mistake: updating the arguments in place.
6. **`sequence_kl(logp, logq)`.** The sum of `p * (log p - log q)`, averaged over the leading axes.
7. **`pass_at_k(n, c, k)`.** The unbiased estimator, for a count or an array of counts. Common mistake: the plug-in $1 - (1 - c/n)^k$.
8. **`judge_win_rate(pref_ab, pref_ba)`.** A verdict of `-1` with B shown first is a win for A. Return the two-order average and the swap consistency.
9. **The pipeline checks** call the given `run_toy_pipeline`, which uses your functions. Across five seeds, SFT must lower the demonstration loss, DPO must raise the margin and the reward, and the KL to SFT must stay under 2 nats.

## Where this breaks

**Rank too small.** A behavior that needs a bigger change than rank $r$ cannot fit. Raise $r$ first.

**A reference that moved.** If you compute the DPO reference with the live adapter instead of a frozen snapshot, the margin is zero on every step. Take the log-probabilities once, before stage 2.

**Gains that are format.** SFT can lift pass@1 by teaching the `Answer:` line alone. Compare the right-answer rate among responses that have the line, and read [Reward hacking, safety, and the frontier](/lessons/083-reward-hacking-and-safe-rl) for what to do when the proxy and the goal diverge.

**One seed, 48 questions.** The Run block is a demonstration. A claim needs several seeds and an interval.

## Further reading

The following sources define the pieces of this chapter:

- [Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)](https://arxiv.org/abs/2106.09685): the update $\Delta W = BA$, the zero initialization of $B$, and the $\alpha / r$ scaling.
- [Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023)](https://arxiv.org/abs/2305.18290): the DPO loss used in stage 2.
- [Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (2024)](https://arxiv.org/abs/2402.03300): GRPO, the alternative for stage 2.
- [Lambert et al., "Tulu 3: Pushing Frontiers in Open Language Model Post-Training" (2024)](https://arxiv.org/abs/2411.15124): an open recipe with SFT, preference tuning, and verifiable-reward RL.
- [Lambert, *Reinforcement Learning from Human Feedback*](https://rlhfbook.com/): the chapters [Training Overview](https://rlhfbook.com/c/03-training-overview), [Instruction Fine-Tuning](https://rlhfbook.com/c/04-instruction-tuning), and [Evaluation](https://rlhfbook.com/c/16-evaluation).

For university courses, see Stanford CS336, *Language Modeling from Scratch* (Spring 2026), Lecture 15, "Mid/post-training (SFT/RLHF)", and Lecture 16, "Post-training - RLVR", and Stanford CS 224R, *Deep Reinforcement Learning* (Spring 2026), Lecture 9, "RL for LLMs: Preference Optimization" (guest lecture), and Lecture 10, "RL for LLMs: Reasoning" (guest lecture).

Open questions that researchers are working on:

- How much of a post-training gain survives a different judge, a different prompt format, and a different seed?
- When does a rank-$r$ adapter match full fine-tuning for reinforcement learning, and when does it cap what the policy can learn?

## The exercise

You build the adapter and the evaluation, and the given pipeline joins them. Write `lora_linear`, `init_lora`, `lora_param_count`, `lora_grads`, and `train_adapter`, then `sequence_kl`, `pass_at_k`, and `judge_win_rate`. The given `run_toy_pipeline` calls them to run SFT and DPO on the toy model, and the given `real_run` calls them on Qwen3-0.6B when a GPU is available.

The checks run on the CPU with toy tensors. They verify the start equals the base and equals $W + \frac{\alpha}{r}BA$ once $B$ is set, the exact parameter count, the gradients against finite differences, and that `train_adapter` leaves `W` untouched. They verify `sequence_kl`, `pass_at_k`, and `judge_win_rate` against hand values, including a judge that always prefers the first position. Last, they run the toy pipeline on five seeds.
