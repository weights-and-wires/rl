---
title: Supervised fine-tuning and the post-training pipeline
slug: 054-sft-and-the-post-training-pipeline
part: "Part 9 — RL for language models"
summary: Mask prompt tokens in the next-token loss, pack examples so attention stays inside each one, and see where the reference policy comes from.
minutes: 40
gpu: false
objectives:
  - Write the next-token supervised fine-tuning loss with prompt masking and the one-token label shift.
  - Explain why this loss is behavior cloning and why a mistake early in a reply compounds.
  - Pack examples so position ids restart and attention cannot cross the boundary between them.
  - Keep the k highest-scoring samples of each prompt and match that loss to reward-weighted regression with weights of 0 or 1.
  - Say which checkpoint later chapters freeze and call the reference policy.
prereqs:
  - 048-imitation-learning
exercise: 054-sft-and-the-post-training-pipeline
---

# Supervised fine-tuning and the post-training pipeline

You ask a base model "What is 2 + 2?" and it continues the pattern: "What is 3 + 3? What is 4 + 4?" A base model was trained to predict the next token of whatever text came before, so a question looks like the start of more questions. You have a pile of questions paired with the replies a person wrote. The first post-training step copies those replies.

That step is *supervised fine-tuning*, also called instruction tuning: the same next-token loss, a *chat template* that marks who said what, and a mask on the prompt. [Language models as policies](/lessons/055-language-models-as-policies) puts a reward on a finished reply and a leash back to a *reference policy*, a frozen copy of this checkpoint. A leash back to the base model pulls toward a model that does not answer.

## What you already know

- Behavior cloning fits a policy to an expert's actions, and the error compounds because the learner trains on the expert's states and then acts on its own.
- A policy $\pi(a \mid s)$ is the probability of each action in a state.
- Next-token training minimizes the negative log-probability of the token that came next.

## Copy the reply, not the question

A *chat template* turns a list of messages into one token row the model can train on. Roles alternate. A short template looks like this:

```text
<|user|>
What is 2 + 2?
<|assistant|>
4<|end|>
```

The user span is the prompt $x$. The assistant span, including the end-of-turn token, is the response $y$. The model has to learn to stop, so that token stays in the loss. Generation starts after `<|assistant|>`.

Pretraining puts the loss on every next token. Supervised fine-tuning keeps a term only when the token being predicted is a response token. Predicting the user's question teaches the model to ask. In the exercise the *prompt mask* is 1 on prompt tokens and 0 on response tokens. The formula below writes $m_t = 1$ for a response token, which is the complement of that array.

### A three-token example

Take a vocabulary of three tokens and the row $u = (0, 1, 2)$, with prompt mask $(1, 1, 0)$. Tokens 0 and 1 are the user turn. Token 2 is the whole reply. The logits at the three positions are $[3, 0, 0]$, $[0, 0, 2]$, and $[1, 1, 1]$.

The network reads left to right. The logits at position $t$ have seen $u_0, \ldots, u_t$ and score the next token, $u_{t+1}$. That one-token *label shift* is the indexing. Position 0 scores token 1, still a prompt token, and the mask drops it. Position 1 is the last prompt token and scores token 2, the reply, so the mask keeps it. Position 2 has no next token, so the shift drops it. Figure 1 draws each contribution with the mask and without it.

```figure
<svg viewBox="0 0 640 310" role="img" aria-label="Per-position loss on a 3-token example. Masking drops a prompt-target loss of 3.0949 to zero. The reply contributes 0.2395. The last position contributes nothing.">
<line x1="72.0" y1="220.0" x2="600.0" y2="220.0" class="stroke-grid"/>
<text x="64.0" y="224.0" class="fig-small" text-anchor="end">0</text>
<line x1="72.0" y1="167.4" x2="600.0" y2="167.4" class="stroke-grid"/>
<text x="64.0" y="171.4" class="fig-small" text-anchor="end">1</text>
<line x1="72.0" y1="114.9" x2="600.0" y2="114.9" class="stroke-grid"/>
<text x="64.0" y="118.9" class="fig-small" text-anchor="end">2</text>
<line x1="72.0" y1="62.3" x2="600.0" y2="62.3" class="stroke-grid"/>
<text x="64.0" y="66.3" class="fig-small" text-anchor="end">3</text>
<text x="16.0" y="128.0" class="fig-small" text-anchor="middle" transform="rotate(-90 16 128)">loss</text>
<rect x="130.0" y="57.3" width="32.0" height="162.7" class="fill-b"/>
<rect x="174.0" y="220.0" width="32.0" height="0.0" class="fill-a"/>
<text x="146.0" y="51.3" class="fig-small" text-anchor="middle">3.0949</text>
<text x="190.0" y="214.0" class="fig-small" text-anchor="middle">0.0000</text>
<text x="168.0" y="242.0" class="fig-small" text-anchor="middle">logits at 0</text>
<text x="168.0" y="258.0" class="fig-small" text-anchor="middle">prompt target</text>
<rect x="310.0" y="207.4" width="32.0" height="12.6" class="fill-b"/>
<rect x="354.0" y="207.4" width="32.0" height="12.6" class="fill-a"/>
<text x="348.0" y="199.4" class="fig-small" text-anchor="middle">0.2395</text>
<text x="348.0" y="242.0" class="fig-small" text-anchor="middle">logits at 1</text>
<text x="348.0" y="258.0" class="fig-small" text-anchor="middle">reply target</text>
<rect x="490.0" y="220.0" width="32.0" height="0.0" class="fill-b"/>
<rect x="534.0" y="220.0" width="32.0" height="0.0" class="fill-a"/>
<text x="528.0" y="212.0" class="fig-small" text-anchor="middle">0.0000</text>
<text x="528.0" y="242.0" class="fig-small" text-anchor="middle">logits at 2</text>
<text x="528.0" y="258.0" class="fig-small" text-anchor="middle">no label</text>
<rect x="150.0" y="274.0" width="14.0" height="14.0" class="fill-b"/>
<text x="170.0" y="286.0" class="fig-small" text-anchor="start">without the prompt mask</text>
<rect x="360.0" y="274.0" width="14.0" height="14.0" class="fill-a"/>
<text x="380.0" y="286.0" class="fig-small" text-anchor="start">with the prompt mask</text>
</svg>
Figure 1. Loss at each logit position of the three-token example. Notice that the mask removes the prompt target's 3.0949 and leaves the reply's 0.2395, and that the last position adds nothing because the shift gives it no label.
```

Work the kept term by hand. The logits are $[0, 0, 2]$ and the target is token 2, whose logit is 2. The normalizer is $\log(e^0 + e^0 + e^2) = \log(2 + e^2)$. With $e^2 = 7.3891$, that sum is 9.3891 and its log is 2.2395. The negative log-probability is $2.2395 - 2 = 0.2395$.

The dropped term is the one a missing mask keeps. Logits $[3, 0, 0]$ scoring token 1 give $\log(e^3 + 2) = 3.0949$. Leave it in and the loss is 3.3345, about fourteen times the masked loss, and the gradient pushes the model to reproduce the prompt.

The gradient of a kept term on its logits is the softmax minus a one-hot vector at the target. On $[0, 0, 2]$ the softmax is $(0.1065, 0.1065, 0.7870)$, so the chosen logit gets $0.7870 - 1 = -0.2130$. Logits that predict a prompt token get 0, and so do the logits at the last position.

> [!WARNING]
> Zeroing every logit that sits on a prompt token drops the first word of the reply. The mask follows the token being predicted, one step to the right of the logits.

For a batch of shape `(B, T, V)`, the sum of the kept terms is:

```python
pred, target = logits[:, :-1, :], tokens[:, 1:]
supervise = prompt_mask[:, 1:] == 0
log_z = logsumexp(pred, axis=-1)
chosen = np.take_along_axis(pred, target[..., None], axis=-1)[..., 0]
loss = np.where(supervise, log_z - chosen, 0.0).sum()
```

Read it aloud: at every position but the last, if the next token is a response token, add how surprised the model was. The exercise returns this sum, so two examples add. Divide by the number of supervised tokens before the optimizer step.

$$
\mathcal{L}(\theta) = -\sum_{t=0}^{T-2} m_{t+1} \log \pi_\theta(u_{t+1} \mid u_{\le t})
$$

The sum stops at $T - 2$ because position $T - 1$ has no label. $m_{t+1}$ is 1 when $u_{t+1}$ is a response token. $\pi_\theta$ is a softmax over the logits at $t$.

> [!KEY]
> Supervised fine-tuning is next-token loss on the reply only. Logits at $t$ score token $t + 1$, and the last prompt position is the one that scores the first response token.

## This is behavior cloning

Treat the prefix so far as the state and the next token as the action. The demonstrations are expert actions on expert prefixes, and the loss copies them. That is behavior cloning from [Imitation learning](/lessons/048-imitation-learning), one token at a time.

Training feeds the demonstration's own prefix, which is called teacher forcing. Generation feeds the model's own prefix. One wrong token makes the next state something the loss never trained on, and the following token is a guess from there. On the tightrope in that chapter, the extra cost of behavior cloning grows with the square of the horizon. A long reply is the same kind of horizon.

Cloning also stops at the demonstrations. If every stored reply is wrong, the minimum of this loss is a model that is confidently wrong.

## Pack examples without mixing them

Rows have different lengths. Padding them spends attention on pad tokens the loss then drops. *Sequence packing* concatenates examples into one row. Two things have to come with that concatenation. Position ids restart at 0, or the second example looks like a continuation of the first. The attention mask is a causal block per example: token $i$ may attend to token $j$ only when they share an example and $j \le i$. A causal mask over the whole row lets the second loss depend on the first example's tokens. Figure 2 is the block-diagonal mask for $(1, 2, 3)$ followed by $(4, 5)$.

```figure
<svg viewBox="0 0 640 330" role="img" aria-label="A 5 by 5 attention mask for two packed examples. Ones form a causal triangle inside each example. Zeros fill every cell that would look across the boundary.">
<text x="228.0" y="22.0" class="fig-small" text-anchor="middle">key</text>
<text x="132.0" y="28.0" class="fig-small fig-bold" text-anchor="middle">A0</text>
<text x="92.0" y="64.0" class="fig-small fig-bold" text-anchor="end">A0</text>
<text x="180.0" y="28.0" class="fig-small fig-bold" text-anchor="middle">A1</text>
<text x="92.0" y="112.0" class="fig-small fig-bold" text-anchor="end">A1</text>
<text x="228.0" y="28.0" class="fig-small fig-bold" text-anchor="middle">A2</text>
<text x="92.0" y="160.0" class="fig-small fig-bold" text-anchor="end">A2</text>
<text x="276.0" y="28.0" class="fig-small fig-bold" text-anchor="middle">B0</text>
<text x="92.0" y="208.0" class="fig-small fig-bold" text-anchor="end">B0</text>
<text x="324.0" y="28.0" class="fig-small fig-bold" text-anchor="middle">B1</text>
<text x="92.0" y="256.0" class="fig-small fig-bold" text-anchor="end">B1</text>
<rect x="108.0" y="36.0" width="48.0" height="48.0" class="fill-soft-a stroke-muted"/>
<text x="132.0" y="64.0" class="fig-small fig-mono" text-anchor="middle">1</text>
<rect x="156.0" y="36.0" width="48.0" height="48.0" class="fill-sunk stroke-muted"/>
<text x="180.0" y="64.0" class="fig-small fig-mono" text-anchor="middle">0</text>
<rect x="204.0" y="36.0" width="48.0" height="48.0" class="fill-sunk stroke-muted"/>
<text x="228.0" y="64.0" class="fig-small fig-mono" text-anchor="middle">0</text>
<rect x="252.0" y="36.0" width="48.0" height="48.0" class="fill-sunk stroke-muted"/>
<text x="276.0" y="64.0" class="fig-small fig-mono" text-anchor="middle">0</text>
<rect x="300.0" y="36.0" width="48.0" height="48.0" class="fill-sunk stroke-muted"/>
<text x="324.0" y="64.0" class="fig-small fig-mono" text-anchor="middle">0</text>
<rect x="108.0" y="84.0" width="48.0" height="48.0" class="fill-soft-a stroke-muted"/>
<text x="132.0" y="112.0" class="fig-small fig-mono" text-anchor="middle">1</text>
<rect x="156.0" y="84.0" width="48.0" height="48.0" class="fill-soft-a stroke-muted"/>
<text x="180.0" y="112.0" class="fig-small fig-mono" text-anchor="middle">1</text>
<rect x="204.0" y="84.0" width="48.0" height="48.0" class="fill-sunk stroke-muted"/>
<text x="228.0" y="112.0" class="fig-small fig-mono" text-anchor="middle">0</text>
<rect x="252.0" y="84.0" width="48.0" height="48.0" class="fill-sunk stroke-muted"/>
<text x="276.0" y="112.0" class="fig-small fig-mono" text-anchor="middle">0</text>
<rect x="300.0" y="84.0" width="48.0" height="48.0" class="fill-sunk stroke-muted"/>
<text x="324.0" y="112.0" class="fig-small fig-mono" text-anchor="middle">0</text>
<rect x="108.0" y="132.0" width="48.0" height="48.0" class="fill-soft-a stroke-muted"/>
<text x="132.0" y="160.0" class="fig-small fig-mono" text-anchor="middle">1</text>
<rect x="156.0" y="132.0" width="48.0" height="48.0" class="fill-soft-a stroke-muted"/>
<text x="180.0" y="160.0" class="fig-small fig-mono" text-anchor="middle">1</text>
<rect x="204.0" y="132.0" width="48.0" height="48.0" class="fill-soft-a stroke-muted"/>
<text x="228.0" y="160.0" class="fig-small fig-mono" text-anchor="middle">1</text>
<rect x="252.0" y="132.0" width="48.0" height="48.0" class="fill-sunk stroke-muted"/>
<text x="276.0" y="160.0" class="fig-small fig-mono" text-anchor="middle">0</text>
<rect x="300.0" y="132.0" width="48.0" height="48.0" class="fill-sunk stroke-muted"/>
<text x="324.0" y="160.0" class="fig-small fig-mono" text-anchor="middle">0</text>
<rect x="108.0" y="180.0" width="48.0" height="48.0" class="fill-sunk stroke-muted"/>
<text x="132.0" y="208.0" class="fig-small fig-mono" text-anchor="middle">0</text>
<rect x="156.0" y="180.0" width="48.0" height="48.0" class="fill-sunk stroke-muted"/>
<text x="180.0" y="208.0" class="fig-small fig-mono" text-anchor="middle">0</text>
<rect x="204.0" y="180.0" width="48.0" height="48.0" class="fill-sunk stroke-muted"/>
<text x="228.0" y="208.0" class="fig-small fig-mono" text-anchor="middle">0</text>
<rect x="252.0" y="180.0" width="48.0" height="48.0" class="fill-soft-b stroke-muted"/>
<text x="276.0" y="208.0" class="fig-small fig-mono" text-anchor="middle">1</text>
<rect x="300.0" y="180.0" width="48.0" height="48.0" class="fill-sunk stroke-muted"/>
<text x="324.0" y="208.0" class="fig-small fig-mono" text-anchor="middle">0</text>
<rect x="108.0" y="228.0" width="48.0" height="48.0" class="fill-sunk stroke-muted"/>
<text x="132.0" y="256.0" class="fig-small fig-mono" text-anchor="middle">0</text>
<rect x="156.0" y="228.0" width="48.0" height="48.0" class="fill-sunk stroke-muted"/>
<text x="180.0" y="256.0" class="fig-small fig-mono" text-anchor="middle">0</text>
<rect x="204.0" y="228.0" width="48.0" height="48.0" class="fill-sunk stroke-muted"/>
<text x="228.0" y="256.0" class="fig-small fig-mono" text-anchor="middle">0</text>
<rect x="252.0" y="228.0" width="48.0" height="48.0" class="fill-soft-b stroke-muted"/>
<text x="276.0" y="256.0" class="fig-small fig-mono" text-anchor="middle">1</text>
<rect x="300.0" y="228.0" width="48.0" height="48.0" class="fill-soft-b stroke-muted"/>
<text x="324.0" y="256.0" class="fig-small fig-mono" text-anchor="middle">1</text>
<text x="80.0" y="150.0" class="fig-small" text-anchor="middle" transform="rotate(-90 80 150)">query</text>
<rect x="380.0" y="80.0" width="16.0" height="16.0" class="fill-soft-a stroke-muted"/>
<text x="404.0" y="93.0" class="fig-small" text-anchor="start">example A may attend</text>
<rect x="380.0" y="112.0" width="16.0" height="16.0" class="fill-soft-b stroke-muted"/>
<text x="404.0" y="125.0" class="fig-small" text-anchor="start">example B may attend</text>
<rect x="380.0" y="144.0" width="16.0" height="16.0" class="fill-sunk stroke-muted"/>
<text x="404.0" y="157.0" class="fig-small" text-anchor="start">blocked</text>
</svg>
Figure 2. Which tokens may attend to which in one packed row. Notice the two causal triangles and the zeros across the boundary. With this mask a one-layer attention model scores the packed row at 12.5230, the sum of the separate losses 7.7072 and 4.8158. One causal mask over the whole row scores it at 9.5887.
```

The prompt mask packs by the same concatenation, and the label shift inside each block is unchanged. With that mask and restarted positions, the packed sum equals the sum of the separate losses. The exercise checks the equality to $10^{-10}$ on the model behind Figure 2.

> [!KEY]
> Packing does not change the loss when position ids restart at each example and attention is confined to that example's causal block.

## Keep the highest-scoring samples, then copy them

Demonstrations are scarce, and a model that already answers a little can write its own. *Rejection sampling* draws several replies for one prompt, scores them, and discards all but the highest-scoring. Supervised fine-tuning then clones the kept replies. The loss has no term for a discarded sample.

Take two prompts and five scored replies, and keep $k = 1$ per prompt. Rewards $(0.2, 0.9, 0.9)$ on the first prompt keep the middle reply: it ties the third at 0.9, and the smaller index wins the tie. Rewards $(0.1, 0.4)$ on the second prompt keep the last reply. The loss is the supervised loss on those two rows and zero on the other three.

That is *reward-weighted regression* with weights of 0 or 1:

$$
\sum_i w_i \, \mathrm{NLL}(y_i \mid x_i), \qquad w_i \in \{0, 1\}
$$

$w_i$ is 1 when sample $i$ is among the $k$ highest rewards for its prompt. The weighted sum equals the supervised loss on the kept rows. The same sum with the raw rewards as weights does not: a reward of 0.9 is a different number from a weight of 1.

The sampler stays frozen, a discarded reply sends no gradient, and the loss has no advantage and no KL term. Reinforcement learning, in the chapters ahead, can give a near miss a small advantage. Drawing the next round from the updated model is expert iteration, which [AI feedback and self-improvement](/lessons/064-ai-feedback-and-self-improvement) builds. One round is still cloning.

> [!KEY]
> Rejection sampling fine-tuning is supervised learning on the kept rows. It matches reward-weighted regression only when the weights are 0 or 1, not the raw rewards.

## Where the reference policy comes from

The classic recipe has three steps after the base model: supervised fine-tuning, a reward model on pairwise preferences, then reinforcement learning from the supervised checkpoint. Tülu 3 has the same shape with a different middle and end: supervised fine-tuning, direct preference optimization, then reinforcement learning against a verifier. The paper also counts a data-curation stage before those three.

The reference policy $\pi_{\text{ref}}$ is a frozen copy of the checkpoint a KL-regularized stage starts from. Supervised fine-tuning has no KL term, so it has no reference. It produces the first checkpoint later stages can freeze. In the three-step recipe the leash points at the supervised model. In the Tülu 3 runs with the highest scores they report, the verifier stage starts from the preference-tuned checkpoint, and the KL is measured from there. A base model is a poor anchor: the KL would spend its budget pulling replies back toward continuations.

The stages differ in what they train on and in which weights they freeze:

| Stage | What it trains on | Frozen reference |
|---|---|---|
| Supervised fine-tuning | Response tokens of demonstrations | None. This checkpoint is the first reference. |
| Preference | Ranked pairs, or a reward model fit to them | The supervised checkpoint |
| Reinforcement learning | Samples from the policy being updated | The checkpoint this stage starts from |

> [!KEY]
> The reference policy is a frozen copy of the checkpoint a later stage starts from. In the three-step recipe, that checkpoint is the supervised model.

> [!RECAP]
> - Logits at position $t$ score token $t + 1$. Prompt targets add nothing, and the last prompt position scores the first response token.
> - The loss returns the sum of those terms, so separate examples add.
> - Packing matches that sum when position ids restart and the attention mask is block diagonal.
> - Keeping the $k$ highest-reward replies per prompt and cloning them is reward-weighted regression with weights of 0 or 1.
> - Later chapters leash the policy to a frozen copy of the checkpoint they start from, which this stage creates.

## Build it step by step

The checks call three functions, in this order. A prompt mask uses 1 for a prompt token and 0 for a response token.

1. **`sft_loss(logits, tokens, prompt_mask)`.** `logits` is `(B, T, V)`. `tokens` and `prompt_mask` are `(B, T)`. Return the sum of the kept negative log-probabilities after the shift, in float64. Subtract the max before the exp. Score `tokens[:, 1:]` from `logits[:, :-1]`, and keep a term only where the next token's prompt mask is 0. The three-token example returns 0.2395, and two supervised tokens add. Scoring `tokens[:, t]` from `logits[:, t]`, zeroing prompt positions, or returning the mean fails.
2. **`pack_sequences(token_rows, prompt_rows)`.** Each argument is a list of 1D arrays, one example per entry, and each prompt row matches its token row in length. Return `(tokens, position_ids, attn_mask, prompt_mask)` for the concatenated row. `attn_mask[i, j]` is 1 when token `i` may attend to token `j`. The usual mistake is one lower-triangular mask over the whole row, or position ids that do not restart.
3. **`rejection_sample_filter(prompt_ids, rewards, k)`.** `prompt_ids` and `rewards` are shape `(N,)`. Return a boolean mask of shape `(N,)` that is true for the `k` highest rewards inside each prompt id. A prompt with fewer than `k` samples keeps all of them. Ties break toward the smaller index.

## Cost and scaling

Attention on a packed row of length $L$ costs the same quadratic as a padded row of that length, and every token in it is a real token. A production kernel runs each block as its own variable-length sequence and never builds the $L \times L$ mask. The pattern is the one you build here.

The sum you return grows with the number of supervised tokens, so divide by that count before the optimizer step. OLMo 2 uses a peak learning rate of $3 \times 10^{-4}$ in pretraining and $1 \times 10^{-5}$ here. Sweep it.

## Where this breaks

**The mask is off by one.** Dropping every prompt position, rather than every prompt target, deletes the gradient on the first response token. The reply then learns from its second token onward, and the first word is whoever the pretrained model would have written.

**The demonstrations are the ceiling.** Rejection sampling keeps a bad reply when every sample for that prompt is bad. A weight of 1 on a wrong answer is behavior cloning of the wrong answer. A verifier that is itself wrong has the same effect, one stage later.

**Multi-turn rows hide an extra choice.** You can put the loss on every assistant turn, or only on the last one. User turns stay masked either way. Pick one rule and keep it for the whole run. Mixing the two makes the same conversation a different target in different rows.

## Further reading

- [Lambert, *Instruction fine-tuning*](https://rlhfbook.com/c/04-instruction-tuning), in *Reinforcement Learning from Human Feedback*. Chat templates, prompt masking, and why the loss is the pretraining loss on different tokens.
- [Lambert, *Training overview*](https://rlhfbook.com/c/03-training-overview). The three-step recipe and the Tülu 3 stages that follow it.
- [CS336, Spring 2026, lecture 15, "Mid/post-training (SFT/RLHF)"](https://cs336.stanford.edu/). Supervised fine-tuning, then RLHF. Lecture 16 the same week is the verifier stage.
- [CS224R, Spring 2026, "RL for LLMs: Preference Optimization"](http://cs224r.stanford.edu/). Preference optimization, after a default project whose first milestone is a supervised run.
- [CS285, Spring 2026, lecture 14, "LLM RL"](https://rail.eecs.berkeley.edu/deeprlcourse/). The [slides](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-14.pdf) draw pretraining, supervised fine-tuning, and RL finetuning as the three blocks.
- [Lambert et al. (2024), Tülu 3](https://arxiv.org/abs/2411.15124). Supervised fine-tuning, then preference optimization, then reinforcement learning with verifiable rewards. The [training commands](https://github.com/allenai/open-instruct/blob/main/docs/tulu3.md) reproduce the supervised stage.

Open questions researchers are working on:

- When a conversation has several assistant turns, which of those turns belong in the loss?
- Does the KL reference stay the supervised checkpoint, or move to the checkpoint each later stage starts from?
- How much of the gain people attribute to later reinforcement learning is already available from the supervised mixture?

## The exercise

You build `sft_loss`, `pack_sequences`, and `rejection_sample_filter`.

The checks match the three-token loss 0.2395, the sum of two supervised tokens, and a finite-difference gradient that is zero on prompt targets and on the last position. They match the packed tokens, restarted position ids, and block-diagonal mask, and they match packed and unpacked losses to $10^{-10}$. They match the filter, including a tie, and the equality with weights of 0 or 1.
