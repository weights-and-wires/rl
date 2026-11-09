---
title: RL for vision, speech, and robots
slug: 060-rl-for-vision-speech-and-robots
part: "Part 10 — Beyond text, and what can go wrong"
summary: Turn a task metric you cannot differentiate, such as box overlap, word error rate, or task cost, into a reward, and see the policy gradient each one produces.
minutes: 45
gpu: false
objectives:
  - Explain why a metric such as IoU or word error rate can be a reward when it cannot be a loss.
  - Compute intersection over union and a detection reward, and show how extra boxes game a recall reward.
  - Compute word error rate and the minimum word error rate loss, and show that its gradient is REINFORCE over an N-best list.
  - Estimate a residual policy's gradient with reward-to-go and a leave-one-out baseline.
prereqs:
  - 031-baselines-and-actor-critic
  - 042-imitation-learning
  - 053-group-relative-policy-optimization
exercise: 060-rl-for-vision-speech-and-robots
---

# RL for vision, speech, and robots

A vision model is asked "where is the red mug?" and answers with a box. A speech recognizer is judged by the fraction of words it got wrong. A robot controller is judged by how far a pendulum drifts from upright. In all three cases you have a number that says how good the output was, and you cannot differentiate it: box overlap gives no gradient with respect to the weights, and a transcript is a list of words that cannot be nudged.

Cross-entropy, the usual loss, scores the wrong thing. So can you train on the score directly? You can. Treat the model as a policy, treat the score as a reward, and follow the policy gradient. The model never differentiates the score. It only samples outputs and is told how good they were.

This chapter shows that one move in three places. In each you build the metric, turn it into a reward, and meet one wrinkle special to that domain.

## What you already know

- A policy gradient raises the log-probability of a sampled output in proportion to its advantage, the reward minus a baseline (see [Baselines and actor-critic](/lessons/031-baselines-and-actor-critic)).
- GRPO standardizes the rewards of a group of samples from one prompt to get advantages (see [Group-relative policy optimization](/lessons/053-group-relative-policy-optimization)).
- A policy trained by imitation drifts into states its demonstrations never covered (see [Imitation learning](/lessons/042-imitation-learning)).

## The shared move: sample, score, weight

Every method here has the same loop. Sample several outputs, score each with the task metric, and subtract a baseline so that better-than-typical outputs get positive weight. Then raise each output's log-probability in proportion to its weight. The metric appears only as a number in that weight, so it can be as jagged as you like.

## Vision: score a box with IoU

A box is four numbers $(x_1, y_1, x_2, y_2)$, its left, top, right, and bottom edges. *Intersection over union* (IoU) compares two boxes by their overlap:

$$
\mathrm{IoU}(b, \hat b) = \frac{|b \cap \hat b|}{|b| + |\hat b| - |b \cap \hat b|}
$$

Read it as: the area the two boxes share, divided by the area they cover together. The denominator subtracts the shared area once because adding the two areas counts it twice. Identical boxes score 1, and boxes that do not touch score 0.

The intersection is itself a box: its left edge is the larger of the two left edges and its right edge the smaller of the two right edges. When those cross, clip the width to zero:

```python
iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
```

Take the target $b = (0, 0, 2, 2)$. A prediction $(1, 1, 3, 3)$ is a 2 by 2 square, so each has area 4. They share a unit square from $(1, 1)$ to $(2, 2)$, so $\mathrm{IoU} = 1/(4 + 4 - 1) = 1/7 = 0.143$. The prediction $(0, 0, 4, 2)$ has area 8 and contains the target, so $\mathrm{IoU} = 4/8 = 0.5$. The prediction $(3, 3, 4, 4)$ does not touch it, so $\mathrm{IoU} = 0$.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="A target box and three predicted boxes with IoU 0.143, 0.5, and 0">
  <rect x="40" y="30" width="240" height="240" class="fill-sunk"/>
  <rect x="40" y="30" width="120" height="120" class="fill-soft-a stroke-a"/>
  <rect x="100" y="90" width="120" height="120" fill="none" class="stroke-b"/>
  <rect x="40" y="30" width="240" height="120" fill="none" class="stroke-c dashed"/>
  <rect x="220" y="210" width="60" height="60" fill="none" class="stroke-d"/>
  <text x="100" y="95" text-anchor="middle" class="fig-small">target</text>
  <line x1="320" y1="60" x2="350" y2="60" class="stroke-a"/>
  <text x="360" y="64">target (0, 0, 2, 2)</text>
  <line x1="320" y1="110" x2="350" y2="110" class="stroke-b"/>
  <text x="360" y="114">(1, 1, 3, 3): IoU 1/7 = 0.143</text>
  <line x1="320" y1="160" x2="350" y2="160" class="stroke-c dashed"/>
  <text x="360" y="164">(0, 0, 4, 2): IoU 0.5</text>
  <line x1="320" y1="210" x2="350" y2="210" class="stroke-d"/>
  <text x="360" y="214">(3, 3, 4, 4): IoU 0</text>
</svg>
Figure 1. IoU gives partial credit. The half-overlapping box earns 0.5 and the offset square earns 0.143, a denser signal than a 0-or-1 reward that only fires on an exact match.
```

IoU is a rule, not a learned model, so it is a *verifiable reward*. Visual-RFT (Liu and colleagues, 2025) and VLM-R1 (Shen and colleagues, 2025) train vision-language models with GRPO on IoU rewards. 

> [!WARNING]
> A policy that samples four numbers can produce a box with $x_2 < x_1$. Clip each width and height at zero, and return 0 when the union is zero. Otherwise one bad sample puts a NaN into every gradient.

### Stop the box spray

Detection asks for every object of a class, so the policy outputs a list of boxes. A natural reward is *recall*: for each true box, take the best IoU of any predicted box, and average over the true boxes. Extra boxes never lower recall, so a policy learns to spray boxes everywhere. VLM-R1 saw this happen and scaled the reward by a length penalty:

$$
R = s \cdot \text{recall}, \qquad s = \min\Big(1, \frac{L_{\text{gt}}}{L_{\text{pred}}}\Big)
$$

Read it as: $L_{\text{gt}}$ is the number of true boxes and $L_{\text{pred}}$ the number predicted. Predicting more boxes than exist shrinks the reward in proportion. One true box and an exact prediction plus three junk boxes score 1 under recall, and $1 \times \min(1, 1/4) = 0.25$ with the penalty. In code, `overlaps = iou(pred[:, None, :], gt[None, :, :])` has shape `(K, M)`, and `overlaps.max(axis=0).mean()` is the recall.

## Speech: score a transcript with word error rate

*Word error rate* (WER) compares a *hypothesis*, the recognizer's transcript, with the *reference*, what was said. It counts the fewest word edits that turn one into the other, where an edit is a substitution, an insertion, or a deletion, and divides by the number of reference words:

$$
\text{WER} = \frac{S + D + I}{N_{\text{ref}}}
$$

Reference `the cat sat` against hypothesis `the cat sit` is one substitution, so $\text{WER} = 1/3$. Insertions can push it above 1: `a b` against `x y z w` gives $4/2 = 2$. You count the edits by dynamic programming, taking the cheapest last move for the first $i$ reference and $j$ hypothesis words:

```python
cur[j] = min(prev[j - 1] + (ref[i - 1] != hyp[j - 1]), prev[j] + 1, cur[j - 1] + 1)
```

WER is a property of a whole transcript and cannot be differentiated. A recognizer assigns a probability $P(y \mid x)$ to every transcript $y$, and what you want to lower is the *expected* number of errors, a sum over every possible sentence. You cannot compute that sum. Minimum word error rate (MWER) training (Prabhavalkar and colleagues, 2018) restricts it to the $N$ best hypotheses from beam search and renormalizes their probabilities to sum to one over the list:

$$
\hat P(y_i) = \frac{P(y_i)}{\sum_{j=1}^{N} P(y_j)}, \qquad \mathcal{L}_{\text{MWER}} = \sum_{i=1}^{N} \hat P(y_i)\big[W(y_i, y^*) - \hat W\big]
$$

Read it as: treat the beam as the whole world, and average each hypothesis's error count $W$ minus a constant $\hat W$, the list's mean error count. The renormalized probabilities are a softmax of the log-scores, so subtract the maximum first.

### The gradient is REINFORCE

Differentiate the expected errors $E = \sum_j \hat P_j W_j$ with respect to a log-score $\ell_i$. The softmax derivative is $\partial \hat P_j/\partial \ell_i = \hat P_j(\mathbb{1}[i = j] - \hat P_i)$, so

$$
\frac{\partial \mathcal{L}_{\text{MWER}}}{\partial \ell_i} = \hat P_i \big(W_i - E\big)
$$

Read it as: a hypothesis with more errors than the list's expected errors gets a positive gradient, so descent lowers its score. One with fewer errors is raised. The constant $\hat W$ has vanished. REINFORCE with baseline $b$ samples hypothesis $k$ from $\hat P$ and uses $(W_k - b)(e_k - \hat P)$, with $e_k$ the one-hot vector. Its average is $\hat P \odot (W - E)$ for any constant $b$, because $\sum_k \hat P_k(e_k - \hat P) = 0$. So MWER is REINFORCE with a baseline, where the samples are the whole beam and the expectation is exact.

Take four hypotheses with log-scores $(2.0, 1.0, 0.5, -1.0)$ and word errors $(0, 1, 2, 3)$. The renormalized probabilities are $\hat P = (0.6095, 0.2242, 0.1360, 0.0303)$, the expected errors are $E = 0.5872$, and the gradient is $(-0.3579, 0.0925, 0.1921, 0.0732)$. It sums to zero, so a step only moves probability around inside the beam.

```figure
<svg viewBox="0 0 600 260" role="img" aria-label="For four hypotheses with 0 to 3 word errors: probability 0.6095, 0.2242, 0.1360, 0.0303 and gradient minus 0.3579, 0.0925, 0.1921, 0.0732">
  <text x="130" y="28" class="fig-bold">Probability in the beam</text>
  <text x="420" y="28" text-anchor="middle" class="fig-bold">MWER gradient</text>
  <line x1="420" y1="40" x2="420" y2="230" class="stroke-muted"/>
  <text x="20" y="72">h1, 0 errors</text>
  <text x="20" y="122">h2, 1 error</text>
  <text x="20" y="172">h3, 2 errors</text>
  <text x="20" y="222">h4, 3 errors</text>
  <rect x="130" y="54" width="182.9" height="24" class="fill-a"/>
  <rect x="130" y="104" width="67.3" height="24" class="fill-a"/>
  <rect x="130" y="154" width="40.8" height="24" class="fill-a"/>
  <rect x="130" y="204" width="9.1" height="24" class="fill-a"/>
  <rect x="312.7" y="54" width="107.3" height="24" class="fill-c"/>
  <rect x="420" y="104" width="27.8" height="24" class="fill-b"/>
  <rect x="420" y="154" width="57.6" height="24" class="fill-b"/>
  <rect x="420" y="204" width="22" height="24" class="fill-b"/>
  <text x="320" y="70" class="fig-small">0.6095</text>
  <text x="205" y="120" class="fig-small">0.2242</text>
  <text x="178" y="170" class="fig-small">0.1360</text>
  <text x="147" y="220" class="fig-small">0.0303</text>
  <text x="240" y="70" class="fig-small">-0.358</text>
  <text x="455" y="120" class="fig-small">+0.093</text>
  <text x="485" y="170" class="fig-small">+0.192</text>
  <text x="450" y="220" class="fig-small">+0.073</text>
</svg>
Figure 2. The best hypothesis has the most probability and a negative gradient, so descent raises its score. The three worse ones are pushed down, and the probability factor shrinks the push on unlikely ones.
```

### Reward hacking in speech synthesis

Text-to-speech has the opposite problem: nobody can write a formula for "sounds good", so the rewards come from other models, such as a recognizer for intelligibility. F5R-TTS (Sun and colleagues, 2025) runs GRPO on such a reward and reports a 29.5% relative cut in WER. Seed-TTS (Anastassiou and colleagues, 2024) reports what happens when you optimize WER alone: the speech gets slower and clearer, and less natural, because slow careful speech is easy to transcribe.

## Robots: learn a correction, not a policy

A robot policy starts by imitating demonstrations, and it drifts into states nobody demonstrated. RL lets the robot practice, but fine-tuning a large pretrained policy can wreck what it learned in the first noisy updates. *Residual RL* (Johannink and colleagues, 2019) leaves the base policy frozen and learns a small correction on top. With $a_{\text{base}}(s)$ the frozen controller's action and $c$ a bound:

$$
a = a_{\text{base}}(s) + c \tanh(z), \qquad z \sim \mathcal{N}\big(w^\top \phi(s),\ \sigma^2\big)
$$

Read it as: the base action plus a correction of at most $c$. The raw value $z$ has mean $w^\top \phi(s)$, a weighted sum of state features, plus exploration noise $\sigma$. With $w = 0$ training starts at the base policy's behavior, and it can never stray more than $c$ from it.

The squash comes after sampling, so the log-probability you need is the Gaussian's, and its gradient with respect to $w$ is $\frac{\epsilon}{\sigma}\phi$, the noise that was added divided by $\sigma$, times the features. To weight it, use the *reward-to-go* $G_t^i = \sum_{t' \ge t} r_{t'}^i$ and subtract a *leave-one-out baseline* $b_t^i$, the mean reward-to-go of the other $N - 1$ episodes at the same step:

$$
\hat g = \frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{T-1}\big(G_t^i - b_t^i\big)\,\frac{\epsilon_t^i}{\sigma}\,\phi(s_t^i), \qquad b_t^i = \frac{1}{N-1}\sum_{j \ne i} G_t^j
$$

Episode $i$'s baseline uses only the other episodes, which are independent of it, so the estimate stays unbiased. In numpy:

```python
to_go = np.cumsum(rewards[:, ::-1], axis=1)[:, ::-1]
baseline = (to_go.sum(0, keepdims=True) - to_go) / (n - 1)
grad = np.sum((to_go - baseline)[:, :, None] * eps[:, :, None] * features / sigma, axis=(0, 1)) / n
```

Check it on a one-step task: base action $0.8$, target $1$, reward $-(a - 1)^2$, and residual $a = 0.8 + w + 0.1\epsilon$ with feature $\phi = 1$. At $w = 0$, the exact gradient of the expected reward is $-2(0.8 - 1) = 0.4$, because the noise has mean zero. The REINFORCE estimate over 100,000 samples lands within 0.02 of it.

The same idea scales to large vision-language-action models. SimpleVLA-RL (Li and colleagues, 2025) runs GRPO on a model that writes actions as tokens, with a reward of 1 for task success.

> [!RECAP]
> - A metric you cannot differentiate becomes a reward: sample, score, subtract a baseline, weight the log-probability.
> - IoU is the box overlap over the box union, and a recall reward needs a length penalty or the policy sprays boxes.
> - WER is word edits over reference length, and the MWER gradient $\hat P_i(W_i - E)$ is REINFORCE over the beam.
> - A residual policy adds $c\tanh(z)$ to a frozen action, and its gradient uses reward-to-go and a leave-one-out baseline.

## Build it step by step

The checks test these functions in order.

1. **`iou(box_a, box_b)`.** Boxes of shape `(..., 4)` that broadcast. Clip each width and height at zero. Common mistake: not returning 0 when the union is zero.
2. **`detection_reward(pred_boxes, gt_boxes, length_penalty)`.** Build the `(K, M)` overlap matrix, take the best prediction for each true box, average, and multiply by $\min(1, M/K)$ when the penalty is on.
3. **`edit_distance(ref, hyp)`** and **`wer(ref, hyp)`.** Keep one row of the table, starting each new row with `[i]`. Divide by the reference length, not the hypothesis length.
4. **`mwer_loss(logscores, errors)`** and **`mwer_grad(logscores, errors)`.** The loss subtracts `errors.mean()`. The gradient is `p_hat * (errors - expected)`, with the $\hat P$-weighted mean. A given helper `_renormalise` computes $\hat P$.
5. **`residual_action(base_action, raw, bound)`.** One line: `base_action + bound * np.tanh(raw)`.
6. **`reinforce_gradient(rewards, features, eps, sigma)`.** Reward-to-go with a reversed cumulative sum, then the leave-one-out baseline. Common mistake: using the mean over all episodes, which includes episode $i$ and biases the estimate.

## Where this breaks

**Every metric is a proxy.** The box spray and the slow speech are the same failure: the policy finds what the metric rewards, not what you meant. [Reward hacking, safety, and the frontier](/lessons/061-reward-hacking-and-safe-rl) measures how fast it happens.

**Zero-variance groups.** If every sample in a group scores 0, every advantage is 0 and nothing learns.

**Learned rewards can be fooled.** VLM-RM (Rocamonde and colleagues, 2023) rewards closeness to a goal prompt in CLIP's embedding space, which a policy can exploit.

## Further reading

- [Shen and colleagues (2025), VLM-R1](https://arxiv.org/abs/2504.07615). GRPO with IoU rewards, and the length penalty against box spray.
- [Prabhavalkar and colleagues (2018), Minimum Word Error Rate Training](https://arxiv.org/abs/1712.01818). The N-best MWER loss.
- [Johannink and colleagues (2019), Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201). Learning a correction on top of a fixed controller.
- [Sun and colleagues (2025), F5R-TTS](https://arxiv.org/abs/2504.02407). GRPO for a flow-matching speech model.

Researchers are still working on these open questions:

- How do you get a dense, hard-to-game reward for tasks like folding a towel, where there is no formula?

## The exercise

You build one metric and its gradient in each domain. Write `iou` and `detection_reward` for vision, `edit_distance`, `wer`, `mwer_loss`, and `mwer_grad` for speech, and `residual_action` and `reinforce_gradient` for robots.

The checks confirm IoU of 1, 0, and 1/7 with no NaN on degenerate boxes, and that three junk boxes score 1 without the length penalty and 0.25 with it. They check WER of $1/3$ and 2, the MWER loss of $-0.9128$, and a gradient that matches finite differences. They check that the residual never moves more than its bound, and that `reinforce_gradient` matches a hand computation and lands within 0.02 of 0.4 on the one-step task.

> [!KEY]
> You can train on a score you cannot differentiate. Sample outputs, score them, and weight each log-probability by its advantage. That answers the opening question: the box overlap, the word error rate, and the pendulum's cost all enter only as numbers in the weight, and the policy gradient does the rest.
