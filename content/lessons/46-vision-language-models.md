---
title: Aligning vision-language models
slug: 46-vision-language-models
part: "Part 9 — Beyond text"
summary: Make a vision-language model look at the image with mDPO, then train it to ground objects with an IoU reward and GRPO.
minutes: 60
gpu: false
objectives:
  - Explain why preference pairs on image-text data can be satisfied by a model that ignores the image, and how mDPO's conditional term stops that.
  - Compute RLHF-V's dense response score and the three mDPO loss terms by hand.
  - Implement intersection over union for batches of boxes, including boxes with zero area.
  - Show how a recall-style detection reward is hacked by extra boxes, and repair it with VLM-R1's length penalty.
  - Train a toy grounding policy with GRPO against an IoU verifier.
prereqs:
  - 36-direct-preference-optimization
  - 37-group-relative-policy-optimization
  - 38-reasoning-with-verifiable-rewards
exercise: 46-vision-language-models
---

# Aligning vision-language models

A vision-language model (VLM) reads an image and a question and writes an
answer. Inside, an image encoder turns the picture into a few hundred vectors,
and a language model reads those vectors as if they were extra words. So the
VLM is still a language model, and every method from Part 8 applies to it
without change: DPO, GRPO, and verifiable rewards all work on its tokens.

Two things do change. The first is that the model can ignore the picture. A
language model trained on captions knows that kitchens tend to hold forks, so
it can describe a fork that isn't there. This is called *hallucination*, and
ordinary preference data can teach it by accident. The second is that many
visual answers can be checked by a rule. When the question is "where is the
dog?", the answer is a box, and you can score a box against the true one with
a number between 0 and 1.

This lesson covers both. You build two preference objectives that target
hallucination, then an overlap reward for boxes, the hack that breaks it, the
fix, and a small GRPO loop that learns to find a rectangle in an image.

## What you already know

- DPO trains a policy on chosen and rejected pairs with the loss
  $-\log \sigma(\beta \Delta)$, where $\Delta$ is the chosen log-ratio minus the
  rejected one, from Direct preference optimisation.
- GRPO samples a group of answers per prompt and uses each answer's
  standardised reward as its advantage, from Group-relative policy
  optimisation.
- A verifiable reward is computed by a rule, not a learned model, from
  Reasoning with verifiable rewards.
- A Gaussian policy's log-probability has a closed form, and its gradient with
  respect to the mean is $(a - \mu) / \sigma^2$.
- A policy that maximises a flawed reward finds the flaw. That's reward
  hacking.

## Why a VLM hallucinates

Think of a student who answers questions about a photo without looking at
it. A guess from context is usually right, so the habit is rarely punished.
That sense of what usually goes together is the *language prior*.

A VLM's *policy* $\pi_\theta(y \mid m, q)$ gives the probability of an answer
$y$ given an image $m$ and a question $q$. Its language model has seen far
more text than images, so the text alone often predicts the answer.
Hallucination is the case where that guess is wrong. Preference learning
attacks it in two ways: make the labels say which words were wrong (RLHF-V),
or make the loss reward only models that use the image (mDPO).

## Put the weight on the corrected words

Collecting a preference label is expensive, and a label that says "answer A is
better" doesn't say *where* B went wrong. RLHF-V (Yu et al., 2023) asks
annotators to rewrite only the hallucinated segments of a response. The
chosen and rejected answers then differ in a few words, and the method counts
those words more.

Write $y_u$ for the tokens left unchanged and $y_c$ for the corrected tokens.
RLHF-V scores a response by a weighted mean of its token log-probabilities:

$$
\log \pi(y \mid x) = \frac{1}{N}\Big[\sum_{y_i \in y_u} \log p(y_i \mid x, y_{<i}) + \gamma \sum_{y_i \in y_c} \log p(y_i \mid x, y_{<i})\Big], \qquad N = |y_u| + \gamma |y_c|
$$

Read it as: add up the log-probabilities of the unchanged tokens, add $\gamma$
times those of the corrected tokens, and divide by the total weight $N$. The
paper uses $\gamma = 5$, so each corrected token counts five times. The score
then goes into the ordinary DPO loss in place of the plain sequence
log-probability. Here $x$ stands for the image and the question together.

In numpy, with a boolean mask `corrected`:

```python
w = np.where(corrected, gamma, 1.0)
score = (w * token_logps).sum(-1) / w.sum(-1)
```

### Worked example

A response has 20 unchanged tokens whose log-probabilities sum to $-6.0$, and
4 corrected tokens that sum to $-3.0$. With $\gamma = 5$:

- The total weight is $N = 20 + 5 \times 4 = 40$.
- The weighted sum is $-6.0 + 5 \times (-3.0) = -21.0$.
- The score is $-21.0 / 40 = -0.525$.

With $\gamma = 1$ it's the plain mean, $-9.0 / 24 = -0.375$. The four
corrected tokens pull the score much further when they carry the weight.

## Make the image matter with mDPO

Here's the trap in ordinary DPO on image data. Both answers in a pair share
the image, and the loss only asks the model to rank them. A model can often do
that from the text alone, so the loss can fall while the model learns nothing
about pictures.

mDPO (Wang et al., 2024) adds a second comparison where the *answer* stays
fixed and the *image* changes. Write the log-ratio of a response under the
policy and the frozen reference $\pi_{\text{ref}}$ as

$$
h(y, m) = \log \frac{\pi_\theta(y \mid m, q)}{\pi_{\text{ref}}(y \mid m, q)}
$$

Read it as: how much more likely the trained policy makes answer $y$ on image
$m$ than the reference does. The implicit reward of DPO is $\beta h$. Let
$m_w$ be the real image and $m_l$ a corrupted copy. mDPO builds $m_l$ by
randomly cropping less than 20 percent of the original, which erases most of
the evidence. The loss has three terms:

$$
\begin{aligned}
\mathcal{L}_{\text{DPO}} &= -\log \sigma\big(\beta h(y_w, m_w) - \beta h(y_l, m_w)\big) \\
\mathcal{L}_{\text{CoPO}} &= -\log \sigma\big(\beta h(y_w, m_w) - \beta h(y_w, m_l)\big) \\
\mathcal{L}_{\text{AncPO}} &= -\log \sigma\big(\beta h(y_w, m_w) - \delta\big)
\end{aligned}
$$

Read them one at a time:

- The first is ordinary DPO: the chosen answer $y_w$ beats the rejected one
  $y_l$ on the real image.
- The second, the *conditional* term, says the chosen answer must be more
  likely on the real image than on the cropped one. A model that ignores the
  image gives both the same log-ratio, so it can't lower this term.
- The third, the *anchor*, keeps the chosen answer's implicit reward above a
  constant $\delta$. DPO can widen a gap by lowering both likelihoods, and the
  anchor forbids the chosen one from falling.

The total is $\mathcal{L}_{\text{mDPO}} = \mathcal{L}_{\text{DPO}} +
\mathcal{L}_{\text{CoPO}} + \mathcal{L}_{\text{AncPO}}$. In numpy, where
`np.logaddexp(0, -z)` computes $-\log \sigma(z)$ without overflow:

```python
nls = lambda z: np.logaddexp(0.0, -z)
dpo = nls(beta * chosen - beta * rejected).mean()
copo = nls(beta * chosen - beta * chosen_corrupt).mean()
anchor = nls(beta * chosen - delta).mean()
```

### Worked example

Take $\beta = 1$ and $\delta = 0$. The chosen answer's log-ratio is $+0.3$ on
the real image and also $+0.3$ on the cropped one. The rejected answer's is
$-0.2$.

- DPO term: $-\log \sigma(0.3 + 0.2) = -\log \sigma(0.5) = 0.4741$.
- Conditional term: $-\log \sigma(0.3 - 0.3) = -\log \sigma(0) = \log 2 = 0.6931$.
- Anchor term: $-\log \sigma(0.3) = 0.5544$.

The conditional term sits at $\log 2$, the value of a coin flip. Its gradient
with respect to the chosen log-ratio on the real image is $\beta(\sigma(0) -
1) = -0.5$, and on the cropped image it's $+0.5$. Gradient descent raises one
and lowers the other, which only the image can tell apart.

> [!KEY]
> Ordinary DPO on image data can be satisfied by the text alone. mDPO's
> conditional term compares one answer on the real and a corrupted image, and
> it stays at $\log 2$ for any model that ignores the picture.

Other methods change the data instead. LLaVA-RLHF (Sun et al., 2023) shows its
reward model the true caption. RLAIF-V (Yu et al., 2024) has an open-source VLM
score each claim in a response. POVID (Zhou et al., 2024) makes rejected
answers by asking GPT-4V to insert hallucinations.

## Score a box with intersection over union

Some visual questions have answers a rule can check. The cleanest is
*grounding*: given "the red mug", output the box around it. A box is four
numbers $b = (x_1, y_1, x_2, y_2)$, its left, top, right, and bottom edges.

*Intersection over union* (IoU) compares two boxes by their overlap:

$$
\mathrm{IoU}(b, \hat b) = \frac{|b \cap \hat b|}{|b| + |\hat b| - |b \cap \hat b|}
$$

Read it as: the area the two boxes share, divided by the area they cover
together. Identical boxes score 1, and boxes that don't touch score 0. The
denominator subtracts the shared area once because adding the two areas counts
it twice.

The intersection is itself a box. Its left edge is the larger of the two left
edges, and its right edge is the smaller of the two right edges. When those
cross, the width is negative, and you clip it to zero:

```python
iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
```

### Worked example

Take $b = (0, 0, 2, 2)$ and $\hat b = (1, 1, 3, 3)$. Each is a 2 by 2 square,
so each has area 4. The intersection runs from $x = 1$ to $x = 2$ and $y = 1$
to $y = 2$, a unit square. So

$$
\mathrm{IoU} = \frac{1}{4 + 4 - 1} = \frac{1}{7} = 0.1429
$$

Against the same target, the box $(0, 0, 4, 2)$ scores $4 / 8 = 0.5$, and
$(3, 3, 4, 4)$ scores 0.

> [!WARNING]
> A policy that samples four numbers can produce a box with $x_2 < x_1$. Its
> naive "area" is negative, and a negative area can make the union zero or
> negative. Clip each width and height at zero, and return 0 when the union is
> zero. Otherwise one bad sample puts a NaN into every gradient.

IoU gives partial credit, a denser signal than the 0-or-1 correctness of a
maths answer. Visual-RFT (Liu et al., 2025) builds its detection reward from
the mean IoU, a confidence term, and a format term:

$$
R_d = \frac{1}{n}\sum_{i=1}^{n} \mathrm{iou}_i + \frac{1}{n}\sum_{i=1}^{n} r_{c,i} + R_{\text{format}}, \qquad r_{c,i} = \begin{cases} c_i & \mathrm{iou}_i \ne 0 \\ 1 - c_i & \mathrm{iou}_i = 0 \end{cases}
$$

Read it as: each predicted box is matched to a true box, and a match below an
IoU threshold counts as $\mathrm{iou}_i = 0$. The model states a confidence
$c_i$ per box, and earns $c_i$ for a matched box and $1 - c_i$ for an
unmatched one. The format term checks the answer's tags.

## Stop the box spray

Detection asks for every object of a class, so the policy outputs a list of
boxes. A natural reward is *recall*: for each true box, take the best IoU of
any predicted box, and average over true boxes. Extra boxes never lower it, so
a policy learns to spray boxes everywhere.

VLM-R1 (Shen et al., 2025) saw this happen with a mean average precision
reward for open-vocabulary detection. Its fix scales the reward by a length
penalty:

$$
R = s \cdot \text{score}, \qquad s = \min\Big(1, \frac{L_{\text{gt}}}{L_{\text{pred}}}\Big)
$$

Read it as: $L_{\text{gt}}$ is the number of true boxes and $L_{\text{pred}}$
the number predicted. Predicting more boxes than exist shrinks the reward in
proportion. Two true boxes and five predictions give $s = 0.4$. The exercise
uses the recall score as a stand-in for mean average precision, because it
shows the same hack with less code.

For one true box, the exact box plus three junk boxes scores 1 under recall
and $1 \times \min(1, 1/4) = 0.25$ under the penalty.

## Train a grounding policy with GRPO

Now put the pieces into one loop of reinforcement learning with a verifiable
reward, where the verifier is IoU. In the exercise, the "image" is a 16 by 16
array with one bright rectangle on a noisy background. A frozen encoder reads
the brightest pixel in every row and column, and a small network maps those 32
numbers to a box.

The *action* is a box, so the policy is a Gaussian over four numbers. Its mean
$\mu_\theta(m)$ comes from the network, and it has one learned standard
deviation $\sigma_j$ per coordinate. The log-probability of a sampled box $a$
is

$$
\log \pi_\theta(a \mid m) = \sum_{j=1}^{4} \Big[-\frac{(a_j - \mu_j)^2}{2\sigma_j^2} - \log \sigma_j - \tfrac12 \log 2\pi\Big]
$$

Read it as: each coordinate adds a squared distance from its mean, scaled by
its variance, and a term that penalises width. The four coordinates are
independent, so their log-densities add.

Each update draws a batch of images, samples a group of $G$ boxes per image,
scores each with IoU, and standardises the rewards within each group. For
rewards $R_1, \dots, R_G$ of one image, the advantage of box $i$ is

$$
A_i = \frac{R_i - \operatorname{mean}(R)}{\operatorname{std}(R) + \epsilon}
$$

Read it as: how much better box $i$ did than its siblings, in units of the
group's spread. The small $\epsilon$ keeps the division safe. The loss is
$-\frac{1}{BG}\sum A_i \log \pi_\theta(a_i \mid m)$, averaged over all $B$
images and $G$ boxes:

```python
adv = torch.tensor(group_advantages(rewards), dtype=torch.float32)
logp = gaussian_logprob(boxes, mean[:, None, :], log_std[:, None, :])
loss = -(adv * logp).mean()
```

### Worked example

The true box is $(0, 0, 2, 2)$. A group of four samples is $(0, 0, 2, 2)$,
$(0, 0, 4, 2)$, $(1, 1, 3, 3)$, and $(3, 3, 4, 4)$, with rewards $1$, $0.5$,
$0.1429$, and $0$.

- The mean reward is $0.4107$ and the standard deviation is $0.3859$.
- The advantages are $1.5271$, $0.2314$, $-0.6941$, and $-1.0643$.

The exact box is pushed up hardest. The box that half-overlaps still gets a
small push up, which is the partial credit a 0-or-1 reward would lose.

One detail of the toy matters. An untrained network outputs means near 0.5
for every coordinate, so its mean box has almost zero width and scores 0. If
every sample in a group scores 0, every advantage is 0 and nothing learns.
Learning starts from the samples whose noise happens to open a real box. This
is the same zero-variance group problem that DAPO filters out for language
models.

> [!KEY]
> Grounding turns a VLM into a policy with a continuous, rule-checked reward.
> IoU gives partial credit, GRPO needs only a group of samples per image, and
> the hard part is keeping the reward from being gamed.

> [!RECAP]
> - IoU clips each width and height at zero, and returns 0 when the union is
>   zero.
> - RLHF-V's score is a weighted mean of token log-probabilities, with weight
>   $\gamma$ on corrected tokens.
> - mDPO adds a conditional term that fixes the answer and varies the image,
>   and an anchor that keeps the chosen reward above $\delta$.
> - Recall rewards reward box spraying, and $\min(1, L_{\text{gt}} /
>   L_{\text{pred}})$ makes it cost.
> - GRPO for grounding samples a group of boxes per image and weights each
>   box's log-probability by its standardised IoU.

## Build it step by step

The checks test the functions in this order.

1. **`iou(box_a, box_b)`.** Takes two arrays of shape `(..., 4)` that
   broadcast, and returns their IoU in the broadcast shape. Use `np.maximum`
   and `np.minimum` for the intersection edges, and `np.clip(..., 0, None)` on
   every width and height, including each box's own. Return 0 where the union
   is zero, using `np.where` with a safe denominator. The common mistake is
   clipping the intersection but not the boxes' own sizes, so a swapped box
   gets a negative area.
2. **`dense_logprob(token_logps, corrected, gamma)`.** Takes arrays of shape
   `(..., T)` and returns shape `(...)`. Build weights with `np.where`, then
   divide the weighted sum by the sum of weights. The common mistake is
   dividing by `T`.
3. **`mdpo_loss(chosen, rejected, chosen_corrupt, beta, delta)`.** Takes three
   `(n,)` arrays of log-ratios and returns three floats. Use
   `np.logaddexp(0, -z)` for $-\log \sigma(z)$. The common mistake is putting
   the rejected answer into the conditional term: that term uses the chosen
   answer twice, on two images.
4. **`detection_reward(pred_boxes, gt_boxes, length_penalty)`.** Takes `(K,
   4)` and `(M, 4)` arrays. Compute the `(K, M)` IoU matrix with
   `iou(pred[:, None], gt[None])`, take `.max(axis=0)`, average, and multiply
   by `min(1, M / K)` when asked. Return 0 when `K` is 0. The common mistake is
   taking the max over the wrong axis, which rewards the best target for each
   prediction instead.
5. **`group_advantages(rewards, eps)`.** Subtract the mean and divide by the
   standard deviation plus `eps`, along the last axis with `keepdims=True`. A
   group of equal rewards must give exact zeros.
6. **`gaussian_logprob(actions, mean, log_std)`.** A torch function that sums
   the per-coordinate log-density over the last axis. Keep it differentiable:
   use `torch.exp(log_std)`, not a numpy conversion.
7. **`train_grounding(steps, batch, group, lr, seed)`.** Seed both
   generators, build a `BoxPolicy` and an Adam optimiser, and run the loop from
   the previous section. Sample boxes inside `torch.no_grad()`, compute the IoU
   rewards with your `iou`, and compute the log-probability of the fixed
   samples with gradients on. The common mistake is sampling with gradients
   on and then differentiating through the sample, which is a different
   estimator from the one the lesson derives.

## Cost and scaling

A real VLM's cost is dominated by the image. The following table compares the
work in each step for a model that turns an image into $V$ visual tokens and
writes $T$ answer tokens:

| Step | Work per example | Grows with |
|---|---|---|
| mDPO | Forward passes for three answer-image pairs on the policy and the reference | $V + T$, six times |
| RLHF-V weighting | One mask multiply on the log-probabilities | $T$, negligible |
| GRPO rollout | $G$ sampled answers per image, one image encoding shared | $G \times T$ decoding steps |
| IoU reward | Four comparisons per box pair | $K \times M$ for $K$ predictions and $M$ targets |

GRPO shares one image encoding across the group, so its extra cost is in
decoding. mDPO's conditional term costs one more image per example. In the
exercise, 300 updates of a 2,400-parameter network take under a second.

## Where this breaks

**The judge shares the model's blind spots.** POVID and RLAIF-V build their
negatives or labels with a model. Scoring the result with a judge from the
same family measures agreement with that family, not truth. Use a different
evaluator from the one that made the data.

**Shorter looks safer.** Longer answers mention more objects, so they
contain more hallucinations. Report hallucination per object mentioned, or a
preference for short answers looks like better grounding.

**The anchor is set too high.** A large $\delta$ forces the chosen answer's
likelihood up even when it's already high, which costs fluency. Start with
$\delta = 0$.

**Box conventions differ.** Some models write $(x, y, w, h)$, some write
corners, and some use pixels while others use a 0-to-1000 grid. A reward that
parses the wrong convention scores every box near 0, and the policy learns
nothing or learns to game the parser. Test the parser on known boxes first.

**Format rewards drown accuracy.** When the format term is as large as the
IoU term, the policy learns the tags and stops improving its boxes. Keep
format a small bonus.

**In-domain gains mislead.** VLM-R1's case for RL over supervised fine-tuning
is out of domain: on LISA-Grounding its 3B RL model scores 63.16 against
54.82, while in domain the two are close. Evaluate out of domain.

## Further reading

- [RLHF-V (Yu et al., 2023)](https://arxiv.org/abs/2312.00849) collects
  corrections of hallucinated segments and weights the corrected tokens in DPO.
- [mDPO (Wang et al., 2024)](https://arxiv.org/abs/2406.11839) shows that
  multimodal DPO can ignore the image and adds the conditional and anchor terms.
- [LLaVA-RLHF (Sun et al., 2023)](https://arxiv.org/abs/2309.14525) trains a
  reward model that sees the true caption, and introduces MMHal-Bench.
- [RLAIF-V (Yu et al., 2024)](https://arxiv.org/abs/2405.17220) scores
  responses claim by claim with an open-source VLM and trains with iterative
  DPO.
- [Visual-RFT (Liu et al., 2025)](https://arxiv.org/abs/2503.01785) defines IoU
  and confidence rewards for detection and trains a VLM with GRPO.
- [VLM-R1 (Shen et al., 2025)](https://arxiv.org/abs/2504.07615) documents the
  box-spraying hack, the length penalty, and RL's out-of-domain advantage.
- [Perception-R1 (Yu et al., 2025)](https://arxiv.org/abs/2504.07954) studies
  when explicit reasoning helps perception tasks and when it doesn't.

Open questions that researchers are working on:

- How do you reward a model for perceiving correctly, and not only for a
  correct final answer that language reasoning reached without looking?
- Can a VLM learn to ask for a closer look, by calling a crop or zoom tool
  mid-answer, from outcome rewards alone?
- Which multimodal reward models can detect hallucination reliably enough to
  train against?

## The exercise

You write `iou`, `dense_logprob`, `mdpo_loss`, `detection_reward`,
`group_advantages`, `gaussian_logprob`, and `train_grounding`. The last runs
GRPO on the toy world at the top of the file: `make_scenes`, `box_features`,
and `BoxPolicy`.

The checks verify IoU on hand cases including $1/7$, on broadcast batches, and
on degenerate boxes that must score 0 without NaN. They check the RLHF-V score
of $-0.525$, the mDPO terms including $\log 2$ for an image-blind policy, and
the detection reward's hack and its repair to $0.25$. They check group
advantages and the Gaussian log-probability and its gradient against torch.
The last check trains for 300 updates and expects held-out mean IoU to rise
from below 0.1 to above 0.6.
