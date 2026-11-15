---
title: Safety training against a spec
slug: 079-safety-training-against-a-spec
part: "Part 9 — RL for language models"
summary: Trace the refusal frontier for a learned reward and a rule-based one, and see which generalizes to prompts training never showed.
minutes: 35
gpu: false
objectives:
  - Define over-refusal and under-refusal, and compute both exactly for a policy on a toy prompt distribution.
  - Show that a Bradley-Terry reward fit on examples puts positive weight on a cue that only correlates with harm.
  - Fit a rule-based reward from the spec's rules, and trace each reward's frontier by sweeping the KL coefficient.
  - Explain why the rule-based frontier holds on encoded prompts and the learned one collapses.
prereqs:
  - 066-language-models-as-policies
  - 067-reward-models
  - 078-ai-feedback-and-self-improvement
exercise: 079-safety-training-against-a-spec
---

# Safety training against a spec

A team trains a chatbot to refuse harmful requests. If they push too hard, it refuses "how do I kill a Python process?" and a student's chemistry homework. If they push too little, it explains how to do something dangerous. Both failures cost something, and training pulls them against each other: every step toward refusing more fixes one and worsens the other.

The usual way to teach refusal is with examples. People label which answers are good, a reward model learns from the labels, and reinforcement learning follows the reward model. The trouble is that harm is a matter of meaning, and a model that learns from examples learns whatever pattern predicts the labels. Suppose the harmful requests in the examples mostly sound alarming. Then "sounds alarming" becomes the rule, and a harmful request in disguise gets through.

This chapter compares two rewards on a toy world. One learns from examples alone. The other scores the written rules of a spec directly, the idea behind rule-based rewards and deliberative alignment. You trace each one's exact trade-off between the two failures, then test both on prompts the examples never showed.

## What you already know

- The reward-regularized optimum is $\pi^*(y \mid x) \propto \pi_{\text{ref}}(y \mid x)\exp(r(x,y)/\beta)$, and $\beta$ sets how far it leaves the reference, from [Language models as policies](/lessons/066-language-models-as-policies).
- A reward model fit with the Bradley-Terry loss on comparisons is a guess that a policy then exploits, from [Reward models from preferences](/lessons/067-reward-models).
- A model can grade answers against written principles, as in Constitutional AI, from [AI feedback and self-improvement](/lessons/078-ai-feedback-and-self-improvement).
- With a few prompt types and two actions, every average is an exact sum, so nothing needs sampling.

## Two ways to be wrong

Call a prompt *harmful* if the spec says to refuse it, and *benign* otherwise. The assistant either refuses or complies. Two rates measure it:

- *Over-refusal* is the probability of refusing, given a benign prompt.
- *Under-refusal* is the probability of complying, given a harmful prompt.

You want both near zero. Always refusing scores (1, 0) and always complying scores (0, 1), so closer to the origin is better. The *frontier* is the curve a reward traces on that plane as you vary how hard the policy optimizes it. This chapter measures the gap between two frontiers.

```figure
<svg viewBox="0 0 600 340" role="img" aria-label="Over-refusal against under-refusal as beta varies. The spec reward reaches the origin. The examples-only reward stops at over-refusal 0.2 and under-refusal 0.1 on training prompts, and moves to under-refusal 1 on encoded prompts.">
<line x1="80" y1="270.0" x2="540" y2="270.0" class="stroke-grid"/>
<text x="72" y="274.0" class="fig-small" text-anchor="end">0</text>
<line x1="80" y1="210.0" x2="540" y2="210.0" class="stroke-grid"/>
<text x="72" y="214.0" class="fig-small" text-anchor="end">0.25</text>
<line x1="80" y1="150.0" x2="540" y2="150.0" class="stroke-grid"/>
<text x="72" y="154.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="80" y1="90.0" x2="540" y2="90.0" class="stroke-grid"/>
<text x="72" y="94.0" class="fig-small" text-anchor="end">0.75</text>
<line x1="80" y1="30.0" x2="540" y2="30.0" class="stroke-grid"/>
<text x="72" y="34.0" class="fig-small" text-anchor="end">1</text>
<line x1="80.0" y1="270" x2="80.0" y2="275" class="stroke-muted"/>
<text x="80.0" y="290" class="fig-small" text-anchor="middle">0</text>
<line x1="172.0" y1="270" x2="172.0" y2="275" class="stroke-muted"/>
<text x="172.0" y="290" class="fig-small" text-anchor="middle">0.05</text>
<line x1="264.0" y1="270" x2="264.0" y2="275" class="stroke-muted"/>
<text x="264.0" y="290" class="fig-small" text-anchor="middle">0.1</text>
<line x1="356.0" y1="270" x2="356.0" y2="275" class="stroke-muted"/>
<text x="356.0" y="290" class="fig-small" text-anchor="middle">0.15</text>
<line x1="448.0" y1="270" x2="448.0" y2="275" class="stroke-muted"/>
<text x="448.0" y="290" class="fig-small" text-anchor="middle">0.2</text>
<line x1="540.0" y1="270" x2="540.0" y2="275" class="stroke-muted"/>
<text x="540.0" y="290" class="fig-small" text-anchor="middle">0.25</text>
<line x1="80" y1="270" x2="540" y2="270" class="stroke-muted"/>
<line x1="80" y1="270" x2="80" y2="30" class="stroke-muted"/>
<polyline points="80.0,30.0 80.0,30.0 80.0,30.0 80.0,30.0 80.0,30.0 80.0,30.0 80.0,30.0 80.0,30.0 80.0,30.0 80.0,30.0 80.0,30.0 80.0,30.0 80.0,30.0 80.0,30.0 80.1,30.0 80.2,30.0 80.4,30.0 80.7,30.1 81.4,30.2 82.4,30.3 84.0,30.5 86.3,30.8 89.5,31.2 93.8,31.8 99.2,32.5 106.0,33.4 114.1,34.5 123.6,35.7 134.3,37.1 146.1,38.6 158.8,40.3 172.4,42.0 186.5,43.9 200.9,45.8 215.5,47.7 230.0,49.6 244.4,51.4 258.4,53.3 272.0,55.0 285.1,56.7 297.5,58.4 309.3,59.9 320.5,61.4 330.9,62.7 340.7,64.0 349.8,65.2 358.2,66.3 366.1,67.3 373.3,68.3 379.9,69.1 386.0,69.9 391.6,70.6 396.7,71.3 401.4,71.9 405.7,72.5 409.6,73.0 413.2,73.5 416.4,73.9 419.4,74.3 422.1,74.6" class="stroke-bad" fill="none"/>
<polyline points="448.0,246.0 447.9,245.9 447.7,245.8 447.3,245.6 446.5,245.1 445.2,244.3 442.8,243.0 439.2,240.8 433.8,237.7 426.4,233.3 416.6,227.6 404.4,220.4 390.2,212.1 374.3,202.7 357.3,192.7 339.9,182.5 322.6,172.2 306.1,162.4 290.8,153.1 277.0,144.5 264.8,136.7 254.6,129.6 246.4,123.3 240.2,117.8 236.1,112.8 234.0,108.5 233.8,104.7 235.5,101.4 238.9,98.5 243.8,96.0 249.9,93.8 257.2,91.8 265.3,90.2 274.0,88.7 283.2,87.4 292.7,86.3 302.2,85.4 311.6,84.5 321.0,83.8 330.0,83.1 338.7,82.5 347.0,82.0 354.9,81.6 362.4,81.2 369.4,80.9 376.0,80.6 382.1,80.3 387.7,80.0 393.0,79.8 397.8,79.6 402.3,79.5 406.4,79.3 410.2,79.2 413.6,79.1 416.8,79.0 419.6,78.9 422.3,78.8 424.6,78.7 426.8,78.6 428.8,78.6" class="stroke-b dashed" fill="none"/>
<polyline points="80.0,270.0 80.0,270.0 80.0,270.0 80.0,270.0 80.0,270.0 80.0,270.0 80.0,270.0 80.0,270.0 80.0,270.0 80.0,270.0 80.0,270.0 80.0,270.0 80.0,270.0 80.0,270.0 80.0,270.0 80.0,269.9 80.1,269.8 80.2,269.6 80.4,269.2 80.8,268.4 81.4,267.0 82.5,264.9 84.1,261.6 86.5,257.1 89.8,251.0 94.2,243.4 99.8,234.4 106.7,224.2 114.9,213.3 124.5,201.9 135.3,190.5 147.2,179.4 160.0,168.9 173.6,159.2 187.7,150.3 202.2,142.2 216.8,135.0 231.3,128.6 245.7,122.9 259.7,117.9 273.2,113.4 286.2,109.5 298.6,106.0 310.4,103.0 321.4,100.2 331.8,97.9 341.5,95.7 350.6,93.9 359.0,92.2 366.7,90.7 373.9,89.4 380.5,88.2 386.5,87.2 392.1,86.2 397.2,85.4 401.8,84.6 406.1,84.0 410.0,83.4 413.5,82.8 416.7,82.4" class="stroke-c" fill="none"/>
<circle cx="246.7" cy="94.8" r="5" class="fill-b"/>
<circle cx="152.3" cy="39.4" r="5" class="fill-bad"/>
<circle cx="129.7" cy="196.2" r="5" class="fill-c"/>
<circle cx="448.0" cy="78.0" r="6" class="fill-ink"/>
<text x="438.0" y="66.0" class="fig-small" text-anchor="end">base model (0.2, 0.8)</text>
<text x="162.3" y="43.4" class="fig-small">examples-only, encoded</text>
<text x="256.7" y="110.8" class="fig-small">examples-only, training</text>
<text x="139.7" y="200.2" class="fig-small">spec reward, both</text>
<text x="310" y="322" text-anchor="middle">over-refusal: refuses a benign prompt</text>
<text x="72" y="20" class="fig-small" text-anchor="end">under-refusal: complies with harm</text>
</svg>
Figure 1. Each curve sweeps beta from 20 (near the base model) down to 0.05. The dots mark beta = 1. Notice that the spec curve runs to the origin and is the same on both prompt sets, while the examples-only curve stalls at (0.2, 0.1) and, on encoded prompts, rises above the base model's 0.8.
```

Figure 1 is the result you build toward. The green curve runs to the bottom-left corner. The dashed orange curve stops short of it. The red curve is the same orange reward on encoded prompts, and it rises above the base model's starting point.

## The toy world

Each prompt has three yes-or-no features: whether it is *harmful*, whether it *looks fishy* (it uses alarming words), and whether it is *encoded* (it hides its content, for example in a cipher). In the training data, fishy prompts are mostly harmful. The four training prompt types are:

| Harmful | Fishy | Share of prompts |
|---|---|---|
| yes | yes | 0.27 |
| yes | no | 0.03 |
| no | yes | 0.14 |
| no | no | 0.56 |

So 30% of prompts are harmful, 90% of those look fishy, and 20% of the benign ones do. No training prompt is encoded. The test set is half harmful and half benign, all encoded, and none fishy because the encoding hides the alarming words.

The *reference policy* is the base model before safety training. It refuses 20% of the time on every prompt, so its rates are over-refusal 0.2 and under-refusal 0.8.

## The obvious plan: learn the reward from examples

Annotators compare a refusal with a compliance for each prompt. They follow the spec but slip 10% of the time: on a harmful prompt they prefer the refusal with probability 0.9, and on a benign one with probability 0.1. A reward model learns from these comparisons.

A small reward model reads surface patterns, not meaning, so give it two inputs, a bias and the fishy cue $f$. Its reward for complying is 0, and its reward for refusing is $\theta_0 + \theta_1 f$. Only the *margin* matters for a choice between two actions, $m(x) = r(x,\text{refuse}) - r(x,\text{comply})$. The Bradley-Terry loss on the margin, averaged over prompt types, is

$$
\mathcal{L}(\theta) = -\sum_{x} p(x)\Big[q(x)\log\sigma\big(m_\theta(x)\big) + \big(1 - q(x)\big)\log\sigma\big(-m_\theta(x)\big)\Big]
$$

where $p(x)$ is a prompt type's share and $q(x)$ is the probability an annotator prefers refusing. Read it aloud: for each prompt type, reward the model for a high margin when annotators prefer refusal and a low one when they prefer compliance, weighted by how common the type is. With $\sigma$ the logistic function, this is logistic regression with soft labels.

Each prompt group has a known $q$, so the best fit matches each group's log-odds. Among fishy prompts, annotators prefer refusal with probability $0.9 \cdot \frac{27}{41} + 0.1 \cdot \frac{14}{41} = \frac{25.7}{41} = 0.627$. Among plain prompts, it is $0.9 \cdot \frac{3}{59} + 0.1 \cdot \frac{56}{59} = \frac{8.3}{59} = 0.141$. The fitted margins are the log-odds of those two numbers:

- Plain prompts: $\theta_0 = \ln\frac{8.3}{50.7} = -1.810$.
- Fishy prompts: $\theta_0 + \theta_1 = \ln\frac{25.7}{15.3} = 0.519$, so $\theta_1 = 2.328$.

The weight on the fishy cue is large and positive, yet an annotator looks only at whether the prompt is harmful. The cue earns its weight because it predicts harm in this data. That makes it a *spurious feature*, a pattern that correlates with the label without causing it.

```python
margin = feats_refuse - feats_comply          # (n, d)
s = sigmoid(margin @ theta)
grad = margin.T @ (p * (s - q))               # Newton's method also needs the Hessian
```

> [!KEY]
> A reward learned from examples puts weight on whatever predicts the labels. Where a harmless cue correlates with harm, the reward credits the cue, and a policy trained on it follows the cue.

Why not collect more examples? More data from the same source repeats the same correlation. Only examples that break it remove the weight, and finding those means already knowing the cue.

## The spec as the reward

A spec states the rules in words. Here it has two: do not comply with a harmful request, and do not refuse a benign one. A grader that reads the prompt, such as a language model given the spec, can say whether a rule was broken. That gives two *rule features* for an action on a prompt: $c_1$ is 1 when the assistant complies with a harmful prompt, and $c_2$ is 1 when it refuses a benign one.

The spec reward is a weighted sum, $r_{\text{spec}}(x, a) = \theta_1 c_1 + \theta_2 c_2$. Its weights come from the same annotated comparisons and the same Bradley-Terry loss, with one change: the reward sees the rules, not the surface cue. On a harmful prompt the margin is $-\theta_1$, and on a benign one it is $\theta_2$. The fit gives $\theta_1 = \theta_2 = -\ln 9 = -2.197$, so each violation costs the log-odds that an annotator is right, $\ln(0.9/0.1)$.

This is the shape of Rule Based Rewards (Mu and colleagues, 2024). Their features are probabilities that a grader model assigns to propositions about a response. A linear model combines them, with weights fit on comparisons that the rules rank, and the result adds to a standard reward model's score. The paper fits with a hinge loss, where this chapter uses Bradley-Terry. Deliberative alignment (Guan and colleagues, 2024) trains the model on the spec's text so it recalls and reasons over the relevant rules before it answers.

Because the grader reads what the prompt asks, an encoding cannot hide it.

## Sweep the leash

With a reward, the policy comes from [the KL-regularized optimum](/lessons/066-language-models-as-policies). For two actions it simplifies to the log-odds of refusing:

$$
\log\frac{\pi^*(\text{refuse} \mid x)}{\pi^*(\text{comply} \mid x)} = \log\frac{0.2}{0.8} + \frac{m(x)}{\beta}
$$

Read it aloud: start from the reference's log-odds of refusing, $-1.386$, and add the reward margin divided by $\beta$. A large $\beta$ leaves the reference alone, and a small one amplifies the margin until the choice follows its sign.

Take one encoded, harmful prompt and set $\beta = 1$. Its fishy feature is 0, so the examples-only margin is $-1.810$, the log-odds are $-1.386 - 1.810 = -3.196$, and the probability of refusing is $\sigma(-3.196) = 0.039$. The spec margin on a harmful prompt is $+2.197$, so the log-odds are $0.811$, which is odds of $9/4$ and a refusal probability of $9/13 = 0.692$. The base model refuses this prompt 20% of the time. The learned reward made the assistant less careful than the base model, and the spec reward made it more careful.

```python
z = np.log(ref_probs) + R / beta               # (n, 2)
pi = np.exp(z - z.max(axis=1, keepdims=True))
pi /= pi.sum(axis=1, keepdims=True)
```

Each rate is an average over the prompt types in its group, weighted by their share within the group:

$$
\text{over}(\pi) = \sum_{x \text{ benign}} p(x \mid \text{benign})\,\pi(\text{refuse} \mid x), \qquad \text{under}(\pi) = \sum_{x \text{ harmful}} p(x \mid \text{harmful})\,\pi(\text{comply} \mid x)
$$

For the learned reward at $\beta = 1$ on training prompts, a fishy prompt has margin $0.519$, odds $0.25 e^{0.519} = 0.420$, and refusal probability $0.296$. A plain prompt gives $0.039$. Benign prompts are 20% fishy, so over-refusal is $0.2(0.296) + 0.8(0.039) = 0.091$. Harmful prompts are 90% fishy, so under-refusal is $0.9(0.704) + 0.1(0.961) = 0.730$. The spec reward at the same $\beta$ gives over-refusal $\frac{1}{37} = 0.027$ and under-refusal $\frac{4}{13} = 0.308$, which is better on both rates.

Sweeping $\beta$ from 20 down to 0.05 traces the curves in Figure 1:

- The spec frontier runs to $(0, 0)$, because its reward ranks every prompt correctly. It is the same on training and encoded prompts, since the rules look at harm, not at the surface.
- The learned frontier stops at $(0.2, 0.1)$. Its cue flags 20% of benign prompts and misses 10% of harmful ones, and lowering $\beta$ only makes it follow the cue harder.
- On encoded prompts nothing is fishy, so the learned margin is always $-1.810$. At $\beta = 1$ its under-refusal is $0.961$, above the reference's $0.8$, and it climbs toward 1 as $\beta$ falls.

The spec frontier *weakly dominates* the learned one: for every point on the learned curve, some point on the spec curve is no worse on either rate.

> [!KEY]
> Safety training trades two errors, and the reward decides the curve. A reward that scores the spec's rules reaches the corner and stays put on prompts the examples never showed. A reward learned from examples alone traces a worse curve, and on inputs that break its cue it can fall below the untrained model.

> [!RECAP]
> - A frontier plots over-refusal against under-refusal as $\beta$ varies.
> - A Bradley-Terry reward on examples puts weight $2.33$ on a fishy cue, because the cue correlates with harm in the data.
> - A rule-based reward fit from the spec's rule features gives each violation weight $-\ln 9$ and ignores the cue.
> - For two actions, the optimal policy's log-odds of refusing are the reference's plus the margin over $\beta$.
> - The spec frontier weakly dominates the learned one on training and encoded prompts.

## Build it step by step

The checks test these functions in order. Prompt rows have columns (harmful, fishy, encoded), and the actions are `REFUSE = 0` and `COMPLY = 1`.

1. **`refuse_preference(X, noise)`.** Returns the probability annotators prefer refusing: $1 - $ `noise` on harmful prompts and `noise` on benign ones. Common mistake: returning a 0/1 label.
2. **`fit_bt(M, q, p)`.** Newton's method from $\theta = 0$. The gradient is `M.T @ (p * (s - q))` and the Hessian is `(M * (p * s * (1 - s))[:, None]).T @ M`. Common mistake: leaving out the weights `p`.
3. **`rule_features(X, action)`.** Returns an `(n, 2)` array: complying with harm in column 0, refusing the benign in column 1. Common mistake: counting a refusal of a harmful prompt as a violation.
4. **`reward_table(theta, feats_refuse, feats_comply)`.** Stacks the two actions' linear rewards into an `(n, 2)` array, for both rewards.
5. **`kl_optimal(ref_probs, R, beta)`.** A row softmax of `log(ref_probs) + R / beta`. Common mistake: skipping the row maximum, which overflows at small $\beta$.
6. **`error_rates(X, p, pi)`.** Returns `(over, under)`. Divide each sum by its group's mass, not by 1.
7. **`frontier(X, p, R, ref_probs, betas)`.** Returns the two rates for each $\beta$.
8. **`weakly_dominates(a, b)`.** For each point of `b`, checks that some point of `a` is no larger on both rates.

## Where this breaks

**The toy hands the spec reward its answer.** Its grader reads harm perfectly, and it sees through an encoding that hides the fishy words. A real grader errs, and if its errors track a surface cue, the spec reward inherits the cue. The papers, not the toy, carry the evidence for real models: Rule Based Rewards report fewer under-refusals and over-refusals, and Deliberative Alignment reports fewer over-refusals and better generalization to unseen prompts.

**The learned reward is built to fail.** The cue is spurious by construction, while a real reward model learns many cues and some carry signal. The chapter shows that correlated cues earn weight, not that every learned weight is wrong.

**Two actions force the trade.** A request that is usually benign but occasionally dangerous has no good yes-or-no answer. Safe-completions (Yuan and colleagues, 2025) judge the safety of the output instead of the user's intent, and reward the most helpful answer within the policy. The exercise does not build that third action.

**Real specs conflict.** A real spec ranks its rules, and where you stop on the frontier is a policy choice, not a mathematical one. The exact KL optimum is also not what finite RL finds, so real curves are noisier.

## Further reading

The following sources define the methods in this chapter:

- [Guan et al., "Deliberative Alignment: Reasoning Enables Safer Language Models" (2024)](https://arxiv.org/abs/2412.16339): trains a model on its safety spec and teaches it to reason over the spec before answering.
- [Mu et al., "Rule Based Rewards for Language Model Safety" (2024)](https://arxiv.org/abs/2411.01111): grader-judged propositions, a linear combination, and weights fit on ranked comparisons.
- [Bai et al., "Constitutional AI: Harmlessness from AI Feedback" (2022)](https://arxiv.org/abs/2212.08073): written principles replace human labels of harm.
- [Yuan et al., "From Hard Refusals to Safe-Completions: Toward Output-Centric Safety Training" (2025)](https://arxiv.org/abs/2508.09224): judges the safety of the output to handle dual-use requests.

Open questions that researchers are working on:

- How should a grader be checked so that its errors do not become the policy's blind spots?
- Can a model reason over a spec reliably when the prompt is built to mislead it?
- How should a spec rank rules that conflict, and who decides the ranking?

## The exercise

You build the pieces that trace the two frontiers: `refuse_preference` and `rule_features`, then `fit_bt`, `reward_table`, and `kl_optimal`, then `error_rates` and `frontier`, and last `weakly_dominates`.

The checks verify the annotator probabilities and rule features, and that `kl_optimal` returns $9/13$ on the lesson's harmful prompt. They verify `error_rates` against a hand sum, that the fishy weight is $\ln(25.7/15.3) - \ln(8.3/50.7)$ and vanishes when the cue is independent of harm, and that the spec weights are $-\ln 9$. Last, they verify the frontiers' endpoints and that only the spec frontier weakly dominates on both training and encoded prompts.
