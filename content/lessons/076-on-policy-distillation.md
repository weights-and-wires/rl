---
title: On-policy distillation
slug: 076-on-policy-distillation
part: "Part 9 — RL for language models"
summary: Train a small student on tokens it samples itself, graded by a teacher, and see why this is RL with a dense per-token reward and a reverse KL.
minutes: 35
gpu: false
objectives:
  - Write the forward and reverse KL between a teacher and a student at every token, averaged over real tokens only.
  - Explain with a bimodal teacher why forward KL covers modes and reverse KL picks one.
  - Argue from exposure bias why a student must train on prefixes it samples, and link that to DAgger.
  - Show that on-policy distillation is policy gradient with the per-token reward log pi_T minus log pi_S, and check its gradient against the exact KL gradient.
prereqs:
  - 031-the-policy-gradient
  - 051-imitation-learning
  - 065-language-models-as-policies
  - 068-rlhf-with-ppo
exercise: 076-on-policy-distillation
---

# On-policy distillation

A large model solves your problem well, but it costs too much to serve. A small model is cheap, but it is worse. The usual fix is *distillation*: train the small model, the *student*, to behave like the large one, the *teacher*. The simplest version has the teacher write answers and the student fine-tune on them, which is ordinary supervised fine-tuning on the teacher's text.

That recipe has a quiet flaw. During training the student always reads text the teacher wrote. At test time it reads its own text, and its own early mistakes put it in places the teacher never showed it. This chapter fixes the flaw by changing who writes the text. The student writes, and the teacher grades every token. The fix turns out to be reinforcement learning with a reward you can compute at every token, and it reuses ideas from three chapters you have already read.

## What you already know

- The REINFORCE gradient is $\mathbb{E}_{y \sim \pi_\theta}[r(y)\nabla_\theta \log \pi_\theta(y)]$, from [The policy gradient and REINFORCE](/lessons/031-the-policy-gradient).
- Behavior cloning on expert states compounds its errors as the learner drifts away from them, and DAgger fixes this by labeling states the learner visits, from [Imitation learning](/lessons/051-imitation-learning).
- A language model is a policy that picks one token at a time, and a response $y$ has log-probability equal to the sum of its token log-probabilities, from [Language models as policies](/lessons/065-language-models-as-policies).
- RLHF maximizes reward minus a KL penalty against a reference policy $\pi_{\text{ref}}$, from [RLHF with PPO](/lessons/068-rlhf-with-ppo).
- $\mathrm{KL}(p \,\|\, q) = \sum_v p(v)\log\frac{p(v)}{q(v)}$ is not symmetric in $p$ and $q$.

## Offline distillation, the starting point

Two older recipes set the baseline. Hinton, Vinyals, and Dean (2015) train a student to match the teacher's whole next-token distribution, not only its top choice. The ML course covers that logit matching, so this chapter leaves it there. Kim and Rush (2016) go one step simpler for sequences: the teacher generates complete responses, and the student fine-tunes on them. The DeepSeek-R1 report (2025) used this recipe at scale. It fine-tuned smaller open models on roughly 800 thousand samples that R1 and related pipelines produced.

Both recipes are *offline*. The text comes from the teacher, so the student never practices on the prefixes it will produce itself. Call that gap *exposure bias*: the student is exposed to teacher prefixes in training and to its own prefixes in use.

## Two directions of KL

Fix a prefix, and let $\pi_T$ and $\pi_S$ be the teacher's and student's next-token distributions. There are two ways to measure their distance, and they ask different questions:

$$
\mathrm{KL}(\pi_T \,\|\, \pi_S) = \sum_v \pi_T(v)\log\frac{\pi_T(v)}{\pi_S(v)}, \qquad
\mathrm{KL}(\pi_S \,\|\, \pi_T) = \sum_v \pi_S(v)\log\frac{\pi_S(v)}{\pi_T(v)}
$$

Read the first aloud: average, under the teacher, how surprised the student is relative to the teacher. It is the *forward* KL, and it is the cross-entropy loss of supervised training up to a constant. Every token the teacher might say must get probability from the student, or the loss blows up. The second is the *reverse* KL. It averages under the student, so it punishes the student for saying anything the teacher considers unlikely, and it ignores what the student never says.

Take one token. The teacher gives $[0.9, 0.1]$ and the student gives $[0.6, 0.4]$. The forward KL is $0.9\ln\frac{0.9}{0.6} + 0.1\ln\frac{0.1}{0.4} = 0.3649 - 0.1386 = 0.2263$. The reverse KL is $0.6\ln\frac{0.6}{0.9} + 0.4\ln\frac{0.4}{0.1} = -0.2433 + 0.5545 = 0.3112$. The numbers differ because the weights differ, and the check `test_hand_example` asks for exactly these two.

A language model produces many tokens, with padding in a batch, so the loss is the mean of the per-token value over real tokens only:

```python
fwd = (pT * (logpT - logpS)).sum(-1)          # (B, T): one value per token
loss_fwd = (fwd * mask).sum() / mask.sum()    # not / (B * T)
```

## A picture: a bimodal teacher

Suppose the teacher is torn between two answers, with half its probability on each, and the student is too simple to hold both. The following figure shows the teacher over a grid of 73 positions, and the best single-mode student under each KL, found by a grid search over the student's center and width.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="A bimodal teacher with two modes at -3 and 3, the reverse-KL student sitting on one mode, and the forward-KL student spread across both">
<line x1="60.0" y1="230.0" x2="580.0" y2="230.0" class="stroke-muted"/>
<line x1="60.0" y1="230.0" x2="60.0" y2="235.0" class="stroke-muted"/>
<text x="60.0" y="250.0" class="fig-small" text-anchor="middle">-9</text>
<line x1="146.7" y1="230.0" x2="146.7" y2="235.0" class="stroke-muted"/>
<text x="146.7" y="250.0" class="fig-small" text-anchor="middle">-6</text>
<line x1="233.3" y1="230.0" x2="233.3" y2="235.0" class="stroke-muted"/>
<text x="233.3" y="250.0" class="fig-small" text-anchor="middle">-3</text>
<line x1="320.0" y1="230.0" x2="320.0" y2="235.0" class="stroke-muted"/>
<text x="320.0" y="250.0" class="fig-small" text-anchor="middle">0</text>
<line x1="406.7" y1="230.0" x2="406.7" y2="235.0" class="stroke-muted"/>
<text x="406.7" y="250.0" class="fig-small" text-anchor="middle">3</text>
<line x1="493.3" y1="230.0" x2="493.3" y2="235.0" class="stroke-muted"/>
<text x="493.3" y="250.0" class="fig-small" text-anchor="middle">6</text>
<line x1="580.0" y1="230.0" x2="580.0" y2="235.0" class="stroke-muted"/>
<text x="580.0" y="250.0" class="fig-small" text-anchor="middle">9</text>
<rect x="57.1" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="64.3" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="71.6" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="78.8" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="86.0" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="93.2" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="100.4" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="107.7" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="114.9" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="122.1" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="129.3" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="136.6" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="143.8" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="151.0" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="158.2" y="229.8" width="5.8" height="0.2" class="fill-soft-a"/>
<rect x="165.4" y="229.5" width="5.8" height="0.5" class="fill-soft-a"/>
<rect x="172.7" y="228.5" width="5.8" height="1.5" class="fill-soft-a"/>
<rect x="179.9" y="226.0" width="5.8" height="4.0" class="fill-soft-a"/>
<rect x="187.1" y="220.8" width="5.8" height="9.2" class="fill-soft-a"/>
<rect x="194.3" y="211.5" width="5.8" height="18.5" class="fill-soft-a"/>
<rect x="201.6" y="197.2" width="5.8" height="32.8" class="fill-soft-a"/>
<rect x="208.8" y="178.8" width="5.8" height="51.2" class="fill-soft-a"/>
<rect x="216.0" y="159.6" width="5.8" height="70.4" class="fill-soft-a"/>
<rect x="223.2" y="144.7" width="5.8" height="85.3" class="fill-soft-a"/>
<rect x="230.4" y="139.1" width="5.8" height="90.9" class="fill-soft-a"/>
<rect x="237.7" y="144.7" width="5.8" height="85.3" class="fill-soft-a"/>
<rect x="244.9" y="159.6" width="5.8" height="70.4" class="fill-soft-a"/>
<rect x="252.1" y="178.8" width="5.8" height="51.2" class="fill-soft-a"/>
<rect x="259.3" y="197.2" width="5.8" height="32.8" class="fill-soft-a"/>
<rect x="266.6" y="211.5" width="5.8" height="18.5" class="fill-soft-a"/>
<rect x="273.8" y="220.8" width="5.8" height="9.2" class="fill-soft-a"/>
<rect x="281.0" y="226.0" width="5.8" height="4.0" class="fill-soft-a"/>
<rect x="288.2" y="228.5" width="5.8" height="1.5" class="fill-soft-a"/>
<rect x="295.4" y="229.5" width="5.8" height="0.5" class="fill-soft-a"/>
<rect x="302.7" y="229.8" width="5.8" height="0.2" class="fill-soft-a"/>
<rect x="309.9" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="317.1" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="324.3" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="331.6" y="229.8" width="5.8" height="0.2" class="fill-soft-a"/>
<rect x="338.8" y="229.5" width="5.8" height="0.5" class="fill-soft-a"/>
<rect x="346.0" y="228.5" width="5.8" height="1.5" class="fill-soft-a"/>
<rect x="353.2" y="226.0" width="5.8" height="4.0" class="fill-soft-a"/>
<rect x="360.4" y="220.8" width="5.8" height="9.2" class="fill-soft-a"/>
<rect x="367.7" y="211.5" width="5.8" height="18.5" class="fill-soft-a"/>
<rect x="374.9" y="197.2" width="5.8" height="32.8" class="fill-soft-a"/>
<rect x="382.1" y="178.8" width="5.8" height="51.2" class="fill-soft-a"/>
<rect x="389.3" y="159.6" width="5.8" height="70.4" class="fill-soft-a"/>
<rect x="396.6" y="144.7" width="5.8" height="85.3" class="fill-soft-a"/>
<rect x="403.8" y="139.1" width="5.8" height="90.9" class="fill-soft-a"/>
<rect x="411.0" y="144.7" width="5.8" height="85.3" class="fill-soft-a"/>
<rect x="418.2" y="159.6" width="5.8" height="70.4" class="fill-soft-a"/>
<rect x="425.4" y="178.8" width="5.8" height="51.2" class="fill-soft-a"/>
<rect x="432.7" y="197.2" width="5.8" height="32.8" class="fill-soft-a"/>
<rect x="439.9" y="211.5" width="5.8" height="18.5" class="fill-soft-a"/>
<rect x="447.1" y="220.8" width="5.8" height="9.2" class="fill-soft-a"/>
<rect x="454.3" y="226.0" width="5.8" height="4.0" class="fill-soft-a"/>
<rect x="461.6" y="228.5" width="5.8" height="1.5" class="fill-soft-a"/>
<rect x="468.8" y="229.5" width="5.8" height="0.5" class="fill-soft-a"/>
<rect x="476.0" y="229.8" width="5.8" height="0.2" class="fill-soft-a"/>
<rect x="483.2" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="490.4" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="497.7" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="504.9" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="512.1" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="519.3" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="526.6" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="533.8" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="541.0" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="548.2" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="555.4" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="562.7" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="569.9" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<rect x="577.1" y="230.0" width="5.8" height="0.0" class="fill-soft-a"/>
<polyline points="60.0,230.0 67.2,230.0 74.4,230.0 81.7,230.0 88.9,230.0 96.1,230.0 103.3,230.0 110.6,230.0 117.8,230.0 125.0,230.0 132.2,230.0 139.4,230.0 146.7,230.0 153.9,229.9 161.1,229.7 168.3,229.0 175.6,226.9 182.8,222.0 190.0,211.7 197.2,193.1 204.4,164.5 211.7,127.6 218.9,89.1 226.1,59.4 233.3,48.2 240.6,59.4 247.8,89.1 255.0,127.6 262.2,164.5 269.4,193.1 276.7,211.7 283.9,222.0 291.1,226.9 298.3,229.0 305.6,229.7 312.8,229.9 320.0,230.0 327.2,230.0 334.4,230.0 341.7,230.0 348.9,230.0 356.1,230.0 363.3,230.0 370.6,230.0 377.8,230.0 385.0,230.0 392.2,230.0 399.4,230.0 406.7,230.0 413.9,230.0 421.1,230.0 428.3,230.0 435.6,230.0 442.8,230.0 450.0,230.0 457.2,230.0 464.4,230.0 471.7,230.0 478.9,230.0 486.1,230.0 493.3,230.0 500.6,230.0 507.8,230.0 515.0,230.0 522.2,230.0 529.4,230.0 536.7,230.0 543.9,230.0 551.1,230.0 558.3,230.0 565.6,230.0 572.8,230.0 580.0,230.0" class="stroke-b" fill="none"/>
<polyline points="60.0,229.4 67.2,229.2 74.4,229.0 81.7,228.8 88.9,228.5 96.1,228.2 103.3,227.8 110.6,227.3 117.8,226.8 125.0,226.2 132.2,225.4 139.4,224.6 146.7,223.7 153.9,222.6 161.1,221.5 168.3,220.2 175.6,218.8 182.8,217.3 190.0,215.6 197.2,213.9 204.4,212.1 211.7,210.2 218.9,208.2 226.1,206.2 233.3,204.2 240.6,202.2 247.8,200.2 255.0,198.3 262.2,196.5 269.4,194.9 276.7,193.4 283.9,192.0 291.1,190.9 298.3,190.0 305.6,189.3 312.8,188.9 320.0,188.8 327.2,188.9 334.4,189.3 341.7,190.0 348.9,190.9 356.1,192.0 363.3,193.4 370.6,194.9 377.8,196.5 385.0,198.3 392.2,200.2 399.4,202.2 406.7,204.2 413.9,206.2 421.1,208.2 428.3,210.2 435.6,212.1 442.8,213.9 450.0,215.6 457.2,217.3 464.4,218.8 471.7,220.2 478.9,221.5 486.1,222.6 493.3,223.7 500.6,224.6 507.8,225.4 515.0,226.2 522.2,226.8 529.4,227.3 536.7,227.8 543.9,228.2 551.1,228.5 558.3,228.8 565.6,229.0 572.8,229.2 580.0,229.4" class="stroke-d dashed" fill="none"/>
<text x="320.0" y="274.0" text-anchor="middle">position on the grid</text>
<rect x="64" y="8" width="14" height="10" class="fill-soft-a"/><text x="84" y="17" class="fig-small">teacher</text>
<line x1="164" y1="13" x2="190" y2="13" class="stroke-b"/><text x="196" y="17" class="fig-small">reverse-KL student</text>
<line x1="344" y1="13" x2="370" y2="13" class="stroke-d dashed"/><text x="376" y="17" class="fig-small">forward-KL student</text>
</svg>
Figure 1. A teacher with two modes, at -3 and 3, and the best one-bump student under each direction. The reverse-KL student sits on one mode and ignores the other. The forward-KL student has mean 0 and width 3.1, and puts most of its mass where the teacher has almost none.
```

The two students are good at different things. The forward-KL student has $\mathrm{KL}(\pi_T\|\pi_S) = 0.785$ and $\mathrm{KL}(\pi_S\|\pi_T) = 2.400$. The reverse-KL student has $\mathrm{KL}(\pi_S\|\pi_T) = 0.693 = \ln 2$ and $\mathrm{KL}(\pi_T\|\pi_S) = 17.7$. The value $\ln 2$ is no accident: the student matches one mode exactly, which carries half the teacher's mass, so each sample the student writes is something the teacher also writes, at half the probability.

Forward KL is *mode-covering*: the student spreads out to cover everything the teacher does, and so writes some answers the teacher never would. Reverse KL is *mode-seeking*: the student commits to one behavior it can do well. For a model that generates answers, a commitment is usually better than a blur, and a blur between two good answers can be a bad answer.

> [!KEY]
> Forward KL averages under the teacher and covers its modes. Reverse KL averages under the student and picks a mode. Reverse KL is the one whose expectation runs over the student's own samples.

## Why the student must write the text

Imitation learning has seen this problem. A cloned policy errs a little, lands in a state the expert never visited, errs more there, and the errors compound, as in [Imitation learning](/lessons/051-imitation-learning). DAgger (Ross and colleagues, 2011) fixes it by running the learner, then asking the expert what to do in the states the learner reached.

On-policy distillation is DAgger for language models. The student samples a response, so the prefixes are the ones it produces. The teacher then gives its next-token distribution at each of those prefixes, which is the expert label for that state. The training distribution matches the one the student will meet at test time, and that is what exposure bias lacked.

## The reward at each token

Reverse KL needs samples from the student, and the student's samples are what we now have. For a token $y_t$ sampled from $\pi_S$ at a prefix, the quantity $\log\pi_S(y_t) - \log\pi_T(y_t)$ is an unbiased one-sample estimate of the reverse KL at that position, because averaging it over $\pi_S$ gives the KL by definition. Negate it and you have a reward:

$$
r_t = \log \pi_T(y_t \mid x, y_{<t}) - \log \pi_S(y_t \mid x, y_{<t})
$$

Read it aloud: the reward is positive when the teacher likes the token more than the student does, and negative when the student is more eager than the teacher. Use the one-token example, student $[0.6, 0.4]$ and teacher $[0.9, 0.1]$. Token 0 earns $\ln\frac{0.9}{0.6} = +0.405$, and token 1 earns $\ln\frac{0.1}{0.4} = -1.386$. The student picks token 0 sixty percent of the time, so its mean reward is $0.6(0.405) + 0.4(-1.386) = -0.311$, minus the reverse KL.

Treat each token as one decision with discount $\gamma = 0$, so each token's return is its own reward. Policy gradient gives the loss:

$$
\mathcal{L}(\theta) = -\frac{1}{N}\sum_{t \in \text{real}} \operatorname{sg}(r_t)\,\log \pi_\theta(y_t \mid x, y_{<t})
$$

where $\operatorname{sg}$ means "no gradient flows through" and $N$ is the number of real tokens. At each prefix its expected gradient is the reverse-KL gradient:

$$
\nabla_\theta \mathrm{KL}(\pi_\theta \,\|\, \pi_T) = \mathbb{E}_{y \sim \pi_\theta}\big[(\log\pi_\theta(y) - \log\pi_T(y))\,\nabla_\theta\log\pi_\theta(y)\big]
$$

The identity comes from differentiating $\sum_y \pi_\theta \log(\pi_\theta/\pi_T)$ with the product rule. The term from differentiating the logarithm is $\sum_y \nabla_\theta\pi_\theta(y) = 0$, so one term is left, and writing $\nabla\pi = \pi\nabla\log\pi$ turns the sum into an expectation. The exercise verifies this with finite differences on a four-token categorical.

```python
reward = (logp_teacher - logp_student).detach()      # (B, T), no gradient
loss = -(mask * reward * logp_student).sum() / mask.sum()
```

> [!WARNING]
> Call `detach()` on the reward. The reward contains $\log\pi_S$, so a version with gradient adds a second term, $-\log\pi_S\,\nabla\log\pi_S$, and the expected gradient is no longer the KL gradient.

## Relation to RLHF and to GKD

RLHF in [RLHF with PPO](/lessons/068-rlhf-with-ppo) shapes the reward with a KL penalty: $r - \beta(\log\pi - \log\pi_{\text{ref}})$. Set $\beta = 1$, drop the task reward $r$, and replace $\pi_{\text{ref}}$ with the teacher, and you get $\log\pi_T - \log\pi_S$, the on-policy distillation reward. The penalty that kept RLHF close to a reference becomes the entire objective: stay close to the teacher, measured by reverse KL. The k1 estimator from that chapter is this same sampled quantity.

Agarwal and colleagues (GKD, 2024) put the idea in a general frame. Their loss is a *generalized Jensen-Shannon divergence* that interpolates between the two KL directions, and a mixing fraction decides how many training sequences the student samples and how many come from a fixed dataset. Supervised distillation and student-sampled distillation are two ends of one family. MiniLLM (Gu and colleagues, 2024) minimizes the reverse KL with policy gradient and several variance-reduction tricks. The Thinking Machines post (Lu and colleagues, 2025) describes the discount-0 recipe in this chapter, with the per-token reverse KL as a dense reward.

One honest caveat: a token also changes which prefixes come next, and the sequence-level reverse KL has gradient terms for that effect. The discount-0 loss drops them. The exercise shows that it still reaches the teacher on a toy, because a student that matches the teacher at every prefix has zero KL on every sequence.

## Dense and cheap

Compare this with RL from a sparse reward, such as the 0/1 check in [Reasoning with verifiable rewards](/lessons/072-reasoning-with-verifiable-rewards). A 1000-token answer earns one number, so the gradient must work out which of the 1000 tokens mattered. On-policy distillation gives 1000 numbers, one per token, each already assigned to the token it grades. It needs no reward model and no verifier, and one teacher forward pass over the student's response produces every reward. Teacher generation costs nothing, because the teacher never writes.

The price is the teacher. It must expose token log-probabilities, share the student's tokenizer, and be good at the task, because the student cannot exceed the model it copies.

## Build it step by step

The exercise defines a toy sequence task: three tokens from a vocabulary of four, a table model, and a fixed random teacher. You write six functions, in the order the checks test them.

1. `token_kl_losses(student_logits, teacher_logits, mask)` takes `(B, T, V)` logits and a `(B, T)` mask. It returns `(forward, reverse)`. Use `log_softmax`, sum over the vocabulary, then divide the masked sum by `mask.sum()`. The common mistake is dividing by $B \times T$, which lets padding dilute the loss.
2. `sampled_reverse_kl(student_logits, teacher_logits, tokens)` gathers both log-probabilities at the sampled tokens and subtracts them. The common mistake is the opposite sign, which estimates minus the KL.
3. `distill_reward(...)` is the negative of step 2.
4. `on_policy_distill_loss(student_logits, teacher_logits, tokens, mask)` is the masked, detached surrogate above. The common mistake is leaving the reward attached.
5. `fit_unimodal_student(teacher_logits, grid, direction)` scores every center and width with `token_kl_losses` and returns the best pair. The common mistake is swapping which direction goes with which name.
6. `train_on_policy_distill(teacher, seed)` samples from the student without gradient, scores with the loss, and takes an Adam step.

## Where this breaks

- **Mode collapse.** Reverse KL is mode-seeking, so a student can lose the teacher's diversity, as in the collapse of [AI feedback and self-improvement](/lessons/075-ai-feedback-and-self-improvement). If you need coverage, use a divergence closer to forward KL, or mix in teacher-written data.
- **A weak start.** A student that rarely produces anything the teacher likes gets uninformative rewards. Common practice starts with offline distillation, then switches to on-policy.
- **Noisy per-token estimates.** One sample per position has high variance when the two distributions disagree sharply. Averaging over many tokens, as the loss does, helps; computing the exact per-token KL from both full distributions removes the sampling noise at the cost of a larger vocabulary sum.

## Further reading

- [Agarwal and colleagues, "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes" (GKD)](https://arxiv.org/abs/2306.13649): a generalized JSD loss and a mixing fraction between dataset and student-sampled sequences. The arXiv version dates from 2023, and it appeared at ICLR 2024.
- [Gu and colleagues, "MiniLLM: On-Policy Distillation of Large Language Models"](https://arxiv.org/abs/2306.08543): reverse KL minimized with policy gradient for generative students.
- [Lu and colleagues, "On-Policy Distillation," Thinking Machines (2025)](https://thinkingmachines.ai/blog/on-policy-distillation/): the per-token reverse-KL reward with discount 0.
- [Hinton, Vinyals, and Dean, "Distilling the Knowledge in a Neural Network" (2015)](https://arxiv.org/abs/1503.02531): matching the teacher's output distribution.
- [Kim and Rush, "Sequence-Level Knowledge Distillation" (2016)](https://arxiv.org/abs/1606.07947): train the student on complete teacher-generated sequences.
- [Ross, Gordon, and Bagnell, "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning" (2011)](https://arxiv.org/abs/1011.0686): DAgger.
- [DeepSeek-AI, "DeepSeek-R1" (2025)](https://arxiv.org/abs/2501.12948): distilled smaller models fine-tuned on teacher samples.
- [Nathan Lambert, *RLHF Book*, Chapter 12, "Synthetic Data & Distillation"](https://rlhfbook.com/): on-policy distillation, framed against exposure bias.

Open questions researchers are working on:

- How should a student keep the teacher's diversity without giving up mode-seeking?
- How much does the dropped future-prefix term cost on long responses?
- Can a model act as its own teacher, for example with extra context in the prompt, without a larger model?

## The exercise

You write `token_kl_losses`, `sampled_reverse_kl`, `distill_reward`, `on_policy_distill_loss`, `fit_unimodal_student`, and `train_on_policy_distill`. The checks compare the two KLs with closed forms and a ragged mask, test the sampled estimator over 20000 draws within four standard errors, compare the loss gradient with its closed form and with finite differences on the exact KL, confirm that a bimodal teacher gives a one-mode reverse student and a two-mode forward student, and train a student that cuts the exact sequence-level reverse KL below 2% of its starting value on three teachers.
