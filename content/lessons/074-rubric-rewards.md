---
title: "Rubric rewards"
slug: 074-rubric-rewards
part: "Part 9 — RL for language models"
summary: "Grade answers a program can't check with a rubric, then mix it with a claims grader so each claim has its own include-or-skip threshold."
minutes: 35
gpu: false
objectives:
  - Explain why a recall-style rubric reward pays a model to list every claim it can, and why a precision-style claims reward pays it to say nothing.
  - Derive the per-claim rule that includes claim i exactly when alpha times its weight beats beta times the cost of it being false.
  - Gate a rubric score behind a test result, so a failed test always earns 0.
  - Compute the exact policy gradient of a Bernoulli policy over claims and train it to the threshold policy.
prereqs:
  - 073-verifiable-rewards
  - 068-judge-evaluation
  - 031-the-policy-gradient
exercise: 074-rubric-rewards
---

# Rubric rewards

Ask a model "why did the Roman Republic fall?" No program can check the answer, because no key holds the one right string. [Verifiable rewards](/lessons/073-verifiable-rewards) stops at questions that have such a key. Most questions people ask do not.

The usual fix is a grader that reads the answer against a checklist. A *rubric* is a list of facts a good answer should contain, each with a weight, and the grader adds up the weights of the items the answer covers. It is cheap to write and easy to read, and it never charges the model for a wrong sentence. This chapter shows what a model does with that flaw, builds the grader that closes it, and mixes the two so each claim is judged on its own terms.

## What you already know

- A reward is one number for a whole response, from [Verifiable rewards](/lessons/073-verifiable-rewards).
- A judge model can score an answer, and its scores have errors you can measure, from [Judge evaluation](/lessons/068-judge-evaluation).
- A policy with parameters $\theta$ improves when you move $\theta$ up the gradient of expected reward, from [The policy gradient](/lessons/031-the-policy-gradient).

## One response is a set of claims

Treat a response as a set of claims it states. Suppose the model has $K$ candidate claims it could make. Claim $i$ has two numbers attached:

- $w_i$ is the rubric weight, the credit the rubric grader gives for stating it.
- $p_i$ is the probability that the claim is true.

The response is a 0/1 vector $z$ with $z_i = 1$ when it states claim $i$. This is a toy, but it keeps the one effect the chapter is about, and the exercise can check it exactly.

Our running example has four claims about the Roman Republic, and a cost $c = 5$ for each false claim:

| Claim | Weight $w_i$ | $p_i$ true |
|---|---|---|
| 1. Generals commanded armies loyal to them | 3 | 0.9 |
| 2. Land reform stalled in the Senate | 2 | 0.7 |
| 3. A particular tax law caused the collapse | 2 | 0.5 |
| 4. An exact date and casualty count | 1 | 0.2 |

## Two graders that pull in opposite directions

The *rubric reward* pays for coverage. It is recall-style: it asks whether the answer says what a good answer says.

$$
R(z) = \sum_{i=1}^{K} w_i \, z_i
$$

Read this aloud: add the weight of every claim you state. In numpy, `(include * w).sum()`.

The *claims reward* pays for precision. A claims grader lists every claim in the response and checks each one. It charges $c$ for each claim that turns out false. Let $t_i = 1$ when claim $i$ is true.

$$
P(z) = -c \sum_{i=1}^{K} z_i \, (1 - t_i)
$$

Read this aloud: count the claims you stated that are false, and pay $c$ for each. A true claim costs nothing, and so does a claim you leave out. In numpy, `-c * (include * (1 - truth)).sum()`.

Train on each grader alone and see what the model learns.

Under $R$, a claim you add never lowers the score. A false claim earns its weight like a true one, because the rubric grader checks coverage and not accuracy. So the best policy states all four claims, including the shaky ones. This is a *brain dump*: the model packs its answer with everything that might score. Its rubric reward is $8$, the highest possible.

Under $P$, the best score is $0$, and you get it by stating nothing. Every claim you add can only lower the reward or leave it alone, so a model trained on $P$ alone learns silence.

You want a model that states a claim when it is likely enough to be true to be worth the risk. Mixing the graders gives a way to choose that.

```figure
<svg viewBox="0 0 560 280" role="img" aria-label="Bars of the net value of including each of four claims under the mix: 2.5, 0.5, minus 0.5, and minus 3.0">
  <line x1="50" y1="130" x2="530" y2="130" class="stroke-muted"/>
  <rect x="90" y="55" width="70" height="75" class="fill-b"/>
  <rect x="200" y="115" width="70" height="15" class="fill-b"/>
  <rect x="310" y="130" width="70" height="15" class="fill-bad"/>
  <rect x="420" y="130" width="70" height="90" class="fill-bad"/>
  <text x="125" y="46" text-anchor="middle" class="fig-bold">+2.5</text>
  <text x="235" y="106" text-anchor="middle" class="fig-bold">+0.5</text>
  <text x="345" y="162" text-anchor="middle" class="fig-bold">-0.5</text>
  <text x="455" y="238" text-anchor="middle" class="fig-bold">-3.0</text>
  <text x="125" y="258" text-anchor="middle">claim 1</text>
  <text x="235" y="258" text-anchor="middle">claim 2</text>
  <text x="345" y="258" text-anchor="middle">claim 3</text>
  <text x="455" y="258" text-anchor="middle">claim 4</text>
  <text x="50" y="24" class="fig-small">net value of stating each claim, alpha = beta = 1, c = 5</text>
</svg>
Figure 1. Each claim's net value is its weight minus its expected cost. Claims 1 and 2 are worth stating. Claims 3 and 4 lose more in expected false-claim cost than they earn, so a recall grader alone would still pay for them.
```

## Mix the graders and the problem splits by claim

Weight the two rewards with $\alpha$ and $\beta$, both non-negative. The mixed reward is $\alpha R + \beta P$. You do not know which claims are true when you answer, so take its expectation over the truth. Claim $i$ is true with probability $p_i$, so it is false, and costs $c$, with probability $1 - p_i$.

$$
\mathbb{E}\big[\alpha R + \beta P\big] = \sum_{i=1}^{K} z_i \,\Big(\alpha w_i - \beta c \,(1 - p_i)\Big)
$$

The bracket is the *net value* of claim $i$: what the rubric pays for it, minus the expected charge from the claims grader. It does not depend on any other claim. The sum has one term per claim, and $z_i$ switches it on. So you maximize the sum by switching on every term whose bracket is positive. That gives the rule:

$$
\text{include claim } i \iff \alpha \, w_i > \beta \, c \, (1 - p_i)
$$

Read this aloud: state a claim when the credit it earns beats the chance it is false times the penalty for being false. When $\beta = 0$, the right side is 0 and every claim passes, which is the brain dump. When $\alpha = 0$, the left side is 0 and no claim passes, which is silence. Between those, each claim has its own cutoff on $p_i$. Rearranged, the cutoff is $p_i > 1 - \alpha w_i / (\beta c)$. A heavily weighted claim can be stated even when it is likely false, and a cheap claim must be nearly certain.

Work the example by hand with $\alpha = \beta = 1$ and $c = 5$:

| Claim | $\alpha w_i$ | $\beta c (1-p_i)$ | Net value | State it? |
|---|---|---|---|---|
| 1 | 3 | $5 \times 0.1 = 0.5$ | $+2.5$ | yes |
| 2 | 2 | $5 \times 0.3 = 1.5$ | $+0.5$ | yes |
| 3 | 2 | $5 \times 0.5 = 2.5$ | $-0.5$ | no |
| 4 | 1 | $5 \times 0.8 = 4.0$ | $-3.0$ | no |

The best subset is claims 1 and 2, with expected reward $2.5 + 0.5 = 3.0$. The brain dump states all four, so its expected reward is the sum of the four net values, $-0.5$. It earns $R = 8$ from the rubric, but the mix also expects $0.1 + 0.3 + 0.5 + 0.8 = 1.7$ false claims, which cost $8.5$. Silence earns $0$. Figure 2 shows the three side by side.

Double $\alpha$ to $2$ and claim 3 enters, because $4 > 2.5$. Claim 4 stays out, because $2 < 4$. The best subset is claims 1, 2, and 3, worth $5.5 + 2.5 + 1.5 = 9.5$. So the ratio $\alpha / \beta$ sets how bold the model is.

```figure
<svg viewBox="0 0 560 240" role="img" aria-label="Expected mixed reward of three policies: say nothing 0, brain dump minus 0.5, threshold subset 3.0">
  <line x1="50" y1="170" x2="530" y2="170" class="stroke-muted"/>
  <circle cx="101" cy="170" r="4" class="fill-muted"/>
  <rect x="215" y="170" width="80" height="15" class="fill-bad"/>
  <rect x="385" y="80" width="80" height="90" class="fill-c"/>
  <text x="101" y="160" text-anchor="middle" class="fig-bold">0</text>
  <text x="255" y="206" text-anchor="middle" class="fig-bold">-0.5</text>
  <text x="425" y="70" text-anchor="middle" class="fig-bold">+3.0</text>
  <text x="101" y="226" text-anchor="middle">say nothing</text>
  <text x="255" y="226" text-anchor="middle">brain dump</text>
  <text x="425" y="226" text-anchor="middle">claims 1 and 2</text>
  <text x="50" y="24" class="fig-small">expected mixed reward, alpha = beta = 1, c = 5</text>
</svg>
Figure 2. The recall grader alone would rank the brain dump first, at a rubric score of 8. The mix ranks it below silence, and ranks the two-claim answer first.
```

> [!KEY]
> A rubric grader pays for recall, so it rewards a brain dump. A claims grader pays for precision, so it rewards silence. Their mix is a sum over claims, so each claim has its own test: state it when $\alpha w_i > \beta c (1 - p_i)$.

## Gate the rubric behind the tests

A coding agent has unit tests, and it also has quality that a test does not capture. If you add a rubric score to the test result, a response that fails every test can still collect credit for tidy explanations. Multiply instead. Let $T = 1$ when the tests pass, and $T = 0$ otherwise. Let $S$ be the rubric score. The reward is

$$
r = T \cdot S
$$

A failed test earns $0$ whatever the rubric says, so the rubric can only rank responses that already pass. A passing response with $S = 0.625$ earns $0.625$. A failing one with $S = 1$ earns $0$. The MiMo-V2.6 technical report (Xiaomi LLM-Core Team, 2026, section 4.3.1) uses this form for coding tasks where most rollouts already pass: the binary test reward times a solution-quality score and a behavior score. The gate also gives signal when every rollout in a group passes, because the rubric separates them.

> [!WARNING]
> Gating protects the test, not the rubric. A response that passes the tests can still brain-dump into the rubric, so the gate does not replace the claims grader.

## Train the threshold policy

A model does not pick a subset by arithmetic. It has a policy, and training moves the policy. Take the simplest policy: include claim $i$ with probability $q_i = \sigma(\theta_i)$, independently, where $\sigma$ is the sigmoid and $\theta_i$ is a learnable logit. Each claim is a two-way softmax between "state it" and "skip it".

The expected reward of this policy is $J(\theta) = \sum_i q_i g_i$, where $g_i = \alpha w_i - \beta c (1 - p_i)$ is the net value. The derivative of a sigmoid is $q(1 - q)$, so

$$
\frac{\partial J}{\partial \theta_i} = g_i \, q_i \, (1 - q_i)
$$

Read this aloud: a claim's logit moves up when its net value is positive and down when it is negative, and it moves fastest at $q_i = 1/2$, where the policy is least sure. In numpy, `g * q * (1 - q)`.

This is the exact version of REINFORCE from [The policy gradient](/lessons/031-the-policy-gradient). The sampled score for a Bernoulli is $z_i - q_i$, and only claim $i$'s own reward term correlates with it, so $\mathbb{E}[(z_i - q_i)\,\text{reward}] = q_i (1 - q_i) g_i$. The toy has a closed form, so you skip the sampling and its noise.

At $\theta = 0$, every $q_i = 0.5$, so $q(1-q) = 0.25$. The net values $(2.5, 0.5, -0.5, -3.0)$ give the gradient $(0.625, 0.125, -0.125, -0.75)$. The first two logits rise and the last two fall, which is the threshold policy.

With $\alpha = \beta = 1$, step size 1, and 600 steps, the trained inclusion probabilities are $(0.999, 0.997, 0.003, 0.001)$. The expected claim count settles near 2. Set $\beta = 0$ and every net value is positive. The expected claim count climbs from 2.0 to 3.87 after 20 steps and to 3.996 after 600. That climb is the brain dump, measured.

```python
q = 1 / (1 + np.exp(-theta))
grad = (alpha * w - beta * c * (1 - p)) * q * (1 - q)
theta = theta + lr * grad
```

> [!RECAP]
> - `rubric_reward` adds the weight of each included claim. `claims_reward` charges $c$ per included false claim. `gated_reward` multiplies a test result by a rubric score.
> - In the example, the best subset under $\alpha = \beta = 1$, $c = 5$ is claims 1 and 2, worth $3.0$. The brain dump is worth $-0.5$.
> - `expected_reward` is a sum of each claim's net value, so `optimal_subset` tests each claim alone.
> - `policy_gradient` is net value times $q(1-q)$, and `train_policy` ascends it.

## Build it step by step

The checks call the functions in this order. The first passes once `rubric_reward` is right.

1. **`rubric_reward(include, w)`.** Takes the 0/1 vector and the weights, and returns a float. Use `np.sum(include * w)`. Common mistake: summing `w` alone.
2. **`claims_reward(include, truth, c)`.** Returns $-c$ times the number of included false claims. Use `include * (1 - truth)`. Common mistake: counting every included claim, so true claims cost too.
3. **`gated_reward(tests_passed, rubric_score)`.** Multiply the two. Common mistake: adding them, which lets a failed test keep its rubric credit.
4. **`optimal_subset(w, p, c, alpha, beta)`.** Returns a boolean array. Compare `alpha * w` with `beta * c * (1 - p)` with a strict `>`. Common mistake: writing `p` where the rule has `1 - p`, which includes the claims most likely to be false. A tie stays out.
5. **`expected_reward(include, w, p, c, alpha, beta)`.** Returns the sum of `include * (alpha * w - beta * c * (1 - p))`. Accept probabilities as well as a mask. Common mistake: charging the penalty for every included claim instead of the expected false ones.
6. **`policy_gradient(theta, w, p, c, alpha, beta)`.** Compute `q = sigmoid(theta)`, then net value times `q * (1 - q)`. Common mistake: dropping the `(1 - q)` factor, which a finite-difference check catches.
7. **`train_policy(w, p, c, alpha, beta, lr, steps, theta0)`.** Ascend the gradient for `steps` iterations from `theta0`, or zeros. Common mistake: modifying `theta0` in place.

## Where this breaks

**The grader is a noisy model.** The toy treats $w_i$ and $t_i$ as exact. A real rubric or claims grader is a language model with errors (see [Judge evaluation](/lessons/068-judge-evaluation)). Mahmoud et al. (2026) train against a single rubric verifier and evaluate with a panel of other judges. Weak verifiers produce large training gains that do not transfer, and gains concentrate in completeness-type criteria while factual correctness and conciseness decline. That matches the brain-dump prediction.

**The model does not know $p_i$.** A policy has only its own belief. A badly calibrated model states claims it should skip, and the claims grader charges it for that.

**Claims are not independent.** One false premise can make several claims false together, and rubric items can overlap. The sum over claims then double-counts, and the per-claim test is only an approximation.

**You still have to tune $\alpha$, $\beta$, and $c$.** Too little $\beta$ and the model brain-dumps. Too much and it hedges or says nothing. The CS 2881 scribe notes mention tuned coefficients and describe no threshold, so the threshold here is derived from the toy and not quoted from them.

**Gating hides the rubric on failures.** A response that fails the tests gets no rubric gradient. The gate suits tasks the model already passes often.

## Further reading

The chapter rests on the following sources:

- [Gunjal et al., "Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains" (2025)](https://arxiv.org/abs/2507.17746): uses instance-specific rubrics as the reward for domains with no automatic check, and compares ways to combine rubric scores.
- [Mahmoud et al., "Reward Hacking in Rubric-Based Reinforcement Learning" (2026)](https://arxiv.org/abs/2605.12474): shows that weak rubric verifiers give training gains that do not transfer, concentrated in completeness-type criteria.
- [Xiaomi LLM-Core Team, "MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement" (2026)](https://arxiv.org/abs/2610.11959): multiplies the binary test reward by rubric scores in section 4.3.1.
- [Barak (ed.), "CS 2881 Fall 26, Lecture 6: Grading, Reward Hacking, and Alignment" (2026)](https://windowsontheory.org/2026/11/08/cs-2881-fall-26-lecture-6-grading-reward-hacking-and-alignment/): scribe notes for John Schulman's lecture on recall-oriented rubrics and precision-oriented claims graders.

Open questions that researchers are working on:

- How do you estimate a claim's chance of being true well enough for the model to apply the threshold?
- How do you write rubrics whose items a policy cannot satisfy with false claims?
- Which mix of graders stays stable as the policy gets stronger than its grader?

## The exercise

You write `rubric_reward`, `claims_reward`, and `gated_reward`, then `optimal_subset`, `expected_reward`, `policy_gradient`, and `train_policy`.

The checks compare `optimal_subset` with the closed-form thresholds on the four-claim example, including a tie. They compare `expected_reward` with an average over all $2^K$ truth worlds, and `optimal_subset` with a brute-force search over all $2^8$ subsets. They check `policy_gradient` against finite differences. They train the policy and require the two good claims above $0.9$ and the two bad ones below $0.1$, and with $\beta = 0$ they require the claim count to rise toward four.
