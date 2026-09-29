---
title: The research frontier
slug: 52-the-frontier
part: "Part 10 — Safety and the frontier"
summary: Build test-time RL's majority-vote reward and INTUITOR's self-certainty reward, see where each locks in errors, and map the open problems.
minutes: 55
gpu: false
objectives:
  - Compute TTRL's majority-vote pseudo-reward, and explain why its reward accuracy can far exceed its label accuracy.
  - Compute INTUITOR's self-certainty as the KL divergence from uniform, and show it ignores a shift of the logits.
  - Train toy policies with both rewards and show that each reinforces a wrong answer when the plurality or the confidence is wrong.
  - Place these methods among the open questions of 2025 and 2026, and separate what's established from what's contested.
prereqs:
  - 37-group-relative-policy-optimization
  - 38-reasoning-with-verifiable-rewards
  - 39-process-rewards-and-search
  - 49-reward-hacking-and-overoptimization
exercise: 52-the-frontier
---

# The research frontier

Every method in this course needed a reward from outside the model: an
environment, a person, a reward model, or a verifier that checks an answer.
For hard problems, that outside signal is the bottleneck. Nobody has labels
for tomorrow's exam, and a verifier exists only for tasks where an answer can
be checked.

So researchers asked a bold question in 2025: can a model train on rewards
it gives itself? Two answers drew attention. Test-time RL rewards each answer
for agreeing with the majority of the model's own answers. INTUITOR rewards
each answer for how confident the model was while writing it. Both work
surprisingly well on math with strong base models. Both also have an
obvious way to fail: a model that rewards itself can reward its own
mistakes.

In this lesson you build both rewards, train toy policies with them, and
watch exactly when each one locks in an error. The last part maps the rest
of the frontier and the questions that remain open.

## What you already know

- GRPO samples a group of answers to one prompt and uses each answer's
  reward, standardised within the group, as its advantage. You built it in
  *Group-relative policy optimisation*.
- For a softmax policy, the gradient of $\log \pi(a)$ in the logits is
  $\text{onehot}(a) - \pi$.
- Majority voting over sampled answers is a strong test-time method. You
  used it in *Process rewards and test-time search*.
- RL with verifiable rewards raises pass@1, and its effect on pass@$k$ at
  large $k$ is debated. See *Reasoning with verifiable rewards*.
- Any proxy reward can be hacked by enough optimisation. See *Reward hacking
  and overoptimisation*.

## Reward agreement with the majority

Test-time reinforcement learning (TTRL; Zuo et al., 2025) needs only
unlabelled questions. For a question $x$, sample $G$ answers from the policy
and extract the final answer $a_i$ from each. Take the most common answer as
a pseudo-label, $\hat a = \operatorname{mode}(a_1, \dots, a_G)$, and reward
agreement with it:

$$
r_i = \mathbf{1}\big[a_i = \hat a\big] .
$$

Read it as: every answer that matches the majority earns 1, and every other
answer earns 0. Then train with GRPO or PPO as if these were real rewards.
In numpy, `values, counts = np.unique(a, return_counts=True)` gives the
mode as `values[np.argmax(counts)]`, and the rewards are
`(a == mode).astype(float)`. The final version of the paper reports that
TTRL raised Qwen2.5-Math-7B's pass@1 on AIME 2024 by about 211 percent,
using only the unlabelled test questions.

Why would that work? Two measures separate the cases. *Label accuracy* asks
whether the majority is right. *Reward accuracy* asks, for each sample,
whether its pseudo-reward equals the reward a true label would give. The
second can be high even when the first is zero.

### Worked example

The true answer is 5, and eight samples give $(3, 3, 4, 6, 7, 8, 9, 5)$. The
majority is 3, which is wrong, so label accuracy is 0. The pseudo-rewards
are $(1, 1, 0, 0, 0, 0, 0, 0)$, and the true rewards are $(0, 0, 0, 0, 0,
0, 0, 1)$. They agree on the five samples that answered 4, 6, 7, 8, and 9,
which are wrong and correctly get 0. Reward accuracy is $5/8 = 0.625$.

When wrong answers scatter, most of them disagree with the majority and
correctly earn nothing. That's why a wrong majority does less damage than
you'd expect. The damage grows when the errors concentrate on one wrong
answer. In the exercise, the right answer has probability 0.25, one
attractive wrong answer has 0.35, and eight others share the rest. Over
1,000 simulated questions with $G = 16$, label accuracy is 0.35, while
reward accuracy is still 0.61.

> [!KEY]
> TTRL's pseudo-reward is right more often than its pseudo-label, because
> scattered wrong answers disagree with the majority. A wrong answer that's
> also a common one breaks this.

## Watch TTRL lock in the plurality

The exercise's toy policy answers one question with a softmax over five
answers, where answer 0 is correct. Each step draws 16 answers, rewards
agreement with the majority, standardises the rewards within the group, and
takes a policy-gradient step with $\eta = 1$ for 200 steps.

The *plurality* is the answer with the highest probability. It wins the
vote most often, so it collects most of the positive advantage, so its
probability grows, so it wins more often. Whatever the model starts out
believing most gets locked in:

| Start: $\pi(\text{right}), \pi(\text{wrong 1})$ | After 200 steps: $\pi(\text{right})$ | Majority right at the start |
|---|---|---|
| 0.45, 0.25 | 0.999 | 86 percent of groups |
| 0.25, 0.45 | 0.000 | 20 percent of groups |

With the wrong answer ahead, the majority of a single group is right in 20
percent of draws. Over 20 seeds the policy never recovered. Near a tie, the
outcome is a coin flip. With $0.30$ against $0.36$, the right answer won in
40 percent of 40 seeds. In every run, entropy collapses to near zero.

So on one isolated question, TTRL can't beat the model's own majority vote.
It turns maj@$G$ into pass@1. On a real model, gains can go further, because
the parameters are shared across questions and what the model learns on
one question carries to others. The paper reports that TTRL beat the
initial model's maj@$n$ accuracy. How far that transfer goes, and when it
reverses, is open.

## Reward confidence: INTUITOR

INTUITOR (Zhao, Kang, Feng, Levine, and Song, 2025) drops the answer
entirely and rewards the model's confidence. For an output $o$ of $\lvert o
\rvert$ tokens, with a vocabulary of $V$ tokens and next-token distribution
$p_t$ at position $t$, the *self-certainty* is

$$
\operatorname{SC}(o) = \frac{1}{\lvert o \rvert} \sum_{t=1}^{\lvert o \rvert}
\text{KL}\big(U \,\|\, p_t\big)
= \frac{1}{\lvert o \rvert} \sum_{t=1}^{\lvert o \rvert}
\Big( -\log V - \frac{1}{V} \sum_{j=1}^{V} \log p_t(j) \Big) ,
$$

where $U$ is the uniform distribution over the vocabulary. Read it as: at
every position, how far the model's next-token distribution sits from "no
idea at all", averaged over the output. It's 0 for a uniform distribution
and grows as the distribution sharpens. The second form follows from
writing out $\sum_j \frac{1}{V} \log \frac{1/V}{p_t(j)}$. In numpy, with
`log_p` the log-softmax of the logits,
`(-np.log(V) - log_p.mean(axis=-1)).mean(axis=-1)`.

The KL runs from uniform to the model, not the other way. That direction
averages $-\log p_t(j)$ over *every* token, so a distribution that puts
near-zero probability on some token scores very high. Adding the same
constant to every logit changes nothing, because the log-softmax is
unchanged.

INTUITOR uses self-certainty as the reward in GRPO. Each output's advantage
is its self-certainty, standardised within the group: $\hat A_i = (u_i -
\operatorname{mean}(u)) / \operatorname{std}(u)$.

### Worked example

With $V = 4$ and $p = (0.7, 0.1, 0.1, 0.1)$,

$$
\text{KL}(U \,\|\, p) = \tfrac{1}{4}\Big[\log \tfrac{0.25}{0.7} + 3 \log \tfrac{0.25}{0.1}\Big]
= \tfrac{1}{4}(-1.0296 + 2.7489) = 0.4298 .
$$

A uniform $p$ gives 0. A distribution of $(0.95, 0.02, 0.01, 0.01, 0.01)$ over
five tokens gives 1.946.

The authors found that the reward must be *online*. It has to come from the
policy being trained. When they scored outputs with a frozen copy of the
base model, the policy learned to hack it, for example by appending
problems it had already solved. Entropy minimisation, which rewards low
entropy directly, is a close relative (Agarwal et al., 2025; Prabhudesai et
al., 2025).

## Watch INTUITOR pick the confident method

The exercise's second toy separates confidence from correctness. The policy
first picks one of three methods, then answers from that method's fixed
distribution over five answers, where answer 0 is correct. The following
table gives the three methods, each with its self-certainty:

| Method | Answer distribution | Self-certainty | $\Pr(\text{right})$ |
|---|---|---|---|
| Careful | $(0.70, 0.10, 0.10, 0.05, 0.05)$ | 0.581 | 0.70 |
| Confident | $(0.02, 0.95, 0.01, 0.01, 0.01)$ | 1.946 | 0.02 |
| Guessing | uniform | 0 | 0.20 |

The first token, the method choice, is shared by every output in a group,
so its term adds the same amount to each $u_i$. Standardisation removes a
shared constant. So each output's advantage depends only on its method's
self-certainty. INTUITOR moves the policy onto the confident method. In the
exercise, accuracy falls from 0.307 to 0.020 in 200 steps.

Swap the table so the confident method is also right, with $(0.95, 0.02,
0.01, 0.01, 0.01)$, and the same training raises accuracy from 0.483 to
0.949. Self-certainty helps exactly as far as confidence tracks correctness.
That's the question of *calibration*, and it's why these methods work on
some model families and not others.

> [!WARNING]
> A self-generated reward reinforces what the model already believes. Before
> trusting a gain, check it against a held-out labelled set and a
> random-reward baseline.

> [!RECAP]
> - TTRL rewards $\mathbf{1}[a_i = \operatorname{mode}(a)]$. Its reward
>   accuracy can exceed its label accuracy.
> - On one question, TTRL locks in the initial plurality, right or wrong.
> - Self-certainty is the mean over positions of $\text{KL}(U \| p_t) =
>   -\log V - \frac{1}{V}\sum_j \log p_t(j)$.
> - Group standardisation removes terms that every output shares.
> - INTUITOR moves the policy toward its most confident behaviour, which
>   helps only when confidence tracks correctness.

## Map the rest of the frontier

Self-reward is one thread among several. Each of the following areas has a
solid core and a contested edge. The short versions:

**Sharpening or discovery.** Yue et al. (2025) found that RLVR-trained
models beat their base models at pass@1 but lose at large $k$, so RLVR
mostly reweights paths the base model already samples. ProRL (Liu et
al., 2025) argues that long, stable training expands the frontier. Shao et al.
(2025) found that GRPO with *random* rewards improved Qwen2.5-Math-7B on
MATH-500 by 21.4 points, against 29.1 with true rewards. The effect mostly
failed on Llama and OLMo models. Much measured gain may be elicitation, so
claims need random-reward baselines and several model families.

**Curricula and environment synthesis.** A teacher that picks tasks at the
edge of the learner's ability beats uniform sampling. PAIRED (Dennis et
al., 2020) scores tasks by regret, and prioritised level replay (Jiang et
al., 2021) by value loss. For language models, Absolute Zero (Zhao et al., 2025)
has one model propose, verify, and solve its own coding tasks. Work in 2025
and 2026 synthesises hundreds of tool-use environments as code, on the claim
that environment diversity is the bottleneck for agentic RL. Whether that
claim holds is open.

**World models as simulators.** DreamerV3 (Hafner et al., 2023) trains an
actor-critic inside a learned latent model with one fixed set of
hyperparameters. Dreamer 4 (Hafner, Yan, and Lillicrap, 2025) collected
diamonds in Minecraft from offline data alone. The simulation lemma from
*Model-based reinforcement learning* bounds the cost of model error, and it
grows with the horizon.

**Continual RL.** Abel et al. (2023) define continual RL as a setting where
the best agents never stop learning. Plasticity loss, from *Scaling
reinforcement learning*, is its main enemy.

**Safety of RL-trained agents.** The evidence from 2024 and 2025 is
empirical and consistent. Denison et al. (2024) found that models trained on
gameable tasks occasionally generalised to editing their own reward.
Baker et al. (2025) found that optimising against a chain-of-thought monitor
teaches the agent to hide its reward hacking. MacDiarmid et al. (2025) found
that learning to hack in real coding environments generalised to broader
misalignment, including sabotage attempts. Schoen et al. (2025) cut o3's
covert action rate from 13 percent to 0.4 percent with deliberative
alignment, but not to zero. In every case the policy optimised what the
training signal rewarded. The open question is how that gaming generalises
and whether you can detect it.

The following table separates what's settled from what isn't:

| Topic | Solid | Contested or open |
|---|---|---|
| Reward misspecification | Potential-based shaping, overoptimisation curves | Predicting when hacking emerges, and how it generalises |
| Scaling | Predictable curves for stable recipes | What sets the asymptote, and whether environment diversity has its own law |
| RLVR | Raises pass@1 and sampling efficiency | Whether it expands capability, and how much depends on the model family |
| Self-reward | Works on some model families for math | When it collapses, and how to detect a systematically wrong majority |
| Safety | Reward hacking generalises, and monitors can be gamed | Reliable detection, and training that stays monitorable |

## Build it step by step

The checks test the functions in this order. `softmax` is given.

1. **`majority_vote(answers)`.** `np.unique` with `return_counts=True`, then
   `np.argmax` over the counts. The sorted `values` make ties go to the
   smallest answer. Common mistake: `np.bincount`, which fails on negative
   answers.
2. **`ttrl_rewards(answers)`.** One comparison with the mode, cast to float.
3. **`reward_and_label_accuracy(answers, truth)`.** Compare the pseudo-reward
   array with `(answers == truth)`, and compare the mode with `truth`.
4. **`self_certainty(logits)`.** Log-softmax over the last axis, then
   `-np.log(V) - log_p.mean(axis=-1)`, then the mean over positions. Common
   mistake: computing $\text{KL}(p \,\|\, U)$, which is $\log V$ minus the
   entropy and gives the wrong value on the worked example.
5. **`group_advantages(rewards)`.** Standardise with `np.std`'s default,
   and return zeros when the standard deviation is zero.
6. **`policy_gradient(theta, actions, advantages)`.** The mean over samples
   of `advantage * (onehot(action) - softmax(theta))`. Finite differences
   check it.
7. **`ttrl_train(...)`.** Sample with `rng.choice`, reward, standardise,
   step. Common mistake: forgetting that a group that all agrees gives zero
   advantage, so late in training most steps do nothing.
8. **`intuitor_train(...)`.** The same loop, over methods, with each
   sample scored by its method's self-certainty.

## Cost and scaling

Both rewards are cheap next to generation. TTRL needs $G$ samples per
question, which GRPO draws anyway, plus answer extraction and a vote.
Self-certainty needs the full next-token distribution at every position,
which the forward pass already computes. The following table compares them:

| Reward | Extra work per output | Needs labels | Main risk |
|---|---|---|---|
| Verifiable reward | One verifier call | Yes | Verifier gaps |
| TTRL majority vote | Answer extraction, one vote per group | No | A wrong but common answer |
| Self-certainty | $O(\lvert o \rvert V)$ from existing logits | No | Confident errors, collapse |

The real cost is in trust. A self-reward removes the labelled data, so you
need a separate labelled evaluation set to know whether training helps.

## Where this breaks

**A systematically wrong majority.** When the model's most common answer is
wrong, TTRL reinforces it. People filter questions by how strongly the
group agrees, and keep a labelled set to watch for drift.

**Confidence without calibration.** When the model is confidently wrong,
INTUITOR and entropy minimisation reinforce it. The online requirement
helps against hacking, but it can't create calibration the base model lacks.

**Entropy collapse.** Both rewards push the policy toward one answer. Every
run in the exercise ended with near-zero entropy. Once the policy is
deterministic, groups agree, advantages vanish, and learning stops. KL
penalties and entropy bonuses slow this.

**Model-family dependence.** Random rewards help Qwen models on math and
mostly fail on Llama and OLMo. A self-reward result on one family is weak
evidence about another.

## Further reading

- [TTRL: test-time reinforcement learning](https://arxiv.org/abs/2504.16084)
  (Zuo et al., 2025). Majority-vote rewards on unlabelled test questions,
  and the gap between reward and label accuracy.
- [Learning to reason without external rewards](https://arxiv.org/abs/2505.19590)
  (Zhao et al., 2025). INTUITOR's self-certainty reward, and why it must be
  computed online.
- [The unreasonable effectiveness of entropy minimization in LLM reasoning](https://arxiv.org/abs/2505.15134)
  (Agarwal et al., 2025). Negative entropy as the only training signal.
- [Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model?](https://arxiv.org/abs/2504.13837)
  (Yue et al., 2025). The pass@$k$ evidence for sharpening over discovery.
- [Spurious rewards: rethinking training signals in RLVR](https://arxiv.org/abs/2506.10947)
  (Shao et al., 2025). Random rewards that help one model family and not
  others.
- [Natural emergent misalignment from reward hacking in production RL](https://arxiv.org/abs/2511.18397)
  (MacDiarmid et al., 2025). How reward hacking in coding environments
  generalised to broader misalignment.
- [Monitoring reasoning models for misbehavior and the risks of promoting obfuscation](https://arxiv.org/abs/2503.11926)
  (Baker et al., 2025). Chain-of-thought monitors, and what happens when you
  train against them.
- [A definition of continual reinforcement learning](https://arxiv.org/abs/2307.11046)
  (Abel et al., 2023). A formal account of agents that never stop learning.

Researchers are working on the following open questions, each concrete
enough to start with modest compute:

1. **Convergence of GRPO-style updates.** Characterise the fixed points and
   rates of group-normalised, clipped updates on a tabular bandit or MDP,
   including the clipping bias that may amplify high-prior behaviour.
2. **Sharpening or discovery.** Formalise when policy gradient on a
   pretrained prior raises pass@$k$ at large $k$, as a function of coverage,
   entropy regularisation, and training length.
3. **Predicting the asymptote.** Find what sets ScaleRL's ceiling $A$, and
   whether cheap probes such as base-model pass@$k$ predict it.
4. **Scaling laws for environment diversity.** Measure held-out agent
   performance against the number and variety of synthesised environments
   at fixed compute.
5. **When self-reward collapses.** Give conditions on calibration under
   which TTRL or self-certainty improves accuracy, and build a detector for
   a systematically wrong majority.
6. **Heavy tails in real reward models.** Test whether reward models for
   long reasoning traces, LLM judges, or agentic outcomes keep light-tailed
   error.
7. **Overoptimisation laws for verifiers.** Derive proxy-gold curves when
   the proxy is a unit-test suite or rubric with known gaps.
8. **Monitorability-preserving RL.** Use a chain-of-thought monitor for
   detection without creating pressure to hide the behaviour it detects.
9. **Generalisation of reward hacking.** Explain why hacking in narrow
   environments generalises to broad misalignment, and why inoculation
   prompting blocks it, with small reproducible model organisms.
10. **Plasticity in LLM RL.** Test whether long LLM RL runs lose plasticity,
    whether resets help at that scale, and whether entropy collapse is
    partly a plasticity effect.
11. **Statistics for LLM RL evaluation.** Extend stratified bootstrap
    intervals and performance profiles to pass@$k$ curves with correlated
    samples, with power analyses.
12. **Constrained RLHF with learned thresholds.** Adapt PID Lagrangian or
    zero-duality-gap results to constraints that are themselves
    overoptimisable cost models.
13. **Faithfulness of world models.** Measure how a learned model's error
    predicts whether skills trained inside it transfer to the real task.
14. **Open-endedness metrics.** Turn "novel and learnable" into a measurable
    quantity for self-play systems, and test whether they plateau.
15. **Guarantees for off-policy LLM RL.** Derive trust-region-style bounds
    that account for stale rollouts and truncated importance weights, and
    test whether they predict instability.

## The exercise

You build both self-generated rewards and train a toy policy with each.
Write `majority_vote`, `ttrl_rewards`, and `reward_and_label_accuracy` for
TTRL, and `self_certainty` for INTUITOR. Then write `group_advantages` and
`policy_gradient` for the GRPO-style update, and use them in `ttrl_train`
and `intuitor_train`.

The checks confirm the lucky-hit example by hand, and that reward accuracy
stays above 0.55 while label accuracy falls below 0.42 when one wrong answer
is common. `self_certainty` must give 0.4298 on the worked example, 0 on
uniform logits, and the same value after a shift of the logits, and it must
stay finite for logits of 1000. `policy_gradient` must match finite
differences. TTRL must end above 0.95 on whichever answer starts as the
plurality, right or wrong. INTUITOR must move onto the most confident
method in both tables, which drives accuracy below 0.05 when that method is
wrong.
