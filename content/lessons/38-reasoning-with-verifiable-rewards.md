---
title: Reasoning with verifiable rewards
slug: 38-reasoning-with-verifiable-rewards
part: "Part 8 — Reinforcement learning for language models"
summary: Reward answers a program can check, measure pass@k without bias, watch entropy collapse, and ask honestly what RL adds.
minutes: 70
gpu: true
objectives:
  - Write a verifiable reward for the R1-Zero answer format that rejects malformed outputs.
  - Implement the unbiased pass@k estimator in a numerically stable form, and check it against exact combinatorics.
  - Derive the first-order entropy change of a softmax as a covariance, and measure entropy collapse during RLVR.
  - "Train a toy arithmetic model with RLVR, and explain why pass@1 can rise while pass@k at large k falls."
  - State the spurious-rewards result, and what a zero-information reward does and doesn't do on-policy.
prereqs:
  - 34-reward-models
  - 37-group-relative-policy-optimization
exercise: 38-reasoning-with-verifiable-rewards
---

# Reasoning with verifiable rewards

A learned reward model is a guess about what people want, and a policy that
pushes hard enough finds where the guess is wrong. For some tasks you don't
need a guess. A maths answer is right or wrong. A program passes its tests or
it doesn't. A response follows the required format or it doesn't. A few lines
of Python can check each one.

Reinforcement learning with verifiable rewards, or RLVR, trains against such
checks. The name comes from Tülu 3 (2024). DeepSeek-R1-Zero (2025) showed how
far it goes. It took a base model that had never been tuned to follow
instructions, gave it a fixed answer template, and trained it with GRPO on
rule-checked maths. Its accuracy rose steadily, and its answers grew longer
and more reflective.

This lesson builds the parts of an RLVR run and the tools to judge one. You
write the reward check, the standard way to measure a model that gets several
tries, and a way to see the model's diversity drain away. Then you train a
toy arithmetic model and face a contested question: does RL teach a model new
answers, or only make it pick answers it could already find?

## What you already know

- GRPO scores a group of responses per prompt and uses group-normalised
  advantages, from *Group-relative policy optimisation*. With one gradient
  step per batch its clipped loss has the same gradient as plain REINFORCE.
- A reward model can be hacked. A policy optimising it finds its mistakes,
  from *Reward models*.
- The *entropy* of a distribution, $H(\pi) = -\sum_a \pi_a \log \pi_a$, is
  zero when one outcome is certain and largest when all are equally likely.
- A softmax policy $\pi = \operatorname{softmax}(z)$ moves when its logits
  $z$ move.
- REINFORCE with a baseline that doesn't depend on the action is unbiased.

## Reward what a program can check

A *verifier* is a function from a response to a reward. DeepSeek-R1-Zero asks
the model to reason inside `<think>` tags and answer inside `<answer>` tags.
The prompt ends with `<think>`, so the model starts reasoning at once. The
**Run** block uses the same template:

```text
A conversation between User and Assistant. The User asks a question, and the
Assistant solves it. The Assistant first thinks about the reasoning process in
the mind and then provides the User with the answer. The reasoning process is
enclosed within <think> </think> and answer is enclosed within <answer>
</answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. User: What is 7 * 43 + 18? Assistant: <think>
```

The exercise's `answer_reward(text, target)` pays 1 only when all of the
following hold:

- The response closes its reasoning with `</think>`.
- An `<answer>` block follows, and it's closed.
- The block holds one integer, with an optional minus sign and nothing else.
- That integer equals the target.

In Python, `re.search(r"</think>\s*<answer>(.*?)</answer>", text, re.DOTALL)`
finds the block and `re.fullmatch(r"-?\d+", content.strip())` accepts it.

> [!WARNING]
> A loose verifier is a reward model with a bug. A check that accepts any
> response *containing* the right number pays a model that lists every
> number from 1 to 1000. Test your verifier on malformed and adversarial
> strings before you train against it.

## Measure a model that gets k tries

Reasoning models are often judged by *pass@k*: the chance that at least one
of $k$ independent samples is correct. It measures whether the model *can*
solve a problem, given tries. pass@1 is the plain accuracy.

Estimating it from exactly $k$ samples is noisy. Chen and colleagues (2021)
draw $n \ge k$ samples per problem, count the $c$ correct ones, and ask: if
you picked $k$ of the $n$ samples at random, without replacement, how likely
is at least one to be correct?

$$
\text{pass@}k = \mathbb{E}_{\text{problems}}\left[1 - \frac{\binom{n - c}{k}}{\binom{n}{k}}\right]
$$

Read it as: $\binom{n - c}{k} / \binom{n}{k}$ is the fraction of size-$k$
subsets that hold only wrong samples. One minus that is the fraction with at
least one correct sample. Averaged over problems, it's an unbiased estimate
of pass@$k$ for each problem.

For example, $n = 10$ samples with $c = 3$ correct and $k = 5$ give
$1 - \binom{7}{5} / \binom{10}{5} = 1 - 21/252 = 0.917$. The tempting shortcut
$1 - (1 - c/n)^k = 1 - 0.7^5 = 0.832$ plugs an estimate into a nonlinear
formula, and that makes it biased.

The binomial coefficients overflow a float once $n$ reaches a few thousand.
Cancel them into a product instead:

$$
\frac{\binom{n - c}{k}}{\binom{n}{k}} = \prod_{j = n - c + 1}^{n}\left(1 - \frac{k}{j}\right)
$$

Read it as: $c$ factors, each a little below one. When $n - c < k$, every
subset holds a correct sample and pass@$k$ is exactly 1. In numpy:

```python
if n - c < k:
    return 1.0
return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
```

> [!TIP]
> On the toy you know each problem's exact success probability $p$. There,
> pass@$k$ is $1 - (1 - p)^k$, the limit of the unbiased estimator as $n$
> grows. The shortcut is biased only when you plug in an estimate of $p$.

## Train a toy arithmetic model

The toy answers 16 addition questions, $a + b$ with $a$ and $b$ from 0 to 3.
Its vocabulary has 15 tokens: ten digits, four *method* tokens, and an end
token. A response is two tokens, a method and then a digit:

- The method comes from one shared distribution that ignores the question.
  Think of it as a habit, such as whether the model writes its working as a
  column sum or counts on its fingers.
- The digit comes from a table indexed by question and method.

Each question has one *suited* method. Ten questions suit method 0, and two
each suit methods 1, 2, and 3. In the base model the four methods are equally
likely. The suited method gives the right digit with probability $0.8$, and any
other method with probability $0.002$. So each question's base success rate
is $0.25 \times 0.8 + 0.75 \times 0.002 = 0.2015$.

Training is RLVR as the GPU run does it. For each of the 16 questions, sample
a group of 8 responses. The reward is 1 when the digit is right. Compute
GRPO advantages and take one step of `pg_loss`, the token-mean policy
gradient, with SGD at step size 1. The following table shows the exact pass@$k$
and the batch entropy during 200 steps:

| Step | pass@1 | pass@8 | pass@64 | Entropy |
|---|---|---|---|---|
| 0 | $0.201$ | $0.835$ | $1.000$ | $1.574$ |
| 25 | $0.451$ | $0.738$ | $0.979$ | $1.086$ |
| 50 | $0.512$ | $0.668$ | $0.858$ | $0.794$ |
| 200 | $0.584$ | $0.638$ | $0.717$ | $0.571$ |

pass@1 nearly triples. pass@64 falls from 1 to $0.717$. The trained model
beats the base model with one try and loses to it with eight or more. The
next two sections explain both halves.

## Watch entropy collapse

Entropy falls from $1.57$ to $0.57$ nats in the table. Most of the drop is
the method habit: the probability of method 0 rises to about $0.99$. Cui and
colleagues (2025) found the same pattern in RLVR on real models. Entropy
collapses early, and performance levels off once it's spent. They fitted an
empirical law, $R = -a e^{H} + b$, between validation performance $R$ and
entropy $H$.

Why does entropy fall? Differentiate $H$ with respect to one logit:

$$
\frac{\partial H}{\partial z_k} = -\pi_k\big(\log \pi_k - \mathbb{E}_{\pi}[\log \pi]\big)
\quad\Longrightarrow\quad
\Delta H \approx -\operatorname{Cov}_{a \sim \pi}\big(\log \pi_a,\ \Delta z_a\big)
$$

Read it as: to first order, entropy falls when the logits that rise belong to
tokens that are already likely, and rises when they belong to unlikely ones.
The covariance is taken under $\pi$. In PyTorch,
`-(p * (logp - (p * logp).sum(-1, keepdim=True)) * (dz - (p * dz).sum(-1, keepdim=True))).sum(-1)`.

A positive advantage on the most likely token is the common case in RLVR,
because the likely answers are the ones sampled and rewarded. Each such step
has positive covariance, so each lowers the entropy. For a worked example, take
$\pi = (0.7, 0.2, 0.1)$, whose entropy is $0.802$, and a natural-gradient step
$\Delta z = 0.1 \times A$ with $A = (+1, -1, 0)$:

- $\log \pi = (-0.357, -1.609, -2.303)$, and $\operatorname{Cov}_\pi(\log \pi, A) = 0.473$.
- The estimate is $-0.1 \times 0.473 = -0.0473$. The exact change is $-0.0487$.
- With $A = (-1, +1, 0)$, the estimate is $+0.0473$ and the exact change is
  $+0.0454$.

The remedies raise the entropy of the update. DAPO's clip-higher lets rare
tokens grow faster. Cui and colleagues' Clip-Cov and KL-Cov restrict the
updates on the tokens with the highest covariance. A plain entropy bonus is
unstable at scale in several reports.

> [!KEY]
> A step changes entropy by about minus the covariance between each token's
> log-probability and its logit change. RLVR rewards likely answers, so it
> drains entropy unless something pushes back.

## Ask whether RL adds capability

Yue and colleagues (2025) compared base and RLVR-trained models across maths,
code, and visual reasoning. The RL models won at small $k$. The base models
won at large $k$. And the reasoning paths the RL models used already had high
probability under the base model. Their reading: RLVR sharpens the
distribution toward answers the base model could already sample, and loses
some rare correct paths along the way.

The toy reproduces that crossover, and you can see the mechanism. The shared
habit is pushed toward method 0, because ten of the sixteen questions reward
it. The six questions that need another method get sampled with their suited
method less and less. Their success rate falls from $0.20$ to about $0.005$.
pass@64 counts those questions, so it falls.

The toy doesn't settle the question, and neither does the crossover. Run the
same toy with Adam at step size $0.1$ instead of SGD. pass@1 reaches $0.998$
and pass@64 stays at $1.0$. Method 0 answers the six minority questions right
with probability $0.002$, so a success turns up now and then. Adam scales
each table entry's step by that entry's own gradient history, so those rare
successes teach the table in full. RL learned answers the base model almost
never gave. The difference isn't magic in Adam. It shows that the outcome
depends on whether rare correct paths get sampled, and on how hard the update
follows them.

The evidence on real models is contested:

- DeepSeekMath (2024) already reported RL raising majority-vote accuracy but
  not pass@$k$.
- ProRL (Liu and colleagues, 2025) trained far longer, with KL control,
  periodic resets of the reference, and diverse tasks. It reports pass@$k$
  gains even at large $k$, including on tasks where the base model fails.
- Wen and colleagues (2025) argue that pass@$k$ on final answers counts lucky
  guesses. With a metric that also checks the reasoning, they find RLVR does
  improve correct reasoning.

> [!KEY]
> RLVR reliably raises pass@1. Whether it raises pass@$k$ at large $k$ depends
> on training length, task diversity, exploration, and the metric, and it
> remains an open question.

## Know what a random reward can do

Shao and colleagues (2025) trained Qwen2.5-Math-7B with GRPO on rewards
assigned at random. MATH-500 accuracy rose by 21.4 points. Spurious rewards
that helped Qwen models often failed for Llama 3 and OLMo 2. The authors trace
the effect to a bias from GRPO's clipping, which amplifies behaviours the base
model already prefers. For Qwen2.5-Math that behaviour is reasoning written
as code, which rose from 65% to over 90% of responses.

One part of this is exact. When the reward doesn't depend on the response,
the advantage is independent of the response, so

$$
\mathbb{E}\big[\hat A_i\, \nabla_\theta \log \pi_\theta(y_i)\big] = \mathbb{E}\big[\hat A_i\big]\, \mathbb{E}\big[\nabla_\theta \log \pi_\theta(y_i)\big] = 0
$$

Read it as: on-policy, with one gradient step per batch, a random reward
moves the policy nowhere on average. The exercise checks this by
enumerating every group and every reward pattern. Any systematic drift must
come from somewhere else: several updates per batch, clipping, normalisation,
or the optimiser's state.

The clipping mechanism itself is the paper's claim, and it's harder to test.
A tabular toy with a dominant token, random rewards, and four clipped updates
per batch does drift. But the drift is away from the dominant token, and
clipping makes it smaller, the opposite of what the paper describes. The paper's
argument concerns whole-sequence probabilities in a real model, which a toy
may not capture. So the exercise checks only the zero-drift identity. The
practical lesson stands either way. Validate an RLVR claim on more than one
model family, and include a random-reward baseline.

## Run R1-Zero-style RLVR on Qwen3-0.6B-Base

The **Run** block trains Qwen3-0.6B-Base, a model with no instruction tuning,
with your functions when a GPU is available. The questions have the form
$a \times b + c$, with $a$ from 2 to 9 and $b$ and $c$ from 10 to 99. Each
step samples 8 questions and 8 responses each, at temperature 1, stopping at
`</answer>` or 128 tokens. The reward is your `answer_reward`. The step is
your `grpo_advantages` and `pg_loss`, with AdamW at learning rate $10^{-6}$.
R1-Zero also used a KL penalty. This run drops it, as DAPO and Dr. GRPO do.

Every 10 steps it draws 16 samples for each of 16 held-out questions. It
counts the correct ones and prints pass@1, pass@4, and pass@16 from your
`pass_at_k`, with the mean reward and response length. Forty steps is a short
run. Watch whether pass@16 moves as much as pass@1.

> [!RECAP]
> - `answer_reward` pays 1 only for `</think>`, then a closed `<answer>`
>   block holding exactly the right integer.
> - pass@$k$ is $1 - \binom{n-c}{k}/\binom{n}{k}$, computed as
>   $1 - \prod_{j=n-c+1}^{n}(1 - k/j)$, and 1 when $n - c < k$.
> - `pg_loss` is minus the token mean of $\hat A_i \log \pi$ over the mask.
> - $\Delta H \approx -\operatorname{Cov}_\pi(\log \pi, \Delta z)$.
> - On the toy, RLVR with SGD raises pass@1 and lowers pass@64. The outcome
>   depends on whether rare correct paths get sampled and followed.

## Build it step by step

The checks test the functions in this order.

1. **`pass_at_k(n, c, k)`.** Takes ints, or an array for `c`. Loop over
   `np.ndindex(c.shape)` and use the product form. Common mistake: calling
   `math.comb` and dividing floats, which overflows at $n = 5000$.
2. **`answer_reward(text, target)`.** Search for the block after `</think>`,
   strip it, and `fullmatch` an integer. Common mistake: `re.search(r"\d+")`
   inside the block, which accepts `42 or 43`.
3. **`grpo_advantages(rewards, group_size, eps)`.** As in *Group-relative
   policy optimisation*, with the population standard deviation.
4. **`pg_loss(token_logps, mask, advantages)`.** Zero the padding with
   `torch.where`, multiply by `advantages[:, None]`, sum, and divide by
   `mask.sum()`. Common mistake: multiplying `-inf` padding by a zero mask,
   which gives `nan`.
5. **The random-reward check** uses your two functions above.
6. **`token_entropy(logits, mask)`.** `log_softmax`, then
   `torch.where(p > 0, p * logp, 0)`, summed over the vocabulary and
   averaged over the mask.
7. **`entropy_change_estimate(logits, delta_logits)`.** Centre both
   quantities under `p` along the last axis, multiply, weight by `p`, sum,
   and negate.
8. **`train_rlvr_toy(...)`.** Build the batch with
   `torch.arange(16).repeat_interleave(group_size)`, so groups are
   contiguous. Record the entropy of the batch's logits, and each pass@$k$
   row from `model.success_probs()`. Common mistake: forgetting
   `opt.zero_grad()`, so gradients pile up across steps.

## Cost and scaling

Sampling dominates an RLVR run, and verification is nearly free. The following
table shows where the work goes:

| Part | Cost | Grows with |
|---|---|---|
| Sampling a group | $G$ generations per prompt | $G$, response length, model size |
| Verifier | A regular expression or a test run | Response length, test suite size |
| Gradient step | One forward and backward pass per response | Total tokens |
| pass@$k$ evaluation | $n$ samples per problem, $n \ge k$ | $n$ and the problem count |
| The `pass_at_k` estimator | $c$ multiplications per problem | $c$, at most $n$ |

Long reasoning makes sampling slower in two ways: more tokens, and a batch
that waits for its longest response. Code verifiers that run tests need a
sandbox and can cost as much as generation. Large-$k$ evaluation is
expensive, because it needs many samples per problem. The **Run** block's
evaluation draws 256 responses each time.

## Where this breaks

**Verifier loopholes.** The model finds what the check doesn't check:
answers in the wrong place, tests it can edit, or formats the parser
misreads. Test the verifier, keep it out of the model's reach, and read
samples.

**Entropy collapse.** Diversity drains early and learning stalls. Clip-higher,
covariance-based clipping, and filtering out prompts the model always or
never solves all help.

**Truncation.** A response cut off at the length limit gets reward 0, the
same as a wrong answer. That punishes long, correct reasoning. DAPO masks or
softly penalises overlong responses instead.

**Contaminated baselines.** A base model's accuracy depends heavily on the
prompt template. Compare base and trained models with the same template and
the same sampling settings.

**Single-model claims.** Effects such as spurious-reward gains appear on one
model family and not another. Test on more than one.

## Further reading

The following sources cover the recipe and the debates in this lesson:

- [DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025)](https://arxiv.org/abs/2501.12948): R1-Zero's template, rule-based rewards, and length growth.
- [Lambert et al., "Tülu 3: Pushing Frontiers in Open Language Model Post-Training" (2024)](https://arxiv.org/abs/2411.15124): names RLVR and uses it alongside preference tuning.
- [Chen et al., "Evaluating Large Language Models Trained on Code" (2021)](https://arxiv.org/abs/2107.03374): the unbiased pass@k estimator and its stable form.
- [Yue et al., "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?" (2025)](https://arxiv.org/abs/2504.13837): the crossover at large k.
- [Liu et al., "ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models" (2025)](https://arxiv.org/abs/2505.24864): longer, more diverse training that reports gains at large k.
- [Wen et al., "Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs" (2025)](https://arxiv.org/abs/2506.14245): a metric that checks the reasoning as well as the answer.
- [Cui et al., "The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models" (2025)](https://arxiv.org/abs/2505.22617): the covariance law and Clip-Cov.
- [Shao et al., "Spurious Rewards: Rethinking Training Signals in RLVR" (2025)](https://arxiv.org/abs/2506.10947): gains from random rewards, and why they're model-specific.

Open questions that researchers are working on:

- Under what training conditions does RLVR add problems to the set a model can solve at any $k$?
- Which entropy controls keep exploration alive without degrading the output?
- Can verifiable rewards extend to tasks without a single checkable answer?

## The exercise

You build the evaluation and training pieces of an RLVR run. Write
`pass_at_k` and `answer_reward`, then `grpo_advantages` and `pg_loss`, then
`token_entropy` and `entropy_change_estimate`. Last, write `train_rlvr_toy`,
which trains `ToyArithmeticLM` with GRPO advantages and records exact pass@$k$
and batch entropy.

The checks verify `pass_at_k` against exact fractions, brute-force subset
counts, and a case whose binomial coefficients overflow a float. They run
`answer_reward` on nine well-formed and malformed responses. They check
`pg_loss`'s gradient in closed form, and they enumerate every group and reward
pattern to confirm that a random reward gives a zero expected update. They
compare the entropy estimate with the exact change. Then they train the toy
twice. With SGD, pass@1 must rise above $0.45$, entropy must fall below 60%
of its start, and pass@64 must fall below the base model's. With Adam, pass@1
must pass $0.9$ and pass@64 must not fall.
