---
title: Process rewards and test-time search
slug: 39-process-rewards-and-search
part: "Part 8 — Reinforcement learning for language models"
summary: Spend compute at answer time with best-of-n, weighted voting, and beam search steered by a process reward model.
minutes: 55
gpu: false
objectives:
  - Compute best-of-n's expected reward exactly from the order statistics of the reward.
  - Implement majority and weighted majority voting, and show when voting on a wrong mode fails.
  - Compute the exact step-level value of a toy reasoning task with one backward pass.
  - Run PRM-guided beam search and explain why a perfect value never leads it to a dead end.
prereqs:
  - 10-planning-and-learning
  - 34-reward-models
  - 38-reasoning-with-verifiable-rewards
exercise: 39-process-rewards-and-search
---

# Process rewards and test-time search

Every method so far in this part changes the model's weights. Training is
expensive, though, and it happens once. There is a second place to spend
compute: at answer time. You can let the model try a hard problem many times
and keep the best try, the way a student works a few drafts before handing
one in.

This lesson builds the three standard ways to do that. Best-of-n draws several
full answers and keeps the one a scorer likes most. Voting draws several
answers and keeps the final answer that comes up most often, optionally
weighted by a scorer's confidence. Beam search goes further. It scores partial
answers step by step and spends its samples on the prefixes that look most
promising.

The scorer that beam search needs is a process reward model, which rates each
reasoning step instead of only the final answer. You met it in
`34-reward-models`. Here you see what a perfect one is: the value function of
the policy that wrote the steps. With that link in hand, test-time search
turns out to be planning, the same idea as Monte Carlo tree search in
`10-planning-and-learning`, run over a language model's reasoning.

## What you already know

- A language model is a policy $\pi(y \mid x)$ over responses $y$ to a prompt
  $x$, and it generates one token at a time (`33-language-models-as-policies`).
- An outcome reward model scores a whole response. A process reward model
  scores each step (`34-reward-models`).
- A verifiable reward, such as "the final answer is correct", is a Python rule
  that returns 0 or 1 (`38-reasoning-with-verifiable-rewards`).
- The value $V^\pi(s)$ of a state is the expected return from $s$ when you
  follow $\pi$, and it satisfies a backward Bellman recursion.
- Beam search and tree search pick which branches of a search tree to expand.

## Pick the best of n samples

Best-of-n (BoN) is the simplest search. Draw $n$ independent responses from
the policy, score each with a reward model, and return the highest scorer. It
needs no training, and it's the usual baseline that every smarter method must
beat.

How good is it? Suppose the scorer is perfect, so the returned reward is the
largest of $n$ independent draws. Sort the $M$ possible outcomes by reward,
$r_{(1)} \le r_{(2)} \le \dots \le r_{(M)}$, and let $F_i$ be the probability
that one draw lands at or below outcome $i$. The largest of $n$ draws is at or
below outcome $i$ exactly when all $n$ are, which has probability $F_i^n$. So:

$$
\mathbb{E}\big[\max(r_1, \dots, r_n)\big] = \sum_{i=1}^{M} r_{(i)} \big(F_i^{\,n} - F_{i-1}^{\,n}\big), \qquad F_0 = 0
$$

Read it as: the chance that the best of $n$ is exactly outcome $i$ is the
chance that all $n$ draws are at or below $i$, minus the chance that they are
all strictly below it. Weight each reward by that chance and add up. This is
the order-statistic formula. Two outcomes with the same reward are fine: their
terms telescope into one.

In numpy, it's a sort and a cumulative sum:

```python
order = np.argsort(rewards, kind="stable")
F = np.cumsum(probs[order])
F_prev = np.concatenate([[0.0], F[:-1]])
expected = np.sum(rewards[order] * (F**n - F_prev**n))
```

### Worked example

Take three outcomes with probabilities $[0.5, 0.3, 0.2]$ and rewards
$[0, 0.5, 1]$. With one draw, the expected reward is
$0.3 \times 0.5 + 0.2 \times 1 = 0.35$. With $n = 2$:

- The cumulative probabilities are $F = [0.5, 0.8, 1]$.
- The best of two is outcome 1 with probability $0.5^2 = 0.25$.
- It's outcome 2 with probability $0.8^2 - 0.5^2 = 0.39$.
- It's outcome 3 with probability $1 - 0.8^2 = 0.36$.
- The expected reward is $0.39 \times 0.5 + 0.36 \times 1 = 0.555$.

With a 0/1 reward, such as a unit test that passes or fails, the formula
collapses to $1 - (1 - p)^n$, where $p$ is the policy's success rate. Best-of-n
fails only when every one of the $n$ draws fails. On this lesson's toy task,
the base policy succeeds 8.6% of the time, and a perfect verifier lifts that
to 30.3% at $n = 4$, 76.4% at $n = 16$, and 99.7% at $n = 64$.

> [!KEY]
> Best-of-n's expected reward is $\sum_i r_{(i)}(F_i^n - F_{i-1}^n)$. With a
> perfect 0/1 verifier it's $1 - (1 - p)^n$, and the cost is $n$ full
> generations.

The real scorer is a learned reward model, not a perfect one. As $n$ grows,
best-of-n finds the responses the reward model overrates. `33-language-models-as-policies`
bounds how far best-of-n moves from the policy, and
`49-reward-hacking-and-overoptimization` measures when the true reward turns
down.

## Vote on the final answer

Many reasoning tasks end in a short final answer, such as a number. Then you
don't need a reward model at all. Self-consistency, from Wang and colleagues
(2022), samples $n$ reasoning chains and returns the final answer that appears
most often. Different chains that reach the same answer by different routes
reinforce each other. It's majority voting.

Weighted majority voting adds a scorer back. Each sampled chain $j$ gets a
weight $w_j$, usually a verifier's probability that the chain is correct. The
vote returns the answer with the largest total weight:

$$
\hat a = \operatorname*{arg\,max}_{a} \sum_{j=1}^{n} w_j\, \mathbf{1}[a_j = a]
$$

Read it as: for each candidate answer $a$, add up the weights of the chains
that ended in $a$, and pick the answer with the biggest pile. With every
$w_j = 1$, it's plain majority voting. Li and colleagues (2022) used a
verifier's scores this way, and Snell and colleagues (2024) call the same rule
"best-of-N weighted". In numpy, build one-hot answers and contract them with
the weights:

```python
onehot = np.eye(n_answers)[answers]                 # (B, n, n_answers)
totals = np.einsum("bn,bnk->bk", weights, onehot)
winner = np.argmax(totals, axis=1)                  # ties: smallest answer
```

### Worked example

Suppose one chain is correct with probability 0.6, and every wrong chain ends
on the same wrong answer. With $n = 5$, the vote is right when at least 3 of
the 5 chains are right:

$$
\sum_{k=3}^{5} \binom{5}{k} 0.6^k\, 0.4^{5 - k} = 0.3456 + 0.2592 + 0.07776 = 0.68256
$$

Read it as: the probability of exactly 3, 4, or 5 correct chains out of 5,
added up. Voting raised accuracy from 0.6 to 0.683.

Voting only helps when the correct answer is the single most likely one. Take
three answers with probabilities 0.55 for a wrong answer, 0.10 for a second
wrong answer, and 0.35 for the correct one. One sample is right 35% of the
time. The majority of 51 samples is right only 5.5% of the time, because the
vote converges on the most likely answer, which is wrong. Instead, weight each
chain by a verifier that gives correct chains 0.9 and wrong ones 0.3. The
correct answer's expected pile per sample is $0.35 \times 0.9 = 0.315$, and
the big wrong answer's is $0.55 \times 0.3 = 0.165$. Weighted voting over 51
samples is right 97.3% of the time.

> [!WARNING]
> More votes make majority voting more confident, not more correct. If the
> policy's most likely answer is wrong, accuracy falls toward zero as $n$
> grows.

## Score steps, not only answers

Best-of-n and voting judge finished responses. Search can do better by
judging partial ones, so it stops spending samples on a chain that already
went wrong. That needs a score for a prefix: how good is this half-written
solution?

The ideal answer is the prefix's value. Treat the prefix as a state $s_t$, the
prompt plus the first $t$ steps. Its value under the policy $\pi$ is the
probability that $\pi$, continuing from there, reaches a correct final answer:
$V^\pi(s_t) = \Pr_\pi(\text{correct} \mid s_t)$. Math-Shepherd, from Wang and
colleagues (2023), trains its process reward model (PRM) on exactly this. It
labels a step with the fraction of Monte Carlo continuations from that step
that reach the gold answer. That's a sample estimate of $V^\pi$. A PRM trained
this way is a learned critic.

### A toy with a known step value

In the exercise, a solution is $T = 4$ digits, each from 0 to 7, and it's
correct when the digits sum to a target such as 14. Each digit is one
reasoning step. The base policy picks digit $a$ at step $t$ with probability
$\pi_t(a)$, independently of earlier steps. The reward depends only on the
running sum, so the state is the pair (step $t$, partial sum $s$). The value
obeys a Bellman backup:

$$
V(t, s) = \sum_{a=0}^{7} \pi_t(a)\, V(t + 1, s + a), \qquad V(T, s) = \mathbf{1}[s = \text{target}]
$$

Read it as: at the last step, a sum is worth 1 if it hits the target and 0
otherwise. One step earlier, a partial sum is worth the average of what it
leads to, weighted by how likely the policy is to write each digit. Run it
backward from $t = T$ to $t = 0$, and $V(0, 0)$ is the policy's success rate.

```python
values[T, target] = 1.0
for t in reversed(range(T)):
    nxt = np.concatenate([values[t + 1], np.zeros(V)])   # sums past 28 are worth 0
    values[t] = np.stack([nxt[s:s + V] for s in range(S)]) @ step_probs[t]
```

Here's a two-step version small enough to do by hand. Digits are 0 or 1, the
target is 1, and the policy writes 1 with probability 0.3 at each step. At the
end, $V(2, 1) = 1$ and every other sum is worth 0. One step back,
$V(1, 0) = 0.3$, because from a sum of 0 you still need a 1. And
$V(1, 1) = 0.7$, because from a sum of 1 you need a 0. At the start,
$V(0, 0) = 0.7 \times 0.3 + 0.3 \times 0.7 = 0.42$.

## Search with a process reward model

Beam search keeps a small set of the most promising prefixes, called the
beam. At each step, it extends every prefix in the beam by every possible
next step, scores the extensions with the PRM, and keeps the best
$k$ of them. $k$ is the beam width. After $T$ steps, it returns the beam's
best complete solution.

In the two-step example, beam search with width 1 compares the first digit's
extensions: digit 0 leads to a state worth 0.3, and digit 1 to a state worth
0.7. It keeps digit 1. Then digit 0 leads to a sum of 1, worth 1, and digit 1
to a sum of 2, worth 0. It keeps digit 0 and ends on the target.

That isn't luck. With the exact value, width-1 search never reaches a dead
end. The Bellman backup says $V(s)$ is an average of its children's values, and
the largest of some numbers is at least their average. So the child that
search picks is worth at least as much as its parent. Start from a state with
$V > 0$, and every state along the path has $V > 0$. A complete solution's
value is its reward, 0 or 1, so the last one is 1.

> [!KEY]
> A perfect PRM is the value function of the policy that wrote the steps.
> Greedy search on it succeeds whenever success is possible, because the best
> child of a state is worth at least the state itself.

A real PRM is a noisy estimate of that value. In the exercise, you add
Gaussian noise with standard deviation 0.1 to the exact values and search
over 21 targets and 20 noise draws. The base policy succeeds about 5% of the
time on these targets. Width-1 search succeeds 76% of the time, and width 8
succeeds 92% of the time. A wider beam gives the search a way around a
misjudged step: a good prefix that the noise pushed to second place survives
to the next step.

On a real model, a step is a line of reasoning, not one of eight digits, so
you can't try every extension. Snell and colleagues (2024) sample instead.
They draw $N$ first steps, keep the top $N / M$ by PRM score, sample $M$
continuations from each, and repeat. Their lookahead variant rolls each
candidate a few steps further before scoring it. At the end, they pick among
the $N$ finished solutions with best-of-N weighted voting.

> [!RECAP]
> - Best-of-n's expected reward is $\sum_i r_{(i)}(F_i^n - F_{i-1}^n)$ over
>   outcomes sorted by reward, and $1 - (1 - p)^n$ with a 0/1 reward.
> - Majority voting returns the most frequent final answer. Weighted voting
>   returns the answer with the largest total verifier score.
> - A perfect PRM is $V^\pi$ of a prefix, computed backward with
>   $V(t, s) = \sum_a \pi_t(a) V(t + 1, s + a)$.
> - Beam search extends every beam by every token, scores the candidates, and
>   keeps the top $k$.

## Build it step by step

The checks test the functions in this order.

1. **`best_of_n_expected_reward(probs, rewards, n)`.** Takes `(M,)` arrays and
   an integer, and returns a float. Sort by reward with
   `np.argsort(..., kind="stable")`, take `np.cumsum` of the sorted
   probabilities, and prepend a zero for $F_0$. Common mistake: summing
   $r_{(i)} F_i^n$ without the difference, which counts each outcome many
   times.
2. **`majority_vote(answers, n_answers)`.** Takes `(B, n)` integers and
   returns `(B,)`. Count each answer per row, and take `np.argmax`, which
   returns the first maximum, so ties go to the smallest answer. You can call
   `weighted_majority_vote` with unit weights.
3. **`weighted_majority_vote(answers, weights, n_answers)`.** The same
   shapes, plus `(B, n)` weights. Use `np.eye(n_answers)[answers]` and
   `np.einsum`. Common mistake: summing weights over the batch axis instead of
   the samples axis.
4. **`step_values(step_probs, target)`.** Takes `(T, V)` probabilities and
   returns a `(T + 1, S)` table, where `S = 29`. Start from the one-hot row at
   $t = T$ and run the backup backward. Common mistake: reading past the end
   of the table when $s + a > 28$. Pad the next row with zeros.
5. **`prm_beam_search(prm, beam_width)`.** Starts from one empty prefix of
   shape `(1, 0)`. Each step builds candidates with `np.repeat` and `np.tile`,
   scores them with `prm`, and keeps the top `beam_width` with
   `np.argsort(-scores, kind="stable")`. Returns the beams best first, and
   their scores. Common mistake: keeping the top $k$ children of each beam
   separately, which is $k$ separate greedy searches and not a beam search.

## Cost and scaling

Search spends inference compute, and the costs differ in what they call. The
following table assumes solutions of $T$ steps, $n$ samples, beam width $k$,
and $b$ candidate extensions per beam at each step:

| Method | Policy generations | Scorer calls | Needs |
|---|---|---|---|
| Best-of-n | $n$ full solutions | $n$ | An outcome scorer |
| Majority voting | $n$ full solutions | 0 | An extractable final answer |
| Weighted voting | $n$ full solutions | $n$ | Both |
| PRM beam search | $k b$ steps per step, $T$ steps | $k b T$ | A process scorer |

Best-of-n and voting run the $n$ samples in parallel, so their latency is one
generation. Beam search must wait for each step's scores before it extends,
so its latency grows with $T$. Memory for all of them grows with $n$ or $kb$,
because the model keeps a key-value cache for every live prefix. On the toy,
$b = 8$, so width 8 costs $8 \times 8 \times 4 = 256$ PRM calls.

Snell and colleagues (2024) found that the best method depends on the
problem. Beam search beat best-of-N on harder questions and at small budgets.
Best-of-N caught up and passed it on easier questions at large budgets.
Choosing the method per question by estimated difficulty matched best-of-N
with up to four times less compute.

## Where this breaks

**The search exploits the PRM.** A PRM is a learned model with blind spots,
and search looks for them on purpose. Snell and colleagues saw beam search
produce repetitive, low-information steps and solutions only one or two steps
long, because the PRM scored them highly. People cap the search width,
aggregate step scores carefully, and fall back to best-of-n at large budgets.

**Aggregating step scores is a choice.** Lightman and colleagues (2023) scored
a solution by the product of step-correctness probabilities. Math-Shepherd
uses the minimum. Snell and colleagues found that the last step's score worked
best for them. Each choice changes which solutions win.

**Monte Carlo labels belong to one policy.** A Math-Shepherd label estimates
$V^\pi$ for the policy that did the rollouts. A step that a weak policy rarely
completes can be easy for a stronger one, so a PRM trained on the weak
policy's rollouts mislabels it.

**Steps are hard to define.** Tokens are too fine, and whole solutions are too
coarse. Most work splits on newlines. DeepSeek-R1 (2025) lists this, the
difficulty of labeling steps at scale, and reward hacking of a learned PRM as
reasons its PRM experiments didn't pay off.

**Voting needs one checkable answer.** Majority voting can't combine proofs,
essays, or code with different but equally correct outputs. It also fails
when the policy's most likely answer is wrong, as the worked example showed.

## Further reading

- [Wang and colleagues (2022), Self-Consistency Improves Chain of Thought
  Reasoning in Language Models](https://arxiv.org/abs/2203.11171). Majority
  voting over sampled reasoning chains, with large gains on arithmetic
  benchmarks.
- [Li and colleagues (2022), Making Large Language Models Better Reasoners
  with Step-Aware Verifier](https://arxiv.org/abs/2206.02336). DiVeRSe, which
  weights votes with a verifier that checks each step.
- [Lightman and colleagues (2023), Let's Verify Step by
  Step](https://arxiv.org/abs/2305.20050). PRM800K, human step labels, and
  process supervision beating outcome supervision for best-of-N reranking.
- [Wang and colleagues (2023), Math-Shepherd](https://arxiv.org/abs/2312.08935).
  PRM labels from Monte Carlo rollouts, which estimate the step's value.
- [Snell and colleagues (2024), Scaling LLM Test-Time Compute Optimally Can be
  More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314).
  Beam search, lookahead search, and best-of-N weighted voting compared across
  budgets and difficulty.
- [Wu and colleagues (2024), Inference Scaling Laws](https://arxiv.org/abs/2408.00724).
  Majority voting, weighted voting, and tree search under a fixed compute
  budget, where a small model with search can beat a larger one.

Researchers are still working on these open questions:

- How can a PRM stay accurate on the prefixes that search itself produces,
  which are unlike the prefixes it was trained on?
- When is compute better spent on more samples at answer time than on more
  reinforcement learning, and can one budget decide both?
- What should a step be for tasks without natural line breaks, such as code
  or long proofs?

## The exercise

You build every piece of test-time search on the digit-sum toy. Write
`best_of_n_expected_reward` from the order-statistic formula,
`majority_vote` and `weighted_majority_vote`, `step_values` for the exact
value of every (step, partial sum), and `prm_beam_search`, which takes any
PRM as a function of a batch of prefixes.

The checks compare best-of-n with a brute-force sum over every $n$-tuple,
with and without tied rewards, and with $1 - (1 - p)^n$. They compute voting
accuracy exactly, confirm 0.68256 for the example, and confirm that majority
voting falls below 0.1 on a wrong mode while weighted voting passes 0.95.
They compare `step_values` with an enumeration of all 4096 solutions,
test beam search's tie-breaking, and confirm that search on the exact value
solves every reachable target at width 1. The last check adds noise to the
PRM and expects width 8 to beat width 1 by at least 0.08.
