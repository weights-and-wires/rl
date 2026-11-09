---
title: Process rewards and test-time search
slug: 055-process-rewards-and-search
part: "Part 9 — RL for language models"
summary: Spend compute at answer time with best-of-n, weighted voting, and beam search steered by a process reward model that is a value function.
minutes: 40
gpu: false
objectives:
  - Compute best-of-n's expected reward exactly from the order statistics of the reward.
  - Implement majority and weighted majority voting, and show when voting on a wrong mode fails.
  - Compute the exact step-level value of a toy reasoning task with one backward pass.
  - Run PRM-guided beam search and explain why a perfect value never leads it to a dead end.
prereqs:
  - 022-monte-carlo-tree-search
  - 050-reward-models
  - 054-reasoning-with-verifiable-rewards
exercise: 055-process-rewards-and-search
---

# Process rewards and test-time search

Every method so far in this part changes the model's weights. Training is expensive, and it happens once. There is a second place to spend compute: at answer time. A student who works a hard problem several times and hands in the best draft does better than one who writes a single draft, and a model can do the same.

The question is how to spend those extra tries. You can draw whole answers and keep the best one, which needs a scorer. You can draw whole answers and keep the most common final answer, which needs no scorer but can be fooled. Or you can score half-finished answers and stop spending samples on a chain that has already gone wrong. That last idea needs a *process reward model* that rates each reasoning step, and this chapter shows what a perfect one is: the value function of the policy that wrote the steps.

## What you already know

- A language model is a policy $\pi(y \mid x)$ that generates one token at a time, from [Language models as policies](/lessons/049-language-models-as-policies).
- An outcome reward model scores a whole response, and a process reward model scores each step, from [Reward models from preferences](/lessons/050-reward-models).
- A verifiable reward is a Python rule that returns 0 or 1, from [Reasoning with verifiable rewards](/lessons/054-reasoning-with-verifiable-rewards).
- The value $V^\pi(s)$ of a state is the expected return from $s$ under $\pi$, and it obeys a backward Bellman recursion.
- Tree search expands the most promising branches first, from [Monte Carlo tree search](/lessons/022-monte-carlo-tree-search).

## Best of n: draw many, keep the best

*Best-of-n* draws $n$ independent responses, scores each, and returns the top one. It needs no training, and every smarter method must beat it.

How good is it with a perfect scorer? The returned reward is the largest of $n$ draws. Sort the $M$ outcomes by reward, $r_{(1)} \le \dots \le r_{(M)}$, and let $F_i$ be the chance one draw lands at or below outcome $i$. The largest of $n$ is at or below $i$ exactly when all $n$ are, which has probability $F_i^n$. So the chance that the best is exactly outcome $i$ is $F_i^n - F_{i-1}^n$, and

$$
\mathbb{E}\big[\max(r_1, \dots, r_n)\big] = \sum_{i=1}^{M} r_{(i)} \big(F_i^{\,n} - F_{i-1}^{\,n}\big), \qquad F_0 = 0
$$

Read it aloud: weight each reward by the chance it is the best, and add up. Take probabilities $[0.5, 0.3, 0.2]$ and rewards $[0, 0.5, 1]$. One draw gives $0.3 \times 0.5 + 0.2 \times 1 = 0.35$. For $n = 2$, $F = [0.5, 0.8, 1]$, so the best is outcome 2 with probability $0.8^2 - 0.5^2 = 0.39$ and outcome 3 with probability $1 - 0.8^2 = 0.36$. The expected reward is $0.39 \times 0.5 + 0.36 \times 1 = 0.555$. In numpy, it is a sort and a cumulative sum:

```python
order = np.argsort(rewards, kind="stable")
F = np.cumsum(probs[order])
F_prev = np.concatenate([[0.0], F[:-1]])
expected = np.sum(rewards[order] * (F**n - F_prev**n))
```

With a 0/1 reward the formula collapses to $1 - (1 - p)^n$, because best-of-n fails only when every draw fails. On this chapter's toy, the policy succeeds 8.6% of the time, and a perfect verifier lifts that to 30.3% at $n = 4$, 76.4% at $n = 16$, and 99.7% at $n = 64$. The price is $n$ full generations. A learned scorer is not perfect, and as $n$ grows best-of-n finds what it overrates, which [Reward hacking, safety, and the frontier](/lessons/061-reward-hacking-and-safe-rl) measures.

## Vote on the final answer

Many tasks end in a short final answer, such as a number, so you can drop the scorer. *Self-consistency* (Wang and colleagues, 2022) samples $n$ chains and returns the answer that appears most often. Chains that reach the same answer by different routes reinforce each other. If one chain is right with probability $0.6$ and every wrong chain lands on the same wrong answer, a vote over $n = 5$ is right when at least 3 chains are right: $0.3456 + 0.2592 + 0.07776 = 0.68256$. Voting lifted accuracy from $0.6$ to $0.683$.

Voting helps only when the correct answer is the single most likely one. Suppose the answers have probabilities $0.55$ and $0.10$ (both wrong) and $0.35$ (correct). One sample is right 35% of the time. A majority of 51 samples is right only 5.5% of the time, because the vote converges on the most likely answer, and that answer is wrong.

*Weighted voting* adds a scorer back. Chain $j$ gets a weight $w_j$, such as a verifier's probability that it is correct, and the vote returns the answer with the largest total weight:

$$
\hat a = \operatorname*{arg\,max}_{a} \sum_{j=1}^{n} w_j\, \mathbf{1}[a_j = a]
$$

Read it aloud: for each candidate answer, add up the weights of the chains that ended there, and pick the biggest pile. With every $w_j = 1$ it is plain majority voting. Give correct chains weight $0.9$ and wrong ones $0.3$. The correct answer's expected pile per sample is $0.35 \times 0.9 = 0.315$, and the big wrong answer's is $0.55 \times 0.3 = 0.165$, so the correct answer wins.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Probability that a vote returns the correct answer as n grows, when a wrong answer is the most likely one: majority voting falls toward zero and weighted voting rises to 0.97">

<line x1="64" y1="252.0" x2="580" y2="252.0" class="stroke-grid"/>
<text x="58" y="256.0" class="fig-small" text-anchor="end">0</text>
<line x1="64" y1="196.8" x2="580" y2="196.8" class="stroke-grid"/>
<text x="58" y="200.8" class="fig-small" text-anchor="end">0.25</text>
<line x1="64" y1="141.5" x2="580" y2="141.5" class="stroke-grid"/>
<text x="58" y="145.5" class="fig-small" text-anchor="end">0.5</text>
<line x1="64" y1="86.3" x2="580" y2="86.3" class="stroke-grid"/>
<text x="58" y="90.3" class="fig-small" text-anchor="end">0.75</text>
<line x1="64" y1="31.0" x2="580" y2="31.0" class="stroke-grid"/>
<text x="58" y="35.0" class="fig-small" text-anchor="end">1</text>
<line x1="64" y1="252" x2="580" y2="252" class="stroke-muted"/>
<line x1="64.0" y1="252" x2="64.0" y2="257" class="stroke-muted"/>
<text x="64.0" y="272" class="fig-small" text-anchor="middle">1</text>
<line x1="167.2" y1="252" x2="167.2" y2="257" class="stroke-muted"/>
<text x="167.2" y="272" class="fig-small" text-anchor="middle">11</text>
<line x1="270.4" y1="252" x2="270.4" y2="257" class="stroke-muted"/>
<text x="270.4" y="272" class="fig-small" text-anchor="middle">21</text>
<line x1="373.6" y1="252" x2="373.6" y2="257" class="stroke-muted"/>
<text x="373.6" y="272" class="fig-small" text-anchor="middle">31</text>
<line x1="476.8" y1="252" x2="476.8" y2="257" class="stroke-muted"/>
<text x="476.8" y="272" class="fig-small" text-anchor="middle">41</text>
<line x1="580.0" y1="252" x2="580.0" y2="257" class="stroke-muted"/>
<text x="580.0" y="272" class="fig-small" text-anchor="middle">51</text>
<polyline points="64.0,174.7 84.6,189.7 105.3,200.0 125.9,201.8 146.6,205.8 167.2,209.8 187.8,212.4 208.5,215.1 229.1,217.7 249.8,219.8 270.4,221.9 291.0,223.8 311.7,225.6 332.3,227.2 353.0,228.7 373.6,230.1 394.2,231.4 414.9,232.6 435.5,233.8 456.2,234.8 476.8,235.8 497.4,236.8 518.1,237.6 538.7,238.4 559.4,239.2 580.0,239.9" class="stroke-bad" fill="none"/>
<polyline points="64.0,174.7 84.6,91.7 105.3,92.1 125.9,80.9 146.6,91.5 167.2,74.2 187.8,62.9 208.5,62.4 229.1,56.3 249.8,55.8 270.4,51.7 291.0,50.8 311.7,48.1 332.3,46.9 353.0,45.1 373.6,43.9 394.2,42.7 414.9,41.6 435.5,40.7 456.2,39.7 476.8,39.0 497.4,38.2 518.1,37.8 538.7,37.7 559.4,37.5 580.0,37.0" class="stroke-c" fill="none"/>
<circle cx="64.0" cy="174.7" r="4" class="fill-ink"/>
<circle cx="580.0" cy="239.9" r="4" class="fill-bad"/>
<circle cx="580.0" cy="37.0" r="4" class="fill-c"/>
<text x="322" y="292" text-anchor="middle">n, the number of sampled chains</text>
<text x="58" y="14" text-anchor="end" class="fig-small">probability the vote is right</text>
<line x1="350" y1="28" x2="380" y2="28" class="stroke-bad"/>
<text x="388" y="32" class="fig-small">majority vote</text>
<line x1="350" y1="46" x2="380" y2="46" class="stroke-c"/>
<text x="388" y="50" class="fig-small">weighted vote</text>
</svg>
Figure 1. Exact accuracy of each vote as $n$ grows, on the example above. Notice that majority voting falls from 0.35 to 0.055 as it grows more confident, and weighted voting rises to 0.973.
```

```python
onehot = np.eye(n_answers)[answers]                 # (B, n, n_answers)
totals = np.einsum("bn,bnk->bk", weights, onehot)
winner = np.argmax(totals, axis=1)                  # ties: smallest answer
```

> [!WARNING]
> More votes make majority voting more confident, not more correct. If the policy's most likely answer is wrong, accuracy falls toward zero as $n$ grows.

## Score steps, not only answers

Best-of-n and voting judge finished responses, so a chain that went wrong at step 2 still costs all its tokens. If you could score a half-written solution, you could stop early. The ideal score of a prefix is its value. Treat the prefix as a state $s_t$ (the prompt plus the first $t$ steps). Its value under $\pi$ is the probability that $\pi$, continuing from there, reaches a correct final answer. Math-Shepherd (Wang and colleagues, 2023) trains a process reward model (PRM) on exactly this: it labels a step with the fraction of sampled continuations that reach the gold answer, a sample estimate of $V^\pi$. A PRM trained this way is a learned critic.

The exercise's toy has a known value. A solution is $T = 4$ digits, each from 0 to 7, and it is correct when the digits sum to a target such as 14. The policy writes digit $a$ at step $t$ with probability $\pi_t(a)$. The state is the pair (step $t$, partial sum $s$), and the value obeys a Bellman backup:

$$
V(t, s) = \sum_{a=0}^{7} \pi_t(a)\, V(t + 1, s + a), \qquad V(T, s) = \mathbf{1}[s = \text{target}]
$$

Read it aloud: at the end, a sum is worth 1 if it hits the target. One step earlier, a partial sum is worth the average of what it leads to, weighted by how likely the policy is to write each digit. Run it backward from $t = T$ to $t = 0$, and $V(0, 0)$ is the policy's success rate.

```python
values[T, target] = 1.0
for t in reversed(range(T)):
    nxt = np.concatenate([values[t + 1], np.zeros(V)])   # sums past 28 are worth 0
    values[t] = np.stack([nxt[s:s + V] for s in range(S)]) @ step_probs[t]
```

By hand, take two steps, digits 0 or 1, target 1, and the policy writes 1 with probability $0.3$ at each step. Then $V(2, 1) = 1$ and every other final sum is worth 0. One step back, $V(1, 0) = 0.3$ (you still need a 1) and $V(1, 1) = 0.7$ (you need a 0). At the start, $V(0, 0) = 0.7 \times 0.3 + 0.3 \times 0.7 = 0.42$.

## Search with the process reward model

*Beam search* keeps a small set of the most promising prefixes, the *beam*. At each step it extends every prefix by every possible next step, scores the extensions with the PRM, and keeps the best $k$, where $k$ is the beam width. After $T$ steps, it returns the best complete solution.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Two-step toy: the policy writes 1 with probability 0.3. Beam search of width 1 follows the child with the higher value, ending on the target">

<defs><marker id="c054-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<line x1="300" y1="66" x2="170" y2="134" class="stroke-muted"/><text x="227.0" y="96.0" class="fig-small" text-anchor="end">digit 0</text>
<line x1="300" y1="66" x2="430" y2="134" class="stroke-b" marker-end="url(#c054-arrow)"/><text x="373.0" y="96.0" class="fig-small" text-anchor="start">digit 1</text>
<line x1="430" y1="174" x2="330" y2="248" class="stroke-b" marker-end="url(#c054-arrow)"/><text x="372.0" y="207.0" class="fig-small" text-anchor="end">digit 0</text>
<line x1="430" y1="174" x2="530" y2="248" class="stroke-muted"/><text x="488.0" y="207.0" class="fig-small" text-anchor="start">digit 1</text>
<line x1="170" y1="174" x2="110" y2="248" class="stroke-muted"/><text x="132.0" y="207.0" class="fig-small" text-anchor="end">digit 0</text>
<line x1="170" y1="174" x2="230" y2="248" class="stroke-muted"/><text x="208.0" y="207.0" class="fig-small" text-anchor="start">digit 1</text>
<rect x="248" y="26" width="104" height="40" rx="6" class="fill-soft-a stroke-muted"/><text x="300" y="43" text-anchor="middle" class="fig-small fig-bold">start: sum 0</text><text x="300" y="58" text-anchor="middle" class="fig-small">V = 0.42</text>
<rect x="118" y="134" width="104" height="40" rx="6" class="fill-soft-a stroke-muted"/><text x="170" y="151" text-anchor="middle" class="fig-small fig-bold">sum 0</text><text x="170" y="166" text-anchor="middle" class="fig-small">V = 0.30</text>
<rect x="378" y="134" width="104" height="40" rx="6" class="fill-soft-b stroke-muted"/><text x="430" y="151" text-anchor="middle" class="fig-small fig-bold">sum 1</text><text x="430" y="166" text-anchor="middle" class="fig-small">V = 0.70</text>
<rect x="58" y="248" width="104" height="40" rx="6" class="fill-soft-a stroke-muted"/><text x="110" y="265" text-anchor="middle" class="fig-small fig-bold">sum 0</text><text x="110" y="280" text-anchor="middle" class="fig-small">reward 0</text>
<rect x="178" y="248" width="104" height="40" rx="6" class="fill-soft-a stroke-muted"/><text x="230" y="265" text-anchor="middle" class="fig-small fig-bold">sum 1</text><text x="230" y="280" text-anchor="middle" class="fig-small">reward 1</text>
<rect x="278" y="248" width="104" height="40" rx="6" class="fill-soft-c stroke-muted"/><text x="330" y="265" text-anchor="middle" class="fig-small fig-bold">sum 1</text><text x="330" y="280" text-anchor="middle" class="fig-small">reward 1</text>
<rect x="478" y="248" width="104" height="40" rx="6" class="fill-soft-a stroke-muted"/><text x="530" y="265" text-anchor="middle" class="fig-small fig-bold">sum 2</text><text x="530" y="280" text-anchor="middle" class="fig-small">reward 0</text>
</svg>
Figure 2. The two-step example. Width-1 beam search compares 0.30 with 0.70 and follows digit 1, then compares reward 1 with reward 0 and follows digit 0. Notice that it lands on the target without trying the other branch.
```

That is not luck. The Bellman backup says a state's value is an average of its children's values, and the largest of some numbers is at least their average. So the child that search picks is worth at least as much as its parent. Start from a state with $V > 0$, and every state along the path has $V > 0$, so the last one is worth 1.

> [!KEY]
> A perfect PRM is the value function of the policy that wrote the steps. Greedy search on it succeeds whenever success is possible, because the best child of a state is worth at least the state itself.

A real PRM is a noisy estimate. In the exercise, you add Gaussian noise with standard deviation $0.1$ to the exact values and search over 21 targets and 20 noise draws. The base policy succeeds about 5% of the time. Width-1 search succeeds 76% of the time, and width 8 succeeds 92%. A wider beam gives the search a way around a misjudged step: a good prefix that noise pushed to second place survives. On a real model a step is a line of reasoning, so you cannot try every extension, and Snell and colleagues (2024) sample $N$ first steps, keep the top $N/M$ by PRM score, sample $M$ continuations from each, and repeat.

> [!KEY]
> You can spend compute at answer time in three ways: draw many answers and keep the best, vote on the final answer, or score steps and search. The last works because a perfect step score is a value function, so planning over a model's reasoning is the same idea as tree search.

> [!RECAP]
> - Best-of-n's expected reward is $\sum_i r_{(i)}(F_i^n - F_{i-1}^n)$, and $1 - (1 - p)^n$ with a 0/1 reward.
> - Majority voting returns the most frequent final answer. Weighted voting returns the answer with the largest total verifier score.
> - A perfect PRM is $V^\pi$ of a prefix, computed backward with $V(t, s) = \sum_a \pi_t(a) V(t + 1, s + a)$.
> - Beam search extends every beam by every token, scores the candidates, and keeps the top $k$.

## Build it step by step

The checks test the functions in this order.

1. **`best_of_n_expected_reward(probs, rewards, n)`.** Takes `(M,)` arrays and an integer, and returns a float. Sort by reward with `np.argsort(..., kind="stable")`, take `np.cumsum`, and prepend a zero for $F_0$. Common mistake: summing $r_{(i)} F_i^n$ without the difference, which counts each outcome many times.
2. **`majority_vote(answers, n_answers)`.** Takes `(B, n)` integers and returns `(B,)`. Count each answer per row and take `np.argmax`, so ties go to the smallest answer.
3. **`weighted_majority_vote(answers, weights, n_answers)`.** Use `np.eye(n_answers)[answers]` and `np.einsum`. Common mistake: summing weights over the batch axis instead of the samples axis.
4. **`step_values(step_probs, target)`.** Takes `(T, V)` probabilities and returns a `(T + 1, S)` table with `S = 29`. Start from the one-hot row at $t = T$ and run the backup backward. Common mistake: reading past the end of the table when $s + a > 28$. Pad the next row with zeros.
5. **`prm_beam_search(prm, beam_width)`.** Start from one empty prefix of shape `(1, 0)`. Each step builds candidates with `np.repeat` and `np.tile`, scores them with `prm`, and keeps the top `beam_width` with `np.argsort(-scores, kind="stable")`. Common mistake: keeping the top $k$ children of each beam separately, which is $k$ separate greedy searches.

## Where this breaks

**The search exploits the PRM.** A PRM is a learned model with blind spots, and search looks for them on purpose. Snell and colleagues saw beam search produce repetitive, low-information steps that the PRM scored highly. People cap the search width and fall back to best-of-n at large budgets.

**Monte Carlo labels belong to one policy.** A Math-Shepherd label estimates $V^\pi$ for the policy that did the rollouts. A step a weak policy rarely completes can be easy for a stronger one, so a PRM trained on the weak policy's rollouts mislabels it.

**Voting needs one checkable answer.** Majority voting cannot combine proofs, essays, or code with different but equally correct outputs.

## Further reading

The following sources cover the methods in this chapter:

- [Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (2022)](https://arxiv.org/abs/2203.11171): majority voting over sampled reasoning chains.
- [Lightman et al., "Let's Verify Step by Step" (2023)](https://arxiv.org/abs/2305.20050): human step labels, and process supervision beating outcome supervision for reranking.
- [Wang et al., "Math-Shepherd" (2023)](https://arxiv.org/abs/2312.08935): PRM labels from Monte Carlo rollouts, which estimate the step's value.
- [Snell et al., "Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Model Parameters" (2024)](https://arxiv.org/abs/2408.03314): beam search, lookahead search, and best-of-N compared across budgets and difficulty.
- [Wu et al., "Inference Scaling Laws" (2024)](https://arxiv.org/abs/2408.00724): voting and tree search under a fixed compute budget.

Open questions that researchers are working on:

- How can a PRM stay accurate on the prefixes that search itself produces, which are unlike the prefixes it was trained on?
- When is compute better spent on more samples at answer time than on more reinforcement learning?
- What should a step be for tasks without natural line breaks, such as code or long proofs?

## The exercise

You build every piece of test-time search on the digit-sum toy. Write `best_of_n_expected_reward`, `majority_vote` and `weighted_majority_vote`, `step_values` for the exact value of every (step, partial sum), and `prm_beam_search`, which takes any PRM as a function of a batch of prefixes.

The checks compare best-of-n with a brute-force sum over every $n$-tuple and with $1 - (1 - p)^n$. They compute voting accuracy exactly, confirm 0.68256 for the example, and confirm that majority voting falls below 0.1 on a wrong mode while weighted voting passes 0.95. They compare `step_values` with an enumeration of all 4096 solutions, test beam search's tie-breaking, and confirm that search on the exact value solves every reachable target at width 1. The last check adds noise to the PRM and expects width 8 to beat width 1 by at least 0.08.
