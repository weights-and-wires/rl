---
title: Language models as policies
slug: 33-language-models-as-policies
part: "Part 8 — Reinforcement learning for language models"
summary: Treat a language model as a policy, solve the KL-regularised objective in closed form, and measure how far best-of-n moves.
minutes: 60
gpu: false
objectives:
  - Score a whole response as the masked sum of its token log-probabilities.
  - Derive the optimum of reward minus beta times KL, and check it exactly on an enumerable toy.
  - Rebuild the same optimum one token at a time with soft backward induction, the maximum-entropy Bellman backup.
  - "Compute the exact best-of-n distribution and its KL, and say when log n - (n-1)/n is only a bound."
prereqs:
  - 15-policy-gradient-theorem
  - 22-maximum-entropy-rl
exercise: 33-language-models-as-policies
---

# Language models as policies

A language model writes one token at a time, and each token is a choice
sampled from a distribution that depends on everything written so far. That's
exactly what a policy does. Once you see it this way, policy gradients, value
functions, and the soft Bellman equations all apply.

The goal of post-training is to change what the model writes so that people,
or a program, score its answers higher. There's a catch. Push too hard on any
score and the model finds strange answers that score well and read badly. So
every method in this part of the course chases reward on a leash that keeps
the model close to where it started.

This lesson builds three ideas the rest of Part 8 leans on. You score whole
responses from token probabilities. You solve the leashed objective exactly,
for whole responses and token by token. And you measure how far best-of-n,
drawing several answers and keeping the best, moves the model.

## What you already know

- A *policy* $\pi(a \mid s)$ gives the probability of each action $a$ in state
  $s$.
- *The policy gradient theorem* improves a policy from sampled episodes, using
  the log-probabilities of the actions it took.
- Maximum-entropy RL, from *Maximum-entropy reinforcement learning*, adds a
  bonus for randomness. Its optimal policy is a Boltzmann distribution over soft
  action values, $\pi^*(a \mid s) \propto \exp(Q^*(s, a) / \alpha)$.
- The KL divergence $\mathrm{KL}(p \,\|\, q) = \sum_i p_i \log(p_i / q_i)$ is
  zero when two distributions match and positive otherwise.

## See generation as a sequence of actions

A *prompt* $x$ is the text the model is given. A *response*
$y = (y_0, y_1, \dots, y_{T-1})$ is the $T$ tokens it writes back. You can cast
this as reinforcement learning in two ways.

In the *token-level MDP*, the state is the prompt plus everything written so
far, $s_t = (x, y_{<t})$. The action is the next token, $a_t = y_t$, chosen
from a vocabulary of $V$ tokens. The transition is fixed: append the token. The
episode ends at an end-of-sequence token or a length limit. A reward model or
a checker scores only the finished response, so the reward is zero at every
token except the last, where $r_{T-1} = r(x, y)$. PPO with a value head, in
*RLHF with PPO*, uses this view.

In the *sequence-level bandit*, the whole response is one action. The context
is $x$ and the reward $r(x, y)$ arrives at once. *Group-relative policy
optimisation* leans on this view.

The two views connect through the chain rule of probability. The model's
probability of a whole response is the product of its token probabilities, so
its log-probability is a sum:

$$
\log \pi(y \mid x) = \sum_{t=0}^{T-1} m_t \log \pi(y_t \mid x, y_{<t})
$$

Read it as: for each position, take the log-probability the model gave the
token it actually wrote, and add them up. The *response mask* $m_t$ is 1 for
tokens that belong to the response and 0 for padding, which fills out a batch
of responses of different lengths. In numpy, with logits of shape `(B, T, V)`
and tokens of shape `(B, T)`:

```python
log_norm = logsumexp(logits, axis=-1)                          # (B, T)
chosen = np.take_along_axis(logits, tokens[..., None], axis=-1)[..., 0]
token_logps = chosen - log_norm                                # (B, T)
seq_logps = np.where(mask, token_logps, 0.0).sum(axis=-1)      # (B,)
```

### Worked example

Take a vocabulary of three tokens and a response of three tokens, padded to
length four. The model's logits at the four positions are $[2, 1, 0]$,
$[0, 0, 0]$, $[1, 3, 0]$, and $[5, 5, 5]$. The response is tokens 0, 2, and 1,
and the mask is $[1, 1, 1, 0]$.

- Position 0: $\pi = e^2 / (e^2 + e + 1) = 0.6652$, log-probability $-0.4076$.
- Position 1: all logits equal, $\pi = 1/3$, log-probability $-1.0986$.
- Position 2: $\pi = e^3 / (e + e^3 + 1) = 0.8438$, log-probability $-0.1698$.
- Position 3 is padding, so it's left out whatever it holds.

The sum is $\log \pi(y \mid x) = -1.6761$, so the whole response has
probability $0.1871$.

> [!WARNING]
> Multiplying by the mask doesn't remove padding. A padded position can hold
> $-\infty$, and $0 \times (-\infty)$ is `nan` in floating point. Select with
> `np.where(mask, token_logps, 0.0)` instead.

## Put the policy on a leash

The *reference policy* $\pi_{\text{ref}}$ is the model before this round of
training, usually the model after supervised fine-tuning. The goal is to raise
the expected reward while staying close to it. The standard objective, for one
prompt, is:

$$
J(\pi) = \mathbb{E}_{y \sim \pi}\big[r(x, y)\big] - \beta\, \mathrm{KL}\big(\pi(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\big)
$$

Read it as: the average reward of the responses $\pi$ writes, minus $\beta$
times how far $\pi$ has moved from the reference. The *KL coefficient*
$\beta > 0$ is the length of the leash. The KL runs from $\pi$ to
$\pi_{\text{ref}}$, the direction called the *reverse KL*. It averages over
responses the trained model writes, so it charges heavily for any response the
reference would almost never write.

For a small set of responses you can write $J$ directly:

```python
kl = np.sum(np.where(p > 0, p * (np.log(p) - ref_logps), 0.0))
J = np.sum(p * rewards) - beta * kl
```

A real model has far too many responses to list. By the same chain rule, the
sequence KL is the expected sum, over the tokens $\pi$ writes, of
$\log \pi(y_t \mid s_t) - \log \pi_{\text{ref}}(y_t \mid s_t)$. *RLHF with
PPO* estimates it that way, from samples.

## Solve the leash in closed form

The objective has an exact solution. The best policy starts from the reference
and multiplies each response's probability by a boost that grows with its
reward:

$$
\pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x)\, \exp\!\Big(\frac{r(x, y)}{\beta}\Big),
\qquad
Z(x) = \sum_{y} \pi_{\text{ref}}(y \mid x) \exp\!\Big(\frac{r(x, y)}{\beta}\Big)
$$

Read it as: take the reference probability of each response, multiply it by
$\exp(r / \beta)$, and divide by the total $Z(x)$ so the result sums to one.
$Z(x)$ is called the *partition function*. In log space it's a softmax:

```python
z = ref_logps + rewards / beta
pi_star = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
```

Here's why it's the best. Expand $\mathrm{KL}(\pi \,\|\, \pi^*)$ with the
formula for $\pi^*$, and collect terms. You get the identity
$J(\pi) = \beta \log Z(x) - \beta\, \mathrm{KL}(\pi \,\|\, \pi^*)$. The first
term doesn't depend on $\pi$, and the KL is zero only at $\pi = \pi^*$. So
$\pi^*$ is the unique maximiser, and the best value is exactly
$J(\pi^*) = \beta \log Z(x)$.

When $\beta$ is huge, $r / \beta$ is near zero and $\pi^* = \pi_{\text{ref}}$.
When $\beta$ is tiny, the best response's boost swamps every other.

This is the maximum-entropy result from *Maximum-entropy reinforcement
learning* in a different coat. There, the optimal policy was
$\exp(Q / \alpha)$, normalised. Here, $\beta$ plays the part of the temperature
$\alpha$, and $\pi_{\text{ref}}$ replaces the uniform distribution as the
prior. With a uniform reference, $-\beta\, \mathrm{KL}(\pi \,\|\, \text{uniform})$
is $\beta$ times the entropy of $\pi$ minus a constant, so the two objectives
are the same. Korbak, Perez, and Buckley (2022) read the formula as Bayesian
inference, with $\pi_{\text{ref}}$ as the prior and $\exp(r / \beta)$ as the
evidence.

> [!KEY]
> The KL-regularised optimum is $\pi^* \propto \pi_{\text{ref}} \exp(r / \beta)$,
> and its value is $\beta \log Z$. It's the Boltzmann policy of maximum-entropy
> RL, with $\beta$ as the temperature and the reference as the prior.

### Worked example

Take three responses with $\pi_{\text{ref}} = (0.5, 0.3, 0.2)$, rewards
$r = (1, 0, 2)$, and $\beta = 1$:

- Boosted weights: $0.5 e^1 = 1.3591$, $0.3 e^0 = 0.3$, and $0.2 e^2 = 1.4778$.
- The total is $Z = 3.1370$, so $\pi^* = (0.4333, 0.0956, 0.4711)$.
- Expected reward under $\pi^*$ is $1.3755$, and
  $\mathrm{KL}(\pi^* \,\|\, \pi_{\text{ref}}) = 0.2322$.
- $J(\pi^*) = 1.3755 - 0.2322 = 1.1433$, which equals $\log Z = 1.1433$.

The reference itself scores only $J(\pi_{\text{ref}}) = 0.9$. A shorter leash,
$\beta = 0.5$, gives $\pi^* = (0.2477, 0.0201, 0.7322)$ at a KL of $0.7218$.

A real model can't compute $Z(x)$, because it sums over every possible
response. *Direct preference optimisation* cancels it, and PPO and GRPO sample
instead.

## Solve it one token at a time

The closed form treats the response as one action. You can also solve the same
problem in the token-level MDP, with the soft Bellman backup from
*Maximum-entropy reinforcement learning*. The reference supplies the prior at
every token, and the only reward is on the last token.

Walk backwards from the end. A finished response's soft value is its reward.
At every earlier prefix, token $a$'s soft action value is the soft value of the
prefix it leads to, and the prefix's soft value is a log-sum-exp over the next
token:

$$
\begin{aligned}
Q(s_t, a) &= V(s_{t+1}) \\
V(s_t) &= \beta \log \sum_{a} \pi_{\text{ref}}(a \mid s_t) \exp\!\Big(\frac{Q(s_t, a)}{\beta}\Big) \\
\pi^*(a \mid s_t) &= \pi_{\text{ref}}(a \mid s_t) \exp\!\Big(\frac{Q(s_t, a) - V(s_t)}{\beta}\Big)
\end{aligned}
$$

Read it as: a prefix's value is a soft maximum over the next token, weighted by
the reference. The policy there is the reference, reweighted by how much better
each token is than the prefix's value. Here $s_{t+1}$ is $s_t$ with $a$
appended, and $V(s_T) = r(x, y)$.

Multiply the token probabilities along a response and the values telescope.
Each $V(s_{t+1})$ that one factor adds, the next factor subtracts. What's left
is $\pi_{\text{ref}}(y \mid x) \exp\big((r(x, y) - V(s_0)) / \beta\big)$. That's
the closed form, with $V(s_0) = \beta \log Z(x)$. Rafailov and colleagues
(2024) build on this fact: the log-ratio
$\beta \log(\pi^* / \pi_{\text{ref}})$ of each token is the soft advantage
$Q(s_t, a) - V(s_t)$.

Store the prefixes as a tree of arrays. Entry $t$ has shape `(V,) * (t + 1)`:
the leading axes pick the prefix and the last axis picks the next token. Then
each step of the backward pass is one line of numpy:

```python
z = ref_cond_logps[t] + v_next / beta     # shape (V,) * (t + 1)
v = beta * logsumexp(z, axis=-1)          # shape (V,) * t
```

The state must be the whole prefix. The toy reference in the exercise depends
only on the previous token. The best policy generally doesn't, once the reward
isn't a sum of per-token scores.

### Worked example

Take two tokens from a vocabulary of two, with $\beta = 1$. The reference
writes $y_0 \in \{0, 1\}$ with probability $0.5$ each. After a 0 it writes
$(0.8, 0.2)$, and after a 1 it writes $(0.4, 0.6)$. Only the response $(0, 1)$
earns a reward, of 1.

- After $y_0 = 0$: $V = \log(0.8 e^0 + 0.2 e^1) = \log 1.3437 = 0.2954$, and
  $\pi^*(\cdot \mid 0) = (0.8, 0.2 e) / 1.3437 = (0.5954, 0.4046)$.
- After $y_0 = 1$: no reward is reachable, so $V = 0$ and the policy stays at
  $(0.4, 0.6)$.
- At the start: $V(s_0) = \log(0.5 e^{0.2954} + 0.5 e^{0}) = \log 1.1718 = 0.1586$,
  and $\pi^*(y_0) = (0.6718, 0.5) / 1.1718 = (0.5733, 0.4267)$.

The response $(0, 1)$ gets $0.5733 \times 0.4046 = 0.2320$. The closed form
agrees: $0.5 \times 0.2 \times e / Z$ with $Z = 1.1718$ is $0.2320$.

> [!KEY]
> The token-level soft Bellman backup and the sequence-level closed form give
> the same policy. The reward sits on the last token, and the values telescope
> to $\beta \log Z$ at the start.

## Measure best-of-n

*Best-of-n* sampling draws $n$ responses from $\pi_{\text{ref}}$ and returns
the one with the highest reward. It needs no training, and Gao, Schulman, and
Hilton (2022) used it to measure how far a reward model can be pushed before it
breaks. Sort the responses by reward, and let $F_i$ be the reference
probability of the $i$-th response and every response below it. The best of
$n$ draws is at or below response $i$ exactly when all $n$ draws are, which
happens with probability $F_i^n$. So:

$$
\pi_{\text{BoN}}(y_{(i)}) = F_i^{\,n} - F_{i-1}^{\,n}
$$

Read it as: the chance the best draw is at or below response $i$, minus the
chance it's at or below the one before. Here $y_{(i)}$ is the $i$-th response in
reward order, $F_0 = 0$, and rewards are distinct. In numpy:

```python
order = np.argsort(rewards)
F = np.cumsum(ref_probs[order])
out[order] = F**n - np.concatenate([[0.0], F[:-1]])**n
```

### How far best-of-n moves

A widely quoted formula says the KL of best-of-n from the reference is
$\log n - (n - 1)/n$. It's exact when the reward is continuous, so no single
response has probability of its own. Then
$\pi_{\text{BoN}} / \pi_{\text{ref}} = n F^{n-1}$, where $F$ is the reward's
cumulative distribution under the reference. $F$ of the chosen response is the
largest of $n$ uniform numbers, whose expected log is $-1/n$, so the KL is
$\log n - (n - 1)/n$.

On a finite set of responses, Beirami and colleagues (2024) showed that the
formula is only an upper bound. The gap is tiny when the reference spreads over
many responses of small probability. It's large when a response near the top of
the ranking holds a lot of probability.

> [!KEY]
> $\log n - (n-1)/n$ is the exact best-of-n KL for a continuous reward and an
> upper bound on any finite set. It's loose when a few likely responses sit near
> the top of the ranking.

### Worked example

Take four responses with $\pi_{\text{ref}} = (0.1, 0.4, 0.2, 0.3)$, rewards
$(2, -1, 0.5, 1)$, and $n = 2$:

- In reward order, the responses are the second, third, fourth, and first, and
  the cumulative mass is $F = (0.4, 0.6, 0.9, 1.0)$.
- $\pi_{\text{BoN}}$ in reward order is $0.4^2 = 0.16$,
  $0.6^2 - 0.4^2 = 0.20$, $0.9^2 - 0.6^2 = 0.45$, and $1 - 0.9^2 = 0.19$.
- In the original order, $\pi_{\text{BoN}} = (0.19, 0.16, 0.20, 0.45)$, and
  expected reward rises from $0.2$ to $0.77$.
- The exact KL is $0.1578$. The formula gives $\log 2 - 1/2 = 0.1931$.

With $n = 4$ and a uniform reference over distinct rewards, the exact KL is
$0.5903$ for 4 responses, $0.6334$ for 16, and $0.63629$ for 1000, against a
bound of $0.63629$. Put half the reference mass on the single best response,
and at $n = 16$ the exact KL is $0.6930$, against a bound of $1.8351$.

> [!RECAP]
> - A response's log-probability is the sum of its token log-probabilities,
>   selected with `np.where` under the mask.
> - $\pi^* = \operatorname{softmax}(\log \pi_{\text{ref}} + r / \beta)$, and
>   $J(\pi^*) = \beta \log Z$.
> - Soft backward induction over the prefix tree gives the same $\pi^*$ token by
>   token, with $V(s_T) = r$ and $V(s_0) = \beta \log Z$.
> - Best-of-n gives response $i$ in reward order $F_i^n - F_{i-1}^n$.
> - Its KL is at most $\log n - (n-1)/n$, with equality only in the continuous
>   limit.

## Build it step by step

The checks test the functions in this order. The starter provides `ToyLM`, a
tabular policy whose logits depend on the position and the previous token, and
`all_sequences`, which lists every response of a given length.

1. **`token_logprobs(logits, tokens)`.** Takes `(B, T, V)` logits and `(B, T)`
   tokens, and returns `(B, T)`. Subtract the log-sum-exp over the last axis,
   then gather with `np.take_along_axis`. Common mistake: exponentiating without
   subtracting the maximum first. The check adds 1000 to every logit.
2. **`sequence_logprob(token_logps, mask)`.** Returns `(B,)`. Use `np.where`,
   then sum over the last axis. Common mistake: multiplying by the mask. The
   third check then scores all $4^3 = 64$ responses of a toy model and expects
   their probabilities to sum to one.
3. **`kl_optimal_policy(ref_logps, rewards, beta)`.** Returns `(N,)`, a softmax
   of `ref_logps + rewards / beta`. Common mistake: skipping the max
   subtraction, which overflows at small `beta`.
4. **`kl_regularised_objective(probs, ref_logps, rewards, beta)`.** Returns a
   float. Common mistake: letting a zero probability produce `nan`. The check
   scores a point mass.
5. **`soft_token_policy(ref_cond_logps, rewards, beta)`.** Loop `t` from
   `T - 1` down to 0, starting from `v_next = rewards`, and store
   `z - v[..., None] / beta` as the log-policy at each step. Return the list and
   the final `v` as a float. Common mistake: dropping the `[..., None]` that
   lines up the prefix axes.
6. **`best_of_n_probs(ref_probs, rewards, n)`.** Returns `(N,)` in the input
   order. Common mistake: returning it in sorted order.
7. **`best_of_n_kl(ref_probs, rewards, n)`.** Returns a float. Skip entries with
   zero best-of-n probability.

## Cost and scaling

Scoring responses is cheap, and solving for $\pi^*$ exactly is not. The
following table gives the costs for $B$ responses of length $T$ over $V$
tokens:

| Computation | Time | What it needs |
|---|---|---|
| Token and sequence log-probs | $O(BTV)$, one forward pass | Only the sampled responses |
| Closed-form $\pi^*$ | $O(V^T)$ | Every possible response |
| Soft backward induction | $O(V^T)$ | Every prefix, $V + V^2 + \dots + V^T$ of them |
| Exact best-of-n | $O(N \log N)$ for a sort | A finite list of $N$ responses |
| Best-of-n by sampling | $n$ generations per prompt | A reward for each |

For Qwen3-0.6B, $V$ is about 150,000, so even two-token responses number more
than $10^{10}$. The exact answers here are for checking methods, not for
training. Best-of-n moves compute from training to inference: every answer
costs $n$ generations.

## Where this breaks

**A policy that can't represent $\pi^*$.** The closed form ranges over every
distribution on responses. A real network reaches, at best, the nearest policy
it can represent. The exercise's toy shows it: its logits read only the previous
token, but $\pi^*$ for a reward that isn't a sum of per-token scores reads the
whole prefix.

**The wrong KL direction.** Swap the arguments to
$\mathrm{KL}(\pi_{\text{ref}} \,\|\, \pi)$ and the leash charges for failing to
cover responses the reference writes, instead of for writing ones it wouldn't.
The optimum changes. Test the direction with a two-response example.

**A reward that's wrong somewhere.** If the reward overrates some odd response,
$\pi^*$ boosts it by $\exp(r / \beta)$, and a small $\beta$ makes that boost
enormous. *Reward models* and *Reward hacking and overoptimisation* return to
this.

**Quoting the best-of-n formula as exact.** Language models often repeat the
same short answer, and all the repeats act as one response with a large
probability. The formula then overstates best-of-n's KL. Estimate it from the
distinct responses, or use the estimator Beirami and colleagues propose.

## Further reading

- [Ziegler et al. (2019), Fine-tuning language models from human preferences](https://arxiv.org/abs/1909.08593)
  sets up the reward-minus-KL objective for a language model.
- [Korbak, Perez, and Buckley (2022), RL with KL penalties is better viewed as Bayesian inference](https://arxiv.org/abs/2205.11275)
  reads $\pi^*$ as a Bayesian posterior.
- [Rafailov et al. (2023), Direct preference optimization](https://arxiv.org/abs/2305.18290)
  derives the closed-form optimum in its appendix, then inverts it to train
  without sampling.
- [Rafailov, Hejna, Park, and Finn (2024), From r to Q*: your language model is secretly a Q-function](https://arxiv.org/abs/2404.12358)
  works out the token-level soft Bellman view of the same optimum.
- [Gao, Schulman, and Hilton (2022), Scaling laws for reward model overoptimization](https://arxiv.org/abs/2210.10760)
  measures how a proxy reward and a true reward come apart under best-of-n.
- [Beirami et al. (2024), Theoretical guarantees on the best-of-n alignment policy](https://arxiv.org/abs/2401.01879)
  proves that $\log n - (n-1)/n$ is an upper bound and gives a tighter
  estimator.

Open questions researchers are working on:

- How close can a trained network get to $\pi^*$, and how much of the gap is
  optimisation rather than architecture?
- Is reverse KL to a fixed reference the right leash, or should the anchor move
  as training goes on?
- How should you measure the KL of a policy that repeats itself, when
  sample-based estimates are noisy and the closed forms don't apply?

## The exercise

You build the scoring and both exact solutions from this lesson:
`token_logprobs`, `sequence_logprob`, `kl_optimal_policy`,
`kl_regularised_objective`, `soft_token_policy`, `best_of_n_probs`, and
`best_of_n_kl`.

The checks compare log-probabilities with the worked example, with logits
shifted by 1000, and with padding that holds $-\infty$ and `nan`. They check
that all 64 toy responses sum to probability one, that $\pi^*$ matches the
worked example and both limits of $\beta$, and that $J(\pi^*) = \beta \log Z$
beats 20 perturbed policies. They check that your token-level policy multiplies
out to the closed form to $10^{-10}$ on a reward that isn't a sum of per-token
scores. Finally, they check best-of-n against 100,000 sampled draws, and its KL
against $\log n - (n-1)/n$ for every $n$ up to 64, for a spread reference and a
peaked one.
