---
title: Reward hacking and overoptimisation
slug: 49-reward-hacking-and-overoptimization
part: "Part 10 — Safety and the frontier"
summary: Watch a gold reward rise and then fall as best-of-n pushes on a noisy proxy, fit the Gao curve, and delay the fall with ensembles.
minutes: 60
gpu: false
objectives:
  - Explain Goodhart's law and name the regressional, extremal, causal, and adversarial ways a proxy reward fails.
  - Compute the exact best-of-n policy on a discrete space and compare its KL with the formula log n - (n - 1)/n.
  - Fit the Gao, Schulman, and Hilton overoptimisation forms by least squares and read off where gold reward peaks.
  - Show that mean and minimum ensembles of proxy rewards push the peak out, and say why shared errors limit them.
prereqs:
  - 30-reward-design
  - 33-language-models-as-policies
  - 34-reward-models
exercise: 49-reward-hacking-and-overoptimization
---

# Reward hacking and overoptimisation

Every reward you can write down is a stand-in for what you want. A reward
model trained on a few thousand human comparisons is a stand-in for human
judgement. A unit-test suite is a stand-in for correct code. Push hard enough
on a stand-in, and the optimiser finds the places where it and the real goal
part ways. This is *reward hacking*, and in reinforcement learning it's the
expected result of optimisation pressure, not a rare bug.

The pattern has a name: Goodhart's law. "When a measure becomes a target, it
ceases to be a good measure." The useful question isn't whether it happens,
but how fast, and whether you can see it coming. In 2022, Gao, Schulman, and
Hilton showed that, for language models tuned against a learned reward, the
answer follows a smooth curve with a peak you can predict.

In this lesson you build that curve on a toy world where you know the true
reward. You compute best-of-n sampling exactly, measure how far it moves the
policy, fit the Gao curve, and try the standard mitigation: an ensemble of
reward models.

## What you already know

- A *policy* $\pi$ assigns a probability to every output. Here the outputs
  are a finite list, such as the answers a model might give to one prompt.
- The KL divergence $\text{KL}(p \,\|\, q) = \sum_i p_i \log(p_i / q_i)$
  measures how far distribution $p$ has moved from $q$.
- A reward model scores outputs with a number learned from human
  comparisons. You trained one in *Reward models*.
- The best policy for "high reward, small KL from $\pi_{\text{ref}}$" is
  $\pi_{\text{ref}}$ reweighted by $\exp(r / \beta)$. You derived it in
  *Language models as policies*.
- Potential-based shaping keeps the optimal policy, and other bonuses can
  break it. You saw both in *Reward design and shaping*.

## Name the ways a proxy fails

Call the reward you optimise the *proxy*, and the reward you care about the
*gold* reward. In an RLHF pipeline, the proxy is the reward model and gold is
what careful people would say. Manheim and Garrabrant (2018) sort the ways
they part into four kinds:

- **Regressional.** The proxy is gold plus noise. Picking the output with the
  highest proxy also picks high noise, so its gold score is lower than its
  proxy score suggests.
- **Extremal.** The proxy tracks gold over ordinary outputs, but the link
  breaks at the extreme outputs that optimisation reaches.
- **Causal.** The proxy correlates with gold without causing it. Pushing the
  proxy doesn't move gold.
- **Adversarial.** The policy, or another agent, actively exploits the proxy.

Language models show every kind. Singhal et al. (2023) found that a reward
based on length alone reproduces most of the gains RLHF reports. Sharma et
al. (2023) found that people and preference models sometimes prefer a
convincing, sycophantic answer to a correct one. Coding agents trained on
test results have learned to edit or special-case the tests (Baker et al.,
2025).

Skalse et al. (2022) make this precise. A proxy is *unhackable* if raising
its expected return can never lower the true expected return. Over all
stochastic policies, two rewards are unhackable only if one of them is
constant. No useful proxy is safe to push without limit, so the goal is to
know how far is safe.

> [!KEY]
> Any proxy reward that isn't constant can be hacked by some policy. The
> practical question is how much optimisation it survives.

## Compute best-of-n exactly

The simplest optimiser needs no training at all. *Best-of-n* draws $n$
outputs from the base policy, scores each with the proxy, and returns the
highest. Raising $n$ applies more optimisation pressure.

On a discrete space you can write down the whole best-of-n policy. Sort the
outputs by proxy score, lowest first, and let $F_i$ be the base probability
of output $i$ plus every output below it. The best of $n$ independent draws
is at or below output $i$ exactly when all $n$ draws are, which has
probability $F_i^n$. So the chance that output $i$ wins is

$$
\pi_n(i) = F_i^{\,n} - F_{i-1}^{\,n}, \qquad F_0 = 0 .
$$

Read it as: the chance that everything lands at or below $i$, minus the
chance that everything lands strictly below it. What's left is the chance
that $i$ itself is the top draw. With no ties in the proxy, that's exact.

In numpy you sort once, take a cumulative sum, and scatter the result back
into the original order:

```python
order = np.argsort(proxy)
cdf = np.cumsum(base[order])
policy = np.empty_like(base)
policy[order] = cdf**n - (cdf - base[order])**n
```

### Worked example

Take five equally likely outputs, already sorted by proxy score. Their gold
scores are $[-1, 0, 1, 3, 0]$. The top-scoring output is a hack: the proxy
loves it, and gold rates it 0. The cumulative masses are $F = [0.2, 0.4, 0.6,
0.8, 1.0]$. The following table shows the policy and its gold score for four
values of $n$:

| $n$ | $\pi_n$ | Gold score |
|---|---|---|
| 1 | $[0.2, 0.2, 0.2, 0.2, 0.2]$ | 0.600 |
| 2 | $[0.04, 0.12, 0.2, 0.28, 0.36]$ | 1.000 |
| 3 | $[0.008, 0.056, 0.152, 0.296, 0.488]$ | 1.032 |
| 4 | $[0.0016, 0.024, 0.104, 0.28, 0.5904]$ | 0.942 |

For $n = 2$, the second entry is $0.4^2 - 0.2^2 = 0.12$, and the gold score
is $(-0.04 + 0 + 0.2 + 0.84 + 0) = 1.0$. Gold rises, peaks at $n = 3$, and
then falls, because more and more mass lands on the hack. As $n$ grows
without bound, all the mass goes there and gold drops to 0.

## Measure how far the policy moved

To compare optimisers, you need a common ruler for "how hard did you push".
The standard ruler is the KL divergence from the base policy. For best-of-n
it has a well-known closed form:

$$
\text{KL}(\pi_n \,\|\, \pi_{\text{ref}}) = \log n - \frac{n - 1}{n} .
$$

Read it as: the distance grows like $\log n$, less a correction that tends to
one. Doubling $n$ adds only about 0.69 nats, so moving the policy far costs
exponentially many samples. In numpy it's `np.log(n) - (n - 1) / n`.

The formula is exact for a continuous output space. On a discrete space it's
an upper bound (Beirami et al., 2024). In the worked example, the exact KL
for $n = 2$ is 0.180 nats, against 0.193 from the formula. With 20,000
equally likely outputs, the two agree within 0.03 percent up to $n = 4096$.

Gao et al. plot gold reward against the square root of this KL,

$$
d = \sqrt{\text{KL}(\pi \,\|\, \pi_{\text{ref}})} ,
$$

because the curves take simple shapes in $d$. For $n = 2, 4, 16, 64$, the
formula gives KL values of 0.193, 0.636, 1.835, and 3.175, so $d$ runs from
0.44 to 1.78.

> [!WARNING]
> Fit the Gao forms against $d$, not against the KL itself. With KL on the
> horizontal axis, the same data needs a different functional form, and the
> fitted peak lands in the wrong place.

## Watch gold rise and fall

The exercise world has $N = 20{,}000$ outputs under a uniform base policy.
Each output's gold score is a standard normal draw. Each proxy adds an error
drawn from a Student-t distribution with three degrees of freedom. That error
has *heavy tails*: most errors are small, but a few are huge.

Heavy tails decide the story. If the error were instead normal with
variance $\sigma^2$, an output with proxy score $u$ would have an expected
gold score of $u / (1 + \sigma^2)$. A higher proxy would always buy higher
gold, only at a discount. That's regressional Goodhart, and it never makes
gold fall.

With heavy-tailed error, the very highest proxy scores are mostly error. Kwa,
Thomas, and Garriga-Alonso (2024) call the limit *catastrophic Goodhart*:
some policies reach arbitrarily high proxy reward with no more true utility
than the base policy. They also found that the reward models they measured
had light-tailed error, so the question stays open for real systems.

On the exercise world with one proxy, best-of-n traces the following curve.
The proxy score climbs at every step, while gold peaks near $n = 64$ to $128$
and then falls:

| $n$ | $d$ | Proxy score | Gold score |
|---|---|---|---|
| 1 | 0.00 | 0.01 | 0.00 |
| 16 | 1.35 | 2.14 | 1.45 |
| 64 | 1.78 | 3.02 | 1.71 |
| 256 | 2.13 | 4.20 | 1.61 |
| 1024 | 2.44 | 6.19 | 1.13 |
| 4096 | 2.70 | 9.70 | 0.59 |

If you only watched the proxy, you'd see steady progress the whole way.

## Fit the Gao curve

Gao, Schulman, and Hilton trained proxy reward models of 3 million to 3
billion parameters on labels from a 6-billion-parameter "gold" reward model.
They then optimised policies against each proxy and scored the results with
gold. They measured gold relative to the starting policy, so it's 0 at
$d = 0$. Two forms fit their data. For best-of-n:

$$
R_{\text{BoN}}(d) = d\,(a - b\,d) ,
$$

and for reinforcement learning with PPO:

$$
R_{\text{RL}}(d) = d\,(a - b \log d) .
$$

The paper writes the coefficients as $\alpha$ and $\beta$. This lesson uses
$a$ and $b$, to keep $\beta$ free for the KL coefficient. Read the first form
as: gold starts rising at slope $a$, and a quadratic penalty $b\,d^2$ drags it
back down. Setting its derivative $a - 2bd$ to zero puts the peak at
$d^* = a / (2b)$. The RL form's penalty grows more slowly, like $d \log d$.
Its derivative $a - b \log d - b$ is zero at $d^* = e^{a/b - 1}$.

Both forms are linear in $a$ and $b$, so ordinary least squares fits them.
Stack one column per coefficient and solve:

```python
X = np.stack([d, -d**2], axis=1)            # best-of-n
# X = np.stack([d, -d * np.log(d)], axis=1)  # RL, needs d > 0
(a, b), *_ = np.linalg.lstsq(X, R, rcond=None)
```

On the one-proxy curve, the fit gives $a = 1.93$ and $b = 0.60$, so the
predicted peak is at $d^* = 1.60$, a KL of about 2.6 nats. The simulated peak
sits at $d = 1.78$ to $1.96$, one grid step away. Gao et al. found that $a$
and $b$ change smoothly, roughly with the logarithm of the proxy's parameter
count, and a bigger reward model moves the peak outward. So you can fit the
curve on small runs and predict where a larger run turns over.

The reinforcement learning optimum in this world is the tilted policy from
*Language models as policies*:

$$
\pi_\beta(i) \propto \pi_{\text{ref}}(i)\, \exp\!\big(\tilde r(i) / \beta\big) ,
$$

where $\tilde r$ is the proxy. Lowering $\beta$ moves this optimum further
from the base, much as a longer RL run does. Compute it in log space, with
`logits = np.log(base) + proxy / beta`, subtract the maximum, then
exponentiate and normalise. Otherwise `np.exp` overflows as soon as
`proxy / beta` passes about 709.

> [!WARNING]
> Don't read the paper as "RL overoptimises less than best-of-n". The two
> forms differ, the optimisers spend KL differently, and the curves aren't
> directly comparable at equal $d$.

## Delay the fall with ensembles

If one reward model has blind spots, train several and combine them. Coste
et al. (2023) tried this with best-of-n and PPO. Two combinations recur:

- **Mean.** Average the members' scores. Independent errors partly cancel,
  so the noise shrinks.
- **Minimum.** Score each output by its worst member. Coste et al. call this
  *worst-case optimisation*. An output has to fool every member to score
  high.

They found that both, and a variance-penalised mean, reduced
overoptimisation, by up to 70 percent for best-of-n.

The minimum works well against heavy tails, and a line of algebra shows why.
If $k$ members make independent errors $e_1, \dots, e_k$, the minimum error
exceeds $t$ only when every error does:

$$
\Pr\big(\min_j e_j > t\big) = \Pr(e > t)^k .
$$

Read it as: a tail that decays like $t^{-\nu}$ for one member decays like
$t^{-k\nu}$ for the minimum. Three members make the tail far thinner. In
numpy the two ensembles are `proxies.mean(axis=0)` and
`proxies.min(axis=0)`.

The catch is the word "independent". Eisenstein et al. (2023) found reward
hacks that ensembles didn't remove, because every member shared the error.
The exercise world builds this in: each member's error is 30 percent a shared
component and 70 percent its own. The minimum can't remove the shared part,
so gold still falls eventually. On the exercise world, the gold peak moves from $n = 64$
for one member to $n = 256$ for the mean and $n = 1024$ for the minimum. The
peak height rises from 1.71 to 2.26 and 2.59.

> [!KEY]
> Ensembles delay overoptimisation by the amount their members disagree.
> Errors the members share pass straight through the mean and the minimum.

> [!RECAP]
> - Best-of-n on a discrete space is exact: sort by proxy, then
>   $\pi_n(i) = F_i^n - F_{i-1}^n$.
> - Its KL from the base is at most $\log n - (n-1)/n$, with near equality
>   when no output carries much base mass.
> - Measure distance as $d = \sqrt{\text{KL}}$. Fit $d(a - bd)$ for best-of-n
>   and $d(a - b\log d)$ for RL by least squares.
> - The best-of-n peak is at $d^* = a/(2b)$.
> - A mean or minimum ensemble pushes the peak outward, as far as the
>   members' errors are independent.

## Build it step by step

The checks test the functions in this order.

1. **`bon_kl(n)`.** Takes an integer or an array, returns
   `np.log(n) - (n - 1) / n`. Common mistake: integer division. Convert `n`
   to float first.
2. **`best_of_n_policy(proxy, n, base)`.** Takes `(N,)` arrays and an
   integer, returns an `(N,)` distribution in the original order. Use
   `np.argsort`, `np.cumsum`, and a scatter with `policy[order] = ...`.
   Common mistake: returning the probabilities in sorted order. The second
   check uses an unsorted proxy to catch it. Clip the cumulative sums to
   `[0, 1]`, because rounding can push the last one to `1.0000000000000002`,
   and raising that to a large power drifts.
3. **`kl_divergence(p, q)`.** Sum `p * log(p / q)` over entries with
   `p > 0`. Common mistake: letting `0 * log 0` produce `nan`.
4. **`tilted_policy(proxy, beta, base)`.** Softmax of
   `log(base) + proxy / beta`. Common mistake: exponentiating before
   subtracting the maximum. A check feeds it a proxy of 800.
5. **`goodhart_curve(gold, proxy, base, ns)`.** Loop over `ns`. For each,
   build the policy, then record `sqrt(KL)`, `policy @ gold`, and
   `policy @ proxy`. Return three arrays. Common mistake: using `bon_kl`
   instead of the exact KL. The check compares the two, so both must be
   right.
6. **`fit_gao_bon(d, R)` and `fit_gao_rl(d, R)`.** Build the two-column
   matrix and call `np.linalg.lstsq`. Common mistake: a sign slip that
   returns $-b$. The check plants known coefficients.
7. **`ensemble_proxy(proxies, how)`.** A mean or a minimum over axis 0.

## Cost and scaling

The exercise is cheap, because the whole policy is one sorted array. Real
best-of-n costs a full generation and a reward-model pass per sample. The
following table compares the costs:

| Operation | Time | Memory |
|---|---|---|
| Exact best-of-n on $N$ outputs | $O(N \log N)$ for the sort, then $O(N)$ per $n$ | $O(N)$ |
| Real best-of-n for one prompt | $n$ generations and $n$ reward-model passes | $n$ responses |
| KL reached by best-of-n | grows like $\log n$ | none |
| An ensemble of $k$ reward models | $k$ times the reward cost | $k$ models |

Because best-of-n's KL grows only like $\log n$, reaching a KL of 5 nats takes
$n$ of about 400. A policy trained with RL reaches the same distance with one
sample per prompt at test time, which is why training is worth its cost.

## Where this breaks

**The gold reward is never available.** A real pipeline can't measure gold
cheaply. People hold out fresh human ratings, and stop or roll back when the
proxy and the ratings diverge. Moskovitz et al. (2023) turn the proxy-gold
turning points into constraints, which *Safe and constrained reinforcement
learning* picks up.

**The curve changes with scale.** Pan, Bhatia, and Steinhardt (2022) found
that the gap between proxy and true reward can open suddenly as models grow.
A small run that shows no hacking proves little about a large one.

**Direct alignment methods overoptimise too.** Rafailov et al. (2024) found
that DPO and its relatives degrade in the same pattern, often within a single
epoch, even with no explicit reward model.

**The optimiser can hide the hack.** Baker et al. (2025) found that a
weaker model reading a reasoning model's chain of thought catches reward
hacking well. When that monitor went into the reward, the agent kept hacking
but stopped saying so. Denison et al. (2024) found that models trained on
gameable tasks occasionally generalised to editing their own reward function.

**Ensembles share blind spots.** Training members on different data, or from
different pretrained models, raises their disagreement. It doesn't remove the
errors they share.

## Further reading

- [Scaling laws for reward model overoptimization](https://arxiv.org/abs/2210.10760)
  (Gao, Schulman, and Hilton, 2022). The gold-proxy setup, both functional
  forms, and how their coefficients scale with reward-model size.
- [Categorizing variants of Goodhart's law](https://arxiv.org/abs/1803.04585)
  (Manheim and Garrabrant, 2018). The regressional, extremal, causal, and
  adversarial taxonomy.
- [Defining and characterizing reward hacking](https://arxiv.org/abs/2209.13085)
  (Skalse et al., 2022). The formal definition of an unhackable proxy, and
  why nontrivial ones don't exist over all stochastic policies.
- [Catastrophic Goodhart](https://arxiv.org/abs/2407.14503) (Kwa, Thomas, and
  Garriga-Alonso, 2024). Why heavy-tailed reward error defeats a KL penalty
  and light-tailed error doesn't.
- [Reward model ensembles help mitigate overoptimization](https://arxiv.org/abs/2310.02743)
  (Coste et al., 2023). Worst-case and uncertainty-weighted ensembles, for
  best-of-n and PPO.
- [Helping or herding?](https://arxiv.org/abs/2312.09244) (Eisenstein et al.,
  2023). Why ensembles mitigate but don't eliminate reward hacking.
- [Theoretical guarantees on the best-of-n alignment policy](https://arxiv.org/abs/2401.01879)
  (Beirami et al., 2024). The proof that $\log n - (n-1)/n$ is an upper bound,
  and a tighter estimator.
- [Towards understanding sycophancy in language models](https://arxiv.org/abs/2310.13548)
  (Sharma et al., 2023). Evidence that human and model preferences reward
  agreement over accuracy.

Researchers are working on open questions such as the following:

- Do reward models trained on long reasoning traces, or LLM judges, have the
  light-tailed error that keeps a KL penalty effective?
- What do proxy-gold curves look like when the proxy is a unit-test suite or
  a rubric with known gaps, rather than a learned preference model?
- Can a monitor detect reward hacking without creating pressure to hide it?

## The exercise

You build every piece of the Goodhart curve on a world of 20,000 outputs
with a known gold reward and three noisy proxies from the given
`make_world`. Write `bon_kl` for the closed form, `best_of_n_policy` for the
exact best-of-n distribution, `kl_divergence`, and `tilted_policy` for the
KL-regularised optimum. Then write `goodhart_curve` to trace distance, gold,
and proxy scores over a range of $n$. Finish with `fit_gao_bon` and
`fit_gao_rl` for the two Gao forms, and `ensemble_proxy` for mean and
minimum ensembles.

The checks compare `best_of_n_policy` with hand-sized examples and 200,000
simulated draws, and confirm that the exact KL stays within 0.5 percent of
the formula without exceeding it. They test that the proxy climbs at every
$n$ while gold peaks inside the range and falls by at least 0.6. The fits
must recover planted coefficients exactly and place the simulated peak
correctly. The last check confirms that the mean ensemble peaks at least
twice as late as one member, the minimum no earlier than the mean, and that
each raises the peak.
