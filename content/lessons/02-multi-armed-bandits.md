---
title: Multi-armed bandits
slug: 02-multi-armed-bandits
part: "Part 1 — The reinforcement learning problem"
summary: Choose between options with unknown payoffs, and measure which strategies waste only logarithmically many tries.
minutes: 55
gpu: false
objectives:
  - Keep a running mean with one line, and say what a constant step size changes.
  - Implement epsilon-greedy, UCB1, and Thompson sampling for Bernoulli arms.
  - Define regret, and tell logarithmic regret growth from linear growth on a curve.
prereqs:
  - 01-the-rl-problem
exercise: 02-multi-armed-bandits
---

# Multi-armed bandits

Picture a row of slot machines, each paying out at its own unknown rate. You
have a thousand coins. Every coin you spend on a poor machine is a coin not
spent on the best one. But you only learn which machine is best by spending
coins on them. That tension, between using what you know and finding out
more, is called the *exploration-exploitation trade-off*.

A *multi-armed bandit* is this problem with the slot machines called *arms*.
It is reinforcement learning with the states taken away. Your action changes
the reward you get, but nothing else. That makes it the cleanest place to
study exploration, which every later method in the course has to handle
somehow.

In this lesson, you build three strategies and a way to score them. You will
see that the obvious strategy wastes a fixed share of its pulls forever, and
two cleverer ones waste a share that shrinks toward zero.

## What you already know

- An agent picks an action, and the environment returns a reward.
- A policy is the agent's rule for picking actions. It can be random.
- The mean of a sample estimates an expected value, and its standard error
  shrinks like one over the square root of the sample size.
- `np.random.default_rng(seed)` gives a reproducible random number generator.

## The bandit problem

A bandit has $k$ arms, numbered $0$ to $k-1$. At each step $t$ you pick an arm
$a_t$ and receive a reward $r_t$. Each arm $a$ has a fixed but unknown mean
reward $\mu_a$. In this lesson, arms are *Bernoulli*: arm $a$ pays 1 with
probability $\mu_a$ and 0 otherwise, like a coin with its own bias.

The best arm's mean is $\mu^* = \max_a \mu_a$. The *gap* of arm $a$ is
$\Delta_a = \mu^* - \mu_a$, the reward you give up, on average, each time you
pull it. The best arm has gap 0.

## Estimate each arm with a running mean

Start with the easy half: estimating how good each arm is. The natural
estimate $Q(a)$ is the average of the rewards the arm has paid so far. You
don't need to store every reward. When the $n$-th reward $r$ arrives for an arm
whose estimate after $n-1$ rewards is $Q_{n}$, update it in place:

$$
Q_{n+1} = Q_n + \frac{1}{n}\left(r - Q_n\right)
$$

Read it as: move the old estimate toward the new reward by a fraction $1/n$ of
the gap between them. The first reward replaces the estimate entirely, since
$1/1 = 1$. The hundredth reward moves it by one hundredth of its surprise. The
result is exactly the sample mean, with no stored history.

In numpy, with `n` the number of rewards already averaged in:

```python
q = q + (reward - q) / (n + 1)
```

For rewards $1, 0, 1, 1$ starting from $Q = 0$:

| Reward number | Reward | Step size | New estimate |
|---|---|---|---|
| 1 | 1 | 1 | $0 + 1 \cdot (1 - 0) = 1$ |
| 2 | 0 | 1/2 | $1 + \tfrac12 (0 - 1) = 0.5$ |
| 3 | 1 | 1/3 | $0.5 + \tfrac13 (1 - 0.5) = 0.6667$ |
| 4 | 1 | 1/4 | $0.6667 + \tfrac14 (1 - 0.6667) = 0.75$ |

The last estimate is $3/4$, the plain average of the four rewards.

### A constant step size

Replace $1/n$ by a fixed step size $\alpha$ between 0 and 1, and the estimate
becomes a weighted average that favours recent rewards. Unrolling the update
from a starting value $Q_1$ gives:

$$
Q_{n+1} = (1 - \alpha)^n Q_1 + \sum_{i=1}^{n} \alpha (1 - \alpha)^{n-i} r_i
$$

Read it as: the reward from $n - i$ steps ago carries weight
$\alpha (1-\alpha)^{n-i}$, which shrinks by a factor $1 - \alpha$ for every
step it ages. The starting value keeps weight $(1-\alpha)^n$. All the weights
add up to exactly one. With $\alpha = 0.5$ and the same rewards, the estimates
are $0.5$, $0.25$, $0.625$, and $0.8125$. The last one is
$0.5 \times (0.125 + 0 + 0.5 + 1) = 0.8125$.

A constant step never settles. It keeps chasing the latest rewards, which is
a flaw when the arm's mean is fixed and a feature when the mean drifts. The
steps $1/n$ do settle on the true mean. Robbins and Monro (1951) give a
sufficient condition: the step sizes must add up to infinity, while their
squares add up to a finite number.

## Score a strategy with regret

The *regret* after $T$ pulls is the reward you lost, in expectation, by not
pulling the best arm every time:

$$
\text{Regret}(T) = \sum_{t=0}^{T-1} \Delta_{a_t}
$$

Read it as: every pull costs the gap of the arm you pulled. Pulling the best
arm costs nothing. Add up the costs. This version uses the gaps rather than
the rewards you happened to receive, so the coin flips inside each arm don't
add noise to the score.

Grouping the sum by arm gives a second view. If $N_T(a)$ is how many times you
pulled arm $a$, then regret is $\sum_a \Delta_a N_T(a)$. A good strategy pulls
each bad arm only as often as it takes to be sure the arm is bad.

In code, a regret curve is one running sum:

```python
total += bandit.regret(arm)   # the gap of the arm just pulled
curve[t] = total
```

The shape of the curve is what matters. A strategy that keeps wasting a fixed
fraction of its pulls has regret that grows *linearly*, in proportion to $T$.
A strategy that learns has regret that grows more slowly, and the best
possible growth is *logarithmic*, in proportion to $\ln T$. Lai and Robbins
(1985) proved that a strategy that does well on every Bernoulli bandit must,
in the long run, have regret that grows at least as fast as $\ln T$ on each
one.

> [!KEY]
> Regret adds up the gap of every arm you pull. Linear growth means you never
> stop wasting pulls. Logarithmic growth, the best possible, means you waste
> fewer and fewer.

## Explore at random with epsilon-greedy

The first strategy is the one most people invent. Usually pick the arm with
the best estimate. That's *exploiting*. But with a small probability
$\varepsilon$ (epsilon), pick an arm uniformly at random instead. That's
*exploring*, and it stops any arm from being written off after one unlucky
pull.

With $k$ arms, epsilon-greedy picks the arm with the best estimate with
probability $1 - \varepsilon + \varepsilon / k$. The $\varepsilon / k$ is the
chance that the random pick lands on it anyway. Each other arm gets
$\varepsilon / k$. For $k = 3$ and $\varepsilon = 0.1$, that's $0.9333$ for
the leader and $0.0333$ for each of the other two.

```python
if rng.random() < epsilon:
    arm = rng.integers(k)
else:
    best = np.flatnonzero(Q == Q.max())   # every arm tied for the lead
    arm = rng.choice(best)
```

> [!WARNING]
> `np.argmax` returns the lowest index among tied values. At the start every
> estimate is 0, so an `argmax` agent always tries arm 0 first and can lock
> onto it. Break ties at random, as the preceding code does.

Epsilon-greedy's flaw shows in its regret. Even after it knows the best arm,
it explores at the same rate forever. Each exploring pull costs the average
gap. So regret grows by $\varepsilon \times \frac{1}{k}\sum_a \Delta_a$ per
pull, which is linear.

## Be optimistic with UCB1

Random exploration wastes pulls on arms that are clearly bad. A better idea is
*optimism in the face of uncertainty*: treat each arm as if it's as good as it
could plausibly be, and pick the best of those. An arm wins because its
estimate is high or because you know little about it. Trying an uncertain arm
either confirms it's good or shrinks its uncertainty, so a mistake doesn't
repeat for long.

UCB1 (Auer, Cesa-Bianchi, and Fischer, 2002) turns this into a score. Let
$N(a)$ be the number of times you have pulled arm $a$, and $t$ the total
number of pulls so far. UCB1 pulls the arm with the largest score:

$$
\text{score}(a) = Q(a) + \sqrt{\frac{2 \ln t}{N(a)}}
$$

Read it as: the arm's average reward plus a *bonus* for uncertainty. The bonus
shrinks as the arm is pulled more, because $N(a)$ is in the denominator. It
grows slowly for arms you ignore, because $\ln t$ keeps rising. An arm that has
never been pulled has an infinite score, so UCB1 tries every arm once first.
The bonus is the width of a confidence interval that comes from Hoeffding's
inequality, a bound on how far a sample mean of values in $[0, 1]$ strays from
its expectation.

```python
scores = np.full(k, np.inf)
seen = counts > 0
scores[seen] = means[seen] + np.sqrt(2 * np.log(t) / counts[seen])
```

### Worked example

Two arms after $t = 7$ pulls. Arm 0 has paid an average of 0.6 over 5 pulls,
and arm 1 an average of 0.4 over 2 pulls. With $\ln 7 = 1.945910$:

| Arm | $Q(a)$ | $N(a)$ | Bonus $\sqrt{2 \ln 7 / N(a)}$ | Score |
|---|---|---|---|---|
| 0 | 0.6 | 5 | 0.882249 | 1.482249 |
| 1 | 0.4 | 2 | 1.394959 | 1.794959 |

UCB1 pulls arm 1, even though its estimate is lower. Two pulls aren't enough
to rule it out.

Auer and colleagues proved that UCB1's expected regret after $T$ pulls is at
most $8 \sum_{a: \Delta_a > 0} \frac{\ln T}{\Delta_a} + \left(1 + \frac{\pi^2}{3}\right)\sum_a \Delta_a$.
That grows like $\ln T$. Arms with small gaps cost the most, because they take
the longest to tell apart from the best arm.

## Sample your beliefs with Thompson sampling

The third strategy is the oldest, from Thompson in 1933. Keep a belief about
each arm's mean as a probability distribution. Each round, draw one plausible
mean from each arm's belief, and pull the arm whose draw is largest. An
uncertain arm sometimes draws high and gets tried. An arm you're sure is bad
almost never does.

For Bernoulli arms, the natural belief is a *Beta distribution*. Start each
arm at $\text{Beta}(1, 1)$, which is uniform on $[0, 1]$. After $S_a$ pulls
that paid 1 and $F_a$ pulls that paid 0, Bayes' rule gives the updated belief,
called the *posterior*:

$$
\mu_a \mid \text{data} \sim \text{Beta}(1 + S_a,\; 1 + F_a)
$$

Read it as: count the successes into the first parameter and the failures into
the second. The mean of $\text{Beta}(\alpha, \beta)$ is
$\alpha / (\alpha + \beta)$, and the distribution narrows as the counts grow.

```python
samples = rng.beta(1 + successes, 1 + failures)   # one draw per arm
arm = np.argmax(samples)
```

Thompson sampling pulls each arm with exactly the posterior probability that
it's the best arm. Take arm A with 3 successes and 1 failure, so
$\text{Beta}(4, 2)$ with mean $2/3$. Take arm B with one of each, so
$\text{Beta}(2, 2)$ with mean $1/2$. The probability that a draw from A beats
a draw from B is an integral of A's density times B's cumulative distribution:

$$
\Pr(\text{A's draw} > \text{B's draw}) = \int_0^1 20 x^3 (1 - x) \left(3x^2 - 2x^3\right) dx = \frac{5}{7}
$$

Read it as: for each value $x$ that A might draw, weight by how likely A is to
draw it, $20 x^3 (1-x)$, and by the chance that B draws less than $x$,
$3x^2 - 2x^3$. The result is $5/7 \approx 0.714$. So Thompson sampling plays A
71 percent of the time, and B the rest, while the evidence is this thin.

Agrawal and Goyal (2012) proved logarithmic regret for Thompson sampling, and
Kaufmann, Korda, and Munos (2012) showed that it matches the Lai-Robbins
lower bound for Bernoulli arms.

> [!KEY]
> UCB1 explores arms whose best plausible value is high. Thompson sampling
> explores arms in proportion to the chance they're best. Both stop exploring
> an arm once the evidence rules it out, which is why their regret grows only
> logarithmically.

## Compare the three on one bandit

Take three Bernoulli arms with means $0.1$, $0.5$, and $0.9$. The gaps are
$0.8$, $0.4$, and $0$. Averaged over four seeded runs, the exercise's
reference solution measures:

| Strategy | Regret at $T = 2000$ | Regret at $T = 8000$ | Growth from 2000 to 8000 |
|---|---|---|---|
| Epsilon-greedy, $\varepsilon = 0.1$ | 96.1 | 344.5 | 3.58 times |
| UCB1 | 35.1 | 48.5 | 1.38 times |
| Thompson sampling | 5.4 | 6.4 | 1.19 times |

Quadrupling $T$ nearly quadruples epsilon-greedy's regret. The predicted
linear rate is $0.1 \times (0.8 + 0.4 + 0) / 3 = 0.04$ per pull, which is
240 over the extra 6000 pulls. It pays 248. For comparison,
$\ln 8000 / \ln 2000 = 1.18$, and UCB1 and Thompson sampling grow by about
that factor.

UCB1's bound for this bandit at $T = 2000$ is
$8 \times 7.60 \times (1/0.8 + 1/0.4) + 4.29 \times 1.2 = 233$. The measured
35 is far below it. Regret bounds guarantee a worst case. They rarely predict
the actual number.

> [!RECAP]
> - The running mean is `q + (reward - q) / (n + 1)`. A constant `alpha`
>   weights recent rewards more and never settles.
> - Regret adds up `bandit.regret(arm)`, the gap of every arm you pull.
> - Epsilon-greedy explores at random forever, so its regret grows linearly.
> - UCB1 adds $\sqrt{2 \ln t / N(a)}$ to each estimate, and unpulled arms score
>   infinity.
> - Thompson sampling draws from $\text{Beta}(1 + S_a, 1 + F_a)$ for every arm
>   and plays the largest draw.

## Build it step by step

The checks test the functions in this order.

1. **`update_estimate(q, n, reward, alpha=None)`.** Takes the current
   estimate, the number of rewards already in it, and a new reward. Returns
   the new estimate. With `alpha=None` the step is `1 / (n + 1)`. Common
   mistake: using `1 / n`, which divides by zero on the first reward. The
   checks compare 200 updates against `np.cumsum(rewards) / n`, and the
   constant-step version against the weighted-sum formula.
2. **`epsilon_greedy_action(Q, epsilon, rng)`.** Returns an int. Draw
   `rng.random()` to decide whether to explore. Common mistake: `np.argmax`
   for the greedy pick. The check gives two arms equal estimates and expects
   each about half the time.
3. **`ucb1_scores(means, counts, t)`.** Returns a `(k,)` array. Start from
   `np.full(k, np.inf)` and fill only the pulled arms. Common mistake:
   `np.log(counts)` in place of `np.log(t)`.
4. **`thompson_action(successes, failures, rng)`.** One call to `rng.beta`
   with array arguments, then `np.argmax`. Common mistake: using the posterior
   means, which makes the strategy greedy. The check expects arm A to be
   picked $5/7$ of the time.
5. **`run_bandit(bandit, method, T, seed, epsilon=0.1)`.** Keep arrays `Q`,
   `N`, and `successes` of length `k`. Each step, choose an arm with the named
   method, pull it, update that arm's `Q` with `update_estimate`, and add
   `bandit.regret(arm)` to a running total. For UCB1, pass
   `max(t, 1)` as the total pulls so far and break ties at random. Return the
   `(T,)` array of running totals. Common mistake: updating every arm's
   counts, not only the pulled arm's. The last three checks compare the
   strategies' regret and its growth from $T = 2000$ to $T = 8000$.

## Cost and scaling

Every strategy here does a little work per pull, and it grows with the number
of arms $k$.

| Strategy | Work per pull | Memory |
|---|---|---|
| Epsilon-greedy | $O(k)$ for the max | $k$ estimates and counts |
| UCB1 | $O(k)$ for the scores | $k$ estimates and counts |
| Thompson sampling | $O(k)$ Beta draws | $k$ success and failure counts |

The harder cost is statistical. UCB1's regret scales like
$\sum_a \ln T / \Delta_a$, so many arms with small gaps cost a lot, and a
million arms cost a million pulls before each is tried once. Large problems
need structure that links the arms, such as the features in the next lesson.

## Where this breaks

**Small gaps.** When the best arms differ by a hair, every strategy needs many
pulls to separate them. UCB1 explores aggressively at first. On five arms with
means from $0.2$ to $0.8$ in steps of $0.15$, the exercise's reference UCB1
trails epsilon-greedy at $T = 2000$. It wins only over longer horizons.

**Arms that change.** Everything here assumes each arm's mean is fixed. If
means drift, sample averages lag behind. People use a constant step size, or
discard old data with a sliding window.

**Delayed or batched feedback.** In a real system, rewards often arrive in
batches, hours later. UCB1 is deterministic, so it pulls the same arm for the
whole batch. Thompson sampling's randomness spreads a batch across arms, which
is one reason it's popular in practice.

**Tuning epsilon.** A fixed $\varepsilon$ wastes pulls forever. Decaying it
helps, but the right schedule depends on the unknown gaps, and decaying too
fast can lock onto a bad arm.

## Further reading

- [Sutton and Barto, *Reinforcement Learning: An Introduction*, Chapter 2](http://incompleteideas.net/book/the-book-2nd.html)
  covers action values, epsilon-greedy, UCB, and optimistic initial values
  with the ten-armed testbed.
- [Auer, Cesa-Bianchi, and Fischer, "Finite-time analysis of the multiarmed bandit problem" (2002)](https://doi.org/10.1023/A:1013689704352)
  introduces UCB1 and proves its logarithmic regret bound at every $T$.
- [Lai and Robbins, "Asymptotically efficient adaptive allocation rules" (1985)](https://doi.org/10.1016/0196-8858(85)90002-8)
  proves the $\ln T$ lower bound that no consistent strategy can beat.
- [Thompson, "On the likelihood that one unknown probability exceeds another" (1933)](https://doi.org/10.1093/biomet/25.3-4.285)
  is the original posterior-sampling idea, written for medical trials.
- [Agrawal and Goyal, "Analysis of Thompson sampling for the multi-armed bandit problem" (2012)](https://arxiv.org/abs/1111.1797)
  gives the first logarithmic regret bound for Thompson sampling.
- [Russo, Van Roy, Kazerouni, Osband, and Wen, "A tutorial on Thompson sampling" (2018)](https://arxiv.org/abs/1707.02038)
  is a readable guide with many worked applications.
- [Lattimore and Szepesvári, *Bandit Algorithms* (2020)](https://tor-lattimore.com/downloads/book/book.pdf)
  is the full theory, free to download.

## The exercise

You build the three strategies and a regret tracker. Write
`update_estimate` for running and constant-step means,
`epsilon_greedy_action`, `ucb1_scores`, and `thompson_action` for the three
ways to choose, and `run_bandit`, which plays an `rlenvs.BernoulliBandit` for
`T` pulls and returns the cumulative regret after each one.

The checks verify the running mean against `np.cumsum`, the constant-step
formula, epsilon-greedy's choice frequencies and tie-breaking, the UCB1 table,
and Thompson sampling's $5/7$. On arms with means $0.1$, $0.5$, and $0.9$,
they check that uniform play pays the mean gap per pull, that UCB1 and
Thompson sampling beat epsilon-greedy by a wide margin at $T = 2000$, and that
from $T = 2000$ to $T = 8000$ their regret grows by less than 2.2 times while
epsilon-greedy's grows by more than 2.8 times.
