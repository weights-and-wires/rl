---
title: "Optimism in the face of uncertainty: UCB"
slug: 004-optimism-and-ucb
part: "Part 1 — Learning from reward"
summary: Explore the arms you know least about by adding an uncertainty bonus to each estimate, and see why regret grows only logarithmically.
minutes: 40
gpu: false
objectives:
  - Explain why epsilon-greedy wastes pulls, and what it means to explore in proportion to uncertainty.
  - Compute the UCB1 score of each arm by hand, and say what the bonus does as an arm is pulled more.
  - Say why the bonus has a logarithm of the total pulls in it.
  - Compare UCB1's regret curve with epsilon-greedy's and with UCB1's own bound.
prereqs:
  - 003-explore-or-exploit
exercise: 004-optimism-and-ucb
---

# Optimism in the face of uncertainty: UCB

Epsilon-greedy has a blind spot. When it explores, it picks an arm uniformly at random, so it spends as many pulls on a machine it has pulled a thousand times, and knows is bad, as on a machine it has pulled twice. Those pulls teach almost nothing, and they come out of your coins.

Take the row of slot machines at a moment when you have made 7 pulls. Machine 0 has paid 3 of 5, an average of 0.6. Machine 1 has paid 1 of 2, an average of 0.5. Epsilon-greedy calls machine 0 the leader and tries machine 1 only by a random draw. But two pulls are thin evidence, and machine 1 could well be the better machine. A better rule would pull machine 1 because you know so little about it.

This chapter turns that thought into a rule called UCB1. Each arm gets a score made of what you have seen plus a bonus for what you have not, and you pull the highest score.

## What you already know

- A running mean $Q(a)$ estimates each arm's average reward.
- The gap $\Delta_a$ is how much less arm $a$ pays than the best arm, and regret adds up the gaps of the arms you pull.
- Epsilon-greedy explores at a fixed rate, so its regret grows linearly.
- A sample mean of $N$ values gets closer to the true mean as $N$ grows.

## Be optimistic about what you do not know

Picture each arm's estimate as a dot with a bar above it. The bar says how high the arm's true mean could plausibly be. An arm pulled many times has a short bar, because its estimate is well pinned down. An arm pulled twice has a long bar. *Optimism in the face of uncertainty* is the rule: act as if each arm is as good as the top of its bar, and pull the arm whose top is highest.

Why does this explore sensibly? An arm wins the comparison for one of two reasons. Either its estimate is high, so pulling it earns well, or its bar is long, so you know little about it. In the second case, the pull either shows it is good, which is worth knowing, or shrinks its bar. Either way, one mistake does not repeat for long, because each wasted pull shortens the bar that caused it. Epsilon-greedy has no such self-correction.

## The UCB1 score

UCB1, by Auer, Cesa-Bianchi, and Fischer (2002), writes the bar as a formula. Let $N(a)$ be the number of times you have pulled arm $a$, and $t$ the total number of pulls so far. UCB1 pulls the arm with the largest score:

$$
\text{score}(a) = Q(a) + \sqrt{\frac{2 \ln t}{N(a)}}
$$

Read it as: the arm's average reward plus a *bonus* for uncertainty. The bonus has two parts that pull in opposite directions. It shrinks as $N(a)$ grows, because $N(a)$ is in the denominator, so a well-tried arm is judged almost by its average alone. It grows slowly with $\ln t$, so an arm you keep passing over sees its bonus creep up until it gets another try. An arm with $N(a) = 0$ has an infinite bonus, so UCB1 pulls every arm once before it repeats any.

The bonus is the width of a confidence interval. Hoeffding's inequality says that for $N$ rewards in $[0, 1]$, the chance that the true mean sits more than $c$ above the sample mean is at most $e^{-2Nc^2}$. Setting $c = \sqrt{2 \ln t / N}$ makes that chance $t^{-4}$, which falls fast as $t$ grows. So the score is a high-probability upper bound on the arm's true mean, and that is where the name, upper confidence bound, comes from.

```python
bonus = np.full(k, np.inf)
seen = counts > 0
bonus[seen] = np.sqrt(2 * np.log(t) / counts[seen])
scores = means + bonus
```

### Worked example

Here are the two machines after $t = 7$ pulls. With $\ln 7 = 1.945910$:

| Arm | $Q(a)$ | $N(a)$ | Bonus $\sqrt{2 \ln 7 / N(a)}$ | Score |
|---|---|---|---|---|
| 0 | 0.6 | 5 | 0.882249 | 1.482249 |
| 1 | 0.5 | 2 | 1.394959 | 1.894959 |

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="UCB1 scores after 7 pulls: arm 0 has an estimate of 0.6 plus a bonus of 0.88, scoring 1.48; arm 1 has an estimate of 0.5 plus a bonus of 1.39, scoring 1.89">
<line x1="60" y1="220" x2="560" y2="220" class="stroke-muted"/>
<line x1="60" y1="220.0" x2="560" y2="220.0" class="stroke-grid"/>
<text x="52" y="224.0" text-anchor="end" class="fig-small">0</text>
<line x1="60" y1="172.5" x2="560" y2="172.5" class="stroke-grid"/>
<text x="52" y="176.5" text-anchor="end" class="fig-small">0.5</text>
<line x1="60" y1="125.0" x2="560" y2="125.0" class="stroke-grid"/>
<text x="52" y="129.0" text-anchor="end" class="fig-small">1</text>
<line x1="60" y1="77.5" x2="560" y2="77.5" class="stroke-grid"/>
<text x="52" y="81.5" text-anchor="end" class="fig-small">1.5</text>
<line x1="60" y1="30.0" x2="560" y2="30.0" class="stroke-grid"/>
<text x="52" y="34.0" text-anchor="end" class="fig-small">2</text>
<rect x="130" y="163.0" width="90" height="57.0" class="fill-a"/>
<rect x="130" y="79.2" width="90" height="83.8" class="fill-soft-b stroke-b"/>
<text x="175" y="195.5" text-anchor="middle" class="fig-bold">Q = 0.6</text>
<text x="175" y="125.1" text-anchor="middle" class="fig-small">bonus 0.882</text>
<text x="175" y="71.2" text-anchor="middle" class="fig-bold">score 1.482</text>
<text x="175" y="240" text-anchor="middle">arm 0, pulled 5 times</text>
<rect x="330" y="172.5" width="90" height="47.5" class="fill-a"/>
<rect x="330" y="40.0" width="90" height="132.5" class="fill-soft-b stroke-b"/>
<text x="375" y="200.2" text-anchor="middle" class="fig-bold">Q = 0.5</text>
<text x="375" y="110.2" text-anchor="middle" class="fig-small">bonus 1.395</text>
<text x="375" y="32.0" text-anchor="middle" class="fig-bold">score 1.895</text>
<text x="375" y="240" text-anchor="middle">arm 1, pulled 2 times</text>
<text x="470" y="60" class="fig-small">UCB1 pulls arm 1</text>
</svg>
Figure 1. Each bar is an estimate (blue) with its uncertainty bonus stacked on top. Arm 1 has the lower estimate but the higher score, because two pulls leave much more doubt than five.
```

UCB1 pulls arm 1, even though arm 0 has the better average. Suppose arm 1 pays 0. Then at $t = 8$ its estimate is $1/3$ with $N = 3$, so its bonus drops from 1.395 to 1.177 and its score to 1.511. Arm 0's score rises to 1.512, because $\ln t$ grew while its count did not. Arm 0 wins the next round by a hair. The bad news did what it should: it shrank the doubt about arm 1 and the lead moved on.

> [!WARNING]
> The logarithm takes $t$, the total number of pulls, and never $N(a)$. Writing `np.log(counts)` makes the bonus shrink with the arm's own count twice over and removes the slow growth that brings neglected arms back.

## Why regret grows only logarithmically

Reason about one bad arm with gap $\Delta$. UCB1 pulls it only when its score beats the best arm's, and the best arm's score is at least about $\mu^*$. The bad arm's score is about $\mu_a + \sqrt{2 \ln t / N_a}$. So it keeps getting pulled only while its bonus exceeds roughly $\Delta$, that is, while $\sqrt{2 \ln t / N_a} \gtrsim \Delta$. Solving for the count gives $N_a \lesssim 2 \ln t / \Delta^2$. Each of those pulls costs $\Delta$ in regret, so arm $a$ costs about $2 \ln t / \Delta$ in all.

The count grows like $\ln t$, not like $t$, because the bonus only has to fall below the gap to stop the pulls, and a bonus that falls like $1/\sqrt{N}$ does that quickly. Auer and colleagues proved a bound that pays a larger constant for the noise in the estimates. For Bernoulli arms, the expected regret after $T$ pulls is at most

$$
8 \sum_{a:\, \Delta_a > 0} \frac{\ln T}{\Delta_a} + \left(1 + \frac{\pi^2}{3}\right) \sum_a \Delta_a
$$

Read it as: each bad arm contributes $8 \ln T / \Delta_a$, so an arm with a small gap costs the most, because it takes the longest to tell apart from the best arm. For the row at $T = 2000$ the bound is $8 \times 7.601 \times (1/0.8 + 1/0.4) + 4.290 \times 1.2 = 233$.

Figure 2 compares the curves. The mean over 100 seeded runs at $T = 2000$ is 41 for UCB1 and 90 for epsilon-greedy.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Cumulative regret over 8000 pulls, mean of 100 runs: epsilon-greedy climbs in a straight line to 330, while UCB1 flattens to 57 and stays under its dashed theoretical bound">
<line x1="64" y1="230" x2="515" y2="230" class="stroke-muted"/>
<line x1="64" y1="230.0" x2="515" y2="230.0" class="stroke-grid"/>
<text x="56" y="234.0" text-anchor="end" class="fig-small">0</text>
<line x1="64" y1="171.1" x2="515" y2="171.1" class="stroke-grid"/>
<text x="56" y="175.1" text-anchor="end" class="fig-small">100</text>
<line x1="64" y1="112.3" x2="515" y2="112.3" class="stroke-grid"/>
<text x="56" y="116.3" text-anchor="end" class="fig-small">200</text>
<line x1="64" y1="53.4" x2="515" y2="53.4" class="stroke-grid"/>
<text x="56" y="57.4" text-anchor="end" class="fig-small">300</text>
<text x="64.0" y="248" text-anchor="middle" class="fig-small">0</text>
<text x="176.8" y="248" text-anchor="middle" class="fig-small">2000</text>
<text x="289.5" y="248" text-anchor="middle" class="fig-small">4000</text>
<text x="402.2" y="248" text-anchor="middle" class="fig-small">6000</text>
<text x="515.0" y="248" text-anchor="middle" class="fig-small">8000</text>
<text x="289.5" y="270" text-anchor="middle" class="fig-small">pulls, T</text>
<polyline points="64.0,230.0 78.1,219.2 92.2,213.4 106.3,207.4 120.4,201.2 134.5,195.0 148.6,189.2 162.7,183.1 176.8,177.1 190.8,171.1 204.9,165.0 219.0,159.0 233.1,152.9 247.2,147.1 261.3,141.3 275.4,135.4 289.5,129.6 303.6,124.0 317.7,118.4 331.8,112.5 345.9,106.5 360.0,100.6 374.1,94.6 388.2,88.7 402.2,83.0 416.3,77.1 430.4,71.2 444.5,65.2 458.6,59.4 472.7,53.7 486.8,47.8 500.9,42.0 515.0,36.1" fill="none" class="stroke-b" stroke-width="2.5"/>
<text x="521.0" y="40.1" class="fig-small">ε-greedy</text>
<polyline points="64.0,230.0 78.1,218.3 92.2,214.4 106.3,212.0 120.4,210.1 134.5,208.8 148.6,207.6 162.7,206.5 176.8,205.6 190.8,204.9 204.9,204.3 219.0,203.7 233.1,203.1 247.2,202.4 261.3,201.9 275.4,201.6 289.5,201.3 303.6,200.8 317.7,200.3 331.8,200.0 345.9,199.6 360.0,199.2 374.1,198.9 388.2,198.5 402.2,198.2 416.3,197.9 430.4,197.6 444.5,197.4 458.6,197.3 472.7,197.1 486.8,196.9 500.9,196.6 515.0,196.4" fill="none" class="stroke-a" stroke-width="2.5"/>
<text x="521.0" y="200.4" class="fig-small">UCB1</text>
<polyline points="78.1,129.5 92.2,117.2 106.3,110.1 120.4,105.0 134.5,101.1 148.6,97.8 162.7,95.1 176.8,92.8 190.8,90.7 204.9,88.8 219.0,87.1 233.1,85.6 247.2,84.2 261.3,82.9 275.4,81.7 289.5,80.5 303.6,79.5 317.7,78.4 331.8,77.5 345.9,76.6 360.0,75.7 374.1,74.9 388.2,74.1 402.2,73.4 416.3,72.6 430.4,71.9 444.5,71.3 458.6,70.6 472.7,70.0 486.8,69.4 500.9,68.8 515.0,68.3" fill="none" class="stroke-muted dashed" stroke-width="2.5"/>
<text x="521.0" y="72.3" class="fig-small">bound</text>
</svg>
Figure 2. Mean regret over 100 runs. Epsilon-greedy (orange) keeps climbing in a straight line. UCB1 (blue) flattens and stays well under its theoretical bound (dashed).
```

Quadrupling $T$ from 2000 to 8000 multiplies epsilon-greedy's regret by 3.67 and UCB1's by 1.38. A logarithmic curve would grow by $\ln 8000 / \ln 2000 = 1.18$, and UCB1 is close to it. The measured 41 sits far under the bound of 233 because a bound has to hold for every bandit, including the worst ones, and so rarely predicts a typical run.

> [!KEY]
> UCB1 explores the arms whose best plausible value is high. A pull shrinks that arm's uncertainty, so no arm keeps being tried once the evidence rules it out, and regret grows like $\ln T$ and not like $T$.

> [!RECAP]
> - The bonus $\sqrt{2 \ln t / N(a)}$ is a confidence-interval width: it shrinks with the arm's pulls and grows with $\ln t$.
> - An unpulled arm has an infinite score, so every arm is tried once first.
> - Ties at the top are broken at random, as in epsilon-greedy.
> - UCB1's expected regret is at most $8 \sum \ln T / \Delta_a + (1 + \pi^2/3) \sum_a \Delta_a$, which grows like $\ln T$.

## Build it step by step

The bandit loop `run_bandit` is given in full: its epsilon-greedy half is your code from the last chapter, and its UCB1 half calls your `ucb1_action`. The checks test the functions in this order.

1. **`ucb_bonus(counts, t)`.** Returns shape `(k,)`. Start from `np.full(k, np.inf)` and fill only the arms with `counts > 0`. The common mistake is `np.log(counts)` in place of `np.log(t)`.
2. **`ucb1_scores(means, counts, t)`.** Returns `means + ucb_bonus(counts, t)`. An unpulled arm gets `np.inf`.
3. **`ucb1_action(means, counts, t, rng)`.** Pick from `np.flatnonzero(scores == scores.max())` with `rng.choice`. The common mistake is `np.argmax`, which always picks the lowest index, so ties always go to the lowest-numbered arm.
4. **`ucb1_regret_bound(gaps, T)`.** Evaluate the bound above. Drop the arms with gap 0 before dividing, because the best arm adds nothing to the first sum.

## Where this breaks

**Small gaps.** When arms differ by a hair, UCB1 explores a lot before it separates them. On five arms with means from $0.2$ to $0.8$ in steps of $0.15$, UCB1's mean regret at $T = 2000$ is 105, against 95 for epsilon-greedy over 20 runs. UCB1 wins only over longer horizons.

**A loose bound.** The constant 8 is chosen so that a proof goes through, and it is far larger than what runs need. People often shrink the bonus in practice, at the cost of the guarantee.

**Rewards outside the range.** The bonus assumes rewards in $[0, 1]$. Rewards on another scale need the bonus scaled to match.

**Delayed feedback.** UCB1 is deterministic, so when rewards arrive in batches it pulls the same arm for the whole batch. The next chapter's strategy spreads a batch across arms.

## Further reading

- [Auer, Cesa-Bianchi, and Fischer, "Finite-time analysis of the multiarmed bandit problem" (2002)](https://doi.org/10.1023/A:1013689704352) introduces UCB1 and proves its logarithmic regret bound at every $T$.
- [Sutton and Barto, *Reinforcement Learning: An Introduction*, Section 2.7](http://incompleteideas.net/book/the-book-2nd.html) explains upper-confidence-bound action selection.
- [Lattimore and Szepesvári, *Bandit Algorithms* (2020)](https://tor-lattimore.com/downloads/book/book.pdf) develops the theory of confidence bounds in full, and it is free to download.

## The exercise

You build UCB1 and check it against its bound. Write `ucb_bonus`, `ucb1_scores`, `ucb1_action` with random tie-breaking, and `ucb1_regret_bound`. The bandit loop `run_bandit` is given, and it calls your `ucb1_action`.

The checks verify the bonus table, that the bonus grows with $t$, and the scores of the worked example. They check that `ucb1_action` picks arm 1 in the example, picks arm 0 once arm 1 pays 0, tries unpulled arms first, and splits ties evenly. They check that the bound is 233 for the row at $T = 2000$, that UCB1 pulls each arm once before repeating any, that its mean regret at $T = 2000$ is under 75 percent of epsilon-greedy's and under the bound, and that its regret grows by less than 2.2 times from $T = 2000$ to $T = 8000$.
