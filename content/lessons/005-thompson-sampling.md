---
title: Thompson sampling
slug: 005-thompson-sampling
part: "Part 1 — Learning from reward"
summary: Keep a Beta belief about each arm's payout rate, draw one plausible rate per arm, and pull the arm with the largest draw.
minutes: 40
gpu: false
objectives:
  - Describe a Beta belief about a payout rate, and update it from successes and failures by counting.
  - Run one round of Thompson sampling by hand, and say why it sometimes pulls the arm with the lower mean.
  - Explain why sampling beliefs plays each arm with the probability that it is the best.
  - Compare Thompson sampling's regret with epsilon-greedy's and UCB1's on the slot-machine row.
prereqs:
  - 003-explore-or-exploit
  - 004-optimism-and-ucb
exercise: 005-thompson-sampling
---

# Thompson sampling

UCB1 explores by adding a bonus, and the bonus comes from a formula with a constant that a proof chose. There is a more direct way to say what you do not know. For each machine, hold a belief about its payout rate: a spread of how plausible each possible rate is. A machine you have pulled twice has a wide belief, and one you have pulled a thousand times has a narrow one.

Then act on the belief with one random step. Each round, draw one possible payout rate from each machine's belief, and pull the machine whose draw is highest. This is *Thompson sampling*, from Thompson in 1933. This chapter shows why it explores the right amount without a bonus, an $\varepsilon$, or a tuning constant, and how it does on the row of slot machines.

## What you already know

- A running mean estimates each arm's payout, and regret adds up the gaps of the arms you pull.
- Epsilon-greedy explores blindly, and UCB1 explores in proportion to a confidence bound.
- A Bernoulli arm pays 1 with probability $\mu_a$ and 0 otherwise.
- `rng.beta(a, b)` draws random numbers between 0 and 1, and with array arguments it draws one per entry.

## A belief about a payout rate

The unknown here is a number between 0 and 1, the arm's payout rate. A belief about it is a curve over $[0, 1]$: high where the rate is plausible, low where it is not. The *Beta distribution* is the family of curves that fits, and it takes two numbers, $\alpha$ and $\beta$, written $\text{Beta}(\alpha, \beta)$.

Start with no information. $\text{Beta}(1, 1)$ is flat, so every rate is equally plausible. Every pull then adds to a count. A pull that pays 1 adds one to $\alpha$, and a pull that pays 0 adds one to $\beta$. After $S$ successes and $F$ failures, the belief, called the *posterior*, is

$$
\mu_a \mid \text{data} \sim \text{Beta}(1 + S,\; 1 + F)
$$

Read it as: count the successes into the first number and the failures into the second. This works because Bayes' rule applied to a Bernoulli observation turns a Beta curve into another Beta curve, so the update is nothing but counting. The mean of $\text{Beta}(\alpha, \beta)$ is $\alpha / (\alpha + \beta)$, and the curve narrows as the counts grow. The next table shows both effects.

| Successes, failures | Belief | Mean | Standard deviation |
|---|---|---|---|
| 0, 0 | Beta(1, 1) | 0.500 | 0.289 |
| 1, 1 | Beta(2, 2) | 0.500 | 0.224 |
| 3, 1 | Beta(4, 2) | 0.667 | 0.178 |
| 30, 10 | Beta(31, 11) | 0.738 | 0.067 |
| 300, 100 | Beta(301, 101) | 0.749 | 0.022 |

Take two arms. Arm A has 3 successes and 1 failure, so its belief is $\text{Beta}(4, 2)$. Arm B has one of each, so its belief is $\text{Beta}(2, 2)$. Figure 1 draws both.

```figure
<svg viewBox="0 0 600 280" role="img" aria-label="Two belief curves over an arm's payout rate: Beta(4, 2) for arm A peaks near 0.75, Beta(2, 2) for arm B peaks at 0.5 and is wider. Arm A's draw beats arm B's with probability 5 out of 7">
<line x1="50" y1="230" x2="560" y2="230" class="stroke-muted"/>
<text x="50.0" y="248" text-anchor="middle" class="fig-small">0</text>
<text x="177.5" y="248" text-anchor="middle" class="fig-small">0.25</text>
<text x="305.0" y="248" text-anchor="middle" class="fig-small">0.5</text>
<text x="432.5" y="248" text-anchor="middle" class="fig-small">0.75</text>
<text x="560.0" y="248" text-anchor="middle" class="fig-small">1</text>
<text x="305.0" y="270" text-anchor="middle" class="fig-small">possible payout rate of the arm</text>
<polygon points="50,230 50.0,230.0 55.1,226.0 60.2,222.2 65.3,218.4 70.4,214.6 75.5,211.0 80.6,207.4 85.7,204.0 90.8,200.6 95.9,197.2 101.0,194.0 106.1,190.8 111.2,187.8 116.3,184.8 121.4,181.8 126.5,179.0 131.6,176.2 136.7,173.6 141.8,171.0 146.9,168.4 152.0,166.0 157.1,163.6 162.2,161.4 167.3,159.2 172.4,157.0 177.5,155.0 182.6,153.0 187.7,151.2 192.8,149.4 197.9,147.6 203.0,146.0 208.1,144.4 213.2,143.0 218.3,141.6 223.4,140.2 228.5,139.0 233.6,137.8 238.7,136.8 243.8,135.8 248.9,134.8 254.0,134.0 259.1,133.2 264.2,132.6 269.3,132.0 274.4,131.4 279.5,131.0 284.6,130.6 289.7,130.4 294.8,130.2 299.9,130.0 305.0,130.0 310.1,130.0 315.2,130.2 320.3,130.4 325.4,130.6 330.5,131.0 335.6,131.4 340.7,132.0 345.8,132.6 350.9,133.2 356.0,134.0 361.1,134.8 366.2,135.8 371.3,136.8 376.4,137.8 381.5,139.0 386.6,140.2 391.7,141.6 396.8,143.0 401.9,144.4 407.0,146.0 412.1,147.6 417.2,149.4 422.3,151.2 427.4,153.0 432.5,155.0 437.6,157.0 442.7,159.2 447.8,161.4 452.9,163.6 458.0,166.0 463.1,168.4 468.2,171.0 473.3,173.6 478.4,176.2 483.5,179.0 488.6,181.8 493.7,184.8 498.8,187.8 503.9,190.8 509.0,194.0 514.1,197.2 519.2,200.6 524.3,204.0 529.4,207.4 534.5,211.0 539.6,214.6 544.7,218.4 549.8,222.2 554.9,226.0 560.0,230.0 560,230" class="fill-soft-d" opacity="0.55"/>
<polyline points="50.0,230.0 55.1,226.0 60.2,222.2 65.3,218.4 70.4,214.6 75.5,211.0 80.6,207.4 85.7,204.0 90.8,200.6 95.9,197.2 101.0,194.0 106.1,190.8 111.2,187.8 116.3,184.8 121.4,181.8 126.5,179.0 131.6,176.2 136.7,173.6 141.8,171.0 146.9,168.4 152.0,166.0 157.1,163.6 162.2,161.4 167.3,159.2 172.4,157.0 177.5,155.0 182.6,153.0 187.7,151.2 192.8,149.4 197.9,147.6 203.0,146.0 208.1,144.4 213.2,143.0 218.3,141.6 223.4,140.2 228.5,139.0 233.6,137.8 238.7,136.8 243.8,135.8 248.9,134.8 254.0,134.0 259.1,133.2 264.2,132.6 269.3,132.0 274.4,131.4 279.5,131.0 284.6,130.6 289.7,130.4 294.8,130.2 299.9,130.0 305.0,130.0 310.1,130.0 315.2,130.2 320.3,130.4 325.4,130.6 330.5,131.0 335.6,131.4 340.7,132.0 345.8,132.6 350.9,133.2 356.0,134.0 361.1,134.8 366.2,135.8 371.3,136.8 376.4,137.8 381.5,139.0 386.6,140.2 391.7,141.6 396.8,143.0 401.9,144.4 407.0,146.0 412.1,147.6 417.2,149.4 422.3,151.2 427.4,153.0 432.5,155.0 437.6,157.0 442.7,159.2 447.8,161.4 452.9,163.6 458.0,166.0 463.1,168.4 468.2,171.0 473.3,173.6 478.4,176.2 483.5,179.0 488.6,181.8 493.7,184.8 498.8,187.8 503.9,190.8 509.0,194.0 514.1,197.2 519.2,200.6 524.3,204.0 529.4,207.4 534.5,211.0 539.6,214.6 544.7,218.4 549.8,222.2 554.9,226.0 560.0,230.0" fill="none" class="stroke-d" stroke-width="2.5"/>
<polygon points="50,230 50.0,230.0 55.1,230.0 60.2,230.0 65.3,230.0 70.4,229.9 75.5,229.8 80.6,229.7 85.7,229.6 90.8,229.4 95.9,229.1 101.0,228.8 106.1,228.4 111.2,228.0 116.3,227.5 121.4,226.9 126.5,226.2 131.6,225.4 136.7,224.6 141.8,223.6 146.9,222.6 152.0,221.5 157.1,220.2 162.2,218.9 167.3,217.5 172.4,216.0 177.5,214.4 182.6,212.7 187.7,210.8 192.8,208.9 197.9,206.9 203.0,204.8 208.1,202.6 213.2,200.3 218.3,197.9 223.4,195.4 228.5,192.8 233.6,190.2 238.7,187.5 243.8,184.6 248.9,181.8 254.0,178.8 259.1,175.8 264.2,172.7 269.3,169.6 274.4,166.4 279.5,163.2 284.6,159.9 289.7,156.6 294.8,153.3 299.9,150.0 305.0,146.7 310.1,143.3 315.2,140.0 320.3,136.7 325.4,133.4 330.5,130.2 335.6,127.0 340.7,123.8 345.8,120.7 350.9,117.7 356.0,114.8 361.1,112.0 366.2,109.2 371.3,106.6 376.4,104.2 381.5,101.8 386.6,99.7 391.7,97.7 396.8,95.8 401.9,94.2 407.0,92.8 412.1,91.6 417.2,90.7 422.3,90.0 427.4,89.5 432.5,89.4 437.6,89.5 442.7,90.0 447.8,90.8 452.9,91.9 458.0,93.5 463.1,95.4 468.2,97.7 473.3,100.4 478.4,103.6 483.5,107.2 488.6,111.3 493.7,115.9 498.8,121.0 503.9,126.6 509.0,132.8 514.1,139.6 519.2,146.9 524.3,154.9 529.4,163.6 534.5,172.8 539.6,182.8 544.7,193.5 549.8,204.9 554.9,217.1 560.0,230.0 560,230" class="fill-soft-a" opacity="0.55"/>
<polyline points="50.0,230.0 55.1,230.0 60.2,230.0 65.3,230.0 70.4,229.9 75.5,229.8 80.6,229.7 85.7,229.6 90.8,229.4 95.9,229.1 101.0,228.8 106.1,228.4 111.2,228.0 116.3,227.5 121.4,226.9 126.5,226.2 131.6,225.4 136.7,224.6 141.8,223.6 146.9,222.6 152.0,221.5 157.1,220.2 162.2,218.9 167.3,217.5 172.4,216.0 177.5,214.4 182.6,212.7 187.7,210.8 192.8,208.9 197.9,206.9 203.0,204.8 208.1,202.6 213.2,200.3 218.3,197.9 223.4,195.4 228.5,192.8 233.6,190.2 238.7,187.5 243.8,184.6 248.9,181.8 254.0,178.8 259.1,175.8 264.2,172.7 269.3,169.6 274.4,166.4 279.5,163.2 284.6,159.9 289.7,156.6 294.8,153.3 299.9,150.0 305.0,146.7 310.1,143.3 315.2,140.0 320.3,136.7 325.4,133.4 330.5,130.2 335.6,127.0 340.7,123.8 345.8,120.7 350.9,117.7 356.0,114.8 361.1,112.0 366.2,109.2 371.3,106.6 376.4,104.2 381.5,101.8 386.6,99.7 391.7,97.7 396.8,95.8 401.9,94.2 407.0,92.8 412.1,91.6 417.2,90.7 422.3,90.0 427.4,89.5 432.5,89.4 437.6,89.5 442.7,90.0 447.8,90.8 452.9,91.9 458.0,93.5 463.1,95.4 468.2,97.7 473.3,100.4 478.4,103.6 483.5,107.2 488.6,111.3 493.7,115.9 498.8,121.0 503.9,126.6 509.0,132.8 514.1,139.6 519.2,146.9 524.3,154.9 529.4,163.6 534.5,172.8 539.6,182.8 544.7,193.5 549.8,204.9 554.9,217.1 560.0,230.0" fill="none" class="stroke-a" stroke-width="2.5"/>
<text x="70" y="70" class="fig-bold">arm B: Beta(2, 2)</text><text x="70" y="88" class="fig-small">1 success, 1 failure</text>
<text x="400" y="40" class="fig-bold">arm A: Beta(4, 2)</text><text x="400" y="58" class="fig-small">3 successes, 1 failure</text>
<circle cx="259.4" cy="230" r="6" class="fill-a"/>
<circle cx="410.8" cy="230" r="6" class="fill-d"/>
<text x="259.4" y="218" text-anchor="middle" class="fig-small">draw A 0.41</text>
<text x="410.8" y="218" text-anchor="middle" class="fig-small">draw B 0.71</text>
</svg>
Figure 1. Beliefs about two arms. Arm A's mean is higher, but the curves overlap a great deal, so a draw from B can land above a draw from A. One seeded round drew 0.41 for A and 0.71 for B.
```

## Sample, then act

Thompson sampling uses the beliefs in two steps:

1. Draw one number from each arm's belief. Each draw is a payout rate that the evidence has not ruled out.
2. Pull the arm whose draw is largest.

In numpy, one call draws for every arm at once:

```python
samples = rng.beta(1 + successes, 1 + failures)   # one draw per arm
arm = np.argmax(samples)
```

In the seeded round of Figure 1, arm A drew 0.41 and arm B drew 0.71. So B is pulled, even though A has the higher mean. That is not a mistake. B's belief is wide enough that 0.71 is a believable rate for it, and trying B is how you find out. A draw like that is how the strategy explores.

Why not pull the arm with the highest posterior mean, 0.667 for A? That is greedy again. It would pull A every round, and B would never get a chance to prove that it is better. The random draw is what keeps B in play, in proportion to how much doubt there is about it.

## How often does each arm get pulled?

The chance that a draw from A beats a draw from B is an integral. For each value $x$ that A might draw, weight by how likely A is to draw it, $20 x^3 (1-x)$, and by the chance that B draws less than $x$, $3x^2 - 2x^3$:

$$
\Pr(\text{A's draw} > \text{B's draw}) = \int_0^1 20 x^3 (1 - x) \left(3x^2 - 2x^3\right) dx = \frac{5}{7}
$$

Read it as: Thompson sampling pulls A in $5/7 \approx 71$ percent of rounds and B in the other 29 percent, while the evidence is this thin. The number has a meaning beyond this pair. A draw from each belief is a possible world, and the arm with the largest draw is the best arm in that world. So Thompson sampling pulls each arm with exactly the posterior probability that it is the best arm. You can estimate that probability by sampling, which is what `prob_best` does in the exercise:

```python
samples = rng.beta(a, b, size=(n_samples, k))      # n_samples rounds of draws
wins = np.bincount(samples.argmax(axis=1), minlength=k) / n_samples
```

This also says why the exploration stops. As an arm is pulled, its belief narrows around its true rate, as the last rows of the table show. An arm that is clearly worse has a narrow belief that sits below the best arm's, and it almost never draws the highest number. So the strategy pulls the arms in proportion to the chance that each is best, and that chance falls toward zero for an arm the evidence has ruled out.

> [!KEY]
> Thompson sampling pulls each arm with the probability that it is the best arm. Beliefs that are wide get tried, and beliefs that have narrowed onto a poor rate are left alone, so the exploration shrinks by itself as evidence arrives.

## Compare the three on one bandit

Run all three strategies on the row, with payout rates $0.1$, $0.5$, and $0.9$. The next table gives the mean regret over 100 seeded runs.

| Strategy | Regret at $T = 2000$ | Regret at $T = 8000$ | Growth from 2000 to 8000 |
|---|---|---|---|
| Epsilon-greedy, $\varepsilon = 0.1$ | 89.8 | 329.5 | 3.67 times |
| UCB1 | 41.4 | 57.1 | 1.38 times |
| Thompson sampling | 6.9 | 8.1 | 1.18 times |

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Cumulative regret over 8000 pulls, mean of 100 runs: epsilon-greedy reaches 330, UCB1 57, and Thompson sampling 8">
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
<polyline points="64.0,230.0 78.1,227.0 92.2,226.7 106.3,226.5 120.4,226.4 134.5,226.2 148.6,226.1 162.7,226.0 176.8,226.0 190.8,225.9 204.9,225.9 219.0,225.8 233.1,225.8 247.2,225.7 261.3,225.7 275.4,225.7 289.5,225.6 303.6,225.6 317.7,225.6 331.8,225.5 345.9,225.5 360.0,225.5 374.1,225.4 388.2,225.4 402.2,225.4 416.3,225.4 430.4,225.4 444.5,225.3 458.6,225.3 472.7,225.3 486.8,225.3 500.9,225.3 515.0,225.2" fill="none" class="stroke-c" stroke-width="2.5"/>
<text x="521.0" y="229.2" class="fig-small">Thompson</text>
</svg>
Figure 2. Mean regret over 100 runs on the row. Epsilon-greedy climbs in a straight line, UCB1 flattens, and Thompson sampling is nearly flat from early on.
```

Thompson sampling's growth, 1.18 times, matches $\ln 8000 / \ln 2000 = 1.18$, the growth of a pure logarithm. Agrawal and Goyal (2012) proved logarithmic regret for Thompson sampling, and Kaufmann, Korda, and Munos (2012) showed that on Bernoulli arms it matches the Lai-Robbins lower bound from the chapter on epsilon-greedy.

> [!WARNING]
> The sizes in this table belong to this row, where the gaps are large. On five arms with means from $0.2$ to $0.8$ in steps of $0.15$, the mean regret at $T = 2000$ over 20 runs was 21.5 for Thompson sampling, 105 for UCB1, and 95 for epsilon-greedy. The order holds, but every strategy pays more when the arms are close, and UCB1 falls behind epsilon-greedy.

> [!RECAP]
> - A belief about a payout rate is $\text{Beta}(1 + S, 1 + F)$ after $S$ successes and $F$ failures.
> - One round draws one rate per arm with `rng.beta` and pulls the `argmax` of the draws, never of the posterior means.
> - Each arm is pulled with the posterior probability that it is the best arm.
> - Regret on the row grows like $\ln T$ and stays far below UCB1's.

## Build it step by step

The bandit loop `run_bandit` is given in full, with its epsilon-greedy and UCB1 halves from the earlier chapters and a Thompson half that calls your `thompson_action`. The checks test the functions in this order.

1. **`beta_posterior(successes, failures)`.** Returns the two Beta parameters of every arm as a pair of float arrays, `1 + successes` and `1 + failures`. The common mistake is forgetting the $+1$, which makes the prior impossible: $\text{Beta}(0, 0)$ is not a distribution.
2. **`thompson_action(successes, failures, rng)`.** One call to `rng.beta` with array arguments, then `np.argmax`. Return an int. The common mistake is taking the argmax of the posterior means, which is greedy. The check expects arm A to be picked $5/7$ of the time, and an arm with 5 successes in 100 pulls to almost never be picked against one with 80.
3. **`prob_best(successes, failures, n_samples, rng)`.** Draw an `(n_samples, k)` array of Beta samples, take the argmax along each row, and count the winners with `np.bincount`. Return shape `(k,)` with entries that add up to 1. The check compares your frequencies of `thompson_action` with `prob_best`.

## Where this breaks

**The prior.** $\text{Beta}(1, 1)$ is one choice of starting belief. A prior that is confident and wrong slows learning, because it takes many pulls for the data to outweigh it.

**Other rewards.** The counting update belongs to Bernoulli rewards. Rewards that are not zero or one need a different belief, or an approximation, which is more work.

**Cost.** Each round draws one sample per arm, so the work is $O(k)$ draws per pull, the same order as the scores of UCB1.

**Drifting arms.** Beliefs built from all past data lag when the payout rates change. A common repair is to discount old counts so that old evidence fades.

## Further reading

- [Thompson, "On the likelihood that one unknown probability exceeds another in view of the evidence of two samples" (1933)](https://doi.org/10.1093/biomet/25.3-4.285) is the original posterior-sampling idea, written for medical trials.
- [Agrawal and Goyal, "Analysis of Thompson sampling for the multi-armed bandit problem" (2012)](https://arxiv.org/abs/1111.1797) gives the first logarithmic regret bound for Thompson sampling.
- [Kaufmann, Korda, and Munos, "Thompson sampling: an asymptotically optimal finite-time analysis" (2012)](https://arxiv.org/abs/1205.4217) shows that it matches the Lai-Robbins lower bound on Bernoulli arms.
- [Russo, Van Roy, Kazerouni, Osband, and Wen, "A tutorial on Thompson sampling" (2018)](https://arxiv.org/abs/1707.02038) is a readable guide with many worked applications.

## The exercise

You build Thompson sampling for Bernoulli arms. Write `beta_posterior`, `thompson_action`, and `prob_best`. The bandit loop `run_bandit` is given, and it calls your `thompson_action`.

The checks verify the posterior parameters and the $5/7$ chance that $\text{Beta}(4, 2)$ beats $\text{Beta}(2, 2)$, both from `thompson_action` and from `prob_best`. They check that a well-evidenced arm almost never loses to a poor one, and that with no data every arm is picked equally often. Then they check that `thompson_action` plays each arm with the probability that `prob_best` reports, that Thompson's regret is under 30 percent of epsilon-greedy's and below UCB1's at $T = 2000$, and that it grows by less than 2.2 times from $T = 2000$ to $T = 8000$.
