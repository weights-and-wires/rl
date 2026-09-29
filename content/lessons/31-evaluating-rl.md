---
title: Evaluating reinforcement learning
slug: 31-evaluating-rl
part: "Part 7 — Rewards, evaluation, and many agents"
summary: Report RL results that survive scrutiny with the interquartile mean, performance profiles, and stratified bootstrap intervals.
minutes: 45
gpu: false
objectives:
  - Explain why a few runs and a point estimate can't separate two algorithms, and measure how often the winner flips.
  - Compute the interquartile mean, optimality gap, performance profile, and probability of improvement by the rliable definitions.
  - Build a stratified bootstrap and a percentile interval, and check its variance against the exact formula.
  - Say what an interval does and doesn't promise when you have only a handful of runs.
prereqs:
  - 12-deep-q-networks
  - 19-proximal-policy-optimization
exercise: 31-evaluating-rl
---

# Evaluating reinforcement learning

Train the same RL algorithm twice with different random seeds, and you can get
two very different agents. One balances the pole for the full episode. The
other never learns. The seed decides the network's starting weights, the
exploration noise, and the order of every minibatch, and deep RL is sensitive
to all of them.

Now picture a paper that runs a new method three times, runs the old method
three times, and reports that the new mean is higher. With spread like that,
the comparison says almost nothing. Agarwal and colleagues (2021) called the
situation a *statistical precipice*. Re-analysing published results, they
found rankings that the evidence didn't support.

This lesson builds the fix they proposed, a small version of their `rliable`
library. It uses robust summary numbers, pictures of the whole score
distribution, and confidence intervals from a bootstrap that respects tasks.

## What you already know

- A seed fixes a random number generator, so a run is reproducible, but a
  different seed is a different experiment.
- The mean is sensitive to outliers. The median isn't, but it ignores most of
  the data.
- A percentile is the value below which a given share of the data falls.
- A confidence interval is a range that, over repeated experiments, contains
  the true value a stated share of the time.
- PPO and DQN results vary widely across seeds (lessons 12 and 19).

## See how little three runs tell you

Henderson and colleagues (2018) showed that two groups of five runs of the
*same* algorithm, differing only in their seeds, can produce learning curves
whose confidence bands don't overlap. In the exercise's demonstration, two
algorithms draw their scores from identical distributions on eight tasks.
With three runs each, algorithm A beats B by mean score in 56% of 200 simulated
experiments. A coin would do about as well at picking the winner, because
there is no winner to pick.

The cure isn't only more runs, though that helps. It's reporting how
uncertain each number is, and choosing numbers that aren't dominated by one
lucky or unlucky run.

## Arrange the scores as a matrix

The results of an evaluation form a *score matrix* $x$ with shape `(n_runs,
n_tasks)`. Entry $x_{n,m}$ is the score of run $n$ on task $m$, with $N$ runs
and $M$ tasks. Scores are *normalised* so that tasks are comparable. On Atari,
for example, a random policy scores 0 and a human reference scores 1. This
is `rliable`'s convention, and every function in the exercise takes it.

The following matrix is the worked example for the rest of the lesson. It
holds four runs on two tasks, and run 4 on task 2 is an outlier:

| | Task 1 | Task 2 |
|---|---|---|
| Run 1 | 0.2 | 1.1 |
| Run 2 | 0.9 | 0.3 |
| Run 3 | 0.4 | 0.8 |
| Run 4 | 0.5 | 5.0 |

## Aggregate with the interquartile mean

The *interquartile mean* (IQM) pools all $NM$ scores, discards the lowest
quarter and the highest quarter, and averages the middle half:

$$
\text{IQM}(x) = \frac{1}{NM - 2k} \sum_{i = k + 1}^{NM - k} x_{(i)}, \qquad k = \lfloor NM / 4 \rfloor
$$

Read it aloud: sort the pooled scores into $x_{(1)} \le x_{(2)} \le \dots$,
drop $k$ from each end, and average the rest. When $NM$ isn't a multiple of 4,
$k$ rounds down, so slightly more than half the scores stay. This is exactly
`scipy.stats.trim_mean(x, 0.25, axis=None)`, which `rliable` uses:

```python
x = np.sort(scores.ravel())
k = int(0.25 * x.size)
iqm = x[k : x.size - k].mean()
```

In the example, the sorted scores are 0.2, 0.3, 0.4, 0.5, 0.8, 0.9, 1.1, and
5.0. With $k = 2$, the IQM averages 0.4, 0.5, 0.8, and 0.9, giving 0.65. The
mean is 1.15, pulled up by the single 5.0. Raise the outlier to 500 and the
mean explodes, while the IQM stays at 0.65.

Agarwal and colleagues prefer the IQM to the median of per-task scores. The
median is robust too, but it rests on one or two numbers and swings a lot
when runs are few. The IQM keeps half the data, so its intervals are much
narrower.

The *optimality gap* asks a different question: how far do runs fall short
of a target, ignoring anything above it?

$$
\text{Gap}(x) = \gamma - \frac{1}{NM} \sum_{n, m} \min(x_{n,m}, \gamma)
$$

Here $\gamma$ is a score threshold, usually 1, and not the discount. Capping
each score at 1 gives 0.2, 1, 0.9, 0.3, 0.4, 0.8, 0.5, and 1, whose mean is
0.6375. So the gap is 0.3625. Lower is better, and a superhuman score on one
task can't hide a failure on another.

> [!KEY]
> The IQM keeps the middle half of all runs pooled across tasks. It resists
> outliers like the median, but it uses enough data to be precise.

## Show the whole distribution: performance profiles

A single number hides shape. A *performance profile* shows, for every
threshold $\tau$, the fraction of runs that score above it:

$$
\hat F(\tau) = \frac{1}{NM} \sum_{n, m} \mathbb{1}\big[x_{n,m} > \tau\big]
$$

Read it aloud: count the runs, over all tasks, that beat $\tau$, and divide by
the total. The inequality is strict. With the same number of runs on every
task, this equals the average over tasks of each task's fraction, which is how
Agarwal and colleagues define it. In numpy, compare every score with every
threshold at once:

```python
profile = (x.ravel()[None, :] > taus[:, None]).mean(axis=1)
```

In the example, $\hat F(0.5) = 4/8 = 0.5$: the 0.5 itself doesn't count. And
$\hat F(1) = 2/8 = 0.25$. The profile starts at 1 and falls to 0. For
non-negative scores, the area under it equals the mean score. If one
algorithm's profile lies above another's at every $\tau$, the first
*stochastically dominates* the second. That's a much stronger claim than a
higher mean.

A related number is the *probability of improvement*, the chance that a random
run of X beats a random run of Y on a random task:

$$
P(X > Y) = \frac{1}{M} \sum_{m=1}^{M} \frac{1}{NK} \sum_{i=1}^{N} \sum_{j=1}^{K} S(x_{i,m}, y_{j,m})
$$

Read it aloud: on each task, compare every run of X with every run of Y. A win
scores 1, a tie scores $\tfrac{1}{2}$, and a loss scores 0. Average over the
pairs, then over the tasks. $S$ is that score, and $K$ is Y's number of runs.
Two identical algorithms give $\tfrac{1}{2}$. For $x = (1, 2)$ and
$y = (2, 0)$ on one task, the four pairs score 0, 1, $\tfrac{1}{2}$, and 1, so
$P(X > Y) = 0.625$.

## Put an interval on it: the stratified bootstrap

You have one score matrix. You'd like to know how much the IQM would move if
you reran the whole experiment. The *bootstrap* pretends that the runs you
have are the population. It builds many fake experiments by redrawing runs
with replacement, and recomputes the statistic on each one. The spread of
those *replicates* estimates the spread you'd see on reruns.

*Stratified* means you redraw runs within each task separately. Every
replicate still holds $N$ runs of every task, so tasks never get mixed or
dropped. Without stratifying, a replicate could hold no runs of a hard task,
and the interval would mostly measure the difference between tasks rather
than the noise within them.

```python
idx = rng.integers(0, n_runs, size=(n_runs, n_tasks))
resample = scores[idx, np.arange(n_tasks)[None, :]]      # column m reads only task m
reps = np.array([iqm(resample_once()) for _ in range(n_boot)])
low, high = np.percentile(reps, [2.5, 97.5])
```

The last line gives a 95% *percentile interval*: the replicates' 2.5th and
97.5th percentiles. `rliable` defaults to 50,000 replicates, the percentile
method, and 95% confidence.

### Check the bootstrap by hand

For the mean, you can work out the bootstrap's answer exactly. Each task's mean
is resampled independently. The variance of a resampled mean of $N$ draws is
$\sigma_m^2 / N$, where $\sigma_m^2$ is the task's variance with divisor $N$.
The grand mean averages $M$ task means, so

$$
\operatorname{Var}^*\big(\bar x^*\big) = \frac{1}{M^2} \sum_{m=1}^{M} \frac{\sigma_m^2}{N}
$$

Read it aloud: add each task's variance divided by its number of runs, then
divide by the number of tasks squared. In the example, task 1 has mean 0.5 and
$\sigma_1^2 = 0.065$. Task 2 has mean 1.8 and $\sigma_2^2 = 3.495$. So the
variance is $(0.065/4 + 3.495/4) / 4 = 0.2225$, a standard deviation of 0.47.
Almost all of it comes from the outlier task. With 20,000 replicates, the
exercise's bootstrap reproduces that variance within 5%.

A second check needs no arithmetic. If every run agrees within each task, every
replicate equals the original matrix, and the interval collapses to a point.
That's true even when tasks differ from each other, which is exactly what a
bootstrap that pools tasks gets wrong.

> [!KEY]
> Redraw runs within each task, recompute the statistic, and read off the tail
> percentiles. Stratifying keeps every task in every replicate, so the interval
> measures run-to-run noise, not the gap between tasks.

> [!WARNING]
> A bootstrap interval from a handful of runs is too narrow. It uses the
> spread of the runs you happen to have, which tends to underestimate the
> truth. In the exercise, a nominal 95% interval for the mean covers the true
> mean about 90% of the time with 10 runs per task. With 3 runs it does worse.

> [!RECAP]
> - Scores are an `(n_runs, n_tasks)` matrix. The IQM pools them and averages
>   the middle half, dropping $\lfloor NM/4 \rfloor$ from each end.
> - The optimality gap is the threshold minus the mean of scores capped at it.
> - A performance profile is the fraction of runs strictly above each $\tau$.
> - The stratified bootstrap redraws runs within each task, and the percentile
>   interval reads the replicates' tails.

## Build it step by step

The checks test the functions in this order.

1. **`iqm(scores)`.** Sort the flattened scores and slice. Common mistake:
   taking the IQM of each task and averaging, which isn't what `rliable`
   computes. The check pools a `(4, 2)` matrix and expects 0.65.
2. **`optimality_gap(scores, threshold)`.** One line with `np.minimum`.
3. **`performance_profile(scores, taus)`.** Broadcast the comparison. Common
   mistake: `>=` instead of `>`, which the check at $\tau = 0.5$ catches.
4. **`probability_of_improvement(scores_x, scores_y)`.** Loop over tasks and
   build the pairwise table with broadcasting. The two inputs can have
   different numbers of runs.
5. **`stratified_resample(scores, rng)`.** One call to `rng.integers`, then
   fancy indexing with a column index. Common mistake: `rng.choice(scores.ravel())`,
   which mixes tasks. The check confirms that each column only holds its own
   task's scores.
6. **`bootstrap_replicates(scores, statistic, n_boot, seed)`.** Make one
   generator and reuse it for every resample. Common mistake: reseeding inside
   the loop, which makes every replicate identical.
7. **`percentile_ci(replicates, confidence)`.** Convert the confidence to two
   percentiles between 0 and 100, and call `np.percentile`.

## Cost and scaling

The expensive part of evaluation is running the agents, not the statistics.
The following table gives the cost of each summary once you have the score
matrix.

| Computation | Time | Notes |
|---|---|---|
| IQM | $O(NM \log NM)$ | One sort |
| Performance profile at $T$ thresholds | $O(TNM)$ | Sort once for $O(NM \log NM + T \log NM)$ |
| Probability of improvement | $O(MNK)$ | All pairs on each task |
| Stratified bootstrap | $O(B \cdot \text{statistic})$ | $B$ replicates, often 2,000 to 50,000 |

Interval width shrinks like $1 / \sqrt{N}$: four times the runs halve it. More
replicates don't narrow the interval. They only make its endpoints less noisy.

## Where this breaks

**Few runs.** With three runs per task, no statistic rescues you. Report the
interval anyway, and say plainly that it is wide and probably too narrow.

**Tuning on the evaluation seeds.** If you pick hyperparameters, checkpoints,
or the best run by the numbers you report, the numbers are biased upward.
Tune on separate seeds, and report every run you trained.

**Different evaluation protocols.** Scores measured after different numbers of
steps, with different episode limits, or with different sticky-action settings
aren't comparable, whatever the statistics say.

**Normalisation choices.** Changing the reference scores changes which tasks
dominate a mean. The IQM and profiles are less sensitive, but not immune.

**Many comparisons.** Comparing ten methods pairwise with 95% intervals gives
several spurious "wins" by chance. Look at the profiles and at the probability
of improvement, and treat small differences as ties.

**Language-model evaluation.** Reasoning benchmarks move by several points
with the seed, the sampling temperature, and the prompt format (Hochlehnert
and colleagues, 2025). Lesson 38 builds the unbiased pass@k estimator for this
setting.

## Further reading

- [Agarwal, Schwarzer, Castro, Courville, and Bellemare (2021), Deep
  Reinforcement Learning at the Edge of the Statistical
  Precipice](https://arxiv.org/abs/2108.13264). The IQM, performance profiles,
  probability of improvement, and stratified bootstrap, with the `rliable`
  library.
- [Henderson and colleagues (2018), Deep Reinforcement Learning that
  Matters](https://arxiv.org/abs/1709.06560). Shows how seeds, hyperparameters,
  and codebases change deep RL results.
- [Colas, Sigaud, and Oudeyer (2018), How Many Random
  Seeds?](https://arxiv.org/abs/1806.08295). Power analysis for choosing the
  number of runs.
- [Jordan and colleagues (2020), Evaluating the Performance of Reinforcement
  Learning Algorithms](https://arxiv.org/abs/2006.16958). Evaluation that
  accounts for hyperparameter sensitivity, not only the best setting.
- [Patterson, Neumann, White, and White (2024), Empirical Design in
  Reinforcement Learning](https://arxiv.org/abs/2304.01315). A long practical
  guide to designing RL experiments.
- [Hochlehnert and colleagues (2025), A Sober Look at Progress in Language
  Model Reasoning](https://arxiv.org/abs/2504.07086). The same problems,
  measured on reasoning benchmarks.

Open questions a researcher might work on:

- How should you evaluate an algorithm's sensitivity to its hyperparameters,
  so that a method that needs heavy tuning isn't ranked above one that doesn't?
- Which interval methods keep their stated coverage with three to five runs,
  where the percentile bootstrap is known to be too narrow?
- How do you compare agents fairly when a single training run costs as much as
  a large language model's post-training?

## The exercise

You build a small `rliable`. Write `iqm`, `optimality_gap`,
`performance_profile`, and `probability_of_improvement` for the point
estimates. Then write `stratified_resample`, `bootstrap_replicates`, and
`percentile_ci` for the intervals.

The checks confirm the worked example's IQM of 0.65, with and without a huge
outlier, and a hand-computed trimmed mean on matrices whose size isn't a
multiple of 4. They check the optimality gap and the profile at hand-picked
thresholds, and that the profile's area equals the mean. The probability of
improvement must match a hand computation with ties and satisfy
$P(X > Y) + P(Y > X) = 1$. The resampler must keep each task's runs in its own
column. The bootstrap's variance must match the exact value 0.2225, its
interval must collapse when runs agree within tasks, and a 95% interval must
cover the true mean in at least 166 of 200 simulated experiments.
