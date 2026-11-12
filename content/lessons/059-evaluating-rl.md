---
title: Evaluating RL honestly
slug: 059-evaluating-rl
part: "Part 8 — Rewards, evaluation, and many agents"
summary: Report RL results that survive a rerun with the interquartile mean, a performance profile, and a stratified bootstrap interval.
minutes: 35
gpu: false
objectives:
  - Explain why three seeds and a mean cannot separate two algorithms, and measure how often the "winner" is only luck.
  - Compute the interquartile mean and a performance profile from a matrix of run scores, and say what each hides.
  - Build a stratified bootstrap and a percentile interval, and check its variance against the exact formula.
  - Say what an interval does and does not promise when you have only a handful of runs.
prereqs:
  - 026-deep-q-networks
  - 036-proximal-policy-optimization
exercise: 059-evaluating-rl
---

# Evaluating RL honestly

You train a candidate algorithm three times and the old baseline three times. The candidate's average score is higher, so you write "our method is better." Would a second group of three runs say the same?

Often it would not. The random seed sets the network's starting weights, the exploration noise, and the order of every minibatch, and deep RL is sensitive to all three. Henderson and colleagues (2018) showed that two groups of five runs of the *same* algorithm can give learning curves whose confidence bands do not overlap. Agarwal and colleagues (2021) called this the *statistical precipice*.

This chapter builds a small version of their fix, the `rliable` library: summary numbers that one lucky run cannot move, a picture of the whole score distribution, and an interval that says how much the answer would change on a rerun.

## What you already know

- A seed fixes a random number generator, so a different seed is a different experiment.
- The mean moves when one value is extreme. The median does not, but it ignores most of the data.
- A percentile is the value below which a given share of the data falls.
- A confidence interval is a range that, over repeated experiments, contains the true value a stated share of the time.
- DQN and PPO results vary widely across seeds.

## Three runs are a coin flip

The obvious method is to run each algorithm a few times and compare the means. Figure 1 shows why that fails. It reruns one algorithm twelve times and plots each rerun's mean score, once with 3 runs per task and once with 30. With 3 runs, the same algorithm reports anything from 0.71 to 1.02, while its true mean score is 0.85. With 30 runs, the reports shrink to 0.81 through 0.92.

```figure
<svg viewBox="0 0 600 230" role="img" aria-label="Mean score reported by twelve reruns of one algorithm, with 3 runs per task and with 30">
<line x1="110.0" y1="40.0" x2="110.0" y2="180.0" class="stroke-grid"/>
<text x="110.0" y="198.0" class="fig-small" text-anchor="middle">0.6</text>
<line x1="202.0" y1="40.0" x2="202.0" y2="180.0" class="stroke-grid"/>
<text x="202.0" y="198.0" class="fig-small" text-anchor="middle">0.7</text>
<line x1="294.0" y1="40.0" x2="294.0" y2="180.0" class="stroke-grid"/>
<text x="294.0" y="198.0" class="fig-small" text-anchor="middle">0.8</text>
<line x1="386.0" y1="40.0" x2="386.0" y2="180.0" class="stroke-grid"/>
<text x="386.0" y="198.0" class="fig-small" text-anchor="middle">0.9</text>
<line x1="478.0" y1="40.0" x2="478.0" y2="180.0" class="stroke-grid"/>
<text x="478.0" y="198.0" class="fig-small" text-anchor="middle">1.0</text>
<line x1="570.0" y1="40.0" x2="570.0" y2="180.0" class="stroke-grid"/>
<text x="570.0" y="198.0" class="fig-small" text-anchor="middle">1.1</text>
<text x="340.0" y="220.0" class="fig-small" text-anchor="middle">mean normalized score reported by one rerun</text>
<text x="100.0" y="84.0" class="fig-bold" text-anchor="end">3 runs</text>
<line x1="110.0" y1="80.0" x2="570.0" y2="80.0" class="stroke-grid"/>
<circle cx="208.4" cy="80.0" r="5" class="fill-b"/>
<circle cx="222.1" cy="80.0" r="5" class="fill-b"/>
<circle cx="271.0" cy="80.0" r="5" class="fill-b"/>
<circle cx="274.4" cy="80.0" r="5" class="fill-b"/>
<circle cx="313.7" cy="80.0" r="5" class="fill-b"/>
<circle cx="394.7" cy="80.0" r="5" class="fill-b"/>
<circle cx="418.4" cy="80.0" r="5" class="fill-b"/>
<circle cx="426.6" cy="80.0" r="5" class="fill-b"/>
<circle cx="472.4" cy="80.0" r="5" class="fill-b"/>
<circle cx="480.3" cy="80.0" r="5" class="fill-b"/>
<circle cx="489.0" cy="80.0" r="5" class="fill-b"/>
<circle cx="493.9" cy="80.0" r="5" class="fill-b"/>
<text x="100.0" y="144.0" class="fig-bold" text-anchor="end">30 runs</text>
<line x1="110.0" y1="140.0" x2="570.0" y2="140.0" class="stroke-grid"/>
<circle cx="306.1" cy="140.0" r="5" class="fill-a"/>
<circle cx="312.5" cy="140.0" r="5" class="fill-a"/>
<circle cx="330.0" cy="140.0" r="5" class="fill-a"/>
<circle cx="341.5" cy="140.0" r="5" class="fill-a"/>
<circle cx="344.4" cy="140.0" r="5" class="fill-a"/>
<circle cx="345.2" cy="140.0" r="5" class="fill-a"/>
<circle cx="348.0" cy="140.0" r="5" class="fill-a"/>
<circle cx="361.4" cy="140.0" r="5" class="fill-a"/>
<circle cx="366.2" cy="140.0" r="5" class="fill-a"/>
<circle cx="387.1" cy="140.0" r="5" class="fill-a"/>
<circle cx="399.3" cy="140.0" r="5" class="fill-a"/>
<circle cx="407.9" cy="140.0" r="5" class="fill-a"/>
<line x1="343.9" y1="40.0" x2="343.9" y2="180.0" class="stroke-ink dashed"/>
<text x="349.9" y="34.0" class="fig-small" text-anchor="start">true mean 0.85</text>
</svg>
Figure 1. Twelve reruns of the same algorithm. With 3 runs per task the reported mean ranges from 0.71 to 1.02, and with 30 runs from 0.81 to 0.92. Notice that a gap of 0.1 between two algorithms is well inside the 3-run spread.
```

The exercise sets up a fair test. Algorithms A and B draw their scores from the *identical* distribution on eight tasks, so there is no winner to find. With three runs each, A still beats B by mean score in 56% of 200 simulated experiments, and each of those papers would have printed a ranking.

More runs help, and the cure has two parts. Pick summary numbers that one lucky run cannot dominate, and report how uncertain each number is.

## Arrange the scores as a matrix

The results of an evaluation form a *score matrix* $x$ with shape `(n_runs, n_tasks)`. Entry $x_{n,m}$ is the score of run $n$ on task $m$, so there are $N$ runs and $M$ tasks. Scores are *normalized* so that tasks are comparable: on Atari, a random policy scores 0 and a human reference scores 1.

The worked example for the rest of the chapter has four runs on two tasks. Run 4 on task 2 is an outlier:

| | Task 1 | Task 2 |
|---|---|---|
| Run 1 | 0.2 | 1.1 |
| Run 2 | 0.9 | 0.3 |
| Run 3 | 0.4 | 0.8 |
| Run 4 | 0.5 | 5.0 |

## Pool the scores and trim: the interquartile mean

The mean of these eight scores is 1.15, and the single 5.0 supplies most of it. If a bug turned that run into 500, the mean would explode to 63.0 and tell you nothing about the other seven runs.

The median would resist, but it rests on one or two numbers, so it swings when runs are few. The *interquartile mean* (IQM) sits between them. Pool all $NM$ scores, discard the lowest quarter and the highest quarter, and average the middle half. Sort the example: 0.2, 0.3, 0.4, 0.5, 0.8, 0.9, 1.1, 5.0. Drop two from each end, and average 0.4, 0.5, 0.8, and 0.9. The IQM is 0.65, and it stays 0.65 when the outlier becomes 500.

```figure
<svg viewBox="0 0 600 260" role="img" aria-label="Eight pooled scores sorted, with the lowest two and highest two trimmed and the IQM of 0.65 against the mean of 1.15">
<line x1="70.0" y1="210.0" x2="570.0" y2="210.0" class="stroke-grid"/>
<text x="62.0" y="214.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="174.0" x2="570.0" y2="174.0" class="stroke-grid"/>
<text x="62.0" y="178.0" class="fig-small" text-anchor="end">1</text>
<line x1="70.0" y1="138.0" x2="570.0" y2="138.0" class="stroke-grid"/>
<text x="62.0" y="142.0" class="fig-small" text-anchor="end">2</text>
<line x1="70.0" y1="102.0" x2="570.0" y2="102.0" class="stroke-grid"/>
<text x="62.0" y="106.0" class="fig-small" text-anchor="end">3</text>
<line x1="70.0" y1="66.0" x2="570.0" y2="66.0" class="stroke-grid"/>
<text x="62.0" y="70.0" class="fig-small" text-anchor="end">4</text>
<line x1="70.0" y1="30.0" x2="570.0" y2="30.0" class="stroke-grid"/>
<text x="62.0" y="34.0" class="fig-small" text-anchor="end">5</text>
<rect x="79.2" y="202.8" width="44.0" height="7.2" class="fill-sunk"/>
<rect x="79.2" y="202.8" width="44" height="7.2" class="stroke-muted" fill="none"/>
<text x="101.2" y="228.0" class="fig-small" text-anchor="middle">0.2</text>
<rect x="141.8" y="199.2" width="44.0" height="10.8" class="fill-sunk"/>
<rect x="141.8" y="199.2" width="44" height="10.8" class="stroke-muted" fill="none"/>
<text x="163.8" y="228.0" class="fig-small" text-anchor="middle">0.3</text>
<rect x="204.2" y="195.6" width="44.0" height="14.4" class="fill-a"/>
<text x="226.2" y="228.0" class="fig-small" text-anchor="middle">0.4</text>
<rect x="266.8" y="192.0" width="44.0" height="18.0" class="fill-a"/>
<text x="288.8" y="228.0" class="fig-small" text-anchor="middle">0.5</text>
<rect x="329.2" y="181.2" width="44.0" height="28.8" class="fill-a"/>
<text x="351.2" y="228.0" class="fig-small" text-anchor="middle">0.8</text>
<rect x="391.8" y="177.6" width="44.0" height="32.4" class="fill-a"/>
<text x="413.8" y="228.0" class="fig-small" text-anchor="middle">0.9</text>
<rect x="454.2" y="170.4" width="44.0" height="39.6" class="fill-sunk"/>
<rect x="454.2" y="170.4" width="44" height="39.6" class="stroke-muted" fill="none"/>
<text x="476.2" y="228.0" class="fig-small" text-anchor="middle">1.1</text>
<rect x="516.8" y="30.0" width="44.0" height="180.0" class="fill-sunk"/>
<rect x="516.8" y="30.0" width="44" height="180.0" class="stroke-muted" fill="none"/>
<text x="538.8" y="228.0" class="fig-small" text-anchor="middle">5</text>
<line x1="70.0" y1="168.6" x2="570.0" y2="168.6" class="stroke-b dashed"/>
<text x="76.0" y="162.6" class="fig-small fig-bold" text-anchor="start">mean 1.15</text>
<line x1="70.0" y1="186.6" x2="570.0" y2="186.6" class="stroke-c"/>
<text x="76.0" y="180.6" class="fig-small fig-bold" text-anchor="start">IQM 0.65</text>
<text x="195.0" y="24.0" class="fig-small" text-anchor="start">kept: the middle half</text>
<text x="81.2" y="24.0" class="fig-small" text-anchor="start">trimmed</text>
<text x="446.2" y="24.0" class="fig-small" text-anchor="start">trimmed</text>
<text x="320.0" y="250.0" class="fig-small" text-anchor="middle">the eight pooled scores of the worked example, sorted</text>
</svg>
Figure 2. The two lowest and two highest of the eight scores are discarded, and the shaded four are averaged. Notice that the lone 5.0 drags the mean to 1.15 but cannot move the IQM, which stays at 0.65.
```

In symbols, with $x_{(i)}$ the $i$-th smallest pooled score:

$$
\text{IQM}(x) = \frac{1}{NM - 2k} \sum_{i = k + 1}^{NM - k} x_{(i)}, \qquad k = \lfloor NM / 4 \rfloor
$$

Read it aloud: sort the pooled scores, drop $k$ from each end, and average the rest. When $NM$ is not a multiple of 4, $k$ rounds down. This equals `scipy.stats.trim_mean(x, 0.25, axis=None)`:

```python
x = np.sort(scores.ravel())
k = int(0.25 * x.size)
iqm = x[k : x.size - k].mean()
```

> [!KEY]
> The IQM keeps the middle half of all runs, pooled across tasks. It resists outliers like the median, but it uses enough data to give narrower intervals than the median does.

## Draw the whole distribution: performance profiles

One number hides shape. Two algorithms can share an IQM while one fails often and the other never does. A *performance profile* answers a different question for every threshold $\tau$: what fraction of runs scored above it?

In the example, $\tau = 0.5$ is beaten by four of eight scores (0.8, 0.9, 1.1, 5.0), and the 0.5 itself does not count, so the fraction is 0.5. At $\tau = 1$, two of eight beat it, so the fraction is 0.25. The profile starts at 1 and falls to 0, and Figure 3 draws it.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Performance profile of the worked example: the fraction of scores above each threshold, falling from 1 to 0.125 by threshold 1.1">
<line x1="80.0" y1="220.0" x2="560.0" y2="220.0" class="stroke-grid"/>
<text x="72.0" y="224.0" class="fig-small" text-anchor="end">0</text>
<line x1="80.0" y1="172.5" x2="560.0" y2="172.5" class="stroke-grid"/>
<text x="72.0" y="176.5" class="fig-small" text-anchor="end">0.25</text>
<line x1="80.0" y1="125.0" x2="560.0" y2="125.0" class="stroke-grid"/>
<text x="72.0" y="129.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="80.0" y1="77.5" x2="560.0" y2="77.5" class="stroke-grid"/>
<text x="72.0" y="81.5" class="fig-small" text-anchor="end">0.75</text>
<line x1="80.0" y1="30.0" x2="560.0" y2="30.0" class="stroke-grid"/>
<text x="72.0" y="34.0" class="fig-small" text-anchor="end">1</text>
<line x1="80.0" y1="220.0" x2="80.0" y2="225.0" class="stroke-muted"/>
<text x="80.0" y="240.0" class="fig-small" text-anchor="middle">0</text>
<line x1="240.0" y1="220.0" x2="240.0" y2="225.0" class="stroke-muted"/>
<text x="240.0" y="240.0" class="fig-small" text-anchor="middle">0.5</text>
<line x1="400.0" y1="220.0" x2="400.0" y2="225.0" class="stroke-muted"/>
<text x="400.0" y="240.0" class="fig-small" text-anchor="middle">1</text>
<line x1="560.0" y1="220.0" x2="560.0" y2="225.0" class="stroke-muted"/>
<text x="560.0" y="240.0" class="fig-small" text-anchor="middle">1.5</text>
<line x1="80.0" y1="220.0" x2="560.0" y2="220.0" class="stroke-muted"/>
<polyline points="80.0,30.0 144.0,30.0 144.0,53.8 176.0,53.8 176.0,77.5 208.0,77.5 208.0,101.2 240.0,101.2 240.0,125.0 336.0,125.0 336.0,148.8 368.0,148.8 368.0,172.5 432.0,172.5 432.0,196.2 560.0,196.2" class="stroke-a" fill="none"/>
<circle cx="240.0" cy="125.0" r="5" class="fill-b"/>
<text x="248.0" y="117.0" class="fig-small fig-bold" text-anchor="start">F(0.5) = 0.5</text>
<circle cx="400.0" cy="172.5" r="5" class="fill-b"/>
<text x="408.0" y="194.5" class="fig-small fig-bold" text-anchor="start">F(1) = 0.25</text>
<text x="556.0" y="188.2" class="fig-small" text-anchor="end">stays at 0.125 until 5.0</text>
<text x="320.0" y="262.0" class="fig-small" text-anchor="middle">threshold τ</text>
<text x="80.0" y="20.0" class="fig-small" text-anchor="start">fraction of scores above τ</text>
</svg>
Figure 3. The profile of the eight example scores. Notice that the score exactly at 0.5 does not count as above 0.5, so F(0.5) is 4/8, and that the 5.0 outlier adds only one-eighth of height, however large it grows.
```

In symbols, with $\mathbb{1}[\cdot]$ equal to 1 when its condition holds and 0 otherwise:

$$
\hat F(\tau) = \frac{1}{NM} \sum_{n, m} \mathbb{1}\big[x_{n,m} > \tau\big]
$$

Read it aloud: count the runs, over all tasks, that beat $\tau$, and divide by the total. The inequality is strict. In numpy, compare every score with every threshold at once:

```python
profile = (x.ravel()[None, :] > taus[:, None]).mean(axis=1)
```

For non-negative scores, the area under the profile equals the mean score, so the profile contains the mean and more. If one algorithm's profile lies above another's at every $\tau$, the first *stochastically dominates* the second. That is a much stronger claim than a higher mean.

## Put an interval on it: the stratified bootstrap

You have one score matrix and want to know how far the IQM would move if you reran the whole experiment. You cannot afford the reruns, so the *bootstrap* pretends that the runs you have are the population. It builds many fake experiments by redrawing runs with replacement, and recomputes the statistic on each one. The spread of those *replicates* estimates the spread you would see on real reruns.

The obvious way to redraw is to sample from all eight scores at once. That mixes tasks. A replicate could hold no runs of a hard task, and the interval would mostly measure the gap between tasks rather than the noise within them. *Stratified* resampling fixes this by redrawing runs within each task's column. Every replicate still holds $N$ runs of every task:

```python
idx = rng.integers(0, n_runs, size=(n_runs, n_tasks))
resample = scores[idx, np.arange(n_tasks)[None, :]]   # column m reads only task m
reps = np.array([iqm(resample_once()) for _ in range(n_boot)])
low, high = np.percentile(reps, [2.5, 97.5])
```

The last line is a 95% *percentile interval*: the replicates' 2.5th and 97.5th percentiles.

### Check the bootstrap by hand

For the mean, you can work out the bootstrap's spread exactly. Each task's mean is resampled independently, and the variance of a resampled mean of $N$ draws is $\sigma_m^2 / N$, where $\sigma_m^2$ is that task's variance with divisor $N$. The grand mean averages $M$ task means, so

$$
\operatorname{Var}^*\big(\bar x^*\big) = \frac{1}{M^2} \sum_{m=1}^{M} \frac{\sigma_m^2}{N}
$$

Read it aloud: add each task's variance divided by its number of runs, then divide by the number of tasks squared. In the example, task 1 has $\sigma_1^2 = 0.065$ and task 2 has $\sigma_2^2 = 3.495$. So the variance is $(0.065/4 + 3.495/4) / 4 = 0.2225$, a standard deviation of 0.47, almost all of it from the outlier task. With 20,000 replicates, the exercise's bootstrap reproduces that variance within 5%.

A second check needs no arithmetic. If every run agrees within each task, every replicate equals the original matrix and the interval collapses to a point, even when the tasks differ from each other. A bootstrap that pools tasks gets this wrong.

> [!KEY]
> Redraw runs within each task, recompute the statistic, and read off the tail percentiles. Stratifying keeps every task in every replicate, so the interval measures run-to-run noise, not the gap between tasks.

> [!WARNING]
> A bootstrap interval from a handful of runs is too narrow, because the spread of the runs you happen to have tends to underestimate the truth. In the exercise, a nominal 95% interval covers the true mean about 90% of the time with 10 runs per task, and less with 3.

> [!RECAP]
> - Scores form an `(n_runs, n_tasks)` matrix. The IQM pools them and averages the middle half, dropping $\lfloor NM/4 \rfloor$ from each end.
> - A performance profile is the fraction of runs strictly above each $\tau$.
> - The stratified bootstrap redraws runs within each task, and the percentile interval reads the replicates' tails.
> - Three runs cannot separate two algorithms, and an interval from few runs is still too narrow.
- Compare fairly: equal tuning budget and size, fresh seeds for the final run, and one ablation per component with its own interval.

## Compare fairly and ablate

A tight interval does not make a comparison fair. Henderson and colleagues (2018) showed that hyperparameters, codebases, and seeds can change a result as much as the algorithm does. Three habits remove most of that.

**Give the baseline the same budget.** If you tune your method over 50 configurations and the baseline over 3, the gap measures effort. Give both the same number of tuning runs, the same search ranges, and the same parameter count. A method with twice the network width has an advantage that has nothing to do with its idea.

**Rerun on fresh seeds.** You chose settings by looking at some seeds, so those seeds favor you. Freeze the settings, then run the final experiment on seeds you have never used, and report what comes out, even when it is worse. Run the baseline on the same fresh seeds.

**Ablate one component at a time.** If your method adds a target network and reward scaling, train a version without each one, changing only that. A component earns its place only when its removal costs score by more than noise.

The bootstrap that you built gives that noise. A component's *contribution* is the full method's mean score minus the ablation's. The full runs and the ablation runs are different runs, so each replicate redraws the two groups independently, and the percentile interval comes from the differences.

Take six seeds of the full method, and six of each ablation:

| | Mean score | Contribution | 95% interval |
|---|---|---|---|
| Full method | 0.855 | | |
| No target network | 0.748 | 0.107 | [0.040, 0.173] |
| No reward scaling | 0.847 | 0.008 | [-0.053, 0.072] |

The target network's interval stays above zero, so it helps. The reward scaling's interval contains zero, so six seeds cannot tell whether it does anything. Do not claim it as part of your method. With few seeds, treat even a positive interval as evidence and not as proof.

## Build it step by step

The checks test the functions in this order.

1. **`iqm(scores)`.** Sort the flattened scores and slice. The common mistake is taking the IQM of each task and averaging, which is not what `rliable` computes. The check pools a `(4, 2)` matrix and expects 0.65.
2. **`performance_profile(scores, taus)`.** Broadcast the comparison. The common mistake is `>=` instead of `>`, which the check at $\tau = 0.5$ catches.
3. **`stratified_resample(scores, rng)`.** One call to `rng.integers`, then fancy indexing with a column index. The common mistake is `rng.choice(scores.ravel())`, which mixes tasks. The check confirms that each column holds only its own task's scores.
4. **`bootstrap_replicates(scores, statistic, n_boot, seed)`.** Make one generator and reuse it for every resample. The common mistake is reseeding inside the loop, which makes every replicate identical.
5. **`percentile_ci(replicates, confidence)`.** Convert the confidence to two percentiles between 0 and 100, and call `np.percentile`.
6. **`ablation_contributions(full, ablations, n_boot, alpha, seed)`.** Return `{name: (estimate, low, high)}`, where `estimate` is `mean(full) - mean(ablation)`. For each component, draw separate index arrays for `full` and for that ablation. The common mistake is one shared index array, which treats seed $i$ of each group as the same run, or resampling only `full`. Both give intervals that are too narrow.

## Where this breaks

**Few runs.** With three runs per task, no statistic rescues you. Report the interval anyway, and say plainly that it is wide and probably too narrow.

**Tuning on the evaluation seeds.** If you pick hyperparameters, checkpoints, or the best run by the numbers you report, those numbers are biased upward. Tune on separate seeds, as the previous section describes.

**Different protocols.** Scores measured after different numbers of steps or with different episode limits are not comparable, whatever the statistics say.

**Debugging a run in progress.** Statistics report a finished experiment. To find out why a run is failing, read its entropy, KL, and explained variance, which [Debugging RL runs](/lessons/060-debugging-rl-runs) builds.

**Other summaries.** `rliable` also defines the optimality gap and the probability of improvement. Both are worth reporting next to the IQM.

**Language-model evaluation.** Reasoning benchmarks move by several points with the seed, the sampling temperature, and the prompt format (Hochlehnert and colleagues, 2025). [Reasoning with verifiable rewards](/lessons/073-reasoning-with-verifiable-rewards) builds the unbiased pass@k estimator for this setting.

## Further reading

- [Agarwal, Schwarzer, Castro, Courville, and Bellemare (2021), Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264). The IQM, performance profiles, probability of improvement, and stratified bootstrap, with the `rliable` library.
- [Henderson and colleagues (2018), Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560). Shows how seeds, hyperparameters, and codebases change deep RL results. [Paper page](https://papers.ifkash.dev/henderson/).
- [Patterson, Neumann, White, and White (2024), Empirical Design in Reinforcement Learning](https://arxiv.org/abs/2304.01315). A long practical guide to designing RL experiments.
- [Hochlehnert and colleagues (2025), A Sober Look at Progress in Language Model Reasoning](https://arxiv.org/abs/2504.07086). The same problems, measured on reasoning benchmarks.

Open questions a researcher might work on:

- Which interval methods keep their stated coverage with three to five runs, where the percentile bootstrap is known to be too narrow?
- How do you compare agents fairly when a single training run costs as much as a large language model's post-training?

## The exercise

You build a small `rliable`. Write `iqm` and `performance_profile` for the point estimates, then `stratified_resample`, `bootstrap_replicates`, and `percentile_ci` for the intervals. Then write `ablation_contributions(full, ablations, n_boot, alpha, seed)`, which takes the full method's per-seed scores and a dict that maps each component's name to its ablation's per-seed scores.

The checks confirm the worked example's IQM of 0.65, with and without a huge outlier, and the profile at hand-picked thresholds. The resampler must keep each task's runs in its own column. The bootstrap's variance must match the exact value 0.2225, its interval must collapse when runs agree within tasks, and a 95% interval must cover the true mean in at least 166 of 200 simulated experiments.

`ablation_contributions` must return, for each component, a tuple of the estimate and the interval's low and high ends. The estimate must equal the difference of means. The interval must match a large reference bootstrap, contain the estimate, and widen with noisier scores. A check feeds it groups where pairing indices gives a zero-width interval, and another where bootstrapping only the full method gives one that is too narrow.
