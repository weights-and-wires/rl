---
title: "Verifiable rewards"
slug: 068-verifiable-rewards
part: "Part 9 — RL for language models"
summary: "Score math and code with a check a program can grade, and drop reward groups that give GRPO nothing to learn."
minutes: 40
gpu: false
objectives:
  - Normalize a math answer so fractions, decimals, units, and a boxed span compare as the same value.
  - Score a format bonus separately from correctness, and show when that bonus gives a wrong answer a positive advantage.
  - Run a code reward in a subprocess under a timeout, and withhold full credit from a program that prints one memorized line.
  - Estimate a pass rate with a Wilson interval and drop reward groups whose rewards are all equal.
prereqs:
  - 067-group-relative-policy-optimization
exercise: 068-verifiable-rewards
---

# Verifiable rewards

A model answers "what is $1/2 + 1/2$?" twice. One reply says the answer is 0.5 meters. The other writes `\boxed{1/2}`. A check that compares the reply with the string `1/2` scores both of them 0.

The two rewards match, so the group advantage in [Group-relative policy optimization](/lessons/067-group-relative-policy-optimization) is 0 and the update does nothing. The check scored two names for one quantity as two failures. This chapter builds a reward that treats them as the same answer, pays a small bonus for putting that answer in a box, runs a program against tests under a timeout, and drops groups whose rewards cannot move the policy. [Reasoning with verifiable rewards](/lessons/069-reasoning-with-verifiable-rewards) then measures how often a model is right when it may try several times.

## What you already know

- GRPO's advantage is the reward minus the group's mean, divided by the group's standard deviation. Equal rewards make that advantage 0, from [Group-relative policy optimization](/lessons/067-group-relative-policy-optimization).
- A prompt is $x$, a response is $y$, and $r(x, y)$ is one number for the whole response.
- $1/2$ and $0.5$ are the same rational, and $2.50$ is $5/2$.
- A confidence interval is a range for an unknown probability, from [Evaluating RL honestly](/lessons/056-evaluating-rl). Here the probability is a prompt's pass rate.

## One string is not one answer

A string compare marks the spelling of the key, so the model learns to copy `1/2` and gets nothing for `0.5`. Parse the value instead.

Take the last `\boxed{...}` span. A later box is the answer the model emitted after an earlier guess. Strip `$`, `\left`, and `\right`. Expand `\frac{a}{b}` into a ratio. Delete commas that sit between digits, so `1,000` and `1000` match. Parse the rest with an exact rational, and write it in lowest terms. `2.50`, `2.5`, and `5/2` all become `5/2`. `1.0` becomes `1`.

A trailing unit is a suffix. `meters` and `m` become `meter`, `kg` becomes `kilogram`, `s` becomes `second`, and `%` becomes `percent`. When both answers carry a unit, the units have to match, so `5 meters` is not `5 kg`. When only one side has a unit, compare the numbers. Keys are often a bare number, and the model repeats the unit from the question.

`0.333` stays `333/1000`. It is a different rational from `1/3`. Rounding to three decimals would mark a wrong answer right.

The response `\boxed{2.50 meters}` against the target `5/2` is the running example. The box holds `2.50 meters`, which becomes `5/2 meter`. The target has no unit. The numbers match, so the answers are equivalent and the format-plus-correctness reward below scores $1.1$.

The exercise checks thirty pairs. These are the pattern:

| Left | Right | Equivalent |
|---|---|---|
| `1/2` | `0.5` | yes |
| `\frac{1}{2}` | `1/2` | yes |
| `2.50` | `5/2` | yes |
| `1.0` | `1` | yes |
| `5 meters` | `5` | yes |
| `5 meters` | `5 kg` | no |
| `1/3` | `0.333` | no |

```python
from fractions import Fraction
Fraction("2.50")  # Fraction(5, 2), rendered as "5/2"
```

> [!KEY]
> Compare answers after you reduce them to one rational and one unit. A string compare pays the model for copying the key's spelling.

## Pay for the box, then pay for the answer

DeepSeek-R1-Zero adds two rule rewards: an accuracy reward on a boxed math answer, or on a compiler's test cases, and a format reward on the requested tags. Tülu 3 named this reinforcement learning with verifiable rewards (RLVR) and pays a constant for a verified answer, otherwise 0. DAPO pays $+1$ or $-1$ on the extracted answer and skips a format term.

Split the score into two bits. $f = 1$ when the response contains `\boxed{`, and $0$ otherwise. $c = 1$ when that boxed span matches the target, and $0$ otherwise. A right number outside a box scores 0. The check has to know where the answer is, or a reply that lists every integer from 0 to 100 collects the credit.

$$
r = c + \lambda f
$$

$c$ is correctness. $f$ is the format bit. $\lambda$ is the format credit. At $\lambda = 0.1$ a correct box scores $1.1$, a wrong box scores $0.1$, and a missing box scores $0$.

Take eight responses to the target `1`. One is `\boxed{1}`, five are `\boxed{9}`, and two have no box. The rewards are $1 + \lambda$ once, $\lambda$ five times, and $0$ twice, so the mean is $(1 + 6\lambda)/8$. The centered advantage of a wrong box is $\lambda - (1 + 6\lambda)/8 = (2\lambda - 1)/8$.

At $\lambda = 0.1$ the mean is $0.2$. The correct box sits at $+0.9$, each wrong box at $-0.1$, and each missing box at $-0.2$. GRPO divides by the standard deviation next. The sign stays, and the correct answer is the one pushed up.

At $\lambda = 0.9$ the mean is $0.8$ and each wrong box sits at $+0.1$. In this group the sign flips once $\lambda > 1/2$. A different mix of replies moves that threshold.

The other trap needs no large $\lambda$. Eight wrong boxes score $\lambda$ every time. Every centered advantage is 0, and dividing by the standard deviation is still 0. The group adds no gradient. A format bonus that every sample already earns is a constant, and a constant inside a GRPO group teaches nothing.

> [!WARNING]
> Once a format bonus beats the group mean, a wrong boxed answer has a positive advantage, and GRPO makes that wrong answer more likely.

## Run the program, and stop it

A math answer is a string you parse. A code answer is a program you run. The reward is the fraction of tests that pass. Each test carries an stdin and an expected stdout. Write the program into a temporary directory, start it with `python -I`, and stop it when it runs past the timeout. The course runner is a resource boundary, not a security boundary, and this check is the same kind of boundary: a timeout and a subprocess, not a sandbox.

Strip whitespace at both ends of stdout. A non-zero exit fails the test. The reward is passes divided by the number of tests.

The adding program reads two integers and prints the sum. On stdin `2 3` and `10 1` it prints `5` and `11`, both tests pass, and the reward is $1$. A program that only prints `5` passes the first test and fails the second, so the reward is $0.5$. Full credit takes every test, and the tests have to expect different lines. One memorized `print` is a solution only for the test that happens to expect that line.

An infinite loop prints nothing. With a timeout of a fraction of a second, each test fails and the reward is $0$, and the call returns. Score the loop inside this process and the check waits until the outer limit.

> [!WARNING]
> Printing one expected line passes every test that expects that line. A second test with a different output is what keeps that program below full credit.

## Keep the prompts that can teach

Some prompts come back all right or all wrong. Every reward is then the same number, the advantage is the zero vector, and the group adds no gradient, as with the eight wrong boxes. DAPO keeps a group only when the number correct is strictly between $0$ and the group size, and it samples until the batch is full of those groups. For a binary reward that is "the rewards differ." With a format bonus, eight copies of $0.1$ are as empty as eight zeros, so drop a group when its rewards are all equal.

The mask has one boolean per reward. The effective batch size is how many entries stay. These sixteen rewards, in groups of four,

$$
(1, 0, 1, 0),\ (0, 0, 0, 0),\ (1, 1, 1, 1),\ (1, 0, 0, 1),
$$

keep the first group and the last. The effective batch size is $8$.

You also want a range on a prompt's pass rate before you spend a run on it. Draw $n$ samples, see $k$ successes, and set $\hat p = k/n$. The normal approximation $\hat p \pm z\sqrt{\hat p(1-\hat p)/n}$ has width $0$ when $k = 0$. It reports a pass rate of exactly $0$. Zero successes in eight tries can still sit next to a pass rate near a third.

The Wilson score interval keeps a term that survives at the boundary. For a 95% interval, $z = 1.96$:

$$
\mathrm{center} = \frac{\hat p + z^{2}/(2n)}{1 + z^{2}/n}, \qquad \mathrm{half} = \frac{z}{1 + z^{2}/n}\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^{2}}{4n^{2}}}.
$$

The interval is $\mathrm{center} \pm \mathrm{half}$. The center pulls $\hat p$ toward $1/2$, less so as $n$ grows. The extra $z^{2}/(4n^{2})$ under the square root is why zero successes still has a positive upper end. Clip each end into $[0, 1]$, since a pass rate lives there and rounding can step a hair outside.

Three successes out of eight give $\hat p = 0.375$ and the interval $0.136842$ to $0.694262$. Zero successes out of eight give $0$ to $0.324416$.

```figure
<svg viewBox="0 0 620 330" role="img" aria-label="Wilson intervals for 0 to 32 successes out of 32 samples, with the band from 0.1 to 0.9 shaded">

<rect x="162.2" y="38.0" width="353.6" height="230.0" class="fill-soft-c"/>
<line x1="118.0" y1="276.0" x2="560.0" y2="276.0" class="stroke-muted"/>
<line x1="118.0" y1="276.0" x2="118.0" y2="281.0" class="stroke-muted"/>
<text x="118.0" y="296.0" text-anchor="middle" class="fig-small">0</text>
<line x1="162.2" y1="276.0" x2="162.2" y2="281.0" class="stroke-muted"/>
<text x="162.2" y="296.0" text-anchor="middle" class="fig-small">0.1</text>
<line x1="339.0" y1="276.0" x2="339.0" y2="281.0" class="stroke-muted"/>
<text x="339.0" y="296.0" text-anchor="middle" class="fig-small">0.5</text>
<line x1="515.8" y1="276.0" x2="515.8" y2="281.0" class="stroke-muted"/>
<text x="515.8" y="296.0" text-anchor="middle" class="fig-small">0.9</text>
<line x1="560.0" y1="276.0" x2="560.0" y2="281.0" class="stroke-muted"/>
<text x="560.0" y="296.0" text-anchor="middle" class="fig-small">1</text>
<text x="339.0" y="316.0" text-anchor="middle">estimated pass rate</text>
<text x="300" y="22" text-anchor="middle" class="fig-bold">32 samples, Wilson interval, z = 1.96</text>
<text x="110" y="50.0" text-anchor="end" class="fig-small">0/32</text>
<line x1="118.0" y1="46.0" x2="165.4" y2="46.0" class="stroke-bad"/>
<line x1="118.0" y1="41.0" x2="118.0" y2="51.0" class="stroke-bad"/>
<line x1="165.4" y1="41.0" x2="165.4" y2="51.0" class="stroke-bad"/>
<circle cx="118.0" cy="46.0" r="3.5" class="fill-bad"/>
<text x="110" y="76.0" text-anchor="end" class="fig-small">2/32</text>
<line x1="125.7" y1="72.0" x2="207.1" y2="72.0" class="stroke-bad"/>
<line x1="125.7" y1="67.0" x2="125.7" y2="77.0" class="stroke-bad"/>
<line x1="207.1" y1="67.0" x2="207.1" y2="77.0" class="stroke-bad"/>
<circle cx="145.6" cy="72.0" r="3.5" class="fill-bad"/>
<text x="110" y="102.0" text-anchor="end" class="fig-small">4/32</text>
<line x1="140.0" y1="98.0" x2="242.1" y2="98.0" class="stroke-bad"/>
<line x1="140.0" y1="93.0" x2="140.0" y2="103.0" class="stroke-bad"/>
<line x1="242.1" y1="93.0" x2="242.1" y2="103.0" class="stroke-bad"/>
<circle cx="173.2" cy="98.0" r="3.5" class="fill-bad"/>
<text x="110" y="128.0" text-anchor="end" class="fig-small">8/32</text>
<line x1="176.6" y1="124.0" x2="304.1" y2="124.0" class="stroke-c"/>
<line x1="176.6" y1="119.0" x2="176.6" y2="129.0" class="stroke-c"/>
<line x1="304.1" y1="119.0" x2="304.1" y2="129.0" class="stroke-c"/>
<circle cx="228.5" cy="124.0" r="3.5" class="fill-c"/>
<text x="110" y="154.0" text-anchor="end" class="fig-small">16/32</text>
<line x1="266.6" y1="150.0" x2="411.4" y2="150.0" class="stroke-c"/>
<line x1="266.6" y1="145.0" x2="266.6" y2="155.0" class="stroke-c"/>
<line x1="411.4" y1="145.0" x2="411.4" y2="155.0" class="stroke-c"/>
<circle cx="339.0" cy="150.0" r="3.5" class="fill-c"/>
<text x="110" y="180.0" text-anchor="end" class="fig-small">24/32</text>
<line x1="373.9" y1="176.0" x2="501.4" y2="176.0" class="stroke-c"/>
<line x1="373.9" y1="171.0" x2="373.9" y2="181.0" class="stroke-c"/>
<line x1="501.4" y1="171.0" x2="501.4" y2="181.0" class="stroke-c"/>
<circle cx="449.5" cy="176.0" r="3.5" class="fill-c"/>
<text x="110" y="206.0" text-anchor="end" class="fig-small">28/32</text>
<line x1="435.9" y1="202.0" x2="538.0" y2="202.0" class="stroke-bad"/>
<line x1="435.9" y1="197.0" x2="435.9" y2="207.0" class="stroke-bad"/>
<line x1="538.0" y1="197.0" x2="538.0" y2="207.0" class="stroke-bad"/>
<circle cx="504.8" cy="202.0" r="3.5" class="fill-bad"/>
<text x="110" y="232.0" text-anchor="end" class="fig-small">30/32</text>
<line x1="470.9" y1="228.0" x2="552.3" y2="228.0" class="stroke-bad"/>
<line x1="470.9" y1="223.0" x2="470.9" y2="233.0" class="stroke-bad"/>
<line x1="552.3" y1="223.0" x2="552.3" y2="233.0" class="stroke-bad"/>
<circle cx="532.4" cy="228.0" r="3.5" class="fill-bad"/>
<text x="110" y="258.0" text-anchor="end" class="fig-small">32/32</text>
<line x1="512.6" y1="254.0" x2="560.0" y2="254.0" class="stroke-bad"/>
<line x1="512.6" y1="249.0" x2="512.6" y2="259.0" class="stroke-bad"/>
<line x1="560.0" y1="249.0" x2="560.0" y2="259.0" class="stroke-bad"/>
<circle cx="560.0" cy="254.0" r="3.5" class="fill-bad"/>
</svg>
Figure 1. Wilson intervals for 32 samples. The shaded band runs from 0.1 to 0.9. Three intervals lie inside it. A point inside the band is not enough: 4 of 32 estimates 0.125, and the interval 0.0497 to 0.2807 still crosses 0.1.
```

Keep a prompt when the whole interval lies inside $[0.1, 0.9]$. In Figure 1 that is $8$, $16$, and $24$ successes out of $32$. Four successes estimate $0.125$, and the interval $0.0497$ to $0.2807$ crosses $0.1$, so that prompt stays out. `dynamic_sampling_filter` is the same rule on a group you already scored. The interval chooses prompts before the run. The mask drops a group whose rewards came back all equal.

> [!KEY]
> An all-equal reward group has GRPO advantage 0. Drop it. Keep a prompt when the Wilson interval around its pass rate lies inside the band where some samples fail and some succeed.

```python
center = (phat + z * z / (2 * n)) / (1 + z * z / n)
```

> [!RECAP]
> - `normalize_math_answer` reduces the last box to a lowest-terms rational and a canonical unit. `is_equivalent` ignores a unit that only one side has.
> - The reward is $c + \lambda f$. In the group of eight, a wrong box is below the mean at $\lambda = 0.1$ and above it at $\lambda = 0.9$.
> - `code_reward` is the fraction of tests that pass in a subprocess before the timeout. `print(5)` scores $0.5$ on the two-test example, and an infinite loop scores $0$.
> - The Wilson interval for $3$ successes out of $8$ is $0.136842$ to $0.694262$. `dynamic_sampling_filter` drops every all-equal group, and the effective batch size is the number of rewards kept.

## Build it step by step

The checks call the functions in this order. The first passes once `normalize_math_answer` is right.

1. **`normalize_math_answer(text)`.** Count braces to take the last `\boxed{...}`, expand `\frac`, and parse with `fractions.Fraction`. Render an integer with no denominator. Common mistake: leaving `1/2` and `0.5` as different strings, or using `float` so `1/3` and `0.333` collapse.
2. **`is_equivalent(left, right)`.** Normalize both sides, then apply the unit rule. Common mistake: rejecting `5 meters` against `5` because one string still contains a unit.
3. **`format_correctness_reward(response, target, format_credit=0.1)`.** Return `0` when `\boxed{` is absent. Otherwise return `format_credit`, plus `1` on a match. Common mistake: paying `1` for any box, or paying correctness for a right number that is not boxed.
4. **`code_reward(program, tests, timeout)`.** Call `subprocess.run` with `timeout`, `python -I`, and a temporary working directory. Divide passes by the number of tests. Common mistake: running the program in this process, so `while True` never returns.
5. **`wilson_interval(successes, n, z=1.96)`.** Use the center and the half-width above, and clip the ends to $[0, 1]$. Common mistake: the normal approximation, which returns a width of $0$ at zero successes. For $0$ out of $8$ the upper end is $0.324416$.
6. **`dynamic_sampling_filter(rewards, group_size)`.** Walk contiguous groups and keep a group when its max and min differ. Common mistake: keeping a group of equal $0.1$s because it is neither all zeros nor all ones.

## Where this breaks

**The format bonus becomes the task.** A large $\lambda$ gives wrong boxed answers a positive advantage. A bonus every sample already earns is a constant, and the group advantage is 0. Read a batch of rewards before you trust the run.

**Tests in the prompt.** A prompt that includes the tests lets the model print those lines. Keep the tests out of the prompt, and give them different expected outputs.

**A unit that is the answer.** Dropping a missing unit is right when the question already named the unit. It is the wrong rule when `5 meters` and `5` are different answers. The pairs in the exercise follow the first rule.

**A timeout that is too short.** A slow correct program fails the same way a loop does. A loop with no timeout holds the batch. The checks use a timeout of `0.4` or `0.5` seconds, and one process per test, so a looping program costs about a second of wall clock rather than the whole run.

## Further reading

The reward and the filter in this chapter come from the following sources:

- [DeepSeek-AI, "DeepSeek-R1" (2025)](https://arxiv.org/abs/2501.12948): accuracy on a boxed answer or on test cases, plus a format reward.
- [Yu et al., "DAPO" (2025)](https://arxiv.org/abs/2503.14476): drop a group whose accuracy is $0$ or $1$, because its GRPO advantage is zero.
- [Lambert et al., "Tülu 3" (2024)](https://arxiv.org/abs/2411.15124): names RLVR and pays $\alpha = 10$ for a verified answer, otherwise $0$.
- [Lambert, "Reinforcement Learning"](https://rlhfbook.com/c/06-policy-gradients): a group with no contrast has little learning signal, and the format weight is worth sweeping.
- [Lambert, "Evaluation"](https://rlhfbook.com/c/16-evaluation): a math score is exact match after answer extraction.
- [CS336, lecture 16, "Post-training - RLVR"](https://cs336.stanford.edu/): verifiable rewards, and an assignment that trains on a checkable math reward.
- [Kumar, "Reinforcement Learning for LLM Reasoning"](https://cs224r.stanford.edu/spring_2025/slides/10_cs224r-rl_for_reasoning_lecture.pdf): the CS224R guest lecture sets the outcome reward to $1$ on a correct answer and $0$ otherwise.

Open questions that researchers are working on:

- How large can $\lambda$ be, on the prompts in the batch, before a wrong boxed answer gets a positive advantage?
- Which normalizer still disagrees with a mathematician once the answer is a set, a matrix, or an expression?
- How do you cap the extra generations dynamic sampling spends on prompts that keep coming back all right or all wrong?

## The exercise

You write `normalize_math_answer`, `is_equivalent`, and `format_correctness_reward`, then `code_reward`, `wilson_interval`, and `dynamic_sampling_filter`.

The checks compare canonical strings, then thirty pairs. A normalizer that only strips a box, called `literal_normalizer` in the checks, fails `1/2` against `0.5`. They check the rewards $1.1$, $0.1$, and $0$, and the sign of a wrong box's advantage at format credits $0.1$ and $0.9$. They run an adding program, an infinite loop, and `print(5)`. They compare the Wilson ends with the formula, including zero successes. They drop every all-equal group, including a group of $0.1$s, and require an effective batch size of $8$. A filter that keeps every group, called `keep_tied_groups` in the checks, reports $20$.
