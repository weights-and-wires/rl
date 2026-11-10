---
title: Gradient bandits
slug: 006-gradient-bandits
part: "Part 1 — Learning from reward"
summary: Learn which arm to pull by nudging arm preferences uphill on expected reward, and see why a baseline cuts the noise without changing the direction.
minutes: 35
gpu: false
objectives:
  - Turn arm preferences into a policy with a softmax that cannot overflow.
  - Apply the gradient bandit update by hand, and say what each factor in it does.
  - Show that the update's average is the exact gradient of expected reward, for any baseline.
  - Explain why a baseline leaves the average alone but makes learning faster.
prereqs:
  - 003-explore-or-exploit
exercise: 006-gradient-bandits
---

# Gradient bandits

Every strategy so far works in two steps. It estimates how good each arm is, then picks an arm by a rule such as epsilon-greedy. Suppose you skip the first step. You keep a number for each arm that says how much you like it, you pull arms with probabilities that follow those numbers, and after each reward you nudge the numbers: up for an arm that paid well, down for one that paid badly.

Two questions follow. How big should each nudge be, and in which direction? The obvious answer is to add a fixed amount to the probability of the arm that just paid. That breaks quickly. Probabilities must stay between 0 and 1 and add up to one, so raising one probability forces you to lower the others by amounts you must compute, and a few lucky pulls can push a probability below zero.

This chapter fixes the problem by nudging a different set of numbers and letting a softmax turn them into probabilities. You then see that, on average, the nudges climb the exact gradient of expected reward. That makes the gradient bandit the smallest policy gradient method, and the same idea fine-tunes language models later in the course.

## What you already know

- A bandit arm pays a random reward with an unknown mean, and your goal is a large total reward (see [Explore or exploit](/lessons/003-explore-or-exploit)).
- An incremental mean updates a running average without storing every reward.
- A softmax turns a vector of scores into probabilities that sum to one.
- Gradient ascent moves parameters a small step along the gradient to raise an objective.
- The average of many random draws approaches their expected value.

## Preferences instead of probabilities

Give each arm $a$ a number $H(a)$, its *preference*. A preference can be any real number, so you can raise or lower it freely. A softmax turns the preferences into the policy $\pi(a)$, the probability of pulling arm $a$:

$$
\pi(a) = \frac{e^{H(a)}}{\sum_b e^{H(b)}}
$$

Read it as: exponentiate every preference, then divide by the total so the probabilities sum to one. With $H = (0, 0, 0)$, every arm gets $1/3$. Raise arm 0's preference to 1, and the policy becomes $(0.576, 0.212, 0.212)$.

Only differences between preferences matter, because adding 50 to every $H(a)$ multiplies each exponential by $e^{50}$, and that cancels in the division. So subtract the largest preference before you exponentiate. The result is the same, and nothing overflows:

```python
z = np.exp(H - H.max())
pi = z / z.sum()
```

## One nudge, by hand

You pull arm $a_t$ and receive reward $r_t$. You need a rule that raises $H(a_t)$ when the reward is good and lowers it when the reward is bad. "Good" needs a reference level, so call it the *baseline* $b$: a reward above $b$ is good news, and a reward below $b$ is bad news. Then update every arm's preference:

$$
H(a) \leftarrow H(a) + \alpha\,(r_t - b)\,\bigl(\mathbf{1}\{a = a_t\} - \pi(a)\bigr)
$$

Read it piece by piece:

- $\alpha$ is the step size.
- $r_t - b$ is the size of the news, with its sign.
- $\mathbf{1}\{a = a_t\}$ is 1 for the arm you pulled and 0 for every other arm.
- For the pulled arm, the last factor is $1 - \pi(a_t)$, which is positive, so good news raises it. An arm that was unlikely moves more, because $1 - \pi$ is larger.
- For every other arm, the factor is $-\pi(a)$, so good news lowers it in proportion to how likely it was.

In numpy, the whole update is one line:

```python
H = H + alpha * (reward - baseline) * (np.eye(k)[arm] - pi)
```

Three arms start at $H = (0, 0, 0)$, so $\pi = (1/3, 1/3, 1/3)$. You pull arm 0 and receive $r = 1$. The baseline is 0 and $\alpha = 0.1$:

- Arm 0: $0 + 0.1 \times 1 \times (1 - 1/3) = 0.0667$.
- Arms 1 and 2: $0 + 0.1 \times 1 \times (0 - 1/3) = -0.0333$ each.

The softmax of $(0.0667, -0.0333, -0.0333)$ is $(0.3559, 0.3220, 0.3220)$. One good reward moves about two percentage points of probability toward arm 0.

```figure
<svg viewBox="0 0 600 305" role="img" aria-label="Bars of the three arm probabilities before and after one update: arm 0 rises from 0.333 to 0.356, the others fall to 0.322">
<line x1="70.0" y1="182.5" x2="560.0" y2="182.5" class="stroke-grid"/>
<line x1="70.0" y1="135.0" x2="560.0" y2="135.0" class="stroke-grid"/>
<line x1="70.0" y1="87.5" x2="560.0" y2="87.5" class="stroke-grid"/>
<text x="62.0" y="234.0" class="fig-small" text-anchor="end">0.0</text>
<text x="62.0" y="186.5" class="fig-small" text-anchor="end">0.1</text>
<text x="62.0" y="139.0" class="fig-small" text-anchor="end">0.2</text>
<text x="62.0" y="91.5" class="fig-small" text-anchor="end">0.3</text>
<text x="62.0" y="44.0" class="fig-small" text-anchor="end">0.4</text>
<line x1="70.0" y1="230.0" x2="560.0" y2="230.0" class="stroke-muted"/>
<text x="300.0" y="18.0" class="fig-bold" text-anchor="middle">Probability of each arm, before and after one pull of arm 0 that paid 1</text>
<rect x="99.7" y="71.7" width="48.0" height="158.3" class="fill-muted"/>
<rect x="155.7" y="60.9" width="48.0" height="169.1" class="fill-a"/>
<text x="123.7" y="65.7" class="fig-small" text-anchor="middle">0.333</text>
<text x="179.7" y="54.9" class="fig-small" text-anchor="middle">0.356</text>
<text x="151.7" y="252.0" class="fig-bold" text-anchor="middle">arm 0</text>
<text x="151.7" y="270.0" class="fig-small" text-anchor="middle">H: 0 → +0.0667</text>
<rect x="263.0" y="71.7" width="48.0" height="158.3" class="fill-muted"/>
<rect x="319.0" y="77.0" width="48.0" height="153.0" class="fill-b"/>
<text x="287.0" y="65.7" class="fig-small" text-anchor="middle">0.333</text>
<text x="343.0" y="71.0" class="fig-small" text-anchor="middle">0.322</text>
<text x="315.0" y="252.0" class="fig-bold" text-anchor="middle">arm 1</text>
<text x="315.0" y="270.0" class="fig-small" text-anchor="middle">H: 0 → -0.0333</text>
<rect x="426.3" y="71.7" width="48.0" height="158.3" class="fill-muted"/>
<rect x="482.3" y="77.0" width="48.0" height="153.0" class="fill-b"/>
<text x="450.3" y="65.7" class="fig-small" text-anchor="middle">0.333</text>
<text x="506.3" y="71.0" class="fig-small" text-anchor="middle">0.322</text>
<text x="478.3" y="252.0" class="fig-bold" text-anchor="middle">arm 2</text>
<text x="478.3" y="270.0" class="fig-small" text-anchor="middle">H: 0 → -0.0333</text>
<rect x="80.0" y="284.0" width="12.0" height="12.0" class="fill-muted"/>
<text x="98.0" y="294.0" class="fig-small" text-anchor="start">before</text>
<rect x="160.0" y="284.0" width="12.0" height="12.0" class="fill-a"/>
<text x="178.0" y="294.0" class="fig-small" text-anchor="start">after, pulled arm</text>
<rect x="310.0" y="284.0" width="12.0" height="12.0" class="fill-b"/>
<text x="328.0" y="294.0" class="fig-small" text-anchor="start">after, other arms</text>
</svg>
Figure 1. One reward of 1 for arm 0 raises its preference by 0.0667 and lowers the other two by 0.0333 each, which moves the policy from 0.333 to 0.356 on arm 0. The nudge is small, so learning takes many pulls.
```

Notice that the three changes add up to zero: $0.0667 - 0.0333 - 0.0333 = 0$. That is no accident. The factors $\mathbf{1}\{a = a_t\} - \pi(a)$ sum to $1 - 1 = 0$ over all arms. The next section uses that fact.

## Why the nudges climb the gradient

Each nudge uses one noisy reward, so it can point the wrong way. The surprise is that, averaged over the arm you pull and the reward you get, it points exactly along the gradient of expected reward. That is why this is gradient *ascent* and not a heuristic.

Write the objective as $J(H) = \sum_x \pi(x)\, q(x)$, where $q(x)$ is arm $x$'s unknown mean reward. Differentiating the softmax gives $\partial \pi(x) / \partial H(a) = \pi(x)\,(\mathbf{1}\{a = x\} - \pi(a))$, so:

$$
\frac{\partial J}{\partial H(a)} = \sum_x q(x)\,\pi(x)\,\bigl(\mathbf{1}\{a = x\} - \pi(a)\bigr) = \pi(a)\,\bigl(q(a) - J\bigr)
$$

Read it as: raise the preference of an arm that pays more than the policy's current average $J$, lower it for an arm that pays less, and scale by how often you pull the arm. In numpy, that is `pi * (q - pi @ q)`. With $H = (0, 0, 0)$ and $q = (1, 2, 3)$, the average is $J = 2$ and the gradient is $\tfrac13(1 - 2,\ 2 - 2,\ 3 - 2) = (-1/3, 0, 1/3)$.

Now average the update itself. You pull arm $x$ with probability $\pi(x)$, and it pays $q(x)$ on average. So the expected update is $\sum_x \pi(x)\,(q(x) - b)\,(\mathbf{1}\{a = x\} - \pi(a))$ for the preference of arm $a$. The $q(x)$ part is the gradient above. The $b$ part is $b$ times $\sum_x \pi(x)(\mathbf{1}\{a = x\} - \pi(a)) = \pi(a) - \pi(a) = 0$. It vanishes, which means any baseline that does not depend on which arm you just pulled leaves the average update unchanged.

There is a compact way to say the same thing. The factor $\mathbf{1}\{a = a_t\} - \pi(a)$ is the gradient of $\ln \pi(a_t)$ with respect to $H(a)$. So the update is the step size times the news times the gradient of the log-probability of the arm you chose. Keep that shape in mind. It returns in [The policy gradient and REINFORCE](/lessons/031-the-policy-gradient) for whole sequences of actions.

> [!KEY]
> The gradient bandit update is, on average, the exact gradient of expected reward. A baseline changes nothing on average, because the direction $\mathbf{1}\{a = a_t\} - \pi(a)$ averages to zero.

## Why use a baseline at all

If the baseline changes nothing on average, why bother? Because the average hides the noise, and the noise is what slows you down. Suppose every arm pays about $+4$, as in many real systems where rewards are never near zero. Take $H = (0, 0, 0)$, pull arm 1, and receive $4.1$.

- With $b = 0$: arm 1 moves by $0.1 \times 4.1 \times (2/3) = 0.273$, and each other arm moves by $-0.137$.
- With $b = 4$: arm 1 moves by $0.1 \times 0.1 \times (2/3) = 0.0067$.

Without a baseline, every pull makes a large nudge toward whichever arm you happened to pull, good or bad. A pulled arm gets raised, so it gets pulled more, so it gets raised more. The small differences between arms, the signal, hide under the shared $+4$. A baseline removes the shared part and leaves the differences.

The usual choice, from Sutton and Barto's Section 2.8, is the average of all rewards received *before* this pull, starting at 0. Figure 2 compares the two on a 10-arm bandit with means near $+4$. In steps 500 to 1000, averaged over 20 runs, the best arm gets 98.6 percent of the pulls with the baseline and 59.2 percent without it.

```figure
<svg viewBox="0 0 600 285" role="img" aria-label="Line chart of how often the best arm is pulled: with a baseline the share reaches about 99 percent, without one it stays near 60 percent">
<line x1="70.0" y1="230.0" x2="570.0" y2="230.0" class="stroke-grid"/>
<text x="62.0" y="234.0" class="fig-small" text-anchor="end">0%</text>
<line x1="70.0" y1="180.0" x2="570.0" y2="180.0" class="stroke-grid"/>
<text x="62.0" y="184.0" class="fig-small" text-anchor="end">25%</text>
<line x1="70.0" y1="130.0" x2="570.0" y2="130.0" class="stroke-grid"/>
<text x="62.0" y="134.0" class="fig-small" text-anchor="end">50%</text>
<line x1="70.0" y1="80.0" x2="570.0" y2="80.0" class="stroke-grid"/>
<text x="62.0" y="84.0" class="fig-small" text-anchor="end">75%</text>
<line x1="70.0" y1="30.0" x2="570.0" y2="30.0" class="stroke-grid"/>
<text x="62.0" y="34.0" class="fig-small" text-anchor="end">100%</text>
<line x1="70.0" y1="230.0" x2="570.0" y2="230.0" class="stroke-muted"/>
<line x1="70.0" y1="30.0" x2="70.0" y2="230.0" class="stroke-muted"/>
<line x1="70.0" y1="230.0" x2="70.0" y2="235.0" class="stroke-muted"/>
<text x="70.0" y="250.0" class="fig-small" text-anchor="middle">0</text>
<line x1="195.0" y1="230.0" x2="195.0" y2="235.0" class="stroke-muted"/>
<text x="195.0" y="250.0" class="fig-small" text-anchor="middle">250</text>
<line x1="320.0" y1="230.0" x2="320.0" y2="235.0" class="stroke-muted"/>
<text x="320.0" y="250.0" class="fig-small" text-anchor="middle">500</text>
<line x1="445.0" y1="230.0" x2="445.0" y2="235.0" class="stroke-muted"/>
<text x="445.0" y="250.0" class="fig-small" text-anchor="middle">750</text>
<line x1="570.0" y1="230.0" x2="570.0" y2="235.0" class="stroke-muted"/>
<text x="570.0" y="250.0" class="fig-small" text-anchor="middle">1000</text>
<text x="320.0" y="270.0" class="fig-small" text-anchor="middle">step</text>
<text x="14" y="130.0" class="fig-small" text-anchor="middle" transform="rotate(-90 14 130.0)">best arm pulled</text>
<polyline points="82.5,197.6 107.5,142.4 132.5,85.8 157.5,67.0 182.5,53.8 207.5,50.0 232.5,47.2 257.5,43.8 282.5,41.4 307.5,38.6 332.5,34.8 357.5,35.6 382.5,32.4 407.5,33.2 432.5,31.0 457.5,32.6 482.5,32.6 507.5,32.4 532.5,31.8 557.5,31.8" class="stroke-a" fill="none"/>
<polyline points="82.5,192.8 107.5,165.6 132.5,156.6 157.5,139.2 182.5,133.6 207.5,132.0 232.5,130.6 257.5,124.0 282.5,122.2 307.5,121.0 332.5,118.6 357.5,119.2 382.5,114.4 407.5,118.8 432.5,119.4 457.5,110.6 482.5,104.6 507.5,105.0 532.5,101.8 557.5,103.0" class="stroke-b" fill="none"/>
<circle cx="82.5" cy="197.6" r="2.5" class="fill-a"/>
<circle cx="107.5" cy="142.4" r="2.5" class="fill-a"/>
<circle cx="132.5" cy="85.8" r="2.5" class="fill-a"/>
<circle cx="157.5" cy="67.0" r="2.5" class="fill-a"/>
<circle cx="182.5" cy="53.8" r="2.5" class="fill-a"/>
<circle cx="207.5" cy="50.0" r="2.5" class="fill-a"/>
<circle cx="232.5" cy="47.2" r="2.5" class="fill-a"/>
<circle cx="257.5" cy="43.8" r="2.5" class="fill-a"/>
<circle cx="282.5" cy="41.4" r="2.5" class="fill-a"/>
<circle cx="307.5" cy="38.6" r="2.5" class="fill-a"/>
<circle cx="332.5" cy="34.8" r="2.5" class="fill-a"/>
<circle cx="357.5" cy="35.6" r="2.5" class="fill-a"/>
<circle cx="382.5" cy="32.4" r="2.5" class="fill-a"/>
<circle cx="407.5" cy="33.2" r="2.5" class="fill-a"/>
<circle cx="432.5" cy="31.0" r="2.5" class="fill-a"/>
<circle cx="457.5" cy="32.6" r="2.5" class="fill-a"/>
<circle cx="482.5" cy="32.6" r="2.5" class="fill-a"/>
<circle cx="507.5" cy="32.4" r="2.5" class="fill-a"/>
<circle cx="532.5" cy="31.8" r="2.5" class="fill-a"/>
<circle cx="557.5" cy="31.8" r="2.5" class="fill-a"/>
<circle cx="82.5" cy="192.8" r="2.5" class="fill-b"/>
<circle cx="107.5" cy="165.6" r="2.5" class="fill-b"/>
<circle cx="132.5" cy="156.6" r="2.5" class="fill-b"/>
<circle cx="157.5" cy="139.2" r="2.5" class="fill-b"/>
<circle cx="182.5" cy="133.6" r="2.5" class="fill-b"/>
<circle cx="207.5" cy="132.0" r="2.5" class="fill-b"/>
<circle cx="232.5" cy="130.6" r="2.5" class="fill-b"/>
<circle cx="257.5" cy="124.0" r="2.5" class="fill-b"/>
<circle cx="282.5" cy="122.2" r="2.5" class="fill-b"/>
<circle cx="307.5" cy="121.0" r="2.5" class="fill-b"/>
<circle cx="332.5" cy="118.6" r="2.5" class="fill-b"/>
<circle cx="357.5" cy="119.2" r="2.5" class="fill-b"/>
<circle cx="382.5" cy="114.4" r="2.5" class="fill-b"/>
<circle cx="407.5" cy="118.8" r="2.5" class="fill-b"/>
<circle cx="432.5" cy="119.4" r="2.5" class="fill-b"/>
<circle cx="457.5" cy="110.6" r="2.5" class="fill-b"/>
<circle cx="482.5" cy="104.6" r="2.5" class="fill-b"/>
<circle cx="507.5" cy="105.0" r="2.5" class="fill-b"/>
<circle cx="532.5" cy="101.8" r="2.5" class="fill-b"/>
<circle cx="557.5" cy="103.0" r="2.5" class="fill-b"/>
<text x="570.0" y="62.0" class="fig-small" text-anchor="end">with baseline: 98.6% in steps 500–1000</text>
<text x="570.0" y="138.0" class="fig-small" text-anchor="end">no baseline: 59.2% in steps 500–1000</text>
</svg>
Figure 2. Share of pulls that go to the best arm, averaged over 20 runs of a 10-arm bandit whose means sit near +4 (step size 0.1). Both versions climb the same gradient on average, but without a baseline each nudge carries the shared +4, and learning lags.
```

> [!WARNING]
> Update the running mean of rewards *after* you use it as the baseline. If the baseline already includes the current reward, it depends on which arm you pulled, and the argument that the baseline vanishes no longer holds exactly.

> [!RECAP]
> - `softmax` subtracts the maximum before `np.exp`.
> - The step is `alpha * (r - b) * (np.eye(k)[arm] - pi)`, and it moves every arm.
> - The step's average over your own arm choices is `pi * (q - pi @ q)`, for every baseline.
> - The baseline is the mean of earlier rewards, updated after each use.

## Build it step by step

The checks test the functions in this order.

1. **`softmax(H)`.** Takes `(k,)` preferences and returns `(k,)` probabilities. Common mistake: skipping the max subtraction. The check passes preferences of 1000.
2. **`gradient_bandit_update(H, arm, reward, baseline, alpha)`.** Returns new preferences and leaves `H` unchanged. Compute `pi` from the old `H` first. Common mistake: updating only the pulled arm and forgetting the $-\pi(a)$ push on the others.
3. **`expected_update(H, q, baseline)`.** Loop over arms $x$, call your update with `alpha=1.0` and reward `q[x]`, subtract `H`, weight by `pi[x]`, and sum. The check compares the result with finite differences of $J(H) = \pi \cdot q$ for three baselines, so a mistake in step 2 shows up here.
4. **`run_gradient_bandit(bandit, T, alpha, use_baseline, seed)`.** Draw arms with `rng.choice(k, p=softmax(H))`. Keep the running mean reward, use it as the baseline, and then fold in the new reward. Return the `(T,)` array of arms.

## Where this breaks

**A step size that is too large.** A few lucky steps can push one preference far above the rest, and the softmax then almost never tries another arm. Smaller steps, or a bonus that rewards spread-out policies, keep it exploring.

**A baseline that peeks.** A baseline that uses the current reward biases the update, as the warning above says.

**Arms whose means drift.** The mean of every past reward adapts slowly when the arms change over time. An exponentially weighted average of recent rewards follows the drift better.

## Further reading

- [Sutton and Barto, *Reinforcement Learning: An Introduction*, Section 2.8](http://incompleteideas.net/book/the-book-2nd.html) derives the gradient bandit as stochastic gradient ascent and shows the baseline experiment.
- [Williams, "Simple statistical gradient-following algorithms for connectionist reinforcement learning" (1992)](https://doi.org/10.1007/BF00992696) introduces REINFORCE, of which the gradient bandit is the one-state case.
- [Greensmith, Bartlett, and Baxter, "Variance reduction techniques for gradient estimates in reinforcement learning" (2004)](https://jmlr.org/papers/v5/greensmith04a.html) studies baselines as a way to cut variance.
- [Mei, Xiao, Szepesvári, and Schuurmans, "On the global convergence rates of softmax policy gradient methods" (2020)](https://arxiv.org/abs/2005.06392) proves when softmax policy gradient reaches the best arm, and how fast.

## The exercise

You build the gradient bandit from the softmax up. Write `softmax`, `gradient_bandit_update`, `expected_update`, and `run_gradient_bandit`.

The checks verify that `softmax` ignores a shared shift and survives preferences of 1000, and that `gradient_bandit_update` reproduces the worked step above, leaves `H` alone, and keeps the preferences' sum fixed. They check that `expected_update` equals a finite-difference gradient of expected reward for three different baselines, including the one with $q = (1, 2, 3)$. Last, they check that on arms with means near $+4$, a run with a baseline takes over the best arm, and a run without one lags well behind.
