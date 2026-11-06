---
title: Value iteration and contraction
slug: 012-value-iteration
part: "Part 2 — Planning with a known world"
summary: Skip the full evaluation by applying the Bellman optimality backup directly, and show that the error shrinks by gamma every sweep.
minutes: 40
gpu: false
objectives:
  - Explain why one sweep of the Bellman optimality backup mixes evaluation and improvement.
  - Run value iteration on the two-state world and read its table of iterates.
  - Show that every sweep shrinks the worst error by at least a factor of gamma.
  - Compute how many sweeps a target accuracy needs, and how small the last change must be before you stop.
prereqs:
  - 010-policy-evaluation
  - 011-policy-iteration
exercise: 012-value-iteration
---

# Value iteration and contraction

Policy iteration solves a linear system in every round to get the exact values of a policy, and then throws that policy away as soon as the greedy step replaces it. That looks wasteful. In the two-room world, a rough guess of the values is already enough to pick the right action: after two sweeps of the update you build in this chapter, the values are $(1.9, 3.8)$, nowhere near the true $(18, 20)$, and yet the greedy action in state 0 is already the best one.

So do you need exact values before you improve? This chapter says no. It applies the improvement idea at every sweep, starting from a guess of zero, and it answers the practical question that follows: how do you know when to stop, and how far from the answer you still are?

## What you already know

- $Q(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a)\, V(s')$, and `rlenvs.q_from_v(mdp, v)` builds the whole table.
- The greedy policy takes the action with the largest $Q$ in each state.
- Evaluating a policy by sweeping repeats $V \leftarrow r_\pi + \gamma P_\pi V$ until it settles.
- Policy iteration reaches $V^* = (18, 20)$ on the two-room world from [Policy iteration](/lessons/011-policy-iteration) in two evaluations, but each one is a linear solve.
- That world has states 0 and 1 and $\gamma = 0.9$. Staying pays 1 in state 0 and 2 in state 1, and going to the other room pays 0.

## Improve inside every sweep

Evaluation by sweeping already updates a value from the values of the next states. Improvement picks the best action. Do both at once: in each sweep, set every state's value to the value of its best action, judged by the current guesses.

$$
V_{k+1}(s) = \max_a \Bigl[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a)\, V_k(s') \Bigr]
$$

Read it as: each state takes the value of its best action, where "best" is measured with the current guess of the next states' values. This is the *Bellman optimality backup*, and this method is *value iteration*. No policy is stored. You read one off at the end with the greedy step. In code, one sweep is one line:

```python
v = rlenvs.q_from_v(mdp, v).max(axis=1)     # one sweep, from the old v
```

Here is the two-state world from $V_0 = (0, 0)$, with the greedy action in state 0 for each iterate:

| $k$ | $V_k(0)$ | $V_k(1)$ | Greedy action in state 0 |
|---|---|---|---|
| 0 | 0 | 0 | stay |
| 1 | $\max(1, 0) = 1$ | $\max(2, 0) = 2$ | stay |
| 2 | $\max(1 + 0.9, 0 + 0.9 \times 2) = 1.9$ | $\max(2 + 0.9 \times 2, 0 + 0.9 \times 1) = 3.8$ | go |
| 3 | $\max(1 + 0.9 \times 1.9, 0.9 \times 3.8) = 3.42$ | $\max(2 + 0.9 \times 3.8, 0.9 \times 1.9) = 5.42$ | go |
| 4 | $\max(1 + 0.9 \times 3.42, 0.9 \times 5.42) = 4.878$ | $\max(2 + 0.9 \times 5.42, 0.9 \times 3.42) = 6.878$ | go |

```figure
<svg viewBox="0 0 570 290" role="img" aria-label="Value iteration on the two-state world: the values of states 0 and 1 climb toward 18 and 20 over 40 sweeps, while the greedy action in state 0 is already go at sweep 2">
<line x1="70.0" y1="240.0" x2="540.0" y2="240.0" class="stroke-grid"/>
<text x="62.0" y="244.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="194.5" x2="540.0" y2="194.5" class="stroke-grid"/>
<text x="62.0" y="198.5" class="fig-small" text-anchor="end">5</text>
<line x1="70.0" y1="149.1" x2="540.0" y2="149.1" class="stroke-grid"/>
<text x="62.0" y="153.1" class="fig-small" text-anchor="end">10</text>
<line x1="70.0" y1="103.6" x2="540.0" y2="103.6" class="stroke-grid"/>
<text x="62.0" y="107.6" class="fig-small" text-anchor="end">15</text>
<line x1="70.0" y1="58.2" x2="540.0" y2="58.2" class="stroke-grid"/>
<text x="62.0" y="62.2" class="fig-small" text-anchor="end">20</text>
<line x1="70.0" y1="240.0" x2="70.0" y2="244.0" class="stroke-muted"/>
<text x="70.0" y="258.0" class="fig-small" text-anchor="middle">0</text>
<line x1="187.5" y1="240.0" x2="187.5" y2="244.0" class="stroke-muted"/>
<text x="187.5" y="258.0" class="fig-small" text-anchor="middle">10</text>
<line x1="305.0" y1="240.0" x2="305.0" y2="244.0" class="stroke-muted"/>
<text x="305.0" y="258.0" class="fig-small" text-anchor="middle">20</text>
<line x1="422.5" y1="240.0" x2="422.5" y2="244.0" class="stroke-muted"/>
<text x="422.5" y="258.0" class="fig-small" text-anchor="middle">30</text>
<line x1="540.0" y1="240.0" x2="540.0" y2="244.0" class="stroke-muted"/>
<text x="540.0" y="258.0" class="fig-small" text-anchor="middle">40</text>
<text x="305.0" y="276.0" text-anchor="middle">sweep k</text>
<line x1="70.0" y1="76.4" x2="540.0" y2="76.4" class="stroke-a dashed"/>
<line x1="70.0" y1="58.2" x2="540.0" y2="58.2" class="stroke-b dashed"/>
<text x="540.0" y="91.4" class="fig-small" text-anchor="end">V*(0) = 18</text>
<text x="540.0" y="52.2" class="fig-small" text-anchor="end">V*(1) = 20</text>
<polyline points="70.0,240.0 81.8,230.9 93.5,222.7 105.2,208.9 117.0,195.7 128.8,183.7 140.5,173.0 152.2,163.3 164.0,154.6 175.8,146.8 187.5,139.8 199.2,133.4 211.0,127.7 222.8,122.6 234.5,118.0 246.2,113.8 258.0,110.1 269.8,106.7 281.5,103.7 293.2,100.9 305.0,98.5 316.8,96.3 328.5,94.3 340.2,92.5 352.0,90.9 363.8,89.4 375.5,88.1 387.2,86.9 399.0,85.9 410.8,84.9 422.5,84.1 434.2,83.3 446.0,82.6 457.8,82.0 469.5,81.4 481.2,80.9 493.0,80.5 504.8,80.1 516.5,79.7 528.2,79.3 540.0,79.1" class="stroke-a"/>
<polyline points="70.0,240.0 81.8,221.8 93.5,205.5 105.2,190.7 117.0,177.5 128.8,165.5 140.5,154.8 152.2,145.1 164.0,136.4 175.8,128.6 187.5,121.6 199.2,115.2 211.0,109.5 222.8,104.4 234.5,99.8 246.2,95.6 258.0,91.9 269.8,88.5 281.5,85.5 293.2,82.7 305.0,80.3 316.8,78.1 328.5,76.1 340.2,74.3 352.0,72.7 363.8,71.2 375.5,69.9 387.2,68.8 399.0,67.7 410.8,66.7 422.5,65.9 434.2,65.1 446.0,64.4 457.8,63.8 469.5,63.2 481.2,62.7 493.0,62.3 504.8,61.9 516.5,61.5 528.2,61.2 540.0,60.9" class="stroke-b"/>
<line x1="93.5" y1="80.0" x2="93.5" y2="240.0" class="stroke-ink dashed"/>
<circle cx="93.5" cy="222.7" r="5" class="fill-a"/>
<text x="101.5" y="90.0" class="fig-small" text-anchor="start">k = 2: the greedy action</text>
<text x="101.5" y="104.0" class="fig-small" text-anchor="start">in state 0 becomes go</text>
<text x="14.0" y="24.0" class="fig-small" text-anchor="start">V_k(0) in blue, V_k(1) in orange</text>
</svg>
Figure 1. Value iteration on the two-state world, from zeros. Solid lines are V_k(0) and V_k(1), and dashed lines are the answers 18 and 20. Notice that the greedy action in state 0 is already the optimal go at k = 2, when V_2(0) is only 1.9.
```

The values climb slowly, but the greedy action settles at sweep 2. The ordering of actions is often right long before the values are. Whether the values themselves arrive, and how fast, is the question for the rest of the chapter.

## Why the error shrinks by gamma each sweep

Write $T$ for one backup, so $V_{k+1} = T V_k$. The answer you want, $V^*$, satisfies $V^* = T V^*$: a sweep leaves it unchanged. Measure the gap between two value vectors by their largest difference over states, written $\lVert V - W \rVert = \max_s |V(s) - W(s)|$. The key fact is that one backup brings any two vectors closer:

$$
\lVert T V - T W \rVert \le \gamma \lVert V - W \rVert
$$

Three observations explain why. First, for one state and one action, the two $Q$ values differ by $\gamma \sum_{s'} P(s' \mid s, a)\,(V(s') - W(s'))$. That is $\gamma$ times an average of differences, and an average is never larger than the largest difference, so it is at most $\gamma \lVert V - W \rVert$. Second, a maximum over actions can't separate two vectors more than the largest gap between the actions' entries: $|\max_a x_a - \max_a y_a| \le \max_a |x_a - y_a|$. Third, combine the two, and the bound holds in every state.

A map with this property is a *contraction* with factor $\gamma$. It has two consequences. It has only one fixed point, because two of them would have to stay the same distance apart while the map shrinks it. So $V^*$ is the only solution of the Bellman optimality equation, which is what the previous chapter needed to call a policy-iteration stopping point optimal. And since $V^* = T V^*$, each sweep shrinks the distance to it:

$$
\lVert V_k - V^* \rVert = \lVert T V_{k-1} - T V^* \rVert \le \gamma \lVert V_{k-1} - V^* \rVert \quad\Rightarrow\quad \lVert V_k - V^* \rVert \le \gamma^k \lVert V_0 - V^* \rVert
$$

Read it as: every sweep cuts the worst error across states by at least a factor $\gamma$, from any starting guess. In the two-state world, $\lVert V_0 - V^* \rVert = 20$, so the errors in the table are 20, 18, 16.2, 14.58, 13.122, exactly $20 \times 0.9^k$. The bound is tight here, because state 1 staying forever collects $2(1 + 0.9 + \cdots + 0.9^{k-1}) = 20(1 - 0.9^k)$ in $k$ sweeps.

```figure
<svg viewBox="0 0 570 300" role="img" aria-label="Error of value iteration against the sweep number on a log scale: both curves fall in a straight line, the random MDP's error always below its gamma to the k bound">
<line x1="70.0" y1="250.0" x2="540.0" y2="250.0" class="stroke-grid"/>
<text x="62.0" y="254.0" class="fig-small" text-anchor="end">1e-4</text>
<line x1="70.0" y1="211.8" x2="540.0" y2="211.8" class="stroke-grid"/>
<text x="62.0" y="215.8" class="fig-small" text-anchor="end">1e-3</text>
<line x1="70.0" y1="173.6" x2="540.0" y2="173.6" class="stroke-grid"/>
<text x="62.0" y="177.6" class="fig-small" text-anchor="end">1e-2</text>
<line x1="70.0" y1="135.5" x2="540.0" y2="135.5" class="stroke-grid"/>
<text x="62.0" y="139.5" class="fig-small" text-anchor="end">1e-1</text>
<line x1="70.0" y1="97.3" x2="540.0" y2="97.3" class="stroke-grid"/>
<text x="62.0" y="101.3" class="fig-small" text-anchor="end">1</text>
<line x1="70.0" y1="59.1" x2="540.0" y2="59.1" class="stroke-grid"/>
<text x="62.0" y="63.1" class="fig-small" text-anchor="end">1e1</text>
<line x1="70.0" y1="250.0" x2="70.0" y2="254.0" class="stroke-muted"/>
<text x="70.0" y="268.0" class="fig-small" text-anchor="middle">0</text>
<line x1="148.3" y1="250.0" x2="148.3" y2="254.0" class="stroke-muted"/>
<text x="148.3" y="268.0" class="fig-small" text-anchor="middle">20</text>
<line x1="226.7" y1="250.0" x2="226.7" y2="254.0" class="stroke-muted"/>
<text x="226.7" y="268.0" class="fig-small" text-anchor="middle">40</text>
<line x1="305.0" y1="250.0" x2="305.0" y2="254.0" class="stroke-muted"/>
<text x="305.0" y="268.0" class="fig-small" text-anchor="middle">60</text>
<line x1="383.3" y1="250.0" x2="383.3" y2="254.0" class="stroke-muted"/>
<text x="383.3" y="268.0" class="fig-small" text-anchor="middle">80</text>
<line x1="461.7" y1="250.0" x2="461.7" y2="254.0" class="stroke-muted"/>
<text x="461.7" y="268.0" class="fig-small" text-anchor="middle">100</text>
<line x1="540.0" y1="250.0" x2="540.0" y2="254.0" class="stroke-muted"/>
<text x="540.0" y="268.0" class="fig-small" text-anchor="middle">120</text>
<text x="305.0" y="286.0" text-anchor="middle">sweep k</text>
<polyline points="70.0,47.8 73.9,48.6 77.8,49.5 81.8,50.3 85.7,51.2 89.6,52.0 93.5,52.9 97.4,53.7 101.3,54.6 105.2,55.4 109.2,56.3 113.1,57.1 117.0,58.0 120.9,58.8 124.8,59.7 128.8,60.5 132.7,61.4 136.6,62.2 140.5,63.1 144.4,63.9 148.3,64.8 152.2,65.6 156.2,66.5 160.1,67.3 164.0,68.2 167.9,69.0 171.8,69.9 175.8,70.7 179.7,71.6 183.6,72.4 187.5,73.3 191.4,74.1 195.3,75.0 199.2,75.8 203.2,76.7 207.1,77.5 211.0,78.4 214.9,79.2 218.8,80.1 222.8,80.9 226.7,81.8 230.6,82.6 234.5,83.5 238.4,84.3 242.3,85.2 246.2,86.0 250.2,86.9 254.1,87.7 258.0,88.6 261.9,89.4 265.8,90.3 269.8,91.1 273.7,92.0 277.6,92.8 281.5,93.7 285.4,94.5 289.3,95.4 293.2,96.2 297.2,97.1 301.1,97.9 305.0,98.8 308.9,99.6 312.8,100.5 316.8,101.3 320.7,102.2 324.6,103.0 328.5,103.9 332.4,104.7 336.3,105.6 340.2,106.4 344.2,107.3 348.1,108.1 352.0,109.0 355.9,109.8 359.8,110.7 363.8,111.5 367.7,112.4 371.6,113.2 375.5,114.1 379.4,114.9 383.3,115.8 387.2,116.6 391.2,117.5 395.1,118.3 399.0,119.2 402.9,120.0 406.8,120.9 410.8,121.7 414.7,122.6 418.6,123.5 422.5,124.3 426.4,125.2 430.3,126.0 434.2,126.9 438.2,127.7 442.1,128.6 446.0,129.4 449.9,130.3 453.8,131.1 457.8,132.0 461.7,132.8 465.6,133.7 469.5,134.5 473.4,135.4 477.3,136.2 481.2,137.1 485.2,137.9 489.1,138.8 493.0,139.6 496.9,140.5 500.8,141.3 504.8,142.2 508.7,143.0 512.6,143.9 516.5,144.7 520.4,145.6 524.3,146.4 528.2,147.3 532.2,148.1 536.1,149.0 540.0,149.8" class="stroke-muted dashed"/>
<polyline points="70.0,47.8 73.9,49.6 77.8,50.6 81.8,51.5 85.7,52.3 89.6,53.2 93.5,54.0 97.4,54.9 101.3,55.7 105.2,56.6 109.2,57.4 113.1,58.3 117.0,59.1 120.9,60.0 124.8,60.8 128.8,61.7 132.7,62.5 136.6,63.4 140.5,64.2 144.4,65.1 148.3,65.9 152.2,66.8 156.2,67.6 160.1,68.5 164.0,69.3 167.9,70.2 171.8,71.0 175.8,71.9 179.7,72.7 183.6,73.6 187.5,74.4 191.4,75.3 195.3,76.1 199.2,77.0 203.2,77.9 207.1,78.7 211.0,79.6 214.9,80.4 218.8,81.3 222.8,82.1 226.7,83.0 230.6,83.8 234.5,84.7 238.4,85.5 242.3,86.4 246.2,87.2 250.2,88.1 254.1,88.9 258.0,89.8 261.9,90.6 265.8,91.5 269.8,92.3 273.7,93.2 277.6,94.0 281.5,94.9 285.4,95.7 289.3,96.6 293.2,97.4 297.2,98.3 301.1,99.1 305.0,100.0 308.9,100.8 312.8,101.7 316.8,102.5 320.7,103.4 324.6,104.2 328.5,105.1 332.4,105.9 336.3,106.8 340.2,107.6 344.2,108.5 348.1,109.3 352.0,110.2 355.9,111.0 359.8,111.9 363.8,112.7 367.7,113.6 371.6,114.4 375.5,115.3 379.4,116.1 383.3,117.0 387.2,117.8 391.2,118.7 395.1,119.5 399.0,120.4 402.9,121.2 406.8,122.1 410.8,122.9 414.7,123.8 418.6,124.6 422.5,125.5 426.4,126.3 430.3,127.2 434.2,128.0 438.2,128.9 442.1,129.7 446.0,130.6 449.9,131.4 453.8,132.3 457.8,133.1 461.7,134.0 465.6,134.8 469.5,135.7 473.4,136.5 477.3,137.4 481.2,138.2 485.2,139.1 489.1,139.9 493.0,140.8 496.9,141.6 500.8,142.5 504.8,143.3 508.7,144.2 512.6,145.0 516.5,145.9 520.4,146.7 524.3,147.6 528.2,148.4 532.2,149.3 536.1,150.1 540.0,151.0" class="stroke-b"/>
<polyline points="70.0,47.6 73.9,49.3 77.8,51.1 81.8,52.8 85.7,54.6 89.6,56.3 93.5,58.1 97.4,59.8 101.3,61.6 105.2,63.3 109.2,65.1 113.1,66.8 117.0,68.6 120.9,70.3 124.8,72.1 128.8,73.8 132.7,75.6 136.6,77.3 140.5,79.0 144.4,80.8 148.3,82.5 152.2,84.3 156.2,86.0 160.1,87.8 164.0,89.5 167.9,91.3 171.8,93.0 175.8,94.8 179.7,96.5 183.6,98.3 187.5,100.0 191.4,101.8 195.3,103.5 199.2,105.3 203.2,107.0 207.1,108.7 211.0,110.5 214.9,112.2 218.8,114.0 222.8,115.7 226.7,117.5 230.6,119.2 234.5,121.0 238.4,122.7 242.3,124.5 246.2,126.2 250.2,128.0 254.1,129.7 258.0,131.5 261.9,133.2 265.8,135.0 269.8,136.7 273.7,138.4 277.6,140.2 281.5,141.9 285.4,143.7 289.3,145.4 293.2,147.2 297.2,148.9 301.1,150.7 305.0,152.4 308.9,154.2 312.8,155.9 316.8,157.7 320.7,159.4 324.6,161.2 328.5,162.9 332.4,164.7 336.3,166.4 340.2,168.1 344.2,169.9 348.1,171.6 352.0,173.4 355.9,175.1 359.8,176.9 363.8,178.6 367.7,180.4 371.6,182.1 375.5,183.9 379.4,185.6 383.3,187.4 387.2,189.1 391.2,190.9 395.1,192.6 399.0,194.4 402.9,196.1 406.8,197.8 410.8,199.6 414.7,201.3 418.6,203.1 422.5,204.8 426.4,206.6 430.3,208.3 434.2,210.1 438.2,211.8 442.1,213.6 446.0,215.3 449.9,217.1 453.8,218.8 457.8,220.6 461.7,222.3 465.6,224.1 469.5,225.8 473.4,227.5 477.3,229.3 481.2,231.0 485.2,232.8 489.1,234.5 493.0,236.3 496.9,238.0 500.8,239.8 504.8,241.5 508.7,243.3 512.6,245.0 516.5,246.8 520.4,248.5 524.3,250.3 528.2,252.0 532.2,253.8 536.1,255.5 540.0,257.2" class="stroke-c"/>
<line x1="290.0" y1="52.0" x2="320.0" y2="52.0" class="stroke-b"/>
<text x="326.0" y="56.0" class="fig-small" text-anchor="start">random MDP, gamma 0.95</text>
<line x1="290.0" y1="70.0" x2="320.0" y2="70.0" class="stroke-muted dashed"/>
<text x="326.0" y="74.0" class="fig-small" text-anchor="start">its bound 0.95^k |V_0 - V*|</text>
<line x1="290.0" y1="88.0" x2="320.0" y2="88.0" class="stroke-c"/>
<text x="326.0" y="92.0" class="fig-small" text-anchor="start">two-state, gamma 0.9: on its bound</text>
<text x="14.0" y="24.0" class="fig-small" text-anchor="start">error |V_k - V*|, log scale</text>
</svg>
Figure 2. Error against sweep on a log scale. Notice that a geometric rate is a straight line on this scale, that the two-state error sits exactly on its bound, and that the random MDP's error (0.11 at sweep 100) stays under its bound (0.12).
```

> [!KEY]
> One sweep of the optimality backup brings any two value vectors at least a factor $\gamma$ closer. So value iteration converges to the one answer $V^*$ from any starting guess, and its error is at most $\gamma^k$ times the starting error after $k$ sweeps.

## How many sweeps, and when to stop

Because the bound is a clean geometric decay, you can say in advance how many sweeps reach an error $\varepsilon$. Solve $\gamma^k \lVert V_0 - V^* \rVert \le \varepsilon$ for $k$:

$$
k \ge \frac{\ln\bigl(\varepsilon / \lVert V_0 - V^* \rVert\bigr)}{\ln \gamma}
$$

Read it as: the count grows with the logarithm of how much you want to shrink the error, divided by the logarithm of the shrink factor. For $\gamma$ near 1, $-\ln \gamma \approx 1 - \gamma$, so the count is about $\frac{1}{1 - \gamma} \ln \frac{\lVert V_0 - V^* \rVert}{\varepsilon}$, and halving $1 - \gamma$ doubles the sweeps.

```python
k = math.ceil(math.log(eps / initial_error) / math.log(gamma))
```

For the two-state world with $\varepsilon = 0.01$, $\ln(0.01 / 20) / \ln 0.9 = 72.14$, so you need 73 sweeps. The tight bound confirms it: the error is $20 \times 0.9^{72} = 0.01015$ after 72 sweeps, just above 0.01, and $0.00914$ after 73.

You usually can't compute $\lVert V_0 - V^* \rVert$, because you don't know $V^*$. What you can watch is the size of the last change, $\lVert V_{k+1} - V_k \rVert$. A small change does not by itself mean a small error, but the contraction turns it into a guarantee:

$$
\lVert V_{k+1} - V^* \rVert \le \frac{\gamma}{1 - \gamma} \lVert V_{k+1} - V_k \rVert
$$

The reason is one line. The error after a sweep is at most $\gamma$ times the error before it, and the error before is at most the last change plus the error after, so $e_{k+1} \le \gamma (\lVert V_{k+1} - V_k \rVert + e_{k+1})$. Move $\gamma e_{k+1}$ to the left and divide by $1 - \gamma$. With $\gamma = 0.95$ the factor is 19. On the exercise's random 20-state MDP, the bound is loose after the first sweep (42.4 against a true error of 17.7) and tight after sweep 101 (0.1038 against 0.1038).

> [!WARNING]
> Stopping when the last change falls below `tol` does not mean the error is below `tol`. To guarantee an error of $\varepsilon$, stop when the last change falls below $\varepsilon (1 - \gamma) / \gamma$.

## How good is the greedy policy?

You will usually stop early and act greedily on the values you have. Singh and Yee (1994) bound what that costs: if $\lVert V - V^* \rVert \le \varepsilon$, then the greedy policy $\pi_V$ satisfies $\lVert V^* - V^{\pi_V} \rVert \le \frac{2 \gamma}{1 - \gamma} \varepsilon$. The bound is loose: $V_1$ has error 18 and its greedy policy, "stay everywhere", loses 8 in state 0, far under the bound's $18 \times 18 = 324$. Still, it guarantees that a small value error can't hide a terrible policy.

Policy iteration and value iteration are two ends of one design: evaluate fully before improving, or evaluate for a single sweep. Schemes in between are called *generalized policy iteration*, and later methods follow that pattern, with samples standing in for the model, starting with [Monte Carlo prediction](/lessons/013-monte-carlo-prediction).

> [!RECAP]
> - One sweep is `v = rlenvs.q_from_v(mdp, v).max(axis=1)`, computed from the old `v`.
> - The backup is a $\gamma$-contraction, so the error after $k$ sweeps is at most $\gamma^k \lVert V_0 - V^* \rVert$ and $V^*$ is the only fixed point.
> - Reaching error $\varepsilon$ takes $\lceil \ln(\varepsilon / \text{initial error}) / \ln \gamma \rceil$ sweeps.
> - The error after a sweep is at most $\gamma / (1 - \gamma)$ times its last change.
> - The greedy policy of an approximate value can be read off at any sweep, and is often right early.

## Build it step by step

The checks test the functions in this order. `two_state_mdp` and `greedy_policy` are given, and you wrote `greedy_policy` in the previous chapter.

1. **`value_iteration(mdp, tol, v0=None)`.** Keep a list that starts with $V_0$ and gains every sweep, including the last. Return `(v, np.array(history))`. Common mistake: leaving $V_0$ out of the history, which shifts every row by one. The checks compare rows 0 to 4 with the table, test the $\gamma^k$ bound and the stopping bound at every sweep, and compare the result with `rlenvs.value_iteration`.
2. **`sweeps_needed(gamma, initial_error, eps)`.** One `math.ceil` of a ratio of logs, with 0 when the initial error is already at most `eps`. Common mistake: `math.floor`, which gives 72 and misses the target. The check confirms that sweeps 72 and 73 straddle 0.01.
3. **`stopping_threshold(gamma, eps)`.** The last change that guarantees an error of `eps`: `eps * (1 - gamma) / gamma`. Common mistake: dividing by `1 - gamma` instead of multiplying. The check runs `value_iteration` with this threshold and confirms the true error stays below `eps`.

## Cost and scaling

Each sweep costs $O(S^2 A)$, because it touches every entry of `P`. The number of sweeps grows like $\frac{1}{1 - \gamma} \ln \frac{1}{\varepsilon}$. On the exercise's 20-state MDP with $\gamma = 0.95$, value iteration needs 449 sweeps to get a change below $10^{-10}$, and policy iteration needs three evaluations.

## Where this breaks

**Discount close to 1.** The sweep count grows like $1 / (1 - \gamma)$, which hurts at $\gamma = 0.999$. With $\gamma = 1$ the contraction argument fails entirely, and convergence needs a policy that is sure to end. People use modified policy iteration instead.

## Further reading

- [Sutton and Barto, *Reinforcement Learning: An Introduction*, Chapter 4](http://incompleteideas.net/book/the-book-2nd.html) covers value iteration and generalized policy iteration.
- [Puterman, *Markov Decision Processes: Discrete Stochastic Dynamic Programming* (1994)](https://www.wiley.com/en-us/Markov+Decision+Processes:+Discrete+Stochastic+Dynamic+Programming-p-9781118625873) proves the contraction and the convergence of every method here.
- [Singh and Yee, "An upper bound on the loss from approximate optimal-value functions" (1994)](https://doi.org/10.1023/A:1022693225949) proves the $2\gamma\varepsilon/(1-\gamma)$ bound on acting greedily with approximate values.

## The exercise

You build value iteration and the two tools that reason about its error. Write `value_iteration(mdp, tol, v0=None)`, `sweeps_needed(gamma, initial_error, eps)`, and `stopping_threshold(gamma, eps)`. The starter gives `two_state_mdp` and `greedy_policy`.

The checks compare value iteration's first five iterates on the two-state world with this chapter's table, and check that it converges to the optimal values of a random 20-state MDP. On that MDP they test the $\gamma^k$ bound and the stopping bound at every sweep, from a random starting guess. They check `sweeps_needed(0.9, 20, 0.01) == 73` against the actual errors, that `stopping_threshold` guarantees the error it promises, and that the greedy action in state 0 switches at $V_2$.
