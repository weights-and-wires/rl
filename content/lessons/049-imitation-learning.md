---
title: Imitation learning
slug: 049-imitation-learning
part: "Part 7 — Models, exploration, and fixed data"
summary: Clone an expert, measure exactly how its small mistakes compound, and close the gap by letting DAgger query the expert.
minutes: 40
gpu: false
objectives:
  - Fit a behavior-cloning policy to expert demonstrations by counting actions per state.
  - Derive the tightrope cost T - (1 - (1 - eps)^T) / eps, and explain why it grows like eps T^2 while a learner that recovers pays about eps T.
  - Measure a learner's disagreement with the expert under the expert's states and under its own, and say why only the second predicts its value.
  - Implement DAgger and show it closes the gap that more expert demonstrations cannot.
prereqs:
  - 008-markov-decision-processes
  - 012-value-iteration
exercise: 049-imitation-learning
---

# Imitation learning

You want a robot to walk along a narrow ledge. Writing a reward that says exactly what a good walk looks like is hard. Showing it is easy: a person walks the ledge, and you record every step. Can the robot learn the walk from the recording alone?

The obvious method is plain supervised learning: treat each recorded state as an input and the expert's action as the label. It works surprisingly often, and it fails in a predictable way. The learner makes one small mistake, lands somewhere the expert never went, has no idea what to do there, and makes a bigger mistake. This chapter measures that failure exactly, on a two-state tightrope and then on a grid-world ledge, and fixes it with DAgger, which asks the expert what it would do where the learner actually ends up.

## What you already know

- A *policy* $\pi(a \mid s)$ is a probability distribution over actions in each state, stored as an array of shape `(S, A)`.
- Value iteration finds an optimal policy, which serves as the expert here.
- The state distribution $d^\pi$ says how much time a policy spends in each state.
- Supervised learning fits a model to labeled examples drawn from some distribution, and its guarantees hold on that distribution.

## Clone the expert

*Behavior cloning* (BC) turns demonstrations into a labeled dataset and fits a policy to it. ALVINN (Pomerleau, 1989) followed roads this way, and fine-tuning a language model on example answers is the same method.

On a tabular problem, the fit is counting. Let $n(s, a)$ be the number of times the expert took action $a$ in state $s$:

$$
\hat\pi(a \mid s) = \frac{n(s, a)}{\sum_b n(s, b)}, \qquad \hat\pi(a \mid s) = \frac{1}{A} \text{ if } s \text{ was never visited}
$$

Read it as: in a state the expert visited, copy its action frequencies. In a state it never visited, you have no information, so choose uniformly among the $A$ actions. This is the maximum-likelihood fit, and on the expert's own states it is perfect. In numpy:

```python
counts = np.zeros((S, A))
np.add.at(counts, (states, actions), 1.0)
totals = counts.sum(axis=1, keepdims=True)
pi = np.where(totals > 0, counts / np.maximum(totals, 1), 1.0 / A)
```

If state 0 has the labels $[1, 1, 2, 1]$ over three actions, its row becomes $[0, 0.75, 0.25]$. Real learners are imperfect even on the states they trained on, so the exercise makes this explicit: every learned policy, when it acts, takes a uniformly random action with probability $\varepsilon$. The expert never does.

The exercise's ledge world has three rows of 10 cells (Figure 1). The expert walks the top row from the start to the goal, which pays 1. The middle row is safe. The bottom row is holes, which cost 1 and end the episode. Moves are deterministic and $\gamma = 0.99$.

```figure
<svg viewBox="0 0 640 250" role="img" aria-label="The ledge world: the expert walks the top row to the goal, the middle row is safe, the bottom row is holes; a mistake from the top row lands in the middle row">
<defs><marker id="c042-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<rect x="50.0" y="50.0" width="50.0" height="44.0" class="fill-soft-a stroke-grid"/>
<rect x="100.0" y="50.0" width="50.0" height="44.0" class="fill-soft-a stroke-grid"/>
<rect x="150.0" y="50.0" width="50.0" height="44.0" class="fill-soft-a stroke-grid"/>
<rect x="200.0" y="50.0" width="50.0" height="44.0" class="fill-soft-a stroke-grid"/>
<rect x="250.0" y="50.0" width="50.0" height="44.0" class="fill-soft-a stroke-grid"/>
<rect x="300.0" y="50.0" width="50.0" height="44.0" class="fill-soft-a stroke-grid"/>
<rect x="350.0" y="50.0" width="50.0" height="44.0" class="fill-soft-a stroke-grid"/>
<rect x="400.0" y="50.0" width="50.0" height="44.0" class="fill-soft-a stroke-grid"/>
<rect x="450.0" y="50.0" width="50.0" height="44.0" class="fill-soft-a stroke-grid"/>
<rect x="500.0" y="50.0" width="50.0" height="44.0" class="fill-soft-a stroke-grid"/>
<rect x="50.0" y="94.0" width="50.0" height="44.0" class="fill-sunk stroke-grid"/>
<rect x="100.0" y="94.0" width="50.0" height="44.0" class="fill-sunk stroke-grid"/>
<rect x="150.0" y="94.0" width="50.0" height="44.0" class="fill-sunk stroke-grid"/>
<rect x="200.0" y="94.0" width="50.0" height="44.0" class="fill-sunk stroke-grid"/>
<rect x="250.0" y="94.0" width="50.0" height="44.0" class="fill-sunk stroke-grid"/>
<rect x="300.0" y="94.0" width="50.0" height="44.0" class="fill-sunk stroke-grid"/>
<rect x="350.0" y="94.0" width="50.0" height="44.0" class="fill-sunk stroke-grid"/>
<rect x="400.0" y="94.0" width="50.0" height="44.0" class="fill-sunk stroke-grid"/>
<rect x="450.0" y="94.0" width="50.0" height="44.0" class="fill-sunk stroke-grid"/>
<rect x="500.0" y="94.0" width="50.0" height="44.0" class="fill-sunk stroke-grid"/>
<rect x="50.0" y="138.0" width="50.0" height="44.0" class="fill-bad stroke-grid"/>
<rect x="100.0" y="138.0" width="50.0" height="44.0" class="fill-bad stroke-grid"/>
<rect x="150.0" y="138.0" width="50.0" height="44.0" class="fill-bad stroke-grid"/>
<rect x="200.0" y="138.0" width="50.0" height="44.0" class="fill-bad stroke-grid"/>
<rect x="250.0" y="138.0" width="50.0" height="44.0" class="fill-bad stroke-grid"/>
<rect x="300.0" y="138.0" width="50.0" height="44.0" class="fill-bad stroke-grid"/>
<rect x="350.0" y="138.0" width="50.0" height="44.0" class="fill-bad stroke-grid"/>
<rect x="400.0" y="138.0" width="50.0" height="44.0" class="fill-bad stroke-grid"/>
<rect x="450.0" y="138.0" width="50.0" height="44.0" class="fill-bad stroke-grid"/>
<rect x="500.0" y="138.0" width="50.0" height="44.0" class="fill-bad stroke-grid"/>
<text x="58.0" y="88.0" class="fig-bold" text-anchor="start">S</text>
<text x="525.0" y="77.0" class="fig-bold" text-anchor="middle">G</text>
<line x1="62.0" y1="70.0" x2="90.0" y2="70.0" class="stroke-b" marker-end="url(#c042-arrow)"/>
<line x1="112.0" y1="70.0" x2="140.0" y2="70.0" class="stroke-b" marker-end="url(#c042-arrow)"/>
<line x1="162.0" y1="70.0" x2="190.0" y2="70.0" class="stroke-b" marker-end="url(#c042-arrow)"/>
<line x1="212.0" y1="70.0" x2="240.0" y2="70.0" class="stroke-b" marker-end="url(#c042-arrow)"/>
<line x1="262.0" y1="70.0" x2="290.0" y2="70.0" class="stroke-b" marker-end="url(#c042-arrow)"/>
<line x1="312.0" y1="70.0" x2="340.0" y2="70.0" class="stroke-b" marker-end="url(#c042-arrow)"/>
<line x1="362.0" y1="70.0" x2="390.0" y2="70.0" class="stroke-b" marker-end="url(#c042-arrow)"/>
<line x1="412.0" y1="70.0" x2="440.0" y2="70.0" class="stroke-b" marker-end="url(#c042-arrow)"/>
<line x1="462.0" y1="70.0" x2="490.0" y2="70.0" class="stroke-b" marker-end="url(#c042-arrow)"/>
<text x="250.0" y="40.0" class="fig-small" text-anchor="middle">expert: always right along the top row (9 cells)</text>
<text x="558.0" y="121.0" class="fig-small" text-anchor="start">safe row: no labels</text>
<text x="558.0" y="165.0" class="fig-small" text-anchor="start">holes: cost 1</text>
<line x1="225.0" y1="82.0" x2="225.0" y2="108.0" class="stroke-bad" marker-end="url(#c042-arrow)"/>
<text x="232.0" y="92.0" class="fig-small" text-anchor="start">a mistake</text>
<text x="350.0" y="216.0" class="fig-small" text-anchor="middle">BC acts uniformly in the safe row, so a quarter of its moves fall into a hole</text>
</svg>
Figure 1. The ledge. The expert's demonstrations cover only the top row. One random step down puts the learner on cells that have no labels at all.
```

## Why small mistakes compound

The *tightrope* makes the problem exact. It has two states: on the rope and off it. Every step spent off the rope costs 1. The learner stays on with probability $1 - \varepsilon$ and falls off with probability $\varepsilon$. The expert never falls, so its demonstrations never show a fallen state. The cloned learner, off the rope, has no idea how to climb back, and in the worst case never does.

The chance that it is still on the rope at step $t$ is $(1 - \varepsilon)^t$, so the chance that it is off is $1 - (1 - \varepsilon)^t$. The expected cost over $T$ steps adds these up:

$$
\text{cost}(T) = \sum_{t=0}^{T-1} \Big(1 - (1 - \varepsilon)^t\Big) = T - \frac{1 - (1 - \varepsilon)^T}{\varepsilon}
$$

Read it as: at each step, pay the probability of having already fallen. The closed form sums the geometric series $\sum_t (1 - \varepsilon)^t$. When $\varepsilon T$ is small, the cost is about $\varepsilon T^2 / 2$: a fall at step $t$, with probability about $\varepsilon$, is paid for at each of the $T - t$ steps that remain, and summing $\varepsilon (T - t)$ over $t$ gives that triangle.

Give the learner the recovery action: off the rope, it climbs straight back on. Each fall then costs one step, and the cost is about $\varepsilon T$. The exact answer is $\frac{\varepsilon}{1 + \varepsilon}\big(T - \frac{1 - (-\varepsilon)^T}{1 + \varepsilon}\big)$. You compute both in numpy by pushing the state distribution forward one step at a time:

```python
d, total = mdp.mu0.copy(), 0.0
for _ in range(T):
    total += d @ r_pi        # expected reward this step
    d = d @ P_pi             # distribution one step later
```

With $\varepsilon = 0.01$, the expected cost is as follows:

| Horizon $T$ | Never recovers | Recovers at once | $\varepsilon T^2 / 2$ | $\varepsilon T$ |
|---|---|---|---|---|
| 10 | 0.44 | 0.09 | 0.5 | 0.1 |
| 100 | 36.60 | 0.98 | 50 | 1 |
| 1,000 | 900.00 | 9.89 | 5,000 | 10 |

At $T = 100$, a learner that is right 99% of the time pays 36.6, while the same learner with a recovery action pays about 1.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Expected cost of the tightrope over T steps at epsilon 0.01: without recovery the cost climbs toward T, with recovery it grows about epsilon T">
<defs><marker id="c042-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<line x1="70.0" y1="270.0" x2="540.0" y2="270.0" class="stroke-grid"/>
<text x="62.0" y="274.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="212.5" x2="540.0" y2="212.5" class="stroke-grid"/>
<text x="62.0" y="216.5" class="fig-small" text-anchor="end">50</text>
<line x1="70.0" y1="155.0" x2="540.0" y2="155.0" class="stroke-grid"/>
<text x="62.0" y="159.0" class="fig-small" text-anchor="end">100</text>
<line x1="70.0" y1="97.5" x2="540.0" y2="97.5" class="stroke-grid"/>
<text x="62.0" y="101.5" class="fig-small" text-anchor="end">150</text>
<line x1="70.0" y1="40.0" x2="540.0" y2="40.0" class="stroke-grid"/>
<text x="62.0" y="44.0" class="fig-small" text-anchor="end">200</text>
<line x1="70.0" y1="270.0" x2="540.0" y2="270.0" class="stroke-muted"/>
<line x1="70.0" y1="270.0" x2="70.0" y2="275.0" class="stroke-muted"/>
<text x="70.0" y="290.0" class="fig-small" text-anchor="middle">0</text>
<line x1="226.7" y1="270.0" x2="226.7" y2="275.0" class="stroke-muted"/>
<text x="226.7" y="290.0" class="fig-small" text-anchor="middle">100</text>
<line x1="383.3" y1="270.0" x2="383.3" y2="275.0" class="stroke-muted"/>
<text x="383.3" y="290.0" class="fig-small" text-anchor="middle">200</text>
<line x1="540.0" y1="270.0" x2="540.0" y2="275.0" class="stroke-muted"/>
<text x="540.0" y="290.0" class="fig-small" text-anchor="middle">300</text>
<text x="305.0" y="312.0" class="fig-small" text-anchor="middle">horizon T (steps)</text>
<polyline points="70.0,270.0 77.8,269.9 85.7,269.5 93.5,268.8 101.3,267.9 109.2,266.8 117.0,265.4 124.8,263.9 132.7,262.1 140.5,260.1 148.3,257.9 156.2,255.6 164.0,253.1 171.8,250.4 179.7,247.6 187.5,244.6 195.3,241.5 203.2,238.3 211.0,235.0 218.8,231.5 226.7,227.9 234.5,224.2 242.3,220.4 250.2,216.5 258.0,212.6 265.8,208.5 273.7,204.4 281.5,200.1 289.3,195.8 297.2,191.5 305.0,187.0 312.8,182.5 320.7,178.0 328.5,173.3 336.3,168.7 344.2,163.9 352.0,159.2 359.8,154.3 367.7,149.5 375.5,144.5 383.3,139.6 391.2,134.6 399.0,129.6 406.8,124.5 414.7,119.4 422.5,114.3 430.3,109.1 438.2,103.9 446.0,98.7 453.8,93.4 461.7,88.2 469.5,82.9 477.3,77.6 485.2,72.2 493.0,66.9 500.8,61.5 508.7,56.1 516.5,50.7 524.3,45.3 532.2,40.0 540.0,40.0" class="stroke-bad" fill="none"/>
<polyline points="70.0,270.0 77.8,270.0 85.7,269.9 93.5,269.8 101.3,269.8 109.2,269.7 117.0,269.7 124.8,269.6 132.7,269.6 140.5,269.5 148.3,269.4 156.2,269.4 164.0,269.3 171.8,269.3 179.7,269.2 187.5,269.2 195.3,269.1 203.2,269.0 211.0,269.0 218.8,268.9 226.7,268.9 234.5,268.8 242.3,268.8 250.2,268.7 258.0,268.6 265.8,268.6 273.7,268.5 281.5,268.5 289.3,268.4 297.2,268.4 305.0,268.3 312.8,268.2 320.7,268.2 328.5,268.1 336.3,268.1 344.2,268.0 352.0,268.0 359.8,267.9 367.7,267.8 375.5,267.8 383.3,267.7 391.2,267.7 399.0,267.6 406.8,267.6 414.7,267.5 422.5,267.4 430.3,267.4 438.2,267.3 446.0,267.3 453.8,267.2 461.7,267.2 469.5,267.1 477.3,267.1 485.2,267.0 493.0,266.9 500.8,266.9 508.7,266.8 516.5,266.8 524.3,266.7 532.2,266.7 540.0,266.6" class="stroke-c" fill="none"/>
<polyline points="70.0,270.0 77.8,269.9 85.7,269.4 93.5,268.7 101.3,267.7 109.2,266.4 117.0,264.8 124.8,263.0 132.7,260.8 140.5,258.4 148.3,255.6 156.2,252.6 164.0,249.3 171.8,245.7 179.7,241.8 187.5,237.7 195.3,233.2 203.2,228.5 211.0,223.4 218.8,218.1 226.7,212.5 234.5,206.6 242.3,200.4 250.2,194.0 258.0,187.2 265.8,180.2 273.7,172.8 281.5,165.2 289.3,157.3 297.2,149.1 305.0,140.6 312.8,131.9 320.7,122.8 328.5,113.5 336.3,103.8 344.2,93.9 352.0,83.7 359.8,73.2 367.7,62.4 375.5,51.4 383.3,40.0 391.2,40.0 399.0,40.0 406.8,40.0 414.7,40.0 422.5,40.0 430.3,40.0 438.2,40.0 446.0,40.0 453.8,40.0 461.7,40.0 469.5,40.0 477.3,40.0 485.2,40.0 493.0,40.0 500.8,40.0 508.7,40.0 516.5,40.0 524.3,40.0 532.2,40.0 540.0,40.0" class="stroke-muted dashed" fill="none"/>
<circle cx="226.7" cy="227.9" r="5" class="fill-bad"/>
<text x="236.7" y="225.9" class="fig-small" text-anchor="start">T = 100: cost 36.6</text>
<line x1="90.0" y1="50.0" x2="120.0" y2="50.0" class="stroke-bad"/>
<text x="126.0" y="54.0" class="fig-small" text-anchor="start">never recovers</text>
<line x1="90.0" y1="68.0" x2="120.0" y2="68.0" class="stroke-c"/>
<text x="126.0" y="72.0" class="fig-small" text-anchor="start">recovers at once</text>
<line x1="90.0" y1="86.0" x2="120.0" y2="86.0" class="stroke-muted dashed"/>
<text x="126.0" y="90.0" class="fig-small" text-anchor="start">epsilon T^2 / 2, the small-epsilon estimate</text>
</svg>
Figure 2. Expected cost against horizon, from the closed forms. The learner that never recovers pays a quadratic cost at first, 36.6 at T = 100, while the one that climbs back pays about 1.
```

Ross and Bagnell (2010) prove that a cloned policy whose error rate on the expert's states is $\varepsilon$ has a cost at most $T^2 \varepsilon$ above the expert's, for costs between 0 and 1. The tightrope shows the bound is tight.

> [!KEY]
> A cloned policy pays for each mistake at every later step, because nobody showed it how to recover. Its extra cost can grow like $\varepsilon T^2$, against $\varepsilon T$ for a learner that recovers.

## Measure the error where the learner goes

Supervised learning promises low error on the distribution it trained on. For BC, that is the expert's state distribution. But the learner is judged on its own. The exercise measures both with one quantity, the learner's *disagreement* with the expert $\pi^*$ under the states of a policy $\pi$:

$$
\text{disagreement}(\hat\pi; d^\pi) = \sum_s d^\pi(s) \Big( 1 - \sum_a \hat\pi(a \mid s)\, \pi^*(a \mid s) \Big)
$$

Read it as: for each state, take the chance that the learner's action differs from the expert's, and weight it by how much time $\pi$ spends there. With $\pi = \pi^*$ this is BC's training error. With $\pi$ the learner itself, it is the error that decides the learner's value. In numpy:

```python
d = rlenvs.state_distribution(mdp, pi_dist)
miss = 1.0 - (learner * expert).sum(axis=1)
error = d @ np.where(mdp.terminal, 0.0, miss)
```

On the ledge with $\varepsilon = 0.1$, the results are as follows:

| Policy | Value from the start | Error on expert states | Error on its own states |
|---|---|---|---|
| Expert | 0.9227 | 0 | 0 |
| Expert, with the same mistakes | 0.8995 | 0 | 0 |
| BC, with mistakes | 0.6979 | 0 | 0.0319 |

The fair reference is the expert making the same random mistakes, because no learner can avoid those. BC loses 0.20 against it, yet its error on the expert's states is exactly zero, so no supervised-learning metric would warn you. A random step down lands the learner in the middle row, which the expert never visited. There it acts uniformly, and a quarter of its moves drop it into a hole. More demonstrations do not help: the expert and the world are deterministic, so one demonstration and fifty cover the same nine cells, and BC's value is 0.6979 either way.

> [!WARNING]
> Low validation error on held-out demonstrations says nothing about how a cloned policy does when it acts. Evaluate it by running it, not by its loss on the expert's data.

## DAgger: ask the expert where you are

If the problem is that the learner trains on the wrong states, the fix is to train it on the right ones. DAgger, short for dataset aggregation (Ross, Gordon, and Bagnell, 2011), collects labels on the learner's own states:

1. Roll out the expert and record the states it visits. Label each with the expert's action, and train a first policy. This is BC.
2. Roll out the current learner, mistakes and all, and record the states it visits.
3. Ask the expert what it would do in each of those states, and add the labeled states to the dataset.
4. Retrain on the whole dataset, and repeat from step 2.

The paper allows mixing the expert into later rollouts but reports that using it only in the first iteration often works best, which is the choice here. Because the dataset keeps growing with states the learner reaches, the learner eventually knows what to do everywhere it goes, including how to recover from its own mistakes. Ross and others prove a cost within about $u T \varepsilon_N$ of the expert's, where $\varepsilon_N$ is the learner's error on its own states and $u$ bounds how much one wrong action raises the expert's remaining cost. That is linear in $T$ when mistakes are recoverable.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="DAgger value per iteration on the ledge with 10 percent mistakes: it rises from 0.698 for behavior cloning to 0.8995, the value of the expert making the same mistakes">
<defs><marker id="c042-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<line x1="70.0" y1="260.0" x2="540.0" y2="260.0" class="stroke-grid"/>
<text x="62.0" y="264.0" class="fig-small" text-anchor="end">0.65</text>
<line x1="70.0" y1="225.0" x2="540.0" y2="225.0" class="stroke-grid"/>
<text x="62.0" y="229.0" class="fig-small" text-anchor="end">0.70</text>
<line x1="70.0" y1="190.0" x2="540.0" y2="190.0" class="stroke-grid"/>
<text x="62.0" y="194.0" class="fig-small" text-anchor="end">0.75</text>
<line x1="70.0" y1="155.0" x2="540.0" y2="155.0" class="stroke-grid"/>
<text x="62.0" y="159.0" class="fig-small" text-anchor="end">0.80</text>
<line x1="70.0" y1="120.0" x2="540.0" y2="120.0" class="stroke-grid"/>
<text x="62.0" y="124.0" class="fig-small" text-anchor="end">0.85</text>
<line x1="70.0" y1="85.0" x2="540.0" y2="85.0" class="stroke-grid"/>
<text x="62.0" y="89.0" class="fig-small" text-anchor="end">0.90</text>
<line x1="70.0" y1="50.0" x2="540.0" y2="50.0" class="stroke-grid"/>
<text x="62.0" y="54.0" class="fig-small" text-anchor="end">0.95</text>
<text x="70.0" y="280.0" class="fig-small" text-anchor="middle">1</text>
<text x="204.3" y="280.0" class="fig-small" text-anchor="middle">5</text>
<text x="372.1" y="280.0" class="fig-small" text-anchor="middle">10</text>
<text x="540.0" y="280.0" class="fig-small" text-anchor="middle">15</text>
<text x="305.0" y="302.0" class="fig-small" text-anchor="middle">DAgger iteration (1 is behavior cloning)</text>
<line x1="70.0" y1="85.3" x2="540.0" y2="85.3" class="stroke-c dashed"/>
<text x="540.0" y="77.3" class="fig-small" text-anchor="end">expert with the same mistakes: 0.8995</text>
<polyline points="70.0,226.5 103.6,127.9 137.1,127.9 170.7,85.6 204.3,85.6 237.9,85.6 271.4,85.6 305.0,85.6 338.6,85.6 372.1,85.6 405.7,85.3 439.3,85.3 472.9,85.3 506.4,85.3 540.0,85.3" class="stroke-b" fill="none"/>
<circle cx="70.0" cy="226.5" r="4" class="fill-b"/>
<circle cx="103.6" cy="127.9" r="4" class="fill-b"/>
<circle cx="137.1" cy="127.9" r="4" class="fill-b"/>
<circle cx="170.7" cy="85.6" r="4" class="fill-b"/>
<circle cx="204.3" cy="85.6" r="4" class="fill-b"/>
<circle cx="237.9" cy="85.6" r="4" class="fill-b"/>
<circle cx="271.4" cy="85.6" r="4" class="fill-b"/>
<circle cx="305.0" cy="85.6" r="4" class="fill-b"/>
<circle cx="338.6" cy="85.6" r="4" class="fill-b"/>
<circle cx="372.1" cy="85.6" r="4" class="fill-b"/>
<circle cx="405.7" cy="85.3" r="4" class="fill-b"/>
<circle cx="439.3" cy="85.3" r="4" class="fill-b"/>
<circle cx="472.9" cy="85.3" r="4" class="fill-b"/>
<circle cx="506.4" cy="85.3" r="4" class="fill-b"/>
<circle cx="540.0" cy="85.3" r="4" class="fill-b"/>
<text x="78.0" y="230.5" class="fig-small" text-anchor="start">BC: 0.6979</text>
<text x="111.6" y="145.9" class="fig-small" text-anchor="start">iteration 2: 0.8386</text>
<text x="210.3" y="105.6" class="fig-small" text-anchor="start">iteration 5: 0.8991</text>
</svg>
Figure 3. Value of the noisy learner at the start, seed 0, 20 rollouts per iteration. Each iteration adds labels on states the learner itself reaches, and the gap to the expert closes within five iterations.
```

On the ledge, with 20 rollouts per iteration and seed 0, the learner's value climbs from 0.6979 for BC to 0.8386 at iteration 2 and 0.8991 at iteration 5, while its error on its own states falls from 0.0319 to 0.0001.

> [!KEY]
> BC trains on the expert's states and is tested on its own. DAgger trains on its own states, so its training error is the error that matters.

The price is an expert on call: someone who can label arbitrary states, including strange ones the learner wandered into. For a simulator with a planning expert that is cheap. For a human it is slow, because labeling a state you are not controlling is harder than driving.

> [!RECAP]
> - BC counts actions per state and uses uniform rows where there is no data.
> - The tightrope cost is $T - (1 - (1 - \varepsilon)^T)/\varepsilon$ without recovery.
> - Disagreement weights $1 - \sum_a \hat\pi \pi^*$ by a state distribution and ignores terminal states.
> - DAgger rolls out the expert once, then the noisy learner, and always labels with the expert.

## Build it step by step

The checks test the functions in this order. The helpers `ledge_world`, `expert_policy`, and `tightrope` are written for you.

1. **`fit_bc(states, actions, n_states, n_actions)`.** Count with `np.add.at`, normalize rows, and fill empty rows with `1 / A`. Do not write `counts[states, actions] += 1`, which counts a repeated pair only once.
2. **`add_mistakes(policy, eps)`.** One line. Each row still sums to one.
3. **`finite_horizon_return(mdp, pi, T)`.** Build `P_pi` with `np.einsum` and `r_pi` with a row sum, zero the terminal rows and entries, then push `d` forward `T` times. This sum is undiscounted.
4. **`collect(mdp, policy, n_episodes, max_steps, rng)`.** Draw the start from `mu0`, then loop: stop at a terminal state, record the state, sample an action, and call `mdp.sample`. Do not record the terminal state.
5. **`disagreement(mdp, learner, expert, pi_dist)`.** Dot `rlenvs.state_distribution(mdp, pi_dist)` with the per-state miss rate, and zero the terminal states.
6. **`dagger(mdp, expert, eps, n_iters, n_episodes, max_steps, seed)`.** Keep one growing array of states, label with `expert.argmax(axis=1)`, refit with `fit_bc` each iteration, and roll out `add_mistakes(learner, eps)` after the first. Refitting on only the latest states forgets the expert's own path.

## Where this breaks

**No expert on call.** DAgger needs interactive labels. With a fixed set of demonstrations, DART (Laskey and others, 2017) injects noise into the expert's actions while it demonstrates, so the data shows how to recover. [Inverse RL and GAIL](/lessons/050-inverse-rl-and-gail) learn a reward instead.

**Several right answers.** When an expert sometimes goes left around an obstacle and sometimes right, a model that predicts the average action goes straight into it. Diffusion policies (Chi and others, 2023) model the whole action distribution.

## Further reading

- [Pomerleau (1989), ALVINN: an autonomous land vehicle in a neural network](https://proceedings.neurips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html). Behavior cloning for road following.
- [Ross and Bagnell (2010), Efficient reductions for imitation learning](https://proceedings.mlr.press/v9/ross10a.html). The $T^2 \varepsilon$ bound for behavior cloning, and why it is tight.
- [Ross, Gordon, and Bagnell (2011), A reduction of imitation learning and structured prediction to no-regret online learning](https://arxiv.org/abs/1011.0686). DAgger and its guarantee that grows linearly in the horizon.
- [Laskey, Lee, Fox, Dragan, and Goldberg (2017), DART: noise injection for robust imitation learning](https://arxiv.org/abs/1703.09327). Recovery data without interactive labels.

## The exercise

You measure behavior cloning's failure exactly and fix it. Write `fit_bc`, `add_mistakes`, `finite_horizon_return` for an exact $T$-step return, `collect` to roll out a policy, and `disagreement` to weight a learner's errors by a state distribution. Then write `dagger`.

The checks verify `finite_horizon_return` against both tightrope closed forms, including 36.60 for $\varepsilon = 0.01$ and $T = 100$, and `disagreement` against the tightrope's closed form. On the ledge, they confirm that BC has zero error on the expert's states yet loses over 0.15 of value with 1 or 50 demonstrations, and that DAgger ends within 0.005 of the expert making the same mistakes on three seeds.
