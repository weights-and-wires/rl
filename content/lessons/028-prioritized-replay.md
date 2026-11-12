---
title: Prioritized experience replay
slug: 028-prioritized-replay
part: "Part 4 — Deep value methods"
summary: Replay the transitions the network predicts worst more often, sample them in logarithmic time with a sum tree, and correct the bias with importance weights.
minutes: 40
gpu: false
objectives:
  - Explain why replaying transitions in proportion to their TD error speeds learning, and why the raw error is a poor sampling weight.
  - Compute sampling probabilities p^alpha / sum p^alpha and importance weights (N P)^-beta by hand.
  - Implement a sum tree and sample from it in logarithmic time by walking down from the root.
  - Build a prioritized replay buffer that gives new items the maximum priority.
prereqs:
  - 026-deep-q-networks
exercise: 028-prioritized-replay
---

# Prioritized experience replay

A student who rereads every flashcard equally spends most of the time on cards they already know. A smarter student spends the time on the cards they get wrong. A DQN replay buffer is the first kind of student. It draws every stored transition with the same probability, so a transition the network already predicts perfectly is replayed as often as one it gets badly wrong.

The useful signal is how surprised the network was by a transition, and you already compute it: the TD error. This chapter replays transitions in proportion to a score built from the TD error. That change creates two problems, and the chapter solves both. The first is cost, because sampling from a non-uniform distribution over a million transitions must be fast. The second is bias, because the network no longer sees the data in the proportions it actually occurred.

## What you already know

- A replay buffer stores transitions and draws minibatches from it, so that consecutive samples are not correlated.
- The TD error $\delta = r + \gamma \max_{a'} Q_{\bar\theta}(s', a') - Q_\theta(s, a)$ measures how wrong the network's prediction was on a transition.
- A probability distribution over $N$ items is a vector of $N$ nonnegative numbers that sums to one.
- Importance sampling reweights samples from one distribution so that their average matches another distribution.
- Looking up or updating one entry on each level of a binary tree of $N$ leaves takes about $\log_2 N$ steps.

## Sample in proportion to surprise

Give transition $i$ a priority $p_i$ that grows with its TD error. The obvious choice is $p_i = |\delta_i|$, and to sample transition $i$ with probability $p_i / \sum_k p_k$. It has two problems.

First, a transition with $\delta_i = 0$ has probability 0 and is never replayed again. Its error was zero a moment ago, not forever, because the network keeps changing. Second, a single transition with a huge, noisy TD error can dominate the batch.

The proportional variant of Schaul et al. (2016) fixes both with two knobs:

$$
p_i = |\delta_i| + \epsilon, \qquad P(i) = \frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}}
$$

Read it as: add a small $\epsilon$ so that no priority is zero, raise each priority to the power $\alpha$, and normalize. When $\alpha = 0$ every item gets the same weight, which is uniform replay. When $\alpha = 1$ the probabilities are fully proportional to the priorities. The paper uses $\alpha = 0.6$, which flattens the big priorities.

### Work an example

Take four transitions with priorities $p = [2, 1, 0.5, 0.5]$.

- With $\alpha = 1$, the sum is 4, so $P = [0.5, 0.25, 0.125, 0.125]$.
- With $\alpha = 0.6$, the powers are $[1.5157, 1, 0.6598, 0.6598]$, which sum to 3.8352. So $P = [0.3952, 0.2607, 0.1720, 0.1720]$.

Figure 1 compares the three settings. The first transition is four times likelier than the third at $\alpha = 1$ and only 2.3 times likelier at $\alpha = 0.6$, so the exponent keeps the favorite from taking over.

```figure
<svg viewBox="0 0 620 270" role="img" aria-label="Sampling probability of four transitions with priorities 2, 1, 0.5, 0.5 under alpha 0, 0.6, and 1">
<line x1="50.0" y1="230.0" x2="580.0" y2="230.0" class="stroke-muted"/>
<line x1="50.0" y1="230.0" x2="580.0" y2="230.0" class="stroke-grid"/>
<text x="42.0" y="234.0" class="fig-small" text-anchor="end">0</text>
<line x1="50.0" y1="155.0" x2="580.0" y2="155.0" class="stroke-grid"/>
<text x="42.0" y="159.0" class="fig-small" text-anchor="end">0.25</text>
<line x1="50.0" y1="80.0" x2="580.0" y2="80.0" class="stroke-grid"/>
<text x="42.0" y="84.0" class="fig-small" text-anchor="end">0.5</text>
<rect x="80.0" y="155.0" width="28.0" height="75.0" class="fill-soft-a"/>
<text x="94.0" y="150.0" class="fig-small" text-anchor="middle">.250</text>
<rect x="112.0" y="111.4" width="28.0" height="118.6" class="fill-b"/>
<text x="126.0" y="106.4" class="fig-small" text-anchor="middle">.395</text>
<rect x="144.0" y="80.0" width="28.0" height="150.0" class="fill-d"/>
<text x="158.0" y="75.0" class="fig-small" text-anchor="middle">.500</text>
<text x="126.0" y="248.0" class="fig-small" text-anchor="middle">p = 2</text>
<rect x="205.0" y="155.0" width="28.0" height="75.0" class="fill-soft-a"/>
<text x="219.0" y="150.0" class="fig-small" text-anchor="middle">.250</text>
<rect x="237.0" y="151.8" width="28.0" height="78.2" class="fill-b"/>
<text x="251.0" y="146.8" class="fig-small" text-anchor="middle">.261</text>
<rect x="269.0" y="155.0" width="28.0" height="75.0" class="fill-d"/>
<text x="283.0" y="150.0" class="fig-small" text-anchor="middle">.250</text>
<text x="251.0" y="248.0" class="fig-small" text-anchor="middle">p = 1</text>
<rect x="330.0" y="155.0" width="28.0" height="75.0" class="fill-soft-a"/>
<text x="344.0" y="150.0" class="fig-small" text-anchor="middle">.250</text>
<rect x="362.0" y="178.4" width="28.0" height="51.6" class="fill-b"/>
<text x="376.0" y="173.4" class="fig-small" text-anchor="middle">.172</text>
<rect x="394.0" y="192.5" width="28.0" height="37.5" class="fill-d"/>
<text x="408.0" y="187.5" class="fig-small" text-anchor="middle">.125</text>
<text x="376.0" y="248.0" class="fig-small" text-anchor="middle">p = 0.5</text>
<rect x="455.0" y="155.0" width="28.0" height="75.0" class="fill-soft-a"/>
<text x="469.0" y="150.0" class="fig-small" text-anchor="middle">.250</text>
<rect x="487.0" y="178.4" width="28.0" height="51.6" class="fill-b"/>
<text x="501.0" y="173.4" class="fig-small" text-anchor="middle">.172</text>
<rect x="519.0" y="192.5" width="28.0" height="37.5" class="fill-d"/>
<text x="533.0" y="187.5" class="fig-small" text-anchor="middle">.125</text>
<text x="501.0" y="248.0" class="fig-small" text-anchor="middle">p = 0.5</text>
<rect x="330.0" y="14.0" width="14.0" height="14.0" class="fill-soft-a"/>
<text x="350.0" y="26.0" class="fig-small" text-anchor="start">alpha = 0</text>
<rect x="420.0" y="14.0" width="14.0" height="14.0" class="fill-b"/>
<text x="440.0" y="26.0" class="fig-small" text-anchor="start">alpha = 0.6</text>
<rect x="520.0" y="14.0" width="14.0" height="14.0" class="fill-d"/>
<text x="540.0" y="26.0" class="fig-small" text-anchor="start">alpha = 1</text>
<text x="50.0" y="14.0" text-anchor="start">P(i)</text>
</svg>
Figure 1. Four transitions with priorities 2, 1, 0.5, and 0.5. At alpha = 0 every transition is equally likely; at alpha = 1 the top transition is four times as likely as each of the last two. The exponent 0.6 sits between: the top transition is 2.3 times as likely as each of the last two.
```

A new transition has no TD error yet, so it has no priority. The buffer gives it the largest raw priority seen so far, which makes it likely to be replayed soon, and updates the priority after the first replay.

## Undo the bias with importance weights

Sampling by $P(i)$ is not free. The network sees high-error transitions far more often than they occurred, so its updates pull toward them. The update is no longer an average over the data the agent actually collected.

Importance sampling fixes this. Uniform sampling would pick each item with probability $1/N$, and you sampled it with probability $P(i)$. So item $i$ is oversampled by the factor $N \cdot P(i)$, and you shrink its loss by that factor, raised to a power $\beta$:

$$
w_i = \frac{\big(N \cdot P(i)\big)^{-\beta}}{\max_j \big(N \cdot P(j)\big)^{-\beta}}
$$

Read it as: $N \cdot P(i)$ says how many times more often than uniform you sample item $i$. Raising it to $-\beta$ gives the shrink factor. Dividing by the largest value keeps every weight at most 1, so updates only ever shrink and the step size stays stable. With $\beta = 1$ the correction is exact, and with $\beta = 0$ there is none. The paper anneals $\beta$ from 0.4 up to 1 over training. Each sampled TD error's loss is multiplied by its $w_i$.

For the example with $\alpha = 0.6$ and $\beta = 0.4$:

- $N \cdot P = [1.5808, 1.0430, 0.6881, 0.6881]$.
- Raising each to $-0.4$ gives $[0.8326, 0.9833, 1.1613, 1.1613]$.
- Dividing by the largest, 1.1613, gives $w = [0.7170, 0.8467, 1, 1]$.

Figure 2 shows the pattern. The most replayed transition gets the smallest weight, which cancels part of its extra sampling.

```figure
<svg viewBox="0 0 620 260" role="img" aria-label="Sampling probability and importance weight for four transitions: the most sampled transition gets the smallest weight">
<line x1="60.0" y1="220.0" x2="580.0" y2="220.0" class="stroke-muted"/>
<rect x="90.0" y="93.5" width="40.0" height="126.5" class="fill-b"/>
<text x="110.0" y="88.5" class="fig-small" text-anchor="middle">0.395</text>
<rect x="136.0" y="105.3" width="40.0" height="114.7" class="fill-c"/>
<text x="156.0" y="100.3" class="fig-small" text-anchor="middle">0.717</text>
<text x="133.0" y="238.0" class="fig-small" text-anchor="middle">p = 2</text>
<rect x="215.0" y="136.6" width="40.0" height="83.4" class="fill-b"/>
<text x="235.0" y="131.6" class="fig-small" text-anchor="middle">0.261</text>
<rect x="261.0" y="84.5" width="40.0" height="135.5" class="fill-c"/>
<text x="281.0" y="79.5" class="fig-small" text-anchor="middle">0.847</text>
<text x="258.0" y="238.0" class="fig-small" text-anchor="middle">p = 1</text>
<rect x="340.0" y="165.0" width="40.0" height="55.0" class="fill-b"/>
<text x="360.0" y="160.0" class="fig-small" text-anchor="middle">0.172</text>
<rect x="386.0" y="60.0" width="40.0" height="160.0" class="fill-c"/>
<text x="406.0" y="55.0" class="fig-small" text-anchor="middle">1.000</text>
<text x="383.0" y="238.0" class="fig-small" text-anchor="middle">p = 0.5</text>
<rect x="465.0" y="165.0" width="40.0" height="55.0" class="fill-b"/>
<text x="485.0" y="160.0" class="fig-small" text-anchor="middle">0.172</text>
<rect x="511.0" y="60.0" width="40.0" height="160.0" class="fill-c"/>
<text x="531.0" y="55.0" class="fig-small" text-anchor="middle">1.000</text>
<text x="508.0" y="238.0" class="fig-small" text-anchor="middle">p = 0.5</text>
<rect x="60.0" y="14.0" width="14.0" height="14.0" class="fill-b"/>
<text x="80.0" y="26.0" class="fig-small" text-anchor="start">sampling probability P(i), alpha = 0.6</text>
<rect x="340.0" y="14.0" width="14.0" height="14.0" class="fill-c"/>
<text x="360.0" y="26.0" class="fig-small" text-anchor="start">importance weight w_i, beta = 0.4</text>
</svg>
Figure 2. The more often a transition is sampled (orange), the smaller its update weight (green). The two least likely transitions both get weight 1, and the most likely gets 0.717.
```

The paper's pseudocode divides by the largest weight without saying over which set. This exercise uses the largest over the whole buffer, as common implementations do. Some implementations use the batch's largest weight instead.

> [!KEY]
> Prioritized replay samples with $P(i) = p_i^\alpha / \sum_k p_k^\alpha$ so the network studies what surprises it, and multiplies each loss by $w_i = (N P(i))^{-\beta}$, divided by the largest weight, so that the oversampling does not bias the update.

## Sample in logarithmic time with a sum tree

To draw from $P$, you could build the running total of the priorities and search it. That costs $O(N)$ per draw, and a buffer of a million transitions makes that far too slow when a priority changes after every update.

A *sum tree* is a binary tree whose leaves hold the $p_i^\alpha$ and whose every inner node holds the sum of its two children. The root holds the total. To sample, draw a mass $u$ uniformly from $[0, \text{total})$ and walk down from the root. At each node, if $u$ is less than the left child's sum, go left. Otherwise subtract the left child's sum from $u$ and go right.

Why does that sample correctly? Picture the leaves laid end to end on a number line, each with a stretch as long as its priority, as in Figure 3. The walk finds the leaf whose stretch contains $u$. Leaf $i$ owns a stretch of length $p_i^\alpha$, so it comes up with probability $p_i^\alpha / \text{total}$, which is exactly $P(i)$.

```figure
<svg viewBox="0 0 620 365" role="img" aria-label="A sum tree over leaves 3, 1, 2, 4 with the walk for mass 7.5 highlighted, above a number line that gives each leaf a stretch as long as its priority">
<line x1="300.0" y1="66.0" x2="180.0" y2="104.0" class="stroke-muted"/>
<line x1="300.0" y1="66.0" x2="420.0" y2="104.0" class="stroke-b"/>
<line x1="180.0" y1="136.0" x2="120.0" y2="174.0" class="stroke-muted"/>
<line x1="180.0" y1="136.0" x2="240.0" y2="174.0" class="stroke-muted"/>
<line x1="420.0" y1="136.0" x2="360.0" y2="174.0" class="stroke-muted"/>
<line x1="420.0" y1="136.0" x2="480.0" y2="174.0" class="stroke-b"/>
<rect x="270.0" y="34.0" width="60.0" height="32.0" class="fill-soft-b stroke-b"/>
<text x="300.0" y="55.0" class="fig-bold" text-anchor="middle">10</text>
<rect x="150.0" y="104.0" width="60.0" height="32.0" class="fill-sunk stroke-muted"/>
<text x="180.0" y="125.0" class="fig-bold" text-anchor="middle">4</text>
<rect x="390.0" y="104.0" width="60.0" height="32.0" class="fill-soft-b stroke-b"/>
<text x="420.0" y="125.0" class="fig-bold" text-anchor="middle">6</text>
<rect x="90.0" y="174.0" width="60.0" height="32.0" class="fill-sunk stroke-muted"/>
<text x="120.0" y="195.0" class="fig-bold" text-anchor="middle">3</text>
<rect x="210.0" y="174.0" width="60.0" height="32.0" class="fill-sunk stroke-muted"/>
<text x="240.0" y="195.0" class="fig-bold" text-anchor="middle">1</text>
<rect x="330.0" y="174.0" width="60.0" height="32.0" class="fill-sunk stroke-muted"/>
<text x="360.0" y="195.0" class="fig-bold" text-anchor="middle">2</text>
<rect x="450.0" y="174.0" width="60.0" height="32.0" class="fill-soft-b stroke-b"/>
<text x="480.0" y="195.0" class="fig-bold" text-anchor="middle">4</text>
<text x="120.0" y="226.0" class="fig-small" text-anchor="middle">leaf 0</text>
<text x="240.0" y="226.0" class="fig-small" text-anchor="middle">leaf 1</text>
<text x="360.0" y="226.0" class="fig-small" text-anchor="middle">leaf 2</text>
<text x="480.0" y="226.0" class="fig-small" text-anchor="middle">leaf 3</text>
<text x="262.0" y="54.0" class="fig-small" text-anchor="end">root = total</text>
<text x="595.0" y="72.0" class="fig-small" text-anchor="end">u = 7.5 >= 4: subtract 4, go right</text>
<text x="595.0" y="252.0" class="fig-small" text-anchor="end">3.5 >= 2: subtract 2, go right, reach leaf 3</text>
<rect x="40.0" y="300.0" width="156.0" height="26.0" class="fill-soft-a stroke-muted"/>
<text x="118.0" y="318.0" class="fig-small" text-anchor="middle">leaf 0</text>
<rect x="196.0" y="300.0" width="52.0" height="26.0" class="fill-soft-c stroke-muted"/>
<text x="222.0" y="318.0" class="fig-small" text-anchor="middle">leaf 1</text>
<rect x="248.0" y="300.0" width="104.0" height="26.0" class="fill-soft-a stroke-muted"/>
<text x="300.0" y="318.0" class="fig-small" text-anchor="middle">leaf 2</text>
<rect x="352.0" y="300.0" width="208.0" height="26.0" class="fill-soft-c stroke-muted"/>
<text x="456.0" y="318.0" class="fig-small" text-anchor="middle">leaf 3</text>
<text x="40.0" y="344.0" class="fig-small" text-anchor="middle">0</text>
<text x="196.0" y="344.0" class="fig-small" text-anchor="middle">3</text>
<text x="248.0" y="344.0" class="fig-small" text-anchor="middle">4</text>
<text x="352.0" y="344.0" class="fig-small" text-anchor="middle">6</text>
<text x="560.0" y="344.0" class="fig-small" text-anchor="middle">10</text>
<line x1="430.0" y1="286.0" x2="430.0" y2="326.0" class="stroke-b"/>
<text x="430.0" y="280.0" class="fig-small fig-bold" text-anchor="middle">u = 7.5</text>
<line x1="206.4" y1="286.0" x2="206.4" y2="326.0" class="stroke-b"/>
<text x="206.4" y="280.0" class="fig-small fig-bold" text-anchor="middle">u = 3.2</text>
</svg>
Figure 3. Each inner node holds the sum of its children, and the walk for u = 7.5 (orange) visits one node per level. The number line shows why it works: leaf i owns a stretch of length p_i, so a uniform u lands on leaf i with probability p_i divided by 10.
```

Take four leaves $[3, 1, 2, 4]$, so the inner nodes are 4 and 6 and the root is 10.

- For $u = 7.5$: $7.5 \ge 4$, so subtract 4 and go right with 3.5. Then $3.5 \ge 2$, so subtract 2 and go right. You land on leaf 3.
- For $u = 3.2$: $3.2 < 4$, so go left. Then $3.2 \ge 3$, so subtract 3 and go right with 0.2. You land on leaf 1.

One array of length $2N$ holds the whole tree. Node 1 is the root, node $k$ has children $2k$ and $2k + 1$, and leaf $i$ sits at node $N + i$. Updating a priority sets its leaf and then walks up through the parents, $k \to k // 2$, resetting each to the sum of its children. Both operations visit one node per level, so they cost $O(\log N)$:

```python
node = 1
while node < capacity:
    left = 2 * node
    if mass < nodes[left]:
        node = left
    else:
        mass, node = mass - nodes[left], left + 1
return node - capacity
```

To draw a batch of $k$, the paper uses *stratified sampling*. It splits $[0, \text{total})$ into $k$ equal slices and draws one mass inside each slice, which spreads the batch across the buffer.

> [!RECAP]
> - The sampling probability is $p^\alpha / \sum p^\alpha$ with $p = |\delta| + \epsilon$, and the tree stores $p^\alpha$.
> - The importance weight is $(N P)^{-\beta}$ divided by its maximum over the buffer.
> - A sum tree is an array of length $2N$. `find` goes left or right by comparing the mass with the left child's sum.
> - New items get the largest raw priority so far.

## Build it step by step

The checks test the pieces in this order:

1. **`SumTree`.** Allocate `np.zeros(2 * capacity)`. `update` sets a leaf and recomputes each parent. `total` reads node 1, and `leaf` reads node `capacity + i`. `find` walks down as described above. The most common mistake is moving right without subtracting the left sum. Any capacity must work, not only powers of two.
2. **`per_probabilities(priorities, alpha)` and `importance_weights(probs, beta)`.** Two lines each. The most common mistake is normalizing the weights before raising to $-\beta$.
3. **`PrioritizedReplay`.** Keep a `SumTree` as `self.tree`, a write index, the size, the largest raw priority so far starting at 1.0, and a numpy generator. `add` gives the slot `max_priority ** alpha`. `sample` draws stratified masses and computes the weights from the tree's leaves and the smallest stored probability. `update_priorities` stores `(abs(delta) + eps) ** alpha` and updates the maximum. The most common mistake is storing $p$ in the tree instead of $p^\alpha$.

## Cost and scaling

Prioritized replay adds $O(B \log N)$ time per update, to sample and to write priorities back for a batch of $B$, and a tree of $2N$ numbers. That is small next to a network's forward pass, but it makes the buffer a serial bottleneck, because every update must write its priorities before the next sample. Distributed agents such as Ape-X spread this bookkeeping across many actors.

## Where this breaks

**Stale priorities.** A transition's priority is its TD error from the last time it was replayed. If it is never replayed again, its priority never changes, even as the network improves. The $\epsilon$ term and the maximum priority for new items keep every transition in circulation.

**Overfitting to noise.** A transition with a noisy reward keeps a large TD error forever, so prioritized replay keeps replaying it. The importance weights soften this, and $\alpha$ below 1 limits it.

**Interactions.** Prioritizing by TD error interacts with every other change to the target. Rainbow (Hessel et al., 2018) combined six improvements, and its ablations found prioritized replay and multi-step targets the most important.

## Further reading

- [Schaul et al., *Prioritized experience replay* (2016)](https://arxiv.org/abs/1511.05952). Proportional and rank-based priorities, importance weights, and the sum tree.
- [Hessel et al., *Rainbow: combining improvements in deep reinforcement learning* (2018)](https://arxiv.org/abs/1710.02298). Combines six improvements and ablates each one. [Paper page](https://papers.ifkash.dev/rainbow/).
- [Horgan et al., *Distributed prioritized experience replay* (2018)](https://arxiv.org/abs/1803.00933). Ape-X, which computes priorities on many actors and shares one prioritized buffer.
- [Obando-Ceron and Castro, *Revisiting Rainbow* (2021)](https://arxiv.org/abs/2011.14826). Reruns Rainbow's ablations on small environments at a fraction of the compute.

## The exercise

You write the `SumTree` class, `per_probabilities`, `importance_weights`, and the `PrioritizedReplay` class.

The checks confirm that the sum tree's total is exact after arbitrary updates, and that `find` maps exactly $p_i$ units of mass to leaf $i$, for capacities that are not powers of two. They check the probabilities and weights against the worked example, and the $\beta = 1$ identity that undoes the prioritization exactly. They check that the replay buffer gives new items the maximum priority, stores $(|\delta| + \epsilon)^\alpha$, returns the paper's weights, and samples in proportion to $P(i)$ by a chi-squared test.
