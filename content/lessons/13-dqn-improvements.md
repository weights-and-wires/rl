---
title: Improving DQN
slug: 13-dqn-improvements
part: "Part 3 — Approximation and deep value methods"
summary: Add Double DQN, dueling heads, prioritised replay with a sum tree, and n-step targets to DQN, and verify each exactly.
minutes: 60
gpu: false
objectives:
  - Compute the Double DQN target and explain why separating selection from evaluation removes the max bias.
  - Build a dueling head whose Q-values average to V, and say why the mean subtraction is needed.
  - Implement a sum tree, sample from it in proportion to p^alpha, and compute the importance weights that correct the bias.
  - Compute n-step targets that stop at episode ends, and name the bias they carry off-policy.
prereqs:
  - 08-double-learning
  - 09-n-step-and-eligibility-traces
  - 12-deep-q-networks
exercise: 13-dqn-improvements
---

# Improving DQN

DQN works, but it wastes effort in four places. Its targets are biased
upward, because the max picks whichever estimate is noisiest. Its network
learns a separate value for every action, even in states where the action
barely matters. Its buffer replays a transition the network already predicts
perfectly as often as one it gets badly wrong. And its target looks only one
reward ahead, so news of a reward creeps backward one step per update.

Each problem has a small, separate fix. Double DQN changes one line of the
target. A dueling head changes the last layer. Prioritised replay changes how
the buffer samples. n-step targets change how many rewards the target sums.
Rainbow (Hessel et al., 2018) combined these with two more ideas, and its
ablations found prioritised replay and multi-step targets the most important.

In this lesson you build all four and check each one exactly. The sum tree
at the heart of prioritised replay gets the most attention, because an
off-by-one error there silently changes which transitions you learn from.

## What you already know

- The DQN target is $r_t + \gamma (1 - d) \max_{a'} Q_{\bar\theta}(s_{t+1}, a')$,
  where $Q_{\bar\theta}$ is the target network and $d$ marks termination.
- A max over noisy estimates is biased upward, and double learning fixes it
  with two independent estimates, as in *Maximisation bias and double
  learning*.
- An n-step return sums $n$ discounted rewards before bootstrapping, as in
  *n-step returns and eligibility traces*.
- The *advantage* $A(s, a) = Q(s, a) - V(s)$ says how much better action $a$
  is than the policy's average in state $s$.
- *Importance sampling* reweights samples drawn from one distribution so
  their average matches another.

## Separate choosing from scoring with Double DQN

Picture ten dice-rolling contestants, all equally skilled. If you crown the
highest roller and then report that same roll as their skill, you
overestimate every time. Rolling again to score the winner fixes it. The
second roll is independent of the choice.

DQN's target does the first thing. The same target network picks the best
next action and reports its value. Double DQN (van Hasselt, Guez, and Silver,
2016) picks with the online network and scores with the target network:

$$
y = r_t + \gamma (1 - d) \, Q_{\bar\theta}\big(s_{t+1}, \operatorname*{arg\,max}_{a'} Q_\theta(s_{t+1}, a')\big)
$$

Read it as: ask the online network $Q_\theta$ which action looks best at the
next state, then ask the target network $Q_{\bar\theta}$ what that action is
worth. The two networks' errors are only partly correlated, so the choice no
longer rewards its own noise. In PyTorch it's a `gather`:

```python
best = next_q_online.argmax(dim=1, keepdim=True)
y = r + gamma * (1 - term) * next_q_target.gather(1, best).squeeze(1)
```

### Worked example

Take $r = 1$ and $\gamma = 0.9$. At the next state, the online network says
$[1.0, 3.0, 2.0]$ and the target network says $[2.5, 1.5, 2.0]$.

- DQN takes the target network's max, 2.5, so $y = 1 + 0.9 \times 2.5 = 3.25$.
- Double DQN lets the online network choose action 1, then reads the target
  network's value for it, 1.5. So $y = 1 + 0.9 \times 1.5 = 2.35$.

With ten actions that are all truly worth 0, and estimates carrying
independent standard normal noise, the DQN target averages 1.539, the
expected maximum of ten standard normals. The Double DQN target averages 0.
The exercise checks both over 20,000 samples.

> [!WARNING]
> Using the target network for both the choice and the score is plain DQN
> again, however the code is arranged. And Double DQN can underestimate, so
> a small downward bias in practice is expected, not a bug.

## Split value from advantage with a dueling head

In many states, what you do barely matters. A car on an empty straight road
has almost the same future whether it drifts left or right. A plain Q-network
still has to learn each action's value separately, from the samples where
that action was taken.

A dueling network (Wang et al., 2016) splits the last layer into two heads.
One outputs a single state value $V(s)$, and it learns from every sample.
The other outputs an advantage $A(s, a)$ per action. A plain sum
$V + A$ has a flaw: add 5 to $V$ and subtract 5 from every $A$, and $Q$
doesn't change. So the network can't tell which head should carry what. The
paper removes that freedom by subtracting the mean advantage:

$$
Q(s, a) = V(s) + A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a')
$$

Read it as: start from the state's value, add how much better this action
looks than the others, and subtract the average of those advantages. Here
$|\mathcal{A}|$ is the number of actions. Averaging both sides over actions
shows that the mean of $Q$ over actions is exactly $V$. Adding a constant to
every advantage cancels.

```python
q = value.reshape(-1, 1) + adv - adv.mean(dim=1, keepdim=True)
```

For example, $V = 2$ and $A = [1, -1, 3]$ give a mean advantage of 1, so
$Q = [2, 0, 4]$, whose mean is 2. The paper also tried subtracting the max
advantage instead, which makes $V$ equal $\max_a Q$. It found the mean
version more stable.

> [!KEY]
> Double DQN changes the target: choose with the online network, score with
> the target network. The dueling head changes the output: $Q = V + A -
> \operatorname{mean}(A)$, which pins down $V$ as the average of $Q$.

## Replay what surprises you: prioritised replay

A student who rereads every flashcard equally spends most of the time on
cards they already know. Prioritised experience replay (Schaul et al., 2016)
replays transitions with large TD errors more often. The proportional variant
gives transition $i$ a priority $p_i = |\delta_i| + \epsilon$, where
$\delta_i$ is its latest TD error and $\epsilon$ keeps every priority above
zero. It samples with probability

$$
P(i) = \frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}}
$$

Read it as: raise each priority to the power $\alpha$, then normalise.
$\alpha = 0$ is uniform replay, and $\alpha = 1$ is fully proportional. The
paper uses $\alpha = 0.6$. New transitions haven't been scored yet, so they
get the largest priority so far and are replayed soon.

Sampling this way changes the data distribution, which biases the updates.
Importance-sampling weights undo the bias:

$$
w_i = \frac{\big(N \cdot P(i)\big)^{-\beta}}{\max_j \big(N \cdot P(j)\big)^{-\beta}}
$$

Read it as: $N \cdot P(i)$ is how many times more often than uniform
transition $i$ is sampled. Raising it to $-\beta$ shrinks the update for
over-sampled transitions. Dividing by the largest weight keeps every weight
at most 1, so updates only ever shrink. With $\beta = 1$ the correction is
exact, and the paper anneals $\beta$ from 0.4 to 1 over training. Each
sampled TD error's loss is multiplied by its $w_i$.

The paper's pseudocode divides by the largest weight without saying over
which set. This exercise uses the largest over the whole buffer, as common
implementations do. Some implementations use the batch's largest weight
instead.

### Worked example

Take four transitions with priorities $p = [2, 1, 0.5, 0.5]$, and use
$\alpha = 0.6$ and $\beta = 0.4$.

- $p^{0.6} = [1.5157, 1, 0.6598, 0.6598]$, which sum to 3.8352.
- $P = [0.3952, 0.2607, 0.1720, 0.1720]$.
- $N \cdot P = [1.5808, 1.0430, 0.6881, 0.6881]$, and raising those to $-0.4$
  gives $[0.8326, 0.9833, 1.1613, 1.1613]$.
- Dividing by the largest, 1.1613, gives $w = [0.7170, 0.8467, 1, 1]$.

The most-replayed transition gets the smallest weight.

## Sample in logarithmic time with a sum tree

Sampling from $P$ by scanning a cumulative sum costs $O(N)$ per draw, which
is too slow for a buffer of a million. A *sum tree* is a binary tree whose
leaves hold the $p_i^\alpha$ and whose every inner node holds the sum of its
two children. The root holds the total.

To sample, draw a mass $u$ uniformly from $[0, \text{total})$ and walk down
from the root. At each node, if $u$ is less than the left child's sum, go
left. Otherwise subtract the left child's sum from $u$ and go right. You land
on leaf $i$ exactly when $u$ falls in leaf $i$'s stretch of the total, which
has length $p_i^\alpha$. So leaf $i$ comes up with probability
$p_i^\alpha / \text{total} = P(i)$.

One array of length $2N$ holds the whole tree. Node 1 is the root, node $k$
has children $2k$ and $2k + 1$, and leaf $i$ sits at node $N + i$. Updating
a priority sets its leaf, then walks up through the parents, $k \to k // 2$,
resetting each to the sum of its children. Both operations visit one node per
level, so they cost $O(\log N)$:

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

### Worked example

Four leaves hold $[3, 1, 2, 4]$. The inner nodes are $3 + 1 = 4$ on the left
and $2 + 4 = 6$ on the right, and the root is 10.

- For $u = 7.5$: $7.5 \ge 4$, so subtract 4 and go right with 3.5. Then
  $3.5 \ge 2$, so subtract 2 and go right. You land on leaf 3.
- For $u = 3.2$: $3.2 < 4$, so go left. Then $3.2 \ge 3$, so go right with
  0.2. You land on leaf 1.

Leaf 3 owns the masses from 6 to 10, a stretch of length 4, and leaf 1 owns
3 to 4. The paper draws a batch of $k$ by *stratified sampling*. It splits
$[0, \text{total})$ into $k$ equal slices and draws one mass in each, which
spreads the batch across the buffer.

> [!KEY]
> A sum tree samples leaf $i$ with probability $p_i^\alpha / \sum_k p_k^\alpha$
> in $O(\log N)$ time. The importance weights $(N P(i))^{-\beta}$, divided by
> their maximum, correct the bias this sampling introduces.

## Look further ahead with n-step targets

With a one-step target, a reward at the end of a long corridor takes one
update per step to travel back to the start. An n-step target sums $n$ real
rewards before it bootstraps:

$$
y^{(n)}_t = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n \max_{a'} Q_{\bar\theta}(s_{t+n}, a')
$$

Read it as: add the next $n$ discounted rewards, then add the discounted
target-network value of the state $n$ steps ahead. If the episode terminates
before $n$ steps, the sum stops at the terminal step and nothing is
bootstrapped. Rainbow uses $n = 3$.

The catch is off-policy bias. The rewards after step $t$ came from the
behaviour policy, including its random epsilon-greedy actions, not from the
greedy policy the target assumes. For small $n$ the bias is usually a fair
price for faster propagation. Retrace (Munos et al., 2016) corrects it with
truncated importance weights.

### Worked example

An episode pays $[1, 0, 2, 1]$ and terminates after the fourth step. Every
bootstrap value is 5, $\gamma = 0.9$, and $n = 3$.

- $t = 0$: $1 + 0.9 \times 0 + 0.81 \times 2 + 0.729 \times 5 = 6.265$.
- $t = 1$: $0 + 0.9 \times 2 + 0.81 \times 1 = 2.61$. The third reward
  ends the episode, so there's no bootstrap.
- $t = 2$: $2 + 0.9 \times 1 = 2.9$, and $t = 3$: $1$.

> [!RECAP]
> - Double DQN chooses $a^*$ with the online network and scores it with the
>   target network.
> - The dueling head returns $V + A - \operatorname{mean}(A)$, so the mean of
>   $Q$ is $V$.
> - A sum tree stores sums in an array of length $2N$. `find` walks left or
>   right by comparing the mass with the left child's sum.
> - Prioritised replay samples with $p^\alpha / \sum p^\alpha$ and weights
>   by $(N P)^{-\beta}$ over its maximum. New items get the max priority.
> - n-step targets stop at termination and bootstrap after $n$ rewards or at
>   the end of the data.

## Build it step by step

The checks test the pieces in this order:

1. **`double_dqn_targets(rewards, next_q_online, next_q_target, terminated,
   gamma)`.** An `argmax` with `keepdim=True`, then a `gather`. Common
   mistake: taking the argmax of the target network, which reproduces DQN.
   The bias check catches it.
2. **`dueling_combine(value, advantages)`.** Reshape `value` to `(B, 1)` and
   subtract `advantages.mean(dim=1, keepdim=True)`. Common mistake: taking
   the mean over the batch instead of over actions.
3. **`SumTree`.** Allocate `np.zeros(2 * capacity)`. `update` sets a leaf and
   recomputes each parent. `total` reads node 1, and `leaf` reads node
   `capacity + i`. `find` walks down as described. Common mistake: moving
   right without subtracting the left sum. Any capacity must work, not only
   powers of two.
4. **`per_probabilities(priorities, alpha)` and `importance_weights(probs,
   beta)`.** Two lines each. Common mistake: normalising the weights before
   raising to $-\beta$.
5. **`PrioritizedReplay`.** Keep a `SumTree` as `self.tree`, a write index,
   the size, the largest raw priority so far starting at 1.0, and a numpy
   generator. `add` gives the slot `max_priority ** alpha`. `sample` draws
   stratified masses and computes the weights from the tree's leaves and the
   smallest stored probability. `update_priorities` stores
   `(abs(delta) + eps) ** alpha` and updates the max. Common mistake:
   storing $p$ in the tree instead of $p^\alpha$.
6. **`n_step_targets(rewards, terminated, bootstrap, gamma, n)`.** A double
   loop over $t$ and $k$. Common mistake: bootstrapping after a termination,
   or using $\gamma^{n}$ when the sum stopped early at the end of the data.

## Cost and scaling

Each improvement adds little compute. The following table shows the extra
cost of each one over plain DQN, for a buffer of $N$ transitions and batches
of $B$:

| Change | Extra time per update | Extra memory |
|---|---|---|
| Double DQN | One online forward pass at $s_{t+1}$ | None |
| Dueling head | One extra output unit | A few parameters |
| Prioritised replay | $O(B \log N)$ to sample and to update priorities | A tree of $2N$ numbers |
| n-step targets | $O(n)$ per stored transition | A queue of the last $n$ steps |

Prioritised replay's cost is small next to a network's forward pass, but it
serialises the buffer: every update has to write priorities back before the
next sample. Distributed agents such as Ape-X spread that bookkeeping across
many actors.

## Where this breaks

**Stale priorities.** A transition's priority is its TD error from the last
time it was replayed. If it's never replayed again, its priority never
changes, even as the network improves. The $\epsilon$ term and new-item max
priority keep every transition in circulation.

**Overfitting to outliers.** A transition with a noisy reward keeps a large
TD error forever, so prioritised replay keeps replaying it. The
importance weights soften this, and $\alpha$ below 1 limits it.

**n-step bias.** With a random behaviour policy and large $n$, the target
estimates the behaviour policy's value, not the greedy one. Keep $n$ small,
or correct with Retrace.

**Interactions.** The improvements aren't independent. Prioritising by TD
error changes which transitions the Double DQN target sees, and n-step
targets change the TD errors that set priorities. Rainbow's ablations are the
best guide to what matters together.

## Further reading

- [van Hasselt, Guez, and Silver, *Deep reinforcement learning with double
  Q-learning* (2016)](https://arxiv.org/abs/1509.06461). The Double DQN
  target, and evidence that DQN overestimates on Atari.
- [Wang et al., *Dueling network architectures for deep reinforcement
  learning* (2016)](https://arxiv.org/abs/1511.06581). The value and
  advantage streams, and the mean-subtracted combination.
- [Schaul et al., *Prioritized experience replay*
  (2016)](https://arxiv.org/abs/1511.05952). Proportional and rank-based
  priorities, importance weights, and the sum tree.
- [Hessel et al., *Rainbow: combining improvements in deep reinforcement
  learning* (2018)](https://arxiv.org/abs/1710.02298). Combines six
  improvements and ablates each one.
- [Munos et al., *Safe and efficient off-policy reinforcement learning*
  (2016)](https://arxiv.org/abs/1606.02647). Retrace, which corrects
  multi-step targets for off-policy data.
- [Obando-Ceron and Castro, *Revisiting Rainbow*
  (2021)](https://arxiv.org/abs/2011.14826). Reruns Rainbow's ablations on
  small environments at a fraction of the compute.

## The exercise

You write `double_dqn_targets`, `dueling_combine`, the `SumTree` class,
`per_probabilities`, `importance_weights`, the `PrioritizedReplay` class, and
`n_step_targets`.

The checks verify the Double DQN target against hand values, and confirm
that it averages 0 where the plain max averages 1.539. They check that the
dueling combination averages to $V$ and ignores a shift in the advantages.
They confirm that the sum tree's total is exact after arbitrary updates, and
that `find` maps exactly $p_i$ units of mass to leaf $i$, for capacities that
aren't powers of two. They check the probabilities and weights against the
worked example and the $\beta = 1$ unbiasedness identity. They check that the
replay buffer gives new items the max priority, stores $(|\delta| +
\epsilon)^\alpha$, returns the paper's weights, and samples in proportion to
$P(i)$ by a chi-squared test. Finally, the n-step targets must match the hand
example, reduce to one step at $n = 1$, and reduce to the discounted return
for large $n$.
