---
title: Double and dueling DQN
slug: 027-double-and-dueling-dqn
part: "Part 4 — Deep value methods"
summary: Fix three leaks in the DQN target and head, so the agent stops overestimating, shares what it learns across actions, and sees rewards sooner.
minutes: 40
gpu: false
objectives:
  - Explain why the max over noisy value estimates is biased upward even when the noise has mean zero.
  - Compute the Double DQN target, choosing the action with the online network and scoring it with the target network.
  - Build a dueling head whose Q-values average to V, and say why the mean subtraction is needed.
  - Compute n-step targets that stop at episode ends, and name the bias they carry off-policy.
prereqs:
  - 018-maximization-bias
  - 019-n-step-returns
  - 026-deep-q-networks
exercise: 027-double-and-dueling-dqn
---

# Double and dueling DQN

Imagine a state with ten actions, and all ten are truly worth 0. Your network has never seen the state before, so each estimate is off by a little in a random direction. A few are too high and a few are too low. When DQN builds a target from this state, it takes the maximum of the ten estimates, and the maximum is almost always one of the too-high ones. The target says the state is worth about 1.5 when it is worth 0.

That error does not stay put. The target for the state before this one includes the inflated value, so the error spreads backward through every state that leads here. This chapter fixes that leak with a one-line change to the target. It then fixes two smaller leaks in DQN with changes of the same size: a head that shares what it learns across actions, and a target that looks several rewards ahead.

Each fix is a separate idea you can add or remove. They fit in one chapter because each one changes a single line of the DQN you already built, and you check each one exactly.

## What you already know

- DQN regresses $Q_\theta(s, a)$ toward $r_t + \gamma (1 - d) \max_{a'} Q_{\bar\theta}(s_{t+1}, a')$, where $Q_{\bar\theta}$ is the target network and $d$ is 1 when the episode terminated.
- A max over noisy estimates is biased upward, which [Maximization bias and double learning](/lessons/018-maximization-bias) fixed by using two independent estimates.
- An n-step return sums $n$ discounted rewards before it bootstraps from a value estimate.
- The advantage $A(s, a) = Q(s, a) - V(s)$ says how much better action $a$ is than the policy's average action in state $s$.
- Off-policy means the data came from a different policy than the one you are evaluating.

## Why the max looks too high

If the noise is as likely to be positive as negative, why doesn't it cancel? It does cancel in an average, but not in a max. The max always keeps the largest draw, so it keeps the noise that points up and throws away the noise that points down.

Figure 1 shows the size of the effect. Each action is truly worth 0, and each estimate carries standard normal noise. With one action the max averages 0, because there is nothing to choose between. With ten actions it averages 1.539, and with twenty it averages 1.867. The more actions the state has, the more DQN overestimates.

```figure
<svg viewBox="0 0 620 310" role="img" aria-label="Average of the largest of k noisy estimates of zero-valued actions rises with k; the Double DQN target stays at zero">
<line x1="70.0" y1="250.0" x2="550.0" y2="250.0" class="stroke-grid"/>
<text x="62.0" y="254.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="195.0" x2="550.0" y2="195.0" class="stroke-grid"/>
<text x="62.0" y="199.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="70.0" y1="140.0" x2="550.0" y2="140.0" class="stroke-grid"/>
<text x="62.0" y="144.0" class="fig-small" text-anchor="end">1</text>
<line x1="70.0" y1="85.0" x2="550.0" y2="85.0" class="stroke-grid"/>
<text x="62.0" y="89.0" class="fig-small" text-anchor="end">1.5</text>
<line x1="70.0" y1="30.0" x2="550.0" y2="30.0" class="stroke-grid"/>
<text x="62.0" y="34.0" class="fig-small" text-anchor="end">2</text>
<line x1="70.0" y1="250.0" x2="70.0" y2="255.0" class="stroke-muted"/>
<text x="70.0" y="270.0" class="fig-small" text-anchor="middle">1</text>
<line x1="171.1" y1="250.0" x2="171.1" y2="255.0" class="stroke-muted"/>
<text x="171.1" y="270.0" class="fig-small" text-anchor="middle">5</text>
<line x1="297.4" y1="250.0" x2="297.4" y2="255.0" class="stroke-muted"/>
<text x="297.4" y="270.0" class="fig-small" text-anchor="middle">10</text>
<line x1="423.7" y1="250.0" x2="423.7" y2="255.0" class="stroke-muted"/>
<text x="423.7" y="270.0" class="fig-small" text-anchor="middle">15</text>
<line x1="550.0" y1="250.0" x2="550.0" y2="255.0" class="stroke-muted"/>
<text x="550.0" y="270.0" class="fig-small" text-anchor="middle">20</text>
<line x1="70.0" y1="250.0" x2="550.0" y2="250.0" class="stroke-muted"/>
<polyline points="70.0,250.0 95.3,187.9 120.5,156.9 145.8,136.8 171.1,122.1 196.3,110.6 221.6,101.3 246.8,93.4 272.1,86.6 297.4,80.7 322.6,75.5 347.9,70.8 373.2,66.5 398.4,62.6 423.7,59.0 448.9,55.7 474.2,52.7 499.5,49.8 524.7,47.1 550.0,44.6" class="stroke-b" fill="none"/>
<line x1="70.0" y1="250.0" x2="550.0" y2="250.0" class="stroke-a dashed"/>
<circle cx="297.4" cy="80.7" r="5" class="fill-b"/>
<text x="307.4" y="96.7" class="fig-small" text-anchor="start">10 actions: 1.539</text>
<text x="550.0" y="34.6" class="fig-small" text-anchor="end">max over k estimates</text>
<text x="550.0" y="242.0" class="fig-small" text-anchor="end">Double DQN: 0</text>
<text x="310.0" y="292.0" text-anchor="middle">number of actions k, each truly worth 0</text>
<text x="20.0" y="16.0" text-anchor="start">average target value</text>
</svg>
Figure 1. Every action is truly worth 0, and each estimate has standard normal noise. The max of the estimates (orange) grows with the number of actions, reaching 1.539 at ten. Choosing with one set of estimates and scoring with another (dashed) averages 0.
```

DQN's target makes the mistake in a specific way. The target network does two jobs at once: it picks the best-looking next action, and it reports that action's value. The action looks best partly because its estimate is too high, and then that same too-high estimate gets reported as its value.

## Choose with one network, score with the other

Picture ten equally skilled dice rollers. If you crown the highest roller and then report that roll as their skill, you overstate it every time. If you ask the winner to roll again, the second roll is independent of why they won, and it averages to their true skill.

Double DQN (van Hasselt, Guez, and Silver, 2016) does the second thing. The online network $Q_\theta$ picks the action, and the target network $Q_{\bar\theta}$ scores it:

$$
y = r_t + \gamma (1 - d)\, Q_{\bar\theta}\big(s_{t+1}, \operatorname*{arg\,max}_{a'} Q_\theta(s_{t+1}, a')\big)
$$

Read it from the inside out. Ask the online network which next action looks best. Then ask the target network what that action is worth, and add the reward and the discount as before. The two networks' errors are only partly related, so the target network's value for the chosen action is not inflated by the reason it was chosen.

### Work an example

Take $r = 1$ and $\gamma = 0.9$. At the next state the online network says $[1.0, 3.0, 2.0]$ and the target network says $[2.5, 1.5, 2.0]$.

- DQN takes the target network's own maximum, 2.5, so $y = 1 + 0.9 \times 2.5 = 3.25$.
- Double DQN lets the online network choose action 1, then reads the target network's value for action 1, which is 1.5. So $y = 1 + 0.9 \times 1.5 = 2.35$.

```figure
<svg viewBox="0 0 620 332" role="img" aria-label="Online and target Q-values at the next state for three actions; DQN reads the target's maximum, Double DQN reads the target value at the online network's best action">
<text x="110.0" y="28.0" class="fig-bold" text-anchor="middle">online network</text>
<line x1="20.0" y1="220.0" x2="200.0" y2="220.0" class="stroke-muted"/>
<rect x="35.0" y="184.0" width="40.0" height="36.0" class="fill-soft-a"/>
<text x="55.0" y="238.0" class="fig-small" text-anchor="middle">a0</text>
<text x="55.0" y="178.0" class="fig-small fig-bold" text-anchor="middle">1</text>
<rect x="90.0" y="112.0" width="40.0" height="108.0" class="fill-b"/>
<text x="110.0" y="238.0" class="fig-small" text-anchor="middle">a1</text>
<text x="110.0" y="106.0" class="fig-small fig-bold" text-anchor="middle">3</text>
<rect x="145.0" y="148.0" width="40.0" height="72.0" class="fill-soft-a"/>
<text x="165.0" y="238.0" class="fig-small" text-anchor="middle">a2</text>
<text x="165.0" y="142.0" class="fig-small fig-bold" text-anchor="middle">2</text>
<text x="340.0" y="28.0" class="fig-bold" text-anchor="middle">target network</text>
<line x1="250.0" y1="220.0" x2="430.0" y2="220.0" class="stroke-muted"/>
<rect x="265.0" y="130.0" width="40.0" height="90.0" class="fill-a"/>
<text x="285.0" y="238.0" class="fig-small" text-anchor="middle">a0</text>
<text x="285.0" y="124.0" class="fig-small fig-bold" text-anchor="middle">2.5</text>
<rect x="320.0" y="166.0" width="40.0" height="54.0" class="fill-b"/>
<text x="340.0" y="238.0" class="fig-small" text-anchor="middle">a1</text>
<text x="340.0" y="160.0" class="fig-small fig-bold" text-anchor="middle">1.5</text>
<rect x="375.0" y="148.0" width="40.0" height="72.0" class="fill-soft-a"/>
<text x="395.0" y="238.0" class="fig-small" text-anchor="middle">a2</text>
<text x="395.0" y="142.0" class="fig-small fig-bold" text-anchor="middle">2</text>
<text x="110.0" y="266.0" class="fig-small" text-anchor="middle">Double DQN picks a1 (orange)</text>
<text x="340.0" y="266.0" class="fig-small" text-anchor="middle">DQN reads a0 (blue, 2.5)</text>
<text x="340.0" y="284.0" class="fig-small" text-anchor="middle">Double DQN reads a1 (orange, 1.5)</text>
<text x="20.0" y="318.0" text-anchor="start">DQN: y = 1 + 0.9 x 2.5 = 3.25.   Double DQN: y = 1 + 0.9 x 1.5 = 2.35.</text>
</svg>
Figure 2. DQN reads the target network's own maximum (blue, a0, 2.5). Double DQN reads the target network's value (orange, 1.5) for the action the online network prefers (orange, a1).
```

In PyTorch the whole change is a `gather` on the target network's output:

```python
best = next_q_online.argmax(dim=1, keepdim=True)
y = r + gamma * (1 - term) * next_q_target.gather(1, best).squeeze(1)
```

Run the ten-action experiment this way and the target averages 0, where the plain max averages 1.539. The exercise checks both numbers over 20,000 samples.

> [!WARNING]
> Taking the `argmax` of the target network and then reading the target network is plain DQN again, however you arrange the code. Double DQN can also underestimate, so a small downward bias in practice is expected and is not a bug.

## Split the value from the advantage

In many states, what you do barely matters. A car on an empty straight road has almost the same future whether it drifts a little left or right. A plain Q-network still has to learn each action's value separately, and it only learns about an action from the samples where that action was taken.

Why not let the network output $Q$ directly, as DQN does? Because the part of $Q$ that every action shares, the value of the state, gets learned $|\mathcal{A}|$ separate times, once per output. A dueling network (Wang et al., 2016) gives the network two heads on top of a shared body. One head outputs a single state value $V(s)$, and every sample trains it. The other outputs an advantage $A(s, a)$ per action.

The obvious combination is $Q = V + A$, and it has a flaw. Add 5 to $V$ and subtract 5 from every $A$, and $Q$ does not change. The network has no way to tell which head should carry the 5, so neither head means what its name says. The paper removes that freedom by subtracting the mean advantage:

$$
Q(s, a) = V(s) + A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a')
$$

Read it as: start from the state's value, add how much better this action looks than the others, and subtract the average of those advantages so that they sum to zero. Averaging both sides over actions shows that the mean of $Q$ over actions is exactly $V$. Adding a constant to every advantage cancels, so $V$ has one possible meaning.

For example, $V = 2$ and $A = [1, -1, 3]$ have a mean advantage of 1, so $Q = [2, 0, 4]$, and the mean of $Q$ is 2, as Figure 3 shows.

```figure
<svg viewBox="0 0 620 250" role="img" aria-label="Dueling head: Q-values 2, 0, 4 are the state value 2 plus the mean-centered advantages 0, -2, 2">
<line x1="60.0" y1="200.0" x2="560.0" y2="200.0" class="stroke-muted"/>
<rect x="110.0" y="140.0" width="70.0" height="60.0" class="fill-a"/>
<text x="145.0" y="218.0" class="fig-small" text-anchor="middle">Q(s, a0) = 2</text>
<text x="145.0" y="132.0" class="fig-small" text-anchor="middle">A - mean(A) = 0</text>
<rect x="260.0" y="200.0" width="70.0" height="0.0" class="fill-a"/>
<text x="295.0" y="218.0" class="fig-small" text-anchor="middle">Q(s, a1) = 0</text>
<text x="295.0" y="192.0" class="fig-small" text-anchor="middle">A - mean(A) = -2</text>
<rect x="410.0" y="80.0" width="70.0" height="120.0" class="fill-a"/>
<text x="445.0" y="218.0" class="fig-small" text-anchor="middle">Q(s, a2) = 4</text>
<text x="445.0" y="72.0" class="fig-small" text-anchor="middle">A - mean(A) = +2</text>
<line x1="60.0" y1="140.0" x2="560.0" y2="140.0" class="stroke-b dashed"/>
<text x="560.0" y="28.0" class="fig-small fig-bold" text-anchor="end">dashed line: V(s) = 2 = mean of Q</text>
<text x="60.0" y="28.0" class="fig-bold" text-anchor="start">V = 2, A = [1, -1, 3], mean(A) = 1</text>
</svg>
Figure 3. The state value V sets the dashed level, and the mean-centered advantages (0, -2, +2) move each bar away from it. Because the advantages average to zero, the Q-values average to V.
```

```python
q = value.reshape(-1, 1) + adv - adv.mean(dim=1, keepdim=True)
```

The paper also tried subtracting the max advantage, which makes $V$ equal $\max_a Q$. It found the mean version more stable.

> [!KEY]
> Double DQN changes the target: choose with the online network and score with the target network, so the max stops rewarding its own noise. The dueling head changes the output: $Q = V + A - \operatorname{mean}(A)$ pins down $V$ as the average of $Q$.

## Look further ahead with n-step targets

With a one-step target, a reward at the end of a long corridor needs one update per step to travel back to the start. Waiting for the full return, as Monte Carlo does, fixes the speed but brings back all the noise of the episode. An n-step target takes the middle road. It sums $n$ real rewards, then bootstraps:

$$
y^{(n)}_t = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n \max_{a'} Q_{\bar\theta}(s_{t+n}, a')
$$

Read it as: add the next $n$ discounted rewards, then add the discounted target-network value of the state $n$ steps ahead. If the episode terminates before $n$ steps, the sum stops at the terminal step and nothing is bootstrapped, because a terminal state has no future. Rainbow (Hessel et al., 2018) uses $n = 3$.

### Work an example

An episode pays $[1, 0, 2, 1]$ and terminates after the fourth step. Every bootstrap value is 5, $\gamma = 0.9$, and $n = 3$.

- At $t = 0$: $1 + 0.9 \times 0 + 0.81 \times 2 + 0.729 \times 5 = 6.265$. Three rewards were available, so the target bootstraps.
- At $t = 1$: $0 + 0.9 \times 2 + 0.81 \times 1 = 2.61$. The third reward ends the episode, so there is no bootstrap.
- At $t = 2$: $2 + 0.9 \times 1 = 2.9$. At $t = 3$: $1$.

There is a catch. The rewards after step $t$ came from the behavior policy, including its random epsilon-greedy actions, but the bootstrap term assumes the greedy policy takes over. For small $n$ the bias is usually a fair price for faster propagation. Retrace (Munos et al., 2016) corrects it with truncated importance weights.

> [!RECAP]
> - Double DQN chooses $a^*$ with the online network and scores it with the target network, which removes the max bias.
> - The dueling head returns $V + A - \operatorname{mean}(A)$, so the mean of $Q$ is $V$.
> - An n-step target sums up to $n$ rewards, stops at termination without bootstrapping, and otherwise bootstraps from the state $n$ steps ahead.

## Build it step by step

The checks test the pieces in this order:

1. **`double_dqn_targets(rewards, next_q_online, next_q_target, terminated, gamma)`.** It takes the rewards of shape `(B,)`, two `(B, A)` tables of next-state Q-values, and a termination mask. It returns the `(B,)` targets with no gradient. Take an `argmax` with `keepdim=True`, then a `gather`. The most common mistake is taking the argmax of the target network, which reproduces DQN. The bias check catches it.
2. **`dueling_combine(value, advantages)`.** Reshape `value` to `(B, 1)` and subtract `advantages.mean(dim=1, keepdim=True)`. The most common mistake is taking the mean over the batch instead of over actions.
3. **`n_step_targets(rewards, terminated, bootstrap, gamma, n)`.** Use a double loop over $t$ and $k$, adding one discounted reward at a time. Stop when a step terminates, and bootstrap after $n$ rewards or at the end of the data. The most common mistake is bootstrapping after a termination, or using $\gamma^n$ when the sum stopped early at the end of the data.

## Where this breaks

**Double DQN can underestimate.** The two networks are correlated, because the target network is a lagged copy of the online one. The bias shrinks a lot, but it does not vanish.

**Dueling needs actions that share structure.** When every action leads somewhere very different, the shared value head has little to share.

**n-step bias grows with $n$.** With a random behavior policy and large $n$, the target estimates the behavior policy's value, not the greedy one. Keep $n$ small, or correct with Retrace.

**The fixes interact.** n-step targets change the TD errors, and Double DQN changes the targets those errors use. Rainbow's ablations are the best guide to what matters in combination.

## Further reading

- [van Hasselt, Guez, and Silver, *Deep reinforcement learning with double Q-learning* (2016)](https://arxiv.org/abs/1509.06461). The Double DQN target, and evidence that DQN overestimates on Atari.
- [Wang et al., *Dueling network architectures for deep reinforcement learning* (2016)](https://arxiv.org/abs/1511.06581). The value and advantage streams, and the mean-subtracted combination.
- [Hessel et al., *Rainbow: combining improvements in deep reinforcement learning* (2018)](https://arxiv.org/abs/1710.02298). Combines six improvements and ablates each one.
- [Munos et al., *Safe and efficient off-policy reinforcement learning* (2016)](https://arxiv.org/abs/1606.02647). Retrace, which corrects multi-step targets for off-policy data.

## The exercise

You write `double_dqn_targets`, `dueling_combine`, and `n_step_targets`.

The checks verify the Double DQN target against hand values, and confirm that it averages 0 where the plain max averages 1.539. They check that the dueling combination matches the worked example, averages to $V$, and ignores a shift in the advantages. They check that the n-step targets match the hand example, reduce to one step at $n = 1$, and reduce to the discounted return for large $n$.
