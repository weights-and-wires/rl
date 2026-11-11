---
title: Maximum-entropy reinforcement learning
slug: 041-maximum-entropy-rl
part: "Part 6 — Continuous control and maximum entropy"
summary: Pay the agent for acting randomly, and the hard max in the Bellman equation becomes a soft max that gives a Boltzmann policy.
minutes: 45
gpu: false
objectives:
  - Explain why adding an entropy bonus to the reward turns the best policy into a Boltzmann distribution.
  - Compute the soft value alpha * logsumexp(Q / alpha) without overflow, and the Boltzmann policy that goes with it.
  - Run soft value iteration, and bound what the temperature costs with alpha log A / (1 - gamma).
  - Measure the soft Bellman residual on sampled transitions.
prereqs:
  - 012-value-iteration
  - 006-gradient-bandits
exercise: 041-maximum-entropy-rl
---

# Maximum-entropy reinforcement learning

Picture a hallway with two doors that lead to the same prize. An ordinary reinforcement learning agent picks one, never tries the other, and forgets that it exists. If door one later jams, the agent has nothing to fall back on. Worse, when one door is better by a hair, the agent commits to it completely, so a small error in its estimates becomes a large error in its behavior.

The agent is brittle because its goal says nothing about keeping options open. Maximum-entropy reinforcement learning changes the goal. The agent earns its reward plus a bonus for acting randomly, so among the ways to succeed it prefers the ones that keep its choices spread out.

This chapter solves that problem exactly on small MDPs. You will see that the hard max in the Bellman equation turns into a soft max, and that the best policy turns into a Boltzmann distribution. The next chapter approximates the same equations with neural networks.

## What you already know

- Value iteration applies the backup $Q(s, a) \leftarrow r(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \max_{a'} Q(s', a')$ until nothing changes (see [Value iteration and contraction](/lessons/012-value-iteration)).
- That backup is a contraction, so value iteration converges from any start.
- A softmax turns a vector of scores into probabilities that sum to one.
- A softmax policy turns preferences into action probabilities, as in [Gradient bandits](/lessons/006-gradient-bandits).

## Why not add randomness afterward?

The obvious fix is $\varepsilon$-greedy: act greedily, and with probability $\varepsilon$ pick a random action. That adds randomness, but the agent does not value it. It plans as if it will always act greedily, so it cannot choose a route because that route leaves it more room to move. To make the agent plan for flexibility, flexibility has to be in the reward.

Measure it with *entropy*. The entropy of a distribution $p$ over actions is

$$
\mathcal{H}(p) = -\sum_a p(a) \log p(a)
$$

Read it as the average surprise of one draw from $p$. A policy that always picks one action has entropy 0. The uniform policy over $A$ actions has the largest entropy, $\log A$, which is $\log 2 = 0.6931$ for two actions. This chapter uses the natural logarithm, so the unit is the *nat*. In numpy it is `-(p * np.log(p)).sum(axis=-1)`, with any $0 \log 0$ term counted as 0.

The maximum-entropy objective adds the entropy of the action choice at each visited state to the reward:

$$
J_\alpha(\pi) = \mathbb{E}_\pi\left[\sum_{t \ge 0} \gamma^t \big(r_t + \alpha\, \mathcal{H}(\pi(\cdot \mid s_t))\big)\right]
$$

Read it as the ordinary discounted return plus $\alpha$ times the entropy at every step. The *temperature* $\alpha > 0$ is the exchange rate between reward and randomness. At $\alpha = 0$ you are back to ordinary reinforcement learning, and as $\alpha$ grows, randomness is worth more than reward.

## Find the best distribution at one state

Start with a single state where you already know each action's value $Q(a)$. Which distribution $p$ maximizes $\sum_a p(a) Q(a) + \alpha \mathcal{H}(p)$? Take $Q = (1, 2)$ and $\alpha = 0.5$. Guess $p = (0.1192, 0.8808)$ and score it:

- The expected value is $0.1192 \times 1 + 0.8808 \times 2 = 1.8808$.
- The entropy is $0.3653$ nats, and $\alpha$ times it is $0.1827$.
- The total is $1.8808 + 0.1827 = 2.0635$.

No other distribution scores higher, and calculus with a Lagrange multiplier shows why. The winner is the *Boltzmann distribution*, a softmax of $Q / \alpha$, and its score is the *soft maximum*:

$$
V = \alpha \log \sum_a \exp\big(Q(a) / \alpha\big), \qquad
p^*(a) = \exp\big((Q(a) - V) / \alpha\big)
$$

Read the first formula as: divide each value by the temperature, take a log-sum-exp, and multiply back. Read the second as: an action's probability falls off exponentially with how far its value sits below the soft maximum. Dividing by $e^{V/\alpha}$ is what makes the probabilities sum to one. For the example, $e^{2} + e^{4} = 61.9872$, so $V = 0.5 \log 61.9872 = 2.0635$, a little above the hard max of 2, and $p^* = (e^2, e^4) / 61.9872 = (0.1192, 0.8808)$. The guess was the answer.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Grouped bars: the probability of the worse action (Q = 1) and the better action (Q = 2) at temperatures 0.1, 0.5, 2, and 10; the better action dominates when the temperature is low and the two approach one half as it grows">
<line x1="70.0" y1="240.0" x2="570.0" y2="240.0" class="stroke-grid"/>
<text x="62.0" y="244.0" class="fig-small" text-anchor="end">0.0</text>
<line x1="70.0" y1="140.0" x2="570.0" y2="140.0" class="stroke-grid"/>
<text x="62.0" y="144.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="70.0" y1="40.0" x2="570.0" y2="40.0" class="stroke-grid"/>
<text x="62.0" y="44.0" class="fig-small" text-anchor="end">1.0</text>
<line x1="70.0" y1="240.0" x2="570.0" y2="240.0" class="stroke-muted"/>
<rect x="100.0" y="240.0" width="38.0" height="0.0" class="fill-a"/>
<rect x="142.0" y="40.0" width="38.0" height="200.0" class="fill-b"/>
<text x="119.0" y="234.0" class="fig-small" text-anchor="middle">0.000</text>
<text x="161.0" y="34.0" class="fig-small" text-anchor="middle">1.000</text>
<text x="140.0" y="260.0" class="fig-small" text-anchor="middle">α = 0.1</text>
<rect x="225.0" y="216.2" width="38.0" height="23.8" class="fill-a"/>
<rect x="267.0" y="63.8" width="38.0" height="176.2" class="fill-b"/>
<text x="244.0" y="210.2" class="fig-small" text-anchor="middle">0.119</text>
<text x="286.0" y="57.8" class="fig-small" text-anchor="middle">0.881</text>
<text x="265.0" y="260.0" class="fig-small" text-anchor="middle">α = 0.5</text>
<rect x="350.0" y="164.5" width="38.0" height="75.5" class="fill-a"/>
<rect x="392.0" y="115.5" width="38.0" height="124.5" class="fill-b"/>
<text x="369.0" y="158.5" class="fig-small" text-anchor="middle">0.378</text>
<text x="411.0" y="109.5" class="fig-small" text-anchor="middle">0.622</text>
<text x="390.0" y="260.0" class="fig-small" text-anchor="middle">α = 2</text>
<rect x="475.0" y="145.0" width="38.0" height="95.0" class="fill-a"/>
<rect x="517.0" y="135.0" width="38.0" height="105.0" class="fill-b"/>
<text x="494.0" y="139.0" class="fig-small" text-anchor="middle">0.475</text>
<text x="536.0" y="129.0" class="fig-small" text-anchor="middle">0.525</text>
<text x="515.0" y="260.0" class="fig-small" text-anchor="middle">α = 10</text>
<rect x="330.0" y="276.0" width="14.0" height="14.0" class="fill-a"/>
<text x="350.0" y="288.0" class="fig-small" text-anchor="start">action with Q = 1</text>
<rect x="470.0" y="276.0" width="14.0" height="14.0" class="fill-b"/>
<text x="490.0" y="288.0" class="fig-small" text-anchor="start">Q = 2</text>
<text x="62.0" y="28.0" class="fig-small" text-anchor="end">probability</text>
</svg>
Figure 1. The Boltzmann policy for Q = (1, 2). Notice that a low temperature puts almost all the probability on the better action, and a high temperature pulls the two toward one half each.
```

Figure 1 shows the two limits. As $\alpha \to 0$, the soft maximum becomes $\max_a Q(a)$ and $p^*$ puts all its mass on the best action. As $\alpha \to \infty$, $p^*$ becomes uniform. In numpy, subtract the row maximum first, or `exp` overflows:

```python
top = q.max(axis=-1)
v = top + alpha * np.log(np.exp((q - top[..., None]) / alpha).sum(axis=-1))
pi = np.exp((q - v[..., None]) / alpha)
```

> [!WARNING]
> With $Q$ near 1000 and $\alpha = 0.01$, the naive `np.exp(q / alpha)` is `inf`. Subtracting the row maximum first keeps every exponent at or below zero.

## Put the soft max into the Bellman equation

In an MDP the value of an action includes the entropy you will collect later. Replace the max over next actions in value iteration with the soft maximum, and you get the *soft Bellman equation*:

$$
Q^*_\alpha(s, a) = r(s, a) + \gamma \sum_{s'} P(s' \mid s, a)\, V^*_\alpha(s'), \qquad
V^*_\alpha(s) = \alpha \log \sum_{a} \exp\big(Q^*_\alpha(s, a) / \alpha\big)
$$

Read it as ordinary Bellman optimality with the hard max swapped for the soft one. The best policy is the Boltzmann policy of $Q^*_\alpha$. The soft maximum moves by at most as much as its largest input moves, so the soft backup is a contraction by $\gamma$, like the hard one. Iterate it from $Q = 0$ and it converges. That is *soft value iteration*.

Try it on two states. In each, action 0 stays and action 1 switches to the other state. Staying in state 1 pays 1, and everything else pays 0. Set $\gamma = 0.9$ and $\alpha = 1$.

- From $Q = 0$, both states have soft value $\log 2 = 0.6931$.
- First backup: $Q = \begin{bmatrix} 0.6238 & 0.6238 \\ 1.6238 & 0.6238 \end{bmatrix}$, since $0.9 \times 0.6931 = 0.6238$.
- Soft values: $V(0) = \log(2 e^{0.6238}) = 1.3170$ and $V(1) = \log(e^{1.6238} + e^{0.6238}) = 1.9371$.
- Second backup: $Q = \begin{bmatrix} 1.1853 & 1.7434 \\ 2.7434 & 1.1853 \end{bmatrix}$.

Keep going and it converges to $V^*_1 = (10.9377, 11.6998)$. The hard optimum is $V^* = (9, 10)$ with a deterministic policy. Each soft value is higher because it counts the entropy collected forever, about $\log 2 / (1 - \gamma) = 6.9$ at most. The Boltzmann policy switches out of state 0 with probability 0.665 and stays in state 1 with probability 0.844. It prefers the better move without ruling out the other.

In numpy, one backup reuses `rlenvs.q_from_v`:

```python
v = np.where(mdp.terminal, 0.0, soft_value(q, alpha))
q = rlenvs.q_from_v(mdp, v)
```

> [!WARNING]
> A terminal state's row of $Q$ is all zeros, and the soft maximum of zeros is $\alpha \log A$, not 0. Mask terminal states before the backup, or every episode ends with a phantom entropy bonus.

## Count what the temperature costs

The soft-optimal policy gives up some reward to stay random. How much? Each step's bonus lies between 0 and $\alpha \log A$, so summed with discounting it lies between 0 and $\alpha \log A / (1 - \gamma)$. The soft value is never below the hard value, since the hard-optimal policy is one choice the soft agent could make. So

$$
0 \le V^*_\alpha(s) - V^*(s) \le \frac{\alpha \log A}{1 - \gamma}
$$

As $\alpha \to 0$ the gap closes and soft value iteration becomes ordinary value iteration. On the frozen lake with $\gamma = 0.95$ and four actions, the hard value at the start is $0.1374$. At $\alpha = 1$ the soft value is $9.7011$, because the entropy bonus dwarfs the single reward of 1 at the goal. At $\alpha = 0.001$ the two differ by $3 \times 10^{-5}$.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Log scale plot of the soft value minus the hard value at the frozen lake start against the temperature, with the upper bound alpha log 4 over 1 minus gamma above it; both fall as the temperature falls">
<line x1="80.0" y1="220.0" x2="560.0" y2="220.0" class="stroke-grid"/>
<text x="72.0" y="224.0" class="fig-small" text-anchor="end">1e-4</text>
<line x1="80.0" y1="190.0" x2="560.0" y2="190.0" class="stroke-grid"/>
<text x="72.0" y="194.0" class="fig-small" text-anchor="end">1e-3</text>
<line x1="80.0" y1="160.0" x2="560.0" y2="160.0" class="stroke-grid"/>
<text x="72.0" y="164.0" class="fig-small" text-anchor="end">1e-2</text>
<line x1="80.0" y1="130.0" x2="560.0" y2="130.0" class="stroke-grid"/>
<text x="72.0" y="134.0" class="fig-small" text-anchor="end">1e-1</text>
<line x1="80.0" y1="100.0" x2="560.0" y2="100.0" class="stroke-grid"/>
<text x="72.0" y="104.0" class="fig-small" text-anchor="end">1</text>
<line x1="80.0" y1="70.0" x2="560.0" y2="70.0" class="stroke-grid"/>
<text x="72.0" y="74.0" class="fig-small" text-anchor="end">1e1</text>
<line x1="80.0" y1="250.0" x2="560.0" y2="250.0" class="stroke-muted"/>
<line x1="540.0" y1="250.0" x2="540.0" y2="255.0" class="stroke-muted"/>
<text x="540.0" y="270.0" class="fig-small" text-anchor="middle">1</text>
<line x1="393.3" y1="250.0" x2="393.3" y2="255.0" class="stroke-muted"/>
<text x="393.3" y="270.0" class="fig-small" text-anchor="middle">0.1</text>
<line x1="246.7" y1="250.0" x2="246.7" y2="255.0" class="stroke-muted"/>
<text x="246.7" y="270.0" class="fig-small" text-anchor="middle">0.01</text>
<line x1="100.0" y1="250.0" x2="100.0" y2="255.0" class="stroke-muted"/>
<text x="100.0" y="270.0" class="fig-small" text-anchor="middle">0.001</text>
<text x="320.0" y="290.0" text-anchor="middle">temperature α</text>
<polyline points="540.0,56.7 393.3,86.7 246.7,116.7 100.0,146.7" class="stroke-muted dashed"/>
<polyline points="540.0,70.6 393.3,102.2 246.7,144.8 100.0,235.4" class="stroke-b"/>
<circle cx="540.0" cy="70.6" r="5" class="fill-b"/>
<circle cx="393.3" cy="102.2" r="5" class="fill-b"/>
<circle cx="246.7" cy="144.8" r="5" class="fill-b"/>
<circle cx="100.0" cy="235.4" r="5" class="fill-b"/>
<circle cx="540.0" cy="56.7" r="4" class="fill-muted"/>
<circle cx="393.3" cy="86.7" r="4" class="fill-muted"/>
<circle cx="246.7" cy="116.7" r="4" class="fill-muted"/>
<circle cx="100.0" cy="146.7" r="4" class="fill-muted"/>
<text x="540.0" y="94.6" class="fig-small" text-anchor="end">gap V_soft - V*</text>
<text x="540.0" y="44.7" class="fig-small" text-anchor="end">bound α log 4 / (1 - γ)</text>
<text x="80.0" y="24.0" class="fig-small" text-anchor="start">value (log scale)</text>
</svg>
Figure 2. On the frozen lake with γ = 0.95, the soft value of the start exceeds the hard value by the orange amount, and never by more than the dashed bound. Notice that the bound falls tenfold with each tenfold drop in temperature, while the gap falls faster, because a low-temperature Boltzmann policy is nearly the greedy one.
```

> [!KEY]
> Paying for entropy turns the hard max into a soft max and the best policy into a Boltzmann distribution. The temperature costs at most $\alpha \log A / (1 - \gamma)$ in value, and as $\alpha \to 0$ the soft Bellman equation becomes the ordinary one. That answers the opening question: an agent that values randomness keeps the second door open, and you choose how much that is worth.

## Measure the soft Bellman residual

With a model you back up every state at once. Without one you have sampled transitions $(s, a, r, s', \text{terminated})$. The *soft Bellman residual* of one transition is

$$
\delta = r + \gamma\, (1 - \text{terminated})\, V_\alpha(s') - Q(s, a), \qquad
V_\alpha(s') = \alpha \log \sum_{a'} \exp\big(Q(s', a') / \alpha\big)
$$

Read it as the TD error of Q-learning with the max over next actions replaced by the soft maximum. At $Q^*_\alpha$ the residual averages to exactly zero over the next state, and anywhere else it does not. Minimizing the squared residual over a replay buffer is *soft Q-learning* (Haarnoja et al., 2017), and the next chapter's critic minimizes nearly the same quantity. In numpy, for integer arrays `s`, `a`, and `s2`:

```python
target = r + gamma * (1.0 - terminated) * soft_value(q[s2], alpha)
delta = target - q[s, a]
```

> [!RECAP]
> - Entropy $\mathcal{H}(p) = -\sum_a p(a)\log p(a)$ is largest for the uniform policy, at $\log A$.
> - The best distribution for $\sum_a p(a)Q(a) + \alpha\mathcal{H}(p)$ is the Boltzmann distribution, and its score is $\alpha \log \sum_a e^{Q(a)/\alpha}$.
> - Soft value iteration is value iteration with the soft maximum, and terminal states have soft value 0.
> - The soft value exceeds the hard value by between 0 and $\alpha \log A / (1 - \gamma)$.
> - The residual uses `soft_value(q[s2], alpha)` times `1 - terminated`.

## Build it step by step

The checks test five functions in this order.

1. `soft_value(q, alpha)` takes `(S, A)` and returns `(S,)`. Subtract each row's maximum, exponentiate, sum, take the log, scale by `alpha`, and add the maximum back. The common mistake is forgetting to add the maximum back.
2. `boltzmann_policy(q, alpha)` returns `(S, A)`. Reuse `soft_value` and broadcast it with `[..., None]`.
3. `soft_bellman_backup(mdp, q, alpha)` returns `(S, A)`. Compute the soft value, set it to 0 where `mdp.terminal` is true, and pass it to `rlenvs.q_from_v`. The common mistake is skipping the terminal mask.
4. `soft_value_iteration(mdp, alpha, tol)` loops backups from zeros until the largest change is below `tol`, and returns $Q^*_\alpha$.
5. `soft_bellman_residual(q, s, a, r, s2, terminated, gamma, alpha)` returns `(N,)`. Index `q[s2]` for the next states and `q[s, a]` for the current values.

## Where this breaks

- **The temperature has units.** The bonus $\alpha \mathcal{H}$ is in nats and the reward is in the environment's units. Scale the reward by 10 and the same $\alpha$ makes a very different agent. [Soft actor-critic](/lessons/042-soft-actor-critic) fixes this by tuning $\alpha$ to a target entropy.
- **You are optimizing a different objective.** With a large $\alpha$, the agent can prefer wandering to reaching the goal, as Figure 2 shows at $\alpha = 1$. Anneal $\alpha$, or report the policy's ordinary return next to its soft return.
- **Entropy is a KL penalty in disguise.** $\mathcal{H}(p) = \log A - D_{\mathrm{KL}}(p \,\Vert\, \text{uniform})$. Replace the uniform policy with a reference policy $\pi_{\text{ref}}$, and the best policy tilts the reference by $\exp(Q / \alpha)$. That is the form [language models as policies](/lessons/062-language-models-as-policies) use.

## Further reading

The following sources develop maximum-entropy reinforcement learning:

- [Ziebart (2010), Modeling purposeful adaptive behavior with the principle of maximum causal entropy](https://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf) develops the soft Bellman equations from maximum causal entropy.
- [Haarnoja, Tang, Abbeel, and Levine (2017), Reinforcement learning with deep energy-based policies](https://arxiv.org/abs/1702.08165) introduces soft Q-learning with neural networks.
- [Schulman, Chen, and Abbeel (2017), Equivalence between policy gradients and soft Q-learning](https://arxiv.org/abs/1704.06440) shows that the entropy-regularized policy gradient and soft Q-learning take the same expected step.
- [Levine (2018), Reinforcement learning and control as probabilistic inference](https://arxiv.org/abs/1805.00909) derives maximum-entropy reinforcement learning as variational inference.

## The exercise

You build soft dynamic programming from the soft maximum up. Write `soft_value`, `boltzmann_policy`, `soft_bellman_backup`, `soft_value_iteration`, and `soft_bellman_residual`.

The checks confirm that `soft_value` gives $2.0635$ for $Q = (1, 2)$ and $\alpha = 0.5$, stays finite near $Q = 1000$, and approaches the max and the mean in its two limits. They check that `boltzmann_policy` beats 3,000 random policies on expected $Q$ plus $\alpha$ times entropy, that one backup matches the two-state worked example with terminal states at zero, and that soft value iteration reaches a fixed point on a random MDP and the frozen lake. They also check the bound $0 \le V^*_\alpha - V^* \le \alpha \log A / (1 - \gamma)$ for $\alpha$ from 1 down to 0.001, and that the residual averages to exactly zero over next states at $Q^*_\alpha$.
