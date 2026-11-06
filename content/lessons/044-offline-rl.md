---
title: Offline reinforcement learning
slug: 044-offline-rl
part: "Part 7 — Models, exploration, and fixed data"
summary: See Q-learning on a fixed dataset trust actions it never saw, then fix it with CQL's penalty and IQL's expectile value.
minutes: 40
gpu: false
objectives:
  - Explain extrapolation error, and measure it exactly on a table where unseen actions keep their starting values.
  - Compute the CQL penalty and its gradient, and show that it sinks the values of actions the data never took.
  - Implement IQL's expectile loss, check its gradient, and show that tau = 0.5 gives the mean and tau near 1 the best in-data action.
prereqs:
  - 017-sarsa-and-q-learning
  - 018-maximization-bias
  - 042-imitation-learning
exercise: 044-offline-rl
---

# Offline reinforcement learning

Suppose you have a log of months of decisions that someone else made, with their outcomes, and you may never act yourself. With real patients, roads, or users, you cannot afford the mistakes that reinforcement learning usually learns from. Can you learn a better policy from the log alone?

*Offline reinforcement learning* does exactly this, with no further contact with the environment. It sounds like Q-learning on a replay buffer that never gets new data, but off-policy methods quietly depend on fresh data to correct their mistakes. Take the fresh data away, and they confidently pick actions that nobody ever tried. This chapter makes that failure exact on a table, where Q-learning on a cautious demonstrator's data decides that stepping off the cliff is the best move. Then you build two fixes: conservative Q-learning (CQL) and implicit Q-learning (IQL).

## What you already know

- Q-learning moves $Q(s, a)$ toward $r + \gamma \max_{a'} Q(s', a')$, using whatever transitions it has.
- A max over noisy estimates is biased upward, because it picks the estimates that happen to be too high (see [Maximization bias and double learning](/lessons/018-maximization-bias)).
- Behavior cloning copies the demonstrator's actions by supervised learning (see [Imitation learning](/lessons/042-imitation-learning)).
- The cliff world pays $-1$ per step and $-100$ for stepping into the cliff, which sends you back to the start.

## Learn from a log you cannot extend

The *dataset* $\mathcal{D}$ is a list of transitions $(s, a, r, s', d)$, where $d$ is 1 when $s'$ is terminal. Some *behavior policy* $\pi_\beta$ produced them. A pair $(s, a)$ is *in the data*, or *in support*, if it appears at least once.

The exercise's behavior policy is a cautious demonstrator on the cliff world with $\gamma = 0.9$. It heads for the goal, takes a random allowed action 40% of the time, and never steps toward the cliff. From 100 episodes you get 2,370 transitions. They cover 37 states, and 17 state-action pairs in those states never appear. The demonstrator's own value at the start is $-8.984$.

## Watch Q-learning trust what it never saw

Run tabular Q-learning over this dataset until it stops changing. Each iteration sets every covered pair to the mean of its targets:

$$
Q(s, a) \leftarrow \operatorname{mean}_{(s, a, r, s', d) \in \mathcal{D}} \Big[ r + \gamma (1 - d) \max_{a'} Q(s', a') \Big]
$$

Read it aloud: for each pair the data covers, average the one-step targets of all its transitions. The max in each target ranges over *every* action at the next state, including ones the data never took. Those keep their starting value of 0, because no data ever updates them:

```python
y = r + gamma * (1 - done) * Q[s2].max(axis=1)     # (N,)
Q = np.where(seen, mean_target_per_pair(y), Q)     # unseen pairs stay put
```

In a world where every step costs $-1$, a value of 0 is wildly optimistic, and the max finds it. Fujimoto and colleagues (2019) called this *extrapolation error*: values for actions outside the data are guesses, and the max systematically selects the guesses that are too high. Online, the agent would try that action and correct itself. Offline, nothing corrects it.

### Work it by hand: a three-cell corridor

Take the corridor `S.G` with $\gamma = 0.9$. Each move costs $-1$, and $G$ is terminal. The demonstrator always moves right, so the data holds only right.

- In the middle cell, right enters the goal: $Q(1, \text{right}) = -1$.
- In the start cell, right lands in the middle cell, whose best estimate is an unseen 0: $Q(0, \text{right}) = -1 + 0.9 \times 0 = -1$.
- Every other action keeps $Q = 0$.

The true values are $Q^*(0, \text{right}) = -1.9$ and $Q^*(0, \text{up}) = -2.71$, as in Figure 1. Offline Q-learning thinks up is worth 0, an overestimate of 2.71. Its greedy policy picks up, which bumps the wall forever, and is worth $-1 / (1 - 0.9) = -10$.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Q values at the start cell of the corridor: offline Q-learning gives 0 to unseen actions up, down and left while the truth is about -2.7">
<defs><marker id="c044-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<line x1="70.0" y1="70.0" x2="560.0" y2="70.0" class="stroke-grid"/>
<text x="62.0" y="74.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="123.3" x2="560.0" y2="123.3" class="stroke-grid"/>
<text x="62.0" y="127.3" class="fig-small" text-anchor="end">-1</text>
<line x1="70.0" y1="176.7" x2="560.0" y2="176.7" class="stroke-grid"/>
<text x="62.0" y="180.7" class="fig-small" text-anchor="end">-2</text>
<line x1="70.0" y1="230.0" x2="560.0" y2="230.0" class="stroke-grid"/>
<text x="62.0" y="234.0" class="fig-small" text-anchor="end">-3</text>
<rect x="70.0" y="70.0" width="26.0" height="2.0" class="fill-bad"/>
<text x="83.0" y="64.0" class="fig-small" text-anchor="middle">0</text>
<rect x="104.0" y="70.0" width="26.0" height="144.5" class="fill-a"/>
<text x="117.0" y="230.5" class="fig-small" text-anchor="middle">-2.71</text>
<text x="102.0" y="264.0" class="fig-small" text-anchor="middle">up (never taken)</text>
<rect x="185.0" y="70.0" width="26.0" height="53.3" class="fill-bad"/>
<text x="198.0" y="139.3" class="fig-small" text-anchor="middle">-1</text>
<rect x="219.0" y="70.0" width="26.0" height="101.3" class="fill-a"/>
<text x="232.0" y="187.3" class="fig-small" text-anchor="middle">-1.9</text>
<text x="217.0" y="264.0" class="fig-small" text-anchor="middle">right (in the data)</text>
<rect x="300.0" y="70.0" width="26.0" height="2.0" class="fill-bad"/>
<text x="313.0" y="64.0" class="fig-small" text-anchor="middle">0</text>
<rect x="334.0" y="70.0" width="26.0" height="144.5" class="fill-a"/>
<text x="347.0" y="230.5" class="fig-small" text-anchor="middle">-2.71</text>
<text x="332.0" y="264.0" class="fig-small" text-anchor="middle">down (never taken)</text>
<rect x="415.0" y="70.0" width="26.0" height="2.0" class="fill-bad"/>
<text x="428.0" y="64.0" class="fig-small" text-anchor="middle">0</text>
<rect x="449.0" y="70.0" width="26.0" height="144.5" class="fill-a"/>
<text x="462.0" y="230.5" class="fig-small" text-anchor="middle">-2.71</text>
<text x="447.0" y="264.0" class="fig-small" text-anchor="middle">left (never taken)</text>
<rect x="80.0" y="34.0" width="12.0" height="12.0" class="fill-bad"/>
<text x="98.0" y="45.0" class="fig-small" text-anchor="start">offline Q-learning</text>
<rect x="260.0" y="34.0" width="12.0" height="12.0" class="fill-a"/>
<text x="278.0" y="45.0" class="fig-small" text-anchor="start">true Q*</text>
</svg>
Figure 1. The start cell's values. Offline Q-learning leaves the three unseen actions at their starting value of 0, which beats every true value, so the greedy policy chooses one of them and bumps the wall forever.
```

On the cliff, the same thing happens at scale. Stepping right from the start into the cliff is truly worth about $-106.7$, but it looks like 0. Every unseen action in a visited state is overestimated by more than 7, and the greedy policy steps into the cliff on every move, worth exactly $-1000$.

> [!KEY]
> Offline, the values of actions the data never took are guesses that nothing can correct. A max over actions picks the most optimistic guess, and the policy follows it off the cliff.

BCQ (Fujimoto and colleagues, 2019) and TD3+BC (Fujimoto and Gu, 2021) keep the policy close to the data's actions. CQL makes unseen actions look bad, and IQL never evaluates an unseen action at all. The rest of this chapter builds the last two.

## Push unseen values down: CQL

Conservative Q-learning (Kumar and colleagues, 2020) adds a penalty to the usual squared TD error. In each state of the data, it pushes down a soft maximum over all actions and pushes up the action the data took:

$$
\mathcal{R}(Q) = \frac{1}{N}\sum_{i=1}^{N} \Big[ \log \sum_{a} e^{Q(s_i, a)} - Q(s_i, a_i) \Big]
$$

Read it aloud: for each transition, take the log-sum-exp of its state's Q values, which is close to the largest one, and subtract the Q of the action actually taken. The penalty is small when the data's action already has the highest value. CQL minimizes $\alpha \mathcal{R}(Q)$ plus the squared TD error, where $\alpha$ sets how conservative to be.

The gradient of a log-sum-exp is a softmax. With respect to row $s$ of the table, each transition from $s$ contributes $(\operatorname{softmax}(Q(s, \cdot)) - e_{a_i}) / N$, where $e_{a_i}$ is the indicator vector of the data's action:

```python
lse = top + np.log(np.exp(Q[s] - top[:, None]).sum(axis=1))  # top = row max
penalty = np.mean(lse - Q[s, a])
np.add.at(grad, s, softmax(Q[s]) / N)
np.add.at(grad, (s, a), -1.0 / N)
```

For $Q(s, \cdot) = (1, 2, 3)$ with the data's action 0, the penalty is $\log(e + e^2 + e^3) - 1 = 3.4076 - 1 = 2.4076$. The gradient is the softmax $(0.090, 0.245, 0.665)$ minus $(1, 0, 0)$, so the step lowers actions 1 and 2 and raises action 0.

For an action the data never took, the only force on its value is the penalty's push, which is $\alpha$ times its softmax weight, always positive. So an unseen value sinks, more and more slowly as its softmax weight fades. After 1,000 gradient steps with $\alpha = 0.5$ on the cliff, every unseen value sits below $-7$. The greedy policy uses only data actions and walks the 13-step safe path, worth $-7.458$. That beats the demonstrator's $-8.984$, because the greedy choice drops the demonstrator's random detours.

> [!WARNING]
> CQL does not make every covered value exact. In the cliff table, it values the move up from the start at $-6.62$ while the truth is $-7.46$. Kumar and colleagues prove a weaker guarantee: with $\alpha$ large enough, the value averaged over the learned policy's actions is a lower bound on that policy's true value.

## Never ask about unseen actions: IQL

CQL still takes a max over all actions and then fights the consequences. Implicit Q-learning (Kostrikov, Nair, and Levine, 2022) removes the max from the backup. It learns a separate state value $V(s)$ and bootstraps from that: $Q(s, a) \leftarrow r + \gamma (1 - d) V(s')$.

The trick is how $V$ is fit. If $V(s)$ were the plain mean of $Q(s, a)$ over the data's actions, you would get the behavior policy's value, as SARSA does. If it were the max over the data's actions, you would get the best policy the data supports. IQL interpolates with an *expectile*, fit by this loss:

$$
L_2^\tau(u) = \big|\tau - \mathbb{1}(u < 0)\big|\, u^2, \qquad u = Q(s, a) - V(s)
$$

Read it aloud: a squared error whose weight depends on the sign. When $Q$ is above $V$, the error gets weight $\tau$. When $Q$ is below $V$, it gets $1 - \tau$. With $\tau = 0.5$ both sides weigh the same, and the minimizing $V$ is the mean. With $\tau$ near 1, falling short of a high $Q$ costs far more than overshooting a low one, so $V$ climbs toward the largest $Q$ among the data's actions. The gradient with respect to $u$ is $2 |\tau - \mathbb{1}(u<0)| u$:

```python
weight = np.where(u < 0, 1 - tau, tau)
loss = np.mean(weight * u**2)
grad = 2 * weight * u / u.size
```

At $\tau = 0.7$, the loss is $L(1) = 0.7$ and $L(-1) = 0.3$. Suppose a state has two data actions with $Q$ values 1 and 3, equally often. The expectile $v$ balances the weighted errors, $\tau (3 - v) = (1 - \tau)(v - 1)$, which gives $v = 1 + 2\tau$. At $\tau = 0.5$ that is $v = 2$, the mean. At $\tau = 0.7$ it is $v = 2.4$, and at $\tau = 0.9$ it is $v = 2.8$, closing in on the max of 3 (Figure 2). In code, weight each value by $\tau$ or $1 - \tau$ according to its side of the current guess, take the weighted mean, and repeat.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="Expectile of the values 1 and 3 against tau: it equals 1 + 2 tau, from the mean 2 at tau 0.5 toward the max 3">
<defs><marker id="c044-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<line x1="80.0" y1="194.4" x2="540.0" y2="194.4" class="stroke-grid"/>
<text x="72.0" y="198.4" class="fig-small" text-anchor="end">2.0</text>
<line x1="80.0" y1="128.8" x2="540.0" y2="128.8" class="stroke-grid"/>
<text x="72.0" y="132.8" class="fig-small" text-anchor="end">2.5</text>
<line x1="80.0" y1="63.1" x2="540.0" y2="63.1" class="stroke-grid"/>
<text x="72.0" y="67.1" class="fig-small" text-anchor="end">3.0</text>
<line x1="80.0" y1="260.0" x2="80.0" y2="265.0" class="stroke-muted"/>
<text x="80.0" y="280.0" class="fig-small" text-anchor="middle">0.5</text>
<line x1="172.0" y1="260.0" x2="172.0" y2="265.0" class="stroke-muted"/>
<text x="172.0" y="280.0" class="fig-small" text-anchor="middle">0.6</text>
<line x1="264.0" y1="260.0" x2="264.0" y2="265.0" class="stroke-muted"/>
<text x="264.0" y="280.0" class="fig-small" text-anchor="middle">0.7</text>
<line x1="356.0" y1="260.0" x2="356.0" y2="265.0" class="stroke-muted"/>
<text x="356.0" y="280.0" class="fig-small" text-anchor="middle">0.8</text>
<line x1="448.0" y1="260.0" x2="448.0" y2="265.0" class="stroke-muted"/>
<text x="448.0" y="280.0" class="fig-small" text-anchor="middle">0.9</text>
<line x1="540.0" y1="260.0" x2="540.0" y2="265.0" class="stroke-muted"/>
<text x="540.0" y="280.0" class="fig-small" text-anchor="middle">1.0</text>
<text x="310.0" y="302.0" class="fig-small" text-anchor="middle">tau</text>
<line x1="80.0" y1="63.1" x2="540.0" y2="63.1" class="stroke-c dashed"/>
<text x="540.0" y="55.1" class="fig-small" text-anchor="end">max of the data's Q values: 3</text>
<polyline points="80.0,194.4 84.6,193.1 89.2,191.8 93.8,190.4 98.4,189.1 103.0,187.8 107.6,186.5 112.2,185.2 116.8,183.9 121.4,182.6 126.0,181.2 130.6,179.9 135.2,178.6 139.8,177.3 144.4,176.0 149.0,174.7 153.6,173.4 158.2,172.1 162.8,170.8 167.4,169.4 172.0,168.1 176.6,166.8 181.2,165.5 185.8,164.2 190.4,162.9 195.0,161.6 199.6,160.3 204.2,158.9 208.8,157.6 213.4,156.3 218.0,155.0 222.6,153.7 227.2,152.4 231.8,151.1 236.4,149.8 241.0,148.4 245.6,147.1 250.2,145.8 254.8,144.5 259.4,143.2 264.0,141.9 268.6,140.6 273.2,139.2 277.8,137.9 282.4,136.6 287.0,135.3 291.6,134.0 296.2,132.7 300.8,131.4 305.4,130.1 310.0,128.8 314.6,127.4 319.2,126.1 323.8,124.8 328.4,123.5 333.0,122.2 337.6,120.9 342.2,119.6 346.8,118.2 351.4,116.9 356.0,115.6 360.6,114.3 365.2,113.0 369.8,111.7 374.4,110.4 379.0,109.1 383.6,107.8 388.2,106.4 392.8,105.1 397.4,103.8 402.0,102.5 406.6,101.2 411.2,99.9 415.8,98.6 420.4,97.2 425.0,95.9 429.6,94.6 434.2,93.3 438.8,92.0 443.4,90.7 448.0,89.4 452.6,88.1 457.2,86.7 461.8,85.4 466.4,84.1 471.0,82.8 475.6,81.5 480.2,80.2 484.8,78.9 489.4,77.6 494.0,76.2 498.6,74.9 503.2,73.6 507.8,72.3 512.4,71.0 517.0,69.7 521.6,68.4 526.2,67.1 530.8,65.8 535.4,64.4" class="stroke-b" fill="none"/>
<circle cx="80.0" cy="194.4" r="5" class="fill-b"/>
<text x="90.0" y="210.4" class="fig-small" text-anchor="start">tau 0.5: 2.0</text>
<circle cx="264.0" cy="141.9" r="5" class="fill-b"/>
<text x="274.0" y="157.9" class="fig-small" text-anchor="start">tau 0.7: 2.4</text>
<circle cx="448.0" cy="89.4" r="5" class="fill-b"/>
<text x="458.0" y="105.4" class="fig-small" text-anchor="start">tau 0.9: 2.8</text>
</svg>
Figure 2. The expectile of two equally frequent values, 1 and 3, rises steadily from the mean toward the max as tau grows, so tau is a dial between SARSA-like evaluation and the best in-data choice.
```

In the exercise, IQL at $\tau = 0.5$ reproduces the exact value of the data's empirical behavior policy, to $10^{-8}$. As $\tau$ rises, $V(\text{start})$ climbs from $-9.02$ to $-8.28$, $-7.70$, and $-7.48$ at $\tau = 0.99$ (Figure 3). It approaches the best in-data value, $-7.458$, and never passes it, because no unseen action ever enters a target.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="IQL value at the start cell against tau: -9.02, -8.28, -7.70 and -7.48, approaching but never passing the best in-data value of -7.458">
<defs><marker id="c044-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<line x1="80.0" y1="221.8" x2="540.0" y2="221.8" class="stroke-grid"/>
<text x="72.0" y="225.8" class="fig-small" text-anchor="end">-9</text>
<line x1="80.0" y1="174.1" x2="540.0" y2="174.1" class="stroke-grid"/>
<text x="72.0" y="178.1" class="fig-small" text-anchor="end">-8.5</text>
<line x1="80.0" y1="126.4" x2="540.0" y2="126.4" class="stroke-grid"/>
<text x="72.0" y="130.4" class="fig-small" text-anchor="end">-8</text>
<line x1="80.0" y1="78.6" x2="540.0" y2="78.6" class="stroke-grid"/>
<text x="72.0" y="82.6" class="fig-small" text-anchor="end">-7.5</text>
<line x1="118.3" y1="260.0" x2="118.3" y2="265.0" class="stroke-muted"/>
<text x="118.3" y="280.0" class="fig-small" text-anchor="middle">0.5</text>
<line x1="271.7" y1="260.0" x2="271.7" y2="265.0" class="stroke-muted"/>
<text x="271.7" y="280.0" class="fig-small" text-anchor="middle">0.7</text>
<line x1="425.0" y1="260.0" x2="425.0" y2="265.0" class="stroke-muted"/>
<text x="425.0" y="280.0" class="fig-small" text-anchor="middle">0.9</text>
<line x1="494.0" y1="260.0" x2="494.0" y2="265.0" class="stroke-muted"/>
<text x="494.0" y="280.0" class="fig-small" text-anchor="middle">0.99</text>
<text x="310.0" y="302.0" class="fig-small" text-anchor="middle">tau</text>
<line x1="80.0" y1="74.6" x2="540.0" y2="74.6" class="stroke-c dashed"/>
<text x="540.0" y="66.6" class="fig-small" text-anchor="end">best value the data supports: -7.458</text>
<line x1="80.0" y1="220.3" x2="540.0" y2="220.3" class="stroke-muted dashed"/>
<text x="540.0" y="236.3" class="fig-small" text-anchor="end">true demonstrator: -8.984</text>
<polyline points="118.3,223.5 271.7,153.1 425.0,97.5 494.0,76.8" class="stroke-b" fill="none"/>
<circle cx="118.3" cy="223.5" r="5" class="fill-b"/>
<text x="126.3" y="241.5" class="fig-small" text-anchor="start">-9.02</text>
<circle cx="271.7" cy="153.1" r="5" class="fill-b"/>
<text x="279.7" y="171.1" class="fig-small" text-anchor="start">-8.28</text>
<circle cx="425.0" cy="97.5" r="5" class="fill-b"/>
<text x="433.0" y="115.5" class="fig-small" text-anchor="start">-7.70</text>
<circle cx="494.0" cy="76.8" r="5" class="fill-b"/>
<text x="502.0" y="94.8" class="fig-small" text-anchor="start">-7.48</text>
</svg>
Figure 3. IQL's V at the start of the cliff world. At tau 0.5 it matches the value of the data's own action frequencies, and as tau rises it approaches the best in-data value from below without crossing it.
```

> [!KEY]
> IQL bootstraps from a learned $V(s')$, never from a max over actions. The expectile $\tau$ slides $V$ from the behavior policy's value toward the best value the data supports. Both fixes answer the opening question the same way: you can improve on the log, but only with actions the log contains.

> [!RECAP]
> - Offline Q-learning leaves unseen pairs at their starting values, and the max selects them.
> - The CQL gradient row is the softmax minus the data action's indicator.
> - The expectile loss weights $u^2$ by $\tau$ above zero and $1 - \tau$ below.
> - IQL targets use $V(s')$, with no max over actions.

## Build it step by step

The checks test the functions in this order. The helpers `make_dataset`, `action_counts`, and `cautious_behavior` are written for you.

1. **`offline_q_iteration(data, n_states, n_actions, gamma, iters)`.** Compute all targets at once, sum them per pair with `np.add.at`, and divide by the counts. Keep unseen pairs at 0 with `np.where(seen, ...)`.
2. **`cql_penalty(Q, s, a)`.** Subtract each row's max before `np.exp`. The check includes a row of `[1000, 999]`.
3. **`cql_penalty_grad(Q, s, a)`.** Use `np.add.at`, not `grad[s] += ...`, because fancy-index assignment drops repeated states.
4. **`cql_q_iteration(data, ..., alpha, iters, lr)`.** Recompute the targets each step, add `alpha * cql_penalty_grad` to the TD gradient, divide each visited row by `counts[s].sum() / N`, and step.
5. **`expectile_loss` and `expectile_loss_grad`.** One `np.where` gives the weight. Divide the gradient by `u.size`, because the loss is a mean.
6. **`expectile(values, weights, tau, iters)`.** Work along the last axis so one call handles every state, and start at the weighted mean.
7. **`iql_values(data, ..., tau, iters)`.** Alternate `V = expectile(Q, counts, tau)` and the mean target `r + gamma * (1 - done) * V[s2]`. A max anywhere here defeats the method.

## Where this breaks

**Coverage limits everything.** No method can learn that an action is good if no data shows it. If the demonstrator never goes near the goal, offline RL will not find the goal.

**Tuning needs evaluation that you do not have.** Picking $\alpha$ or $\tau$ by running the policy in the real environment defeats the purpose, and offline policy evaluation is its own hard problem.

## Further reading

- [Levine, Kumar, Tucker, and Fu (2020), Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643). A survey of why offline RL is hard.
- [Fujimoto, Meger, and Precup (2019), Off-Policy Deep Reinforcement Learning without Exploration](https://arxiv.org/abs/1812.02900). Names extrapolation error and introduces BCQ.
- [Kumar, Zhou, Tucker, and Levine (2020), Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779). The CQL penalty and its lower-bound theorems.
- [Kostrikov, Nair, and Levine (2022), Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169). The expectile value loss, and advantage-weighted policy extraction.

## The exercise

You build all three methods on the cliff world, from a cautious demonstrator's 2,370 transitions. Write `offline_q_iteration` to see the failure, then `cql_penalty`, `cql_penalty_grad`, and `cql_q_iteration`. For IQL, write `expectile_loss`, `expectile_loss_grad`, `expectile`, and `iql_values`.

The checks confirm the corridor's exact numbers and the cliff table's fixed point, with a greedy value near $-1000$. They test the CQL penalty against $2.4076$ and its gradient by finite differences, and require CQL's greedy policy to use only data actions and reach the best in-data value. The expectile checks cover the loss at $\tau = 0.7$, its gradient, and the mean at $\tau = 0.5$. IQL at $\tau = 0.5$ must equal policy evaluation of the empirical behavior policy, and as $\tau$ rises its values must climb within 0.1 of the best in-data value without passing it.
