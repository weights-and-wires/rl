---
title: Cooperative multi-agent RL
slug: 064-cooperative-multi-agent-rl
part: "Part 8 — Rewards, evaluation, and many agents"
summary: Teach a team to share one reward with VDN and QMIX, and see why independent learners pick the safe action.
minutes: 40
gpu: false
objectives:
  - "State the Dec-POMDP setting and centralized training with decentralized execution (CTDE)."
  - Explain why independent learners on a shared reward face a moving target and fall for the safe action on the climbing game.
  - Write VDN's sum and check that the greedy joint action equals each agent's own greedy action (IGM).
  - Build QMIX's monotonic mixer with absolute-value weights and verify dQ_tot/dQ_i >= 0 by finite differences.
  - Say which payoff tables an additive or monotonic mixer can represent, and which it cannot.
prereqs:
  - 026-deep-q-networks
  - 061-multi-agent-and-self-play
exercise: 064-cooperative-multi-agent-rl
---

# Cooperative multi-agent RL

Two robots carry a table through a doorway. Each sees only its own side of the room, and each decides its own move. They share one reward: the table arrives in one piece, or it doesn't. Neither robot can tell, from that single number, how much of the credit is its own.

You could give each robot its own Q-learner and let them train side by side. It sometimes works, and it fails in a particular way. While robot A learns, robot B changes how it moves, so the world A is learning from keeps shifting. Worse, when the team can score big only if both robots commit, each robot's average experience says that committing is a bad idea.

This chapter fixes both problems with one trick: train with a team-wide view, then act with local views. It builds two methods that do this, VDN and QMIX, and checks what each can and cannot represent. [Many agents and self-play](/lessons/061-multi-agent-and-self-play) covers the competitive case, where agents' rewards oppose each other. Here every agent receives the same reward.

## What you already know

- A Q-learner keeps $Q(a)$ for each action and moves it toward the observed reward, as in [Deep Q-networks](/lessons/026-deep-q-networks).
- A matrix game gives each pair of actions a payoff, as in the games of [Many agents and self-play](/lessons/061-multi-agent-and-self-play).
- A greedy action is the one with the largest Q value.
- PPO clips a policy-gradient step, as in [Proximal policy optimization](/lessons/036-proximal-policy-optimization).

## The setting: one team reward, many local views

A *decentralized partially observable Markov decision process* (Dec-POMDP; Oliehoek and Amato, 2016) describes the team. There are $N$ agents. At each step the world is in a hidden state $s$, agent $i$ sees only its own observation $o_i$, and every agent picks an action $a_i$ at once. The world moves on, and the whole team receives a single reward $r$ that depends on the state and the joint action $\mathbf{a} = (a_1, \dots, a_N)$.

At run time each agent must choose from its own observation alone, because the robots can't talk to a central controller. That is *decentralized execution*. At training time you can usually do better: the simulator can show the learner every agent's observation, the true state, and the joint action. *Centralized training with decentralized execution* (CTDE) uses that extra information to shape learning, and throws it away when the agents act.

## Why independent learners fall for the safe action

The simplest team gives each agent its own Q table and trains it on the team reward as if the partner were part of the world. These are *independent learners*. Claus and Boutilier (1998) showed how they fail on a small payoff table, the climbing game. Both agents choose a row or column from $\{0, 1, 2\}$ and receive the same payoff:

| agent 1 \ agent 2 | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 11 | -30 | 0 |
| 1 | -30 | 7 | 6 |
| 2 | 0 | 0 | 5 |

The best joint action is $(0, 0)$, worth 11. Miss it by one agent and the team loses 30. The joint action $(2, 2)$ pays only 5, but it never punishes a partner who strays.

An independent learner can't see its partner, so its Q value for an action is the average payoff over whatever its partner plays. Suppose agent 2 plays all three actions equally. Agent 1's action 0 then averages $(11 - 30 + 0) / 3 = -6.33$, action 1 averages $(-30 + 7 + 6) / 3 = -5.67$, and action 2 averages $(0 + 0 + 5) / 3 = 1.67$. Action 2 looks best.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="The climbing game payoff grid, and each row's mean payoff against a uniform partner: -6.3, -5.7, and 1.7">
<text x="195.0" y="38" class="fig-small fig-bold" text-anchor="middle">agent 2 plays</text>
<text x="34.0" y="165.0" class="fig-small fig-bold" text-anchor="middle" transform="rotate(-90 34.0 165.0)">agent 1 plays</text>
<text x="125.0" y="54.0" class="fig-small" text-anchor="middle">0</text>
<text x="80.0" y="99.0" class="fig-small" text-anchor="end">0</text>
<text x="195.0" y="54.0" class="fig-small" text-anchor="middle">1</text>
<text x="80.0" y="169.0" class="fig-small" text-anchor="end">1</text>
<text x="265.0" y="54.0" class="fig-small" text-anchor="middle">2</text>
<text x="80.0" y="239.0" class="fig-small" text-anchor="end">2</text>
<rect x="90.0" y="60.0" width="70.0" height="70.0" class="fill-soft-c stroke-muted"/>
<text x="125.0" y="100.0" class="fig-bold" text-anchor="middle">11</text>
<rect x="160.0" y="60.0" width="70.0" height="70.0" class="fill-sunk stroke-muted"/>
<text x="195.0" y="100.0" class="fig-bold" text-anchor="middle">-30</text>
<rect x="230.0" y="60.0" width="70.0" height="70.0" class="fill-sunk stroke-muted"/>
<text x="265.0" y="100.0" class="fig-bold" text-anchor="middle">0</text>
<rect x="90.0" y="130.0" width="70.0" height="70.0" class="fill-sunk stroke-muted"/>
<text x="125.0" y="170.0" class="fig-bold" text-anchor="middle">-30</text>
<rect x="160.0" y="130.0" width="70.0" height="70.0" class="fill-sunk stroke-muted"/>
<text x="195.0" y="170.0" class="fig-bold" text-anchor="middle">7</text>
<rect x="230.0" y="130.0" width="70.0" height="70.0" class="fill-sunk stroke-muted"/>
<text x="265.0" y="170.0" class="fig-bold" text-anchor="middle">6</text>
<rect x="90.0" y="200.0" width="70.0" height="70.0" class="fill-sunk stroke-muted"/>
<text x="125.0" y="240.0" class="fig-bold" text-anchor="middle">0</text>
<rect x="160.0" y="200.0" width="70.0" height="70.0" class="fill-sunk stroke-muted"/>
<text x="195.0" y="240.0" class="fig-bold" text-anchor="middle">0</text>
<rect x="230.0" y="200.0" width="70.0" height="70.0" class="fill-soft-b stroke-muted"/>
<text x="265.0" y="240.0" class="fig-bold" text-anchor="middle">5</text>
<text x="440.0" y="38" class="fig-small fig-bold" text-anchor="middle">mean payoff, uniform partner</text>
<line x1="400.0" y1="60.0" x2="400.0" y2="270.0" class="stroke-muted"/>
<rect x="324.0" y="80.0" width="76.0" height="30" class="fill-a"/>
<text x="406.0" y="100.0" class="fig-small" text-anchor="start">-6.33</text>
<rect x="332.0" y="150.0" width="68.0" height="30" class="fill-a"/>
<text x="406.0" y="170.0" class="fig-small" text-anchor="start">-5.67</text>
<rect x="400.0" y="220.0" width="20.0" height="30" class="fill-b"/>
<text x="426.0" y="240.0" class="fig-small" text-anchor="start">1.67</text>
<text x="300" y="290" class="fig-small" text-anchor="middle">green: the best joint action. orange: the safe one.</text>
</svg>
Figure 1. The climbing game and each row's mean payoff against a uniform partner. Notice that the best joint action (green) belongs to the row with the lowest mean, and the safe action (orange) to the row with the highest.
```

Learning makes this worse. As soon as both agents lean toward action 2, the partner's exploration costs little there and a lot everywhere else. Each agent drifts further toward action 2, and from the inside it looks like a good decision. This is *relative overgeneralization*: an action's value is pulled down by the partner's mistakes, so it is judged relative to a partner who doesn't coordinate.

Figure 2 shows one run. Agent 1's value for action 0 climbs early, while both agents still pick action 0 and score 11. It then collapses as the partner's exploration lands on $-30$ cells, and action 2 ends highest.

```figure
<svg viewBox="0 0 600 280" role="img" aria-label="One independent learner's three action values over 3000 steps. The value of action 2 ends highest.">
<line x1="70.0" y1="220.0" x2="570.0" y2="220.0" class="stroke-grid"/>
<text x="62.0" y="224.0" class="fig-small" text-anchor="end">-8</text>
<line x1="70.0" y1="172.5" x2="570.0" y2="172.5" class="stroke-grid"/>
<text x="62.0" y="176.5" class="fig-small" text-anchor="end">-4</text>
<line x1="70.0" y1="125.0" x2="570.0" y2="125.0" class="stroke-muted"/>
<text x="62.0" y="129.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="77.5" x2="570.0" y2="77.5" class="stroke-grid"/>
<text x="62.0" y="81.5" class="fig-small" text-anchor="end">4</text>
<line x1="70.0" y1="30.0" x2="570.0" y2="30.0" class="stroke-grid"/>
<text x="62.0" y="34.0" class="fig-small" text-anchor="end">8</text>
<text x="70.0" y="240.0" class="fig-small" text-anchor="middle">0</text>
<text x="236.7" y="240.0" class="fig-small" text-anchor="middle">1000</text>
<text x="403.3" y="240.0" class="fig-small" text-anchor="middle">2000</text>
<text x="570.0" y="240.0" class="fig-small" text-anchor="middle">3000</text>
<polyline points="70.0,125.0 78.3,68.7 86.7,30.0 95.0,68.4 103.3,49.1 111.7,102.0 120.0,109.2 128.3,120.0 136.7,121.3 145.0,120.5 153.3,132.0 161.7,129.4 170.0,139.2 178.3,135.4 186.7,116.4 195.0,113.7 203.3,119.2 211.7,127.9 220.0,127.1 228.3,120.5 236.7,181.5 245.0,174.7 253.3,189.5 261.7,150.1 270.0,133.9 278.3,149.2 286.7,136.8 295.0,149.9 303.3,135.6 311.7,127.5 320.0,143.2 328.3,146.2 336.7,134.2 345.0,144.4 353.3,156.2 361.7,145.1 370.0,143.1 378.3,145.5 386.7,141.7 395.0,130.7 403.3,144.7 411.7,133.7 420.0,137.4 428.3,166.8 436.7,159.0 445.0,156.9 453.3,167.4 461.7,177.3 470.0,181.3 478.3,152.2 486.7,146.9 495.0,167.9 503.3,163.7 511.7,160.6 520.0,167.1 528.3,144.8 536.7,145.7 545.0,141.0 553.3,119.8 561.7,115.0 570.0,144.3" class="stroke-a" fill="none"/>
<text x="574.0" y="148.3" class="fig-small" text-anchor="start">action 0</text>
<polyline points="70.0,125.0 78.3,134.6 86.7,211.0 95.0,220.0 103.3,220.0 111.7,220.0 120.0,220.0 128.3,220.0 136.7,197.5 145.0,183.3 153.3,139.2 161.7,121.9 170.0,113.4 178.3,135.1 186.7,112.4 195.0,107.2 203.3,118.1 211.7,105.6 220.0,103.6 228.3,107.4 236.7,89.7 245.0,77.8 253.3,107.4 261.7,119.1 270.0,117.0 278.3,102.1 286.7,97.7 295.0,125.5 303.3,116.6 311.7,124.8 320.0,105.4 328.3,100.4 336.7,84.7 345.0,85.4 353.3,132.0 361.7,120.3 370.0,110.8 378.3,95.2 386.7,85.2 395.0,101.5 403.3,92.6 411.7,81.7 420.0,85.9 428.3,101.6 436.7,104.6 445.0,91.1 453.3,104.5 461.7,108.9 470.0,104.6 478.3,100.9 486.7,89.2 495.0,91.3 503.3,97.5 511.7,99.7 520.0,109.6 528.3,146.7 536.7,162.7 545.0,143.9 553.3,109.6 561.7,93.3 570.0,82.7" class="stroke-d" fill="none"/>
<text x="574.0" y="86.7" class="fig-small" text-anchor="start">action 1</text>
<polyline points="70.0,125.0 78.3,125.0 86.7,122.9 95.0,119.1 103.3,118.7 111.7,119.3 120.0,112.5 128.3,102.9 136.7,82.6 145.0,82.5 153.3,82.0 161.7,76.1 170.0,83.2 178.3,86.9 186.7,80.3 195.0,82.4 203.3,85.6 211.7,84.1 220.0,77.8 228.3,78.8 236.7,79.2 245.0,73.1 253.3,84.3 261.7,84.0 270.0,78.6 278.3,94.3 286.7,82.0 295.0,76.8 303.3,75.0 311.7,79.1 320.0,78.6 328.3,81.0 336.7,80.9 345.0,78.5 353.3,82.7 361.7,84.3 370.0,81.7 378.3,76.4 386.7,81.5 395.0,90.1 403.3,83.4 411.7,82.1 420.0,78.8 428.3,88.8 436.7,80.2 445.0,76.3 453.3,79.8 461.7,84.3 470.0,79.7 478.3,85.0 486.7,85.1 495.0,82.3 503.3,81.2 511.7,87.5 520.0,78.5 528.3,81.2 536.7,81.7 545.0,82.4 553.3,74.7 561.7,83.6 570.0,77.9" class="stroke-b" fill="none"/>
<text x="574.0" y="81.9" class="fig-small" text-anchor="start">action 2</text>
<text x="320.0" y="270" class="fig-small" text-anchor="middle">step</text>
</svg>
Figure 2. One independent learner's action values over 3000 steps, with epsilon = 0.4 and step size 0.05. Notice that action 0 rises to about 8 and then falls below zero, while action 2 ends highest at 3.97.
```

The partner's behavior also changes as it learns, so each agent chases a target that moves. That is the *non-stationarity* problem: the convergence guarantees of single-agent Q-learning assume the world holds still.

A joint learner, with one Q value per joint action, doesn't have this problem. Its estimate for $(0, 0)$ is 11 no matter what anyone explores. It also needs $|A|^N$ values and a single controller that sees everything, so it can't run on separate robots. The question is how to keep each agent's local choice and still learn from the team's joint reward.

## VDN: add the agents' utilities

*Value decomposition networks* (VDN; Sunehag et al., 2017) give each agent a *utility* $Q_i(o_i, a_i)$ that depends only on its own view, and define the team's value as their sum:

$$
Q_{\text{tot}}(\mathbf{o}, \mathbf{a}) = \sum_{i=1}^{N} Q_i(o_i, a_i)
$$

You train the $Q_i$ end to end: compute $Q_{\text{tot}}$ for the joint action that was taken, and move it toward the team's TD target, as DQN does. The gradient flows back into every $Q_i$ at once. Credit assignment comes from the sum, because raising the team's value means raising some agent's utility.

The reason to sum is *individual-global-max* (IGM): the joint action that maximizes $Q_{\text{tot}}$ is the tuple of each agent's own greedy action,

$$
\arg\max_{\mathbf{a}} Q_{\text{tot}}(\mathbf{o}, \mathbf{a}) = \big( \arg\max_{a_1} Q_1(o_1, a_1), \dots, \arg\max_{a_N} Q_N(o_N, a_N) \big)
$$

For a sum, this is true because each term depends on one agent's action only, so you can maximize the terms separately. At run time, each agent picks its own argmax with no communication, and the team still plays the best joint action of $Q_{\text{tot}}$. Training uses the whole team's reward. Execution uses one local network per agent.

In numpy, with `qs` of shape `(n_agents, n_actions)`:

```python
q_tot = qs[np.arange(len(qs)), actions].sum()
greedy = qs.argmax(axis=1)
```

### A worked example

Take `qs = [[1, 3, 0], [2, -1, 4]]`. Agent 1's best action is 1 with utility 3, and agent 2's is 2 with utility 4. The joint action $(1, 2)$ has $Q_{\text{tot}} = 3 + 4 = 7$. Every other joint action has less: $(0, 2)$ gives $1 + 4 = 5$, and $(1, 0)$ gives $3 + 2 = 5$.

If you average instead of summing, $Q_{\text{tot}}$ becomes 3.5. The greedy joint action is the same, because dividing by $N$ changes no ordering. The values are wrong, though, so the TD target the network trains toward would be wrong too. Check the value, not only the argmax.

### What a sum cannot say

A sum has no interaction term. Take a table that pays 1 only when both agents play action 1, and 0 otherwise. Call it AND. The best additive fit, by least squares, is $\text{row mean} + \text{column mean} - \text{grand mean}$, which gives the cells $-0.25, 0.25, 0.25, 0.75$. Each cell is off by $0.25$, so the mean squared error is $1/16 = 0.0625$. The fit still picks the right joint action here, but its values are wrong, and wrong values make bad bootstrap targets.

On the climbing game, the same fit over all nine cells has a mean squared error of 175.09 and picks $(2, 2)$ again. A sum can represent a table only if payoffs add up across agents, and the climbing game is nowhere near that.

## QMIX: any monotonic mixer keeps IGM

IGM doesn't need a sum. It needs only that $Q_{\text{tot}}$ never decreases when one agent's utility goes up. *QMIX* (Rashid et al., 2018) replaces the sum with a small network $f$ that mixes the utilities, and forces

$$
\frac{\partial Q_{\text{tot}}}{\partial Q_i} \ge 0 \quad \text{for every agent } i
$$

Here is why this keeps IGM. Each agent's greedy action gives its largest utility $Q_i^{\max}$. Any other joint action has utilities no larger than those, so by monotonicity $Q_{\text{tot}}$ at the greedy actions is at least as large as at any other joint action. The greedy tuple is therefore the joint argmax, and a sum is the special case with all slopes equal to 1.

A network is monotonic in its inputs if all the weights between the inputs and the output are nonnegative, and the activation never decreases. QMIX gets nonnegative weights with *hypernetworks*: networks that read the global state $s$ and output the mixer's weights. The mixer, in numpy, is:

```python
w1 = np.abs(state @ w1_A + w1_c).reshape(B, N, H)   # first layer weights, abs
b1 = state @ b1_A + b1_c                             # first layer bias, any sign
hidden = elu(np.einsum("bn,bnh->bh", qs, w1) + b1)
w2 = np.abs(state @ w2_A + w2_c)                     # second layer weights, abs
q_tot = (hidden * w2).sum(axis=1) + (state @ b2_A + b2_c)[:, 0]
```

The state enters through the hypernetworks only. That is a CTDE trade: the mixer is used in training, so it can read the state. The agents never see it, and at run time the mixer is dropped. The biases can be negative, because adding a constant never changes a slope.

### Check one number

Take $N = 2$, $H = 1$, state $s = 1$, utilities $(1, 2)$, raw first-layer weights $(-0.5, 1.0)$, bias $0.25$, raw second-layer weight $-2$, and final bias $0.5$. The abs turns the weights into $(0.5, 1.0)$ and $2$. The hidden value is $0.5 \cdot 1 + 1 \cdot 2 + 0.25 = 2.75$, which ELU leaves alone because it is positive. So $Q_{\text{tot}} = 2 \cdot 2.75 + 0.5 = 6$. Skip the abs on the first layer alone and the hidden value is $-0.5 + 2 + 0.25 = 1.75$, so $Q_{\text{tot}} = 2 \cdot 1.75 + 0.5 = 4$, and raising agent 1's utility lowers the team's value, with slope $2 \cdot (-0.5) = -1$.

### What a monotonic mixer cannot say

Monotonic still doesn't mean any table. Take XOR, which pays 1 when the agents match. Row 0 pays more for column 0, so a monotonic mixer needs $Q_2(0) > Q_2(1)$. Row 1 pays more for column 1, so it needs $Q_2(1) > Q_2(0)$. One utility table can't satisfy both, so no monotonic mixer fits XOR. The climbing game has the same flaw. Row 0 ranks the columns $0, 2, 1$ and row 1 ranks them $1, 2, 0$, and one ordering can't match both.

Table 1 summarizes what each method does on the two $2 \times 2$ tables in the exercise.

| Payoff table | VDN's mean squared error | Trained QMIX's mean squared error |
|---|---|---|
| AND (monotone) | 0.0625 | below 0.0005 in five seeds |
| XOR (not monotone) | 0.25 | 0.25 |

Table 1. Fits to two team payoff tables. QMIX fits AND, where a sum can't. On XOR, both settle on a constant 0.5 in the exercise's runs.

> [!KEY]
> VDN and QMIX both keep IGM, so each agent can act greedily on its own utility. VDN adds the utilities and QMIX mixes them with nonnegative weights. Monotonic mixing represents more tables than a sum, but still not a table whose best response depends on who else is playing.

## MAPPO: a centralized critic for policy gradients

The same CTDE idea works for policy gradients. *Multi-agent PPO* (MAPPO; Yu et al., 2022) gives each agent a policy $\pi_i(a_i \mid o_i)$ that acts on local observations, and trains a single *centralized critic* $V(s)$ that sees the global state. Each agent's advantage comes from the critic, and each policy takes the clipped PPO step from [Proximal policy optimization](/lessons/036-proximal-policy-optimization). At run time you drop the critic. Yu et al. report that this simple setup, with the PPO implementation details of [PPO in practice](/lessons/037-ppo-in-practice), is competitive with off-policy methods such as QMIX on several cooperative benchmarks.

## Build it step by step

The exercise builds the pieces in the order the checks test them.

1. `vdn_q_tot(qs, actions)` takes utilities of shape `(n_agents, n_actions)` and one action per agent, and returns the sum of the chosen utilities. The usual mistake is `.mean()`, which keeps IGM but halves the value for two agents.
2. `greedy_actions(qs)` returns each agent's own argmax, with `argmax(axis=1)`.
3. `best_joint_action(qs)` searches every joint action with `itertools.product` and returns the one with the largest `vdn_q_tot`. The checks confirm it equals `greedy_actions` on random utilities.
4. `qmix_mixer(qs, state, hyper)` follows the code above. The mistake to avoid is skipping `np.abs` on `w1` or `w2`, or adding it to the biases.
5. The monotonicity check nudges each `qs[:, i]` by plus and minus $10^{-5}$ and requires a nonnegative slope on random states and random hypernetworks, with weights of both signs.
6. `vdn_fit(table)` returns the closed-form additive fit and its mean squared error. The checks verify with finite differences that its gradient is zero at the fit.
7. `fit_qmix(table, seed)` trains free utilities and a small mixer with Adam in torch. It must fit AND and cannot fit XOR.
8. `expected_individual_values(table, partner)` is `table @ partner`. It reproduces the means of Figure 1.
9. `independent_learners` and `joint_learner` run the climbing game over five seeds. The independent pair must settle on $(2, 2)$. The joint learner must find $(0, 0)$.

## Where this breaks

- **Monotonic is a restriction.** A task whose best action for one agent flips with its partner's action, like XOR, needs a richer factorization. Later methods relax the monotonic constraint, at the cost of extra machinery. The climbing game is the same trap, so QMIX does not remove relative overgeneralization in general.
- **The global state is a training-time luxury.** CTDE assumes a simulator that reports the state. Real fleets may not have one.
- **A shared reward blurs credit.** With many agents, one team reward carries little information about any single agent. The sum or mixer helps, but the signal gets noisier as the team grows.
- **Parameter sharing needs identity.** Agents that share one network need an agent ID in their input, or they can't specialize.

## Further reading

- [Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296) (Sunehag et al., 2017). The sum of per-agent utilities, with end-to-end training.
- [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485) (Rashid et al., 2018). The hypernetwork mixer and its monotonicity constraint.
- [The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games](https://arxiv.org/abs/2103.01955) (Yu et al., 2022). MAPPO and the implementation details that matter.
- [The Dynamics of Reinforcement Learning in Cooperative Multiagent Systems](https://www.cs.toronto.edu/~cebly/Papers/multirl2-abs.html) (Claus and Boutilier, 1998). Independent learners against joint-action learners, and the climbing game.
- [A Concise Introduction to Decentralized POMDPs](https://doi.org/10.1007/978-3-319-28929-8) (Oliehoek and Amato, 2016). The Dec-POMDP model and its solution methods.

The Berkeley CS285 and Stanford CS224R schedules, as fetched, list no lecture on multi-agent RL, so this chapter rests on the papers.

## The exercise

You write `vdn_q_tot`, `greedy_actions`, `best_joint_action`, `qmix_mixer`, `vdn_fit`, `fit_qmix`, `expected_individual_values`, `independent_learners`, and `joint_learner`. The checks confirm the sum with exact values, IGM on random utilities, the mixer against a hand-computed number, monotonicity by finite differences, the exact least-squares fit, QMIX's fit of AND, the shared failure on XOR, and the safe-action trap of independent learners across five seeds.

The `__main__` block prints the greedy joint action, the VDN and QMIX errors on AND and XOR, and what independent and joint learners choose on three seeds.
