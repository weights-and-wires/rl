---
title: Reward design and shaping
slug: 055-reward-design-and-shaping
part: "Part 8 — Rewards, evaluation, and many agents"
summary: Add a potential-based shaping reward that provably keeps the optimal policy, and see how an ordinary bonus breaks the task.
minutes: 35
gpu: false
objectives:
  - "State the shaping theorem: F = gamma * Phi(s') - Phi(s) leaves the optimal policy unchanged and shifts Q* by Phi(s)."
  - Explain the telescoping sum behind the theorem and verify it exactly on random MDPs.
  - Show how an ordinary bonus lets the agent farm it and abandon the goal.
  - Say why the potential must be zero at terminal states.
prereqs:
  - 012-value-iteration
  - 017-sarsa-and-q-learning
exercise: 055-reward-design-and-shaping
---

# Reward design and shaping

In 1998, Randløv and Alstrøm rewarded a simulated bicycle for moving toward its goal. The agent learned to ride in small circles, collecting the reward on each approach. Both the designers and the agent did what they were told. The true reward, reaching the goal, was sparse, so the designers added a hint, and the hint quietly changed which behavior was best.

A sparse reward, one that arrives only at the goal, gives a learner almost nothing to go on, so hints are tempting. Can you add a hint that provably cannot change what the best behavior is?

This chapter shows how an ordinary hint goes wrong on a five-state chain, and then builds the one kind of hint that cannot, from Ng, Harada, and Russell (1999). You check the guarantee exactly, and break it by one careless choice at the end of an episode.

## What you already know

- Value iteration computes the optimal action values $Q^*$ of a tabular MDP.
- A policy is optimal when, in every state, it only takes actions that maximize $Q^*(s, \cdot)$.
- The discounted return adds rewards weighted by $\gamma^t$.
- A terminal state ends the episode, and its value is zero by definition.

## Rewards are a specification

The reward is a contract. The agent maximizes it, whatever it takes, so any gap between the reward and your intent is a gap the agent can find. Krakovna and colleagues (2020) keep a public list of dozens of such cases, which they call *specification gaming*. In one, an agent rewarded for the height of a block's bottom face flipped the block over instead of stacking it. A *shaping reward* $F(s, a, s')$ is extra reward paid on each transition, on top of the task's own reward $r$. The question is which shaping rewards leave the task's optimal policy alone.

## The obvious hint, and how it breaks

Take five states in a row, 0 to 4, with state 4 terminal. Action 0 moves left (state 0 stays put), and action 1 moves right. Entering state 4 pays 1, and $\gamma = 0.9$. Moving right is optimal everywhere, and the optimal values are:

| State | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| $Q^*(s, \text{left})$ | 0.6561 | 0.6561 | 0.729 | 0.81 |
| $Q^*(s, \text{right})$ | 0.729 | 0.81 | 0.9 | 1.0 |

The obvious hint rewards progress: pay $+0.3$ for every transition into state 1, which is one step closer to the goal. But the agent can collect it again and again. Stepping $0 \to 1 \to 0 \to 1$ earns 0.3 every two steps, worth $0.3 / (1 - 0.81) = 1.579$ from state 0. Walking to the goal then earns $0.3 + 0.9^3 = 1.029$, so the loop wins (Figure 1).

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Value from state 0 under the +0.3 bonus: walking to the goal is worth 1.029, looping between states 0 and 1 is worth 1.579">
<defs><marker id="c045-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<text x="20.0" y="80.0" class="fig-small" text-anchor="start">original task: walk to the goal</text>
<rect x="320.0" y="64.0" width="101.2" height="26.0" class="fill-a"/>
<text x="429.2" y="82.0" class="fig-small" text-anchor="start">0.729</text>
<text x="20.0" y="140.0" class="fig-small" text-anchor="start">bonus task: walk to the goal</text>
<rect x="320.0" y="124.0" width="142.9" height="26.0" class="fill-a"/>
<text x="470.9" y="142.0" class="fig-small" text-anchor="start">1.029</text>
<text x="20.0" y="200.0" class="fig-small" text-anchor="start">bonus task: loop 0, 1, 0, 1</text>
<rect x="320.0" y="184.0" width="219.3" height="26.0" class="fill-bad"/>
<text x="547.3" y="202.0" class="fig-small" text-anchor="start">1.579</text>
<line x1="320.0" y1="50.0" x2="320.0" y2="240.0" class="stroke-muted"/>
<text x="300.0" y="270.0" class="fig-small" text-anchor="middle">discounted value at state 0 (gamma = 0.9)</text>
</svg>
Figure 1. The bonus pays 0.3 on every entry to state 1, so the loop beats the walk to the goal, and the optimal policy of the bonus task never finishes the real task.
```

Value iteration confirms it: $Q^*_{\text{bonus}}(0, \cdot) = (1.421, 1.579)$. In states 2 and 3, left becomes the only optimal action, and in state 1 left and right tie. The optimal policy of the bonus task never reaches the goal, so its value in the original task is exactly 0. In code, this bonus is one slice of the reward table:

```python
R3 = full_rewards(mdp).copy()
R3[:, :, state] += bonus          # every (s, a) that lands in `state`
```

> [!KEY]
> A bonus for *being* somewhere can be collected again and again. The agent optimizes the bonus instead of the task.

## Pay for climbing, and take it back on the way down

The loop paid because the agent was rewarded for arriving but never charged for leaving. Fix that by giving every state a number $\Phi(s)$, called a *potential*. Think of it as height on a map: how promising a state looks. Pay the agent for climbing and charge it for descending:

$$
F(s, a, s') = \gamma\, \Phi(s') - \Phi(s)
$$

Read it aloud: the bonus for a transition is the discounted potential of where you land minus the potential of where you started. Moving to a more promising state earns a bonus, and moving to a less promising one costs you. In numpy, over the full `(S, A, S)` reward table:

```python
F = gamma * phi[None, None, :] - phi[:, None, None]     # (S, A, S)
R3_shaped = R3 + F
```

Why can this not move the optimal policy? Add up the bonuses along any trajectory $s_0, s_1, s_2, \dots$, each weighted by $\gamma^t$ as the return weights them:

$$
\sum_{t=0}^{T-1} \gamma^t \big( \gamma \Phi(s_{t+1}) - \Phi(s_t) \big) = \gamma^T \Phi(s_T) - \Phi(s_0)
$$

Read it aloud: each term's positive half cancels the next term's negative half, so only the two ends survive. If the episode ends in a terminal state with $\Phi = 0$, the sum is exactly $-\Phi(s_0)$. That number depends only on where you started, not on anything you did. Every policy's return drops by the same amount, so the ranking of policies does not change. Around any loop the gains from climbing are paid back on the way down, so there is nothing to farm.

The same argument works one action at a time. Ng, Harada, and Russell proved that if $F$ is potential-based, then $Q^*_{\text{shaped}}(s, a) = Q^*(s, a) - \Phi(s)$, and every optimal policy of the shaped task is optimal in the original, and the reverse. They also proved the converse: if $F$ is not potential-based, some choice of transitions and rewards makes it change the optimal policy. Subtracting $\Phi(s)$ shifts a whole row of the Q table by one number, so the largest entry in each row stays the largest.

### Work it by hand

Shape the chain with $\Phi = (0, 1, 2, 3, 0)$, zero at the terminal state. Moving right from states 0, 1, 2, and 3 pays the bonuses $0.9 \times 1 - 0 = 0.9$, then $0.8$, then $0.7$, and then $0 - 3 = -3$ on the last step into the goal (Figure 2). From state 1, the three bonuses on the way to the goal total $0.8 + 0.9 \times 0.7 + 0.81 \times (-3) = -1$, which is exactly $-\Phi(1)$. From state 3, moving right enters the goal: $1 + 0.9 \times 0 - 3 = -2$, a large penalty on the winning move. Yet value iteration on the shaped chain gives $Q^*_{\text{shaped}}(3, \cdot) = (-2.19, -2.0)$, which is $Q^*(3, \cdot) - 3$, and right is still best. Figure 3 shows the whole table: every row of $Q^*$ drops by its own $\Phi(s)$, and right stays on top in every row.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Five-state chain with potential 0, 1, 2, 3, 0 drawn as bar heights and the shaping bonus on each rightward move: 0.9, 0.8, 0.7 and minus 3">
<defs><marker id="c045-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<circle cx="80.0" cy="280.0" r="18" class="fill-soft-a"/>
<text x="80.0" y="285.0" class="fig-bold" text-anchor="middle">0</text>
<text x="80.0" y="242.0" class="fig-small" text-anchor="middle">Phi = 0</text>
<rect x="172.0" y="210.0" width="36.0" height="40.0" class="fill-soft-b"/>
<circle cx="190.0" cy="280.0" r="18" class="fill-soft-a"/>
<text x="190.0" y="285.0" class="fig-bold" text-anchor="middle">1</text>
<text x="190.0" y="202.0" class="fig-small" text-anchor="middle">Phi = 1</text>
<rect x="282.0" y="170.0" width="36.0" height="80.0" class="fill-soft-b"/>
<circle cx="300.0" cy="280.0" r="18" class="fill-soft-a"/>
<text x="300.0" y="285.0" class="fig-bold" text-anchor="middle">2</text>
<text x="300.0" y="162.0" class="fig-small" text-anchor="middle">Phi = 2</text>
<rect x="392.0" y="130.0" width="36.0" height="120.0" class="fill-soft-b"/>
<circle cx="410.0" cy="280.0" r="18" class="fill-soft-a"/>
<text x="410.0" y="285.0" class="fig-bold" text-anchor="middle">3</text>
<text x="410.0" y="122.0" class="fig-small" text-anchor="middle">Phi = 3</text>
<circle cx="520.0" cy="280.0" r="18" class="fill-soft-a"/>
<text x="520.0" y="285.0" class="fig-bold" text-anchor="middle">4</text>
<text x="520.0" y="242.0" class="fig-small" text-anchor="middle">Phi = 0</text>
<line x1="50.0" y1="250.0" x2="570.0" y2="250.0" class="stroke-muted"/>
<line x1="102.0" y1="280.0" x2="168.0" y2="280.0" class="stroke-b" marker-end="url(#c045-arrow)"/>
<text x="135.0" y="318.0" class="fig-small" text-anchor="middle">F = 0.9</text>
<line x1="212.0" y1="280.0" x2="278.0" y2="280.0" class="stroke-b" marker-end="url(#c045-arrow)"/>
<text x="245.0" y="318.0" class="fig-small" text-anchor="middle">F = 0.8</text>
<line x1="322.0" y1="280.0" x2="388.0" y2="280.0" class="stroke-b" marker-end="url(#c045-arrow)"/>
<text x="355.0" y="318.0" class="fig-small" text-anchor="middle">F = 0.7</text>
<line x1="432.0" y1="280.0" x2="498.0" y2="280.0" class="stroke-b" marker-end="url(#c045-arrow)"/>
<text x="465.0" y="318.0" class="fig-small" text-anchor="middle">F = -3.0</text>
<text x="520.0" y="318.0" class="fig-small" text-anchor="middle">goal</text>
<text x="300.0" y="36.0" class="fig-bold" text-anchor="middle">From state 1 to the goal: 0.8 + 0.9 x 0.7 + 0.81 x (-3.0) = -1.0 = -Phi(1)</text>
<text x="300.0" y="58.0" class="fig-small" text-anchor="middle">The ends of the sum survive; everything in the middle cancels</text>
</svg>
Figure 2. Each rightward move pays gamma Phi(next) minus Phi(here). Climbing the bars earns bonuses, and the final step into the goal pays back everything at once, so the total along the path is minus the potential at the start.
```

```figure
<svg viewBox="0 0 600 345" role="img" aria-label="Q values for left and right in states 0 to 3 before and after shaping: shaping lowers each state's pair by the same amount, and right stays higher">
<defs><marker id="c045-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<text x="130.0" y="34.0" class="fig-bold" text-anchor="middle">Q* of the original chain</text>
<line x1="20.0" y1="127.9" x2="250.0" y2="127.9" class="stroke-muted"/>
<rect x="38.0" y="87.4" width="11.0" height="40.5" class="fill-a"/>
<rect x="52.0" y="82.9" width="11.0" height="45.0" class="fill-b"/>
<text x="52.0" y="286.0" class="fig-small" text-anchor="middle">state 0</text>
<rect x="93.0" y="87.4" width="11.0" height="40.5" class="fill-a"/>
<rect x="107.0" y="77.9" width="11.0" height="50.0" class="fill-b"/>
<text x="107.0" y="286.0" class="fig-small" text-anchor="middle">state 1</text>
<rect x="148.0" y="82.9" width="11.0" height="45.0" class="fill-a"/>
<rect x="162.0" y="72.4" width="11.0" height="55.6" class="fill-b"/>
<text x="162.0" y="286.0" class="fig-small" text-anchor="middle">state 2</text>
<rect x="203.0" y="77.9" width="11.0" height="50.0" class="fill-a"/>
<rect x="217.0" y="66.2" width="11.0" height="61.8" class="fill-b"/>
<text x="217.0" y="286.0" class="fig-small" text-anchor="middle">state 3</text>
<text x="430.0" y="34.0" class="fig-bold" text-anchor="middle">Q* of the shaped chain</text>
<line x1="320.0" y1="127.9" x2="550.0" y2="127.9" class="stroke-muted"/>
<rect x="338.0" y="87.4" width="11.0" height="40.5" class="fill-a"/>
<rect x="352.0" y="82.9" width="11.0" height="45.0" class="fill-b"/>
<text x="352.0" y="286.0" class="fig-small" text-anchor="middle">state 0</text>
<rect x="393.0" y="127.9" width="11.0" height="21.2" class="fill-a"/>
<rect x="407.0" y="127.9" width="11.0" height="11.7" class="fill-b"/>
<text x="407.0" y="286.0" class="fig-small" text-anchor="middle">state 1</text>
<rect x="448.0" y="127.9" width="11.0" height="78.5" class="fill-a"/>
<rect x="462.0" y="127.9" width="11.0" height="67.9" class="fill-b"/>
<text x="462.0" y="286.0" class="fig-small" text-anchor="middle">state 2</text>
<rect x="503.0" y="127.9" width="11.0" height="135.3" class="fill-a"/>
<rect x="517.0" y="127.9" width="11.0" height="123.5" class="fill-b"/>
<text x="517.0" y="286.0" class="fig-small" text-anchor="middle">state 3</text>
<rect x="250.0" y="302.0" width="11.0" height="11.0" class="fill-a"/>
<text x="266.0" y="312.0" class="fig-small" text-anchor="start">left</text>
<rect x="320.0" y="302.0" width="11.0" height="11.0" class="fill-b"/>
<text x="336.0" y="312.0" class="fig-small" text-anchor="start">right</text>
<text x="300.0" y="332.0" class="fig-small" text-anchor="middle">shaped values = original values minus Phi(s): 0, 1, 2, 3 in states 0 to 3</text>
</svg>
Figure 3. The orange right bar is the taller of each pair before and after shaping. The shaped bars sit lower by 0, 1, 2, and 3 in states 0 to 3, but the gap inside each pair is unchanged.
```

> [!WARNING]
> Writing $F = \Phi(s') - \Phi(s)$ without the $\gamma$ is not potential-based when $\gamma < 1$. It equals the correct term plus $(1 - \gamma)\Phi(s')$, a reward for sitting in high-potential states, and the agent can profit from loitering there.

## Keep the potential zero where episodes end

The telescoping sum leaves $\gamma^T \Phi(s_T)$ behind. When $s_T$ is terminal and $\Phi(s_T) \ne 0$, that leftover depends on when and where the episode ends, so the agent can change its shaped return by choosing its ending.

Consider the row `HS...G` with $\gamma = 0.9$. Moves are free, the goal pays 1, and the hole ends the episode with nothing. Walking right to the goal is optimal, worth $0.9^3 = 0.729$ from the start. Shape with the constant potential $\Phi = 10$ everywhere, terminals included. A constant looks harmless, but every step then pays $(0.9 - 1) \times 10 = -1$. Walking to the goal costs four such steps: $-(1 + 0.9 + 0.81 + 0.729) + 0.729 = -2.71$. Jumping into the hole costs one: $-1$. The shaped agent jumps. With $\Phi = 0$ at both terminals and 10 elsewhere, entering a terminal pays $0 - 10$, the sum telescopes to $-10$ for every policy, and walking right is optimal again.

> [!KEY]
> Potential-based shaping pays $\gamma \Phi(s') - \Phi(s)$. Along any trajectory it telescopes to $-\Phi(s_0)$, so $Q^*$ shifts by $\Phi(s)$ in each state and the optimal actions never change, provided $\Phi$ is zero at terminal states. That is the hint you can add without changing the answer.

> [!RECAP]
> - Shape with $F = \gamma \Phi(s') - \Phi(s)$, computed over the whole `(S, A, S)` reward table.
> - The shaped task's optimal values are $Q^* - \Phi(s)$, row by row, with the same optimal actions.
> - $\Phi$ must be zero at terminal states, or the agent can profit from how it ends an episode.
> - A bonus for entering a state is not potential-based, and a loop can farm it.

## Build it step by step

The checks test the functions in this order. The helpers `full_rewards` and `with_rewards` are written for you.

1. **`optimal_actions(q, tol)`.** Returns a boolean `(S, A)` mask. Compare with the row max minus `tol`, so floating-point ties count as ties. Comparing `np.argmax` results hides a tie and makes two equally good policies look different.
2. **`shaped_mdp(mdp, phi)`.** Build `F` with broadcasting, add it to `full_rewards(mdp)`, and return `with_rewards(mdp, ...)`. Put $\gamma$ on the potential of the next state. The check on 10 random MDPs catches the wrong placement.
3. **`bonus_mdp(mdp, state, bonus)`.** Copy the reward table, add `bonus` to the slice of transitions into `state`, and rebuild. Adding it to `R[state, :]` pays for leaving the state instead.

## Where this breaks

**A bad potential slows learning.** The theorem protects the answer, not the speed. Wiewiora (2003) showed that Q-learning on shaped rewards behaves exactly like Q-learning on the original rewards started from a table shifted by $\Phi$, so a good potential is a good initial guess at the values, and a potential that points the wrong way acts like a pessimistic start.

**A potential that depends on the action, or changes during training.** The theorem does not cover $\Phi(s, a)$ or a learned $\Phi$ that moves. Wiewiora, Cottrell, and Elkan (2003) and Devlin and Kudenko (2012) extend it to some of these cases.

**Truncated episodes.** If you treat a time limit as a real ending, the leftover $\gamma^T \Phi(s_T)$ at the cut changes the return. Bootstrap from the cut state instead, and shaping stays consistent.

**Shaping is not the only answer to a sparse reward.** When the task is to reach any of many goals, you can leave the reward alone and relabel the data instead, as [Goals and hindsight](/lessons/048-goal-conditioned-rl-and-her) shows.

**The underlying reward is still a proxy.** Shaping preserves whatever the original reward rewards. If that reward is gameable, a perfectly shaped version is exactly as gameable. The next chapter, [Preferences over trajectories](/lessons/056-preferences-over-trajectories), learns the reward from comparisons instead of writing it.

## Further reading

- [Ng, Harada, and Russell (1999), Policy Invariance under Reward Transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf). The shaping theorem, with both the sufficiency and necessity halves.
- [Wiewiora (2003), Potential-Based Shaping and Q-Value Initialization are Equivalent](https://arxiv.org/abs/1106.5267). Shaping equals a shifted starting table for Q-learning.
- [Krakovna and colleagues (2020), Specification Gaming: The Flip Side of AI Ingenuity](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/). A catalog of agents that satisfied the letter of their reward.
- [Amodei and colleagues (2016), Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565). Frames reward hacking as a research problem.

## The exercise

You shape rewards on a five-state chain and on random MDPs. Write `optimal_actions`, `shaped_mdp`, and `bonus_mdp`.

The checks confirm the chain's shaped rewards by hand, and $Q^*_{\text{shaped}} = Q^* - \Phi(s)$ with unchanged optimal actions on 10 random MDPs. A constant potential of 10 that includes the terminals must send the agent into the hole, with the exact values $(-1.9, -2.71, -1.9, -1.0)$ at the start. The $+0.3$ bonus must give $Q^*(0, \cdot) = (1.421, 1.579)$ and a policy that never reaches the goal.
