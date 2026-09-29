---
title: Reward design and shaping
slug: 30-reward-design
part: "Part 7 — Rewards, evaluation, and many agents"
summary: Add a potential-based shaping reward that provably keeps the optimal policy, and see how an ordinary bonus breaks the task.
minutes: 50
gpu: false
objectives:
  - "State the shaping theorem: F = gamma * Phi(s') - Phi(s) leaves the optimal policy unchanged and shifts Q* by Phi(s)."
  - Verify the theorem exactly on random MDPs, and explain the telescoping sum behind it.
  - Break the guarantee two ways, with a bonus that isn't potential-based and with a nonzero potential at terminal states.
  - Show that shaping equals a different starting Q table, so it changes how fast you learn, not what you learn.
prereqs:
  - 05-dynamic-programming
  - 07-temporal-difference-learning
exercise: 30-reward-design
---

# Reward design and shaping

A reward is the only way you tell an agent what you want. Get it wrong, and
the agent does what you said instead of what you meant. In 1998, Randløv and
Alstrøm rewarded a simulated bicycle for moving toward its goal. The agent
learned to ride in small circles, collecting the reward on each approach. In
2016, OpenAI's boat-racing agent found it could score more by circling to hit
the same respawning targets than by finishing the race.

Both failures started with a good intention: the true reward was sparse, so
the designers added hints. A sparse reward, one that arrives only at the goal,
gives a learner almost nothing to go on. Hints help, but most hints quietly
change which behaviour is best.

This lesson builds the one kind of hint that provably can't change the best
behaviour, from Ng, Harada, and Russell (1999). You check the guarantee exactly,
break it two ways, and measure how much it speeds up learning.

## What you already know

- Value iteration computes the optimal action values $Q^*$ of a tabular MDP
  (lesson 5).
- A policy is optimal when, in every state, it only takes actions that
  maximise $Q^*(s, \cdot)$.
- The discounted return adds rewards weighted by $\gamma^t$.
- Tabular Q-learning moves $Q(s, a)$ a step $\alpha$ toward $r + \gamma
  \max_{a'} Q(s', a')$ (lesson 7).
- A terminal state ends the episode, and its value is zero by definition.

## Rewards are a specification

The reward is a contract. The agent maximises it, whatever it takes, so any
gap between the reward and your intent is a gap the agent can find.
Krakovna and colleagues (2020) keep a public list of dozens of such cases,
which they call *specification gaming*. In one, an agent rewarded for the
height of a block's bottom face flipped the block over instead of stacking it.

This is Goodhart's law applied to optimisation: when a measure becomes a
target, it stops being a good measure. Part 8 meets it again with learned
reward models, and lesson 49 measures it. This lesson handles the simplest
version, where you add a bonus to a reward you trust.

A *shaping reward* $F(s, a, s')$ is extra reward paid on each transition, on
top of the task's own reward $r$. The question is which shaping rewards leave
the task's optimal policy alone.

## Shape with a potential

Pick any number $\Phi(s)$ for each state, called a *potential*. Think of it as
height on a map: how promising a state looks. Pay the agent for climbing:

$$
F(s, a, s') = \gamma\, \Phi(s') - \Phi(s)
$$

Read it aloud: the bonus for a transition is the discounted potential of where
you land minus the potential of where you started. Moving to a more promising
state earns a bonus. Moving to a less promising one costs you. In numpy, over
the full `(S, A, S)` reward table:

```python
F = gamma * phi[None, None, :] - phi[:, None, None]     # (S, A, S)
R3_shaped = R3 + F
R_shaped = (P * R3_shaped).sum(axis=2)                   # expected reward, (S, A)
```

### Why the optimal policy can't move

Add up the bonuses along any trajectory $s_0, s_1, s_2, \dots$, each weighted
by $\gamma^t$ as the return weights them:

$$
\sum_{t=0}^{T-1} \gamma^t \big( \gamma \Phi(s_{t+1}) - \Phi(s_t) \big) = \gamma^T \Phi(s_T) - \Phi(s_0)
$$

Read it aloud: each term's positive half cancels the next term's negative half,
so only the two ends survive. If the episode ends in a terminal state with
$\Phi = 0$, or runs forever with $\gamma < 1$ and a bounded $\Phi$, the sum is exactly
$-\Phi(s_0)$. That number depends only on where you started, not on anything
you did. Every policy's return drops by the same amount, so the ranking of
policies doesn't change.

The same argument works one action at a time. Ng, Harada, and Russell proved
the following:

- **Sufficiency.** If $F$ is potential-based, then
  $Q^*_{\text{shaped}}(s, a) = Q^*(s, a) - \Phi(s)$. Every optimal policy of
  the shaped task is optimal in the original, and the reverse.
- **Necessity.** If $F$ is not potential-based, some choice of transitions and
  rewards makes it change the optimal policy.

Subtracting $\Phi(s)$ shifts a whole row of the Q table by one number. The
largest entry in each row stays the largest.

> [!KEY]
> Potential-based shaping pays $\gamma \Phi(s') - \Phi(s)$. Along any
> trajectory it telescopes to $-\Phi(s_0)$, so $Q^*$ shifts by $\Phi(s)$ in
> each state and the optimal actions never change.

### Worked example: a five-state chain

Take five states in a row, 0 to 4, with state 4 terminal. Action 0 moves left
(state 0 stays put), and action 1 moves right. Entering state 4 pays 1, and
$\gamma = 0.9$. Moving right is optimal everywhere, and the optimal values are
the following:

| State | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| $Q^*(s, \text{left})$ | 0.6561 | 0.6561 | 0.729 | 0.81 |
| $Q^*(s, \text{right})$ | 0.729 | 0.81 | 0.9 | 1.0 |

Shape with $\Phi = (0, 1, 2, 3, 0)$, zero at the terminal state. From state 1,
moving right lands in state 2, so it earns $0.9 \times 2 - 1 = 0.8$. Moving
left lands in state 0 and earns $0 - 1 = -1$. From state 3, moving right
enters the goal: $1 + 0.9 \times 0 - 3 = -2$. That's a large penalty on the
winning move! Yet value iteration on the shaped chain gives
$Q^*_{\text{shaped}}(3, \cdot) = (-2.19, -2.0)$, which is $Q^*(3, \cdot) - 3$.
Right is still best.

> [!WARNING]
> Writing $F = \Phi(s') - \Phi(s)$ without the $\gamma$ isn't potential-based
> when $\gamma < 1$. It equals the correct term plus $(1 - \gamma)\Phi(s')$, a
> reward for sitting in high-potential states, and the agent can profit from
> loitering there.

## Keep the potential zero where episodes end

The telescoping sum leaves $\gamma^T \Phi(s_T)$ behind. When $s_T$ is terminal
and $\Phi(s_T) \ne 0$, that leftover depends on when and where the episode
ends. So the agent can change its shaped return by choosing its ending.

Consider the row `HS...G` with $\gamma = 0.9$. Moves are free, the goal pays 1,
and the hole ends the episode with nothing. Walking right to the goal is
optimal, worth $0.9^3 = 0.729$ from the start. Now shape with the constant
potential $\Phi = 10$ everywhere, terminals included. A constant looks
harmless. But every step now pays $(0.9 - 1) \times 10 = -1$. Walking to the
goal costs four such steps: $-(1 + 0.9 + 0.81 + 0.729) + 0.729 = -2.71$.
Jumping into the hole costs one: $-1$. The shaped agent jumps.

Set $\Phi = 0$ at both terminals and keep 10 elsewhere. Now entering a terminal
pays $0 - 10$, the sum telescopes to $-10$ for every policy, and walking right
is optimal again.

## A bonus that isn't a potential

Go back to the chain and add an ordinary bonus: $+0.3$ for every transition
into state 1. It seems to reward progress, since state 1 is one step closer to
the goal. But the agent can collect it again and again. Stepping $0 \to 1 \to 0
\to 1$ earns 0.3 every two steps, worth $0.3 / (1 - 0.81) = 1.579$ from state 0.
That beats the 0.729 the goal is worth.

Value iteration confirms it: $Q^*_{\text{bonus}}(0, \cdot) = (1.421, 1.579)$.
In states 2 and 3, left becomes the only optimal action, and in state 1 left
and right tie. The optimal policy of the bonus task never reaches the goal, so
its value in the original task is exactly 0.

A potential-based bonus can't do this. Around any loop, the gains from
climbing are paid back when you descend, so there is nothing to farm.

```python
R3 = full_rewards(mdp).copy()
R3[:, :, state] += bonus          # every (s, a) that lands in `state`
```

## Shaping changes speed, not the answer

If shaping can't change the optimal policy, why bother? Because it changes the
path a learner takes to get there. Wiewiora (2003) proved a clean version of
this for Q-learning.

Q-learning on shaped rewards, started from a table $Q_0$, behaves exactly like
Q-learning on the original rewards started from $Q_0 + \Phi(s)$. Feed both the
same transitions, and after every update the tables differ by exactly
$\Phi(s)$ in each row. So they make the same greedy choices at every step. To
see why, substitute $Q_{\text{shaped}} = Q - \Phi(s)$ into the shaped update:

$$
r + \gamma \Phi(s') - \Phi(s) + \gamma \max_{a'} \big(Q(s', a') - \Phi(s')\big) - \big(Q(s, a) - \Phi(s)\big)
= r + \gamma \max_{a'} Q(s', a') - Q(s, a)
$$

Read it aloud: every potential term cancels, and what's left is the ordinary TD
error. At a terminal $s'$ there is no max, so the $\gamma \Phi(s')$ term only
cancels when $\Phi(s') = 0$. This is the terminal rule again.

So a potential is prior knowledge about values. The ideal potential is
$\Phi = V^*$, which makes the shaped values equal the advantage
$A^*(s, a) = Q^*(s, a) - V^*(s)$. That is zero for optimal actions and negative
for the rest. A rough guess at $V^*$, such as a distance heuristic, often
helps a lot.

On the exercise's 5-by-6 maze, use $\Phi(s) = -(\text{Manhattan distance to the
goal})$. With $\epsilon$-greedy Q-learning from a zero table, the first
episode averages 105 steps over five seeds without shaping and 25 with it. Over 30 episodes,
seed 0 takes 947 steps without shaping and 308 with it. Both learners end on the
same optimal path.

> [!KEY]
> Shaping with a potential is the same as starting Q-learning from a table
> shifted by $\Phi$. A good potential is a good initial guess at the values.

> [!RECAP]
> - Shape with $F = \gamma \Phi(s') - \Phi(s)$, computed over the whole
>   `(S, A, S)` reward table.
> - The shaped task's optimal values are $Q^* - \Phi(s)$, row by row, with the
>   same optimal actions.
> - $\Phi$ must be zero at terminal states, or the agent can profit from how
>   it ends an episode.
> - A bonus for entering a state isn't potential-based, and a loop can farm
>   it.
> - Shaped Q-learning from $Q_0$ equals unshaped Q-learning from $Q_0 + \Phi$,
>   minus $\Phi$.

## Build it step by step

The checks test the functions in this order. The helpers `full_rewards`,
`with_rewards`, and `distance_potential` are written for you.

1. **`optimal_actions(q, tol)`.** Returns a boolean `(S, A)` mask. Compare
   with the row max minus `tol`, so floating-point ties count as ties. Common
   mistake: comparing `np.argmax` results, which hides a tie and makes two
   equally good policies look different.
2. **`shaped_mdp(mdp, phi)`.** Build `F` with broadcasting, add it to
   `full_rewards(mdp)`, and return `with_rewards(mdp, ...)`. Common mistake:
   putting $\gamma$ on the wrong potential. The check on 10 random MDPs catches
   it.
3. **`bonus_mdp(mdp, state, bonus)`.** Copy the reward table, add `bonus` to
   the slice of transitions into `state`, and rebuild. Common mistake: adding
   the bonus to `R[state, :]`, which pays for leaving the state instead.
4. **`q_learning_updates(Q0, transitions, alpha, gamma, phi)`.** Loop over the
   transitions in order. Shape the reward, then build the target, skipping
   the bootstrap when `done` is 1. Common mistake: updating `Q0` in place,
   which corrupts the second run in the Wiewiora check.
5. **`run_q_learning(mdp, episodes, alpha, epsilon, seed, phi)`.** The same
   update inside an $\epsilon$-greedy loop over `mdp.reset` and `mdp.step`.
   Break greedy ties at random. The check expects shaping to cut the total
   steps over 30 episodes by more than half.

## Cost and scaling

Shaping adds one subtraction per transition, so it costs nothing at run time.
The following table compares the work each check needs.

| Operation | Time | Memory |
|---|---|---|
| Shape the reward table | $O(S^2 A)$ | One extra `(S, A, S)` table |
| Value iteration per sweep | $O(S^2 A)$ | $O(SA)$ |
| One shaped Q-learning update | $O(A)$ | None extra |

The real cost is designing $\Phi$. With function approximation, $\Phi$ can be
a learned value estimate, a distance heuristic, or a model's guess of progress.
The guarantee needs only that $\Phi$ depends on the state alone and is zero
where episodes end.

## Where this breaks

**A potential that depends on the action.** $\Phi(s, a)$ isn't covered by the
theorem. Wiewiora, Cottrell, and Elkan (2003) showed which corrections restore
it, under the name "look-ahead advice".

**A potential that changes during training.** If you learn $\Phi$ as you go,
the fixed-point argument no longer applies directly. Devlin and Kudenko (2012)
extended the guarantee to time-varying potentials of the form
$\gamma \Phi(s', t+1) - \Phi(s, t)$.

**A bad potential slows learning.** The theorem protects the answer, not the
speed. A potential that points the wrong way acts like a pessimistic starting
table, and the learner takes longer to find the goal.

**Truncated episodes.** If you treat a time limit as a real ending, the
leftover $\gamma^T \Phi(s_T)$ at the cut changes the return. Bootstrap from the
cut state instead, as for any truncation, and shaping stays consistent.

**The underlying reward is still a proxy.** Shaping preserves whatever the
original reward rewards. If that reward is gameable, a perfectly shaped version
is exactly as gameable.

## Further reading

- [Ng, Harada, and Russell (1999), Policy Invariance under Reward
  Transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).
  The shaping theorem, with both the sufficiency and necessity halves.
- [Wiewiora (2003), Potential-Based Shaping and Q-Value Initialization are
  Equivalent](https://arxiv.org/abs/1106.5267). The proof that shaping equals a
  shifted starting table for Q-learning.
- [Krakovna and colleagues (2020), Specification Gaming: The Flip Side of AI
  Ingenuity](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/).
  A catalogue of agents that satisfied the letter of their reward.
- [Amodei and colleagues (2016), Concrete Problems in AI
  Safety](https://arxiv.org/abs/1606.06565). Frames reward hacking and
  negative side effects as research problems.
- [Skalse, Howe, Krasheninnikov, and Krueger (2022), Defining and
  Characterizing Reward Hacking](https://arxiv.org/abs/2209.13085). A formal
  definition of when a proxy reward can be hacked, and why almost every proxy
  can be.
- [Setlur and colleagues (2024), Rewarding Progress: Scaling Automated Process
  Verifiers for LLM Reasoning](https://arxiv.org/abs/2410.08146). Rewards each
  reasoning step by the change in a prover's chance of success, a
  potential-based idea at language-model scale.

Open questions a researcher might work on:

- How can you learn a potential from data, or from a language model's
  judgement, that speeds learning without ever encoding a wrong preference?
- Which reward specifications resist gaming by optimisers much stronger than
  the people who wrote them?
- Does the equivalence between shaping and initialisation survive in deep
  networks, where a starting table isn't something you can set directly?

## The exercise

You shape rewards on a five-state chain, on random MDPs, and on a small maze.
Write `optimal_actions`, `shaped_mdp`, and `bonus_mdp`, then
`q_learning_updates` for Q-learning over a fixed list of transitions and
`run_q_learning` for the full $\epsilon$-greedy loop.

The checks confirm the chain's shaped rewards by hand, and
$Q^*_{\text{shaped}} = Q^* - \Phi(s)$ with unchanged optimal actions on 10
random MDPs and on the maze. A constant potential of 10 that includes the
terminals must send the agent into the hole, with the exact values
$(-1.9, -2.71, -1.9, -1.0)$ at the start. The $+0.3$ bonus must give
$Q^*(0, \cdot) = (1.421, 1.579)$ and a policy that never reaches the goal.
Shaped Q-learning must equal unshaped Q-learning from $Q_0 + \Phi$, minus
$\Phi$, over 400 random transitions. On the maze, the distance potential must
cut the steps by more than half and still find the optimal path.
