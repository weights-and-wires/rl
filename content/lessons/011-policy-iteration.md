---
title: Policy iteration
slug: 011-policy-iteration
part: "Part 2 — Planning with a known world"
summary: Improve a policy by acting greedily on its values, then alternate evaluating and improving until the policy stops changing.
minutes: 35
gpu: false
objectives:
  - Turn a policy's state values into a greedy policy by comparing action values.
  - Explain why acting greedily on a policy's values never makes any state worse.
  - Run policy iteration by hand on the two-state world, and in code on a random MDP.
  - Say why the loop stops, and why it stops at an optimal policy.
prereqs:
  - 009-the-bellman-equation
  - 010-policy-evaluation
exercise: 011-policy-iteration
---

# Policy iteration

Suppose you know exactly how a two-room world works. In each room you can stay or walk to the other room, and each choice pays a known reward. You pick a policy, "stay in whatever room you are in", and evaluate it. The values come out as 10 for room 0 and 20 for room 1. Is that policy any good?

The values alone don't say. They tell you what this policy earns, not whether a different policy would earn more. The obvious way to find out is to evaluate every policy and keep the best. That works for two rooms, but a world with 20 states and 4 actions has $4^{20}$, about a trillion, deterministic policies. You need a way to improve a policy directly, one informed step at a time.

This chapter builds that step, which asks in every state whether a different first action would do better, and then repeats it until nothing can improve. The method is called policy iteration.

## What you already know

- A policy $\pi(a \mid s)$ is an `(S, A)` array whose rows sum to one.
- The value $V^\pi(s)$ is the expected return from $s$ when you follow $\pi$, and you can get it exactly by solving $(I - \gamma P_\pi) V^\pi = r_\pi$.
- The action value is $Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a)\, V^\pi(s')$: take $a$ now, then follow $\pi$.
- This chapter switches from the four-cell hallway of the last three chapters to a smaller *two-room world*, so you can check every step by hand. It has two states, 0 and 1, and $\gamma = 0.9$. Staying in state 0 pays 1, staying in state 1 pays 2, and going to the other room pays 0.
- `rlenvs.q_from_v(mdp, v)` computes that whole table of $Q$ values from a vector $v$.

## Compare actions, not policies

Start from "stay everywhere". Staying in state 0 pays 1 every step, so the value is $1 / (1 - 0.9) = 10$. Staying in state 1 pays 2 every step, so it is $2 / 0.1 = 20$.

Now ask a smaller question than "is this the best policy?". In state 0, what if you took a different action for one step and then went back to following the old policy? That is exactly $Q^\pi(s, a)$, and you already have the values to compute it:

| State | Action | $Q^\pi(s, a) = R + 0.9 \times V^\pi(\text{next})$ |
|---|---|---|
| 0 | stay | $1 + 0.9 \times 10 = 10$ |
| 0 | go | $0 + 0.9 \times 20 = 18$ |
| 1 | stay | $2 + 0.9 \times 20 = 20$ |
| 1 | go | $0 + 0.9 \times 10 = 9$ |

In state 0, walking to room 1 is worth 18 and staying is worth 10. The old policy leaves 8 on the table. In state 1, staying is best. So the improved policy is "go from state 0, stay in state 1". It takes in each state the action with the largest $Q$, which is called the *greedy* policy with respect to $V^\pi$:

$$
\pi'(s) = \operatorname*{arg\,max}_a Q^\pi(s, a)
$$

Read it as: in each state, pick the action whose one step of lookahead, followed by the old policy, scores highest. In code it is one table and one `argmax` per row:

```python
q = rlenvs.q_from_v(mdp, v)                 # (S, A) action values
pi = np.zeros_like(q)
pi[np.arange(S), q.argmax(axis=1)] = 1.0    # one-hot rows
```

```figure
<svg viewBox="0 0 570 250" role="img" aria-label="Policy iteration on the two-state world: stay everywhere has values 10 and 20, the greedy policy goes in state 0, and the next round changes nothing">
<defs><marker id="c011-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<text x="14.0" y="30.0" class="fig-bold" text-anchor="start">round 1</text>
<rect x="14.0" y="40.0" width="130.0" height="56.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="79.0" y="65.0" class="fig-bold" text-anchor="middle">policy</text>
<text x="79.0" y="83.0" class="fig-small" text-anchor="middle">stay, stay</text>
<line x1="146.0" y1="68.0" x2="214.0" y2="68.0" class="stroke-ink" marker-end="url(#c011-arrow)"/>
<text x="180.0" y="58.0" class="fig-small" text-anchor="middle">evaluate</text>
<rect x="216.0" y="40.0" width="130.0" height="56.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="281.0" y="65.0" class="fig-bold" text-anchor="middle">values</text>
<text x="281.0" y="83.0" class="fig-small" text-anchor="middle">V = (10, 20)</text>
<line x1="348.0" y1="68.0" x2="416.0" y2="68.0" class="stroke-ink" marker-end="url(#c011-arrow)"/>
<text x="382.0" y="58.0" class="fig-small" text-anchor="middle">greedy</text>
<rect x="418.0" y="40.0" width="130.0" height="56.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="483.0" y="65.0" class="fig-bold" text-anchor="middle">policy</text>
<text x="483.0" y="83.0" class="fig-small" text-anchor="middle">go, stay</text>
<text x="280.0" y="116.0" class="fig-small" text-anchor="middle">state 0: go is worth 18, stay is worth 10</text>
<text x="14.0" y="140.0" class="fig-bold" text-anchor="start">round 2</text>
<rect x="14.0" y="150.0" width="130.0" height="56.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="79.0" y="175.0" class="fig-bold" text-anchor="middle">policy</text>
<text x="79.0" y="193.0" class="fig-small" text-anchor="middle">go, stay</text>
<line x1="146.0" y1="178.0" x2="214.0" y2="178.0" class="stroke-ink" marker-end="url(#c011-arrow)"/>
<text x="180.0" y="168.0" class="fig-small" text-anchor="middle">evaluate</text>
<rect x="216.0" y="150.0" width="130.0" height="56.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="281.0" y="175.0" class="fig-bold" text-anchor="middle">values</text>
<text x="281.0" y="193.0" class="fig-small" text-anchor="middle">V = (18, 20)</text>
<line x1="348.0" y1="178.0" x2="416.0" y2="178.0" class="stroke-ink" marker-end="url(#c011-arrow)"/>
<text x="382.0" y="168.0" class="fig-small" text-anchor="middle">greedy</text>
<rect x="418.0" y="150.0" width="130.0" height="56.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="483.0" y="175.0" class="fig-bold" text-anchor="middle">policy</text>
<text x="483.0" y="193.0" class="fig-small" text-anchor="middle">go, stay</text>
<text x="280.0" y="226.0" class="fig-small" text-anchor="middle">no state improves, so stop</text>
<polyline points="483.0,98.0 483.0,134.0 79.0,134.0 79.0,140.0" class="stroke-ink dashed"/>
<line x1="79.0" y1="140.0" x2="79.0" y2="148.0" class="stroke-ink" marker-end="url(#c011-arrow)"/>
</svg>
Figure 1. Policy iteration on the two-state world. Notice that round 1 changes the policy and lifts the value of state 0 from 10 to 18, while round 2 finds nothing to change, so the loop stops.
```

## Why the greedy policy cannot be worse

The comparison above changes the action for one step only, and then returns to the old policy. The improved policy follows the greedy action at every step. Why should that be at least as good?

Take any state $s$. By construction, the greedy action's value is at least the old policy's average value there:

$$
Q^\pi(s, \pi'(s)) \ge \sum_a \pi(a \mid s)\, Q^\pi(s, a) = V^\pi(s)
$$

The right-hand side is an average of the $Q$ values, and a maximum is never below an average. So "take $\pi'$ for one step, then $\pi$" is no worse than $\pi$ in every state. Now replace the next step too. After the first step you land in some state $s'$, and there "take $\pi'$ for one step, then $\pi$" is again no worse than $\pi$. Swapping one more step of $\pi$ for $\pi'$ never hurts, and if you keep swapping you end with $\pi'$ at every step. This is the *policy improvement theorem*:

$$
V^{\pi'}(s) \ge V^\pi(s) \quad \text{forevery state } s
$$

In the two-state world, the improved policy has values $V^{\pi'}(0) = 0.9 \times 20 = 18$ and $V^{\pi'}(1) = 20$. State 0 rose from 10 to 18, which is exactly the $Q$ value you computed. State 1 stayed at 20. Nothing got worse.

> [!KEY]
> Acting greedily on a policy's values gives a policy that is at least as good in every state, never worse in any. You can improve without trying anything out, because the values you already have contain the comparison.

## Repeat until nothing changes

The improved policy has its own values, and those may reveal better actions still. So evaluate it, improve it again, and keep going. This loop is *policy iteration*:

1. Evaluate the current policy exactly, by solving the linear system.
2. Replace it with the greedy policy for those values.
3. If the greedy policy equals the current one, stop.

On the two-state world, starting from "stay everywhere":

- Evaluation 1 gives $V = (10, 20)$. The greedy policy is "go, stay".
- Evaluation 2 gives $V = (18, 20)$. The table reads: state 0, stay $1 + 0.9 \times 18 = 17.2$ against go $0 + 0.9 \times 20 = 18$. State 1, stay $2 + 0.9 \times 20 = 20$ against go $0 + 0.9 \times 18 = 16.2$. The greedy policy is "go, stay" again, so you stop.

Two evaluations were enough, and $(18, 20)$ is the best this world allows.

### Why it stops, and why it stops at the right place

The values never go down, and each round either lifts some state's value or leaves the policy alone. When some value rises, the policy is a different, better one, and a policy that has been beaten can't return. A finite MDP has finitely many deterministic policies, $A^S$ of them, so you can't keep climbing forever. When the values stay the same, the greedy policy computed from them is the same policy, so the loop stops. Break ties the same way every time, such as the first best action, so that equal actions can't make the policy flip back and forth.

When the loop stops, the policy is greedy with respect to its own values: $V^\pi(s) = \max_a Q^\pi(s, a)$ in every state. That is the Bellman optimality equation, and the next chapter, [value iteration](/lessons/012-value-iteration), shows that its only solution is the optimal value $V^*$. So the policy it stops on is optimal.

## Try it on a bigger world

The random MDP in the exercise has 20 states, 4 actions, and $\gamma = 0.95$. Starting from "action 0 everywhere", policy iteration needs three evaluations. The first greedy step changes the action in 15 of the 20 states, the second changes 1, and the third changes none.

```figure
<svg viewBox="0 0 570 310" role="img" aria-label="Values of all 20 states of a random MDP across three policy iteration evaluations: they jump from about 0 to about 18, then barely move">
<line x1="70.0" y1="223.8" x2="540.0" y2="223.8" class="stroke-grid"/>
<text x="62.0" y="227.8" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="180.0" x2="540.0" y2="180.0" class="stroke-grid"/>
<text x="62.0" y="184.0" class="fig-small" text-anchor="end">5</text>
<line x1="70.0" y1="136.2" x2="540.0" y2="136.2" class="stroke-grid"/>
<text x="62.0" y="140.2" class="fig-small" text-anchor="end">10</text>
<line x1="70.0" y1="92.5" x2="540.0" y2="92.5" class="stroke-grid"/>
<text x="62.0" y="96.5" class="fig-small" text-anchor="end">15</text>
<line x1="70.0" y1="48.8" x2="540.0" y2="48.8" class="stroke-grid"/>
<text x="62.0" y="52.8" class="fig-small" text-anchor="end">20</text>
<line x1="120.0" y1="34.0" x2="120.0" y2="254.0" class="stroke-grid"/>
<text x="120.0" y="272.0" class="fig-small" text-anchor="middle">evaluation 1</text>
<line x1="300.0" y1="34.0" x2="300.0" y2="254.0" class="stroke-grid"/>
<text x="300.0" y="272.0" class="fig-small" text-anchor="middle">evaluation 2</text>
<line x1="480.0" y1="34.0" x2="480.0" y2="254.0" class="stroke-grid"/>
<text x="480.0" y="272.0" class="fig-small" text-anchor="middle">evaluation 3</text>
<text x="120.0" y="288.0" class="fig-small" text-anchor="middle">action 0 everywhere</text>
<text x="300.0" y="288.0" class="fig-small" text-anchor="middle">after one greedy step</text>
<text x="480.0" y="288.0" class="fig-small" text-anchor="middle">after two</text>
<polyline points="120.0,223.5 300.0,64.9 480.0,64.6" class="stroke-a"/>
<circle cx="120.0" cy="223.5" r="3" class="fill-b"/>
<circle cx="300.0" cy="64.9" r="3" class="fill-b"/>
<circle cx="480.0" cy="64.6" r="3" class="fill-b"/>
<polyline points="120.0,232.3 300.0,53.7 480.0,53.4" class="stroke-a"/>
<circle cx="120.0" cy="232.3" r="3" class="fill-b"/>
<circle cx="300.0" cy="53.7" r="3" class="fill-b"/>
<circle cx="480.0" cy="53.4" r="3" class="fill-b"/>
<polyline points="120.0,228.2 300.0,63.9 480.0,63.7" class="stroke-a"/>
<circle cx="120.0" cy="228.2" r="3" class="fill-b"/>
<circle cx="300.0" cy="63.9" r="3" class="fill-b"/>
<circle cx="480.0" cy="63.7" r="3" class="fill-b"/>
<polyline points="120.0,215.1 300.0,53.9 480.0,53.6" class="stroke-a"/>
<circle cx="120.0" cy="215.1" r="3" class="fill-b"/>
<circle cx="300.0" cy="53.9" r="3" class="fill-b"/>
<circle cx="480.0" cy="53.6" r="3" class="fill-b"/>
<polyline points="120.0,215.8 300.0,62.9 480.0,62.6" class="stroke-a"/>
<circle cx="120.0" cy="215.8" r="3" class="fill-b"/>
<circle cx="300.0" cy="62.9" r="3" class="fill-b"/>
<circle cx="480.0" cy="62.6" r="3" class="fill-b"/>
<polyline points="120.0,225.0 300.0,64.3 480.0,64.0" class="stroke-a"/>
<circle cx="120.0" cy="225.0" r="3" class="fill-b"/>
<circle cx="300.0" cy="64.3" r="3" class="fill-b"/>
<circle cx="480.0" cy="64.0" r="3" class="fill-b"/>
<polyline points="120.0,239.7 300.0,66.7 480.0,66.1" class="stroke-a"/>
<circle cx="120.0" cy="239.7" r="3" class="fill-b"/>
<circle cx="300.0" cy="66.7" r="3" class="fill-b"/>
<circle cx="480.0" cy="66.1" r="3" class="fill-b"/>
<polyline points="120.0,232.0 300.0,68.7 480.0,68.4" class="stroke-a"/>
<circle cx="120.0" cy="232.0" r="3" class="fill-b"/>
<circle cx="300.0" cy="68.7" r="3" class="fill-b"/>
<circle cx="480.0" cy="68.4" r="3" class="fill-b"/>
<polyline points="120.0,218.5 300.0,63.0 480.0,62.7" class="stroke-a"/>
<circle cx="120.0" cy="218.5" r="3" class="fill-b"/>
<circle cx="300.0" cy="63.0" r="3" class="fill-b"/>
<circle cx="480.0" cy="62.7" r="3" class="fill-b"/>
<polyline points="120.0,212.0 300.0,50.6 480.0,50.4" class="stroke-a"/>
<circle cx="120.0" cy="212.0" r="3" class="fill-b"/>
<circle cx="300.0" cy="50.6" r="3" class="fill-b"/>
<circle cx="480.0" cy="50.4" r="3" class="fill-b"/>
<polyline points="120.0,245.3 300.0,69.4 480.0,69.1" class="stroke-a"/>
<circle cx="120.0" cy="245.3" r="3" class="fill-b"/>
<circle cx="300.0" cy="69.4" r="3" class="fill-b"/>
<circle cx="480.0" cy="69.1" r="3" class="fill-b"/>
<polyline points="120.0,221.3 300.0,67.1 480.0,66.8" class="stroke-a"/>
<circle cx="120.0" cy="221.3" r="3" class="fill-b"/>
<circle cx="300.0" cy="67.1" r="3" class="fill-b"/>
<circle cx="480.0" cy="66.8" r="3" class="fill-b"/>
<polyline points="120.0,225.1 300.0,59.0 480.0,58.7" class="stroke-a"/>
<circle cx="120.0" cy="225.1" r="3" class="fill-b"/>
<circle cx="300.0" cy="59.0" r="3" class="fill-b"/>
<circle cx="480.0" cy="58.7" r="3" class="fill-b"/>
<polyline points="120.0,208.5 300.0,54.8 480.0,54.5" class="stroke-a"/>
<circle cx="120.0" cy="208.5" r="3" class="fill-b"/>
<circle cx="300.0" cy="54.8" r="3" class="fill-b"/>
<circle cx="480.0" cy="54.5" r="3" class="fill-b"/>
<polyline points="120.0,232.4 300.0,58.7 480.0,58.4" class="stroke-a"/>
<circle cx="120.0" cy="232.4" r="3" class="fill-b"/>
<circle cx="300.0" cy="58.7" r="3" class="fill-b"/>
<circle cx="480.0" cy="58.4" r="3" class="fill-b"/>
<polyline points="120.0,236.9 300.0,71.3 480.0,71.0" class="stroke-a"/>
<circle cx="120.0" cy="236.9" r="3" class="fill-b"/>
<circle cx="300.0" cy="71.3" r="3" class="fill-b"/>
<circle cx="480.0" cy="71.0" r="3" class="fill-b"/>
<polyline points="120.0,234.4 300.0,73.4 480.0,73.1" class="stroke-a"/>
<circle cx="120.0" cy="234.4" r="3" class="fill-b"/>
<circle cx="300.0" cy="73.4" r="3" class="fill-b"/>
<circle cx="480.0" cy="73.1" r="3" class="fill-b"/>
<polyline points="120.0,223.4 300.0,63.3 480.0,63.0" class="stroke-a"/>
<circle cx="120.0" cy="223.4" r="3" class="fill-b"/>
<circle cx="300.0" cy="63.3" r="3" class="fill-b"/>
<circle cx="480.0" cy="63.0" r="3" class="fill-b"/>
<polyline points="120.0,215.1 300.0,57.9 480.0,57.6" class="stroke-a"/>
<circle cx="120.0" cy="215.1" r="3" class="fill-b"/>
<circle cx="300.0" cy="57.9" r="3" class="fill-b"/>
<circle cx="480.0" cy="57.6" r="3" class="fill-b"/>
<polyline points="120.0,240.6 300.0,63.9 480.0,63.6" class="stroke-a"/>
<circle cx="120.0" cy="240.6" r="3" class="fill-b"/>
<circle cx="300.0" cy="63.9" r="3" class="fill-b"/>
<circle cx="480.0" cy="63.6" r="3" class="fill-b"/>
<text x="14.0" y="22.0" class="fig-small" text-anchor="start">value of each of the 20 states</text>
</svg>
Figure 2. Policy iteration on a random 20-state MDP with gamma 0.95. Notice that no line goes down, that the first greedy step lifts every state from about 0 to about 18, and that the third evaluation moves any value by at most 0.06, which is why the loop stops.
```

The third evaluation differs from the second by at most 0.06, and the greedy policy does not change, so the loop stops. The final values equal $V^*$ to the precision of the linear solve, about $10^{-11}$.

> [!WARNING]
> Each round pays for an exact evaluation, which solves an $S \times S$ linear system. That cost is why people also use the method in the next chapter, which skips the full solve.

> [!RECAP]
> - $Q^\pi(s, a)$ is the value of taking $a$ now and following $\pi$ afterward, and `rlenvs.q_from_v` builds the whole table.
> - The greedy policy takes `q.argmax(axis=1)` in every row, with ties going to the first action.
> - The policy improvement theorem says the greedy policy is at least as good as the policy it came from in every state.
> - Policy iteration alternates exact evaluation and greedy improvement, and stops when the policy is unchanged.
> - At the stopping point, the values satisfy the Bellman optimality equation, so the policy is optimal.

## Build it step by step

The checks test the three functions in this order. `two_state_mdp` and `policy_matrices` are given.

1. **`greedy_policy(mdp, v)`.** Takes values of shape `(S,)` and returns a one-hot policy of shape `(S, A)`. Build `q` with `rlenvs.q_from_v`, then set a 1 at each row's `argmax`. Common mistake: breaking ties at random. The checks use an MDP whose two actions are identical and expect action 0 everywhere.
2. **`improve_policy(mdp, pi)`.** Takes a policy and returns `(new_pi, v)`. Get `P_pi` and `r_pi` from `policy_matrices`, solve `np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)`, and call `greedy_policy` on the result. Common mistake: returning the value of `new_pi` instead of the value of the policy you started with. The checks compare against $(10, 20)$ and test the improvement theorem on ten random MDPs.
3. **`policy_iteration(mdp, pi0=None)`.** Start from "action 0 everywhere" when `pi0` is `None`. Loop `improve_policy`, record each value, and stop with `np.array_equal(new_pi, pi)`. Return `(pi, v, np.array(values))`. Common mistake: forgetting to record the last evaluation. The checks expect `(10, 20)` then `(18, 20)` from "stay everywhere", and three rows from "go everywhere".

## Cost and scaling

One round costs $O(S^3)$ for the linear solve plus $O(S^2 A)$ to build the $Q$ table. Policy iteration usually needs a handful of rounds, and for a fixed $\gamma$ the number of rounds is bounded by a polynomial in $S$ and $A$ (Ye, 2011). Memory is dominated by `P`, which holds $S^2 A$ numbers.

## Where this breaks

**It needs the model.** Policy iteration reads `P` and `R`. When you can only sample transitions, later chapters replace the exact evaluation with sampled experience, starting with [Monte Carlo prediction](/lessons/013-monte-carlo-prediction).

**Large state spaces.** An $O(S^3)$ solve is out of reach when $S$ is in the millions. A later part replaces the table with a function approximator.

**Undiscounted tasks.** With $\gamma = 1$, the linear system is singular for a policy that never ends. Start from a policy that reaches a terminal state.

## Further reading

- [Sutton and Barto, *Reinforcement Learning: An Introduction*, Chapter 4](http://incompleteideas.net/book/the-book-2nd.html) covers policy evaluation, policy improvement, and policy iteration, with the grid-world and gambler's examples.
- [Puterman, *Markov Decision Processes: Discrete Stochastic Dynamic Programming* (1994)](https://www.wiley.com/en-us/Markov+Decision+Processes:+Discrete+Stochastic+Dynamic+Programming-p-9781118625873) proves that policy iteration converges, and covers modified policy iteration.
- [Ye, "The simplex and policy-iteration methods are strongly polynomial for the Markov decision problem with a fixed discount rate" (2011)](https://doi.org/10.1287/moor.1110.0516) bounds the number of policy iteration rounds.

## The exercise

You build the improvement step and the loop around it. Write `greedy_policy(mdp, v)`, `improve_policy(mdp, pi)`, and `policy_iteration(mdp, pi0=None)`. The starter gives `two_state_mdp` and `policy_matrices`.

The checks verify that the greedy policy from $V^* = (18, 20)$ is "go, stay" and that ties go to action 0. They check that one round from "stay everywhere" returns the value $(10, 20)$ and the policy "go, stay", and that one round of improvement never lowers any state's value on ten random MDPs. They check that `policy_iteration` evaluates $(10, 20)$ then $(18, 20)$ from "stay everywhere", and $(0, 0)$, $(10, 20)$, $(18, 20)$ from "go everywhere". Last, they check that it finds $V^*$ on the 20-state MDP and never lowers a value on the way.
