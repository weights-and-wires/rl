---
title: Options and hierarchical RL
slug: 049-options-and-hierarchical-rl
part: "Part 7 — Models, exploration, and fixed data"
summary: Add temporal abstraction with options, plan over them as a semi-MDP, and learn their values with SMDP and intra-option Q-learning.
minutes: 40
gpu: false
objectives:
  - Define an option by its initiation set, intra-option policy, and termination function, and say why a primitive action is a one-step option.
  - Compute an option's reward model and discounted transition model by solving a linear system, and check it against rollouts.
  - Explain why an option that runs for k steps discounts its bootstrap value by gamma to the k, and run SMDP value iteration over options.
  - Write the SMDP Q-learning update and the intra-option Q-learning update, and say what each one learns from.
  - Say how interruption changes termination, and how option-critic, feudal RL, and HIRO scale the idea up.
prereqs:
  - 008-markov-decision-processes
  - 010-policy-evaluation
  - 012-value-iteration
  - 017-sarsa-and-q-learning
exercise: 049-options-and-hierarchical-rl
---

# Options and hierarchical RL

You plan a trip to a café across town as "go downstairs, walk to the station, take the train, walk to the café", not as ten thousand muscle twitches. An agent that picks one primitive action at a time has to rediscover this structure, and it pays for it: a reward a hundred steps away reaches the start of the episode only after a hundred rounds of bootstrapping.

This chapter gives the agent steps of different lengths. An *option* is a small policy with a start condition and a stopping rule, such as "walk to the door of this room". The chapter shows how to plan with options, how to learn their values, and why the discount must be handled with care when one decision lasts many steps.

## What you already know

- A value function satisfies a Bellman equation, and value iteration applies that equation as an update until the values stop changing (see [Value iteration](/lessons/012-value-iteration)).
- Q-learning moves $Q(s, a)$ toward $r + \gamma \max_{a'} Q(s', a')$ (see [SARSA and Q-learning](/lessons/017-sarsa-and-q-learning)).
- A policy and an MDP define a Markov chain, and its values come from a linear system (see [Policy evaluation](/lessons/010-policy-evaluation)).
- A reward $k$ steps away is worth $\gamma^k$ times a reward now.

## A picture: four rooms

The running example is the four-rooms world of Sutton, Precup, and Singh (1999). The world has 104 open cells in four rooms, joined by four one-cell hallways. The agent starts in the top-left corner and wants to reach the bottom-right corner. Each step costs $-1$, a move into a wall or a slip (probability 0.2) leaves the agent where it is, and $\gamma = 0.9$. The goal pays nothing and ends the episode. Rooms are numbered 0 to 3 from the top-left, row by row.

```figure
<svg viewBox="0 0 600 410" role="img" aria-label="The four-rooms map with the plan from the top-left corner to the goal in the bottom-right corner: two hallway options with primitive moves between and after them">
<defs><marker id="c048-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<rect x="20" y="20" width="30" height="30" class="fill-ink"/>
<rect x="50" y="20" width="30" height="30" class="fill-ink"/>
<rect x="80" y="20" width="30" height="30" class="fill-ink"/>
<rect x="110" y="20" width="30" height="30" class="fill-ink"/>
<rect x="140" y="20" width="30" height="30" class="fill-ink"/>
<rect x="170" y="20" width="30" height="30" class="fill-ink"/>
<rect x="200" y="20" width="30" height="30" class="fill-ink"/>
<rect x="230" y="20" width="30" height="30" class="fill-ink"/>
<rect x="260" y="20" width="30" height="30" class="fill-ink"/>
<rect x="290" y="20" width="30" height="30" class="fill-ink"/>
<rect x="320" y="20" width="30" height="30" class="fill-ink"/>
<rect x="350" y="20" width="30" height="30" class="fill-ink"/>
<rect x="380" y="20" width="30" height="30" class="fill-ink"/>
<rect x="20" y="50" width="30" height="30" class="fill-ink"/>
<rect x="50" y="50" width="30" height="30" class="fill-soft-a stroke-grid"/>
<rect x="80" y="50" width="30" height="30" class="fill-soft-a stroke-grid"/>
<rect x="110" y="50" width="30" height="30" class="fill-soft-a stroke-grid"/>
<rect x="140" y="50" width="30" height="30" class="fill-soft-a stroke-grid"/>
<rect x="170" y="50" width="30" height="30" class="fill-soft-a stroke-grid"/>
<rect x="200" y="50" width="30" height="30" class="fill-ink"/>
<rect x="230" y="50" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="260" y="50" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="290" y="50" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="320" y="50" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="350" y="50" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="380" y="50" width="30" height="30" class="fill-ink"/>
<rect x="20" y="80" width="30" height="30" class="fill-ink"/>
<rect x="50" y="80" width="30" height="30" class="fill-soft-a stroke-grid"/>
<rect x="80" y="80" width="30" height="30" class="fill-soft-a stroke-grid"/>
<rect x="110" y="80" width="30" height="30" class="fill-soft-a stroke-grid"/>
<rect x="140" y="80" width="30" height="30" class="fill-soft-a stroke-grid"/>
<rect x="170" y="80" width="30" height="30" class="fill-soft-a stroke-grid"/>
<rect x="200" y="80" width="30" height="30" class="fill-ink"/>
<rect x="230" y="80" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="260" y="80" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="290" y="80" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="320" y="80" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="350" y="80" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="380" y="80" width="30" height="30" class="fill-ink"/>
<rect x="20" y="110" width="30" height="30" class="fill-ink"/>
<rect x="50" y="110" width="30" height="30" class="fill-soft-a stroke-grid"/>
<rect x="80" y="110" width="30" height="30" class="fill-soft-a stroke-grid"/>
<rect x="110" y="110" width="30" height="30" class="fill-soft-a stroke-grid"/>
<rect x="140" y="110" width="30" height="30" class="fill-soft-a stroke-grid"/>
<rect x="170" y="110" width="30" height="30" class="fill-soft-a stroke-grid"/>
<rect x="200" y="110" width="30" height="30" class="fill-b stroke-grid"/>
<rect x="230" y="110" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="260" y="110" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="290" y="110" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="320" y="110" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="350" y="110" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="380" y="110" width="30" height="30" class="fill-ink"/>
<rect x="20" y="140" width="30" height="30" class="fill-ink"/>
<rect x="50" y="140" width="30" height="30" class="fill-soft-a stroke-grid"/>
<rect x="80" y="140" width="30" height="30" class="fill-soft-a stroke-grid"/>
<rect x="110" y="140" width="30" height="30" class="fill-soft-a stroke-grid"/>
<rect x="140" y="140" width="30" height="30" class="fill-soft-a stroke-grid"/>
<rect x="170" y="140" width="30" height="30" class="fill-soft-a stroke-grid"/>
<rect x="200" y="140" width="30" height="30" class="fill-ink"/>
<rect x="230" y="140" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="260" y="140" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="290" y="140" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="320" y="140" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="350" y="140" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="380" y="140" width="30" height="30" class="fill-ink"/>
<rect x="20" y="170" width="30" height="30" class="fill-ink"/>
<rect x="50" y="170" width="30" height="30" class="fill-soft-a stroke-grid"/>
<rect x="80" y="170" width="30" height="30" class="fill-soft-a stroke-grid"/>
<rect x="110" y="170" width="30" height="30" class="fill-soft-a stroke-grid"/>
<rect x="140" y="170" width="30" height="30" class="fill-soft-a stroke-grid"/>
<rect x="170" y="170" width="30" height="30" class="fill-soft-a stroke-grid"/>
<rect x="200" y="170" width="30" height="30" class="fill-ink"/>
<rect x="230" y="170" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="260" y="170" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="290" y="170" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="320" y="170" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="350" y="170" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="380" y="170" width="30" height="30" class="fill-ink"/>
<rect x="20" y="200" width="30" height="30" class="fill-ink"/>
<rect x="50" y="200" width="30" height="30" class="fill-ink"/>
<rect x="80" y="200" width="30" height="30" class="fill-b stroke-grid"/>
<rect x="110" y="200" width="30" height="30" class="fill-ink"/>
<rect x="140" y="200" width="30" height="30" class="fill-ink"/>
<rect x="170" y="200" width="30" height="30" class="fill-ink"/>
<rect x="200" y="200" width="30" height="30" class="fill-ink"/>
<rect x="230" y="200" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="260" y="200" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="290" y="200" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="320" y="200" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="350" y="200" width="30" height="30" class="fill-soft-b stroke-grid"/>
<rect x="380" y="200" width="30" height="30" class="fill-ink"/>
<rect x="20" y="230" width="30" height="30" class="fill-ink"/>
<rect x="50" y="230" width="30" height="30" class="fill-soft-c stroke-grid"/>
<rect x="80" y="230" width="30" height="30" class="fill-soft-c stroke-grid"/>
<rect x="110" y="230" width="30" height="30" class="fill-soft-c stroke-grid"/>
<rect x="140" y="230" width="30" height="30" class="fill-soft-c stroke-grid"/>
<rect x="170" y="230" width="30" height="30" class="fill-soft-c stroke-grid"/>
<rect x="200" y="230" width="30" height="30" class="fill-ink"/>
<rect x="230" y="230" width="30" height="30" class="fill-ink"/>
<rect x="260" y="230" width="30" height="30" class="fill-ink"/>
<rect x="290" y="230" width="30" height="30" class="fill-b stroke-grid"/>
<rect x="320" y="230" width="30" height="30" class="fill-ink"/>
<rect x="350" y="230" width="30" height="30" class="fill-ink"/>
<rect x="380" y="230" width="30" height="30" class="fill-ink"/>
<rect x="20" y="260" width="30" height="30" class="fill-ink"/>
<rect x="50" y="260" width="30" height="30" class="fill-soft-c stroke-grid"/>
<rect x="80" y="260" width="30" height="30" class="fill-soft-c stroke-grid"/>
<rect x="110" y="260" width="30" height="30" class="fill-soft-c stroke-grid"/>
<rect x="140" y="260" width="30" height="30" class="fill-soft-c stroke-grid"/>
<rect x="170" y="260" width="30" height="30" class="fill-soft-c stroke-grid"/>
<rect x="200" y="260" width="30" height="30" class="fill-ink"/>
<rect x="230" y="260" width="30" height="30" class="fill-soft-d stroke-grid"/>
<rect x="260" y="260" width="30" height="30" class="fill-soft-d stroke-grid"/>
<rect x="290" y="260" width="30" height="30" class="fill-soft-d stroke-grid"/>
<rect x="320" y="260" width="30" height="30" class="fill-soft-d stroke-grid"/>
<rect x="350" y="260" width="30" height="30" class="fill-soft-d stroke-grid"/>
<rect x="380" y="260" width="30" height="30" class="fill-ink"/>
<rect x="20" y="290" width="30" height="30" class="fill-ink"/>
<rect x="50" y="290" width="30" height="30" class="fill-soft-c stroke-grid"/>
<rect x="80" y="290" width="30" height="30" class="fill-soft-c stroke-grid"/>
<rect x="110" y="290" width="30" height="30" class="fill-soft-c stroke-grid"/>
<rect x="140" y="290" width="30" height="30" class="fill-soft-c stroke-grid"/>
<rect x="170" y="290" width="30" height="30" class="fill-soft-c stroke-grid"/>
<rect x="200" y="290" width="30" height="30" class="fill-b stroke-grid"/>
<rect x="230" y="290" width="30" height="30" class="fill-soft-d stroke-grid"/>
<rect x="260" y="290" width="30" height="30" class="fill-soft-d stroke-grid"/>
<rect x="290" y="290" width="30" height="30" class="fill-soft-d stroke-grid"/>
<rect x="320" y="290" width="30" height="30" class="fill-soft-d stroke-grid"/>
<rect x="350" y="290" width="30" height="30" class="fill-soft-d stroke-grid"/>
<rect x="380" y="290" width="30" height="30" class="fill-ink"/>
<rect x="20" y="320" width="30" height="30" class="fill-ink"/>
<rect x="50" y="320" width="30" height="30" class="fill-soft-c stroke-grid"/>
<rect x="80" y="320" width="30" height="30" class="fill-soft-c stroke-grid"/>
<rect x="110" y="320" width="30" height="30" class="fill-soft-c stroke-grid"/>
<rect x="140" y="320" width="30" height="30" class="fill-soft-c stroke-grid"/>
<rect x="170" y="320" width="30" height="30" class="fill-soft-c stroke-grid"/>
<rect x="200" y="320" width="30" height="30" class="fill-ink"/>
<rect x="230" y="320" width="30" height="30" class="fill-soft-d stroke-grid"/>
<rect x="260" y="320" width="30" height="30" class="fill-soft-d stroke-grid"/>
<rect x="290" y="320" width="30" height="30" class="fill-soft-d stroke-grid"/>
<rect x="320" y="320" width="30" height="30" class="fill-soft-d stroke-grid"/>
<rect x="350" y="320" width="30" height="30" class="fill-soft-d stroke-grid"/>
<rect x="380" y="320" width="30" height="30" class="fill-ink"/>
<rect x="20" y="350" width="30" height="30" class="fill-ink"/>
<rect x="50" y="350" width="30" height="30" class="fill-soft-c stroke-grid"/>
<rect x="80" y="350" width="30" height="30" class="fill-soft-c stroke-grid"/>
<rect x="110" y="350" width="30" height="30" class="fill-soft-c stroke-grid"/>
<rect x="140" y="350" width="30" height="30" class="fill-soft-c stroke-grid"/>
<rect x="170" y="350" width="30" height="30" class="fill-soft-c stroke-grid"/>
<rect x="200" y="350" width="30" height="30" class="fill-ink"/>
<rect x="230" y="350" width="30" height="30" class="fill-soft-d stroke-grid"/>
<rect x="260" y="350" width="30" height="30" class="fill-soft-d stroke-grid"/>
<rect x="290" y="350" width="30" height="30" class="fill-soft-d stroke-grid"/>
<rect x="320" y="350" width="30" height="30" class="fill-soft-d stroke-grid"/>
<rect x="350" y="350" width="30" height="30" class="fill-soft-d stroke-grid"/>
<rect x="380" y="350" width="30" height="30" class="fill-ink"/>
<rect x="20" y="380" width="30" height="30" class="fill-ink"/>
<rect x="50" y="380" width="30" height="30" class="fill-ink"/>
<rect x="80" y="380" width="30" height="30" class="fill-ink"/>
<rect x="110" y="380" width="30" height="30" class="fill-ink"/>
<rect x="140" y="380" width="30" height="30" class="fill-ink"/>
<rect x="170" y="380" width="30" height="30" class="fill-ink"/>
<rect x="200" y="380" width="30" height="30" class="fill-ink"/>
<rect x="230" y="380" width="30" height="30" class="fill-ink"/>
<rect x="260" y="380" width="30" height="30" class="fill-ink"/>
<rect x="290" y="380" width="30" height="30" class="fill-ink"/>
<rect x="320" y="380" width="30" height="30" class="fill-ink"/>
<rect x="350" y="380" width="30" height="30" class="fill-ink"/>
<rect x="380" y="380" width="30" height="30" class="fill-ink"/>
<text x="365" y="370" class="fig-bold" text-anchor="middle">G</text>
<polyline points="65,65 95,65 95,95 95,125 95,155 95,185 95,215 95,245 125,245 155,245 185,245 185,275 185,305 215,305 245,305 245,335 245,365 275,365 305,365 335,365 365,365" class="stroke-ink" fill="none" marker-end="url(#c048-arrow)"/>
<circle cx="65" cy="65" r="5" class="fill-ink"/>
<circle cx="95" cy="245" r="5" class="fill-ink"/>
<text x="65" y="69" class="fig-small fig-bold" text-anchor="middle">S</text>
<text x="434" y="40" class="fig-bold" text-anchor="start">Plan from S</text>
<text x="434" y="64" class="fig-small" text-anchor="start">(1, 1): room0->(6, 2)</text>
<text x="434" y="82" class="fig-small" text-anchor="start">(6, 2): 1 primitive move</text>
<text x="434" y="100" class="fig-small" text-anchor="start">(7, 2): room2->(9, 6)</text>
<text x="434" y="118" class="fig-small" text-anchor="start">(9, 6): 7 primitive moves</text>
<rect x="434" y="144" width="14" height="14" class="fill-b stroke-grid"/>
<text x="456" y="156" class="fig-small" text-anchor="start">hallway cell</text>
</svg>
Figure 1. The plan that a planner with primitive actions and eight hallway options picks from the top-left corner. It uses two hallway options and eight primitive moves, 20 steps in total. Both routes around the walls are equally short, so the planner could also have gone through the top-right room.
```

Each room offers two options, one for each of its hallways. The option "go to hallway $h$" can start anywhere in its room, follows the shortest path to $h$, and stops when it gets there. The agent can still use primitive actions wherever it likes.

## Options and the one-step special case

An **option** has three parts:

- an **initiation set** $\mathcal{I}$, the states where the option can start;
- an **intra-option policy** $\pi_o(a \mid s)$, which picks primitive actions while the option runs;
- a **termination function** $\beta(s) \in [0, 1]$, the probability that the option stops on arriving in state $s$.

A primitive action $a$ is an option too. It can start everywhere, always picks $a$, and has $\beta = 1$ everywhere, so it stops after one step. So one framework covers primitives and abstractions, and a planner can mix them.

## Why an option that runs k steps discounts by gamma^k

Suppose the agent starts option $o$ in state $s$, and it runs for $k$ steps and ends in state $s'$. The agent collects $r_0, r_1, \dots, r_{k-1}$ along the way. From $s$, the total value of that choice is the discounted sum of those rewards plus the value of where it ends up. The ending is $k$ steps away, so it counts $\gamma^k$ times:

$$
G = r_0 + \gamma r_1 + \dots + \gamma^{k-1} r_{k-1} + \gamma^k V(s')
$$

Here is a worked example. An option runs five steps at $-1$ each, so $\gamma = 0.9$ gives a return of $-(1 + 0.9 + 0.81 + 0.729 + 0.6561) = -4.0951$. The agent ends in a state worth $V(s') = -6$. The correct target is $-4.0951 + 0.9^5 \cdot (-6) = -7.6380$. Discounting the ending by $\gamma$ alone gives $-4.0951 - 5.4 = -9.4951$, which makes the option look worse than it is. Summing the rewards without discounting errs in the other direction.

Because $k$ changes from run to run, fold $\gamma^k$ into a model. The **option model** has two pieces, defined for every state $s$:

$$
R_o(s) = \mathbb{E}\Big[\sum_{t=0}^{k-1} \gamma^t r_t\Big], \qquad P_o(s, x) = \mathbb{E}\big[\gamma^k \, \mathbb{1}(s_k = x)\big]
$$

Read $R_o(s)$ as the expected discounted reward collected while the option runs. Read $P_o(s, x)$ as the chance of ending in $x$, weighted by how much its $k$ steps of delay shrink it. A row of $P_o$ sums to $\mathbb{E}[\gamma^k]$.

## Computing a model by solving a linear system

An option's model satisfies a one-step recursion. After the first step the option either stops, with probability $\beta(s')$, or carries on from $s'$. Write $P_\pi$ for the transition matrix under $\pi_o$ and $r_\pi$ for its one-step rewards. Then

$$
R_o = r_\pi + \gamma\, P_\pi (I - B) R_o, \qquad P_o = \gamma\, P_\pi B + \gamma\, P_\pi (I - B) P_o
$$

where $B = \text{diag}(\beta)$. Each equation says: what you get from $s$ is the first step, plus $\gamma$ times what you get from wherever you land if the option keeps going. Solving gives

$$
R_o = (I - \gamma P_\pi (I - B))^{-1} r_\pi, \qquad P_o = (I - \gamma P_\pi (I - B))^{-1} \gamma P_\pi B
$$

In numpy this is two calls to `np.linalg.solve`.

A one-line sanity check works from the cell $(3, 5)$, next to the hallway at $(3, 6)$. The option steps right. It succeeds with probability 0.8 and otherwise stays put, so $R_o = -1 + 0.2 \cdot 0.9 \cdot R_o$, which gives $R_o = -1 / 0.82 = -1.2195$. The same argument gives $P_o(\text{hallway}) = 0.72 / 0.82 = 0.8780$. From the start cell, the same option has $R_o = -5.9763$ and a row sum of 0.4024.

## Planning over options as a semi-MDP

With models in hand, the choice among options is an MDP with a twist: each decision lasts a random number of steps. Such a process is a **semi-Markov decision process**, or SMDP. Its Bellman optimality equation is the familiar one, with the model standing in for the one-step reward and the discounted transition:

$$
V^*(s) = \max_{o \,:\, s \in \mathcal{I}_o} \Big[ R_o(s) + \sum_x P_o(s, x)\, V^*(x) \Big]
$$

Read it as: pick the option that pays the most reward now plus the discounted value of where it ends. There is no separate $\gamma$ in the formula because $P_o$ already holds $\gamma^k$. Sutton, Precup, and Singh show that choosing among options and running each to termination is an SMDP, so value iteration and Q-learning carry over.

Hallway options are built from primitive moves, so they cannot change the best achievable values. Value iteration over primitives and the eight hallway options returns the same $V^*$ as primitives alone, to within $10^{-12}$, with $V^*(\text{start}) = -9.2581$.

What the options change is how fast information travels. Figure 2 plots the largest value error after each sweep.

```figure
<svg viewBox="0 0 600 340" role="img" aria-label="Largest value error after each sweep on a log scale: planning with hallway options reaches the 1e-6 tolerance in 25 sweeps and primitives alone in 41">
<line x1="70.0" y1="270.0" x2="540.0" y2="270.0" class="stroke-grid"/>
<text x="62.0" y="274.0" class="fig-small" text-anchor="end">1e-7</text>
<line x1="70.0" y1="212.5" x2="540.0" y2="212.5" class="stroke-grid"/>
<text x="62.0" y="216.5" class="fig-small" text-anchor="end">1e-5</text>
<line x1="70.0" y1="155.0" x2="540.0" y2="155.0" class="stroke-grid"/>
<text x="62.0" y="159.0" class="fig-small" text-anchor="end">1e-3</text>
<line x1="70.0" y1="97.5" x2="540.0" y2="97.5" class="stroke-grid"/>
<text x="62.0" y="101.5" class="fig-small" text-anchor="end">1e-1</text>
<line x1="70.0" y1="40.0" x2="540.0" y2="40.0" class="stroke-grid"/>
<text x="62.0" y="44.0" class="fig-small" text-anchor="end">1e1</text>
<line x1="70.0" y1="270.0" x2="540.0" y2="270.0" class="stroke-muted"/>
<line x1="70.0" y1="270.0" x2="70.0" y2="275.0" class="stroke-muted"/>
<text x="70.0" y="290.0" class="fig-small" text-anchor="middle">0</text>
<line x1="174.4" y1="270.0" x2="174.4" y2="275.0" class="stroke-muted"/>
<text x="174.4" y="290.0" class="fig-small" text-anchor="middle">10</text>
<line x1="278.9" y1="270.0" x2="278.9" y2="275.0" class="stroke-muted"/>
<text x="278.9" y="290.0" class="fig-small" text-anchor="middle">20</text>
<line x1="383.3" y1="270.0" x2="383.3" y2="275.0" class="stroke-muted"/>
<text x="383.3" y="290.0" class="fig-small" text-anchor="middle">30</text>
<line x1="487.8" y1="270.0" x2="487.8" y2="275.0" class="stroke-muted"/>
<text x="487.8" y="290.0" class="fig-small" text-anchor="middle">40</text>
<text x="305.0" y="318" class="fig-small" text-anchor="middle">sweep</text>
<line x1="70.0" y1="241.2" x2="540.0" y2="241.2" class="stroke-bad dashed"/>
<text x="536.0" y="236.2" class="fig-small" text-anchor="end">tolerance 1e-6</text>
<polyline points="70.0,41.6 80.4,43.2 90.9,44.9 101.3,46.5 111.8,48.1 122.2,49.7 132.7,51.4 143.1,53.0 153.6,54.6 164.0,56.2 174.4,57.9 184.9,59.4 195.3,60.7 205.8,62.1 216.2,63.5 226.7,65.0 237.1,66.5 247.6,68.0 258.0,69.5 268.4,70.8 278.9,72.2 289.3,73.6 299.8,75.5 310.2,78.5 320.7,82.7 331.1,88.0 341.6,94.3 352.0,101.5 362.4,109.5 372.9,118.3 383.3,127.6 393.8,137.6 404.2,148.2 414.7,159.2 425.1,170.6 435.6,182.5 446.0,194.7 456.4,207.3 466.9,220.3 477.3,233.5 487.8,247.0 498.2,260.7 508.7,270.0 519.1,270.0 529.6,270.0 540.0,270.0" class="stroke-a" fill="none"/>
<line x1="100.0" y1="50" x2="130.0" y2="50" class="stroke-a"/>
<text x="136.0" y="54" class="fig-small" text-anchor="start">primitives only</text>
<polyline points="70.0,41.6 80.4,43.2 90.9,44.9 101.3,46.5 111.8,48.1 122.2,49.7 132.7,51.4 143.1,53.0 153.6,56.6 164.0,63.3 174.4,69.4 184.9,73.0 195.3,81.0 205.8,90.9 216.2,102.0 226.7,114.3 237.1,127.4 247.6,141.3 258.0,155.7 268.4,170.7 278.9,186.1 289.3,201.9 299.8,218.1 310.2,234.5 320.7,251.2 331.1,268.2 341.6,270.0 352.0,270.0 362.4,270.0 372.9,270.0 383.3,270.0 393.8,270.0 404.2,270.0 414.7,270.0 425.1,270.0 435.6,270.0 446.0,270.0 456.4,270.0 466.9,270.0 477.3,270.0 487.8,270.0 498.2,270.0 508.7,270.0 519.1,270.0 529.6,270.0 540.0,270.0" class="stroke-b" fill="none"/>
<line x1="100.0" y1="68" x2="130.0" y2="68" class="stroke-b"/>
<text x="136.0" y="72" class="fig-small" text-anchor="start">primitives and hallway options</text>
</svg>
Figure 2. Largest error against the optimal values after each sweep. Hallway options carry the goal's value across a room in one sweep, so the error falls faster and reaches the 1e-6 tolerance after 25 sweeps. Primitives alone need 41.
```

> [!KEY]
> An option that lasts $k$ steps discounts what follows it by $\gamma^k$, because the next decision happens $k$ steps later. A planner that treats a long option as one step with discount $\gamma$ is solving a different problem and gets different values.

Planning starts every non-goal state at $\min(R) / (1 - \gamma) = -10$, a lower bound on the true values, and keeps the goal at 0. The start matters. From the optimistic $V = 0$, a primitive step looks cheaper than a multi-step option in the early sweeps, and both planners take 39 sweeps.

## Learning option values: SMDP Q-learning

When the model is unknown, the agent learns $Q(s, o)$ from experience. It picks an option, runs it to termination, and records the discounted return `ret`, the number of steps `k`, and the ending state `s2`. The SMDP Q-learning update is then ordinary Q-learning with the duration in the exponent:

$$
Q(s, o) \leftarrow Q(s, o) + \alpha \Big[ \underbrace{\textstyle\sum_{t=0}^{k-1} \gamma^t r_t}_{\text{ret}} + \gamma^k \max_{o'} Q(s', o') - Q(s, o) \Big]
$$

The max runs over options that can start in $s'$. After a terminal state the bootstrap term is 0.

```python
best = 0.0 if terminated else np.max(np.where(avail[s2], Q[s2], -np.inf))
Q[s, o] += alpha * (ret + gamma ** k * best - Q[s, o])
```

SMDP Q-learning learns only about the option it ran, and it updates once per option however many steps that took.

## Learning faster: intra-option Q-learning

Suppose the agent runs the option "go to hallway $h$" and takes one step to the right. Any other option whose intra-option policy also says "right" in this state would have made the same step. The step is a valid sample for all of them. **Intra-option learning** uses it: after every primitive step $(s, a, r, s')$, update every option $o$ that can start in $s$ and would pick $a$ there:

$$
Q(s, o) \leftarrow Q(s, o) + \alpha \big[ r + \gamma\, U(s', o) - Q(s, o) \big]
$$

$$
U(s', o) = (1 - \beta_o(s'))\, Q(s', o) + \beta_o(s') \max_{o'} Q(s', o')
$$

Read $U$ as the value of arriving in $s'$ while "in" option $o$: if the option continues, its own value carries on, and if it stops, the agent picks the best option there. The agent can behave under one option and learn about every other option consistent with its action, which is off-policy learning about options.

On four rooms, with 100000 primitive steps and epsilon-greedy behavior, the mean error against the optimal option values is 0.867 for SMDP Q-learning and 0.221 for intra-option Q-learning, both averaged over two seeds. Intra-option learning makes more updates from the same experience, so it is closer. The exercise runs exactly this comparison.

## Termination and interruption

An option is a commitment, and commitments go stale: the agent may be halfway to a hallway when another option becomes clearly better. **Interruption** adds a stopping rule. The option also stops wherever some available option has a strictly higher value than continuing:

$$
\beta'(s) = \begin{cases} 1 & \text{if } \max_{o'} Q(s, o') > Q(s, o) \\ \beta(s) & \text{otherwise} \end{cases}
$$

Sutton, Precup, and Singh prove that, with the true values, this never makes the policy worse. The inequality is strict on purpose: when two options tie, interrupting buys nothing.

> [!WARNING]
> Interruption compares $Q(s, o)$ with the best alternative, so it needs a $Q$ that you trust. With noisy early estimates it can switch options at random.

## Deep versions of the idea

Three methods learn the hierarchy with function approximation. The exercise does not implement them.

- **Option-critic** (Bacon, Harb, and Precup, 2017) learns the intra-option policies and the termination functions by policy gradient, using the option-value function as a critic.
- **Feudal RL** (Dayan and Hinton, 1993) stacks managers and workers. A manager sets a goal for the level below and rewards it for reaching that goal.
- **HIRO** (Nachum et al., 2018) is a goal-based hierarchy that trains off-policy. The high-level policy picks a goal in state space every few steps, and the low-level policy gets reward for getting close to it. Old high-level experience is relabeled with new goals, so it stays valid while the low level changes. The low-level policy is a goal-conditioned policy, one that takes the goal as an extra input.

> [!KEY]
> Options, feudal levels, and HIRO's goals all answer one question: how can a decision last more than one step without breaking the Bellman equation? The answer is to track how long it lasts and discount by that.

> [!RECAP]
> - An option is $(\mathcal{I}, \pi_o, \beta)$, and a primitive action is the option that stops after one step.
> - An option's model is $R_o = (I - \gamma P_\pi (I - B))^{-1} r_\pi$ and $P_o = (I - \gamma P_\pi (I - B))^{-1} \gamma P_\pi B$, where $\gamma^k$ is folded into $P_o$.
> - SMDP value iteration and SMDP Q-learning use $\gamma^k$ for the bootstrap. Using $\gamma$ or dropping the discount inside the option gives wrong values.
> - Intra-option Q-learning updates every option consistent with the action taken, with the target $r + \gamma U(s', o)$.
> - Interruption ends an option wherever another option is strictly better.

## Build it step by step

The checks test the functions in this order. The starter file already holds the four-rooms world, the options, `run_option`, and the `learn` loop. `P` has shape `(S, A, S)`, `R` has shape `(S, A)`, and an option's `init`, `pi`, and `beta` each have shape `(S,)`.

1. **`discounted_return(rewards, gamma)`.** Multiply the rewards by `gamma ** np.arange(len(rewards))` and sum. The mistake is summing the rewards without the powers.
2. **`option_model(env, option)`.** Pick the rows `env.P[np.arange(S), option.pi]`, build the matrix `M = I - gamma * P_pi * (1 - beta)[None, :]`, and call `np.linalg.solve` twice. The mistake is applying `beta` to the rows instead of the columns.
3. **`smdp_value_iteration(env, options, tol)`.** Each sweep, stack `R_o + P_o @ V` for every option, mask the options that cannot start with `-np.inf`, and take the max. Keep the goal at 0 and count every sweep, including the last. The mistake is a second `gamma` in the backup.
4. **`smdp_q_update(Q, s, o, ret, k, s2, terminated, avail, alpha, gamma)`.** Use `gamma ** k`, and take the max only over `avail[s2]`.
5. **`intra_option_q_update(Q, s, a, r, s2, terminated, options, alpha, gamma)`.** Loop over the options that can start in `s` and have `pi[s] == a`. The mistake is updating only the option that is running.
6. **`interrupted_beta(options, Q, o)`.** Compare the best available $Q$ in each state with `Q[:, o]` using `>`, and keep the option's own `beta` elsewhere.

## Where this breaks

**Someone has to supply the options.** The hallway options work because a person knew where the hallways were. Learning which options to have is an open problem, and learned options can collapse into one option that never terminates or into options that stop after every step.

**Markov options only.** Intra-option learning needs each option's policy and termination to depend on the current state alone. An option with a timer or a memory breaks the update above.

## Further reading

- [Sutton, Precup, and Singh (1999), Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning](https://doi.org/10.1016/S0004-3702(99)00052-1). Artificial Intelligence 112, pages 181 to 211. Options, option models, SMDP planning, intra-option learning, interruption, and the four-rooms example.
- [Sutton and Barto, Reinforcement Learning: An Introduction, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html). Section 17.2, "Temporal Abstraction via Options", in the chapter on frontiers.
- [Bacon, Harb, and Precup (2017), The option-critic architecture](https://arxiv.org/abs/1609.05140). AAAI 2017. Policy gradients for intra-option policies and terminations.
- [Dayan and Hinton (1993), Feudal reinforcement learning](https://proceedings.neurips.cc/paper/1992/hash/d14220ee66aeec73c49038385428ec4c-Abstract.html). Advances in Neural Information Processing Systems 5. Managers set goals for sub-managers at coarser time scales.
- [Nachum, Gu, Lee, and Levine (2018), Data-efficient hierarchical reinforcement learning](https://arxiv.org/abs/1805.08296). NeurIPS 2018. HIRO, with its off-policy correction.
- [CMU 10-703, Deep Reinforcement Learning and Control, Spring 2018](https://www.cs.cmu.edu/~rsalakhu/10703/lectures.html). The schedule lists one lecture, "Hierarchical RL and Tranfer Learning" (spelled as on the page), on April 25, with no lecture numbers.

## The exercise

You build the pieces that let an agent plan and learn with options. Write `discounted_return` and `option_model`, then `smdp_value_iteration`, `smdp_q_update`, `intra_option_q_update`, and `interrupted_beta`.

The checks verify an option model against the closed form at $(3, 5)$ and against Monte Carlo rollouts. Value iteration over primitives must take 41 sweeps, and with hallway options it must return the same optimal values in 25. The checks also verify the $\gamma^k$ in `smdp_q_update`, that `intra_option_q_update` changes exactly the consistent options, and that `interrupted_beta` ignores ties. Last, both learners run for 100000 steps, and intra-option learning must have less than half of SMDP learning's error.
