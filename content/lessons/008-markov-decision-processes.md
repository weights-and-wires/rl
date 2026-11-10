---
title: Markov decision processes
slug: 008-markov-decision-processes
part: "Part 2 — Planning with a known world"
summary: Model a world where actions change what happens next, sample trajectories from a policy, and average the policy away to get a Markov chain.
minutes: 35
gpu: false
objectives:
  - Name the five parts of a Markov decision process and store them as numpy arrays.
  - Explain the Markov property and why the state must carry everything that matters.
  - Sample a trajectory by following a policy through the transition model.
  - Build the policy's state-to-state matrix and expected reward from the model and the policy.
prereqs:
  - 002-returns-and-discounting
  - 003-explore-or-exploit
exercise: 008-markov-decision-processes
---

# Markov decision processes

In a bandit, your action decides the reward and nothing else. The next round looks exactly like this one. Most real problems are not like that. A chess move changes the board. A thermostat setting changes tomorrow's temperature. A robot that steps left in a hallway ends up somewhere else, and where it ends up decides what it can do next. An action can pay nothing now and still be the right one, because of where it leaves you.

Try the obvious approach first: treat every step as a bandit and pick the action with the best immediate reward. In the hallway you meet in this chapter, every move costs 1, so immediate rewards cannot tell left from right. Yet one direction leads to the exit and the other leads away from it. The difference lives in *where you land*, and a bandit has no notion of that.

A *Markov decision process*, or MDP, adds that notion. You write the world down as a few arrays, and then a policy turns into something you can compute with. This chapter builds the world, samples from it, and averages the policy away. The next two chapters use the result to compute exactly how good a policy is. The same small hallway returns in every chapter of Part 2.

## What you already know

- The agent sees a state, picks an action, and receives a reward and a next state (see [Learning by trial and error](/lessons/001-learning-by-trial-and-error)).
- The return $G_t = r_t + \gamma G_{t+1}$ adds up discounted rewards from step $t$ on (see [Returns and discounting](/lessons/002-returns-and-discounting)).
- A policy gives the probability of each action, as the softmax policy in [Gradient bandits](/lessons/006-gradient-bandits) does.
- `np.random.default_rng(seed).choice(n, p=probs)` draws an index with the given probabilities.
- A matrix row that sums to one is a probability distribution.

## The hallway

Picture four cells in a row. Cell 0 is the start, and it has a wall on its left. Cell 3 is the exit. From cells 0, 1, and 2, you choose *left* or *right*. Every move costs a reward of $-1$, and reaching the exit ends the episode. A move left from cell 0 bumps the wall and leaves you in cell 0. The discount is $\gamma = 0.9$.

```figure
<svg viewBox="0 0 560 205" role="img" aria-label="Four cells in a row with a wall on the left and an exit on the right; right and left arrows each pay minus one">
<defs><marker id="c008-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<rect x="45.0" y="85.0" width="100.0" height="60.0" class="fill-soft-a"/>
<rect x="45" y="85" width="100" height="60" class="stroke-muted" fill="none"/>
<text x="95.0" y="112.0" class="fig-bold" text-anchor="middle">cell 0</text>
<text x="95.0" y="132.0" class="fig-small" text-anchor="middle">start</text>
<rect x="165.0" y="85.0" width="100.0" height="60.0" class="fill-sunk"/>
<rect x="165" y="85" width="100" height="60" class="stroke-muted" fill="none"/>
<text x="215.0" y="112.0" class="fig-bold" text-anchor="middle">cell 1</text>
<text x="215.0" y="132.0" class="fig-small" text-anchor="middle"></text>
<rect x="285.0" y="85.0" width="100.0" height="60.0" class="fill-sunk"/>
<rect x="285" y="85" width="100" height="60" class="stroke-muted" fill="none"/>
<text x="335.0" y="112.0" class="fig-bold" text-anchor="middle">cell 2</text>
<text x="335.0" y="132.0" class="fig-small" text-anchor="middle"></text>
<rect x="405.0" y="85.0" width="100.0" height="60.0" class="fill-soft-c"/>
<rect x="405" y="85" width="100" height="60" class="stroke-muted" fill="none"/>
<text x="455.0" y="112.0" class="fig-bold" text-anchor="middle">goal</text>
<text x="455.0" y="132.0" class="fig-small" text-anchor="middle">exit, episode ends</text>
<line x1="45.0" y1="80.0" x2="45.0" y2="150.0" class="stroke-ink"/>
<text x="45.0" y="72.0" class="fig-small" text-anchor="middle">wall</text>
<line x1="115.0" y1="62.0" x2="195.0" y2="62.0" class="stroke-b" marker-end="url(#c008-arrow)"/>
<text x="155.0" y="52.0" class="fig-small" text-anchor="middle">right, r = −1</text>
<line x1="235.0" y1="62.0" x2="315.0" y2="62.0" class="stroke-b" marker-end="url(#c008-arrow)"/>
<text x="275.0" y="52.0" class="fig-small" text-anchor="middle">right, r = −1</text>
<line x1="355.0" y1="62.0" x2="435.0" y2="62.0" class="stroke-b" marker-end="url(#c008-arrow)"/>
<text x="395.0" y="52.0" class="fig-small" text-anchor="middle">right, r = −1</text>
<line x1="195.0" y1="168.0" x2="115.0" y2="168.0" class="stroke-d" marker-end="url(#c008-arrow)"/>
<text x="155.0" y="190.0" class="fig-small" text-anchor="middle">left, r = −1</text>
<line x1="315.0" y1="168.0" x2="235.0" y2="168.0" class="stroke-d" marker-end="url(#c008-arrow)"/>
<text x="275.0" y="190.0" class="fig-small" text-anchor="middle">left, r = −1</text>
<line x1="75.0" y1="168.0" x2="52.0" y2="168.0" class="stroke-d" marker-end="url(#c008-arrow)"/>
<text x="80.0" y="190.0" class="fig-small" text-anchor="middle">left, bump</text>
</svg>
Figure 1. The hallway MDP. Every move costs −1. Right walks toward the exit, and left walks back toward the wall, where a bump leaves you in cell 0. The reward is the same for both actions, so only the cell you land in tells them apart.
```

An MDP has five parts, and the hallway shows each one:

- A finite set of *states*, numbered $0$ to $S - 1$. The hallway has $S = 4$.
- A finite set of *actions*, numbered $0$ to $A - 1$. Action 0 is left and action 1 is right, so $A = 2$.
- *Transition probabilities* $P(s' \mid s, a)$: the chance of landing in state $s'$ after taking action $a$ in state $s$. Here, $P(2 \mid 1, \text{right}) = 1$.
- *Expected rewards* $R(s, a)$: the average reward $r_t$ for taking $a$ in $s$. Here, $R(s, a) = -1$ in cells 0 to 2.
- A discount $\gamma$ and a start-state distribution $\mu_0$. Here, $\mu_0$ puts all its weight on cell 0.

In code, these are arrays. `mdp.P` has shape `(S, A, S)`, and each `mdp.P[s, a]` is a probability distribution over next states. `mdp.R` has shape `(S, A)`. Some states are *terminal*: the episode ends there, and nothing happens afterward. The hallway is deterministic, so each `P[s, a]` puts all its weight on one cell. In a slippery hallway, it would spread across several cells.

The word *Markov* names the key assumption. The next state and reward depend only on the current state and action, not on how you got there. The state carries everything about the past that matters for the future. If your "state" leaves something out, such as a car's speed when you only record its position, your model's predictions go wrong. When you design a state, ask whether knowing more history would change what happens next.

Here is a number for the obvious approach to fail on. A policy that picks the action with the best immediate reward sees a tie at every cell. Break ties toward left, and you bump the wall forever, for a return of $-1 / (1 - 0.9) = -10$. Walking right three times costs $-1 - 0.9 - 0.81 = -2.71$. Only a model of where each action leads can find that.

## Policies and trajectories

A *policy* $\pi(a \mid s)$ gives the probability of each action in each state. For a finite MDP, it is an array of shape `(S, A)` whose rows sum to one. The *uniform policy* picks each action with probability $1/2$ in every state.

To follow a policy, start in a state drawn from $\mu_0$, draw an action from $\pi$, and ask the model where you land and what you earn. Repeat until you reach a terminal state. The record $s_0, a_0, r_0, s_1, a_1, r_1, \dots$ is a *trajectory*, or an *episode*:

```python
s = rng.choice(S, p=mdp.mu0)
while not mdp.terminal[s]:
    a = rng.choice(A, p=pi[s])
    s2, r = mdp.sample(s, a, rng)
    s = s2
```

One uniform-policy trajectory from cell 0, drawn with seed 0, visits cells $0, 1, 0, 1, 2, 3$. It takes actions right, left, right, right, right. Each step pays $-1$, so its return is $-(1 + 0.9 + 0.81 + 0.729 + 0.6561) = -4.0951$.

Figure 2 shows that trajectories from the same policy differ a lot. Seed 2 wanders for 19 steps and earns $-8.649$. A single trajectory is one random draw, so it cannot tell you how good the policy is. To say something about the policy, you must average over many draws, or you must average over the randomness exactly. The second option is the path of this part of the course.

```figure
<svg viewBox="0 0 600 260" role="img" aria-label="Three jagged paths over cells against time; two reach the goal in about 5 steps and one wanders for 19 steps">
<line x1="70.0" y1="185.8" x2="570.0" y2="185.8" class="stroke-grid"/>
<text x="62.0" y="189.8" class="fig-small" text-anchor="end">cell 0</text>
<line x1="70.0" y1="138.6" x2="570.0" y2="138.6" class="stroke-grid"/>
<text x="62.0" y="142.6" class="fig-small" text-anchor="end">cell 1</text>
<line x1="70.0" y1="91.4" x2="570.0" y2="91.4" class="stroke-grid"/>
<text x="62.0" y="95.4" class="fig-small" text-anchor="end">cell 2</text>
<line x1="70.0" y1="44.2" x2="570.0" y2="44.2" class="stroke-grid"/>
<text x="62.0" y="48.2" class="fig-small" text-anchor="end">goal</text>
<line x1="70.0" y1="200.0" x2="570.0" y2="200.0" class="stroke-muted"/>
<line x1="70.0" y1="30.0" x2="70.0" y2="200.0" class="stroke-muted"/>
<line x1="70.0" y1="200.0" x2="70.0" y2="205.0" class="stroke-muted"/>
<text x="70.0" y="220.0" class="fig-small" text-anchor="middle">0</text>
<line x1="201.6" y1="200.0" x2="201.6" y2="205.0" class="stroke-muted"/>
<text x="201.6" y="220.0" class="fig-small" text-anchor="middle">5</text>
<line x1="333.2" y1="200.0" x2="333.2" y2="205.0" class="stroke-muted"/>
<text x="333.2" y="220.0" class="fig-small" text-anchor="middle">10</text>
<line x1="464.7" y1="200.0" x2="464.7" y2="205.0" class="stroke-muted"/>
<text x="464.7" y="220.0" class="fig-small" text-anchor="middle">15</text>
<line x1="570.0" y1="200.0" x2="570.0" y2="205.0" class="stroke-muted"/>
<text x="570.0" y="220.0" class="fig-small" text-anchor="middle">19</text>
<text x="320.0" y="240.0" class="fig-small" text-anchor="middle">time step t</text>
<polyline points="70.0,182.8 96.3,135.6 122.6,182.8 148.9,135.6 175.3,88.4 201.6,41.2" class="stroke-a" fill="none"/>
<circle cx="70.0" cy="182.8" r="2.5" class="fill-a"/>
<circle cx="96.3" cy="135.6" r="2.5" class="fill-a"/>
<circle cx="122.6" cy="182.8" r="2.5" class="fill-a"/>
<circle cx="148.9" cy="135.6" r="2.5" class="fill-a"/>
<circle cx="175.3" cy="88.4" r="2.5" class="fill-a"/>
<circle cx="201.6" cy="41.2" r="2.5" class="fill-a"/>
<polyline points="70.0,185.8 96.3,138.6 122.6,185.8 148.9,185.8 175.3,138.6 201.6,91.4 227.9,44.2" class="stroke-b" fill="none"/>
<circle cx="70.0" cy="185.8" r="2.5" class="fill-b"/>
<circle cx="96.3" cy="138.6" r="2.5" class="fill-b"/>
<circle cx="122.6" cy="185.8" r="2.5" class="fill-b"/>
<circle cx="148.9" cy="185.8" r="2.5" class="fill-b"/>
<circle cx="175.3" cy="138.6" r="2.5" class="fill-b"/>
<circle cx="201.6" cy="91.4" r="2.5" class="fill-b"/>
<circle cx="227.9" cy="44.2" r="2.5" class="fill-b"/>
<polyline points="70.0,188.8 96.3,188.8 122.6,141.6 148.9,94.4 175.3,141.6 201.6,188.8 227.9,141.6 254.2,188.8 280.5,188.8 306.8,141.6 333.2,188.8 359.5,188.8 385.8,141.6 412.1,188.8 438.4,188.8 464.7,188.8 491.1,188.8 517.4,141.6 543.7,94.4 570.0,47.2" class="stroke-d" fill="none"/>
<circle cx="70.0" cy="188.8" r="2.5" class="fill-d"/>
<circle cx="96.3" cy="188.8" r="2.5" class="fill-d"/>
<circle cx="122.6" cy="141.6" r="2.5" class="fill-d"/>
<circle cx="148.9" cy="94.4" r="2.5" class="fill-d"/>
<circle cx="175.3" cy="141.6" r="2.5" class="fill-d"/>
<circle cx="201.6" cy="188.8" r="2.5" class="fill-d"/>
<circle cx="227.9" cy="141.6" r="2.5" class="fill-d"/>
<circle cx="254.2" cy="188.8" r="2.5" class="fill-d"/>
<circle cx="280.5" cy="188.8" r="2.5" class="fill-d"/>
<circle cx="306.8" cy="141.6" r="2.5" class="fill-d"/>
<circle cx="333.2" cy="188.8" r="2.5" class="fill-d"/>
<circle cx="359.5" cy="188.8" r="2.5" class="fill-d"/>
<circle cx="385.8" cy="141.6" r="2.5" class="fill-d"/>
<circle cx="412.1" cy="188.8" r="2.5" class="fill-d"/>
<circle cx="438.4" cy="188.8" r="2.5" class="fill-d"/>
<circle cx="464.7" cy="188.8" r="2.5" class="fill-d"/>
<circle cx="491.1" cy="188.8" r="2.5" class="fill-d"/>
<circle cx="517.4" cy="141.6" r="2.5" class="fill-d"/>
<circle cx="543.7" cy="94.4" r="2.5" class="fill-d"/>
<circle cx="570.0" cy="47.2" r="2.5" class="fill-d"/>
<line x1="228.2" y1="44.0" x2="246.2" y2="44.0" class="stroke-a"/>
<text x="254.2" y="48.0" class="fig-small" text-anchor="start">episode 1: 5 steps, return -4.095</text>
<line x1="228.2" y1="60.0" x2="246.2" y2="60.0" class="stroke-b"/>
<text x="254.2" y="64.0" class="fig-small" text-anchor="start">episode 2: 6 steps, return -4.686</text>
<line x1="228.2" y1="76.0" x2="246.2" y2="76.0" class="stroke-d"/>
<text x="254.2" y="80.0" class="fig-small" text-anchor="start">episode 3: 19 steps, return -8.649</text>
</svg>
Figure 2. Three episodes of the uniform random policy from cell 0, drawn from seeds 0, 1, and 2. Same policy, same start, and yet they last 5, 6, and 19 steps, with returns of -4.095, -4.686, -8.649.
```

## Averaging the policy away

The randomness in a trajectory comes from two places: the policy's action choice and the world's transition. The world's part sits in `P`. The policy's part sits in `pi`. Average over the policy's action once, at each state, and what remains is one random move per state. That is a plain *Markov chain* with rewards.

$$
P_\pi(s, s') = \sum_a \pi(a \mid s)\, P(s' \mid s, a), \qquad r_\pi(s) = \sum_a \pi(a \mid s)\, R(s, a)
$$

Read it as: $P_\pi(s, s')$ is the chance of moving from $s$ to $s'$ in one step when you act by $\pi$, and $r_\pi(s)$ is the reward you expect in $s$. $P_\pi$ is an $S \times S$ matrix. For a terminal state, set both its row of $P_\pi$ and its $r_\pi$ to zero, because nothing happens after the episode ends.

For the uniform policy in the hallway, each row is an even mix of the *left* row and the *right* row of `P`:

$$
P_\pi = \begin{bmatrix} 0.5 & 0.5 & 0 & 0 \\ 0.5 & 0 & 0.5 & 0 \\ 0 & 0.5 & 0 & 0.5 \\ 0 & 0 & 0 & 0 \end{bmatrix}, \qquad r_\pi = (-1, -1, -1, 0)
$$

Check row 0 by hand. Left bumps the wall and stays in cell 0, with probability $0.5$. Right goes to cell 1, with probability $0.5$. The last row is zero because cell 3 is terminal.

In numpy, one `einsum` does the averaging. The string `"sa,sat->st"` says: for each $s$ and $t$, multiply `pi[s, a]` by `P[s, a, t]` and add over $a$.

```python
P_pi = np.einsum("sa,sat->st", pi, mdp.P)   # (S, S)
r_pi = (pi * mdp.R).sum(axis=1)             # (S,)
live = ~mdp.terminal
P_pi, r_pi = P_pi * live[:, None], r_pi * live
```

The matrix and your sampler must agree. If you draw many first steps from cell 1 under the uniform policy, about half should land in cell 0 and half in cell 2, which are the entries of row 1.

> [!KEY]
> An MDP plus a policy is a Markov chain with rewards. The immediate reward cannot tell left from right in the hallway, but the transition does, and $P_\pi$ and $r_\pi$ hold both.

> [!WARNING]
> Zero the terminal rows. A terminal state with a nonzero reward collects it forever in later calculations, and that inflates the value of every state that can reach it.

> [!RECAP]
> - An MDP is `P` with shape `(S, A, S)`, `R` with shape `(S, A)`, `gamma`, `mu0`, and a `terminal` mask.
> - A policy is an `(S, A)` array whose rows sum to one.
> - A trajectory comes from alternating `rng.choice(A, p=pi[s])` with `mdp.sample(s, a, rng)` until a terminal state.
> - `P_pi = np.einsum("sa,sat->st", pi, P)` and `r_pi = (pi * R).sum(1)`, with terminal rows zeroed.

## Build it step by step

The starter gives you `hallway_mdp`, which builds the MDP above. The checks test your functions in this order.

1. **`sample_episode(mdp, pi, rng)`.** Draws the first state from `mdp.mu0`, then repeats until a terminal state: draw an action from `pi[s]`, call `mdp.sample(s, a, rng)`, and record the step. Returns `(states, actions, rewards)` as lists, where `states` has one more entry than the other two. Common mistake: forgetting to record the terminal state, which leaves `states` the same length as `actions`.
2. **`policy_matrices(mdp, pi)`.** Returns `(P_pi, r_pi)` with shapes `(S, S)` and `(S,)`. Use the `einsum` above and zero the terminal rows. Common mistake: the string `"sa,sat->ts"`, which returns the transpose. The checks test the hallway matrices above, row sums, and the sampler's frequencies.

## Where this breaks

**State that is not Markov.** If the observation leaves out something that matters, such as a moving object's velocity, the true process is not an MDP in those states. People stack recent observations, or give the agent memory.

**An unknown model.** This chapter reads `P` and `R` directly. Most agents never get them. Later chapters estimate values from sampled experience, and [Learning a model and planning with it](/lessons/041-model-based-rl) learns the model itself.

**Huge state spaces.** `P` holds $S^2 A$ numbers. A grid of 1000 cells with 4 actions needs 4 million, which is fine. A board game with $10^{20}$ states needs more memory than exists.

## Further reading

- [Sutton and Barto, *Reinforcement Learning: An Introduction*, Chapter 3](http://incompleteideas.net/book/the-book-2nd.html) defines finite MDPs, value functions, and both Bellman equations.
- [Bellman, "A Markovian decision process" (1957)](http://www.iumj.indiana.edu/IUMJ/FULLTEXT/1957/6/56038) is Bellman's short paper on the model, whose title gave it its name.
- [Puterman, *Markov Decision Processes: Discrete Stochastic Dynamic Programming* (1994)](https://www.wiley.com/en-us/Markov+Decision+Processes:+Discrete+Stochastic+Dynamic+Programming-p-9781118625873) is the standard reference for the theory.
- [Kaelbling, Littman, and Moore, "Reinforcement learning: A survey" (1996)](https://arxiv.org/abs/cs/9605103) reviews the MDP framework and the classical methods built on it.

## The exercise

You build the two basic operations on an MDP. Write `sample_episode(mdp, pi, rng)` to follow a policy through the model, and `policy_matrices(mdp, pi)` to average the policy away. The hallway is defined in the starter file as `hallway_mdp`.

The checks verify that a sampled episode starts in cell 0, ends in the exit, and uses only transitions the model allows, and that the always-right policy produces the episode $0, 1, 2, 3$ with return $-2.71$. They verify that `policy_matrices` reproduces the matrices above, matches the definition on a random MDP, and zeroes terminal rows on the frozen lake. Last, they check that the frequencies in thousands of your sampled steps match the rows of $P_\pi$.
