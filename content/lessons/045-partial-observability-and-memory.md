---
title: Partial observability and memory
slug: 045-partial-observability-and-memory
part: "Part 7 — Models, exploration, and fixed data"
summary: When the agent sees only part of the state, track a belief with Bayes' rule or train a recurrent policy, and see why memoryless policies stall.
minutes: 40
gpu: false
objectives:
  - Define a POMDP and update a belief with Bayes' rule on the tiger problem.
  - Compute exactly what the best policy without memory earns on a T-maze, and say why it falls short.
  - Reset a recurrent hidden state at episode boundaries and compute a masked loss over padded episodes.
  - Train a recurrent policy by policy gradient and compare recurrence with frame stacking.
prereqs:
  - 008-markov-decision-processes
  - 031-the-policy-gradient
  - 033-baselines-and-actor-critic
exercise: 045-partial-observability-and-memory
---

# Partial observability and memory

You walk down a corridor. At the start, a light flashes red or blue to tell you which of two doors at the far end hides the prize. Then the corridor goes dark and every step looks the same. At the end you must pick a door. A clerk who sees only the current moment, a dark corridor and two doors, cannot do better than a coin flip. You can do better, because you remember the light.

Every method so far assumed the agent sees the full state, so the current observation was all it needed. Real problems rarely work that way. A robot sees one camera frame, not the whole room. A chatbot sees a transcript, not the user's intent. This chapter shows exactly how much a policy without memory loses, and gives you two ways to add memory: keep a probability distribution over the hidden state, or let a recurrent network learn its own summary of the past.

## What you already know

- An MDP has states $s$, actions $a$, transitions, and rewards, and the next state depends only on the current state and action (see [Markov decision processes](/lessons/008-markov-decision-processes)).
- A policy $\pi(a \mid s)$ is trained by following the gradient of expected return (see [The policy gradient](/lessons/031-the-policy-gradient)).
- A baseline lowers the variance of that gradient without changing its mean (see [Baselines and actor-critic](/lessons/033-baselines-and-actor-critic)).
- Bayes' rule turns a prior and a likelihood into a posterior.
- A tabular policy's exact value comes from solving a linear system.

## What goes wrong without memory

A *partially observable MDP* (POMDP) is an MDP where the agent receives an observation $o$ instead of the state $s$. It adds an observation rule $O(o \mid s', a)$: the chance of seeing $o$ after the world lands in $s'$ following action $a$. Two different states can produce the same observation, and that is where the trouble starts.

Figure 1 draws the corridor as a T-maze. The signal says "left" or "right" with equal probability. The agent advances four steps, then opens a door. The correct door pays $+1$, the wrong door pays $-1$, and any other action walks into a wall and ends the episode with $-1$. With $\gamma = 0.95$, the reward arrives four steps after the signal.

```figure
<svg viewBox="0 0 600 260" role="img" aria-label="A T-maze: a signal at the start, four identical corridor steps, and a junction with two doors">
<defs><marker id="c042-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<rect x="30" y="110" width="76" height="46" rx="6" class="fill-soft-a stroke-muted"/>
<text x="68" y="138" class="fig-bold" text-anchor="middle">start</text>
<line x1="108" y1="133" x2="124" y2="133" class="stroke-ink" marker-end="url(#c042-arrow)"/>
<rect x="126" y="110" width="76" height="46" rx="6" class="fill-sunk stroke-muted"/>
<text x="164" y="138" class="fig-bold" text-anchor="middle">step 1</text>
<line x1="204" y1="133" x2="220" y2="133" class="stroke-ink" marker-end="url(#c042-arrow)"/>
<rect x="222" y="110" width="76" height="46" rx="6" class="fill-sunk stroke-muted"/>
<text x="260" y="138" class="fig-bold" text-anchor="middle">step 2</text>
<line x1="300" y1="133" x2="316" y2="133" class="stroke-ink" marker-end="url(#c042-arrow)"/>
<rect x="318" y="110" width="76" height="46" rx="6" class="fill-sunk stroke-muted"/>
<text x="356" y="138" class="fig-bold" text-anchor="middle">step 3</text>
<line x1="396" y1="133" x2="412" y2="133" class="stroke-ink" marker-end="url(#c042-arrow)"/>
<rect x="414" y="110" width="76" height="46" rx="6" class="fill-soft-b stroke-muted"/>
<text x="452" y="138" class="fig-bold" text-anchor="middle">junction</text>
<text x="68" y="76" class="fig-small" text-anchor="middle">sees the signal</text>
<text x="68" y="92" class="fig-small" text-anchor="middle">left or right, 50/50</text>
<line x1="68" y1="98" x2="68" y2="108" class="stroke-a" marker-end="url(#c042-arrow)"/>
<text x="260" y="196" class="fig-small" text-anchor="middle">steps 1 to 3 look identical: the signal is gone</text>
<text x="260" y="214" class="fig-small" text-anchor="middle">so the agent must carry it in memory</text>
<line x1="452" y1="108" x2="452" y2="66" class="stroke-b" marker-end="url(#c042-arrow)"/>
<line x1="452" y1="158" x2="452" y2="200" class="stroke-b" marker-end="url(#c042-arrow)"/>
<text x="452" y="54" class="fig-small" text-anchor="middle">left door: +1 or -1</text>
<text x="452" y="222" class="fig-small" text-anchor="middle">right door: +1 or -1</text>
<text x="540" y="246" class="fig-small" text-anchor="end">full state: 0.8145   memoryless: 0.0000   (0.95^4 = 0.8145)</text>
</svg>
Figure 1. The T-maze. The signal at the start is the only clue, and four identical steps separate it from the decision. Full-state value is 0.8145 and the best memoryless value is 0.
```

An agent that sees the full state, signal included, advances and opens the right door, and earns $\gamma^4 = 0.8145$. Now take a *memoryless* policy, one that picks its action from the current observation alone. At the junction the observation is the same for both signals, so the policy cannot match the door to the signal. Whatever door it opens, it is right half the time and wrong half the time, which is worth $0.5 \times (+1) + 0.5 \times (-1) = 0$. The exercise enumerates every deterministic memoryless policy on this maze and finds the best is exactly 0. Random stochastic policies do no better, because mixing doors only averages two equal values.

> [!KEY]
> A policy that sees only the current observation cannot use information that arrived earlier. When the observation hides the state, a memoryless policy can fall arbitrarily far below the best policy: here 0 against 0.8145, and the gap grows with the stakes.

## Belief: remembering with Bayes' rule

If the agent knew the model, it could keep a *belief* $b$, a probability distribution over states that summarizes everything it has seen. The belief is a sufficient summary: a policy that acts on $b$ loses nothing compared with one that sees the whole history. That is why Silver's lecture treats the belief state as a Markov state, so a POMDP becomes an MDP over beliefs (see Further reading).

The classic example is the tiger problem. A tiger sits behind the left or right door, equally likely. You can listen, which costs 1 and tells you the tiger's side correctly 85% of the time, or you can open a door. Opening a door re-places the tiger at random and gives an uninformative observation.

### Work it by hand

Start with $b = (0.5, 0.5)$ for tiger-left and tiger-right. Listen and hear the tiger on the left.

- Listening leaves the state alone, so the predicted belief is still $(0.5, 0.5)$.
- The likelihood of hearing "left" is 0.85 if the tiger is left and 0.15 if it is right. Multiplying gives $(0.425, 0.075)$.
- The sum is 0.5, so the new belief is $(0.85, 0.15)$.

Listen again and hear "left" again. Multiply $(0.85, 0.15)$ by $(0.85, 0.15)$ to get $(0.7225, 0.0225)$. The sum is 0.745, so the belief is $(0.9698, 0.0302)$. A third listen that hears "right" multiplies by $(0.15, 0.85)$ and lands back at $(0.85, 0.15)$, because one "right" cancels one "left". Opening a door is different: from $(0.97, 0.03)$, the transition re-places the tiger, so the belief becomes $(0.5, 0.5)$ before the observation is applied, and the coin-flip observation changes nothing.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="Belief that the tiger is on the left after k listens that all hear left, rising from 0.5 toward 1">
<defs><marker id="c042-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<line x1="70.0" y1="250.0" x2="540.0" y2="250.0" class="stroke-grid"/>
<text x="62.0" y="254.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="70.0" y1="208.0" x2="540.0" y2="208.0" class="stroke-grid"/>
<text x="62.0" y="212.0" class="fig-small" text-anchor="end">0.6</text>
<line x1="70.0" y1="166.0" x2="540.0" y2="166.0" class="stroke-grid"/>
<text x="62.0" y="170.0" class="fig-small" text-anchor="end">0.7</text>
<line x1="70.0" y1="124.0" x2="540.0" y2="124.0" class="stroke-grid"/>
<text x="62.0" y="128.0" class="fig-small" text-anchor="end">0.8</text>
<line x1="70.0" y1="82.0" x2="540.0" y2="82.0" class="stroke-grid"/>
<text x="62.0" y="86.0" class="fig-small" text-anchor="end">0.9</text>
<line x1="70.0" y1="40.0" x2="540.0" y2="40.0" class="stroke-grid"/>
<text x="62.0" y="44.0" class="fig-small" text-anchor="end">1.0</text>
<line x1="70.0" y1="250.0" x2="540.0" y2="250.0" class="stroke-muted"/>
<text x="70.0" y="268.0" class="fig-small" text-anchor="middle">0</text>
<text x="148.3" y="268.0" class="fig-small" text-anchor="middle">1</text>
<text x="226.7" y="268.0" class="fig-small" text-anchor="middle">2</text>
<text x="305.0" y="268.0" class="fig-small" text-anchor="middle">3</text>
<text x="383.3" y="268.0" class="fig-small" text-anchor="middle">4</text>
<text x="461.7" y="268.0" class="fig-small" text-anchor="middle">5</text>
<text x="540.0" y="268.0" class="fig-small" text-anchor="middle">6</text>
<polyline points="70.0,250.0 148.3,103.0 226.7,52.7 305.0,42.3 383.3,40.4 461.7,40.1 540.0,40.0" class="stroke-a" fill="none"/>
<circle cx="70.0" cy="250.0" r="4" class="fill-a"/>
<text x="78.0" y="268.0" class="fig-small">0.5000</text>
<circle cx="148.3" cy="103.0" r="4" class="fill-a"/>
<text x="156.3" y="121.0" class="fig-small">0.8500</text>
<circle cx="226.7" cy="52.7" r="4" class="fill-a"/>
<text x="234.7" y="70.7" class="fig-small">0.9698</text>
<circle cx="305.0" cy="42.3" r="4" class="fill-a"/>
<text x="313.0" y="60.3" class="fig-small">0.9945</text>
<circle cx="383.3" cy="40.4" r="4" class="fill-a"/>
<circle cx="461.7" cy="40.1" r="4" class="fill-a"/>
<circle cx="540.0" cy="40.0" r="4" class="fill-a"/>
<text x="305.0" y="292" class="fig-small" text-anchor="middle">listens, each one hearing the tiger on the left</text>
<text x="22" y="145.0" class="fig-small" text-anchor="middle" transform="rotate(-90 22 145.0)">belief: tiger on the left</text>
</svg>
Figure 2. The belief that the tiger is on the left after k listens that all hear left. Each listen multiplies the odds by 0.85 / 0.15, so confidence grows fast and never reaches 1.
```

The rule has two steps, predict and correct. Written for the new state $s'$, with $T(s' \mid s, a)$ the transition probability:

$$
b'(s') = \frac{O(o \mid s', a) \sum_{s} T(s' \mid s, a)\, b(s)}{\sum_{s''} O(o \mid s'', a) \sum_{s} T(s'' \mid s, a)\, b(s)}
$$

Read it as: push the old belief through the transition to predict where the world is now, weight each state by how likely it makes the observation you got, and divide by the total so the result sums to 1. In numpy, with `T` of shape `(A, S, S)` and `O` of shape `(A, S, n_obs)`:

```python
predicted = b @ T[a]                  # predict
posterior = predicted * O[a, :, o]    # correct
b_new = posterior / posterior.sum()   # normalize
```

Kaelbling, Littman, and Cassandra (1998) develop this update and exact planning on beliefs. Exact planning is expensive, because the belief space is continuous and grows with the number of states. For large problems, people let a network learn the summary instead.

## Recurrence: a memory the network learns

A recurrent policy keeps a hidden vector $h_t$ and updates it at every step from the new observation: $h_t = f(o_t, h_{t-1})$. Then it picks the action from $h_t$. The function $f$ is a *gated recurrent unit* (GRU) in this chapter, a small network with learnable gates that decide what to keep and what to overwrite. Nobody tells it to store the signal. Policy gradient rewards it when storing the signal helps, and the training loop discovers the habit.

Two details decide whether this works.

**Reset the hidden state at episode boundaries.** The state carried out of one episode describes a different world. If it leaks into the next one, the policy acts on a stale signal, and the gradient teaches it to attend to something that does not exist. So before the step that starts an episode, multiply $h$ by zero. The reset also cuts the gradient: nothing computed after the boundary depends on anything before it, so no learning signal flows back across it.

**Pad and mask episodes of different lengths.** Episodes end at different times. Stack them in a `(T, B)` array, pad the short ones, and keep a mask that is 1 on real steps and 0 on padding. The loss sums over real steps and divides by their count:

$$
L(\theta) = -\frac{1}{\sum_{t,b} m_{t,b}} \sum_{t,b} m_{t,b}\, A_{t,b} \log \pi_\theta(a_{t,b} \mid h_{t,b})
$$

Here $m$ is the mask and $A$ the advantage. Because padding has $m = 0$, it adds nothing to the loss or the gradient, however much garbage it holds. The test of correctness is strict: the gradient of the padded batch must equal the gradient you get by running each episode alone, one step at a time, and summing.

```python
h = reset_hidden(h, first[t])     # zero the rows that start an episode
h = cell(obs[t], h)               # one GRU step, shape (B, H)
```

Real systems also cut long streams into chunks of fixed length and backpropagate only inside each chunk. That is *truncated backpropagation through time*: the hidden state carries over between chunks, but the gradient does not. The T-maze episodes here last five steps, so the exercise backpropagates through the whole episode.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="Greedy value during training: the recurrent policy rises to 0.8145 while the memoryless policy stays at 0">
<defs><marker id="c042-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<line x1="70.0" y1="239.5" x2="540.0" y2="239.5" class="stroke-grid"/>
<text x="62.0" y="243.5" class="fig-small" text-anchor="end">-1.0</text>
<line x1="70.0" y1="187.0" x2="540.0" y2="187.0" class="stroke-grid"/>
<text x="62.0" y="191.0" class="fig-small" text-anchor="end">-0.5</text>
<line x1="70.0" y1="134.5" x2="540.0" y2="134.5" class="stroke-grid"/>
<text x="62.0" y="138.5" class="fig-small" text-anchor="end">0.0</text>
<line x1="70.0" y1="82.0" x2="540.0" y2="82.0" class="stroke-grid"/>
<text x="62.0" y="86.0" class="fig-small" text-anchor="end">0.5</text>
<text x="305.0" y="290.0" class="fig-small" text-anchor="middle">policy-gradient iteration</text>
<text x="70.0" y="272.0" class="fig-small" text-anchor="middle">0</text>
<text x="226.7" y="272.0" class="fig-small" text-anchor="middle">50</text>
<text x="383.3" y="272.0" class="fig-small" text-anchor="middle">100</text>
<text x="540.0" y="272.0" class="fig-small" text-anchor="middle">150</text>
<line x1="70.0" y1="49.0" x2="540.0" y2="49.0" class="stroke-muted dashed"/>
<text x="540.0" y="43.0" class="fig-small" text-anchor="end">full-state optimum 0.8145</text>
<polyline points="70.0,239.5 85.7,220.0 101.3,220.0 117.0,220.0 132.7,220.0 148.3,134.5 164.0,134.5 179.7,134.5 195.3,134.5 211.0,134.5 226.7,134.5 242.3,134.5 258.0,134.5 273.7,134.5 289.3,134.5 305.0,134.5 320.7,134.5 336.3,134.5 352.0,134.5 367.7,134.5 383.3,134.5 399.0,134.5 414.7,134.5 430.3,134.5 446.0,134.5 461.7,134.5 477.3,134.5 493.0,134.5 508.7,134.5 524.3,134.5 540.0,134.5" class="stroke-a" fill="none"/>
<polyline points="70.0,239.5 85.7,220.0 101.3,220.0 117.0,134.5 132.7,134.5 148.3,134.5 164.0,134.5 179.7,49.0 195.3,49.0 211.0,49.0 226.7,49.0 242.3,49.0 258.0,49.0 273.7,49.0 289.3,49.0 305.0,49.0 320.7,49.0 336.3,49.0 352.0,49.0 367.7,49.0 383.3,49.0 399.0,49.0 414.7,49.0 430.3,49.0 446.0,49.0 461.7,49.0 477.3,49.0 493.0,49.0 508.7,49.0 524.3,49.0 540.0,49.0" class="stroke-b" fill="none"/>
<line x1="300" y1="190" x2="330" y2="190" class="stroke-b"/>
<text x="336" y="194" class="fig-small">with memory: 0.8145 at iteration 150</text>
<line x1="300" y1="210" x2="330" y2="210" class="stroke-a"/>
<text x="336" y="214" class="fig-small">no memory: 0.0000 at iteration 150</text>
</svg>
Figure 3. Exact greedy value during policy-gradient training, seed 0. Both learners reach 0, the memoryless limit. Only the recurrent policy goes on to read the signal and reach 0.8145.
```

Figure 3 trains the policy on the maze. Greedy value, the exact return of the action the policy considers most likely, starts at $-1$ because an untrained policy walks into a wall. Both learners soon learn to advance and reach 0. Only the one with memory goes on to read the signal, and reaches the full-state optimum 0.8145.

## Frame stacking or recurrence

A second way to add memory needs no recurrent network: feed the policy the last $k$ observations as one input. DQN on Atari does this with four frames. It is a window of fixed length, so the policy can use anything that happened within it and nothing that happened earlier.

The T-maze shows the limit exactly. The signal appears at step 0 and the decision comes at step 4, so the window must reach back five observations. The exercise machinery gives the best window policy for each $k$:

| Window $k$ | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Best value | 0 | 0 | 0 | 0 | 0.8145 |

At $k = 4$ the window still shows only corridor and junction, so the policy earns nothing. At $k = 5$ it sees the signal and matches the full-state optimum. A recurrent policy does not need to know the delay in advance, and its cost does not grow with the memory it might use. A window grows linearly in $k$ and must be chosen before training. Hausknecht and Stone (2015) compared the two on Atari with a recurrent layer in DQN. They found recurrence is a viable alternative to stacking frames and gives no systematic learning advantage, but it adapts better when the quality of observations changes at test time.

> [!RECAP]
> - A POMDP gives the agent observations that can hide the state, so the current observation is not a Markov state.
> - A belief updates in two steps, predict through the transition and correct with the observation likelihood, then normalize.
> - The best memoryless policy on the T-maze earns exactly 0. Full state earns $\gamma^4 = 0.8145$.
> - A recurrent policy resets its hidden state at episode starts, and a mask keeps padding out of the loss and the gradient.
> - A window of $k$ observations needs $k = 5$ here. Recurrence needs no choice of $k$.

## Build it step by step

You fill in `starter.py` in this order. The tiger arrays, the T-maze, the environment, and the sampling helpers are given.

1. `belief_update(b, a, o, T, O)` takes a belief of shape `(S,)` and returns a new one. Multiply by `T[a]`, then by the observation column, then normalize. The common mistake is skipping the transition. Listening hides the mistake, because `T` is the identity there, and opening a door exposes it.
2. `memoryless_value(mdp, obs_of_state, pi_obs)` expands `pi_obs[o, a]` to a state policy with `pi_obs[obs_of_state]` and returns `mdp.mu0 @ v` from `rlenvs.policy_evaluation`.
3. `best_memoryless_value(mdp, obs_of_state)` tries every one-hot assignment of actions to the observations that occur in non-terminal states, and returns the best value and its policy. Use `itertools.product`.
4. `reset_hidden(h, first)` zeroes the rows of `h` where `first` is 1, and returns a new tensor.
5. `unroll(cell, obs, first, h0=None)` loops over time, calls `reset_hidden` before each cell step, and stacks the hidden states into shape `(T, B, H)`. The mistake is resetting after the step, which wipes the output of the first step instead of its input.
6. `masked_pg_loss(logits, actions, advantages, mask)` gathers the log-probability of each action and divides by `mask.sum()`. The mistake is dividing by the number of array entries, which counts padding as steps.
7. `sequence_loss(cell, head, obs, actions, advantages, mask, first)` chains `unroll`, `head`, and `masked_pg_loss`.
8. `train_policy(seed, memory=True)` collects a batch, computes discounted returns and a per-time-step baseline over the running steps, and takes an Adam step on `sequence_loss`. For `memory=False`, pass a `first` of all ones, which resets every step.

## Where this breaks

- **Long memory.** A GRU trained by backpropagation through time struggles when the signal comes hundreds of steps before the decision. The gradient shrinks or grows as it passes back through each step. Attention over a long context, as language-model agents use, sidesteps this by keeping the past as readable input.
- **Truncation.** Cutting streams into chunks stops the gradient at the cut, so the network cannot learn dependencies longer than a chunk.
- **Stored hidden states.** In a replay buffer, a hidden state saved from an older policy no longer matches what the current network would compute. Methods store it and refresh it with a short burn-in sequence.
- **Exact beliefs.** The belief update needs the model $T$ and $O$, and exact planning over beliefs does not scale. Learned memory trades the guarantees for generality.

## Further reading

- [CS 185/285 Deep Reinforcement Learning, UC Berkeley](https://rail.eecs.berkeley.edu/deeprlcourse/): the lecture list has Lecture 5, Policy Gradients, which is the base of this chapter's training loop. No lecture on the list is devoted to POMDPs, so this chapter fills that gap.
- [David Silver's UCL reinforcement learning lectures](https://davidstarsilver.wordpress.com/teaching/): Lecture 2, Markov Decision Processes, ends with extensions to MDPs, including partially observable MDPs, belief states that satisfy the Markov property, and the reduction of a POMDP to a tree of belief states.
- [Sutton and Barto, Reinforcement Learning: An Introduction, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html): section 17.3, Observations and State, in the chapter on frontiers.
- [Kaelbling, Littman, and Cassandra, Planning and acting in partially observable stochastic domains (1998)](https://doi.org/10.1016/S0004-3702(98)00023-X): the standard reference for POMDPs and belief-state planning, with the tiger problem.
- [Hausknecht and Stone, Deep Recurrent Q-Learning for Partially Observable MDPs (2015)](https://arxiv.org/abs/1507.06527): replaces DQN's first fully connected layer with an LSTM and compares recurrence with frame stacking on Atari.

## The exercise

You build the memory pieces for a policy that sees only part of the world. `belief_update` is Bayes' rule on the tiger problem, checked against exact posteriors with 0.85 observation accuracy. `memoryless_value` and `best_memoryless_value` compute exactly what a policy without memory earns on the T-maze. `reset_hidden`, `unroll`, `masked_pg_loss`, and `sequence_loss` form the masked sequence loss, and the checks compare its gradient with the step-by-step gradient and with finite differences, and confirm that no gradient crosses an episode boundary. `train_policy` ties them together: the checks train a GRU policy on five seeds and require it to beat the best memoryless value by at least 0.5, while the same learner with its memory switched off stays at or below that value.
