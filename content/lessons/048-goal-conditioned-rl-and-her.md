---
title: Goals and hindsight
slug: 048-goal-conditioned-rl-and-her
part: "Part 7 — Models, exploration, and fixed data"
summary: Learn one policy for every goal, then turn each failed episode into a success for the goal it did reach, with hindsight relabeling.
minutes: 40
gpu: false
objectives:
  - Write the goal-conditioned MDP with the sparse reward 0 when the goal is reached and -1 otherwise.
  - Train one Q(s, a, g) for every goal, and say what it shares across goals.
  - Relabel an episode with the final, future, and episode strategies, and recompute each reward for the new goal.
  - Explain why relabeling is valid for off-policy Q-learning and not for an on-policy policy gradient.
  - Build the supervised targets that goal-conditioned supervised learning (GCSL) trains on.
prereqs:
  - 026-deep-q-networks
  - 031-the-policy-gradient
  - 046-exploration-bonuses
exercise: 048-goal-conditioned-rl-and-her
---

# Goals and hindsight

A robot arm must push a puck to a spot on a table. The reward is the one you would write first: 1 when the puck is on the spot, 0 otherwise. Early in training the arm waves around at random, and the puck almost never lands on the spot. Every episode scores zero, so the learner has nothing to learn from. Worse, you want an arm that reaches any spot you name, not one fixed spot.

This chapter fixes both problems with one idea. Give the policy the goal as an input, so one network serves every goal. Then notice that an episode that failed to reach the goal it was given did reach *some* goal, namely the state where it ended. Pretend that was the goal all along, and the same episode becomes a success. Andrychowicz and others (2017) call this hindsight experience replay (HER).

## What you already know

- A Q-function $Q(s, a)$ estimates the return of taking $a$ in $s$ and acting well afterward, and DQN fits it from a replay buffer with a target network (see [Deep Q-networks](/lessons/026-deep-q-networks)).
- Q-learning is off-policy: its update is valid for any transition $(s, a, r, s')$, whatever policy collected it.
- A policy gradient needs actions sampled from the policy being improved (see [The policy gradient](/lessons/031-the-policy-gradient)).
- A sparse reward gives almost no signal until the agent stumbles on the goal by luck (see [Exploration beyond epsilon](/lessons/046-exploration-bonuses)). [Reward design and shaping](/lessons/057-reward-design-and-shaping) fixes this by redesigning the reward, and this chapter fixes it by relabeling the data.

## A goal as part of the state

In a *goal-conditioned MDP*, each episode comes with a goal $g$ that the agent can read. The dynamics $P(s' \mid s, a)$ do not depend on $g$. Only the reward does:

$$
r(s, a, g) = \begin{cases} 0 & \text{if the goal is reached} \\ -1 & \text{otherwise} \end{cases}
$$

Read it as: every step costs 1 until the goal is reached, so maximizing return means reaching the goal quickly. This is the *sparse* goal reward. Schaul and others (2015) call a value function that takes the goal as an input a *universal value function approximator* (UVFA), $Q(s, a, g)$. In code, you join the state and goal vectors and feed them to one network:

```python
q_values = q_net(torch.cat([state, goal], dim=1))      # shape (B, n_actions)
```

The goal makes one network share what it learned between goals. Learning to flip a bit on the way to goal A also teaches the network something about reaching goal B.

### The running example

The task is *bit-flipping*. The state is a string of $n$ bits, the goal is another string of $n$ bits, and action $i$ flips bit $i$. An episode lasts at most $n$ steps, and the reward is 0 when the state equals the goal and -1 otherwise. The best policy flips each wrong bit once, so a goal at Hamming distance $d$ takes $d$ steps.

With $\gamma = 0.9$, the optimal value for a goal $d$ steps away is $Q^* = -(1 - 0.9^{d-1}) / (1 - 0.9)$. That gives $0$ at $d = 1$, because the single step reaches the goal and earns 0, then $-1$ at $d = 2$, $-1.9$ at $d = 3$, and $-2.71$ at $d = 4$.

The trouble is the first reward. A random policy flips a random bit each step. Figure 1 shows the exact chance that it reaches a random goal within $n$ steps, from a Markov chain on the Hamming distance.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="Chance that a random policy reaches a random goal, falling from 0.22 at 4 bits to 1.8e-05 at 20 bits">
<line x1="70.0" y1="220.0" x2="540.0" y2="220.0" class="stroke-grid"/>
<text x="62.0" y="224.0" class="fig-small" text-anchor="end">1e-6</text>
<line x1="70.0" y1="160.0" x2="540.0" y2="160.0" class="stroke-grid"/>
<text x="62.0" y="164.0" class="fig-small" text-anchor="end">1e-4</text>
<line x1="70.0" y1="100.0" x2="540.0" y2="100.0" class="stroke-grid"/>
<text x="62.0" y="104.0" class="fig-small" text-anchor="end">1e-2</text>
<line x1="70.0" y1="40.0" x2="540.0" y2="40.0" class="stroke-grid"/>
<text x="62.0" y="44.0" class="fig-small" text-anchor="end">1</text>
<polyline points="70.0,59.8 148.3,72.6 226.7,86.7 305.0,101.6 383.3,117.1 461.7,149.2 540.0,182.3" class="stroke-b" fill="none"/>
<circle cx="70.0" cy="59.8" r="4" class="fill-b"/>
<text x="70.0" y="270.0" class="fig-small" text-anchor="middle">4</text>
<text x="78.0" y="49.8" class="fig-small" text-anchor="start">0.22</text>
<circle cx="148.3" cy="72.6" r="4" class="fill-b"/>
<text x="148.3" y="270.0" class="fig-small" text-anchor="middle">6</text>
<text x="148.3" y="62.6" class="fig-small" text-anchor="middle">0.082</text>
<circle cx="226.7" cy="86.7" r="4" class="fill-b"/>
<text x="226.7" y="270.0" class="fig-small" text-anchor="middle">8</text>
<text x="226.7" y="76.7" class="fig-small" text-anchor="middle">0.028</text>
<circle cx="305.0" cy="101.6" r="4" class="fill-b"/>
<text x="305.0" y="270.0" class="fig-small" text-anchor="middle">10</text>
<text x="305.0" y="91.6" class="fig-small" text-anchor="middle">0.0088</text>
<circle cx="383.3" cy="117.1" r="4" class="fill-b"/>
<text x="383.3" y="270.0" class="fig-small" text-anchor="middle">12</text>
<text x="383.3" y="107.1" class="fig-small" text-anchor="middle">0.0027</text>
<circle cx="461.7" cy="149.2" r="4" class="fill-b"/>
<text x="461.7" y="270.0" class="fig-small" text-anchor="middle">16</text>
<text x="461.7" y="139.2" class="fig-small" text-anchor="middle">0.00023</text>
<circle cx="540.0" cy="182.3" r="4" class="fill-b"/>
<text x="540.0" y="270.0" class="fig-small" text-anchor="middle">20</text>
<text x="540.0" y="172.3" class="fig-small" text-anchor="middle">1.8e-05</text>
<text x="305.0" y="292.0" class="fig-small" text-anchor="middle">number of bits n (episode length n)</text>
</svg>
Figure 1. The exact chance that a random policy reaches a random goal within n steps. Every two extra bits cut it by a factor of about three, so at 12 bits a reward is a 1-in-370 event.
```

At $n = 12$ the chance is 0.27%, so a buffer of a few thousand random episodes holds only a handful of rewards, and a plain goal-conditioned DQN at 12 bits learns nothing: its greedy success rate stays at 0.00 on all ten seeds we tried. Each failed episode is wasted data, because the agent gets -1 on every step whatever it does.

## Hindsight: change the goal, keep the episode

Look at the episode again. The agent aimed at goal $g$ and ended at state $s_T$. It did not reach $g$, but it did reach $s_T$, and it did so by a specific sequence of actions. That sequence is a perfect demonstration of how to reach $s_T$. If the goal had been $s_T$, the last step would have earned a reward of 0.

HER stores, next to each real transition, copies whose goal is replaced by a state the episode actually visited. The reward is *recomputed* for the new goal. Only the goal and the reward change, because the transition itself, $(s, a, s')$, really happened.

### Work it by hand

Take $n = 3$ and an episode that starts at 000 with goal 111. The agent flips bit 0, then bit 1, then bit 0 again, and visits 000, 100, 110, 010. It never reaches 111, so all three rewards are -1.

Relabel with the final state, 010. The three transitions become:

| Step | $s$ | $a$ | $s'$ | goal $g'$ | $r(s', g')$ |
|---|---|---|---|---|---|
| 0 | 000 | 0 | 100 | 010 | -1 |
| 1 | 100 | 1 | 110 | 010 | -1 |
| 2 | 110 | 0 | 010 | 010 | 0 |

The last transition is a success, so the buffer now holds one reward of 0 that came from an episode with none. The other two teach the agent that moving to 100 or 110 does not reach 010, and a Q-function needs that signal too.

### Three ways to choose the new goal

Let the episode visit states $s_0, \dots, s_T$. The goal that step $t$ reaches is $s_{t+1}$. HER has three common strategies for picking $g'$ for the transition at step $t$:

| Strategy | New goal for step $t$ | Copies per step |
|---|---|---|
| `final` | the last state $s_T$ | 1 |
| `future` | $s_j$ with $j$ drawn uniformly from $t+1, \dots, T$ | $k$ |
| `episode` | $s_j$ with $j$ drawn uniformly from $1, \dots, T$ | $k$ |

The `future` strategy draws only from the steps after $t$, so for every relabeled transition, the goal is a state that the agent reached after that step. Some of those relabeled goals make the transition itself a success (when $j = t + 1$), and the others make it a step on the way. Andrychowicz and others (2017) found `future` the best of the three, and Figure 2 shows the same order on bit-flipping.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Greedy success rate on 12-bit bit-flipping against training episodes for four relabeling strategies">
<line x1="70.0" y1="250.0" x2="540.0" y2="250.0" class="stroke-grid"/>
<text x="62.0" y="254.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="197.5" x2="540.0" y2="197.5" class="stroke-grid"/>
<text x="62.0" y="201.5" class="fig-small" text-anchor="end">0.25</text>
<line x1="70.0" y1="145.0" x2="540.0" y2="145.0" class="stroke-grid"/>
<text x="62.0" y="149.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="70.0" y1="92.5" x2="540.0" y2="92.5" class="stroke-grid"/>
<text x="62.0" y="96.5" class="fig-small" text-anchor="end">0.75</text>
<line x1="70.0" y1="40.0" x2="540.0" y2="40.0" class="stroke-grid"/>
<text x="62.0" y="44.0" class="fig-small" text-anchor="end">1</text>
<text x="70.0" y="270.0" class="fig-small" text-anchor="middle">0</text>
<text x="164.0" y="270.0" class="fig-small" text-anchor="middle">100</text>
<text x="258.0" y="270.0" class="fig-small" text-anchor="middle">200</text>
<text x="352.0" y="270.0" class="fig-small" text-anchor="middle">300</text>
<text x="446.0" y="270.0" class="fig-small" text-anchor="middle">400</text>
<text x="540.0" y="270.0" class="fig-small" text-anchor="middle">500</text>
<text x="305.0" y="292.0" class="fig-small" text-anchor="middle">training episodes</text>
<polyline points="70.0,250.0 117.0,250.0 164.0,250.0 211.0,249.3 258.0,250.0 305.0,248.6 352.0,250.0 399.0,250.0 446.0,250.0 493.0,250.0 540.0,250.0" class="stroke-muted" fill="none"/>
<line x1="330" y1="56" x2="360" y2="56" class="stroke-muted"/>
<text x="366" y="60" class="fig-small" text-anchor="start">no relabeling: 0.00 at episode 500</text>
<polyline points="70.0,250.0 117.0,250.0 164.0,250.0 211.0,248.6 258.0,249.3 305.0,240.2 352.0,208.0 399.0,184.9 446.0,132.4 493.0,94.6 540.0,83.4" class="stroke-a" fill="none"/>
<line x1="330" y1="74" x2="360" y2="74" class="stroke-a"/>
<text x="366" y="78" class="fig-small" text-anchor="start">final: 0.79 at episode 500</text>
<polyline points="70.0,250.0 117.0,250.0 164.0,249.3 211.0,244.4 258.0,227.6 305.0,189.8 352.0,148.5 399.0,112.8 446.0,91.8 493.0,74.3 540.0,68.0" class="stroke-c" fill="none"/>
<line x1="330" y1="92" x2="360" y2="92" class="stroke-c"/>
<text x="366" y="96" class="fig-small" text-anchor="start">episode: 0.87 at episode 500</text>
<polyline points="70.0,250.0 117.0,249.3 164.0,248.6 211.0,218.5 258.0,178.6 305.0,119.1 352.0,89.7 399.0,70.1 446.0,68.7 493.0,48.4 540.0,51.2" class="stroke-b" fill="none"/>
<line x1="330" y1="110" x2="360" y2="110" class="stroke-b"/>
<text x="366" y="114" class="fig-small" text-anchor="start">future: 0.95 at episode 500</text>
</svg>
Figure 2. Greedy success rate on 12-bit bit-flipping, mean of three seeds. With no relabeling the agent never learns. All three strategies learn, and future relabeling learns fastest.
```

> [!KEY]
> A relabeled transition changes only the goal and the reward. The state, action, and next state are unchanged, and the reward is recomputed with `goal_reward(s2, g)`. Keeping the old reward with the new goal marks a success as a failure.

## Why relabeling works for Q-learning and not for a policy gradient

The Q-learning target for a goal-conditioned Q is

$$
y = r(s', g) + \gamma \, (1 - \text{done}) \max_{a'} Q(s', a', g)
$$

Read it as: the reward for landing on $s'$ when the goal is $g$, plus the discounted best value from $s'$ with the same goal. Every term on the right depends on the transition $(s, a, s')$ and on $g$, and on nothing else. The dynamics do not depend on $g$, and Q-learning is off-policy. So the equation holds for *any* $g$, including one picked after the fact. The transition is data about the world, and the goal only chooses which question to ask of it. The loss is the same squared error as in DQN, where the action value comes from `q_net` and the target comes from a frozen copy:

```python
y = r + gamma * (1 - done) * target_net(torch.cat([s2, g], dim=1)).max(dim=1).values
loss = ((q_net(torch.cat([s, g], dim=1)).gather(1, a[:, None])[:, 0] - y.detach()) ** 2).mean()
```

A policy gradient estimates an expectation over actions that the *current* policy $\pi(a \mid s, g)$ chooses when it aims at $g$. An episode collected while aiming at $g$ holds actions chosen for $g$, and those actions are not what $\pi(\cdot \mid s, g')$ would choose for a different goal $g'$. Attaching $g'$ to the data breaks the match between the actions and the policy that is supposed to have taken them. You could correct it with importance weights, as in [Off-policy importance sampling](/lessons/015-off-policy-importance-sampling), but the weights are products of ratios over the whole episode and are close to 0 or enormous. Relabeling is a tool for off-policy value learning.

> [!WARNING]
> Relabeling assumes the goal can be changed without changing the dynamics, and that the achieved goal can be read from the state. With a stochastic environment the relabeled data still gives a valid Q-target. With a goal that you cannot compute from the state, you cannot recompute the reward, and relabeling stops.

## Goal-conditioned supervised learning

The value function does the heavy lifting in HER, and it is the part that is hardest to train. Ghosh and others (2019) ask whether you can skip it. Their method, goal-conditioned supervised learning (GCSL), uses relabeling and then trains the policy by plain supervised learning:

1. Collect episodes with the current policy, aiming at goals you sample.
2. For each step $t$, pick a later state $s_j$ with $j > t$ and treat $(s_t, s_j)$ as an input and $a_t$ as the label. The episode really did get from $s_t$ to $s_j$ by taking $a_t$ first.
3. Maximize $\log \pi(a_t \mid s_t, s_j)$ on those labels, and repeat.

This is behavior cloning on the agent's own relabeled successes (see [Imitation learning](/lessons/051-imitation-learning)). There is no reward and no bootstrapping. Ghosh and others show that the procedure optimizes a bound on the reinforcement learning objective. The cost is that the label is *an* action that led there, not the best one, so GCSL improves only as fast as the data does, and a noisy environment makes the labels noisier.

> [!RECAP]
> - The goal-conditioned reward is `goal_reward(achieved, goal)`: 0 where all bits match, -1 elsewhere.
> - Relabeling keeps `s, a, s2`, swaps in a goal that the episode reached, and recomputes `r` and `done`.
> - `future` draws from steps $t+1$ to $T$, `episode` from steps $1$ to $T$, and `final` uses $s_T$.
> - The goal-conditioned DQN target uses `(1 - done)` and carries no gradient through the target network.
> - GCSL's targets are `(s_t, s_j)` as input and `a_t` as the label, for $j > t$.

## Build it step by step

The checks test the functions in this order. The environment `BitFlipEnv`, the network `make_qnet`, and the helpers `collect_episode`, `to_batch`, and `success_rate` are in the starter file.

1. **`goal_reward(achieved, goal)`.** Compare along the last axis with `np.all(achieved == goal, axis=-1)`, then map `True` to 0.0 and `False` to -1.0 with `np.where`. It must accept a batch and a single goal.
2. **`relabel(episode, strategy, k, rng)`.** Return the $T$ original transitions first, then the relabeled copies step by step. The goal that step $t$ achieved is `states[t + 1]`. The most common mistake is an off-by-one in `rng.integers(t + 1, T + 1, size=k)`: starting at `t` lets a step use the state it began in as its goal, and starting at `t + 2` drops the step's own result.
3. **Recompute the reward.** For every relabeled transition, set `r = goal_reward(s2, g)` and `done = (r == 0)`. Copying the original `r` fails the check on the `final` strategy.
4. **`dqn_loss(q_net, target_net, batch, gamma)`.** Build the target under `torch.no_grad()` and multiply the bootstrap by `(1 - done)`. The check compares your gradient with finite differences.
5. **`train_gcdqn(n, strategy, seed, ...)`.** Collect an episode, extend a list with `relabel(...)`, then take `n` gradient steps on random batches. Strategy `"none"` gives plain goal-conditioned DQN.
6. **`gcsl_targets(episode, rng)`.** Return `(states[:T], goals, actions)`, with `goals[t]` drawn from `states[t + 1 : T + 1]`.

## Where this breaks

**Goals must be computable from states.** HER needs the achieved goal at every step and a reward function you can call for any goal. For a task described only by a human's judgment, there is no function to call.

**The goal space has to match what the agent can reach.** Relabeled goals are goals the agent has already reached, so they cover only the region it explores. If the real goals lie outside that region, the relabeled data does not help, and exploration bonuses from [Exploration beyond epsilon](/lessons/046-exploration-bonuses) have to push the region outward.

**Relabeling shifts the goal distribution.** The relabeled goals follow what the agent did, not what you will ask for at test time. A buffer that holds many relabeled goals and few real ones can leave the agent weak on the goals you care about, which is why HER keeps the original transitions too.

## Further reading

- [Andrychowicz and others (2017), Hindsight experience replay](https://arxiv.org/abs/1707.01495). HER with the `final`, `future`, and `episode` strategies, tested on robot-arm pushing, sliding, and pick-and-place with a binary reward.
- [Schaul, Horgan, Gregor, and Silver (2015), Universal value function approximators](https://proceedings.mlr.press/v37/schaul15.html). Value functions $V(s, g)$ and $Q(s, a, g)$ that generalize across goals.
- [Ghosh and others (2019), Learning to reach goals via iterated supervised learning](https://arxiv.org/abs/1912.06088). GCSL: relabel, then imitate, with a bound on the RL objective.
- [CS285 (Berkeley), Lecture 24: Multi-task RL](https://rail.eecs.berkeley.edu/deeprlcourse/). The multi-task lecture of the Deep Reinforcement Learning course. The schedule page does not name goal-conditioned RL or hindsight relabeling in the lecture title.
- [CS224R (Stanford), Lecture 12: Multi-Task and Goal-Conditioned RL](https://cs224r.stanford.edu/). The lecture on this chapter's topic, which lists the HER paper as optional reading.

## The exercise

You build hindsight relabeling and use it to train a goal-conditioned DQN on bit-flipping. Write `goal_reward`, then `relabel` with the `"none"`, `"final"`, `"future"`, and `"episode"` strategies, then `dqn_loss`, `train_gcdqn`, and `gcsl_targets`.

The checks verify `goal_reward` on a batch, and `relabel` on a hand-built three-step episode, where every relabeled goal and reward is exact. They draw `future` goals 40 times on an episode of distinct states and require every goal to come from a later step and every later step to appear. They verify `dqn_loss` against a hand-computed value and against finite differences, and require that no gradient reaches the target network. They check that `gcsl_targets` draws goals from later states and keeps the actions. Last, `train_gcdqn` on 12 bits must stay under a 0.15 success rate with no relabeling and exceed 0.7 with `future`. The reference solution scores 0.94 to 0.98 with `future` across ten seeds, and 0.00 without.
