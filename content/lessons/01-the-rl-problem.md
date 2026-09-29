---
title: Agents, rewards, and returns
slug: 01-the-rl-problem
part: "Part 1 — The reinforcement learning problem"
summary: Meet the agent-environment loop, turn rewards into discounted returns, and estimate a policy's value by averaging episodes.
minutes: 45
gpu: false
objectives:
  - Say what reinforcement learning is, and how its feedback differs from supervised learning's labels.
  - Run the agent-environment loop, and tell termination from truncation.
  - Compute discounted returns for every step of an episode in one backward pass.
  - Estimate a policy's value by averaging returns, and report the standard error of that estimate.
exercise: 01-the-rl-problem
---

# Agents, rewards, and returns

*Reinforcement learning* is learning to act by trial and error. A program,
called the *agent*, takes an action, and the world it acts in, called the
*environment*, answers with a number called a *reward*. The agent's job is to
find a way of acting that earns as much reward as possible over time. Nobody
tells it which action was right. It has to find out by trying.

That is the difference from supervised learning, which you already know. A
supervised model gets a labelled answer for every example, and its loss says
exactly how far off it was. A reinforcement learning agent gets only a score
for what it did, never the answer it should have given. Its own choices decide
which situations it sees next, so it gathers its own training data. And a
reward can arrive long after the action that earned it, like a chess game won
thirty moves after the decisive move.

This course builds reinforcement learning from the ground up. Part 1 starts
with one decision under uncertainty, the multi-armed bandit, then adds states
and plans with a known model. Parts 2 to 7 learn from experience, scale up
with neural networks, optimise policies directly, and handle continuous
control, exploration, offline data, and many agents. Parts 8 to 10 use the
same ideas to post-train language models, image generators, speech models,
and robot policies, and then look at safety and open research problems.

Every lesson works the same way. You read the ideas, with a small example
worked by hand, and then build them in the exercise pane next to the lesson.
The pane holds a starter file whose functions raise `NotImplementedError`.
**Run** executes the file's example at the bottom, and **Check** runs a set
of checks that compare your functions against exact answers: hand-computed
values, closed forms, and the true value of a policy. This first lesson sets
up the vocabulary for everything that follows, and you build the loop that
every later method runs inside.

## What you already know

- A supervised model learns from pairs of inputs and correct outputs.
- An expected value is the long-run average of a random quantity.
- The mean of $n$ samples estimates an expected value, with a standard error
  of about the sample standard deviation divided by $\sqrt{n}$.
- `np.random.default_rng(seed)` makes a reproducible random number generator.

## The agent-environment loop

Everything in reinforcement learning happens inside one loop. At each time
step $t$:

1. The agent observes the environment's *state* $s_t$, a description of the
   situation it's in.
2. The agent picks an *action* $a_t$.
3. The environment returns a reward $r_t$ and the next state $s_{t+1}$.

Then the loop repeats from $s_{t+1}$. Throughout the course, $r_t$ is the
reward for taking $a_t$ in $s_t$. Some books, including Sutton and Barto, call
the same number $R_{t+1}$. The meaning is identical, and only the label
differs.

The agent's rule for picking actions is its *policy*, written
$\pi(a \mid s)$: the probability of taking action $a$ in state $s$. A policy
can be deterministic, always picking the same action in a state, or random.
In code, a policy for a small problem is an array `pi` of shape
`(n_states, n_actions)` whose rows each sum to one.

Many tasks come in *episodes*: runs that start in a start state and end in a
*terminal state*, such as a won game or a reached goal. One episode is a
sequence $s_0, a_0, r_0, s_1, a_1, r_1, \dots, s_T$, where $s_T$ is terminal
and $r_{T-1}$ is the last reward.

The course's environments use a slimmed-down version of the interface from
the Gymnasium library. The following code runs one episode:

```python
s = mdp.reset(seed=0)
while True:
    a = rng.choice(mdp.n_actions, p=pi[s])     # sample from the policy
    s2, r, terminated, truncated = mdp.step(a)
    # record (s, a, r) here: all three belong to step t
    s = s2
    if terminated or truncated:
        break
```

### A room to walk in

The exercise uses a 3 by 3 room with a pillar in the middle. You start in the
top-left corner `S`, and the episode ends when you reach the goal `G` in the
bottom-right corner:

```text
S . .
. # .
. . G
```

The actions are up, right, down, and left. Walking into a wall or the pillar
leaves you where you are. Every step pays a reward of $-1$, so the reward
punishes wasted time. The room has eight states, one per open cell.

> [!KEY]
> The reward says what you want, not how to get it. Here, $-1$ per step means
> "reach the goal quickly", and the agent has to discover the route.

### Termination and truncation

An episode can stop for two different reasons, and later lessons depend on
telling them apart. It *terminates* when it reaches a terminal state, where
nothing more can happen. It's *truncated* when a time limit cuts it off, even
though the task could have continued. The room's environment truncates after
500 steps by default. A truncated episode stopped early, so its last state
still has a future that the recorded rewards don't include.

## Returns: the total reward from now on

A single reward is short-sighted. What the agent cares about is the total it
collects from a step onward, called the *return*. Rewards far in the future
usually count for less, like money promised years from now. A *discount
factor* $\gamma$ (gamma), between 0 and 1, shrinks each later reward by one
more factor of $\gamma$:

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots + \gamma^{T-1-t} r_{T-1}
$$

Read it as: the return from step $t$ is this step's reward at full weight, the
next reward at weight $\gamma$, the one after at $\gamma^2$, and so on to the
end of the episode. With $\gamma = 0$ the agent cares only about the next
reward. With $\gamma = 1$ every reward counts the same. With $\gamma = 0.9$ a
reward ten steps away counts $0.9^{10} = 0.35$ as much as one now, so
$1 / (1 - \gamma) = 10$ steps is a rough planning horizon.

Discounting also keeps the sum finite in tasks that never end, because
$\sum_k \gamma^k$ is $1/(1-\gamma)$ when $\gamma < 1$.

### Compute every return in one backward pass

Pull the first reward out of the sum, and what's left is $\gamma$ times the
next return:

$$
G_t = r_t + \gamma G_{t+1}, \qquad G_T = 0
$$

Read it as: the return from now is this reward plus the discounted return from
the next step. After the last step there are no rewards, so the return is 0.
That turns a sum per step into one loop that runs backward from the end:

```python
running = 0.0
for t in reversed(range(len(rewards))):
    running = rewards[t] + gamma * running
    returns[t] = running
```

Recomputing the full sum at every step costs time proportional to $T^2$. The
backward loop costs time proportional to $T$.

### Worked example

Take rewards $r_0 = 1$, $r_1 = 0$, and $r_2 = 2$, then termination, with
$\gamma = 0.9$. Work backward:

| Step $t$ | Reward $r_t$ | $G_t = r_t + 0.9\, G_{t+1}$ |
|---|---|---|
| 2 | 2 | $2 + 0.9 \times 0 = 2$ |
| 1 | 0 | $0 + 0.9 \times 2 = 1.8$ |
| 0 | 1 | $1 + 0.9 \times 1.8 = 2.62$ |

The direct sum agrees: $1 + 0.9 \times 0 + 0.81 \times 2 = 2.62$.

In the room, the shortest route takes four steps: right, right, down, down.
Its return from the start is
$-(1 + 0.9 + 0.81 + 0.729) = -3.439$.

## Values: how good a state is under a policy

A return depends on luck: which actions the policy happened to pick, and how
the environment happened to respond. The *value* of a state averages that luck
away. The value of state $s$ under policy $\pi$ is the expected return when you
start in $s$ and follow $\pi$:

$$
V^\pi(s) = \mathbb{E}_\pi\left[\, G_t \mid s_t = s \,\right]
$$

Read it as: over all the episodes that could unfold from $s$ when you act by
$\pi$, the average return. The subscript $\pi$ on the expectation says whose
actions generate those episodes. Values are the central object of the course.
Almost every method estimates them, and then uses them to act better.

### Estimate a value by averaging

The simplest estimate follows from the definition. Run $n$ episodes from the
start state, compute each one's return $G_0$, and average. This is called a
*Monte Carlo* estimate, after the casino, because it uses random sampling:

$$
\hat{V} = \frac{1}{n} \sum_{i=1}^{n} G_0^{(i)}, \qquad
\text{SE} = \frac{\text{sd}\left(G_0^{(1)}, \dots, G_0^{(n)}\right)}{\sqrt{n}}
$$

Read it as: the estimate is the average of the $n$ episode returns, where
$G_0^{(i)}$ is the return of episode $i$. Its *standard error* SE is the
sample standard deviation of those returns divided by $\sqrt{n}$. For large
$n$, the true value lies within two standard errors of the estimate roughly 95
percent of the time. Quadruple the episodes and the error halves.

```python
g0 = np.array([discounted_returns(run_episode(mdp, pi, rng)[2], gamma)[0]
               for _ in range(n)])
estimate, se = g0.mean(), g0.std(ddof=1) / np.sqrt(n)
```

### Worked example

Take the policy that picks each of the four actions with probability $1/4$,
called the *uniform random policy*, in the room with $\gamma = 0.9$. It bumps
into walls a lot. Its exact expected episode length from the start is 32
steps, and its exact value is $V^\pi(\text{start}) = -8.551$. The shortest
route is worth $-3.439$, so a better policy exists, which is what the rest of
the course learns to find.

The exercise's reference solution, averaging more and more episodes, gives:

| Episodes | Estimate | Standard error |
|---|---|---|
| 100 | $-8.631$ | 0.174 |
| 400 | $-8.610$ | 0.082 |
| 1600 | $-8.551$ | 0.040 |

Each estimate sits within one standard error of the truth, and the
error halves each time the episode count quadruples. The exact value comes
from `rlenvs.policy_evaluation`, which solves a linear system that lesson 04,
"Markov decision processes", derives.

> [!KEY]
> A value is an expected return. You can estimate it by averaging the returns
> of many episodes, and the standard error tells you how far to trust the
> average.

> [!RECAP]
> - The loop is observe $s_t$, act $a_t$, receive $r_t$ and $s_{t+1}$, and
>   repeat until terminated or truncated.
> - A policy `pi[s]` is a row of action probabilities.
> - $G_t = r_t + \gamma G_{t+1}$ with $G_T = 0$ gives every return in one
>   backward pass.
> - $V^\pi(s)$ is the expected return from $s$, and the mean of many $G_0$
>   values estimates it.
> - The standard error is `np.std(g0, ddof=1) / np.sqrt(n)`.

## Build it step by step

The checks test the functions in this order.

1. **`discounted_returns(rewards, gamma)`.** Takes a `(T,)` array of rewards
   and returns a `(T,)` array of returns. Loop backward with a running total,
   as in the preceding code. Common mistake: discounting the first reward,
   which starts every return at $\gamma r_t$ instead of $r_t$. The checks use
   the worked example and compare against the direct sum for four values of
   $\gamma$.
2. **`run_episode(mdp, pi, rng)`.** Reset with
   `mdp.reset(seed=int(rng.integers(2**31)))`, so the whole episode follows
   from `rng`. Draw each action with `rng.choice(mdp.n_actions, p=pi[s])`.
   Return `(states, actions, rewards, terminated)`, where `states[t]` is the
   state *before* step $t$. Common mistake: appending the next state `s2`
   instead of `s`, which shifts every state by one step. The checks walk the
   shortest route, cut an episode off with a five-step time limit, and count
   how often each action is drawn.
3. **`mc_value_estimate(mdp, pi, n_episodes, seed)`.** Make one generator
   from `seed` and reuse it for every episode. Return the mean of the $G_0$
   values and their standard error, using `ddof=1`. Common mistake: making a
   fresh generator with the same seed for every episode, which repeats one
   episode `n` times and reports a standard error of zero. The checks compare
   against the exact value within four standard errors, and check that the
   error halves when the episode count quadruples.

## Cost and scaling

A Monte Carlo estimate costs one environment step per reward, so its running
time is the number of episodes times their average length.

| Quantity | Grows with |
|---|---|
| Time for `discounted_returns` | Episode length $T$ |
| Time for one Monte Carlo estimate | Episodes $n$ times average length |
| Standard error | $1 / \sqrt{n}$ |
| Memory | One episode's rewards at a time |

The square root is the expensive part. Each extra digit of accuracy costs 100
times as many episodes. In the room, 1000 episodes of about 32 steps take
under half a second in plain Python. A task with thousand-step episodes and
noisy rewards needs far more. Later lessons cut this cost by *bootstrapping*,
updating from estimated values of later states instead of waiting for the
episode to end.

## Where this breaks

**Episodes that never end.** A Monte Carlo estimate waits for the episode to
finish. A policy that walks into a wall forever never finishes, so the loop
needs a time limit. Truncating then biases the estimate, because the missing
rewards are dropped. People set the limit far past the typical episode length,
or add an estimate of the remaining value.

**High variance.** Long episodes with random rewards make $G_0$ swing widely,
so the standard error stays large. Discounting with a smaller $\gamma$ helps by
shortening the effective horizon, at the cost of changing what you measure.

**The reward says the wrong thing.** The agent maximises the reward you wrote,
not the one you meant. A reward for touching a ball can teach a robot to
vibrate against it. Lesson 30, "Reward design and shaping", returns to this.

**Using the same seed twice.** Reusing one seed for every episode repeats the
same episode. The average then looks precise and is wrong, with a standard
error of zero.

## Further reading

- [Sutton and Barto, *Reinforcement Learning: An Introduction*, Chapters 1 and 3](http://incompleteideas.net/book/the-book-2nd.html)
  introduce the agent-environment loop, returns, and values. The book is free
  online and underlies Parts 1 to 3 of this course.
- [Mnih and colleagues, "Human-level control through deep reinforcement learning" (2015)](https://doi.org/10.1038/nature14236)
  trains one network design, separately on each of 49 Atari games, from
  pixels and score alone.
- [Ouyang and colleagues, "Training language models to follow instructions with human feedback" (2022)](https://arxiv.org/abs/2203.02155)
  uses reinforcement learning from human preferences to post-train a language
  model, the recipe behind many chat assistants.
- [Black and colleagues, "Training diffusion models with reinforcement learning" (2023)](https://arxiv.org/abs/2305.13301)
  treats each denoising step of an image generator as an action.
- [DeepSeek-AI, "DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning" (2025)](https://arxiv.org/abs/2501.12948)
  trains long step-by-step reasoning with rewards that a program can verify.

## The exercise

You build the loop and the first value estimate. Write
`discounted_returns(rewards, gamma)` for the backward recursion,
`run_episode(mdp, pi, rng)` for one episode of the agent-environment loop, and
`mc_value_estimate(mdp, pi, n_episodes, seed)` for the Monte Carlo estimate
and its standard error. The environment is the 3 by 3 room from this lesson,
built with `rlenvs.gridworld`.

The checks verify the worked returns $(2.62, 1.8, 2)$ and the direct sum at
four discount factors. They check that `run_episode` walks the four-step route
exactly, reproduces an episode from a seeded generator, stops at a time limit
with `terminated` set to `False`, and draws actions with the policy's
probabilities. Last, they check that `mc_value_estimate` returns exactly
$-3.439$ with zero error on the shortest route, lands within four standard
errors of the random policy's exact value $-8.551$, and that its standard
error halves when the episode count quadruples.
