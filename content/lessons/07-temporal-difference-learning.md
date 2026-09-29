---
title: Temporal-difference learning
slug: 07-temporal-difference-learning
part: "Part 2 — Learning from experience"
summary: Update values after every step toward a bootstrapped target, then learn the cliff walk with SARSA and Q-learning.
minutes: 55
gpu: false
objectives:
  - Update a value estimate toward a bootstrapped target after every step.
  - Show that batch TD(0) converges to the values of the maximum-likelihood model of the data.
  - Implement SARSA and Q-learning, and explain why they walk different paths on the cliff.
  - "Say what bootstrapping trades: lower variance for bias from the current estimate."
prereqs:
  - 05-dynamic-programming
  - 06-monte-carlo-methods
exercise: 07-temporal-difference-learning
---

# Temporal-difference learning

Monte Carlo methods wait. They can't update anything until an episode ends,
because they need the whole return. If episodes are long, learning is slow.
If episodes never end, Monte Carlo can't learn at all.

Temporal-difference learning, or *TD learning*, doesn't wait. After every
single step it compares two guesses: what it expected before the step, and
what it expects after seeing one reward and the next state. It nudges the
first guess toward the second. Think of a weather forecast for Saturday that
you revise every day of the week. You don't wait until Saturday to learn that
Monday's forecast was too sunny. Tuesday's forecast already tells you.

In this lesson you build TD(0) for predicting values, see what it converges
to on a fixed dataset, and then build two control methods, SARSA and
Q-learning. On the cliff walk, the two learn different paths from the same
experience. The reason why is the most useful idea in the lesson.

## What you already know

- The return $G_t = r_t + \gamma G_{t+1}$ is the immediate reward plus the
  discounted return from the next step.
- The Bellman equation says $V^\pi(s)$ is the expected value of $r_t + \gamma
  V^\pi(s_{t+1})$ when you start in $s$ and follow $\pi$.
- [Monte Carlo methods](06-monte-carlo-methods) move an estimate a fraction
  $\alpha$ toward an observed return.
- An $\varepsilon$-greedy policy picks a random action with probability
  $\varepsilon$ and the greedy action otherwise.
- `rlenvs.cliff_walking()` is Sutton and Barto's 4 by 12 cliff. The best path
  from the start is worth exactly $-13$.

## Update after every step

Monte Carlo moves $V(s_t)$ toward the full return $G_t$. The Bellman equation
says $G_t$ and $r_t + \gamma V(s_{t+1})$ have the same expected value when $V$
is right. TD(0) uses the second one as its target. It's available after one
step, because it replaces all the future rewards with the current estimate of
their total.

Using your own estimate inside your own target is called *bootstrapping*.
The TD(0) update is

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad
V(s_t) \leftarrow V(s_t) + \alpha\, \delta_t
$$

Read it as follows. The *TD error* $\delta_t$ is the reward you got, plus the
discounted value of where you landed, minus what you predicted. If it's
positive, the step went better than expected, so raise $V(s_t)$ by a fraction
$\alpha$ of the surprise. A terminal state has value zero, so when $s_{t+1}$
is terminal the target is $r_t$ alone.

```python
target = r + gamma * V[s2] * (not terminated)
V[s] += alpha * (target - V[s])
```

Here's a worked example with $V = (0, 0.5, 1.0)$, $\alpha = 0.1$, and $\gamma
= 0.9$. A step from state 1 pays 1 and lands in state 2:

- If state 2 isn't terminal, the target is $1 + 0.9 \times 1.0 = 1.9$, the
  error is $1.9 - 0.5 = 1.4$, and $V(1)$ becomes $0.5 + 0.14 = 0.64$.
- If state 2 is terminal, the target is $1$, the error is $0.5$, and $V(1)$
  becomes $0.55$.

### Bias against variance

The Monte Carlo target $G_t$ is unbiased, but it adds up every random reward
until the end, so it's noisy. The TD target depends on one reward and one
transition, so it's much less noisy. But it's biased whenever $V(s_{t+1})$ is
wrong, and early in learning it's always wrong. Bootstrapping trades variance
for bias.

The two are tied together exactly. If $V$ doesn't change during an episode,
the Monte Carlo error is the discounted sum of the TD errors along the way:

$$
G_t - V(s_t) = \sum_{k=t}^{T-1} \gamma^{k-t} \delta_k
$$

Read it as: each step's surprise adds up, discounted, into the total surprise
of the whole return. The terms $\gamma V(s_{k+1})$ cancel in pairs, leaving
only the rewards and $-V(s_t)$. TD spreads the Monte Carlo update over the
episode, one piece per step.

On the five-state random walk from the previous lesson, with every estimate
started at 0.5, TD(0) with $\alpha = 0.05$ reaches an RMS error of about 0.036
after 100 episodes, averaged over 40 runs. Constant-step Monte Carlo with
$\alpha = 0.02$ reaches about 0.08. On this problem, lower
variance wins.

> [!KEY]
> TD(0) updates toward $r_t + \gamma V(s_{t+1})$ after every step. It learns
> from incomplete episodes and has lower variance than Monte Carlo, at the
> cost of bias from its own current estimates.

## Learn from a fixed batch

Suppose you have only a fixed set of episodes, and you sweep them again and
again until the values stop changing. This is *batch* learning. In each
sweep, hold $V$ fixed, add up the update for every step, and then apply the
total. With a small enough $\alpha$, batch Monte Carlo and batch TD(0) each
converge, but to different answers.

Sutton and Barto's Example 6.4 makes the difference plain. There are eight
episodes over two states, $A$ and $B$:

- One episode goes $A \to B$ with reward 0, then ends from $B$ with reward 0.
- Six episodes start in $B$ and end with reward 1.
- One episode starts in $B$ and ends with reward 0.

Both methods agree that $V(B) = 6/8 = 0.75$. They disagree about $A$. The only
return ever seen from $A$ was 0, so batch Monte Carlo says $V(A) = 0$. Batch
TD says $V(A) = 0.75$, because $A$ always went to $B$ with reward 0, and $B$
is worth 0.75.

TD's answer comes from a model. Estimate each transition probability as a
fraction of observed transitions, and each reward as a mean. This is the
*maximum-likelihood model* of the data. Solve that model's Bellman equation
exactly, and you get the *certainty-equivalence* estimate. Batch TD(0)
converges to it. Batch Monte Carlo instead minimises squared error on the
returns it saw.

For the example, the model says $\hat P(B \mid A) = 1$ with reward 0, and $B$
ends with mean reward $6/8$. Solving $V(B) = 0.75$ and $V(A) = 0 + V(B)$ gives
$0.75$ for both. In numpy, count transitions, divide by row totals, and solve
the linear system on the states that were ever left:

```python
P_hat[seen] = counts[seen] / counts[seen].sum(axis=1, keepdims=True)
V[idx] = np.linalg.solve(np.eye(len(idx)) - gamma * P_hat[np.ix_(idx, idx)], r_hat[idx])
```

> [!KEY]
> On a fixed batch, TD(0) converges to the exact values of the
> maximum-likelihood model of the data. Monte Carlo converges to the average
> observed returns, and ignores the Markov structure.

## Control on-policy with SARSA

For control you need action values, for the same reason as in Monte Carlo
control: without a model, only $Q$ tells you which action is best. SARSA is
TD(0) for $Q$. Its update uses the five quantities $(s_t, a_t, r_t, s_{t+1},
a_{t+1})$, which give it its name:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \bigl[r_t + \gamma\, Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)\bigr]
$$

Read it as: the target is the reward plus the discounted value of the next
state and the action you will actually take there. SARSA picks $a_{t+1}$ with
its $\varepsilon$-greedy policy before updating, and then takes that same
action. SARSA is *on-policy*: it learns the value of the policy it follows,
exploration included.

```python
a2 = epsilon_greedy_action(Q[s2], epsilon, rng)
target = r + gamma * Q[s2, a2] * (not terminated)
Q[s, a] += alpha * (target - Q[s, a])
s, a = s2, a2
```

## Control off-policy with Q-learning

Q-learning changes one thing. Its target uses the best action in the next
state, whatever action it takes next:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \bigl[r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)\bigr]
$$

Read it as: the target is the reward plus the discounted value of acting
greedily from the next state on. Q-learning is *off-policy*: it behaves
$\varepsilon$-greedily, but it learns the value of the greedy policy. In code,
`Q[s2, a2]` becomes `Q[s2].max()`.

Here's a worked example with $Q(s, a) = 0$, $r = -1$, $\gamma = 1$, $\alpha =
0.5$, and next-state values $Q(s', \cdot) = (2, 5)$. Suppose the policy's next
action is action 0, the exploratory one:

- SARSA uses $Q(s', 0) = 2$. The update gives $0 + 0.5(-1 + 2 - 0) = 0.5$.
- Q-learning uses $\max = 5$. The update gives $0 + 0.5(-1 + 5 - 0) = 2.0$.

Tabular Q-learning converges to the optimal $Q^*$ with probability one under
two conditions. Every state-action pair must be tried infinitely often, and
the step sizes must shrink, with $\sum_t \alpha_t = \infty$ and $\sum_t
\alpha_t^2 < \infty$. With a constant $\alpha$, as in the exercise, the
estimates keep jittering, but the greedy policy often settles anyway.

> [!TIP]
> Always pick SARSA's next action before the update, and then take that same
> action. Choosing a second, fresh action to execute turns SARSA into a
> confused mix of two policies.

## Walk the cliff two ways

The cliff walk is a 4 by 12 grid. You start at the bottom left, and the goal
is the bottom right. The cells between them, along the bottom row, are the
cliff. Every step pays $-1$. Stepping into the cliff pays $-100$ and sends you
back to the start. The shortest path runs along the edge, right above the
cliff, in 13 steps worth $-13$.

Train both methods for 500 episodes with $\alpha = 0.1$ and $\varepsilon =
0.1$, and then look at their greedy policies:

- Q-learning's greedy policy walks the edge. Its exact value from the start is
  $-13$. Q-learning learns the value of acting greedily, and the greedy walker
  never slips.
- SARSA's greedy policy takes a detour one or two rows farther from the
  cliff, worth $-15$ or $-17$. SARSA learns the value of its $\varepsilon$-greedy policy. Next to
  the edge, that policy sometimes steps down into the cliff. Those falls are
  in SARSA's values, so the edge looks dangerous to it.

The ordering flips during training. The agents still explore, and Q-learning
explores right beside the cliff. In one seeded run, over the last 100 training
episodes, SARSA earns about $-24$ per episode and Q-learning about $-39$,
because Q-learning keeps falling off. Which answer is right depends on the question. If
exploration stops after training, Q-learning's path is better. If the agent
must keep exploring, SARSA's is.

> [!KEY]
> SARSA learns the value of the policy it follows, slips included, so it
> learns a safe path. Q-learning learns the value of the greedy policy, so it
> learns the optimal edge path and falls off it while exploring.

> [!WARNING]
> `rlenvs.policy_evaluation` returns `nan` for any state from which the
> greedy policy never ends, such as a rarely visited cell whose greedy action
> walks into a wall. Compare a value with `np.isclose`, never with
> `abs(x - y) > tol`, because every comparison with `nan` is false.

> [!RECAP]
> - The TD target is `r + gamma * V[s2] * (not terminated)`. Move `V[s]` a
>   fraction `alpha` toward it.
> - Batch TD(0) holds $V$ fixed within a sweep, sums the increments, and
>   converges to the certainty-equivalence values.
> - The maximum-likelihood model is transition counts divided by row totals,
>   and mean rewards. Solve it with `np.linalg.solve`.
> - SARSA's target uses `Q[s2, a2]` for the action it will take. Q-learning's
>   uses `Q[s2].max()`.
> - Break ties between greedy actions at random.

## Build it step by step

The checks test the functions in this order. `random_walk` is already
written.

1. **`td0_update(V, s, r, s2, terminated, alpha, gamma)`.** Updates `V[s]` in
   place and returns `V`. Common mistake: bootstrapping from a terminal
   state. Multiply `V[s2]` by `not terminated`.
2. **`certainty_equivalence(episodes, n_states, gamma)`.** Returns `V`, shape
   `(S,)`. Each episode is `(states, rewards)`, where `states` has one extra
   entry, the terminal state. Build a count matrix and a reward sum per state,
   normalise the rows that have data, and solve on those states only. Common
   mistake: solving the full system, which is singular because terminal rows
   are all zero.
3. **`batch_td0(episodes, n_states, gamma, alpha, n_sweeps)`.** Returns `V`.
   Inside each sweep, add every `alpha * delta` into a separate array, and add
   it to `V` after the sweep. Common mistake: updating `V` inside the sweep,
   which is online TD and settles near, but not exactly on, the batch answer.
4. **`td0_prediction(mdp, pi, n_episodes, alpha, seed, v_init=0.5)`.** Returns
   `V`. Start terminal states at 0 and others at `v_init`. Draw actions with
   `rng.choice(A, p=pi[s])` and steps with `mdp.sample(s, a, rng)`.
5. **`sarsa_update` and `q_learning_update`.** Each updates `Q[s, a]` in place
   and returns `Q`. They differ only in the bootstrap term.
6. **`epsilon_greedy_action(q_row, epsilon, rng)`.** Returns an `int`. Draw
   `rng.random()` to decide whether to explore, and pick among
   `np.flatnonzero(q_row == q_row.max())` to break ties. Common mistake:
   `q_row.argmax()`, which always picks the first tie. With $Q = 0$, that sends
   the cliff walker up into the wall.
7. **`q_learning(mdp, n_episodes, alpha, epsilon, seed)`.** Returns `(Q,
   returns)`, where `returns[i]` is the total reward of training episode `i`.
   Choose, step, update, repeat, until `mdp.terminal[s2]` or `mdp.max_steps`.
8. **`sarsa(mdp, n_episodes, alpha, epsilon, seed)`.** Same outputs. Choose
   the first action before the step loop, and carry `a2` into the next step.

## Cost and scaling

Each TD update touches one table entry and costs $O(1)$, or $O(A)$ for
Q-learning's max. Memory is one number per state or per state-action pair.
The following table compares the methods in this lesson:

| Method | Time per step | Memory | Must episodes end? |
|---|---|---|---|
| TD(0) | $O(1)$ | $S$ | No |
| Batch TD(0), per sweep | $O(\text{dataset steps})$ | $S$ plus the dataset | No |
| Certainty equivalence | $O(S^3)$ once | $S^2$ | No |
| SARSA | $O(A)$ to pick the action | $S \times A$ | No |
| Q-learning | $O(A)$ for the max | $S \times A$ | No |

Certainty equivalence is exact but needs a full model and a linear solve, so
it's cheap for 7 states and hopeless for a million. TD reaches the same
answer on a batch with memory that grows with $S$, not $S^2$.

## Where this breaks

**Slow propagation.** One TD update moves information back by one step. A
reward at the end of a 100-step corridor needs many episodes to reach the
start. [n-step returns and eligibility traces](09-n-step-and-eligibility-traces)
send it back faster.

**Overestimation.** Q-learning's max picks the action whose estimate is
luckiest, so noisy rewards make it overestimate. The next lesson,
[Maximisation bias and double learning](08-double-learning), measures this and
fixes it.

**Constant step sizes.** With a constant $\alpha$, values never stop moving in
a stochastic world. People decay $\alpha$ over time, or accept the jitter and
evaluate the greedy policy.

**Exploration that never happens.** With $Q = 0$ and ties broken by `argmax`,
a greedy agent can repeat one action forever. Random tie-breaking,
$\varepsilon$-greedy exploration, or optimistic starting values fix it.

**Large state spaces.** A table needs one entry per state. Part 3 replaces it
with a function approximator, and TD's bootstrapping then brings its own
risk of divergence.

## Further reading

- [Sutton, "Learning to predict by the methods of temporal differences"
  (1988)](https://doi.org/10.1007/BF00115009). The paper that introduced TD
  learning, including the batch convergence result.
- [Watkins and Dayan, "Q-learning" (1992)](https://doi.org/10.1007/BF00992698).
  The convergence proof for tabular Q-learning.
- [van Seijen, van Hasselt, Whiteson, and Wiering, "A theoretical and
  empirical analysis of Expected Sarsa"
  (2009)](https://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/vanseijenadprl09.pdf).
  A variant that averages over the next action, covered in the next lesson.
- [Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd edition
  (2018), Chapter 6](http://incompleteideas.net/book/the-book-2nd.html). The
  random walk, Example 6.4, and the cliff walk.

## The exercise

You build TD prediction and TD control. Write `td0_update`,
`certainty_equivalence`, `batch_td0`, and `td0_prediction` for prediction.
Then write `sarsa_update`, `q_learning_update`, `epsilon_greedy_action`,
`q_learning`, and `sarsa` for control on the cliff.

The checks verify both TD updates by hand, and that `certainty_equivalence`
and `batch_td0` give $V(A) = V(B) = 0.75$ on Example 6.4 and agree on
random-walk data. TD(0) must beat constant-step Monte Carlo on the random walk,
and `epsilon_greedy_action` must split ties evenly. On the cliff, Q-learning's
greedy value from the start must be exactly $-13$. SARSA's must be finite and
below $-13$ in two of three seeded runs, and SARSA must out-earn Q-learning
during training.
