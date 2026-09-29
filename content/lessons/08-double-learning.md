---
title: Maximisation bias and double learning
slug: 08-double-learning
part: "Part 2 — Learning from experience"
summary: Measure how a max over noisy estimates overshoots, fix it with double Q-learning, and average the target with Expected SARSA.
minutes: 50
gpu: false
objectives:
  - Explain why the max of unbiased noisy estimates overestimates the max of the true values, and compute the size of the bias.
  - Implement the double estimator and double Q-learning, and show that they remove the upward bias.
  - Implement the Expected SARSA target, and explain why it tolerates a step size of one in a deterministic world.
prereqs:
  - 02-multi-armed-bandits
  - 07-temporal-difference-learning
exercise: 08-double-learning
---

# Maximisation bias and double learning

Q-learning's target takes a max over estimated action values. That looks
harmless. If every estimate is right on average, surely their max is right on
average too? It isn't. Among several noisy estimates, the largest one is
usually the one whose noise happened to be positive. So the max of the
estimates overshoots the max of the true values, even when no single estimate
is biased.

Think of ten people guessing the weight of a pumpkin. Each guess is unbiased,
but if you always report the highest guess, you overestimate every time. This
is *maximisation bias*. In Q-learning it compounds, because each biased
target feeds the next update. It can steer an agent toward actions that only
look good because they're noisy.

In this lesson you measure the bias exactly, then remove it with *double
learning*: one estimate chooses the action and a second, independent estimate
scores it. You also build Expected SARSA, which replaces the sampled next
action with an average and removes a different source of noise.

## What you already know

- An estimate is *unbiased* if its average over many repetitions equals the
  true value.
- The Q-learning target is $r_t + \gamma \max_{a'} Q(s_{t+1}, a')$.
- The SARSA target is $r_t + \gamma Q(s_{t+1}, a_{t+1})$, using the action
  actually taken next.
- An $\varepsilon$-greedy policy gives every action $\varepsilon / A$ and the
  greedy action another $1 - \varepsilon$.
- On the cliff walk from
  [Temporal-difference learning](07-temporal-difference-learning), SARSA
  learned a safe path and Q-learning learned the edge.

## See the bias in a max

Suppose two actions both have true value 0. You estimate each one from a
single noisy sample, $X_1$ and $X_2$, each drawn from a standard normal
distribution. Each estimate is unbiased. But the max of the two is positive
more often than not, because it's negative only when both are. The expected
max is

$$
\mathbb{E}\bigl[\max(X_1, X_2)\bigr] = \frac{1}{\sqrt{\pi}} \approx 0.564
$$

Read it as: two unbiased guesses at a value of 0, and their max averages
0.564. That's more than half a standard deviation of bias from nowhere.

The general statement is an inequality. For any estimates $\hat Q(a)$,

$$
\mathbb{E}\Bigl[\max_a \hat Q(a)\Bigr] \ge \max_a \mathbb{E}\bigl[\hat Q(a)\bigr]
$$

Read it as: the average of the max is at least the max of the averages. It
holds because the max function is *convex*: the max of an average of vectors
is at most the average of their maxes. Jensen's inequality turns that into
the statement about expectations. The gap grows
with the number of actions and with the noise. With 10 actions, all worth 0,
each estimated from a standard normal sample, the expected max is 1.539. If
each estimate averages 5 samples, its standard deviation shrinks by
$\sqrt 5$, and the bias shrinks to $1.539 / \sqrt 5 = 0.688$.

In numpy, draw a matrix with one row per trial and one column per action, and
take the max along the last axis:

```python
X = rng.standard_normal((n_trials, n_samples, n_actions)).mean(axis=1)
bias = X.max(axis=-1).mean()          # about 0.688 for 5 samples and 10 actions
```

> [!KEY]
> The max of unbiased noisy estimates is biased upward. The bias grows with
> the number of actions and the size of the noise.

## Separate choosing from scoring

The bias comes from using the same noise twice. The estimate $X$ picks the
action that looks best, and then that same $X$ reports how good it is. The
action was picked partly because its noise was high, so its reported value is
high too.

The *double estimator* keeps two independent sets of estimates, $X$ and $Y$.
It uses one to choose and the other to score:

$$
\hat a = \operatorname*{arg\,max}_a X(a), \qquad \text{estimate} = Y(\hat a)
$$

Read it as: let $X$ pick the action that looks best, and then ask $Y$, which
never saw $X$'s noise, how good that action is. Given the choice $\hat a$,
$Y(\hat a)$ is an unbiased estimate of that action's true value. So the double
estimate averages the true values of the actions $X$ tends to pick. That
average can't exceed the best true value.

```python
pick = np.argmax(X, axis=-1)
double = np.take_along_axis(Y, pick[..., None], axis=-1)[..., 0]
```

When all true values are equal, the double estimate is exactly unbiased. When
they differ, it underestimates, because $X$ sometimes picks a worse action.
Take true values $(0, 0.5)$ with unit noise. $X$ picks the better action with
probability $\Phi(0.5 / \sqrt 2) = 0.638$, where $\Phi$ is the standard normal
cumulative distribution function. So the double estimate averages $0.638
\times 0.5 = 0.319$, below the true max of 0.5. The single estimate averages
0.849, above it.

Here's a worked example with three actions. The estimates are $X = (1, 3, 2)$
and $Y = (0.5, 1, 4)$:

- The single estimate is $\max X = 3$.
- $X$ picks action 1, and $Y$ scores it at 1, so the double estimate is 1.
- Swap the roles: $Y$ picks action 2, and $X$ scores it at 2.

## Watch the bias steer an agent

Sutton and Barto's Example 6.7 turns the bias into bad behaviour. The episode
starts in state $A$, which has two actions. *Right* ends the episode with
reward 0. *Left* moves to state $B$ with reward 0. State $B$ has 10 actions,
and each one ends the episode with a reward drawn from a normal distribution
with mean $-0.1$ and standard deviation 1. Going left is worth $-0.1$, so
right is optimal.

Q-learning's update for $Q(A, \text{left})$ uses $\max_b Q(B, b)$. After a few
visits to $B$, some of the 10 estimates have seen lucky positive rewards, so
the max is positive. $Q(A, \text{left})$ climbs above $Q(A, \text{right}) =
0$, and the agent goes left. It keeps going left until the $B$ estimates
average out enough noise, which takes many episodes.

With $\alpha = 0.1$, $\varepsilon = 0.1$, and $\gamma = 1$, averaged over 100
seeded runs:

| Episodes | Q-learning, fraction going left | Double Q-learning, fraction going left |
|---|---|---|
| 1 to 30 | 0.84 | 0.38 |
| 181 to 200 | 0.17 | 0.08 |

A perfect $\varepsilon$-greedy agent still goes left with probability
$\varepsilon / 2 = 0.05$, because of random exploration. Double Q-learning
gets close to that floor. Q-learning is still well above it after 200
episodes.

## Double Q-learning

Double Q-learning (van Hasselt, 2010) applies the double estimator inside
Q-learning. It keeps two tables, $Q_1$ and $Q_2$. After each step it flips a
fair coin. On heads it updates $Q_1$:

$$
Q_1(s_t, a_t) \leftarrow Q_1(s_t, a_t) + \alpha \Bigl[r_t + \gamma\, Q_2\bigl(s_{t+1}, \operatorname*{arg\,max}_{a'} Q_1(s_{t+1}, a')\bigr) - Q_1(s_t, a_t)\Bigr]
$$

Read it as: $Q_1$ picks the best next action, $Q_2$ says what it's worth, and
$Q_1(s_t, a_t)$ moves toward that target. On tails, swap the roles of the two
tables. Each table learns from about half the steps, so the two sets of noise
stay independent. The agent behaves $\varepsilon$-greedily on $Q_1 + Q_2$,
which uses all the data.

```python
if rng.random() < 0.5:
    target = r + gamma * Q2[s2][np.argmax(Q1[s2])] * (not done)
    Q1[s][a] += alpha * (target - Q1[s][a])
else:
    target = r + gamma * Q1[s2][np.argmax(Q2[s2])] * (not done)
    Q2[s][a] += alpha * (target - Q2[s][a])
```

Here's a worked example with $r = -1$ and $\gamma = 0.9$. In the next state,
$Q_1 = (1, 3)$ and $Q_2 = (4, 0)$:

- Updating $Q_1$: $Q_1$ picks action 1, and $Q_2$ scores it at 0. The target
  is $-1 + 0.9 \times 0 = -1$.
- Updating $Q_2$: $Q_2$ picks action 0, and $Q_1$ scores it at 1. The target
  is $-1 + 0.9 \times 1 = -0.1$.
- Plain Q-learning on $Q_1$ would use $\max Q_1 = 3$, a target of $1.7$.

> [!WARNING]
> Using $\max_{a'} Q_2(s_{t+1}, a')$ as the target is not double Q-learning.
> That lets $Q_2$ both choose and score, which is the single estimator again,
> with all its bias.

> [!KEY]
> Double Q-learning chooses the next action with one table and scores it with
> the other. The noise that made an action look best never also reports its
> value.

## Average over the next action with Expected SARSA

SARSA's target has noise from two places: the environment's randomness, and
the random choice of the next action $a_{t+1}$. Expected SARSA removes the
second. Instead of the one next action you happen to sample, it uses the
average over all next actions, weighted by the policy:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \Bigl[r_t + \gamma \sum_{a'} \pi(a' \mid s_{t+1})\, Q(s_{t+1}, a') - Q(s_t, a_t)\Bigr]
$$

Read it as: the target is the reward plus the discounted value you'd expect
in the next state, if you picked the next action with your policy. For an
$\varepsilon$-greedy policy, the weights are $\varepsilon / A$ on every action
plus $1 - \varepsilon$ split among the greedy ones. In numpy, the sum is a dot
product:

```python
probs = epsilon / A + (1 - epsilon) * best / best.sum()   # best = q_next == q_next.max()
target = r + gamma * probs @ q_next * (not terminated)
```

Here's a worked example with $Q(s, a) = 0$, $r = -1$, $\gamma = 1$, $\alpha =
0.5$, and $Q(s', \cdot) = (2, 5)$. With $\varepsilon = 0.2$, the policy's
probabilities are $(0.1, 0.9)$. The target is $-1 + 0.1 \times 2 + 0.9 \times
5 = 3.7$, and the update gives $0 + 0.5 \times 3.7 = 1.85$. SARSA would give
0.5 or 2.0, depending on the sampled next action. Expected SARSA lands between
them, every time.

Two limits connect it to what you know. With $\varepsilon = 0$ the policy is
greedy, the weighted average is the max, and Expected SARSA is Q-learning.
With $\varepsilon = 1$ the average is uniform, and the target here is $-1 +
3.5 = 2.5$.

### Take a full step in a deterministic world

On the cliff, every move is deterministic. SARSA's only noise is its own next
action, and Expected SARSA has none. So Expected SARSA can use $\alpha = 1$,
replacing each estimate with its target outright. Van Seijen and colleagues
(2009) found that Expected SARSA does well at large step sizes on the cliff
walk, where SARSA does not. In the exercise, 300
episodes at $\alpha = 1$ and $\varepsilon = 0.1$ give Expected SARSA a safe
greedy path worth $-15$, and about $-22$ per episode over the last 100
training episodes. Plain SARSA at $\alpha = 1$ earns between about $-46$ and
$-167$, depending on the seed, because each sampled exploratory action
overwrites its estimate completely.

> [!RECAP]
> - `single_estimate` is `max` along the last axis. `double_estimate` indexes
>   one array at the other's `argmax`.
> - The $\varepsilon$-greedy probabilities are `epsilon / A` plus
>   `(1 - epsilon) * best / best.sum()`, where `best` marks the ties.
> - The Expected SARSA target is `r + gamma * probs @ q_next`, or `r` at the
>   end.
> - The double Q target is `r + gamma * q_evaluate[np.argmax(q_select)]`.
> - Double Q-learning flips a coin per step to pick which table to update,
>   and acts on `Q1[s] + Q2[s]`.

## Build it step by step

The checks test the functions in this order. `MaxBiasMDP` and
`epsilon_greedy_action` are already written. In `MaxBiasMDP`, a table is a
list of rows from `env.new_q()`, so write `Q[s][a]` rather than `Q[s, a]`.

1. **`single_estimate(means)`.** Takes `(..., A)` and returns `(...)`. One
   call to `np.max` with `axis=-1`.
2. **`double_estimate(means_a, means_b)`.** Same shapes. Take
   `np.argmax(means_a, axis=-1)` and gather from `means_b` with
   `np.take_along_axis`. Common mistake: taking the max of `means_b`, which
   is the single estimator again.
3. **`epsilon_greedy_probs(q_row, epsilon)`.** Returns `(A,)` summing to one.
   Build a boolean mask of the ties and split $1 - \varepsilon$ over it.
   Common mistake: putting all of $1 - \varepsilon$ on the first tie, which
   no longer matches a policy that breaks ties at random.
4. **`expected_sarsa_target(q_next, r, terminated, gamma, epsilon)`.** Returns
   a float. Dot the probabilities with `q_next`. Common mistake: forgetting
   that a terminal step's target is `r` alone.
5. **`double_q_target(q_select, q_evaluate, r, terminated, gamma)`.** Returns
   a float. One `np.argmax`, one index.
6. **`q_learning(env, n_episodes, alpha, epsilon, seed)`.** Returns a boolean
   `(n_episodes,)` array, `True` where the episode went left from $A$. Record
   the action taken in `env.A` before stepping.
7. **`double_q_learning(env, n_episodes, alpha, epsilon, seed)`.** Same
   output. Act on `Q1[s] + Q2[s]`, then flip the coin with `rng.random() <
   0.5`.
8. **`expected_sarsa(mdp, n_episodes, alpha, epsilon, seed)`.** Returns `(Q,
   returns)` on a `TabularMDP`, like `q_learning` from the previous lesson,
   with the Expected SARSA target in place of the max.

## Cost and scaling

The following table compares the cost of each target:

| Method | Target cost | Memory | Bias from the max |
|---|---|---|---|
| Q-learning | $O(A)$ for the max | One table | Upward |
| Double Q-learning | $O(A)$ for the argmax | Two tables | None, can be downward |
| SARSA | $O(1)$ | One table | None, on-policy |
| Expected SARSA | $O(A)$ for the dot product | One table | None, on-policy |

Double Q-learning doubles the memory and gives each table about half the
updates, so each table learns more slowly than one Q-learning table would.
Expected SARSA costs $O(A)$ per step instead of $O(1)$. That's cheap for a
few actions and matters when actions number in the thousands.

## Where this breaks

**Underestimation.** The double estimator can err low when true values
differ, as the $(0, 0.5)$ example shows. In most problems a small downward
bias does less harm than a compounding upward one, but it isn't free.

**Slower learning per table.** Each table in double Q-learning sees half the
data. On problems with little noise, plain Q-learning can learn faster.

**Bias with function approximation.** With a neural network, overestimation
comes from approximation error as well as reward noise, and it can be large.
Double DQN, covered in [Improving DQN](13-dqn-improvements), reuses the online
network to choose and the target network to score.

**Expected SARSA in stochastic worlds.** The step-size advantage comes from
removing action noise. When transitions or rewards are noisy, $\alpha = 1$
fails for Expected SARSA too, and it needs a smaller step.

## Further reading

- [van Hasselt, "Double Q-learning"
  (2010)](https://papers.nips.cc/paper_files/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf).
  The double estimator, its underestimation bound, and the algorithm.
- [van Hasselt, Guez, and Silver, "Deep reinforcement learning with double
  Q-learning" (2016)](https://arxiv.org/abs/1509.06461). The same idea inside
  DQN, which lesson 13 builds.
- [van Seijen, van Hasselt, Whiteson, and Wiering, "A theoretical and
  empirical analysis of Expected Sarsa"
  (2009)](https://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/vanseijenadprl09.pdf).
  Why Expected SARSA has lower variance, and its results on the cliff.
- [Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd edition
  (2018), Sections 6.6 and 6.7](http://incompleteideas.net/book/the-book-2nd.html).
  Expected SARSA and the maximisation-bias example.
- [Zhang, reference code for Sutton and Barto's
  figures](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter06/maximization_bias.py).
  The settings used here for Example 6.7.

## The exercise

You build the two estimators, the two targets, and three learners. Write
`single_estimate` and `double_estimate`, then `epsilon_greedy_probs`,
`expected_sarsa_target`, and `double_q_target`. Then write `q_learning` and
`double_q_learning` for `MaxBiasMDP`, and `expected_sarsa` for the cliff.

The checks verify both estimators by hand, and that over seeded trials the
single estimate matches the exact expected max, such as $1 / \sqrt\pi$ for two
actions, while the double estimate averages 0 and underestimates when true
values differ. They check the targets against the worked examples, including
3.7 and the Q-learning limit. Over 100 seeded runs, Q-learning must go left
from $A$ in about 0.84 of early episodes against about 0.38 for double
Q-learning, and double Q-learning must fall below 0.12 by episodes 181 to 200.
Last, Expected SARSA at $\alpha = 1$ must find a safe cliff path and out-earn
plain SARSA.
