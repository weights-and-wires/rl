---
title: Offline reinforcement learning
slug: 29-offline-rl
part: "Part 6 — Models, exploration, and fixed data"
summary: See Q-learning on a fixed dataset trust actions it never saw, then fix it with CQL's penalty and IQL's expectile value.
minutes: 60
gpu: false
objectives:
  - Explain extrapolation error, and measure it exactly on a table where unseen actions keep their starting values.
  - Compute the CQL penalty and its gradient, and show that it sinks the values of actions the data never took.
  - Implement IQL's expectile loss, check its gradient, and show that tau = 0.5 gives the mean and tau near 1 the best in-data action.
  - Extract a policy by advantage weighting that puts no probability on unseen actions.
prereqs:
  - 07-temporal-difference-learning
  - 08-double-learning
  - 27-imitation-learning
exercise: 29-offline-rl
---

# Offline reinforcement learning

Reinforcement learning usually learns by trying things, and each mistake
teaches it something. In medicine, driving, or a recommendation system with
real users, you can't afford those mistakes. What you often do have is a log: months of decisions that
someone else made, with their outcomes.

*Offline reinforcement learning* learns a policy from such a fixed dataset,
with no further contact with the environment. It sounds like Q-learning on a
replay buffer that never gets new data. It isn't that easy. Off-policy
methods quietly depend on fresh data to correct their mistakes. Take the fresh
data away, and they confidently pick actions that nobody ever tried.

This lesson makes that failure exact on a table. A cautious demonstrator walks
the cliff world without ever stepping toward the edge, and Q-learning on its
data concludes that stepping off the cliff is the best move. Then you build two
fixes: conservative Q-learning (CQL) and implicit Q-learning (IQL).

## What you already know

- Q-learning moves $Q(s, a)$ toward $r + \gamma \max_{a'} Q(s', a')$, using
  whatever transitions it has (lesson 7).
- A max over noisy estimates is biased upward, because it picks the estimates
  that happen to be too high (lesson 8).
- Behaviour cloning copies the demonstrator's actions by supervised learning
  (lesson 27).
- The cliff world pays $-1$ per step and $-100$ for stepping into the cliff,
  which sends you back to the start.
- Policy evaluation computes $V^\pi$ exactly on a tabular MDP.

## Learn from a log you can't extend

The *dataset* $\mathcal{D}$ is a list of transitions $(s, a, r, s', d)$, where
$d$ is 1 when $s'$ is terminal. Some *behaviour policy* $\pi_\beta$ produced
them. You don't need to know $\pi_\beta$. Counting how often each action
appears in each state gives its empirical version $\hat\pi_\beta(a \mid s)$.
A pair $(s, a)$ is *in the data*, or *in support*, if it appears at least once.

The exercise's behaviour policy is a cautious demonstrator on the cliff world
with $\gamma = 0.9$. It heads for the goal, takes a random allowed action 40%
of the time, and never steps toward the cliff. From 100 episodes you get 2,370
transitions. They cover 37 states, and 17 state-action pairs in those states
never appear. The demonstrator's own value at the start is $-8.984$.

## Watch Q-learning trust what it never saw

Run tabular Q-learning over this dataset until it stops changing. Each
iteration sets every covered pair to the mean of its targets:

$$
Q(s, a) \leftarrow \operatorname{mean}_{(s, a, r, s', d) \in \mathcal{D}} \Big[ r + \gamma (1 - d) \max_{a'} Q(s', a') \Big]
$$

Read it aloud: for each pair the data covers, average the one-step targets of
all its transitions. The max in each target ranges over *every* action at the
next state, including ones the data never took. Those keep their starting value
of 0, because no data ever updates them. Only the covered pairs change:

```python
y = r + gamma * (1 - done) * Q[s2].max(axis=1)     # (N,)
Q = np.where(seen, mean_target_per_pair(y), Q)     # unseen pairs stay put
```

In a world where every step costs $-1$, a value of 0 is wildly optimistic.
The max finds it. Fujimoto and colleagues (2019) called this *extrapolation
error*: values for actions outside the data are guesses, and the max
systematically selects the guesses that are too high. Online, the agent would
try that action and correct itself. Offline, nothing corrects it.

### Worked example: a three-cell corridor

Take the corridor `S.G` with $\gamma = 0.9$. Each move costs $-1$, and $G$ is
terminal. The demonstrator always moves right, so the data holds only right.

- In the middle cell, right enters the goal: $Q(1, \text{right}) = -1$.
- In the start cell, right lands in the middle cell, whose best estimate is an
  unseen 0: $Q(0, \text{right}) = -1 + 0.9 \times 0 = -1$.
- Every other action keeps $Q = 0$.

The true values are $Q^*(0, \text{right}) = -1.9$ and $Q^*(0, \text{up}) =
-2.71$. Offline Q-learning thinks up is worth 0, an overestimate of 2.71. Its
greedy policy picks up, which bumps the wall forever, and is worth
$-1 / (1 - 0.9) = -10$.

On the cliff, the same thing happens at scale. Stepping into the cliff from the
row above it is truly worth about $-106.7$, but it looks like 0. The greedy
policy steps into the cliff on every move and is worth exactly $-1000$.

> [!KEY]
> Offline, the values of actions the data never took are guesses that nothing
> can correct. A max over actions picks the most optimistic guess, and the
> policy follows it off the cliff.

The fixes fall into a few families. BCQ (Fujimoto and colleagues, 2019) and
TD3+BC (Fujimoto and Gu, 2021) keep the policy close to the data's actions.
CQL makes unseen actions look bad. IQL never evaluates an unseen action at
all. The rest of this lesson builds the last two.

## Push unseen values down: CQL

Conservative Q-learning (Kumar and colleagues, 2020) adds a penalty to the
usual squared TD error. In each state of the data, it pushes down a soft
maximum over all actions and pushes up the action the data took:

$$
\mathcal{R}(Q) = \frac{1}{N}\sum_{i=1}^{N} \Big[ \log \sum_{a} e^{Q(s_i, a)} - Q(s_i, a_i) \Big]
$$

Read it aloud: for each transition, take the log-sum-exp of its state's Q
values, which is close to the largest one, and subtract the Q of the action
actually taken. The penalty is small when the data's action already has the
highest value. CQL minimises $\alpha \mathcal{R}(Q)$ plus the squared TD
error, where $\alpha$ sets how conservative to be.

Its gradient follows from one fact: the gradient of a log-sum-exp is a
softmax. With respect to row $s$ of the table, each transition from $s$
contributes $(\operatorname{softmax}(Q(s, \cdot)) - e_{a_i}) / N$, where
$e_{a_i}$ is the indicator vector of the data's action.

```python
lse = top + np.log(np.exp(Q[s] - top[:, None]).sum(axis=1))  # top = row max
penalty = np.mean(lse - Q[s, a])
np.add.at(grad, s, softmax(Q[s]) / N)
np.add.at(grad, (s, a), -1.0 / N)
```

For $Q(s, \cdot) = (1, 2, 3)$ with the data's action 0, the penalty is
$\log(e + e^2 + e^3) - 1 = 3.4076 - 1 = 2.4076$. The gradient is the softmax
$(0.090, 0.245, 0.665)$ minus $(1, 0, 0)$, so the step lowers actions 1 and 2
and raises action 0.

### What the penalty does in a table

Look at one state and hold the TD targets $\bar y(s, a)$ fixed. For an action
the data never took, the only force is the penalty's push, $\alpha$ times its
softmax weight. That is always positive, so an unseen value sinks without limit,
more and more slowly as its softmax weight fades. For an action the data does
take, the gradient vanishes where

$$
Q(s, a) = \bar y(s, a) - \alpha \left( \frac{\mu(a \mid s)}{\hat\pi_\beta(a \mid s)} - 1 \right)
$$

Read it aloud: a covered value sits at its TD target, shifted down when the
softmax $\mu$ favours the action more than the data does, and shifted up when
it favours the action less. So individual values can land above the truth. In
the exercise's cliff table, CQL values the move up from the start at $-6.62$,
while the truth is $-7.46$. Kumar and colleagues prove a weaker, averaged
guarantee instead. With $\alpha$ large enough, the value averaged over the
learned policy's actions is a lower bound on that policy's true value.

The exercise takes gradient steps on the whole loss. It divides each state's
row of the gradient by that state's share of the data, so rare states learn as
fast as common ones. A positive step size per state changes the speed but not
where the minimum is. After 1,000 steps with $\alpha = 0.5$, every unseen value
sits below $-3$. The greedy policy uses only data actions, and it walks the
13-step safe path, worth $-7.458$. That beats the demonstrator's $-8.984$,
because the greedy choice drops the demonstrator's random detours.

## Never ask about unseen actions: IQL

CQL still takes a max over all actions and then fights the consequences.
Implicit Q-learning (Kostrikov, Nair, and Levine, 2022) removes the max from
the backup. It learns a separate state value $V(s)$ and bootstraps from that:

$$
Q(s, a) \leftarrow r + \gamma (1 - d) V(s')
$$

The trick is how $V$ is fit. If $V(s)$ were the plain mean of $Q(s, a)$ over
the data's actions, you'd get the behaviour policy's value, as SARSA does. If
it were the max over the data's actions, you'd get the best policy the data
supports. IQL interpolates with an *expectile*, fit by this loss:

$$
L_2^\tau(u) = \big|\tau - \mathbb{1}(u < 0)\big|\, u^2, \qquad u = Q(s, a) - V(s)
$$

Read it aloud: a squared error whose weight depends on the sign. When $Q$ is
above $V$, the error gets weight $\tau$. When $Q$ is below $V$, it gets
$1 - \tau$. With $\tau = 0.5$ both sides weigh the same, and the minimising
$V$ is the mean. With $\tau$ near 1, falling short of a high $Q$ costs far more
than overshooting a low one, so $V$ climbs toward the largest $Q$ among the
data's actions. The weight is constant on each side, so the gradient with
respect to $u$ is $2 |\tau - \mathbb{1}(u<0)| u$.

```python
weight = np.where(u < 0, 1 - tau, tau)
loss = np.mean(weight * u**2)
grad = 2 * weight * u / u.size
```

### Worked example: two actions

At $\tau = 0.7$, the loss is $L(1) = 0.7$ and $L(-1) = 0.3$. Now suppose a
state has two data actions with $Q$ values 1 and 3, equally often. The
expectile $v$ balances the weighted errors: $\tau (3 - v) = (1 - \tau)(v - 1)$.

- At $\tau = 0.5$: $v = 2$, the mean.
- At $\tau = 0.7$: $2.1 - 0.7 v = 0.3 v - 0.3$, so $v = 2.4$.
- At $\tau = 0.9$: $v = 2.8$, closing in on the max of 3.

The balance equation gives a quick way to compute an expectile. Weight each
value by $\tau$ or $1 - \tau$, according to its side of the current guess.
Take the weighted mean. Repeat until the sides stop changing, which takes a
few steps.

In the exercise, IQL at $\tau = 0.5$ reproduces the exact value of the data's
empirical behaviour policy, to $10^{-8}$. As $\tau$ rises, $V(\text{start})$
climbs from $-9.02$ to $-8.28$, $-7.70$, and $-7.48$ at $\tau = 0.99$. It
approaches the best in-data value, $-7.458$, and never passes it. No unseen
action ever enters a target.

> [!KEY]
> IQL bootstraps from a learned $V(s')$, never from a max over actions. The
> expectile $\tau$ slides $V$ from the behaviour policy's value toward the best
> value the data supports.

### Get a policy out

IQL's values never name a policy, so you extract one by *advantage-weighted
regression*. Reweight each data action by the exponential of its advantage:

$$
\pi(a \mid s) \propto \hat\pi_\beta(a \mid s) \exp\big(\beta\, (Q(s, a) - V(s))\big)
$$

Read it aloud: start from how often the data takes each action, and boost the
ones whose value beats the state's value. Here $\beta$ is an inverse
temperature, not the behaviour policy. At $\beta = 0$ you get behaviour cloning
back. As $\beta$ grows, you approach the best data action. An action with no
data has $\hat\pi_\beta = 0$, so no advantage can rescue it. In the network
version, IQL fits this by weighted maximum likelihood, with the weights
clipped (at 100 in the authors' code) so that one large advantage can't
dominate.

> [!RECAP]
> - Offline Q-learning leaves unseen pairs at their starting values, and the
>   max in every target selects them.
> - The CQL penalty is the mean of log-sum-exp minus the data action's Q, and
>   its gradient row is the softmax minus the data action's indicator.
> - The expectile loss weights $u^2$ by $\tau$ above zero and $1 - \tau$ below,
>   and $\tau = 0.5$ gives the mean.
> - IQL targets use $V(s')$, and its policy is the data's action frequencies
>   times $e^{\beta A}$, normalised.

## Build it step by step

The checks test the functions in this order. The helpers `make_dataset`,
`action_counts`, and `cautious_behavior` are written for you.

1. **`offline_q_iteration(data, n_states, n_actions, gamma, iters)`.** Returns
   `Q` with shape `(S, A)`. Compute all targets at once, sum them per pair
   with `np.add.at`, and divide by the counts. Common mistake: dividing by
   zero for unseen pairs, or writing their 0 targets over them. Use
   `np.where(seen, ...)`.
2. **`cql_penalty(Q, s, a)`.** Returns a float. Subtract each row's max
   before `np.exp`. The check includes a row of `[1000, 999]`.
3. **`cql_penalty_grad(Q, s, a)`.** Returns `(S, A)`. Use `np.add.at`, not
   `grad[s] += ...`, because fancy-index assignment drops repeated states.
4. **`cql_q_iteration(data, ..., alpha, iters, lr)`.** Recompute the targets
   each step. Add `alpha * cql_penalty_grad` to the TD gradient, divide each
   visited row by `counts[s].sum() / N`, and step. Common mistake: letting
   gradients flow into the target, which isn't part of this update.
5. **`expectile_loss` and `expectile_loss_grad`.** One `np.where` gives the
   weight. The gradient divides by `u.size` because the loss is a mean.
6. **`expectile(values, weights, tau, iters)`.** Work along the last axis so
   that one call handles every state. Start at the weighted mean and repeat the
   reweighting. Return 0 for rows with no weight.
7. **`iql_values(data, ..., tau, iters)`.** Alternate `V = expectile(Q,
   counts, tau)` and the mean target `r + gamma * (1 - done) * V[s2]`. Common
   mistake: a max anywhere in this function.
8. **`iql_policy(Q, V, counts, beta)`.** Mask unseen actions before
   normalising, and subtract each row's largest advantage before `np.exp`.

## Cost and scaling

All three methods cost about the same per update: one pass over a batch of
transitions. The following table compares what each adds.

| Method | Extra work per update | Extra memory | Main knob |
|---|---|---|---|
| Offline Q-learning | None | None | None |
| CQL | A log-sum-exp over actions; sampled actions when continuous | None | $\alpha$ |
| IQL | A second network for $V$, and a weighted policy fit | One value network | $\tau$, $\beta$ |
| BCQ | A generative model of the behaviour policy | One VAE | Perturbation size |

With continuous actions, CQL estimates its log-sum-exp from sampled actions,
which costs several extra Q evaluations per state. IQL never samples actions
for its value update, which makes it one of the cheapest offline methods.

## Where this breaks

**Coverage limits everything.** No method can learn that an action is good
if no data shows it. If the demonstrator never goes near the goal, offline RL
won't find the goal. The best these methods do is the best policy the data
supports.

**Tuning needs evaluation that you don't have.** Picking $\alpha$ or $\tau$ by
running the policy in the real environment defeats the purpose. Offline
policy evaluation, which estimates a policy's value from the logs alone, is
its own hard problem.

**CQL's $\alpha$ is sensitive to the dataset.** Too small, and unseen actions
still look good. Too large, and the $1 / \hat\pi_\beta$ term distorts the
values of rare data actions, so the policy collapses toward behaviour cloning.

**Narrow expert data leaves little to improve.** If the data comes from a
single good policy, every method reduces to imitating it. TD3+BC's plain
behaviour-cloning term then does as well as anything.


## Further reading

- [Levine, Kumar, Tucker, and Fu (2020), Offline Reinforcement Learning:
  Tutorial, Review, and Perspectives on Open
  Problems](https://arxiv.org/abs/2005.01643). A broad survey of why offline RL
  is hard and how the families of fixes relate.
- [Fujimoto, Meger, and Precup (2019), Off-Policy Deep Reinforcement Learning
  without Exploration](https://arxiv.org/abs/1812.02900). Names extrapolation
  error and introduces BCQ, which only considers actions a model of the data
  proposes.
- [Kumar, Zhou, Tucker, and Levine (2020), Conservative Q-Learning for Offline
  Reinforcement Learning](https://arxiv.org/abs/2006.04779). The CQL penalty,
  its tabular update, and the lower-bound theorems.
- [Kostrikov, Nair, and Levine (2022), Offline Reinforcement Learning with
  Implicit Q-Learning](https://arxiv.org/abs/2110.06169). The expectile value
  loss and advantage-weighted policy extraction.
- [Fujimoto and Gu (2021), A Minimalist Approach to Offline Reinforcement
  Learning](https://arxiv.org/abs/2106.06860). TD3+BC, which adds one
  behaviour-cloning term to TD3 and matches far more complex methods.
- [Chen and colleagues (2021), Decision Transformer: Reinforcement Learning via
  Sequence Modeling](https://arxiv.org/abs/2106.01345). Treats offline RL as
  predicting actions conditioned on a desired return, with no bootstrapping.

## The exercise

You build all three methods on the cliff world, from a cautious
demonstrator's 2,370 transitions. Write `offline_q_iteration` to see the
failure, then `cql_penalty`, `cql_penalty_grad`, and `cql_q_iteration` for
CQL. For IQL, write `expectile_loss`, `expectile_loss_grad`, `expectile`,
`iql_values`, and `iql_policy`.

The checks confirm the corridor's exact numbers and the cliff table's fixed
point, with unseen actions overestimated by more than 5 and a greedy value
near $-1000$. They test the CQL penalty against $2.4076$ and its gradient by
finite differences. CQL's greedy policy must use only data actions and reach
the best in-data value exactly. The expectile checks cover the loss values at
$\tau = 0.7$, the gradient by finite differences, and the mean at $\tau = 0.5$.
IQL at $\tau = 0.5$ must equal policy evaluation of the empirical behaviour
policy. As $\tau$ rises, its values must climb within 0.1 of the best in-data
value without passing it, and its policy must beat the demonstrator.
