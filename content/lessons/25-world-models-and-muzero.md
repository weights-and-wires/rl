---
title: World models and MuZero
slug: 25-world-models-and-muzero
part: "Part 6 — Models, exploration, and fixed data"
summary: Train on imagined rollouts with lambda-returns, see why a model only needs the right values, and search it with pUCT.
minutes: 65
gpu: false
objectives:
  - Compute Dreamer's lambda-returns over an imagined trajectory, with continuation flags, and relate them to GAE.
  - "State the value-equivalence principle: a wrong model is good enough when its Bellman backups match the real ones."
  - Implement MuZero's pUCT selection, expansion, and backup, and trace a small search by hand.
  - Explain why a learned value function lets a short search see far past its depth.
prereqs:
  - 24-model-based-rl
  - 10-planning-and-learning
  - 17-generalized-advantage-estimation
exercise: 25-world-models-and-muzero
---

# World models and MuZero

The last lesson predicted the next state of CartPole: four numbers. A game of
Atari hands the agent a picture of 210 by 160 pixels. Predicting every pixel
of the next frame is expensive, and most of those pixels don't matter. The
colour of the sky doesn't change which way to steer.

Two families of methods answer this. The first learns a compact *world model*
that works in a small learned code instead of pixels, and trains the agent on
trajectories it imagines inside that model. Dreamer is the best known. The
second, MuZero, doesn't try to predict what the world looks like at all. Its
model predicts only rewards, values, and good actions, and a tree search plans
with them.

This lesson builds the three pieces that make those methods work. You compute
the returns Dreamer trains on, check exactly when a model that gets the world
wrong can still plan correctly, and write MuZero's tree search.

## What you already know

- A model predicts the next state and reward, and its error compounds over an
  open-loop rollout.
- The lambda-return, from the lesson on n-step returns and eligibility
  traces, blends every n-step return with weights that shrink by $\lambda$.
- GAE, from the lesson on generalised advantage estimation, computes
  advantages by a backward recursion.
- The Bellman expectation equation says $V^\pi = r_\pi + \gamma P_\pi V^\pi$.
- Monte Carlo tree search with UCT, from the lesson on planning, learning,
  and search, balances trying good actions against trying rare ones.

## Imagine in a learned latent space

World Models (Ha and Schmidhuber, 2018) and the Dreamer family (Hafner and
others, 2020 to 2025) share one design. An encoder compresses each
observation into a small *latent state*, a vector the network learns. A
dynamics network predicts the next latent state from the current one and an
action. Small heads predict the reward and whether the episode continues.
These are trained on real experience.

Then the agent stops looking at the world. It starts from latent states of
real experience, rolls its policy forward for a short horizon inside the
model, and trains an actor and a critic on those imagined trajectories. This
is called *learning in imagination*. Dreamer imagines 15 steps ahead in its
first version and 16 in its third, all in the small latent space.

The critic needs a target to learn from. A one-step target $r_t + \gamma v(s_{t+1})$ leans entirely on the critic's own guess. A full imagined return
leans entirely on the model's rewards, which drift the further out they go.
Dreamer blends the two.

## Compute lambda-returns in imagination

The *lambda-return* $G^\lambda_t$ mixes one-step, two-step, and longer
returns, with the weight on longer ones controlled by $\lambda$. Over an
imagined trajectory of length $H$, it is computed backward from the end:

$$
G^\lambda_t = r_t + \gamma \, c_t \Big( (1 - \lambda)\, v(s_{t+1}) + \lambda \, G^\lambda_{t+1} \Big), \qquad G^\lambda_H = v(s_H)
$$

Read it piece by piece:

- $r_t$ is the model's predicted reward for taking $a_t$ in $s_t$.
- $c_t$ is the *continuation flag*: 1 if $s_{t+1}$ isn't terminal, 0 if it
  is. A zero cuts off everything after the episode ends.
- Inside the brackets, the target for what comes after step $t$ blends the
  critic's value of the next state, with weight $1 - \lambda$, and the
  lambda-return from the next step on, with weight $\lambda$.
- At the end of the horizon there's no more imagination, so the recursion
  starts from the critic's value $v(s_H)$.

With $\lambda = 0$ it's the one-step target. With $\lambda = 1$ it's the
discounted sum of all $H$ imagined rewards plus $\gamma^{H-t} v(s_H)$, and
the values in between don't matter at all. Dreamer uses $\lambda = 0.95$. In
numpy, the recursion is one backward loop:

```python
following = values[H]
for t in reversed(range(H)):
    following = rewards[t] + gamma * continues[t] * (
        (1 - lam) * values[t + 1] + lam * following)
    out[t] = following
```

### Worked example

Take $H = 3$, $\gamma = 0.9$, rewards $[0, 0, 1]$, values $[0.5, 0.6, 0.8, 0]$ for $s_0$ to $s_3$, and continuation flags $[1, 1, 0]$, so the episode
ends on the third step. Start at the end with $\lambda = 0.5$:

- $G_2 = 1 + 0.9 \times 0 \times (\ldots) = 1$. The zero flag cuts off $v(s_3)$.
- $G_1 = 0 + 0.9 \times (0.5 \times 0.8 + 0.5 \times 1) = 0.81$.
- $G_0 = 0 + 0.9 \times (0.5 \times 0.6 + 0.5 \times 0.81) = 0.6345$.

With $\lambda = 0$ the returns are $[0.54, 0.72, 1]$, which trust the critic.
With $\lambda = 1$ they're $[0.81, 0.9, 1]$, the discounted reward. The
$\lambda = 0.5$ answer sits between.

### The link to GAE

Subtract the value from the lambda-return and you get GAE's advantage. Define
the TD error $\delta_t = r_t + \gamma c_t v(s_{t+1}) - v(s_t)$. Then
$G^\lambda_t - v(s_t) = \delta_t + \gamma \lambda c_t (G^\lambda_{t+1} - v(s_{t+1}))$. That's exactly GAE's backward recursion, and both end at zero
at step $H$. So Dreamer's critic target and the GAE advantage PPO uses are the same
quantity viewed from two sides.

There's one more exact fact worth knowing. If the model is exact, the
policy and the model are deterministic, and the critic is the true $V^\pi$,
then every lambda-return equals $V^\pi(s_t)$, for every $\lambda$. Each term
becomes $r_t + \gamma c_t V^\pi(s_{t+1})$, which is the Bellman equation. The
exercise checks this on a grid world.

> [!KEY]
> The lambda-return trades the critic's bias against the model's drift:
> $\lambda = 0$ trusts the critic after one step, and $\lambda = 1$ trusts
> the model's rewards for the whole horizon.

## Predict only what planning needs

Dreamer still learns to reconstruct observations. MuZero (Schrittwieser and
others, 2020) drops that. Its model has three learned functions:

- A *representation* function turns past observations into a latent state.
- A *dynamics* function takes a latent state and an action, and returns a
  predicted reward and the next latent state.
- A *prediction* function takes a latent state, and returns a policy prior
  over actions and a value.

MuZero unrolls the model 5 steps along real action sequences and trains it
so that the predicted rewards, values, and policies match real rewards,
bootstrapped value targets, and the search's own action distribution. No
loss ever asks the latent state to look like the world. MuZero matched
AlphaZero in Go, chess, and shogi without being told the rules.

Why is that enough? The *value-equivalence principle* (Grimm and others,
2020) gives the answer. Planning only ever uses a model through Bellman
backups. So two models that give the same backups, for the policies and
values you plan with, give the same plans, even if one of them gets the
transitions wrong. Formally, a model $(\hat P, \hat R)$
is *value-equivalent* to the real environment, for a set of policies and a
set of value functions, when for every policy $\pi$ and value function $v$
in those sets:

$$
\hat r_\pi + \gamma \hat P_\pi v = r_\pi + \gamma P_\pi v
$$

Read it as: apply one step of the Bellman expectation backup for $\pi$ to
$v$, once with the model and once with reality. Here $r_\pi(s) = \sum_a \pi(a \mid s) R(s, a)$ is the expected reward under $\pi$ and $P_\pi$ is the
state-to-state transition matrix under $\pi$. The model passes when the two
answers agree at every state. In numpy, one backup is two lines:

```python
P_pi = np.einsum("sa,sat->st", pi, P)
backup = (pi * R).sum(axis=1)[:, None] + gamma * P_pi @ V   # V is (S, m)
```

The exercise's `value_equivalence_loss` is the mean squared gap between the
two backups, over every policy, state, and value function.

### Worked example

Take three states with values $v = [0, 1, 2]$. From some state, the real
next-state probabilities are $[0.2, 0.4, 0.4]$, and a model says $[0.3, 0.2, 0.5]$. The model is wrong about every single probability. But the expected
next value is the same:

- Real: $0.2 \times 0 + 0.4 \times 1 + 0.4 \times 2 = 1.2$.
- Model: $0.3 \times 0 + 0.2 \times 1 + 0.5 \times 2 = 1.2$.

The difference $[0.1, -0.2, 0.1]$ sums to zero, so both rows stay
probabilities, and it's orthogonal to $v$, so the backup can't see it. The
exercise builds a whole model this way, with every row perturbed in a
direction orthogonal to the value functions.

There's a strong consequence. Suppose the model is value-equivalent for a
policy $\pi$ and its true value $V^\pi$. Then the model's backup leaves
$V^\pi$ unchanged, because it matches the real backup, which leaves $V^\pi$
unchanged. When the model's rows are probability vectors, its backup is a
contraction with a single fixed point, so evaluating $\pi$ inside the wrong
model returns the true $V^\pi$ exactly.

> [!KEY]
> A model only has to get the Bellman backups right for the policies and
> values you plan with. Its transitions can be wrong in every direction the
> values can't see.

## Search with pUCT

MuZero plans by Monte Carlo tree search inside its model. Every node holds,
for each action $a$, a visit count $N(a)$, a mean backed-up return $Q(a)$,
and a prior $P(a)$ from the prediction function. To choose which action to
follow from a node, MuZero uses the *pUCT* rule:

$$
\text{score}(a) = Q(a) + P(a) \, \frac{\sqrt{\sum_b N(b)}}{1 + N(a)} \left( c_1 + \log \frac{\sum_b N(b) + c_2 + 1}{c_2} \right)
$$

Read it as two terms. $Q(a)$ rewards actions that have done well. The second
term rewards actions the prior likes and that have few visits compared with
their siblings. It shrinks as $N(a)$ grows. $c_1 = 1.25$ sets its size, and
the logarithm with $c_2 = 19652$ grows it slowly once a node has tens of
thousands of visits. In numpy:

```python
total = visits.sum()
weight = c1 + np.log((total + c2 + 1) / c2)
scores = q + prior * np.sqrt(total) / (1 + visits) * weight
```

Each simulation walks down the tree by pUCT until it picks an action it has
never tried. It asks the model for that action's reward and next state, and
asks the prediction function for the added node's prior and value. Then it
walks back up. The leaf value becomes a return, $G \leftarrow r + \gamma G$,
at each edge on the way, and each edge adds $G$ to its total and 1 to its
count. A terminal next state contributes a leaf value of 0.

Two details differ from the paper. MuZero rescales $Q$ to the range 0 to 1,
using the smallest and largest values in the tree, so that one $c_1$ works
for every game. The exercise skips that, because its rewards already lie
between 0 and 1. And the exercise scores an action that has never been tried
by the node's own predicted value, so a good value function isn't drowned
out by zeros.

### Worked example

A toy tree has two actions at the root. Action 0 pays 0.5 and ends the
episode. Action 1 pays nothing and leads to a state B, where action 1 pays 1
and ends. The best first move is action 1, worth $0.9 \times 1 = 0.9$ with
$\gamma = 0.9$. The prior is uniform, $[0.5, 0.5]$, and every predicted
value is 0. The following table traces the root for the first four
simulations:

| Simulation | $N$ before | $Q$ before | Scores | Picks | Return backed up |
|---|---|---|---|---|---|
| 1 | $[0, 0]$ | $[0, 0]$ | $[0, 0]$ | 0 (tie) | $0.5$ |
| 2 | $[1, 0]$ | $[0.5, 0]$ | $[0.8125, 0.6251]$ | 0 | $0.5$ |
| 3 | $[2, 0]$ | $[0.5, 0]$ | $[0.7947, 0.8840]$ | 1 | $0 + 0.9 \times 0 = 0$ |
| 4 | $[2, 1]$ | $[0.5, 0]$ | $[0.8609, 0.5414]$ | 0 | $0.5$ |

In simulation 2, the weight is $1.25 + \log(19654 / 19652) = 1.2501$, and
action 0 scores $0.5 + 0.5 \times 1 / 2 \times 1.2501 = 0.8125$. After four
simulations the root has visits $[3, 1]$ and $Q = [0.5, 0]$. The search
hasn't found the reward under B yet, because B's value prediction is 0. By
20 simulations it has, and visits are $[10, 10]$. After 200 they are $[21, 179]$, with $Q(1) = 0.86$ approaching the true 0.9.

## A value function makes search short

A search with a zero value can see only as far as its tree reaches. The
exercise shows it on a 3 by 4 grid world whose goal is 5 steps from the
start. With a zero value and 200 simulations, the root's visits split
exactly $[50, 50, 50, 50]$: no simulation reaches the goal, so no action
looks better than another.

Give the same search values computed inside a model learned from 400 random
transitions, and 200 simulations from the start give visits $[43, 60, 54, 43]$. Actions 1 and 2, right and down, are the optimal ones. From every
non-terminal cell, the most visited action is optimal. The value function
summarises everything beyond the leaves, so a shallow search acts as if it
could see to the goal. That's one reason MuZero learns a value at all.

> [!RECAP]
> - The lambda-return recursion runs backward from $G_H = v(s_H)$, and the
>   continuation flag multiplies the whole bracket.
> - Lambda-return minus value is GAE's advantage.
> - A model is value-equivalent when its Bellman backups equal the real ones
>   for the chosen policies and values.
> - pUCT is $Q$ plus prior times $\sqrt{\sum N} / (1 + N)$ times the
>   $c_1, c_2$ weight, and ties go to the lowest action.
> - Backup turns the leaf value into $G \leftarrow r + \gamma G$ at each edge.

## Build it step by step

The checks test the functions in this order.

1. **`lambda_returns(rewards, values, continues, gamma, lam)`.** Takes arrays
   of length $H$, $H + 1$, and $H$. Returns length $H$. One reversed loop,
   seeded with `values[H]`. Common mistake: multiplying only $v(s_{t+1})$ by
   the flag and not the $\lambda$ term, which leaks value across an episode
   end.
2. **`imagine(step_fn, policy_fn, value_fn, s0, horizon)`.** Loop `horizon`
   times: pick an action, step the model, record the reward, the flag, and
   the value of the next state. Return float arrays of shapes $(H,)$,
   $(H + 1,)$, and $(H,)$. Common mistake: forgetting to start with
   `values[0] = value_fn(s0)`, which shifts every value by one step.
3. **`bellman_backup(P, R, gamma, pi, v)`.** Build `P_pi` with `np.einsum` and
   `r_pi` with a row sum. Support `v` of shape `(S,)` and `(S, m)`: broadcast
   `r_pi[:, None]` in the second case.
4. **`value_equivalence_loss(P_model, R_model, P, R, gamma, policies, values)`.**
   For each policy, subtract the two backups of `values`, square, and sum.
   Divide by the total number of entries. Common mistake: comparing the
   models' matrices instead of their backups, which fails the check with a
   wrong model that plans correctly.
5. **`puct_scores(q, prior, visits, c1, c2)`.** One vectorised line. With no
   visits, `np.sqrt(0)` makes the exploration term zero.
6. **`mcts(model_step, predict, root_state, n_actions, n_simulations, gamma)`.**
   Keep a small node class holding the state, prior, value, and arrays `N`,
   `W`, and `R`, plus a dictionary of children. Mark an edge to a terminal
   state with `None`. Each simulation records the `(node, action)` path, then
   backs up in reverse. Common mistake: adding the reward after the discount,
   $\gamma (G + r)$, instead of $r + \gamma G$.

## Cost and scaling

The following table shows how each piece grows, for a horizon $H$, $S$ states,
$A$ actions, $m$ value functions, $n$ simulations, and tree depth $D$.

| Piece | Time | Memory |
|---|---|---|
| Lambda-returns | $O(H)$ per trajectory | $O(H)$ |
| One Bellman backup of $m$ functions | $O(S^2 A + S^2 m)$ | $O(S^2 A)$ |
| One MCTS simulation | $O(D \, A)$ plus one model call | one node per simulation |
| A search | $O(n \, D \, A)$ plus $n$ model calls | $O(n A)$ |

For MuZero, the model calls dominate: every simulation runs the dynamics and
prediction networks once. That's why its search budget is small, 50
simulations per move in Atari and 800 in board games. Dreamer imagines
thousands of trajectories in parallel on a GPU, at a cost that grows with the
batch size times $H$, and it needs no search when acting.

## Where this breaks

**The actor exploits the model.** Dreamer's policy trains only in
imagination, so it chases any reward the model invents. Short horizons,
lambda-returns that lean on the critic, and a model retrained on fresh real
data keep this in check.

**Value equivalence is only as broad as its sets.** A model that's
value-equivalent for some policies and values can be wrong for others. As a
policy improves, it visits situations the model wasn't trained for. MuZero
keeps training the model on data from its most recent policy for this reason.

**Stochastic worlds break a deterministic search.** The search in this lesson
assumes each action has one outcome. MuZero's original model is
deterministic too. Games with chance, such as dice, need a model that
represents several outcomes, and a search that averages over them.

**Scale matters for pUCT.** With rewards in the hundreds, $Q$ swamps the
exploration term and pUCT stops exploring. Normalise $Q$ within the tree, as
MuZero does.

**Search with a poor value is myopic.** Without a useful value, the search
needs a tree deep enough to reach every reward, and the tree grows
exponentially with depth.

## Further reading

- [Ha and Schmidhuber (2018), World models](https://arxiv.org/abs/1803.10122).
  A variational autoencoder, a recurrent dynamics model, and a tiny
  controller trained inside the model's dream.
- [Hafner, Lillicrap, Ba, and Norouzi (2020), Dream to control: learning behaviors by latent imagination](https://arxiv.org/abs/1912.01603).
  Dreamer: lambda-returns over imagined latent trajectories, with the actor
  trained by backpropagating through the model.
- [Hafner, Pasukonis, Ba, and Lillicrap (2025), Mastering diverse control tasks through world models](https://doi.org/10.1038/s41586-025-08744-2).
  DreamerV3 in Nature: one configuration across more than 150 tasks,
  including diamonds in Minecraft from scratch.
- [Hafner, Yan, and Lillicrap (2025), Training agents inside of scalable world models](https://arxiv.org/abs/2509.24527).
  Dreamer 4: a transformer world model that learns Minecraft diamonds from
  offline data alone.
- [Oh, Singh, and Lee (2017), Value prediction network](https://arxiv.org/abs/1707.03497).
  An early model that predicts rewards and values instead of observations.
- [Schrittwieser and others (2020), Mastering Atari, Go, chess and shogi by planning with a learned model](https://arxiv.org/abs/1911.08265).
  MuZero: representation, dynamics, and prediction functions with pUCT
  search.
- [Grimm, Barreto, Singh, and Silver (2020), The value equivalence principle for model-based reinforcement learning](https://arxiv.org/abs/2011.03506).
  The formal case that a model needs only to match Bellman backups.

## The exercise

You build the three ideas of this lesson. Write `lambda_returns` for
Dreamer's critic targets and `imagine` to roll a policy through a
deterministic model. Write `bellman_backup` and `value_equivalence_loss` to
measure when a model plans correctly. Then write `puct_scores` and `mcts` for
MuZero's tree search.

The checks verify `lambda_returns` against the worked example, its
$\lambda = 0$ and $\lambda = 1$ limits, a continuation cut, and the GAE
identity. They check that an exact imagined rollout with $V^\pi$ gives
lambda-returns equal to $V^\pi$ for every $\lambda$. They check
`bellman_backup` against its fixed point, and that `value_equivalence_loss`
is zero for a model whose transitions differ from the truth but agree on the
value functions. Finally they check `puct_scores` by hand, replay the
four-simulation trace, confirm that 200 simulations find the delayed reward
on the toy tree, and confirm that search over a learned grid-world model
picks an optimal action from every state.
