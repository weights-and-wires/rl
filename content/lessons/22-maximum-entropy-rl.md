---
title: Maximum-entropy reinforcement learning
slug: 22-maximum-entropy-rl
part: "Part 5 — Continuous control and maximum entropy"
summary: Add entropy to the reward, solve the soft Bellman equation exactly, and watch it become ordinary value iteration as the temperature falls.
minutes: 55
gpu: false
objectives:
  - Compute the soft value alpha * logsumexp(Q / alpha) stably, and show the Boltzmann policy maximises expected Q plus alpha times entropy.
  - Implement soft value iteration and soft policy iteration, and verify that both reach the same fixed point of the soft Bellman equation.
  - "Bound what the temperature costs: 0 <= V_soft - V* <= alpha log A / (1 - gamma), and recover value iteration as alpha goes to 0."
prereqs:
  - 05-dynamic-programming
  - 20-policy-optimization-theory
exercise: 22-maximum-entropy-rl
---

# Maximum-entropy reinforcement learning

An ordinary reinforcement learning agent wants the most reward, and it doesn't
care how. When two actions are equally good, it picks one and forgets the
other. When one is a hair better, it commits to it completely. That makes the
agent brittle. It stops exploring early, it can't tell you about the second-best
way to do the job, and a small change in the world can leave it stranded.

Maximum-entropy reinforcement learning changes the goal. The agent earns its
reward plus a bonus for acting randomly. It must still succeed, but among the
ways to succeed, it prefers the ones that keep its options open. A temperature
sets the exchange rate between reward and randomness.

This lesson solves that problem exactly on small MDPs. The hard max in the
Bellman equation turns into a soft max, and the optimal policy turns into a
Boltzmann distribution. The next lesson, on soft actor-critic, approximates
the same equations with neural networks for continuous actions.

## What you already know

- Value iteration applies the backup $Q(s, a) \leftarrow r(s, a) + \gamma
  \sum_{s'} P(s' \mid s, a) \max_{a'} Q(s', a')$ until it stops changing.
- The backup is a contraction, so value iteration converges from any start.
- Policy iteration alternates exact evaluation with a greedy improvement
  step.
- A softmax turns a vector of scores into probabilities that sum to one.
- The natural policy gradient multiplies a policy by the exponential of its
  advantage, and at its largest step it becomes soft policy iteration.

## Add entropy to the reward

The *entropy* of a distribution $p$ over actions measures how spread out it
is:

$$
\mathcal{H}(p) = -\sum_a p(a) \log p(a)
$$

Read it as the average surprise of a draw from $p$. A policy that always picks
one action has entropy 0. The uniform policy over $A$ actions has the largest
possible entropy, $\log A$. Over two actions, that's $\log 2 = 0.6931$ nats.
In numpy, it's `-(p * np.log(p)).sum(axis=-1)`, with any `0 * log 0` term
counted as 0.

The maximum-entropy objective adds the entropy of the policy at each visited
state to the reward:

$$
J_\alpha(\pi) = \mathbb{E}_\pi\left[\sum_{t \ge 0} \gamma^t \big(r_t + \alpha\, \mathcal{H}(\pi(\cdot \mid s_t))\big)\right]
$$

Read it as the ordinary discounted return, plus $\alpha$ times the entropy of
the action choice at every step. The *temperature* $\alpha > 0$ is the
exchange rate. At $\alpha = 0$, it's ordinary reinforcement learning. As
$\alpha$ grows, randomness becomes worth more than reward.

> [!KEY]
> Maximum-entropy reinforcement learning pays the agent $\alpha$ times the
> entropy of its action choice at every step, so among equally good ways to
> succeed, it prefers the most random one.

## Replace the max with a soft max

Start with one state. You know the value $Q(a)$ of each action. Which
distribution $p$ maximises $\sum_a p(a) Q(a) + \alpha \mathcal{H}(p)$? A bit
of calculus with a Lagrange multiplier gives the answer. It's the *Boltzmann
distribution*, a softmax of $Q / \alpha$, and its score is the *soft
maximum*:

$$
V = \alpha \log \sum_a \exp\big(Q(a) / \alpha\big), \qquad
p^*(a) = \exp\big((Q(a) - V) / \alpha\big)
$$

Read the first formula as: divide each value by the temperature, take a
log-sum-exp, and multiply back by the temperature. Read the second as: an
action's probability falls off exponentially with how far its value sits
below the soft maximum. Dividing by $e^{V / \alpha}$ is what makes the
probabilities sum to one.

The soft maximum sits between two familiar numbers:

- As $\alpha \to 0$, it becomes $\max_a Q(a)$, and $p^*$ puts all its mass on
  the best action.
- As $\alpha \to \infty$, it becomes the mean of $Q$ plus $\alpha \log A$,
  and $p^*$ becomes uniform.

In numpy, subtract the row maximum first, or `exp` overflows:

```python
top = q.max(axis=-1)
v = top + alpha * np.log(np.exp((q - top[..., None]) / alpha).sum(axis=-1))
pi = np.exp((q - v[..., None]) / alpha)
```

### Worked example

Take $Q = (1, 2)$ and $\alpha = 0.5$:

- $Q / \alpha = (2, 4)$, and $e^2 + e^4 = 7.3891 + 54.5982 = 61.9872$.
- $V = 0.5 \log 61.9872 = 2.0635$. It's a little above the hard max of 2.
- $p^* = (e^2, e^4) / 61.9872 = (0.1192, 0.8808)$.

Check the claim that $V$ is the best score. The expected value under $p^*$ is
$0.1192 + 2 \times 0.8808 = 1.8808$. The entropy is $0.3653$ nats, and half
of it is $0.1827$. Their sum is $2.0635$, which matches $V$.

> [!WARNING]
> Compute the soft maximum with the row maximum subtracted. With $Q$ near
> 1000 and $\alpha = 0.01$, the naive `np.exp(q / alpha)` is `inf`.

## Solve the soft Bellman equation

In an MDP, the value of an action includes the entropy you collect later. The
soft-optimal action values $Q^*_\alpha$ satisfy the *soft Bellman equation*:

$$
Q^*_\alpha(s, a) = r(s, a) + \gamma \sum_{s'} P(s' \mid s, a)\, V^*_\alpha(s'), \qquad
V^*_\alpha(s) = \alpha \log \sum_{a} \exp\big(Q^*_\alpha(s, a) / \alpha\big)
$$

Read it as ordinary Bellman optimality with the max over next actions
replaced by the soft maximum. The optimal policy is the Boltzmann policy of
$Q^*_\alpha$, $\pi^*_\alpha(a \mid s) = \exp\big((Q^*_\alpha(s, a) -
V^*_\alpha(s)) / \alpha\big)$. A terminal state ends the episode, so no more
entropy accrues, and its soft value is 0.

The soft maximum moves by at most as much as its largest input moves. So the
soft backup is a contraction by $\gamma$, like the hard one. Iterate it from
$Q = 0$, and it converges to $Q^*_\alpha$. That's *soft value iteration*. In
numpy, one backup reuses `rlenvs.q_from_v`:

```python
v = np.where(mdp.terminal, 0.0, soft_value(q, alpha))
q = rlenvs.q_from_v(mdp, v)
```

> [!WARNING]
> A terminal state's row of $Q$ is all zeros, and the soft maximum of zeros
> is $\alpha \log A$, not 0. Mask terminal states before the backup, or every
> episode ends with a phantom entropy bonus.

### Worked example

Take two states. In each, action 0 stays and action 1 switches to the other
state. Staying in state 1 pays 1, and everything else pays 0. Set $\gamma =
0.9$ and $\alpha = 1$.

- Start from $Q = 0$. Both states have soft value $\log 2 = 0.6931$.
- First backup: $Q = \begin{bmatrix} 0.6238 & 0.6238 \\ 1.6238 & 0.6238 \end{bmatrix}$,
  since $0.9 \times 0.6931 = 0.6238$.
- Soft values: $V(0) = \log(2 e^{0.6238}) = 1.3170$, and
  $V(1) = \log(e^{1.6238} + e^{0.6238}) = 1.9371$.
- Second backup: $Q = \begin{bmatrix} 1.1853 & 1.7434 \\ 2.7434 & 1.1853 \end{bmatrix}$.

Keep going, and soft value iteration converges to

$$
Q^*_1 = \begin{bmatrix} 9.8439 & 10.5298 \\ 11.5298 & 9.8439 \end{bmatrix}, \qquad
V^*_1 = (10.9377, 11.6998)
$$

The Boltzmann policy switches out of state 0 with probability 0.665 and stays
in state 1 with probability 0.844. The hard optimum has $V^* = (9, 10)$ and a
deterministic policy. Each soft value is higher because it counts the
entropy the agent collects forever. At $\alpha = 0.1$, soft value iteration
already matches $V^* = (9, 10)$ to four decimal places.

## See what the temperature costs

The soft-optimal policy gives up some reward to stay random. How much? The
entropy bonus at each step lies between 0 and $\alpha \log A$. Summed with
discounting, it lies between 0 and $\alpha \log A / (1 - \gamma)$. So the soft
value and the hard optimal value satisfy

$$
0 \le V^*_\alpha(s) - V^*(s) \le \frac{\alpha \log A}{1 - \gamma}
$$

Read it as: the soft value is never below the hard value, since the hard
optimal policy is one choice the soft agent could make. It's never more than
the most entropy the agent could collect. As $\alpha \to 0$, the gap closes,
and soft value iteration becomes `rlenvs.value_iteration`.

On the frozen lake, with $\gamma = 0.95$ and four actions, the start state's
values are as follows.

| $\alpha$ | $V^*_\alpha(\text{start})$ | $V^*_\alpha - V^*$ | Bound $\alpha \log 4 / 0.05$ |
|---|---|---|---|
| 1 | 9.7011 | 9.5638 | 27.73 |
| 0.1 | 0.9826 | 0.8452 | 2.773 |
| 0.01 | 0.1694 | 0.0320 | 0.2773 |
| 0.001 | 0.1374 | 0.0000 | 0.0277 |

The hard value is $V^*(\text{start}) = 0.1374$. At $\alpha = 1$ the entropy
bonus dwarfs the single reward of 1 for reaching the goal. At $\alpha =
0.001$ the two agree to four decimal places.

> [!KEY]
> The temperature costs at most $\alpha \log A / (1 - \gamma)$ in value, and
> as $\alpha \to 0$ the soft Bellman equation becomes the ordinary one.

## Improve a policy softly

Value iteration has a partner, policy iteration. The soft version alternates
two steps:

1. **Soft policy evaluation.** Compute the soft action values of the current
   policy $\pi$. They satisfy
   $Q^\pi(s, a) = r(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^\pi(s')$, with
   $V^\pi(s) = \sum_a \pi(a \mid s)\big(Q^\pi(s, a) - \alpha \log \pi(a \mid s)\big)$.
2. **Soft policy improvement.** Replace $\pi$ with the Boltzmann policy of
   $Q^\pi$.

Read the evaluation equation as ordinary evaluation with one extra reward:
acting in state $s$ also pays $-\alpha \log \pi(a \mid s)$. Its average
over actions is $\alpha \mathcal{H}(\pi(\cdot \mid s))$. So the soft values
of a fixed policy come from one linear solve, `rlenvs.policy_evaluation`,
on a copy of the MDP whose reward is `R - alpha * np.log(pi)`.

Haarnoja et al. (2018) prove that the improvement step never lowers any soft
action value, and that the loop converges to $\pi^*_\alpha$. It usually takes
a handful of iterations. On the frozen lake at $\alpha = 0.1$, the distance
to $Q^*_\alpha$ is $0.135$, then $0.0023$, then $1.6 \times 10^{-6}$, then
rounding error.

Soft policy iteration is the algorithm that soft actor-critic approximates.
Its critic does soft policy evaluation from samples, and its actor moves
toward the Boltzmann policy of the critic. The lesson on why policy
optimisation works met the same step from the other side: natural policy
gradient with an entropy bonus, at its largest step size, is exactly soft
policy iteration.

## Measure the soft Bellman residual

With a model, you back up every state at once. Without one, you have sampled
transitions $(s, a, r, s', \text{terminated})$. The *soft Bellman residual*
on one transition is

$$
\delta = r + \gamma\, (1 - \text{terminated})\, V_\alpha(s') - Q(s, a), \qquad
V_\alpha(s') = \alpha \log \sum_{a'} \exp\big(Q(s', a') / \alpha\big)
$$

Read it as the TD error of Q-learning, with the max over next actions
replaced by the soft maximum. At $Q^*_\alpha$, the residual averages to
exactly zero over the next state. Anywhere else, it doesn't. So
minimising the mean squared residual over a replay buffer is *soft
Q-learning* (Haarnoja et al., 2017). Soft actor-critic's critic minimises the
same squared residual, halved, with the soft value estimated from a sampled
next action.

In numpy, for integer arrays `s`, `a`, and `s2`:

```python
target = r + gamma * (1.0 - terminated) * soft_value(q[s2], alpha)
delta = target - q[s, a]
```

### Relate entropy to a KL penalty

Entropy is a KL penalty toward the uniform policy in disguise:
$\mathcal{H}(p) = \log A - D_{\mathrm{KL}}(p \,\Vert\, \text{uniform})$.
Replace the uniform policy with any reference policy $\pi_{\text{ref}}$, and
the soft maximum becomes
$\alpha \log \sum_a \pi_{\text{ref}}(a \mid s) \exp(Q(s, a) / \alpha)$. The
optimal policy tilts the reference by $\exp(Q / \alpha)$. That's the
KL-regularised objective that RLHF uses, which Part 8 builds on.

Levine (2018) shows a third view. Shift the rewards to be at most 0, treat
"acting optimally" as an observed event with probability $e^{r}$, and ask
which actions explain it. Variational
inference with the dynamics held fixed recovers this lesson's equations at
$\alpha = 1$.

> [!RECAP]
> - `soft_value` is the log-sum-exp of `q / alpha`, times `alpha`, with the
>   row maximum subtracted first.
> - `boltzmann_policy` is `exp((q - v[..., None]) / alpha)`.
> - One soft backup masks terminal states to 0, then calls
>   `rlenvs.q_from_v`.
> - Soft policy evaluation is `rlenvs.policy_evaluation` with the reward
>   `R - alpha * log(pi)`.
> - The residual uses `soft_value(q[s2], alpha)` times `1 - terminated`.

## Build it step by step

The checks test the functions in this order.

1. **`soft_value(q, alpha)`.** Takes `q` of shape `(S, A)` and returns
   `(S,)`. Subtract each row's maximum, exponentiate `(q - top) / alpha`,
   sum, take the log, scale by `alpha`, and add the maximum back. Common
   mistake: forgetting to add the maximum back. The check feeds values near
   1000 at $\alpha = 0.01$.
2. **`boltzmann_policy(q, alpha)`.** Returns `(S, A)`. Reuse `soft_value`, and
   broadcast it with `[..., None]`. The check confirms the policy beats 3,000
   random policies on expected $Q$ plus $\alpha$ times entropy.
3. **`soft_bellman_backup(mdp, q, alpha)`.** Returns `(S, A)`. Compute the
   soft value, set it to 0 where `mdp.terminal` is true, and pass it to
   `rlenvs.q_from_v`. Common mistake: skipping the terminal mask. The frozen
   lake check catches it.
4. **`soft_value_iteration(mdp, alpha, tol)`.** Returns $Q^*_\alpha$. Loop
   backups from zeros until the largest change is below `tol`. The checks
   test the fixed point, and the bound against `rlenvs.value_iteration` for
   $\alpha$ from 1 down to 0.001.
5. **`soft_policy_evaluation(mdp, pi, alpha)`.** Returns `(S, A)`. Build a
   `rlenvs.TabularMDP` with reward `mdp.R - alpha * log_pi`, evaluate `pi` on
   it, and convert with `rlenvs.q_from_v` on the original MDP. Treat
   `log 0` as 0, which matters only when $\alpha = 0$.
6. **`soft_bellman_residual(q, s, a, r, s2, terminated, gamma, alpha)`.**
   Returns `(N,)`. Index `q[s2]` for the next states' rows and `q[s, a]` for
   the current values. The check weights every possible next state by its
   probability and expects an exact zero at $Q^*_\alpha$.
7. **`soft_policy_iteration(mdp, alpha, iters)`.** Returns
   `(iters + 1, S, A)`. Start uniform, evaluate, record, and improve. Record
   before improving, so entry 0 belongs to the uniform policy.

## Cost and scaling

Soft dynamic programming costs the same as the hard version, plus one
exponential and one log per state-action pair. The following table assumes
$S$ states and $A$ actions.

| Method | Cost per iteration | Iterations |
|---|---|---|
| Soft value iteration | $O(S^2 A)$ | About $\log(1/\text{tol}) / \log(1/\gamma)$ |
| Soft policy evaluation | $O(S^3 + S^2 A)$ for a linear solve | One |
| Soft policy iteration | One evaluation, plus $O(SA)$ | Usually under ten |

Soft value iteration needs about 260 backups to reach $10^{-12}$ at
$\gamma = 0.9$, and about 540 at $\gamma = 0.95$. Memory is $O(S^2 A)$ for the
model and $O(SA)$ for $Q$. Tabular methods stop there. With continuous
actions, the sum inside the soft maximum becomes an integral, and soft
actor-critic replaces it with samples from a learned policy.

## Where this breaks

**The temperature is a unit problem.** The bonus $\alpha \mathcal{H}$ is in
nats, and the reward is in whatever units the environment uses. Scale the
reward by 10, and the same $\alpha$ means a very different agent. Soft
actor-critic fixes this by tuning $\alpha$ to hit a target entropy.

**You optimise a different objective.** The soft optimum isn't the hard
optimum. With a large $\alpha$, the agent can prefer wandering over reaching
the goal, as the frozen lake table shows at $\alpha = 1$. Anneal $\alpha$, or
report the policy's ordinary return next to its soft return.

**Small temperatures bring back the hard problem.** As $\alpha \to 0$, the
Boltzmann policy becomes greedy, and the soft residual inherits Q-learning's
maximisation bias. The soft maximum is convex, so by Jensen's inequality, a soft
maximum of noisy estimates is biased upward too.

**Entropy is relative to the action coordinates.** For discrete actions,
uniform is the natural reference. For continuous actions, differential
entropy depends on how you scale the action. Squashing actions into bounds,
as the next lesson does, changes the entropy, and the log-probability needs
a correction for it.

**Exact inference is optimistic.** The inference view with exact posterior
inference lets the posterior change the dynamics. The backup becomes
$\log \mathbb{E}_{s'} e^{V(s')}$, which acts as if the agent could pick
lucky transitions. Keep the dynamics fixed, as soft value iteration does.

## Further reading

The following sources develop maximum-entropy reinforcement learning:

- [Ziebart (2010), Modeling purposeful adaptive behavior with the principle of maximum causal entropy](https://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf)
  develops the soft Bellman equations and maximum causal entropy.
- [Haarnoja, Tang, Abbeel, and Levine (2017), Reinforcement learning with deep energy-based policies](https://arxiv.org/abs/1702.08165)
  introduces soft Q-learning with neural networks.
- [Haarnoja, Zhou, Abbeel, and Levine (2018), Soft actor-critic](https://arxiv.org/abs/1801.01290)
  proves soft policy evaluation, improvement, and convergence, then
  approximates them.
- [Schulman, Chen, and Abbeel (2017), Equivalence between policy gradients and soft Q-learning](https://arxiv.org/abs/1704.06440)
  shows the entropy-regularised policy gradient and soft Q-learning take the
  same expected step.
- [Nachum, Norouzi, Xu, and Schuurmans (2017), Bridging the gap between value and policy based reinforcement learning](https://arxiv.org/abs/1702.08892)
  derives path consistency, a multi-step form of the soft Bellman equation.
- [Geist, Scherrer, and Pietquin (2019), A theory of regularized Markov decision processes](https://arxiv.org/abs/1901.11275)
  treats entropy and KL as two cases of one regularised framework.
- [Levine (2018), Reinforcement learning and control as probabilistic inference](https://arxiv.org/abs/1805.00909)
  derives maximum-entropy reinforcement learning as variational inference.

## The exercise

You build soft dynamic programming from the soft maximum up. Write
`soft_value` and `boltzmann_policy`, then `soft_bellman_backup` and
`soft_value_iteration`. Write `soft_policy_evaluation` for the exact soft
values of a fixed policy, `soft_bellman_residual` for sampled transitions,
and `soft_policy_iteration` to alternate evaluation and Boltzmann
improvement.

The checks confirm the soft value and policy on the worked example, stability
near $Q = 1000$, and both temperature limits. They test that the Boltzmann
policy beats random policies on expected $Q$ plus entropy, that the backup
zeroes terminal states, and that soft value iteration reaches a fixed point
on a random MDP and the frozen lake. They check the bound $0 \le V^*_\alpha -
V^* \le \alpha \log A / (1 - \gamma)$ as $\alpha$ falls to 0.001. They also
check that the Boltzmann policy of $Q^*_\alpha$ has soft value $Q^*_\alpha$,
that the residual averages to exactly zero at $Q^*_\alpha$, and that soft
policy iteration improves every soft action value until it matches soft value
iteration.
