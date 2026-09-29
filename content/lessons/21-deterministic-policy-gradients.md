---
title: Deterministic policy gradients
slug: 21-deterministic-policy-gradients
part: "Part 5 — Continuous control and maximum entropy"
summary: Check the deterministic policy gradient theorem exactly, then build TD3's smoothed clipped double-Q update and train it on Pendulum.
minutes: 60
gpu: false
objectives:
  - State the deterministic policy gradient theorem, and verify it against finite differences on a scalar linear-quadratic problem.
  - Build DDPG's critic target and actor loss, and say why the actor must never train the critic.
  - Implement TD3's clipped double-Q target, target policy smoothing, Polyak averaging, and delayed updates, and train it on Pendulum.
prereqs:
  - 12-deep-q-networks
  - 08-double-learning
  - 16-actor-critic
exercise: 21-deterministic-policy-gradients
---

# Deterministic policy gradients

A robot arm doesn't choose from four actions. It chooses a torque, any real
number in a range. A deep Q-network can't handle that. It picks the action
with the largest value, and finding the largest of infinitely many values is
an optimisation problem at every single step.

The stochastic policy gradient from earlier lessons works with continuous
actions, but it's noisy. It learns only by trying random actions and
comparing the results. This lesson takes a different route. Keep a
*deterministic* policy that outputs one action per state. Learn a critic
$Q(s, a)$ that's differentiable in the action. Then move the action in the
direction the critic says is uphill.

That idea is the deterministic policy gradient. DDPG turned it into a deep
learning method, and TD3 fixed DDPG's worst habit: trusting a critic that
overestimates. You check the theorem exactly on a problem with a closed-form
answer, then build TD3 piece by piece and train it to swing up a pendulum.

## What you already know

- A deep Q-network learns $Q(s, a)$ from a replay buffer, with a slowly
  updated target network for the regression target.
- The max of noisy estimates is biased upward. Double learning fixes this by
  choosing and evaluating with different estimates.
- The policy gradient theorem says how the return changes with the policy's
  parameters, without differentiating the state distribution.
- An actor-critic method trains a policy, the *actor*, with help from a
  learned value function, the *critic*.
- The discounted return from state $s_t$ is $G_t = \sum_k \gamma^k r_{t+k}$.

## Follow the critic's slope

A *deterministic policy* $\mu_\theta(s)$ outputs an action, not a
distribution. The critic $Q^\mu(s, a)$ says how good each action would be.
If $Q$ increases when you nudge the action to the right, the policy needs to
output a slightly larger action. Chain that through the policy's parameters
$\theta$, and you have a gradient.

Silver et al. (2014) proved that this is the exact gradient of the return
$J(\theta)$:

$$
\nabla_\theta J(\theta) = \sum_s \rho^\mu(s)\, \nabla_\theta \mu_\theta(s)\, \nabla_a Q^\mu(s, a) \Big|_{a = \mu_\theta(s)}
$$

Read it from the right. $\nabla_a Q^\mu$ is the critic's slope in the
action, measured at the action the policy takes. $\nabla_\theta
\mu_\theta(s)$ turns that into a direction in parameter space, by the chain
rule. $\rho^\mu(s) = \sum_t \gamma^t \Pr(s_t = s)$ weights each state by its
discounted visits under the policy. It's unnormalised, so it sums to
$1 / (1 - \gamma)$.

Like the stochastic theorem, this one ignores how changing $\theta$ changes
which states you visit. There's also no expectation over actions. That's why
it's cheaper to estimate than the stochastic gradient. Silver et al. also
show it's the limit of the stochastic policy gradient as the policy's noise
shrinks to zero.

In PyTorch, autograd applies the chain rule for you. Minimise the negative
critic value at the policy's own actions:

```python
loss = -critic(s, actor(s)).mean()   # gradient flows through actor(s) into theta
actor_opt.zero_grad()
loss.backward()
actor_opt.step()                     # step only the actor's parameters
```

> [!KEY]
> The deterministic policy gradient moves each action along the critic's
> slope $\nabla_a Q(s, a)$, then chains that through the policy's
> parameters. No sampling over actions is needed.

### Worked example

A *linear-quadratic* problem has linear dynamics and a quadratic reward, so
every value has a closed form. Take a scalar state with

$$
s_{t+1} = c\, s_t + b\, a_t, \qquad r_t = -\big(s_t^2 + \lambda a_t^2\big)
$$

Read it as: the next state is a weighted sum of the current state and
action, and the reward penalises both being far from zero. Use $c = 1$, $b =
1$, $\lambda = 0.5$, $\gamma = 0.9$, and the linear policy $\mu_\theta(s) =
\theta s$ with $\theta = -0.5$.

- **The state shrinks geometrically.** Each step multiplies it by $m = c + b
  \theta = 0.5$.
- **The value is quadratic.** Each step pays $-(1 + \lambda \theta^2)
  s_t^2$, and the discounted sum of $s_t^2$ is $s^2 / (1 - \gamma m^2)$. So
  $V(s) = -P s^2$ with $P = (1 + \lambda \theta^2) / (1 - \gamma m^2) =
  1.125 / 0.775 = 1.4516$.
- **The action value adds one step.** $Q(s, a) = -(s^2 + \lambda a^2) -
  \gamma P (c s + b a)^2$.
- **The critic's slope.** Differentiate in $a$: $\partial Q / \partial a =
  -2 \lambda a - 2 \gamma P b (c s + b a)$. At $s = 1$ and $a = \mu(1) =
  -0.5$, it's $0.5 - 1.3065 = -0.8065$.
- **The policy's slope.** $\partial \mu / \partial \theta = s$.

The gradient sums $\gamma^t s_t \cdot \partial Q / \partial a$ along the
trajectory $s_t = 0.5^t$. Every term is $-0.8065 \times (\gamma m^2)^t =
-0.8065 \times 0.225^t$, so the sum is $-0.8065 / 0.775 = -1.0406$.
Differentiate $J(\theta) = -P(\theta)$ directly, and you get the same
$-1.0406$. The gradient is negative, so a stronger negative feedback gain
does better. Follow it, and it vanishes at $\theta^* = -0.7092$, the gain the
discounted Riccati equation gives.

> [!WARNING]
> Hold $Q^\mu$ fixed when you differentiate in the action. The theorem
> differentiates the critic in $a$ only. The $P$ inside $Q$ belongs to the
> current policy, and it doesn't move with $a$.

## Build DDPG from DQN's parts

Lillicrap et al. (2016) combined the deterministic policy gradient with the
two stabilisers from deep Q-networks: a replay buffer and target networks.
The result, *deep deterministic policy gradient* (DDPG), learns from stored
transitions $(s, a, r, s', \text{terminated})$. The critic $Q_\phi$ regresses
toward

$$
y = r + \gamma (1 - \text{terminated})\, Q_{\phi'}\big(s', \mu_{\theta'}(s')\big)
$$

Read it as the DQN target, with the max over next actions replaced by the
target actor's choice. $\phi'$ and $\theta'$ are *target networks*, slow
copies of the critic and actor. The `(1 - terminated)` factor stops the
target from bootstrapping past a true terminal state. A time limit isn't
terminal, so Pendulum, which only truncates, always bootstraps.

The actor minimises $-Q_\phi(s, \mu_\theta(s))$ over states from the buffer.
After each update, the targets move a fraction $\tau$ toward the online
networks. This is *Polyak averaging*:

$$
\phi' \leftarrow \tau \phi + (1 - \tau) \phi'
$$

Read it as a running average of the online weights. With $\tau = 0.005$, a
target forgets its past with a half-life of about 140 updates. In PyTorch,
update in place without tracking gradients:

```python
with torch.no_grad():
    for t, p in zip(target.parameters(), source.parameters()):
        t.mul_(1 - tau).add_(tau * p)
```

A deterministic policy never explores on its own. DDPG adds noise to the
actions it takes in the environment. The original paper used a correlated
Ornstein-Uhlenbeck process. Fujimoto et al. (2018) found that plain Gaussian
noise works as well.

> [!WARNING]
> The actor's loss runs through the critic, so `loss.backward()` also writes
> gradients into the critic's parameters. Step only the actor's optimiser,
> and zero the critic's gradients before its own step. Otherwise the critic
> learns to flatter the actor.

## Fix overestimation with TD3

DDPG's actor climbs whatever the critic says. If the critic overestimates
some action, the actor finds it and exploits it. That's the maximisation bias
from the double learning lesson, with an optimiser pushing on it. Fujimoto et
al. (2018) measured large overestimation in DDPG and proposed *twin delayed
DDPG* (TD3). It makes three changes.

### Clipped double Q-learning

Train two critics, $Q_{\phi_1}$ and $Q_{\phi_2}$. Build one target from the
smaller of their target copies:

$$
y = r + \gamma (1 - \text{terminated}) \min_{i = 1, 2} Q_{\phi'_i}(s', \tilde{a})
$$

Read it as the DDPG target, taking the more pessimistic of two independent
opinions. Two critics rarely overestimate the same action by the same amount,
so the minimum cancels most of the upward bias. It can underestimate a
little, and that's the lesser evil, since the actor doesn't chase low values.
Both critics regress to the same $y$. In PyTorch it's
`r + gamma * (1 - terminated) * torch.minimum(q1_next, q2_next)`.

### Target policy smoothing

A critic can develop a narrow, wrong spike in $a$. To stop the target from
riding that spike, TD3 adds clipped noise to the target action:

$$
\tilde{a} = \operatorname{clip}\big(\mu_{\theta'}(s') + \operatorname{clip}(\varepsilon, -c, c),\ a_{\text{low}},\ a_{\text{high}}\big), \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2)
$$

Read it from the inside out. Draw Gaussian noise, clip it to at most $c$ in
size, add it to the target actor's action, then clip the result to the
action bounds. The target then averages $Q$ over a small neighbourhood of
actions. The paper uses $\sigma = 0.2$ and $c = 0.5$ for actions in $[-1,
1]$. The authors' code scales both by the action bound, and so does the
exercise.

### Delayed policy updates

Update the critics every step, but the actor and all three target networks
only every $d = 2$ steps. The actor then climbs a critic that has had time to
settle. TD3 uses $\tau = 0.005$ and exploration noise with standard deviation
$0.1$ times the action bound.

### Worked example

Take $r = 1$, $\gamma = 0.99$, and target critics that score the smoothed
next action at $3.0$ and $2.5$. The target is $y = 1 + 0.99 \times 2.5 =
3.475$. DDPG with the first critic alone would have used $3.97$.

For the smoothing, take action bound 2, $\sigma = 0.4$, and $c = 1$:

- $\mu = 0$ and $\varepsilon = 1$: noise $0.4$, so $\tilde{a} = 0.4$.
- $\mu = 1.8$ and $\varepsilon = 2$: noise $0.8$, and $2.6$ is clipped to
  $2$.
- $\mu = -1$ and $\varepsilon = -10$: noise $-4$ is clipped to $-1$, then
  $-2$ stays at the bound.

> [!KEY]
> TD3 targets the smaller of two critics at a slightly jittered action, and
> updates the actor half as often as the critics. Each change starves the
> actor of overestimated values to exploit.

> [!RECAP]
> - The deterministic policy gradient sums $\nabla_\theta \mu \cdot
>   \nabla_a Q$ over discounted visits. On the scalar problem, it's
>   $-1.0406$ at $\theta = -0.5$.
> - The smoothed target action clips the noise first, then the action.
> - The target is `r + gamma * (1 - terminated) * min(q1, q2)`, computed
>   under `torch.no_grad()` from the target networks.
> - The actor minimises `-critic1(s, actor(s)).mean()`.
> - Critics update every call. The actor and the three targets update when
>   `step % policy_delay == 0`.

## Build it step by step

The checks test the functions in this order.

1. **`lqr_value_coef(theta, lqr)`.** Returns $P$. Compute $m = c + b\theta$,
   then $(1 + \lambda \theta^2) / (1 - \gamma m^2)$. The check compares
   $-P s_0^2$ with a 600-step simulated return.
2. **`lqr_q(s, a, theta, lqr)`.** Returns one step's reward plus $\gamma$
   times $V$ of the next state, $-P (c s + b a)^2$.
3. **`lqr_dpg(theta, lqr, s0, horizon)`.** Loop over the policy's own
   trajectory. At each state, multiply $\gamma^t$, $\partial \mu /
   \partial \theta = s_t$, and $\partial Q / \partial a$ at $a = \theta s_t$.
   Common mistake: differentiating $P$ as well, which double-counts. The check
   compares with finite differences of the simulated $J$.
4. **`smoothed_target_action(mu_next, eps, sigma, noise_clip, max_action)`.**
   Two `clamp` calls. Common mistake: clipping the sum only, so a huge draw
   moves the action by more than `noise_clip`.
5. **`td3_target(r, terminated, q1_next, q2_next, gamma)`.** One line with
   `torch.minimum`. The check includes the lesson's 3.475 and a terminal
   transition.
6. **`critic_loss(agent, batch, eps)`.** Inside `torch.no_grad()`, run
   `agent.actor_target` on `batch["s2"]`, smooth with `agent.policy_noise`
   and `agent.noise_clip`, and score with both target critics. Outside it,
   sum the two critics' mean squared errors. Common mistake: using the online
   actor or critics in the target.
7. **`actor_loss(agent, s)`.** `-agent.critic1(s, agent.actor(s)).mean()`.
   The check compares its gradient with $\nabla_\theta \mu \cdot \nabla_a
   Q_1$ assembled by hand. Common mistake: using `min(Q1, Q2)`.
8. **`soft_update(target, source, tau)`.** In place, under
   `torch.no_grad()`.
9. **`td3_update(agent, batch, eps, step)`.** Critic step first. Then, when
   `step % agent.policy_delay == 0`, an actor step and three soft updates.
   Call `zero_grad()` before each `backward()`.

The last check runs the given `train_td3` for 8,000 steps and expects the
noise-free actor to average above $-800$. A random policy averages about
$-1300$, and the reference solution about $-180$.

## Cost and scaling

The following table shows what a TD3 update costs with batch size $N$, a
network of $W$ weights, and a replay buffer of $M$ transitions.

| Piece | Time per update | Memory |
|---|---|---|
| Critic step | Forward and backward passes through two critics, and forward passes through three target networks: $O(NW)$ | Two critics, two targets, and Adam state |
| Actor step, every $d$ updates | Forward and backward passes through the actor and one critic: $O(NW)$ | One actor, one target, and Adam state |
| Polyak averaging | $O(W)$ per network | None extra |
| Replay buffer | $O(N)$ to sample | $O(M)$ transitions |

TD3 costs about twice what DDPG costs per update, and gets that back by
wasting fewer updates on a fooled actor. The Pendulum run in the exercise
takes 8,000 environment steps and 7,000 updates, and finishes in about five
seconds on one CPU thread with 128-unit layers. Harder robot tasks need a
million steps and a GPU.

## Where this breaks

**Exploration is bolted on.** A deterministic policy explores only through
the noise you add. Too little, and the critic never sees the good actions.
Too much, and the data doesn't reflect the policy. Soft actor-critic, the
next lesson, builds exploration into the objective instead.

**The critic's slope must be meaningful.** The actor follows $\nabla_a Q$,
which a neural network can get badly wrong away from the data. Clipped
double Q and smoothing reduce this, and don't remove it. When a run
collapses, the critic's values have usually run away first. Plot them.

**The sample gradient is off-policy in the states.** The theorem weights
states by the current policy's visits. The buffer holds states from older,
noisier policies. Silver et al.'s off-policy version drops a term to allow
this, so the update is an approximation.

**Hyperparameters and seeds matter.** DDPG is notoriously sensitive to
learning rates, noise, and network size, and results vary widely across
seeds (Henderson et al., 2018). The Pendulum check in the exercise was
calibrated over 42 seeds before it was trusted.

**Saturated actions stall learning.** A `tanh` output layer keeps actions in
bounds, but when it saturates, its gradient vanishes, and the actor stops
moving. Keep pre-activations moderate, or penalise them.

## Further reading

The following papers introduce and analyse the methods in this lesson:

- [Silver et al. (2014), Deterministic policy gradient algorithms](https://proceedings.mlr.press/v32/silver14.html)
  proves the deterministic policy gradient theorem and its link to the
  stochastic one.
- [Lillicrap et al. (2016), Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
  introduces DDPG, with replay, target networks, and Polyak averaging.
- [Fujimoto, van Hoof, and Meger (2018), Addressing function approximation error in actor-critic methods](https://arxiv.org/abs/1802.09477)
  measures DDPG's overestimation and introduces TD3's three fixes.
- [Henderson et al. (2018), Deep reinforcement learning that matters](https://arxiv.org/abs/1709.06560)
  shows how much DDPG-style results depend on seeds and implementation details.
- [Haarnoja et al. (2018), Soft actor-critic](https://arxiv.org/abs/1801.01290)
  keeps TD3's twin critics and replaces the deterministic actor with a
  maximum-entropy stochastic one.

## The exercise

You check the theorem, then build TD3. Write `lqr_value_coef`, `lqr_q`, and
`lqr_dpg` for the scalar linear-quadratic problem. Then write
`smoothed_target_action`, `td3_target`, `critic_loss`, `actor_loss`,
`soft_update`, and `td3_update`. The networks, the `TD3Agent`, the replay
buffer, and `train_td3` are given.

The checks compare the value coefficient and $Q$ with simulated returns, and
the deterministic policy gradient with finite differences of the return on
two systems. They test the smoothing noise's two clips, the clipped
double-Q target on hand-worked cases, and that `critic_loss` uses the target
networks without sending them gradient. They compare `actor_loss`'s gradient
with the chain rule assembled by hand, check Polyak averaging exactly, and
confirm that the actor and targets move only on delayed steps. The last
check trains TD3 on Pendulum for 8,000 steps.
