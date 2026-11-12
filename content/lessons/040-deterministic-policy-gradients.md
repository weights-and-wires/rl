---
title: "Deterministic policy gradients: DDPG and TD3"
slug: 040-deterministic-policy-gradients
part: "Part 6 — Continuous control and maximum entropy"
summary: Move an actor's action up the critic's slope, then harden that update with TD3's twin critics, smoothed targets, and delayed actor steps.
minutes: 45
gpu: false
objectives:
  - Explain why a continuous-action agent moves its action along the critic's slope instead of taking a max.
  - Check the deterministic policy gradient against finite differences on a scalar linear-quadratic problem.
  - "Build TD3's update: a smoothed clipped double-Q target, the actor and critic losses, Polyak averaging, and delayed actor steps."
prereqs:
  - 026-deep-q-networks
  - 027-double-and-dueling-dqn
  - 033-baselines-and-actor-critic
exercise: 040-deterministic-policy-gradients
---

# Deterministic policy gradients: DDPG and TD3

A robot arm does not choose from four buttons. It chooses a torque, any real number between $-2$ and $2$. A deep Q-network cannot do that, because it acts by scoring every action and taking the best one, and an interval holds infinitely many actions. You would have to run a small optimization inside every decision.

The policy gradient from Part 5 does work with continuous actions, but it learns by trying random actions and comparing the results, which is slow and noisy. This chapter takes a shortcut. Keep an *actor* that outputs one action for each state, learn a *critic* that scores actions, and nudge the actor's action in the direction the critic says is uphill.

That idea has one weak spot. The actor climbs whatever the critic says, so if the critic overestimates some action, the actor finds it and exploits it. By the end you will have the fix, TD3, and a check that trains it to swing up a pendulum.

## What you already know

- A deep Q-network regresses $Q(s, a)$ toward a bootstrapped target computed by a slowly moving target network (see [Deep Q-networks](/lessons/026-deep-q-networks)).
- The maximum of noisy estimates is biased upward, and double learning removes the bias by using two estimates (see [Double and dueling DQN](/lessons/027-double-and-dueling-dqn)).
- An actor-critic method trains a policy, the *actor*, with help from a learned value function, the *critic* (see [Baselines and actor-critic](/lessons/033-baselines-and-actor-critic)).
- Autograd applies the chain rule to any composition of differentiable functions.

## Why not take the maximum?

A Q-network picks $a = \arg\max_a Q(s, a)$. With four actions, that is a comparison of four numbers. With a torque, it is a search over a continuous range, repeated at every step. You could run gradient ascent on $a$ each time, but the search in one state looks much like the search in the next, so that repeats work.

So learn the answer instead. A network $\mu_\theta(s)$, the *deterministic policy*, outputs the one action it believes is best. The remaining question is how to train $\theta$ so that $\mu_\theta(s)$ lands where $Q$ is high.

## Follow the critic's slope

Look at Figure 1. It plots the critic's value of every action in one state. The actor outputs $a = -0.5$. The curve slopes downward there, which means a slightly smaller action would score higher. The critic has told the actor which way to move.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Critic value Q(1, a) against the action a: the actor picks a = -0.5, where the slope is -0.81, so a larger value lies to the left, with the peak at a = -0.72">
<defs><marker id="c036-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<line x1="70.0" y1="215.0" x2="560.0" y2="215.0" class="stroke-grid"/>
<text x="62.0" y="219.0" class="fig-small" text-anchor="end">-6</text>
<line x1="70.0" y1="145.0" x2="560.0" y2="145.0" class="stroke-grid"/>
<text x="62.0" y="149.0" class="fig-small" text-anchor="end">-4</text>
<line x1="70.0" y1="75.0" x2="560.0" y2="75.0" class="stroke-grid"/>
<text x="62.0" y="79.0" class="fig-small" text-anchor="end">-2</text>
<line x1="70.0" y1="250.0" x2="560.0" y2="250.0" class="stroke-muted"/>
<line x1="70.0" y1="250.0" x2="70.0" y2="255.0" class="stroke-muted"/>
<text x="70.0" y="270.0" class="fig-small" text-anchor="middle">-2</text>
<line x1="233.3" y1="250.0" x2="233.3" y2="255.0" class="stroke-muted"/>
<text x="233.3" y="270.0" class="fig-small" text-anchor="middle">-1</text>
<line x1="396.7" y1="250.0" x2="396.7" y2="255.0" class="stroke-muted"/>
<text x="396.7" y="270.0" class="fig-small" text-anchor="middle">0</text>
<line x1="560.0" y1="250.0" x2="560.0" y2="255.0" class="stroke-muted"/>
<text x="560.0" y="270.0" class="fig-small" text-anchor="middle">1</text>
<text x="315.0" y="288.0" text-anchor="middle">action a (state s = 1)</text>
<text x="62.0" y="24.0" text-anchor="end">Q(1, a)</text>
<polyline points="70.0,155.7 74.1,151.7 78.2,147.8 82.2,144.0 86.3,140.2 90.4,136.5 94.5,132.9 98.6,129.4 102.7,126.0 106.8,122.6 110.8,119.3 114.9,116.1 119.0,113.0 123.1,109.9 127.2,107.0 131.2,104.1 135.3,101.3 139.4,98.5 143.5,95.9 147.6,93.3 151.7,90.8 155.8,88.4 159.8,86.1 163.9,83.8 168.0,81.6 172.1,79.5 176.2,77.5 180.2,75.6 184.3,73.7 188.4,71.9 192.5,70.2 196.6,68.6 200.7,67.0 204.7,65.6 208.8,64.2 212.9,62.9 217.0,61.6 221.1,60.5 225.2,59.4 229.2,58.4 233.3,57.5 237.4,56.7 241.5,55.9 245.6,55.2 249.7,54.6 253.8,54.1 257.8,53.7 261.9,53.3 266.0,53.0 270.1,52.8 274.2,52.7 278.2,52.7 282.3,52.7 286.4,52.8 290.5,53.0 294.6,53.3 298.7,53.6 302.8,54.0 306.8,54.6 310.9,55.1 315.0,55.8 319.1,56.6 323.2,57.4 327.2,58.3 331.3,59.3 335.4,60.3 339.5,61.5 343.6,62.7 347.7,64.0 351.8,65.4 355.8,66.8 359.9,68.4 364.0,70.0 368.1,71.7 372.2,73.4 376.2,75.3 380.3,77.2 384.4,79.2 388.5,81.3 392.6,83.5 396.7,85.7 400.7,88.1 404.8,90.5 408.9,92.9 413.0,95.5 417.1,98.1 421.2,100.9 425.2,103.7 429.3,106.5 433.4,109.5 437.5,112.5 441.6,115.7 445.7,118.9 449.8,122.1 453.8,125.5 457.9,128.9 462.0,132.4 466.1,136.0 470.2,139.7 474.3,143.4 478.3,147.3 482.4,151.2 486.5,155.1 490.6,159.2 494.7,163.4 498.8,167.6 502.8,171.9 506.9,176.3 511.0,180.7 515.1,185.3 519.2,189.9 523.2,194.6 527.3,199.4 531.4,204.2 535.5,209.1 539.6,214.2 543.7,219.2 547.8,224.4 551.8,229.7 555.9,235.0 560.0,240.4" class="stroke-a"/>
<polyline points="233.3,41.7 396.7,69.9" class="stroke-muted dashed"/>
<circle cx="315.0" cy="55.8" r="6" class="fill-b"/>
<line x1="311.0" y1="106.5" x2="282.5" y2="106.5" class="stroke-b" marker-end="url(#c036-arrow)"/>
<text x="295.4" y="124.5" class="fig-small" text-anchor="middle">move left</text>
<text x="190.0" y="196.0" class="fig-small" text-anchor="start">orange dot: actor outputs a = μ(1) = -0.5</text>
<text x="190.0" y="216.0" class="fig-small" text-anchor="start">dashed line: slope ∂Q/∂a = -0.8065 there</text>
<text x="190.0" y="236.0" class="fig-small" text-anchor="start">green dot: the peak, a = -0.723</text>
<circle cx="278.5" cy="52.7" r="5" class="fill-c"/>
</svg>
Figure 1. The critic's value of each action in state 1, for the worked example below. The slope at the actor's action is negative, so the actor should move left, toward the peak at -0.723.
```

To make that concrete, use a problem with exact answers. A *linear-quadratic* problem has a scalar state with $s_{t+1} = c\, s_t + b\, a_t$ and reward $r_t = -(s_t^2 + \lambda a_t^2)$, so the agent wants state and action both near zero. Take $c = b = 1$, $\lambda = 0.5$, $\gamma = 0.9$, and the linear policy $\mu_\theta(s) = \theta s$ with $\theta = -0.5$. Every quantity has a closed form:

- The state shrinks by $m = c + b\theta = 0.5$ each step, so $s_t = 0.5^t s_0$.
- Each step pays $-(1 + \lambda\theta^2) s_t^2$, and the discounted sum of $s_t^2$ is $s^2 / (1 - \gamma m^2)$. So $V(s) = -P s^2$ with $P = 1.125 / 0.775 = 1.4516$.
- The action value adds one free step: $Q(s, a) = -(s^2 + \lambda a^2) - \gamma P (c s + b a)^2$.
- Differentiate in the action: $\partial Q / \partial a = -2\lambda a - 2\gamma P b (c s + b a)$. At $s = 1$ and $a = -0.5$ that is $0.5 - 1.3065 = -0.8065$.

The slope $-0.8065$ is the dashed line in Figure 1. It says how the *action* should change. The policy has a parameter $\theta$, not an action, so use the chain rule: $\partial\mu / \partial\theta = s$, because $\mu = \theta s$. At $s = 1$, the push on $\theta$ is $1 \times (-0.8065)$.

The return $J(\theta)$ sums that push over every state the policy visits, weighted by $\gamma^t$. Here $s_t = 0.5^t$, so every term is $-0.8065 \times 0.225^t$, and the sum is $-0.8065 / 0.775 = -1.0406$. Differentiating $J(\theta) = -P(\theta)$ directly gives the same $-1.0406$, which is the theorem below at work.

$$
\nabla_\theta J(\theta) = \sum_s \rho^\mu(s)\, \nabla_\theta \mu_\theta(s)\, \nabla_a Q^\mu(s, a) \Big|_{a = \mu_\theta(s)}
$$

Read it from the right. $\nabla_a Q^\mu$ is the critic's slope in the action, measured at the action the policy takes. $\nabla_\theta \mu_\theta(s)$ turns that into a direction in parameter space. $\rho^\mu(s) = \sum_t \gamma^t \Pr(s_t = s)$ weights each state by its discounted visits, so it sums to $1/(1-\gamma)$. This is Silver et al.'s (2014) deterministic policy gradient theorem.

The theorem needs no expectation over actions, because the policy picks only one, so it is cheaper to estimate than the stochastic gradient. Like the stochastic version, it does not differentiate how $\theta$ changes which states you visit. In PyTorch, autograd does the chain rule for you:

```python
loss = -critic(s, actor(s)).mean()   # gradient flows through actor(s) into theta
actor_opt.zero_grad()
loss.backward()
actor_opt.step()                     # step only the actor's parameters
```

> [!WARNING]
> Hold $Q^\mu$ fixed when you differentiate in the action. The $P$ inside $Q$ belongs to the current policy and does not move with $a$. Differentiating it too counts the same effect twice.

## Build DDPG from DQN's parts

Lillicrap et al. (2016) combined this gradient with the two stabilizers from deep Q-networks, a replay buffer and target networks. The result is *deep deterministic policy gradient* (DDPG). The critic $Q_\phi$ regresses toward

$$
y = r + \gamma (1 - \text{terminated})\, Q_{\phi'}\big(s', \mu_{\theta'}(s')\big)
$$

Read it as the DQN target with the max over next actions replaced by the target actor's choice. The primes mark *target networks*, slow copies of the critic and actor. The factor $(1 - \text{terminated})$ stops the target from bootstrapping past a true end. A time limit is not an end, so Pendulum, which only truncates, always bootstraps.

After each update, each target moves a fraction $\tau$ toward its online network. This is *Polyak averaging*, $\phi' \leftarrow \tau\phi + (1-\tau)\phi'$. With $\tau = 0.005$ a target forgets its past with a half-life of about 140 updates. In PyTorch, update in place:

```python
with torch.no_grad():
    for t, p in zip(target.parameters(), source.parameters()):
        t.mul_(1 - tau).add_(tau * p)
```

A deterministic policy never explores on its own, so DDPG adds Gaussian noise to the actions it takes in the environment.

> [!WARNING]
> The actor's loss runs through the critic, so `loss.backward()` also writes gradients into the critic's parameters. Step only the actor's optimizer, and call `zero_grad()` on the critic's optimizer before its own step. Otherwise the critic learns from the actor's objective.

## Stop the actor from exploiting the critic

Why does DDPG misbehave? The actor climbs whatever the critic says. A neural critic has small errors, and the actor finds the action where the error is most positive. That is the maximization bias from double learning, with an optimizer pushing on it. Fujimoto et al. (2018) measured large overestimation in DDPG and proposed *twin delayed DDPG* (TD3), which makes three changes.

**Clipped double Q-learning.** Train two critics, and build one target from the smaller of their target copies:

$$
y = r + \gamma (1 - \text{terminated}) \min_{i = 1, 2} Q_{\phi'_i}(s', \tilde{a})
$$

Two critics rarely overestimate the same action by the same amount, so the minimum cancels most of the upward bias. It can underestimate a little, which is the lesser evil. Take $r = 1$, $\gamma = 0.99$, and target critics that score the next action at $3.0$ and $2.5$. The target is $y = 1 + 0.99 \times 2.5 = 3.475$. Critic 1 alone would have given $3.97$, as Figure 2 shows. Both critics regress to the same $y$.

```figure
<svg viewBox="0 0 600 260" role="img" aria-label="Bars: critic 1 scores the next action 3.0 and critic 2 scores it 2.5; the target from critic 1 alone is 3.97 and the clipped target is 3.475">
<text x="20.0" y="49.0" class="fig-small" text-anchor="start">critic 1, next action</text>
<rect x="210.0" y="30.0" width="214.3" height="28.0" class="fill-a"/>
<text x="432.3" y="49.0" class="fig-small" text-anchor="start">3.0</text>
<text x="20.0" y="101.0" class="fig-small" text-anchor="start">critic 2, next action</text>
<rect x="210.0" y="82.0" width="178.6" height="28.0" class="fill-a"/>
<text x="396.6" y="101.0" class="fig-small" text-anchor="start">2.5</text>
<text x="20.0" y="171.0" class="fig-small" text-anchor="start">target using critic 1 only</text>
<rect x="210.0" y="152.0" width="283.6" height="28.0" class="fill-bad"/>
<text x="501.6" y="171.0" class="fig-small" text-anchor="start">3.970</text>
<text x="20.0" y="223.0" class="fig-small" text-anchor="start">target using the smaller one</text>
<rect x="210.0" y="204.0" width="248.2" height="28.0" class="fill-b"/>
<text x="466.2" y="223.0" class="fig-small" text-anchor="start">3.475</text>
<line x1="210.0" y1="24.0" x2="210.0" y2="250.0" class="stroke-muted"/>
<line x1="30.0" y1="134.0" x2="580.0" y2="134.0" class="stroke-grid"/>
<text x="230.0" y="128.0" class="fig-small" text-anchor="start">y = r + γ × (value), with r = 1 and γ = 0.99</text>
</svg>
Figure 2. Two critics disagree about the same next action. The target built from the smaller score, 3.475, sits below the 3.97 that critic 1 alone would give. Notice that the gap is the upward bias that the actor would otherwise chase.
```

**Target policy smoothing.** A critic can grow a narrow, wrong spike in $a$. To stop the target from riding it, TD3 adds clipped noise to the target action, then clips the result to the bounds:

$$
\tilde{a} = \operatorname{clip}\big(\mu_{\theta'}(s') + \operatorname{clip}(\varepsilon, -c, c),\ -a_{\max},\ a_{\max}\big), \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2)
$$

Read it from the inside out. Draw Gaussian noise, clip it to size $c$, add it to the target action, and clip to the bounds. The target then averages $Q$ over a small neighborhood of actions. With bound $2$, $\sigma = 0.4$, and $c = 1$: for $\mu = 0$ and $\varepsilon = 1$ the noise is $0.4$ and $\tilde{a} = 0.4$. For $\mu = 1.8$ and $\varepsilon = 2$ the noise is $0.8$, and $2.6$ clips to the bound, $2$. For $\varepsilon = -10$ the noise $-4$ clips to $-1$ first, so one wild draw cannot move the action far.

**Delayed policy updates.** Update the critics every step, but the actor and the three target networks only every $d = 2$ steps. The actor then climbs a critic that has had time to settle.

> [!KEY]
> A deterministic actor needs no max over actions: it moves along the critic's slope, $\nabla_\theta \mu \cdot \nabla_a Q$. That makes the critic's mistakes dangerous, so TD3 takes the smaller of two critics at a jittered action, and updates the actor half as often. You can now answer how an agent picks a continuous action without searching: it learns a network that already outputs the uphill answer.

> [!RECAP]
> - The deterministic policy gradient sums $\nabla_\theta \mu \cdot \nabla_a Q$ over discounted visits. On the scalar problem it is $-1.0406$ at $\theta = -0.5$.
> - The smoothed target action clips the noise first, then the action.
> - The target is `r + gamma * (1 - terminated) * min(q1, q2)`, computed under `torch.no_grad()` from the target networks.
> - The actor minimizes `-critic1(s, actor(s)).mean()`.
> - The critics update on every call. The actor and the three targets update when `step % policy_delay == 0`.

## Build it step by step

The exercise gives you the networks, `TD3Agent`, the replay buffer, `train_td3`, and the helpers `lqr_value_coef` and `lqr_q`. The checks test your functions in this order.

1. `lqr_dpg(theta, lqr, s0, horizon)` returns the gradient as a float. Loop over the policy's own trajectory and add $\gamma^t \cdot s_t \cdot \partial Q/\partial a$ at $a = \theta s_t$. The common mistake is differentiating $P$ as well.
2. `smoothed_target_action(mu_next, eps, sigma, noise_clip, max_action)` takes and returns `(N, act_dim)`. Use two `clamp` calls. The common mistake is clipping only the sum, so a huge draw moves the action by more than `noise_clip`.
3. `td3_target(r, terminated, q1_next, q2_next, gamma)` is one line with `torch.minimum`. The common mistake is dropping the `(1 - terminated)` factor.
4. `critic_loss(agent, batch, eps)` builds the target under `torch.no_grad()` from `agent.actor_target` and both target critics, then sums the two critics' mean squared errors. The common mistake is using the online actor or critics in the target.
5. `actor_loss(agent, s)` returns `-agent.critic1(s, agent.actor(s)).mean()`. The common mistake is using `min(Q1, Q2)`.
6. `soft_update(target, source, tau)` works in place under `torch.no_grad()`.
7. `td3_update(agent, batch, eps, step)` takes the critic step first. When `step % agent.policy_delay == 0` it also takes an actor step and three soft updates. Call `zero_grad()` before each `backward()`.

## Where this sits

DDPG and TD3 are a bridge. The actor is a policy that you train by gradient, as in Part 5, and the critic is a Q-function that you train from a replay buffer, as in Part 4. The table shows how the three families of this course compare. Each cell is a rule of thumb, not a guarantee.

| | Policy optimization (031 to 039) | Q-learning (026 to 030) | Actor-critic hybrids (040 to 042) |
|---|---|---|---|
| Data | On-policy: fresh samples after each update. IMPALA (038) tolerates slightly stale data with V-trace. | Off-policy: replay buffer | Off-policy: replay buffer |
| Stability | Steady with clipping or trust regions (035 to 036) | Fragile: the deadly triad (025) | Fragile: sensitive to the critic, so TD3 and SAC add twin critics and targets |
| Sample efficiency | Low: needs many environment steps | Medium to high | High on continuous control |
| Action spaces | Discrete or continuous | Discrete only: the policy takes an `argmax` | Continuous |

Read it as a trade. Policy optimization pays in samples to get steady updates. Q-learning reuses every sample but needs a finite set of actions to take a max over. The hybrids keep the replay buffer and get continuous actions by training an actor to climb the critic, and they pay in tuning. When you pick a method, ask first whether your actions are discrete, and second whether environment steps are cheap.

## Where this breaks

- **Exploration is bolted on.** A deterministic policy explores only through the noise you add. [Soft actor-critic](/lessons/042-soft-actor-critic) builds exploration into the objective instead.
- **The critic's slope must be meaningful.** The actor follows $\nabla_a Q$, which a neural network can get badly wrong away from its data. When a run collapses, plot the critic's values first.
- **Results vary widely.** DDPG is sensitive to learning rates, noise, and network size, and seeds differ a lot (Henderson et al., 2018).

## Further reading

The following papers introduce and analyze the methods in this chapter:

- [Silver et al. (2014), Deterministic policy gradient algorithms](https://proceedings.mlr.press/v32/silver14.html) proves the theorem and its link to the stochastic policy gradient.
- [Lillicrap et al. (2016), Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971) introduces DDPG.
- [Fujimoto, van Hoof, and Meger (2018), Addressing function approximation error in actor-critic methods](https://arxiv.org/abs/1802.09477) measures DDPG's overestimation and introduces TD3's three fixes. [Paper page](https://papers.ifkash.dev/td3/).
- [Henderson et al. (2018), Deep reinforcement learning that matters](https://arxiv.org/abs/1709.06560) shows how much results depend on seeds and implementation details. [Paper page](https://papers.ifkash.dev/henderson/).

## The exercise

You check the theorem, then build TD3. Write `lqr_dpg`, `smoothed_target_action`, `td3_target`, `critic_loss`, `actor_loss`, `soft_update`, and `td3_update`.

The checks compare `lqr_dpg` with finite differences of the simulated return on two systems, test the smoothing noise's two clips, and check the clipped double-Q target on hand-worked cases. They confirm that `critic_loss` uses the target networks and sends them no gradient, compare `actor_loss`'s gradient with the chain rule assembled by hand, check Polyak averaging exactly, and confirm that the actor and targets move only on delayed steps. The last check trains TD3 on Pendulum for 8,000 steps and expects the noise-free actor to average above $-800$. A random policy averages about $-1300$.
