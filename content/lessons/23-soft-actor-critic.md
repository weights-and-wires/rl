---
title: Soft actor-critic
slug: 23-soft-actor-critic
part: "Part 5 — Continuous control and maximum entropy"
summary: Compute a squashed Gaussian's log-probability, build SAC's reparameterised actor loss, and tune the temperature to a target entropy.
minutes: 55
gpu: false
objectives:
  - Derive the tanh-Gaussian log-probability with its change-of-variables term, compute it stably, and check it against a numerical density.
  - Build SAC's reparameterised actor loss and soft critic target, and verify the actor gradient against the exact expected loss.
  - Write the temperature loss, and show that its gradient steps drive the policy's entropy to a target.
prereqs:
  - 21-deterministic-policy-gradients
  - 22-maximum-entropy-rl
exercise: 23-soft-actor-critic
---

# Soft actor-critic

The previous two lessons built two halves of one method. Maximum-entropy
reinforcement learning gave you a better goal: earn reward, but stay as
random as you can while doing it. TD3 gave you a stable way to learn a
continuous-action actor off-policy, with twin critics and a replay buffer.
Soft actor-critic, or SAC, joins them. It's one of the most widely used
methods for continuous control.

Three details make SAC work, and each is a classic source of bugs. The
policy must output bounded actions, so it squashes a Gaussian with `tanh`,
and the log-probability needs a correction for that squash. The actor's
gradient must flow through a random sample, which needs the
reparameterisation trick. And the temperature that prices entropy is hard to
set by hand, so SAC learns it.

You build each of those pieces and check it against exact mathematics: a
density from a numerical change of variables, a gradient of an expectation
computed by quadrature, and a temperature that must settle where the entropy
hits its target.

## What you already know

- The maximum-entropy objective adds $\alpha$ times the policy's entropy to
  every reward, and its optimal policy is a Boltzmann distribution over soft
  action values.
- Soft policy iteration alternates soft policy evaluation with a step toward
  the Boltzmann policy of the current soft $Q$.
- TD3 trains two critics toward a target built from the smaller of two
  target critics, and moves the targets by Polyak averaging.
- The deterministic policy gradient moves an action along the critic's slope
  $\nabla_a Q(s, a)$.
- A Gaussian $\mathcal{N}(\mu, \sigma^2)$ has log-density $-\frac{1}{2}((u -
  \mu)/\sigma)^2 - \log \sigma - \frac{1}{2} \log 2\pi$.

## Turn soft policy iteration into three losses

SAC approximates soft policy iteration with neural networks and samples
(Haarnoja et al., 2018). It trains a stochastic actor $\pi_\theta(a \mid
s)$, two critics $Q_{\phi_1}$ and $Q_{\phi_2}$, and target copies of the
critics. Every update samples a batch of transitions from a replay buffer.

**The critics** do soft policy evaluation. Each regresses toward

$$
y = r + \gamma (1 - \text{terminated}) \Big( \min_{i = 1, 2} Q_{\phi'_i}(s', a') - \alpha \log \pi_\theta(a' \mid s') \Big), \qquad a' \sim \pi_\theta(\cdot \mid s')
$$

Read it as TD3's clipped double-Q target, plus an entropy bonus. The next
action $a'$ is a fresh sample from the *current* policy, not from a target
actor. SAC has no target actor. The term $-\alpha \log \pi(a' \mid s')$ is a
one-sample estimate of $\alpha$ times the entropy at the next state. The
critic loss is half the mean squared error, $\frac{1}{2}(Q_{\phi_i}(s, a) -
y)^2$, averaged over the batch.

**The actor** does soft policy improvement. The ideal step moves $\pi$ to the
Boltzmann policy $\exp(Q / \alpha) / Z$. A Gaussian can't be that policy
exactly, so SAC minimises the KL divergence to it instead. Drop the constant
$\log Z$ and multiply by $\alpha$, and the actor loss is

$$
\mathcal{L}_\pi(\theta) = \mathbb{E}_{s \sim \mathcal{D},\, a \sim \pi_\theta(\cdot \mid s)} \Big[ \alpha \log \pi_\theta(a \mid s) - \min_{i = 1, 2} Q_{\phi_i}(s, a) \Big]
$$

Read it as: prefer actions the critics score highly, and pay $\alpha$ for
every nat of certainty. Since $\mathbb{E}[-\log \pi]$ is the entropy,
minimising this loss maximises expected $Q$ plus $\alpha$ times entropy,
state by state. Haarnoja et al. (2018b) use the smaller critic here too.

**The temperature** is learned. That comes after the two pieces the actor
loss needs: a log-probability for bounded actions, and a gradient through a
sample.

## Squash a Gaussian into bounds

A Gaussian can output any real number. A motor accepts torques in $[-2, 2]$.
SAC draws $u \sim \mathcal{N}(\mu_\theta(s), \sigma_\theta(s)^2)$ and outputs
$a = a_{\max} \tanh(u)$, which always lands inside the bounds.

Squashing moves probability around. A small interval near $u = 0$ maps to an
interval of nearly the same width, but near the bounds, `tanh` flattens and a
wide range of $u$ piles into a thin sliver of $a$. The density of $a$ must go
up wherever the squash compresses. The *change of variables* formula says by
how much: divide the density of $u$ by $\lvert da / du \rvert$. For one
dimension, $da / du = a_{\max}(1 - \tanh^2 u)$. With $D$ independent action
dimensions,

$$
\log \pi(a \mid s) = \sum_{i=1}^{D} \Big[ \log \mathcal{N}\big(u_i; \mu_i, \sigma_i^2\big) - \log\big(1 - \tanh^2 u_i\big) - \log a_{\max} \Big]
$$

Read it as the Gaussian log-density of the pre-squash sample, minus the log
of how much the squash stretched each dimension. Always compute it from the
stored $u$. Recovering $u$ from $a$ with `atanh` blows up near the bounds,
where `tanh` rounds to exactly 1.

The term $\log(1 - \tanh^2 u)$ has its own trap. For $\lvert u \rvert$
above about 9 in float32, `torch.tanh(u)` rounds to 1, and the log is
$-\infty$. Since $1 - \tanh^2 u = 4 / (e^u + e^{-u})^2$, an equal form is

$$
\log\big(1 - \tanh^2 u\big) = 2\big(\log 2 - u - \operatorname{softplus}(-2u)\big)
$$

where $\operatorname{softplus}(x) = \log(1 + e^x)$. Read it as the same
number, computed from pieces that never overflow or round to zero. In
PyTorch:

```python
log_det = (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(dim=-1)
log_prob = gauss_log_prob - log_det - D * math.log(max_action)
```

> [!WARNING]
> Forgetting the squash correction is the most common SAC bug. The
> log-probability is wrong most at the bounds, so the entropy estimate is
> biased, the temperature chases the wrong target, and learning degrades
> without an error message.

### Worked example

Take one dimension with $\mu = 0$, $\sigma = 1$, and a draw of $u = 0.5$:

- Gaussian log-density: $-\frac{1}{2}(0.5)^2 - 0 - 0.9189 = -1.0439$.
- Squash: $\tanh 0.5 = 0.4621$, so $1 - \tanh^2 u = 0.7864$, and its log is
  $-0.2402$.
- With $a_{\max} = 1$: $\log \pi(a) = -1.0439 + 0.2402 = -0.8037$.
- With $a_{\max} = 2$: subtract $\log 2 = 0.6931$ more, for $-1.4968$.

The stable form agrees: $2(0.6931 - 0.5 - \operatorname{softplus}(-1)) =
2(0.6931 - 0.5 - 0.3133) = -0.2402$.

> [!KEY]
> A tanh-squashed Gaussian's log-probability is the Gaussian log-density of
> the pre-squash sample, minus $\sum_i \log(1 - \tanh^2 u_i)$, minus $D \log
> a_{\max}$. Compute it from $u$ with the softplus form.

## Differentiate through the sample

The actor loss is an expectation over actions that the actor itself samples.
How do you differentiate that? The REINFORCE route multiplies each sample's
loss by the score $\nabla_\theta \log \pi$, and it works with any loss, but
it's noisy.

SAC uses the *reparameterisation trick* instead. Write the sample as a
fixed function of the parameters and some parameter-free noise:

$$
a = f_\theta(\varepsilon; s) = a_{\max} \tanh\big(\mu_\theta(s) + \sigma_\theta(s)\, \varepsilon\big), \qquad \varepsilon \sim \mathcal{N}(0, I)
$$

Read it as: draw standard normal noise first, then shape it with the
network's mean and standard deviation. The randomness no longer depends on
$\theta$, so the expectation over $\varepsilon$ and the gradient in $\theta$
swap places. The loss becomes an average over fixed noise draws, and
autograd differentiates it directly. The gradient flows through $a$ into the
critic, as in the deterministic policy gradient, and through $\log \pi$ both
directly and through $a$.

In PyTorch, the only rule is to build the action from `eps` without
detaching anything:

```python
mean, log_std = policy(s)
a, log_prob = sample_action(mean, log_std, eps, policy.max_action)
loss = (alpha * log_prob - torch.minimum(q1(s, a), q2(s, a))).mean()
```

The estimate is unbiased: its average over noise draws is the true gradient
of the expected loss. The exercise checks that exactly. It computes the
expected loss by Gauss-Hermite quadrature, differentiates it by finite
differences, and compares with your autograd gradient over 20,000 stratified
noise draws.

## Tune the temperature automatically

The right $\alpha$ depends on the reward's scale, and it changes during
training. Haarnoja et al. (2018b) remove the guesswork. They ask for the
highest return subject to the policy's entropy staying at least a target
$\bar{\mathcal{H}}$, and treat $\alpha$ as the constraint's Lagrange
multiplier. The multiplier's step minimises

$$
J(\alpha) = \mathbb{E}_{a \sim \pi} \big[ -\alpha \log \pi(a \mid s) - \alpha \bar{\mathcal{H}} \big]
$$

Read it as $\alpha$ times the gap between the entropy $\mathcal{H} =
\mathbb{E}[-\log \pi]$ and the target. Implementations learn $\log \alpha$,
which keeps $\alpha$ positive. With $\alpha = e^{\log \alpha}$, the gradient
is

$$
\frac{\partial J}{\partial \log \alpha} = \alpha \big(\mathcal{H} - \bar{\mathcal{H}}\big)
$$

Read the sign. When the entropy is below the target, the gradient is
negative, so a descent step raises $\alpha$, and entropy becomes worth more.
When the entropy is above the target, $\alpha$ falls. The log-probabilities
are constants in this loss: the temperature step must not move the policy.
In PyTorch:

```python
loss = -(log_alpha.exp() * (log_prob + target_entropy).detach()).mean()
```

The paper's default target is $\bar{\mathcal{H}} = -\dim(\mathcal{A})$, so
$-1$ for Pendulum's single torque. It's a heuristic, not a derived value.
Differential entropy can be negative, so a negative target is fine.

### Worked example

Suppose a batch's log-probabilities average $-0.5$, so the entropy estimate
is $\mathcal{H} = 0.5$, and the target is $-1$. At $\alpha = 0.2$, the loss
is $0.2 \times (0.5 + 1) = 0.3$, and so is its gradient in $\log \alpha$.
The entropy is above target, so a descent step lowers $\alpha$.

Next, see where the loop settles. Against $Q(a) = -a^2/2$, the best Gaussian at
temperature $\alpha$ maximises $-\frac{1}{2}(\mu^2 + \sigma^2) + \alpha
\cdot \frac{1}{2} \log(2\pi e \sigma^2)$. That gives $\mu = 0$ and $\sigma^2
= \alpha$. Its entropy $\frac{1}{2}\log(2\pi e \alpha)$ equals a target of 1
nat at $\alpha^* = e^{2} / (2\pi e) = 0.4326$. Start from $\alpha = 1$, and
Adam steps on $J$ reach $0.4326$ within about 150 steps. The exercise runs
this loop.

> [!KEY]
> The temperature loss has gradient $\alpha(\mathcal{H} - \bar{\mathcal{H}})$
> in $\log \alpha$, so descent raises $\alpha$ when the policy is too
> certain and lowers it when the policy is too random.

## Assemble one SAC update

With the pieces in hand, one SAC update on a batch runs these steps:

1. Sample $a' \sim \pi_\theta(\cdot \mid s')$ with its log-probability, and
   build the critic target $y$ under `torch.no_grad()`.
2. Take one gradient step on both critics' squared errors.
3. Sample fresh actions at $s$ by reparameterisation, and take one step on
   the actor loss.
4. Take one step on the temperature loss, with the actor's log-probabilities
   detached.
5. Move both target critics toward the online critics with Polyak averaging.

The paper's defaults are Adam with learning rate $3 \times 10^{-4}$ for every
loss, batch size 256, two hidden layers of 256 units, $\gamma = 0.99$, and
$\tau = 0.005$. Use the value of $\alpha$ as a plain number in steps 1 and 3,
never as a tensor that carries gradient.

> [!RECAP]
> - `gaussian_log_prob` sums the Gaussian log-density over the last
>   dimension. At $u = 0.5$, $\mu = 0$, and $\sigma = 1$, it's $-1.0439$.
> - `tanh_log_det` is `2 * (log 2 - u - softplus(-2u))`, summed over the
>   last dimension.
> - `sample_action` builds `u` from `eps`, squashes it, and subtracts both
>   corrections from the Gaussian log-density.
> - The actor loss is `(alpha * log_prob - min(q1, q2)).mean()` at
>   reparameterised actions.
> - The temperature loss detaches `log_prob + target_entropy`, and its
>   gradient is $\alpha(\mathcal{H} - \bar{\mathcal{H}})$.

## Build it step by step

The checks test the functions in this order.

1. **`gaussian_log_prob(u, mean, log_std)`.** Takes three `(N, D)` tensors and
   returns `(N,)`. Standardise with `torch.exp(-log_std)`, then sum over
   `dim=-1`. Common mistake: forgetting the sum, which returns `(N, D)`.
2. **`tanh_log_det(u)`.** Returns `(N,)`. Use
   `torch.nn.functional.softplus`. The check feeds $u = \pm 20$ in float32,
   where the naive formula gives $-\infty$.
3. **`sample_action(mean, log_std, eps, max_action)`.** Returns
   `(a, log_prob)`. Build `u = mean + exp(log_std) * eps`, then subtract
   `tanh_log_det(u)` and `D * log(max_action)`. Common mistake: dropping the
   `max_action` term. The check compares `exp(log_prob)` with the derivative
   of the action's cumulative distribution function, integrates the density
   to 1, and compares a 2-D case with PyTorch's `TanhTransform`. A second
   check confirms gradients reach `mean` and `log_std`.
4. **`actor_loss(policy, critic1, critic2, s, eps, alpha)`.** Call
   `policy(s)`, sample with `policy.max_action`, and average `alpha *
   log_prob - min(q1, q2)`. Common mistake: detaching the action before the
   critics, which throws away the reparameterised gradient. The check
   compares your gradient with the exact expected loss's gradient.
5. **`critic_target(r, terminated, q1_next, q2_next, next_log_prob, gamma,
   alpha)`.** One line. Common mistake: adding $\alpha \log \pi$ instead of
   subtracting it.
6. **`temperature_loss(log_alpha, log_prob, target_entropy)`.** One line.
   Common mistake: forgetting `.detach()`, which lets the temperature step
   push on the policy. The last check runs 400 Adam steps and expects
   $\alpha$ within 1% of $0.4326$.

## Cost and scaling

The following table compares one SAC update with one TD3 update, for batch
size $N$ and networks of $W$ weights each.

| Piece | SAC | TD3 |
|---|---|---|
| Critic target | Actor forward at $s'$, two target critics | Target actor forward at $s'$, two target critics |
| Critic step | Two critics, forward and backward | The same |
| Actor step | Every update, through two critics | Every second update, through one critic |
| Temperature step | $O(N)$, one scalar | None |
| Networks kept | Actor, two critics, two target critics | Actor, two critics, three targets |

Each piece costs $O(NW)$, so SAC's update costs about the same as TD3's,
with one fewer target network and a more frequent actor step. The squash
correction and the log-probability add $O(ND)$ for $D$ action dimensions,
which is negligible. Every check in the exercise runs in well under a
second.

## Where this breaks

**The squash correction is easy to get wrong.** Summing it over the wrong
axis, dropping the $a_{\max}$ term, or computing it with `atanh` of a clipped
action all bias the log-probability. The temperature then chases the wrong
entropy. Test the density against a histogram, as the exercise's **Run**
block does.

**The temperature can run away.** If the entropy estimate is biased, or the
target is unreachable, $\log \alpha$ drifts without bound. Clamp
`log_std`, and watch $\alpha$ in your logs.

**The target entropy is a heuristic.** $-\dim(\mathcal{A})$ works well on
standard benchmarks, and it isn't derived from anything. For discrete
actions, SAC needs a different target, such as a fraction of $\log
\lvert \mathcal{A} \rvert$ (Christodoulou, 2019).

**Gaussian policies are unimodal.** The Boltzmann policy SAC chases can have
several peaks. A tanh-Gaussian has one, so the KL projection picks one mode.
Normalising flows and diffusion policies relax this at extra cost.

**It's still off-policy with function approximation.** SAC inherits the
deadly triad of bootstrapping, off-policy data, and function approximation.
Twin critics and target networks make it stable in practice, not in theory.

## Further reading

The following papers introduce SAC and the ideas it relies on:

- [Haarnoja, Zhou, Abbeel, and Levine (2018), Soft actor-critic: off-policy maximum entropy deep reinforcement learning with a stochastic actor](https://arxiv.org/abs/1801.01290)
  derives soft policy iteration and introduces SAC, with the tanh correction
  in its appendix.
- [Haarnoja et al. (2018b), Soft actor-critic algorithms and applications](https://arxiv.org/abs/1812.05905)
  adds automatic temperature tuning, twin critics, and drops the separate
  value network.
- [Kingma and Welling (2014), Auto-encoding variational Bayes](https://arxiv.org/abs/1312.6114)
  introduces the reparameterisation trick for gradients through samples.
- [Fujimoto, van Hoof, and Meger (2018), Addressing function approximation error in actor-critic methods](https://arxiv.org/abs/1802.09477)
  introduces the clipped double-Q target SAC borrows.
- [Christodoulou (2019), Soft actor-critic for discrete action settings](https://arxiv.org/abs/1910.07207)
  adapts SAC and its entropy target to discrete actions.
- [Levine (2018), Reinforcement learning and control as probabilistic inference](https://arxiv.org/abs/1805.00909)
  explains why the actor's KL projection is variational inference.

## The exercise

You build SAC's mathematical core. Write `gaussian_log_prob` and
`tanh_log_det`, then `sample_action` for a reparameterised tanh-Gaussian
sample and its corrected log-probability. Write `actor_loss` for the
reparameterised actor loss, `critic_target` for the soft clipped double-Q
target, and `temperature_loss` for the temperature's dual loss. The
`TanhGaussianPolicy` network is given.

The checks compare the Gaussian log-density with PyTorch's `Normal`, and the
stable squash term with the naive formula and at $u = \pm 20$ in float32.
They compare `exp(log_prob)` with the density from a numerical change of
variables, integrate it to 1, and check a 2-D case against `TanhTransform`.
They confirm that gradients reach the mean and log standard deviation, that
the actor loss uses the smaller critic, and that its gradient matches the
exact expected loss's. They check the critic target on hand-worked cases,
the temperature loss's gradient $\alpha(\mathcal{H} - \bar{\mathcal{H}})$
and its detached log-probabilities, and that the temperature settles where
the entropy meets its target.
