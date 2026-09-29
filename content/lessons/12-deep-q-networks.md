---
title: Deep Q-networks
slug: 12-deep-q-networks
part: "Part 3 — Approximation and deep value methods"
summary: Build DQN's replay buffer, target network, and Huber loss, check each exactly, and train a network that balances CartPole.
minutes: 60
gpu: false
objectives:
  - Explain the two failures of naive neural Q-learning that experience replay and a target network address.
  - Implement a ring replay buffer, the DQN target with terminal masking, and Polyak target updates.
  - Derive the Huber loss's gradient and explain why it clips the TD error.
  - Train a DQN on CartPole, and say why truncation must not be stored as termination.
prereqs:
  - 07-temporal-difference-learning
  - 11-value-function-approximation
exercise: 12-deep-q-networks
---

# Deep Q-networks

Tile coding needs you to design the features by hand, and the number of
tiles explodes with the size of the input. A neural network learns its own
features. Feed it the raw state, and it outputs one Q-value per action.
That's the idea behind the deep Q-network, or DQN, which learned to play 49
Atari games from pixels with one architecture and one set of settings.

Plugging a network into Q-learning isn't enough, though. Consecutive
transitions look almost identical, so the network overfits to whatever it
saw last. The target moves every time the weights change, so the network
chases its own tail. The previous lesson showed that bootstrapping, function
approximation, and off-policy data together can diverge, and Q-learning with
a network has all three.

DQN keeps the plain Q-learning update and adds two stabilisers. A *replay
buffer* stores past transitions and trains on random samples from them. A
*target network* is a slowly updated copy of the network that computes the
targets. A Huber loss keeps a single bad target from yanking the weights. In
this lesson you build each piece, check it exactly, and then train a network
that balances a pole on a cart.

## What you already know

- Q-learning updates $Q(s_t, a_t)$ toward $r_t + \gamma \max_{a'} Q(s_{t+1}, a')$,
  whatever action the agent takes next. You built it in *Temporal-difference
  learning*.
- An *epsilon-greedy* policy takes a random action with probability
  $\varepsilon$ and the highest-valued action otherwise.
- Semi-gradient TD treats the target as a constant and steps along the
  gradient of the estimate only, as in *Value-function approximation*.
- Function approximation, bootstrapping, and off-policy updates together are
  the deadly triad.
- `rlenvs.CartPole` returns `(obs, reward, terminated, truncated)`. It pays 1
  per step and truncates at `max_steps`.

## Turn Q-learning into regression

A Q-network $Q_\theta(s, \cdot)$ takes a state and returns a vector of
Q-values, one per action. Here $\theta$ is the network's weights. For a
transition $(s_t, a_t, r_t, s_{t+1})$, Q-learning's target is a number, and
the network's job is to predict it. That's regression. The DQN loss over a
batch of $B$ transitions is

$$
\mathcal{L}(\theta) = \frac{1}{B} \sum_{i=1}^{B} \ell\big(y_i - Q_\theta(s_i, a_i)\big), \qquad y_i = r_i + \gamma (1 - d_i) \max_{a'} Q_{\bar\theta}(s'_i, a')
$$

Read it piece by piece:

- $Q_\theta(s_i, a_i)$ is the online network's estimate for the action that
  was taken.
- $y_i$ is the target: the reward plus the discounted best Q-value at the
  next state $s'_i$.
- $d_i$ is 1 when $s'_i$ is terminal, so the bootstrap term vanishes.
- $Q_{\bar\theta}$ is the *target network*, a copy with its own weights
  $\bar\theta$ that changes slowly.
- $\ell$ is a per-sample loss, the Huber loss described later in this lesson.

The target is computed without gradient. That's the semi-gradient update
from the previous lesson, made explicit in code:

```python
with torch.no_grad():
    y = r + gamma * (1 - term) * target_net(s2).max(dim=1).values
q_sa = q_net(s).gather(1, a[:, None]).squeeze(1)
loss = huber_loss(y - q_sa)
```

### Worked example

Take $r = 1$ and $\gamma = 0.9$, with target-network Q-values at the next
state of $[2.5, 1.5, 2.0]$. The best is 2.5, so
$y = 1 + 0.9 \times 2.5 = 3.25$. If that step ended the episode, $d = 1$ and
$y = 1$. If the next values were all negative, $[-1, -3, -2]$, and the
reward 0.5, the target would be $0.5 + 0.9 \times (-1) = -0.4$.

> [!WARNING]
> Only a real termination sets $d = 1$. When CartPole stops at `max_steps`,
> the pole is still up and the future is still worth something. Storing
> `truncated` as terminal teaches the network that balancing leads nowhere.

## Break correlations with a replay buffer

Picture studying for an exam by reading only the last page you wrote. You'd
remember that page perfectly and forget the rest. Training on transitions in
the order they arrive does that to a network. CartPole's successive states
differ by a hair, so a batch of consecutive steps is nearly one example
repeated.

A replay buffer, from Lin (1992), fixes this. It keeps the most recent $N$
transitions in a fixed-size array. Each update trains on a batch drawn
uniformly at random from the whole array, so one batch mixes moments from
many episodes. Each transition also gets reused in many updates, which makes
better use of expensive experience.

The array works as a *ring buffer*. A write index points at the next slot.
Each `add` writes there and moves the index forward by one, wrapping to 0 at
the end. Once the buffer is full, every write overwrites the oldest entry:

```python
self.s[self.next] = s               # and the same for a, r, s2, terminated
self.next = (self.next + 1) % self.capacity
self.size = min(self.size + 1, self.capacity)
idx = self.rng.integers(0, self.size, size=batch_size)
```

With capacity 5, after 7 adds the slots hold transitions 5, 6, 2, 3, and 4.
Transitions 0 and 1 are gone.

Replay has a cost. The data in the buffer came from older, more random
versions of the policy, so DQN is firmly off-policy. Q-learning tolerates
that, because its target doesn't depend on which action the behaviour policy
took next.

## Hold the target still with a target network

If the online network also computes the targets, every update moves the
thing it's aiming at. Raising $Q(s_t, a_t)$ usually raises $Q(s_{t+1}, \cdot)$
too, because the two states look alike, so the target runs away. A target
network slows the target down.

The Nature DQN (Mnih et al., 2015) copied the online weights into the target
network every 10,000 updates and held them fixed in between. A smoother
alternative, *Polyak averaging*, blends a little of the online weights in
after every update:

$$
\bar\theta \leftarrow (1 - \tau)\, \bar\theta + \tau\, \theta
$$

Read it as: keep a fraction $1 - \tau$ of the old target weights and mix in
a fraction $\tau$ of the online weights. With $\tau = 0.005$, the target
forgets old weights slowly. After 200 updates the original weights still
make up $0.995^{200} = 0.37$ of the mix. With $\tau = 1$ it's a full copy.

```python
with torch.no_grad():
    for p_target, p in zip(target_net.parameters(), q_net.parameters()):
        p_target.mul_(1 - tau).add_(tau * p)
```

> [!KEY]
> Replay breaks the correlation between consecutive samples. The target
> network stops the target from moving with every update. Together they
> turn Q-learning into something close to ordinary supervised regression.

## Clip the error with the Huber loss

Early in training, some targets are wildly wrong. A squared loss gives an
error of 30 a gradient 30 times larger than an error of 1, so one bad sample
can wreck the weights. The Nature paper clipped the error term to
$[-1, 1]$, which is the same as using the Huber loss with $\kappa = 1$:

$$
\ell_\kappa(u) = \begin{cases} \tfrac12 u^2 & |u| \le \kappa \\ \kappa\big(|u| - \tfrac12 \kappa\big) & |u| > \kappa \end{cases}
$$

Read it as: small errors cost half their square, like mean squared error.
Past $\kappa$, the cost grows in a straight line instead. The two pieces meet
smoothly at $|u| = \kappa$, where both equal $\kappa^2 / 2$ and both have
slope $\kappa$.

The derivative is the point of it. With $u = y - Q$ and the batch mean,

$$
\frac{\partial \mathcal{L}}{\partial u_i} = \frac{1}{B} \operatorname{clip}(u_i, -\kappa, \kappa)
$$

Read it as: each sample pushes with a force equal to its error, but never
more than $\kappa$. A huge error pushes no harder than a moderate one.
Because $u = y - Q$, the gradient with respect to $Q$ is the negative of
this.

```python
abs_u = u.abs()
loss = torch.where(abs_u <= kappa, 0.5 * u**2, kappa * (abs_u - 0.5 * kappa)).mean()
```

### Worked example

Take two TD errors, $u = [0.5, 3]$, with $\kappa = 1$.

- Losses: $0.5 \times 0.25 = 0.125$ and $1 \times (3 - 0.5) = 2.5$, mean
  1.3125.
- Gradient: $[0.5, 1] / 2 = [0.25, 0.5]$.
- With plain half-squared error, the gradient would be $[0.5, 3] / 2 =
  [0.25, 1.5]$. The large error's push is three times smaller under Huber.

> [!TIP]
> PyTorch's `nn.functional.smooth_l1_loss` with `beta=1` equals this Huber
> loss at $\kappa = 1$. Write your own first, so you know what it computes.

## Put the loop together

The training loop interleaves acting and learning. It follows these steps:

1. Observe $s_t$ and pick $a_t$ epsilon-greedily from $Q_\theta$. Decay
   $\varepsilon$ linearly from 1 to a small floor, so early data is random
   and later data follows the network.
2. Step the environment, and store $(s_t, a_t, r_t, s_{t+1}, \text{terminated})$
   in the buffer. Reset when the episode terminates or truncates.
3. Every few steps, once the buffer holds enough data, sample a batch, take
   one optimiser step on the loss, and Polyak-update the target network.

The exercise's defaults follow that shape at CartPole scale: 30,000 steps,
one update every 4 steps with batches of 256, $\tau = 0.005$, and $\gamma =
0.98$. The learning rate decays linearly to zero. A step size that shrinks
makes the final network settle rather than keep jumping between good and bad
policies.

> [!RECAP]
> - The ring buffer overwrites the oldest transition once full and samples
>   uniformly from what it holds.
> - The target is $r + \gamma (1 - d) \max_{a'} Q_{\bar\theta}(s', a')$,
>   computed with no gradient. Only termination sets $d = 1$.
> - The Huber loss is quadratic inside $\kappa$ and linear outside. Its
>   gradient is the error clipped to $[-\kappa, \kappa]$, over $B$.
> - Polyak averaging moves each target weight a fraction $\tau$ toward the
>   online weight.
> - Epsilon decays linearly, and `train_dqn` wires the pieces into one loop.

## Build it step by step

The checks test the pieces in this order. `greedy_return` is already
written for you.

1. **`ReplayBuffer`.** Preallocate numpy arrays of length `capacity` in
   `__init__`, plus a write index, a size, and `np.random.default_rng(seed)`.
   `add` writes one slot and advances the index modulo capacity. `sample`
   returns five arrays drawn at the same random indices. Common mistake:
   drawing a separate index for each field, which mixes up transitions. The
   first check catches that. The second runs a chi-squared test for
   uniformity.
2. **`dqn_targets(rewards, next_q, terminated, gamma)`.** One line under
   `torch.no_grad()`, with `next_q.max(dim=1).values`. Common mistake:
   `next_q.max()` without a dimension, which takes one maximum over the whole
   batch.
3. **`huber_loss(td_errors, kappa)`.** `torch.where` over the two pieces,
   then `.mean()`. Common mistake: writing the linear piece as
   $\kappa |u|$, which leaves a jump at $\kappa$. The checks compare the
   gradient against finite differences.
4. **`dqn_loss(q_net, target_net, batch, gamma, kappa)`.** Convert the batch
   with `torch.as_tensor`, build `y` from the target network, and select
   $Q(s, a)$ with `gather`. Common mistake: computing `y` from `q_net`, or
   outside `no_grad`, so gradients leak into the target.
5. **`soft_update(target_net, q_net, tau)`.** In-place `mul_` and `add_`
   under `no_grad`. Common mistake: `target_net = q_net`, which makes both
   names point at one network.
6. **`epsilon_at`, `make_q_network`, and `select_action`.** The schedule is
   a clamped linear interpolation. The network is an `nn.Sequential` with two
   hidden layers. For `select_action`, compute the Q-values, then draw
   `rng.random()` and compare it with $\varepsilon$.
7. **`train_dqn(env, ...)`.** Follow the recipe in the docstring. Common
   mistake: storing `terminated or truncated` in the buffer.

## Cost and scaling

DQN's cost splits into acting and learning. The following table shows how
each part grows:

| Part | Time | Memory |
|---|---|---|
| Acting | One forward pass per environment step | The network |
| Learning | One forward and backward pass on $B$ samples per update, plus a target forward pass | Two copies of the network |
| Replay buffer | Constant per add and per sampled row | $N$ transitions: $O(N \cdot \text{obs size})$ |

On CartPole the network has under 10,000 parameters, and each update costs
a fraction of a millisecond, so 7,250 updates take a few seconds. On Atari,
the Nature buffer held a million transitions of stacked $84 \times 84$ frames,
which is why real implementations store each frame once and rebuild stacks
on sampling. The number of updates per environment step is called the
*replay ratio*. Raising it trades compute for sample efficiency, up to the
point where the network overfits the buffer.

## Where this breaks

**Overestimation.** The max in the target picks whichever action's estimate
is noisiest upward, so Q-values drift high. *Maximisation bias and double
learning* showed this in tables, and the next lesson's Double DQN fixes it
for networks.

**Collapse after success.** Once the agent balances well, the buffer fills
with near-identical upright states. The network can forget how to recover
from a tilt, and its return drops. Larger buffers, slower learning rates, and
keeping some old data help.

**Sensitivity to settings.** DQN's result depends on the learning rate,
$\tau$, the buffer size, and the exploration schedule, and it varies a lot
between seeds. Report several seeds, never one.

**Short horizons.** With $\gamma = 0.98$, rewards more than about 50 steps
ahead barely count. The exercise's network keeps the pole up but often lets
the cart drift off the track after 150 to 300 steps. A larger $\gamma$ fixes
that in principle but needs more training.

**Continuous actions.** The max over actions needs a finite list. For
continuous actions, Part 5 replaces it with a learned actor.

## Further reading

- [Mnih et al., *Playing Atari with deep reinforcement learning*
  (2013)](https://arxiv.org/abs/1312.5602). The first DQN, with replay and no
  target network.
- [Mnih et al., *Human-level control through deep reinforcement learning*
  (2015)](https://doi.org/10.1038/nature14236). Adds the target network and
  error clipping, and reports results on 49 Atari games.
- [Lin, *Self-improving reactive agents based on reinforcement learning,
  planning and teaching* (1992)](https://doi.org/10.1007/BF00992699). The
  origin of experience replay.
- [Riedmiller, *Neural fitted Q iteration* (2005)](https://doi.org/10.1007/11564096_32).
  Batch Q-learning that refits a network to the whole stored data set at
  every iteration.
- [van Hasselt et al., *Deep reinforcement learning and the deadly triad*
  (2018)](https://arxiv.org/abs/1812.02648). Studies how target networks,
  multi-step returns, and other parts of DQN affect divergence in practice.
- [Fedus et al., *Revisiting fundamentals of experience replay*
  (2020)](https://arxiv.org/abs/2007.06700). How replay capacity and the
  replay ratio affect Q-learning agents, and why n-step returns help.

## The exercise

You build a DQN from its parts. Write the `ReplayBuffer` class, then
`dqn_targets`, `huber_loss`, `dqn_loss`, and `soft_update`. Write
`epsilon_at`, `make_q_network`, and `select_action` for acting, and wire
everything into `train_dqn`.

The checks verify that the buffer overwrites its oldest entries, keeps each
transition whole, and samples uniformly by a chi-squared test. They check the
targets and terminal masking against hand values, and the Huber loss's
values and gradient against the clipped-error formula and finite
differences. They confirm that `dqn_loss` matches a hand-built loss and sends
no gradient into the target network, and that `soft_update` mixes weights
exactly and keeps a separate copy. Finally, `train_dqn` must train a network
whose greedy policy averages a return above 100 over 10 CartPole episodes. A
random policy scores about 22, and the reference solution scores 283 on the
checked seed.
