---
title: Deep Q-networks
slug: 025-deep-q-networks
part: "Part 4 — Deep value methods"
summary: Train a neural Q-function stably with a replay buffer, a target network, and a Huber loss, then balance CartPole with it.
minutes: 40
gpu: false
objectives:
  - Explain the two failures of naive neural Q-learning that experience replay and a target network address.
  - Implement a ring replay buffer, the DQN target with terminal masking, and Polyak target updates.
  - Derive the Huber loss's gradient and explain why it clips the TD error.
  - Train a DQN on CartPole, and say why truncation must not be stored as termination.
prereqs:
  - 017-sarsa-and-q-learning
  - 023-function-approximation
  - 024-the-deadly-triad
exercise: 025-deep-q-networks
---

# Deep Q-networks

Tile coding asks you to design the features by hand, and the number of tiles explodes with the size of the input. A neural network learns its own features. Feed it the raw state, and it outputs one Q-value per action. That is the deep Q-network, or DQN, which learned to play 49 Atari games from pixels with one architecture and one set of settings.

Plugging a network into Q-learning and pressing go does not work. [The deadly triad](/lessons/024-the-deadly-triad) showed that function approximation, bootstrapping, and off-policy data together can diverge, and Q-learning with a network has all three. The question is how to keep the update you know and make it stable.

DQN's answer is three small pieces: a buffer of past steps to train on, a slowly moving copy of the network to compute targets, and a loss that does not let one bad target wreck the weights. You will build each one and check it exactly, and then balance a pole on a cart.

## What you already know

- Q-learning moves $Q(s_t, a_t)$ toward $r_t + \gamma \max_{a'} Q(s_{t+1}, a')$, whatever action the agent takes next, as in [SARSA and Q-learning](/lessons/017-sarsa-and-q-learning).
- An *epsilon-greedy* policy takes a random action with probability $\varepsilon$ and the highest-valued action otherwise.
- Semi-gradient TD treats the target as a constant and steps along the gradient of the estimate only, as in [Value-function approximation](/lessons/023-function-approximation).
- Function approximation, bootstrapping, and off-policy updates together can diverge.
- `rlenvs.CartPole` returns `(obs, reward, terminated, truncated)`. It pays 1 per step and truncates at `max_steps`.

## Two things go wrong

A Q-network $Q_\theta(s, \cdot)$ takes a state and returns a vector of Q-values, one per action. Here $\theta$ is the network's weights. The obvious plan is to run Q-learning as before: after each real step, compute the target and take one gradient step toward it. Two problems appear.

First, consecutive steps are almost the same example. CartPole's successive states differ by a hair, so a run of 32 steps is nearly one data point repeated 32 times. Training on that is like studying for an exam by reading only the last page you wrote: you remember it perfectly and forget the rest.

Second, the target moves. The target $r + \gamma \max_{a'} Q_\theta(s', a')$ uses the same weights you are updating. Raising $Q(s_t, a_t)$ usually raises $Q(s_{t+1}, \cdot)$ too, because the two states look alike, so the target runs away from the estimate. The network chases its own tail.

```figure
<svg viewBox="0 0 600 275" role="img" aria-label="DQN data flow: the environment feeds a replay buffer, batches go to the online and target networks, and a Huber loss updates the online network">
<defs><marker id="c025-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" class="fill-muted"/></marker></defs>
<rect x="20.0" y="30.0" width="120.0" height="50.0" class="fill-soft-a stroke-a" rx="6"/>
<text x="80.0" y="59.0" text-anchor="middle">environment</text>
<rect x="220.0" y="30.0" width="140.0" height="50.0" class="fill-soft-b stroke-b" rx="6"/>
<text x="290.0" y="51.0" text-anchor="middle">online network</text>
<text x="290.0" y="67.0" text-anchor="middle">Q(s, a)</text>
<rect x="430.0" y="30.0" width="140.0" height="50.0" class="fill-sunk stroke-muted" rx="6"/>
<text x="500.0" y="51.0" text-anchor="middle">Huber loss</text>
<text x="500.0" y="67.0" text-anchor="middle">of y - Q(s, a)</text>
<rect x="20.0" y="150.0" width="140.0" height="56.0" class="fill-sunk stroke-muted" rx="6"/>
<text x="90.0" y="174.0" text-anchor="middle">replay buffer</text>
<text x="90.0" y="190.0" text-anchor="middle">last N steps</text>
<rect x="220.0" y="150.0" width="140.0" height="56.0" class="fill-soft-d stroke-d" rx="6"/>
<text x="290.0" y="174.0" text-anchor="middle">batch of B</text>
<text x="290.0" y="190.0" text-anchor="middle">(s, a, r, s', d)</text>
<rect x="430.0" y="150.0" width="140.0" height="56.0" class="fill-soft-c stroke-c" rx="6"/>
<text x="500.0" y="174.0" text-anchor="middle">target network</text>
<text x="500.0" y="190.0" text-anchor="middle">y = r + gamma max Q</text>
<line x1="140.0" y1="45.0" x2="220.0" y2="45.0" class="stroke-muted" marker-end="url(#c025-arrow)"/>
<text x="180.0" y="38.0" class="fig-small" text-anchor="middle">action</text>
<line x1="220.0" y1="66.0" x2="140.0" y2="66.0" class="stroke-muted" marker-end="url(#c025-arrow)"/>
<text x="180.0" y="82.0" class="fig-small" text-anchor="middle">state</text>
<line x1="80.0" y1="80.0" x2="80.0" y2="150.0" class="stroke-muted" marker-end="url(#c025-arrow)"/>
<text x="52.0" y="120.0" class="fig-small" text-anchor="middle">store</text>
<line x1="160.0" y1="178.0" x2="220.0" y2="178.0" class="stroke-muted" marker-end="url(#c025-arrow)"/>
<text x="190.0" y="170.0" class="fig-small" text-anchor="middle">sample</text>
<line x1="360.0" y1="178.0" x2="430.0" y2="178.0" class="stroke-muted" marker-end="url(#c025-arrow)"/>
<text x="395.0" y="170.0" class="fig-small" text-anchor="middle">s'</text>
<line x1="500.0" y1="150.0" x2="500.0" y2="80.0" class="stroke-muted" marker-end="url(#c025-arrow)"/>
<text x="512.0" y="120.0" class="fig-small" text-anchor="middle">y</text>
<line x1="290.0" y1="150.0" x2="290.0" y2="80.0" class="stroke-muted" marker-end="url(#c025-arrow)"/>
<text x="312.0" y="120.0" class="fig-small" text-anchor="middle">s, a</text>
<line x1="430.0" y1="55.0" x2="360.0" y2="55.0" class="stroke-muted" marker-end="url(#c025-arrow)"/>
<text x="300.0" y="240.0" class="fig-small" text-anchor="middle">Loss gradients flow into the online network only.</text>
<text x="300.0" y="258.0" class="fig-small" text-anchor="middle">After each update the target network moves a fraction tau toward the online one.</text>
</svg>
Figure 1. The DQN loop. A replay buffer fixes the first problem by serving random batches. A target network fixes the second by holding the target still between updates. Notice that gradients reach only the online network.
```

## Turn Q-learning into regression

For a transition $(s_t, a_t, r_t, s_{t+1})$, Q-learning's target is a number, and the network's job is to predict it. That is regression. The DQN loss over a batch of $B$ transitions is

$$
\mathcal{L}(\theta) = \frac{1}{B} \sum_{i=1}^{B} \ell\big(y_i - Q_\theta(s_i, a_i)\big), \qquad y_i = r_i + \gamma (1 - d_i) \max_{a'} Q_{\bar\theta}(s'_i, a')
$$

Read it piece by piece:

- $Q_\theta(s_i, a_i)$ is the online network's estimate for the action that was taken.
- $y_i$ is the target: the reward plus the discounted best Q-value at the next state $s'_i$.
- $d_i$ is 1 when $s'_i$ is terminal, so the bootstrap term vanishes.
- $Q_{\bar\theta}$ is the *target network*, a copy with its own weights $\bar\theta$ that changes slowly.
- $\ell$ is a per-sample loss, the Huber loss described later in this chapter.

The target is computed without gradient. That is the semi-gradient update from [Value-function approximation](/lessons/023-function-approximation), made explicit in code:

```python
with torch.no_grad():
    y = r + gamma * (1 - term) * target_net(s2).max(dim=1).values
q_sa = q_net(s).gather(1, a[:, None]).squeeze(1)
loss = huber_loss(y - q_sa)
```

Here is a worked example. Take $r = 1$ and $\gamma = 0.9$, with target-network Q-values at the next state of $[2.5, 1.5, 2.0]$. The best is 2.5, so $y = 1 + 0.9 \times 2.5 = 3.25$. If that step ended the episode, $d = 1$ and $y = 1$. If the next values were all negative, $[-1, -3, -2]$, and the reward 0.5, the target would be $0.5 + 0.9 \times (-1) = -0.4$.

> [!WARNING]
> Only a real termination sets $d = 1$. When CartPole stops at `max_steps`, the pole is still up and the future is still worth something. Storing `truncated` as terminal teaches the network that balancing leads nowhere.

## Break correlations with a replay buffer

A replay buffer (Lin, 1992) keeps the most recent $N$ transitions in a fixed-size array. Each update trains on a batch drawn uniformly at random from the whole array, so one batch mixes moments from many episodes, and the batch no longer looks like one repeated example. Each transition is also reused in many updates, which makes better use of expensive experience.

The array works as a *ring buffer*. A write index points at the next slot. Each `add` writes there and moves the index forward by one, wrapping to 0 at the end. Once the buffer is full, every write overwrites the oldest entry:

```python
self.s[self.next] = s               # and the same for a, r, s2, terminated
self.next = (self.next + 1) % self.capacity
self.size = min(self.size + 1, self.capacity)
idx = self.rng.integers(0, self.size, size=batch_size)
```

With capacity 5, after 7 adds the slots hold transitions 5, 6, 2, 3, and 4. Transitions 0 and 1 are gone.

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="A ring buffer of five slots after seven adds: slots hold transitions 5, 6, 2, 3, 4 and the next write goes to slot 2">
<defs><marker id="c025-arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" class="fill-muted"/></marker></defs>
<text x="20.0" y="22.0" class="fig-bold" text-anchor="start">Capacity 5, after adding transitions 0 to 6</text>
<rect x="60.0" y="60.0" width="80.0" height="50.0" class="fill-soft-b stroke-b" rx="6"/>
<text x="100.0" y="90.0" class="fig-bold" text-anchor="middle">#5</text>
<text x="100.0" y="130.0" class="fig-small" text-anchor="middle">slot 0</text>
<rect x="160.0" y="60.0" width="80.0" height="50.0" class="fill-soft-b stroke-b" rx="6"/>
<text x="200.0" y="90.0" class="fig-bold" text-anchor="middle">#6</text>
<text x="200.0" y="130.0" class="fig-small" text-anchor="middle">slot 1</text>
<rect x="260.0" y="60.0" width="80.0" height="50.0" class="fill-soft-a stroke-a" rx="6"/>
<text x="300.0" y="90.0" class="fig-bold" text-anchor="middle">#2</text>
<text x="300.0" y="130.0" class="fig-small" text-anchor="middle">slot 2</text>
<rect x="360.0" y="60.0" width="80.0" height="50.0" class="fill-soft-a stroke-a" rx="6"/>
<text x="400.0" y="90.0" class="fig-bold" text-anchor="middle">#3</text>
<text x="400.0" y="130.0" class="fig-small" text-anchor="middle">slot 3</text>
<rect x="460.0" y="60.0" width="80.0" height="50.0" class="fill-soft-a stroke-a" rx="6"/>
<text x="500.0" y="90.0" class="fig-bold" text-anchor="middle">#4</text>
<text x="500.0" y="130.0" class="fig-small" text-anchor="middle">slot 4</text>
<line x1="260.0" y1="175.0" x2="260.0" y2="118.0" class="stroke-muted" marker-end="url(#c025-arrow2)"/>
<text x="260.0" y="192.0" class="fig-small" text-anchor="middle">next write goes here (slot 2)</text>
<text x="20.0" y="225.0" class="fig-small" text-anchor="start">Transitions 5 and 6 (orange) wrapped around and overwrote 0 and 1.</text>
<text x="20.0" y="245.0" class="fig-small" text-anchor="start">Transitions 2, 3, and 4 survive. Sampling draws from slots 0 to 4 uniformly.</text>
</svg>
Figure 2. A ring buffer of capacity 5 after seven adds. The write index wrapped around, so transitions 5 and 6 replaced 0 and 1. Notice that the buffer always holds the most recent transitions, so old behavior fades out as the policy improves.
```

Replay has a cost. The data in the buffer came from older, more random versions of the policy, so DQN is firmly off-policy. Q-learning tolerates that, because its target does not depend on which action the behavior policy took next.

## Hold the target still with a target network

If the online network also computes the targets, every update moves the thing it is aiming at. A target network slows the target down. The Nature DQN (Mnih et al., 2015) copied the online weights into the target network every 10,000 updates and held them fixed in between. A smoother alternative, *Polyak averaging*, blends a little of the online weights in after every update:

$$
\bar\theta \leftarrow (1 - \tau)\, \bar\theta + \tau\, \theta
$$

Read it aloud: keep a fraction $1 - \tau$ of the old target weights and mix in a fraction $\tau$ of the online weights. With $\tau = 0.005$, the target forgets old weights slowly. After 200 updates the original weights still make up $0.995^{200} = 0.37$ of the mix. With $\tau = 1$ it is a full copy.

```python
with torch.no_grad():
    for p_target, p in zip(target_net.parameters(), q_net.parameters()):
        p_target.mul_(1 - tau).add_(tau * p)
```

> [!KEY]
> Replay breaks the correlation between consecutive samples. The target network stops the target from moving with every update. Together they turn Q-learning into something close to ordinary supervised regression.

## Clip the error with the Huber loss

Early in training, some targets are wildly wrong. A squared loss gives an error of 30 a gradient 30 times larger than an error of 1, so one bad sample can wreck the weights. The Nature paper clipped the error term to $[-1, 1]$, which is the same as using the Huber loss with $\kappa = 1$:

$$
\ell_\kappa(u) = \begin{cases} \tfrac12 u^2 & |u| \le \kappa \\ \kappa\big(|u| - \tfrac12 \kappa\big) & |u| > \kappa \end{cases}
$$

Read it aloud: small errors cost half their square, like mean squared error. Past $\kappa$, the cost grows in a straight line instead. The two pieces meet smoothly at $|u| = \kappa$, where both equal $\kappa^2 / 2$ and both have slope $\kappa$.

The derivative is the point of it. With $u = y - Q$ and the batch mean,

$$
\frac{\partial \mathcal{L}}{\partial u_i} = \frac{1}{B} \operatorname{clip}(u_i, -\kappa, \kappa)
$$

Read it aloud: each sample pushes with a force equal to its error, but never more than $\kappa$. A huge error pushes no harder than a moderate one. Because $u = y - Q$, the gradient with respect to $Q$ is the negative of this.

```python
abs_u = u.abs()
loss = torch.where(abs_u <= kappa, 0.5 * u**2, kappa * (abs_u - 0.5 * kappa)).mean()
```

Take two TD errors, $u = [0.5, 3]$, with $\kappa = 1$. The losses are $0.5 \times 0.25 = 0.125$ and $1 \times (3 - 0.5) = 2.5$, with mean 1.3125. The gradient is $[0.5, 1] / 2 = [0.25, 0.5]$. With plain half-squared error, the gradient would be $[0.5, 3] / 2 = [0.25, 1.5]$, so the large error's push is three times smaller under Huber.

```figure
<svg viewBox="0 0 600 275" role="img" aria-label="Half squared error against Huber loss with kappa = 1: they agree inside 1 and Huber grows linearly outside, so an error of 3 costs 2.5 instead of 4.5">
<line x1="70.0" y1="230.0" x2="570.0" y2="230.0" class="stroke-grid"/>
<text x="62.0" y="234.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="182.5" x2="570.0" y2="182.5" class="stroke-grid"/>
<text x="62.0" y="186.5" class="fig-small" text-anchor="end">2</text>
<line x1="70.0" y1="135.0" x2="570.0" y2="135.0" class="stroke-grid"/>
<text x="62.0" y="139.0" class="fig-small" text-anchor="end">4</text>
<line x1="70.0" y1="87.5" x2="570.0" y2="87.5" class="stroke-grid"/>
<text x="62.0" y="91.5" class="fig-small" text-anchor="end">6</text>
<line x1="70.0" y1="40.0" x2="570.0" y2="40.0" class="stroke-grid"/>
<text x="62.0" y="44.0" class="fig-small" text-anchor="end">8</text>
<text x="70.0" y="248.0" class="fig-small" text-anchor="middle">-4</text>
<text x="195.0" y="248.0" class="fig-small" text-anchor="middle">-2</text>
<text x="320.0" y="248.0" class="fig-small" text-anchor="middle">0</text>
<text x="445.0" y="248.0" class="fig-small" text-anchor="middle">2</text>
<text x="570.0" y="248.0" class="fig-small" text-anchor="middle">4</text>
<text x="300.0" y="270.0" text-anchor="middle">TD error u = y - Q(s, a)</text>
<text x="62.0" y="22.0" class="fig-small" text-anchor="start">loss</text>
<polyline points="70.0,40.0 76.2,49.4 82.5,58.5 88.7,67.4 95.0,76.1 101.2,84.5 107.5,92.7 113.8,100.7 120.0,108.4 126.2,115.9 132.5,123.1 138.8,130.1 145.0,136.9 151.2,143.4 157.5,149.7 163.8,155.8 170.0,161.6 176.2,167.2 182.5,172.5 188.8,177.6 195.0,182.5 201.2,187.1 207.5,191.5 213.8,195.7 220.0,199.6 226.2,203.3 232.5,206.7 238.8,209.9 245.0,212.9 251.2,215.6 257.5,218.1 263.8,220.4 270.0,222.4 276.2,224.2 282.5,225.7 288.8,227.0 295.0,228.1 301.2,228.9 307.5,229.5 313.8,229.9 320.0,230.0 326.2,229.9 332.5,229.5 338.8,228.9 345.0,228.1 351.2,227.0 357.5,225.7 363.8,224.2 370.0,222.4 376.2,220.4 382.5,218.1 388.8,215.6 395.0,212.9 401.2,209.9 407.5,206.7 413.8,203.3 420.0,199.6 426.2,195.7 432.5,191.5 438.8,187.1 445.0,182.5 451.2,177.6 457.5,172.5 463.8,167.2 470.0,161.6 476.2,155.8 482.5,149.7 488.8,143.4 495.0,136.9 501.2,130.1 507.5,123.1 513.8,115.9 520.0,108.4 526.2,100.7 532.5,92.7 538.8,84.5 545.0,76.1 551.2,67.4 557.5,58.5 563.8,49.4 570.0,40.0" class="stroke-a dashed" fill="none"/>
<polyline points="70.0,146.9 76.2,149.2 82.5,151.6 88.7,154.0 95.0,156.4 101.2,158.8 107.5,161.1 113.8,163.5 120.0,165.9 126.2,168.2 132.5,170.6 138.8,173.0 145.0,175.4 151.2,177.8 157.5,180.1 163.8,182.5 170.0,184.9 176.2,187.2 182.5,189.6 188.8,192.0 195.0,194.4 201.2,196.8 207.5,199.1 213.8,201.5 220.0,203.9 226.2,206.2 232.5,208.6 238.8,211.0 245.0,213.4 251.2,215.8 257.5,218.1 263.8,220.4 270.0,222.4 276.2,224.2 282.5,225.7 288.8,227.0 295.0,228.1 301.2,228.9 307.5,229.5 313.8,229.9 320.0,230.0 326.2,229.9 332.5,229.5 338.8,228.9 345.0,228.1 351.2,227.0 357.5,225.7 363.8,224.2 370.0,222.4 376.2,220.4 382.5,218.1 388.8,215.8 395.0,213.4 401.2,211.0 407.5,208.6 413.8,206.2 420.0,203.9 426.2,201.5 432.5,199.1 438.8,196.8 445.0,194.4 451.2,192.0 457.5,189.6 463.8,187.2 470.0,184.9 476.2,182.5 482.5,180.1 488.8,177.8 495.0,175.4 501.2,173.0 507.5,170.6 513.8,168.2 520.0,165.9 526.2,163.5 532.5,161.1 538.8,158.8 545.0,156.4 551.2,154.0 557.5,151.6 563.8,149.2 570.0,146.9" class="stroke-b" fill="none"/>
<circle cx="507.5" cy="170.6" r="5" class="fill-b"/>
<circle cx="507.5" cy="123.1" r="5" class="fill-a"/>
<text x="497.5" y="115.1" class="fig-small" text-anchor="end">squared: 4.5</text>
<text x="513.5" y="194.6" class="fig-small" text-anchor="end">Huber: 2.5</text>
<line x1="110.0" y1="52.0" x2="140.0" y2="52.0" class="stroke-a dashed"/>
<text x="148.0" y="56.0" class="fig-small" text-anchor="start">half squared error</text>
<line x1="110.0" y1="70.0" x2="140.0" y2="70.0" class="stroke-b"/>
<text x="148.0" y="74.0" class="fig-small" text-anchor="start">Huber, kappa = 1</text>
</svg>
Figure 3. Half squared error and Huber loss with kappa = 1. They agree for errors inside 1. Notice that at an error of 3, squared error costs 4.5 and keeps curving up, while Huber costs 2.5 and grows linearly, so a single bad target cannot dominate a batch.
```

> [!TIP]
> PyTorch's `nn.functional.smooth_l1_loss` with `beta=1` equals this Huber loss at $\kappa = 1$. Write your own first, so you know what it computes.

## Put the loop together

The training loop interleaves acting and learning:

1. Observe $s_t$ and pick $a_t$ epsilon-greedily from $Q_\theta$. Decay $\varepsilon$ linearly from 1 to a small floor, so early data is random and later data follows the network.
2. Step the environment, and store $(s_t, a_t, r_t, s_{t+1}, \text{terminated})$ in the buffer. Reset when the episode terminates or truncates.
3. Every few steps, once the buffer holds enough data, sample a batch, take one optimizer step on the loss, and Polyak-update the target network.

The exercise's defaults follow that shape at CartPole scale: 30,000 steps, one update every 4 steps with batches of 256, $\tau = 0.005$, and $\gamma = 0.98$. The learning rate decays linearly to zero, so the final network settles.

> [!RECAP]
> - The ring buffer overwrites the oldest transition once full and samples uniformly from what it holds.
> - The target is $r + \gamma (1 - d) \max_{a'} Q_{\bar\theta}(s', a')$, computed with no gradient. Only termination sets $d = 1$.
> - The Huber loss is quadratic inside $\kappa$ and linear outside. Its gradient is the error clipped to $[-\kappa, \kappa]$, over $B$.
> - Polyak averaging moves each target weight a fraction $\tau$ toward the online weight.
> - Epsilon decays linearly, and `train_dqn` wires the pieces into one loop.

## Build it step by step

The checks test the pieces in this order. `greedy_return` is already written.

1. **`ReplayBuffer`.** Preallocate numpy arrays of length `capacity` in `__init__`, plus a write index, a size, and `np.random.default_rng(seed)`. `add` writes one slot and advances the index modulo capacity. `sample` returns five arrays drawn at the same random indices. Common mistake: drawing a separate index for each field, which mixes up transitions.
2. **`dqn_targets(rewards, next_q, terminated, gamma)`.** One line under `torch.no_grad()`, with `next_q.max(dim=1).values`. Common mistake: `next_q.max()` without a dimension, which takes one maximum over the whole batch.
3. **`huber_loss(td_errors, kappa)`.** `torch.where` over the two pieces, then `.mean()`. Common mistake: writing the linear piece as $\kappa |u|$, which leaves a jump at $\kappa$. The checks compare the gradient against finite differences.
4. **`dqn_loss(q_net, target_net, batch, gamma, kappa)`.** Convert the batch with `torch.as_tensor`, build `y` from the target network, and select $Q(s, a)$ with `gather`. Common mistake: computing `y` from `q_net`, or outside `no_grad`, so gradients leak into the target.
5. **`soft_update(target_net, q_net, tau)`.** In-place `mul_` and `add_` under `no_grad`. Common mistake: `target_net = q_net`, which makes both names point at one network.
6. **`epsilon_at`, `make_q_network`, and `select_action`.** The schedule is a clamped linear interpolation. The network is an `nn.Sequential` with two hidden layers. For `select_action`, compute the Q-values, then draw `rng.random()` and compare it with $\varepsilon$.
7. **`train_dqn(env, ...)`.** Follow the recipe in the docstring. Common mistake: storing `terminated or truncated` in the buffer.

## Where this breaks

**Overestimation.** The max in the target picks whichever action's estimate is noisiest upward, so Q-values drift high. [Maximization bias and double learning](/lessons/018-maximization-bias) showed this in tables, and [Double and dueling DQN](/lessons/026-double-and-dueling-dqn) fixes it for networks.

**Collapse after success.** Once the agent balances well, the buffer fills with near-identical upright states. The network can forget how to recover from a tilt, and its return drops. Larger buffers, lower learning rates, and keeping some old data help.

**Sensitivity to settings.** DQN's result depends on the learning rate, $\tau$, the buffer size, and the exploration schedule, and it varies a lot between seeds. Report several seeds, never one.

**Short horizons.** With $\gamma = 0.98$, rewards more than about 50 steps ahead barely count, so the exercise's network often lets the cart drift off the track after 150 to 300 steps.

**Continuous actions.** The max over actions needs a finite list. For continuous actions, [Deterministic policy gradients](/lessons/037-deterministic-policy-gradients) replaces it with a learned actor.

## Further reading

- [Mnih and colleagues, "Playing Atari with deep reinforcement learning" (2013)](https://arxiv.org/abs/1312.5602). The first DQN, with replay and no target network.
- [Mnih and colleagues, "Human-level control through deep reinforcement learning" (2015)](https://doi.org/10.1038/nature14236). Adds the target network and error clipping, and reports results on 49 Atari games.
- [Lin, "Self-improving reactive agents based on reinforcement learning, planning and teaching" (1992)](https://doi.org/10.1007/BF00992699). The origin of experience replay.
- [van Hasselt and colleagues, "Deep reinforcement learning and the deadly triad" (2018)](https://arxiv.org/abs/1812.02648). Studies how target networks, multi-step returns, and other parts of DQN affect divergence in practice.

## The exercise

You build a DQN from its parts. Write the `ReplayBuffer` class, then `dqn_targets`, `huber_loss`, `dqn_loss`, and `soft_update`. Write `epsilon_at`, `make_q_network`, and `select_action` for acting, and wire everything into `train_dqn`.

The checks verify that the buffer overwrites its oldest entries and samples uniformly, the targets and terminal masking against hand values, and the Huber loss's gradient against finite differences. They confirm that `dqn_loss` sends no gradient into the target network, and that `soft_update` mixes weights exactly and keeps a separate copy. Finally, `train_dqn` must train a network whose greedy policy averages a return above 100 over 10 CartPole episodes. A random policy scores about 22, and the reference solution scores 283 on the checked seed.
