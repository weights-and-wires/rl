---
title: Distributional reinforcement learning
slug: 14-distributional-rl
part: "Part 3 — Approximation and deep value methods"
summary: Learn the whole distribution of returns with C51's projection and QR-DQN's quantile loss, and check what each preserves exactly.
minutes: 60
gpu: false
objectives:
  - State the distributional Bellman equation and explain what a return distribution adds to a value.
  - Implement the C51 projection and show that it keeps total probability and, away from the edges, the mean.
  - Train categorical distributions with cross-entropy, and evaluate a policy's return distributions on a small MDP.
  - Implement the quantile Huber loss and explain why its minimiser is a set of quantiles.
prereqs:
  - 12-deep-q-networks
exercise: 14-distributional-rl
---

# Distributional reinforcement learning

Two slot machines both pay 1 coin per pull on average. The first always pays
exactly 1. The second pays 100 once in a hundred pulls and nothing otherwise.
A Q-value can't tell them apart, because it's only the mean. Yet they're
different machines, and an agent that knew the difference could plan around
risk, or at least learn something richer about the world.

Distributional reinforcement learning learns the whole distribution of the
return, not only its average. The greedy policy still picks the action with
the highest mean, so on paper nothing changes. In practice, C51, the first
deep distributional agent, beat DQN by a wide margin on Atari. The richer
target seems to give the network better features to learn from.

This lesson builds the two main ways to represent a return distribution. C51
fixes a grid of possible returns and learns a probability for each. QR-DQN
fixes a set of probability levels and learns the returns at those levels.
C51 needs a projection step, and it's easy to get subtly wrong. QR-DQN needs
an asymmetric loss. You check both exactly.

## What you already know

- The *return* $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$ is a
  random quantity. $Q^\pi(s, a)$ is its mean after taking $a$ in $s$.
- DQN regresses $Q_\theta(s, a)$ toward $r_t + \gamma \max_{a'} Q_{\bar\theta}(s_{t+1}, a')$,
  with a target network $Q_{\bar\theta}$.
- A softmax turns a vector of scores, called *logits*, into probabilities
  that sum to one.
- Cross-entropy $-\sum_i m_i \log p_i$ measures how badly a distribution $p$
  predicts a target distribution $m$.
- The Huber loss is quadratic near zero and linear beyond a threshold
  $\kappa$, as in *Deep Q-networks*.

## Learn the distribution, not the mean

Write $Z(s, a)$ for the random return after taking action $a$ in state $s$.
Its mean is $Q(s, a)$. The ordinary Bellman equation relates means. The
*distributional Bellman equation* relates the random returns themselves:

$$
Z(s, a) \overset{D}{=} r + \gamma \, Z(s', a')
$$

Read it as: the return from $(s, a)$ has the same distribution as the
reward, plus $\gamma$ times the return from wherever you land next. Here
$s'$ is the random next state, $a'$ is the next action, and the $D$ over the
equals sign means "has the same distribution as". Take the mean of both
sides and you get the ordinary Bellman equation back.

The operator on the right, applied to a whole table of distributions, is a
$\gamma$-contraction for policy evaluation, in the worst-case Wasserstein
distance (Bellemare, Dabney, and Munos, 2017). Repeating it converges to
the true return distributions, the same way repeating the ordinary Bellman
update converges to $V^\pi$. The control version, with a max over next
actions, isn't a contraction in any metric over distributions. The means
still converge to $Q^*$, which is what greedy action selection needs.

> [!KEY]
> A distributional agent learns the whole return distribution $Z(s, a)$ and
> acts on its mean. The distributional Bellman equation says the return is
> the reward plus a discounted, random next return.

## Put the distribution on a grid: C51

C51 represents each $Z(s, a)$ as a probability distribution over $N$ fixed
returns, called *atoms*, spread evenly from $V_{\min}$ to $V_{\max}$:

$$
z_i = V_{\min} + i \, \Delta z, \qquad \Delta z = \frac{V_{\max} - V_{\min}}{N - 1}, \qquad i = 0, \ldots, N - 1
$$

Read it as: a ruler of $N$ evenly spaced marks from $V_{\min}$ to
$V_{\max}$. The network outputs $N$ logits per action, and a softmax turns
them into probabilities $p_i(s, a)$ for the atoms. The mean is
$Q(s, a) = \sum_i z_i \, p_i(s, a)$. The name comes from Atari's settings:
51 atoms from $-10$ to $10$, so $\Delta z = 0.4$. In numpy the support is
`np.linspace(v_min, v_max, n_atoms)`.

### Project the target back onto the grid

The target distribution for a transition moves every next-state atom $z_j$
to $r + \gamma z_j$. Those shifted positions almost never land on the grid.
So C51 *projects* them back. Each shifted atom splits its probability
between the two grid atoms on either side, in proportion to how close it is
to each:

$$
\hat z_j = \big[r + \gamma (1 - d) z_j\big]_{V_{\min}}^{V_{\max}}, \qquad b_j = \frac{\hat z_j - V_{\min}}{\Delta z}, \qquad l = \lfloor b_j \rfloor, \quad u = \lceil b_j \rceil
$$

$$
m_l \mathrel{+}= p_j \, (u - b_j), \qquad m_u \mathrel{+}= p_j \, (b_j - l)
$$

Read it piece by piece:

- $\hat z_j$ is the shifted atom, clipped into $[V_{\min}, V_{\max}]$. The
  factor $1 - d$ collapses every atom to $r$ when the step terminates.
- $b_j$ is its position measured in atom widths, a fractional index between
  0 and $N - 1$.
- $l$ and $u$ are the grid atoms below and above it.
- The lower atom gets the fraction $u - b_j$ of $p_j$, and the upper atom
  gets $b_j - l$. The closer neighbour gets more.

Here $p_j$ is the next-state probability for the greedy next action, the one
with the highest mean under the target network.

```python
tz = np.clip(r[:, None] + gamma * (1 - term[:, None]) * z[None, :], v_min, v_max)
b = (tz - v_min) / dz
lower, upper = np.floor(b).astype(int), np.ceil(b).astype(int)
np.add.at(m, (rows, lower), p * (upper - b))
np.add.at(m, (rows, upper), p * (b - lower))
```

> [!WARNING]
> When $b_j$ is a whole number, $l = u$, and both fractions $u - b_j$ and
> $b_j - l$ are zero. The formulas as written drop that atom's probability
> entirely. Fix it by moving $l$ down one, or $u$ up one, whenever they're
> equal. Then the shared atom receives all of $p_j$.

### Worked example

Take the support $z = [-2, -1, 0, 1, 2]$, so $\Delta z = 1$. The next state
has $p = [0.1, 0.2, 0.4, 0.2, 0.1]$, and $r = 1$ with $\gamma = 0.9$.

| $j$ | $p_j$ | $r + \gamma z_j$ | Clipped $\hat z_j$ | $b_j$ | Goes to |
|---|---|---|---|---|---|
| 0 | 0.1 | $-0.8$ | $-0.8$ | 1.2 | 0.08 to atom 1, 0.02 to atom 2 |
| 1 | 0.2 | 0.1 | 0.1 | 2.1 | 0.18 to atom 2, 0.02 to atom 3 |
| 2 | 0.4 | 1.0 | 1.0 | 3.0 | 0.4 to atom 3 |
| 3 | 0.2 | 1.9 | 1.9 | 3.9 | 0.02 to atom 3, 0.18 to atom 4 |
| 4 | 0.1 | 2.8 | 2.0 | 4.0 | 0.1 to atom 4 |

Adding up gives $m = [0, 0.08, 0.20, 0.44, 0.28]$, which sums to 1. Rows 2
and 4 are the $l = u$ case. The projected mean is $0.92$. The unprojected
target's mean is $1 + 0.9 \times 0 = 1$. The gap of $0.08$ is exactly the
clipping at row 4: $0.1 \times (2.8 - 2.0)$.

### What the projection keeps

Splitting an atom at position $t$ between neighbours $z_l$ and $z_u$ keeps
its mean. The split puts mass $u - b$ at $z_l$ and $b - l$ at $z_u$, and
$z_l (u - b) + z_u (b - l) = z_l + \Delta z (b - l) = t$. So each atom's
contribution to the mean survives, and the total probability survives too.
Wherever nothing is clipped, the projected mean is exactly
$r + \gamma (1 - d) \sum_j z_j p_j$.

That fact has a pleasant consequence. Run the projected distributional
Bellman update to convergence on a small MDP, with a support wide enough
that nothing clips. The means follow the ordinary Bellman update step for
step, so they converge to $V^\pi$ exactly. Rowland et al. (2018) proved
that the projected operator is a $\sqrt{\gamma}$-contraction in the Cramér
distance, so the distributions converge too. The exercise's
`categorical_evaluation` checks the means against
`rlenvs.policy_evaluation`.

## Train C51 with cross-entropy

The projected target $m$ is a distribution over the same atoms as the
prediction, so C51 trains with cross-entropy, treating $m$ as a constant:

$$
\mathcal{L} = -\frac{1}{B} \sum_{b=1}^{B} \sum_{i=0}^{N-1} m_{b,i} \log p_i(s_b, a_b)
$$

Read it as: for each transition in the batch, score how much probability the
predicted distribution gives each target atom, weighted by the target's
probability, and average over the batch. With $p = \operatorname{softmax}$ of
the logits, the gradient with respect to the logits is
$(p - m) / B$. That's the same gradient as classification with soft labels,
which is why the paper describes C51 as turning the Bellman update into
multiclass classification.

```python
loss = -(m * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
```

The greedy next action comes from the means: compute
`next_probs @ support` for each action, take the argmax, and project that
action's distribution.

> [!KEY]
> The C51 projection keeps total probability always, and keeps the mean
> wherever no atom is clipped. So the means of C51's targets are DQN's
> targets, and the extra information is in the shape.

## Flip the grid: quantile regression

C51 needs you to guess $V_{\min}$ and $V_{\max}$ in advance, and its
projection loses information at the edges. QR-DQN (Dabney et al., 2018)
turns the representation around. It keeps $N$ fixed probability levels and
learns the return at each one. Each learned location $\theta_i$ is meant to
be the $\hat\tau_i$ quantile of the return, at the midpoints

$$
\hat\tau_i = \frac{2i - 1}{2N}, \qquad i = 1, \ldots, N
$$

Read it as: cut $[0, 1]$ into $N$ equal slices and take each slice's centre.
For $N = 4$ that's $[0.125, 0.375, 0.625, 0.875]$. The paper proves those
midpoints give the closest $N$-point approximation in the 1-Wasserstein
distance. There's no projection, and no bounds to choose.

### The quantile loss

The $\tau$ quantile minimises an asymmetric absolute error. Overshooting a
sample costs $1 - \tau$ per unit, and undershooting costs $\tau$ per unit.
At the minimum, the pushes balance: a fraction $\tau$ of samples lies below
$\theta$. QR-DQN smooths the kink at zero with a Huber loss $L_\kappa$:

$$
\rho^\kappa_\tau(u) = \big|\tau - \mathbb{1}\{u < 0\}\big| \, L_\kappa(u), \qquad \mathcal{L} = \sum_{i=1}^{N} \frac{1}{M} \sum_{j=1}^{M} \rho^\kappa_{\hat\tau_i}\big(y_j - \theta_i\big)
$$

Read it piece by piece:

- $u = y_j - \theta_i$ is how far target sample $j$ lies above prediction
  $i$. The targets $y_j$ are $r + \gamma \theta_j(s', a^*)$, held constant.
- $\mathbb{1}\{u < 0\}$ is 1 when the target lies below the prediction.
- The weight is $\tau$ when the target is above and $1 - \tau$ when it's
  below. That asymmetry pulls $\theta_i$ to the $\tau_i$ quantile.
- $L_\kappa$ is the Huber loss. With $\kappa = 0$ it's replaced by $|u|$,
  the plain quantile loss.
- Every prediction is compared with every target, averaged over targets and
  summed over predictions.

```python
u = target[:, None, :] - pred[:, :, None]          # (B, N, M): i on axis 1
weight = (taus[None, :, None] - (u < 0).float()).abs()
loss = (weight * huber(u, kappa)).mean(dim=2).sum(dim=1).mean()
```

With $\kappa = 0$, the gradient with respect to $\theta_i$ is the fraction
of targets below $\theta_i$ minus $\tau_i$. It's zero exactly when
$\theta_i$ is a $\tau_i$ quantile of the targets.

### Worked example

Take two predictions $\theta = [0, 1]$ at levels $\tau = [0.25, 0.75]$, two
target samples $y = [0.5, 2]$, and $\kappa = 1$.

- For $\theta_1 = 0$, both targets lie above, so both weights are 0.25. The
  errors are 0.5 and 2, with Huber losses 0.125 and 1.5. The terms are
  0.03125 and 0.375, with mean 0.203125.
- For $\theta_2 = 1$, the errors are $-0.5$ and 1. The first target lies
  below, weight $1 - 0.75 = 0.25$, and the second lies above, weight 0.75.
  The terms are $0.25 \times 0.125 = 0.03125$ and $0.75 \times 0.5 = 0.375$,
  with mean 0.203125.
- The loss is the sum, 0.40625. With $\kappa = 0$ it's 0.75.

Fitting four locations to 10,000 samples from an exponential distribution
shows what $\kappa$ does. The true quantiles are $-\log(1 - \tau) = [0.134,
0.470, 0.981, 2.079]$. With $\kappa = 0$, the fit lands on the sample
quantiles, $[0.138, 0.477, 0.992, 2.133]$. With $\kappa = 1$ it lands at
$[0.379, 0.694, 1.029, 1.849]$, pulled toward the middle, because inside
$\kappa$ the loss behaves like a weighted squared error rather than an
absolute one. QR-DQN accepts that bias for smoother gradients.

> [!TIP]
> Check the orientation of `u` first. Predictions index axis 1 and pair
> with `taus`, and targets index axis 2. Swapping them pairs each $\tau$
> with the wrong index, and the loss still runs without error.

> [!RECAP]
> - The support is `np.linspace(v_min, v_max, n_atoms)`.
> - The projection clips $r + \gamma (1 - d) z_j$, finds $b_j$, and splits
>   $p_j$ between $\lfloor b_j \rfloor$ and $\lceil b_j \rceil$, with the
>   $l = u$ case getting all of $p_j$.
> - Projection keeps total probability always, and the mean when nothing
>   clips. C51 trains by cross-entropy, with gradient $(p - m)/B$.
> - The QR midpoints are $(2i - 1) / 2N$, and the loss weights each pairwise
>   Huber error by $|\tau_i - \mathbb{1}\{u < 0\}|$.

## Build it step by step

The checks test the functions in this order:

1. **`categorical_support(v_min, v_max, n_atoms)`.** One `np.linspace`.
2. **`c51_project(next_probs, rewards, terminated, gamma, support)`.**
   Broadcast to `(B, N)`, clip, compute `b`, `floor`, and `ceil`. Fix the
   $l = u$ case, then accumulate with `np.add.at`. Common mistake: writing
   `m[rows, lower] += ...`, which silently drops repeated indices when two
   atoms share a neighbour. Three checks cover the worked example, the
   integer case, and conservation of mass and mean.
3. **`c51_greedy(next_probs, support)`.** Means with `next_probs @ support`,
   an `argmax` over actions, and fancy indexing to pick each row's
   distribution.
4. **`c51_loss(logits, target_probs)`.** `torch.log_softmax` along axis 1.
   Common mistake: `torch.log(torch.softmax(...))`, which underflows to
   `-inf` for tiny probabilities.
5. **`categorical_evaluation(P, R, pi, gamma, support, iters)`.** Start with
   all mass on the atom nearest 0. Each sweep, for every $(s, a)$, project
   all $S$ next-state distributions with reward `R[s, a]` in one call, mix
   the rows with `P[s, a] @ projected`, and weight by `pi[s, a]`.
6. **`quantile_midpoints(n)`.** One line with `np.arange`.
7. **`quantile_huber_loss(pred, target, taus, kappa)`.** Build `u` with
   shape `(B, N, M)`, the Huber term with `torch.where`, and the weight with
   the indicator `u < 0`. Handle `kappa == 0` with `u.abs()`. The checks
   compare the gradient with finite differences and with the closed form,
   then fit quantiles of an exponential sample.

## Cost and scaling

Both methods multiply the network's output by $N$. The following table
compares them with DQN, for $|\mathcal{A}|$ actions and a batch of $B$:

| Method | Outputs per state | Target cost per sample | Loss cost per sample |
|---|---|---|---|
| DQN | $\lvert\mathcal{A}\rvert$ | One max | $O(1)$ |
| C51 | $N \lvert\mathcal{A}\rvert$ | Projection, $O(N)$ | Cross-entropy, $O(N)$ |
| QR-DQN | $N \lvert\mathcal{A}\rvert$ | Shift, $O(N)$ | Pairwise, $O(N M)$ |

C51 used $N = 51$ and QR-DQN used $N = 200$ on Atari. The pairwise
$(B, N, M)$ tensor dominates QR-DQN's memory. With $N = M = 200$ and
$B = 32$ it holds 1.28 million entries per update, still small next to the
convolutional network. `categorical_evaluation` costs $O(S^2 A N)$ per
sweep, which is fine for the tiny MDPs checks use.

## Where this breaks

**A bad support.** If returns fall outside $[V_{\min}, V_{\max}]$, the
projection piles probability on the end atoms and biases the mean, as in
the worked example. If the support is far too wide, most atoms are never
used and resolution suffers. QR-DQN avoids choosing bounds at all.

**Reward scale.** C51's fixed support ties it to the reward scale. On Atari
it relied on DQN's clipping of rewards to $[-1, 1]$ to keep returns on a
known scale.

**No convergence guarantee for control.** The distributional optimality
operator isn't a contraction. In practice the means behave like DQN's, but
the shape of the learned distribution for a greedy policy can oscillate.

**Unclear benefit.** Lyle, Castro, and Bellemare (2019) showed that in many
tabular and linear settings, distributional updates produce exactly the same
means as expected ones. The gains appear with deep networks, and why they
appear is still debated. One common explanation is that the richer target
helps the network learn better features.

**Quantile crossing.** Nothing forces $\theta_1 \le \theta_2 \le \cdots$.
Learned quantiles can cross, especially early in training. Some later
methods sort them or parameterise increments.

## Further reading

- [Bellemare, Dabney, and Munos, *A distributional perspective on
  reinforcement learning* (2017)](https://arxiv.org/abs/1707.06887). The
  distributional Bellman operator, its contraction properties, and C51.
- [Dabney et al., *Distributional reinforcement learning with quantile
  regression* (2018)](https://arxiv.org/abs/1710.10044). QR-DQN, the
  quantile midpoints, and the quantile Huber loss.
- [Dabney et al., *Implicit quantile networks for distributional
  reinforcement learning* (2018)](https://arxiv.org/abs/1806.06923). IQN,
  which samples $\tau$ and feeds it to the network. Its loss divides the
  Huber term by $\kappa$, which matches QR-DQN's at $\kappa = 1$.
- [Rowland et al., *An analysis of categorical distributional reinforcement
  learning* (2018)](https://arxiv.org/abs/1802.08163). Proves the projected
  operator converges, using the Cramér distance.
- [Lyle, Castro, and Bellemare, *A comparative analysis of expected and
  distributional reinforcement learning* (2019)](https://arxiv.org/abs/1901.11084).
  When distributional and expected RL behave identically, and when they
  don't.
- [Bellemare, Dabney, and Rowland, *Distributional Reinforcement Learning*
  (2023)](https://www.distributional-rl.org/). The textbook, free online.
- [Dabney et al., *A distributional code for value in dopamine-based
  reinforcement learning* (2020)](https://doi.org/10.1038/s41586-019-1924-6).
  Evidence that dopamine neurons encode a distribution of predicted rewards.

## The exercise

You write `categorical_support`, `c51_project`, `c51_greedy`, `c51_loss`,
and `categorical_evaluation` for C51. Then you write `quantile_midpoints`
and `quantile_huber_loss` for QR-DQN.

The checks verify the support's spacing and the projection against the
worked example, including the clipped mean of 0.92. They check that the
$l = u$ case keeps its mass, that terminal steps collapse to $r$, and that
the projection conserves probability always and the mean whenever nothing
clips. They check the greedy action and the cross-entropy loss and its
gradient. They confirm that `categorical_evaluation`'s means equal $V^\pi$ on
a random MDP. Finally, they check the quantile loss against the hand example,
its gradient against finite differences and the closed form, and that
minimising it on 10,000 exponential samples recovers the quantiles.
