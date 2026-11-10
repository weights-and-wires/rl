---
title: Distributional reinforcement learning
slug: 029-distributional-rl
part: "Part 4 — Deep value methods"
summary: Learn the whole distribution of returns instead of its mean, with C51's projection onto a grid and QR-DQN's quantile Huber loss.
minutes: 45
gpu: false
objectives:
  - Explain what a return distribution tells you that a Q-value does not.
  - Project a shifted return distribution back onto a fixed grid, and show that the projection keeps total probability and, away from the edges, the mean.
  - Train a categorical distribution with cross-entropy against the projected target.
  - Implement the quantile Huber loss and explain why its minimizer is a set of quantiles.
prereqs:
  - 026-deep-q-networks
exercise: 029-distributional-rl
---

# Distributional reinforcement learning

Two slot machines both pay 1 coin per pull on average. The first always pays exactly 1. The second pays 100 once in a hundred pulls and nothing otherwise. A Q-value cannot tell them apart, because it is only the mean. Yet they are different machines, and an agent that knew the difference could plan around risk.

Distributional reinforcement learning learns the whole distribution of the return, not only its average. The greedy policy still picks the action with the highest mean, so on paper nothing changes. In practice, C51, the first deep distributional agent, beat DQN by a wide margin on Atari, though why a richer target helps is still debated.

This chapter builds the two standard ways to represent a return distribution. C51 fixes a grid of possible returns and learns a probability for each one, which needs a projection step. QR-DQN fixes a set of probability levels and learns the return at each one, which needs an asymmetric loss. You check both exactly.

## What you already know

- The return $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$ is a random quantity, and $Q^\pi(s, a)$ is its mean after taking $a$ in $s$.
- DQN regresses $Q_\theta(s, a)$ toward $r_t + \gamma \max_{a'} Q_{\bar\theta}(s_{t+1}, a')$, with a target network $Q_{\bar\theta}$.
- A softmax turns a vector of scores, called logits, into probabilities that sum to one.
- Cross-entropy $-\sum_i m_i \log p_i$ measures how badly a distribution $p$ predicts a target distribution $m$.
- The Huber loss is quadratic near zero and linear beyond a threshold $\kappa$.

## Keep the shape, not only the mean

Write $Z(s, a)$ for the random return after taking action $a$ in state $s$, so that $Q(s, a)$ is its mean. Figure 1 shows the two machines as distributions. Both have mean 1, and the picture shows what the mean hides.

```figure
<svg viewBox="0 0 620 262" role="img" aria-label="Two payout distributions with the same mean of 1: machine A always pays 1, machine B pays 100 with probability 0.01">
<text x="160.0" y="26.0" class="fig-bold" text-anchor="middle">machine A: always pays 1</text>
<line x1="30.0" y1="190.0" x2="290.0" y2="190.0" class="stroke-muted"/>
<text x="460.0" y="26.0" class="fig-bold" text-anchor="middle">machine B: pays 100 once in 100 pulls</text>
<line x1="330.0" y1="190.0" x2="590.0" y2="190.0" class="stroke-muted"/>
<rect x="26.4" y="40.0" width="12.0" height="150.0" class="fill-a"/>
<text x="46.4" y="50.0" class="fig-small" text-anchor="start">probability 1.00</text>
<rect x="324.0" y="41.5" width="12.0" height="148.5" class="fill-b"/>
<text x="346.0" y="52.0" class="fig-small" text-anchor="start">0 pays: probability 0.99</text>
<rect x="564.0" y="188.5" width="12.0" height="1.5" class="fill-b"/>
<text x="570.0" y="178.0" class="fig-small" text-anchor="end">100: probability 0.01</text>
<text x="30.0" y="208.0" class="fig-small" text-anchor="middle">0</text>
<text x="150.0" y="208.0" class="fig-small" text-anchor="middle">50</text>
<text x="270.0" y="208.0" class="fig-small" text-anchor="middle">100</text>
<text x="160.0" y="226.0" class="fig-small" text-anchor="middle">payout</text>
<text x="330.0" y="208.0" class="fig-small" text-anchor="middle">0</text>
<text x="450.0" y="208.0" class="fig-small" text-anchor="middle">50</text>
<text x="570.0" y="208.0" class="fig-small" text-anchor="middle">100</text>
<text x="460.0" y="226.0" class="fig-small" text-anchor="middle">payout</text>
<text x="160.0" y="250.0" class="fig-bold" text-anchor="middle">mean 1</text>
<text x="460.0" y="250.0" class="fig-bold" text-anchor="middle">mean 0.99 x 0 + 0.01 x 100 = 1</text>
</svg>
Figure 1. Both machines have a Q-value of 1. A single number cannot say that A is a sure thing and B is a lottery, but the whole distribution can.
```

The ordinary Bellman equation relates means. The distributional Bellman equation relates the random returns themselves:

$$
Z(s, a) \overset{D}{=} r + \gamma \, Z(s', a')
$$

Read it as: the return from $(s, a)$ has the same distribution as the reward plus $\gamma$ times the return from wherever you land next. Here $s'$ is the random next state, $a'$ is the next action, and $\overset{D}{=}$ means "has the same distribution as". Take the mean of both sides and you get the ordinary Bellman equation back.

For policy evaluation, applying this update to a whole table of distributions is a $\gamma$-contraction in the worst-case Wasserstein distance (Bellemare, Dabney, and Munos, 2017), so repeating it converges. The practical question is how to store a distribution in a network's output layer. A distribution is not a fixed-size vector, so you choose a representation, and the two standard choices differ in what they hold fixed.

## Put the distribution on a grid: C51

C51 holds the returns fixed and learns their probabilities. It represents each $Z(s, a)$ as a probability distribution over $N$ evenly spaced returns, called *atoms*:

$$
z_i = V_{\min} + i \, \Delta z, \qquad \Delta z = \frac{V_{\max} - V_{\min}}{N - 1}, \qquad i = 0, \ldots, N - 1
$$

The network outputs $N$ logits per action, and a softmax turns them into probabilities $p_i(s, a)$. The mean is $Q(s, a) = \sum_i z_i \, p_i(s, a)$. The name comes from 51 atoms from $-10$ to $10$ on Atari. In numpy the grid is `np.linspace(v_min, v_max, n_atoms)`.

### Why the target needs a projection

To build a training target, you apply the Bellman equation to the next state's distribution. Every atom $z_j$ moves to $r + \gamma z_j$, and keeps its probability $p_j$. The trouble is that the shifted atoms almost never land on the grid. The network can only output probabilities on the grid atoms, so the target must be a distribution on the grid too.

The obvious fix is to round each shifted atom to the nearest grid point, but that moves the mean by up to half a grid step on every update. C51 instead *projects*: each shifted atom splits its probability between the two grid atoms on either side, and the closer neighbor gets more.

With the shifted atom clipped into $[V_{\min}, V_{\max}]$, write $b_j$ for its position measured in grid widths, and $l$ and $u$ for the grid indexes just below and above it:

$$
b_j = \frac{\hat z_j - V_{\min}}{\Delta z}, \qquad m_l \mathrel{+}= p_j (u - b_j), \qquad m_u \mathrel{+}= p_j (b_j - l)
$$

Here $\hat z_j = \big[r + \gamma (1 - d) z_j\big]_{V_{\min}}^{V_{\max}}$ is the clipped shifted atom, so a terminal step moves every atom to $r$. The probability $p_j$ is the next-state probability for the greedy next action, the one with the highest mean under the target network.

### Work an example

Take the grid $z = [-2, -1, 0, 1, 2]$, so $\Delta z = 1$. The next state has $p = [0.1, 0.2, 0.4, 0.2, 0.1]$, and $r = 1$ with $\gamma = 0.9$. The following table shows where each atom's probability goes:

| $j$ | $p_j$ | $r + \gamma z_j$ | Clipped | $b_j$ | Goes to |
|---|---|---|---|---|---|
| 0 | 0.1 | $-0.8$ | $-0.8$ | 1.2 | 0.08 to atom 1, 0.02 to atom 2 |
| 1 | 0.2 | 0.1 | 0.1 | 2.1 | 0.18 to atom 2, 0.02 to atom 3 |
| 2 | 0.4 | 1.0 | 1.0 | 3.0 | 0.4 to atom 3 |
| 3 | 0.2 | 1.9 | 1.9 | 3.9 | 0.02 to atom 3, 0.18 to atom 4 |
| 4 | 0.1 | 2.8 | 2.0 | 4.0 | 0.1 to atom 4 |

Adding up each atom's share gives $m = [0, 0.08, 0.20, 0.44, 0.28]$, which sums to 1, as Figure 2 shows. The projected mean is $0.92$. The unprojected target's mean is $1 + 0.9 \times 0 = 1$. The gap of $0.08$ is exactly the clipping at row 4: $0.1 \times (2.8 - 2.0)$.

```figure
<svg viewBox="0 0 620 300" role="img" aria-label="Shifted return atoms before projection and the projected probabilities on the five-atom grid">
<text x="20.0" y="22.0" class="fig-bold" text-anchor="start">before: r + 0.9 z_j with probability p_j (r = 1)</text>
<line x1="40.0" y1="120.0" x2="590.0" y2="120.0" class="stroke-muted"/>
<rect x="243.7" y="100.0" width="8.0" height="20.0" class="fill-b"/>
<text x="247.7" y="94.0" class="fig-small" text-anchor="middle">-0.8</text>
<rect x="337.7" y="80.0" width="8.0" height="40.0" class="fill-b"/>
<text x="341.7" y="74.0" class="fig-small" text-anchor="middle">0.1</text>
<rect x="431.8" y="40.0" width="8.0" height="80.0" class="fill-b"/>
<text x="435.8" y="34.0" class="fig-small" text-anchor="middle">1.0</text>
<rect x="525.8" y="80.0" width="8.0" height="40.0" class="fill-b"/>
<text x="529.8" y="74.0" class="fig-small" text-anchor="middle">1.9</text>
<rect x="536.2" y="100.0" width="8.0" height="20.0" class="fill-b"/>
<text x="40.0" y="52.0" class="fig-small" text-anchor="start">2.8 is clipped to 2.0 (last orange bar)</text>
<text x="20.0" y="160.0" class="fig-bold" text-anchor="start">after: projected onto the grid z = -2, -1, 0, 1, 2</text>
<line x1="40.0" y1="270.0" x2="590.0" y2="270.0" class="stroke-muted"/>
<rect x="108.2" y="270.0" width="28.0" height="0.0" class="fill-a"/>
<text x="122.2" y="264.0" class="fig-small" text-anchor="middle">0.00</text>
<text x="122.2" y="288.0" class="fig-small" text-anchor="middle">-2</text>
<line x1="122.2" y1="120.0" x2="122.2" y2="126.0" class="stroke-muted"/>
<text x="122.2" y="138.0" class="fig-small" text-anchor="middle">-2</text>
<rect x="212.8" y="252.4" width="28.0" height="17.6" class="fill-a"/>
<text x="226.8" y="246.4" class="fig-small" text-anchor="middle">0.08</text>
<text x="226.8" y="288.0" class="fig-small" text-anchor="middle">-1</text>
<line x1="226.8" y1="120.0" x2="226.8" y2="126.0" class="stroke-muted"/>
<text x="226.8" y="138.0" class="fig-small" text-anchor="middle">-1</text>
<rect x="317.2" y="226.0" width="28.0" height="44.0" class="fill-a"/>
<text x="331.2" y="220.0" class="fig-small" text-anchor="middle">0.20</text>
<text x="331.2" y="288.0" class="fig-small" text-anchor="middle">0</text>
<line x1="331.2" y1="120.0" x2="331.2" y2="126.0" class="stroke-muted"/>
<text x="331.2" y="138.0" class="fig-small" text-anchor="middle">0</text>
<rect x="421.8" y="173.2" width="28.0" height="96.8" class="fill-a"/>
<text x="435.8" y="167.2" class="fig-small" text-anchor="middle">0.44</text>
<text x="435.8" y="288.0" class="fig-small" text-anchor="middle">1</text>
<line x1="435.8" y1="120.0" x2="435.8" y2="126.0" class="stroke-muted"/>
<text x="435.8" y="138.0" class="fig-small" text-anchor="middle">1</text>
<rect x="526.2" y="208.4" width="28.0" height="61.6" class="fill-a"/>
<text x="540.2" y="202.4" class="fig-small" text-anchor="middle">0.28</text>
<text x="540.2" y="288.0" class="fig-small" text-anchor="middle">2</text>
<line x1="540.2" y1="120.0" x2="540.2" y2="126.0" class="stroke-muted"/>
<text x="540.2" y="138.0" class="fig-small" text-anchor="middle">2</text>
</svg>
Figure 2. The five shifted atoms (orange) land between grid points, so each splits its probability between its two neighbors. The projected probabilities (blue) are 0, 0.08, 0.20, 0.44, and 0.28; the atom that shifted to 2.8 is clipped to the top of the grid, which is why the mean drops from 1 to 0.92.
```

> [!WARNING]
> When $b_j$ is a whole number, $l = u$, and both fractions $u - b_j$ and $b_j - l$ are zero. The formulas as written drop that atom's probability. Rows 2 and 4 above are this case. Fix it by moving $l$ down one, or $u$ up one, whenever they are equal, so the shared atom receives all of $p_j$.

### What the projection keeps

Splitting an atom at position $b$ between neighbors $l$ and $u$ keeps its mean. The split puts mass $u - b$ at $z_l$ and $b - l$ at $z_u$, and $z_l (u - b) + z_u (b - l) = z_l + \Delta z\,(b - l)$, which is the shifted atom's own position. So each atom's contribution to the mean survives, and the total probability survives too. Wherever nothing is clipped, the projected mean is exactly $r + \gamma (1 - d) \sum_j z_j p_j$, which is DQN's target for the mean. The extra information is in the shape.

```python
tz = np.clip(r[:, None] + gamma * (1 - term[:, None]) * z[None, :], v_min, v_max)
b = (tz - v_min) / dz
lower, upper = np.floor(b).astype(int), np.ceil(b).astype(int)
np.add.at(m, (rows, lower), p * (upper - b))
np.add.at(m, (rows, upper), p * (b - lower))
```

### Train with cross-entropy

The projected target $m$ is a distribution over the same atoms as the prediction, so C51 trains with cross-entropy, holding $m$ constant. The gradient with respect to the logits is $(p - m) / B$ for a batch of $B$, the same as classification with soft labels:

```python
loss = -(m * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
```

> [!KEY]
> The C51 projection keeps total probability always, and keeps the mean wherever no atom is clipped. So the means of C51's targets are DQN's targets, and the extra information lives in the shape.

## Flip the grid: quantile regression

C51 makes you guess $V_{\min}$ and $V_{\max}$ in advance, and its projection loses information at the edges. QR-DQN (Dabney et al., 2018) flips the representation. It holds $N$ probability levels fixed and learns the return at each one. Each learned location $\theta_i$ should be the $\hat\tau_i$ quantile of the return, at the midpoints

$$
\hat\tau_i = \frac{2i - 1}{2N}, \qquad i = 1, \ldots, N
$$

Cut $[0, 1]$ into $N$ equal slices and take each slice's center. For $N = 4$ that gives $[0.125, 0.375, 0.625, 0.875]$. Figure 3 shows what the network learns for an exponential return: the returns where the cumulative curve crosses those four levels. There is no projection and no bounds to choose.

```figure
<svg viewBox="0 0 620 305" role="img" aria-label="Cumulative distribution of an exponential return with the four quantile levels 0.125, 0.375, 0.625, 0.875 and the returns where the curve reaches them">
<line x1="70.0" y1="240.0" x2="550.0" y2="240.0" class="stroke-grid"/>
<text x="62.0" y="244.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="187.5" x2="550.0" y2="187.5" class="stroke-grid"/>
<text x="62.0" y="191.5" class="fig-small" text-anchor="end">0.25</text>
<line x1="70.0" y1="135.0" x2="550.0" y2="135.0" class="stroke-grid"/>
<text x="62.0" y="139.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="70.0" y1="82.5" x2="550.0" y2="82.5" class="stroke-grid"/>
<text x="62.0" y="86.5" class="fig-small" text-anchor="end">0.75</text>
<line x1="70.0" y1="30.0" x2="550.0" y2="30.0" class="stroke-grid"/>
<text x="62.0" y="34.0" class="fig-small" text-anchor="end">1</text>
<text x="70.0" y="260.0" class="fig-small" text-anchor="middle">0</text>
<text x="230.0" y="260.0" class="fig-small" text-anchor="middle">1</text>
<text x="390.0" y="260.0" class="fig-small" text-anchor="middle">2</text>
<text x="550.0" y="260.0" class="fig-small" text-anchor="middle">3</text>
<line x1="70.0" y1="240.0" x2="550.0" y2="240.0" class="stroke-muted"/>
<polyline points="70.0,240.0 74.0,234.8 78.0,229.8 82.0,224.8 86.0,220.0 90.0,215.3 94.0,210.7 98.0,206.3 102.0,201.9 106.0,197.7 110.0,193.5 114.0,189.5 118.0,185.6 122.0,181.7 126.0,178.0 130.0,174.3 134.0,170.8 138.0,167.3 142.0,163.9 146.0,160.6 150.0,157.4 154.0,154.2 158.0,151.2 162.0,148.2 166.0,145.3 170.0,142.4 174.0,139.6 178.0,136.9 182.0,134.3 186.0,131.7 190.0,129.2 194.0,126.7 198.0,124.4 202.0,122.0 206.0,119.8 210.0,117.5 214.0,115.4 218.0,113.3 222.0,111.2 226.0,109.2 230.0,107.3 234.0,105.3 238.0,103.5 242.0,101.7 246.0,99.9 250.0,98.2 254.0,96.5 258.0,94.9 262.0,93.3 266.0,91.7 270.0,90.2 274.0,88.7 278.0,87.2 282.0,85.8 286.0,84.4 290.0,83.1 294.0,81.8 298.0,80.5 302.0,79.3 306.0,78.0 310.0,76.9 314.0,75.7 318.0,74.6 322.0,73.5 326.0,72.4 330.0,71.4 334.0,70.3 338.0,69.3 342.0,68.4 346.0,67.4 350.0,66.5 354.0,65.6 358.0,64.7 362.0,63.9 366.0,63.0 370.0,62.2 374.0,61.4 378.0,60.6 382.0,59.9 386.0,59.1 390.0,58.4 394.0,57.7 398.0,57.0 402.0,56.4 406.0,55.7 410.0,55.1 414.0,54.5 418.0,53.9 422.0,53.3 426.0,52.7 430.0,52.1 434.0,51.6 438.0,51.1 442.0,50.5 446.0,50.0 450.0,49.5 454.0,49.1 458.0,48.6 462.0,48.1 466.0,47.7 470.0,47.2 474.0,46.8 478.0,46.4 482.0,46.0 486.0,45.6 490.0,45.2 494.0,44.8 498.0,44.5 502.0,44.1 506.0,43.8 510.0,43.4 514.0,43.1 518.0,42.8 522.0,42.5 526.0,42.1 530.0,41.8 534.0,41.6 538.0,41.3 542.0,41.0 546.0,40.7 550.0,40.5" class="stroke-a" fill="none"/>
<line x1="70.0" y1="213.8" x2="91.4" y2="213.8" class="stroke-b dashed"/>
<line x1="91.4" y1="213.8" x2="91.4" y2="240.0" class="stroke-b dashed"/>
<circle cx="91.4" cy="213.8" r="5" class="fill-b"/>
<text x="99.4" y="232.0" class="fig-small" text-anchor="start">0.134</text>
<line x1="70.0" y1="161.2" x2="145.2" y2="161.2" class="stroke-b dashed"/>
<line x1="145.2" y1="161.2" x2="145.2" y2="240.0" class="stroke-b dashed"/>
<circle cx="145.2" cy="161.2" r="5" class="fill-b"/>
<text x="153.2" y="232.0" class="fig-small" text-anchor="start">0.470</text>
<line x1="70.0" y1="108.8" x2="226.9" y2="108.8" class="stroke-b dashed"/>
<line x1="226.9" y1="108.8" x2="226.9" y2="240.0" class="stroke-b dashed"/>
<circle cx="226.9" cy="108.8" r="5" class="fill-b"/>
<text x="234.9" y="232.0" class="fig-small" text-anchor="start">0.981</text>
<line x1="70.0" y1="56.2" x2="402.7" y2="56.2" class="stroke-b dashed"/>
<line x1="402.7" y1="56.2" x2="402.7" y2="240.0" class="stroke-b dashed"/>
<circle cx="402.7" cy="56.2" r="5" class="fill-b"/>
<text x="410.7" y="232.0" class="fig-small" text-anchor="start">2.079</text>
<text x="310.0" y="282.0" text-anchor="middle">return</text>
<text x="62.0" y="20.0" text-anchor="start">probability the return is at most this value</text>
</svg>
Figure 3. QR-DQN learns the four orange x-values: the returns at which the cumulative curve crosses the midpoint levels 0.125, 0.375, 0.625, and 0.875. For this exponential return they are 0.134, 0.470, 0.981, and 2.079.
```

### Why an asymmetric loss finds a quantile

Squared error finds a mean, so you need a different loss to find a quantile. Charge $\tau$ per unit for every sample above your guess $\theta$, and $1 - \tau$ per unit for every sample below it. If more than a fraction $\tau$ of the samples lie below $\theta$, moving down saves more than it costs, so $\theta$ moves down. It stops where the pushes balance, which is the $\tau$ quantile.

QR-DQN smooths the kink at zero with a Huber loss $L_\kappa$:

$$
\rho^\kappa_\tau(u) = \big|\tau - \mathbb{1}\{u < 0\}\big| \, L_\kappa(u), \qquad \mathcal{L} = \sum_{i=1}^{N} \frac{1}{M} \sum_{j=1}^{M} \rho^\kappa_{\hat\tau_i}\big(y_j - \theta_i\big)
$$

Read it piece by piece:

- $u = y_j - \theta_i$ is how far target sample $j$ lies above prediction $i$. The targets $y_j$ are $r + \gamma \theta_j(s', a^*)$, held constant.
- $\mathbb{1}\{u < 0\}$ is 1 when the target lies below the prediction, so the weight is $\tau$ when the target is above and $1 - \tau$ when it is below.
- Every prediction is compared with every target, averaged over targets and summed over predictions.
- With $\kappa = 0$, $L_\kappa$ is replaced by $|u|$, the plain quantile loss.

```python
u = target[:, None, :] - pred[:, :, None]          # (B, N, M): i on axis 1
weight = (taus[None, :, None] - (u < 0).float()).abs()
loss = (weight * huber(u, kappa)).mean(dim=2).sum(dim=1).mean()
```

With $\kappa = 0$, the gradient with respect to $\theta_i$ is the fraction of targets below $\theta_i$ minus $\tau_i$. It is zero exactly when $\theta_i$ is a $\tau_i$ quantile of the targets.

### Work an example

Take two predictions $\theta = [0, 1]$ at levels $\tau = [0.25, 0.75]$, two target samples $y = [0.5, 2]$, and $\kappa = 1$.

- For $\theta_1 = 0$, both targets lie above, so both weights are 0.25. The errors are 0.5 and 2, with Huber losses 0.125 and 1.5. The terms are 0.03125 and 0.375, with mean 0.203125.
- For $\theta_2 = 1$, the errors are $-0.5$ and 1. The first target lies below, with weight $1 - 0.75 = 0.25$, and the second lies above, with weight 0.75. The terms are $0.25 \times 0.125 = 0.03125$ and $0.75 \times 0.5 = 0.375$, with mean 0.203125.
- The loss is the sum, 0.40625. With $\kappa = 0$ it is 0.75.

> [!TIP]
> Check the orientation of `u` first. Predictions index axis 1 and pair with `taus`, and targets index axis 2. Swapping them pairs each $\tau$ with the wrong index, and the loss still runs without error.

> [!RECAP]
> - C51's grid is `np.linspace(v_min, v_max, n_atoms)`. The projection clips $r + \gamma (1 - d) z_j$, finds $b_j$, and splits $p_j$ between the two neighbors, with $l = u$ getting all of $p_j$.
> - The projection keeps total probability always, and the mean when nothing clips. C51 trains by cross-entropy.
> - The QR midpoints are $(2i - 1) / 2N$, and the loss weights each pairwise Huber error by $|\tau_i - \mathbb{1}\{u < 0\}|$.

## Build it step by step

The checks test the functions in this order:

1. **`categorical_support(v_min, v_max, n_atoms)`.** One `np.linspace`.
2. **`c51_project(next_probs, rewards, terminated, gamma, support)`.** Broadcast to `(B, N)`, clip, compute `b`, `floor`, and `ceil`. Fix the $l = u$ case, then accumulate with `np.add.at`. The most common mistake is writing `m[rows, lower] += ...`, which silently drops repeated indexes.
3. **`c51_loss(logits, target_probs)`.** Use `torch.log_softmax` along axis 1. The most common mistake is `torch.log(torch.softmax(...))`, which underflows to `-inf` for tiny probabilities.
4. **`quantile_midpoints(n)`.** One line with `np.arange`.
5. **`quantile_huber_loss(pred, target, taus, kappa)`.** Build `u` with shape `(B, N, M)`, the Huber term with `torch.where`, and the weight with the indicator `u < 0`. Handle `kappa == 0` with `u.abs()`. The checks compare the gradient with finite differences and the closed form, then fit quantiles of an exponential sample.

## Where this breaks

**A bad support.** If returns fall outside $[V_{\min}, V_{\max}]$, the projection piles probability on the end atoms and biases the mean, as in the worked example. If the support is far too wide, most atoms go unused. QR-DQN avoids choosing bounds.

**No convergence guarantee for control.** The distributional optimality operator is not a contraction. In practice the means behave like DQN's, but the learned shape can oscillate.

**Unclear benefit.** Lyle, Castro, and Bellemare (2019) showed that in many tabular and linear settings, distributional updates produce exactly the same means as expected ones. The gains appear with deep networks, and why is still debated.

**Quantile crossing.** Nothing forces $\theta_1 \le \theta_2 \le \cdots$, so learned quantiles can cross early in training.

## Further reading

- [Bellemare, Dabney, and Munos, *A distributional perspective on reinforcement learning* (2017)](https://arxiv.org/abs/1707.06887). The distributional Bellman operator, its contraction properties, and C51.
- [Dabney et al., *Distributional reinforcement learning with quantile regression* (2018)](https://arxiv.org/abs/1710.10044). QR-DQN, the quantile midpoints, and the quantile Huber loss.
- [Rowland et al., *An analysis of categorical distributional reinforcement learning* (2018)](https://arxiv.org/abs/1802.08163). Proves the projected operator converges, using the Cramér distance.
- [Lyle, Castro, and Bellemare, *A comparative analysis of expected and distributional reinforcement learning* (2019)](https://arxiv.org/abs/1901.11084). When distributional and expected RL behave identically, and when they do not.

## The exercise

You write `categorical_support`, `c51_project`, and `c51_loss` for C51. Then you write `quantile_midpoints` and `quantile_huber_loss` for QR-DQN.

The checks verify the projection against the worked example, including the clipped mean of 0.92, the $l = u$ case, and terminal steps. They check that it conserves probability always and the mean whenever nothing clips. They check the cross-entropy loss and its gradient. Finally, they check the quantile loss against the hand example and against finite differences, and that fitting four locations to 10,000 exponential samples recovers the true quantiles.
