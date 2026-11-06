---
title: Contextual bandits and LinUCB
slug: 007-contextual-bandits
part: "Part 1 — Learning from reward"
summary: Pick the best arm for each context by fitting a linear reward model per arm and adding an optimism bonus where the data is thin.
minutes: 35
gpu: false
objectives:
  - Explain why a bandit that ignores the context loses reward every round.
  - Fit one ridge regression per arm from running sums, and predict an arm's reward for a context.
  - Score arms with LinUCB's prediction plus confidence width, and say why the width shrinks only where data arrives.
  - Update an inverse matrix after one new data point with the Sherman-Morrison formula.
prereqs:
  - 004-optimism-and-ucb
exercise: 007-contextual-bandits
---

# Contextual bandits and LinUCB

A news site picks one of three headlines for every visitor. In the bandit you have studied so far, one headline is best for everyone, and your job is to find it. But visitors differ. A sports fan clicks the match report, and a cook clicks the recipe. The best headline depends on who walks in. Before you choose, you can see something about the visitor, such as age, location, or past clicks. That information is the *context*.

The obvious approach is to ignore the context and find the single best headline. That wastes reward on every visitor for whom some other headline would do better. In this chapter's exercise, the best single arm loses about 0.61 reward per round compared with the best choice for each context. Over 2000 rounds, that adds up to about 1220. Another obvious approach is to run a separate bandit for each kind of visitor. That works when contexts repeat, but real contexts are lists of numbers that almost never repeat exactly. Each separate bandit would see too little data to learn anything.

This chapter shows a way out. You assume the reward is a straight-line function of the context, so every visitor's feedback teaches you about all similar visitors. Then you add the optimism you met in [Optimism in the face of uncertainty](/lessons/004-optimism-and-ucb), measured in the space of features.

## What you already know

- UCB1 picks the arm with the largest estimate plus a bonus that is large when you have little data (see [Optimism in the face of uncertainty](/lessons/004-optimism-and-ucb)).
- Regret adds up how much less each pulled arm paid than the best arm.
- Ridge regression fits weights by solving $(X^\top X + \lambda I)\,\theta = X^\top y$.
- The outer product $x x^\top$ of a vector with itself is a matrix.
- `np.linalg.solve` and `np.linalg.inv` solve and invert small matrices.

## The best arm depends on the context

In a *contextual bandit*, each round starts with a context vector $x \in \mathbb{R}^d$ that you observe before you choose. The mean reward of arm $a$ depends on $x$. The simplest useful model is linear, with one weight vector $\theta_a$ per arm:

$$
\mathbb{E}[r \mid x, a] = \theta_a^\top x
$$

Read it as: each arm scores the context with its own weights. Different contexts favor different arms, so no arm wins everywhere. In the exercise, $d = 4$ and there are three arms. Each context is a leading 1, for an intercept, followed by three standard-normal features. Figure 1 shows one slice of that world, where only the first feature varies.

```figure
<svg viewBox="0 0 600 350" role="img" aria-label="Three straight lines of mean reward against the first context feature; arm 1 is highest on the left, arm 0 on the right, and the lines cross at 1.49">
<line x1="70.0" y1="230.0" x2="520.0" y2="230.0" class="stroke-grid"/>
<text x="62.0" y="234.0" class="fig-small" text-anchor="end">-2</text>
<line x1="70.0" y1="190.0" x2="520.0" y2="190.0" class="stroke-grid"/>
<text x="62.0" y="194.0" class="fig-small" text-anchor="end">-1</text>
<line x1="70.0" y1="150.0" x2="520.0" y2="150.0" class="stroke-grid"/>
<text x="62.0" y="154.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="110.0" x2="520.0" y2="110.0" class="stroke-grid"/>
<text x="62.0" y="114.0" class="fig-small" text-anchor="end">1</text>
<line x1="70.0" y1="70.0" x2="520.0" y2="70.0" class="stroke-grid"/>
<text x="62.0" y="74.0" class="fig-small" text-anchor="end">2</text>
<line x1="70.0" y1="30.0" x2="520.0" y2="30.0" class="stroke-grid"/>
<text x="62.0" y="34.0" class="fig-small" text-anchor="end">3</text>
<line x1="70.0" y1="230.0" x2="520.0" y2="230.0" class="stroke-muted"/>
<line x1="70.0" y1="30.0" x2="70.0" y2="230.0" class="stroke-muted"/>
<line x1="115.0" y1="230.0" x2="115.0" y2="235.0" class="stroke-muted"/>
<text x="115.0" y="250.0" class="fig-small" text-anchor="middle">-2</text>
<line x1="205.0" y1="230.0" x2="205.0" y2="235.0" class="stroke-muted"/>
<text x="205.0" y="250.0" class="fig-small" text-anchor="middle">-1</text>
<line x1="295.0" y1="230.0" x2="295.0" y2="235.0" class="stroke-muted"/>
<text x="295.0" y="250.0" class="fig-small" text-anchor="middle">0</text>
<line x1="385.0" y1="230.0" x2="385.0" y2="235.0" class="stroke-muted"/>
<text x="385.0" y="250.0" class="fig-small" text-anchor="middle">1</text>
<line x1="475.0" y1="230.0" x2="475.0" y2="235.0" class="stroke-muted"/>
<text x="475.0" y="250.0" class="fig-small" text-anchor="middle">2</text>
<text x="295.0" y="270.0" class="fig-small" text-anchor="middle">first feature x1 (the other two features are 0)</text>
<text x="14" y="130.0" class="fig-small" text-anchor="middle" transform="rotate(-90 14 130.0)">mean reward</text>
<polyline points="70.0,218.3 520.0,54.0" class="stroke-a" fill="none"/>
<text x="528.0" y="58.0" class="fig-small" text-anchor="start">arm 0</text>
<polyline points="70.0,158.4 520.0,69.1" class="stroke-b" fill="none"/>
<text x="528.0" y="73.1" class="fig-small" text-anchor="start">arm 1</text>
<polyline points="70.0,164.8 520.0,106.0" class="stroke-c" fill="none"/>
<text x="528.0" y="110.0" class="fig-small" text-anchor="start">arm 2</text>
<rect x="70.0" y="282.0" width="359.3" height="10.0" class="fill-b"/>
<rect x="429.3" y="282.0" width="90.7" height="10.0" class="fill-a"/>
<text x="249.6" y="310.0" class="fig-small" text-anchor="middle">arm 1 is best</text>
<text x="474.6" y="310.0" class="fig-small" text-anchor="middle">arm 0 is best</text>
<circle cx="429.3" cy="87.2" r="4" class="fill-ink"/>
</svg>
Figure 1. The exercise's three arms along one slice of context space. Arm 1 pays most when x1 is below 1.49, and arm 0 pays most above it, so no single arm is best for every visitor.
```

You do not know $\theta_a$. You must learn it from the rewards of the rounds where you pulled arm $a$.

## Fit each arm with ridge regression

For each arm, collect the contexts from the rounds you pulled it, and fit its weights by ridge regression. Keep two running sums per arm:

$$
A_a = I + \sum_{t:\, a_t = a} x_t x_t^\top, \qquad b_a = \sum_{t:\, a_t = a} r_t\, x_t, \qquad \hat\theta_a = A_a^{-1} b_a
$$

Read it as: $A_a$ is the identity plus the outer product of every past context, $b_a$ adds up each past context times its reward, and $\hat\theta_a$ solves ridge regression with penalty $\lambda = 1$. The identity matters because it makes $A_a$ invertible before you have any data. With no data, $b_a = 0$ and $\hat\theta_a = 0$, a sensible blank slate.

Here is one arm with $d = 2$. It has seen context $(1, 0)$ with reward 1, and context $(1, 1)$ with reward 0:

$$
A = I + \begin{bmatrix}1&0\\0&0\end{bmatrix} + \begin{bmatrix}1&1\\1&1\end{bmatrix} = \begin{bmatrix}3&1\\1&2\end{bmatrix}, \qquad A^{-1} = \begin{bmatrix}0.4&-0.2\\-0.2&0.6\end{bmatrix}
$$

With $b = (1, 0)$, the estimate is $\hat\theta = A^{-1} b = (0.4, -0.2)$. For a new context $x = (1, 0.5)$, the predicted reward is $0.4 \times 1 - 0.2 \times 0.5 = 0.3$.

## Add optimism where the data is thin

Why not pull the arm with the highest prediction? For the reason greedy fails in a plain bandit. An arm that looks poor after a few unlucky rewards is never pulled again, so you never learn that it was unlucky. You need a bonus that is large where an arm's data is thin. LinUCB (Li, Chu, Langford, and Schapire, 2010) scores each arm as:

$$
\text{score}(a) = \hat\theta_a^\top x + \alpha \sqrt{x^\top A_a^{-1} x}
$$

Read it as: the predicted reward for this context, plus $\alpha$ times a *confidence width*. Here $\alpha$ sets how much you explore. It is not a step size. The width $\sqrt{x^\top A_a^{-1} x}$ is small when many past contexts pointed the same way as $x$, because those contexts made $A_a$ large in that direction, and so $A_a^{-1}$ small. Under a Bayesian reading, with unit-variance noise and a standard-normal prior on $\theta_a$, $A_a^{-1}$ is the posterior covariance of $\theta_a$, so the width is the posterior standard deviation of the prediction.

```python
theta_hat = A_inv @ b
mean = theta_hat @ x
width = np.sqrt(x @ A_inv @ x)
score = mean + alpha * width
```

In the worked example, $x^\top A^{-1} x = 0.4 - 0.2 + 0.15 = 0.35$, so the width is $\sqrt{0.35} = 0.5916$. Figure 2 shows how the width behaves. If you see a unit direction $u$ again and again, $A = I + n\,u u^\top$, and the width in that direction is exactly $\sqrt{1/(1+n)}$. A direction at right angles to $u$ keeps its full width of 1, because nothing you have seen says anything about it.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Line chart of confidence width against the number of observations: it falls along the observed direction and stays at 1 for an orthogonal direction">
<line x1="70.0" y1="210.0" x2="570.0" y2="210.0" class="stroke-grid"/>
<text x="62.0" y="214.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="167.1" x2="570.0" y2="167.1" class="stroke-grid"/>
<text x="62.0" y="171.1" class="fig-small" text-anchor="end">0.25</text>
<line x1="70.0" y1="124.3" x2="570.0" y2="124.3" class="stroke-grid"/>
<text x="62.0" y="128.3" class="fig-small" text-anchor="end">0.5</text>
<line x1="70.0" y1="81.4" x2="570.0" y2="81.4" class="stroke-grid"/>
<text x="62.0" y="85.4" class="fig-small" text-anchor="end">0.75</text>
<line x1="70.0" y1="38.6" x2="570.0" y2="38.6" class="stroke-grid"/>
<text x="62.0" y="42.6" class="fig-small" text-anchor="end">1</text>
<line x1="70.0" y1="210.0" x2="570.0" y2="210.0" class="stroke-muted"/>
<line x1="70.0" y1="30.0" x2="70.0" y2="210.0" class="stroke-muted"/>
<line x1="70.0" y1="210.0" x2="70.0" y2="215.0" class="stroke-muted"/>
<text x="70.0" y="230.0" class="fig-small" text-anchor="middle">0</text>
<line x1="195.0" y1="210.0" x2="195.0" y2="215.0" class="stroke-muted"/>
<text x="195.0" y="230.0" class="fig-small" text-anchor="middle">5</text>
<line x1="320.0" y1="210.0" x2="320.0" y2="215.0" class="stroke-muted"/>
<text x="320.0" y="230.0" class="fig-small" text-anchor="middle">10</text>
<line x1="445.0" y1="210.0" x2="445.0" y2="215.0" class="stroke-muted"/>
<text x="445.0" y="230.0" class="fig-small" text-anchor="middle">15</text>
<line x1="570.0" y1="210.0" x2="570.0" y2="215.0" class="stroke-muted"/>
<text x="570.0" y="230.0" class="fig-small" text-anchor="middle">20</text>
<text x="320.0" y="250.0" class="fig-small" text-anchor="middle">times you have seen the direction u</text>
<text x="14" y="120.0" class="fig-small" text-anchor="middle" transform="rotate(-90 14 120.0)">confidence width</text>
<polyline points="70.0,38.6 95.0,88.8 120.0,111.0 145.0,124.3 170.0,133.3 195.0,140.0 220.0,145.2 245.0,149.4 270.0,152.9 295.0,155.8 320.0,158.3 345.0,160.5 370.0,162.5 395.0,164.2 420.0,165.7 445.0,167.1 470.0,168.4 495.0,169.6 520.0,170.7 545.0,171.7 570.0,172.6" class="stroke-a" fill="none"/>
<circle cx="70.0" cy="38.6" r="3" class="fill-a"/>
<circle cx="95.0" cy="88.8" r="3" class="fill-a"/>
<circle cx="120.0" cy="111.0" r="3" class="fill-a"/>
<circle cx="145.0" cy="124.3" r="3" class="fill-a"/>
<circle cx="170.0" cy="133.3" r="3" class="fill-a"/>
<circle cx="195.0" cy="140.0" r="3" class="fill-a"/>
<circle cx="220.0" cy="145.2" r="3" class="fill-a"/>
<circle cx="245.0" cy="149.4" r="3" class="fill-a"/>
<circle cx="270.0" cy="152.9" r="3" class="fill-a"/>
<circle cx="295.0" cy="155.8" r="3" class="fill-a"/>
<circle cx="320.0" cy="158.3" r="3" class="fill-a"/>
<circle cx="345.0" cy="160.5" r="3" class="fill-a"/>
<circle cx="370.0" cy="162.5" r="3" class="fill-a"/>
<circle cx="395.0" cy="164.2" r="3" class="fill-a"/>
<circle cx="420.0" cy="165.7" r="3" class="fill-a"/>
<circle cx="445.0" cy="167.1" r="3" class="fill-a"/>
<circle cx="470.0" cy="168.4" r="3" class="fill-a"/>
<circle cx="495.0" cy="169.6" r="3" class="fill-a"/>
<circle cx="520.0" cy="170.7" r="3" class="fill-a"/>
<circle cx="545.0" cy="171.7" r="3" class="fill-a"/>
<circle cx="570.0" cy="172.6" r="3" class="fill-a"/>
<line x1="70.0" y1="38.6" x2="570.0" y2="38.6" class="stroke-b"/>
<text x="566.0" y="56.6" class="fig-small" text-anchor="end">a direction at right angles to u: width stays 1</text>
<text x="220.0" y="133.2" class="fig-small" text-anchor="start">direction u: sqrt(1/(1+n)), 0.378 at n = 6</text>
</svg>
Figure 2. Width sqrt(x A⁻¹ x) after A = I + n u uᵀ, for a unit vector u. Seeing a direction n times shrinks the width in that direction to sqrt(1/(1+n)), and leaves directions at right angles untouched, so LinUCB keeps exploring where it has no data.
```

> [!KEY]
> LinUCB is UCB1 in feature space. The prediction comes from a ridge regression per arm, and the bonus is the width of its confidence region in the direction of the current context.

## Update the inverse without inverting

Each round adds one outer product to one arm's $A$. Solving with that $A$ from scratch costs time proportional to $d^3$. The Sherman-Morrison formula gives the new inverse directly from the old one:

$$
\left(A + x x^\top\right)^{-1} = A^{-1} - \frac{u\, u^\top}{1 + x^\top u}, \qquad u = A^{-1} x
$$

Read it as: apply the old inverse to the new context to get $u$, then subtract a rank-one correction. That costs time proportional to $d^2$. In the worked example, adding $x = (1, 0.5)$ gives $u = (0.3, 0.1)$ and $x^\top u = 0.35$. The updated inverse is $A^{-1} - u u^\top / 1.35$, with entries $0.3333$, $-0.2222$, $-0.2222$, and $0.5926$, which matches `np.linalg.inv` of $A + x x^\top$. The width for that same context shrinks from $0.5916$ to $\sqrt{0.35 / 1.35} = 0.5092$, because you have seen it once.

```python
u = A_inv @ x
A_inv = A_inv - np.outer(u, u) / (1.0 + x @ u)
```

## The full loop

Each round, observe $x$, score every arm, and pull the arm with the highest score. Then update only that arm: `A_inv[arm]` with Sherman-Morrison and `b[arm] += reward * x`. Figure 3 shows the result on the exercise's world with $\alpha = 1$. LinUCB's regret after 2000 rounds is 13.0, compared with about 1219 for the best arm that ignores context. Setting $\alpha = 0$ on the same run gives 30.7: the bonus pays for itself.

```figure
<svg viewBox="0 0 600 285" role="img" aria-label="Cumulative regret over 2000 rounds: the context-blind arm climbs to about 1219 while LinUCB stays near 13">
<line x1="70.0" y1="230.0" x2="570.0" y2="230.0" class="stroke-grid"/>
<text x="62.0" y="234.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="191.5" x2="570.0" y2="191.5" class="stroke-grid"/>
<text x="62.0" y="195.5" class="fig-small" text-anchor="end">250</text>
<line x1="70.0" y1="153.1" x2="570.0" y2="153.1" class="stroke-grid"/>
<text x="62.0" y="157.1" class="fig-small" text-anchor="end">500</text>
<line x1="70.0" y1="114.6" x2="570.0" y2="114.6" class="stroke-grid"/>
<text x="62.0" y="118.6" class="fig-small" text-anchor="end">750</text>
<line x1="70.0" y1="76.2" x2="570.0" y2="76.2" class="stroke-grid"/>
<text x="62.0" y="80.2" class="fig-small" text-anchor="end">1000</text>
<line x1="70.0" y1="37.7" x2="570.0" y2="37.7" class="stroke-grid"/>
<text x="62.0" y="41.7" class="fig-small" text-anchor="end">1250</text>
<line x1="70.0" y1="230.0" x2="570.0" y2="230.0" class="stroke-muted"/>
<line x1="70.0" y1="30.0" x2="70.0" y2="230.0" class="stroke-muted"/>
<line x1="70.0" y1="230.0" x2="70.0" y2="235.0" class="stroke-muted"/>
<text x="70.0" y="250.0" class="fig-small" text-anchor="middle">0</text>
<line x1="195.0" y1="230.0" x2="195.0" y2="235.0" class="stroke-muted"/>
<text x="195.0" y="250.0" class="fig-small" text-anchor="middle">500</text>
<line x1="320.0" y1="230.0" x2="320.0" y2="235.0" class="stroke-muted"/>
<text x="320.0" y="250.0" class="fig-small" text-anchor="middle">1000</text>
<line x1="445.0" y1="230.0" x2="445.0" y2="235.0" class="stroke-muted"/>
<text x="445.0" y="250.0" class="fig-small" text-anchor="middle">1500</text>
<line x1="570.0" y1="230.0" x2="570.0" y2="235.0" class="stroke-muted"/>
<text x="570.0" y="250.0" class="fig-small" text-anchor="middle">2000</text>
<text x="320.0" y="270.0" class="fig-small" text-anchor="middle">round</text>
<text x="14" y="130.0" class="fig-small" text-anchor="middle" transform="rotate(-90 14 130.0)">cumulative regret</text>
<polyline points="70.2,229.9 75.2,228.0 80.2,226.2 85.2,224.3 90.2,222.4 95.2,220.5 100.2,218.7 105.2,216.8 110.2,214.9 115.2,213.0 120.2,211.2 125.2,209.3 130.2,207.4 135.2,205.5 140.2,203.7 145.2,201.8 150.2,199.9 155.2,198.0 160.2,196.2 165.2,194.3 170.2,192.4 175.2,190.5 180.2,188.7 185.2,186.8 190.2,184.9 195.2,183.0 200.2,181.2 205.2,179.3 210.2,177.4 215.2,175.5 220.2,173.7 225.2,171.8 230.2,169.9 235.2,168.0 240.2,166.2 245.2,164.3 250.2,162.4 255.2,160.5 260.2,158.7 265.2,156.8 270.2,154.9 275.2,153.0 280.2,151.2 285.2,149.3 290.2,147.4 295.2,145.5 300.2,143.7 305.2,141.8 310.2,139.9 315.2,138.0 320.2,136.2 325.2,134.3 330.2,132.4 335.2,130.5 340.2,128.7 345.2,126.8 350.2,124.9 355.2,123.0 360.2,121.2 365.2,119.3 370.2,117.4 375.2,115.5 380.2,113.7 385.2,111.8 390.2,109.9 395.2,108.0 400.2,106.2 405.2,104.3 410.2,102.4 415.2,100.5 420.2,98.7 425.2,96.8 430.2,94.9 435.2,93.0 440.2,91.2 445.2,89.3 450.2,87.4 455.2,85.5 460.2,83.7 465.2,81.8 470.2,79.9 475.2,78.0 480.2,76.2 485.2,74.3 490.2,72.4 495.2,70.5 500.2,68.7 505.2,66.8 510.2,64.9 515.2,63.0 520.2,61.2 525.2,59.3 530.2,57.4 535.2,55.5 540.2,53.7 545.2,51.8 550.2,49.9 555.2,48.0 560.2,46.2 565.2,44.3 570.0,42.5" class="stroke-b" fill="none"/>
<polyline points="70.2,229.7 75.2,229.0 80.2,228.9 85.2,228.9 90.2,228.7 95.2,228.7 100.2,228.7 105.2,228.7 110.2,228.6 115.2,228.5 120.2,228.4 125.2,228.4 130.2,228.4 135.2,228.4 140.2,228.3 145.2,228.3 150.2,228.3 155.2,228.3 160.2,228.3 165.2,228.3 170.2,228.2 175.2,228.2 180.2,228.2 185.2,228.2 190.2,228.2 195.2,228.2 200.2,228.2 205.2,228.2 210.2,228.2 215.2,228.2 220.2,228.2 225.2,228.2 230.2,228.1 235.2,228.1 240.2,228.1 245.2,228.1 250.2,228.1 255.2,228.1 260.2,228.1 265.2,228.1 270.2,228.1 275.2,228.1 280.2,228.1 285.2,228.1 290.2,228.1 295.2,228.1 300.2,228.1 305.2,228.1 310.2,228.1 315.2,228.1 320.2,228.1 325.2,228.1 330.2,228.1 335.2,228.1 340.2,228.1 345.2,228.1 350.2,228.1 355.2,228.1 360.2,228.1 365.2,228.1 370.2,228.1 375.2,228.1 380.2,228.1 385.2,228.1 390.2,228.1 395.2,228.1 400.2,228.0 405.2,228.0 410.2,228.0 415.2,228.0 420.2,228.0 425.2,228.0 430.2,228.0 435.2,228.0 440.2,228.0 445.2,228.0 450.2,228.0 455.2,228.0 460.2,228.0 465.2,228.0 470.2,228.0 475.2,228.0 480.2,228.0 485.2,228.0 490.2,228.0 495.2,228.0 500.2,228.0 505.2,228.0 510.2,228.0 515.2,228.0 520.2,228.0 525.2,228.0 530.2,228.0 535.2,228.0 540.2,228.0 545.2,228.0 550.2,228.0 555.2,228.0 560.2,228.0 565.2,228.0 570.0,228.0" class="stroke-a" fill="none"/>
<text x="100.0" y="46.9" class="fig-small" text-anchor="start">best arm that ignores context: 1219 at round 2000</text>
<text x="564.0" y="218.0" class="fig-small" text-anchor="end">LinUCB: 13.0</text>
</svg>
Figure 3. Cumulative regret over 2000 rounds. The best single arm loses about 0.61 per round, so its line climbs to about 1219. LinUCB loses 8.2 in the first 100 rounds and 13.0 in total, because it learns each arm's weights and then picks the arm that suits each context.
```

> [!WARNING]
> Keep one $A^{-1}$ and one $b$ per arm. If you share them across arms, every arm learns the same weights, and the arms become indistinguishable.

> [!RECAP]
> - Each arm keeps `A_inv` (shape `(d, d)`, starting at the identity) and `b` (shape `(d,)`, starting at zero).
> - The prediction is `(A_inv @ b) @ x` and the width is `np.sqrt(x @ A_inv @ x)`.
> - The score is the prediction plus `alpha` times the width, and you pull the arm with the largest score.
> - Sherman-Morrison updates `A_inv` with `u = A_inv @ x`.

## Build it step by step

The starter gives you the environment `LinearContextualBandit`. The checks test your functions in this order.

1. **`linucb_terms(A_inv, b, x)`.** Takes one arm's `A_inv` `(d, d)`, `b` `(d,)`, and a context `x` `(d,)`. Returns `(mean, width)` as floats. Common mistake: computing the estimate as `A_inv @ x` instead of `A_inv @ b`.
2. **`sherman_morrison(A_inv, x)`.** Two lines: `u = A_inv @ x`, then the rank-one correction with `np.outer(u, u)`. Common mistake: dividing by `x @ u` without the `1 +`.
3. **`run_linucb(bandit, T, alpha)`.** Arrays `A_inv` of shape `(k, d, d)`, starting as identities, and `b` of shape `(k, d)`, starting at zero. Each round: `x = bandit.observe()`, score every arm with `linucb_terms`, pull the best, update that arm only, and add `bandit.regret(arm)` to a running total. Return the `(T,)` array of cumulative regret.

## Cost and scaling

Scoring all arms costs $O(k d^2)$ per round, and the Sherman-Morrison update costs $O(d^2)$, against $O(d^3)$ for re-solving. Memory is $k$ matrices of $d^2$ entries. With $d$ in the thousands, even $d^2$ per arm hurts, so large systems use a diagonal approximation of $A$ or share one model across arms by putting the arm into the features.

## Where this breaks

**A model that is not linear.** LinUCB's widths are only honest when rewards really are linear in the features. When they are not, the regression is confidently wrong, and the bonus does not fix it. Richer features help, and so do neural networks with approximate uncertainty.

**Choosing alpha.** The theory gives a width multiplier that depends on the noise level, the dimension, and your confidence target, but the constants are loose. In practice, people tune $\alpha$ on logged data.

## Further reading

- [Sutton and Barto, *Reinforcement Learning: An Introduction*, Section 2.9](http://incompleteideas.net/book/the-book-2nd.html) introduces contextual bandits under the name *associative search*.
- [Li, Chu, Langford, and Schapire, "A contextual-bandit approach to personalized news article recommendation" (2010)](https://arxiv.org/abs/1003.0146) introduces LinUCB and tests it on Yahoo! front-page news.
- [Abbasi-Yadkori, Pál, and Szepesvári, "Improved algorithms for linear stochastic bandits" (2011)](https://papers.nips.cc/paper/2011/hash/e1d5be1c7f2f456670de3d53c7b54f4a-Abstract.html) proves a tight confidence ellipsoid for the ridge estimate and regret bounds for the resulting algorithm.
- [Lattimore and Szepesvári, *Bandit Algorithms* (2020), Chapters 19 and 20](https://tor-lattimore.com/downloads/book/book.pdf) work through linear bandits and their confidence bounds in full.

## The exercise

You build the LinUCB learner. Write `linucb_terms`, `sherman_morrison`, and `run_linucb`. The environment `LinearContextualBandit` is given in the starter file.

The checks verify `linucb_terms` against the worked example above, against a ridge fit on 30 random rows, and against the closed-form width $\sqrt{1/(1+n)}$ for a context seen $n$ times. They check `sherman_morrison` on the worked update and against `np.linalg.inv` after twenty updates. Last, `run_linucb` must end 2000 rounds with less than a tenth of the best context-blind arm's regret, and with much less regret in the second half than in the first.
