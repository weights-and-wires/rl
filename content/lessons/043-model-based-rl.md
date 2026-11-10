---
title: Learning a model and planning with it
slug: 043-model-based-rl
part: "Part 7 — Models, exploration, and fixed data"
summary: Fit a dynamics model by regression, measure how its error compounds over a rollout, and plan through it with CEM and model-predictive control.
minutes: 50
gpu: false
objectives:
  - Fit a linear dynamics model by ridge regression and predict next states with it.
  - Explain with the error recursion why open-loop prediction error grows with the horizon, and measure it on CartPole.
  - Implement the cross-entropy method and use it for model-predictive control that balances CartPole.
  - Say why replanning at every step lets a model that drifts over long horizons still control well.
prereqs:
  - 021-dyna-planning-with-a-learned-model
  - 022-monte-carlo-tree-search
  - 026-deep-q-networks
exercise: 043-model-based-rl
---

# Learning a model and planning with it

Every method so far learns a policy or a value function straight from experience. Each real step feeds one update and is then mostly forgotten. On a simulator that costs nothing. On a robot, a real step takes time, wears out motors, and can break things, and model-free methods such as PPO often need millions of steps.

A model-based method first learns how the world moves: given this state and this action, where do I end up? That is ordinary supervised learning, with one labeled example per real step. Once the model exists, the agent can try thousands of imagined action sequences inside it and pick the best, without touching the real system.

The catch is that the model is wrong, and more wrong the further ahead you ask it to look. This chapter fits a model, measures how its error grows, and then plans through it anyway with a loop that corrects itself at every step. It balances CartPole with a model fitted to 2,000 random transitions.

## What you already know

- A *policy* $\pi(a \mid s)$ picks actions, and the *return* $G_t$ adds up discounted rewards.
- Dyna learns a model from real steps and replays it to update values (see [Planning with a learned model: Dyna](/lessons/021-dyna-planning-with-a-learned-model)).
- Monte Carlo tree search plans by simulating ahead with a model it is given (see [Monte Carlo tree search](/lessons/022-monte-carlo-tree-search)).
- CartPole pays 1 per step while the pole stays within 12 degrees of upright and the cart stays within 2.4 m of center (see [Deep Q-networks](/lessons/026-deep-q-networks)).

## Learn the dynamics by regression

A *dynamics model* $\hat f$ predicts the next state from the current state and action, $\hat s_{t+1} = \hat f(s_t, a_t)$. Every real step $(s_t, a_t, s_{t+1})$ is a training example, with the next state as the label. A linear model is the easiest place to start:

$$
\hat s_{t+1} = A s_t + B a_t + c
$$

Stack each input as a row $x = [s, a, 1]$. Then the whole model is one weight matrix $W$ of shape $(d + k + 1, d)$ for a state of size $d$ and an action of size $k$, and a prediction is $xW$. Ridge regression on $N$ transitions gives

$$
W = (X^\top X + \lambda I)^{-1} X^\top Y
$$

Read it as follows. $X$ stacks the $N$ input rows, $Y$ stacks the $N$ observed next states, and $\lambda \ge 0$ is the ridge penalty that shrinks the weights. Setting the gradient $X^\top(XW - Y) + \lambda W$ to zero gives this closed form. In numpy:

```python
X = np.concatenate([states, actions, np.ones((N, 1))], axis=1)
W = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ next_states)
```

Check it by hand on a one-dimensional system with true rule $s_{t+1} = 0.9 s_t + 0.5 a_t + 0.1$. Three transitions pin down three weights: $(s, a) = (0, 1)$ gives $0.6$, $(1, 0)$ gives $1.0$, and $(2, 1)$ gives $2.4$. With $\lambda = 0$ the solve returns $W = [0.9, 0.5, 0.1]^\top$: the weight on $s$, on $a$, and the bias. 

CartPole is not linear, because the pole angle enters through a sine and a cosine. Near upright, though, $\sin\theta \approx \theta$ and $\cos\theta \approx 1$, so a linear model fitted near upright predicts one step ahead well. Its mean one-step error is about $4 \times 10^{-4}$.

## Watch the error compound

A one-step error of $4 \times 10^{-4}$ sounds harmless. Planning asks a longer question: if I take these 30 actions, where do I end up? The model answers by feeding its own prediction back in as the next input. That is an *open-loop rollout*, because nothing from the real world corrects it on the way:

```python
states = [s0]
for t in range(H):
    states.append(step_fn(states[-1], actions[:, t]))
traj = np.stack(states, axis=1)          # (N, H + 1, d)
```

Each step inherits the previous step's error and adds its own. For a linear model with learned $\hat A, \hat B$ and truth $A, B$, the error $e_k = \hat s_k - s_k$ obeys

$$
e_k = \hat A \, e_{k-1} + (\hat A - A)\, s_{k-1} + (\hat B - B)\, a_{k-1}
$$

Read it in two parts. The first term carries the old error forward through the model's own dynamics, which can grow it or shrink it. The other two terms are the fresh one-step error at step $k$, made even from the correct state.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Log plot of the k-step error of a model that overestimates a growth rate by 0.01, for true rates 0.9, 1.0, and 1.1: it stays near 0.04 for 0.9, grows to 0.64 for 1.0, and reaches 67 for 1.1">
<line x1="80.0" y1="250.0" x2="560.0" y2="250.0" class="stroke-grid"/>
<text x="72.0" y="254.0" class="fig-small" text-anchor="end">1e-3</text>
<line x1="80.0" y1="208.0" x2="560.0" y2="208.0" class="stroke-grid"/>
<text x="72.0" y="212.0" class="fig-small" text-anchor="end">1e-2</text>
<line x1="80.0" y1="166.0" x2="560.0" y2="166.0" class="stroke-grid"/>
<text x="72.0" y="170.0" class="fig-small" text-anchor="end">1e-1</text>
<line x1="80.0" y1="124.0" x2="560.0" y2="124.0" class="stroke-grid"/>
<text x="72.0" y="128.0" class="fig-small" text-anchor="end">1</text>
<line x1="80.0" y1="82.0" x2="560.0" y2="82.0" class="stroke-grid"/>
<text x="72.0" y="86.0" class="fig-small" text-anchor="end">1e1</text>
<line x1="80.0" y1="40.0" x2="560.0" y2="40.0" class="stroke-grid"/>
<text x="72.0" y="44.0" class="fig-small" text-anchor="end">1e2</text>
<line x1="80.0" y1="250.0" x2="560.0" y2="250.0" class="stroke-muted"/>
<line x1="89.6" y1="250.0" x2="89.6" y2="255.0" class="stroke-muted"/>
<text x="89.6" y="270.0" class="fig-small" text-anchor="middle">1</text>
<line x1="176.0" y1="250.0" x2="176.0" y2="255.0" class="stroke-muted"/>
<text x="176.0" y="270.0" class="fig-small" text-anchor="middle">10</text>
<line x1="272.0" y1="250.0" x2="272.0" y2="255.0" class="stroke-muted"/>
<text x="272.0" y="270.0" class="fig-small" text-anchor="middle">20</text>
<line x1="368.0" y1="250.0" x2="368.0" y2="255.0" class="stroke-muted"/>
<text x="368.0" y="270.0" class="fig-small" text-anchor="middle">30</text>
<line x1="464.0" y1="250.0" x2="464.0" y2="255.0" class="stroke-muted"/>
<text x="464.0" y="270.0" class="fig-small" text-anchor="middle">40</text>
<line x1="560.0" y1="250.0" x2="560.0" y2="255.0" class="stroke-muted"/>
<text x="560.0" y="270.0" class="fig-small" text-anchor="middle">50</text>
<text x="320.0" y="288.0" text-anchor="middle">steps ahead, k</text>
<text x="72.0" y="24.0" class="fig-small" text-anchor="end">error</text>
<polyline points="89.6,208.0 99.2,197.2 108.8,191.6 118.4,188.2 128.0,185.9 137.6,184.4 147.2,183.4 156.8,182.8 166.4,182.5 176.0,182.4 185.6,182.5 195.2,182.7 204.8,183.1 214.4,183.5 224.0,184.1 233.6,184.7 243.2,185.4 252.8,186.2 262.4,187.0 272.0,187.9 281.6,188.8 291.2,189.8 300.8,190.8 310.4,191.9 320.0,192.9 329.6,194.0 339.2,195.2 348.8,196.3 358.4,197.5 368.0,198.7 377.6,199.9 387.2,201.1 396.8,202.4 406.4,203.7 416.0,205.0 425.6,206.3 435.2,207.6 444.8,208.9 454.4,210.2 464.0,211.6 473.6,212.9 483.2,214.3 492.8,215.7 502.4,217.1 512.0,218.5 521.6,219.9 531.2,221.3 540.8,222.8 550.4,224.2 560.0,225.6" class="stroke-c"/>
<polyline points="89.6,208.0 99.2,195.3 108.8,187.8 118.4,182.4 128.0,178.3 137.6,174.9 147.2,172.0 156.8,169.4 166.4,167.2 176.0,165.2 185.6,163.3 195.2,161.7 204.8,160.1 214.4,158.7 224.0,157.3 233.6,156.0 243.2,154.8 252.8,153.7 262.4,152.6 272.0,151.6 281.6,150.6 291.2,149.7 300.8,148.8 310.4,147.9 320.0,147.1 329.6,146.3 339.2,145.5 348.8,144.7 358.4,144.0 368.0,143.3 377.6,142.6 387.2,141.9 396.8,141.2 406.4,140.6 416.0,140.0 425.6,139.4 435.2,138.8 444.8,138.2 454.4,137.6 464.0,137.1 473.6,136.5 483.2,136.0 492.8,135.4 502.4,134.9 512.0,134.4 521.6,133.9 531.2,133.4 540.8,132.9 550.4,132.5 560.0,132.0" class="stroke-a"/>
<polyline points="89.6,208.0 99.2,193.5 108.8,184.3 118.4,177.2 128.0,171.4 137.6,166.2 147.2,161.6 156.8,157.3 166.4,153.3 176.0,149.6 185.6,146.0 195.2,142.6 204.8,139.4 214.4,136.2 224.0,133.1 233.6,130.1 243.2,127.2 252.8,124.3 262.4,121.5 272.0,118.7 281.6,116.0 291.2,113.3 300.8,110.7 310.4,108.1 320.0,105.5 329.6,103.0 339.2,100.5 348.8,98.0 358.4,95.5 368.0,93.1 377.6,90.7 387.2,88.3 396.8,85.9 406.4,83.5 416.0,81.2 425.6,78.8 435.2,76.5 444.8,74.2 454.4,71.9 464.0,69.6 473.6,67.3 483.2,65.1 492.8,62.8 502.4,60.6 512.0,58.3 521.6,56.1 531.2,53.9 540.8,51.7 550.4,49.5 560.0,47.3" class="stroke-b"/>
<text x="554.0" y="39.3" class="fig-small" text-anchor="end">rate 1.1: 67.2</text>
<text x="554.0" y="124.0" class="fig-small" text-anchor="end">rate 1.0: 0.645</text>
<text x="554.0" y="243.6" class="fig-small" text-anchor="end">rate 0.9: 0.0038</text>
</svg>
Figure 1. The same one-step error of 0.01, rolled out k steps, for three true growth rates. Notice that the stable system forgets the error, the neutral one lets it add up, and the unstable one multiplies it, so 0.01 becomes 67 after 50 steps.
```

Take a scalar system $s_k = a\, s_{k-1}$ with $s_0 = 1$ and a model that overestimates the rate by 0.01, so $\hat a = a + 0.01$. The one-step error is always 0.01, and the $k$-step error is $\hat a^k - a^k$. Figure 1 shows three cases. For $a = 0.9$, the system is stable, so it forgets old errors and the error rises to about 0.04, then dies away. For $a = 1$ the error is $1.01^k - 1$, which is $0.645$ at $k = 50$. For $a = 1.1$ the system multiplies errors, and the prediction at $k = 50$ is off by 67.

An upright pole is unstable, so CartPole behaves like the unstable case. Figure 2 measures it. The linear model's mean error under random pushes is $4.3 \times 10^{-4}$ after one step, $2.1 \times 10^{-3}$ after 10, $0.036$ after 20, and $0.34$ after 30, about 780 times the one-step error.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Log plot of the mean k-step prediction error of a linear model on CartPole: 0.00043 after one step, 0.0021 after 10, 0.036 after 20, and 0.34 after 30">
<line x1="80.0" y1="250.0" x2="560.0" y2="250.0" class="stroke-grid"/>
<text x="72.0" y="254.0" class="fig-small" text-anchor="end">1e-4</text>
<line x1="80.0" y1="197.5" x2="560.0" y2="197.5" class="stroke-grid"/>
<text x="72.0" y="201.5" class="fig-small" text-anchor="end">1e-3</text>
<line x1="80.0" y1="145.0" x2="560.0" y2="145.0" class="stroke-grid"/>
<text x="72.0" y="149.0" class="fig-small" text-anchor="end">1e-2</text>
<line x1="80.0" y1="92.5" x2="560.0" y2="92.5" class="stroke-grid"/>
<text x="72.0" y="96.5" class="fig-small" text-anchor="end">1e-1</text>
<line x1="80.0" y1="40.0" x2="560.0" y2="40.0" class="stroke-grid"/>
<text x="72.0" y="44.0" class="fig-small" text-anchor="end">1</text>
<line x1="80.0" y1="250.0" x2="560.0" y2="250.0" class="stroke-muted"/>
<line x1="96.0" y1="250.0" x2="96.0" y2="255.0" class="stroke-muted"/>
<text x="96.0" y="270.0" class="fig-small" text-anchor="middle">1</text>
<line x1="160.0" y1="250.0" x2="160.0" y2="255.0" class="stroke-muted"/>
<text x="160.0" y="270.0" class="fig-small" text-anchor="middle">5</text>
<line x1="240.0" y1="250.0" x2="240.0" y2="255.0" class="stroke-muted"/>
<text x="240.0" y="270.0" class="fig-small" text-anchor="middle">10</text>
<line x1="320.0" y1="250.0" x2="320.0" y2="255.0" class="stroke-muted"/>
<text x="320.0" y="270.0" class="fig-small" text-anchor="middle">15</text>
<line x1="400.0" y1="250.0" x2="400.0" y2="255.0" class="stroke-muted"/>
<text x="400.0" y="270.0" class="fig-small" text-anchor="middle">20</text>
<line x1="480.0" y1="250.0" x2="480.0" y2="255.0" class="stroke-muted"/>
<text x="480.0" y="270.0" class="fig-small" text-anchor="middle">25</text>
<line x1="560.0" y1="250.0" x2="560.0" y2="255.0" class="stroke-muted"/>
<text x="560.0" y="270.0" class="fig-small" text-anchor="middle">30</text>
<text x="320.0" y="288.0" text-anchor="middle">steps ahead, k</text>
<text x="72.0" y="24.0" class="fig-small" text-anchor="end">mean error</text>
<polyline points="96.0,216.7 112.0,213.6 128.0,207.7 144.0,205.2 160.0,202.4 176.0,201.3 192.0,198.6 208.0,192.3 224.0,190.3 240.0,181.0 256.0,171.5 272.0,164.0 288.0,156.1 304.0,144.6 320.0,139.3 336.0,134.2 352.0,128.4 368.0,126.2 384.0,119.0 400.0,115.9 416.0,108.3 432.0,102.6 448.0,99.0 464.0,95.3 480.0,90.0 496.0,83.7 512.0,79.4 528.0,73.6 544.0,68.9 560.0,64.9" class="stroke-b"/>
<circle cx="96.0" cy="216.7" r="5" class="fill-b"/>
<circle cx="240.0" cy="181.0" r="5" class="fill-b"/>
<circle cx="400.0" cy="115.9" r="5" class="fill-b"/>
<circle cx="560.0" cy="64.9" r="5" class="fill-b"/>
<text x="110.0" y="60.0" class="fig-small" text-anchor="start">k = 1: 0.00043</text>
<text x="110.0" y="80.0" class="fig-small" text-anchor="start">k = 10: 0.0021</text>
<text x="110.0" y="100.0" class="fig-small" text-anchor="start">k = 20: 0.036</text>
<text x="110.0" y="120.0" class="fig-small" text-anchor="start">k = 30: 0.34</text>
</svg>
Figure 2. A linear model fitted to 2,000 random CartPole transitions, rolled out open loop under random pushes. Notice that the error is tiny after one step and about 780 times larger after 30, because an upright pole is unstable.
```

> [!KEY]
> A model trained to be accurate one step ahead can be badly wrong many steps ahead, because each step feeds its error into the next. Unstable dynamics can make the growth exponential.

## Plan by random search: the cross-entropy method

Once you have a model, planning means choosing actions $a_0, \ldots, a_{H-1}$ whose imagined rollout scores well. Score a plan by running the model forward from the real state $s$ and adding up a planning reward $r$ at every state it lands in:

$$
J(a_{0:H-1}) = \sum_{t=0}^{H-1} r(\hat s_{t+1}), \qquad \hat s_0 = s, \quad \hat s_{t+1} = \hat f(\hat s_t, a_t)
$$

For CartPole, $r$ is high when the pole is upright and the cart is centered, and strongly negative once the rollout crosses a failure line. Why not follow the gradient of $J$ to improve the plan? CartPole's actions are discrete, so $J$ has no useful gradient. The *cross-entropy method* (CEM) needs none. It is guided random search:

1. Keep a Gaussian over plans, with a mean and standard deviation per step.
2. Sample $n$ plans and score each with the model.
3. Keep the $n_e$ best plans, the *elites*.
4. Refit the Gaussian to the elites, and repeat.

The refit is $\mu \leftarrow \tfrac{1}{n_e}\sum_{i \in E} u_i$ and $\sigma^2 \leftarrow \tfrac{1}{n_e}\sum_{i \in E}(u_i - \mu)^2$. Read it as: the next search centers on the average of the winners and spreads as widely as the winners disagree, so when the elites agree, the search narrows. In numpy:

```python
elite = samples[np.argsort(-scores)[:n_elite]]
mean, std = elite.mean(axis=0), elite.std(axis=0)
```

For CartPole, a plan is a real vector $u$ of length $H$, and step $t$ pushes right when $u_t > 0$ and left otherwise. Now work one refit by hand, maximizing $-(u - 1)^2$ over a scalar $u$ with five samples and two elites. The samples $-1, 0.5, 2, 1.2, 0.3$ score $-4, -0.25, -1, -0.04, -0.49$. The two best are $1.2$ and $0.5$. Their mean is $0.85$ and their standard deviation is $\sqrt{((1.2-0.85)^2 + (0.5-0.85)^2)/2} = 0.35$. One refit moves the search center most of the way to the optimum at 1, as Figure 3 shows.

```figure
<svg viewBox="0 0 600 280" role="img" aria-label="One cross-entropy refit on the score -(u - 1)^2: five samples, the two best are 1.2 and 0.5, and the new mean is 0.85">
<line x1="60.0" y1="200.0" x2="560.0" y2="200.0" class="stroke-muted"/>
<polyline points="110.0,196.8 115.0,190.9 120.0,185.1 125.0,179.4 130.0,173.8 135.0,168.4 140.0,163.0 145.0,157.8 150.0,152.7 155.0,147.6 160.0,142.8 165.0,138.0 170.0,133.3 175.0,128.8 180.0,124.3 185.0,120.0 190.0,115.8 195.0,111.7 200.0,107.7 205.0,103.8 210.0,100.1 215.0,96.4 220.0,92.9 225.0,89.5 230.0,86.2 235.0,83.0 240.0,80.0 245.0,77.0 250.0,74.1 255.0,71.4 260.0,68.8 265.0,66.3 270.0,63.9 275.0,61.6 280.0,59.5 285.0,57.4 290.0,55.5 295.0,53.7 300.0,52.0 305.0,50.4 310.0,48.9 315.0,47.5 320.0,46.3 325.0,45.1 330.0,44.1 335.0,43.2 340.0,42.4 345.0,41.7 350.0,41.2 355.0,40.7 360.0,40.4 365.0,40.1 370.0,40.0 375.0,40.0 380.0,40.1 385.0,40.4 390.0,40.7 395.0,41.2 400.0,41.7 405.0,42.4 410.0,43.2 415.0,44.1 420.0,45.1 425.0,46.3 430.0,47.5 435.0,48.9 440.0,50.4 445.0,52.0 450.0,53.7 455.0,55.5 460.0,57.4 465.0,59.5 470.0,61.6 475.0,63.9 480.0,66.3 485.0,68.8 490.0,71.4 495.0,74.1 500.0,77.0 505.0,80.0 510.0,83.0 515.0,86.2 520.0,89.5 525.0,92.9 530.0,96.4 535.0,100.1 540.0,103.8 545.0,107.7 550.0,111.7 555.0,115.8 560.0,120.0" class="stroke-a"/>
<line x1="122.5" y1="200.0" x2="122.5" y2="205.0" class="stroke-muted"/>
<text x="122.5" y="220.0" class="fig-small" text-anchor="middle">-1</text>
<line x1="247.5" y1="200.0" x2="247.5" y2="205.0" class="stroke-muted"/>
<text x="247.5" y="220.0" class="fig-small" text-anchor="middle">0</text>
<line x1="372.5" y1="200.0" x2="372.5" y2="205.0" class="stroke-muted"/>
<text x="372.5" y="220.0" class="fig-small" text-anchor="middle">1</text>
<line x1="497.5" y1="200.0" x2="497.5" y2="205.0" class="stroke-muted"/>
<text x="497.5" y="220.0" class="fig-small" text-anchor="middle">2</text>
<text x="310.0" y="242.0" text-anchor="middle">sample u (score is -(u - 1)²)</text>
<circle cx="122.5" cy="182.2" r="6" class="fill-muted"/>
<text x="110.5" y="178.2" class="fig-small" text-anchor="end">-1: -4.00</text>
<circle cx="310.0" cy="48.9" r="6" class="fill-b"/>
<text x="304.0" y="34.9" class="fig-small" text-anchor="end">0.5: -0.25</text>
<circle cx="497.5" cy="75.6" r="6" class="fill-muted"/>
<text x="506.5" y="67.6" class="fig-small" text-anchor="start">2: -1.00</text>
<circle cx="397.5" cy="41.4" r="6" class="fill-b"/>
<text x="406.5" y="29.4" class="fig-small" text-anchor="start">1.2: -0.04</text>
<circle cx="285.0" cy="57.4" r="6" class="fill-muted"/>
<text x="295.0" y="83.4" class="fig-small" text-anchor="start">0.3: -0.49</text>
<line x1="353.8" y1="198.0" x2="353.8" y2="54.2" class="stroke-c"/>
<text x="361.8" y="153.8" class="fig-small" text-anchor="start">new mean 0.85, std 0.35</text>
</svg>
Figure 3. One CEM refit on a score with its peak at u = 1. Notice that the two best samples, in orange, sit on either side of the peak, so their mean, 0.85, lands near it, and their spread, 0.35, sets the next search width.
```

## Replan at every step: model-predictive control

Figure 2 says a 30-step plan rests on predictions that are badly wrong by the end. *Model-predictive control* (MPC) sidesteps most of that damage. At every real step it does the following:

1. Plan $H$ steps ahead from the real current state, using CEM through the model.
2. Execute only the first action of the best plan.
3. Observe the real next state, throw away the rest of the plan, and plan again.

The window slides forward one step at a time, which is called a *receding horizon*. Every plan starts from a real state, so model error cannot build up from one real step to the next. The first action depends most on the first few predicted steps, which are the accurate ones. Replanning turns the open-loop model into a closed-loop controller, and feedback corrects what the model gets wrong. With a 15-step horizon, 64 samples, 8 elites, and 4 CEM iterations, MPC with the fitted linear model balances CartPole for all 200 steps. Across six seeds the pole never leans past 0.05 rad, well inside the 0.21 rad failure line.

> [!KEY]
> MPC executes only the first action of each plan and replans from the real state. Model error then matters over the few steps that decide the first action, not over the whole horizon. That answers the opening question: you can act well in the real world after learning a model that is only right for a short way ahead.

> [!RECAP]
> - Ridge regression on rows $[s, a, 1]$ fits $W = (X^\top X + \lambda I)^{-1} X^\top Y$.
> - An open-loop rollout feeds each prediction back in, so the $k$-step error carries every earlier error forward.
> - For $\hat a = 1.01$ against $a = 1$, the $k$-step error is exactly $1.01^k - 1$.
> - CEM refits a Gaussian to the top $n_e$ samples, using the population standard deviation.
> - MPC plans from the real state, executes the first action, and replans.

## Build it step by step

The checks test these functions in this order.

1. `fit_linear_model(states, actions, next_states, ridge)` takes `(N, d)`, `(N, k)`, and `(N, d)` arrays and returns `W` of shape `(d + k + 1, d)`. Build `X` with `np.concatenate` and a column of ones, then call `np.linalg.solve`. The common mistake is putting the bias column first. The checks expect $[A^\top; B^\top; c]$ from top to bottom.
2. `model_step(W, states, actions)` builds the same `X` and returns `X @ W`. Reshape an `(N,)` action array to `(N, 1)` first.
3. `rollout(step_fn, s0, actions)` returns `(N, H + 1, d)` with `s0` first. The common mistake is calling `step_fn` on `s0` at every step instead of on the last output.
4. `k_step_errors(true_step, learned_step, s0, actions)` rolls out both functions, drops entry 0, takes `np.linalg.norm` over the last axis, and averages over the batch, giving shape `(H,)`. The common mistake is averaging the squared error.
5. `cem_update(samples, scores, n_elite)` sorts with `np.argsort(-scores)`, keeps the top rows, and returns their mean and `std(axis=0)`. The common mistake is keeping the lowest scores, because higher is better here.
6. `cem_optimize(score_fn, mean, std, n_samples, n_elite, n_iters, rng)` loops: draw `mean + std * rng.standard_normal((n_samples, D))`, score, and refit. Return the final mean.
7. `mpc_action(step_fn, reward_fn, state, horizon, rng, ...)` writes a score function that turns plans `(n, H)` into actions `(n, H, 1)` with `plans > 0`, rolls out from the real state, and sums `reward_fn(traj[:, 1:])` over time. Run `cem_optimize` from mean 0 and std 1, and return `int(plan[0] > 0)`. The common mistake is returning the whole plan instead of one int.

## Where this breaks

- **Model exploitation.** The planner searches for plans the model rates highly, so it finds places where the model is wrong in its favor. The plan looks great in imagination and fails in reality. Ensembles help.
- **Distribution shift.** A model is accurate only where it was trained, so a controller that wanders into unfamiliar states needs the model refitted on the data it collects there.
- **Objective mismatch.** The model is trained to predict the next state, but the agent cares about return. A better one-step model does not always give better control (Lambert et al., 2020). The next chapter covers models that predict only what planning needs.

## Further reading

The following papers cover learned models, CEM, and model-based control:

- [de Boer, Kroese, Mannor, and Rubinstein (2005), A tutorial on the cross-entropy method](https://doi.org/10.1007/s10479-005-5724-z) explains why refitting to elites works as optimization.
- [Chua, Calandra, McAllister, and Levine (2018), Deep reinforcement learning in a handful of trials using probabilistic dynamics models](https://arxiv.org/abs/1805.12114) combines probabilistic ensembles with CEM and MPC.
- [Janner, Fu, Zhang, and Levine (2019), When to trust your model: model-based policy optimization](https://arxiv.org/abs/1906.08253) bounds the return gap by rollout length and uses short branched rollouts.
- [Lambert, Amos, Yadan, and Calandra (2020), Objective mismatch in model-based reinforcement learning](https://arxiv.org/abs/2002.04523) shows that one-step accuracy and control performance can disagree.

## The exercise

You build a model-based controller for CartPole from the ground up. Write `fit_linear_model` and `model_step` for the linear model, `rollout` and `k_step_errors` to measure drift, `cem_update` and `cem_optimize` for the cross-entropy method, and `mpc_action` to plan through a model and return the first action.

The checks verify that `fit_linear_model` recovers a known affine system exactly and zeroes the ridge objective's gradient. They check `rollout` against the closed form, `k_step_errors` against $1.01^k - 1$, and that the learned model's 30-step CartPole error exceeds 100 times its one-step error. They check one CEM refit against the worked example and `cem_optimize` against a quadratic's maximizer, and run `mpc_action` for a full 200-step episode with a linear model fitted to 2,000 transitions, keeping the pole under 0.15 rad.
