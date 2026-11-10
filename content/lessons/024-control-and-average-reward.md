---
title: Control, better features, and average reward
slug: 024-control-and-average-reward
part: "Part 4 — Deep value methods"
summary: Add Fourier and RBF features and n-step SARSA for control, then learn on a task that never ends by tracking the average reward with differential TD.
minutes: 40
gpu: false
objectives:
  - Build Fourier and radial basis features, and say what each generalizes across.
  - Compute an n-step SARSA return, including a window cut short by the end of an episode.
  - Train an n-step semi-gradient SARSA controller on Mountain Car and say why it learns faster than one-step SARSA.
  - Define the average reward of a policy and the differential TD error, and apply one differential update by hand.
  - Say why a discount factor stops being a useful objective for a continuing task with function approximation.
prereqs:
  - 019-n-step-returns
  - 023-function-approximation
exercise: 024-control-and-average-reward
---

# Control, better features, and average reward

[Value-function approximation](/lessons/023-function-approximation) trained a Mountain Car controller with tile features and one-step SARSA. That worked, but tiles are a blocky choice of feature, and one-step SARSA passes news about the goal back one state per visit. Mountain Car also ends, so a return always has a finish line to count toward.

Many tasks have no finish line. A thermostat, a server deciding which requests to admit, or a factory line never reaches a terminal state. This chapter improves the control machinery first, with smoother features and a longer look-ahead. Then it changes the question from "how much return until the end" to "how much reward per step, forever", which is the right question for a continuing task.

The running example for the second half is a factory line of four stations in a ring. A part lingers at a station for a random time and then moves on. Only station 3 pays, 4 per step.

## What you already know

- A linear estimate is $\hat q_w(s, a) = w^\top x(s, a)$, and semi-gradient SARSA moves $w$ by $\alpha \, \delta \, x$, as in [Value-function approximation](/lessons/023-function-approximation).
- An $n$-step return sums $n$ real rewards and then bootstraps, as in [n-step returns](/lessons/019-n-step-returns).
- The stationary distribution $\mu$ is the long-run fraction of time a policy spends in each state.

## Smoother features: Fourier and radial basis

A tile feature is 1 inside a tile and 0 outside, so the learned value is a staircase. That is fine for a rough shape but wasteful for a smooth one. The *Fourier basis* swaps tiles for cosines (Konidaris, Osentoski, and Thomas, 2011). Scale each input to $[0, 1]$. For each integer vector $c$ with entries in $0, 1, \ldots, \text{order}$, one feature is

$$
x_c(s) = \cos\big(\pi \, c^\top s\big)
$$

Read it aloud: $c$ says how many half-waves of cosine to fit along each input, and the feature is the height of that wave at $s$. The vector $c = 0$ gives the constant 1, and $c = 1$ gives one half-wave across the whole range. With $d$ inputs there are $(\text{order} + 1)^d$ features. In numpy:

```python
C = np.array(list(itertools.product(range(order + 1), repeat=d)), dtype=float)
features = np.cos(np.pi * (C @ x))
```

Take one input at $x = 0.25$ and order 2. The three features are $\cos 0 = 1$, $\cos(\pi/4) = 0.7071$, and $\cos(\pi/2) = 0$. With two inputs at $(0.25, 0.5)$ and order 1, the coefficient vectors are $(0,0), (0,1), (1,0), (1,1)$, and the features are $1$, $0$, $0.7071$, and $-0.7071$.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Four Fourier basis functions on the interval from 0 to 1: the constant 1 and cosines with 1, 2, and 3 half-waves">
<line x1="70.0" y1="210.0" x2="570.0" y2="210.0" class="stroke-grid"/>
<text x="62.0" y="214.0" class="fig-small" text-anchor="end">-1</text>
<line x1="70.0" y1="125.0" x2="570.0" y2="125.0" class="stroke-grid"/>
<text x="62.0" y="129.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="40.0" x2="570.0" y2="40.0" class="stroke-grid"/>
<text x="62.0" y="44.0" class="fig-small" text-anchor="end">1</text>
<text x="70.0" y="228.0" class="fig-small" text-anchor="middle">0</text>
<text x="195.0" y="228.0" class="fig-small" text-anchor="middle">0.25</text>
<text x="320.0" y="228.0" class="fig-small" text-anchor="middle">0.5</text>
<text x="445.0" y="228.0" class="fig-small" text-anchor="middle">0.75</text>
<text x="570.0" y="228.0" class="fig-small" text-anchor="middle">1</text>
<text x="320.0" y="258.0" text-anchor="middle">x</text>
<polyline points="70.0,40.0 75.0,40.0 80.0,40.0 85.0,40.0 90.0,40.0 95.0,40.0 100.0,40.0 105.0,40.0 110.0,40.0 115.0,40.0 120.0,40.0 125.0,40.0 130.0,40.0 135.0,40.0 140.0,40.0 145.0,40.0 150.0,40.0 155.0,40.0 160.0,40.0 165.0,40.0 170.0,40.0 175.0,40.0 180.0,40.0 185.0,40.0 190.0,40.0 195.0,40.0 200.0,40.0 205.0,40.0 210.0,40.0 215.0,40.0 220.0,40.0 225.0,40.0 230.0,40.0 235.0,40.0 240.0,40.0 245.0,40.0 250.0,40.0 255.0,40.0 260.0,40.0 265.0,40.0 270.0,40.0 275.0,40.0 280.0,40.0 285.0,40.0 290.0,40.0 295.0,40.0 300.0,40.0 305.0,40.0 310.0,40.0 315.0,40.0 320.0,40.0 325.0,40.0 330.0,40.0 335.0,40.0 340.0,40.0 345.0,40.0 350.0,40.0 355.0,40.0 360.0,40.0 365.0,40.0 370.0,40.0 375.0,40.0 380.0,40.0 385.0,40.0 390.0,40.0 395.0,40.0 400.0,40.0 405.0,40.0 410.0,40.0 415.0,40.0 420.0,40.0 425.0,40.0 430.0,40.0 435.0,40.0 440.0,40.0 445.0,40.0 450.0,40.0 455.0,40.0 460.0,40.0 465.0,40.0 470.0,40.0 475.0,40.0 480.0,40.0 485.0,40.0 490.0,40.0 495.0,40.0 500.0,40.0 505.0,40.0 510.0,40.0 515.0,40.0 520.0,40.0 525.0,40.0 530.0,40.0 535.0,40.0 540.0,40.0 545.0,40.0 550.0,40.0 555.0,40.0 560.0,40.0 565.0,40.0 570.0,40.0" class="stroke-muted dashed" fill="none"/>
<polyline points="70.0,40.0 75.0,40.0 80.0,40.2 85.0,40.4 90.0,40.7 95.0,41.0 100.0,41.5 105.0,42.0 110.0,42.7 115.0,43.4 120.0,44.2 125.0,45.0 130.0,46.0 135.0,47.0 140.0,48.1 145.0,49.3 150.0,50.5 155.0,51.8 160.0,53.2 165.0,54.7 170.0,56.2 175.0,57.8 180.0,59.5 185.0,61.2 190.0,63.0 195.0,64.9 200.0,66.8 205.0,68.8 210.0,70.8 215.0,72.9 220.0,75.0 225.0,77.2 230.0,79.5 235.0,81.7 240.0,84.1 245.0,86.4 250.0,88.8 255.0,91.2 260.0,93.7 265.0,96.2 270.0,98.7 275.0,101.3 280.0,103.9 285.0,106.5 290.0,109.1 295.0,111.7 300.0,114.3 305.0,117.0 310.0,119.7 315.0,122.3 320.0,125.0 325.0,127.7 330.0,130.3 335.0,133.0 340.0,135.7 345.0,138.3 350.0,140.9 355.0,143.5 360.0,146.1 365.0,148.7 370.0,151.3 375.0,153.8 380.0,156.3 385.0,158.8 390.0,161.2 395.0,163.6 400.0,165.9 405.0,168.3 410.0,170.5 415.0,172.8 420.0,175.0 425.0,177.1 430.0,179.2 435.0,181.2 440.0,183.2 445.0,185.1 450.0,187.0 455.0,188.8 460.0,190.5 465.0,192.2 470.0,193.8 475.0,195.3 480.0,196.8 485.0,198.2 490.0,199.5 495.0,200.7 500.0,201.9 505.0,203.0 510.0,204.0 515.0,205.0 520.0,205.8 525.0,206.6 530.0,207.3 535.0,208.0 540.0,208.5 545.0,209.0 550.0,209.3 555.0,209.6 560.0,209.8 565.0,210.0 570.0,210.0" class="stroke-a" fill="none"/>
<polyline points="70.0,40.0 75.0,40.2 80.0,40.7 85.0,41.5 90.0,42.7 95.0,44.2 100.0,46.0 105.0,48.1 110.0,50.5 115.0,53.2 120.0,56.2 125.0,59.5 130.0,63.0 135.0,66.8 140.0,70.8 145.0,75.0 150.0,79.5 155.0,84.1 160.0,88.8 165.0,93.7 170.0,98.7 175.0,103.9 180.0,109.1 185.0,114.3 190.0,119.7 195.0,125.0 200.0,130.3 205.0,135.7 210.0,140.9 215.0,146.1 220.0,151.3 225.0,156.3 230.0,161.2 235.0,165.9 240.0,170.5 245.0,175.0 250.0,179.2 255.0,183.2 260.0,187.0 265.0,190.5 270.0,193.8 275.0,196.8 280.0,199.5 285.0,201.9 290.0,204.0 295.0,205.8 300.0,207.3 305.0,208.5 310.0,209.3 315.0,209.8 320.0,210.0 325.0,209.8 330.0,209.3 335.0,208.5 340.0,207.3 345.0,205.8 350.0,204.0 355.0,201.9 360.0,199.5 365.0,196.8 370.0,193.8 375.0,190.5 380.0,187.0 385.0,183.2 390.0,179.2 395.0,175.0 400.0,170.5 405.0,165.9 410.0,161.2 415.0,156.3 420.0,151.3 425.0,146.1 430.0,140.9 435.0,135.7 440.0,130.3 445.0,125.0 450.0,119.7 455.0,114.3 460.0,109.1 465.0,103.9 470.0,98.7 475.0,93.7 480.0,88.8 485.0,84.1 490.0,79.5 495.0,75.0 500.0,70.8 505.0,66.8 510.0,63.0 515.0,59.5 520.0,56.2 525.0,53.2 530.0,50.5 535.0,48.1 540.0,46.0 545.0,44.2 550.0,42.7 555.0,41.5 560.0,40.7 565.0,40.2 570.0,40.0" class="stroke-b" fill="none"/>
<polyline points="70.0,40.0 75.0,40.4 80.0,41.5 85.0,43.4 90.0,46.0 95.0,49.3 100.0,53.2 105.0,57.8 110.0,63.0 115.0,68.8 120.0,75.0 125.0,81.7 130.0,88.8 135.0,96.2 140.0,103.9 145.0,111.7 150.0,119.7 155.0,127.7 160.0,135.7 165.0,143.5 170.0,151.3 175.0,158.8 180.0,165.9 185.0,172.8 190.0,179.2 195.0,185.1 200.0,190.5 205.0,195.3 210.0,199.5 215.0,203.0 220.0,205.8 225.0,208.0 230.0,209.3 235.0,210.0 240.0,209.8 245.0,209.0 250.0,207.3 255.0,205.0 260.0,201.9 265.0,198.2 270.0,193.8 275.0,188.8 280.0,183.2 285.0,177.1 290.0,170.5 295.0,163.6 300.0,156.3 305.0,148.7 310.0,140.9 315.0,133.0 320.0,125.0 325.0,117.0 330.0,109.1 335.0,101.3 340.0,93.7 345.0,86.4 350.0,79.5 355.0,72.9 360.0,66.8 365.0,61.2 370.0,56.2 375.0,51.8 380.0,48.1 385.0,45.0 390.0,42.7 395.0,41.0 400.0,40.2 405.0,40.0 410.0,40.7 415.0,42.0 420.0,44.2 425.0,47.0 430.0,50.5 435.0,54.7 440.0,59.5 445.0,64.9 450.0,70.8 455.0,77.2 460.0,84.1 465.0,91.2 470.0,98.7 475.0,106.5 480.0,114.3 485.0,122.3 490.0,130.3 495.0,138.3 500.0,146.1 505.0,153.8 510.0,161.2 515.0,168.3 520.0,175.0 525.0,181.2 530.0,187.0 535.0,192.2 540.0,196.8 545.0,200.7 550.0,204.0 555.0,206.6 560.0,208.5 565.0,209.6 570.0,210.0" class="stroke-c" fill="none"/>
<line x1="150.0" y1="16.0" x2="174.0" y2="16.0" class="stroke-muted dashed"/>
<text x="180.0" y="20.0" class="fig-small" text-anchor="start">c = 0</text>
<line x1="250.0" y1="16.0" x2="274.0" y2="16.0" class="stroke-a"/>
<text x="280.0" y="20.0" class="fig-small" text-anchor="start">c = 1</text>
<line x1="350.0" y1="16.0" x2="374.0" y2="16.0" class="stroke-b"/>
<text x="380.0" y="20.0" class="fig-small" text-anchor="start">c = 2</text>
<line x1="450.0" y1="16.0" x2="474.0" y2="16.0" class="stroke-c"/>
<text x="480.0" y="20.0" class="fig-small" text-anchor="start">c = 3</text>
</svg>
Figure 1. The Fourier basis functions for c = 0, 1, 2, and 3 on [0, 1]. Notice that each one is smooth and spans the whole range, so one update changes the estimate everywhere, and higher c adds finer detail.
```

A *radial basis function* (RBF) is the local alternative. Pick centers $c_i$ and a width $\sigma$, and let feature $i$ be

$$
x_i(s) = \exp\!\Big(-\frac{\lVert s - c_i \rVert^2}{2\sigma^2}\Big)
$$

Read it aloud: the feature is 1 at its center and falls off smoothly with distance, like a tile with soft edges. With centers at 0, 0.5, and 1, $\sigma = 0.5$, and $s = 0.5$, the features are $0.6065$, $1$, and $0.6065$.

Fourier features are global and need no placement decisions, but they ring near sharp edges. RBFs are local, but you place the centers and pick $\sigma$. Both grow exponentially with the number of inputs on a full grid, so they suit low-dimensional inputs, as tiles do.

> [!KEY]
> Features decide what generalizes to what. Tiles are blocky and local, RBFs are smooth and local, and the Fourier basis is smooth and global.

## A longer look-ahead: n-step semi-gradient SARSA

One-step SARSA moves the estimate at $(s_t, a_t)$ toward a target that uses one real reward and a guess for everything after. In Mountain Car every reward is $-1$, so the only true information is "the episode has not ended yet". The flag's value reaches the start of the climb one visit at a time, and each visit costs an episode. Using $n$ real rewards before guessing moves that information $n$ steps per update.

Define the $n$-step return from step $\tau$ as

$$
G_{\tau:\tau+n} = r_\tau + \gamma r_{\tau+1} + \cdots + \gamma^{n-1} r_{\tau+n-1} + \gamma^{n}\, \hat q_w(s_{\tau+n}, a_{\tau+n})
$$

Read it aloud: add up $n$ discounted rewards, then bootstrap from the value of the state and action $n$ steps later. If the episode ends before step $\tau + n$, there is no state to bootstrap from, so you drop the last term and sum only the rewards that happened. The weight update is the usual semi-gradient one, with $G_{\tau:\tau+n}$ as the target (Sutton and Barto, Section 10.2):

$$
w \leftarrow w + \alpha \big(G_{\tau:\tau+n} - \hat q_w(s_\tau, a_\tau)\big)\, x(s_\tau, a_\tau)
$$

At step $t$ the algorithm updates $\tau = t - n + 1$, because only then has it seen $n$ rewards after $\tau$. After the episode ends, it keeps updating the last $n - 1$ steps without a bootstrap.

Here is a worked example with $n = 3$, $\gamma = 0.9$, three rewards of $-1$, and $\hat q_w(s_{\tau+3}, a_{\tau+3}) = -10$:

- Rewards: $-1 - 0.9 - 0.81 = -2.71$.
- Bootstrap: $0.9^3 \times (-10) = -7.29$.
- Return: $G = -10$. If $\hat q_w(s_\tau, a_\tau) = -8$, the error is $-2$.
- If the episode had ended after two rewards, the return would be $-1.9$, with no bootstrap.

```python
def n_step_return(rewards, q_boot, gamma, terminated):
    G = sum(gamma**i * r for i, r in enumerate(rewards))
    return G if terminated else G + gamma ** len(rewards) * q_boot
```

On Mountain Car with the same tile features and step size as before, $n = 8$ reaches the flag faster than $n = 1$ on every one of 5 seeds. Averaged over 30 episodes, one-step SARSA takes about 342 steps per episode and 8-step SARSA about 279. By episode 30 the means are 218 and 143.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Mean steps per episode on Mountain Car over 5 seeds for one-step and 8-step semi-gradient SARSA; the 8-step curve sits below the one-step curve">
<line x1="70.0" y1="220.0" x2="570.0" y2="220.0" class="stroke-grid"/>
<text x="62.0" y="224.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="130.0" x2="570.0" y2="130.0" class="stroke-grid"/>
<text x="62.0" y="134.0" class="fig-small" text-anchor="end">500</text>
<line x1="70.0" y1="40.0" x2="570.0" y2="40.0" class="stroke-grid"/>
<text x="62.0" y="44.0" class="fig-small" text-anchor="end">1000</text>
<text x="70.0" y="238.0" class="fig-small" text-anchor="middle">1</text>
<text x="225.2" y="238.0" class="fig-small" text-anchor="middle">10</text>
<text x="397.6" y="238.0" class="fig-small" text-anchor="middle">20</text>
<text x="570.0" y="238.0" class="fig-small" text-anchor="middle">30</text>
<text x="62.0" y="22.0" class="fig-small" text-anchor="start">steps to reach the flag (cap 1000)</text>
<text x="320.0" y="262.0" text-anchor="middle">episode</text>
<polyline points="70.0,40.0 87.2,82.7 104.5,84.6 121.7,133.1 139.0,119.4 156.2,135.4 173.4,145.7 190.7,160.5 207.9,164.1 225.2,157.6 242.4,157.8 259.7,174.3 276.9,163.8 294.1,164.3 311.4,176.7 328.6,171.9 345.9,176.5 363.1,173.8 380.3,181.0 397.6,178.5 414.8,183.7 432.1,181.7 449.3,176.4 466.6,180.8 483.8,179.5 501.0,180.4 518.3,184.7 535.5,182.4 552.8,183.7 570.0,180.8" class="stroke-a" fill="none"/>
<polyline points="70.0,40.0 87.2,98.6 104.5,90.7 121.7,114.9 139.0,150.3 156.2,125.3 173.4,156.7 190.7,175.9 207.9,178.3 225.2,174.7 242.4,172.4 259.7,194.9 276.9,181.8 294.1,187.1 311.4,182.1 328.6,182.9 345.9,187.0 363.1,186.5 380.3,187.1 397.6,194.0 414.8,185.7 432.1,193.6 449.3,192.2 466.6,191.8 483.8,190.4 501.0,195.0 518.3,195.7 535.5,199.4 552.8,195.5 570.0,194.2" class="stroke-b" fill="none"/>
<line x1="400.0" y1="52.0" x2="430.0" y2="52.0" class="stroke-a"/>
<text x="438.0" y="56.0" class="fig-small" text-anchor="start">n = 1</text>
<line x1="400.0" y1="70.0" x2="430.0" y2="70.0" class="stroke-b"/>
<text x="438.0" y="74.0" class="fig-small" text-anchor="start">n = 8</text>
</svg>
Figure 2. Mean steps per episode on Mountain Car over 5 seeds, for n = 1 and n = 8 with the same tile features and step size. Notice that both start at the 1,000-step cap, and the 8-step curve drops faster and stays lower.
```

A middle $n$ works best, as in the tabular case of [n-step returns](/lessons/019-n-step-returns).

> [!KEY]
> An $n$-step target moves news about the goal $n$ steps per update. Near the end of an episode the window is shorter than $n$ and has no bootstrap term.

## A task with no end: average reward

Remove the finish line. A continuing task has no terminal state, so there is no episode to total up. The factory line earns forever, and its total reward is infinite whatever you do. A useful objective compares policies by reward per step. The *average reward* of a policy $\pi$ is

$$
r(\pi) = \sum_{s} \mu_\pi(s) \sum_{a} \pi(a \mid s)\, r(s, a)
$$

Read it aloud: weight the expected reward of each state by the fraction of time the policy spends there, and add. For a chain with transition matrix `P` and a reward `r[s]` for each state, it is `stationary_distribution(P) @ r`.

On the factory ring, station 0 keeps a part with probability 0.5, and stations 1 to 3 keep it with probability 0.2. A station's share of time is proportional to the expected stay, $1/(1 - \text{stay})$, which is $2, 1.25, 1.25, 1.25$. So $\mu = (0.3478, 0.2174, 0.2174, 0.2174)$, and only station 3 pays 4:

$$
r(\pi) = 4 \times 0.2174 = 0.8696
$$

A single average cannot say which state is better, so each state gets a *differential value* $v(s)$, the expected sum of $r_k - r(\pi)$ from $s$. It measures how much more than average you collect starting here. Differential values obey a Bellman equation with no discount, $v(s) = r(s) - r(\pi) + \sum_{s'} P(s' \mid s)\, v(s')$, and they are fixed only up to adding one constant to every state.

You rarely know $r(\pi)$, so you learn an estimate $\bar r$ alongside the values. The *differential TD error* is

$$
\delta_t = r_t - \bar r + \hat v_w(s_{t+1}) - \hat v_w(s_t)
$$

Read it aloud: the usual TD error with no discount, and with the average reward subtracted from the reward. Then both the weights and the estimate move along $\delta_t$ (Sutton and Barto, Section 10.3):

$$
w \leftarrow w + \alpha\, \delta_t\, x(s_t), \qquad \bar r \leftarrow \bar r + \beta\, \delta_t
$$

The second update looks odd, because it does not move $\bar r$ toward $r_t$. It works because averaged over the stationary distribution, the value terms cancel: $\sum_s \mu(s)\,(v(s') - v(s))$ is zero. So the average error is $r(\pi) - \bar r$, and the update pushes $\bar r$ toward $r(\pi)$ exactly. There is no terminal mask, because the task has no end. In numpy:

```python
delta = r - r_bar + w @ x2 - w @ x
w = w + alpha * delta * x
r_bar = r_bar + beta * delta
```

Here is a worked example. Take $w = [1, 2, 0]$, $x(s_t) = [1, 0, 1]$, $x(s_{t+1}) = [0, 1, 1]$, $r_t = 3$, $\bar r = 1.5$, $\alpha = 0.1$, and $\beta = 0.02$.

- Estimates: $\hat v(s_t) = 1$ and $\hat v(s_{t+1}) = 2$.
- Error: $\delta = 3 - 1.5 + 2 - 1 = 2.5$.
- Update: $w = [1, 2, 0] + 0.25 \times [1, 0, 1] = [1.25, 2, 0.25]$ and $\bar r = 1.5 + 0.02 \times 2.5 = 1.55$.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="The learned average reward on a four-state ring rises from 0 toward the exact value 0.87 and then wobbles around it">
<line x1="70.0" y1="220.0" x2="570.0" y2="220.0" class="stroke-grid"/>
<text x="62.0" y="224.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="160.0" x2="570.0" y2="160.0" class="stroke-grid"/>
<text x="62.0" y="164.0" class="fig-small" text-anchor="end">0.4</text>
<line x1="70.0" y1="100.0" x2="570.0" y2="100.0" class="stroke-grid"/>
<text x="62.0" y="104.0" class="fig-small" text-anchor="end">0.8</text>
<line x1="70.0" y1="40.0" x2="570.0" y2="40.0" class="stroke-grid"/>
<text x="62.0" y="44.0" class="fig-small" text-anchor="end">1.2</text>
<text x="70.0" y="238.0" class="fig-small" text-anchor="middle">0</text>
<text x="195.0" y="238.0" class="fig-small" text-anchor="middle">5000</text>
<text x="320.0" y="238.0" class="fig-small" text-anchor="middle">10000</text>
<text x="445.0" y="238.0" class="fig-small" text-anchor="middle">15000</text>
<text x="570.0" y="238.0" class="fig-small" text-anchor="middle">20000</text>
<text x="62.0" y="22.0" class="fig-small" text-anchor="start">estimate r_bar</text>
<text x="320.0" y="262.0" text-anchor="middle">step</text>
<line x1="70.0" y1="89.6" x2="570.0" y2="89.6" class="stroke-c dashed"/>
<text x="570.0" y="83.6" class="fig-small" text-anchor="end">exact 0.8696</text>
<polyline points="70.0,220.0 72.5,194.3 75.0,170.5 77.5,154.2 80.0,143.3 82.5,135.3 85.0,128.5 87.5,124.6 90.0,119.4 92.5,111.4 95.0,107.4 97.5,104.7 100.0,101.5 102.5,99.9 105.0,95.9 107.5,91.5 110.0,92.1 112.5,90.2 115.0,89.9 117.5,96.2 120.0,97.8 122.5,96.1 125.0,94.5 127.5,96.3 130.0,92.6 132.5,84.3 135.0,86.7 137.5,87.7 140.0,88.3 142.5,85.0 145.0,87.4 147.5,85.9 150.0,85.7 152.5,85.1 155.0,87.4 157.5,86.1 160.0,88.1 162.5,95.4 165.0,93.2 167.5,91.1 170.0,88.5 172.5,87.4 175.0,85.1 177.5,81.3 180.0,77.5 182.5,77.7 185.0,79.5 187.5,84.6 190.0,84.9 192.5,84.1 195.0,84.2 197.5,85.3 200.0,87.6 202.5,83.9 205.0,83.2 207.5,83.6 210.0,86.5 212.5,87.1 215.0,87.0 217.5,89.1 220.0,90.6 222.5,96.3 225.0,92.6 227.5,92.9 230.0,95.2 232.5,96.5 235.0,99.1 237.5,90.8 240.0,85.4 242.5,86.8 245.0,85.9 247.5,86.1 250.0,84.5 252.5,85.5 255.0,84.6 257.5,87.1 260.0,85.7 262.5,86.1 265.0,85.1 267.5,83.2 270.0,82.6 272.5,87.1 275.0,88.2 277.5,82.4 280.0,86.1 282.5,84.6 285.0,85.8 287.5,83.9 290.0,88.3 292.5,83.2 295.0,84.6 297.5,78.2 300.0,77.3 302.5,78.7 305.0,83.9 307.5,85.6 310.0,83.0 312.5,85.9 315.0,88.2 317.5,91.8 320.0,89.0 322.5,89.1 325.0,94.3 327.5,96.0 330.0,94.4 332.5,92.8 335.0,91.4 337.5,91.1 340.0,90.5 342.5,94.3 345.0,94.1 347.5,92.8 350.0,92.1 352.5,91.2 355.0,90.2 357.5,88.0 360.0,83.9 362.5,85.0 365.0,85.3 367.5,87.8 370.0,87.7 372.5,87.2 375.0,87.2 377.5,85.7 380.0,87.3 382.5,85.3 385.0,82.3 387.5,83.3 390.0,85.2 392.5,88.6 395.0,89.6 397.5,89.7 400.0,87.8 402.5,90.3 405.0,87.5 407.5,89.7 410.0,91.9 412.5,95.5 415.0,95.0 417.5,91.7 420.0,90.1 422.5,86.1 425.0,87.4 427.5,85.5 430.0,87.8 432.5,85.7 435.0,86.0 437.5,85.3 440.0,86.8 442.5,89.9 445.0,90.9 447.5,90.1 450.0,91.2 452.5,83.9 455.0,83.2 457.5,85.7 460.0,90.8 462.5,93.5 465.0,91.5 467.5,89.9 470.0,91.6 472.5,88.9 475.0,93.8 477.5,93.9 480.0,93.7 482.5,85.1 485.0,87.3 487.5,88.8 490.0,87.0 492.5,86.6 495.0,89.1 497.5,90.8 500.0,92.4 502.5,89.8 505.0,90.9 507.5,91.2 510.0,91.5 512.5,89.8 515.0,89.2 517.5,90.2 520.0,87.2 522.5,89.3 525.0,88.6 527.5,94.0 530.0,87.4 532.5,87.8 535.0,86.6 537.5,86.7 540.0,88.1 542.5,88.8 545.0,86.9 547.5,83.5 550.0,85.2 552.5,86.2 555.0,83.8 557.5,86.3 560.0,82.1 562.5,86.7 565.0,88.5 567.5,89.8 570.0,94.2" class="stroke-b" fill="none"/>
</svg>
Figure 3. The estimate of the average reward during one run of differential TD on the factory ring, with one-hot features, alpha = 0.05, and beta = 0.002. Notice that it climbs from 0 to the exact 0.8696 and then wobbles around it, because the transitions are random.
```

On a deterministic cycle of four stations with reward 4 at station 3, the same update converges to the exact answer. The average reward is 1, and the differential values minus the value of station 0 are $0, 1, 2, 3$.

Control works the same way. Differential semi-gradient SARSA replaces $\hat v$ with $\hat q_w(s, a)$ at the next action the agent takes, and updates $\bar r$ with $\beta \delta_t$. Section 10.5 of Sutton and Barto gives the $n$-step version.

> [!KEY]
> In a continuing task, compare policies by reward per step. Differential TD measures each reward against a running average $\bar r$, and the same error that fixes the values also fixes $\bar r$.

## Why discounting stops working here

You could keep a discount factor and ask for the discounted return in a continuing task. For a tabular task that is fine. With function approximation it is not, and Sutton and Barto devote Section 10.4 to why.

The first fact is that the discount buys you nothing as an objective. On the factory ring, solve for the discounted values $v_\gamma$ and average them with weights $\mu$. For $\gamma = 0.9$ the result is $8.6957$, and for $\gamma = 0.99$ it is $86.9565$. Both equal $r(\pi)/(1 - \gamma)$. The discounted objective is the average reward multiplied by a constant that depends only on $\gamma$, so it ranks policies in the same order as the average reward, and $\gamma$ changes only the scale.

The second fact is the cost. With function approximation, states are no longer improved one at a time, so the policy improvement guarantee that holds for tables is lost. Sutton and Barto conclude that continuing tasks with function approximation call for the average-reward setting. Treat that as their argument, not settled law, because discounted methods with a $\gamma$ chosen for stability are still widely used.

## Where the triad comes back

None of this removes the instability of [the deadly triad](/lessons/025-the-deadly-triad): bootstrapping, function approximation, and off-policy data together can diverge. *Gradient-TD* methods follow the true gradient of the projected Bellman error, at the cost of a second set of weights (Sutton and Barto, Section 11.7). *Emphatic-TD* methods reweight each update by how much the state matters under the target policy, which restores convergence for linear off-policy TD (Section 11.8). This chapter builds neither.

## Build it step by step

The checks test the functions in this order. `stationary_distribution`, `MountainCar`, and `state_action_features` are already written.

1. **`fourier_features(x, order)`.** Returns a float vector of length `(order + 1) ** d`. Stack every coefficient vector from `itertools.product` as the rows of `C`, and return `np.cos(np.pi * C @ x)`. Common mistake: using `2 * np.pi`, which doubles the frequency.
2. **`rbf_features(x, centers, sigma)`.** One squared distance per center, `((centers - x) ** 2).sum(axis=1)`, then `np.exp(-sq / (2 * sigma ** 2))`. Common mistake: dropping the 2 in the denominator.
3. **`n_step_return(rewards, q_boot, gamma, terminated)`.** The discounted sum of the rewards you have, plus the bootstrap term only when the episode has not ended. Common mistake: bootstrapping from `q_boot` on a terminal window.
4. **`n_step_sarsa(n, n_episodes, alpha, gamma, epsilon, max_steps, seed)`.** Keep lists of states, actions, and rewards. At step `t`, update `tau = t - n + 1` when it is at least 0, bootstrapping from `(states[tau + n], actions[tau + n])` only if `tau + n < T`. After the terminal step, keep looping until `tau` reaches `T - 1`. Common mistake: skipping the final updates, so the last states of every episode never learn.
5. **`average_reward(P, r)`.** One dot product, `stationary_distribution(P) @ r`.
6. **`differential_td_update(w, r_bar, x, r, x2, alpha, beta)`.** Returns the updated `(w, r_bar)` and leaves the inputs alone. Common mistake: updating `r_bar` by `beta * (r - r_bar)`. That tracks the mean reward but ignores the value terms, so the exact differential values stop being a fixed point.
7. **`differential_td(P, r, Phi, n_steps, alpha, beta, seed)`.** Start at state 0, sample `s2 = rng.choice(S, p=P[s])`, apply the update with `Phi[s]`, `Phi[s2]`, and `r[s]`, and move on.

## Where this breaks

**Step sizes.** High-order Fourier features change fast, so one step size is too large for them. Konidaris and colleagues scale the step for feature $c$ by $1/\lVert c \rVert$.

**Two step sizes.** If $\beta$ is large, $\bar r$ chases noise. If it is tiny, the values learn against a wrong average for a long time. Compare differential values only as differences between states, because they are fixed up to a constant.

## Further reading

- [Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd edition (2018), Chapters 9, 10, and 11](http://incompleteideas.net/book/the-book-2nd.html). Section 9.5.3 covers the Fourier basis and Section 9.5.6 radial basis functions. In Chapter 10, Section 10.2 covers semi-gradient $n$-step SARSA, Section 10.3 the average-reward setting, Section 10.4 the case against discounting, and Section 10.5 differential semi-gradient $n$-step SARSA. Sections 11.7 and 11.8 introduce gradient-TD and emphatic-TD.
- [David Silver, *Reinforcement Learning* (COMPM050/COMPGI13), Lecture 6, "Value Function Approximation"](https://davidstarsilver.wordpress.com/teaching/). Slides on linear approximation and control with function approximation.
- [Konidaris, Osentoski, and Thomas, "Value function approximation in reinforcement learning using the Fourier basis" (AAAI 2011)](https://doi.org/10.1609/aaai.v25i1.7903). Introduces the Fourier basis for value functions and compares it with RBFs and polynomials.
- [Mahadevan, "Average reward reinforcement learning: Foundations, algorithms, and empirical results" (1996)](https://doi.org/10.1007/BF00114725). A survey of the average-reward objective and the methods built on it.
- [Sutton, Mahmood, and White, "An emphatic approach to the problem of off-policy temporal-difference learning" (2016)](https://jmlr.org/papers/v17/14-488.html). The emphatic-TD method and its convergence proof.

## The exercise

You build the features, the $n$-step controller, and the average-reward learner. Write `fourier_features`, `rbf_features`, `n_step_return`, and `n_step_sarsa`. Then write `average_reward`, `differential_td_update`, and `differential_td`. The `MountainCar` class and its tile features are given.

The first checks verify exact values for both feature families and the $n$-step return, including the terminal window. The controller check trains `n_step_sarsa` with $n = 1$ and $n = 8$ on five seeds, and requires $n = 8$ to take fewer steps. The last four checks cover the average reward against its closed form, the hand-computed differential update, the exact differential values as a fixed point on a deterministic cycle, and recovery of the average reward on the cycle and on the factory ring.
