---
title: Sim to real
slug: 082-sim-to-real
part: "Part 10 — Beyond text, and what can go wrong"
summary: Train in a simulator whose physics you can randomize, identify the real physics from a few transitions, and report the reality gap honestly.
minutes: 40
gpu: false
objectives:
  - Describe the reality gap as a shift in the dynamics parameters, not only in the states a policy visits.
  - Train on a distribution of physics parameters and measure the robust-versus-nominal trade-off.
  - Fit mass and friction from transitions by least squares, and say why the regression target is the acceleration.
  - Use identified parameters to adapt a controller, and explain why history lets a policy do the same implicitly.
  - Report a simulator score, a held-out real score, and a worst case.
prereqs:
  - 081-rl-for-vision-speech-and-robots
  - 043-model-based-rl
  - 045-partial-observability-and-memory
exercise: 082-sim-to-real
---

# Sim to real

You train a controller for a robot in a simulator, because the simulator is fast, safe, and free. In the simulator the controller is excellent. Then you load it onto the real robot, which has a heavier arm, a stiffer joint, or a worn motor, and it shakes itself apart. Nothing in the controller is wrong. It is tuned to one set of physics, and the real robot has different physics.

This chapter shows the failure in a system small enough to compute by hand, then gives you three tools against it. Train on many randomized physics, so reality looks like one more draw. Measure the real physics from a few transitions and adapt to it. Or give the policy a short memory, so it infers the physics while it acts. The chapter closes with how to report the result without flattering it.

## What you already know

- A simulator is a model of $P(s' \mid s, a)$, and a policy tuned in it inherits the model's mistakes (see [Model-based RL](/lessons/043-model-based-rl)).
- A hidden quantity makes a problem partially observable, and memory can recover it (see [Partial observability and memory](/lessons/045-partial-observability-and-memory)).
- Least squares fits a linear model to data in closed form.
- A residual policy adds a bounded correction to a fixed controller (see [RL for vision, speech, and robots](/lessons/081-rl-for-vision-speech-and-robots)).

## The reality gap is a change in the dynamics

The running example is a point mass on a line. The state is $s = (x, v)$, the position and velocity. The action $a = u$ is a force. The physics has two parameters, the mass $m$ and a friction coefficient $b$, which together I call $\xi = (m, b)$:

$$
v' = v + \Delta t \, \frac{u - b\,v}{m}, \qquad x' = x + \Delta t \, v'.
$$

Read it aloud: the new velocity is the old one plus the acceleration times the time step, and the acceleration is the force minus the friction drag, divided by the mass. The time step $\Delta t$ is 0.05 seconds. The task is to bring the mass from $x = 1$ to rest at $x = 0$ in 100 steps, at a cost per step of $x^2 + 0.1\,v^2 + 0.002\,u^2$. The policy is the feedback law $u = -k_x x - k_v v$.

The simulator uses $\xi_0 = (1.0, 0.5)$. Reality is some other $\xi$. This is a *reality gap*: the policy sees the same kind of observation, but the function that maps a state and action to the next state has changed. This differs from a shift in the states a policy visits: the same state and action produce a different next state, so covering more training states does not repair it.

Why does a change in $\xi$ matter? Tune the two gains on $\xi_0$ and the best answer is aggressive, because force is cheap in the cost: about $k_x = 17.7$ and $k_v = 7.25$. Look at the velocity loop alone. Each step multiplies the velocity by about $1 - \Delta t\,(k_v + b)/m$. At the simulator mass of 1.0 that factor is $1 - 0.05 \cdot 7.75 / 1.0 = 0.61$, so the velocity decays smoothly. At a mass of 0.15 it is $1 - 0.05 \cdot 7.75 / 0.15 = -1.6$. A factor below $-1$ flips the sign and grows each step, so the controller oscillates harder every step. On a held-out grid of real robots the mean cost is 0.516 and the worst is 3.33, against 0.0875 on the simulator.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Cost against mass for three controllers: the gain tuned on the simulator blows up for light robots, the randomized gain stays flat, and the adaptive controller is lowest">
<line x1="70" y1="182.9" x2="570" y2="182.9" class="stroke-grid"/>
<text x="62" y="186.9" class="fig-small" text-anchor="end">0.1</text>
<line x1="70" y1="92.9" x2="570" y2="92.9" class="stroke-grid"/>
<text x="62" y="96.9" class="fig-small" text-anchor="end">1</text>
<line x1="70" y1="210" x2="570" y2="210" class="stroke-muted"/>
<text x="70.0" y="228" class="fig-small" text-anchor="middle">0.1</text>
<text x="185.7" y="228" class="fig-small" text-anchor="middle">0.2</text>
<text x="338.6" y="228" class="fig-small" text-anchor="middle">0.5</text>
<text x="454.3" y="228" class="fig-small" text-anchor="middle">1</text>
<text x="570.0" y="228" class="fig-small" text-anchor="middle">2</text>
<text x="320" y="248" class="fig-small" text-anchor="middle">true mass (log scale), friction 0.5</text>
<text x="62" y="20" class="fig-small" text-anchor="end">cost (log)</text>
<polyline points="70.0,37.1 82.8,42.4 95.6,47.3 108.5,52.0 121.3,56.4 134.1,60.5 146.9,64.3 159.7,67.8 172.6,71.7 185.4,75.1 198.2,175.0 211.0,186.8 223.8,190.2 236.7,191.7 249.5,192.5 262.3,192.9 275.1,193.1 287.9,193.1 300.8,193.1 313.6,193.0 326.4,192.8 339.2,192.5 352.1,192.2 364.9,191.8 377.7,191.4 390.5,191.0 403.3,190.5 416.2,190.0 429.0,189.4 441.8,188.8 454.6,188.1 467.4,187.4 480.3,186.6 493.1,185.8 505.9,185.0 518.7,184.0 531.5,183.1 544.4,182.1 557.2,181.0 570.0,179.9" fill="none" class="stroke-bad" stroke-width="2.5"/>
<polyline points="70.0,191.5 82.8,194.3 95.6,195.5 108.5,196.1 121.3,196.5 134.1,196.6 146.9,196.7 159.7,196.6 172.6,196.5 185.4,196.4 198.2,196.2 211.0,195.9 223.8,195.7 236.7,195.4 249.5,195.0 262.3,194.6 275.1,194.2 287.9,193.7 300.8,193.2 313.6,192.7 326.4,192.1 339.2,191.5 352.1,190.8 364.9,190.1 377.7,189.4 390.5,188.6 403.3,187.8 416.2,186.9 429.0,185.9 441.8,184.9 454.6,183.9 467.4,182.8 480.3,181.6 493.1,180.4 505.9,179.2 518.7,177.8 531.5,176.5 544.4,175.0 557.2,173.5 570.0,172.0" fill="none" class="stroke-b" stroke-width="2.5"/>
<polyline points="70.0,191.4 82.8,191.4 95.6,191.4 108.5,191.3 121.3,191.3 134.1,191.3 146.9,191.3 159.7,191.3 172.6,191.3 185.4,191.3 198.2,191.2 211.0,191.2 223.8,191.2 236.7,191.1 249.5,191.1 262.3,191.0 275.1,191.0 287.9,190.9 300.8,190.8 313.6,190.7 326.4,190.6 339.2,190.5 352.1,190.3 364.9,190.1 377.7,189.9 390.5,189.7 403.3,189.4 416.2,189.1 429.0,188.7 441.8,188.2 454.6,187.7 467.4,187.1 480.3,186.5 493.1,185.7 505.9,184.9 518.7,184.0 531.5,183.0 544.4,182.0 557.2,180.9 570.0,179.8" fill="none" class="stroke-c" stroke-width="2.5"/>
<line x1="454.3" y1="30" x2="454.3" y2="210" class="stroke-muted dashed"/>
<text x="460.3" y="40" class="fig-small">simulator mass</text>
<rect x="80" y="54" width="10" height="10" class="fill-bad"/>
<text x="96" y="63" class="fig-small">tuned on the simulator</text>
<rect x="80" y="70" width="10" height="10" class="fill-b"/>
<text x="96" y="79" class="fig-small">tuned on randomized physics</text>
<rect x="80" y="86" width="10" height="10" class="fill-c"/>
<text x="96" y="95" class="fig-small">adaptive, identified physics</text>
</svg>
Figure 1. Cost against the true mass for three controllers, computed on the exercise's point mass. The gain tuned on the simulator is the best near mass 1 and diverges below about 0.2. The randomized gain and the adaptive controller stay flat.
```

> [!KEY]
> A reality gap is a change in the dynamics parameters $\xi$. A policy tuned to one $\xi_0$ can be unstable at another, because the best policy for one physics often sits near the edge of stability for others.

## Train on a distribution of physics

The first tool is *domain randomization*: do not train on one $\xi$, but on a distribution $p(\xi)$, drawing a fresh $\xi$ for each episode. The objective becomes the expected return over physics:

$$
\max_{\pi} \; \mathbb{E}_{\xi \sim p}\big[ J(\pi; \xi) \big].
$$

Read it aloud: score the policy on a random robot, and average. Tobin and colleagues (2017) randomized textures and lighting in rendered scenes so that a real camera image looked like one more rendering. Peng and colleagues (2018) randomized the dynamics, including masses and friction, which is this chapter's version. OpenAI and colleagues (2019) went further for a robot hand that solves a Rubik's Cube. Their automatic domain randomization starts narrow and widens each range whenever the policy does well.

The range is the main design choice. Mass and friction are scales, so you draw them log-uniformly: 0.1 to 0.2 matters as much as 1 to 2. A narrow range leaves the real robot outside the distribution. A wide range makes one policy hedge across physics that need different behaviors.

The hedge has a price, and you can read it in the numbers. Tuning the gain by the cross-entropy method on 128 randomized draws gives $k_x = 8.8$ and $k_v = 2.8$, a gentler gain than the nominal one. Table 1 compares the costs, where lower is better.

| Controller | Simulator ($\xi_0$) | Real set, mean | Real set, worst |
|---|---|---|---|
| Tuned on the simulator | 0.0875 | 0.516 | 3.334 |
| Tuned on randomized physics | 0.0974 | 0.0999 | 0.1410 |
| Adaptive, identified physics | 0.0884 | 0.0921 | 0.1086 |

Table 1. Mean per-step cost for seed 0. The real set is a held-out 6 by 5 grid of masses and frictions.

The randomized gain gives up 11 percent on the simulator, from 0.0875 to 0.0974, and cuts the worst case on the real set by a factor of 24. That is the robust-versus-nominal trade-off: a small loss where you tuned buys a large gain elsewhere. Randomization optimizes the average over $p(\xi)$, so a rare dangerous corner counts only as much as its probability. For a dangerous tail, weight the tail.

## Identify the physics from transitions

The second tool is *system identification*: measure $\xi$ from real transitions, then use it. Here the physics is linear in two coefficients. Rearrange the dynamics into an acceleration:

$$
\underbrace{\frac{v' - v}{\Delta t}}_{\text{acceleration}} = \frac{1}{m}\, u \;-\; \frac{b}{m}\, v .
$$

Read it aloud: the measured acceleration is one coefficient times the force plus another coefficient times the velocity. Call the coefficients $c_u = 1/m$ and $c_v = -b/m$. Least squares over $N$ transitions finds both, and then $m = 1/c_u$ and $b = -c_v / c_u$.

Work it by hand with two transitions of a robot whose true mass is 2 and friction is 0.5. In the first, $v = 1$ and $u = 4$, so $v' = 1 + 0.05\,(4 - 0.5)/2 = 1.0875$ and the acceleration is $(1.0875 - 1)/0.05 = 1.75$. In the second, $v = 0$ and $u = 2$, so $v' = 0.05$ and the acceleration is $1.0$. The second equation says $1.0 = 2\,c_u$, so $c_u = 0.5$ and $m = 2$. The first says $1.75 = 4 \cdot 0.5 + c_v \cdot 1$, so $c_v = -0.25$ and $b = 0.25 / 0.5 = 0.5$. Two noiseless transitions recover both parameters exactly, and `np.linalg.lstsq` handles any number of noisy ones.

```python
accel = (S2[:, 1] - S[:, 1]) / dt          # regress the acceleration ...
A = np.stack([U, S[:, 1]], axis=1)          # ... on force and velocity
(c_u, c_v), *_ = np.linalg.lstsq(A, accel, rcond=None)
mass, friction = 1 / c_u, -c_v / c_u
```

> [!WARNING]
> Regress the acceleration, not the next position. The position change is $\Delta t$ times the new velocity, so its coefficients carry extra factors of $\Delta t$ and the mass you read back is wrong.

Identification needs data that varies the force and the velocity independently, because two columns that move together cannot be told apart. With noise of 0.002 on the velocity and 300 transitions, the fitted mass stays within 2 percent.

## Adapt to what you identified

With $\hat m$ and $\hat b$ in hand, you do not need a gain that hedges. Choose the force that cancels the estimated physics and imposes the closed loop you want:

$$
u = \hat m \,\big(-k_x x - k_v v\big) + \hat b\, v .
$$

Read it aloud: ask for the acceleration $-k_x x - k_v v$, multiply by the estimated mass to get the force, and add back the friction drag that you expect to lose. When $\hat\xi = \xi$, the robot follows $x'' = -k_x x - k_v v$ for any mass and friction, so every real robot behaves like the same well-tuned one. This chapter uses $k_x = 16$ and $k_v = 8$, a critically damped loop. In Table 1 the adaptive controller costs 0.0921 on average and 0.1086 at worst, below the randomized gain's 0.0999 and 0.1410, because it does not hedge. Over 30 seeds, its worst case stays below 83 percent of the randomized gain's and its mean below 96 percent.

## History as implicit identification

The randomized gain is one number pair for every robot, because a policy of the form $\pi(a \mid s)$ cannot tell a light robot from a heavy one. The physics $\xi$ is a hidden state of the kind in [Partial observability and memory](/lessons/045-partial-observability-and-memory): the current state does not reveal it, but the recent transitions do. A policy with a short memory can therefore act differently on different robots, which the randomized gain cannot.

Figure 2 shows the idea with the estimator from the exercise. At each step, `history_estimate` fits mass and friction from only the last 10 transitions. The true mass drops from 1.0 to 0.3 at step 50, as if the robot picked up a payload. The estimate lags by a few steps, since the window still holds old transitions, and then tracks the new mass exactly. A window of all 100 transitions would average the two robots. The controller adds force noise to keep the system excited.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Mass estimate from the last ten transitions while the true mass drops from 1.0 to 0.3 at step 50">
<line x1="60" y1="190.0" x2="570" y2="190.0" class="stroke-grid"/>
<text x="52" y="194.0" class="fig-small" text-anchor="end">0</text>
<line x1="60" y1="123.3" x2="570" y2="123.3" class="stroke-grid"/>
<text x="52" y="127.3" class="fig-small" text-anchor="end">0.5</text>
<line x1="60" y1="56.7" x2="570" y2="56.7" class="stroke-grid"/>
<text x="52" y="60.7" class="fig-small" text-anchor="end">1</text>
<text x="315" y="224" class="fig-small" text-anchor="middle">time step</text>
<text x="60.0" y="206" class="fig-small" text-anchor="middle">0</text>
<text x="188.8" y="206" class="fig-small" text-anchor="middle">25</text>
<text x="317.6" y="206" class="fig-small" text-anchor="middle">50</text>
<text x="446.4" y="206" class="fig-small" text-anchor="middle">75</text>
<text x="570.0" y="206" class="fig-small" text-anchor="middle">99</text>
<polyline points="60.0,56.7 65.2,56.7 70.3,56.7 75.5,56.7 80.6,56.7 85.8,56.7 90.9,56.7 96.1,56.7 101.2,56.7 106.4,56.7 111.5,56.7 116.7,56.7 121.8,56.7 127.0,56.7 132.1,56.7 137.3,56.7 142.4,56.7 147.6,56.7 152.7,56.7 157.9,56.7 163.0,56.7 168.2,56.7 173.3,56.7 178.5,56.7 183.6,56.7 188.8,56.7 193.9,56.7 199.1,56.7 204.2,56.7 209.4,56.7 214.5,56.7 219.7,56.7 224.8,56.7 230.0,56.7 235.2,56.7 240.3,56.7 245.5,56.7 250.6,56.7 255.8,56.7 260.9,56.7 266.1,56.7 271.2,56.7 276.4,56.7 281.5,56.7 286.7,56.7 291.8,56.7 297.0,56.7 302.1,56.7 307.3,56.7 312.4,56.7 317.6,150.0 322.7,150.0 327.9,150.0 333.0,150.0 338.2,150.0 343.3,150.0 348.5,150.0 353.6,150.0 358.8,150.0 363.9,150.0 369.1,150.0 374.2,150.0 379.4,150.0 384.5,150.0 389.7,150.0 394.8,150.0 400.0,150.0 405.2,150.0 410.3,150.0 415.5,150.0 420.6,150.0 425.8,150.0 430.9,150.0 436.1,150.0 441.2,150.0 446.4,150.0 451.5,150.0 456.7,150.0 461.8,150.0 467.0,150.0 472.1,150.0 477.3,150.0 482.4,150.0 487.6,150.0 492.7,150.0 497.9,150.0 503.0,150.0 508.2,150.0 513.3,150.0 518.5,150.0 523.6,150.0 528.8,150.0 533.9,150.0 539.1,150.0 544.2,150.0 549.4,150.0 554.5,150.0 559.7,150.0 564.8,150.0 570.0,150.0" fill="none" class="stroke-a dashed" stroke-width="2"/>
<polyline points="60.0,56.7 65.2,56.7 70.3,56.7 75.5,56.7 80.6,56.7 85.8,56.7 90.9,56.7 96.1,56.7 101.2,56.7 106.4,56.7 111.5,56.7 116.7,56.7 121.8,56.7 127.0,56.7 132.1,56.7 137.3,56.7 142.4,56.7 147.6,56.7 152.7,56.7 157.9,56.7 163.0,56.7 168.2,56.7 173.3,56.7 178.5,56.7 183.6,56.7 188.8,56.7 193.9,56.7 199.1,56.7 204.2,56.7 209.4,56.7 214.5,56.7 219.7,56.7 224.8,56.7 230.0,56.7 235.2,56.7 240.3,56.7 245.5,56.7 250.6,56.7 255.8,56.7 260.9,56.7 266.1,56.7 271.2,56.7 276.4,56.7 281.5,56.7 286.7,56.7 291.8,56.7 297.0,56.7 302.1,56.7 307.3,56.7 312.4,56.7 317.6,56.7 322.7,88.6 327.9,121.5 333.0,125.8 338.2,127.2 343.3,134.5 348.5,137.9 353.6,142.7 358.8,148.7 363.9,150.3 369.1,150.0 374.2,150.0 379.4,150.0 384.5,150.0 389.7,150.0 394.8,150.0 400.0,150.0 405.2,150.0 410.3,150.0 415.5,150.0 420.6,150.0 425.8,150.0 430.9,150.0 436.1,150.0 441.2,150.0 446.4,150.0 451.5,150.0 456.7,150.0 461.8,150.0 467.0,150.0 472.1,150.0 477.3,150.0 482.4,150.0 487.6,150.0 492.7,150.0 497.9,150.0 503.0,150.0 508.2,150.0 513.3,150.0 518.5,150.0 523.6,150.0 528.8,150.0 533.9,150.0 539.1,150.0 544.2,150.0 549.4,150.0 554.5,150.0 559.7,150.0 564.8,150.0 570.0,150.0" fill="none" class="stroke-b" stroke-width="2.5"/>
<text x="343.3" y="172.0" class="fig-small">true mass</text>
<text x="85.8" y="44.7" class="fig-small">estimate from the last 10 steps</text>
</svg>
Figure 2. The mass estimate from the last 10 transitions follows a mass drop from 1.0 to 0.3 at step 50. It reads 0.514 at step 52 and 0.300 by step 60.
```

Rapid motor adaptation (RMA, Kumar and colleagues, 2021) learns this idea for a legged robot. In phase 1 it trains a policy in simulation with the privileged physics $\xi$, which an encoder compresses into a vector $z$. In phase 2 it trains an *adaptation module*, a network that predicts $z$ from the last 50 states and actions, and that needs no $\xi$. The adaptation module is a learned counterpart of `history_estimate`: implicit identification, with no gradient step at deployment.

> [!KEY]
> A memoryless policy must hedge across all the physics it might meet, and hedging costs return. History lets the policy infer the physics from recent transitions, so it can act as if it knew them.

## Report the gap honestly

A sim-to-real result is easy to flatter. Four habits keep it honest:

1. **Report the simulator score and the held-out score.** The gap is their difference.
2. **Keep the evaluation physics out of training.** The exercise's real set is a fixed grid, and training draws come from the random generator.
3. **Report the worst case beside the mean.** The simulator-tuned gain has a mean of 0.516, but the worst case of 3.33 is what breaks a robot.
4. **Say what the held-out set is.** Here it is the same simulator with other parameters, so it covers only the variation you modeled. A real robot adds latency, contact, and sensor effects outside $\xi$.

## Build it step by step

> [!RECAP]
> - The reality gap is a change in the dynamics parameters $\xi = (m, b)$.
> - Domain randomization trains on a distribution of $\xi$ and trades a little nominal return for a much smaller worst case.
> - System identification regresses the acceleration on force and velocity by least squares to recover $\xi$.
> - An adaptive controller uses $\hat\xi$ to cancel the physics, and a window over recent history estimates $\xi$ as it changes.

The starter defines `step`, `batch_cost`, `collect_transitions`, and `adaptive_cost`. You write five functions, in the order the checks run them:

1. **`randomized_params(rng, ranges, n)`.** Return an `(n, 2)` array of `[mass, friction]`. Draw log-uniformly from the `rng` you are given: `np.exp(lo + (hi - lo) * rng.random((n, 2)))`. The common mistake is creating `np.random.default_rng(0)` inside the function, so every episode gets the same physics.
2. **`sysid_fit(S, U, S2)`.** Build the acceleration from the velocity column, stack force and velocity as columns, call `np.linalg.lstsq`, and convert. The common mistake is regressing the position.
3. **`history_estimate(S, U, S2, window, prior)`.** Keep the last `window` rows and call `sysid_fit` on them. Return `prior` when fewer than three rows remain or `np.linalg.matrix_rank` of the design matrix is below 2.
4. **`adaptive_action(state, params_hat, gains)`.** One expression: the estimated mass times the wanted acceleration plus the estimated friction times the velocity. It must work on a batch.
5. **`tune_gain(params, seed)`.** Run the cross-entropy method over the two gains. Start from mean `[8, 4]` and standard deviation `[6, 3]`. Sample 60 gains, take absolute values, keep the 10 with the lowest mean cost over `params`, and refit.

## Where this breaks

- **Unmodeled physics.** Randomization covers only the variation you list. Latency, backlash, and contact stay in the gap.
- **Too much randomization.** A wide range trains a policy that excels nowhere.
- **Identification needs excitation.** A robot resting at its goal produces transitions that look alike, and the regression cannot separate the parameters.
- **Estimation error.** A bad $\hat\xi$ makes the controller cancel the wrong physics.
- **The model structure can be wrong.** Randomizing parameters cannot fix a missing term. Correct the model with real data, as in [Model-based RL](/lessons/043-model-based-rl), or add a residual policy, as in [RL for vision, speech, and robots](/lessons/081-rl-for-vision-speech-and-robots).

## Further reading

- [Tobin and colleagues (2017), Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907). Randomized renderings transfer an object detector to a real camera.
- [Peng and colleagues (2018), Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537). Randomized masses, friction, and damping, with a recurrent policy that adapts to them.
- [OpenAI and colleagues (2019), Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113). Automatic domain randomization widens each range as the policy improves.
- [Kumar and colleagues (2021), RMA: Rapid Motor Adaptation for Legged Robots](https://arxiv.org/abs/2107.04034). A privileged encoder in phase 1 and an adaptation module over recent history in phase 2.
- [Zhao, Queralta, and Westerlund (2020), Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303). A map of the methods, including randomization, identification, and adaptation.
- Stanford CS 224R, Spring 2026, Lecture 16, ["RL for Robots: Sim-to-Real Transfer (Guest Lecture: Guanya Shi)"](https://cs224r.stanford.edu/). The course schedule lists it as the sim-to-real lecture.

## The exercise

You build the tools on a point mass whose mass and friction change. Write `randomized_params`, `sysid_fit`, `history_estimate`, `adaptive_action`, and `tune_gain`.

The checks confirm that draws stay in range, differ from row to row and call to call, and are log-uniform. They check that `sysid_fit` recovers the physics to 8 decimals without noise and to 2 percent with it, that `history_estimate` follows a mass change and returns the prior on too little data, and that `adaptive_action` gives a closed-loop acceleration of $-16x - 8v$. The last three checks tune gains: the tuned gain lands within 5 percent of a grid search, the randomized gain cuts the worst case below 25 percent of the nominal gain's at a 2 to 50 percent simulator cost, and the adaptive controller beats the randomized gain on the held-out set, over five seeds.
