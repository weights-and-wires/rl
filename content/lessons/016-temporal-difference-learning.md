---
title: Temporal-difference learning
slug: 016-temporal-difference-learning
part: "Part 3 — Learning from experience"
summary: Update a value estimate after every step toward the reward plus your own guess for the next state, and see why that learns faster than waiting.
minutes: 35
gpu: false
objectives:
  - Update a value estimate toward a bootstrapped target after every step.
  - Explain why the target uses the current estimate of the next state.
  - Show that batch TD(0) uses the structure of the data, which Monte Carlo ignores.
  - "Say what bootstrapping trades: lower variance for bias from the current estimate."
prereqs:
  - 009-the-bellman-equation
  - 013-monte-carlo-prediction
exercise: 016-temporal-difference-learning
---

# Temporal-difference learning

You drive home from work and guess that the trip takes 30 minutes. Ten minutes in, you hit traffic on the bridge. You do not wait until you park to change your guess. At the bridge you already know the trip will be longer, so you revise the estimate for the start of the drive, and for every road you took to get here.

Monte Carlo prediction cannot do this. It waits for the end of the episode, reads off the return, and only then corrects each state it visited. That has two costs. A long episode delays every update, and an episode that never ends gives no update at all. The return is also noisy, because it adds up every random event until the end.

This chapter builds the alternative: *temporal-difference learning*, or *TD learning*. After each single step, it compares the guess you had with the guess you have after the step, and moves the old guess toward the new one. You will see why this is a fair thing to do, work it by hand on a three-state chain, and then write it.

## What you already know

- The return $G_t = r_t + \gamma G_{t+1}$ is the reward now plus the discounted return from the next step.
- The Bellman equation says $V^\pi(s)$ is the expected value of $r_t + \gamma V^\pi(s_{t+1})$ when you start in $s$ and follow $\pi$ (see [The Bellman equation](/lessons/009-the-bellman-equation)).
- Monte Carlo prediction moves $V(s_t)$ a fraction $\alpha$ toward the return it saw (see [Monte Carlo prediction](/lessons/013-monte-carlo-prediction)).
- A terminal state has value zero, because nothing follows it.
- In the five-state random walk, each step goes left or right with probability one half, and leaving on the right pays 1. The true value of state $k$ is $k/6$.

## Wait for the end, or guess the rest

Monte Carlo moves $V(s_t)$ toward $G_t$, the sum of all rewards to the end. The Bellman equation offers a shortcut. It says $G_t$ and $r_t + \gamma V(s_{t+1})$ have the same expected value when $V$ is right. The second expression needs only one reward and one lookup in your own table, so you can compute it after one step.

So the shortcut is to replace "all the rewards from here on" with "my current estimate of them". Using your own estimate inside your own target is called *bootstrapping*, after the story of the man who lifts himself by his bootstraps. It feels circular, because the estimate you rely on is still wrong. The next section shows why the circle still closes.

## Watch news travel back

Take a chain of three states. You start in state 0, step to state 1 with reward 0, and step to the end with reward 1. The only action is "go forward", so every episode is identical. Start every estimate at 0.5 and use $\alpha = 1$, which replaces an estimate with its target outright, so that every number is easy to follow. Terminal states have value 0, and $\gamma = 1$.

In episode 1, the agent takes two steps:

1. From state 0 to state 1 with reward 0. The target is $0 + V(1) = 0.5$, so $V(0)$ stays 0.5. The agent learned nothing, because state 1's estimate was still the starting guess.
2. From state 1 to the end with reward 1. A terminal state has no future, so the target is 1 alone. $V(1)$ becomes 1.

In episode 2, the first step finds $V(1) = 1$. The target is $0 + 1 = 1$, so $V(0)$ becomes 1. Both values are now right.

```figure
<svg viewBox="0 0 560 320" role="img" aria-label="A three-state chain; after one episode only the state next to the reward has learned, and after two the first state has too">
<defs><marker id="c016-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<text x="20.0" y="34.0" class="fig-bold" text-anchor="start">Before: every guess is 0.5</text>
<circle cx="110.0" cy="70.0" r="28" class="fill-soft-a"/>
<circle cx="110" cy="70" r="28" class="stroke-a" fill="none"/>
<text x="110.0" y="75.0" class="fig-bold" text-anchor="middle">V = 0.5</text>
<circle cx="280.0" cy="70.0" r="28" class="fill-soft-a"/>
<circle cx="280" cy="70" r="28" class="stroke-a" fill="none"/>
<text x="280.0" y="75.0" class="fig-bold" text-anchor="middle">V = 0.5</text>
<circle cx="450.0" cy="70.0" r="28" class="fill-sunk"/>
<circle cx="450" cy="70" r="28" class="stroke-muted" fill="none"/>
<text x="450.0" y="75.0" class="fig-bold" text-anchor="middle">end</text>
<line x1="140.0" y1="70.0" x2="248.0" y2="70.0" class="stroke-ink" marker-end="url(#c016-arrow)"/>
<line x1="310.0" y1="70.0" x2="418.0" y2="70.0" class="stroke-ink" marker-end="url(#c016-arrow)"/>
<text x="195.0" y="58.0" class="fig-small" text-anchor="middle">reward 0</text>
<text x="365.0" y="58.0" class="fig-small" text-anchor="middle">reward 1</text>
<text x="20.0" y="134.0" class="fig-bold" text-anchor="start">After episode 1</text>
<circle cx="110.0" cy="170.0" r="28" class="fill-soft-a"/>
<circle cx="110" cy="170" r="28" class="stroke-a" fill="none"/>
<text x="110.0" y="175.0" class="fig-bold" text-anchor="middle">V = 0.5</text>
<circle cx="280.0" cy="170.0" r="28" class="fill-soft-a"/>
<circle cx="280" cy="170" r="28" class="stroke-a" fill="none"/>
<text x="280.0" y="175.0" class="fig-bold" text-anchor="middle">V = 1</text>
<circle cx="450.0" cy="170.0" r="28" class="fill-sunk"/>
<circle cx="450" cy="170" r="28" class="stroke-muted" fill="none"/>
<text x="450.0" y="175.0" class="fig-bold" text-anchor="middle">end</text>
<line x1="140.0" y1="170.0" x2="248.0" y2="170.0" class="stroke-ink" marker-end="url(#c016-arrow)"/>
<line x1="310.0" y1="170.0" x2="418.0" y2="170.0" class="stroke-ink" marker-end="url(#c016-arrow)"/>
<text x="195.0" y="158.0" class="fig-small" text-anchor="middle">reward 0</text>
<text x="365.0" y="158.0" class="fig-small" text-anchor="middle">reward 1</text>
<text x="20.0" y="234.0" class="fig-bold" text-anchor="start">After episode 2</text>
<circle cx="110.0" cy="270.0" r="28" class="fill-soft-a"/>
<circle cx="110" cy="270" r="28" class="stroke-a" fill="none"/>
<text x="110.0" y="275.0" class="fig-bold" text-anchor="middle">V = 1</text>
<circle cx="280.0" cy="270.0" r="28" class="fill-soft-a"/>
<circle cx="280" cy="270" r="28" class="stroke-a" fill="none"/>
<text x="280.0" y="275.0" class="fig-bold" text-anchor="middle">V = 1</text>
<circle cx="450.0" cy="270.0" r="28" class="fill-sunk"/>
<circle cx="450" cy="270" r="28" class="stroke-muted" fill="none"/>
<text x="450.0" y="275.0" class="fig-bold" text-anchor="middle">end</text>
<line x1="140.0" y1="270.0" x2="248.0" y2="270.0" class="stroke-ink" marker-end="url(#c016-arrow)"/>
<line x1="310.0" y1="270.0" x2="418.0" y2="270.0" class="stroke-ink" marker-end="url(#c016-arrow)"/>
<text x="195.0" y="258.0" class="fig-small" text-anchor="middle">reward 0</text>
<text x="365.0" y="258.0" class="fig-small" text-anchor="middle">reward 1</text>
</svg>
Figure 1. With a step size of 1, the reward reaches state 1 in the first episode and state 0 only in the second. Notice that bootstrapping moves news back one state per update.
```

This is why the circle closes. Each update is only as wrong as the next state's estimate, and the reward at the end is real. The real reward fixes the last state first, then that fixed estimate fixes the one before it, and so on back down the chain. Information moves back one state per update, and every update uses data from only one step.

## Write the update

The pattern in both steps is the same: compute a target, find the gap to the current estimate, and close a fraction $\alpha$ of the gap. The gap has its own name, the *TD error*:

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad
V(s_t) \leftarrow V(s_t) + \alpha\, \delta_t
$$

Read it aloud: $\delta_t$ is the reward you got, plus the discounted value of where you landed, minus what you predicted. If it is positive, the step went better than you expected, so you raise $V(s_t)$ by a fraction $\alpha$ of the surprise. When $s_{t+1}$ is terminal, drop the $\gamma V(s_{t+1})$ term. This method is called TD(0), where the 0 means "bootstrap after zero extra steps". In numpy:

```python
target = r + gamma * V[s2] * (not terminated)
V[s] += alpha * (target - V[s])
```

Here is a second check with $V = (0, 0.5, 1.0)$, $\alpha = 0.1$, and $\gamma = 0.9$. A step from state 1 pays 1 and lands in state 2. If state 2 is not terminal, the target is $1 + 0.9 \times 1.0 = 1.9$, the error is $1.9 - 0.5 = 1.4$, and $V(1)$ becomes $0.5 + 0.14 = 0.64$. If state 2 is terminal, the target is 1, the error is 0.5, and $V(1)$ becomes 0.55.

> [!WARNING]
> Forgetting the terminal mask bootstraps from a state that has no future. The target then includes a value that never pays anything, so the estimate before the end of every episode is wrong.

## Why TD is quieter than Monte Carlo

Why trade a whole return for one step and a guess? Look at how big the errors are. Take the random walk with every estimate set to its true value, $k/6$, and one episode that visits states 3, 4, 3, 4, 5 and exits on the right with reward 1. If the estimates are already right, a good method would barely move them. Here is what each method sees at each step:

| Step | State | Monte Carlo error $G_t - V(s_t)$ | TD error $\delta_t$ |
|---|---|---|---|
| 0 | 3 | $1 - 3/6 = 0.50$ | $4/6 - 3/6 = 0.17$ |
| 1 | 4 | $1 - 4/6 = 0.33$ | $3/6 - 4/6 = -0.17$ |
| 2 | 3 | $1 - 3/6 = 0.50$ | $4/6 - 3/6 = 0.17$ |
| 3 | 4 | $1 - 4/6 = 0.33$ | $5/6 - 4/6 = 0.17$ |
| 4 | 5 | $1 - 5/6 = 0.17$ | $1 - 5/6 = 0.17$ |

```figure
<svg viewBox="0 0 580 290" role="img" aria-label="Monte Carlo errors against TD errors for each step of one episode on the random walk, with values at their true numbers">
<line x1="70.0" y1="200.8" x2="540.0" y2="200.8" class="stroke-grid"/>
<text x="62.0" y="204.8" class="fig-small" text-anchor="end">-0.2</text>
<line x1="70.0" y1="171.5" x2="540.0" y2="171.5" class="stroke-grid"/>
<text x="62.0" y="175.5" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="142.3" x2="540.0" y2="142.3" class="stroke-grid"/>
<text x="62.0" y="146.3" class="fig-small" text-anchor="end">0.2</text>
<line x1="70.0" y1="113.1" x2="540.0" y2="113.1" class="stroke-grid"/>
<text x="62.0" y="117.1" class="fig-small" text-anchor="end">0.4</text>
<line x1="70.0" y1="83.8" x2="540.0" y2="83.8" class="stroke-grid"/>
<text x="62.0" y="87.8" class="fig-small" text-anchor="end">0.6</text>
<line x1="70.0" y1="54.6" x2="540.0" y2="54.6" class="stroke-grid"/>
<text x="62.0" y="58.6" class="fig-small" text-anchor="end">0.8</text>
<line x1="70.0" y1="230.0" x2="540.0" y2="230.0" class="stroke-muted"/>
<text x="62.0" y="28.0" text-anchor="end">error</text>
<line x1="70.0" y1="171.5" x2="540.0" y2="171.5" class="stroke-ink"/>
<rect x="89.0" y="98.5" width="26.0" height="73.1" class="fill-b"/>
<text x="102.0" y="93.5" class="fig-small" text-anchor="middle">0.50</text>
<rect x="119.0" y="147.2" width="26.0" height="24.4" class="fill-a"/>
<text x="132.0" y="142.2" class="fig-small" text-anchor="middle">0.17</text>
<text x="117.0" y="250.0" class="fig-small" text-anchor="middle">step 0</text>
<text x="117.0" y="266.0" class="fig-small" text-anchor="middle">state 3</text>
<rect x="183.0" y="122.8" width="26.0" height="48.7" class="fill-b"/>
<text x="196.0" y="117.8" class="fig-small" text-anchor="middle">0.33</text>
<rect x="213.0" y="171.5" width="26.0" height="24.4" class="fill-a"/>
<text x="226.0" y="209.9" class="fig-small" text-anchor="middle">-0.17</text>
<text x="211.0" y="250.0" class="fig-small" text-anchor="middle">step 1</text>
<text x="211.0" y="266.0" class="fig-small" text-anchor="middle">state 4</text>
<rect x="277.0" y="98.5" width="26.0" height="73.1" class="fill-b"/>
<text x="290.0" y="93.5" class="fig-small" text-anchor="middle">0.50</text>
<rect x="307.0" y="147.2" width="26.0" height="24.4" class="fill-a"/>
<text x="320.0" y="142.2" class="fig-small" text-anchor="middle">0.17</text>
<text x="305.0" y="250.0" class="fig-small" text-anchor="middle">step 2</text>
<text x="305.0" y="266.0" class="fig-small" text-anchor="middle">state 3</text>
<rect x="371.0" y="122.8" width="26.0" height="48.7" class="fill-b"/>
<text x="384.0" y="117.8" class="fig-small" text-anchor="middle">0.33</text>
<rect x="401.0" y="147.2" width="26.0" height="24.4" class="fill-a"/>
<text x="414.0" y="142.2" class="fig-small" text-anchor="middle">0.17</text>
<text x="399.0" y="250.0" class="fig-small" text-anchor="middle">step 3</text>
<text x="399.0" y="266.0" class="fig-small" text-anchor="middle">state 4</text>
<rect x="465.0" y="147.2" width="26.0" height="24.4" class="fill-b"/>
<text x="478.0" y="142.2" class="fig-small" text-anchor="middle">0.17</text>
<rect x="495.0" y="147.2" width="26.0" height="24.4" class="fill-a"/>
<text x="508.0" y="142.2" class="fig-small" text-anchor="middle">0.17</text>
<text x="493.0" y="250.0" class="fig-small" text-anchor="middle">step 4</text>
<text x="493.0" y="266.0" class="fig-small" text-anchor="middle">state 5</text>
<rect x="330.0" y="44.0" width="14.0" height="14.0" class="fill-b"/>
<text x="350.0" y="56.0" class="fig-small" text-anchor="start">Monte Carlo error G − V</text>
<rect x="330.0" y="66.0" width="14.0" height="14.0" class="fill-a"/>
<text x="350.0" y="78.0" class="fig-small" text-anchor="start">TD error δ</text>
</svg>
Figure 2. One episode 3, 4, 3, 4, 5, exit with the true values k/6 as the estimates. Notice that every TD error is 1/6 in size, while the Monte Carlo errors swing from 1/6 to 1/2, because each carries the whole episode's luck.
```

Every TD error is the size of one coin flip. Every Monte Carlo error carries the luck of the whole rest of the episode, so it is larger and varies more. The two are tied together. When $V$ does not change during the episode, the Monte Carlo error is the discounted sum of the TD errors from there on. For step 0 in the table, $0.17 - 0.17 + 0.17 + 0.17 + 0.17 = 0.50$ (exactly, $3/6$). The $\gamma V(s_{k+1})$ terms cancel in pairs, so TD spreads one Monte Carlo update into one small piece per step.

The price is bias. The Monte Carlo target $G_t$ is correct on average even when $V$ is wrong, whereas the TD target is wrong whenever $V(s_{t+1})$ is, and early in learning it always is. Bootstrapping trades variance for bias. On the random walk, with every estimate starting at 0.5, the trade pays off.

```figure
<svg viewBox="0 0 580 290" role="img" aria-label="RMS error of TD(0) and constant-step Monte Carlo on the five-state random walk over 100 episodes">
<line x1="70.0" y1="230.0" x2="540.0" y2="230.0" class="stroke-grid"/>
<text x="62.0" y="234.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="190.0" x2="540.0" y2="190.0" class="stroke-grid"/>
<text x="62.0" y="194.0" class="fig-small" text-anchor="end">0.1</text>
<line x1="70.0" y1="150.0" x2="540.0" y2="150.0" class="stroke-grid"/>
<text x="62.0" y="154.0" class="fig-small" text-anchor="end">0.2</text>
<line x1="70.0" y1="110.0" x2="540.0" y2="110.0" class="stroke-grid"/>
<text x="62.0" y="114.0" class="fig-small" text-anchor="end">0.3</text>
<line x1="70.0" y1="70.0" x2="540.0" y2="70.0" class="stroke-grid"/>
<text x="62.0" y="74.0" class="fig-small" text-anchor="end">0.4</text>
<line x1="70.0" y1="30.0" x2="540.0" y2="30.0" class="stroke-grid"/>
<text x="62.0" y="34.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="70.0" y1="230.0" x2="540.0" y2="230.0" class="stroke-muted"/>
<line x1="70.0" y1="230.0" x2="70.0" y2="235.0" class="stroke-muted"/>
<text x="70.0" y="250.0" class="fig-small" text-anchor="middle">0</text>
<line x1="187.5" y1="230.0" x2="187.5" y2="235.0" class="stroke-muted"/>
<text x="187.5" y="250.0" class="fig-small" text-anchor="middle">25</text>
<line x1="305.0" y1="230.0" x2="305.0" y2="235.0" class="stroke-muted"/>
<text x="305.0" y="250.0" class="fig-small" text-anchor="middle">50</text>
<line x1="422.5" y1="230.0" x2="422.5" y2="235.0" class="stroke-muted"/>
<text x="422.5" y="250.0" class="fig-small" text-anchor="middle">75</text>
<line x1="540.0" y1="230.0" x2="540.0" y2="235.0" class="stroke-muted"/>
<text x="540.0" y="250.0" class="fig-small" text-anchor="middle">100</text>
<text x="305.0" y="272.0" text-anchor="middle">episodes</text>
<text x="62.0" y="18.0" text-anchor="end">RMS error</text>
<polyline points="70.0,135.7 74.7,137.1 79.4,139.3 84.1,140.5 88.8,142.1 93.5,142.7 98.2,143.7 102.9,145.2 107.6,146.6 112.3,148.5 117.0,150.9 121.7,152.8 126.4,154.4 131.1,156.0 135.8,158.1 140.5,159.4 145.2,160.1 149.9,160.8 154.6,162.1 159.3,162.9 164.0,164.0 168.7,165.4 173.4,167.1 178.1,168.0 182.8,169.2 187.5,168.7 192.2,169.8 196.9,170.8 201.6,170.8 206.3,172.0 211.0,173.7 215.7,174.5 220.4,174.9 225.1,176.2 229.8,177.2 234.5,177.3 239.2,178.4 243.9,180.3 248.6,180.7 253.3,181.0 258.0,181.3 262.7,181.3 267.4,182.2 272.1,183.8 276.8,184.2 281.5,184.6 286.2,185.6 290.9,186.7 295.6,186.5 300.3,186.5 305.0,186.8 309.7,187.2 314.4,188.1 319.1,188.1 323.8,187.8 328.5,186.3 333.2,187.6 337.9,188.4 342.6,187.6 347.3,188.7 352.0,190.7 356.7,191.3 361.4,191.3 366.1,192.5 370.8,194.1 375.5,194.1 380.2,194.2 384.9,194.6 389.6,196.2 394.3,196.2 399.0,195.1 403.7,195.7 408.4,195.9 413.1,196.3 417.8,195.5 422.5,195.4 427.2,194.8 431.9,197.0 436.6,196.9 441.3,197.8 446.0,196.0 450.7,196.5 455.4,196.5 460.1,196.9 464.8,195.4 469.5,195.8 474.2,196.8 478.9,196.8 483.6,198.9 488.3,198.5 493.0,199.2 497.7,197.4 502.4,198.2 507.1,198.7 511.8,199.5 516.5,199.6 521.2,199.4 525.9,198.4 530.6,198.4 535.3,197.9 540.0,196.4" class="stroke-b" fill="none"/>
<polyline points="70.0,135.7 74.7,138.5 79.4,141.2 84.1,143.8 88.8,146.4 93.5,148.9 98.2,151.3 102.9,153.6 107.6,155.9 112.3,158.1 117.0,160.2 121.7,162.3 126.4,164.5 131.1,166.5 135.8,168.5 140.5,170.3 145.2,172.1 149.9,173.9 154.6,175.6 159.3,177.4 164.0,178.9 168.7,180.3 173.4,181.7 178.1,183.3 182.8,184.7 187.5,186.2 192.2,187.4 196.9,188.6 201.6,189.8 206.3,191.2 211.0,192.3 215.7,193.3 220.4,194.2 225.1,194.9 229.8,195.9 234.5,196.9 239.2,197.5 243.9,198.2 248.6,199.5 253.3,200.2 258.0,200.6 262.7,201.4 267.4,202.2 272.1,202.5 276.8,203.3 281.5,204.2 286.2,204.8 290.9,205.4 295.6,206.3 300.3,207.1 305.0,207.6 309.7,207.9 314.4,208.3 319.1,208.5 323.8,209.1 328.5,210.0 333.2,210.1 337.9,210.7 342.6,211.5 347.3,211.6 352.0,212.2 356.7,212.0 361.4,211.8 366.1,212.0 370.8,212.4 375.5,212.3 380.2,212.9 384.9,213.2 389.6,213.9 394.3,214.4 399.0,214.5 403.7,214.7 408.4,214.7 413.1,215.0 417.8,214.5 422.5,214.5 427.2,214.9 431.9,215.4 436.6,215.2 441.3,215.1 446.0,215.3 450.7,214.9 455.4,215.2 460.1,215.3 464.8,216.2 469.5,215.5 474.2,215.9 478.9,215.5 483.6,215.5 488.3,215.4 493.0,214.8 497.7,214.6 502.4,214.6 507.1,214.8 511.8,214.9 516.5,215.5 521.2,215.8 525.9,215.6 530.6,215.6 535.3,215.8 540.0,216.2" class="stroke-a" fill="none"/>
<line x1="380.0" y1="52.0" x2="410.0" y2="52.0" class="stroke-a"/>
<text x="416.0" y="56.0" class="fig-small" text-anchor="start">TD(0), α = 0.05: 0.034</text>
<line x1="380.0" y1="74.0" x2="410.0" y2="74.0" class="stroke-b"/>
<text x="416.0" y="78.0" class="fig-small" text-anchor="start">Monte Carlo, α = 0.02: 0.084</text>
</svg>
Figure 3. RMS error over the five states, averaged over 40 runs, with every estimate starting at 0.5. Notice that TD(0) ends lower even though its step size is larger.
```

> [!KEY]
> TD(0) updates toward $r_t + \gamma V(s_{t+1})$ after every step. It learns from incomplete episodes, and it has lower variance than Monte Carlo, at the cost of bias from its own current estimates.

## Learn from a fixed batch

Suppose you keep a fixed set of episodes and replay them until the values stop changing. In each sweep, hold $V$ fixed, add up the update for every step, and then apply the total. With a small $\alpha$, batch Monte Carlo and batch TD(0) each converge, but to different answers.

Sutton and Barto's Example 6.4 shows the difference. There are two states, $A$ and $B$, and eight episodes. One episode goes $A \to B$ with reward 0, then ends from $B$ with reward 0. Six episodes start in $B$ and end with reward 1. One episode starts in $B$ and ends with reward 0.

Both methods give $V(B) = 6/8 = 0.75$, because $B$ ended with reward 1 in six of its eight visits. They disagree about $A$. The only return ever seen from $A$ was 0, so batch Monte Carlo says $V(A) = 0$. Batch TD says $V(A) = 0.75$, because every visit to $A$ went to $B$ with reward 0, and $B$ is worth 0.75.

TD's answer is the one you get by fitting a model to the data. Estimate each transition probability as a fraction of observed transitions, and each reward as a mean, then solve that model's Bellman equation exactly. Batch TD(0) converges to those values, the *certainty-equivalence* estimate. Monte Carlo instead matches the returns it saw, and ignores the fact that $A$ always leads to $B$. In this example, TD uses structure that Monte Carlo throws away.

> [!KEY]
> You revise your guess at the bridge, not at the parking space, because the next state's estimate already says how the rest of the drive goes. TD(0) does exactly that after every step. It uses the Markov structure that Monte Carlo ignores, so batch TD(0) agrees with the values of the best-fit model, and Monte Carlo can disagree with it.

> [!RECAP]
> - The TD target is `r + gamma * V[s2] * (not terminated)`. Move `V[s]` a fraction `alpha` toward it.
> - The target is available after one step, so TD learns from incomplete episodes.
> - Each update passes news back one state, so a reward takes several episodes to reach early states.
> - TD errors are small and Monte Carlo errors are large, so TD has lower variance and some bias.
> - Batch TD(0) holds $V$ fixed in a sweep, sums the increments, and converges to the model-based values.

## Build it step by step

The checks test the functions in this order. `random_walk` is already written.

1. **`td0_update(V, s, r, s2, terminated, alpha, gamma)`.** Takes the value array and one transition. It updates `V[s]` in place and returns `V`. The common mistake is bootstrapping from a terminal state, so multiply `V[s2]` by `not terminated`.
2. **`td0_prediction(mdp, pi, n_episodes, alpha, seed, v_init=0.5)`.** Returns `V` with shape `(S,)`. Start terminal states at 0 and the others at `v_init`. Draw actions with `rng.choice(A, p=pi[s])` and steps with `mdp.sample(s, a, rng)`, and call `td0_update` after every step. The common mistake is to wait until the end of the episode to update.
3. **`batch_td0(episodes, n_states, gamma, alpha, n_sweeps)`.** Each episode is `(states, rewards)`, and `states` has one extra entry, the terminal state. Inside each sweep, add every `alpha * delta` into a separate array, and add that array to `V` after the sweep. The common mistake is updating `V` inside the sweep, which is online TD and settles near, but not exactly on, the batch answer.

## Where this breaks

- **Slow propagation.** One TD update moves news back by one step, as the chain showed. A reward at the end of a 100-step corridor needs many episodes to reach the start. [n-step returns](/lessons/019-n-step-returns) send it back faster.
- **A bad start.** Bias from the starting values fades only as fast as the news travels. Estimates that begin far from the truth distort the targets for a long time.
- **Constant step sizes.** With a constant $\alpha$, the values keep moving in a noisy world. People decay $\alpha$ over time, or accept the jitter.
- **Control.** This chapter predicts the values of one policy. [SARSA and Q-learning](/lessons/017-sarsa-and-q-learning) use the same update to choose actions.

## Further reading

- [Sutton, "Learning to predict by the methods of temporal differences" (1988)](https://doi.org/10.1007/BF00115009). The paper that introduced TD learning, including the batch convergence result.
- [Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd edition (2018), Chapter 6](http://incompleteideas.net/book/the-book-2nd.html). The random walk and Example 6.4.
- [Tesauro, "Temporal difference learning and TD-Gammon" (1995)](https://doi.org/10.1145/203330.203343). TD learning applied to backgammon.

## The exercise

You write `td0_update`, `td0_prediction`, and `batch_td0`. The checks verify the update by hand, including the terminal case. They run `td0_prediction` on the three-state chain and expect values `[0.5, 1, 0]` after one episode and `[1, 1, 0]` after two. `batch_td0` must give $V(A) = V(B) = 0.75$ on Example 6.4 and must hold $V$ fixed within a sweep. On the random walk, `td0_prediction` must keep terminal values at zero and have a lower RMS error than constant-step Monte Carlo after 100 episodes.
