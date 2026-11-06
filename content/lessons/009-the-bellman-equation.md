---
title: The Bellman equation
slug: 009-the-bellman-equation
part: "Part 2 — Planning with a known world"
summary: Write a state's value as the reward now plus the discounted value of the next state, then solve that equation exactly with one linear solve.
minutes: 35
gpu: false
objectives:
  - Explain why a state's value equals the expected reward plus gamma times the expected value of the next state.
  - Solve a policy's values exactly as a linear system, by hand on a small MDP and with one call to np.linalg.solve.
  - Compute action values from state values, and check that they average back to the state values.
  - Say why exact evaluation and sampling agree, and where each one costs more.
prereqs:
  - 002-returns-and-discounting
  - 008-markov-decision-processes
exercise: 009-the-bellman-equation
---

# The Bellman equation

In the hallway from [Markov decision processes](/lessons/008-markov-decision-processes), suppose you follow the uniform random policy from cell 0. How good is that? "Good" needs a number: the average return you collect, over all the random paths the policy can take. Call that number the value of cell 0.

The obvious way to get it is to try. Run thousands of episodes, compute each return, and average. That works, but slowly. The three episodes in the last chapter earned $-4.1$, $-4.7$, and $-8.6$, so one episode tells you almost nothing, and every extra digit of accuracy costs about a hundred times more episodes. You also have the whole model written down as arrays. It feels wasteful to simulate a world you can read.

This chapter shows how to read the answer straight off the arrays. The trick is that a state's value depends on the values of the states it leads to, which gives you an equation to solve instead of a random average to estimate.

## What you already know

- A policy $\pi(a \mid s)$ is an `(S, A)` array whose rows sum to one (see [Markov decision processes](/lessons/008-markov-decision-processes)).
- `policy_matrices` turns a policy into $P_\pi$ and $r_\pi$, which hold its average moves and average rewards.
- The return satisfies $G_t = r_t + \gamma G_{t+1}$ (see [Returns and discounting](/lessons/002-returns-and-discounting)).
- The value $V^\pi(s)$ is the expected return from state $s$ when you follow $\pi$. A terminal state has value 0.
- `np.linalg.solve(M, b)` solves $M x = b$ without forming $M^{-1}$.

## Value is reward now plus value next

Start from $G_t = r_t + \gamma G_{t+1}$: the return is the first reward plus the discounted return of everything after it. Take the average of both sides, starting in state $s$ and following $\pi$. The left side is $V^\pi(s)$ by definition. On the right, the first reward averages to $r_\pi(s)$. The rest averages to $\gamma$ times the value of wherever you land, because from there on you again follow $\pi$:

$$
V^\pi(s) = r_\pi(s) + \gamma \sum_{s'} P_\pi(s, s')\, V^\pi(s')
$$

Read it as: a state's value is the reward you expect now, plus the discounted average value of the next state. This is the *Bellman equation* for a policy, named for Richard Bellman. It holds at every state at once.

Try it on the always-right policy in the hallway. Nothing is random, so each value is one number, and you can work backward from the exit. The exit has value 0.

- Cell 2: $-1 + 0.9 \times 0 = -1$.
- Cell 1: $-1 + 0.9 \times (-1) = -1.9$.
- Cell 0: $-1 + 0.9 \times (-1.9) = -2.71$.

That matches the return you computed in the last chapter for walking right three times, because every episode of this policy is the same.

## When the values depend on each other

Take the uniform policy instead, which sends you left or right with probability $1/2$. Cell 0 can lead back to cell 0, and cell 1 can lead back to cell 0, so you cannot start at the exit and work backward. Every value depends on another value that depends on it. Write the three equations with $\gamma = 0.9$, so each neighbor carries a weight of $0.9 \times 0.5 = 0.45$:

$$
\begin{aligned}
V(0) &= -1 + 0.45\,V(0) + 0.45\,V(1) \\
V(1) &= -1 + 0.45\,V(0) + 0.45\,V(2) \\
V(2) &= -1 + 0.45\,V(1) + 0.45 \times 0
\end{aligned}
$$

Three equations in three unknowns have an answer, and you can find it by substitution:

1. The third equation gives $V(2) = -1 + 0.45\,V(1)$.
2. Put that into the second: $V(1) = -1.45 + 0.45\,V(0) + 0.2025\,V(1)$, so $V(1) = -1.8182 + 0.5643\,V(0)$.
3. Put that into the first: $0.55\,V(0) = -1 + 0.45(-1.8182 + 0.5643\,V(0))$, so $0.2961\,V(0) = -1.8182$.
4. So $V(0) = -6.1408$, then $V(1) = -5.2832$ and $V(2) = -3.3774$.

Figure 1 compares these values with the always-right ones. Check one equation by plugging in: $-1 + 0.45 \times (-6.1408) + 0.45 \times (-3.3774) = -5.2832$, which is $V(1)$. Figure 2 draws that check.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Bars of state values for cells 0 to 2: always right gives minus 2.71, minus 1.9, minus 1; the uniform policy gives minus 6.14, minus 5.28, minus 3.38">
<line x1="70.0" y1="175.7" x2="560.0" y2="175.7" class="stroke-grid"/>
<text x="62.0" y="179.7" class="fig-small" text-anchor="end">-6</text>
<line x1="70.0" y1="127.1" x2="560.0" y2="127.1" class="stroke-grid"/>
<text x="62.0" y="131.1" class="fig-small" text-anchor="end">-4</text>
<line x1="70.0" y1="78.6" x2="560.0" y2="78.6" class="stroke-grid"/>
<text x="62.0" y="82.6" class="fig-small" text-anchor="end">-2</text>
<line x1="70.0" y1="30.0" x2="560.0" y2="30.0" class="stroke-grid"/>
<text x="62.0" y="34.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="30.0" x2="560.0" y2="30.0" class="stroke-muted"/>
<rect x="87.7" y="30.0" width="60.0" height="65.8" class="fill-b"/>
<text x="117.7" y="111.8" class="fig-small" text-anchor="middle">−2.71</text>
<rect x="155.7" y="30.0" width="60.0" height="149.1" class="fill-d"/>
<text x="185.7" y="195.1" class="fig-small" text-anchor="middle">−6.14</text>
<text x="151.7" y="22.0" class="fig-bold" text-anchor="middle">cell 0</text>
<rect x="251.0" y="30.0" width="60.0" height="46.1" class="fill-b"/>
<text x="281.0" y="92.1" class="fig-small" text-anchor="middle">−1.90</text>
<rect x="319.0" y="30.0" width="60.0" height="128.3" class="fill-d"/>
<text x="349.0" y="174.3" class="fig-small" text-anchor="middle">−5.28</text>
<text x="315.0" y="22.0" class="fig-bold" text-anchor="middle">cell 1</text>
<rect x="414.3" y="30.0" width="60.0" height="24.3" class="fill-b"/>
<text x="444.3" y="70.3" class="fig-small" text-anchor="middle">−1.00</text>
<rect x="482.3" y="30.0" width="60.0" height="82.0" class="fill-d"/>
<text x="512.3" y="128.0" class="fig-small" text-anchor="middle">−3.38</text>
<text x="478.3" y="22.0" class="fig-bold" text-anchor="middle">cell 2</text>
<rect x="80.0" y="246.0" width="12.0" height="12.0" class="fill-b"/>
<text x="98.0" y="256.0" class="fig-small" text-anchor="start">always right</text>
<rect x="200.0" y="246.0" width="12.0" height="12.0" class="fill-d"/>
<text x="218.0" y="256.0" class="fig-small" text-anchor="start">uniform random</text>
</svg>
Figure 1. Exact state values in the hallway with γ = 0.9. Always right is worth −2.71 from cell 0, because it walks straight to the exit. The uniform policy is worth −6.14, because it wastes steps wandering back toward the wall.
```

```figure
<svg viewBox="0 0 600 230" role="img" aria-label="A tree: the value of cell 1 equals minus one plus 0.9 times the average of the values of cells 0 and 2">
<defs><marker id="c009-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<rect x="220.0" y="20.0" width="160.0" height="50.0" class="fill-soft-d"/>
<rect x="220" y="20" width="160" height="50" class="stroke-d" fill="none"/>
<text x="300.0" y="42.0" class="fig-bold" text-anchor="middle">V(1) = −5.283</text>
<text x="300.0" y="60.0" class="fig-small" text-anchor="middle">cell 1, uniform policy</text>
<line x1="260.0" y1="72.0" x2="150.0" y2="128.0" class="stroke-b" marker-end="url(#c009-arrow)"/>
<line x1="340.0" y1="72.0" x2="450.0" y2="128.0" class="stroke-b" marker-end="url(#c009-arrow)"/>
<text x="176.0" y="94.0" class="fig-small" text-anchor="end">left, π = 0.5</text>
<text x="424.0" y="94.0" class="fig-small" text-anchor="start">right, π = 0.5</text>
<text x="176.0" y="110.0" class="fig-small" text-anchor="end">reward −1, land in cell 0</text>
<text x="424.0" y="110.0" class="fig-small" text-anchor="start">reward −1, land in cell 2</text>
<rect x="70.0" y="130.0" width="160.0" height="50.0" class="fill-soft-a"/>
<rect x="70" y="130" width="160" height="50" class="stroke-a" fill="none"/>
<text x="150.0" y="152.0" class="fig-bold" text-anchor="middle">V(0) = −6.141</text>
<text x="150.0" y="170.0" class="fig-small" text-anchor="middle">cell 0</text>
<rect x="370.0" y="130.0" width="160.0" height="50.0" class="fill-soft-a"/>
<rect x="370" y="130" width="160" height="50" class="stroke-a" fill="none"/>
<text x="450.0" y="152.0" class="fig-bold" text-anchor="middle">V(2) = −3.377</text>
<text x="450.0" y="170.0" class="fig-small" text-anchor="middle">cell 2</text>
<text x="300.0" y="214.0" class="fig-mono" text-anchor="middle">−1 + 0.9 × (0.5 × −6.141 + 0.5 × −3.377) = −5.283</text>
</svg>
Figure 2. One Bellman equation, drawn for cell 1. The value of a state is the reward you expect now plus γ times the average value of the cell you land in. Plugging the solved values of cells 0 and 2 back in returns the value of cell 1.
```

Does the number agree with simulation? Yes. Averaged over 2000 episodes of the uniform policy, the return from cell 0 is $-6.159$, and the exact value is $-6.141$ (Figure 3). The equation gives all the digits at once, and the average still wobbles in the second decimal.

```figure
<svg viewBox="0 0 600 275" role="img" aria-label="Line chart of a running average of returns settling near minus 6.14, with a horizontal line at the exact value">
<line x1="70.0" y1="210.0" x2="570.0" y2="210.0" class="stroke-grid"/>
<text x="62.0" y="214.0" class="fig-small" text-anchor="end">-9</text>
<line x1="70.0" y1="180.0" x2="570.0" y2="180.0" class="stroke-grid"/>
<text x="62.0" y="184.0" class="fig-small" text-anchor="end">-8</text>
<line x1="70.0" y1="150.0" x2="570.0" y2="150.0" class="stroke-grid"/>
<text x="62.0" y="154.0" class="fig-small" text-anchor="end">-7</text>
<line x1="70.0" y1="120.0" x2="570.0" y2="120.0" class="stroke-grid"/>
<text x="62.0" y="124.0" class="fig-small" text-anchor="end">-6</text>
<line x1="70.0" y1="90.0" x2="570.0" y2="90.0" class="stroke-grid"/>
<text x="62.0" y="94.0" class="fig-small" text-anchor="end">-5</text>
<line x1="70.0" y1="60.0" x2="570.0" y2="60.0" class="stroke-grid"/>
<text x="62.0" y="64.0" class="fig-small" text-anchor="end">-4</text>
<line x1="70.0" y1="30.0" x2="570.0" y2="30.0" class="stroke-grid"/>
<text x="62.0" y="34.0" class="fig-small" text-anchor="end">-3</text>
<line x1="70.0" y1="210.0" x2="570.0" y2="210.0" class="stroke-muted"/>
<line x1="70.0" y1="30.0" x2="70.0" y2="210.0" class="stroke-muted"/>
<line x1="70.0" y1="210.0" x2="70.0" y2="215.0" class="stroke-muted"/>
<text x="70.0" y="230.0" class="fig-small" text-anchor="middle">0</text>
<line x1="195.0" y1="210.0" x2="195.0" y2="215.0" class="stroke-muted"/>
<text x="195.0" y="230.0" class="fig-small" text-anchor="middle">500</text>
<line x1="320.0" y1="210.0" x2="320.0" y2="215.0" class="stroke-muted"/>
<text x="320.0" y="230.0" class="fig-small" text-anchor="middle">1000</text>
<line x1="445.0" y1="210.0" x2="445.0" y2="215.0" class="stroke-muted"/>
<text x="445.0" y="230.0" class="fig-small" text-anchor="middle">1500</text>
<line x1="570.0" y1="210.0" x2="570.0" y2="215.0" class="stroke-muted"/>
<text x="570.0" y="230.0" class="fig-small" text-anchor="middle">2000</text>
<text x="320.0" y="250.0" class="fig-small" text-anchor="middle">episodes averaged</text>
<text x="14" y="120.0" class="fig-small" text-anchor="middle" transform="rotate(-90 14 120.0)">average return from cell 0</text>
<line x1="70.0" y1="124.2" x2="570.0" y2="124.2" class="stroke-b"/>
<polyline points="70.2,62.9 70.5,42.1 70.8,54.9 71.0,65.3 71.2,93.0 71.5,109.1 71.8,111.2 72.0,102.7 72.2,113.5 72.5,118.5 72.8,117.8 73.0,114.7 73.2,113.3 73.5,112.1 73.8,112.9 74.0,107.2 74.2,109.4 74.5,106.8 74.8,103.5 75.0,109.9 75.2,105.6 75.5,106.5 75.8,102.8 76.0,99.4 76.2,100.3 76.5,98.9 76.8,102.1 77.0,100.7 77.2,101.0 77.5,101.8 77.8,101.6 78.0,101.4 78.2,104.1 78.5,101.7 78.8,103.7 79.0,104.6 79.2,106.7 79.5,107.5 79.8,106.3 80.0,104.2 80.2,104.4 82.8,102.6 85.2,104.7 87.8,108.1 90.2,107.8 92.8,115.7 95.2,117.4 97.8,116.2 100.2,116.9 102.8,119.6 105.2,121.5 107.8,121.8 110.2,120.8 112.8,118.6 115.2,118.9 117.8,117.5 120.2,118.1 122.8,117.4 125.2,117.2 127.8,117.2 130.2,117.8 132.8,117.9 135.2,118.8 137.8,119.5 140.2,120.9 142.8,121.1 145.2,121.5 147.8,121.2 150.2,120.5 152.8,120.9 155.2,120.9 157.8,120.9 160.2,121.5 162.8,122.2 165.2,122.1 167.8,120.0 170.2,120.4 172.8,120.0 175.2,120.7 177.8,120.6 180.2,121.1 182.8,121.3 185.2,121.5 187.8,122.6 190.2,122.7 192.8,123.6 195.2,123.1 197.8,123.3 200.2,123.2 202.8,123.0 205.2,122.7 207.8,122.8 210.2,122.8 212.8,122.6 215.2,122.5 217.8,122.2 220.2,121.9 222.8,121.5 225.2,121.4 227.8,121.3 230.2,121.7 232.8,121.5 235.2,121.1 237.8,121.3 240.2,121.4 242.8,121.7 245.2,121.6 247.8,121.2 250.2,121.5 252.8,121.5 255.2,121.4 257.8,121.8 260.2,122.1 262.8,122.3 265.2,121.9 267.8,121.6 270.2,121.0 272.8,120.7 275.2,120.5 277.8,120.4 280.2,120.5 282.8,121.0 285.2,121.1 287.8,121.4 290.2,121.4 292.8,121.3 295.2,121.3 297.8,121.1 300.2,121.5 302.8,121.8 305.2,121.5 307.8,121.6 310.2,121.9 312.8,122.0 315.2,121.9 317.8,121.8 320.2,122.0 322.8,121.7 325.2,121.9 327.8,121.8 330.2,122.0 332.8,122.3 335.2,122.4 337.8,122.6 340.2,122.5 342.8,122.6 345.2,122.6 347.8,122.3 350.2,122.3 352.8,122.4 355.2,122.0 357.8,122.3 360.2,122.4 362.8,122.2 365.2,122.3 367.8,122.6 370.2,123.0 372.8,122.8 375.2,123.0 377.8,123.1 380.2,123.2 382.8,122.9 385.2,122.7 387.8,122.7 390.2,122.9 392.8,123.1 395.2,123.2 397.8,123.2 400.2,123.3 402.8,123.4 405.2,123.3 407.8,123.4 410.2,123.2 412.8,123.1 415.2,123.3 417.8,123.3 420.2,123.5 422.8,123.5 425.2,123.5 427.8,123.7 430.2,123.7 432.8,123.8 435.2,123.7 437.8,123.7 440.2,124.1 442.8,124.1 445.2,123.9 447.8,123.9 450.2,124.0 452.8,124.1 455.2,124.2 457.8,124.2 460.2,124.3 462.8,124.4 465.2,124.2 467.8,124.1 470.2,124.1 472.8,124.1 475.2,124.2 477.8,124.1 480.2,124.4 482.8,124.5 485.2,124.4 487.8,124.4 490.2,124.5 492.8,124.6 495.2,124.6 497.8,124.7 500.2,124.8 502.8,124.5 505.2,124.5 507.8,124.5 510.2,124.2 512.8,124.2 515.2,124.3 517.8,124.2 520.2,124.1 522.8,124.2 525.2,124.3 527.8,124.3 530.2,124.5 532.8,124.5 535.2,124.6 537.8,124.8 540.2,124.9 542.8,124.9 545.2,124.9 547.8,124.7 550.2,124.9 552.8,125.1 555.2,124.9 557.8,124.9 560.2,125.0 562.8,124.8 565.2,124.6 567.8,124.6 570.0,124.8" class="stroke-a" fill="none"/>
<text x="566.0" y="116.2" class="fig-small" text-anchor="end">exact value −6.141</text>
<text x="566.0" y="144.8" class="fig-small" text-anchor="end">2000 episodes: −6.159</text>
</svg>
Figure 3. The average return of the uniform policy from cell 0, as you add episodes. It wanders for hundreds of episodes and still sits 0.02 from the exact value after 2000. The linear solve gets all digits at once.
```

## One linear solve

Substitution is how you solve three equations by hand. For $S$ states, stack the equations into one matrix equation. The vector $V$ holds every state's value, and $P_\pi V$ is the vector of "average value of the next state":

$$
V = r_\pi + \gamma P_\pi V \quad\Longleftrightarrow\quad (I - \gamma P_\pi)\, V = r_\pi
$$

Read it as: one linear system whose solution is the whole value vector. The left form says "value equals reward plus discounted next value". The right form moves the unknown to one side, so it has the shape $M x = b$ that `np.linalg.solve` handles:

```python
V = np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)
```

Terminal states stay in the system. Their rows of $P_\pi$ and $r_\pi$ are zero, so their equation reads $V(s) = 0$, which is what a terminal state is worth.

Why does a solution always exist? When $\gamma < 1$, the matrix $I - \gamma P_\pi$ is invertible. Multiplying a vector by $P_\pi$ averages its entries, and an average is never larger than the largest entry, so $\gamma P_\pi$ shrinks the largest absolute entry of every nonzero vector by at least a factor $\gamma$. No nonzero $V$ can then satisfy $V = \gamma P_\pi V$, which means the system has exactly one solution.

> [!KEY]
> A state's value is the reward you expect now plus the discounted value you expect next. With a known model, that is a linear system, and one call to `np.linalg.solve` gives every state's value exactly.

## From state values to action values

The value $V^\pi(s)$ averages over the policy's first action. The *action value* $Q^\pi(s, a)$ fixes the first action and follows $\pi$ afterward. It answers the question "what if I did $a$ here, just this once?"

$$
Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a)\, V^\pi(s')
$$

Read it as: the reward for $a$ in $s$, plus the discounted value of where $a$ takes you. In numpy, `mdp.P @ V` has shape `(S, A)`, and entry `[s, a]` is the expected value of the next state:

```python
Q = mdp.R + gamma * mdp.P @ V
Q[mdp.terminal] = 0.0
```

For the uniform policy in the hallway, $V = (-6.1408, -5.2832, -3.3774, 0)$ gives this table:

| Cell | $Q(s, \text{left})$ | $Q(s, \text{right})$ |
|---|---|---|
| 0 | $-1 + 0.9 \times (-6.1408) = -6.5267$ | $-1 + 0.9 \times (-5.2832) = -5.7549$ |
| 1 | $-6.5267$ | $-1 + 0.9 \times (-3.3774) = -4.0397$ |
| 2 | $-5.7549$ | $-1 + 0.9 \times 0 = -1$ |

Each row averages to its state value, because the uniform policy weighs both actions equally: $(-6.5267 - 5.7549)/2 = -6.1408$. In general, $V^\pi(s) = \sum_a \pi(a \mid s)\, Q^\pi(s, a)$. The gap $Q^\pi(s, a) - V^\pi(s)$ is the *advantage* $A^\pi(s, a)$. In cell 0, going right has an advantage of $-5.7549 + 6.1408 = 0.3859$: it beats the policy's average action by that much. The course uses advantages heavily from Part 4 on.

> [!WARNING]
> Zero the terminal rows of `Q` as well. Without that, a terminal state keeps collecting reward in later calculations and inflates the value of every state that can reach it.

> [!RECAP]
> - $V^\pi = r_\pi + \gamma P_\pi V^\pi$, because the return is the first reward plus the discounted rest.
> - `V = np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)`.
> - `Q = R + gamma * P @ V`, with terminal rows set to zero.
> - $V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s, a)$.

## Build it step by step

The starter gives you `hallway_mdp` and, in full, the `policy_matrices` function from the last chapter. The checks test your functions in this order.

1. **`evaluate_exact(mdp, pi)`.** Get `P_pi` and `r_pi` from `policy_matrices`, then solve one system. Returns `V` of shape `(S,)`. Common mistake: solving with `P_pi.T`, which answers a different question. The checks compare against the hand-computed hallway values, against `rlenvs.policy_evaluation`, and against the Bellman equation itself.
2. **`q_from_v(mdp, v)`.** `mdp.R + mdp.gamma * mdp.P @ v`, then zero the terminal rows. Returns `Q` of shape `(S, A)`. The checks use the table above and the identity $\sum_a \pi(a \mid s) Q(s, a) = V(s)$.

## Cost and scaling

Building $P_\pi$ costs $O(S^2 A)$, and the dense solve costs $O(S^3)$ time with $O(S^2)$ memory. A thousand states solve in a blink, and a million states cannot even store the matrix. That gap is why the next chapter replaces the solve with repeated sweeps, and why later chapters learn from samples.

## Where this breaks

**No discount and no guaranteed ending.** With $\gamma = 1$, the matrix $I - P_\pi$ is singular whenever the policy can loop forever without reaching a terminal state. The values of those states are infinite or undefined. `rlenvs.policy_evaluation` returns `nan` for them. Every exercise in this course uses $\gamma < 1$ or a policy that always ends.

**An unknown model.** The solve reads `P` and `R` directly. When you can only sample transitions, you fall back on averages like the ones in Figure 3.

## Further reading

- [Sutton and Barto, *Reinforcement Learning: An Introduction*, Sections 3.5 and 3.6](http://incompleteideas.net/book/the-book-2nd.html) define value functions and derive the Bellman equation.
- [Bellman, "A Markovian decision process" (1957)](http://www.iumj.indiana.edu/IUMJ/FULLTEXT/1957/6/56038) is the paper that gave the equation its name.
- [Puterman, *Markov Decision Processes: Discrete Stochastic Dynamic Programming* (1994)](https://www.wiley.com/en-us/Markov+Decision+Processes:+Discrete+Stochastic+Dynamic+Programming-p-9781118625873) proves that the linear system has a unique solution and covers the theory in full.

## The exercise

You solve for a policy's values exactly. Write `evaluate_exact(mdp, pi)` for the linear solve and `q_from_v(mdp, v)` for action values. The starter file gives `hallway_mdp` and `policy_matrices`, with the exercise from the last chapter solved for you.

The checks verify the always-right values $(-2.71, -1.9, -1, 0)$ and the uniform-policy values above. They compare `evaluate_exact` against `rlenvs.policy_evaluation` on random MDPs and on the frozen lake, and check that the values satisfy the Bellman equation. They check the action-value table, the identity between $Q$ and $V$, and that the exact value agrees with the average return of simulated episodes.
