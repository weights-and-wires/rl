---
title: Policy evaluation by sweeping
slug: 010-policy-evaluation
part: "Part 2 — Planning with a known world"
summary: Evaluate a policy by applying the Bellman equation over and over from a guess, and stop with a guaranteed error bound.
minutes: 30
gpu: false
objectives:
  - Apply one Bellman sweep to a value vector, and say what V_k means after k sweeps from zero.
  - Explain why each sweep shrinks the error by at least a factor of gamma.
  - Stop iterating with a guaranteed error, and say why a small last change is not enough.
  - Check that sweeping converges to the same answer as the linear solve.
prereqs:
  - 009-the-bellman-equation
exercise: 010-policy-evaluation
---

# Policy evaluation by sweeping

The linear solve from [The Bellman equation](/lessons/009-the-bellman-equation) is exact, but it has a price. A dense solve takes time proportional to $S^3$ and stores an $S \times S$ matrix. With a thousand states that is nothing. With a million states, you cannot even store the matrix. A tempting fix is to give up exactness and solve the equation approximately. The question is how to get a good answer without ever solving a system.

Look at the equation again: $V = r_\pi + \gamma P_\pi V$. The unknown $V$ appears on both sides. Suppose you guess $V$, plug the guess into the right side, and read off the left side. If the guess was already right, you get it back. If the guess was wrong, you get a different vector. Is that vector closer to the answer? This chapter shows that it is, and by a guaranteed amount. Repeating the step gives a way to find values that needs only multiplications, and it is the first of the *planning* algorithms in this part of the course.

## What you already know

- A policy's values satisfy $V^\pi = r_\pi + \gamma P_\pi V^\pi$, and the linear solve gives them exactly (see [The Bellman equation](/lessons/009-the-bellman-equation)).
- `policy_matrices` returns $P_\pi$ and $r_\pi$, with zero rows for terminal states (see [Markov decision processes](/lessons/008-markov-decision-processes)).
- The hallway MDP has four cells and a discount of $\gamma = 0.9$, and the uniform policy has values $(-6.141, -5.283, -3.377, 0)$.
- The largest absolute entry of a vector measures how far it is from zero.
- An average of numbers is never larger than the largest of them.

## One sweep

Start with a guess $V_0$, such as all zeros. Plug it into the right side of the Bellman equation, and call the result $V_1$. Then repeat:

$$
V_{k+1} = r_\pi + \gamma P_\pi V_k
$$

Read it as: each *sweep* replaces every state's value with its expected reward plus the discounted average of the current guesses at the next states. In numpy, it is one line, and a terminal state stays at 0 because its rows are zero:

```python
v_new = r_pi + gamma * P_pi @ v
```

Work it by hand on the hallway with the uniform policy, where each neighbor carries a weight of $0.45$, starting from $V_0 = (0, 0, 0, 0)$:

- Sweep 1: every non-terminal cell gets $-1 + 0.45 \times 0 + 0.45 \times 0 = -1$, so $V_1 = (-1, -1, -1, 0)$.
- Sweep 2: $V_2(0) = -1 + 0.45 \times (-1) + 0.45 \times (-1) = -1.9$. Cell 1 gets the same, $-1.9$. Cell 2 reaches the exit, which is worth 0, so $V_2(2) = -1 + 0.45 \times (-1) + 0 = -1.45$.
- Sweep 3: $V_3(0) = -1 + 0.45 \times (-1.9) + 0.45 \times (-1.9) = -2.71$, $V_3(1) = -1 + 0.45 \times (-1.9) + 0.45 \times (-1.45) = -2.5075$, and $V_3(2) = -1 + 0.45 \times (-1.9) = -1.855$.

Why does this move toward the answer? After $k$ sweeps from zero, $V_k(s)$ is the expected discounted reward of the *first $k$ steps*. Sweep 1 sees one step of reward, sweep 2 sees two, and every sweep adds one more step of lookahead. The part of the return beyond $k$ steps is discounted by $\gamma^k$, so as $k$ grows, the missing part shrinks and $V_k$ approaches $V^\pi$. Figure 1 shows the values creeping down toward the exact answers.

```figure
<svg viewBox="0 0 640 270" role="img" aria-label="Three curves of value estimates falling from zero toward dashed exact values of about minus 6.14, minus 5.28, and minus 3.38">
<line x1="70.0" y1="210.0" x2="570.0" y2="210.0" class="stroke-grid"/>
<text x="62.0" y="214.0" class="fig-small" text-anchor="end">-7</text>
<line x1="70.0" y1="186.0" x2="570.0" y2="186.0" class="stroke-grid"/>
<text x="62.0" y="190.0" class="fig-small" text-anchor="end">-6</text>
<line x1="70.0" y1="162.0" x2="570.0" y2="162.0" class="stroke-grid"/>
<text x="62.0" y="166.0" class="fig-small" text-anchor="end">-5</text>
<line x1="70.0" y1="138.0" x2="570.0" y2="138.0" class="stroke-grid"/>
<text x="62.0" y="142.0" class="fig-small" text-anchor="end">-4</text>
<line x1="70.0" y1="114.0" x2="570.0" y2="114.0" class="stroke-grid"/>
<text x="62.0" y="118.0" class="fig-small" text-anchor="end">-3</text>
<line x1="70.0" y1="90.0" x2="570.0" y2="90.0" class="stroke-grid"/>
<text x="62.0" y="94.0" class="fig-small" text-anchor="end">-2</text>
<line x1="70.0" y1="66.0" x2="570.0" y2="66.0" class="stroke-grid"/>
<text x="62.0" y="70.0" class="fig-small" text-anchor="end">-1</text>
<line x1="70.0" y1="42.0" x2="570.0" y2="42.0" class="stroke-grid"/>
<text x="62.0" y="46.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="210.0" x2="570.0" y2="210.0" class="stroke-muted"/>
<line x1="70.0" y1="30.0" x2="70.0" y2="210.0" class="stroke-muted"/>
<line x1="70.0" y1="210.0" x2="70.0" y2="215.0" class="stroke-muted"/>
<text x="70.0" y="230.0" class="fig-small" text-anchor="middle">0</text>
<line x1="153.3" y1="210.0" x2="153.3" y2="215.0" class="stroke-muted"/>
<text x="153.3" y="230.0" class="fig-small" text-anchor="middle">5</text>
<line x1="236.7" y1="210.0" x2="236.7" y2="215.0" class="stroke-muted"/>
<text x="236.7" y="230.0" class="fig-small" text-anchor="middle">10</text>
<line x1="320.0" y1="210.0" x2="320.0" y2="215.0" class="stroke-muted"/>
<text x="320.0" y="230.0" class="fig-small" text-anchor="middle">15</text>
<line x1="403.3" y1="210.0" x2="403.3" y2="215.0" class="stroke-muted"/>
<text x="403.3" y="230.0" class="fig-small" text-anchor="middle">20</text>
<line x1="486.7" y1="210.0" x2="486.7" y2="215.0" class="stroke-muted"/>
<text x="486.7" y="230.0" class="fig-small" text-anchor="middle">25</text>
<line x1="570.0" y1="210.0" x2="570.0" y2="215.0" class="stroke-muted"/>
<text x="570.0" y="230.0" class="fig-small" text-anchor="middle">30</text>
<text x="320.0" y="250.0" class="fig-small" text-anchor="middle">sweeps k</text>
<text x="14" y="120.0" class="fig-small" text-anchor="middle" transform="rotate(-90 14 120.0)">V_k(s)</text>
<line x1="70.0" y1="189.4" x2="570.0" y2="189.4" class="stroke-muted dashed"/>
<polyline points="70.0,42.0 86.7,66.0 103.3,87.6 120.0,107.0 136.7,122.3 153.3,135.1 170.0,145.3 186.7,153.7 203.3,160.4 220.0,165.9 236.7,170.3 253.3,173.9 270.0,176.9 286.7,179.2 303.3,181.2 320.0,182.7 336.7,184.0 353.3,185.0 370.0,185.8 386.7,186.5 403.3,187.0 420.0,187.5 436.7,187.8 453.3,188.1 470.0,188.4 486.7,188.6 503.3,188.7 520.0,188.8 536.7,188.9 553.3,189.0 570.0,189.1" class="stroke-a" fill="none"/>
<text x="574.0" y="193.1" class="fig-small" text-anchor="start">cell 0</text>
<line x1="70.0" y1="168.8" x2="570.0" y2="168.8" class="stroke-muted dashed"/>
<polyline points="70.0,42.0 86.7,66.0 103.3,87.6 120.0,102.2 136.7,115.3 153.3,125.1 170.0,133.6 186.7,140.1 203.3,145.6 220.0,150.0 236.7,153.5 253.3,156.4 270.0,158.8 286.7,160.7 303.3,162.2 320.0,163.4 336.7,164.5 353.3,165.3 370.0,165.9 386.7,166.5 403.3,166.9 420.0,167.3 436.7,167.6 453.3,167.8 470.0,168.0 486.7,168.1 503.3,168.3 520.0,168.4 536.7,168.4 553.3,168.5 570.0,168.6" class="stroke-b" fill="none"/>
<text x="574.0" y="172.6" class="fig-small" text-anchor="start">cell 1</text>
<line x1="70.0" y1="123.1" x2="570.0" y2="123.1" class="stroke-muted dashed"/>
<polyline points="70.0,42.0 86.7,66.0 103.3,76.8 120.0,86.5 136.7,93.1 153.3,99.0 170.0,103.4 186.7,107.2 203.3,110.2 220.0,112.6 236.7,114.6 253.3,116.2 270.0,117.5 286.7,118.5 303.3,119.4 320.0,120.1 336.7,120.7 353.3,121.1 370.0,121.5 386.7,121.8 403.3,122.0 420.0,122.2 436.7,122.4 453.3,122.5 470.0,122.6 486.7,122.7 503.3,122.8 520.0,122.8 536.7,122.9 553.3,122.9 570.0,122.9" class="stroke-c" fill="none"/>
<text x="574.0" y="126.9" class="fig-small" text-anchor="start">cell 2</text>
</svg>
Figure 1. Values after k sweeps from V = 0 in the hallway with the uniform policy. The first three sweeps match the numbers worked out by hand, and every cell creeps toward its exact value (dashed lines, from the linear solve) without overshooting it.
```

For the always-right policy, the process finishes exactly. Its first three sweeps give $-1$, then $-1.9$, then $-2.71$ at cell 0, and the fourth sweep changes nothing. A policy that always reaches the exit within $n$ steps is fully evaluated after $n$ sweeps.

This is a *synchronous* sweep: every new entry comes from the old vector. You can also update entries in place, so later states in a sweep see earlier states' new values. That also converges, often faster, but its iterates differ, and the bounds below are for the synchronous version.

## Why sweeping converges

Subtract the exact equation $V^\pi = r_\pi + \gamma P_\pi V^\pi$ from the sweep $V_{k+1} = r_\pi + \gamma P_\pi V_k$. The rewards cancel, and what remains is:

$$
V_{k+1} - V^\pi = \gamma P_\pi\,(V_k - V^\pi)
$$

Read it as: each sweep multiplies the old error by $\gamma P_\pi$. Multiplying by $P_\pi$ averages the entries of the error vector, and an average is never larger than the largest entry. So, writing $\lVert x \rVert$ for the largest absolute entry of $x$:

$$
\lVert V_{k+1} - V^\pi \rVert \le \gamma\, \lVert V_k - V^\pi \rVert
$$

Every sweep cuts the worst error across states by at least a factor of $\gamma$. After $k$ sweeps from zero, the error is at most $\gamma^k \lVert V^\pi \rVert$. In the hallway, that is $0.9^k \times 6.14$. Figure 2 compares the guarantee with the actual error.

```figure
<svg viewBox="0 0 640 270" role="img" aria-label="Log-scale error against sweeps: the actual error falls faster than the straight line that bounds it">
<line x1="70.0" y1="210.0" x2="570.0" y2="210.0" class="stroke-grid"/>
<text x="62.0" y="214.0" class="fig-small" text-anchor="end">0.001</text>
<line x1="70.0" y1="165.0" x2="570.0" y2="165.0" class="stroke-grid"/>
<text x="62.0" y="169.0" class="fig-small" text-anchor="end">0.01</text>
<line x1="70.0" y1="120.0" x2="570.0" y2="120.0" class="stroke-grid"/>
<text x="62.0" y="124.0" class="fig-small" text-anchor="end">0.1</text>
<line x1="70.0" y1="75.0" x2="570.0" y2="75.0" class="stroke-grid"/>
<text x="62.0" y="79.0" class="fig-small" text-anchor="end">1</text>
<line x1="70.0" y1="30.0" x2="570.0" y2="30.0" class="stroke-grid"/>
<text x="62.0" y="34.0" class="fig-small" text-anchor="end">10</text>
<line x1="70.0" y1="210.0" x2="570.0" y2="210.0" class="stroke-muted"/>
<line x1="70.0" y1="30.0" x2="70.0" y2="210.0" class="stroke-muted"/>
<line x1="70.0" y1="210.0" x2="70.0" y2="215.0" class="stroke-muted"/>
<text x="70.0" y="230.0" class="fig-small" text-anchor="middle">0</text>
<line x1="195.0" y1="210.0" x2="195.0" y2="215.0" class="stroke-muted"/>
<text x="195.0" y="230.0" class="fig-small" text-anchor="middle">10</text>
<line x1="320.0" y1="210.0" x2="320.0" y2="215.0" class="stroke-muted"/>
<text x="320.0" y="230.0" class="fig-small" text-anchor="middle">20</text>
<line x1="445.0" y1="210.0" x2="445.0" y2="215.0" class="stroke-muted"/>
<text x="445.0" y="230.0" class="fig-small" text-anchor="middle">30</text>
<line x1="570.0" y1="210.0" x2="570.0" y2="215.0" class="stroke-muted"/>
<text x="570.0" y="230.0" class="fig-small" text-anchor="middle">40</text>
<text x="320.0" y="250.0" class="fig-small" text-anchor="middle">sweeps k</text>
<text x="14" y="120.0" class="fig-small" text-anchor="middle" transform="rotate(-90 14 120.0)">largest error over cells</text>
<polyline points="70.0,39.5 82.5,41.6 95.0,43.6 107.5,45.7 120.0,47.8 132.5,49.8 145.0,51.9 157.5,53.9 170.0,56.0 182.5,58.1 195.0,60.1 207.5,62.2 220.0,64.2 232.5,66.3 245.0,68.4 257.5,70.4 270.0,72.5 282.5,74.5 295.0,76.6 307.5,78.7 320.0,80.7 332.5,82.8 345.0,84.8 357.5,86.9 370.0,88.9 382.5,91.0 395.0,93.1 407.5,95.1 420.0,97.2 432.5,99.2 445.0,101.3 457.5,103.4 470.0,105.4 482.5,107.5 495.0,109.5 507.5,111.6 520.0,113.7 532.5,115.7 545.0,117.8 557.5,119.8 570.0,121.9" class="stroke-b" fill="none"/>
<polyline points="70.0,39.5 82.5,43.0 95.0,46.8 107.5,50.9 120.0,54.9 132.5,59.1 145.0,63.1 157.5,67.3 170.0,71.3 182.5,75.4 195.0,79.5 207.5,83.6 220.0,87.7 232.5,91.8 245.0,95.9 257.5,100.0 270.0,104.1 282.5,108.2 295.0,112.3 307.5,116.4 320.0,120.5 332.5,124.6 345.0,128.7 357.5,132.8 370.0,136.9 382.5,141.0 395.0,145.1 407.5,149.2 420.0,153.3 432.5,157.4 445.0,161.5 457.5,165.6 470.0,169.7 482.5,173.8 495.0,177.9 507.5,182.0 520.0,186.1 532.5,190.2 545.0,194.3 557.5,198.3 570.0,202.4" class="stroke-a" fill="none"/>
<circle cx="70.0" cy="39.5" r="2.5" class="fill-a"/>
<circle cx="132.5" cy="59.1" r="2.5" class="fill-a"/>
<circle cx="195.0" cy="79.5" r="2.5" class="fill-a"/>
<circle cx="257.5" cy="100.0" r="2.5" class="fill-a"/>
<circle cx="320.0" cy="120.5" r="2.5" class="fill-a"/>
<circle cx="382.5" cy="141.0" r="2.5" class="fill-a"/>
<circle cx="445.0" cy="161.5" r="2.5" class="fill-a"/>
<circle cx="507.5" cy="182.0" r="2.5" class="fill-a"/>
<circle cx="570.0" cy="202.4" r="2.5" class="fill-a"/>
<line x1="82.5" y1="140.2" x2="120.0" y2="140.2" class="stroke-b"/>
<text x="132.5" y="144.2" class="fig-small" text-anchor="start">guarantee: 6.14 × 0.9^k</text>
<line x1="82.5" y1="162.8" x2="120.0" y2="162.8" class="stroke-a"/>
<text x="132.5" y="166.8" class="fig-small" text-anchor="start">actual error, 0.0015 at k = 40</text>
</svg>
Figure 2. The largest error over cells after k sweeps, on a log scale. Every sweep shrinks it by at least a factor of γ = 0.9, which is the straight orange line. Here it shrinks by about 0.81 per sweep, so the guarantee is safe but loose.
```

The actual error shrinks faster, by about 0.81 per sweep, because every walk that reaches the exit lands on a state whose error is always zero. The factor $\gamma$ is a guarantee that holds for every MDP, so it must also cover the worst one, where nothing leaks away.

> [!KEY]
> Applying the Bellman equation as an update shrinks the error by at least a factor of $\gamma$ per sweep, so iterating from any starting guess converges to the policy's exact values.

## When to stop

A real loop cannot run forever, so it stops when one sweep changes no value by more than a threshold `tol`. Does a small last change mean the answer is within `tol` of $V^\pi$? It does not, and the contraction tells you how far off you can be. The error of the latest sweep is at most $\gamma$ times the previous error, and the previous error is at most the last change plus the latest error:

$$
\lVert V_{k+1} - V^\pi \rVert \le \gamma\,\bigl(\lVert V_{k+1} - V_k \rVert + \lVert V_{k+1} - V^\pi \rVert\bigr)
$$

Move the error terms to one side and divide by $1 - \gamma$:

$$
\lVert V_{k+1} - V^\pi \rVert \le \frac{\gamma}{1 - \gamma}\, \lVert V_{k+1} - V_k \rVert
$$

Read it as: the true error is at most $\gamma / (1 - \gamma)$ times the size of the last change. With $\gamma = 0.9$, that factor is 9. With $\gamma = 0.95$, it is 19.

In the hallway with the uniform policy, stopping at `tol = 1e-3` takes 35 sweeps. The last change is $9.8 \times 10^{-4}$ and the true error is $4.2 \times 10^{-3}$, about four times `tol` but under the bound of $8.8 \times 10^{-3}$. On a random 20-state MDP with 4 actions and $\gamma = 0.95$, stopping at `tol = 1e-8` takes 332 sweeps, with a true error of $1.82 \times 10^{-7}$ against a bound of $1.90 \times 10^{-7}$.

```python
v = np.zeros(S)
while True:
    new = r_pi + gamma * P_pi @ v
    if np.max(np.abs(new - v)) < tol:
        break
    v = new
```

> [!WARNING]
> A small last change does not mean a small error. To guarantee an error of $\varepsilon$, stop when the last change drops below $\varepsilon (1 - \gamma) / \gamma$.

> [!RECAP]
> - One sweep is `r_pi + gamma * P_pi @ v`, computed from the old vector for every state.
> - After $k$ sweeps from zero, $V_k$ is the expected reward of the first $k$ steps.
> - Each sweep shrinks the largest error by at least a factor of $\gamma$.
> - The true error is at most $\gamma / (1 - \gamma)$ times the last change.

## Build it step by step

The starter gives you `hallway_mdp` and `policy_matrices` in full. For comparisons, `rlenvs.policy_evaluation(mdp, pi)` returns the exact values from the linear solve. The checks test your functions in this order.

1. **`bellman_sweep(mdp, pi, v)`.** Takes values `v` of shape `(S,)` and returns `r_pi + gamma * P_pi @ v`, a new array of shape `(S,)`. Common mistake: updating `v` in place entry by entry, which makes later states see the new values.
2. **`iterative_policy_evaluation(mdp, pi, tol)`.** Starts from zeros and calls `bellman_sweep` until the largest change drops below `tol`. Returns `(v, sweeps)`, where `v` is the last sweep and `sweeps` counts the sweeps performed. Common mistake: returning the vector from before the last sweep.
3. **`stopping_bound(gamma, last_change)`.** Returns the error guarantee $\gamma / (1 - \gamma)$ times the last change. The checks compare it with the true errors that your loop leaves behind.

## Cost and scaling

A dense sweep costs $O(S^2)$ time, since it multiplies a matrix by a vector, and the number of sweeps to reach error $\varepsilon$ is about $\frac{1}{1-\gamma}\ln\frac{\text{initial error}}{\varepsilon}$. The solve costs $O(S^3)$ once. Sweeping wins when $P_\pi$ is sparse, because a sweep then costs only the number of nonzero transitions, and when you accept a small error.

## Where this breaks

**A discount close to 1.** The sweep count grows like $1 / (1 - \gamma)$. At $\gamma = 0.999$, you need thousands of sweeps. With $\gamma = 1$, the contraction argument fails, and sweeping converges only for policies that are sure to reach a terminal state.

**No model.** Both the solve and the sweep read `P` and `R`. When you can only sample transitions, later chapters replace the expectation in each sweep with sampled experience.

## Further reading

- [Sutton and Barto, *Reinforcement Learning: An Introduction*, Section 4.1](http://incompleteideas.net/book/the-book-2nd.html) introduces iterative policy evaluation with the grid-world example.
- [Puterman, *Markov Decision Processes: Discrete Stochastic Dynamic Programming* (1994)](https://www.wiley.com/en-us/Markov+Decision+Processes:+Discrete+Stochastic+Dynamic+Programming-p-9781118625873) proves convergence and error bounds for successive approximation.
- [Bellman, *Dynamic Programming* (1957)](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) is the book that named the family of methods.

## The exercise

You evaluate a policy by sweeping. Write `bellman_sweep(mdp, pi, v)`, `iterative_policy_evaluation(mdp, pi, tol)`, and `stopping_bound(gamma, last_change)`. The starter file gives `hallway_mdp` and `policy_matrices`.

The checks verify that `bellman_sweep` reproduces the three hand sweeps above, keeps terminal values at 0, leaves its input alone, and returns the exact values unchanged. They check that the always-right policy is fully evaluated after four sweeps, that the uniform policy lands within the stopping bound of the exact values on the hallway and on a random 20-state MDP, and that a smaller `tol` takes more sweeps. Last, they check that the true error never exceeds `stopping_bound` for three values of `tol`.
