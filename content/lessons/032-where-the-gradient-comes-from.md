---
title: The policy gradient theorem
slug: 032-where-the-gradient-comes-from
part: "Part 5 — Policy gradients"
summary: Write the exact gradient of a policy's return as visitation times probability times advantage, and verify it against finite differences.
minutes: 40
gpu: false
objectives:
  - Compute the discounted state-visitation distribution of a policy by solving one linear system.
  - State the policy gradient theorem and explain why you do not need the gradient of the visitation.
  - Simplify the theorem for a tabular softmax policy to visitation times probability times advantage.
  - Verify the exact gradient against finite differences of the expected return on random MDPs.
prereqs:
  - 009-the-bellman-equation
  - 031-the-policy-gradient
exercise: 032-where-the-gradient-comes-from
---

# The policy gradient theorem

The last chapter gave you REINFORCE, an estimator that weights each action's score by the return that followed. It works, but you took the formula on trust. Why a return, and why a $\gamma^t$? Is there a formula for the true gradient of the expected return, one you could compute exactly on a small problem and compare against?

There is, and it looks like it should not exist. The return depends on your policy in two ways. A change to the policy changes which action you pick in a state, and it also changes which states you reach. To differentiate the return you seem to need the derivative of the *distribution of states you visit*, and that depends on the policy in a complicated way through the whole world.

The policy gradient theorem says you do not need it. The gradient is a sum over states, weighted by how often you visit them, of how much each action beats the average. This chapter builds that weighting, states the theorem, and verifies it exactly on a small MDP.

## What you already know

- $V^\pi(s)$ is the expected return from $s$, and it solves the linear system $(I - \gamma P_\pi) V = r_\pi$ from the chapter on the Bellman equation.
- $Q^\pi(s, a) = r + \gamma \sum_{s'} P(s' \mid s, a) V^\pi(s')$ is the expected return after taking $a$ in $s$, and $V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s, a)$.
- The *advantage* $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ says how much better $a$ is than the policy's average action in $s$.
- The softmax policy's score is `onehot(a) - pi[s]`, and it averages to zero over actions.
- $J(\theta) = \sum_s \mu_0(s) V^{\pi_\theta}(s)$ is the expected return from the start distribution $\mu_0$.

## Where a policy spends its time

A change to the policy in state $s$ matters only when you are in $s$. So the first thing the gradient needs is a count of how often you are there. Count visits with a discount, because the objective discounts later rewards. The *discounted visitation* of $\pi$ from $\mu_0$ is

$$
d^\pi(s) = \sum_{t=0}^{\infty} \gamma^t \Pr(s_t = s)
$$

Read it as: add up the probability of being in $s$ at every time step, weighting step $t$ by $\gamma^t$. It is the expected discounted number of visits to $s$. With no terminal states, the weights $\gamma^t$ sum to $1 / (1 - \gamma)$, so $d^\pi$ sums to $1 / (1 - \gamma)$ as well.

You do not need to sum the series. A visit to $s$ comes from starting there, or from flowing in from a neighbor one step earlier, which costs a factor of $\gamma$. That gives the flow equation $d^\pi = \mu_0 + \gamma P_\pi^\top d^\pi$, where $P_\pi[s, s'] = \sum_a \pi(a \mid s) P(s' \mid s, a)$. Rearranged, it is one linear solve, the same matrix as policy evaluation but transposed:

$$
d^\pi = (I - \gamma P_\pi^\top)^{-1} \mu_0
$$

```python
P_pi = np.einsum("sa,sat->st", pi, P)
d = np.linalg.solve((np.eye(S) - gamma * P_pi).T, mu0)
```

### Work an example

Use the two-state MDP from the last chapter: *stay* or *switch*, and staying in state 1 pays 1. Start in state 0 with $\gamma = 0.9$ under the uniform policy. At $t = 0$ you are in state 0 for sure. At every later step you are in each state with probability $\tfrac{1}{2}$, because each action from either state is equally likely to leave you in either state. So

$$
d(0) = 1 + 0.9 \times 0.5 + 0.81 \times 0.5 + \cdots = 1 + 0.5 \times \frac{0.9}{0.1} = 5.5, \qquad d(1) = 0.5 \times \frac{0.9}{0.1} = 4.5
$$

Figure 1 draws the terms. The two totals sum to $10 = 1 / (1 - 0.9)$. The flow equation checks out: $d(0) = 1 + 0.9 \times (0.5 \times 5.5 + 0.5 \times 4.5) = 5.5$.

```figure
<svg viewBox="0 0 620 310" role="img" aria-label="Stacked bars of the discounted probability of being in state 0 and state 1 at each time step, which add up to visitations 5.5 and 4.5">
<line x1="70.0" y1="230.0" x2="550.0" y2="230.0" class="stroke-grid"/>
<text x="62.0" y="234.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="180.0" x2="550.0" y2="180.0" class="stroke-grid"/>
<text x="62.0" y="184.0" class="fig-small" text-anchor="end">0.25</text>
<line x1="70.0" y1="130.0" x2="550.0" y2="130.0" class="stroke-grid"/>
<text x="62.0" y="134.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="70.0" y1="80.0" x2="550.0" y2="80.0" class="stroke-grid"/>
<text x="62.0" y="84.0" class="fig-small" text-anchor="end">0.75</text>
<line x1="70.0" y1="30.0" x2="550.0" y2="30.0" class="stroke-grid"/>
<text x="62.0" y="34.0" class="fig-small" text-anchor="end">1</text>
<line x1="70.0" y1="230.0" x2="550.0" y2="230.0" class="stroke-muted"/>
<rect x="76.0" y="30.0" width="28.0" height="200.0" class="fill-a"/>
<text x="90.0" y="248.0" class="fig-small" text-anchor="middle">0</text>
<rect x="116.0" y="140.0" width="28.0" height="90.0" class="fill-a"/>
<rect x="116.0" y="50.0" width="28.0" height="90.0" class="fill-b"/>
<text x="130.0" y="248.0" class="fig-small" text-anchor="middle">1</text>
<rect x="156.0" y="149.0" width="28.0" height="81.0" class="fill-a"/>
<rect x="156.0" y="68.0" width="28.0" height="81.0" class="fill-b"/>
<text x="170.0" y="248.0" class="fig-small" text-anchor="middle">2</text>
<rect x="196.0" y="157.1" width="28.0" height="72.9" class="fill-a"/>
<rect x="196.0" y="84.2" width="28.0" height="72.9" class="fill-b"/>
<text x="210.0" y="248.0" class="fig-small" text-anchor="middle">3</text>
<rect x="236.0" y="164.4" width="28.0" height="65.6" class="fill-a"/>
<rect x="236.0" y="98.8" width="28.0" height="65.6" class="fill-b"/>
<text x="250.0" y="248.0" class="fig-small" text-anchor="middle">4</text>
<rect x="276.0" y="171.0" width="28.0" height="59.0" class="fill-a"/>
<rect x="276.0" y="111.9" width="28.0" height="59.0" class="fill-b"/>
<text x="290.0" y="248.0" class="fig-small" text-anchor="middle">5</text>
<rect x="316.0" y="176.9" width="28.0" height="53.1" class="fill-a"/>
<rect x="316.0" y="123.7" width="28.0" height="53.1" class="fill-b"/>
<text x="330.0" y="248.0" class="fig-small" text-anchor="middle">6</text>
<rect x="356.0" y="182.2" width="28.0" height="47.8" class="fill-a"/>
<rect x="356.0" y="134.3" width="28.0" height="47.8" class="fill-b"/>
<text x="370.0" y="248.0" class="fig-small" text-anchor="middle">7</text>
<rect x="396.0" y="187.0" width="28.0" height="43.0" class="fill-a"/>
<rect x="396.0" y="143.9" width="28.0" height="43.0" class="fill-b"/>
<text x="410.0" y="248.0" class="fig-small" text-anchor="middle">8</text>
<rect x="436.0" y="191.3" width="28.0" height="38.7" class="fill-a"/>
<rect x="436.0" y="152.5" width="28.0" height="38.7" class="fill-b"/>
<text x="450.0" y="248.0" class="fig-small" text-anchor="middle">9</text>
<rect x="476.0" y="195.1" width="28.0" height="34.9" class="fill-a"/>
<rect x="476.0" y="160.3" width="28.0" height="34.9" class="fill-b"/>
<text x="490.0" y="248.0" class="fig-small" text-anchor="middle">10</text>
<rect x="516.0" y="198.6" width="28.0" height="31.4" class="fill-a"/>
<rect x="516.0" y="167.2" width="28.0" height="31.4" class="fill-b"/>
<text x="530.0" y="248.0" class="fig-small" text-anchor="middle">11</text>
<text x="310.0" y="270.0" text-anchor="middle">time step t (first 12 steps shown)</text>
<text x="62.0" y="20.0" text-anchor="start">gamma^t Pr(s_t = s)</text>
<rect x="400.0" y="36.0" width="14.0" height="14.0" class="fill-a"/>
<text x="420.0" y="48.0" class="fig-small" text-anchor="start">state 0: sums to d(0) = 5.5</text>
<rect x="400.0" y="58.0" width="14.0" height="14.0" class="fill-b"/>
<text x="420.0" y="70.0" class="fig-small" text-anchor="start">state 1: sums to d(1) = 4.5</text>
</svg>
Figure 1. Each bar is the discounted chance of being in each state at one time step. The first bar is all state 0 because that is where you start, and every later bar splits evenly. Added over all steps, the blue parts total 5.5 and the orange parts total 4.5.
```

The visitation also rewrites the objective as an average over states: $J = \sum_s d^\pi(s)\, r_\pi(s)$, where $r_\pi(s) = \sum_a \pi(a \mid s) r(s, a)$ is the expected reward in $s$. Here $r_\pi = [0, 0.5]$, so $J = 5.5 \times 0 + 4.5 \times 0.5 = 2.25$, which matches the last chapter.

## State the theorem

Put the pieces together. Raising the logit of action $a$ in state $s$ changes the policy's probabilities in $s$. That change is worth something in proportion to how often you are in $s$, which is $d^\pi(s)$, and in proportion to how good each action is, which is $Q^\pi(s, a)$. Sutton, McAllester, Singh, and Mansour (2000) proved that this is all there is:

$$
\nabla_\theta J(\theta) = \sum_s d^\pi(s) \sum_a Q^\pi(s, a)\, \nabla_\theta \pi_\theta(a \mid s) = \sum_s d^\pi(s) \sum_a \pi_\theta(a \mid s)\, Q^\pi(s, a)\, \nabla_\theta \log \pi_\theta(a \mid s)
$$

The two forms are the same, because $\nabla \pi = \pi \nabla \log \pi$. Read the second one from the inside out:

- $Q^\pi(s, a)\, \nabla \log \pi(a \mid s)$ pushes on action $a$ in proportion to how good it is.
- The sum over $a$, weighted by $\pi$, averages that push over the actions the policy takes.
- The outer sum weights each state by $d^\pi(s)$, so states you visit often and early count most.

Why can you ignore how $d^\pi$ itself changes? The idea is an unrolling argument in three steps. Start from $V(s) = \sum_a \pi(a \mid s) Q(s, a)$ and apply the product rule:

1. One term differentiates $\pi$ and holds $Q$ fixed. That is the term in the theorem.
2. The other term differentiates $Q$. The reward does not depend on $\theta$, so this is $\gamma$ times the average of $\nabla V$ at the next state.
3. Unrolling that second term visits every future state $s_t$ with weight $\gamma^t$. Adding those weights over the start distribution gives exactly $d^\pi$.

The effect of the policy on which states you reach is not lost. It is already inside the term that differentiates $Q$, and the unrolling turns it into more copies of the first term at later states.

## Collapse it for a softmax policy

For the tabular softmax, put the score into the theorem. Only row $s$ of the score is nonzero, and the sum over actions $a'$ collapses:

$$
\sum_{a'} \pi(a' \mid s)\, Q(s, a')\, \big(\mathbb{1}[a' = a] - \pi(a \mid s)\big) = \pi(a \mid s) \big(Q(s, a) - V(s)\big)
$$

The second half of the left side is $\pi(a \mid s)\sum_{a'} \pi(a' \mid s) Q(s, a') = \pi(a \mid s) V(s)$. So

$$
\frac{\partial J}{\partial \theta_{s,a}} = d^\pi(s)\; \pi_\theta(a \mid s)\; A^\pi(s, a)
$$

Read it as: the logit for $a$ in $s$ should rise when $a$ beats the state's average, $A^\pi(s, a) > 0$. It rises faster when you visit $s$ often and when $a$ is already likely. Subtracting $V(s)$ cost nothing, because the score averages to zero, which is why the advantage appears instead of $Q$. Every piece is one call to `rlenvs`:

```python
v = rlenvs.policy_evaluation(mdp, pi)
adv = rlenvs.q_from_v(mdp, v) - v[:, None]
grad = d[:, None] * pi * adv        # d from the visitation solve above
```

> [!WARNING]
> `rlenvs.state_distribution` returns the visitation *normalized* to sum to one. The theorem needs the unnormalized $d^\pi$, which sums to $1 / (1 - \gamma)$ with no terminal states. Forget the factor and your gradient is too small by a factor of 10 when $\gamma = 0.9$. This chapter's `discounted_visitation` skips the normalization.

### Work an example

Take the two-state MDP with $\theta = 0$, so $\pi = 0.5$ everywhere. The following table shows each quantity:

| Quantity | State 0 | State 1 |
|---|---|---|
| $V^\pi(s)$ | 2.25 | 2.75 |
| $Q^\pi(s, \text{stay})$ | $0 + 0.9 \times 2.25 = 2.025$ | $1 + 0.9 \times 2.75 = 3.475$ |
| $Q^\pi(s, \text{switch})$ | $0.9 \times 2.75 = 2.475$ | $0.9 \times 2.25 = 2.025$ |
| $A^\pi(s, \text{stay})$ | $-0.225$ | $+0.725$ |
| $d^\pi(s)$ | 5.5 | 4.5 |
| $\partial J / \partial \theta_{s, \text{stay}}$ | $5.5 \times 0.5 \times (-0.225) = -0.61875$ | $4.5 \times 0.5 \times 0.725 = +1.63125$ |

The switch column is the negative of the stay column, because each state's advantages average to zero under $\pi$. You can check $V$ from the Bellman equation: $V(0) = 0.45 V(0) + 0.45 V(1)$ and $V(1) = 0.5 + 0.45 V(0) + 0.45 V(1)$. These are the same numbers a finite-difference measurement gave in the last chapter, $-0.619$ and $+1.631$. Figure 2 shows what the gradient says: in state 0, prefer switching, because it leads to the state that pays. In state 1, prefer staying, and push harder there, because the advantage is larger.

```figure
<svg viewBox="0 0 620 300" role="img" aria-label="Two-state stay or switch MDP with, for each state, the visitation, probability, advantage, and the resulting gradient on the stay logit">
<defs><marker id="c030-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<circle cx="150.0" cy="110.0" r="34" class="fill-soft-a stroke-a"/>
<text x="150.0" y="115.0" class="fig-bold" text-anchor="middle">state 0</text>
<circle cx="470.0" cy="110.0" r="34" class="fill-soft-a stroke-a"/>
<text x="470.0" y="115.0" class="fig-bold" text-anchor="middle">state 1</text>
<path d="M136,78 C110,30 190,30 164,78" class="stroke-muted" fill="none" marker-end="url(#c030-arrow)"/>
<text x="150.0" y="20.0" class="fig-small" text-anchor="middle">stay: 0</text>
<path d="M456,78 C430,30 510,30 484,78" class="stroke-muted" fill="none" marker-end="url(#c030-arrow)"/>
<text x="470.0" y="20.0" class="fig-small" text-anchor="middle">stay: +1</text>
<line x1="186.0" y1="102.0" x2="434.0" y2="102.0" class="stroke-muted" marker-end="url(#c030-arrow)"/>
<line x1="434.0" y1="120.0" x2="186.0" y2="120.0" class="stroke-muted" marker-end="url(#c030-arrow)"/>
<text x="310.0" y="96.0" class="fig-small" text-anchor="middle">switch: 0</text>
<text x="310.0" y="138.0" class="fig-small" text-anchor="middle">switch: 0</text>
<text x="150.0" y="190.0" class="fig-small" text-anchor="middle">visits d = 5.5</text>
<text x="150.0" y="210.0" class="fig-small" text-anchor="middle">pi(stay) = 0.5</text>
<text x="150.0" y="230.0" class="fig-small" text-anchor="middle">advantage of stay = -0.225</text>
<text x="150.0" y="258.0" class="fig-small fig-bold" text-anchor="middle">dJ / d theta(stay) = -0.61875</text>
<text x="150.0" y="280.0" class="fig-bold" text-anchor="middle">lower the stay logit</text>
<text x="470.0" y="190.0" class="fig-small" text-anchor="middle">visits d = 4.5</text>
<text x="470.0" y="210.0" class="fig-small" text-anchor="middle">pi(stay) = 0.5</text>
<text x="470.0" y="230.0" class="fig-small" text-anchor="middle">advantage of stay = +0.725</text>
<text x="470.0" y="258.0" class="fig-small fig-bold" text-anchor="middle">dJ / d theta(stay) = +1.63125</text>
<text x="470.0" y="280.0" class="fig-bold" text-anchor="middle">raise the stay logit</text>
</svg>
Figure 2. The three factors multiply to the gradient on each state's stay logit. State 0 is visited more, but staying there is worse than average, so its stay logit falls by 0.619. State 1 is visited less, but staying there pays, so its stay logit rises by 1.631.
```

> [!TIP]
> Before you trust any gradient formula, compare it with central finite differences of $J$. `rlcheck.finite_difference` nudges each logit by $\pm 10^{-6}$ and re-solves for $J$. On a random MDP the two agree to about $10^{-9}$.

> [!KEY]
> The gradient of a softmax policy's return is $d^\pi(s)\, \pi(a \mid s)\, A^\pi(s, a)$: how often you are in the state, how likely the action already is, and how much better it is than average. You never need the gradient of the visitation.

> [!RECAP]
> - The discounted visitation $d^\pi$ solves $(I - \gamma P_\pi^\top) d = \mu_0$ and sums to $1 / (1 - \gamma)$ when no state is terminal.
> - $J = \sum_s d^\pi(s) r_\pi(s)$.
> - The theorem is $\nabla J = \sum_s d^\pi(s) \sum_a \pi(a \mid s) Q^\pi(s, a) \nabla \log \pi(a \mid s)$.
> - For a tabular softmax it collapses to $\partial J / \partial \theta_{s,a} = d^\pi(s)\,\pi(a \mid s)\,A^\pi(s, a)$.

## Build it step by step

The checks test the functions in this order. `softmax_policy`, `log_policy_grad`, `objective`, and `two_state_mdp` are given in full, because you wrote the first three in the last chapter.

1. **`discounted_visitation(mdp, pi)`.** Build `P_pi` with `np.einsum`, zero the rows of terminal states, and solve the transposed system with `np.linalg.solve`. It returns `(S,)` and does not normalize. The most common mistake is forgetting the transpose, which gives a vector that fails the flow equation. The checks use the two-state example, the series sum, the flow equation, and the identity $J = \sum_s d(s) r_\pi(s)$.
2. **`advantages(mdp, pi)`.** Use `rlenvs.policy_evaluation` and `rlenvs.q_from_v`, then subtract `v[:, None]`. Each row must average to zero under $\pi$.
3. **`exact_gradient(mdp, theta)`.** Multiply `d[:, None] * pi * adv`. The most common mistake is the normalized visitation. The finite-difference check catches the missing factor.
4. **`score_form_gradient(mdp, theta)`.** Loop over every state and action, and add `d[s] * pi[s, a] * q[s, a] * log_policy_grad(theta, s, a)`. It must equal `exact_gradient`, which shows that the collapse is exact.

## Cost and scaling

The exact gradient needs linear solves, so it works only on small tabular problems, in $O(S^3 + S^2 A)$ time. Finite differences cost $SA$ extra solves. The exact gradient is a measuring stick for checking estimators, not a training method.

## Where this breaks

**It needs a model.** The theorem's inputs, $d^\pi$ and $Q^\pi$, come from the transition probabilities. In a real problem you have only samples. REINFORCE replaces $d^\pi$ with "the states you visited, each with weight $\gamma^t$" and $Q^\pi$ with the sampled return.

**The weighting is often dropped.** Almost every deep RL implementation uses the visitation without the discount, a state distribution that weights all steps equally. That is a different direction from the true gradient, as the last chapter noted.

**Plain gradient ascent can stall.** When $\pi(a \mid s)$ is tiny for the best action, the gradient $d^\pi \pi A^\pi$ is tiny too, and ascent can crawl for a long time. The chapter on why policy optimization works makes this precise.

## Further reading

- [Sutton, McAllester, Singh, and Mansour (2000), *Policy gradient methods for reinforcement learning with function approximation*](https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html). Proves the theorem in the form this chapter uses.
- [Sutton and Barto (2018), *Reinforcement Learning: An Introduction*, chapter 13](http://incompleteideas.net/book/the-book-2nd.html). Derives the theorem step by step.
- [Nota and Thomas (2020), *Is the policy gradient a gradient?*](https://arxiv.org/abs/1906.07073). Shows what goes wrong when implementations drop the discount from the state weighting.
- [Agarwal, Kakade, Lee, and Mahajan (2021), *On the theory of policy gradient methods*](https://arxiv.org/abs/1908.00261). Analyzes when softmax policy gradients converge and how fast.

## The exercise

You write `discounted_visitation`, `advantages`, `exact_gradient`, and `score_form_gradient`.

The checks confirm that the visitation is $(5.5, 4.5)$ on the worked example, equals the series sum, obeys the flow equation, and turns $J$ into a weighted reward. They check the advantages against the hand table. They check `exact_gradient` against the hand-worked table and against finite differences of `objective` on four random MDPs. Finally, they check that the unsimplified score form gives the same gradient.
