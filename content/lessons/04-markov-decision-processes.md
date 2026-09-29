---
title: Markov decision processes
slug: 04-markov-decision-processes
part: "Part 1 — The reinforcement learning problem"
summary: Write a policy's values as a linear system and solve it exactly, then find optimal values and where a policy spends its time.
minutes: 55
gpu: false
objectives:
  - Define a Markov decision process by its states, actions, transitions, rewards, and discount.
  - Solve the Bellman expectation equation for a policy's values with one linear solve.
  - Convert state values to action values, and state the Bellman optimality equation.
  - Compute the discounted state-visitation distribution, and relate it to the expected return.
prereqs:
  - 01-the-rl-problem
  - 02-multi-armed-bandits
exercise: 04-markov-decision-processes
---

# Markov decision processes

In a bandit, your action decides the reward and nothing else. The next round
looks exactly like this one. Most real problems aren't like that. A chess move
changes the board. A thermostat setting changes tomorrow's temperature. An
action can pay little now and still be the right one, because of where it
leaves you.

A *Markov decision process*, or MDP, is the model that captures this. It adds
states to the bandit, and each action moves you between them. With the model
written down as arrays, the value of a policy stops being something you
estimate by averaging episodes. It becomes the solution of a linear system,
exact to machine precision, in one call to `np.linalg.solve`.

This lesson sets up that algebra. Every later method, even the ones that
never see the arrays, is trying to solve the equations in it.

## What you already know

- The return $G_t = r_t + \gamma G_{t+1}$ adds up discounted rewards from step
  $t$ on.
- A policy $\pi(a \mid s)$ gives the probability of each action in each state.
- The value $V^\pi(s)$ is the expected return from state $s$ under $\pi$.
- `np.linalg.solve(M, b)` solves $M x = b$ without forming $M^{-1}$.
- `np.einsum` sums products of arrays over the axes you name.

## The pieces of an MDP

An MDP has five parts:

- A finite set of *states*, numbered $0$ to $S - 1$.
- A finite set of *actions*, numbered $0$ to $A - 1$.
- *Transition probabilities* $P(s' \mid s, a)$: the chance of landing in state
  $s'$ after taking action $a$ in state $s$.
- *Expected rewards* $R(s, a)$: the average reward $r_t$ for taking $a$ in $s$.
- A discount factor $\gamma$ and a start-state distribution $\mu_0$.

In code, these are arrays. `mdp.P` has shape `(S, A, S)`, and each
`mdp.P[s, a]` is a probability distribution over next states. `mdp.R` has
shape `(S, A)`. Some states are *terminal*: an episode ends there, and a
terminal state's value is 0 by definition.

The word *Markov* names the key assumption. The next state and reward depend
only on the current state and action, not on how you got there. The state
carries everything about the past that matters for the future. If your
"state" leaves something out, such as a car's speed when you only record its
position, the model's predictions go wrong.

### A two-state example

The following MDP appears throughout this lesson and the next. In each of two
states, you can *stay* (action 0) or *go* to the other state (action 1):

| State | Action | Reward | Next state |
|---|---|---|---|
| 0 | stay | 1 | 0 |
| 0 | go | 0 | 1 |
| 1 | stay | 2 | 1 |
| 1 | go | 0 | 0 |

The discount is $\gamma = 0.9$. State 1 pays more for staying, but getting
there from state 0 costs a step with no reward. Is the trip worth it? The
Bellman equations answer that.

## From a policy to a Markov chain

Fix a policy $\pi$. Averaging over its actions turns the MDP into a plain
Markov chain with rewards:

$$
P_\pi(s, s') = \sum_a \pi(a \mid s)\, P(s' \mid s, a), \qquad r_\pi(s) = \sum_a \pi(a \mid s)\, R(s, a)
$$

Read it as: $P_\pi(s, s')$ is the chance of moving from $s$ to $s'$ in one
step when you act by $\pi$, and $r_\pi(s)$ is the reward you expect in $s$.
$P_\pi$ is an $S \times S$ matrix whose rows sum to one. For a terminal state,
set both its row of $P_\pi$ and its $r_\pi$ to zero, because nothing happens
after the episode ends.

```python
P_pi = np.einsum("sa,sat->st", pi, mdp.P)   # (S, S)
r_pi = (pi * mdp.R).sum(axis=1)             # (S,)
live = ~mdp.terminal
P_pi, r_pi = P_pi * live[:, None], r_pi * live
```

For the uniform random policy in the two-state MDP, each state stays or goes
with probability $1/2$, so
$P_\pi = \begin{bmatrix}0.5&0.5\\0.5&0.5\end{bmatrix}$ and
$r_\pi = (0.5, 1)$.

## The Bellman expectation equation

The return satisfies $G_t = r_t + \gamma G_{t+1}$. Take expectations of both
sides, starting in state $s$ and acting by $\pi$. The left side becomes
$V^\pi(s)$. On the right, the first reward averages to $r_\pi(s)$, and the rest
averages to $\gamma$ times the value of wherever you land:

$$
V^\pi(s) = r_\pi(s) + \gamma \sum_{s'} P_\pi(s, s')\, V^\pi(s')
$$

Read it as: a state's value is the reward you expect now plus the discounted
average value of the next state. This is the *Bellman expectation equation*.
It holds at every state at once, so it's a system of $S$ linear equations in
$S$ unknowns. In matrix form, $V = r_\pi + \gamma P_\pi V$, which rearranges
to:

$$
(I - \gamma P_\pi)\, V^\pi = r_\pi
$$

Read it as: one linear system whose solution is the whole value vector. When
$\gamma < 1$, the matrix $I - \gamma P_\pi$ is always invertible. Multiplying
by $\gamma P_\pi$ averages a vector's entries and scales them by $\gamma$, so
it shrinks the largest absolute entry of every nonzero vector. No nonzero $V$
can then satisfy $V = \gamma P_\pi V$.

```python
V = np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)
```

### Worked example

For the uniform policy in the two-state MDP, both rows of $P_\pi$ are
$(0.5, 0.5)$. So both equations share the term $0.9 \times m$, where
$m = (V(0) + V(1))/2$ is the average value:

- $V(0) = 0.5 + 0.9\, m$
- $V(1) = 1 + 0.9\, m$

Average the two equations: $m = 0.75 + 0.9\, m$, so $m = 7.5$. Then
$V(0) = 0.5 + 6.75 = 7.25$ and $V(1) = 1 + 6.75 = 7.75$. A random walker
starting in state 1 is worth half a reward more, because it collects the
bigger reward on its first step.

> [!KEY]
> A policy's values solve the linear system $(I - \gamma P_\pi) V = r_\pi$. With
> the model known, policy evaluation is one call to `np.linalg.solve`.

## Action values

The state value $V^\pi(s)$ averages over the policy's first action. The *action
value* $Q^\pi(s, a)$ fixes the first action and follows $\pi$ afterward. It
answers the question "what if I did $a$ here, just this once?"

$$
Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a)\, V^\pi(s')
$$

Read it as: the reward for $a$ in $s$, plus the discounted average value of
where $a$ takes you. Averaging $Q^\pi$ over the policy's own action choices
gives back $V^\pi(s) = \sum_a \pi(a \mid s)\, Q^\pi(s, a)$.

In numpy, `mdp.P @ V` has shape `(S, A)`: entry `[s, a]` is the expected value
of the next state. So:

```python
Q = mdp.R + gamma * mdp.P @ V
Q[mdp.terminal] = 0.0
```

For the uniform policy, $V = (7.25, 7.75)$ gives:

| State | $Q(s, \text{stay})$ | $Q(s, \text{go})$ |
|---|---|---|
| 0 | $1 + 0.9 \times 7.25 = 7.525$ | $0 + 0.9 \times 7.75 = 6.975$ |
| 1 | $2 + 0.9 \times 7.75 = 8.975$ | $0 + 0.9 \times 7.25 = 6.525$ |

Each row averages to its $V$: $(7.525 + 6.975)/2 = 7.25$. The difference
$Q^\pi(s, a) - V^\pi(s)$ is called the *advantage* $A^\pi(s, a)$. It measures
how much better $a$ is than the policy's average action in $s$. The course
uses it heavily from Part 4 on.

## Optimal values and the Bellman optimality equation

The best any policy can do from state $s$ is the *optimal value* $V^*(s)$. An
optimal agent takes the best action in every state, so the average over the
policy becomes a maximum over actions:

$$
V^*(s) = \max_a \Bigl[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a)\, V^*(s') \Bigr]
$$

Read it as: the optimal value of a state is the value of its best action,
assuming you act optimally afterward. This is the *Bellman optimality
equation*. The max makes it nonlinear, so no single linear solve gives
$V^*$. The next lesson, "Dynamic programming", solves it by iteration.

For the two-state MDP, you can reason it out. Staying in state 1 forever is
worth $2 + 0.9 \times 2 + 0.81 \times 2 + \cdots = 2 / (1 - 0.9) = 20$. From
state 0, staying forever is worth $1 / 0.1 = 10$, while going is worth
$0 + 0.9 \times 20 = 18$. So $V^* = (18, 20)$. The optimal action values are
$Q^*(0, \cdot) = (1 + 0.9 \times 18,\ 0 + 0.9 \times 20) = (17.2, 18)$ and
$Q^*(1, \cdot) = (2 + 0.9 \times 20,\ 0 + 0.9 \times 18) = (20, 16.2)$. The
maxima are 18 and 20, so $(18, 20)$ does satisfy the equation. The trip is
worth it.

Once you know $Q^*$, acting optimally is easy: in each state, pick an action
that maximises $Q^*(s, a)$. In a finite MDP with $\gamma < 1$, that greedy
policy is optimal from every state at once. You never need a different policy
for different starting points.

> [!WARNING]
> Terminal states need care in both equations. If you forget to zero their
> rows, a terminal state keeps collecting reward forever and inflates the
> value of every state that can reach it.

## The optimality backup is a contraction

Write the right-hand side of the optimality equation as an operator: $T^*$
takes a value vector $V$ and returns $\max_a [R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V(s')]$
at every state. Then $V^*$ is a *fixed point*: $T^* V^* = V^*$.

$T^*$ has a useful property. For any two value vectors $U$ and $V$:

$$
\max_s \bigl| (T^* U)(s) - (T^* V)(s) \bigr| \le \gamma \max_s \bigl| U(s) - V(s) \bigr|
$$

Read it as: applying $T^*$ to two guesses brings them at least a factor
$\gamma$ closer, measured by their largest disagreement across states. An
operator with this property is a *$\gamma$-contraction in the max norm*.

The proof idea takes three sentences. The difference of two maxima is at most
the largest difference between the things being maximised. Each of those is
$\gamma$ times an average of $U - V$ over next states. An average is never
larger than the largest entry. A contraction has exactly one fixed point, so
$V^*$ is unique, and the next lesson uses the same fact to show that iterating
$T^*$ converges to it.

## Where a policy spends its time

The last object in this lesson answers "how often does $\pi$ visit each
state?" It counts visits with a discount, so early visits count more. The
*discounted state-visitation distribution* of $\pi$ from $\mu_0$ is:

$$
d^\pi(s) = (1 - \gamma) \sum_{t=0}^{\infty} \gamma^t \Pr(s_t = s)
$$

Read it as: add up the probability of being in $s$ at every step, weight step
$t$ by $\gamma^t$, and scale by $1 - \gamma$ so the weights sum to one. Since
$\Pr(s_t = \cdot) = \mu_0^\top P_\pi^t$, the sum is a geometric series of
matrices, and $d^\pi = (1 - \gamma)\, \mu_0^\top (I - \gamma P_\pi)^{-1}$. It
satisfies a *flow equation*, $d^\pi = (1 - \gamma)\mu_0 + \gamma\, d^\pi P_\pi$:
the visits to a state come from starting there or from flowing in from a
neighbour.

```python
d = (1 - gamma) * np.linalg.solve((np.eye(S) - gamma * P_pi).T, mu0)
d = d / d.sum()
```

The transpose appears because $d^\pi$ is a row vector multiplying from the
left. With terminal states, episodes end and the unnormalised $d$ sums to less
than one, so the code divides by its sum. `rlenvs.state_distribution` does
the same.

### Worked example

In the two-state MDP starting in state 0, the uniform policy is in state 0 at
$t = 0$ and then in each state with probability $1/2$ forever. So
$d(0) = 0.1 \times (1 + 0.5 \times 0.9 / 0.1) = 0.55$ and $d(1) = 0.45$. The
optimal policy goes to state 1 and stays, so its $d = (0.1, 0.9)$.

The visitation distribution rewrites the expected return as an average over
states:

$$
\mu_0^\top V^\pi = \frac{1}{1 - \gamma} \sum_s d^\pi(s)\, r_\pi(s)
$$

Read it as: the expected return from the start equals the reward rate at each
state, weighted by how often you're there, times the effective horizon
$1/(1 - \gamma)$. For the uniform policy,
$10 \times (0.55 \times 0.5 + 0.45 \times 1) = 7.25$, which is $V(0)$. Lesson
15, "The policy gradient theorem", builds on this identity.

> [!KEY]
> The same matrix $I - \gamma P_\pi$ gives both the values, by solving on the
> right, and the visitation distribution, by solving on the left.

> [!RECAP]
> - `P_pi = np.einsum("sa,sat->st", pi, P)` and `r_pi = (pi * R).sum(1)`, with
>   terminal rows zeroed.
> - $V^\pi$ solves $(I - \gamma P_\pi) V = r_\pi$.
> - `Q = R + gamma * P @ V`, with terminal rows zeroed, and
>   $V = \sum_a \pi(a \mid s) Q(s, a)$.
> - The optimality backup is `Q.max(axis=1)`, a $\gamma$-contraction whose
>   fixed point is $V^* = (18, 20)$ in the example.
> - $d^\pi$ solves the transposed system with $\mu_0$ on the right, scaled by
>   $1 - \gamma$ and normalised.

## Build it step by step

The checks test the functions in this order.

1. **`policy_matrices(mdp, pi)`.** Returns `(P_pi, r_pi)` with shapes `(S, S)`
   and `(S,)`. Use the `einsum` above and zero terminal rows. Common mistake:
   the einsum string `"sa,sat->ts"`, which returns the transpose. The checks
   test row sums and one entry by hand.
2. **`evaluate_exact(mdp, pi)`.** One solve. Common mistake: solving with
   `P_pi.T`. The checks compare against the worked example, against
   `rlenvs.policy_evaluation`, and against the Bellman residual.
3. **`q_from_v(mdp, v)`.** `mdp.R + mdp.gamma * mdp.P @ v`, then zero the
   terminal rows. The checks use both tables from this lesson and the identity
   $\sum_a \pi(a \mid s) Q(s, a) = V(s)$.
4. **`bellman_optimality_backup(mdp, v)`.** The max over actions of your
   `q_from_v`. The checks confirm that $(18, 20)$ is fixed, and that 200
   random pairs of vectors move closer by at least a factor of $\gamma$.
5. **`discounted_visitation(mdp, pi)`.** Solve the transposed system, scale by
   $1 - \gamma$, and normalise. Common mistake: forgetting the transpose,
   which gives a vector that doesn't satisfy the flow equation. The checks
   use both worked distributions, the flow equation, and the return identity.

## Cost and scaling

Everything in this lesson is dense linear algebra on arrays of size $S$ and
$A$.

| Operation | Time | Memory |
|---|---|---|
| Build $P_\pi$ and $r_\pi$ | $O(S^2 A)$ | $O(S^2)$ |
| Solve for $V^\pi$ or $d^\pi$ | $O(S^3)$ | $O(S^2)$ |
| Action values from $V$ | $O(S^2 A)$ | $O(SA)$ |
| One optimality backup | $O(S^2 A)$ | $O(SA)$ |

Storing `P` itself takes $S^2 A$ numbers. A grid of 1000 cells with 4 actions
needs 4 million, which is fine. A board game with $10^{20}$ states needs more
memory than exists. Exact solves are for small problems and for checking
answers. Parts 2 and 3 learn from sampled transitions instead.

## Where this breaks

**No discount and no guaranteed ending.** With $\gamma = 1$, the matrix
$I - P_\pi$ is singular whenever the policy can loop forever without reaching
a terminal state. The values of those states are infinite or undefined.
`rlenvs.policy_evaluation` returns `nan` for them, and every exercise in this
course uses $\gamma < 1$ or a policy that always ends.

**State that isn't Markov.** If the observation leaves out something that
matters, like a moving object's velocity, the true process isn't an MDP in
those states. The equations still solve, but they describe the wrong world.
People stack recent observations, or give the agent memory.

**An unknown model.** Everything here reads `P` and `R` directly. Most agents
never get them. Later lessons estimate values from sampled experience, and
lesson 24, "Model-based reinforcement learning", learns the model itself.

**Huge state spaces.** A dense solve at $O(S^3)$ is hopeless beyond a few
thousand states. Iterative methods from the next lesson and function
approximation from Part 3 take over.

## Further reading

- [Sutton and Barto, *Reinforcement Learning: An Introduction*, Chapter 3](http://incompleteideas.net/book/the-book-2nd.html)
  defines finite MDPs, value functions, and both Bellman equations.
- [Bellman, "A Markovian decision process" (1957)](http://www.iumj.indiana.edu/IUMJ/FULLTEXT/1957/6/56038)
  is Bellman's short paper on the model, whose title gave it its name.
- [Puterman, *Markov Decision Processes: Discrete Stochastic Dynamic Programming* (1994)](https://www.wiley.com/en-us/Markov+Decision+Processes:+Discrete+Stochastic+Dynamic+Programming-p-9781118625873)
  is the standard reference for the theory, including contractions and
  optimality of deterministic policies.
- [Kaelbling, Littman, and Moore, "Reinforcement learning: A survey" (1996)](https://arxiv.org/abs/cs/9605103)
  reviews the MDP framework and the classical methods built on it.

## The exercise

You build the linear algebra of an MDP. Write `policy_matrices(mdp, pi)` for
$P_\pi$ and $r_\pi$, `evaluate_exact(mdp, pi)` for the value solve,
`q_from_v(mdp, v)` for action values, `bellman_optimality_backup(mdp, v)` for
one application of $T^*$, and `discounted_visitation(mdp, pi)` for $d^\pi$.
The two-state MDP from this lesson is defined in the starter file as
`two_state_mdp`.

The checks verify $P_\pi$, $r_\pi$, $V = (7.25, 7.75)$, and the action-value
tables from this lesson. They compare `evaluate_exact` against
`rlenvs.policy_evaluation` on random MDPs and on a grid with a terminal goal,
and check the Bellman residual. They confirm that $(18, 20)$ is the fixed point
of your backup and that the backup contracts 200 random pairs by at least
$\gamma$. Last, they check the visitation distributions $(0.55, 0.45)$ and
$(0.1, 0.9)$, the flow equation, and the identity between the expected return
and the visitation-weighted reward.
