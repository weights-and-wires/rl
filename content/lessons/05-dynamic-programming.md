---
title: Dynamic programming
slug: 05-dynamic-programming
part: "Part 1 — The reinforcement learning problem"
summary: Find optimal policies with a known model by repeated Bellman sweeps, and bound the error after every sweep.
minutes: 55
gpu: false
objectives:
  - Evaluate a policy by repeated Bellman sweeps, and stop with a guaranteed error bound.
  - Implement policy iteration, and explain why each round never lowers any state's value.
  - Implement value iteration, and show that its error shrinks by at least a factor of gamma every sweep.
  - Compute how many sweeps a target accuracy needs from the contraction rate.
prereqs:
  - 04-markov-decision-processes
exercise: 05-dynamic-programming
---

# Dynamic programming

The previous lesson wrote down what optimal values must satisfy, the Bellman
optimality equation, but it didn't say how to solve it. The max over actions
makes the equation nonlinear, so one linear solve won't do. This lesson solves
it by turning the equation into an update and applying it over and over.

*Dynamic programming* is the name Richard Bellman gave to this family of
methods in the 1950s. They need the full model, the arrays `P` and `R`, so
they're planning methods rather than learning methods. But they matter far
beyond small problems. Almost every reinforcement learning algorithm is a
sampled, approximate version of one of the two you build here: policy
iteration and value iteration.

The lesson also answers a practical question: when do you stop? The
contraction property from the previous lesson gives an exact answer, and on
the running two-state example the bound it gives is tight at every sweep.

## What you already know

- A policy's values solve $(I - \gamma P_\pi) V^\pi = r_\pi$.
- Action values are $Q(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V(s')$.
- The optimality backup $T^*$ takes the max over actions of $Q$, and $V^*$ is
  its fixed point.
- $T^*$ is a $\gamma$-contraction: it brings any two value vectors at least a
  factor $\gamma$ closer in the max norm.
- In the two-state MDP, $V^* = (18, 20)$: go from state 0, stay in state 1.

## Evaluate a policy by sweeping

Start with a simpler job: find $V^\pi$ for a fixed policy without a linear
solve. Take any guess $V_0$, say all zeros. Plug it into the right side of the
Bellman expectation equation, and call the result $V_1$. Repeat:

$$
V_{k+1} = r_\pi + \gamma P_\pi V_k
$$

Read it as: each *sweep* replaces every state's value with its expected reward
plus the discounted average of the current guesses at the next states. After
$k$ sweeps from zero, $V_k(s)$ is exactly the expected discounted reward of the
first $k$ steps. The rest of the return, discounted by $\gamma^k$, is still
missing, so $V_k$ approaches $V^\pi$ as $k$ grows.

```python
v = np.zeros(S)
while True:
    new = r_pi + gamma * P_pi @ v   # one synchronous sweep
    if np.max(np.abs(new - v)) < tol:
        break
    v = new
```

This is a *synchronous* sweep: every new entry is computed from the old vector.
Updating entries in place, so later states in a sweep see earlier states' new
values, also converges, often faster. But its iterates differ, and the bounds
in this lesson are stated for the synchronous version.

### When to stop

The loop stops when one sweep changes no value by more than `tol`. That does
*not* mean the answer is within `tol` of $V^\pi$. The contraction gives the
true guarantee. Write $\lVert x \rVert$ for $\max_s |x(s)|$. Then:

$$
\lVert V_{k+1} - V^\pi \rVert \le \frac{\gamma}{1 - \gamma} \lVert V_{k+1} - V_k \rVert
$$

Read it as: the true error of the latest sweep is at most
$\gamma / (1 - \gamma)$ times the size of its last change. With
$\gamma = 0.95$, that factor is 19. Here's why in one line. The error shrinks
by $\gamma$ per sweep, so
$\lVert V_{k+1} - V^\pi \rVert \le \gamma \lVert V_k - V^\pi \rVert \le \gamma \left(\lVert V_k - V_{k+1} \rVert + \lVert V_{k+1} - V^\pi \rVert\right)$.
Move the $V_{k+1} - V^\pi$ terms to one side and divide by $1 - \gamma$.

On the exercise's random MDP with 20 states, 4 actions, and $\gamma = 0.95$,
evaluating the uniform policy with `tol = 1e-8` stops after 332 sweeps. The
true error is $1.82 \times 10^{-7}$, about 18 times `tol`, and just under the
bound of $1.90 \times 10^{-7}$.

> [!WARNING]
> A small last change doesn't mean a small error. To guarantee an error of
> $\varepsilon$, stop when the last change drops below
> $\varepsilon (1 - \gamma) / \gamma$.

## Improve a policy greedily

Suppose you know $Q^\pi$ for some policy. In each state, you could switch to
the action with the highest $Q^\pi(s, a)$. That new policy $\pi'$ is *greedy*
with respect to $Q^\pi$. Is it better?

The *policy improvement theorem* says yes. If
$Q^\pi(s, \pi'(s)) \ge V^\pi(s)$ in every state, then
$V^{\pi'}(s) \ge V^\pi(s)$ in every state. The greedy policy satisfies the
condition by construction, because the best action is at least as good as the
policy's average. The proof idea: taking $\pi'$ for one step and then $\pi$ is
no worse than $\pi$. Unroll that one step at a time, and following $\pi'$
forever is no worse either.

```python
q = rlenvs.q_from_v(mdp, v)            # R + gamma * P @ v, terminal rows zero
pi = np.zeros_like(q)
pi[np.arange(S), q.argmax(axis=1)] = 1.0   # first maximiser wins ties
```

## Policy iteration

*Policy iteration* (Howard, 1960) alternates the two steps until nothing
changes:

1. Evaluate the current policy exactly.
2. Replace it with the greedy policy with respect to its values.
3. If the greedy policy equals the current one, stop.

Each round improves every state's value or leaves the policy unchanged. A
finite MDP has finitely many deterministic policies, $A^S$ of them, and the
values never go down, so no policy repeats and the loop must stop. When it
stops, the policy is greedy with respect to its own values. So its values
satisfy the Bellman optimality equation, and the policy is optimal.

### Worked example

Run policy iteration on the two-state MDP from "stay everywhere":

1. Evaluate: staying pays 1 forever in state 0 and 2 forever in state 1, so
   $V = (1 / 0.1,\ 2 / 0.1) = (10, 20)$.
2. Improve: in state 0, stay gives $1 + 0.9 \times 10 = 10$ and go gives
   $0 + 0.9 \times 20 = 18$, so switch to go. In state 1, stay gives
   $2 + 0.9 \times 20 = 20$ and go gives $0 + 0.9 \times 10 = 9$, so keep stay.
3. Evaluate the new policy: $V = (0.9 \times 20,\ 20) = (18, 20)$.
4. Improve: nothing changes. Stop.

Two evaluations reach $V^* = (18, 20)$ exactly. On the exercise's random
20-state MDP, policy iteration needs three evaluations.

> [!KEY]
> Policy iteration never makes any state worse, and it stops at an optimal
> policy after finitely many rounds. It usually needs only a handful.

## Value iteration

Policy iteration evaluates each policy completely before improving it. That's
a lot of work for a policy you're about to throw away. *Value iteration* goes
to the other extreme. It does one sweep of evaluation and improvement at once,
by applying the optimality backup directly:

$$
V_{k+1}(s) = \max_a \Bigl[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a)\, V_k(s') \Bigr]
$$

Read it as: each sweep, every state takes the value of its best action,
judged by the current guesses at the next states. No policy is stored. You
read one off at the end with `greedy_policy`.

```python
v = rlenvs.q_from_v(mdp, v).max(axis=1)   # one sweep of T*
```

### The error shrinks by gamma every sweep

Since $V^* = T^* V^*$, the contraction gives the rate directly:
$\lVert V_{k} - V^* \rVert = \lVert T^* V_{k-1} - T^* V^* \rVert \le \gamma \lVert V_{k-1} - V^* \rVert$.
Apply it $k$ times:

$$
\lVert V_k - V^* \rVert \le \gamma^k \lVert V_0 - V^* \rVert
$$

Read it as: every sweep cuts the worst error across states by at least a
factor of $\gamma$. The same stopping bound as for policy evaluation,
$\gamma / (1 - \gamma)$ times the last change, holds too.

### Worked example

Value iteration on the two-state MDP, from $V_0 = (0, 0)$, where
$\lVert V_0 - V^* \rVert = 20$:

| $k$ | $V_k(0)$ | $V_k(1)$ | Greedy action in state 0 | $\lVert V_k - V^* \rVert$ | $20 \times 0.9^k$ |
|---|---|---|---|---|---|
| 0 | 0 | 0 | stay | 20 | 20 |
| 1 | $\max(1, 0) = 1$ | $\max(2, 0) = 2$ | stay | 18 | 18 |
| 2 | $\max(1.9, 1.8) = 1.9$ | $\max(3.8, 0.9) = 3.8$ | go | 16.2 | 16.2 |
| 3 | $\max(2.71, 3.42) = 3.42$ | $\max(5.42, 1.71) = 5.42$ | go | 14.58 | 14.58 |
| 4 | $\max(4.078, 4.878) = 4.878$ | $\max(6.878, 3.078) = 6.878$ | go | 13.122 | 13.122 |

The "greedy action" column is the best action in state 0 when you judge by
$V_k$. It switches to the optimal "go" at $k = 2$, while the values are still
far from $(18, 20)$. That's typical. The greedy policy is often right long
before the values are.

The bound is exactly tight here. State 1 staying forever collects
$2(1 + 0.9 + \cdots + 0.9^{k-1}) = 20(1 - 0.9^k)$ in $k$ sweeps, so its error
is exactly $20 \times 0.9^k$.

### How many sweeps

The bound tells you in advance how many sweeps reach an error $\varepsilon$.
Solve $\gamma^k \lVert V_0 - V^* \rVert \le \varepsilon$ for $k$:

$$
k \ge \frac{\ln\left(\varepsilon / \lVert V_0 - V^* \rVert\right)}{\ln \gamma}
$$

Read it as: the number of sweeps grows with the log of how much you need to
shrink the error, divided by the log of the per-sweep factor. Since
$-\ln \gamma \approx 1 - \gamma$ for $\gamma$ near 1, that's about
$\frac{1}{1 - \gamma} \ln \frac{\lVert V_0 - V^* \rVert}{\varepsilon}$. Halving
$1 - \gamma$ doubles the sweeps.

```python
k = math.ceil(math.log(eps / initial_error) / math.log(gamma))
```

For the two-state MDP with $\varepsilon = 0.01$:
$\ln(0.01 / 20) / \ln 0.9 = 72.14$, so 73 sweeps. Because the bound is tight,
the error after 72 sweeps is $20 \times 0.9^{72} = 0.01015$, just above
$0.01$, and after 73 it's $0.00914$. Policy iteration reached the exact answer
in two evaluations.

> [!KEY]
> Value iteration's error shrinks by at least $\gamma$ per sweep, so reaching
> error $\varepsilon$ takes about $\frac{1}{1-\gamma}\ln\frac{\text{initial error}}{\varepsilon}$
> sweeps.

### How good is the greedy policy?

You usually stop value iteration early and act greedily. Singh and Yee (1994)
bound what that costs. If $\lVert V - V^* \rVert \le \varepsilon$, the greedy
policy $\pi_V$ satisfies
$\lVert V^* - V^{\pi_V} \rVert \le \frac{2 \gamma}{1 - \gamma} \varepsilon$.
The bound is loose: in the table, $V_1$ has error 18 and its greedy policy,
"stay everywhere", loses 8 in state 0, far under $18 \times 18 = 324$. But it
guarantees that a small value error can't hide a terrible policy.

## Generalised policy iteration

Policy iteration and value iteration are two ends of one design.
*Generalised policy iteration* is Sutton and Barto's name for any scheme that
interleaves some evaluation with some greedy improvement. Evaluate fully, and
you get policy iteration. Do one sweep, and you get value iteration. Do a few
sweeps between improvements, and you get *modified policy iteration*, which is
often fastest in practice. Nearly every reinforcement learning algorithm in
the course fits this pattern, with sampled experience in place of the model.

> [!RECAP]
> - Iterative evaluation sweeps `new = r_pi + gamma * P_pi @ v` until the
>   largest change is below `tol`.
> - The true error after stopping is at most $\gamma / (1 - \gamma)$ times the
>   last change.
> - The greedy policy is a one-hot at `rlenvs.q_from_v(mdp, v).argmax(1)`.
> - Policy iteration evaluates exactly and improves greedily until the policy
>   stops changing.
> - Value iteration sweeps `v = q_from_v(mdp, v).max(1)`, with error at most
>   $\gamma^k \lVert V_0 - V^* \rVert$.

## Build it step by step

The checks test the functions in this order.

1. **`iterative_policy_evaluation(mdp, pi, tol)`.** Get `P_pi` and `r_pi`
   from the given `policy_matrices`, sweep from zeros, and return the last
   sweep and the sweep count. Common mistake: returning `v` from before the
   last sweep. The checks compare against the exact value within the
   $\gamma / (1 - \gamma)$ bound.
2. **`greedy_policy(mdp, v)`.** Build `q` with `rlenvs.q_from_v`, then a
   one-hot row per state. Common mistake: breaking ties at random. The checks
   use an MDP whose two actions are identical and expect action 0.
3. **`value_iteration(mdp, tol, v0=None)`.** Keep a list that starts with `V_0`
   and gains every sweep, including the last. Return `(v, np.array(history))`.
   The checks compare rows 0 to 4 with the table, test the $\gamma^k$ bound
   and the stopping bound at every sweep, and check that the greedy action
   switches at $V_2$.
4. **`sweeps_needed(gamma, initial_error, eps)`.** One `math.ceil` of a ratio
   of logs, with 0 when the initial error is already at most `eps`. Common
   mistake: `math.floor`, which gives 72 and misses the target. The check
   confirms that sweeps 72 and 73 straddle 0.01.
5. **`policy_iteration(mdp, pi0=None)`.** Loop: solve for `v` with
   `np.linalg.solve`, record it, take `greedy_policy`, and stop with
   `np.array_equal`. The checks compare the recorded values with
   $(10, 20)$ and $(18, 20)$, check that no state's value ever drops on a
   random MDP, and compare the result with $V^*$.

## Cost and scaling

Each method's cost is sweeps times the cost of a sweep. The table uses $S$
states and $A$ actions.

| Method | Cost of one round | Rounds |
|---|---|---|
| Iterative policy evaluation | $O(S^2)$ per sweep | About $\frac{1}{1-\gamma}\ln\frac{1}{\varepsilon}$ |
| Value iteration | $O(S^2 A)$ per sweep | About $\frac{1}{1-\gamma}\ln\frac{1}{\varepsilon}$ |
| Policy iteration | $O(S^3 + S^2 A)$ per round | Usually a handful |

On the exercise's 20-state MDP with $\gamma = 0.95$, value iteration takes 449
sweeps to reach a change below $10^{-10}$, while policy iteration takes three
evaluations. For fixed $\gamma$, the number of policy iteration rounds is
bounded by a polynomial in $S$ and $A$ (Ye, 2011). Value iteration's sweeps
grow like $1 / (1 - \gamma)$, which hurts at $\gamma = 0.999$. Memory is
dominated by `P`, which holds $S^2 A$ numbers for all three methods.

## Where this breaks

**The curse of dimensionality.** Bellman's own phrase. A robot arm with seven
joints, each discretised into 100 positions, has $10^{14}$ states. A sweep
over them is impossible. Part 3 replaces the table with a function
approximator.

**No model.** Dynamic programming reads `P` and `R`. When you can only sample
transitions, Part 2 replaces the expectation in each backup with sampled
experience.

**Discount close to 1.** Value iteration's sweep count grows like
$1 / (1 - \gamma)$. With $\gamma = 1$, the contraction argument fails
entirely, and convergence needs a policy that is sure to end. Policy
iteration suffers less, and people also use modified policy iteration.

**Oscillating ties.** If policy iteration breaks ties differently each round,
it can switch between equally good actions and never stop. Break ties the same
way every time, such as the first maximiser, or stop when values stop
changing.

## Further reading

- [Sutton and Barto, *Reinforcement Learning: An Introduction*, Chapter 4](http://incompleteideas.net/book/the-book-2nd.html)
  covers policy evaluation, policy iteration, value iteration, and
  generalised policy iteration, with the grid-world and gambler's examples.
- [Puterman, *Markov Decision Processes: Discrete Stochastic Dynamic Programming* (1994)](https://www.wiley.com/en-us/Markov+Decision+Processes:+Discrete+Stochastic+Dynamic+Programming-p-9781118625873)
  proves convergence of every method here, and covers modified policy
  iteration.
- [Singh and Yee, "An upper bound on the loss from approximate optimal-value functions" (1994)](https://doi.org/10.1023/A:1022693225949)
  proves the $2\gamma\varepsilon/(1-\gamma)$ bound on acting greedily with
  approximate values.
- [Ye, "The simplex and policy-iteration methods are strongly polynomial for the Markov decision problem with a fixed discount rate" (2011)](https://doi.org/10.1287/moor.1110.0516)
  bounds the number of policy iteration rounds.

## The exercise

You build both dynamic programming methods and the tools to reason about their
error. Write `iterative_policy_evaluation(mdp, pi, tol)`,
`greedy_policy(mdp, v)`, `value_iteration(mdp, tol, v0=None)`,
`sweeps_needed(gamma, initial_error, eps)`, and
`policy_iteration(mdp, pi0=None)`. The starter file gives `two_state_mdp` and
`policy_matrices`.

The checks verify iterative evaluation against the exact value within its
stopping bound, and the greedy policy's choices and tie-breaking. On the
two-state MDP, they compare value iteration's first five iterates with this
lesson's table, and check that the greedy action in state 0 switches at $V_2$.
On a random 20-state MDP, they check the $\gamma^k$ bound and the stopping bound
at every sweep. They check `sweeps_needed(0.9, 20, 0.01) == 73` against the
actual errors, and that policy iteration evaluates $(10, 20)$ and then
$(18, 20)$. Last, they check that policy iteration never lowers any state's
value and finds $V^*$ on the random MDP.
