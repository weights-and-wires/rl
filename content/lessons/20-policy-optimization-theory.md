---
title: Why policy optimisation works
slug: 20-policy-optimization-theory
part: "Part 4 — Policy optimisation"
summary: Verify the performance difference lemma exactly, measure softmax policy gradient's slow 1/t rate, and show NPG is multiplicative weights.
minutes: 60
gpu: false
objectives:
  - Verify the performance difference lemma to rounding error, and say why the surrogate that TRPO and PPO optimise is only first-order correct.
  - Measure softmax policy gradient's 1/t rate on a bandit against Mei et al.'s bound, and explain why a bad start makes it crawl.
  - Show that the natural policy gradient for a softmax policy is a multiplicative-weights step, and check Agarwal et al.'s dimension-free bound.
prereqs:
  - 15-policy-gradient-theorem
  - 18-natural-gradients-and-trpo
  - 19-proximal-policy-optimization
exercise: 20-policy-optimization-theory
---

# Why policy optimisation works

The last five lessons built policy gradient methods that work in practice.
REINFORCE, actor-critic, TRPO, and PPO all nudge a policy uphill. None of
those lessons asked the harder questions. Does the nudge ever reach the best
policy, given that the objective isn't concave? How fast? And why does a
trust region help?

This lesson answers them on problems small enough to solve exactly. One
identity, the performance difference lemma, sits under almost every result.
You verify it to fifteen decimal places. Then you see why plain softmax
policy gradient reaches the optimum but slowly, sometimes painfully slowly.
Finally, you see that the natural gradient turns into a simple
multiplicative-weights rule that escapes the slowdown.

## What you already know

- A policy $\pi(a \mid s)$ gives a probability for each action in each state.
  $J(\pi) = \sum_s \mu_0(s) V^\pi(s)$ is its expected discounted return from
  the start distribution $\mu_0$.
- The advantage $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ says how much better
  action $a$ is than the policy's average in state $s$.
- The policy gradient theorem gives the gradient of $J$ for a softmax policy
  exactly, from the visitation of each state.
- The natural gradient preconditions the gradient with the Fisher
  information matrix, so step size is measured in KL divergence.
- TRPO and PPO maximise a surrogate built from the old policy's samples.

## Compare two policies with the performance difference lemma

How much better is a candidate policy $\pi'$ than an old policy $\pi$? Run
$\pi'$, and at each state it visits, score its action with the *old*
policy's advantage. Add up those scores, and you get the exact improvement.

An everyday version: to know how much a new commute saves, sum, over the
route you actually take, how much faster each turn is than your old habit
at that corner.

The lemma needs the *normalised discounted state distribution*
$d^{\pi'}(s) = (1 - \gamma) \sum_{t \ge 0} \gamma^t \Pr(s_t = s)$, where the
states come from running $\pi'$ from $\mu_0$. It sums to one.
`rlenvs.state_distribution` returns it. The lesson on the policy gradient
theorem used the unnormalised version, which is this one divided by
$1 - \gamma$. Kakade and Langford (2002) state the lemma as

$$
J(\pi') - J(\pi) = \frac{1}{1 - \gamma} \sum_s d^{\pi'}(s) \sum_a \pi'(a \mid s)\, A^{\pi}(s, a)
$$

Read it from the inside out. In each state $s$, average the old advantage
over the new policy's actions. Weight each state by how often the new policy
visits it. Divide by $1 - \gamma$ to undo the normalisation. The result is
the exact gain, with no approximation.

The proof idea takes three sentences. Write the gain as a sum along the new
policy's trajectory of $r_t + \gamma V^\pi(s_{t+1}) - V^\pi(s_t)$, discounted
by $\gamma^t$. The $V^\pi$ terms telescope and leave $J(\pi') - J(\pi)$. The
expected value of each bracket, given $s_t$ and $a_t$, is $A^\pi(s_t, a_t)$.

In numpy, the whole lemma is three lines:

```python
d_new = rlenvs.state_distribution(mdp, pi_new)
per_state = (pi_new * advantage(mdp, pi_old)).sum(axis=1)
gain = d_new @ per_state / (1 - mdp.gamma)
```

> [!KEY]
> The performance difference lemma is an identity: the new policy's visits,
> scored by the old policy's advantage, give the exact improvement.

### Worked example

Take two states. In each, action 0 stays and action 1 switches to the other
state. Staying in state 1 pays 1, and everything else pays 0. The discount is
$\gamma = 0.9$, and every episode starts in state 0.

Under the uniform policy, $V = (2.25, 2.75)$ and the advantages are

$$
A = \begin{bmatrix} -0.225 & 0.225 \\ 0.725 & -0.725 \end{bmatrix}
$$

Switch to the greedy policy: leave state 0, then stay in state 1. Its return
is $J = 9$, so the true gain is $9 - 2.25 = 6.75$. The lemma gets there as follows:

- The greedy policy's state distribution is $d' = (0.1, 0.9)$. It spends
  one step in state 0, then stays in state 1 forever.
- The advantage of the greedy action is $0.225$ in state 0 and $0.725$ in
  state 1.
- The lemma gives $(0.1 \times 0.225 + 0.9 \times 0.725) / 0.1 = 6.75$.
  It's exact.

## Swap the distribution: the surrogate

The lemma has a catch. It needs $d^{\pi'}$, the state distribution of a
policy you haven't run yet. Your samples come from $\pi$. So TRPO and PPO
replace $d^{\pi'}$ with $d^{\pi}$. That gives the *surrogate gain*:

$$
L_\pi(\pi') = \frac{1}{1 - \gamma} \sum_s d^{\pi}(s) \sum_a \pi'(a \mid s)\, A^{\pi}(s, a)
$$

Read it as the lemma with one change: weight states by where the old policy
goes, not the new one. In code, pass `pi_old` to `state_distribution`.

In the worked example, the old uniform policy has $d = (0.55, 0.45)$. The
surrogate gives $(0.55 \times 0.225 + 0.45 \times 0.725) / 0.1 = 4.5$, not
6.75. The greedy policy spends more time in the good state than the old
policy did, and the surrogate can't see that.

For a small step, the swap costs little. Mix the policies,
$\pi_\alpha = (1 - \alpha)\pi + \alpha \pi'$. The surrogate and the true gain
agree in value and in slope at $\alpha = 0$. Their difference grows like
$\alpha^2$. Halve the step, and the error falls by a factor of four. The
exercise measures exactly that.

How far can you trust the surrogate? Schulman et al. (2015) bound the error.
With $\epsilon = \max_{s, a} \lvert A^\pi(s, a) \rvert$,

$$
J(\pi') - J(\pi) \ge L_\pi(\pi') - \frac{4 \epsilon \gamma}{(1 - \gamma)^2} \max_s D_{\mathrm{KL}}\big(\pi(\cdot \mid s) \,\Vert\, \pi'(\cdot \mid s)\big)
$$

Read it as: the true gain is at least the surrogate, minus a penalty that
grows with the largest per-state KL divergence. Maximise the right-hand side
each step, and the true return can never fall. This is the argument behind
trust regions.

The constant is large. With $\gamma = 0.9$ and $\epsilon = 1$, it's
$4 \times 0.9 / 0.01 = 360$, so a per-state KL of 0.01 could cost 3.6 units
of return. TRPO replaces the penalty with a constraint on the *average* KL,
which drops the guarantee but takes useful steps. PPO's clipping inherits no
guarantee at all.

> [!KEY]
> The surrogate uses the old policy's states, so it's correct only to first
> order. Trust regions exist to keep the step small enough that the second-order
> error stays small.

## Softmax policy gradient converges, slowly

A tabular *softmax policy* gives each state and action a logit
$\theta_{s, a}$, and $\pi_\theta(\cdot \mid s)$ is the softmax of row $s$.
$J(\theta)$ isn't concave in $\theta$. Even so, exact gradient ascent
reaches the optimal policy, as long as the step is small enough and every
state has positive start probability (Agarwal et al., 2021). The question is
how fast.

Mei et al. (2020) answer it cleanly for a bandit, a one-state problem with
$K$ arms and mean rewards $r \in [0, 1]^K$. The objective is the expected
reward $\pi_\theta^\top r$, and the update is

$$
\theta_{t+1}(a) = \theta_t(a) + \eta\, \pi_{\theta_t}(a) \big(r(a) - \pi_{\theta_t}^\top r\big)
$$

Read it as: push each arm's logit up by how much it beats the current
average reward, scaled by its own probability. In numpy that's
`theta + eta * pi * (r - pi @ r)`. The factor $\pi_{\theta_t}(a)$ is the
problem. An arm the policy rarely picks gets a tiny push, even when it's the
best arm.

Mei et al. prove the following with $\eta = 2/5$, which suits the
objective's smoothness constant of $5/2$:

- Let $c_t = \min_{1 \le j \le t} \pi_{\theta_j}(a^*)$ be the smallest
  probability the best arm $a^*$ has had so far. Then the gap satisfies
  $(\pi^* - \pi_{\theta_t})^\top r \le 5 / (t\, c_t^2)$.
- From the uniform start, $c_t \ge 1/K$, so the gap is at most $5K^2/t$.
- The rate can't beat $1/t$. For large $t$, the gap is at least
  $\Delta^2 / (6t)$, where $\Delta$ is the reward gap between the best and
  second-best arms.

Why $1/t$ and not faster? Near the optimum, the gap is roughly the
probability left on the runner-up arm, times $\Delta$. Shrinking that
probability to $\varepsilon$ needs a logit gap of about $\log(1/\varepsilon)$.
But each step moves the logit gap by an amount proportional to
$\varepsilon$ itself. So $\varepsilon$ falls like $1/t$, and every tenfold
improvement costs ten times as many steps.

> [!WARNING]
> The PMLR version of Mei et al. prints the arbitrary-start bound without
> the constant 5. The arXiv version, and the proof, carry it. The exercise
> uses $5 / (t\, c_t^2)$.

### Worked example

Take three arms with $r = (1, 0.9, 0.1)$, uniform logits, and $\eta = 0.4$.
The average reward is $\pi^\top r = 2/3$.

- Gradient: $\frac{1}{3}(r - \frac{2}{3}) = (0.1111, 0.0778, -0.1889)$.
- New logits: $\theta_2 = 0.4 \times$ gradient $= (0.0444, 0.0311, -0.0756)$.
- New policy: $\pi_{\theta_2} = (0.3480, 0.3434, 0.3086)$, with a gap of
  $0.3121$.

Carry on, and the gap is
0.195 at $t = 10$, 0.0273 at $t = 100$, 0.00183 at $t = 1{,}000$, and
0.000169 at $t = 10{,}000$. The product $t \times$ gap stays between 1.7 and
2.7, well inside the bound of $5K^2 = 45$.

Next, start from $\theta_1 = (0, 5, 0)$, which puts 99% of the mass on the
runner-up. The best arm starts at probability 0.0066. From uniform, the gap
drops below 0.01 at $t = 226$. From the bad start, it takes until $t =
7{,}062$, about 31 times longer.

In MDPs, Mei et al. prove an $O(1/t)$ rate whose constant depends on the
smallest optimal-action probability along the way. Li et al. (2021) show some
MDPs need a number of steps exponential in the number of states and in
$1/(1 - \gamma)$.

## Natural gradient is multiplicative weights

The natural gradient lesson preconditioned the gradient with the Fisher
matrix $F$. For a tabular softmax policy, that preconditioned step has a
closed form. Agarwal et al. (2021) show that
$\theta \leftarrow \theta + \eta F^{\dagger} \nabla_\theta J$, where
$F^\dagger$ is the pseudo-inverse, gives

$$
\pi_{t+1}(a \mid s) = \frac{\pi_t(a \mid s) \exp\big(\eta A^{\pi_t}(s, a) / (1 - \gamma)\big)}{Z_t(s)}
$$

Read it as: in every state, multiply each action's probability by the
exponential of its advantage, scaled by $\eta / (1 - \gamma)$. Then divide
by $Z_t(s)$, the sum over actions, so the row sums to one. The state
distribution $d^{\pi}$ is gone. The Fisher matrix carries the same $d(s)$
factor as the gradient, and the pseudo-inverse cancels it.

In numpy, row by row:

```python
logits = eta * adv / (1 - mdp.gamma)
weights = pi * np.exp(logits - logits.max(axis=1, keepdims=True))
pi_new = weights / weights.sum(axis=1, keepdims=True)
```

This is the classic *multiplicative-weights* rule from online learning.
Three facts follow from it:

- **No more crawling.** The factor $\pi(a \mid s)$ that slowed softmax
  policy gradient is gone. A rarely chosen but good action grows by the same
  factor as a common one.
- **It's mirror descent.** In each state, the new row maximises
  $\frac{\eta}{1 - \gamma} \sum_a p(a) A(s, a) - D_{\mathrm{KL}}(p \,\Vert\, \pi(\cdot \mid s))$.
  That's "follow the advantage, but pay for moving away in KL". The
  literature calls this *policy mirror descent*.
- **Its limit is policy iteration.** As $\eta \to \infty$, all the mass goes
  to the action with the largest advantage. Agarwal et al. call the update
  "NPG as soft policy iteration".

Agarwal et al. also prove a rate. With rewards in $[0, 1]$, the uniform start,
and any $\eta > 0$,

$$
J(\pi^*) - J(\pi_T) \le \frac{\log A}{\eta T} + \frac{1}{(1 - \gamma)^2 T}
$$

Read it as: after $T$ steps, the gap to optimal is at most two terms that
both shrink like $1/T$, where $A$ is the number of actions. The bound doesn't
depend on the number of states or on how small any probability gets. Each
step also improves $J$.

> [!KEY]
> For a softmax policy, the natural gradient multiplies each action's
> probability by $\exp(\eta A / (1 - \gamma))$. That removes the
> $\pi(a \mid s)$ factor that makes plain policy gradient crawl.

### Worked example

Go back to the two-state MDP and the uniform policy. Take $\eta = 0.1$, so
$\eta / (1 - \gamma) = 1$ and the multiplier is $\exp(A)$.

- State 0: weights $0.5 e^{-0.225}$ and $0.5 e^{0.225}$, which normalise to
  $(0.3894, 0.6106)$.
- State 1: weights $0.5 e^{0.725}$ and $0.5 e^{-0.725}$, which normalise to
  $(0.8100, 0.1900)$.

One step lifts $J$ from 2.25 to 5.4249. With $\eta = 1$, one step reaches
8.9900, within 0.01 of the optimal 9.

### With an entropy bonus, the rate turns linear

Add an entropy bonus with temperature $\alpha$ to the reward, as the lesson
on maximum-entropy reinforcement learning does. Cen et al. (2022) show that
NPG then converges linearly.
At the largest step they allow, the update becomes *soft policy iteration*,
$\pi_{t+1} \propto \exp(Q^{\pi_t}_\alpha / \alpha)$, and the error shrinks by
a factor of at least $\gamma$ every step. The catch is that it converges to the
regularised optimum, which differs from the true one by at most
$\alpha \log A / (1 - \gamma)$.

> [!RECAP]
> - `advantage` is `q_from_v(mdp, v) - v[:, None]`, and it averages to zero
>   under the policy in every state.
> - The lemma weights states by the new policy's $d$; the surrogate uses the
>   old policy's $d$. Both divide by $1 - \gamma$.
> - Softmax policy gradient on a bandit adds `eta * pi * (r - pi @ r)` to the
>   logits, and its gap falls like $1/t$.
> - NPG multiplies `pi` by `exp(eta * A / (1 - gamma))` and renormalises each
>   row.

## Build it step by step

The checks test the functions in this order.

1. **`advantage(mdp, pi)`.** Returns `A`, shape `(S, A)`, from
   `rlenvs.policy_evaluation` and `rlenvs.q_from_v`. Common mistake:
   subtracting `v` without `[:, None]`.
2. **`performance_difference(mdp, pi_old, pi_new)`.** Returns a float. Weight
   the per-state sum `(pi_new * A_old).sum(axis=1)` by
   `state_distribution(mdp, pi_new)`, then divide by `1 - gamma`. Common
   mistake: using the old policy's distribution, which gives the surrogate.
3. **`surrogate_gain(mdp, pi_old, pi_new)`.** The same, with
   `state_distribution(mdp, pi_old)`. The check halves a mixing step and
   expects the error to fall by about four.
4. **`bandit_gradient(r, theta)`.** Returns shape `(K,)`. Softmax with the
   maximum subtracted, then `pi * (r - pi @ r)`.
5. **`softmax_pg_bandit(r, theta, eta, steps)`.** Returns `(steps, K)`
   policies. Record the policy *before* each update, so row 0 is the start.
   Common mistake: updating the caller's `theta` in place. Copy it first.
6. **`npg_step(mdp, pi, eta)`.** Returns the new policy, shape `(S, A)`.
   Multiply `pi` by the exponential of `eta * A / (1 - gamma)`, subtracting
   each row's maximum exponent first, and renormalise. The check sets
   `eta = 1e4` and expects the greedy policy, which overflows if you skip the
   maximum.
7. **`run_npg(mdp, eta, iters)`.** Returns `iters + 1` values of $J$,
   starting with the uniform policy.

## Cost and scaling

Everything in this lesson needs the exact advantage, which costs one linear
solve per policy. The following table shows how each piece grows with $S$
states and $A$ actions.

| Piece | Time | Memory |
|---|---|---|
| Policy evaluation | $O(S^3 + S^2 A)$ for the solve | $O(S^2 A)$ for the model |
| State distribution | $O(S^3)$, another solve | $O(S^2)$ |
| NPG step, closed form | $O(SA)$ after evaluation | $O(SA)$ |
| NPG step, explicit Fisher | $O((SA)^3)$ for the pseudo-inverse | $O((SA)^2)$ |
| Softmax PG on a bandit | $O(K)$ per step, times $O(1/\epsilon)$ steps | $O(K)$ |

With neural network policies, the closed form is gone, and TRPO
approximates $F^{-1} g$ with conjugate gradient instead.

## Where this breaks

**Exact gradients are an assumption.** Every rate here assumes the true
advantage. Sampled advantages add noise, and step sizes and rates then need a
different analysis.

**The start distribution matters.** States the policy never reaches never
improve. General bounds carry a *distribution mismatch coefficient*, the
largest ratio of the optimal policy's visitation to $\mu_0$, for this
reason.

**The constants can be astronomical.** The $1/t$ rate hides a factor of
$1/c^2$. A bad start, or an MDP designed like Li et al.'s, makes it huge.
Entropy bonuses and natural gradients are the standard fixes.

**Function approximation breaks the closed forms.** The multiplicative
weights form holds only for tabular softmax policies. With neural networks,
the guarantees depend on how well the network represents the advantage.

**The bounds are loose.** Treat them as explanations of why the methods
work, not as step-size recipes.

## Further reading

The following sources prove the results in this lesson:

- [Kakade and Langford (2002), Approximately optimal approximate reinforcement learning](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf)
  introduces the performance difference lemma and conservative policy
  iteration.
- [Schulman et al. (2015), Trust region policy optimization](https://arxiv.org/abs/1502.05477)
  turns the lemma into the KL-penalised lower bound behind TRPO.
- [Agarwal, Kakade, Lee, and Mahajan (2021), On the theory of policy gradient methods](https://arxiv.org/abs/1908.00261)
  proves global convergence of softmax policy gradient, NPG as soft policy
  iteration, and the dimension-free NPG rate.
- [Mei, Xiao, Szepesvári, and Schuurmans (2020), On the global convergence rates of softmax policy gradient methods](https://arxiv.org/abs/2005.06392)
  proves the $1/t$ upper and lower bounds and the linear rate with entropy.
- [Li, Wei, Chi, Gu, and Chen (2021), Softmax policy gradient methods can take exponential time to converge](https://arxiv.org/abs/2102.11270)
  builds the MDPs that make the constant exponential.
- [Cen, Cheng, Chen, Wei, and Chi (2022), Fast global convergence of natural policy gradient methods with entropy regularization](https://arxiv.org/abs/2007.06558)
  proves the linear rate of regularised NPG and its link to soft policy
  iteration.
- [Agarwal, Jiang, Kakade, and Sun, Reinforcement learning: theory and algorithms](https://rltheorybook.github.io/)
  collects these results in a free monograph.

## The exercise

You verify the theory on exact MDPs. Write `advantage`, then
`performance_difference` and `surrogate_gain`, which differ only in whose
state distribution weights the per-state gain. Write `bandit_gradient` and
`softmax_pg_bandit` for exact softmax policy gradient on a bandit. Finally,
write `npg_step` for the closed-form natural gradient step and `run_npg` to
iterate it from the uniform policy.

The checks compare the lemma with the exact change in $J$ on 100 random
pairs, and the surrogate's error as the step halves. They test the bandit
gradient, Mei et al.'s bound, the $1/t$ slope, and the bad start. They
compare `npg_step` with an explicit Fisher step and a mirror descent
solution, and check `run_npg` against Agarwal et al.'s bound.
