---
title: Safe and constrained reinforcement learning
slug: 50-safe-and-constrained-rl
part: "Part 10 — Safety and the frontier"
summary: Solve a constrained MDP exactly as a linear program, then reach it with a Lagrangian primal-dual method and a Safe RLHF cost model.
minutes: 60
gpu: false
objectives:
  - Write a constrained MDP as a linear program over occupancy measures and read the policy back from its solution.
  - Explain why the constrained optimum can be stochastic when no unconstrained optimum needs to be.
  - Compute the dual function, confirm the zero duality gap, and read the Lagrange multiplier as the price of the budget.
  - Run projected gradient descent-ascent on the Lagrangian and explain why you average its iterates.
  - Build the Safe RLHF cost-model loss and its gradient.
prereqs:
  - 04-markov-decision-processes
  - 15-policy-gradient-theorem
  - 34-reward-models
exercise: 50-safe-and-constrained-rl
---

# Safe and constrained reinforcement learning

Most tasks come with more than one goal. A delivery robot must be fast and
must not hit people. A chat assistant must be helpful and must not give
dangerous instructions. The usual trick is to fold everything into one
reward, with a penalty for the bad thing. Then you tune the penalty by hand,
and every change to the task means tuning it again.

A *constrained* problem states the second goal as a budget instead: earn as
much reward as you can, while keeping expected cost below a limit you choose.
The limit has a meaning you can defend, such as "at most one unsafe answer in
a hundred". The penalty weight becomes something the algorithm finds for
itself.

In this lesson you solve a small constrained problem exactly, as a linear
program. You then reach the same answer with the Lagrangian method that
large systems use, and you build the cost model from Safe RLHF, which applies
the same idea to language models.

## What you already know

- A *policy* $\pi(a \mid s)$ is an `(S, A)` array whose rows sum to one. Its
  *return* is the discounted sum of rewards $\sum_t \gamma^t r_t$.
- The Bellman expectation equation gives a policy's exact value by one
  linear solve. You did this in *Markov decision processes*.
- The discounted visitation distribution weights each state by how often and
  how early the policy reaches it.
- For a softmax policy, the policy gradient weights each state-action pair by
  how often it's visited and by its *advantage* $A^\pi(s, a) = Q^\pi(s, a) -
  V^\pi(s)$. You derived this in *The policy gradient theorem*.
- A reward model learns scores from pairwise comparisons with the
  Bradley-Terry loss. You trained one in *Reward models*.

## Add a budget to the MDP

A *constrained MDP* (CMDP) is an ordinary MDP with a second table, a cost
$c(s, a) \ge 0$ for taking action $a$ in state $s$, and a budget $d$. Write
$J_r(\pi)$ for the expected discounted reward from the start distribution
$\mu_0$, and $J_c(\pi)$ for the same thing with cost in place of reward. The
problem is

$$
\max_\pi \; J_r(\pi) \quad \text{subject to} \quad J_c(\pi) \le d .
$$

Read it as: among all policies whose expected discounted cost stays within
the budget, find the one with the most reward. The constraint is on the
*expected cumulative* cost, not on every step or every episode. A policy can
exceed the budget in one episode, provided it stays under it on average.

## Write the problem as a linear program

The trick that makes CMDPs tractable is to stop thinking about the policy and
think about where it spends its time. The *occupancy measure* of a policy is

$$
\rho^\pi(s, a) = \sum_{t=0}^{\infty} \gamma^t \Pr(s_t = s, a_t = a) ,
$$

with $s_0$ drawn from $\mu_0$. Read it as: the discounted number of times the
policy takes $a$ in $s$. Its entries sum to $1/(1-\gamma)$. Every return is
now a dot product with it: $J_r = \sum_{s,a} \rho(s, a)\, r(s, a)$, and $J_c$
is the same with $c$. In numpy you find the state part with one linear solve,
then multiply by the policy:

```python
P_pi = np.einsum("sa,sat->st", pi, mdp.P)
state = np.linalg.solve(np.eye(S) - gamma * P_pi.T, mdp.mu0)
rho = state[:, None] * pi
```

Not every table of non-negative numbers is an occupancy measure. It has to
obey a *flow constraint* at every state $s'$:

$$
\sum_a \rho(s', a) = \mu_0(s') + \gamma \sum_{s, a} P(s' \mid s, a)\, \rho(s, a) .
$$

Read it as: the discounted time spent in $s'$ equals the chance of starting
there, plus the discounted flow arriving from every state. Any $\rho \ge 0$
that satisfies all $S$ of these equations is the occupancy measure of the
stationary policy $\pi(a \mid s) = \rho(s, a) / \sum_{a'} \rho(s, a')$.

So the CMDP becomes a *linear program* (LP). Maximise $\sum \rho\, r$,
subject to $\sum \rho\, c \le d$, the $S$ flow equations, and $\rho \ge 0$.
The objective, the budget, and the flow are all linear in $\rho$. That's why
constrained MDPs are a solved problem when the model is known (Altman,
1999).

### Worked example

Take one state, two actions, and $\gamma = 0$, so the problem is a single
decision. Action A pays reward 1 and costs 1. Action B pays and costs
nothing. The budget is $d = 0.3$. The flow constraint reads $\rho_A + \rho_B
= 1$, the objective is $\rho_A$, and the budget is $\rho_A \le 0.3$. The LP's
answer is $\rho_A = 0.3$, $\rho_B = 0.7$, for a reward of 0.3. The policy is
"play A with probability 0.3".

## See why the optimum can be random

Every unconstrained MDP has a deterministic optimal policy. The worked
example shows a constrained one needn't. Always playing A costs 1, over the
budget. Always playing B is feasible but earns 0. Only the mixture reaches
0.3.

The LP explains how much randomness you need. Some optimum always sits at a
corner of the feasible region, and a corner of a region cut out by $S$ equations
and one active inequality has at most $S + 1$ nonzero entries. Every visited
state needs at least one, so at most one state takes two actions. With $k$ active
constraints, at most $k$ states randomise.

The exercise builds a four-state CMDP whose budget sits halfway between the
cost of the reward-optimal policy and the lowest reachable cost. Its LP
optimum earns 2.88 and randomises in exactly one state. The best of all 81
deterministic policies that fit the budget earns only 1.91.

> [!KEY]
> A constrained MDP is a linear program over occupancy measures. Its optimum
> randomises in at most one state per active constraint, and no deterministic
> policy may come close to it.

## Price the constraint with a Lagrange multiplier

An LP needs the full model $P$, which you rarely have. The method that scales
turns the budget into a penalty whose weight adjusts itself. Define the
*Lagrangian*

$$
\mathcal{L}(\pi, \lambda) = J_r(\pi) - \lambda \big(J_c(\pi) - d\big),
\qquad \lambda \ge 0 .
$$

Read it as: reward, minus $\lambda$ times how far you're over budget. The
number $\lambda$ is the *Lagrange multiplier*. For a fixed $\lambda$,
maximising $\mathcal{L}$ over $\pi$ is an ordinary MDP with reward $r -
\lambda c$. That maximum is the *dual function*

$$
g(\lambda) = \max_\pi \mathcal{L}(\pi, \lambda) = \max_\pi J_{r - \lambda c}(\pi) + \lambda d .
$$

Read it as: solve the penalised MDP, then add back $\lambda d$. In numpy
it's value iteration on a shifted reward:

```python
shifted = rlenvs.TabularMDP(P=mdp.P, R=mdp.R - lam * cost, gamma=mdp.gamma, mu0=mdp.mu0)
g = rlenvs.value_iteration(shifted) @ mdp.mu0 + lam * budget
```

Two facts make $g$ useful. First, *weak duality*: $g(\lambda)$ is at least
the constrained optimum for every $\lambda \ge 0$. Any feasible policy has
$J_c - d \le 0$, so the penalty can only add to its reward. Second, for a
CMDP the gap closes: the smallest $g(\lambda)$ equals the constrained
optimum. This *strong duality* follows from LP duality. Paternain et al.
(2019) showed that the gap stays at zero for rich parameterised policies too,
under mild conditions.

In the worked example, $g(\lambda) = \max(1 - \lambda, 0) + 0.3\lambda$.
That's $g(0) = 1$, $g(0.5) = 0.65$, $g(1) = 0.3$, and $g(2) = 0.6$. The
minimum is at $\lambda^* = 1$, where $g$ equals the LP's 0.3. At $\lambda^*$,
both arms score 0 under $r - \lambda c$, so the penalised problem is
indifferent between them. That indifference is what lets the optimum mix.

The multiplier also has a price tag. $\lambda^*$ is how much optimal reward
you gain per unit of extra budget. In the example, raising $d$ from 0.3 to
0.31 raises the best reward by 0.01, a slope of 1.

> [!TIP]
> `scipy.optimize.linprog` reports the multiplier for you. For a constraint
> written as `A_ub @ x <= b_ub` in a minimisation, `-res.ineqlin.marginals`
> is $\lambda^*$. The checks use it to test your dual function.

## Run primal-dual

Without a model, you can't solve the LP or evaluate $g$ exactly. The
*primal-dual* method runs two gradient steps at once. It climbs
$\mathcal{L}$ in the policy parameters $\theta$, and it moves $\lambda$ in
the direction that punishes violation:

$$
\theta \leftarrow \theta + \eta_\theta \nabla_\theta \mathcal{L}(\theta, \lambda),
\qquad
\lambda \leftarrow \max\!\big(0,\; \lambda + \eta_\lambda (J_c(\pi_\theta) - d)\big) .
$$

Read it as: improve the policy for the current penalty, then raise the
penalty if you're over budget and lower it if you're under. The $\max(0,
\cdot)$ keeps $\lambda$ non-negative. A negative multiplier would reward
cost.

For a tabular softmax policy, the policy gradient theorem gives the first
step exactly. The gradient in $\theta(s, a)$ is the occupancy times the
advantage under the penalised reward, $\rho(s, a)\, A_{r - \lambda c}(s, a)$:

```python
v = rlenvs.policy_evaluation(shifted, pi)
advantage = rlenvs.q_from_v(shifted, v) - v[:, None]
grad = occupancy(mdp, pi) * advantage
```

Plain gradient descent-ascent circles the answer rather than landing on it.
In the worked example, any $\lambda < 1$ makes "always A" best and pushes
cost over budget, so $\lambda$ rises. Any $\lambda > 1$ makes "always B"
best, so cost falls and $\lambda$ drops. The last iterate keeps swinging
between the two. The *average* of the iterates converges.

Average the occupancy measures, not the logits. If you run each of $K$
policies with probability $1/K$, the occupancy is the mean of theirs, and
the stationary policy read off that mean has the same occupancy. On the
exercise's four-state CMDP, 4,000 steps with $\eta_\theta = 1$ and
$\eta_\lambda = 0.1$ give an average $\lambda$ of 0.851 over the second half,
against the LP's 0.846. The averaged occupancy earns 2.816 at a cost of 4.617,
against the LP's 2.882 at the budget of 4.693.

Stooke, Achiam, and Abbeel (2020) damp the swing another way. They treat
$\lambda$ as the output of a PID controller on the violation, adding
proportional and derivative terms to the integral that plain ascent already
is.

> [!WARNING]
> Report the averaged policy, not the last iterate. A primal-dual run can
> look feasible in its final step and still overshoot on average, or the
> reverse.

## Apply it to language models: Safe RLHF

Safe RLHF (Dai et al., 2023) applies the CMDP recipe to a chat model. People
label helpfulness and harmlessness separately. A reward model $R_\phi$
learns helpfulness with the usual Bradley-Terry loss. A *cost model*
$C_\psi$ learns harm, with positive scores for harmful answers.

The cost model sees two kinds of label. For a pair, $y_w$ is the *more
harmful* answer and $y_l$ the less harmful one. Each answer also carries a
sign $s(y)$, which is $+1$ if people judged it harmful and $-1$ if safe. The
loss adds a ranking term to two classification terms:

$$
\mathcal{L}_C = -\log \sigma\big(C_w - C_l\big)
- \log \sigma\big(s_w C_w\big) - \log \sigma\big(s_l C_l\big) ,
$$

averaged over pairs, where $C_w = C_\psi(y_w, x)$ and $C_l = C_\psi(y_l, x)$.
Read it as: rank the more harmful answer higher, push harmful answers to
positive cost, and push safe answers to negative cost. The sign terms pin
zero as the boundary between safe and harmful, which the ranking term alone
can't do, because it sees only differences.

Its gradient follows from $\frac{d}{dz}[-\log \sigma(z)] = -\sigma(-z)$:

$$
\frac{\partial \mathcal{L}_C}{\partial C_w} = -\sigma\big(-(C_w - C_l)\big) - s_w\, \sigma(-s_w C_w) ,
$$

and for $C_l$ the ranking term flips sign: $+\sigma(-(C_w - C_l)) - s_l\,
\sigma(-s_l C_l)$. Divide both by the number of pairs for the mean. Compute
$\log \sigma(z)$ as `-np.logaddexp(0, -z)`, which never overflows.

For one pair with $C_w = 1$, $C_l = -1$, $s_w = +1$, and $s_l = -1$, the loss
is $-\log\sigma(2) - 2\log\sigma(1) = 0.1269 + 2 \times 0.3133 = 0.7535$. The
gradient for $C_w$ is $-\sigma(-2) - \sigma(-1) = -0.1192 - 0.2689 =
-0.3881$. Raising the harmful answer's cost lowers the loss, as it should.

With both models trained, Safe RLHF solves

$$
\max_\theta \; \mathbb{E}\big[R_\phi(y, x)\big]
\quad \text{subject to} \quad \mathbb{E}\big[C_\psi(y, x)\big] + d \le 0 ,
$$

where $y$ is sampled from the policy and $d$ is a threshold hyperparameter.
It uses PPO for $\theta$ and a primal-dual update for $\lambda$, taken in log
space so the multiplier stays positive: $\ln \lambda \leftarrow \ln \lambda +
\alpha\, \lambda\, \big(\mathbb{E}[C_\psi] + d\big)$. The expected cost comes
from a moving average of the cost model's scores during training.

> [!KEY]
> Safe RLHF is a CMDP with learned reward and cost models. The multiplier
> rises while the policy's expected cost is over the threshold and falls
> once it's under.

> [!RECAP]
> - The occupancy measure turns every return into a dot product, and the
>   CMDP into a linear program with $S$ flow equations and one budget row.
> - The LP optimum can randomise, in at most one state per active
>   constraint.
> - $g(\lambda) = \max_\pi J_{r - \lambda c}(\pi) + \lambda d$ bounds the
>   optimum from above and touches it at $\lambda^*$.
> - Primal-dual steps $\theta$ up the Lagrangian and $\lambda$ up the
>   violation, clipped at 0. Average the occupancies it visits.
> - The Safe RLHF cost loss ranks pairs and classifies each answer by its
>   harm sign.

## Build it step by step

The checks test the functions in this order. `make_cmdp` and
`softmax_policy` are given.

1. **`occupancy(mdp, pi)`.** Returns `(S, A)`. Build $P_\pi$ with
   `np.einsum`, solve with the *transpose* $P_\pi^\top$, and multiply by
   `pi`. Common mistake: solving with $P_\pi$ instead of its transpose, which
   gives values instead of visitation. The check compares the total with
   $1/(1-\gamma)$ and the state marginal with `rlenvs.state_distribution`.
2. **`policy_from_occupancy(rho)`.** Divide each row by its sum, and use a
   uniform row where the sum is zero. `np.where` avoids dividing by zero.
3. **`cmdp_lp(mdp, cost, budget)`.** Returns `c`, `A_ub`, `b_ub`, `A_eq`, and
   `b_eq` for `linprog`, with $\rho$ flattened so column `s * A + a` holds
   $\rho(s, a)$. Common mistake: forgetting that `linprog` minimises, so `c`
   is minus the reward. The checks solve your program and test the one-state
   example and the four-state CMDP.
4. **`dual_function(mdp, cost, budget, lam)`.** Value iteration on `R - lam *
   cost`, read at `mu0`, plus `lam * budget`. The check tests weak duality on
   a grid of $\lambda$ and equality at the LP's multiplier.
5. **`lagrangian_grad(mdp, cost, budget, theta, lam)`.** Returns the gradient
   and $J_c - d$. Common mistake: computing the advantage under $r$ alone.
   Finite differences catch it.
6. **`primal_dual(...)`.** The loop, with $\lambda$ clipped at zero and the
   occupancies averaged over the second half of the run.
7. **`cost_model_loss` and `cost_model_grad`.** Use the stable $\log\sigma$.
   Common mistake: dropping the $1/n$ from the mean in the gradient.

## Cost and scaling

The exact methods scale with the size of the state and action spaces. The
sampled ones scale with the size of the models. The following table
summarises the costs:

| Method | Needs | Cost per solve or step |
|---|---|---|
| Occupancy LP | The full model $P$ | An LP with $SA$ variables and $S + 1$ rows |
| Dual function | The full model | One value iteration, $O(S^2 A)$ per sweep |
| Exact primal-dual | The full model | Two linear solves, $O(S^3)$, per step |
| Sampled primal-dual | Rollouts | One policy-gradient step, plus a cost estimate |
| Safe RLHF | Reward and cost models | PPO with a second reward model, so one extra forward pass per response |

The LP is the right tool when $S$ is small and $P$ is known. Primal-dual is
the one that scales, because each step is one ordinary policy-gradient step
plus a scalar update. Each extra constraint adds one multiplier and one cost
estimate.

## Where this breaks

**The last iterate oscillates.** Plain descent-ascent circles the solution.
Average the iterates, lower $\eta_\lambda$, or use a PID update for
$\lambda$.

**The constraint holds only on average, and only at the end.** A CMDP bounds
expected cost. It says nothing about the worst episode, and nothing about
the episodes spent learning. Safe exploration methods, such as constrained
policy optimisation (Achiam et al., 2017), try to stay feasible at every
update.

**Per-step and cumulative budgets differ.** Constraining the average cost
per step and constraining the total discounted cost give different
problems. Check which one your specification means before you set $d$.

**The cost model can be hacked too.** A learned cost is a proxy, like any
reward model, and *Reward hacking and overoptimisation* applies to it in
full. Moskovitz et al. (2023) set each reward component's threshold near the
point where it stops tracking human judgement. A threshold that's too loose
invites the policy to exploit the proxy.

**$\lambda$ can blow up.** If the budget is infeasible, no policy satisfies
it, and $\lambda$ grows without limit. Check feasibility first, for example
by minimising cost alone.

## Further reading

- [Constrained Markov decision processes](https://doi.org/10.1201/9781315140223)
  (Altman, 1999). The occupancy-measure LP, the existence of optimal
  stationary policies, and why they can need randomisation.
- [Constrained reinforcement learning has zero duality gap](https://arxiv.org/abs/1910.13393)
  (Paternain et al., 2019). Strong duality for parameterised policies, which
  justifies primal-dual methods beyond the tabular case.
- [Constrained policy optimization](https://arxiv.org/abs/1705.10528)
  (Achiam et al., 2017). A trust-region method that aims to keep every
  update feasible.
- [Responsive safety in reinforcement learning by PID Lagrangian methods](https://arxiv.org/abs/2007.03964)
  (Stooke, Achiam, and Abbeel, 2020). Why the multiplier oscillates and how
  PID control damps it.
- [Safe RLHF](https://arxiv.org/abs/2310.12773) (Dai et al., 2023).
  Separate helpfulness and harmlessness labels, a cost model, and a
  Lagrangian PPO for language models.
- [Confronting reward model overoptimization with constrained RLHF](https://arxiv.org/abs/2310.04373)
  (Moskovitz et al., 2023). Constraint thresholds placed at the points where
  proxy rewards stop tracking gold.

Researchers are working on open questions such as the following:

- How do you set a threshold when the constraint is itself a learned,
  overoptimisable cost model?
- Can PID or zero-duality-gap guarantees survive when the cost estimate is
  stale, as in asynchronous LLM training?
- Can a policy stay within budget during training as well as at the end?

## The exercise

You solve a constrained MDP three ways and build the Safe RLHF cost model.
Write `occupancy` for the discounted occupancy of a policy,
`policy_from_occupancy` to read a policy back, and `cmdp_lp` to build the
occupancy LP in `scipy.optimize.linprog` form. Then write `dual_function`
for $g(\lambda)$, `lagrangian_grad` for the exact softmax gradient of the
Lagrangian, and `primal_dual` for projected gradient descent-ascent. Finish
with `cost_model_loss` and `cost_model_grad`.

The checks solve your LP with `linprog`. On the one-state example it must
play the costly action with probability 0.3, with multiplier 1. On a
four-state CMDP it must be feasible and stochastic, and it must beat every
feasible deterministic policy. Your dual function must match hand values,
stay above the LP value, and meet it at the LP's multiplier. Your gradient
must match finite differences. Primal-dual must bring the average $\lambda$
within 25 percent of the LP's multiplier, and the averaged policy within 12
percent of its value. The cost-model loss must match the hand value and
survive scores of $\pm 800$, and its gradient must match finite differences.
