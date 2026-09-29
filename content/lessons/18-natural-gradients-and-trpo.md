---
title: Natural gradients and TRPO
slug: 18-natural-gradients-and-trpo
part: "Part 4 — Policy optimisation"
summary: Measure policy steps by how much behaviour changes, solve for the natural gradient with Fisher-vector products, and take KL-bounded steps.
minutes: 65
gpu: false
objectives:
  - Explain why a step of fixed length in parameter space can change a policy's behaviour a little or a lot.
  - Compute Fisher-vector products by double backpropagation through the mean KL, and check them against an explicit Fisher matrix.
  - Solve F x = g with conjugate gradient using only matrix-vector products.
  - Implement TRPO's natural step and the backtracking line search that keeps the measured KL inside the trust region.
prereqs:
  - 15-policy-gradient-theorem
  - 16-actor-critic
exercise: 18-natural-gradients-and-trpo
---

# Natural gradients and TRPO

A policy gradient tells you which way is uphill. It doesn't tell you how far
to go. Plain gradient ascent moves the parameters a fixed distance, scaled by
a learning rate. But the parameters aren't what you care about. You care
about the policy's behaviour, and the same distance in parameters can change
behaviour a lot or barely at all.

Think of giving directions on a map drawn at different scales in different
places. "Walk two centimetres north" means a short stroll in one corner and a
long hike in another. A single learning rate has the same problem. Near a
uniform policy, a small step in the logits changes the action probabilities
noticeably. Near a nearly deterministic policy, the same step does almost
nothing, and a larger one can overshoot and wreck the policy in a single
update.

The fix is to measure steps in terms of behaviour. The *natural gradient*
takes the steepest step among all steps that change the policy by the same
small amount, measured by the KL divergence. Trust Region Policy Optimisation
(TRPO) turns that idea into an algorithm for neural networks. In this lesson
you build its three working parts: a product with the Fisher matrix that never
builds the matrix, conjugate gradient to solve with it, and a line search that
checks every step before accepting it.

## What you already know

- The policy gradient $\nabla J$ is a sum over states and actions of the
  advantage times the score $\nabla \log \pi_\theta(a \mid s)$.
- The *KL divergence*
  $\operatorname{KL}(p \,\|\, q) = \sum_a p(a) \log (p(a) / q(a))$ measures
  how different two distributions are. It's zero only when they match, and
  it isn't symmetric.
- The score averages to zero over actions drawn from the policy.
- Advantages, from *Generalised advantage estimation*, say how much better
  each sampled action was than usual.
- PyTorch's `torch.autograd.grad` returns gradients, and with
  `create_graph=True` you can differentiate those gradients again.

## Measure steps by how much the policy changes

Suppose you move the parameters from $\theta$ to $\theta + \Delta$. Averaged
over the states you visit, the change in behaviour is the mean KL,
$\bar D(\Delta) = \mathbb{E}_s[\operatorname{KL}(\pi_\theta(\cdot \mid s) \,\|\,
\pi_{\theta + \Delta}(\cdot \mid s))]$. For small steps it's a quadratic:

$$
\bar D(\Delta) \approx \tfrac{1}{2}\, \Delta^\top F\, \Delta, \qquad
F = \mathbb{E}_{s,\, a \sim \pi_\theta}\big[\nabla_\theta \log \pi_\theta(a
\mid s)\, \nabla_\theta \log \pi_\theta(a \mid s)^\top\big]
$$

Read it as: $F$, the *Fisher information matrix*, averages the outer product
of the score with itself. The KL of a small step is half the step's squared
length measured with $F$ instead of the identity. There's no linear term,
because the KL is at its minimum, zero, when $\Delta = 0$.

Directions where $F$ is large are directions where the policy is sensitive. A
small step there changes behaviour a lot. Directions where $F$ is small change
behaviour very little. For a softmax over logits $z$ in one state, the Fisher
matrix with respect to $z$ is $\operatorname{diag}(\pi) - \pi \pi^\top$.

```python
F = np.diag(pi) - np.outer(pi, pi)     # Fisher of a softmax, in its logits
kl_estimate = 0.5 * step @ F @ step
```

## Take the steepest step of a given size

Ask for the step that raises the objective most, to first order, among all
steps whose KL is at most $\delta$. Maximise $g^\top \Delta$ subject to
$\tfrac12 \Delta^\top F \Delta \le \delta$, where $g = \nabla J$. The answer
is

$$
\Delta^* = \sqrt{\frac{2\delta}{g^\top F^{-1} g}}\; F^{-1} g
$$

Read it right to left. $F^{-1} g$ is the *natural gradient*: the ordinary
gradient, reweighted so that sensitive directions shrink and insensitive ones
grow. The square root scales it so that the quadratic KL estimate equals
$\delta$ exactly. Amari (1998) introduced the natural gradient, and Kakade
(2001) brought it to policy gradients.

The natural gradient has a property plain gradients lack. To first order, it
doesn't depend on how you parameterise the policy. Two parameterisations that
describe the same policies give the same small change in behaviour, because
both measure the step with the KL.

### Worked example

Take one state, two actions, and rewards $[0, 1]$. The policy is nearly
certain to pick the worse action: $\pi = [0.99, 0.01]$. Take the logits as the
parameters. The ordinary gradient is
$g = \pi \odot (r - \pi \cdot r) = [-0.0099, 0.0099]$, which is tiny. The
Fisher matrix is
$\operatorname{diag}(\pi) - \pi\pi^\top = 0.0099 \begin{bmatrix} 1 & -1 \\ -1
& 1 \end{bmatrix}$.

This $F$ is singular, because adding a constant to both logits changes
nothing. Use its pseudo-inverse, `np.linalg.pinv`, or add damping, as in the
next section. The natural gradient is $F^{+} g = [-0.5, 0.5]$. The two
vectors point the same way. The natural one is 50 times longer, because it
isn't shrunk by the policy's near-certainty.

With $\delta = 0.01$, the scale is $\sqrt{0.02 / (g^\top F^{+} g)} =
\sqrt{0.02 / 0.0099} = 1.4213$, so the step is $[-0.7107, 0.7107]$. It moves
$\pi$ to $[0.9598, 0.0402]$, which quadruples the chance of the better action.
Plain gradient ascent would need a learning rate of about 72 to take the same
step. At the uniform policy, the same rewards give a step of only
$[-0.1414, 0.1414]$. The natural step adapts to where the policy is.

## Multiply by the Fisher matrix without building it

A policy network with a million parameters has a Fisher matrix with a
trillion entries. You can't build it, let alone invert it. TRPO never does.
It needs only products $F v$, and those come from automatic differentiation.

The trick is that $F$ is the Hessian of the mean KL at the current
parameters. Hold the old policy fixed at $\theta_{\text{old}}$, define
$\bar D(\theta) = \mathbb{E}_s[\operatorname{KL}(\pi_{\theta_{\text{old}}}
\,\|\, \pi_\theta)]$, and differentiate twice at $\theta = \theta_{\text{old}}$.
The product with any vector $v$ is a gradient of a gradient:

$$
F v = \nabla_\theta \Big( \big(\nabla_\theta \bar D(\theta)\big)^\top v
\Big) \Big|_{\theta = \theta_{\text{old}}}
$$

Read it as: take the gradient of the KL, dot it with $v$ to get a scalar,
then take the gradient of that scalar. The result is the Hessian times $v$,
at the cost of about two backward passes. In PyTorch:

```python
logits = logits_fn(params)
kl = mean_kl(logits.detach(), logits)        # old policy held fixed
grad = torch.autograd.grad(kl, params, create_graph=True)[0]
fv = torch.autograd.grad(grad @ v, params)[0]
```

The `detach()` matters. The first argument is the old policy, a constant. The
second is the policy being differentiated. The KL's gradient at this point is
zero, because the KL is at its minimum. Its Hessian isn't zero, and that's
what you want.

> [!WARNING]
> Don't forget `detach()` on the old logits. Without it, both arguments of the
> KL move together, the KL is identically zero, and every Fisher-vector
> product comes out zero.

Implementations add *damping*, a small multiple of the identity:
$(F + \xi I) v$ with $\xi$ around 0.1. It makes the system solvable when $F$
is singular, as in the worked example, and it keeps the solver stable when
$F$ is estimated from a small batch.

> [!KEY]
> The Fisher matrix is the Hessian of the mean KL from a frozen copy of the
> policy. Double backpropagation gives $F v$ for any $v$ at the cost of two
> backward passes, without ever forming $F$.

## Solve with conjugate gradient

You need $x = F^{-1} g$, and you can only multiply by $F$. *Conjugate
gradient* (CG) solves $A x = b$ for a symmetric positive definite matrix $A$
using one product $A p$ per iteration. It picks a sequence of search
directions that don't undo each other's progress, and in exact arithmetic it
reaches the exact answer in at most $n$ iterations for an $n$ by $n$ system.
TRPO stops after about 10 iterations, which is enough for a useful direction.

Each iteration updates the estimate $x$, the residual $r = b - Ax$, and the
search direction $p$:

$$
\alpha = \frac{r^\top r}{p^\top A p}, \qquad x \leftarrow x + \alpha p,
\qquad r_{\text{new}} = r - \alpha A p, \qquad p \leftarrow r_{\text{new}} +
\frac{r_{\text{new}}^\top r_{\text{new}}}{r^\top r}\, p
$$

Read it as: step along $p$ by the amount that minimises the error along that
line, update the residual without a second product, then build the next
direction from the new residual plus a correction that keeps it independent
of the previous ones. Start from $x = 0$, so $r = p = b$. Stop early once
$r^\top r$ drops below a small tolerance.

```python
x = torch.zeros_like(b); r = b.clone(); p = r.clone(); rr = r @ r
for _ in range(iters):
    Ap = Avp(p)
    alpha = rr / (p @ Ap)
    # update x and r, then p, then rr
```

Once you have $x \approx F^{-1} g$, the step is $\sqrt{2\delta / (x^\top F x)}\,
x$. That needs one more product, $F x$, and $x^\top F x = g^\top F^{-1} g$
when $x$ is exact.

## Assemble TRPO

TRPO, from Schulman, Levine, Moritz, Jordan, and Abbeel (2015), applies this
machinery to a batch of experience from the current policy
$\pi_{\theta_{\text{old}}}$. It maximises a *surrogate objective* inside a KL
*trust region*:

$$
\max_\theta\; L(\theta) = \hat{\mathbb{E}}\left[\frac{\pi_\theta(a \mid
s)}{\pi_{\theta_{\text{old}}}(a \mid s)}\, \hat A\right] \quad \text{subject
to} \quad \hat{\mathbb{E}}_s\big[\operatorname{KL}(\pi_{\theta_{\text{old}}}
\,\|\, \pi_\theta)\big] \le \delta
$$

Read it as: reweight each sampled advantage by how much more likely the new
policy makes that action than the policy that sampled it. That ratio is an
*importance weight*. It corrects for the batch having come from the old
policy. At $\theta = \theta_{\text{old}}$ every ratio is one, $L$ equals the
mean advantage, and the gradient of $L$ is exactly the policy gradient
$\hat{\mathbb{E}}[\hat A \nabla \log \pi]$. The standard trust-region size is
$\delta = 0.01$.

Why is this surrogate worth maximising? Kakade and Langford (2002) proved
that the true improvement is at least the surrogate's improvement minus a
penalty, for mixtures of the old and new policies. Schulman et al. (2015)
extended the bound to any pair of policies, with a penalty that grows with the
largest KL between them. So while the policy stays close, raising the
surrogate raises the return. The penalty's coefficient is so large that it
would force tiny steps, so TRPO replaces it with the constraint $\delta$. *Why policy optimisation works* derives the
exact identity behind this bound.

One TRPO update has four steps:

1. Record $\log \pi_{\theta_{\text{old}}}(a \mid s)$ for every sample, and the
   old logits for the KL.
2. Compute $g = \nabla L$ at $\theta_{\text{old}}$.
3. Solve $(F + \xi I) x = g$ with CG, and scale $x$ to the full step $s$.
4. Try $\theta_{\text{old}} + s$, then $+ s/2$, $+ s/4$, and so on. Accept the
   first candidate whose *measured* mean KL is at most $\delta$ and whose
   surrogate is higher than at $\theta_{\text{old}}$.

The line search matters because the quadratic KL is only an estimate. In the
worked example, the full natural step's quadratic estimate was 0.01, but its
actual KL is 0.0167, which breaks the constraint. Halving the step gives an
actual KL of 0.0032 and still doubles the better action's probability, from
0.01 to 0.0201. The line search accepts that. Schulman's original code also
checks that the improvement is a reasonable fraction of what the linear model
predicted. The exercise uses the simpler test: the surrogate must improve.

> [!KEY]
> TRPO takes the natural-gradient step sized for a quadratic KL of $\delta$,
> then backtracks until the measured KL is within $\delta$ and the surrogate
> improves. CG and Fisher-vector products make this work for millions of
> parameters.

> [!TIP]
> Check a Fisher-vector product on a model small enough to enumerate. Build
> $F$ exactly as $\sum_a \pi(a \mid s)\, \text{score}\, \text{score}^\top$
> from `torch.autograd.functional.jacobian`, averaged over states, and compare
> $F v$ with your product to $10^{-10}$ in float64.

> [!RECAP]
> - `mean_kl(old, new)` averages $\sum_a \pi_{\text{old}} (\log
>   \pi_{\text{old}} - \log \pi_{\text{new}})$ over rows.
> - $F v$ is the gradient of $(\nabla \bar D)^\top v$, with the old logits
>   detached and `create_graph=True` on the first gradient.
> - CG solves $F x = g$ with one product per iteration.
> - The full step is $\sqrt{2\delta / (x^\top F x)}\, x$.
> - The line search halves the step until the measured KL is within $\delta$
>   and the surrogate improves.

## Build it step by step

The checks test the functions in this order. The contextual bandit, its
linear softmax policy, `flat_grad`, `sample_batch`, and `expected_reward` are
given. Everything is float64.

1. **`mean_kl(logits_old, logits_new)`.** Takes two `(N, A)` tensors and
   returns a scalar. Use `torch.log_softmax` on both, so shifted logits give
   the same answer. Common mistake: swapping the arguments. KL isn't
   symmetric, and TRPO's KL is from the old policy to the new one.
2. **`fisher_vector_product(logits_fn, params, v, damping)`.** Returns `(P,)`.
   Make a leaf copy of `params` with `requires_grad_(True)`, compute the
   logits once, and differentiate the KL between their detached copy and
   themselves twice. The checks compare the result with an explicit Fisher
   matrix and with the KL's Hessian.
3. **`conjugate_gradient(Avp, b, iters, tol)`.** Returns `(P,)`. Common
   mistake: calling `Avp` twice per iteration. You can update the residual
   with the `Ap` you already have. The check solves 10 by 10 and 5 by 5 systems
   and compares with `np.linalg.solve`.
4. **`surrogate(logits, actions, advantages, logp_old)`.** Returns a scalar.
   Gather the log-probability of each sampled action with
   `.gather(-1, actions[:, None])`. The ratio is
   `torch.exp(logp - logp_old)`.
5. **`natural_step(g, fvp, max_kl, cg_iters)`.** Returns `(P,)`. Common
   mistake: scaling by $\sqrt{\delta / \ldots}$ and forgetting the 2. The
   check requires $\tfrac12 s^\top F s = \delta$.
6. **`line_search(surrogate_fn, kl_fn, params, full_step, max_kl, ...)`.**
   Returns `(new_params, fraction)`. Wrap it in `torch.no_grad()`. Return
   `(params.clone(), 0.0)` if nothing is accepted.
7. **`trpo_update(...)`.** Ties the pieces together as the four numbered
   steps describe. Index the logits with `contexts` so the surrogate, the KL,
   and the Fisher matrix all use the batch's states. The last check runs 10
   updates and requires every measured KL to stay within `max_kl`.

## Cost and scaling

For a policy with $P$ parameters and a batch of $N$ states, the following
table compares the costs of each approach to the natural step:

| Approach | Time | Memory |
|---|---|---|
| Build and invert $F$ | $O(N P^2 + P^3)$ | $O(P^2)$ |
| One Fisher-vector product | About two backward passes, $O(N P)$ | $O(P)$ plus the graph |
| CG with $k$ iterations | $k$ products, $O(k N P)$ | $O(P)$ |
| Line search with $m$ tries | $m$ forward passes, $O(m N P)$ | $O(P)$ |

With $k = 10$ on the full batch, the Fisher-vector products dominate the
cost of an update. Schulman et al. (2015) computed them on 10% of the batch
instead, which made all of them together cost about as much as the gradient.
The Fisher matrix only sets the metric, so a subsample estimates it well
enough.

## Where this breaks

**A second-order method per update.** Each update needs CG, double
backpropagation, and a line search. That's harder to implement than a plain
optimiser step, and it doesn't combine well with architectures that share
layers between the policy and the value function, or with dropout. PPO, in
the next lesson, gets most of the benefit with first-order steps.

**The KL goes the wrong way.** TRPO constrains
$\operatorname{KL}(\pi_{\text{old}} \,\|\, \pi_\theta)$. Reversing the
arguments still gives a valid-looking number, but a different constraint and
a different Hessian.

**The Fisher matrix is only estimated.** With a small batch, $F$ is noisy and
can be badly conditioned. CG then amplifies the noise in poorly measured
directions. Damping and a modest number of CG iterations help.

**Stale or mismatched batches.** The gradient, the Fisher-vector products, and
the line search must all use the same old policy. Recomputing $\log
\pi_{\text{old}}$ after the step has started makes every ratio one.

**The trust region is only local.** A KL of 0.01 per update is a small change,
but TRPO's guarantee rests on a bound that needs every state's KL to be
small, not only the average. In practice the average works, but a few states
can still change a lot.

## Further reading

- [Amari (1998), Natural gradient works efficiently in learning](https://doi.org/10.1162/089976698300017746)
  introduces the natural gradient and its invariance to parameterisation.
- [Kakade (2001), A natural policy gradient](https://proceedings.neurips.cc/paper/2001/hash/4b86abe48d358ecf194c56c69108433e-Abstract.html)
  applies the Fisher matrix of the policy to policy gradients.
- [Kakade and Langford (2002), Approximately optimal approximate reinforcement learning](https://dl.acm.org/doi/10.5555/645531.656005)
  proves the policy improvement bound that TRPO builds on.
- [Schulman, Levine, Moritz, Jordan, and Abbeel (2015), Trust region policy optimization](https://arxiv.org/abs/1502.05477)
  introduces TRPO, its Fisher-vector products, and the backtracking line
  search.
- [Shewchuk (1994), An introduction to the conjugate gradient method without the agonizing pain](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)
  derives conjugate gradient from scratch with pictures.
- [Martens (2014), New insights and perspectives on the natural gradient method](https://arxiv.org/abs/1412.1193)
  connects the Fisher matrix to second-order optimisation and discusses its
  practical approximations.

## The exercise

You build a TRPO update for a linear softmax policy on a small contextual
bandit. Write `mean_kl`, `fisher_vector_product` by double backpropagation,
`conjugate_gradient`, the importance-weighted `surrogate`, `natural_step`,
the backtracking `line_search`, and `trpo_update`, which combines them.

The checks verify `mean_kl` against a hand computation. They compare your
Fisher-vector products with an explicit Fisher matrix built by enumerating
actions, and with the Hessian of the mean KL. They compare conjugate gradient
with `np.linalg.solve`. They confirm that the surrogate's gradient at the old
parameters is the policy gradient. They check that the natural step equals
the closed form and that $\tfrac12 s^\top F s = \delta$. They run the line
search on toy functions with known answers. Finally, 10 TRPO updates must
each keep the measured KL within `max_kl` and together raise the expected
reward by more than 0.25.
