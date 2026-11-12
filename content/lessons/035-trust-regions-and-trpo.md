---
title: Natural gradients and trust regions
slug: 035-trust-regions-and-trpo
part: "Part 5 — Policy gradients"
summary: Measure policy steps by how much behavior changes, solve for the natural gradient with Fisher-vector products, and take KL-bounded steps.
minutes: 40
gpu: false
objectives:
  - Explain why a step of fixed length in parameter space can change a policy's behavior a little or a lot.
  - Compute Fisher-vector products by double backpropagation through the mean KL, and check them against an explicit Fisher matrix.
  - Solve F x = g with conjugate gradient using only matrix-vector products.
  - Implement TRPO's natural step and the backtracking line search that keeps the measured KL inside the trust region.
prereqs:
  - 032-where-the-gradient-comes-from
  - 033-baselines-and-actor-critic
exercise: 035-trust-regions-and-trpo
---

# Natural gradients and trust regions

A policy gradient tells you which way is uphill. It does not tell you how far to go. Plain gradient ascent moves the parameters a fixed distance, set by a learning rate. But you do not care about the parameters. You care about the policy's behavior, and the same distance in parameters can change behavior a lot or barely at all.

Think of directions on a map drawn at different scales in different places. "Walk two centimeters north" is a short stroll in one corner and a long hike in another. Near a uniform policy, a small step in the logits changes the action probabilities noticeably. Near a nearly deterministic policy, the same step does almost nothing, and a larger one can overshoot and wreck the policy in one update.

The fix is to measure steps in behavior. The *natural gradient* takes the steepest step among all steps that change the policy by the same small amount, measured by the KL divergence. *Trust region policy optimization* (TRPO) turns that idea into an algorithm for neural networks. You build its three working parts: a product with the Fisher matrix that never builds the matrix, conjugate gradient to solve with it, and a line search that checks every step before accepting it.

## What you already know

- The policy gradient $\nabla J$ is a sum over states and actions of the advantage times the score $\nabla \log \pi_\theta(a \mid s)$.
- The *KL divergence* $\operatorname{KL}(p \,\|\, q) = \sum_a p(a) \log (p(a) / q(a))$ measures how different two distributions are. It is zero only when they match, and it is not symmetric.
- The score averages to zero over actions drawn from the policy.
- `torch.autograd.grad` returns gradients, and with `create_graph=True` you can differentiate those gradients again.

## See why parameter distance misleads

For two actions, the policy is $\pi = [p, 1 - p]$ with $p = 1/(1 + e^{-z})$, where $z$ is the logit gap. Move $z$ by one unit and watch $p$ (Figure 1). Starting at $z = 0$, the policy is uniform and $p$ goes from 0.5 to 0.731. Starting at $p = 0.99$, the same unit step moves $p$ from 0.990 to 0.996. Same distance, 37 times less change.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Probability of the first action against the logit gap: a unit step changes it by 0.231 near 0.5 and by 0.006 near 0.99">
<line x1="70" y1="230.0" x2="560" y2="230.0" class="stroke-grid"/><text x="62" y="234.0" class="fig-small" text-anchor="end">0</text>
<line x1="70" y1="135.0" x2="560" y2="135.0" class="stroke-grid"/><text x="62" y="139.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="70" y1="40.0" x2="560" y2="40.0" class="stroke-grid"/><text x="62" y="44.0" class="fig-small" text-anchor="end">1</text>
<text x="70.0" y="250" class="fig-small" text-anchor="middle">-1</text>
<text x="192.5" y="250" class="fig-small" text-anchor="middle">1</text>
<text x="315.0" y="250" class="fig-small" text-anchor="middle">3</text>
<text x="437.5" y="250" class="fig-small" text-anchor="middle">5</text>
<text x="560.0" y="250" class="fig-small" text-anchor="middle">7</text>
<text x="315" y="272" text-anchor="middle">logit gap z</text><text x="70" y="24" class="fig-small">probability p</text>
<polyline points="70.0,178.9 76.1,175.1 82.2,171.1 88.4,167.0 94.5,162.7 100.6,158.3 106.8,153.8 112.9,149.1 119.0,144.5 125.1,139.7 131.2,135.0 137.4,130.3 143.5,125.5 149.6,120.9 155.8,116.2 161.9,111.7 168.0,107.3 174.1,103.0 180.2,98.9 186.4,94.9 192.5,91.1 198.6,87.5 204.8,84.0 210.9,80.7 217.0,77.6 223.1,74.7 229.2,71.9 235.4,69.3 241.5,67.0 247.6,64.7 253.8,62.6 259.9,60.7 266.0,59.0 272.1,57.3 278.2,55.8 284.4,54.4 290.5,53.1 296.6,52.0 302.8,50.9 308.9,49.9 315.0,49.0 321.1,48.2 327.2,47.4 333.4,46.8 339.5,46.1 345.6,45.6 351.8,45.1 357.9,44.6 364.0,44.2 370.1,43.8 376.2,43.4 382.4,43.1 388.5,42.8 394.6,42.5 400.8,42.3 406.9,42.1 413.0,41.9 419.1,41.7 425.3,41.6 431.4,41.4 437.5,41.3 443.6,41.2 449.8,41.0 455.9,40.9 462.0,40.9 468.1,40.8 474.3,40.7 480.4,40.6 486.5,40.6 492.6,40.5 498.8,40.5 504.9,40.4 511.0,40.4 517.1,40.3 523.2,40.3 529.4,40.3 535.5,40.3 541.6,40.2 547.8,40.2 553.9,40.2 560.0,40.2" class="stroke-ink" fill="none"/>
<line x1="131.2" y1="135.0" x2="192.5" y2="135.0" class="stroke-b"/>
<line x1="192.5" y1="135.0" x2="192.5" y2="91.1" class="stroke-b"/>
<circle cx="131.2" cy="135.0" r="4" class="fill-b"/><circle cx="192.5" cy="91.1" r="4" class="fill-b"/>
<text x="143.2" y="175.0" class="fig-small" text-anchor="start">z: 0 → 1, p: 0.500 → 0.731</text>
<line x1="412.7" y1="41.9" x2="474.0" y2="41.9" class="stroke-a"/>
<line x1="474.0" y1="41.9" x2="474.0" y2="40.7" class="stroke-a"/>
<circle cx="412.7" cy="41.9" r="4" class="fill-a"/><circle cx="474.0" cy="40.7" r="4" class="fill-a"/>
<text x="406.7" y="91.9" class="fig-small" text-anchor="end">z: 4.6 → 5.6, p: 0.990 → 0.996</text>
</svg>
Figure 1. The probability of the first action as the logit gap z grows. Notice that a step of one unit in z changes the probability by 0.231 at the uniform policy but by only 0.006 where the policy is 99% certain, so equal steps in parameters are unequal steps in behavior.
```

So parameter distance is the wrong ruler. The right ruler asks how different the two policies are as distributions, which is what the KL divergence measures. Averaged over the states you visit, the change in behavior from $\theta$ to $\theta + \Delta$ is the mean KL, $\bar D(\Delta) = \mathbb{E}_s[\operatorname{KL}(\pi_\theta(\cdot \mid s) \,\|\, \pi_{\theta + \Delta}(\cdot \mid s))]$. For small steps it is a quadratic:

$$
\bar D(\Delta) \approx \tfrac{1}{2}\, \Delta^\top F\, \Delta, \qquad
F = \mathbb{E}_{s,\, a \sim \pi_\theta}\big[\nabla_\theta \log \pi_\theta(a \mid s)\, \nabla_\theta \log \pi_\theta(a \mid s)^\top\big]
$$

Read it as: $F$, the *Fisher information matrix*, averages the outer product of the score with itself. The KL of a small step is half the step's squared length measured with $F$ instead of the identity. There is no linear term, because the KL is at its minimum, zero, when $\Delta = 0$. Directions where $F$ is large are directions where the policy is sensitive. For a softmax over logits $z$ in one state, $F = \operatorname{diag}(\pi) - \pi \pi^\top$.

## Take the steepest step of a given size

Ask for the step that raises the objective most, to first order, among all steps whose KL is at most $\delta$. Maximize $g^\top \Delta$ subject to $\tfrac12 \Delta^\top F \Delta \le \delta$, where $g = \nabla J$. The answer is

$$
\Delta^* = \sqrt{\frac{2\delta}{g^\top F^{-1} g}}\; F^{-1} g
$$

Read it right to left. $F^{-1} g$ is the *natural gradient*: the ordinary gradient, reweighted so that sensitive directions shrink and insensitive ones grow. The square root scales it so the quadratic KL estimate equals $\delta$ exactly. Amari (1998) introduced the natural gradient, and Kakade (2001) brought it to policy gradients.

Work one case. One state, two actions, rewards $[0, 1]$, and a policy that almost always picks the worse action, $\pi = [0.99, 0.01]$. Take the logits as parameters.

- The gradient is $g = \pi \odot (r - \pi \cdot r) = [-0.0099, 0.0099]$, which is tiny.
- The Fisher matrix is $0.0099 \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$. It is singular, because adding a constant to both logits changes nothing, so use the pseudo-inverse.
- The natural gradient is $F^{+} g = [-0.5, 0.5]$. It points the same way as $g$ and is 50 times longer.
- With $\delta = 0.01$, the scale is $\sqrt{0.02 / 0.0099} = 1.4213$, so the step is $[-0.7107, 0.7107]$. It moves $\pi$ to $[0.9598, 0.0402]$, which quadruples the chance of the better action. Plain gradient ascent would need a learning rate of about 72 to take that step.
- At the uniform policy, the same rewards give a step of only $[-0.1414, 0.1414]$. The natural step adapts to where the policy is.

## Multiply by the Fisher matrix without building it

A network with a million parameters has a Fisher matrix with a trillion entries. You cannot build it, let alone invert it. TRPO needs only products $F v$, and automatic differentiation provides them.

The trick is that $F$ is the Hessian of the mean KL at the current parameters. Hold the old policy fixed at $\theta_{\text{old}}$, define $\bar D(\theta) = \mathbb{E}_s[\operatorname{KL}(\pi_{\theta_{\text{old}}} \,\|\, \pi_\theta)]$, and differentiate twice at $\theta_{\text{old}}$. The product with any vector $v$ is a gradient of a gradient:

$$
F v = \nabla_\theta \Big( \big(\nabla_\theta \bar D(\theta)\big)^\top v \Big) \Big|_{\theta = \theta_{\text{old}}}
$$

Read it as: take the gradient of the KL, dot it with $v$ to get a scalar, then take the gradient of that scalar. The cost is about two backward passes.

```python
logits = logits_fn(params)
kl = mean_kl(logits.detach(), logits)        # old policy held fixed
grad = torch.autograd.grad(kl, params, create_graph=True)[0]
fv = torch.autograd.grad(grad @ v, params)[0]
```

> [!WARNING]
> Do not forget `detach()` on the old logits. Without it, both arguments of the KL move together, the KL is identically zero, and every Fisher-vector product is zero.

Implementations add *damping*, a small multiple of the identity: $(F + \xi I) v$ with $\xi$ around 0.1. It makes the system solvable when $F$ is singular.

## Solve with conjugate gradient

You need $x = F^{-1} g$, and you can only multiply by $F$. *Conjugate gradient* (CG) solves $A x = b$ for a symmetric positive definite $A$ using one product $A p$ per iteration. It picks search directions that do not undo each other's progress, so in exact arithmetic it finishes in at most $n$ iterations for an $n$ by $n$ system. TRPO stops after about 10, which gives a useful direction. Each iteration updates the estimate $x$, the residual $r = b - Ax$, and the direction $p$:

$$
\alpha = \frac{r^\top r}{p^\top A p}, \qquad x \leftarrow x + \alpha p, \qquad r_{\text{new}} = r - \alpha A p, \qquad p \leftarrow r_{\text{new}} + \frac{r_{\text{new}}^\top r_{\text{new}}}{r^\top r}\, p
$$

Read it as: step along $p$ by the amount that minimizes the error along that line, update the residual without a second product, then build the next direction from the new residual plus a correction that keeps it independent of the earlier ones. Start from $x = 0$, so $r = p = b$. Once you have $x \approx F^{-1} g$, the step is $\sqrt{2\delta / (x^\top F x)}\, x$, which needs one more product.

## Check every step with a line search

TRPO, from Schulman, Levine, Moritz, Jordan, and Abbeel (2015), applies this to a batch collected by the current policy. It maximizes a *surrogate* inside a KL trust region:

$$
\max_\theta\; L(\theta) = \hat{\mathbb{E}}\left[\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)}\, \hat A\right] \quad \text{subject to} \quad \hat{\mathbb{E}}_s\big[\operatorname{KL}(\pi_{\theta_{\text{old}}} \,\|\, \pi_\theta)\big] \le \delta
$$

Read it as: reweight each sampled advantage by how much more likely the new policy makes that action than the policy that sampled it. That ratio is an *importance weight*. At $\theta_{\text{old}}$ every ratio is one, and the gradient of $L$ is exactly the policy gradient. The usual trust-region size is $\delta = 0.01$. [Why policy optimization works](/lessons/039-why-policy-optimization-works) explains why this surrogate is worth maximizing.

One update has four steps:

1. Record $\log \pi_{\theta_{\text{old}}}(a \mid s)$ and the old logits for every sample.
2. Compute $g = \nabla L$ at $\theta_{\text{old}}$.
3. Solve $(F + \xi I) x = g$ with CG, and scale $x$ to the full step $s$.
4. Try $\theta_{\text{old}} + s$, then $+ s/2$, $+ s/4$, and so on. Accept the first candidate whose *measured* mean KL is at most $\delta$ and whose surrogate beats the surrogate at $\theta_{\text{old}}$.

The line search matters because the quadratic KL is only an estimate. In the worked example, the full step's quadratic estimate is 0.01, but its measured KL is 0.0167, which breaks the constraint. Halving the step gives a measured KL of 0.0032, and it still doubles the better action's probability from 0.01 to 0.0201. The line search accepts that (Figure 2).

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="Measured KL and quadratic KL estimate against the fraction of the natural step, with the trust region limit 0.01: the full step breaks it and half the step meets it">
<line x1="70" y1="240.0" x2="560" y2="240.0" class="stroke-grid"/><text x="62" y="244.0" class="fig-small" text-anchor="end">0</text>
<line x1="70" y1="140.0" x2="560" y2="140.0" class="stroke-bad dashed"/><text x="62" y="144.0" class="fig-small" text-anchor="end">0.01</text>
<line x1="70" y1="40.0" x2="560" y2="40.0" class="stroke-grid"/><text x="62" y="44.0" class="fig-small" text-anchor="end">0.02</text>
<text x="560" y="134.0" class="fig-small" text-anchor="end">trust region δ = 0.01</text>
<text x="70.0" y="260" class="fig-small" text-anchor="middle">0</text>
<text x="192.5" y="260" class="fig-small" text-anchor="middle">0.25</text>
<text x="315.0" y="260" class="fig-small" text-anchor="middle">0.5</text>
<text x="437.5" y="260" class="fig-small" text-anchor="middle">0.75</text>
<text x="560.0" y="260" class="fig-small" text-anchor="middle">1</text>
<polyline points="70.0,240.0 79.8,240.0 89.6,239.8 99.4,239.6 109.2,239.3 119.0,239.0 128.8,238.5 138.6,237.9 148.4,237.2 158.2,236.5 168.0,235.6 177.8,234.6 187.6,233.5 197.4,232.3 207.2,231.0 217.0,229.6 226.8,228.1 236.6,226.4 246.4,224.6 256.2,222.6 266.0,220.6 275.8,218.4 285.6,216.0 295.4,213.5 305.2,210.9 315.0,208.1 324.8,205.1 334.6,202.0 344.4,198.7 354.2,195.2 364.0,191.6 373.8,187.7 383.6,183.7 393.4,179.5 403.2,175.1 413.0,170.5 422.8,165.7 432.6,160.6 442.4,155.4 452.2,149.9 462.0,144.2 471.8,138.2 481.6,132.0 491.4,125.5 501.2,118.8 511.0,111.8 520.8,104.6 530.6,97.1 540.4,89.2 550.2,81.1 560.0,72.7" class="stroke-b" fill="none"/>
<polyline points="70.0,240.0 79.8,240.0 89.6,239.8 99.4,239.6 109.2,239.4 119.0,239.0 128.8,238.6 138.6,238.0 148.4,237.4 158.2,236.8 168.0,236.0 177.8,235.2 187.6,234.2 197.4,233.2 207.2,232.2 217.0,231.0 226.8,229.8 236.6,228.4 246.4,227.0 256.2,225.6 266.0,224.0 275.8,222.4 285.6,220.6 295.4,218.8 305.2,217.0 315.0,215.0 324.8,213.0 334.6,210.8 344.4,208.6 354.2,206.4 364.0,204.0 373.8,201.6 383.6,199.0 393.4,196.4 403.2,193.8 413.0,191.0 422.8,188.2 432.6,185.2 442.4,182.2 452.2,179.2 462.0,176.0 471.8,172.8 481.6,169.4 491.4,166.0 501.2,162.6 511.0,159.0 520.8,155.4 530.6,151.6 540.4,147.8 550.2,144.0 560.0,140.0" class="stroke-a dashed" fill="none"/>
<circle cx="560.0" cy="72.7" r="5" class="fill-bad"/><text x="550.0" y="62.7" class="fig-small" text-anchor="end">rejected: 0.0167</text>
<circle cx="315.0" cy="208.1" r="5" class="fill-c"/><text x="325.0" y="198.1" class="fig-small" text-anchor="start">accepted: 0.0032</text>
<line x1="90" y1="62" x2="120" y2="62" class="stroke-b"/><text x="128" y="66" class="fig-small">measured KL</text>
<line x1="90" y1="82" x2="120" y2="82" class="stroke-a dashed"/><text x="128" y="86" class="fig-small">quadratic estimate δ f²</text>
<text x="315" y="284" text-anchor="middle">fraction f of the full natural step</text><text x="70" y="24" class="fig-small">KL(old, new)</text>
</svg>
Figure 2. The KL of the worked example as the step shrinks. Notice that the quadratic estimate hits 0.01 at the full step while the measured KL is 0.0167, so the line search rejects f = 1 and accepts f = 0.5.
```

> [!KEY]
> TRPO takes the natural-gradient step sized for a quadratic KL of $\delta$, then backtracks until the measured KL is within $\delta$ and the surrogate improves. CG and Fisher-vector products make this work for millions of parameters.

> [!RECAP]
> - `mean_kl(old, new)` averages $\sum_a \pi_{\text{old}} (\log \pi_{\text{old}} - \log \pi_{\text{new}})$ over rows.
> - $F v$ is the gradient of $(\nabla \bar D)^\top v$, with the old logits detached and `create_graph=True` on the first gradient.
> - CG solves $F x = g$ with one product per iteration.
> - The full step is $\sqrt{2\delta / (x^\top F x)}\, x$.
> - The line search halves the step until the measured KL is within $\delta$ and the surrogate improves.

## Build it step by step

The checks test the functions in this order. The contextual bandit, its linear softmax policy, `flat_grad`, `sample_batch`, and `expected_reward` are given. Everything is float64.

1. **`mean_kl(logits_old, logits_new)`.** Takes two `(N, A)` tensors and returns a scalar. Use `torch.log_softmax` on both. The common mistake is to swap the arguments, because KL is not symmetric.
2. **`fisher_vector_product(logits_fn, params, v, damping)`.** Returns `(P,)`. Make a leaf copy of `params`, compute the logits once, and differentiate the KL between their detached copy and themselves twice.
3. **`conjugate_gradient(Avp, b, iters, tol)`.** Returns `(P,)`. The common mistake is to call `Avp` twice per iteration. Update the residual with the `Ap` you already have.
4. **`surrogate(logits, actions, advantages, logp_old)`.** Returns a scalar. Gather the sampled actions' log-probabilities with `.gather(-1, actions[:, None])`, and take the ratio with `torch.exp(logp - logp_old)`.
5. **`natural_step(g, fvp, max_kl, cg_iters)`.** Returns `(P,)`. The common mistake is to scale by $\sqrt{\delta / \ldots}$ and forget the 2.
6. **`line_search(surrogate_fn, kl_fn, params, full_step, max_kl, ...)`.** Returns `(new_params, fraction)`. Wrap it in `torch.no_grad()`, and return `(params.clone(), 0.0)` if nothing is accepted.
7. **`trpo_update(...)`.** Ties the pieces together as the four steps above. Index the logits with `contexts` so the surrogate, KL, and Fisher products all use the batch's states.

## Where this breaks

**It is a second-order method.** Each update needs CG, double backpropagation, and a line search. That is harder to implement than an optimizer step and does not combine well with layers shared between policy and value function, or with dropout. [Proximal policy optimization](/lessons/036-proximal-policy-optimization) gets most of the benefit with first-order steps.

**The KL goes one way.** TRPO constrains $\operatorname{KL}(\pi_{\text{old}} \,\|\, \pi_\theta)$. Reversing the arguments gives a valid-looking number but a different constraint and a different Hessian.

## Further reading

- [Kakade (2001), A natural policy gradient](https://proceedings.neurips.cc/paper/2001/hash/4b86abe48d358ecf194c56c69108433e-Abstract.html) applies the Fisher matrix of the policy to policy gradients. [Paper page](https://papers.ifkash.dev/npg/).
- [Schulman, Levine, Moritz, Jordan, and Abbeel (2015), Trust region policy optimization](https://arxiv.org/abs/1502.05477) introduces TRPO, its Fisher-vector products, and the backtracking line search.
- [Shewchuk (1994), An introduction to the conjugate gradient method without the agonizing pain](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf) derives conjugate gradient with pictures.
- [Martens (2014), New insights and perspectives on the natural gradient method](https://arxiv.org/abs/1412.1193) connects the Fisher matrix to second-order optimization.

## The exercise

You build a TRPO update for a linear softmax policy on a small contextual bandit. Write `mean_kl`, `fisher_vector_product`, `conjugate_gradient`, `surrogate`, `natural_step`, `line_search`, and `trpo_update`.

The checks compare your Fisher-vector products with an explicit Fisher matrix and with the Hessian of the mean KL, and your conjugate gradient with `np.linalg.solve`. They check that the surrogate's gradient at the old parameters is the policy gradient, that the natural step equals the closed form with $\tfrac12 s^\top F s = \delta$, and that the line search finds the known answers on toy functions. Finally, 10 TRPO updates must each keep the measured KL within `max_kl` and together raise the expected reward by more than 0.25.
