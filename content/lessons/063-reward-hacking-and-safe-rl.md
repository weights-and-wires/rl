---
title: Reward hacking, safety, and the frontier
slug: 063-reward-hacking-and-safe-rl
part: "Part 10 — Beyond text, and what can go wrong"
summary: Measure how a proxy reward diverges from the goal under best-of-n, delay the fall with ensembles, and enforce a cost budget with a Lagrange multiplier.
minutes: 45
gpu: false
objectives:
  - Explain Goodhart's law for learned rewards, and why proxy reward can climb while the true reward falls.
  - Compute the exact best-of-n policy and its KL divergence from the base policy.
  - Explain why a mean or minimum ensemble of reward models delays overoptimization, and where it stops helping.
  - Write the dual function of a constrained problem and run a primal-dual loop that finds the policy and the multiplier together.
prereqs:
  - 050-language-models-as-policies
  - 051-reward-models
  - 057-process-rewards-and-search
exercise: 063-reward-hacking-and-safe-rl
---

# Reward hacking, safety, and the frontier

You tune a chat model against a reward model and watch the reward score climb. After a thousand steps the score is far higher than where you started. Is the model better?

Not necessarily. The reward model is a stand-in for what you want, built from a few thousand human comparisons. Push hard enough on a stand-in, and the optimizer finds where it and the real goal part ways. The score keeps rising and the quality can fall. This is *reward hacking*, and it is the expected result of optimization pressure, not a rare bug.

This last chapter asks the question every earlier one leaves open: how far can you push a reward before it stops meaning what you wanted? You measure that on a world where you know the true reward, delay the damage with ensembles, and replace a hand-tuned penalty with a budget the algorithm enforces.

## What you already know

- The KL divergence $\text{KL}(p \,\|\, q) = \sum_i p_i \log(p_i/q_i)$ measures how far a policy $p$ has moved from $q$ (see [Language models as policies](/lessons/050-language-models-as-policies)).
- A reward model scores outputs with a number learned from human comparisons (see [Reward models from preferences](/lessons/051-reward-models)).
- Best-of-$n$ draws $n$ outputs and keeps the one with the highest score (see [Process rewards and test-time search](/lessons/057-process-rewards-and-search)).
- A policy that maximizes a flawed reward finds the flaw.

## A proxy and a goal

Call the reward you optimize the *proxy*, and the reward you care about the *gold* reward. In an RLHF pipeline, the proxy is the reward model and gold is what careful people would say. This is Goodhart's law: when a measure becomes a target, it ceases to be a good measure. Manheim and Garrabrant (2018) name four ways the two part: the proxy is gold plus noise, the link breaks at the extremes optimization reaches, the proxy correlates without causing, or the policy exploits it. Skalse and colleagues (2022) show that, over all stochastic policies, a proxy is unhackable only if one of the rewards is constant. So the goal is to know how far you can safely go.

## Best-of-n, computed exactly

The simplest optimizer needs no training. Best-of-$n$ draws $n$ outputs from the base policy, scores each with the proxy, and returns the best. Raising $n$ applies more pressure. On a discrete space you can write down its policy. Sort the outputs by proxy score, lowest first, and let $F_i$ be the base probability of output $i$ plus every output below it. The best of $n$ draws is at or below output $i$ exactly when all $n$ draws are, which has probability $F_i^n$. So the chance that output $i$ wins is

$$
\pi_n(i) = F_i^{\,n} - F_{i-1}^{\,n}, \qquad F_0 = 0
$$

Read it as: the chance that everything lands at or below $i$, minus the chance that everything lands strictly below it. What is left is the chance that $i$ itself is the top draw. In numpy you sort once, take a cumulative sum, and scatter the result back into the original order:

```python
order = np.argsort(proxy)
cdf = np.cumsum(base[order])
policy = np.empty_like(base)
policy[order] = cdf**n - (cdf - base[order])**n
```

Take five equally likely outputs, sorted by proxy score, with gold scores $[-1, 0, 1, 3, 0]$. The top-scoring output is a hack: the proxy loves it and gold rates it 0. The cumulative masses are $F = [0.2, 0.4, 0.6, 0.8, 1.0]$, so for $n = 2$ the policy is $[0.04, 0.12, 0.2, 0.28, 0.36]$, and its gold is $-0.04 + 0 + 0.2 + 0.84 + 0 = 1.0$. Gold is 0.6 at $n = 1$, 1.0 at $n = 2$, 1.032 at $n = 3$, and 0.942 at $n = 4$. It peaks and then falls, because more and more mass lands on the hack, and in the limit gold drops to 0.

To compare optimizers you need a ruler for how hard you pushed. The standard ruler is the KL divergence from the base policy, which for best-of-$n$ has a well-known closed form:

$$
\text{KL}(\pi_n \,\|\, \pi_{\text{ref}}) = \log n - \frac{n - 1}{n}
$$

Read it as: the distance grows like $\log n$, so doubling $n$ adds only about 0.69 nats, and moving the policy far costs exponentially many samples. The formula is exact for a continuous output space and an upper bound on a discrete one (Beirami and colleagues, 2024). In the worked example the exact KL at $n = 2$ is 0.180 against 0.193. Gao, Schulman, and Hilton (2022) plot gold against $d = \sqrt{\text{KL}}$, where the curves take simple shapes.

## Watch gold rise and fall

The exercise world has 20,000 equally likely outputs. Each output's gold score is a standard normal draw, and each proxy adds an error drawn from a Student-t distribution with three degrees of freedom. That error has *heavy tails*: most errors are small, but a few are huge, so the very highest proxy scores are mostly error. Kwa, Thomas, and Garriga-Alonso (2024) call the limit *catastrophic Goodhart*, though the reward models they measured had light-tailed error.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Best-of-n on one proxy: the proxy score climbs to 9.70 while the gold score peaks at 1.71 near n equal 64 and falls to 0.59 at n equal 4096">
  <line x1="60" y1="210" x2="560" y2="210" class="stroke-muted"/>
  <line x1="60" y1="210" x2="60" y2="20" class="stroke-muted"/>
  <line x1="60" y1="110" x2="560" y2="110" class="stroke-grid"/>
  <text x="52" y="214" text-anchor="end" class="fig-small">0</text>
  <text x="52" y="114" text-anchor="end" class="fig-small">5</text>
  <text x="52" y="24" text-anchor="end" class="fig-small">10</text>
  <text x="60" y="232" text-anchor="middle" class="fig-small">0</text>
  <text x="239" y="232" text-anchor="middle" class="fig-small">1</text>
  <text x="418" y="232" text-anchor="middle" class="fig-small">2</text>
  <text x="310" y="256" text-anchor="middle">d = sqrt(KL from the base policy)</text>
  <polyline points="60.0,209.9 138.6,196.7 202.9,185.7 256.4,176.1 301.1,167.3 342.1,158.6 377.9,149.5 410.0,139.0 440.4,126.0 468.9,109.0 495.7,86.1 518.9,55.4 542.1,16.0" fill="none" class="stroke-b"/>
  <polyline points="60.0,209.9 138.6,200.2 202.9,192.2 256.4,185.9 301.1,181.1 342.1,177.6 377.9,175.7 410.0,175.7 440.4,177.8 468.9,181.9 495.7,187.5 518.9,194.0 542.1,198.1" fill="none" class="stroke-c"/>
  <circle cx="377.9" cy="175.7" r="5" class="fill-c"/>
  <circle cx="542.1" cy="198.1" r="5" class="fill-c"/>
  <circle cx="542.1" cy="16" r="5" class="fill-b"/>
  <text x="380" y="196" class="fig-small">gold peaks: 1.71 at n = 64</text>
  <text x="430" y="216" class="fig-small">0.59 at n = 4096</text>
  <text x="300" y="40" class="fig-small">proxy: 9.70 at n = 4096</text>
  <line x1="262" y1="36" x2="294" y2="36" class="stroke-b"/>
  <line x1="262" y1="60" x2="294" y2="60" class="stroke-c"/>
  <text x="300" y="64" class="fig-small">gold</text>
</svg>
Figure 1. Best-of-n on one proxy, with $n$ from 1 to 4096. The proxy score (orange) climbs at every step, while gold (green) peaks near $n = 64$ to $128$ and falls. If you only watched the proxy, you would see steady progress the whole way.
```

Gao and colleagues fit gold for best-of-$n$ as $R(d) = d(a - b\,d)$: gold starts rising at slope $a$, and a quadratic penalty drags it back, so the peak sits at $d^* = a/(2b)$. The form is linear in $a$ and $b$, so least squares fits it. Here it gives $a = 1.93$ and $b = 0.60$, a peak at $d^* = 1.60$, near the simulated one at $d = 1.78$ to $1.96$. Because $a$ and $b$ change smoothly with reward-model size, small runs predict where a larger run turns over.

> [!KEY]
> Proxy reward climbs at every step of optimization while gold peaks and falls. Watching the proxy alone cannot tell you where the peak is.

## Delay the fall with ensembles

If one reward model has blind spots, train several and combine them. Two combinations recur (Coste and colleagues, 2023). The *mean* averages the members' scores, so independent errors partly cancel. The *minimum* scores each output by its worst member, so an output must fool every member. If $k$ members err independently, $\Pr(\min_j e_j > t) = \Pr(e > t)^k$, a much thinner tail.

The catch is the word *independent*. Eisenstein and colleagues (2023) found reward hacks that ensembles did not remove, because every member shared the error. The exercise builds this in: each member's error is 30% a shared component and 70% its own. On this world, the gold peak moves from $n = 64$ for one member to $n = 256$ for the mean and $n = 1024$ for the minimum, and the peak height rises from 1.71 to 2.26 and 2.59. The minimum cannot remove the shared part, so gold still falls eventually.

> [!KEY]
> Ensembles delay overoptimization by the amount their members disagree. Errors the members share pass straight through the mean and the minimum.

## A budget instead of a penalty

The other defense is a penalty for the bad thing, whose weight you tune by hand. A *constrained* problem states the second goal as a budget instead: earn as much reward as you can while keeping expected cost below a limit $d$ that you can defend, such as "at most one unsafe answer in a hundred". The penalty weight becomes something the algorithm finds. Write $J_r$ and $J_c$ for the expected reward and cost of a policy:

$$
\max_\pi J_r(\pi) \quad \text{subject to} \quad J_c(\pi) \le d
$$

Take one decision with two arms. Arm A pays reward 1 and costs 1. Arm B pays and costs nothing. The budget is $d = 0.3$. Always playing A is over budget, and always playing B earns 0. Only a mixture reaches the best reward of 0.3, by playing A with probability 0.3.

To find the budget's price automatically, add it to the objective with a *Lagrange multiplier* $\lambda \ge 0$:

$$
\mathcal{L}(\pi, \lambda) = J_r(\pi) - \lambda\big(J_c(\pi) - d\big), \qquad g(\lambda) = \max_\pi \mathcal{L}(\pi, \lambda)
$$

Read it as: reward, minus $\lambda$ times how far you are over budget, and $g$ is the best you can do for a fixed $\lambda$. For a one-decision problem, $g(\lambda) = \max_a (r_a - \lambda c_a) + \lambda d$. On the example, $g(\lambda) = \max(1 - \lambda, 0) + 0.3\lambda$, so $g(0) = 1$, $g(0.5) = 0.65$, $g(1) = 0.3$, and $g(2) = 0.6$. Figure 2 plots it.

```figure
<svg viewBox="0 0 600 260" role="img" aria-label="The dual function g of lambda for the one-decision example: 1 at 0, 0.65 at 0.5, 0.3 at 1, and 0.6 at 2, with its minimum equal to the constrained optimum 0.3">
  <line x1="60" y1="200" x2="560" y2="200" class="stroke-muted"/>
  <line x1="60" y1="200" x2="60" y2="30" class="stroke-muted"/>
  <line x1="60" y1="152" x2="560" y2="152" class="stroke-bad dashed"/>
  <polyline points="60,40 310,152 560,104" fill="none" class="stroke-b"/>
  <circle cx="60" cy="40" r="5" class="fill-b"/>
  <circle cx="185" cy="96" r="5" class="fill-b"/>
  <circle cx="310" cy="152" r="5" class="fill-b"/>
  <circle cx="560" cy="104" r="5" class="fill-b"/>
  <text x="72" y="44" class="fig-small">g(0) = 1</text>
  <text x="195" y="88" class="fig-small">g(0.5) = 0.65</text>
  <text x="320" y="176" class="fig-small">g(1) = 0.3, the minimum</text>
  <text x="474" y="96" class="fig-small">g(2) = 0.6</text>
  <text x="70" y="146" class="fig-small">constrained optimum 0.3</text>
  <text x="60" y="220" text-anchor="middle" class="fig-small">0</text>
  <text x="185" y="220" text-anchor="middle" class="fig-small">0.5</text>
  <text x="310" y="220" text-anchor="middle" class="fig-small">1</text>
  <text x="435" y="220" text-anchor="middle" class="fig-small">1.5</text>
  <text x="560" y="220" text-anchor="middle" class="fig-small">2</text>
  <text x="310" y="244" text-anchor="middle">multiplier lambda</text>
</svg>
Figure 2. The dual function stays at or above the constrained optimum 0.3 and touches it at $\lambda = 1$. There both arms score 0 under $r - \lambda c$, so the penalized problem is indifferent between them, which is what lets the optimum mix.
```

That $g$ stays above 0.3 everywhere is *weak duality*: any feasible policy has $J_c - d \le 0$, so the penalty can only add to its reward. That its minimum equals 0.3 is *strong duality*. The minimizing $\lambda^*$ is also a price: the optimal reward you gain per unit of extra budget.

Without a model you cannot compute $g$, so you run two gradient steps at once. *Primal-dual* climbs $\mathcal{L}$ in the policy parameters and moves $\lambda$ to punish violation:

$$
\theta \leftarrow \theta + \eta_\theta \nabla_\theta \mathcal{L}(\theta, \lambda), \qquad \lambda \leftarrow \max\big(0,\ \lambda + \eta_\lambda (J_c(\pi_\theta) - d)\big)
$$

Read it as: improve the policy for the current penalty, then raise the penalty if you are over budget and lower it if you are under. The $\max(0, \cdot)$ keeps $\lambda$ non-negative, because a negative multiplier would reward cost. For a softmax policy over arms with $u = r - \lambda c$, the first step has an exact gradient, $p_a(u_a - \sum_b p_b u_b)$. Plain descent-ascent circles the answer, because any $\lambda < 1$ makes "always A" best and pushes cost over budget, and any $\lambda > 1$ does the opposite. The *average* of the iterates converges, to playing A with probability 0.3 and $\lambda$ near 1. Safe RLHF (Dai and colleagues, 2023) applies this recipe to a chat model with a learned cost model.

> [!RECAP]
> - Proxy reward can rise while gold falls, and no non-constant proxy is unhackable.
> - Best-of-$n$ is exact on a discrete space, $\pi_n(i) = F_i^n - F_{i-1}^n$, with KL at most $\log n - (n-1)/n$.
> - The gold curve $d(a - bd)$ peaks at $d^* = a/(2b)$. A mean or minimum ensemble moves it outward, as far as members' errors are independent.
> - A budget turns a hand-tuned penalty into a multiplier. The dual function $g(\lambda)$ lies above the constrained optimum, and primal-dual finds both the policy and $\lambda$.

## Build it step by step

The checks test these functions in order.

1. **`bon_kl(n)`.** Returns `np.log(n) - (n - 1) / n`. Common mistake: integer division, so convert `n` to float.
2. **`best_of_n_policy(proxy, n, base)`.** Use `np.argsort`, `np.cumsum`, and a scatter `policy[order] = ...`. Common mistake: returning the probabilities in sorted order.
3. **`kl_divergence(p, q)`.** Sum `p * log(p / q)` over entries with `p > 0`, so `0 * log 0` does not give `nan`.
4. **`goodhart_curve(gold, proxy, base, ns)`.** For each `n`, build the policy, take the square root of its KL for `d`, and dot it with gold and with the proxy.
5. **`ensemble_proxy(proxies, how)`.** `proxies.mean(axis=0)` or `proxies.min(axis=0)`.
6. **`dual_value(reward, cost, budget, lam)`.** The maximum over arms of the penalized reward, plus `lam * budget`.
7. **`dual_step(lam, expected_cost, budget, lr)`.** One projected step. Common mistake: forgetting the `max(0, ...)`.
8. **`primal_dual(reward, cost, budget)`.** Compute the cost from the policy before the update, and average the policy and `lam` over the second half. Common mistake: returning the last iterate.

## Where this breaks

**Heavy tails.** A KL penalty protects you only if the proxy's error is light-tailed, and whether real reward models are is unmeasured.

**Hidden hacks.** Training against a monitor of a model's reasoning can teach it to hide the hack (Baker and colleagues, 2025).

**Budgets need estimates.** If the cost estimate is itself a learned model, you have moved the proxy problem rather than removed it.

## What comes next

The course ends here, but threads run on. ScaleRL (Khatri and colleagues, 2025) found that RL compute curves are S-shaped in log compute and can be extrapolated, but only after the bend. Test-time RL rewards agreement with a majority vote on unlabeled questions (TTRL, Zuo and colleagues, 2025), and INTUITOR (Zhao and colleagues, 2025) rewards the model's own confidence. Both can lock in whatever the model already believes. Whether RL discovers new reasoning or sharpens what the base model had is contested (Yue and colleagues, 2025).

## Further reading

- [Gao, Schulman, and Hilton (2022), Scaling laws for reward model overoptimization](https://arxiv.org/abs/2210.10760). The gold-proxy setup and both functional forms.
- [Kwa, Thomas, and Garriga-Alonso (2024), Catastrophic Goodhart](https://arxiv.org/abs/2407.14503). Why heavy-tailed error defeats a KL penalty.
- [Coste and colleagues (2023), Reward model ensembles help mitigate overoptimization](https://arxiv.org/abs/2310.02743). Worst-case ensembles for best-of-n and PPO.
- [Dai and colleagues (2023), Safe RLHF](https://arxiv.org/abs/2310.12773). A cost model and a Lagrangian PPO for language models.
- [Khatri, Madaan, and colleagues (2025), The art of scaling RL compute for LLMs](https://arxiv.org/abs/2510.13786). The sigmoidal compute fit.

Researchers are still working on these open questions:

- Do reward models trained on long reasoning traces, or LLM judges, have light-tailed error?
- Can a monitor detect reward hacking without creating pressure to hide it?

## The exercise

You build the Goodhart curve on a world of 20,000 outputs with a known gold reward and three noisy proxies from the given `make_world`. Write `bon_kl`, `best_of_n_policy`, `kl_divergence`, and `goodhart_curve`, then `ensemble_proxy` for mean and minimum ensembles. Then write `dual_value`, `dual_step`, and `primal_dual` for the one-decision constrained problem.

The checks confirm the best-of-$n$ policy on hand-sized examples, and that the proxy rises with every $n$ while gold peaks inside the range and falls. They check that the ensembles move the peak later and raise it, that $g$ matches the four hand values and stays above the linear-program optimum on a three-arm problem while touching it, and that `primal_dual` plays the costly arm with probability 0.3 with a multiplier near 1.

> [!KEY]
> Optimizing a proxy pushes the policy toward the places where proxy and goal disagree, so the proxy climbs while the goal peaks and falls. You can predict the peak, delay it with ensembles, and enforce a budget with a multiplier, but you cannot make it vanish. That answers the question this course ends on: push a reward only as far as you can measure what it still means.
