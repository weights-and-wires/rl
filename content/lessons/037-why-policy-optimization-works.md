---
title: Why policy optimization works
slug: 037-why-policy-optimization-works
part: "Part 5 — Policy gradients"
summary: Verify the performance difference lemma exactly, measure softmax policy gradient's slow 1/t rate, and show NPG is multiplicative weights.
minutes: 40
gpu: false
objectives:
  - Verify the performance difference lemma to rounding error, and say why the surrogate that TRPO and PPO optimize is only first-order correct.
  - Measure softmax policy gradient's 1/t rate on a bandit against the bound of Mei et al., and explain why a bad start makes it crawl.
  - Show that the natural policy gradient for a softmax policy is a multiplicative-weights step, and check the dimension-free bound of Agarwal et al.
prereqs:
  - 030-where-the-gradient-comes-from
  - 033-trust-regions-and-trpo
  - 034-proximal-policy-optimization
exercise: 037-why-policy-optimization-works
---

# Why policy optimization works

The last four chapters built policy gradient methods that work in practice. REINFORCE, actor-critic, TRPO, and PPO all nudge a policy uphill. None of them asked the harder questions. Does the nudge ever reach the best policy, given that the objective is not concave? How fast? And why does a trust region help?

This chapter answers those on problems small enough to solve exactly. One identity, the performance difference lemma, sits under almost every result, and you verify it to fifteen decimal places. Then you see that plain softmax policy gradient reaches the optimum but can crawl, and that the natural gradient turns into a multiplicative-weights rule that escapes the crawl.

## What you already know

- $J(\pi) = \sum_s \mu_0(s) V^\pi(s)$ is a policy's expected discounted return from the start distribution $\mu_0$.
- The advantage $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ says how much better $a$ is than the policy's average in $s$.
- The policy gradient theorem weights states by their discounted visitation.
- TRPO and PPO maximize a surrogate built from the old policy's samples.

## Compare two policies exactly

How much better is a candidate policy $\pi'$ than an old policy $\pi$? Run $\pi'$, and at each state it visits, score its action with the *old* policy's advantage. Add up those scores, and you get the exact improvement.

The lemma needs the *normalized discounted state distribution* $d^{\pi'}(s) = (1 - \gamma) \sum_{t \ge 0} \gamma^t \Pr(s_t = s)$ for states visited by $\pi'$ from $\mu_0$. It sums to one, and `rlenvs.state_distribution` returns it. Kakade and Langford (2002) state the lemma as

$$
J(\pi') - J(\pi) = \frac{1}{1 - \gamma} \sum_s d^{\pi'}(s) \sum_a \pi'(a \mid s)\, A^{\pi}(s, a)
$$

Read it from the inside out. In each state, average the old advantage over the new policy's actions. Weight each state by how often the new policy visits it. Divide by $1 - \gamma$ to undo the normalization. The proof idea takes three sentences: write the gain as a sum along the new policy's trajectory of $r_t + \gamma V^\pi(s_{t+1}) - V^\pi(s_t)$, discounted by $\gamma^t$; the $V^\pi$ terms telescope and leave $J(\pi') - J(\pi)$; and each bracket averages to $A^\pi(s_t, a_t)$.

```python
d_new = rlenvs.state_distribution(mdp, pi_new)
per_state = (pi_new * advantage(mdp, pi_old)).sum(axis=1)
gain = d_new @ per_state / (1 - mdp.gamma)
```

Work one case. Two states: in each, action 0 stays and action 1 switches to the other state. Staying in state 1 pays 1, everything else pays 0, $\gamma = 0.9$, and every episode starts in state 0. Under the uniform policy, $V = (2.25, 2.75)$ and the advantages are

$$
A = \begin{bmatrix} -0.225 & 0.225 \\ 0.725 & -0.725 \end{bmatrix}
$$

Switch to the greedy policy: leave state 0, then stay in state 1. Its return is $J = 9$, so the true gain is $9 - 2.25 = 6.75$. The lemma gets there as follows. The greedy policy spends one step in state 0 and then stays in state 1, so $d' = (0.1, 0.9)$. Its action has advantage $0.225$ in state 0 and $0.725$ in state 1. Then $(0.1 \times 0.225 + 0.9 \times 0.725) / 0.1 = 6.75$, exactly.

> [!KEY]
> The performance difference lemma is an identity: the new policy's visits, scored by the old policy's advantage, give the exact improvement.

## See what the surrogate gets wrong

The lemma has a catch. It needs $d^{\pi'}$, the visits of a policy you have not run yet. Your samples come from $\pi$. So TRPO and PPO swap in $d^{\pi}$, which gives the *surrogate gain*:

$$
L_\pi(\pi') = \frac{1}{1 - \gamma} \sum_s d^{\pi}(s) \sum_a \pi'(a \mid s)\, A^{\pi}(s, a)
$$

In code, pass `pi_old` to `state_distribution`. In the worked example, the uniform policy has $d = (0.55, 0.45)$, so the surrogate gives $(0.55 \times 0.225 + 0.45 \times 0.725) / 0.1 = 4.5$, not 6.75. The greedy policy spends more time in the good state than the old policy did, and the surrogate cannot see that (Figure 1).

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Stacked contributions of the two states to the true gain 6.75 and the surrogate gain 4.5 in the two-state example">
<line x1="70" y1="250" x2="570" y2="250" class="stroke-muted"/>
<line x1="70" y1="250" x2="570" y2="250" class="stroke-grid"/><text x="64" y="254" class="fig-small" text-anchor="end">0</text>
<line x1="70" y1="186" x2="570" y2="186" class="stroke-grid"/><text x="64" y="190" class="fig-small" text-anchor="end">2</text>
<line x1="70" y1="122" x2="570" y2="122" class="stroke-grid"/><text x="64" y="126" class="fig-small" text-anchor="end">4</text>
<line x1="70" y1="58" x2="570" y2="58" class="stroke-grid"/><text x="64" y="62" class="fig-small" text-anchor="end">6</text>
<rect x="150" y="242.8" width="90" height="7.2" class="fill-a"/>
<rect x="150" y="34.0" width="90" height="208.8" class="fill-b"/>
<text x="195" y="250.4" class="fig-small" text-anchor="middle">0.225</text>
<text x="195" y="142.4" class="fig-small" text-anchor="middle">6.525</text>
<text x="195" y="26.0" class="fig-bold" text-anchor="middle">6.75</text>
<text x="195" y="270" class="fig-small" text-anchor="middle">true gain: new policy's visits</text>
<rect x="360" y="210.4" width="90" height="39.6" class="fill-a"/>
<rect x="360" y="106.0" width="90" height="104.4" class="fill-b"/>
<text x="405" y="234.2" class="fig-small" text-anchor="middle">1.238</text>
<text x="405" y="162.2" class="fig-small" text-anchor="middle">3.262</text>
<text x="405" y="98.0" class="fig-bold" text-anchor="middle">4.5</text>
<text x="405" y="270" class="fig-small" text-anchor="middle">surrogate: old policy's visits</text>
<rect x="430" y="16" width="14" height="14" class="fill-a"/><text x="450" y="28" class="fig-small">state 0</text>
<rect x="430" y="38" width="14" height="14" class="fill-b"/><text x="450" y="50" class="fig-small">state 1</text>
<text x="70" y="24" class="fig-small">gain in J</text>
</svg>
Figure 1. Each state's share of the gain from the uniform policy to the greedy policy, weighted by d/(1 − γ). Notice that the greedy policy spends 90% of its time in the good state 1, so the true gain is 6.75, while the old policy's visits give the surrogate only 4.5.
```

For a small step, the swap costs little. Mix the policies, $\pi_\alpha = (1 - \alpha)\pi + \alpha \pi'$. The surrogate and the true gain agree in value and slope at $\alpha = 0$, and their difference grows like $\alpha^2$. Halve the step, and the error falls by a factor of four. Schulman et al. (2015) turn this into a bound: with $\epsilon = \max_{s, a} \lvert A^\pi(s, a) \rvert$,

$$
J(\pi') - J(\pi) \ge L_\pi(\pi') - \frac{4 \epsilon \gamma}{(1 - \gamma)^2} \max_s D_{\mathrm{KL}}\big(\pi(\cdot \mid s) \,\Vert\, \pi'(\cdot \mid s)\big)
$$

Read it as: the true gain is at least the surrogate minus a penalty that grows with the largest per-state KL. Maximize the right side each step and the return never falls, which is the argument behind trust regions. The constant is large, 360 for $\gamma = 0.9$ and $\epsilon = 1$, so TRPO constrains the average KL instead, which gives up the guarantee but takes useful steps. PPO's clipping has no guarantee at all.

> [!KEY]
> The surrogate uses the old policy's states, so it is correct only to first order. Trust regions keep the step small enough that the second-order error stays small.

## Watch softmax policy gradient crawl

A tabular *softmax policy* gives each state and action a logit $\theta_{s, a}$. $J(\theta)$ is not concave in $\theta$, yet exact gradient ascent reaches the optimal policy when the step is small enough and every state has positive start probability (Agarwal et al., 2021). The question is how fast. Mei et al. (2020) answer it for a bandit with $K$ arms and mean rewards $r \in [0, 1]^K$. The update is

$$
\theta_{t+1}(a) = \theta_t(a) + \eta\, \pi_{\theta_t}(a) \big(r(a) - \pi_{\theta_t}^\top r\big)
$$

Read it as: push each arm's logit up by how much it beats the current average reward, scaled by its own probability. In numpy that is `theta + eta * pi * (r - pi @ r)`. The factor $\pi_{\theta_t}(a)$ is the problem. An arm the policy rarely picks gets a tiny push, even when it is the best arm.

With $\eta = 2/5$, Mei et al. prove that the gap $(\pi^* - \pi_{\theta_t})^\top r$ is at most $5 / (t\, c_t^2)$, where $c_t$ is the smallest probability the best arm has had so far, so from the uniform start it is at most $5K^2/t$. They also prove that the rate cannot beat $1/t$.

Why $1/t$? Near the optimum, the gap is roughly the probability $\varepsilon$ left on the runner-up arm times the reward difference. Each step moves the logit gap by an amount proportional to $\varepsilon$ itself, so $\varepsilon$ falls like $1/t$, and every tenfold improvement costs ten times as many steps.

> [!WARNING]
> The PMLR version of Mei et al. prints the arbitrary-start bound without the constant 5. The arXiv version, and the proof, carry it. The exercise uses $5 / (t\, c_t^2)$.

Take three arms with $r = (1, 0.9, 0.1)$ and $\eta = 0.4$. From uniform logits, the gap is $0.333$ at $t = 1$, then $0.195$, $0.0273$, $0.00183$, and $0.000169$ at $t = 10$, $100$, $1{,}000$, and $10{,}000$. The product $t \times \text{gap}$ stays between 1.7 and 2.7, inside the bound $5K^2 = 45$. Now start from $\theta_1 = (0, 5, 0)$, which puts 99% of the mass on the runner-up, so the best arm starts at probability $0.0066$. From uniform, the gap drops below 0.01 at $t = 226$. From the bad start, it takes $t = 7{,}062$, about 31 times longer (Figure 2).

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="Log-log plot of the gap against the step count for softmax policy gradient from a uniform start and from a bad start: the bad start stays flat near 0.1 for thousands of steps">
<line x1="70" y1="240.0" x2="560" y2="240.0" class="stroke-grid"/><text x="62" y="244.0" class="fig-small" text-anchor="end">1e-4</text>
<line x1="70" y1="190.0" x2="560" y2="190.0" class="stroke-grid"/><text x="62" y="194.0" class="fig-small" text-anchor="end">1e-3</text>
<line x1="70" y1="140.0" x2="560" y2="140.0" class="stroke-grid"/><text x="62" y="144.0" class="fig-small" text-anchor="end">1e-2</text>
<line x1="70" y1="90.0" x2="560" y2="90.0" class="stroke-grid"/><text x="62" y="94.0" class="fig-small" text-anchor="end">1e-1</text>
<line x1="70" y1="40.0" x2="560" y2="40.0" class="stroke-grid"/><text x="62" y="44.0" class="fig-small" text-anchor="end">1e0</text>
<text x="70.0" y="260" class="fig-small" text-anchor="middle">1</text>
<text x="192.5" y="260" class="fig-small" text-anchor="middle">10</text>
<text x="315.0" y="260" class="fig-small" text-anchor="middle">100</text>
<text x="437.5" y="260" class="fig-small" text-anchor="middle">1,000</text>
<text x="560.0" y="260" class="fig-small" text-anchor="middle">10,000</text>
<polyline points="70.0,24.9 106.9,40.0 128.4,48.8 143.8,55.1 155.6,59.9 165.3,63.9 173.5,67.2 180.6,70.1 186.9,72.7 192.5,74.9 197.6,77.0 202.2,78.9 206.5,80.6 214.1,83.8 217.5,85.2 223.8,87.7 229.4,90.0 234.4,92.1 241.2,94.8 245.3,96.5 250.9,98.8 257.6,101.5 262.1,103.4 267.6,105.6 273.7,108.1 279.2,110.3 285.1,112.7 290.4,114.9 296.0,117.2 301.1,119.3 307.0,121.7 312.3,123.8 317.6,126.0 323.4,128.4 329.0,130.6 334.4,132.9 339.7,135.0 345.4,137.3 350.8,139.6 356.2,141.8 361.8,144.0 367.2,146.3 372.7,148.5 378.4,150.8 383.9,153.1 389.3,155.3 394.8,157.5 400.3,159.8 405.9,162.0 411.4,164.3 416.8,166.5 422.3,168.8 427.8,171.0 433.4,173.3 438.9,175.5 444.4,177.8 449.9,180.0 455.4,182.3 460.9,184.5 466.4,186.8 471.9,189.0 477.4,191.2 482.9,193.5 488.4,195.7 493.9,198.0 499.4,200.2 504.9,202.5 510.4,204.7 516.0,207.0 521.5,209.2 527.0,211.5 532.5,213.7 538.0,216.0 543.5,218.2 549.0,220.5 554.5,222.7 560.0,224.9" class="stroke-muted dashed" fill="none"/>
<polyline points="70.0,63.9 106.9,65.3 128.4,66.7 143.8,68.1 155.6,69.4 165.3,70.7 173.5,72.0 180.6,73.2 186.9,74.3 192.5,75.4 197.6,76.5 202.2,77.5 206.5,78.5 214.1,80.4 217.5,81.3 223.8,82.9 229.4,84.5 234.4,86.0 241.2,88.0 245.3,89.3 250.9,91.1 257.6,93.4 262.1,95.0 267.6,97.0 273.7,99.4 279.2,101.6 285.1,104.1 290.4,106.4 296.0,109.0 301.1,111.4 307.0,114.2 312.3,116.8 317.6,119.5 323.4,122.4 329.0,125.2 334.4,128.0 339.7,130.7 345.4,133.5 350.8,136.3 356.2,139.0 361.8,141.7 367.2,144.4 372.7,147.1 378.4,149.8 383.9,152.4 389.3,155.0 394.8,157.5 400.3,160.1 405.9,162.7 411.4,165.2 416.8,167.6 422.3,170.1 427.8,172.6 433.4,175.0 438.9,177.4 444.4,179.9 449.9,182.3 455.4,184.7 460.9,187.0 466.4,189.4 471.9,191.7 477.4,194.1 482.9,196.4 488.4,198.8 493.9,201.1 499.4,203.4 504.9,205.7 510.4,208.0 516.0,210.3 521.5,212.6 527.0,214.9 532.5,217.2 538.0,219.5 543.5,221.8 549.0,224.1 554.5,226.3 560.0,228.6" class="stroke-a" fill="none"/>
<polyline points="70.0,89.0 106.9,89.0 128.4,89.0 143.8,89.0 155.6,89.0 165.3,89.0 173.5,89.0 180.6,89.0 186.9,89.0 192.5,89.0 197.6,89.1 202.2,89.1 206.5,89.1 214.1,89.1 217.5,89.1 223.8,89.1 229.4,89.1 234.4,89.1 241.2,89.1 245.3,89.1 250.9,89.1 257.6,89.1 262.1,89.1 267.6,89.2 273.7,89.2 279.2,89.2 285.1,89.2 290.4,89.2 296.0,89.2 301.1,89.3 307.0,89.3 312.3,89.3 317.6,89.3 323.4,89.3 329.0,89.4 334.4,89.4 339.7,89.4 345.4,89.4 350.8,89.5 356.2,89.5 361.8,89.5 367.2,89.5 372.7,89.6 378.4,89.6 383.9,89.6 389.3,89.7 394.8,89.7 400.3,89.7 405.9,89.7 411.4,89.7 416.8,89.8 422.3,89.8 427.8,89.8 433.4,89.8 438.9,89.8 444.4,89.9 449.9,89.9 455.4,89.9 460.9,89.9 466.4,89.9 471.9,89.9 477.4,89.9 482.9,90.0 488.4,90.0 493.9,90.0 499.4,90.0 504.9,90.0 510.4,90.0 516.0,90.0 521.5,90.1 527.0,90.1 532.5,90.3 538.0,91.2 543.5,162.1 549.0,185.6 554.5,197.5 560.0,205.7" class="stroke-b" fill="none"/>
<line x1="90" y1="52" x2="120" y2="52" class="stroke-a"/><text x="128" y="56" class="fig-small">uniform start: below 0.01 at t = 226</text>
<line x1="90" y1="72" x2="120" y2="72" class="stroke-b"/><text x="128" y="76" class="fig-small">bad start θ = (0, 5, 0): at t = 7,062</text>
<line x1="90" y1="92" x2="120" y2="92" class="stroke-muted dashed"/><text x="128" y="96" class="fig-small">2 / t, for reference</text>
<text x="315" y="284" text-anchor="middle">step t</text><text x="70" y="24" class="fig-small">gap (π* − π_t)ᵀ r</text>
</svg>
Figure 2. The gap of softmax policy gradient on the three-arm bandit, r = (1, 0.9, 0.1), η = 0.4, on log axes. Notice that the uniform start falls along a line of slope −1, which is the 1/t rate, while the bad start sits near 0.1 for thousands of steps before it escapes.
```

## Fix the crawl with the natural gradient

The previous chapter preconditioned the gradient with the Fisher matrix $F$. For a tabular softmax policy, that step has a closed form. Agarwal et al. (2021) show that $\theta \leftarrow \theta + \eta F^{\dagger} \nabla_\theta J$, with $F^\dagger$ the pseudo-inverse, gives

$$
\pi_{t+1}(a \mid s) = \frac{\pi_t(a \mid s) \exp\big(\eta A^{\pi_t}(s, a) / (1 - \gamma)\big)}{Z_t(s)}
$$

Read it as: in every state, multiply each action's probability by the exponential of its advantage, scaled by $\eta / (1 - \gamma)$, then divide by $Z_t(s)$, the sum over actions, so the row sums to one. The state distribution $d^{\pi}$ is gone, because the Fisher matrix carries the same $d(s)$ factor as the gradient and the pseudo-inverse cancels it.

```python
logits = eta * adv / (1 - mdp.gamma)
weights = pi * np.exp(logits - logits.max(axis=1, keepdims=True))
pi_new = weights / weights.sum(axis=1, keepdims=True)
```

This is the classic *multiplicative-weights* rule from online learning, and it explains three things. The factor $\pi(a \mid s)$ that slowed softmax policy gradient is gone, so a rarely chosen but good action grows by the same factor as a common one. Each row maximizes $\frac{\eta}{1 - \gamma} \sum_a p(a) A(s, a) - D_{\mathrm{KL}}(p \,\Vert\, \pi(\cdot \mid s))$, which is "follow the advantage, but pay for moving away in KL". And as $\eta \to \infty$ all the mass goes to the action with the largest advantage, which is a policy iteration step.

Agarwal et al. also prove a rate. With rewards in $[0, 1]$, the uniform start, and any $\eta > 0$,

$$
J(\pi^*) - J(\pi_T) \le \frac{\log A}{\eta T} + \frac{1}{(1 - \gamma)^2 T}
$$

where $A$ is the number of actions. The bound does not depend on the number of states or on how small any probability gets.

In the two-state example with $\eta = 0.1$, the multiplier is $\exp(A)$. State 0 has weights $0.5 e^{-0.225}$ and $0.5 e^{0.225}$, which normalize to $(0.3894, 0.6106)$. State 1 normalizes to $(0.8100, 0.1900)$. One step lifts $J$ from 2.25 to 5.4249. With $\eta = 1$, one step reaches 8.9900, within 0.01 of the optimal 9.

> [!KEY]
> For a softmax policy, the natural gradient multiplies each action's probability by $\exp(\eta A / (1 - \gamma))$. That removes the $\pi(a \mid s)$ factor that makes plain policy gradient crawl.

> [!RECAP]
> - `advantage` is `q_from_v(mdp, v) - v[:, None]`, and it averages to zero under the policy in every state.
> - The lemma weights states by the new policy's $d$, and the surrogate uses the old policy's $d$. Both divide by $1 - \gamma$.
> - Softmax policy gradient on a bandit adds `eta * pi * (r - pi @ r)` to the logits, and its gap falls like $1/t$.
> - NPG multiplies `pi` by `exp(eta * A / (1 - gamma))` and renormalizes each row.

## Build it step by step

The checks test the functions in this order.

1. **`advantage(mdp, pi)`.** Returns `A`, shape `(S, A)`, from `rlenvs.policy_evaluation` and `rlenvs.q_from_v`. The common mistake is to subtract `v` without `[:, None]`.
2. **`performance_difference(mdp, pi_old, pi_new)`.** Returns a float. Weight the per-state sum `(pi_new * A_old).sum(axis=1)` by `state_distribution(mdp, pi_new)`, then divide by `1 - gamma`. Using the old policy's distribution gives the surrogate instead.
3. **`surrogate_gain(mdp, pi_old, pi_new)`.** The same, with `state_distribution(mdp, pi_old)`.
4. **`bandit_gradient(r, theta)`.** Returns `(K,)`: softmax with the maximum subtracted, then `pi * (r - pi @ r)`.
5. **`softmax_pg_bandit(r, theta, eta, steps)`.** Returns `(steps, K)` policies. Record the policy before each update, so row 0 is the start. Copy `theta` first, or you change the caller's array.
6. **`npg_step(mdp, pi, eta)`.** Returns `(S, A)`. Subtract each row's maximum exponent before `np.exp`. The check sets `eta = 1e4` and expects the greedy policy, which overflows without the subtraction.
7. **`run_npg(mdp, eta, iters)`.** Returns `iters + 1` values of $J$, starting with the uniform policy.

## Where this breaks

**Exact advantages are an assumption.** Every rate here assumes the true advantage. Sampled advantages add noise and need a different analysis.

**The start distribution matters.** States the policy never reaches never improve. General bounds carry a *distribution mismatch coefficient*, the largest ratio of the optimal policy's visitation to $\mu_0$.

**The constants can be astronomical.** The $1/t$ rate hides a factor of $1/c^2$, and Li et al. (2021) build MDPs where softmax policy gradient needs exponentially many steps ([arXiv:2102.11270](https://arxiv.org/abs/2102.11270)).

**Function approximation breaks the closed forms.** The multiplicative-weights form holds only for tabular softmax policies. With neural networks, TRPO approximates $F^{-1} g$ with conjugate gradient instead.

## Further reading

- [Kakade and Langford (2002), Approximately optimal approximate reinforcement learning](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf) introduces the performance difference lemma.
- [Agarwal, Kakade, Lee, and Mahajan (2021), On the theory of policy gradient methods](https://arxiv.org/abs/1908.00261) proves global convergence of softmax policy gradient and the dimension-free NPG rate.
- [Mei, Xiao, Szepesvári, and Schuurmans (2020), On the global convergence rates of softmax policy gradient methods](https://arxiv.org/abs/2005.06392) proves the $1/t$ upper and lower bounds.

## The exercise

You verify the theory on exact MDPs. Write `advantage`, `performance_difference`, and `surrogate_gain`, which differ only in whose state distribution weights the per-state gain. Write `bandit_gradient` and `softmax_pg_bandit` for exact softmax policy gradient, then `npg_step` and `run_npg` for the natural policy gradient.

The checks compare the lemma with the exact change in $J$ on 100 random policy pairs, and the surrogate's error as the step halves. They test the bound of Mei et al., the $1/t$ slope, and the bad start. They compare `npg_step` with an explicit Fisher step and a mirror descent solution, and check `run_npg` against the bound of Agarwal et al.
