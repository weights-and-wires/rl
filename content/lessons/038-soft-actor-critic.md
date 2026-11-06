---
title: Soft actor-critic
slug: 038-soft-actor-critic
part: "Part 6 — Continuous control and maximum entropy"
summary: Build SAC's three tricky pieces, a tanh-Gaussian log-probability, a reparameterized actor loss, and an automatic temperature, and check each exactly.
minutes: 50
gpu: false
objectives:
  - Derive the log-probability of a tanh-squashed Gaussian action and compute it without overflow.
  - Explain why the actor's gradient must flow through the sampled action, and build SAC's reparameterized actor loss.
  - Write the temperature loss and show that its gradient steps drive the policy's entropy to a target.
prereqs:
  - 036-deterministic-policy-gradients
  - 037-maximum-entropy-rl
exercise: 038-soft-actor-critic
---

# Soft actor-critic

The last two chapters built two halves of one method. Maximum-entropy reinforcement learning gave a better goal: earn reward, and stay as random as you can while doing it. TD3 gave a stable way to train a continuous-action actor from a replay buffer. Soft actor-critic, or SAC, joins them, and it is among the most widely used methods for continuous control.

Joining them raises one question: how do you build an actor that outputs a random, bounded action, whose randomness the objective can price, and whose parameters you can still train by gradient? The Boltzmann policy from the last chapter cannot be the actor itself, because over a continuous range you cannot normalize it or sample from it. SAC uses a Gaussian instead, and three details have to be right: the log-probability of a squashed action, the gradient through a random sample, and the price of entropy.

You will check each of these against exact mathematics, which is the only reliable way to find the bugs that SAC implementations are known for.

## What you already know

- The maximum-entropy objective adds $\alpha$ times the policy's entropy to every reward, and its best policy is a Boltzmann distribution over soft action values (see [Maximum-entropy reinforcement learning](/lessons/037-maximum-entropy-rl)).
- TD3 trains two critics toward a target built from the smaller of two target critics, and moves the targets by Polyak averaging (see [Deterministic policy gradients](/lessons/036-deterministic-policy-gradients)).
- The deterministic policy gradient moves an action along the critic's slope $\nabla_a Q(s, a)$.
- A Gaussian $\mathcal{N}(\mu, \sigma^2)$ has log-density $-\tfrac12((u - \mu)/\sigma)^2 - \log \sigma - \tfrac12 \log 2\pi$.

## Turn soft policy iteration into two losses

SAC trains a stochastic actor $\pi_\theta(a \mid s)$, two critics, and target copies of the critics, from a replay buffer (Haarnoja et al., 2018). The critics regress toward TD3's target plus an entropy bonus:

$$
y = r + \gamma (1 - \text{terminated}) \Big( \min_{i = 1, 2} Q_{\phi'_i}(s', a') - \alpha \log \pi_\theta(a' \mid s') \Big), \qquad a' \sim \pi_\theta(\cdot \mid s')
$$

Read it as the clipped double-Q target, with the next action sampled fresh from the *current* policy. SAC has no target actor. The term $-\alpha \log \pi(a' \mid s')$ is a one-sample estimate of $\alpha$ times the entropy at the next state.

The actor does the improvement step. The ideal move is to the Boltzmann policy $\exp(Q/\alpha)/Z$, but a Gaussian cannot match it, so SAC minimizes the KL divergence to it. Dropping the constant $\log Z$ and multiplying by $\alpha$ gives the actor loss:

$$
\mathcal{L}_\pi(\theta) = \mathbb{E}_{s \sim \mathcal{D},\, a \sim \pi_\theta(\cdot \mid s)} \Big[ \alpha \log \pi_\theta(a \mid s) - \min_{i = 1, 2} Q_{\phi_i}(s, a) \Big]
$$

Read it as: prefer actions the critics score highly, and pay $\alpha$ for every nat of certainty. Minimizing it maximizes expected $Q$ plus $\alpha$ times entropy, state by state. Both formulas need $\log \pi(a \mid s)$ for a bounded action, so start there.

## Squash a Gaussian into bounds

A Gaussian can output any real number, but a motor accepts torques in $[-2, 2]$. The obvious fix is to clip the sample. That is wrong for two reasons: clipping has zero gradient outside the bounds, and it piles probability on the endpoints, where no density formula applies. SAC squashes smoothly instead. Draw $u \sim \mathcal{N}(\mu_\theta(s), \sigma_\theta(s)^2)$ and output $a = a_{\max} \tanh(u)$, which always lands inside the bounds.

Squashing moves probability around. Near $u = 0$ the map barely changes widths, but near the bounds `tanh` flattens, so a wide range of $u$ crowds into a thin sliver of $a$. The density of $a$ must rise wherever the map compresses, which Figure 1 shows.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Density of the squashed action for mean 0.3, standard deviation 0.8, and bound 2: the correct density rises toward the bounds, while the density without the squash term is lower there and integrates to only 0.661">
<line x1="70.0" y1="203.3" x2="560.0" y2="203.3" class="stroke-grid"/>
<text x="62.0" y="207.3" class="fig-small" text-anchor="end">0.1</text>
<line x1="70.0" y1="156.7" x2="560.0" y2="156.7" class="stroke-grid"/>
<text x="62.0" y="160.7" class="fig-small" text-anchor="end">0.2</text>
<line x1="70.0" y1="110.0" x2="560.0" y2="110.0" class="stroke-grid"/>
<text x="62.0" y="114.0" class="fig-small" text-anchor="end">0.3</text>
<line x1="70.0" y1="63.3" x2="560.0" y2="63.3" class="stroke-grid"/>
<text x="62.0" y="67.3" class="fig-small" text-anchor="end">0.4</text>
<line x1="70.0" y1="250.0" x2="560.0" y2="250.0" class="stroke-muted"/>
<line x1="70.0" y1="250.0" x2="70.0" y2="255.0" class="stroke-muted"/>
<text x="70.0" y="270.0" class="fig-small" text-anchor="middle">-2</text>
<line x1="192.5" y1="250.0" x2="192.5" y2="255.0" class="stroke-muted"/>
<text x="192.5" y="270.0" class="fig-small" text-anchor="middle">-1</text>
<line x1="315.0" y1="250.0" x2="315.0" y2="255.0" class="stroke-muted"/>
<text x="315.0" y="270.0" class="fig-small" text-anchor="middle">0</text>
<line x1="437.5" y1="250.0" x2="437.5" y2="255.0" class="stroke-muted"/>
<text x="437.5" y="270.0" class="fig-small" text-anchor="middle">1</text>
<line x1="560.0" y1="250.0" x2="560.0" y2="255.0" class="stroke-muted"/>
<text x="560.0" y="270.0" class="fig-small" text-anchor="middle">2</text>
<text x="315.0" y="288.0" text-anchor="middle">action a = 2 tanh(u)</text>
<text x="62.0" y="24.0" class="fig-small" text-anchor="end">density</text>
<polyline points="73.7,239.0 77.7,226.5 81.7,216.8 85.7,209.2 89.8,203.1 93.8,198.1 97.8,194.0 101.8,190.4 105.9,187.4 109.9,184.8 113.9,182.4 117.9,180.4 121.9,178.6 126.0,176.9 130.0,175.4 134.0,174.1 138.0,172.8 142.1,171.7 146.1,170.6 150.1,169.6 154.1,168.7 158.1,167.8 162.2,167.0 166.2,166.2 170.2,165.5 174.2,164.7 178.2,164.0 182.3,163.3 186.3,162.7 190.3,162.0 194.3,161.4 198.4,160.8 202.4,160.2 206.4,159.6 210.4,159.0 214.4,158.4 218.5,157.8 222.5,157.2 226.5,156.6 230.5,156.0 234.6,155.4 238.6,154.8 242.6,154.2 246.6,153.6 250.6,152.9 254.7,152.3 258.7,151.7 262.7,151.0 266.7,150.4 270.8,149.7 274.8,149.1 278.8,148.4 282.8,147.7 286.8,147.0 290.9,146.2 294.9,145.5 298.9,144.7 302.9,144.0 307.0,143.2 311.0,142.4 315.0,141.5 319.0,140.7 323.0,139.8 327.1,138.9 331.1,138.0 335.1,137.1 339.1,136.2 343.2,135.2 347.2,134.2 351.2,133.1 355.2,132.1 359.2,131.0 363.3,129.9 367.3,128.7 371.3,127.6 375.3,126.4 379.4,125.1 383.4,123.8 387.4,122.5 391.4,121.2 395.4,119.8 399.5,118.3 403.5,116.8 407.5,115.3 411.5,113.7 415.6,112.1 419.6,110.4 423.6,108.7 427.6,106.9 431.6,105.0 435.7,103.1 439.7,101.1 443.7,99.1 447.7,97.0 451.8,94.8 455.8,92.5 459.8,90.2 463.8,87.8 467.8,85.3 471.9,82.7 475.9,80.0 479.9,77.3 483.9,74.4 487.9,71.5 492.0,68.5 496.0,65.4 500.0,62.3 504.0,59.1 508.1,55.9 512.1,52.8 516.1,49.7 520.1,46.8 524.1,44.2 528.2,42.1 532.2,40.8 536.2,40.8 540.2,43.0 544.3,48.9 548.3,61.3 552.3,86.7 556.3,141.8" class="stroke-b"/>
<polyline points="73.7,249.7 77.7,248.5 81.7,246.9 85.7,244.9 89.8,242.7 93.8,240.4 97.8,238.0 101.8,235.5 105.9,233.0 109.9,230.5 113.9,228.0 117.9,225.4 121.9,222.9 126.0,220.4 130.0,218.0 134.0,215.5 138.0,213.1 142.1,210.7 146.1,208.4 150.1,206.1 154.1,203.8 158.1,201.5 162.2,199.3 166.2,197.1 170.2,195.0 174.2,192.9 178.2,190.8 182.3,188.8 186.3,186.8 190.3,184.8 194.3,182.9 198.4,181.0 202.4,179.2 206.4,177.3 210.4,175.5 214.4,173.8 218.5,172.1 222.5,170.4 226.5,168.8 230.5,167.2 234.6,165.6 238.6,164.0 242.6,162.5 246.6,161.1 250.6,159.6 254.7,158.2 258.7,156.9 262.7,155.6 266.7,154.3 270.8,153.0 274.8,151.8 278.8,150.6 282.8,149.4 286.8,148.3 290.9,147.2 294.9,146.2 298.9,145.2 302.9,144.2 307.0,143.3 311.0,142.4 315.0,141.5 319.0,140.7 323.0,140.0 327.1,139.2 331.1,138.5 335.1,137.9 339.1,137.3 343.2,136.7 347.2,136.2 351.2,135.7 355.2,135.3 359.2,134.9 363.3,134.6 367.3,134.3 371.3,134.0 375.3,133.9 379.4,133.7 383.4,133.7 387.4,133.6 391.4,133.7 395.4,133.8 399.5,134.0 403.5,134.2 407.5,134.5 411.5,134.9 415.6,135.3 419.6,135.8 423.6,136.4 427.6,137.1 431.6,137.9 435.7,138.7 439.7,139.7 443.7,140.7 447.7,141.9 451.8,143.1 455.8,144.5 459.8,146.0 463.8,147.6 467.8,149.4 471.9,151.3 475.9,153.3 479.9,155.5 483.9,157.9 487.9,160.5 492.0,163.2 496.0,166.2 500.0,169.3 504.0,172.8 508.1,176.4 512.1,180.4 516.1,184.7 520.1,189.2 524.1,194.2 528.2,199.5 532.2,205.2 536.2,211.4 540.2,218.0 544.3,225.0 548.3,232.4 552.3,239.9 556.3,246.8" class="stroke-muted dashed"/>
<line x1="100.0" y1="62.0" x2="140.0" y2="62.0" class="stroke-b"/>
<text x="148.0" y="66.0" class="fig-small" text-anchor="start">with the squash term: area 1.000</text>
<line x1="100.0" y1="84.0" x2="140.0" y2="84.0" class="stroke-muted dashed"/>
<text x="148.0" y="88.0" class="fig-small" text-anchor="start">without it: area 0.661</text>
</svg>
Figure 1. The density of the squashed action when u is drawn from a Gaussian with mean 0.3 and standard deviation 0.8. Notice that tanh piles probability up near the bounds, and that the density without the squash term integrates to only 0.661, so it is not a probability density.
```

The *change of variables* formula says by how much: divide the density of $u$ by $\lvert da/du \rvert = a_{\max}(1 - \tanh^2 u)$. With $D$ independent action dimensions,

$$
\log \pi(a \mid s) = \sum_{i=1}^{D} \Big[ \log \mathcal{N}\big(u_i; \mu_i, \sigma_i^2\big) - \log\big(1 - \tanh^2 u_i\big) - \log a_{\max} \Big]
$$

Read it as the Gaussian log-density of the pre-squash sample, minus the log of how much the squash stretched each dimension. Take one dimension with $\mu = 0$, $\sigma = 1$, and $u = 0.5$:

- The Gaussian log-density is $-\tfrac12(0.5)^2 - 0 - 0.9189 = -1.0439$.
- $\tanh 0.5 = 0.4621$, so $1 - \tanh^2 u = 0.7864$ and its log is $-0.2402$.
- With $a_{\max} = 1$: $\log \pi(a) = -1.0439 + 0.2402 = -0.8037$.
- With $a_{\max} = 2$, subtract $\log 2 = 0.6931$ more, for about $-1.497$.

Always compute the log-probability from the stored $u$. Recovering $u$ from $a$ with `atanh` blows up near the bounds, where `tanh` rounds to exactly 1. The term $\log(1 - \tanh^2 u)$ has its own trap: for $\lvert u \rvert$ above about 9 in float32, `torch.tanh(u)` rounds to 1 and the log is $-\infty$. Since $1 - \tanh^2 u = 4/(e^u + e^{-u})^2$, an equal form that never overflows is

$$
\log\big(1 - \tanh^2 u\big) = 2\big(\log 2 - u - \operatorname{softplus}(-2u)\big)
$$

where $\operatorname{softplus}(x) = \log(1 + e^x)$. At $u = 0.5$ it gives $2(0.6931 - 0.5 - 0.3133) = -0.2402$, the same as before. In PyTorch:

```python
log_det = (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(dim=-1)
log_prob = gauss_log_prob - log_det - D * math.log(max_action)
```

> [!WARNING]
> Forgetting the squash correction is the most common SAC bug. The log-probability is wrong most at the bounds, so the entropy estimate is biased, the temperature chases the wrong target, and learning degrades without an error message.

## Differentiate through the sample

The actor loss is an expectation over actions that the actor itself samples. To differentiate it, you could multiply each sample's loss by the score $\nabla_\theta \log \pi$, as REINFORCE does. That works for any loss, but it is noisy, and here you can do better because the critic is differentiable in the action.

The *reparameterization trick* writes the sample as a fixed function of the parameters and parameter-free noise:

$$
a = a_{\max} \tanh\big(\mu_\theta(s) + \sigma_\theta(s)\, \varepsilon\big), \qquad \varepsilon \sim \mathcal{N}(0, I)
$$

Read it as: draw standard normal noise first, then shape it with the network's mean and standard deviation. The randomness no longer depends on $\theta$, so the gradient and the expectation over $\varepsilon$ swap places. The gradient flows through $a$ into the critic, as in the deterministic policy gradient, and through $\log\pi$ both directly and through $a$. In PyTorch the only rule is to build the action from `eps` without detaching anything:

```python
mean, log_std = policy(s)
a, log_prob = sample_action(mean, log_std, eps, policy.max_action)
loss = (alpha * log_prob - torch.minimum(q1(s, a), q2(s, a))).mean()
```

The estimate is unbiased, and the exercise proves it. It computes the expected loss by quadrature, differentiates that by finite differences, and compares with your autograd gradient over 20,000 stratified noise draws.

## Tune the temperature automatically

The right $\alpha$ depends on the reward's scale and changes during training, so SAC learns it (Haarnoja et al., 2018b). The idea is to ask for the highest return while keeping the policy's entropy at least a target $\bar{\mathcal{H}}$, and to treat $\alpha$ as that constraint's Lagrange multiplier. Implementations learn $\log\alpha$ to keep $\alpha$ positive, and the temperature loss is

$$
J(\alpha) = \mathbb{E}_{a \sim \pi} \big[ -\alpha \log \pi(a \mid s) - \alpha \bar{\mathcal{H}} \big], \qquad
\frac{\partial J}{\partial \log \alpha} = \alpha \big(\mathcal{H} - \bar{\mathcal{H}}\big)
$$

Read the sign. When the entropy $\mathcal{H} = \mathbb{E}[-\log \pi]$ is below the target, the gradient is negative, so a descent step raises $\alpha$ and entropy becomes worth more. When it is above the target, $\alpha$ falls. Suppose a batch's log-probabilities average $-0.5$, so $\mathcal{H} = 0.5$, and the target is $-1$. At $\alpha = 0.2$ the gradient is $0.2 \times 1.5 = 0.3$. The entropy is above target, so descent lowers $\alpha$.

The log-probabilities are constants in this loss, because the temperature step must not move the policy:

```python
loss = -(log_alpha.exp() * (log_prob + target_entropy).detach()).mean()
```

The paper's default target is $-\dim(\mathcal{A})$, a heuristic. Differential entropy can be negative, so a negative target is fine.

Figure 2 shows where the loop settles. Against $Q(a) = -a^2/2$ the best Gaussian at temperature $\alpha$ has $\mu = 0$ and $\sigma^2 = \alpha$, so its entropy is $\tfrac12\log(2\pi e\,\alpha)$. A target of 1 nat gives $\alpha^* = e^2/(2\pi e) = 0.4326$. Adam steps on $J$ reach it within about 150 steps.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Temperature alpha against Adam step: it starts at 1, dips to 0.383 at step 36, and settles at 0.4326, the value where the entropy equals the target">
<line x1="70.0" y1="222.0" x2="560.0" y2="222.0" class="stroke-grid"/>
<text x="62.0" y="226.0" class="fig-small" text-anchor="end">0.4</text>
<line x1="70.0" y1="166.0" x2="560.0" y2="166.0" class="stroke-grid"/>
<text x="62.0" y="170.0" class="fig-small" text-anchor="end">0.6</text>
<line x1="70.0" y1="110.0" x2="560.0" y2="110.0" class="stroke-grid"/>
<text x="62.0" y="114.0" class="fig-small" text-anchor="end">0.8</text>
<line x1="70.0" y1="54.0" x2="560.0" y2="54.0" class="stroke-grid"/>
<text x="62.0" y="58.0" class="fig-small" text-anchor="end">1.0</text>
<line x1="70.0" y1="250.0" x2="560.0" y2="250.0" class="stroke-muted"/>
<line x1="70.0" y1="250.0" x2="70.0" y2="255.0" class="stroke-muted"/>
<text x="70.0" y="270.0" class="fig-small" text-anchor="middle">0</text>
<line x1="192.5" y1="250.0" x2="192.5" y2="255.0" class="stroke-muted"/>
<text x="192.5" y="270.0" class="fig-small" text-anchor="middle">50</text>
<line x1="315.0" y1="250.0" x2="315.0" y2="255.0" class="stroke-muted"/>
<text x="315.0" y="270.0" class="fig-small" text-anchor="middle">100</text>
<line x1="437.5" y1="250.0" x2="437.5" y2="255.0" class="stroke-muted"/>
<text x="437.5" y="270.0" class="fig-small" text-anchor="middle">150</text>
<line x1="560.0" y1="250.0" x2="560.0" y2="255.0" class="stroke-muted"/>
<text x="560.0" y="270.0" class="fig-small" text-anchor="middle">200</text>
<text x="315.0" y="288.0" text-anchor="middle">Adam step</text>
<text x="62.0" y="24.0" text-anchor="end">α</text>
<line x1="70.0" y1="212.9" x2="560.0" y2="212.9" class="stroke-c dashed"/>
<text x="560.0" y="202.9" class="fig-small" text-anchor="end">α* = 0.4326, where the entropy equals the target 1</text>
<polyline points="70.0,54.0 72.5,67.7 74.9,80.6 77.3,92.8 79.8,104.3 82.2,115.1 84.7,125.2 87.2,134.7 89.6,143.4 92.0,151.6 94.5,159.2 97.0,166.2 99.4,172.6 101.8,178.5 104.3,184.0 106.8,188.9 109.2,193.5 111.7,197.6 114.1,201.4 116.5,204.8 119.0,207.8 121.4,210.6 123.9,213.1 126.3,215.3 128.8,217.2 131.2,219.0 133.7,220.5 136.2,221.8 138.6,222.9 141.1,223.9 143.5,224.7 145.9,225.4 148.4,225.9 150.9,226.3 153.3,226.6 155.8,226.7 158.2,226.8 160.7,226.8 163.1,226.6 165.6,226.4 168.0,226.2 170.4,225.8 172.9,225.5 175.3,225.0 177.8,224.5 180.2,224.0 182.7,223.4 185.1,222.8 187.6,222.2 190.1,221.6 192.5,220.9 194.9,220.3 197.4,219.6 199.8,219.0 202.3,218.3 204.8,217.7 207.2,217.0 209.6,216.4 212.1,215.8 214.5,215.3 217.0,214.7 219.4,214.2 221.9,213.8 224.3,213.3 226.8,212.9 229.2,212.6 231.7,212.2 234.2,211.9 236.6,211.7 239.0,211.4 241.5,211.3 243.9,211.1 246.4,211.0 248.8,210.9 251.3,210.8 253.8,210.8 256.2,210.8 258.6,210.8 261.1,210.8 263.6,210.9 266.0,211.0 268.5,211.1 270.9,211.2 273.4,211.3 275.8,211.4 278.2,211.5 280.7,211.6 283.1,211.8 285.6,211.9 288.1,212.0 290.5,212.2 293.0,212.3 295.4,212.4 297.9,212.5 300.3,212.6 302.8,212.7 305.2,212.8 307.6,212.9 310.1,213.0 312.6,213.0 315.0,213.1 317.4,213.1 319.9,213.2 322.4,213.2 324.8,213.2 327.2,213.3 329.7,213.3 332.2,213.3 334.6,213.3 337.1,213.3 339.5,213.3 342.0,213.2 344.4,213.2 346.8,213.2 349.3,213.2 351.8,213.2 354.2,213.1 356.6,213.1 359.1,213.1 361.6,213.0 364.0,213.0 366.4,213.0 368.9,213.0 371.4,212.9 373.8,212.9 376.2,212.9 378.7,212.9 381.1,212.9 383.6,212.8 386.1,212.8 388.5,212.8 390.9,212.8 393.4,212.8 395.9,212.8 398.3,212.8 400.8,212.8 403.2,212.8 405.7,212.8 408.1,212.8 410.5,212.8 413.0,212.8 415.4,212.8 417.9,212.8 420.3,212.8 422.8,212.8 425.2,212.8 427.7,212.8 430.1,212.8 432.6,212.8 435.1,212.8 437.5,212.8 439.9,212.8 442.4,212.9 444.9,212.9 447.3,212.9 449.8,212.9 452.2,212.9 454.7,212.9 457.1,212.9 459.6,212.9 462.0,212.9 464.5,212.9 466.9,212.9 469.3,212.9 471.8,212.9 474.2,212.9 476.7,212.9 479.1,212.9 481.6,212.9 484.1,212.9 486.5,212.9 488.9,212.9 491.4,212.9 493.9,212.9 496.3,212.9 498.8,212.9 501.2,212.9 503.6,212.9 506.1,212.9 508.6,212.9 511.0,212.9 513.5,212.9 515.9,212.9 518.4,212.9 520.8,212.9 523.2,212.9 525.7,212.9 528.2,212.9 530.6,212.9 533.0,212.9 535.5,212.9 538.0,212.9 540.4,212.9 542.8,212.9 545.3,212.9 547.8,212.9 550.2,212.9 552.6,212.9 555.1,212.9 557.5,212.9 560.0,212.9" class="stroke-b"/>
<circle cx="70.0" cy="54.0" r="5" class="fill-b"/>
<text x="80.0" y="58.0" class="fig-small" text-anchor="start">start: α = 1, entropy 1.419</text>
<circle cx="158.2" cy="226.8" r="5" class="fill-b"/>
<text x="168.2" y="242.8" class="fig-small" text-anchor="start">overshoot: 0.3829 at step 36</text>
</svg>
Figure 2. The temperature tuned against the policy that best responds to Q(a) = -a²/2, with a target entropy of 1 nat. Notice that the entropy starts above the target, so α falls, overshoots a little, and settles at the value where the gradient α(H - target) is zero.
```

> [!KEY]
> A tanh-squashed Gaussian's log-probability is the Gaussian log-density of the pre-squash sample, minus $\sum_i \log(1 - \tanh^2 u_i)$ and $D \log a_{\max}$. The actor trains by reparameterization, so the critic's slope reaches the policy's parameters. And $\alpha$ rises when the policy is too certain and falls when it is too random. Together these answer the opening question: SAC's actor is a Gaussian whose log-probability, gradient, and price you can all compute exactly.

> [!RECAP]
> - `tanh_log_det` is `2 * (log 2 - u - softplus(-2u))`, summed over the last dimension.
> - `sample_action` builds `u` from `eps`, squashes it, and subtracts both corrections from the Gaussian log-density.
> - The actor loss is `(alpha * log_prob - min(q1, q2)).mean()` at reparameterized actions.
> - The critic target is `r + gamma * (1 - terminated) * (min(q1, q2) - alpha * log_prob)`.
> - The temperature loss detaches `log_prob + target_entropy`.

## Build it step by step

The exercise gives you `TanhGaussianPolicy` and `gaussian_log_prob`, which sums the Gaussian log-density over the last dimension. The checks test your functions in this order.

1. `tanh_log_det(u)` takes `(N, D)` and returns `(N,)`. Use `torch.nn.functional.softplus`. The check feeds $u = \pm 20$ in float32, where the naive formula gives $-\infty$.
2. `sample_action(mean, log_std, eps, max_action)` returns `(a, log_prob)`. Build `u = mean + exp(log_std) * eps`, then subtract `tanh_log_det(u)` and `D * log(max_action)`. The common mistake is dropping the `max_action` term.
3. `actor_loss(policy, critic1, critic2, s, eps, alpha)` calls `policy(s)`, samples with `policy.max_action`, and averages `alpha * log_prob - min(q1, q2)`. The common mistake is detaching the action before the critics, which throws away the reparameterized gradient.
4. `critic_target(r, terminated, q1_next, q2_next, next_log_prob, gamma, alpha)` is one line. The common mistake is adding $\alpha \log \pi$ instead of subtracting it.
5. `temperature_loss(log_alpha, log_prob, target_entropy)` is one line. The common mistake is forgetting `.detach()`, which lets the temperature step push on the policy.

## Where this breaks

- **The temperature can run away.** If the entropy estimate is biased or the target is unreachable, $\log\alpha$ drifts without bound. Clamp `log_std` and log $\alpha$.
- **The target entropy is a heuristic.** $-\dim(\mathcal{A})$ works on standard benchmarks and is not derived from anything. Discrete actions need a different target (Christodoulou, 2019).
- **Gaussian policies are unimodal.** The Boltzmann policy SAC chases can have several peaks, and a tanh-Gaussian picks one.
- **The deadly triad remains.** SAC bootstraps from off-policy data with function approximation. Twin critics and target networks make it stable in practice, not in theory.

## Further reading

The following papers introduce SAC and the ideas it relies on:

- [Haarnoja, Zhou, Abbeel, and Levine (2018), Soft actor-critic](https://arxiv.org/abs/1801.01290) derives soft policy iteration and introduces SAC, with the tanh correction in its appendix.
- [Haarnoja et al. (2018b), Soft actor-critic algorithms and applications](https://arxiv.org/abs/1812.05905) adds automatic temperature tuning and drops the separate value network.
- [Kingma and Welling (2014), Auto-encoding variational Bayes](https://arxiv.org/abs/1312.6114) introduces the reparameterization trick.
- [Christodoulou (2019), Soft actor-critic for discrete action settings](https://arxiv.org/abs/1910.07207) adapts SAC and its entropy target to discrete actions.

## The exercise

You build SAC's mathematical core. Write `tanh_log_det`, `sample_action`, `actor_loss`, `critic_target`, and `temperature_loss`.

The checks compare the stable squash term with the naive formula and at $u = \pm 20$ in float32. They compare `exp(log_prob)` with the density from a numerical change of variables, integrate it to 1, and check a 2-D case against PyTorch's `TanhTransform`. They confirm that gradients reach the mean and log standard deviation, that the actor loss uses the smaller critic, and that its gradient matches the exact expected loss. They check the critic target on hand-worked cases, the temperature loss's gradient $\alpha(\mathcal{H} - \bar{\mathcal{H}})$ and its detached log-probabilities, and that the temperature settles where the entropy meets its target.
