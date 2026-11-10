---
title: Proximal policy optimization
slug: 034-proximal-policy-optimization
part: "Part 5 — Policy gradients"
summary: Reuse each batch for several epochs safely with PPO's clipped objective, then train an agent that balances CartPole.
minutes: 40
gpu: false
objectives:
  - Explain why the clipped surrogate removes the incentive to push a ratio past 1 plus or minus epsilon, and only in the direction the advantage favors.
  - Compute the clipped surrogate and the entropy bonus, and check each against hand-computed values.
  - Monitor an update with the k3 estimate of the KL and the clip fraction.
  - Stop a PPO update early when the KL estimate passes a target.
  - Assemble a PPO training loop with GAE, minibatch epochs, advantage normalization, and gradient clipping, and train it on CartPole.
prereqs:
  - 025-deep-q-networks
  - 032-generalized-advantage-estimation
  - 033-trust-regions-and-trpo
exercise: 034-proximal-policy-optimization
---

# Proximal policy optimization

Collecting experience is the slow part of on-policy learning. A policy gradient method runs the policy for a few thousand steps, takes one gradient step, and throws the data away, because the next step needs data from the updated policy. It would be far cheaper to take several gradient steps on each batch. But every step moves the policy further from the one that collected the data, and eventually the batch describes a policy that no longer exists.

TRPO handled this with a hard limit on the KL divergence, enforced by conjugate gradient and a line search. It works, and it is complicated. *Proximal policy optimization* (PPO), from Schulman, Wolski, Dhariwal, Radford, and Klimov (2017), gets most of the same stability with ordinary minibatch gradient descent. It changes the objective so that moving the policy too far earns nothing, and then lets Adam do the rest.

PPO became the default policy gradient method for games, robotics, and the reinforcement learning stage of training large language models. You build its pieces one at a time, check each against numbers you can compute by hand, and then assemble an agent that learns to balance CartPole in about a second of CPU time.

## What you already know

- The surrogate $\hat{\mathbb{E}}[\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)} \hat A]$ has the policy gradient as its gradient at $\theta = \theta_{\text{old}}$.
- TRPO keeps the mean KL between the old and new policies below $\delta$.
- GAE computes advantages from a rollout, and $\hat A_t + V(s_t)$ are the value targets.
- A truncated step bootstraps from the value of the state it reached, and a terminated one does not.
- CartPole pays 1 per step until the pole falls or 500 steps pass.

## Why reusing a batch goes wrong

Write the *probability ratio* of one sample as $\rho_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t \mid s_t)$. It starts at 1 and measures how much more likely the new policy makes the sampled action. The surrogate $\rho_t \hat A_t$ rewards raising $\rho_t$ when $\hat A_t > 0$ and lowering it when $\hat A_t < 0$, and it never says stop. Many epochs of gradient ascent on it push some ratios very far, and the batch's advantages are only trustworthy near $\rho_t = 1$.

The obvious fix is to take fewer steps, but then you are back to one step per batch. The better fix is to make the objective itself go flat once a ratio has moved far enough.

## Clip the ratio

PPO's *clipped surrogate* caps the reward for moving a ratio:

$$
L^{\text{CLIP}}_t(\theta) = \min\Big(\rho_t(\theta)\, \hat A_t,\; \operatorname{clip}\big(\rho_t(\theta), 1 - \epsilon, 1 + \epsilon\big)\, \hat A_t\Big)
$$

Read it piece by piece. $\rho_t \hat A_t$ is the plain surrogate. $\operatorname{clip}(\rho_t, 1 - \epsilon, 1 + \epsilon)$ holds the ratio inside a band around 1, and the paper uses $\epsilon = 0.2$. The $\min$ takes the more pessimistic of the two, which is what makes the clip one-sided.

Consider the two signs of the advantage. When $\hat A_t > 0$, the objective grows with $\rho_t$ until $\rho_t = 1 + \epsilon$ and then stays flat, so there is no reward for making a good action more than 20% likelier. When $\hat A_t < 0$, the objective grows as $\rho_t$ falls, until $\rho_t = 1 - \epsilon$, and then stays flat. In the other direction, the plain term always wins the $\min$. If an update accidentally made a bad action much likelier, the objective keeps its full slope and pulls the ratio back.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="The clipped objective against the probability ratio for advantage +2 and advantage -1: flat beyond 1.2 for a positive advantage and below 0.8 for a negative one, with the four worked-example points marked">
<line x1="70" y1="250.0" x2="560" y2="250.0" class="stroke-grid"/><text x="62" y="254.0" class="fig-small" text-anchor="end">-2</text>
<line x1="70" y1="208.0" x2="560" y2="208.0" class="stroke-grid"/><text x="62" y="212.0" class="fig-small" text-anchor="end">-1</text>
<line x1="70" y1="166.0" x2="560" y2="166.0" class="stroke-muted"/><text x="62" y="170.0" class="fig-small" text-anchor="end">0</text>
<line x1="70" y1="124.0" x2="560" y2="124.0" class="stroke-grid"/><text x="62" y="128.0" class="fig-small" text-anchor="end">1</text>
<line x1="70" y1="82.0" x2="560" y2="82.0" class="stroke-grid"/><text x="62" y="86.0" class="fig-small" text-anchor="end">2</text>
<line x1="70" y1="40.0" x2="560" y2="40.0" class="stroke-grid"/><text x="62" y="44.0" class="fig-small" text-anchor="end">3</text>
<text x="70.0" y="270" class="fig-small" text-anchor="middle">0.5</text>
<text x="168.0" y="270" class="fig-small" text-anchor="middle">0.7</text>
<text x="217.0" y="270" class="fig-small" text-anchor="middle">0.8</text>
<text x="315.0" y="270" class="fig-small" text-anchor="middle">1</text>
<text x="413.0" y="270" class="fig-small" text-anchor="middle">1.2</text>
<text x="462.0" y="270" class="fig-small" text-anchor="middle">1.3</text>
<text x="560.0" y="270" class="fig-small" text-anchor="middle">1.5</text>
<line x1="217.0" y1="40" x2="217.0" y2="250" class="stroke-muted dashed"/>
<line x1="413.0" y1="40" x2="413.0" y2="250" class="stroke-muted dashed"/>
<polyline points="70.0,124.0 74.9,123.2 79.8,122.3 84.7,121.5 89.6,120.6 94.5,119.8 99.4,119.0 104.3,118.1 109.2,117.3 114.1,116.4 119.0,115.6 123.9,114.8 128.8,113.9 133.7,113.1 138.6,112.2 143.5,111.4 148.4,110.6 153.3,109.7 158.2,108.9 163.1,108.0 168.0,107.2 172.9,106.4 177.8,105.5 182.7,104.7 187.6,103.8 192.5,103.0 197.4,102.2 202.3,101.3 207.2,100.5 212.1,99.6 217.0,98.8 221.9,98.0 226.8,97.1 231.7,96.3 236.6,95.4 241.5,94.6 246.4,93.8 251.3,92.9 256.2,92.1 261.1,91.2 266.0,90.4 270.9,89.6 275.8,88.7 280.7,87.9 285.6,87.0 290.5,86.2 295.4,85.4 300.3,84.5 305.2,83.7 310.1,82.8 315.0,82.0 319.9,81.2 324.8,80.3 329.7,79.5 334.6,78.6 339.5,77.8 344.4,77.0 349.3,76.1 354.2,75.3 359.1,74.4 364.0,73.6 368.9,72.8 373.8,71.9 378.7,71.1 383.6,70.2 388.5,69.4 393.4,68.6 398.3,67.7 403.2,66.9 408.1,66.0 413.0,65.2 417.9,64.4 422.8,63.5 427.7,62.7 432.6,61.8 437.5,61.0 442.4,60.2 447.3,59.3 452.2,58.5 457.1,57.6 462.0,56.8 466.9,56.0 471.8,55.1 476.7,54.3 481.6,53.4 486.5,52.6 491.4,51.8 496.3,50.9 501.2,50.1 506.1,49.2 511.0,48.4 515.9,47.6 520.8,46.7 525.7,45.9 530.6,45.0 535.5,44.2 540.4,43.4 545.3,42.5 550.2,41.7 555.1,40.8 560.0,40.0" class="stroke-a dashed" fill="none"/>
<polyline points="70.0,124.0 74.9,123.2 79.8,122.3 84.7,121.5 89.6,120.6 94.5,119.8 99.4,119.0 104.3,118.1 109.2,117.3 114.1,116.4 119.0,115.6 123.9,114.8 128.8,113.9 133.7,113.1 138.6,112.2 143.5,111.4 148.4,110.6 153.3,109.7 158.2,108.9 163.1,108.0 168.0,107.2 172.9,106.4 177.8,105.5 182.7,104.7 187.6,103.8 192.5,103.0 197.4,102.2 202.3,101.3 207.2,100.5 212.1,99.6 217.0,98.8 221.9,98.0 226.8,97.1 231.7,96.3 236.6,95.4 241.5,94.6 246.4,93.8 251.3,92.9 256.2,92.1 261.1,91.2 266.0,90.4 270.9,89.6 275.8,88.7 280.7,87.9 285.6,87.0 290.5,86.2 295.4,85.4 300.3,84.5 305.2,83.7 310.1,82.8 315.0,82.0 319.9,81.2 324.8,80.3 329.7,79.5 334.6,78.6 339.5,77.8 344.4,77.0 349.3,76.1 354.2,75.3 359.1,74.4 364.0,73.6 368.9,72.8 373.8,71.9 378.7,71.1 383.6,70.2 388.5,69.4 393.4,68.6 398.3,67.7 403.2,66.9 408.1,66.0 413.0,65.2 417.9,65.2 422.8,65.2 427.7,65.2 432.6,65.2 437.5,65.2 442.4,65.2 447.3,65.2 452.2,65.2 457.1,65.2 462.0,65.2 466.9,65.2 471.8,65.2 476.7,65.2 481.6,65.2 486.5,65.2 491.4,65.2 496.3,65.2 501.2,65.2 506.1,65.2 511.0,65.2 515.9,65.2 520.8,65.2 525.7,65.2 530.6,65.2 535.5,65.2 540.4,65.2 545.3,65.2 550.2,65.2 555.1,65.2 560.0,65.2" class="stroke-a" fill="none"/>
<polyline points="70.0,187.0 74.9,187.4 79.8,187.8 84.7,188.3 89.6,188.7 94.5,189.1 99.4,189.5 104.3,189.9 109.2,190.4 114.1,190.8 119.0,191.2 123.9,191.6 128.8,192.0 133.7,192.5 138.6,192.9 143.5,193.3 148.4,193.7 153.3,194.1 158.2,194.6 163.1,195.0 168.0,195.4 172.9,195.8 177.8,196.2 182.7,196.7 187.6,197.1 192.5,197.5 197.4,197.9 202.3,198.3 207.2,198.8 212.1,199.2 217.0,199.6 221.9,200.0 226.8,200.4 231.7,200.9 236.6,201.3 241.5,201.7 246.4,202.1 251.3,202.5 256.2,203.0 261.1,203.4 266.0,203.8 270.9,204.2 275.8,204.6 280.7,205.1 285.6,205.5 290.5,205.9 295.4,206.3 300.3,206.7 305.2,207.2 310.1,207.6 315.0,208.0 319.9,208.4 324.8,208.8 329.7,209.3 334.6,209.7 339.5,210.1 344.4,210.5 349.3,210.9 354.2,211.4 359.1,211.8 364.0,212.2 368.9,212.6 373.8,213.0 378.7,213.5 383.6,213.9 388.5,214.3 393.4,214.7 398.3,215.1 403.2,215.6 408.1,216.0 413.0,216.4 417.9,216.8 422.8,217.2 427.7,217.7 432.6,218.1 437.5,218.5 442.4,218.9 447.3,219.3 452.2,219.8 457.1,220.2 462.0,220.6 466.9,221.0 471.8,221.4 476.7,221.9 481.6,222.3 486.5,222.7 491.4,223.1 496.3,223.5 501.2,224.0 506.1,224.4 511.0,224.8 515.9,225.2 520.8,225.6 525.7,226.1 530.6,226.5 535.5,226.9 540.4,227.3 545.3,227.7 550.2,228.2 555.1,228.6 560.0,229.0" class="stroke-b dashed" fill="none"/>
<polyline points="70.0,199.6 74.9,199.6 79.8,199.6 84.7,199.6 89.6,199.6 94.5,199.6 99.4,199.6 104.3,199.6 109.2,199.6 114.1,199.6 119.0,199.6 123.9,199.6 128.8,199.6 133.7,199.6 138.6,199.6 143.5,199.6 148.4,199.6 153.3,199.6 158.2,199.6 163.1,199.6 168.0,199.6 172.9,199.6 177.8,199.6 182.7,199.6 187.6,199.6 192.5,199.6 197.4,199.6 202.3,199.6 207.2,199.6 212.1,199.6 217.0,199.6 221.9,200.0 226.8,200.4 231.7,200.9 236.6,201.3 241.5,201.7 246.4,202.1 251.3,202.5 256.2,203.0 261.1,203.4 266.0,203.8 270.9,204.2 275.8,204.6 280.7,205.1 285.6,205.5 290.5,205.9 295.4,206.3 300.3,206.7 305.2,207.2 310.1,207.6 315.0,208.0 319.9,208.4 324.8,208.8 329.7,209.3 334.6,209.7 339.5,210.1 344.4,210.5 349.3,210.9 354.2,211.4 359.1,211.8 364.0,212.2 368.9,212.6 373.8,213.0 378.7,213.5 383.6,213.9 388.5,214.3 393.4,214.7 398.3,215.1 403.2,215.6 408.1,216.0 413.0,216.4 417.9,216.8 422.8,217.2 427.7,217.7 432.6,218.1 437.5,218.5 442.4,218.9 447.3,219.3 452.2,219.8 457.1,220.2 462.0,220.6 466.9,221.0 471.8,221.4 476.7,221.9 481.6,222.3 486.5,222.7 491.4,223.1 496.3,223.5 501.2,224.0 506.1,224.4 511.0,224.8 515.9,225.2 520.8,225.6 525.7,226.1 530.6,226.5 535.5,226.9 540.4,227.3 545.3,227.7 550.2,228.2 555.1,228.6 560.0,229.0" class="stroke-b" fill="none"/>
<circle cx="462.0" cy="65.2" r="5" class="fill-a"/><text x="454.0" y="55.2" class="fig-small" text-anchor="end">2.4</text>
<circle cx="168.0" cy="107.2" r="5" class="fill-a"/><text x="160.0" y="97.2" class="fig-small" text-anchor="end">1.4</text>
<circle cx="168.0" cy="199.6" r="5" class="fill-b"/><text x="178.0" y="221.6" class="fig-small" text-anchor="start">-0.8</text>
<circle cx="462.0" cy="220.6" r="5" class="fill-b"/><text x="454.0" y="210.6" class="fig-small" text-anchor="end">-1.3</text>
<line x1="90" y1="52" x2="120" y2="52" class="stroke-a"/><text x="128" y="56" class="fig-small">advantage +2, clipped</text>
<line x1="90" y1="72" x2="120" y2="72" class="stroke-b"/><text x="128" y="76" class="fig-small">advantage −1, clipped</text>
<text x="360" y="56" class="fig-small">dashed: plain surrogate ρA</text>
<text x="315" y="294" text-anchor="middle">probability ratio ρ</text><text x="70" y="24" class="fig-small">objective per sample</text>
</svg>
Figure 1. The clipped objective for ε = 0.2 against the ratio ρ, with the plain surrogate dashed. Notice that the objective goes flat past 1.2 when the advantage is positive and below 0.8 when it is negative, and that it keeps its slope where a ratio moved the wrong way.
```

In PyTorch, the per-sample objective is two lines:

```python
ratio = torch.exp(logp_new - logp_old)
obj = torch.min(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv)
```

Where the clipped term is the minimum, `torch.clamp` has zero gradient, so those samples stop contributing to the update.

Take $\epsilon = 0.2$ and four samples, one for each combination of a ratio outside the band and an advantage sign. The last column is the gradient of $L^{\text{CLIP}}$ with respect to $\log \pi_\theta(a \mid s)$, which is $\rho \hat A$ where the plain term is the minimum, and zero otherwise.

| Ratio $\rho$ | Advantage $\hat A$ | $\rho \hat A$ | $\operatorname{clip}(\rho) \hat A$ | $L^{\text{CLIP}}$ | Gradient in $\log \pi$ |
|---|---|---|---|---|---|
| 1.3 | 2 | 2.6 | 2.4 | 2.4 | 0, clipped |
| 0.7 | $-1$ | $-0.7$ | $-0.8$ | $-0.8$ | 0, clipped |
| 1.3 | $-1$ | $-1.3$ | $-1.2$ | $-1.3$ | $-1.3$ |
| 0.7 | 2 | 1.4 | 1.6 | 1.4 | 1.4 |

The first two rows moved the way their advantage wanted and went too far, so the objective stops pushing them. The last two moved the wrong way, so the objective keeps pulling them back at full strength. At $\theta = \theta_{\text{old}}$ every ratio is exactly 1, inside the band, so the first gradient step on a fresh batch is the ordinary policy gradient. The clip matters from the second minibatch onward.

> [!KEY]
> The clipped surrogate takes the minimum of the plain and clipped terms. It removes the incentive to move a ratio past $1 \pm \epsilon$ in the direction the advantage favors. It never stops the objective from pulling a ratio back.

## Add a critic loss and an entropy bonus

PPO usually trains the critic alongside the policy and adds a bonus for keeping the policy random. The full objective, to maximize, is

$$
L_t(\theta) = \hat{\mathbb{E}}_t\big[L^{\text{CLIP}}_t(\theta) - c_1 L^{\text{VF}}_t(\theta) + c_2\, H[\pi_\theta](s_t)\big]
$$

Read it as: the clipped surrogate, minus $c_1$ times the critic's squared error $L^{\text{VF}}_t = (V_\theta(s_t) - \hat R_t)^2$ against the value target, plus $c_2$ times the policy's entropy. Common values are $c_1 = 0.5$ and $c_2 = 0.01$.

The *entropy* of a categorical policy in one state is $H = -\sum_a \pi(a \mid s) \log \pi(a \mid s)$. It is largest, $\log k$, for a uniform policy over $k$ actions and zero for a deterministic one. For $\pi = [0.25, 0.75]$ it is $0.5623$, a little below $\log 2 = 0.6931$. The bonus slows the policy's collapse onto one action before the advantages are reliable.

```python
logp = torch.log_softmax(logits, dim=-1)
entropy = -(logp.exp() * logp).sum(dim=-1)     # log_softmax avoids log(0)
```

## Watch the update with two diagnostics

The clip does not guarantee a small KL, because ratios routinely leave the band during training (Engstrom et al., 2020). So implementations measure how far each update went with two numbers.

The first estimates $\operatorname{KL}(\pi_{\text{old}} \,\|\, \pi_\theta)$ from samples drawn from $\pi_{\text{old}}$. The obvious estimate, the mean of $-\log \rho$, is noisy and single terms can be negative. Schulman's note on KL approximations (2020) suggests the *k3* estimator:

$$
\widehat{\operatorname{KL}} = \hat{\mathbb{E}}\big[(\rho - 1) - \log \rho\big]
$$

Each term is at least zero, because $\log \rho \le \rho - 1$, and the average is still unbiased, because $\rho - 1$ averages to zero under $\pi_{\text{old}}$. For example, with $\pi_{\text{old}} = [0.5, 0.3, 0.2]$ and $\pi_\theta = [0.2, 0.5, 0.3]$, the ratios are $[0.4, 1.667, 1.5]$ and the k3 terms are $[0.3163, 0.1558, 0.0945]$. Weighted by $\pi_{\text{old}}$, they average to $0.2238$, the exact KL.

The second is the *clip fraction*, the share of samples whose ratio left $[1 - \epsilon, 1 + \epsilon]$. As a rule of thumb, a KL estimate well above 0.02 or a clip fraction above 0.3 means the updates are too aggressive.

```python
log_ratio = logp_new - logp_old
approx_kl = (log_ratio.exp() - 1 - log_ratio).mean()
clipfrac = ((log_ratio.exp() - 1).abs() > eps).float().mean()
```

The KL also gives you a brake. The clip does not undo an epoch that has already drifted, so implementations measure the KL after each epoch and skip the remaining epochs once it passes a target such as `target_kl = 0.015`. Compare the updated policy with the stored `logp_old`:

```python
with torch.no_grad():
    kl = approx_kl(logp_current(batch), batch["logp"])   # stored logp_old, never recomputed
if target_kl is not None and kl > target_kl:
    break
```

Figure 2 runs 10 epochs on one seeded batch with a learning rate of 0.05, large on purpose. The full-batch KL after epochs 1 to 3 is 0.0049, 0.0059, and 0.0167, so the stop fires after epoch 3, when 28.7% of the ratios are outside the band, and it saves 7 epochs. Without it the KL wanders up to 0.0441 at epoch 9, where the clip fraction is 0.474. It does not climb steadily, because each epoch shuffles the minibatches differently. The target is a trigger, not a bound.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="Approximate KL after each of 10 epochs on one batch, against a stopping target of 0.015: the KL first exceeds the target at epoch 3, where the early stop ends the update">
<line x1="70" y1="250.0" x2="560" y2="250.0" class="stroke-muted"/><text x="62" y="254.0" class="fig-small" text-anchor="end">0.00</text>
<line x1="70" y1="208.0" x2="560" y2="208.0" class="stroke-grid"/><text x="62" y="212.0" class="fig-small" text-anchor="end">0.01</text>
<line x1="70" y1="166.0" x2="560" y2="166.0" class="stroke-grid"/><text x="62" y="170.0" class="fig-small" text-anchor="end">0.02</text>
<line x1="70" y1="124.0" x2="560" y2="124.0" class="stroke-grid"/><text x="62" y="128.0" class="fig-small" text-anchor="end">0.03</text>
<line x1="70" y1="82.0" x2="560" y2="82.0" class="stroke-grid"/><text x="62" y="86.0" class="fig-small" text-anchor="end">0.04</text>
<line x1="70" y1="40.0" x2="560" y2="40.0" class="stroke-grid"/><text x="62" y="44.0" class="fig-small" text-anchor="end">0.05</text>
<text x="70.0" y="270" class="fig-small" text-anchor="middle">1</text>
<text x="124.4" y="270" class="fig-small" text-anchor="middle">2</text>
<text x="178.9" y="270" class="fig-small" text-anchor="middle">3</text>
<text x="233.3" y="270" class="fig-small" text-anchor="middle">4</text>
<text x="287.8" y="270" class="fig-small" text-anchor="middle">5</text>
<text x="342.2" y="270" class="fig-small" text-anchor="middle">6</text>
<text x="396.7" y="270" class="fig-small" text-anchor="middle">7</text>
<text x="451.1" y="270" class="fig-small" text-anchor="middle">8</text>
<text x="505.6" y="270" class="fig-small" text-anchor="middle">9</text>
<text x="560.0" y="270" class="fig-small" text-anchor="middle">10</text>
<line x1="70" y1="187.0" x2="560" y2="187.0" class="stroke-bad dashed"/>
<text x="556" y="181.0" class="fig-small" text-anchor="end">target_kl = 0.015</text>
<polyline points="70.0,229.2 124.4,225.3 178.9,180.0 233.3,241.4 287.8,190.6 342.2,183.2 396.7,226.9 451.1,185.2 505.6,65.0 560.0,197.8" class="stroke-muted dashed" fill="none"/>
<polyline points="70.0,229.2 124.4,225.3 178.9,180.0" class="stroke-b" fill="none"/>
<circle cx="70.0" cy="229.2" r="4.5" class="fill-b"/>
<circle cx="124.4" cy="225.3" r="4.5" class="fill-b"/>
<circle cx="178.9" cy="180.0" r="4.5" class="fill-b"/>
<circle cx="233.3" cy="241.4" r="4.5" class="fill-muted"/>
<circle cx="287.8" cy="190.6" r="4.5" class="fill-muted"/>
<circle cx="342.2" cy="183.2" r="4.5" class="fill-muted"/>
<circle cx="396.7" cy="226.9" r="4.5" class="fill-muted"/>
<circle cx="451.1" cy="185.2" r="4.5" class="fill-muted"/>
<circle cx="505.6" cy="65.0" r="4.5" class="fill-muted"/>
<circle cx="560.0" cy="197.8" r="4.5" class="fill-muted"/>
<text x="170.9" y="172.0" class="fig-small" text-anchor="end">stop: 0.0167</text>
<text x="315" y="294" text-anchor="middle">epoch on one batch (learning rate 0.05)</text><text x="70" y="24" class="fig-small">full-batch approx_kl after the epoch</text>
</svg>
Figure 2. The full-batch KL after each of 10 epochs on one batch, with the stop at target_kl = 0.015. Notice that the first epoch above the target is epoch 3, and that the dashed epochs after it are the ones the stop skips, one of which reaches 0.0441.
```

## Assemble the training loop

One PPO iteration has two phases. First, run the current policy in $N$ environments for $T$ steps, record every observation, action, log-probability, and value, and compute GAE advantages and value targets. Second, run $K$ epochs over the batch: shuffle the $N T$ samples, split them into minibatches, and take one gradient step per minibatch. If the KL after an epoch passes `target_kl`, skip the remaining epochs.

Huang et al. (2022) cataloged the details that make PPO work. The exercise uses the core ones: advantages normalized per minibatch, Adam with $\epsilon = 10^{-5}$ and a learning rate that falls linearly to zero, gradient norm clipped to 0.5, separate tanh networks for policy and critic, and bootstrapping from $V$ at time limits and at the rollout's end.

For CartPole, the exercise runs 8 environments for 128 steps per iteration, with 4 epochs of 4 minibatches, for 20 iterations, about 20,000 environment steps. Over 20 seeds, the greedy policy after training balanced the pole for between 255 and 500 steps on average. Figure 3 shows seed 0.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="Mean return of the last 10 training episodes over 20 PPO iterations on CartPole, rising from 20 to about 280">
<line x1="70" y1="250.0" x2="560" y2="250.0" class="stroke-grid"/><text x="62" y="254.0" class="fig-small" text-anchor="end">0</text>
<line x1="70" y1="180.0" x2="560" y2="180.0" class="stroke-grid"/><text x="62" y="184.0" class="fig-small" text-anchor="end">100</text>
<line x1="70" y1="110.0" x2="560" y2="110.0" class="stroke-grid"/><text x="62" y="114.0" class="fig-small" text-anchor="end">200</text>
<line x1="70" y1="40.0" x2="560" y2="40.0" class="stroke-grid"/><text x="62" y="44.0" class="fig-small" text-anchor="end">300</text>
<text x="90.0" y="270" class="fig-small" text-anchor="middle">1</text>
<text x="184.7" y="270" class="fig-small" text-anchor="middle">5</text>
<text x="303.2" y="270" class="fig-small" text-anchor="middle">10</text>
<text x="421.6" y="270" class="fig-small" text-anchor="middle">15</text>
<text x="540.0" y="270" class="fig-small" text-anchor="middle">20</text>
<polyline points="90.0,236.1 113.7,231.2 137.4,217.3 161.1,219.6 184.7,193.2 208.4,192.4 232.1,183.3 255.8,150.7 279.5,132.3 303.2,138.1 326.8,112.2 350.5,105.1 374.2,82.5 397.9,52.0 421.6,122.8 445.3,122.0 468.9,102.5 492.6,92.3 516.3,73.0 540.0,55.0" class="stroke-b" fill="none"/>
<circle cx="90.0" cy="236.1" r="3.5" class="fill-b"/>
<circle cx="113.7" cy="231.2" r="3.5" class="fill-b"/>
<circle cx="137.4" cy="217.3" r="3.5" class="fill-b"/>
<circle cx="161.1" cy="219.6" r="3.5" class="fill-b"/>
<circle cx="184.7" cy="193.2" r="3.5" class="fill-b"/>
<circle cx="208.4" cy="192.4" r="3.5" class="fill-b"/>
<circle cx="232.1" cy="183.3" r="3.5" class="fill-b"/>
<circle cx="255.8" cy="150.7" r="3.5" class="fill-b"/>
<circle cx="279.5" cy="132.3" r="3.5" class="fill-b"/>
<circle cx="303.2" cy="138.1" r="3.5" class="fill-b"/>
<circle cx="326.8" cy="112.2" r="3.5" class="fill-b"/>
<circle cx="350.5" cy="105.1" r="3.5" class="fill-b"/>
<circle cx="374.2" cy="82.5" r="3.5" class="fill-b"/>
<circle cx="397.9" cy="52.0" r="3.5" class="fill-b"/>
<circle cx="421.6" cy="122.8" r="3.5" class="fill-b"/>
<circle cx="445.3" cy="122.0" r="3.5" class="fill-b"/>
<circle cx="468.9" cy="102.5" r="3.5" class="fill-b"/>
<circle cx="492.6" cy="92.3" r="3.5" class="fill-b"/>
<circle cx="516.3" cy="73.0" r="3.5" class="fill-b"/>
<circle cx="540.0" cy="55.0" r="3.5" class="fill-b"/>
<text x="315" y="294" text-anchor="middle">PPO iteration (8 environments × 128 steps)</text><text x="70" y="24" class="fig-small">mean return of the last 10 episodes (limit 500)</text>
</svg>
Figure 3. PPO on CartPole with seed 0, 20 iterations and about 20,000 environment steps. Notice that the return climbs from about 20 to about 280 and dips once around iteration 15, which is why you report several seeds.
```

> [!WARNING]
> Record `logp_old` during the rollout, with the policy that acted. If you recompute it from the network during the update, every ratio is 1, the clip never engages, and PPO turns into an unstable multi-epoch policy gradient.

> [!RECAP]
> - `clipped_surrogate` is `torch.min(ratio * adv, clamp(ratio) * adv)`, one value per sample.
> - Entropy uses `log_softmax`, and a uniform policy gives $\log k$.
> - The k3 KL estimate is the mean of $(\rho - 1) - \log \rho$.
> - The loss is $-L^{\text{CLIP}} + c_1 L^{\text{VF}} - c_2 H$, with advantages normalized per minibatch.
- `ppo_epochs` returns the per-epoch KL and the number of epochs run, and stops after the first epoch whose KL, measured against the stored `logp_old`, exceeds `target_kl`.

## Build it step by step

The checks test the functions in this order. The network, `VecCartPole`, `collect_rollout`, `gae`, and `evaluate` are given.

1. **`clipped_surrogate(logp_new, logp_old, advantages, clip_eps)`.** Returns `(N,)`, not a mean. The common mistake is `torch.clamp(ratio * adv, ...)`, which clips the product instead of the ratio.
2. **`categorical_entropy(logits)`.** Returns `(N,)`. The common mistake is `torch.log(torch.softmax(...))`, which gives `nan` when a probability underflows to zero.
3. **`approx_kl(logp_new, logp_old)`** and **`clip_fraction(logp_new, logp_old, clip_eps)`.** Each returns a scalar. The KL check draws 200,000 actions from the old policy and compares with the exact KL.
4. **`ppo_loss(model, batch, ...)`.** Returns a scalar tensor. Gather each action's log-probability, normalize the advantages with `adv.std() + 1e-8`, and combine the three terms with the signs shown in the recap.
5. **`ppo_epochs(model, optimizer, batch, epochs, minibatches, clip_eps, target_kl)`.** Runs the epoch loop on one batch and returns `(kls, epochs_run)`. Measure the full-batch `approx_kl` after each epoch, with no gradient, and stop after the first one above `target_kl`. The common mistake is to recompute the old log-probabilities from the current network, which makes the KL always 0, so it never stops.
6. **`train_ppo(seed, ..., target_kl=None)`.** Returns `(model, history)`. Call `ppo_epochs` once per iteration. The common mistake is to index some batch tensors with the minibatch indices and forget others, which pairs each observation with another sample's advantage.

> [!TIP]
> Test the loss before you train. With `logp_new` equal to `logp_old`, the clipped surrogate's gradient must equal the vanilla policy gradient. A failed learning run tells you almost nothing about which piece is wrong.

## Where this breaks

**The clip does not bound the KL.** Ratios can drift well past $1 \pm \epsilon$ over several epochs, because other samples' gradients move shared parameters. Monitor `approx_kl` and stop the epochs early when it passes a target such as 0.015, as `ppo_epochs` does with `target_kl`.

**Implementation details carry much of the performance.** Engstrom et al. (2020) found that code-level choices, such as reward scaling, orthogonal initialization, and learning-rate annealing, explain much of PPO's advantage over TRPO. OpenAI's reference code also clips the critic's update, which the exercise leaves out, and Andrychowicz et al. (2021) found that it can hurt. [PPO in practice](/lessons/035-ppo-in-practice) carries this clip to a diagonal Gaussian and builds those details. Compare implementations with their details listed, and run several seeds.

**Seed sensitivity.** Even here, 20 seeds gave greedy scores from 255 to 500. Report a spread over seeds, never one run.

**Entropy collapse.** Without the entropy bonus, or with a large learning rate, the policy can become deterministic before it finds the best actions, and the advantages then carry no signal about alternatives.

## Further reading

- [Schulman, Wolski, Dhariwal, Radford, and Klimov (2017), Proximal policy optimization algorithms](https://arxiv.org/abs/1707.06347) introduces the clipped surrogate and the combined objective.
- [Huang, Dossa, Raffin, Kanervisto, and Wang (2022), The 37 implementation details of proximal policy optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) lists every detail of the reference implementation, with code.
- [Engstrom et al. (2020), Implementation matters in deep policy gradients](https://arxiv.org/abs/2005.12729) shows that code-level optimizations drive much of PPO's performance.
- [Schulman (2020), Approximating KL divergence](http://joschu.net/blog/kl-approx.html) compares the k1, k2, and k3 estimators.
- [UC Berkeley CS285, Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/) has Lecture 9, "Advanced Policy Gradients Part 1", and Lecture 10, "Advanced Policy Gradients Part 2", on the surrogate and the step-size limits that PPO approximates.

[Why policy optimization works](/lessons/037-why-policy-optimization-works) proves the identity behind these surrogates.

## The exercise

You build PPO's pieces and then a PPO agent. Write `clipped_surrogate`, `categorical_entropy`, `approx_kl`, and `clip_fraction`, then `ppo_loss`, which combines them on a minibatch, then `ppo_epochs`, which runs the minibatch epochs on one batch and can stop early on the KL, and `train_ppo`, which runs the rollout, GAE, and `ppo_epochs` on CartPole.

The checks reproduce the worked table for both signs of the advantage, including zero gradients on the clipped rows, and confirm that the surrogate's gradient at the old policy is the policy gradient. They check the entropy of a uniform policy, that the k3 estimate averages to the exact KL, the clip fraction, and the combined loss against an independent computation. Two checks run `ppo_epochs` on one seeded batch with a large learning rate. With `target_kl=None` it runs every epoch, and the last KL it reports matches an independent computation against `logp_old`. With `target_kl=0.015` it stops after the first epoch whose full-batch KL exceeds the target, and it reports the same KLs up to that epoch. Finally, `train_ppo` must produce a greedy policy that balances the pole for at least 150 steps on average over 10 episodes.
