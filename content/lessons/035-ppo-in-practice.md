---
title: PPO in practice
slug: 035-ppo-in-practice
part: "Part 5 — Policy gradients"
summary: Score continuous actions with a diagonal Gaussian, then add the normalizers and clips that make PPO train.
minutes: 40
gpu: false
objectives:
  - Compute the log-probability, entropy, and KL of a diagonal Gaussian in closed form.
  - Merge two running mean-and-variance estimates so they match one pass over the combined batch.
  - Normalize advantages per minibatch, and explain why that differs from scaling the reward by the return standard deviation.
  - Anneal the learning rate, clip the value loss, and take one continuous PPO step on numbers you can check by hand.
prereqs:
  - 031-baselines-and-actor-critic
  - 032-generalized-advantage-estimation
  - 034-proximal-policy-optimization
exercise: 035-ppo-in-practice
---

# PPO in practice

A pendulum takes a torque, any real number from $-2$ to $2$. [Proximal policy optimization](/lessons/034-proximal-policy-optimization) trains a discrete CartPole policy with a clipped ratio, and that clip needs a log-probability. A button has one. A torque needs a density.

Use a Gaussian. The network outputs a preferred torque and a width, and the density turns them into a log-probability, so the same clip applies. Engstrom et al. (2020) found that code-level choices account for most of PPO's gain in return over TRPO. You build those choices here: a diagonal Gaussian, a running normalizer, advantage normalization, a learning-rate schedule, a clipped value loss, and a batch from several environments at once.

## What you already know

- The clipped surrogate in [Proximal policy optimization](/lessons/034-proximal-policy-optimization) stops rewarding a ratio that has moved past $1 \pm \epsilon$ in the direction the advantage favors.
- GAE turns a rollout into advantages and value targets.
- An actor-critic method trains the policy with a learned value.
- The ratio uses the log-probability stored when the action was sampled. Recomputing it from the updated network makes every ratio 1.
- Pendulum's observation is $\cos \theta$, $\sin \theta$, and angular velocity. The reward is negative, and the episode ends only by truncation.

## Score a torque with a Gaussian

Picture a hill centered on the torque the network prefers. A sample near the center is ordinary, and a sample in the tail is rare. The width is how widely you explore.

The network outputs the center $\mu(s)$ and $\log \sigma$, and the width is $\sigma = \exp(\log \sigma)$, which stays positive. Action dimensions stay independent: each has its own width, and the joint log-probability is the sum. Huang et al. (2022) use one state-independent $\log \sigma$, started at 0, so each width starts at 1.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="Log-probability of a Gaussian with mean 0.5 and standard deviation 0.5. Removing the constant lifts the curve by 0.919 at every action.">
<line x1="80" y1="250" x2="560" y2="250" class="stroke-muted"/>
<line x1="80" y1="222.1" x2="560" y2="222.1" class="stroke-grid"/><text x="72" y="226.1" class="fig-small" text-anchor="end">-3</text><line x1="80" y1="175.6" x2="560" y2="175.6" class="stroke-grid"/><text x="72" y="179.6" class="fig-small" text-anchor="end">-2</text><line x1="80" y1="129.0" x2="560" y2="129.0" class="stroke-grid"/><text x="72" y="133.0" class="fig-small" text-anchor="end">-1</text><line x1="80" y1="82.5" x2="560" y2="82.5" class="stroke-grid"/><text x="72" y="86.5" class="fig-small" text-anchor="end">0</text><line x1="80" y1="36.0" x2="560" y2="36.0" class="stroke-grid"/><text x="72" y="40.0" class="fig-small" text-anchor="end">1</text>
<line x1="140.0" y1="250" x2="140.0" y2="256" class="stroke-muted"/><text x="140.0" y="272" class="fig-small" text-anchor="middle">0</text><line x1="260.0" y1="250" x2="260.0" y2="256" class="stroke-muted"/><text x="260.0" y="272" class="fig-small" text-anchor="middle">0.5</text><line x1="380.0" y1="250" x2="380.0" y2="256" class="stroke-muted"/><text x="380.0" y="272" class="fig-small" text-anchor="middle">1</text><line x1="500.0" y1="250" x2="500.0" y2="256" class="stroke-muted"/><text x="500.0" y="272" class="fig-small" text-anchor="middle">1.5</text>
<polyline points="80.0,102.6 88.0,98.1 96.0,93.7 104.0,89.6 112.0,85.7 120.0,81.9 128.0,78.4 136.0,75.1 144.0,72.0 152.0,69.1 160.0,66.4 168.0,63.9 176.0,61.7 184.0,59.6 192.0,57.7 200.0,56.1 208.0,54.6 216.0,53.4 224.0,52.4 232.0,51.5 240.0,50.9 248.0,50.5 256.0,50.3 264.0,50.3 272.0,50.5 280.0,50.9 288.0,51.5 296.0,52.4 304.0,53.4 312.0,54.6 320.0,56.1 328.0,57.7 336.0,59.6 344.0,61.7 352.0,63.9 360.0,66.4 368.0,69.1 376.0,72.0 384.0,75.1 392.0,78.4 400.0,81.9 408.0,85.7 416.0,89.6 424.0,93.7 432.0,98.1 440.0,102.6 448.0,107.4 456.0,112.3 464.0,117.5 472.0,122.9 480.0,128.5 488.0,134.2 496.0,140.2 504.0,146.4 512.0,152.9 520.0,159.5 528.0,166.3 536.0,173.3 544.0,180.6 552.0,188.0 560.0,195.7" class="stroke-a dashed" fill="none"/>
<polyline points="80.0,145.4 88.0,140.8 96.0,136.5 104.0,132.3 112.0,128.4 120.0,124.7 128.0,121.2 136.0,117.9 144.0,114.8 152.0,111.9 160.0,109.2 168.0,106.7 176.0,104.4 184.0,102.4 192.0,100.5 200.0,98.8 208.0,97.4 216.0,96.2 224.0,95.1 232.0,94.3 240.0,93.7 248.0,93.3 256.0,93.1 264.0,93.1 272.0,93.3 280.0,93.7 288.0,94.3 296.0,95.1 304.0,96.2 312.0,97.4 320.0,98.8 328.0,100.5 336.0,102.4 344.0,104.4 352.0,106.7 360.0,109.2 368.0,111.9 376.0,114.8 384.0,117.9 392.0,121.2 400.0,124.7 408.0,128.4 416.0,132.3 424.0,136.5 432.0,140.8 440.0,145.4 448.0,150.1 456.0,155.1 464.0,160.2 472.0,165.6 480.0,171.2 488.0,177.0 496.0,183.0 504.0,189.2 512.0,195.6 520.0,202.2 528.0,209.0 536.0,216.1 544.0,223.3 552.0,230.8 560.0,238.4" class="stroke-b" fill="none"/>
<line x1="500.0" y1="143.3" x2="500.0" y2="186.1" class="stroke-ink"/>
<circle cx="500.0" cy="143.3" r="4.5" class="fill-a"/>
<circle cx="500.0" cy="186.1" r="4.5" class="fill-b"/>
<line x1="96" y1="200" x2="128" y2="200" class="stroke-b"/>
<text x="134" y="204" class="fig-small">log-probability</text>
<line x1="96" y1="220" x2="128" y2="220" class="stroke-a dashed"/>
<text x="134" y="224" class="fig-small">constant removed</text>
<text x="320" y="300" text-anchor="middle">action a</text>
<text x="78" y="24" class="fig-small" text-anchor="end">log π(a)</text>
</svg>
Figure 1. Log-probability of a Gaussian with mean 0.5 and standard deviation 0.5. At action 1.5 the log-probability is −2.2258. Dropping the constant lifts that point to −1.3069. The gap, 0.919, is the same at every action.
```

At $\mu = 0.5$, $\sigma = 0.5$, and action $1.5$, the distance is $z = 2$ and the log-probability is $-2.2258$. Dropping $-\frac{1}{2}\log(2\pi)$ scores the same action at $-1.3069$. Figure 1 draws both curves. The gap, $0.9189$, does not depend on $a$, $\mu$, or $\sigma$, so it cancels in a ratio and a ratio-only check stays quiet. The log-probability and the entropy keep it.

> [!WARNING]
> Omitting $-\frac{1}{2}\log(2\pi)$ on each action dimension still produces the right ratio when both log-probabilities omit it. Compare the log-probability with the table, not only the ratio.

The case you implement has two dimensions. The current policy has mean $[0, 1]$ and widths $[1, 0.5]$, the action is $[0, 0]$, and the old policy has mean $[0.5, 1]$ and widths $[1, 1]$. The KL is $\mathrm{KL}(\pi_{\text{old}} \,\|\, \pi)$.

| Dimension | $z$ | $\log \pi(a)$ | Entropy | $\mathrm{KL}(\pi_{\text{old}} \,\|\, \pi)$ |
|---|---|---|---|---|
| 0 | 0 | $-0.9189$ | $1.4189$ | $0.1250$ |
| 1 | $-2$ | $-2.2258$ | $0.7258$ | $0.8069$ |
| Sum |  | $-3.1447$ | $2.1447$ | $0.9319$ |

The gradient of that log-probability in the mean, on dimension 1, is $(0 - 1) / 0.5^2 = -4$. A finite-difference check has to recover that slope.

Write $z_i = (a_i - \mu_i) / \sigma_i$. The three closed forms, summed over action dimensions, are:

$$
\log \pi(a \mid s) = \sum_i \left(-\frac{1}{2}\log(2\pi) - \log \sigma_i - \frac{1}{2} z_i^2\right)
$$

$$
H[\pi](s) = \sum_i \left(\frac{1}{2}\log(2\pi e) + \log \sigma_i\right)
$$

$$
\mathrm{KL}(\pi_{\text{old}} \,\|\, \pi) = \sum_i \left(\log\frac{\sigma_i}{\sigma_{\text{old},i}} + \frac{\sigma_{\text{old},i}^2 + (\mu_{\text{old},i} - \mu_i)^2}{2\sigma_i^2} - \frac{1}{2}\right)
$$

Read them in order: how surprising the action was, how spread out the policy is, and how far the current Gaussian has moved from the one that collected the data. In numpy the log-probability is two lines:

```python
z = (actions - mean) / np.exp(log_std)
logp = (-0.5 * np.log(2 * np.pi) - log_std - 0.5 * z**2).sum(-1)
```

> [!KEY]
> The log-probability, the entropy, and the KL of a diagonal Gaussian are closed forms. The constant $-\frac{1}{2}\log(2\pi)$ per dimension cancels in a ratio and remains in the log-probability and the entropy.

## Keep a running mean and variance

Pendulum mixes a cosine, a sine, and a velocity that can reach 8. Training is steadier when each coordinate arrives with mean near 0 and spread near 1. You learn those moments from the stream you have seen, so you keep a running mean and variance and fold in every batch.

Two batches have to give the same answer as one. Split $[1, 3, 2, 6]$ into $[1, 3]$ and $[2, 6]$. Each piece has a mean and a population variance, the variance that divides by the count:

| Chunk | Count | Mean | Variance |
|---|---|---|---|
| $[1, 3]$ | 2 | 2 | 1 |
| $[2, 6]$ | 2 | 4 | 4 |
| Both | 4 | 3 | 3.5 |

Chan's merge recovers the combined row from the two pieces. Let $\delta$ be the difference of means, here $4 - 2 = 2$, and let $n$ be the total count. $M_2$ is the sum of squared deviations, which is the variance times the count, so the two pieces contribute $M_2$ of $2$ and $8$.

$$
\begin{aligned}
\text{mean} &= \text{mean}_a + \delta \cdot n_b / n \\
M_2 &= M_{2,a} + M_{2,b} + \delta^2 \cdot n_a \cdot n_b / n
\end{aligned}
$$

The mean is $2 + 2 \cdot 2 / 4 = 3$. Then $M_2 = 2 + 8 + 4 \cdot 2 \cdot 2 / 4 = 14$, and the variance is $M_2 / n = 3.5$, the same as one pass. Divide by the count. The unbiased variance does not merge to `numpy.var`. Normalize an observation with $(x - \text{mean}) / \sqrt{\text{var} + \varepsilon}$ and $\varepsilon = 10^{-8}$.

Reward scaling uses a second running variance, and it does not subtract the mean. Keep a discounted return $G \leftarrow \gamma G + r$, fold $G$ into the variance, and divide the reward by $\sqrt{\text{var} + \varepsilon}$. Huang et al. (2022) describe this as the standard deviation of a rolling discounted sum. On the stream $2, 0, 4$ with $\gamma = 0.5$ and $\varepsilon = 10^{-8}$, the returns and the scaled rewards are:

| Step | Reward | Return $G$ | Std of $G$ so far | Scaled reward |
|---|---|---|---|---|
| 0 | 2 | 2 | 0 | 20,000 |
| 1 | 0 | 1 | 0.5 | 0 |
| 2 | 4 | 4.5 | 1.4720 | 2.7175 |

The first reward is $2 / \sqrt{\varepsilon} = 20{,}000$, because one sample has variance 0. After an episode ends, include that reward, then set the return to 0 so the next episode does not inherit it.

## Normalize advantages, and scale rewards, for different reasons

Advantage normalization and reward scaling both change the size of a policy step. They change different objects.

Take advantages $[1, 3, 5]$ and returns $[0, 6, 12]$. The advantages have mean 3 and population standard deviation $1.6330$. The returns have standard deviation $4.8990$, exactly three times larger. Subtract the mean and divide by the advantage standard deviation, and the batch becomes $[-1.2247, 0, 1.2247]$. Divide by the return standard deviation instead, and it becomes $[-0.4082, 0, 0.4082]$.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Three advantages after two normalizations. Dividing by the advantage standard deviation reaches plus or minus 1.225. Dividing by the return standard deviation reaches plus or minus 0.408.">
<line x1="70" y1="135.0" x2="560" y2="135.0" class="stroke-muted"/>
<text x="60" y="202.3" class="fig-small" text-anchor="end">-1</text>
<text x="60" y="139.0" class="fig-small" text-anchor="end">0</text>
<text x="60" y="75.7" class="fig-small" text-anchor="end">1</text>
<rect x="108.0" y="135.0" width="36" height="25.9" class="fill-a"/>
<rect x="156.0" y="135.0" width="36" height="77.6" class="fill-b"/>
<text x="150.0" y="262" class="fig-small" text-anchor="middle">sample 1</text>
<text x="126.0" y="174.9" class="fig-small" text-anchor="middle">-0.408</text>
<text x="174.0" y="226.6" class="fig-small" text-anchor="middle">-1.225</text>
<text x="310.0" y="262" class="fig-small" text-anchor="middle">sample 2</text>
<text x="310.0" y="127.0" class="fig-small" text-anchor="middle">0</text>
<rect x="428.0" y="109.1" width="36" height="25.9" class="fill-a"/>
<rect x="476.0" y="57.4" width="36" height="77.6" class="fill-b"/>
<text x="470.0" y="262" class="fig-small" text-anchor="middle">sample 3</text>
<text x="446.0" y="101.1" class="fig-small" text-anchor="middle">0.408</text>
<text x="494.0" y="49.4" class="fig-small" text-anchor="middle">1.225</text>
<rect x="78" y="28" width="16" height="12" class="fill-a"/>
<text x="100" y="39" class="fig-small">divided by std of returns</text>
<rect x="78" y="48" width="16" height="12" class="fill-b"/>
<text x="100" y="59" class="fig-small">divided by std of advantages</text>
<text x="315" y="288" text-anchor="middle">one minibatch, mean removed</text>
</svg>
Figure 2. One minibatch, centered, then divided two ways. Dividing by the return standard deviation (blue) is one third the length of dividing by the advantage standard deviation (orange). The policy step shrinks by that factor.
```

Reward scaling divides the reward before GAE, so advantages and value targets share a scale and the reward's mean stays. Advantage normalization happens after GAE, inside one minibatch: it recenters the advantages and rescales only the policy loss. Huang et al. (2022) do this per minibatch, and they report that Andrychowicz et al. (2021) found little change in return. Dividing by the return standard deviation still changes the step. Use the population standard deviation, the one `numpy.std` returns:

```python
(advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

> [!KEY]
> Advantage normalization rescales one minibatch of the policy loss. Reward scaling rescales the reward before the advantages exist, and it does not remove the mean.

## Anneal the learning rate and clip the value

The reference continuous-control schedule starts at $3 \times 10^{-4}$ and falls in a straight line toward 0 (Huang et al., 2022). With `update` updates already finished out of `total`, the next rate is

$$
\text{lr} = \text{lr}_0 \left(1 - \frac{\text{update}}{\text{total}}\right)
$$

With $\text{lr}_0 = 0.001$ and a total of 4, the rates are $0.001$, $0.00075$, $0.0005$, $0.00025$, and $0$. A parameter at $0.4$ with loss gradient $2$ moves to $0.398$ on the first update and to $0.399$ when two updates are already finished. Same parameter, same gradient, half the step.

The value clip refuses to credit a critic move past a band of width $\epsilon$ around the value that produced the advantages:

$$
\begin{aligned}
V_{\text{clip}} &= V_{\text{old}} + \operatorname{clip}(V - V_{\text{old}}, -\epsilon, \epsilon) \\
L^{V} &= \max\big((V - \hat R)^2,\; (V_{\text{clip}} - \hat R)^2\big)
\end{aligned}
$$

The max keeps the worse error, so a move past $\epsilon$ stops reducing the loss. Take $\epsilon = 0.2$, an old value of $1$, and a target of $2$.

| Value | Clipped value | Unclipped square | Clipped square | Loss |
|---|---|---|---|---|
| 1.5 | 1.2 | 0.25 | 0.64 | 0.64 |
| 1.1 | 1.1 | 0.81 | 0.81 | 0.81 |

At $1.5$ the clipped branch wins, and it does not depend on the updated value, so the gradient is $0$. At $1.1$ the value is inside the band, both squares equal $0.81$, and the gradient is $2(1.1 - 2) = -1.8$. Engstrom et al. (2020) found no evidence that this clip raises return, and Huang et al. report that Andrychowicz et al. (2021) found it can hurt. The reference code still uses it, so you implement the loss in the table.

## Step several environments at once

One pendulum fills a batch slowly. The rollout in Huang et al. (2022) steps $N$ environments together, which is also how synchronous A2C collects data. You wait for every environment, stack arrays of shape $(T, N)$, and update once. An environment that hits its time limit resets alone. Store the state the step reached, before the reset, as the value target, and let the next step see the reset state. The actor and the critic are the split in the [Hugging Face Deep RL course, unit 6](https://huggingface.co/learn/deep-rl-course/en/unit6/advantage-actor-critic).

## Take one continuous PPO step

For the clipped surrogate, see [Proximal policy optimization](/lessons/034-proximal-policy-optimization). Normalize the advantages, form the ratio from the stored old log-probability, and minimize:

```python
loss = -surrogate.mean() + 0.5 * value_loss - 0.01 * entropy.mean()
```

Fit the value to the returns, not to the normalized advantages. Record the sum of the batch rewards. The check compares that sum with the rewards you passed in. A short step has no Pendulum return cutoff the solution is guaranteed to clear.

> [!KEY]
> A continuous PPO step is the discrete clip, fed by a Gaussian log-probability, with the advantage scale, the learning-rate fraction, and the value clip made explicit.

> [!RECAP]
> - `gaussian_logprob_entropy_kl` returns log-probability, entropy, and $\mathrm{KL}(\pi_{\text{old}} \,\|\, \pi)$.
> - `RunningMeanStd.merge` matches one pass. `scale_reward_stream` divides by the return standard deviation and keeps the mean.
> - `normalize_advantages` divides by the advantage standard deviation. `anneal_learning_rate` falls linearly to 0.
> - `clipped_value_loss` takes the worse squared error. `ppo_continuous_step` minimizes $-\text{clip} + c_1 L^V - c_2 H$ and records the reward sum.

## Build it step by step

The checks call the functions in this order. Population variance divides by the count.

1. **`gaussian_logprob_entropy_kl(mean, log_std, actions, old_mean, old_log_std)`.** Returns log-probability, entropy, and $\mathrm{KL}(\text{old} \,\|\, \text{current})$, each shape `(N,)` and summed over dimensions. `log_std` is `(D,)` or `(N, D)`. The common mistake is dropping $-\frac{1}{2}\log(2\pi)$ per dimension.
2. **`RunningMeanStd`.** `update(batch)` takes shape `(N, *shape)`. `merge` uses Chan's formula. `normalize` returns $(x - \text{mean}) / \sqrt{\text{var} + \varepsilon}$. The common mistake is the unbiased variance.
3. **`scale_reward_stream(rewards, dones, gamma, epsilon=1e-8)`.** Both arrays are `(T, N)`. Set $G \leftarrow \gamma G + r$, divide the reward by the running return standard deviation, then zero $G$ where `dones` is 1. The common mistake is subtracting the mean, or clearing $G$ before that reward.
4. **`normalize_advantages(advantages, epsilon=1e-8)`.** Center the batch and divide by its population standard deviation. The common mistake is dividing by the standard deviation of the returns.
5. **`anneal_learning_rate(update, total_updates, initial_lr)`.** Return `initial_lr * (1 - update / total_updates)`. The common mistake is omitting the `1 -`, so the rate grows.
6. **`clipped_value_loss(values, old_values, targets, clip_eps)`.** Return the mean of the worse squared error. The common mistake is the minimum, which rewards a large move.
7. **`vectorized_rollout(n_envs, actions, seed, max_steps=200)`.** `actions` is `(T, n_envs)`. Start environment `i` with `reset(seed + i)`, and call later resets with no seed. Return `obs`, `rewards`, `final_obs` before the reset, `next_obs` after it, `terminated`, and `truncated`. The common mistake is storing the reset state in `final_obs`.
8. **`ppo_continuous_step(...)`.** Build the loss in the recap from the stored `old_logp`, and put the reward sum in `rollout_return`. The common mistake is fitting the value to the normalized advantages.

> [!TIP]
> Match the two-dimensional table before you assemble the step. At the old policy the ratio is 1 even when the constant is missing, so a training curve does not reveal it.

## Where this breaks

**The value clip changes the loss, and it may not change the return.** Engstrom et al. (2020) found no evidence that it raises return, and Huang et al. report a sweep in which it hurts. Implement it so you can see the loss switch branches.

**A diagonal Gaussian ignores dependence across action dimensions.** Huang et al. (2022) leave a full covariance, or an autoregressive head, as an open alternative.

**Scoring a torque after clipping it into $[-2, 2]$ biases the density.** Score the unclipped sample, and let the environment clip what it applies. The tanh correction for a squashed Gaussian is in [Soft actor-critic](/lessons/040-soft-actor-critic).

**The ratio clip still does not bound the KL.** [Spinning Up's PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) stops an epoch when the mean KL passes a target such as $0.01$. Compare that target with the closed-form KL from this chapter.

## Further reading

- [Huang, Dossa, Raffin, Kanervisto, and Wang (2022), The 37 implementation details of proximal policy optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) catalogs the Gaussian, the normalizers, the value clip, and the vectorized rollout.
- [Engstrom, Ilyas, Santurkar, Tsipras, Janoos, Rudolph, and Mądry (2020), Implementation matters in deep policy gradients](https://arxiv.org/abs/2005.12729) finds that these choices account for most of PPO's gain over TRPO.
- [OpenAI Spinning Up, Proximal policy optimization](https://spinningup.openai.com/en/latest/algorithms/ppo.html) covers continuous actions and early stopping on the KL.
- [Hugging Face Deep RL course, unit 6](https://huggingface.co/learn/deep-rl-course/en/unit6/advantage-actor-critic) separates the actor from the critic, and [unit 8](https://huggingface.co/learn/deep-rl-course/en/unit8/introduction) adds the clipped ratio.
- [CS 185/285, Spring 2026](http://rail.eecs.berkeley.edu/deeprlcourse/) lists Lecture 9 and Lecture 10 as Advanced Policy Gradients, plus Homework 2 and Homework 3. [Fall 2018 homework 2](https://rail.eecs.berkeley.edu/deeprlcourse-fa18/static/homeworks/hw2.pdf) uses a Gaussian mean and a log standard deviation.
- [CMU 10-703, Fall 2025](https://cmudeeprl.github.io/703website_f25/lectures/) lists Lecture 7 and Lecture 8 as Advanced Policy Gradient Methods, and Lecture 7's readings include the PPO paper.

## The exercise

Write `gaussian_logprob_entropy_kl`, `RunningMeanStd`, `scale_reward_stream`, `normalize_advantages`, `anneal_learning_rate`, `clipped_value_loss`, `vectorized_rollout`, and `ppo_continuous_step`.

The checks match the log-probability, entropy, and KL to the table, to finite differences, and to `torch.distributions`. They match a merged running estimate to one pass, scale the three-reward stream, and reject advantages divided by the return standard deviation. They check the learning-rate schedule, the value-clip table, a zero gradient outside the band, and a vectorized Pendulum rollout against independent environments. The PPO step matches the assembled loss and records the reward sum, with no return cutoff.
