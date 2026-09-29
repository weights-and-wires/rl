---
title: Proximal policy optimisation
slug: 19-proximal-policy-optimization
part: "Part 4 — Policy optimisation"
summary: Reuse each batch for several epochs safely with PPO's clipped objective, then train an agent that balances CartPole.
minutes: 70
gpu: false
objectives:
  - Explain why the clipped surrogate removes the incentive to push a ratio past 1 plus or minus epsilon, and only in the direction the advantage favours.
  - Compute the clipped surrogate, the clipped value loss, and the entropy bonus, and check each against hand-computed values.
  - Monitor an update with the k3 estimate of the KL and the clip fraction.
  - Assemble a PPO training loop with GAE, minibatch epochs, advantage normalisation, and gradient clipping, and train it on CartPole.
prereqs:
  - 12-deep-q-networks
  - 17-generalized-advantage-estimation
  - 18-natural-gradients-and-trpo
exercise: 19-proximal-policy-optimization
---

# Proximal policy optimisation

Collecting experience is the slow part of on-policy learning. A policy
gradient method runs the policy for a few thousand steps, takes one gradient
step, and throws the data away, because the next step needs data from the
updated policy. It would be far cheaper to squeeze several gradient steps out
of each batch. But every step moves the policy further from the one that
collected the data, and eventually the batch describes a policy that no
longer exists.

TRPO handled this with a hard limit on the KL divergence, enforced by
conjugate gradient and a line search. It works, and it's complicated.
Proximal policy optimisation (PPO), from Schulman, Wolski, Dhariwal, Radford,
and Klimov (2017), gets most of the same stability with ordinary minibatch
gradient descent. It changes the objective so that moving the policy too far
earns nothing, and then it lets Adam do the rest.

PPO became the default policy gradient method for games, robotics, and the
reinforcement learning stage of training large language models. In this
lesson you build its pieces one at a time, check each against numbers you
can compute by hand, and then assemble a PPO agent that learns to balance
CartPole in about a second of CPU time.

## What you already know

- The surrogate objective
  $\hat{\mathbb{E}}[\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a
  \mid s)} \hat A]$ has the policy gradient as its gradient at
  $\theta = \theta_{\text{old}}$.
- TRPO keeps the mean KL between the old and new policies below $\delta$.
- GAE computes advantages from a rollout, and $\hat A_t + V(s_t)$ are the
  value targets.
- A *truncated* step bootstraps from the value of the state it reached, and a
  *terminated* one doesn't.
- CartPole pays 1 per step until the pole falls or 500 steps pass. You met it
  in *Deep Q-networks*.

## Clip the probability ratio

Write the *probability ratio* of one sample as
$\rho_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t
\mid s_t)$. It starts at 1 and measures how much more likely the new policy
makes the sampled action. The surrogate $\rho_t \hat A_t$ rewards pushing
$\rho_t$ up when $\hat A_t > 0$ and down when $\hat A_t < 0$, without limit.
Many epochs of gradient ascent on it would push some ratios very far.

PPO's *clipped surrogate* caps the reward for moving a ratio:

$$
L^{\text{CLIP}}_t(\theta) = \min\Big(\rho_t(\theta)\, \hat A_t,\;
\operatorname{clip}\big(\rho_t(\theta), 1 - \epsilon, 1 + \epsilon\big)\,
\hat A_t\Big)
$$

Read it piece by piece:

- $\rho_t \hat A_t$ is the plain surrogate.
- $\operatorname{clip}(\rho_t, 1 - \epsilon, 1 + \epsilon)$ holds the ratio
  inside a band around 1. The paper uses $\epsilon = 0.2$.
- The $\min$ takes the more pessimistic of the two. That's what makes the
  clip one-sided.

Consider the two signs of the advantage. When $\hat A_t > 0$, the objective
grows with $\rho_t$ until $\rho_t = 1 + \epsilon$ and then stays flat. There's
no reward for making a good action more than 20% likelier. When
$\hat A_t < 0$, the objective grows as $\rho_t$ falls, until
$\rho_t = 1 - \epsilon$, and then stays flat. But in the other direction, the
unclipped term always wins the $\min$. If an update accidentally made a bad
action much likelier, $\rho_t > 1 + \epsilon$ with $\hat A_t < 0$, the
objective keeps its full slope and pulls the ratio back.

In PyTorch, the per-sample objective is two lines:

```python
ratio = torch.exp(logp_new - logp_old)
obj = torch.min(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv)
```

Where the clipped term is the minimum and the ratio is outside the band,
`torch.clamp` has zero gradient. So those samples stop contributing to the
update. The objective to maximise is the mean of `obj`, and the loss to
minimise is its negative.

### Worked example

Take $\epsilon = 0.2$. The following table shows four samples, one for each
combination of a ratio outside the band and an advantage sign. The last
column is the gradient of $L^{\text{CLIP}}$ with respect to
$\log \pi_\theta(a \mid s)$, which is $\rho \hat A$ when the unclipped term
is the minimum, and zero otherwise.

| Ratio $\rho$ | Advantage $\hat A$ | $\rho \hat A$ | $\operatorname{clip}(\rho) \hat A$ | $L^{\text{CLIP}}$ | Gradient in $\log \pi$ |
|---|---|---|---|---|---|
| 1.3 | 2 | 2.6 | 2.4 | 2.4 | 0, clipped |
| 0.7 | $-1$ | $-0.7$ | $-0.8$ | $-0.8$ | 0, clipped |
| 1.3 | $-1$ | $-1.3$ | $-1.2$ | $-1.3$ | $-1.3$ |
| 0.7 | 2 | 1.4 | 1.6 | 1.4 | 1.4 |

The first two rows moved in the direction their advantage wanted and went too
far, so the objective stops pushing them. The last two moved the wrong way,
so the objective keeps pulling them back at full strength.

At $\theta = \theta_{\text{old}}$, every ratio is exactly 1, which is inside
the band. So the first gradient step on a fresh batch is the ordinary policy
gradient $\hat{\mathbb{E}}[\hat A \nabla \log \pi]$. The clip only matters
from the second minibatch onward.

> [!KEY]
> The clipped surrogate takes the minimum of the plain and clipped terms. It
> removes the incentive to move a ratio past $1 \pm \epsilon$ in the
> direction the advantage favours. It never stops the objective from pulling
> a ratio back.

## Train the value function and keep exploring

PPO usually trains the critic alongside the policy, and adds a bonus for
keeping the policy random. The paper's full objective, to maximise, is

$$
L_t(\theta) = \hat{\mathbb{E}}_t\big[L^{\text{CLIP}}_t(\theta) - c_1
L^{\text{VF}}_t(\theta) + c_2\, H[\pi_\theta](s_t)\big]
$$

Read it as: the clipped surrogate, minus $c_1$ times the critic's squared
error $L^{\text{VF}}_t = (V_\theta(s_t) - \hat R_t)^2$ against the value
target, plus $c_2$ times the policy's entropy. Common values are $c_1 = 0.5$
and $c_2 = 0.01$. With separate networks for the policy and the critic, the
value term only trains the critic, and adding it to one loss is a
convenience.

The *entropy* of a categorical policy in one state is
$H = -\sum_a \pi(a \mid s) \log \pi(a \mid s)$. It's largest, $\log k$, for a
uniform policy over $k$ actions, and zero for a deterministic one. For
$\pi = [0.25, 0.75]$ it's $-(0.25 \log 0.25 + 0.75 \log 0.75) = 0.5623$, a
little below $\log 2 = 0.6931$. The bonus slows the policy's collapse onto
one action before the advantages are reliable.

```python
logp = torch.log_softmax(logits, dim=-1)
entropy = -(logp.exp() * logp).sum(dim=-1)     # log_softmax avoids log(0)
```

### Value clipping

OpenAI's reference implementation also clips the critic's update, an idea
that isn't in the paper's main algorithm. Let $V_{\text{old}}$ be the
prediction recorded during the rollout:

$$
L^{\text{VF}} = \max\Big[\big(V - \hat R\big)^2,\; \big(V_{\text{old}} +
\operatorname{clip}(V - V_{\text{old}}, -\epsilon, \epsilon) - \hat
R\big)^2\Big]
$$

Read it as: limit how far the prediction may move from its old value, and
take the worse of the clipped and unclipped errors. For $V_{\text{old}} = 1$,
$V = 1.5$, $\hat R = 2$, and $\epsilon = 0.2$, the clipped prediction is 1.2.
The loss is $\max(0.25, 0.64) = 0.64$, and its gradient is zero. Engstrom
et al. (2020) found no evidence that value clipping helps, and Andrychowicz et
al. (2021) found that it can hurt. The exercise implements it and
leaves it off by default.

## Watch the update with two diagnostics

The clip doesn't guarantee a small KL. Engstrom et al. (2020) showed that
ratios routinely leave the band during training. So PPO implementations
measure how far each update went, with two numbers.

The first is an estimate of $\operatorname{KL}(\pi_{\text{old}} \,\|\,
\pi_\theta)$ from the samples, which were drawn from $\pi_{\text{old}}$. The
obvious estimate, the mean of $-\log \rho$, is unbiased but noisy, and single
terms can be negative. Schulman's note on KL approximations (2020) suggests
the *k3* estimator instead:

$$
\widehat{\operatorname{KL}} = \hat{\mathbb{E}}\big[(\rho - 1) - \log
\rho\big]
$$

Read it as: the mean of the ratio minus one, minus the log of the ratio.
Each term is at least zero, because $\log \rho \le \rho - 1$. The average is
still unbiased, because $\rho - 1$ averages to zero under $\pi_{\text{old}}$.
For example, with $\pi_{\text{old}} = [0.5, 0.3, 0.2]$ and
$\pi_\theta = [0.2, 0.5, 0.3]$, the ratios are $[0.4, 1.667, 1.5]$ and the
k3 terms are $[0.3163, 0.1558, 0.0945]$. Weighted by $\pi_{\text{old}}$, they
average to $0.2238$, the exact KL.

The second is the *clip fraction*, the share of samples whose ratio left
$[1 - \epsilon, 1 + \epsilon]$. As a rule of thumb, a KL estimate well above
0.02 or a clip fraction above 0.3 means the updates are too aggressive.
Lower the learning rate, use fewer epochs, or stop the epochs early.

```python
log_ratio = logp_new - logp_old
approx_kl = (log_ratio.exp() - 1 - log_ratio).mean()
clipfrac = ((log_ratio.exp() - 1).abs() > eps).float().mean()
```

## Assemble the training loop

One PPO iteration has two phases. First, run the current policy in $N$
environments for $T$ steps each, and record every observation, action,
log-probability, and value. Then compute GAE advantages and value targets.
Second, run $K$ epochs over the batch. Each epoch shuffles the $N T$ samples,
splits them into minibatches, and takes one gradient step per minibatch.

Huang et al. (2022) catalogued the details that make PPO work as well as the
reference implementation. The exercise uses the core ones:

- Advantages are normalised to zero mean and unit standard deviation within
  each minibatch, after the value targets are computed.
- Adam uses $\epsilon = 10^{-5}$, and the learning rate falls linearly to
  zero over training.
- The gradient's global norm is clipped to 0.5.
- The policy and critic are separate small networks with tanh activations
  and orthogonal initialisation. The policy's last layer starts with gain
  0.01, so the initial policy is close to uniform.
- The rollout bootstraps from $V$ at both time limits and the rollout's end,
  as in the previous lesson.

For CartPole, the exercise runs 8 environments for 128 steps per iteration,
4 epochs of 4 minibatches, for 20 iterations. That's about 20,000
environment steps. The learning rate starts at $2.5 \times 10^{-3}$, ten
times the usual default, to fit the check's time budget. Over 20 seeds, the
greedy policy after training balanced the pole for between 212 and 500 steps
on average.

> [!WARNING]
> Record `logp_old` during the rollout, with the policy that acted. If you
> recompute it from the network during the update, every ratio is 1, the clip
> never engages, and PPO turns into an unstable multi-epoch policy gradient.

> [!TIP]
> Test the loss before you train. With `logp_new` equal to `logp_old`, the
> clipped surrogate's gradient must equal the vanilla policy gradient. A
> failed learning run tells you almost nothing about which piece is wrong.

> [!RECAP]
> - `clipped_surrogate` is `torch.min(ratio * adv, clamp(ratio) * adv)`, one
>   value per sample.
> - The clipped value loss is the mean of the larger squared error.
> - Entropy uses `log_softmax`, and a uniform policy gives $\log k$.
> - The k3 KL estimate is the mean of $(\rho - 1) - \log \rho$.
> - The loss is $-L^{\text{CLIP}} + c_1 L^{\text{VF}} - c_2 H$, with
>   advantages normalised per minibatch.

## Build it step by step

The checks test the functions in this order. The network, `VecCartPole`,
`collect_rollout`, `gae`, and `evaluate` are given.

1. **`clipped_surrogate(logp_new, logp_old, advantages, clip_eps)`.** Returns
   `(N,)`, not a mean. Common mistake: `torch.clamp(ratio * adv, ...)`, which
   clips the product instead of the ratio. The check compares every row of
   the worked table and its gradient.
2. The second check confirms that your surrogate's gradient at
   $\theta = \theta_{\text{old}}$ is the policy gradient. It needs nothing
   new.
3. **`clipped_value_loss(values, old_values, returns, clip_eps)`.** Returns a
   scalar. Build the clipped prediction with
   `old_values + torch.clamp(values - old_values, -eps, eps)`.
4. **`categorical_entropy(logits)`.** Returns `(N,)`. Common mistake: using
   `torch.log(torch.softmax(...))`, which gives `nan` when a probability
   underflows to zero.
5. **`approx_kl(logp_new, logp_old)`** and **`clip_fraction(logp_new,
   logp_old, clip_eps)`.** Each returns a scalar. The KL check draws 200,000
   actions from the old policy and compares with the exact KL.
6. **`ppo_loss(model, batch, ...)`.** Returns a scalar tensor. Gather each
   action's log-probability, normalise the advantages with
   `adv.std() + 1e-8`, and combine the three terms with the signs shown in
   the recap.
7. **`train_ppo(seed, ...)`.** Returns `(model, history)`. Follow the
   docstring's four steps. Common mistake: indexing some tensors in the batch
   with the minibatch indices and forgetting others, which pairs each
   observation with another sample's advantage. The check trains with seed 0
   and requires the greedy policy to average at least 150 steps.

## Cost and scaling

Each iteration costs one rollout and $K \times M$ minibatch gradient steps,
where $M$ is the number of minibatches. The following table gives the costs
for $N$ environments, $T$ steps, and a network with $P$ parameters:

| Phase | Time | Memory |
|---|---|---|
| Rollout | $N T$ environment steps and forward passes | $O(N T)$ for the batch |
| GAE | $T$ sequential vector steps of size $N$ | $O(N T)$ |
| Update | $K$ epochs over $N T$ samples, $O(K N T P)$ | $O(P)$ plus one minibatch's graph |

The rollout is usually the bottleneck, which is why PPO reuses each batch for
several epochs. In the exercise, one training run takes about a second on one
CPU thread. On large problems, the rollout runs on many machines, and the
update phase on accelerators.

## Where this breaks

**The clip doesn't bound the KL.** Ratios can drift well past $1 \pm
\epsilon$ over several epochs, because other samples' gradients move shared
parameters. Monitor `approx_kl` and stop the epochs early when it passes a
target, such as 0.015.

**Implementation details carry much of the performance.** Engstrom et al.
(2020) found that code-level choices, including value clipping, reward
scaling, orthogonal initialisation, and learning-rate annealing, explain much
of PPO's advantage over TRPO. A PPO with only the clipped objective performed
noticeably worse. Compare implementations with their details listed, and run
several seeds.

**Seed sensitivity.** Even in the exercise, 20 seeds gave greedy scores
between 212 and 500. Report a spread over seeds, never one run. *Evaluating
reinforcement learning* covers how.

**Stale data.** If rollouts come from an older copy of the policy than the
one that recorded `logp_old`, the ratios are wrong. This is a real risk in
distributed training, and *Infrastructure for LLM reinforcement learning*
returns to it.

**Entropy collapse.** Without the entropy bonus, or with a large learning
rate, the policy can become deterministic before it finds the best actions,
and the advantages then carry no signal about alternatives.

## Further reading

- [Schulman, Wolski, Dhariwal, Radford, and Klimov (2017), Proximal policy optimization algorithms](https://arxiv.org/abs/1707.06347)
  introduces the clipped surrogate, the combined objective, and the
  adaptive-KL variant that performed worse in its experiments.
- [Huang, Dossa, Raffin, Kanervisto, and Wang (2022), The 37 implementation details of proximal policy optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
  lists every detail of the reference implementation, with code.
- [Engstrom et al. (2020), Implementation matters in deep policy gradients: a case study on PPO and TRPO](https://arxiv.org/abs/2005.12729)
  shows that code-level optimisations drive much of PPO's performance and that
  the clip doesn't keep ratios in the band.
- [Andrychowicz et al. (2021), What matters in on-policy reinforcement learning? A large-scale empirical study](https://arxiv.org/abs/2006.05990)
  tests many PPO design choices across a large number of training runs.
- [Schulman (2020), Approximating KL divergence](http://joschu.net/blog/kl-approx.html)
  compares the k1, k2, and k3 estimators of the KL from samples.

The next lesson, *Why policy optimisation works*, proves the identity behind
all these surrogates. Part 8 applies PPO to language models in
*RLHF with PPO*.

## The exercise

You build PPO's pieces and then a PPO agent. Write `clipped_surrogate`,
`clipped_value_loss`, `categorical_entropy`, `approx_kl`, and
`clip_fraction`. Then write `ppo_loss`, which combines them on a minibatch,
and `train_ppo`, which runs the rollout, GAE, and minibatch epochs on
CartPole.

The checks reproduce the worked table for both signs of the advantage,
including zero gradients on the clipped rows. They confirm that the
surrogate's gradient at the old policy is the policy gradient. They check the
value-clipping example, the entropy of a uniform policy, and that the k3
estimate averages to the exact KL. They check the clip fraction and the
combined loss against an independent computation. Finally, `train_ppo` must
produce a greedy policy that balances the pole for at least 150 steps on
average over 10 episodes.
