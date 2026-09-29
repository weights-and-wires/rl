---
title: RLHF with PPO
slug: 35-rlhf-with-ppo
part: "Part 8 — Reinforcement learning for language models"
summary: Train a language model with PPO, a value head, and a per-token KL penalty, from the toy policy up to Qwen3-0.6B.
minutes: 75
gpu: true
objectives:
  - Spread the KL penalty over tokens so the task reward lands on the last token and the return equals reward minus beta times the log-ratio.
  - Compare the k1, k2, and k3 KL estimators, and say which KL each one's gradient follows when you use it as a loss.
  - Compute GAE over response tokens with a value head, and assemble a clipped PPO update with a clipped value loss.
  - Run PPO on a toy policy to within a few percent of the exact KL-regularised optimum, and on Qwen3-0.6B against a rule-based reward.
prereqs:
  - 17-generalized-advantage-estimation
  - 19-proximal-policy-optimization
  - 33-language-models-as-policies
  - 34-reward-models
exercise: 35-rlhf-with-ppo
---

# RLHF with PPO

Reinforcement learning from human feedback (RLHF) made chat models follow
instructions. The recipe has three stages: fine-tune on example answers, fit a
reward model on comparisons, then run PPO against that reward while a penalty
keeps the model near where it started. This lesson builds the third stage.

You already know PPO from Part 4 and the leashed objective from *Language
models as policies*. Putting them together takes a few ideas that are specific
to text. The penalty is paid token by token. The reward arrives only at the end.
The critic is a small layer bolted onto the language model. And the exact KL is
out of reach, so you estimate it from the sampled tokens, which has traps of its
own.

You build each piece as a function on plain tensors. The checks run them on a
toy policy, where PPO gets within a few percent of the exact optimum. On the GPU
worker, the same functions train Qwen3-0.6B to describe a topic in exactly ten
words.

## What you already know

- The KL-regularised objective $\mathbb{E}[r] - \beta\, \mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})$
  has the exact optimum $\pi^* \propto \pi_{\text{ref}} \exp(r / \beta)$, with
  value $\beta \log Z$.
- A reward model scores a finished response with one number, $r_\phi(x, y)$.
- PPO maximises a clipped surrogate of the probability ratio times the
  advantage, from *Proximal policy optimisation*.
- GAE computes $\hat A_t = \delta_t + \gamma\lambda \hat A_{t+1}$ backwards from
  TD errors, from *Generalised advantage estimation*.
- A response's tokens, log-probabilities, and mask are `(B, T)` tensors, and
  its reward is one entry of a `(B,)` tensor.

## Put the KL penalty in the reward

In the token-level MDP, the state is the prompt plus the tokens so far, and the
action is the next token. The KL between whole responses splits into a sum of
per-token log-ratios. So you can charge the penalty one token at a time, and pay
the reward model's score on the last token. Ziegler and colleagues (2019),
Stiennon and colleagues (2020), and InstructGPT all do this:

$$
r_t = -\beta\, \big(\log \pi_{\text{old}}(y_t \mid s_t) - \log \pi_{\text{ref}}(y_t \mid s_t)\big) + \mathbf{1}[t = T - 1]\; r_\phi(x, y)
$$

Read it as: every token pays $\beta$ times how much more likely the current
policy made it than the reference did. The last token, $t = T - 1$, also
collects the task reward. $\pi_{\text{old}}$ is the policy that sampled the
batch. Add the $r_t$ up and you get $r_\phi(x, y) - \beta \log(\pi_{\text{old}}(y) / \pi_{\text{ref}}(y))$,
whose average is the KL-regularised objective. In torch:

```python
rewards = -beta * (logp - ref_logp) * mask
last = mask.sum(dim=1).long() - 1
rewards[torch.arange(B), last] += seq_rewards
```

### Worked example

A response has three tokens, padded to four. The policy's log-probabilities are
$(-0.5, -1.0, -0.2)$ and the reference's are $(-0.7, -0.9, -0.6)$. The reward
model gives $1$, and $\beta = 0.1$.

- Log-ratios: $(0.2, -0.1, 0.4)$.
- KL terms: $(-0.02, +0.01, -0.04)$.
- Shaped rewards: $(-0.02, 0.01, 0.96, 0)$. The task reward lands on the third
  token, not on the padding.
- The sum is $0.95 = 1 - 0.1 \times 0.5$.

The second token earns a tiny bonus, because the policy made it *less* likely
than the reference did. A single token's log-ratio can have either sign. Only
its average over the policy's samples is a KL, and that's never negative.

Ziegler and colleagues also adapt $\beta$ during training. With a target KL
$\mathrm{KL}_{\text{target}}$, they set
$e_t = \operatorname{clip}\big((\mathrm{KL}_t - \mathrm{KL}_{\text{target}}) / \mathrm{KL}_{\text{target}}, -0.2, 0.2\big)$
and $\beta_{t+1} = \beta_t (1 + 0.1\, e_t)$. The exercise keeps $\beta$ fixed.

## Estimate the KL from samples

The per-token log-ratio is one of three single-sample estimators that Schulman
(2020) compares. Let $q = \pi_\theta$ be the policy you sample from and
$p = \pi_{\text{ref}}$. For a sampled token, let the ratio be
$\rho = p(y_t) / q(y_t)$. Then:

$$
k_1 = -\log \rho,
\qquad
k_2 = \tfrac{1}{2} (\log \rho)^2,
\qquad
k_3 = (\rho - 1) - \log \rho
$$

Read it as: $k_1$ is the log-ratio itself, $k_2$ is half its square, and $k_3$
adds $\rho - 1$, a term whose average under $q$ is exactly zero. So $k_1$ and
$k_3$ are both unbiased for $\mathrm{KL}(q \,\|\, p)$. $k_3$ is never negative,
because $\log \rho \le \rho - 1$, and it usually has much lower variance when $q$
and $p$ are close. $k_2$ is biased, but it matches the KL to second order when
$q \approx p$.

```python
log_r = ref_logp - logp
k1, k2, k3 = -log_r, 0.5 * log_r**2, torch.expm1(log_r) - log_r
```

### Worked example

Take three tokens with $q = (0.5, 0.3, 0.2)$ and $p = (0.3, 0.3, 0.4)$. The
exact $\mathrm{KL}(q \,\|\, p) = 0.1168$. The estimators on each token are:

| Estimator | Token 1 | Token 2 | Token 3 | Mean under $q$ | Variance under $q$ |
|---|---|---|---|---|---|
| $k_1$ | 0.5108 | 0 | $-0.6931$ | 0.1168 | 0.2129 |
| $k_2$ | 0.1305 | 0 | 0.2402 | 0.1133 | 0.0072 |
| $k_3$ | 0.1108 | 0 | 0.3069 | 0.1168 | 0.0113 |

$k_3$ has a variance 19 times smaller than $k_1$'s, with no bias. Here $k_2$'s
variance is lower still, but it pays with bias. None of them is best in every
case.

### Estimators as losses

There's a second question: what happens if you put the estimator in the loss
and backpropagate? Autodiff differentiates the estimator at the sampled tokens.
It doesn't differentiate through the sampling. Work out the expected gradient
under $q_\theta$ for each:

- $k_1$: $\mathbb{E}_q[\nabla \log q] = 0$. As a loss it does nothing on
  average.
- $k_2$: $\mathbb{E}_q[\log(q / p)\, \nabla \log q] = \nabla\, \mathrm{KL}(q \,\|\, p)$,
  the gradient of the reverse KL the objective asks for.
- $k_3$: $\mathbb{E}_q[(1 - p / q)\, \nabla \log q] = \nabla\, \mathrm{KL}(p \,\|\, q)$,
  the gradient of the *forward* KL.

So an estimator can be unbiased for the KL's value and still follow the
gradient of a different quantity. Using $k_1$ in the *reward*, as this lesson
does, is also correct: the policy gradient sends it through the score function,
which gives the reverse-KL gradient. GRPO puts $k_3$ in the loss, which
*Group-relative policy optimisation* returns to. Shah and colleagues (2025)
study how these choices change training.

> [!KEY]
> $k_1$ and $k_3$ are unbiased for the KL's value. As loss terms, $k_1$ has zero
> expected gradient, $k_2$ follows the reverse KL, and $k_3$ follows the forward
> KL. Put $k_1$ in the reward, or $k_2$ in the loss.

## Assign credit with a value head

The reward model speaks once, at the end. PPO needs an advantage for every
token, so it needs a critic. The *value head* is a linear layer that maps the
language model's last hidden state at each position to one number, $V(s_t)$.
The hidden state at the position before token $t$ has read the prompt and every
earlier token, so it's a summary of the state $s_t$. Many implementations
initialise the value head from the reward model.

GAE then runs over the response tokens, as in *Generalised advantage
estimation*. RLHF usually sets $\gamma = 1$ and $\lambda = 0.95$. The response
ends after its last token, so the value after it is zero. Padded positions get
no advantage and no return:

$$
\delta_t = r_t + \gamma\, m_{t+1} V(s_{t+1}) - V(s_t),
\qquad
\hat A_t = \delta_t + \gamma \lambda\, m_{t+1} \hat A_{t+1}
$$

Read it as: the usual TD error and backward recursion, where $m_{t+1}$ is the
mask of the next position. It's 0 after the last response token, which turns
off both the bootstrap and the recursion. The value targets are
$\hat R_t = \hat A_t + V(s_t)$.

A response that hits the length limit without an end-of-sequence token is also
treated as finished, because the reward model scores it as it stands. Huang and
colleagues (2024) recommend penalising such responses, and the GPU run gives
them a task reward of 0.

### Worked example

Continue with the shaped rewards $(-0.02, 0.01, 0.96)$. Let the value head
predict $V = (0.5, 0.6, 0.8)$, with $\gamma = 1$ and $\lambda = 0.95$.

- $\delta_2 = 0.96 + 0 - 0.8 = 0.16$, with no bootstrap after the last token.
- $\delta_1 = 0.01 + 0.8 - 0.6 = 0.21$, and $\delta_0 = -0.02 + 0.6 - 0.5 = 0.08$.
- $\hat A_2 = 0.16$, $\hat A_1 = 0.21 + 0.95 \times 0.16 = 0.362$, and
  $\hat A_0 = 0.08 + 0.95 \times 0.362 = 0.4239$.
- Returns: $\hat R = (0.9239, 0.962, 0.96)$.

With $\lambda = 1$ you'd get the reward-to-go minus $V$:
$(0.95 - 0.5, 0.97 - 0.6, 0.96 - 0.8) = (0.45, 0.37, 0.16)$.

## Clip the update

The policy loss is PPO's clipped surrogate from *Proximal policy optimisation*,
with one ratio per token, $\rho_t = \pi_\theta(y_t \mid s_t) / \pi_{\text{old}}(y_t \mid s_t)$:

$$
L^{\text{CLIP}}(\theta) = -\frac{1}{\sum_{b,t} m_{b,t}} \sum_{b,t} m_{b,t} \min\big(\rho_{b,t} \hat A_{b,t},\; \operatorname{clip}(\rho_{b,t}, 1 - \epsilon, 1 + \epsilon)\, \hat A_{b,t}\big)
$$

Read it as: for every response token in the batch, take the smaller of the
unclipped and clipped products, then average over all response tokens and
negate. This is a *per-token mean*: a response with more tokens gets more
weight. The alternative, averaging within each response and then across
responses, weights every response equally. The choice matters, and
*Group-relative policy optimisation* shows how much.

With $\epsilon = 0.2$, the clip decides which tokens still get a gradient:

| $\rho_t$ | $\hat A_t$ | Unclipped | Clipped | Loss term | Gradient flows? |
|---|---|---|---|---|---|
| 1.3 | +1 | 1.3 | 1.2 | $-1.2$ | No: already pushed up enough |
| 0.7 | +1 | 0.7 | 0.8 | $-0.7$ | Yes |
| 0.7 | $-1$ | $-0.7$ | $-0.8$ | 0.8 | No: already pushed down enough |
| 1.3 | $-1$ | $-1.3$ | $-1.2$ | 1.3 | Yes, without limit |

The value loss clips too. With old prediction $V_{\text{old}}$, it takes the
larger of the plain squared error and the error of a prediction held within
$\epsilon$ of $V_{\text{old}}$, and halves the mean. With $V_{\text{old}} = 1$,
$V = 1.5$, $\hat R = 2$, and $\epsilon = 0.2$, that's
$0.5 \max(0.25, (1.2 - 2)^2) = 0.32$.

```python
ratio = torch.exp(logp - old_logp)
surr = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv)
policy_loss = -(surr * mask).sum() / mask.sum()
```

The inner loop, `ppo_update`, runs a few epochs over the batch. Each epoch
shuffles the responses into minibatches. For each minibatch it recomputes the
log-probabilities and values, adds the policy loss to `vf_coef` times the value
loss, clips the global gradient norm, and steps. Advantages are whitened, to
mean 0 and variance 1 over response tokens, before the loop starts.

> [!KEY]
> RLHF with PPO is ordinary PPO in the token MDP. KL-shaped rewards go into GAE
> with a value head, and the clipped losses average over response tokens under
> the mask.

## Run it on Qwen3-0.6B

On the GPU worker, **Run** trains Qwen3-0.6B with your functions. The prompt
asks the model to describe a topic in exactly ten words. A Python rule replaces
the reward model: a response of exactly ten words scores 1, and each word too
many or too few costs 0.1. Qwen3-0.6B, in non-thinking mode, writes sentences of
anywhere from 3 to 20 words, so there's room to improve.

The run uses the following settings, each for a reason:

- **Sampling at temperature 1 with `top_p=1.0` and `top_k=0`.** Qwen3's default
  generation settings truncate the distribution. The log-probabilities in the
  loss then describe a different policy from the one that sampled.
- **A mask up to the first end-of-sequence token.** Everything after it is
  padding. The attention mask and the position IDs come from the same mask, so
  the training forward pass sees exactly what generation saw.
- **Float32 master weights, bfloat16 compute.** bfloat16 keeps about three
  significant digits, so most weights can't represent an update of
  $10^{-6}$ and wouldn't change. The frozen reference stays in
  bfloat16.
- **A value head on detached hidden states.** The value loss then trains only
  the head, and can't pull the language model's representation around.
- **A small learning rate, $10^{-6}$.** Adam's first steps move every weight by
  about the learning rate, whatever its gradient. Across 600 million weights
  that adds up: in a test with four responses per batch, one step moved some
  token ratios past 2. Larger batches average out that noise.

The run prints mean reward, the fraction of exactly-ten-word responses, the
mean word count, the KL, and the clip fraction every few steps. Expect the mean
reward to start near 0.7 and climb within the first dozen steps, while the KL
grows to a few nats.

> [!WARNING]
> If the clip fraction at the first minibatch of the first step isn't 0, your
> old log-probabilities don't match your training forward pass. Look for a
> sampling setting, a dtype, dropout, or a position-ID difference.

> [!RECAP]
> - Shaped rewards are $-\beta$ times the log-ratio on every response token,
>   plus the task reward on the last one.
> - $k_1$ and $k_3$ are unbiased for the KL. As losses, $k_1$'s expected gradient
>   is 0, $k_2$'s is the reverse-KL gradient, and $k_3$'s is the forward-KL one.
> - GAE runs over tokens with $m_{t+1}$ switching off the bootstrap after the
>   last token.
> - Both losses are per-token means under the mask.
> - `ppo_update` shuffles, recomputes, sums the losses, clips the gradient norm,
>   and steps.

## Build it step by step

The checks test the functions in this order. Every function takes and returns
torch tensors. The starter provides `ToyPolicy`, `rising_reward`,
`train_toy_ppo`, and `real_run`, which call your functions.

1. **`kl_shaped_rewards(logp, ref_logp, seq_rewards, mask, beta)`.** Returns
   `(B, T)`. Find each row's last response index with `mask.sum(dim=1) - 1`.
   Common mistake: adding the task reward at index `T - 1`, which is padding for
   short responses.
2. **`kl_estimators(logp, ref_logp)`.** Returns `(k1, k2, k3)`. Use
   `torch.expm1` for $\rho - 1$. Common mistake: flipping the ratio, so $k_1$
   comes out as the negative log-ratio. The third check puts each estimator in a
   loss and compares exact expected gradients.
3. **`gae(rewards, values, mask, gamma, lam)`.** Returns `(advantages,
   returns)`. Loop backwards over `t`. Common mistake: bootstrapping from
   `values[:, t + 1]` at the last response token, where it's padding.
4. **`masked_whiten(x, mask)`.** Returns `(B, T)` with zeros on padding. Use the
   variance over $N$, not $N - 1$.
5. **`ppo_policy_loss(logp, old_logp, advantages, mask, clip_eps)`.** Returns a
   scalar. Common mistake: `torch.mean` over the whole tensor, which counts
   padding.
6. **`value_loss(values, old_values, returns, mask, clip_eps)`.** Returns a
   scalar, with the 0.5 factor.
7. **`ppo_update(forward, optimizer, batch, ...)`.** Returns a dict of
   averaged `policy_loss`, `value_loss`, and `clip_frac`. Common mistake:
   clipping the gradient norm of only some parameters. The check takes one step
   with SGD and compares it with the exact gradient of the combined loss.
8. The last check runs 40 PPO iterations on the toy policy with your functions.
   It computes the exact objective of the result, and the exact optimum
   $\beta \log Z$ by soft backward induction, and expects PPO to close at least
   85 percent of the gap from the reference.

## Cost and scaling

RLHF with PPO holds four models: the policy, the value head or a separate
critic, the reward model, and the reference. The following table gives rough
costs for a batch of $B$ responses of $T$ tokens from a model with $P$
parameters:

| Stage | Time | Memory |
|---|---|---|
| Generation | $T$ sequential forward passes | KV cache, $O(B T)$ per layer |
| Scoring with policy and reference | Two forward passes | Logits, $O(B T V)$ |
| Reward model | One forward pass | One more model |
| PPO epochs | $E$ times forward plus backward | Weights, gradients, and Adam state: about $16 P$ bytes in float32 |

Generation dominates, because it's sequential. For Qwen3-0.6B, float32 weights,
gradients, and Adam state take about 10 GB, and the bfloat16 reference adds
1.2 GB. The logits are large too: $V$ is about 150,000, so compute the
log-softmax one minibatch at a time. A separate critic the size of the policy
doubles training memory. That's one reason critic-free methods became popular.

## Where this breaks

**Reward hacking.** A learned reward has errors, and PPO finds them: longer
answers, flattery, and formatting tricks. The KL penalty slows it. Gao and
colleagues (2022) found that in their setup, the penalty acted mostly like early
stopping.

**Mismatched log-probabilities.** Sampling with one temperature and scoring
with another, dropout in training mode, a bfloat16 engine against a float32
trainer, or wrong position IDs all make $\rho \ne 1$ before any update. The
first minibatch's clip fraction is the test.

**A critic that lags.** Early in training the value head knows nothing, so GAE
with $\lambda < 1$ leans on bad values. Warm up the value head first, or start
from the reward model's weights.

**Too large a step.** Adam normalises each weight's update, so a large model
moves a lot even at a tiny learning rate. Watch the clip fraction and the KL.
Lower the learning rate or add warmup if either jumps.

**Length bias.** The per-token mean gives long responses more weight, and the
KL penalty grows with length. Both push the length around for reasons that have
nothing to do with the task.

## Further reading

- [Schulman et al. (2017), Proximal policy optimization algorithms](https://arxiv.org/abs/1707.06347)
  introduces the clipped surrogate.
- [Ziegler et al. (2019), Fine-tuning language models from human preferences](https://arxiv.org/abs/1909.08593)
  puts the KL penalty in the reward and adapts $\beta$ toward a target KL.
- [Stiennon et al. (2020), Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)
  scales the recipe to summarisation, with a human-preference reward model.
- [Ouyang et al. (2022), Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
  is InstructGPT, which adds a pretraining loss to limit regressions.
- [Huang et al. (2024), The N+ implementation details of RLHF with PPO](https://arxiv.org/abs/2403.17031)
  lists the details that decide whether a reproduction works.
- [Schulman (2020), Approximating KL divergence](http://joschu.net/blog/kl-approx.html)
  defines $k_1$, $k_2$, and $k_3$ and compares their bias and variance.
- [Shah et al. (2025), A comedy of estimators: on KL regularization in RL training of LLMs](https://arxiv.org/abs/2512.21852)
  shows that common KL setups don't give the gradient of the stated objective.

Open questions researchers are working on:

- Is the critic worth its memory for language models, or do group baselines do
  as well?
- Which KL estimator, in the reward or in the loss, gives stable training at
  scale?
- How should the loss aggregate over tokens so that length doesn't leak into
  the objective?

## The exercise

You build the RLHF-specific pieces of PPO. Write `kl_shaped_rewards`,
`kl_estimators`, `gae`, `masked_whiten`, `ppo_policy_loss`, `value_loss`, and
`ppo_update`. They work on `(B, T)` tensors, so the starter's `train_toy_ppo`
and `real_run` use them unchanged.

The checks confirm the shaped rewards on the worked example and on random
masks, with the task reward on each row's last token. They check each KL
estimator's value and bias exactly, $k_3 \ge 0$, and the exact expected gradient
of each estimator as a loss. They check GAE on the worked example and at
$\lambda = 0$ and $\lambda = 1$, masked whitening, every row of the clipping
table, and the clipped value loss. They check that one `ppo_update` step equals
one SGD step on the combined loss, and that gradient clipping caps it. Finally,
PPO from your functions must close 85 percent of the gap to $\beta \log Z$ on
the toy policy in 40 iterations.
