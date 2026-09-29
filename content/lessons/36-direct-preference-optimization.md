---
title: Direct preference optimisation
slug: 36-direct-preference-optimization
part: "Part 8 — Reinforcement learning for language models"
summary: Derive DPO from the KL-regularised optimum, build IPO, KTO, and SimPO, and watch DPO lower both responses of a pair.
minutes: 65
gpu: false
objectives:
  - Derive the DPO loss from the KL-regularised optimum, and say why the partition function cancels.
  - Implement the DPO loss and its gradient, and confirm the gradient with finite differences.
  - Show on an enumerable toy that DPO's implicit reward equals the Bradley-Terry reward up to a per-prompt constant.
  - Implement IPO, KTO, and SimPO as their papers write them, and say what each one changes.
  - "Reproduce likelihood displacement: the DPO loss falls while both the chosen and the rejected log-probabilities fall too."
prereqs:
  - 33-language-models-as-policies
  - 34-reward-models
  - 35-rlhf-with-ppo
exercise: 36-direct-preference-optimization
---

# Direct preference optimisation

RLHF with PPO has three stages. You fit a reward model to human comparisons.
You sample responses from the policy, score them, and run reinforcement
learning against the score. You keep a reference model, a value head, and the
reward model in memory the whole time. It works, and it is slow, fiddly, and
expensive.

Direct preference optimisation, or DPO, from Rafailov and colleagues (2023),
removes the middle stage. It rests on one observation. For the objective RLHF
optimises, the best policy and the reward determine each other exactly. So you
can write the reward in terms of the policy, plug that into the reward model's
own loss, and train the policy on the comparisons directly. What comes out is
a classification loss on pairs of responses. It needs no sampling, no reward
model, and no critic.

This lesson derives DPO step by step and checks every step in code. Then it
builds three relatives that change one piece each: IPO, KTO, and SimPO.
Finally, it reproduces the failure all of them share. The loss can go down
while the probability of the answer people preferred goes down too.

## What you already know

- A language model is a *policy* $\pi(y \mid x)$ over responses $y$ to a prompt
  $x$. Its log-probability of a whole response is the sum of its per-token
  log-probabilities, from *Language models as policies*.
- The KL-regularised objective rewards a policy for high reward and charges it
  $\beta$ times its KL divergence from a reference policy $\pi_{\text{ref}}$.
  Its optimum has a closed form.
- A Bradley-Terry reward model says the chance that response $y_w$ beats $y_l$
  is $\sigma(r(x, y_w) - r(x, y_l))$, where $\sigma(z) = 1 / (1 + e^{-z})$ is
  the sigmoid. It sees only reward differences, from *Reward models*.
- RLHF with PPO optimises that objective by sampling, from *RLHF with PPO*.
- Gradient descent on a softmax's logits moves every probability, because they
  must sum to one.

## Derive DPO from the KL-regularised optimum

The derivation has four steps of one line each. The data is a set of *preference pairs* $(x, y_w, y_l)$: a prompt, the response a person
preferred (the *chosen* response $y_w$), and the one they passed over (the
*rejected* response $y_l$).

### Step 1: write down the best policy

RLHF maximises, for each prompt,

$$
J(\pi) = \mathbb{E}_{y \sim \pi(\cdot \mid x)}\big[r(x, y)\big] - \beta\, \mathrm{KL}\big(\pi(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\big)
$$

Read it as: the average reward of the policy's responses, minus $\beta$ times
how far the policy has drifted from the reference. $\beta > 0$ is the length of
the leash.

Define the *Gibbs policy* $\pi^\star$ and its normaliser $Z(x)$, also called
the *partition function*:

$$
\pi^\star(y \mid x) = \frac{\pi_{\text{ref}}(y \mid x)\, e^{r(x, y) / \beta}}{Z(x)},
\qquad
Z(x) = \sum_{y} \pi_{\text{ref}}(y \mid x)\, e^{r(x, y) / \beta}
$$

Read it as: take the reference's probability of each response, boost it by
$e^{r/\beta}$, and divide by the total so the result sums to one.

Expand $\mathrm{KL}(\pi \,\|\, \pi^\star) = \mathbb{E}_\pi[\log \pi - \log \pi^\star]$
with $\log \pi^\star = \log \pi_{\text{ref}} + r / \beta - \log Z$, and multiply
by $\beta$. Every term rearranges into one identity:

$$
J(\pi) = \beta \log Z(x) - \beta\, \mathrm{KL}\big(\pi \,\|\, \pi^\star\big)
$$

The first term doesn't depend on $\pi$. The second is zero only when
$\pi = \pi^\star$ and positive otherwise. So $\pi^\star$ is the unique best
policy, and the best value of $J$ is $\beta \log Z(x)$.

### Step 2: solve the optimum for the reward

Take the log of $\pi^\star$ and solve for $r$:

$$
r(x, y) = \beta \log \frac{\pi^\star(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

Read it as: the reward of a response is $\beta$ times how much more likely the
best policy makes it than the reference, plus a number that depends only on
the prompt. You can't compute $Z(x)$ for a language model, because it sums
over every possible response.

### Step 3: plug it into Bradley-Terry

Bradley-Terry needs only the reward gap between two responses to the same
prompt. The $\beta \log Z(x)$ term appears in both rewards, so it cancels:

$$
p(y_w \succ y_l \mid x) = \sigma\Big(\beta \log \frac{\pi^\star(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^\star(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\Big)
$$

The unknown normaliser is gone. The preference probability is now a function
of log-probabilities you can compute.

### Step 4: fit the policy by maximum likelihood

Replace $\pi^\star$ with the policy you train, $\pi_\theta$, and minimise the
negative log-likelihood of the observed preferences. Write the *log-ratio
margin* of a pair as

$$
h_\theta = \Big(\log \pi_\theta(y_w \mid x) - \log \pi_{\text{ref}}(y_w \mid x)\Big) - \Big(\log \pi_\theta(y_l \mid x) - \log \pi_{\text{ref}}(y_l \mid x)\Big)
$$

Read it as: how much the policy has raised the chosen response relative to the
reference, minus the same for the rejected one. The DPO loss is

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\Big[\log \sigma\big(\beta\, h_\theta\big)\Big]
$$

Read it as: the reward-model loss from *Reward models*, applied to the
*implicit reward* $\hat r_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$.
In numpy, with sequence log-probabilities of shape `(B,)`:

```python
h = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
loss = np.mean(-log_sigmoid(beta * h))     # log_sigmoid is -np.logaddexp(0, -z)
```

> [!KEY]
> DPO is the Bradley-Terry loss applied to the implicit reward
> $\beta \log (\pi_\theta / \pi_{\text{ref}})$. The partition function cancels
> because Bradley-Terry compares two responses to the same prompt.

### The gradient

The derivative of $-\log \sigma(u)$ is $-\sigma(-u)$, and $u = \beta h$ moves by
$\beta$ for each unit of $\log \pi_\theta(y_w)$. So for a batch of $B$ pairs,

$$
\frac{\partial \mathcal{L}}{\partial \log \pi_\theta(y_w)} = -\frac{\beta}{B}\, \sigma\big(-\beta h\big),
\qquad
\frac{\partial \mathcal{L}}{\partial \log \pi_\theta(y_l)} = +\frac{\beta}{B}\, \sigma\big(-\beta h\big)
$$

Read it as: raise the chosen log-probability and lower the rejected one, with
weight $\sigma(-\beta h)$. The weight is near one for a pair the implicit
reward ranks the wrong way round, and near zero for a pair it already ranks
right by a wide margin. The reference log-probabilities are constants. In
numpy, `g = -beta * sigmoid(-beta * h) / B`, and the rejected gradient is `-g`.

### Worked example

One pair, $\beta = 0.1$. The policy gives the chosen response log-probability
$-10$ and the rejected one $-12$. The reference gives them $-11$ and $-11.5$.

- Margin: $h = (-10 + 11) - (-12 + 11.5) = 1 + 0.5 = 1.5$.
- Implicit rewards: $\hat r_w = 0.1$ and $\hat r_l = -0.05$, so $\beta h = 0.15$.
- Loss: $-\log \sigma(0.15) = \log(1 + e^{-0.15}) = 0.6210$.
- Weight: $\sigma(-0.15) = 0.4626$. At the start of training $h = 0$ and the
  weight is $0.5$.
- Gradient for the chosen log-probability: $-0.1 \times 0.4626 = -0.04626$.
  Gradient descent moves it up.

## Check the derivation on a table

The claim to test is this: with enough data covering every response, DPO's
implicit reward equals the true Bradley-Terry reward up to a constant for each
prompt. The derivation predicts the constant too. At the optimum
$\pi_\theta = \pi^\star$, so Step 2 gives
$\hat r_\theta(x, y) = r(x, y) - \beta \log Z(x)$.

To test it without sampling noise, use the *expected* DPO loss. Take a few
prompts, each with a handful of responses, and a policy that is one free
logit per response (a *tabular* policy). For every ordered pair $(i, j)$, the
chance that the data shows $i$ as chosen and $j$ as rejected is
$\pi_{\text{ref}}(i)\, \pi_{\text{ref}}(j)\, \sigma(r_i - r_j)$: both responses
were sampled from the reference, and Bradley-Terry picked $i$. Weight each
pair's DPO loss by that number, normalised to sum to one per prompt, and run
gradient descent on the logits.

Your `dpo_grad` gives gradients with respect to log-probabilities. The logits
need one more chain-rule step through the log-softmax. Because
$\partial \log \pi_k / \partial z_m = [k = m] - \pi_m$,

```python
g_logits = g_logp - np.exp(logp) * g_logp.sum()
```

The check fits two prompts with nine responses each, then confirms three
things to within $10^{-5}$:

- $\hat r - r$ is constant across the responses of each prompt.
- That constant is $-\beta \log Z(x)$, and differs between the two prompts.
- The fitted policy equals $\pi_{\text{ref}}\, e^{r/\beta} / Z$.

> [!WARNING]
> The recovery holds only for responses the data covers, and only up to the
> per-prompt constant. Comparing implicit rewards across prompts is
> meaningless, and a response that never appears in a pair is pushed around
> only by the softmax's normalisation.

## Watch both log-probabilities fall

DPO's loss sees only the margin $h$. It's satisfied when the chosen response
rises *relative to* the rejected one. Nothing stops both from falling, with the
rejected one falling faster. Razin and colleagues (2025) call this *likelihood
displacement*. The mass that leaves the pair goes somewhere else, sometimes to
responses nobody wanted.

It happens because responses share parameters. Take a softmax policy over 32
responses whose logits are $z_y = \theta^\top e_y$, with a unit-length
embedding $e_y \in \mathbb{R}^4$ for each response. The gradient of
$\log \pi(y)$ with respect to $\theta$ is $e_y - \bar e$, where
$\bar e = \sum_{y'} \pi(y') e_{y'}$. The two DPO gradients have equal size and
opposite signs, so the $\bar e$ terms cancel, and each step moves $\theta$
along $e_w - e_l$.

Now make the chosen and rejected embeddings nearly parallel, with cosine
similarity $0.95$. Then each response's logit moves in proportion to
$e_y^\top (e_w - e_l)$:

- The chosen response: $1 - 0.95 = 0.05$.
- The rejected response: $0.95 - 1 = -0.05$.
- The response most aligned with $e_w - e_l$ in the seeded toy: $0.258$.

The third response gains five times faster than the chosen one. It takes over
the normaliser, and the chosen log-probability falls. The exercise runs 500
steps with $\beta = 0.1$ and step size 1 from a uniform start, where every log-probability is
$\log(1/32) = -3.466$. The following table shows the run:

| Step | $\log \pi(y_w)$ | $\log \pi(y_l)$ | DPO loss |
|---|---|---|---|
| 0 | $-3.466$ | $-3.466$ | $0.693$ |
| 100 | $-3.356$ | $-3.850$ | $0.669$ |
| 500 | $-5.688$ | $-8.039$ | $0.583$ |

The chosen log-probability rises for about 70 steps, then falls for good. By
step 500 the aligned third response holds $45\%$ of the mass, up from about
$3\%$. With orthogonal embeddings the effect is much weaker, because
$e_w^\top(e_w - e_l) = 1$.

> [!KEY]
> A falling DPO loss says the margin grew. It doesn't say the chosen response
> became more likely. Track $\log \pi(y_w)$ and $\log \pi(y_l)$ separately.

## Change one piece: IPO, KTO, and SimPO

Each relative keeps DPO's shape and changes one thing.

### IPO: a finite target for the margin

When a preference is deterministic, so that $y_w$ always wins, Bradley-Terry
needs an infinite reward gap to fit it. DPO then pushes $h$ upward forever,
whatever $\beta$ is, and the KL leash stops working. Azar and colleagues (2023)
derive a general family of objectives and pick the identity mapping. The
resulting *identity preference optimisation* loss, IPO, is a squared error
toward a finite margin. The paper's sampled loss is

$$
\mathcal{L}_{\text{IPO}} = \mathbb{E}_{(x, y_w, y_l)}\Big[\Big(h_\theta - \frac{1}{2\tau}\Big)^2\Big]
$$

Read it as: push the log-ratio margin toward $1/(2\tau)$, and push back if it
overshoots. $\tau$ plays the role of $\beta$. In numpy,
`np.mean((h - 1 / (2 * tau)) ** 2)`, and the gradient for the chosen
log-probability is `2 * (h - 1 / (2 * tau)) / B`. With the DPO example's
$h = 1.5$ and $\tau = 0.1$, the target is $5$ and the loss is
$(1.5 - 5)^2 = 12.25$.

The exercise trains both losses on one deterministic pair. IPO's margin settles
at $1/(2\tau)$ to six decimal places. DPO's margin is 7.7 after 1000 steps and
9.1 after 2000, and it keeps climbing.

### KTO: thumbs up or thumbs down

Pairs are expensive to collect, and a thumbs up or down on one response is
cheap. KTO, from Ethayarajh and colleagues (2024), learns from such unpaired
labels. Borrowing from prospect theory, it judges each response as a gain or a
loss relative to a *reference point* $z_0$. With $r_\theta = \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$,

$$
v(x, y) = \begin{cases}
\lambda_D\, \sigma\big(\beta (r_\theta - z_0)\big) & \text{if } y \text{ is desirable} \\
\lambda_U\, \sigma\big(\beta (z_0 - r_\theta)\big) & \text{if } y \text{ is undesirable}
\end{cases}
\qquad
\mathcal{L}_{\text{KTO}} = \mathbb{E}_{x, y}\big[\lambda_y - v(x, y)\big]
$$

Read it as: a desirable response scores well when its log-ratio is above the
reference point, and an undesirable one when its log-ratio is below it.
$\lambda_y$ is $\lambda_D$ or $\lambda_U$, whichever matches the label. The
paper defines $z_0$ as the KL divergence between the policy and the reference.
In practice it estimates $z_0$ from mismatched pairs in a microbatch, pairing
prompt $x_i$ with another example's response $y_j$. It clamps the estimate at
zero and passes no gradient through it:

$$
\hat z_0 = \max\Big(0,\ \frac{1}{m}\sum_{i=1}^{m} \log \frac{\pi_\theta(y_j \mid x_i)}{\pi_{\text{ref}}(y_j \mid x_i)}\Big)
$$

The paper recommends $\beta = 0.1$ and weights with
$\lambda_D n_D / (\lambda_U n_U) \in [1, 3/2]$, where $n_D$ and $n_U$ count
desirable and undesirable examples. For example, with $\beta = 0.1$,
$z_0 = 0.2$, and both weights 1, a desirable response with $r_\theta = 1$ costs
$1 - \sigma(0.08) = 0.4800$. An undesirable one with $r_\theta = -0.5$ costs
$1 - \sigma(0.07) = 0.4825$.

### SimPO: no reference, and average log-probability

SimPO, from Meng, Xia, and Chen (2024), drops the reference model. It argues
that the quantity generation cares about is the *average* log-probability per
token, and uses that as the reward. It also adds a target margin $\gamma$:

$$
\mathcal{L}_{\text{SimPO}} = -\mathbb{E}\Big[\log \sigma\Big(\frac{\beta}{\lvert y_w \rvert} \log \pi_\theta(y_w \mid x) - \frac{\beta}{\lvert y_l \rvert} \log \pi_\theta(y_l \mid x) - \gamma\Big)\Big]
$$

Read it as: the chosen response's average token log-probability, times
$\beta$, must beat the rejected one's by at least $\gamma$. $\lvert y \rvert$
is the number of response tokens, `mask.sum(axis=-1)`. The authors report good
results with $\beta$ between 2 and 2.5 and $\gamma$ between 0.5 and 1.5. For
example, a chosen response of 10 tokens with total log-probability $-10$
averages $-1$. A rejected one of 8 tokens with total $-12$ averages $-1.5$.
With $\beta = 2$ and $\gamma = 1$, the argument is $-2 + 3 - 1 = 0$, so the
loss is $\log 2 = 0.693$.

Length normalisation makes SimPO blind to repetition. Repeat every token of a
response and its average is unchanged, so the SimPO loss is unchanged. The
summed log-probability doubles, so the DPO loss moves.

> [!TIP]
> SimPO without a reference isn't DPO with averaged log-probs. Averaging the
> log-probs inside DPO changes DPO's objective, and the derivation in this
> lesson no longer applies.

> [!RECAP]
> - DPO follows from the KL-regularised optimum: invert it for the reward,
>   plug that into Bradley-Terry, and $\beta \log Z(x)$ cancels.
> - The DPO gradient for the chosen log-prob is $-\beta \sigma(-\beta h) / B$.
>   The rejected one gets the opposite sign.
> - At the optimum, the implicit reward is $r - \beta \log Z(x)$: right up to
>   one constant per prompt.
> - IPO regresses $h$ onto $1/(2\tau)$. KTO scores single responses against
>   $z_0$. SimPO uses average log-probs and a margin $\gamma$, with no
>   reference.
> - The margin can grow while both log-probabilities fall.

## Build it step by step

The checks test the functions in this order.

1. **`sequence_logprob(token_logps, mask)`.** Takes `(B, T)` arrays and
   returns `(B,)`. Multiply by the mask and sum over the last axis. Common
   mistake: counting padding tokens, whose log-probabilities are garbage.
2. **`dpo_loss(...)`.** Takes four `(B,)` arrays, `beta`, and optional
   `weights`. Build `h`, then take the weighted mean of `-log_sigmoid(beta * h)`
   with `pair_weights`. Common mistake: `-np.log(sigmoid(z))`, which returns
   `inf` for large negative `z`.
3. **`dpo_grad(...)`.** Returns `(grad_chosen, grad_rejected)`. Multiply
   `-beta * sigmoid(-beta * h)` by the weights. Common mistake: dropping the
   factor $\beta$ from the chain rule. The finite-difference check catches it.
4. **`ipo_loss` and `ipo_grad`.** The target is `1 / (2 * tau)`. Common
   mistake: using `1 / tau`.
5. **The margin check.** It uses only your two gradient functions.
6. **`kto_reference_point` and `kto_loss`.** The reference point is
   `max(0, mean(policy - ref))`. The loss uses `np.where` on the `desirable`
   mask for both $v$ and $\lambda_y$. Common mistake: flipping the sign inside
   the undesirable branch's sigmoid.
7. **`simpo_loss(...)`.** Divide each masked sum by its own
   `mask.sum(axis=-1)`, then apply `gamma` outside the $\beta$ terms.
8. **`fit_dpo_exact(ref_logprobs, reward, beta, steps, lr)`.** Build all
   ordered pairs with `np.meshgrid` and drop the diagonal. Compute the pair
   weights per prompt, call `dpo_grad` with them, then scatter the gradients
   into response slots with `np.bincount(i, g_w, N)` and chain through the
   log-softmax. Common mistake: forgetting that pairs $(i, j)$ and $(j, i)$
   both appear, with weights $\sigma(r_i - r_j)$ and $\sigma(r_j - r_i)$.
9. **`train_dpo_linear(E, theta0, chosen, rejected, beta, lr, steps)`.** Keep
   the reference log-probs fixed at the start. Each step, call `dpo_grad` on
   one pair and form the $\theta$ gradient with
   `g_w * (E[w] - e_bar) + g_l * (E[l] - e_bar)`. Return all `steps + 1` rows.
   Common mistake: updating the reference as you go, which makes the margin
   zero every step.

## Cost and scaling

DPO costs about the same as supervised fine-tuning on twice as many sequences.
The following table compares the methods by what each training pair or example
costs:

| Method | Models in memory | Forward passes per example | Samples during training |
|---|---|---|---|
| PPO RLHF | Policy, reference, reward model, value head | Generation, then scoring | Yes, every step |
| DPO, IPO | Policy and reference | 2 policy and 2 reference | No |
| KTO | Policy and reference | 1 of each, plus the mismatched pairs for $z_0$ | No |
| SimPO | Policy only | 2 policy | No |

You can precompute the reference log-probabilities once, before training, so
DPO holds one model in memory. The exercise's losses cost time linear in the
batch size. The exact fit enumerates $N(N-1)$ pairs per prompt, which is why
it stays on tiny tables.

## Where this breaks

**Likelihood displacement.** The chosen response's probability falls, and the
lost mass goes to responses outside the data. Razin and colleagues link it to
chosen and rejected responses with similar hidden representations, and show
that filtering such pairs helps. People also add a supervised loss on the
chosen responses to hold them up.

**Offline data.** The pairs usually come from other models, so DPO learns
about responses its own policy rarely produces. Online and iterative DPO
sample fresh pairs from the current policy and label them with a reward model
or a judge.

**Deterministic preferences.** DPO drives the margin up without bound when a
pair always goes one way, as the IPO check shows. IPO's finite target, label
smoothing, and early stopping all limit it.

**Length exploitation.** Preferred responses are often longer. DPO's summed
log-ratio lets length act as a shortcut, and trained models grow verbose.
SimPO normalises by length, and others add an explicit length term.

## Further reading

The following sources cover the derivation and the relatives in more depth:

- [Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023)](https://arxiv.org/abs/2305.18290): the derivation this lesson follows.
- [Azar et al., "A General Theoretical Paradigm to Understand Learning from Human Preferences" (2023)](https://arxiv.org/abs/2310.12036): the $\Psi$-preference family and IPO.
- [Ethayarajh et al., "KTO: Model Alignment as Prospect Theoretic Optimization" (2024)](https://arxiv.org/abs/2402.01306): learning from unpaired binary labels.
- [Meng, Xia, and Chen, "SimPO: Simple Preference Optimization with a Reference-Free Reward" (2024)](https://arxiv.org/abs/2405.14734): the length-normalised, reference-free loss.
- [Razin et al., "Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization" (2025)](https://arxiv.org/abs/2410.08847): why both log-probabilities fall, and a filter for the pairs that cause it.
- [Rafailov et al., "From r to Q*: Your Language Model is Secretly a Q-Function" (2024)](https://arxiv.org/abs/2404.12358): the token-level reading of DPO's implicit reward.
- [Guo et al., "Direct Language Model Alignment from Online AI Feedback" (2024)](https://arxiv.org/abs/2402.04792): online DPO with a judge labelling fresh samples.

Open questions that researchers are working on:

- Which properties of a preference dataset predict displacement before training starts?
- When does offline DPO match online RLHF, given equal labels and compute?
- Can one loss keep DPO's simplicity and IPO's bounded margin without a tuned target?

## The exercise

You build the losses and the two experiments in this lesson. Write
`sequence_logprob`, then `dpo_loss` and `dpo_grad`, then `ipo_loss` and
`ipo_grad`, `kto_reference_point` and `kto_loss`, and `simpo_loss`. Then write
`fit_dpo_exact`, which minimises the expected DPO loss over every pair of a
tabular policy, and `train_dpo_linear`, which runs DPO on one pair of a
softmax policy with shared embeddings.

The checks verify each loss against the worked examples in this lesson, and
both gradients against finite differences. They train IPO and DPO on a
deterministic pair and confirm that only IPO's margin settles. They confirm
that `fit_dpo_exact` recovers the hidden reward up to $-\beta \log Z(x)$ for
each prompt. Last, they confirm that `train_dpo_linear` lowers the loss while
both log-probabilities fall, that the mass goes to the response aligned with
$e_w - e_l$, and that dissimilar pairs lose less.
