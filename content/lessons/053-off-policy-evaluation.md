---
title: Evaluating a policy from logs
slug: 053-off-policy-evaluation
part: "Part 7 — Models, exploration, and fixed data"
summary: Estimate a new policy's value from logged data with IPS, per-decision IS, doubly robust estimates, and FQE, and know when the data cannot tell you.
minutes: 45
gpu: false
objectives:
  - Compute the IPS, self-normalized IPS, and doubly robust estimates from logged bandit data, and say which are unbiased.
  - Explain why the doubly robust correction stays unbiased for any model and cuts variance when the model is close.
  - Compute the per-decision importance-sampling and recursive doubly robust estimates for trajectories.
  - Fit action values from logged transitions with fitted Q evaluation (FQE).
  - Use the effective sample size to judge an estimate before you trust it, and say when evaluation from logs is hopeless.
prereqs:
  - 007-contextual-bandits
  - 015-off-policy-importance-sampling
  - 051-offline-rl
exercise: 053-off-policy-evaluation
---

# Evaluating a policy from logs

You trained a new policy offline, and before it touches a patient, a road, or a user, someone asks how good it is. You cannot run it to find out, because that is what you are trying to avoid. All you have is the log that an older policy wrote.

This is *off-policy evaluation* (OPE): estimating a target policy's value from data that a different policy collected. Importance sampling already reweights returns for this, but one weight per whole trajectory is blunt, and its variance grows fast with the length of the trajectory. This chapter builds the estimators that practitioners use and ends with the checks that tell you when no estimator can save you.

## What you already know

- Reweighting by $\pi / b$ turns data from the behavior policy $b$ into data about the target policy $\pi$, and needs coverage: $b$ must take every action that $\pi$ might take. Ordinary importance sampling is unbiased with high variance, and the weighted form is biased but stays in range (see [Learning about one policy from another](/lessons/015-off-policy-importance-sampling)).
- A contextual bandit logs a context, an action, and a reward for each round (see [Contextual bandits and LinUCB](/lessons/007-contextual-bandits)).
- Offline RL fits Q values to a fixed log of transitions and fails where the log has no coverage (see [Offline reinforcement learning](/lessons/051-offline-rl)).

## Start with one-step logs

Begin with a contextual bandit, where the algebra is smallest. A log holds $N$ samples $(x_i, a_i, r_i)$ and the *propensity* $b(a_i \mid x_i)$, the probability that the logger chose $a_i$ in context $x_i$. You want $V^\pi = \mathbb{E}_{x}\,\mathbb{E}_{a \sim \pi}[\,r\,]$ for a new policy $\pi$.

A sample whose action $\pi$ likes more than $b$ did should count for more, so give sample $i$ the weight $w_i = \pi(a_i \mid x_i) / b(a_i \mid x_i)$. Averaging $w_i r_i$ gives the *inverse propensity score* (IPS) estimator. Dividing by $\sum_i w_i$ instead of $N$ gives its self-normalized form (SNIPS). They are the ordinary and weighted estimators of the previous chapter, one step long.

The *doubly robust* (DR) estimator adds a model $\hat q(x, a)$, a guess at the expected reward. Let $\hat V(x) = \sum_a \pi(a \mid x)\,\hat q(x, a)$ be the model's value of $\pi$ in context $x$. DR starts from the model and corrects it with the data:

$$
\hat V_{\text{IPS}} = \frac{1}{N}\sum_i w_i r_i,
\qquad
\hat V_{\text{SNIPS}} = \frac{\sum_i w_i r_i}{\sum_i w_i},
\qquad
\hat V_{\text{DR}} = \frac{1}{N}\sum_i \Big[\hat V(x_i) + w_i\,\big(r_i - \hat q(x_i, a_i)\big)\Big]
$$

Read DR aloud: for each sample, take the model's answer for $\pi$ in that context, then add the importance-weighted *residual*, the reward you saw minus what the model predicted for that action. In code, with `w = pi[x, a] / b[x, a]` and `v_hat = (pi * q_hat).sum(axis=1)`:

```python
ips_value = np.mean(w * r)
snips_value = np.sum(w * r) / np.sum(w)
dr_value = np.mean(v_hat[x] + w * (r - q_hat[x, a]))
```

### Work it by hand: a four-sample log

Take two contexts and two actions. The target is $\pi = [[0.8, 0.2], [0.1, 0.9]]$ and the logger played uniformly, $b = 0.5$ everywhere. The model is $\hat q = [[0.6, 0.1], [0.3, 0.4]]$. The log is:

| $x$ | $a$ | $r$ | $w$ | $\hat q(x, a)$ | $\hat V(x)$ | DR term |
|---|---|---|---|---|---|---|
| 0 | 0 | 1.0 | 1.6 | 0.6 | 0.50 | $0.50 + 1.6 \times 0.4 = 1.14$ |
| 0 | 0 | 0.8 | 1.6 | 0.6 | 0.50 | $0.50 + 1.6 \times 0.2 = 0.82$ |
| 1 | 1 | 0.5 | 1.8 | 0.4 | 0.39 | $0.39 + 1.8 \times 0.1 = 0.57$ |
| 1 | 0 | 0.2 | 0.2 | 0.3 | 0.39 | $0.39 + 0.2 \times (-0.1) = 0.37$ |

The weighted rewards sum to $1.6 + 1.28 + 0.9 + 0.04 = 3.82$, and the weights sum to $5.2$. So IPS is $3.82 / 4 = 0.955$, SNIPS is $3.82 / 5.2 = 0.735$, and DR is $(1.14 + 0.82 + 0.57 + 0.37) / 4 = 0.725$. Three estimators give three answers from the same four rows.

Why is DR right on average, whatever $\hat q$ is? Over the logger's action choice, $\mathbb{E}_{a \sim b}[\,w\,(r - \hat q)\,] = \mathbb{E}_{a \sim \pi}[\,r - \hat q\,]$, which is the true value of $\pi$ minus the model's value $\hat V(x)$. Adding $\hat V(x)$ back cancels the model's error. This needs $\hat q$ fixed before you look at the log, or fitted on a separate split. Why does it help? IPS multiplies the whole reward by a weight that can be large. DR multiplies only the residual, which is small when $\hat q$ is close.

```figure
<svg viewBox="0 0 570 250" role="img" aria-label="Mean and one standard deviation of IPS, SNIPS, and doubly robust over 4000 logs of 20 samples, against the true value">
<line x1="90.0" y1="196" x2="90.0" y2="202" class="stroke-muted"/>
<text x="90.0" y="218" class="fig-small" text-anchor="middle">0.0</text>
<line x1="154.3" y1="196" x2="154.3" y2="202" class="stroke-muted"/>
<text x="154.3" y="218" class="fig-small" text-anchor="middle">0.1</text>
<line x1="218.6" y1="196" x2="218.6" y2="202" class="stroke-muted"/>
<text x="218.6" y="218" class="fig-small" text-anchor="middle">0.2</text>
<line x1="282.9" y1="196" x2="282.9" y2="202" class="stroke-muted"/>
<text x="282.9" y="218" class="fig-small" text-anchor="middle">0.3</text>
<line x1="347.1" y1="196" x2="347.1" y2="202" class="stroke-muted"/>
<text x="347.1" y="218" class="fig-small" text-anchor="middle">0.4</text>
<line x1="411.4" y1="196" x2="411.4" y2="202" class="stroke-muted"/>
<text x="411.4" y="218" class="fig-small" text-anchor="middle">0.5</text>
<line x1="475.7" y1="196" x2="475.7" y2="202" class="stroke-muted"/>
<text x="475.7" y="218" class="fig-small" text-anchor="middle">0.6</text>
<line x1="540.0" y1="196" x2="540.0" y2="202" class="stroke-muted"/>
<text x="540.0" y="218" class="fig-small" text-anchor="middle">0.7</text>
<line x1="90" y1="196" x2="540" y2="196" class="stroke-muted"/>
<line x1="278.4" y1="30" x2="278.4" y2="196" class="stroke-c dashed"/>
<text x="284.4" y="24" class="fig-small">true value 0.293</text>
<text x="14" y="74" class="fig-bold">IPS</text>
<line x1="205.0" y1="70" x2="348.8" y2="70" class="stroke-b" stroke-width="3"/>
<circle cx="276.9" cy="70" r="6" class="fill-b"/>
<text x="356.8" y="74" class="fig-small">mean 0.291, std 0.112</text>
<text x="14" y="129" class="fig-bold">SNIPS</text>
<line x1="208.3" y1="125" x2="369.6" y2="125" class="stroke-d" stroke-width="3"/>
<circle cx="288.9" cy="125" r="6" class="fill-d"/>
<text x="377.6" y="129" class="fig-small">mean 0.309, std 0.125</text>
<text x="14" y="184" class="fig-bold">DR</text>
<line x1="239.1" y1="180" x2="317.1" y2="180" class="stroke-a" stroke-width="3"/>
<circle cx="278.1" cy="180" r="6" class="fill-a"/>
<text x="325.1" y="184" class="fig-small">mean 0.293, std 0.061</text>
<text x="315" y="240" class="fig-small" text-anchor="middle">estimate of the target policy value</text>
</svg>
Figure 1. IPS, SNIPS, and DR over 4,000 logs of 20 samples, with a model $\hat q$ that is off by about 0.05. Notice that IPS and DR center on the true value 0.293, while SNIPS sits above it at 0.310, and that DR's spread, 0.061, is about half of IPS's, 0.112.
```

SNIPS is biased, and here its spread (0.126) is no smaller than IPS's, because its guarantee is staying in the range of the rewards, not lower spread in every problem. DR keeps IPS's unbiasedness and halves the spread, because the model absorbs most of what the weights carried.

> [!KEY]
> IPS and DR are unbiased when the logger's propensities are right. DR has lower variance when the model is good, and a bad model costs variance, not bias. That is where the name comes from.

## Trust an estimate by its effective sample size

A weighted average of $N$ samples is not worth $N$ samples when a few weights are huge. The *effective sample size* (ESS) counts the equally weighted samples that would carry the same information:

$$
\operatorname{ESS}(w) = \frac{\big(\sum_i w_i\big)^2}{\sum_i w_i^2}
$$

Read it aloud: the square of the total weight over the total of squared weights. Equal weights give $N$. One nonzero weight gives 1. Weights $(3, 1)$ give $16 / 10 = 1.6$. On the four-sample log above, the weights $(1.6, 1.6, 1.8, 0.2)$ give $5.2^2 / 8.4 = 3.22$, so four samples are worth about three. When the ESS is a tiny fraction of $N$, every importance-weighted estimate rests on a handful of samples, whatever the estimator.

## Weight each reward only by the steps before it

With trajectories, a single whole-episode weight multiplies one ratio per step, so its variance grows with the horizon. *Per-decision importance sampling* (Precup, Sutton, and Singh, 2000) cuts this. A reward $r_t$ cannot depend on actions taken after step $t$, so it needs only the ratios up to step $t$. Let $\rho_{0:t} = \prod_{k=0}^{t} \pi(a_k \mid s_k) / b(a_k \mid s_k)$, which is the *prefix* product through step $t$ (the previous chapter used the tail product to the end). Then

$$
\hat V_{\text{PDIS}} = \frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{H-1} \gamma^t\,\rho^{(i)}_{0:t}\, r^{(i)}_t
$$

Read it aloud: sum the discounted rewards, each weighted by the ratios up to and including the step that earned it. Take one two-step trajectory with step ratios $1.6$ and $1.8$, rewards $1$ and $2$, and $\gamma = 0.9$. Per-decision weighting gives $1 \times 1.6 + 0.9 \times 2 \times (1.6 \times 1.8) = 6.784$. Whole-trajectory weighting multiplies both rewards by $2.88$ and gives $8.064$. Both have the right mean, but the first multiplies fewer ratios into the early reward.

```python
ratio = pi[states, actions] / b[states, actions]       # (N, H)
rho = np.cumprod(ratio, axis=1)                         # rho[:, t] is rho_{0:t}
est = (rho * rewards * gamma ** np.arange(H)).sum(axis=1)
```

### Add a model with a backward recursion

The DR idea extends to trajectories with a model $Q_t(s, a)$ of the expected discounted reward from step $t$ to the end. Let $\bar V_t(s) = \sum_a \pi(a \mid s)\,Q_t(s, a)$ and let $\rho_t = \pi(a_t \mid s_t)/b(a_t \mid s_t)$ be a *single* step's ratio. Jiang and Li (2016) define the estimate by a recursion that runs backward from the last step:

$$
\hat V_{H} = 0,
\qquad
\hat V_t = \bar V_t(s_t) + \rho_t\,\big(r_t + \gamma \hat V_{t+1} - Q_t(s_t, a_t)\big),
\qquad
\hat V_{\text{DR}} = \frac{1}{N}\sum_i \hat V_0^{(i)}
$$

Read it aloud: at each step, start from the model's value of the state under $\pi$, then add the one-step residual (reward plus discounted future estimate, minus the model's value of the action taken), weighted by this step's ratio. Unrolled, step $t$'s residual carries the prefix product $\rho_{0:t}$, as in per-decision IS. Take the same trajectory with $Q_1 = [[0.4, 0.1], [0.5, 1.8]]$ and $Q_0 = [[0.5, 0.2], [0.3, 1.0]]$. At $t = 1$: $\bar V_1(1) = 0.1 \times 0.5 + 0.9 \times 1.8 = 1.67$, so $\hat V_1 = 1.67 + 1.8 \times (2 - 1.8) = 2.03$. At $t = 0$: $\bar V_0(0) = 0.8 \times 0.5 + 0.2 \times 0.2 = 0.44$, so $\hat V_0 = 0.44 + 1.6 \times (1 + 0.9 \times 2.03 - 0.5) = 4.1632$.

```python
est = np.zeros(N)
for t in reversed(range(H)):
    s, a = states[:, t], actions[:, t]
    est = v_bar[t, s] + ratio[:, t] * (rewards[:, t] + gamma * est - q[t, s, a])
```

On a random MDP with 4 states, 2 actions, and 5-step trajectories, the exact value is $-0.393$. Over 20,000 logged trajectories, whole-trajectory IS has a per-trajectory standard deviation of 3.77, per-decision IS has 3.17, and DR with the exact $Q_t$ has 2.21. DR's spread is not zero, because the next state is still random. It would be zero if rewards and transitions were both deterministic.

## Learn the model from the log: FQE

Where does $Q$ come from? You fit it from the same log. *Fitted Q evaluation* (FQE; Le, Voloshin, and Yue, 2019) is Q-iteration for the *target* policy, with no max. Given logged transitions $(s, a, r, s')$, repeat

$$
Q(s, a) \leftarrow \operatorname{mean}_{(s, a, r, s') \in \mathcal{D}}\Big[\, r + \gamma \sum_{a'} \pi(a' \mid s')\,Q(s', a') \Big]
$$

Read it aloud: for each pair the log covers, average the one-step targets of its transitions, and bootstrap from the *expected* next value under $\pi$, not from the best next action. Replacing the max of [Offline reinforcement learning](/lessons/051-offline-rl) with an expectation under $\pi$ gives an operator with a unique fixed point, the value of $\pi$ under the log's empirical model. In code:

```python
y = r + gamma * (pi[s2] * Q[s2]).sum(axis=1)           # (N,)
Q = mean_per_pair(y)                                    # pairs not in the log stay at 0
```

If the log holds every transition at the model's exact frequencies, the fixed point is the exact $Q^\pi$, and the exercise checks that against `policy_evaluation`. In practice the log is a sample, and a pair that $\pi$ visits but the log never took keeps its starting value, with no warning. The usual fix is to feed FQE's $\hat Q$ into a DR estimator, which uses the model to cut variance and the weights to correct its bias.

## When evaluation from logs is hopeless

The estimators are only as good as the log supports. Three failures matter most.

*Support violations.* If $\pi$ takes an action where $b(a \mid s) = 0$, no sample carries information about it. IPS silently counts it as worth zero, and DR leans on a model with no data there. Check coverage first.

*Long horizons.* Whole-trajectory weights multiply one ratio per step. Figure 2 shows the ESS of those weights on the random MDP: from 79 percent of the samples at $H = 1$ down to 3.6 percent at $H = 20$.

```figure
<svg viewBox="0 0 570 270" role="img" aria-label="Effective sample size as a fraction of 5000 trajectories against horizon: it falls from 1 at horizon 1 to near 0.02 at horizon 20">
<line x1="70" y1="210.0" x2="540" y2="210.0" class="stroke-grid"/>
<text x="62" y="214.0" class="fig-small" text-anchor="end">0</text>
<line x1="70" y1="165.0" x2="540" y2="165.0" class="stroke-grid"/>
<text x="62" y="169.0" class="fig-small" text-anchor="end">0.25</text>
<line x1="70" y1="120.0" x2="540" y2="120.0" class="stroke-grid"/>
<text x="62" y="124.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="70" y1="75.0" x2="540" y2="75.0" class="stroke-grid"/>
<text x="62" y="79.0" class="fig-small" text-anchor="end">0.75</text>
<line x1="70" y1="30.0" x2="540" y2="30.0" class="stroke-grid"/>
<text x="62" y="34.0" class="fig-small" text-anchor="end">1</text>
<line x1="70.0" y1="210" x2="70.0" y2="216" class="stroke-muted"/>
<text x="70.0" y="232" class="fig-small" text-anchor="middle">1</text>
<line x1="168.9" y1="210" x2="168.9" y2="216" class="stroke-muted"/>
<text x="168.9" y="232" class="fig-small" text-anchor="middle">5</text>
<line x1="292.6" y1="210" x2="292.6" y2="216" class="stroke-muted"/>
<text x="292.6" y="232" class="fig-small" text-anchor="middle">10</text>
<line x1="416.3" y1="210" x2="416.3" y2="216" class="stroke-muted"/>
<text x="416.3" y="232" class="fig-small" text-anchor="middle">15</text>
<line x1="540.0" y1="210" x2="540.0" y2="216" class="stroke-muted"/>
<text x="540.0" y="232" class="fig-small" text-anchor="middle">20</text>
<polyline points="70.0,67.4 94.7,95.5 119.5,117.8 144.2,136.2 168.9,149.7 193.7,161.8 218.4,171.9 243.2,178.3 267.9,182.9 292.6,189.0 317.4,190.7 342.1,195.5 366.8,200.2 391.6,204.2 416.3,201.5 441.1,203.8 465.8,202.5 490.5,203.5 515.3,203.8 540.0,203.5" class="stroke-b" fill="none" stroke-width="2.5"/>
<circle cx="70.0" cy="67.4" r="4" class="fill-b"/>
<text x="78.0" y="59.4" class="fig-small">0.792</text>
<circle cx="168.9" cy="149.7" r="4" class="fill-b"/>
<text x="176.9" y="141.7" class="fig-small">0.335</text>
<circle cx="292.6" cy="189.0" r="4" class="fill-b"/>
<text x="300.6" y="181.0" class="fig-small">0.117</text>
<circle cx="540.0" cy="203.5" r="4" class="fill-b"/>
<text x="548.0" y="195.5" class="fig-small">0.036</text>
<text x="305" y="262" class="fig-small" text-anchor="middle">horizon H (steps in each logged trajectory)</text>
<text x="14" y="22" class="fig-small">ESS / N</text>
</svg>
Figure 2. The effective sample size of whole-trajectory importance weights, as a fraction of 5,000 logged trajectories, against the horizon. Notice that it halves within three steps and keeps falling, so at $H = 20$ the log is worth about 180 samples.
```

*Unbiased is not accurate.* DR is unbiased for any fixed model, but that is an average over many logs. With few effective samples, one log's estimate can be far off, so report an interval, not just a number.

> [!KEY]
> IPS, per-decision IS, and DR change how the data is spent, but none creates information the log lacks. When the ESS collapses or coverage fails, the honest answer is that the log cannot tell you.

> [!RECAP]
> - IPS averages $w\,r$ with $w = \pi / b$. SNIPS divides by $\sum w$, which adds bias. DR averages $\hat V(x) + w\,(r - \hat q)$.
> - DR is unbiased for any fixed $\hat q$, and has lower variance when $\hat q$ is close to the true reward.
> - Per-decision IS weights $r_t$ by the prefix product $\rho_{0:t}$. The DR recursion runs backward with a single step ratio $\rho_t$.
> - FQE averages $r + \gamma \sum_{a'} \pi(a' \mid s')\,Q(s', a')$ over each pair's transitions, which has the value of $\pi$ as its fixed point.
> - $\operatorname{ESS} = (\sum w)^2 / \sum w^2$ tells you how many samples the weights leave you.

## Build it step by step

The checks test the functions in this order.

1. **`effective_sample_size(w)`.** Takes `(N,)` weights and returns a float. Return `0.0` when every weight is zero.
2. **`ips` and `snips`.** Take `(N,)` arrays `x, a, r` and the `(K, A)` tables `pi, b`, and return floats. Index `pi[x, a] / b[x, a]` for all weights at once. Common mistake: dividing by `np.sum(w)` in `ips`, which is SNIPS and is biased.
3. **`doubly_robust(x, a, r, pi, b, q_hat)`.** With `q_hat` of shape `(K, A)`, compute `v_hat = (pi * q_hat).sum(axis=1)` and average `v_hat[x] + w * (r - q_hat[x, a])`. With `q_hat` all zeros it equals `ips`.
4. **`per_decision_is(states, actions, rewards, pi, b, gamma)`.** Takes `(N, H)` arrays and returns `(N,)`. Use `np.cumprod` along axis 1. Common mistake: using the product through the last step for every reward.
5. **`doubly_robust_trajectory(..., q)`.** `q` has shape `(H, S, A)`. Loop backward over `t` with `est` of shape `(N,)`. Use the single step ratio `ratio[:, t]`, not the previous step's ratio and not a cumulative product.
6. **`fqe(s, a, r, s2, pi, gamma, n_states, n_actions, iters)`.** Returns `(S, A)`. Group targets by pair with `np.bincount(s * n_actions + a, weights=y)`, divide by the pair counts, and leave unseen pairs at zero.

## Where this breaks

**Estimated propensities.** Real logs often lack $b$. Errors in a fitted $b$ bias IPS and DR, and a tiny estimate makes a huge weight. Clipping weights trades that for a bias you chose.

## Further reading

- [Stanford CS234, Winter 2026 schedule](https://web.stanford.edu/class/cs234/) lists "Offline RL, Imitation Learning", "Offline RL, RLHF", and "Offline RL / Bandits" in weeks 4 to 6. It gives no lecture numbers.
- [Berkeley CS285, Lecture 17 "Offline RL Part 1" and Lecture 18 "Offline RL Part 2"](https://rail.eecs.berkeley.edu/deeprlcourse/) cover the setting that makes evaluation matter.
- [Precup, Sutton, and Singh (2000)](https://dl.acm.org/doi/10.5555/645529.658134) introduces per-decision importance sampling.
- [Dudík, Langford, and Li, "Doubly robust policy evaluation and learning" (2011)](https://arxiv.org/abs/1103.4601) introduces DR for contextual bandits.
- [Jiang and Li, "Doubly robust off-policy value evaluation for reinforcement learning" (2016)](https://arxiv.org/abs/1511.03722) gives the recursive estimator.
- [Thomas and Brunskill, "Data-efficient off-policy policy evaluation for reinforcement learning" (2016)](https://arxiv.org/abs/1604.00923) adds a weighted DR estimator.
- [Le, Voloshin, and Yue, "Batch policy learning under constraints" (2019)](https://arxiv.org/abs/1903.08738) introduces FQE.

## The exercise

You write `effective_sample_size`, `ips`, `snips`, and `doubly_robust` for logged bandit data, and `per_decision_is`, `doubly_robust_trajectory`, and `fqe` for a tabular MDP. The starter gives `finite_horizon_q` and `sample_episodes`.

The checks verify the four-sample log (IPS 0.955, SNIPS 0.735, DR 0.725). They average IPS and DR over 4,000 logs and require the mean to land within four standard errors of the exact value, which the self-normalized estimate fails. DR must have under half of IPS's variance with a close model, and be exact with the true $q$ on one context. The trajectory estimators match the hand examples (6.784 and 4.1632) and the exact finite-horizon value over 20,000 trajectories. Last, `fqe` must match `policy_evaluation` on a log with exact transition frequencies.
