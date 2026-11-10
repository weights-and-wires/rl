---
title: IMPALA and V-trace
slug: 038-impala-and-v-trace
part: "Part 5 — Policy gradients"
summary: Correct for actors that lag behind the learner with V-trace targets, and see exactly which policy they converge to.
minutes: 40
gpu: false
objectives:
  - Explain why parallel actors act with older parameters than the learner holds, and what that costs.
  - Compute V-trace targets with clipped weights by a backward recursion, and say what each clip controls.
  - Show that V-trace reduces to the n-step return on-policy and to a one-step TD target when c-bar is zero.
  - State the fixed point of the V-trace operator, the policy it evaluates, and check it on a tabular MDP.
  - Build the V-trace policy-gradient advantage from the targets.
prereqs:
  - 015-off-policy-importance-sampling
  - 019-n-step-returns
  - 033-baselines-and-actor-critic
  - 037-ppo-in-practice
exercise: 038-impala-and-v-trace
---

# IMPALA and V-trace

You have 32 actors, each playing its own copy of a game, and one learner on a GPU. In a synchronous design such as A2C, every actor plays a few steps with the same parameters, then everyone stops while the learner updates, then everyone starts again. The learner idles while the actors play, and the actors idle while the learner computes. Most of the machine does nothing at any moment.

The fix is to let nobody wait. Actors keep playing and drop their trajectories into a queue, and the learner takes whatever is there. This is the idea behind A3C (Mnih et al., 2016) and IMPALA (Espeholt et al., 2018). It creates a problem. By the time the learner reads a trajectory, the learner's parameters have moved on, so the actions in the trajectory came from an older policy than the one you are training.

This chapter measures that lag, shows why ignoring it or correcting it naively fails, and builds the correction IMPALA uses, called V-trace. Its payoff is unusual: you can say exactly which policy it learns the value of, and you can check that claim numerically on a small MDP.

## What you already know

- Importance sampling reweights a sample from a behavior policy $\mu$ by $\pi(a \mid s) / \mu(a \mid s)$ so that it estimates an average under $\pi$ (see [Learning about one policy from another](/lessons/015-off-policy-importance-sampling)).
- An $n$-step return adds up $n$ rewards and then bootstraps from the value of the state reached (see [n-step returns](/lessons/019-n-step-returns)).
- An actor-critic method updates the policy with an advantage and the critic with a target (see [Baselines and actor-critic](/lessons/033-baselines-and-actor-critic)).
- A terminal step has no future, so nothing bootstraps from it.
- The reward for taking $a_t$ in $s_t$ is $r_t$. The IMPALA paper writes the state as $x_t$. This chapter keeps $s_t$.

## Measure the lag

Picture an assembly line. An actor takes 8 ticks to produce an unroll, a short trajectory. The learner takes 1 tick to update on one unroll, and each update raises a version counter by one. An actor reads the newest version when it starts an unroll. The learner takes the oldest unroll in a queue that holds 4. When the queue is full, a finished actor waits, still holding its old version.

Define the *lag* of an unroll as the number of learner updates between the version its actor used and the version the learner holds when it takes the unroll. A synchronous A2C round has lag 0, because every actor uses the current version. It costs idle time instead: each round takes 8 ticks of acting plus one tick of learning per unroll.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Learner utilization against the number of actors for synchronous A2C and asynchronous actors, with the mean staleness of the asynchronous batches">
<line x1="70" y1="250" x2="560" y2="250" class="stroke-muted"/>
<line x1="70" y1="250.0" x2="560" y2="250.0" class="stroke-grid"/><text x="62" y="254.0" class="fig-small" text-anchor="end">0</text>
<line x1="70" y1="145.0" x2="560" y2="145.0" class="stroke-grid"/><text x="62" y="149.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="70" y1="40.0" x2="560" y2="40.0" class="stroke-grid"/><text x="62" y="44.0" class="fig-small" text-anchor="end">1</text>
<text x="70.0" y="270" class="fig-small" text-anchor="middle">1</text>
<text x="168.0" y="270" class="fig-small" text-anchor="middle">2</text>
<text x="266.0" y="270" class="fig-small" text-anchor="middle">4</text>
<text x="364.0" y="270" class="fig-small" text-anchor="middle">8</text>
<text x="462.0" y="270" class="fig-small" text-anchor="middle">16</text>
<text x="560.0" y="270" class="fig-small" text-anchor="middle">32</text>
<polyline points="70.0,226.7 168.0,208.0 266.0,180.0 364.0,145.0 462.0,110.0 560.0,82.0" class="stroke-a" fill="none"/>
<circle cx="70.0" cy="226.7" r="4" class="fill-a"/>
<circle cx="168.0" cy="208.0" r="4" class="fill-a"/>
<circle cx="266.0" cy="180.0" r="4" class="fill-a"/>
<circle cx="364.0" cy="145.0" r="4" class="fill-a"/>
<circle cx="462.0" cy="110.0" r="4" class="fill-a"/>
<circle cx="560.0" cy="82.0" r="4" class="fill-a"/>
<polyline points="70.0,223.8 168.0,197.5 266.0,145.4 364.0,43.6 462.0,43.6 560.0,43.6" class="stroke-b" fill="none"/>
<circle cx="70.0" cy="223.8" r="4" class="fill-b"/>
<circle cx="168.0" cy="197.5" r="4" class="fill-b"/>
<circle cx="266.0" cy="145.4" r="4" class="fill-b"/>
<circle cx="364.0" cy="43.6" r="4" class="fill-b"/>
<circle cx="462.0" cy="43.6" r="4" class="fill-b"/>
<circle cx="560.0" cy="43.6" r="4" class="fill-b"/>
<text x="70.0" y="290" class="fig-small fig-bold" text-anchor="middle">1.0</text>
<text x="168.0" y="290" class="fig-small fig-bold" text-anchor="middle">2.5</text>
<text x="266.0" y="290" class="fig-small fig-bold" text-anchor="middle">5.5</text>
<text x="364.0" y="290" class="fig-small fig-bold" text-anchor="middle">10.1</text>
<text x="462.0" y="290" class="fig-small fig-bold" text-anchor="middle">18.5</text>
<text x="560.0" y="290" class="fig-small fig-bold" text-anchor="middle">33.4</text>
<text x="62" y="290" class="fig-small" text-anchor="end">lag</text>
<text x="315.0" y="314" text-anchor="middle">number of actors</text>
<text x="62" y="28" class="fig-small" text-anchor="end">learner busy</text>
<line x1="380" y1="62" x2="408" y2="62" class="stroke-b"/><text x="414" y="66" class="fig-small">asynchronous</text>
<line x1="380" y1="82" x2="408" y2="82" class="stroke-a"/><text x="414" y="86" class="fig-small">synchronous A2C</text>
</svg>
Figure 1. Learner busy time against the number of actors, from an exact count with 8 ticks per unroll, 1 tick per update, and a queue of 4. Asynchronous actors keep the learner busy, and the number under each point is the mean lag in updates, which grows with the actors.
```

Figure 1 comes from an exact tick-by-tick count, in `scripts/figures/038-impala-and-v-trace.py`. With 8 actors the synchronous learner is busy half the time, and the asynchronous learner is busy 98% of it, nearly twice the throughput. The price is a mean lag of 10.1 updates. With 32 actors the lag grows to 33.4.

> [!KEY]
> Asynchronous actors trade idle time for lag. More actors keep the learner busy, and they make every trajectory older.

## Why not ignore the lag

A 10-update lag means the actor's policy $\mu$ and the learner's policy $\pi$ differ. If you treat the trajectory as on-policy, you push the policy toward actions that $\mu$ liked, which $\pi$ may no longer prefer. In [Stale samples and importance weights](/lessons/064-rl-infrastructure-for-llms), the gradient from stale samples points partly backward at a lag of 8.

The obvious correction is importance sampling. Weight each term of the $n$-step return by the product of ratios along the trajectory. That is unbiased, and its variance grows with the length of the product. A ratio of 2 at five steps in a row multiplies the weight by 32, and a single ratio near 0 erases everything after it.

Truncation tames the product. You give up some bias, and you keep control of the variance. V-trace chooses where to truncate, and it uses two separate clips for two separate jobs.

## Build V-trace in words

Start from the TD error you know. At step $t$, the critic's one-step surprise is $r_t + \gamma V(s_{t+1}) - V(s_t)$. V-trace does three things with it:

1. It scales the surprise by a clipped ratio $\rho_t = \min(\bar\rho, \pi/\mu)$. A step where $\pi$ would rarely have acted gets less say.
2. It lets the surprise at step $t$ also correct the estimates of earlier states $s_{t-1}, s_{t-2}, \dots$, as an $n$-step return does. Each hop back multiplies by $\gamma$ and by a second clipped ratio $c_t = \min(\bar c, \pi/\mu)$. This is the *trace*.
3. It resets at episode ends. After a terminal step, the discount is zero, so the next episode's values never flow back.

The two clips do different jobs. $\bar\rho$ decides which policy the values describe. $\bar c$ decides how far a correction travels: it sets the speed at which information moves back, and it controls variance, but it does not change the answer.

## Work one example

Take $\gamma = 0.9$ and a three-step rollout with rewards $[1, 0, 1]$, critic values $V = [0.5, 0.4, 0.3]$, and a value $V(s_3) = 0.2$ for the state after the last step. The ratios $\pi/\mu$ are $[0.5, 1.5, 2.0]$. With $\bar\rho = \bar c = 1$, the clipped weights are $\rho = c = [0.5, 1, 1]$, because the ratios 1.5 and 2.0 clip to 1.

| $t$ | $r_t$ | $V(s_t)$ | $\rho_t$ | $c_t$ | $\delta_t = \rho_t (r_t + \gamma V(s_{t+1}) - V(s_t))$ |
|---|---|---|---|---|---|
| 0 | 1 | 0.5 | 0.5 | 0.5 | $0.5 \times (1 + 0.9 \times 0.4 - 0.5) = 0.43$ |
| 1 | 0 | 0.4 | 1 | 1 | $0 + 0.9 \times 0.3 - 0.4 = -0.13$ |
| 2 | 1 | 0.3 | 1 | 1 | $1 + 0.9 \times 0.2 - 0.3 = 0.88$ |

Work backward from the end, where the correction is 0. Call the running correction $a_t$.

- $a_2 = 0.88$.
- $a_1 = -0.13 + 0.9 \times 1 \times 0.88 = 0.662$.
- $a_0 = 0.43 + 0.9 \times 0.5 \times 0.662 = 0.7279$.

The target is the critic's value plus the correction: $v = [0.5 + 0.7279,\ 0.4 + 0.662,\ 0.3 + 0.88] = [1.2279, 1.062, 1.18]$.

Now set every ratio to 1, as when the actor is fully up to date. Then the targets become $[1.9558, 1.062, 1.18]$. The first is $1 + 0.9 \times 0 + 0.81 \times 1 + 0.729 \times 0.2 = 1.9558$, the 3-step return. The critic's values $0.5$, $0.4$, and $0.3$ cancel out of it entirely. Set $\bar c = 0$ instead, and the correction cannot travel: the targets become $V_t + \rho_t \delta$ terms alone, $[0.93, 0.27, 1.18]$, which is a one-step TD target weighted by $\rho_t$.

## The formula

For a rollout $t = 0, \dots, T-1$, define $d_t = \gamma (1 - \text{terminated}_t)$ and

$$
\delta_t = \rho_t \left( r_t + d_t V(s_{t+1}) - V(s_t) \right).
$$

Read it aloud: the critic's surprise at step $t$, shrunk by how plausible the step is under $\pi$. The V-trace target at $s_t$ is

$$
v_t = V(s_t) + \sum_{k=t}^{T-1} \left( \prod_{i=t}^{k-1} d_i c_i \right) \delta_k.
$$

Read it aloud: start from the critic's value, then add every later surprise, each discounted and multiplied by the trace weights it passed through. An empty product is 1. Pull out the first term of the sum and the rest is the same expression one step later, which gives the recursion you will write:

$$
v_t - V(s_t) = \delta_t + d_t c_t \left( v_{t+1} - V(s_{t+1}) \right), \qquad v_T = V(s_T).
$$

In numpy this is a loop from $T - 1$ down to 0 over a running scalar, which is the table above.

## Check the limits

The formula contains three familiar methods.

- **On-policy.** With $\pi = \mu$ every ratio is 1, so $\rho_t = c_t = 1$ for $\bar\rho, \bar c \geq 1$. The sum telescopes, the values cancel, and $v_t$ is the $n$-step return that ends at the rollout's end or the episode's end.
- **No trace.** With $\bar c = 0$ every product is empty past its first term, so $v_t = V(s_t) + \delta_t$. This is TD(0) with the surprise scaled by $\rho_t$.
- **No clipping.** With $\bar\rho = \bar c = \infty$ you get the full importance-sampled $n$-step return, unbiased and with the largest variance.

## Find the fixed point

A target is only useful if repeated updates settle somewhere sensible. Apply the V-trace operator to a value table over and over, averaging over actions taken by $\mu$. Where does it stop?

Not at $V^\pi$, unless $\bar\rho$ is large. It stops at the value of an *interpolated policy*:

$$
\pi_{\bar\rho}(a \mid s) = \frac{\min\left(\bar\rho\, \mu(a \mid s),\ \pi(a \mid s)\right)}{\sum_{b} \min\left(\bar\rho\, \mu(b \mid s),\ \pi(b \mid s)\right)}.
$$

Read it aloud: take $\pi$, but never give an action more than $\bar\rho$ times the probability $\mu$ gave it, then renormalize. If $\bar\rho$ exceeds every ratio, $\pi_{\bar\rho} = \pi$. If $\bar\rho$ is tiny, $\pi_{\bar\rho}$ approaches $\mu$.

The reason takes four sentences. Setting the operator's expected update to zero gives $\sum_a \mu(a \mid s)\, \rho(s, a) \left(r + \gamma \mathbb{E}[V(s')] - V(s)\right) = 0$ at every state. Because $\mu \rho = \min(\bar\rho \mu, \pi)$, this is the same equation weighted by $\pi_{\bar\rho}$ up to the positive constant in the denominator. Dividing it out leaves the Bellman equation of $\pi_{\bar\rho}$. The trace weight $c$ never appears, which is why $\bar c$ cannot change the answer.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="Value of the V-trace fixed point at one state as rho-bar grows, moving from the behavior policy value toward the target policy value">
<line x1="80" y1="250" x2="560" y2="250" class="stroke-muted"/>
<line x1="80" y1="209.0" x2="560" y2="209.0" class="stroke-a dashed"/><text x="560" y="203.0" class="fig-small" text-anchor="end">V of mu = 0.05</text>
<line x1="80" y1="58.7" x2="560" y2="58.7" class="stroke-c dashed"/><text x="86" y="52.7" class="fig-small">V of pi = 1.66</text>
<polyline points="80.0,195.0 128.2,188.9 167.1,202.5 191.8,215.8 220.0,227.7 240.0,231.3 268.2,224.9 288.2,205.2 316.3,168.5 351.8,155.8 400.0,145.0 476.3,104.1 560.0,58.7" class="stroke-b" fill="none"/>
<circle cx="80.0" cy="195.0" r="4" class="fill-b"/>
<circle cx="128.2" cy="188.9" r="4" class="fill-b"/>
<circle cx="167.1" cy="202.5" r="4" class="fill-b"/>
<circle cx="191.8" cy="215.8" r="4" class="fill-b"/>
<circle cx="220.0" cy="227.7" r="4" class="fill-b"/>
<circle cx="240.0" cy="231.3" r="4" class="fill-b"/>
<circle cx="268.2" cy="224.9" r="4" class="fill-b"/>
<circle cx="288.2" cy="205.2" r="4" class="fill-b"/>
<circle cx="316.3" cy="168.5" r="4" class="fill-b"/>
<circle cx="351.8" cy="155.8" r="4" class="fill-b"/>
<circle cx="400.0" cy="145.0" r="4" class="fill-b"/>
<circle cx="476.3" cy="104.1" r="4" class="fill-b"/>
<circle cx="560.0" cy="58.7" r="4" class="fill-b"/>
<line x1="80.0" y1="250" x2="80.0" y2="255" class="stroke-muted"/><text x="80.0" y="270" class="fig-small" text-anchor="middle">0.1</text>
<line x1="191.8" y1="250" x2="191.8" y2="255" class="stroke-muted"/><text x="191.8" y="270" class="fig-small" text-anchor="middle">0.5</text>
<line x1="240.0" y1="250" x2="240.0" y2="255" class="stroke-muted"/><text x="240.0" y="270" class="fig-small" text-anchor="middle">1</text>
<line x1="316.3" y1="250" x2="316.3" y2="255" class="stroke-muted"/><text x="316.3" y="270" class="fig-small" text-anchor="middle">3</text>
<line x1="400.0" y1="250" x2="400.0" y2="255" class="stroke-muted"/><text x="400.0" y="270" class="fig-small" text-anchor="middle">10</text>
<line x1="560.0" y1="250" x2="560.0" y2="255" class="stroke-muted"/><text x="560.0" y="270" class="fig-small" text-anchor="middle">100</text>
<text x="320.0" y="294" text-anchor="middle">rho-bar (log scale)</text>
<text x="72" y="28" class="fig-small" text-anchor="end">fixed-point value at state 0</text>
</svg>
Figure 2. The fixed point of the V-trace operator at state 0 as the clip rho-bar grows. It starts near the value of the behavior policy, ends at the value of the target policy, and need not move in a straight line between them.
```

Figure 2 checks this on a random MDP with 5 states and 3 actions, from `rlenvs.random_mdp` with $\gamma = 0.9$. For each $\bar\rho$, the operator is iterated from zero and compared with `policy_evaluation` on $\pi_{\bar\rho}$. They agree to $10^{-6}$ at all 13 values. At state 0, the fixed point moves from near $V^\mu = 0.05$ to $V^\pi = 1.66$ as $\bar\rho$ reaches 100, the largest ratio in this MDP (95.4). The path between is not monotone: it dips to $-0.18$ at $\bar\rho = 1$, because the interpolated policy changes at every state at once, so the value at one state can move either way.

> [!KEY]
> $\bar\rho$ chooses the policy whose value you learn, anywhere from $\mu$ to $\pi$. $\bar c$ chooses how fast information flows and how much variance it carries. Neither is a free hyperparameter: $\bar\rho = 1$ is not "no correction", it is a decision to learn $\pi_1$.

## Use the targets in the policy gradient

The critic regresses $V(s_t)$ toward $v_t$. The actor needs an advantage. IMPALA replaces the critic's next value with the next target:

$$
A_t = \rho_t \left( r_t + d_t v_{t+1} - V(s_t) \right), \qquad v_T = V(s_T).
$$

Read it aloud: the one-step advantage you know from [Baselines and actor-critic](/lessons/033-baselines-and-actor-critic), with $v_{t+1}$ in place of $V(s_{t+1})$ and an importance weight in front. In the worked example, $A = [0.7279, 0.662, 0.88]$. On-policy, $A_t$ is the $n$-step return minus $V(s_t)$. The policy gradient is $A_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$ with $A_t$ held constant, and the clip $\bar\rho$ bounds the weight, at the price of the bias toward $\mu$ that Figure 2 shows.

## Build it step by step

> [!RECAP]
> - The two clips act separately: $\rho = \min(\bar\rho, \pi/\mu)$ and $c = \min(\bar c, \pi/\mu)$.
> - Targets come from a backward loop over $\delta_t$, with a running correction multiplied by $d_t c_t$.
> - Terminal steps zero $d_t$, which stops the loop from crossing episodes.
> - On-policy the targets are the $n$-step return, and $\bar c = 0$ gives a one-step target.
> - The fixed point is the value of $\pi_{\bar\rho}$, for any $\bar c$.

1. `clipped_ratios(log_rhos, rho_bar, c_bar)` takes $\log \pi - \log \mu$ and returns `(rho, c)`. Exponentiate once and use `np.minimum` twice. The common mistake is clipping the log instead of the ratio.
2. `vtrace_targets(rewards, values, bootstrap_value, terminated, log_rhos, gamma, rho_bar, c_bar)` returns $v$ of shape `(T,)`. Build `d = gamma * (1 - terminated)` first, shift `values` left and append `bootstrap_value`, then loop backward. The common mistake is using `gamma` instead of `d`, or `rho` instead of `c`, inside the loop.
3. `vtrace_advantages(rewards, values, vs, bootstrap_value, terminated, log_rhos, gamma, rho_bar)` returns $A_t$. Shift `vs` left and append `bootstrap_value`. The common mistake is using the critic's next value in place of the next target.
4. `interpolated_policy(pi, mu, rho_bar)` returns $\pi_{\bar\rho}$ with rows summing to 1. Use `np.minimum(rho_bar * mu, pi)` and normalize each row.
5. `vtrace_operator(mdp, pi, mu, V, rho_bar, c_bar)` applies the operator once in expectation. Compute $g$ with `np.einsum("sat,t->sa", mdp.P, V)`, build $M$ from `mu`, the clipped `c`, and `mdp.P` with one `np.einsum`, and call `np.linalg.solve` on $I - \gamma M$. The common mistake is weighting the transition matrix by $\rho$ instead of $c$.

## Where this breaks

**Lag past the clip.** If $\pi$ and $\mu$ drift far enough that most ratios sit at 0 or above $\bar\rho$, the small ratios shrink $\delta_t$ toward 0 and the targets collapse to the critic's own values, while the large ratios clip, so you learn $\pi_{\bar\rho}$, which can sit far from $\pi$. Bound the lag, as the bounded queue in Figure 1 tries to, and log the mean clipped ratio and the fraction of ratios at the clip.

**No reset at episode ends.** Using $\gamma$ at a terminal step leaks the next episode's values backward. The error is silent: targets still look plausible.

**Mixing up the two clips.** Using $\rho$ in the trace product changes how far corrections travel and makes $\bar c$ meaningless. The code looks almost identical.

**Clipping is bias.** At $\bar\rho = 1$ you evaluate $\pi_1$, not $\pi$. IMPALA accepts this because the variance of the unclipped product is worse.

## Further reading

- [Espeholt et al. (2018), IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561). Introduces the actor-learner split and V-trace, and proves the contraction and fixed-point results used here.
- [Mnih et al. (2016), Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783). A3C: asynchronous actors that apply their own gradient updates instead of queuing trajectories.
- [Silver, UCL Advanced Topics 2015 (COMPM050/COMPGI13) reinforcement learning course, Lecture 7: Policy Gradient Methods](https://davidstarsilver.wordpress.com/teaching/). Actor-critic methods, the starting point for the advantage used here.
- [CMU 10-703, Deep Reinforcement Learning, schedule](https://www.andrew.cmu.edu/course/10-703/). The lectures "Policy Gradients I" and "Policy Gradients II" cover the actor-critic base; the schedule has no lecture titled for A3C or IMPALA.
- [Hugging Face Deep RL course, Unit 6: Advantage Actor Critic (A2C)](https://huggingface.co/learn/deep-rl-course/unit6/hands-on). The synchronous baseline this chapter compares against.

## The exercise

You write five functions: `clipped_ratios`, `vtrace_targets`, `vtrace_advantages`, `interpolated_policy`, and `vtrace_operator`. The ten checks run from the smallest piece to the whole. They compare targets with the three-step example above and with a double-sum definition that uses no recursion, confirm that on-policy targets equal the $n$-step return even when the critic's values change, and confirm the TD limit at $\bar c = 0$. They test a rollout with a terminal step, and they test separate clips where $\bar\rho \neq \bar c$. They check the operator on a random MDP: the value of $\pi_{\bar\rho}$ is a fixed point for several $\bar\rho$ and $\bar c$, and iterating from zero reaches it.
