---
title: Baselines and actor-critic
slug: 031-baselines-and-actor-critic
part: "Part 5 — Policy gradients"
summary: Grade each action against a typical outcome to cut the policy gradient's noise, then let a learned critic supply that comparison after every step.
minutes: 40
gpu: false
objectives:
  - Explain why subtracting a baseline that ignores the action changes the gradient's noise but not its average.
  - Compute the exact variance of the score-function estimator and the constant baseline that minimizes it.
  - Say what an actor-critic trades by replacing the sampled return with a one-step TD error.
  - Train a tabular one-step actor-critic and compare it with the optimal policy.
prereqs:
  - 016-temporal-difference-learning
  - 029-the-policy-gradient
  - 030-where-the-gradient-comes-from
exercise: 031-baselines-and-actor-critic
---

# Baselines and actor-critic

A slot machine has three arms. Arm 0 pays 1 on average, arm 1 pays 0, and arm 2 pays 2, with some noise on every pull. You learn which arm to favor by REINFORCE: after each pull, you raise the probability of the arm you pulled, in proportion to the reward it paid.

That sounds right, and on average it is. But look at a single update. Every reward here is zero or larger, so every pull raises the probability of the arm you pulled. Arm 2 gets pushed up a little more than arm 0, and that small difference is the only signal. The large push that every arm shares carries no information, and it makes each individual update point in a wild direction. You must average many pulls before the signal shows.

This chapter fixes that with a *baseline*, a reference number that you subtract from the reward, so each pull is graded as "better or worse than usual". You will see that this costs no bias, measure how much noise it removes, and then replace the reference with a learned *critic* that grades every single step.

## What you already know

- REINFORCE moves the policy along $R \,\nabla_\theta \log \pi(a)$, and its average is the true gradient $\nabla J$.
- For a softmax policy, the score $\nabla_\theta \log \pi(a)$ is `onehot(a) - pi`, and it averages to zero over the policy's own actions.
- The advantage $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ says how much better $a$ is than the policy's average in $s$.
- The TD error $\delta = r + \gamma V(s') - V(s)$ measures how far one step's outcome was from the value estimate.

## Grade on a curve

The picture to hold is a teacher who grades on a curve. A student who scores 70 out of 100 deserves praise if the class average was 50 and a warning if it was 90. The raw score is the same, and the right reaction is opposite. The baseline $b$ is the class average.

For a bandit, one pull of arm $a$ with reward $R$ gives the estimate

$$
\hat g = (R - b)\, \nabla_\theta \log \pi(a)
$$

Read it as: the score of the arm you pulled, scaled by how much the reward beat the baseline. With $b = 0$ you get REINFORCE. With $b$ near the typical reward, an arm that paid above average gets pushed up, and an arm that paid below average gets pushed down.

Figure 1 shows the size and sign of $R - b$ for the three arms' mean rewards, first with no baseline and then with the average reward $b = 1.25$.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Push size and sign for three arms: without a baseline no push is negative, with a baseline of 1.25 arms 0 and 1 are pushed down">
<line x1="70" y1="150" x2="570" y2="150" class="stroke-muted"/>
<text x="64" y="154" class="fig-small" text-anchor="end">0</text>
<line x1="70" y1="195" x2="570" y2="195" class="stroke-grid"/>
<text x="64" y="199" class="fig-small" text-anchor="end">-1</text>
<line x1="70" y1="105" x2="570" y2="105" class="stroke-grid"/>
<text x="64" y="109" class="fig-small" text-anchor="end">1</text>
<line x1="70" y1="60" x2="570" y2="60" class="stroke-grid"/>
<text x="64" y="64" class="fig-small" text-anchor="end">2</text>
<rect x="130" y="105.0" width="44" height="45.0" class="fill-a"/>
<text x="152" y="99.0" class="fig-small" text-anchor="middle">+1.00</text>
<rect x="180" y="150.0" width="44" height="11.2" class="fill-b"/>
<text x="202" y="175.2" class="fig-small" text-anchor="middle">-0.25</text>
<text x="177" y="270" text-anchor="middle">arm 0 (mean 1)</text>
<text x="322" y="144.0" class="fig-small" text-anchor="middle">+0.00</text>
<rect x="350" y="150.0" width="44" height="56.2" class="fill-b"/>
<text x="372" y="220.2" class="fig-small" text-anchor="middle">-1.25</text>
<text x="347" y="270" text-anchor="middle">arm 1 (mean 0)</text>
<rect x="470" y="60.0" width="44" height="90.0" class="fill-a"/>
<text x="492" y="54.0" class="fig-small" text-anchor="middle">+2.00</text>
<rect x="520" y="116.2" width="44" height="33.8" class="fill-b"/>
<text x="542" y="110.2" class="fig-small" text-anchor="middle">+0.75</text>
<text x="517" y="270" text-anchor="middle">arm 2 (mean 2)</text>
<rect x="380" y="20" width="14" height="14" class="fill-a"/><text x="400" y="32" class="fig-small">R − b with b = 0</text>
<rect x="380" y="42" width="14" height="14" class="fill-b"/><text x="400" y="54" class="fig-small">R − b with b = 1.25</text>
<text x="70" y="24" class="fig-small">push size</text>
</svg>
Figure 1. The size and sign of R − b for each arm's mean reward. Notice that without a baseline no arm is ever pushed down, while with b = 1.25 the two weaker arms are pushed down and arm 2 is pushed up.
```

## Why the average does not move

Subtracting a number from every update sounds like it must change the answer. It does not, and the reason takes two lines. The baseline's contribution to the average update is

$$
\mathbb{E}\big[b\, \nabla_\theta \log \pi(a)\big] = b \sum_a \pi(a)\, \nabla_\theta \log \pi(a) = 0
$$

because the score averages to zero over the policy's own actions. So $\mathbb{E}[\hat g] = \nabla J$ for every constant $b$. The same argument holds in an MDP for any baseline $b(s_t)$ that depends on the state but not on the action. A baseline that depends on $a_t$ breaks it, because $b$ no longer factors out of the sum over actions.

In numpy, for a batch of pulls:

```python
scores = np.eye(k)[arms] - pi                  # shape (N, k)
samples = (rewards - b)[:, None] * scores
```

## Work one case by hand

Take $\pi = [0.25, 0.25, 0.5]$ and means $m = [1, 0, 2]$. The expected reward is $V = 0.25 \cdot 1 + 0.25 \cdot 0 + 0.5 \cdot 2 = 1.25$. The exact gradient of $J = \sum_a \pi(a) m_a$ is $\pi \odot (m - V) = [-0.0625, -0.3125, 0.375]$.

Pull arm 2 and get reward 2. Its score is $e_2 - \pi = [-0.25, -0.25, 0.5]$.

- With $b = 0$, the estimate is $2 \times$ score $= [-0.5, -0.5, 1.0]$.
- With $b = 1.25$, the estimate is $0.75 \times$ score $= [-0.1875, -0.1875, 0.375]$.

Pull arm 1 and get reward 0. Its score is $[-0.25, 0.75, -0.25]$.

- With $b = 0$, the estimate is $0 \times$ score $= [0, 0, 0]$. The worst arm produces no push at all.
- With $b = 1.25$, the estimate is $-1.25 \times$ score $= [0.3125, -0.9375, 0.3125]$. The worst arm is pushed down.

Both settings have the same average. The baseline estimates are smaller, and they use the fact that arm 1 is bad.

## Measure the noise exactly

For a vector estimate, use the *total variance*: the expected squared distance between one estimate and its mean. Say arm $a$'s reward has mean $m_a$ and noise with standard deviation $\sigma$. One sample's squared length is $(R - b)^2 \lVert e_a - \pi \rVert^2$, where $e_a$ is the one-hot vector for $a$. Averaging over arms and noise, and subtracting the squared length of the mean, gives

$$
\operatorname{Var}(b) = \sum_a \pi(a) \big[(m_a - b)^2 + \sigma^2\big]
\lVert e_a - \pi \rVert^2 - \lVert \nabla J \rVert^2
$$

Read it piece by piece. $\lVert e_a - \pi \rVert^2$ is how long arm $a$'s score is, and rare arms have long scores. $(m_a - b)^2 + \sigma^2$ is the average of $(R - b)^2$ for arm $a$: the gap between its mean and the baseline, plus its noise. The last term turns an average squared length into a variance. The noise term does not depend on $b$, so no baseline removes the noise in the rewards themselves.

```python
score_sq = ((np.eye(k) - pi) ** 2).sum(axis=1)        # |e_a - pi|^2
var = np.sum(pi * ((means - b) ** 2 + sigma**2) * score_sq) - g @ g
```

With $\sigma = 1$, the score lengths are $[0.875, 0.875, 0.375]$. Figure 2 plots $\operatorname{Var}(b)$ from this formula.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="Total variance of the gradient estimate against the baseline b, a parabola with its lowest point at b = 0.95">
<line x1="70" y1="260.0" x2="560" y2="260.0" class="stroke-grid"/><text x="62" y="264.0" class="fig-small" text-anchor="end">0</text>
<line x1="70" y1="186.7" x2="560" y2="186.7" class="stroke-grid"/><text x="62" y="190.7" class="fig-small" text-anchor="end">1</text>
<line x1="70" y1="113.3" x2="560" y2="113.3" class="stroke-grid"/><text x="62" y="117.3" class="fig-small" text-anchor="end">2</text>
<line x1="70" y1="40.0" x2="560" y2="40.0" class="stroke-grid"/><text x="62" y="44.0" class="fig-small" text-anchor="end">3</text>
<line x1="151.7" y1="260" x2="151.7" y2="265" class="stroke-muted"/><text x="151.7" y="280" class="fig-small" text-anchor="middle">0</text>
<line x1="233.3" y1="260" x2="233.3" y2="265" class="stroke-muted"/><text x="233.3" y="280" class="fig-small" text-anchor="middle">0.5</text>
<line x1="315.0" y1="260" x2="315.0" y2="265" class="stroke-muted"/><text x="315.0" y="280" class="fig-small" text-anchor="middle">1</text>
<line x1="396.7" y1="260" x2="396.7" y2="265" class="stroke-muted"/><text x="396.7" y="280" class="fig-small" text-anchor="middle">1.5</text>
<line x1="478.3" y1="260" x2="478.3" y2="265" class="stroke-muted"/><text x="478.3" y="280" class="fig-small" text-anchor="middle">2</text>
<line x1="560.0" y1="260" x2="560.0" y2="265" class="stroke-muted"/><text x="560.0" y="280" class="fig-small" text-anchor="middle">2.5</text>
<polyline points="70.0,105.9 78.2,112.4 86.3,118.7 94.5,124.8 102.7,130.6 110.8,136.2 119.0,141.6 127.2,146.8 135.3,151.7 143.5,156.4 151.7,160.9 159.8,165.1 168.0,169.1 176.2,172.9 184.3,176.5 192.5,179.8 200.7,182.9 208.8,185.8 217.0,188.4 225.2,190.8 233.3,193.0 241.5,194.9 249.7,196.6 257.8,198.1 266.0,199.4 274.2,200.4 282.3,201.2 290.5,201.8 298.7,202.1 306.8,202.2 315.0,202.1 323.2,201.8 331.3,201.2 339.5,200.4 347.7,199.4 355.8,198.1 364.0,196.6 372.2,194.9 380.3,193.0 388.5,190.8 396.7,188.4 404.8,185.8 413.0,182.9 421.2,179.8 429.3,176.5 437.5,172.9 445.7,169.1 453.8,165.1 462.0,160.9 470.2,156.4 478.3,151.7 486.5,146.8 494.7,141.6 502.8,136.2 511.0,130.6 519.2,124.8 527.3,118.7 535.5,112.4 543.7,105.9 551.8,99.1 560.0,92.1" class="stroke-a" fill="none"/>
<circle cx="151.7" cy="160.9" r="5" class="fill-bad"/>
<text x="161.7" y="152.9" class="fig-small" text-anchor="start">no baseline: 1.3516</text>
<circle cx="355.8" cy="198.1" r="5" class="fill-b"/>
<text x="365.8" y="186.1" class="fig-small" text-anchor="start">b = V = 1.25: 0.8438</text>
<circle cx="306.8" cy="202.2" r="5" class="fill-c"/>
<text x="298.8" y="228.2" class="fig-small" text-anchor="end">b* = 0.95: 0.7875</text>
<text x="315" y="302" text-anchor="middle">baseline b</text><text x="70" y="24" class="fig-small">total variance</text>
</svg>
Figure 2. Total variance of one gradient estimate as the baseline b moves, for the three-arm example with σ = 1. Notice that the curve is a parabola, that the expected reward 1.25 sits close to the bottom, and that the lowest point b* = 0.95 is a little to its left.
```

The curve is a parabola, because $b$ enters squared. Its lowest point is where the derivative is zero:

$$
b^* = \frac{\sum_a \pi(a) \lVert e_a - \pi \rVert^2\, m_a}{\sum_a \pi(a) \lVert e_a - \pi \rVert^2}
$$

Read it as a weighted average of the arm means. Each arm's weight is its probability times its score's squared length, here $[0.21875, 0.21875, 0.1875]$, which sum to $0.625$. So $b^* = (0.21875 + 0 + 0.375) / 0.625 = 0.95$.

| Baseline | $b$ | Total variance |
|---|---|---|
| None | 0 | 1.3516 |
| Expected reward $V$ | 1.25 | 0.8438 |
| Optimal | 0.95 | 0.7875 |

The expected reward $V$ removes most of what a baseline can remove. The optimal baseline leans toward arms with long scores, because those samples carry the most variance. In practice, people use a learned state value $V(s)$, because it is nearly as good and easy to learn.

> [!KEY]
> Any baseline that ignores the action leaves the gradient's average unchanged, because the score averages to zero. A good baseline removes the variance that comes from rewards being far from typical. It cannot remove the rewards' own noise.

## Let a critic grade every step

A baseline $V(s_t)$ turns $G_t - V(s_t)$ into an estimate of the advantage. But $G_t$ still adds up every later reward with all their noise, and the policy cannot learn until the episode ends.

An *actor-critic* goes further. The *actor* is the policy. The *critic* is a learned value table or network $V_\phi$. Instead of waiting for the whole return, the critic's one-step guess $r_t + \gamma V_\phi(s_{t+1})$ replaces $G_t$, which makes the actor's signal the TD error:

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

Read it as: the reward you got, plus the discounted value of where you landed, minus the value you expected. If $\delta_t > 0$, the step went better than expected, so the action you took becomes more likely. The same $\delta_t$ trains the critic by TD(0).

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="One step of an actor-critic: the actor picks an action, the environment returns a reward and next state, the critic forms the TD error, and the TD error updates both">
<defs><marker id="c031-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<rect x="30" y="40" width="150" height="56" rx="6" class="fill-soft-b stroke-muted"/>
<text x="105.0" y="68.0" class="fig-bold" text-anchor="middle">actor</text>
<text x="105.0" y="84.0" class="fig-small" text-anchor="middle">π(a | s), θ</text>
<rect x="420" y="40" width="150" height="56" rx="6" class="fill-soft-a stroke-muted"/>
<text x="495.0" y="68.0" class="fig-bold" text-anchor="middle">environment</text>
<text x="495.0" y="84.0" class="fig-small" text-anchor="middle">returns r and s2</text>
<rect x="225" y="170" width="150" height="56" rx="6" class="fill-soft-c stroke-muted"/>
<text x="300.0" y="198.0" class="fig-bold" text-anchor="middle">critic</text>
<text x="300.0" y="214.0" class="fig-small" text-anchor="middle">V(s), V(s2)</text>
<line x1="180" y1="68" x2="420" y2="68" class="stroke-ink" marker-end="url(#c031-arrow)"/>
<text x="300" y="60" class="fig-small" text-anchor="middle">action a</text>
<line x1="495" y1="96" x2="375" y2="185" class="stroke-ink" marker-end="url(#c031-arrow)"/>
<text x="470" y="155" class="fig-small" text-anchor="middle">r, s2</text>
<line x1="225" y1="198" x2="105" y2="100" class="stroke-ink" marker-end="url(#c031-arrow)"/>
<text x="60" y="150" class="fig-small" text-anchor="middle">TD error</text>
<text x="60" y="166" class="fig-small" text-anchor="middle">δ = r + γV(s2) − V(s)</text>
<text x="105" y="122" class="fig-small" text-anchor="start">actor: θ += α δ score</text>
<text x="300" y="250" class="fig-small" text-anchor="middle">critic: V(s) += α δ</text>
</svg>
Figure 3. One step of a tabular actor-critic. Notice that the same TD error δ drives both updates: it moves the critic toward the one-step target, and it scales the actor's step along the score.
```

If the critic is exact, the average TD error after taking $a$ in $s$ is the advantage:

$$
\mathbb{E}[\delta_t \mid s_t = s, a_t = a] = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a)\, V^\pi(s') - V^\pi(s) = A^\pi(s, a)
$$

The first two terms are $Q^\pi(s, a)$, and subtracting $V^\pi(s)$ leaves the advantage. In numpy, the expected TD error for every state and action is `mdp.R + gamma * mdp.P @ v - v[:, None]`.

### What an imperfect critic costs

Write the critic's error as $e = V_\phi - V^\pi$. The average TD error becomes $A^\pi(s, a) + \gamma \sum_{s'} P(s' \mid s, a)\, e(s') - e(s)$. The two error terms behave differently:

- $-e(s)$ depends only on the current state, so it acts as a baseline. It adds variance but no bias.
- $\gamma \sum_{s'} P(s' \mid s, a)\, e(s')$ depends on the action through where the action leads. It biases the actor toward actions that lead to overvalued states.

A critic that is off by the same constant everywhere does no harm, because that constant cancels in both terms. This is the trade in an actor-critic: lower variance, paid for with bias from the critic's mistakes.

> [!KEY]
> An actor-critic trades variance for bias. With an exact critic, the TD error averages to the advantage. With an inexact one, its error at the current state is a harmless baseline, but its error at the next state biases the update.

> [!RECAP]
> - A baseline $b$ scales the score by $R - b$ and leaves the average unchanged.
> - Total variance is $\sum_a \pi(a)[(m_a - b)^2 + \sigma^2] \lVert e_a - \pi \rVert^2 - \lVert \nabla J \rVert^2$.
> - $b^*$ is the average of $m_a$ weighted by $\pi(a) \lVert e_a - \pi \rVert^2$.
> - The expected TD error is `R + gamma * P @ v - v[:, None]`, and it equals $A^\pi$ when `v` is $V^\pi$.
> - One step of a tabular actor-critic updates the critic by $\delta$ and the actor by $\delta$ times the score.

## Build it step by step

The checks test the functions in this order. `softmax` is given.

1. **`bandit_gradient_samples(theta, arms, rewards, baseline)`.** Takes `(k,)` logits and `(N,)` arms and rewards, and returns `(N, k)`. Build all scores at once with `np.eye(k)[arms] - pi`. The common mistake is to scale by `rewards` and forget the baseline.
2. **`exact_bandit_gradient(theta, means)`.** Returns `(k,)`: `pi * (means - pi @ means)`.
3. **`estimator_variance(theta, means, sigma, baseline)`.** Returns a float. The common mistake is to return the second moment without subtracting $\lVert \nabla J \rVert^2$.
4. **`optimal_baseline(theta, means)`.** Returns a float: the average of `means` weighted by `pi * score_sq`. It does not depend on `sigma`.
5. **`expected_td_error(mdp, v)`.** Returns `(S, A)`. The common mistake is to subtract `v` without `[:, None]`, which broadcasts over the wrong axis when `S == A`.
6. **`actor_critic(mdp, steps, alpha_actor, alpha_critic, seed)`.** Returns `(theta, v)`. Draw everything from one `np.random.default_rng(seed)`, and use `mdp.sample(s, a, rng)` for each transition. Update `v[s]`, not `v[s2]`. Then continue from `s2`, except that with probability $1 - \gamma$ you restart from a fresh draw of `mdp.mu0`. The restart makes the states you learn from follow the discounted visitation that the policy gradient theorem weights by.

> [!TIP]
> Give the critic a larger step size than the actor. The actor's signal is only as good as the critic, so the critic has to keep up with the policy it evaluates.

## Where this breaks

**A bad critic misleads the actor.** Early in training, the critic knows nothing, and the actor follows its errors. People warm the critic up first, or use multi-step targets that lean less on it. The next chapter, [Generalized advantage estimation](/lessons/032-generalized-advantage-estimation), makes that dial explicit.

**Two step sizes to tune.** If the actor moves faster than the critic can track, the TD errors describe a policy that no longer exists. Konda and Tsitsiklis (2000) proved convergence when the critic runs on a faster timescale.

## Further reading

- [Greensmith, Bartlett, and Baxter (2004), Variance reduction techniques for gradient estimates in reinforcement learning](https://www.jmlr.org/papers/v5/greensmith04a.html) derives optimal baselines.
- [Konda and Tsitsiklis (2000), Actor-critic algorithms](https://proceedings.neurips.cc/paper/1999/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) proves convergence with a faster critic.
- [Sutton and Barto (2018), Reinforcement Learning: An Introduction, chapter 13](http://incompleteideas.net/book/the-book-2nd.html) covers REINFORCE with a baseline and the one-step actor-critic.

## The exercise

You build both halves of this chapter. For the bandit, write `bandit_gradient_samples`, `exact_bandit_gradient`, `estimator_variance`, and `optimal_baseline`. For the MDP, write `expected_td_error` and the online learner `actor_critic`.

The checks verify the per-sample estimates and $b^* = 0.95$ against the worked example, confirm the gradient by finite differences, and compare `estimator_variance` with sampled variance. They check that `expected_td_error` with the true $V^\pi$ equals the advantage. Finally, `actor_critic` must learn $V^\pi$ of the uniform policy when the actor is frozen, and must close at least 90% of the gap to the optimum on a random five-state MDP.
