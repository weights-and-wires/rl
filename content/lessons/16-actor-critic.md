---
title: Baselines and actor-critic
slug: 16-actor-critic
part: "Part 4 — Policy optimisation"
summary: Cut the variance of policy gradients with a baseline, find the best constant one, then replace returns with a learned critic.
minutes: 60
gpu: false
objectives:
  - Show that subtracting any baseline that ignores the action leaves the policy gradient unbiased.
  - Compute the exact variance of the score-function estimator and the constant baseline that minimises it.
  - Explain which part of a critic's error biases the actor and which part acts as a harmless baseline.
  - Train a tabular one-step actor-critic and compare it with the optimal policy.
prereqs:
  - 07-temporal-difference-learning
  - 15-policy-gradient-theorem
exercise: 16-actor-critic
---

# Baselines and actor-critic

REINFORCE is unbiased, and it's painfully noisy. Suppose every reward in a
game is positive. Then every sampled action gets its probability pushed up,
the good ones slightly more than the bad ones. The signal you want is that
small difference. The large push that every action shares is noise, and the
estimator spends most of its samples averaging it away.

A *baseline* fixes this by grading on a curve. Instead of asking "was the
return high?", you ask "was the return higher than usual?" Subtracting the
usual amount removes the shared push and keeps the difference. It turns out
this never changes the average direction, so it costs nothing in bias.

This lesson measures exactly how much variance a baseline removes and finds
the constant baseline that removes the most. Then it goes one step further.
An *actor-critic* learns a value function, the *critic*, and uses its one-step
prediction error in place of the whole sampled return. That trades a little
bias for a large cut in variance, and it lets the policy, the *actor*, learn
after every step instead of after every episode.

## What you already know

- REINFORCE estimates $\nabla J$ with
  $\sum_t \gamma^t G_t \nabla \log \pi(a_t \mid s_t)$, and its average over
  episodes equals the exact gradient.
- The score $\nabla \log \pi(a \mid s)$ of a softmax is `onehot(a) - pi[s]`,
  and its average over actions is zero.
- The advantage $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ says how much better
  $a$ is than the policy's average in $s$.
- The TD error $\delta = r + \gamma V(s') - V(s)$ measures how far one step's
  outcome was from the value estimate, as in
  *Temporal-difference learning*.
- The standard error of a sample mean is the standard deviation over
  $\sqrt{N}$.

## Subtract a baseline

Start with the simplest case, a *bandit*: one state, $k$ arms, and a softmax
policy $\pi = \operatorname{softmax}(\theta)$ over them. Arm $a$ pays a
random reward with mean $m_a$. The objective is the expected reward,
$J(\theta) = \sum_a \pi(a) m_a$. One pull of arm $a$ with reward $R$ gives
the estimate

$$
\hat g = (R - b)\, \nabla_\theta \log \pi(a)
$$

Read it as: take the score of the arm you pulled, and scale it by how much
the reward beat the baseline $b$. With $b = 0$ it's REINFORCE. With $b$ near
the typical reward, arms that paid above average get pushed up and arms that
paid below average get pushed down.

Why doesn't $b$ change the average? Its contribution is
$b \sum_a \pi(a) \nabla \log \pi(a)$, and that sum is the zero-mean property
of the score. So $\mathbb{E}[\hat g] = \nabla J$ for every constant $b$. In
numpy, for a batch of pulls:

```python
scores = np.eye(k)[arms] - pi          # one row per sample, shape (N, k)
samples = (rewards - b)[:, None] * scores
```

The same argument works in an MDP for any baseline $b(s_t)$ that depends on
the state but not on the action. The estimator becomes
$\sum_t \gamma^t (G_t - b(s_t)) \nabla \log \pi(a_t \mid s_t)$, and it's still
unbiased. A baseline that depends on the action $a_t$ breaks the argument,
because it no longer factors out of the sum over actions.

The exact gradient has a closed form you can check against:
$\partial J / \partial \theta_a = \pi(a) (m_a - \sum_{a'} \pi(a') m_{a'})$.
It's the bandit version of $d^\pi \pi A^\pi$ from the previous lesson.

## Measure the variance exactly

For a vector-valued estimator, use the *total variance*: the expected squared
distance between one estimate and its mean. It's the sum of the variances of
the components. Say arm $a$'s reward has mean $m_a$ and noise with standard
deviation $\sigma$. The squared length of one sample is
$(R - b)^2 \lVert e_a - \pi \rVert^2$, where $e_a$ is the one-hot vector for
$a$. Averaging over arms and noise gives

$$
\operatorname{Var}(b) = \sum_a \pi(a) \big[(m_a - b)^2 + \sigma^2\big]
\lVert e_a - \pi \rVert^2 - \lVert \nabla J \rVert^2
$$

Read it piece by piece:

- $\lVert e_a - \pi \rVert^2$ is how long arm $a$'s score vector is. Rare arms
  have long scores.
- $(m_a - b)^2 + \sigma^2$ is the average of $(R - b)^2$ for arm $a$: the gap
  between its mean and the baseline, plus its noise.
- The first sum is the average squared length. Subtracting the squared length
  of the mean turns it into a variance.

The noise term doesn't depend on $b$. No baseline can remove the noise in the
rewards themselves. What the baseline removes is the part of the variance
that comes from the arms' means being far from $b$.

```python
score_sq = ((np.eye(k) - pi) ** 2).sum(axis=1)       # |e_a - pi|^2
second = np.sum(pi * ((means - b) ** 2 + sigma**2) * score_sq)
var = second - g @ g
```

## Find the best constant baseline

$\operatorname{Var}(b)$ is a quadratic in $b$ that opens upward. Set its
derivative to zero:

$$
b^* = \frac{\sum_a \pi(a) \lVert e_a - \pi \rVert^2\, m_a}{\sum_a \pi(a)
\lVert e_a - \pi \rVert^2}
$$

Read it as: a weighted average of the arm means. Each arm's weight is its
probability times its score's squared length. Compare it with the obvious
choice, the expected reward $V = \sum_a \pi(a) m_a$, which weights by
probability alone. The optimal baseline leans toward the arms whose scores
are long, because those are the samples that carry the most variance.
Greensmith, Bartlett, and Baxter (2004) derived optimal baselines in this
form and showed that the expected reward can be a poor substitute.

### Worked example

Take three arms with $\pi = [0.25, 0.25, 0.5]$, means $m = [1, 0, 2]$, and
noise $\sigma = 1$. The expected reward is $V = 0.25 + 0 + 1 = 1.25$. The
exact gradient is $\pi \odot (m - V) = [-0.0625, -0.3125, 0.375]$.

The score lengths are $\lVert e_a - \pi \rVert^2 = [0.875, 0.875, 0.375]$.
For arm 0, that's $0.75^2 + 0.25^2 + 0.5^2$. The weights
$\pi(a) \lVert e_a - \pi \rVert^2$ are $[0.21875, 0.21875, 0.1875]$, which sum
to $0.625$. So

$$
b^* = \frac{0.21875 \times 1 + 0.21875 \times 0 + 0.1875 \times 2}{0.625} = 0.95
$$

The following table shows the total variance for three baselines:

| Baseline | $b$ | Total variance |
|---|---|---|
| None | 0 | 1.3516 |
| Expected reward | 1.25 | 0.8438 |
| Optimal | 0.95 | 0.7875 |

The expected reward removes most of what a baseline can remove, and $b^*$
removes a little more. In practice people use a state value $V(s)$, because it's nearly as
good and it's easy to learn.

> [!KEY]
> Any baseline that doesn't depend on the action leaves the gradient
> unbiased, because the score averages to zero. A good baseline removes the
> variance that comes from rewards being far from typical. It can't remove
> the rewards' own noise.

## Replace the return with a critic

A baseline $V(s_t)$ turns $G_t - V(s_t)$ into an estimate of the advantage.
That's better, but $G_t$ still sums every later reward, with all their noise.
An actor-critic goes further. It keeps a learned value table or network
$V_\phi$, the critic, and replaces $G_t$ with a one-step guess
$r_t + \gamma V_\phi(s_{t+1})$. The actor's signal becomes the TD error:

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

Read it as: the reward you got, plus the discounted value of where you
landed, minus the value you expected. If $\delta_t > 0$, the step went better
than expected, so the action you took gets more likely. The same $\delta_t$
also trains the critic, by TD(0).

One step of a tabular actor-critic is four lines:

```python
delta = r + gamma * v[s2] - v[s]
v[s] += alpha_critic * delta                 # critic: TD(0)
score = -pi[s]; score[a] += 1.0
theta[s] += alpha_actor * delta * score      # actor: policy gradient step
```

### What a perfect critic gives you

If the critic is exact, $V_\phi = V^\pi$, then the average TD error after
taking $a$ in $s$ is exactly the advantage:

$$
\mathbb{E}[\delta_t \mid s_t = s, a_t = a] = R(s, a) + \gamma \sum_{s'}
P(s' \mid s, a) V^\pi(s') - V^\pi(s) = A^\pi(s, a)
$$

Read it as: the first two terms are $Q^\pi(s, a)$ by definition, and
subtracting $V^\pi(s)$ leaves the advantage. So $\delta_t$ is a one-sample,
low-noise estimate of $A^\pi$. In numpy the expected TD error for every state
and action is one line: `mdp.R + gamma * mdp.P @ v - v[:, None]`.

### What an imperfect critic costs you

Write the critic's error as $e = V_\phi - V^\pi$. Then the average TD error
is $A^\pi(s, a) + \gamma \sum_{s'} P(s' \mid s, a) e(s') - e(s)$. The two
error terms behave very differently:

- $-e(s)$ depends only on the current state. It acts as a baseline, so it
  adds variance but no bias.
- $\gamma \sum_{s'} P(s' \mid s, a) e(s')$ depends on the action, through
  where the action leads. That term biases the actor.

A critic that is off by the same constant everywhere does no harm at all.
Its error at the next state is the same for every action, so it cancels too.
A critic whose errors differ from state to state pushes the actor toward
actions that lead to overvalued states.

> [!KEY]
> An actor-critic trades variance for bias. With an exact critic, the TD
> error averages to the advantage. With an inexact one, its error at the
> current state is a harmless baseline, but its error at the next state
> biases the update.

### Worked example

Go back to the two-state MDP from *The policy gradient theorem*: stay or
switch, staying in state 1 pays 1, $\gamma = 0.9$, and the policy is uniform.
The true values are $V^\pi = [2.25, 2.75]$. The following table shows the
average TD error, centred within each state, for three critics:

| Critic $v$ | State 0, stay | State 0, switch | State 1, stay | State 1, switch |
|---|---|---|---|---|
| $V^\pi = [2.25, 2.75]$ | $-0.225$ | $+0.225$ | $+0.725$ | $-0.725$ |
| $V^\pi + 3 = [5.25, 5.75]$ | $-0.225$ | $+0.225$ | $+0.725$ | $-0.725$ |
| $V^\pi + [0, 1] = [2.25, 3.75]$ | $-0.675$ | $+0.675$ | $+1.175$ | $-1.175$ |

The constant shift changes nothing. Overvaluing state 1 by 1 inflates every
push toward state 1 by $0.9 \times 1 / 2 = 0.45$. The direction happens to
stay right here, but the size is wrong, and on other problems the sign can
flip.

### Which states the actor learns from

The policy gradient theorem weights states by the discounted visitation
$d^\pi$. A single long trajectory visits states in its long-run proportions
instead, which ignore $\mu_0$ and $\gamma$. There's a tidy fix. After every
step, restart from a fresh draw of $\mu_0$ with probability $1 - \gamma$, and
otherwise continue from $s_{t+1}$. The long-run state distribution $\rho$ of
that process solves $\rho = (1 - \gamma) \mu_0 + \gamma \rho P_\pi$, whose
solution is exactly the normalised visitation from `rlenvs.state_distribution`.
The restart only chooses which state you learn from next. The TD target
still uses the true next state.

With the restart and an exact critic, the average actor update per step is
$(1 - \gamma) \nabla J$. For component $b$ of state $s$, it's

$$
\bar u(s, b) = \rho(s)\, \pi(b \mid s) \Big(\bar\delta(s, b) - \sum_a
\pi(a \mid s)\, \bar\delta(s, a)\Big)
$$

Read it as: the visitation of $s$, times the probability of $b$, times how
much $b$'s average TD error $\bar\delta$ beats the policy's average in that
state. With the exact critic, the bracket is $A^\pi(s, b)$. Most deep
implementations run ordinary episodes and drop this weighting, which carries
the same caveat as dropping $\gamma^t$ in REINFORCE.

> [!TIP]
> Give the critic a larger step size than the actor. The actor's signal is
> only as good as the critic, so the critic has to keep up with the policy it
> evaluates.

> [!RECAP]
> - A baseline $b$ scales the score by $R - b$ and leaves the mean unchanged.
> - Total variance is $\sum_a \pi(a)[(m_a - b)^2 + \sigma^2] \lVert e_a - \pi
>   \rVert^2 - \lVert \nabla J \rVert^2$.
> - $b^*$ is the average of $m_a$ weighted by $\pi(a) \lVert e_a - \pi
>   \rVert^2$.
> - The expected TD error is `R + gamma * P @ v - v[:, None]`, and it equals
>   $A^\pi$ when $v = V^\pi$.
> - Restarting with probability $1 - \gamma$ makes the visited states follow
>   the discounted visitation.

## Build it step by step

The checks test the functions in this order. `softmax` is given.

1. **`bandit_gradient_samples(theta, arms, rewards, baseline)`.** Takes `(k,)`
   logits and `(N,)` arms and rewards. Returns `(N, k)`. Build all scores at
   once with `np.eye(k)[arms] - pi`. Common mistake: multiplying by `rewards`
   and forgetting to subtract the baseline.
2. **`exact_bandit_gradient(theta, means)`.** Returns `(k,)`:
   `pi * (means - pi @ means)`.
3. The third check reuses your samples with four different baselines and
   requires every mean to be within four standard errors of the exact
   gradient. There's nothing new to write.
4. **`estimator_variance(theta, means, sigma, baseline)`.** Returns a float.
   Common mistake: returning the second moment without subtracting
   $\lVert \nabla J \rVert^2$. The check compares your value with the
   variance of 200,000 samples.
5. **`optimal_baseline(theta, means)`.** Returns a float. Compute the weights
   `pi * score_sq` and take the weighted average of `means`. It doesn't depend
   on `sigma`.
6. **`expected_td_error(mdp, v)`.** Returns `(S, A)`. Common mistake:
   subtracting `v` without `[:, None]`, which broadcasts over the wrong axis
   when `S == A`.
7. **`expected_actor_update(mdp, theta, v)`.** Returns `(S, A)`. Centre the TD
   errors within each state, multiply by `pi`, then by
   `rlenvs.state_distribution(mdp, pi)[:, None]`. The check compares it with
   $(1 - \gamma)$ times finite differences of $J$.
8. **`actor_critic(mdp, steps, alpha_actor, alpha_critic, seed)`.** Returns
   `(theta, v)`. Use one `rng = np.random.default_rng(seed)` for every draw,
   and `mdp.sample(s, a, rng)` for the transition. Common mistake: updating
   `v[s2]` instead of `v[s]`. The checks train the critic alone for 20,000
   steps, then the pair for 5,000 steps.

## Cost and scaling

A tabular actor-critic costs a constant amount of work per step, and it
learns online, so memory doesn't grow with the episode. The following table
compares it with REINFORCE:

| Method | Work per step | Memory | Learns when? |
|---|---|---|---|
| REINFORCE | $O(A)$ for the score | Whole episode, $O(T)$ | At the end of each episode |
| REINFORCE with $V(s)$ baseline | $O(A)$ plus a critic update | $O(T)$ | At the end of each episode |
| One-step actor-critic | $O(A)$ plus a critic update | $O(1)$ | After every step |

With neural networks, every step costs a forward and a backward pass through
both the actor and the critic. Mnih et al. (2016) ran many copies of the
environment in parallel, which decorrelates the steps that each update sees.
A later synchronous variant, A2C, batches those steps on one device.

## Where this breaks

**A bad critic misleads the actor.** Early in training, the critic knows
nothing, and the actor follows its errors. People warm the critic up first,
or use multi-step targets that lean less on it. The next lesson,
*Generalised advantage estimation*, makes that dial explicit.

**Two learning rates to tune.** If the actor moves faster than the critic can
track, the TD errors describe a policy that no longer exists. Konda and
Tsitsiklis (2000) proved convergence when the critic runs on a faster
timescale than the actor. In practice, people tune the ratio.

**Action-dependent baselines.** A baseline $b(s, a)$ can cut variance
further, but it biases the gradient unless you add a correction term. Tucker
et al. (2018) found that learned action-dependent baselines didn't reduce
variance over a state baseline on common benchmarks, and that the reported
gains came from implementation details.

**Gradients through the target.** In a deep actor-critic, the TD target
$r + \gamma V_\phi(s')$ must be treated as a constant when you train the
critic, and $\delta$ must be detached in the actor's loss. Otherwise the
actor's loss trains the critic, and the critic's loss chases a moving target.

**Premature certainty.** The actor can make one action nearly certain before
the critic is accurate, and then it never samples the alternatives. Deep
implementations add an entropy bonus to the actor's objective to slow this
down. *Proximal policy optimisation* includes one.

## Further reading

- [Greensmith, Bartlett, and Baxter (2004), Variance reduction techniques for gradient estimates in reinforcement learning](https://www.jmlr.org/papers/v5/greensmith04a.html)
  derives optimal baselines and analyses what actor-critic estimates cost in
  bias and variance.
- [Konda and Tsitsiklis (2000), Actor-critic algorithms](https://proceedings.neurips.cc/paper/1999/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
  proves convergence for actor-critic methods with a faster critic.
- [Sutton and Barto (2018), Reinforcement Learning: An Introduction, chapter 13](http://incompleteideas.net/book/the-book-2nd.html)
  covers REINFORCE with a baseline and the one-step actor-critic.
- [Mnih et al. (2016), Asynchronous methods for deep reinforcement learning](https://arxiv.org/abs/1602.01783)
  introduces A3C, the parallel deep actor-critic with an entropy bonus.
- [Tucker et al. (2018), The mirage of action-dependent baselines in reinforcement learning](https://arxiv.org/abs/1802.10031)
  shows that learned action-dependent baselines don't beat a state baseline
  on common benchmarks.

## The exercise

You build both halves of this lesson. For the bandit, write
`bandit_gradient_samples`, `exact_bandit_gradient`, the exact variance
`estimator_variance`, and `optimal_baseline`. For the MDP, write
`expected_td_error`, `expected_actor_update`, and the online learner
`actor_critic`.

The checks verify the per-sample estimates and $b^* = 0.95$ against the
worked example. They confirm the gradient by finite differences and require
the sampled means with four different baselines to stay within four standard
errors of it. They compare your exact variance with the spread of 200,000
samples, and they check that the variance has zero slope at $b^*$. On a
random MDP they check that the expected TD error equals $A^\pi$. They also
check that the expected actor update equals $(1 - \gamma) \nabla J$, both for
the exact critic and for one shifted by a constant. Finally, `actor_critic`
must learn $V^\pi$ with the actor frozen, then close at least 90% of the gap
between the uniform policy and the optimum in 5,000 steps.
