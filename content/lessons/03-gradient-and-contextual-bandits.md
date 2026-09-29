---
title: Gradient and contextual bandits
slug: 03-gradient-and-contextual-bandits
part: "Part 1 — The reinforcement learning problem"
summary: Learn arm preferences by gradient ascent with a baseline, then pick arms by context with LinUCB's confidence widths.
minutes: 55
gpu: false
objectives:
  - Update softmax preferences with the gradient bandit rule, and show its expected update is the exact gradient.
  - Explain why a baseline leaves the expected update unchanged but speeds up learning.
  - Fit per-arm ridge regressions, and score arms with LinUCB's prediction plus confidence width.
  - Update an inverse matrix after one new data point with the Sherman-Morrison formula.
prereqs:
  - 02-multi-armed-bandits
exercise: 03-gradient-and-contextual-bandits
---

# Gradient and contextual bandits

The bandit strategies in the previous lesson all estimate how good each arm
is, then act on those estimates. This lesson takes two different turns, and
each one leads straight to a big idea later in the course.

The first turn skips value estimates. It keeps a *preference* for each arm,
turns preferences into probabilities, and nudges them uphill on expected
reward. That is gradient ascent on the policy itself. It's the one-state
version of the policy gradient methods that, many lessons later, post-train
language models.

The second turn gives the agent something to look at before it chooses. A
news site picking a headline sees who the reader is. A doctor picking a
treatment sees the patient's chart. The best arm now depends on this
*context*. LinUCB handles it by fitting a small linear regression per arm and
adding UCB's optimism, measured as a confidence width in feature space.

## What you already know

- A softmax turns a vector of scores into probabilities that sum to one.
- Gradient ascent moves parameters a small step along the gradient to
  increase an objective.
- Ridge regression fits weights by solving $(X^\top X + \lambda I)\,\theta = X^\top y$.
- UCB1 picks the arm with the largest estimate plus an uncertainty bonus.
- Regret adds up the gap between the best arm's mean and the pulled arm's mean.

## Preferences and a softmax policy

Give each arm $a$ a number $H(a)$, its preference. The policy picks arms with
probabilities from a softmax:

$$
\pi(a) = \frac{e^{H(a)}}{\sum_b e^{H(b)}}
$$

Read it as: exponentiate each preference, then divide by the total so the
probabilities add up to one. Only differences between preferences matter.
Adding 50 to every $H(a)$ multiplies every term by $e^{50}$, which cancels. So
subtract the largest preference before exponentiating, and nothing overflows:

```python
z = np.exp(H - H.max())
pi = z / z.sum()
```

## The gradient bandit update

The goal is a large expected reward,
$J(H) = \sum_x \pi(x)\, q(x)$, where $q(x)$ is arm $x$'s unknown mean reward.
You can't compute $J$, because you don't know $q$. But each pull gives one
noisy reward, and that turns out to be enough.

After pulling arm $a_t$ and receiving reward $r_t$, update every preference:

$$
H(a) \leftarrow H(a) + \alpha\,(r_t - b)\,\bigl(\mathbf{1}\{a = a_t\} - \pi(a)\bigr)
$$

Read it piece by piece:

- $\alpha$ is the step size.
- $b$ is a *baseline*, a reference level for rewards. A reward above it counts
  as good news, and one below it as bad news.
- $\mathbf{1}\{a = a_t\}$ is 1 for the arm you pulled and 0 for every other.
- For the pulled arm, the last factor is $1 - \pi(a_t)$, which is positive. Good
  news raises its preference. For every other arm, the factor is $-\pi(a)$, so
  good news lowers them, each in proportion to its probability.

In numpy, the whole update is one line:

```python
H = H + alpha * (reward - baseline) * (np.eye(k)[arm] - pi)
```

Sutton and Barto (Section 2.8) use as the baseline the average of the rewards
received *before* the current step. This lesson does the same, starting at 0.

### Worked example

Three arms start at $H = (0, 0, 0)$, so $\pi = (1/3, 1/3, 1/3)$. You pull arm
0 and receive $r = 1$. The baseline is 0 and $\alpha = 0.1$:

- Arm 0: $0 + 0.1 \times 1 \times (1 - 1/3) = 0.066667$.
- Arms 1 and 2: $0 + 0.1 \times 1 \times (0 - 1/3) = -0.033333$.

The new policy is $\text{softmax}(0.066667, -0.033333, -0.033333) = (0.355913, 0.322043, 0.322043)$.
One good reward shifted about 2 percentage points toward arm 0.

## Why the update climbs the true gradient

The update uses one random reward, yet on average it points exactly along the
gradient of $J$. Here's the argument in three steps.

First, the gradient of the softmax. Differentiating $\pi(x)$ with respect to
$H(a)$ gives $\pi(x)\,(\mathbf{1}\{a = x\} - \pi(a))$. So the exact gradient of
expected reward is:

$$
\frac{\partial J}{\partial H(a)} = \sum_x q(x)\, \pi(x)\, \bigl(\mathbf{1}\{a = x\} - \pi(a)\bigr) = \pi(a)\,\bigl(q(a) - J\bigr)
$$

Read it as: raise the preference of an arm that pays more than the policy's
current average, lower it for an arm that pays less, and scale by how often
the arm is pulled. In numpy: `pi * (q - pi @ q)`.

Second, the expected update. Arm $x$ is pulled with probability $\pi(x)$ and
pays $q(x)$ on average, so the average of $(r_t - b)(\mathbf{1}\{a = a_t\} - \pi(a))$
is $\sum_x \pi(x)\,(q(x) - b)\,(\mathbf{1}\{a = x\} - \pi(a))$. The part with
$q(x)$ is the exact gradient.

Third, the baseline's part vanishes. It is $b$ times
$\sum_x \pi(x)(\mathbf{1}\{a = x\} - \pi(a)) = \pi(a) - \pi(a) = 0$. Any
baseline that doesn't depend on the current pull leaves the expected update
unchanged. The update is an *unbiased* estimate of the gradient, which makes
the gradient bandit *stochastic gradient ascent*.

For example, with $H = (0, 0, 0)$ and $q = (1, 2, 3)$, the policy's average is
$J = 2$, and the exact gradient is $\tfrac13 (1 - 2, 2 - 2, 3 - 2) = (-1/3, 0, 1/3)$.

> [!KEY]
> The gradient bandit update is, on average, the exact gradient of expected
> reward. A baseline changes nothing on average, because the update's
> direction $\mathbf{1}\{a = a_t\} - \pi(a)$ averages to zero.

### So why use a baseline

If the baseline changes nothing on average, it must change something else: the
noise. Suppose every arm pays around $+4$. With no baseline, $r_t - b$ is
always near $+4$, so every pull raises the pulled arm's preference by a lot,
whether that arm is good or bad. Early lucky arms get pulled more, and pulled
more means raised more. The signal that tells arms apart is the small
difference between rewards, buried under the shared $+4$. Subtracting the
running average removes the shared part and leaves the difference.

Sutton and Barto's Figure 2.5 shows this on ten arms whose means are drawn
around $+4$. The exercise reproduces it. On one such bandit, averaged over 20
runs of 1000 steps with $\alpha = 0.1$, the reference solution pulls the best
arm 98.6 percent of the time in the last 500 steps with a baseline, and 59.2
percent without one.

> [!WARNING]
> Update the running mean of rewards *after* you use it as the baseline. If the
> baseline already includes the current reward, it depends on which arm you
> pulled, and the unbiasedness argument no longer holds exactly.

## Contextual bandits

In a *contextual bandit*, each round starts with a context vector
$x \in \mathbb{R}^d$ that you observe before choosing. The mean reward of arm
$a$ depends on it. The simplest useful model is linear, with one weight vector
$\theta_a$ per arm:

$$
\mathbb{E}[r \mid x, a] = \theta_a^\top x
$$

Read it as: each arm scores the context with its own weights. Different
contexts favour different arms, so no single arm is best. In the exercise,
$d = 4$, there are three arms, and each context is a leading 1, for an
intercept, followed by three standard-normal features. The best arm that
ignores the context loses about 0.61 reward per round against the best
choice for each context. Over 2000 rounds, that's about 1220 regret, and no
strategy that ignores the context can expect to do better.

### Fit each arm with ridge regression

For each arm, collect the contexts from the rounds you pulled it, and fit its
weights by ridge regression. Keep two running sums per arm:

$$
A_a = I + \sum_{t:\, a_t = a} x_t x_t^\top, \qquad b_a = \sum_{t:\, a_t = a} r_t\, x_t, \qquad \hat\theta_a = A_a^{-1} b_a
$$

Read it as: $A_a$ is the identity plus the outer products of that arm's past
contexts, $b_a$ adds up each past context times its reward, and the estimate
$\hat\theta_a$ solves the ridge regression with penalty $\lambda = 1$. That's
the same $\hat\theta = (X^\top X + I)^{-1} X^\top y$ you know, with the rows of
$X$ and $y$ added one at a time.

### Score each arm with LinUCB

LinUCB (Li, Chu, Langford, and Schapire, 2010) adds optimism, as UCB1 does. The
bonus is large in directions of feature space where this arm has seen little
data:

$$
\text{score}(a) = \hat\theta_a^\top x + \alpha \sqrt{x^\top A_a^{-1} x}
$$

Read it as: the predicted reward for this context, plus $\alpha$ times a
*confidence width*. The width $\sqrt{x^\top A_a^{-1} x}$ is small when many
past contexts pointed the same way as $x$, because those contexts made $A_a$
large in that direction. Under a Bayesian reading, with unit-variance noise and
a standard-normal prior on $\theta_a$, $A_a^{-1}$ is exactly the posterior
covariance of $\theta_a$, so the width is the posterior standard deviation of
the prediction. The multiplier $\alpha$ here is LinUCB's exploration strength,
not a step size.

```python
theta_hat = A_inv @ b
mean = theta_hat @ x
width = np.sqrt(x @ A_inv @ x)
score = mean + alpha * width
```

### Worked example

One arm with $d = 2$ has seen context $(1, 0)$ with reward 1 and context
$(1, 1)$ with reward 0. Then:

$$
A = I + \begin{bmatrix}1&0\\0&0\end{bmatrix} + \begin{bmatrix}1&1\\1&1\end{bmatrix} = \begin{bmatrix}3&1\\1&2\end{bmatrix}, \qquad A^{-1} = \begin{bmatrix}0.4&-0.2\\-0.2&0.6\end{bmatrix}
$$

With $b = (1, 0)$, the estimate is $\hat\theta = A^{-1} b = (0.4, -0.2)$. For a
new context $x = (1, 0.5)$:

- Prediction: $0.4 \times 1 - 0.2 \times 0.5 = 0.3$.
- Width: $x^\top A^{-1} x = 0.4 - 0.2 + 0.15 = 0.35$, so the width is
  $\sqrt{0.35} = 0.5916$.

### Update the inverse without inverting

Solving with $A_a$ from scratch every round costs time proportional to $d^3$.
Each round adds one outer product, and the Sherman-Morrison formula gives the
new inverse directly from the old one:

$$
\left(A + x x^\top\right)^{-1} = A^{-1} - \frac{u\, u^\top}{1 + x^\top u}, \qquad u = A^{-1} x
$$

Read it as: compute $u$, the old inverse applied to the new context, and
subtract a rank-one correction. That costs time proportional to $d^2$. In the
worked example, adding $x = (1, 0.5)$ gives $u = (0.3, 0.1)$ and
$x^\top u = 0.35$. The new inverse is
$A^{-1} - u u^\top / 1.35$, with entries $0.3333$, $-0.2222$, $-0.2222$, and
$0.5926$, which matches `np.linalg.inv` of $A + x x^\top$. The width for that
same context shrinks from $0.5916$ to $\sqrt{0.35 / 1.35} = 0.5092$.

> [!KEY]
> LinUCB is UCB1 in feature space. The prediction comes from a ridge
> regression per arm, and the bonus is the width of its confidence region in
> the direction of the current context.

> [!RECAP]
> - `softmax` subtracts the max, and the gradient bandit step is
>   `alpha * (r - b) * (np.eye(k)[arm] - pi)`.
> - The expected update over the policy's own arm choices equals
>   `pi * (q - pi @ q)` for every baseline.
> - The baseline is the mean of earlier rewards, updated after each use.
> - LinUCB's score is `theta_hat @ x + alpha * sqrt(x @ A_inv @ x)`, with
>   `theta_hat = A_inv @ b`.
> - Sherman-Morrison updates `A_inv` with `u = A_inv @ x`.

## Build it step by step

The checks test the functions in this order.

1. **`softmax(H)`.** Takes `(k,)` preferences and returns `(k,)`
   probabilities. Common mistake: skipping the max subtraction. The check
   passes preferences of 1000.
2. **`gradient_bandit_update(H, arm, reward, baseline, alpha)`.** Returns new
   preferences, leaving `H` unchanged. Compute `pi` from the old `H` first.
   Common mistake: updating only the pulled arm and forgetting the $-\pi(a)$
   push on the rest.
3. **`expected_update(H, q, baseline)`.** Loop over arms $x$, call your
   update with `alpha=1.0` and reward `q[x]`, subtract `H`, weight by
   `pi[x]`, and sum. The check compares against finite differences of
   $J(H) = \pi \cdot q$ for three baselines, so any error in step 2 shows here.
4. **`run_gradient_bandit(bandit, T, alpha, use_baseline, seed)`.** Draw arms
   with `rng.choice(k, p=softmax(H))`. Keep the running mean reward, use it as
   the baseline, then fold in the new reward. Return the `(T,)` array of arms.
   The check runs 20 seeds on means near $+4$ and expects the baseline version
   to win by a wide margin.
5. **`sherman_morrison(A_inv, x)`.** Two lines: `u = A_inv @ x`, then the
   rank-one correction with `np.outer(u, u)`. Common mistake: dividing by
   `x @ u` without the `1 +`.
6. **`linucb_terms(A_inv, b, x)`.** Returns `(mean, width)`. The checks use
   the worked example, a ridge fit on 30 random rows, and a context seen $n$
   times, whose width must equal $\sqrt{1 / (1 + n)}$.
7. **`run_linucb(bandit, T, alpha)`.** Arrays `A_inv` of shape `(k, d, d)`,
   starting as identities, and `b` of shape `(k, d)`, starting at zero. Each
   round: `x = bandit.observe()`, score every arm, pull the best, update that
   arm only, and add `bandit.regret(arm)`. Common mistake: sharing one `A_inv`
   across arms.

## Cost and scaling

The gradient bandit costs $O(k)$ per step for the softmax and the update. LinUCB
does more linear algebra, and it grows with the feature dimension $d$.

| Operation | Time per round | Memory |
|---|---|---|
| Gradient bandit step | $O(k)$ | $k$ preferences |
| LinUCB scores, all arms | $O(k d^2)$ | $k$ matrices of $d^2$ entries |
| Ridge update by re-solving | $O(d^3)$ | Same |
| Ridge update by Sherman-Morrison | $O(d^2)$ | Same |

With $d$ in the thousands, even $d^2$ per arm hurts. Large systems use a
diagonal approximation of $A$, or share one model across arms by putting the
arm into the features.

## Where this breaks

**No baseline, shifted rewards.** Without a baseline, the gradient bandit's
behaviour depends on where zero sits. Shift every reward by a constant, and
learning speeds up or stalls. Always subtract a baseline.

**A step size that's too large.** The gradient bandit can drive one
preference far above the rest in a few lucky steps, and softmax then almost
never tries another arm. Smaller steps, or an entropy bonus that rewards
spread-out policies, keep it exploring.

**A model that isn't linear.** LinUCB's widths are only honest when rewards
really are linear in the features. When they aren't, the regression is
confidently wrong, and the bonus doesn't fix it. Richer features help, and so
do neural networks with approximate uncertainty.

**Choosing alpha.** The theory behind LinUCB gives a width multiplier that
depends on the noise level, the dimension, and how confident you want to be,
but the constants are loose. In practice, people tune $\alpha$ on logged
data.

## Further reading

- [Sutton and Barto, *Reinforcement Learning: An Introduction*, Sections 2.8 and 2.9](http://incompleteideas.net/book/the-book-2nd.html)
  derive the gradient bandit as stochastic gradient ascent and introduce
  contextual bandits under the name *associative search*.
- [Williams, "Simple statistical gradient-following algorithms for connectionist reinforcement learning" (1992)](https://doi.org/10.1007/BF00992696)
  introduces REINFORCE, which the gradient bandit is the one-state case of.
- [Li, Chu, Langford, and Schapire, "A contextual-bandit approach to personalized news article recommendation" (2010)](https://arxiv.org/abs/1003.0146)
  introduces LinUCB and tests it on Yahoo! front-page news.
- [Abbasi-Yadkori, Pál, and Szepesvári, "Improved algorithms for linear stochastic bandits" (2011)](https://papers.nips.cc/paper/2011/hash/e1d5be1c7f2f456670de3d53c7b54f4a-Abstract.html)
  proves a tight confidence ellipsoid for the ridge estimate and regret bounds
  for the resulting algorithm, OFUL.
- [Lattimore and Szepesvári, *Bandit Algorithms* (2020), Chapters 19 and 20](https://tor-lattimore.com/downloads/book/book.pdf)
  work through linear bandits and their confidence bounds in full.

## The exercise

You build both halves of this lesson. For the gradient bandit, write
`softmax`, `gradient_bandit_update`, `expected_update`, and
`run_gradient_bandit`. For LinUCB, write `sherman_morrison`, `linucb_terms`,
and `run_linucb`. The environment `LinearContextualBandit` is given in the
starter file.

The checks verify the worked gradient bandit step and the policy it produces,
and that `expected_update` equals a finite-difference gradient of expected
reward for three different baselines. They check that a baseline wins by a
wide margin on arms whose means sit near $+4$. Then they check
`sherman_morrison` against `np.linalg.inv` after twenty updates, and
`linucb_terms` against the worked example, a ridge fit, and the closed-form
width $\sqrt{1/(1+n)}$. Last, `run_linucb` must end 2000 rounds with less than a
tenth of the best context-blind arm's regret, and with much less regret in the
second half than in the first.
