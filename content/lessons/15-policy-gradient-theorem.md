---
title: The policy gradient theorem
slug: 15-policy-gradient-theorem
part: "Part 4 — Policy optimisation"
summary: Differentiate a policy's expected return exactly, then estimate that same gradient from sampled episodes with REINFORCE.
minutes: 55
gpu: false
objectives:
  - Compute the score of a tabular softmax policy and explain why its expectation over actions is zero.
  - State the policy gradient theorem and evaluate it exactly with rlenvs.policy_evaluation and rlenvs.state_distribution.
  - Confirm the exact gradient against finite differences of J on a random MDP.
  - Implement REINFORCE and test that its mean matches the exact gradient within four standard errors.
prereqs:
  - 03-gradient-and-contextual-bandits
  - 04-markov-decision-processes
  - 06-monte-carlo-methods
exercise: 15-policy-gradient-theorem
---

# The policy gradient theorem

Every method in Part 3 learned values and then acted greedily on them. The
policy was an afterthought, read off the value table with an `argmax`. That
works well when actions are few and the values are accurate. It struggles
when the best behaviour is random, when actions are continuous, or when a
small change in a value flips the greedy action and the agent's behaviour
jumps.

This part of the course turns the problem around. You write the policy as a
function with parameters, and you climb the expected return directly by
gradient ascent. The obstacle is that the return depends on the parameters
through random sampling. You can't backpropagate through "pick an action at
random, then let the world respond."

The policy gradient theorem removes that obstacle. It says the gradient of
the expected return is an average you can estimate from episodes, and it
needs no model of the world. In this lesson you compute that gradient
exactly on a tiny MDP, confirm it with finite differences, and then show
that REINFORCE, the simplest sampling method, gets the same answer on
average.

## What you already know

- A *policy* $\pi(a \mid s)$ gives the probability of taking action $a$ in
  state $s$. In code it's an `(S, A)` array whose rows sum to one.
- The *return* $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$ is the
  discounted sum of rewards from step $t$ onward.
- $V^\pi(s)$ is the expected return from $s$, and $Q^\pi(s, a)$ is the
  expected return after taking $a$ in $s$. `rlenvs.policy_evaluation` and
  `rlenvs.q_from_v` compute both exactly.
- The *discounted visitation* weights each state by how often, and how early,
  the policy visits it. You met it in
  *Markov decision processes*.
- The gradient bandit from
  *Gradient and contextual bandits*
  nudged softmax preferences up for actions that paid more than a baseline.

## Write the policy as a softmax

Give every state its own row of scores, called *logits*. The policy turns
each row into probabilities with a softmax. The parameters are the whole
table $\theta$, of shape `(S, A)`:

$$
\pi_\theta(a \mid s) = \frac{e^{\theta_{s,a}}}{\sum_{b} e^{\theta_{s,b}}}
$$

Read it as: exponentiate the scores in row $s$, then divide by their total so
they sum to one. A higher score means a likelier action. Adding the same
number to a whole row changes nothing, because it cancels in the ratio.

```python
z = theta - theta.max(axis=1, keepdims=True)   # same softmax, no overflow
pi = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
```

The quantity you want to raise is the *objective*, the expected return from
a start state drawn from $\mu_0$:

$$
J(\theta) = \sum_s \mu_0(s)\, V^{\pi_\theta}(s)
$$

Read it as: the value of each start state, weighted by how likely you are to
start there. On a small MDP you can compute it exactly with one linear solve:
`mdp.mu0 @ rlenvs.policy_evaluation(mdp, pi)`.

## Differentiate through a random choice

Here's the trick that makes everything work. Suppose $x$ is drawn from a
distribution $p_\theta(x)$, and you want the gradient of the average of some
score $f(x)$. The gradient of a probability equals the probability times the
gradient of its log, because $\nabla \log p = \nabla p / p$. So

$$
\nabla_\theta\, \mathbb{E}_{x \sim p_\theta}\big[f(x)\big]
= \mathbb{E}_{x \sim p_\theta}\big[f(x)\, \nabla_\theta \log p_\theta(x)\big]
$$

Read it as: to learn how the average of $f$ moves, sample $x$, and push up
its log-probability in proportion to how good $f(x)$ was. You never
differentiate $f$ itself. That's why the world can stay a black box.

The vector $\nabla_\theta \log \pi_\theta(a \mid s)$ is called the *score*.
For the tabular softmax it has a clean form. Only row $s$ is nonzero, and in
that row the score is the one-hot vector for $a$ minus the probabilities:

$$
\frac{\partial \log \pi_\theta(a \mid s)}{\partial \theta_{s,b}}
= \mathbb{1}[a = b] - \pi_\theta(b \mid s)
$$

Read it as: raising the logit of the action you took makes it likelier, and
every other action in that state gives up a share in proportion to its
probability. In numpy:

```python
grad = np.zeros_like(pi)
grad[s] = -pi[s]
grad[s, a] += 1.0
```

The score has one property you use again and again. Averaged over actions
drawn from the policy, it's zero:
$\sum_a \pi(a \mid s) \nabla \log \pi(a \mid s) = \nabla \sum_a \pi(a \mid s)
= \nabla 1 = 0$. The next lesson builds baselines on this fact.

> [!KEY]
> The log-derivative trick turns the gradient of an expectation into the
> expectation of "how good was it" times the score
> $\nabla \log \pi(a \mid s)$. You can estimate that from samples without
> knowing how the world works.

## State the policy gradient theorem

An episode is a chain of random choices: the policy picks actions, and the
world picks next states. The world's probabilities don't depend on $\theta$,
so their logs have zero gradient. Only the policy's terms survive. Sutton,
McAllester, Singh, and Mansour (2000) collected the result into one formula.
First, define the *unnormalised discounted visitation*
$d^\pi(s) = \sum_{t \ge 0} \gamma^t \Pr(s_t = s)$. It's the expected
discounted number of visits to $s$. Then

$$
\nabla_\theta J(\theta) = \sum_s d^\pi(s) \sum_a \pi_\theta(a \mid s)\,
Q^\pi(s, a)\, \nabla_\theta \log \pi_\theta(a \mid s)
$$

Read it from the inside out:

- $Q^\pi(s, a)\, \nabla \log \pi(a \mid s)$ pushes on action $a$ in
  proportion to how good it is.
- The sum over $a$ weighted by $\pi$ averages that push over the actions the
  policy takes.
- The outer sum weights each state by $d^\pi(s)$, so states you visit often
  and early count most.

Why is it true? Write $V(s) = \sum_a \pi(a \mid s) Q(s, a)$ and apply the
product rule. One term differentiates $\pi$ and holds $Q$ fixed. The other
differentiates $Q$, which leads to $\gamma$ times the gradient of $V$ at the
next state. Unrolling that recursion visits every future state $s_t$ with
weight $\gamma^t$, and adding the weights gives $d^\pi$.

For the tabular softmax, put the score formula into the theorem. Only row $s$
of the score is nonzero, and the sum over actions collapses:
$\sum_{a'} \pi(a' \mid s) Q(s, a') (\mathbb{1}[a' = a] - \pi(a \mid s)) =
\pi(a \mid s)(Q(s, a) - V(s))$. So

$$
\frac{\partial J}{\partial \theta_{s,a}} = d^\pi(s)\, \pi_\theta(a \mid s)\,
A^\pi(s, a)
$$

Read it as: the logit for $a$ in $s$ should rise when $a$ beats the state's
average, $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s) > 0$. It rises faster when you
visit $s$ often and when $a$ is already likely. The *advantage* $A^\pi$ says
how much better $a$ is than the policy's usual behaviour in $s$.

Every piece is one call to `rlenvs`:

```python
v = rlenvs.policy_evaluation(mdp, pi)
adv = rlenvs.q_from_v(mdp, v) - v[:, None]
d = rlenvs.state_distribution(mdp, pi) / (1 - mdp.gamma)
grad = d[:, None] * pi * adv
```

> [!WARNING]
> `rlenvs.state_distribution` returns the visitation *normalised* to sum to
> one. With no terminal states, the unnormalised $d^\pi$ sums to
> $1 / (1 - \gamma)$, so you divide by $1 - \gamma$. Forget that and your
> gradient is too small by a factor of ten when $\gamma = 0.9$.

### Worked example

Take two states and two actions, *stay* and *switch*. Switching moves you to
the other state, and staying keeps you where you are. Staying in state 1 pays
1, and every other choice pays 0. You start in state 0, $\gamma = 0.9$, and
$\theta = 0$, so the policy is uniform: $\pi = 0.5$ everywhere.

The following table shows each quantity, computed with `rlenvs` and checkable
by hand:

| Quantity | State 0 | State 1 |
|---|---|---|
| $V^\pi(s)$ | 2.25 | 2.75 |
| $Q^\pi(s, \text{stay})$ | $0 + 0.9 \times 2.25 = 2.025$ | $1 + 0.9 \times 2.75 = 3.475$ |
| $Q^\pi(s, \text{switch})$ | $0.9 \times 2.75 = 2.475$ | $0.9 \times 2.25 = 2.025$ |
| $A^\pi(s, \text{stay})$ | $-0.225$ | $+0.725$ |
| $d^\pi(s)$ | 5.5 | 4.5 |
| $\partial J / \partial \theta_{s, \text{stay}}$ | $5.5 \times 0.5 \times (-0.225) = -0.61875$ | $4.5 \times 0.5 \times 0.725 = +1.63125$ |

The switch column of the gradient is the negative of the stay column, since
each state's advantages average to zero. You can check $V$ from the Bellman
equation: $V(0) = 0.45 V(0) + 0.45 V(1)$ and $V(1) = 0.5 + 0.45 V(0) + 0.45
V(1)$. The visitation $d^\pi$ sums to $10 = 1 / (1 - 0.9)$, as it must. So
$J = V(0) = 2.25$.

The gradient says: in state 0, prefer switching, because it leads to the
state that pays. In state 1, prefer staying, and push harder there, because
the advantage is larger.

> [!TIP]
> Before trusting any gradient formula, compare it with central finite
> differences of $J$. `rlcheck.finite_difference` nudges each logit by
> $\pm 10^{-6}$ and re-solves for $J$. On a three-state MDP the two agree to
> about $10^{-9}$.

## Estimate the gradient from episodes

On a real problem you don't have $d^\pi$ or $Q^\pi$. You have episodes.
REINFORCE, from Williams (1992), replaces each unknown with a sample. The
state weighting $d^\pi$ becomes "the states you visited, each with weight
$\gamma^t$." The action value $Q^\pi(s_t, a_t)$ becomes the return $G_t$ you
actually got from step $t$ on. One episode gives the estimate

$$
\hat g = \sum_{t=0}^{T-1} \gamma^t\, G_t\, \nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

Read it as: for every step, take the score of the action you chose, scale it
by the discounted return that followed, and scale again by $\gamma^t$ for how
late the step came. Then add up the steps.

This estimate is *unbiased*: its average over many episodes equals the exact
gradient. The reason is that $G_t$ averages to $Q^\pi(s_t, a_t)$ given the
state and action, and the $\gamma^t$ weights average to $d^\pi$. Only the
rewards from step $t$ onward appear in $G_t$. An action can't change rewards
that came before it, so leaving them out removes noise and keeps the average
the same.

In numpy, for one episode:

```python
weights = gamma ** np.arange(T) * G              # G from discounted_returns
scores = -pi[states]
scores[np.arange(T), actions] += 1.0
np.add.at(grad, states, weights[:, None] * scores)
```

`np.add.at` matters here. Plain `grad[states] += ...` keeps only one update
when a state repeats.

### Worked example

In the two-state MDP with the uniform policy, suppose one episode goes
$s = [0, 1, 1]$, $a = [\text{switch}, \text{stay}, \text{switch}]$, with
rewards $r = [0, 1, 0]$. The returns are
$G = [0 + 0.9 \times 1, 1, 0] = [0.9, 1, 0]$. The weights
$\gamma^t G_t$ are $[0.9, 0.9, 0]$. Step 0 adds $0.9 \times [-0.5, 0.5]$ to row
0. Step 1 adds $0.9 \times [0.5, -0.5]$ to row 1. Step 2 adds nothing. The
estimate is

$$
\hat g = \begin{bmatrix} -0.45 & 0.45 \\ 0.45 & -0.45 \end{bmatrix}
$$

It points the right way in state 1 and the wrong way in state 0. One episode
is noisy. Averaged over thousands, the estimates converge to the exact
gradient from the table.

### Test for unbiasedness honestly

A single tolerance, like `atol=0.05`, is the wrong test. Some components of
the gradient are much noisier than others. Instead, compute each component's
*standard error*, the standard deviation of the per-episode estimates divided
by $\sqrt{N}$. Then check that the mean is within four standard errors of the
exact value. An unbiased estimator fails this test for a component about
once in 16,000 tries. A biased one fails it reliably once $N$ is large.

> [!KEY]
> REINFORCE is unbiased: the average of $\sum_t \gamma^t G_t \nabla \log
> \pi(a_t \mid s_t)$ over episodes equals $\nabla J$. It pays for that with
> high variance, which the next three lessons attack.

> [!RECAP]
> - The softmax score in row $s$ is `onehot(a) - pi[s]`, and it averages to
>   zero over actions.
> - $J(\theta)$ is `mu0 @ policy_evaluation(mdp, pi)`.
> - The tabular gradient is $d^\pi(s)\,\pi(a \mid s)\,A^\pi(s, a)$, with the
>   unnormalised visitation `state_distribution(...) / (1 - gamma)`.
> - REINFORCE weights each step's score by $\gamma^t G_t$.
> - Test unbiasedness with standard errors, not a fixed tolerance.

## Build it step by step

The checks test the functions in this order. `sample_episodes` is given.

1. **`softmax_policy(theta)`.** Takes `(S, A)` logits and returns `(S, A)`
   probabilities. Subtract `theta.max(axis=1, keepdims=True)` before
   `np.exp`. Common mistake: taking the max over the whole array, which works
   until two rows differ by 800.
2. **`log_policy_grad(theta, s, a)`.** Returns an `(S, A)` array that is zero
   outside row `s`. Common mistake: filling row `s` with `onehot(a)` alone,
   which forgets that raising one logit lowers the others' probabilities.
3. **`objective(mdp, theta)`.** Returns a float: `mdp.mu0 @` the values from
   `rlenvs.policy_evaluation`.
4. **`exact_gradient(mdp, theta)`.** Returns `(S, A)`. Build the advantage
   with `rlenvs.q_from_v`, the visitation with `rlenvs.state_distribution`,
   and multiply elementwise. Common mistake: using the normalised visitation.
   The finite-difference check catches the missing $1 / (1 - \gamma)$.
5. **`discounted_returns(rewards, gamma)`.** Loop backwards over the last
   axis with `running = rewards[..., t] + gamma * running`. Using `...` makes
   the same code work for one episode or a batch.
6. **`reinforce_estimate(theta, states, actions, rewards, gamma)`.** Returns
   `(S, A)`. Build the weights $\gamma^t G_t$ and the scores, then use
   `np.add.at`. Common mistake: leaving out $\gamma^t$. The check names it if
   you do.
7. **`reinforce_gradient(mdp, theta, episodes, horizon, seed)`.** Make
   `rng = np.random.default_rng(seed)`, call `sample_episodes`, and stack one
   estimate per episode into an `(N, S, A)` array. The check runs 2,000
   episodes of 100 steps.

## Cost and scaling

The exact gradient needs linear solves, so it only works on small tabular
problems. REINFORCE's cost is set by how many steps you sample. The following
table compares them:

| Method | Time | Memory | Needs a model? |
|---|---|---|---|
| Exact gradient | $O(S^3 + S^2 A)$ for the solves | $O(S^2 A)$ for `P` | Yes |
| Finite differences | $SA$ extra solves, so $O(S^4 A)$ | $O(S^2 A)$ | Yes |
| REINFORCE | $O(N H)$ steps plus one score per step | $O(H)$ per episode | No |

The catch is the constant hidden in $N$. The standard error shrinks like
$1 / \sqrt{N}$, so halving the noise costs four times the episodes. With a
neural network policy, each step also costs a forward and a backward pass.
Deep learning libraries compute the whole sum at once by building the loss
`-(weights * logp).sum()` and calling `backward()`.

## Where this breaks

**Variance.** The return $G_t$ carries the noise of every later reward and
action. Long episodes and rewards far from zero make the estimate so noisy
that learning crawls. Baselines and critics, the subject of
*Baselines and actor-critic*, cut this noise without adding
bias.

**Dropping $\gamma^t$.** Almost every deep RL implementation weights all steps
equally and keeps $\gamma$ only inside $G_t$. Nota and Thomas (2020) showed
that the resulting direction isn't the gradient of any objective. It often
works in practice, but it isn't the unbiased estimator. In this lesson's
check, dropping $\gamma^t$ moves the mean by more than ten standard errors.

**On-policy data only.** The theorem averages over the current policy's
states and actions. Once you update $\theta$, old episodes come from the
wrong distribution. Reusing them needs importance weights, which
*Proximal policy optimisation* controls with
clipping.

**Step size.** A gradient step that is small in $\theta$ can be large in
behaviour, especially when a softmax is nearly deterministic. Too large a
step collapses the policy onto one action, and it stops exploring.
*Natural gradients and TRPO* measure steps in
terms of how much the policy changes instead.

**Slow escape from bad regions.** When $\pi(a \mid s)$ is tiny for the best
action, the gradient $d^\pi \pi A^\pi$ is tiny too, and plain gradient ascent
can stall for a long time. *Why policy optimisation works*
makes this precise.

## Further reading

- [Williams (1992), Simple statistical gradient-following algorithms for connectionist reinforcement learning](https://doi.org/10.1007/BF00992696)
  introduces REINFORCE and the score-function gradient for learning systems.
- [Sutton, McAllester, Singh, and Mansour (2000), Policy gradient methods for reinforcement learning with function approximation](https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)
  proves the policy gradient theorem in the form this lesson uses.
- [Sutton and Barto (2018), Reinforcement Learning: An Introduction, chapter 13](http://incompleteideas.net/book/the-book-2nd.html)
  derives the theorem step by step and presents REINFORCE with the $\gamma^t$
  factor.
- [Nota and Thomas (2020), Is the policy gradient a gradient?](https://arxiv.org/abs/1906.07073)
  shows what goes wrong when implementations drop $\gamma^t$ from the state
  weighting.
- [Agarwal, Kakade, Lee, and Mahajan (2021), On the theory of policy gradient methods](https://arxiv.org/abs/1908.00261)
  analyses when softmax policy gradients converge and how fast.

## The exercise

You build the tabular policy gradient from the ground up. Write
`softmax_policy` and its score `log_policy_grad`, the exact objective
`objective(mdp, theta)`, and the theorem itself in
`exact_gradient(mdp, theta)`. Then write `discounted_returns`, the
one-episode estimator `reinforce_estimate`, and `reinforce_gradient`, which
returns one estimate per sampled episode.

The checks confirm the softmax's values and stability, the score against
finite differences and its zero mean, and $J = 2.25$ on the worked example.
They check `exact_gradient` against the hand-worked table and against finite
differences of `objective` on four random MDPs. Then they check the returns
and the single-episode estimate against hand computations. Finally, they run
2,000 episodes on a three-state MDP and require every component of the
REINFORCE mean to sit within four standard errors of the exact gradient.
