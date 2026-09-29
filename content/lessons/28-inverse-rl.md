---
title: Inverse reinforcement learning and GAIL
slug: 28-inverse-rl
part: "Part 6 — Models, exploration, and fixed data"
summary: Recover a reward that explains an expert by matching feature counts, then imitate through a GAIL discriminator instead.
minutes: 60
gpu: false
objectives:
  - Run soft value iteration backward and state visitation forward to get a reward's expected feature counts.
  - Derive the maximum-entropy IRL gradient as expert counts minus expected counts, and check it by finite differences.
  - Explain why the learned reward matches the expert's behaviour but not its exact reward.
  - Build the GAIL discriminator loss, its optimum rho_E / (rho_E + rho_pi), and the reward -log(1 - D).
prereqs:
  - 22-maximum-entropy-rl
  - 27-imitation-learning
exercise: 28-inverse-rl
---

# Inverse reinforcement learning and GAIL

Behaviour cloning copies what an expert does. It never asks why. A driver who
slows near a school is not following a rule about one street. They care about
children, and that one preference explains thousands of choices on streets you
never recorded. Inverse reinforcement learning (IRL) tries to recover that
preference: given demonstrations, find a reward the expert seems to be
maximising.

A reward is a compact, portable summary of behaviour. You can plan with it in a
world whose layout changed. The catch is that the question has too many
answers: a reward of zero everywhere makes every behaviour optimal.

This lesson resolves that ambiguity with the principle of maximum entropy, and
builds the algorithm of Ziebart and colleagues (2008) on a grid world. Then it
turns to GAIL, which skips the reward entirely and trains a classifier to tell
expert from learner. That classifier's output becomes the learner's reward.

## What you already know

- Behaviour cloning fits $\pi(a \mid s)$ to expert pairs, and its errors
  compound once the learner leaves the expert's states (lesson 27).
- Soft value iteration replaces the max in the Bellman backup with a
  log-sum-exp, and its optimal policy is a softmax of $Q$ (lesson 22).
- A policy's state-visitation distribution says how often it is in each state
  (lesson 4).
- Logistic regression trains a sigmoid classifier with cross-entropy.
- The gradient of $\log \sum_i e^{z_i}$ with respect to $z_i$ is the softmax
  weight of $i$.

## Why one expert fits many rewards

A *reward* here is a number $r(s)$ earned for each step spent in state $s$. A
*trajectory* $\tau$ is a sequence of states and actions $s_0, a_0, \dots,
s_{T-1}, a_{T-1}$ over a fixed *horizon* of $T$ steps. You describe each state
by a *feature vector* $f(s)$ of length $K$, and you look for a reward that is
linear in those features, $r(s) = \theta^\top f(s)$.

The expert's *feature counts* are the features summed along a trajectory and
averaged over demonstrations, $\tilde f_E = \frac{1}{N}\sum_i \sum_t f(s_t^{(i)})$.
Abbeel and Ng (2004) observed that if your policy produces the same expected
feature counts as the expert, then it earns the same expected return as the
expert under every reward of the form $\theta^\top f$. The same weights apply
to both, and expected return is $\theta$ dotted with the expected counts.

Many rewards still explain the same behaviour. Adding a constant to $r$
changes nothing when every trajectory lasts $T$ steps, and lesson 30 shows a
whole family of such changes, called shaping. You can't recover *the* reward.
You can find a reward whose behaviour matches the expert's.

## Choose the most random explanation

The principle of maximum entropy says: among all distributions over
trajectories that match the expert's feature counts, pick the one that commits
to nothing else. That is the one with the highest entropy. The solution has an
exponential form:

$$
P_\theta(\tau) = \frac{1}{Z(\theta)} \exp\Big(\sum_{t=0}^{T-1} \theta^\top f(s_t)\Big)
$$

Read it aloud: a trajectory's probability grows exponentially with the reward it
collects. Trajectories that tie on reward are equally likely. $Z(\theta)$ is
the sum of the numerator over every trajectory, which makes the probabilities
add to one.

This formula assumes deterministic dynamics, as in this lesson's exercise.
With random dynamics, it lets the agent "choose" lucky transitions. Ziebart and
colleagues (2010) fixed that with *causal* entropy, which is exactly the soft
value iteration you are about to write. With deterministic dynamics, the two
agree.

## Run the backward pass: soft value iteration

You never enumerate trajectories. Instead, you compute soft values backward
from the horizon, as in lesson 22 but with a finite horizon and no discount:

$$
Q_t(s, a) = r(s) + \sum_{s'} P(s' \mid s, a)\, V_{t+1}(s'), \qquad
V_t(s) = \log \sum_a e^{Q_t(s, a)}, \qquad V_T = 0
$$

Read it aloud: the soft value of acting at step $t$ is this state's reward plus
the expected soft value of wherever you land. The soft value of a state is a
smooth maximum over actions. The policy follows as a softmax:

$$
\pi_t(a \mid s) = \exp\big(Q_t(s, a) - V_t(s)\big)
$$

In numpy, one step backward is three lines. `P @ V[t + 1]` contracts the last
axis of `P`, shape `(S, A, S)`, with `V[t + 1]`, shape `(S,)`, and returns
`(S, A)`.

```python
Q = r[:, None] + P @ V[t + 1]              # (S, A)
V[t] = logsumexp(Q, axis=1)                # subtract the row max first
pi[t] = np.exp(Q - V[t][:, None])          # rows sum to one
```

With deterministic dynamics, multiplying these policies along a trajectory
gives exactly $P_\theta(\tau)$. The soft values telescope, and what remains is
$\exp(\text{return} - V_0(s_0))$. So $V_0(s_0) = \log Z(\theta)$.

### Worked example: a two-cell corridor

Take the corridor `S.`, two cells side by side, with the start on the left. The
four actions are up, right, down, and left. From the left cell, only right
moves. Up, down, and left bump into walls and stay. Let $r = (0, 1)$: the
right cell pays 1 per step. Use $T = 2$.

- At $t = 1$, every action is the last one, so $Q_1(s, a) = r(s)$ and
  $V_1 = r + \log 4 = (1.3863, 2.3863)$.
- At $t = 0$ in the left cell, the three bumping actions stay:
  $Q_0 = 0 + V_1(\text{left}) = 1.3863$. Moving right gives
  $Q_0 = 0 + V_1(\text{right}) = 2.3863$.
- $V_0(\text{left}) = \log(3e^{1.3863} + e^{2.3863}) = \log(12 + 4e) = 3.1300$.
- $\pi_0(\text{right} \mid \text{left}) = e^{2.3863 - 3.1300} = e / (e + 3) = 0.4754$.

Check it by counting. Of the 16 action sequences, the 4 that start with right
collect reward 1, and the other 12 collect 0. Maximum entropy gives the first
group $4e / (4e + 12) = 0.4754$ of the probability.

## Run the forward pass: where the policy goes

Expected feature counts need to know how often the policy is in each state at
each step. That is a forward pass. $D_t(s)$ is the probability of being in $s$
at step $t$:

$$
D_0 = \mu_0, \qquad D_{t+1}(s') = \sum_{s, a} D_t(s)\, \pi_t(a \mid s)\, P(s' \mid s, a)
$$

Read it aloud: start from the start distribution $\mu_0$. To get the next
step's distribution, push each state's probability through the policy and then
through the dynamics. The expected feature counts are then
$\mathbb{E}_\theta[f] = \sum_t \sum_s D_t(s) f(s)$.

```python
D[t + 1] = np.einsum("s,sa,sap->p", D[t], pi[t], P)
f_expected = D.sum(axis=0) @ features       # (K,)
```

In the corridor, $D_0 = (1, 0)$ and $D_1 = (0.5246, 0.4754)$. With one
indicator feature per cell, the expected counts are $(1.5246, 0.4754)$. They
sum to $T = 2$, one visit per step.

> [!KEY]
> A backward pass turns a reward into a policy. A forward pass turns that
> policy into expected feature counts. IRL runs both every iteration.

## Learn the reward by matching features

Fit $\theta$ by maximum likelihood: make the demonstrations as probable as
possible under $P_\theta$. The mean log-likelihood of $N$ demonstrations is

$$
\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \log P_\theta(\tau_i)
= \theta^\top \tilde f_E - \log Z(\theta)
$$

Read it aloud: reward the demonstrations' feature counts, then subtract the log
of the total weight of all trajectories. The derivative of $\log Z(\theta)$ is
the expected feature count under $P_\theta$, the same way the derivative of a
log-sum-exp is a softmax average. So the gradient is

$$
\nabla_\theta \mathcal{L} = \tilde f_E - \mathbb{E}_{\tau \sim P_\theta}\Big[\sum_t f(s_t)\Big]
$$

Read it aloud: the gradient is what the expert does minus what your current
reward predicts. Where the expert visits a feature more often than predicted,
raise that feature's weight. At the optimum, the two counts match exactly, which
is the feature-matching condition.

The log-likelihood is concave in $\theta$, so gradient ascent doesn't get stuck
in a bad local optimum. Each step costs one backward pass and one forward pass:

```python
r = features @ theta
_, pi = soft_value_iteration(P, r, T)
grad = f_expert - feature_expectations(state_visitation(P, pi, mu0), features)
theta = theta + lr * grad
```

### Worked example, continued

Suppose the corridor's expert data shows counts $\tilde f_E = (1.2, 0.8)$: the
expert moves right 80% of the time. At $r = (0, 1)$, the model predicts
$(1.5246, 0.4754)$, so the gradient is $(-0.3246, +0.3246)$. It lowers the left
cell's weight and raises the right cell's.

Where does ascent stop? The model moves right with probability
$e^{\Delta} / (e^{\Delta} + 3)$, where $\Delta = \theta_{\text{right}} -
\theta_{\text{left}}$. Setting that to 0.8 gives $e^\Delta = 12$, or
$\Delta = \log 12 = 2.4849$. Only the difference is pinned down. The pair
$(0, 2.4849)$ and the pair $(-5, -2.5151)$ explain the data equally well.

> [!KEY]
> Maximum-entropy IRL recovers a reward whose soft-optimal behaviour matches
> the expert's feature counts. It doesn't recover the expert's reward itself,
> which the data leaves ambiguous.

## GAIL: match the expert with a classifier

IRL has an inner loop: every reward update needs a full solve of the
reinforcement learning problem. Ho and Ermon (2016) showed that running
max-entropy IRL and then RL on the result amounts to one goal, softened
by a regularizer on the reward: make the learner's *occupancy measure* match
the expert's. The occupancy measure
$\rho_\pi(s, a)$ is how often policy $\pi$ visits the pair $(s, a)$. Their
method, *generative adversarial imitation learning* (GAIL), chases that goal
directly.

A *discriminator* $D(s, a)$ is a classifier. It outputs the probability that a
pair came from the expert rather than the learner. Train it with the usual
cross-entropy:

$$
\mathcal{L}_D = -\sum_{s, a} \rho_E(s, a) \log D(s, a) - \sum_{s, a} \rho_\pi(s, a) \log\big(1 - D(s, a)\big)
$$

Read it aloud: push $D$ toward 1 on pairs the expert visits and toward 0 on
pairs the learner visits, weighted by how often each visits them. Write
$D = \sigma(\ell)$ with a logit $\ell$. Then the gradient with respect to the
logit is $\rho_\pi D - \rho_E (1 - D)$.

Setting that gradient to zero gives the best discriminator in closed form:

$$
D^*(s, a) = \frac{\rho_E(s, a)}{\rho_E(s, a) + \rho_\pi(s, a)}
$$

Read it aloud: the best classifier reports the expert's share of the visits to
that pair. Its logit is $\log \rho_E - \log \rho_\pi$, the log-ratio of the two
occupancies. At this optimum, the loss equals $2 \log 2 - 2\,\mathrm{JS}(\rho_E,
\rho_\pi)$, where JS is the Jensen-Shannon divergence. It is the same identity
as in the original GAN paper, so a better learner leaves the classifier with a
higher loss.

The learner then takes one step of policy optimisation, such as TRPO or PPO,
with the reward

$$
r(s, a) = -\log\big(1 - D(s, a)\big)
$$

Read it aloud: the reward is large where the classifier is sure the pair looks
expert. Ho and Ermon wrote the objective with $D$ as the probability of the
*learner*, with cost $\log D$. Flipping the convention turns their cost into
this reward. Another popular choice, from AIRL (Fu and colleagues, 2018), is
$\log D - \log(1 - D)$, which is the logit itself. At the optimum, its expected
value under the learner is $-\mathrm{KL}(\rho_\pi \,\|\, \rho_E)$.

```python
log_sig = lambda z: -np.logaddexp(0.0, -z)            # log sigmoid(z)
loss = -(rho_e * log_sig(l)).sum() - (rho_p * log_sig(-l)).sum()
reward = np.logaddexp(0.0, l)                         # -log(1 - sigmoid(l))
```

### Worked example: four state-action pairs

Let $\rho_E = (0.5, 0.3, 0.2, 0)$ and $\rho_\pi = (0.1, 0.2, 0.3, 0.4)$.

- The best discriminator is $D^* = (0.833, 0.6, 0.4, 0)$.
- The GAIL rewards are $-\log(1 - D^*) = (1.792, 0.916, 0.511, 0)$.
- The Jensen-Shannon divergence is $0.2215$, so the best loss is
  $2 \log 2 - 0.4429 = 0.9433$.

The fourth pair, which the expert never takes, earns reward 0, the lowest
possible. The learner shifts its visits toward the first pair.

> [!RECAP]
> - Soft value iteration runs backward from $V_T = 0$ with a log-sum-exp, and
>   its policy is $\exp(Q_t - V_t)$.
> - State visitation runs forward from $\mu_0$, and summing it times the
>   features gives expected counts.
> - The IRL gradient is expert counts minus expected counts. At the optimum
>   they match, but the reward is only pinned down up to ambiguity.
> - The best GAIL discriminator is $\rho_E / (\rho_E + \rho_\pi)$, and the
>   learner's reward is $-\log(1 - D)$.

## Build it step by step

The checks test the functions in this order.

1. **`soft_value_iteration(P, r, horizon)`.** Returns `V` with shape
   `(T + 1, S)` and `pi` with shape `(T, S, A)`. Loop `t` backward from
   `T - 1` to `0`. Write a stable log-sum-exp that subtracts each row's max
   before `np.exp`. Common mistake: taking a plain max instead of the
   log-sum-exp. That is ordinary value iteration, and its policy ignores
   near-ties that the expert's randomness depends on.
2. **`state_visitation(P, pi, mu0)`.** Returns `D` with shape `(T, S)`. Set
   `D[0] = mu0`, then use one `np.einsum` per step. Common mistake: computing
   `T + 1` rows, which counts a step the reward never pays.
3. **`feature_expectations(D, features)`.** One line: `D.sum(axis=0) @
   features`.
4. **`empirical_feature_counts(states, features)`.** `features[states]` has
   shape `(N, T, K)`. Sum over time, then average over trajectories. Common
   mistake: averaging over time too, which divides the counts by `T`.
5. **`maxent_irl_gradient(P, features, mu0, horizon, theta, f_expert)`.** Chain
   the first three functions and subtract. The check compares it with the
   finite-difference gradient of the demonstrations' log-likelihood. Common
   mistake: flipping the sign.
6. **`maxent_irl(...)`.** Start at `theta = 0` and add `lr` times the gradient
   `iters` times. The check expects the counts to match within 0.02.
7. **`discriminator_loss` and `discriminator_grad`.** Use `np.logaddexp` for
   both log-sigmoid terms. The checks test the loss at `D = 1/2`, which is
   $2 \log 2$, the gradient by finite differences, and the optimum.
8. **`gail_reward(logits)`.** `np.logaddexp(0, logits)`. Common mistake:
   computing `-np.log(1 - sigmoid(l))` directly, which returns infinity at
   large logits.

## Cost and scaling

One IRL iteration costs a backward and a forward pass, each $T$ steps of work
proportional to $S \cdot A \cdot S$ for a dense transition array. The
following table summarises the costs.

| Method | Work per reward update | Memory | Needs the dynamics? |
|---|---|---|---|
| Max-ent IRL, tabular | $O(T S^2 A)$ | $O(T S A)$ for the policies | Yes |
| Max-ent IRL, sampled (guided cost learning) | A batch of rollouts | A reward network | No |
| GAIL | One discriminator step and one PPO step | Two networks | No |

Tabular IRL is exact but needs states you can enumerate. Deep IRL estimates
the expected counts from samples instead. GAIL costs about as much as ordinary
PPO plus a classifier.

## Where this breaks

**The reward isn't identifiable.** Any constant, and any potential-based
shaping term, explains the same behaviour. A reward learned in one world can
behave badly in another where those extra terms matter. AIRL (Fu and
colleagues, 2018) constrains the reward's form so it transfers better across
dynamics.

**The expert is assumed to be noisily rational.** Maximum entropy models the
expert as picking trajectories in proportion to $e^{\text{return}}$. A
systematically biased expert, such as one who never considers a shortcut,
produces a reward that encodes the bias as a preference.

**Features limit what you can learn.** With one feature per state, states the
demonstrations never visit get counts of zero, and their weights drift toward
minus infinity.

**GAIL's discriminator wins too fast.** A perfect classifier gives a flat,
useless reward. People add gradient penalties or input noise.

**Reward sign creates a bias at episode ends.** $-\log(1 - D)$ is always
positive, so the learner is paid for surviving longer, whatever it does. $\log
D$ is always negative, so the learner wants to end episodes early. Kostrikov
and colleagues (2019) traced this to how absorbing states are handled and fixed
it by learning a reward for them too.

## Further reading

- [Ziebart, Maas, Bagnell, and Dey (2008), Maximum Entropy Inverse
  Reinforcement Learning](https://cdn.aaai.org/AAAI/2008/AAAI08-227.pdf). The
  algorithm in this lesson, with the backward and forward passes, applied to
  taxi drivers' route choices.
- [Abbeel and Ng (2004), Apprenticeship Learning via Inverse Reinforcement
  Learning](https://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf). The
  feature-matching argument that a policy with the expert's feature counts
  earns the expert's return.
- [Ng and Russell (2000), Algorithms for Inverse Reinforcement
  Learning](https://ai.stanford.edu/~ang/papers/icml00-irl.pdf). The first
  statement of the problem and of its ambiguity.
- [Ho and Ermon (2016), Generative Adversarial Imitation
  Learning](https://arxiv.org/abs/1606.03476). Shows that IRL followed by RL is
  occupancy matching, and introduces GAIL.
- [Fu, Luo, and Levine (2018), Learning Robust Rewards with Adversarial Inverse
  Reinforcement Learning](https://arxiv.org/abs/1710.11248). AIRL, a
  discriminator whose logit is a reward that transfers to new dynamics.
- [Kostrikov and colleagues (2019),
  Discriminator-Actor-Critic](https://arxiv.org/abs/1809.02925). Explains the
  survival and termination biases of GAIL's reward and how to fix them.

## The exercise

You build maximum-entropy IRL on a small `rlenvs.gridworld` and a tabular
GAIL discriminator. Write `soft_value_iteration`, `state_visitation`,
`feature_expectations`, and `empirical_feature_counts`, then
`maxent_irl_gradient` and the gradient-ascent loop `maxent_irl`. Finish with
`discriminator_loss`, `discriminator_grad`, and `gail_reward`.

The checks confirm the corridor example's numbers, and that soft value
iteration gives each of 64 action sequences a probability proportional to the
exponential of its return. They compare the visitation with sampled
trajectories, and the IRL gradient with finite differences of the
demonstrations' log-likelihood. They also check that the gradient is zero at
the reward that generated exact counts. The loop must match the expert's
feature counts within 0.02 and its policy wherever the expert goes. For GAIL,
the checks test the loss gradient by finite differences, the optimal logit
$\log(\rho_E / \rho_\pi)$ with its loss of $2 \log 2 - 2\,\mathrm{JS}$, and a
reward that stays finite at logits of 800.
