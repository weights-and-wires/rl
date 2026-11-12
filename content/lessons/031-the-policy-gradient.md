---
title: The policy gradient and REINFORCE
slug: 031-the-policy-gradient
part: "Part 5 — Policy gradients"
summary: Climb the expected return directly by gradient ascent, using the log-derivative trick to estimate the gradient from sampled episodes.
minutes: 40
gpu: false
objectives:
  - Explain why you cannot backpropagate through a random choice, and how the log-derivative trick gets around it.
  - Compute the score of a tabular softmax policy and show that its average over actions is zero.
  - Build the REINFORCE estimate of the gradient of the expected return from one episode.
  - Test that an estimator is unbiased by comparing its mean with a reference gradient in standard errors.
prereqs:
  - 006-gradient-bandits
  - 002-returns-and-discounting
  - 009-the-bellman-equation
  - 013-monte-carlo-prediction
exercise: 031-the-policy-gradient
---

# The policy gradient and REINFORCE

Every method so far learned values and then acted greedily on them. The policy was an afterthought, read off a value table with an `argmax`. That works when actions are few and the values are accurate. It struggles when the best behavior is random, when actions are continuous, or when a tiny change in a value flips the greedy action and the agent's behavior jumps.

This part of the course turns the problem around. You write the policy as a function with parameters, and you climb the expected return directly by gradient ascent, the way you trained a network in your machine learning course. The obstacle is that the return depends on the parameters through a random choice. You cannot backpropagate through "pick an action at random, then let the world respond."

This chapter shows how to get a gradient anyway. A one-line identity turns the gradient of an average into an average you can estimate from episodes, with no model of the world. You build the simplest estimator, REINFORCE, and test that its average matches a gradient you measure independently.

## What you already know

- A *policy* $\pi(a \mid s)$ gives the probability of taking action $a$ in state $s$. In code it is an `(S, A)` array whose rows sum to one.
- The *return* $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$ is the discounted sum of rewards from step $t$ onward.
- $V^\pi(s)$ is the expected return from $s$, and `rlenvs.policy_evaluation` computes it exactly on a small MDP.
- The gradient bandit nudged softmax preferences up for actions that paid more than a baseline.
- A Monte Carlo estimate averages sampled returns, and its error shrinks like $1/\sqrt{N}$.

## Write the policy as a softmax

Give every state its own row of scores, called *logits*. The policy turns each row into probabilities with a softmax. The parameters are the whole table $\theta$, of shape `(S, A)`:

$$
\pi_\theta(a \mid s) = \frac{e^{\theta_{s,a}}}{\sum_{b} e^{\theta_{s,b}}}
$$

Read it as: exponentiate the scores in row $s$, then divide by their total so they sum to one. A higher score means a likelier action. Adding the same number to a whole row changes nothing, because it cancels in the ratio, so you can subtract the row maximum to avoid overflow:

```python
z = theta - theta.max(axis=1, keepdims=True)   # same softmax, no overflow
pi = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
```

The quantity to raise is the *objective*, the expected return from a start state drawn from $\mu_0$:

$$
J(\theta) = \sum_s \mu_0(s)\, V^{\pi_\theta}(s)
$$

Read it as: the value of each start state, weighted by how likely you are to start there. On a small MDP you compute it with one call: `mdp.mu0 @ rlenvs.policy_evaluation(mdp, pi)`.

The running example is a two-state MDP. In each state you can *stay* or *switch*, and switching moves you to the other state. Staying in state 1 pays 1, and every other choice pays 0. You start in state 0 with $\gamma = 0.9$. With $\theta = 0$ the policy is uniform, and $J = 2.25$.

## Why you cannot backpropagate through a choice

Suppose you want $\nabla_\theta J$. Two obvious approaches both fail.

*Backpropagate through the sample.* An action is drawn from $\pi_\theta$, and the draw is a discrete jump. Nudging $\theta$ a little leaves the sampled action unchanged until the nudge flips it, and then it jumps. There is no slope to follow.

*Measure by finite differences.* Nudge one logit up and down by $10^{-6}$, recompute $J$, and divide the difference by $2 \times 10^{-6}$. This works, and you use it in this chapter as a reference. But it needs the exact $J$, which needs the full model of the world, and it costs two solves per parameter. A real policy has millions of parameters and no model to solve.

You want an estimator built from sampled episodes. The log-derivative trick builds one.

## The log-derivative trick

Suppose $x$ is drawn from a distribution $p_\theta(x)$, and you want the gradient of the average of some score $f(x)$. The gradient of a probability equals the probability times the gradient of its log, because $\nabla \log p = \nabla p / p$. So

$$
\nabla_\theta\, \mathbb{E}_{x \sim p_\theta}\big[f(x)\big] = \sum_x f(x)\, \nabla_\theta p_\theta(x) = \mathbb{E}_{x \sim p_\theta}\big[f(x)\, \nabla_\theta \log p_\theta(x)\big]
$$

Read it as: to learn how the average of $f$ moves, sample $x$, and push up its log-probability in proportion to how good $f(x)$ was. You never differentiate $f$ itself, so the world can stay a black box. The vector $\nabla_\theta \log p_\theta(x)$ is called the *score*.

For a tabular softmax policy the score has a clean form. Only row $s$ is nonzero, and in that row the score is the one-hot vector for $a$ minus the probabilities:

$$
\frac{\partial \log \pi_\theta(a \mid s)}{\partial \theta_{s,b}} = \mathbb{1}[a = b] - \pi_\theta(b \mid s)
$$

Read it as: raising the logit of the action you took makes it likelier, and every other action in that state gives up a share in proportion to its probability. In numpy:

```python
grad = np.zeros_like(pi)
grad[s] = -pi[s]
grad[s, a] += 1.0
```

### Check the trick by hand

Take one state with three actions and $\pi = [0.5, 0.3, 0.2]$. Action 0 pays 1 and the others pay 0, so the average reward is $\pi_0$. Differentiating $\pi_0$ directly gives $\partial \pi_0 / \partial \theta_0 = \pi_0(1 - \pi_0) = 0.25$, then $-\pi_0 \pi_1 = -0.15$ and $-\pi_0 \pi_2 = -0.10$.

The trick gives the same numbers. Only action 0 has $f \neq 0$, so the average is $\pi_0 \times 1 \times \text{score}$, and the score for action 0 is $[1, 0, 0] - \pi = [0.5, -0.3, -0.2]$. Multiplying by $\pi_0 = 0.5$ gives $[0.25, -0.15, -0.10]$.

```figure
<svg viewBox="0 0 620 262" role="img" aria-label="The score of a softmax policy for two chosen actions: plus the chosen action's share, minus every other action's probability">
<text x="160.0" y="22.0" class="fig-bold" text-anchor="middle">the agent took a0: score = onehot - pi</text>
<line x1="30.0" y1="130.0" x2="290.0" y2="130.0" class="stroke-muted"/>
<rect x="55.0" y="80.0" width="50.0" height="50.0" class="fill-b"/>
<text x="80.0" y="74.0" class="fig-small fig-bold" text-anchor="middle">+0.5</text>
<text x="80.0" y="222.0" class="fig-small" text-anchor="middle">a0 (pi 0.5)</text>
<rect x="135.0" y="130.0" width="50.0" height="30.0" class="fill-d"/>
<text x="160.0" y="176.0" class="fig-small fig-bold" text-anchor="middle">-0.3</text>
<text x="160.0" y="222.0" class="fig-small" text-anchor="middle">a1 (pi 0.3)</text>
<rect x="215.0" y="130.0" width="50.0" height="20.0" class="fill-d"/>
<text x="240.0" y="166.0" class="fig-small fig-bold" text-anchor="middle">-0.2</text>
<text x="240.0" y="222.0" class="fig-small" text-anchor="middle">a2 (pi 0.2)</text>
<text x="460.0" y="22.0" class="fig-bold" text-anchor="middle">the agent took a2: score = onehot - pi</text>
<line x1="330.0" y1="130.0" x2="590.0" y2="130.0" class="stroke-muted"/>
<rect x="355.0" y="130.0" width="50.0" height="50.0" class="fill-d"/>
<text x="380.0" y="196.0" class="fig-small fig-bold" text-anchor="middle">-0.5</text>
<text x="380.0" y="222.0" class="fig-small" text-anchor="middle">a0 (pi 0.5)</text>
<rect x="435.0" y="130.0" width="50.0" height="30.0" class="fill-d"/>
<text x="460.0" y="176.0" class="fig-small fig-bold" text-anchor="middle">-0.3</text>
<text x="460.0" y="222.0" class="fig-small" text-anchor="middle">a1 (pi 0.3)</text>
<rect x="515.0" y="50.0" width="50.0" height="80.0" class="fill-b"/>
<text x="540.0" y="44.0" class="fig-small fig-bold" text-anchor="middle">+0.8</text>
<text x="540.0" y="222.0" class="fig-small" text-anchor="middle">a2 (pi 0.2)</text>
<text x="310.0" y="250.0" class="fig-small" text-anchor="middle">Weighted by pi = [0.5, 0.3, 0.2], the scores of all three actions add to [0, 0, 0].</text>
</svg>
Figure 1. For a policy with probabilities 0.5, 0.3, and 0.2, the score of the chosen action is +1 minus its probability, and every other entry is minus that action's probability. Raising the chosen logit (orange) lowers the rest (purple).
```

The score has one property you use again and again. Averaged over actions drawn from the policy, it is zero, as Figure 1 shows: $\sum_a \pi(a \mid s) \nabla \log \pi(a \mid s) = \nabla \sum_a \pi(a \mid s) = \nabla 1 = 0$.

> [!KEY]
> The log-derivative trick turns the gradient of an expectation into the expectation of "how good was it" times the score $\nabla \log \pi(a \mid s)$. You can estimate that from samples without knowing how the world works.

## From one choice to a whole episode

An episode is a chain of random choices: the policy picks actions, and the world picks next states. The probability of a whole episode is a product of the start probability, the policy's probabilities, and the world's probabilities. Take the log of the product and it becomes a sum. The world's terms do not contain $\theta$, so their gradient is zero. Only the policy's terms survive, which gives the score of the episode as $\sum_t \nabla \log \pi_\theta(a_t \mid s_t)$.

Apply the trick with $f$ equal to the discounted return $\sum_k \gamma^k r_k$ of the episode. You get every reward multiplied by every score, including scores of actions taken after the reward. That is wasteful, because a reward $r_k$ cannot depend on an action taken later at step $t > k$. Each such term has an average of zero, because the score averages to zero given everything before it. Dropping them removes noise without changing the average, which leaves

$$
\hat g = \sum_{t=0}^{T-1} \gamma^t\, G_t\, \nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

This is REINFORCE (Williams, 1992). Read it as: for every step, take the score of the action you chose, scale it by the discounted return that followed, and scale again by $\gamma^t$ for how late the step came. Then add up the steps. The $\gamma^t$ is there because the objective counts rewards from the start of the episode, so a choice made late can change less of the discounted total.

In numpy, for one episode:

```python
weights = gamma ** np.arange(T) * G              # G from discounted_returns
scores = -pi[states]
scores[np.arange(T), actions] += 1.0
np.add.at(grad, states, weights[:, None] * scores)
```

`np.add.at` matters here. Plain `grad[states] += ...` keeps only one update when a state repeats.

### Work an example

In the two-state MDP with the uniform policy, suppose one episode goes $s = [0, 1, 1]$, $a = [\text{switch}, \text{stay}, \text{switch}]$, with rewards $r = [0, 1, 0]$. The returns are $G = [0 + 0.9 \times 1, 1, 0] = [0.9, 1, 0]$. The weights $\gamma^t G_t$ are $[0.9, 0.9, 0]$, as Figure 2 tabulates.

```figure
<svg viewBox="0 0 620 260" role="img" aria-label="One three-step episode with each step's state, action, reward, return, and weight in the REINFORCE estimate">
<text x="30.0" y="30.0" class="fig-bold" text-anchor="start"></text>
<text x="140.0" y="30.0" class="fig-bold" text-anchor="start">state</text>
<text x="250.0" y="30.0" class="fig-bold" text-anchor="start">action</text>
<text x="360.0" y="30.0" class="fig-bold" text-anchor="start">reward r_t</text>
<text x="450.0" y="30.0" class="fig-bold" text-anchor="start">G_t</text>
<text x="530.0" y="30.0" class="fig-bold" text-anchor="start">gamma^t G_t</text>
<line x1="20.0" y1="40.0" x2="600.0" y2="40.0" class="stroke-muted"/>
<rect x="24.0" y="48.0" width="572.0" height="34.0" class="fill-soft-a"/>
<text x="30.0" y="70.0" text-anchor="start">t = 0</text>
<text x="140.0" y="70.0" text-anchor="start">state 0</text>
<text x="250.0" y="70.0" text-anchor="start">switch</text>
<text x="360.0" y="70.0" class="fig-mono" text-anchor="start">0</text>
<text x="450.0" y="70.0" class="fig-mono" text-anchor="start">0.9</text>
<text x="530.0" y="70.0" class="fig-mono" text-anchor="start">0.9</text>
<rect x="24.0" y="92.0" width="572.0" height="34.0" class="fill-sunk"/>
<text x="30.0" y="114.0" text-anchor="start">t = 1</text>
<text x="140.0" y="114.0" text-anchor="start">state 1</text>
<text x="250.0" y="114.0" text-anchor="start">stay</text>
<text x="360.0" y="114.0" class="fig-mono" text-anchor="start">1</text>
<text x="450.0" y="114.0" class="fig-mono" text-anchor="start">1</text>
<text x="530.0" y="114.0" class="fig-mono" text-anchor="start">0.9</text>
<rect x="24.0" y="136.0" width="572.0" height="34.0" class="fill-soft-a"/>
<text x="30.0" y="158.0" text-anchor="start">t = 2</text>
<text x="140.0" y="158.0" text-anchor="start">state 1</text>
<text x="250.0" y="158.0" text-anchor="start">switch</text>
<text x="360.0" y="158.0" class="fig-mono" text-anchor="start">0</text>
<text x="450.0" y="158.0" class="fig-mono" text-anchor="start">0</text>
<text x="530.0" y="158.0" class="fig-mono" text-anchor="start">0</text>
<text x="30.0" y="216.0" class="fig-small" text-anchor="start">Step 0 adds 0.9 x [-0.5, +0.5] to row 0. Step 1 adds 0.9 x [+0.5, -0.5] to row 1. Step 2 adds 0.</text>
<text x="30.0" y="240.0" class="fig-small fig-bold" text-anchor="start">Gradient estimate: row 0 = [-0.45, +0.45], row 1 = [+0.45, -0.45].</text>
</svg>
Figure 2. Each step contributes its score scaled by gamma^t G_t. The last step earns zero weight because nothing follows it, so it contributes nothing to the estimate.
```

Step 0 adds $0.9 \times [-0.5, 0.5]$ to row 0. Step 1 adds $0.9 \times [0.5, -0.5]$ to row 1. Step 2 adds nothing. The estimate is

$$
\hat g = \begin{bmatrix} -0.45 & 0.45 \\ 0.45 & -0.45 \end{bmatrix}
$$

It points the right way in state 1 and the wrong way in state 0. A finite-difference measurement of $J$ gives the reference $\begin{bmatrix} -0.619 & 0.619 \\ 1.631 & -1.631 \end{bmatrix}$. One episode is noisy. Averaged over thousands, the estimates should converge to the reference.

## Test for unbiasedness honestly

The estimate is *unbiased* if its average over many episodes equals the true gradient. To test that, you cannot use a fixed tolerance like `atol=0.05`, because some components are much noisier than others. Use each component's *standard error* instead, the standard deviation of the per-episode estimates divided by $\sqrt{N}$. Then check that the mean is within four standard errors of the reference. An unbiased estimator fails this test for a component about once in 16,000 tries. A biased one fails it reliably once $N$ is large.

Figure 3 shows one component for a random three-state MDP. The running average starts far from the reference and settles inside a band that narrows as episodes accumulate. The per-episode standard deviation is about 1.4, so after 2,000 episodes the standard error is 0.031 and the band is 0.12 wide on each side.

```figure
<svg viewBox="0 0 620 320" role="img" aria-label="Running average of one REINFORCE gradient component against the number of episodes, settling inside a shrinking four-standard-error band around the finite-difference value">
<line x1="70.0" y1="234.4" x2="550.0" y2="234.4" class="stroke-grid"/>
<text x="62.0" y="238.4" class="fig-small" text-anchor="end">-1.2</text>
<line x1="70.0" y1="183.3" x2="550.0" y2="183.3" class="stroke-grid"/>
<text x="62.0" y="187.3" class="fig-small" text-anchor="end">-0.8</text>
<line x1="70.0" y1="132.2" x2="550.0" y2="132.2" class="stroke-grid"/>
<text x="62.0" y="136.2" class="fig-small" text-anchor="end">-0.4</text>
<line x1="70.0" y1="81.1" x2="550.0" y2="81.1" class="stroke-grid"/>
<text x="62.0" y="85.1" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="30.0" x2="550.0" y2="30.0" class="stroke-grid"/>
<text x="62.0" y="34.0" class="fig-small" text-anchor="end">0.4</text>
<text x="70.0" y="280.0" class="fig-small" text-anchor="middle">10</text>
<text x="278.6" y="280.0" class="fig-small" text-anchor="middle">100</text>
<text x="487.2" y="280.0" class="fig-small" text-anchor="middle">1000</text>
<line x1="70.0" y1="260.0" x2="550.0" y2="260.0" class="stroke-muted"/>
<polygon points="70.0,30.0 132.8,30.0 215.8,77.1 278.6,106.6 341.4,129.4 424.4,142.8 487.2,152.7 550.0,158.1 550.0,189.4 487.2,194.8 424.4,204.6 341.4,218.1 278.6,240.9 215.8,260.0 132.8,260.0 70.0,260.0" class="fill-soft-b"/>
<line x1="70.0" y1="173.7" x2="550.0" y2="173.7" class="stroke-c dashed"/>
<polyline points="70.0,218.2 132.8,216.1 215.8,205.5 278.6,180.1 341.4,176.6 424.4,186.4 487.2,178.9 550.0,181.3" class="stroke-b" fill="none"/>
<circle cx="70.0" cy="218.2" r="3.5" class="fill-b"/>
<circle cx="132.8" cy="216.1" r="3.5" class="fill-b"/>
<circle cx="215.8" cy="205.5" r="3.5" class="fill-b"/>
<circle cx="278.6" cy="180.1" r="3.5" class="fill-b"/>
<circle cx="341.4" cy="176.6" r="3.5" class="fill-b"/>
<circle cx="424.4" cy="186.4" r="3.5" class="fill-b"/>
<circle cx="487.2" cy="178.9" r="3.5" class="fill-b"/>
<circle cx="550.0" cy="181.3" r="3.5" class="fill-b"/>
<text x="550.0" y="165.7" class="fig-small" text-anchor="end">finite difference: -0.725</text>
<text x="310.0" y="302.0" text-anchor="middle">episodes averaged (log scale)</text>
<text x="62.0" y="20.0" text-anchor="start">average REINFORCE estimate of one gradient component</text>
</svg>
Figure 3. One component of the gradient, averaged over more and more episodes (orange line). The shaded band is four standard errors around the finite-difference value (dashed green); it shrinks like 1 over the square root of the episode count, and the average stays inside it.
```

> [!KEY]
> REINFORCE is unbiased: the average of $\sum_t \gamma^t G_t \nabla \log \pi(a_t \mid s_t)$ over episodes equals $\nabla J$. It pays for that with high variance, which the next chapters attack.

> [!RECAP]
> - The softmax score in row $s$ is `onehot(a) - pi[s]`, and it averages to zero over actions.
> - $J(\theta)$ is `mu0 @ policy_evaluation(mdp, pi)`.
> - The log-derivative trick gives $\nabla \mathbb{E}[f] = \mathbb{E}[f \nabla \log p]$, so you can estimate a gradient from samples.
> - REINFORCE weights each step's score by $\gamma^t G_t$.
> - Test unbiasedness with standard errors, not a fixed tolerance.

## Build it step by step

The checks test the functions in this order. `sample_episodes` is given.

1. **`softmax_policy(theta)`.** Takes `(S, A)` logits and returns `(S, A)` probabilities. Subtract `theta.max(axis=1, keepdims=True)` before `np.exp`. The most common mistake is taking the max over the whole array, which works until two rows differ by 800.
2. **`log_policy_grad(theta, s, a)`.** Returns an `(S, A)` array that is zero outside row `s`. The most common mistake is filling row `s` with `onehot(a)` alone, which forgets that raising one logit lowers the others' probabilities.
3. **`objective(mdp, theta)`.** Returns a float: `mdp.mu0 @` the values from `rlenvs.policy_evaluation`.
4. **`discounted_returns(rewards, gamma)`.** Loop backwards over the last axis with `running = rewards[..., t] + gamma * running`. Using `...` makes the same code work for one episode or a batch.
5. **`reinforce_estimate(theta, states, actions, rewards, gamma)`.** Returns `(S, A)`. Build the weights $\gamma^t G_t$ and the scores, then use `np.add.at`. The most common mistake is leaving out $\gamma^t$. The check names it if you do.
6. **`reinforce_gradient(mdp, theta, episodes, horizon, seed)`.** Make `rng = np.random.default_rng(seed)`, call `sample_episodes`, and stack one estimate per episode into an `(N, S, A)` array. The check runs 2,000 episodes of 100 steps.

## Cost and scaling

REINFORCE needs no model of the world, and its cost is the $O(NH)$ steps you sample. The standard error shrinks like $1/\sqrt{N}$, so halving the noise costs four times the episodes. With a neural network policy, deep learning libraries compute the whole sum by building the loss `-(weights * logp).sum()` and calling `backward()`.

## Where this breaks

**Variance.** The return $G_t$ carries the noise of every later reward and action. Long episodes and rewards far from zero make the estimate so noisy that learning crawls. Baselines and critics cut this noise without adding bias, and the chapter on baselines and actor-critic builds them.

**Dropping $\gamma^t$.** Almost every deep RL implementation weights all steps equally and keeps $\gamma$ only inside $G_t$. Nota and Thomas (2020) showed that the resulting direction is not the gradient of any objective. It often works in practice, but it is not the unbiased estimator. In this chapter's check, dropping $\gamma^t$ moves the mean by more than ten standard errors.

**Truncated episodes.** The test samples 100 steps per episode, and the rest of the return is cut off. With $\gamma = 0.9$ the missing part is worth at most $0.9^{100} \approx 3 \times 10^{-5}$ of a reward, far below the noise, but a discount near 1 would need a longer horizon.

**The loss value is not a score.** In a framework, you build `loss = -(weights * logp).mean()` and call `backward()`. That surrogate has the right gradient, but its value does not measure performance. It depends on the batch the policy itself sampled, so it moves when the policy moves and when the batch changes. Suppose every action pays 2 and the policy is uniform over three actions. The loss is $2 \log 3 = 2.197$. If the policy drifts to $[0.5, 0.3, 0.2]$, its expected loss falls to 2.059, yet the return is still exactly 2. So a falling loss can sit beside flat returns. Track the average episode return instead.

**On-policy data only.** Once you update $\theta$, old episodes come from the wrong distribution, and reusing them needs importance weights.

## Further reading

- [Williams (1992), *Simple statistical gradient-following algorithms for connectionist reinforcement learning*](https://doi.org/10.1007/BF00992696). Introduces REINFORCE and the score-function gradient for learning systems.
- [Sutton, McAllester, Singh, and Mansour (2000), *Policy gradient methods for reinforcement learning with function approximation*](https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html). Proves the policy gradient theorem, the subject of the next chapter. [Paper page](https://papers.ifkash.dev/pg-theorem/).
- [Sutton and Barto (2018), *Reinforcement Learning: An Introduction*, chapter 13](http://incompleteideas.net/book/the-book-2nd.html). Derives the theorem step by step and presents REINFORCE with the $\gamma^t$ factor.
- [Nota and Thomas (2020), *Is the policy gradient a gradient?*](https://arxiv.org/abs/1906.07073). Shows what goes wrong when implementations drop $\gamma^t$.

## The exercise

You build the tabular policy gradient from the ground up. Write `softmax_policy` and its score `log_policy_grad`, the exact objective `objective(mdp, theta)`, the returns `discounted_returns`, the one-episode estimator `reinforce_estimate`, and `reinforce_gradient`, which returns one estimate per sampled episode.

The checks confirm the softmax's values and stability, the score against finite differences and its zero mean, and $J = 2.25$ on the worked example. They check the returns and the single-episode estimate against hand computations. Finally, they run 2,000 episodes on a three-state MDP and require every component of the REINFORCE mean to sit within four standard errors of the gradient you get by nudging each logit and re-solving for $J$.
