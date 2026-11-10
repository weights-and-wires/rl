---
title: Inverse RL and GAIL
slug: 049-inverse-rl-and-gail
part: "Part 7 — Models, exploration, and fixed data"
summary: Recover a reward that explains an expert by matching feature counts, then imitate through a GAIL discriminator instead.
minutes: 45
gpu: false
objectives:
  - Run soft value iteration backward and state visitation forward to get a reward's expected feature counts.
  - Derive the maximum-entropy IRL gradient as expert counts minus expected counts, and check it by finite differences.
  - Explain why the learned reward matches the expert's behavior but not its exact reward.
  - Build the GAIL discriminator loss, its optimum rho_E / (rho_E + rho_pi), and the reward -log(1 - D).
prereqs:
  - 041-maximum-entropy-rl
  - 048-imitation-learning
exercise: 049-inverse-rl-and-gail
---

# Inverse RL and GAIL

A driver who slows near a school is not following a rule about that one street. They care about children, and that one preference explains thousands of choices on streets you never recorded. Behavior cloning copies what an expert does and never asks why. Can you recover the *why*, a reward the expert seems to be maximizing, from the demonstrations alone?

A reward is a compact, portable summary of behavior: you can plan with it in a world whose layout changed. The catch is that the question has too many answers, since a reward of zero everywhere makes every behavior optimal. This chapter resolves that with the principle of maximum entropy, building the algorithm of Ziebart and colleagues (2008). Then it turns to GAIL, which trains a classifier to tell expert from learner and uses its output as the learner's reward.

## What you already know

- Behavior cloning fits $\pi(a \mid s)$ to expert pairs, and its errors compound once the learner leaves the expert's states (see [Imitation learning](/lessons/048-imitation-learning)).
- Soft value iteration replaces the max in the Bellman backup with a log-sum-exp, and its policy is a softmax of $Q$ (see [Maximum-entropy RL](/lessons/041-maximum-entropy-rl)).
- Logistic regression trains a sigmoid classifier with cross-entropy.
- The gradient of $\log \sum_i e^{z_i}$ with respect to $z_i$ is the softmax weight of $i$.

## Why one expert fits many rewards

A *reward* here is a number $r(s)$ earned for each step spent in state $s$. A *trajectory* $\tau$ is a sequence of states and actions over a fixed *horizon* of $T$ steps. You describe each state by a *feature vector* $f(s)$ of length $K$, and you look for a reward that is linear in those features, $r(s) = \theta^\top f(s)$.

The expert's *feature counts* are the features summed along a trajectory and averaged over demonstrations, $\tilde f_E = \frac{1}{N}\sum_i \sum_t f(s_t^{(i)})$. Abbeel and Ng (2004) observed that a policy with the same expected feature counts as the expert earns the same expected return under every reward $\theta^\top f$, because expected return is $\theta$ dotted with the expected counts. So matching counts is the right target. Many rewards still explain the same behavior: adding a constant to $r$ changes nothing when every trajectory lasts $T$ steps, and [Reward design and shaping](/lessons/051-reward-design-and-shaping) shows a whole family of such changes. You cannot recover *the* reward, but you can find one whose behavior matches the expert's.

## Choose the most random explanation

Many distributions over trajectories match the expert's feature counts. The principle of maximum entropy says to pick the one that commits to nothing else, the one with the highest entropy. Its form is exponential:

$$
P_\theta(\tau) = \frac{1}{Z(\theta)} \exp\Big(\sum_{t=0}^{T-1} \theta^\top f(s_t)\Big)
$$

Read it aloud: a trajectory's probability grows exponentially with the reward it collects, and trajectories that tie on reward are equally likely. $Z(\theta)$ is the sum of the numerator over every trajectory, which makes the probabilities add to one. This formula assumes deterministic dynamics, as in the exercise. With random dynamics, Ziebart and colleagues (2010) use *causal* entropy, which is exactly the soft value iteration below. You never enumerate trajectories, because there are far too many. Instead you compute soft values backward from the horizon:

$$
Q_t(s, a) = r(s) + \sum_{s'} P(s' \mid s, a)\, V_{t+1}(s'), \qquad
V_t(s) = \log \sum_a e^{Q_t(s, a)}, \qquad V_T = 0
$$

$$
\pi_t(a \mid s) = \exp\big(Q_t(s, a) - V_t(s)\big)
$$

Read it aloud: the soft value of acting at step $t$ is this state's reward plus the expected soft value of wherever you land. The soft value of a state is a smooth maximum over actions, and the policy is a softmax of $Q$. With deterministic dynamics, multiplying these policies along a trajectory gives exactly $P_\theta(\tau)$. In numpy:

```python
Q = r[:, None] + P @ V[t + 1]              # (S, A)
V[t] = logsumexp(Q, axis=1)                # subtract the row max first
pi[t] = np.exp(Q - V[t][:, None])          # rows sum to one
```

### Work it by hand: a two-cell corridor

Take the corridor `S.`, two cells side by side, with the start on the left. From the left cell only right moves, and the other three actions bump into walls and stay. Let $r = (0, 1)$, so the right cell pays 1 per step, and use $T = 2$.

- At $t = 1$, every action is the last one, so $Q_1(s, a) = r(s)$ and $V_1 = r + \log 4 = (1.3863, 2.3863)$.
- At $t = 0$ in the left cell, the three bumping actions stay, so $Q_0 = 0 + V_1(\text{left}) = 1.3863$. Moving right gives $Q_0 = 0 + V_1(\text{right}) = 2.3863$.
- $V_0(\text{left}) = \log(3e^{1.3863} + e^{2.3863}) = \log(12 + 4e) = 3.1300$.
- $\pi_0(\text{right} \mid \text{left}) = e^{2.3863 - 3.1300} = e / (e + 3) = 0.4754$.

Check it by counting, as in Figure 1. Of the 16 action sequences, the 4 that start with right collect reward 1 and the other 12 collect 0, so the first group gets $4e / (4e + 12) = 0.4754$ of the probability.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Sixteen action sequences from the left cell: 4 begin with right and carry weight e each, 12 carry weight 1 each, so the right-first group gets 0.4754">
<defs><marker id="c043-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<text x="20.0" y="32.0" class="fig-bold" text-anchor="start">Weight of each group of action sequences (exp of return)</text>
<text x="20.0" y="80.0" class="fig-small" text-anchor="start">12 sequences that bump a wall first</text>
<rect x="250.0" y="60.0" width="270.0" height="32.0" class="fill-a"/>
<text x="530.0" y="82.0" class="fig-small" text-anchor="start">12 x 1 = 12</text>
<text x="20.0" y="140.0" class="fig-small" text-anchor="start">4 sequences that move right first</text>
<rect x="250.0" y="120.0" width="244.6" height="32.0" class="fill-b"/>
<text x="504.6" y="142.0" class="fig-small" text-anchor="start">4 x e = 10.87</text>
<rect x="60.0" y="190.0" width="262.3" height="30.0" class="fill-a"/>
<rect x="322.3" y="190.0" width="237.7" height="30.0" class="fill-b"/>
<text x="191.2" y="210.0" class="fig-small" text-anchor="middle">0.5246</text>
<text x="441.2" y="210.0" class="fig-small" text-anchor="middle">0.4754</text>
<text x="60.0" y="184.0" class="fig-small" text-anchor="start">Probability after normalizing by Z = 12 + 4e = 22.87</text>
</svg>
Figure 1. Maximum entropy gives every sequence a weight of exp(its return). Moving right first earns 1 in the second step, so those four sequences carry 0.4754 of the probability, which matches pi_0(right | left) = e / (e + 3).
```

## Run the forward pass, then match counts

Expected feature counts need to know how often the policy is in each state at each step. That is a forward pass. $D_t(s)$ is the probability of being in $s$ at step $t$:

$$
D_0 = \mu_0, \qquad D_{t+1}(s') = \sum_{s, a} D_t(s)\, \pi_t(a \mid s)\, P(s' \mid s, a)
$$

Read it aloud: start from the start distribution $\mu_0$, and push each state's probability through the policy and then through the dynamics. The expected feature counts are $\mathbb{E}_\theta[f] = \sum_t \sum_s D_t(s) f(s)$.

```python
D[t + 1] = np.einsum("s,sa,sap->p", D[t], pi[t], P)
f_expected = D.sum(axis=0) @ features       # (K,)
```

In the corridor, $D_0 = (1, 0)$ and $D_1 = (0.5246, 0.4754)$, so with one indicator feature per cell the expected counts are $(1.5246, 0.4754)$.

> [!KEY]
> A backward pass turns a reward into a policy. A forward pass turns that policy into expected feature counts. IRL runs both on every iteration.

Fit $\theta$ by maximum likelihood: make the demonstrations as probable as possible under $P_\theta$. The mean log-likelihood of $N$ demonstrations is $\mathcal{L}(\theta) = \theta^\top \tilde f_E - \log Z(\theta)$. The derivative of $\log Z(\theta)$ is the expected feature count under $P_\theta$, the same way the derivative of a log-sum-exp is a softmax average. So the gradient is

$$
\nabla_\theta \mathcal{L} = \tilde f_E - \mathbb{E}_{\tau \sim P_\theta}\Big[\sum_t f(s_t)\Big]
$$

Read it aloud: the gradient is what the expert does minus what your current reward predicts. Where the expert visits a feature more often than predicted, raise that feature's weight. At the optimum the two counts match, which is the feature-matching condition. The log-likelihood is concave in $\theta$, so gradient ascent finds the optimum:

```python
r = features @ theta
_, pi = soft_value_iteration(P, r, T)
grad = f_expert - feature_expectations(state_visitation(P, pi, mu0), features)
theta = theta + lr * grad
```

Suppose the corridor's expert data shows counts $\tilde f_E = (1.2, 0.8)$, so the expert moves right 80% of the time. At $r = (0, 1)$ the model predicts $(1.5246, 0.4754)$, so the gradient is $(-0.3246, +0.3246)$, lowering the left cell's weight and raising the right cell's. Ascent stops where $e^{\Delta} / (e^{\Delta} + 3) = 0.8$ for $\Delta = \theta_{\text{right}} - \theta_{\text{left}}$, which gives $\Delta = \log 12 = 2.4849$. Only the difference is pinned down: the pairs $(0, 2.4849)$ and $(-5, -2.5151)$ explain the data equally well.

On a small grid with one indicator feature per cell, the loop recovers an expert's counts. In Figure 2, the learned counts match the expert's to within 0.00003 on every cell.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="Expert and learned expected visits per grid cell: the two sets of bars match on all 11 cells">
<defs><marker id="c043-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<line x1="60.0" y1="260.0" x2="570.0" y2="260.0" class="stroke-grid"/>
<text x="52.0" y="264.0" class="fig-small" text-anchor="end">0</text>
<line x1="60.0" y1="185.0" x2="570.0" y2="185.0" class="stroke-grid"/>
<text x="52.0" y="189.0" class="fig-small" text-anchor="end">1</text>
<line x1="60.0" y1="110.0" x2="570.0" y2="110.0" class="stroke-grid"/>
<text x="52.0" y="114.0" class="fig-small" text-anchor="end">2</text>
<rect x="65.2" y="165.9" width="18.0" height="94.0" class="fill-a"/>
<rect x="83.2" y="166.0" width="18.0" height="94.0" class="fill-b"/>
<text x="83.2" y="276.0" class="fig-small" text-anchor="middle">(0,0)</text>
<rect x="111.5" y="195.9" width="18.0" height="64.1" class="fill-a"/>
<rect x="129.5" y="196.0" width="18.0" height="64.0" class="fill-b"/>
<text x="129.5" y="276.0" class="fig-small" text-anchor="middle">(0,1)</text>
<rect x="157.9" y="208.4" width="18.0" height="51.6" class="fill-a"/>
<rect x="175.9" y="208.4" width="18.0" height="51.6" class="fill-b"/>
<text x="175.9" y="276.0" class="fig-small" text-anchor="middle">(0,2)</text>
<rect x="204.3" y="242.6" width="18.0" height="17.4" class="fill-a"/>
<rect x="222.3" y="242.6" width="18.0" height="17.4" class="fill-b"/>
<text x="222.3" y="276.0" class="fig-small" text-anchor="middle">(0,3)</text>
<rect x="250.6" y="229.8" width="18.0" height="30.2" class="fill-a"/>
<rect x="268.6" y="229.9" width="18.0" height="30.1" class="fill-b"/>
<text x="268.6" y="276.0" class="fig-small" text-anchor="middle">(1,0)</text>
<rect x="297.0" y="226.4" width="18.0" height="33.6" class="fill-a"/>
<rect x="315.0" y="226.4" width="18.0" height="33.6" class="fill-b"/>
<text x="315.0" y="276.0" class="fig-small" text-anchor="middle">(1,2)</text>
<rect x="343.4" y="223.8" width="18.0" height="36.1" class="fill-a"/>
<rect x="361.4" y="223.9" width="18.0" height="36.1" class="fill-b"/>
<text x="361.4" y="276.0" class="fig-small" text-anchor="middle">(1,3)</text>
<rect x="389.7" y="234.9" width="18.0" height="25.1" class="fill-a"/>
<rect x="407.7" y="235.0" width="18.0" height="25.0" class="fill-b"/>
<text x="407.7" y="276.0" class="fig-small" text-anchor="middle">(2,0)</text>
<rect x="436.1" y="238.2" width="18.0" height="21.8" class="fill-a"/>
<rect x="454.1" y="238.2" width="18.0" height="21.8" class="fill-b"/>
<text x="454.1" y="276.0" class="fig-small" text-anchor="middle">(2,1)</text>
<rect x="482.5" y="221.6" width="18.0" height="38.4" class="fill-a"/>
<rect x="500.5" y="221.6" width="18.0" height="38.4" class="fill-b"/>
<text x="500.5" y="276.0" class="fig-small" text-anchor="middle">(2,2)</text>
<rect x="528.8" y="72.2" width="18.0" height="187.8" class="fill-a"/>
<rect x="546.8" y="72.2" width="18.0" height="187.8" class="fill-b"/>
<text x="546.8" y="276.0" class="fig-small" text-anchor="middle">(2,3)</text>
<text x="315.0" y="296.0" class="fig-small" text-anchor="middle">grid cell (row, column)</text>
<rect x="70.0" y="52.0" width="12.0" height="12.0" class="fill-a"/>
<text x="88.0" y="63.0" class="fig-small" text-anchor="start">expert, 500 demonstrations</text>
<rect x="70.0" y="70.0" width="12.0" height="12.0" class="fill-b"/>
<text x="88.0" y="81.0" class="fig-small" text-anchor="start">learned reward, exact expected counts</text>
<text x="546.8" y="64.2" class="fig-small" text-anchor="middle">2.50</text>
</svg>
Figure 2. Visits per cell over a horizon of 8 steps. The expert's hidden reward pays 3 per step in the bottom-right cell (2,3), which draws 2.50 visits. The learned counts sit on top of the expert's, which is what feature matching guarantees.
```

> [!KEY]
> Maximum-entropy IRL recovers a reward whose soft-optimal behavior matches the expert's feature counts. It does not recover the expert's reward itself, which the data leaves ambiguous.

## GAIL: match the expert with a classifier

IRL has an inner loop: every reward update needs a full RL solve. Ho and Ermon (2016) showed that max-entropy IRL followed by RL amounts to one goal: make the learner's *occupancy measure* $\rho_\pi(s, a)$, how often $\pi$ visits the pair $(s, a)$, match the expert's. Their method, *generative adversarial imitation learning* (GAIL), chases that goal directly.

A *discriminator* $D(s, a)$ is a classifier. It outputs the probability that a pair came from the expert rather than the learner. Train it with the usual cross-entropy:

$$
\mathcal{L}_D = -\sum_{s, a} \rho_E(s, a) \log D(s, a) - \sum_{s, a} \rho_\pi(s, a) \log\big(1 - D(s, a)\big)
$$

Read it aloud: push $D$ toward 1 on pairs the expert visits and toward 0 on pairs the learner visits, weighted by how often each visits them. With $D = \sigma(\ell)$, the gradient with respect to the logit is $\rho_\pi D - \rho_E (1 - D)$, and setting it to zero gives the best discriminator:

$$
D^*(s, a) = \frac{\rho_E(s, a)}{\rho_E(s, a) + \rho_\pi(s, a)}
$$

Read it aloud: the best classifier reports the expert's share of the visits to that pair. Its logit is $\log \rho_E - \log \rho_\pi$. At this optimum the loss equals $2 \log 2 - 2\,\mathrm{JS}(\rho_E, \rho_\pi)$, where JS is the Jensen-Shannon divergence.

The learner then takes a step of policy optimization, such as TRPO or PPO, with the reward $r(s, a) = -\log(1 - D(s, a))$, which is large where the classifier is sure the pair looks expert.

```python
loss = -(rho_e * -np.logaddexp(0, -l)).sum() - (rho_p * -np.logaddexp(0, l)).sum()
reward = np.logaddexp(0.0, l)                         # -log(1 - sigmoid(l))
```

Let $\rho_E = (0.5, 0.3, 0.2, 0)$ and $\rho_\pi = (0.1, 0.2, 0.3, 0.4)$, as in Figure 3. The best discriminator is $D^* = (0.833, 0.6, 0.4, 0)$, and the GAIL rewards are $-\log(1 - D^*) = (1.792, 0.916, 0.511, 0)$. The Jensen-Shannon divergence is 0.2215, so the best loss is $2 \log 2 - 0.4429 = 0.9433$. The fourth pair, which the expert never takes, earns reward 0, so the learner shifts its visits toward the first pair.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Four state-action pairs: expert occupancy, learner occupancy, best discriminator and GAIL reward">
<defs><marker id="c043-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<text x="160.0" y="34.0" class="fig-bold" text-anchor="middle">pair 1</text>
<text x="270.0" y="34.0" class="fig-bold" text-anchor="middle">pair 2</text>
<text x="380.0" y="34.0" class="fig-bold" text-anchor="middle">pair 3</text>
<text x="490.0" y="34.0" class="fig-bold" text-anchor="middle">pair 4</text>
<text x="10.0" y="82.0" class="fig-small" text-anchor="start">expert occupancy</text>
<rect x="120.0" y="64.0" width="35.0" height="26.0" class="fill-a"/>
<text x="161.0" y="82.0" class="fig-small" text-anchor="start">0.500</text>
<rect x="230.0" y="64.0" width="21.0" height="26.0" class="fill-a"/>
<text x="257.0" y="82.0" class="fig-small" text-anchor="start">0.300</text>
<rect x="340.0" y="64.0" width="14.0" height="26.0" class="fill-a"/>
<text x="360.0" y="82.0" class="fig-small" text-anchor="start">0.200</text>
<text x="456.0" y="82.0" class="fig-small" text-anchor="start">0</text>
<text x="10.0" y="137.0" class="fig-small" text-anchor="start">learner occupancy</text>
<rect x="120.0" y="119.0" width="7.0" height="26.0" class="fill-b"/>
<text x="133.0" y="137.0" class="fig-small" text-anchor="start">0.100</text>
<rect x="230.0" y="119.0" width="14.0" height="26.0" class="fill-b"/>
<text x="250.0" y="137.0" class="fig-small" text-anchor="start">0.200</text>
<rect x="340.0" y="119.0" width="21.0" height="26.0" class="fill-b"/>
<text x="367.0" y="137.0" class="fig-small" text-anchor="start">0.300</text>
<rect x="450.0" y="119.0" width="28.0" height="26.0" class="fill-b"/>
<text x="484.0" y="137.0" class="fig-small" text-anchor="start">0.400</text>
<text x="10.0" y="192.0" class="fig-small" text-anchor="start">best D*</text>
<rect x="120.0" y="174.0" width="58.3" height="26.0" class="fill-c"/>
<text x="184.3" y="192.0" class="fig-small" text-anchor="start">0.833</text>
<rect x="230.0" y="174.0" width="42.0" height="26.0" class="fill-c"/>
<text x="278.0" y="192.0" class="fig-small" text-anchor="start">0.600</text>
<rect x="340.0" y="174.0" width="28.0" height="26.0" class="fill-c"/>
<text x="374.0" y="192.0" class="fig-small" text-anchor="start">0.400</text>
<text x="456.0" y="192.0" class="fig-small" text-anchor="start">0</text>
<text x="10.0" y="247.0" class="fig-small" text-anchor="start">reward -log(1-D*)</text>
<rect x="120.0" y="229.0" width="62.7" height="26.0" class="fill-d"/>
<text x="188.7" y="247.0" class="fig-small" text-anchor="start">1.792</text>
<rect x="230.0" y="229.0" width="32.1" height="26.0" class="fill-d"/>
<text x="268.1" y="247.0" class="fig-small" text-anchor="start">0.916</text>
<rect x="340.0" y="229.0" width="17.9" height="26.0" class="fill-d"/>
<text x="363.9" y="247.0" class="fig-small" text-anchor="start">0.511</text>
<text x="456.0" y="247.0" class="fig-small" text-anchor="start">0</text>
</svg>
Figure 3. The best discriminator reports the expert's share of each pair, and the reward grows with it. Pair 4, which only the learner visits, gets D* = 0 and reward 0, so the learner is pushed toward pair 1.
```

> [!RECAP]
> - Soft value iteration runs backward with a log-sum-exp, and state visitation runs forward.
> - The IRL gradient is expert counts minus expected counts, and the reward is only pinned down up to ambiguity.
> - The best GAIL discriminator is $\rho_E / (\rho_E + \rho_\pi)$, and the learner's reward is $-\log(1 - D)$.

## Build it step by step

The checks test the functions in this order.

1. **`soft_value_iteration(P, r, horizon)`.** Returns `V` with shape `(T + 1, S)` and `pi` with shape `(T, S, A)`. Loop `t` backward from `T - 1` to `0`, with a stable log-sum-exp that subtracts each row's max. A plain max is ordinary value iteration, not this.
2. **`state_visitation(P, pi, mu0)`.** Returns `D` with shape `(T, S)`. Set `D[0] = mu0`, then one `np.einsum` per step. Compute `T` rows, not `T + 1`.
3. **`feature_expectations(D, features)`.** One line: `D.sum(axis=0) @ features`.
4. **`empirical_feature_counts(states, features)`.** Sum `features[states]` over time, then average over trajectories. Do not average over time too.
5. **`maxent_irl_gradient(P, features, mu0, horizon, theta, f_expert)`.** Chain the first three functions and subtract. The check compares it with finite differences of the log-likelihood, so a flipped sign shows up at once.
6. **`maxent_irl(...)`.** Start at `theta = 0` and add `lr` times the gradient `iters` times.
7. **`discriminator_loss` and `discriminator_grad`.** Use `np.logaddexp` for both log-sigmoid terms.
8. **`gail_reward(logits)`.** `np.logaddexp(0, logits)`. Computing `-np.log(1 - sigmoid(l))` directly returns infinity at large logits.

## Where this breaks

**The expert is assumed to be noisily rational.** Maximum entropy models the expert as picking trajectories in proportion to $e^{\text{return}}$. A systematically biased expert produces a reward that encodes the bias as a preference.

**GAIL's discriminator can win too fast.** A perfect classifier gives a flat, useless reward, so people add gradient penalties or input noise. Also, $-\log(1 - D)$ is always positive, so the learner is paid for surviving longer. Kostrikov and colleagues (2019) traced this bias to how absorbing states are handled.

## Further reading

- [Ziebart, Maas, Bagnell, and Dey (2008), Maximum Entropy Inverse Reinforcement Learning](https://cdn.aaai.org/AAAI/2008/AAAI08-227.pdf). The algorithm in this chapter, applied to taxi drivers' route choices.
- [Abbeel and Ng (2004), Apprenticeship Learning via Inverse Reinforcement Learning](https://doi.org/10.1145/1015330.1015430). The feature-matching argument.
- [Ho and Ermon (2016), Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476). Shows that IRL followed by RL is occupancy matching, and introduces GAIL.
- [Fu, Luo, and Levine (2018), Learning Robust Rewards with Adversarial Inverse Reinforcement Learning](https://arxiv.org/abs/1710.11248). AIRL, a discriminator whose logit is a reward that transfers to new dynamics.

## The exercise

You build maximum-entropy IRL on a small `rlenvs.gridworld` and a tabular GAIL discriminator. Write `soft_value_iteration`, `state_visitation`, `feature_expectations`, `empirical_feature_counts`, `maxent_irl_gradient`, and the gradient-ascent loop `maxent_irl`. Finish with `discriminator_loss`, `discriminator_grad`, and `gail_reward`.

The checks confirm the corridor example's numbers, compare the IRL gradient with finite differences of the demonstrations' log-likelihood, and require the loop to match the expert's feature counts within 0.02. For GAIL, they test the loss gradient by finite differences, the optimal logit $\log(\rho_E / \rho_\pi)$ with its loss of $2 \log 2 - 2\,\mathrm{JS}$, and a reward that stays finite at logits of 800.
