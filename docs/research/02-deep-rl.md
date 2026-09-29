# Deep RL algorithms: research notes for the course

This document collects source material for the deep reinforcement learning (RL) part of
the course. Each topic follows the same template:

1. **Intuition**: the core idea in plain sentences.
2. **Equations**: the key equations in KaTeX-compatible LaTeX.
3. **Worked example**: a tiny numeric example. Every number in these examples was
   recomputed with NumPy while writing this document.
4. **References**: canonical papers with year and arXiv link.
5. **Exercise**: a NumPy or CPU PyTorch exercise that runs in seconds, with checks that
   are mathematically verifiable.
6. **Pitfalls**: common mistakes and failure modes.

Anything that couldn't be confirmed against a primary source is marked **UNVERIFIED**.
Research-frontier claims are dated because they describe a moving field.

## Notation and shared conventions

The document uses the following notation throughout:

- An MDP is $(\mathcal{S}, \mathcal{A}, P, r, \gamma, \rho_0)$ with discount
  $\gamma \in [0, 1)$ and initial-state distribution $\rho_0$.
- A policy $\pi_\theta(a \mid s)$ has parameters $\theta$.
- The return from time $t$ is $G_t = \sum_{k \ge 0} \gamma^k r_{t+k}$.
- Value functions are $V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$ and
  $Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]$. The advantage is
  $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$.
- The objective is $J(\theta) = \mathbb{E}_{s_0 \sim \rho_0}[V^{\pi_\theta}(s_0)]$.
- The unnormalized discounted state visitation is
  $d^\pi(s) = \sum_{t \ge 0} \gamma^t \Pr(s_t = s \mid \pi, \rho_0)$. It sums to
  $1/(1-\gamma)$. The normalized version is $d^\pi_\rho = (1-\gamma) d^\pi$.
- The TD error is $\delta_t = r_t + \gamma (1 - d_{t+1}) V(s_{t+1}) - V(s_t)$, where
  $d_{t+1} \in \{0,1\}$ is the termination flag.

### Tabular exact quantities for checks

Many exercises in this document compare a sampled estimator against an exact quantity on
a small tabular MDP. For a finite MDP and policy $\pi$, define
$P_\pi(s, s') = \sum_a \pi(a \mid s) P(s' \mid s, a)$ and
$r_\pi(s) = \sum_a \pi(a \mid s) r(s,a)$. Then:

$$
V^\pi = (I - \gamma P_\pi)^{-1} r_\pi, \qquad
Q^\pi = r + \gamma P V^\pi, \qquad
(d^\pi)^\top = \rho_0^\top (I - \gamma P_\pi)^{-1}.
$$

These closed forms make $J(\theta) = \rho_0^\top V^\pi$ differentiable end to end, so
`torch.autograd` or central finite differences give the exact gradient. The course's
check suites can reuse one shared `tabular_mdp.py` helper that builds a seeded random MDP
(for example, 3-5 states, 2-3 actions, `numpy.random.default_rng(seed)`, transition rows
drawn from a Dirichlet distribution).

### Suggested deterministic environments

The following environments are small enough for sub-second rollouts in NumPy:

- **Chain MDP** with $N$ states, where the reward sits only at the far end. It's the
  standard hard-exploration test (see [Exploration](#exploration)).
- **Gridworld** (4x4 or 5x5) with walls and a goal, deterministic or with slip
  probability.
- **CartPole in NumPy**, which ports the Gymnasium dynamics equations directly (Euler
  integration, $\tau = 0.02$ s, pole half-length 0.5 m, force 10 N). It's vectorizable over
  a batch of environments.
- **Pendulum in NumPy**, which uses
  $\dot\theta_{t+1} = \dot\theta_t + (3g/(2l)\sin\theta_t + 3/(ml^2) u_t)\,\Delta t$,
  clipped to $[-8, 8]$, with torque $u \in [-2, 2]$ and cost
  $\theta^2 + 0.1\dot\theta^2 + 0.001u^2$. This matches Gymnasium `Pendulum-v1`.
- **Linear-quadratic regulator (LQR)**, where the optimal policy and value are known in
  closed form through the discrete Riccati equation. It's the best check for continuous
  control and for deterministic policy gradients.

Seed everything: `numpy.random.default_rng(seed)`, `torch.manual_seed(seed)`, and
`torch.use_deterministic_algorithms(True)` when the check compares exact values.

---

## Value-based deep RL

### DQN

**Intuition.** DQN approximates $Q^*(s,a)$ with a neural network and regresses it toward
the one-step Bellman optimality target. Naive online Q-learning with a neural network is
unstable for two reasons. Consecutive samples are strongly correlated, and the regression
target moves every time the network updates. DQN adds two stabilizers: an **experience
replay** buffer, which samples past transitions uniformly and breaks temporal correlation,
and a **target network** $Q_{\bar\theta}$, which is a periodically copied snapshot used to
compute targets. DQN reached human-level play on many of 49 Atari games from pixels with a
single architecture and hyperparameter set.

**Equations.** The DQN loss is:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s',d) \sim \mathcal{D}}
\Big[ \ell\big( y - Q_\theta(s,a) \big) \Big], \qquad
y = r + \gamma (1-d) \max_{a'} Q_{\bar\theta}(s', a').
$$

The Nature DQN uses the Huber loss $\ell$ (described as "error clipping"). The target
weights update as $\bar\theta \leftarrow \theta$ every $C$ steps ($C = 10^4$ in Nature
DQN), or with Polyak averaging, $\bar\theta \leftarrow \tau\theta + (1-\tau)\bar\theta$.

The Nature DQN setup uses the following settings: a 1M-transition replay buffer, RMSProp
with learning rate $2.5 \times 10^{-4}$, $\varepsilon$-greedy exploration annealed from
1.0 to 0.1 over 1M frames, frame skip 4, and 4 stacked 84x84 grayscale frames.

**Worked example.** Suppose $r = 1$, $\gamma = 0.9$, and the target network gives
$Q_{\bar\theta}(s', \cdot) = [2.5, 1.5, 2.0]$. Then $y = 1 + 0.9 \cdot 2.5 = 3.25$.

**References.**

- Mnih et al. (2013). *Playing Atari with Deep Reinforcement Learning*.
  [arXiv:1312.5602](https://arxiv.org/abs/1312.5602)
- Mnih et al. (2015). *Human-level control through deep reinforcement learning*. Nature
  518, 529-533. [doi:10.1038/nature14236](https://doi.org/10.1038/nature14236)
- Lin (1992). *Self-improving reactive agents based on reinforcement learning, planning
  and teaching*. Machine Learning 8. (Origin of experience replay.)

**Exercise: DQN on the chain or CartPole.** Implement a replay buffer as a NumPy ring
buffer and a target network in PyTorch. Use the following checks:

- The ring buffer overwrites the oldest entry after capacity is reached, and uniform
  sampling returns each index with frequency $1/N$ within a chi-squared tolerance.
- The target computation masks terminal states: `y == r` wherever `done == 1`.
- The gradient doesn't flow into the target network: after `loss.backward()`, every
  target parameter has `grad is None`.
- On a 5-state deterministic chain with known $Q^*$ (computable by value iteration), a
  tabular-feature (one-hot) DQN converges to $Q^*$ within $10^{-2}$ in under 5 s.

**Pitfalls.**

- Bootstrapping through a time-limit truncation as if it were a termination. Only true
  terminations set $d = 1$. Gymnasium separates `terminated` from `truncated` for this
  reason.
- Forgetting `torch.no_grad()` or `.detach()` on the target.
- Updating the target too often, which brings back the moving-target instability, or too
  rarely, which slows learning.
- Setting the replay buffer too small, which gives correlated samples and catastrophic
  forgetting.

### Double DQN

**Intuition.** The $\max$ over noisy estimates is biased upward because
$\mathbb{E}[\max_a \hat Q(a)] \ge \max_a \mathbb{E}[\hat Q(a)]$. Double DQN decouples
action *selection*, done by the online network, from action *evaluation*, done by the
target network.

**Equations.**

$$
y^{\text{DDQN}} = r + \gamma (1-d)\, Q_{\bar\theta}\big(s', \arg\max_{a'} Q_\theta(s', a')\big).
$$

**Worked example.** With $Q_\theta(s', \cdot) = [1.0, 3.0, 2.0]$ and
$Q_{\bar\theta}(s', \cdot) = [2.5, 1.5, 2.0]$, the online argmax is action 1, so
$y^{\text{DDQN}} = 1 + 0.9 \cdot 1.5 = 2.35$. Standard DQN would give 3.25.

**References.** van Hasselt, Guez, and Silver (2016). *Deep Reinforcement Learning with
Double Q-learning*. AAAI. [arXiv:1509.06461](https://arxiv.org/abs/1509.06461). The
tabular origin is van Hasselt (2010), *Double Q-learning*, NeurIPS.

**Exercise: overestimation bias.** Let all true $Q$ values be 0, and let estimates be
$\hat Q(a) = \epsilon_a$ with $\epsilon_a \sim \mathcal{N}(0, 1)$ for $m$ actions. Check
that the Monte Carlo mean of $\max_a \hat Q(a)$ matches the expected maximum of $m$
standard normals (for $m = 10$ it's about 1.5388; compute the reference with numerical
integration of $m\,x\,\phi(x)\Phi(x)^{m-1}$). Check that the double estimator with two
independent noise draws has mean 0 within $4\sigma/\sqrt{n}$.

**Pitfalls.** Using the target network for both selection and evaluation, which is plain
DQN. Double DQN reduces overestimation but can underestimate.

### Dueling networks

**Intuition.** In many states the choice of action barely matters. Splitting the head
into a state value $V(s)$ and an advantage $A(s,a)$ lets the network learn $V$ from every
sample, whatever action was taken.

**Equations.** The naive sum $V + A$ isn't identifiable because you can add a constant to
$V$ and subtract it from $A$. The paper uses the mean-subtracted form:

$$
Q(s,a) = V(s) + A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a').
$$

**Worked example.** With $V = 2$ and $A = [1, -1, 3]$, the mean of $A$ is 1, so
$Q = [2, 0, 4]$. Check that the mean of $Q$ equals $V$.

**References.** Wang et al. (2016). *Dueling Network Architectures for Deep
Reinforcement Learning*. ICML. [arXiv:1511.06581](https://arxiv.org/abs/1511.06581)

**Exercise.** Implement the head and assert that `Q.mean(-1) == V` and that adding a
constant $c$ to all advantages leaves $Q$ unchanged.

**Pitfalls.** Forgetting the mean subtraction, or using the max-subtraction variant
without knowing it changes the semantics (with max subtraction, $V = \max_a Q$).

### Prioritized experience replay (PER)

**Intuition.** Transitions with large TD error carry more learning signal, so PER samples
them more often. Non-uniform sampling biases the expected gradient, so importance
sampling weights correct it, and the correction is annealed to full strength by the end
of training.

**Equations.** The proportional variant uses the following sampling probability and
weights:

$$
p_i = |\delta_i| + \epsilon, \qquad
P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}, \qquad
w_i = \frac{\big(N \cdot P(i)\big)^{-\beta}}{\max_j w_j}.
$$

The paper uses $\alpha = 0.6$ and anneals $\beta$ from 0.4 to 1 for the proportional
variant, and uses $\alpha = 0.7$ and $\beta_0 = 0.5$ for the rank-based variant, where
$p_i = 1/\operatorname{rank}(i)$. A sum-tree gives $O(\log N)$ sampling and updates.

**Worked example.** With priorities $p = [2, 1, 0.5, 0.5]$, $\alpha = 0.6$, and
$\beta = 0.4$, the result is $P \approx [0.395, 0.261, 0.172, 0.172]$ and normalized
weights $w \approx [0.717, 0.847, 1.0, 1.0]$.

**References.** Schaul et al. (2016). *Prioritized Experience Replay*. ICLR.
[arXiv:1511.05952](https://arxiv.org/abs/1511.05952)

**Exercise: sum-tree.** Implement a sum-tree with the following checks:

- The root equals the sum of the leaves after arbitrary updates.
- The empirical sampling frequencies over $10^5$ draws match $P(i)$ (chi-squared test).
- With $\beta = 1$, the importance-weighted mean of a function $f(i)$ under PER sampling
  is an unbiased estimate of its uniform mean. That is,
  $\mathbb{E}_{i \sim P}[\tfrac{1}{N P(i)} f(i)] = \tfrac{1}{N}\sum_i f(i)$, which you can
  check exactly by enumeration.

**Pitfalls.**

- Forgetting to give new transitions maximal priority, so they're never sampled.
- Normalizing weights by the batch max instead of the global max, which is a common and
  mostly harmless variant, but inconsistent with the paper.
- Stale priorities that are never updated after a transition is replayed.

### n-step returns

**Intuition.** Bootstrapping after $n$ real rewards propagates reward information $n$
times faster and reduces reliance on an inaccurate $Q$. The target becomes off-policy
biased when the behavior policy differs from the greedy policy. DQN variants usually
ignore this bias for small $n$ (3 in Rainbow).

**Equations.**

$$
y^{(n)}_t = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n \max_{a'} Q_{\bar\theta}(s_{t+n}, a').
$$

If the episode terminates before $t+n$, the sum truncates and the bootstrap term is
dropped.

**Worked example.** Rewards $[1, 0, 2]$, $\gamma = 0.9$, and bootstrap value 5 give
$1 + 0 + 0.81 \cdot 2 + 0.729 \cdot 5 = 6.265$.

**References.**

- Sutton (1988). *Learning to predict by the methods of temporal differences*.
- Hessel et al. (2018). Rainbow, cited later in this section.
- For the off-policy correction alternatives, see Munos et al. (2016). *Safe and Efficient
  Off-Policy Reinforcement Learning* (Retrace). NeurIPS.
  [arXiv:1606.02647](https://arxiv.org/abs/1606.02647)

**Exercise.** Write a vectorized n-step target function and check it against a slow loop
reference on random episodes with random termination positions. Check that $n = 1$
recovers the one-step target and that $n \ge T$ with $\gamma$ bootstrap masked equals the
Monte Carlo return.

**Pitfalls.** Off-by-one errors in $\gamma^n$ when the episode ends early, and
accidentally crossing episode boundaries in the buffer.

### Distributional RL: C51 and QR-DQN

**Intuition.** Instead of learning the mean return, learn the whole return distribution
$Z(s,a)$, which satisfies a distributional Bellman equation
$Z(s,a) \overset{D}{=} R + \gamma Z(S', A')$. The mean still drives action selection, but
the richer target acts as a strong auxiliary signal and improves representations.

**C51 equations.** C51 uses a fixed support of $N = 51$ atoms,
$z_i = V_{\min} + i\,\Delta z$ with $\Delta z = (V_{\max} - V_{\min})/(N-1)$ and
$[V_{\min}, V_{\max}] = [-10, 10]$ on Atari. The network outputs probabilities
$p_i(s,a)$. The target distribution from $s'$ with greedy
$a^* = \arg\max_a \sum_i z_i p_i(s',a)$ is shifted, clipped, and projected back onto the
support:

$$
\hat{\mathcal{T}} z_j = \big[r + \gamma (1-d) z_j\big]_{V_{\min}}^{V_{\max}}, \quad
b_j = \frac{\hat{\mathcal{T}} z_j - V_{\min}}{\Delta z}, \quad
l = \lfloor b_j \rfloor, \; u = \lceil b_j \rceil,
$$

$$
m_l \mathrel{+}= p_j(s', a^*)\,(u - b_j), \qquad m_u \mathrel{+}= p_j(s', a^*)\,(b_j - l).
$$

The loss is the cross-entropy $-\sum_i m_i \log p_i(s,a)$. When $b_j$ is an integer,
$l = u$ and the whole mass must go to $m_l$. A naive implementation drops that mass.

**C51 worked example.** Take support $z = [-2, -1, 0, 1, 2]$ ($\Delta z = 1$), next-state
probabilities $p = [0.1, 0.2, 0.4, 0.2, 0.1]$, $r = 1$, and $\gamma = 0.9$. Then
$\hat{\mathcal T} z = [-0.8, 0.1, 1.0, 1.9, 2.0]$ (the last is clipped from 2.8),
$b = [1.2, 2.1, 3.0, 3.9, 4.0]$, and the projection gives
$m = [0, 0.08, 0.20, 0.44, 0.28]$. The projected mean is 0.92 rather than
$1 + 0.9 \cdot 0 = 1.0$. The gap comes entirely from clipping at $V_{\max}$, which makes a
useful test case.

**QR-DQN equations.** QR-DQN flips the parameterization: fixed probabilities $1/N$ and
learned locations $\theta_i(s,a)$ that estimate quantiles at midpoints
$\hat\tau_i = \frac{2i - 1}{2N}$. It minimizes the quantile Huber loss over all pairs of
predicted and target quantiles:

$$
\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{N} \rho^\kappa_{\hat\tau_i}\big(\mathcal{T}\theta_j - \theta_i\big), \qquad
\rho^\kappa_\tau(u) = \big|\tau - \mathbb{1}\{u < 0\}\big| \frac{L_\kappa(u)}{\kappa},
$$

where $L_\kappa$ is the Huber loss with threshold $\kappa$ (usually 1). For $N = 4$, the
midpoints are $[0.125, 0.375, 0.625, 0.875]$. QR-DQN removes the projection step and the
need to pick $V_{\min}$ and $V_{\max}$. IQN extends QR-DQN by sampling $\tau$ and
embedding it as an input.

**References.**

- Bellemare, Dabney, and Munos (2017). *A Distributional Perspective on Reinforcement
  Learning*. ICML. [arXiv:1707.06887](https://arxiv.org/abs/1707.06887)
- Dabney et al. (2018). *Distributional Reinforcement Learning with Quantile
  Regression*. AAAI. [arXiv:1710.10044](https://arxiv.org/abs/1710.10044)
- Dabney et al. (2018). *Implicit Quantile Networks for Distributional RL*. ICML.
  [arXiv:1806.06923](https://arxiv.org/abs/1806.06923)
- Bellemare, Dabney, and Rowland (2023). *Distributional Reinforcement Learning*. MIT
  Press. Free at [distributional-rl.org](https://www.distributional-rl.org/).

**Exercise: projection and quantile regression.** Use the following checks:

- The projection conserves mass: `m.sum() == 1`.
- The projected distribution has the same mean as the unprojected shifted distribution
  whenever no atom is clipped. This follows because linear interpolation onto the two
  neighbors preserves the first moment.
- The integer-$b_j$ case retains all mass.
- The previous worked example reproduces exactly.
- Quantile regression on 10,000 samples of a known distribution (for example, an
  exponential) converges to the true quantiles $F^{-1}(\hat\tau_i)$ within 0.05.

**Pitfalls.**

- The lost-mass bug when $l = u$.
- Using the target network for the greedy action in C51 without deciding whether to do
  Double-DQN-style selection.
- In QR-DQN, broadcasting the pairwise difference in the wrong orientation, so
  $\hat\tau$ indexes the target instead of the prediction.

### Noisy nets

**Intuition.** Replace $\varepsilon$-greedy with learned, state-dependent exploration by
adding parameterized Gaussian noise to the weights of the final linear layers. The
network learns to shrink the noise $\sigma$ where exploration isn't useful.

**Equations.** A noisy linear layer computes:

$$
y = (\mu^w + \sigma^w \odot \varepsilon^w)\, x + \mu^b + \sigma^b \odot \varepsilon^b.
$$

Factorized Gaussian noise uses $\varepsilon^w_{ij} = f(\varepsilon_i) f(\varepsilon_j)$
and $\varepsilon^b_j = f(\varepsilon_j)$ with $f(x) = \operatorname{sgn}(x)\sqrt{|x|}$.
The initialization is $\mu \sim \mathcal{U}[-1/\sqrt{p}, 1/\sqrt{p}]$ and
$\sigma = \sigma_0/\sqrt{p}$ with $\sigma_0 = 0.5$, where $p$ is the fan-in.

**References.** Fortunato et al. (2018). *Noisy Networks for Exploration*. ICLR.
[arXiv:1706.10295](https://arxiv.org/abs/1706.10295)

**Exercise.** Check that with $\sigma = 0$ the layer equals `nn.Linear`. Check that the
empirical output covariance over $10^4$ noise samples for a fixed input matches the
analytic covariance $\operatorname{Var}(y_j) = \sum_i (\sigma^w_{ji} x_i)^2 + (\sigma^b_j)^2$
for independent (non-factorized) noise.

**Pitfalls.** Resampling noise at the wrong frequency (the paper resamples per step for
acting and per minibatch for learning), and forgetting to disable noise at evaluation
when that's the intended protocol.

### Rainbow

**Intuition.** Rainbow combines six DQN extensions: Double DQN, prioritized replay,
dueling heads, multi-step returns ($n = 3$), distributional C51, and noisy nets. Ablations
showed that prioritized replay and multi-step returns mattered most, and that the
distributional component helped late in training. Rainbow is the standard reference
point for Atari value-based agents, and later work (BBF) builds on it.

**References.** Hessel et al. (2018). *Rainbow: Combining Improvements in Deep
Reinforcement Learning*. AAAI. [arXiv:1710.02298](https://arxiv.org/abs/1710.02298)

**Exercise.** Treat Rainbow as an integration exercise. Combine the distributional loss,
with Double-DQN action selection from the online net, and use the per-sample
cross-entropy as the new PER priority. Unit-test each part separately, then run on the
chain MDP.

**Pitfalls.** Using TD-error priorities with a distributional loss. Rainbow uses the KL
or cross-entropy loss as the priority.

---

## Policy gradient methods

### Log-derivative trick and the policy gradient theorem

**Intuition.** You can't differentiate through sampling from a discrete distribution
directly. The log-derivative (score function) trick rewrites the gradient of an
expectation as an expectation of a gradient, which you can estimate by sampling.
Actions that led to high return get their log-probability pushed up.

**Equations.** The log-derivative identity is
$\nabla_\theta p_\theta(x) = p_\theta(x) \nabla_\theta \log p_\theta(x)$, so:

$$
\nabla_\theta \mathbb{E}_{x\sim p_\theta}[f(x)] = \mathbb{E}_{x \sim p_\theta}\big[f(x)\,\nabla_\theta \log p_\theta(x)\big].
$$

For trajectories, the dynamics terms don't depend on $\theta$, so
$\nabla_\theta \log p_\theta(\tau) = \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$.
The **policy gradient theorem** states:

$$
\nabla_\theta J(\theta) = \sum_s d^{\pi}(s) \sum_a \pi_\theta(a\mid s)\, Q^{\pi}(s,a)\, \nabla_\theta \log \pi_\theta(a \mid s)
= \mathbb{E}\Big[\sum_{t\ge 0} \gamma^t\, Q^\pi(s_t,a_t)\,\nabla_\theta \log\pi_\theta(a_t\mid s_t)\Big].
$$

For a tabular softmax policy $\pi(a\mid s) \propto e^{\theta_{s,a}}$, the score is
$\partial \log \pi(a \mid s)/\partial\theta_{s,b} = \mathbb{1}[a=b] - \pi(b\mid s)$, so:

$$
\frac{\partial J}{\partial \theta_{s,a}} = d^\pi(s)\,\pi(a\mid s)\,A^\pi(s,a).
$$

**Worked example.** Take a two-armed bandit with $\theta = [0, 0]$ and rewards
$[1, 0]$. Then $\pi = [0.5, 0.5]$, $J = \pi_1$, and
$\partial J/\partial\theta_1 = \pi_1(1-\pi_1)(r_1 - r_2) = 0.25$. The score-function
estimator gives
$\mathbb{E}[r\,(\mathbb{1}[a=1] - \pi_1)] = 0.5 \cdot 1 \cdot 0.5 + 0.5 \cdot 0 \cdot (-0.5) = 0.25$.

**References.**

- Williams (1992). *Simple statistical gradient-following algorithms for connectionist
  reinforcement learning*. Machine Learning 8, 229-256.
- Sutton, McAllester, Singh, and Mansour (2000). *Policy Gradient Methods for
  Reinforcement Learning with Function Approximation*. NeurIPS.
- Nota and Thomas (2020). *Is the Policy Gradient a Gradient?* AAMAS.
  [arXiv:1906.07073](https://arxiv.org/abs/1906.07073)

**Exercise: exact versus sampled gradient on a tabular MDP.** This is the flagship
verifiable exercise. It was prototyped for this document, and it runs in under 2 s:

1. Build a seeded random MDP with 3 states, 2 actions, and $\gamma = 0.9$.
2. Compute $J(\theta)$ in closed form and its gradient by central finite differences or
   autograd.
3. Check that the policy gradient theorem formula $d^\pi(s)\pi(a\mid s)A^\pi(s,a)$ matches
   the finite-difference gradient to $10^{-8}$. In the prototype, the maximum error was
   $1.2 \times 10^{-9}$.
4. Run REINFORCE with the $\gamma^t$ weighting for 4,000 episodes of length 60. Check that
   each gradient component lies within 4 standard errors of the exact value. Compute the
   standard errors from the per-episode estimates. Don't use a fixed absolute tolerance,
   because the variance differs across components.
5. Show that dropping $\gamma^t$ gives a *different* expectation. It's the biased
   direction discussed by Nota and Thomas, and the test should detect a statistically
   significant difference on a seed where the gap is large.

**Pitfalls.**

- Dropping $\gamma^t$ from the state weighting. Almost every deep RL implementation does
  this. The resulting direction isn't the gradient of any objective (Nota and Thomas,
  2020). In practice it works as a gradient of the undiscounted-state-weighted objective
  heuristic, but the course should say so honestly.
- Sign errors, because optimizers minimize, so the loss is
  $-\log\pi \cdot \hat A$.
- Letting the return $G$ carry gradient. It must be detached.

### REINFORCE, reward-to-go, and baselines

**Intuition.** REINFORCE is the Monte Carlo version of the theorem: replace $Q^\pi$ with
the sampled return. Two variance reductions keep it unbiased. The first is *causality*:
an action can't affect past rewards, so use the reward-to-go $G_t$ rather than the full
return. The second is a *baseline*: subtract any function $b(s_t)$ that doesn't depend on
$a_t$, because $\mathbb{E}_{a\sim\pi}[\nabla\log\pi(a\mid s)] = 0$.

**Equations.**

$$
\hat g = \frac{1}{N}\sum_{i=1}^N \sum_{t} \gamma^t \big(G^{(i)}_t - b(s^{(i)}_t)\big)\nabla_\theta \log\pi_\theta(a^{(i)}_t\mid s^{(i)}_t).
$$

The baseline is unbiased because
$\sum_a \pi(a\mid s)\nabla\log\pi(a\mid s) = \nabla\sum_a \pi(a \mid s) = \nabla 1 = 0$.
The variance-minimizing constant baseline for a single-step problem with scalar
parameter is:

$$
b^* = \frac{\mathbb{E}\big[(\nabla\log\pi)^2\, R\big]}{\mathbb{E}\big[(\nabla\log\pi)^2\big]}.
$$

$V^\pi(s)$ is the usual practical choice, though not exactly optimal.

**References.**

- Williams (1992), cited earlier.
- Greensmith, Bartlett, and Baxter (2004). *Variance Reduction Techniques for Gradient
  Estimates in Reinforcement Learning*. JMLR 5.
- Tucker et al. (2018). *The Mirage of Action-Dependent Baselines in Reinforcement
  Learning*. ICML. [arXiv:1802.10031](https://arxiv.org/abs/1802.10031)

**Exercise: variance reduction.** On a bandit or a 3-state MDP, estimate the mean and the
variance of the gradient estimator with no baseline, with a constant baseline, with the
$V^\pi$ baseline, and with $b^*$. Use the following checks:

- All four means agree with the exact gradient within 4 standard errors, which confirms
  unbiasedness.
- The variance ordering holds: $b^*$ has the lowest variance among constant baselines,
  and the analytic variance formula matches the empirical variance.
- An *action-dependent* baseline $b(s,a)$ without a correction term makes the estimator
  biased, and the test detects it.

**Pitfalls.**

- Normalizing returns per batch (a common trick) changes the scale and effectively the
  baseline. That's fine as a heuristic, but it isn't the unbiased estimator.
- Letting the baseline depend on the current action.
- Fitting the baseline on the same batch without detaching.

---

## Actor-critic methods

### A2C and A3C

**Intuition.** Replace the Monte Carlo return with a learned critic to cut variance at
the cost of bias. A3C runs many actor-learners asynchronously on CPU threads, each
computing $n$-step advantage gradients and applying them to shared parameters with
Hogwild-style updates. The parallel actors decorrelate data without a replay buffer. A2C
is the synchronous, batched version. It's usually as good as or better than A3C, and it
uses GPUs well.

**Equations.** The A3C/A2C loss combines three terms with $n$-step advantage
$\hat A_t = \sum_{k=0}^{n-1}\gamma^k r_{t+k} + \gamma^n V_\phi(s_{t+n}) - V_\phi(s_t)$:

$$
\mathcal{L} = -\mathbb{E}\big[\hat A_t \log\pi_\theta(a_t\mid s_t)\big]
+ c_v\,\mathbb{E}\big[(\hat R_t - V_\phi(s_t))^2\big]
- c_e\,\mathbb{E}\big[\mathcal{H}(\pi_\theta(\cdot\mid s_t))\big],
$$

where $\hat R_t = \hat A_t + V_\phi(s_t)$ is detached. A3C used $c_e = 0.01$, $n = 5$,
and 16 actor threads.

**References.**

- Mnih et al. (2016). *Asynchronous Methods for Deep Reinforcement Learning*. ICML.
  [arXiv:1602.01783](https://arxiv.org/abs/1602.01783)
- Konda and Tsitsiklis (2000). *Actor-Critic Algorithms*. NeurIPS.
- OpenAI Baselines blog (2017). *OpenAI Baselines: ACKTR & A2C*.
  [openai.com/research/openai-baselines-acktr-a2c](https://openai.com/research/openai-baselines-acktr-a2c)

**Exercise.** Implement A2C on NumPy CartPole with 8 vectorized environments. Use
the following checks:

- The $n$-step return function matches a loop reference, including terminations mid-roll.
- The entropy of a uniform categorical over $k$ actions equals $\log k$.
- Averaged over 3 seeds, A2C reaches a mean return of at least 195 within a fixed step
  budget. Keep this as a soft check, and keep the step budget within the 60 s limit.

**Pitfalls.**

- Bootstrapping from $V(s_{t+n})$ across an episode boundary.
- A critic loss that dominates when networks are shared.
- Entropy coefficients that are too large, so the policy never commits.

### Generalized advantage estimation (GAE)

**Intuition.** GAE is an exponentially weighted average of $k$-step advantage
estimators. It's the TD($\lambda$) idea applied to advantages. $\lambda$ trades bias
against variance. With $\lambda = 0$, it's the one-step TD advantage, which is low
variance and biased by critic error. With $\lambda = 1$, it's the Monte Carlo return
minus a baseline, which is unbiased (given a correct $\gamma$) and high variance. The
discount $\gamma$ plays a *different* role: it defines the problem you're solving, and
a lower $\gamma$ reduces variance by down-weighting distant rewards but biases the
objective itself, even with a perfect critic. Schulman et al. frame
$\gamma$ as a variance-reduction parameter too. The key difference is that $\lambda < 1$
introduces bias only when $V$ is inaccurate, while $\gamma < 1$ introduces bias
regardless.

**Equations.**

$$
\hat A^{\text{GAE}(\gamma,\lambda)}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l\, \delta_{t+l}, \qquad
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).
$$

The backward recursion, with termination masking, is:

$$
\hat A_t = \delta_t + \gamma\lambda(1 - d_{t+1})\,\hat A_{t+1}.
$$

The two limiting cases are as follows:

$$
\hat A^{(\lambda=0)}_t = \delta_t, \qquad
\hat A^{(\lambda=1)}_t = \sum_{l \ge 0}\gamma^l r_{t+l} - V(s_t).
$$

The value target used in PPO implementations is $\hat R_t = \hat A_t + V(s_t)$, which is
the $\lambda$-return.

**Worked example.** Rewards $r = [1, 0, 2]$, values $V = [0.5, 1.0, 1.5]$, a terminal
state after the third step ($V = 0$), $\gamma = 0.9$, and $\lambda = 0.8$ give
$\delta = [1.4, 0.35, 0.5]$ and $\hat A = [1.9112, 0.71, 0.5]$. With $\lambda = 1$, the
result is $\hat A = [2.12, 0.8, 0.5]$, which equals $G - V = [2.62 - 0.5, 1.8 - 1.0, 2.0 - 1.5]$.

**References.** Schulman, Moritz, Levine, Jordan, and Abbeel (2016). *High-Dimensional
Continuous Control Using Generalized Advantage Estimation*. ICLR.
[arXiv:1506.02438](https://arxiv.org/abs/1506.02438)

**Exercise: GAE identities.** Implement `gae(rewards, values, dones, last_value, gamma,
lam)` for a batch of vectorized rollouts. Use the following checks:

- `lam=1` equals the Monte Carlo return minus $V$, exactly (to float tolerance).
- `lam=0` equals $\delta$.
- For intermediate $\lambda$, the result equals the explicit weighted sum
  $(1-\lambda)\sum_{k\ge1}\lambda^{k-1}\hat A^{(k)}_t$ of $k$-step advantages, with the
  tail mass assigned to the longest estimator at episode end.
- The previous worked example reproduces exactly.
- With a perfect $V = V^\pi$ on a tabular MDP, the expected GAE is $A^\pi(s,a)$ for
  *every* $\lambda$, checked by averaging over many rollouts from a fixed $(s,a)$.

**Pitfalls.**

- Using `done[t]` where `done[t+1]` is needed. The indexing convention differs between
  codebases, which is the most frequent GAE bug.
- Not bootstrapping from the value of the final observation for truncated rollouts.
- Treating truncation as termination.
- Normalizing advantages before computing value targets, which corrupts the targets.

---

## Trust regions: natural policy gradient, TRPO, and PPO

### Natural policy gradient

**Intuition.** Vanilla gradient ascent measures step size in parameter space, which
depends on how you parameterize the policy. The natural gradient measures step size in
distribution space with the KL divergence, whose local quadratic approximation is the
Fisher information matrix. The update direction becomes $F^{-1}\nabla J$, which is
invariant to reparameterization.

**Equations.**

$$
F(\theta) = \mathbb{E}_{s\sim d^\pi_\rho,\, a\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a\mid s)\nabla_\theta\log\pi_\theta(a\mid s)^\top\big], \qquad
\bar{D}_{\mathrm{KL}}(\theta \,\|\, \theta + \Delta) \approx \tfrac12 \Delta^\top F \Delta.
$$

Maximizing $g^\top\Delta$ subject to $\frac12\Delta^\top F\Delta \le \delta$ gives:

$$
\Delta^* = \sqrt{\frac{2\delta}{g^\top F^{-1} g}}\; F^{-1} g.
$$

For the tabular softmax policy, the natural gradient has a closed form (Agarwal et al.,
2021): $F^{\dagger}\nabla_\theta V^{\pi}(\rho) = \frac{1}{1-\gamma}A^\pi(s,a)$, up to an
additive per-state constant that doesn't change the policy. So NPG with step size $\eta$
is $\pi_{t+1}(a\mid s)\propto \pi_t(a\mid s)\exp\big(\tfrac{\eta}{1-\gamma}A^{\pi_t}(s,a)\big)$,
a soft policy iteration step. This identity was checked numerically for this document.
The residual was constant within each state to $10^{-14}$.

**References.**

- Kakade (2001). *A Natural Policy Gradient*. NeurIPS.
- Amari (1998). *Natural Gradient Works Efficiently in Learning*. Neural Computation.
- Agarwal, Kakade, Lee, and Mahajan (2021). *On the Theory of Policy Gradient Methods:
  Optimality, Approximation, and Distribution Shift*. JMLR.
  [arXiv:1908.00261](https://arxiv.org/abs/1908.00261)

**Exercise.** On a tabular MDP, build $F$ explicitly as a $(|S||A|)^2$ matrix. Compute
`pinv(F) @ grad` and check that it equals $A^\pi/(1-\gamma)$ plus a per-state constant:
`(w - A/(1-g)).std(axis=1) < 1e-8`. Note the normalization. Use the normalized
$d^\pi_\rho$ in $F$ and the unnormalized $d^\pi$ in the gradient.

**Pitfalls.** $F$ is singular for softmax policies because adding a constant to all
logits in a state changes nothing. Use a pseudo-inverse or damping.

### TRPO

**Intuition.** TRPO turns the natural gradient into a practical algorithm with a
monotonic-improvement flavor. It maximizes an importance-weighted surrogate subject to an
average KL constraint. It solves $Fx = g$ approximately with conjugate gradient (CG),
using only Fisher-vector products and never forming $F$. Then it backtracks along the
step until the surrogate improves and the KL constraint holds.

**Equations.** The surrogate problem is:

$$
\max_\theta\; L_{\theta_{\text{old}}}(\theta) = \mathbb{E}_{s,a\sim\pi_{\text{old}}}\Big[\frac{\pi_\theta(a\mid s)}{\pi_{\theta_{\text{old}}}(a\mid s)}\hat A(s,a)\Big]
\quad \text{s.t.}\quad \mathbb{E}_{s}\big[D_{\mathrm{KL}}(\pi_{\theta_{\text{old}}}(\cdot\mid s)\,\|\,\pi_\theta(\cdot\mid s))\big] \le \delta.
$$

The theory behind it is the following lower bound (Kakade and Langford, 2002; Schulman
et al., 2015):

$$
\eta(\tilde\pi) \ge L_\pi(\tilde\pi) - \frac{4\epsilon\gamma}{(1-\gamma)^2} D^{\max}_{\mathrm{KL}}(\pi,\tilde\pi), \qquad \epsilon = \max_{s,a}|A^\pi(s,a)|.
$$

The **Fisher-vector product** is computed with double backpropagation through the mean
KL, where $\theta_{\text{old}}$ is held fixed at the current $\theta$:

$$
F v = \nabla_\theta\Big( \big(\nabla_\theta \bar D_{\mathrm{KL}}(\theta_{\text{old}}\,\|\,\theta)\big)^\top v \Big)\Big|_{\theta=\theta_{\text{old}}}.
$$

Implementations add damping $(F + \xi I)v$ with $\xi \approx 0.1$. The algorithm runs
about 10 CG iterations to get $x \approx F^{-1}g$, sets the full step
$\beta x$ with $\beta = \sqrt{2\delta/(x^\top F x)}$, and backtracks with factor 0.5
until the KL is at most $\delta$ and the surrogate improves. The standard setting is
$\delta = 0.01$.

**References.**

- Schulman, Levine, Moritz, Jordan, and Abbeel (2015). *Trust Region Policy
  Optimization*. ICML. [arXiv:1502.05477](https://arxiv.org/abs/1502.05477)
- Kakade and Langford (2002). *Approximately Optimal Approximate Reinforcement
  Learning*. ICML.

**Exercise: CG and Fisher-vector products.** Use the following checks:

- CG on a random symmetric positive definite (SPD) $10\times10$ matrix converges to
  `np.linalg.solve` within $10^{-6}$ in at most 10 iterations. In exact arithmetic, CG
  converges in at most $n$ iterations.
- For a small PyTorch categorical policy, the double-backprop FVP equals the explicitly
  built Fisher matrix times $v$, to $10^{-5}$. Build the Fisher matrix as the expectation
  of outer products of score vectors, computed exactly by enumerating actions.
- The FVP also equals the Hessian of the mean KL at $\theta_{\text{old}}$ times $v$,
  computed with `torch.autograd.functional.hessian`.
- After the line search, the measured KL is at most $\delta$.

**Pitfalls.**

- Computing the KL in the wrong direction.
- Forgetting to detach $\pi_{\text{old}}$ inside the KL. The gradient of the KL at
  $\theta = \theta_{\text{old}}$ is zero, but its Hessian isn't.
- Evaluating the FVP on a different batch than the gradient.
- Omitting damping, which causes CG blow-ups.

### PPO

**Intuition.** PPO gets most of TRPO's stability with first-order optimization. It
clips the probability ratio so the objective gives no incentive to move the ratio
outside $[1-\epsilon, 1+\epsilon]$ in the direction the advantage favors. Taking the
minimum with the unclipped term makes the objective a pessimistic lower bound. The
algorithm then runs several epochs of minibatch SGD on each batch of rollouts.

**Equations.** With $r_t(\theta) = \pi_\theta(a_t\mid s_t)/\pi_{\theta_{\text{old}}}(a_t\mid s_t)$,
the clipped surrogate is:

$$
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t\Big[\min\big(r_t(\theta)\hat A_t,\; \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat A_t\big)\Big].
$$

The full objective to *maximize* is:

$$
L^{\text{CLIP+VF+S}}_t = \hat{\mathbb{E}}_t\big[L^{\text{CLIP}}_t - c_1 L^{\text{VF}}_t + c_2\,\mathcal{H}[\pi_\theta](s_t)\big].
$$

Value clipping, which appears in the OpenAI Baselines implementation and not in the
paper's main algorithm, is:

$$
L^{\text{VF}} = \max\Big[\big(V_\phi(s_t) - \hat R_t\big)^2,\; \big(\operatorname{clip}(V_\phi(s_t), V_{\text{old}}(s_t) - \epsilon, V_{\text{old}}(s_t)+\epsilon) - \hat R_t\big)^2\Big].
$$

Baselines multiplies this by 0.5. The paper also proposes an adaptive-KL-penalty
variant, which performed worse than clipping in its experiments.

**Worked example.** Take $\epsilon = 0.2$ and the following four cases:

| Ratio $r$ | Advantage $\hat A$ | $r\hat A$ | $\operatorname{clip}(r)\hat A$ | $L^{\text{CLIP}}$ | Gradient w.r.t. $r$ |
|---|---|---|---|---|---|
| 1.3 | 2 | 2.6 | 2.4 | 2.4 | 0 (clipped) |
| 0.7 | -1 | -0.7 | -0.8 | -0.8 | 0 (clipped) |
| 1.3 | -1 | -1.3 | -1.2 | -1.3 | -1 (not clipped) |
| 0.7 | 2 | 1.4 | 1.6 | 1.4 | 2 (not clipped) |

The last two rows are the important lesson: the clip only removes the incentive to move
further in the *favorable* direction. It never stops the objective from pulling a bad
ratio back.

For value clipping with $V_{\text{old}} = 1.0$, $V = 1.5$, $\hat R = 2.0$, and
$\epsilon = 0.2$, the loss is $\max(0.25, (1.2-2)^2) = 0.64$.

**The implementation details.** Huang et al. (2022) catalog 37 details that CleanRL uses
to match OpenAI Baselines' `ppo2`. The 13 core details are as follows:

1. Vectorized environments ($N$ environments by $M$ steps per rollout).
2. Orthogonal initialization: hidden layers with gain $\sqrt2$, policy head gain 0.01,
   value head gain 1, and zero biases.
3. Adam $\epsilon = 10^{-5}$ rather than PyTorch's default $10^{-8}$.
4. Linear learning-rate annealing to 0 (from $2.5\times10^{-4}$ on Atari and
   $3\times10^{-4}$ on MuJoCo).
5. GAE with $\lambda = 0.95$ and $\gamma = 0.99$, bootstrapping from the value of the
   final observation.
6. Shuffled minibatches over the $N \times M$ batch, with several epochs.
7. Per-minibatch advantage normalization.
8. The clipped surrogate ($\epsilon = 0.1$ on Atari and $0.2$ on MuJoCo).
9. Value-loss clipping.
10. The overall loss: policy loss, minus entropy coefficient times entropy, plus value
    coefficient times value loss (entropy coefficient 0.01 on Atari and 0 on MuJoCo; value
    coefficient 0.5).
11. Global gradient-norm clipping at 0.5.
12. Debug variables: `approx_kl`, `clipfrac`, policy loss, value loss, and entropy.
13. Shared or separate networks. Continuous control uses separate 64x64 tanh MLPs.

The continuous-action details include a state-independent learnable log standard
deviation initialized to 0, action clipping to bounds (storing the *unclipped* action for
the log-prob), running observation normalization clipped to $[-10, 10]$, and reward
scaling by the running standard deviation of the discounted return, clipped to
$[-10, 10]$. The Atari details are the standard wrappers: no-op reset, frame skip 4
with max-pooling, episodic life, fire reset, 84x84 grayscale, reward clipping to
$\{-1, 0, 1\}$, and a 4-frame stack. The defaults on Atari are 128 steps, 4 minibatches,
and 4 epochs. On MuJoCo, they're 2048 steps, 32 minibatches, and 10 epochs.

**Findings from "Implementation Matters."** Engstrom et al. (2020) showed that
code-level optimizations explain much of PPO's advantage over TRPO. These optimizations
include value clipping, reward scaling, orthogonal initialization, learning-rate
annealing, observation normalization and clipping, tanh activations, and gradient
clipping. They also showed that PPO's clipping alone doesn't keep the ratio in the trust
region: ratios routinely exceed $1 \pm \epsilon$. The paper's "PPO-M" (minimal PPO)
performs noticeably worse. Andrychowicz et al. (2021) ran more than 250,000 agents and
found, among other things, that value clipping didn't help or slightly hurt, and that
observation normalization and a careful policy initialization matter.

**References.**

- Schulman, Wolski, Dhariwal, Radford, and Klimov (2017). *Proximal Policy Optimization
  Algorithms*. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Huang, Dossa, Raffin, Kanervisto, and Wang (2022). *The 37 Implementation Details of
  Proximal Policy Optimization*. ICLR Blog Track.
  [iclr-blog-track.github.io](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- Engstrom et al. (2020). *Implementation Matters in Deep Policy Gradients: A Case Study
  on PPO and TRPO*. ICLR. [arXiv:2005.12729](https://arxiv.org/abs/2005.12729)
- Andrychowicz et al. (2021). *What Matters in On-Policy Reinforcement Learning? A
  Large-Scale Empirical Study*. ICLR. [arXiv:2006.05990](https://arxiv.org/abs/2006.05990)
- Shao et al. (2024). *DeepSeekMath* (introduces GRPO, a critic-free PPO variant with
  group-relative advantages, now standard in LLM RL).
  [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)

**Exercise: PPO loss and diagnostics.** Implement `ppo_loss(logp_new, logp_old, adv,
eps)`. Use the following checks:

- The four table rows reproduce exactly, and autograd gradients with respect to
  `logp_new` are zero exactly for the clipped rows.
- At $\theta = \theta_{\text{old}}$, the gradient of $L^{\text{CLIP}}$ equals the vanilla
  policy gradient $\hat{\mathbb{E}}[\hat A\nabla\log\pi]$.
- The `approx_kl` estimator $\hat{\mathbb{E}}[(r-1) - \log r]$ (the "k3" estimator from
  Schulman's blog) is non-negative and unbiased for the KL: compare with the exact
  categorical KL.
- `clipfrac` equals the fraction of $|r - 1| > \epsilon$.
- The value-clipping example gives 0.64.
- PPO on NumPy CartPole reaches 475 or more within the time budget on 2 of 3 seeds.

**Pitfalls.**

- Computing `logp_old` with the *current* network after an update, so the ratio is
  always 1.
- Advantage normalization over the whole batch versus per minibatch. Both are used, but
  pick one.
- Storing clipped actions and computing log-probs of the clipped action.
- Forgetting learning-rate annealing when replicating results.
- Too many epochs, where the KL blows up. Monitor `approx_kl` and consider early
  stopping at a target KL.
- Entropy bonus on a squashed distribution without a closed-form entropy. Use a
  sample estimate instead.

---

## Deterministic policy gradients: DPG, DDPG, and TD3

### Deterministic policy gradient and DDPG

**Intuition.** For continuous actions, a deterministic policy $\mu_\theta(s)$ can
improve by moving its action along the critic's action-gradient. The gradient doesn't
need an integral over actions, which makes it off-policy friendly. DDPG is DPG with the
DQN stabilizers: replay, target networks with Polyak averaging, and exploration noise
added to the actions.

**Equations.** The deterministic policy gradient theorem states:

$$
\nabla_\theta J(\mu_\theta) = \mathbb{E}_{s\sim \rho^\mu}\Big[\nabla_\theta\mu_\theta(s)\,\nabla_a Q^\mu(s,a)\big|_{a=\mu_\theta(s)}\Big].
$$

The DDPG critic target is
$y = r + \gamma(1-d)\,Q_{\bar\phi}(s', \mu_{\bar\theta}(s'))$. The actor loss is
$-\mathbb{E}_s[Q_\phi(s,\mu_\theta(s))]$. Targets update with Polyak averaging
$\bar\phi \leftarrow \tau\phi + (1-\tau)\bar\phi$ with $\tau = 0.001$. The original paper
uses Ornstein-Uhlenbeck noise ($\theta = 0.15$, $\sigma = 0.2$). Later work found plain
Gaussian noise works as well.

**References.**

- Silver et al. (2014). *Deterministic Policy Gradient Algorithms*. ICML.
  [PMLR v32](https://proceedings.mlr.press/v32/silver14.html)
- Lillicrap et al. (2016). *Continuous control with deep reinforcement learning*. ICLR.
  [arXiv:1509.02971](https://arxiv.org/abs/1509.02971)

**Exercise: DPG on LQR.** For scalar LQR $s' = as + bu$ with cost $qs^2 + ru^2$ and
linear policy $u = -ks$, the value and its derivative with respect to $k$ are analytic.
Use the following checks:

- The DPG formula, with the exact $Q^\mu$ computed from the closed-form quadratic value,
  equals $dJ/dk$ from finite differences.
- DDPG with a quadratic-feature critic converges to the Riccati-optimal gain $k^*$
  within 5%.

**Pitfalls.**

- DDPG overestimates $Q$ and is brittle to hyperparameters.
- Letting actor gradients flow into the critic parameters.
- Forgetting to clip actions to bounds after adding noise.

### TD3

**Intuition.** TD3 fixes DDPG's overestimation and instability with three changes.
**Clipped double-Q learning** uses the minimum of two critics in the target.
**Target policy smoothing** adds clipped noise to the target action so the critic can't
exploit narrow peaks. **Delayed policy updates** update the actor and targets less
often than the critics.

**Equations.**

$$
\tilde a = \operatorname{clip}\big(\mu_{\bar\theta}(s') + \operatorname{clip}(\varepsilon, -c, c),\, a_{\text{low}}, a_{\text{high}}\big), \quad \varepsilon\sim\mathcal N(0,\sigma^2),
$$

$$
y = r + \gamma(1-d)\min_{i=1,2} Q_{\bar\phi_i}(s', \tilde a).
$$

Both critics regress to the same $y$. The actor maximizes $Q_{\phi_1}(s, \mu_\theta(s))$
every $d$ critic steps. The defaults are $\sigma = 0.2$, $c = 0.5$, $d = 2$,
$\tau = 0.005$, and exploration noise $\mathcal N(0, 0.1^2)$ (scaled to the action range).

**Worked example.** With $Q_{\bar\phi_1} = 3.0$, $Q_{\bar\phi_2} = 2.5$, $r = 1$, and
$\gamma = 0.99$, the result is $y = 1 + 0.99 \cdot 2.5 = 3.475$.

**References.** Fujimoto, van Hoof, and Meger (2018). *Addressing Function Approximation
Error in Actor-Critic Methods*. ICML. [arXiv:1802.09477](https://arxiv.org/abs/1802.09477)

**Exercise.** Use the following checks:

- The target noise never exceeds $c$ in absolute value, and the final action stays in
  bounds.
- Overestimation experiment: with two critics equal to true $Q$ plus independent noise,
  the expected single-critic target overestimates. The clipped double target is at most
  the true value in expectation. Check both signs statistically.
- The actor parameters change only on steps divisible by $d$.
- TD3 solves NumPy Pendulum (mean return above -200) within about 15k steps on CPU in
  under 60 s. This is a soft check, and the budget is UNVERIFIED for pure NumPy or CPU
  torch.

**Pitfalls.**

- Applying the smoothing noise to the *current* action in the actor loss.
- Using $\min(Q_1, Q_2)$ in the actor loss. The paper uses $Q_1$, though using the min or
  mean is a known variant.
- Delaying target updates but not actor updates, or the reverse.

---

## Maximum-entropy RL and SAC

**Intuition.** Maximum-entropy RL adds the policy's entropy to the reward, so the agent
succeeds while acting as randomly as possible. This gives robust exploration, multi-modal
behavior, and smoother optimization. The soft Bellman equations replace the hard max with
a log-sum-exp, and the optimal policy is a Boltzmann distribution over soft Q values.
SAC is an off-policy actor-critic that approximates soft policy iteration with a
reparameterized Gaussian actor, twin critics (clipped double Q, as in TD3), and a
temperature $\alpha$ that's tuned automatically to hit a target entropy.

**Equations.** The objective is:

$$
J(\pi) = \sum_t \mathbb{E}\big[r(s_t,a_t) + \alpha\,\mathcal H(\pi(\cdot\mid s_t))\big].
$$

The soft Bellman equations for a fixed policy are:

$$
Q(s,a) = r(s,a) + \gamma\,\mathbb{E}_{s'}\big[V(s')\big], \qquad
V(s) = \mathbb{E}_{a\sim\pi}\big[Q(s,a) - \alpha\log\pi(a\mid s)\big].
$$

The optimal soft policy and value are:

$$
\pi^*(a\mid s) = \exp\Big(\frac{Q^*(s,a) - V^*(s)}{\alpha}\Big), \qquad
V^*(s) = \alpha\log\sum_a \exp\big(Q^*(s,a)/\alpha\big).
$$

As $\alpha \to 0$, this recovers the hard max. The SAC losses are as follows. The critic
target is:

$$
y = r + \gamma(1-d)\Big(\min_{i}Q_{\bar\phi_i}(s', \tilde a') - \alpha\log\pi_\theta(\tilde a'\mid s')\Big), \quad \tilde a'\sim\pi_\theta(\cdot\mid s').
$$

The actor loss uses reparameterization $a = f_\theta(\xi; s)$:

$$
\mathcal L_\pi = \mathbb{E}_{s,\xi}\big[\alpha\log\pi_\theta(f_\theta(\xi;s)\mid s) - \min_i Q_{\phi_i}(s, f_\theta(\xi;s))\big].
$$

The temperature loss for automatic tuning, with target entropy
$\bar{\mathcal H} = -\dim(\mathcal A)$, is:

$$
\mathcal L(\alpha) = \mathbb{E}_{a\sim\pi}\big[-\alpha\log\pi(a\mid s) - \alpha\bar{\mathcal H}\big].
$$

Implementations optimize $\log\alpha$ for positivity. If the entropy falls below the
target, the gradient increases $\alpha$.

**The tanh-squash log-prob correction.** SAC samples $u\sim\mathcal N(\mu,\sigma^2)$ and
outputs $a = \tanh(u)$. By the change of variables formula, with
$da/du = 1 - \tanh^2(u)$ elementwise:

$$
\log\pi(a\mid s) = \log\mathcal N(u;\mu,\sigma^2) - \sum_{i=1}^{D}\log\big(1-\tanh^2(u_i)\big).
$$

The numerically stable form, used in Spinning Up and CleanRL-style code, is:

$$
\log\big(1-\tanh^2(u)\big) = 2\big(\log 2 - u - \operatorname{softplus}(-2u)\big).
$$

For action bounds $[l, h]$ with $a' = \frac{h-l}{2}a + \frac{h+l}{2}$, subtract an extra
$\sum_i\log\frac{h_i - l_i}{2}$.

**Worked example.** Soft value: with $Q = [1, 2]$ and $\alpha = 0.5$,
$V^* = 0.5\log(e^2 + e^4) = 2.0635$ and $\pi^* = [0.1192, 0.8808]$. Check that
$\sum_a\pi(Q - \alpha\log\pi) = 2.0635$ as well.

Tanh correction: with $u = 0.5$, $\mu = 0$, and $\sigma = 1$, the Gaussian log-density
is $-1.0439$, $a = \tanh(0.5) = 0.4621$, and $\log(1-a^2) = -0.2402$. Both the naive and
stable formulas agree, so $\log\pi(a) = -1.0439 + 0.2402 = -0.8037$.

**References.**

- Haarnoja, Zhou, Abbeel, and Levine (2018). *Soft Actor-Critic: Off-Policy Maximum
  Entropy Deep RL with a Stochastic Actor*. ICML.
  [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)
- Haarnoja et al. (2018). *Soft Actor-Critic Algorithms and Applications*. (Automatic
  temperature, removes the separate V network.)
  [arXiv:1812.05905](https://arxiv.org/abs/1812.05905)
- Haarnoja, Tang, Abbeel, and Levine (2017). *Reinforcement Learning with Deep
  Energy-Based Policies* (soft Q-learning). ICML.
  [arXiv:1702.08165](https://arxiv.org/abs/1702.08165)
- Ziebart (2010). *Modeling Purposeful Adaptive Behavior with the Principle of Maximum
  Causal Entropy*. PhD thesis, CMU.
- Levine (2018). *Reinforcement Learning and Control as Probabilistic Inference*.
  [arXiv:1805.00909](https://arxiv.org/abs/1805.00909)

**Exercise: soft RL identities and the squash.** Use the following checks:

- **Tanh correction by numerical change of variables.** Sample $10^6$ values of
  $u\sim\mathcal N(\mu,\sigma^2)$ in 1-D, histogram $a = \tanh(u)$, and compare the
  histogram density with $\exp(\log\pi(a))$ at bin centers (relative error below 2% away
  from $\pm1$). Also check that `torch.distributions.TransformedDistribution(Normal,
  TanhTransform())` gives the same log-prob, and that the stable and naive formulas agree
  to $10^{-6}$ for $|u| < 5$ while the naive one returns `-inf` for large $|u|$ in
  float32.
- **Soft value iteration on a tabular MDP.** Show that it converges to a fixed point
  satisfying both soft Bellman equations. As $\alpha\to0$, $V_{\text{soft}}\to V^*$ from
  standard value iteration, and $V_{\text{soft}} - V^* \le \alpha\log|\mathcal A|/(1-\gamma)$.
- **Temperature gradient sign.** Check that the gradient of the temperature loss with
  respect to $\log\alpha$ has sign $\operatorname{sign}(\mathcal H - \bar{\mathcal H})$,
  so a minimizing step raises $\alpha$ when entropy is too low.

**Pitfalls.**

- Omitting the tanh correction, which causes a large bias in entropy, runaway $\alpha$,
  and poor learning. It's the most common SAC bug.
- Summing the correction over the wrong axis.
- Computing $\log\pi$ from the clipped action with `atanh`. It's unstable, so keep the
  pre-squash $u$ instead.
- Using $\alpha$ as a tensor that requires gradients inside the actor and critic losses.
  Detach it.
- A target entropy of $-\dim\mathcal A$ is a heuristic. Discrete SAC needs a different
  target, such as a fraction of $\log|\mathcal A|$.

---

## Model-based RL

**Intuition.** Learn a model of dynamics (and reward), then use it to plan, to generate
synthetic experience, or to backpropagate value gradients. Model-based methods are often
much more sample-efficient. The central danger is *model exploitation*: the policy
finds actions where the model is wrong in its favor. Uncertainty estimation (ensembles),
short rollouts, and planning in latent space are the main defenses.

### Dyna

Dyna interleaves real updates with $k$ planning updates on transitions sampled from a
learned model. Dyna-Q on a tabular maze is the classic demonstration: more planning
steps per real step speeds learning dramatically.

- Sutton (1990). *Integrated Architectures for Learning, Planning, and Reacting Based on
  Approximating Dynamic Programming*. ICML.
- Sutton (1991). *Dyna, an Integrated Architecture for Learning, Planning, and Reacting*.
  SIGART Bulletin 2(4).

**Exercise.** Dyna-Q on a deterministic gridworld. With a deterministic tabular model,
check that $k$ planning steps per real step reduce the episodes-to-optimal count
monotonically on average over seeds, and that the model's predictions match the true
transitions exactly for visited pairs.

### PETS

PETS learns an ensemble of probabilistic neural networks, each outputting a Gaussian
over $s_{t+1}$. That captures both aleatoric uncertainty (the predicted variance) and
epistemic uncertainty (ensemble disagreement). PETS plans with the cross-entropy method
(CEM) in a model-predictive control (MPC) loop and propagates uncertainty by *trajectory
sampling* (TS1 resamples the ensemble member each step, and TS$\infty$ fixes one member
per particle).

- Chua, Calandra, McAllister, and Levine (2018). *Deep Reinforcement Learning in a
  Handful of Trials using Probabilistic Dynamics Models*. NeurIPS.
  [arXiv:1805.12114](https://arxiv.org/abs/1805.12114)

**Exercise.** Implement CEM on a known quadratic objective and check convergence to the
optimum. Then run CEM-MPC with the *true* NumPy pendulum model and check that it swings
up. Then substitute a fitted ensemble of linear-feature models.

### MBPO

MBPO runs short $k$-step *branched* rollouts from real replay states using a
probabilistic ensemble, adds them to a model buffer, and trains SAC on a mixture
dominated by model data. The analysis bounds the return gap in terms of model error and
policy shift, and justifies short rollouts because compounding error grows with $k$.

- Janner, Fu, Zhang, and Levine (2019). *When to Trust Your Model: Model-Based Policy
  Optimization*. NeurIPS. [arXiv:1906.08253](https://arxiv.org/abs/1906.08253)
- The real-data mixing ratio of 5% is from the reference implementation. It's
  UNVERIFIED against the paper text.

### World Models and Dreamer v1-v3

World Models (Ha and Schmidhuber, 2018) trains a variational autoencoder (VAE) (V) that
compresses frames, a mixture density network recurrent neural network (MDN-RNN) (M) that
predicts the next latent, and a tiny linear controller (C) trained by the covariance
matrix adaptation evolution strategy (CMA-ES), including entirely inside the "dream."
[arXiv:1803.10122](https://arxiv.org/abs/1803.10122)

The Dreamer line learns a recurrent state-space model (RSSM) with a deterministic
recurrent state $h_t$ and a stochastic state $z_t$. It trains the actor and critic
purely on imagined latent trajectories. The critic learns the $\lambda$-return:

$$
V^\lambda_t = r_t + \gamma_t\big((1-\lambda)\,v(s_{t+1}) + \lambda V^\lambda_{t+1}\big),
$$

where $\gamma_t$ includes the predicted continuation probability.

- **PlaNet** (Hafner et al., 2019) introduced the RSSM with CEM planning.
  [arXiv:1811.04551](https://arxiv.org/abs/1811.04551)
- **DreamerV1** (Hafner et al., 2020, ICLR) trains the actor by backpropagating analytic
  value gradients through the learned dynamics.
  [arXiv:1912.01603](https://arxiv.org/abs/1912.01603)
- **DreamerV2** (Hafner et al., 2021, ICLR) uses 32 categorical latents with 32 classes
  each, trained with straight-through gradients, and KL balancing with weight 0.8. It
  reached human-level Atari at 200M frames.
  [arXiv:2010.02193](https://arxiv.org/abs/2010.02193)
- **DreamerV3** (Hafner, Pasukonis, Ba, and Lillicrap) uses fixed hyperparameters across
  more than 150 tasks and was the first to collect diamonds in Minecraft from scratch
  without human data. It appeared in Nature 640, 647-653 (2025).
  [arXiv:2301.04104](https://arxiv.org/abs/2301.04104),
  [doi:10.1038/s41586-025-08744-2](https://doi.org/10.1038/s41586-025-08744-2)
- **Dreamer 4** (Hafner, Yan, and Lillicrap, September 2025) uses a transformer world
  model trained with a "shortcut forcing" objective that runs in real time on one GPU. It
  was the first agent to obtain Minecraft diamonds purely from an offline dataset.
  [arXiv:2509.24527](https://arxiv.org/abs/2509.24527)

The DreamerV3 robustness techniques are good lesson material because each is a small,
testable function:

- **Symlog** is $\operatorname{symlog}(x) = \operatorname{sign}(x)\ln(|x|+1)$, with
  inverse $\operatorname{symexp}(x) = \operatorname{sign}(x)(e^{|x|}-1)$. For example,
  $\operatorname{symlog}(100) = 4.615$.
- **Two-hot encoding** puts reward and value targets onto 255 exponentially spaced bins,
  spanning $[-20, 20]$ in symlog space, and trains them with cross-entropy.
- **Return normalization** divides by $\max\big(1, \operatorname{Per}(R, 95) - \operatorname{Per}(R, 5)\big)$,
  using an exponential moving average (EMA) of percentiles.
- **KL loss** uses free bits (clip below 1 nat) with $\beta_{\text{dyn}} = 0.5$ and
  $\beta_{\text{rep}} = 0.1$.
- **Unimix** mixes 1% uniform into categorical distributions.

**Exercise.** Use the following checks:

- `symexp(symlog(x)) == x` to float tolerance.
- The two-hot encoding of $x$ has expectation exactly $x$ for $x$ inside the bin range,
  with at most two nonzero entries summing to 1.
- The $\lambda$-return recursion equals GAE plus $V$ (connect back to GAE).
- Percentile normalization leaves returns with a spread under 1 unchanged.

### MuZero

MuZero plans with Monte Carlo tree search (MCTS) inside a learned latent model that
predicts only what planning needs: reward, value, and policy. It never reconstructs
observations. It has three functions: representation $s^0 = h_\theta(o_{1:t})$, dynamics
$(r^k, s^k) = g_\theta(s^{k-1}, a^k)$, and prediction $(p^k, v^k) = f_\theta(s^k)$. The
model unrolls $K = 5$ steps:

$$
\ell = \sum_{k=0}^{K}\Big[\ell^r(u_{t+k}, r^k) + \ell^v(z_{t+k}, v^k) + \ell^p(\pi_{t+k}, p^k)\Big] + c\|\theta\|^2,
$$

where $\pi$ is the MCTS visit distribution and $z$ is an $n$-step bootstrapped value
target. The pUCT selection rule is:

$$
a^* = \arg\max_a\Big[Q(s,a) + P(s,a)\frac{\sqrt{\sum_b N(s,b)}}{1+N(s,a)}\Big(c_1 + \log\frac{\sum_b N(s,b) + c_2 + 1}{c_2}\Big)\Big],
$$

with $c_1 = 1.25$ and $c_2 = 19652$. Values and rewards use the transform
$h(x) = \operatorname{sign}(x)(\sqrt{|x|+1}-1) + \epsilon x$ with a categorical
representation.

- Schrittwieser et al. (2020). *Mastering Atari, Go, Chess and Shogi by Planning with a
  Learned Model*. Nature 588. [arXiv:1911.08265](https://arxiv.org/abs/1911.08265)
- Danihelka et al. (2022). *Policy improvement by planning with Gumbel* (Gumbel MuZero).
  ICLR.

**EfficientZero** adds a self-supervised temporal consistency loss (SimSiam-style), a
value-prefix LSTM, and off-policy value correction. It reached 194.3% mean and 109.0%
median human-normalized score on Atari 100k (2 hours of play). Ye et al. (2021), NeurIPS.
[arXiv:2111.00210](https://arxiv.org/abs/2111.00210). **EfficientZero V2** (Wang et al.,
2024, ICML) extends it to continuous control and low-dimensional inputs, and outperforms
DreamerV3 on 50 of 66 tasks. [arXiv:2403.00564](https://arxiv.org/abs/2403.00564)

**Exercise.** Implement pUCT MCTS on tic-tac-toe or a small deterministic chain with a
*perfect* model and uniform priors. Use the following checks:

- The visit distribution at the root concentrates on the optimal move as simulations
  increase.
- The backed-up $Q$ at the root equals minimax or expectimax values for small depth.
- The value transform $h$ and its inverse round-trip. The inverse is
  $h^{-1}(x) = \operatorname{sign}(x)\Big(\big(\frac{\sqrt{1+4\epsilon(|x|+1+\epsilon)}-1}{2\epsilon}\big)^2-1\Big)$.

### TD-MPC2

TD-MPC2 learns an implicit, decoder-free latent world model with a latent consistency
loss, reward prediction, TD-learned $Q$, and a policy prior. It plans with model
predictive path integral (MPPI) control over a short horizon, bootstrapped with the
terminal $Q$. Its robustness features include SimNorm latents (softmax over groups),
discrete regression for rewards and values, and an ensemble of 5 $Q$ functions. A single
317M-parameter agent performs 80 tasks across DMControl and Meta-World.

- Hansen, Wang, and Su (2022). *Temporal Difference Learning for Model Predictive
  Control*. ICML. [arXiv:2203.04955](https://arxiv.org/abs/2203.04955)
- Hansen, Su, and Wang (2024). *TD-MPC2: Scalable, Robust World Models for Continuous
  Control*. ICLR. [arXiv:2310.16828](https://arxiv.org/abs/2310.16828)

**Exercise.** Implement MPPI on the NumPy pendulum with the true model and check that the
cost decreases monotonically in expectation across iterations. Also check that MPPI with
temperature $\to 0$ reduces to picking the best sampled trajectory.

**Pitfalls across model-based RL.**

- Long model rollouts that compound error.
- A single deterministic model that the policy exploits.
- Reconstructing pixels that are irrelevant to control (motivates value-equivalent models
  like MuZero).
- Evaluation with a different number of planning samples than training.
- Forgetting that planning cost dominates wall-clock time.

---

## Exploration

**Intuition.** $\varepsilon$-greedy explores by dithering, which takes time exponential in
the length of a chain that needs deliberate progress. Directed exploration adds an
*intrinsic reward* for novelty (counts, prediction error) or samples plausible
hypotheses of the world and acts optimally for one (posterior sampling).

### Count-based bonuses and pseudo-counts

Model-based interval estimation with exploration bonus (MBIE-EB) adds
$\beta/\sqrt{N(s,a)}$ to the reward and has polynomial sample complexity guarantees in
tabular MDPs (Strehl and Littman, 2008, JCSS 74). For large state spaces, pseudo-counts
derive $\hat N$ from a density model. Let $\rho_n(s)$ be the model's probability of $s$
before observing it and $\rho'_n(s)$ the "recoding probability" after one more update on
$s$:

$$
\hat N_n(s) = \frac{\rho_n(s)\big(1-\rho'_n(s)\big)}{\rho'_n(s) - \rho_n(s)}.
$$

**Worked example.** With $\rho = 0.1$ and $\rho' = 0.12$, the pseudo-count is
$0.1\cdot0.88/0.02 = 4.4$. For an empirical-count model, $\rho = N/n$ and
$\rho' = (N+1)/(n+1)$, and the formula recovers $\hat N = N$ exactly, which makes a clean
test.

- Bellemare et al. (2016). *Unifying Count-Based Exploration and Intrinsic Motivation*.
  NeurIPS. [arXiv:1606.01868](https://arxiv.org/abs/1606.01868)
- Tang et al. (2017). *#Exploration: A Study of Count-Based Exploration for Deep RL*
  (hash-based counts). NeurIPS. [arXiv:1611.04717](https://arxiv.org/abs/1611.04717)

### RND and ICM

**Random network distillation (RND)** trains a predictor $\hat f$ to match a fixed,
randomly initialized target network $f$. The intrinsic reward
$\|\hat f(s') - f(s')\|^2$ is high on novel states and decays with visitation. Key
details: normalize observations and clip them, normalize the intrinsic reward by a
running estimate of the standard deviation of intrinsic returns, use separate value
heads for intrinsic and extrinsic rewards, and treat the intrinsic stream as
non-episodic. RND was the first method to exceed average human performance on
Montezuma's Revenge without demonstrations.

- Burda, Edwards, Storkey, and Klimov (2019). *Exploration by Random Network
  Distillation*. ICLR. [arXiv:1810.12894](https://arxiv.org/abs/1810.12894)

**Intrinsic curiosity module (ICM)** learns features $\phi$ with an inverse model
(predict $a_t$ from $\phi(s_t)$ and $\phi(s_{t+1})$) so the features ignore uncontrollable
distractors. The curiosity reward is the forward model's error in that feature space:

$$
r^i_t = \frac{\eta}{2}\big\|\hat\phi(s_{t+1}) - \phi(s_{t+1})\big\|^2, \qquad
\min\; (1-\beta)L_I + \beta L_F, \;\beta = 0.2.
$$

- Pathak, Agrawal, Efros, and Darrell (2017). *Curiosity-driven Exploration by
  Self-supervised Prediction*. ICML. [arXiv:1705.05363](https://arxiv.org/abs/1705.05363)
- Burda et al. (2019). *Large-Scale Study of Curiosity-Driven Learning*. ICLR.
  [arXiv:1808.04355](https://arxiv.org/abs/1808.04355). This paper documents the
  *noisy-TV problem*: prediction-error bonuses get stuck on stochastic sources.

### Go-Explore

Go-Explore maintains an archive of "cells" (downscaled frames or domain features), selects
promising cells, *returns* to them first (by restoring simulator state or with a
goal-conditioned policy), and then explores from there. Afterward, it robustifies the
discovered trajectories into a policy (the backward algorithm from demonstrations). It
addresses *detachment* (forgetting frontiers) and *derailment* (failing to return). It
achieved very large scores on Montezuma's Revenge and Pitfall. The specific scores are
UNVERIFIED here.

- Ecoffet et al. (2019). *Go-Explore: a New Approach for Hard-Exploration Problems*.
  [arXiv:1901.10995](https://arxiv.org/abs/1901.10995)
- Ecoffet et al. (2021). *First return, then explore*. Nature 590.
  [arXiv:2004.12919](https://arxiv.org/abs/2004.12919)

### Posterior sampling

Posterior sampling for RL (PSRL) samples an MDP from the posterior at the start of each
episode and follows its optimal policy. It's Thompson sampling for MDPs, with Bayesian
regret $\tilde O(\tau S\sqrt{AT})$. Deep approximations include bootstrapped DQN (an
ensemble of heads trained on bootstrapped masks, one head per episode) and randomized
prior functions.

- Osband, Russo, and Van Roy (2013). *(More) Efficient Reinforcement Learning via
  Posterior Sampling*. NeurIPS. [arXiv:1306.0940](https://arxiv.org/abs/1306.0940)
- Osband, Blundell, Pritzel, and Van Roy (2016). *Deep Exploration via Bootstrapped DQN*.
  NeurIPS. [arXiv:1602.04621](https://arxiv.org/abs/1602.04621)
- Osband, Aslanides, and Cassirer (2018). *Randomized Prior Functions for Deep RL*.
  NeurIPS. [arXiv:1806.03335](https://arxiv.org/abs/1806.03335)

**Exercise: chain MDP exploration.** Build a chain of length $N$ where the agent starts
at the left, "right" moves right with reward 0 except $+1$ at the end, and "left"
gives $+0.001$ (a distractor). Use the following checks:

- $\varepsilon$-greedy Q-learning fails to find the $+1$ within a budget for $N = 20$
  on most seeds, while count-bonus Q-learning (with $\beta/\sqrt{N}$) or PSRL
  (Dirichlet-categorical posterior over transitions, known rewards) succeeds on all
  seeds.
- The pseudo-count formula recovers exact counts for the empirical model.
- An RND predictor trained on a fixed set of states gives lower error on those states
  than on held-out states. It's a statistical check over seeds.
- Thompson sampling on a Beta-Bernoulli bandit has cumulative regret growing
  sublinearly. Check that the regret at $T = 2{,}000$ divided by that at $T = 500$ is well
  below 4.

**Pitfalls.**

- Intrinsic rewards that never decay (noisy TV).
- Episodic resets of intrinsic returns that remove the incentive to reach new rooms.
- Unnormalized intrinsic rewards that swamp extrinsic ones.
- For RND, a predictor with the same architecture that learns the target instantly on
  low-dimensional inputs.

---

## Offline RL

**Intuition.** Offline RL learns from a fixed dataset $\mathcal D$ collected by a
behavior policy $\pi_\beta$, without further interaction. The core failure is
**distribution shift**: Bellman backups query $Q(s', a')$ at actions the data never
covers. Those out-of-distribution (OOD) values are arbitrarily wrong, the max picks the
overestimated ones, and there's no new data to correct them. The remedies fall into
four families:

- Constrain the policy to the data support (BCQ, TD3+BC, AWAC).
- Penalize OOD values (CQL).
- Never query OOD actions at all (IQL).
- Recast RL as conditional sequence modeling (Decision Transformer).

### BCQ

BCQ trains a generative model (conditional VAE) of $\pi_\beta(a\mid s)$ and a small
perturbation network $\xi_\phi(s,a,\Phi)$ bounded in $[-\Phi,\Phi]$. It picks the best of
$n$ sampled, perturbed actions under $Q$. Its target uses a soft clipped double Q:
$\lambda\min_j Q_j + (1-\lambda)\max_j Q_j$ with $\lambda = 0.75$.

- Fujimoto, Meger, and Precup (2019). *Off-Policy Deep Reinforcement Learning without
  Exploration*. ICML. [arXiv:1812.02900](https://arxiv.org/abs/1812.02900)

### CQL

CQL learns a conservative Q whose expected value under the learned policy lower-bounds
the true value. The $\text{CQL}(\mathcal H)$ variant is:

$$
\min_Q\; \alpha\,\mathbb{E}_{s\sim\mathcal D}\Big[\log\sum_a\exp Q(s,a) - \mathbb{E}_{a\sim\pi_\beta(\cdot\mid s)}[Q(s,a)]\Big] + \frac12\mathbb{E}_{(s,a,s')\sim\mathcal D}\Big[\big(Q(s,a) - \hat{\mathcal B}^\pi\bar Q(s,a)\big)^2\Big].
$$

The first term pushes down a soft maximum over all actions and pushes up dataset
actions. **Worked example.** With $Q(s,\cdot) = [1, 2, 3]$ and the dataset action 0,
the penalty is $\log(e^1 + e^2 + e^3) - 1 = 2.4076$.

- Kumar, Zhou, Tucker, and Levine (2020). *Conservative Q-Learning for Offline
  Reinforcement Learning*. NeurIPS. [arXiv:2006.04779](https://arxiv.org/abs/2006.04779)

### IQL

IQL avoids OOD queries entirely. It fits $V$ as an upper *expectile* of $Q$ over dataset
actions, and fits $Q$ with a backup that uses $V(s')$, never $\max_{a'}$. It then extracts
a policy with advantage-weighted regression.

$$
L_V = \mathbb{E}_{(s,a)\sim\mathcal D}\big[L_2^\tau\big(\bar Q(s,a) - V(s)\big)\big], \quad L_2^\tau(u) = \big|\tau - \mathbb{1}\{u<0\}\big|u^2,
$$

$$
L_Q = \mathbb{E}\big[(r + \gamma V(s') - Q(s,a))^2\big], \qquad
L_\pi = -\mathbb{E}\big[\exp\big(\beta(\bar Q(s,a) - V(s))\big)\log\pi(a\mid s)\big].
$$

With $\tau = 0.5$, $V$ fits the mean (SARSA-like). As $\tau\to1$, it approaches the max
over in-support actions. **Worked example.** With $\tau = 0.7$, $L^\tau_2(1) = 0.7$ and
$L^\tau_2(-1) = 0.3$. In practice, clip the exponentiated advantage weights (for example,
at 100).

- Kostrikov, Nair, and Levine (2022). *Offline Reinforcement Learning with Implicit
  Q-Learning*. ICLR. [arXiv:2110.06169](https://arxiv.org/abs/2110.06169)

### AWAC and AWR

The KL-constrained policy improvement problem has the closed-form solution
$\pi^*(a\mid s)\propto\pi_\beta(a\mid s)\exp(A(s,a)/\lambda)$, which you fit by weighted
maximum likelihood. AWAC uses an off-policy $Q$ critic, which suits offline-to-online
fine-tuning.

- Nair, Gupta, Dalal, and Levine (2020). *AWAC: Accelerating Online Reinforcement
  Learning with Offline Datasets*. [arXiv:2006.09359](https://arxiv.org/abs/2006.09359)
- Peng, Kumar, Zhang, and Levine (2019). *Advantage-Weighted Regression*.
  [arXiv:1910.00177](https://arxiv.org/abs/1910.00177)

### TD3+BC

TD3+BC adds a behavior cloning (BC) term to the TD3 actor loss and normalizes the Q
scale:

$$
\pi = \arg\max_\pi\;\mathbb{E}_{(s,a)\sim\mathcal D}\big[\lambda Q(s,\pi(s)) - (\pi(s) - a)^2\big], \qquad
\lambda = \frac{\alpha}{\frac1N\sum_i|Q(s_i,a_i)|},\;\alpha = 2.5.
$$

It also normalizes states. **Worked example.** With minibatch $Q$ values
$[10, -20, 30]$, the result is $\lambda = 2.5/20 = 0.125$. The value of $\lambda$ is
treated as a constant (detached).

- Fujimoto and Gu (2021). *A Minimalist Approach to Offline Reinforcement Learning*.
  NeurIPS. [arXiv:2106.06860](https://arxiv.org/abs/2106.06860)

### Decision Transformer

Decision Transformer models trajectories as sequences of
$(\hat R_t, s_t, a_t)$ tokens with return-to-go $\hat R_t = \sum_{t'\ge t} r_{t'}$, and
trains a causal transformer to predict actions. At test time, you condition on a desired
return. It doesn't bootstrap. It struggles to "stitch" suboptimal trajectories, and
conditioning on returns is unreliable in stochastic environments.

- Chen et al. (2021). *Decision Transformer: Reinforcement Learning via Sequence
  Modeling*. NeurIPS. [arXiv:2106.01345](https://arxiv.org/abs/2106.01345)
- Janner, Li, and Levine (2021). *Offline RL as One Big Sequence Modeling Problem*
  (Trajectory Transformer). NeurIPS. [arXiv:2106.02039](https://arxiv.org/abs/2106.02039)

**Broader references.**

- Levine, Kumar, Tucker, and Fu (2020). *Offline Reinforcement Learning: Tutorial,
  Review, and Perspectives on Open Problems*. [arXiv:2005.01643](https://arxiv.org/abs/2005.01643)
- Fu et al. (2020). *D4RL: Datasets for Deep Data-Driven RL*.
  [arXiv:2004.07219](https://arxiv.org/abs/2004.07219). Park et al. (2025). *OGBench*
  (ICLR 2025) is the successor benchmark used by FQL.

**Exercise: offline RL on a tabular MDP.** Generate a dataset from a behavior policy that
never takes action 2 in some states. Use the following checks:

- Offline tabular Q-learning with a max backup assigns the unseen action its (arbitrary)
  initialization, which is optimistic if initialized high, and the greedy policy selects
  it. That's the distribution shift failure made visible.
- Tabular CQL with $\alpha > 0$ produces $\mathbb{E}_{a\sim\pi}Q(s,a) \le V^\pi(s)$ for
  the evaluated policy. Compute $V^\pi$ exactly with the true MDP.
- IQL with $\tau\to1$ approaches the in-support optimal value (value iteration restricted
  to dataset actions), while $\tau = 0.5$ gives the behavior value $V^{\pi_\beta}$.
- The expectile loss minimizer on samples matches the expectile computed by bisection.
- The CQL penalty and TD3+BC $\lambda$ reproduce the worked examples.

**Pitfalls.**

- Tuning hyperparameters with online evaluation, which defeats the offline premise.
- Unclipped AWR weights that explode.
- CQL's $\alpha$ is very dataset-sensitive.
- Using BC-regularized methods on narrow expert data, where the optimum is BC anyway.
- Decision Transformer target returns outside the data range.

---

## Imitation learning and inverse RL

### Behavior cloning and DAgger

**Intuition.** Behavior cloning is supervised learning of $\pi(a\mid s)$ on expert
pairs. Small errors take the learner to states the expert never visited, where errors
grow further. That's *compounding error*: expected cost grows as $O(\epsilon T^2)$ in the
horizon $T$ for per-step error $\epsilon$. DAgger fixes this by rolling out the
*learner*, asking the expert to label the visited states, aggregating all data, and
retraining. That gives $O(\epsilon T)$ (no-regret reduction).

**Equations.** DAgger iteration $i$ uses the mixture
$\pi_i = \beta_i\pi^* + (1-\beta_i)\hat\pi_i$, collects $\mathcal D_i = \{(s, \pi^*(s)) : s\sim d^{\pi_i}\}$,
sets $\mathcal D \leftarrow \mathcal D\cup\mathcal D_i$, and trains
$\hat\pi_{i+1}$ on $\mathcal D$.

- Pomerleau (1989). *ALVINN: An Autonomous Land Vehicle in a Neural Network*. NeurIPS.
- Ross and Bagnell (2010). *Efficient Reductions for Imitation Learning*. AISTATS.
- Ross, Gordon, and Bagnell (2011). *A Reduction of Imitation Learning and Structured
  Prediction to No-Regret Online Learning*. AISTATS.
  [arXiv:1011.0686](https://arxiv.org/abs/1011.0686)

**Exercise.** On a gridworld, or a "narrow corridor" task where one wrong step leads to
an off-distribution region, use an expert from value iteration. Check that BC with a
restricted policy class has error growing superlinearly with $T$, while DAgger's policy
matches the expert on all states reachable by the learner after a few iterations. The
check is exact: count the mismatched states.

### GAIL

GAIL matches the learner's occupancy measure to the expert's with a GAN-style
discriminator. The policy is trained with TRPO or PPO using a reward derived from $D$.

$$
\min_\pi\max_D\;\mathbb{E}_{\pi}\big[\log D(s,a)\big] + \mathbb{E}_{\pi_E}\big[\log(1-D(s,a))\big] - \lambda\mathcal H(\pi).
$$

In the paper's convention, $D$ outputs the probability that $(s,a)$ came from the
*learner*, and the cost is $\log D(s,a)$. Many implementations flip the convention and use
reward $-\log(1 - D)$ or $\log D - \log(1-D)$, where $D$ scores expert-likeness. These
choices introduce survival or termination biases (Kostrikov et al., 2019,
*Discriminator-Actor-Critic*, [arXiv:1809.02925](https://arxiv.org/abs/1809.02925)).

- Ho and Ermon (2016). *Generative Adversarial Imitation Learning*. NeurIPS.
  [arXiv:1606.03476](https://arxiv.org/abs/1606.03476)
- Fu, Luo, and Levine (2018). *Learning Robust Rewards with Adversarial Inverse
  Reinforcement Learning* (AIRL). ICLR. [arXiv:1710.11248](https://arxiv.org/abs/1710.11248)

### Inverse RL and maximum-entropy IRL

**Intuition.** Inverse RL infers a reward that explains the expert's behavior. It's
ill-posed because many rewards, including $r = 0$, make the expert optimal. Max-ent IRL
resolves the ambiguity by choosing the trajectory distribution with maximum entropy
subject to matching expert feature expectations. That yields
$P(\tau)\propto\exp(\theta^\top f_\tau)$, where $f_\tau = \sum_t f(s_t)$.

**Equations.** The log-likelihood gradient is the difference between expert feature
counts and expected feature counts under the current reward's soft-optimal policy:

$$
\nabla_\theta\mathcal L = \tilde f_{E} - \sum_s D_\theta(s) f(s),
$$

where the state visitation frequencies $D_\theta$ come from a backward pass (soft value
iteration) and a forward pass (propagation of visitation).

- Ng and Russell (2000). *Algorithms for Inverse Reinforcement Learning*. ICML.
- Abbeel and Ng (2004). *Apprenticeship Learning via Inverse Reinforcement Learning*.
  ICML.
- Ziebart, Maas, Bagnell, and Dey (2008). *Maximum Entropy Inverse Reinforcement
  Learning*. AAAI.

**Exercise.** Use a 5x5 gridworld with one-hot state features and a known reward.
Generate soft-optimal expert trajectories. Use the following checks:

- The max-ent IRL gradient computed by the forward-backward algorithm matches autograd
  through the exact log-likelihood on enumerated short trajectories (horizon 3-4 on a
  tiny grid).
- After training, the learned policy's expected feature counts match the expert's within
  a tolerance. That's the moment-matching condition at the optimum. Don't check that the
  reward matches exactly, because it's identifiable only up to shaping.

**Pitfalls.**

- Expecting to recover the true reward. At best, you recover it up to potential-based
  shaping and constants.
- GAIL discriminators that overfit instantly, which requires gradient penalties or
  noise.
- Absorbing-state handling in adversarial imitation.
- BC on multi-modal demonstrations with a unimodal Gaussian averages the modes. This
  motivates diffusion policies (see [Diffusion and flow policies](#diffusion-and-flow-policies-for-control)).

---

## Multi-agent RL

**Intuition.** With several learning agents, each agent's environment is non-stationary
because the others change. Three patterns dominate practice:

- **Self-play** trains against copies or past versions of yourself, as in the AlphaGo
  line and OpenAI Five.
- **Centralized training with decentralized execution (CTDE)** gives critics global
  information during training, while each actor acts on local observations at execution
  time. Examples are MADDPG, COMA, QMIX, and MAPPO.
- **Population-based or league training** keeps a population of opponents to avoid
  cycling and forgetting, as in AlphaStar and PSRO.

**Key equations.** Value decomposition networks (VDN) use
$Q_{\text{tot}} = \sum_i Q_i(o_i, a_i)$. QMIX generalizes this with a monotonic mixing
network, $\partial Q_{\text{tot}}/\partial Q_i\ge0$, enforced by non-negative
hypernetwork weights. That guarantees
$\arg\max_{\mathbf a}Q_{\text{tot}} = (\arg\max_{a_1}Q_1,\dots,\arg\max_{a_n}Q_n)$.

**Lessons from the large systems.**

- **AlphaStar** (StarCraft II, Grandmaster level) was pretrained with supervised learning
  on human replays and then trained in a league of main agents, main exploiters, and
  league exploiters with prioritized fictitious self-play. Without the league, agents
  cycled and forgot strategies.
- **OpenAI Five** (Dota 2) scaled PPO massively over about 10 months, used "surgery" to
  keep training through architecture changes, used a team-spirit reward mixing, and beat
  the world champions (OG) in 2019.
- **MAPPO** showed that well-tuned PPO with a centralized value function is a strong
  cooperative baseline. Its details include value normalization, agent-specific global
  state, and fewer epochs.

**References.**

- Lowe et al. (2017). *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive
  Environments* (MADDPG). NeurIPS. [arXiv:1706.02275](https://arxiv.org/abs/1706.02275)
- Foerster et al. (2018). *Counterfactual Multi-Agent Policy Gradients* (COMA). AAAI.
  [arXiv:1705.08926](https://arxiv.org/abs/1705.08926)
- Sunehag et al. (2018). *Value-Decomposition Networks*. AAMAS.
  [arXiv:1706.05296](https://arxiv.org/abs/1706.05296)
- Rashid et al. (2018). *QMIX*. ICML. [arXiv:1803.11485](https://arxiv.org/abs/1803.11485)
- Yu et al. (2022). *The Surprising Effectiveness of PPO in Cooperative Multi-Agent
  Games* (MAPPO). NeurIPS Datasets and Benchmarks.
  [arXiv:2103.01955](https://arxiv.org/abs/2103.01955)
- Vinyals et al. (2019). *Grandmaster level in StarCraft II using multi-agent
  reinforcement learning*. Nature 575, 350-354.
- Berner et al. (2019). *Dota 2 with Large Scale Deep Reinforcement Learning*.
  [arXiv:1912.06680](https://arxiv.org/abs/1912.06680)
- Lanctot et al. (2017). *A Unified Game-Theoretic Approach to Multiagent RL* (PSRO).
  NeurIPS. [arXiv:1711.00832](https://arxiv.org/abs/1711.00832)
- Heinrich and Silver (2016). *Deep Reinforcement Learning from Self-Play in
  Imperfect-Information Games* (NFSP). [arXiv:1603.01121](https://arxiv.org/abs/1603.01121)

**Exercise.** Use the following checks:

- In rock-paper-scissors, simultaneous gradient ascent for both players cycles, and the
  last iterate doesn't converge. Fictitious play's *average* strategy converges to
  $(1/3, 1/3, 1/3)$ within 0.02 after 5,000 iterations. Both are exactly checkable.
- A QMIX-style mixer with `abs()` weights has non-negative partial derivatives
  everywhere (autograd check), and the decentralized argmax equals the joint argmax by
  enumeration over a 3-agent, 3-action game.

**Pitfalls.**

- Evaluating self-play agents only against themselves.
- Relative overgeneralization in cooperative games, where monotonic factorization can't
  represent the optimum.
- Sharing parameters without agent IDs.

---

## Hierarchical RL and options

**Intuition.** Temporal abstraction lets an agent choose extended behaviors. An *option*
is $\omega = (\mathcal I_\omega, \pi_\omega, \beta_\omega)$: an initiation set, an
intra-option policy, and a termination function. Options turn an MDP into a semi-MDP.
The SMDP Q-learning update, for an option lasting $k$ steps with discounted reward sum
$R$, is:

$$
Q(s,\omega) \leftarrow Q(s,\omega) + \alpha\big[R + \gamma^k\max_{\omega'}Q(s',\omega') - Q(s,\omega)\big].
$$

Option-critic learns $\pi_\omega$ and $\beta_\omega$ end to end with policy-gradient
theorems for both. In the termination gradient, a high option advantage discourages
termination. Feudal and goal-conditioned approaches (FeUdal Networks, HIRO, HAC) have a
manager set subgoals for a worker, and handle the non-stationarity of the lower level
with off-policy relabeling.

**References.**

- Sutton, Precup, and Singh (1999). *Between MDPs and semi-MDPs: A framework for temporal
  abstraction in reinforcement learning*. Artificial Intelligence 112.
- Bacon, Harb, and Precup (2017). *The Option-Critic Architecture*. AAAI.
  [arXiv:1609.05140](https://arxiv.org/abs/1609.05140)
- Vezhnevets et al. (2017). *FeUdal Networks for Hierarchical RL*. ICML.
  [arXiv:1703.01161](https://arxiv.org/abs/1703.01161)
- Nachum, Gu, Lee, and Levine (2018). *Data-Efficient Hierarchical RL* (HIRO). NeurIPS.
  [arXiv:1805.08296](https://arxiv.org/abs/1805.08296)
- Levy, Konidaris, Platt, and Saenko (2019). *Learning Multi-Level Hierarchies with
  Hindsight* (HAC). ICLR. [arXiv:1712.00948](https://arxiv.org/abs/1712.00948)
- Andrychowicz et al. (2017). *Hindsight Experience Replay*. NeurIPS.
  [arXiv:1707.01495](https://arxiv.org/abs/1707.01495)

**Exercise.** Use the four-rooms gridworld with hand-coded hallway options. Check that
SMDP Q-learning with options reaches the goal in fewer episodes than primitive Q-learning,
averaged over seeds. Also check that the SMDP update with $k = 1$ options that equal
primitive actions reduces exactly to Q-learning.

**Pitfalls.** Options that collapse to one-step or never terminate, and a manager that
learns on stale lower-level behavior.

---

## Sim-to-real and robotics RL

**Intuition.** Real-robot data is slow and risky to collect. The dominant recipe is to
train in massively parallel simulation and close the reality gap with the following
techniques:

- **Domain randomization** of dynamics, visuals, latency, and noise, so reality looks
  like one more sample.
- **System identification** or learned actuator models.
- **Privileged teacher-student distillation**: a teacher sees true terrain and
  friction, and a student learns from proprioceptive history to imitate it.
- **Online adaptation** of an inferred latent environment extrinsic (RMA).
- **Asymmetric actor-critic**: the critic sees privileged state while the actor sees
  observations.

PPO in GPU simulators (Isaac Gym, Isaac Lab, MuJoCo Playground or MJX) with thousands of
parallel environments is the workhorse for locomotion. Off-policy methods (FastTD3,
SAC variants) and real-world RL with human interventions (HIL-SERL) are increasingly
used for manipulation.

**References.**

- Tobin et al. (2017). *Domain Randomization for Transferring Deep Neural Networks from
  Simulation to the Real World*. IROS. [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)
- Peng et al. (2018). *Sim-to-Real Transfer of Robotic Control with Dynamics
  Randomization*. ICRA. [arXiv:1710.06537](https://arxiv.org/abs/1710.06537)
- Pinto et al. (2018). *Asymmetric Actor Critic for Image-Based Robot Learning*. RSS.
  [arXiv:1710.06542](https://arxiv.org/abs/1710.06542)
- OpenAI et al. (2019). *Solving Rubik's Cube with a Robot Hand* (automatic domain
  randomization). [arXiv:1910.07113](https://arxiv.org/abs/1910.07113)
- Hwangbo et al. (2019). *Learning agile and dynamic motor skills for legged robots*.
  Science Robotics. [arXiv:1901.08652](https://arxiv.org/abs/1901.08652)
- Lee et al. (2020). *Learning quadrupedal locomotion over challenging terrain*. Science
  Robotics. [arXiv:2010.11251](https://arxiv.org/abs/2010.11251)
- Kumar, Fu, Pathak, and Malik (2021). *RMA: Rapid Motor Adaptation for Legged Robots*.
  RSS. [arXiv:2107.04034](https://arxiv.org/abs/2107.04034)
- Rudin, Hoeller, Reist, and Hutter (2021). *Learning to Walk in Minutes Using Massively
  Parallel Deep RL*. CoRL. [arXiv:2109.11978](https://arxiv.org/abs/2109.11978)
- Radosavovic et al. (2024). *Real-world humanoid locomotion with reinforcement
  learning*. Science Robotics. [arXiv:2303.03381](https://arxiv.org/abs/2303.03381)
- Luo, Xu, Wu, and Levine (2025). *Precise and Dexterous Robotic Manipulation via
  Human-in-the-Loop Reinforcement Learning* (HIL-SERL). Science Robotics 10(105). It
  reports near-100% success within 1-2.5 hours of real training on most tasks.
  [arXiv:2410.21845](https://arxiv.org/abs/2410.21845)
- Seo et al. (2025). *FastTD3: Simple, Fast, and Capable RL for Humanoid Control*. It
  uses parallel simulation, large batches, and a distributional critic, and solves
  HumanoidBench tasks in under 3 hours on one A100.
  [arXiv:2505.22642](https://arxiv.org/abs/2505.22642)
- Ball, Smith, Kostrikov, and Levine (2023). *Efficient Online RL with Offline Data*
  (RLPD). ICML. [arXiv:2302.02948](https://arxiv.org/abs/2302.02948)

**Exercise: domain randomization on a pendulum or CartPole.** Train a policy (or use LQR
gains) on a nominal pendulum mass. Evaluate across a grid of masses and friction values.
Use the following checks:

- A policy trained with randomized mass has lower worst-case cost across the grid than
  the nominal policy.
- The teacher-student distillation loss on a linear system drives the student's actions
  to the teacher's within tolerance, when the privileged parameter is identifiable from a
  short history.

Both checks work with linear policies and closed-form LQR, so they're fast.

**Pitfalls.**

- Randomization ranges so wide that the task becomes unsolvable, which gives an overly
  conservative policy.
- Ignoring action latency and control frequency.
- Rewards that exploit simulator artifacts.
- Evaluating sim-to-real only on the training randomization distribution.

---

## Diffusion and flow policies for control

**Intuition.** Demonstrations are multi-modal: you can go around an obstacle to the left
or the right. A Gaussian policy regresses to the mean, which is often invalid.
Diffusion (and flow-matching) policies model $p(a\mid s)$ as an iterative denoising
process and represent arbitrary multi-modal distributions. They became the default
imitation backbone in manipulation, and flow-matching action experts power
vision-language-action (VLA) models like $\pi_0$. The RL problem is how to improve such a
policy, because exact log-likelihoods are expensive and backpropagating through many
denoising steps is unstable. The main approaches are as follows:

- **Q-guided training.** Diffusion-QL adds $-\alpha Q(s, a_\theta)$ to the diffusion BC
  loss and backpropagates through the denoising chain.
- **Rejection or resampling.** IDQL samples many actions from a diffusion behavior model
  and reweights them with IQL's critic.
- **Policy gradient on the denoising MDP.** DPPO treats each denoising step as an action
  in an augmented MDP with Gaussian transitions, so the log-probs are exact, and runs
  PPO on it.
- **Distill to one step.** Flow Q-learning (FQL) trains a one-step noise-to-action policy
  to maximize $Q$ while distilling from a BC flow policy, which avoids recursive
  backpropagation.
- **Steer the noise.** DSRL runs RL in the latent-noise space of a frozen diffusion
  policy, so it needs only black-box access.
- **Inject noise into flows.** ReinFlow and $\pi_{\text{RL}}$ make flow sampling
  stochastic for tractable likelihoods, and use policy gradients.

**Equations.** The DDPM training loss for a policy is:

$$
\mathcal L_{\text{BC}}(\theta) = \mathbb{E}_{k,\,(s,a^0)\sim\mathcal D,\,\epsilon\sim\mathcal N(0,I)}\Big[\big\|\epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_k}a^0 + \sqrt{1-\bar\alpha_k}\,\epsilon,\; s,\; k\big)\big\|^2\Big].
$$

The conditional flow-matching loss, with linear interpolation
$a^t = (1-t)\epsilon + t a^1$, is:

$$
\mathcal L_{\text{FM}} = \mathbb{E}\big[\|v_\theta(a^t, s, t) - (a^1 - \epsilon)\|^2\big].
$$

The DPPO denoising MDP uses the fact that each DDPM reverse step is Gaussian,
$p_\theta(a^{k-1}\mid a^k, s) = \mathcal N(\mu_\theta(a^k, s, k), \sigma_k^2 I)$. The
log-probability of the whole chain is then a sum of Gaussian log-densities, and the PPO
ratio is a product over the chain (DPPO clips per denoising step).

**References.**

- Chi et al. (2023). *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*.
  RSS. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)
- Janner, Du, Tenenbaum, and Levine (2022). *Planning with Diffusion for Flexible
  Behavior Synthesis* (Diffuser). ICML. [arXiv:2205.09991](https://arxiv.org/abs/2205.09991)
- Wang, Hunt, and Zhou (2023). *Diffusion Policies as an Expressive Policy Class for
  Offline RL* (Diffusion-QL). ICLR. [arXiv:2208.06193](https://arxiv.org/abs/2208.06193)
- Hansen-Estruch et al. (2023). *IDQL: Implicit Q-Learning as an Actor-Critic Method with
  Diffusion Policies*. [arXiv:2304.10573](https://arxiv.org/abs/2304.10573)
- Ren et al. (2025). *Diffusion Policy Policy Optimization* (DPPO). ICLR.
  [arXiv:2409.00588](https://arxiv.org/abs/2409.00588)
- Park, Li, and Levine (2025). *Flow Q-Learning*. ICML.
  [arXiv:2502.02538](https://arxiv.org/abs/2502.02538)
- Wagenmaker et al. (2025). *Steering Your Diffusion Policy with Latent Space
  Reinforcement Learning* (DSRL). [arXiv:2506.15799](https://arxiv.org/abs/2506.15799)
- Zhang et al. (2025). *ReinFlow: Fine-tuning Flow Matching Policy with Online RL*.
  NeurIPS. [arXiv:2505.22094](https://arxiv.org/abs/2505.22094). The first-author name is
  UNVERIFIED.
- Black et al. (2024). *$\pi_0$: A Vision-Language-Action Flow Model for General Robot
  Control*. [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
- *$\pi_{\text{RL}}$: Online RL Fine-tuning for Flow-based Vision-Language-Action
  Models* (2025). [arXiv:2510.25889](https://arxiv.org/abs/2510.25889). The author list is
  UNVERIFIED.

**Exercise: multi-modal BC in 1-D.** Build a dataset where, for state $s$, the action is
$+1$ or $-1$ with equal probability. Use the following checks:

- A Gaussian MSE policy predicts about 0, which is an invalid action, checked exactly.
- A tiny flow-matching MLP (CPU, a few thousand steps, well under 60 s) produces samples
  whose histogram is bimodal. Less than 10% of the mass is within $\pm0.3$ of 0, and each
  mode holds $50\% \pm 5\%$.
- The DPPO chain log-probability equals the sum of per-step Gaussian log-densities,
  checked against `torch.distributions.Normal`.
- The DDPM forward process gives $\operatorname{Var}(a^k) = \bar\alpha_k\operatorname{Var}(a^0) + (1-\bar\alpha_k)$
  empirically.

**Pitfalls.**

- Differentiating through all denoising steps (unstable and slow).
- Ignoring action chunking and temporal ensembling, which matter a lot in Diffusion
  Policy.
- Evaluating with a different number of denoising steps than training.
- Treating the diffusion model's log-prob as available when it isn't. Only the
  per-step Gaussian chain likelihood is exact.

---

## Scaling deep RL and the 2023-2026 frontier

This section reflects the state of the field as of September 2026. Revisit the claims
before publishing.

**Intuition.** For years, bigger networks made deep RL *worse*, unlike supervised
learning. From 2023 to 2026, a series of papers showed that the right architecture and
regularization let value-based and actor-critic methods scale with parameters and with
the updates-to-data (UTD) ratio. Recurring causes of failure are *plasticity loss* and
*primacy bias*: networks overfit early data and stop learning. Unbounded feature and
weight norms, and nonstationary regression targets, add to the problem. Recurring fixes
are the following:

- Normalization: LayerNorm, or $\ell_2$ and hyperspherical normalization.
- Residual MLP blocks.
- Periodic resets or shrink-and-perturb.
- Weight decay.
- Classification (cross-entropy) value losses instead of MSE.
- Observation and reward normalization.
- Higher UTD ratios made safe by the preceding techniques.

**Key papers.**

- **Primacy bias and resets.** Nikishin et al. (2022). *The Primacy Bias in Deep
  Reinforcement Learning*. ICML. [arXiv:2205.07802](https://arxiv.org/abs/2205.07802).
  D'Oro et al. (2023) showed that resets let the replay ratio scale up (SR-SPR), in
  *Sample-Efficient RL by Breaking the Replay Ratio Barrier* (ICLR). Lyle et al. (2023).
  *Understanding Plasticity in Neural Networks*. ICML.
  [arXiv:2303.01486](https://arxiv.org/abs/2303.01486)
- **BBF** (Schwarzer et al., 2023, ICML). *Bigger, Better, Faster: Human-level Atari with
  human-level efficiency*. It reached superhuman Atari 100k with a scaled Impala-style
  ResNet, higher replay ratio, periodic shrink-and-perturb resets, n-step annealed from
  10 to 3, an increasing discount, and weight decay, on top of SPR.
  [arXiv:2305.19452](https://arxiv.org/abs/2305.19452)
- **HL-Gauss / "Stop Regressing"** (Farebrother et al., 2024, ICML). Training value
  functions with cross-entropy on histogram targets (Gaussian-smoothed two-hot)
  scales better than MSE. [arXiv:2403.03950](https://arxiv.org/abs/2403.03950)
- **CrossQ** (Bhatt et al., 2024, ICLR) removes target networks and uses batch
  normalization with joint current and next batches for sample-efficient SAC at UTD 1.
  [arXiv:1902.05605](https://arxiv.org/abs/1902.05605)
- **TD7** (Fujimoto et al., 2023, NeurIPS). *For SALE: State-Action Representation
  Learning for Deep RL*. [arXiv:2306.02451](https://arxiv.org/abs/2306.02451)
- **BRO** (Nauman et al., 2024, NeurIPS spotlight). *Bigger, Regularized, Optimistic*.
  Strong regularization (BroNet: residual blocks and LayerNorm) lets the critic scale.
  Combined with optimistic exploration and high UTD, it was the first model-free method to
  reach near-optimal Dog and Humanoid on DMC.
  [arXiv:2405.16158](https://arxiv.org/abs/2405.16158)
- **SimBa** (Lee et al., 2025, ICLR) adds running observation normalization, residual
  feedforward blocks, and post-LayerNorm. SAC with SimBa improves as parameters grow,
  while SAC with an MLP degrades. [arXiv:2410.09754](https://arxiv.org/abs/2410.09754).
  **SimbaV2** (Lee et al., 2025, ICML) uses hyperspherical ($\ell_2$) normalization of
  weights and features, learnable interpolation (LERP) residuals, a distributional
  critic with KL loss, and reward scaling. It reports state-of-the-art results on 57
  continuous control tasks. [arXiv:2502.15280](https://arxiv.org/abs/2502.15280)
- **MR.Q** (Fujimoto et al., 2025, ICLR). *Towards General-Purpose Model-Free RL*. It
  learns model-based representations that approximately linearize the value function,
  and uses one hyperparameter set across Gym, DMC (state and pixels), and Atari.
  [arXiv:2501.16142](https://arxiv.org/abs/2501.16142)
- **PQN** (Gallici et al., 2025, ICLR). *Simplifying Deep Temporal Difference Learning*.
  It shows LayerNorm can make TD provably convergent without target networks. Its
  parallel Q-learning with vectorized environments and no replay buffer matches Rainbow
  on Atari and is up to 50x faster than DQN.
  [arXiv:2407.04811](https://arxiv.org/abs/2407.04811)
- **Scaling laws for value-based RL.** Rybkin, Nauman, Fu, Snell, Abbeel, Levine, and
  Kumar (2025). *Value-Based Deep RL Scales Predictably*. Data and compute requirements
  lie on a Pareto frontier controlled by the UTD ratio, and power-law fits predict
  large-scale hyperparameters. [arXiv:2502.04327](https://arxiv.org/abs/2502.04327).
  A follow-up is *Compute-Optimal Scaling for Value-Based Deep RL* (2025),
  [arXiv:2508.14881](https://arxiv.org/abs/2508.14881).
- **Mixtures of experts.** Obando-Ceron et al. (2024). *Mixtures of Experts Unlock
  Parameter Scaling for Deep RL*. ICML. [arXiv:2402.08609](https://arxiv.org/abs/2402.08609)
- **Depth scaling.** Wang, Javali, Bortkiewicz, Trzciński, and Eysenbach (2025). *1000
  Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching
  Capabilities*. NeurIPS 2025 Best Paper. Contrastive goal-conditioned RL with residual
  connections, LayerNorm, and Swish scales to more than 1,000 layers with 2x-50x gains,
  while SAC saturates. [arXiv:2503.14858](https://arxiv.org/abs/2503.14858)
- **World-model agents at scale.** DreamerV3 appeared in Nature (2025). Dreamer 4
  (September 2025) reached Minecraft diamonds from offline data only. See
  [World Models and Dreamer v1-v3](#world-models-and-dreamer-v1-v3).
- **RL for robot foundation models.** Physical Intelligence's *$\pi^*_{0.6}$: a VLA That
  Learns From Experience* (November 2025) introduced RECAP (RL with Experience and
  Corrections via Advantage-conditioned Policies). It trains a value function and
  conditions the VLA on advantage indicators, using autonomous rollouts and human
  corrections. It reports more than doubled throughput and roughly halved failure rates
  on laundry folding, box assembly, and espresso making.
  [arXiv:2511.14759](https://arxiv.org/abs/2511.14759). SimpleVLA-RL (ICLR 2026) applied
  LLM-style outcome-reward RL (built on veRL) to OpenVLA-OFT. It raised LIBERO-Long from
  17.3 to 91.7 with one demonstration per task for supervised fine-tuning (SFT).
  [arXiv:2509.09674](https://arxiv.org/abs/2509.09674). 2026 preprints such as EXPO-FT
  ([arXiv:2605.25477](https://arxiv.org/abs/2605.25477)) and its real-time variant
  ([arXiv:2609.18207](https://arxiv.org/abs/2609.18207)) report sample-efficient
  real-world RL fine-tuning of VLAs. Their headline numbers (for example, 30/30 success on
  eight tasks, and 42% to 97% with 10 minutes of robot data) come from abstracts and
  weren't independently checked, so treat them as UNVERIFIED preprint claims.
- **Other 2025-2026 continuous-control work** includes FastTD3 (discussed in the
  preceding sim-to-real section) and FoG (*A Forget-and-Grow Strategy for Deep RL
  Scaling in Continuous Control*, [arXiv:2507.02712](https://arxiv.org/abs/2507.02712)),
  which reports gains over SimBa, BRO, and TD-MPC2. The peer-review status and exact
  claims of FoG are UNVERIFIED.

**Exercise: scaling pathologies you can reproduce on CPU.** Use the following checks:

- **Feature-norm growth.** Train a plain MLP Q-network on a chain with high UTD and log
  the feature norms. Add LayerNorm and check that the norms stay bounded (a
  deterministic seed with a ratio threshold). Tie this to the PQN theory with a linear-TD
  divergence counterexample: Baird's star, or the two-state "$w \to 2w$" example, where
  semi-gradient TD diverges off-policy. Check that the iterates' norm grows geometrically
  without regularization and converges with a sufficiently strong $\ell_2$ or LayerNorm-like
  normalization. The PQN bound is in the paper. The exact exercise constants need
  deriving, and they're UNVERIFIED here.
- **HL-Gauss target.** Check that the histogram target for $y$ with smoothing $\sigma$
  has mean close to $y$ (exact up to bin truncation), and that the cross-entropy gradient
  is bounded while MSE gradients scale with $|y|$.
- **Plasticity.** Train on task A and then task B, with and without a reset of the last
  layer. Check that the reset run fits task B to a lower loss in a fixed number of steps
  (a statistical check over 5 seeds).

---

## Cross-cutting common mistakes

The following bugs recur across algorithms. They're worth a dedicated debugging lesson:

1. **Truncation versus termination.** Bootstrap on time limits, and don't bootstrap on
   true terminal states.
2. **Detaching targets.** Targets, advantages, returns, $\lambda$ in TD3+BC, and $\alpha$
   in SAC's actor loss must not carry gradients.
3. **Off-by-one `done` indexing** in GAE and n-step code.
4. **Log-prob bookkeeping.** Store `logp_old` at collection time, compute log-probs of
   the *pre-clip* or *pre-squash* action, and apply the tanh correction.
5. **Normalization mismatches.** Use the same observation normalizer statistics at train
   and evaluation time, and freeze or save them.
6. **Seeds and evaluation.** Report several seeds with confidence intervals. Agarwal et
   al. (2021), *Deep RL at the Edge of the Statistical Precipice*, NeurIPS,
   [arXiv:2108.13264](https://arxiv.org/abs/2108.13264), recommends the interquartile mean
   (IQM) and stratified bootstrap confidence intervals. Henderson et al. (2018), *Deep
   Reinforcement Learning that Matters*, AAAI,
   [arXiv:1709.06560](https://arxiv.org/abs/1709.06560), documents the seed sensitivity
   problem.
7. **Deadly triad.** Function approximation, bootstrapping, and off-policy data together
   can diverge. See Sutton and Barto (2018), chapter 11.
8. **Reward scale.** Many algorithms (SAC's entropy term, PPO's value loss) are sensitive
   to it. Normalize, or use scale-free losses (two-hot, symlog).

## Test harness notes for the course

The following notes apply to the check suites:

- **Exact checks.** Closed-form tabular values, finite-difference gradients, the PPO
  table, GAE identities, C51 projection, the tanh correction, and the NPG identity are
  deterministic and fast. Assert with `numpy.testing.assert_allclose(rtol=1e-6)`.
- **Statistical checks.** For unbiasedness tests, use a z-test at 4 standard errors
  computed from per-sample estimates, with a fixed seed. Choose sample sizes so each
  check runs in under 5 s. The REINFORCE prototype used 4,000 episodes of length 60 in
  about 1.7 s in pure Python. Vectorizing over episodes makes it much faster.
- **Learning checks** (for example, "CartPole reaches 475") are the flakiest. Use soft
  thresholds, 2-of-3 seeds, and small networks, and cap wall-clock time with a timeout
  so the suite stays under 60 s.
- Prefer `float64` in NumPy checks. For PyTorch, call `torch.set_default_dtype(torch.float64)`
  in exact-comparison tests, and keep `float32` where numerical-stability lessons (tanh
  correction) need it.

## Unverified items

The following items couldn't be confirmed against a primary source:

- MBPO's 5% real-data ratio (from the reference code, not checked against the paper).
- Go-Explore's specific Montezuma's Revenge and Pitfall scores (omitted on purpose).
- The CPU wall-clock budgets for the TD3 Pendulum and PPO CartPole learning checks. These
  are estimates and need measuring on the grading hardware.
- The first-author name for ReinFlow and the author list for $\pi_{\text{RL}}$.
- The EXPO-FT, real-time EXPO-FT, and FoG headline claims (2025-2026 preprints; numbers
  taken from abstracts).
- The exact constants for a PQN-style LayerNorm convergence exercise.

Everything else was cross-checked against arXiv, proceedings pages, or the cited blog,
or was recomputed numerically. That includes all worked examples, the tabular policy
gradient theorem check (error $1.2\times10^{-9}$), and the tabular NPG identity.
