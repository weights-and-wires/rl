# Research notes: classical reinforcement learning foundations

These notes collect the verified material for the classical part of the course:
lessons `01` to `11` in `docs/CURRICULUM.md`, plus the lambda-return background
that lesson `17-generalized-advantage-estimation` builds on. Each topic follows
the same six-part layout:

1. Intuition in two or three plain sentences.
2. Key equations in KaTeX, written in the course notation.
3. A small worked example with numbers that were computed and checked in
   numpy.
4. Canonical references.
5. An exercise idea with checks that are mathematically verifiable and fit the
   20-second suite budget in `docs/CONTENT_CONTRACT.md`.
6. Common student mistakes and failure modes.

A verification log at the end lists what was checked, how, and what remains
UNVERIFIED.

## Notation and conventions

The course contract fixes the notation, and these notes follow it. The
contract differs from Sutton and Barto (S&B) in one place that matters for
every equation, as follows:

- The course writes $r_t$ for the reward received for taking $a_t$ in $s_t$.
  S&B write $R_{t+1}$ for the same quantity. Every S&B equation in these notes
  is translated: S&B's $R_{t+1} + \gamma V(S_{t+1})$ becomes
  $r_t + \gamma V(s_{t+1})$.
- An episode is $s_0, a_0, r_0, s_1, a_1, r_1, \dots, s_T$, where $s_T$ is
  terminal. The last reward is $r_{T-1}$.
- Values are $V^\pi(s)$ and $Q^\pi(s,a)$. Optimal values are $V^*$ and $Q^*$.
- A tabular policy is an array `pi` of shape `(S, A)` whose rows sum to one,
  matching `rlenvs`.
- The value of a terminal state is zero by definition. Code must mask the
  bootstrap term with `(1 - terminated)`.

The book citation for every "S&B" reference is: Richard S. Sutton and Andrew G.
Barto, *Reinforcement Learning: An Introduction*, 2nd edition, MIT Press, 2018.
The authors host a free PDF at
[incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html).

## 1. The RL problem and the agent-environment loop

### Intuition

An agent repeatedly observes a state, picks an action, and receives a reward and
a next state from the environment. The agent's goal is to pick actions that make
the total discounted reward large, not the next reward alone. Everything else in
the course is a way to estimate "how good is this state or action in the long
run" and to act on that estimate.

### Key equations

The environment is described by transition probabilities and an expected reward
function:

$$
P(s' \mid s, a) = \Pr(s_{t+1} = s' \mid s_t = s, a_t = a), \qquad
R(s, a) = \mathbb{E}[r_t \mid s_t = s, a_t = a].
$$

The discounted return from time $t$ is:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k} = r_t + \gamma G_{t+1},
\qquad 0 \le \gamma \le 1.
$$

In an episodic task the sum stops at $T-1$, and $G_T = 0$. The recursion
$G_t = r_t + \gamma G_{t+1}$ is what the exercise implements by a backward loop.

The value functions are expectations of the return:

$$
V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s], \qquad
Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a].
$$

The expectation is over the policy's action choices and the environment's
transitions, starting from the conditioned state.

### Worked example

Take rewards $r_0 = 1$, $r_1 = 0$, $r_2 = 2$ and then termination, with
$\gamma = 0.9$. The backward recursion gives the following returns:

- $G_2 = 2$.
- $G_1 = 0 + 0.9 \cdot 2 = 1.8$.
- $G_0 = 1 + 0.9 \cdot 1.8 = 2.62$.

The direct sum $1 + 0.9 \cdot 0 + 0.81 \cdot 2 = 2.62$ agrees.

### References

- S&B, Chapter 1 and Sections 3.1 to 3.4.
- Puterman, *Markov Decision Processes: Discrete Stochastic Dynamic
  Programming*, Wiley, 1994. The standard reference text for MDP theory.

### Exercise idea

The reader implements the following functions:

- `discounted_returns(rewards, gamma)`: the backward recursion. Check it
  against the direct sum $\sum_k \gamma^k r_{t+k}$ for every $t$ on a seeded
  random reward vector, to `atol=1e-12`.
- `run_episode(mdp, pi, seed)`: the loop over `mdp.reset(seed)` and
  `mdp.step(a)`. Check reproducibility by calling it twice with the same seed
  and comparing trajectories with `equal`.
- `mc_value_estimate(mdp, pi, n_episodes, seed)`: the average of $G_0$. Check
  it against `rlenvs.policy_evaluation(mdp, pi)` at the start state within
  four standard errors. Compute the standard error from the sample, so the
  threshold scales with the variance and holds under every seed.

### Common mistakes

- Off-by-one indexing between S&B's $R_{t+1}$ and the course's $r_t$.
- Bootstrapping past a terminal state, or treating a time-limit truncation as
  termination. Truncation needs $\gamma V(s_{\text{last}})$ added; termination
  adds nothing.
- Forward loops that recompute the sum for each $t$, which costs $O(T^2)$
  instead of $O(T)$.
- Discounting the first reward: $G_t$ starts with $\gamma^0 r_t$.

## 2. Multi-armed bandits

A bandit is an MDP with one state. It isolates the exploration-exploitation
trade-off: to find the best arm you must try arms that look worse.

### 2.1 Action values, incremental means, and epsilon-greedy

#### Intuition

Estimate each arm's mean reward by averaging what you saw. Usually pick the arm
with the highest estimate, but with probability $\varepsilon$ pick an arm
uniformly at random so no arm is starved. A constant step size tracks a mean
that drifts over time, because it weights recent rewards more.

#### Key equations

The incremental sample mean after the $n$-th reward $r_n$ from an arm is:

$$
Q_{n+1} = Q_n + \frac{1}{n}\left(r_n - Q_n\right).
$$

With a constant step size $\alpha \in (0, 1]$ the estimate is an exponentially
weighted average (S&B equation 2.6):

$$
Q_{n+1} = (1-\alpha)^n Q_1 + \sum_{i=1}^{n} \alpha (1-\alpha)^{n-i} r_i .
$$

The weights $(1-\alpha)^n + \sum_i \alpha(1-\alpha)^{n-i}$ sum to exactly one.
A step-size sequence converges to the true mean with probability one when
$\sum_n \alpha_n = \infty$ and $\sum_n \alpha_n^2 < \infty$ (the
Robbins-Monro conditions, S&B equation 2.7). The sequence $1/n$ satisfies both.
A constant $\alpha$ fails the second, which is why it keeps tracking.

#### Worked example

Rewards $1, 0, 1, 1$ from one arm with $Q_1 = 0$ and $\alpha_n = 1/n$ give
$Q_2 = 1$, $Q_3 = 0.5$, $Q_4 = 2/3$, $Q_5 = 0.75$. This equals the plain mean
$3/4$.

#### References

- S&B, Sections 2.1 to 2.6.
- Robbins and Monro, "A stochastic approximation method," *Annals of
  Mathematical Statistics* 22(3), 1951,
  [doi:10.1214/aoms/1177729586](https://doi.org/10.1214/aoms/1177729586).

#### Common mistakes

- Using `np.argmax` for ties. It always picks the lowest index, which biases
  early play. Break ties at random with a seeded generator.
- Counting a pull in `N` but updating a different arm's `Q`.
- Decaying $\varepsilon$ to zero too fast and locking onto a suboptimal arm.
- Measuring regret with realised rewards instead of the gap
  $\mu^* - \mu_{a_t}$. The gap version has much lower variance.

### 2.2 Upper confidence bound (UCB)

#### Intuition

Add a bonus to each arm's estimate that is large when the arm has been tried
rarely. Pick the arm with the largest estimate plus bonus. An arm is chosen
either because it looks good or because you are unsure about it, which is
"optimism in the face of uncertainty".

#### Key equations

S&B's form (equation 2.10), with $t$ counting from 1 and $N_t(a)$ the number
of pulls of $a$ before time $t$:

$$
a_t = \operatorname*{arg\,max}_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right].
$$

An arm with $N_t(a) = 0$ counts as maximising. UCB1 (Auer, Cesa-Bianchi, and
Fischer, 2002) is the case $c = \sqrt{2}$ for rewards in $[0, 1]$:

$$
a_t = \operatorname*{arg\,max}_a \left[ \bar{x}_a + \sqrt{\frac{2 \ln t}{N_t(a)}} \right].
$$

The regret of a policy after $T$ pulls is the expected shortfall against always
pulling the best arm. It decomposes over arms by gap
$\Delta_a = \mu^* - \mu_a$:

$$
\mathcal{R}_T = T\mu^* - \mathbb{E}\left[\sum_{t=1}^{T} \mu_{a_t}\right]
= \sum_{a} \Delta_a \, \mathbb{E}[N_T(a)] .
$$

Auer et al. (2002), Theorem 1, bound UCB1's regret for rewards in $[0,1]$ at
every $T$:

$$
\mathcal{R}_T \le 8 \sum_{a:\Delta_a > 0} \frac{\ln T}{\Delta_a}
+ \left(1 + \frac{\pi^2}{3}\right) \sum_{a} \Delta_a .
$$

Lai and Robbins (1985) prove that no consistent policy does better
asymptotically. For every suboptimal arm,

$$
\liminf_{T \to \infty} \frac{\mathbb{E}[N_T(a)]}{\ln T} \ge
\frac{1}{\mathrm{KL}(\nu_a \,\Vert\, \nu^*)},
\quad\text{so}\quad
\liminf_{T\to\infty} \frac{\mathcal{R}_T}{\ln T} \ge
\sum_{a:\Delta_a>0} \frac{\Delta_a}{\mathrm{KL}(\nu_a \,\Vert\, \nu^*)} .
$$

For Bernoulli arms,
$\mathrm{KL}(p \,\Vert\, q) = p \ln\frac{p}{q} + (1-p)\ln\frac{1-p}{1-q}$.
Here "consistent" means the policy's regret is $o(T^\beta)$ for every
$\beta > 0$ on every instance.

#### Worked example

Two arms have been pulled 7 times in total. Auer et al. write the bonus with
$\ln n$, where $n$ is the number of pulls so far, so the table uses $\ln 7$.
S&B's form uses $\ln t$ at the current step, which would be $\ln 8$ here. State
which convention the exercise uses, because both appear in the literature.

| Arm | $\bar{x}_a$ | $N(a)$ | UCB1 bonus $\sqrt{2\ln 7 / N}$ | Score |
|---|---|---|---|---|
| 1 | 0.6 | 5 | 0.8822 | 1.4822 |
| 2 | 0.4 | 2 | 1.3950 | 1.7950 |

UCB1 picks arm 2 even though its estimate is lower. With $\ln 7 = 1.945910$,
the bonuses are $\sqrt{0.778364} = 0.882249$ and $\sqrt{1.945910} = 1.394959$.

For the constants in a regret check, take Bernoulli means $(0.5, 0.6)$, so
$\Delta = 0.1$:

- $\mathrm{KL}(0.5 \Vert 0.6) = 0.020411$, so the Lai-Robbins coefficient is
  $\Delta/\mathrm{KL} = 4.899$ per unit of $\ln T$.
- The UCB1 bound coefficient is $8/\Delta = 80$ per unit of $\ln T$.

A seeded simulation (10 seeds) gave the following mean regret:

| $T$ | UCB1 | $\varepsilon$-greedy, $\varepsilon = 0.1$ | Thompson sampling |
|---|---|---|---|
| 1,000 | 23.1 | 23.3 | 7.5 |
| 5,000 | 57.5 | 48.0 | 18.1 |
| 10,000 | 78.1 | 73.0 | 20.8 |
| 20,000 | 102.4 | 123.9 | 23.5 |

Between $T = 10{,}000$ and $20{,}000$, $\varepsilon$-greedy adds 50.9. Its
predicted linear rate is $\varepsilon \Delta (K-1)/K = 0.005$ per step, which
is 50 per 10,000 steps. UCB1 adds 24.3, close to the 20.6 it added between
5,000 and 10,000, which is the signature of logarithmic growth. The whole
simulation took under two seconds in pure Python.

#### References

- Auer, Cesa-Bianchi, and Fischer, "Finite-time analysis of the multiarmed
  bandit problem," *Machine Learning* 47, 235-256, 2002,
  [doi:10.1023/A:1013689704352](https://doi.org/10.1023/A:1013689704352).
- Lai and Robbins, "Asymptotically efficient adaptive allocation rules,"
  *Advances in Applied Mathematics* 6(1), 4-22, 1985,
  [doi:10.1016/0196-8858(85)90002-8](https://doi.org/10.1016/0196-8858(85)90002-8).
- Lattimore and Szepesvári, *Bandit Algorithms*, Cambridge University Press,
  2020. Free draft at [tor-lattimore.com/downloads/book/book.pdf](https://tor-lattimore.com/downloads/book/book.pdf)
  (URL UNVERIFIED in this session).
- S&B, Section 2.7.

#### Exercise idea

Use `rlenvs.BernoulliBandit([0.5, 0.6], seed)` and its `.regret(arm)`. The
checks are as follows:

- `ucb1_scores(means, counts, t)` returns the score vector. Check it against
  the worked-example table to `atol=1e-6`, and check that an unpulled arm
  scores `inf`.
- The mean UCB1 regret over 20 seeds stays below the Auer bound at
  $T \in \{1000, 5000, 20000\}$. The margin is roughly a factor of eight, so
  this check is robust.
- Growth shape: the ratio
  $[\mathcal{R}(20000) - \mathcal{R}(10000)] / [\mathcal{R}(10000) - \mathcal{R}(5000)]$
  is below 1.6 for UCB1. For $\varepsilon$-greedy, the increment
  $\mathcal{R}(20000) - \mathcal{R}(10000)$ is within 30 percent of
  $\varepsilon\Delta(K-1)/K \cdot 10000$. Tune both thresholds under at least
  five seeds before fixing the seed, as the contract requires.
- Thompson sampling's regret at $T = 20000$ is below UCB1's. This held with a
  factor-of-four margin in the simulation.

#### Common mistakes

- Using $\ln N(a)$ instead of $\ln t$ in the bonus.
- Forgetting to pull each arm once first, which divides by zero.
- Plotting regret on a linear time axis and calling it "logarithmic". Plot
  against $\ln T$ and look for a straight line.
- Expecting UCB1 to beat $\varepsilon$-greedy at small $T$. With a small gap,
  UCB1 explores heavily at first and can trail a well-tuned
  $\varepsilon$-greedy for thousands of steps, as the table shows at
  $T = 5000$.

### 2.3 Thompson sampling

#### Intuition

Keep a posterior belief over each arm's mean. Each round, draw one plausible
mean per arm from its posterior and pull the arm whose draw is largest. Arms
are pulled in proportion to the probability that they are best, so exploration
fades exactly as certainty grows.

#### Key equations

For Bernoulli arms with a $\mathrm{Beta}(\alpha_0, \beta_0)$ prior, after
$S_a$ successes and $F_a$ failures the posterior is:

$$
\theta_a \mid \text{data} \sim \mathrm{Beta}(\alpha_0 + S_a,\; \beta_0 + F_a).
$$

Each round draws $\tilde\theta_a$ from each posterior and plays
$a_t = \operatorname*{arg\,max}_a \tilde\theta_a$. The probability of playing
arm $a$ equals the posterior probability that $a$ is the best arm.

#### Worked example

Arm A has 3 successes and 1 failure, so its posterior is
$\mathrm{Beta}(4, 2)$ with mean $2/3$. Arm B has 1 of each, so
$\mathrm{Beta}(2, 2)$ with mean $1/2$. The probability that Thompson sampling
plays A is:

$$
\Pr(\theta_A > \theta_B) = \int_0^1 20\,x^3(1-x)\,\bigl(3x^2 - 2x^3\bigr)\,dx = \frac{5}{7} \approx 0.714286 .
$$

The integrand uses the $\mathrm{Beta}(4,2)$ density $20x^3(1-x)$ and the
$\mathrm{Beta}(2,2)$ CDF $3x^2 - 2x^3$. Numerical integration gave
0.7142857.

#### References

- Thompson, "On the likelihood that one unknown probability exceeds another in
  view of the evidence of two samples," *Biometrika* 25(3-4), 285-294, 1933,
  [doi:10.1093/biomet/25.3-4.285](https://doi.org/10.1093/biomet/25.3-4.285).
- Agrawal and Goyal, "Analysis of Thompson sampling for the multi-armed
  bandit problem," COLT 2012, [arXiv:1111.1797](https://arxiv.org/abs/1111.1797).
  First logarithmic regret bound for Thompson sampling.
- Kaufmann, Korda, and Munos, "Thompson sampling: An asymptotically optimal
  finite-time analysis," ALT 2012, [arXiv:1205.4217](https://arxiv.org/abs/1205.4217).
  Shows the Lai-Robbins constant is attained for Bernoulli rewards.
- Russo, Van Roy, Kazerouni, Osband, and Wen, "A tutorial on Thompson
  sampling," *Foundations and Trends in Machine Learning*, 2018,
  [arXiv:1707.02038](https://arxiv.org/abs/1707.02038).

#### Exercise idea

- `beta_posterior(successes, failures, prior=(1, 1))`: exact parameter check
  after a fixed reward sequence.
- `prob_best_mc(alpha, beta, n, seed)`: a Monte Carlo estimate of
  $\Pr(\theta_A > \theta_B)$. Check it against $5/7$ within 0.005 with
  $n = 200{,}000$. The standard error is about 0.001.
- Regret check against UCB1, as in section 2.2.

#### Common mistakes

- Updating the posterior of every arm instead of only the pulled arm.
- Using the posterior mean instead of a sample. That is greedy, not Thompson
  sampling, and it can lock onto a bad arm.
- Using one shared random draw for all arms.

### 2.4 Gradient bandits

#### Intuition

Keep a numerical preference for each arm and turn preferences into
probabilities with a softmax. After each reward, raise the preference of the
pulled arm if the reward beat a baseline and lower it otherwise, pushing the
other arms the opposite way. This is stochastic gradient ascent on expected
reward, and it is the one-state version of REINFORCE.

#### Key equations

The policy and the update (S&B equations 2.11 and 2.12, in course notation)
are:

$$
\pi_t(a) = \frac{e^{H_t(a)}}{\sum_b e^{H_t(b)}}, \qquad
H_{t+1}(a) = H_t(a) + \alpha\,(r_t - \bar r_t)\,\bigl(\mathbf{1}\{a = a_t\} - \pi_t(a)\bigr).
$$

Here $\bar r_t$ is a running average of past rewards used as a baseline. The
2nd edition defines it as the average up to but not including time $t$. The
2015 draft included the current reward. Either choice gives a valid algorithm,
because a baseline that does not depend on $a_t$ leaves the expected update
unchanged.

The update is an unbiased sample of the exact gradient. With true means $q(x)$,

$$
\frac{\partial\, \mathbb{E}[r_t]}{\partial H_t(a)}
= \sum_x q(x)\,\pi_t(x)\bigl(\mathbf{1}\{a=x\} - \pi_t(a)\bigr)
= \mathbb{E}\Bigl[(r_t - b)\bigl(\mathbf{1}\{a=a_t\} - \pi_t(a)\bigr)\Bigr]
$$

for any constant $b$, because $\sum_x \pi_t(x)\bigl(\mathbf{1}\{a=x\} - \pi_t(a)\bigr) = 0$.

#### Worked example

Three arms, $H = (0, 0, 0)$, so $\pi = (1/3, 1/3, 1/3)$. The agent pulls arm 1
and gets $r = 1$, with baseline $0$ and $\alpha = 0.1$. The update is:

- Arm 1: $0 + 0.1 \cdot 1 \cdot (1 - 1/3) = 0.066667$.
- Arms 2 and 3: $0 - 0.1 \cdot 1 \cdot 1/3 = -0.033333$.

The new policy is $(0.355913, 0.322043, 0.322043)$.

#### References

- S&B, Section 2.8.
- Williams, "Simple statistical gradient-following algorithms for
  connectionist reinforcement learning," *Machine Learning* 8, 229-256, 1992,
  [doi:10.1007/BF00992696](https://doi.org/10.1007/BF00992696) (DOI
  UNVERIFIED in this session).

#### Exercise idea

- `softmax(H)`: check that rows sum to one and that it is shift invariant.
- `gradient_bandit_step(H, a, r, baseline, alpha)`: check against the worked
  example.
- `expected_update(H, q)`: sum the update over all actions weighted by
  $\pi(a)$ with reward $q(a)$. Check it against a finite-difference gradient
  of $J(H) = \sum_a \pi_H(a) q(a)$ with `rlcheck.finite_difference`, for two
  different baselines. This is the contract's preferred style of check.
- Learning check: on `GaussianBandit` with means shifted by $+4$, the
  version with a baseline reaches a higher probability on the best arm after
  1,000 steps than the version without one. S&B Figure 2.5 shows this
  effect.

#### Common mistakes

- Leaving the baseline out and then shifting all rewards by a constant. With
  no baseline, the learning curve changes badly.
- Updating only the chosen arm and forgetting the $-\pi_t(a)$ term for the
  others.
- Overflow in the softmax. Subtract the maximum preference first.

### 2.5 Contextual bandits and LinUCB

#### Intuition

In a contextual bandit you see a feature vector before choosing, for example
a user profile. The best arm depends on the context. LinUCB assumes each arm's
expected reward is linear in the features, fits the weights by ridge
regression, and adds an uncertainty bonus that is large in feature directions
it has seen rarely.

#### Key equations

Disjoint LinUCB (Li, Chu, Langford, and Schapire, 2010, Algorithm 1) keeps,
per arm $a$, a matrix $A_a$ and a vector $b_a$:

$$
A_a = I_d + \sum_{\tau:\, a_\tau = a} x_\tau x_\tau^\top, \qquad
b_a = \sum_{\tau:\, a_\tau = a} r_\tau x_\tau, \qquad
\hat\theta_a = A_a^{-1} b_a .
$$

The score for context $x_{t,a}$ is the ridge prediction plus a bonus, and the
agent picks the arm with the largest score:

$$
p_{t,a} = \hat\theta_a^\top x_{t,a} + \alpha \sqrt{x_{t,a}^\top A_a^{-1} x_{t,a}} .
$$

The bonus is the width of a confidence ellipsoid around $\hat\theta_a$. OFUL
(Abbasi-Yadkori, Pál, and Szepesvári, 2011) gives the version with a proven
confidence radius.

#### Worked example

One arm with $d = 2$ has seen $x = (1, 0)$ with reward 1 and $x = (1, 1)$
with reward 0. Then:

- $A = I + \begin{bmatrix}1&0\\0&0\end{bmatrix} + \begin{bmatrix}1&1\\1&1\end{bmatrix} = \begin{bmatrix}3&1\\1&2\end{bmatrix}$
  and $b = (1, 0)$.
- $A^{-1} = \frac{1}{5}\begin{bmatrix}2&-1\\-1&3\end{bmatrix}$, so
  $\hat\theta = (0.4, -0.2)$.
- For a new context $x = (1, 0.5)$, the mean is $0.4 - 0.1 = 0.3$. The
  quadratic form is $x^\top A^{-1} x = (2 - 1 + 0.75)/5 = 0.35$, so the bonus
  is $\alpha \cdot 0.5916$.

#### References

- Li, Chu, Langford, and Schapire, "A contextual-bandit approach to
  personalized news article recommendation," WWW 2010,
  [arXiv:1003.0146](https://arxiv.org/abs/1003.0146).
- Abbasi-Yadkori, Pál, and Szepesvári, "Improved algorithms for linear
  stochastic bandits," NeurIPS 2011,
  [proceedings page](https://papers.nips.cc/paper/2011/hash/e1d5be1c7f2f456670de3d53c7b54f4a-Abstract.html)
  (URL UNVERIFIED in this session).
- S&B, Section 2.9, for the term "associative search".

#### Exercise idea

The exercise defines a linear contextual bandit in the starter file with a
fixed $\theta^*$ per arm and Gaussian noise. The checks are as follows:

- $\hat\theta$ matches `np.linalg.solve(X.T @ X + I, X.T @ y)` to
  `atol=1e-10`.
- A Sherman-Morrison rank-one update of $A^{-1}$ matches `np.linalg.inv` to
  `atol=1e-10`.
- The worked example's mean 0.3 and bonus 0.5916 match.
- Over 2,000 rounds, LinUCB's cumulative regret is below half that of a
  context-blind UCB1 and below a tenth of uniform random play. Tune the
  thresholds under five seeds.

#### Common mistakes

- Sharing one $A$ across arms in the disjoint model.
- Inverting $A$ from scratch every step. It works but costs $O(d^3)$.
- Using $\alpha = 0$ and calling it LinUCB.

## 3. Markov decision processes and Bellman equations

### Intuition

An MDP adds states to the bandit. An action changes both the reward and which
state comes next, so a good action now can be worth a bad reward. The Bellman
equation says the value of a state equals the expected immediate reward plus
the discounted value of where you land. For a fixed policy this is a linear
system you can solve exactly.

### Key equations

The Bellman expectation equations, in course notation, are:

$$
V^\pi(s) = \sum_a \pi(a \mid s)\Bigl[R(s,a) + \gamma \sum_{s'} P(s' \mid s,a)\, V^\pi(s')\Bigr],
$$

$$
Q^\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s' \mid s,a) \sum_{a'} \pi(a' \mid s')\, Q^\pi(s',a').
$$

In matrix form, with $P_\pi[s,s'] = \sum_a \pi(a \mid s) P(s' \mid s,a)$ and
$r_\pi[s] = \sum_a \pi(a \mid s) R(s,a)$:

$$
V^\pi = r_\pi + \gamma P_\pi V^\pi
\quad\Longrightarrow\quad
V^\pi = (I - \gamma P_\pi)^{-1} r_\pi .
$$

The inverse exists for $\gamma < 1$ because $P_\pi$ is a stochastic matrix, so
every eigenvalue of $\gamma P_\pi$ has modulus at most $\gamma$. The numpy form
is `np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)`.

The Bellman optimality equations are:

$$
V^*(s) = \max_a \Bigl[R(s,a) + \gamma \sum_{s'} P(s' \mid s,a)\, V^*(s')\Bigr],
\qquad
Q^*(s,a) = R(s,a) + \gamma \sum_{s'} P(s' \mid s,a) \max_{a'} Q^*(s',a').
$$

These are nonlinear because of the max, so no single linear solve gives $V^*$.

Define the Bellman operators $(T^\pi V)(s)$ and $(T^* V)(s)$ as the right-hand
sides. Both are $\gamma$-contractions in the max norm:

$$
\lVert T V - T U \rVert_\infty \le \gamma \lVert V - U \rVert_\infty .
$$

The proof idea for $T^*$ takes three sentences. The difference of two maxima is
at most the maximum of the differences. Each difference is $\gamma$ times an
average of $V - U$ over next states. An average never exceeds the largest
entry. By the Banach fixed-point theorem, each operator has a unique fixed
point, and repeated application converges to it geometrically.

The discounted state-visitation distribution from start distribution $\mu_0$
is:

$$
d^\pi = (1-\gamma)\,\mu_0^\top (I - \gamma P_\pi)^{-1} .
$$

This is what `rlenvs.state_distribution` computes. Lesson 04 lists it as an
exercise deliverable.

### Worked example

This two-state, two-action MDP is reused through sections 3 and 4:

| State | Action | Reward | Next state |
|---|---|---|---|
| $s_1$ | stay | 1 | $s_1$ |
| $s_1$ | go | 0 | $s_2$ |
| $s_2$ | stay | 2 | $s_2$ |
| $s_2$ | go | 0 | $s_1$ |

Use $\gamma = 0.9$. For the uniform random policy,
$P_\pi = \begin{bmatrix}0.5&0.5\\0.5&0.5\end{bmatrix}$ and $r_\pi = (0.5, 1)$.
Let $m$ be the average of the two values. Then $m = 0.75 + 0.9 m$, so
$m = 7.5$ and $V^\pi = (0.5 + 6.75,\ 1 + 6.75) = (7.25, 7.75)$. The
action values are $Q^\pi(s_1, \cdot) = (7.525, 6.975)$ and
$Q^\pi(s_2, \cdot) = (8.975, 6.525)$.

For the optimal values, staying in $s_2$ forever is worth
$2/(1-0.9) = 20$. From $s_1$, staying forever is worth $10$, while going is
worth $0 + 0.9 \cdot 20 = 18$. So $V^* = (18, 20)$ and the optimal policy is
go in $s_1$, stay in $s_2$. The optimal action values are
$Q^*(s_1,\cdot) = (17.2, 18)$ and $Q^*(s_2,\cdot) = (20, 16.2)$. Check:
$\max(17.2, 18) = 18$ and $\max(20, 16.2) = 20$.

### References

- S&B, Sections 3.5 to 3.7.
- Bellman, *Dynamic Programming*, Princeton University Press, 1957.
- Puterman, 1994, Chapter 6, for the contraction and fixed-point results.

### Exercise idea

Use `rlenvs.random_mdp(n_states=6, n_actions=3, gamma=0.9, seed=0)`. The checks
are as follows:

- `policy_matrices(mdp, pi)` returns `P_pi` of shape `(S, S)` and `r_pi` of
  shape `(S,)`. Check that rows of `P_pi` sum to one.
- `evaluate_exact(mdp, pi)` matches `rlenvs.policy_evaluation` and the worked
  example's $(7.25, 7.75)$.
- The Bellman residual $\lVert r_\pi + \gamma P_\pi V - V \rVert_\infty$ is
  below `1e-10`.
- `q_from_v` satisfies $\sum_a \pi(a \mid s) Q(s,a) = V(s)$.
- Contraction: for 200 seeded random pairs $(U, V)$, the ratio
  $\lVert T^* U - T^* V\rVert_\infty / \lVert U - V \rVert_\infty$ never
  exceeds $\gamma + 10^{-12}$.
- The discounted visitation `d_pi` sums to one and satisfies
  $d^\pi = (1-\gamma)\mu_0 + \gamma\, d^\pi P_\pi$.

### Common mistakes

- Solving with `P_pi.T` instead of `P_pi`, or building `P_pi` with the wrong
  einsum axes. Use `np.einsum("sa,sat->st", pi, P)`.
- Forgetting to zero out terminal states, so a terminal state "earns" rewards
  forever.
- Believing the optimality equation is linear. The max makes it nonlinear.
- Confusing $V^*$ with $\max_\pi$ of the value at one state only. One policy is
  simultaneously optimal at every state in a finite discounted MDP.

## 4. Dynamic programming

### Intuition

When the model $P$ and $R$ are known, you can compute values by repeatedly
applying the Bellman equation as an update. Policy evaluation computes one
policy's value. Policy improvement makes the policy greedy with respect to that
value. Alternating the two, fully or partially, converges to the optimal
policy.

### Key equations

Iterative policy evaluation applies $V_{k+1} = T^\pi V_k$ and converges to
$V^\pi$ at rate $\gamma^k$.

The policy improvement theorem (S&B Section 4.2) states that if
$Q^\pi(s, \pi'(s)) \ge V^\pi(s)$ for every $s$, then
$V^{\pi'}(s) \ge V^\pi(s)$ for every $s$. The greedy policy
$\pi'(s) = \operatorname*{arg\,max}_a Q^\pi(s,a)$ satisfies the condition.

Policy iteration alternates exact evaluation with greedy improvement. It stops
when the policy no longer changes. Because a finite MDP has finitely many
deterministic policies and each step improves strictly until optimal, it
terminates in finitely many iterations.

Value iteration applies $V_{k+1} = T^* V_k$:

$$
V_{k+1}(s) = \max_a \Bigl[R(s,a) + \gamma \sum_{s'} P(s' \mid s,a) V_k(s')\Bigr].
$$

Two error bounds follow from the contraction. They give the stopping rule:

$$
\lVert V_k - V^* \rVert_\infty \le \gamma^k \lVert V_0 - V^* \rVert_\infty,
\qquad
\lVert V_{k+1} - V^* \rVert_\infty \le \frac{\gamma}{1-\gamma}\lVert V_{k+1} - V_k \rVert_\infty .
$$

The second bound follows from
$\lVert V_{k+1} - V^*\rVert \le \gamma \lVert V_k - V^* \rVert \le \gamma\bigl(\lVert V_k - V_{k+1}\rVert + \lVert V_{k+1} - V^*\rVert\bigr)$.

The greedy policy $\pi_V$ with respect to any $V$ loses at most

$$
\lVert V^* - V^{\pi_V} \rVert_\infty \le \frac{2\gamma}{1-\gamma} \lVert V - V^* \rVert_\infty ,
$$

per Singh and Yee (1994), who tighten a result of Bertsekas (1987).

Generalized policy iteration (GPI) is S&B's name for any interleaving of
partial evaluation and partial improvement. Value iteration is GPI with one
evaluation sweep per improvement.

### Worked example

Value iteration on the two-state MDP from section 3, starting at $V_0 = (0, 0)$:

| $k$ | $V_k(s_1)$ | $V_k(s_2)$ | Greedy action in $s_1$ | $\lVert V_k - V^*\rVert_\infty$ | $\gamma^k \lVert V_0 - V^* \rVert_\infty$ |
|---|---|---|---|---|---|
| 1 | $\max(1, 0) = 1$ | $\max(2, 0) = 2$ | stay | 18 | 18 |
| 2 | $\max(1.9, 1.8) = 1.9$ | $\max(3.8, 0.9) = 3.8$ | stay | 16.2 | 16.2 |
| 3 | $\max(2.71, 3.42) = 3.42$ | $5.42$ | go | 14.58 | 14.58 |
| 4 | 4.878 | 6.878 | go | 13.122 | 13.122 |

The greedy action in $s_1$ flips to the optimal "go" at $k = 3$, long before the
values converge. On this MDP the $\gamma^k$ bound is tight from $k = 1$.

Policy iteration from "stay everywhere":

1. Evaluate: $V = (1/0.1,\ 2/0.1) = (10, 20)$.
2. Improve: in $s_1$, stay gives $1 + 9 = 10$ and go gives $0 + 18 = 18$, so
   switch to go. In $s_2$, stay gives $20$ and go gives $9$, so keep stay.
3. Evaluate the new policy: $V = (0.9 \cdot 20, 20) = (18, 20)$.
4. Improve: no change. Stop with $V^* = (18, 20)$.

Policy iteration needed two evaluations. Value iteration needs about
$\ln(10^{-6}/20)/\ln 0.9 \approx 160$ sweeps to reach $10^{-6}$ accuracy.

### References

- S&B, Chapter 4.
- Howard, *Dynamic Programming and Markov Processes*, MIT Press, 1960. The
  origin of policy iteration.
- Singh and Yee, "An upper bound on the loss from approximate optimal-value
  functions," *Machine Learning* 16, 227-233, 1994,
  [doi:10.1023/A:1022693225949](https://doi.org/10.1023/A:1022693225949).
- Puterman, 1994, Chapter 6.

### Exercise idea

The checks run on `rlenvs.random_mdp(20, 4, 0.95, seed=0)` and the worked MDP:

- `iterative_policy_evaluation(mdp, pi, tol)` matches `evaluate_exact` within
  `tol / (1 - gamma)`.
- `value_iteration(mdp, tol)` returns $V$ and a history. Check that every
  iterate satisfies
  $\lVert V_k - V^* \rVert_\infty \le \gamma^k \lVert V_0 - V^* \rVert_\infty + 10^{-10}$.
- The stopping bound $\frac{\gamma}{1-\gamma}\lVert V_{k+1} - V_k\rVert$ upper
  bounds the true error at every iterate.
- `policy_iteration(mdp)` returns the same greedy policy as value iteration,
  and its value matches $V^*$ to `1e-8`. It takes fewer than 20 improvement
  steps on this MDP. Confirm the exact count on the seed while writing the
  check.
- The policy value is monotone: each policy-iteration step does not decrease
  $V^{\pi_k}(s)$ at any state, within `1e-10`.
- The worked-example history for $k = 1$ to $4$ matches exactly.

Every check is exact linear algebra, so the suite runs in well under a second.

### Common mistakes

- Updating $V$ in place during a sweep and then claiming the Jacobi error
  bound. In-place (Gauss-Seidel) sweeps also converge, often faster, but the
  per-iterate history differs.
- Stopping policy iteration on "values stopped changing" with a tolerance,
  while ties in the argmax make the policy oscillate between equal actions.
  Break ties consistently, for example with `rlenvs.greedy`, which picks the
  first maximiser.
- Stopping value iteration when $\lVert V_{k+1} - V_k \rVert < \epsilon$ and
  reporting $\epsilon$ as the error. The true error can be
  $\gamma/(1-\gamma)$ times larger, which is 19 times at $\gamma = 0.95$.

## 5. Monte Carlo methods and importance sampling

### 5.1 Monte Carlo prediction and control

#### Intuition

Without a model, you can estimate $V^\pi(s)$ by running episodes and averaging
the returns that followed visits to $s$. No bootstrapping is involved, so the
estimate is unbiased, but it waits until the episode ends and its variance
grows with episode length. For control, estimate $Q$ instead of $V$, since
acting greedily on $V$ needs a model.

#### Key equations

First-visit MC averages $G_t$ over the first visit to $s$ in each episode.
Every-visit MC averages over all visits. Both converge to $V^\pi(s)$. First
visit is unbiased. Every visit is biased for finite samples but consistent.

On-policy MC control uses an $\varepsilon$-soft policy. With
$\mathcal{A}(s)$ the action set,

$$
\pi(a \mid s) =
\begin{cases}
1 - \varepsilon + \dfrac{\varepsilon}{|\mathcal{A}(s)|} & a = \operatorname*{arg\,max}_{a'} Q(s,a') \\[4pt]
\dfrac{\varepsilon}{|\mathcal{A}(s)|} & \text{otherwise.}
\end{cases}
$$

This converges to the best $\varepsilon$-soft policy, not to the optimal
policy (S&B Section 5.4).

#### Worked example

One episode visits $A, B, A$ and then terminates, with rewards $r = (1, 0, 2)$
and $\gamma = 1$. The returns are $G_0 = 3$, $G_1 = 2$, and $G_2 = 2$.

- First-visit: $V(A) = G_0 = 3$ and $V(B) = 2$.
- Every-visit: $V(A) = (3 + 2)/2 = 2.5$ and $V(B) = 2$.

#### References

- S&B, Sections 5.1 to 5.4.
- Singh and Sutton, "Reinforcement learning with replacing eligibility
  traces," *Machine Learning* 22, 123-158, 1996,
  [doi:10.1007/BF00114726](https://doi.org/10.1007/BF00114726). This paper
  analyses first-visit against every-visit MC.

### 5.2 Off-policy prediction by importance sampling

#### Intuition

You collect episodes with a behaviour policy $b$ but want the value of a
different target policy $\pi$. Reweight each return by how much more or less
likely its actions were under $\pi$ than under $b$. Ordinary importance
sampling averages the reweighted returns. It is unbiased but can have huge,
even infinite, variance. Weighted importance sampling divides by the sum of
the weights instead. It is biased for finite samples but far more stable.

#### Key equations

The importance-sampling ratio from $t$ to the end of the episode is:

$$
\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(a_k \mid s_k)}{b(a_k \mid s_k)} .
$$

The transition probabilities cancel, so the ratio needs no model. This
requires coverage: $\pi(a \mid s) > 0$ must imply $b(a \mid s) > 0$.

Let $\mathcal{T}(s)$ be the set of visit times to $s$ and $T(t)$ the end of the
episode containing $t$. The two estimators are (S&B equations 5.5 and 5.6):

$$
V_{\text{OIS}}(s) = \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1}\, G_t}{|\mathcal{T}(s)|},
\qquad
V_{\text{WIS}}(s) = \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1}\, G_t}{\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1}} .
$$

WIS is a convex combination of observed returns, so it always lies between the
smallest and the largest observed return. The incremental form of WIS (S&B
Section 5.6) keeps a cumulative weight $C$:

$$
C \leftarrow C + W, \qquad V \leftarrow V + \frac{W}{C}\,(G - V).
$$

#### Worked example: one step

A one-step problem has actions $L$ and $R$. $L$ pays 1 and $R$ pays 0. The
target policy has $\pi(L) = 0.9$, so $V^\pi = 0.9$. The behaviour policy has
$b(L) = 0.5$. The ratios are $\rho(L) = 0.9/0.5 = 1.8$ and
$\rho(R) = 0.1/0.5 = 0.2$.

Three behaviour episodes give $(L, 1)$, $(R, 0)$, and $(L, 1)$:

- OIS: $(1.8 + 0 + 1.8)/3 = 1.2$. This is outside the range of possible
  returns.
- WIS: $3.6/(1.8 + 0.2 + 1.8) = 0.947$.

The per-sample variance of OIS is
$\mathbb{E}_b[(\rho G)^2] - (V^\pi)^2 = 0.5 \cdot 1.8^2 - 0.81 = 0.81$.

#### Worked example: infinite variance

S&B Example 5.5 has one nonterminal state $s$ and two actions. One action ends
the episode with reward 0. The other, which the book calls "back" in the 2015
draft, returns to $s$ with probability 0.9 and otherwise terminates with reward
$+1$. The target policy always picks back, so $V^\pi(s) = 1$. The behaviour
policy picks each action with probability 0.5.

An episode that loops $k$ times and then terminates with reward has
probability $(0.5 \cdot 0.9)^k (0.5 \cdot 0.1)$ under $b$ and weight
$\rho = 2^{k+1}$. The second moment of the OIS sample is therefore:

$$
\mathbb{E}_b\bigl[(\rho G)^2\bigr] = \sum_{k=0}^{\infty} 0.05 \cdot 0.45^k \cdot 4^{k+1}
= 0.2 \sum_{k=0}^{\infty} 1.8^k = \infty .
$$

The mean is still exactly 1, so OIS is unbiased with infinite variance. WIS
returns exactly 1 after the first episode that ends with the back action,
because every such episode has return 1.

#### References

- S&B, Sections 5.5 to 5.9, including discounting-aware and per-decision
  importance sampling.
- Precup, Sutton, and Singh, "Eligibility traces for off-policy policy
  evaluation," ICML 2000, 759-766,
  [ACM entry](https://dl.acm.org/doi/10.5555/645529.658134).
- Mahmood, van Hasselt, and Sutton, "Weighted importance sampling for
  off-policy learning with linear function approximation," NeurIPS 2014,
  [ACM entry](https://dl.acm.org/doi/10.5555/2969033.2969163).

### 5.3 Exercise idea for Monte Carlo

The checks are as follows:

- `first_visit_mc` on the five-state random walk of S&B Example 6.2 converges
  to the true values $(1/6, 2/6, 3/6, 4/6, 5/6)$. The walk starts in the
  centre, moves left or right with equal probability, and pays 1 only on the
  right exit. Check within four standard errors after 2,000 seeded episodes.
- `returns_first_and_every_visit` reproduces the $A, B, A$ worked example
  exactly.
- `importance_ratios(episode, pi, b)` matches a hand-computed product.
- Across 2,000 seeded batches of 10 one-step episodes, the OIS mean is within
  three standard errors of 0.9, confirming unbiasedness. The WIS variance
  across batches is lower than the OIS variance, and every WIS estimate lies
  in $[0, 1]$.
- `truncated_second_moment(K)` for the infinite-variance example equals
  $0.2 \sum_{k<K} 1.8^k$ exactly, and grows without bound in $K$.
- On-policy $\varepsilon$-soft MC control on a small `gridworld` produces a
  policy whose exact value, computed with `rlenvs.policy_evaluation`, is
  within a tolerance of the best $\varepsilon$-soft value.

### 5.4 Common mistakes

- Multiplying ratios in the wrong direction, $b/\pi$ instead of $\pi/b$.
- Accumulating $\rho$ from the start of the episode instead of from time $t$.
  S&B's backward loop avoids this: it multiplies $W$ by the ratio after using
  it.
- In off-policy MC control with a greedy target, forgetting to break out of
  the backward loop when $a_t \ne \pi(s_t)$. The ratio is zero from there on.
- Reporting OIS as "wrong" because a finite sample is far from the truth. It is
  unbiased; its variance is the problem.
- Coverage violations that silently produce `0/0` in WIS.

## 6. Temporal-difference learning

### 6.1 TD(0) prediction

#### Intuition

TD learning updates a state's value toward the reward it received plus the
estimated value of the next state, after every single step. It does not wait
for the episode to end. It bootstraps from its own current guesses, which adds
bias but usually cuts variance a lot.

#### Key equations

The TD(0) update and the TD error are:

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad
V(s_t) \leftarrow V(s_t) + \alpha\, \delta_t .
$$

If $V$ is held fixed during an episode, the Monte Carlo error is a discounted
sum of TD errors (S&B equation 6.6):

$$
G_t - V(s_t) = \sum_{k=t}^{T-1} \gamma^{k-t} \delta_k .
$$

Batch TD(0) converges to the certainty-equivalence estimate. That is the exact
value of the maximum-likelihood Markov model fitted to the data. Batch MC
converges to the estimate that minimises squared error on the observed returns.

#### Worked example

S&B Example 6.4 uses eight episodes: one episode $A \to B$ with rewards 0 and
0, six episodes that start in $B$ and end with reward 1, and one episode that
starts in $B$ and ends with reward 0.

- Batch MC gives $V(B) = 6/8 = 0.75$ and $V(A) = 0$, because the only return
  from $A$ was 0.
- Batch TD(0) gives $V(B) = 0.75$ and $V(A) = 0.75$, because $A$ always
  moves to $B$ with reward 0.

A single TD step with $V(B) = 0.5$, reward 1 on termination, and
$\alpha = 0.1$ gives $V(B) \leftarrow 0.5 + 0.1(1 + 0 - 0.5) = 0.55$.

#### References

- Sutton, "Learning to predict by the methods of temporal differences,"
  *Machine Learning* 3, 9-44, 1988,
  [doi:10.1007/BF00115009](https://doi.org/10.1007/BF00115009).
- S&B, Sections 6.1 to 6.3.

### 6.2 SARSA, Q-learning, and Expected SARSA

#### Intuition

All three update $Q(s,a)$ toward the reward plus a discounted estimate of the
next state's value. They differ in which next-state estimate they use. SARSA
uses the action it will actually take. Q-learning uses the best action. Expected
SARSA uses the average under its policy. That one choice decides whether the
method learns about the policy it follows (on-policy) or about the greedy
policy (off-policy).

#### Key equations

Each method updates
$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\bigl[y_t - Q(s_t, a_t)\bigr]$ with
a different target $y_t$:

$$
\begin{aligned}
\text{SARSA:}\quad & y_t = r_t + \gamma\, Q(s_{t+1}, a_{t+1}), \\
\text{Q-learning:}\quad & y_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a'), \\
\text{Expected SARSA:}\quad & y_t = r_t + \gamma \sum_{a'} \pi(a' \mid s_{t+1})\, Q(s_{t+1}, a').
\end{aligned}
$$

On a terminal transition, $y_t = r_t$. Expected SARSA with a greedy target
policy is Q-learning. In a deterministic environment, Expected SARSA's update
has zero variance from action selection, so it tolerates $\alpha = 1$ (van
Seijen et al., 2009).

Tabular Q-learning converges to $Q^*$ with probability one if every
state-action pair is updated infinitely often and the step sizes satisfy the
Robbins-Monro conditions (Watkins and Dayan, 1992). SARSA converges to $Q^*$
if, in addition, the policy becomes greedy in the limit while still exploring
infinitely often (Singh et al., 2000, "Convergence results for single-step
on-policy reinforcement-learning algorithms," *Machine Learning* 38; citation
details UNVERIFIED in this session).

#### Worked example

Take $Q(s,a) = 0$, $r = -1$, $\gamma = 1$, $\alpha = 0.5$, and
$Q(s', \cdot) = (2, 5)$. The $\varepsilon$-greedy policy with $\varepsilon = 0.2$
over two actions puts probability $0.9$ on action 2 and $0.1$ on action 1.
Suppose the sampled next action is action 1.

- SARSA: $0 + 0.5(-1 + 2 - 0) = 0.5$.
- Q-learning: $0 + 0.5(-1 + 5 - 0) = 2.0$.
- Expected SARSA: $0 + 0.5(-1 + 0.1 \cdot 2 + 0.9 \cdot 5) = 1.85$.

On S&B's cliff walk (Example 6.6: a 4 by 12 grid, reward $-1$ per step,
$-100$ and a reset to start for stepping into the cliff, $\varepsilon = 0.1$),
Q-learning's greedy policy follows the cliff edge in 13 steps, which is worth
$-13$. SARSA learns a safer, longer path, because its values include the cost
of its own exploratory slips. During training, SARSA's online return is
higher, because Q-learning's exploratory steps along the edge sometimes fall
off.

#### References

- Watkins, *Learning from Delayed Rewards*, PhD thesis, University of
  Cambridge, 1989.
- Watkins and Dayan, "Q-learning," *Machine Learning* 8, 279-292, 1992,
  [doi:10.1007/BF00992698](https://doi.org/10.1007/BF00992698).
- Rummery and Niranjan, "On-line Q-learning using connectionist systems,"
  Technical Report CUED/F-INFENG/TR 166, Cambridge University, 1994. The
  origin of SARSA, under the name "modified connectionist Q-learning".
- van Seijen, van Hasselt, Whiteson, and Wiering, "A theoretical and empirical
  analysis of Expected Sarsa," IEEE ADPRL 2009, 177-184,
  [PDF](https://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/vanseijenadprl09.pdf).
- S&B, Sections 6.4 to 6.6.

### 6.3 Maximisation bias and double Q-learning

#### Intuition

If you take the maximum of several noisy estimates, you tend to pick one whose
noise happened to be positive. The maximum of the estimates overestimates the
maximum of the true values, even when each estimate alone is unbiased.
Double learning keeps two independent estimates. One picks the best action and
the other scores it, so the selection noise and the evaluation noise are
independent.

#### Key equations

Because $\max$ is convex, Jensen's inequality gives:

$$
\mathbb{E}\Bigl[\max_a \hat Q(a)\Bigr] \ge \max_a \mathbb{E}\bigl[\hat Q(a)\bigr] .
$$

Double Q-learning (van Hasselt, 2010) keeps $Q_1$ and $Q_2$. On each step, with
probability $0.5$, it updates $Q_1$:

$$
Q_1(s_t,a_t) \leftarrow Q_1(s_t,a_t) + \alpha\Bigl[r_t + \gamma\, Q_2\bigl(s_{t+1}, \operatorname*{arg\,max}_{a'} Q_1(s_{t+1}, a')\bigr) - Q_1(s_t,a_t)\Bigr].
$$

Otherwise it updates $Q_2$ with the roles swapped. The behaviour policy is
$\varepsilon$-greedy on $Q_1 + Q_2$. The double estimator can underestimate,
but it removes the systematic upward bias.

#### Worked example

Two actions with true value 0 each have one estimate drawn from
$\mathcal{N}(0, 1)$. The expected maximum of two independent standard normals
is:

$$
\mathbb{E}[\max(X_1, X_2)] = \frac{1}{\sqrt{\pi}} \approx 0.5642 .
$$

A seeded simulation with $10^6$ draws gave 0.5653. The double estimator picks
the argmax with $X$ and reads the value from an independent $Y$, so its
expectation is exactly 0 here.

S&B Example 6.7 (Figure 6.5) makes this an MDP. From state A, "right"
terminates with reward 0 and "left" moves to B with reward 0. From B, each of
many actions terminates with reward drawn from $\mathcal{N}(-0.1, 1)$. The
settings are $\varepsilon = 0.1$, $\alpha = 0.1$, and $\gamma = 1$. "Left" is
worse in expectation. The reference implementation by Zhang uses 10 actions in
B. A seeded replication with 1,000 runs gave the following fractions of "left"
choices:

| Episodes | Q-learning | Double Q-learning |
|---|---|---|
| 1 to 10 | 0.69 | 0.50 |
| 11 to 30 | 0.92 | 0.33 |
| 281 to 300 | 0.12 | 0.071 |

The floor for $\varepsilon$-greedy with two actions is 0.05. The replication
took about 1.4 seconds per method in pure Python.

#### References

- van Hasselt, "Double Q-learning," NeurIPS 2010, 2613-2621,
  [PDF](https://papers.nips.cc/paper_files/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf).
- van Hasselt, Guez, and Silver, "Deep reinforcement learning with double
  Q-learning," AAAI 2016, [arXiv:1509.06461](https://arxiv.org/abs/1509.06461).
  The bridge to lesson 13.
- S&B, Section 6.7.
- Zhang, reference code for S&B figures,
  [`chapter06/maximization_bias.py`](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter06/maximization_bias.py).

### 6.4 Exercise ideas for TD

The following checks cover lessons 07 and 08:

- `td0_update` against a hand-computed step, as in the contract's own example.
- Batch TD(0) on the eight-episode dataset converges to $V(A) = 0.75$, and
  batch MC gives $V(A) = 0$, both to `1e-6`. Batch updates repeat sweeps over
  the fixed dataset with a small $\alpha$ until the values stop changing.
- `certainty_equivalence(dataset)` builds the MLE model and solves it exactly.
  Check it matches batch TD.
- TD(0) on the five-state random walk has lower RMS error than constant-step
  MC after 100 episodes, averaged over 50 seeded runs, starting from
  $V = 0.5$. This is S&B's Example 6.2 effect. A replication under five seed
  blocks gave 0.035 to 0.040 for TD at $\alpha = 0.05$ and 0.083 to 0.088 for
  MC at $\alpha = 0.02$, in 0.3 seconds for all runs. A threshold such as
  "TD below 0.06, MC above 0.07" has margin.
- Q-learning on `rlenvs.cliff_walking()` with 500 episodes, $\alpha = 0.5$,
  and $\varepsilon = 0.1$ produces a greedy policy whose rollout from the
  start earns exactly $-13$ in 13 steps. A replication under five seeds gave
  $-13$ every time, in about 0.5 seconds per learner.
- Evaluate the greedy policy with a capped rollout, not with
  `rlenvs.policy_evaluation`. `cliff_walking()` has $\gamma = 1$. Any state
  whose greedy action bumps into a wall, such as a rarely visited top-row
  state with all-zero $Q$ picking "up", makes $I - P_\pi$ singular. In the
  replication, `policy_evaluation(mdp, greedy(Q))` raised `LinAlgError` for
  all five Q-learning seeds. The contract's own example check in
  `docs/CONTENT_CONTRACT.md` uses this call and is likely to hit the same
  error. Alternatives are a rollout from the start with a step cap, or exact
  evaluation on `cliff_walking(gamma=0.99)`.
- SARSA's greedy policy is less reliable as a check target. In the
  replication, three seeds gave a safe 17-step path worth $-17$, and two gave
  a greedy policy that looped until the cap. A robust SARSA check is that its
  mean online return over the last 100 training episodes is higher than
  Q-learning's, which is S&B Figure 6.4's effect, and that its greedy rollout
  never enters the cliff. Five seeds gave SARSA $-24.2$ to $-27.8$ and
  Q-learning $-42.1$ to $-60.7$, so a check such as "SARSA above $-35$,
  Q-learning below $-35$" has margin.
- `expected_sarsa_target` matches the worked example's 1.85.
- `max_bias(n_actions, n_samples, seed)` estimates $\mathbb{E}[\max]$ for two
  actions within 0.01 of $1/\sqrt{\pi}$.
- On the Example 6.7 MDP with 300 seeded runs, Q-learning's mean left fraction
  over episodes 1 to 30 exceeds 0.6, and double Q-learning's mean over
  episodes 281 to 300 is below 0.1.

### 6.5 Common mistakes

- Bootstrapping from a terminal state. The fix is
  `target = r + gamma * V[s2] * (1 - terminated)`.
- In SARSA, choosing $a_{t+1}$ for the update and then choosing a different
  action to execute. The same $a_{t+1}$ must be both.
- In double Q-learning, using $\max_{a'} Q_2(s', a')$ instead of
  $Q_2(s', \operatorname*{arg\,max}_{a'} Q_1(s',a'))$. That is still a biased
  single estimator.
- Evaluating the learned policy by its $\varepsilon$-greedy training returns
  when the question is about the greedy policy, or the reverse. The cliff
  lesson hinges on this distinction.
- Declaring "Q-learning converges" with a constant $\alpha$. With constant
  $\alpha$ and stochastic rewards, the estimates keep fluctuating.

## 7. n-step methods, eligibility traces, and GAE

### Intuition

One-step TD bootstraps after one reward and Monte Carlo after all of them.
n-step methods sit in between: sum $n$ rewards, then bootstrap. The
$\lambda$-return averages all $n$-step returns with geometrically decaying
weights. Eligibility traces compute that average incrementally, by keeping a
fading memory of recently visited states and crediting each TD error back to
them.

### Key equations

The n-step return, in course notation, is:

$$
G_{t:t+n} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n})
\quad\text{if } t + n < T, \qquad
G_{t:t+n} = G_t \quad\text{if } t + n \ge T .
$$

It has an error-reduction property (S&B equation 7.3). The worst-case error of
its expectation is at most $\gamma^n$ times the worst-case error of $V$:

$$
\max_s \Bigl| \mathbb{E}_\pi[G_{t:t+n} \mid s_t = s] - V^\pi(s) \Bigr|
\le \gamma^n \max_s \bigl| V(s) - V^\pi(s) \bigr| .
$$

The $\lambda$-return in its episodic form (S&B equation 12.3) is:

$$
G_t^\lambda = (1-\lambda) \sum_{n=1}^{T-t-1} \lambda^{n-1} G_{t:t+n} + \lambda^{T-t-1} G_t .
$$

The weights sum to one. $\lambda = 0$ gives the one-step TD target and
$\lambda = 1$ gives the Monte Carlo return.

Semi-gradient TD($\lambda$) with accumulating traces (S&B Section 12.2) uses
the following updates, with $z_{-1} = 0$:

$$
z_t = \gamma\lambda\, z_{t-1} + \nabla_w \hat V(s_t, w_t), \qquad
\delta_t = r_t + \gamma \hat V(s_{t+1}, w_t) - \hat V(s_t, w_t), \qquad
w_{t+1} = w_t + \alpha\, \delta_t\, z_t .
$$

In the tabular case, $\nabla_w \hat V(s_t)$ is the one-hot vector of $s_t$.
Replacing traces (Singh and Sutton, 1996) set the trace of $s_t$ to 1 instead of
adding 1.

The offline equivalence is the key identity for lesson 09. If $V$ is held
fixed during an episode and the updates are summed and applied at the end,
accumulating-trace TD($\lambda$) produces exactly the same total change as
updating every visited state toward its $\lambda$-return:

$$
\sum_{t=0}^{T-1} \alpha\, \delta_t\, z_t = \sum_{t=0}^{T-1} \alpha\,\bigl(G^\lambda_t - V(s_t)\bigr)\, \mathbf{e}_{s_t} .
$$

A seeded numpy check on a random 12-step episode over 5 states matched to
$8.9 \times 10^{-16}$. Online updating breaks the exact match. True online
TD($\lambda$) (van Seijen and Sutton, 2014) restores an exact match with the
online $\lambda$-return for linear $\hat V = w^\top x$, using a dutch trace:

$$
z_t = \gamma\lambda z_{t-1} + \bigl(1 - \alpha\gamma\lambda\, z_{t-1}^\top x_t\bigr) x_t,
\qquad
w_{t+1} = w_t + \alpha\delta_t z_t + \alpha\bigl(w_t^\top x_t - w_{t-1}^\top x_t\bigr)(z_t - x_t).
$$

Here $\delta_t$ uses $w_t$ for both values. The dutch trace was confirmed from
secondary notes. The weight update is from memory of S&B equation 12.11 and is
UNVERIFIED against the book text in this session.

Generalized advantage estimation (Schulman et al., 2015) applies the same
weighting to advantages. With $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$:

$$
\hat A_t^{\mathrm{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l\, \delta_{t+l}
= G_t^\lambda - V(s_t) ,
$$

where $G_t^\lambda$ is the $\lambda$-return built from the same $V$. The limits
are $\mathrm{GAE}(\gamma, 0) = \delta_t$ and
$\mathrm{GAE}(\gamma, 1) = G_t - V(s_t)$. The backward recursion that code
uses is:

$$
\hat A_t = \delta_t + \gamma\lambda\,(1 - d_{t+1})\,\hat A_{t+1},
$$

where $d_{t+1} = 1$ when $s_{t+1}$ is terminal. On truncation by a time limit,
$\delta_t$ must still bootstrap from $V(s_{t+1})$, and the recursion is cut
there. PPO-style value targets $\hat A_t + V(s_t)$ are therefore
$\lambda$-returns. The GAE paper indexes rewards as $r_t$, the same as this
course.

### Worked example

Three steps: states $s_0, s_1, s_2$, then terminal. Rewards $r = (1, 0, 2)$,
$V = (0.5, 1.0, 0.8)$, $\gamma = 0.9$, and $\lambda = 0.5$.

The n-step returns from $t = 0$ are:

- $G_{0:1} = 1 + 0.9 \cdot 1.0 = 1.9$.
- $G_{0:2} = 1 + 0 + 0.81 \cdot 0.8 = 1.648$.
- $G_{0:3} = G_0 = 1 + 0 + 0.81 \cdot 2 = 2.62$.

The $\lambda$-return is
$G_0^\lambda = 0.5 \cdot 1.9 + 0.5 \cdot 0.5 \cdot 1.648 + 0.5^2 \cdot 2.62 = 0.95 + 0.412 + 0.655 = 2.017$.

The TD errors are $\delta_0 = 1 + 0.9 - 0.5 = 1.4$,
$\delta_1 = 0 + 0.72 - 1.0 = -0.28$, and $\delta_2 = 2 + 0 - 0.8 = 1.2$.
With $\gamma\lambda = 0.45$:

$$
\hat A_0 = 1.4 + 0.45(-0.28) + 0.2025(1.2) = 1.4 - 0.126 + 0.243 = 1.517 = G_0^\lambda - V(s_0).
$$

### References

- S&B, Chapter 7 (n-step bootstrapping, including tree backup in Section 7.5)
  and Chapter 12 (eligibility traces).
- Sutton, 1988, for TD($\lambda$).
- Singh and Sutton, 1996, for replacing traces.
- van Seijen and Sutton, "True online TD($\lambda$)," ICML 2014, PMLR 32,
  692-700, [proceedings page](https://proceedings.mlr.press/v32/seijen14.html).
  The JMLR version is van Seijen et al., "True online temporal-difference
  learning," 2016, [arXiv:1512.04087](https://arxiv.org/abs/1512.04087).
- Schulman, Moritz, Levine, Jordan, and Abbeel, "High-dimensional continuous
  control using generalized advantage estimation," ICLR 2016,
  [arXiv:1506.02438](https://arxiv.org/abs/1506.02438).

### Exercise idea

- `n_step_return(rewards, values, t, n, gamma)` matches the three worked
  values exactly, including the $t + n \ge T$ case.
- `lambda_return(rewards, values, gamma, lam)` matches 2.017, and matches the
  n-step return at $\lambda = 0$ and the MC return at $\lambda = 1$.
- `td_lambda_offline(episode, V, gamma, lam)` matches the forward
  $\lambda$-return update on a seeded random episode to `atol=1e-12`. This is
  the offline equivalence identity.
- `gae(rewards, values, dones, gamma, lam)` matches $G^\lambda - V$ to
  `1e-12`, and has the right limits at $\lambda \in \{0, 1\}$. A truncated
  episode must bootstrap from the final value.
- The error-reduction property: on `rlenvs.random_mdp`, the exact expected
  n-step return (a linear map of $V$) has error at most $\gamma^n$ times the
  error of $V$, for $n = 1, \dots, 5$.
- A soft learning check on the 19-state random walk of S&B Figure 7.2: an
  intermediate $n$ such as 4 beats both $n = 1$ and $n = 64$ in RMS error
  after 10 episodes, averaged over 50 seeded runs at a good $\alpha$ for each
  $n$. Tune under five seeds, because this effect has a small margin.

### Common mistakes

- Omitting the tail weight $\lambda^{T-t-1}$ on the Monte Carlo return, so the
  weights no longer sum to one.
- Letting traces carry across episode boundaries. Reset $z$ to zero at the
  start of each episode.
- Forgetting to decay traces by $\gamma\lambda$ before adding the current
  gradient, or decaying twice.
- In GAE, treating a time-limit truncation as termination. This drops
  $\gamma V(s_{t+1})$ from the last TD error, which biases every advantage
  near the time limit. Lesson 17 covers it.
- Assuming online TD($\lambda$) equals the $\lambda$-return method exactly. It
  does only offline, or with the true online variant.

## 8. Function approximation

### 8.1 Linear value functions and tile coding

#### Intuition

With too many states for a table, you represent $V$ with a parameter vector
$w$ and a feature vector $x(s)$. Updating $w$ for one state changes the value
of every state that shares features with it, which is generalisation. Tile
coding builds binary features by overlaying several offset grids. Exactly one
tile per grid is active, so the number of active features is fixed and step
sizes are easy to choose.

#### Key equations

The prediction objective is the $\mu$-weighted mean squared value error
(S&B equation 9.1):

$$
\overline{\mathrm{VE}}(w) = \sum_s \mu(s)\bigl[V^\pi(s) - \hat V(s, w)\bigr]^2 .
$$

Here $\mu$ is the on-policy state distribution. Gradient Monte Carlo is true
SGD on this objective:

$$
w \leftarrow w + \alpha\bigl[G_t - \hat V(s_t, w)\bigr]\nabla_w \hat V(s_t, w).
$$

Semi-gradient TD(0) replaces $G_t$ with a bootstrapped target but still takes
the gradient of the prediction only. It ignores the target's dependence on
$w$:

$$
w \leftarrow w + \alpha\bigl[r_t + \gamma \hat V(s_{t+1}, w) - \hat V(s_t, w)\bigr]\nabla_w \hat V(s_t, w).
$$

For tile coding with $m$ tilings and binary features, S&B suggest a step
size such as $\alpha = 1/(10m)$. This moves the prediction one tenth of the way to
the target per update. $\alpha = 1/m$ gives exact one-trial learning: after
one update, the prediction at the updated state equals the target.

#### Worked example

One input $x \in [0, 1)$ uses $m = 2$ tilings with tile width 0.5. Tiling 1
has tiles $[0, 0.5)$ and $[0.5, 1)$. Tiling 2 is offset by 0.25, so its tile
index is $\lfloor (x + 0.25)/0.5 \rfloor \in \{0, 1, 2\}$. Features are
numbered 0 and 1 for tiling 1, and 2 to 4 for tiling 2.

- For $x = 0.6$, the active features are 1 and 3, since
  $\lfloor 1.7 \rfloor = 1$.
- Starting from $w = 0$ with target 1 and $\alpha = 1/m = 0.5$, both active
  weights become 0.5, so $\hat V(0.6) = 1$ exactly.
- For $x = 0.3$, the active features are 0 and 3. It shares feature 3, so
  $\hat V(0.3) = 0.5$. That shared half is generalisation.

#### References

- S&B, Sections 9.1 to 9.5. Tile coding is Section 9.5.4.
- Albus, "A new approach to manipulator control: The cerebellar model
  articulation controller (CMAC)," *Journal of Dynamic Systems, Measurement,
  and Control* 97(3), 1975. The origin of tile coding (citation details
  UNVERIFIED in this session).
- Sutton's tile-coding software page, linked from the S&B book site.

### 8.2 The TD fixed point, the deadly triad, and Baird's counterexample

#### Intuition

Linear semi-gradient TD does not minimise $\overline{\mathrm{VE}}$. It converges
to a different point, the TD fixed point, whose error is bounded by a multiple
of the best achievable error. This guarantee needs on-policy data. Combine
function approximation, bootstrapping, and off-policy training (the "deadly
triad") and the weights can diverge to infinity, even when the true value is
exactly representable.

#### Key equations

For linear $\hat V(s,w) = w^\top x(s)$ under the on-policy distribution,
expected semi-gradient TD(0) converges to $w_{\text{TD}} = A^{-1} b$, with:

$$
A = \mathbb{E}\bigl[x_t (x_t - \gamma x_{t+1})^\top\bigr] = X^\top D (I - \gamma P_\pi) X,
\qquad
b = \mathbb{E}[r_t\, x_t] = X^\top D\, r_\pi .
$$

Here $X$ stacks the feature rows and $D = \mathrm{diag}(\mu)$. LSTD estimates
$A$ and $b$ from samples and solves directly. The error at the fixed point is
bounded as follows:

$$
\overline{\mathrm{VE}}(w_{\text{TD}}) \le \frac{1}{1-\gamma^2} \min_w \overline{\mathrm{VE}}(w)
\le \frac{1}{1-\gamma}\min_w \overline{\mathrm{VE}}(w) .
$$

S&B equation 9.14 states the looser right-hand form. The tighter
$1/(1-\gamma^2)$ follows in three steps. Let $\Pi$ be the $\mu$-weighted
projection onto the feature span. $X w_{\text{TD}}$ is the fixed point of
$\Pi T^\pi$, which is a $\gamma$-contraction in the $\mu$-norm. Pythagoras then
gives
$\lVert X w_{\text{TD}} - V^\pi\rVert^2 = \lVert \Pi V^\pi - V^\pi\rVert^2 + \lVert X w_{\text{TD}} - \Pi V^\pi\rVert^2$,
and the last term is at most $\gamma^2 \lVert X w_{\text{TD}} - V^\pi\rVert^2$.
Tsitsiklis and Van Roy (1997) prove the TD($\lambda$) version with factor
$(1-\lambda\gamma)/(1-\gamma)$ on the unsquared norm.

Baird's counterexample (S&B Section 11.2, Figure 11.1) has seven states and
eight weights, all rewards 0, and $\gamma = 0.99$:

- Upper states $i = 1, \dots, 6$ have $\hat V(s_i) = 2w_i + w_8$.
- The lower state has $\hat V(s_7) = w_7 + 2w_8$.
- The solid action moves to $s_7$. The dashed action moves to one of the six
  upper states uniformly.
- The target policy always takes solid. The behaviour policy takes dashed with
  probability $6/7$ and solid with $1/7$. The behaviour policy's state
  distribution is uniform.
- S&B's figure starts from $w = (1, 1, 1, 1, 1, 1, 10, 1)$ with
  $\alpha = 0.01$.

The true value is $V^\pi = 0$, which $w = 0$ represents exactly. With
$\rho = 7$ on solid and $0$ on dashed, the expected off-policy semi-gradient
update is $w \leftarrow w - \alpha A_{\text{B}} w$, where:

$$
A_{\text{B}} = \frac{1}{7} \sum_{s=1}^{7} x(s)\bigl(x(s) - \gamma\, x(s_7)\bigr)^\top .
$$

This is also S&B's "semi-gradient DP" update in Figure 11.2.

#### Worked example: TD fixed point

Two states, one feature: $x(s_1) = 1$, $x(s_2) = 2$. From either state the
chain moves to each state with probability 0.5. Rewards are
$r_\pi = (0, 1)$, $\gamma = 0.9$, and $\mu = (0.5, 0.5)$, which is
stationary.

- The true values are $V^\pi = (4.5, 5.5)$. The average $m$ satisfies
  $m = 0.5 + 0.9m$, so $m = 5$.
- $(I - \gamma P)X = (0.55 - 0.9,\ -0.45 + 1.1) = (-0.35, 0.65)$, so
  $A = 0.5(-0.35 + 2 \cdot 0.65) = 0.475$ and $b = 0.5(0 + 2) = 1$.
- $w_{\text{TD}} = 1/0.475 = 2.1053$.
- The VE-optimal weight is $w_{\text{VE}} = (0.5 \cdot 4.5 + 0.5 \cdot 11)/(0.5 \cdot 5) = 3.1$.
- $\overline{\mathrm{VE}}(w_{\text{TD}}) = 3.699$ and
  $\min \overline{\mathrm{VE}} = 1.225$. The bound is
  $1.225/(1-0.81) = 6.447$, which holds.

TD's answer is far from the best linear fit, but within the bound.

#### Worked example: Baird divergence

The eigenvalues of $A_{\text{B}}$, computed in numpy, are approximately
$-0.2393$, $-0.0222$, $0$, and $0.5714$ with multiplicity five. The zero comes
from having eight weights for seven states. Two eigenvalues are negative. Each
negative eigenvalue $\lambda$ gives a factor $|1 - \alpha\lambda| > 1$ for
every $\alpha > 0$, so the expected update diverges at any step size. From
S&B's initial weights with $\alpha = 0.01$, the weight norm grows from 10.3 to
437 after 1,000 expected updates.

Under the target policy's on-policy distribution, all mass sits on $s_7$. Then
$A = (1-\gamma)\, x(s_7) x(s_7)^\top$ is positive semidefinite, and the same
update converges. The off-policy distribution is what breaks it.

#### References

- S&B, Chapter 9 (Section 9.4 for the TD fixed point) and Chapter 11
  (Section 11.2 for Baird, Section 11.3 for the deadly triad, Sections 11.7 and
  11.8 for gradient-TD and emphatic-TD fixes).
- Tsitsiklis and Van Roy, "An analysis of temporal-difference learning with
  function approximation," *IEEE Transactions on Automatic Control* 42(5),
  674-690, 1997,
  [author PDF of the NeurIPS version](http://papers.neurips.cc/paper/1269-analysis-of-temporal-diffference-learning-with-function-approximation.pdf).
- Baird, "Residual algorithms: Reinforcement learning with function
  approximation," ICML 1995, 30-37.
- Bradtke and Barto, "Linear least-squares algorithms for temporal difference
  learning," *Machine Learning* 22, 1996. The origin of LSTD (citation details
  UNVERIFIED in this session).
- van Hasselt, Doron, Strub, Hessel, Sonnerat, and Modayil, "Deep
  reinforcement learning and the deadly triad," 2018,
  [arXiv:1812.02648](https://arxiv.org/abs/1812.02648). An empirical study of
  when the triad bites in deep RL, which leads into lesson 12.
- Zhang, reference code,
  [`chapter11/counterexample.py`](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter11/counterexample.py).
  It confirms the features, $\gamma$, the policies, and the initial weights.

### 8.3 Exercise idea for function approximation

The following checks fit lesson 11:

- `tile_features(x, n_tilings, width, offsets)` returns exactly `n_tilings`
  active indices, matches the worked example, and gives exact one-trial
  learning at $\alpha = 1/m$.
- `td_fixed_point(X, P_pi, r_pi, mu, gamma)` returns $A^{-1}b$. Check it
  equals 2.1053 on the worked example.
- Semi-gradient TD(0), run on 20,000 sampled transitions from the worked
  chain with constant $\alpha = 0.005$, has an average weight over the last
  10,000 steps within 0.05 of $w_{\text{TD}} = 2.1053$. Five seeds gave 2.088
  to 2.096, in under 0.1 seconds each. It is far from $w_{\text{VE}} = 3.1$,
  which the check can also assert. A decaying schedule such as
  $\alpha_k = 0.5/(k+10)$ decays too fast here: it stalls near 1.78 after
  20,000 steps.
- The bound $\overline{\mathrm{VE}}(w_{\text{TD}}) \le \min\overline{\mathrm{VE}}/(1-\gamma^2)$
  holds on 20 seeded random linear problems from `rlenvs.random_mdp` with
  random features. It is a theorem, so it never fails.
- LSTD from 5,000 sampled transitions converges to $w_{\text{TD}}$ within a
  tolerance.
- `baird_expected_matrix()` matches the eigenvalues above to `1e-4`, and has
  an eigenvalue with negative real part.
- `semi_gradient_dp_baird(w0, alpha, n_steps)` has a weight norm above 100
  after 1,000 steps. The same routine with the on-policy weighting on $s_7$
  converges toward $w_7 + 2w_8 = 0$.

### 8.4 Common mistakes

- Backpropagating through the TD target. That gives the residual-gradient
  algorithm, which is a different method with a different fixed point.
- Using a fixed-size step with tile coding while changing the number of
  tilings, so the effective step size changes.
- Believing semi-gradient TD minimises $\overline{\mathrm{VE}}$. It reaches the
  TD fixed point instead.
- Thinking Baird's divergence comes from sampling noise. The expected update
  diverges too. The cause is the mismatch between the update's state
  distribution and the target policy.
- Drawing the wrong lesson from the triad. Any two of the three are fine.
  Target networks and replay in DQN mitigate the triad; they do not remove it.

## 9. Planning and learning

### 9.1 Dyna-Q

#### Intuition

Every real step teaches you two things: a value update and a fact about the
world. Dyna-Q stores those facts in a model and replays them as imagined
experience. Each imagined step is a Q-learning update, so the agent squeezes
many updates out of each real interaction.

#### Key equations

Dyna-Q (S&B Section 8.2, tabular Dyna-Q box) repeats the following loop:

1. Observe $s$ and pick $a$ $\varepsilon$-greedily from $Q$.
2. Take $a$, observe $r$ and $s'$.
3. Apply the Q-learning update to $Q(s, a)$ with target
   $r + \gamma \max_{a'} Q(s', a')$.
4. Store the model entry $\mathrm{Model}(s, a) \leftarrow (r, s')$. This
   assumes a deterministic environment.
5. Repeat $n$ times: pick a previously observed $(\tilde s, \tilde a)$ at
   random, read $(\tilde r, \tilde s')$ from the model, and apply the same
   Q-learning update.

With $n = 0$, Dyna-Q is Q-learning. Dyna-Q+ adds a bonus
$\kappa\sqrt{\tau}$ to modelled rewards, where $\tau$ is the time since the
pair was last tried, so it re-explores after the environment changes (S&B
Section 8.3).

#### Worked example

A chain $s_0 \to s_1 \to s_2$ has one action "right", with reward 1 on reaching
the terminal $s_2$. Use $\gamma = 0.9$, $\alpha = 1$, and $Q = 0$ at the start.

- The first real episode updates $Q(s_0) = 0 + 0.9 \cdot 0 = 0$, then
  $Q(s_1) = 1$.
- One planning update on $(s_0, \text{right})$ gives $Q(s_0) = 0.9 \cdot 1 = 0.9$
  with no new real experience.
- Without planning, $Q(s_0)$ reaches 0.9 only during the second real episode.

### 9.2 Prioritized sweeping

#### Intuition

Random replay wastes updates on pairs whose values would not change. Prioritized
sweeping keeps a priority queue ordered by how much a pair's value would change.
After updating a pair, it pushes that pair's predecessors onto the queue.
Changes propagate backward from where the news is.

#### Key equations

For a deterministic model, the priority of $(s, a)$ with model output
$(r, s')$ is:

$$
p(s, a) = \Bigl| r + \gamma \max_{a'} Q(s', a') - Q(s, a) \Bigr| .
$$

A pair enters the queue if $p > \theta$ for a small threshold $\theta$. After
updating $(s, a)$, the algorithm recomputes the priority of each predecessor
$(\bar s, \bar a)$ that the model predicts leads to $s$, with that
predecessor's modelled reward $\bar r$:

$$
p(\bar s, \bar a) = \Bigl| \bar r + \gamma \max_{a'} Q(s, a') - Q(\bar s, \bar a) \Bigr| .
$$

In the chain example, after $Q(s_1)$ changes to 1, the predecessor
$(s_0, \text{right})$ gets priority $|0 + 0.9 \cdot 1 - 0| = 0.9$.

### 9.3 Monte Carlo tree search and UCT

#### Intuition

At decision time, grow a search tree from the current state. For each
simulation, walk down the tree by treating each node's actions as a bandit and
using UCB to pick. Add a node at the frontier, estimate its value with a
rollout, and back the result up the path. Visit counts concentrate on the most
promising moves.

#### Key equations

The four phases of each simulation are selection, expansion, simulation
(rollout), and backup. UCT (Kocsis and Szepesvári, 2006) selects with a UCB
rule at each tree node:

$$
a = \operatorname*{arg\,max}_{a}\left[ Q(s,a) + c\sqrt{\frac{\ln N(s)}{N(s,a)}} \right].
$$

The backup is an incremental mean:
$N(s,a) \leftarrow N(s,a) + 1$ and
$Q(s,a) \leftarrow Q(s,a) + \bigl(G - Q(s,a)\bigr)/N(s,a)$. With $c = \sqrt 2$
and returns in $[0, 1]$, the rule is UCB1 at each node. Kocsis and Szepesvári
show that the probability of choosing a suboptimal root action goes to zero as
the number of simulations grows. Practical systems tune $c$. AlphaGo and
AlphaZero replace the bonus with a prior-weighted PUCT variant, covered in
lesson 25.

#### Worked example

A root has $N = 10$. Action $a$ has $N = 6$ and $Q = 0.5$. Action $b$ has
$N = 4$ and $Q = 0.4$. With $c = \sqrt 2$ and $\ln 10 = 2.302585$:

- $a$: $0.5 + 1.41421\sqrt{0.383764} = 0.5 + 0.87609 = 1.3761$.
- $b$: $0.4 + 1.41421\sqrt{0.575646} = 0.4 + 1.07298 = 1.4730$.

UCT selects $b$.

### 9.4 References for planning

- Sutton, "Integrated architectures for learning, planning, and reacting
  based on approximating dynamic programming," ICML 1990, 216-224. The origin
  of Dyna.
- Moore and Atkeson, "Prioritized sweeping: Reinforcement learning with less
  data and less time," *Machine Learning* 13, 103-130, 1993,
  [ACM entry](https://dl.acm.org/doi/abs/10.1023/A:1022635613229).
- Kocsis and Szepesvári, "Bandit based Monte-Carlo planning," ECML 2006, LNCS
  4212, 282-293,
  [doi:10.1007/11871842_29](https://doi.org/10.1007/11871842_29).
- Browne et al., "A survey of Monte Carlo tree search methods," *IEEE
  Transactions on Computational Intelligence and AI in Games* 4(1), 1-43,
  2012, [doi:10.1109/TCIAIG.2012.2186810](https://doi.org/10.1109/TCIAIG.2012.2186810).
- S&B, Chapter 8. Sections 8.2 (Dyna), 8.4 (prioritized sweeping), and 8.11
  (MCTS).

### 9.5 Exercise idea for planning

The following checks fit lesson 10:

- `dyna_q_update` reproduces the chain example: one planning step sets
  $Q(s_0) = 0.9$.
- Dyna-Q on a deterministic `rlenvs.gridworld` with $n = 50$ planning steps
  reaches a greedy policy with the optimal value (from
  `rlenvs.value_iteration`) at the start in fewer real episodes than $n = 0$.
  Use a fixed seed and a threshold with margin, and confirm under five seeds.
- With enough planning on a fully explored deterministic model, $Q$ converges
  to $Q^*$ from `rlenvs.q_from_v(mdp, value_iteration(mdp))` to `1e-6`.
- Prioritized sweeping reaches $\lVert Q - Q^* \rVert_\infty < 10^{-3}$ with
  fewer backups than uniformly random Dyna planning on the same learned model.
  Counting backups is deterministic given the seed, so the check is exact.
- `uct_scores` matches the worked example.
- UCT on a small depth-3 tree with Bernoulli leaves, whose exact expectimax
  values the starter computes, picks the correct root action after 3,000
  simulations. The best root action gets more than half the visits.

### 9.6 Common mistakes for planning

- Planning from pairs never actually visited, which reads uninitialised model
  entries.
- Storing only the last outcome in a stochastic environment and treating it as
  the truth. Tabular Dyna-Q assumes determinism.
- In MCTS, backing up returns from the wrong player's perspective in
  two-player games. Negate at each ply, or store values per player.
- Choosing the final move by the highest $Q$ with few visits instead of the
  most visited child. Most practical systems pick the most visited child.
- In prioritized sweeping, forgetting to update the priority of a pair that is
  already in the queue. Keep only the higher priority.

## 10. Partially observable MDPs (brief)

### Intuition

In a POMDP the agent sees an observation, not the state. The right thing to
condition on is the belief: a probability distribution over states given
everything seen so far. The belief is itself a Markov state, so a POMDP is an
MDP over beliefs. That MDP has a continuous state space, which is why exact
solutions are hard.

### Key equations

With observation model $O(o \mid s', a)$, the belief after taking $a$ and
seeing $o$ is:

$$
b'(s') = \frac{O(o \mid s', a) \sum_s P(s' \mid s, a)\, b(s)}{\sum_{s''} O(o \mid s'', a) \sum_s P(s'' \mid s, a)\, b(s)} .
$$

The denominator is $\Pr(o \mid b, a)$. In numpy this is
`bp = O[a][:, o] * (b @ P[:, a, :]); bp /= bp.sum()`. The finite-horizon
optimal value function over beliefs is piecewise linear and convex (Kaelbling,
Littman, and Cassandra, 1998).

### Worked example

In the tiger problem (Cassandra, Kaelbling, and Littman, 1994), a tiger is
behind the left or right door with equal prior probability. Listening costs 1
and reports the correct side with probability 0.85. Opening the tiger's door
costs 100 and opening the other door pays 10. Listening does not move the
tiger. Starting from $b(\text{left}) = 0.5$ and hearing "left" repeatedly:

- After one: $0.85 \cdot 0.5 / (0.85 \cdot 0.5 + 0.15 \cdot 0.5) = 0.85$.
- After two: $0.7225/(0.7225 + 0.0225) = 0.9698$.
- After three: $0.9945$.

Hearing "left" then "right" returns the belief to exactly 0.5.

### References

- Kaelbling, Littman, and Cassandra, "Planning and acting in partially
  observable stochastic domains," *Artificial Intelligence* 101(1-2), 99-134,
  1998, [doi:10.1016/S0004-3702(98)00023-X](https://doi.org/10.1016/S0004-3702(98)00023-X).
- S&B, Section 17.3, "Observations and state".

### Exercise idea

- `belief_update(b, a, o, P, O)` reproduces the tiger sequence, returns a
  vector that sums to one, and returns to 0.5 after "left" then "right".
- A frame-stacking or last-observation Q-learner on a small aliased gridworld
  scores lower than the same learner given the true state. This shows why
  memory matters, and it motivates the recurrent policies of later parts.

### Common mistakes

- Normalising before multiplying by the observation likelihood.
- Treating the observation as the state, then blaming the learning rule when
  aliased states get the same value.

## 11. The average-reward setting (brief)

### Intuition

In a continuing task with no end, discounting is a modelling choice, not part
of the problem. The average-reward setting ranks policies by long-run reward
per step. Values become "differential": how much better than average it is to
start from a state.

### Key equations

For an ergodic MDP, the average reward of $\pi$ is:

$$
\bar r(\pi) = \lim_{h\to\infty} \frac{1}{h}\sum_{t=0}^{h-1} \mathbb{E}_\pi[r_t]
= \sum_s \mu_\pi(s) \sum_a \pi(a \mid s)\, R(s,a),
$$

where $\mu_\pi$ is the stationary distribution, with $\mu_\pi^\top P_\pi = \mu_\pi^\top$.
The differential value satisfies a Bellman equation with no discount. The
solution is unique up to an additive constant:

$$
V^\pi(s) = \sum_a \pi(a \mid s)\Bigl[R(s,a) - \bar r(\pi) + \sum_{s'} P(s' \mid s,a)\, V^\pi(s')\Bigr].
$$

The differential TD error, in course notation, is
$\delta_t = r_t - \bar R_t + \hat Q(s_{t+1}, a_{t+1}) - \hat Q(s_t, a_t)$.
Differential semi-gradient SARSA (S&B Section 10.3) updates
$w \leftarrow w + \alpha\delta_t \nabla_w \hat Q(s_t,a_t)$. The average-reward
estimate is updated by $\bar R \leftarrow \bar R + \beta\,\delta_t$. That
$\bar R$ update is recalled from the S&B algorithm box and is UNVERIFIED
against the book text in this session. The book discusses it in Exercise 10.8,
which shows it has lower variance than $\bar R \leftarrow \bar R + \beta(r_t - \bar R)$.

As $\gamma \to 1$, the discounted value behaves like
$V_\gamma(s) \approx \bar r/(1-\gamma) + h(s) + \text{const}$, where $h$ is the
differential value.

### Worked example

Take a two-state chain with $P_\pi = \begin{bmatrix}0.5&0.5\\0.2&0.8\end{bmatrix}$
and rewards $(1, 3)$.

- The stationary distribution is $\mu = (2/7, 5/7)$, so
  $\bar r = 2/7 + 15/7 = 17/7 \approx 2.4286$.
- With the normalisation $h(s_1) = 0$, the first Bellman row gives
  $0 = 1 - 17/7 + 0.5 h(s_2)$, so $h(s_2) = 20/7 \approx 2.857$.
- Discounted values at $\gamma = 0.999$ give
  $V_\gamma(s_2) - V_\gamma(s_1) = 2.856$, which approaches $20/7$.

### References

- S&B, Sections 10.3 to 10.5.
- Schwartz, "A reinforcement learning method for maximizing undiscounted
  rewards," ICML 1993. R-learning (citation details UNVERIFIED in this
  session).
- Mahadevan, "Average reward reinforcement learning: Foundations, algorithms,
  and empirical results," *Machine Learning* 22, 1996 (citation details
  UNVERIFIED in this session).
- Wan, Naik, and Sutton, "Learning and planning in average-reward Markov
  decision processes," ICML 2021, arXiv:2006.16318 (arXiv ID UNVERIFIED in
  this session).

### Exercise idea

- `stationary_distribution(P_pi)` returns $(2/7, 5/7)$ and satisfies
  $\mu^\top P = \mu^\top$.
- `average_reward(mdp, pi)` matches a long seeded simulation's mean reward
  within four standard errors.
- `differential_values(P_pi, r_pi)` satisfies the differential Bellman
  equation to `1e-10` with $h(s_1) = 0$.
- $(1-\gamma)V_\gamma \to \bar r$ as $\gamma \to 1$: check at
  $\gamma = 0.999$ within 0.01.

### Common mistakes

- Expecting a unique differential value. Fix a reference state or require
  $\mu^\top h = 0$.
- Using the average-reward formulation on a periodic or multichain MDP without
  checking that the limit exists.

## 12. Notes for exercise authors

These points apply across lessons 01 to 11:

- Match the contract's budget. The contract asks for a reference solution that
  passes all checks in under 20 seconds, and a `timeout` of about four times
  that. Every check proposed here is either exact linear algebra, which takes
  milliseconds, or a seeded simulation measured at two seconds or less.
- Prefer exact identities over learning curves: offline TD($\lambda$)
  equivalence, GAE against $G^\lambda - V$, the TD fixed point, value
  iteration's $\gamma^k$ bound, and Baird's eigenvalues. They cannot flake.
- When a check must train, check a property with a margin: a regret ratio, a
  fraction of "left" actions, or a greedy policy's exact value from
  `rlenvs.policy_evaluation`. Do not compare against a specific learned
  number.
- Use `rlenvs` worlds. `cliff_walking()` has an optimal start value of exactly
  $-13$, and `random_mdp` gives varied exact test cases.
- Guard exact evaluation when $\gamma = 1$. A deterministic greedy policy that
  loops anywhere, even in an unreachable state, makes
  `rlenvs.policy_evaluation` raise `LinAlgError`. Use a capped rollout or a
  discount below one for those checks.
- Break argmax ties at random with the seeded generator in learning code, but
  use `rlenvs.greedy`, which picks the first maximiser, when the check
  evaluates a policy.
- Keep the terminal mask explicit in every function signature, such as a
  `terminated` argument, so the most common bug is visible in the interface.

## 13. Verification log

The following items were verified in this session:

- Bibliographic details, DOIs, arXiv IDs, and page ranges were checked with web
  search for: Auer et al. 2002, Lai and Robbins 1985, Thompson 1933, Agrawal
  and Goyal 2012, Kaufmann et al. 2012, Russo et al. 2018, Li et al. 2010,
  Abbasi-Yadkori et al. 2011, Watkins and Dayan 1992, Sutton 1988, Sutton
  1990, van Hasselt 2010, van Seijen et al. 2009, Singh and Sutton 1996,
  Precup et al. 2000, Mahmood et al. 2014, van Seijen and Sutton 2014,
  Schulman et al. 2015, Tsitsiklis and Van Roy 1997, Baird 1995, Moore and
  Atkeson 1993, Kocsis and Szepesvári 2006, Browne et al. 2012, Kaelbling et
  al. 1998, Singh and Yee 1994, and van Hasselt et al. 2018.
- S&B section titles for Chapters 2, 8, 9, 10, 11, and 12 were checked
  against secondary chapter notes. Chapter 6 and 7 titles were checked against
  search snippets.
- The S&B 2015 draft PDF (Stanford course mirror) was read directly for the
  UCB rule, the gradient-bandit update, Example 5.5 (infinite variance),
  Example 6.4 (batch TD against MC), tile-coding step sizes, and prioritized
  sweeping. Equation numbers in the draft differ from the 2018 edition. The
  equation numbers cited in these notes are the 2018 edition's, recalled from
  memory, and are UNVERIFIED against the 2018 text.
- Baird's features, $\gamma = 0.99$, policies, initial weights
  $(1,1,1,1,1,1,10,1)$, and $\alpha = 0.01$ were checked against Zhang's
  reference implementation. The maximisation-bias settings ($\varepsilon = 0.1$,
  $\alpha = 0.1$, $\gamma = 1$, 10 actions in B) were checked against the same
  source.
- Every worked number was computed in numpy: the bandit tables, the Beta
  integral, LinUCB, the two-state MDP values and iteration tables, importance
  sampling, $\lambda$-returns and GAE, the TD fixed point and its bound, Baird's
  eigenvalues and divergence, the tiger beliefs, and the average-reward
  example. The offline TD($\lambda$) identity was checked to machine
  precision.

The following items remain UNVERIFIED:

- The 2018 edition's definition of $\bar r_t$ in the gradient bandit as the
  average excluding the current reward. The 2015 draft includes it.
- The true online TD($\lambda$) weight update, recalled from S&B
  equation 12.11.
- The differential SARSA update $\bar R \leftarrow \bar R + \beta\delta$,
  recalled from the S&B algorithm box.
- Citation details for Singh, Jaakkola, Littman, and Szepesvári 2000 (SARSA
  convergence), Albus 1975, Bradtke and Barto 1996, Schwartz 1993, Mahadevan
  1996, Wan et al. 2021, and the Williams 1992 DOI.
- The URLs for the Lattimore and Szepesvári draft and the Abbasi-Yadkori
  NeurIPS abstract page.
- The name of the looping action in Example 5.5 in the 2018 edition. The 2015
  draft calls the actions "back" and "end".
- The regret and maximisation-bias tables come from one set of seeds each.
  Anyone who turns them into check thresholds must re-tune them under at least
  five seeds, as the contract requires.
