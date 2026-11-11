---
title: Exploration beyond epsilon
slug: 046-exploration-bonuses
part: "Part 7 — Models, exploration, and fixed data"
summary: Reach a far-away reward by making uncertainty attractive, with count bonuses, posterior sampling, and a random-network novelty bonus.
minutes: 40
gpu: false
objectives:
  - Explain why epsilon-greedy needs a run of lucky steps whose chance shrinks geometrically with the distance to the reward.
  - Add a count bonus beta / sqrt(N) to a learned model's reward and plan with it.
  - Sample a model from a Dirichlet and Gaussian posterior and act optimally for it.
  - Compute a random network distillation bonus and say why it stays high on states the agent has not visited.
prereqs:
  - 002-returns-and-discounting
  - 003-explore-or-exploit
  - 004-optimism-and-ucb
  - 005-thompson-sampling
  - 043-model-based-rl
exercise: 046-exploration-bonuses
---

# Exploration beyond epsilon

A swimmer in a river has six positions to choose from. Drifting left with the current always works and pays a tiny reward at the bank. Swimming right against the current usually fails, and it pays nothing until the far end, where the reward is 200 times larger. How does an agent that has only ever seen the tiny reward find out that the far end exists?

In a bandit, every arm is one pull away. In a sequential problem, a good reward can sit many steps away, behind actions that each look pointless alone. This chapter shows how badly random dithering fails there, and fixes it three ways: a bonus for rarely tried actions, a plan made in a randomly drawn plausible world, and a novelty signal for problems with too many states to count.

## What you already know

- Epsilon-greedy takes a random action with probability $\varepsilon$ and the best-looking one otherwise (see [Explore or exploit](/lessons/003-explore-or-exploit)).
- UCB adds a bonus that shrinks as an arm's count grows, and Thompson sampling acts on a sample from the posterior.
- Value iteration computes optimal action values from a known model.
- A learned tabular model estimates transition probabilities from counts.

## Why dithering fails

Figure 1 draws the river that `rlenvs.river_swim` builds. States run from 0 to 5. Swimming left always works, and in state 0 it pays 0.005. Swimming right moves one state right with probability 0.35, stays put with 0.6, and slips back with 0.05. Swimming right in state 5 pays 1 on every step. With $\gamma = 0.95$, always swimming right is worth 8.739 from the start, and always swimming left is worth 0.0975.

```figure
<svg viewBox="0 0 600 230" role="img" aria-label="RiverSwim: six states in a row; swimming left always works, swimming right moves forward with probability 0.35">
<defs><marker id="c041-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<circle cx="70.0" cy="110.0" r="24" class="fill-soft-a"/>
<circle cx="70.0" cy="110.0" r="24" class="fill-soft-a"/>
<text x="70.0" y="115.0" class="fig-bold" text-anchor="middle">0</text>
<circle cx="162.0" cy="110.0" r="24" class="fill-soft-a"/>
<circle cx="162.0" cy="110.0" r="24" class="fill-soft-a"/>
<text x="162.0" y="115.0" class="fig-bold" text-anchor="middle">1</text>
<circle cx="254.0" cy="110.0" r="24" class="fill-soft-a"/>
<circle cx="254.0" cy="110.0" r="24" class="fill-soft-a"/>
<text x="254.0" y="115.0" class="fig-bold" text-anchor="middle">2</text>
<circle cx="346.0" cy="110.0" r="24" class="fill-soft-a"/>
<circle cx="346.0" cy="110.0" r="24" class="fill-soft-a"/>
<text x="346.0" y="115.0" class="fig-bold" text-anchor="middle">3</text>
<circle cx="438.0" cy="110.0" r="24" class="fill-soft-a"/>
<circle cx="438.0" cy="110.0" r="24" class="fill-soft-a"/>
<text x="438.0" y="115.0" class="fig-bold" text-anchor="middle">4</text>
<circle cx="530.0" cy="110.0" r="24" class="fill-soft-a"/>
<circle cx="530.0" cy="110.0" r="24" class="fill-soft-a"/>
<text x="530.0" y="115.0" class="fig-bold" text-anchor="middle">5</text>
<line x1="96.0" y1="102.0" x2="136.0" y2="102.0" class="stroke-b" marker-end="url(#c041-arrow)"/>
<text x="116.0" y="96.0" class="fig-small" text-anchor="middle">0.35</text>
<line x1="136.0" y1="120.0" x2="96.0" y2="120.0" class="stroke-a" marker-end="url(#c041-arrow)"/>
<line x1="188.0" y1="102.0" x2="228.0" y2="102.0" class="stroke-b" marker-end="url(#c041-arrow)"/>
<text x="208.0" y="96.0" class="fig-small" text-anchor="middle">0.35</text>
<line x1="228.0" y1="120.0" x2="188.0" y2="120.0" class="stroke-a" marker-end="url(#c041-arrow)"/>
<line x1="280.0" y1="102.0" x2="320.0" y2="102.0" class="stroke-b" marker-end="url(#c041-arrow)"/>
<text x="300.0" y="96.0" class="fig-small" text-anchor="middle">0.35</text>
<line x1="320.0" y1="120.0" x2="280.0" y2="120.0" class="stroke-a" marker-end="url(#c041-arrow)"/>
<line x1="372.0" y1="102.0" x2="412.0" y2="102.0" class="stroke-b" marker-end="url(#c041-arrow)"/>
<text x="392.0" y="96.0" class="fig-small" text-anchor="middle">0.35</text>
<line x1="412.0" y1="120.0" x2="372.0" y2="120.0" class="stroke-a" marker-end="url(#c041-arrow)"/>
<line x1="464.0" y1="102.0" x2="504.0" y2="102.0" class="stroke-b" marker-end="url(#c041-arrow)"/>
<text x="484.0" y="96.0" class="fig-small" text-anchor="middle">0.35</text>
<line x1="504.0" y1="120.0" x2="464.0" y2="120.0" class="stroke-a" marker-end="url(#c041-arrow)"/>
<text x="346.0" y="158.0" class="fig-small" text-anchor="middle">swim left: always moves left (blue)</text>
<text x="346.0" y="176.0" class="fig-small" text-anchor="middle">swim right: 0.35 forward, 0.6 stay, 0.05 back (orange)</text>
<text x="70.0" y="62.0" class="fig-small" text-anchor="middle">left pays 0.005</text>
<text x="530.0" y="62.0" class="fig-small" text-anchor="middle">right pays 1 per step</text>
<text x="70.0" y="200.0" class="fig-small" text-anchor="middle">near bank</text>
<text x="530.0" y="200.0" class="fig-small" text-anchor="middle">far end</text>
</svg>
Figure 1. The river. The small reward sits at the left bank, and the 200-times-larger one sits five successful swims away. Every swim right against the current is a gamble, so one lucky swim is not enough.
```

An agent that has only seen the 0.005 believes left is best. Epsilon-greedy swims right only when its coin flip says to, which happens with probability $\varepsilon / 2$, because half of the random actions are rightward. Getting from state 1 to state 5 needs four successful swims before the current pushes the agent back. A crude estimate of the chance of four in a row is $(0.35\,\varepsilon / 2)^4$:

| $\varepsilon$ | 0.1 | 0.3 | 0.5 | 1.0 |
|---|---|---|---|---|
| $(0.35\,\varepsilon / 2)^4$ | $9.4 \times 10^{-8}$ | $7.6 \times 10^{-6}$ | $5.9 \times 10^{-5}$ | $9.4 \times 10^{-4}$ |

A longer river makes dithering hopeless. Over three seeds and 2,000 steps, epsilon-greedy with $\varepsilon = 0.1$ never visits state 5 once.

> [!KEY]
> Random dithering explores locally. Reaching a reward $k$ deliberate steps away needs a run of lucky choices whose probability falls geometrically in $k$. Exploration has to commit to a plan for many steps.

## Optimism: pay for the unknown

The fix from the bandit chapters carries over: pretend that anything you have not tried much is better than it looks, so the agent tries it on purpose. Count how often each state-action pair has been tried, $N(s, a)$, and build an empirical model $\hat P$ and mean reward $\hat R$ from the counts. Then plan with value iteration on a reward that carries a bonus. This is MBIE-EB (Strehl and Littman, 2008):

$$
\tilde R(s, a) = \hat R(s, a) + \frac{\beta}{\sqrt{N(s, a)}}
$$

Read it as: the working reward is the average reward so far, plus a bonus that is large for pairs tried a few times and shrinks like one over the square root of the count. The square root matches how fast the uncertainty in a mean shrinks. The scale $\beta$ sets how optimistic the agent is. The exercise treats an untried pair as $N = 1$, so its bonus is $\beta$:

```python
bonus = beta / np.sqrt(np.maximum(N, 1))
Q = plan(P_hat, R_hat + bonus, gamma)      # value iteration on the bonus reward
```

### Work it by hand

Take $\beta = 0.5$. A pair tried once gets a bonus of 0.5, a pair tried 4 times gets 0.25, and a pair tried 100 times gets 0.05. An untried pair stays put in the empirical model, so it earns its 0.5 on every step, which is worth at least $0.5 / (1 - 0.95) = 10$. That is far more than the 0.1 the agent has found by swimming left. So the planner sends the agent to try every untried pair, including swimming right in states it has barely reached. Each visit shrinks that pair's bonus. Only the far end, where the real reward is 1 per step, stays attractive after the bonuses fade.

Replanning every 20 steps, this agent ends with the optimal policy on every one of three seeds. In Figure 2 it reaches state 5 within 200 steps, and by step 2,000 it has spent 1,494 steps there.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Cumulative steps spent in state 5 over 2000 steps: epsilon-greedy stays at zero, optimism reaches 1494 and posterior sampling 1325">
<defs><marker id="c041-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<line x1="70.0" y1="270.0" x2="540.0" y2="270.0" class="stroke-grid"/>
<text x="62.0" y="274.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="198.1" x2="540.0" y2="198.1" class="stroke-grid"/>
<text x="62.0" y="202.1" class="fig-small" text-anchor="end">500</text>
<line x1="70.0" y1="126.2" x2="540.0" y2="126.2" class="stroke-grid"/>
<text x="62.0" y="130.2" class="fig-small" text-anchor="end">1000</text>
<line x1="70.0" y1="54.4" x2="540.0" y2="54.4" class="stroke-grid"/>
<text x="62.0" y="58.4" class="fig-small" text-anchor="end">1500</text>
<line x1="70.0" y1="270.0" x2="540.0" y2="270.0" class="stroke-muted"/>
<line x1="70.0" y1="270.0" x2="70.0" y2="275.0" class="stroke-muted"/>
<text x="70.0" y="290.0" class="fig-small" text-anchor="middle">0</text>
<line x1="187.5" y1="270.0" x2="187.5" y2="275.0" class="stroke-muted"/>
<text x="187.5" y="290.0" class="fig-small" text-anchor="middle">500</text>
<line x1="305.0" y1="270.0" x2="305.0" y2="275.0" class="stroke-muted"/>
<text x="305.0" y="290.0" class="fig-small" text-anchor="middle">1000</text>
<line x1="422.5" y1="270.0" x2="422.5" y2="275.0" class="stroke-muted"/>
<text x="422.5" y="290.0" class="fig-small" text-anchor="middle">1500</text>
<line x1="540.0" y1="270.0" x2="540.0" y2="275.0" class="stroke-muted"/>
<text x="540.0" y="290.0" class="fig-small" text-anchor="middle">2000</text>
<text x="305.0" y="312.0" class="fig-small" text-anchor="middle">environment steps</text>
<polyline points="70.0,270.0 93.5,270.0 117.0,270.0 140.5,270.0 164.0,270.0 187.5,270.0 211.0,270.0 234.5,270.0 258.0,270.0 281.5,270.0 305.0,270.0 328.5,270.0 352.0,270.0 375.5,270.0 399.0,270.0 422.5,270.0 446.0,270.0 469.5,270.0 493.0,270.0 516.5,270.0 540.0,270.0" class="stroke-a" fill="none"/>
<polyline points="70.0,270.0 93.5,270.0 117.0,258.8 140.5,246.7 164.0,233.6 187.5,221.1 211.0,208.6 234.5,198.6 258.0,185.6 281.5,176.4 305.0,164.2 328.5,153.7 352.0,141.2 375.5,131.9 399.0,121.1 422.5,109.7 446.0,96.8 469.5,87.4 493.0,74.5 516.5,66.4 540.0,55.2" class="stroke-b" fill="none"/>
<polyline points="70.0,270.0 93.5,270.0 117.0,270.0 140.5,265.7 164.0,252.6 187.5,242.8 211.0,230.3 234.5,220.3 258.0,207.3 281.5,200.7 305.0,188.5 328.5,178.0 352.0,165.5 375.5,156.2 399.0,145.4 422.5,134.0 446.0,121.1 469.5,111.7 493.0,98.8 516.5,90.7 540.0,79.5" class="stroke-c" fill="none"/>
<line x1="100.0" y1="50.0" x2="130.0" y2="50.0" class="stroke-a"/>
<text x="136.0" y="54.0" class="fig-small" text-anchor="start">epsilon-greedy: 0 at step 2000</text>
<line x1="100.0" y1="68.0" x2="130.0" y2="68.0" class="stroke-b"/>
<text x="136.0" y="72.0" class="fig-small" text-anchor="start">optimism: 1494 at step 2000</text>
<line x1="100.0" y1="86.0" x2="130.0" y2="86.0" class="stroke-c"/>
<text x="136.0" y="90.0" class="fig-small" text-anchor="start">posterior sampling: 1325 at step 2000</text>
</svg>
Figure 2. Cumulative steps spent in the far state 5, seed 0. Epsilon-greedy never leaves the left end. Optimism and posterior sampling find the far reward and then keep collecting it.
```

## Posterior sampling: act on one plausible world

Optimism inflates every uncertain value. Posterior sampling takes a different route. Keep a probability distribution over what the world might be, which is the *posterior*. Every so often, draw one complete model from it, plan in that model, and follow its optimal policy until the next draw. Osband, Russo, and Van Roy (2013) call this PSRL and prove it has low regret.

For transitions, the natural posterior is a *Dirichlet* distribution, a distribution over probability vectors. With a prior of one pseudo-observation on every next state, row $(s, a)$ after counts $C(s, a, \cdot)$ is

$$
P(\cdot \mid s, a) \sim \operatorname{Dirichlet}\big(C(s, a, \cdot) + 1\big)
$$

Read it as: each next state gets a weight of one plus the number of times you saw it, and sampled probability vectors cluster around those weights, normalized. More data makes the cluster tighter. To sample one, draw a gamma variable for each weight and divide by their sum:

```python
g = rng.standard_gamma(counts + prior)       # (S, A, S)
P = g / g.sum(axis=2, keepdims=True)
```

For rewards, the exercise uses a Gaussian prior with mean 0 and variance 1, and unit noise on each observed reward. After $N$ visits with reward total $\Sigma$, the posterior is Gaussian with mean $\Sigma / (N + 1)$ and standard deviation $1 / \sqrt{N + 1}$.

Say that in state 1, swimming right has been tried 8 times. It moved right 3 times, stayed 5 times, and never slipped. The Dirichlet weights over states 0 to 5 are $[1, 6, 4, 1, 1, 1]$, which sum to 14. The posterior mean chance of moving right is $4/14 = 0.286$, close to the true 0.35. The posterior still gives $1/14 = 0.071$ to jumping straight to state 5, which is impossible. Early on, that stray mass makes some sampled models believe the far end is close, so the agent swims right to find out. As counts grow, the stray mass shrinks. An untried pair has reward standard deviation 1, so some draws make it look rewarding. In Figure 2, the posterior-sampling agent starts slower than the optimist, because its first draws are random guesses, and then climbs at the same rate.

> [!KEY]
> Optimism explores by inflating every uncertain value. Posterior sampling explores by acting on one plausible world at a time. Both commit to a plan for many steps, which is what dithering cannot do.

## Too many states to count: random network distillation

A count needs states it can compare. In Atari, no two frames are exactly alike, so every count is zero. Bellemare and others (2016) turn a density model into *pseudo-counts*, but a simpler trick works for any observation. Random network distillation (RND; Burda and others, 2019) fixes a randomly initialized network $f$, the *target*, and never trains it. It trains a second network $\hat f_\phi$, the *predictor*, to match the target's output on every state the agent visits. The bonus for a state is the predictor's error:

$$
b(s) = \big\lVert \hat f_\phi(s) - f(s) \big\rVert^2
$$

Read it as: the target maps each state to a fixed random vector. On states the predictor has trained on, it has learned to reproduce that vector, so the bonus is small. On states it has never seen, it has not, so the bonus stays large. In PyTorch:

```python
with torch.no_grad():
    bonus = ((predictor(states) - target(states)) ** 2).sum(dim=1)
```

In the exercise, both networks take a two-dimensional state, with one hidden layer of 32 units and 8 outputs. The predictor trains for 300 Adam steps on 256 states from the square $[0, 0.5]^2$. The mean bonus there falls from 0.49 to 0.00006, while on 256 states from $[0.5, 1]^2$, which it never saw, the bonus is 0.011 (Figure 3). That is over 100 times higher, and it is what pulls the agent toward unvisited regions.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="RND bonus on a log scale: visited states fall from 0.49 to 0.00006 after training, while novel states stay at 0.011">
<defs><marker id="c041-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<line x1="80.0" y1="250.0" x2="540.0" y2="250.0" class="stroke-grid"/>
<text x="72.0" y="254.0" class="fig-small" text-anchor="end">1e-5</text>
<line x1="80.0" y1="208.0" x2="540.0" y2="208.0" class="stroke-grid"/>
<text x="72.0" y="212.0" class="fig-small" text-anchor="end">1e-4</text>
<line x1="80.0" y1="166.0" x2="540.0" y2="166.0" class="stroke-grid"/>
<text x="72.0" y="170.0" class="fig-small" text-anchor="end">1e-3</text>
<line x1="80.0" y1="124.0" x2="540.0" y2="124.0" class="stroke-grid"/>
<text x="72.0" y="128.0" class="fig-small" text-anchor="end">1e-2</text>
<line x1="80.0" y1="82.0" x2="540.0" y2="82.0" class="stroke-grid"/>
<text x="72.0" y="86.0" class="fig-small" text-anchor="end">1e-1</text>
<line x1="80.0" y1="40.0" x2="540.0" y2="40.0" class="stroke-grid"/>
<text x="72.0" y="44.0" class="fig-small" text-anchor="end">1</text>
<rect x="110.0" y="53.0" width="30.0" height="197.0" class="fill-a"/>
<rect x="160.0" y="43.3" width="30.0" height="206.7" class="fill-b"/>
<text x="125.0" y="47.0" class="fig-small" text-anchor="middle">0.49</text>
<text x="175.0" y="37.3" class="fig-small" text-anchor="middle">0.83</text>
<text x="150.0" y="270.0" class="fig-small" text-anchor="middle">before training</text>
<rect x="340.0" y="216.0" width="30.0" height="34.0" class="fill-a"/>
<rect x="390.0" y="122.1" width="30.0" height="127.9" class="fill-b"/>
<text x="355.0" y="210.0" class="fig-small" text-anchor="middle">0.00006</text>
<text x="405.0" y="116.1" class="fig-small" text-anchor="middle">0.011</text>
<text x="380.0" y="270.0" class="fig-small" text-anchor="middle">after 300 steps</text>
<rect x="380.0" y="48.0" width="12.0" height="12.0" class="fill-a"/>
<text x="398.0" y="59.0" class="fig-small" text-anchor="start">visited states</text>
<rect x="380.0" y="66.0" width="12.0" height="12.0" class="fill-b"/>
<text x="398.0" y="77.0" class="fig-small" text-anchor="start">novel states</text>
</svg>
Figure 3. The mean RND bonus before and after training on the lower-left square. Training drives the bonus on visited states down by a factor of about 7,600, while novel states keep a bonus 170 times larger.
```

> [!WARNING]
> A bonus for the prediction error of the *next state*, rather than of a fixed target, gets stuck on anything random. A television showing static is always hard to predict, so the agent stares at it. Burda and others (2019) call this the noisy-TV problem. RND removes the part that comes from random transitions, because its target depends only on the current state.

> [!RECAP]
> - The count bonus is $\beta / \sqrt{\max(N, 1)}$, added to the empirical reward before planning.
> - A posterior model samples each row from $\operatorname{Dirichlet}(C + 1)$ and each reward from a Gaussian with mean $\Sigma / (N + 1)$.
> - All three agents replan every 20 steps and report the greedy policy of the final empirical model, with no bonus.
> - The RND bonus is the squared distance between predictor and target.

## Build it step by step

The checks test the functions in this order.

1. **`count_bonus(counts, beta)`.** One line with `np.maximum` and `np.sqrt`. Do not divide by zero for unvisited pairs.
2. **`empirical_model(transition_counts, reward_sums)`.** Sum counts over the last axis to get `N`, and divide only where `N > 0`. Set `P[s, a, s] = 1` for each unvisited pair, because a zero row is not a probability vector. Return `(P, R, N)`.
3. **`plan(P, R, gamma)`.** Iterate `Q = R + gamma * P @ Q.max(axis=1)` from zeros until it stops moving.
4. **`sample_posterior_mdp(transition_counts, reward_sums, rng, prior)`.** Gamma draws for `P`, then a Gaussian draw for `R`. Use `N + 1` in the reward posterior, not `N`.
5. **`explore(mdp, method, steps, seed, ...)`.** One loop for all three methods: replan every `replan_every` steps, act, update the counts, and reset on termination or truncation. Return the greedy policy of the final empirical model without the bonus, because the checks want what the agent believes.
6. **`make_rnd`, `rnd_bonus`, and `rnd_train`.** Freeze the target with `requires_grad_(False)`, and give the optimiser only the predictor's parameters.

## Where this breaks

**Bonus scale and prior are guesses.** Too small a $\beta$ and optimism behaves like greedy. Too large and the agent wanders after it finds the reward. A prior that rules out the true world can stall posterior sampling. [Sample complexity and regret](/lessons/047-sample-complexity-and-regret) shows what a bonus with a proven scale costs.

**Novelty is not usefulness.** A bonus rewards every novel state, including ones that do not matter. RND works best with its bonus normalized by a running estimate of its spread.

**Goals can replace bonuses.** When many goals share one environment, relabeling failed episodes with the goals they reached gives the learner rewards without any bonus. [Goals and hindsight](/lessons/048-goal-conditioned-rl-and-her) builds this.

## Further reading

- [Strehl and Littman (2008), An analysis of model-based interval estimation for Markov decision processes](https://doi.org/10.1016/j.jcss.2007.08.009). MBIE-EB, the count bonus used here, tested on RiverSwim.
- [Osband, Russo, and Van Roy (2013), (More) efficient reinforcement learning via posterior sampling](https://arxiv.org/abs/1306.0940). PSRL and its regret bound.
- [Bellemare and others (2016), Unifying count-based exploration and intrinsic motivation](https://arxiv.org/abs/1606.01868). Pseudo-counts from density models.
- [Burda, Edwards, Storkey, and Klimov (2019), Exploration by random network distillation](https://arxiv.org/abs/1810.12894). RND and its normalization details.

## The exercise

You build three exploring agents that share one model-based loop, and a novelty bonus. Write `count_bonus`, `empirical_model`, `plan`, and `sample_posterior_mdp`, then `explore`, which runs the `"epsilon"`, `"optimism"`, or `"posterior"` agent on RiverSwim. Finish with `make_rnd`, `rnd_bonus`, and `rnd_train`.

The checks verify the bonus, the empirical model on hand-built counts, `plan` against $Q^*$, and the posterior's Dirichlet and Gaussian means over 4,000 samples. They run each agent for 2,000 steps on three seeds: epsilon-greedy must never reach state 5, while optimism and posterior sampling must end with the optimal policy and spend over 500 steps there. Finally, training must cut the RND bonus on visited states by 99% while novel states keep a bonus over 10 times higher.
