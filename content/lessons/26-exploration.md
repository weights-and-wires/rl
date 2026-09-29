---
title: Exploration
slug: 26-exploration
part: "Part 6 — Models, exploration, and fixed data"
summary: Find a reward that random dithering never reaches, with count bonuses, posterior sampling, and random network distillation.
minutes: 60
gpu: false
objectives:
  - Explain why epsilon-greedy needs a run of lucky steps whose chance shrinks geometrically with the distance to the reward.
  - Add a count-based bonus to a learned model and plan with it, and compute pseudo-counts from a density model.
  - Sample a model from a Dirichlet and Gaussian posterior and act optimally for it.
  - Build a random network distillation bonus that stays high on states the agent hasn't visited.
prereqs:
  - 02-multi-armed-bandits
  - 24-model-based-rl
exercise: 26-exploration
---

# Exploration

An agent learns only about the things it tries. In the bandit lesson, trying
an arm cost one pull, and every arm was always one pull away. In a sequential
problem, the good reward can sit many steps away, behind a series of actions
that each look pointless on their own. An agent that explores by taking a
random action now and then almost never strings together the whole series.

RiverSwim is the classic example. A swimmer in a river of six positions can
drift left with the current, which pays a tiny reward at the bank, or swim
right against it, which usually fails and pays nothing until the far end.
The far end pays 200 times more. Every method in this lesson faces the same
river, and the difference in what they find is stark.

You build three ways to explore on purpose: a bonus for rarely tried actions,
a posterior that samples plausible worlds, and a novelty signal from a
neural network, for problems with too many states to count.

## What you already know

- Epsilon-greedy takes a random action with probability $\varepsilon$ and
  the best-looking one otherwise.
- UCB adds a bonus that shrinks as an arm's count grows, and Thompson
  sampling acts on a sample from the posterior.
- Value iteration computes optimal action values from a known model.
- A learned tabular model estimates transition probabilities from counts.
- A *policy* is a rule for choosing actions, and its *value* from a start
  state is the expected discounted return.

## Why dithering fails on RiverSwim

The river in `rlenvs.river_swim` has states 0 to 5. Action 0 swims left and
always works. Swimming left in state 0 pays 0.005. Action 1 swims right, and
it moves right with probability 0.35, stays put with probability 0.6, and
slips left with probability 0.05. Swimming right in state 5 pays 1. With
$\gamma = 0.95$, and averaged over the two start states, 0 and 1, always
swimming right is worth 8.739 and always swimming left is worth 0.0975.

An agent that has only ever seen the small reward believes left is best.
Epsilon-greedy then swims right only when its coin flip says to, with
probability $\varepsilon / 2$, since a random action is right half the time.
To get from state 1 to state 5, it needs four successful swims to the right
before the current pushes it back. A crude estimate of the chance of four in
a row is $(0.35 \, \varepsilon / 2)^4$:

| $\varepsilon$ | 0.1 | 0.3 | 0.5 | 1.0 |
|---|---|---|---|---|
| $(0.35 \, \varepsilon / 2)^4$ | $9.4 \times 10^{-8}$ | $7.6 \times 10^{-6}$ | $5.9 \times 10^{-5}$ | $9.4 \times 10^{-4}$ |

The exercise measures it. Over ten seeds and 2,000 steps, epsilon-greedy with
$\varepsilon = 0.1$, 0.3, or 0.5 never once reaches state 5. Even a purely
random agent, $\varepsilon = 1$, spends only 0 to 10 of its steps there. The
chance of a lucky run shrinks geometrically with the length of the chain, so
a longer river makes dithering hopeless.

> [!KEY]
> Random dithering explores locally. Reaching a reward $k$ deliberate steps
> away needs a run of lucky choices whose probability falls geometrically in
> $k$.

## Optimism in the face of uncertainty

The fix from the bandit lesson carries over. Pretend that anything you
haven't tried much is better than it looks. The agent then tries it on
purpose, and either finds a reward or learns the pretence was wrong.

MBIE-EB (Strehl and Littman, 2008) puts this into a learned model. Count how
often each pair $(s, a)$ has been tried, $N(s, a)$. Build the empirical model
$\hat P$ and mean rewards $\hat R$ from the counts. Then plan with value
iteration on a reward that carries a bonus:

$$
\tilde R(s, a) = \hat R(s, a) + \frac{\beta}{\sqrt{N(s, a)}}
$$

Read it as: the agent's working reward is the average reward seen so far,
plus a bonus that is large for pairs tried a few times and shrinks like one
over the square root of the count. $\beta$ sets how optimistic the agent is.
The square root matches how fast the uncertainty in a mean shrinks. The
exercise treats an unvisited pair as $N = 1$, so its bonus is $\beta$. In
numpy:

```python
bonus = beta / np.sqrt(np.maximum(N, 1))
Q = plan(P_hat, R_hat + bonus, gamma)      # value iteration
```

### Worked example

With $\beta = 0.5$, a pair tried once gets a bonus of 0.5, a pair tried 4
times gets 0.25, and a pair tried 100 times gets 0.05. An unvisited pair
stays put in the empirical model, so it earns its bonus of 0.5 at every step
and is worth at least $0.5 / (1 - 0.95) = 10$. That's more than the 0.1 the agent has
found by swimming left. So the planner sends it to try every untried pair,
including swimming right in states it has barely reached. Each visit shrinks
the bonus, and only the far end, where the real reward is 1, stays
attractive once the bonuses fade.

In the exercise, this agent replans every 20 steps. It finds the far end and
ends with the optimal policy on every one of 20 seeds within 1,000 steps. In
2,000 steps it spends between 1,414 and 1,565 of them in state 5.

### Pseudo-counts for large spaces

Counting works when states repeat. In Atari, no two frames are exactly alike,
so every count is zero. Bellemare and others (2016) derive a *pseudo-count*
from any density model $\rho$ over states. Let $\rho(s)$ be the model's
probability of $s$ before one more training step on $s$, and $\rho'(s)$ its
probability after. Then

$$
\hat N(s) = \frac{\rho(s)\,\big(1 - \rho'(s)\big)}{\rho'(s) - \rho(s)}
$$

Read it as: a state the model has seen often barely moves when trained on
once more, so $\rho' - \rho$ is small and $\hat N$ is large. A novel state
jumps, so $\hat N$ is small. The formula comes from asking what count $\hat N$ and total $\hat n$ would give $\rho = \hat N / \hat n$ and $\rho' = (\hat N + 1)/(\hat n + 1)$, and solving. For a model that is empirical
counts, it returns the exact count. For example, $N = 3$ of $n = 10$ gives
$\rho = 0.3$ and $\rho' = 4/11$, and the formula returns 3. With $\rho = 0.1$
and $\rho' = 0.12$ it returns $0.1 \times 0.88 / 0.02 = 4.4$.

## Posterior sampling

Optimism adds a fixed bonus. Posterior sampling, also called Thompson
sampling for MDPs, explores by acting on a guess instead. Keep a probability
distribution over what the world might be, the *posterior*. Every so often,
draw one complete model from it, plan in that model, and follow its optimal
policy until the next draw. Osband, Russo, and Van Roy (2013) call this PSRL
and prove it has low regret.

For transitions, the natural posterior is a *Dirichlet* distribution. It's a
distribution over probability vectors. With a prior of 1 pseudo-observation
on every next state, the posterior for row $(s, a)$ after counts $C[s, a, \cdot]$ is

$$
P(\cdot \mid s, a) \sim \operatorname{Dirichlet}\big(C[s, a, \cdot] + 1\big)
$$

Read it as: each next state gets a weight of one plus the number of times you
saw it, and sampled probability vectors cluster around those weights,
normalised. More data means a tighter cluster. To sample one, draw a gamma
variable for each weight and divide by their sum:

```python
g = rng.standard_gamma(counts + prior)       # (S, A, S)
P = g / g.sum(axis=2, keepdims=True)
```

For rewards, the exercise uses a Gaussian prior with mean 0 and variance 1,
and assumes unit noise on each observed reward. After $N$ visits with reward
total $\Sigma$, the posterior is Gaussian with mean $\Sigma / (N + 1)$ and
standard deviation $1 / \sqrt{N + 1}$.

### Worked example

Say that in state 1, swimming right has been tried 8 times: it moved right 3
times, stayed 5 times, and never slipped. The Dirichlet weights over states 0
to 5 are $[1, 6, 4, 1, 1, 1]$, which sum to 14. The posterior mean chance of
moving right is $4/14 = 0.286$, close to the true 0.35. The posterior still
gives $1/14 = 0.071$ to jumping straight to state 5, which is impossible.
Early on, that stray mass makes some sampled models think the far end is
close, so the agent swims right to find out. As counts grow, the stray mass
shrinks, and the sampled models agree with reality.

The reward posterior after those 8 visits, with total reward 0, has mean 0
and standard deviation $1/3$. An unvisited pair has standard deviation 1, so
some draws make it look rewarding. The exercise's posterior-sampling agent
finds the optimal policy on all 20 seeds within 1,000 steps, and spends 1,043
to 1,388 of 2,000 steps in state 5.

> [!KEY]
> Optimism explores by inflating every uncertain value. Posterior sampling
> explores by acting on one plausible world at a time. Both commit to a plan
> for many steps, which is what dithering can't do.

## Random network distillation

A count or a density model needs states it can compare. For images or other
large observations, random network distillation (RND; Burda and others, 2019)
uses a trick. Fix a randomly initialised network $f$, the *target*, and never
train it. Train a second network $\hat f_\phi$, the *predictor*, to match the
target's output on every state the agent visits. The bonus for a state is
the predictor's error:

$$
b(s) = \big\lVert \hat f_\phi(s) - f(s) \big\rVert^2
$$

Read it as: the target maps each state to a fixed random vector. On states
the predictor has trained on, it has learned to reproduce that vector, so
the bonus is small. On states it has never seen, it hasn't, so the bonus is
large. The target is a fixed function of the state, so on familiar states the
error can shrink toward zero. In PyTorch:

```python
with torch.no_grad():
    bonus = ((predictor(states) - target(states)) ** 2).sum(dim=1)
```

In the exercise, both networks take a two-dimensional state, with one hidden
layer of 32 units and 8 outputs. After 300 Adam steps on 256 states from the
square $[0, 0.5]^2$, the mean bonus there falls from 0.49 to 0.0001. On 256
states from $[0.5, 1]^2$, which it never saw, the mean bonus is 0.011, over
100 times higher. Across eight seeds the ratio stays above 90.

> [!WARNING]
> A novelty bonus for prediction error of the *next state*, rather than of a
> fixed target, gets stuck on anything random. A television showing static
> is always hard to predict, so the agent stares at it. Burda and others
> (2019) call this the noisy-TV problem. RND removes the part that comes from
> random transitions, because its target depends only on the current state.
> A screen that keeps showing images the agent has never seen still looks
> novel to it.

> [!RECAP]
> - The count bonus is $\beta / \sqrt{\max(N, 1)}$, added to the empirical
>   reward before planning.
> - An unvisited pair in the empirical model stays put and has reward 0.
> - A posterior model samples each row from $\operatorname{Dirichlet}(C + 1)$
>   and each reward from a Gaussian with mean $\Sigma / (N + 1)$.
> - All three agents replan every 20 steps and report the greedy policy of
>   the final empirical model, with no bonus.
> - The RND bonus is the squared distance between predictor and target.

## Build it step by step

The checks test the functions in this order.

1. **`count_bonus(counts, beta)`.** One line with `np.maximum` and
   `np.sqrt`. Common mistake: dividing by zero for unvisited pairs.
2. **`pseudo_count(rho, rho_prime)`.** One line, elementwise.
3. **`empirical_model(transition_counts, reward_sums)`.** Sum counts over the
   last axis to get `N`. Divide where `N > 0`. For unvisited pairs, find them
   with `np.nonzero(N == 0)` and set `P[s, a, s] = 1`. Return `(P, R, N)`.
   Common mistake: leaving unvisited rows as zeros, so they aren't
   probability vectors and value iteration treats them as dead ends.
4. **`plan(P, R, gamma)`.** Iterate `Q = R + gamma * P @ Q.max(axis=1)` from
   zeros until it stops moving. `P @ v` with `P` of shape `(S, A, S)` gives
   shape `(S, A)`.
5. **`sample_posterior_mdp(transition_counts, reward_sums, rng, prior)`.**
   Gamma draws for `P`, then a Gaussian draw for `R`, in that order. Common
   mistake: using `N` instead of `N + 1` in the reward posterior, which
   divides by zero on unvisited pairs.
6. **`explore(mdp, method, steps, seed, ...)`.** One loop for all three
   methods. Replan when `t % replan_every == 0`, pick an action, step, update
   `counts[s, a, s2]` and `reward_sums[s, a]`, and reset on termination or
   truncation. Common mistake: returning the greedy policy of the bonus
   model. The checks want what the agent believes without the bonus.
7. **`make_rnd`, `rnd_bonus`, and `rnd_train`.** Seed, build two
   `nn.Sequential` networks, and freeze the target with
   `requires_grad_(False)`. The bonus sums squared differences over the
   output axis under `torch.no_grad()`. Training runs Adam on the mean bonus.
   Common mistake: passing the target's parameters to the optimiser, which
   lets both networks chase each other.

## Cost and scaling

The following table shows how each method's cost grows with $S$ states, $A$
actions, $T$ steps, a replanning interval $K$, and $I$ value-iteration
sweeps.

| Method | Time | Memory |
|---|---|---|
| Epsilon-greedy with a model | $O(T/K \cdot I S^2 A)$ planning | $O(S^2 A)$ counts |
| Count-based optimism | the same | the same |
| Posterior sampling | the same, plus $O(S^2 A)$ per sample | the same |
| RND | one forward pass per state, one backward pass per update | two networks |

The tabular methods all pay for planning, which grows with the square of the
number of states. That's why they stop at small problems. RND's cost doesn't
depend on the number of states at all, which is its point. Deep versions of
posterior sampling, such as bootstrapped DQN, replace the sampled model with
one of several value-function heads, chosen at random each episode.

## Where this breaks

**The bonus scale is a guess.** Too small a $\beta$ and optimism behaves like
greedy. Too large and the agent wanders long after it has found the reward.
Theory gives values of $\beta$ that are far too cautious in practice, so
people tune it.

**The prior is a guess too.** Posterior sampling is only as good as its
prior. A prior that rules out the true world, or a reward noise model far
from reality, can stall it or send it wandering.

**Novelty isn't usefulness.** A bonus rewards every novel state, including
ones that don't matter. In a large world, an agent can spend its whole budget
on novel but useless states. Burda and others (2019) find that curiosity
alone produces surprisingly competent play in many games, but it gets
distracted by sources of randomness.

**Intrinsic rewards need care.** RND works best with the intrinsic reward
normalised by a running estimate of its spread, with a separate value head
for it, and with the intrinsic return not reset at episode ends. Without
those, the bonus either swamps the real reward or vanishes.

**Detachment.** An agent can find a promising frontier, get distracted, and
forget how to return. Go-Explore (Ecoffet and others, 2021) keeps an archive
of frontier states and returns to them before exploring further.

## Further reading

- [Strehl and Littman (2008), An analysis of model-based interval estimation for Markov decision processes](https://doi.org/10.1016/j.jcss.2007.08.009).
  MBIE and MBIE-EB, the count bonus used here, with RiverSwim as a test.
- [Osband, Russo, and Van Roy (2013), (More) efficient reinforcement learning via posterior sampling](https://arxiv.org/abs/1306.0940).
  PSRL and its regret bound.
- [Bellemare and others (2016), Unifying count-based exploration and intrinsic motivation](https://arxiv.org/abs/1606.01868).
  Pseudo-counts from density models, with large gains on Montezuma's Revenge.
- [Osband, Blundell, Pritzel, and Van Roy (2016), Deep exploration via bootstrapped DQN](https://arxiv.org/abs/1602.04621).
  Posterior-style exploration with an ensemble of value heads.
- [Burda, Edwards, Storkey, and Klimov (2019), Exploration by random network distillation](https://arxiv.org/abs/1810.12894).
  RND, and the normalisation details that make it work.
- [Burda and others (2019), Large-scale study of curiosity-driven learning](https://arxiv.org/abs/1808.04355).
  Curiosity without any external reward, and the noisy-TV problem.
- [Ecoffet and others (2021), First return, then explore](https://arxiv.org/abs/2004.12919).
  Go-Explore's archive of frontier states.

## The exercise

You build three exploring agents that share one model-based loop, and a
novelty bonus for large state spaces. Write `count_bonus`, `pseudo_count`,
`empirical_model`, and `plan` as the building blocks, and
`sample_posterior_mdp` for the posterior. Then write `explore`, which runs
the `"epsilon"`, `"optimism"`, or `"posterior"` agent on RiverSwim. Finish
with `make_rnd`, `rnd_bonus`, and `rnd_train`.

The checks verify the bonus and pseudo-count formulas exactly, the empirical
model on hand-built counts, and `plan` against $Q^*$. They check that the
average of 4,000 posterior samples matches the Dirichlet and Gaussian means.
Then they run each agent for 2,000 steps on three seeds: epsilon-greedy must
never reach state 5, while optimism and posterior sampling must end with the
optimal policy and spend over 500 steps there. Finally, they check
`rnd_bonus` on a hand-set network, and that training cuts the bonus on
visited states by 99% while novel states keep a bonus over 10 times higher.
