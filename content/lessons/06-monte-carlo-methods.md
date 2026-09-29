---
title: Monte Carlo methods
slug: 06-monte-carlo-methods
part: "Part 2 — Learning from experience"
summary: Estimate values by averaging the returns of real episodes, improve a policy with them, and reuse another policy's data.
minutes: 55
gpu: false
objectives:
  - Compute first-visit and every-visit Monte Carlo value estimates from complete episodes.
  - Run on-policy Monte Carlo control with an epsilon-greedy policy, and say which policy it converges toward.
  - Compute ordinary and weighted importance-sampling estimates, and explain why one is unbiased and the other is stable.
prereqs:
  - 01-the-rl-problem
  - 04-markov-decision-processes
  - 05-dynamic-programming
exercise: 06-monte-carlo-methods
---

# Monte Carlo methods

Dynamic programming finds exact values, but only if you hand it the whole
model: every transition probability and every expected reward. Most real
problems don't come with that table. A robot doesn't know its gripper's
physics to four decimal places. What it can do is try things and see what
happens.

Monte Carlo methods learn from exactly that. Play an episode to the end, add
up the rewards that followed each state, and average those totals over many
episodes. It's how you'd estimate the length of a commute: drive it twenty
times and take the mean.

In this lesson you estimate a fixed policy's values from episodes, use those
estimates to improve the policy while it keeps exploring, and estimate one
policy's value from episodes that a different policy played.

## What you already know

- A *policy* $\pi(a \mid s)$ is the probability of taking action $a$ in state
  $s$. In code it's an `(S, A)` array whose rows sum to one.
- The *return* $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$ is the
  discounted sum of rewards from step $t$ to the end of the episode.
- $V^\pi(s)$ is the expected return from $s$ when you follow $\pi$, and
  $Q^\pi(s, a)$ is the expected return after taking $a$ in $s$ first.
- `rlenvs.policy_evaluation` solves the Bellman equation exactly, as you did
  in [Markov decision processes](04-markov-decision-processes).
- Policy iteration from [Dynamic programming](05-dynamic-programming)
  alternates two steps: evaluate the policy, then make it greedy with respect
  to the values.

## Average returns to estimate values

A state's value is an expected return, and a sample mean estimates an
expectation. So run episodes with $\pi$, record the return that followed each
visit to $s$, and average them.

An *episode* runs from a start state until it reaches a *terminal* state, a
state with no future. Here an episode is three arrays of length $T$: states
$s_0, \dots, s_{T-1}$, actions $a_0, \dots, a_{T-1}$, and rewards $r_0,
\dots, r_{T-1}$. The terminal state isn't stored, because its value is zero.

### Compute every return in one backward pass

Adding up the future rewards separately for every step would cost $T^2$
work. The return has a recursive shape that makes it cost $T$:

$$
G_t = r_t + \gamma\, G_{t+1}, \qquad G_T = 0
$$

Read it as: the return from step $t$ is the reward you get right away, plus
the discounted return from the next step. Start at the end, where nothing is
left, and walk backward.

```python
running = 0.0
for t in reversed(range(T)):
    running = rewards[t] + gamma * running
    G[t] = running
```

### First visit or every visit

A state can appear more than once in an episode. *First-visit* Monte Carlo
uses only the return after the first visit to $s$ in each episode.
*Every-visit* Monte Carlo uses the return after each visit. Both converge to
$V^\pi(s)$. First-visit is also unbiased after any number of episodes,
because each episode adds one independent sample. Every-visit is biased for
finitely many episodes, because returns from one episode overlap.

Here's a worked example. One episode visits states $A, B, A$ and then ends,
with rewards $(1, 0, 2)$ and $\gamma = 1$. The backward pass gives:

- $G_2 = 2$.
- $G_1 = 0 + 2 = 2$.
- $G_0 = 1 + 2 = 3$.

First-visit Monte Carlo takes $V(A) = G_0 = 3$ and $V(B) = G_1 = 2$.
Every-visit Monte Carlo averages both visits to $A$, so $V(A) = (3 + 2) / 2 =
2.5$, and still $V(B) = 2$.

In code, keep a total and a count per state, and a `set` of states already
seen in the episode:

```python
seen = set()
for t, s in enumerate(states):
    if first_visit and s in seen:
        continue
    seen.add(s)
    total[s] += G[t]
    count[s] += 1
```

### Check it on the random walk

Sutton and Barto's random walk is five states in a row, labelled 1 to 5,
with an exit at each end. Every step moves left or right with probability one
half, and leaving on the right pays 1. Episodes start at state 3. With
$\gamma = 1$, the value of state $k$ is the probability of leaving on the
right, which is $k / 6$.

Each return is 0 or 1, so an average over $n$ first visits has standard error
$\sqrt{V(1 - V) / n}$. After 2,000 episodes the middle state has 2,000
visits and a standard error of about 0.011.

> [!KEY]
> Monte Carlo prediction needs no model. It averages the returns that
> actually followed each state, so it's unbiased, but it must wait for every
> episode to end.

## Improve the policy with action values

To improve a policy you compare actions. With a model, $V$ and a one-step
lookahead are enough. Without one, Monte Carlo control estimates $Q(s, a)$,
the average return after taking $a$ in $s$, and the greedy action is
`Q[s].argmax()`.

The loop is policy iteration with samples: play an episode, move $Q$ toward
the returns you saw, make the policy greedier, and repeat. This alternation is
called *generalised policy iteration*.

### Keep exploring

A fully greedy policy has a fatal flaw here. If it never tries an action, that
action's $Q$ never changes. The standard fix is an *$\varepsilon$-greedy* policy. With
probability $\varepsilon$ it picks an action uniformly at random, and
otherwise it picks the greedy action. With $A$ actions:

$$
\pi(a \mid s) =
\begin{cases}
1 - \varepsilon + \dfrac{\varepsilon}{A} & \text{if } a = \operatorname*{arg\,max}_{a'} Q(s, a') \\[4pt]
\dfrac{\varepsilon}{A} & \text{otherwise}
\end{cases}
$$

Read it as: every action gets a floor of $\varepsilon / A$, its share of
the random pick, and the greedy action also gets the remaining $1 -
\varepsilon$. A policy that gives every action at least $\varepsilon / A$ is
*$\varepsilon$-soft*. The $\varepsilon$-greedy policy is the
$\varepsilon$-soft policy closest to greedy.

```python
pi = np.full((S, A), epsilon / A)
pi[np.arange(S), Q.argmax(axis=1)] += 1.0 - epsilon
```

For $Q(s, \cdot) = (0.5, 2.0, 1.0, -1.0)$ and $\varepsilon = 0.2$, the floor
is $0.2 / 4 = 0.05$, and the greedy action 1 gets $0.8 + 0.05 = 0.85$. The
row is $(0.05, 0.85, 0.05, 0.05)$.

### Use a constant step size

The update for each first-visited pair in an episode is

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \bigl(G_t - Q(s_t, a_t)\bigr)
$$

Read it as: find the gap between the return you saw and your current
estimate, and close a fraction $\alpha$ of it. With $\alpha = 1/N$, where
$N$ counts the visits so far, this is the running mean. The exercise uses a
constant $\alpha$ instead. The policy keeps changing, so early returns come
from a worse policy, and a running mean weights those stale returns forever.
A constant $\alpha$ shrinks an old return's weight by $1 - \alpha$ per
update.

### Know what it converges to

On-policy control always explores, so it can't converge to the optimal
policy. It approaches the best $\varepsilon$-soft policy. Sutton and Barto
describe that policy with a modified world, in which every action you choose
is replaced, with probability $\varepsilon$, by a uniformly random one. The
best $\varepsilon$-soft policy is worth exactly what the optimal policy is
worth in the modified world. Build that world's model in two lines and solve
it with value iteration:

```python
P_soft = (1 - eps) * mdp.P + eps * mdp.P.mean(axis=1, keepdims=True)
R_soft = (1 - eps) * mdp.R + eps * mdp.R.mean(axis=1, keepdims=True)
```

On the exercise's grid, three rows of four cells with one wall, the optimal
policy reaches the goal in 3 steps, worth $-3$. The best $\varepsilon$-soft
policy with $\varepsilon = 0.2$ is worth $-3.818$, because it sometimes
stumbles.

> [!KEY]
> On-policy control with $\varepsilon$-greedy exploration converges toward
> the best $\varepsilon$-soft policy, not the optimal one. The price of
> exploring is built into the values it learns.

> [!WARNING]
> No theorem guarantees that this loop settles. With a constant $\alpha$, the
> estimates keep fluctuating, and an unlucky run can lock in a bad greedy
> action for a while. The exercise's check takes the median of five seeded
> runs for that reason.

## Evaluate one policy with another's data

Sometimes a different policy played your episodes. Call it the *behaviour
policy* $b$, and call the one you want to evaluate the *target policy* $\pi$.
Learning about $\pi$ from $b$'s data is *off-policy* learning. Learning about
the policy that generated the data is *on-policy*.

Reweight each of $b$'s returns by how much more likely its actions were under
$\pi$ than under $b$. An episode that $\pi$ would play twice as often counts
twice as much.

### The importance-sampling ratio

The weight for the part of an episode from step $t$ to the end is

$$
\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(a_k \mid s_k)}{b(a_k \mid s_k)}
$$

Read it as: at each step, divide the target policy's probability of the
action taken by the behaviour policy's, and multiply the ratios. The
transition probabilities appear in both policies' trajectory probabilities, so
they cancel, and the ratio needs no model. It does need *coverage*: any action
$\pi$ might take, $b$ must take with some probability.

Every tail product comes from one reversed cumulative product:

```python
step = pi[states, actions] / b[states, actions]
rho = np.cumprod(step[::-1])[::-1]     # rho[t] is rho_{t:T-1}
```

For step ratios $(1.8, 1.6, 0.2)$, the tails are $1.8 \times 1.6 \times 0.2 =
0.576$, then $1.6 \times 0.2 = 0.32$, then $0.2$.

### Ordinary and weighted estimates

Given $n$ episodes from the same start state, with returns $G^{(i)}$ and
weights $\rho^{(i)} = \rho^{(i)}_{0:T-1}$, there are two estimators:

$$
V_{\text{OIS}} = \frac{1}{n} \sum_{i=1}^{n} \rho^{(i)} G^{(i)},
\qquad
V_{\text{WIS}} = \frac{\sum_{i=1}^{n} \rho^{(i)} G^{(i)}}{\sum_{i=1}^{n} \rho^{(i)}}
$$

Read them as follows. *Ordinary importance sampling* (OIS) averages the
weighted returns. It's unbiased, because the weights undo the change of policy
on average. *Weighted importance sampling* (WIS) divides by the total weight
instead of by $n$, so it's a weighted average of returns you saw and can't
leave their range. It's biased: with one episode it returns that episode's
return, which estimates $V^b$. The bias shrinks to zero as $n$ grows.

```python
ois = np.mean(w * G)
wis = np.sum(w * G) / np.sum(w)
```

Here's a worked example. A one-step problem has two actions: $L$ pays 1 and
$R$ pays 0. The target policy has $\pi(L) = 0.9$, so $V^\pi = 0.9$. The
behaviour policy has $b(L) = 0.5$. The ratios are $\rho(L) = 0.9 / 0.5 = 1.8$
and $\rho(R) = 0.1 / 0.5 = 0.2$. Three behaviour episodes give $(L, 1)$,
$(R, 0)$, and $(L, 1)$:

- OIS gives $(1.8 + 0 + 1.8) / 3 = 1.2$. That's above 1, the largest return
  this problem can pay.
- WIS gives $3.6 / (1.8 + 0.2 + 1.8) = 0.947$.

> [!KEY]
> Ordinary importance sampling is unbiased but can have huge variance. Weighted
> importance sampling is biased for small samples but always stays within the
> range of the returns you saw.

### When the variance is infinite

Sutton and Barto's Example 5.5 shows how bad OIS can get. One state has two
actions. One ends the episode with reward 0. The other returns to the state
with probability 0.9, and otherwise ends it with reward 1. The target policy
always loops, so $V^\pi = 1$. The behaviour policy picks each action half the
time.

An episode that loops $k$ times and then ends with reward 1 has probability
$(0.5 \times 0.9)^k (0.5 \times 0.1)$ under $b$, and weight $2^{k+1}$. The
expected squared OIS sample is

$$
\mathbb{E}_b\bigl[(\rho G)^2\bigr] = \sum_{k=0}^{\infty} 0.05 \times 0.45^k \times 4^{k+1} = 0.2 \sum_{k=0}^{\infty} 1.8^k
$$

Read it as: add up probability times squared weight over every episode
length. Each extra loop multiplies the probability by 0.45 and the squared
weight by 4, so each term is 1.8 times the last, and the sum diverges. Its
first 10 terms add to 89, and its first 20 to 31,870. OIS still has mean
exactly 1, so it converges eventually, but with infinite variance no error bar
applies and progress is painfully slow. WIS returns exactly 1 as soon as one
episode with nonzero weight arrives, because each such episode returns 1.

> [!RECAP]
> - Compute all returns with one backward pass: `running = r[t] + gamma * running`.
> - First-visit Monte Carlo averages one return per state per episode. It's
>   unbiased. Every-visit averages all of them.
> - The $\varepsilon$-greedy row is a floor of $\varepsilon / A$, plus
>   $1 - \varepsilon$ on the first greedy action.
> - Control moves $Q$ a fraction $\alpha$ toward each first-visit return and
>   approaches the best $\varepsilon$-soft policy.
> - The ratio $\rho_{t:T-1}$ multiplies $\pi / b$ from $t$ to the end. OIS
>   divides by $n$ and WIS divides by the total weight.

## Build it step by step

The checks test the functions in this order. `random_walk` and
`generate_episode` are already written for you.

1. **`discounted_returns(rewards, gamma)`.** Takes a `(T,)` array and returns
   a `(T,)` array of returns. Loop backward with one running total. Common
   mistake: looping forward, which needs the future before you have it.
2. **`mc_prediction(episodes, n_states, gamma, first_visit=True)`.** Returns
   `V`, shape `(S,)`. Call `discounted_returns` once per episode, then add
   each counted return to `total[s]` and 1 to `count[s]`. Divide with
   `np.divide(..., where=count > 0)` so unvisited states stay 0. Common
   mistake: resetting the `seen` set once for all episodes, instead of once
   per episode.
3. **`epsilon_soft(Q, epsilon)`.** Returns an `(S, A)` policy. Fill with
   `epsilon / A`, then add `1 - epsilon` at `Q.argmax(axis=1)`. Common
   mistake: giving the greedy action $1 - \varepsilon$ and forgetting its
   share of the random pick, so rows sum to less than one.
4. **`mc_control(mdp, n_episodes, epsilon, alpha, seed)`.** Returns `Q`,
   shape `(S, A)`. Make one `np.random.default_rng(seed)`. For each episode,
   draw the start from `mdp.mu0`, then act: with probability `epsilon` pick
   `rng.integers(A)`, otherwise pick at random among
   `np.flatnonzero(Q[s] == Q[s].max())`. Step with `mdp.sample(s, a, rng)`
   until `mdp.terminal[s2]` or `mdp.max_steps` steps. Then update each
   first-visited `(s, a)` pair. Common mistake: always taking `argmax` among
   ties. With $Q = 0$ everywhere, that walks into the same wall for hundreds
   of steps.
5. **`importance_ratios(states, actions, pi, b)`.** Returns a `(T,)` array.
   Index `pi[states, actions]` and `b[states, actions]` to get every step's
   probability at once, divide, and take the reversed cumulative product.
   Common mistake: dividing $b$ by $\pi$.
6. **`importance_sampling_estimates(episodes, pi, b, gamma)`.** Returns
   `(ois, wis)`. For each episode, take `importance_ratios(...)[0]` and
   `discounted_returns(...)[0]`. Common mistake: returning `nan` when every
   weight is zero. Return `0.0` instead.

## Cost and scaling

Time grows with the total number of steps you play, and memory with the
number of states or state-action pairs. The following table summarises the
cost:

| Method | Time per episode | Memory | What grows the error |
|---|---|---|---|
| Monte Carlo prediction | $O(T)$ | $S$ values and counts | Long, noisy episodes |
| Monte Carlo control | $O(T)$ | $S \times A$ values | Rarely tried actions |
| Ordinary importance sampling | $O(T)$ | One weight per episode | Weights that grow with $T$ |
| Weighted importance sampling | $O(T)$ | Two running sums | Bias with few episodes |

The hidden cost is variance. A return sums $T$ random rewards, and an
importance weight multiplies $T$ ratios, so its variance can grow
exponentially with $T$.

## Where this breaks

**Episodes that never end.** Monte Carlo needs complete returns, and a
policy that walks into a wall forever never produces one. Cap the episode
length, or learn from partial episodes, the subject of
[Temporal-difference learning](07-temporal-difference-learning).

**Noisy returns on long episodes.** Every return carries the noise of every
later step. People switch to bootstrapped methods or average more episodes.

**Stuck greedy actions.** An action with an unlucky early estimate gets tried
only $\varepsilon / A$ of the time, so it recovers slowly. A constant step
size, a larger $\varepsilon$, or *exploring starts*, which begin each episode
from a random state and action, all help.

**Exploding importance weights.** One rare long episode can carry a weight in
the thousands. People use weighted importance sampling, cap the weights, or
use per-decision methods that weight each reward only by the ratios up to it.

**Missing coverage.** If the target policy takes an action that the behaviour
policy never takes, no amount of data reveals its value.

## Further reading

- [Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd edition
  (2018), Chapter 5](http://incompleteideas.net/book/the-book-2nd.html).
  The source of the random walk, the modified-world argument for
  $\varepsilon$-soft policies, and the infinite-variance example.
- [Singh and Sutton, "Reinforcement learning with replacing eligibility
  traces" (1996)](https://doi.org/10.1007/BF00114726). Analyses first-visit
  against every-visit Monte Carlo, including the bias of every-visit.
- [Precup, Sutton, and Singh, "Eligibility traces for off-policy policy
  evaluation" (2000)](https://dl.acm.org/doi/10.5555/645529.658134).
  Per-decision importance sampling, which cuts the variance of long-horizon
  weights.
- [Mahmood, van Hasselt, and Sutton, "Weighted importance sampling for
  off-policy learning with linear function approximation"
  (2014)](https://dl.acm.org/doi/10.5555/2969033.2969163). Carries the
  weighted estimator beyond tables.

## The exercise

You build Monte Carlo prediction, control, and off-policy evaluation. Write
`discounted_returns`, `mc_prediction`, `epsilon_soft`, and `mc_control`. Then
write `importance_ratios` for the tail products $\rho_{t:T-1}$, and
`importance_sampling_estimates` for the ordinary and weighted estimates of a
start value.

The checks verify the $A, B, A$ example and the $\varepsilon$-greedy table
by hand, and that random-walk estimates land within four standard errors of
$k / 6$. The median of five seeded `mc_control` runs must come within 0.5 of
the best $\varepsilon$-soft value, computed exactly from the modified world.
The last checks confirm the 1.2 and 0.947 example, and that over 2,000
batches OIS is unbiased while WIS has lower variance and stays in $[0, 1]$.
