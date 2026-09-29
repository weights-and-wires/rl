---
title: Imitation learning
slug: 27-imitation-learning
part: "Part 6 — Models, exploration, and fixed data"
summary: Clone an expert, measure exactly how its small mistakes compound, and close the gap by letting DAgger query the expert.
minutes: 55
gpu: false
objectives:
  - Fit a behaviour-cloning policy to expert demonstrations and evaluate it exactly.
  - Derive the tightrope cost T - (1 - (1 - eps)^T) / eps, and explain why it grows like eps T^2 while a learner that recovers pays about eps T.
  - Measure a learner's disagreement with the expert under the expert's states and under its own, and say why only the second predicts its value.
  - Implement DAgger and show it closes the gap that more expert demonstrations can't.
prereqs:
  - 04-markov-decision-processes
  - 05-dynamic-programming
exercise: 27-imitation-learning
---

# Imitation learning

Writing a reward that says exactly what you want is hard. Showing what you
want is often easy. A driver can drive, a surgeon can operate, and a person
can write a good answer to a question, even when none of them could turn
their skill into a formula. *Imitation learning* trains a policy from those
demonstrations instead of from a reward.

The obvious method is plain supervised learning: record what the expert did
in each state, and train a model to predict it. It works surprisingly often.
It also fails in a specific, predictable way. The learner makes a small
mistake, lands somewhere the expert never went, has no idea what to do
there, and makes a bigger mistake. The errors feed on each other.

This lesson measures that failure exactly, on a two-state tightrope and on a
grid world ledge. Then it fixes it with DAgger, which asks the expert what it
would do in the places the learner actually ends up.

## What you already know

- A *policy* $\pi(a \mid s)$ is a probability distribution over actions in
  each state, stored as an array of shape `(S, A)`.
- The value $V^\pi$ of a policy on a tabular MDP comes from one linear solve.
- Value iteration finds an optimal policy, which serves as the expert here.
- The discounted state distribution $d^\pi$ says how much time a policy spends
  in each state.
- Supervised learning fits a model to labelled examples drawn from some
  distribution, and its guarantees hold on that distribution.

## Clone the expert

*Behaviour cloning* (BC) turns demonstrations into a labelled dataset. Each
visited state is an input, and the expert's action there is its label. Then
it fits a policy by supervised learning. ALVINN (Pomerleau, 1989) followed
roads this way, with a small neural network mapping camera images to a
steering direction.
Fine-tuning a language model on example answers is the same method.

On a tabular problem, the fit is counting. With $n(s, a)$ the number of times
the expert took action $a$ in state $s$:

$$
\hat\pi(a \mid s) = \frac{n(s, a)}{\sum_b n(s, b)}, \qquad \hat\pi(a \mid s) = \frac{1}{A} \text{ if } s \text{ was never visited}
$$

Read it as: in a state the expert visited, copy its action frequencies. In a
state it never visited, you have no information, so choose uniformly among
the $A$ actions. This is the maximum-likelihood fit, and on the expert's own
states it's perfect. In numpy:

```python
counts = np.zeros((S, A))
np.add.at(counts, (states, actions), 1.0)
totals = counts.sum(axis=1, keepdims=True)
pi = np.where(totals > 0, counts / np.maximum(totals, 1), 1.0 / A)
```

For example, if state 0 has labels $[1, 1, 2, 1]$ over three actions, its row
becomes $[0, 0.75, 0.25]$. A state with no labels gets $[1/3, 1/3, 1/3]$.

Real learners aren't perfect even on the states they trained on. A network
that fits 99% of its labels still gets 1% wrong. The exercise makes this
explicit. Every learned policy, when it acts, takes a uniformly random action
with probability $\varepsilon$. The expert never does.

## Why small mistakes compound

The *tightrope* makes the problem exact. It has two states: on the rope and
off it. Every step spent off the rope costs 1. The learner stays on with
probability $1 - \varepsilon$ and falls off with probability $\varepsilon$.
The expert never falls, so its demonstrations never show a fallen state. The
cloned learner, off the rope, has no idea how to climb back, and in the
worst case never does.

The chance that it is still on the rope at step $t$ is $(1 - \varepsilon)^t$,
so the chance it is off is $1 - (1 - \varepsilon)^t$. The expected cost over
$T$ steps adds these up:

$$
\text{cost}(T) = \sum_{t=0}^{T-1} \Big(1 - (1 - \varepsilon)^t\Big) = T - \frac{1 - (1 - \varepsilon)^T}{\varepsilon}
$$

Read it as: at each step, pay the probability of already having fallen. The
closed form comes from summing the geometric series $\sum_t (1 - \varepsilon)^t$.
When $\varepsilon T$ is small, this is about $\varepsilon T^2 / 2$. A fall at
step $t$, which happens with probability about $\varepsilon$, is paid for at
each of the $T - t$ steps that remain. Summing $\varepsilon (T - t)$ over $t$
gives the triangle $\varepsilon T^2 / 2$.

Give the learner the recovery action: off the rope, it climbs straight
back on. Each fall then costs only one step, and the cost is about
$\varepsilon T$. The exact answer is
$\frac{\varepsilon}{1 + \varepsilon}\big(T - \frac{1 - (-\varepsilon)^T}{1 + \varepsilon}\big)$.
You compute both in numpy by pushing the state distribution forward:

```python
d, total = mu0.copy(), 0.0
for _ in range(T):
    total += d @ r_pi        # expected reward this step
    d = d @ P_pi             # distribution one step later
```

### Worked example

With $\varepsilon = 0.01$, the following table shows the expected cost:

| Horizon $T$ | Never recovers | Recovers at once | $\varepsilon T^2 / 2$ | $\varepsilon T$ |
|---|---|---|---|---|
| 10 | 0.44 | 0.09 | 0.5 | 0.1 |
| 100 | 36.60 | 0.98 | 50 | 1 |
| 1,000 | 900.00 | 9.89 | 5,000 | 10 |

At $T = 100$, a learner that is right 99% of the time pays 36.6, while the
same learner with a recovery action pays about 1. At $T = 1{,}000$ the first
cost stops growing quadratically, because it can't exceed the $T$ steps
available. It's still 90 times the second.

This isn't a quirk of the example. Ross and Bagnell (2010) prove that a
cloned policy whose error rate on the expert's states is $\varepsilon$ has a
cost at most $T^2 \varepsilon$ above the expert's, for costs between 0 and 1.
The tightrope shows the bound is tight: some problems do pay the
square.

> [!KEY]
> A cloned policy pays for each mistake at every later step, because nobody
> showed it how to recover. Its extra cost can grow like $\varepsilon T^2$,
> against $\varepsilon T$ for a learner that recovers.

## Measure the error where the learner goes

Supervised learning promises low error on the distribution it trained on.
For BC that's the expert's state distribution. But the learner is judged on
its own. The exercise measures both with one quantity, the learner's
*disagreement* with the expert $\pi^*$ under a policy $\pi$'s states:

$$
\text{disagreement}(\hat\pi; d^\pi) = \sum_s d^\pi(s) \Big( 1 - \sum_a \hat\pi(a \mid s)\, \pi^*(a \mid s) \Big)
$$

Read it as: for each state, take the chance that the learner's action differs
from the expert's, and weight it by how much time $\pi$ spends there. With
$\pi = \pi^*$ this is BC's training error. With $\pi$ the learner itself,
it's the error that decides the learner's value. In numpy:

```python
d = rlenvs.state_distribution(mdp, pi_dist)
miss = 1.0 - (learner * expert).sum(axis=1)
error = d @ np.where(mdp.terminal, 0.0, miss)
```

The exercise's ledge world has three rows of 10 cells. The expert walks the
top row from the start to the goal, which pays 1. The middle row is safe. The
bottom row is holes, which cost 1 and end the episode. Moves are
deterministic, and $\gamma = 0.99$. With $\varepsilon = 0.1$, the results are
as follows:

| Policy | Value from the start | Error on expert states | Error on its own states |
|---|---|---|---|
| Expert | 0.9227 | 0 | 0 |
| Expert, with the same mistakes | 0.8995 | 0 | 0 |
| BC, with mistakes | 0.6979 | 0 | 0.0319 |

The fairest reference is the expert making the same random mistakes, because
no learner can avoid those. BC loses 0.20 against it. Its error on the
expert's states is exactly zero, so no supervised-learning metric would warn
you. A random mistake that steps down lands the learner in the middle row,
which the expert never visited. There the learner acts uniformly, and a
quarter of its moves drop it into a hole.

More demonstrations don't help at all here. The expert and the world are
both deterministic, so one demonstration and fifty cover exactly the same
nine cells. BC's value is 0.6979 either way.

> [!WARNING]
> Low validation error on held-out demonstrations says nothing about how a
> cloned policy will do when it acts. Evaluate it by running it, not by its
> loss on the expert's data.

## DAgger: ask the expert where you are

DAgger, short for dataset aggregation (Ross, Gordon, and Bagnell, 2011),
collects labels on the learner's own states:

1. Roll out the expert and record the states it visits. Label each with the
   expert's action, and train a first policy. This is BC.
2. Roll out the current learner, mistakes and all. Record the states it
   visits.
3. Ask the expert what it would do in each of those states, and add the
   labelled states to the dataset.
4. Retrain on the whole dataset, and repeat from step 2.

The paper allows mixing the expert into the rollouts, acting with
probability $\beta_i$ at iteration $i$. It reports that the simplest choice,
$\beta_1 = 1$ and $\beta_i = 0$ afterward, often works best. That's the
choice described above.

Because the dataset keeps growing with states the learner reaches, the
learner eventually knows what to do everywhere it goes, including how to
recover from its own mistakes. Ross and others prove a cost within about
$u T \varepsilon_N$ of the expert's, where $\varepsilon_N$ is the learner's
error on its own distribution and $u$ bounds how much one wrong action can
raise the expert's remaining cost. That's linear in $T$ when mistakes are
recoverable.

On the ledge, with 20 rollouts per iteration and seed 0:

| Iteration | Value, with mistakes | Error on its own states |
|---|---|---|
| 1 (BC) | 0.6979 | 0.0319 |
| 2 | 0.8386 | 0.0091 |
| 5 | 0.8991 | 0.0001 |
| 15 | 0.8995 | 0.0000 |

By iteration 5 the learner is within 0.0004 of the expert making the same
mistakes. Over 300 seeds, every run of 15 iterations ends within 0.0004.

> [!KEY]
> BC trains on the expert's states and is tested on its own. DAgger trains
> on its own states, so its training error is the error that matters.

The price is an expert on call. DAgger needs someone who can label arbitrary
states, including strange ones the learner wandered into, not only record
their own demonstrations. For a driving simulator with a planning expert
that's cheap. For a human expert it can be slow or awkward, since labelling a
state you aren't controlling is harder than driving.

> [!RECAP]
> - BC counts actions per state and uses uniform rows where there's no data.
> - Every learned policy acts with $\varepsilon$ mistakes, through
>   `add_mistakes`, and the expert doesn't.
> - The tightrope cost is $T - (1 - (1 - \varepsilon)^T)/\varepsilon$ without
>   recovery.
> - Disagreement weights $1 - \sum_a \hat\pi \pi^*$ by a state distribution
>   and ignores terminal states.
> - DAgger rolls out the expert once, then the noisy learner, and always
>   labels with the expert.

## Build it step by step

The checks test the functions in this order.

1. **`fit_bc(states, actions, n_states, n_actions)`.** Count with `np.add.at`,
   normalise rows, and fill empty rows with `1 / A`. Common mistake: using
   `counts[states, actions] += 1`, which counts a repeated pair only once.
2. **`add_mistakes(policy, eps)`.** One line. Each row still sums to one.
3. **`finite_horizon_return(mdp, pi, T)`.** Build `P_pi` with `np.einsum` and
   `r_pi` with a row sum, zero the terminal rows and entries, then push `d`
   forward `T` times. Common mistake: discounting. This sum is undiscounted.
4. **`collect(mdp, policy, n_episodes, max_steps, rng)`.** Draw the start from
   `mu0`, then loop: stop at a terminal state, record the state, sample an
   action, and call `mdp.sample`. Common mistake: recording the terminal
   state, which gives the expert a label in the goal cell.
5. **`disagreement(mdp, learner, expert, pi_dist)`.** Take
   `rlenvs.state_distribution(mdp, pi_dist)` and dot it with the per-state
   miss rate. Zero the terminal states.
6. **`dagger(mdp, expert, eps, n_iters, n_episodes, max_steps, seed)`.** Keep
   one growing array of states. Label with `expert.argmax(axis=1)`, refit with
   `fit_bc` each iteration, and switch the rollout policy to
   `add_mistakes(learner, eps)` after the first. Common mistake: refitting
   on only the most recent iteration's states, which forgets how to walk the
   expert's own path.

## Cost and scaling

The following table shows how each piece grows, for $S$ states, $A$ actions,
$n$ labelled states, $T$ horizon steps, and $K$ DAgger iterations.

| Piece | Time | Memory |
|---|---|---|
| Tabular BC fit | $O(n + S A)$ | $O(S A)$ |
| Finite-horizon return | $O(T S^2)$ | $O(S^2)$ |
| Disagreement | one linear solve, $O(S^3)$ | $O(S^2)$ |
| DAgger | $K$ rollouts and refits, and one expert label per visited state | the aggregated dataset |

For a neural policy, each DAgger iteration retrains on a dataset that grows
by one batch of rollouts. The expensive resource is rarely compute. It's the
expert's labels, which DAgger needs $K$ times over, on states the expert
didn't choose.

## Where this breaks

**No expert on call.** DAgger needs interactive labels. When you only have a
fixed set of demonstrations, other fixes apply. DART (Laskey and others,
2017) injects noise into the expert's actions while it demonstrates, so the
expert shows how to recover from small perturbations. Inverse RL and GAIL,
in the next lesson, learn a reward from demonstrations and then optimise it.

**Causal confusion.** A cloned policy can latch onto a cue that correlates
with the expert's action without causing it. de Haan and others (2019) show a
driving policy that sees its own brake light learns to brake when the light
is on, and that giving it more information can make it worse.

**Several right answers.** When an expert sometimes goes left around an
obstacle and sometimes right, a model that predicts the average action goes
straight into it. Policies that model the whole action distribution, such as
diffusion policies (Chi and others, 2023), handle this.

**An expert the learner can't copy.** If the learner sees less than the
expert, or can't represent the expert's policy, some disagreement never goes
away. DAgger's guarantee is relative to the best policy in the learner's
class.

## Further reading

- [Pomerleau (1989), ALVINN: an autonomous land vehicle in a neural network](https://proceedings.neurips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html).
  Behaviour cloning for road following, from camera images to steering.
- [Ross and Bagnell (2010), Efficient reductions for imitation learning](https://proceedings.mlr.press/v9/ross10a.html).
  The $T^2 \varepsilon$ bound for behaviour cloning, and why it's tight.
- [Ross, Gordon, and Bagnell (2011), A reduction of imitation learning and structured prediction to no-regret online learning](https://arxiv.org/abs/1011.0686).
  DAgger and its guarantee that grows linearly in the horizon.
- [Laskey, Lee, Fox, Dragan, and Goldberg (2017), DART: noise injection for robust imitation learning](https://arxiv.org/abs/1703.09327).
  Recovery data without interactive labels, by perturbing the demonstrator.
- [de Haan, Jayaraman, and Levine (2019), Causal confusion in imitation learning](https://arxiv.org/abs/1905.11979).
  When more information makes a cloned policy worse.
- [Chi and others (2023), Diffusion policy: visuomotor policy learning via action diffusion](https://arxiv.org/abs/2303.04137).
  Behaviour cloning with a model that represents several good actions.

## The exercise

You measure behaviour cloning's failure exactly and fix it. Write `fit_bc`
for the counting fit and `add_mistakes` for a learner's random errors. Write
`finite_horizon_return` to compute an exact $T$-step return, `collect` to roll
out a policy, and `disagreement` to weight a learner's errors by a state
distribution. Then write `dagger`.

The checks verify `fit_bc` and `add_mistakes` on small arrays, and
`finite_horizon_return` against both tightrope closed forms, including 36.60
for $\varepsilon = 0.01$ and $T = 100$. They check that `collect` skips
terminal states and visits states at the rate the exact distribution
predicts, and that `disagreement` matches the tightrope's closed form. On the
ledge, they confirm that BC has zero error on the expert's states yet loses
over 0.15 of value with 1 or 50 demonstrations, and that DAgger ends within
0.005 of the expert making the same mistakes on three seeds.
