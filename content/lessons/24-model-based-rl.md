---
title: Model-based reinforcement learning
slug: 24-model-based-rl
part: "Part 6 — Models, exploration, and fixed data"
summary: Fit a dynamics model, measure how its error compounds over a rollout, and plan through it with the cross-entropy method.
minutes: 60
gpu: false
objectives:
  - Fit a linear dynamics model by ridge regression and predict next states with it.
  - Measure k-step prediction error, and explain with the error recursion why it grows with the horizon.
  - Implement the cross-entropy method and use it for model-predictive control that balances CartPole.
  - Say why replanning at every step makes MPC tolerate a model that drifts over long horizons.
prereqs:
  - 10-planning-and-learning
  - 12-deep-q-networks
exercise: 24-model-based-rl
---

# Model-based reinforcement learning

Every method since Part 3 learns a policy or a value function straight from
experience. Each real step feeds one update and is then mostly forgotten. On a
simulator that costs nothing. On a robot, a real step takes time, wears out
motors, and can break things. Model-free methods such as PPO often need
millions of steps, which a real robot can't afford.

A model-based method first learns how the world moves: given this state and
this action, where do I end up? That's an ordinary supervised learning
problem, and each real step gives one labelled example. Once the model exists,
the agent can try out thousands of imagined action sequences inside it and
pick the best one, without touching the real system.

The catch is that the model is wrong, and it's more wrong the further ahead
you ask it to look. In this lesson you fit a model, measure how its error
grows over a rollout, and then plan through it anyway with a loop that
corrects itself at every step. It balances CartPole with a model fitted from
2,000 random transitions.

## What you already know

- A *policy* $\pi(a \mid s)$ picks actions, and the *return* $G_t$ adds up
  the discounted rewards from step $t$ onward.
- Dyna, from the lesson on planning, learning, and search, learns a table of
  transitions and replays it to update values.
- Least squares fits a linear map by solving the normal equations, and a ridge
  penalty keeps the solve well posed.
- CartPole, from the lesson on deep Q-networks, pays 1 per step while the pole
  stays within 12 degrees of upright and the cart stays within 2.4 m of the
  centre.
- Monte Carlo tree search plans by simulating ahead with a model it's given.

## Learn a model of the dynamics

A *dynamics model* $\hat f$ predicts the next state from the current state and
action: $\hat s_{t+1} = \hat f(s_t, a_t)$. The true dynamics $f$ are unknown,
but every real step $(s_t, a_t, s_{t+1})$ is a training example for $\hat f$.
Learning the model is regression, with the next state as the label.

A linear model is the easiest place to start. It predicts $\hat s_{t+1} = A s_t + B a_t + c$,
where $A$ is a $d \times d$ matrix, $B$ is $d \times k$, and $c$ is a bias, for
a state of size $d$ and an action of size $k$. Stack each input as a row
$x = [s, a, 1]$ of length $d + k + 1$. Then the whole model is one weight
matrix $W$ of shape $(d + k + 1, d)$, and a prediction is $x W$. Fit $W$ by
ridge regression on $N$ transitions:

$$
W = \operatorname*{arg\,min}_W \; \tfrac{1}{2}\lVert X W - Y \rVert^2 + \tfrac{\lambda}{2} \lVert W \rVert^2
= (X^\top X + \lambda I)^{-1} X^\top Y
$$

Read it as follows. $X$ holds the $N$ input rows, $Y$ holds the $N$ observed
next states, and $\lambda \ge 0$ is the ridge penalty. The first term is the
squared prediction error summed over every transition and every state
coordinate. The second term shrinks the weights. Setting the gradient
$X^\top(XW - Y) + \lambda W$ to zero gives the closed form on the right. In
numpy:

```python
X = np.concatenate([states, actions, np.ones((N, 1))], axis=1)
W = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ next_states)
```

### Worked example

Take a one-dimensional system whose true rule is
$s_{t+1} = 0.9 s_t + 0.5 a_t + 0.1$. Three transitions are enough to pin down
three weights:

| $s_t$ | $a_t$ | $s_{t+1}$ |
|---|---|---|
| 0 | 1 | $0.9 \cdot 0 + 0.5 + 0.1 = 0.6$ |
| 1 | 0 | $0.9 + 0 + 0.1 = 1.0$ |
| 2 | 1 | $1.8 + 0.5 + 0.1 = 2.4$ |

With $\lambda = 0$, the solve returns $W = [0.9, 0.5, 0.1]^\top$ exactly: the
weight on $s$, the weight on $a$, and the bias. Noise-free data from a truly
linear system always gives back the true weights, provided $X^\top X$ is
invertible. The first check in the exercise tests exactly that.

CartPole isn't linear. Its pole angle enters through a sine and a cosine. But
near upright, $\sin\theta \approx \theta$ and $\cos\theta \approx 1$, so a
linear model fitted on states near upright predicts one step ahead well.
In the exercise, a linear model fitted on 2,000 random transitions has a mean
one-step error of about $4 \times 10^{-4}$.

## Watch the error compound

A one-step error of $4 \times 10^{-4}$ sounds harmless. Planning, though,
asks a longer question: if I take these 30 actions, where do I end up? The
model answers by feeding its own prediction back in as the next input. That's
an *open-loop rollout*: nothing from the real world corrects it on the way.

Each step inherits the previous step's error and adds its own. For the linear
case with a learned $\hat A, \hat B$ and the truth $A, B$, the error
$e_k = \hat s_k - s_k$ after $k$ steps obeys

$$
e_k = \hat A \, e_{k-1} + (\hat A - A)\, s_{k-1} + (\hat B - B)\, a_{k-1}
$$

Read it in two parts. The first term, $\hat A e_{k-1}$, carries the old error
forward through the model's own dynamics, so the model's dynamics can grow it
or shrink it. The last two terms are the fresh one-step error the model makes
at step $k$, even from the correct state. To derive it, subtract
$s_k = A s_{k-1} + B a_{k-1}$ from
$\hat s_k = \hat A \hat s_{k-1} + \hat B a_{k-1}$, then add and subtract
$\hat A s_{k-1}$.

In numpy, the rollout is a loop that keeps its own output:

```python
states = [s0]
for t in range(H):
    states.append(step_fn(states[-1], actions[:, t]))
traj = np.stack(states, axis=1)          # (N, H + 1, d)
```

### Worked example

Take a scalar system $s_{k} = a\, s_{k-1}$ with no actions, a start state
$s_0 = 1$, and a model that overestimates the rate by one percent:
$\hat a = a + 0.01$. The one-step error is always 0.01. The $k$-step error is
$\hat a^k - a^k$:

| True rate $a$ | $k = 1$ | $k = 5$ | $k = 10$ | $k = 20$ | $k = 50$ |
|---|---|---|---|---|---|
| 0.9 (stable) | 0.0100 | 0.0335 | 0.0407 | 0.0301 | 0.0038 |
| 1.0 (neutral) | 0.0100 | 0.0510 | 0.1046 | 0.2202 | 0.6446 |
| 1.1 (unstable) | 0.0100 | 0.0745 | 0.2457 | 1.3348 | 67.1740 |

For $a = 1$, the error is $1.01^k - 1$, which grows about linearly at first
and then faster. A stable system forgets old errors, so the error rises and
then dies away as both predictions decay toward zero. An unstable system
multiplies them, so after 50 steps the prediction is off by 67 when the
one-step error was 0.01.

An upright pole is unstable: left alone, it falls. So CartPole behaves like
the bottom row. The exercise measures it. The linear model's mean error under
random pushes is about $4.3 \times 10^{-4}$ after one step,
$2.1 \times 10^{-3}$ after 10, $0.036$ after 20, and $0.34$ after 30. That's
roughly 800 times the one-step error. By then the rollout has carried the pole
into angles where the linear fit was never trained.

> [!KEY]
> A model trained to be accurate one step ahead can be badly wrong many steps
> ahead, because each step feeds its error into the next. Unstable dynamics
> can make the growth exponential.

## Plan with the cross-entropy method

Once you have a model, planning means choosing an action sequence
$a_0, \ldots, a_{H-1}$ that makes the imagined rollout score well. Write the
score of a plan as

$$
J(a_{0:H-1}) = \sum_{t=0}^{H-1} r(\hat s_{t+1}), \qquad \hat s_0 = s, \quad \hat s_{t+1} = \hat f(\hat s_t, a_t)
$$

Read it as: start from the real current state $s$, run the model forward
under the plan, and add up a reward $r$ for every state the rollout lands in.
Here $r$ is a planning reward you choose. For CartPole it's high when the
pole is upright and the cart is centred, and strongly negative once the
rollout crosses a failure line.

$J$ has no useful gradient for CartPole, because its actions are discrete. The
*cross-entropy method* (CEM) needs no gradient. It's a guided random search:

1. Keep a Gaussian over plans, with a mean $\mu$ and a standard deviation
   $\sigma$ per step.
2. Sample $n$ plans from it and score each one with the model.
3. Keep the $n_e$ best plans, called the *elites*.
4. Refit the Gaussian to the elites, and repeat.

The refit is the mean and spread of the elite set $E$:

$$
\mu \leftarrow \frac{1}{n_e} \sum_{i \in E} u_i, \qquad
\sigma^2 \leftarrow \frac{1}{n_e} \sum_{i \in E} (u_i - \mu)^2
$$

Read it as: the next search centres on the average of the winners, and
spreads as widely as the winners disagree. When the elites agree, $\sigma$
shrinks and the search narrows. In numpy:

```python
elite = samples[np.argsort(-scores)[:n_elite]]
mean, std = elite.mean(axis=0), elite.std(axis=0)
```

For CartPole, each plan is a real vector $u$ of length $H$, and step $t$
pushes right when $u_t > 0$ and left otherwise. A Gaussian over $u$ then
describes a probability of pushing right at each step, which the elites
sharpen.

### Worked example

Maximise $-(u - 1)^2$ over a scalar $u$, with five samples and two elites:

| Sample $u$ | $-1$ | $0.5$ | $2$ | $1.2$ | $0.3$ |
|---|---|---|---|---|---|
| Score | $-4$ | $-0.25$ | $-1$ | $-0.04$ | $-0.49$ |

The two best are $1.2$ and $0.5$. Their mean is $0.85$, and their standard
deviation is $\sqrt{((1.2 - 0.85)^2 + (0.5 - 0.85)^2)/2} = 0.35$. One refit
moves the search centre to $0.85$, most of the way to the optimum at 1.

## Close the loop with model-predictive control

The compounding-error table says a 30-step plan is built on predictions that
are badly wrong by the end. Model-predictive control (MPC) sidesteps most of
that damage. At every real step, it does the following:

1. Plan $H$ steps ahead from the real current state, with CEM through the
   model.
2. Execute only the first action of the best plan.
3. Observe the real next state, throw the rest of the plan away, and plan
   again.

This is called a *receding horizon*: the agent always looks $H$ steps ahead,
but the window slides forward one step at a time. Because every plan starts
from a real state, the model's error can't build up from one real step to the
next. The first action of a plan depends most on the first few predicted
steps, which are the accurate ones. Replanning turns the open-loop model into
a closed-loop controller, and feedback corrects what the model gets wrong.

> [!KEY]
> MPC executes only the first action of each plan and replans from the real
> state. Model error then matters over the few steps that decide the first
> action, not over the whole horizon.

In the exercise, MPC with a 15-step horizon, 64 samples, 8 elites, and 4 CEM
iterations balances CartPole for all 200 steps, with the true physics and
with the fitted linear model. Across ten seeds the pole never leans past
0.05 rad, well inside the 0.21 rad failure line.

> [!TIP]
> Warm-start CEM in practice: drop the first step of the previous mean plan,
> append a zero, and use that as the next starting mean. The exercise starts from zero
> each step to keep the function self-contained.

## Model-based methods in practice

Three designs cover most of the field. Each answers compounding error
differently:

- **Dyna** (Sutton, 1991) uses the model to generate extra training data for
  a model-free learner.
- **PETS** (Chua and others, 2018) plans with CEM and MPC, as you do here,
  through an ensemble of neural networks that each predict a Gaussian over the
  next state. Where the networks disagree, data was scarce, and the planner
  learns not to trust the prediction.
- **MBPO** (Janner and others, 2019) trains soft actor-critic on short model
  rollouts that branch off real states. Its bound on the gap between model
  return and real return grows with the rollout length, so it keeps rollouts
  short.

> [!RECAP]
> - Ridge regression on rows $[s, a, 1]$ fits $W = (X^\top X + \lambda I)^{-1} X^\top Y$.
> - An open-loop rollout feeds each prediction back in, so the $k$-step error
>   carries forward every earlier error.
> - For $\hat a = 1.01$ against $a = 1$, the $k$-step error is exactly $1.01^k - 1$.
> - CEM refits a Gaussian to the top $n_e$ samples, using the population
>   standard deviation.
> - MPC plans from the real state, executes the first action, and replans.

## Build it step by step

The checks test the functions in this order.

1. **`fit_linear_model(states, actions, next_states, ridge)`.** Takes `(N, d)`,
   `(N, k)`, and `(N, d)` arrays. Returns `W` of shape `(d + k + 1, d)`. Build
   `X` with `np.concatenate` and a column of ones, then call `np.linalg.solve`
   on the regularised normal equations. Common mistake: putting the bias
   column first. The checks expect `[A^T; B^T; c]` from top to bottom.
2. **`model_step(W, states, actions)`.** Builds the same `X` and returns
   `X @ W`, shape `(N, d)`. Reshape a `(N,)` action array to `(N, 1)` first.
3. **`rollout(step_fn, s0, actions)`.** Loops over `H` steps, appends each
   output, and stacks into `(N, H + 1, d)` with `s0` first. Common mistake:
   calling `step_fn` on `s0` at every step instead of on the last output.
4. **`k_step_errors(true_step, learned_step, s0, actions)`.** Roll out both
   functions with `rollout`, drop entry 0, take `np.linalg.norm` over the last
   axis, and average over the batch. Returns shape `(H,)`. Common mistake:
   averaging the squared error instead of the distance.
5. **`cem_update(samples, scores, n_elite)`.** Sort by descending score with
   `np.argsort(-scores)`, keep the top rows, and return their mean and
   `std(axis=0)`. Common mistake: keeping the lowest scores. Higher is better
   here.
6. **`cem_optimize(score_fn, mean, std, n_samples, n_elite, n_iters, rng)`.**
   Loop: draw `mean + std * rng.standard_normal((n_samples, D))`, score, and
   refit with `cem_update`. Return the final mean.
7. **`mpc_action(step_fn, reward_fn, state, horizon, rng, ...)`.** Write a
   score function that turns plans `(n, H)` into actions `(n, H, 1)` with
   `plans > 0`, repeats `state` into `n` start states, calls `rollout`, and
   sums `reward_fn(traj[:, 1:])` over time. Run `cem_optimize` from mean 0
   and std 1, and return `int(plan[0] > 0)`. Common mistake: returning the
   whole plan, or a numpy boolean, instead of one int. The environment takes
   the first action only.

## Cost and scaling

The following table shows where the time goes, for $N$ transitions, state
size $d$, action size $k$, $n$ CEM samples, $I$ CEM iterations, and a
horizon $H$.

| Piece | Time | Memory |
|---|---|---|
| Fit the linear model | $O(N (d + k)^2 + (d + k)^3)$ | $O((d + k)^2)$ |
| One model step on a batch of $n$ | $O(n (d + k) d)$ | $O(n d)$ |
| One MPC decision | $O(I \, n \, H)$ model steps | $O(n H d)$ |
| An episode of $T$ steps | $O(T \, I \, n \, H)$ model steps | same |

Planning dominates. Every real step costs $I \times n \times H$ model calls,
which is $4 \times 64 \times 15 = 3{,}840$ in the exercise. That's cheap for a
linear model. With a large neural model it makes MPC slow, which is why MBPO
and Dreamer learn a policy from the model instead of planning at every step.

## Where this breaks

**Model exploitation.** The planner searches for plans the model rates
highly. Where the model is wrong in the planner's favour, the search finds
that error and exploits it. The plan looks great in imagination and fails in
reality. Ensembles help: PETS scores a plan across several models, so an
error that only one model makes gets averaged away.

**Compounding error over long horizons.** As the table shows, long rollouts
on unstable dynamics are untrustworthy. Keep planning horizons short, bootstrap
the end of a short plan with a learned value function, as TD-MPC2 does, or
branch rollouts from real states, as MBPO does.

**Distribution shift in the model's data.** A model is accurate only where it
was trained. The exercise fits CartPole near upright, and MPC keeps the pole
there. A controller that wanders into unfamiliar regions needs the model refitted on
the data it collects there, so practical systems alternate between acting and
refitting.

**Objective mismatch.** The model is trained to predict the next state, but
the agent cares about return. A model can spend its capacity on details that
don't affect control and still miss the one that does. Lambert and others
(2020) show that one-step likelihood and control performance often disagree.
The next lesson covers models trained to predict only what planning needs.

## Further reading

- [Sutton (1991), Dyna, an integrated architecture for learning, planning, and reacting](https://doi.org/10.1145/122344.122377).
  The original case for learning a model and using it to generate experience.
- [de Boer, Kroese, Mannor, and Rubinstein (2005), A tutorial on the cross-entropy method](https://doi.org/10.1007/s10479-005-5724-z).
  Where CEM comes from, and why refitting to elites works as optimisation.
- [Nagabandi, Kahn, Fearing, and Levine (2018), Neural network dynamics for model-based deep reinforcement learning with model-free fine-tuning](https://arxiv.org/abs/1708.02596).
  A neural dynamics model with random-shooting MPC for locomotion, then used
  to initialise a model-free learner.
- [Chua, Calandra, McAllister, and Levine (2018), Deep reinforcement learning in a handful of trials using probabilistic dynamics models](https://arxiv.org/abs/1805.12114).
  PETS: probabilistic ensembles, CEM, and trajectory sampling.
- [Janner, Fu, Zhang, and Levine (2019), When to trust your model: model-based policy optimization](https://arxiv.org/abs/1906.08253).
  MBPO: a return bound that grows with rollout length, and short branched
  rollouts as the response.
- [Lambert, Amos, Yadan, and Calandra (2020), Objective mismatch in model-based reinforcement learning](https://arxiv.org/abs/2002.04523).
  Evidence that a better one-step model doesn't always give better control.

## The exercise

You build a model-based controller for CartPole from the ground up. Write
`fit_linear_model` and `model_step` for a linear dynamics model, `rollout` for
open-loop prediction, and `k_step_errors` to measure how the model drifts.
Then write `cem_update` and `cem_optimize` for the cross-entropy method, and
`mpc_action` to plan through a model and return the first action.

The checks verify that `fit_linear_model` recovers a known affine system
exactly and that its ridge solution zeroes the objective's gradient. They
check `rollout` against the closed form $A^k s_0 + \sum_j A^{k-1-j} B a_j$,
check `k_step_errors` against $1.01^k - 1$, and confirm that on CartPole the
learned model's 30-step error exceeds 100 times its one-step error. They
check one CEM refit against the worked example, check that `cem_optimize`
finds a quadratic's maximiser within $10^{-3}$, and run `mpc_action` for a
full 200-step episode, first with the true physics and then with a linear
model fitted to 2,000 transitions, keeping the pole under 0.15 rad.
