---
title: Robotics and embodied agents
slug: 48-robotics-and-embodied-agents
part: "Part 9 — Beyond text"
summary: Learn a bounded residual on a fixed controller, measure what domain randomisation buys, and score progress with a VLM reward.
minutes: 60
gpu: false
objectives:
  - Say how RL fine-tuning changes for each kind of robot action head, from action tokens to flow matching.
  - Train a bounded residual policy on top of a fixed, imperfect controller with REINFORCE and a leave-one-out baseline.
  - Vectorise a simulator so a batch of randomised robots steps at once, and check it against the reference physics.
  - Measure what domain randomisation costs at the nominal setting and what it saves in the worst case.
  - Compute VLM-RM's goal-baseline reward, and show how it removes a direction a policy could hack.
prereqs:
  - 27-imitation-learning
  - 37-group-relative-policy-optimization
  - 45-flow-models-and-grpo
exercise: 48-robotics-and-embodied-agents
---

# Robotics and embodied agents

A robot policy starts the way a language model does: it imitates. People
drive the robot through a task a few hundred times, and a large model learns
to map camera images and an instruction to motor commands. These
*vision-language-action* models, or VLAs, are good at the demonstrated tasks
and fragile away from them. Small mistakes carry the robot into states no
demonstration visited, and the mistakes compound.

Reinforcement learning lets the robot practise and recover. But robots change
the problem in three ways. Real practice is slow, expensive, and can break
things, so most of it happens in simulation, and the simulator is always
somewhat wrong. The pretrained policy is valuable, so you want to improve it
without destroying it. And nobody hands you a reward: "fold the towel" has no
formula.

This lesson takes each in turn. You survey how VLAs are fine-tuned, then
build the two tools that answer the first two problems on a pendulum: a
*residual* policy that corrects a fixed controller, and *domain
randomisation* that trains across many versions of the simulator. You finish
with a reward read off a vision-language model.

## What you already know

- A policy trained by imitation drifts into states its demonstrations never
  covered, and its errors compound, from Imitation learning.
- REINFORCE weights the gradient of each action's log-probability by the
  return that followed, and a baseline that ignores the action keeps it
  unbiased.
- RLOO and GRPO compare each rollout with the others from the same start,
  from Group-relative policy optimisation.
- A diffusion or flow policy has no per-step probability until you add noise,
  from Flow models and GRPO.
- A policy that maximises a flawed reward finds the flaw.

## Fine-tune a VLA with RL

The algorithm you use depends on how the VLA outputs its actions:

- **Action tokens.** Some VLAs, such as OpenVLA, write each action as discrete
  tokens, like words. They're language models, so GRPO works unchanged.
  SimpleVLA-RL (Li et al., 2025) runs GRPO on OpenVLA-OFT with a reward of 1
  for task success and 0 for failure, like the 0-or-1 rewards of Reasoning
  with verifiable rewards.
- **Flow-matching heads.** Models such as $\pi_0$ generate a chunk of actions
  by integrating a velocity field, so an action has no tractable
  log-probability. $\pi_{\text{RL}}$ (2025) fixes this the two ways from Flow
  models and GRPO: *Flow-Noise* learns a noise network so each denoising step
  is an explicit Gaussian, and *Flow-SDE* converts the ODE into an SDE. On
  LIBERO, few-shot supervised $\pi_{0.5}$ plus RL reaches 98.3 percent success,
  against 96.9 for supervised training on the full dataset.
- **Diffusion heads.** DPPO (Ren et al., 2024) runs PPO on a two-layer
  decision process: an outer layer of environment steps and, inside each, an
  inner layer of denoising steps. It often fine-tunes only the last few
  denoising steps and keeps the sampling noise above a floor so exploration
  stays alive.
- **No policy gradient at all.** Recap, used to train $\pi^*_{0.6}$ (Physical
  Intelligence, 2025), learns a value function and labels each action as
  better or worse than typical by thresholding its advantage. The policy is
  trained by plain supervised learning with the text "Advantage: positive" or
  "Advantage: negative" in its input. At test time it's asked for positive
  advantage. This works with any action head.

Robotics adds a second clock. An environment episode has its steps, and a
diffusion or flow policy has denoising steps inside each one. When a VLA
outputs a chunk of eight actions that run open-loop, the decision is the
chunk, not the single action.

## Learn a correction, not a policy

Fine-tuning a large pretrained policy with RL can wreck what it learned in
the first few noisy updates. *Residual RL* (Johannink et al., 2019; Silver et
al., 2018) leaves the base policy frozen and learns a small correction on top.
Think of a driving instructor with a second brake pedal. The learner does the
driving, and the instructor only adds a nudge.

Write $a_{\text{base}}(s)$ for the frozen controller's action in state $s$.
The residual policy samples a raw value from a Gaussian and squashes it into a
bounded correction:

$$
a = a_{\text{base}}(s) + c \tanh(z), \qquad z \sim \mathcal{N}\big(w^\top \phi(s),\ \sigma^2\big)
$$

Read it as: the base action, plus a correction of at most $c$ in either
direction. The raw value $z$ has mean $w^\top \phi(s)$, a weighted sum of a
few state features $\phi(s)$, and exploration noise $\sigma$. With $w = 0$ the
mean correction is zero, so training starts from the base policy's behaviour.
Policy Decorator (Yuan et al., 2024) uses this tanh bound, scaled by a
hyperparameter, and also mixes in more residual actions as training goes on.

In numpy, with `eps` a standard normal draw:

```python
z = phi @ w + sigma * eps
action = base_action + bound * np.tanh(z)
```

### The gradient

The squash comes after sampling, so the reward is a function of $z$ and the
log-probability you need is the Gaussian's. Its gradient with respect to $w$
is

$$
\nabla_w \log \mathcal{N}\big(z;\ w^\top \phi,\ \sigma^2\big) = \frac{z - w^\top \phi}{\sigma^2}\,\phi = \frac{\epsilon}{\sigma}\,\phi
$$

Read it as: the noise that was added, divided by $\sigma$, times the features.
A lucky nudge in some direction, followed by a good return, pulls the mean
toward that nudge.

You estimate the policy gradient from $N$ episodes of $T$ steps. For each
step, use the *reward-to-go* $G_t^i = \sum_{t' \ge t} r_{t'}^i$, the reward
collected from step $t$ to the end of episode $i$. Subtract a *leave-one-out
baseline* $b_t^i$, the mean reward-to-go of the other $N - 1$ episodes at the
same step. This is the RLOO baseline of Group-relative policy optimisation,
applied at every step:

$$
\hat g = \frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{T-1}\big(G_t^i - b_t^i\big)\,\frac{\epsilon_t^i}{\sigma}\,\phi(s_t^i), \qquad b_t^i = \frac{1}{N-1}\sum_{j \ne i} G_t^j
$$

Read it as: every step's noise is credited with the return that came after
it, compared with what the other episodes earned from the same step. Episode
$i$'s baseline uses only the other episodes, which are independent of it, so
it keeps the estimate unbiased. Adding a constant to every reward adds the
same amount to $G_t^i$ and $b_t^i$, so the estimate doesn't change.

```python
to_go = np.cumsum(rewards[:, ::-1], axis=1)[:, ::-1]
baseline = (to_go.sum(0, keepdims=True) - to_go) / (n - 1)
grad = np.sum((to_go - baseline)[:, :, None] * eps[:, :, None] * features / sigma, axis=(0, 1)) / n
```

### Worked example

A one-step task: the base action is $0.8$, the target is $1$, and the reward
is $r = -(a - 1)^2$. Take a residual with no squash, $a = 0.8 + w +
0.1\epsilon$, and one feature, $\phi = 1$. At $w = 0$, the exact gradient of
the expected reward is

$$
\frac{d}{dw}\,\mathbb{E}\big[-(0.8 + w + 0.1\epsilon - 1)^2\big]\Big|_{w=0} = -2(0.8 - 1) = 0.4
$$

The noise term drops out because its mean is zero. The REINFORCE estimate over
100,000 samples lands within $0.02$ of $0.4$, and the exercise checks exactly
that. The best residual is $w = 0.2$, which closes the base policy's gap.

> [!KEY]
> A residual policy starts exactly at the base policy's behaviour and can
> never move more than its bound away from it. RL then only has to learn the
> correction, which is smaller and safer to explore than the whole policy.

## Build a pendulum with a miscalibrated motor

The exercise uses the pendulum from `rlenvs`. The angle $\theta$ is measured
from upright, and each step of $\Delta t = 0.05$ seconds updates the angular
velocity with gravity and the motor torque $u$, clipped to $[-2, 2]$:

$$
\dot\theta_{t+1} = \dot\theta_t + \Big(\frac{3g}{2l}\sin\theta_t + \frac{3}{ml^2}\,u_t\Big)\Delta t, \qquad \theta_{t+1} = \theta_t + \dot\theta_{t+1}\,\Delta t
$$

Read it as: gravity pulls the pendulum further from upright, and the torque
pushes back. A longer pendulum falls more slowly, and a heavier one responds
less to the motor. The new velocity is clipped to $[-8, 8]$, and the reward
is $r_t = -(\theta_t^2 + 0.1\dot\theta_t^2 + 0.001u_t^2)$, with the angle
wrapped to $[-\pi, \pi)$.

You write this for a whole batch at once. A randomised robot needs many
simulated robots, each with its own parameters, and stepping them together is
what makes the training fast. GPU simulators run thousands of robots in
parallel for the same reason.

The task is to hold the pendulum upright for 100 steps from a start within
$0.3$ of the top. The base controller is a PD controller, $u = -(10\sin\theta
+ 1.5\dot\theta)$. It would be good, except the motor has a calibration
offset $d$: every command $u$ comes out as $u + d$. The simulator's best
guess is $d = 1$.

### Worked example

With $g = 10$ and $m = l = 1$, the pendulum rests where gravity and torque
cancel: $15\sin\theta + 3(-10\sin\theta + d) = 0$. So

$$
\sin\theta = \frac{d}{5}
$$

At $d = 1$ the base controller holds the pendulum $0.20$ radians off
vertical, which costs about $0.04$ per step. Over 100 steps that's about $-4$,
and the measured return of the base controller is $-4.3$. A residual whose
constant weight cancels the offset removes almost all of it.

## Randomise the simulator

A residual trained only at $d = 1$ learns to cancel exactly that offset. That's
the danger: the real motor's offset isn't known, and a policy that has
memorised the simulator's guess pushes the wrong way when the guess is wrong.

*Domain randomisation* trains across a spread of simulators instead of one.
Tobin et al. (2017) randomised rendering so a vision network trained on
synthetic images worked on real ones. Peng et al. (2018) randomised dynamics,
such as masses and friction, so a pushing policy transferred to a real arm.
Here you randomise the offset. Each batch of 64 episodes uses offsets spread
evenly over $[-1.5, 1.5]$.

The following table shows the mean return over 50 starts, with exploration
noise off, for the base controller and for residuals trained at the nominal
offset and across the range:

| Motor offset | Base | Nominal residual | Randomised residual |
|---|---|---|---|
| $-1.5$ | $-9.41$ | $-507.55$ | $-6.30$ |
| $-1.0$ | $-4.27$ | $-228.28$ | $-2.83$ |
| $-0.5$ | $-1.29$ | $-7.80$ | $-0.85$ |
| $0.0$ | $-0.30$ | $-3.49$ | $-0.29$ |
| $0.5$ | $-1.29$ | $-0.99$ | $-1.13$ |
| $1.0$ | $-4.27$ | $-0.31$ | $-3.38$ |
| $1.5$ | $-9.42$ | $-1.48$ | $-7.12$ |

Read it row by row. At the nominal offset, the nominal residual is the best
by far: $-0.31$ against the base's $-4.27$. It also helps at the other
positive offsets, where its learned push points the right way. At $d = 0$,
where the base is nearly perfect, it scores $-3.49$. At negative offsets its
push adds to the motor's error, and the pendulum falls. The randomised
residual never falls. It learns a stiffer controller, which shrinks every
offset's error a little, and gives up most of the nominal gain.

> [!KEY]
> Domain randomisation trades performance at the simulator's guess for
> robustness across the range. A policy that can't observe the parameter can
> only hedge against it. To cancel it, the policy needs information that
> reveals it, such as a history of recent states.

This is why the randomised residual beats the base only modestly, $-3.13$
against $-4.33$ averaged over the seven offsets. Its state features,
$\sin\theta$, $\dot\theta$, and a constant, say nothing about $d$.
Randomisation made it safe, not smart.

## Score progress with a vision-language model

Hand-written rewards are the bottleneck of robot RL. Foundation models offer
three substitutes: compare camera images with a text description of the goal,
ask a VLM whether the task is done, or have a language model write reward
code.

VLM-RM (Rocamonde et al., 2023) uses CLIP, which embeds images and text in one
space. The reward is high when the camera image's embedding $s$ is close to
the embedding $g$ of a goal prompt, such as "a humanoid robot kneeling". A
second, *baseline* prompt $b$, such as "a humanoid robot", describes the scene
without the goal. With all three normalised to unit length, and $L$ the line
through $b$ and $g$, the reward is

$$
R(s) = 1 - \tfrac12 \big\lVert \alpha\, \mathrm{proj}_L\, s + (1 - \alpha)\, s - g \big\rVert_2^2
$$

Read it as: move the image embedding part of the way onto the goal-baseline
line, then score how close it is to the goal. With $\alpha = 0$ the reward is
$1 - \tfrac12\lVert s - g\rVert^2 = s^\top g$, the cosine similarity, because
both are unit vectors. With $\alpha = 1$ only the component along $g - b$
counts. Everything the goal and baseline prompts share, such as the setting
or the robot's appearance, is projected away.

```python
d = (g - b) / np.linalg.norm(g - b)
proj = b + ((s - b) @ d)[..., None] * d
reward = 1 - 0.5 * np.sum((alpha * proj + (1 - alpha) * s - g) ** 2, axis=-1)
```

### Worked example

In two dimensions, let $g = (1, 0)$, $b = (0, 1)$, and the image embedding
$s = (0.6, 0.8)$. The direction of $L$ is $(1, -1)/\sqrt2$.

- With $\alpha = 0$: $s - g = (-0.4, 0.8)$, so $R = 1 - \tfrac12(0.16 + 0.64) =
  0.60$, the cosine $s^\top g$.
- With $\alpha = 1$: the projection of $s$ onto $L$ is $(0.4, 0.6)$, so
  $\mathrm{proj}_L s - g = (-0.6, 0.6)$ and $R = 1 - \tfrac12(0.36 + 0.36) = 0.64$.

The projection matters when an irrelevant direction leaks into the reward.
Suppose both prompts also evoke a floor texture, $g \propto e_1 + 1.5e_3$ and
$b \propto e_2 + 1.5e_3$, where $e_1$ is the kneeling pose, $e_2$ standing, and
$e_3$ the texture. Kneeling on a plain floor, $s = e_1$, scores $0.555$ at
$\alpha = 0$. Standing on the textured floor scores $0.692$. A policy would
learn to find the texture. At $\alpha = 1$, the texture direction is
orthogonal to $g - b$ and is projected out: kneeling scores $0.950$ and
standing $0.692$.

Two related ideas complete the picture. RoboCLIP (Sontakke et al., 2023)
scores a video of the whole episode against a single demonstration, a video
or a sentence, with a pretrained video-language model. Eureka (Ma et al.,
2023) has GPT-4 write reward code, trains a policy with each candidate, and
feeds the training statistics back for the next round. It beats
expert-written rewards on 83 percent of 29 environments.

> [!RECAP]
> - A residual policy adds $c \tanh(z)$ to a frozen base action, so it starts
>   at the base behaviour and never strays more than $c$.
> - Its score is $\epsilon \phi / \sigma$, and the gradient estimate weights
>   each step by reward-to-go minus a leave-one-out baseline.
> - The batched pendulum must match `rlenvs.Pendulum` exactly, including both
>   clips.
> - Randomising the offset gives up nominal return and prevents the falls a
>   nominal-only policy suffers.
> - VLM-RM's projection onto the goal-baseline line removes directions the two
>   prompts share, which a policy could otherwise exploit.

## Build it step by step

The checks test the functions in this order.

1. **`pendulum_step(theta, theta_dot, torque, m, l)`.** Takes `(N,)` arrays
   and returns the new angle, velocity, and reward. Read the constants from
   `rlenvs.Pendulum`, clip the torque with `np.clip`, compute the reward from
   the *current* state and the clipped torque, then update. The common
   mistake is updating the angle with the old velocity, which isn't what
   `rlenvs.Pendulum` does.
2. **`residual_action(base_action, raw, bound)`.** One line with `np.tanh`.
3. **`reinforce_gradient(rewards, features, eps, sigma)`.** Reward-to-go with a
   reversed `np.cumsum`, the leave-one-out mean for every step at once, then a
   sum over episodes and steps. The common mistake is including episode $i$ in
   its own baseline, which biases the estimate.
4. **`rollout(w, offsets, theta0, theta_dot0, eps, sigma, bound)`.** Loop over
   the $T$ columns of `eps`. Record the features before stepping, and add the
   offset to the command after the residual, since the motor adds it.
5. **`clip_reward(state_emb, goal_emb, baseline_emb, alpha)`.** Normalise with
   `np.linalg.norm(..., axis=-1, keepdims=True)`, project with `@`, and sum
   the squared error over the last axis. The common mistake is projecting onto
   the direction through the origin instead of the line through $b$ and $g$.
6. **`train_residual(offset_low, offset_high, seed, ...)`.** Start from
   `w = np.zeros(3)` and an empty Adam state, and repeat: spread the offsets,
   draw starts and noise, roll out, estimate the gradient, and call
   `adam_step`, which ascends. The checks train twice, at offset 1 and over
   $[-1.5, 1.5]$, and evaluate both on `rlenvs.Pendulum` itself.

## Cost and scaling

A robot's cost is its samples. The following table compares where the
samples come from:

| Source | Cost per sample | What limits it |
|---|---|---|
| Real robot | Seconds of wall-clock time, plus resets and wear | Safety and supervision |
| CPU simulator, one robot | Microseconds for a pendulum, milliseconds for an arm | Python overhead per step |
| Batched simulator | Nearly the cost of one step, for the whole batch | Memory, and the gap to reality |
| VLM reward | One image-encoder pass per scored frame | Encoder size, so batch the frames |

In the exercise, the batched pendulum steps 64 robots in about twice the time
`rlenvs.Pendulum` takes to step one, and both training runs finish in under a
second. The residual has three weights, so its update is negligible. A
residual on a real VLA is a small network next to a frozen model with billions
of weights, which is the point: the expensive model only runs forward.

## Where this breaks

**The residual's noise swamps the base.** If $\sigma$ and the bound are large,
the first rollouts are worse than the base, and the returns carry little
signal. Start with a small bound, or mix in residual actions gradually, as
Policy Decorator does.

**Randomising too wide.** A range that includes simulators no policy can
handle, such as offsets beyond the motor's torque, fills the batch with
failures. The gradient then chases them. Widen the range as the policy
improves.

**Randomising the wrong thing.** Randomisation only protects against the
parameters you vary. A real robot's error is often something the simulator
doesn't model at all, such as cable drag or latency.

**Dense VLM rewards on unreliable frames.** A VLM can judge a finished task
far better than a half-finished one. Scoring every frame rewards the
mid-task poses it misjudges. That's one reason RoboCLIP scores whole episodes.

**Privileged reward code.** An LLM-written reward can read simulator state,
such as exact object poses, that doesn't exist on the real robot. The policy
trains fine and can't be deployed.

## Further reading

- [Johannink et al., Residual Reinforcement Learning for Robot Control
  (2019)](https://arxiv.org/abs/1812.03201) adds an RL correction to a
  conventional controller for contact-rich assembly.
- [Yuan et al., Policy Decorator (2024)](https://arxiv.org/abs/2412.13630)
  refines large imitation policies with a bounded residual and a progressive
  exploration schedule.
- [Peng et al., Sim-to-Real Transfer of Robotic Control with Dynamics
  Randomization (2018)](https://arxiv.org/abs/1710.06537) randomises physics so
  a simulated pushing policy works on a real arm.
- [Ren et al., Diffusion Policy Policy Optimization
  (2024)](https://arxiv.org/abs/2409.00588) fine-tunes diffusion policies with
  PPO on the two-layer decision process.
- [$\pi_{\text{RL}}$ (2025)](https://arxiv.org/abs/2510.25889) brings online RL
  to flow-matching VLAs with Flow-Noise and Flow-SDE.
- [Physical Intelligence, $\pi^*_{0.6}$ and Recap
  (2025)](https://arxiv.org/abs/2511.14759) trains a VLA with
  advantage-conditioned supervised learning on real-robot experience.
- [Rocamonde et al., Vision-Language Models are Zero-Shot Reward Models
  (2023)](https://arxiv.org/abs/2310.12921) defines the goal-baseline CLIP
  reward and finds larger CLIP models give better rewards.
- [Ma et al., Eureka (2023)](https://arxiv.org/abs/2310.12931) evolves reward
  code written by a language model.

Open questions that researchers are working on:

- Can a learned video world model replace the simulator for RL, with rewards
  that stay trustworthy inside it?
- How should credit be split between the environment steps and the denoising
  steps of a diffusion or flow policy?
- Which VLM progress rewards stay reliable when the policy optimises hard
  against them?

## The exercise

You write `pendulum_step`, `residual_action`, `reinforce_gradient`, `rollout`,
`clip_reward`, and `train_residual`. The robot is given at the top of the
file: `base_controller`, `residual_features`, `start_states`, and
`adam_step`, with `HORIZON` and `NOMINAL_OFFSET`.

The checks compare your batched `pendulum_step` with `rlenvs.Pendulum` for
nominal and randomised masses and lengths, including both clips. They check
the residual's bound, the gradient estimate against a hand computation and the
worked example's $0.4$, its invariance to a shift in reward, and that a
zero-weight rollout reproduces the base controller. They check the CLIP
reward's $0.60$ and $0.64$ and the texture hack. The last two train residuals
and evaluate them on `rlenvs.Pendulum`: the nominal residual must beat the
base at the nominal offset, and the randomised one must have a worst case
over 100 better while losing to the nominal one at offset 1.
