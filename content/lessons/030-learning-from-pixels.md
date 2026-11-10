---
title: Learning from pixels
slug: 030-learning-from-pixels
part: "Part 4 — Deep value methods"
summary: Turn game frames into inputs, add noisy layers, see how Rainbow combines the DQN fixes, and shift images at random to regularize.
minutes: 45
gpu: false
objectives:
  - Preprocess two game frames into one grayscale frame and stack the last k frames into an observation.
  - Implement a NoisyLinear layer with factorized Gaussian noise and derive its gradients for the means and the noise scales.
  - Name the six components Rainbow combines and read its ablation without overstating it.
  - Apply a DrQ-style random shift to the observations in both Q and the target.
  - "Say when augmentation helps and when it does not: it needs shifts that keep the best action the same."
prereqs:
  - 026-deep-q-networks
  - 027-double-and-dueling-dqn
  - 028-prioritized-replay
  - 029-distributional-rl
exercise: 030-learning-from-pixels
---

# Learning from pixels

Every environment so far gave the agent a short vector: a cart's position and velocity, or a grid cell index. A person playing a video game gets none of that. They get a picture, 60 times a second, and they work out where the ball is and which way it moves. To learn from pictures, an agent has to turn a pile of pixels into something a Q-network can use, and that raises three problems this chapter solves.

First, a single frame hides motion, and it can hide objects that flicker. Second, a DQN explores with random actions that have nothing to do with what the network knows, and the six fixes in the last three chapters leave that alone. Third, a network with thousands of inputs can memorize its training frames. Augmenting the images attacks that at almost no cost.

You build all three pieces, then train a small agent on a pixel game called Catch.

## What you already know

- DQN regresses $Q_\theta(s, a)$ toward $r_t + \gamma \max_{a'} Q_{\bar\theta}(s_{t+1}, a')$ from a replay buffer ([Deep Q-networks](/lessons/026-deep-q-networks)).
- Double, dueling, and n-step targets each fix one weakness of that target ([Double and dueling DQN](/lessons/027-double-and-dueling-dqn)).
- Prioritized replay samples surprising transitions more often and corrects the bias with importance weights ([Prioritized experience replay](/lessons/028-prioritized-replay)).
- C51 learns a distribution of returns and trains with a cross-entropy loss ([Distributional reinforcement learning](/lessons/029-distributional-rl)).
- A linear layer computes $y = Wx + b$, and its gradients with respect to $W$ and $b$ are outer products you can check by finite differences.

## From screen to observation

Mnih et al. (2015) fed their DQN a processed screen, and most Atari agents since use the same recipe. It has four steps.

- **Max over two frames.** Some Atari sprites show only on every other frame. Taking the pixelwise maximum of two consecutive frames keeps them. Take the maximum on the RGB values, before you convert to gray.
- **Grayscale.** Color rarely matters for the action, and one channel costs a third of three. Luminance is $0.299R + 0.587G + 0.114B$.
- **Scale.** Divide by 255, so every input lies in $[0, 1]$.
- **Stack $k$ frames.** One frame shows where the ball is, not where it is going. Stacking the last $k$ frames, usually 4, lets the network read velocity from the differences.

Work one pixel by hand. Frame A has a pure red pixel $(255, 0, 0)$ and frame B has a pure green pixel $(0, 255, 0)$ in the same place. The RGB maximum is $(255, 255, 0)$, and its luminance is $(0.299 + 0.587) \cdot 255 = 226$, which scales to $0.886$. Converting to gray first gives $\max(0.299, 0.587) = 0.587$ instead, which is the most common bug in this step.

A stack of 3 from a history of only two frames, $f_0$ and $f_1$, is $[f_0, f_0, f_1]$: at the start of an episode you repeat the oldest frame, so the observation always has $k$ channels.

```python
merged = np.maximum(frame_a, frame_b)          # (H, W, 3), on RGB
gray = merged @ np.array([0.299, 0.587, 0.114]) / 255.0
stack = np.stack(([frames[0]] * (k - len(recent))) + recent)
```

## Explore with noise in the weights

DQN explores with epsilon-greedy: with probability $\epsilon$, it picks a random action, whatever the state. Fortunato et al. (2018) put the randomness inside the network instead. Each weight becomes a mean plus a learned amount of noise:

$$
y = (\mu^w + \sigma^w \odot \varepsilon^w)\, x + \mu^b + \sigma^b \odot \varepsilon^b
$$

Here $\mu$ are the usual weights, $\sigma$ are learned noise scales of the same shape, $\odot$ multiplies elementwise, and $\varepsilon$ is noise drawn once and held fixed while you act. Gradient descent can shrink a $\sigma$ toward zero where exploring does not pay, which epsilon-greedy cannot do, because $\epsilon$ is one number set in advance. One noise draw also shifts the action values the same way across states, so the agent tries a coherent alternative instead of a random twitch.

### Factorized noise

Independent noise needs $n_\text{out} \times n_\text{in}$ random numbers per draw. The factorized version draws $n_\text{in} + n_\text{out}$ and combines them with an outer product. Define $f(u) = \operatorname{sign}(u)\sqrt{|u|}$, draw $u_\text{in}$ and $u_\text{out}$ from a standard normal, and set

$$
\varepsilon^w = f(u_\text{out})\, f(u_\text{in})^\top, \qquad \varepsilon^b = f(u_\text{out})
$$

The transform $f$ keeps the sign of each draw and shrinks large ones, and it is the transform Fortunato et al. specify. What matters for the code is that $\varepsilon^w$ has rank one.

Work a layer with two inputs and one output. Take $u_\text{in} = (4, -0.25)$ and $u_\text{out} = (9)$, so $f(u_\text{in}) = (2, -0.5)$ and $f(u_\text{out}) = (3)$. Then $\varepsilon^w = [6, -1.5]$ and $\varepsilon^b = 3$. With $\mu^w = [1, 2]$, $\sigma^w = [0.1, 0.1]$, $\mu^b = 0$, $\sigma^b = 0.1$, and $x = (1, 1)$:

- The noisy weights are $[1 + 0.6,\ 2 - 0.15] = [1.6, 1.85]$ and the noisy bias is $0.3$.
- The output is $1.6 + 1.85 + 0.3 = 3.75$, against $3$ with the noise off.

### The gradient

Let $L$ be a loss and $g = \partial L / \partial y$, one row per example. Because $y$ is linear in $\mu$ and in $\sigma$ once $\varepsilon$ is fixed, the gradients look like those of a linear layer, with the noise as an extra factor on the $\sigma$ terms:

$$
\frac{\partial L}{\partial \mu^w} = g^\top x, \qquad
\frac{\partial L}{\partial \sigma^w} = (g^\top x) \odot \varepsilon^w, \qquad
\frac{\partial L}{\partial \mu^b} = \textstyle\sum g, \qquad
\frac{\partial L}{\partial \sigma^b} = \big(\textstyle\sum g\big) \odot \varepsilon^b
$$

Read it as: a $\sigma$ gets the gradient of the matching $\mu$, scaled by how much noise it multiplied on this draw. In the layer above with $g = 1$, $\partial L / \partial \sigma^w = [6, -1.5]$. You do not write these by hand. Build the weights from `weight_mu + weight_sigma * eps_w` with torch operations, and autograd produces them. The exercise checks the result against finite differences.

> [!KEY]
> A noisy layer is a linear layer with a learned noise scale on every weight. Gradient descent can lower the noise where exploration stops paying, which epsilon-greedy cannot do.

## Rainbow: all the pieces at once

Each of the last three chapters changed one part of DQN. Hessel et al. (2018) asked what happens when you use them together, and called the result Rainbow. The following table lists the six parts and where you met each one.

| Component | What it changes | Where |
|---|---|---|
| Double Q-learning | The online network picks the next action and the target network scores it | [026](/lessons/027-double-and-dueling-dqn) |
| Dueling heads | The network outputs a state value and advantages | [026](/lessons/027-double-and-dueling-dqn) |
| Multi-step targets | The target uses $n$ rewards before bootstrapping (Rainbow uses $n = 3$) | [026](/lessons/027-double-and-dueling-dqn) |
| Prioritized replay | Transitions are sampled in proportion to their loss, with importance weights | [027](/lessons/028-prioritized-replay) |
| Distributional output | The network predicts a distribution, trained with C51's projected cross-entropy | [028](/lessons/029-distributional-rl) |
| Noisy layers | The output layers are NoisyLinear, and there is no epsilon-greedy | This chapter |

The parts fit because each touches a different place. The loss is the cross-entropy between the projected $n$-step target and the prediction, the online network picks the bootstrap action, and the same loss sets each transition's priority.

The paper's ablation removes one component at a time. Read it with care, because it is one benchmark, 57 Atari games:

- Removing prioritized replay or multi-step targets hurt the median score most. Multi-step also lowered the final score, not only the early learning.
- Removing the distributional output matched the full agent for about 40 million frames, then fell behind.
- Replacing noisy layers with epsilon-greedy lowered the median, with large drops in some games and small gains in others.
- Removing dueling or double Q-learning made no significant difference to the median. The authors suggest that clipped rewards may limit overestimation, so double Q-learning might matter more with other reward scales.

The honest reading is that a median hides single games, that no median difference is not the same as useless, and that this is an Atari result with clipped rewards. Test the parts on your own task.

## Shift the pixels: DrQ

A network that sees 84 by 84 pixels can memorize: it can store the right answer for one exact frame without learning what a ball is. Image augmentation, an idea from supervised learning, shows the network the same situation in slightly different pictures, so it has to learn what stays the same.

Kostrikov, Yarats, and Fergus (2020) use the most basic version in DrQ: a random shift. Pad the image by a few pixels, copying the edge pixels outward, then crop a window of the original size at a random offset. Laskin et al. (2020) compared several augmentations in RAD and found that cropping and translating the image worked well on the control tasks they tried. Figure 1 shows `random_shift` on a 5 by 5 image.

```figure
<svg viewBox="0 0 620 160" role="img" aria-label="A 5 by 5 image and three shifted copies, with the left edge column replicated when the window moves left">
<text x="20" y="20" class="fig-small">original</text>
<rect x="20" y="30" width="21" height="21" class="fill-sunk"/>
<rect x="42" y="30" width="21" height="21" class="fill-sunk"/>
<rect x="64" y="30" width="21" height="21" class="fill-sunk"/>
<rect x="86" y="30" width="21" height="21" class="fill-sunk"/>
<rect x="108" y="30" width="21" height="21" class="fill-sunk"/>
<rect x="20" y="52" width="21" height="21" class="fill-a"/>
<rect x="42" y="52" width="21" height="21" class="fill-sunk"/>
<rect x="64" y="52" width="21" height="21" class="fill-sunk"/>
<rect x="86" y="52" width="21" height="21" class="fill-sunk"/>
<rect x="108" y="52" width="21" height="21" class="fill-sunk"/>
<rect x="20" y="74" width="21" height="21" class="fill-a"/>
<rect x="42" y="74" width="21" height="21" class="fill-sunk"/>
<rect x="64" y="74" width="21" height="21" class="fill-sunk"/>
<rect x="86" y="74" width="21" height="21" class="fill-sunk"/>
<rect x="108" y="74" width="21" height="21" class="fill-sunk"/>
<rect x="20" y="96" width="21" height="21" class="fill-a"/>
<rect x="42" y="96" width="21" height="21" class="fill-a"/>
<rect x="64" y="96" width="21" height="21" class="fill-sunk"/>
<rect x="86" y="96" width="21" height="21" class="fill-sunk"/>
<rect x="108" y="96" width="21" height="21" class="fill-sunk"/>
<rect x="20" y="118" width="21" height="21" class="fill-sunk"/>
<rect x="42" y="118" width="21" height="21" class="fill-sunk"/>
<rect x="64" y="118" width="21" height="21" class="fill-sunk"/>
<rect x="86" y="118" width="21" height="21" class="fill-sunk"/>
<rect x="108" y="118" width="21" height="21" class="fill-sunk"/>
<text x="170" y="20" class="fig-small">dy=+0, dx=-1</text>
<rect x="170" y="30" width="21" height="21" class="fill-sunk"/>
<rect x="192" y="30" width="21" height="21" class="fill-sunk"/>
<rect x="214" y="30" width="21" height="21" class="fill-sunk"/>
<rect x="236" y="30" width="21" height="21" class="fill-sunk"/>
<rect x="258" y="30" width="21" height="21" class="fill-sunk"/>
<rect x="170" y="52" width="21" height="21" class="fill-a"/>
<rect x="192" y="52" width="21" height="21" class="fill-a"/>
<rect x="214" y="52" width="21" height="21" class="fill-sunk"/>
<rect x="236" y="52" width="21" height="21" class="fill-sunk"/>
<rect x="258" y="52" width="21" height="21" class="fill-sunk"/>
<rect x="170" y="74" width="21" height="21" class="fill-a"/>
<rect x="192" y="74" width="21" height="21" class="fill-a"/>
<rect x="214" y="74" width="21" height="21" class="fill-sunk"/>
<rect x="236" y="74" width="21" height="21" class="fill-sunk"/>
<rect x="258" y="74" width="21" height="21" class="fill-sunk"/>
<rect x="170" y="96" width="21" height="21" class="fill-a"/>
<rect x="192" y="96" width="21" height="21" class="fill-a"/>
<rect x="214" y="96" width="21" height="21" class="fill-a"/>
<rect x="236" y="96" width="21" height="21" class="fill-sunk"/>
<rect x="258" y="96" width="21" height="21" class="fill-sunk"/>
<rect x="170" y="118" width="21" height="21" class="fill-sunk"/>
<rect x="192" y="118" width="21" height="21" class="fill-sunk"/>
<rect x="214" y="118" width="21" height="21" class="fill-sunk"/>
<rect x="236" y="118" width="21" height="21" class="fill-sunk"/>
<rect x="258" y="118" width="21" height="21" class="fill-sunk"/>
<text x="320" y="20" class="fig-small">dy=+0, dx=+1</text>
<rect x="320" y="30" width="21" height="21" class="fill-sunk"/>
<rect x="342" y="30" width="21" height="21" class="fill-sunk"/>
<rect x="364" y="30" width="21" height="21" class="fill-sunk"/>
<rect x="386" y="30" width="21" height="21" class="fill-sunk"/>
<rect x="408" y="30" width="21" height="21" class="fill-sunk"/>
<rect x="320" y="52" width="21" height="21" class="fill-sunk"/>
<rect x="342" y="52" width="21" height="21" class="fill-sunk"/>
<rect x="364" y="52" width="21" height="21" class="fill-sunk"/>
<rect x="386" y="52" width="21" height="21" class="fill-sunk"/>
<rect x="408" y="52" width="21" height="21" class="fill-sunk"/>
<rect x="320" y="74" width="21" height="21" class="fill-sunk"/>
<rect x="342" y="74" width="21" height="21" class="fill-sunk"/>
<rect x="364" y="74" width="21" height="21" class="fill-sunk"/>
<rect x="386" y="74" width="21" height="21" class="fill-sunk"/>
<rect x="408" y="74" width="21" height="21" class="fill-sunk"/>
<rect x="320" y="96" width="21" height="21" class="fill-a"/>
<rect x="342" y="96" width="21" height="21" class="fill-sunk"/>
<rect x="364" y="96" width="21" height="21" class="fill-sunk"/>
<rect x="386" y="96" width="21" height="21" class="fill-sunk"/>
<rect x="408" y="96" width="21" height="21" class="fill-sunk"/>
<rect x="320" y="118" width="21" height="21" class="fill-sunk"/>
<rect x="342" y="118" width="21" height="21" class="fill-sunk"/>
<rect x="364" y="118" width="21" height="21" class="fill-sunk"/>
<rect x="386" y="118" width="21" height="21" class="fill-sunk"/>
<rect x="408" y="118" width="21" height="21" class="fill-sunk"/>
<text x="470" y="20" class="fig-small">dy=-1, dx=+1</text>
<rect x="470" y="30" width="21" height="21" class="fill-sunk"/>
<rect x="492" y="30" width="21" height="21" class="fill-sunk"/>
<rect x="514" y="30" width="21" height="21" class="fill-sunk"/>
<rect x="536" y="30" width="21" height="21" class="fill-sunk"/>
<rect x="558" y="30" width="21" height="21" class="fill-sunk"/>
<rect x="470" y="52" width="21" height="21" class="fill-sunk"/>
<rect x="492" y="52" width="21" height="21" class="fill-sunk"/>
<rect x="514" y="52" width="21" height="21" class="fill-sunk"/>
<rect x="536" y="52" width="21" height="21" class="fill-sunk"/>
<rect x="558" y="52" width="21" height="21" class="fill-sunk"/>
<rect x="470" y="74" width="21" height="21" class="fill-sunk"/>
<rect x="492" y="74" width="21" height="21" class="fill-sunk"/>
<rect x="514" y="74" width="21" height="21" class="fill-sunk"/>
<rect x="536" y="74" width="21" height="21" class="fill-sunk"/>
<rect x="558" y="74" width="21" height="21" class="fill-sunk"/>
<rect x="470" y="96" width="21" height="21" class="fill-sunk"/>
<rect x="492" y="96" width="21" height="21" class="fill-sunk"/>
<rect x="514" y="96" width="21" height="21" class="fill-sunk"/>
<rect x="536" y="96" width="21" height="21" class="fill-sunk"/>
<rect x="558" y="96" width="21" height="21" class="fill-sunk"/>
<rect x="470" y="118" width="21" height="21" class="fill-a"/>
<rect x="492" y="118" width="21" height="21" class="fill-sunk"/>
<rect x="514" y="118" width="21" height="21" class="fill-sunk"/>
<rect x="536" y="118" width="21" height="21" class="fill-sunk"/>
<rect x="558" y="118" width="21" height="21" class="fill-sunk"/>
</svg>
Figure 1. random_shift with pad=1 pads the image by copying its edge cells, then crops a 5 x 5 window. Moving the window left (dx=-1) duplicates the bright left column, and moving it right (dx=+1) drops that column and leaves a blank one.
```

DrQ applies the shift in two places in the TD loss, not one:

- **In the target.** Shift the next observation $K$ times, compute $\max_{a'} Q_{\bar\theta}$ on each, and average. The target $y$ is then less noisy, and it carries no gradient.
- **In Q.** Shift the observation $M$ times, and average the Huber loss between $Q_\theta(\text{shifted } s, a)$ and $y$ over the shifts.

$$
y = r_t + \gamma (1 - d_t)\, \frac{1}{K} \sum_{k=1}^{K} \max_{a'} Q_{\bar\theta}\big(\text{shift}_k(s_{t+1}), a'\big), \qquad
\mathcal{L} = \frac{1}{M} \sum_{m=1}^{M} \ell\big(Q_\theta(\text{shift}_m(s_t), a_t) - y\big)
$$

Here $d_t$ is 1 when $s_{t+1}$ is terminal and $\ell$ is the Huber loss. With $K = M = 1$ and a shift of zero this is the ordinary DQN loss.

Augmentation is the cheapest sample-efficiency gain on offer: it costs no environment steps and no extra network, and the DrQ and RAD papers report large gains on pixel control tasks.

### Shifts must keep the best action

The gain has a condition: the shifted picture must still call for the same action. In Catch, shifting everything by one pixel keeps the ball and paddle in the same relative position, so the right move is the same, as long as nothing leaves the picture. Figure 2 shows that edge. The exercise's Catch frame carries a blank one-pixel margin, so a shift only moves objects around a blank canvas.

```figure
<svg viewBox="0 0 620 190" role="img" aria-label="A Catch frame shifted down one row; without a blank margin the paddle leaves the image, with a margin it stays">
<text x="15" y="16" class="fig-small">no margin</text>
<rect x="15" y="24" width="13" height="13" class="fill-sunk"/>
<rect x="29" y="24" width="13" height="13" class="fill-sunk"/>
<rect x="43" y="24" width="13" height="13" class="fill-sunk"/>
<rect x="57" y="24" width="13" height="13" class="fill-sunk"/>
<rect x="71" y="24" width="13" height="13" class="fill-sunk"/>
<rect x="15" y="38" width="13" height="13" class="fill-sunk"/>
<rect x="29" y="38" width="13" height="13" class="fill-sunk"/>
<rect x="43" y="38" width="13" height="13" class="fill-sunk"/>
<rect x="57" y="38" width="13" height="13" class="fill-sunk"/>
<rect x="71" y="38" width="13" height="13" class="fill-sunk"/>
<rect x="15" y="52" width="13" height="13" class="fill-sunk"/>
<rect x="29" y="52" width="13" height="13" class="fill-sunk"/>
<rect x="43" y="52" width="13" height="13" class="fill-sunk"/>
<rect x="57" y="52" width="13" height="13" class="fill-sunk"/>
<rect x="71" y="52" width="13" height="13" class="fill-sunk"/>
<rect x="15" y="66" width="13" height="13" class="fill-sunk"/>
<rect x="29" y="66" width="13" height="13" class="fill-sunk"/>
<rect x="43" y="66" width="13" height="13" class="fill-sunk"/>
<rect x="57" y="66" width="13" height="13" class="fill-sunk"/>
<rect x="71" y="66" width="13" height="13" class="fill-sunk"/>
<rect x="15" y="80" width="13" height="13" class="fill-sunk"/>
<rect x="29" y="80" width="13" height="13" class="fill-b"/>
<rect x="43" y="80" width="13" height="13" class="fill-sunk"/>
<rect x="57" y="80" width="13" height="13" class="fill-sunk"/>
<rect x="71" y="80" width="13" height="13" class="fill-sunk"/>
<rect x="15" y="94" width="13" height="13" class="fill-sunk"/>
<rect x="29" y="94" width="13" height="13" class="fill-sunk"/>
<rect x="43" y="94" width="13" height="13" class="fill-sunk"/>
<rect x="57" y="94" width="13" height="13" class="fill-sunk"/>
<rect x="71" y="94" width="13" height="13" class="fill-sunk"/>
<rect x="15" y="108" width="13" height="13" class="fill-sunk"/>
<rect x="29" y="108" width="13" height="13" class="fill-sunk"/>
<rect x="43" y="108" width="13" height="13" class="fill-sunk"/>
<rect x="57" y="108" width="13" height="13" class="fill-sunk"/>
<rect x="71" y="108" width="13" height="13" class="fill-sunk"/>
<rect x="15" y="122" width="13" height="13" class="fill-sunk"/>
<rect x="29" y="122" width="13" height="13" class="fill-sunk"/>
<rect x="43" y="122" width="13" height="13" class="fill-sunk"/>
<rect x="57" y="122" width="13" height="13" class="fill-sunk"/>
<rect x="71" y="122" width="13" height="13" class="fill-sunk"/>
<rect x="15" y="136" width="13" height="13" class="fill-sunk"/>
<rect x="29" y="136" width="13" height="13" class="fill-sunk"/>
<rect x="43" y="136" width="13" height="13" class="fill-sunk"/>
<rect x="57" y="136" width="13" height="13" class="fill-sunk"/>
<rect x="71" y="136" width="13" height="13" class="fill-sunk"/>
<rect x="15" y="150" width="13" height="13" class="fill-sunk"/>
<rect x="29" y="150" width="13" height="13" class="fill-a"/>
<rect x="43" y="150" width="13" height="13" class="fill-sunk"/>
<rect x="57" y="150" width="13" height="13" class="fill-sunk"/>
<rect x="71" y="150" width="13" height="13" class="fill-sunk"/>
<text x="165" y="16" class="fig-small">no margin, shifted down</text>
<rect x="165" y="24" width="13" height="13" class="fill-sunk"/>
<rect x="179" y="24" width="13" height="13" class="fill-sunk"/>
<rect x="193" y="24" width="13" height="13" class="fill-sunk"/>
<rect x="207" y="24" width="13" height="13" class="fill-sunk"/>
<rect x="221" y="24" width="13" height="13" class="fill-sunk"/>
<rect x="165" y="38" width="13" height="13" class="fill-sunk"/>
<rect x="179" y="38" width="13" height="13" class="fill-sunk"/>
<rect x="193" y="38" width="13" height="13" class="fill-sunk"/>
<rect x="207" y="38" width="13" height="13" class="fill-sunk"/>
<rect x="221" y="38" width="13" height="13" class="fill-sunk"/>
<rect x="165" y="52" width="13" height="13" class="fill-sunk"/>
<rect x="179" y="52" width="13" height="13" class="fill-sunk"/>
<rect x="193" y="52" width="13" height="13" class="fill-sunk"/>
<rect x="207" y="52" width="13" height="13" class="fill-sunk"/>
<rect x="221" y="52" width="13" height="13" class="fill-sunk"/>
<rect x="165" y="66" width="13" height="13" class="fill-sunk"/>
<rect x="179" y="66" width="13" height="13" class="fill-sunk"/>
<rect x="193" y="66" width="13" height="13" class="fill-sunk"/>
<rect x="207" y="66" width="13" height="13" class="fill-sunk"/>
<rect x="221" y="66" width="13" height="13" class="fill-sunk"/>
<rect x="165" y="80" width="13" height="13" class="fill-sunk"/>
<rect x="179" y="80" width="13" height="13" class="fill-sunk"/>
<rect x="193" y="80" width="13" height="13" class="fill-sunk"/>
<rect x="207" y="80" width="13" height="13" class="fill-sunk"/>
<rect x="221" y="80" width="13" height="13" class="fill-sunk"/>
<rect x="165" y="94" width="13" height="13" class="fill-sunk"/>
<rect x="179" y="94" width="13" height="13" class="fill-b"/>
<rect x="193" y="94" width="13" height="13" class="fill-sunk"/>
<rect x="207" y="94" width="13" height="13" class="fill-sunk"/>
<rect x="221" y="94" width="13" height="13" class="fill-sunk"/>
<rect x="165" y="108" width="13" height="13" class="fill-sunk"/>
<rect x="179" y="108" width="13" height="13" class="fill-sunk"/>
<rect x="193" y="108" width="13" height="13" class="fill-sunk"/>
<rect x="207" y="108" width="13" height="13" class="fill-sunk"/>
<rect x="221" y="108" width="13" height="13" class="fill-sunk"/>
<rect x="165" y="122" width="13" height="13" class="fill-sunk"/>
<rect x="179" y="122" width="13" height="13" class="fill-sunk"/>
<rect x="193" y="122" width="13" height="13" class="fill-sunk"/>
<rect x="207" y="122" width="13" height="13" class="fill-sunk"/>
<rect x="221" y="122" width="13" height="13" class="fill-sunk"/>
<rect x="165" y="136" width="13" height="13" class="fill-sunk"/>
<rect x="179" y="136" width="13" height="13" class="fill-sunk"/>
<rect x="193" y="136" width="13" height="13" class="fill-sunk"/>
<rect x="207" y="136" width="13" height="13" class="fill-sunk"/>
<rect x="221" y="136" width="13" height="13" class="fill-sunk"/>
<rect x="165" y="150" width="13" height="13" class="fill-sunk"/>
<rect x="179" y="150" width="13" height="13" class="fill-sunk"/>
<rect x="193" y="150" width="13" height="13" class="fill-sunk"/>
<rect x="207" y="150" width="13" height="13" class="fill-sunk"/>
<rect x="221" y="150" width="13" height="13" class="fill-sunk"/>
<text x="315" y="16" class="fig-small">blank margin</text>
<rect x="315" y="24" width="13" height="13" class="fill-sunk"/>
<rect x="329" y="24" width="13" height="13" class="fill-sunk"/>
<rect x="343" y="24" width="13" height="13" class="fill-sunk"/>
<rect x="357" y="24" width="13" height="13" class="fill-sunk"/>
<rect x="371" y="24" width="13" height="13" class="fill-sunk"/>
<rect x="385" y="24" width="13" height="13" class="fill-sunk"/>
<rect x="399" y="24" width="13" height="13" class="fill-sunk"/>
<rect x="315" y="38" width="13" height="13" class="fill-sunk"/>
<rect x="329" y="38" width="13" height="13" class="fill-sunk"/>
<rect x="343" y="38" width="13" height="13" class="fill-sunk"/>
<rect x="357" y="38" width="13" height="13" class="fill-sunk"/>
<rect x="371" y="38" width="13" height="13" class="fill-sunk"/>
<rect x="385" y="38" width="13" height="13" class="fill-sunk"/>
<rect x="399" y="38" width="13" height="13" class="fill-sunk"/>
<rect x="315" y="52" width="13" height="13" class="fill-sunk"/>
<rect x="329" y="52" width="13" height="13" class="fill-sunk"/>
<rect x="343" y="52" width="13" height="13" class="fill-sunk"/>
<rect x="357" y="52" width="13" height="13" class="fill-sunk"/>
<rect x="371" y="52" width="13" height="13" class="fill-sunk"/>
<rect x="385" y="52" width="13" height="13" class="fill-sunk"/>
<rect x="399" y="52" width="13" height="13" class="fill-sunk"/>
<rect x="315" y="66" width="13" height="13" class="fill-sunk"/>
<rect x="329" y="66" width="13" height="13" class="fill-sunk"/>
<rect x="343" y="66" width="13" height="13" class="fill-sunk"/>
<rect x="357" y="66" width="13" height="13" class="fill-sunk"/>
<rect x="371" y="66" width="13" height="13" class="fill-sunk"/>
<rect x="385" y="66" width="13" height="13" class="fill-sunk"/>
<rect x="399" y="66" width="13" height="13" class="fill-sunk"/>
<rect x="315" y="80" width="13" height="13" class="fill-sunk"/>
<rect x="329" y="80" width="13" height="13" class="fill-sunk"/>
<rect x="343" y="80" width="13" height="13" class="fill-sunk"/>
<rect x="357" y="80" width="13" height="13" class="fill-sunk"/>
<rect x="371" y="80" width="13" height="13" class="fill-sunk"/>
<rect x="385" y="80" width="13" height="13" class="fill-sunk"/>
<rect x="399" y="80" width="13" height="13" class="fill-sunk"/>
<rect x="315" y="94" width="13" height="13" class="fill-sunk"/>
<rect x="329" y="94" width="13" height="13" class="fill-sunk"/>
<rect x="343" y="94" width="13" height="13" class="fill-b"/>
<rect x="357" y="94" width="13" height="13" class="fill-sunk"/>
<rect x="371" y="94" width="13" height="13" class="fill-sunk"/>
<rect x="385" y="94" width="13" height="13" class="fill-sunk"/>
<rect x="399" y="94" width="13" height="13" class="fill-sunk"/>
<rect x="315" y="108" width="13" height="13" class="fill-sunk"/>
<rect x="329" y="108" width="13" height="13" class="fill-sunk"/>
<rect x="343" y="108" width="13" height="13" class="fill-sunk"/>
<rect x="357" y="108" width="13" height="13" class="fill-sunk"/>
<rect x="371" y="108" width="13" height="13" class="fill-sunk"/>
<rect x="385" y="108" width="13" height="13" class="fill-sunk"/>
<rect x="399" y="108" width="13" height="13" class="fill-sunk"/>
<rect x="315" y="122" width="13" height="13" class="fill-sunk"/>
<rect x="329" y="122" width="13" height="13" class="fill-sunk"/>
<rect x="343" y="122" width="13" height="13" class="fill-sunk"/>
<rect x="357" y="122" width="13" height="13" class="fill-sunk"/>
<rect x="371" y="122" width="13" height="13" class="fill-sunk"/>
<rect x="385" y="122" width="13" height="13" class="fill-sunk"/>
<rect x="399" y="122" width="13" height="13" class="fill-sunk"/>
<rect x="315" y="136" width="13" height="13" class="fill-sunk"/>
<rect x="329" y="136" width="13" height="13" class="fill-sunk"/>
<rect x="343" y="136" width="13" height="13" class="fill-sunk"/>
<rect x="357" y="136" width="13" height="13" class="fill-sunk"/>
<rect x="371" y="136" width="13" height="13" class="fill-sunk"/>
<rect x="385" y="136" width="13" height="13" class="fill-sunk"/>
<rect x="399" y="136" width="13" height="13" class="fill-sunk"/>
<rect x="315" y="150" width="13" height="13" class="fill-sunk"/>
<rect x="329" y="150" width="13" height="13" class="fill-sunk"/>
<rect x="343" y="150" width="13" height="13" class="fill-sunk"/>
<rect x="357" y="150" width="13" height="13" class="fill-sunk"/>
<rect x="371" y="150" width="13" height="13" class="fill-sunk"/>
<rect x="385" y="150" width="13" height="13" class="fill-sunk"/>
<rect x="399" y="150" width="13" height="13" class="fill-sunk"/>
<rect x="315" y="164" width="13" height="13" class="fill-sunk"/>
<rect x="329" y="164" width="13" height="13" class="fill-sunk"/>
<rect x="343" y="164" width="13" height="13" class="fill-a"/>
<rect x="357" y="164" width="13" height="13" class="fill-sunk"/>
<rect x="371" y="164" width="13" height="13" class="fill-sunk"/>
<rect x="385" y="164" width="13" height="13" class="fill-sunk"/>
<rect x="399" y="164" width="13" height="13" class="fill-sunk"/>
<rect x="315" y="178" width="13" height="13" class="fill-sunk"/>
<rect x="329" y="178" width="13" height="13" class="fill-sunk"/>
<rect x="343" y="178" width="13" height="13" class="fill-sunk"/>
<rect x="357" y="178" width="13" height="13" class="fill-sunk"/>
<rect x="371" y="178" width="13" height="13" class="fill-sunk"/>
<rect x="385" y="178" width="13" height="13" class="fill-sunk"/>
<rect x="399" y="178" width="13" height="13" class="fill-sunk"/>
<text x="465" y="16" class="fig-small">blank margin, shifted down</text>
<rect x="465" y="24" width="13" height="13" class="fill-sunk"/>
<rect x="479" y="24" width="13" height="13" class="fill-sunk"/>
<rect x="493" y="24" width="13" height="13" class="fill-sunk"/>
<rect x="507" y="24" width="13" height="13" class="fill-sunk"/>
<rect x="521" y="24" width="13" height="13" class="fill-sunk"/>
<rect x="535" y="24" width="13" height="13" class="fill-sunk"/>
<rect x="549" y="24" width="13" height="13" class="fill-sunk"/>
<rect x="465" y="38" width="13" height="13" class="fill-sunk"/>
<rect x="479" y="38" width="13" height="13" class="fill-sunk"/>
<rect x="493" y="38" width="13" height="13" class="fill-sunk"/>
<rect x="507" y="38" width="13" height="13" class="fill-sunk"/>
<rect x="521" y="38" width="13" height="13" class="fill-sunk"/>
<rect x="535" y="38" width="13" height="13" class="fill-sunk"/>
<rect x="549" y="38" width="13" height="13" class="fill-sunk"/>
<rect x="465" y="52" width="13" height="13" class="fill-sunk"/>
<rect x="479" y="52" width="13" height="13" class="fill-sunk"/>
<rect x="493" y="52" width="13" height="13" class="fill-sunk"/>
<rect x="507" y="52" width="13" height="13" class="fill-sunk"/>
<rect x="521" y="52" width="13" height="13" class="fill-sunk"/>
<rect x="535" y="52" width="13" height="13" class="fill-sunk"/>
<rect x="549" y="52" width="13" height="13" class="fill-sunk"/>
<rect x="465" y="66" width="13" height="13" class="fill-sunk"/>
<rect x="479" y="66" width="13" height="13" class="fill-sunk"/>
<rect x="493" y="66" width="13" height="13" class="fill-sunk"/>
<rect x="507" y="66" width="13" height="13" class="fill-sunk"/>
<rect x="521" y="66" width="13" height="13" class="fill-sunk"/>
<rect x="535" y="66" width="13" height="13" class="fill-sunk"/>
<rect x="549" y="66" width="13" height="13" class="fill-sunk"/>
<rect x="465" y="80" width="13" height="13" class="fill-sunk"/>
<rect x="479" y="80" width="13" height="13" class="fill-sunk"/>
<rect x="493" y="80" width="13" height="13" class="fill-sunk"/>
<rect x="507" y="80" width="13" height="13" class="fill-sunk"/>
<rect x="521" y="80" width="13" height="13" class="fill-sunk"/>
<rect x="535" y="80" width="13" height="13" class="fill-sunk"/>
<rect x="549" y="80" width="13" height="13" class="fill-sunk"/>
<rect x="465" y="94" width="13" height="13" class="fill-sunk"/>
<rect x="479" y="94" width="13" height="13" class="fill-sunk"/>
<rect x="493" y="94" width="13" height="13" class="fill-sunk"/>
<rect x="507" y="94" width="13" height="13" class="fill-sunk"/>
<rect x="521" y="94" width="13" height="13" class="fill-sunk"/>
<rect x="535" y="94" width="13" height="13" class="fill-sunk"/>
<rect x="549" y="94" width="13" height="13" class="fill-sunk"/>
<rect x="465" y="108" width="13" height="13" class="fill-sunk"/>
<rect x="479" y="108" width="13" height="13" class="fill-sunk"/>
<rect x="493" y="108" width="13" height="13" class="fill-b"/>
<rect x="507" y="108" width="13" height="13" class="fill-sunk"/>
<rect x="521" y="108" width="13" height="13" class="fill-sunk"/>
<rect x="535" y="108" width="13" height="13" class="fill-sunk"/>
<rect x="549" y="108" width="13" height="13" class="fill-sunk"/>
<rect x="465" y="122" width="13" height="13" class="fill-sunk"/>
<rect x="479" y="122" width="13" height="13" class="fill-sunk"/>
<rect x="493" y="122" width="13" height="13" class="fill-sunk"/>
<rect x="507" y="122" width="13" height="13" class="fill-sunk"/>
<rect x="521" y="122" width="13" height="13" class="fill-sunk"/>
<rect x="535" y="122" width="13" height="13" class="fill-sunk"/>
<rect x="549" y="122" width="13" height="13" class="fill-sunk"/>
<rect x="465" y="136" width="13" height="13" class="fill-sunk"/>
<rect x="479" y="136" width="13" height="13" class="fill-sunk"/>
<rect x="493" y="136" width="13" height="13" class="fill-sunk"/>
<rect x="507" y="136" width="13" height="13" class="fill-sunk"/>
<rect x="521" y="136" width="13" height="13" class="fill-sunk"/>
<rect x="535" y="136" width="13" height="13" class="fill-sunk"/>
<rect x="549" y="136" width="13" height="13" class="fill-sunk"/>
<rect x="465" y="150" width="13" height="13" class="fill-sunk"/>
<rect x="479" y="150" width="13" height="13" class="fill-sunk"/>
<rect x="493" y="150" width="13" height="13" class="fill-sunk"/>
<rect x="507" y="150" width="13" height="13" class="fill-sunk"/>
<rect x="521" y="150" width="13" height="13" class="fill-sunk"/>
<rect x="535" y="150" width="13" height="13" class="fill-sunk"/>
<rect x="549" y="150" width="13" height="13" class="fill-sunk"/>
<rect x="465" y="164" width="13" height="13" class="fill-sunk"/>
<rect x="479" y="164" width="13" height="13" class="fill-sunk"/>
<rect x="493" y="164" width="13" height="13" class="fill-sunk"/>
<rect x="507" y="164" width="13" height="13" class="fill-sunk"/>
<rect x="521" y="164" width="13" height="13" class="fill-sunk"/>
<rect x="535" y="164" width="13" height="13" class="fill-sunk"/>
<rect x="549" y="164" width="13" height="13" class="fill-sunk"/>
<rect x="465" y="178" width="13" height="13" class="fill-sunk"/>
<rect x="479" y="178" width="13" height="13" class="fill-sunk"/>
<rect x="493" y="178" width="13" height="13" class="fill-a"/>
<rect x="507" y="178" width="13" height="13" class="fill-sunk"/>
<rect x="521" y="178" width="13" height="13" class="fill-sunk"/>
<rect x="535" y="178" width="13" height="13" class="fill-sunk"/>
<rect x="549" y="178" width="13" height="13" class="fill-sunk"/>
</svg>
Figure 2. The ball (orange) and paddle (blue) in a Catch frame. Without a margin, a one-row shift down pushes the paddle out of the window. With a one-pixel blank margin, the same shift keeps both objects.
```

> [!WARNING]
> Shifts rely on replicated edge pixels. If anything the agent needs touches the border, a shift can delete it or smear it. Check what sits at the image edge before you augment.

### What happens on Catch

The exercise trains the same small noisy DQN with and without shifts. Figure 3 shows the result over eight seeds, and it is not the result the DrQ paper would lead you to expect.

```figure
<svg viewBox="0 0 620 185" role="img" aria-label="Catch rate of eight trained agents with and without random shifts, against the random policy rate of 0.20">
<line x1="70" y1="130" x2="590" y2="130" class="stroke-muted"/>
<line x1="70" y1="130.0" x2="590" y2="130.0" class="stroke-grid"/>
<text x="62" y="134.0" text-anchor="end" class="fig-small">0.0</text>
<line x1="70" y1="80.0" x2="590" y2="80.0" class="stroke-grid"/>
<text x="62" y="84.0" text-anchor="end" class="fig-small">0.5</text>
<line x1="70" y1="30.0" x2="590" y2="30.0" class="stroke-grid"/>
<text x="62" y="34.0" text-anchor="end" class="fig-small">1.0</text>
<line x1="70" y1="110.0" x2="590" y2="110.0" class="stroke-bad dashed"/>
<text x="590" y="105.0" text-anchor="end" class="fig-small">random policy 0.20</text>
<circle cx="83.0" cy="72.0" r="5" class="fill-a"/>
<circle cx="105.0" cy="30.0" r="5" class="fill-a"/>
<circle cx="127.0" cy="30.0" r="5" class="fill-a"/>
<circle cx="149.0" cy="30.0" r="5" class="fill-a"/>
<circle cx="171.0" cy="30.0" r="5" class="fill-a"/>
<circle cx="193.0" cy="30.0" r="5" class="fill-a"/>
<circle cx="215.0" cy="72.0" r="5" class="fill-a"/>
<circle cx="237.0" cy="30.0" r="5" class="fill-a"/>
<text x="160" y="152" text-anchor="middle" class="fig-bold">no shifts</text>
<text x="160" y="170" text-anchor="middle" class="fig-small">mean 0.90</text>
<circle cx="323.0" cy="30.0" r="5" class="fill-b"/>
<circle cx="345.0" cy="30.0" r="5" class="fill-b"/>
<circle cx="367.0" cy="77.7" r="5" class="fill-b"/>
<circle cx="389.0" cy="45.7" r="5" class="fill-b"/>
<circle cx="411.0" cy="70.3" r="5" class="fill-b"/>
<circle cx="433.0" cy="30.0" r="5" class="fill-b"/>
<circle cx="455.0" cy="30.0" r="5" class="fill-b"/>
<circle cx="477.0" cy="70.7" r="5" class="fill-b"/>
<text x="400" y="152" text-anchor="middle" class="fig-bold">random shifts</text>
<text x="400" y="170" text-anchor="middle" class="fig-small">mean 0.82</text>
</svg>
Figure 3. Catch rate over 300 evaluation episodes for eight seeds, after 4000 steps. Both versions beat the random policy's 0.20 on every seed. Shifts did not beat plain DQN on this toy.
```

Both versions beat the random policy's exact catch rate of 0.20, which is $1/5$ because the ball's column is uniform. The shifts did not help. Plain DQN already solves this tiny game, so there is little to regularize. A network that flattens the image also has no built-in shift symmetry, so it must learn the same relation at every offset, which makes the augmented problem harder. The papers' gains come with convolutional encoders on richer images, where memorization is the real failure. Measure augmentation before you believe it.

> [!KEY]
> Random shifts regularize a pixel encoder for free, if a shifted frame calls for the same action. On a task the network can already solve, they cost sample efficiency, as Figure 3 shows.

> [!RECAP]
> - Take the RGB maximum of two frames, then luminance, then scale. Stack the last $k$ frames.
> - Factorized noise is the outer product of two $f(u)$ vectors, and `mu + sigma * eps` lets autograd train both.
> - `random_shift` pads with replicated edges and crops a random window per image.
> - The augmented TD loss averages the target over $K$ shifts and Q over $M$ shifts.

## Build it step by step

The checks test the functions in this order:

1. **`preprocess(frame_a, frame_b)`.** Take `np.maximum` on the RGB arrays, then a dot product with the luminance weights, then divide by 255. The most common mistake is to convert each frame to gray before you take the maximum.
2. **`frame_stack(frames, k)`.** Take the last $k$ frames, left-pad with the oldest frame, and `np.stack` them. It returns shape `(k, H, W)`.
3. **`noise_f(x)` and `factorized_noise(n_in, n_out, rng)`.** Draw `n_in` normals before `n_out` normals. Return `np.outer(f_out, f_in)` with shape `(n_out, n_in)` and `f_out`.
4. **`NoisyLinear.reset_noise` and `forward`.** Copy the noise into the buffers. In `forward`, build `weight_mu + weight_sigma * eps_w` and call `F.linear`. Fall back to the means when `self.noisy` is false. Do not draw noise inside `forward`: it would change the output on every call and break the finite-difference check.
5. **`random_shift(imgs, pad, rng)`.** Pad with `np.pad(..., mode="edge")`, draw offsets with one `rng.integers` call, and crop per image. Zero is not the offset for "no shift": the identity offset is `pad`.
6. **`augmented_td_loss(...)`.** Compute the target under `torch.no_grad()` from $K$ shifted copies of `next_obs`, then the loss from $M$ shifted copies of `obs`. The most common mistake is to shift only the Q input and leave the target unshifted.

## Where this breaks

**Augmentations that change the answer.** A flip turns left into right, and a crop can delete the one pixel that matters. Look at augmented frames before you train.

**Noise and evaluation.** Evaluate with the noise off, as `evaluate_catch` does, and resample the noise at every training step.

**Atari scale.** Real Atari frames need a convolutional encoder and millions of steps. The 12 by 7 board shows each piece, not the benchmark.

## Further reading

- [Mnih et al., *Human-level control through deep reinforcement learning* (Nature, 2015)](https://doi.org/10.1038/nature14236). The Atari DQN and its frame preprocessing.
- [Hessel et al., *Rainbow: Combining improvements in deep reinforcement learning* (2018)](https://arxiv.org/abs/1710.02298). The six components together, and the ablation.
- [Fortunato et al., *Noisy networks for exploration* (2018)](https://arxiv.org/abs/1706.10295). The noisy layer, with independent and factorized noise.
- [Kostrikov, Yarats, and Fergus, *Image augmentation is all you need* (2020)](https://arxiv.org/abs/2004.13649). DrQ, random shifts in Q and in the target.
- [Laskin et al., *Reinforcement learning with augmented data* (2020)](https://arxiv.org/abs/2004.14990). RAD, a comparison of augmentations on pixel control tasks.
- [CS285 (Berkeley), Lecture 8, "Q-learning in Practice"](https://rail.eecs.berkeley.edu/deeprlcourse/). Replay buffers, target networks, and Q-learning practice, after Lecture 7, "Value-Based RL".
- [Hugging Face Deep RL course, "Deep Q-Learning" lesson](https://huggingface.co/learn/deep-rl-course/unit3/introduction). DQN on Atari, with frame stacking.

## The exercise

You write `preprocess` and `frame_stack` for the frames, then `noise_f`, `factorized_noise`, and `NoisyLinear` for exploration. After that you write `random_shift` and `augmented_td_loss`. The Catch game, the network, and the training loop in `train_catch` are given, and they call your functions.

The checks verify `preprocess` on the red and green frames (0.886), the stack values, the noise shapes and rank, `NoisyLinear` against the formula, and its gradients against finite differences. They check `random_shift` for replicated-edge crops, determinism, and a zero pad, and `augmented_td_loss` against the plain Huber loss and against a hand computation for $K = 2$ and $M = 3$. The last check trains a noisy DQN on Catch, with and without shifts, and requires a catch rate of at least 0.40, against 0.20 for a random policy.
