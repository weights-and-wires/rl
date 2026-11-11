---
title: Preferences over trajectories
slug: 053-preferences-over-trajectories
part: "Part 8 — Rewards, evaluation, and many agents"
summary: Learn a per-step reward from comparisons of short trajectory segments, and see what a preference can and cannot pin down.
minutes: 40
gpu: false
objectives:
  - Score a segment as the sum of a learned per-step reward, and say why a mean breaks the comparison.
  - Fit a reward with the Bradley-Terry loss over segment returns, and check its gradient.
  - Say which changes to the reward preferences cannot see, and which change the optimal policy.
  - Pick the next pair to ask about by the disagreement of a reward ensemble.
prereqs:
  - 012-value-iteration
  - 052-reward-design-and-shaping
exercise: 053-preferences-over-trajectories
---

# Preferences over trajectories

You want a simulated robot to walk with a smooth, upright gait. You can recognize one when you see it, but any reward formula you write, such as forward speed minus torque, gets exploited by a robot that lunges and falls. The previous chapter showed that even a harmless-looking hint can change the task.

You can do something easier than writing a reward. Show two short clips and ask which one is better. Asking for a score is harder, because scores drift from day to day, while "this one" is quick and fairly consistent. Christiano and colleagues (2017) learn a reward from such answers and train an agent against it.

This chapter builds that idea on a small grid. You learn a reward from comparisons of trajectory segments, check that its policy matches the one from the hidden true reward, and find out which changes to the reward the comparisons can never reveal.

## What you already know

- Value iteration computes the optimal policy of a tabular MDP for any reward you give it.
- Potential-based shaping changes rewards without changing the optimal policy, and a constant potential misbehaves when episodes end.
- The sigmoid $\sigma(z) = 1 / (1 + e^{-z})$ turns a gap into a probability.

## Compare two segments, not two rewards

A *trajectory segment* is a short run of consecutive steps. A person compares two segments, A and B, and says which is better. You never see a reward, only these answers (Figure 1).

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="A four by four grid with two six-step segments drawn on it, one in blue and one in orange, with their total true rewards">
<rect x="30" y="20" width="62" height="62" class="fill-paper stroke-grid"/>
<rect x="92" y="20" width="62" height="62" class="fill-paper stroke-grid"/>
<rect x="154" y="20" width="62" height="62" class="fill-paper stroke-grid"/>
<rect x="216" y="20" width="62" height="62" class="fill-paper stroke-grid"/>
<rect x="30" y="82" width="62" height="62" class="fill-paper stroke-grid"/>
<rect x="92" y="82" width="62" height="62" class="fill-sunk stroke-grid"/>
<rect x="154" y="82" width="62" height="62" class="fill-paper stroke-grid"/>
<rect x="216" y="82" width="62" height="62" class="fill-paper stroke-grid"/>
<rect x="30" y="144" width="62" height="62" class="fill-paper stroke-grid"/>
<rect x="92" y="144" width="62" height="62" class="fill-paper stroke-grid"/>
<rect x="154" y="144" width="62" height="62" class="fill-bad stroke-grid"/>
<text x="185" y="180" class="fig-bold" text-anchor="middle">H</text>
<rect x="216" y="144" width="62" height="62" class="fill-paper stroke-grid"/>
<rect x="30" y="206" width="62" height="62" class="fill-paper stroke-grid"/>
<rect x="92" y="206" width="62" height="62" class="fill-paper stroke-grid"/>
<rect x="154" y="206" width="62" height="62" class="fill-paper stroke-grid"/>
<rect x="216" y="206" width="62" height="62" class="fill-soft-c stroke-grid"/>
<text x="247" y="242" class="fig-bold" text-anchor="middle">G</text>
<polyline points="123.0,237.0 185.0,237.0 185.0,175.0 185.0,175.0 185.0,175.0 185.0,175.0" class="stroke-a" fill="none"/>
<circle cx="123.0" cy="237.0" r="7" class="fill-a"/>
<circle cx="185.0" cy="237.0" r="5" class="fill-a"/>
<circle cx="185.0" cy="175.0" r="5" class="fill-a"/>
<circle cx="185.0" cy="175.0" r="5" class="fill-a"/>
<circle cx="185.0" cy="175.0" r="5" class="fill-a"/>
<circle cx="185.0" cy="175.0" r="5" class="fill-a"/>
<polyline points="61.0,237.0 61.0,175.0 123.0,175.0 123.0,237.0 123.0,175.0 123.0,175.0" class="stroke-b" fill="none"/>
<circle cx="61.0" cy="237.0" r="7" class="fill-b"/>
<circle cx="61.0" cy="175.0" r="5" class="fill-b"/>
<circle cx="123.0" cy="175.0" r="5" class="fill-b"/>
<circle cx="123.0" cy="237.0" r="5" class="fill-b"/>
<circle cx="123.0" cy="175.0" r="5" class="fill-b"/>
<circle cx="123.0" cy="175.0" r="5" class="fill-b"/>
<text x="320" y="70" class="fig-bold">Segment A (blue)</text>
<text x="320" y="92" class="fig-small">total true reward -4.45</text>
<text x="320" y="140" class="fig-bold">Segment B (orange)</text>
<text x="320" y="162" class="fig-small">total true reward -2.70</text>
<text x="320" y="210" class="fig-small">The large dot is the first state entered.</text>
<text x="320" y="232" class="fig-small">The labeler only says which total is larger.</text>
</svg>
Figure 1. Two six-step segments from the grid world, drawn from the data set the exercise uses. Segment A walks into the hole and stays there, so its total true reward is -4.45. Segment B wanders in the lower-left corner and totals -2.70. A labeler who sees both says B is better, and that one bit is all the learner gets.
```

To fit "A or B", the learner needs a number for each segment. It keeps a reward model $\hat r_\phi(s)$ that scores each state, and it scores a segment $\sigma = (s_1, \dots, s_T)$ by adding up the scores of the states it enters:

$$
\hat R_\phi(\sigma) = \sum_{t=1}^{T} \hat r_\phi(s_t)
$$

Read it aloud: a segment is worth the total of its per-step rewards. It is a sum because the return in RL is a sum. Take a segment A of three steps with rewards 0.2, 0.5, and 1.0, and a segment C of one step with reward 1.0. The sum says A is worth 1.7 and C is worth 1.0. The mean says 0.57 and 1.0, which reverses the answer.

## Turn a score gap into a probability

People are noisy, so you cannot model the labeler as "always picks the larger total". The Bradley-Terry model says the chance of preferring A is the sigmoid of the return gap:

$$
P(\sigma_A \succ \sigma_B) = \sigma\big(\hat R_\phi(\sigma_A) - \hat R_\phi(\sigma_B)\big)
$$

Read it aloud: equal totals give even odds, and a bigger gap gives a more confident answer. The exercise's simulated labeler uses the same form. A *Boltzmann labeler* with inverse temperature $\beta$ prefers A with probability $\sigma(\beta \cdot (R(\sigma_A) - R(\sigma_B)))$ under the true reward $R$. A large $\beta$ is a careful labeler, and $\beta = 0$ is a coin flip.

### Work one pair by hand

Four states have rewards $0.2, 0.5, 1.0, 0.3$. Segment A enters states 0, 1, 2, so its return is $0.2 + 0.5 + 1.0 = 1.7$. Segment B enters states 1, 1, 3, so its return is $0.5 + 0.5 + 0.3 = 1.3$. The gap is $0.4$, and A is preferred with probability $\sigma(0.4) = 0.5987$.

If the labeler said A won, the loss is $-\ln 0.5987 = 0.5130$. If B won, it is $-\ln(1 - 0.5987) = 0.9130$.

To fit a reward, give each state a feature vector $x(s)$ and let $\hat r_\phi(s) = \phi^\top x(s)$. The return gap is then $\phi^\top d$ with $d = \sum_{s \in \sigma_A} x(s) - \sum_{s \in \sigma_B} x(s)$. With one-hot features, $d$ counts visits: here $d = (1, -1, 1, -1)$, because A visits states 0, 1, 2 once and B visits state 1 twice and state 3 once. With $y = 1$ when A won, the loss and its gradient over $N$ pairs are

$$
L(\phi) = \frac{1}{N} \sum_{i=1}^{N} \Big[ \log\big(1 + e^{\phi^\top d_i}\big) - y_i\, \phi^\top d_i \Big], \qquad
\nabla_\phi L = \frac{1}{N} \sum_{i=1}^{N} \big(\sigma(\phi^\top d_i) - y_i\big)\, d_i
$$

Read the gradient aloud: each pair pushes $\phi$ along its visit-count difference, scaled by how far the model's probability is from the label. For the pair above with $y = 1$, the gradient is $-0.4013 \cdot (1, -1, 1, -1)$, so a descent step raises the reward of states 0 and 2 and lowers states 1 and 3. In numpy, with `X` holding one feature row per state:

```python
d = X[segs_a].sum(axis=1) - X[segs_b].sum(axis=1)   # (N, d): feature sums, A minus B
z = d @ w                                           # (N,) predicted return gaps
loss = np.mean(np.logaddexp(0.0, z) - labels * z)
grad = d.T @ (sigmoid(z) - labels) / len(labels)
```

> [!KEY]
> A preference over two segments is one Bradley-Terry observation on the gap of their summed rewards. Part 9 uses the same loss for whole language-model responses.

## Does the learned reward give the right policy?

The exercise world is a 4x4 grid with a wall, a hole, and a goal. The goal and the hole keep the agent where it is, so the task never ends. The true reward is $-0.15$ per step of Manhattan distance to the goal, $+1$ at the goal, and $-1$ at the hole. A Boltzmann labeler with $\beta = 3$ compares random-walk segments of six steps. You fit one weight per state and run value iteration on the learned reward.

Judge the method by the policy. *Policy regret* is the most value that the greedy policy for the learned reward loses against the optimal policy, both measured under the true reward. Zero regret means the learned reward picks actions that are optimal for the true one (Figure 2). Comparing argmax actions state by state would mislead, because on a grid two directions often tie.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="Policy regret against the number of preference labels: it falls to zero for the fitted reward and stays high when every label is flipped">
<line x1="80.0" y1="250.0" x2="540.0" y2="250.0" class="stroke-muted"/>
<line x1="80.0" y1="250.0" x2="80.0" y2="30.0" class="stroke-muted"/>
<line x1="80.0" y1="250.0" x2="540.0" y2="250.0" class="stroke-grid"/>
<text x="72.0" y="254.0" class="fig-small" text-anchor="end">0</text>
<line x1="80.0" y1="195.0" x2="540.0" y2="195.0" class="stroke-grid"/>
<text x="72.0" y="199.0" class="fig-small" text-anchor="end">5</text>
<line x1="80.0" y1="140.0" x2="540.0" y2="140.0" class="stroke-grid"/>
<text x="72.0" y="144.0" class="fig-small" text-anchor="end">10</text>
<line x1="80.0" y1="85.0" x2="540.0" y2="85.0" class="stroke-grid"/>
<text x="72.0" y="89.0" class="fig-small" text-anchor="end">15</text>
<line x1="80.0" y1="30.0" x2="540.0" y2="30.0" class="stroke-grid"/>
<text x="72.0" y="34.0" class="fig-small" text-anchor="end">20</text>
<text x="110.0" y="268.0" class="fig-small" text-anchor="middle">25</text>
<text x="210.0" y="268.0" class="fig-small" text-anchor="middle">50</text>
<text x="310.0" y="268.0" class="fig-small" text-anchor="middle">100</text>
<text x="410.0" y="268.0" class="fig-small" text-anchor="middle">200</text>
<text x="510.0" y="268.0" class="fig-small" text-anchor="middle">300</text>
<polyline points="110.0,163.9 210.0,173.2 310.0,229.8 410.0,237.0 510.0,250.0" class="stroke-a" fill="none"/>
<circle cx="110.0" cy="163.9" r="4" class="fill-a"/>
<circle cx="210.0" cy="173.2" r="4" class="fill-a"/>
<circle cx="310.0" cy="229.8" r="4" class="fill-a"/>
<circle cx="410.0" cy="237.0" r="4" class="fill-a"/>
<circle cx="510.0" cy="250.0" r="4" class="fill-a"/>
<polyline points="110.0,33.6 210.0,31.8 310.0,30.0 410.0,30.0 510.0,30.0" class="stroke-bad" fill="none"/>
<circle cx="110.0" cy="33.6" r="4" class="fill-bad"/>
<circle cx="210.0" cy="31.8" r="4" class="fill-bad"/>
<circle cx="310.0" cy="30.0" r="4" class="fill-bad"/>
<circle cx="410.0" cy="30.0" r="4" class="fill-bad"/>
<circle cx="510.0" cy="30.0" r="4" class="fill-bad"/>
<text x="310" y="290.0" class="fig-small" text-anchor="middle">number of preference labels</text>
<text x="80.0" y="20" class="fig-small">policy regret under the true reward (mean of 20 seeds)</text>
<text x="310.0" y="18.0" class="fig-small" text-anchor="middle">every label flipped</text>
<text x="310.0" y="217.8" class="fig-small" text-anchor="middle">labels as given</text>
</svg>
Figure 2. Policy regret under the true reward, averaged over 20 seeds, as the number of labels grows. With 25 labels the policy gives up about 8 units of value. By 300 labels it gives up nothing, for the same fit. The red line flips every label before fitting: the learner then seeks the hole, and its regret sits at 20.
```

The learned reward is not the true reward. Figure 3 plots each state's learned weight against its true reward for 300 labels. The points track a line with a correlation of 0.85 and a slope of 1.1, not 1. The next section says why that is fine.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="Learned reward of each of 15 states plotted against its true reward: the points fall close to a rising line">
<line x1="80.0" y1="140.0" x2="540.0" y2="140.0" class="stroke-grid"/>
<line x1="310.0" y1="30.0" x2="310.0" y2="250.0" class="stroke-grid"/>
<line x1="80.0" y1="250.0" x2="540.0" y2="250.0" class="stroke-muted"/>
<line x1="80.0" y1="250.0" x2="80.0" y2="30.0" class="stroke-muted"/>
<text x="118.3" y="268.0" class="fig-small" text-anchor="middle">-1</text>
<text x="214.2" y="268.0" class="fig-small" text-anchor="middle">-0.5</text>
<text x="310.0" y="268.0" class="fig-small" text-anchor="middle">0</text>
<text x="405.8" y="268.0" class="fig-small" text-anchor="middle">0.5</text>
<text x="501.7" y="268.0" class="fig-small" text-anchor="middle">1</text>
<text x="72.0" y="254.0" class="fig-small" text-anchor="end">-2</text>
<text x="72.0" y="199.0" class="fig-small" text-anchor="end">-1</text>
<text x="72.0" y="144.0" class="fig-small" text-anchor="end">0</text>
<text x="72.0" y="89.0" class="fig-small" text-anchor="end">1</text>
<text x="72.0" y="34.0" class="fig-small" text-anchor="end">2</text>
<line x1="80.0" y1="187.4" x2="540.0" y2="41.6" class="stroke-muted dashed"/>
<circle cx="137.5" cy="188.9" r="5" class="fill-b"/>
<circle cx="166.2" cy="180.3" r="5" class="fill-b"/>
<circle cx="195.0" cy="159.1" r="5" class="fill-b"/>
<circle cx="223.8" cy="134.8" r="5" class="fill-b"/>
<circle cx="166.2" cy="157.0" r="5" class="fill-b"/>
<circle cx="223.8" cy="131.5" r="5" class="fill-b"/>
<circle cx="252.5" cy="107.9" r="5" class="fill-b"/>
<circle cx="195.0" cy="153.1" r="5" class="fill-b"/>
<circle cx="223.8" cy="116.2" r="5" class="fill-b"/>
<circle cx="118.3" cy="203.5" r="5" class="fill-b"/>
<circle cx="281.2" cy="120.6" r="5" class="fill-b"/>
<circle cx="223.8" cy="128.9" r="5" class="fill-b"/>
<circle cx="252.5" cy="127.3" r="5" class="fill-b"/>
<circle cx="281.2" cy="109.1" r="5" class="fill-b"/>
<circle cx="501.7" cy="81.9" r="5" class="fill-b"/>
<text x="310" y="290.0" class="fig-small" text-anchor="middle">true reward of the state</text>
<text x="80.0" y="20" class="fig-small">learned reward (weight on the state)</text>
<text x="540.0" y="48.0" class="fig-small" text-anchor="end">fitted slope 1.10, correlation 0.85</text>
</svg>
Figure 3. Learned weight against true reward for the 15 states, from one seed with 300 labels. The weights rise with the truth but sit on their own scale and offset, and the states the segments visit rarely are noisier.
```

## What a preference cannot tell you

Preferences compare returns, so a change to the reward that leaves every comparison alone is invisible to the learner. Check each case by arithmetic.

**A constant added to every step.** Add $c$ to each step's reward. A segment of $T$ steps gains $cT$, so the gap between two segments of the same length does not change, and neither does any preference probability. For A and B above, adding $0.3$ per step gives 2.6 and 2.2, a gap of 0.4 again. Segments of different lengths break this: A against C gains $0.9 - 0.3$ in its gap, and the probability moves from $\sigma(0.7) = 0.668$ to $\sigma(1.3) = 0.786$. Compare segments of one fixed length, and the constant is not identified.

Whether the constant matters for the policy depends on the task. With no terminal states, every Q value gains the same $c / (1 - \gamma)$, so the optimal policy cannot move. That is potential-based shaping with a constant potential. When episodes end, the previous chapter showed that a constant per step can reward the agent for avoiding the end. The exercise world has no terminal states for this reason.

**A positive scale.** Multiply every reward by $\alpha > 0$ and the Bradley-Terry gaps scale by $\alpha$. The learner cannot tell a larger reward from a more careful labeler, so the fitted scale mixes the two, which is why the slope in Figure 3 is not 1. A positive scale never changes the optimal policy. A negative scale does, and flipping every label amounts to one.

**A potential.** Add $F = \gamma \Phi(s') - \Phi(s)$ from the previous chapter. With $\gamma = 1$, the extra reward over a segment telescopes to $\Phi(\text{end}) - \Phi(\text{start})$. Two segments with the same start and end keep their gap, so those comparisons cannot see $F$. Segments with different endpoints change their gap, so preferences can tell a shaped reward from the original even though both have the same optimal policy. Even so, they only constrain states that segments visit. A state that never appears has no evidence behind its weight, and the L2 penalty in `fit_reward` pulls it to zero.

> [!WARNING]
> Never compare the mean reward of segments of different lengths, and think about the constant before you sample pairs of different lengths. Both let a shift of every reward change the answer, and the learned reward then carries a length bias that the labeler never expressed.

## Ask the questions that teach the most

Labels cost human time, so choose the pairs. Christiano and colleagues train an *ensemble* of reward models, so the members differ where the data is thin. For a batch of candidate pairs, each member predicts $P(\sigma_A \succ \sigma_B)$, and they ask about the pair whose predictions have the highest variance across members. Where the members agree, a label adds little. Where they split, a label settles it.

```python
d = X[segs_a].sum(axis=1) - X[segs_b].sum(axis=1)          # (N, d)
probs = sigmoid(ws @ d.T)                                   # (K, N): one row per member
index = int(np.argmax(probs.var(axis=0)))                   # the pair they disagree about most
```

Take three members whose predicted return gaps on a pair are $(3, -3, 0)$. Their probabilities are $0.953$, $0.047$, and $0.5$, with variance about $0.14$. A pair on which all three predict a gap of 4 has probabilities near $0.982$ and variance 0. The first pair is worth a label even though the second has the larger gaps.

> [!RECAP]
> - A segment's score is the sum of a per-step reward. A mean of unequal-length segments gives wrong answers.
> - The chance of preferring A is $\sigma$ of the return gap. The loss is $\log(1 + e^z) - yz$, and its gradient is $(\sigma(z) - y)\, d$.
> - Preferences cannot see a constant per step (for equal lengths) or a positive scale, and neither changes the optimal policy in a task with no terminal states.
> - You judge the learned reward by the policy it induces, measured as regret under the true reward.
> - Ensemble variance picks the pair worth a label.

## Build it step by step

The checks test the functions in this order. The world, the segment sampler, and `policy_regret` are written for you.

1. **`segment_return(reward_fn, segment)`.** Apply `reward_fn` to the array of states and sum. The common mistake is a mean, and the first check catches it.
2. **`preference_prob(return_a, return_b)`.** Return $\sigma$ of the gap. The naive `1 / (1 + np.exp(-gap))` overflows at $-800$, so write the sigmoid as `0.5 * (1 + np.tanh(gap / 2))`.
3. **`bt_segment_loss(w, X, segs_a, segs_b, labels)`.** Build `d`, compute `z = d @ w`, and return the loss and gradient. Label 1 means A won. A flipped sign trains a reward that points at the hole, and the hand-worked pair catches it.
4. **`simulate_preference(true_r, seg_a, seg_b, beta, rng)`.** Compute the true return gap and return `int(rng.random() < p)` with $p = \sigma(\beta \cdot \text{gap})$. Draw exactly one number, so the labels are reproducible.
5. **`fit_reward(X, segs_a, segs_b, labels, l2, steps, lr)`.** Start from zeros and take full-batch steps on the loss plus `l2 * w`.
6. **`disagreement_query(ws, X, segs_a, segs_b)`.** Compute each member's probability for each pair and return the index of the largest variance across members.

## Where this breaks

**Reward hacking moves into the learned reward.** The policy finds states the reward model overrates and the labeler never saw. Retraining the reward model while the policy trains labels that behavior, yet Ibarz and colleagues (2018) still found reward hacking in Atari.

**Disagreement is a heuristic.** It asks about pairs the model is unsure of, not pairs that change the policy, and it can hurt on some tasks.

**Sample cost.** The grid needs hundreds of labels for 15 states. Later methods stretch labels by relabeling stored experience and pretraining without a reward (Lee, Smith, and Abbeel, 2021).


## Further reading

- [Christiano, Leike, Brown, Martic, Legg, and Amodei (2017), Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741). Defines the method of this chapter: preferences between pairs of trajectory segments, with feedback on under one percent of the agent's interactions.
- [Ibarz, Leike, Pohlen, Irving, Legg, and Amodei (2018), Reward learning from human preferences and demonstrations in Atari](https://arxiv.org/abs/1811.06521). Adds expert demonstrations to the preferences and studies the noise and the reward hacking that appear.
- [Lee, Smith, and Abbeel (2021), PEBBLE](https://arxiv.org/abs/2106.05091). An off-policy method that relabels experience when the reward model changes and pretrains with unsupervised exploration.
- [Stanford CS224R, Deep Reinforcement Learning, Spring 2026, Lecture 8, Reward Learning](https://cs224r.stanford.edu/). The course's lecture on this topic, with Christiano and colleagues (2017) as its reading. Lecture 9 is "RL for LLMs: Preference Optimization."
- [Lambert, Reward Modeling](https://rlhfbook.com/c/05-reward-models), in *Reinforcement Learning from Human Feedback*. The Bradley-Terry reward model for language models, the setting of [Reward models from preferences](/lessons/061-reward-models).

## The exercise

You learn a reward from segment preferences on the grid. Write `segment_return`, `preference_prob`, `bt_segment_loss`, `simulate_preference`, `fit_reward`, and `disagreement_query`.

The checks cover a sum against a mean, the sigmoid at gaps of $\pm 800$, and a constant per step that leaves equal-length probabilities unchanged. They compute the hand-worked pair's loss and gradient for both labels and check the gradient with finite differences. On five seeds, the fit must track the true reward and its greedy policy must lose under 0.05 of true value, also after adding 5 per step, while flipped labels give a clearly worse policy. Last, the ensemble must pick the highest-variance pair.
