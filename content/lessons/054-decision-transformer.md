---
title: "Decision transformer: RL as sequence modeling"
slug: 054-decision-transformer
part: "Part 7 — Models, exploration, and fixed data"
summary: Turn a log of trajectories into token sequences, train a causal transformer to predict actions, and ask for the return you want.
minutes: 40
gpu: false
objectives:
  - Compute returns-to-go, with and without a discount, and lay a trajectory out as interleaved (return, state, action) tokens.
  - Build a causal mask so that the action at step t depends on the returns and states up to t and the actions before t, and check it by changing future tokens.
  - Condition on a target return at test time and lower it by each observed reward.
  - Explain why a return-conditioned model cannot stitch suboptimal trajectories the way Q-learning does, and why a lucky return misleads it under stochastic dynamics.
prereqs:
  - 049-imitation-learning
  - 051-offline-rl
  - 052-advantage-weighted-policy-improvement
exercise: 054-decision-transformer
---

# Decision transformer: RL as sequence modeling

You have a log of a robot's runs. Some runs were careful and reached the goal in eight moves. Others wandered for twenty. You want a policy that is as good as the careful runs, and you cannot collect more data. [Behavior cloning](/lessons/049-imitation-learning) copies the average of the log, careless runs included. [Offline RL](/lessons/051-offline-rl) fits a value function and pays for it with a penalty that keeps the policy near the data.

This chapter takes a third route. Treat each trajectory as a sentence made of numbers, and train a language-model-style network to predict the next token. Put the return the trajectory eventually earned at the front of every step. The model learns what careless and careful runs look like. To get a careful run, you tell it the return you want.

The method is the decision transformer (Chen and others, 2021). It needs no value function and no bootstrapping, only supervised learning. This chapter also shows exactly what that simplicity costs: the model can only reproduce returns the data already contains, and it can mistake luck for skill.

## What you already know

- A trajectory is a sequence of states, actions, and rewards, and the return $G_t$ adds the rewards from step $t$ on, with discount $\gamma$.
- [Behavior cloning](/lessons/049-imitation-learning) fits $\pi(a \mid s)$ to logged actions by supervised learning.
- [Offline Q-iteration](/lessons/051-offline-rl) backs values up through transitions that appear in a fixed dataset.
- [Weighted regression](/lessons/052-advantage-weighted-policy-improvement) can favor the better actions in a log without a new rollout.
- Language models minimize a [masked next-token loss](/lessons/062-sft-and-the-post-training-pipeline), and a causal mask hides the future from each position.

## Tag every step with its future

Take one trajectory with rewards $r_0, r_1, \dots, r_{T-1}$. The *return-to-go* at step $t$ is the return from $t$ onward:

$$
G_t = \sum_{k=t}^{T-1} \gamma^{\,k-t}\, r_k = r_t + \gamma\, G_{t+1}, \qquad G_T = 0
$$

Read it as: the reward you get now, plus everything still to come. Entry $0$ is the whole return, and the last entry is the last reward. The decision transformer paper sets $\gamma = 1$. In numpy, one backward pass does it:

```python
out, running = np.zeros_like(rewards), 0.0
for t in range(len(rewards) - 1, -1, -1):
    running = rewards[t] + gamma * running
    out[t] = running
```

Two hand examples to check against. Rewards $[-1, -1, -1]$ with $\gamma = 1$ give $[-3, -2, -1]$. Rewards $[1, 0, 2]$ with $\gamma = 0.5$ give $G_2 = 2$, $G_1 = 0 + 0.5 \cdot 2 = 1$, and $G_0 = 1 + 0.5 \cdot 1 = 1.5$, so $[1.5, 1, 2]$.

The mistake to avoid is an off-by-one. A sum that leaves out $r_t$, or one that includes $r_{t-1}$, gives a number that is not the future from step $t$. The model would then learn to condition on a quantity it cannot know at test time.

## Lay the trajectory out as tokens

For each step, the model reads three tokens in a fixed order: the return-to-go $G_t$, the state $s_t$, and the action $a_t$. A two-step trajectory with $G = [-2, -1]$, states $[0, 1]$, and actions $[2, 1]$ becomes six tokens with these kinds, timesteps, and values:

| Position | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Token | $G_0$ | $s_0$ | $a_0$ | $G_1$ | $s_1$ | $a_1$ |
| Timestep | 0 | 0 | 0 | 1 | 1 | 1 |
| Value | $-2$ | $0$ | $2$ | $-1$ | $1$ | $1$ |

Each kind gets its own embedding: a linear layer for the return, a lookup table for a discrete state or action. A learned embedding of the timestep $t$ is added to all three tokens of step $t$, so the model knows where in the episode it is. The network predicts the action at step $t$ from the output at the **state** token $s_t$.

```figure
<svg viewBox="0 0 600 370" role="img" aria-label="Attention mask over six tokens G0 s0 a0 G1 s1 a1: each token sees itself and the tokens before it; the s1 row sees G0 to s1 but not a1">
<text x="172.0" y="50.0" class="fig-small" text-anchor="middle">G0</text>
<text x="216.0" y="50.0" class="fig-small" text-anchor="middle">s0</text>
<text x="260.0" y="50.0" class="fig-small" text-anchor="middle">a0</text>
<text x="304.0" y="50.0" class="fig-small" text-anchor="middle">G1</text>
<text x="348.0" y="50.0" class="fig-small" text-anchor="middle">s1</text>
<text x="392.0" y="50.0" class="fig-small" text-anchor="middle">a1</text>
<text x="140.0" y="86.0" class="fig-small" text-anchor="end">G0</text>
<rect x="150.0" y="60.0" width="44.0" height="44.0" class="fill-soft-a stroke-grid"/>
<rect x="194.0" y="60.0" width="44.0" height="44.0" class="fill-sunk stroke-grid"/>
<rect x="238.0" y="60.0" width="44.0" height="44.0" class="fill-sunk stroke-grid"/>
<rect x="282.0" y="60.0" width="44.0" height="44.0" class="fill-sunk stroke-grid"/>
<rect x="326.0" y="60.0" width="44.0" height="44.0" class="fill-sunk stroke-grid"/>
<rect x="370.0" y="60.0" width="44.0" height="44.0" class="fill-sunk stroke-grid"/>
<text x="140.0" y="130.0" class="fig-small" text-anchor="end">s0</text>
<rect x="150.0" y="104.0" width="44.0" height="44.0" class="fill-soft-b stroke-grid"/>
<rect x="194.0" y="104.0" width="44.0" height="44.0" class="fill-soft-b stroke-grid"/>
<rect x="238.0" y="104.0" width="44.0" height="44.0" class="fill-sunk stroke-grid"/>
<rect x="282.0" y="104.0" width="44.0" height="44.0" class="fill-sunk stroke-grid"/>
<rect x="326.0" y="104.0" width="44.0" height="44.0" class="fill-sunk stroke-grid"/>
<rect x="370.0" y="104.0" width="44.0" height="44.0" class="fill-sunk stroke-grid"/>
<text x="140.0" y="174.0" class="fig-small" text-anchor="end">a0</text>
<rect x="150.0" y="148.0" width="44.0" height="44.0" class="fill-soft-a stroke-grid"/>
<rect x="194.0" y="148.0" width="44.0" height="44.0" class="fill-soft-a stroke-grid"/>
<rect x="238.0" y="148.0" width="44.0" height="44.0" class="fill-soft-a stroke-grid"/>
<rect x="282.0" y="148.0" width="44.0" height="44.0" class="fill-sunk stroke-grid"/>
<rect x="326.0" y="148.0" width="44.0" height="44.0" class="fill-sunk stroke-grid"/>
<rect x="370.0" y="148.0" width="44.0" height="44.0" class="fill-sunk stroke-grid"/>
<text x="140.0" y="218.0" class="fig-small" text-anchor="end">G1</text>
<rect x="150.0" y="192.0" width="44.0" height="44.0" class="fill-soft-a stroke-grid"/>
<rect x="194.0" y="192.0" width="44.0" height="44.0" class="fill-soft-a stroke-grid"/>
<rect x="238.0" y="192.0" width="44.0" height="44.0" class="fill-soft-a stroke-grid"/>
<rect x="282.0" y="192.0" width="44.0" height="44.0" class="fill-soft-a stroke-grid"/>
<rect x="326.0" y="192.0" width="44.0" height="44.0" class="fill-sunk stroke-grid"/>
<rect x="370.0" y="192.0" width="44.0" height="44.0" class="fill-sunk stroke-grid"/>
<text x="140.0" y="262.0" class="fig-small" text-anchor="end">s1</text>
<rect x="150.0" y="236.0" width="44.0" height="44.0" class="fill-soft-b stroke-grid"/>
<rect x="194.0" y="236.0" width="44.0" height="44.0" class="fill-soft-b stroke-grid"/>
<rect x="238.0" y="236.0" width="44.0" height="44.0" class="fill-soft-b stroke-grid"/>
<rect x="282.0" y="236.0" width="44.0" height="44.0" class="fill-soft-b stroke-grid"/>
<rect x="326.0" y="236.0" width="44.0" height="44.0" class="fill-soft-b stroke-grid"/>
<rect x="370.0" y="236.0" width="44.0" height="44.0" class="fill-sunk stroke-grid"/>
<text x="140.0" y="306.0" class="fig-small" text-anchor="end">a1</text>
<rect x="150.0" y="280.0" width="44.0" height="44.0" class="fill-soft-a stroke-grid"/>
<rect x="194.0" y="280.0" width="44.0" height="44.0" class="fill-soft-a stroke-grid"/>
<rect x="238.0" y="280.0" width="44.0" height="44.0" class="fill-soft-a stroke-grid"/>
<rect x="282.0" y="280.0" width="44.0" height="44.0" class="fill-soft-a stroke-grid"/>
<rect x="326.0" y="280.0" width="44.0" height="44.0" class="fill-soft-a stroke-grid"/>
<rect x="370.0" y="280.0" width="44.0" height="44.0" class="fill-soft-a stroke-grid"/>
<text x="282.0" y="352.0" class="fig-small" text-anchor="middle">columns: the token being read. rows: the token reading. s rows are orange: they predict the action.</text>
<text x="30" y="262.0" class="fig-small" text-anchor="start">predicts a1</text>
</svg>
Figure 1. The attention mask over the six tokens of a two-step trajectory. Each row reads the tokens that are shaded in it. The row for s1, which predicts a1, sees G0 through s1 and never a1 itself.
```

## Hide the future with a causal mask

A transformer layer lets every token read every other token. For prediction, that is cheating: if $s_t$ could read $a_t$, the model would copy the answer from its own input and learn nothing. The fix is a *causal mask*, a table whose entry $(i, j)$ says whether token $i$ may read token $j$. Allow $j \le i$ and block $j > i$:

$$
M_{ij} = \begin{cases} 1 & j \le i \\ 0 & j > i \end{cases}
$$

Read it as: each token sees itself and everything before it. Because $a_t$ comes after $s_t$ in the sequence, the output at $s_t$ depends on $G_{\le t}$, $s_{\le t}$, and $a_{<t}$, which is what you know when you must choose $a_t$. Attention applies the mask by setting blocked scores to $-\infty$ before the softmax, so each blocked token gets weight zero:

```python
scores = (x @ Wq) @ (x @ Wk).T / np.sqrt(d)
scores = np.where(mask, scores, -np.inf)
```

You can test a mask without training anything. Change the tokens after position $i$ and check that the outputs at positions up to $i$ do not move. A mask that is shifted one column too far, `np.tril(ones, k=1)`, lets the state token read the action it must predict. The output at $s_t$ then changes when you change $a_t$, and the property test catches it.

> [!KEY]
> A causal mask turns "predict the next token" into "predict the action from the past." The only thing that makes the model more than behavior cloning is the first token of each step: the return you want.

## Ask for a return, then lower the ask

At test time you pick a target return $G^\star$ for the whole episode, put it in the first $G$ slot, and act. After each real reward, the target must fall by what you got. If the target stayed at $-8$ after you had already lost three points, the model would be asked for an impossible rest of the episode. The update is the recursion run forward, $G_{t+1} = (G_t - r_t)/\gamma$, and with $\gamma = 1$ it is $G_t - r_t$.

Take a 5 by 5 grid where every step pays $-1$ and the goal is eight moves away. Ask for $-8$. After the first move, $r_0 = -1$, so the next target is $-8 - (-1) = -7$. The targets read $-8, -7, -6, \dots, -1$, and the last step asks for $-1$, exactly the reward of the last move.

The exercise trains one masked attention layer, followed by a small network, on 200 episodes from this grid. Half of the episodes come from a near-optimal policy and half from a nearly random one, so returns run from $-8$ to $-24$. Figure 2 asks the trained model for nine different returns. It averages 20 episodes per target, for five training seeds.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Mean realized return of a tiny decision transformer against the target return it is asked for; realized return rises with the target and follows the diagonal">
<line x1="80.0" y1="62.0" x2="560.0" y2="62.0" class="stroke-grid"/>
<text x="72.0" y="66.0" class="fig-small" text-anchor="end">-8</text>
<text x="512.0" y="282.0" class="fig-small" text-anchor="middle">-8</text>
<line x1="80.0" y1="106.0" x2="560.0" y2="106.0" class="stroke-grid"/>
<text x="72.0" y="110.0" class="fig-small" text-anchor="end">-12</text>
<text x="416.0" y="282.0" class="fig-small" text-anchor="middle">-12</text>
<line x1="80.0" y1="150.0" x2="560.0" y2="150.0" class="stroke-grid"/>
<text x="72.0" y="154.0" class="fig-small" text-anchor="end">-16</text>
<text x="320.0" y="282.0" class="fig-small" text-anchor="middle">-16</text>
<line x1="80.0" y1="194.0" x2="560.0" y2="194.0" class="stroke-grid"/>
<text x="72.0" y="198.0" class="fig-small" text-anchor="end">-20</text>
<text x="224.0" y="282.0" class="fig-small" text-anchor="middle">-20</text>
<line x1="80.0" y1="238.0" x2="560.0" y2="238.0" class="stroke-grid"/>
<text x="72.0" y="242.0" class="fig-small" text-anchor="end">-24</text>
<text x="128.0" y="282.0" class="fig-small" text-anchor="middle">-24</text>
<line x1="128.0" y1="238.0" x2="512.0" y2="62.0" class="stroke-muted dashed"/>
<text x="426.0" y="98.0" class="fig-small">realized = target</text>
<line x1="512.0" y1="75.8" x2="512.0" y2="62.6" class="stroke-b"/>
<line x1="464.0" y1="108.8" x2="464.0" y2="83.4" class="stroke-b"/>
<line x1="416.0" y1="128.0" x2="416.0" y2="107.7" class="stroke-b"/>
<line x1="368.0" y1="157.1" x2="368.0" y2="135.2" class="stroke-b"/>
<line x1="320.0" y1="209.4" x2="320.0" y2="135.7" class="stroke-b"/>
<line x1="272.0" y1="205.6" x2="272.0" y2="177.5" class="stroke-b"/>
<line x1="224.0" y1="218.2" x2="224.0" y2="189.6" class="stroke-b"/>
<line x1="176.0" y1="229.2" x2="176.0" y2="210.5" class="stroke-b"/>
<line x1="128.0" y1="231.9" x2="128.0" y2="221.5" class="stroke-b"/>
<polyline points="512.0,67.4 464.0,96.4 416.0,118.9 368.0,147.6 320.0,176.7 272.0,193.8 224.0,202.4 176.0,224.1 128.0,228.1" class="stroke-a" fill="none"/>
<circle cx="512.0" cy="67.4" r="4" class="fill-a"/>
<circle cx="464.0" cy="96.4" r="4" class="fill-a"/>
<circle cx="416.0" cy="118.9" r="4" class="fill-a"/>
<circle cx="368.0" cy="147.6" r="4" class="fill-a"/>
<circle cx="320.0" cy="176.7" r="4" class="fill-a"/>
<circle cx="272.0" cy="193.8" r="4" class="fill-a"/>
<circle cx="224.0" cy="202.4" r="4" class="fill-a"/>
<circle cx="176.0" cy="224.1" r="4" class="fill-a"/>
<circle cx="128.0" cy="228.1" r="4" class="fill-a"/>
<text x="320" y="306.0" class="fig-small" text-anchor="middle">target return asked for</text>
<text x="20" y="150" class="fig-small" text-anchor="middle" transform="rotate(-90 20 150)">realized return</text>
<text x="550.0" y="30.0" class="fig-small" text-anchor="end">bars: lowest and highest of five training seeds</text>
</svg>
Figure 2. Realized return against the return asked for. The model's mean return rises with the target and stays close to the diagonal, from about -8.5 at a target of -8 to about -20.8 at a target of -20. Across the five seeds, the gap between those two targets runs from 11.2 to 14.1.
```

The model does not search for a good policy. It has seen what trajectories with a return near $-8$ look like in each state, and it reproduces them. A simpler baseline does the same without a transformer: *return-conditioned behavior cloning* feeds $(s_t, G_t)$ to a small network and predicts $a_t$ (the RvS recipe of Emmons and others, 2022). It matches the transformer whenever the current state and return-to-go are enough to choose an action. A transformer helps when history matters, for example when the state is only partly observed, as in [Partial observability and memory](/lessons/045-partial-observability-and-memory).

## What conditioning cannot do

A return-conditioned model learns the *distribution of actions among trajectories with a given return*. Two limits follow.

**It cannot stitch.** Figure 3 shows two trajectories that meet at state $s_1$. A starts at $s_0$ and earns $0 + 1 = 1$. B starts at $s_3$ and earns $0 + 10 = 10$. A Q-learning method backs up through $s_1$: $Q(s_1, \text{B's action}) = 10$, so $Q(s_0, \text{A's first action}) = 0 + 10 = 10$. It combines A's start with B's finish. The best single trajectory from $s_0$ in the data earns $1$, and a model conditioned on a target of $10$ at $s_0$ has no example of a run from $s_0$ that earned it. Return conditioning reproduces trajectories. Dynamic programming recombines them.

```figure
<svg viewBox="0 0 600 260" role="img" aria-label="Two trajectories meet at state s1. Trajectory A starts at s0 and ends with return 1. Trajectory B starts at s3 and ends with return 10. Offline Q-iteration values s0 at 10; the best trajectory from s0 is worth 1">
<defs><marker id="c052-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<circle cx="70" cy="80" r="24" class="fill-soft-a stroke-grid"/><text x="70" y="84" class="fig-small" text-anchor="middle">s0</text>
<circle cx="70" cy="190" r="24" class="fill-soft-a stroke-grid"/><text x="70" y="194" class="fig-small" text-anchor="middle">s3</text>
<circle cx="260" cy="135" r="24" class="fill-soft-a stroke-grid"/><text x="260" y="139" class="fig-small" text-anchor="middle">s1</text>
<circle cx="470" cy="80" r="24" class="fill-sunk stroke-grid"/><text x="470" y="84" class="fig-small" text-anchor="middle">end</text>
<circle cx="470" cy="190" r="24" class="fill-sunk stroke-grid"/><text x="470" y="194" class="fig-small" text-anchor="middle">end</text>
<line x1="95.0" y1="87.2" x2="233.1" y2="127.2" class="stroke-a" marker-end="url(#c052-arrow)"/>
<text x="165.0" y="97.5" class="fig-small" text-anchor="middle">A: reward 0</text>
<line x1="95.0" y1="182.8" x2="233.1" y2="142.8" class="stroke-b" marker-end="url(#c052-arrow)"/>
<text x="165.0" y="184.5" class="fig-small" text-anchor="middle">B: reward 0</text>
<line x1="285.2" y1="128.4" x2="442.9" y2="87.1" class="stroke-a" marker-end="url(#c052-arrow)"/>
<text x="365.0" y="97.5" class="fig-small" text-anchor="middle">A: reward 1</text>
<line x1="285.2" y1="141.6" x2="442.9" y2="182.9" class="stroke-b" marker-end="url(#c052-arrow)"/>
<text x="365.0" y="184.5" class="fig-small" text-anchor="middle">B: reward 10</text>
<text x="300" y="240" class="fig-small" text-anchor="middle">best return from s0 in the data: 1. Offline Q-iteration, Q(s0, a0): 10.</text>
</svg>
Figure 3. Two logged trajectories share state s1. The best return from s0 in the data is 1. Offline Q-iteration, which backs a value through the shared state, values s0 at 10.
```

**It mistakes luck for skill.** Suppose a risky arm pays 10 with probability 0.1 and 0 otherwise, so its mean is 1. A safe arm pays 2 every time. Collect 100 pulls of each. Ten of the risky pulls hit 10, and no safe pull does. Conditioned on a return of 10, every matching sample used the risky arm, so the model picks it and earns 1 on average, worse than the safe arm's 2. The return in the log reflects a lucky draw, not a choice the agent controlled. Paster and others (2022) show this failure for decision transformers and RvS in stochastic environments and propose conditioning on an estimate of what the policy controls. On deterministic dynamics, such as the grid here, a return is a fact about the actions, and the problem does not arise.

> [!WARNING]
> Asking for a return higher than anything in the data does not give you a better policy. The model extrapolates, and nothing in the loss rewards it for being right.

## Build it step by step

The checks test these functions in this order. The grid, the dataset collector, the `TinyDT` network, `train_dt`, and `dt_policy` are written for you.

1. **`returns_to_go(rewards, gamma=1.0)`.** Take the reward array and return an array of the same length. Run a backward loop with a running sum. The common mistake is to add the reward after storing, which drops $r_t$ from entry $t$.
2. **`build_tokens(rtg, states, actions)`.** Return three arrays, the token kinds, the timesteps, and the values, interleaved as $G_t, s_t, a_t$. When there is one fewer action than steps, stop after the last state.
3. **`causal_mask(n)`.** Return an $(n, n)$ boolean array that is true where $j \le i$. `np.tril` does it. Do not pass `k=1`.
4. **`masked_attention(x, mask, Wq, Wk, Wv)`.** Compute scaled scores, replace blocked entries with `-np.inf`, subtract each row's maximum, and apply the softmax. Subtract the maximum after masking so a row of finite numbers stays finite.
5. **`rollout_with_target(mdp, act, target, horizon, gamma, seed)`.** At each step, append the current target to the history, call `act`, step the environment, and set the target to `(target - r) / gamma`. Return the total reward and the targets used.
6. **`conditional_action_probs(actions, returns, target, n_actions)`.** Count the actions among samples whose return matches the target with `np.isclose`, and normalize. With no match, return the uniform distribution.
7. **`offline_q_iteration(transitions, n_states, n_actions, gamma, iters)`.** Give only the seen $(s, a)$ pairs a value, start them at zero, and back up through the max over seen actions in the next state. Skip the next-state term on a terminal step.

Then `train_dt` and `mean_return` run your functions on a real model: the `__main__` block trains the transformer and prints the realized return for three targets.

## Where this breaks

**Stochastic dynamics.** The return in the log mixes the agent's choices with chance. When chance matters, condition on something the agent controls, such as an estimate of the expected return, instead of the realized one.

**A target outside the data.** The model is only as good as the best trajectories it has seen at each state, so a high target does not create skill. Offline RL methods with a value function, such as [IQL and CQL](/lessons/051-offline-rl), can beat the best trajectory in the data by stitching.

**Long contexts.** Attention over $3T$ tokens costs time and memory that grow with $T^2$. Real systems cut the sequence to the last $K$ steps.

**Return as the only goal.** A single number says little about what the agent should do. Later work conditions on goals or on other outcomes, and language-model policies condition on a prompt, as in [Language models as policies](/lessons/063-language-models-as-policies).

## Further reading

- [Chen, Lu, Rajeswaran, Lee, Grover, Laskin, Abbeel, Srinivas, and Mordatch (2021), Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345). The return, state, and action token layout and the causal transformer.
- [Janner, Li, and Levine (2021), Offline Reinforcement Learning as One Big Sequence Modeling Problem](https://arxiv.org/abs/2106.02039). The Trajectory Transformer models whole trajectories and plans by beam search.
- [Emmons, Eysenbach, Kostrikov, and Levine (2022), RvS: What is Essential for Offline RL via Supervised Learning?](https://arxiv.org/abs/2112.10751). A plain MLP conditioned on return or goal matches much of what the transformer achieves.
- [Paster, McIlraith, and Ba (2022), You Can't Count on Luck: Why Decision Transformers and RvS Fail in Stochastic Environments](https://arxiv.org/abs/2205.15967). Why conditioning on a realized return rewards luck, and a fix.
- [CS285, Spring 2026, lectures 17 and 18: "Offline RL Part 1" and "Offline RL Part 2"](https://rail.eecs.berkeley.edu/deeprlcourse/). The value-based offline methods that this chapter contrasts with return conditioning.

## The exercise

You build the decision transformer's parts and the two arguments about its limits. Write `returns_to_go`, `build_tokens`, `causal_mask`, and `masked_attention`, then `rollout_with_target`, `conditional_action_probs`, and `offline_q_iteration`.

The checks verify returns-to-go against hand values for $\gamma = 1$ and $\gamma = 0.5$, the token layout, and the mask rows for $s_0$ and $s_1$. A property test changes the future tokens of a 12-token sequence and requires that no earlier output moves, and that the output at each state token ignores the action after it. The rollout must produce the targets $-8, -7, \dots, -1$, and the lucky-arm example must give an expected return of 1 against the safe arm's 2. Offline Q-iteration must value $s_0$ at 10 against a best in-data return of 1. A last check trains the transformer on the grid and requires a mean return above $-12$ when you ask for $-8$, and at least 5 points better than when you ask for $-20$.
