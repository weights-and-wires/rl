---
title: World models and MuZero
slug: 041-world-models-and-muzero
part: "Part 7 — Models, exploration, and fixed data"
summary: A model only has to get the Bellman backups right. See why, train on imagined lambda-returns, and search a learned model with pUCT.
minutes: 50
gpu: false
objectives:
  - Compute Dreamer's lambda-returns over an imagined trajectory, with continuation flags.
  - "State the value-equivalence principle: a wrong model is good enough when its Bellman backups match the real ones."
  - Implement MuZero's pUCT score and search, and trace a small search by hand.
  - Explain why a learned value function lets a short search see far past its depth.
prereqs:
  - 040-model-based-rl
  - 009-the-bellman-equation
  - 020-eligibility-traces
  - 022-monte-carlo-tree-search
exercise: 041-world-models-and-muzero
---

# World models and MuZero

The last chapter predicted the next state of CartPole: four numbers. A game of Atari hands the agent a picture of 210 by 160 pixels. Predicting every pixel of the next frame is expensive, and most of those pixels do not matter, because the color of the sky does not change which way to steer.

So what does a model need to predict? This chapter gives one answer: only what planning uses. Two families of methods follow from it. A *world model*, as in Dreamer, works in a small learned code and trains the agent on trajectories it imagines. MuZero goes further and does not predict what the world looks like at all. Its model predicts rewards, values, and good actions, and a tree search plans with them.

You will build three pieces: the return Dreamer trains on, a test for when a wrong model still plans correctly, and MuZero's tree search.

## What you already know

- A model predicts the next state and reward, and its error compounds over an open-loop rollout (see [Learning a model and planning with it](/lessons/040-model-based-rl)).
- The Bellman expectation equation says $V^\pi = r_\pi + \gamma P_\pi V^\pi$ (see [The Bellman equation](/lessons/009-the-bellman-equation)).
- The lambda-return blends every $n$-step return, with weights that shrink by $\lambda$ (see [Eligibility traces and TD(lambda)](/lessons/020-eligibility-traces)).
- Monte Carlo tree search balances trying good actions against trying rare ones (see [Monte Carlo tree search](/lessons/022-monte-carlo-tree-search)).

## Imagine in a small code

The Dreamer family (Hafner et al., 2020 onward) follows World Models (Ha and Schmidhuber, 2018). An encoder compresses each observation into a small *latent state*. A dynamics network predicts the next latent state from the current one and an action, and small heads predict the reward and whether the episode continues. All are trained on real experience.

Then the agent stops looking at the world. It starts from latent states of real experience, rolls its policy forward for a short horizon inside the model, and trains an actor and a critic on those imagined trajectories, which is *learning in imagination*. The critic needs a target. A one-step target $r_t + \gamma v(s_{t+1})$ leans entirely on the critic's own guess. A full imagined return leans entirely on the model's rewards, which drift the further out they go. Dreamer blends the two.

## Blend model and critic with the lambda-return

Over an imagined trajectory of length $H$, the *lambda-return* is computed backward from the end:

$$
G^\lambda_t = r_t + \gamma \, c_t \Big( (1 - \lambda)\, v(s_{t+1}) + \lambda \, G^\lambda_{t+1} \Big), \qquad G^\lambda_H = v(s_H)
$$

Read it piece by piece. $r_t$ is the model's predicted reward. $c_t$ is the *continuation flag*, 1 if $s_{t+1}$ is not terminal and 0 if it is, so a zero cuts off everything after the episode ends. Inside the brackets, what comes after step $t$ blends the critic's value of the next state, with weight $1 - \lambda$, and the lambda-return from the next step on, with weight $\lambda$. At the horizon there is no more imagination, so the recursion starts from $v(s_H)$.

Work an example. Take $H = 3$, $\gamma = 0.9$, rewards $[0, 0, 1]$, values $[0.5, 0.6, 0.8, 0]$ for $s_0$ to $s_3$, and flags $[1, 1, 0]$, so the episode ends on the third step. With $\lambda = 0.5$, go backward:

- $G_2 = 1 + 0.9 \times 0 \times (\ldots) = 1$. The zero flag cuts off $v(s_3)$.
- $G_1 = 0 + 0.9 \times (0.5 \times 0.8 + 0.5 \times 1) = 0.81$.
- $G_0 = 0 + 0.9 \times (0.5 \times 0.6 + 0.5 \times 0.81) = 0.6345$.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Grouped bars: the lambda-return at the three steps of the worked example for lambda 0, 0.5, and 1; returns grow with lambda at the first two steps and all equal 1 at the last step">
<line x1="70.0" y1="240.0" x2="570.0" y2="240.0" class="stroke-grid"/>
<text x="62.0" y="244.0" class="fig-small" text-anchor="end">0.0</text>
<line x1="70.0" y1="140.0" x2="570.0" y2="140.0" class="stroke-grid"/>
<text x="62.0" y="144.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="70.0" y1="40.0" x2="570.0" y2="40.0" class="stroke-grid"/>
<text x="62.0" y="44.0" class="fig-small" text-anchor="end">1.0</text>
<line x1="70.0" y1="240.0" x2="570.0" y2="240.0" class="stroke-muted"/>
<rect x="100.0" y="132.0" width="38.0" height="108.0" class="fill-a"/>
<text x="119.0" y="126.0" class="fig-small" text-anchor="middle">0.54</text>
<rect x="142.0" y="113.1" width="38.0" height="126.9" class="fill-c"/>
<text x="161.0" y="107.1" class="fig-small" text-anchor="middle">0.6345</text>
<rect x="184.0" y="78.0" width="38.0" height="162.0" class="fill-b"/>
<text x="203.0" y="72.0" class="fig-small" text-anchor="middle">0.81</text>
<text x="160.0" y="260.0" class="fig-small" text-anchor="middle">step t = 0</text>
<rect x="265.0" y="96.0" width="38.0" height="144.0" class="fill-a"/>
<text x="284.0" y="90.0" class="fig-small" text-anchor="middle">0.72</text>
<rect x="307.0" y="78.0" width="38.0" height="162.0" class="fill-c"/>
<text x="326.0" y="72.0" class="fig-small" text-anchor="middle">0.81</text>
<rect x="349.0" y="60.0" width="38.0" height="180.0" class="fill-b"/>
<text x="368.0" y="54.0" class="fig-small" text-anchor="middle">0.9</text>
<text x="325.0" y="260.0" class="fig-small" text-anchor="middle">step t = 1</text>
<rect x="430.0" y="40.0" width="38.0" height="200.0" class="fill-a"/>
<text x="449.0" y="34.0" class="fig-small" text-anchor="middle">1</text>
<rect x="472.0" y="40.0" width="38.0" height="200.0" class="fill-c"/>
<text x="491.0" y="34.0" class="fig-small" text-anchor="middle">1</text>
<rect x="514.0" y="40.0" width="38.0" height="200.0" class="fill-b"/>
<text x="533.0" y="34.0" class="fig-small" text-anchor="middle">1</text>
<text x="490.0" y="260.0" class="fig-small" text-anchor="middle">step t = 2</text>
<rect x="300.0" y="276.0" width="14.0" height="14.0" class="fill-a"/>
<text x="320.0" y="288.0" class="fig-small" text-anchor="start">λ = 0</text>
<rect x="380.0" y="276.0" width="14.0" height="14.0" class="fill-c"/>
<text x="400.0" y="288.0" class="fig-small" text-anchor="start">λ = 0.5</text>
<rect x="480.0" y="276.0" width="14.0" height="14.0" class="fill-b"/>
<text x="500.0" y="288.0" class="fig-small" text-anchor="start">λ = 1</text>
</svg>
Figure 1. The lambda-return at each step of the worked example, with γ = 0.9, rewards 0, 0, 1, and an episode that ends at step 2. Notice that λ = 0 trusts the critic after one step, λ = 1 trusts the model's rewards, and the last step is 1 for every λ because the episode ends there.
```

With $\lambda = 0$ the returns are $[0.54, 0.72, 1]$, one-step targets that trust the critic. With $\lambda = 1$ they are $[0.81, 0.9, 1]$, the discounted rewards. Dreamer uses $\lambda = 0.95$. In numpy:

```python
following = values[H]
for t in reversed(range(H)):
    following = rewards[t] + gamma * continues[t] * (
        (1 - lam) * values[t + 1] + lam * following)
    out[t] = following
```

Subtract the value and you get the advantage from [Generalized advantage estimation](/lessons/032-generalized-advantage-estimation), because both obey the same backward recursion. Dreamer's critic target and PPO's advantage are one quantity viewed from two sides.

> [!KEY]
> The lambda-return trades the critic's bias against the model's drift. $\lambda = 0$ trusts the critic after one step, and $\lambda = 1$ trusts the model's rewards for the whole horizon.

## Predict only what planning needs

Dreamer still learns to reconstruct observations. MuZero (Schrittwieser et al., 2020) drops that. Its model has three learned functions: a *representation* function from observations to a latent state, a *dynamics* function from a latent state and action to a reward and the next latent state, and a *prediction* function from a latent state to a policy prior and a value. No loss asks the latent state to look like the world. MuZero matched AlphaZero in Go, chess, and shogi without being told the rules.

Why is that enough? Planning only ever touches a model through Bellman backups. So two models that give the same backups, for the policies and values you plan with, give the same plans, even if one of them gets the transitions wrong. This is the *value-equivalence principle* (Grimm et al., 2020). A model $(\hat P, \hat R)$ is *value-equivalent* to the real environment, for a set of policies and a set of value functions, when for every policy $\pi$ and value function $v$ in those sets

$$
\hat r_\pi + \gamma \hat P_\pi v = r_\pi + \gamma P_\pi v
$$

Read it as: apply one Bellman expectation backup of $\pi$ to $v$, once with the model and once with reality, and ask whether the two answers agree at every state. Here $r_\pi(s) = \sum_a \pi(a \mid s) R(s, a)$ is the expected reward and $P_\pi$ is the state-to-state transition matrix under $\pi$. In numpy one backup is two lines:

```python
P_pi = np.einsum("sa,sat->st", pi, P)
backup = (pi * R).sum(axis=1)[:, None] + gamma * P_pi @ V   # V is (S, m)
```

Take three states with values $v = [0, 1, 2]$. From some state the real next-state probabilities are $[0.2, 0.4, 0.4]$, and a model says $[0.3, 0.2, 0.5]$. The model is wrong about every single probability, as Figure 2 shows, yet the expected next value is the same: $0.2 \times 0 + 0.4 \times 1 + 0.4 \times 2 = 1.2$ for reality and $0.3 \times 0 + 0.2 \times 1 + 0.5 \times 2 = 1.2$ for the model. The difference $[0.1, -0.2, 0.1]$ is orthogonal to $v$, so the backup cannot see it.

```figure
<svg viewBox="0 0 600 280" role="img" aria-label="Bars: the real next-state probabilities 0.2, 0.4, 0.4 and a wrong model 0.3, 0.2, 0.5 over three states with values 0, 1, 2; both give an expected next value of 1.2">
<line x1="70.0" y1="210.0" x2="570.0" y2="210.0" class="stroke-grid"/>
<text x="62.0" y="214.0" class="fig-small" text-anchor="end">0.00</text>
<line x1="70.0" y1="135.0" x2="570.0" y2="135.0" class="stroke-grid"/>
<text x="62.0" y="139.0" class="fig-small" text-anchor="end">0.25</text>
<line x1="70.0" y1="60.0" x2="570.0" y2="60.0" class="stroke-grid"/>
<text x="62.0" y="64.0" class="fig-small" text-anchor="end">0.50</text>
<line x1="70.0" y1="210.0" x2="570.0" y2="210.0" class="stroke-muted"/>
<rect x="120.0" y="150.0" width="50.0" height="60.0" class="fill-a"/>
<text x="145.0" y="144.0" class="fig-small" text-anchor="middle">0.2</text>
<rect x="174.0" y="120.0" width="50.0" height="90.0" class="fill-b"/>
<text x="199.0" y="114.0" class="fig-small" text-anchor="middle">0.3</text>
<text x="172.0" y="230.0" class="fig-small" text-anchor="middle">next state 0, v = 0</text>
<rect x="285.0" y="90.0" width="50.0" height="120.0" class="fill-a"/>
<text x="310.0" y="84.0" class="fig-small" text-anchor="middle">0.4</text>
<rect x="339.0" y="150.0" width="50.0" height="60.0" class="fill-b"/>
<text x="364.0" y="144.0" class="fig-small" text-anchor="middle">0.2</text>
<text x="337.0" y="230.0" class="fig-small" text-anchor="middle">next state 1, v = 1</text>
<rect x="450.0" y="90.0" width="50.0" height="120.0" class="fill-a"/>
<text x="475.0" y="84.0" class="fig-small" text-anchor="middle">0.4</text>
<rect x="504.0" y="60.0" width="50.0" height="150.0" class="fill-b"/>
<text x="529.0" y="54.0" class="fig-small" text-anchor="middle">0.5</text>
<text x="502.0" y="230.0" class="fig-small" text-anchor="middle">next state 2, v = 2</text>
<rect x="110.0" y="250.0" width="14.0" height="14.0" class="fill-a"/>
<text x="130.0" y="262.0" class="fig-small" text-anchor="start">real: 0.2×0 + 0.4×1 + 0.4×2 = 1.2</text>
<rect x="360.0" y="250.0" width="14.0" height="14.0" class="fill-b"/>
<text x="380.0" y="262.0" class="fig-small" text-anchor="start">model: 0.3×0 + 0.2×1 + 0.5×2 = 1.2</text>
<text x="62.0" y="28.0" class="fig-small" text-anchor="end">probability</text>
</svg>
Figure 2. A model that is wrong about every transition probability, yet gives the same expected next value as reality. Notice that the Bellman backup only sees that expected value, so this model plans exactly as well as the true one.
```

There is a strong consequence. If the model is value-equivalent for a policy $\pi$ and its true value $V^\pi$, the model's backup leaves $V^\pi$ unchanged, because it equals the real backup, which does. A backup with probability rows has a single fixed point, so evaluating $\pi$ inside the wrong model returns the true $V^\pi$ exactly.

> [!KEY]
> A model only has to get the Bellman backups right for the policies and values you plan with. Its transitions can be wrong in every direction the values cannot see.

## Search with pUCT

MuZero plans with Monte Carlo tree search inside its model. Every node stores, for each action $a$, a visit count $N(a)$, a mean backed-up return $Q(a)$, and a prior $P(a)$ from the prediction function. The *pUCT* rule picks which action to follow:

$$
\text{score}(a) = Q(a) + P(a) \, \frac{\sqrt{\sum_b N(b)}}{1 + N(a)} \left( c_1 + \log \frac{\sum_b N(b) + c_2 + 1}{c_2} \right)
$$

Read it as two terms. $Q(a)$ rewards actions that have done well. The second rewards actions the prior likes and that have few visits, and it shrinks as $N(a)$ grows. $c_1 = 1.25$ sets its size, and $c_2 = 19652$ lets it grow slowly at tens of thousands of visits. In numpy:

```python
total = visits.sum()
weight = c1 + np.log((total + c2 + 1) / c2)
scores = q + prior * np.sqrt(total) / (1 + visits) * weight
```

Each simulation walks down the tree by pUCT until it picks an action it has never tried. It asks the model for that action's reward and next state, and the prediction function for the new node's prior and value. Then it walks back up, turning the leaf value into a return $G \leftarrow r + \gamma G$ at each edge, and each edge adds $G$ to its total and 1 to its count. A terminal next state has a leaf value of 0. This exercise skips MuZero's rescaling of $Q$ to the range 0 to 1, because its rewards already lie there, and scores an untried action by the node's own predicted value.

A toy tree shows the whole mechanism. At the root, action 0 pays 0.5 and ends the episode. Action 1 pays nothing and leads to a state B, where action 1 pays 1 and ends. The best first move is action 1, worth $0.9 \times 1 = 0.9$ with $\gamma = 0.9$. The prior is $[0.5, 0.5]$ and every predicted value is 0.

| Simulation | $N$ before | $Q$ before | Scores | Picks | Return backed up |
|---|---|---|---|---|---|
| 1 | $[0, 0]$ | $[0, 0]$ | $[0, 0]$ | 0 (tie) | $0.5$ |
| 2 | $[1, 0]$ | $[0.5, 0]$ | $[0.8125, 0.6251]$ | 0 | $0.5$ |
| 3 | $[2, 0]$ | $[0.5, 0]$ | $[0.7947, 0.8840]$ | 1 | $0 + 0.9 \times 0 = 0$ |
| 4 | $[2, 1]$ | $[0.5, 0]$ | $[0.8609, 0.5414]$ | 0 | $0.5$ |

After four simulations the root has visits $[3, 1]$ and has not yet found the reward under B, because B's predicted value is 0. Figure 3 shows what follows: by 20 simulations the visits are $[10, 10]$, and after 200 they are $[21, 179]$, with $Q(1) = 0.86$ approaching the true 0.9.

```figure
<svg viewBox="0 0 600 280" role="img" aria-label="Root visit counts of the search on the toy tree after 4, 20, and 200 simulations: 3 and 1, then 10 and 10, then 21 and 179 for action 0 and action 1">
<line x1="70.0" y1="210.0" x2="570.0" y2="210.0" class="stroke-grid"/>
<text x="62.0" y="214.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="135.0" x2="570.0" y2="135.0" class="stroke-grid"/>
<text x="62.0" y="139.0" class="fig-small" text-anchor="end">100</text>
<line x1="70.0" y1="60.0" x2="570.0" y2="60.0" class="stroke-grid"/>
<text x="62.0" y="64.0" class="fig-small" text-anchor="end">200</text>
<line x1="70.0" y1="210.0" x2="570.0" y2="210.0" class="stroke-muted"/>
<rect x="120.0" y="207.8" width="50.0" height="2.2" class="fill-a"/>
<text x="145.0" y="201.8" class="fig-small" text-anchor="middle">3</text>
<rect x="174.0" y="209.2" width="50.0" height="0.8" class="fill-b"/>
<text x="199.0" y="203.2" class="fig-small" text-anchor="middle">1</text>
<text x="172.0" y="230.0" class="fig-small" text-anchor="middle">4 simulations</text>
<rect x="285.0" y="202.5" width="50.0" height="7.5" class="fill-a"/>
<text x="310.0" y="196.5" class="fig-small" text-anchor="middle">10</text>
<rect x="339.0" y="202.5" width="50.0" height="7.5" class="fill-b"/>
<text x="364.0" y="196.5" class="fig-small" text-anchor="middle">10</text>
<text x="337.0" y="230.0" class="fig-small" text-anchor="middle">20 simulations</text>
<rect x="450.0" y="194.2" width="50.0" height="15.8" class="fill-a"/>
<text x="475.0" y="188.2" class="fig-small" text-anchor="middle">21</text>
<rect x="504.0" y="75.8" width="50.0" height="134.2" class="fill-b"/>
<text x="529.0" y="69.8" class="fig-small" text-anchor="middle">179</text>
<text x="502.0" y="230.0" class="fig-small" text-anchor="middle">200 simulations</text>
<rect x="150.0" y="250.0" width="14.0" height="14.0" class="fill-a"/>
<text x="170.0" y="262.0" class="fig-small" text-anchor="start">action 0: pays 0.5 and ends</text>
<rect x="380.0" y="250.0" width="14.0" height="14.0" class="fill-b"/>
<text x="400.0" y="262.0" class="fig-small" text-anchor="start">action 1: pays 1 one step later</text>
<text x="62.0" y="28.0" class="fig-small" text-anchor="end">visits</text>
</svg>
Figure 3. How the root's visits move as the search runs, with a uniform prior and every predicted value equal to 0. Notice that the search first sticks with the sure 0.5, then finds the delayed reward worth 0.9 and sends most visits to it.
```

A learned value makes the search short. On a 3 by 4 grid world whose goal is 5 steps away, a search with zero values splits 200 root visits exactly $[50, 50, 50, 50]$, because no simulation reaches the goal. With values computed inside a model learned from 400 random transitions, the visits become $[43, 60, 54, 43]$, favoring right and down, the optimal actions. The value summarizes everything beyond the leaves.

> [!RECAP]
> - The lambda-return recursion runs backward from $G_H = v(s_H)$, and the continuation flag multiplies the whole bracket.
> - A model is value-equivalent when its Bellman backups equal the real ones for the chosen policies and values.
> - pUCT is $Q$ plus the prior times $\sqrt{\sum N}/(1 + N)$ times the $c_1, c_2$ weight, and ties go to the lowest action.
> - Backup turns the leaf value into $G \leftarrow r + \gamma G$ at each edge on the way up.

## Build it step by step

The exercise gives you the toy tree and the grid-world model learner. The checks test your functions in this order.

1. `lambda_returns(rewards, values, continues, gamma, lam)` takes arrays of length $H$, $H + 1$, and $H$ and returns length $H$. Use one reversed loop seeded with `values[H]`. The common mistake is multiplying only $v(s_{t+1})$ by the flag and not the $\lambda$ term, which leaks value across an episode end.
2. `bellman_backup(P, R, gamma, pi, v)` builds `P_pi` with `np.einsum` and `r_pi` with a row sum, and supports `v` of shape `(S,)` and `(S, m)`.
3. `value_equivalence_loss(P_model, R_model, P, R, gamma, policies, values)` subtracts the two backups of `values` for each policy, squares, sums, and divides by the number of entries. The common mistake is comparing the two models' matrices instead of their backups.
4. `puct_scores(q, prior, visits, c1, c2)` is one vectorized line. With no visits, `np.sqrt(0)` makes the exploration term zero.
5. `mcts(model_step, predict, root_state, n_actions, n_simulations, gamma)` keeps a node class with arrays `N`, `W`, and `R`, and a dictionary of children, and marks an edge to a terminal state with `None`. Record the `(node, action)` path, then back up in reverse. The common mistake is writing $\gamma (G + r)$ instead of $r + \gamma G$.

## Where this breaks

- **The actor exploits the model.** Dreamer's policy trains only in imagination, so it chases any reward the model invents. Short horizons, lambda-returns that lean on the critic, and fresh real data keep this in check.
- **Value equivalence is only as broad as its sets.** A model that is value-equivalent for some policies and values can be wrong for others, and an improving policy reaches situations the model never saw.
- **Scale matters for pUCT.** With rewards in the hundreds, $Q$ swamps the exploration term and the search stops exploring. Normalize $Q$ within the tree, as MuZero does.

## Further reading

The following papers cover world models, MuZero, and value equivalence:

- [Hafner, Lillicrap, Ba, and Norouzi (2020), Dream to control: learning behaviors by latent imagination](https://arxiv.org/abs/1912.01603) introduces Dreamer, with lambda-returns over imagined latent trajectories.
- [Schrittwieser et al. (2020), Mastering Atari, Go, chess and shogi by planning with a learned model](https://arxiv.org/abs/1911.08265) introduces MuZero.
- [Grimm, Barreto, Singh, and Silver (2020), The value equivalence principle for model-based reinforcement learning](https://arxiv.org/abs/2011.03506) makes the formal case that a model needs only to match Bellman backups.

## The exercise

You build the three ideas of this chapter. Write `lambda_returns` for Dreamer's critic targets, `bellman_backup` and `value_equivalence_loss` to measure when a model plans correctly, and `puct_scores` and `mcts` for MuZero's tree search.

The checks verify `lambda_returns` against the worked example, its limits, and a continuation cut. They check that `value_equivalence_loss` is zero for a model whose transitions differ from the truth but agree on the value functions, check `puct_scores` by hand, replay the four-simulation trace, and confirm that search over a learned grid-world model picks an optimal action from every state.
