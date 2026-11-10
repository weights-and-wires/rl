---
title: World models and MuZero
slug: 042-world-models-and-muzero
part: "Part 7 — Models, exploration, and fixed data"
summary: A model only has to get the Bellman backups right. See why, train on imagined lambda-returns, and search a learned model with pUCT.
minutes: 55
gpu: false
objectives:
  - Compute Dreamer's lambda-returns over an imagined trajectory, with continuation flags.
  - "State the value-equivalence principle: a wrong model is good enough when its Bellman backups match the real ones."
  - Fit a deterministic latent by least squares and compare its k-step return with the real one.
  - Implement MuZero's pUCT score and search, and trace a small search by hand.
  - Explain why a learned value function lets a short search see far past its depth.
prereqs:
  - 041-model-based-rl
  - 009-the-bellman-equation
  - 020-eligibility-traces
  - 022-monte-carlo-tree-search
exercise: 042-world-models-and-muzero
---

# World models and MuZero

The last chapter predicted the next state of CartPole: four numbers. Atari hands the agent a picture of 210 by 160 pixels, and most of those pixels do not matter. The color of the sky does not change which way to steer.

So what does a model need to predict? Only what planning uses. Dreamer trains on trajectories it imagines inside a small code. MuZero predicts rewards, values, and actions, and a tree search plans with them.

You will build the lambda-return, a value-equivalence test, a $k$-step latent, and MuZero's search.

## What you already know

- A model predicts the next state and reward, and its error compounds over an open-loop rollout (see [Learning a model and planning with it](/lessons/041-model-based-rl)).
- The Bellman expectation equation says $V^\pi = r_\pi + \gamma P_\pi V^\pi$ (see [The Bellman equation](/lessons/009-the-bellman-equation)).
- The lambda-return blends every $n$-step return, with weights that shrink by $\lambda$ (see [Eligibility traces and TD(lambda)](/lessons/020-eligibility-traces)).
- Monte Carlo tree search balances trying good actions against trying rare ones (see [Monte Carlo tree search](/lessons/022-monte-carlo-tree-search)).

## Imagine in a small code

The Dreamer family (Hafner et al., 2020 onward) follows World Models (Ha and Schmidhuber, 2018). An encoder compresses each observation into a *latent state*, and a dynamics model predicts the next latent and the reward. From latents of real experience the agent imagines a short trajectory and trains on it. A one-step target $r_t + \gamma v(s_{t+1})$ leans on the critic. The model's rewards drift as the horizon grows, so Dreamer blends the two. That blend is *learning in imagination*.

## Blend model and critic with the lambda-return

Over an imagined trajectory of length $H$, the *lambda-return* is computed backward from the end:

$$
G^\lambda_t = r_t + \gamma \, c_t \Big( (1 - \lambda)\, v(s_{t+1}) + \lambda \, G^\lambda_{t+1} \Big), \qquad G^\lambda_H = v(s_H)
$$

$r_t$ is the predicted reward. The *continuation flag* $c_t$ is 0 at a terminal next state, and that zero drops the rest of the episode. Weight $1 - \lambda$ sits on the critic's next value, and weight $\lambda$ sits on the next lambda-return. The recursion starts at $v(s_H)$.

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

With $\lambda = 0$ the returns are $[0.54, 0.72, 1]$, and with $\lambda = 1$ they are $[0.81, 0.9, 1]$. Dreamer uses $\lambda = 0.95$. In numpy:

```python
following = values[H]
for t in reversed(range(H)):
    following = rewards[t] + gamma * continues[t] * (
        (1 - lam) * values[t + 1] + lam * following)
    out[t] = following
```

Subtract $v(s_t)$ and the same recursion is the advantage in [Generalized advantage estimation](/lessons/032-generalized-advantage-estimation).

> [!KEY]
> The lambda-return trades the critic's bias against the model's drift. $\lambda = 0$ trusts the critic after one step, and $\lambda = 1$ trusts the model's rewards for the whole horizon.

## Predict only what planning needs

Dreamer still learns to reconstruct observations.

Dreamer imagines a deterministic latent. A dynamics step maps that latent and an action to the next latent and a reward. After $k$ steps, a value head reads a scalar off the latent. The imagined return adds the discounted rewards to $\gamma^k$ times that scalar.

MuZero (Schrittwieser et al., 2020) drops the reconstruction. Its model has three learned functions: a *representation* function from observations to a latent state, a *dynamics* function from a latent state and action to a reward and the next latent state, and a *prediction* function from a latent state to a policy prior and a value. No loss asks the latent state to look like the world. MuZero matched AlphaZero in Go, chess, and shogi without being told the rules.

Planning uses a model only through Bellman backups, so two models that agree on those backups give the same plans. This is the *value-equivalence principle* (Grimm et al., 2020). A model $(\hat P, \hat R)$ is *value-equivalent* for a set of policies and value functions when, for every policy $\pi$ and value $v$ in those sets,

$$
\hat r_\pi + \gamma \hat P_\pi v = r_\pi + \gamma P_\pi v
$$

Apply one Bellman backup of $\pi$ to $v$ with the model and with reality, and ask whether the two answers agree at every state. Here $r_\pi(s) = \sum_a \pi(a \mid s) R(s, a)$ is the expected reward, and $P_\pi$ is the transition matrix under $\pi$. In numpy:

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

If the model is value-equivalent for a policy $\pi$ and the true $V^\pi$, its backup leaves $V^\pi$ fixed, because the real backup does. That backup has one fixed point, so evaluating $\pi$ inside the model returns the true $V^\pi$.

> [!KEY]
> A model only has to get the Bellman backups right for the policies and values you plan with. Its transitions can be wrong in every direction the values cannot see.

## Unroll a latent for k steps

One backup can match a model that gets every transition wrong. Unrolling $k$ steps exposes that model unless the latent keeps the coordinate the value depends on.

Take $D = 2$, $R = 1$, and $A = 2$. An observation is $[x, \text{noise}]$, and the value of a state is $x$. Action 0 adds $-1$ to $x$ and action 1 adds $+1$. The reward is the new $x$.

Encoder $[1, 0]$ sends $[2, 5]$ to $z = 2$. The dynamics row and the reward row are $[1, -1, 1]$, applied to $z$ joined with a one-hot action. Action 1, input $[2, 0, 1]$, returns latent 3, reward 3, and value 3. Action 0, input $[2, 1, 0]$, returns 1, 1, and 1.

At $\gamma = 0.5$ the 1-step unroll is $3 + 0.5 \times 3 = 4.5$, the return from the next value. Actions $[1, 0, 1]$ move $x$ through $2 \to 3 \to 2 \to 3$, with rewards $3, 2, 3$. The value three actions later is 3, and the value head reads 3. The 3-step unroll is $3 + 0.5 \times 2 + 0.25 \times 3 + 0.125 \times 3 = 5.125$.

$$
G^{(k)} = \sum_{t=0}^{k-1} \gamma^{t} \hat{r}_t + \gamma^{k} w^{\top} z_k
$$

$\hat{r}_t$ is the predicted reward, $z_k$ the latent after the last action, and $w$ the value head. The $k$-step value error is the mean square of $G^{(k)}$ against the real return.

A noise-only encoder reads 5 and predicts 10.75. The squared error against 5.125 is 31.640625. The mean square of $[4.5, 10.75]$ against $[4.5, 5.125]$ is 15.8203125, and a matching pair has mean square 0.

Fit the encoder, the dynamics, and the value head by least squares. `values[i]` is the value of the next observation $i$, so the noise weight is 0 and one step matches the next latent and the reward. The transitions are as follows:

| Observation | Action | Next observation | Reward | Value |
| --- | --- | --- | --- | --- |
| [2, 5] | 1 | [3, 9] | 3 | 3 |
| [3, 9] | 0 | [2, 1] | 2 | 2 |
| [0, 4] | 1 | [1, 8] | 1 | 1 |
| [1, 8] | 0 | [0, 2] | 0 | 0 |

The fit returns rows $[1, 0]$ and $[1, -1, 1]$, unrolls 4.5 and 5.125, and a $k$-step value error of 0. The following lines compute one step:

```python
inp = np.concatenate([z, one_hot])
z_next = dynamics["W"] @ inp
reward = float(dynamics["reward"] @ inp)
```

## Search with pUCT

MuZero plans with Monte Carlo tree search inside its model. Every node stores, for each action $a$, a visit count $N(a)$, a mean backed-up return $Q(a)$, and a prior $P(a)$ from the prediction function. The *pUCT* rule picks which action to follow:

$$
\text{score}(a) = Q(a) + P(a) \, \frac{\sqrt{\sum_b N(b)}}{1 + N(a)} \left( c_1 + \log \frac{\sum_b N(b) + c_2 + 1}{c_2} \right)
$$

$Q(a)$ is large when an action has paid off. The second term is large for a high prior and a low visit count, and it shrinks as $N(a)$ grows. $c_1 = 1.25$ sets its size, and $c_2 = 19652$ lets it grow slowly at tens of thousands of visits. In numpy:

```python
total = visits.sum()
weight = c1 + np.log((total + c2 + 1) / c2)
scores = q + prior * np.sqrt(total) / (1 + visits) * weight
```

Each simulation follows pUCT to an untried action, queries the model for the reward and next state, and stores the new prior and value. A terminal next state has leaf value 0. Walking back, each edge sets $G \leftarrow r + \gamma G$ and adds $G$ to its total and 1 to its count. An untried action uses the node's own value as $Q$. Rewards already lie in 0 to 1, so this search skips MuZero's rescaling of $Q$.

A toy tree shows the mechanism. From the root, action 0 pays 0.5 and ends. Action 1 pays nothing and leads to state B, where action 1 pays 1 and ends. The best first move is action 1, worth $0.9 \times 1 = 0.9$ at $\gamma = 0.9$. The prior is $[0.5, 0.5]$ and every predicted value is 0.

| Simulation | $N$ before | $Q$ before | Scores | Picks | Return backed up |
|---|---|---|---|---|---|
| 1 | $[0, 0]$ | $[0, 0]$ | $[0, 0]$ | 0 (tie) | $0.5$ |
| 2 | $[1, 0]$ | $[0.5, 0]$ | $[0.8125, 0.6251]$ | 0 | $0.5$ |
| 3 | $[2, 0]$ | $[0.5, 0]$ | $[0.7947, 0.8840]$ | 1 | $0 + 0.9 \times 0 = 0$ |
| 4 | $[2, 1]$ | $[0.5, 0]$ | $[0.8609, 0.5414]$ | 0 | $0.5$ |

After four simulations the visits are $[3, 1]$, and the search has not found the reward under B, because B's predicted value is 0. By 20 simulations the visits are $[10, 10]$, and after 200 they are $[21, 179]$, with $Q(1) = 0.86$ approaching the true 0.9. Figure 3 plots that shift.

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
> - Run the lambda-return backward from $G_H = v(s_H)$, and multiply the bracket by the continuation flag.
> - Value equivalence means the Bellman backups match for the policies and values you chose.
> - pUCT adds $Q$ to the prior times $\sqrt{\sum N}/(1 + N)$ times the $c_1, c_2$ weight. Ties take the lowest action.
> - On the way up, set $G \leftarrow r + \gamma G$ at each edge.
> - A $k$-step unroll sums the discounted predicted rewards and adds $\gamma^k$ times the value on the final latent.

## Build it step by step

The exercise gives you the toy tree and the grid-world model learner. The checks call your functions in this order.

1. `lambda_returns(rewards, values, continues, gamma, lam)` takes lengths $H$, $H + 1$, and $H$ and returns length $H$. Walk backward from `values[H]`. Multiply the whole bracket by the flag, or a finished episode still leaks value.
2. `bellman_backup(P, R, gamma, pi, v)` forms `P_pi` with `np.einsum` and `r_pi` by a row sum. Accept `v` shaped `(S,)` or `(S, m)`.
3. `value_equivalence_loss(P_model, R_model, P, R, gamma, policies, values)` averages the squared gap between the two backups. The common mistake is to compare the matrices instead.
4. `puct_scores(q, prior, visits, c1, c2)` is one vectorized line. With no visits, `np.sqrt(0)` zeroes the exploration term.
5. `mcts(model_step, predict, root_state, n_actions, n_simulations, gamma)` stores `N`, `W`, `R`, and children, and marks a terminal edge with `None`. Record the path and back up with $G \leftarrow r + \gamma G$. Writing $\gamma(G + r)$ is the usual slip.
6. `encode(observation, weights)` returns `weights @ observation`. The shapes are `(D,)`, `(R, D)`, and `(R,)`.
7. `latent_step(z, action, dynamics)` joins `z` with a one-hot of length $A$. `W` is `(R, R + A)` and `reward` is `(R + A,)`. Return `(W @ inp, reward @ inp)`. The usual slip is to pass $-1$ or $+1$ in place of the one-hot.
8. `unroll_value(z, actions, dynamics, value_head, gamma)` steps once per action, adds `gamma**t * reward`, then `gamma**k * (value_head @ z_k)`. Returning the 1-step value when $k = 3$ fails.
9. `k_step_value_error(predicted, target)` is the mean of the squared differences. Summing the squares misses the hand-computed mean.
10. `learn_latent_model(observations, actions, rewards, next_observations, values, rank, n_actions)` calls `np.linalg.lstsq`. Observations are `(N, D)`, and `values[i]` is the value of `next_observations[i]`. Fit the encoder on that pair, then fit `W` and the reward row on `concat(latent, one_hot)`. Return `encoder`, `dynamics`, and `value_head`. Keeping only the noise column fails the value check.

## Where this breaks

- **The actor exploits the model.** Dreamer's actor trains only in imagination, so it chases rewards the model invents. Short horizons and fresh real data limit that.
- **Value equivalence holds for the sets you chose.** An improving policy reaches situations the model never saw, and the backups can then disagree.
- **Scale matters for pUCT.** With rewards in the hundreds, $Q$ swamps the exploration term. Normalize $Q$ inside the tree, as MuZero does.

## Further reading

The following papers cover world models, MuZero, and value equivalence:

- [Hafner, Lillicrap, Ba, and Norouzi (2020), Dream to control: learning behaviors by latent imagination](https://arxiv.org/abs/1912.01603) introduces Dreamer, with lambda-returns over imagined latent trajectories.
- [Schrittwieser et al. (2020), Mastering Atari, Go, chess and shogi by planning with a learned model](https://arxiv.org/abs/1911.08265) introduces MuZero.
- [Grimm, Barreto, Singh, and Silver (2020), The value equivalence principle for model-based reinforcement learning](https://arxiv.org/abs/2011.03506) shows why matching Bellman backups is enough.

## The exercise

You write the functions in the order the checks call them: `lambda_returns`, `bellman_backup`, `value_equivalence_loss`, `puct_scores`, `mcts`, `encode`, `latent_step`, `unroll_value`, `k_step_value_error`, and `learn_latent_model`.

The checks match the lambda-returns, including the continuation cut, and they accept a wrong transition matrix whose backups agree. They replay the four-simulation trace and the grid-world search. They match the latent unrolls to 4.5 and 5.125, and they check that `learn_latent_model` brings the $k$-step value error to about 0. A noise-only encoder does not.
