---
title: Meta-RL and multi-task RL
slug: 050-meta-rl
part: "Part 7 — Models, exploration, and fixed data"
summary: Learn a policy that adapts to a new task within a few episodes, by recurrence (RL^2), by Bayes' rule, or by one gradient step (MAML).
minutes: 40
gpu: false
objectives:
  - Tell multi-task RL from meta-RL, and say why meta-RL is a partially observable problem over tasks.
  - Build the RL^2 input from the previous action, reward, and done flag, and say why the hidden state resets only at a trial boundary.
  - Compute the Bayes posterior over a family of bandit tasks, and the exact values of the Bayes-optimal, Thompson, and fixed policies.
  - Differentiate one MAML adaptation step, name the second-order term, and say what first-order MAML drops.
  - Say when meta-learning helps and when it does not.
prereqs:
  - 005-thompson-sampling
  - 031-the-policy-gradient
  - 045-partial-observability-and-memory
  - 046-exploration-bonuses
exercise: 050-meta-rl
---

# Meta-RL and multi-task RL

Every morning you walk into a casino you have never seen. It has two machines, and each casino is rigged in one of a few ways. You do not know which today, and you get a handful of pulls before you leave. A policy trained on one fixed casino is useless here. A policy trained on thousands of casinos can use its first pulls to work out which casino it is in, then play for it.

That skill is *meta-reinforcement learning*, or learning to learn. The agent trains on a whole family of tasks, and what it learns is a procedure that finds the answer to a new task quickly. This chapter shows three ways to build that procedure, and computes exactly how good the best one can be on a small casino.

## What you already know

- [Thompson sampling](/lessons/005-thompson-sampling) keeps a belief about each arm, updates it with Bayes' rule, and acts on a draw from the belief.
- The [policy gradient](/lessons/031-the-policy-gradient) moves the parameters $\theta$ uphill on the expected return $J(\theta)$.
- A [recurrent policy](/lessons/045-partial-observability-and-memory) carries a hidden state that summarizes the past.
- [Posterior sampling](/lessons/046-exploration-bonuses) explores by acting as if one sampled world were the true one.

## One task, many tasks, and a new task

| Setup | Training | At test time | The agent must |
|---|---|---|---|
| Single-task RL | One task | The same task | Solve it |
| Multi-task RL | A list of tasks | One of them, announced | Run the policy for that task |
| Meta-RL | Tasks drawn from a distribution | A new task, not announced | Learn it within a few episodes |

In multi-task RL the policy gets the task identity as an input, so it never learns anything at test time. In meta-RL the task is new, so there is nothing to look up. The agent gets a *trial*, a fixed budget of episodes on one task, and must spend the start finding out which task it faces and the rest exploiting that. The objective is the expected return of the whole trial, averaged over tasks $\tau$ from a distribution $p(\tau)$:

$$
\max_\theta \; \mathbb{E}_{\tau \sim p(\tau)} \; \mathbb{E}\Big[\sum_{t=0}^{H-1} r_t \;\Big|\; \pi_\theta, \tau\Big]
$$

Read it as: score the policy on a full trial, including the pulls it spends learning, so the objective tells the agent how much to explore. The chapters on [goal-conditioned RL](/lessons/048-goal-conditioned-rl-and-her) and on [options](/lessons/049-options-and-hierarchical-rl) meet a related setup where the task arrives as an input. Meta-RL is the case where it must be inferred.

## A POMDP in which the task is the hidden state

Within a trial the task is fixed but unseen. That is the partially observable problem of [Partial observability and memory](/lessons/045-partial-observability-and-memory), with a hidden part that never changes. The exact summary of the past is a *belief*, a probability for each task given everything seen, and a policy that acts on the belief is optimal for the trial. A recurrent network learns a belief-like summary (RL^2), Bayes' rule computes it exactly, and MAML skips it.

The running example is a casino of three kinds. Each row gives the payout probability of each arm and how often the kind occurs.

| Kind | Arm 0 pays | Arm 1 pays | Prior |
|---|---|---|---|
| A | 0.8 | 0.2 | 0.5 |
| B | 0.4 | 0.6 | 0.2 |
| C | 0.2 | 0.8 | 0.3 |

## RL^2: let the hidden state do the learning

Duan and others (2016) and Wang and others (2016) had the same idea: use a recurrent policy, and train it with an ordinary policy-gradient method on trial returns. The change is in what the network reads. At step $t$ it sees the observation $o_t$ and also the previous action $a_{t-1}$, the previous reward $r_{t-1}$, and a flag $d_{t-1}$ that is 1 when the previous step ended an episode.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="A recurrent policy unrolled over five steps of one trial. Each step reads the observation and the previous action, reward, and done flag. The hidden state passes along every step, across the episode boundary, and starts at zero only at the trial start">
<defs><marker id="c048-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<rect x="40" y="100" width="80" height="50" rx="6" class="fill-soft-b stroke-muted"/>
<text x="80" y="130" class="fig-bold" text-anchor="middle">GRU</text>
<line x1="80" y1="206" x2="80" y2="152" class="stroke-a" marker-end="url(#c048-arrow)"/>
<text x="80" y="222" class="fig-small" text-anchor="middle">o0, a∅</text>
<text x="80" y="238" class="fig-small" text-anchor="middle">r0, d0</text>
<line x1="80" y1="98" x2="80" y2="62" class="stroke-b" marker-end="url(#c048-arrow)"/>
<text x="80" y="50" class="fig-small" text-anchor="middle">action a0</text>
<line x1="122" y1="125" x2="148" y2="125" class="stroke-ink" marker-end="url(#c048-arrow)"/>
<rect x="150" y="100" width="80" height="50" rx="6" class="fill-soft-b stroke-muted"/>
<text x="190" y="130" class="fig-bold" text-anchor="middle">GRU</text>
<line x1="190" y1="206" x2="190" y2="152" class="stroke-a" marker-end="url(#c048-arrow)"/>
<text x="190" y="222" class="fig-small" text-anchor="middle">o1, a0</text>
<text x="190" y="238" class="fig-small" text-anchor="middle">r0, d0</text>
<line x1="190" y1="98" x2="190" y2="62" class="stroke-b" marker-end="url(#c048-arrow)"/>
<text x="190" y="50" class="fig-small" text-anchor="middle">action a1</text>
<line x1="232" y1="125" x2="258" y2="125" class="stroke-ink" marker-end="url(#c048-arrow)"/>
<rect x="260" y="100" width="80" height="50" rx="6" class="fill-soft-b stroke-muted"/>
<text x="300" y="130" class="fig-bold" text-anchor="middle">GRU</text>
<line x1="300" y1="206" x2="300" y2="152" class="stroke-a" marker-end="url(#c048-arrow)"/>
<text x="300" y="222" class="fig-small" text-anchor="middle">o2, a1</text>
<text x="300" y="238" class="fig-small" text-anchor="middle">r1, d1</text>
<line x1="300" y1="98" x2="300" y2="62" class="stroke-b" marker-end="url(#c048-arrow)"/>
<text x="300" y="50" class="fig-small" text-anchor="middle">action a2</text>
<line x1="342" y1="125" x2="368" y2="125" class="stroke-ink" marker-end="url(#c048-arrow)"/>
<rect x="370" y="100" width="80" height="50" rx="6" class="fill-soft-b stroke-muted"/>
<text x="410" y="130" class="fig-bold" text-anchor="middle">GRU</text>
<line x1="410" y1="206" x2="410" y2="152" class="stroke-a" marker-end="url(#c048-arrow)"/>
<text x="410" y="222" class="fig-small" text-anchor="middle">o3, a2</text>
<text x="410" y="238" class="fig-small" text-anchor="middle">r2, d2</text>
<line x1="410" y1="98" x2="410" y2="62" class="stroke-b" marker-end="url(#c048-arrow)"/>
<text x="410" y="50" class="fig-small" text-anchor="middle">action a3</text>
<line x1="452" y1="125" x2="478" y2="125" class="stroke-ink" marker-end="url(#c048-arrow)"/>
<rect x="480" y="100" width="80" height="50" rx="6" class="fill-soft-b stroke-muted"/>
<text x="520" y="130" class="fig-bold" text-anchor="middle">GRU</text>
<line x1="520" y1="206" x2="520" y2="152" class="stroke-a" marker-end="url(#c048-arrow)"/>
<text x="520" y="222" class="fig-small" text-anchor="middle">o4, a3</text>
<text x="520" y="238" class="fig-small" text-anchor="middle">r3, d3</text>
<line x1="520" y1="98" x2="520" y2="62" class="stroke-b" marker-end="url(#c048-arrow)"/>
<text x="520" y="50" class="fig-small" text-anchor="middle">action a4</text>
<line x1="355" y1="90" x2="355" y2="200" class="stroke-bad dashed"/>
<text x="355" y="176" class="fig-small" text-anchor="middle">episode</text>
<text x="355" y="192" class="fig-small" text-anchor="middle">ends</text>
<text x="40" y="82" class="fig-small">h = 0</text>
<text x="300" y="22" class="fig-bold" text-anchor="middle">one trial: the hidden state is not reset at the episode end</text>
</svg>
Figure 1. A recurrent policy over one trial. The previous action, reward, and done flag are the only way the network can see what its last action did. The hidden state runs on across the episode boundary and starts at zero only when the trial begins.
```

A plain observation does not say what happened. With the extra inputs, the hidden state can hold "arm 0 paid twice and failed once", a sufficient statistic for the casino. The hidden state resets at a trial boundary and not at an episode boundary, because a reset after every episode would erase what the network learned about the task just when it could use it.

Why does this count as learning? The weights are fixed while a trial runs. The *outer loop*, the policy gradient, changes them over many trials. The *inner loop* is the recurrence itself, which can implement any update rule, including Bayes' rule. One input row, in numpy:

```python
x = np.zeros(n_obs + n_actions + 2)
x[obs] = 1.0                                   # one-hot observation
if prev_action is not None:                    # None on the first step of a trial
    x[n_obs + prev_action] = 1.0               # one-hot previous action
x[n_obs + n_actions] = prev_reward
x[n_obs + n_actions + 1] = float(done)
```

## Task inference is Bayes' rule

On the three-kind casino you can skip the network and compute what a perfect learner believes. Start from the prior $b_k$ for each kind $k$. For each observed pull, multiply by its likelihood and divide by the sum:

$$
b'_k = \frac{b_k \, \ell_k}{\sum_j b_j \, \ell_j}, \qquad
\ell_k = \begin{cases} \theta_{k,a} & \text{if the pull paid 1} \\ 1 - \theta_{k,a} & \text{if it paid 0} \end{cases}
$$

Read it as: each kind gets credit in proportion to how well it predicted what happened. Pull arm 0 and win. The likelihoods are $0.8$, $0.4$, and $0.2$, so the weights $b_k \ell_k$ are $0.4$, $0.08$, and $0.06$, which add to $0.54$, and the posterior is $(0.741, 0.148, 0.111)$. Add one more win and one loss on arm 0, and the posterior is $(0.690, 0.207, 0.103)$. A posterior that drops the prior starts from $(1/3, 1/3, 1/3)$ and answers $(0.5, 0.375, 0.125)$. In numpy:

```python
like = thetas[:, arm] if reward == 1 else 1.0 - thetas[:, arm]
post = post * like / (post * like).sum()
```

## Three ways to act on the posterior

Compare three policies over a trial of $H$ pulls, with a fourth number as a ceiling. A *fixed* policy pulls the best prior-mean arm $H$ times. *Thompson sampling* draws a kind from the posterior and pulls that kind's best arm. The *Bayes-optimal* policy picks, at every step, the arm that maximizes the expected reward of the rest of the trial, including what the pull teaches. The *oracle* knows the kind before the first pull, and no learner can reach it.

Write $m_a(b) = \sum_k b_k \theta_{k,a}$ for the chance that arm $a$ pays under belief $b$. With $h$ pulls left, the Bayes-optimal value obeys a recursion on the belief:

$$
V_h(b) = \max_a \Big[ m_a(b) + m_a(b)\, V_{h-1}(b'_{a,1}) + \big(1 - m_a(b)\big)\, V_{h-1}(b'_{a,0}) \Big], \qquad V_0 = 0
$$

Read it as: the reward now, plus the value after a win weighted by the chance of a win, plus the value after a loss weighted by the chance of a loss. The belief is the state of an MDP, so this is dynamic programming over beliefs. Thompson sampling uses the same recursion with the max replaced by an average over arms, weighted by the posterior mass of the kinds whose best arm that is.

```figure
<svg viewBox="0 0 600 280" role="img" aria-label="Expected return against horizon H on the bandit family: the oracle is highest, then the Bayes-optimal adaptive policy, then Thompson sampling, then a fixed arm. At H = 3 they are 2.28, 1.85, 1.69, and 1.62">
<line x1="70" y1="230" x2="470" y2="230" class="stroke-muted"/>
<line x1="70" y1="230.0" x2="470" y2="230.0" class="stroke-grid"/>
<text x="62" y="234.0" text-anchor="end" class="fig-small">0</text>
<line x1="70" y1="170.6" x2="470" y2="170.6" class="stroke-grid"/>
<text x="62" y="174.6" text-anchor="end" class="fig-small">1</text>
<line x1="70" y1="111.2" x2="470" y2="111.2" class="stroke-grid"/>
<text x="62" y="115.2" text-anchor="end" class="fig-small">2</text>
<line x1="70" y1="51.9" x2="470" y2="51.9" class="stroke-grid"/>
<text x="62" y="55.9" text-anchor="end" class="fig-small">3</text>
<text x="70.0" y="248" text-anchor="middle" class="fig-small">1</text>
<text x="203.3" y="248" text-anchor="middle" class="fig-small">2</text>
<text x="336.7" y="248" text-anchor="middle" class="fig-small">3</text>
<text x="470.0" y="248" text-anchor="middle" class="fig-small">4</text>
<text x="270" y="270" text-anchor="middle" class="fig-small">horizon H (pulls in one trial)</text>
<polyline points="70.0,184.9 203.3,139.8 336.7,94.6 470.0,49.5" fill="none" class="stroke-ink" stroke-width="2.5"/>
<circle cx="70.0" cy="184.9" r="3.5" class="fill-ink"/>
<circle cx="203.3" cy="139.8" r="3.5" class="fill-ink"/>
<circle cx="336.7" cy="94.6" r="3.5" class="fill-ink"/>
<circle cx="470.0" cy="49.5" r="3.5" class="fill-ink"/>
<polyline points="70.0,197.9 203.3,159.5 336.7,120.4 470.0,79.1" fill="none" class="stroke-a" stroke-width="2.5"/>
<circle cx="70.0" cy="197.9" r="3.5" class="fill-a"/>
<circle cx="203.3" cy="159.5" r="3.5" class="fill-a"/>
<circle cx="336.7" cy="120.4" r="3.5" class="fill-a"/>
<circle cx="470.0" cy="79.1" r="3.5" class="fill-a"/>
<polyline points="70.0,200.3 203.3,166.1 336.7,129.5 470.0,91.0" fill="none" class="stroke-b" stroke-width="2.5"/>
<circle cx="70.0" cy="200.3" r="3.5" class="fill-b"/>
<circle cx="203.3" cy="166.1" r="3.5" class="fill-b"/>
<circle cx="336.7" cy="129.5" r="3.5" class="fill-b"/>
<circle cx="470.0" cy="91.0" r="3.5" class="fill-b"/>
<polyline points="70.0,197.9 203.3,165.9 336.7,133.8 470.0,101.8" fill="none" class="stroke-c" stroke-width="2.5"/>
<circle cx="70.0" cy="197.9" r="3.5" class="fill-c"/>
<circle cx="203.3" cy="165.9" r="3.5" class="fill-c"/>
<circle cx="336.7" cy="133.8" r="3.5" class="fill-c"/>
<circle cx="470.0" cy="101.8" r="3.5" class="fill-c"/>
<text x="480" y="53.5" class="fig-small">oracle 3.04</text>
<text x="480" y="83.1" class="fig-small">Bayes-optimal 2.54</text>
<text x="480" y="95.0" class="fig-small">Thompson 2.34</text>
<text x="480" y="107.8" class="fig-small">fixed arm 2.16</text>
</svg>
Figure 2. Exact expected return of a trial on the three-kind casino. Bayes-optimal play stays above Thompson sampling at every horizon, and the oracle sits above both.
```

| Horizon $H$ | Fixed arm | Thompson | Bayes-optimal | Oracle |
|---|---|---|---|---|
| 1 | 0.5400 | 0.5000 | 0.5400 | 0.7600 |
| 2 | 1.0800 | 1.0758 | 1.1880 | 1.5200 |
| 3 | 1.6200 | 1.6931 | 1.8456 | 2.2800 |
| 4 | 2.1600 | 2.3408 | 2.5416 | 3.0400 |

With one pull there is nothing to learn for, so Bayes-optimal play equals the fixed policy, while Thompson sampling plays the arm with the lower mean half the time. At $H = 2$ it is still a little below the fixed arm, since it pays for exploring and has no time to profit. The gap between Bayes-optimal play and Thompson sampling is the price of not valuing information: Thompson sampling explores by chance, while the recursion explores when the information is worth the pull.

> [!KEY]
> Meta-RL on a family of tasks is a POMDP whose hidden state is the task. The Bayes-optimal adaptive policy acts on the posterior and prices exploration inside the trial. RL^2 asks a recurrent network to learn that policy from experience.

## MAML: learn a starting point, adapt by gradient

RL^2 learns inside the forward pass. *Model-agnostic meta-learning* (MAML; Finn, Abbeel, and Levine, 2017) learns inside gradient descent: it looks for parameters $\theta$ from which one policy-gradient step on a new task already works well. For each task $i$, the *inner step* adapts the parameters, and the *outer objective* scores the adapted parameters:

$$
\theta_i' = \theta + \alpha \, \nabla J_i(\theta), \qquad
\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} J_i(\theta_i')
$$

To raise $\mathcal{L}$, differentiate through the inner step. The adapted parameters depend on $\theta$ twice, directly and through the gradient, so the chain rule gives

$$
\nabla_\theta J_i(\theta_i') = \big(I + \alpha \, \nabla^2 J_i(\theta)\big)\, \nabla J_i(\theta_i')
$$

where $\nabla^2 J_i$ is the Hessian, the matrix of second derivatives. The part $\alpha \nabla^2 J_i(\theta)\,\nabla J_i(\theta_i')$ is the *second-order term*. *First-order MAML* drops it and uses $\nabla J_i(\theta_i')$ alone. That is cheaper but biased, and its error is exactly $\alpha$ times the mean of the second-order terms.

To make the numbers checkable, use a task with closed forms. The policy picks an action $a \sim \mathcal{N}(\theta, \sigma^2 I)$ in $d$ dimensions, and the reward is a bump $\exp(-\lVert a - c \rVert^2 / (2 w^2))$ around a target $c$ of width $w$. Averaging over the action gives another bump, with $s^2 = w^2 + \sigma^2$:

$$
J(\theta) = \left(\frac{w^2}{s^2}\right)^{d/2} \exp\!\left(-\frac{\lVert \theta - c \rVert^2}{2 s^2}\right), \quad
\nabla J = -J\,\frac{\theta - c}{s^2}, \quad
\nabla^2 J = J\left(\frac{(\theta - c)(\theta - c)^\top}{s^4} - \frac{I}{s^2}\right)
$$

Work one task in one dimension: $c = 1$, $w = 0.5$, $\sigma = 0.5$ (so $s^2 = 0.5$), $\theta = 0$, and $\alpha = 1$. Then $J(0) = 0.2601$, $\nabla J(0) = 0.5203$, and $\nabla^2 J(0) = 0.5203$. The inner step moves to $\theta' = 0.5203$, where $J(\theta') = 0.5617$ and $\nabla J(\theta') = 0.5390$. The full outer gradient is $(1 + 0.5203) \times 0.5390 = 0.8194$, and first-order MAML gives $0.5390$. A finite difference of $J(\theta')$ with respect to $\theta$ gives $0.8194$, and the second-order term, $0.2804$, accounts for the whole gap.

Now train on three tasks in two dimensions, with targets $(1.2, 0)$, $(-0.6, 1)$, and $(-0.6, -1)$, widths $0.5$, $0.8$, and $1.2$, and $\alpha = 4$. Start from the best *multi-task* initialization, the point that maximizes the mean return with no adaptation, and run 150 steps of gradient ascent on $\mathcal{L}$.

```figure
<svg viewBox="0 0 600 280" role="img" aria-label="Reward after one adaptation step on each of three tasks and on average, starting from the multi-task initialization and from the MAML initialization. MAML trades a worse average before adapting for a better average after">
<line x1="60" y1="220" x2="560" y2="220" class="stroke-muted"/>
<line x1="60" y1="220.0" x2="560" y2="220.0" class="stroke-grid"/>
<text x="52" y="224.0" text-anchor="end" class="fig-small">0</text>
<line x1="60" y1="175.0" x2="560" y2="175.0" class="stroke-grid"/>
<text x="52" y="179.0" text-anchor="end" class="fig-small">0.25</text>
<line x1="60" y1="130.0" x2="560" y2="130.0" class="stroke-grid"/>
<text x="52" y="134.0" text-anchor="end" class="fig-small">0.5</text>
<line x1="60" y1="85.0" x2="560" y2="85.0" class="stroke-grid"/>
<text x="52" y="89.0" text-anchor="end" class="fig-small">0.75</text>
<line x1="60" y1="40.0" x2="560" y2="40.0" class="stroke-grid"/>
<text x="52" y="44.0" text-anchor="end" class="fig-small">1</text>
<rect x="80" y="208.0" width="36" height="12.0" class="fill-b"/>
<text x="98" y="203.0" text-anchor="middle" class="fig-small">0.07</text>
<rect x="122" y="152.2" width="36" height="67.8" class="fill-a"/>
<text x="140" y="147.2" text-anchor="middle" class="fig-small">0.38</text>
<text x="119" y="238" text-anchor="middle" class="fig-small">task 1</text>
<rect x="200" y="146.9" width="36" height="73.1" class="fill-b"/>
<text x="218" y="141.9" text-anchor="middle" class="fig-small">0.41</text>
<rect x="242" y="96.4" width="36" height="123.6" class="fill-a"/>
<text x="260" y="91.4" text-anchor="middle" class="fig-small">0.69</text>
<text x="239" y="238" text-anchor="middle" class="fig-small">task 2</text>
<rect x="320" y="69.1" width="36" height="150.9" class="fill-b"/>
<text x="338" y="64.1" text-anchor="middle" class="fig-small">0.84</text>
<rect x="362" y="75.1" width="36" height="144.9" class="fill-a"/>
<text x="380" y="70.1" text-anchor="middle" class="fig-small">0.80</text>
<text x="359" y="238" text-anchor="middle" class="fig-small">task 3</text>
<rect x="440" y="141.3" width="36" height="78.7" class="fill-b"/>
<text x="458" y="136.3" text-anchor="middle" class="fig-small">0.44</text>
<rect x="482" y="107.9" width="36" height="112.1" class="fill-a"/>
<text x="500" y="102.9" text-anchor="middle" class="fig-small">0.62</text>
<text x="479" y="238" text-anchor="middle" class="fig-small">mean</text>
<rect x="70" y="258" width="12" height="12" class="fill-b"/><text x="88" y="268" class="fig-small">multi-task initialization</text>
<rect x="290" y="258" width="12" height="12" class="fill-a"/><text x="308" y="268" class="fig-small">MAML initialization</text>
</svg>
Figure 3. Reward after one adaptation step, on each task and on average. The MAML initialization is worse on average before adapting (0.32 against 0.36) and better after (0.62 against 0.44), because it sits where one step helps every task.
```

The meta-learned start is a worse policy before adapting, and it is worth more because of what one step does from it. That is what it means to optimize for adaptability. In this run, first-order MAML from the same start reaches $0.660$, above the exact method's $0.623$, because the objective has several local peaks and a biased direction can land on a better one. It is still not the gradient of the objective you named.

In real RL the inner gradient is estimated from sampled trajectories, which depend on $\theta$ too, so differentiating the estimate drops terms. DiCE (Foerster and others, 2018) and ProMP (Rothfuss and others, 2018) put them back. PEARL (Rakelly and others, 2019) and VariBAD (Zintgraf and others, 2020) take a third route: an encoder maps recent experience to a latent task variable that the policy reads. PEARL samples the latent from a learned posterior and reuses off-policy data, and VariBAD learns a belief that approximates the Bayes-optimal policy above.

## When meta-learning helps

Meta-learning pays when tasks share structure that a few samples can pin down. In the casino a few pulls tell three kinds apart, and the gain over a fixed arm in Figure 2 grows with the horizon. If every machine were rigged at random, the trial would reveal nothing transferable. For MAML, far-apart targets leave no starting point that is one step from all of them, while clustered targets let one step finish the job.

## Build it step by step

> [!RECAP]
> - The RL^2 input is the observation plus the previous action, reward, and done flag. The hidden state resets only when a trial begins.
> - The posterior is prior times likelihood, normalized. Dropping the prior breaks it.
> - The Bayes-optimal value is dynamic programming over beliefs.
> - The MAML outer gradient is $(I + \alpha \nabla^2 J_i)\,\nabla J_i(\theta_i')$ averaged over tasks. First-order MAML drops the Hessian factor.

The checks test the functions in this order.

1. **`rl2_inputs(obs, prev_action, prev_reward, done, n_obs, n_actions)`.** Fill four slots of a zero vector. Treat `prev_action=None` as an all-zero block.
2. **`rl2_trial_inputs(...)`.** Row $t$ uses `actions[t - 1]`, `rewards[t - 1]`, and `dones[t - 1]`. Using index $t$ leaks an action the policy has not chosen yet.
3. **`posterior_update` and `task_posterior`.** Multiply, then normalize. Start from `prior`, not from a uniform vector.
4. **`bayes_optimal_value(prior, thetas, H)`.** Recurse on the posterior, and return 0 when `H == 0`.
5. **`thompson_value`, `fixed_policy_value`, and `oracle_value`.** For Thompson sampling, the probability of each arm is the posterior mass of the tasks whose best arm it is.
6. **`task_objective`, `task_grad`, and `task_hessian`.** The three closed forms.
7. **`adapt`, `maml_objective`, and `maml_grad`.** Multiply `eye + alpha * task_hessian(theta, t)` by `task_grad(adapted, t)`, with the Hessian taken at `theta`, not at the adapted point.
8. **`fomaml_grad`, `maml_step`, and `maml_train`.** The first-order gradient, then one ascent step of size `beta`, repeated.

## Where this breaks

**Meta-overfitting.** A policy trained on too few tasks memorizes them, and a task outside the training distribution gets a confidently wrong response.

**Exploration inside the trial.** With sparse reward the agent must explore to infer the task, which a gradient-trained recurrent policy learns badly.

## Further reading

- [Duan and others (2016), RL^2: Fast reinforcement learning via slow reinforcement learning](https://arxiv.org/abs/1611.02779). The recurrent policy with previous action, reward, and done flag.
- [Wang and others (2016), Learning to reinforcement learn](https://arxiv.org/abs/1611.05763). The same idea, with the outer loop as slow learning and the recurrence as fast learning.
- [Finn, Abbeel, and Levine (2017), Model-agnostic meta-learning for fast adaptation of deep networks](https://arxiv.org/abs/1703.03400). MAML, with its policy-gradient version and first-order approximation.
- [Rakelly and others (2019), Efficient off-policy meta-reinforcement learning via probabilistic context variables](https://arxiv.org/abs/1903.08254). PEARL.
- [Zintgraf and others (2020), VariBAD: A very good method for Bayes-adaptive deep RL via meta-learning](https://arxiv.org/abs/1910.08348). A learned belief that approximates the Bayes-adaptive policy.
- [Foerster and others (2018), DiCE](https://arxiv.org/abs/1802.05098) and [Rothfuss and others (2018), ProMP](https://arxiv.org/abs/1810.06784). Second-order estimates for MAML in RL.
- Berkeley CS 285, [Lecture 24, "Multi-task RL"](https://rail.eecs.berkeley.edu/deeprlcourse/), Spring 2026 schedule.
- Stanford CS 330, [Deep Multi-Task and Meta Learning](https://cs330.stanford.edu/), "Optimization-based meta-learning" lecture, Fall 2023 schedule.

## The exercise

You write the functions above, from the recurrent policy's input to `maml_train`. The checks verify the input encoding on hand-built trials and the posterior against the worked numbers, with a non-uniform prior. They compare `bayes_optimal_value` with brute force over all 32 policies at $H = 2$, and `thompson_value` with a 20,000-run simulation. For MAML, they check `maml_grad` against a finite difference of the post-adaptation objective, the gap to `fomaml_grad` against the exact Hessian term, and that training lifts the post-adaptation reward by more than $0.1$.
