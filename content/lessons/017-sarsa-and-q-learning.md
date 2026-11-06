---
title: SARSA and Q-learning
slug: 017-sarsa-and-q-learning
part: "Part 3 — Learning from experience"
summary: Turn the TD update into control with SARSA and Q-learning, and see why one next-action choice sends them along different paths on the cliff.
minutes: 35
gpu: false
objectives:
  - Update an action value toward a one-step bootstrapped target.
  - Say what differs between the SARSA and Q-learning targets, and why that makes one on-policy and the other off-policy.
  - Explain why SARSA learns a safer path than Q-learning on the cliff walk.
  - Say what Q-learning needs in order to converge to the optimal action values.
prereqs:
  - 014-monte-carlo-control
  - 016-temporal-difference-learning
exercise: 017-sarsa-and-q-learning
---

# SARSA and Q-learning

A robot walks along the top of a cliff. The shortest path to the goal runs right along the edge, but the robot explores: one step in ten, it moves in a random direction. Near the edge, a random step is a fall.

So what is the edge worth? If the robot asks "what is the best I could do from here?", the edge is great. If it asks "what will I really get, given that I sometimes stumble?", the edge is dangerous. Both are fair questions, and the two answers lead to different paths. This chapter builds the two learning rules that ask them, SARSA and Q-learning, and shows that they differ by one symbol.

## What you already know

- TD(0) moves an estimate a fraction $\alpha$ toward $r_t + \gamma V(s_{t+1})$ after every step (see [Temporal-difference learning](/lessons/016-temporal-difference-learning)).
- Without a model, only action values $Q(s, a)$ tell you which action is best, because $V$ alone does not say what an action leads to (see [Monte Carlo control](/lessons/014-monte-carlo-control)).
- An $\varepsilon$-greedy policy picks a random action with probability $\varepsilon$, and a greedy action otherwise.
- A terminal state has value zero.
- `rlenvs.cliff_walking()` is a 4 by 12 grid. The best path from the start is worth exactly $-13$.

## Put the TD update on action values

The TD(0) update works the same on $Q(s, a)$ as it does on $V(s)$. The target is the reward plus the discounted value of what comes next, and $Q(s_t, a_t)$ moves a fraction $\alpha$ toward it. The only new question is what "what comes next" means. After landing in $s_{t+1}$, the agent has a whole row of action values $Q(s_{t+1}, \cdot)$, and the target must use one number from that row.

There are two natural choices. You can use the value of the action the agent will actually take next, or you can use the value of the best action. Take a next state with $Q(s', \cdot) = (2, 5)$, $r = -1$, $\gamma = 1$, and $\alpha = 0.5$, and suppose the agent's next action is action 0, an exploratory one.

```figure
<svg viewBox="0 0 580 230" role="img" aria-label="Two bars for the next state's action values and the targets SARSA and Q-learning build from them">
<defs><marker id="c017-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<text x="20.0" y="28.0" class="fig-bold" text-anchor="start">Next state s′: Q(s′, ·) = (2, 5). The policy happens to take action 0.</text>
<rect x="215.0" y="152.0" width="60.0" height="48.0" class="fill-soft-a"/>
<rect x="215" y="152" width="60" height="48" class="stroke-a" fill="none"/>
<text x="245.0" y="144.0" class="fig-small" text-anchor="middle">Q = 2</text>
<text x="245.0" y="218.0" class="fig-small" text-anchor="middle">action 0</text>
<rect x="325.0" y="80.0" width="60.0" height="120.0" class="fill-soft-a"/>
<rect x="325" y="80" width="60" height="120" class="stroke-a" fill="none"/>
<text x="355.0" y="72.0" class="fig-small" text-anchor="middle">Q = 5</text>
<text x="355.0" y="218.0" class="fig-small" text-anchor="middle">action 1</text>
<line x1="195.0" y1="200.0" x2="400.0" y2="200.0" class="stroke-muted"/>
<rect x="10.0" y="100.0" width="170.0" height="44.0" class="fill-soft-d"/>
<text x="20.0" y="118.0" class="fig-bold" text-anchor="start">SARSA target</text>
<text x="20.0" y="134.0" class="fig-small" text-anchor="start">−1 + 2 = 1, so Q = 0.5</text>
<line x1="180.0" y1="128.0" x2="212.0" y2="160.0" class="stroke-d" marker-end="url(#c017-arrow)"/>
<rect x="410.0" y="60.0" width="160.0" height="44.0" class="fill-soft-b"/>
<text x="420.0" y="78.0" class="fig-bold" text-anchor="start">Q-learning target</text>
<text x="420.0" y="94.0" class="fig-small" text-anchor="start">−1 + 5 = 4, so Q = 2.0</text>
<line x1="410.0" y1="84.0" x2="378.0" y2="110.0" class="stroke-b" marker-end="url(#c017-arrow)"/>
</svg>
Figure 1. With Q(s, a) = 0, r = −1, γ = 1, and α = 0.5, SARSA moves Q(s, a) to 0.5 and Q-learning moves it to 2.0. Notice that the two targets differ only in which next value they read.
```

The first choice reads 2, which gives the target $-1 + 2 = 1$. The second reads 5, which gives the target $-1 + 5 = 4$. The same experience yields two different lessons, depending only on which number you read.

## SARSA uses the action you take

SARSA reads the value of the action the agent actually takes next. Its update uses five quantities, $(s_t, a_t, r_t, s_{t+1}, a_{t+1})$, which give it its name:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \bigl[r_t + \gamma\, Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)\bigr]
$$

Read it aloud: the target is the reward plus the discounted value of the next state and the action you will take there. On the example, the target is 1, so $Q(s, a)$ becomes $0 + 0.5(1 - 0) = 0.5$.

SARSA picks $a_{t+1}$ with its $\varepsilon$-greedy policy before it updates, and then takes that same action. Because the value it learns includes the exploratory steps, SARSA is *on-policy*: it learns the value of the policy it follows, exploration included.

```python
a2 = epsilon_greedy_action(Q[s2], epsilon, rng)
target = r + gamma * Q[s2, a2] * (not terminated)
Q[s, a] += alpha * (target - Q[s, a])
s, a = s2, a2
```

> [!TIP]
> Pick SARSA's next action before the update, and then take that same action. Choosing a second, fresh action to execute turns SARSA into a mix of two policies.

## Q-learning uses the best action

Q-learning changes one thing. Its target reads the best next action, whatever the agent does next:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \bigl[r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)\bigr]
$$

Read it aloud: the target is the reward plus the discounted value of acting greedily from the next state on. On the example, the target is 4, so $Q(s, a)$ becomes $0 + 0.5(4 - 0) = 2.0$. In code, `Q[s2, a2]` becomes `Q[s2].max()`.

Q-learning is *off-policy*: the agent behaves $\varepsilon$-greedily, but the values it learns belong to the greedy policy, which is a different policy. That separation is what lets Q-learning learn about the best behavior while it explores.

Tabular Q-learning converges to the optimal $Q^*$ with probability one under two conditions. Every state-action pair must be tried infinitely often, and the step sizes must shrink, with $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$. The first condition says the agent must keep exploring, because a value it never updates cannot improve. With a constant $\alpha$, as in the exercise, the estimates keep jittering, but the greedy policy often settles anyway.

## Walk the cliff both ways

The cliff walk is a 4 by 12 grid. You start at the bottom left, and the goal is the bottom right. The cells between them along the bottom row are the cliff. Every step pays $-1$. Stepping into the cliff pays $-100$ and sends you back to the start. The shortest path runs along the edge, right above the cliff, in 13 steps worth $-13$.

Train both methods for 500 episodes with $\alpha = 0.1$ and $\varepsilon = 0.1$, then follow each one's greedy policy from the start.

```figure
<svg viewBox="0 0 580 236" role="img" aria-label="The cliff walk grid with the greedy path of Q-learning along the cliff edge and of SARSA two rows above it">
<rect x="50.0" y="20.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="50" y="20" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="90.0" y="20.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="90" y="20" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="130.0" y="20.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="130" y="20" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="170.0" y="20.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="170" y="20" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="210.0" y="20.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="210" y="20" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="250.0" y="20.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="250" y="20" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="290.0" y="20.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="290" y="20" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="330.0" y="20.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="330" y="20" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="370.0" y="20.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="370" y="20" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="410.0" y="20.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="410" y="20" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="450.0" y="20.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="450" y="20" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="490.0" y="20.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="490" y="20" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="50.0" y="60.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="50" y="60" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="90.0" y="60.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="90" y="60" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="130.0" y="60.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="130" y="60" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="170.0" y="60.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="170" y="60" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="210.0" y="60.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="210" y="60" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="250.0" y="60.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="250" y="60" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="290.0" y="60.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="290" y="60" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="330.0" y="60.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="330" y="60" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="370.0" y="60.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="370" y="60" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="410.0" y="60.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="410" y="60" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="450.0" y="60.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="450" y="60" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="490.0" y="60.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="490" y="60" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="50.0" y="100.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="50" y="100" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="90.0" y="100.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="90" y="100" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="130.0" y="100.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="130" y="100" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="170.0" y="100.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="170" y="100" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="210.0" y="100.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="210" y="100" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="250.0" y="100.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="250" y="100" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="290.0" y="100.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="290" y="100" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="330.0" y="100.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="330" y="100" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="370.0" y="100.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="370" y="100" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="410.0" y="100.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="410" y="100" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="450.0" y="100.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="450" y="100" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="490.0" y="100.0" width="40.0" height="40.0" class="fill-sunk"/>
<rect x="490" y="100" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="50.0" y="140.0" width="40.0" height="40.0" class="fill-soft-c"/>
<rect x="50" y="140" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="90.0" y="140.0" width="40.0" height="40.0" class="fill-bad"/>
<rect x="90" y="140" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="130.0" y="140.0" width="40.0" height="40.0" class="fill-bad"/>
<rect x="130" y="140" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="170.0" y="140.0" width="40.0" height="40.0" class="fill-bad"/>
<rect x="170" y="140" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="210.0" y="140.0" width="40.0" height="40.0" class="fill-bad"/>
<rect x="210" y="140" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="250.0" y="140.0" width="40.0" height="40.0" class="fill-bad"/>
<rect x="250" y="140" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="290.0" y="140.0" width="40.0" height="40.0" class="fill-bad"/>
<rect x="290" y="140" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="330.0" y="140.0" width="40.0" height="40.0" class="fill-bad"/>
<rect x="330" y="140" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="370.0" y="140.0" width="40.0" height="40.0" class="fill-bad"/>
<rect x="370" y="140" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="410.0" y="140.0" width="40.0" height="40.0" class="fill-bad"/>
<rect x="410" y="140" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="450.0" y="140.0" width="40.0" height="40.0" class="fill-bad"/>
<rect x="450" y="140" width="40" height="40" class="stroke-grid" fill="none"/>
<rect x="490.0" y="140.0" width="40.0" height="40.0" class="fill-soft-c"/>
<rect x="490" y="140" width="40" height="40" class="stroke-grid" fill="none"/>
<text x="70.0" y="196.0" class="fig-small fig-bold" text-anchor="middle">S start</text>
<text x="510.0" y="196.0" class="fig-small fig-bold" text-anchor="middle">G goal</text>
<text x="290.0" y="196.0" class="fig-small" text-anchor="middle">red cells: the cliff, −100 and back to start</text>
<polyline points="70.0,165.0 70.0,125.0 110.0,125.0 150.0,125.0 190.0,125.0 230.0,125.0 270.0,125.0 310.0,125.0 350.0,125.0 390.0,125.0 430.0,125.0 470.0,125.0 510.0,125.0 510.0,165.0" class="stroke-b" fill="none" stroke-width="3"/>
<polyline points="70.0,155.0 70.0,115.0 70.0,75.0 70.0,35.0 110.0,35.0 150.0,35.0 190.0,35.0 230.0,35.0 270.0,35.0 310.0,35.0 350.0,35.0 390.0,35.0 430.0,35.0 470.0,35.0 510.0,35.0 510.0,75.0 510.0,115.0 510.0,155.0" class="stroke-d" fill="none" stroke-width="3"/>
<line x1="60.0" y1="214.0" x2="90.0" y2="214.0" class="stroke-b" stroke-width="3"/>
<text x="96.0" y="218.0" class="fig-small" text-anchor="start">Q-learning: 13 steps, value -13</text>
<line x1="310.0" y1="214.0" x2="340.0" y2="214.0" class="stroke-d" stroke-width="3"/>
<text x="346.0" y="218.0" class="fig-small" text-anchor="start">SARSA: 17 steps, value -17</text>
</svg>
Figure 2. Greedy paths after 500 episodes with α = 0.1 and ε = 0.1, seed 0. Notice that SARSA keeps away from the red cliff cells, because its own values include the exploratory slips that happen next to them.
```

The two paths come from the two targets:

- **Q-learning walks the edge.** Its target assumes the best action next, so an edge cell looks as good as the shortest path makes it. The greedy walker never slips, so the learned value, $-13$, is exact.
- **SARSA detours.** In the run above, it takes 17 steps, worth $-17$, and other seeds give 15 steps. Its target includes the next action it will really take, which is sometimes a random one. Next to the cliff, that random action is a fall worth $-100$, and those falls are inside SARSA's values. So the edge cells look dangerous to it, and it walks farther from the cliff.

The order of the two flips while they train, because both keep exploring.

```figure
<svg viewBox="0 0 580 290" role="img" aria-label="Training return per episode for Q-learning and SARSA on the cliff walk, averaged over five seeds">
<line x1="70.0" y1="230.0" x2="540.0" y2="230.0" class="stroke-grid"/>
<text x="62.0" y="234.0" class="fig-small" text-anchor="end">-100</text>
<line x1="70.0" y1="180.0" x2="540.0" y2="180.0" class="stroke-grid"/>
<text x="62.0" y="184.0" class="fig-small" text-anchor="end">-75</text>
<line x1="70.0" y1="130.0" x2="540.0" y2="130.0" class="stroke-grid"/>
<text x="62.0" y="134.0" class="fig-small" text-anchor="end">-50</text>
<line x1="70.0" y1="80.0" x2="540.0" y2="80.0" class="stroke-grid"/>
<text x="62.0" y="84.0" class="fig-small" text-anchor="end">-25</text>
<line x1="70.0" y1="30.0" x2="540.0" y2="30.0" class="stroke-grid"/>
<text x="62.0" y="34.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="230.0" x2="540.0" y2="230.0" class="stroke-muted"/>
<line x1="70.0" y1="230.0" x2="70.0" y2="235.0" class="stroke-muted"/>
<text x="70.0" y="250.0" class="fig-small" text-anchor="middle">0</text>
<line x1="164.0" y1="230.0" x2="164.0" y2="235.0" class="stroke-muted"/>
<text x="164.0" y="250.0" class="fig-small" text-anchor="middle">100</text>
<line x1="258.0" y1="230.0" x2="258.0" y2="235.0" class="stroke-muted"/>
<text x="258.0" y="250.0" class="fig-small" text-anchor="middle">200</text>
<line x1="352.0" y1="230.0" x2="352.0" y2="235.0" class="stroke-muted"/>
<text x="352.0" y="250.0" class="fig-small" text-anchor="middle">300</text>
<line x1="446.0" y1="230.0" x2="446.0" y2="235.0" class="stroke-muted"/>
<text x="446.0" y="250.0" class="fig-small" text-anchor="middle">400</text>
<line x1="540.0" y1="230.0" x2="540.0" y2="235.0" class="stroke-muted"/>
<text x="540.0" y="250.0" class="fig-small" text-anchor="middle">500</text>
<text x="305.0" y="272.0" text-anchor="middle">training episodes</text>
<text x="62.0" y="18.0" text-anchor="end">return</text>
<polyline points="79.4,230.0 80.3,230.0 81.3,230.0 82.2,230.0 83.2,230.0 84.1,230.0 85.0,230.0 86.0,230.0 86.9,230.0 87.9,230.0 88.8,230.0 89.7,230.0 90.7,230.0 91.6,230.0 92.6,230.0 93.5,230.0 94.4,230.0 95.4,230.0 96.3,230.0 97.3,230.0 98.2,230.0 99.1,230.0 100.1,230.0 101.0,230.0 102.0,230.0 102.9,230.0 103.8,230.0 104.8,230.0 105.7,230.0 106.7,230.0 107.6,230.0 108.5,230.0 109.5,230.0 110.4,230.0 111.4,230.0 112.3,229.6 113.2,230.0 114.2,230.0 115.1,230.0 116.1,230.0 117.0,211.4 117.9,225.7 118.9,230.0 119.8,230.0 120.8,230.0 121.7,230.0 122.6,230.0 123.6,230.0 124.5,230.0 125.5,230.0 126.4,230.0 127.3,230.0 128.3,230.0 129.2,230.0 130.2,230.0 131.1,230.0 132.0,218.5 133.0,214.5 133.9,204.9 134.9,215.2 135.8,215.2 136.7,191.0 137.7,192.1 138.6,196.8 139.6,210.1 140.5,217.0 141.4,197.5 142.4,189.7 143.3,189.0 144.3,182.8 145.2,201.5 146.1,190.2 147.1,186.0 148.0,180.8 149.0,170.0 149.9,181.6 150.8,188.7 151.8,188.2 152.7,178.6 153.7,173.9 154.6,149.9 155.5,151.9 156.5,156.4 157.4,151.9 158.4,155.4 159.3,150.6 160.2,153.0 161.2,163.2 162.1,167.4 163.1,168.8 164.0,190.7 164.9,193.0 165.9,187.4 166.8,189.2 167.8,187.0 168.7,185.1 169.6,186.2 170.6,171.4 171.5,171.0 172.5,166.7 173.4,156.7 174.3,163.2 175.3,181.6 176.2,182.4 177.2,183.1 178.1,171.0 179.0,167.8 180.0,174.0 180.9,173.5 181.9,177.7 182.8,163.0 183.7,161.8 184.7,142.6 185.6,148.0 186.6,147.8 187.5,164.7 188.4,162.6 189.4,155.0 190.3,157.8 191.3,149.8 192.2,150.2 193.1,139.5 194.1,134.5 195.0,125.6 196.0,126.8 196.9,108.5 197.8,107.5 198.8,116.8 199.7,109.1 200.7,116.8 201.6,127.5 202.5,127.4 203.5,128.3 204.4,131.6 205.4,124.9 206.3,124.4 207.2,119.9 208.2,116.5 209.1,115.3 210.1,106.5 211.0,100.8 211.9,110.1 212.9,119.2 213.8,119.2 214.8,123.2 215.7,127.5 216.6,136.4 217.6,139.1 218.5,140.0 219.5,139.7 220.4,139.5 221.3,137.7 222.3,137.4 223.2,137.5 224.2,132.9 225.1,127.7 226.0,119.4 227.0,131.9 227.9,135.2 228.9,152.9 229.8,162.0 230.7,153.2 231.7,161.1 232.6,159.5 233.6,166.2 234.5,166.9 235.4,174.6 236.4,153.2 237.3,144.6 238.3,130.8 239.2,125.4 240.1,126.3 241.1,116.5 242.0,113.2 243.0,115.7 243.9,119.6 244.8,115.0 245.8,116.3 246.7,124.5 247.7,133.5 248.6,128.7 249.5,127.2 250.5,119.4 251.4,132.5 252.4,140.6 253.3,136.4 254.2,136.4 255.2,140.0 256.1,131.9 257.1,121.9 258.0,126.2 258.9,136.7 259.9,148.6 260.8,134.2 261.8,120.4 262.7,120.4 263.6,115.5 264.6,118.9 265.5,131.6 266.5,128.2 267.4,119.9 268.3,110.3 269.3,100.8 270.2,107.0 271.2,106.3 272.1,110.0 273.0,113.8 274.0,114.8 274.9,105.9 275.9,105.7 276.8,109.3 277.7,113.0 278.7,113.7 279.6,108.6 280.6,122.2 281.5,123.2 282.4,122.9 283.4,117.1 284.3,122.3 285.3,134.8 286.2,134.5 287.1,138.1 288.1,142.1 289.0,145.3 290.0,132.4 290.9,131.3 291.8,128.0 292.8,124.2 293.7,122.8 294.7,118.2 295.6,122.7 296.5,114.5 297.5,114.6 298.4,111.0 299.4,119.3 300.3,123.9 301.2,122.7 302.2,135.4 303.1,131.6 304.1,131.8 305.0,131.0 305.9,130.8 306.9,121.9 307.8,129.5 308.8,121.8 309.7,117.5 310.6,121.9 311.6,130.3 312.5,131.2 313.5,121.6 314.4,113.5 315.3,122.1 316.3,127.4 317.2,127.7 318.2,122.0 319.1,116.9 320.0,117.1 321.0,94.8 321.9,97.7 322.9,97.6 323.8,98.0 324.7,97.1 325.7,95.6 326.6,87.0 327.6,95.7 328.5,100.0 329.4,99.6 330.4,104.4 331.3,107.9 332.3,133.2 333.2,136.8 334.1,136.6 335.1,132.7 336.0,150.8 337.0,142.4 337.9,146.4 338.8,141.8 339.8,141.6 340.7,133.4 341.7,108.1 342.6,108.0 343.5,104.3 344.5,117.0 345.4,98.4 346.4,97.8 347.3,93.4 348.2,102.4 349.2,106.7 350.1,110.5 351.1,110.5 352.0,115.0 352.9,115.3 353.9,102.6 354.8,102.5 355.8,120.0 356.7,124.1 357.6,137.5 358.6,146.0 359.5,150.6 360.5,155.1 361.4,158.9 362.3,162.6 363.3,171.2 364.2,171.2 365.2,166.4 366.1,167.2 367.0,145.0 368.0,127.9 368.9,116.1 369.9,111.7 370.8,99.0 371.7,103.2 372.7,94.4 373.6,94.9 374.6,86.4 375.5,77.2 376.4,77.7 377.4,86.4 378.3,85.4 379.3,89.8 380.2,90.0 381.1,81.9 382.1,82.0 383.0,87.0 384.0,82.9 384.9,83.0 385.8,91.4 386.8,99.3 387.7,112.5 388.7,112.5 389.6,112.7 390.5,116.3 391.5,116.1 392.4,114.9 393.4,114.7 394.3,123.3 395.2,118.4 396.2,119.1 397.1,126.5 398.1,122.1 399.0,121.5 399.9,118.5 400.9,118.8 401.8,124.0 402.8,140.4 403.7,140.3 404.6,144.8 405.6,136.4 406.5,120.2 407.5,124.8 408.4,141.9 409.3,136.8 410.3,149.3 411.2,140.1 412.2,123.7 413.1,115.0 414.0,106.9 415.0,98.0 415.9,93.5 416.9,89.2 417.8,76.7 418.7,76.4 419.7,72.6 420.6,89.5 421.6,93.6 422.5,105.7 423.4,114.2 424.4,113.8 425.3,117.8 426.3,138.6 427.2,146.8 428.1,147.0 429.1,138.2 430.0,125.8 431.0,130.0 431.9,131.0 432.8,131.0 433.8,152.4 434.7,152.7 435.7,140.6 436.6,144.8 437.5,169.6 438.5,173.3 439.4,177.4 440.4,168.8 441.3,160.2 442.2,155.8 443.2,148.1 444.1,152.2 445.1,156.4 446.0,139.6 446.9,115.0 447.9,111.1 448.8,102.5 449.8,102.7 450.7,102.9 451.6,107.3 452.6,107.0 453.5,111.5 454.5,98.8 455.4,103.6 456.3,120.4 457.3,124.8 458.2,124.8 459.2,128.7 460.1,124.1 461.0,116.3 462.0,108.0 462.9,107.9 463.9,107.4 464.8,102.8 465.7,93.8 466.7,98.1 467.6,103.0 468.6,112.2 469.5,120.7 470.4,132.6 471.4,127.7 472.3,123.7 473.3,123.7 474.2,136.2 475.1,127.8 476.1,119.5 477.0,114.6 478.0,109.6 478.9,127.8 479.8,131.5 480.8,144.2 481.7,135.4 482.7,153.2 483.6,149.1 484.5,153.5 485.5,153.8 486.4,162.3 487.4,171.2 488.3,153.6 489.2,137.4 490.2,129.2 491.1,128.8 492.1,111.5 493.0,120.4 493.9,128.6 494.9,136.5 495.8,137.0 496.8,119.8 497.7,119.0 498.6,131.5 499.6,131.4 500.5,131.7 501.5,131.3 502.4,122.6 503.3,114.6 504.3,136.2 505.2,127.4 506.2,127.4 507.1,119.3 508.0,111.0 509.0,106.9 509.9,110.9 510.9,115.2 511.8,115.9 512.7,115.8 513.7,94.2 514.6,94.1 515.6,111.8 516.5,111.7 517.4,116.4 518.4,120.6 519.3,129.5 520.3,130.4 521.2,133.9 522.1,130.4 523.1,126.2 524.0,142.8 525.0,134.3 525.9,147.8 526.8,150.9 527.8,159.7 528.7,152.1 529.7,155.4 530.6,146.9 531.5,150.5 532.5,146.6 533.4,135.0 534.4,130.7 535.3,121.6 536.2,126.2 537.2,121.7 538.1,116.2 539.1,116.8 540.0,116.8" class="stroke-b" fill="none"/>
<polyline points="79.4,230.0 80.3,230.0 81.3,230.0 82.2,230.0 83.2,230.0 84.1,230.0 85.0,230.0 86.0,230.0 86.9,230.0 87.9,230.0 88.8,230.0 89.7,230.0 90.7,230.0 91.6,230.0 92.6,230.0 93.5,230.0 94.4,230.0 95.4,230.0 96.3,230.0 97.3,230.0 98.2,230.0 99.1,230.0 100.1,230.0 101.0,230.0 102.0,230.0 102.9,230.0 103.8,230.0 104.8,230.0 105.7,230.0 106.7,230.0 107.6,230.0 108.5,230.0 109.5,230.0 110.4,230.0 111.4,230.0 112.3,230.0 113.2,230.0 114.2,230.0 115.1,230.0 116.1,230.0 117.0,230.0 117.9,230.0 118.9,230.0 119.8,230.0 120.8,230.0 121.7,230.0 122.6,230.0 123.6,230.0 124.5,230.0 125.5,229.1 126.4,221.0 127.3,213.0 128.3,196.6 129.2,181.9 130.2,188.5 131.1,184.1 132.0,182.1 133.0,161.0 133.9,163.9 134.9,156.0 135.8,164.5 136.7,161.6 137.7,175.1 138.6,175.3 139.6,176.6 140.5,173.0 141.4,170.3 142.4,167.2 143.3,168.9 144.3,169.4 145.2,160.0 146.1,155.2 147.1,146.4 148.0,147.3 149.0,132.3 149.9,144.4 150.8,152.7 151.8,148.9 152.7,153.0 153.7,149.8 154.6,148.0 155.5,148.6 156.5,145.2 157.4,142.6 158.4,145.0 159.3,132.3 160.2,120.6 161.2,130.9 162.1,125.0 163.1,129.8 164.0,134.4 164.9,142.0 165.9,158.8 166.8,157.3 167.8,153.6 168.7,153.0 169.6,156.3 170.6,159.4 171.5,160.8 172.5,173.7 173.4,176.4 174.3,165.4 175.3,150.0 176.2,152.0 177.2,151.6 178.1,157.4 179.0,158.1 180.0,143.5 180.9,136.7 181.9,119.7 182.8,111.1 183.7,120.8 184.7,114.5 185.6,113.3 186.6,113.8 187.5,107.2 188.4,110.8 189.4,115.6 190.3,116.1 191.3,120.1 192.2,134.9 193.1,130.4 194.1,129.4 195.0,133.2 196.0,134.3 196.9,141.8 197.8,132.6 198.8,126.8 199.7,130.8 200.7,133.7 201.6,132.8 202.5,127.8 203.5,137.4 204.4,133.7 205.4,140.9 206.3,133.8 207.2,133.4 208.2,134.1 209.1,128.2 210.1,119.0 211.0,105.7 211.9,105.4 212.9,105.3 213.8,108.8 214.8,99.0 215.7,98.0 216.6,101.9 217.6,100.5 218.5,101.4 219.5,101.1 220.4,99.7 221.3,99.1 222.3,89.6 223.2,89.3 224.2,88.8 225.1,89.0 226.0,84.4 227.0,94.1 227.9,97.9 228.9,99.5 229.8,99.9 230.7,99.2 231.7,98.8 232.6,98.8 233.6,107.8 234.5,113.0 235.4,113.4 236.4,109.0 237.3,104.6 238.3,108.8 239.2,113.8 240.1,114.1 241.1,113.1 242.0,109.3 243.0,101.2 243.9,94.5 244.8,94.3 245.8,89.6 246.7,89.5 247.7,87.6 248.6,95.0 249.5,95.6 250.5,96.0 251.4,94.7 252.4,98.7 253.3,99.0 254.2,106.6 255.2,105.5 256.1,106.0 257.1,106.5 258.0,93.5 258.9,93.2 259.9,98.6 260.8,99.1 261.8,95.0 262.7,95.0 263.6,91.3 264.6,91.8 265.5,91.6 266.5,87.1 267.4,87.5 268.3,86.8 269.3,81.9 270.2,82.5 271.2,81.7 272.1,85.9 273.0,81.1 274.0,80.5 274.9,84.3 275.9,88.6 276.8,87.9 277.7,87.4 278.7,87.2 279.6,86.1 280.6,85.1 281.5,81.8 282.4,81.9 283.4,86.0 284.3,90.0 285.3,90.5 286.2,90.5 287.1,94.7 288.1,95.1 289.0,99.0 290.0,104.8 290.9,107.4 291.8,107.5 292.8,107.9 293.7,100.3 294.7,100.6 295.6,100.2 296.5,100.0 297.5,104.8 298.4,100.9 299.4,99.9 300.3,96.0 301.2,96.3 302.2,92.9 303.1,92.6 304.1,87.2 305.0,87.4 305.9,83.4 306.9,77.8 307.8,81.4 308.8,77.4 309.7,77.5 310.6,77.3 311.6,76.0 312.5,74.8 313.5,74.6 314.4,74.4 315.3,74.0 316.3,73.9 317.2,70.0 318.2,74.4 319.1,74.1 320.0,78.6 321.0,78.4 321.9,79.0 322.9,79.6 323.8,83.5 324.7,83.6 325.7,83.2 326.6,82.6 327.6,82.2 328.5,86.4 329.4,81.8 330.4,82.1 331.3,82.0 332.3,81.8 333.2,77.4 334.1,78.1 335.1,78.2 336.0,78.1 337.0,73.4 337.9,73.5 338.8,78.2 339.8,77.9 340.7,76.9 341.7,77.2 342.6,78.1 343.5,77.2 344.5,81.7 345.4,81.7 346.4,81.9 347.3,76.8 348.2,71.6 349.2,71.9 350.1,76.2 351.1,76.0 352.0,75.6 352.9,76.1 353.9,72.6 354.8,76.9 355.8,76.9 356.7,77.5 357.6,81.4 358.6,81.1 359.5,81.7 360.5,81.9 361.4,81.3 362.3,80.9 363.3,79.9 364.2,80.1 365.2,84.4 366.1,84.1 367.0,80.4 368.0,80.4 368.9,76.4 369.9,75.6 370.8,75.6 371.7,75.4 372.7,75.3 373.6,70.5 374.6,66.0 375.5,65.8 376.4,66.1 377.4,65.8 378.3,65.6 379.3,65.8 380.2,66.1 381.1,66.5 382.1,67.0 383.0,71.2 384.0,71.3 384.9,71.5 385.8,75.6 386.8,75.9 387.7,79.4 388.7,83.3 389.6,83.2 390.5,87.0 391.5,87.0 392.4,82.9 393.4,83.2 394.3,83.4 395.2,87.4 396.2,91.4 397.1,87.1 398.1,87.4 399.0,87.2 399.9,83.3 400.9,82.7 401.8,82.5 402.8,81.6 403.7,81.0 404.6,72.1 405.6,67.7 406.5,68.1 407.5,64.1 408.4,64.5 409.3,68.9 410.3,73.0 411.2,73.0 412.2,73.6 413.1,74.0 414.0,74.6 415.0,74.3 415.9,74.1 416.9,74.1 417.8,73.8 418.7,69.0 419.7,65.2 420.6,65.6 421.6,66.1 422.5,66.3 423.4,70.6 424.4,70.6 425.3,70.4 426.3,70.0 427.2,69.9 428.1,70.2 429.1,70.3 430.0,70.0 431.0,69.9 431.9,73.6 432.8,69.0 433.8,69.2 434.7,69.2 435.7,69.0 436.6,69.0 437.5,69.0 438.5,68.4 439.4,68.6 440.4,68.0 441.3,64.3 442.2,64.4 443.2,64.9 444.1,65.2 445.1,65.7 446.0,66.0 446.9,70.0 447.9,70.4 448.8,70.3 449.8,70.1 450.7,70.2 451.6,70.5 452.6,74.2 453.5,74.2 454.5,74.0 455.4,77.8 456.3,73.6 457.3,73.7 458.2,73.7 459.2,73.8 460.1,77.9 461.0,77.6 462.0,73.5 462.9,81.3 463.9,81.0 464.8,76.7 465.7,81.8 466.7,81.7 467.6,81.7 468.6,82.0 469.5,78.4 470.4,78.4 471.4,83.1 472.3,75.3 473.3,75.5 474.2,79.3 475.1,74.4 476.1,74.3 477.0,74.2 478.0,74.0 478.9,72.9 479.8,73.0 480.8,68.5 481.7,72.6 482.7,72.9 483.6,69.5 484.5,74.3 485.5,74.0 486.4,74.1 487.4,74.4 488.3,74.4 489.2,74.5 490.2,74.6 491.1,70.3 492.1,70.4 493.0,70.1 493.9,65.2 494.9,69.2 495.8,69.2 496.8,68.7 497.7,68.8 498.6,68.7 499.6,72.3 500.5,72.4 501.5,72.2 502.4,72.6 503.3,72.7 504.3,69.3 505.2,69.0 506.2,69.4 507.1,77.5 508.0,77.8 509.0,73.8 509.9,73.7 510.9,74.0 511.8,73.7 512.7,73.6 513.7,73.2 514.6,73.7 515.6,73.9 516.5,66.1 517.4,65.8 518.4,70.3 519.3,70.7 520.3,70.3 521.2,70.4 522.1,70.3 523.1,70.2 524.0,69.8 525.0,69.4 525.9,73.3 526.8,77.4 527.8,72.6 528.7,76.4 529.7,75.9 530.6,84.3 531.5,84.4 532.5,84.6 533.4,84.8 534.4,84.7 535.3,80.6 536.2,76.6 537.2,76.6 538.1,76.7 539.1,82.0 540.0,73.6" class="stroke-d" fill="none"/>
<line x1="330.0" y1="190.0" x2="360.0" y2="190.0" class="stroke-b"/>
<text x="366.0" y="194.0" class="fig-small" text-anchor="start">Q-learning, end -43</text>
<line x1="330.0" y1="210.0" x2="360.0" y2="210.0" class="stroke-d"/>
<text x="366.0" y="214.0" class="fig-small" text-anchor="start">SARSA, end -22</text>
</svg>
Figure 3. Return per training episode, averaged over five seeds and smoothed over 10 episodes. Notice that SARSA earns more while it explores, even though Q-learning has the shorter greedy path.
```

In the seed-0 run, over the last 100 training episodes, SARSA earns about $-24$ per episode and Q-learning about $-39$. Q-learning keeps falling off because its path hugs the cliff, and its exploration still takes random steps there. Which answer is right depends on the question. If exploration stops after training, Q-learning's path is better. If the agent must keep exploring, SARSA's is.

> [!KEY]
> SARSA learns the value of the policy it follows, slips included, so it learns a safe path. Q-learning learns the value of the greedy policy, so it learns the shortest edge path and falls off it while it explores. Back at the cliff, the robot's two questions are these two methods.

> [!WARNING]
> `rlenvs.policy_evaluation` returns `nan` for any state from which the greedy policy never ends, such as a rarely visited cell whose greedy action walks into a wall. Compare a value with `np.isclose`, never with `abs(x - y) > tol`, because every comparison with `nan` is false.

> [!RECAP]
> - Both rules move `Q[s, a]` a fraction `alpha` toward `r + gamma * (next value) * (not terminated)`.
> - SARSA's next value is `Q[s2, a2]` for the action it will take. Q-learning's is `Q[s2].max()`.
> - SARSA is on-policy and learns the value of its $\varepsilon$-greedy policy. Q-learning is off-policy and learns the value of the greedy policy.
> - On the cliff, Q-learning's greedy path is worth $-13$, and SARSA's is longer and safer.
> - Break ties between greedy actions at random.

## Build it step by step

The checks test the functions in this order.

1. **`epsilon_greedy_action(q_row, epsilon, rng)`.** Takes one row of `Q` and returns an `int`. Draw `rng.random()` to decide whether to explore, and pick among `np.flatnonzero(q_row == q_row.max())` to break ties. The common mistake is `q_row.argmax()`, which always picks the first tie. With $Q = 0$, that sends the cliff walker up into the wall every time.
2. **`sarsa_update(Q, s, a, r, s2, a2, terminated, alpha, gamma)`.** Updates `Q[s, a]` in place and returns `Q`. The common mistake is forgetting to zero the bootstrap term at the end of an episode.
3. **`q_learning_update(Q, s, a, r, s2, terminated, alpha, gamma)`.** Same shape, with `Q[s2].max()` in place of `Q[s2, a2]`.
4. **`q_learning(mdp, n_episodes, alpha, epsilon, seed)`.** Returns `(Q, returns)`, where `returns[i]` is the total reward of training episode `i`. Choose, step, update, repeat, until `mdp.terminal[s2]` or `mdp.max_steps`.
5. **`sarsa(mdp, n_episodes, alpha, epsilon, seed)`.** Same outputs. Choose the first action before the step loop, choose `a2` before each update, and carry `a2` into the next step.

## Where this breaks

- **Overestimation.** Q-learning's max picks the action whose estimate is luckiest, so noisy rewards make it overestimate. [Maximization bias and double learning](/lessons/018-maximization-bias) measures this and fixes it.
- **Constant step sizes.** With a constant $\alpha$, values never stop moving in a stochastic world. People decay $\alpha$ over time, or accept the jitter and evaluate the greedy policy.
- **Exploration that never happens.** With $Q = 0$ and ties broken by `argmax`, a greedy agent can repeat one action forever. Random tie-breaking, $\varepsilon$-greedy exploration, or optimistic starting values fix it.
- **Large state spaces.** A table needs one entry per state-action pair. [Value-function approximation](/lessons/023-function-approximation) replaces the table with a function, and bootstrapping then brings its own risk of divergence.

## Further reading

- [Watkins and Dayan, "Q-learning" (1992)](https://doi.org/10.1007/BF00992698). The convergence proof for tabular Q-learning.
- [van Seijen, van Hasselt, Whiteson, and Wiering, "A theoretical and empirical analysis of Expected Sarsa" (2009)](https://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/vanseijenadprl09.pdf). A variant that averages over the next action.
- [Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd edition (2018), Chapter 6](http://incompleteideas.net/book/the-book-2nd.html). The cliff walk, Example 6.6.

## The exercise

You write `epsilon_greedy_action`, `sarsa_update`, `q_learning_update`, `q_learning`, and `sarsa`. The checks verify both updates on the worked example, and that `epsilon_greedy_action` splits ties evenly. On the cliff, Q-learning's greedy value from the start must be exactly $-13$. SARSA's must be finite and below $-13$ in two of three seeded runs, and SARSA must out-earn Q-learning during training.
