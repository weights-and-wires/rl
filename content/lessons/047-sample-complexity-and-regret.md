---
title: Sample complexity and regret
slug: 047-sample-complexity-and-regret
part: "Part 7 — Models, exploration, and fixed data"
summary: Bound how a wrong model hurts a plan, show why optimism keeps UCB-VI's values above the truth, and measure regret on RiverSwim.
minutes: 40
gpu: false
objectives:
  - State the simulation lemma, and check it numerically on random pairs of MDPs.
  - Build UCB-VI with a Hoeffding bonus and verify that its values stay above the optimal values.
  - Measure cumulative regret on RiverSwim and say why a bonus with the proven constant explores for so long.
  - State the Robbins-Monro step-size conditions and show what each one buys with an exact variance.
prereqs:
  - 004-optimism-and-ucb
  - 012-value-iteration
  - 016-temporal-difference-learning
  - 043-model-based-rl
  - 046-exploration-bonuses
exercise: 047-sample-complexity-and-regret
---

# Sample complexity and regret

The previous chapter made an agent swim to the far end of RiverSwim by paying it a bonus for the unknown. Two questions stay open. How many samples does an agent need before you can trust its plan? How much reward does it give up while it learns?

A bandit has an answer to the second question: regret, the reward you lost against always pulling the best arm. This chapter carries regret to sequential problems, adds its partner *sample complexity*, and checks the inequality behind most such guarantees, the simulation lemma, on 200 random worlds.

## What you already know

- A model-based agent estimates transition probabilities from counts and plans in the estimated model (see [Model-based reinforcement learning](/lessons/043-model-based-rl)).
- A count bonus $\beta / \sqrt{N}$ makes rarely tried actions look good, so the agent tries them (see [Exploration beyond epsilon](/lessons/046-exploration-bonuses)).
- UCB adds a confidence width to each arm's mean (see [Optimism and UCB](/lessons/004-optimism-and-ucb)).
- A temporal-difference update moves an estimate a step $\alpha$ toward its target.

## Two ways to count mistakes

Fix a small error $\varepsilon$ and a failure probability $\delta$. The **sample complexity** of an algorithm is the number of steps it needs before it returns a policy within $\varepsilon$ of the best, with probability at least $1 - \delta$. It charges you only for the final answer. The **regret** charges you for every mistake on the way. Run $K$ episodes, and let $\pi_k$ be the policy played in episode $k$:

$$
\text{Regret}(K) = \sum_{k=1}^{K} \big( V^*(s_0) - V^{\pi_k}(s_0) \big)
$$

Read it as: add up how much less the policy in use earned than the best policy would have. An agent that stops improving has regret that grows in a straight line. An agent that learns has *sublinear* regret, which flattens: regret divided by $K$ goes to zero.

Figure 1 previews the result: regret on RiverSwim over 300 episodes of 20 steps, for three agents that share one model-based loop.

```figure
<svg viewBox="0 0 600 340" role="img" aria-label="Cumulative regret over 300 episodes on RiverSwim: epsilon-greedy and UCB-VI with the full bonus grow linearly, UCB-VI with a tenth of the bonus flattens">
<line x1="70.0" y1="270.0" x2="540.0" y2="270.0" class="stroke-grid"/>
<text x="62.0" y="274.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="212.5" x2="540.0" y2="212.5" class="stroke-grid"/>
<text x="62.0" y="216.5" class="fig-small" text-anchor="end">500</text>
<line x1="70.0" y1="155.0" x2="540.0" y2="155.0" class="stroke-grid"/>
<text x="62.0" y="159.0" class="fig-small" text-anchor="end">1000</text>
<line x1="70.0" y1="97.5" x2="540.0" y2="97.5" class="stroke-grid"/>
<text x="62.0" y="101.5" class="fig-small" text-anchor="end">1500</text>
<line x1="70.0" y1="40.0" x2="540.0" y2="40.0" class="stroke-grid"/>
<text x="62.0" y="44.0" class="fig-small" text-anchor="end">2000</text>
<line x1="70.0" y1="270.0" x2="540.0" y2="270.0" class="stroke-muted"/>
<line x1="70.0" y1="270.0" x2="70.0" y2="275.0" class="stroke-muted"/>
<text x="70.0" y="290.0" class="fig-small" text-anchor="middle">0</text>
<line x1="226.7" y1="270.0" x2="226.7" y2="275.0" class="stroke-muted"/>
<text x="226.7" y="290.0" class="fig-small" text-anchor="middle">100</text>
<line x1="383.3" y1="270.0" x2="383.3" y2="275.0" class="stroke-muted"/>
<text x="383.3" y="290.0" class="fig-small" text-anchor="middle">200</text>
<line x1="540.0" y1="270.0" x2="540.0" y2="275.0" class="stroke-muted"/>
<text x="540.0" y="290.0" class="fig-small" text-anchor="middle">300</text>
<text x="305.0" y="318" class="fig-small" text-anchor="middle">episode</text>
<polyline points="71.6,269.3 79.4,266.0 87.2,262.7 95.1,259.4 102.9,256.1 110.7,252.8 118.6,249.5 126.4,246.2 134.2,242.9 142.1,239.6 149.9,236.3 157.7,233.0 165.6,229.7 173.4,226.4 181.2,223.1 189.1,219.8 196.9,216.5 204.7,213.2 212.6,209.9 220.4,206.6 228.2,203.3 236.1,200.0 243.9,196.7 251.7,193.4 259.6,190.1 267.4,186.8 275.2,183.5 283.1,180.2 290.9,176.9 298.7,173.6 306.6,170.3 314.4,167.0 322.2,163.6 330.1,160.3 337.9,157.0 345.7,153.7 353.6,150.4 361.4,147.1 369.2,143.8 377.1,140.5 384.9,137.2 392.7,133.9 400.6,130.6 408.4,127.3 416.2,124.0 424.1,120.7 431.9,117.4 439.7,114.1 447.6,110.8 455.4,107.5 463.2,104.2 471.1,100.9 478.9,97.6 486.7,94.3 494.6,91.0 502.4,87.7 510.2,84.4 518.1,81.1 525.9,77.8 533.7,74.5" class="stroke-a" fill="none"/>
<line x1="100.0" y1="50" x2="130.0" y2="50" class="stroke-a"/>
<text x="136.0" y="54" class="fig-small" text-anchor="start">epsilon-greedy: 1723 at episode 300</text>
<polyline points="71.6,269.3 79.4,266.0 87.2,262.7 95.1,259.4 102.9,256.1 110.7,252.8 118.6,249.5 126.4,246.2 134.2,242.9 142.1,239.6 149.9,236.4 157.7,233.1 165.6,229.8 173.4,226.5 181.2,223.2 189.1,219.9 196.9,216.6 204.7,213.3 212.6,209.9 220.4,206.6 228.2,203.2 236.1,199.9 243.9,196.5 251.7,193.2 259.6,189.8 267.4,186.4 275.2,183.1 283.1,179.7 290.9,176.4 298.7,173.0 306.6,169.7 314.4,166.3 322.2,163.0 330.1,159.6 337.9,156.2 345.7,152.9 353.6,149.5 361.4,146.2 369.2,142.8 377.1,139.5 384.9,136.1 392.7,132.8 400.6,129.4 408.4,126.1 416.2,122.7 424.1,119.4 431.9,116.1 439.7,112.7 447.6,109.4 455.4,106.0 463.2,102.7 471.1,99.3 478.9,96.0 486.7,92.6 494.6,89.3 502.4,86.0 510.2,82.6 518.1,79.3 525.9,75.9 533.7,72.6" class="stroke-b" fill="none"/>
<line x1="100.0" y1="68" x2="130.0" y2="68" class="stroke-b"/>
<text x="136.0" y="72" class="fig-small" text-anchor="start">UCB-VI, bonus scale 1: 1740 at episode 300</text>
<polyline points="71.6,269.3 79.4,266.0 87.2,262.6 95.1,259.3 102.9,256.3 110.7,254.9 118.6,254.2 126.4,253.9 134.2,253.7 142.1,253.5 149.9,253.3 157.7,253.1 165.6,252.9 173.4,252.8 181.2,252.7 189.1,252.5 196.9,252.4 204.7,252.3 212.6,252.1 220.4,252.0 228.2,251.9 236.1,251.8 243.9,251.7 251.7,251.6 259.6,251.4 267.4,251.3 275.2,251.2 283.1,251.1 290.9,251.0 298.7,250.9 306.6,250.8 314.4,250.7 322.2,250.6 330.1,250.5 337.9,250.4 345.7,250.4 353.6,250.3 361.4,250.2 369.2,250.1 377.1,250.1 384.9,250.0 392.7,249.9 400.6,249.8 408.4,249.8 416.2,249.7 424.1,249.7 431.9,249.6 439.7,249.5 447.6,249.5 455.4,249.4 463.2,249.4 471.1,249.3 478.9,249.2 486.7,249.2 494.6,249.1 502.4,249.1 510.2,249.0 518.1,248.9 525.9,248.9 533.7,248.8" class="stroke-c" fill="none"/>
<line x1="100.0" y1="86" x2="130.0" y2="86" class="stroke-c"/>
<text x="136.0" y="90" class="fig-small" text-anchor="start">UCB-VI, bonus scale 0.1: 185 at episode 300</text>
</svg>
Figure 1. Mean cumulative regret on RiverSwim, five seeds. Epsilon-greedy never finds the far reward, so its regret climbs in a straight line. UCB-VI with the bonus constant from the proof does no better in 300 episodes. With a tenth of that bonus it finds the reward within about 50 episodes and then flattens.
```

The middle curve is a surprise, and the chapter explains it. First, the inequality that every guarantee rests on.

## From a wrong model to a wrong plan: the simulation lemma

An agent that plans in an estimated model $\hat P$ gets the right answer for the wrong world. How wrong can it be? Take the smallest example: two states and one action, so every policy is the same policy. State 0 pays 1 and state 1 pays 0 forever. With $\gamma = 0.5$, the true model keeps state 0 in place with probability 0.8, so its value is $1 / (1 - 0.5 \cdot 0.8) = 1.6667$. An estimate with probability 0.6 gives $1 / (1 - 0.5 \cdot 0.6) = 1.4286$. The planner's value is off by 0.2381.

The **simulation lemma** says exactly where that 0.2381 comes from. Write $\Delta = Q^\pi - \hat Q^\pi$. Each $Q$ is the reward plus $\gamma$ times the expected next value, so

$$
\Delta = \gamma P V^\pi - \gamma \hat P \hat V^\pi = \gamma (P - \hat P) V^\pi + \gamma \hat P (V^\pi - \hat V^\pi) = \gamma (P - \hat P) V^\pi + \gamma \hat P^\pi \Delta
$$

where $\hat P^\pi$ takes one step in the estimated model and then one action from $\pi$, which turns $V^\pi - \hat V^\pi$ into $\Delta$. Solve for $\Delta$:

$$
Q^\pi - \hat Q^\pi = \gamma \, (I - \gamma \hat P^\pi)^{-1} (P - \hat P) V^\pi
$$

Read it as: the one-step model error, $(P - \hat P)V^\pi$, passed through $(I - \gamma \hat P^\pi)^{-1}$, which adds it up over every later step. In the example, the one-step error is $0.2 \times 1.6667 = 0.3333$ in state 0, the inverse contributes $1 / (1 - 0.3) = 1.4286$, and $0.5 \times 1.4286 \times 0.3333 = 0.2381$. The identity is exact.

To get a bound, take the worst case. Let $\varepsilon = \max_{s,a} \lVert \hat P(\cdot \mid s, a) - P(\cdot \mid s, a) \rVert_1$ and $R_{\max} = \max |r|$. Every row of $(I - \gamma \hat P^\pi)^{-1}$ adds up to at most $1 / (1 - \gamma)$, and $\lVert V^\pi \rVert_\infty \le R_{\max} / (1 - \gamma)$, so

$$
\lVert Q^\pi - \hat Q^\pi \rVert_\infty \le \frac{\gamma}{(1 - \gamma)^2} \, \varepsilon \, R_{\max}
$$

For the example, $\varepsilon = 0.2 + 0.2 = 0.4$ and $\gamma / (1 - \gamma)^2 = 2$, so the bound is 0.8, and the true gap is under a third of it. The bound covers every policy at once, so it can be loose for any one.

```python
eps = np.abs(P_hat - P).sum(axis=2).max()          # largest L1 distance over rows (s, a)
bound = gamma / (1 - gamma) ** 2 * eps * np.abs(R).max()
```

Figure 2 tests it on 200 random pairs of MDPs with 5 states, 3 actions, and $\gamma = 0.9$. Each $\hat P$ mixes the true $P$ with a second random model at a weight between 0.02 and 0.5.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Scatter plot on log axes: the true Q error of 200 random MDP pairs against the simulation-lemma bound; every point lies far below the diagonal where the two are equal">
<line x1="70.0" y1="270.0" x2="70.0" y2="40.0" class="stroke-grid"/>
<text x="70.0" y="288.0" class="fig-small" text-anchor="middle">1e0</text>
<line x1="226.7" y1="270.0" x2="226.7" y2="40.0" class="stroke-grid"/>
<text x="226.7" y="288.0" class="fig-small" text-anchor="middle">1e1</text>
<line x1="383.3" y1="270.0" x2="383.3" y2="40.0" class="stroke-grid"/>
<text x="383.3" y="288.0" class="fig-small" text-anchor="middle">1e2</text>
<line x1="540.0" y1="270.0" x2="540.0" y2="40.0" class="stroke-grid"/>
<text x="540.0" y="288.0" class="fig-small" text-anchor="middle">1e3</text>
<line x1="70.0" y1="270.0" x2="540.0" y2="270.0" class="stroke-grid"/>
<text x="62.0" y="274.0" class="fig-small" text-anchor="end">1e-2</text>
<line x1="70.0" y1="193.3" x2="540.0" y2="193.3" class="stroke-grid"/>
<text x="62.0" y="197.3" class="fig-small" text-anchor="end">1e-1</text>
<line x1="70.0" y1="116.7" x2="540.0" y2="116.7" class="stroke-grid"/>
<text x="62.0" y="120.7" class="fig-small" text-anchor="end">1e0</text>
<line x1="70.0" y1="40.0" x2="540.0" y2="40.0" class="stroke-grid"/>
<text x="62.0" y="44.0" class="fig-small" text-anchor="end">1e1</text>
<line x1="70.0" y1="270.0" x2="540.0" y2="270.0" class="stroke-muted"/>
<line x1="70.0" y1="270.0" x2="70.0" y2="40.0" class="stroke-muted"/>
<line x1="70.0" y1="116.7" x2="226.7" y2="40.0" class="stroke-bad dashed"/>
<circle cx="379.7" cy="140.0" r="2.5" class="fill-a"/>
<circle cx="350.0" cy="146.9" r="2.5" class="fill-a"/>
<circle cx="294.4" cy="180.9" r="2.5" class="fill-a"/>
<circle cx="243.9" cy="204.6" r="2.5" class="fill-a"/>
<circle cx="407.2" cy="103.8" r="2.5" class="fill-a"/>
<circle cx="353.8" cy="177.1" r="2.5" class="fill-a"/>
<circle cx="340.0" cy="181.0" r="2.5" class="fill-a"/>
<circle cx="345.2" cy="174.1" r="2.5" class="fill-a"/>
<circle cx="341.3" cy="157.6" r="2.5" class="fill-a"/>
<circle cx="397.6" cy="122.3" r="2.5" class="fill-a"/>
<circle cx="387.9" cy="141.2" r="2.5" class="fill-a"/>
<circle cx="278.0" cy="164.1" r="2.5" class="fill-a"/>
<circle cx="304.5" cy="157.2" r="2.5" class="fill-a"/>
<circle cx="389.8" cy="153.3" r="2.5" class="fill-a"/>
<circle cx="400.3" cy="152.8" r="2.5" class="fill-a"/>
<circle cx="372.7" cy="139.5" r="2.5" class="fill-a"/>
<circle cx="375.5" cy="143.0" r="2.5" class="fill-a"/>
<circle cx="395.7" cy="177.8" r="2.5" class="fill-a"/>
<circle cx="325.0" cy="156.5" r="2.5" class="fill-a"/>
<circle cx="367.7" cy="137.1" r="2.5" class="fill-a"/>
<circle cx="339.2" cy="162.6" r="2.5" class="fill-a"/>
<circle cx="372.3" cy="126.1" r="2.5" class="fill-a"/>
<circle cx="355.7" cy="124.7" r="2.5" class="fill-a"/>
<circle cx="398.1" cy="168.3" r="2.5" class="fill-a"/>
<circle cx="315.6" cy="183.1" r="2.5" class="fill-a"/>
<circle cx="307.0" cy="202.6" r="2.5" class="fill-a"/>
<circle cx="333.0" cy="141.5" r="2.5" class="fill-a"/>
<circle cx="382.3" cy="125.2" r="2.5" class="fill-a"/>
<circle cx="369.4" cy="145.5" r="2.5" class="fill-a"/>
<circle cx="245.0" cy="194.1" r="2.5" class="fill-a"/>
<circle cx="320.0" cy="181.2" r="2.5" class="fill-a"/>
<circle cx="389.0" cy="157.0" r="2.5" class="fill-a"/>
<circle cx="303.4" cy="176.8" r="2.5" class="fill-a"/>
<circle cx="350.9" cy="145.8" r="2.5" class="fill-a"/>
<circle cx="175.1" cy="258.7" r="2.5" class="fill-a"/>
<circle cx="325.8" cy="176.7" r="2.5" class="fill-a"/>
<circle cx="297.8" cy="183.6" r="2.5" class="fill-a"/>
<circle cx="383.0" cy="156.8" r="2.5" class="fill-a"/>
<circle cx="368.2" cy="155.4" r="2.5" class="fill-a"/>
<circle cx="335.8" cy="178.7" r="2.5" class="fill-a"/>
<circle cx="366.9" cy="148.5" r="2.5" class="fill-a"/>
<circle cx="403.9" cy="152.3" r="2.5" class="fill-a"/>
<circle cx="345.3" cy="128.7" r="2.5" class="fill-a"/>
<circle cx="395.5" cy="104.1" r="2.5" class="fill-a"/>
<circle cx="263.1" cy="210.5" r="2.5" class="fill-a"/>
<circle cx="354.5" cy="153.1" r="2.5" class="fill-a"/>
<circle cx="393.5" cy="146.6" r="2.5" class="fill-a"/>
<circle cx="391.2" cy="118.2" r="2.5" class="fill-a"/>
<circle cx="298.3" cy="161.3" r="2.5" class="fill-a"/>
<circle cx="365.2" cy="178.8" r="2.5" class="fill-a"/>
<circle cx="377.8" cy="150.0" r="2.5" class="fill-a"/>
<circle cx="418.0" cy="143.3" r="2.5" class="fill-a"/>
<circle cx="381.4" cy="137.5" r="2.5" class="fill-a"/>
<circle cx="179.5" cy="263.6" r="2.5" class="fill-a"/>
<circle cx="289.6" cy="186.3" r="2.5" class="fill-a"/>
<circle cx="393.0" cy="106.9" r="2.5" class="fill-a"/>
<circle cx="311.7" cy="171.1" r="2.5" class="fill-a"/>
<circle cx="362.5" cy="148.6" r="2.5" class="fill-a"/>
<circle cx="297.1" cy="181.2" r="2.5" class="fill-a"/>
<circle cx="374.0" cy="157.7" r="2.5" class="fill-a"/>
<circle cx="299.6" cy="141.1" r="2.5" class="fill-a"/>
<circle cx="347.8" cy="153.2" r="2.5" class="fill-a"/>
<circle cx="349.6" cy="147.2" r="2.5" class="fill-a"/>
<circle cx="393.4" cy="133.1" r="2.5" class="fill-a"/>
<circle cx="368.1" cy="127.2" r="2.5" class="fill-a"/>
<circle cx="245.7" cy="227.0" r="2.5" class="fill-a"/>
<circle cx="390.9" cy="134.3" r="2.5" class="fill-a"/>
<circle cx="363.5" cy="177.5" r="2.5" class="fill-a"/>
<circle cx="341.6" cy="181.9" r="2.5" class="fill-a"/>
<circle cx="316.4" cy="180.4" r="2.5" class="fill-a"/>
<circle cx="351.6" cy="159.1" r="2.5" class="fill-a"/>
<circle cx="310.6" cy="182.0" r="2.5" class="fill-a"/>
<circle cx="397.3" cy="127.7" r="2.5" class="fill-a"/>
<circle cx="370.0" cy="135.2" r="2.5" class="fill-a"/>
<circle cx="402.2" cy="120.0" r="2.5" class="fill-a"/>
<circle cx="287.5" cy="164.0" r="2.5" class="fill-a"/>
<circle cx="398.5" cy="154.5" r="2.5" class="fill-a"/>
<circle cx="362.5" cy="144.0" r="2.5" class="fill-a"/>
<circle cx="337.4" cy="136.5" r="2.5" class="fill-a"/>
<circle cx="419.8" cy="147.5" r="2.5" class="fill-a"/>
<circle cx="381.0" cy="134.6" r="2.5" class="fill-a"/>
<circle cx="340.0" cy="192.7" r="2.5" class="fill-a"/>
<circle cx="401.4" cy="127.2" r="2.5" class="fill-a"/>
<circle cx="348.9" cy="140.2" r="2.5" class="fill-a"/>
<circle cx="360.8" cy="160.2" r="2.5" class="fill-a"/>
<circle cx="244.2" cy="201.8" r="2.5" class="fill-a"/>
<circle cx="365.6" cy="165.1" r="2.5" class="fill-a"/>
<circle cx="376.3" cy="131.6" r="2.5" class="fill-a"/>
<circle cx="280.5" cy="177.3" r="2.5" class="fill-a"/>
<circle cx="362.6" cy="164.7" r="2.5" class="fill-a"/>
<circle cx="367.6" cy="193.8" r="2.5" class="fill-a"/>
<circle cx="307.6" cy="195.6" r="2.5" class="fill-a"/>
<circle cx="389.7" cy="155.4" r="2.5" class="fill-a"/>
<circle cx="401.5" cy="166.2" r="2.5" class="fill-a"/>
<circle cx="297.9" cy="220.7" r="2.5" class="fill-a"/>
<circle cx="354.9" cy="211.1" r="2.5" class="fill-a"/>
<circle cx="366.5" cy="120.9" r="2.5" class="fill-a"/>
<circle cx="413.3" cy="97.2" r="2.5" class="fill-a"/>
<circle cx="424.4" cy="102.3" r="2.5" class="fill-a"/>
<circle cx="333.7" cy="147.7" r="2.5" class="fill-a"/>
<circle cx="377.6" cy="120.0" r="2.5" class="fill-a"/>
<circle cx="436.3" cy="132.5" r="2.5" class="fill-a"/>
<circle cx="278.1" cy="171.7" r="2.5" class="fill-a"/>
<circle cx="319.9" cy="194.4" r="2.5" class="fill-a"/>
<circle cx="349.5" cy="172.5" r="2.5" class="fill-a"/>
<circle cx="376.0" cy="199.1" r="2.5" class="fill-a"/>
<circle cx="402.2" cy="103.1" r="2.5" class="fill-a"/>
<circle cx="360.2" cy="137.3" r="2.5" class="fill-a"/>
<circle cx="416.7" cy="133.8" r="2.5" class="fill-a"/>
<circle cx="369.9" cy="172.3" r="2.5" class="fill-a"/>
<circle cx="384.9" cy="95.1" r="2.5" class="fill-a"/>
<circle cx="299.8" cy="185.9" r="2.5" class="fill-a"/>
<circle cx="261.3" cy="187.8" r="2.5" class="fill-a"/>
<circle cx="272.1" cy="223.2" r="2.5" class="fill-a"/>
<circle cx="399.6" cy="126.7" r="2.5" class="fill-a"/>
<circle cx="384.3" cy="155.0" r="2.5" class="fill-a"/>
<circle cx="267.0" cy="202.6" r="2.5" class="fill-a"/>
<circle cx="377.4" cy="181.7" r="2.5" class="fill-a"/>
<circle cx="409.6" cy="107.5" r="2.5" class="fill-a"/>
<circle cx="393.9" cy="156.4" r="2.5" class="fill-a"/>
<circle cx="371.0" cy="117.3" r="2.5" class="fill-a"/>
<circle cx="391.0" cy="153.5" r="2.5" class="fill-a"/>
<circle cx="376.8" cy="157.8" r="2.5" class="fill-a"/>
<circle cx="368.0" cy="137.4" r="2.5" class="fill-a"/>
<circle cx="385.0" cy="163.6" r="2.5" class="fill-a"/>
<circle cx="409.4" cy="105.9" r="2.5" class="fill-a"/>
<circle cx="187.4" cy="217.9" r="2.5" class="fill-a"/>
<circle cx="257.6" cy="203.3" r="2.5" class="fill-a"/>
<circle cx="368.3" cy="135.2" r="2.5" class="fill-a"/>
<circle cx="374.1" cy="209.1" r="2.5" class="fill-a"/>
<circle cx="342.6" cy="157.7" r="2.5" class="fill-a"/>
<circle cx="381.4" cy="120.5" r="2.5" class="fill-a"/>
<circle cx="237.3" cy="174.5" r="2.5" class="fill-a"/>
<circle cx="398.8" cy="122.2" r="2.5" class="fill-a"/>
<circle cx="375.0" cy="164.9" r="2.5" class="fill-a"/>
<circle cx="361.2" cy="150.7" r="2.5" class="fill-a"/>
<circle cx="275.2" cy="184.8" r="2.5" class="fill-a"/>
<circle cx="350.8" cy="123.9" r="2.5" class="fill-a"/>
<circle cx="369.0" cy="153.8" r="2.5" class="fill-a"/>
<circle cx="344.9" cy="221.9" r="2.5" class="fill-a"/>
<circle cx="369.6" cy="162.6" r="2.5" class="fill-a"/>
<circle cx="387.8" cy="143.8" r="2.5" class="fill-a"/>
<circle cx="250.1" cy="237.1" r="2.5" class="fill-a"/>
<circle cx="357.2" cy="141.9" r="2.5" class="fill-a"/>
<circle cx="407.3" cy="119.8" r="2.5" class="fill-a"/>
<circle cx="374.0" cy="166.3" r="2.5" class="fill-a"/>
<circle cx="394.6" cy="144.9" r="2.5" class="fill-a"/>
<circle cx="431.6" cy="95.8" r="2.5" class="fill-a"/>
<circle cx="391.0" cy="135.8" r="2.5" class="fill-a"/>
<circle cx="216.5" cy="194.4" r="2.5" class="fill-a"/>
<circle cx="182.8" cy="254.7" r="2.5" class="fill-a"/>
<circle cx="372.4" cy="91.1" r="2.5" class="fill-a"/>
<circle cx="297.9" cy="184.6" r="2.5" class="fill-a"/>
<circle cx="389.1" cy="102.7" r="2.5" class="fill-a"/>
<circle cx="369.8" cy="170.7" r="2.5" class="fill-a"/>
<circle cx="348.5" cy="149.3" r="2.5" class="fill-a"/>
<circle cx="291.2" cy="209.4" r="2.5" class="fill-a"/>
<circle cx="274.4" cy="217.7" r="2.5" class="fill-a"/>
<circle cx="422.5" cy="118.0" r="2.5" class="fill-a"/>
<circle cx="386.7" cy="156.6" r="2.5" class="fill-a"/>
<circle cx="339.4" cy="160.8" r="2.5" class="fill-a"/>
<circle cx="375.5" cy="138.4" r="2.5" class="fill-a"/>
<circle cx="255.9" cy="210.3" r="2.5" class="fill-a"/>
<circle cx="408.9" cy="160.7" r="2.5" class="fill-a"/>
<circle cx="249.7" cy="200.6" r="2.5" class="fill-a"/>
<circle cx="352.8" cy="183.5" r="2.5" class="fill-a"/>
<circle cx="391.6" cy="130.1" r="2.5" class="fill-a"/>
<circle cx="278.5" cy="191.5" r="2.5" class="fill-a"/>
<circle cx="316.2" cy="236.6" r="2.5" class="fill-a"/>
<circle cx="355.0" cy="151.6" r="2.5" class="fill-a"/>
<circle cx="343.4" cy="163.6" r="2.5" class="fill-a"/>
<circle cx="276.2" cy="210.9" r="2.5" class="fill-a"/>
<circle cx="302.6" cy="158.8" r="2.5" class="fill-a"/>
<circle cx="253.6" cy="212.0" r="2.5" class="fill-a"/>
<circle cx="423.3" cy="168.6" r="2.5" class="fill-a"/>
<circle cx="359.5" cy="159.5" r="2.5" class="fill-a"/>
<circle cx="378.5" cy="115.9" r="2.5" class="fill-a"/>
<circle cx="353.2" cy="189.5" r="2.5" class="fill-a"/>
<circle cx="380.7" cy="136.5" r="2.5" class="fill-a"/>
<circle cx="391.8" cy="139.9" r="2.5" class="fill-a"/>
<circle cx="348.2" cy="181.7" r="2.5" class="fill-a"/>
<circle cx="407.5" cy="136.8" r="2.5" class="fill-a"/>
<circle cx="292.8" cy="212.9" r="2.5" class="fill-a"/>
<circle cx="399.3" cy="129.9" r="2.5" class="fill-a"/>
<circle cx="196.8" cy="223.3" r="2.5" class="fill-a"/>
<circle cx="282.2" cy="205.0" r="2.5" class="fill-a"/>
<circle cx="365.5" cy="209.4" r="2.5" class="fill-a"/>
<circle cx="303.7" cy="194.2" r="2.5" class="fill-a"/>
<circle cx="265.1" cy="205.5" r="2.5" class="fill-a"/>
<circle cx="390.1" cy="160.1" r="2.5" class="fill-a"/>
<circle cx="365.9" cy="191.8" r="2.5" class="fill-a"/>
<circle cx="405.9" cy="137.9" r="2.5" class="fill-a"/>
<circle cx="378.8" cy="152.6" r="2.5" class="fill-a"/>
<circle cx="352.5" cy="151.0" r="2.5" class="fill-a"/>
<circle cx="388.5" cy="121.8" r="2.5" class="fill-a"/>
<circle cx="215.6" cy="247.7" r="2.5" class="fill-a"/>
<circle cx="407.8" cy="140.4" r="2.5" class="fill-a"/>
<circle cx="411.5" cy="159.7" r="2.5" class="fill-a"/>
<circle cx="376.7" cy="151.5" r="2.5" class="fill-a"/>
<circle cx="377.8" cy="132.6" r="2.5" class="fill-a"/>
<text x="234.7" y="52.0" class="fig-small" text-anchor="start">gap = bound</text>
<text x="305.0" y="314" class="fig-small" text-anchor="middle">simulation-lemma bound</text>
<text x="14" y="155.0" class="fig-small" text-anchor="middle" transform="rotate(-90 14 155.0)">true max |Q error|</text>
</svg>
Figure 2. True gap against the simulation-lemma bound for 200 random MDP pairs. Every point sits below the dashed line where the two are equal. The median gap is 0.5% of the bound and the largest is 2.5%, so the bound is safe and loose.
```

The lemma turns a statement about samples into a statement about plans. The empirical distribution of $N$ samples over $S$ outcomes has an $L_1$ error of about $\sqrt{S / N}$, so $\varepsilon$ shrinks like $1 / \sqrt{N}$, and the plan's error follows it, scaled by at most $\gamma / (1 - \gamma)^2$. A sharper variance argument reaches the known minimax rate of about $SA / \big((1 - \gamma)^3 \varepsilon^2\big)$ samples with a generative sampler (Agarwal, Jiang, Kakade, and Sun, chapter 2).

> [!KEY]
> The simulation lemma separates two jobs. Statistics says how close $\hat P$ is to $P$, and that shrinks like $1 / \sqrt{N}$. The lemma says how much a planner's error grows from that closeness. Sample complexity is the product of the two.

> [!WARNING]
> The bound is a ceiling, not a prediction. On the 200 pairs it overstates the true gap 40 to 1,400 times. Use it to read off how error scales, never to size a dataset.

## Optimism and UCB-VI

The lemma assumes $\hat P$ is close everywhere. An exploring agent visits some pairs a thousand times and others never. UCB-VI (Azar, Osband, and Munos, 2017) makes the planner *optimistic* about pairs it knows little about, so it goes and learns them.

The setting is episodic: $H$ steps per episode, known rewards in $[0, 1]$, and the same transitions $P$ at every step. Let $N(s, a)$ count visits, and let $\hat P$ be the empirical model, where an untried pair is a self-loop. UCB-VI adds a bonus to the reward and runs backward induction from step $H - 1$ to step 0, clipping at the most the future could still pay:

$$
\tilde Q_h(s, a) = \min\Big( H - h, \; r(s, a) + b(s, a) + \sum_{s'} \hat P(s' \mid s, a) \max_{a'} \tilde Q_{h+1}(s', a') \Big)
$$

with the Hoeffding bonus

$$
b(s, a) = H \sqrt{\frac{\log (2 S A H / \delta)}{2 \max(N(s, a), 1)}}
$$

Read it as: a pair is worth its reward, plus how unsure the agent is about it, plus the estimated future. The cap $H - h$ keeps an untried pair from looking better than the steps left could pay. The agent plays greedily on $\tilde Q$ for an episode, updates the counts, and repeats.

Optimism is the property that makes this work: $\tilde Q_h \ge Q^*_h$ for every step, state, and action. The proof is an induction from the last step backward. Suppose $\tilde Q_{h+1} \ge Q^*_{h+1}$. Then $\tilde Q_h \ge Q^*_h$ whenever the bonus covers the sampling error, $b \ge (\hat P - P) V^*_{h+1}$. Hoeffding's inequality bounds that error by $H \sqrt{\log(2SAH/\delta) / (2N)}$ with probability $1 - \delta / (SAH)$, because $V^*_{h+1}$ is a fixed function with values in $[0, H]$. A union bound over the $SAH$ cases gives probability $1 - \delta$.

### Work it by hand

For $N = [[0, 1], [4, 100]]$, $H = 10$, $\delta = 0.1$, and $S A = 4$, the log term is $\log 800 = 6.685$. A pair with $N \le 1$ gets $10 \sqrt{6.685 / 2} = 18.28$, one with $N = 4$ gets 9.14, and one with $N = 100$ gets 1.83. Twenty-five times the visits cut the bonus 5 times.

```python
bonus = H * np.sqrt(np.log(2 * N.size * H / delta) / (2 * np.maximum(N, 1)))
Q = backward_induction(P_hat, R + bonus, H, clip=True)
```

Because $\tilde V$ upper-bounds what any policy can earn, the regret of episode $k$ is at most $\tilde V - V^{\pi_k}$, which the bonuses along the trajectory bound. Azar and others (2017) prove that a sharper Bernstein bonus gives regret of order $\sqrt{HSAT}$ for $T = HK$ steps, up to logarithmic factors, and that no algorithm beats that order in the worst case.

### Why the proven constant explores for so long

On RiverSwim with $H = 20$, $S A = 12$, and $\delta = 0.1$, an untried pair has a bonus of 41, and 1,000 visits still leave 1.30, on a reward scale of 1 per step. In 300 episodes, which is 6,000 steps over 12 pairs, the bonus never drops below the rewards it should be weighing, so the agent keeps exploring everywhere. Its regret is 1740 against 1723 for epsilon-greedy.

Scaling the bonus by 0.1 gives 0.13 at 1,000 visits. Regret falls to 185, 9 times below epsilon-greedy, and the mean per episode drops from 2.9 in the first 50 episodes to 0.10 in the last 50. Optimism is no longer proved at that scale, so the exercise checks it only at the full scale.

> [!KEY]
> If the bonus covers the sampling error, the agent's values never fall below the optimal values, so it can only err by trying something worse than it hoped, and each try shrinks a bonus. The price is a constant large enough to hide the effect on short runs.

## The model-free side: step sizes

UCB-VI keeps counts and a model. Q-learning keeps one number per pair and nudges it with a step size $\alpha_n$ at the $n$th update. Robbins and Monro (1951) give two conditions for convergence:

$$
\sum_{n=1}^{\infty} \alpha_n = \infty \qquad \text{and} \qquad \sum_{n=1}^{\infty} \alpha_n^2 < \infty
$$

The first says the steps can carry the estimate any distance, so a bad start is forgotten. The second says they shrink fast enough that sample noise averages out. 
Take a running average of noisy samples $x_n$ with variance $\sigma^2$, updated as $q \leftarrow q + \alpha_n (x_n - q)$. A sample $x_k$ reaches the final estimate with weight $w_k = \alpha_k \prod_{j > k} (1 - \alpha_j)$, so the variance after $n$ updates is exactly $\sigma^2 \sum_k w_k^2$:

| schedule | $\sum \alpha_n$, 1,000 steps | $\sum \alpha_n^2$, 1,000 steps | variance after 1,000 steps |
|---|---|---|---|
| $\alpha_n = 1/n$ | 7.485 | 1.644 | $0.001\,\sigma^2$ |
| $\alpha_n = 0.1$ | 100 | 10 | $0.0526\,\sigma^2$ |

With $\alpha_n = 1/n$ every weight is $1/n$, so the estimate is the sample mean and the variance is $\sigma^2 / n$. With a constant $\alpha$ the second sum grows like $\alpha^2 n$, so the variance settles at $\sigma^2 \alpha / (2 - \alpha)$ and never reaches zero. In exchange the estimate forgets old samples, so it tracks a moving target, which is why deep RL keeps constant step sizes.

> [!RECAP]
> - Sample complexity counts samples to an $\varepsilon$-optimal policy. Regret sums the shortfall of every policy played.
> - The simulation lemma gives $Q^\pi - \hat Q^\pi = \gamma (I - \gamma \hat P^\pi)^{-1} (P - \hat P) V^\pi$, and the bound $\gamma \varepsilon R_{\max} / (1 - \gamma)^2$.
> - UCB-VI plans on reward plus $H \sqrt{\log(2SAH/\delta) / (2N)}$, clipped at $H - h$, and its values stay at or above $Q^*$ with probability $1 - \delta$.
> - Robbins-Monro needs $\sum \alpha = \infty$ and $\sum \alpha^2 < \infty$. A constant $\alpha$ breaks the second and keeps a variance floor of $\sigma^2 \alpha / (2 - \alpha)$.

## Build it step by step

The checks test the functions in this order. `P` has shape `(S, A, S)` and `R` has shape `(S, A)`.

1. **`simulation_lemma_bound(P_hat, P, R, gamma)`.** Use `np.abs(P_hat - P).sum(axis=2).max()` for $\varepsilon$. The mistake is $1 / (1 - \gamma)$ without the square, which random MDPs hide.
2. **`policy_q(P, R, gamma, pi)`.** Solve `(I - gamma * P_pi) v = r_pi` with `P_pi = np.einsum("sa,sat->st", pi, P)`, then return `R + gamma * P @ v`.
3. **`simulation_gap(P_hat, P, R, gamma, pi)`.** Reshape $\hat P^\pi$ to `(S * A, S * A)` and use `np.linalg.solve` on `(P - P_hat) @ v`. The mistake is building the matrix from `P` instead of `P_hat`.
4. **`hoeffding_bonus(N, horizon, delta)`.** Use `N.size` for $SA$ and `np.maximum(N, 1)` for the count.
5. **`backward_induction(P, R, horizon, clip)`.** Loop `h` from `horizon - 1` to 0 with `Q[h] = R + P @ v_next`. With `clip`, cap `Q[h]` at `horizon - h`.
6. **`finite_horizon_value(P, R, pi)`.** The same loop with the policy's average in place of the max.
7. **`ucbvi(...)`.** Each episode: build `P_hat`, plan with the bonus, record `Q`, compute the exact regret of the policy you will play, sample the episode, update the counts. The mistake is leaving the clip out.
8. **`step_size_sums` and `estimator_variance`.** The second needs the product of $(1 - \alpha_j)$ over the *later* steps, a reversed cumulative product.

## Where this breaks

**The constants do not transfer.** A bound for every MDP carries a constant sized for the worst one. Real implementations scale the bonus down, which gives up the guarantee and keeps the idea.

**The setting is simplified.** The exercise is tabular, with known rewards and a fixed horizon. The paper also learns the rewards, and the book's version lets the transitions change with the step.

Exploration also gets cheaper when one decision lasts many steps, which is the subject of [Options and hierarchical RL](/lessons/049-options-and-hierarchical-rl).

## Further reading

- [Agarwal, Jiang, Kakade, and Sun, Reinforcement Learning: Theory and Algorithms](https://rltheorybook.github.io/). Chapter 2, "Sample Complexity", states the simulation lemma and the minimax model-based rate, and chapter 6, "Strategic Exploration in Tabular MDPs", gives UCB-VI. Chapter numbers are from the December 2020 working draft.
- [Azar, Osband, and Munos (2017), Minimax regret bounds for reinforcement learning](https://arxiv.org/abs/1703.05449). UCB-VI, with Hoeffding and Bernstein bonuses.
- [Strehl and Littman (2008), An analysis of model-based interval estimation for Markov decision processes](https://doi.org/10.1016/j.jcss.2007.08.009). The count-bonus algorithm from the previous chapter, with its analysis.
- [Robbins and Monro (1951), A stochastic approximation method](https://doi.org/10.1214/aoms/1177729586). The two step-size conditions.
- [CS 285, Deep Reinforcement Learning, UC Berkeley, Spring 2026](https://rail.eecs.berkeley.edu/deeprlcourse/). Lecture 19, "Exploration", lecture 20, "RL Theory", and lecture 23, "Advanced Exploration".
- [CS234, Reinforcement Learning, Stanford, Winter 2026](https://web.stanford.edu/class/cs234/). The schedule lists exploration on February 11, 18, and 23, with no lecture numbers.

## The exercise

You build the pieces of the guarantee and then run it. Write `simulation_lemma_bound`, `policy_q`, and `simulation_gap`, then `hoeffding_bonus`, `backward_induction`, `finite_horizon_value`, and `ucbvi`, and finish with `step_size_sums` and `estimator_variance`.

The checks verify the bound on a hand-worked example and on 200 random MDP pairs, and the exact identity against `policy_q`. On RiverSwim, `ucbvi` with the full bonus must keep $\tilde Q \ge Q^*$ everywhere in every episode on five seeds, and with a tenth of the bonus it must have under a quarter of epsilon-greedy's regret on the same seeds. The last check confirms the Robbins-Monro sums and the exact variance.
