---
title: Attack and defense as a game
slug: 088-attack-and-defense-as-a-game
part: "Part 10 — Beyond text, and what can go wrong"
summary: Treat red-teaming as a zero-sum game, and learn attack and defense mixes that end the patch-and-counterpatch cycle.
minutes: 40
gpu: false
objectives:
  - Say why patching the most recent attack cycles forever instead of settling.
  - Build the attacker-versus-defender loss table with an over-refusal cost, and measure a pair of mixes with exploitability.
  - Update EXP3 weights from success-or-failure feedback with the 1/p importance weight, and Hedge weights from full losses.
  - Trace the defender's attack-success against over-refusal frontier as the cost weight grows.
prereqs:
  - 006-gradient-bandits
  - 061-multi-agent-and-self-play
  - 079-safety-training-against-a-spec
exercise: 088-attack-and-defense-as-a-game
---

# Attack and defense as a game

A lab ships a chat model with a safety filter in front of it. A red team finds a role-play framing that gets past the filter, so the lab patches the filter. A week later, someone finds an encoded request the patch misses, so the lab adds a second patch. Each patch also refuses some harmless requests. Does this ever finish?

It doesn't, and the reason is structural. The attacker picks an attack knowing the current defense, and the defender picks a defense knowing the most recent attack. Each side answers the other's last move, the chase that [Many agents and self-play](/lessons/061-multi-agent-and-self-play) showed going in circles. This chapter writes the chase as a game, shows the cycle with numbers, and replaces it with two learners that mix over their options. Their averages settle at the game's value, and the defender's mix traces a trade-off between letting attacks through and refusing harmless requests.

An attack here is the name of a family, never a prompt, and nothing in this chapter is a working jailbreak.

## What you already know

- Exploitability measures how much two players could gain by best-responding, and it is zero only at an equilibrium ([061](/lessons/061-multi-agent-and-self-play)).
- A no-regret learner's average strategy approaches equilibrium in a zero-sum game, while its last strategy might not.
- In a bandit you see only the reward of the arm you pulled ([006](/lessons/006-gradient-bandits)).
- A safety spec trades refusing harmful requests against refusing harmless ones ([079](/lessons/079-safety-training-against-a-spec)).

## A game on a table of attacks and defenses

Picture a table. The attacker picks a row, one of three attack families: a role-play framing, an encoded request, or a multi-turn drift. The defender picks a column: a wrapper filter, a decoding patch, or a conversation classifier. Each cell holds the chance that the attack gets through. Each defense also refuses a share of *harmless* requests, its *over-refusal*.

The following table holds the numbers the whole chapter uses. They are made up to teach the idea, and a real table comes from running each family against each defense and counting.

| Attack family | Wrapper filter | Decoding patch | Conversation classifier |
|---|---|---|---|
| Role-play | 0.10 | 0.70 | 0.50 |
| Encoded | 0.60 | 0.10 | 0.70 |
| Multi-turn | 0.80 | 0.50 | 0.10 |
| Over-refusal | 0.05 | 0.10 | 0.25 |

Call the success table $S$ and the over-refusal row $c$. The defender pays a *loss* that adds the two costs, with a weight $\lambda$ for how many points of attack success you would give up to save one point of over-refusal:

$$
L_{ij} = S_{ij} + \lambda\, c_j
$$

Read it aloud: the loss of defense $j$ against attack $i$ is the chance the attack succeeds, plus $\lambda$ times the share of harmless requests that defense refuses. In numpy, that is `L = S + lam * c[None, :]`. At $\lambda = 1$ the losses are:

| Attack family | Wrapper filter | Decoding patch | Conversation classifier |
|---|---|---|---|
| Role-play | 0.15 | 0.80 | 0.75 |
| Encoded | 0.65 | 0.20 | 0.95 |
| Multi-turn | 0.85 | 0.60 | 0.35 |

The attacker wants the loss high and the defender wants it low, the zero-sum matrix game of lesson 061. The attacker never sees the $\lambda c_j$ term, but it adds the same amount to every row of column $j$, so it never changes which attack is best. The attacker can learn from success alone.

## Patching the most recent attack goes in circles

Why not patch each attack as it appears? Try it. The defender starts with the wrapper filter. The attacker picks the largest entry in that column, the multi-turn drift, at loss 0.85. The defender answers with the smallest entry in that row, the classifier, at 0.35. The attacker answers the classifier with the encoded request, at 0.95, the defender patches with the decoding patch, at 0.20, and the attacker answers that with role-play, at 0.80. The defender returns to the wrapper filter at 0.15, where it started.

```figure
<svg viewBox="0 0 600 340" role="img" aria-label="Six steps in a loop: the attacker picks the attack that beats the current defense, then the defender patches that attack, and after six steps the loop returns to the start">
<defs>
<marker id="c088-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker>
</defs>
<line x1="361.7" y1="77.0" x2="402.8" y2="91.0" class="stroke-muted" marker-end="url(#c088-arrow)"/>
<text x="396.3" y="78.0" class="fig-small fig-mono" text-anchor="middle">0.85</text>
<line x1="464.5" y1="133.0" x2="464.5" y2="203.0" class="stroke-muted" marker-end="url(#c088-arrow)"/>
<text x="484.5" y="172.0" class="fig-small fig-mono" text-anchor="middle">0.35</text>
<line x1="402.8" y1="245.0" x2="361.7" y2="259.0" class="stroke-muted" marker-end="url(#c088-arrow)"/>
<text x="396.3" y="266.0" class="fig-small fig-mono" text-anchor="middle">0.95</text>
<line x1="238.3" y1="259.0" x2="197.2" y2="245.0" class="stroke-muted" marker-end="url(#c088-arrow)"/>
<text x="203.7" y="266.0" class="fig-small fig-mono" text-anchor="middle">0.20</text>
<line x1="135.5" y1="203.0" x2="135.5" y2="133.0" class="stroke-muted" marker-end="url(#c088-arrow)"/>
<text x="115.5" y="172.0" class="fig-small fig-mono" text-anchor="middle">0.80</text>
<line x1="197.2" y1="91.0" x2="238.3" y2="77.0" class="stroke-muted" marker-end="url(#c088-arrow)"/>
<text x="203.7" y="78.0" class="fig-small fig-mono" text-anchor="middle">0.15</text>
<rect x="236.0" y="39.0" width="128" height="34" rx="8" class="fill-soft-a stroke-a"/>
<text x="300.0" y="60.0" class="fig-small fig-bold" text-anchor="middle">wrapper filter</text>
<rect x="400.5" y="95.0" width="128" height="34" rx="8" class="fill-soft-b stroke-b"/>
<text x="464.5" y="116.0" class="fig-small fig-bold" text-anchor="middle">multi-turn</text>
<rect x="400.5" y="207.0" width="128" height="34" rx="8" class="fill-soft-a stroke-a"/>
<text x="464.5" y="228.0" class="fig-small fig-bold" text-anchor="middle">classifier</text>
<rect x="236.0" y="263.0" width="128" height="34" rx="8" class="fill-soft-b stroke-b"/>
<text x="300.0" y="284.0" class="fig-small fig-bold" text-anchor="middle">encoded</text>
<rect x="71.5" y="207.0" width="128" height="34" rx="8" class="fill-soft-a stroke-a"/>
<text x="135.5" y="228.0" class="fig-small fig-bold" text-anchor="middle">decoding patch</text>
<rect x="71.5" y="95.0" width="128" height="34" rx="8" class="fill-soft-b stroke-b"/>
<text x="135.5" y="116.0" class="fig-small fig-bold" text-anchor="middle">role-play</text>
<text x="300" y="164" class="fig-small" text-anchor="middle">each arrow is a best reply;</text>
<text x="300" y="182" class="fig-small" text-anchor="middle">its number is the defender's loss</text>
</svg>
Figure 1. Patching the most recent attack at λ = 1, starting from the wrapper filter. Blue boxes are defenses, orange boxes are attacks. Notice that the loop closes after six steps, and no pair in it is stable.
```

Figure 1 shows the loop. Each step is a best reply to one move, so the sequence is *best-response dynamics*, and no pair in it is stable. Against (role-play, wrapper filter), the best attack pays 0.85 and the best defense against role-play pays 0.15, so the exploitability is $0.85 - 0.15 = 0.70$. The other two pairs sit at 0.60. In code, one round is:

```python
a = int(np.argmax(L[:, d]))   # the attacker answers the current defense
d = int(np.argmin(L[a]))      # the defender patches that attack
```

## The value of the game, by hand

The fix is to mix. Let the defender pick the wrapper filter, decoding patch, and classifier with probabilities $y = (14, 17, 16)/47$. Each attack then costs the defender the same amount:

- Role-play: $0.15 \cdot 14 + 0.80 \cdot 17 + 0.75 \cdot 16 = 2.1 + 13.6 + 12.0 = 27.7$.
- Encoded: $9.1 + 3.4 + 15.2 = 27.7$.
- Multi-turn: $11.9 + 10.2 + 5.6 = 27.7$.

Every attack costs $27.7/47 \approx 0.589$, so the attacker gains nothing by switching. The attacker mix $x = (60, 35, 93)/188$ does the same to every defense, so this is an equilibrium and 0.589 is the game's *value*. It splits into over-refusal $6.4/47 \approx 0.136$ and attack success $21.3/47 \approx 0.453$.

To measure how far any pair of mixes sits from this point, reuse exploitability from lesson 061, with $L$ in place of $A$:

$$
\operatorname{expl}(x, y) = \max_i\, (L y)_i - \min_j\, (x^\top L)_j
$$

It adds the most the attacker could gain by switching to its best attack and the most the defender could save by switching to its best defense. In numpy, that is `np.max(L @ y) - np.min(x @ L)`.

## No-regret learners reach the mix

The chase fails because each side reacts to one move. A *no-regret* learner reacts to the whole history, shifting weight slowly toward options that would have done better. As in lesson 061, the average strategies' exploitability equals the players' total regret divided by $T$, so averages approach the value when regret grows more slowly than $T$.

The defender uses *Hedge*. After each round it sees the attack and can score every defense against it, so it sees a full row of losses $L_{i_t, \cdot}$. It lowers each log weight by that loss:

$$
y_{t+1}(j) \propto \exp\Big(-\eta \sum_{s \le t} L_{i_s j}\Big)
$$

In numpy, that is `log_w -= eta * L[i]` and then `y = softmax(log_w)`.

The attacker sees only whether its own attack worked, which is *bandit feedback*, the setting of lesson 006. It uses *EXP3* (Auer, Cesa-Bianchi, Freund, and Schapire, 2002). With $K$ attacks and an exploration rate $\gamma$, it plays

$$
p_t(i) = (1 - \gamma)\,\frac{w_i}{\sum_k w_k} + \frac{\gamma}{K}
$$

and, after attack $i_t$ earns reward $r_t$, updates only that attack's weight:

$$
\hat r_{i_t} = \frac{r_t}{p_t(i_t)}, \qquad \log w_{i_t} \leftarrow \log w_{i_t} + \frac{\gamma\, \hat r_{i_t}}{K}
$$

The estimate divides by the chance of playing the attack, because an attack played rarely must count for more when it is played. That division makes the estimate *unbiased*: the expected estimate for attack $i$ is $p_t(i) \cdot r_i / p_t(i) = r_i$. Without it, an attack played often would look better only because it was played often.

### One EXP3 step

Take $K = 3$, $\gamma = 0.3$, and all log weights at 0, so each $p = 1/3$. The multi-turn drift is played and works with reward 0.8. Its estimate is $0.8/(1/3) = 2.4$, and its log weight grows by $0.3 \cdot 2.4/3 = 0.24$. The weights become $(1, 1, e^{0.24} = 1.271)$, and the probabilities become $(0.314, 0.314, 0.372)$.

Figure 2 runs EXP3 against Hedge on the loss table, with the attacker's reward set to its success rate. The average strategies' exploitability falls toward zero, while the strategies held in the last round stay far above it.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Exploitability of the average strategies of EXP3 against Hedge falls from about 0.07 to 0.013 over 10 to 10000 rounds, while the strategies held in the last round rise to 0.17">
<line x1="80.0" y1="230.0" x2="560.0" y2="230.0" class="stroke-muted"/>
<text x="72.0" y="234.0" class="fig-small" text-anchor="end">0</text>
<line x1="80.0" y1="182.5" x2="560.0" y2="182.5" class="stroke-grid"/>
<text x="72.0" y="186.5" class="fig-small" text-anchor="end">0.05</text>
<line x1="80.0" y1="135.0" x2="560.0" y2="135.0" class="stroke-grid"/>
<text x="72.0" y="139.0" class="fig-small" text-anchor="end">0.1</text>
<line x1="80.0" y1="87.5" x2="560.0" y2="87.5" class="stroke-grid"/>
<text x="72.0" y="91.5" class="fig-small" text-anchor="end">0.15</text>
<line x1="80.0" y1="40.0" x2="560.0" y2="40.0" class="stroke-grid"/>
<text x="72.0" y="44.0" class="fig-small" text-anchor="end">0.2</text>
<line x1="80.0" y1="230.0" x2="80.0" y2="235.0" class="stroke-muted"/>
<text x="80.0" y="250.0" class="fig-small" text-anchor="middle">10</text>
<line x1="240.0" y1="230.0" x2="240.0" y2="235.0" class="stroke-muted"/>
<text x="240.0" y="250.0" class="fig-small" text-anchor="middle">100</text>
<line x1="400.0" y1="230.0" x2="400.0" y2="235.0" class="stroke-muted"/>
<text x="400.0" y="250.0" class="fig-small" text-anchor="middle">1,000</text>
<line x1="560.0" y1="230.0" x2="560.0" y2="235.0" class="stroke-muted"/>
<text x="560.0" y="250.0" class="fig-small" text-anchor="middle">10,000</text>
<polyline points="80.0,165.2 156.3,163.1 240.0,154.5 316.3,148.4 400.0,167.5 476.3,207.5 560.0,218.1" class="stroke-a" fill="none"/>
<polyline points="80.0,165.4 156.3,157.6 240.0,148.7 316.3,146.8 400.0,152.0 476.3,112.5 560.0,68.4" class="stroke-b dashed" fill="none"/>
<circle cx="80.0" cy="165.2" r="4" class="fill-a"/>
<circle cx="156.3" cy="163.1" r="4" class="fill-a"/>
<circle cx="240.0" cy="154.5" r="4" class="fill-a"/>
<circle cx="316.3" cy="148.4" r="4" class="fill-a"/>
<circle cx="400.0" cy="167.5" r="4" class="fill-a"/>
<circle cx="476.3" cy="207.5" r="4" class="fill-a"/>
<circle cx="560.0" cy="218.1" r="4" class="fill-a"/>
<circle cx="80.0" cy="165.4" r="4" class="fill-b"/>
<circle cx="156.3" cy="157.6" r="4" class="fill-b"/>
<circle cx="240.0" cy="148.7" r="4" class="fill-b"/>
<circle cx="316.3" cy="146.8" r="4" class="fill-b"/>
<circle cx="400.0" cy="152.0" r="4" class="fill-b"/>
<circle cx="476.3" cy="112.5" r="4" class="fill-b"/>
<circle cx="560.0" cy="68.4" r="4" class="fill-b"/>
<text x="156.3" y="201.5" class="fig-small fig-bold" text-anchor="start">average strategies (solid)</text>
<text x="462.3" y="87.5" class="fig-small fig-bold" text-anchor="end">last strategies (dashed)</text>
<text x="560.0" y="28.0" class="fig-small" text-anchor="end">patch-the-last-attack: 0.6 to 0.7, off this scale</text>
<text x="320.0" y="284" class="fig-small" text-anchor="middle">rounds T (log scale)</text>
<text x="80.0" y="22" class="fig-small" text-anchor="start">exploitability, mean of 5 seeds</text>
</svg>
Figure 2. EXP3 (attacker) against Hedge (defender) at λ = 1, with γ = 0.03 and η = 0.02. Notice that the average strategies' exploitability sits near 0.09 for the first 300 rounds, then falls to 0.013, while the strategies held in the last round end at 0.17.
```

> [!KEY]
> Patching the most recent attack cycles, because each side answers one move. EXP3 and Hedge answer the whole history, and their average strategies approach the game's value.

## The frontier of attack success against over-refusal

The weight $\lambda$ is a choice, so sweep it. For each $\lambda$, solve the game and record the defender's mix $y$ by its over-refusal and its worst-case attack success. In numpy, the two numbers are:

```python
over = c @ y               # share of harmless requests refused
attack = (S @ y).max()     # success of the best attack against the mix
```

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Frontier of worst-case attack success against over-refusal: three points, with attack success 0.45 at over-refusal 0.136, 0.57 at 0.089, and 0.80 at 0.05">
<line x1="90.0" y1="230.0" x2="560.0" y2="230.0" class="stroke-grid"/>
<text x="82.0" y="234.0" class="fig-small" text-anchor="end">0.4</text>
<line x1="90.0" y1="187.8" x2="560.0" y2="187.8" class="stroke-grid"/>
<text x="82.0" y="191.8" class="fig-small" text-anchor="end">0.5</text>
<line x1="90.0" y1="145.6" x2="560.0" y2="145.6" class="stroke-grid"/>
<text x="82.0" y="149.6" class="fig-small" text-anchor="end">0.6</text>
<line x1="90.0" y1="103.3" x2="560.0" y2="103.3" class="stroke-grid"/>
<text x="82.0" y="107.3" class="fig-small" text-anchor="end">0.7</text>
<line x1="90.0" y1="61.1" x2="560.0" y2="61.1" class="stroke-grid"/>
<text x="82.0" y="65.1" class="fig-small" text-anchor="end">0.8</text>
<line x1="90.0" y1="230.0" x2="90.0" y2="235.0" class="stroke-muted"/>
<text x="90.0" y="250.0" class="fig-small" text-anchor="middle">0</text>
<line x1="207.5" y1="230.0" x2="207.5" y2="235.0" class="stroke-muted"/>
<text x="207.5" y="250.0" class="fig-small" text-anchor="middle">0.04</text>
<line x1="325.0" y1="230.0" x2="325.0" y2="235.0" class="stroke-muted"/>
<text x="325.0" y="250.0" class="fig-small" text-anchor="middle">0.08</text>
<line x1="442.5" y1="230.0" x2="442.5" y2="235.0" class="stroke-muted"/>
<text x="442.5" y="250.0" class="fig-small" text-anchor="middle">0.12</text>
<line x1="560.0" y1="230.0" x2="560.0" y2="235.0" class="stroke-muted"/>
<text x="560.0" y="250.0" class="fig-small" text-anchor="middle">0.16</text>
<line x1="90.0" y1="230.0" x2="560.0" y2="230.0" class="stroke-muted"/>
<polyline points="490.0,207.5 351.1,159.6 236.9,61.1" class="stroke-b" fill="none"/>
<circle cx="490.0" cy="207.5" r="5" class="fill-b"/>
<text x="500.0" y="199.5" class="fig-small fig-bold" text-anchor="start">λ = 0, 1, 2</text>
<circle cx="351.1" cy="159.6" r="5" class="fill-b"/>
<text x="361.1" y="151.6" class="fig-small fig-bold" text-anchor="start">λ = 4</text>
<circle cx="236.9" cy="61.1" r="5" class="fill-b"/>
<text x="246.9" y="53.1" class="fig-small fig-bold" text-anchor="start">λ = 8</text>
<text x="325.0" y="284" class="fig-small" text-anchor="middle">over-refusal rate of the defender's mix</text>
<text x="90.0" y="22" class="fig-small" text-anchor="start">worst-case attack success</text>
</svg>
Figure 3. The defender's equilibrium mix at five values of the over-refusal weight λ. Notice that the mix does not move for λ from 0 to 2, then trades attack success for lower over-refusal in two steps.
```

Figure 3 plots the sweep. The mix does not move for $\lambda$ from 0 to 2, because it leaves the attacker indifferent among its attacks, and that condition uses $S$ alone. At $\lambda = 4$ the defender drops the classifier, and at $\lambda = 8$ it plays the cheap wrapper filter alone. Over-refusal falls from 0.136 to 0.089 to 0.050, and worst-case attack success rises from 0.453 to 0.567 to 0.800.

The points never go backward. Write $g$ for worst-case attack success and $h$ for over-refusal, so the mix at weight $\lambda$ minimizes $g + \lambda h$. For $\lambda < \lambda'$ with mixes $y$ and $y'$, each mix is no worse than the other under its own weight. Adding the two inequalities gives $(\lambda' - \lambda)(h(y') - h(y)) \le 0$, so over-refusal does not rise, and attack success does not fall.

> [!KEY]
> The over-refusal weight picks a point on the frontier, and a larger weight buys lower over-refusal with higher attack success.

> [!RECAP]
> - The loss is $S_{ij} + \lambda c_j$. The attacker maximizes it and the defender minimizes it.
> - Patching the last attack loops through pairs with exploitability of at least 0.6.
> - The value at $\lambda = 1$ is 27.7/47.
> - EXP3 divides the reward by $p_t(i)$, Hedge subtracts $\eta$ times the loss row, and the averages converge.
> - Over-refusal never rises as $\lambda$ grows.

## Build it step by step

The checks test these functions in this order. `SUCCESS`, `OVER_REFUSAL`, and `softmax` are written for you.

1. **`loss_matrix(success, over_refusal, lam)`.** Add `lam * over_refusal[None, :]`. The common mistake is charging each attack, not each defense.
2. **`exploitability(L, x, y)`.** One line. The common mistake is `max` for the second term.
3. **`best_response_dynamics(L, first_defense, rounds)`.** Each round, `argmax` the defense's column, then `argmin` that attack's row, and return one `(attack, defense)` pair per round. The common mistake is `argmax` for the defender.
4. **`exp3_strategy(log_w, gamma)` and `exp3_update(log_w, arm, reward, gamma)`.** Mix `softmax(log_w)` with the uniform floor. In the update, divide by `p[arm]` and return a copy. The common mistake is leaving out the division.
5. **`hedge_update(log_w, losses, eta)`.** Subtract `eta * losses`. The common mistake is the sign.
6. **`train_no_regret(success, over_refusal, lam, rounds, gamma, eta, seed)`.** Draw the attack, then the defense, from one generator. Update EXP3 with `success[i, j]` and Hedge with `L[i]`, and average the strategies held. The common mistake is averaging the actions drawn.
7. **`solve_game(L)`.** For each pair of equal-size supports, solve each side's linear system with `np.linalg.solve`. Accept the first pair with non-negative mixes and no profitable deviation outside the supports. The common mistake is skipping the deviation test.
8. **`frontier(success, over_refusal, lams)`.** For each weight, solve the game and report the two numbers above. The common mistake is reporting `L @ y`, which adds them.

## Where this breaks

**The attack bank is a stand-in.** A real attacker searches an open space. Zou and colleagues (2023) automate that search with gradient-guided greedy token swaps, and Perez and colleagues (2022) use a language model to write test cases. The equilibrium of the table is not an equilibrium against a family you left out.

**The cells are estimates.** A small gap between two defenses can be noise, so put an interval on each cell, as lesson 059 does.

**Mixing needs randomized deployment.** A defender must draw a defense per request or per session, because a fixed defense that an attacker can probe brings the cycle back.

**The guarantees are looser than the picture.** EXP3's regret bound holds in expectation, a fixed seed hides the spread, and the last strategies carry no guarantee, as Figure 2 shows. One weight $\lambda$ also puts a leaked attack and a refused harmless request on one scale, which is a policy decision.

## Further reading

- [Zou, Wang, Carlini, Nasr, Kolter, and Fredrikson (2023), Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043). Greedy, gradient-based search for adversarial suffixes.
- [Perez and colleagues (2022), Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286). A language model writes test cases for another.
- [Sharma and colleagues (2025), Constitutional Classifiers](https://arxiv.org/abs/2501.18837). Classifiers tested in over 3,000 estimated hours of red teaming.
- [Zaremba and colleagues (2025), Trading Inference-Time Compute for Adversarial Robustness](https://arxiv.org/abs/2501.18841). More inference-time compute often lowers attack success, with exceptions.
- [Auer, Cesa-Bianchi, Freund, and Schapire (2002), The Nonstochastic Multiarmed Bandit Problem](https://doi.org/10.1137/S0097539701398375). Introduces EXP3 and its regret bound.
- [Pinto, Davidson, Sukthankar, and Gupta (2017), Robust Adversarial Reinforcement Learning](https://arxiv.org/abs/1703.02702). A policy trained against a learned adversary.
- [CS 2881 AI Safety, Lecture 3: Robustness](https://www.youtube.com/watch?v=pfKO4MlvM-Y). Adversarial attacks and classifier defenses.

Researchers are still working on the following questions:

- How large must an attack bank be before its equilibrium predicts what a search-based attacker finds?
- Which robustness gains from extra inference-time compute survive an attacker who targets that compute?

## The exercise

You build the tools for the attack-and-defense game. Write `loss_matrix` and `exploitability`, then `best_response_dynamics` to see the cycle. Write `exp3_strategy`, `exp3_update`, and `hedge_update`, then `train_no_regret` to run them against each other. Finish with `solve_game` and `frontier`.

The checks confirm the hand computations. Exploitability is 0.7 for (role-play, wrapper filter), best-response dynamics repeat every three rounds, and an EXP3 step gives the increment 0.24. The value at $\lambda = 1$ is 27.7/47, and `solve_game` agrees with a linear program. After 10,000 rounds the average strategies are exploitable by under 0.04, and the frontier's over-refusal never rises as $\lambda$ grows.
