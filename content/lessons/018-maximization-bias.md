---
title: Maximization bias and double learning
slug: 018-maximization-bias
part: "Part 3 — Learning from experience"
summary: Show that a max over noisy estimates overshoots, and fix Q-learning by choosing the action with one table and scoring it with another.
minutes: 35
gpu: false
objectives:
  - Explain why the max of unbiased noisy estimates is biased upward.
  - Separate choosing an action from scoring it with the double estimator.
  - Implement the double Q-learning target and show it removes the bias on the maximization-bias MDP.
  - Build the Expected SARSA target and say what noise it removes.
prereqs:
  - 017-sarsa-and-q-learning
exercise: 018-maximization-bias
---

# Maximization bias and double learning

Ten people guess the weight of a pumpkin. Each guess is fair: on average it lands on the true weight. Now suppose you report only the highest guess. You overestimate every time, because the highest guess is usually the one whose error happened to be positive.

Q-learning does the same thing. Its target takes a max over estimated action values, and each of those estimates is noisy. If every estimate is right on average, is their max right on average too? It is not. This is *maximization bias*, and it compounds, because each biased target feeds the next update.

This chapter measures the bias, shows how it steers an agent to a worse action, and removes it with *double learning*: one set of estimates picks the action and a second, independent set scores it. A short section at the end covers Expected SARSA, which removes a different kind of noise from the target.

## What you already know

- An estimate is *unbiased* if its average over many repetitions equals the true value.
- The Q-learning target is $r_t + \gamma \max_{a'} Q(s_{t+1}, a')$ (see [SARSA and Q-learning](/lessons/017-sarsa-and-q-learning)).
- The SARSA target is $r_t + \gamma Q(s_{t+1}, a_{t+1})$, using the action actually taken next.
- An $\varepsilon$-greedy policy gives every action $\varepsilon / A$, and gives the greedy action another $1 - \varepsilon$.
- A noisy average gets less noisy as you add samples: its standard deviation shrinks by $\sqrt{n}$ for $n$ samples.

## See the bias in a max

Start with the smallest case. Two actions are both truly worth 0. You estimate each from one noisy sample, $X_1$ and $X_2$, each a standard normal draw. Each estimate is unbiased, because its average is 0.

Now take the max. It is negative only when both samples are negative, which happens a quarter of the time. In the other three quarters, at least one sample is positive, and the max picks it. So the max is positive far more often than negative, and its average is

$$
\mathbb{E}\bigl[\max(X_1, X_2)\bigr] = \frac{1}{\sqrt{\pi}} \approx 0.564
$$

Read it aloud: two unbiased guesses at a value of 0, and their max averages 0.564. That is more than half a standard deviation of bias, and nothing was wrong with either guess. More actions give more chances for one of them to be lucky.

```figure
<svg viewBox="0 0 580 270" role="img" aria-label="Average of the max of n noisy estimates of actions that are all worth zero, growing with n">
<line x1="70.0" y1="220.0" x2="540.0" y2="220.0" class="stroke-grid"/>
<text x="62.0" y="224.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="172.5" x2="540.0" y2="172.5" class="stroke-grid"/>
<text x="62.0" y="176.5" class="fig-small" text-anchor="end">0.5</text>
<line x1="70.0" y1="125.0" x2="540.0" y2="125.0" class="stroke-grid"/>
<text x="62.0" y="129.0" class="fig-small" text-anchor="end">1</text>
<line x1="70.0" y1="77.5" x2="540.0" y2="77.5" class="stroke-grid"/>
<text x="62.0" y="81.5" class="fig-small" text-anchor="end">1.5</text>
<line x1="70.0" y1="30.0" x2="540.0" y2="30.0" class="stroke-grid"/>
<text x="62.0" y="34.0" class="fig-small" text-anchor="end">2</text>
<line x1="70.0" y1="220.0" x2="540.0" y2="220.0" class="stroke-muted"/>
<text x="62.0" y="18.0" text-anchor="end">max</text>
<rect x="86.2" y="220.0" width="46.0" height="0.0" class="fill-b"/>
<text x="109.2" y="214.0" class="fig-small" text-anchor="middle">0.00</text>
<text x="109.2" y="238.0" class="fig-small" text-anchor="middle">1 action</text>
<rect x="164.5" y="166.4" width="46.0" height="53.6" class="fill-b"/>
<text x="187.5" y="160.4" class="fig-small" text-anchor="middle">0.56</text>
<text x="187.5" y="238.0" class="fig-small" text-anchor="middle">2 actions</text>
<rect x="242.8" y="139.6" width="46.0" height="80.4" class="fill-b"/>
<text x="265.8" y="133.6" class="fig-small" text-anchor="middle">0.85</text>
<text x="265.8" y="238.0" class="fig-small" text-anchor="middle">3 actions</text>
<rect x="321.2" y="109.5" width="46.0" height="110.5" class="fill-b"/>
<text x="344.2" y="103.5" class="fig-small" text-anchor="middle">1.16</text>
<text x="344.2" y="238.0" class="fig-small" text-anchor="middle">5 actions</text>
<rect x="399.5" y="73.8" width="46.0" height="146.2" class="fill-b"/>
<text x="422.5" y="67.8" class="fig-small" text-anchor="middle">1.54</text>
<text x="422.5" y="238.0" class="fig-small" text-anchor="middle">10 actions</text>
<rect x="477.8" y="42.6" width="46.0" height="177.4" class="fill-b"/>
<text x="500.8" y="36.6" class="fig-small" text-anchor="middle">1.87</text>
<text x="500.8" y="238.0" class="fig-small" text-anchor="middle">20 actions</text>
<line x1="70.0" y1="220.0" x2="540.0" y2="220.0" class="stroke-ink"/>
</svg>
Figure 1. Every action is truly worth 0, and each estimate is one standard normal draw, so each estimate is unbiased. Notice that the max of them, plotted as its average, climbs steadily with the number of actions.
```

With 10 actions, all worth 0, the max averages 1.54. If you average 5 samples for each estimate, the noise shrinks by $\sqrt 5$, and so does the bias: $1.539 / \sqrt 5 = 0.688$. Less noise helps, but it does not remove the bias. The general fact is an inequality, true for any estimates $\hat Q(a)$:

$$
\mathbb{E}\Bigl[\max_a \hat Q(a)\Bigr] \ge \max_a \mathbb{E}\bigl[\hat Q(a)\bigr]
$$

Read it aloud: the average of the max is at least the max of the averages. It holds because the max function is convex, and Jensen's inequality turns that into a statement about averages. In numpy, draw one row per trial and one column per action, and take the max along the last axis:

```python
X = rng.standard_normal((n_trials, n_samples, n_actions)).mean(axis=1)
bias = X.max(axis=-1).mean()          # about 0.688 for 5 samples and 10 actions
```

> [!KEY]
> The max of unbiased noisy estimates is biased upward. The bias grows with the number of actions and the size of the noise.

## Choose with one estimate, score with another

Why does the max overshoot? It uses the same noise twice. The estimate $X$ picks the action that looks best, and then that same $X$ reports how good the action is. The action was chosen partly because its noise was high, so its reported value is high too.

The fix is to stop reusing the noise. The *double estimator* keeps two independent sets of estimates, $X$ and $Y$. One chooses, and the other scores:

$$
\hat a = \operatorname*{arg\,max}_a X(a), \qquad \text{estimate} = Y(\hat a)
$$

Read it aloud: let $X$ pick the action that looks best, and then ask $Y$, which never saw $X$'s noise, how good that action is. Given the choice $\hat a$, $Y(\hat a)$ is an unbiased estimate of that action's true value. In numpy:

```python
pick = np.argmax(X, axis=-1)
double = np.take_along_axis(Y, pick[..., None], axis=-1)[..., 0]
```

Here is a worked example with three actions, $X = (1, 3, 2)$ and $Y = (0.5, 1, 4)$.

```figure
<svg viewBox="0 0 580 275" role="img" aria-label="Two arrays of three action estimates: X picks action 1, and Y gives its estimate of that action, 1, instead of X's own 3">
<defs><marker id="c018-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<text x="100.0" y="30.0" class="fig-bold" text-anchor="middle">X chooses: argmax is a1</text>
<rect x="30.0" y="162.0" width="40.0" height="28.0" class="fill-soft-a"/>
<text x="50.0" y="156.0" class="fig-small" text-anchor="middle">1</text>
<text x="50.0" y="206.0" class="fig-small" text-anchor="middle">a0</text>
<rect x="82.0" y="106.0" width="40.0" height="84.0" class="fill-soft-a"/>
<text x="102.0" y="100.0" class="fig-small" text-anchor="middle">3</text>
<text x="102.0" y="206.0" class="fig-small" text-anchor="middle">a1</text>
<rect x="134.0" y="134.0" width="40.0" height="56.0" class="fill-soft-a"/>
<text x="154.0" y="128.0" class="fig-small" text-anchor="middle">2</text>
<text x="154.0" y="206.0" class="fig-small" text-anchor="middle">a2</text>
<rect x="79" y="103" width="46" height="90" class="stroke-ink" fill="none" stroke-width="2"/>
<line x1="24.0" y1="190.0" x2="186.0" y2="190.0" class="stroke-muted"/>
<text x="330.0" y="30.0" class="fig-bold" text-anchor="middle">Y scores a1: reads 1</text>
<rect x="260.0" y="176.0" width="40.0" height="14.0" class="fill-soft-d"/>
<text x="280.0" y="170.0" class="fig-small" text-anchor="middle">0.5</text>
<text x="280.0" y="206.0" class="fig-small" text-anchor="middle">a0</text>
<rect x="312.0" y="162.0" width="40.0" height="28.0" class="fill-soft-d"/>
<text x="332.0" y="156.0" class="fig-small" text-anchor="middle">1</text>
<text x="332.0" y="206.0" class="fig-small" text-anchor="middle">a1</text>
<rect x="364.0" y="78.0" width="40.0" height="112.0" class="fill-soft-d"/>
<text x="384.0" y="72.0" class="fig-small" text-anchor="middle">4</text>
<text x="384.0" y="206.0" class="fig-small" text-anchor="middle">a2</text>
<rect x="309" y="159" width="46" height="34" class="stroke-ink" fill="none" stroke-width="2"/>
<line x1="254.0" y1="190.0" x2="416.0" y2="190.0" class="stroke-muted"/>
<line x1="186.0" y1="120.0" x2="254.0" y2="120.0" class="stroke-ink" marker-end="url(#c018-arrow)"/>
<text x="220.0" y="110.0" class="fig-small" text-anchor="middle">a1</text>
<text x="30.0" y="238.0" class="fig-small fig-bold" text-anchor="start">single estimate: max X = 3</text>
<text x="260.0" y="238.0" class="fig-small fig-bold" text-anchor="start">double estimate: Y[a1] = 1</text>
<text x="30.0" y="258.0" class="fig-small" text-anchor="start">X lifts a1, then vouches for it.</text>
<text x="260.0" y="258.0" class="fig-small" text-anchor="start">Y never saw that noise.</text>
</svg>
Figure 2. The same three actions estimated twice. The single estimate reads X at its own favorite and gets 3. The double estimate reads Y there and gets 1. Notice that the two arrays do not have to agree.
```

The single estimate is $\max X = 3$. $X$ picks action 1, and $Y$ scores it at 1, so the double estimate is 1. If you swap the roles, $Y$ picks action 2, and $X$ scores it at 2.

When all true values are equal, the double estimate is exactly unbiased. When they differ, it errs low, because $X$ sometimes picks a worse action. Take true values $(0, 0.5)$ with unit noise. $X$ picks the better action with probability $\Phi(0.5 / \sqrt 2) = 0.638$, where $\Phi$ is the standard normal cumulative distribution function. So the double estimate averages $0.638 \times 0.5 = 0.319$, below the true max of 0.5. The single estimate averages 0.850, above it. A small underestimate is the price of removing the overestimate.

## Watch the bias steer an agent

Sutton and Barto's Example 6.7 turns the bias into bad behavior. An episode starts in state $A$ with two actions. *Right* ends the episode with reward 0. *Left* moves to state $B$ with reward 0. State $B$ has 10 actions, and each one ends the episode with a reward drawn from a normal distribution with mean $-0.1$ and standard deviation 1. Going left is worth $-0.1$, so right is optimal.

Q-learning updates $Q(A, \text{left})$ toward $\max_b Q(B, b)$. After a few visits to $B$, some of the 10 estimates have seen lucky positive rewards, so the max is positive. That lifts $Q(A, \text{left})$ above $Q(A, \text{right}) = 0$, and the agent goes left. It keeps going left until the estimates at $B$ average out enough noise, which takes many episodes.

```figure
<svg viewBox="0 0 580 290" role="img" aria-label="Fraction of runs that go left from state A over 200 episodes for Q-learning and double Q-learning">
<line x1="70.0" y1="230.0" x2="540.0" y2="230.0" class="stroke-grid"/>
<text x="62.0" y="234.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="180.0" x2="540.0" y2="180.0" class="stroke-grid"/>
<text x="62.0" y="184.0" class="fig-small" text-anchor="end">0.25</text>
<line x1="70.0" y1="130.0" x2="540.0" y2="130.0" class="stroke-grid"/>
<text x="62.0" y="134.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="70.0" y1="80.0" x2="540.0" y2="80.0" class="stroke-grid"/>
<text x="62.0" y="84.0" class="fig-small" text-anchor="end">0.75</text>
<line x1="70.0" y1="30.0" x2="540.0" y2="30.0" class="stroke-grid"/>
<text x="62.0" y="34.0" class="fig-small" text-anchor="end">1</text>
<line x1="70.0" y1="230.0" x2="540.0" y2="230.0" class="stroke-muted"/>
<line x1="70.0" y1="230.0" x2="70.0" y2="235.0" class="stroke-muted"/>
<text x="70.0" y="250.0" class="fig-small" text-anchor="middle">0</text>
<line x1="187.5" y1="230.0" x2="187.5" y2="235.0" class="stroke-muted"/>
<text x="187.5" y="250.0" class="fig-small" text-anchor="middle">50</text>
<line x1="305.0" y1="230.0" x2="305.0" y2="235.0" class="stroke-muted"/>
<text x="305.0" y="250.0" class="fig-small" text-anchor="middle">100</text>
<line x1="422.5" y1="230.0" x2="422.5" y2="235.0" class="stroke-muted"/>
<text x="422.5" y="250.0" class="fig-small" text-anchor="middle">150</text>
<line x1="540.0" y1="230.0" x2="540.0" y2="235.0" class="stroke-muted"/>
<text x="540.0" y="250.0" class="fig-small" text-anchor="middle">200</text>
<text x="305.0" y="272.0" text-anchor="middle">episodes</text>
<text x="62.0" y="18.0" text-anchor="end">share left</text>
<line x1="70.0" y1="220.0" x2="540.0" y2="220.0" class="stroke-muted" stroke-dasharray="4 4"/>
<polyline points="81.8,120.4 84.1,112.0 86.5,101.2 88.8,90.4 91.2,77.6 93.5,71.6 95.8,62.0 98.2,56.4 100.5,52.8 102.9,51.6 105.2,48.8 107.6,46.4 110.0,44.8 112.3,44.4 114.7,43.2 117.0,41.6 119.3,44.8 121.7,45.2 124.1,44.4 126.4,44.8 128.8,45.2 131.1,43.6 133.4,44.4 135.8,45.2 138.1,45.6 140.5,45.2 142.8,45.6 145.2,48.4 147.6,50.4 149.9,53.6 152.2,57.6 154.6,61.2 156.9,62.0 159.3,64.0 161.7,63.2 164.0,64.4 166.3,63.2 168.7,61.2 171.1,62.4 173.4,64.8 175.8,65.6 178.1,68.0 180.4,70.0 182.8,71.2 185.1,71.6 187.5,73.2 189.9,76.0 192.2,79.2 194.6,78.8 196.9,79.6 199.2,78.8 201.6,76.8 203.9,76.8 206.3,78.4 208.7,77.6 211.0,77.2 213.3,76.8 215.7,76.8 218.1,75.2 220.4,78.0 222.8,78.0 225.1,82.4 227.5,84.8 229.8,88.8 232.1,90.0 234.5,96.0 236.8,97.2 239.2,100.4 241.5,102.8 243.9,105.2 246.2,107.6 248.6,108.4 251.0,109.2 253.3,109.6 255.7,113.6 258.0,114.8 260.4,116.8 262.7,118.0 265.0,120.4 267.4,118.8 269.8,118.4 272.1,118.8 274.4,120.0 276.8,122.4 279.1,123.6 281.5,125.6 283.9,128.0 286.2,128.4 288.6,128.0 290.9,130.4 293.2,129.2 295.6,129.2 297.9,130.4 300.3,129.6 302.6,128.4 305.0,130.4 307.4,132.4 309.7,134.4 312.1,138.4 314.4,141.2 316.8,143.6 319.1,141.6 321.5,141.6 323.8,141.2 326.2,142.8 328.5,144.4 330.9,148.8 333.2,151.2 335.5,152.8 337.9,155.6 340.2,156.4 342.6,158.4 344.9,159.2 347.3,160.4 349.6,158.0 352.0,156.8 354.3,154.0 356.7,153.6 359.1,154.4 361.4,155.6 363.8,158.0 366.1,161.2 368.4,161.6 370.8,160.8 373.2,163.2 375.5,165.2 377.9,165.2 380.2,167.6 382.6,170.4 384.9,171.2 387.2,170.4 389.6,172.8 392.0,174.0 394.3,176.4 396.6,175.6 399.0,175.6 401.3,176.0 403.7,176.0 406.1,174.8 408.4,174.8 410.8,176.0 413.1,176.8 415.4,179.2 417.8,179.2 420.1,183.2 422.5,184.0 424.9,184.4 427.2,184.4 429.6,187.2 431.9,189.6 434.2,194.0 436.6,197.6 438.9,198.4 441.3,200.4 443.7,200.0 446.0,198.8 448.4,197.6 450.7,198.0 453.0,198.0 455.4,196.8 457.8,196.0 460.1,195.2 462.4,194.4 464.8,192.4 467.1,192.8 469.5,194.4 471.8,195.6 474.2,197.6 476.6,198.0 478.9,197.6 481.2,198.0 483.6,198.4 485.9,197.2 488.3,199.6 490.7,200.8 493.0,200.0 495.4,198.4 497.7,198.4 500.1,196.4 502.4,194.4 504.8,193.6 507.1,196.0 509.5,195.6 511.8,194.0 514.1,194.4 516.5,196.0 518.8,194.0 521.2,194.0 523.5,195.6 525.9,196.4 528.2,194.4 530.6,194.0 533.0,193.6 535.3,192.8 537.6,193.2 540.0,193.2" class="stroke-b" fill="none"/>
<polyline points="81.8,138.4 84.1,133.2 86.5,127.6 88.8,123.6 91.2,126.8 93.5,128.8 95.8,134.4 98.2,137.2 100.5,138.8 102.9,137.6 105.2,140.4 107.6,141.2 110.0,147.2 112.3,152.8 114.7,154.8 117.0,158.8 119.3,166.0 121.7,169.6 124.1,172.0 126.4,176.8 128.8,178.8 131.1,176.4 133.4,175.6 135.8,174.0 138.1,174.0 140.5,174.4 142.8,176.0 145.2,179.2 147.6,181.2 149.9,183.6 152.2,184.4 154.6,187.6 156.9,188.4 159.3,190.4 161.7,190.0 164.0,191.2 166.3,189.6 168.7,189.2 171.1,190.8 173.4,192.8 175.8,192.8 178.1,194.4 180.4,194.4 182.8,195.6 185.1,194.8 187.5,196.0 189.9,194.8 192.2,196.0 194.6,194.4 196.9,196.0 199.2,196.4 201.6,198.0 203.9,198.0 206.3,199.2 208.7,200.0 211.0,200.8 213.3,200.8 215.7,202.4 218.1,202.8 220.4,203.6 222.8,203.6 225.1,205.6 227.5,204.8 229.8,206.8 232.1,207.2 234.5,207.2 236.8,208.4 239.2,210.4 241.5,209.6 243.9,208.4 246.2,210.0 248.6,209.2 251.0,208.0 253.3,208.8 255.7,210.4 258.0,212.4 260.4,214.0 262.7,216.4 265.0,217.6 267.4,218.4 269.8,217.2 272.1,216.4 274.4,216.0 276.8,217.2 279.1,218.0 281.5,216.8 283.9,217.6 286.2,218.0 288.6,217.2 290.9,216.4 293.2,216.0 295.6,215.2 297.9,215.6 300.3,214.8 302.6,215.6 305.0,217.2 307.4,217.6 309.7,217.6 312.1,218.4 314.4,217.2 316.8,217.2 319.1,218.8 321.5,219.2 323.8,218.8 326.2,219.6 328.5,220.0 330.9,220.4 333.2,218.4 335.5,218.4 337.9,217.2 340.2,216.0 342.6,214.0 344.9,215.2 347.3,215.6 349.6,216.8 352.0,217.2 354.3,218.0 356.7,216.8 359.1,215.6 361.4,213.6 363.8,212.8 366.1,212.4 368.4,212.4 370.8,212.8 373.2,211.6 375.5,212.0 377.9,210.4 380.2,209.6 382.6,208.0 384.9,208.0 387.2,207.6 389.6,207.6 392.0,207.2 394.3,208.0 396.6,211.2 399.0,211.2 401.3,212.4 403.7,213.6 406.1,215.2 408.4,215.2 410.8,216.4 413.1,216.8 415.4,216.4 417.8,215.6 420.1,214.8 422.5,215.6 424.9,213.2 427.2,211.2 429.6,209.6 431.9,209.2 434.2,207.6 436.6,207.2 438.9,210.4 441.3,211.6 443.7,212.0 446.0,212.0 448.4,213.2 450.7,214.4 453.0,215.6 455.4,215.6 457.8,215.2 460.1,216.4 462.4,214.8 464.8,213.6 467.1,213.6 469.5,214.8 471.8,214.0 474.2,214.0 476.6,214.4 478.9,214.0 481.2,214.4 483.6,214.8 485.9,215.2 488.3,215.2 490.7,215.2 493.0,213.6 495.4,214.0 497.7,214.8 500.1,213.2 502.4,214.0 504.8,216.0 507.1,214.8 509.5,212.8 511.8,212.0 514.1,211.6 516.5,210.0 518.8,211.6 521.2,214.8 523.5,216.8 525.9,218.8 528.2,219.2 530.6,218.4 533.0,216.4 535.3,218.4 537.6,216.4 540.0,215.2" class="stroke-a" fill="none"/>
<line x1="380.0" y1="50.0" x2="410.0" y2="50.0" class="stroke-b"/>
<text x="416.0" y="54.0" class="fig-small" text-anchor="start">Q-learning</text>
<line x1="380.0" y1="70.0" x2="410.0" y2="70.0" class="stroke-a"/>
<text x="416.0" y="74.0" class="fig-small" text-anchor="start">double Q-learning</text>
<line x1="380.0" y1="90.0" x2="410.0" y2="90.0" class="stroke-muted" stroke-dasharray="4 4"/>
<text x="416.0" y="94.0" class="fig-small" text-anchor="start">0.05, the floor from exploring</text>
</svg>
Figure 3. Share of 100 runs that go left from A, smoothed over 5 episodes, with α = 0.1 and ε = 0.1. Notice that Q-learning starts far above the 0.05 floor and takes many episodes to fall, while double Q-learning is near it by about episode 80.
```

An $\varepsilon$-greedy agent that has learned the right answer still goes left with probability $\varepsilon / 2 = 0.05$ from exploration alone. In the run above, Q-learning goes left in 0.84 of runs over episodes 1 to 30 and 0.18 over episodes 181 to 200. Double Q-learning goes left in 0.38 and 0.07, close to the floor.

## Double Q-learning

Double Q-learning (van Hasselt, 2010) puts the double estimator inside Q-learning. It keeps two tables, $Q_1$ and $Q_2$. After each step it flips a fair coin. On heads, it updates $Q_1$:

$$
Q_1(s_t, a_t) \leftarrow Q_1(s_t, a_t) + \alpha \Bigl[r_t + \gamma\, Q_2\bigl(s_{t+1}, \operatorname*{arg\,max}_{a'} Q_1(s_{t+1}, a')\bigr) - Q_1(s_t, a_t)\Bigr]
$$

Read it aloud: $Q_1$ picks the best next action, $Q_2$ says what it is worth, and $Q_1(s_t, a_t)$ moves toward that target. On tails, swap the two tables. Each table learns from about half the steps, so the two sets of noise stay independent. The agent behaves $\varepsilon$-greedily on $Q_1 + Q_2$, which uses all the data.

```python
if rng.random() < 0.5:
    target = r + gamma * Q2[s2][np.argmax(Q1[s2])] * (not done)
    Q1[s][a] += alpha * (target - Q1[s][a])
else:
    target = r + gamma * Q1[s2][np.argmax(Q2[s2])] * (not done)
    Q2[s][a] += alpha * (target - Q2[s][a])
```

Here is a worked example with $r = -1$, $\gamma = 0.9$, and next-state values $Q_1 = (1, 3)$ and $Q_2 = (4, 0)$. When updating $Q_1$, $Q_1$ picks action 1, and $Q_2$ scores it at 0, so the target is $-1 + 0.9 \times 0 = -1$. When updating $Q_2$, $Q_2$ picks action 0, and $Q_1$ scores it at 1, so the target is $-1 + 0.9 \times 1 = -0.1$. Plain Q-learning would use $\max Q_1 = 3$ and get $1.7$.

> [!WARNING]
> Using $\max_{a'} Q_2(s_{t+1}, a')$ as the target is not double Q-learning. That lets $Q_2$ both choose and score, which is the single estimator again, with all its bias.

> [!KEY]
> Double Q-learning chooses the next action with one table and scores it with the other. The noise that made an action look best never also reports its value, so the agent in the example stops overrating the way left.

## Average over the next action with Expected SARSA

The noise in SARSA's target comes from two places: the environment's randomness, and the random choice of the next action $a_{t+1}$. Expected SARSA removes the second. Instead of the one next action you happened to sample, it averages over all next actions, weighted by the policy:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \Bigl[r_t + \gamma \sum_{a'} \pi(a' \mid s_{t+1})\, Q(s_{t+1}, a') - Q(s_t, a_t)\Bigr]
$$

Read it aloud: the target is the reward plus the value you expect in the next state if you pick the next action with your policy. For an $\varepsilon$-greedy policy, the weights are $\varepsilon / A$ on every action, plus $1 - \varepsilon$ split among the greedy ones. In numpy, the sum is a dot product:

```python
probs = epsilon / A + (1 - epsilon) * best / best.sum()   # best = q_next == q_next.max()
target = r + gamma * probs @ q_next * (not terminated)
```

Take $r = -1$, $\gamma = 1$, and $Q(s', \cdot) = (2, 5)$ with $\varepsilon = 0.2$. The probabilities are $(0.1, 0.9)$, so the target is $-1 + 0.1 \times 2 + 0.9 \times 5 = 3.7$. SARSA would give 1 or 4, depending on the sampled action. Expected SARSA lands between them every time. At $\varepsilon = 0$ the average is the max, and Expected SARSA is Q-learning.

> [!RECAP]
> - The max of unbiased noisy estimates averages above the max of their true values, and the gap grows with the number of actions and the noise.
> - The double estimator picks with one array and reads the other: `Y[np.argmax(X)]`.
> - The double Q target is `r + gamma * q_evaluate[np.argmax(q_select)]`, or `r` at the end.
> - Double Q-learning flips a coin per step to pick which table to update, and acts on `Q1[s] + Q2[s]`.
> - The Expected SARSA target is `r + gamma * probs @ q_next`, with `probs` the $\varepsilon$-greedy probabilities.

## Build it step by step

The checks test the functions in this order. `MaxBiasMDP` and `epsilon_greedy_action` are already written. In `MaxBiasMDP`, a table is a list of rows from `env.new_q()`, so write `Q[s][a]` rather than `Q[s, a]`.

1. **`single_estimate(means)`.** Takes `(..., A)` and returns `(...)`. One call to `np.max` with `axis=-1`.
2. **`double_estimate(means_a, means_b)`.** Same shapes. Take `np.argmax(means_a, axis=-1)` and gather from `means_b` with `np.take_along_axis`. The common mistake is taking the max of `means_b`, which is the single estimator again.
3. **`double_q_target(q_select, q_evaluate, r, terminated, gamma)`.** Returns a float. One `np.argmax`, one index. A terminal step's target is `r` alone.
4. **`q_learning(env, n_episodes, alpha, epsilon, seed)`.** Returns a boolean `(n_episodes,)` array, `True` where the episode went left from $A$. Record the action taken in `env.A` before you step.
5. **`double_q_learning(env, n_episodes, alpha, epsilon, seed)`.** Same output. Act on `Q1[s] + Q2[s]`, then flip the coin with `rng.random() < 0.5`.
6. **`epsilon_greedy_probs(q_row, epsilon)`.** Returns `(A,)` summing to one. Build a boolean mask of the ties and split $1 - \varepsilon$ over it. The common mistake is giving all of $1 - \varepsilon$ to the first tie, which no longer matches a policy that breaks ties at random.
7. **`expected_sarsa_target(q_next, r, terminated, gamma, epsilon)`.** Returns a float. Dot the probabilities with `q_next`.

## Where this breaks

- **Underestimation.** The double estimator can err low when true values differ, as the $(0, 0.5)$ example shows. A small downward bias usually does less harm than a compounding upward one, but it is not free.
- **Slower learning per table.** Each table in double Q-learning sees half the data. On problems with little noise, plain Q-learning can learn faster.
- **Bias with function approximation.** With a neural network, overestimation comes from approximation error as well as reward noise, and it can be large. [Double and dueling DQN](/lessons/027-double-and-dueling-dqn) reuses the online network to choose and the target network to score.
- **Expected SARSA in stochastic worlds.** Its advantage comes from removing action noise. When transitions or rewards are noisy, it still needs a small step size.

## Further reading

- [van Hasselt, "Double Q-learning" (2010)](https://papers.nips.cc/paper_files/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf). The algorithm and the bias analysis from this chapter.
- [van Hasselt, Guez, and Silver, "Deep reinforcement learning with double Q-learning" (2016)](https://arxiv.org/abs/1509.06461). The same idea for deep Q-networks.
- [Smith and Winkler, "The optimizer's curse" (2006)](https://doi.org/10.1287/mnsc.1050.0451). The same bias in decision analysis outside reinforcement learning.
- [van Seijen, van Hasselt, Whiteson, and Wiering, "A theoretical and empirical analysis of Expected Sarsa" (2009)](https://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/vanseijenadprl09.pdf). The analysis of the Expected SARSA target.
- [Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd edition (2018), Chapter 6](http://incompleteideas.net/book/the-book-2nd.html). Example 6.7, the maximization-bias MDP.

## The exercise

You write `single_estimate`, `double_estimate`, `double_q_target`, `q_learning`, `double_q_learning`, `epsilon_greedy_probs`, and `expected_sarsa_target`. The checks verify the two estimators by hand, and that the single estimate is biased up by the expected max of normals while the double estimate is not. They verify the double Q target on the worked example. Over 100 seeded runs, Q-learning must go left from $A$ far more often than double Q-learning early on, and double Q-learning must settle near the 0.05 floor. The last two checks confirm that the Expected SARSA target is 3.7 on the worked example, and that it reduces to Q-learning's target at $\varepsilon = 0$.
