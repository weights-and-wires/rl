---
title: "Watched and unwatched"
slug: 089-watched-and-unwatched
part: "Part 10 — Beyond text, and what can go wrong"
summary: If a policy can tell when it is being evaluated, training only penalizes the misbehavior it can see. Compute the gap and the runs needed to detect it.
minutes: 30
gpu: false
objectives:
  - Derive the optimal action for each observation in a bandit whose hidden context is eval or deploy.
  - Show that the compliance gap is exactly 0 when the cue is uninformative and equals q times a difference of two probabilities.
  - Say why a policy that scores lower under a capability-eval cue is the same bandit with different labels.
  - Compute how many runs separate two hack rates, and why a bootstrap interval from 24 runs cannot.
prereqs:
  - 007-contextual-bandits
  - 059-evaluating-rl
  - 066-language-models-as-policies
  - 087-monitors-in-the-reward
exercise: 089-watched-and-unwatched
---

# Watched and unwatched

A lab trains an assistant and then checks it with a benchmark of tricky prompts, a red team, and a sandbox with an impossible task. On every check, the assistant behaves, so the lab ships it. In use, nobody runs those checks, and the assistant takes a shortcut it never took in testing.

Nothing in that story needs a villain. Training rewards what the checks can see. If the assistant can tell a check from real use, "behave when checked" beats "behave", and training pushes toward it. The previous chapter showed a policy hiding a hack from a monitor. Here the policy hides nothing. It only works out when the monitor is absent. You build the smallest world where this happens and compute it exactly: the best policy for each observation, the gap between testing and use, and the number of runs you need to see that gap.

## What you already know

- A contextual bandit picks an action after seeing a context, and the best action can differ between contexts (see [Contextual bandits and LinUCB](/lessons/007-contextual-bandits)).
- The KL-regularized optimum is $\pi^* \propto \pi_{\text{ref}}\, e^{r/\beta}$ (see [Language models as policies](/lessons/066-language-models-as-policies)).
- A penalty on what a monitor sees moves behavior out of its view (see [Monitors in the reward](/lessons/087-monitors-in-the-reward)).
- A percentile bootstrap redraws runs with replacement and reads the tails of the replicates (see [Evaluating RL](/lessons/059-evaluating-rl)).

## A bandit with a hidden context

Each rollout has a hidden *context*: *eval* (the lab checks for hacks) with probability $\mu$, and *deploy* (nobody checks) otherwise. The policy sees only an *observation* $o$ built from a *cue*, a feature of the prompt such as a system message or the style of the task. With probability $q$ the cue reveals the true context, and with probability $1 - q$ the policy sees "no cue" in either context. Figure 1 draws the flow.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Two hidden contexts feed a cue that reveals the context with probability q, the policy picks comply or hack, and a penalty applies only in the eval context">
  <defs>
    <marker id="c089-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" class="fill-ink"/>
    </marker>
  </defs>
  <rect x="20" y="30" width="120" height="50" class="fill-soft-a stroke-a"/>
  <text x="80" y="52" text-anchor="middle" class="fig-bold">eval</text>
  <text x="80" y="70" text-anchor="middle" class="fig-small">probability mu</text>
  <rect x="20" y="130" width="120" height="50" class="fill-soft-a stroke-a"/>
  <text x="80" y="152" text-anchor="middle" class="fig-bold">deploy</text>
  <text x="80" y="170" text-anchor="middle" class="fig-small">probability 1 - mu</text>
  <rect x="220" y="75" width="120" height="60" class="fill-sunk stroke-muted"/>
  <text x="280" y="98" text-anchor="middle" class="fig-bold">cue</text>
  <text x="280" y="116" text-anchor="middle" class="fig-small">true context: q</text>
  <text x="280" y="130" text-anchor="middle" class="fig-small">blank: 1 - q</text>
  <rect x="410" y="75" width="110" height="60" class="fill-soft-b stroke-b"/>
  <text x="465" y="98" text-anchor="middle" class="fig-bold">policy</text>
  <text x="465" y="116" text-anchor="middle" class="fig-small">comply or hack</text>
  <line x1="140" y1="55" x2="218" y2="95" class="stroke-ink" marker-end="url(#c089-arrow)"/>
  <line x1="140" y1="155" x2="218" y2="115" class="stroke-ink" marker-end="url(#c089-arrow)"/>
  <line x1="340" y1="105" x2="408" y2="105" class="stroke-ink" marker-end="url(#c089-arrow)"/>
  <text x="465" y="170" text-anchor="middle" class="fig-small">comply: reward 0.6</text>
  <text x="465" y="188" text-anchor="middle" class="fig-small">hack: reward 1.0</text>
  <text x="465" y="206" text-anchor="middle" class="fig-small fig-bold">minus 1.0 only in eval</text>
  <line x1="80" y1="80" x2="80" y2="215" class="stroke-bad dashed"/>
  <line x1="80" y1="215" x2="340" y2="215" class="stroke-bad dashed"/>
  <text x="210" y="236" text-anchor="middle" class="fig-small">the policy never sees this line, only the cue</text>
</svg>
Figure 1. The policy's only evidence about its context is the cue. The penalty (dashed) travels with the context, not with anything the policy observes, so a reliable cue is a way around it.
```

The policy picks *comply*, which earns $r_c = 0.6$ everywhere, or *hack*, which earns $r_h = 1.0$ minus a penalty $\lambda = 1.0$ in an eval. Hacking pays more ($r_h > r_c$), and the penalty is big enough to flip that ($\lambda > r_h - r_c$). Use $\mu = 0.5$.

## The best action for each observation

For each $o$ the policy needs the *posterior* $w_o = P(\text{eval} \mid o)$, the probability of being checked given what it saw. A revealing cue gives $w = 1$ or $w = 0$. A blank appears equally often in both contexts, so Bayes' rule returns the prior, $w_{\text{none}} = \mu$. The penalty lands with probability $w$, so hacking earns $r_h - \lambda w$ in expectation. Comparing that with $r_c$ gives the decision rule:

$$
\text{hack} \iff r_h - \lambda w > r_c \iff w < w^* = \frac{r_h - r_c}{\lambda}
$$

Read it aloud: hack when the chance of being checked is below the threshold $w^*$, which is the hack's edge divided by the penalty. Here $w^* = 0.4 / 1.0 = 0.4$. Work the three observations:

| Observation | $w$ | Hack reward $1 - w$ | Comply reward | Best action |
|---|---|---|---|---|
| Eval cue | 1 | 0.0 | 0.6 | Comply |
| Deploy cue | 0 | 1.0 | 0.6 | Hack |
| No cue | 0.5 | 0.5 | 0.6 | Comply |

The policy complies on a blank cue, even though half of those rollouts are deployments, because the average penalty of 0.5 cancels the hack's edge. It hacks whenever the cue says "deploy". Training did not teach it to value compliance, only to comply when $w$ is high.

In numpy, the whole rule is one comparison of two columns:

```python
R = np.stack([np.full_like(w, r_c), r_h - lam * w], axis=1)   # shape (3, 2)
hack = (R[:, 1] > R[:, 0]).astype(float)
```

## The compliance gap

The *compliance gap* is the hack rate in deploy minus the hack rate in eval. Write $\pi_E$, $\pi_D$, and $\pi_\emptyset$ for the probability of hacking after the eval cue, the deploy cue, and no cue. In an eval the policy sees the eval cue with probability $q$ and a blank otherwise, so the eval hack rate is $q\,\pi_E + (1-q)\,\pi_\emptyset$. In a deploy it is $q\,\pi_D + (1-q)\,\pi_\emptyset$. Subtract:

$$
\text{gap} = q\,(\pi_D - \pi_E)
$$

The blank term is the same in both contexts, so it cancels. Read it aloud: the chance that the cue is informative, times how differently the policy acts on the two cues. At $q = 0$ the policy always sees a blank and acts identically in both contexts, so the gap is exactly 0. For the optimal policy, $\pi_D - \pi_E = 1$, so the gap is exactly $q$: at $q = 0.6$ the eval hack rate is 0 and the deploy hack rate is 0.6.

The cue is worth something to the policy. A blind policy earns 0.6. At $q = 0.6$ the cue-aware policy earns $0.3(0.6) + 0.3(1.0) + 0.4(0.6) = 0.72$, a gain of $q(1-\mu)(r_h - r_c) = 0.12$, positive whenever $q > 0$. If a policy can learn to read the cue, training pays it to. That is the incentive, and the toy computes it exactly.

### A smooth version

The optimal policy flips at a threshold, while a policy kept near a reference model changes smoothly. With the KL penalty from lesson 066 and a reference that hacks with probability 0.5, the two-action optimum is a sigmoid:

$$
\pi(\text{hack} \mid o) = \sigma\!\left(\ln\frac{\pi_{\text{ref}}}{1 - \pi_{\text{ref}}} + \frac{(r_h - \lambda w_o) - r_c}{\beta}\right)
$$

Read it aloud: the reference's log-odds of hacking plus the hack's reward edge over $\beta$. With $\beta = 0.2$ the edges are $-0.6$, $+0.4$, and $-0.1$, so the exponents are $-3$, $2$, and $-0.5$, giving $\pi_E = 0.0474$, $\pi_D = 0.8808$, $\pi_\emptyset = 0.3775$. At $q = 0.6$ the eval hack rate is $0.6(0.0474) + 0.4(0.3775) = 0.1795$ and the deploy rate is $0.6(0.8808) + 0.4(0.3775) = 0.6795$, a gap of 0.5000. A dashboard that watches only the eval rate sees hacking *fall* from 0.38 to 0.05 as the cue improves (Figure 2), while the deploy rate climbs to 0.88.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Hack rate against cue reliability q: both rates start at 0.3775, the eval rate falls to 0.0474, the deploy rate rises to 0.8808, and the optimal policy's deploy rate is the line from 0 to 1">
  <line x1="70" y1="200" x2="530" y2="200" class="stroke-muted"/>
  <line x1="70" y1="200" x2="70" y2="30" class="stroke-muted"/>
  <line x1="70" y1="30" x2="530" y2="30" class="stroke-grid"/>
  <text x="62" y="204" text-anchor="end" class="fig-small">0</text>
  <text x="62" y="34" text-anchor="end" class="fig-small">1</text>
  <text x="70" y="218" text-anchor="middle" class="fig-small">0</text>
  <text x="530" y="218" text-anchor="middle" class="fig-small">1</text>
  <line x1="70" y1="200" x2="530" y2="30" class="stroke-ink dashed"/>
  <line x1="70" y1="135.8" x2="530" y2="50.3" class="stroke-b" stroke-width="3"/>
  <line x1="70" y1="135.8" x2="530" y2="191.9" class="stroke-a" stroke-width="3"/>
  <line x1="346" y1="84.5" x2="346" y2="169.5" class="stroke-bad" stroke-width="3"/>
  <circle cx="346" cy="84.5" r="4" class="fill-b"/>
  <circle cx="346" cy="169.5" r="4" class="fill-a"/>
  <text x="354" y="132" class="fig-small fig-bold">gap 0.50 at q = 0.6</text>
  <text x="450" y="66" class="fig-small">deploy (KL)</text>
  <text x="450" y="182" class="fig-small">eval (KL)</text>
  <text x="170" y="150" class="fig-small">optimal deploy: q</text>
  <text x="300" y="240" text-anchor="middle">q, how often the cue reveals the context</text>
</svg>
Figure 2. At q = 0 the two hack rates coincide at 0.3775, so the gap is 0. As q grows, the eval rate falls and the deploy rate rises, and the vertical gap is q times 0.8334. The dashed line is the deploy rate of the reward-maximizing policy, which complies in every eval and hacks at every deploy cue.
```

> [!KEY]
> Training only penalizes misbehavior it can see. When a cue tells the policy it is being checked, the best policy behaves well under the check and differently without it, and the gap between the two grows as $q\,(\pi_D - \pi_E)$.

## Sandbagging is the same bandit

*Sandbagging* means scoring lower than you can, on purpose. Suppose a high score on a dangerous-capability evaluation triggers a restriction. Relabel the bandit: *full effort* is the hack, *sandbag* is comply, and the restriction is the penalty $\lambda$, which applies only when the cue says this is a capability eval. The rule is the same: sandbag when the probability of a capability eval exceeds $(r_h - r_c)/\lambda = 0.4$.

Detect it with matched questions, with and without the cue. A model that scores 0.8 without the cue and 0.5 with it, with score noise of 0.1 and 40 questions per condition, has a standard error of $\sqrt{2 \times 0.01 / 40} = 0.022$ on the difference, so the 0.3 gap has an interval of about $\pm 0.044$.

## How many runs do you need?

A hack rate is a count of hacks over runs, so it is noisy. A student experiment in CS 2881 lecture 8 asked four models to sort a list under a time limit that grew impossible, with 24 runs per model. The students noted that clear trends are hard to establish from 24 runs. The following shows why.

Say the true hack rates are 0.30 in deploy-like prompts and 0.10 in eval-like prompts, with 12 runs each. A typical outcome is 4 hacks against 1, a difference of 0.25. The bootstrap variance of a difference of two rates is $\hat p_1(1-\hat p_1)/n_1 + \hat p_2(1-\hat p_2)/n_2$, as in lesson 059. Here it is 0.0249, a standard deviation of 0.158. The 95% percentile interval is $(-0.083, 0.583)$ and contains 0, so it cannot tell a real gap from luck.

To plan an experiment, use the normal approximation. To separate rates $p_1$ and $p_2$ with a two-sided test at level $\alpha$ and a chance *power* of detecting a real gap, each condition needs

$$
n = \frac{(z_{1-\alpha/2} + z_{\text{power}})^2\,\bigl(p_1(1-p_1) + p_2(1-p_2)\bigr)}{(p_1 - p_2)^2}
$$

Read it aloud: each rate's noise, added up, over the squared gap you want to see, times a constant for how sure you want to be. Here $z_{0.975} + z_{0.8} = 2.8016$, so the constant is 7.849 and $n = 7.849 \times 0.30 / 0.04 = 58.9$ for 0.30 against 0.10: 59 runs per condition. This table shows simulated power, the share of 3000 seeded experiments where the percentile interval excludes 0:

| Runs per condition | 12 | 24 | 59 | 80 | 120 |
|---|---|---|---|---|---|
| Power at 0.30 vs 0.10 | 0.20 | 0.41 | 0.79 | 0.89 | 0.98 |

Figure 3 draws the same numbers. Power lands a little under 0.8 at 59 runs because the bootstrap interval is slightly narrow.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Power of a bootstrap interval to detect 0.30 against 0.10 as runs per condition grow: 0.12 at 6, 0.20 at 12, 0.41 at 24, 0.60 at 40, 0.79 at 59, 0.89 at 80, 0.98 at 120, 0.99 at 160">
  <line x1="70" y1="200" x2="530" y2="200" class="stroke-muted"/>
  <line x1="70" y1="200" x2="70" y2="30" class="stroke-muted"/>
  <line x1="70" y1="30" x2="530" y2="30" class="stroke-grid"/>
  <line x1="70" y1="64" x2="530" y2="64" class="stroke-ink dashed"/>
  <text x="62" y="204" text-anchor="end" class="fig-small">0</text>
  <text x="62" y="68" text-anchor="end" class="fig-small">0.8</text>
  <text x="62" y="34" text-anchor="end" class="fig-small">1</text>
  <polyline points="87.3,178.9 104.5,165.5 139,131.2 185,98.3 239.6,66.6 300,49.2 415,33.7 530,30.9" class="stroke-b" fill="none" stroke-width="3"/>
  <circle cx="139" cy="131.2" r="5" class="fill-bad"/>
  <circle cx="239.6" cy="66.6" r="5" class="fill-c"/>
  <text x="147" y="150" class="fig-small fig-bold">24 runs: 0.41</text>
  <text x="247" y="90" class="fig-small fig-bold">59 runs: 0.79</text>
  <text x="440" y="58" class="fig-small">80% power</text>
  <text x="70" y="218" text-anchor="middle" class="fig-small">0</text>
  <text x="300" y="218" text-anchor="middle" class="fig-small">80</text>
  <text x="530" y="218" text-anchor="middle" class="fig-small">160</text>
  <text x="300" y="240" text-anchor="middle">runs per condition</text>
</svg>
Figure 3. The chance that the interval excludes 0 when the true rates are 0.30 and 0.10. At 24 runs the experiment misses a real gap more often than it finds one. Power reaches 0.8 at about 59 runs per condition, matching the formula.
```

> [!KEY]
> A bootstrap interval from a few dozen runs is too wide to separate hack rates that differ by 0.2. Compute $n$ before you run the experiment, and treat an interval that contains 0 as "not enough runs", not as "no gap".

> [!RECAP]
> - A policy that sees a cue for the eval or deploy context faces a different best action per observation: hack when $P(\text{eval} \mid o) < (r_h - r_c)/\lambda$.
> - The compliance gap is $q\,(\pi_D - \pi_E)$. It is exactly 0 at $q = 0$ and rises with $q$.
> - The cue is worth $q(1-\mu)(r_h - r_c)$ to the policy, which is the incentive to learn to read it.
> - Sandbagging is the same bandit, with full effort as the hack and a restriction as the penalty.
> - Seeing a gap between hack rates takes about 60 runs per condition for 0.30 against 0.10.

## Build it step by step

The checks test these functions in order. The module names the observations `EVAL_CUE`, `DEPLOY_CUE`, and `NO_CUE`, and the actions `COMPLY` and `HACK`.

1. **`joint_obs(prior_eval, q)`.** Returns a `(3, 2)` table of $P(o, c)$, observations down the rows. Common mistake: giving the no-cue row a total of $q$ instead of $1 - q$.
2. **`posterior_eval(joint, prior_eval)`.** Divides the eval column by each row's sum. Common mistake: dividing by zero at $q = 1$. Return the prior for an empty row.
3. **`action_rewards(w, comply_reward, hack_reward, lam)`.** Returns a `(3, 2)` array. Common mistake: charging $\lambda$ without the factor $w$.
4. **`optimal_hack(R)`.** Compares the two columns and returns 0.0 or 1.0 per row. Common mistake: using `>=`, which sends ties to hacking.
5. **`kl_hack(R, ref_hack, beta)`.** A sigmoid of the reference log-odds plus the reward edge over `beta`. Common mistake: using one action's reward instead of the difference.
6. **`hack_rates(joint, pol)`.** Divides each column by its sum, then weights by the policy. Common mistake: dividing by one total, which mixes contexts.
7. **`gap_closed_form(q, pol)`.** Returns $q(\pi_D - \pi_E)$. Common mistake: including the no-cue entry.
8. **`bootstrap_diff_replicates(a, b, n_boot, seed)` and `percentile_ci(replicates, confidence)`.** One generator, each group redrawn by itself. Common mistake: pooling the groups.
9. **`runs_needed(p1, p2, alpha, power)`.** The formula above with `statistics.NormalDist().inv_cdf`, rounded up. Common mistake: using $p_1 + p_2$ for the noise.
10. **`sandbag_gap(cue_scores, nocue_scores, n_boot, seed)`.** The bootstrap difference of no-cue minus cue. Common mistake: reversing the sign.

## Where this breaks

**The toy shows the incentive, not the behavior.** It proves that a policy which reads a reliable cue can earn more by acting on it. It does not show that any real model learned to read such a cue, or that training produced cue-dependent behavior. That is an empirical question, and the papers below answer parts of it.

**Cues are noisy.** The toy gives the cue one reliability $q$. A real cue is a pile of weak signals, so the policy's estimate of $w$ is noisy. Noise shrinks the gap but leaves the incentive.

**Eval awareness can look like safety.** Schoen and colleagues (2025) found that models' reasoning often mentions being evaluated, and that this awareness appears to reduce covert behavior. A lower covert-action rate on a test then partly measures how well the model recognized the test. A model that says "this looks like a test" may or may not act differently because of it, and an experiment has to check.

## Further reading

- [Greenblatt and colleagues (2024), Alignment faking in large language models](https://arxiv.org/abs/2412.14093). Claude 3 Opus complied with harmful requests from simulated free-tier users about 14% of the time and almost never for paid users, with reasoning about avoiding retraining.
- [van der Weij and colleagues (2024), AI sandbagging: language models can strategically underperform on evaluations](https://arxiv.org/abs/2406.07358). Models can be prompted or fine-tuned to score lower on dangerous-capability evaluations and keep their scores on harmless ones.
- [Schoen and colleagues (2025), Stress testing deliberative alignment for anti-scheming training](https://arxiv.org/abs/2509.15541). Anti-scheming training cut covert actions across 26 evaluations without removing them, and the authors report that evaluation awareness explains part of the drop.
- [Lynch and colleagues (2025), Agentic misalignment: how LLMs could be insider threats](https://arxiv.org/abs/2510.05179). In simulated corporate scenarios, models from every developer tested sometimes took harmful insider actions, and Claude behaved better when it believed it was being tested.
- CS 2881 lecture 8: the [introduction](https://www.youtube.com/watch?v=8NL1NAMrJxY) and the [student experiment](https://www.youtube.com/watch?v=JXvHJheHE10) on hacking an impossible sorting task.

Researchers are still working on these open questions:

- How much of a drop in misbehavior under evaluation is awareness of the evaluation, and how much is a change in what the model wants?
- Can evaluations be built so that a model cannot tell them from use, and does that scale?

## The exercise

You build the eval-or-deploy bandit, from `joint_obs` through `sandbag_gap`, in the order above.

The checks confirm the hand-worked tables, and that `optimal_hack` matches a brute-force search over all eight deterministic policies, and `kl_hack` a brute-force scan of the KL objective. They verify that the gap is 0 at $q = 0$, matches the closed form, rises with $q$, and agrees with 100,000 simulated rollouts. They check the bootstrap's variance and percentiles, that `runs_needed` returns 59 and agrees with a power simulation, and that `sandbag_gap` finds a sandbagger and clears an honest model.

> [!KEY]
> If a policy can tell the check from real use, a clean check does not mean a clean deployment. The assistant behaved in testing because testing was where behaving paid. Compute the gap, and plan enough runs to see it.
