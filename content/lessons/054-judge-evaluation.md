---
title: Evaluate a reward model and a judge
slug: 054-judge-evaluation
part: "Part 9 — RL for language models"
summary: Measure whether a reward model or a judge agrees with people, or only with answer order and length.
minutes: 40
gpu: false
objectives:
  - Compute pairwise accuracy, agreement with human labels, and expected calibration error.
  - Measure position bias by swapping the order of the two answers.
  - Fit a length-controlled win rate and Bradley-Terry strengths, counting a tie as half a win.
  - Explain how flipped labels shrink the reward gap a preference loss fits.
prereqs:
  - 053-reward-models
exercise: 054-judge-evaluation
---

# Evaluate a reward model and a judge

You already fit a reward model. It turns a person's choice between two answers into a score, and a judge does the same job in one prompt. Either number can look decisive and still measure the wrong thing. Mira writes four tokens more than Noor, the judge picks Mira, and a leaderboard reports a win rate near 80 percent. Swap the order and the judge picks the left-hand slot again.

This chapter measures a score you already have. [Reward models from preferences](/lessons/053-reward-models) trains the Bradley-Terry loss, and you do not fit that loss again. You ask whether the score agrees with people, whether a stated probability is honest, and whether order or length is doing the work.

## What you already know

- A Bradley-Terry model says the chance one answer is preferred is the sigmoid of a reward gap. Only gaps are identified, up to one constant per prompt.
- The sigmoid \(\sigma(z) = 1/(1+e^{-z})\) turns a gap into a probability. The logit, \(\log(p/(1-p))\), turns a probability back into a gap.
- Logistic regression models a probability as the sigmoid of a weighted sum of features.
- Accuracy is the fraction of predictions that match a label.

## Rank the pair, then check the probability

Pairwise accuracy asks one yes-or-no question: did the higher score land on the answer the person preferred? Agreement asks the same question and also counts a tie, where the person could not choose and the two scores are equal. Calibration asks a different question. When the model states 0.85, does the event happen 85 percent of the time?

Four prompts, with a label of \(+1\) when the person prefers Mira, \(-1\) when they prefer Noor, and \(0\) for a human tie:

| Prompt | Mira | Noor | Label | Scores |
|---|---|---|---|---|
| 1 | 0.2 | 0.0 | \(+1\) | Mira, correct |
| 2 | 1.0 | 0.4 | \(-1\) | Mira, wrong |
| 3 | 0.0 | 0.0 | \(0\) | Tie, and the label is a tie |
| 4 | 0.5 | 0.8 | \(+1\) | Noor, wrong |

Prompt 3 has no winner, so it leaves the accuracy. Of the other three, only prompt 1 is ranked correctly, and the accuracy is \(1/3\). Agreement keeps the tie: prompts 1 and 3 match, prompts 2 and 4 do not, and the agreement is \(1/2\). A score tie on a decisive label is a miss in both numbers. It is not half a point. The higher score has to be strictly higher.

A probability can be wrong even when you have not asked about accuracy. On four prompts Mira is on the left and the judge states 0.85. On four prompts Mira is on the right and the judge states 0.15. The stated probability follows the slot. Mira wins two prompts in each group. Figure 1 puts both groups on the line at 0.5, off the diagonal a calibrated judge would land on.

```figure
<svg viewBox="0 0 620 300" role="img" aria-label="Reliability diagram. The judge predicts 0.15 and 0.85, but Mira wins half the time in both groups.">
<line x1="90" y1="210.0" x2="560" y2="210.0" class="stroke-grid"/>
<text x="82" y="214.0" class="fig-small" text-anchor="end">0</text>
<line x1="90" y1="166.5" x2="560" y2="166.5" class="stroke-grid"/>
<text x="82" y="170.5" class="fig-small" text-anchor="end">0.25</text>
<line x1="90" y1="123.0" x2="560" y2="123.0" class="stroke-grid"/>
<text x="82" y="127.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="90" y1="79.5" x2="560" y2="79.5" class="stroke-grid"/>
<text x="82" y="83.5" class="fig-small" text-anchor="end">0.75</text>
<line x1="90" y1="36.0" x2="560" y2="36.0" class="stroke-grid"/>
<text x="82" y="40.0" class="fig-small" text-anchor="end">1</text>
<line x1="90.0" y1="210" x2="90.0" y2="216" class="stroke-muted"/>
<text x="90.0" y="232" class="fig-small" text-anchor="middle">0</text>
<line x1="325.0" y1="210" x2="325.0" y2="216" class="stroke-muted"/>
<text x="325.0" y="232" class="fig-small" text-anchor="middle">0.5</text>
<line x1="560.0" y1="210" x2="560.0" y2="216" class="stroke-muted"/>
<text x="560.0" y="232" class="fig-small" text-anchor="middle">1</text>
<line x1="90.0" y1="210.0" x2="560.0" y2="36.0" class="stroke-muted dashed"/>
<line x1="160.5" y1="183.9" x2="160.5" y2="123.0" class="stroke-bad"/>
<circle cx="160.5" cy="183.9" r="4" class="fill-a"/>
<circle cx="160.5" cy="123.0" r="5" class="fill-b"/>
<line x1="489.5" y1="62.1" x2="489.5" y2="123.0" class="stroke-bad"/>
<circle cx="489.5" cy="62.1" r="4" class="fill-a"/>
<circle cx="489.5" cy="123.0" r="5" class="fill-b"/>
<text x="172.5" y="111.0" class="fig-small fig-bold">0.15 predicted, 0.50 won</text>
<text x="477.5" y="145.0" class="fig-small fig-bold" text-anchor="end">0.85 predicted, 0.50 won</text>
<text x="325" y="258" class="fig-small" text-anchor="middle">stated probability that Mira wins</text>
<text x="90" y="22" class="fig-small" text-anchor="start">how often Mira won</text>
<line x1="90" y1="276" x2="112" y2="276" class="stroke-muted dashed"/>
<text x="118" y="280" class="fig-small">calibrated</text>
<circle cx="210" cy="276" r="4" class="fill-b"/>
<text x="220" y="280" class="fig-small">observed</text>
<circle cx="300" cy="276" r="4" class="fill-a"/>
<text x="310" y="280" class="fig-small">stated probability</text>
</svg>
Figure 1. The judge's stated probability follows the left-hand slot. Both groups have Mira winning half the time, 0.35 away from the probability the judge stated.
```

Five equal-width bins cover \([0, 1]\). Bin \(b\) is \([b/5, (b+1)/5)\), and the last bin includes 1 so a probability of 1 is not dropped. The 0.15 bin holds four prompts, mean confidence 0.15, frequency 0.50. The 0.85 bin holds the other four, mean confidence 0.85, same frequency. Expected calibration error weights each gap by the bin's share of the prompts:

$$
\mathrm{ECE} = \sum_b \frac{n_b}{n} \left| \mathrm{acc}_b - \mathrm{conf}_b \right|
$$

Read it aloud: weight each absolute gap between observed frequency and mean stated probability by that bin's share of the prompts. Empty bins add nothing. Here the sum is \(0.5 \times 0.35 + 0.5 \times 0.35 = 0.35\). Using bin middles, 0.10 and 0.90, scores 0.40 on the same outcomes.

> [!WARNING]
> One bin hides this judge. The eight probabilities average to 0.50, and Mira wins half the time, so a single bin reports an error of 0. Use enough bins that 0.15 and 0.85 cannot cancel.

The three numbers in code, with `label` holding \(+1\), \(-1\), or \(0\):

```python
decisive = label != 0
accuracy = np.mean(np.sign(score_a - score_b)[decisive] == label[decisive])
```

Agreement keeps every row. A label of 0 matches equal scores, and any other label matches only a strict sign. The calibration loop adds `(count / n) * abs(acc - conf)` and skips an empty bin.

> [!KEY]
> Pairwise accuracy ignores human ties and gives a score tie no credit. Calibration error is the weighted gap on the reliability diagram.

## Swap the two answers

Trusting the named winner fails when the winner is the slot. Zheng and colleagues (2023) show each pair twice, in opposite orders, and call the judge consistent when the same answer wins both times. Most judges they tried favored the first position, and on similar answers GPT-4 was consistent on a bit over 60 percent of pairs.

`swap_consistency` returns the fraction of pairs where `judge(right, left) == -judge(left, right)`. Zero equals its own negation, so a tie in both orders counts as consistent. On two pairs the judge always returns \(+1\), and the swap does not flip. On a third the scores are 5 and 2, so the calls return \(+1\) then \(-1\). On a fourth the scores are 2 and 2, and both calls return 0. Consistency is \(2/4 = 0.5\), and the position-bias rate is \(1 - 0.5 = 0.5\). A judge that always returns \(+1\) has consistency 0. Calling it twice in the same order, and treating agreement as consistency, scores that judge at 1.

> [!TIP]
> Keep a winner only when both orders name the same answer. When they disagree, record a tie.

> [!KEY]
> Swap consistency is the fraction of pairs whose preference flips when the order flips. One minus that fraction is the position-bias rate when the other pairs follow the true scores.

## Take length out of the win rate

Length is the other confounder. AlpacaEval reports how often a judge prefers a candidate to a baseline. Dubois, Galambosi, Liang, and Hashimoto (2024) fit a logistic regression on the model, the length gap, and the instruction, then predict again with the length term removed. The question is what the win rate would be at equal length. That control raised the rank correlation with Chatbot Arena from 0.94 to 0.98. The exercise keeps the model term and the length term:

$$
P(\text{Mira wins}) = \sigma(b_0 + b_1 \cdot \Delta \ell)
$$

Read it aloud: \(b_0\) is the log-odds at a gap of zero, \(b_1\) is the change per extra token, and the controlled win rate is \(\sigma(b_0)\). Run Newton from zeros for 20 steps, with ridge \(10^{-6}\) on \(b_1\) only, and clip \(p(1-p)\) at \(10^{-12}\).

A length-only judge wins with probability \(\sigma(1.5 \cdot \Delta \ell)\). On gaps \(+4, +4, +4, +4, -2\) that is \(\sigma(6) = 0.997527\) four times and \(\sigma(-3) = 0.047426\) once, mean 0.8075. The fit returns 0.500, and Figure 2 shows both rates. The ridge shifts the controlled rate by about \(10^{-6}\). The check allows \(10^{-5}\).

```figure
<svg viewBox="0 0 560 250" role="img" aria-label="Raw win rate near 0.81 beside the length-controlled win rate of 0.50.">
<line x1="70" y1="50" x2="520" y2="50" class="stroke-grid"/>
<text x="62" y="54" class="fig-small" text-anchor="end">1</text>
<line x1="70" y1="120.0" x2="520" y2="120.0" class="stroke-muted dashed"/>
<text x="62" y="124.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="70" y1="190" x2="520" y2="190" class="stroke-muted"/>
<text x="62" y="194" class="fig-small" text-anchor="end">0</text>
<rect x="150" y="76.9" width="80" height="113.1" class="fill-b"/>
<rect x="330" y="120.0" width="80" height="70.0" class="fill-c"/>
<text x="190" y="68.9" class="fig-small fig-bold" text-anchor="middle">0.808</text>
<text x="370" y="112.0" class="fig-small fig-bold" text-anchor="middle">0.500</text>
<text x="190" y="214" class="fig-small" text-anchor="middle">raw win rate</text>
<text x="370" y="214" class="fig-small" text-anchor="middle">equal length</text>
<text x="295" y="240" class="fig-small" text-anchor="middle">chance the judge prefers Mira</text>
</svg>
Figure 2. The length-only judge prefers Mira on 80.8 percent of these prompts, because Mira is longer on four of the five. At a length gap of zero the same fit says 50 percent.
```

When every gap is 0 and Mira wins 3 of 4 prompts, nothing is left for \(b_1\) to explain, and \(\sigma(b_0) = 0.75\). Returning 0.5 fails that case. Returning the raw mean fails the length-only case.

## Strengths from wins, including ties

A rating puts one number on each player, so a table of wins becomes a ranking. Player \(i\) has strength \(s_i\) and weight \(\pi_i = e^{s_i}\), and the chance \(i\) beats \(j\) is \(\pi_i / (\pi_i + \pi_j)\). That is the same sigmoid as the reward model, comparing players rather than answers. You fit these strengths to wins you already counted. You do not train \(\phi\) again.

A tie is half a win for each player. Record a comparison as \((i, j, r)\), the share credited to \(i\): 1, 0, or 0.5. Three wins for Mira (player 0), one win for Noor (player 1), and one tie give totals 3.5 and 1.5.

With two players the odds equal the win ratio. Centering splits the log of that ratio in half:

$$
s_0 - s_1 = \log\frac{3.5}{1.5} = \log\frac{7}{3},
\qquad
s_0 = \frac{1}{2}\log\frac{7}{3} = 0.4236,
\qquad
s_1 = -0.4236
$$

Read it aloud: centering splits the log win-ratio in half, so Mira sits at \(+0.4236\) and Noor at \(-0.4236\). The Elo scale uses base 10 and 400 points, and the same gap is \((400 / \ln 10) \times \log(7/3) = 147.19\) points. Counting the tie as a full win for the player listed first changes the totals to 4 and 1, and the centered strength to \(\tfrac{1}{2}\log 4 = 0.6931\).

For three or more players, start from \(\pi_i = 1\) and repeat minorization-maximization. \(W_i\) is the win total, ties included as halves, and \(N_{ij}\) is the number of comparisons between \(i\) and \(j\):

$$
\pi_i \leftarrow \frac{W_i}{\sum_{j \neq i} N_{ij} / (\pi_i + \pi_j)}
$$

Then set \(s_i = \log \pi_i\) and subtract the mean. One step from equal weights hits the two-player closed form, because the two denominators match. For three players, set strengths \(1\), \(0\), and \(-1\) and pass one comparison per pair at the Bradley-Terry probability, \(\sigma(1) = 0.731059\), \(\sigma(2) = 0.880797\), and \(\sigma(1)\). Those are the expected wins, the log-likelihood gradient at the true strengths is 0, and 32 iterations leave an error below \(10^{-7}\). The check allows \(10^{-5}\).

> [!KEY]
> A tie adds half a win to each player. Two players then have a closed form, half the log of the win ratio, and the same update recovers known strengths for three players once you center them.

## What a flipped label does to the loss

Some recorded labels point the wrong way. Suppose the true gap is \(\Delta\) and each label flips independently with probability \(\varepsilon\). The loss in the previous chapter sees the recorded winner. The probability that this winner is Mira shrinks toward one half:

$$
p_{\text{noisy}} = (1 - 2\varepsilon)\,\sigma(\Delta) + \varepsilon
$$

The gap that minimizes the same loss is the logit of \(p_{\text{noisy}}\), not \(\Delta\). For \(\Delta = 2\), \(\sigma(2) = 0.880797\) and \(\varepsilon = 0.1\), so \(p_{\text{noisy}} = 0.804638\) and the fitted gap is 1.416. At \(\varepsilon = 0.5\) every true gap looks like a coin flip, and the fitted gap is 0.

[Direct preference optimization](/lessons/056-direct-preference-optimization) trains on these same recorded pairs, so the same shrinkage sits in that loss. This chapter does not derive it.

> [!RECAP]
> - Accuracy on the four prompts is \(1/3\). Agreement, which keeps the tie, is \(1/2\). The eight-prompt calibration error is 0.35.
> - Swap consistency is the fraction of pairs that flip. One minus that fraction is a planted position-bias rate.
> - A length-only judge with raw win rate 0.8075 scores 0.500 at equal length.
> - A tie is half a win. Totals 3.5 and 1.5 have centered strengths \(\pm 0.4236\).

## Build it step by step

The checks call the functions in this order. Scores and labels are one-dimensional numpy arrays. A judge is a function you call, not a model you train.

1. **`pairwise_accuracy(score_a, score_b, label)`.** `label` is \(+1\), \(-1\), or \(0\). Drop the zeros. Return the fraction of the rest whose `sign(score_a - score_b)` equals the label. The common mistake is giving a score tie half credit.
2. **`expected_calibration_error(prob, outcome, n_bins)`.** `prob` is in \([0, 1]\) and `outcome` is 0 or 1. Equal-width bins, last bin closed at 1. Skip empty bins. Confidence is the mean of `prob` in the bin. The common mistake is using the bin center, which scores the figure's judge at 0.40 instead of 0.35.
3. **`human_agreement(score_a, score_b, label)`.** Same arrays. A row agrees when the label is 0 and the scores are equal, or the label's sign matches a strict score gap. The common mistake is dropping ties, which turns the four-prompt agreement of \(1/2\) into the accuracy of \(1/3\).
4. **`swap_consistency(judge, pairs)`.** `pairs` is a sequence of `(left, right)`. `judge(left, right)` returns \(+1\), \(-1\), or \(0\). Return the fraction of pairs with `judge(right, left) == -judge(left, right)`. The common mistake is calling `judge(left, right)` twice and treating agreement as consistency.
5. **`bt_fit(wins, iters=32)`.** `wins` is a sequence of `(i, j, result)` with `result` in \([0, 1]\). Add `result` to \(i\) and `1 - result` to \(j\), run the multiplicative update, and return log-strengths centered to mean 0. The common mistake is counting a result of 0.5 as a full win for \(i\).
6. **`length_controlled_winrate(length_diff, win)`.** `win` is 0, 1, or a probability in between. Fit \(b_0 + b_1 \cdot \Delta\ell\) for 20 Newton steps from zeros, with ridge \(10^{-6}\) on \(b_1\) only. Return \(\sigma(b_0)\). The common mistake is returning `win.mean()`.

## Where this breaks

**A consistent judge can still be unfair.** A swap catches a preference for the left-hand slot. A preference for a name or a format stays with the answer, so both orders still agree.

**Length carries real quality some of the time.** If the longer answer is longer because it contains the proof, removing the length term removes that signal along with the bias.

**Players who never meet do not share a constant.** Each group of players with no comparisons between groups gets its own additive constant, the same limit a reward model has across prompts.

**A one-sided labeler is not a random flip.** The noise formula assumes each label flips at random. A labeler who always prefers the longer answer is a bias, and a smaller fitted gap does not remove it.

## Further reading

- [Lambert, *Preference data*](https://rlhfbook.com/c/11-preference-data). How pairwise labels are collected, and why disagreement can be noise or a real split.
- [Lambert, *Reward modeling*](https://rlhfbook.com/c/05-reward-models). A held-out accuracy check for a Bradley-Terry reward model, and a language model used as a judge.
- [Lambert, *Evaluation*](https://rlhfbook.com/c/16-evaluation). Agreement between people and reward models, and Elo ratings from pairwise judgments.
- [CS336, lecture 12, Evaluation](https://www.youtube.com/watch?v=JpAxdTWQJxM). Elo from pairwise judgments, and AlpacaEval's length bias. The [course schedule](https://cs336.stanford.edu/) numbers this lecture 12.
- [CS224R, reward learning and preference optimization](https://cs224r.stanford.edu/). Reward learning fits a reward from rankings, and the preference-optimization lecture uses the same pairs.
- [Lambert and colleagues (2024), RewardBench](https://arxiv.org/abs/2403.13787). Accuracy is the fraction of triples where the chosen completion outscores the rejected one.
- [Dubois, Galambosi, Liang, and Hashimoto (2024), Length-controlled AlpacaEval](https://arxiv.org/abs/2404.04475). Predicts the win rate again after removing the length term.
- [Zheng and colleagues (2023), Judging LLM-as-a-judge](https://arxiv.org/abs/2306.05685). Defines swap consistency and reports a first-position bias.

Open questions researchers are working on:

- When a longer answer is also a better answer, how do you split the length coefficient into quality and judge bias?
- Which property, beyond pairwise accuracy on one benchmark, predicts that training on the score still agrees with people?
- When two annotators disagree, the open choice is a tie, a soft label, or dropping the pair. Which one keeps the fitted gap honest?

## The exercise

You write `pairwise_accuracy`, `human_agreement`, `expected_calibration_error`, `swap_consistency`, `bt_fit`, and `length_controlled_winrate`. The checks compare them with this chapter: accuracy \(1/3\), agreement \(1/2\), calibration error 0.35, a planted position-bias rate, and centered strengths \(\pm 0.4236\) plus the three-player strengths. The length-only judge scores 50 percent after the control, and three wins out of four equal-length prompts score 75 percent.

A swap score that ignores the swapped order reports a bias rate of 0 when the planted rate is 0.25. An Elo fit that treats a tie as a win for the first player returns 0.6931 instead of 0.4236.
