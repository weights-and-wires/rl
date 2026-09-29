---
title: Speech and audio
slug: 47-speech-and-audio
part: "Part 9 — Beyond text"
summary: Train speech recognisers on word error rate with MWER, and see how text-to-speech rewards from other models get hacked.
minutes: 55
gpu: false
objectives:
  - Compute word error rate from a word-level edit distance.
  - Derive the MWER gradient over an N-best list and show that it's REINFORCE with a baseline, computed exactly.
  - Measure how much a constant baseline cuts the variance of the sampled estimate.
  - Build a text-to-speech reward from intelligibility and naturalness scores, and show that a WER-only reward slows speech down.
  - Show that DPO on reward-labelled pairs lands on the same KL-regularised optimum, hack included.
prereqs:
  - 16-actor-critic
  - 36-direct-preference-optimization
  - 37-group-relative-policy-optimization
exercise: 47-speech-and-audio
---

# Speech and audio

A speech recogniser turns audio into words. It's trained like a language
model, one token at a time with cross-entropy, but it's judged by *word error
rate*: the fraction of words it got wrong. The two don't match. Cross-entropy
cares about every token's probability. Word error rate cares only about the
final transcript, and it can't be differentiated. Reinforcement learning is
the standard way to train on a score you can't differentiate.

Speech synthesis has the opposite problem. A text-to-speech (TTS) model turns
words into audio, and nobody can write down a formula for "sounds good". So
the rewards come from other models. A recogniser checks that the words are
intelligible, a speaker model checks that the voice matches, and a naturalness
predictor guesses what listeners would rate it. Each of those judges can be
fooled.

This lesson covers both directions. You build edit distance and word error
rate, derive the minimum word error rate loss, and prove that its gradient is
REINFORCE with a baseline. Then you build a toy TTS reward, watch it make the
speech slow and robotic, and fix it.

## What you already know

- REINFORCE estimates a policy gradient as the reward times the gradient of the
  log-probability of the sampled action, from The policy gradient theorem.
- Subtracting a constant baseline from the reward leaves that estimate
  unbiased and can cut its variance, from Baselines and actor-critic.
- The best policy for reward $R$ under a KL penalty of strength $\beta$ is
  $\pi^*(y) \propto \pi_{\text{ref}}(y) \exp(R(y)/\beta)$, from Language models as
  policies.
- DPO fits that optimum from preference pairs without sampling, from Direct
  preference optimisation.
- A policy that maximises a flawed reward finds the flaw.

## Count errors with edit distance

Word error rate compares a *hypothesis*, the recogniser's transcript, with the
*reference*, what was said. It counts the fewest word edits that turn one into
the other. An edit is a substitution, an insertion, or a deletion, and each
costs 1. That count is the *edit distance*, or Levenshtein distance, and word
error rate divides it by the number of reference words:

$$
\text{WER} = \frac{S + D + I}{N_{\text{ref}}}
$$

Read it as: substitutions plus deletions plus insertions, over the length of
the reference. Reference `the cat sat` and hypothesis `the cat sit` differ by
one substitution, so $\text{WER} = 1/3$. Insertions can push it above 1:
reference `a b` against hypothesis `x y z w` needs two substitutions and two
insertions, so $\text{WER} = 4/2 = 2$.

You compute the edit distance by dynamic programming. Let $d[i][j]$ be the
distance between the first $i$ reference words and the first $j$ hypothesis
words. Then

$$
d[i][j] = \min\big(d[i-1][j-1] + \mathbb{1}[r_i \ne h_j],\ d[i-1][j] + 1,\ d[i][j-1] + 1\big)
$$

Read it as: the cheapest of three last moves. Match or substitute the last two
words, delete the last reference word, or insert the last hypothesis word. The
first row is $0, 1, 2, \dots$ and so is the first column, because turning
nothing into $j$ words takes $j$ insertions. You only need the previous row to
fill the next:

```python
cur[j] = min(prev[j - 1] + (ref[i - 1] != hyp[j - 1]), prev[j] + 1, cur[j - 1] + 1)
```

## Minimise the expected errors over an N-best list

A recogniser gives a probability $P(y \mid x)$ to every transcript $y$ of audio
$x$. The quantity you'd like to lower is the *expected* number of word errors,
$\sum_y P(y \mid x)\, W(y, y^*)$, where $W$ counts the errors against the
reference $y^*$. The sum runs over every possible sentence, so you can't
compute it.

Minimum word error rate (MWER) training (Prabhavalkar et al., 2018) restricts
the sum to the $N$ best hypotheses from beam search, $y_1, \dots, y_N$, and
renormalises their probabilities to sum to one over that list:

$$
\hat P(y_i \mid x) = \frac{P(y_i \mid x)}{\sum_{j=1}^{N} P(y_j \mid x)}, \qquad \mathcal{L}_{\text{MWER}} = \sum_{i=1}^{N} \hat P(y_i \mid x)\big[W(y_i, y^*) - \hat W\big]
$$

Read it as: treat the beam as the whole world, reweight it so it sums to one,
and average each hypothesis's error count minus $\hat W$. The paper calls
$\hat W$ the average number of word errors over the N-best list, "applied as a
form of variance reduction, since it does not affect the gradient." It's a
constant: in the exercise it's the plain mean of the $N$ error counts.

In numpy, with `logscores` holding $\ell_i = \log P(y_i \mid x)$:

```python
p_hat = np.exp(logscores - logscores.max())
p_hat /= p_hat.sum()
loss = np.sum(p_hat * (errors - errors.mean()))
```

The renormalised probabilities are a softmax of the log-scores, so the shift
by the maximum changes nothing and prevents overflow. The paper trains with
$\mathcal{L}_{\text{MWER}} + \lambda \mathcal{L}_{\text{CE}}$, keeping a little
cross-entropy with $\lambda = 0.01$ and $N = 4$, and starts from a model
already trained with cross-entropy alone. It reports up to 8.2 percent
relative improvement in word error rate.

## Show that the gradient is REINFORCE

Differentiate the expected errors over the list, $E = \sum_j \hat P_j W_j$,
with respect to the log-score $\ell_i$. The softmax derivative is
$\partial \hat P_j / \partial \ell_i = \hat P_j(\mathbb{1}[i = j] - \hat P_i)$, so

$$
\frac{\partial \mathcal{L}_{\text{MWER}}}{\partial \ell_i} = \hat P_i \big(W_i - E\big), \qquad E = \sum_{j=1}^{N} \hat P_j W_j
$$

Read it as: a hypothesis with more errors than the list's expected errors gets
a positive gradient, so gradient descent lowers its score. One with fewer
errors gets raised. The push is weighted by how likely the hypothesis already
is. The constant $\hat W$ has vanished, as the paper says.

Now compare with REINFORCE. Sample a hypothesis $k$ from $\hat P$. The
gradient of its log-probability is $\nabla_\ell \log \hat P_k = e_k - \hat P$,
where $e_k$ is the one-hot vector for $k$. The REINFORCE estimate with a
baseline $b$ is $(W_k - b)(e_k - \hat P)$. Its average over $k \sim \hat P$
is

$$
\sum_{k=1}^{N} \hat P_k (W_k - b)(e_k - \hat P) = \hat P \odot (W - E)
$$

Read it as: the exact expectation of the sampled estimate is the MWER
gradient, whatever constant $b$ you choose. The baseline drops out because
$\sum_k \hat P_k (e_k - \hat P) = \hat P - \hat P = 0$. So MWER is REINFORCE with
a baseline, where the "samples" are the whole beam and the expectation is
computed exactly instead of estimated.

```python
expected = np.sum(p_hat * errors)
grad = p_hat * (errors - expected)
```

> [!KEY]
> MWER's gradient, $\hat P_i(W_i - E)$, is the exact expectation of REINFORCE
> over the beam. Every constant baseline gives the same exact gradient. The
> baseline matters only when you sample, where it sets the variance.

> [!WARNING]
> If you compute $\hat W$ as the $\hat P$-weighted mean and let gradients flow
> through it, the loss is $E - E = 0$ for every input, and so is its gradient.
> The baseline must be a constant with respect to the log-scores.

### Worked example

Four hypotheses have log-scores $(2.0, 1.0, 0.5, -1.0)$ and word errors
$(0, 1, 2, 3)$:

- The renormalised probabilities are $\hat P = (0.6095, 0.2242, 0.1360, 0.0303)$.
- The expected errors are $E = 0.5872$.
- The gradient is $\hat P_i(W_i - E) = (-0.3579, 0.0925, 0.1921, 0.0732)$. It sums
  to zero, so a step only moves probability around inside the beam.
- The plain mean of the errors is $\hat W = 1.5$, so the loss is
  $0.5872 - 1.5 = -0.9128$.

The variance of one sampled estimate depends on $b$, even though its mean
doesn't. Summed over the four coordinates, it's $1.057$ with $b = 0$ and $0.467$
with $b = E$, a cut by a factor of $2.26$. The plain mean $b = 1.5$ does better
here, $0.315$. The best constant, from the formula in Baselines and
actor-critic, is $b^* = 1.192$ with variance $0.262$. Self-critical sequence
training (Rennie et al., 2017) uses a different baseline: the reward of the
model's own greedy output.

## Preference tuning for text-to-speech

TTS models come in two kinds, and they need different machinery:

- **Codec language models** write audio as a sequence of discrete tokens from
  a neural codec, one token at a time. They're language models, so DPO, PPO,
  and GRPO apply unchanged. Seed-TTS (Anastassiou et al., 2024) compares
  PPO and REINFORCE with external rewards against DPO, and finds both work.
  SpeechAlign (Zhang et al., 2024) prefers tokens from real speech over the
  model's own tokens and iterates.
- **Flow-matching models** such as F5-TTS generate audio features by
  integrating a velocity field, like the image flows of Flow models and GRPO.
  They have no per-step probability. F5R-TTS (Sun et al., 2025) gives the
  network a Gaussian output, so each step has a log-probability, and pretrains
  it with a Gaussian negative log-likelihood.

The F5R-TTS pretraining loss predicts a mean $\mu_\theta$ and a standard
deviation $\sigma_\theta$ for the flow target $x_1 - x_0$:

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}\Big[\frac{\big(\mu_\theta - (x_1 - x_0)\big)^2}{2\sigma_\theta^2} + \log \sigma_\theta\Big]
$$

Read it as: the ordinary flow-matching squared error, divided by a learned
variance, plus a term that stops the variance from growing without limit. It's
a Gaussian negative log-likelihood without its constant. F5R-TTS then runs GRPO
with the reward $R = \lambda_W(1 - \text{WER}) + \lambda_S\, \text{SIM}$, where
SIM is the cosine similarity of speaker embeddings. It reports a 29.5 percent
relative cut in WER and a 4.6 percent relative gain in SIM on zero-shot voice
cloning.

Every term of that reward comes from another model. WER comes from an ASR
model run on the synthesised audio. SIM comes from a speaker-verification
model. Naturalness comes from a mean opinion score predictor such as UTMOS
(Saeki et al., 2022), trained to guess listener ratings. The same machinery
works for audio understanding: R1-AQA (Li et al., 2025) trains an audio
question-answering model with GRPO, rewarding a correct answer and the right
answer format.

## Watch a WER reward slow the speech down

Seed-TTS reports what happens when you optimise WER alone: the speech gets
slower and clearer, the WER falls, and the naturalness drops. Slow, careful
speech is easy to transcribe. Nothing in a WER reward says it should also
sound human.

The exercise reproduces this with a toy policy that picks one speaking rate
$\rho$ from eleven options between 0.6 and 1.6 times normal speed. Two given
functions stand in for the judge models. The stand-in ASR model's error rises
with speed, from $0.045$ at $\rho = 0.6$ to $0.113$ at $\rho = 1.0$. The
stand-in naturalness score peaks at $4.5$ for $\rho = 1.0$ and falls to $1.47$
at $\rho = 0.6$. The reward mixes them:

$$
R(\rho) = w_{\text{intel}}\big(1 - \text{WER}(\rho)\big) + w_{\text{nat}}\, \frac{\text{UTMOS}(\rho)}{5}
$$

Read it as: intelligibility plus a weighted naturalness score, scaled to a
similar range. With $w_{\text{nat}} = 0$, it's the WER-only reward.

The pretrained policy $\pi_{\text{ref}}$ talks a little fast, with a mean rate
of $1.1$. The KL-regularised optimum is a softmax of $\log \pi_{\text{ref}} +
R/\beta$, which you compute exactly. The following table shows that optimum at
$\beta = 0.01$:

| Policy | Mean rate | Expected WER | Expected UTMOS |
|---|---|---|---|
| Reference | 1.100 | 0.1450 | 3.138 |
| WER-only optimum | 0.672 | 0.0538 | 2.018 |
| WER plus UTMOS optimum | 1.000 | 0.1130 | 4.499 |

The WER-only optimum wins on its own metric and loses more than a point of
naturalness. Adding the naturalness term moves the optimum to normal speed,
at a WER between the two.

### DPO lands on the same optimum

Suppose you skip the reward optimisation and run DPO instead, on pairs of
rates labelled by the same reward. For every pair $(i, j)$ drawn from
$\pi_{\text{ref}}$, a Bradley-Terry annotator prefers $i$ with probability
$\sigma(R_i - R_j)$. With the implicit rewards $u_i = \beta(\log \pi_\theta(i) -
\log \pi_{\text{ref}}(i))$, the expected DPO loss is

$$
\mathcal{L}(\theta) = \frac{\sum_{i \ne j} w_{ij}\,\big[-\sigma(R_i - R_j) \log \sigma(u_i - u_j)\big]}{\sum_{i \ne j} w_{ij}}, \qquad w_{ij} = \pi_{\text{ref}}(i)\,\pi_{\text{ref}}(j)
$$

Read it as: for every ordered pair, the chance that $i$ wins times the DPO loss
for "$i$ beats $j$", weighted by how often the reference would propose that
pair. Each unordered pair contributes a cross-entropy between the label
$\sigma(R_i - R_j)$ and the prediction $\sigma(u_i - u_j)$. That's smallest when
$u_i - u_j = R_i - R_j$ for every pair, which means $\pi_\theta \propto
\pi_{\text{ref}} \exp(R/\beta)$. DPO fits the KL optimum, and it inherits the
reward's hack.

```python
margin = u[:, None] - u[None, :]
label = torch.sigmoid(R[:, None] - R[None, :])
loss = (w * -label * F.logsigmoid(margin)).sum() / w.sum()
```

> [!KEY]
> A TTS reward is a stack of other models' opinions. Whatever they miss, the
> policy exploits, and DPO on pairs ranked by the same reward reaches the same
> exploit.

> [!RECAP]
> - WER is word-level edit distance over the reference length, and it can
>   exceed 1.
> - MWER softmaxes the N-best log-scores and subtracts a constant $\hat W$ from
>   the error counts.
> - Its gradient is $\hat P_i(W_i - E)$, the exact expectation of REINFORCE
>   with any constant baseline.
> - The baseline changes only the variance of the sampled estimate.
> - A WER-only TTS reward drives the KL optimum to slow speech, and DPO on
>   reward-ranked pairs converges to the same optimum.

## Build it step by step

The checks test the functions in this order.

1. **`edit_distance(ref, hyp)`.** Takes two lists of tokens and returns an
   int. Keep one row of the table, `prev = list(range(len(hyp) + 1))`, and
   build each next row starting from `[i]`. The common mistake is starting
   the first column at 0, which makes deletions free.
2. **`wer(ref, hyp)`.** Split both strings with `.split()`, call
   `edit_distance`, and divide by the number of reference words, not
   hypothesis words.
3. **`mwer_loss(logscores, errors)`.** Softmax with the maximum subtracted,
   then `np.sum(p_hat * (errors - errors.mean()))`.
4. **`mwer_grad(logscores, errors)`.** Return `p_hat * (errors - expected)`.
   The checks compare it with finite differences of your `mwer_loss`, so the
   two must agree. The common mistake is using `errors.mean()` in the
   gradient instead of the $\hat P$-weighted mean.
5. **`reinforce_mwer_samples(logscores, errors, n_samples, baseline, seed)`.**
   Draw indices with `rng.choice(N, size=n_samples, p=p_hat)`, build the score
   rows with `np.eye(N)[k] - p_hat`, and multiply each row by `errors[k] -
   baseline`. Return shape `(n_samples, N)`.
6. **`tts_reward(rate, w_intel, w_nat)`.** One line using the given `asr_wer`
   and `utmos`.
7. **`kl_optimal_policy(ref_logprobs, rewards, beta)`.** A softmax of
   `ref_logprobs + rewards / beta` along the last axis, with the maximum
   subtracted first. For $\beta = 10^{-4}$ the logits reach the tens of
   thousands, so the subtraction is required.
8. **`dpo_population_loss(logits, ref_logprobs, rewards, beta)`.** In torch.
   Take `torch.log_softmax(logits, -1)`, form `u`, build the `(K, K)` margin
   and label matrices by broadcasting, zero the diagonal of the weights, and
   use `torch.nn.functional.logsigmoid`. The check confirms the gradient
   vanishes at the closed-form optimum, then minimises your loss and compares.

## Cost and scaling

MWER needs an N-best list for every training utterance, and beam search is the
expensive part. The following table shows how each piece grows:

| Piece | Cost per utterance | Notes |
|---|---|---|
| Beam search for the N-best list | About $N$ times greedy decoding | The main cost; the paper uses $N = 4$ |
| Edit distance per hypothesis | $O(L_{\text{ref}} \times L_{\text{hyp}})$ | Microseconds for sentences |
| MWER loss and gradient | $O(N)$ once the log-scores exist | One backward pass through $N$ sequences |
| TTS reward | One ASR pass, one speaker model pass, one MOS pass per sample | Judges often cost more than the TTS model |

For TTS with GRPO, each prompt needs a group of synthesised utterances, and
each one must be run through every judge. Batch the judges, because running
them one utterance at a time dominates training time.

## Where this breaks

**The beam isn't the world.** MWER only moves probability among the $N$
hypotheses it sees, because its gradient sums to zero over the list. A
correct transcript outside the beam gets no push at all. The cross-entropy
term keeps the model anchored to the reference, which is why the paper finds
it important for stability.

**Normalisation mismatches.** Beam search often ranks hypotheses by
length-normalised scores. If the loss uses unnormalised ones, the beam and the
loss disagree about which hypotheses are likely. Use the same scores in both.

**Text normalisation.** Casing, punctuation, and numerals ("5" against
"five") change WER. Normalise the text the same way for the reward and for
evaluation.

**The judge is also the exam.** A TTS policy trained against one ASR model
learns that model's quirks. Evaluate WER with a different recogniser, and
speaker similarity with a different speaker model.

**Naturalness predictors are hackable too.** A MOS predictor rewards clean,
studio-like audio and can prefer flat, over-smoothed prosody. Adding it fixes
the slow-speech hack in the toy, but in practice it introduces its own. Mix
several rewards, keep a KL penalty, and listen to samples.

## Further reading

- [Prabhavalkar et al., Minimum Word Error Rate Training for Attention-based
  Sequence-to-Sequence Models (2018)](https://arxiv.org/abs/1712.01818) defines
  the N-best MWER loss and compares it with sampling.
- [Rennie et al., Self-critical Sequence Training for Image Captioning
  (2017)](https://arxiv.org/abs/1612.00563) uses the greedy decode's reward as
  the REINFORCE baseline for a sequence metric.
- [Anastassiou et al., Seed-TTS (2024)](https://arxiv.org/abs/2406.02430)
  compares RL with WER, speaker, and emotion rewards against DPO, and reports
  the slow-speech hack.
- [Sun et al., F5R-TTS (2025)](https://arxiv.org/abs/2504.02407) adds a Gaussian
  head to a flow-matching TTS model so GRPO can train it.
- [Zhang et al., SpeechAlign (2024)](https://arxiv.org/abs/2404.05600) builds
  preference pairs from real and synthetic codec tokens.
- [Saeki et al., UTMOS (2022)](https://arxiv.org/abs/2204.02152) is the
  naturalness predictor that many TTS rewards borrow.
- [Li et al., R1-AQA (2025)](https://arxiv.org/abs/2503.11197) applies GRPO to
  audio question answering.

Open questions that researchers are working on:

- Which learned judges of naturalness and expressiveness agree with human
  listeners well enough to optimise against?
- How should a full-duplex spoken dialogue model be rewarded for turn-taking,
  when to speak and when to stop?
- Can a TTS model be trained with RL on several judges without drifting from
  the transcript it was asked to read?

## The exercise

You build both halves of the lesson. For speech recognition, write
`edit_distance`, `wer`, `mwer_loss`, `mwer_grad`, and
`reinforce_mwer_samples`. For speech synthesis, write `tts_reward`,
`kl_optimal_policy`, and `dpo_population_loss`, using the given toy world:
`SPEAKING_RATES`, `asr_wer`, `utmos`, and `reference_logprobs`.

The checks verify edit distance on hand cases, including empty sequences and
symmetry, and WER of $1/3$ and of $2$. They check the MWER loss of $-0.9128$,
the gradient against the worked example and finite differences, and its
invariance to shifting log-scores or error counts. They confirm the sampled
REINFORCE estimate is unbiased with and without a baseline, and that the
baseline $E$ cuts its variance by at least 1.8 times. Last, they check that a
WER-only reward pushes the mean rate below 0.7 and loses over a point of
naturalness, that the combined reward's optimum sits at normal speed, and that
DPO's loss has zero gradient at the KL optimum and converges to it.
