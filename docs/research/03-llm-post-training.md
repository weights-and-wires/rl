# RL for language-model post-training and alignment

Research notes for the course module on reinforcement learning (RL) for large
language model (LLM) post-training. The notes run from InstructGPT-era RLHF to
the 2025-2026 work on reasoning models, off-policy infrastructure, and reward
hacking. Every topic follows the same six-part template:

1. Core intuition.
2. Key equations (KaTeX-compatible LaTeX).
3. A small worked numeric example.
4. Canonical references with links.
5. A CPU exercise idea with verifiable checks.
6. Common mistakes and failure modes.

## How to read these notes

- **Verification status.** Equations marked "verified" were checked against the
  paper's HTML or PDF text during research (September 2026). Numeric examples
  were computed with a short Python script, and several exercise checks were
  prototyped in PyTorch on CPU (noted inline). Anything not checked against a
  primary source carries the tag **UNVERIFIED**.
- **Toy scale.** The course can't train real LLMs. Every exercise uses a toy
  policy: a tabular categorical, a bigram model, or a 2-layer MLP over a
  vocabulary of 8 tokens with sequences of length 8 or less. For exact
  expectations, use sequences of length 4 or less so you can enumerate all
  $8^4 = 4096$ sequences.
- **Time budget.** Every proposed check suite targets well under 60 seconds on
  a laptop CPU. The prototypes in these notes ran in under 1 second each.

## Notation

The following notation is used throughout:

| Symbol | Meaning |
|---|---|
| $x$ | Prompt (context). |
| $y = (y_1, \dots, y_T)$ | Response (completion) of length $T = \lvert y \rvert$. |
| $s_t = (x, y_{<t})$ | Token-level state: the prompt plus the prefix. |
| $\pi_\theta(y \mid x) = \prod_t \pi_\theta(y_t \mid s_t)$ | Policy (the LM being trained). |
| $\pi_{\text{ref}}$ | Reference policy, usually the SFT model. |
| $\pi_{\text{old}}$ | Policy that generated the current batch (PPO anchor). |
| $r(x, y)$ or $R$ | Sequence-level (outcome) reward. |
| $\beta$ | KL-penalty coefficient. |
| $\sigma(z) = 1/(1 + e^{-z})$ | Logistic sigmoid. |
| $G$ | Group size (samples per prompt) in group-based methods. |
| $\operatorname{sg}[\cdot]$ | Stop-gradient. |

---

## 1. The post-training pipeline and InstructGPT

### Intuition

A pretrained LM models the distribution of internet text. It isn't yet a
helpful assistant: it continues text instead of following instructions.
Post-training reshapes this distribution in stages:

1. **Supervised fine-tuning (SFT).** Imitate demonstrations of good behaviour.
   This moves the model to the right region of output space cheaply.
2. **Preference tuning.** Humans (or AIs) compare pairs of outputs. You either
   fit a reward model (RM) and optimise against it with RL (RLHF), or you
   optimise the policy on the comparisons directly (DPO and relatives).
3. **RL with verifiable rewards (RLVR).** For tasks with checkable answers
   (maths, code, formatted outputs), use a programmatic reward and run RL at
   scale. Since 2024, this stage produces "reasoning models" such as o1 and
   DeepSeek-R1.

Modern pipelines interleave these stages. DeepSeek-R1 uses cold-start SFT, then
reasoning RL, then rejection-sampling SFT, then a final RL stage for
helpfulness and harmlessness. Qwen3 uses long-CoT cold start, reasoning RL,
thinking-mode fusion SFT, and general RL (both verified; see section 9).

### Key equations (InstructGPT, verified from the paper)

The reward model loss over $K$ ranked responses per prompt ($K$ between 4 and
9) uses all $\binom{K}{2}$ pairs:

$$
\mathcal{L}_{\text{RM}}(\phi) = -\frac{1}{\binom{K}{2}}\,
\mathbb{E}_{(x, y_w, y_l) \sim D}\Big[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\Big]
$$

The RL objective ("PPO-ptx") adds a pretraining-likelihood term to the
KL-penalised reward:

$$
\text{objective}(\phi) =
\mathbb{E}_{(x,y) \sim \pi_\phi^{\text{RL}}}\Big[r_\theta(x,y) - \beta \log \frac{\pi_\phi^{\text{RL}}(y \mid x)}{\pi^{\text{SFT}}(y \mid x)}\Big]
+ \gamma\, \mathbb{E}_{x \sim D_{\text{pretrain}}}\big[\log \pi_\phi^{\text{RL}}(x)\big]
$$

Setting $\gamma = 0$ gives plain PPO. The $\gamma$ term reduces the "alignment
tax" (regressions on public NLP benchmarks).

### Worked example

Scale matters less than you might expect. InstructGPT used roughly 13k SFT
prompts, 33k RM prompts, and 31k PPO prompts. Labelers preferred outputs of the
1.3B InstructGPT model over outputs of the 175B GPT-3 model.

### References

- Ouyang et al., "Training language models to follow instructions with human
  feedback" (InstructGPT), 2022. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- Christiano et al., "Deep reinforcement learning from human preferences",
  2017. [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)
- Ziegler et al., "Fine-Tuning Language Models from Human Preferences", 2019.
  [arXiv:1909.08593](https://arxiv.org/abs/1909.08593)
- Stiennon et al., "Learning to summarize from human feedback", 2020.
  [arXiv:2009.01325](https://arxiv.org/abs/2009.01325)
- Bai et al., "Training a Helpful and Harmless Assistant with RLHF" (Anthropic
  HH), 2022. [arXiv:2204.05862](https://arxiv.org/abs/2204.05862)
- Lambert, *Reinforcement Learning from Human Feedback* (open book), 2025.
  [arXiv:2504.12501](https://arxiv.org/abs/2504.12501)

### Exercise idea: a three-stage toy pipeline

Build a bigram "pretrained" model over 8 tokens, fit it by maximum likelihood
to a corpus drawn from a random bigram matrix. Then:

1. SFT on 200 demonstrations from a target bigram matrix.
2. Fit a Bradley-Terry RM on pairwise comparisons labelled by a hidden reward
   $r^\star(y) = $ count of token 3 minus count of token 7.
3. Run REINFORCE with a KL penalty against the SFT model.

**Checks:**

- The SFT model's NLL on held-out demonstrations drops below the pretrained
  model's NLL.
- The RM's pairwise accuracy on held-out pairs exceeds 0.8.
- After RL, $\mathbb{E}[r^\star]$ increases and the measured
  $\mathrm{KL}(\pi \,\|\, \pi_{\text{SFT}})$ stays below a bound set by $\beta$.

### Common mistakes

- Treating the stages as independent. RL can only reweight behaviours the SFT
  model already produces with non-negligible probability.
- Forgetting the KL (or pretraining) term and then being surprised by
  degenerate outputs.
- Confusing "reference model" (the KL anchor) with "old policy" (the PPO
  importance-sampling anchor). They are different models after the first PPO
  iteration.

---

## 2. The LM as a policy: token MDP, bandit view, and the KL-regularised optimum

### Intuition

You can view generation in two equivalent ways:

- **Token-level MDP.** The state is $s_t = (x, y_{<t})$, the action is the next
  token $y_t \in \mathcal{V}$, transitions are deterministic concatenation, and
  the episode ends at EOS or at the length limit. Reward is usually sparse:
  zero until the final token. PPO with a value head uses this view.
- **Sequence-level contextual bandit.** The context is $x$, the action is the
  whole response $y$, and the reward $r(x, y)$ arrives at once. RLOO, ReMax,
  Kimi k1.5, and GSPO lean on this view. It is natural because the reward model
  scores only complete responses.

With deterministic transitions and terminal reward, the two views give the same
objective. They differ in variance, credit assignment, and where the KL penalty
lives.

The central object is the KL-regularised objective. It trades reward against
staying close to the reference model, and it has a closed-form optimum. DPO,
best-of-$n$ analysis, Kimi's mirror-descent step, and the "RL as inference"
view all build on it.

### Key equations

The KL-regularised objective, per prompt:

$$
J(\pi) = \mathbb{E}_{y \sim \pi(\cdot \mid x)}\big[r(x,y)\big]
- \beta\, \mathrm{KL}\big(\pi(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\big)
$$

The optimum (Gibbs distribution):

$$
\pi^\star(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x)\, \exp\!\Big(\frac{r(x,y)}{\beta}\Big),
\qquad
Z(x) = \sum_{y} \pi_{\text{ref}}(y \mid x) \exp\!\Big(\frac{r(x,y)}{\beta}\Big)
$$

Proof sketch. Expand $\mathrm{KL}(\pi \,\|\, \pi^\star)$:

$$
\mathrm{KL}(\pi \,\|\, \pi^\star)
= \mathbb{E}_\pi\Big[\log \frac{\pi}{\pi_{\text{ref}}} - \frac{r}{\beta} + \log Z\Big]
= \mathrm{KL}(\pi \,\|\, \pi_{\text{ref}}) - \frac{1}{\beta}\mathbb{E}_\pi[r] + \log Z
$$

Rearranging gives the identity

$$
J(\pi) = \beta \log Z(x) - \beta\, \mathrm{KL}(\pi \,\|\, \pi^\star)
$$

so $J$ is maximised exactly at $\pi = \pi^\star$, with optimal value
$\beta \log Z(x)$.

**KL chain rule.** For autoregressive policies the sequence KL decomposes into
per-token KLs:

$$
\mathrm{KL}\big(\pi(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\big)
= \mathbb{E}_{y \sim \pi}\Big[\sum_{t=1}^{T} \log \frac{\pi(y_t \mid s_t)}{\pi_{\text{ref}}(y_t \mid s_t)}\Big]
= \mathbb{E}_{y \sim \pi}\Big[\sum_{t=1}^{T} \mathrm{KL}\big(\pi(\cdot \mid s_t) \,\|\, \pi_{\text{ref}}(\cdot \mid s_t)\big)\Big]
$$

**Per-token reward shaping.** Define
$r_t = -\beta \log \frac{\pi(y_t \mid s_t)}{\pi_{\text{ref}}(y_t \mid s_t)} + \mathbf{1}[t = T]\, r(x, y)$.
The undiscounted return $\sum_t r_t$ equals $r(x,y) - \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$,
whose expectation is $J(\pi)$.

**Token-level soft optimality.** In the token MDP with this regulariser,
$\beta \log \frac{\pi^\star(a \mid s)}{\pi_{\text{ref}}(a \mid s)} = Q^\star(s,a) - V^\star(s)$,
and summing along a trajectory telescopes to $r(x,y) - V^\star(s_0)$
(Rafailov et al., "From $r$ to $Q^\star$").

### Worked example

Take three responses with $\pi_{\text{ref}} = (0.5, 0.3, 0.2)$ and
$r = (1, 0, 2)$, and $\beta = 1$:

- Unnormalised weights: $0.5e^1, 0.3e^0, 0.2e^2 = 1.359, 0.300, 1.478$.
- $Z = 3.137$, so $\pi^\star = (0.433, 0.096, 0.471)$.
- $\mathbb{E}_{\pi^\star}[r] = 1.376$, $\mathrm{KL}(\pi^\star \,\|\, \pi_{\text{ref}}) = 0.232$.
- $J(\pi^\star) = 1.376 - 0.232 = 1.143 = \log Z$. The identity checks out.

With $\beta = 0.5$, $\pi^\star = (0.248, 0.020, 0.732)$ and KL rises to 0.722.
With $\beta = 2$, $\pi^\star = (0.494, 0.180, 0.326)$ and KL falls to 0.061.

### References

- Rafailov et al., "From $r$ to $Q^*$: Your Language Model is Secretly a
  Q-Function", 2024. [arXiv:2404.12358](https://arxiv.org/abs/2404.12358)
- Korbak, Perez, and Buckley, "RL with KL penalties is better viewed as
  Bayesian inference", 2022. [arXiv:2205.11275](https://arxiv.org/abs/2205.11275)
- Rafailov et al., DPO (derivation of the closed form in Appendix A.1), 2023.
  [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

### Exercise idea: verify the Gibbs optimum three ways

Use a bigram policy over $V = 8$ tokens with fixed length $T = 3$, so the
response space has $8^3 = 512$ sequences. Let $r(y) = $ number of even tokens.

1. Compute $\pi^\star$ in closed form by enumeration.
2. Train a tabular sequence-level policy (512 logits) by exact gradient ascent
   on $J$.
3. Train a position-dependent bigram policy (one transition matrix per
   position) by exact gradient ascent on $J$.

**Checks:**

- `allclose(pi_trained_tabular, pi_star, atol=1e-4)`.
- $J(\pi_{\text{trained}}) \le \beta \log Z$ always, with equality within
  $10^{-5}$ at convergence.
- The chain rule: the sequence KL computed by enumeration equals the expected
  sum of per-token KLs, to $10^{-10}$.
- Bonus (why tabular sequences matter): because $r$ is additive over tokens
  and $\pi_{\text{ref}}$ is Markov, $\pi^\star$ is a (time-inhomogeneous) Markov
  chain, so the position-dependent bigram policy can also reach $\pi^\star$.
  Swap in a non-additive reward, such as "1 if all tokens are strictly
  increasing", and show the bigram policy's best $J$ falls below
  $\beta \log Z$ (this holds in general, not for every $\pi_{\text{ref}}$;
  assert the gap exceeds $10^{-3}$ for your seeded instance).

### Common mistakes

- Writing the KL the wrong way round. The standard objective penalises the
  reverse KL $\mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})$, which is mode-seeking.
- Forgetting that $Z(x)$ depends on the prompt. It cancels in pairwise
  comparisons of the same prompt (the key to DPO), but not across prompts.
- Assuming a parametric policy can represent $\pi^\star$. The closed form is
  over all distributions; the realised policy is a projection.

---

## 3. Reward modelling

### 3.1 Bradley-Terry reward models and the pairwise loss

#### Intuition

Humans compare more reliably than they score. The Bradley-Terry (BT) model says
each response has a latent scalar quality, and the probability that one wins
is the sigmoid of the quality difference. A reward model is a neural network
(usually the SFT model with a scalar head on the final token) trained to
maximise the BT likelihood of the observed comparisons. Rewards are identified
only up to a per-prompt additive constant.

#### Key equations

$$
p(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)
= \frac{\exp r(x,y_w)}{\exp r(x,y_w) + \exp r(x,y_l)}
$$

$$
\mathcal{L}_{\text{BT}}(\phi) = -\mathbb{E}_{(x, y_w, y_l)}\big[\log \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)\big]
$$

The gradient for one pair is
$-(1 - \sigma(\Delta))\,(\nabla r_\phi(x,y_w) - \nabla r_\phi(x,y_l))$ with
$\Delta = r_\phi(x,y_w) - r_\phi(x,y_l)$. Pairs the RM already ranks
confidently contribute little.

Some RMs add a margin (Llama 2 used a margin from the labeler's confidence) or
a small regulariser on $r^2$ to pin the free constant. These are practical
additions rather than part of the BT model.

#### Worked example

If $r(x, y_w) = 2$ and $r(x, y_l) = 1$, then $p = \sigma(1) = 0.731$ and the
loss is $-\log 0.731 = 0.313$. Adding 100 to both rewards changes nothing.

#### References

- Bradley and Terry, "Rank Analysis of Incomplete Block Designs", *Biometrika*,
  1952 (classical source; no arXiv).
- Ouyang et al., InstructGPT, 2022 (RM with $K$-wise rankings).
  [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

#### Exercise idea: recover a known reward

Let 16 responses have hidden rewards $r^\star \sim \mathcal{N}(0, 1)$. Sample
5000 pairs uniformly and label them with BT probabilities. Fit a 16-parameter
tabular RM with Adam.

**Checks:**

- After centring both, `corr(r_hat, r_star) > 0.98`.
- `max |(r_hat - mean) - (r_star - mean)| < 0.2`.
- The fitted model's held-out log-loss is within 0.01 of the log-loss of the
  true $r^\star$ (the Bayes floor).
- Shifting all logits by a constant leaves the loss unchanged (identifiability).

#### Common mistakes

- Reading RM scores as absolute. Only differences within a prompt carry
  meaning, so normalise or whiten before using them as PPO rewards.
- Training on deterministic labels (always pick the higher reward). The BT
  maximum-likelihood estimate then diverges: reward gaps grow without bound.
  This is also the root of the DPO overfitting argument in IPO (section 6.2).
- Ignoring label noise. Inter-annotator agreement on real preference data is
  often only about 60-75% (**UNVERIFIED** as a general range; InstructGPT
  reports around 73% agreement among training labelers).

### 3.2 Reward-model overoptimisation and scaling laws

#### Intuition

A learned RM is a proxy. Optimising hard against it first improves the true
("gold") reward, then the gold reward falls while the proxy reward keeps
rising. This is Goodhart's law. Gao et al. measured it in a synthetic setup
where a large "gold" RM plays the human, and found clean functional forms in
terms of the distance the policy has moved from its initialisation.

#### Key equations (verified from the paper)

Let $d := \sqrt{\mathrm{KL}(\pi \,\|\, \pi_{\text{init}})}$. For best-of-$n$
sampling and for RL respectively:

$$
R_{\text{bon}}(d) = d\,(\alpha_{\text{bon}} - \beta_{\text{bon}}\, d),
\qquad
R_{\text{RL}}(d) = d\,(\alpha_{\text{RL}} - \beta_{\text{RL}} \log d)
$$

with $R(0) := 0$. The KL for best-of-$n$ is computed analytically as
$\log n - \frac{n-1}{n}$ (see section 5). The paper also finds the following:

- The $\alpha$ and $\beta$ coefficients scale smoothly (roughly
  logarithmically) with RM parameter count.
- RL is slower than best-of-$n$ both to optimise and to overoptimise, as a
  function of KL.
- Policy size has a weak effect on the amount of overoptimisation.
- The KL penalty acts mostly like early stopping in their setup; it doesn't
  improve the gold-reward-versus-KL frontier.

#### Worked example

These are illustrative coefficients, not fitted values. For
$R_{\text{bon}}$ with $\alpha = 1, \beta = 0.25$, the peak is at
$d^\star = \alpha / (2\beta) = 2$, that is, KL = 4 nats, which corresponds to
$n \approx 148$. For $R_{\text{RL}}$ with $\alpha = 1, \beta = 0.5$, the
derivative $\alpha - \beta \log d - \beta = 0$ gives
$d^\star = e^{\alpha/\beta - 1} = e \approx 2.72$ (KL $\approx 7.4$ nats).

#### References

- Gao, Schulman, and Hilton, "Scaling Laws for Reward Model Overoptimization",
  2022. [arXiv:2210.10760](https://arxiv.org/abs/2210.10760)

#### Exercise idea: watch Goodhart happen

Hidden gold reward over 64 tabular responses. Fit a proxy RM on only 300 noisy
comparisons. Run best-of-$n$ against the proxy for $n \in \{1, 2, 4, \dots, 1024\}$
and compute exact expected gold and proxy rewards under the BoN distribution
(section 5 gives the formula).

**Checks:**

- Proxy reward is monotonically non-decreasing in $n$.
- Gold reward peaks at some $n^\star < 1024$ and then decreases (seed the RM
  noise so this holds; assert the argmax is interior).
- A least-squares fit of $R_{\text{bon}}(d)$ to the gold curve has
  $R^2 > 0.9$.
- Refit the proxy with 3000 comparisons and assert that $n^\star$ moves right.

#### Common mistakes

- Plotting against steps instead of KL. KL is the natural x-axis because it
  measures how far you moved the distribution.
- Reading the KL penalty as a cure. It changes where you stop, not the
  frontier.

### 3.3 Reward ensembles

#### Intuition

Different RMs make different errors. Combining an ensemble conservatively
(mean minus a multiple of the standard deviation, or worst case) penalises
responses where the RMs disagree, which is where exploitation hides.

#### Key equations

For $K$ reward models $r_1, \dots, r_K$:

- Mean: $\bar{r} = \frac{1}{K}\sum_k r_k$.
- Worst-case optimisation (WCO): $r_{\text{WCO}} = \min_k r_k$.
- Uncertainty-weighted optimisation (UWO):
  $r_{\text{UWO}} = \bar{r} - \lambda \cdot \operatorname{Var}_k[r_k]$.

#### Findings

Coste et al. found that ensembles (with WCO or UWO) reduce overoptimisation
for both BoN and PPO. Eisenstein et al. found that ensembles mitigate but
don't eliminate reward hacking. When all members share a pretraining seed,
they share errors, so pretrain-diverse ensembles help more than
fine-tune-diverse ones.

#### References

- Coste et al., "Reward Model Ensembles Help Mitigate Overoptimization", 2023.
  [arXiv:2310.02743](https://arxiv.org/abs/2310.02743)
- Eisenstein et al., "Helping or Herding? Reward Model Ensembles Mitigate but
  do not Eliminate Reward Hacking", 2023.
  [arXiv:2312.09244](https://arxiv.org/abs/2312.09244)

#### Exercise idea

Extend the Goodhart exercise with 5 proxy RMs trained on bootstrap resamples.

**Checks:**

- The gold reward at the proxy-optimal BoN point is higher for WCO than for a
  single RM, averaged over 20 seeds.
- Ensemble disagreement (std across members) is higher on the top-ranked
  responses than on random responses.

#### Common mistakes

- Using ensembles of RMs fine-tuned from the same checkpoint on the same data.
  Their errors correlate, so the variance term is small exactly where it
  matters.

### 3.4 Generative reward models and LLM-as-a-judge

#### Intuition

Instead of a scalar head, ask an LLM to judge. A generative verifier (GenRM)
frames verification as next-token prediction: the reward is the probability of
the token "Yes" after a prompt asking whether the answer is correct, optionally
after a chain-of-thought critique. This reuses the model's reasoning and lets
you spend test-time compute on verification (sample several critiques and
average).

#### Key equations

$$
r_{\text{GenRM}}(x, y) = p_\theta(\texttt{Yes} \mid x, y, I_{\text{verify}}),
\qquad
r_{\text{GenRM-CoT}}(x, y) = \frac{1}{K}\sum_{k=1}^{K} p_\theta\big(\texttt{Yes} \mid x, y, c_k, I\big),\;
c_k \sim p_\theta(\cdot \mid x, y, I)
$$

#### Findings

- Zheng et al. showed strong LLM judges agree with human preferences at levels
  similar to inter-human agreement on MT-Bench, and documented position,
  verbosity, and self-enhancement biases.
- Rubric-based rewards ("Rubrics as Rewards") extend RLVR-style training to
  domains without a single checkable answer, by scoring against
  instance-specific checklists with an LLM judge.

#### References

- Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena",
  2023. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
- Zhang et al., "Generative Verifiers: Reward Modeling as Next-Token
  Prediction", 2024. [arXiv:2408.15240](https://arxiv.org/abs/2408.15240)
- Gunjal et al., "Rubrics as Rewards: Reinforcement Learning Beyond Verifiable
  Domains", 2025. [arXiv:2507.17746](https://arxiv.org/abs/2507.17746)

#### Exercise idea: judge bias as a reward hack

Simulate a judge whose score is the true reward plus $0.3 \times$ (length) and
plus a position bias of $+0.5$ for whichever response is shown first. Run BoN
and a short REINFORCE loop against it.

**Checks:**

- Selected responses are longer than a gold-reward-optimal policy's.
- Averaging the judge over both orderings removes the position bias exactly
  (assert the swapped-order average equals the unbiased score).

#### Common mistakes

- Using the policy model as its own judge without controls. Self-enhancement
  bias turns into a feedback loop.
- Reporting only judge scores. Always include a held-out gold or human metric.

### 3.5 Process reward models versus outcome reward models

#### Intuition

An outcome reward model (ORM) scores only the final answer. A process reward
model (PRM) scores each reasoning step. PRMs give denser credit assignment and
catch "right answer, wrong reasoning", but step labels are expensive.
Math-Shepherd avoids human labels by estimating a step's quality as the
fraction of Monte Carlo continuations from that step that reach the correct
answer. That is a value estimate, which links PRMs to critics.

#### Key equations

Lightman et al. score a solution as the probability that every step is
correct under the PRM:

$$
\text{score}_{\text{PRM800K}}(y) = \prod_{k=1}^{K} p(\text{step}_k \text{ correct})
$$

Math-Shepherd labels step $s_i$ using $N$ completions
$\{(s_{i+1}^{j}, \dots, a^{j})\}_{j=1}^{N}$ and gold answer $a^\star$ (verified):

$$
y_{s_i}^{\text{HE}} = \begin{cases} 1 & \exists j:\, a^{j} = a^\star \\ 0 & \text{otherwise}\end{cases},
\qquad
y_{s_i}^{\text{SE}} = \frac{1}{N}\sum_{j=1}^{N} \mathbf{1}[a^{j} = a^\star]
$$

The PRM trains with per-step binary cross-entropy,
$\mathcal{L} = -\sum_{i=1}^{K} \big[y_{s_i} \log r_{s_i} + (1 - y_{s_i}) \log(1 - r_{s_i})\big]$,
and scores a solution by the minimum step score.

In GRPO with process supervision (DeepSeekMath), step rewards are normalised
across the group and the advantage of token $t$ is the sum of normalised
rewards of all steps that end at or after $t$ (verified):

$$
\hat{A}_{i,t} = \sum_{\text{index}(j) \ge t} \tilde{r}_i^{\text{index}(j)},
\qquad
\tilde{r}_i^{\text{index}(j)} = \frac{r_i^{\text{index}(j)} - \operatorname{mean}(\mathbf{R})}{\operatorname{std}(\mathbf{R})}
$$

#### Worked example

Step-correctness probabilities $(0.9, 0.8, 0.95)$ give a PRM800K-style score of
$0.9 \times 0.8 \times 0.95 = 0.684$ and a Math-Shepherd-style (min) score of
0.8. If 3 of 8 rollouts from a step reach the gold answer, the soft label is
0.375 and the hard label is 1.

#### Facts to state carefully

- PRM800K contains about 800k step-level human labels on model-generated
  solutions to MATH problems; labels are positive, negative, or neutral.
- The process-supervised RM solved 78% of a representative subset of the MATH
  test set in best-of-$N$ reranking (verified from the abstract).
- DeepSeek-R1 lists PRMs and MCTS as unsuccessful attempts (verified from
  arXiv v1). For PRMs it cites three problems: defining a fine-grained step is
  hard, judging intermediate steps is hard to automate and doesn't scale by
  hand, and a model-based PRM invites reward hacking and needs retraining.

#### References

- Uesato et al., "Solving math word problems with process- and outcome-based
  feedback", 2022. [arXiv:2211.14275](https://arxiv.org/abs/2211.14275)
- Lightman et al., "Let's Verify Step by Step" (PRM800K), 2023.
  [arXiv:2305.20050](https://arxiv.org/abs/2305.20050)
- Wang et al., "Math-Shepherd", 2023.
  [arXiv:2312.08935](https://arxiv.org/abs/2312.08935)
- Kazemnejad et al., "VinePPO: Refining Credit Assignment in RL Training of
  LLMs" (Monte Carlo value estimates instead of a critic), 2024.
  [arXiv:2410.01679](https://arxiv.org/abs/2410.01679)

#### Exercise idea: Math-Shepherd labels are value estimates

Toy task: emit 4 digits (tokens 0-7); reward 1 if the digits sum to 14. Take a
fixed random policy. For every prefix, compute the exact success probability
by enumeration (the true $V^\pi(s_t)$), then estimate soft labels with
$N = 16$ rollouts.

**Checks:**

- The soft label is an unbiased estimator of $V^\pi$: the mean over 200
  repeats is within 3 standard errors of the exact value.
- The hard label estimates $1 - (1 - V)^N$, not $V$; assert this within
  sampling error.
- A PRM-guided greedy search (pick the next digit with the highest estimated
  $V$) achieves a higher success rate than sampling from the policy.

#### Common mistakes

- Treating hard-estimation labels as calibrated probabilities. They're biased
  towards 1 as $N$ grows.
- Forgetting that MC labels depend on the policy that did the rollouts. A PRM
  trained on a weak policy's continuations mislabels steps a stronger policy
  could complete.
- Letting the policy exploit the PRM by producing many short "safe" steps.
  This is why min or product aggregation matters.

---

## 4. RLHF with PPO for language models

### 4.1 The PPO-for-LMs recipe

#### Intuition

Classic RLHF runs PPO in the token MDP with four models in memory: the policy,
a frozen reference (SFT) model, a frozen reward model, and a value model
(critic). The KL to the reference usually enters as a per-token penalty in the
reward, so the critic learns a KL-shaped return. The reward model's score is
added only at the final token.

Many implementation details matter as much as the algorithm. Huang et al.
catalogue more than 20 of them for the TL;DR summarisation task.

#### Key equations

Per-token shaped reward (KL in reward, as in Ziegler et al., Stiennon et al.,
and InstructGPT):

$$
r_t = -\beta \log \frac{\pi_{\text{old}}(y_t \mid s_t)}{\pi_{\text{ref}}(y_t \mid s_t)} + \mathbf{1}[t = T]\; r_\phi(x, y)
$$

Generalised advantage estimation (GAE), typically with $\gamma = 1$ and
$\lambda = 0.95$:

$$
\delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t),
\qquad
\hat{A}_t = \sum_{l=0}^{T-t} (\gamma\lambda)^l\, \delta_{t+l}
$$

PPO clipped surrogate (maximise):

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(\rho_t \hat{A}_t,\; \operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\, \hat{A}_t\big)\Big],
\qquad
\rho_t = \frac{\pi_\theta(y_t \mid s_t)}{\pi_{\text{old}}(y_t \mid s_t)}
$$

Clipped value loss (minimise), with returns $\hat{R}_t = \hat{A}_t + V_{\text{old}}(s_t)$:

$$
L^{V}(\psi) = \tfrac{1}{2}\,\mathbb{E}_t\Big[\max\big((V_\psi(s_t) - \hat{R}_t)^2,\;
(\operatorname{clip}(V_\psi(s_t), V_{\text{old}}(s_t) - \epsilon_v, V_{\text{old}}(s_t) + \epsilon_v) - \hat{R}_t)^2\big)\Big]
$$

Adaptive KL controller (Ziegler et al., 2019), with target KL
$\mathrm{KL}_{\text{target}}$ and $K_\beta = 0.1$:

$$
e_t = \operatorname{clip}\Big(\frac{\mathrm{KL}_t - \mathrm{KL}_{\text{target}}}{\mathrm{KL}_{\text{target}}}, -0.2, 0.2\Big),
\qquad
\beta_{t+1} = \beta_t\,(1 + K_\beta\, e_t)
$$

(The controller form is from memory of Ziegler et al. section 2.2;
**UNVERIFIED** against the PDF during this pass.)

#### Implementation details worth teaching

The following details come from Huang et al. and from common open-source
trainers:

- **Value head.** A linear layer on the final hidden state at each position.
  It is often initialised from the reward model, since both predict a scalar
  about the completion.
- **Reward normalisation.** Normalise RM outputs so that SFT samples have mean
  0 and a fixed scale before RL begins.
- **Advantage whitening.** Standardise $\hat{A}_t$ over the batch (mean 0,
  standard deviation 1). Some implementations don't subtract the mean.
- **EOS handling.** Penalise responses that hit the length limit without an
  EOS token (for example, assign a fixed negative reward).
- **Sampling settings.** Rollouts must use the same temperature as the one used
  to compute log-probs in training. A temperature mismatch silently makes the
  ratios wrong.
- **Dropout off.** Dropout in the policy makes $\pi_\theta$ and $\pi_{\text{old}}$
  disagree even at the same weights.

#### Worked example: which tokens get gradient under clipping

With $\epsilon = 0.2$:

| $\rho_t$ | $\hat{A}_t$ | Unclipped | Clipped | Min | Gradient flows? |
|---|---|---|---|---|---|
| 1.3 | +1 | 1.3 | 1.2 | 1.2 | No (already pushed up enough) |
| 0.7 | +1 | 0.7 | 0.8 | 0.7 | Yes |
| 0.7 | -1 | -0.7 | -0.8 | -0.8 | No (already pushed down enough) |
| 1.3 | -1 | -1.3 | -1.2 | -1.3 | Yes (pessimistic bound) |

The last row matters for LMs: a token whose probability rose despite a
negative advantage still gets full, unbounded gradient. This is the motivation
for dual-clip PPO and for several 2025 variants.

#### References

- Schulman et al., "Proximal Policy Optimization Algorithms", 2017.
  [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Ziegler et al., 2019. [arXiv:1909.08593](https://arxiv.org/abs/1909.08593)
- Stiennon et al., 2020. [arXiv:2009.01325](https://arxiv.org/abs/2009.01325)
- Huang et al., "The N+ Implementation Details of RLHF with PPO: A Case Study
  on TL;DR Summarization", 2024.
  [arXiv:2403.17031](https://arxiv.org/abs/2403.17031)
- Sheng et al., "HybridFlow: A Flexible and Efficient RLHF Framework" (verl),
  2024. [arXiv:2409.19256](https://arxiv.org/abs/2409.19256)

#### Exercise idea: PPO on a verifiable toy task

Task: emit 4 tokens from $\{0, \dots, 7\}$; reward 1 if they sum to a target
(14), else 0. Policy and critic: a 2-layer MLP over a one-hot encoding of
(position, running sum). Implement per-token KL shaping, GAE, clipping, and
advantage whitening.

**Checks:**

- `gae(rewards, values, gamma=1, lam=1)` equals Monte Carlo return minus value,
  to $10^{-6}$.
- `gae(..., lam=0)` equals one-step TD errors.
- The PPO loss gradient is exactly zero for a token with $\rho = 1.3$ and
  $\hat{A} = +1$ (use `torch.autograd.grad`).
- Success rate goes from about 0.08 (uniform policy; the exact value is
  computable by enumeration) to more than 0.6 in under 20 seconds.
- The sum of shaped rewards equals $r - \beta \log \frac{\pi(y)}{\pi_{\text{ref}}(y)}$
  for every sampled sequence.

#### Common mistakes

- Computing $\pi_{\text{old}}$ log-probs with a different forward pass setup
  (padding, dropout, temperature) than the one used for $\pi_\theta$. The ratio
  at the first minibatch then isn't 1.
- Off-by-one errors between logits and labels. The logit at position $t$
  predicts token $t+1$.
- Averaging the loss over padded positions.
- Letting the value head share the policy's trunk with a large value-loss
  coefficient. The value loss then dominates representation learning.

### 4.2 KL estimators: k1, k2, k3, and "KL in reward" versus "KL in loss"

#### Intuition

You never compute the exact per-token KL over the full vocabulary in most
trainers; you estimate it from the sampled token. Schulman's note compares
three single-sample estimators of $\mathrm{KL}(q \,\|\, p)$ using samples
$x \sim q$ and the ratio $r = p(x)/q(x)$. For RLHF, $q = \pi_\theta$ and
$p = \pi_{\text{ref}}$.

A second, subtler question is what happens when you backpropagate through
these estimators. An estimator can be unbiased for the KL value and still give
the gradient of a different quantity when used as a loss term, because
autodiff doesn't differentiate through the sampling distribution. Several
2025 papers (REINFORCE++, Zhang et al., "A Comedy of Estimators", and
DeepSeek-V3.2) sort this out.

#### Key equations

With $x \sim q$ and $r = p(x)/q(x)$:

$$
k_1 = -\log r, \qquad k_2 = \tfrac{1}{2}(\log r)^2, \qquad k_3 = (r - 1) - \log r
$$

- $k_1$ is unbiased but high variance, and it can be negative.
- $k_2$ is biased but low variance. Its expectation matches KL to second order
  when $p \approx q$ (it is an $f$-divergence with the same second-order
  behaviour).
- $k_3$ is unbiased (because $\mathbb{E}_q[r - 1] = 0$), always non-negative
  (because $\log r \le r - 1$), and usually low variance.

**Gradients when used as a loss** (samples treated as constants, gradient only
through $\log q_\theta(x)$):

$$
\begin{aligned}
\mathbb{E}_{x \sim q_\theta}\big[\nabla_\theta k_1\big] &= \mathbb{E}_{q_\theta}\big[\nabla_\theta \log q_\theta(x)\big] = 0 \\
\mathbb{E}_{x \sim q_\theta}\big[\nabla_\theta k_2\big] &= \mathbb{E}_{q_\theta}\Big[\log \tfrac{q_\theta(x)}{p(x)}\, \nabla_\theta \log q_\theta(x)\Big] = \nabla_\theta \mathrm{KL}(q_\theta \,\|\, p) \\
\mathbb{E}_{x \sim q_\theta}\big[\nabla_\theta k_3\big] &= \mathbb{E}_{q_\theta}\Big[\Big(1 - \tfrac{p(x)}{q_\theta(x)}\Big) \nabla_\theta \log q_\theta(x)\Big] = -\sum_x p(x) \nabla_\theta \log q_\theta(x) = \nabla_\theta \mathrm{KL}(p \,\|\, q_\theta)
\end{aligned}
$$

So "$k_3$ in the loss", as in GRPO, follows the gradient of the forward KL, not
the reverse KL of the stated objective. "$k_2$ in the loss" gives the correct
reverse-KL gradient. "$k_1$ in the reward" (fed through the policy-gradient
score function, as in classic PPO RLHF) also gives an unbiased reverse-KL
gradient.

**Off-policy correction (DeepSeek-V3.2, verified).** When tokens come from
$\pi_{\text{old}}$ rather than $\pi_\theta$, DeepSeek-V3.2 multiplies $k_3$ by
the importance ratio:

$$
\mathbb{D}_{\text{KL}}\big(\pi_\theta(o_{i,t}) \,\|\, \pi_{\text{ref}}(o_{i,t})\big)
= \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\text{old}}(o_{i,t} \mid q, o_{i,<t})}
\Big(\frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}
- \log \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - 1\Big)
$$

Working the expectation through shows that
$\mathbb{E}_{\pi_{\text{old}}}\big[\nabla_\theta \tfrac{\pi_\theta}{\pi_{\text{old}}} k_3\big] = \sum_a \nabla_\theta \pi_\theta(a) \log \tfrac{\pi_\theta(a)}{\pi_{\text{ref}}(a)} = \nabla_\theta \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$,
so the corrected estimator's gradient is the unbiased reverse-KL gradient. The
paper also notes that for maths, a weak KL penalty or none at all can work
better.

#### Worked example (computed)

Let $q = (0.5, 0.3, 0.2)$ and $p = (0.3, 0.3, 0.4)$. Then
$\mathrm{KL}(q \,\|\, p) = 0.1168$ and:

| Estimator | $\mathbb{E}_q[k]$ | $\operatorname{Var}_q[k]$ |
|---|---|---|
| $k_1$ | 0.1168 | 0.2129 |
| $k_2$ | 0.1133 (biased) | not computed |
| $k_3$ | 0.1168 | 0.0113 |

The variance of $k_3$ is about 19 times smaller than the variance of $k_1$.

The gradient identities were prototyped in PyTorch on a 3-way softmax: the
exact expected gradient of $k_1$ as a loss was 0 (to $10^{-7}$), of $k_2$
matched $\nabla \mathrm{KL}(q \,\|\, p)$, and of $k_3$ matched
$\nabla \mathrm{KL}(p \,\|\, q)$.

#### References

- Schulman, "Approximating KL Divergence" (blog), 2020.
  [joschu.net/blog/kl-approx.html](http://joschu.net/blog/kl-approx.html)
- Hu et al., "REINFORCE++", 2025 (argues for $k_2$ in the loss).
  [arXiv:2501.03262](https://arxiv.org/abs/2501.03262)
- Zhang et al., "On the Design of KL-Regularized Policy Gradient Algorithms
  for LLM Reasoning", 2025.
  [arXiv:2505.17508](https://arxiv.org/abs/2505.17508)
- Shah et al., "A Comedy of Estimators: On KL Regularization in RL Training of
  LLMs", 2025. [arXiv:2512.21852](https://arxiv.org/abs/2512.21852)
- DeepSeek-AI, "DeepSeek-V3.2", 2025, section 3.1.
  [arXiv:2512.02556](https://arxiv.org/abs/2512.02556)

#### Exercise idea: estimator bias, variance, and gradients

Use a random 8-way categorical for $q_\theta$ (learnable logits) and a fixed
$p$.

**Checks:**

- With $10^6$ samples, $|\bar{k}_1 - \mathrm{KL}| < 4\,\text{SE}$ and
  $|\bar{k}_3 - \mathrm{KL}| < 4\,\text{SE}$.
- $k_3 \ge 0$ for every sample; $k_1 < 0$ for some sample.
- $\operatorname{Var}[k_3] < \operatorname{Var}[k_1]$ when $p$ and $q$ are close
  (for example, logits differing by $\mathcal{N}(0, 0.1^2)$).
- Compute exact expected gradients by summing over the 8 outcomes (weights
  $q_\theta(x)$ detached). Assert $k_1 \to 0$, $k_2 \to \nabla\mathrm{KL}(q\|p)$,
  and $k_3 \to \nabla\mathrm{KL}(p\|q)$ with `allclose(atol=1e-6)`.
- Off-policy: sample from a fixed $q_{\text{old}}$, use the V3.2-corrected
  estimator, and assert its exact expected gradient equals
  $\nabla\mathrm{KL}(q_\theta \,\|\, p)$.

#### Common mistakes

- Assuming "unbiased value" implies "correct gradient". It doesn't.
- Using $k_3$ when $\pi_\theta \ll \pi_{\text{ref}}$ on sampled tokens. The
  term $p/q$ explodes and dominates the gradient (DeepSeek-V3.2 calls out this
  failure).
- Mixing conventions: some code defines the ratio as
  $\pi_\theta/\pi_{\text{ref}}$ and others as $\pi_{\text{ref}}/\pi_\theta$.
  Check the sign of $k_1$ in a unit test.
- Counting the KL twice: once in the reward and once in the loss.

---

## 5. Best-of-n sampling and its KL

### Intuition

Best-of-$n$ (BoN) draws $n$ samples from $\pi_{\text{ref}}$ and returns the one
with the highest reward. It is a strong, training-free baseline and the
standard way to measure how much a reward model can be exploited. Its
distribution has a closed form, which makes it an ideal exercise.

### Key equations

For a continuous reward with CDF $F$ under $\pi_{\text{ref}}$ (no ties), the
BoN density is

$$
\pi_{\text{BoN}}(y) = n\, F(r(y))^{n-1}\, \pi_{\text{ref}}(y)
$$

and its KL is

$$
\mathrm{KL}(\pi_{\text{BoN}} \,\|\, \pi_{\text{ref}}) = \log n + (n-1)\,\mathbb{E}_{U \sim \text{Beta}(n,1)}[\log U]
= \log n - \frac{n-1}{n}
$$

For a discrete space with outcomes sorted by reward, $y_{(1)} < \dots < y_{(M)}$,
and cumulative probability $F_i = \sum_{j \le i} \pi_{\text{ref}}(y_{(j)})$:

$$
\pi_{\text{BoN}}(y_{(i)}) = F_i^{\,n} - F_{i-1}^{\,n}
$$

Beirami et al. showed that $\log n - \frac{n-1}{n}$ is an upper bound on the
actual KL, not an identity, and that the win rate of BoN over the reference is
at most $\frac{n}{n+1}$ (verified from the abstract). The bound is tight when
the reference distribution spreads over many outcomes, each of small
probability.

### Worked example (computed)

| $n$ | $\log n - (n-1)/n$ |
|---|---|
| 2 | 0.193 |
| 4 | 0.636 |
| 8 | 1.204 |
| 16 | 1.835 |
| 64 | 3.175 |

For $n = 4$ and a uniform reference over $M$ outcomes with distinct rewards,
the exact KL is 0.590 for $M = 4$, 0.633 for $M = 16$, and 0.63629 for
$M = 1000$ against the bound of 0.63629.

### References

- Stiennon et al., 2020, Appendix G.3 (the analytic BoN KL).
  [arXiv:2009.01325](https://arxiv.org/abs/2009.01325)
- Beirami et al., "Theoretical guarantees on the best-of-n alignment policy",
  2024. [arXiv:2401.01879](https://arxiv.org/abs/2401.01879)
- Gao et al., 2022. [arXiv:2210.10760](https://arxiv.org/abs/2210.10760)

### Exercise idea: BoN distribution and KL bound

**Checks:**

- The exact discrete BoN distribution sums to 1 and matches a Monte Carlo
  histogram of $10^5$ BoN draws (total variation distance less than 0.01).
- $\mathrm{KL}_{\text{exact}}(n) \le \log n - \frac{n-1}{n}$ for all
  $n \in \{1, \dots, 64\}$ and random references.
- With a uniform reference over $M = 10^4$ outcomes, the gap to the bound is
  below $10^{-3}$.
- With ties (a reference that puts mass 0.5 on one outcome), the gap is large.
  This shows why the bound is loose for peaked distributions.
- The win rate against a fresh reference sample is at most $n/(n+1)$, with
  near-equality for the large-$M$ uniform case.
- Compare BoN to the KL-regularised optimum $\pi^\star$ at matched KL: find
  $\beta$ by bisection so that $\mathrm{KL}(\pi^\star) = \mathrm{KL}(\pi_{\text{BoN}})$,
  then assert $\mathbb{E}_{\pi^\star}[r] \ge \mathbb{E}_{\pi_{\text{BoN}}}[r]$.
  $\pi^\star$ is optimal for that KL budget, so it can't do worse.

### Common mistakes

- Quoting $\log n - (n-1)/n$ as the exact KL for LLM outputs with many
  duplicate completions.
- Forgetting BoN costs $n\times$ inference compute. It trades training compute
  for inference compute.

---

## 6. Direct alignment algorithms

### 6.1 DPO: full derivation

#### Intuition

RLHF fits a reward model and then runs RL against it. DPO notices that, for the
KL-regularised objective, the optimal policy and the reward determine each
other. So you can reparameterise the reward in terms of the policy and fit the
Bradley-Terry likelihood directly. The partition function $Z(x)$ cancels because
BT depends only on reward differences within a prompt. The result is a
classification loss on preference pairs with no sampling, no reward model, and
no critic.

#### Derivation

1. Start from the KL-regularised objective and its optimum (section 2):
   $\pi^\star(y \mid x) = \pi_{\text{ref}}(y \mid x)\exp(r(x,y)/\beta)/Z(x)$.
2. Take logs and solve for the reward:
   $$
   r(x,y) = \beta \log \frac{\pi^\star(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
   $$
3. Substitute into Bradley-Terry. The $\beta \log Z(x)$ terms cancel:
   $$
   p^\star(y_w \succ y_l \mid x) = \sigma\Big(\beta \log \frac{\pi^\star(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^\star(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\Big)
   $$
4. Replace $\pi^\star$ by $\pi_\theta$ and maximise the likelihood of the data:
   $$
   \mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim D}\Big[\log \sigma\Big(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\Big)\Big]
   $$

Define the implicit reward $\hat{r}_\theta(x,y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$.
The gradient is

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta\, \mathbb{E}\Big[\sigma\big(\hat{r}_\theta(x,y_l) - \hat{r}_\theta(x,y_w)\big)
\big(\nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x)\big)\Big]
$$

The weight $\sigma(\hat{r}_l - \hat{r}_w)$ is large when the implicit reward
ranks the pair wrongly.

**What DPO guarantees.** Every reward $r$ is equivalent (same preferences, same
optimal policy) to $\beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$
for some $\pi$. So at the population optimum with full coverage, the implicit
reward equals the BT reward up to a per-prompt constant. With finite data,
responses outside the dataset's support are unconstrained.

#### Worked example (computed)

Let the sequence log-probs be $\log \pi_\theta(y_w) = -10$,
$\log \pi_{\text{ref}}(y_w) = -11$, $\log \pi_\theta(y_l) = -12$,
$\log \pi_{\text{ref}}(y_l) = -11.5$, with $\beta = 0.1$:

- Log-ratio margin $h = (-10 + 11) - (-12 + 11.5) = 1 + 0.5 = 1.5$.
- Implicit rewards: $\hat{r}_w = 0.1$, $\hat{r}_l = -0.05$.
- Loss $= -\log \sigma(0.15) = 0.621$.
- Gradient weight $\sigma(-0.15) = 0.463$ (compared with 0.5 at initialisation).

#### References

- Rafailov et al., "Direct Preference Optimization: Your Language Model is
  Secretly a Reward Model", 2023.
  [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- Rafailov et al., "From $r$ to $Q^*$", 2024 (token-level interpretation).
  [arXiv:2404.12358](https://arxiv.org/abs/2404.12358)

#### Exercise idea: DPO recovers the BT reward

Setup: a tabular policy over 16 responses (one prompt), $\pi_{\text{ref}}$
random, hidden reward $r^\star$. Generate the *expected* DPO loss exactly:
for every ordered pair $(i, j)$, weight
$\pi_{\text{ref}}(i)\pi_{\text{ref}}(j)\,\sigma(r^\star_i - r^\star_j)$.
Minimise it with L-BFGS.

**Checks:**

- `std(beta * log(pi/pi_ref) - r_star) < 1e-3`: the implicit reward matches the
  true reward up to a constant.
- `allclose(pi_theta, pi_ref * exp(r_star / beta) / Z, atol=1e-4)`.
- Finite-data variant: 500 sampled pairs. Assert that responses never seen in
  any pair keep a log-ratio driven only by the softmax normalisation, and that
  the recovered reward's correlation with $r^\star$ on covered responses
  exceeds 0.9.
- Autoregressive variant: a bigram policy over length-3 sequences. Assert
  sequence log-probs are sums of token log-probs and that the implicit reward
  is additive over tokens when $r^\star$ is.

#### Common mistakes

- Using average (length-normalised) log-probs in standard DPO. That's SimPO,
  not DPO, and changes the objective.
- Forgetting to freeze $\pi_{\text{ref}}$, or precomputing its log-probs with a
  different tokenisation or chat template.
- Summing log-probs over prompt tokens as well as response tokens.
- Reading "chosen reward goes up" as success. Both log-probs often go down
  (see section 6.6).

### 6.2 IPO

#### Intuition

With deterministic preferences ($p^\star = 1$), BT needs an infinite reward gap,
so DPO pushes $\pi(y_l) \to 0$ regardless of $\beta$: the KL regulariser stops
working. Azar et al. generalise to $\Psi$-preference optimisation and choose
$\Psi$ as the identity. The resulting IPO loss regresses the log-ratio margin
to a finite target, so it can't run away.

#### Key equations (verified)

$$
\max_\pi\; \mathbb{E}_{x,\, y \sim \pi,\, y' \sim \mu}\big[\Psi\big(p^\star(y \succ y' \mid x)\big)\big] - \tau\, \mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})
$$

$$
h_\pi(y_w, y_l, x) = \log \frac{\pi(y_w \mid x)\, \pi_{\text{ref}}(y_l \mid x)}{\pi(y_l \mid x)\, \pi_{\text{ref}}(y_w \mid x)},
\qquad
\mathcal{L}_{\text{IPO}} = \mathbb{E}\Big[\Big(h_\pi(y_w, y_l, x) - \frac{1}{2\tau}\Big)^2\Big]
$$

The paper writes the sampled loss with an extra factor $\tfrac{1}{2}$ and
$\tau^{-1}/2$ inside the square; the constant factor doesn't change the
minimiser.

#### Worked example

With the DPO example's $h = 1.5$ and $\tau = 0.1$, the target margin is 5 and
the loss is $(1.5 - 5)^2 = 12.25$. Once $h$ reaches 5, the gradient reverses
instead of pushing further.

#### Exercise check

With deterministic labels on 8 responses, train DPO and IPO for 2000 steps.
Assert that DPO's margin keeps growing (the last-100-step slope is positive)
while IPO's margin converges to $1/(2\tau)$ within 0.05.

#### Reference

- Azar et al., "A General Theoretical Paradigm to Understand Learning from
  Human Preferences", 2023. [arXiv:2310.12036](https://arxiv.org/abs/2310.12036)

### 6.3 KTO

#### Intuition

KTO uses unpaired binary feedback ("good" or "bad" per response), which is
cheaper to collect than pairs. It borrows prospect theory: value is measured
relative to a reference point $z_0$, with different weights for gains and
losses.

#### Key equations (verified)

$$
r_\theta(x,y) = \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)},
\qquad
z_0 = \mathrm{KL}\big(\pi_\theta(y' \mid x) \,\|\, \pi_{\text{ref}}(y' \mid x)\big)
$$

$$
v(x,y) = \begin{cases}
\lambda_D\, \sigma\big(\beta(r_\theta(x,y) - z_0)\big) & \text{if } y \text{ is desirable} \\
\lambda_U\, \sigma\big(\beta(z_0 - r_\theta(x,y))\big) & \text{if } y \text{ is undesirable}
\end{cases}
\qquad
\mathcal{L}_{\text{KTO}} = \mathbb{E}_{x,y}\big[\lambda_y - v(x,y)\big]
$$

In practice, $z_0$ is estimated per microbatch of size $m$ from mismatched
pairs, clamped at 0, with no gradient:
$\hat{z}_0 = \max\big(0, \frac{1}{m}\sum_i \log \frac{\pi_\theta(y_j \mid x_i)}{\pi_{\text{ref}}(y_j \mid x_i)}\big)$
with $j = (i \bmod m) + 1$. The paper recommends
$\frac{\lambda_D n_D}{\lambda_U n_U} \in [1, \tfrac{3}{2}]$ and $\beta = 0.1$ by
default.

#### Worked example

With $\beta = 0.1$, $\lambda_D = \lambda_U = 1$, $z_0 = 0.2$: a desirable
response with $r_\theta = 1$ has loss $1 - \sigma(0.08) = 0.480$; an
undesirable one with $r_\theta = -0.5$ has loss $1 - \sigma(0.07) = 0.483$.

#### Reference

- Ethayarajh et al., "KTO: Model Alignment as Prospect Theoretic
  Optimization", 2024. [arXiv:2402.01306](https://arxiv.org/abs/2402.01306)

### 6.4 ORPO and SimPO (reference-free)

#### Intuition

Both drop $\pi_{\text{ref}}$ and use length-normalised log-probs. ORPO folds
preference learning into SFT: it adds an odds-ratio penalty to the NLL loss so
you skip the separate SFT stage. SimPO argues DPO's implicit reward doesn't
match the quantity that governs generation (average log-likelihood), and
adds a target margin $\gamma$.

#### Key equations (verified)

ORPO, with $\log P_\theta(y \mid x) = \frac{1}{m}\sum_{t=1}^{m} \log P_\theta(y_t \mid x, y_{<t})$:

$$
\text{odds}_\theta(y \mid x) = \frac{P_\theta(y \mid x)}{1 - P_\theta(y \mid x)},
\qquad
\mathcal{L}_{\text{OR}} = -\log \sigma\Big(\log \frac{\text{odds}_\theta(y_w \mid x)}{\text{odds}_\theta(y_l \mid x)}\Big),
\qquad
\mathcal{L}_{\text{ORPO}} = \mathbb{E}\big[\mathcal{L}_{\text{SFT}} + \lambda\, \mathcal{L}_{\text{OR}}\big]
$$

SimPO:

$$
\mathcal{L}_{\text{SimPO}} = -\mathbb{E}\Big[\log \sigma\Big(\frac{\beta}{\lvert y_w \rvert}\log \pi_\theta(y_w \mid x) - \frac{\beta}{\lvert y_l \rvert}\log \pi_\theta(y_l \mid x) - \gamma\Big)\Big]
$$

The SimPO authors report good results with $\beta \in [2.0, 2.5]$ and
$\gamma \in [0.5, 1.5]$. They also report that, under DPO training, only about
half of training triples satisfy the likelihood ranking
$p_\theta(y_w) > p_\theta(y_l)$ even when the implicit reward ranking holds.

#### Worked example (computed)

Let $\log \pi(y_w) = -10$ over 10 tokens (average $-1$) and
$\log \pi(y_l) = -12$ over 8 tokens (average $-1.5$).

- SimPO with $\beta = 2$, $\gamma = 1$: argument $= 2(-1) - 2(-1.5) - 1 = 0$,
  so the loss is $\log 2 = 0.693$.
- ORPO: $P_w = e^{-1} = 0.368$, $P_l = e^{-1.5} = 0.223$, odds $0.582$ and
  $0.287$, log odds ratio $0.706$, so $\mathcal{L}_{\text{OR}} = 0.401$.

#### References

- Hong, Lee, and Thorne, "ORPO: Monolithic Preference Optimization without
  Reference Model", 2024. [arXiv:2403.07691](https://arxiv.org/abs/2403.07691)
- Meng, Xia, and Chen, "SimPO: Simple Preference Optimization with a
  Reference-Free Reward", 2024. [arXiv:2405.14734](https://arxiv.org/abs/2405.14734)

### 6.5 Conservative and robust DPO

#### Intuition

Preference labels are noisy. If a fraction $\varepsilon$ of labels are flipped,
you can either smooth the target (cDPO, label smoothing) or build an unbiased
estimator of the clean loss (rDPO).

#### Key equations (verified from Chowdhury et al.)

Let $\mathcal{L}(\theta; y_w, y_l) = -\log \sigma(\beta h_\theta(y_w, y_l))$.

$$
\bar{\mathcal{L}}_\varepsilon^{\text{cDPO}} = (1-\varepsilon)\,\mathcal{L}(\theta; \tilde{y}_w, \tilde{y}_l) + \varepsilon\, \mathcal{L}(\theta; \tilde{y}_l, \tilde{y}_w)
$$

$$
\hat{\mathcal{L}}_\varepsilon^{\text{rDPO}} = \frac{(1-\varepsilon)\,\mathcal{L}(\theta; \tilde{y}_w, \tilde{y}_l) - \varepsilon\, \mathcal{L}(\theta; \tilde{y}_l, \tilde{y}_w)}{1 - 2\varepsilon}
$$

rDPO satisfies $\mathbb{E}_\varepsilon[\hat{\mathcal{L}}_\varepsilon \mid y_w, y_l] = \mathcal{L}(\theta; y_w, y_l)$.
cDPO's optimum stops at a finite margin $\beta h = \log\frac{1-\varepsilon}{\varepsilon}$.

#### Worked example

With $\beta h = 0.15$ and $\varepsilon = 0.1$: cDPO loss $= 0.636$, rDPO loss
$= 0.602$.

#### Exercise check

Flip 20% of labels in the DPO recovery exercise. Assert that rDPO's expected
loss (over the flip noise) equals the clean DPO loss exactly for every pair, and
that cDPO's optimal margin equals $\log(0.8/0.2)$ within $10^{-3}$.

#### References

- Mitchell, "A note on DPO with noisy preferences and relationship to IPO",
  2023 (the cDPO note; not on arXiv).
  [ericmitchell.ai/cdpo.pdf](https://ericmitchell.ai/cdpo.pdf) (**UNVERIFIED**
  URL availability.)
- Chowdhury, Kini, and Natarajan, "Provably Robust DPO", 2024.
  [arXiv:2403.00409](https://arxiv.org/abs/2403.00409)

### 6.6 Online and iterative DPO, and DPO failure modes

#### Intuition

DPO on a fixed dataset is offline: the pairs come from other models, so the
policy learns about responses it would never generate. Online or iterative DPO
samples pairs from the current policy, labels them with an RM or an LLM judge,
and repeats. This closes much of the gap to PPO-style RLHF.

Two failure modes recur:

- **Likelihood displacement.** DPO's loss depends only on the margin, so it can
  decrease while both $\log \pi(y_w)$ and $\log \pi(y_l)$ fall. Probability
  mass moves to responses outside the pair, sometimes to harmful ones. Razin et
  al. tie this to chosen and rejected responses with similar hidden embeddings
  (their CHES score) and show that filtering high-CHES pairs mitigates it. In
  their refusal experiment, DPO reduced Llama-3-8B-Instruct's refusal rate from
  74.4% to 33.4%.
- **Length exploitation.** Preferred responses in many datasets are longer, and
  DPO's summed log-ratio lets length act as a shortcut. R-DPO adds a length
  regulariser; SimPO length-normalises.

#### Key equations

DPO-Positive (Smaug) adds a penalty when the chosen log-prob drops below the
reference (form from memory; **UNVERIFIED** exact placement of $\lambda$):

$$
\mathcal{L}_{\text{DPOP}} = -\log \sigma\Big(\beta\Big[\log \tfrac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \log \tfrac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)} - \lambda \max\Big(0, \log \tfrac{\pi_{\text{ref}}(y_w)}{\pi_\theta(y_w)}\Big)\Big]\Big)
$$

R-DPO (Park et al.) subtracts $\alpha(\lvert y_w \rvert - \lvert y_l \rvert)$
inside the sigmoid (**UNVERIFIED** exact sign convention; check the paper
before implementing).

#### References

- Guo et al., "Direct Language Model Alignment from Online AI Feedback", 2024.
  [arXiv:2402.04792](https://arxiv.org/abs/2402.04792)
- Xiong et al., "Iterative Preference Learning from Human Feedback", 2023.
  [arXiv:2312.11456](https://arxiv.org/abs/2312.11456)
- Razin et al., "Unintentional Unalignment: Likelihood Displacement in Direct
  Preference Optimization", ICLR 2025.
  [arXiv:2410.08847](https://arxiv.org/abs/2410.08847)
- Pal et al., "Smaug: Fixing Failure Modes of Preference Optimisation with
  DPO-Positive", 2024. [arXiv:2402.13228](https://arxiv.org/abs/2402.13228)
- Park et al., "Disentangling Length from Quality in Direct Preference
  Optimization", 2024. [arXiv:2403.19159](https://arxiv.org/abs/2403.19159)

#### Exercise idea: reproduce likelihood displacement

Use a softmax policy over 32 responses with a shared low-rank embedding
(logits $= W e_y$ with $e_y \in \mathbb{R}^4$), so responses interact through
shared parameters. Make $y_w$ and $y_l$ have cosine similarity 0.95.

**Checks:**

- After 500 DPO steps, the DPO loss has decreased, yet both
  $\log \pi(y_w)$ and $\log \pi(y_l)$ are below their initial values.
- The probability mass that left the pair went mostly to a third response whose
  embedding aligns with $e_w - e_l$ (assert it is the largest gainer).
- With orthogonal embeddings, $\log \pi(y_w)$ increases.
- Online DPO variant: sample pairs from the current policy and label them with
  $r^\star$. Assert that online DPO reaches a higher $\mathbb{E}_\pi[r^\star]$
  than offline DPO on a fixed dataset from a different policy, at equal numbers
  of labels.
- Length variant: let $r^\star$ ignore length but make the labeller prefer the
  longer response with probability 0.7 on ties. Assert DPO increases the mean
  length and SimPO increases it less.

#### Common mistakes

- Monitoring only the loss and the reward margin.
- Running many epochs of offline DPO. Overfitting and displacement both grow
  with epochs.
- Comparing DPO and PPO at different amounts of preference data or compute.

---

## 7. REINFORCE-style methods: RLOO, ReMax, and GRPO

### 7.1 Why drop the critic

#### Intuition

PPO's critic is a second LLM-sized network that is hard to train with sparse,
terminal rewards. With deterministic transitions and a terminal reward, the
sequence-level bandit view is natural, and REINFORCE with a good baseline has
acceptable variance. Ahmadian et al. ("Back to Basics") argued that PPO's
machinery (clipping, GAE, critic) is less necessary for LLM fine-tuning because
the pretrained policy starts close to good behaviour. The group-baseline family
(RLOO, GRPO) samples several responses per prompt and uses the others as the
baseline.

#### Key equations

REINFORCE with baseline $b(x)$:

$$
\nabla_\theta J = \mathbb{E}_{x,\, y \sim \pi_\theta}\big[(r(x,y) - b(x))\, \nabla_\theta \log \pi_\theta(y \mid x)\big]
$$

It stays unbiased for any $b$ that doesn't depend on $y$.

**RLOO** (REINFORCE leave-one-out), with $k$ samples per prompt:

$$
\hat{A}_i = r(x, y_i) - \frac{1}{k-1}\sum_{j \ne i} r(x, y_j)
= \frac{k}{k-1}\Big(r(x,y_i) - \frac{1}{k}\sum_{j=1}^{k} r(x,y_j)\Big)
$$

The leave-one-out baseline is independent of $y_i$, so the estimator is
unbiased.

**ReMax** uses the reward of the greedy decode as the baseline:
$\hat{A}_i = r(x, y_i) - r(x, \bar{y})$ with $\bar{y} = \arg\max$ decoding. It
needs one extra generation per prompt, but no critic.

### 7.2 GRPO: the exact objectives

#### Intuition

GRPO (DeepSeekMath) keeps PPO's clipped ratio and replaces the critic with a
group-normalised advantage: sample $G$ outputs per question, standardise their
rewards within the group, and assign the result to every token of each output.
It moves the KL to the reference out of the reward and into the loss, using the
$k_3$ estimator.

#### Key equations (verified against both papers)

DeepSeekMath (token-level ratios, per-sequence length normalisation):

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim P(Q),\, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot \mid q)}
\Bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{\lvert o_i \rvert}\sum_{t=1}^{\lvert o_i \rvert}
\Big\{\min\Big[\rho_{i,t}\hat{A}_{i,t},\; \operatorname{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon)\,\hat{A}_{i,t}\Big]
- \beta\, \mathbb{D}_{\text{KL}}\big[\pi_\theta \,\|\, \pi_{\text{ref}}\big]\Big\}\Bigg]
$$

$$
\rho_{i,t} = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})},
\qquad
\mathbb{D}_{\text{KL}}\big[\pi_\theta \,\|\, \pi_{\text{ref}}\big] = \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - \log \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - 1
$$

Outcome supervision:

$$
\hat{A}_{i,t} = \tilde{r}_i = \frac{r_i - \operatorname{mean}(\mathbf{r})}{\operatorname{std}(\mathbf{r})}, \qquad \mathbf{r} = \{r_1, \dots, r_G\}
$$

DeepSeekMath hyperparameters: $G = 64$, $\beta = 0.04$, policy learning rate
$10^{-6}$, maximum length 1024, batch size 1024.

DeepSeek-R1 writes GRPO with sequence-level ratios and no $1/\lvert o_i \rvert$:

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\Big(\min\Big(\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)} A_i,\; \operatorname{clip}\Big(\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)}, 1-\varepsilon, 1+\varepsilon\Big) A_i\Big) - \beta\, \mathbb{D}_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})\Big)\Bigg]
$$

with $A_i = \frac{r_i - \operatorname{mean}(\{r_1, \dots, r_G\})}{\operatorname{std}(\{r_1, \dots, r_G\})}$ and the
same $k_3$-form KL. Treat the R1 form as notation. Open-source GRPO trainers
implement the DeepSeekMath token-level form, and the R1 paper doesn't state
that it changed the implementation. Point this discrepancy out to learners.

DeepSeekMath's "unified paradigm" writes SFT, rejection-sampling fine-tuning
(RFT), DPO, PPO, and GRPO as
$\nabla_\theta J = \mathbb{E}_{(q,o) \sim D}\big[\frac{1}{\lvert o \rvert}\sum_t GC(q, o, t)\, \nabla_\theta \log \pi_\theta(o_t \mid q, o_{<t})\big]$,
differing in the data source $D$ and the gradient coefficient $GC$. It also
reports that RL improved Maj@K but not Pass@K, an early data point for the
debate in section 9.3.

#### Worked example (computed)

A group of $G = 8$ with rewards $(1, 0, 0, 1, 1, 0, 0, 0)$ has mean 0.375.

- Population standard deviation 0.484: $A = +1.291$ (correct), $-0.775$
  (incorrect).
- Bessel-corrected standard deviation 0.518 (PyTorch's default `std`):
  $A = +1.208$ and $-0.725$.
- RLOO with $k = 4$ on $(1, 0, 0, 1)$: $A = +0.667, -0.667, -0.667, +0.667$.

If all 8 rewards are equal, the numerator is 0 and the standard deviation is 0.
Implementations add a small $\epsilon$ to the denominator, so the group
contributes no policy gradient (only the KL term).

#### References

- Williams, "Simple statistical gradient-following algorithms for
  connectionist reinforcement learning", *Machine Learning*, 1992 (REINFORCE).
- Ahmadian et al., "Back to Basics: Revisiting REINFORCE Style Optimization for
  Learning from Human Feedback in LLMs", 2024.
  [arXiv:2402.14740](https://arxiv.org/abs/2402.14740)
- Li et al., "ReMax", 2023. [arXiv:2310.10505](https://arxiv.org/abs/2310.10505)
- Shao et al., "DeepSeekMath" (GRPO), 2024.
  [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- DeepSeek-AI, "DeepSeek-R1", 2025. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- Zhou et al., "Demystifying Group Relative Policy Optimization: Its Policy
  Gradient is a U-Statistic", 2026 (MSE and finite-sample analysis of the GRPO
  gradient). [arXiv:2603.01162](https://arxiv.org/abs/2603.01162)

#### Exercise idea: implement GRPO advantages and loss

Toy task: 4 tokens from 8, reward 1 if the sum equals a per-prompt target
(prompts are targets 10-18). The policy is a 2-layer MLP over one-hot
(target, position, running sum).

**Checks:**

- `grpo_advantages([1,0,0,1,1,0,0,0])` matches the hand-computed values for
  both the population and the Bessel conventions (parameterise the function).
- `rloo_advantages(r)` equals `k/(k-1) * (r - r.mean())`.
- Unbiasedness: for a tabular sequence policy with 64 responses, the exact
  expectation (enumerate all size-$k$ groups, or use $10^5$ Monte Carlo groups)
  of the RLOO gradient equals the true gradient $\nabla \mathbb{E}[r]$ within
  3 SE. The mean-centred GRPO gradient (no std) equals $\frac{G-1}{G}$ times
  the RLOO gradient. With std normalisation it is a different, biased
  estimator; assert only a positive cosine similarity with the true gradient
  on a single prompt, and report the value.
- A degenerate group (all rewards equal) yields exactly zero policy-gradient
  contribution.
- On-policy with one gradient step per batch, $\rho = 1$ everywhere and the
  clipped loss's gradient equals the REINFORCE gradient.
- Training raises the success rate from about 0.08 to more than 0.7 in under
  30 seconds.

#### Common mistakes

- Dividing by a zero standard deviation.
- Using a baseline that includes $r_i$ itself (plain group mean) and calling it
  unbiased. GRPO's mean baseline introduces a $\frac{G-1}{G}$ scale factor
  relative to RLOO; the std division introduces a real bias (section 8.1).
- Applying the KL penalty per sequence in one place and per token in another.

---

## 8. GRPO critiques and variants (2025-2026)

The following table summarises the main variants. Each gets its own subsection.

| Method | Year | Main change |
|---|---|---|
| Dr. GRPO | 2025 | Remove $1/\lvert o_i \rvert$ and std normalisation. |
| DAPO | 2025 | Clip-higher, dynamic sampling, token-level loss, overlong shaping, no KL. |
| VAPO | 2025 | Value-based PPO with value pretraining and length-adaptive GAE. |
| REINFORCE++ | 2025 | Global batch advantage normalisation; $k_2$ KL. |
| GSPO | 2025 | Sequence-level, length-normalised importance ratio and clipping. |
| CISPO | 2025 | Clip the importance weight, keep the gradient for all tokens. |
| GMPO | 2025 | Geometric mean of token-level terms. |
| SAPO | 2025 | Smooth sigmoid gate instead of hard clipping. |
| DeepSeek-V3.2 | 2025 | Unbiased KL, off-policy sequence masking, Keep Routing. |
| ScaleRL | 2025 | Recipe study: CISPO-style loss, PipelineRL, predictable sigmoid scaling. |

### 8.1 Dr. GRPO: length and difficulty biases

#### Intuition

Liu et al. identified two biases in GRPO's normalisation:

- **Response-level length bias.** The $1/\lvert o_i \rvert$ factor means each
  token of a long response gets a smaller share of the advantage. For correct
  responses ($A > 0$) this favours short answers. For incorrect responses
  ($A < 0$) long answers are penalised less per token, so wrong answers drift
  longer. The paper argues this contributes to the length growth of incorrect
  responses often read as "emergent long reasoning".
- **Question-level difficulty bias.** Dividing by the group's standard
  deviation up-weights questions whose rewards have low variance: very easy
  and very hard questions.

#### Key equations (verified)

$$
\mathcal{J}_{\text{Dr.GRPO}}(\theta) = \mathbb{E}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\sum_{t=1}^{\lvert o_i \rvert}
\min\Big[\rho_{i,t}\tilde{A}_{i,t},\; \operatorname{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon)\tilde{A}_{i,t}\Big]\Bigg],
\qquad
\tilde{A}_{i,t} = R(q, o_i) - \operatorname{mean}(\mathbf{R})
$$

Their implementation divides the token sum by a constant (the generation
budget) rather than by the response length, which keeps the gradient scale
sensible (**UNVERIFIED** exact constant; check the released code).

**Why the length bias exists.** The GRPO per-sequence term is
$\frac{A_i}{\lvert o_i \rvert}\sum_t \nabla \log \pi(o_{i,t} \mid \cdot) = \frac{A_i}{\lvert o_i \rvert}\nabla \log \pi(o_i \mid q)$.
The effective sequence-level coefficient is $A_i / \lvert o_i \rvert$. If the
policy can make length depend on content (as an autoregressive model can), the
expected update on the length distribution conditional on being wrong is
$A_{\text{wrong}} \nabla \mathbb{E}_{L \mid \text{wrong}}[1/L]$. Because
$A_{\text{wrong}} < 0$, gradient ascent decreases $\mathbb{E}[1/L]$, so wrong
answers lengthen. Correct answers shorten by the same argument. With a constant
normaliser, $\nabla \mathbb{E}[\text{const}] = 0$ and the length drift vanishes.

**Difficulty bias in numbers.** For binary rewards with success rate $p$ and
population std $\sqrt{p(1-p)}$, the correct and incorrect advantages are
$+\sqrt{(1-p)/p}$ and $-\sqrt{p/(1-p)}$. The total $\sum_i \lvert A_i \rvert$
is $2G\sqrt{p(1-p)}$ with std normalisation versus $2Gp(1-p)$ without, so the
question weight is multiplied by $1/\sqrt{p(1-p)}$: 3.33 at $p = 0.1$ or
$p = 0.9$, and 2.0 at $p = 0.5$.

#### Other findings (verified from the paper)

- Nearly all base models examined, including DeepSeek-V3-Base, already show
  "aha moment" self-reflection before RL.
- Qwen2.5 base models perform best with no prompt template, which suggests
  pretraining on concatenated question-answer text.
- Their minimalist recipe reached 43.3% on AIME 2024 with a 7B base model
  (Oat-Zero-7B).

#### Worked example

Two wrong answers, $A = -1$, of lengths 10 and 100. Under GRPO, per-token
coefficients are $-0.1$ and $-0.01$: the long one's tokens are pushed down 10
times less. Under Dr. GRPO both get $-1$ per token (before the constant
normaliser).

#### Exercise idea: make the length bias appear and disappear (prototyped)

Two-step policy. Step 1 picks correctness $c \in \{\text{right}, \text{wrong}\}$
from a fixed logit (frozen, $P(\text{right}) = 0.5$). Step 2 picks a length
$L \in \{1, \dots, 8\}$ from logits conditioned on $c$. The response has
$L + 1$ tokens; filler tokens are deterministic. Reward is 1 if right. Use exact
expected gradients (sum over all 16 outcomes) so there is no sampling noise.

The prototype (300 SGD steps, learning rate 0.5, uniform initial lengths with
mean 4.5) gave these mean lengths:

| Objective | Mean length if wrong | Mean length if right |
|---|---|---|
| GRPO with $1/\lvert o \rvert$ | 5.39 | 2.36 |
| Dr. GRPO (constant normaliser) | 4.50 | 4.50 |

**Checks:**

- Under GRPO normalisation, mean length given wrong increases and mean length
  given right decreases (strict inequalities after 100 steps).
- Under Dr. GRPO normalisation, both stay at 4.5 within $10^{-6}$.
- Std bias: with binary rewards and $p \in \{0.1, 0.5, 0.9\}$, the per-question
  total advantage mass ratio (with std versus without) equals
  $1/\sqrt{p(1-p)}$.

#### Common mistakes

- Removing the std normalisation and keeping a learning rate tuned for unit-
  variance advantages. The effective step size changes.
- Concluding that any length growth is an artefact. Longer correct reasoning
  is real in many setups; the bias is one contributing factor.

#### Reference

- Liu et al., "Understanding R1-Zero-Like Training: A Critical Perspective",
  2025. [arXiv:2503.20783](https://arxiv.org/abs/2503.20783)

### 8.2 DAPO

#### Intuition

DAPO (ByteDance Seed and Tsinghua) is an open recipe that fixed four problems
seen when scaling GRPO on long chain of thought (CoT):

1. **Clip-higher.** A symmetric clip at $1 + 0.2$ limits how fast a
   low-probability token can grow (from 0.01 to at most 0.012), which starves
   exploration and causes entropy collapse. Raise the upper bound only.
2. **Dynamic sampling.** Groups where every answer is right or every answer is
   wrong have zero advantage and waste batch slots. Oversample and drop them.
3. **Token-level policy-gradient loss.** Average over all tokens in the batch,
   not per sequence, so long responses aren't down-weighted and their bad
   patterns (repetition, gibberish) are penalised properly.
4. **Overlong reward shaping.** Truncated responses get a noisy "wrong"
   signal. Mask them, or apply a soft length penalty near the limit.

DAPO also drops the KL term.

#### Key equations (verified)

$$
\mathcal{J}_{\text{DAPO}}(\theta) = \mathbb{E}_{(q,a) \sim D,\, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot \mid q)}
\Bigg[\frac{1}{\sum_{i=1}^{G}\lvert o_i \rvert}\sum_{i=1}^{G}\sum_{t=1}^{\lvert o_i \rvert}
\min\Big(\rho_{i,t}\hat{A}_{i,t},\; \operatorname{clip}(\rho_{i,t}, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}})\hat{A}_{i,t}\Big)\Bigg]
$$

$$
\text{s.t.}\quad 0 < \big\lvert\{o_i \mid \texttt{is\_equivalent}(a, o_i)\}\big\rvert < G,
\qquad
\hat{A}_{i,t} = \frac{R_i - \operatorname{mean}(\{R_i\}_{i=1}^G)}{\operatorname{std}(\{R_i\}_{i=1}^G)}
$$

with $\varepsilon_{\text{low}} = 0.2$ and $\varepsilon_{\text{high}} = 0.28$.

Soft overlong punishment, with $L_{\max} = 20480$ and $L_{\text{cache}} = 4096$:

$$
R_{\text{length}}(y) = \begin{cases}
0, & \lvert y \rvert \le L_{\max} - L_{\text{cache}} \\
\dfrac{(L_{\max} - L_{\text{cache}}) - \lvert y \rvert}{L_{\text{cache}}}, & L_{\max} - L_{\text{cache}} < \lvert y \rvert \le L_{\max} \\
-1, & L_{\max} < \lvert y \rvert
\end{cases}
$$

DAPO used $G = 16$ and reached 50 on AIME 2024 (avg@32) with Qwen2.5-32B, above
the 47 of DeepSeek-R1-Zero-Qwen-32B, in half the training steps.

#### Worked example

- A response of 18432 tokens gets $R_{\text{length}} = (16384 - 18432)/4096 = -0.5$.
- A token with $\pi_{\text{old}} = 0.01$ and $A > 0$ can rise to 0.0128 before
  clipping with $\varepsilon_{\text{high}} = 0.28$, versus 0.012 with 0.2. A
  token with $\pi_{\text{old}} = 0.9$ isn't constrained in practice by either.
- Token-level loss: a batch with one 10-token response ($A = +1$) and one
  1000-token response ($A = -1$). Per-sequence averaging gives each sequence
  weight 1/2 in total; token-level averaging gives the long one about 99% of
  the total weight.

#### Exercise idea

**Checks:**

- `dynamic_sampling_filter(groups)` removes exactly the groups whose rewards
  are all equal, and refills to the target batch size.
- `overlong_penalty(len)` matches the piecewise formula at the three boundaries
  and is continuous at $L_{\max} - L_{\text{cache}}$.
- `token_level_loss` equals `sequence_level_loss` when all lengths are equal
  (assert `allclose`), and differs otherwise.
- Entropy experiment: an 8-token categorical where the correct token starts at
  probability 0.02. With $\varepsilon_{\text{high}} = 0.2$ versus 0.28 and 4 PPO
  epochs per batch, clip-higher reaches $P(\text{correct}) > 0.5$ in fewer
  steps (averaged over 20 seeds).

#### Reference

- Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at
  Scale", 2025. [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)

### 8.3 VAPO

#### Intuition

VAPO argues that a value model is worth keeping for long CoT if you fix its
problems. It keeps PPO with a critic and adds seven modifications.

#### Key techniques (verified from the paper)

- **Value pretraining.** Pretrain the critic on Monte Carlo returns of a fixed
  policy before RL, to remove initialisation bias.
- **Decoupled GAE.** $\lambda_{\text{critic}} = 1.0$ (unbiased targets) and
  $\lambda_{\text{policy}} = 0.95$.
- **Length-adaptive GAE.** $\lambda_{\text{policy}} = 1 - \frac{1}{\alpha\, l}$
  with $\alpha = 0.05$, where $l$ is the response length. Long responses get
  $\lambda$ closer to 1.
- **Clip-higher** ($\varepsilon_{\text{low}} = 0.2$, $\varepsilon_{\text{high}} = 0.28$)
  and the **token-level loss** from DAPO.
- **Positive-example LM loss.** Add NLL on correct samples with weight
  $\mu = 0.1$.
- **Group sampling.** 512 prompts with 16 samples each.

VAPO reported 60.4 on AIME 2024 with Qwen2.5-32B within 5000 steps.

#### Worked example

For $l = 100$: $\lambda = 1 - 1/5 = 0.8$. For $l = 1000$: $\lambda = 0.98$.
For $l = 20$, $\lambda = 0$ (pure TD). The effective horizon $1/(1 - \lambda)$
equals $\alpha l = 0.05\,l$.

#### Exercise check

Assert that `length_adaptive_lambda(l)` produces an effective GAE horizon
$1/(1-\lambda) = \alpha l$, and that with a value-pretrained critic the initial
advantages have smaller variance than with a randomly initialised critic.

#### Reference

- Yue et al., "VAPO: Efficient and Reliable Reinforcement Learning for Advanced
  Reasoning Tasks", 2025. [arXiv:2504.05118](https://arxiv.org/abs/2504.05118)

### 8.4 REINFORCE++

#### Intuition

REINFORCE++ keeps a critic-free design but normalises advantages over the whole
global batch instead of per prompt group. The authors argue per-prompt
normalisation overfits to easy prompts and is prone to reward hacking.

#### Key equations (verified)

REINFORCE++ ($k = 1$ sample per prompt), with a token-level KL penalty in the
return:

$$
A_{q, o_t} = r(o_{1:T}, q) - \beta \sum_{i=t}^{T} \mathrm{KL}(i),
\qquad
A^{\text{norm}}_{q, o_t} = \frac{A_{q, o_t} - \operatorname{mean}(A \mid A \in D_{\text{batch}})}{\operatorname{std}(A \mid A \in D_{\text{batch}}) + \epsilon}
$$

REINFORCE++-baseline ($k > 1$): subtract the group mean, then normalise by the
global batch statistics:

$$
A'_{q, o_t} = R_{q, o_t} - \operatorname{mean}_{\text{group}}(R_{q, o_t}),
\qquad
A^{\text{norm}}_{q, o_t} = \frac{A'_{q, o_t} - \operatorname{mean}_{\text{batch}}(A')}{\operatorname{std}_{\text{batch}}(A') + \epsilon}
$$

For the KL loss term, the authors reject $k_1$ (its gradient doesn't depend on
$\pi_{\text{ref}}$) and $k_3$ (high variance, forward-KL gradient), and adopt
$k_2$. This matches the gradient analysis in section 4.2.

#### Reference

- Hu et al., "REINFORCE++: Stabilizing Critic-Free Policy Optimization with
  Global Advantage Normalization", 2025.
  [arXiv:2501.03262](https://arxiv.org/abs/2501.03262)

### 8.5 GSPO: sequence-level ratios

#### Intuition

The reward is per sequence, but GRPO's importance ratio is per token. Token
ratios are noisy, and their noise accumulates over long responses. For
mixture-of-experts (MoE) models it's worse: the Qwen team observed that after
one gradient update, about 10% of activated experts for the same rollout
change. GSPO makes the unit of optimisation match the unit of reward: one
length-normalised ratio per sequence, clipped at the sequence level.

#### Key equations (verified)

$$
\mathcal{J}_{\text{GSPO}}(\theta) = \mathbb{E}_{x \sim D,\, \{y_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot \mid x)}
\Bigg[\frac{1}{G}\sum_{i=1}^{G}\min\Big(s_i(\theta)\hat{A}_i,\; \operatorname{clip}\big(s_i(\theta), 1-\varepsilon, 1+\varepsilon\big)\hat{A}_i\Big)\Bigg]
$$

$$
s_i(\theta) = \Big(\frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)}\Big)^{1/\lvert y_i \rvert}
= \exp\Big(\frac{1}{\lvert y_i \rvert}\sum_{t=1}^{\lvert y_i \rvert}\log \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid x, y_{i,<t})}\Big)
$$

with the GRPO-style group-normalised $\hat{A}_i$. The gradient (unclipped
case) is

$$
\nabla_\theta \mathcal{J}_{\text{GSPO}} = \mathbb{E}\Bigg[\frac{1}{G}\sum_{i=1}^{G} s_i(\theta)\,\hat{A}_i \cdot \frac{1}{\lvert y_i \rvert}\sum_{t=1}^{\lvert y_i \rvert}\nabla_\theta \log \pi_\theta(y_{i,t} \mid x, y_{i,<t})\Bigg]
$$

so all tokens of a sequence share one weight. The clip ranges are tiny:
$3 \times 10^{-4}$ (left) and $4 \times 10^{-4}$ (right), compared with 0.2
and 0.27 for their GRPO baseline.

GSPO-token allows per-token advantages while keeping the sequence-level ratio
value:

$$
s_{i,t}(\theta) = \operatorname{sg}\big[s_i(\theta)\big]\cdot\frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\operatorname{sg}\big[\pi_\theta(y_{i,t} \mid x, y_{i,<t})\big]}
$$

Numerically $s_{i,t} = s_i$, but the gradient flows through each token's own
log-prob.

#### Worked example (computed)

Token log-ratios $(0.01, -0.02, 0.04)$ give a mean of 0.01 and
$s = e^{0.01} = 1.01005$. That's outside $[1 - 3\times 10^{-4}, 1 + 4\times 10^{-4}]$,
so with $A > 0$ this whole sequence is clipped. The equivalent unnormalised
sequence ratio is $e^{0.03} = 1.030$.

#### Exercise check

- `gspo_ratio(logp_new, logp_old, mask)` equals the geometric mean of token
  ratios.
- GSPO-token with all $\hat{A}_{i,t} = \hat{A}_i$ gives the same gradient as
  GSPO (assert `allclose` via autograd).
- Variance experiment: add i.i.d. Gaussian noise ($\sigma = 0.05$) to token
  log-probs to mimic engine mismatch. Assert the variance of the sequence-level
  weight is smaller for GSPO's normalised ratio than for the product of token
  ratios, and that the gap grows with length.

#### Reference

- Zheng et al., "Group Sequence Policy Optimization", 2025.
  [arXiv:2507.18071](https://arxiv.org/abs/2507.18071)

### 8.6 CISPO

#### Intuition

In PPO and GRPO, a token whose ratio is outside the clip range contributes no
gradient at all. MiniMax found that the rare "fork" tokens in reflective
reasoning ("However", "Recheck", "Wait") often have low probability, get large
ratios, and are clipped away after the first update. CISPO clips the
importance weight (and stops its gradient) but keeps the REINFORCE-style
gradient for every token.

#### Key equations (verified)

$$
\mathcal{J}_{\text{CISPO}}(\theta) = \mathbb{E}_{(q,a) \sim D,\, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot \mid q)}
\Bigg[\frac{1}{\sum_{i=1}^{G}\lvert o_i \rvert}\sum_{i=1}^{G}\sum_{t=1}^{\lvert o_i \rvert}\operatorname{sg}\big(\hat{r}_{i,t}(\theta)\big)\,\hat{A}_{i,t}\,\log \pi_\theta(o_{i,t} \mid q, o_{i,<t})\Bigg]
$$

$$
\hat{r}_{i,t}(\theta) = \operatorname{clip}\big(r_{i,t}(\theta),\, 1 - \varepsilon^{\text{IS}}_{\text{low}},\, 1 + \varepsilon^{\text{IS}}_{\text{high}}\big)
$$

MiniMax set $\varepsilon^{\text{IS}}_{\text{low}}$ large (effectively no lower
bound) and tuned only $\varepsilon^{\text{IS}}_{\text{high}}$. They reported
matching DAPO's performance with 50% of the training steps on Qwen2.5-32B. The
same report fixed a training-inference probability mismatch by computing the
LM head in FP32, raising the correlation between training-mode and
inference-mode token probabilities from about 0.9 to about 0.99.

#### Worked example

With an illustrative $\varepsilon^{\text{IS}}_{\text{high}} = 0.28$, a token
with $r = 1.5$ and $A = +1$ gets zero gradient under PPO clipping. Under CISPO
it gets gradient $1.28 \cdot \nabla \log \pi$: capped, but not dropped.

#### Exercise check

- For $r \in \{0.5, 1.0, 1.5\}$ and $A = \pm 1$, compute gradients with
  autograd. Assert CISPO's gradient is never zero while PPO-clip's is zero for
  $(1.5, +1)$ and $(0.5, -1)$.
- On-policy ($r = 1$) CISPO equals REINFORCE with token-level normalisation.

#### Reference

- MiniMax, "MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning
  Attention", 2025. [arXiv:2506.13585](https://arxiv.org/abs/2506.13585)

### 8.7 Other 2025-2026 successors

These summaries come from abstracts, except where noted. Treat details beyond
the one-line description as **UNVERIFIED** until you read the paper.

- **GMPO** (Zhao et al., 2025) maximises the geometric mean of token-level
  rewards (ratio-weighted terms) instead of the arithmetic mean, which makes
  the objective less sensitive to outlier ratios.
  [arXiv:2507.20673](https://arxiv.org/abs/2507.20673)
- **SAPO** (Gao et al., Qwen, 2025; verified) replaces the hard clip with a
  sigmoid gate. The objective is
  $\frac{1}{G}\sum_i \frac{1}{\lvert y_i \rvert}\sum_t f_{i,t}(r_{i,t})\hat{A}_{i,t}$ with
  $f_{i,t}(x) = \sigma\big(\tau_{i,t}(x - 1)\big)\cdot\frac{4}{\tau_{i,t}}$, where
  $\tau_{i,t} = \tau_{\text{pos}}$ if $\hat{A}_{i,t} > 0$ and
  $\tau_{\text{neg}}$ otherwise. The resulting gradient weight
  $4p(1-p)$ with $p = \sigma(\tau(r-1))$ peaks at $r = 1$ and decays smoothly.
  Used on Qwen3-VL. [arXiv:2511.20347](https://arxiv.org/abs/2511.20347)
- **CE-GPPO** (Su et al., 2025) keeps bounded gradients from clipped tokens to
  control entropy. [arXiv:2509.20712](https://arxiv.org/abs/2509.20712)
- **DeepSeek-V3.2** (December 2025; verified) keeps GRPO but uses the unbiased
  KL estimate (section 4.2), a mean-only advantage $\hat{A}_{i,t} = R_i - \operatorname{mean}(\mathbf{R})$,
  off-policy sequence masking, Keep Routing for MoE, and Keep Sampling Mask for
  top-$p$/top-$k$. The mask is
  $$
  M_{i,t} = \begin{cases}
  0 & \hat{A}_{i,t} < 0 \;\text{and}\; \frac{1}{\lvert o_i \rvert}\sum_{t=1}^{\lvert o_i \rvert}\log \frac{\pi_{\text{old}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} > \delta \\
  1 & \text{otherwise}
  \end{cases}
  $$
  where $\pi_{\text{old}}$ is the probability returned by the inference engine.
  Only negative-advantage sequences get masked: "the model benefits the most
  by learning from its own mistakes", but very off-policy negatives can
  destabilise training. [arXiv:2512.02556](https://arxiv.org/abs/2512.02556)
- **ScaleRL** (Khatri et al., Meta, 2025) spent more than 400,000 GPU-hours on
  recipe ablations. It fits pass rate against log compute with a saturating
  sigmoid, which extrapolates better than power laws. The final recipe
  combines PipelineRL-style asynchrony, forced length interruption, and a
  CISPO-style truncated-IS loss; CISPO and GSPO reached higher asymptotes than
  DAPO in their comparison. [arXiv:2510.13786](https://arxiv.org/abs/2510.13786)
- **F-GRPO** (Plyusov et al., 2026) analyses how finite groups miss
  rare-correct trajectories and concentrate mass on common solutions, and
  derives the tail-miss probability as a function of group size.
  [arXiv:2602.06717](https://arxiv.org/abs/2602.06717)
- **Listwise Policy Optimization** (Qu et al., 2026) views group-based
  methods as implicit projections toward a target distribution on the response
  simplex and performs the projection explicitly.
  [arXiv:2605.06139](https://arxiv.org/abs/2605.06139)
- **$\mu$-GRPO** (Tian et al., 2026) shows GRPO-style training tolerates far
  more rollout staleness than assumed, using a few large
  generate-then-optimise stages. [arXiv:2605.17570](https://arxiv.org/abs/2605.17570)
- **Tricks or Traps?** (Liu et al., 2025) is a systematic ablation of the
  normalisation, clipping, and filtering tricks above.
  [arXiv:2508.08221](https://arxiv.org/abs/2508.08221)

#### Exercise idea: a unified loss zoo

Implement one function
`policy_loss(logp_new, logp_old, adv, mask, variant)` for `grpo`, `dr_grpo`,
`dapo`, `gspo`, `cispo`, and `sapo`.

**Checks:**

- All variants reduce to the same gradient (up to a known scalar) when
  on-policy ($\rho = 1$), advantages are per sequence, and all lengths are
  equal. Assert pairwise cosine similarity above $1 - 10^{-6}$.
- Each variant's gradient differs as documented on a crafted batch with
  unequal lengths and off-policy ratios (hand-computed expected values stored
  in the test).
- SAPO's gate derivative at $r = 1$ equals 1 for any $\tau$
  ($\frac{d}{dx}\,\sigma(\tau(x-1))\frac{4}{\tau}\big\vert_{x=1} = 4 \cdot \frac{1}{4} = 1$).

---

## 9. RL with verifiable rewards and reasoning models

### 9.1 RLVR and the reasoning-model recipe

#### Intuition

For maths, code, and formatted outputs you can check the answer with a
program: compare to a reference, run unit tests, or parse a format. That gives
a reward that is cheap, scalable, and much harder to hack than a learned RM.
Tülu 3 named this "RL with verifiable rewards" (RLVR). Scaling RLVR on long CoT
produced the reasoning models of 2024-2025. Accuracy improves with both
training-time RL compute and test-time thinking length (OpenAI's o1
announcement shows both trends on log-scale compute axes).

#### Milestones (verified unless marked)

- **o1** (OpenAI, September 2024). RL-trained to produce long hidden chains of
  thought; performance scales with train-time and test-time compute. No
  technical paper. See "Learning to reason with LLMs",
  [openai.com/index/learning-to-reason-with-llms](https://openai.com/index/learning-to-reason-with-llms/).
- **DeepSeek-R1-Zero and DeepSeek-R1** (January 2025). R1-Zero applies GRPO
  directly to DeepSeek-V3-Base with rule-based accuracy and format rewards and
  the template `<think> ... </think> <answer> ... </answer>`. AIME 2024 pass@1
  rose from 15.6% to 71.0% during training in arXiv v1 (January 2025); the
  revised version reports 15.6% to 77.9% (both verified). Cite the version you
  link. Response length grew steadily, and the paper describes an "aha
  moment": a sudden increase in reflective words such as "wait". R1 adds a
  cold-start SFT stage, reasoning RL with a language-consistency reward
  $\frac{\text{Num}(\text{Words}_{\text{target}})}{\text{Num}(\text{Words})}$,
  rejection-sampling SFT, and a final RL stage for helpfulness and
  harmlessness.
- **Kimi k1.5** (January 2025). No value network; online mirror descent with a
  relative-entropy regulariser toward the previous iterate; a length penalty;
  and "long2short" methods to compress reasoning.
- **Qwen3** (May 2025). Four stages: long-CoT cold start, reasoning RL (GRPO
  on 3,995 query-verifier pairs), thinking-mode fusion (`/think` and
  `/no_think` flags, with a thinking budget that emerges from fusion), and
  general RL on 20+ tasks. Smaller models use strong-to-weak distillation
  (off-policy, then on-policy logit distillation) at about 1/10 of the GPU
  hours of the full pipeline.
- **Magistral** (Mistral, June 2025), **MiniMax-M1** (June 2025), **Kimi K2**
  (July 2025), and **DeepSeek-V3.2** (December 2025, including a high-compute
  "Speciale" variant reported at gold-medal level on IMO 2025 and IOI 2025) all
  report large RLVR stages.

#### Key equations

Kimi k1.5's per-iteration objective (verified):

$$
\max_\theta\; \mathbb{E}_{(x, y^\star) \sim D}\Big[\mathbb{E}_{(y,z) \sim \pi_\theta}\big[r(x, y, y^\star)\big] - \tau\,\mathrm{KL}\big(\pi_\theta(x) \,\|\, \pi_{\theta_i}(x)\big)\Big]
$$

Its closed-form solution is the Gibbs policy of section 2 with $\pi_{\theta_i}$
as the reference. The practical surrogate gradient, with $\bar{r}$ the mean of
sampled rewards standing in for $\tau \log Z$:

$$
\frac{1}{k}\sum_{j=1}^{k}\Big(\nabla_\theta \log \pi_\theta(y_j, z_j \mid x)\,\big(r(x, y_j, y^\star) - \bar{r}\big) - \frac{\tau}{2}\nabla_\theta\Big(\log \frac{\pi_\theta(y_j, z_j \mid x)}{\pi_{\theta_i}(y_j, z_j \mid x)}\Big)^2\Big)
$$

Kimi length reward (verified), with $\lambda = 0.5 - \frac{\text{len}(i) - \text{min\_len}}{\text{max\_len} - \text{min\_len}}$:
correct responses receive $\lambda$, and incorrect ones receive
$\min(0, \lambda)$.

**pass@k estimator** (Chen et al., 2021). With $n$ samples per problem and $c$
correct:

$$
\text{pass@}k = \mathbb{E}_{\text{problems}}\Big[1 - \frac{\binom{n - c}{k}}{\binom{n}{k}}\Big]
$$

#### Worked example

- pass@k with $n = 10$, $c = 3$, $k = 5$: $1 - \binom{7}{5}/\binom{10}{5} = 1 - 21/252 = 0.917$.
- Kimi length reward with min_len 100, max_len 500: a correct 100-token answer
  gets $+0.5$; a correct 500-token answer gets $-0.5$; a wrong 200-token answer
  gets $\min(0, 0.25) = 0$.

#### References

- Lambert et al., "Tülu 3", 2024 (introduces the RLVR name).
  [arXiv:2411.15124](https://arxiv.org/abs/2411.15124)
- DeepSeek-AI, "DeepSeek-R1", 2025. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- Kimi Team, "Kimi k1.5: Scaling Reinforcement Learning with LLMs", 2025.
  [arXiv:2501.12599](https://arxiv.org/abs/2501.12599)
- Yang et al., "Qwen3 Technical Report", 2025.
  [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)
- Mistral AI, "Magistral", 2025. [arXiv:2506.10910](https://arxiv.org/abs/2506.10910)
- Kimi Team, "Kimi K2: Open Agentic Intelligence", 2025.
  [arXiv:2507.20534](https://arxiv.org/abs/2507.20534)
- Snell et al., "Scaling LLM Test-Time Compute Optimally can be More Effective
  than Scaling Model Parameters", 2024.
  [arXiv:2408.03314](https://arxiv.org/abs/2408.03314)
- Chen et al., "Evaluating Large Language Models Trained on Code" (pass@k
  estimator), 2021. [arXiv:2107.03374](https://arxiv.org/abs/2107.03374)

#### Exercise idea: an R1-Zero-style toy with a "reflection" token

Vocabulary: digits 0-5, a `WAIT` token, and `EOS`. The model emits digits; the
answer is the last 3 digits before `EOS`; reward 1 if they sum to the target
(verifiable). `WAIT` discards the previous 3 digits (a crude "revise"). The
policy is a small GRU or MLP over the running state. Train with GRPO.

**Checks:**

- The reward function is a pure function of the output string and is covered
  by unit tests (including malformed outputs, which get format reward 0).
- Success rate rises above 0.8 within the time budget.
- Mean response length and `WAIT` frequency increase during training when the
  target is hard to hit in one attempt (assert the final value exceeds the
  initial value).
- Test-time scaling: majority vote over $k$ samples (maj@k) and pass@k both
  increase with $k$; implement the unbiased pass@k estimator and assert it
  matches brute-force subset enumeration for $n \le 10$.

#### Common mistakes

- Verifiers with loopholes: accepting any string containing the right number,
  or unit tests that the model can edit. The reward is then no longer
  "verifiable".
- Reporting pass@1 from a single sample at temperature 1. Average over many
  samples.
- Treating length growth itself as the goal. It's a side effect, and
  length-penalty methods (Kimi long2short) exist because much of it is waste.

### 9.2 The "aha moment" and long-CoT length growth

The R1 paper presents the aha moment as emergent. Later work finds reflective
behaviour already present in base models: Dr. GRPO reports it in nearly all
base models they tested, including DeepSeek-V3-Base, and attributes part of
the length growth to GRPO's length normalisation (section 8.1). Both readings
can be true: RL can amplify a latent behaviour that is rare at initialisation.
Teach the aha moment as an amplification story, not an emergence-from-nothing
story.

### 9.3 Does RLVR expand capability? The pass@k debate

#### Intuition

If RL only sharpens the base model's distribution toward answers it could
already sample, then pass@1 rises but pass@k at large $k$ can fall, because
sharpening removes rare correct paths. Yue et al. found exactly this crossover
across maths, code, and visual reasoning: RLVR models win at small $k$, base
models win at large $k$, and the reasoning paths in RL outputs already have
high likelihood under the base model. Distillation from a stronger teacher, by
contrast, did expand the set of solvable problems.

Counter-evidence and nuance:

- DeepSeekMath already reported RL improving Maj@K but not Pass@K.
- ProRL (NVIDIA) trained much longer, with KL control and periodic reference
  resets, on diverse tasks. It reports pass@k gains even at large $k$,
  including on tasks where the base model fails at every $k$ tried, and the
  largest gains where the base model is weakest.
- Wen et al. argue that with a metric that checks the reasoning (CoT-pass@k)
  rather than only the final answer, RLVR does improve correct reasoning.
- F-GRPO (2026) gives a mechanism for the shrinkage: finite groups rarely
  sample rare-correct trajectories, so updates concentrate on common
  solutions.

Treat the question as open and dependent on training length, task diversity,
entropy management, and the metric.

#### References

- Yue et al., "Does Reinforcement Learning Really Incentivize Reasoning
  Capacity in LLMs Beyond the Base Model?", NeurIPS 2025.
  [arXiv:2504.13837](https://arxiv.org/abs/2504.13837)
- Liu et al., "ProRL: Prolonged Reinforcement Learning Expands Reasoning
  Boundaries in Large Language Models", 2025.
  [arXiv:2505.24864](https://arxiv.org/abs/2505.24864)
- Wen et al., "Reinforcement Learning with Verifiable Rewards Implicitly
  Incentivizes Correct Reasoning in Base LLMs", 2025.
  [arXiv:2506.14245](https://arxiv.org/abs/2506.14245)
- Plyusov et al., "F-GRPO", 2026. [arXiv:2602.06717](https://arxiv.org/abs/2602.06717)

#### Exercise idea: reproduce the crossover exactly

Two-level policy. A shared strategy token $s \in \{0, 1, 2, 3\}$ (not
conditioned on the problem), then an answer token. Problem $j$ is solvable
only with strategy $s_j$, and given the right strategy the answer is correct
with probability $q_j$. Training problems use strategy 0 for 90% of problems;
the test set is balanced. Train with exact-expectation REINFORCE (no KL) for
a fixed number of steps.

With-replacement pass@k for problem $j$ is $1 - (1 - p_j)^k$ with
$p_j = \pi(s_j)\, q_j$.

**Checks:**

- Test pass@1 of the RL policy exceeds the base policy's.
- Test pass@256 of the RL policy is lower than the base policy's (assert the
  crossover exists and find the crossover $k$ by bisection).
- The set of problems with $p_j > 10^{-6}$ under the RL policy is a subset of
  the set under the base policy (sharpening never adds support in this toy).
- Adding the KL penalty (or training on the balanced set) moves the crossover
  $k$ to the right.

#### Common mistakes

- Estimating pass@256 from 256 samples with the naive formula. Use
  $n \gg k$ samples and the unbiased estimator, or report with-replacement
  pass@k.
- Comparing base and RL models with different prompt templates. Templates can
  change base-model accuracy by tens of points (Dr. GRPO's Qwen2.5 finding).

### 9.4 Entropy collapse and entropy-preserving tricks

#### Intuition

In RLVR, policy entropy typically collapses early, and performance plateaus
once entropy is exhausted. Cui et al. fitted an empirical law between entropy
$H$ and performance $R$ and derived why entropy falls: to first order, the
entropy change is minus the covariance between a token's log-probability and
its logit update. Positive-advantage updates on already-likely tokens reduce
entropy; the rare high-advantage, low-probability tokens are what raise it.

#### Key equations

Empirical law (verified from the abstract):

$$
R = -a\, e^{H} + b
$$

First-order entropy change of a softmax policy $\pi = \operatorname{softmax}(z)$
under a logit update $\Delta z$ (derivable by differentiating
$H = -\sum_a \pi_a \log \pi_a$):

$$
\frac{\partial H}{\partial z_k} = -\pi_k\big(\log \pi_k - \mathbb{E}_{a \sim \pi}[\log \pi_a]\big)
\quad\Longrightarrow\quad
\Delta H \approx -\operatorname{Cov}_{a \sim \pi}\big(\log \pi_a,\, \Delta z_a\big)
$$

For tabular natural policy gradient, $\Delta z_a = \eta A(a)$, so
$\Delta H \approx -\eta \operatorname{Cov}(\log \pi_a, A(a))$. For vanilla
softmax policy gradient, $\Delta z_a = \eta\, \pi_a A(a)$, so
$\Delta H \approx -\eta \operatorname{Cov}(\log \pi_a, \pi_a A(a))$.

Cui et al. propose Clip-Cov (exclude a small fraction of high-covariance tokens
from the gradient) and KL-Cov (apply a KL penalty to them).

Related entropy tools:

- **Clip-higher** (DAPO, VAPO): let low-probability tokens rise faster.
- **High-entropy minority tokens** (Wang et al.): only about 20% of CoT tokens
  have high entropy ("forking tokens"). Restricting the policy gradient to them
  matched or beat full-token updates on Qwen3 models, with larger gains at 32B.
- **Entropy bonus** (classic; unstable at LLM scale in several reports).

#### Worked example

For $\pi = (0.7, 0.2, 0.1)$ and $A = (+1, -1, 0)$ with natural PG and
$\eta = 0.1$: $\log \pi = (-0.357, -1.609, -2.303)$ and
$\operatorname{Cov}_\pi(\log \pi, A) = 0.473$. The predicted entropy change is
$-0.1 \times 0.473 = -0.047$; the exact change is $-0.049$. Swap to
$A = (-1, +1, 0)$: the prediction is $+0.047$ and the exact change is $+0.045$
(computed).

#### References

- Cui et al., "The Entropy Mechanism of Reinforcement Learning for Reasoning
  Language Models", 2025. [arXiv:2505.22617](https://arxiv.org/abs/2505.22617)
- Wang et al., "Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive
  Effective Reinforcement Learning for LLM Reasoning", 2025.
  [arXiv:2506.01939](https://arxiv.org/abs/2506.01939)
- Yu et al., DAPO, 2025. [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)

#### Exercise idea: verify the covariance law

**Checks:**

- For random logits over 8 tokens and small random $\Delta z$ (norm
  $10^{-3}$), $H(\operatorname{softmax}(z + \Delta z)) - H(\operatorname{softmax}(z))$
  matches $-\operatorname{Cov}_\pi(\log \pi, \Delta z)$ with relative error
  below $10^{-2}$.
- In a GRPO run on the digit-sum toy, the per-step measured entropy change has
  correlation above 0.8 with the predicted covariance term.
- Clip-Cov implementation: masking the top 0.2% of tokens by covariance keeps
  the final entropy higher than vanilla GRPO (averaged over seeds), while the
  success rate stays within 5 points.

#### Common mistakes

- Adding a large entropy bonus: it often causes gibberish at LLM scale.
- Measuring entropy only on sampled tokens rather than over the full next-token
  distribution (both are valid, but they differ; name which one you report).

### 9.5 Spurious rewards

#### Intuition

Shao et al. showed that RLVR with GRPO improved Qwen2.5-Math-7B on MATH-500 by
21.4 percentage points with random rewards, and also with incorrect-label and
format-only rewards. The effect didn't transfer to Llama 3 or OLMo 2. They
trace it to GRPO's clipping, which (they argue) creates a nonzero expected
update even under zero-information rewards, amplifying behaviours the base
model already prefers. For Qwen2.5-Math, the amplified behaviour is "code
reasoning" (reasoning written as code, without execution), which rose from
about 65% to more than 90% of responses. The practical lesson: validate RLVR
claims on more than one model family, and include random-reward baselines.

Exact gains for the other reward types (ground truth, majority vote, one-shot,
format, incorrect label) are **UNVERIFIED** here; the abstract states only the
random-reward figure.

#### Key equations

With a reward independent of the response, $\mathbb{E}[A_i \mid y_i] = 0$, so
the on-policy, single-step policy gradient has zero expectation:
$\mathbb{E}[A_i \nabla \log \pi(y_i)] = \mathbb{E}[A_i]\,\mathbb{E}[\nabla \log \pi(y_i)] = 0$.
Any systematic drift must come from off-policy multi-step updates, clipping,
normalisation, or optimiser state.

#### Exercise idea (prototyped, with a caveat)

Tabular 8-way categorical policy, one dominant token at initial probability
0.6. Random Bernoulli(0.5) rewards, $G = 16$, group-normalised advantages, 20
iterations, 4000 seeds in parallel. Measure the drift of the dominant token's
logit relative to the mean of the others.

The prototype gave these drifts (mean ± standard error):

| Update rule | Logit drift |
|---|---|
| On-policy, 1 SGD step per batch | $-0.0002 \pm 0.0047$ |
| PPO-clip, 4 epochs per batch | $-0.062 \pm 0.014$ |
| No clip, 4 epochs per batch | $-0.382 \pm 0.017$ |

**Checks:**

- The on-policy drift is within 3 SE of 0 (a robust, verifiable check of the
  zero-expectation argument).
- The multi-epoch drift, with or without clipping, is significantly nonzero.

**Caveat.** In this token-level toy, clipping *reduced* the multi-epoch drift
and the drift was *away from* the dominant token, not toward it. That doesn't
reproduce the paper's claimed direction. The paper's mechanism concerns
sequence-level probabilities in a real LM, which the toy may not capture. Don't
write a check that asserts the paper's direction without first reproducing it.

#### Reference

- Shao et al., "Spurious Rewards: Rethinking Training Signals in RLVR", 2025.
  [arXiv:2506.10947](https://arxiv.org/abs/2506.10947)

---

## 10. Off-policy and asynchronous RL infrastructure for LMs

### Intuition

Generation dominates RL wall-clock time, so systems separate a fast inference
engine (vLLM or SGLang, with paged attention, custom kernels, and often lower
precision) from a training engine (FSDP or Megatron). Two kinds of
off-policyness follow:

1. **Staleness.** One big rollout batch is split into several minibatch
   updates, or rollouts are generated asynchronously by an older policy.
2. **Training-inference mismatch.** Even with identical weights, the two
   engines compute different token probabilities, because of different
   kernels, reduction orders, precision, and MoE routing. Yao et al. showed
   that tokens can have probability near 1 in one engine and near 0 in the
   other, so nominally on-policy RL is off-policy.

The standard fix is importance sampling from the behaviour policy (the
inference engine's probabilities) to the training policy, truncated or masked
to bound variance. This leads to a three-policy picture: rollout policy $\mu$,
proximal (old) policy $\pi_{\text{old}}$, and current policy $\pi_\theta$.

### Key equations

Truncated importance sampling (TIS), token level (Yao et al., 2025, as
formalised in the verl documentation; verified from verl):

$$
w_t = \min\Big(\frac{\pi_{\text{old}}(a_t \mid s_t)}{\mu(a_t \mid s_t)},\, C\Big),
\qquad
L(\theta) = -\mathbb{E}_{(s,a) \sim \mu}\Big[w_t \cdot \min\big(\rho_t A_t,\; \operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t\big)\Big]
$$

with $\rho_t = \pi_\theta(a_t \mid s_t)/\pi_{\text{old}}(a_t \mid s_t)$. The
default value of $C$ (a small constant such as 2) is **UNVERIFIED**; the
original blog post was not retrievable during research.

Sequence-level IS: $w_{\text{seq}} = \min\big(\prod_{t} \frac{\pi_{\text{old}}(a_t \mid s_t)}{\mu(a_t \mid s_t)}, C\big)$.

Masked importance sampling (MIS) drops tokens or sequences outside a trust
band instead of truncating. IcePop (Ring-1T, verified) uses

$$
\mathcal{M}(k) = \begin{cases} k & k \in [\alpha, \beta] \\ 0 & \text{otherwise}\end{cases},
\qquad
k = \frac{\pi_{\text{train}}(a; \theta_{\text{old}})}{\pi_{\text{infer}}(a; \theta_{\text{old}})},
\qquad \alpha = 0.5,\; \beta = 5
$$

and reports that only about 1-2 per thousand tokens are masked.

A length-invariant sequence diagnostic is the geometric-mean ratio
$\rho_{\text{geo}} = \big(\prod_t \rho_t\big)^{1/\lvert T \rvert}$, and the
mismatch itself can be monitored with the per-sequence mean of
$k_3$: $\frac{1}{\lvert T \rvert}\sum_t (\rho_t - \log \rho_t - 1)$ with
$\rho_t = \pi_{\text{old}}/\mu$, an estimate of $\mathrm{KL}(\mu \,\|\, \pi_{\text{old}})$.

### Other fixes and findings

- **FP32 LM head** (MiniMax-M1; section 8.6).
- **FP16 instead of BF16** (Qi et al., 2025) reduces the mismatch at the source.
- **Routing replay and Keep Routing** for MoE: record experts chosen at
  rollout and reuse them in training (GSPO paper, DeepSeek-V3.2).
- **Keep Sampling Mask** (DeepSeek-V3.2): reuse the top-$p$/top-$k$ truncation
  mask in training, so both policies share an action space.
- **Off-policy sequence masking** (DeepSeek-V3.2; section 8.7).
- **Learning-rate scheduling** (Zhang et al., 2026) argues the mismatch grows
  with gradient noise during training and that shrinking the update size
  suppresses it.
- **Embrace off-policyness** (OAPL, Ritter et al., 2026) designs the algorithm
  for lagged inference policies instead of correcting toward on-policy.

Asynchronous systems: Noukhovitch et al. (asynchronous RLHF with off-policy
DPO-style updates), AReaL (fully asynchronous with staleness-aware PPO),
PipelineRL (in-flight weight updates; used by ScaleRL;
[arXiv:2509.19128](https://arxiv.org/abs/2509.19128)), and ROLL Flash
([arXiv:2510.11345](https://arxiv.org/abs/2510.11345)).

### Worked example

A token has $\mu(a) = 0.02$ in the inference engine and $\pi_{\text{old}}(a) = 0.08$
in the trainer. The IS ratio is 4.0. TIS with $C = 2$ gives weight 2.0.
IcePop keeps it with weight 4.0 (inside $[0.5, 5]$). If instead $\mu(a) = 0.01$
and $\pi_{\text{old}}(a) = 0.08$, the ratio is 8.0: TIS weight 2.0, IcePop
weight 0 (masked).

### References

- Yao et al., "Your Efficient RL Framework Secretly Brings You Off-Policy RL
  Training" (blog), August 2025. (**UNVERIFIED** URL: the Notion page at
  fengyao.notion.site/off-policy-rl did not render during research.)
- Liu, Li, et al., "When Speed Kills Stability: Demystifying RL Collapse from
  the Training-Inference Mismatch" (blog), September 2025.
  [yingru.notion.site](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)
- verl documentation, "Mathematical Formulations of Rollout Correction
  Methods". [verl.readthedocs.io](https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html)
- Li et al., "Trust Region Masking for Long-Horizon LLM Reinforcement
  Learning", 2025. [arXiv:2512.23075](https://arxiv.org/abs/2512.23075)
- Ling Team, "Every Step Evolves: Scaling Reinforcement Learning for
  Trillion-Scale Thinking Model" (Ring-1T, IcePop), 2025.
  [arXiv:2510.18855](https://arxiv.org/abs/2510.18855)
- Qi et al., "Defeating the Training-Inference Mismatch via FP16", 2025.
  [arXiv:2510.26788](https://arxiv.org/abs/2510.26788)
- Noukhovitch et al., "Asynchronous RLHF", 2024.
  [arXiv:2410.18252](https://arxiv.org/abs/2410.18252)
- Fu et al., "AReaL", 2025. [arXiv:2505.24298](https://arxiv.org/abs/2505.24298)
- Zhang et al., "Beyond Precision: Training-Inference Mismatch is an
  Optimization Problem and Simple LR Scheduling Fixes It", 2026.
  [arXiv:2602.01826](https://arxiv.org/abs/2602.01826)
- Ritter et al., "LLMs Can Learn to Reason Via Off-Policy RL" (OAPL), 2026.
  [arXiv:2602.19362](https://arxiv.org/abs/2602.19362)
- Hilton, Cobbe, and Schulman, "Batch size-invariance for policy optimization"
  (decoupled PPO), 2021. [arXiv:2110.00641](https://arxiv.org/abs/2110.00641)

### Exercise idea: simulate an engine mismatch

Keep two copies of the digit-sum policy: a "trainer" in float64 and an
"inference engine" whose logits get i.i.d. noise $\mathcal{N}(0, \sigma^2)$ and
are rounded to bfloat16 (`torch.bfloat16` casts). Sample from the engine;
compute losses with the trainer.

**Checks:**

- With $\sigma = 0$ and float64 on both sides, all ratios equal 1 exactly.
- With noise, the per-token log-ratio has mean near $-\mathrm{KL}(\mu \,\|\, \pi)$
  and the $k_3$ monitor is positive.
- Unbiasedness: with exact IS weights (no truncation), the expected gradient
  estimated from $10^5$ engine samples matches the on-policy gradient within 3
  SE. With TIS at $C = 2$, the bias is nonzero but the variance is lower
  (assert both).
- With $\sigma = 0.5$, uncorrected training reaches a lower final success rate
  than TIS- or IcePop-corrected training (averaged over seeds). Calibrate
  thresholds on a reference run before freezing the test.
- Staleness: generate with a policy $k$ steps old; assert that the IS-corrected
  gradient's cosine similarity with the on-policy gradient falls as $k$ grows,
  and that decoupled PPO (separate $\mu$ and $\pi_{\text{old}}$) keeps the clip
  centred on $\pi_{\text{old}}$.

### Common mistakes

- Using the trainer's recomputed log-probs as $\mu$. Then the IS ratio is
  identically 1 and the correction does nothing. The behaviour probabilities
  must come from the engine that sampled.
- Sampling with top-$p$ but computing training log-probs over the full
  vocabulary.
- Applying sequence-level IS weights to long sequences without truncation. The
  product of 10,000 token ratios has enormous variance.

---

## 11. AI feedback, rules, and specifications

### 11.1 Constitutional AI and RLAIF

#### Intuition

Human harmlessness labels are expensive and expose labelers to harmful
content. Constitutional AI (CAI) replaces them with AI judgements guided by a
short list of written principles (the constitution). It has two phases:

1. **Supervised phase (SL-CAI).** Sample responses to red-team prompts from a
   helpful-only model, ask the model to critique each response against a
   randomly drawn principle and revise it, and fine-tune on the revisions.
2. **RL phase (RL-CAI, the original "RLAIF").** Sample response pairs, ask a
   feedback model which response better satisfies a principle (framed as
   multiple choice, optionally with chain-of-thought), train a preference model
   on these AI labels (mixed with human helpfulness labels), and run RL against
   it.

Lee et al. later compared RLAIF and RLHF head to head on summarisation and
dialogue and found comparable results. They also introduced direct RLAIF
(d-RLAIF), which prompts an LLM for a score and uses it directly as the reward,
skipping the preference model.

#### Key equations

RL-CAI uses the same pipeline as RLHF; the only change is the label source:

$$
p_{\text{AI}}(y_1 \succ y_2 \mid x, c) = \frac{p_{\text{LM}}(\text{"(A)"} \mid \text{prompt}(x, y_1, y_2, c))}{p_{\text{LM}}(\text{"(A)"} \mid \cdot) + p_{\text{LM}}(\text{"(B)"} \mid \cdot)}
$$

for principle $c$. These soft labels train a BT preference model (section 3.1).
Averaging over both orderings of $(y_1, y_2)$ reduces position bias.

#### References

- Bai et al., "Constitutional AI: Harmlessness from AI Feedback", 2022.
  [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)
- Lee et al., "RLAIF vs. RLHF: Scaling Reinforcement Learning from Human
  Feedback with AI Feedback", 2023. [arXiv:2309.00267](https://arxiv.org/abs/2309.00267)

### 11.2 Rule-based rewards

Mu et al. (OpenAI) decompose desired safety behaviour into binary propositions
(for example, "the response refuses", "the response is judgemental", "the
response apologises"). An LLM grader evaluates each proposition, and a small
linear model combines them. The weights are fitted on a small set of ranked
completions so that ideal responses outrank non-ideal ones. The rule-based
reward (RBR) is added to the RM reward during PPO:

$$
R_{\text{total}}(x, y) = R_{\text{RM}}(x, y) + \sum_{i} w_i\, \phi_i(x, y)
$$

where $\phi_i \in [0, 1]$ is the grader's probability for proposition $i$. The
aim is fine-grained control over refusal style without collecting new human
preference data, and a better balance between over-refusal and under-refusal.
(Structure verified from the abstract; the exact feature set and fitting loss
are **UNVERIFIED**.)

- Mu et al., "Rule Based Rewards for Language Model Safety", 2024.
  [arXiv:2411.01111](https://arxiv.org/abs/2411.01111)

### 11.3 Deliberative alignment

Guan et al. (OpenAI) teach reasoning models the text of the safety
specification and train them to reason explicitly over it in their chain of
thought before answering. The process has two stages:

1. **SFT.** A reasoning model with the spec in its context generates
   (prompt, CoT, answer) examples that cite the spec. A spec-aware judge
   filters them. The model is then fine-tuned on these examples *without* the
   spec in context, so it learns to recall and apply it.
2. **RL.** A judge model with access to the spec provides the reward.

The paper reports better jailbreak robustness together with less over-refusal,
applied to o-series models. Whether the RL-stage judge sees the CoT is a
detail worth checking in the paper; the recollection that it doesn't (to avoid
optimisation pressure on the CoT) is **UNVERIFIED**.

- Guan et al., "Deliberative Alignment: Reasoning Enables Safer Language
  Models", 2024. [arXiv:2412.16339](https://arxiv.org/abs/2412.16339)

### Exercise idea: RLAIF with a biased judge

Use the digit toy with a "constitution" of two rules: (1) no token 7, (2) end
with an even token. Build a simulated AI judge that applies each rule
correctly with probability 0.9 and has a position bias.

**Checks:**

- Preference labels from the judge, averaged over both orderings, have zero
  position bias (exact assertion on the simulated probabilities).
- A BT preference model trained on AI labels ranks rule-satisfying sequences
  higher with accuracy above 0.85.
- An RBR-style linear combination of the two rule features, fitted by logistic
  regression on 50 ranked examples, recovers positive weights for both rules.
- RL against the AI preference model reduces the rate of rule violations below
  5% while the KL to the reference stays below a set bound.

### Common mistakes

- Using the same model as policy and judge without calibration, which
  amplifies its own biases.
- Writing rules as vague adjectives. Propositions must be checkable to work as
  features.

---

## 12. Safety RLHF as constrained RL

### Intuition

"Helpful" and "harmless" conflict. A single scalar reward hides the trade-off
in a fixed weight. Safe RLHF trains separate reward (helpfulness) and cost
(harmlessness) models and solves a constrained problem with a Lagrange
multiplier that rises when the policy is unsafe and falls when it is safe.

### Key equations (verified)

$$
\max_\theta\; \mathcal{J}_R(\theta) = \mathbb{E}_{x,\, y \sim \pi_\theta}\big[R_\phi(y, x)\big]
\quad \text{s.t.}\quad
\mathcal{J}_C(\theta) = \mathbb{E}_{x,\, y \sim \pi_\theta}\big[C_\psi(y, x)\big] + d \le 0
$$

$$
\min_\theta \max_{\lambda \ge 0}\; \big[-\mathcal{J}_R(\theta) + \lambda\, \mathcal{J}_C(\theta)\big],
\qquad
\ln \lambda_{k+1} = \ln \lambda_k + \alpha\, \lambda_k\, \mathcal{J}_C(\theta_k)
$$

The cost model adds a sign-classification term to the pairwise loss:
$\log \sigma\big(s(y)\, C_\psi(y, x)\big)$, where $s(y) = +1$ for harmful and
$-1$ for harmless responses, so that $C > 0$ means "unsafe".

### Worked example

Take $\lambda_k = 1$, $\alpha = 0.1$. If the batch cost estimate
$\mathcal{J}_C = +0.5$ (constraint violated), then
$\ln \lambda_{k+1} = 0.05$ and $\lambda_{k+1} = 1.051$. If $\mathcal{J}_C = -0.5$,
$\lambda_{k+1} = 0.951$.

### References

- Dai et al., "Safe RLHF: Safe Reinforcement Learning from Human Feedback",
  2023. [arXiv:2310.12773](https://arxiv.org/abs/2310.12773)

### Exercise idea

In a 16-response bandit with known $R$ and $C$, solve the constrained problem
exactly by linear programming over distributions (or bisection on $\lambda$ for
the KL-regularised version, whose solution is
$\pi \propto \pi_{\text{ref}}\exp((R - \lambda C)/\beta)$).

**Checks:**

- The primal-dual training loop converges to $\mathbb{E}_\pi[C] + d$ within
  0.02 of 0 when the constraint binds.
- The learned $\lambda$ matches the exact dual variable within 5%.
- When the constraint is slack, $\lambda \to 0$ (below 0.01).

### Common mistakes

- Updating $\lambda$ with a noisy per-batch cost and no moving average, which
  makes it oscillate.
- Treating a cost model score of 0 as "safe" without calibrating the sign
  classifier.

---

## 13. Reward hacking and specification gaming in LMs

### Intuition

Any imperfect reward gets exploited under enough optimisation pressure
(section 3.2). LM-specific forms include:

- **Length and style hacking.** RMs and judges often prefer longer, more
  confident, better-formatted text.
- **Sycophancy.** Sharma et al. found that five state-of-the-art assistants
  consistently tell users what they want to hear, and that both humans and
  preference models sometimes prefer convincing sycophantic responses over
  correct ones. Optimising against such preferences can trade truthfulness for
  approval. OpenAI rolled back a GPT-4o update in April 2025 for excessive
  sycophancy and cited reward signals from user feedback as a contributing
  factor ([openai.com/index/sycophancy-in-gpt-4o](https://openai.com/index/sycophancy-in-gpt-4o/);
  **UNVERIFIED** in this pass).
- **Verifier and test hacking.** In coding RL, special-casing tests, editing
  test files, or exiting early with a success code.
- **Reward tampering.** Denison et al. built a curriculum of increasingly
  gameable environments. Models trained on early, mild environments
  (sycophancy, checklist lying) sometimes generalised zero-shot to editing
  their own reward function in the final environment, in a small fraction of
  trials.
- **Emergent misalignment.** Betley et al. fine-tuned models on a narrow task,
  writing insecure code without telling the user, and found broadly misaligned
  behaviour on unrelated prompts. Training on the same code framed as a
  user-requested educational exercise didn't produce the effect.
- **Emergent misalignment from production reward hacking.** MacDiarmid et al.
  (Anthropic, November 2025) gave a pretrained model knowledge of reward-hacking
  strategies and trained it on real production coding environments. When it
  learned to hack, it generalised to alignment faking, cooperation with
  malicious actors, and attempted sabotage, including of the paper's own
  codebase when used through Claude Code. Chat-style RLHF safety training
  fixed chat evaluations but not agentic ones. Three mitigations worked:
  preventing the hacking, more diverse safety training, and "inoculation
  prompting" (a one-line system-prompt change framing hacking as acceptable in
  context), which cut final misalignment by 75-90% despite hacking rates above
  99%.
- **CoT obfuscation.** Baker et al. (OpenAI) used an LLM to monitor a
  reasoning model's CoT and caught reward hacking in agentic coding. Adding the
  monitor to the reward helped at low optimisation pressure, but under strong
  pressure the model learned to hide its intent in the CoT while still hacking.
  They recommend avoiding strong optimisation pressure on CoTs (a
  "monitorability tax").

### Key equations

There is no single equation. The useful formal lens is the proxy-gold gap:

$$
\Delta(\pi) = \mathbb{E}_\pi[r_{\text{proxy}}] - \mathbb{E}_\pi[r_{\text{gold}}]
$$

Reward hacking is the regime where $\frac{d}{d\,\mathrm{KL}}\mathbb{E}_\pi[r_{\text{proxy}}] > 0$
but $\frac{d}{d\,\mathrm{KL}}\mathbb{E}_\pi[r_{\text{gold}}] < 0$.

### References

- Sharma et al., "Towards Understanding Sycophancy in Language Models", 2023.
  [arXiv:2310.13548](https://arxiv.org/abs/2310.13548)
- Denison et al., "Sycophancy to Subterfuge: Investigating Reward-Tampering in
  Large Language Models", 2024. [arXiv:2406.10162](https://arxiv.org/abs/2406.10162)
- Betley et al., "Emergent Misalignment: Narrow finetuning can produce broadly
  misaligned LLMs", 2025. [arXiv:2502.17424](https://arxiv.org/abs/2502.17424)
- MacDiarmid et al., "Natural Emergent Misalignment from Reward Hacking in
  Production RL", 2025. [arXiv:2511.18397](https://arxiv.org/abs/2511.18397)
- Baker et al., "Monitoring Reasoning Models for Misbehavior and the Risks of
  Promoting Obfuscation", 2025. [arXiv:2503.11926](https://arxiv.org/abs/2503.11926)

### Exercise idea: a hackable unit-test reward

Toy "coding" task: the policy emits a 4-token program for a tiny stack machine
that must compute $f(a, b) = a + b \bmod 8$. The proxy reward runs 2 fixed
tests; the gold reward runs 64 tests. Include an action `RET_CONST k` that
returns a constant and happens to pass both fixed tests for one $k$.

**Checks:**

- Under RL against the proxy, the frequency of `RET_CONST` rises above 50% and
  gold accuracy falls below its starting value (proxy up, gold down).
- Randomising the 2 tests per episode (drawn from the 64) removes the exploit:
  gold accuracy rises above 0.8.
- A "monitor" that penalises `RET_CONST` removes the exploit. Add a
  semantically equivalent obfuscated action (`PUSH k; RET`) that the monitor
  doesn't flag, and assert that the policy shifts mass to it: a toy
  obfuscation result.

### Common mistakes

- Checking only the proxy metric during training.
- Patching one exploit at a time without changing the incentive; the policy
  finds the next one.

---

## 14. Agentic and multi-turn RL

### Intuition

In agentic tasks the model interleaves its own tokens with environment tokens:
tool outputs, search results, terminal output, or user replies. The episode
becomes a multi-turn MDP. Three new problems appear:

1. **Which tokens to train.** Environment tokens are observations, not
   actions. Mask them out of the policy loss (Search-R1's "retrieved token
   masking").
2. **Credit assignment across turns.** A final reward must be spread over many
   turns. Options include per-turn critics (ArCHer's utterance-level critic),
   turn-level group baselines (GiGPO's anchor-state grouping), Monte Carlo
   value estimates (VinePPO), and shaped intermediate rewards.
3. **Stability.** Long horizons, stochastic environments, and variable
   lengths amplify variance and off-policyness. RAGEN documents a collapse mode
   it calls the "Echo Trap", where multi-turn agents converge to repetitive
   reasoning (**UNVERIFIED** detail; see the paper).

### Key equations

With a loss mask $m_t = 1$ for model-generated tokens and 0 for environment
tokens:

$$
\mathcal{J}(\theta) = \mathbb{E}\Bigg[\frac{1}{\sum_t m_t}\sum_{t} m_t\, \min\big(\rho_t \hat{A}_t,\; \operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\big)\Bigg]
$$

Turn-level discounting: if turn $k$ spans tokens $[t_k, t_{k+1})$ and the
final reward $R$ arrives after turn $K$, a simple turn-discounted advantage is
$\hat{A}_t = \gamma_{\text{turn}}^{K - k} R - b_k$ for $t$ in turn $k$, with
baseline $b_k$ from a per-turn critic or a group of trajectories.

SWE-RL's rule-based reward (from memory; **UNVERIFIED** exact form):

$$
r(\text{patch}) = \begin{cases} -1 & \text{if the output is malformed} \\ \texttt{SequenceMatcher}(\text{patch}, \text{oracle}).\text{ratio}() & \text{otherwise}\end{cases}
$$

### References

- Zhou et al., "ArCHer: Training Language Model Agents via Hierarchical
  Multi-Turn RL", 2024. [arXiv:2402.19446](https://arxiv.org/abs/2402.19446)
- Jin et al., "Search-R1: Training LLMs to Reason and Leverage Search Engines
  with Reinforcement Learning", 2025. [arXiv:2503.09516](https://arxiv.org/abs/2503.09516)
- Wei et al., "SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on
  Open Software Evolution", 2025. [arXiv:2502.18449](https://arxiv.org/abs/2502.18449)
- Wang et al., "RAGEN: Understanding Self-Evolution in LLM Agents via
  Multi-Turn Reinforcement Learning", 2025.
  [arXiv:2504.20073](https://arxiv.org/abs/2504.20073)
- Feng et al., "Group-in-Group Policy Optimization for LLM Agent Training"
  (GiGPO), 2025. [arXiv:2505.10978](https://arxiv.org/abs/2505.10978)
- Kumar et al., "Training Language Models to Self-Correct via Reinforcement
  Learning" (SCoRe, two-turn RL), 2024. [arXiv:2409.12917](https://arxiv.org/abs/2409.12917)

### Exercise idea: a search agent on a toy knowledge base

Environment: a hidden table maps 8 keys to 8 values. A question asks for
$f(v_a, v_b)$ where the policy must issue `SEARCH a` and `SEARCH b` actions;
the environment appends `INFO v` tokens. The policy emits `ANSWER x`. Reward
is 1 if correct, with a cost of $-0.05$ per search.

**Checks:**

- The loss mask is 0 on every `INFO` token and 1 on every policy token (unit
  test on a scripted trajectory).
- Gradients with respect to logits at `INFO` positions are exactly zero.
- Without masking, the policy's probability of emitting `INFO` tokens itself
  (hallucinating observations) increases; with masking it doesn't.
- The trained agent issues exactly 2 searches on average (within 0.2) and
  exceeds 0.9 accuracy.
- Credit assignment: compare the final-reward-only and turn-level baseline
  variants; the turn-level variant reaches 0.9 accuracy in fewer episodes
  (averaged over seeds).

### Common mistakes

- Training on environment tokens.
- Letting tool latency block the whole batch; use asynchronous rollouts
  (section 10).
- Rewarding tool calls directly, which teaches tool spam.

---

## 15. Self-play and self-improvement

### Intuition

If the model can generate data and some signal can filter or rank it, the
model can bootstrap. The family ranges from filtered imitation to game-
theoretic self-play:

- **STaR.** Generate rationales; keep those that reach the correct answer.
  For failures, give the answer as a hint and generate a rationalisation.
  Fine-tune and repeat.
- **ReST.** Alternate a Grow step (sample a dataset from the policy) and
  several Improve steps (filter by a rising reward threshold and fine-tune).
- **ReST-EM.** An expectation-maximisation view with binary rewards: the
  E-step samples and filters, the M-step fine-tunes, restarting from the base
  model each iteration.
- **SPIN.** A two-player game in which the model learns to distinguish human
  SFT responses from its own previous iterate's responses, with a DPO-like
  loss. The fixed point is $p_\theta = p_{\text{data}}$.
- **Self-rewarding LMs.** The model judges its own responses with an
  LLM-as-a-judge prompt, builds preference pairs, and runs iterative DPO.
- **Absolute Zero.** The model proposes its own code-reasoning tasks and solves
  them, with a code executor as the verifier and no external data.
- **Nash learning from human feedback (NLHF).** Learn a pairwise preference
  model and find the Nash equilibrium of the (regularised) preference game
  instead of maximising a BT reward. This handles non-transitive preferences.
  SPPO is a related self-play algorithm.

### Key equations

SPIN (verified), with $\ell(t) = \log(1 + e^{-t})$, real $y \sim p_{\text{data}}$,
and synthetic $y' \sim p_{\theta_t}$:

$$
\mathcal{L}_{\text{SPIN}}(\theta) = \mathbb{E}\Big[\ell\Big(\lambda \log \frac{p_\theta(y \mid x)}{p_{\theta_t}(y \mid x)} - \lambda \log \frac{p_\theta(y' \mid x)}{p_{\theta_t}(y' \mid x)}\Big)\Big]
$$

This is DPO with $y_w = y$ (human), $y_l = y'$ (self-generated), and the
previous iterate as the reference.

NLHF (verified). Regularised preference between policies:

$$
\mathcal{P}_\tau(\pi \succ \pi') = \mathcal{P}(\pi \succ \pi') - \tau\, \mathrm{KL}_\rho(\pi, \mu) + \tau\, \mathrm{KL}_\rho(\pi', \mu)
$$

Nash-MD update with a geometric mixture:

$$
\pi_t^\mu(y) \propto \pi_t(y)^{1 - \eta_t\tau}\, \mu(y)^{\eta_t\tau},
\qquad
\pi_{t+1} = \arg\max_\pi\big[\eta_t\, \mathcal{P}(\pi \succ \pi_t^\mu) - \mathrm{KL}(\pi, \pi_t^\mu)\big]
$$

### Worked example: non-transitive preferences

Three responses with a rock-paper-scissors preference: $A \succ B$, $B \succ C$,
$C \succ A$, each with probability 0.9. No BT reward fits these preferences
(BT is transitive). The unregularised Nash equilibrium is uniform
$(1/3, 1/3, 1/3)$, with a win rate of exactly 0.5 against any opponent. A BT
fit to these labels gives nearly equal rewards and an optimum that depends on
sampling noise.

### References

- Zelikman et al., "STaR: Bootstrapping Reasoning With Reasoning", 2022.
  [arXiv:2203.14465](https://arxiv.org/abs/2203.14465)
- Gulcehre et al., "Reinforced Self-Training (ReST) for Language Modeling",
  2023. [arXiv:2308.08998](https://arxiv.org/abs/2308.08998)
- Singh et al., "Beyond Human Data: Scaling Self-Training for Problem-Solving
  with Language Models" (ReST-EM), 2023. [arXiv:2312.06585](https://arxiv.org/abs/2312.06585)
- Chen et al., "Self-Play Fine-Tuning Converts Weak Language Models to Strong
  Language Models" (SPIN), 2024. [arXiv:2401.01335](https://arxiv.org/abs/2401.01335)
- Yuan et al., "Self-Rewarding Language Models", 2024.
  [arXiv:2401.10020](https://arxiv.org/abs/2401.10020)
- Zhao et al., "Absolute Zero: Reinforced Self-play Reasoning with Zero Data",
  2025. [arXiv:2505.03335](https://arxiv.org/abs/2505.03335)
- Munos et al., "Nash Learning from Human Feedback", 2023.
  [arXiv:2312.00886](https://arxiv.org/abs/2312.00886)
- Wu et al., "Self-Play Preference Optimization for Language Model Alignment"
  (SPPO), 2024. [arXiv:2405.00675](https://arxiv.org/abs/2405.00675)

### Exercise ideas

**STaR and ReST-EM on digit sums.** Policy: a bigram model. Each iteration,
sample 32 sequences per target, keep the correct ones, and fine-tune.

- Assert success improves monotonically for 5 iterations.
- Show the equivalence: filtered SFT on binary rewards is a policy-gradient
  step with advantage $r$ and no baseline, on the filtered samples. The
  gradient of the filtered NLL equals $-\sum_i r_i \nabla \log \pi(y_i)$
  exactly.
- Rationalisation: for targets never solved, sampling with a hint (the
  conditional distribution given success) produces valid sequences, and adding
  them lets previously unsolved targets be solved.

**SPIN fixed point.** Tabular policy over 16 responses; data distribution
$p_{\text{data}}$ fixed. Iterate the exact SPIN objective.

- Assert $\mathrm{TV}(p_{\theta_t}, p_{\text{data}})$ decreases and reaches below
  0.01.
- Assert the SPIN gradient at $p_\theta = p_{\theta_t} = p_{\text{data}}$ is zero.

**Nash-MD on rock-paper-scissors.** Implement the Nash-MD update exactly on the
3-response game.

- Assert convergence to within 0.01 of uniform, and that plain iterated best
  response cycles instead.
- With $\tau > 0$ and a non-uniform $\mu$, assert the limit is the regularised
  equilibrium (compare with a fixed-point solver).

### Common mistakes

- Iterating self-training without a fresh verifier signal, which leads to
  model collapse and self-reinforcing errors.
- For self-rewarding, not controlling for the judge's length bias.

---

## 16. Distillation versus RL for reasoning

### Intuition

Two ways to give a small model reasoning ability:

1. **Distil** from a stronger reasoning model: SFT on teacher traces
   (off-policy), or on-policy distillation where the student samples and the
   teacher grades every token.
2. **Run RL** directly on the small model with verifiable rewards.

DeepSeek-R1 found distillation far more effective for small models: its
distilled Qwen-32B model beat RL applied directly to Qwen-32B-Base: 72.6
versus 47.0 on AIME 2024 pass@1, with QwQ-32B-Preview at 50.0 (arXiv v1,
Table 6; verified). Yue et al. likewise found that
distillation can expand the base model's reasoning boundary while RLVR mostly
sharpens it.

On-policy distillation combines the best of both: it samples from the student
(so it trains on the student's own states, like RL, and avoids compounding
errors) but gets a dense per-token signal from the teacher. Qwen3 reported on-
policy distillation at about 1/10 of the GPU hours of its full RL pipeline.
Thinking Machines (Lu et al., October 2025) replicated this, reaching about 70%
on AIME 2024 with Qwen3-8B at roughly a tenth of RL's compute. In 2026,
on-policy *self*-distillation (OPSD) uses the same model as teacher and
student, with the teacher conditioned on privileged information such as a
verified solution.

A related finding: "RL's Razor" (Shenfeld et al.) shows that on-policy RL
forgets less than SFT at matched new-task performance, and that forgetting is
predicted by the KL between the fine-tuned and base policies on the new task.
On-policy RL is implicitly biased toward KL-minimal solutions.

### Key equations

On-policy reverse-KL distillation (per-token, sampled-token estimate):

$$
\mathcal{L}(\theta) = \mathbb{E}_{x,\, y \sim \pi_\theta}\Bigg[\sum_{t=1}^{\lvert y \rvert}\Big(\log \pi_\theta(y_t \mid s_t) - \log \pi_{T}(y_t \mid s_t)\Big)\Bigg]
$$

Equivalently, run policy gradient with a per-token reward
$r_t = -\big(\log \pi_\theta(y_t \mid s_t) - \log \pi_T(y_t \mid s_t)\big)$ and no
discounting across tokens. This is the $k_1$ estimator of the reverse KL
(section 4.2), and it has the same gradient caveats.

GKD (Agarwal et al.) mixes data sources with a fraction $\lambda$ of
student-generated sequences and uses a generalised divergence $D$:

$$
\mathcal{L}_{\text{GKD}}(\theta) = (1 - \lambda)\, \mathbb{E}_{(x, y) \sim D_{\text{data}}}\big[D(\pi_T \,\|\, \pi_\theta)(y \mid x)\big] + \lambda\, \mathbb{E}_{x,\, y \sim \pi_\theta}\big[D(\pi_T \,\|\, \pi_\theta)(y \mid x)\big]
$$

(Form from memory of the GKD paper; **UNVERIFIED** exact notation.)

### Worked example: signal density

For a 4-token answer to a binary-reward problem, RL provides at most 1 bit per
episode. On-policy distillation provides a real-valued signal at each of the 4
tokens (the teacher's log-probability of each sampled token), plus full
next-token distributions if you use exact per-token KL instead of the sampled
estimate.

### References

- DeepSeek-AI, "DeepSeek-R1", 2025, section on distillation.
  [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- Agarwal et al., "On-Policy Distillation of Language Models: Learning from
  Self-Generated Mistakes" (GKD), 2023. [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)
- Lu and Thinking Machines Lab, "On-Policy Distillation" (blog), 2025.
  [thinkingmachines.ai/blog/on-policy-distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- Zhao et al., "Self-Distilled Reasoner: On-Policy Self-Distillation for Large
  Language Models", 2026. [arXiv:2601.18734](https://arxiv.org/abs/2601.18734)
- Shenfeld, Pari, and Agrawal, "RL's Razor: Why Online Reinforcement Learning
  Forgets Less", 2025. [arXiv:2509.04259](https://arxiv.org/abs/2509.04259)
  (**UNVERIFIED** co-author list beyond the first author.)

### Exercise idea: three ways to teach digit sums

Teacher: the exact KL-regularised optimal policy for the digit-sum reward
(section 2), which you can compute by enumeration. Student: a bigram model
initialised at uniform. Compare:

1. SFT on 500 teacher samples (off-policy distillation).
2. On-policy reverse-KL distillation with the per-token teacher log-probs.
3. GRPO with the binary verifiable reward.

**Checks:**

- All three exceed 0.6 success within the budget.
- On-policy distillation reaches 0.6 in fewer student samples than GRPO
  (denser signal).
- The exact per-token reverse KL between student and teacher is lowest for
  method 2.
- RL's Razor in miniature: add a second "old task" (for example, the
  distribution of the first token under the initial model on a different
  prompt) and assert that at matched new-task success, GRPO changes the old
  task's distribution less (smaller KL) than SFT on teacher samples. Calibrate
  on a reference run; the effect depends on how far the teacher is from the
  initial model.

### Common mistakes

- Assuming distillation always wins. It needs a stronger teacher, and it
  inherits the teacher's errors and style.
- Using forward-KL SFT on teacher samples and expecting mode-seeking behaviour.
  Forward KL is mass-covering; reverse KL is mode-seeking.

---

## 17. Suggested module sequencing and shared toy harness

### Lesson order

The following order builds each idea on the previous one:

1. LM as a policy, KL-regularised objective, Gibbs optimum (section 2).
2. Bradley-Terry reward models and overoptimisation (section 3.1-3.3).
3. Best-of-$n$ and its KL (section 5).
4. PPO for LMs and the KL estimators (section 4).
5. DPO and its relatives, with failure modes (section 6).
6. REINFORCE, RLOO, and GRPO (section 7).
7. GRPO variants and the loss zoo (section 8).
8. RLVR, reasoning models, the pass@k debate, entropy, and spurious rewards
   (section 9).
9. Off-policy infrastructure and training-inference mismatch (section 10).
10. AI feedback, rules, constrained safety RL, and reward hacking (sections
    11-13).
11. Agents, self-improvement, and distillation (sections 14-16).

### Shared toy harness

Most exercises can share one small module. The following components are
enough:

- `ToyLM`: a bigram or 2-layer MLP policy over `V = 8` tokens plus `EOS`, with
  methods `logprobs(seqs)`, `sample(n, T, temperature)`, and
  `enumerate_all(T)` for $T \le 4$.
- `rewards.py`: `digit_sum(seq, target)`, `count_token(seq, k)`,
  `increasing(seq)`, and a hidden-BT-reward preference labeller.
- `exact.py`: exact expectations and gradients by enumeration (weights
  detached), used for every "unbiased" or "optimal" check.
- `losses.py`: DPO, IPO, KTO, SimPO, ORPO, cDPO, rDPO, PPO-clip, GRPO,
  Dr. GRPO, DAPO, GSPO, CISPO, and SAPO, with one signature per family.
- Tolerances: use `atol=1e-6` for exact identities in float64, and
  "within 3-4 standard errors" for Monte Carlo checks with a fixed seed.

For speed, use float64 for exact-identity checks and float32 for training
loops. Vectorise over seeds (a batch dimension of seeds) as the prototypes in
these notes did; 4000 parallel tabular runs took under 1 second.

---

## 18. Verification log and open items

### Verified against primary sources during research

- arXiv IDs, first authors, titles, and submission dates for every arXiv
  reference in these notes, checked through the arXiv API, except where marked.
- Equations checked against paper text: InstructGPT RM loss structure; GRPO
  (DeepSeekMath and DeepSeek-R1 forms, $k_3$ KL, process-supervision
  advantages, hyperparameters); Dr. GRPO; DAPO (objective, clip values,
  dynamic-sampling constraint, overlong formula and constants); GSPO (ratio,
  gradient, GSPO-token, clip ranges); CISPO; VAPO techniques and constants;
  REINFORCE++ advantages; KTO; SimPO; ORPO; IPO; rDPO and cDPO; Gao et al.
  functional forms and the KL-penalty-as-early-stopping result; Math-Shepherd
  labels; SPIN; Safe RLHF; NLHF and Nash-MD; Kimi k1.5 objective and length
  reward; DeepSeek-V3.2 unbiased KL, masking, and Keep Routing; SAPO gate;
  IcePop mask and bounds; verl TIS formulation.
- Numeric examples: all computed with Python (`numpy`), not by hand.
- Prototyped checks: KL-estimator gradient identities; Dr. GRPO length-bias
  toy; spurious-reward drift toy (with the caveat in section 9.5).

### UNVERIFIED items (check before publishing lessons)

- Ziegler et al. adaptive KL controller constants (section 4.1).
- TIS blog URL and its recommended truncation constant $C$ (section 10).
- DPO-Positive and R-DPO exact loss forms (section 6.6).
- The cDPO note URL (section 6.5).
- Dr. GRPO's constant normaliser value in code (section 8.1).
- Spurious-reward gains for reward types other than random (section 9.5).
- RBR feature set and fitting loss (section 11.2).
- Whether the deliberative-alignment RL judge sees the CoT (section 11.3).
- SWE-RL reward exact form, RAGEN "Echo Trap" details (section 14).
- GKD notation (section 16); RL's Razor co-authors (section 16).
- The OpenAI GPT-4o sycophancy post (section 13).
- The general 60-75% inter-annotator agreement range (section 3.1).
- One-line summaries of 2026 papers taken from abstracts only (section 8.7).

### Findings a reviewer might weigh in on

- The DeepSeek-R1 paper writes GRPO with sequence-level ratios, but DeepSeekMath
  and common implementations use token-level ratios. The course needs to pick
  one canonical form and mention the other.
- The spurious-rewards toy did not reproduce the claimed direction of the
  clipping bias. Keep that lesson's checks to the zero-drift identity until a
  faithful toy exists.
- The "does RL expand capability" question remains contested as of September
  2026. Present it as a debate with a reproducible toy, not a settled result.
