# Research notes 04: RL and preference alignment beyond text

This document collects research material for the course modules on reinforcement
learning (RL) and preference alignment for image and video generation,
vision-language models (VLMs), speech and audio, and embodied foundation models. It
covers work from 2017 to September 2026.

Each topic has the following parts:

1. Core intuition.
2. Key equations, written in KaTeX-compatible LaTeX.
3. A small worked numeric example, where one fits.
4. Canonical references with arXiv links.
5. An exercise idea that runs on CPU in seconds and has verifiable checks.
6. Common mistakes and failure modes.

## How to read this document

The following conventions apply throughout:

- **Verification status.** Equations marked *(verified)* were checked against the
  paper's HTML or PDF text during research. Numeric examples were recomputed in
  Python (numpy and scipy). Anything marked **UNVERIFIED** comes from memory,
  an abstract, or a search snippet, and must be checked before it goes into a lesson.
- **2026 papers.** Many 2026 papers are cited from their arXiv abstract or a search
  result only. They are listed so the course can point at the frontier, but their
  details are **UNVERIFIED** unless stated otherwise.
- **Time conventions differ between papers.** This is the single largest source of
  sign errors in this area. Each section states its convention. The three that
  appear are the following:

| Convention | Noise end | Data end | Used by |
|---|---|---|---|
| DDPM discrete | $t = T$ | $t = 0$ | DDPO, DPOK, Diffusion-DPO, D3PO |
| Rectified flow, "noise at 1" | $t = 1$ | $t = 0$ | Flow-GRPO, DanceGRPO, SD3 |
| Adjoint Matching, "noise at 0" | $t = 0$ | $t = 1$ | Domingo-Enrich et al. |

- **Notation.** $c$ is the conditioning prompt, $x_0$ (or $x_1$ in the Adjoint
  Matching convention) is the clean sample, $r$ is a scalar reward, $\pi_\theta$ is
  the policy being trained, and $\pi_{\text{ref}}$ is the frozen reference model.

---

## Part A. Diffusion and flow models as policies

### A1. Denoising as a multi-step MDP: DDPO and DPOK

#### Intuition

A text-to-image diffusion sampler draws $x_T \sim \mathcal{N}(0, I)$ and then applies
$T$ stochastic denoising steps. Each step samples $x_{t-1}$ from a Gaussian whose mean
depends on $x_t$. If you treat each denoising step as an *action*, the sampler is a
policy in a finite-horizon Markov decision process (MDP) with a reward only at the
final step. The per-step Gaussian has a closed-form log-probability, so REINFORCE and
PPO apply directly. The exact likelihood of the final image $p_\theta(x_0 \mid c)$ is
intractable, but the likelihood of the *whole trajectory* is a product of Gaussians and
is cheap to compute.

#### Key equations

The denoising MDP from DDPO (Black et al.) is defined as follows *(verified against
the paper's formulation; notation adapted)*:

$$
\begin{aligned}
s_t &= (c,\, t,\, x_t), \qquad a_t = x_{t-1}, \\
\pi_\theta(a_t \mid s_t) &= p_\theta(x_{t-1} \mid x_t, c), \\
P(s_{t+1} \mid s_t, a_t) &= \delta\big(c,\, t-1,\, x_{t-1}\big), \\
\rho_0(s_0) &= \big(p(c),\, \delta_T,\, \mathcal{N}(0, I)\big), \\
R(s_t, a_t) &= \begin{cases} r(x_0, c) & t = 0 \\ 0 & \text{otherwise.} \end{cases}
\end{aligned}
$$

The DDPM reverse step in the $\epsilon$-parameterization is a Gaussian:

$$
p_\theta(x_{t-1} \mid x_t, c) = \mathcal{N}\!\Big(x_{t-1};\ \mu_\theta(x_t, t, c),\ \sigma_t^2 I\Big),
\qquad
\mu_\theta = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t, c)\Big).
$$

For a $d$-dimensional sample, the per-step log-probability is:

$$
\log p_\theta(x_{t-1} \mid x_t, c) = -\frac{\lVert x_{t-1} - \mu_\theta \rVert^2}{2\sigma_t^2} - d \log \sigma_t - \frac{d}{2}\log(2\pi).
$$

DDPO uses DDIM sampling with $\eta > 0$, so the step variance is
$\sigma_t^2 = \eta^2 \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\big(1 - \frac{\bar\alpha_t}{\bar\alpha_{t-1}}\big)$.
With $\eta = 0$ the step is deterministic and has no density, so it can't be a
stochastic policy.

DDPO gives two gradient estimators. The score-function version (DDPO$_\text{SF}$) is
REINFORCE over the trajectory:

$$
\nabla_\theta \mathcal{J} = \mathbb{E}\Big[\sum_{t=1}^{T} \nabla_\theta \log p_\theta(x_{t-1} \mid x_t, c)\; r(x_0, c)\Big].
$$

The importance-sampled version (DDPO$_\text{IS}$) reuses samples from
$p_{\theta_\text{old}}$ and applies PPO clipping to the per-step ratio:

$$
\nabla_\theta \mathcal{J} = \mathbb{E}_{p_{\theta_\text{old}}}\Big[\sum_{t=1}^{T} \frac{p_\theta(x_{t-1} \mid x_t, c)}{p_{\theta_\text{old}}(x_{t-1} \mid x_t, c)}\, \nabla_\theta \log p_\theta(x_{t-1} \mid x_t, c)\; r(x_0, c)\Big].
$$

In practice DDPO replaces $r$ with an advantage normalized per prompt (running mean
and standard deviation of rewards for that prompt).

DPOK (Fan et al.) adds a KL penalty to the pretrained model *(verified)*:

$$
\min_\theta\; \alpha\, \mathbb{E}_{p(c)}\mathbb{E}_{p_\theta(x_0 \mid c)}\big[-r(x_0, c)\big] + \beta\, \mathbb{E}_{p(c)}\big[\mathrm{KL}\big(p_\theta(x_0 \mid c)\,\Vert\, p_\text{pre}(x_0 \mid c)\big)\big].
$$

The final-sample KL is intractable, so DPOK uses an upper bound (their Lemma 4.2):

$$
\mathrm{KL}\big(p_\theta(x_0 \mid c) \,\Vert\, p_\text{pre}(x_0 \mid c)\big) \le \sum_{t=1}^{T} \mathbb{E}_{p_\theta(x_t \mid c)}\Big[\mathrm{KL}\big(p_\theta(x_{t-1} \mid x_t, c) \,\Vert\, p_\text{pre}(x_{t-1} \mid x_t, c)\big)\Big].
$$

The right-hand side is the *path* KL. Section E1 explains why it equals a sum of
per-step KLs. DPOK also learns a value baseline $V(x_t, c) \approx \mathbb{E}[r(x_0,c) \mid x_t]$
and uses $r - \hat V$ in place of $r$.

#### Worked example: one Gaussian denoising step

Take a 1-D step with $\mu_\theta = 0.5$, $\sigma_t = 0.2$, and sampled
$x_{t-1} = 0.3$:

$$
\log p_\theta = -\tfrac{1}{2}\Big(\tfrac{0.3 - 0.5}{0.2}\Big)^2 - \log 0.2 - \tfrac{1}{2}\log 2\pi = -0.5 + 1.6094 - 0.9189 = 0.1905.
$$

This matches `scipy.stats.norm.logpdf(0.3, 0.5, 0.2)`. Log-densities can be positive.

Now suppose the old policy had mean $0.45$. Then
$\log p_{\theta_\text{old}} = -\tfrac12(0.75)^2 + 1.6094 - 0.9189 = 0.4093$, and the
per-step ratio is $\exp(0.1905 - 0.4093) = 0.8035$. With a typical LLM clip range of
$0.2$, this ratio is inside $[0.8, 1.2]$. The reference DDPO implementation uses a much
smaller clip range, reported as $10^{-4}$ in the `ddpo-pytorch` config
(**UNVERIFIED**, check the repository), because per-step ratios in high dimensions
move far from 1 after a tiny parameter change.

#### KL between two DDPM steps

When both steps share $\sigma_t$, the KL reduces to a scaled squared distance between
noise predictions:

$$
\mathrm{KL}\big(\mathcal{N}(\mu_\theta, \sigma_t^2 I) \,\Vert\, \mathcal{N}(\mu_\text{ref}, \sigma_t^2 I)\big) = \frac{\lVert \mu_\theta - \mu_\text{ref}\rVert^2}{2\sigma_t^2} = \frac{\beta_t^2}{2\sigma_t^2\,\alpha_t\,(1-\bar\alpha_t)}\,\big\lVert \epsilon_\theta - \epsilon_\text{ref}\big\rVert^2.
$$

With the DDPM choice $\sigma_t^2 = \beta_t$, $\beta_t = 0.02$, $\alpha_t = 0.98$,
$\bar\alpha_t = 0.5$, and $\lVert\epsilon_\theta - \epsilon_\text{ref}\rVert^2 = 0.25$, the
coefficient is $0.02 / (2 \cdot 0.98 \cdot 0.5) = 0.020408$ and the KL is
$0.005102$.

#### References

- Black, Janner, Du, Kostrikov, Levine. *Training Diffusion Models with
  Reinforcement Learning.* ICLR 2024. [arXiv:2305.13301](https://arxiv.org/abs/2305.13301)
- Fan, Watkins, Du, Liu, Ryu, Boutilier, Abbeel, Ghavamzadeh, Lee, Lee. *DPOK:
  Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models.* NeurIPS
  2023. [arXiv:2305.16381](https://arxiv.org/abs/2305.16381)
- Zhang et al. *Confronting Reward Overoptimization for Diffusion Models: A
  Perspective of Inductive and Primacy Biases* (TDPO-R). ICML 2024.
  [arXiv:2402.08552](https://arxiv.org/abs/2402.08552)

#### Exercise: toy 1-D diffusion policy with DDPO

Build a 1-D DDPM with $T = 5$ steps whose "pretrained" $\epsilon$-network is the exact
optimal denoiser for a two-component Gaussian mixture (the exact posterior mean is
available in closed form, so no pretraining is needed). Add a small trainable MLP
correction $\epsilon_\theta = \epsilon^\star + g_\phi(x_t, t)$ with $g_\phi$ initialized to
zero. Reward: $r(x_0) = \mathbb{1}[x_0 > 1]$ or the smooth version
$\log \sigma(4(x_0 - 1))$.

Checks:

1. `step_logprob(x_prev, mu, sigma)` matches `scipy.stats.norm.logpdf` to `1e-10`.
2. At initialization, the sum of per-step log-ratios
   $\sum_t \log p_\theta / p_\text{old}$ is exactly zero.
3. The DDPO$_\text{SF}$ gradient estimate from 20,000 trajectories, projected on the
   bias of $g_\phi$, agrees in sign with a finite-difference estimate of
   $\nabla \mathbb{E}[r]$ (use common random numbers for the finite difference).
4. After 200 PPO updates, $\Pr[x_0 > 1]$ rises from its initial value (about 0.5 for
   a symmetric mixture at $\pm 1.5$) to above 0.9.

Runtime: under 10 seconds with numpy and a closed-form gradient for a linear $g_\phi$,
or with torch for a 2-layer MLP.

#### Common mistakes

- Using $\eta = 0$ (deterministic DDIM) and then computing a "log-probability." There
  is no density; any number you compute is meaningless.
- Computing $\log p_\theta$ with the *current* network on a stored $x_t$ but the
  *stored* $\mu_\text{old}$ for the ratio denominator, then mixing up which one gets
  the gradient.
- Summing log-probs over latent dimensions but averaging over time, or the reverse.
  Pick one and use it in both the ratio and the KL.
- Applying the terminal reward without per-prompt normalization. Prompts differ in
  difficulty, so raw rewards give the policy a signal about the prompt, not the image.
- Treating the path-KL penalty as the KL on images. The path KL is an upper bound and
  can be much larger (Section E1 has a numeric example where it is 20 times larger).

---

### A2. Reward-weighted regression for diffusion

#### Intuition

The simplest alignment method samples images, scores them, and fine-tunes on the good
ones with the ordinary denoising loss, weighted by reward. This is reward-weighted
regression (RWR): an EM-style method in which the E-step reweights samples and the
M-step is supervised learning. It needs no per-step log-probabilities and works with
any sampler, but it is offline and improves slowly.

#### Key equations

Lee et al. (2023) fine-tune with a reward-weighted denoising loss plus a pretraining
term (equation reconstructed from the paper's description; **UNVERIFIED** exact
notation):

$$
\mathcal{L}(\theta) = \mathbb{E}_{(x, c) \sim \mathcal{D}_\text{model}}\big[-r_\phi(x, c)\,\log p_\theta(x \mid c)\big] + \beta\,\mathbb{E}_{(x, c) \sim \mathcal{D}_\text{pre}}\big[-\log p_\theta(x \mid c)\big].
$$

Here $-\log p_\theta(x \mid c)$ is replaced by its evidence lower bound, which in
practice is the standard noise-prediction loss
$\mathbb{E}_{t, \epsilon}\lVert \epsilon - \epsilon_\theta(x_t, t, c)\rVert^2$.

The exponentiated version has a clean justification. The KL-regularized objective
$\max_p \mathbb{E}_p[r] - \beta\,\mathrm{KL}(p \Vert p_\text{ref})$ has the solution
$p^\star(x) \propto p_\text{ref}(x)\exp(r(x)/\beta)$. Fitting $p_\theta$ to $p^\star$ by
maximum likelihood on samples from $p_\text{ref}$ gives weights
$w = \exp(r/\beta)$, normalized over the batch.

VideoAlign's Flow-RWR is the same idea applied to a rectified-flow model with a
velocity-matching loss in place of noise prediction.

#### Worked example

Four samples have rewards $r = (0.2, 0.5, 0.9, 0.4)$ and $\beta = 0.25$. The weights
$\exp(r/\beta)$ are $(2.23, 7.39, 36.60, 4.95)$. Normalized, they are
$(0.043, 0.144, 0.715, 0.097)$. One sample dominates: RWR with a small $\beta$ behaves
like "fine-tune on the best sample," which reduces diversity fast.

#### References

- Lee, Liu, Ryu, Watkins, Du, Boutilier, Abbeel, Ghavamzadeh, Gu. *Aligning
  Text-to-Image Models using Human Feedback.* 2023.
  [arXiv:2302.12192](https://arxiv.org/abs/2302.12192)
- Peters and Schaal. *Reinforcement learning by reward-weighted regression for
  operational space control.* ICML 2007.
- Liu et al. *Improving Video Generation with Human Feedback* (Flow-RWR, Flow-DPO,
  VideoReward). 2025. [arXiv:2501.13918](https://arxiv.org/abs/2501.13918)

#### Exercise

Using the toy mixture model from A1, sample 5,000 points, compute weights
$\exp(r/\beta)$, and refit a Gaussian mixture by weighted maximum likelihood.

Checks:

1. For $\beta \to \infty$ the refit recovers the original mixture weights within
   `0.02`.
2. The refit mixture weight on the rewarded component matches the analytic
   posterior weight $\pi_k \mathbb{E}_k[e^{r/\beta}] / \sum_j \pi_j \mathbb{E}_j[e^{r/\beta}]$
   within `0.02`.
3. The effective sample size $(\sum w)^2 / \sum w^2$ decreases monotonically as
   $\beta$ decreases.

#### Common mistakes

- Using unnormalized weights with a fixed learning rate. The effective step size then
  changes with the reward scale.
- Forgetting the pretraining term. Without it the model collapses toward a few
  high-reward modes after a few rounds.

---

### A3. Direct reward backpropagation: DRaFT, AlignProp, ReFL

#### Intuition

If the reward model is differentiable (a neural network on pixels), you can skip RL
and backpropagate $\nabla_x r$ through the sampling chain into the parameters, as in
backpropagation through time. This gives low-variance, sample-efficient gradients. The
costs are memory (every step's activations), exploding gradients through long chains,
and fast reward hacking, because the optimizer follows the reward model's gradient
into regions where the reward model is wrong.

#### Key equations

DRaFT maximizes *(verified)*:

$$
J(\theta) = \mathbb{E}_{c \sim p_c,\ x_T \sim \mathcal{N}(0, I)}\big[r(\text{sample}(\theta, c, x_T), c)\big].
$$

The three variants differ in how far the gradient flows back:

- **DRaFT**: full backpropagation through all 50 DDIM steps, with gradient
  checkpointing and LoRA.
- **DRaFT-K**: insert $\operatorname{stop\_grad}(x_t)$ at step $K$ so only the last $K$
  steps receive gradient. Small $K$ (even $K = 1$) often works best; gradient norms
  grow with $K$.
- **DRaFT-LV** ($K = 1$): re-noise the final sample $n$ times,
  $x_1 = \alpha_1 \operatorname{stop\_grad}(x_0) + \sigma_1 \epsilon$, denoise each copy one
  step, and average the reward gradients. The paper uses $n = 2$ and reports about 2x
  faster training than ReFL.

AlignProp is full backpropagation with LoRA and gradient checkpointing, plus
randomized truncation of the backpropagation depth to control memory and
over-optimization.

ReFL (from the ImageReward paper) takes the gradient at a single randomly chosen late
step *(verified)*. It samples to a step $t \in [T_1, T_2] = [1, 10]$ out of $T = 40$,
predicts $\hat x_0$ in one step, and applies:

$$
\mathcal{L}_\text{reward} = \lambda\, \mathbb{E}_{y_i \sim \mathcal{Y}}\big[\phi\big(r(y_i, g_\theta(y_i))\big)\big], \qquad \phi = \text{ReLU},\ \lambda = 10^{-3},
$$

interleaved with the pretraining loss
$\mathcal{L}_\text{pre} = \mathbb{E}\lVert \epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y_i))\rVert^2$.
The ReLU is applied to a negated reward in the original code, so the loss saturates
once the reward exceeds a threshold (**UNVERIFIED** exact sign convention).

#### Worked example: gradient through a linear chain

Take a 1-D deterministic sampler $x_{t-1} = a\,x_t + \theta$ for $T = 3$ steps from
$x_3 = 1$ with $a = 0.9$, and reward $r(x_0) = -(x_0 - 2)^2$.
Then $x_0 = a^3 x_3 + \theta(1 + a + a^2) = 0.729 + 2.71\,\theta$. At $\theta = 0$,
$\partial r / \partial x_0 = -2(0.729 - 2) = 2.542$, and
$\partial x_0 / \partial\theta = 2.71$, so $\partial r/\partial\theta = 6.889$. With
DRaFT-1 you keep only the last step's contribution, $\partial x_0/\partial\theta = 1$,
giving $2.542$. Truncation biases the gradient toward the last steps. In this
example the sign stays correct and only the magnitude shrinks, which is why small $K$
works in practice.

#### References

- Clark, Vicol, Swersky, Fleet. *Directly Fine-Tuning Diffusion Models on
  Differentiable Rewards.* ICLR 2024. [arXiv:2309.17400](https://arxiv.org/abs/2309.17400)
- Prabhudesai, Goyal, Pathak, Fragkiadaki. *Aligning Text-to-Image Diffusion Models
  with Reward Backpropagation* (AlignProp). 2023.
  [arXiv:2310.03739](https://arxiv.org/abs/2310.03739)
- Xu, Liu, Wu, et al. *ImageReward: Learning and Evaluating Human Preferences for
  Text-to-Image Generation* (ImageReward and ReFL). NeurIPS 2023.
  [arXiv:2304.05977](https://arxiv.org/abs/2304.05977)

#### Exercise

In torch, implement the 1-D linear chain and a small nonlinear chain
($x_{t-1} = x_t - h\,f_\theta(x_t, t)$ with a 2-layer MLP, $T = 10$). Reward: a
differentiable log-density of a target region.

Checks:

1. Autograd $\partial r / \partial \theta$ for the linear chain equals the closed form
   $r'(x_0)(1 + a + \dots + a^{T-1})$ to `1e-6`.
2. DRaFT-K gradient equals the full gradient when $K = T$, and equals the last-step
   term when $K = 1$.
3. Gradient norm grows with $K$ for $a > 1$ (an expanding chain) and shrinks for
   $a < 1$.
4. After 300 steps of full backpropagation with no KL term, the sample variance
   drops below 10% of its initial value (a measurable collapse). Adding a
   pretraining loss term keeps it above 50%.

#### Common mistakes

- Backpropagating through a VAE decoder and a large reward model without
  checkpointing, then running out of memory and reducing $T$ instead.
- Declaring success from the reward curve alone. Direct backpropagation reaches
  high reward in few steps and also hacks the reward faster than any other method.
  DRaFT documents collapse to near-identical high-reward images.
- Mixing up $\hat x_0$ (a one-step prediction) with the actual final sample. ReFL
  scores $\hat x_0$, which is blurrier at larger $t$.

---

### A4. Preference optimization without an explicit reward: Diffusion-DPO, D3PO, SPO, Diffusion-KTO

#### Intuition

DPO shows that, for a KL-regularized objective, the optimal policy defines an implicit
reward $\beta \log \pi_\theta / \pi_\text{ref}$. You can then fit preference pairs with a
logistic loss on the difference of implicit rewards. For diffusion models the problem
is that $\log p_\theta(x_0 \mid c)$ is intractable. Diffusion-DPO works around it by
applying DPO to whole *trajectories*, replacing the intractable reverse posterior with
the forward noising process, and using Jensen's inequality to move the expectation over
timesteps outside the log-sigmoid. The result is a loss that compares denoising errors
of the policy and the reference on the winning and losing images.

#### Diffusion-DPO derivation

The paper's derivation proceeds in four steps *(final loss verified; intermediate
steps follow the paper's structure)*.

**Step 1: DPO on chains.** Define a reward on whole chains $R(c, x_{0:T})$ with
$r(c, x_0) = \mathbb{E}_{p_\theta(x_{1:T} \mid x_0, c)}[R(c, x_{0:T})]$. The
KL-regularized optimum over chains is
$p^\star(x_{0:T} \mid c) \propto p_\text{ref}(x_{0:T} \mid c)\exp(R(c, x_{0:T})/\beta)$.
Substituting into the Bradley-Terry likelihood gives:

$$
\mathcal{L}_1(\theta) = -\mathbb{E}_{(c, x_0^w, x_0^l)}\log\sigma\Big(\beta\,\mathbb{E}_{x_{1:T}^w, x_{1:T}^l}\Big[\log\frac{p_\theta(x_{0:T}^w)}{p_\text{ref}(x_{0:T}^w)} - \log\frac{p_\theta(x_{0:T}^l)}{p_\text{ref}(x_{0:T}^l)}\Big]\Big).
$$

**Step 2: approximate the reverse posterior.** Sampling
$x_{1:T} \sim p_\theta(x_{1:T} \mid x_0)$ is intractable, so replace it with the forward
process $q(x_{1:T} \mid x_0)$. Write the chain log-ratio as a sum over $t$ and
write the sum as $T\,\mathbb{E}_t[\cdot]$.

**Step 3: Jensen.** Because $-\log\sigma(\cdot)$ is convex, move $\mathbb{E}_t$ and the
expectation over $x_t$ outside:

$$
\mathcal{L}(\theta) \le -\mathbb{E}_{(c, x_0^w, x_0^l),\, t,\, x_t^w, x_t^l}\log\sigma\Big(-\beta T\Big[\mathrm{KL}(q(x_{t-1}^w \mid x_{0,t}^w) \Vert p_\theta(x_{t-1}^w \mid x_t^w)) - \mathrm{KL}(q(x_{t-1}^w \mid x_{0,t}^w) \Vert p_\text{ref}(x_{t-1}^w \mid x_t^w)) - \big(\text{same terms for } l\big)\Big]\Big).
$$

**Step 4: Gaussian KLs become noise-prediction errors.** Both reverse steps share
variance with the forward posterior, so each KL is a weighted squared error in
$\epsilon$. The final loss is *(verified)*:

$$
\mathcal{L}_\text{Diffusion-DPO}(\theta) = -\mathbb{E}\,\log\sigma\Big(-\beta T\,\omega(\lambda_t)\Big[\big(\lVert\epsilon^w - \epsilon_\theta(x_t^w, t)\rVert_2^2 - \lVert\epsilon^w - \epsilon_\text{ref}(x_t^w, t)\rVert_2^2\big) - \big(\lVert\epsilon^l - \epsilon_\theta(x_t^l, t)\rVert_2^2 - \lVert\epsilon^l - \epsilon_\text{ref}(x_t^l, t)\rVert_2^2\big)\Big]\Big),
$$

where $x_t^\ast = \alpha_t x_0^\ast + \sigma_t \epsilon^\ast$ with
$\epsilon^\ast \sim \mathcal{N}(0, I)$, $t \sim \mathcal{U}(0, T)$, and $\omega(\lambda_t)$
is a weighting function of the log-SNR (constant in practice). The paper uses
$\beta = 2000$ for SD1.5 and $\beta = 5000$ for SDXL. These large values make sense
because $\beta T$ multiplies a squared error that is averaged over latent dimensions.

Reading the loss: the policy is rewarded when it lowers its denoising error on the
winner, relative to the reference, by more than it lowers its error on the loser.
The loss depends on *relative* improvement, so both errors can rise and the loss can
still fall. This produces the likelihood-displacement failure also seen in text DPO.

Implementations differ in constants. The `diffusers` example script averages the
squared error over latent dimensions and uses a factor of $-\tfrac12\beta$ inside the
sigmoid (**UNVERIFIED**, check the script you follow). Keep constants consistent
between the course's reference solution and its tests.

#### Worked example

Take $\beta T \omega = 1$ and the following per-sample mean squared errors:

| | Policy $\theta$ | Reference |
|---|---|---|
| Winner $w$ | 0.30 | 0.50 |
| Loser $l$ | 0.60 | 0.40 |

The inner term is $(0.30 - 0.50) - (0.60 - 0.40) = -0.40$. The loss is
$-\log\sigma(-1 \cdot (-0.40)) = -\log\sigma(0.40) = 0.5130$. At initialization
($\theta = \text{ref}$) the inner term is 0 and the loss is $\log 2 = 0.6931$. A test
can check both values.

#### D3PO

D3PO (Yang et al.) takes a different route. It treats each denoising step as an MDP
step and applies the DPO loss *per step* to winner and loser trajectories that share
the same initial noise *(verified)*:

$$
\mathcal{L}_i(\theta) = -\mathbb{E}_{(s_i, \sigma_w, \sigma_l)}\Big[\log\sigma\Big(\beta\log\frac{\pi_\theta(a_i^w \mid s_i^w)}{\pi_\text{ref}(a_i^w \mid s_i^w)} - \beta\log\frac{\pi_\theta(a_i^l \mid s_i^l)}{\pi_\text{ref}(a_i^l \mid s_i^l)}\Big)\Big], \quad i \in [0, T-1].
$$

It assumes the final-image preference transfers to every intermediate
state-action pair. Unlike Diffusion-DPO, D3PO needs the *actual sampled
trajectories* (the stored $x_t$ values), so it is on-policy or near on-policy data, not
an offline dataset of images.

#### SPO: step-aware preference optimization

SPO (Liang et al.) argues that transferring one image-level label to every step is
wrong: early steps decide layout, late steps decide fine detail. At each step it
samples a pool of candidates from a shared $x_t$, ranks them with a *step-aware*
preference model that scores noisy intermediate latents, and applies a DPO-style loss
to the chosen pair at that step only. The trajectory then continues from one of the
candidates (step-wise resampling). The published version is titled *Aesthetic
Post-Training Diffusion Models from Generic Preferences with Step-by-step Preference
Optimization* (CVPR 2025).

#### Diffusion-KTO

Diffusion-KTO (Li et al.) uses per-image binary labels (desirable or undesirable)
instead of pairs. It maximizes a prospect-theory utility of the per-step implicit
reward *(verified)*:

$$
\max_{\pi_\theta}\ \mathbb{E}_{x_0 \sim \mathcal{D},\, t}\Big[U\Big(w(x_0)\Big(\beta\log\frac{\pi_\theta(x_{t-1} \mid x_t)}{\pi_\text{ref}(x_{t-1} \mid x_t)} - Q_\text{ref}\Big)\Big)\Big],
$$

with $U = \sigma$ (the logistic function), $w(x_0) = +1$ for desirable and $-1$ for
undesirable images, and $Q_\text{ref}$ a reference point set to a batch estimate of
the KL between policy and reference. As in Diffusion-DPO, the log-ratio is replaced by
a difference of denoising errors.

#### References

- Wallace, Dang, Rafailov, Zhou, Lou, Purushwalkam, Ermon, Xiong, Joty, Naik.
  *Diffusion Model Alignment Using Direct Preference Optimization.* CVPR 2024.
  [arXiv:2311.12908](https://arxiv.org/abs/2311.12908)
- Yang, Tao, Lyu, Ge, Li, Chen, Shen, Zhu, Li. *Using Human Feedback to Fine-tune
  Diffusion Models without Any Reward Model* (D3PO). CVPR 2024.
  [arXiv:2311.13231](https://arxiv.org/abs/2311.13231)
- Liang, Yuan, Gu, Chen, Hang, Cheng, Li, Zheng. *Aesthetic Post-Training Diffusion
  Models from Generic Preferences with Step-by-step Preference Optimization* (SPO).
  CVPR 2025. [arXiv:2406.04314](https://arxiv.org/abs/2406.04314)
- Li, Kallidromitis, Gokul, Kato, Kozuka. *Aligning Diffusion Models by Optimizing
  Human Utility* (Diffusion-KTO). NeurIPS 2024 (**UNVERIFIED** venue).
  [arXiv:2404.04465](https://arxiv.org/abs/2404.04465)
- Rafailov et al. *Direct Preference Optimization.* NeurIPS 2023.
  [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

#### Exercise: Diffusion-DPO on a 2-D toy

Use a 2-D DDPM with a small MLP $\epsilon_\theta$, pretrained for about 2,000 steps on a
mixture of 4 Gaussians (or load pretrained weights shipped with the exercise to stay
under the time budget). Create synthetic preferences: of two samples, the one with
higher $r(x) = -\lVert x - (2, 2)\rVert$ wins.

Checks:

1. `diffusion_dpo_loss` returns exactly $\log 2$ when the policy equals the reference.
2. On the hand-computed example (errors 0.30, 0.50, 0.60, 0.40), the loss is
   `0.5130` to `1e-4`.
3. The gradient of the loss with respect to $\epsilon_\theta(x_t^w)$ points toward
   $\epsilon^w$ (negative inner product with $\epsilon_\theta - \epsilon^w$).
4. After 300 updates, the fraction of samples in the rewarded quadrant rises by at
   least 15 percentage points, while the per-step KL to the reference stays below a
   threshold chosen from the reference solution.
5. Optional: log $\lVert\epsilon^w - \epsilon_\theta\rVert^2$ and show that it can
   *increase* while the loss decreases.

#### Common mistakes

- Using *different* noise $\epsilon$ or timestep $t$ for the policy and the reference
  on the same image. The difference of errors is then dominated by noise.
- Using different $t$ for winner and loser. The paper samples one $t$ per pair.
- Choosing $\beta$ by analogy with text DPO ($\beta \approx 0.1$). Diffusion-DPO needs
  $\beta$ in the thousands because of the scale of the squared error.
- Forgetting that Diffusion-DPO is a bound. Lowering the loss doesn't guarantee lowering
  the true chain-level DPO loss.

---

### A5. Flow models with GRPO: Flow-GRPO, DanceGRPO, MixGRPO, and successors

#### Intuition

Rectified-flow and flow-matching models (SD3, FLUX, Wan, HunyuanVideo) sample by
integrating a deterministic ODE $\mathrm{d}x = v_\theta(x, t)\,\mathrm{d}t$. A
deterministic sampler has no per-step density, so policy gradients don't apply. The
fix is to replace the ODE with an SDE that has the *same marginal distribution at
every time* $t$. The SDE adds Gaussian noise at each step and corrects the drift with
a score term so the marginals don't change. Each discretized step is then a Gaussian
with a known mean and variance, and the whole GRPO machinery from LLMs applies: sample
a group of $G$ images per prompt, normalize rewards within the group, and apply a
clipped per-step ratio loss.

#### Key equations

Flow-GRPO uses the convention $x_t = (1-t)\,x_0 + t\,\epsilon$, where $t = 1$ is noise
and $t = 0$ is data, and sampling runs from $t = 1$ to $t = 0$. The equations are
*(verified)* as follows.

The marginal-preserving SDE family is:

$$
\mathrm{d}x_t = \Big[v_t(x_t) - \frac{\sigma_t^2}{2}\nabla\log p_t(x_t)\Big]\mathrm{d}t + \sigma_t\,\mathrm{d}w.
$$

For rectified flow, the score is a function of the velocity. From
$x_t = \hat x_0 + t\,v$ and $\hat\epsilon = x_t + (1-t)v$, the score is
$\nabla\log p_t(x_t) = -\hat\epsilon / t = -\big(x_t + (1-t)v_t(x_t)\big)/t$. Substituting:

$$
\mathrm{d}x_t = \Big[v_t(x_t) + \frac{\sigma_t^2}{2t}\big(x_t + (1-t)\,v_t(x_t)\big)\Big]\mathrm{d}t + \sigma_t\,\mathrm{d}w.
$$

The Euler-Maruyama step used for sampling and for the policy density is:

$$
x_{t+\Delta t} = x_t + \Big[v_\theta(x_t, t) + \frac{\sigma_t^2}{2t}\big(x_t + (1-t)\,v_\theta(x_t, t)\big)\Big]\Delta t + \sigma_t\sqrt{\Delta t}\,\varepsilon, \qquad \sigma_t = a\sqrt{\frac{t}{1-t}}.
$$

Sampling runs toward $t = 0$, so $\Delta t$ is negative in this equation and
$\sqrt{\Delta t}$ means $\sqrt{|\Delta t|}$. The official code writes `dt = sigma_next - sigma`
(negative) and `std = sigma_t * sqrt(-dt)` (**UNVERIFIED** variable names). A cleaner
form for teaching, with a positive step size $h$, is:

$$
x_{t-h} = x_t - h\Big[v_\theta + \frac{\sigma_t^2}{2t}\big(x_t + (1-t)v_\theta\big)\Big] + \sigma_t\sqrt{h}\,\varepsilon.
$$

So each step is the policy
$\pi_\theta(x_{t-h} \mid x_t, c) = \mathcal{N}\big(\mu_\theta(x_t, t),\ \sigma_t^2 h\, I\big)$.
The Flow-GRPO paper uses $a = 0.7$.

The objective is GRPO with a per-step clipped ratio and a KL penalty *(verified)*:

$$
\mathcal{J}(\theta) = \mathbb{E}\Big[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T}\sum_{t=0}^{T-1}\min\Big(r_t^i(\theta)\hat A_t^i,\ \operatorname{clip}\big(r_t^i(\theta), 1-\varepsilon, 1+\varepsilon\big)\hat A_t^i\Big) - \beta\, D_\text{KL}(\pi_\theta \Vert \pi_\text{ref})\Big],
$$

$$
r_t^i(\theta) = \frac{p_\theta(x_{t-1}^i \mid x_t^i, c)}{p_{\theta_\text{old}}(x_{t-1}^i \mid x_t^i, c)}, \qquad \hat A_t^i = \frac{R(x_0^i, c) - \operatorname{mean}\big(\{R(x_0^j, c)\}_{j=1}^G\big)}{\operatorname{std}\big(\{R(x_0^j, c)\}_{j=1}^G\big)}.
$$

Every step in a trajectory gets the same advantage. Because the two policies share
variance, the per-step KL is closed form *(verified)*:

$$
D_\text{KL}(\pi_\theta \Vert \pi_\text{ref}) = \frac{\Delta t}{2}\Big(\frac{\sigma_t(1-t)}{2t} + \frac{1}{\sigma_t}\Big)^2\big\lVert v_\theta(x_t, t) - v_\text{ref}(x_t, t)\big\rVert^2.
$$

To derive it, note the means differ by
$\Delta\mu = \big(1 + \tfrac{\sigma_t^2(1-t)}{2t}\big)\Delta t\,\Delta v$ and the variance is
$\sigma_t^2\Delta t$, so
$\mathrm{KL} = \lVert\Delta\mu\rVert^2 / (2\sigma_t^2\Delta t)$.

**Denoising reduction.** Flow-GRPO trains with $T = 10$ sampling steps but evaluates
with $T = 40$. Training rollouts only need to be good enough to rank within a group.
The paper reports more than a 4x speedup without loss of final reward.

**Results reported.** GenEval accuracy on SD3.5-M rises from 63% to 95%, and visual
text rendering accuracy rises from 59% to 92%.

#### Worked example: Flow-GRPO numbers

Advantages: a group of $G = 4$ with rewards $(0.2, 0.5, 0.9, 0.4)$ has mean $0.5$ and
population standard deviation $0.2550$. The advantages are
$(-1.177, 0, 1.569, -0.392)$.

KL: at $t = 0.6$, $\Delta t = 0.1$, $a = 0.7$, you get
$\sigma_t = 0.7\sqrt{1.5} = 0.8573$. The bracket is
$0.8573 \cdot 0.4 / 1.2 + 1/0.8573 = 0.2858 + 1.1664 = 1.4522$, and its square is
$2.1089$. With $\lVert\Delta v\rVert = 0.2$, the KL is
$0.05 \cdot 2.1089 \cdot 0.04 = 0.004218$. A direct computation from
$\Delta\mu^2/(2\sigma_t^2\Delta t)$ gives the same number.

#### DanceGRPO

DanceGRPO (Xue et al.) applies the same recipe across two paradigms (diffusion and
rectified flow), three tasks (text-to-image, text-to-video, image-to-video), four base
models (Stable Diffusion, FLUX, HunyuanVideo, SkyReels-I2V), and five reward types
(image aesthetics, video aesthetics, text-image alignment, video motion quality, and a
binary reward). It uses the same kind of ODE-to-SDE step and reports that sharing the
initial noise within a group, and training on a subset of timesteps, matter for
stability. It reports gains of up to 181% on benchmarks including HPS-v2.1, CLIP
Score, VideoAlign, and GenEval.

#### MixGRPO

MixGRPO (Li et al.) notes that making every step stochastic, and computing ratios at
every step, is expensive. It uses SDE sampling only inside a *sliding window* of
timesteps and deterministic ODE sampling outside it. Only steps in the window are
optimized. The window moves from high noise to low noise during training. The paper
reports about 50% less training time than DanceGRPO, and a MixGRPO-Flash variant with
71% less. The ODE steps outside the window don't contribute log-probabilities, and
their use of higher-order solvers is allowed because they don't need a density.

#### Successors from 2025 to 2026

The following papers refine the flow-GRPO recipe. Each entry names the problem it
targets.

- **Credit assignment across time: TempFlow-GRPO**
  ([arXiv:2508.04324](https://arxiv.org/abs/2508.04324)). Branches trajectories at
  chosen timesteps to get per-step "process" rewards without an intermediate reward
  model, and weights the loss by the exploration potential of each timestep.
- **Noise artifacts from the SDE: Coefficients-Preserving Sampling (CPS)**
  ([arXiv:2509.05952](https://arxiv.org/abs/2509.05952)). Shows that the Flow-GRPO SDE
  injects excess noise that shows up as artifacts in images the reward model scores.
  It proposes a DDIM-inspired stochastic step that preserves the signal and noise
  coefficients.
- **Reward hacking from group normalization: Pref-GRPO**
  ([arXiv:2508.20751](https://arxiv.org/abs/2508.20751)). Identifies an *illusory
  advantage*: when group rewards are tightly clustered, dividing by a tiny standard
  deviation amplifies meaningless differences. It replaces pointwise scores with
  win rates from a pairwise preference reward model.
- **Likelihood-free training: DiffusionNFT**
  ([arXiv:2509.16117](https://arxiv.org/abs/2509.16117), ICLR 2026 Oral). Trains on the
  *forward* process with a flow-matching loss, so no SDE sampler or per-step density
  is needed. Covered next.
- **Chunk-level policy optimization**
  ([arXiv:2510.21583](https://arxiv.org/abs/2510.21583)). Groups consecutive timesteps
  into chunks as the unit of policy optimization (**UNVERIFIED** details).
- **Other 2026 work, abstracts only (UNVERIFIED details):** UniGRPO for
  reasoning-driven generation ([arXiv:2603.23500](https://arxiv.org/abs/2603.23500)),
  V-GRPO ([arXiv:2604.23380](https://arxiv.org/abs/2604.23380)), MAR-GRPO for AR-diffusion
  hybrids ([arXiv:2604.06966](https://arxiv.org/abs/2604.06966)), JAGG
  ([arXiv:2607.17572](https://arxiv.org/abs/2607.17572)), and GeMPO
  ([arXiv:2603.10250](https://arxiv.org/abs/2603.10250)).

The DiffusionNFT objective is *(verified)*:

$$
\mathcal{L}(\theta) = \mathbb{E}_{c,\ x_0 \sim \pi^\text{old}(\cdot \mid c),\ t}\Big[r\,\big\lVert v_\theta^+(x_t, c, t) - v\big\rVert_2^2 + (1 - r)\,\big\lVert v_\theta^-(x_t, c, t) - v\big\rVert_2^2\Big],
$$

$$
v_\theta^+ = (1-\beta)\,v^\text{old} + \beta\,v_\theta, \qquad v_\theta^- = (1+\beta)\,v^\text{old} - \beta\,v_\theta,
$$

$$
r(x_0, c) = \frac12 + \frac12\operatorname{clip}\Big[\frac{r^\text{raw}(x_0, c) - \mathbb{E}_{\pi^\text{old}(\cdot \mid c)}\, r^\text{raw}(x_0, c)}{Z_c},\ -1,\ 1\Big].
$$

Here $v$ is the flow-matching target velocity for the noised clean sample $x_0$. The
loss pulls an implicit "positive" policy toward good samples and an implicit
"negative" policy toward bad ones; both are linear in $v_\theta$. For example, with raw
reward $0.8$, group mean $0.6$, and $Z_c = 0.4$, you get
$r = 0.5 + 0.5 \cdot 0.5 = 0.75$.

#### References

- Liu, Liu, et al. *Flow-GRPO: Training Flow Matching Models via Online RL.* 2025.
  [arXiv:2505.05470](https://arxiv.org/abs/2505.05470)
- Xue et al. *DanceGRPO: Unleashing GRPO on Visual Generation.* 2025.
  [arXiv:2505.07818](https://arxiv.org/abs/2505.07818)
- Li et al. *MixGRPO: Unlocking Flow-based GRPO Efficiency with Mixed ODE-SDE.* 2025;
  listed as an ECCV 2026 poster (**UNVERIFIED** venue).
  [arXiv:2507.21802](https://arxiv.org/abs/2507.21802)
- Song et al. *Score-Based Generative Modeling through Stochastic Differential
  Equations* (the probability-flow ODE and the SDE family). ICLR 2021.
  [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
- Shao et al. *DeepSeekMath* (GRPO). 2024.
  [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)

#### Exercise: ODE-to-SDE conversion preserves marginals

Use a 1-D rectified flow whose data distribution is $\mathcal{N}(m, s^2)$ with
$m = 2$ and $s = 0.5$. The exact velocity is available in closed form. For
$x_t = (1-t)x_0 + t\epsilon$, the pair $(x_0, \epsilon, x_t)$ is jointly Gaussian, so:

$$
v(x, t) = \mathbb{E}[\epsilon - x_0 \mid x_t = x] = \frac{t - (1-t)s^2}{(1-t)^2 s^2 + t^2}\big(x - (1-t)m\big) - m.
$$

Students implement the SDE step and simulate 200,000 particles with 500 steps for
$a \in \{0, 0.3, 0.7, \sqrt2\}$.

Checks:

1. For every $a$, the final sample mean is $2.00 \pm 0.02$ and the standard deviation
   is $0.50 \pm 0.02$. The reference simulation gives mean 2.000, 2.001, 2.000, and
   2.002, with standard deviations between 0.499 and 0.501.
2. For $a = 0$, the correlation between initial noise and final sample is 1.0. For
   $a = \sqrt2$ it is $0.00 \pm 0.02$, which is the memoryless schedule from A6.
3. Implement `step_logprob` and `step_kl`; check that the KL formula from the paper
   matches $\Delta\mu^2 / (2\sigma^2 h)$ to `1e-12` at random $(t, h, a, \Delta v)$.
4. Run Flow-GRPO on a 2-D toy (a learned velocity MLP, target reward = log-density of
   one mixture component) and check that the reward rises while the average per-step
   KL stays below a bound.

Runtime: checks 1 to 3 take about 3 seconds in numpy.

Implementation note: $\sigma_t = a\sqrt{t/(1-t)}$ diverges at $t = 1$. Start the grid
at $t = 1 - 10^{-3}$ or cap $\sigma_t$. With a cap too close to 1 (for example
$1 - 10^{-4}$) and a coarse grid, the reference run with $a = 0.7$ gave a standard
deviation of 0.523 instead of 0.500. The error came from the Euler-Maruyama step,
not from the math. Tests must set the grid, or they will fail for correct code.

#### Exercise: the illusory advantage

Compute GRPO advantages for rewards $(0.801, 0.802, 0.800, 0.803)$. The advantages are
$(-0.447, 0.447, -1.342, 1.342)$: the same magnitude as for rewards that differ by
0.5. Check that adding $\epsilon_\text{std} = 0.01$ to the denominator shrinks them
below 0.15, and that Pref-GRPO-style win rates within the group are $(1/3, 2/3, 0, 1)$.

#### Common mistakes

- Using the SDE for training rollouts but computing $\pi_\text{old}$ with a different
  $a$, step count, or CFG scale than was used for sampling. The ratio at the first
  update is then not 1.
- Forgetting that classifier-free guidance changes the velocity. The policy is the
  *guided* velocity; both the rollout and the ratio must use it.
- Dividing by the group standard deviation with no floor. See the illusory advantage.
- Sampling a different initial noise for each group member when the paper shares
  it, or the reverse. Report which one the exercise uses.
- Letting $\sigma_t$ blow up near $t = 1$.

---

### A6. The stochastic optimal control view: Adjoint Matching

#### Intuition

Reward fine-tuning with a KL penalty has a known target: the tilted distribution
$p^\star(x) \propto p^\text{base}(x)\exp(r(x))$. You can view fine-tuning as a
stochastic optimal control (SOC) problem: add a control $u$ to the base SDE drift, pay
$\tfrac12\lVert u\rVert^2$ (this equals the path KL), and receive $r$ at the end. Domingo-Enrich
et al. show two results:

1. The optimal controlled process samples $p^\star$ only if the base process is
   *memoryless*, meaning the final sample is independent of the initial noise.
   Otherwise the solution is tilted by an unwanted value function of the initial noise.
   This is the "initial value function bias."
2. The optimal control can be learned by regression onto a "lean adjoint" that you
   compute by integrating a linear ODE backward along sampled trajectories. This
   avoids high-variance policy gradients and avoids differentiating through the
   sampler with respect to parameters.

#### Key equations

Adjoint Matching uses the convention $\bar X_t = \beta_t \bar X_0 + \alpha_t \bar X_1$,
where $\bar X_0$ is noise and $\bar X_1$ is data *(verified)*.

The SOC problem is:

$$
\min_u\ \mathbb{E}\Big[\int_0^1 \Big(\tfrac12\lVert u(X_t^u, t)\rVert^2 + f(X_t^u, t)\Big)\mathrm{d}t + g(X_1^u)\Big], \quad \mathrm{d}X_t^u = \big(b(X_t^u, t) + \sigma(t)u(X_t^u, t)\big)\mathrm{d}t + \sigma(t)\,\mathrm{d}B_t,
$$

with $f = 0$ and $g = -r$ for reward fine-tuning.

The optimal path measure tilts the base path measure:
$\mathbb{P}^\star(X) = \mathbb{P}^\text{base}(X)\exp\big(r(X_1) + V(X_0, 0)\big)$, where
$V(x, t) = -\log\mathbb{E}_\text{base}[\exp r(X_1) \mid X_t = x]$. Marginalizing gives
*(verified)*:

$$
p^\star(X_1) \propto \int p^\text{base}(X_0, X_1)\exp\big(r(X_1) + V(X_0, 0)\big)\,\mathrm{d}X_0.
$$

If $p^\text{base}(X_0, X_1) = p^\text{base}(X_0)\,p^\text{base}(X_1)$ (memoryless), then
$\exp V(X_0, 0)$ is a constant and $p^\star(X_1) \propto p^\text{base}(X_1)\exp r(X_1)$ as
desired. For an ODE sampler, $X_1$ is a deterministic function of $X_0$, so the bias is
maximal.

The memoryless noise schedule for flow matching is *(verified)*:

$$
\sigma(t) = \sqrt{2\eta_t}, \qquad \eta_t = \beta_t\Big(\frac{\dot\alpha_t}{\alpha_t}\beta_t - \dot\beta_t\Big),
$$

and the drift of the fine-tuned process becomes
$b(x, t) + \sigma(t)u(x, t) = 2v^\text{finetune}(x, t) - \frac{\dot\alpha_t}{\alpha_t}x$.

The lean adjoint ODE, integrated backward from $t = 1$ along a sampled trajectory, is
*(verified form; sign of terminal condition follows from $g = -r$)*:

$$
\frac{\mathrm{d}\tilde a(t)}{\mathrm{d}t} = -\tilde a(t)^\top\nabla_x b(X_t, t), \qquad \tilde a(1) = \nabla_x g(X_1) = -\nabla_x r(X_1).
$$

The Adjoint Matching loss is:

$$
\mathcal{L}_\text{Adj-Match}(u; X) = \frac12\int_0^1 \big\lVert u(X_t, t) + \sigma(t)^\top\tilde a(t; X)\big\rVert^2\,\mathrm{d}t,
$$

and for flow matching with the memoryless schedule it becomes:

$$
\mathcal{L} = \frac12\int_0^1\Big\lVert\sqrt{\tfrac{2}{\eta_t}}\big(v^\text{finetune}(X_t, t) - v^\text{base}(X_t, t)\big) + \sigma(t)\,\tilde a(t; X)\Big\rVert^2\mathrm{d}t.
$$

The factor follows from $u = (b^\text{ft} - b^\text{base})/\sigma = 2(v^\text{ft} - v^\text{base})/\sqrt{2\eta_t}$.
Trajectories are sampled from the current fine-tuned model with gradients stopped,
and $\nabla_x b$ uses the *base* drift in the lean adjoint, which is what makes it
"lean."

**Some papers present the terminal condition as $\tilde a(1) = \nabla r$ with the loss
sign flipped.** Check sign consistency with a unit test (the optimal $u$ must point up
the reward gradient) rather than trusting any single transcription, including this one.

#### Connecting Flow-GRPO to the memoryless schedule

This cross-check is derived here, not taken from either paper, so treat it as a
worked exercise. For rectified flow in the Adjoint Matching convention,
$\alpha_t = t$ and $\beta_t = 1 - t$, so:

$$
\eta_t = (1-t)\Big(\frac{1-t}{t} + 1\Big) = \frac{1-t}{t}, \qquad \sigma(t) = \sqrt{\frac{2(1-t)}{t}}.
$$

Map to the Flow-GRPO convention with $t_\text{FG} = 1 - t$: the Flow-GRPO noise level
is $a\sqrt{t_\text{FG}/(1 - t_\text{FG})} = a\sqrt{(1-t)/t}$. The two agree when $a = \sqrt2$.
So Flow-GRPO's $a = 0.7$ is about half the memoryless noise level. It is a trade-off:
less noise gives cleaner rollouts, but the sampler keeps memory of the initial noise.
The simulation in the A5 exercise confirms this: at $a = \sqrt2$ the noise-to-sample
correlation is $-0.002$; at $a = 0.7$ it is $-0.18$; at $a = 0$ it is $1.0$.

For a policy-gradient method with a KL penalty on the path, the memoryless argument
also applies: the KL-regularized optimum on paths gives the tilted marginal only when
the reference process is memoryless.

#### References

- Domingo-Enrich, Drozdzal, Karrer, Chen. *Adjoint Matching: Fine-tuning Flow and
  Diffusion Generative Models with Memoryless Stochastic Optimal Control.* ICLR 2025.
  [arXiv:2409.08861](https://arxiv.org/abs/2409.08861)
- Uehara et al. *Fine-Tuning of Continuous-Time Diffusion Models as Entropy-Regularized
  Control.* 2024. [arXiv:2402.15194](https://arxiv.org/abs/2402.15194)
  (**UNVERIFIED** ID; check before citing)
- Tang. *Fine-tuning of diffusion models via stochastic control: entropy
  regularization and beyond.* 2024. (**UNVERIFIED** citation details)

#### Exercise: tilted Gaussian by Adjoint Matching

Base: the 1-D rectified flow with data $\mathcal{N}(0, 1)$ in the Adjoint Matching
convention (data at $t = 1$). Reward: $r(x) = \lambda x$ with $\lambda = 1$. The tilted
target is $\mathcal{N}(\lambda, 1)$, which is analytic. Parameterize
$v^\text{ft} = v^\text{base} + w_1(t)\,x + w_0(t)$ with a small MLP in $t$.

Checks:

1. The memoryless $\sigma(t)$ implementation matches $\sqrt{2(1-t)/t}$.
2. For $r(x) = \lambda x$ and a linear base drift, the lean adjoint has a closed form.
   Compare numerical backward integration against it to `1e-4`.
3. After training, the final samples have mean $1.00 \pm 0.05$ and standard deviation
   $1.00 \pm 0.05$.
4. Negative control: train the same model with the ODE ($\sigma \approx 0$) and
   reward-gradient backpropagation plus a KL term. Samples won't match
   $\mathcal{N}(1, 1)$. Record the mean and variance mismatch.

#### Common mistakes

- Mixing the three time conventions from the table at the top of this document.
  Unit-test every formula against a Gaussian case with known answers.
- Using the memoryless SDE for fine-tuning but evaluating with the ODE. The adjusted
  velocity field defines both, so this is valid, but you must use the fine-tuned
  $v$ in both.
- Using $\nabla_x b$ of the *fine-tuned* model in the adjoint. The lean adjoint uses
  the base drift.

---

### A7. Reward models for images and reward hacking

#### Intuition

Image rewards fall into four families:

- **Preference models** trained on human pairwise choices: PickScore, HPSv2, HPSv3,
  and ImageReward.
- **Aesthetic predictors**: a small head on CLIP embeddings, trained on 1-to-10
  ratings. The LAION aesthetic predictor is the common one.
- **VQA-based alignment scores**: ask a VLM whether the image shows the prompt, and use
  $P(\text{Yes})$. VQAScore is the common one.
- **Rule-based verifiable rewards**: object counts from a detector (GenEval), OCR
  accuracy for text rendering, and JPEG compressibility.

Each has its own blind spots, and RL finds them.

#### Key equations

PickScore and HPSv2 fine-tune CLIP-H with a softmax over the two images in a pair. With
$s(x, y) = \tau\,E_\text{txt}(x)\cdot E_\text{img}(y)$:

$$
\hat p_i = \frac{\exp s(x, y_i)}{\exp s(x, y_1) + \exp s(x, y_2)}, \qquad \mathcal{L} = \sum_{i=1}^2 p_i\big(\log p_i - \log\hat p_i\big),
$$

with human label $p = (1, 0)$, $(0, 1)$, or $(0.5, 0.5)$ for a tie (PickScore form;
**UNVERIFIED** exact tie handling for HPSv2).

ImageReward trains a BLIP-based scorer $f_\theta$ with a Bradley-Terry loss over all
ranked pairs from a $k$-way ranking:

$$
\mathcal{L} = -\mathbb{E}_{(T, x_i, x_j) \sim \mathcal{D}}\big[\log\sigma\big(f_\theta(T, x_i) - f_\theta(T, x_j)\big)\big], \quad x_i \succ x_j.
$$

The ImageReward dataset has 136,892 comparison pairs over 8,878 prompts *(verified)*.

VQAScore is *(verified)*:

$$
\text{VQAScore}(\text{image}, \text{text}) = P\big(\text{"Yes"} \mid \text{image},\ \text{"Does this figure show '\{text\}'? Please answer yes or no."}\big).
$$

The paper uses a CLIP-FlanT5 model. The suffix "Please answer yes or no." is part of
the published template (**UNVERIFIED** exact wording). Example: if the yes and no
logits are 2.0 and 0.5, the two-way softmax gives $\sigma(1.5) = 0.8176$.

HPSv3 uses a VLM backbone, the HPDv3 dataset (1.08M text-image pairs and 1.17M
annotated comparisons), and an uncertainty-aware ranking loss *(verified at abstract
level)*. The ranking loss models each score as a Gaussian and the win probability as
$\Phi\big((\mu_a - \mu_b)/\sqrt{\sigma_a^2 + \sigma_b^2}\big)$ (**UNVERIFIED** exact form).

#### Reward hacking catalog

The following failures are documented in the literature:

- **Aesthetic score.** DDPO fine-tuning drifts toward saturated, stylized, and
  cartoon-like images, and prompt adherence drops.
- **JPEG compressibility.** The policy produces flat, low-detail images with blank
  backgrounds. This is a useful teaching reward because it's cheap and clearly hackable.
- **Direct backpropagation.** DRaFT reports collapse to nearly identical high-reward
  images.
- **Pointwise scores with GRPO.** Pref-GRPO reports scores rising while image quality
  degrades, driven by the illusory advantage.
- **Timestep weighting in Flow-DPO.** VideoAlign found that a timestep-dependent
  $\beta_t = \beta(1-t)^2$ causes reward hacking, and that a constant $\beta$ works
  better.
- **Overoptimization in diffusion RL.** TDPO-R links it to a mismatch with the
  temporal inductive bias of denoising and to primacy bias in the critic.

The standard mitigations are a KL penalty to the reference, early stopping on a
held-out reward, multi-reward mixtures (for example, an aesthetic score plus an
alignment score), pairwise rewards, and evaluation with a *different* reward model from
the one used in training.

#### References

- Kirstain et al. *Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image
  Generation* (PickScore). NeurIPS 2023. [arXiv:2305.01569](https://arxiv.org/abs/2305.01569)
- Wu et al. *Human Preference Score v2.* 2023. [arXiv:2306.09341](https://arxiv.org/abs/2306.09341)
- Ma, Wu, et al. *HPSv3: Towards Wide-Spectrum Human Preference Score.* ICCV 2025.
  [arXiv:2508.03789](https://arxiv.org/abs/2508.03789)
- Xu et al. *ImageReward.* NeurIPS 2023. [arXiv:2304.05977](https://arxiv.org/abs/2304.05977)
- Lin, Pathak, Li, Li, Xia, Neubig, Zhang, Ramanan. *Evaluating Text-to-Visual
  Generation with Image-to-Text Generation* (VQAScore). ECCV 2024.
  [arXiv:2404.01291](https://arxiv.org/abs/2404.01291)
- Ghosh, Hajishirzi, Schmidt. *GenEval.* NeurIPS 2023 Datasets and Benchmarks.
  [arXiv:2310.11513](https://arxiv.org/abs/2310.11513)
- Schuhmann et al. LAION aesthetic predictor (a CLIP ViT-L/14 embedding with an MLP
  head). Released on GitHub; no paper (**UNVERIFIED** training details).
- Survey: *Alignment and Safety of Diffusion Models via Reinforcement Learning and
  Reward Modeling: A Survey.* 2025. [arXiv:2505.17352](https://arxiv.org/abs/2505.17352)

#### Exercise: a hackable reward

Train a 2-D flow policy with Flow-GRPO against a learned reward model: an MLP fitted
by Bradley-Terry on 500 synthetic preference pairs drawn from a known true reward
$r^\ast$. Keep $r^\ast$ hidden from the policy.

Checks:

1. The Bradley-Terry fit reaches at least 85% pairwise accuracy on held-out pairs.
2. As training runs, the learned reward rises monotonically. The true reward rises and
   then falls: its peak must occur before the last checkpoint (Goodhart curve).
3. With a KL coefficient chosen from a grid, the true reward at the end is higher than
   with no KL.
4. Evaluating with a second reward model trained on different pairs detects the
   hacking earlier than the training reward does.

#### Common mistakes

- Reporting the training reward model's score as the evaluation metric.
- Using GenEval-style detector rewards without checking the detector on the fine-tuned
  model's outputs. A detector can be fooled by texture patterns.
- Mixing rewards on different scales without normalization. One reward then dominates.

---

### A8. Video generation RL

#### Intuition

Video models are flow or diffusion models over spatiotemporal latents, so the same
algorithms apply. The new difficulties are cost (a rollout is a whole video), reward
dimensionality (visual quality, motion quality, and text alignment can conflict), and
new hacks such as static videos that score well on visual quality or artificial jitter
that scores well on motion.

#### Key results

VideoAlign (Liu et al., 2025) *(verified)* makes the following contributions:

- A dataset of 182,000 annotated triplets from 12 text-to-video models, with labels on
  three dimensions: visual quality (VQ), motion quality (MQ), and text alignment (TA).
- **VideoReward**, a multi-dimensional reward model on Qwen2-VL-2B, trained with a
  Bradley-Terry model extended to ties. It uses separate tokens per dimension to
  prevent context leakage between dimensions.
- **Flow-DPO**, which is Diffusion-DPO with velocity-prediction errors. A constant
  $\beta$ outperforms the timestep-dependent $\beta_t = \beta(1-t)^2$, which causes
  reward hacking.
- **Flow-RWR**, reward-weighted regression for flow models.
- **Flow-NRG**, inference-time reward guidance with a reward model on noisy latents.
  Users can weight the three dimensions at inference.

A standard model for ties is the Rao-Kupper extension of Bradley-Terry with a tie
parameter $\theta \ge 1$:

$$
P(A \succ B) = \frac{p_A}{p_A + \theta p_B}, \qquad P(A = B) = \frac{(\theta^2 - 1)\,p_A p_B}{(p_A + \theta p_B)(p_B + \theta p_A)},
$$

with $p = e^{r}$. Whether VideoAlign uses exactly this parameterization is
**UNVERIFIED**. Worked example: $r_A = 1$, $r_B = 0$, $\theta = 1.5$ gives
$p_A = 2.718$, $P(A \succ B) = 2.718/(2.718 + 1.5) = 0.644$,
$P(B \succ A) = 1/(1 + 1.5 \cdot 2.718) = 0.197$, and
$P(\text{tie}) = 1 - 0.644 - 0.197 = 0.159$. The closed-form tie formula gives the same
value.

#### Frontier, 2025 to 2026 (abstract level; UNVERIFIED details)

- DanceGRPO covers text-to-video and image-to-video with HunyuanVideo and SkyReels.
- *Rethinking Reward Signals in Video GRPO: When Scores Become Targets* (TaRoS)
  ([arXiv:2511.19356](https://arxiv.org/abs/2511.19356)) documents two failures:
  shortcut optimization under composite rewards (for example, motion jitter) and
  reward saturation that removes within-group ranking signal. It reweights reward
  components by their within-group separation.
- TAGRPO for image-to-video ([arXiv:2601.05729](https://arxiv.org/abs/2601.05729)).
- SAGE-GRPO, manifold-aware exploration for video RL
  ([arXiv:2603.21872](https://arxiv.org/abs/2603.21872)).
- VGGRPO with a 4D latent reward for world consistency
  ([arXiv:2603.26599](https://arxiv.org/abs/2603.26599)).
- *Human detectors are surprisingly powerful reward models* (HuDA)
  ([arXiv:2601.14037](https://arxiv.org/abs/2601.14037)) for human motion.
- Wan-R1, verifiable RL for video reasoning such as maze solving
  ([arXiv:2603.27866](https://arxiv.org/abs/2603.27866)).
- RAVEN, consistency-model GRPO for real-time autoregressive video
  ([arXiv:2605.15190](https://arxiv.org/abs/2605.15190)).

#### Exercise: multi-objective reward saturation

Simulate a "video" as a 1-D trajectory of 8 frames produced by a small flow policy.
Define a visual-quality reward (per-frame log-density under a target) and a motion
reward (mean absolute frame difference). Run GRPO on the sum.

Checks:

1. With the unweighted sum, the motion reward rises while the frame-to-frame
   smoothness (a hidden metric) degrades: jitter emerges.
2. When one component saturates (its within-group standard deviation falls below
   `1e-3`), a TaRoS-style reweighting assigns it a weight under 0.1.
3. The Rao-Kupper probabilities sum to 1 for random inputs.

#### Common mistakes

- Averaging dimension scores into one number before training. The balance then
  depends on each head's scale.
- Training on short clips and evaluating on long ones. Motion rewards don't transfer
  across durations.

---

## Part B. Vision-language models

### B1. Hallucination-focused preference learning: LLaVA-RLHF, RLHF-V, POVID, mDPO, RLAIF-V

#### Intuition

A VLM hallucinates when it describes objects or attributes not in the image. The
underlying cause is that the language prior can produce a fluent answer that ignores
the image. Preference learning against hallucination has two problems. First,
preference labels are expensive. Second, standard DPO on (image, winner, loser)
triples can be satisfied by learning *language* preferences that ignore the image.
The methods in this section address these problems in different ways.

#### LLaVA-RLHF: Factually Augmented RLHF

Sun et al. train a reward model for PPO and give the *reward model* extra facts, such
as ground-truth image captions and multiple-choice options. The reward model can then
check facts instead of rewarding fluent confident answers. They also introduce
MMHal-Bench, which penalizes ungrounded claims.

#### RLHF-V: dense direct preference optimization

Annotators correct the hallucinated segments of a response, so the winner and loser
differ only in the corrected segments. RLHF-V's DDPO (no relation to diffusion DDPO)
upweights the changed segments in the response log-probability *(verified)*:

$$
\log\pi(y \mid x) = \frac{1}{N}\Big[\sum_{y_i \in y_u}\log p(y_i \mid x, y_{<i}) + \gamma\sum_{y_i \in y_c}\log p(y_i \mid x, y_{<i})\Big], \qquad N = |y_u| + \gamma|y_c|,
$$

where $y_u$ is the unchanged tokens, $y_c$ the corrected tokens, and $\gamma = 5$. This
score is used inside the standard DPO loss.

Worked example: unchanged tokens have log-probs summing to $-6.0$ over $|y_u| = 20$
tokens, and corrected tokens sum to $-3.0$ over $|y_c| = 4$ tokens. With $\gamma = 5$,
$N = 20 + 20 = 40$ and the score is $(-6.0 + 5 \cdot (-3.0))/40 = -0.525$. Unweighted
length normalization would give $-9.0/24 = -0.375$.

#### POVID

POVID (Zhou et al.) generates dispreferred responses automatically in two ways. It
uses GPT-4V to inject plausible hallucinations into ground-truth answers, and it
distorts the image with noise to trigger the model's *own* hallucinations. The
ground-truth answer is the preferred response. Both kinds of negatives are then used
in DPO.

#### mDPO: make the image matter

Wang et al. observe that multimodal DPO can ignore the image. mDPO adds two terms
*(verified)*:

$$
\mathcal{L}_\text{DPO} = -\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w \mid m, q)}{\pi_\text{ref}(y_w \mid m, q)} - \beta\log\frac{\pi_\theta(y_l \mid m, q)}{\pi_\text{ref}(y_l \mid m, q)}\Big),
$$

$$
\mathcal{L}_\text{CoPO} = -\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w \mid m_w, q)}{\pi_\text{ref}(y_w \mid m_w, q)} - \beta\log\frac{\pi_\theta(y_w \mid m_l, q)}{\pi_\text{ref}(y_w \mid m_l, q)}\Big),
$$

$$
\mathcal{L}_\text{AncPO} = -\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w \mid m_w, q)}{\pi_\text{ref}(y_w \mid m_w, q)} - \delta\Big), \qquad \mathcal{L}_\text{mDPO} = \mathcal{L}_\text{DPO} + \mathcal{L}_\text{CoPO} + \mathcal{L}_\text{AncPO}.
$$

In the conditional term $\mathcal{L}_\text{CoPO}$, the *response* is fixed and the
*image* varies: $m_l$ is a corrupted image made by randomly cropping the original so
that less than 20% of it remains. The anchor term keeps the chosen response's implicit
reward above $\delta$, which prevents both likelihoods from falling together.

Worked example (with $\beta = 1$): the chosen log-ratio on the real image is $+0.3$,
and the chosen log-ratio on the cropped image is $+0.3$ as well. The model ignores
the image. Then $\mathcal{L}_\text{CoPO} = -\log\sigma(0) = 0.693$, the maximum
uninformative value, and its gradient pushes the two apart. With $\delta = 0$,
$\mathcal{L}_\text{AncPO} = -\log\sigma(0.3) = 0.554$.

#### RLAIF-V

RLAIF-V (Yu et al.) removes humans and proprietary models from the loop. It generates
candidates with controlled decoding (so differences are about content, not style),
splits each response into atomic claims, and scores each claim with an open-source
VLM ("divide and conquer"). It trains with iterative DPO and adds inference-time
self-feedback. It reports that the 7B model reduces object hallucination by 80.7% and
overall hallucination by 33.7%.

#### References

- Sun et al. *Aligning Large Multimodal Models with Factually Augmented RLHF*
  (LLaVA-RLHF). 2023; Findings of ACL 2024 (**UNVERIFIED** venue).
  [arXiv:2309.14525](https://arxiv.org/abs/2309.14525)
- Yu et al. *RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from
  Fine-grained Correctional Human Feedback.* CVPR 2024.
  [arXiv:2312.00849](https://arxiv.org/abs/2312.00849)
- Zhou et al. *Aligning Modalities in Vision Large Language Models via Preference
  Fine-tuning* (POVID). 2024. [arXiv:2402.11411](https://arxiv.org/abs/2402.11411)
- Wang et al. *mDPO: Conditional Preference Optimization for Multimodal Large Language
  Models.* EMNLP 2024 (**UNVERIFIED** venue). [arXiv:2406.11839](https://arxiv.org/abs/2406.11839)
- Yu et al. *RLAIF-V: Open-Source AI Feedback Leads to Super GPT-4V Trustworthiness.*
  2024; CVPR 2025 (**UNVERIFIED** venue). [arXiv:2405.17220](https://arxiv.org/abs/2405.17220)

#### Exercise: does the policy look at the image?

Build a toy VLM: the input is a one-hot "image" over 5 object classes plus a
"question" token, and the output is a categorical distribution over 5 answer tokens.
The policy is `softmax(W_img @ image + W_txt @ q + b)`. Pretrain it so that it partly
ignores the image (a strong prior on class 0).

Checks:

1. The DPO, CoPO, and AncPO losses match hand-computed values on fixed logits to
   `1e-6`, including the CoPO value $\log 2$ when image and cropped image give the same
   log-ratio.
2. The CoPO gradient with respect to `W_img` is exactly zero when `W_img` is zero and
   the two images differ only in content that `W_img` reads, while the DPO gradient
   is nonzero. (This isolates what each term trains.)
3. After training with standard DPO only on data where the winner is always the
   correct class, measure the "image reliance" metric
   $\mathbb{E}\big[\mathrm{KL}(\pi(\cdot \mid m, q)\,\Vert\,\pi(\cdot \mid m_\text{blank}, q))\big]$.
   With mDPO, image reliance is higher at equal DPO accuracy.
4. RLHF-V weighting: the dense score matches the formula for random segment masks.

#### Common mistakes

- Evaluating hallucination with the same model family that generated the negatives
  (POVID-style data evaluated by the same judge).
- Length bias: longer responses contain more hallucinations, so a preference for short
  answers looks like hallucination reduction.
- Anchoring $\delta$ too high, which forces the chosen likelihood up at the cost of
  fluency.

---

### B2. Visual reasoning with RLVR: Visual-RFT, VLM-R1, Vision-R1, and successors

#### Intuition

DeepSeek-R1 showed that GRPO with rule-based rewards can produce long reasoning in
LLMs. The same recipe applies to VLMs when the answer is verifiable: a multiple-choice
letter, a math answer, a count, or a bounding box scored by intersection over union
(IoU). Perception tasks add something new: the reward is *continuous*. IoU gives
partial credit, which gives a denser signal than 0-or-1 correctness, but it also
creates new hacks.

#### Key equations

IoU for axis-aligned boxes $b = (x_1, y_1, x_2, y_2)$:

$$
\mathrm{IoU}(b, \hat b) = \frac{|b \cap \hat b|}{|b| + |\hat b| - |b \cap \hat b|}.
$$

Example: $b = (0, 0, 2, 2)$ and $\hat b = (1, 1, 3, 3)$ intersect in a unit square, so
$\mathrm{IoU} = 1/(4 + 4 - 1) = 1/7 = 0.1429$.

Visual-RFT detection reward *(verified)*:

$$
R_d = R_\text{IoU} + R_\text{conf} + R_\text{format}, \qquad R_\text{IoU} = \frac{1}{n}\sum_{i=1}^n \mathrm{iou}_i, \qquad r_{c,i} = \begin{cases} c_i & \mathrm{iou}_i \ne 0 \\ 1 - c_i & \mathrm{iou}_i = 0.\end{cases}
$$

Here $c_i$ is the model's stated confidence for box $i$, so the model is rewarded for
being confident on correct boxes and unconfident on wrong ones. Classification uses
$R_\text{cls} = R_\text{acc} + R_\text{format}$ with a 0-or-1 accuracy term. The format
reward checks for `<think>` and `<answer>` tags.

VLM-R1 found that a raw mAP reward for open-vocabulary detection is hacked by
emitting many redundant boxes. Its fix is a length-penalized reward *(verified)*:

$$
R_\text{ovd} = s_\text{ovd}\cdot\mathrm{mAP}(b_\text{pred}, b_\text{gt}), \qquad s_\text{ovd} = \min\Big(1, \frac{L_\text{gt}}{L_\text{pred}}\Big).
$$

Example: 2 ground-truth boxes and 5 predictions give $s_\text{ovd} = 0.4$.

Vision-R1 (Huang et al.) builds a 200K multimodal chain-of-thought cold-start dataset
by bridging modalities (an MLLM describes the image in text, then DeepSeek-R1 reasons
over the text). It then applies GRPO with Progressive Thinking Suppression Training
(PTST), which caps reasoning length early in RL and relaxes the cap in later stages.
The reward is a hard format-and-result reward on a 10K multimodal math set. It reports
73.5% on MathVista for the 7B model. Another paper uses the same name: Zhan et al.,
*Vision-R1: Evolving Human-Free Alignment in LVLMs via Vision-Guided RL*
(arXiv:2503.18013, **UNVERIFIED**). Disambiguate the two in lessons.

The findings the course can teach from this line of work:

- **RL generalizes better than SFT out of domain.** VLM-R1 reports 63.16 versus 54.82
  on out-of-domain LISA-Grounding for RL versus SFT.
- **Thinking doesn't always help perception.** Perception-R1 finds that an explicit
  thinking process doesn't consistently improve perception tasks. R1-AQA (Part C) finds
  the same for audio.
- **"Thinking over seeing."** Vision-SR1 (ICLR 2026 per its listing) argues that R1-style
  RL leans on language reasoning and demotes perception, and it adds a self-reward for
  visual perception.
- **Agentic perception.** DeepEyes trains "thinking with images" end to end with RL:
  the model calls a crop or zoom tool mid-reasoning and is rewarded for correct
  answers.

#### References

- Liu et al. *Visual-RFT: Visual Reinforcement Fine-Tuning.* 2025.
  [arXiv:2503.01785](https://arxiv.org/abs/2503.01785)
- Shen et al. *VLM-R1: A Stable and Generalizable R1-style Large Vision-Language
  Model.* 2025. [arXiv:2504.07615](https://arxiv.org/abs/2504.07615)
- Huang, Jia, Zhai, et al. *Vision-R1: Incentivizing Reasoning Capability in
  Multimodal Large Language Models.* 2025. [arXiv:2503.06749](https://arxiv.org/abs/2503.06749)
- Yu et al. *Perception-R1: Pioneering Perception Policy with Reinforcement
  Learning.* 2025. [arXiv:2504.07954](https://arxiv.org/abs/2504.07954)
- Zheng et al. *DeepEyes: Incentivizing "Thinking with Images" via Reinforcement
  Learning.* 2025. [arXiv:2505.14362](https://arxiv.org/abs/2505.14362)
  (**UNVERIFIED** first author)
- *Self-Rewarding Vision-Language Model via Reasoning Decomposition* (Vision-SR1).
  [arXiv:2508.19652](https://arxiv.org/abs/2508.19652) (**UNVERIFIED** details)
- VisionReasoner. [arXiv:2505.12081](https://arxiv.org/abs/2505.12081)
- Survey: *Reinforcement Learning in Vision: A Survey.* 2025.
  [arXiv:2508.08189](https://arxiv.org/abs/2508.08189)
- Perceval, a perception-centric process reward model, 2026.
  [arXiv:2604.24583](https://arxiv.org/abs/2604.24583) (**UNVERIFIED** details)

#### Exercise: GRPO for toy grounding

The "image" is a 16x16 array with one bright rectangle; the "VLM" is a Gaussian policy
over 4 box coordinates whose mean comes from a linear layer on pooled features. Reward
is IoU plus a format term. For the detection variant, the policy emits $K$ boxes.

Checks:

1. `iou` matches hand-computed cases: identical boxes give 1, disjoint boxes give 0,
   and the offset squares give $1/7$.
2. `iou` handles degenerate boxes (zero area, or $x_2 < x_1$) without NaN. The
   exercise specifies the expected return value.
3. GRPO raises mean IoU from below 0.2 to above 0.7 within 300 updates (group
   size 8).
4. Reward hacking demo: with a raw recall-style reward and a free number of boxes $K$,
   the policy grows $K$. With the $\min(1, L_\text{gt}/L_\text{pred})$ penalty, $K$
   converges to the ground-truth count.

#### Common mistakes

- Box coordinate conventions: $(x, y, w, h)$ versus $(x_1, y_1, x_2, y_2)$, and pixel
  versus normalized coordinates (Qwen-VL versions differ).
- Format rewards that are too large compared with accuracy, so the model learns tags
  and nothing else.
- Reporting in-domain gains only. The RL-versus-SFT claim is about out-of-domain
  generalization.

---

### B3. Multimodal reward models

#### Intuition

Multimodal reward models come in three styles, and they fail in different ways:

- **Scalar reward models**: a VLM with a value head, trained with Bradley-Terry.
- **Generative judges**: a VLM that writes a critique and a verdict. LLaVA-Critic is
  one; UnifiedReward-Think adds reasoning.
- **Rule-based or verifiable rewards**: IoU, exact match, OCR.

Benchmarks show that even strong judges are mediocre at hallucination detection.
VL-RewardBench reports 65.4% accuracy for GPT-4o on its 1,250 examples. Multimodal
RewardBench reports about 72% for the best judges on 5,211 triplets.

#### Key facts (verified at abstract level)

- UnifiedReward ([arXiv:2503.05236](https://arxiv.org/abs/2503.05236)) is one model for
  pairwise ranking and pointwise scoring across image and video understanding and
  generation. It is used to build DPO pairs and as the pairwise reward model in
  Pref-GRPO.
- VL-RewardBench ([arXiv:2411.17451](https://arxiv.org/abs/2411.17451)), CVPR 2025.
- Multimodal RewardBench ([arXiv:2502.14191](https://arxiv.org/abs/2502.14191)) covers six
  domains: general correctness, preference, knowledge, reasoning, safety, and VQA.
- Skywork-VL Reward ([arXiv:2505.07263](https://arxiv.org/abs/2505.07263)).
- LLaVA-Critic ([arXiv:2410.02712](https://arxiv.org/abs/2410.02712), **UNVERIFIED** ID).

#### Exercise

Give students 200 synthetic (image features, response, label) triples and have them
fit a Bradley-Terry scalar reward model and a "generative judge" (a logistic model on
$P(\text{Yes})$).

Checks:

1. The BT loss at initialization equals $\log 2$.
2. On a planted spurious feature (longer responses always win in training), the
   model's accuracy on a length-balanced test set is measurably lower than on the
   training distribution.
3. Pairwise accuracy is invariant to adding a constant to all scores.

#### Common mistakes

- Position bias in pairwise generative judges. Evaluate both orders and average.
- Treating a judge's verbal confidence as a calibrated probability.

---

## Part C. Speech and audio

### C1. ASR: sequence-level training as REINFORCE and minimum word error rate

#### Intuition

ASR models are trained with token-level cross-entropy but evaluated on word error rate
(WER), a non-differentiable sequence-level metric. Minimum word error rate (MWER)
training minimizes *expected* WER under the model. Computing the expectation over all
sequences is impossible, so MWER approximates it over an N-best list from beam search,
with probabilities renormalized over the list and the list's mean WER subtracted as a
baseline. MWER is REINFORCE with a baseline, where the "samples" are the beam and the
expectation is computed exactly over that finite set.

#### Key equations

Let $\mathcal{B}(x) = \{y_1, \dots, y_N\}$ be the N-best list and $W(y, y^\ast)$ the
number of word errors. The renormalized probabilities and loss are:

$$
\hat P(y_i \mid x) = \frac{P(y_i \mid x)}{\sum_{j=1}^N P(y_j \mid x)}, \qquad \mathcal{L}_\text{MWER} = \sum_{i=1}^N \hat P(y_i \mid x)\big[W(y_i, y^\ast) - \bar W\big], \qquad \bar W = \sum_{i=1}^N \hat P(y_i \mid x)\,W(y_i, y^\ast).
$$

In the paper $\bar W$ is the average number of word errors over the N-best list (the
paper's precise averaging, uniform or probability-weighted, is **UNVERIFIED**; the
gradient identity in the next paragraph holds for any constant baseline). The training
loss interpolates with cross-entropy: $\mathcal{L} = \mathcal{L}_\text{MWER} + \lambda\,\mathcal{L}_\text{CE}$.
The paper compares sampling and N-best approximations and finds N-best better. It
reports up to 8.2% relative improvement.

The gradient with respect to the unnormalized log-score $\ell_i = \log P(y_i \mid x)$
is:

$$
\frac{\partial\mathcal{L}_\text{MWER}}{\partial\ell_i} = \hat P(y_i \mid x)\big[W(y_i, y^\ast) - \bar W\big].
$$

This is the REINFORCE estimator $\mathbb{E}_{\hat P}\big[(W - b)\nabla\log\hat P\big]$
with baseline $b = \bar W$, computed exactly over the finite set. Because
$\sum_i\hat P_i\nabla\log\hat P_i = 0$, any constant baseline gives the same exact
gradient. The baseline matters when the expectation is estimated by sampling, where it
reduces variance. The same structure appears in self-critical sequence training
(Rennie et al., 2017), where the baseline is the greedy decode's reward.

#### Worked example

Four hypotheses have log-scores $(2.0, 1.0, 0.5, -1.0)$ and word errors
$(0, 1, 2, 3)$:

- The renormalized probabilities are $(0.6095, 0.2242, 0.1360, 0.0303)$.
- The expected errors are $\bar W = 0.5872$.
- The gradient is $\hat P_i(W_i - \bar W) = (-0.3579, 0.0926, 0.1921, 0.0732)$.

A finite-difference check on $\sum_i\hat P_iW_i$ gives the same vector to 5 decimal
places. Note that the gradients sum to zero, so gradient descent only redistributes
probability within the beam.

WER itself is the Levenshtein distance divided by the reference length. For example,
reference `the cat sat` and hypothesis `the cat sit` give $\text{WER} = 1/3$.

#### References

- Prabhavalkar, Sainath, Wu, Nguyen, Chen, Chiu, Kannan. *Minimum Word Error Rate
  Training for Attention-based Sequence-to-Sequence Models.* ICASSP 2018.
  [arXiv:1712.01818](https://arxiv.org/abs/1712.01818)
- Rennie et al. *Self-critical Sequence Training for Image Captioning.* CVPR 2017.
  [arXiv:1612.00563](https://arxiv.org/abs/1612.00563)
- Ranzato et al. *Sequence Level Training with Recurrent Neural Networks* (MIXER).
  ICLR 2016. [arXiv:1511.06732](https://arxiv.org/abs/1511.06732)
- Povey. *Discriminative training for large vocabulary speech recognition* (MPE and
  sMBR precursors). PhD thesis, 2003 (**UNVERIFIED** details).

#### Exercise: MWER equals REINFORCE with a baseline

Students implement `levenshtein`, `wer`, `mwer_loss(logscores, errors)`, and a
sampling-based REINFORCE estimator.

Checks:

1. `levenshtein` matches known pairs, including empty strings, and is symmetric.
2. The autograd gradient of `mwer_loss` equals $\hat P_i(W_i - \bar W)$ to `1e-8` and
   matches the worked example.
3. The gradient doesn't change when a constant is added to all log-scores.
4. The Monte Carlo REINFORCE estimate over the renormalized distribution (50,000
   samples) with baseline $\bar W$ matches the exact gradient to `1e-2`. Without the
   baseline its variance is at least 2x larger on the provided example.
5. On a toy "acoustic model" (a categorical model over 3-word sequences from a 5-word
   vocabulary), MWER fine-tuning after cross-entropy reduces expected WER.

#### Common mistakes

- Using log-probabilities that are length-normalized in beam search but not in the
  loss, or the reverse.
- Forgetting the cross-entropy interpolation. Pure MWER can drift because it only
  compares hypotheses within the beam.
- Computing WER at the character level for a language where the evaluation is at the
  word level, or the reverse.

---

### C2. TTS: preference and RL fine-tuning

#### Intuition

Two families of TTS models need different RL machinery:

- **Codec language models** (VALL-E style, CosyVoice's LM stage, and Seed-TTS's
  autoregressive stage) produce discrete tokens, so LLM RL (DPO, PPO, GRPO) applies
  unchanged.
- **Flow-matching TTS** (F5-TTS, the flow stage of CosyVoice) has the same problem as
  image flow models: no per-step density. Solutions mirror Part A: add a Gaussian
  head, or convert the ODE to an SDE.

Rewards come from other models: an ASR model gives WER or character error rate (CER)
for intelligibility, a speaker-verification model gives cosine similarity (SIM) to the
reference speaker, and MOS predictors such as UTMOS give naturalness. Emotion
recognizers give control accuracy. Every one of these can be hacked.

#### Key facts and equations

**SpeechAlign** (Zhang et al., 2024) builds a preference dataset in which *golden*
codec tokens (from real speech) are preferred over the model's *synthetic* tokens. It
then runs preference optimization iteratively (self-improvement). The paper compares
several preference algorithms (DPO, PPO-based RLHF, best-of-N, and Chain-of-Hindsight
are listed in the paper's experiments; **UNVERIFIED** from the abstract).

**Seed-TTS** (ByteDance, 2024) *(verified from paper HTML)* compares RL with external
reward models (PPO, REINFORCE) against reward-free DPO, and reports both effective:

- **Seed-TTS$_\text{RL-SIM-WER}$** uses WER and SIM rewards. English WER drops from
  2.249 to 1.945 and Mandarin from 1.115 to 1.002, SIM improves, and CMOS is +0.14.
- **Seed-TTS$_\text{RL-SER}$** uses emotion-recognition rewards. For example, accuracy
  on "surprise" rises from 0.13 to 0.82.
- **Reward hacking.** Optimizing WER produced *slower, clearer* speech with lower WER
  but reduced naturalness.

**F5R-TTS** (Sun et al., Tencent, 2025) *(verified)* makes F5-TTS probabilistic: the
flow network outputs a mean and a standard deviation for the velocity target
$x_1 - x_0$, and pretraining uses a Gaussian negative log-likelihood:

$$
\mathcal{L}_\text{CFM}(\theta) = \mathbb{E}\Big[\frac{\big(\mu_\theta - (x_1 - x_0)\big)^2}{2\sigma_\theta^2} + \log\sigma_\theta\Big].
$$

The Gaussian gives each flow step a log-probability, and GRPO then uses the reward
$R = \lambda_W R_W + \lambda_S R_S$ with $R_W = 1 - \text{WER}$ and $R_S$ the speaker
cosine similarity. It reports a 29.5% relative WER reduction and a 4.6% relative SIM
improvement on zero-shot voice cloning.

Worked example: for a single dimension with target $0.8$, prediction $\mu = 0.6$, and
$\sigma = 0.5$, the per-dimension NLL term is
$0.04/(2 \cdot 0.25) + \log 0.5 = 0.08 - 0.6931 = -0.6131$.

**2025 to 2026 work (abstract level; UNVERIFIED details):**

- *Group Relative Policy Optimization for Text-to-Speech with Large Language Models*
  ([arXiv:2509.18798](https://arxiv.org/abs/2509.18798)) combines a CER reward with an
  NLL reward for LLM-based TTS.
- *Multi-Reward GRPO for Stable and Prosodic Single-Codebook TTS LLMs at Scale*
  ([arXiv:2511.21270](https://arxiv.org/abs/2511.21270)) uses WER, SIM, a length
  penalty, entropy regularization, and an LLM-annotated prosody reward.
- FlowTTS-GRPO ([arXiv:2606.23190](https://arxiv.org/abs/2606.23190)) is Flow-GRPO for TTS:
  ODE-to-SDE conversion on CosyVoice 3.0 and F5-TTS with multi-objective rewards.
- GROW ([arXiv:2608.03215](https://arxiv.org/abs/2608.03215)) targets
  autoregressive-diffusion TTS.
- *When Does Predictor-Based RL Align with Human Perception?*
  ([arXiv:2608.31035](https://arxiv.org/abs/2608.31035)) studies learned subjective
  rewards (naturalness, likability, arousal) with a CER constraint against transcript
  drift.

**UTMOS** (Saeki et al., 2022, [arXiv:2204.02152](https://arxiv.org/abs/2204.02152)) is
an SSL-based MOS predictor from the VoiceMOS Challenge 2022. As a reward it is
hackable: it rewards clean, studio-like audio and can prefer over-smoothed prosody.

#### Exercise: the WER-versus-naturalness trade-off

Build a toy TTS "policy" with two continuous controls, speaking rate $\rho$ and pitch
variance $\nu$, as a Gaussian policy. Simulated metrics: $\text{WER}(\rho) = 0.02 + 0.3\,\sigma(4(\rho - 1.2))$
(faster speech has more errors), and $\text{MOS}(\rho, \nu)$ peaked at $\rho = 1$ and
$\nu = 1$. Train with GRPO.

Checks:

1. With the WER-only reward, $\rho$ goes to the lower bound (slow speech): reward
   hacking reproduced.
2. With $R = (1 - \text{WER}) + 0.5\,\text{MOS}$, $\rho$ converges within `0.1` of the
   analytic optimum of the combined reward (the exercise computes it by grid search).
3. With a KL penalty to the initial policy, $\rho$ stays within the region predicted
   by the closed-form optimum $\pi^\star \propto \pi_\text{ref}\exp(R/\beta)$, evaluated
   on a grid.
4. F5R-style Gaussian NLL matches `torch.distributions.Normal(...).log_prob` up to the
   constant $\tfrac12\log 2\pi$.

#### Common mistakes

- Using the same ASR model for the WER reward and for evaluation. The policy learns
  that ASR model's quirks.
- Normalizing text differently when computing the reward WER and the evaluation WER
  (punctuation, casing, numerals).
- Speaker similarity measured against the prompt audio with the same verification
  model used as reward. Evaluate with a second model.

---

### C3. Audio LLMs, spoken dialogue, and music

#### Audio question answering with RL

- **R1-AQA** *(verified)* fine-tunes Qwen2-Audio-7B-Instruct with GRPO on about 38K
  AVQA samples, with a reward of +1 for a correct answer and +1 for the correct
  `<think>`/`<answer>` format. On MMAU Test-mini it reaches 64.5%, compared with 51.8%
  for SFT on the same data. Prompting *without* explicit thinking scored better (64.5%
  versus 61.1%). Human performance is 82.23%.
- **Audio-Reasoner** *(verified)* isn't an RL method. It is SFT on CoTA, 1.2M
  structured chain-of-thought samples, and reports large gains on MMAU-mini
  (+25.42%). Lessons must present it as the SFT baseline that RL papers compare
  against.
- **SARI** ([arXiv:2504.15900](https://arxiv.org/abs/2504.15900)) combines structured
  chain-of-thought supervision with curriculum-guided GRPO.
- **Audio-Thinker** ([arXiv:2508.08039](https://arxiv.org/abs/2508.08039)) learns when
  and how to think (**UNVERIFIED** details).
- **Audio-DeepThinker** ([arXiv:2604.18187](https://arxiv.org/abs/2604.18187)) uses
  two-stage GRPO and reports 74.0% on MMAR and 78.5% on MMAU test-mini. It won the
  single-model track of the Interspeech 2026 Audio Reasoning Challenge
  ([arXiv:2602.14224](https://arxiv.org/abs/2602.14224)) (**UNVERIFIED** details).
- *Reinforcement Learning with Evolving Rubrics as Rewards for Audio Reasoning*
  ([arXiv:2608.02831](https://arxiv.org/abs/2608.02831)) (**UNVERIFIED** details).

#### Spoken dialogue

- **Align-SLM** *(verified at abstract level; ACL 2025)* generates several speech
  continuations from a textless spoken language model, vocodes them, transcribes them
  with ASR, rates their semantics with an LLM, and trains the SLM with DPO (a LoRA
  adapter) on the resulting pairs.
- *Optimizing Conversational Quality in Spoken Dialogue Systems with RLAIF*
  ([arXiv:2601.19063](https://arxiv.org/abs/2601.19063)) (**UNVERIFIED** details).
- Full-duplex models such as Moshi ([arXiv:2410.00037](https://arxiv.org/abs/2410.00037))
  raise RL questions about turn-taking rewards; the course can mention them as open
  problems.

#### Music: MusicRL

MusicRL (Cideron et al., Google DeepMind, 2024) *(verified at abstract level)*
fine-tunes MusicLM, an autoregressive model over audio tokens. It has three variants:

- **MusicRL-R** optimizes designed rewards for text adherence (a MuLan-based score) and
  audio quality.
- **MusicRL-U** uses a reward model trained on 300,000 pairwise user preferences.
- **MusicRL-RU** combines both and is preferred by raters.

The paper notes that text adherence and quality explain only part of human preference.
The RL algorithm is a KL-regularized policy gradient on tokens (**UNVERIFIED** exact
algorithm).

#### References

- Li et al. *Reinforcement Learning Outperforms Supervised Fine-Tuning: A Case Study
  on Audio Question Answering* (R1-AQA). 2025. [arXiv:2503.11197](https://arxiv.org/abs/2503.11197)
- Xie et al. *Audio-Reasoner: Improving Reasoning Capability in Large Audio Language
  Models.* 2025. [arXiv:2503.02318](https://arxiv.org/abs/2503.02318)
- Lin et al. *Align-SLM: Textless Spoken Language Models with Reinforcement Learning
  from AI Feedback.* ACL 2025. [arXiv:2411.01834](https://arxiv.org/abs/2411.01834)
- Cideron et al. *MusicRL: Aligning Music Generation to Human Preferences.* ICML 2024
  (**UNVERIFIED** venue). [arXiv:2402.04229](https://arxiv.org/abs/2402.04229)
- Zhang et al. *SpeechAlign: Aligning Speech Generation to Human Preferences.* 2024.
  [arXiv:2404.05600](https://arxiv.org/abs/2404.05600)
- Anastassiou et al. *Seed-TTS: A Family of High-Quality Versatile Speech Generation
  Models.* 2024. [arXiv:2406.02430](https://arxiv.org/abs/2406.02430)
- Sun et al. *F5R-TTS: Improving Flow-Matching based Text-to-Speech with Group
  Relative Policy Optimization.* 2025. [arXiv:2504.02407](https://arxiv.org/abs/2504.02407)

#### Exercise: GRPO for a toy audio QA model

Features are 8-dimensional "audio embeddings" from 4 synthetic sound classes; the
policy is a categorical distribution over 4 answers plus a binary "think" action that
costs length.

Checks:

1. The reward function returns 2 for a correct, well-formatted answer and 0 for an
   unformatted one.
2. The GRPO advantage for a group where every answer is correct is exactly zero (no
   learning signal from solved prompts).
3. Accuracy rises from 25% to above 90% in under 300 updates.
4. Optional: when "thinking" doesn't change accuracy, a length penalty drives the
   think-rate down, mirroring the R1-AQA observation.

#### Common mistakes

- Treating Audio-Reasoner as an RL result.
- Evaluating on MMAU test-mini only, which has 1,000 items; differences of 1 point are
  within noise.

---

## Part D. Embodied and robotics foundation models

### D1. RL fine-tuning of VLAs

#### Intuition

A vision-language-action (VLA) model maps images and an instruction to robot actions.
It is pretrained by imitation. Imitation suffers from compounding errors: small
mistakes put the robot in states the demonstrations never visited. RL fine-tuning
lets the robot practice and recover. The algorithmic question depends on the action
head:

- **Autoregressive discrete-token VLAs** (OpenVLA, OpenVLA-OFT with discrete tokens)
  are LLMs, so PPO or GRPO on action tokens applies directly.
- **Flow-matching action heads** ($\pi_0$, $\pi_{0.5}$) have no per-step density. The
  fixes are the image-generation fixes: ODE-to-SDE conversion or learned noise.
- **Advantage-conditioned supervised learning** skips policy gradients and conditions
  the policy on "was this action good?"

Robotics has two layers of time: the environment episode and, for diffusion or flow
policies, the denoising chain inside each action. That two-layer MDP is the same
structure as Part A, nested inside the environment.

#### Key methods (verified at abstract or summary level unless noted)

**SimpleVLA-RL** ([arXiv:2509.09674](https://arxiv.org/abs/2509.09674), ICLR 2026). GRPO on
OpenVLA-OFT with binary task-success rewards, built on veRL with VLA-specific rollout
and parallel simulation. It raises LIBERO-Long to 97.6. With one demonstration per task
for cold-start SFT, RL raises performance from 17.3 to 91.7.

**RIPT-VLA** ([arXiv:2505.17016](https://arxiv.org/abs/2505.17016)). Interactive
post-training with sparse binary success, *dynamic rollout sampling* (drop contexts
where every rollout succeeds or every rollout fails, because they give zero advantage),
and leave-one-out (RLOO) advantages. It reports 97.5% on LIBERO with OpenVLA-OFT, and
raises a 4% one-demonstration model to 97% within 15 iterations.

The RLOO advantage for rollout $i$ of $K$ is:

$$
A_i = R_i - \frac{1}{K-1}\sum_{j \ne i}R_j.
$$

Example: successes $(1, 0, 1, 1)$ give advantages
$(1 - 2/3,\ 0 - 1,\ 1 - 2/3,\ 1 - 2/3) = (0.333, -1, 0.333, 0.333)$. They sum to zero.

**VLA-RL** ([arXiv:2505.18719](https://arxiv.org/abs/2505.18719)). Online PPO on an
autoregressive VLA, framing the trajectory as a multi-turn conversation. It
fine-tunes a VLM as a *robotic process reward model* on pseudo-labels to densify sparse
rewards.

**RL4VLA: What Can RL Bring to VLA Generalization?**
([arXiv:2505.19789](https://arxiv.org/abs/2505.19789)). An empirical study that finds
PPO beats SFT on out-of-distribution execution and semantics, and ties on visual
robustness.

**$\pi_\text{RL}$** ([arXiv:2510.25889](https://arxiv.org/abs/2510.25889)). Online RL for
$\pi_0$ and $\pi_{0.5}$ with two variants: **Flow-Noise** learns a noise network so
each denoising step has an exact Gaussian log-likelihood (a one-layer MDP), and
**Flow-SDE** uses ODE-to-SDE conversion inside a two-layer MDP. Flow-Noise is slightly
better. It is part of the RLinf framework.

**$\pi^\ast_{0.6}$ and Recap** ([arXiv:2511.14759](https://arxiv.org/abs/2511.14759),
Physical Intelligence). "RL with Experience and Corrections via Advantage-conditioned
Policies" *(verified from paper HTML)*:

- A distributional value function with 201 bins predicts (negative) steps to task
  completion, trained by Monte Carlo on returns:
  $\min_\phi\mathbb{E}_\tau\big[\sum_t H\big(R_t^B(\tau),\ p_\phi(V \mid o_t, \ell)\big)\big]$.
- The n-step advantage is
  $A(o_t, a_t, \ell) = \mathbb{E}\big[\sum_{t'=t}^{t+N-1}r_{t'} + V(o_{t+N})\big] - V(o_t)$.
- The improvement indicator is $I_t = \mathbb{1}\big[A(o_t, a_t, \ell) > \epsilon_\ell\big]$,
  with $\epsilon_\ell$ set at the 30th percentile of predicted values for the task.
- The policy sees the text "Advantage: positive" or "Advantage: negative" before the
  actions, and trains on *all* data with
  $\min_\theta\mathbb{E}\big[-\log\pi_\theta(a_t \mid o_t, \ell) - \alpha\log\pi_\theta(a_t \mid I_t, o_t, \ell)\big]$.
- The theory: for KL-regularized improvement,
  $\hat\pi(a \mid o, \ell) \propto \pi_\text{ref}(a \mid o, \ell)\big(\pi_\text{ref}(a \mid I, o, \ell)/\pi_\text{ref}(a \mid o, \ell)\big)^\beta$.
  This is classifier-free guidance on the improvement indicator, with $\beta > 1$ at
  test time.

Recap is the robotics analogue of reward-conditioned or "advantage-token" training.
It works with any action head, including flow matching, because it only needs
supervised likelihood training.

**World-model RL (2025 to 2026, abstract level; UNVERIFIED details):**

- VLA-RFT ([arXiv:2510.00406](https://arxiv.org/abs/2510.00406)) uses a learned video
  world model as a simulator with verified rewards from goal references.
- *Reinforcing VLAs in Task-Agnostic World Models*
  ([arXiv:2605.12334](https://arxiv.org/abs/2605.12334)).
- WCM, a world critic model ([arXiv:2607.29613](https://arxiv.org/abs/2607.29613)).
- Test-time RL for VLAs ([arXiv:2601.06748](https://arxiv.org/abs/2601.06748)).
- Robo-Dopamine, a general process reward model
  ([arXiv:2512.23703](https://arxiv.org/abs/2512.23703)).

#### Exercise: GRPO and RLOO on a toy VLA

Build a 2-D point-mass "robot" that must reach one of 3 goals named by an instruction
token. The policy is a Gaussian over 2-D velocity, conditioned on position and a goal
embedding. Pretrain it by behavior cloning on 5 noisy demonstrations per goal, so it
succeeds about 40% of the time. The reward is 1 if the robot is within 0.1 of the goal
at step 20.

Checks:

1. `rloo_advantages([1, 0, 1, 1])` returns $(1/3, -1, 1/3, 1/3)$, and advantages
   always sum to 0.
2. Dynamic sampling drops every context whose group rewards are all equal. The test
   counts the remaining contexts.
3. Success rate rises above 90% within 200 iterations with group size 8.
4. Recap-style variant: train a value function, threshold advantages at the 30th
   percentile, and train an advantage-conditioned policy by supervised learning.
   Sampling with $I = 1$ and guidance $\beta = 1.5$ beats unconditional sampling.

#### Common mistakes

- Averaging token log-probabilities for action chunks with a different
  normalization than was used during rollout.
- Measuring success in the same simulator seeds used for training.
- Ignoring the action-chunk structure: if a VLA outputs 8 actions per inference and
  the environment executes them open-loop, the MDP step is the chunk, not the action.

---

### D2. Residual RL and diffusion policy RL

#### Intuition

Fine-tuning a large pretrained policy with RL can destroy what it learned. *Residual
RL* freezes the base policy and learns a small correction on top:
$a = a_\text{base}(s) + a_\text{res}(s)$. The residual starts at zero, so the initial
behavior is the base policy, and the residual can be kept small by bounding or
penalizing it.

*DPPO* instead fine-tunes the diffusion policy itself, using the two-layer MDP: an
outer environment MDP whose actions are produced by an inner denoising MDP. It
fine-tunes only the last $K'$ denoising steps and clips the denoising noise from below,
so that exploration stays structured.

#### Key equations

Residual policy (Johannink et al.; Silver et al.):

$$
\pi(a \mid s) = \mathcal{N}\big(a_\text{base}(s) + \alpha\,f_\phi(s, a_\text{base}(s)),\ \Sigma_\phi\big),
$$

with $f_\phi$ initialized to zero. Policy Decorator adds a bound on the residual and a
schedule that mixes base-only and residual actions during exploration. ResiP trains a
closed-loop residual on top of a frozen, chunked BC diffusion policy with PPO and
sparse rewards; it reports going from 5% to 99% success on a 0.2 mm-clearance
peg-in-hole task.

DPPO's two-layer MDP: an environment step $\bar t$ has state $\bar s_{\bar t}$; inside
it, denoising step $k$ has state $(\bar s_{\bar t}, a^{k})$ and action $a^{k-1}$, with
Gaussian likelihood
$\pi_\theta(a^{k-1} \mid a^k, \bar s_{\bar t}) = \mathcal{N}\big(\mu_\theta(a^k, k, \bar s_{\bar t}),\ \sigma_k^2 I\big)$.
The environment reward arrives only when $a^0$ is executed. The PPO objective sums the
clipped ratio over both indices, using an advantage computed at the environment level
and shared by all denoising steps of that environment step (**UNVERIFIED** exact
discounting across the inner chain; DPPO uses a denoising discount factor, check the
paper).

#### Worked example

A 1-D base policy outputs $a_\text{base} = 0.8$ and the target is $1.0$, with
$r = -(a - 1)^2$. With $\alpha = 1$ and a scalar residual $w$, the optimal residual is
$0.2$. Starting from $w = 0$ with exploration noise $\sigma = 0.1$, the REINFORCE
gradient $\mathbb{E}[r\,(a - \mu)/\sigma^2]$ at $w = 0$ equals
$-\partial_\mu\mathbb{E}[(a-1)^2] = -2(\mu - 1) = 0.4$. You can check this estimate
numerically.

#### References

- Johannink et al. *Residual Reinforcement Learning for Robot Control.* ICRA 2019.
  [arXiv:1812.03201](https://arxiv.org/abs/1812.03201)
- Silver, Allen, Tenenbaum, Kaelbling. *Residual Policy Learning.* 2018.
  [arXiv:1812.06298](https://arxiv.org/abs/1812.06298)
- Yuan et al. *Policy Decorator: Model-Agnostic Online Refinement for Large Policy
  Model.* ICLR 2025. [arXiv:2412.13630](https://arxiv.org/abs/2412.13630)
  (**UNVERIFIED** first author)
- Ankile et al. *From Imitation to Refinement: Residual RL for Precise Assembly*
  (ResiP). 2024. [arXiv:2407.16677](https://arxiv.org/abs/2407.16677)
- Ren, Lidard, Ankile, Simeonov, Agrawal, Majumdar, Burchfiel, Dai, Simchowitz.
  *Diffusion Policy Policy Optimization.* ICLR 2025 (**UNVERIFIED** venue).
  [arXiv:2409.00588](https://arxiv.org/abs/2409.00588)
- Chi et al. *Diffusion Policy.* RSS 2023. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)

#### Exercise: DPPO versus residual RL

Environment: 1-D reach with a bimodal "pretrained" diffusion policy (4 denoising
steps) that picks the wrong mode 50% of the time.

Checks:

1. The REINFORCE gradient for the residual matches $0.4$ in the worked example within
   `0.02` using 100,000 samples.
2. DPPO with only the last $K' = 2$ denoising steps trainable: the gradient on the
   first two steps' parameters is exactly zero.
3. With the noise floor $\sigma_k \ge \sigma_\text{min}$, the log-probability never
   goes to $+\infty$ (no degenerate densities). A test sets $\sigma_\text{min} = 0.01$
   and checks finiteness.
4. Both methods reach 95% success. The residual method never produces an action more
   than its bound away from the base action.

#### Common mistakes

- Letting the residual policy's noise dominate the base action at the start, which
  destroys the base behavior before learning begins.
- Fine-tuning all denoising steps of a diffusion policy with a tiny final-step
  variance: ratios explode.

---

### D3. Rewards from foundation models: VLM-as-reward, RoboCLIP, Eureka

#### Intuition

Hand-written rewards are the bottleneck of robot RL. Foundation models offer three
substitutes:

- **Embedding similarity**: score the current frame or episode against a language or
  video description of the goal (VLM-RM, RoboCLIP).
- **VLM judges and progress estimators**: ask a VLM whether the task is done, or how far
  along it is (VLA-RL's process reward model, and many 2026 progress-reward papers).
- **LLM-written reward code**: Eureka asks an LLM to write the reward function, trains
  RL with it, and feeds back training statistics for the next round.

#### Key equations

VLM-RM (Rocamonde et al.) with goal-baseline regularization (**UNVERIFIED** exact form,
reconstructed from the paper's description):

$$
R(s) = 1 - \tfrac12\big\lVert\big(\alpha\,\mathrm{proj}_L + (1 - \alpha)I\big)\big(\mathrm{CLIP}_I(\psi(s)) - g\big)\big\rVert_2^2,
$$

where $g$ is the normalized embedding of the goal prompt, $b$ that of a baseline prompt
(for example "a humanoid robot"), $L$ is the line through $b$ and $g$, and $\alpha \in [0, 1]$.
With $\alpha = 0$ this is an affine function of cosine similarity for unit vectors.
The paper finds that larger CLIP models make better reward models.

RoboCLIP (Sontakke et al.) uses a pretrained video-language model (S3D trained on
HowTo100M) and gives a *sparse, end-of-episode* reward equal to the similarity between
the episode's video embedding and the embedding of one demonstration (a video or text).
It reports 2x to 3x higher zero-shot performance than competing imitation methods.

Eureka (Ma et al.) runs an evolutionary loop: sample $K$ reward programs from GPT-4
given the environment source code, train a policy with each, and summarize training
statistics ("reward reflection") into the next prompt. It outperforms expert human
rewards on 83% of 29 environments.

#### Worked example: goal-baseline projection

In 2-D, let $g = (1, 0)$ and $b = (0, 1)$. The line $L$ through them has direction
$d = (1, -1)/\sqrt2$. For a state embedding $e = (0.6, 0.8)$: $e - g = (-0.4, 0.8)$; its
projection onto $d$ is $\big((-0.4 - 0.8)/\sqrt2\big)\,d = (-0.6, 0.6)$. With
$\alpha = 1$, $R = 1 - \tfrac12(0.36 + 0.36) = 0.64$. With $\alpha = 0$,
$R = 1 - \tfrac12(0.16 + 0.64) = 0.60$. The projection removes the component
orthogonal to the goal-baseline direction, which carries information irrelevant to
the task.

The projection here is onto the direction of $L$ applied to $e - g$. Whether the paper
projects onto the direction or the affine line matters for off-line points; the course
must fix one convention in its tests.

#### References

- Rocamonde, Montesinos, Nava, Perez, Lindner. *Vision-Language Models are Zero-Shot
  Reward Models for Reinforcement Learning.* ICLR 2024.
  [arXiv:2310.12921](https://arxiv.org/abs/2310.12921)
- Sontakke et al. *RoboCLIP: One Demonstration is Enough to Learn Robot Policies.*
  NeurIPS 2023. [arXiv:2310.07899](https://arxiv.org/abs/2310.07899)
- Ma et al. *Eureka: Human-Level Reward Design via Coding Large Language Models.* ICLR
  2024. [arXiv:2310.12931](https://arxiv.org/abs/2310.12931)
- Baumli et al. *Vision-Language Models as a Source of Rewards.* 2023.
  [arXiv:2312.09187](https://arxiv.org/abs/2312.09187)
- Ma et al. *Vision Language Models are In-Context Value Learners* (GVL). 2024.
  [arXiv:2411.04549](https://arxiv.org/abs/2411.04549)

#### Exercise: a hackable embedding reward

Create synthetic "CLIP embeddings" for states in a 2-D grid world: a goal direction, a
baseline direction, and a nuisance direction correlated with a background texture that
the agent can change.

Checks:

1. The projection reward matches the worked example ($0.64$ and $0.60$).
2. With $\alpha = 0$, a tabular Q-learning agent learns to change the background
   texture (reward hacking); with $\alpha = 1$, it reaches the goal.
3. Eureka-lite: given 3 candidate reward functions as Python callables, the selection
   loop picks the one with the highest true success after short training. The test
   fixes the seeds.

#### Common mistakes

- Using a dense per-frame CLIP reward when the VLM is only reliable at recognizing the
  final state. RoboCLIP's reward is sparse for this reason.
- Running a VLM reward at every environment step without batching, which dominates
  training cost.
- Letting an LLM-written reward read privileged simulator state that doesn't exist on
  the real robot.

---

## Part E. Cross-cutting topics

### E1. KL regularization for diffusion and flow policies

#### Why the trajectory KL decomposes into per-step Gaussian KLs

Both the fine-tuned and reference samplers are Markov chains with the same initial
distribution $p(x_T) = \mathcal{N}(0, I)$:

$$
p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^T p_\theta(x_{t-1} \mid x_t), \qquad p_\text{ref}(x_{0:T}) = p(x_T)\prod_{t=1}^T p_\text{ref}(x_{t-1} \mid x_t).
$$

The log-ratio of the two path densities telescopes into a sum of per-step log-ratios,
because the initial terms cancel:

$$
\log\frac{p_\theta(x_{0:T})}{p_\text{ref}(x_{0:T})} = \sum_{t=1}^T\log\frac{p_\theta(x_{t-1} \mid x_t)}{p_\text{ref}(x_{t-1} \mid x_t)}.
$$

Taking the expectation under $p_\theta$ and applying the tower property to each term
gives the chain rule for KL:

$$
\mathrm{KL}\big(p_\theta(x_{0:T})\,\Vert\,p_\text{ref}(x_{0:T})\big) = \sum_{t=1}^T\mathbb{E}_{x_t \sim p_\theta}\Big[\mathrm{KL}\big(p_\theta(\cdot \mid x_t)\,\Vert\,p_\text{ref}(\cdot \mid x_t)\big)\Big].
$$

Each inner KL is between two Gaussians and is closed form. For shared variance it is
$\lVert\mu_\theta - \mu_\text{ref}\rVert^2/(2\sigma_t^2)$. For different variances:

$$
\mathrm{KL}\big(\mathcal{N}(\mu_1, \sigma_1^2)\,\Vert\,\mathcal{N}(\mu_2, \sigma_2^2)\big) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac12.
$$

For example, $\mathcal{N}(0.5, 0.2^2)$ against $\mathcal{N}(0.4, 0.25^2)$ gives $0.1231$.

The equality holds *exactly* for the path. The KL between *final* marginals is smaller,
by the data processing inequality:
$\mathrm{KL}(p_\theta(x_0)\,\Vert\,p_\text{ref}(x_0)) \le \mathrm{KL}(p_\theta(x_{0:T})\,\Vert\,p_\text{ref}(x_{0:T}))$.
This is DPOK's Lemma 4.2.

Three consequences for practice:

1. **The per-step KL needs the state from the policy's own rollouts.** The expectation
   is over $x_t \sim p_\theta$, so a KL computed on reference rollouts is a different
   quantity.
2. **The gap can be large.** Penalizing the path KL penalizes changes that don't affect
   the output distribution. In the linear-Gaussian chain in the next exercise, the path
   KL is 1.168 while the marginal KL is 0.054.
3. **It requires the same initial distribution and the same step variances** for
   the closed form to be cheap. If $\sigma_t$ differs, use the general formula; if the
   sampler is deterministic, the path KL is infinite or undefined.

The Flow-GRPO KL, the DDPM $\epsilon$-KL, the DPOK regularizer, and the Adjoint Matching
control cost $\tfrac12\int\lVert u\rVert^2\,\mathrm{d}t$ (Girsanov's theorem gives this as
the continuous-time path KL) are all instances of this one identity.

#### Exercise: verify the chain rule

Use a 1-D linear-Gaussian chain with $T = 4$:
$x_{t-1} = a_t x_t + b_t + s_t\varepsilon$ for both policies with the same $s_t$ and
$x_T \sim \mathcal{N}(0, 1)$. The reference solution uses the following parameters:

| Parameter | Step 1 | Step 2 | Step 3 | Step 4 |
|---|---|---|---|---|
| $a^\theta$ | 0.90 | 0.80 | 0.85 | 0.95 |
| $b^\theta$ | 0.10 | 0.20 | 0.00 | 0.30 |
| $a^\text{ref}$ | 0.90 | 0.70 | 0.85 | 1.00 |
| $b^\text{ref}$ | 0.00 | 0.20 | 0.10 | 0.00 |
| $s$ | 0.50 | 0.40 | 0.30 | 0.20 |

Checks:

1. The analytic path KL, computed by propagating the mean and variance of $x_t$ under
   $p_\theta$ and summing
   $\mathbb{E}[(\Delta a\,x_t + \Delta b)^2]/(2s_t^2)$, equals `1.16826`.
2. A Monte Carlo estimate of the summed per-step log-ratios with 400,000 samples
   agrees within 3 standard errors (the reference run gave $1.1715 \pm 0.0025$).
3. The KL between the Gaussian final marginals is `0.05419`, which is smaller than the
   path KL.
4. Setting $\theta = \text{ref}$ gives exactly zero for all three.

Runtime: under 1 second.

---

### E2. Credit assignment across denoising steps

#### The problem

Most methods give the terminal reward to every denoising step equally (DDPO,
Flow-GRPO). This is correct in expectation for REINFORCE, but it is high variance and
ignores the structure: early high-noise steps fix layout and composition, and late
low-noise steps fix texture. A bad image might have come from one bad early step.

#### Approaches

The approaches differ in where the per-step signal comes from:

| Approach | Source of per-step signal | Example |
|---|---|---|
| Uniform terminal reward | None | DDPO, Flow-GRPO |
| Learned value baseline | $V(x_t) \approx \mathbb{E}[r \mid x_t]$ | DPOK |
| Step-aware preference model | A model that scores noisy latents | SPO |
| Branching rollouts | Compare children of a shared $x_t$ | TempFlow-GRPO |
| Restrict stochastic steps | Only a window of steps is random and trained | MixGRPO |
| Train only late steps | The last $K$ or $K'$ steps get gradient | DRaFT-K, DPPO |
| Control-theoretic | The lean adjoint gives a per-time regression target | Adjoint Matching |

The branching idea is worth teaching in detail because it is exact. If you branch
$M$ continuations from the same $x_t$ and use the mean reward of the children as a
Monte Carlo estimate of $V(x_t)$, the difference between a child's reward and that mean
is an unbiased advantage for the action at step $t$. The price is $M$ extra rollouts
per branch point.

A related fact: when only the steps in a window are stochastic, the policy gradient
over those steps is still unbiased for the reward of the whole chain, because the
deterministic steps are part of the environment dynamics.

#### Exercise: per-step credit

Use a 3-step 1-D diffusion policy where only step 1 (the highest noise) matters for
the reward (the reward depends on the sign of $x_0$, which is almost fixed after step
1).

Checks:

1. The Monte Carlo variance of the per-step policy-gradient estimate with a uniform
   terminal reward is larger than with a branching baseline at step 1, with 10,000
   samples, by at least 2x.
2. The branching advantage has mean zero within 3 standard errors.
3. Gradients from steps 2 and 3 have expected value near zero (within 3 standard
   errors), matching the construction.

---

### E3. Reward model design for non-text modalities

The recurring design questions and the answers the literature gives:

- **Pointwise or pairwise?** Pairwise reward models (Bradley-Terry, win rates) are
  more robust to scale drift and to the illusory advantage in GRPO (Pref-GRPO,
  VideoAlign). Pointwise scores are cheaper and needed for best-of-N.
- **Ties.** Human annotators often can't pick. Model ties explicitly (Rao-Kupper, as in
  A8) or use soft labels (PickScore's 0.5 and 0.5).
- **Multiple dimensions.** Separate heads or tokens per dimension prevent leakage
  (VideoAlign). Reweight components that saturate (TaRoS).
- **Verifiable where possible.** IoU, OCR accuracy, object counts, WER, and task
  success are cheaper and harder to fool than learned preferences, but they cover
  narrow properties and have their own hacks: duplicate boxes, slow speech, and
  detector-fooling textures.
- **Noise-aware rewards.** Rewards on noisy latents enable guidance and step-aware
  credit (SPO, Flow-NRG) but must be trained on the model's own noise schedule.
- **Evaluation separation.** Always hold out a reward model, or human evaluation,
  that was not used in training.
- **Perception versus reasoning.** For VLMs and audio LLMs, pure outcome rewards
  can teach "thinking over seeing." Perception-specific rewards (grounding IoU,
  claim-level checks in RLAIF-V, and process reward models such as Perceval) target
  the perception step directly.

---

## Part F. Suggested exercise catalog

The following table summarizes the exercises and their verification hooks. All run on
CPU with numpy or torch in under 60 seconds.

| ID | Topic | Core check | Tolerance | Approx. runtime |
|---|---|---|---|---|
| F1 | Gaussian step log-prob | Equals `scipy.stats.norm.logpdf`; example value `0.1905` | `1e-10` | < 1 s |
| F2 | Path KL chain rule | Analytic `1.16826`, MC within 3 SE, marginal `0.05419` | 3 SE | 1 s |
| F3 | DDPO on 1-D mixture | Ratio is 1 at init; $P(x_0 > 1)$ above 0.9 | Fixed seed | 10 s |
| F4 | RWR | Weighted refit matches analytic posterior weights | `0.02` | 2 s |
| F5 | DRaFT-K | Autograd matches closed-form chain gradient | `1e-6` | 3 s |
| F6 | Diffusion-DPO loss | $\log 2$ at init; example `0.5130` | `1e-4` | < 1 s |
| F7 | ODE-to-SDE marginals | Mean `2.00`, std `0.50`, for all $a$ | `0.02` | 3 s |
| F8 | Memoryless schedule | Noise-sample correlation near 0 at $a = \sqrt2$ | `0.02` | 3 s |
| F9 | Flow-GRPO KL | Paper formula equals $\Delta\mu^2/(2\sigma^2 h)$ | `1e-12` | < 1 s |
| F10 | Illusory advantage | Advantages $(\pm0.447, \pm1.342)$ | `1e-3` | < 1 s |
| F11 | Adjoint Matching tilt | Samples match $\mathcal{N}(1, 1)$ | `0.05` | 30 s |
| F12 | mDPO terms | Hand-computed losses; CoPO equals $\log 2$ when image is ignored | `1e-6` | < 1 s |
| F13 | IoU GRPO | IoU cases; mean IoU above 0.7 | Fixed seed | 15 s |
| F14 | MWER | Gradient $\hat P(W - \bar W)$; example vector | `1e-8` | < 1 s |
| F15 | TTS reward hacking | WER-only drives rate to bound; combined reward hits grid optimum | `0.1` | 5 s |
| F16 | RLOO and dynamic sampling | $(1/3, -1, 1/3, 1/3)$; drop constant groups | Exact | < 1 s |
| F17 | Residual RL | REINFORCE gradient equals `0.4` | `0.02` | 2 s |
| F18 | VLM-RM projection | Rewards `0.64` and `0.60` | `1e-6` | < 1 s |
| F19 | Rao-Kupper ties | Probabilities sum to 1; example `0.644`, `0.197`, `0.159` | `1e-3` | < 1 s |
| F20 | DiffusionNFT reward map | $r = 0.75$ for the example; clipping at bounds | `1e-9` | < 1 s |

### Pitfalls for test design

- **Fix seeds and grids.** Stochastic checks must use fixed seeds and tolerances set
  from at least 20 reference runs, not 1.
- **Separate math checks from learning checks.** Math checks (log-probs, KLs, losses)
  must be exact. Learning checks (the reward goes up) need generous thresholds and a
  fixed seed.
- **Test sign conventions explicitly.** For every time-reversed formula, include a
  test with a Gaussian case whose answer is known.

---

## Verification log

The following claims were checked against paper text (HTML or PDF) during research:

- Flow-GRPO SDE, update rule, $\sigma_t = a\sqrt{t/(1-t)}$, objective, closed-form KL,
  $T = 10$ for training and 40 for inference, and GenEval and OCR numbers.
- The final Diffusion-DPO loss and the $\beta$ values (2000 for SD1.5, 5000 for SDXL).
- Adjoint Matching: tilted target, SOC problem, memoryless condition, flow-matching
  drift $2v - (\dot\alpha_t/\alpha_t)x$, lean adjoint ODE, loss, and initial value
  function bias.
- The DPOK objective and KL upper bound; DRaFT variants; ReFL step range and $\lambda$;
  the D3PO per-step loss; the Diffusion-KTO objective; the DiffusionNFT objective.
- VideoAlign dataset size, dimensions, and the constant-$\beta$ finding.
- mDPO's three terms; RLHF-V's weighting with $\gamma = 5$; Visual-RFT and VLM-R1
  rewards.
- Seed-TTS RL results and its reward-hacking observation; F5R-TTS NLL loss and
  rewards; R1-AQA settings and numbers.
- The Recap value function, advantage, indicator, and objective.

Numeric examples recomputed in Python: the step log-prob, both Gaussian KLs, the DDPM
$\epsilon$-KL coefficient, the Flow-GRPO KL, GRPO advantages, the illusory advantage,
the Diffusion-DPO example, the MWER gradient (autograd and finite differences), the
chain-rule path KL (analytic and Monte Carlo), ODE-to-SDE marginal preservation for
$a \in \{0, 0.3, 0.7, \sqrt2\}$, and the memoryless decorrelation.

The following items are **UNVERIFIED** and are marked in place:

- The `ddpo-pytorch` clip range of $10^{-4}$, and variable names in the Flow-GRPO code.
- The Lee et al. RWR loss notation and the ReFL ReLU sign convention.
- The constants in the `diffusers` Diffusion-DPO script.
- Tie handling in HPSv2, the HPSv3 loss form, and VideoAlign's exact tie
  parameterization.
- The exact VQAScore prompt suffix.
- The VLM-RM projection formula (reconstructed), and projection onto a direction versus
  an affine line.
- The DPPO denoising discount details.
- Several venues: Diffusion-KTO, LLaVA-RLHF, mDPO, RLAIF-V, MusicRL, DPPO, MixGRPO.
- Some author lists: DeepEyes, Policy Decorator.
- The Uehara et al. arXiv ID and the Tang citation.
- The LLaVA-Critic arXiv ID and the second Vision-R1 paper (arXiv:2503.18013).
- MWER's baseline averaging (uniform or probability-weighted).
- All 2026 papers cited from abstracts or search snippets only.
