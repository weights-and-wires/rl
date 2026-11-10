# Curriculum

Sixty-seven chapters, in ten parts. Each chapter teaches one idea. Each row is one
Markdown file under `content/lessons/` and one directory under
`content/exercises/`. Every check runs on the CPU in the app container. Three
chapters, marked **GPU** in the following tables, also post-train a real small
language model in their **Run** block, on a RunPod GPU.

The course starts at a single decision under uncertainty and ends at the
methods that post-train language, image, speech, and robot models. Parts 1 to 8
are the discipline itself. Parts 9 and 10 are what the discipline became once
large pretrained models turned into policies.

The course replaced an earlier 52-lesson version. If you started that version,
you keep your progress: `content/renames.json` maps each old slug to the new
chapters that cover it, and the app carries your completed work across.

## Part 1 — Learning from reward

| Slug | Title | Exercise delivers |
|---|---|---|
| `001-learning-by-trial-and-error` | Learning by trial and error | A uniform random policy, an episode runner that separates termination from truncation, visit counts, and mean episode length in a 3x3 room |
| `002-returns-and-discounting` | Returns and discounting | Discount weights, discounted returns by backward recursion, and a Monte Carlo value estimate with its standard error |
| `003-explore-or-exploit` | Explore or exploit: epsilon-greedy | Arm gaps, incremental and constant-step estimates, epsilon-greedy action choice, and regret on a three-arm bandit |
| `004-optimism-and-ucb` | Optimism in the face of uncertainty: UCB | The UCB1 bonus and scores, the UCB1 action, the regret bound, and a bandit run that beats epsilon-greedy |
| `005-thompson-sampling` | Thompson sampling | Beta posteriors, Thompson sampling actions, the probability that each arm is best, and logarithmic regret |
| `006-gradient-bandits` | Gradient bandits | Softmax preferences, the gradient bandit update with a baseline, and its expected update checked against the exact gradient |
| `007-contextual-bandits` | Contextual bandits and LinUCB | LinUCB's prediction and width, the Sherman-Morrison update, and regret that flattens out on a contextual bandit |

## Part 2 — Planning with a known world

| Slug | Title | Exercise delivers |
|---|---|---|
| `008-markov-decision-processes` | Markov decision processes | Episode sampling in the four-cell hallway and the policy's transition matrix and reward vector |
| `009-the-bellman-equation` | The Bellman equation | Exact policy evaluation as a linear system, action values from state values, and the Bellman equation checked on random MDPs |
| `010-policy-evaluation` | Policy evaluation by sweeping | The Bellman sweep, iterative policy evaluation with a tolerance, and the stopping bound on the error |
| `011-policy-iteration` | Policy iteration | Greedy policies, policy improvement, and policy iteration on a two-state MDP and a random 20-state MDP |
| `012-value-iteration` | Value iteration and contraction | Value iteration, the gamma-to-the-k error bound, the stopping threshold, and the number of sweeps needed |

## Part 3 — Learning from experience

| Slug | Title | Exercise delivers |
|---|---|---|
| `013-monte-carlo-prediction` | Monte Carlo prediction | First-visit and every-visit Monte Carlo prediction, checked on the random walk against its exact values |
| `014-monte-carlo-control` | Monte Carlo control | Epsilon-soft policies, the soft world, and Monte Carlo control that reaches the best epsilon-soft value on a grid |
| `015-off-policy-importance-sampling` | Learning about one policy from another | Importance ratios, and ordinary against weighted importance sampling checked for bias and variance |
| `016-temporal-difference-learning` | Temporal-difference learning | The TD(0) update, TD(0) prediction, batch TD(0), and a comparison with constant-step Monte Carlo on the random walk |
| `017-sarsa-and-q-learning` | SARSA and Q-learning | SARSA and Q-learning updates, and the cliff, where Q-learning walks the edge and SARSA takes the safer path |
| `018-maximization-bias` | Maximization bias and double learning | The upward bias of a max over noisy estimates, double Q-learning targets, and the Expected SARSA target |
| `019-n-step-returns` | n-step returns | The n-step return, its expected error, n-step TD, and the best n on the 19-state walk |
| `020-eligibility-traces` | Eligibility traces and TD(lambda) | The lambda-return, offline and online TD(lambda) with eligibility traces, and the equivalence of the forward and backward views |
| `021-dyna-planning-with-a-learned-model` | Planning with a learned model: Dyna | The Q update, planning steps with a model, Dyna-Q, and prioritized sweeping |
| `022-monte-carlo-tree-search` | Monte Carlo tree search | UCT scores and Monte Carlo tree search that finds the winning Nim move |

## Part 4 — Deep value methods

| Slug | Title | Exercise delivers |
|---|---|---|
| `023-function-approximation` | Value-function approximation | Tile features, the semi-gradient TD update, the value error, convergence toward the TD fixed point, and semi-gradient SARSA on Mountain Car with 2-D tile features |
| `024-control-and-average-reward` | Control and average reward | Fourier and RBF features, the n-step return, n-step semi-gradient SARSA that beats one-step SARSA on Mountain Car, the closed-form average reward, and differential TD that recovers it exactly on a deterministic cycle |
| `025-the-deadly-triad` | The deadly triad | The TD fixed point, LSTD, and Baird's counterexample diverging off-policy |
| `026-deep-q-networks` | Deep Q-networks | A replay buffer, DQN targets, the Huber loss, soft target updates, and a DQN that balances CartPole |
| `027-double-and-dueling-dqn` | Double and dueling DQN | Double DQN targets, the dueling combination, and n-step targets |
| `028-prioritized-replay` | Prioritized experience replay | A sum tree, prioritized sampling probabilities, importance weights, and a prioritized replay buffer |
| `029-distributional-rl` | Distributional reinforcement learning | The categorical support, the C51 projection and loss, and the quantile Huber loss |
| `030-learning-from-pixels` | Learning from pixels | Frame preprocessing and stacking, a NoisyLinear layer with factorized noise checked by finite differences, random-shift augmentation, a DrQ-style TD loss, and a noisy DQN trained on pixel Catch |

## Part 5 — Policy gradients

| Slug | Title | Exercise delivers |
|---|---|---|
| `031-the-policy-gradient` | The policy gradient and REINFORCE | A softmax policy, the score function, the objective J, and REINFORCE checked against finite differences |
| `032-where-the-gradient-comes-from` | The policy gradient theorem | Discounted visitation, advantages, and the exact policy gradient checked against finite differences and the score form |
| `033-baselines-and-actor-critic` | Baselines and actor-critic | Baseline variance, the optimal baseline, expected TD errors, and a one-step actor-critic |
| `034-generalized-advantage-estimation` | Generalized advantage estimation | TD errors, next-state values that respect termination and truncation, and GAE on single and batched episodes |
| `035-trust-regions-and-trpo` | Natural gradients and trust regions | Mean KL, Fisher-vector products, conjugate gradient, the natural step, a KL line search, and a TRPO update |
| `036-proximal-policy-optimization` | Proximal policy optimization | The clipped surrogate, entropy, approximate KL, clip fraction, the PPO loss, KL early stopping, and a PPO agent on CartPole |
| `037-ppo-in-practice` | PPO in practice | A diagonal Gaussian log-probability, entropy, and KL; a parallel running mean and variance; reward scaling; advantage normalization; a linear learning-rate schedule; a clipped value loss; a vectorized Pendulum rollout; and one continuous PPO step |
| `038-impala-and-v-trace` | IMPALA and V-trace | Clipped importance weights, V-trace targets and advantages by backward recursion with episode ends, the n-step and TD(0) limits, and the operator's fixed point checked as the value of an interpolated policy on a random MDP |
| `039-why-policy-optimization-works` | Why policy optimization works | The performance difference identity, softmax policy gradient rates on a bandit, and natural policy gradient as mirror descent |

## Part 6 — Continuous control and maximum entropy

| Slug | Title | Exercise delivers |
|---|---|---|
| `040-deterministic-policy-gradients` | Deterministic policy gradients: DDPG and TD3 | The deterministic policy gradient on an LQR problem, TD3's smoothed target and clipped double Q, and TD3 on Pendulum |
| `041-maximum-entropy-rl` | Maximum-entropy reinforcement learning | The soft value, the Boltzmann policy, the soft Bellman backup, soft value iteration, and its residual |
| `042-soft-actor-critic` | Soft actor-critic | The tanh-Gaussian log-probability, a reparameterized sampler, the SAC actor and critic losses, and temperature tuning |

## Part 7 — Models, exploration, and fixed data

| Slug | Title | Exercise delivers |
|---|---|---|
| `043-model-based-rl` | Learning a model and planning with it | A linear dynamics model fitted by least squares, rollouts and compounding error, and MPC with the cross-entropy method on CartPole |
| `044-world-models-and-muzero` | World models and MuZero | Lambda-returns in imagination, value-equivalence loss, pUCT scores, and MCTS over a learned model |
| `045-exploration-bonuses` | Exploration beyond epsilon | Count bonuses, an empirical model, posterior sampling, and random network distillation on RiverSwim |
| `046-imitation-learning` | Imitation learning | Behavior cloning, the compounding-error gap measured on a tightrope, and DAgger closing it |
| `047-inverse-rl-and-gail` | Inverse RL and GAIL | Soft value iteration, state visitation, MaxEnt IRL feature matching, and the GAIL discriminator and reward |
| `048-offline-rl` | Offline reinforcement learning | Offline Q-iteration, the CQL penalty, IQL's expectile, and the TD3+BC actor loss |

## Part 8 — Rewards, evaluation, and many agents

| Slug | Title | Exercise delivers |
|---|---|---|
| `049-reward-design-and-shaping` | Reward design and shaping | Potential-based shaping that keeps the optimal policy, and a bonus that breaks it |
| `050-evaluating-rl` | Evaluating RL honestly | The interquartile mean, performance profiles, stratified bootstrap intervals, and their coverage |
| `051-multi-agent-and-self-play` | Many agents and self-play | Regret matching, exploitability, fictitious play on zero-sum and general-sum games, and Nim self-play |

## Part 9 — RL for language models

| Slug | Title | Exercise delivers |
|---|---|---|
| `052-sft-and-the-post-training-pipeline` | Supervised fine-tuning and the post-training pipeline | Masked next-token loss with the label shift, sequence packing with a block-diagonal mask, and rejection sampling as reward-weighted regression with weights of 0 or 1 |
| `053-language-models-as-policies` | Language models as policies | Token and sequence log-probabilities, the KL-regularized optimum, and best-of-n's distribution and KL |
| `054-reward-models` | Reward models from preferences | The Bradley-Terry loss, reward-model fitting, reward ensembles, and prefix values with process rewards |
| `055-judge-evaluation` | Evaluate a reward model and a judge | Pairwise accuracy, calibration, agreement, swap consistency, a length-controlled win rate, and Bradley-Terry strengths with ties |
| `056-rlhf-with-ppo` | RLHF with PPO | KL-shaped rewards, the k1, k2, and k3 estimators, and PPO on a toy language model that nears the KL-regularized optimum — **GPU**: PPO with a KL penalty on Qwen3-0.6B against a rule-based reward |
| `057-direct-preference-optimization` | Direct preference optimization | The DPO loss and gradient, IPO, SimPO, KTO, and likelihood displacement in a linear model |
| `058-group-relative-policy-optimization` | Group-relative policy optimization | RLOO and GRPO advantages, their bias, the clipped surrogate with separate bounds, loss aggregation, GSPO's sequence ratio (`gspo_ratio`), and the GRPO loss — **GPU**: GRPO on Qwen3-0.6B, with and without the Dr. GRPO fixes |
| `059-verifiable-rewards` | Verifiable rewards | Math-answer normalization, a format-plus-correctness reward, a timed code reward, a Wilson interval, and a filter that drops all-equal reward groups |
| `060-reasoning-with-verifiable-rewards` | Reasoning with verifiable rewards | The pass@k estimator, the zero-signal probability of a reward group, a verifiable answer reward, the policy-gradient loss, entropy collapse, and RLVR on a toy task — **GPU**: R1-Zero-style RLVR on Qwen3-0.6B-Base for arithmetic, with pass@k tracked |
| `061-process-rewards-and-search` | Process rewards and test-time search | Best-of-n expected reward, majority and weighted voting, step values, and PRM-guided beam search |
| `062-ai-feedback-and-self-improvement` | AI feedback and self-improvement | Filtered SFT loss, expert iteration as reward-weighted regression, an EM step, and the SPIN loss |
| `063-agents-and-multi-turn-rl` | Agents and multi-turn RL | Loss masks over tool output, a masked policy-gradient loss, turn-level returns, and a toy tool-using agent |
| `064-rl-infrastructure-for-llms` | Stale samples and importance weights | Truncated and masked importance weights, their bias and variance, sequence ratios, and stale-rollout drift |

## Part 10 — Beyond text, and what can go wrong

| Slug | Title | Exercise delivers |
|---|---|---|
| `065-diffusion-and-flow-policies` | Diffusion and flow models as policies | Gaussian log-probabilities, the DDPM mean, a denoising rollout, DDPO, the Diffusion-DPO loss, and a toy DDPO training run |
| `066-rl-for-vision-speech-and-robots` | RL for vision, speech, and robots | IoU and detection rewards, word error rate, minimum-WER training, residual action with a REINFORCE gradient, and action-to-token encoding and decoding for vision-language-action models |
| `067-reward-hacking-and-safe-rl` | Reward hacking, safety, and the frontier | Best-of-n KL, the Goodhart curve with ensembles, and the Lagrangian dual value and primal-dual update |
