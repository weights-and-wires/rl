# Curriculum

Eighty-two chapters, in ten parts. Each chapter teaches one idea. Each row is one
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
| `045-partial-observability-and-memory` | Partial observability and memory | Tiger-problem belief updates, the exact best memoryless value on a T-maze, hidden-state resets with a masked sequence loss, and a GRU policy trained by policy gradient |
| `046-exploration-bonuses` | Exploration beyond epsilon | Count bonuses, an empirical model, posterior sampling, and random network distillation on RiverSwim |
| `047-sample-complexity-and-regret` | Sample complexity and regret | The simulation lemma checked on 200 random MDP pairs, UCB-VI with a Hoeffding bonus kept above Q*, regret against epsilon-greedy on RiverSwim, and the Robbins-Monro step-size sums |
| `048-goal-conditioned-rl-and-her` | Goals and hindsight | Sparse goal rewards, the final, future, and episode relabeling strategies with exact checks, a goal-conditioned DQN that fails on 12-bit bit-flipping without relabeling and succeeds with it, and GCSL targets |
| `049-options-and-hierarchical-rl` | Options and hierarchical RL | A four-rooms world with option models solved as linear systems and checked against Monte Carlo, SMDP value iteration that matches primitive values in 25 sweeps instead of 41, an SMDP Q-update that discounts by gamma^k, and an intra-option Q-update that changes every consistent option |
| `050-meta-rl` | Meta-RL and multi-task RL | RL^2 inputs with trial-boundary resets, the exact task posterior on a bandit family, Bayes-optimal, Thompson, fixed, and oracle values by dynamic programming over beliefs, and the MAML gradient with its second-order term checked against finite differences |
| `051-imitation-learning` | Imitation learning | Behavior cloning, the compounding-error gap measured on a tightrope, and DAgger closing it |
| `052-inverse-rl-and-gail` | Inverse RL and GAIL | Soft value iteration, state visitation, MaxEnt IRL feature matching, and the GAIL discriminator and reward |
| `053-offline-rl` | Offline reinforcement learning | Offline Q-iteration, the CQL penalty, IQL's expectile, and the TD3+BC actor loss |
| `054-advantage-weighted-policy-improvement` | Advantage-weighted policy improvement | The AWR weights `exp(A / beta)` with a clip, the KL-optimal improved policy and its objective, MPO's weights, dual, and temperature, the weighted-likelihood loss and Gaussian fit, and a tabular AWAC loop that beats behavior cloning |
| `055-off-policy-evaluation` | Evaluating a policy from logs | Effective sample size, IPS and self-normalized IPS, the doubly robust estimate, per-decision importance sampling and the recursive doubly robust estimate on trajectories, and fitted Q evaluation checked against exact policy evaluation |
| `056-decision-transformer` | Decision transformer: RL as sequence modeling | Return-to-go values, the (return, state, action) token layout, a causal mask checked by changing future tokens, a target-return rollout, an exact luck example, offline Q-iteration that stitches what return conditioning cannot, and a tiny decision transformer whose return rises with its target |

## Part 8 — Rewards, evaluation, and many agents

| Slug | Title | Exercise delivers |
|---|---|---|
| `057-reward-design-and-shaping` | Reward design and shaping | Potential-based shaping that keeps the optimal policy, and a bonus that breaks it |
| `058-preferences-over-trajectories` | Preferences over trajectories | Segment returns as sums, preference probabilities, the Bradley-Terry segment loss with a finite-difference gradient, a Boltzmann labeler, a reward fit whose greedy policy matches the true one on a grid, invariance to a constant per step, and ensemble-disagreement queries |
| `059-evaluating-rl` | Evaluating RL honestly | The interquartile mean, performance profiles, stratified bootstrap intervals, and their coverage |
| `060-debugging-rl-runs` | Debugging RL runs | Explained variance, value-target correlation, relative entropy, and gradient norm; a reward-scale invariance check on an advantage-normalized policy gradient; an overfit-one-batch test; and `diagnose`, which classifies training logs |
| `061-multi-agent-and-self-play` | Many agents and self-play | Regret matching, exploitability, fictitious play on zero-sum and general-sum games, and Nim self-play |
| `062-alphazero-self-play` | AlphaZero: learning from self-play | Visit-count policy targets with a temperature, Dirichlet root noise, the policy-value loss checked against finite differences, the eight board symmetries, self-play games, and a tic-tac-toe network that never loses to random play and holds minimax to a draw |
| `063-counterfactual-regret-minimization` | Counterfactual regret minimization | Kuhn poker's 12 information sets, exact best-response values, exploitability as the sum of both gains, and CFR with regret matching per information set, checked against the game value of -1/18 |
| `064-cooperative-multi-agent-rl` | Cooperative multi-agent RL | VDN's sum and the IGM check, QMIX's monotonic mixer checked by finite differences, exact and trained fits of AND and XOR payoff tables, and independent against joint learners on the climbing game |

## Part 9 — RL for language models

| Slug | Title | Exercise delivers |
|---|---|---|
| `065-sft-and-the-post-training-pipeline` | Supervised fine-tuning and the post-training pipeline | Masked next-token loss with the label shift, sequence packing with a block-diagonal mask, and rejection sampling as reward-weighted regression with weights of 0 or 1 |
| `066-language-models-as-policies` | Language models as policies | Token and sequence log-probabilities, the KL-regularized optimum, and best-of-n's distribution and KL |
| `067-reward-models` | Reward models from preferences | The Bradley-Terry loss, reward-model fitting, reward ensembles, and prefix values with process rewards |
| `068-judge-evaluation` | Evaluate a reward model and a judge | Pairwise accuracy, calibration, agreement, swap consistency, a length-controlled win rate, and Bradley-Terry strengths with ties |
| `069-rlhf-with-ppo` | RLHF with PPO | KL-shaped rewards, the k1, k2, and k3 estimators, and PPO on a toy language model that nears the KL-regularized optimum — **GPU**: PPO with a KL penalty on Qwen3-0.6B against a rule-based reward |
| `070-direct-preference-optimization` | Direct preference optimization | The DPO loss and gradient, IPO, SimPO, KTO, and likelihood displacement in a linear model |
| `071-group-relative-policy-optimization` | Group-relative policy optimization | RLOO and GRPO advantages, their bias, the clipped surrogate with separate bounds, loss aggregation, GSPO's sequence ratio (`gspo_ratio`), and the GRPO loss — **GPU**: GRPO on Qwen3-0.6B, with and without the Dr. GRPO fixes |
| `072-verifiable-rewards` | Verifiable rewards | Math-answer normalization, a format-plus-correctness reward, a timed code reward, a Wilson interval, and a filter that drops all-equal reward groups |
| `073-reasoning-with-verifiable-rewards` | Reasoning with verifiable rewards | The pass@k estimator, the zero-signal probability of a reward group, a verifiable answer reward, the policy-gradient loss, entropy collapse, and RLVR on a toy task — **GPU**: R1-Zero-style RLVR on Qwen3-0.6B-Base for arithmetic, with pass@k tracked |
| `074-process-rewards-and-search` | Process rewards and test-time search | Best-of-n expected reward, majority and weighted voting, step values, and PRM-guided beam search |
| `075-credit-assignment-for-llms` | Credit assignment for language models | Exact prefix values on a toy, Monte Carlo values with a standard error, VinePPO step advantages that telescope, Math-Shepherd hard and soft labels, implicit process rewards from a log ratio, and uniform versus per-step credit error |
| `076-ai-feedback-and-self-improvement` | AI feedback and self-improvement | Filtered SFT loss, expert iteration as reward-weighted regression, an EM step, and the SPIN loss |
| `077-on-policy-distillation` | On-policy distillation | Forward and reverse token KL with a padding mask, the sampled reverse-KL estimator, the per-token reward log pi_T - log pi_S, its score-function loss checked against the exact KL gradient, a bimodal-teacher fit, and on-policy distillation of a toy sequence model |
| `078-agents-and-multi-turn-rl` | Agents and multi-turn RL | Loss masks over tool output, a masked policy-gradient loss, turn-level returns, and a toy tool-using agent |
| `079-rl-infrastructure-for-llms` | Stale samples and importance weights | Truncated and masked importance weights, their bias and variance, sequence ratios, and stale-rollout drift |

## Part 10 — Beyond text, and what can go wrong

| Slug | Title | Exercise delivers |
|---|---|---|
| `080-diffusion-and-flow-policies` | Diffusion and flow models as policies | Gaussian log-probabilities, the DDPM mean, a denoising rollout, DDPO, the Diffusion-DPO loss, and a toy DDPO training run |
| `081-rl-for-vision-speech-and-robots` | RL for vision, speech, and robots | IoU and detection rewards, word error rate, minimum-WER training, residual action with a REINFORCE gradient, and action-to-token encoding and decoding for vision-language-action models |
| `082-reward-hacking-and-safe-rl` | Reward hacking, safety, and the frontier | Best-of-n KL, the Goodhart curve with ensembles, and the Lagrangian dual value and primal-dual update |
