# Curriculum

Fifty-two lessons, in ten parts. Each row is one Markdown file under
`content/lessons/` and one directory under `content/exercises/`. Every check
runs on the CPU in the app container. Three lessons, marked **GPU** in the
following tables, also post-train a real small language model in their **Run**
block, on a RunPod GPU.

The course starts at a single decision under uncertainty and ends at the
methods that post-train language, image, speech, and robot models. Parts 1 to 7
are the discipline itself. Parts 8 to 10 are what the discipline became once
large pretrained models turned into policies.

## Part 1 — The reinforcement learning problem

| Slug | Title | Exercise delivers |
|---|---|---|
| `01-the-rl-problem` | Agents, rewards, and returns | The agent-environment loop, discounted returns by backward recursion, and a Monte Carlo value estimate checked against the exact value |
| `02-multi-armed-bandits` | Multi-armed bandits | Incremental means, epsilon-greedy, UCB1, and Thompson sampling, with regret curves that show which grow logarithmically |
| `03-gradient-and-contextual-bandits` | Gradient and contextual bandits | The gradient bandit with a baseline, its gradient checked exactly, and LinUCB on a linear contextual bandit |
| `04-markov-decision-processes` | Markov decision processes | The Bellman expectation equation solved as a linear system, action values from state values, and the discounted state-visitation distribution |
| `05-dynamic-programming` | Dynamic programming | Iterative policy evaluation, policy iteration, value iteration, and the contraction rate that bounds their error |

## Part 2 — Learning from experience

| Slug | Title | Exercise delivers |
|---|---|---|
| `06-monte-carlo-methods` | Monte Carlo methods | First-visit prediction, on-policy control with epsilon-soft policies, and ordinary against weighted importance sampling |
| `07-temporal-difference-learning` | Temporal-difference learning | TD(0) on the random walk, SARSA and Q-learning on the cliff, and why they learn different paths |
| `08-double-learning` | Maximisation bias and double learning | The bias of a max over noisy estimates, double Q-learning, and Expected SARSA |
| `09-n-step-and-eligibility-traces` | n-step returns and eligibility traces | n-step TD, the lambda-return, and TD(lambda) with traces matching the offline lambda-return |
| `10-planning-and-learning` | Planning, learning, and search | Dyna-Q, prioritised sweeping, and Monte Carlo tree search with UCT |

## Part 3 — Approximation and deep value methods

| Slug | Title | Exercise delivers |
|---|---|---|
| `11-value-function-approximation` | Value-function approximation | Tile coding, semi-gradient TD, LSTD's fixed point, and Baird's counterexample diverging |
| `12-deep-q-networks` | Deep Q-networks | A replay buffer, a target network, the Huber TD loss, and a DQN that balances CartPole |
| `13-dqn-improvements` | Improving DQN | The Double DQN target, dueling heads, prioritised replay with a sum tree and importance weights, and n-step targets |
| `14-distributional-rl` | Distributional reinforcement learning | The C51 projection onto a fixed support and the quantile-regression Huber loss |

## Part 4 — Policy optimisation

| Slug | Title | Exercise delivers |
|---|---|---|
| `15-policy-gradient-theorem` | The policy gradient theorem | REINFORCE, checked for unbiasedness against the exact gradient of a small MDP |
| `16-actor-critic` | Baselines and actor-critic | The variance a baseline removes, the optimal constant baseline, and a one-step actor-critic |
| `17-generalized-advantage-estimation` | Generalised advantage estimation | GAE by backward recursion, its lambda limits, and the difference between termination and truncation |
| `18-natural-gradients-and-trpo` | Natural gradients and TRPO | Fisher-vector products, conjugate gradient, and a KL-constrained line search |
| `19-proximal-policy-optimization` | Proximal policy optimisation | The clipped surrogate, value clipping, the entropy bonus, and a PPO agent on CartPole |
| `20-policy-optimization-theory` | Why policy optimisation works | The performance difference lemma verified exactly, softmax policy gradient rates, and natural gradient as soft policy iteration |

## Part 5 — Continuous control and maximum entropy

| Slug | Title | Exercise delivers |
|---|---|---|
| `21-deterministic-policy-gradients` | Deterministic policy gradients | DDPG's actor and critic updates, TD3's clipped double Q and target smoothing, on Pendulum |
| `22-maximum-entropy-rl` | Maximum-entropy reinforcement learning | Soft value iteration, the Boltzmann optimal policy, and the soft Bellman residual |
| `23-soft-actor-critic` | Soft actor-critic | The tanh-Gaussian log-probability with its change-of-variables term, the reparameterised actor loss, and automatic temperature tuning |

## Part 6 — Models, exploration, and fixed data

| Slug | Title | Exercise delivers |
|---|---|---|
| `24-model-based-rl` | Model-based reinforcement learning | A learned dynamics model, compounding error measured, and model-predictive control with the cross-entropy method |
| `25-world-models-and-muzero` | World models and MuZero | Value-equivalent models, lambda-returns in imagination, and MCTS over a learned model |
| `26-exploration` | Exploration | Count-based bonuses, optimism on RiverSwim, posterior sampling, and random network distillation |
| `27-imitation-learning` | Imitation learning | Behaviour cloning, the compounding-error gap measured, and DAgger closing it |
| `28-inverse-rl` | Inverse reinforcement learning and GAIL | Maximum-entropy IRL by feature matching on a grid world, and a GAIL discriminator reward |
| `29-offline-rl` | Offline reinforcement learning | Extrapolation error made visible, the CQL penalty, and IQL's expectile value loss |

## Part 7 — Rewards, evaluation, and many agents

| Slug | Title | Exercise delivers |
|---|---|---|
| `30-reward-design` | Reward design and shaping | Potential-based shaping that keeps the optimal policy, and a bonus that breaks it |
| `31-evaluating-rl` | Evaluating reinforcement learning | The interquartile mean, stratified bootstrap intervals, and performance profiles across seeds |
| `32-multi-agent-and-self-play` | Multi-agent learning and self-play | Regret matching and fictitious play converging to Nash equilibria, and self-play on a small game |

## Part 8 — Reinforcement learning for language models

| Slug | Title | Exercise delivers |
|---|---|---|
| `33-language-models-as-policies` | Language models as policies | Sequence log-probabilities, the KL-regularised optimum in closed form, and best-of-n's KL |
| `34-reward-models` | Reward models | Bradley-Terry training, reward ensembles, and outcome against process rewards |
| `35-rlhf-with-ppo` | RLHF with PPO | Per-token KL rewards, the k1, k2, and k3 estimators, and PPO on a toy language model — **GPU**: PPO with a KL penalty on Qwen3-0.6B against a rule-based reward |
| `36-direct-preference-optimization` | Direct preference optimisation | DPO derived and implemented, IPO, KTO, and SimPO, and the likelihood displacement they share |
| `37-group-relative-policy-optimization` | Group-relative policy optimisation | RLOO, GRPO, Dr. GRPO's bias fix, DAPO's clip-higher, and GSPO's sequence ratio — **GPU**: GRPO on Qwen3-0.6B, with and without the Dr. GRPO fixes |
| `38-reasoning-with-verifiable-rewards` | Reasoning with verifiable rewards | The unbiased pass@k estimator, entropy collapse, and RLVR on a toy arithmetic task — **GPU**: R1-Zero-style RLVR on Qwen3-0.6B-Base for arithmetic, with pass@k tracked |
| `39-process-rewards-and-search` | Process rewards and test-time search | Best-of-n, weighted majority voting, and PRM-guided beam search |
| `40-ai-feedback-and-self-improvement` | AI feedback and self-improvement | Expert iteration as reward-weighted regression, STaR, and a SPIN-style self-play loss |
| `41-agents-and-multi-turn-rl` | Agents and multi-turn reinforcement learning | Loss masks over tool output, turn-level credit assignment, and a toy tool-using agent |
| `42-rl-infrastructure-for-llms` | Infrastructure for LLM reinforcement learning | Stale rollouts, train-inference mismatch, and truncated importance sampling |

## Part 9 — Beyond text

| Slug | Title | Exercise delivers |
|---|---|---|
| `43-diffusion-models-as-policies` | Diffusion models as policies | The denoising MDP, per-step Gaussian log-probabilities, DDPO, and reward backpropagation |
| `44-preference-tuning-for-diffusion` | Preference tuning for diffusion | The Diffusion-DPO loss, trajectory KL as a sum of step KLs, and image reward hacking |
| `45-flow-models-and-grpo` | Flow models and GRPO | The ODE-to-SDE conversion that keeps marginals, and Flow-GRPO on a two-dimensional flow |
| `46-vision-language-models` | Aligning vision-language models | Hallucination-aware preferences, IoU rewards for grounding, and visual RLVR |
| `47-speech-and-audio` | Speech and audio | Minimum word error rate training as REINFORCE, and preference tuning for text-to-speech |
| `48-robotics-and-embodied-agents` | Robotics and embodied agents | Residual RL on a fixed base policy, domain randomisation, and VLM-scored rewards |

## Part 10 — Safety and the frontier

| Slug | Title | Exercise delivers |
|---|---|---|
| `49-reward-hacking-and-overoptimization` | Reward hacking and overoptimisation | The Goodhart curve under best-of-n and RL, the Gao scaling fit, and ensembles that delay it |
| `50-safe-and-constrained-rl` | Safe and constrained reinforcement learning | Constrained MDPs, the Lagrangian primal-dual update, and a Safe-RLHF style cost model |
| `51-scaling-rl` | Scaling reinforcement learning | Sigmoidal compute-performance fits, plasticity loss measured, and resets that restore it |
| `52-the-frontier` | The research frontier | Test-time RL with majority-vote rewards, self-certainty rewards, and where each collapses |
