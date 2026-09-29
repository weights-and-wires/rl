---
title: Agents and multi-turn reinforcement learning
slug: 41-agents-and-multi-turn-rl
part: "Part 8 — Reinforcement learning for language models"
summary: Train a model that calls tools over several turns, with tool output masked from the loss and credit assigned per turn.
minutes: 55
gpu: false
objectives:
  - Build a loss mask that excludes tool output, and show that its gradient on masked tokens is exactly zero.
  - Compute discounted turn returns, and explain why discounting per token misassigns credit.
  - Train a tabular tool-using agent with REINFORCE and a per-turn baseline.
  - Describe how Search-R1 and software-engineering agents apply these ideas at scale.
prereqs:
  - 16-actor-critic
  - 17-generalized-advantage-estimation
  - 37-group-relative-policy-optimization
exercise: 41-agents-and-multi-turn-rl
---

# Agents and multi-turn reinforcement learning

So far in this part, a response has been one block of text that the model
writes from start to finish. An agent works differently. It writes a little,
calls a tool such as a search engine, a calculator, or a terminal, reads what
comes back, and writes some more. The final answer can be several turns away
from the first token.

That changes the reinforcement learning problem in two ways. First, some of
the tokens in the transcript came from the tool, not from the model. They're
observations, and training the model to produce them is a mistake. Second, a
single reward at the end has to be shared among several turns. The turn that
called the right tool deserves credit even though the reward arrived later.

This lesson handles both. You mask tool output out of the loss and check that
no gradient leaks through. You compute returns per turn and see why
discounting per token gets credit wrong. Then you train a small agent that
learns to use a calculator because it can't do the sum on its own.

## What you already know

- REINFORCE with a baseline pushes up the log-probability of each action in
  proportion to its advantage (`16-actor-critic`).
- A discounted return is computed by a backward recursion,
  $G_t = r_t + \gamma G_{t+1}$ (`17-generalized-advantage-estimation`).
- GRPO and its variants average a per-token loss, and whether you average per
  sequence or per token changes the gradient
  (`37-group-relative-policy-optimization`).
- A language model's per-token log-probabilities have shape `(B, T)`, and a
  response mask of the same shape marks which tokens count.

## An agent's episode is a multi-turn MDP

A *turn* is one stretch of text the model writes before the environment
replies. In the exercise, the prompt is two digits $a$ and $b$, and the
correct answer is $(a + b) \bmod 8$. On each turn, the agent writes one token.
A digit is its final answer. The token `CALC` calls a calculator, and the
environment appends the calculator's output, the correct digit, as a tool
token. The episode ends at an answer or after three turns.

Every position in the transcript has a *role*: prompt, agent, tool, or
padding. Here's a transcript for $a = 5$ and $b = 6$, where the agent calls
the tool and then answers:

| Position | 0 | 1 | 2 | 3 | 4 | 5 to 7 |
|---|---|---|---|---|---|---|
| Token | 5 | 6 | `CALC` | 3 | 3 | padding |
| Role | prompt | prompt | agent | tool | agent | none |
| Turn | none | none | 0 | 0 | 1 | none |

The agent chose positions 2 and 4. The environment wrote position 3. The
reward is $-0.1$ for the tool call on turn 0 and $+1$ for the correct answer
on turn 1.

The agent is a table of logits $\theta[k, \text{last token}, \text{action}]$.
It sees only the turn number and the previous token. After the prompt, it
sees $b$ alone, so it can't compute the sum, and guessing is right one time
in eight. After a tool call, it sees the calculator's answer and only has to
copy it. Using the tool is the only way to do well, which makes the agent's
behavior easy to read.

## Mask the tool's tokens out of the loss

A policy gradient raises the log-probability of the actions the policy took,
weighted by how well things went. The tool's tokens aren't actions. The
policy didn't choose them, and nothing it does changes them. Search-R1, from
Jin and colleagues (2025), calls the fix retrieved-token masking. It defines a
mask that is 1 on tokens the model generated and 0 on tokens the retriever
returned.

With a mask $m_{i,t}$ that is 1 on the agent's tokens, the loss for a batch of
$B$ episodes is

$$
\mathcal{L}(\theta) = -\frac{1}{\sum_{i,t} m_{i,t}} \sum_{i=1}^{B} \sum_{t=1}^{T} m_{i,t}\, \hat A_{i,t} \log \pi_\theta(y_{i,t} \mid s_{i,t})
$$

Read it as: for every token the agent wrote, multiply its log-probability by
its advantage $\hat A_{i,t}$. Add these up over the batch, divide by the total
number of agent tokens, and flip the sign. Tool, prompt, and padding tokens
are multiplied by zero. This is a per-token mean over the whole batch. A
per-sequence mean, which averages within each episode first, weights short
episodes more heavily. `37-group-relative-policy-optimization` explains why
that choice matters.

```python
mask = (roles == AGENT).float()
loss = -(mask * advantages * token_logps).sum() / mask.sum()
```

The gradient with respect to a token's log-probability is
$-m_{i,t} \hat A_{i,t} / \sum m$. On a masked token, that's exactly zero, so no
gradient reaches the logits at that position either. Here's a worked example
with two episodes of four positions each:

- Mask $[0, 1, 0, 1]$ and $[0, 1, 1, 0]$, so four agent tokens.
- Advantages on those tokens: $1, 1$ and $-0.5, -0.5$.
- Log-probabilities on those tokens: $-1, -2$ and $-0.5, -1.5$.
- The sum of products is $-1 - 2 + 0.25 + 0.75 = -2$.
- The loss is $-(-2) / 4 = 0.5$.

The masked positions can hold any values at all, such as a log-probability of
$-9$ and an advantage of 5, and the loss and gradient don't change.

What goes wrong without the mask? The model is trained to predict what the
tool will say. In the exercise, training on tool tokens changes the logits the
agent would use right after a `CALC` token, a context in which the agent
never acts. With the mask, those logits stay at exactly zero. In a real
language model, where all contexts share weights, training on tool output
teaches the model to write tool-like text itself: made-up search results or
imagined terminal output.

> [!KEY]
> Tool output is an observation, not an action. Multiply its tokens by a zero
> mask in the policy loss, and their gradient is exactly zero.

## Assign credit to turns, not tokens

A final reward has to be spread back over the turns that led to it. With a
discount $\gamma$ per turn, the return from turn $k$ is

$$
G_k = r_k + \gamma\, G_{k+1}, \qquad G_K = 0
$$

Read it as: the reward on this turn, plus a discounted copy of everything
after it, computed backward from the last turn. Each agent token in turn $k$
then gets the turn's advantage, $\hat A_k = G_k - b_k$, where $b_k$ is a
baseline for that turn.

```python
running = torch.zeros(B)
for k in reversed(range(K)):
    running = (turn_rewards[:, k] + gamma * running) * turn_mask[:, k]
    returns[:, k] = running
```

The alternative is to treat every token as a time step and discount per token,
$G_j = r_j + \gamma G_{j+1}$ over positions. That's what you get by running a
single-turn RLHF recipe on a multi-turn transcript. It ties credit to length,
and tool output counts as time the agent spent.

Here's a worked example with $\gamma = 0.9$. The agent calls the tool, for
$-0.1$, and then answers correctly, for $+1$:

- **Per turn.** The return at the tool call is $-0.1 + 0.9 \times 1 = 0.8$,
  whatever the tool returns.
- **Per token, one tool token.** The answer is two positions later, so the
  return at `CALC` is $-0.1 + 0.9^2 = 0.71$.
- **Per token, three tool tokens.** The answer is four positions later, so
  it's $-0.1 + 0.9^4 = 0.5561$.

The agent made the same decision in both token-level cases. Only the length of
the tool's reply changed, and it cut the credit by about a fifth. With real tools,
which return hundreds of tokens, per-token discounting makes almost any tool
call look bad.

> [!WARNING]
> Discounting per token charges the agent for the length of text it didn't
> write. Discount per turn, or set $\gamma = 1$ within a turn.

The baseline matters too. The exercise uses the batch mean of $G_k$ over the
episodes that reached turn $k$, a separate baseline per turn. ArCHer, from
Zhou and colleagues (2024), goes further. It learns an utterance-level critic
with temporal-difference updates over turns, and uses that critic's advantage
to train the token-level policy.

## Train a calculator agent

The training loop is REINFORCE with a per-turn baseline. Each iteration does
the following:

1. Play 128 episodes with the current policy, recording tokens, roles, turn
   indices, and turn rewards.
2. Compute turn returns with $\gamma = 0.9$, and subtract the per-turn
   baseline.
3. Copy each turn's advantage to the agent tokens in that turn.
4. Take one Adam step on the masked loss.

The uniform starting policy answers correctly 12.5% of the time and calls the
tool on the first turn 11% of the time. After 200 iterations, the agent calls
the tool 99.6% of the time and is correct 99.5% of the time. The copying
behavior is learned too. After a tool call, the probability that the agent's
answer equals the tool's output is above 0.998 for every digit.

With $\gamma = 0.9$, calling first is worth $0.9 - 0.1 = 0.8$, and a guess
is worth 0.125 on average, so calling is optimal. That's what the agent
learned. Raise the cost above 0.775 and guessing becomes better. Reward design
decides whether the agent uses the tool at all.

> [!RECAP]
> - The loss mask is 1 on agent tokens and 0 on prompt, tool, and padding.
> - The masked loss is $-\sum m \hat A \log \pi / \sum m$, a per-token mean
>   over the batch, and its gradient on masked tokens is exactly zero.
> - Turn returns follow $G_k = r_k + \gamma G_{k+1}$, one step per turn.
> - Per-token discounting shrinks credit by $\gamma$ for every tool token.
> - Each agent token gets its turn's advantage, $G_k$ minus a per-turn
>   baseline.

## Build it step by step

The checks test the functions in this order. The environment, `rollout`,
`token_logprobs`, and `evaluate` are given.

1. **`loss_mask(roles)`.** Takes `(B, T)` integer roles and returns a float
   mask of the same shape. Common mistake: including tool tokens because
   they're part of the transcript.
2. **`masked_pg_loss(token_logps, mask, advantages)`.** Three `(B, T)`
   tensors, returns a scalar. Divide by `mask.sum()`, clamped to at least 1.
   Common mistake: `torch.mean` over all positions, which divides by
   $B \times T$ and lets padding dilute the loss.
3. **`turn_returns(turn_rewards, turn_mask, gamma)`.** `(B, K)` in and out.
   Walk backward over turns and multiply by the turn mask, so turns that
   didn't happen stay at zero.
4. **`token_level_returns(token_rewards, gamma)`.** The same recursion over
   positions, for comparison. It doesn't need a mask.
5. **`turn_to_token(turn_values, turn, mask)`.** Use
   `turn_values.gather(1, turn.clamp(min=0))` and multiply by the mask.
   Common mistake: gathering with the raw `turn` array. Its $-1$ entries
   make `gather` raise an index error.
6. **`train_agent(...)`.** Put the pieces together as in the loop in the
   previous section. Start from `init_policy()`, and use the given
   `token_logprobs(theta, tokens, turn)`. With `mask_tool_tokens=False`, add
   the tool positions to the mask. Common mistake: computing the baseline over
   all episodes, including those that ended before turn $k$, which pulls the
   baseline toward zero.

## Cost and scaling

Multi-turn RL costs more than single-turn RL, because the environment sits
inside the generation loop. The following table lists where the extra time
and memory go:

| Cost | Grows with | Why |
|---|---|---|
| Generation | Turns times tokens per turn | Each turn waits for the tool before the next one starts |
| Tool calls | Batch size times turns | Search, code execution, and test suites run outside the GPU |
| Context length | Everything in the transcript | Tool output counts toward the context, although it isn't trained |
| Loss and backward pass | Total tokens, masked or not | The model still reads masked tokens in its forward pass |

Tool latency is the main problem at scale. One slow test run stalls every
episode in a synchronous batch. Systems overlap generation, tool calls, and
training, which makes rollouts come from slightly older policies.
`42-rl-infrastructure-for-llms` covers how to correct for that. The exercise's
agent costs almost nothing: 200 iterations of 128 episodes of at most eight
tokens.

## Search agents and software agents

Two families of agents show these ideas at scale.

**Search-R1.** Jin and colleagues (2025) train Qwen2.5 models of 3, 7, and 14
billion parameters to interleave reasoning with search. The model writes
reasoning between `<think>` tags. It writes a query between `<search>` tags,
and the retriever's passages come back between `<information>` tags. The
answer goes between `<answer>` tags. The reward is exact match between the
predicted and gold answers, with no format reward and no learned reward model.
Retrieved tokens are masked from the loss, as in this lesson. The paper trains
with both PPO and GRPO, allows at most four searches, and reports average
relative gains of 24% at 7B and 20% at 3B over retrieval-augmented baselines.

**Software-engineering agents.** SWE-agent, from Yang and colleagues (2024),
gives a model an agent-computer interface: commands to open, search, and edit
files and to run tests. Each command's output is an observation, and a task
can take dozens of turns. SWE-Gym, from Pan and colleagues (2024), supplies
2,438 real Python repository tasks with executable environments and unit
tests. It trains agents by fine-tuning on the trajectories that pass the
tests, which is expert iteration from `40-ai-feedback-and-self-improvement`.
With a verifier that picks among sampled attempts, it reaches 32.0% on
SWE-bench Verified. SWE-RL, from Wei and colleagues
(2025), is single-turn, not agentic. It applies GRPO to Llama 3.3 70B with a
rule-based reward: $-1$ for a malformed output, and otherwise the similarity
between the predicted patch and the real one from `difflib.SequenceMatcher`.
It reports 41.0% on SWE-bench Verified.

## Where this breaks

**Rewards are sparse and far away.** A software task can take fifty turns
with one pass-or-fail signal at the end. Most episodes fail, and a group of
failures has zero advantage under GRPO. People use easier tasks first, partial
credit from individual tests, and learned per-turn critics.

**Tool spam and tool avoidance.** Reward the tool call itself and the agent
calls tools it doesn't need. Charge too much and it never calls them. The
exercise's break-even cost is 0.775. Real tasks have no such clean number, so
people tune the cost and watch tool-call rates during training.

**The environment is noisy.** A search engine's results change, and flaky
tests fail at random. The same actions then get different rewards, which adds
variance that no baseline removes.

**Long contexts crowd out the policy.** Tool output fills the context
window. Agents truncate or summarise observations, and that changes the state
the policy sees between training and deployment.

**Masks drift from the tokenizer.** The mask is built from role boundaries,
but a chat template can merge a tool's first character with the agent's last
token. Check the mask on decoded examples before training.

## Further reading

- [Jin and colleagues (2025), Search-R1: Training LLMs to Reason and Leverage
  Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516).
  Interleaved search, retrieved-token masking, and an exact-match reward.
- [Zhou and colleagues (2024), ArCHer: Training Language Model Agents via
  Hierarchical Multi-Turn RL](https://arxiv.org/abs/2402.19446). An
  utterance-level critic that supplies advantages to a token-level policy.
- [Yang and colleagues (2024), SWE-agent: Agent-Computer Interfaces Enable
  Automated Software Engineering](https://arxiv.org/abs/2405.15793). The
  command interface that turns software repair into a multi-turn episode.
- [Pan and colleagues (2024), Training Software Engineering Agents and
  Verifiers with SWE-Gym](https://arxiv.org/abs/2412.21139). Executable
  training environments for software agents, and fine-tuning on passing
  trajectories.
- [Wei and colleagues (2025), SWE-RL: Advancing LLM Reasoning via
  Reinforcement Learning on Open Software Evolution](https://arxiv.org/abs/2502.18449).
  GRPO on real pull requests with a patch-similarity reward.
- [Kumar and colleagues (2024), Training Language Models to Self-Correct via
  Reinforcement Learning](https://arxiv.org/abs/2409.12917). SCoRe, a
  two-turn RL setup in which the second turn revises the first.

Researchers are still working on these open questions:

- How should credit be split across fifty turns when only the last one gets
  a reward, without a critic as large as the policy?
- How do you train on environments that change under you, such as live
  websites, without the policy exploiting quirks of a frozen copy?
- Which parts of an agent's skill come from RL, and which from fine-tuning on
  successful trajectories?

## The exercise

You build the pieces of multi-turn policy-gradient training and use them to
train a calculator agent. Write `loss_mask`, `masked_pg_loss`,
`turn_returns`, `token_level_returns`, and `turn_to_token`, then combine
them in `train_agent`, which runs REINFORCE with a per-turn baseline.

The checks confirm the mask on scripted episodes, the hand value of 0.5 for
the masked loss, and a gradient that is exactly zero on masked tokens, both
for the log-probabilities and for the logits behind them. They check turn
returns by hand, and they check that token-level returns fall from 0.71 to
0.5561 when the tool output grows from one token to three, while the turn
return stays at 0.8. The trained agent must be right, and call the tool,
more than 90% of the time, computed exactly over all 64 prompts. The last
check confirms that masked training leaves the logits after a `CALC` token at
exactly zero, and that unmasked training moves them.
