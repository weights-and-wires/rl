---
title: Agents and multi-turn RL
slug: 075-agents-and-multi-turn-rl
part: "Part 9 — RL for language models"
summary: Train a model that calls tools by masking tool output out of the loss and giving each turn, not each token, its share of the reward.
minutes: 40
gpu: false
objectives:
  - Explain why tool output is an observation and not an action, and build a loss mask that removes it.
  - Show that the masked loss has exactly zero gradient on masked tokens.
  - Compute discounted returns per turn, and explain why discounting per token misassigns credit.
  - Train a tabular agent that learns to call a calculator, using REINFORCE with a per-turn baseline.
prereqs:
  - 033-baselines-and-actor-critic
  - 034-generalized-advantage-estimation
  - 070-group-relative-policy-optimization
exercise: 075-agents-and-multi-turn-rl
---

# Agents and multi-turn RL

You ask a model for the sum of two numbers. It writes `CALC`, a calculator replies with `3`, and the model writes `3` as its answer. The final reward, $+1$ for a correct answer, arrives after that last token. Which tokens should training push up?

Not all of them. The model wrote `CALC` and the final `3`, but the calculator wrote the middle `3`. If you train on it anyway, you teach the model to produce text that only the tool should produce. And the reward is one number for a transcript with several turns, so you also need a way to say that the tool call, two tokens before the reward, deserves credit.

This chapter fixes both problems with two small ideas. A mask removes the tool's tokens from the loss. A discount that counts turns, not tokens, spreads the reward back over the decisions that earned it.

## What you already know

- A policy gradient raises the log-probability of each action in proportion to its advantage (see [Baselines and actor-critic](/lessons/033-baselines-and-actor-critic)).
- A discounted return comes from a backward recursion, $G_t = r_t + \gamma G_{t+1}$ (see [Generalized advantage estimation](/lessons/034-generalized-advantage-estimation)).
- A language model's per-token log-probabilities have shape `(B, T)`, and a mask of the same shape marks which tokens count (see [Group-relative policy optimization](/lessons/070-group-relative-policy-optimization)).

## A transcript has more than one author

A *turn* is one stretch of text that the model writes before the environment replies. In the exercise, the prompt is two digits $a$ and $b$, and the correct answer is $(a + b) \bmod 8$. On each turn, the agent writes one token. A digit is its final answer. The token `CALC` calls the calculator, and the environment appends the calculator's output as a *tool token*. An episode ends at an answer or after three turns.

Every position in the transcript has a *role*: prompt, agent, tool, or padding. Figure 1 shows the transcript for $a = 5$ and $b = 6$, where the agent calls the tool and then answers. The reward is $-0.1$ for the tool call on turn 0 and $+1$ for the correct answer on turn 1.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="A transcript of eight positions with roles prompt, prompt, agent, tool, agent, and three padding, and the loss mask below it">
  <text x="20" y="30" class="fig-bold">Transcript for a = 5, b = 6</text>
  <rect x="40" y="50" width="60" height="44" class="fill-sunk stroke-muted"/>
  <rect x="100" y="50" width="60" height="44" class="fill-sunk stroke-muted"/>
  <rect x="160" y="50" width="60" height="44" class="fill-soft-b stroke-b"/>
  <rect x="220" y="50" width="60" height="44" class="fill-soft-a stroke-a"/>
  <rect x="280" y="50" width="60" height="44" class="fill-soft-b stroke-b"/>
  <rect x="340" y="50" width="60" height="44" class="fill-paper stroke-muted dashed"/>
  <rect x="400" y="50" width="60" height="44" class="fill-paper stroke-muted dashed"/>
  <rect x="460" y="50" width="60" height="44" class="fill-paper stroke-muted dashed"/>
  <text x="70" y="78" text-anchor="middle" class="fig-mono">5</text>
  <text x="130" y="78" text-anchor="middle" class="fig-mono">6</text>
  <text x="190" y="78" text-anchor="middle" class="fig-mono">CALC</text>
  <text x="250" y="78" text-anchor="middle" class="fig-mono">3</text>
  <text x="310" y="78" text-anchor="middle" class="fig-mono">3</text>
  <text x="370" y="78" text-anchor="middle" class="fig-mono fig-small">pad</text>
  <text x="430" y="78" text-anchor="middle" class="fig-mono fig-small">pad</text>
  <text x="490" y="78" text-anchor="middle" class="fig-mono fig-small">pad</text>
  <text x="70" y="114" text-anchor="middle" class="fig-small">prompt</text>
  <text x="130" y="114" text-anchor="middle" class="fig-small">prompt</text>
  <text x="190" y="114" text-anchor="middle" class="fig-small">agent</text>
  <text x="250" y="114" text-anchor="middle" class="fig-small">tool</text>
  <text x="310" y="114" text-anchor="middle" class="fig-small">agent</text>
  <text x="190" y="134" text-anchor="middle" class="fig-small">turn 0</text>
  <text x="250" y="134" text-anchor="middle" class="fig-small">turn 0</text>
  <text x="310" y="134" text-anchor="middle" class="fig-small">turn 1</text>
  <text x="20" y="190" class="fig-bold">Loss mask</text>
  <circle cx="70" cy="184" r="14" class="fill-sunk stroke-muted"/>
  <circle cx="130" cy="184" r="14" class="fill-sunk stroke-muted"/>
  <circle cx="190" cy="184" r="14" class="fill-b"/>
  <circle cx="250" cy="184" r="14" class="fill-sunk stroke-muted"/>
  <circle cx="310" cy="184" r="14" class="fill-b"/>
  <circle cx="370" cy="184" r="14" class="fill-sunk stroke-muted"/>
  <circle cx="430" cy="184" r="14" class="fill-sunk stroke-muted"/>
  <circle cx="490" cy="184" r="14" class="fill-sunk stroke-muted"/>
  <text x="70" y="189" text-anchor="middle" class="fig-small">0</text>
  <text x="130" y="189" text-anchor="middle" class="fig-small">0</text>
  <text x="190" y="189" text-anchor="middle" class="fig-small">1</text>
  <text x="250" y="189" text-anchor="middle" class="fig-small">0</text>
  <text x="310" y="189" text-anchor="middle" class="fig-small">1</text>
  <text x="370" y="189" text-anchor="middle" class="fig-small">0</text>
  <text x="430" y="189" text-anchor="middle" class="fig-small">0</text>
  <text x="490" y="189" text-anchor="middle" class="fig-small">0</text>
  <text x="20" y="232" class="fig-small">Reward: -0.1 on turn 0 (tool call), +1 on turn 1 (correct answer).</text>
</svg>
Figure 1. The agent wrote positions 2 and 4, and only those carry a 1 in the mask. The calculator wrote position 3, so the policy must not be trained to produce it.
```

The agent is a table of logits $\theta[k, \text{last token}, \text{action}]$. It sees only the turn number and the previous token. After the prompt it sees $b$ alone, so it can't compute the sum, and guessing is right one time in eight. After a tool call it sees the calculator's answer and only has to copy it. Using the tool is the only way to do well, so the agent's behavior is easy to read.

## Mask what the agent did not write

The obvious approach is to run the single-turn recipe on the whole transcript: multiply every token's log-probability by an advantage and average. Why does that fail? A policy gradient raises the probability of the actions the policy took. The tool's tokens aren't actions. The policy didn't choose them, and nothing it does changes them. Training on them makes the model predict what the tool will say. In a real language model, where every context shares weights, that teaches it to write made-up search results or imagined terminal output.

Search-R1 (Jin and colleagues, 2025) calls the fix retrieved-token masking: a mask that is 1 on tokens the model generated and 0 on tokens the retriever returned. With a mask $m_{i,t}$, the loss for a batch of $B$ episodes is

$$
\mathcal{L}(\theta) = -\frac{1}{\sum_{i,t} m_{i,t}} \sum_{i=1}^{B} \sum_{t=1}^{T} m_{i,t}\, \hat A_{i,t} \log \pi_\theta(y_{i,t} \mid s_{i,t})
$$

Read it as: for every token the agent wrote, multiply its log-probability by its advantage $\hat A_{i,t}$, add these up over the batch, divide by the number of agent tokens, and flip the sign. Tool, prompt, and padding tokens are multiplied by zero. In code it is two lines:

```python
mask = (roles == AGENT).float()
loss = -(mask * advantages * token_logps).sum() / mask.sum()
```

Work one example by hand. Two episodes have four positions each. The masks are $[0, 1, 0, 1]$ and $[0, 1, 1, 0]$, so there are four agent tokens. Their advantages are $1, 1$ and $-0.5, -0.5$, and their log-probabilities are $-1, -2$ and $-0.5, -1.5$. The sum of products is $-1 - 2 + 0.25 + 0.75 = -2$, so the loss is $-(-2)/4 = 0.5$.

Masked positions can hold any values at all, such as a log-probability of $-9$ and an advantage of 5. The gradient with respect to a token's log-probability is $-m_{i,t} \hat A_{i,t} / \sum m$, which is exactly zero where $m = 0$. So no gradient reaches the logits at that position either.

> [!KEY]
> Tool output is an observation, not an action. Multiply its tokens by a zero mask in the policy loss, and their gradient is exactly zero.

## Give credit to turns, not tokens

The reward arrives at the end, so each decision needs a share of it. With a discount $\gamma$ per turn, the return from turn $k$ is

$$
G_k = r_k + \gamma\, G_{k+1}, \qquad G_K = 0
$$

Read it as: the reward on this turn, plus a discounted copy of everything after it, computed backward from the last turn. Every agent token in turn $k$ then gets the turn's advantage, $\hat A_k = G_k - b_k$, where $b_k$ is a baseline for that turn. The recursion is a short loop:

```python
running = torch.zeros(B)
for k in reversed(range(K)):
    running = (turn_rewards[:, k] + gamma * running) * turn_mask[:, k]
    returns[:, k] = running
```

Why not discount per token, as single-turn RLHF does? Because then the tool's reply counts as time the agent spent. Take $\gamma = 0.9$. The agent calls the tool, for $-0.1$, and then answers correctly, for $+1$.

- **Per turn.** The return at the tool call is $-0.1 + 0.9 \times 1 = 0.8$, whatever the tool returns.
- **Per token, one tool token.** The answer is two positions later, so the return at `CALC` is $-0.1 + 0.9^2 = 0.71$.
- **Per token, three tool tokens.** The answer is four positions later, so it is $-0.1 + 0.9^4 = 0.5561$.

```figure
<svg viewBox="0 0 600 230" role="img" aria-label="Bars for the return at the tool call: 0.8 per turn, 0.71 per token with one tool token, and 0.5561 per token with three tool tokens">
  <text x="20" y="28" class="fig-bold">Return at the CALC token, gamma = 0.9</text>
  <line x1="190" y1="44" x2="190" y2="196" class="stroke-muted"/>
  <rect x="190" y="56" width="320" height="30" class="fill-c"/>
  <rect x="190" y="106" width="284" height="30" class="fill-b"/>
  <rect x="190" y="156" width="222.4" height="30" class="fill-b"/>
  <text x="180" y="76" text-anchor="end">per turn</text>
  <text x="180" y="126" text-anchor="end">per token, 1 tool token</text>
  <text x="180" y="176" text-anchor="end">per token, 3 tool tokens</text>
  <text x="518" y="76" class="fig-bold">0.8000</text>
  <text x="482" y="126" class="fig-bold">0.7100</text>
  <text x="420" y="176" class="fig-bold">0.5561</text>
</svg>
Figure 2. The agent made the same decision in all three cases. Only the length of the tool's reply changed, and per-token discounting cut the credit by about a third while the per-turn return stayed at 0.8.
```

With real tools that return hundreds of tokens, per-token discounting makes almost any tool call look bad.

> [!WARNING]
> Discounting per token charges the agent for the length of text it did not write. Discount per turn, or set $\gamma = 1$ within a turn.

The baseline matters too. The exercise uses the batch mean of $G_k$ over the episodes that reached turn $k$, a separate baseline per turn. ArCHer (Zhou and colleagues, 2024) goes further. It learns an utterance-level critic with temporal-difference updates over turns, and uses that critic's advantage to train the token-level policy.

## Train a calculator agent

The training loop is REINFORCE with a per-turn baseline. Each iteration plays 128 episodes, computes turn returns with $\gamma = 0.9$, subtracts the per-turn baseline, copies each turn's advantage to the agent tokens in that turn, and takes one Adam step on the masked loss.

The uniform starting policy answers correctly 12.5% of the time and calls the tool on the first turn 11% of the time. After 200 iterations, the agent calls the tool 99.6% of the time and is correct 99.5% of the time. With $\gamma = 0.9$, calling first is worth $0.9 - 0.1 = 0.8$, and a guess is worth 0.125 on average, so calling is optimal. If you raise the tool cost above 0.775, guessing becomes better. Reward design decides whether the agent uses the tool at all.

The mask leaves a footprint you can check. With it, the logits for the context right after a `CALC` token stay at exactly zero, because the agent never acts there. Without it, they move.

> [!RECAP]
> - The loss mask is 1 on agent tokens and 0 on prompt, tool, and padding.
> - The masked loss is $-\sum m \hat A \log \pi / \sum m$, and its gradient on masked tokens is exactly zero.
> - Turn returns follow $G_k = r_k + \gamma G_{k+1}$, one step per turn.
> - Per-token discounting shrinks credit by $\gamma$ for every tool token.
> - Each agent token gets its turn's advantage: $G_k$ minus a per-turn baseline.

## Build it step by step

The checks test the functions in this order. The environment, `rollout`, `token_logprobs`, and `evaluate` are given.

1. **`loss_mask(roles)`.** Takes `(B, T)` integer roles and returns a float mask of the same shape. Common mistake: including tool tokens because they are part of the transcript.
2. **`masked_pg_loss(token_logps, mask, advantages)`.** Three `(B, T)` tensors in, a scalar out. Divide by `mask.sum()`. Common mistake: `torch.mean` over all positions, which divides by $B \times T$ and lets padding dilute the loss.
3. **`turn_returns(turn_rewards, turn_mask, gamma)`.** `(B, K)` in and out. Walk backward over turns and multiply by the turn mask, so turns that did not happen stay at zero.
4. **`token_level_returns(token_rewards, gamma)`.** The same recursion over positions, for comparison. It needs no mask.
5. **`turn_to_token(turn_values, turn, mask)`.** Use `turn_values.gather(1, turn.clamp(min=0))` and multiply by the mask. Common mistake: gathering with the raw `turn` array. Its $-1$ entries make `gather` raise an index error.
6. **`train_agent(...)`.** Put the pieces together as in the loop above, starting from `init_policy()` and using the given `token_logprobs`. With `mask_tool_tokens=False`, add the tool positions to the mask. Common mistake: computing the baseline over all episodes, including those that ended before turn $k$, which pulls the baseline toward zero.

## Where this breaks

**Rewards are sparse and far away.** A software task can take fifty turns with one pass-or-fail signal at the end. Most episodes fail, and a group of failures has zero advantage under GRPO. People use easier tasks first, partial credit from individual tests, and learned per-turn critics.

**Tool spam and tool avoidance.** Reward the tool call itself and the agent calls tools it does not need. Charge too much and it never calls them. The exercise's break-even cost is 0.775, and real tasks have no such clean number.

**Slow tools.** One slow test run stalls every episode in a synchronous batch. Systems overlap generation and training, which makes rollouts come from slightly older policies. [Stale samples and importance weights](/lessons/076-rl-infrastructure-for-llms) covers the correction.

## Further reading

- [Jin and colleagues (2025), Search-R1](https://arxiv.org/abs/2503.09516). Trains 3B to 14B Qwen2.5 models to interleave reasoning with search, with retrieved-token masking and an exact-match reward.
- [Zhou and colleagues (2024), ArCHer](https://arxiv.org/abs/2402.19446). An utterance-level critic that supplies advantages to a token-level policy.
- [Yang and colleagues (2024), SWE-agent](https://arxiv.org/abs/2405.15793). The command interface that turns software repair into a multi-turn episode.
- [Pan and colleagues (2024), SWE-Gym](https://arxiv.org/abs/2412.21139). Executable training environments for software agents.

Researchers are still working on these open questions:

- How should credit be split across fifty turns when only the last one gets a reward, without a critic as large as the policy?
- How do you train on environments that change under you, such as live websites, without the policy exploiting quirks of a frozen copy?
- Which parts of an agent's skill come from RL, and which from fine-tuning on successful trajectories?

## The exercise

You build the pieces of multi-turn policy-gradient training and use them to train a calculator agent. Write `loss_mask`, `masked_pg_loss`, `turn_returns`, `token_level_returns`, and `turn_to_token`, then combine them in `train_agent`, which runs REINFORCE with a per-turn baseline.

The checks confirm the mask on scripted episodes, the hand value of 0.5 for the masked loss, and a gradient that is exactly zero on masked tokens. They check turn returns by hand, and that token-level returns fall from 0.71 to 0.5561 when the tool output grows from one token to three, while the turn return stays at 0.8. The trained agent must be right, and call the tool, more than 90% of the time over all 64 prompts. The last check confirms that masked training leaves the logits after a `CALC` token at exactly zero, and that unmasked training moves them.

> [!KEY]
> A multi-turn transcript has two authors. Train only on the tokens the agent wrote, and share the reward across turns instead of tokens. That answers the opening question: push up `CALC` and the final answer, never the calculator's reply.
