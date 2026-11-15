"""Agents and multi-turn reinforcement learning.

Fill in the six functions that raise NotImplementedError, in order. The
environment, `rollout`, `token_logprobs`, and `evaluate` are given.

The environment is a tiny calculator task. A prompt is two digits a and b,
and the correct answer is (a + b) mod 8. On each turn the agent writes one
token: a digit, which is its final answer, or CALC, which calls the
calculator. After a CALC, the environment appends the tool's output, the
digit (a + b) mod 8, as a tool token. An episode ends at an answer or after
MAX_TURNS turns.

The agent is a tabular policy that sees only the turn number and the last
token, theta[k, last_token, action]. After the prompt it sees b alone, so it
can't work out the sum. After a tool call it sees the tool's answer. Tool use
is the only way to be right more than 1 time in 8.

Token rows follow the layout of the language-model chapters: tokens, roles,
and per-token
log-probabilities are all (B, T_MAX), and rewards arrive per turn, (B, MAX_TURNS).
"""

import torch

torch.set_num_threads(1)

N_DIGITS = 8
CALC = 8                  # the tool-call token
PAD = 9
N_TOKENS = 10             # digits, CALC, PAD
N_ACTIONS = 9             # the agent writes a digit or CALC
MAX_TURNS = 3
T_MAX = 2 + 2 * MAX_TURNS
PROMPT, AGENT, TOOL, NONE = 0, 1, 2, 3
TOOL_COST = 0.1


def init_policy() -> torch.Tensor:
    """Uniform policy logits, shape (MAX_TURNS, N_TOKENS, N_ACTIONS)."""
    return torch.zeros(MAX_TURNS, N_TOKENS, N_ACTIONS)


def rollout(theta: torch.Tensor, batch: int, generator: torch.Generator):
    """Play `batch` episodes with the policy `theta`, without gradients.

    Returns:
        tokens: shape (B, T_MAX), PAD after the episode ends.
        roles: shape (B, T_MAX), PROMPT, AGENT, TOOL, or NONE per position.
        turn: shape (B, T_MAX), the turn index of every agent and tool token,
            -1 elsewhere.
        turn_rewards: shape (B, MAX_TURNS). -TOOL_COST for a tool call, 1 for
            a correct answer, 0 for a wrong one.
        turn_mask: shape (B, MAX_TURNS), 1.0 for every turn that happened.
    """
    a = torch.randint(0, N_DIGITS, (batch,), generator=generator)
    b = torch.randint(0, N_DIGITS, (batch,), generator=generator)
    answer = (a + b) % N_DIGITS
    tokens = torch.full((batch, T_MAX), PAD)
    roles = torch.full((batch, T_MAX), NONE)
    turn = torch.full((batch, T_MAX), -1)
    tokens[:, 0], tokens[:, 1] = a, b
    roles[:, :2] = PROMPT
    turn_rewards = torch.zeros(batch, MAX_TURNS)
    turn_mask = torch.zeros(batch, MAX_TURNS)
    alive = torch.ones(batch, dtype=torch.bool)
    pos = torch.full((batch,), 2)
    last = b.clone()
    with torch.no_grad():
        for k in range(MAX_TURNS):
            probs = torch.softmax(theta[k, last], dim=-1)
            act = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            rows = alive.nonzero().squeeze(1)
            tokens[rows, pos[rows]] = act[rows]
            roles[rows, pos[rows]] = AGENT
            turn[rows, pos[rows]] = k
            turn_mask[rows, k] = 1.0
            answered = alive & (act < N_DIGITS)
            called = alive & (act == CALC)
            turn_rewards[answered, k] = (act[answered] == answer[answered]).float()
            turn_rewards[called, k] = -TOOL_COST
            rows = called.nonzero().squeeze(1)
            tokens[rows, pos[rows] + 1] = answer[rows]
            roles[rows, pos[rows] + 1] = TOOL
            turn[rows, pos[rows] + 1] = k
            last = torch.where(called, answer, last)
            pos = pos + torch.where(called, 2, 1)
            alive = called
    return tokens, roles, turn, turn_rewards, turn_mask


def token_logprobs(theta: torch.Tensor, tokens: torch.Tensor, turn: torch.Tensor) -> torch.Tensor:
    """log pi(token_j | turn, previous token) at every position, shape (B, T_MAX).

    Position 0 has no previous token and gets 0. Positions whose token is not
    an action, such as PAD, also get a finite placeholder value. The loss mask
    decides which positions count.
    """
    prev = tokens[:, :-1]
    k = turn[:, 1:].clamp(min=0)
    logp = torch.log_softmax(theta[k, prev], dim=-1)            # (B, T - 1, A)
    target = tokens[:, 1:].clamp(max=N_ACTIONS - 1)
    picked = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return torch.cat([torch.zeros(tokens.shape[0], 1), picked], dim=1)


def loss_mask(roles: torch.Tensor) -> torch.Tensor:
    """1.0 on tokens the agent wrote and 0.0 everywhere else, shape (B, T)."""
    raise NotImplementedError("loss_mask")


def masked_pg_loss(
    token_logps: torch.Tensor, mask: torch.Tensor, advantages: torch.Tensor
) -> torch.Tensor:
    """Policy-gradient loss averaged over the agent's tokens in the whole batch.

    Args:
        token_logps: shape (B, T).
        mask: shape (B, T), 1.0 on the tokens to train.
        advantages: one advantage per token, shape (B, T).

    Returns:
        A scalar: -sum(mask * advantages * token_logps) / sum(mask).
    """
    raise NotImplementedError("masked_pg_loss")


def turn_returns(turn_rewards: torch.Tensor, turn_mask: torch.Tensor, gamma: float) -> torch.Tensor:
    """Discounted return from each turn, counting one step per turn.

    Args:
        turn_rewards: shape (B, K).
        turn_mask: shape (B, K), 1.0 for turns that happened.
        gamma: the per-turn discount.

    Returns:
        G[:, k] = r_k + gamma * G[:, k + 1] on turns that happened, and 0 on
        turns that didn't, shape (B, K).
    """
    raise NotImplementedError("turn_returns")


def token_level_returns(token_rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Discounted return from each position, counting one step per token.

    Args:
        token_rewards: shape (B, T), with each reward on the token where it
            arrives.
        gamma: the per-token discount.

    Returns:
        G[:, j] = r_j + gamma * G[:, j + 1], shape (B, T).
    """
    raise NotImplementedError("token_level_returns")


def turn_to_token(turn_values: torch.Tensor, turn: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Give every trained token the value of the turn it belongs to.

    Args:
        turn_values: shape (B, K), such as advantages per turn.
        turn: shape (B, T), the turn index of each token, -1 outside turns.
        mask: shape (B, T), the loss mask.

    Returns:
        shape (B, T): turn_values[b, turn[b, j]] where mask is 1, else 0.
    """
    raise NotImplementedError("turn_to_token")


def train_agent(
    iterations: int = 200,
    batch_size: int = 128,
    lr: float = 0.1,
    gamma: float = 0.9,
    seed: int = 0,
    mask_tool_tokens: bool = True,
) -> torch.Tensor:
    """REINFORCE with a per-turn baseline on the calculator task.

    Each iteration plays `batch_size` episodes, computes turn returns, takes
    as baseline the batch mean of G[:, k] over the episodes that reached turn
    k, broadcasts the advantages to tokens, and takes one Adam step on
    `masked_pg_loss`. With `mask_tool_tokens=False`, tool tokens are trained
    too, as if the agent had written them.

    Returns:
        The trained logits, shape (MAX_TURNS, N_TOKENS, N_ACTIONS).
    """
    raise NotImplementedError("train_agent")


def evaluate(theta: torch.Tensor) -> tuple[float, float]:
    """Exact accuracy and tool-call rate of `theta`, over all 64 prompts."""
    p = torch.softmax(theta, dim=-1)
    accuracy = 0.0
    calls = 0.0
    for a in range(N_DIGITS):
        for b in range(N_DIGITS):
            s = (a + b) % N_DIGITS
            reach, last = 1.0, b
            for k in range(MAX_TURNS):
                accuracy += reach * p[k, last, s].item()
                if k == 0:
                    calls += p[0, b, CALC].item()
                reach *= p[k, last, CALC].item()
                last = s
    return accuracy / 64, calls / 64


if __name__ == "__main__":
    acc, calls = evaluate(init_policy())
    print(f"uniform policy: accuracy {acc:.3f}, first-turn tool calls {calls:.3f}")
    theta = train_agent(seed=0)
    acc, calls = evaluate(theta)
    print(f"trained agent:  accuracy {acc:.3f}, first-turn tool calls {calls:.3f}")
    probs = torch.softmax(theta[1], dim=-1)
    print("after a tool call, P(answer = tool output) for outputs 0..7:")
    print("  ", [round(probs[s, s].item(), 3) for s in range(N_DIGITS)])
    same = torch.equal(theta[:, CALC], init_policy()[:, CALC])
    print(f"logits after a CALC token untouched by masked training: {same}")
