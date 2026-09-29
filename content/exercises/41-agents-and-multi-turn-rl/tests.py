"""Checks for the agents and multi-turn RL exercise."""

import numpy as np
import torch

from rlcheck import almost, check, close, equal, fail, finite_difference, has, shape

torch.set_num_threads(1)

P, A, TL, N = 0, 1, 2, 3   # PROMPT, AGENT, TOOL, NONE, as in the exercise


def _scripted_roles():
    """Two episodes: call, read the tool, answer; and a direct answer."""
    return torch.tensor([
        [P, P, A, TL, A, N, N, N],
        [P, P, A, N, N, N, N, N],
    ])


def _exact_eval(theta):
    """Exact accuracy and first-turn tool-call rate over all 64 prompts."""
    p = torch.softmax(theta.double(), dim=-1)
    acc, calls = 0.0, 0.0
    for a in range(8):
        for b in range(8):
            s = (a + b) % 8
            reach, last = 1.0, b
            for k in range(theta.shape[0]):
                acc += reach * float(p[k, last, s])
                if k == 0:
                    calls += float(p[0, b, 8])
                reach *= float(p[k, last, 8])
                last = s
    return acc / 64, calls / 64


@check("loss_mask is 1 on the agent's tokens and 0 on prompt, tool, and padding tokens")
def test_loss_mask(sub):
    loss_mask = has(sub, "loss_mask")
    got = loss_mask(_scripted_roles())
    shape(got, (2, 8), label="loss_mask output")
    want = np.array([[0, 0, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0]], dtype=float)
    close(np.asarray(got, dtype=float), want, atol=0, label="mask on the scripted episodes")


@check("masked_pg_loss matches a hand value, and its gradient is exactly zero on masked tokens")
def test_masked_loss_gradient(sub):
    loss_fn = has(sub, "masked_pg_loss")
    mask = torch.tensor([[0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]], dtype=torch.float64)
    adv = torch.tensor([[5.0, 1.0, 5.0, 1.0], [5.0, -0.5, -0.5, 5.0]], dtype=torch.float64)
    logps = torch.tensor([[-9.0, -1.0, -9.0, -2.0], [-9.0, -0.5, -1.5, -9.0]],
                         dtype=torch.float64, requires_grad=True)
    loss = loss_fn(logps, mask, adv)
    # -(1*-1 + 1*-2 + -0.5*-0.5 + -0.5*-1.5) / 4 = -(-3 + 1) / 4 = 0.5
    almost(float(loss), 0.5, 1e-12, label="the hand-computed loss")
    (grad,) = torch.autograd.grad(loss, logps)
    if torch.any(grad[mask == 0] != 0):
        fail(f"the gradient on masked tokens must be exactly 0, got {grad[mask == 0].tolist()}")
    fd = finite_difference(lambda x: float(loss_fn(torch.as_tensor(x), mask, adv)),
                           logps.detach().numpy().copy())
    close(grad.numpy(), fd, atol=1e-7, label="gradient against finite differences")


@check("Through a softmax, the gradient on the logits at tool-output positions is exactly zero")
def test_logit_gradient_zero(sub):
    loss_mask = has(sub, "loss_mask")
    loss_fn = has(sub, "masked_pg_loss")
    g = torch.Generator().manual_seed(0)
    roles = _scripted_roles()
    logits = torch.randn(2, 8, 9, generator=g, dtype=torch.float64, requires_grad=True)
    tokens = torch.randint(0, 9, (2, 8), generator=g)
    logps = torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    mask = loss_mask(roles).to(torch.float64)
    adv = torch.randn(2, 8, generator=g, dtype=torch.float64)
    (grad,) = torch.autograd.grad(loss_fn(logps, mask, adv), logits)
    masked = grad[mask == 0]
    if torch.any(masked != 0):
        fail("the logits at prompt, tool, and padding positions received gradient "
             f"{masked.abs().max().item():.3g}; it must be exactly 0")
    if torch.all(grad[mask == 1] == 0):
        fail("the logits at the agent's positions received no gradient at all")


@check("turn_returns discounts once per turn and stops at turns that didn't happen")
def test_turn_returns(sub):
    turn_returns = has(sub, "turn_returns")
    rewards = torch.tensor([[-0.1, 1.0, 0.0], [0.0, 0.0, 0.0], [-0.1, -0.1, 0.0]],
                           dtype=torch.float64)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                        dtype=torch.float64)
    got = turn_returns(rewards, mask, 0.9)
    want = torch.tensor([[-0.1 + 0.9, 1.0, 0.0], [0.0, 0.0, 0.0],
                         [-0.1 - 0.09, -0.1, 0.0]], dtype=torch.float64)
    close(got, want, atol=1e-12, label="turn returns with gamma = 0.9")


@check("Token-level returns depend on the tool output's length; turn-level returns don't")
def test_token_vs_turn(sub):
    token_level = has(sub, "token_level_returns")
    turn_returns = has(sub, "turn_returns")
    gamma = 0.9
    got = {}
    for tool_len in (1, 3):
        # CALC (reward -0.1), then tool_len tool tokens, then the answer (reward 1).
        rewards = torch.zeros(1, 2 + tool_len, dtype=torch.float64)
        rewards[0, 0], rewards[0, -1] = -0.1, 1.0
        got[tool_len] = float(token_level(rewards, gamma)[0, 0])
    almost(got[1], -0.1 + gamma**2, 1e-12, label="token-level return at CALC, 1 tool token")
    almost(got[3], -0.1 + gamma**4, 1e-12, label="token-level return at CALC, 3 tool tokens")
    turn = turn_returns(torch.tensor([[-0.1, 1.0]], dtype=torch.float64),
                        torch.ones(1, 2, dtype=torch.float64), gamma)
    almost(float(turn[0, 0]), -0.1 + gamma, 1e-12, label="turn-level return at CALC")


@check("turn_to_token copies each turn's value to its trained tokens and zeroes the rest")
def test_turn_to_token(sub):
    turn_to_token = has(sub, "turn_to_token")
    loss_mask = has(sub, "loss_mask")
    turn = torch.tensor([[-1, -1, 0, 0, 1, -1, -1, -1], [-1, -1, 0, -1, -1, -1, -1, -1]])
    values = torch.tensor([[0.5, -2.0, 7.0], [3.0, 7.0, 7.0]], dtype=torch.float64)
    mask = loss_mask(_scripted_roles()).to(torch.float64)
    got = turn_to_token(values, turn, mask)
    want = torch.tensor([[0, 0, 0.5, 0, -2.0, 0, 0, 0], [0, 0, 3.0, 0, 0, 0, 0, 0]],
                        dtype=torch.float64)
    close(got, want, atol=0, label="token advantages")


@check("The trained agent calls the calculator and answers correctly over 90% of the time")
def test_train_agent(sub):
    train = has(sub, "train_agent")
    theta = train(iterations=200, batch_size=128, lr=0.1, gamma=0.9, seed=0)
    shape(theta, (3, 10, 9), label="trained logits")
    acc, calls = _exact_eval(theta.detach())
    if acc < 0.9 or calls < 0.9:
        fail(f"expected exact accuracy and first-turn tool-call rate above 0.9, got "
             f"{acc:.3f} and {calls:.3f}. Without the tool, accuracy can't pass 0.125.")


@check("Masked training never touches the logits after a CALC token; unmasked training does")
def test_mask_protects_tool_context(sub):
    train = has(sub, "train_agent")
    calc = 8
    masked = train(iterations=60, batch_size=128, lr=0.1, gamma=0.9, seed=1)
    if torch.any(masked[:, calc] != 0):
        fail("with the loss mask, no agent token follows a CALC token, so "
             "theta[:, CALC] must stay exactly at its initial zeros; its largest "
             f"entry moved to {masked[:, calc].abs().max().item():.3g}")
    unmasked = train(iterations=60, batch_size=128, lr=0.1, gamma=0.9, seed=1,
                     mask_tool_tokens=False)
    if torch.all(unmasked[:, calc] == 0):
        fail("training on tool tokens should change theta[:, CALC], the policy's "
             "guess at what the calculator will say, but it stayed at zero")
