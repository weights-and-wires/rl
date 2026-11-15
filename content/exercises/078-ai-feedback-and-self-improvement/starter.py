"""AI feedback and self-improvement.

Fill in the five functions that raise NotImplementedError, in order.

The pieces here are the small computations inside expert iteration (STaR and
ReST) and SPIN. The shapes follow the rest of Part 9: per-token
log-probabilities and response masks are (B, T), and per-sequence rewards or weights are (B,).

The expert-iteration toy is the digit-sum task from the previous chapter. A
response is T = 4 digits from 0 to 7, it is correct when the digits sum to a
target, and the policy picks each digit from its own softmax, so the policy
is a (T, V) table of probabilities.
"""

import itertools

import numpy as np
import torch

torch.set_num_threads(1)

V = 8
T = 4


def toy_policy(seed: int = 0) -> np.ndarray:
    """A fixed base policy: one softmax over the V digits per step, shape (T, V)."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(T, V))
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def all_sequences() -> np.ndarray:
    """Every one of the V**T responses, shape (V**T, T), in lexicographic order."""
    return np.array(list(itertools.product(range(V), repeat=T)))


def token_logprobs(logits: torch.Tensor, seqs: torch.Tensor) -> torch.Tensor:
    """Per-token log-probabilities of `seqs` under per-step logits.

    Args:
        logits: shape (T, V), one row of logits per step.
        seqs: integer tokens, shape (B, T).

    Returns:
        log pi(y_t) for every token, shape (B, T).
    """
    logp = torch.log_softmax(logits, dim=-1)
    steps = torch.arange(seqs.shape[1]).expand_as(seqs)
    return logp[steps, seqs]


def sequence_logprobs(token_logps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """log pi(y | x) for each response: the masked sum of its token log-probs.

    Args:
        token_logps: shape (B, T).
        mask: 1.0 on response tokens and 0.0 on padding, shape (B, T).

    Returns:
        shape (B,).
    """
    raise NotImplementedError("sequence_logprobs")


def filtered_sft_loss(
    token_logps: torch.Tensor, mask: torch.Tensor, rewards: torch.Tensor
) -> torch.Tensor:
    """Expert iteration's loss: the NLL of the kept responses, averaged over them.

    Args:
        token_logps: shape (B, T).
        mask: shape (B, T).
        rewards: 1.0 for a correct response and 0.0 otherwise, shape (B,).

    Returns:
        A scalar: minus the mean sequence log-probability over the responses
        with reward 1. Zero, still attached to the graph, when none is kept.
    """
    raise NotImplementedError("filtered_sft_loss")


def rwr_loss(
    token_logps: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Reward-weighted regression: minus the weighted mean sequence log-probability.

    Args:
        token_logps: shape (B, T).
        mask: shape (B, T).
        weights: one non-negative weight per response, shape (B,).

    Returns:
        A scalar: -(1 / B) * sum_i weights_i * log pi(y_i | x).
    """
    raise NotImplementedError("rwr_loss")


def em_step(step_probs: np.ndarray, target: int) -> np.ndarray:
    """One exact expert-iteration step for the per-step policy class.

    Args:
        step_probs: the current policy, shape (T, V).
        target: the digit sum that earns reward 1.

    Returns:
        The policy of the same class that maximises E_old[r(y) log pi(y)]:
        the per-step marginals of the old policy conditioned on success,
        shape (T, V).
    """
    raise NotImplementedError("em_step")


def spin_loss(
    policy_real_logps: torch.Tensor,
    policy_synth_logps: torch.Tensor,
    prev_real_logps: torch.Tensor,
    prev_synth_logps: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """The SPIN loss for each (real, synthetic) pair, as Chen et al. (2024) write it.

    Args:
        policy_real_logps: log p_theta(y | x) for the human response, shape (B,).
        policy_synth_logps: log p_theta(y' | x) for the previous iterate's
            own response, shape (B,).
        prev_real_logps: log p_theta_t(y | x), shape (B,).
        prev_synth_logps: log p_theta_t(y' | x), shape (B,).
        lam: the regularisation weight lambda.

    Returns:
        ell(lam * log(p_theta(y) / p_theta_t(y)) - lam * log(p_theta(y') /
        p_theta_t(y'))) with ell(t) = log(1 + exp(-t)), shape (B,). The mean
        over pairs estimates L_SPIN.
    """
    raise NotImplementedError("spin_loss")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    target = 14
    seqs = all_sequences()

    print("Expert iteration as exact EM on the digit-sum toy")
    p = toy_policy(0)
    for step in range(8):
        success = np.sum(np.prod(p[np.arange(T), seqs], axis=1) * (seqs.sum(axis=1) == target))
        entropy = -np.sum(p * np.log(np.clip(p, 1e-300, None))) / T
        print(f"  step {step}: success {success:.4f}, mean per-step entropy {entropy:.3f}")
        p = em_step(p, target)

    print("SPIN on a 16-response toy: total variation to the data distribution")
    g = torch.Generator().manual_seed(0)
    p_data = torch.softmax(1.5 * torch.randn(16, generator=g), dim=0)
    theta = torch.zeros(16)
    for it in range(5):
        prev = torch.log_softmax(theta, dim=0)
        p_prev = prev.exp()
        pair_w = p_data[:, None] * p_prev[None, :]
        theta = theta.clone().requires_grad_(True)
        for _ in range(100):
            logp = torch.log_softmax(theta, dim=0)
            real = logp[:, None].expand(16, 16).reshape(-1)
            synth = logp[None, :].expand(16, 16).reshape(-1)
            prev_real = prev[:, None].expand(16, 16).reshape(-1)
            prev_synth = prev[None, :].expand(16, 16).reshape(-1)
            losses = spin_loss(real, synth, prev_real, prev_synth, lam=1.0)
            loss = (pair_w.reshape(-1) * losses).sum()
            (grad,) = torch.autograd.grad(loss, theta)
            theta = (theta - 1.0 * grad).detach().requires_grad_(True)
        theta = theta.detach()
        tv = 0.5 * (torch.softmax(theta, dim=0) - p_data).abs().sum().item()
        print(f"  iteration {it + 1}: TV {tv:.4f}")
