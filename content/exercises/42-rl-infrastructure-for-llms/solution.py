"""Infrastructure for LLM reinforcement learning: reference solution.

Fast RL systems sample with an inference engine and train with a separate
training engine. The engine's probabilities mu differ from the trainer's pi,
because of staleness (mu is an older policy) and numerical mismatch (the two
engines compute the same policy differently). Importance sampling corrects
for mu != pi, and truncation or masking bounds its variance at the price of
bias.

Every quantity here follows the Part 8 layout: per-token log-probabilities
and masks are (B, T), and per-sequence values are (B,).
"""

import torch

torch.set_num_threads(1)


def tis_weights(ratio: torch.Tensor, cap: float) -> torch.Tensor:
    """Truncated importance weights: min(ratio, cap), elementwise."""
    return torch.clamp(ratio, max=cap)


def mis_weights(ratio: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Masked importance weights: the ratio inside [low, high], 0 outside."""
    keep = (ratio >= low) & (ratio <= high)
    return torch.where(keep, ratio, torch.zeros_like(ratio))


def estimator_bias_variance(
    mu: torch.Tensor, pi: torch.Tensor, f: torch.Tensor, weights: torch.Tensor
) -> tuple[float, float]:
    """Exact bias and variance of the one-sample estimate w(y) f(y), y ~ mu.

    Args:
        mu: the behaviour distribution that samples y, shape (M,).
        pi: the target distribution, shape (M,).
        f: the quantity whose mean under pi you want, shape (M,).
        weights: the weight w(y) the estimator applies to each outcome, shape (M,).

    Returns:
        (bias, variance): E_mu[w f] - E_pi[f], and Var_mu[w f].
    """
    x = weights * f
    mean = torch.sum(mu * x)
    bias = mean - torch.sum(pi * f)
    variance = torch.sum(mu * x**2) - mean**2
    return float(bias), float(variance)


def sequence_ratio(
    logp_target: torch.Tensor, logp_behavior: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """The sequence-level importance ratio prod_t pi(y_t) / mu(y_t), shape (B,)."""
    return torch.exp(((logp_target - logp_behavior) * mask).sum(dim=1))


def k3_mismatch(
    logp_train: torch.Tensor, logp_infer: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Per-sequence mean of the k3 estimator of KL(mu_infer || pi_train).

    With rho = pi_train / mu_infer at each sampled token, k3 = rho - 1 - log rho.

    Returns:
        shape (B,): the masked mean of k3 over each sequence's tokens.
    """
    log_rho = logp_train - logp_infer
    k3 = torch.exp(log_rho) - 1.0 - log_rho
    return (k3 * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def is_pg_loss(
    logp: torch.Tensor,
    logp_behavior: torch.Tensor,
    mask: torch.Tensor,
    advantages: torch.Tensor,
    level: str = "token",
    cap: float = float("inf"),
) -> torch.Tensor:
    """Importance-corrected REINFORCE loss for samples drawn from mu.

    Args:
        logp: log pi_theta of each sampled token, with gradients, shape (B, T).
        logp_behavior: log mu of the same tokens from the sampling engine,
            shape (B, T).
        mask: response mask, shape (B, T).
        advantages: one advantage per sequence, shape (B,).
        level: "token" weights token t by min(rho_t, cap). "sequence" weights
            every token of a sequence by min(prod_t rho_t, cap).
        cap: the truncation threshold C. Infinity means no truncation.

    Returns:
        A scalar: -(1 / B) * sum_i sum_t mask * w * A_i * logp. The weights
        are computed from logp.detach(), so no gradient flows through them.
    """
    log_rho = (logp.detach() - logp_behavior) * mask
    if level == "token":
        w = tis_weights(torch.exp(log_rho), cap)
    elif level == "sequence":
        w = tis_weights(torch.exp(log_rho.sum(dim=1, keepdim=True)), cap).expand_as(logp)
    else:
        raise ValueError(f"level must be 'token' or 'sequence', got {level!r}")
    per_token = mask * w * advantages[:, None] * logp
    return -per_token.sum(dim=1).mean()


def engine_logprobs(logits: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """A stand-in inference engine that rounds the logits to `dtype`.

    Args:
        logits: the trainer's logits, float64, shape (..., V).
        dtype: the engine's precision, such as torch.bfloat16 or torch.float16.

    Returns:
        log_softmax of the rounded logits, computed in float64, shape (..., V).
    """
    return torch.log_softmax(logits.to(dtype).to(torch.float64), dim=-1)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    mu = torch.tensor([0.1, 0.1, 0.8])      # the inference engine
    pi = torch.tensor([0.1, 0.3, 0.6])      # the trainer
    f = torch.tensor([1.0, 1.0, 0.0])
    rho = pi / mu
    print("ratios pi / mu:", [round(r, 3) for r in rho.tolist()])
    for name, w in [
        ("no correction", torch.ones(3)),
        ("IS", rho),
        ("TIS, C = 2", tis_weights(rho, 2.0)),
        ("MIS, [0.5, 5]", mis_weights(rho, 0.5, 5.0)),
        ("MIS, [0.5, 2]", mis_weights(rho, 0.5, 2.0)),
    ]:
        bias, var = estimator_bias_variance(mu, pi, f, w)
        print(f"{name:14s} bias {bias:+.3f}  variance {var:.3f}  "
              f"MSE at n = 16: {bias**2 + var / 16:.4f}  at n = 1000: {bias**2 + var / 1000:.5f}")

    g = torch.Generator().manual_seed(0)
    logits = 3.0 * torch.randn(2000, 16, generator=g)
    train = torch.log_softmax(logits, dim=-1)
    for dtype in (torch.bfloat16, torch.float16):
        infer = engine_logprobs(logits, dtype)
        kl = (infer.exp() * (infer - train)).sum(dim=-1).mean()
        print(f"{str(dtype):15s} mean KL(mu || pi) per token: {kl.item():.2e}")
