"""Infrastructure for LLM reinforcement learning.

Fill in the five functions that raise NotImplementedError, in order.

Fast RL systems sample with an inference engine and train with a separate
training engine. The engine's probabilities mu differ from the trainer's pi,
because of staleness (mu is an older policy) and numerical mismatch (the two
engines compute the same policy differently). Importance sampling corrects
for mu != pi, and truncation or masking bounds its variance at the price of
bias.

Every quantity here follows the layout of the language-model chapters:
per-token log-probabilities and masks are (B, T), and per-sequence values
are (B,).
"""

import torch

torch.set_num_threads(1)


def tis_weights(ratio: torch.Tensor, cap: float) -> torch.Tensor:
    """Truncated importance weights: min(ratio, cap), elementwise."""
    raise NotImplementedError("tis_weights")


def mis_weights(ratio: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Masked importance weights: the ratio inside [low, high], 0 outside."""
    raise NotImplementedError("mis_weights")


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
    raise NotImplementedError("estimator_bias_variance")


def sequence_ratio(
    logp_target: torch.Tensor, logp_behavior: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """The sequence-level importance ratio prod_t pi(y_t) / mu(y_t), shape (B,)."""
    raise NotImplementedError("sequence_ratio")


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
    raise NotImplementedError("is_pg_loss")


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
