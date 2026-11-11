"""Asynchronous RL for language models.

Fill in the nine functions that raise NotImplementedError, in order.

Fast RL systems sample with an inference engine and train with a separate
training engine. The engine's probabilities mu differ from the trainer's pi,
because of staleness (mu is an older policy) and numerical mismatch (the two
engines compute the same policy differently). Importance sampling corrects
for mu != pi, and truncation or masking bounds its variance at the price of
bias.

The second half of the file adds the systems pieces: a queue simulation with a
staleness cap, a measurement of the gap between a bf16 sampler and an fp32
trainer, and sequence packing for rollouts.

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


def async_schedule(n_updates: int, gen_time: float, train_time: float, max_staleness: int) -> dict:
    """Simulate one sampler and one trainer joined by a queue.

    Batch j is the batch that update j consumes. The sampler generates batches
    one after another, each taking gen_time. The trainer runs updates one after
    another, each taking train_time, and update j finishing produces policy
    version j + 1. Weight sync is instant.

    Rules:
        * The sampler starts batch j at the earliest time it is free and, when
          j - max_staleness - 1 >= 0, update j - max_staleness - 1 has finished.
        * It generates with the newest version finished by its start time
          (a finish exactly at the start time counts).
        * Update j starts when batch j is ready and update j - 1 has finished.
        * The staleness of batch j is j minus the version that generated it.

    Returns:
        A dict with "versions" (list of ints, the version that generated each
        batch), "staleness" (list of ints), "makespan" (finish time of the last
        update), "trainer_utilization" (n_updates * train_time / makespan), and
        "sampler_utilization" (n_updates * gen_time / makespan).
    """
    raise NotImplementedError("async_schedule")


def logprob_mismatch(
    logits_fp32: torch.Tensor, logits_lowprec: torch.Tensor
) -> tuple[torch.Tensor, float]:
    """Measure the gap between the trainer (fp32) and the sampler (low precision).

    Args:
        logits_fp32: the trainer's logits, shape (N, V).
        logits_lowprec: the sampler's logits for the same positions, shape
            (N, V). The sampler's distribution is mu = softmax(logits_lowprec),
            and the trainer's is pi = softmax(logits_fp32).

    Returns:
        (kl, max_abs): kl has shape (N,) and holds KL(mu || pi) per position,
        sum_v mu(v) (log mu(v) - log pi(v)). max_abs is the largest
        |log mu(v) - log pi(v)| over every position and token, as a float.
    """
    raise NotImplementedError("logprob_mismatch")


def toy_logits(tokens, position_ids, attn_mask, seed=0):
    """A one-layer attention model, given. Returns logits of shape (L, 8)."""
    gen = torch.Generator().manual_seed(seed)
    width, vocab = 4, 8
    token_table = torch.randn(16, width, generator=gen)
    position_table = torch.randn(16, width, generator=gen)
    query_w = torch.randn(width, width, generator=gen) / width**0.5
    key_w = torch.randn(width, width, generator=gen) / width**0.5
    value_w = torch.randn(width, width, generator=gen) / width**0.5
    out_w = torch.randn(width, vocab, generator=gen) / width**0.5
    hidden = token_table[tokens] + position_table[position_ids]
    scores = (hidden @ query_w) @ (hidden @ key_w).T / width**0.5
    scores = torch.where(attn_mask > 0, scores, torch.full_like(scores, -1e9))
    weights = torch.softmax(scores, dim=-1)
    return (weights @ (hidden @ value_w)) @ out_w


def pack_rollouts(token_rows, response_rows, advantages):
    """Concatenate variable-length rollouts into one packed row.

    Args:
        token_rows: list of 1D int tensors, prompt plus response, one per rollout.
        response_rows: list of 1D tensors, 1 on response tokens and 0 on the
            prompt. Row i has the length of token_rows[i].
        advantages: list of floats, one per rollout.

    Returns:
        tokens (L,) int64, position_ids (L,) int64 restarting at 0 for each
        rollout, attn_mask (L, L) float64 with a causal block per rollout,
        targets (L,) int64, and weights (L,) float64. Position i predicts the
        next token of its own rollout: targets[i] = tokens[i + 1] and
        weights[i] = A_k * response[i + 1] / (number of response tokens in
        rollout k). The last position of each rollout has weight 0 and target 0.
    """
    raise NotImplementedError("pack_rollouts")


def rollout_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """-sum_i weights[i] * log softmax(logits[i])[targets[i]], a scalar.

    logits has shape (L, V), targets and weights have shape (L,).
    """
    raise NotImplementedError("rollout_loss")

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

    print()
    out = async_schedule(6, gen_time=1.0, train_time=2.0, max_staleness=1)
    sync = async_schedule(6, gen_time=1.0, train_time=2.0, max_staleness=0)
    print("async, K = 1: staleness", out["staleness"],
          "trainer utilization", round(out["trainer_utilization"], 3))
    print("sync,  K = 0: staleness", sync["staleness"],
          "trainer utilization", round(sync["trainer_utilization"], 3))
    g = torch.Generator().manual_seed(0)
    logits = 3.0 * torch.randn(1000, 32, generator=g)
    kl, max_abs = logprob_mismatch(logits, logits.to(torch.bfloat16).to(torch.float64))
    print(f"bf16 logits: mean KL {kl.mean():.2e}, max |log-prob gap| {max_abs:.4f}")
    rows = [torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6, 7])]
    resp = [torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 1])]
    adv = [1.0, -0.5]
    packed = pack_rollouts(rows, resp, adv)
    whole = rollout_loss(toy_logits(*packed[:3]), packed[3], packed[4])
    parts = 0.0
    for r, s, a in zip(rows, resp, adv):
        one = pack_rollouts([r], [s], [a])
        parts = parts + rollout_loss(toy_logits(*one[:3]), one[3], one[4])
    print(f"packed loss {float(whole):.10f}, sum of separate losses {float(parts):.10f}")
