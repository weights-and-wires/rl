"""Group-relative policy optimization and the Dr. GRPO fixes: reference solution.

Every function works on plain tensors, so the same code trains the toy in the
checks and a real language model in the GPU run. Shapes follow the course's
toy-harness convention:

- per-token log-probabilities and response masks are `(B, T)`,
- sequence rewards and advantages are `(B,)`.

A batch holds `B = P * G` responses: `P` prompts with `G` responses each,
stored group by group, so `rewards.view(P, G)` puts one group in each row.
"""

import math
import re
import time

import torch

torch.set_num_threads(1)


# The exercise. ------------------------------------------------------------


def rloo_advantages(rewards, group_size):
    """Leave-one-out advantages: each reward minus the mean of the others in its group."""
    r = rewards.view(-1, group_size)
    g = group_size
    return (g / (g - 1) * (r - r.mean(dim=1, keepdim=True))).reshape(-1)


def grpo_advantages(rewards, group_size, scale_by_std=True, eps=1e-6):
    """Group-normalised advantages: (r - group mean) / (group std + eps)."""
    r = rewards.view(-1, group_size)
    a = r - r.mean(dim=1, keepdim=True)
    if scale_by_std:
        a = a / (r.std(dim=1, unbiased=False, keepdim=True) + eps)
    return a.reshape(-1)


def k3_kl(logp, ref_logp):
    """Schulman's k3 estimator per token: pi_ref / pi - log(pi_ref / pi) - 1."""
    log_ratio = ref_logp - logp
    return torch.exp(log_ratio) - log_ratio - 1.0


def clipped_surrogate(logp_new, logp_old, advantages, eps_low=0.2, eps_high=0.2):
    """Per-token min(rho * A, clip(rho, 1 - eps_low, 1 + eps_high) * A)."""
    ratio = torch.exp(logp_new - logp_old)
    adv = advantages[:, None]
    clipped = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high)
    return torch.minimum(ratio * adv, clipped * adv)


def aggregate(per_token, mask, mode, max_tokens=None):
    """Reduce a (B, T) array of per-token terms to one number."""
    mask = mask.to(per_token.dtype)
    masked = torch.where(mask > 0, per_token, torch.zeros_like(per_token))
    if mode == "sequence_mean":
        return (masked.sum(dim=1) / mask.sum(dim=1)).mean()
    if mode == "token_mean":
        return masked.sum() / mask.sum()
    if mode == "constant":
        if max_tokens is None:
            max_tokens = mask.shape[1]
        return (masked.sum(dim=1) / max_tokens).mean()
    raise ValueError(f"unknown aggregation mode {mode!r}")


def grpo_loss(logp_new, logp_old, advantages, mask, ref_logp=None, beta=0.0,
              eps_low=0.2, eps_high=0.2, mode="sequence_mean", max_tokens=None):
    """The GRPO loss to minimise: minus the aggregated clipped surrogate, plus beta * k3."""
    per_token = clipped_surrogate(logp_new, logp_old, advantages, eps_low, eps_high)
    if ref_logp is not None and beta != 0.0:
        per_token = per_token - beta * k3_kl(logp_new, ref_logp)
    return -aggregate(per_token, mask, mode, max_tokens)


def length_bias_toy(mode, steps=300, lr=0.5):
    """Run the two-step length toy with exact expected gradients.

    The response is one correctness token, one length token L in 1..8, and
    L - 1 filler tokens, so it has L + 1 tokens. Correctness is frozen at
    P(right) = 0.5, and the reward is 1 when right. Both objectives use the
    mean-centred advantage r - 0.5, so only the aggregation differs.
    """
    dtype = torch.float64
    lengths = torch.arange(1, 9)
    logits = torch.zeros(2, 8, dtype=dtype, requires_grad=True)   # row 0 wrong, row 1 right
    advantage = torch.tensor([-0.5, 0.5], dtype=dtype)
    correct = torch.arange(2).repeat_interleave(8)                  # (16,)
    length = lengths.repeat(2)                                      # (16,)
    max_tokens = 9
    mask = (torch.arange(max_tokens)[None, :] < (length + 1)[:, None]).to(dtype)
    for _ in range(steps):
        logp_len = torch.log_softmax(logits, dim=1)
        # The probability of each of the 16 outcomes, a constant in the objective.
        prob = (0.5 * torch.exp(logp_len)).detach().reshape(-1)
        token_logps = torch.zeros(16, max_tokens, dtype=dtype)
        token_logps[:, 0] = math.log(0.5)                            # frozen correctness token
        token_logps[:, 1] = logp_len[correct, length - 1]            # the length token
        # A batch mean over the 16 rows, weighted by 16 * prob, is the exact expectation.
        weight = 16.0 * prob * advantage[correct]
        objective = aggregate(weight[:, None] * token_logps, mask, mode, max_tokens)
        (grad,) = torch.autograd.grad(objective, logits)
        with torch.no_grad():
            logits += lr * grad
    probs = torch.softmax(logits.detach(), dim=1)
    mean_len = (probs * lengths.to(dtype)).sum(dim=1)
    return float(mean_len[0]), float(mean_len[1])


# The real run on a GPU. ---------------------------------------------------

ANSWER = re.compile(r"Answer:\s*\**\s*(-?[\d,]+)")


def multiplication_reward(text, target):
    """1.0 when the last 'Answer: N' in the text equals target, else 0.0."""
    found = ANSWER.findall(text)
    if not found:
        return 0.0
    try:
        return float(int(found[-1].replace(",", "")) == target)
    except ValueError:
        return 0.0


def _response_mask(gen_ids, eos_ids):
    """1 for generated tokens up to and including the first end token."""
    is_eos = torch.zeros_like(gen_ids, dtype=torch.bool)
    for e in eos_ids:
        is_eos |= gen_ids == e
    # Count end tokens strictly before each position; keep positions with none.
    before = torch.cumsum(is_eos.int(), dim=1) - is_eos.int()
    return (before == 0).float()


def _token_logps(model, input_ids, attention_mask, n_prompt, autocast):
    with autocast():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    logits = logits[:, n_prompt - 1 : -1].float()
    targets = input_ids[:, n_prompt:]
    return torch.log_softmax(logits, dim=-1).gather(-1, targets[..., None]).squeeze(-1)


def real_run(model_id="Qwen/Qwen3-0.6B", device="cuda", steps=40, prompts_per_step=8,
             group_size=8, micro_batch=8, mode="grpo", max_new_tokens=96, lr=1e-6,
             beta=0.04, seed=0, time_limit=540.0):
    """GRPO on a real model: 2-digit multiplication with a rule-based reward.

    mode="grpo" uses the original aggregation: std-scaled advantages and a
    per-sequence mean over tokens. mode="dr_grpo" uses the Dr. GRPO fixes:
    mean-centred advantages and a constant normaliser, max_new_tokens.
    """
    from contextlib import nullcontext

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    started = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    # The policy keeps float32 master weights; forward passes run in bfloat16.
    policy = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device)
    policy.gradient_checkpointing_enable()
    ref = None
    if beta > 0:
        ref = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
        ref.eval()
    eos_ids = policy.generation_config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
    if device.startswith("cuda"):
        def autocast():
            return torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast = nullcontext

    scale_by_std = mode == "grpo"
    agg = "sequence_mean" if mode == "grpo" else "constant"
    print(f"GRPO on {model_id}, mode={mode}: advantages "
          f"{'scaled by std' if scale_by_std else 'mean-centred'}, aggregation {agg}")
    print(f"{'step':>4} {'reward':>7} {'len right':>9} {'len wrong':>9} {'KL':>7} {'secs':>6}")

    for step in range(1, steps + 1):
        a = torch.randint(10, 100, (prompts_per_step,), generator=gen)
        b = torch.randint(10, 100, (prompts_per_step,), generator=gen)
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": f"What is {x} * {y}? Work it out briefly, "
                  "then end with 'Answer: ' followed by the number."}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for x, y in zip(a.tolist(), b.tolist())
        ]
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        prompt_ids = enc.input_ids.repeat_interleave(group_size, dim=0)
        prompt_mask = enc.attention_mask.repeat_interleave(group_size, dim=0)
        policy.eval()
        with torch.no_grad(), autocast():
            out = policy.generate(input_ids=prompt_ids, attention_mask=prompt_mask,
                                  max_new_tokens=max_new_tokens, do_sample=True,
                                  temperature=1.0, top_p=1.0, top_k=0,
                                  pad_token_id=tok.pad_token_id)
        n_prompt = prompt_ids.shape[1]
        gen_ids = out[:, n_prompt:]
        mask = _response_mask(gen_ids, eos_ids)
        targets = (a * b).repeat_interleave(group_size)
        decoded = tok.batch_decode(gen_ids, skip_special_tokens=True)
        rewards = torch.tensor([multiplication_reward(t, int(v)) for t, v in zip(decoded, targets)])
        adv = grpo_advantages(rewards, group_size, scale_by_std=scale_by_std).to(device)
        attn = torch.cat([prompt_mask, mask.to(prompt_mask.dtype)], dim=1)

        # One gradient step per batch, accumulated over micro-batches so the
        # result equals the loss on the whole batch.
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        n_seq = out.shape[0]
        kl_sum = 0.0
        for lo in range(0, n_seq, micro_batch):
            sl = slice(lo, lo + micro_batch)
            logp = _token_logps(policy, out[sl], attn[sl], n_prompt, autocast)
            ref_logp = None
            if ref is not None:
                with torch.no_grad():
                    ref_logp = _token_logps(ref, out[sl], attn[sl], n_prompt, autocast)
                kl_sum += float((k3_kl(logp.detach(), ref_logp) * mask[sl]).sum())
            loss = grpo_loss(logp, logp.detach(), adv[sl], mask[sl], ref_logp=ref_logp,
                             beta=beta, mode=agg, max_tokens=max_new_tokens)
            (loss * (logp.shape[0] / n_seq)).backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        lengths = mask.sum(dim=1).cpu()
        right = rewards > 0.5
        len_right = float(lengths[right].mean()) if right.any() else float("nan")
        len_wrong = float(lengths[~right].mean()) if (~right).any() else float("nan")
        kl = kl_sum / float(mask.sum())
        elapsed = time.time() - started
        if step == 1 or step % 5 == 0 or step == steps:
            print(f"{step:>4} {rewards.mean():>7.3f} {len_right:>9.1f} {len_wrong:>9.1f} "
                  f"{kl:>7.4f} {elapsed:>6.0f}")
        if elapsed > time_limit:
            print(f"stopping at step {step}: the {time_limit:.0f} s budget is spent")
            break


def toy_run():
    """The CPU fallback: the length-bias toy under both aggregations."""
    print(f"{'objective':<28} {'len if wrong':>12} {'len if right':>12}")
    for label, mode in (("GRPO, 1/|o| per sequence", "sequence_mean"),
                        ("Dr. GRPO, constant", "constant")):
        wrong, right = length_bias_toy(mode)
        print(f"{label:<28} {wrong:>12.2f} {right:>12.2f}")


if __name__ == "__main__":
    try:
        import transformers  # noqa: F401
        have_transformers = True
    except ImportError:
        have_transformers = False
    if torch.cuda.is_available() and have_transformers:
        import gc

        # Two runs share the ten-minute budget.
        for mode in ("grpo", "dr_grpo"):
            real_run(mode=mode, steps=20, time_limit=270.0)
            gc.collect()
            torch.cuda.empty_cache()
    else:
        print("The real run needs the GPU worker. Running the toy on the CPU instead.")
        toy_run()
