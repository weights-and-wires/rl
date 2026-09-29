"""Reasoning with verifiable rewards: reference solution.

The pieces of an RLVR run: a rule that checks an answer, the unbiased pass@k
estimator, group advantages with an on-policy policy-gradient loss, and two
ways to watch entropy. The toy below is a tiny "language model" that answers
addition questions, so the checks can train it in well under a second.

Shapes follow the course's toy-harness convention:

- per-token log-probabilities and response masks are `(B, T)`,
- next-token logits are `(B, T, V)`,
- sequence rewards and advantages are `(B,)`.
"""

import math
import re
import time

import numpy as np
import torch

torch.set_num_threads(1)


# The toy arithmetic model. The same code sits in starter.py. ---------------

DIGITS = list(range(10))          # token ids 0-9 are the digits
METHODS = [10, 11, 12, 13]        # token ids 10-13 are four "methods"
VOCAB = 15                        # token id 14 is EOS, which the toy never samples


class ToyArithmeticLM(torch.nn.Module):
    """A two-token policy for 16 addition questions, a + b with a, b in 0..3.

    A response is (method, digit). The method token comes from one shared
    distribution that ignores the question: a habit. The digit comes from a
    table indexed by question and method. Each question has one suited
    method: ten questions suit method 0, and two each suit methods 1, 2, and
    3. In the base model the methods are equally likely, the suited method
    gives the right digit with probability `q`, and any other method with
    probability `eps`.
    """

    def __init__(self, q=0.8, eps=2e-3):
        super().__init__()
        a = torch.arange(4).repeat_interleave(4)
        b = torch.arange(4).repeat(4)
        self.questions = torch.stack([a, b], dim=1)            # (16, 2)
        self.answers = a + b                                   # (16,)
        self.suited = torch.tensor([0] * 10 + [1, 1, 2, 2, 3, 3])
        table = torch.empty(16, 4, 10)
        for j in range(16):
            for m in range(4):
                p_right = q if m == int(self.suited[j]) else eps
                table[j, m] = math.log((1.0 - p_right) / 9.0)
                table[j, m, self.answers[j]] = math.log(p_right)
        self.method_logits = torch.nn.Parameter(torch.zeros(4))
        self.digit_logits = torch.nn.Parameter(table)

    def logits(self, problems, tokens):
        """Next-token logits over the full vocabulary, shape (B, 2, VOCAB).

        Tokens the toy can't emit at a position get logit -inf.
        """
        batch = problems.shape[0]
        out = torch.full((batch, 2, VOCAB), float("-inf"))
        out[:, 0, METHODS[0] : METHODS[-1] + 1] = self.method_logits
        method = (tokens[:, 0] - METHODS[0]).clamp(0, 3)
        out[:, 1, : len(DIGITS)] = self.digit_logits[problems, method]
        return out

    def sample(self, problems, generator):
        """Sample one response per entry of `problems`, shape (B, 2)."""
        with torch.no_grad():
            p_method = torch.softmax(self.method_logits, dim=0).expand(problems.shape[0], 4)
            method = torch.multinomial(p_method, 1, generator=generator).squeeze(1)
            p_digit = torch.softmax(self.digit_logits[problems, method], dim=-1)
            digit = torch.multinomial(p_digit, 1, generator=generator).squeeze(1)
        return torch.stack([method + METHODS[0], digit], dim=1)

    def success_probs(self):
        """The exact probability that one sample answers each question right, (16,)."""
        with torch.no_grad():
            p_method = torch.softmax(self.method_logits, dim=0)
            p_digit = torch.softmax(self.digit_logits, dim=-1)
            p_right = p_digit[torch.arange(16), :, self.answers]   # (16, 4)
            return (p_right * p_method).sum(dim=1)


# The exercise. ------------------------------------------------------------


def pass_at_k(n, c, k):
    """Unbiased pass@k from n samples with c correct: 1 - C(n - c, k) / C(n, k)."""
    c_arr = np.asarray(c)
    out = np.ones(c_arr.shape, dtype=float)
    for idx in np.ndindex(c_arr.shape):
        ci = int(c_arr[idx])
        if n - ci >= k:
            # C(n-c, k) / C(n, k) = prod over j = n-c+1 .. n of (1 - k / j).
            out[idx] = 1.0 - np.prod(1.0 - k / np.arange(n - ci + 1, n + 1))
    return float(out) if out.ndim == 0 else out


INTEGER = re.compile(r"-?\d+")


def answer_reward(text, target, close_tag="</think>"):
    """1.0 when the response closes its thinking and answers target, else 0.0."""
    found = re.search(re.escape(close_tag) + r"\s*<answer>(.*?)</answer>", text, re.DOTALL)
    if found is None:
        return 0.0
    content = found.group(1).strip()
    if INTEGER.fullmatch(content) is None:
        return 0.0
    return float(int(content) == target)


def grpo_advantages(rewards, group_size, eps=1e-6):
    """Group-normalised advantages: (r - group mean) / (population std + eps)."""
    r = rewards.view(-1, group_size)
    a = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, unbiased=False, keepdim=True) + eps)
    return a.reshape(-1)


def pg_loss(token_logps, mask, advantages):
    """On-policy policy-gradient loss with a token-level mean."""
    mask = mask.to(token_logps.dtype)
    per_token = torch.where(mask > 0, advantages[:, None] * token_logps,
                            torch.zeros_like(token_logps))
    return -per_token.sum() / mask.sum()


def token_entropy(logits, mask):
    """Mean entropy of the next-token distribution over the masked positions."""
    logp = torch.log_softmax(logits, dim=-1)
    p = torch.exp(logp)
    h = -torch.where(p > 0, p * logp, torch.zeros_like(p)).sum(dim=-1)   # (B, T)
    mask = mask.to(h.dtype)
    return (h * mask).sum() / mask.sum()


def entropy_change_estimate(logits, delta_logits):
    """First-order change in entropy: -Cov_pi(log pi, delta z), along the last axis."""
    logp = torch.log_softmax(logits, dim=-1)
    p = torch.exp(logp)
    mean_logp = (p * logp).sum(dim=-1, keepdim=True)
    mean_delta = (p * delta_logits).sum(dim=-1, keepdim=True)
    return -(p * (logp - mean_logp) * (delta_logits - mean_delta)).sum(dim=-1)


def train_rlvr_toy(steps=200, group_size=8, lr=1.0, optimizer="sgd", seed=0, ks=(1, 8, 64)):
    """RLVR on the toy: GRPO advantages, one on-policy step per batch."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = ToyArithmeticLM()
    if optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    problems = torch.arange(16).repeat_interleave(group_size)
    mask = torch.ones(problems.shape[0], 2)

    def pass_row():
        p = model.success_probs().double().numpy()
        # The n -> infinity limit of the unbiased estimator: 1 - (1 - p)^k.
        return [float(np.mean(1.0 - (1.0 - p) ** k)) for k in ks]

    history = {"pass@k": [pass_row()], "entropy": [], "reward": []}
    for _ in range(steps):
        tokens = model.sample(problems, generator)
        rewards = (tokens[:, 1] == model.answers[problems]).float()
        adv = grpo_advantages(rewards, group_size)
        logits = model.logits(problems, tokens)
        logp = torch.log_softmax(logits, dim=-1).gather(-1, tokens[..., None]).squeeze(-1)
        loss = pg_loss(logp, mask, adv)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history["entropy"].append(float(token_entropy(logits.detach(), mask)))
        history["reward"].append(float(rewards.mean()))
        history["pass@k"].append(pass_row())
    return {
        "ks": list(ks),
        "pass@k": np.array(history["pass@k"]),
        "entropy": np.array(history["entropy"]),
        "reward": np.array(history["reward"]),
    }


# The real run on a GPU. ---------------------------------------------------

# R1-Zero's template, with <reasoning> in place of <think>. Qwen3's tokenizer
# turns "<think>" and "</think>" into two reserved tokens that the base model
# almost never saw in pretraining, so it never writes "</think>" and every
# reward is zero. Plain-text tags are ordinary tokens it can write.
OPEN_TAG, CLOSE_TAG = "<reasoning>", "</reasoning>"
TEMPLATE = (
    "A conversation between User and Assistant. The User asks a question, and the "
    "Assistant solves it. The Assistant first thinks about the reasoning process in "
    "the mind and then provides the User with the answer. The reasoning process is "
    f"enclosed within {OPEN_TAG} {CLOSE_TAG} and answer is enclosed within <answer> "
    f"</answer> tags, respectively, i.e., {OPEN_TAG} reasoning process here {CLOSE_TAG} "
    "<answer> answer here </answer>. User: {question} Assistant: " + OPEN_TAG
)


def _questions(n, generator):
    a = torch.randint(2, 10, (n,), generator=generator)
    b = torch.randint(10, 100, (n,), generator=generator)
    c = torch.randint(10, 100, (n,), generator=generator)
    texts = [f"What is {x} * {y} + {z}?" for x, y, z in zip(a.tolist(), b.tolist(), c.tolist())]
    return texts, (a * b + c).tolist()


def _response_mask(gen_ids, eos_ids, pad_id):
    """1 on generated tokens, up to and including the first end token."""
    is_eos = torch.zeros_like(gen_ids, dtype=torch.bool)
    for e in eos_ids:
        is_eos |= gen_ids == e
    before_eos = torch.cumsum(is_eos.int(), dim=1) - is_eos.int()
    not_pad = torch.cumsum((gen_ids == pad_id).int(), dim=1) == 0
    return ((before_eos == 0) & not_pad).float()


def real_run(model_id="Qwen/Qwen3-0.6B-Base", device="cuda", steps=40, prompts_per_step=8,
             group_size=8, micro_batch=8, max_new_tokens=160, lr=1e-6, eval_every=10,
             eval_questions=16, eval_samples=16, seed=0, time_limit=480.0):
    """R1-Zero-style RLVR: a base model, the R1-Zero template, and a rule-based reward."""
    from contextlib import nullcontext

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    started = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Pad generated sequences with a token the model never writes, so the mask
    # can tell padding from a real end token.
    gen_pad_id = tok.convert_tokens_to_ids("<|vision_pad|>")
    policy = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device)
    policy.gradient_checkpointing_enable()
    eos_ids = policy.generation_config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
    if device.startswith("cuda"):
        def autocast():
            return torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast = nullcontext

    def generate(texts, n):
        enc = tok([TEMPLATE.format(question=q) for q in texts], return_tensors="pt",
                  padding=True).to(device)
        ids = enc.input_ids.repeat_interleave(n, dim=0)
        attn = enc.attention_mask.repeat_interleave(n, dim=0)
        policy.eval()
        with torch.no_grad(), autocast():
            out = policy.generate(input_ids=ids, attention_mask=attn, max_new_tokens=max_new_tokens,
                                  do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                  stop_strings=["</answer>"], tokenizer=tok,
                                  pad_token_id=gen_pad_id)
        gen_ids = out[:, ids.shape[1]:]
        mask = _response_mask(gen_ids, eos_ids, gen_pad_id)
        decoded = tok.batch_decode(gen_ids, skip_special_tokens=True)
        return out, attn, ids.shape[1], mask, decoded

    eval_texts, eval_targets = _questions(eval_questions, torch.Generator().manual_seed(seed + 1))
    ks = [k for k in (1, 4, 16) if k <= eval_samples]

    def evaluate():
        correct = []
        for lo in range(0, eval_questions, 4):
            _, _, _, _, decoded = generate(eval_texts[lo : lo + 4], eval_samples)
            for q in range(len(decoded) // eval_samples):
                block = decoded[q * eval_samples : (q + 1) * eval_samples]
                correct.append(sum(answer_reward(t, eval_targets[lo + q], CLOSE_TAG) for t in block))
        c = np.array(correct)
        return [float(np.mean(pass_at_k(eval_samples, c, k))) for k in ks]

    print(f"R1-Zero-style RLVR on {model_id}: a * b + c, reward 1 for a correct <answer>")
    header = " ".join(f"{'pass@' + str(k):>8}" for k in ks)
    print(f"{'step':>4} {'reward':>7} {'length':>7} {header} {'secs':>6}")

    def report(step, reward, length):
        row = " ".join(f"{v:>8.3f}" for v in evaluate())
        print(f"{step:>4} {reward:>7.3f} {length:>7.1f} {row} {time.time() - started:>6.0f}")

    report(0, float("nan"), float("nan"))
    for step in range(1, steps + 1):
        texts, targets = _questions(prompts_per_step, gen)
        out, attn, n_prompt, mask, decoded = generate(texts, group_size)
        targets = [t for t in targets for _ in range(group_size)]
        rewards = torch.tensor([answer_reward(t, v, CLOSE_TAG) for t, v in zip(decoded, targets)])
        adv = grpo_advantages(rewards, group_size).to(device)
        full_attn = torch.cat([attn, mask.to(attn.dtype)], dim=1)

        policy.train()
        optimizer.zero_grad(set_to_none=True)
        total_tokens = float(mask.sum())
        for lo in range(0, out.shape[0], micro_batch):
            sl = slice(lo, lo + micro_batch)
            with autocast():
                logits = policy(input_ids=out[sl], attention_mask=full_attn[sl],
                                use_cache=False).logits
            logits = logits[:, n_prompt - 1 : -1].float()
            targets_ids = out[sl, n_prompt:]
            logp = torch.log_softmax(logits, dim=-1).gather(-1, targets_ids[..., None]).squeeze(-1)
            # Scale each micro-batch so the sum equals the token mean over the batch.
            loss = pg_loss(logp, mask[sl], adv[sl]) * (float(mask[sl].sum()) / total_tokens)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        if step % eval_every == 0 or step == steps or time.time() - started > time_limit:
            report(step, float(rewards.mean()), float(mask.sum(dim=1).mean()))
        if time.time() - started > time_limit:
            print(f"stopping at step {step}: the {time_limit:.0f} s budget is spent")
            break


def toy_run():
    """The CPU fallback: RLVR on the toy, with pass@k and entropy."""
    hist = train_rlvr_toy()
    ks = hist["ks"]
    print(f"{'step':>4} " + " ".join(f"{'pass@' + str(k):>8}" for k in ks) + f" {'entropy':>8}")
    for step in (0, 10, 25, 50, 100, 200):
        ent = hist["entropy"][max(step - 1, 0)]
        row = " ".join(f"{v:>8.3f}" for v in hist["pass@k"][step])
        print(f"{step:>4} {row} {ent:>8.3f}")


if __name__ == "__main__":
    try:
        import transformers  # noqa: F401
        have_transformers = True
    except ImportError:
        have_transformers = False
    if torch.cuda.is_available() and have_transformers:
        real_run()
    else:
        print("The real run needs the GPU worker. Running the toy on the CPU instead.")
        toy_run()
