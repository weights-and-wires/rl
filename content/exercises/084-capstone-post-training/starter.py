"""Capstone: LoRA, a two-stage post-training pipeline, and a written evaluation.

Fill in the functions under "The exercise". The helpers above them, the toy
pipeline, and the real run are given: they call your functions on a tiny
categorical language model, and on Qwen3-0.6B when a GPU is available.

Conventions:

- A linear layer's weight `W` has shape `(out, in)` and computes `x @ W.T`.
- A LoRA adapter has `A` of shape `(r, in)` and `B` of shape `(out, r)`.
  The adapted layer computes `x @ (W + (alpha / r) * B @ A).T`.
- The toy language model has 4 tokens. A sequence is a prompt token followed by
  3 response tokens, and its next-token logits depend only on the token before.
  Its logit table `table[prev, next]` has shape `(4, 4)`, and it is one adapted
  layer applied to one-hot inputs.
"""

import itertools
import math
import re
import time

import torch

torch.set_num_threads(1)
D = torch.float64

V = 4            # toy vocabulary size, and the number of toy prompts
L = 3            # toy response length
RESPONSES = torch.tensor(list(itertools.product(range(V), repeat=L)))   # (64, 3)


# Helpers given to you. The same code sits in starter.py. -----------------


def table_logits(W, A, B, alpha, r):
    """The toy model's `(V, V)` logit table: row `prev` scores the next token."""
    return lora_linear(torch.eye(V, dtype=W.dtype), W, A, B, alpha, r)


def token_logps(table, seqs):
    """Log-probability of each response token given the one before: `(N, L)`.

    `seqs` is `(N, L + 1)`: a prompt token, then L response tokens.
    """
    logp = torch.log_softmax(table, dim=-1)
    return logp[seqs[:, :-1], seqs[:, 1:]]


def response_logps(table):
    """Log-probability of every response to every prompt: `(V, 64)`. Rows sum to 1 in probability."""
    prompts = torch.arange(V).repeat_interleave(len(RESPONSES))
    seqs = torch.cat([prompts[:, None], RESPONSES.repeat(V, 1)], dim=1)
    return token_logps(table, seqs).sum(dim=1).reshape(V, len(RESPONSES))


def sft_loss(logps, mask):
    """Mean negative log-likelihood over response tokens: `(N, T)` log-probabilities and mask."""
    return -(logps * mask).sum() / mask.sum()


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    """Mean of -log sigmoid(beta * h), where h is the log-ratio margin."""
    h = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
    return -torch.nn.functional.logsigmoid(beta * h).mean()


def sample_responses(table, prompts, generator):
    """Sample one response per prompt from the toy model: `(N, L + 1)` sequences."""
    seqs = prompts[:, None]
    probs = torch.softmax(table, dim=-1)
    for _ in range(L):
        nxt = torch.multinomial(probs[seqs[:, -1]], 1, generator=generator)
        seqs = torch.cat([seqs, nxt], dim=1)
    return seqs


def toy_reward(seqs):
    """The toy preference signal: the number of response tokens equal to 0."""
    return (seqs[:, 1:] == 0).sum(dim=1).to(D)


# The exercise. ------------------------------------------------------------


def lora_linear(x, W, A, B, alpha, r):
    """A linear layer with a LoRA adapter: x @ W.T + (alpha / r) * (x @ A.T) @ B.T.

    Shapes: x (..., in), W (out, in), A (r, in), B (out, r). Returns (..., out).
    It never forms the (out, in) matrix B @ A.
    """
    raise NotImplementedError


def init_lora(in_features, out_features, r, seed):
    """Initialize a LoRA pair so the adapted layer starts equal to the base layer.

    Returns (A, B) in float64. A has shape (r, in) and holds normal draws with
    standard deviation 1 / sqrt(in), made with torch.Generator().manual_seed(seed).
    B has shape (out, r) and is all zeros.
    """
    raise NotImplementedError


def lora_param_count(shapes, r):
    """Trainable LoRA parameters for a list of (out, in) layer shapes at rank r."""
    raise NotImplementedError


def lora_grads(x, W, A, B, alpha, r, y):
    """Gradients (dA, dB) of 0.5 * sum((lora_linear(x, W, A, B, alpha, r) - y) ** 2).

    The base weight W is frozen: no gradient is computed for it.
    """
    raise NotImplementedError


def train_adapter(loss_fn, W, A, B, alpha, r, steps, lr):
    """Gradient descent on the adapter only, and return the new (A, B).

    `loss_fn` takes the (V, V) logit table of the adapted layer and returns a
    scalar. W is frozen. A and B start from the arguments, which stay unchanged.
    """
    raise NotImplementedError


def sequence_kl(logp, logq):
    """KL(p || q) from log-probabilities over an enumerated set: sum p * (log p - log q).

    Shapes (..., N). Returns the mean over the leading axes, a Python float.
    """
    raise NotImplementedError


def pass_at_k(n, c, k):
    """Unbiased pass@k from n samples with c correct: 1 - C(n - c, k) / C(n, k)."""
    raise NotImplementedError


def judge_win_rate(pref_ab, pref_ba):
    """Win rate of model A over model B, and the judge's swap consistency.

    `pref_ab[i]` is the verdict on pair i with A shown first: +1 the first
    response wins, -1 the second wins, 0 a tie. `pref_ba[i]` is the verdict on
    the same pair with B shown first. A tie counts half a win. Average over
    both orders, so a judge that always prefers the first position scores 0.5.
    Returns (win_rate, swap_consistency), both floats. A pair is consistent
    when the winner is the same in both orders.
    """
    raise NotImplementedError


# The toy pipeline, given: SFT, then DPO, with a LoRA adapter on one layer. ---


def run_toy_pipeline(seed=0, r=2, alpha=4.0, sft_steps=400, dpo_steps=40,
                     lr_sft=0.5, lr_dpo=0.02, beta=0.5):
    """SFT on demonstrations, then DPO on preference pairs, training only A and B.

    Returns a dict of the numbers the checks and the figure read.
    """
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(V, V, generator=g, dtype=D) * 0.5          # the frozen base model
    W_before = W.clone()

    # The "teacher" that wrote the demonstrations is the base plus a rank-r change.
    At = torch.randn(r, V, generator=g, dtype=D)
    Bt = torch.randn(V, r, generator=g, dtype=D)
    teacher = table_logits(W, At, Bt, alpha, r)
    prompts = torch.randint(0, V, (96,), generator=g)
    demos = sample_responses(teacher, prompts, g)
    ones = torch.ones(len(demos), L, dtype=D)

    A, B = init_lora(V, V, r, seed)
    base_table = W.T.contiguous()
    init_gap = float((table_logits(W, A, B, alpha, r) - base_table).abs().max())

    def nll(table):
        return sft_loss(token_logps(table, demos), ones)

    sft_nll_before = float(nll(table_logits(W, A, B, alpha, r)))
    A, B = train_adapter(nll, W, A, B, alpha, r, sft_steps, lr_sft)
    sft_table = table_logits(W, A, B, alpha, r).detach()
    sft_nll_after = float(nll(sft_table))

    # Preference pairs come from the SFT policy, scored by toy_reward. Ties are dropped.
    def make_pairs(n):
        p = torch.randint(0, V, (n,), generator=g)
        s1 = sample_responses(sft_table, p, g)
        s2 = sample_responses(sft_table, p, g)
        r1, r2 = toy_reward(s1), toy_reward(s2)
        keep = r1 != r2
        s1, s2, r1, r2 = s1[keep], s2[keep], r1[keep], r2[keep]
        win = (r1 > r2)[:, None]
        return torch.where(win, s1, s2), torch.where(win, s2, s1)

    chosen, rejected = make_pairs(400)
    held_c, held_r = make_pairs(400)
    ref_c = token_logps(sft_table, chosen).sum(dim=1)
    ref_r = token_logps(sft_table, rejected).sum(dim=1)

    def dpo(table):
        return dpo_loss(token_logps(table, chosen).sum(dim=1),
                        token_logps(table, rejected).sum(dim=1), ref_c, ref_r, beta)

    def margin(table):
        return float((token_logps(table, held_c).sum(dim=1)
                      - token_logps(table, held_r).sum(dim=1)).mean())

    A, B = train_adapter(dpo, W, A, B, alpha, r, dpo_steps, lr_dpo)
    dpo_table = table_logits(W, A, B, alpha, r).detach()

    lp_sft, lp_dpo = response_logps(sft_table), response_logps(dpo_table)
    reward = toy_reward(torch.cat([torch.arange(V).repeat_interleave(len(RESPONSES))[:, None],
                                   RESPONSES.repeat(V, 1)], dim=1)).reshape(V, -1)
    return {
        "init_gap": init_gap,
        "sft_nll_before": sft_nll_before,
        "sft_nll_after": sft_nll_after,
        "margin_before": margin(sft_table),
        "margin_after": margin(dpo_table),
        "kl_to_sft": sequence_kl(lp_dpo, lp_sft),
        "reward_before": float((torch.exp(lp_sft) * reward).sum(dim=1).mean()),
        "reward_after": float((torch.exp(lp_dpo) * reward).sum(dim=1).mean()),
        "w_drift": float((W - W_before).abs().max()),
    }


# The real run on a GPU, given. --------------------------------------------

ANSWER = re.compile(r"Answer:\s*\**\s*(-?[\d,]+)")
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def multiplication_reward(text, target):
    """1.0 when the last 'Answer: N' in the text equals target, else 0.0."""
    found = ANSWER.findall(text)
    if not found:
        return 0.0
    try:
        return float(int(found[-1].replace(",", "")) == target)
    except ValueError:
        return 0.0


class LoRALinear(torch.nn.Module):
    """A frozen nn.Linear plus a trainable LoRA pair, computed by your lora_linear."""

    def __init__(self, base, r, alpha, seed):
        super().__init__()
        self.base, self.r, self.alpha = base, r, alpha
        A, B = init_lora(base.in_features, base.out_features, r, seed)
        device = base.weight.device
        self.A = torch.nn.Parameter(A.float().to(device))   # float32 masters
        self.B = torch.nn.Parameter(B.float().to(device))

    def forward(self, x):
        out = lora_linear(x, self.base.weight, self.A, self.B, self.alpha, self.r)
        return out if self.base.bias is None else out + self.base.bias


def attach_lora(model, r, alpha, seed=0):
    """Freeze the model, wrap every target projection, and return the wrapped shapes."""
    model.requires_grad_(False)
    shapes = []
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if name in TARGETS and isinstance(child, torch.nn.Linear):
                setattr(parent, name, LoRALinear(child, r, alpha, seed + len(shapes)))
                shapes.append((child.out_features, child.in_features))
    return shapes


def adapter_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def adapter_state(model):
    return [p.detach().clone() for p in adapter_params(model)]


def load_adapter(model, state):
    with torch.no_grad():
        for p, s in zip(adapter_params(model), state):
            p.copy_(s)


def collate(prompts, responses, pad_id, device):
    """Right-pad prompt + response ids. Returns ids, attention mask, response mask."""
    rows = [p + r for p, r in zip(prompts, responses)]
    width = max(len(row) for row in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attn = torch.zeros(len(rows), width, dtype=torch.long)
    resp = torch.zeros(len(rows), width)
    for i, (p, r) in enumerate(zip(prompts, responses)):
        ids[i, : len(p) + len(r)] = torch.tensor(p + r)
        attn[i, : len(p) + len(r)] = 1
        resp[i, len(p) : len(p) + len(r)] = 1.0
    return ids.to(device), attn.to(device), resp.to(device)


def lm_token_logps(model, ids, attn, autocast):
    """Log-probability of each token given the ones before: `(B, T - 1)`."""
    with autocast():
        logits = model(input_ids=ids, attention_mask=attn, use_cache=False).logits
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    return logp.gather(-1, ids[:, 1:, None]).squeeze(-1)


def seq_logps(model, prompts, responses, pad_id, device, autocast, micro=8):
    """Summed response log-probabilities for lists of prompt and response ids: `(N,)`."""
    out = []
    for lo in range(0, len(prompts), micro):
        ids, attn, resp = collate(prompts[lo : lo + micro], responses[lo : lo + micro], pad_id, device)
        out.append((lm_token_logps(model, ids, attn, autocast) * resp[:, 1:]).sum(dim=1))
    return torch.cat(out)


def prompt_text(tok, a, b):
    return tok.apply_chat_template(
        [{"role": "user", "content": f"What is {a} * {b}? Work it out briefly, "
          "then end with 'Answer: ' followed by the number."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def demonstration(a, b):
    tens, ones = divmod(b, 10)
    return (f"{a} * {b} = {a} * {tens * 10} + {a} * {ones} = {a * tens * 10} + {a * ones} "
            f"= {a * b}. Answer: {a * b}")


def sample_group(model, tok, questions, n, max_new_tokens, device, autocast, eos_ids, chunk=128):
    """Sample n responses per question. Returns (prompt_ids, response_ids, question index) lists."""
    texts = [prompt_text(tok, a, b) for a, b in questions]
    prompt_ids = [tok(t).input_ids for t in texts]
    jobs = [(qi, k) for qi in range(len(questions)) for k in range(n)]
    out_p, out_r, out_q = [], [], []
    model.eval()
    for lo in range(0, len(jobs), chunk):
        part = jobs[lo : lo + chunk]
        width = max(len(prompt_ids[qi]) for qi, _ in part)
        ids = torch.full((len(part), width), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros(len(part), width, dtype=torch.long)
        for i, (qi, _) in enumerate(part):          # left-pad, as generation needs
            p = prompt_ids[qi]
            ids[i, width - len(p) :] = torch.tensor(p)
            attn[i, width - len(p) :] = 1
        with torch.no_grad(), autocast():
            gen = model.generate(input_ids=ids.to(device), attention_mask=attn.to(device),
                                 max_new_tokens=max_new_tokens, do_sample=True, temperature=1.0,
                                 top_p=1.0, top_k=0, pad_token_id=tok.pad_token_id)
        for i, (qi, _) in enumerate(part):
            resp = gen[i, width:].tolist()
            for j, t in enumerate(resp):            # keep up to and including the first end token
                if t in eos_ids:
                    resp = resp[: j + 1]
                    break
            out_p.append(prompt_ids[qi])
            out_r.append(resp)
            out_q.append(qi)
    return out_p, out_r, out_q


def evaluate(model, tok, questions, n, max_new_tokens, device, autocast, eos_ids):
    """Per-question correct counts, mean response length, and the samples themselves."""
    p, r, q = sample_group(model, tok, questions, n, max_new_tokens, device, autocast, eos_ids)
    counts = [0] * len(questions)
    for resp, qi in zip(r, q):
        a, b = questions[qi]
        counts[qi] += int(multiplication_reward(tok.decode(resp, skip_special_tokens=True), a * b))
    return counts, sum(len(x) for x in r) / len(r), (p, r, q)


def post_train(model, tok, device="cuda", r=16, alpha=32.0, sft_steps=60, sft_batch=16,
               dpo_steps=30, dpo_batch=8, pair_questions=48, eval_questions=48, n_samples=8,
               max_new_tokens=80, lr_sft=2e-4, lr_dpo=3e-5, beta=0.1, seed=0, time_limit=540.0):
    """SFT then DPO with a LoRA adapter on two-digit multiplication, then the written evaluation."""
    from contextlib import nullcontext

    started = time.time()
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    if device.startswith("cuda"):
        def autocast():
            return torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast = nullcontext
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    eos_id = tok.convert_tokens_to_ids("<|im_end|>")
    pad_id = tok.pad_token_id

    def questions(n):
        a = torch.randint(10, 100, (n,), generator=rng).tolist()
        b = torch.randint(10, 100, (n,), generator=rng).tolist()
        return list(zip(a, b))

    shapes = attach_lora(model, r, alpha, seed)
    n_lora = lora_param_count(shapes, r)
    n_base = sum(p.numel() for n_, p in model.named_parameters() if not p.requires_grad)
    print(f"LoRA rank {r}, alpha {alpha}: {n_lora:,} trainable parameters, "
          f"{100 * n_lora / n_base:.2f}% of the {n_base:,} frozen ones")
    params = adapter_params(model)
    held_out = questions(eval_questions)
    report = {}

    def stage_report(name, ref_state=None):
        counts, length, samples = evaluate(model, tok, held_out, n_samples, max_new_tokens,
                                           device, autocast, eos_ids)
        p1 = sum(pass_at_k(n_samples, c, 1) for c in counts) / len(counts)
        p4 = sum(pass_at_k(n_samples, c, 4) for c in counts) / len(counts)
        kl = float("nan")
        if ref_state is not None:   # mean log-ratio on the policy's own samples: the k1 estimate
            mine = seq_logps(model, samples[0], samples[1], pad_id, device, autocast)
            now = adapter_state(model)
            load_adapter(model, ref_state)
            ref = seq_logps(model, samples[0], samples[1], pad_id, device, autocast)
            load_adapter(model, now)
            kl = float((mine - ref).mean() / max(length, 1.0))
        report[name] = counts
        print(f"{name:<6} {p1:>7.3f} {p4:>7.3f} {length:>8.1f} {kl:>14.4f} {time.time() - started:>6.0f}")

    print(f"{'stage':<6} {'pass@1':>7} {'pass@4':>7} {'mean len':>8} {'KL/token vs SFT':>14} {'secs':>6}")
    with torch.no_grad():
        stage_report("base")        # B = 0, so this is the base model exactly

    # Stage 1: SFT on worked solutions, with the loss on the response only.
    opt = torch.optim.AdamW(params, lr=lr_sft, weight_decay=0.0)
    model.train()
    for step in range(sft_steps):
        qs = questions(sft_batch)
        ps = [tok(prompt_text(tok, a, b)).input_ids for a, b in qs]
        rs = [tok(demonstration(a, b)).input_ids + [eos_id] for a, b in qs]
        ids, attn, resp = collate(ps, rs, pad_id, device)
        loss = sft_loss(lm_token_logps(model, ids, attn, autocast), resp[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 20 == 0 or step == sft_steps - 1:
            print(f"  sft step {step:>3} loss {float(loss.detach()):.4f}")
    sft_state = adapter_state(model)
    with torch.no_grad():
        stage_report("sft")

    # Stage 2: DPO on pairs sampled from the SFT policy and scored by the reward rule.
    pq = questions(pair_questions)
    p, resp, q = sample_group(model, tok, pq, n_samples, max_new_tokens, device, autocast, eos_ids)
    good = [multiplication_reward(tok.decode(x, skip_special_tokens=True), pq[i][0] * pq[i][1])
            for x, i in zip(resp, q)]
    pairs = []
    for qi in range(len(pq)):
        rows = [k for k in range(len(q)) if q[k] == qi]
        right = [k for k in rows if good[k] > 0.5]
        wrong = [k for k in rows if good[k] < 0.5]
        if right and wrong:
            pairs.append((p[right[0]], resp[right[0]], resp[wrong[0]]))
    print(f"  {len(pairs)} preference pairs from {len(pq)} questions")
    if pairs:
        pp = [x[0] for x in pairs]
        pc = [x[1] for x in pairs]
        pr = [x[2] for x in pairs]
        with torch.no_grad():
            ref_c = seq_logps(model, pp, pc, pad_id, device, autocast)
            ref_r = seq_logps(model, pp, pr, pad_id, device, autocast)
        opt = torch.optim.AdamW(params, lr=lr_dpo, weight_decay=0.0)
        model.train()
        for step in range(dpo_steps):
            idx = torch.randperm(len(pairs), generator=rng)[:dpo_batch].tolist()
            lc = seq_logps(model, [pp[i] for i in idx], [pc[i] for i in idx], pad_id, device, autocast, micro=len(idx))
            lr_ = seq_logps(model, [pp[i] for i in idx], [pr[i] for i in idx], pad_id, device, autocast, micro=len(idx))
            loss = dpo_loss(lc, lr_, ref_c[idx], ref_r[idx], beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if step % 10 == 0 or step == dpo_steps - 1:
                margin = float(((lc - ref_c[idx]) - (lr_ - ref_r[idx])).mean().detach())
                print(f"  dpo step {step:>3} loss {float(loss.detach()):.4f} margin {margin:+.3f}")
            if time.time() - started > time_limit:
                print(f"stopping DPO at step {step}: the {time_limit:.0f} s budget is spent")
                break
    with torch.no_grad():
        stage_report("dpo", ref_state=sft_state)

    # The written evaluation: a head-to-head win rate, DPO against SFT, by correct counts.
    sft_c, dpo_c = report["sft"], report["dpo"]
    ab = [(d > s) - (d < s) for d, s in zip(dpo_c, sft_c)]        # DPO shown first
    ba = [-v for v in ab]                                          # SFT shown first; a rule judge is consistent
    win, consistency = judge_win_rate(ab, ba)
    print(f"win rate of DPO over SFT, by correct count: {win:.3f} "
          f"(rule judge, swap consistency {consistency:.2f})")


def real_run(model_id="Qwen/Qwen3-0.6B", device="cuda", **kwargs):
    """Load Qwen3-0.6B in bfloat16, freeze it, and post-train a LoRA adapter on it."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    print(f"SFT then DPO with LoRA on {model_id}")
    post_train(model, tok, device=device, **kwargs)


def toy_run():
    """The CPU fallback: the toy pipeline, then the evaluation functions on its numbers."""
    print(f"{'seed':>4} {'SFT nll':>16} {'DPO margin':>18} {'KL to SFT':>10} {'reward':>14}")
    for seed in range(3):
        out = run_toy_pipeline(seed)
        print(f"{seed:>4} {out['sft_nll_before']:>7.3f} -> {out['sft_nll_after']:<6.3f} "
              f"{out['margin_before']:>7.3f} -> {out['margin_after']:<7.3f} "
              f"{out['kl_to_sft']:>10.3f} {out['reward_before']:>6.3f} -> {out['reward_after']:<6.3f}")
    print(f"pass@4 with 8 samples, 2 correct: {pass_at_k(8, 2, 4):.3f}")


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
