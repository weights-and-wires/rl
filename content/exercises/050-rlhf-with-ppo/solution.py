"""RLHF with PPO: KL-shaped rewards, KL estimators, GAE on tokens, and PPO.

This is the reference solution. It has the same public surface as the starter.

Every function you write works on plain torch tensors in the shapes used through
Part 9: per-token log-probabilities, values, rewards, and response masks are
`(B, T)`, and sequence rewards are `(B,)`. The same functions train the toy
policy in the checks and Qwen3-0.6B on the GPU worker.
"""

import math
import time

import torch

torch.set_num_threads(1)


# Your functions -------------------------------------------------------------


def kl_shaped_rewards(logp, ref_logp, seq_rewards, mask, beta):
    """Per-token rewards with the KL penalty spread over tokens.

    Every response token pays `-beta * (logp - ref_logp)`, the k1 estimate of
    its KL to the reference. The last response token also receives the
    sequence's task reward.

    Args:
        logp: log-probs of the sampled tokens under the policy that sampled them,
            `(B, T)`.
        ref_logp: log-probs of the same tokens under the reference, `(B, T)`.
        seq_rewards: task reward of each response, `(B,)`.
        mask: response mask, `(B, T)`, 1 for response tokens. Each row is a run
            of ones followed by zeros, and has at least one 1.
        beta: KL coefficient.

    Returns:
        `(B, T)` rewards, exactly 0 at padded positions.
    """
    mask = mask.to(logp.dtype)
    rewards = -beta * (logp - ref_logp) * mask
    last = mask.sum(dim=1).long() - 1
    rewards[torch.arange(len(last), device=last.device), last] += seq_rewards.to(logp.dtype)
    return rewards


def kl_estimators(logp, ref_logp):
    """The three single-sample KL estimators from Schulman's note.

    For a token sampled from the policy q, with ratio r = p / q where p is the
    reference, the estimators of KL(q || p) are k1 = -log r,
    k2 = (log r) ** 2 / 2, and k3 = (r - 1) - log r.

    Args:
        logp: `log q` of sampled tokens, any shape.
        ref_logp: `log p` of the same tokens, same shape.

    Returns:
        A tuple `(k1, k2, k3)` of tensors, each the shape of `logp`.
    """
    log_r = ref_logp - logp
    return -log_r, 0.5 * log_r**2, torch.expm1(log_r) - log_r


def gae(rewards, values, mask, gamma=1.0, lam=0.95):
    """Generalised advantage estimates over the tokens of each response.

    `values[:, t]` is V(s_t), the value of the prefix before token t. The
    response ends after its last masked token, so the value after that token
    is 0: there is no bootstrap at the end.

    Args:
        rewards: per-token rewards, `(B, T)`.
        values: per-token values, `(B, T)`.
        mask: response mask, `(B, T)`, a run of ones then zeros.
        gamma: discount.
        lam: GAE lambda.

    Returns:
        A tuple `(advantages, returns)`, each `(B, T)`, 0 at padded positions.
        `returns = advantages + values` on response tokens.
    """
    mask = mask.to(values.dtype)
    steps = rewards.shape[1]
    # Pad one column so position t + 1 exists at the end: value 0, not continuing.
    next_values = torch.cat([values[:, 1:], torch.zeros_like(values[:, :1])], dim=1)
    next_mask = torch.cat([mask[:, 1:], torch.zeros_like(mask[:, :1])], dim=1)
    adv = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[:, 0])
    for t in reversed(range(steps)):
        delta = rewards[:, t] + gamma * next_values[:, t] * next_mask[:, t] - values[:, t]
        running = (delta + gamma * lam * next_mask[:, t] * running) * mask[:, t]
        adv[:, t] = running
    return adv, (adv + values) * mask


def masked_whiten(x, mask, eps=1e-8):
    """Shift and scale `x` to mean 0 and variance 1 over its masked entries.

    Args:
        x: `(B, T)` values, such as advantages.
        mask: `(B, T)`, 1 for entries that count.
        eps: added to the variance before the square root.

    Returns:
        `(B, T)`, 0 at unmasked positions. The mean and the variance (over N,
        not N - 1) use masked entries only.
    """
    mask = mask.to(x.dtype)
    n = mask.sum()
    mean = (x * mask).sum() / n
    var = (((x - mean) * mask) ** 2).sum() / n
    return (x - mean) / torch.sqrt(var + eps) * mask


def ppo_policy_loss(logp, old_logp, advantages, mask, clip_eps=0.2):
    """The PPO clipped surrogate, negated for minimisation, per-token mean.

    Args:
        logp: log-probs under the policy being trained, `(B, T)`, with gradient.
        old_logp: log-probs under the policy that sampled the batch, `(B, T)`.
        advantages: `(B, T)`, treated as constants.
        mask: response mask, `(B, T)`.
        clip_eps: the clip range epsilon.

    Returns:
        A scalar: minus the mean over every response token in the batch of
        min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A).
    """
    mask = mask.to(logp.dtype)
    ratio = torch.exp(logp - old_logp)
    surrogate = torch.minimum(ratio * advantages,
                              torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages)
    return -(surrogate * mask).sum() / mask.sum()


def value_loss(values, old_values, returns, mask, clip_eps=0.2):
    """Clipped value loss: half the per-token mean of the larger squared error.

    Args:
        values: the value head's predictions, `(B, T)`, with gradient.
        old_values: predictions when the batch was collected, `(B, T)`.
        returns: value targets, `(B, T)`.
        mask: response mask, `(B, T)`.
        clip_eps: how far a prediction may move from `old_values` before its
            clipped error takes over.

    Returns:
        A scalar, 0.5 * mean over response tokens of
        max((V - R) ** 2, (clip(V, V_old - eps, V_old + eps) - R) ** 2).
    """
    mask = mask.to(values.dtype)
    clipped = old_values + torch.clamp(values - old_values, -clip_eps, clip_eps)
    err = torch.maximum((values - returns) ** 2, (clipped - returns) ** 2)
    return 0.5 * (err * mask).sum() / mask.sum()


def ppo_update(forward, optimizer, batch, epochs=4, minibatch_size=64, clip_eps=0.2,
               vf_coef=0.5, max_grad_norm=1.0, generator=None):
    """Run PPO's inner loop: several epochs of minibatch steps on one batch.

    Args:
        forward: a function taking a 1-D tensor of row indices and returning
            `(logp, values)` for those rows, each `(len(idx), T)`, with gradient.
        optimizer: a torch optimiser over every trainable parameter.
        batch: a dict of `(B, T)` tensors with keys `"old_logp"`, `"old_values"`,
            `"advantages"`, `"returns"`, and `"mask"`.
        epochs: passes over the batch.
        minibatch_size: rows per gradient step. Shuffle the rows each epoch.
        clip_eps: clip range for both losses.
        vf_coef: weight of the value loss in the total loss.
        max_grad_norm: clip the global gradient norm of every parameter the
            optimiser holds to this before each step.
        generator: optional `torch.Generator` for the shuffle.

    Returns:
        A dict of floats averaged over minibatch steps: `"policy_loss"`,
        `"value_loss"`, and `"clip_frac"`, the fraction of response tokens whose
        ratio lies outside `[1 - eps, 1 + eps]`.
    """
    rows = batch["mask"].shape[0]
    params = [p for group in optimizer.param_groups for p in group["params"]]
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "clip_frac": 0.0}
    count = 0
    for _ in range(epochs):
        order = torch.randperm(rows, generator=generator)
        for start in range(0, rows, minibatch_size):
            idx = order[start:start + minibatch_size]
            mask = batch["mask"][idx]
            logp, values = forward(idx)
            pl = ppo_policy_loss(logp, batch["old_logp"][idx], batch["advantages"][idx], mask, clip_eps)
            vl = value_loss(values, batch["old_values"][idx], batch["returns"][idx], mask, clip_eps)
            optimizer.zero_grad()
            (pl + vf_coef * vl).backward()
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                ratio = torch.exp(logp - batch["old_logp"][idx])
                outside = ((ratio - 1).abs() > clip_eps).to(ratio.dtype) * mask
                stats["clip_frac"] += float(outside.sum() / mask.sum())
            stats["policy_loss"] += float(pl.detach())
            stats["value_loss"] += float(vl.detach())
            count += 1
    return {k: v / count for k, v in stats.items()}


# The toy language model -------------------------------------------------------


class ToyPolicy(torch.nn.Module):
    """A tabular policy with a value head, over `vocab` tokens and `length` positions.

    The logits and the value both depend on the position `t` and the previous
    token (the start symbol, index `vocab`, at `t = 0`), like `ToyLM` in the
    previous lessons. `ToyPolicy(seed=s).logits` equals `ToyLM(seed=s).table`.
    """

    def __init__(self, vocab=8, length=6, seed=0, scale=1.0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        table = scale * torch.randn(length, vocab + 1, vocab, generator=gen, dtype=torch.float64)
        self.vocab, self.length = vocab, length
        self.logits = torch.nn.Parameter(table)
        self.values = torch.nn.Parameter(torch.zeros(length, vocab + 1, dtype=torch.float64))

    def forward(self, tokens):
        """Log-probs of `tokens` `(B, T)` and the values of their prefixes `(B, T)`."""
        batch, steps = tokens.shape
        start = torch.full((batch, 1), self.vocab, dtype=torch.long)
        prev = torch.cat([start, tokens[:, :-1]], dim=1)
        pos = torch.arange(steps).expand(batch, steps)
        logp_all = torch.log_softmax(self.logits[pos, prev], dim=-1)
        logp = logp_all.gather(-1, tokens[..., None])[..., 0]
        return logp, self.values[pos, prev]

    @torch.no_grad()
    def sample(self, n, generator):
        """Draw `n` responses, shape `(n, length)`, token by token."""
        tokens = torch.zeros(n, self.length, dtype=torch.long)
        prev = torch.full((n,), self.vocab, dtype=torch.long)
        for t in range(self.length):
            probs = torch.softmax(self.logits[t, prev], dim=-1)
            tokens[:, t] = torch.multinomial(probs, 1, generator=generator)[:, 0]
            prev = tokens[:, t]
        return tokens


def rising_reward(tokens):
    """The toy task reward: the fraction of adjacent pairs where the token goes up."""
    return (tokens[:, 1:] > tokens[:, :-1]).to(torch.float64).mean(dim=1)


def train_toy_ppo(iterations=40, batch_size=256, beta=0.05, lr=0.05, seed=0, verbose=False):
    """PPO with a KL penalty on the toy policy, built from your functions.

    Returns a list of per-iteration dicts with the mean task reward and the
    mean sequence KL to the reference (the k1 sum over tokens).
    """
    gen = torch.Generator().manual_seed(seed)
    ref = ToyPolicy(seed=seed)
    policy = ToyPolicy(seed=seed)
    for p in ref.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    history = []
    for it in range(iterations):
        tokens = policy.sample(batch_size, gen)
        mask = torch.ones(tokens.shape, dtype=torch.float64)
        with torch.no_grad():
            old_logp, old_values = policy(tokens)
            ref_logp, _ = ref(tokens)
        task = rising_reward(tokens)
        rewards = kl_shaped_rewards(old_logp, ref_logp, task, mask, beta)
        advantages, returns = gae(rewards, old_values, mask, gamma=1.0, lam=0.95)
        batch = {
            "old_logp": old_logp,
            "old_values": old_values,
            "advantages": masked_whiten(advantages, mask),
            "returns": returns,
            "mask": mask,
        }
        stats = ppo_update(lambda idx: policy(tokens[idx]), optimizer, batch, epochs=4,
                           minibatch_size=64, generator=gen)
        kl = float((old_logp - ref_logp).sum(dim=1).mean())
        history.append({"reward": float(task.mean()), "kl": kl, **stats})
        if verbose and (it % 5 == 0 or it == iterations - 1):
            print(f"{it:4d}  reward {task.mean():.3f}  KL {kl:6.3f}  "
                  f"clip frac {stats['clip_frac']:.3f}")
    return history


# The real run: PPO on Qwen3-0.6B -------------------------------------------------

TOPICS = [
    "the ocean", "a cat", "winter", "coffee", "a library", "the moon", "a bicycle",
    "rain", "a mountain", "bread", "a train station", "music", "a garden", "the desert",
    "a thunderstorm", "chess", "a lighthouse", "autumn leaves", "a volcano", "a city at night",
    "honey bees", "a violin", "the Sahara", "a snowman", "a river", "friendship", "a rainbow",
    "the internet", "a castle", "tea", "a forest", "the sun", "a spider web", "a market",
    "a clock", "a whale", "a campfire", "a newspaper", "a bridge", "the wind",
]
TARGET_WORDS = 10


def word_count_reward(text):
    """The rule-based task reward: 1 for exactly ten words, falling by 0.1 per word off."""
    return 1.0 - min(abs(len(text.split()) - TARGET_WORDS), TARGET_WORDS) / TARGET_WORDS


def real_run(model_id="Qwen/Qwen3-0.6B", device="cuda", steps=60, batch_size=32,
             minibatch_size=8, epochs=2, max_new_tokens=48, lr=1e-6, value_lr=1e-3,
             beta=0.05, log_every=5, seed=0):
    """PPO with a value head and a per-token KL penalty on a real language model.

    The task is a rule: describe a topic in exactly ten words. The policy is
    fully fine-tuned, with float32 master weights and bfloat16 autocast on CUDA.
    The reference is a frozen bfloat16 copy. The value head is one linear layer
    on the policy's last hidden state, detached, so the value loss trains only
    the head and never the language model.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    started = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    policy = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device).float()
    ref = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    policy.config.use_cache = False
    value_head = torch.nn.Linear(policy.config.hidden_size, 1).to(device)
    torch.nn.init.zeros_(value_head.weight)
    torch.nn.init.zeros_(value_head.bias)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in policy.parameters() if p.requires_grad], "lr": lr, "weight_decay": 0.0},
        {"params": value_head.parameters(), "lr": value_lr, "weight_decay": 0.0},
    ])
    stop = policy.generation_config.eos_token_id
    stop = set(stop if isinstance(stop, list) else [stop]) | {tok.eos_token_id}
    eos_ids = torch.tensor(sorted(stop), device=device)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast(
        "cpu", enabled=False)
    print(f"loaded {model_id} on {device} in {time.time() - started:.0f}s")
    print(" step  reward  exact-10  words    KL   clip  seconds")

    def score(model, seq, attn, prompt_len, with_values):
        """Log-probs of the response tokens (and values of their prefixes)."""
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        with autocast:
            out = model(input_ids=seq, attention_mask=attn, position_ids=pos,
                        output_hidden_states=with_values)
        logits = out.logits[:, prompt_len - 1:-1].float()
        logp = torch.log_softmax(logits, -1).gather(-1, seq[:, prompt_len:, None])[..., 0]
        if not with_values:
            return logp, None
        hidden = out.hidden_states[-1][:, prompt_len - 1:-1].detach().float()
        return logp, value_head(hidden)[..., 0]

    for step in range(steps):
        topics = [TOPICS[i] for i in torch.randint(len(TOPICS), (batch_size,), generator=gen)]
        chats = [tok.apply_chat_template(
            [{"role": "user", "content": f"Describe {t} in exactly ten words."}],
            add_generation_prompt=True, enable_thinking=False, tokenize=False) for t in topics]
        enc = tok(chats, return_tensors="pt", padding=True).to(device)
        prompt_len = enc.input_ids.shape[1]
        policy.eval()
        with torch.no_grad(), autocast:
            seq = policy.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                  max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                                  use_cache=True)
        response = seq[:, prompt_len:]
        is_eos = torch.isin(response, eos_ids)
        # A token is in the response if no end-of-sequence token came before it.
        mask = (torch.cumsum(is_eos, 1) - is_eos.long() == 0).float()
        finished = is_eos.any(dim=1)
        attn = torch.cat([enc.attention_mask, mask.long()], dim=1)
        texts = tok.batch_decode(response * mask.long() + tok.pad_token_id * (1 - mask.long()),
                                 skip_special_tokens=True)
        task = torch.tensor([word_count_reward(t) if f else 0.0 for t, f in zip(texts, finished)],
                            device=device)

        with torch.no_grad():
            old_logp, old_values = score(policy, seq, attn, prompt_len, True)
            ref_logp, _ = score(ref, seq, attn, prompt_len, False)
        rewards = kl_shaped_rewards(old_logp, ref_logp, task, mask, beta)
        advantages, returns = gae(rewards, old_values, mask, gamma=1.0, lam=0.95)
        batch = {"old_logp": old_logp, "old_values": old_values,
                 "advantages": masked_whiten(advantages, mask), "returns": returns, "mask": mask}

        policy.train()
        stats = ppo_update(lambda idx: score(policy, seq[idx], attn[idx], prompt_len, True),
                           optimizer, batch, epochs=epochs, minibatch_size=minibatch_size,
                           clip_eps=0.2, vf_coef=0.5, max_grad_norm=1.0, generator=gen)
        if step % log_every == 0 or step == steps - 1:
            k1, _, _ = kl_estimators(old_logp, ref_logp)
            kl = float((k1 * mask).sum(1).mean())
            words = sum(len(t.split()) for t in texts) / len(texts)
            exact = sum(len(t.split()) == TARGET_WORDS for t in texts) / len(texts)
            print(f"{step:5d}  {float(task.mean()):6.3f}  {exact:8.2f}  {words:5.1f}  {kl:5.3f}  "
                  f"{stats['clip_frac']:5.3f}  {time.time() - started:7.0f}")
    print("\nsamples from the last batch:")
    for t, text in zip(topics[:4], texts[:4]):
        print(f"  {t}: {text.strip()!r} ({len(text.split())} words)")


if __name__ == "__main__":
    try:
        import transformers  # noqa: F401
        have_transformers = True
    except ImportError:
        have_transformers = False
    if torch.cuda.is_available() and have_transformers:
        real_run()
    else:
        print("The Qwen3-0.6B run needs the GPU worker. Running PPO on the toy policy instead.")
        print("iter  reward  (the fraction of adjacent tokens that go up)")
        train_toy_ppo(verbose=True)
