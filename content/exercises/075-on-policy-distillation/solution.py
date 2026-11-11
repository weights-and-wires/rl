"""On-policy distillation.

Reference solution.

A student language model learns from a teacher by sampling its own tokens and
asking the teacher how likely each one is. The pieces: the two token-level KL
directions, the per-token reward log pi_T - log pi_S, the loss whose gradient
follows that reward, a fit that shows what each KL direction picks when the
student is too simple, and a training loop.

Shapes follow the course's toy-harness convention:

- next-token logits are `(B, T, V)`,
- token ids and masks are `(B, T)`; a mask is 1 on real tokens and 0 on padding.
"""

import numpy as np
import torch

torch.set_num_threads(1)


# The toy sequence task. The same code sits in solution.py. -----------------

SEQ_LEN = 3                  # tokens per sequence
VOCAB = 4                    # token ids 0-3
START = VOCAB                # a fifth "previous token" that opens every sequence


def make_teacher(seed):
    """A fixed teacher: a table of logits of shape (SEQ_LEN, VOCAB + 1, VOCAB).

    Row [t, p] holds the logits for the token at position t given that the
    previous token was p (p = START at t = 0).
    """
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.normal(0.0, 1.5, size=(SEQ_LEN, VOCAB + 1, VOCAB)))


def logits_for(table, tokens):
    """Next-token logits (N, SEQ_LEN, VOCAB) of the table model for sequences (N, SEQ_LEN)."""
    n = tokens.shape[0]
    prev = torch.cat([torch.full((n, 1), START, dtype=torch.long), tokens[:, :-1]], dim=1)
    return table[torch.arange(SEQ_LEN).unsqueeze(0), prev]


def sample_sequences(table, n, generator):
    """Draw n sequences (n, SEQ_LEN) from the table model, one token at a time."""
    tokens = torch.zeros(n, SEQ_LEN, dtype=torch.long)
    prev = torch.full((n,), START, dtype=torch.long)
    for t in range(SEQ_LEN):
        probs = torch.softmax(table[t, prev], dim=-1)
        tokens[:, t] = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        prev = tokens[:, t]
    return tokens


def all_sequences():
    """Every one of the VOCAB ** SEQ_LEN sequences, shape (64, SEQ_LEN)."""
    grids = torch.meshgrid(*[torch.arange(VOCAB)] * SEQ_LEN, indexing="ij")
    return torch.stack([g.reshape(-1) for g in grids], dim=1)


def sequence_logprobs(table, tokens):
    """Log-probability of each whole sequence, shape (N,)."""
    logp = torch.log_softmax(logits_for(table, tokens), dim=-1)
    return logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1).sum(dim=1)


def exact_sequence_kl(student, teacher):
    """The exact reverse KL(student || teacher) over whole sequences, by enumeration."""
    seqs = all_sequences()
    lp_s = sequence_logprobs(student, seqs)
    lp_t = sequence_logprobs(teacher, seqs)
    return float((lp_s.exp() * (lp_s - lp_t)).sum())


# The exercise --------------------------------------------------------------


def token_kl_losses(student_logits, teacher_logits, mask):
    logp_s = torch.log_softmax(student_logits, dim=-1)
    logp_t = torch.log_softmax(teacher_logits, dim=-1)
    forward = (logp_t.exp() * (logp_t - logp_s)).sum(dim=-1)   # (B, T)
    reverse = (logp_s.exp() * (logp_s - logp_t)).sum(dim=-1)
    n = mask.sum()
    return (forward * mask).sum() / n, (reverse * mask).sum() / n


def sampled_reverse_kl(student_logits, teacher_logits, tokens):
    logp_s = torch.log_softmax(student_logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    logp_t = torch.log_softmax(teacher_logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    return logp_s - logp_t


def distill_reward(student_logits, teacher_logits, tokens):
    return -sampled_reverse_kl(student_logits, teacher_logits, tokens)


def on_policy_distill_loss(student_logits, teacher_logits, tokens, mask):
    reward = distill_reward(student_logits, teacher_logits, tokens).detach()
    logp = torch.log_softmax(student_logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    return -(mask * reward * logp).sum() / mask.sum()


def fit_unimodal_student(teacher_logits, grid, direction):
    t = torch.as_tensor(teacher_logits, dtype=torch.float64).reshape(1, 1, -1)
    g = np.asarray(grid, dtype=float)
    ones = torch.ones(1, 1, dtype=torch.float64)
    best, best_loss = None, np.inf
    for mu in np.linspace(-5, 5, 101):
        for sigma in np.linspace(0.3, 6, 58):
            s = torch.tensor(-((g - mu) ** 2) / (2 * sigma**2)).reshape(1, 1, -1)
            fwd, rev = token_kl_losses(s, t, ones)
            loss = float(fwd if direction == "forward" else rev)
            if loss < best_loss:
                best, best_loss = (float(mu), float(sigma)), loss
    return best


def train_on_policy_distill(teacher, seed, steps=300, batch=128, lr=0.05):
    gen = torch.Generator().manual_seed(seed)
    table = torch.zeros_like(teacher, requires_grad=True)
    opt = torch.optim.Adam([table], lr=lr)
    mask = torch.ones(batch, SEQ_LEN, dtype=teacher.dtype)
    for _ in range(steps):
        with torch.no_grad():
            tokens = sample_sequences(table, batch, gen)
        loss = on_policy_distill_loss(
            logits_for(table, tokens), logits_for(teacher, tokens), tokens, mask
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    return table.detach()


if __name__ == "__main__":
    teacher = make_teacher(0)
    student = torch.zeros_like(teacher)
    print(f"reverse KL of the untrained student: {exact_sequence_kl(student, teacher):.3f}")
    trained = train_on_policy_distill(teacher, seed=0)
    print(f"reverse KL after on-policy distillation: {exact_sequence_kl(trained, teacher):.3f}")
