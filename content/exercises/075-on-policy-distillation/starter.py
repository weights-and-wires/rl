"""On-policy distillation.

Fill in the functions under "The exercise". The toy sequence task above them
is given.

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
    """Forward and reverse KL at every real token, averaged over real tokens.

    student_logits, teacher_logits: (B, T, V) float tensors.
    mask: (B, T), 1.0 on real tokens and 0.0 on padding.

    Returns (forward, reverse), two scalar tensors:
      forward = mean over real tokens of KL(pi_T || pi_S)
      reverse = mean over real tokens of KL(pi_S || pi_T)
    Divide by mask.sum(), not by B * T. Padded positions can hold any logits.
    """
    raise NotImplementedError("compute both KLs per token, then take the masked mean")


def sampled_reverse_kl(student_logits, teacher_logits, tokens):
    """One-sample estimate of KL(pi_S || pi_T) at each position.

    student_logits, teacher_logits: (N, V). tokens: (N,) ints drawn from the
    student's own distribution at those positions.
    Returns (N,): log pi_S(token) - log pi_T(token).
    """
    raise NotImplementedError("gather the two log-probabilities at the sampled tokens")


def distill_reward(student_logits, teacher_logits, tokens):
    """The per-token reward of on-policy distillation, shape (N,): log pi_T - log pi_S."""
    raise NotImplementedError("it is the negative of the sampled reverse-KL estimate")


def on_policy_distill_loss(student_logits, teacher_logits, tokens, mask):
    """A score-function surrogate whose gradient follows the distillation reward.

    student_logits, teacher_logits: (B, T, V). tokens: (B, T) sampled from the
    student. mask: (B, T).
    Returns a scalar: -(1 / mask.sum()) * sum over real tokens of
    reward * log pi_S(token), where the reward is treated as a constant
    (no gradient flows through it).
    """
    raise NotImplementedError("weight each token's log-probability by its detached reward")


def fit_unimodal_student(teacher_logits, grid, direction):
    """Best single-mode (discretized Gaussian) student for a teacher over a 1-D grid.

    teacher_logits: (V,) logits over the grid points. grid: (V,) positions.
    direction: "forward" minimizes KL(pi_T || pi_S); "reverse" minimizes KL(pi_S || pi_T).
    Search mu over np.linspace(-5, 5, 101) and sigma over np.linspace(0.3, 6, 58);
    a student has logits -(grid - mu)**2 / (2 * sigma**2).
    Returns (mu, sigma) of the best pair. Reuse token_kl_losses.
    """
    raise NotImplementedError("score every (mu, sigma) with the right KL direction")


def train_on_policy_distill(teacher, seed, steps=300, batch=128, lr=0.05):
    """Train a table student (zeros at the start) toward the teacher.

    Each step: sample `batch` sequences from the student with no gradient, get
    student logits with gradient and teacher logits, call on_policy_distill_loss
    with an all-ones mask, and take an Adam step. Seed a torch.Generator for the
    sampling. Returns the trained student table.
    """
    raise NotImplementedError("sample, score, take a gradient step, repeat")


if __name__ == "__main__":
    teacher = make_teacher(0)
    student = torch.zeros_like(teacher)
    print(f"reverse KL of the untrained student: {exact_sequence_kl(student, teacher):.3f}")
    trained = train_on_policy_distill(teacher, seed=0)
    print(f"reverse KL after on-policy distillation: {exact_sequence_kl(trained, teacher):.3f}")
