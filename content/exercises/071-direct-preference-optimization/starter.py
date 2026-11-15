"""Direct preference optimization, IPO, and likelihood displacement.

Fill in the functions under "The exercise". The helpers above them are given.

Every loss here works on log-probabilities, never on text. A response's
log-probability under a policy is the sum of its per-token log-probabilities,
so the same functions serve a toy table and a real language model.

Shapes follow the course's toy-harness convention:

- per-token log-probabilities and response masks are `(B, T)`,
- sequence log-probabilities, rewards, and labels are `(B,)`.

"chosen" is the preferred response `y_w` and "rejected" is `y_l`.
"""

import numpy as np


# Helpers given to you. The same code sits in solution.py. ----------------


def sigmoid(z):
    """Logistic sigmoid, elementwise, without overflow for large |z|."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=float)))


def log_sigmoid(z):
    """log(sigmoid(z)), elementwise, computed as -log(1 + exp(-z)) stably."""
    return -np.logaddexp(0.0, -np.asarray(z, dtype=float))


def log_softmax(z, axis=-1):
    """Log-softmax along `axis`, shifted by the max for stability."""
    z = np.asarray(z, dtype=float)
    shifted = z - z.max(axis=axis, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def pair_weights(n, weights):
    """Per-pair weights for a batch of n pairs: 1/n each when `weights` is None."""
    if weights is None:
        return np.full(n, 1.0 / n)
    return np.asarray(weights, dtype=float)


def similar_pair_embeddings(n, dim, cosine, seed):
    """Unit embeddings for n responses. Rows 0 and 1 have the given cosine similarity."""
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(n, dim))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    a = rng.normal(size=dim)
    a /= np.linalg.norm(a)
    b = rng.normal(size=dim)
    b -= (b @ a) * a
    b /= np.linalg.norm(b)
    E[0] = a
    E[1] = cosine * a + np.sqrt(1.0 - cosine**2) * b
    return E


# The exercise. ------------------------------------------------------------


def sequence_logprob(token_logps, mask):
    """Sum each response's per-token log-probabilities over its mask.

    Args:
        token_logps: log pi(y_t | x, y_<t) for every position, shape (B, T).
            Padding positions can hold anything, including -inf.
        mask: 1 (or True) on response tokens and 0 on padding, shape (B, T).

    Returns:
        log pi(y | x) for each response, shape (B,).
    """
    raise NotImplementedError("sequence_logprob")


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta, weights=None):
    """The DPO loss: a weighted mean of -log sigmoid(beta * h) over pairs.

    h = (log pi(y_w) - log pi_ref(y_w)) - (log pi(y_l) - log pi_ref(y_l)) is
    the log-ratio margin of each pair.

    Args:
        policy_chosen, policy_rejected: sequence log-probs under the policy, shape (B,).
        ref_chosen, ref_rejected: sequence log-probs under the frozen reference, shape (B,).
        beta: the KL coefficient, a positive float.
        weights: optional pair weights, shape (B,), summing to 1. None means 1/B each.

    Returns:
        The loss as a Python float.
    """
    raise NotImplementedError("dpo_loss")


def dpo_grad(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta, weights=None):
    """Gradient of dpo_loss with respect to the two policy log-probabilities.

    The reference log-probabilities are constants.

    Returns:
        A tuple (grad_chosen, grad_rejected), each shape (B,).
    """
    raise NotImplementedError("dpo_grad")


def ipo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, tau):
    """The sampled IPO loss: the mean of (h - 1 / (2 tau))^2 over pairs.

    Args:
        policy_chosen, policy_rejected, ref_chosen, ref_rejected: shape (B,), as in dpo_loss.
        tau: IPO's regularisation strength, a positive float.

    Returns:
        The loss as a Python float.
    """
    raise NotImplementedError("ipo_loss")


def ipo_grad(policy_chosen, policy_rejected, ref_chosen, ref_rejected, tau):
    """Gradient of ipo_loss with respect to the two policy log-probabilities.

    Returns:
        A tuple (grad_chosen, grad_rejected), each shape (B,).
    """
    raise NotImplementedError("ipo_grad")


def train_dpo_linear(E, theta0, chosen, rejected, beta, lr, steps):
    """Run DPO on one pair, for a softmax policy whose logits are E @ theta.

    The reference is the policy at theta0. Each step takes one gradient
    descent step on theta for the single pair (chosen, rejected).

    Args:
        E: one embedding per response, shape (N, d).
        theta0: the starting parameters, shape (d,).
        chosen, rejected: the indices of y_w and y_l.
        beta, lr, steps: the DPO coefficient, the step size, and the step count.

    Returns:
        The policy's log-probabilities over all N responses before training
        and after every step, shape (steps + 1, N).
    """
    raise NotImplementedError("train_dpo_linear")


def simpo_loss(logp_chosen, logp_rejected, len_chosen, len_rejected, beta, gamma):
    """The SimPO loss, averaged over the batch. There is no reference model.

    The reward of a response is beta times its average per-token log-probability:
    r = beta * log pi(y|x) / |y|. The loss is mean(-log sigmoid(r_w - r_l - gamma)),
    where gamma is a target margin.

    Args:
        logp_chosen, logp_rejected: summed sequence log-probs under the policy, shape (B,).
        len_chosen, len_rejected: token counts |y_w| and |y_l|, shape (B,).
        beta: the reward scale, a positive float.
        gamma: the target margin, a non-negative float.

    Returns:
        The loss as a Python float.
    """
    raise NotImplementedError("simpo_loss")


def kto_ref_point(policy_mismatched, ref_mismatched):
    """The KTO reference point z_ref: the mean log-ratio of mismatched outputs, clamped at 0.

    Args:
        policy_mismatched, ref_mismatched: log-probs of outputs paired with the
            wrong prompts, under the policy and the reference, shape (m,).

    Returns:
        z_ref as a Python float. It is a constant: no gradient flows through it.
    """
    raise NotImplementedError("kto_ref_point")


def kto_loss(policy_logp, ref_logp, desirable, beta, lambda_d, lambda_u, z_ref):
    """The KTO loss, averaged over a batch of single labelled examples.

    With r = log pi(y|x) - log pi_ref(y|x), a desirable example costs
    lambda_d * (1 - sigmoid(beta * (r - z_ref))) and an undesirable one costs
    lambda_u * (1 - sigmoid(beta * (z_ref - r))). This is the paper's lambda_y - v(x, y).

    Args:
        policy_logp, ref_logp: sequence log-probs, shape (B,).
        desirable: True for a thumbs-up example and False for a thumbs-down one, shape (B,).
        beta, lambda_d, lambda_u: the risk-aversion scale and the two loss weights.
        z_ref: the reference point, a float.

    Returns:
        The loss as a Python float.
    """
    raise NotImplementedError("kto_loss")


if __name__ == "__main__":
    # 1. The worked example from the lesson.
    beta = 0.1
    loss = dpo_loss(np.array([-10.0]), np.array([-12.0]), np.array([-11.0]), np.array([-11.5]), beta)
    g_w, g_l = dpo_grad(np.array([-10.0]), np.array([-12.0]), np.array([-11.0]), np.array([-11.5]), beta)
    print(f"DPO loss on the worked example: {loss:.4f}, d loss / d log pi(y_w) = {g_w[0]:.4f}")

    # 2. Likelihood displacement on a pair of similar responses.
    E = similar_pair_embeddings(32, 4, cosine=0.95, seed=0)
    hist = train_dpo_linear(E, np.zeros(4), 0, 1, beta=0.1, lr=1.0, steps=500)
    print(f"{'step':>5} {'log pi(y_w)':>12} {'log pi(y_l)':>12}")
    for step in (0, 100, 250, 500):
        print(f"{step:>5} {hist[step, 0]:>12.3f} {hist[step, 1]:>12.3f}")

    # 3. SimPO and KTO on the tiny batches from the lesson.
    s = simpo_loss(np.array([-20.0]), np.array([-36.0]), np.array([10]), np.array([12]), 2.0, 0.5)
    z = kto_ref_point(np.array([-9.5, -11.1]), np.array([-10.0, -11.0]))
    k = kto_loss(np.array([-9.0, -12.0]), np.array([-10.0, -11.5]), np.array([True, False]), 1.0, 1.0, 1.5, z)
    print(f"SimPO loss: {s:.4f}, KTO z_ref: {z:.2f}, KTO loss: {k:.4f}")
