"""Direct preference optimisation and its relatives.

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


def kto_reference_point(policy_logps_mismatched, ref_logps_mismatched):
    """KTO's reference point z0, estimated from mismatched prompt-response pairs.

    Args:
        policy_logps_mismatched: log pi(y_j | x_i) for pairs whose response
            belongs to a different prompt, shape (m,).
        ref_logps_mismatched: log pi_ref(y_j | x_i) for the same pairs, shape (m,).

    Returns:
        max(0, mean log-ratio), as a Python float. It is a constant in the loss.
    """
    raise NotImplementedError("kto_reference_point")


def kto_loss(policy_logps, ref_logps, desirable, z0, beta, lambda_d=1.0, lambda_u=1.0):
    """The KTO loss: the mean of lambda_y - v(x, y) over unpaired responses.

    With r = log pi(y) - log pi_ref(y), v is lambda_d * sigmoid(beta * (r - z0))
    for a desirable response and lambda_u * sigmoid(beta * (z0 - r)) for an
    undesirable one.

    Args:
        policy_logps, ref_logps: sequence log-probs, shape (B,).
        desirable: booleans, shape (B,). True marks a response labelled good.
        z0: the reference point, a float.
        beta: KTO's temperature, a positive float.
        lambda_d, lambda_u: weights on desirable and undesirable responses.

    Returns:
        The loss as a Python float.
    """
    raise NotImplementedError("kto_loss")


def simpo_loss(token_logps_chosen, mask_chosen, token_logps_rejected, mask_rejected, beta, gamma):
    """The SimPO loss, which needs no reference model.

    The mean over pairs of -log sigmoid(beta * avg_w - beta * avg_l - gamma),
    where avg is a response's log-probability divided by its token count.

    Args:
        token_logps_chosen, mask_chosen: per-token log-probs and mask of y_w, shape (B, T).
        token_logps_rejected, mask_rejected: the same for y_l, shape (B, T2).
        beta: the reward scale, a positive float.
        gamma: the target margin, a float.

    Returns:
        The loss as a Python float.
    """
    raise NotImplementedError("simpo_loss")


def fit_dpo_exact(ref_logprobs, reward, beta, steps=2000, lr=20.0):
    """Minimise the expected DPO loss exactly, on a tabular policy per prompt.

    For each prompt, every ordered pair (i, j) with i != j is one DPO pair
    with chosen i and rejected j. Its weight is
    pi_ref(i) * pi_ref(j) * sigmoid(reward[i] - reward[j]), normalised to sum
    to 1 over the prompt's pairs. Start the policy's logits at ref_logprobs
    and run `steps` steps of gradient descent on the logits.

    Args:
        ref_logprobs: log pi_ref over each prompt's responses, shape (P, N).
        reward: the hidden Bradley-Terry reward, shape (P, N).
        beta: the DPO coefficient.
        steps, lr: the number of gradient steps and the step size.

    Returns:
        The fitted policy's log-probabilities, shape (P, N).
    """
    raise NotImplementedError("fit_dpo_exact")


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


if __name__ == "__main__":
    # 1. The worked example from the lesson.
    beta = 0.1
    loss = dpo_loss(np.array([-10.0]), np.array([-12.0]), np.array([-11.0]), np.array([-11.5]), beta)
    g_w, g_l = dpo_grad(np.array([-10.0]), np.array([-12.0]), np.array([-11.0]), np.array([-11.5]), beta)
    print(f"DPO loss on the worked example: {loss:.4f}, d loss / d log pi(y_w) = {g_w[0]:.4f}")

    # 2. DPO recovers a hidden reward up to a per-prompt constant.
    rng = np.random.default_rng(0)
    ref = log_softmax(rng.normal(size=(2, 9)))
    reward = rng.normal(size=(2, 9))
    fitted = fit_dpo_exact(ref, reward, beta=0.5)
    gap = 0.5 * (fitted - ref) - reward
    print("implicit reward minus true reward, per prompt (mean, spread):")
    for p in range(2):
        print(f"  prompt {p}: {gap[p].mean():+.4f}, {gap[p].std():.2e}")

    # 3. Likelihood displacement on a pair of similar responses.
    E = similar_pair_embeddings(32, 4, cosine=0.95, seed=0)
    hist = train_dpo_linear(E, np.zeros(4), 0, 1, beta=0.1, lr=1.0, steps=500)
    print(f"{'step':>5} {'log pi(y_w)':>12} {'log pi(y_l)':>12}")
    for step in (0, 100, 250, 500):
        print(f"{step:>5} {hist[step, 0]:>12.3f} {hist[step, 1]:>12.3f}")
