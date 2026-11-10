"""Direct preference optimization, IPO, and likelihood displacement: reference solution.

Every loss here works on log-probabilities, never on text. A response's
log-probability under a policy is the sum of its per-token log-probabilities,
so the same functions serve a toy table and a real language model.

Shapes follow the course's toy-harness convention:

- per-token log-probabilities and response masks are `(B, T)`,
- sequence log-probabilities, rewards, and labels are `(B,)`.

"chosen" is the preferred response `y_w` and "rejected" is `y_l`.
"""

import numpy as np


# Helpers given to you. The same code sits in starter.py. -----------------


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
    """Sum the per-token log-probabilities of each response over its mask."""
    mask = np.asarray(mask, dtype=bool)
    # np.where, not a product: padding can hold -inf, and 0 * -inf is nan.
    return np.where(mask, token_logps, 0.0).sum(axis=-1)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta, weights=None):
    """Mean of -log sigmoid(beta * h), where h is the log-ratio margin."""
    h = (np.asarray(policy_chosen) - ref_chosen) - (np.asarray(policy_rejected) - ref_rejected)
    w = pair_weights(h.shape[0], weights)
    return float(np.sum(w * -log_sigmoid(beta * h)))


def dpo_grad(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta, weights=None):
    """Gradient of dpo_loss with respect to the two policy log-probabilities."""
    h = (np.asarray(policy_chosen) - ref_chosen) - (np.asarray(policy_rejected) - ref_rejected)
    w = pair_weights(h.shape[0], weights)
    # d/du [-log sigmoid(u)] = -sigmoid(-u), and u = beta * h.
    g = -beta * sigmoid(-beta * h) * w
    return g, -g


def ipo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, tau):
    """Mean of (h - 1 / (2 tau))^2, the sampled IPO loss of Azar et al."""
    h = (np.asarray(policy_chosen) - ref_chosen) - (np.asarray(policy_rejected) - ref_rejected)
    return float(np.mean((h - 1.0 / (2.0 * tau)) ** 2))


def ipo_grad(policy_chosen, policy_rejected, ref_chosen, ref_rejected, tau):
    """Gradient of ipo_loss with respect to the two policy log-probabilities."""
    h = (np.asarray(policy_chosen) - ref_chosen) - (np.asarray(policy_rejected) - ref_rejected)
    g = 2.0 * (h - 1.0 / (2.0 * tau)) / h.shape[0]
    return g, -g


def train_dpo_linear(E, theta0, chosen, rejected, beta, lr, steps):
    """Train a softmax policy with logits E @ theta on one DPO pair."""
    E = np.asarray(E, dtype=float)
    theta = np.array(theta0, dtype=float)
    ref = log_softmax(E @ theta)
    history = [ref]
    for _ in range(steps):
        logp = log_softmax(E @ theta)
        g_w, g_l = dpo_grad(
            logp[[chosen]], logp[[rejected]], ref[[chosen]], ref[[rejected]], beta
        )
        # d logp_y / d theta = e_y - E_pi[e].
        mean_e = np.exp(logp) @ E
        grad_theta = g_w[0] * (E[chosen] - mean_e) + g_l[0] * (E[rejected] - mean_e)
        theta = theta - lr * grad_theta
        history.append(log_softmax(E @ theta))
    return np.array(history)


def simpo_loss(logp_chosen, logp_rejected, len_chosen, len_rejected, beta, gamma):
    """The SimPO loss of Meng, Xia, and Chen (2024), averaged over the batch.

    The reward of a response is beta times its average per-token log-probability,
    so this function takes the summed log-probabilities and the token counts.
    There is no reference model. loss = mean(-log sigmoid(r_w - r_l - gamma)).
    """
    r_w = beta * np.asarray(logp_chosen, dtype=float) / np.asarray(len_chosen, dtype=float)
    r_l = beta * np.asarray(logp_rejected, dtype=float) / np.asarray(len_rejected, dtype=float)
    return float(np.mean(-log_sigmoid(r_w - r_l - gamma)))


def kto_ref_point(policy_mismatched, ref_mismatched):
    """The KTO reference point z_ref of Ethayarajh et al. (2024).

    Inputs are the log-probabilities of outputs paired with the wrong prompts,
    shape (m,). z_ref is the mean log-ratio of those outputs, clamped at zero.
    It is a constant in the loss, so no gradient flows through it.
    """
    log_ratio = np.asarray(policy_mismatched, dtype=float) - np.asarray(ref_mismatched, dtype=float)
    return float(max(0.0, np.mean(log_ratio)))


def kto_loss(policy_logp, ref_logp, desirable, beta, lambda_d, lambda_u, z_ref):
    """The KTO loss, averaged over a batch of single labelled examples.

    A desirable example costs lambda_D * (1 - sigmoid(beta * (r - z_ref))) and an
    undesirable one costs lambda_U * (1 - sigmoid(beta * (z_ref - r))), where
    r = log pi(y|x) - log pi_ref(y|x). This is the paper's lambda_y - v(x, y).
    """
    r = np.asarray(policy_logp, dtype=float) - np.asarray(ref_logp, dtype=float)
    good = np.asarray(desirable, dtype=bool)
    value = np.where(good, lambda_d * sigmoid(beta * (r - z_ref)), lambda_u * sigmoid(beta * (z_ref - r)))
    lam = np.where(good, lambda_d, lambda_u)
    return float(np.mean(lam - value))


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
