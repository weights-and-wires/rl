"""Direct preference optimisation and its relatives: reference solution.

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


def kto_reference_point(policy_logps_mismatched, ref_logps_mismatched):
    """KTO's reference point z0: the clamped mean log-ratio on mismatched pairs."""
    ratio = np.asarray(policy_logps_mismatched, dtype=float) - ref_logps_mismatched
    return float(max(0.0, np.mean(ratio)))


def kto_loss(policy_logps, ref_logps, desirable, z0, beta, lambda_d=1.0, lambda_u=1.0):
    """Mean of lambda_y - v(x, y) over a batch of unpaired, labelled responses."""
    r = np.asarray(policy_logps, dtype=float) - ref_logps
    desirable = np.asarray(desirable, dtype=bool)
    v = np.where(
        desirable,
        lambda_d * sigmoid(beta * (r - z0)),
        lambda_u * sigmoid(beta * (z0 - r)),
    )
    lam = np.where(desirable, lambda_d, lambda_u)
    return float(np.mean(lam - v))


def simpo_loss(token_logps_chosen, mask_chosen, token_logps_rejected, mask_rejected, beta, gamma):
    """Mean of -log sigmoid(beta * avg_w - beta * avg_l - gamma), with no reference."""
    avg_w = sequence_logprob(token_logps_chosen, mask_chosen) / np.sum(mask_chosen, axis=-1)
    avg_l = sequence_logprob(token_logps_rejected, mask_rejected) / np.sum(mask_rejected, axis=-1)
    return float(np.mean(-log_sigmoid(beta * avg_w - beta * avg_l - gamma)))


def fit_dpo_exact(ref_logprobs, reward, beta, steps=2000, lr=20.0):
    """Minimise the expected DPO loss over every ordered pair of responses."""
    ref_logprobs = np.asarray(ref_logprobs, dtype=float)
    reward = np.asarray(reward, dtype=float)
    n_prompts, n = ref_logprobs.shape
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    off = i != j
    i, j = i[off], j[off]
    ref_probs = np.exp(ref_logprobs)
    # Weight of "i was shown first, j second, and i won" under Bradley-Terry.
    w = ref_probs[:, i] * ref_probs[:, j] * sigmoid(reward[:, i] - reward[:, j])
    w = w / w.sum(axis=1, keepdims=True)

    logits = ref_logprobs.copy()
    for _ in range(steps):
        logp = log_softmax(logits)
        grad_logits = np.zeros_like(logits)
        for p in range(n_prompts):
            g_w, g_l = dpo_grad(
                logp[p, i], logp[p, j], ref_logprobs[p, i], ref_logprobs[p, j], beta, w[p]
            )
            g_logp = np.bincount(i, g_w, n) + np.bincount(j, g_l, n)
            # Chain rule through log-softmax: d logp_k / d z_m = [k == m] - pi_m.
            grad_logits[p] = g_logp - np.exp(logp[p]) * g_logp.sum()
        logits -= lr * grad_logits
    return log_softmax(logits)


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
