"""Safety training against a spec: reference solution.

A prompt is a row of three binary features: harmful (h), looks fishy (f), and
encoded (e). The assistant refuses (action 0) or complies (action 1). Two
rewards score the two actions: one learned from labelled examples that can only
see the cue f, and one built from the spec's two rules, which a grader applies
by reading what the prompt asks.
"""

import numpy as np

REFUSE, COMPLY = 0, 1

# Columns of a prompt row: harmful h, looks fishy f, encoded e.
TRAIN_TYPES = np.array([[1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
TRAIN_P = np.array([0.27, 0.03, 0.14, 0.56])  # fishy prompts are mostly harmful
OOD_TYPES = np.array([[1, 0, 1], [0, 0, 1]], dtype=float)  # encoded: never fishy
OOD_P = np.array([0.5, 0.5])
REF_REFUSE = 0.2  # the base model refuses a fifth of the time


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def reference_policy(n: int) -> np.ndarray:
    """The base model: refuse with probability 0.2 on every prompt, shape (n, 2)."""
    return np.tile([REF_REFUSE, 1.0 - REF_REFUSE], (n, 1))


def rm_features(X: np.ndarray, action: int) -> np.ndarray:
    """What the examples-only reward model sees: a bias and the cue f, shape (n, 2)."""
    n = len(X)
    if action == REFUSE:
        return np.stack([np.ones(n), X[:, 1]], axis=1)
    return np.zeros((n, 2))


def refuse_preference(X: np.ndarray, noise: float) -> np.ndarray:
    """Probability that an annotator prefers refusing over complying, per prompt."""
    return np.where(X[:, 0] == 1, 1.0 - noise, noise)


def fit_bt(M: np.ndarray, q: np.ndarray, p: np.ndarray, iters: int = 30) -> np.ndarray:
    """Fit a Bradley-Terry reward by Newton's method on the population loss."""
    theta = np.zeros(M.shape[1])
    for _ in range(iters):
        s = sigmoid(M @ theta)
        grad = M.T @ (p * (s - q))
        hess = (M * (p * s * (1.0 - s))[:, None]).T @ M
        theta = theta - np.linalg.solve(hess, grad)
    return theta


def rule_features(X: np.ndarray, action: int) -> np.ndarray:
    """The spec's two rule violations for taking `action`, shape (n, 2).

    Column 0 counts complying with a harmful prompt. Column 1 counts refusing
    a benign one.
    """
    h = X[:, 0]
    if action == REFUSE:
        return np.stack([np.zeros_like(h), 1.0 - h], axis=1)
    return np.stack([h, np.zeros_like(h)], axis=1)


def reward_table(theta: np.ndarray, feats_refuse: np.ndarray, feats_comply: np.ndarray) -> np.ndarray:
    """Linear reward of each action on each prompt, shape (n, 2)."""
    return np.stack([feats_refuse @ theta, feats_comply @ theta], axis=1)


def kl_optimal(ref_probs: np.ndarray, R: np.ndarray, beta: float) -> np.ndarray:
    """The KL-regularized optimal policy: softmax of log pi_ref + R / beta, per row."""
    z = np.log(ref_probs) + R / beta
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def error_rates(X: np.ndarray, p: np.ndarray, pi: np.ndarray) -> tuple:
    """Exact (over_refusal, under_refusal) of a policy on a prompt distribution."""
    h = X[:, 0]
    over = np.sum(p * (1.0 - h) * pi[:, REFUSE]) / np.sum(p * (1.0 - h))
    under = np.sum(p * h * pi[:, COMPLY]) / np.sum(p * h)
    return float(over), float(under)


def frontier(X, p, R, ref_probs, betas) -> tuple:
    """Sweep beta and return arrays (over, under), one point per beta."""
    pts = [error_rates(X, p, kl_optimal(ref_probs, R, b)) for b in betas]
    over, under = zip(*pts)
    return np.array(over), np.array(under)


def weakly_dominates(a: tuple, b: tuple, tol: float = 1e-12) -> bool:
    """True when every point of frontier b has a point of a that is no worse on both rates."""
    a_over, a_under = a
    b_over, b_under = b
    for o, u in zip(b_over, b_under):
        if not np.any((a_over <= o + tol) & (a_under <= u + tol)):
            return False
    return True


if __name__ == "__main__":
    noise = 0.1
    q = refuse_preference(TRAIN_TYPES, noise)
    margin_rm = rm_features(TRAIN_TYPES, REFUSE) - rm_features(TRAIN_TYPES, COMPLY)
    theta_rm = fit_bt(margin_rm, q, TRAIN_P)
    margin_spec = rule_features(TRAIN_TYPES, REFUSE) - rule_features(TRAIN_TYPES, COMPLY)
    theta_spec = fit_bt(margin_spec, q, TRAIN_P)
    print("examples-only weights (bias, fishy):", np.round(theta_rm, 3))
    print("spec weights (comply-harmful, refuse-benign):", np.round(theta_spec, 3))

    betas = np.geomspace(0.05, 20.0, 40)
    fronts = {}
    for name, X, p in [("train", TRAIN_TYPES, TRAIN_P), ("encoded", OOD_TYPES, OOD_P)]:
        ref = reference_policy(len(X))
        R_rm = reward_table(theta_rm, rm_features(X, REFUSE), rm_features(X, COMPLY))
        R_spec = reward_table(theta_spec, rule_features(X, REFUSE), rule_features(X, COMPLY))
        fronts[name] = (frontier(X, p, R_rm, ref, betas), frontier(X, p, R_spec, ref, betas))
        for label, R in [("examples-only", R_rm), ("spec", R_spec)]:
            over, under = error_rates(X, p, kl_optimal(ref, R, 1.0))
            print(f"{name:8s} {label:14s} beta=1: over {over:.4f}, under {under:.4f}")
    for name, (rm_front, spec_front) in fronts.items():
        print(f"spec weakly dominates examples-only on {name}:", weakly_dominates(spec_front, rm_front))
