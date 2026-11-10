"""Debugging RL runs: the diagnostics you read before you touch a hyperparameter.

Every function here works on plain arrays. A training *log* is a dict of 1-D
arrays with one entry per update and these keys:

    "rel_entropy"  mean policy entropy divided by its maximum, log(n_actions)
    "approx_kl"    the k3 estimate of the KL from the old to the new policy
    "ev"           explained variance of the critic on the value targets
    "target_std"   standard deviation of the value targets (the returns)

Fill in the functions below. Use numpy, and torch for overfit_one_batch.
"""

import numpy as np
import torch

torch.set_num_threads(1)

# Thresholds that diagnose() applies to the mean of the last quarter of the log.
KL_MAX = 0.05
ENTROPY_MIN = 0.05
EV_MIN = 0.05
TARGET_STD_MIN = 0.01
TARGET_STD_MAX = 100.0


def explained_variance(y_pred: np.ndarray, y: np.ndarray) -> float:
    """1 - Var(y - y_pred) / Var(y): the share of the targets' variance the prediction explains.

    A perfect prediction gives 1. A constant prediction gives 0, whatever the
    constant. A prediction worse than a constant gives a negative number. When
    the targets are constant, Var(y) is 0 and the ratio is undefined, so
    return nan.
    """
    raise NotImplementedError("explained_variance")


def value_target_correlation(y_pred: np.ndarray, y: np.ndarray) -> float:
    """The Pearson correlation between predictions and targets.

    Return nan when either one is constant.
    """
    raise NotImplementedError("value_target_correlation")


def relative_entropy(probs: np.ndarray) -> float:
    """Mean entropy of a batch of categorical policies, divided by log(n_actions).

    Args:
        probs: shape (B, A), each row sums to 1 and may contain exact zeros.

    Returns:
        1 for uniform rows, 0 for one-hot rows. A zero probability adds 0 to
        the entropy (0 * log 0 = 0), so the result is never nan.
    """
    raise NotImplementedError("relative_entropy")


def grad_norm(grads) -> float:
    """The global L2 norm of a list of gradient arrays, as if they were one long vector."""
    raise NotImplementedError("grad_norm")


def policy_loss(logits: np.ndarray, actions: np.ndarray, adv: np.ndarray, normalize: bool) -> float:
    """The policy-gradient loss -mean(A * log pi(a | s)) for a categorical policy.

    With normalize=True the advantages are first standardized to mean 0 and
    standard deviation 1: (A - mean) / (std + 1e-8). Args: logits (N, A),
    actions (N,) integers, adv (N,).
    """
    raise NotImplementedError("policy_loss")


def policy_loss_grad(logits: np.ndarray, actions: np.ndarray, adv: np.ndarray, normalize: bool) -> np.ndarray:
    """The gradient of policy_loss with respect to logits, shape (N, A).

    The advantages are data, not parameters: do not differentiate through the
    normalization.
    """
    raise NotImplementedError("policy_loss_grad")


def overfit_one_batch(net, x, y, steps: int = 300, lr: float = 1e-2):
    """Train `net` on one fixed batch with Adam and mean squared error.

    Args:
        net: a torch.nn.Module mapping (N, D) to (N, 1).
        x, y: one fixed batch, as torch tensors of shape (N, D) and (N, 1).

    Returns:
        (initial_loss, final_loss) as floats, measured on that same batch.
    """
    raise NotImplementedError("overfit_one_batch")


def diagnose(log: dict) -> str:
    """Name the most urgent problem in a training log, or "healthy".

    Average each series over its last quarter, max(1, T // 4) updates, then
    apply these rules in order and return the first that fires:

        "exploding_kl"     mean approx_kl      > 0.05
        "entropy_collapse" mean rel_entropy    < 0.05
        "reward_scale"     mean target_std     > 100 or < 0.01
        "dead_critic"      mean ev             < 0.05
        "healthy"          none of the above
    """
    raise NotImplementedError("diagnose")


if __name__ == "__main__":
    y = np.array([0.0, 2.0, 4.0, 6.0])
    print("explained variance of y_pred = [1, 2, 3, 4]:", explained_variance(np.array([1.0, 2.0, 3.0, 4.0]), y))
    print("explained variance of a constant:", explained_variance(np.full(4, 3.0), y))
    print("explained variance of 3 * y:", explained_variance(3 * y, y), "(correlation",
          value_target_correlation(3 * y, y), ")")
    print("relative entropy of [0.9, 0.1]:", round(relative_entropy(np.array([[0.9, 0.1]])), 4))

    rng = np.random.default_rng(0)
    logits = rng.normal(size=(64, 3))
    actions = rng.integers(0, 3, size=64)
    adv = rng.normal(size=64)
    for scale in (1.0, 100.0):
        for normalize in (False, True):
            g = policy_loss_grad(logits, actions, scale * adv, normalize)
            print(f"reward scale {scale:5.0f}, normalize={normalize!s:5}: grad norm {grad_norm([g]):.4f}")

    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
    xb = torch.randn(16, 4)
    yb = torch.sin(xb.sum(dim=1, keepdim=True))
    first, last = overfit_one_batch(net, xb, yb)
    print(f"overfit one batch: loss {first:.4f} -> {last:.6f}")

    healthy = {"rel_entropy": np.full(20, 0.5), "approx_kl": np.full(20, 0.01),
               "ev": np.full(20, 0.8), "target_std": np.full(20, 5.0)}
    print("diagnose(healthy log):", diagnose(healthy))
