"""Debugging RL runs: the diagnostics you read before you touch a hyperparameter.

Every function here works on plain arrays, so the same code serves a toy policy
and a large one. A training *log* is a dict of 1-D arrays with one entry per
update and these keys:

    "rel_entropy"  mean policy entropy divided by its maximum, log(n_actions)
    "approx_kl"    the k3 estimate of the KL from the old to the new policy
    "ev"           explained variance of the critic on the value targets
    "target_std"   standard deviation of the value targets (the returns)
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
    constant, because subtracting a constant does not change a variance. A
    prediction worse than a constant gives a negative number. When the targets
    are constant, Var(y) is 0 and the ratio is undefined, so return nan.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y = np.asarray(y, dtype=float)
    var_y = np.var(y)
    if var_y == 0.0:
        return float("nan")
    return float(1.0 - np.var(y - y_pred) / var_y)


def value_target_correlation(y_pred: np.ndarray, y: np.ndarray) -> float:
    """The Pearson correlation between predictions and targets.

    Return nan when either one is constant. Correlation ignores scale and
    offset, so it can be 1 for a critic whose explained variance is negative.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(y_pred) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(y_pred, y)[0, 1])


def relative_entropy(probs: np.ndarray) -> float:
    """Mean entropy of a batch of categorical policies, divided by log(n_actions).

    Args:
        probs: shape (B, A), each row sums to 1 and may contain exact zeros.

    Returns:
        1 for uniform rows, 0 for one-hot rows. A zero probability adds 0 to
        the entropy (0 * log 0 = 0), so the result is never nan.
    """
    p = np.asarray(probs, dtype=float)
    safe = np.where(p > 0.0, p, 1.0)
    entropy = -(p * np.log(safe)).sum(axis=-1)
    return float(entropy.mean() / np.log(p.shape[-1]))


def grad_norm(grads) -> float:
    """The global L2 norm of a list of gradient arrays, as if they were one long vector."""
    return float(np.sqrt(sum(float(np.sum(np.asarray(g, dtype=float) ** 2)) for g in grads)))


def _advantages(adv: np.ndarray, normalize: bool) -> np.ndarray:
    adv = np.asarray(adv, dtype=float)
    if normalize:
        return (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def policy_loss(logits: np.ndarray, actions: np.ndarray, adv: np.ndarray, normalize: bool) -> float:
    """The policy-gradient loss -mean(A * log pi(a | s)) for a categorical policy.

    With normalize=True the advantages are first standardized to mean 0 and
    standard deviation 1: (A - mean) / (std + 1e-8). Args: logits (N, A),
    actions (N,) integers, adv (N,).
    """
    logits = np.asarray(logits, dtype=float)
    logp = np.log(_softmax(logits))
    chosen = logp[np.arange(len(actions)), actions]
    return float(-(_advantages(adv, normalize) * chosen).mean())


def policy_loss_grad(logits: np.ndarray, actions: np.ndarray, adv: np.ndarray, normalize: bool) -> np.ndarray:
    """The gradient of policy_loss with respect to logits, shape (N, A).

    The advantages are data, not parameters: do not differentiate through the
    normalization.
    """
    logits = np.asarray(logits, dtype=float)
    n = len(actions)
    onehot = np.zeros_like(logits)
    onehot[np.arange(n), actions] = 1.0
    a = _advantages(adv, normalize)
    return -(a[:, None] * (onehot - _softmax(logits))) / n


def overfit_one_batch(net, x, y, steps: int = 300, lr: float = 1e-2):
    """Train `net` on one fixed batch with Adam and mean squared error.

    Args:
        net: a torch.nn.Module mapping (N, D) to (N, 1).
        x, y: one fixed batch, as torch tensors of shape (N, D) and (N, 1).

    Returns:
        (initial_loss, final_loss) as floats, measured on that same batch.
        A correct network, loss, and optimizer drive the loss near zero.
    """
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        initial = float(loss_fn(net(x), y))
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(net(x), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return initial, float(loss_fn(net(x), y))


def critic_td_loss(v, rewards, next_v, dones, gamma):
    """Mean squared TD error of a critic: mean((v - (r + gamma * (1 - done) * next_v)) ** 2).

    Args:
        v: critic values, shape (N,) or (N, 1).
        rewards, dones: shape (N,). dones is 1 where the episode ended.
        next_v: critic values of the next states, shape (N,) or (N, 1).
        gamma: the discount.

    A value array of shape (N, 1) is reshaped to (N,). Any other shape that does
    not match (N,) raises ValueError. Without this, (N, 1) - (N,) broadcasts to an
    (N, N) grid and the loss silently trains the critic toward the mean target.
    The target is data, so no gradient flows through next_v.
    """
    rewards = np.asarray(rewards, dtype=float)
    dones = np.asarray(dones, dtype=float)
    n = rewards.shape[0]

    def flat(x, name):
        x = np.asarray(x, dtype=float)
        if x.shape == (n, 1):
            x = x.reshape(n)
        if x.shape != (n,):
            raise ValueError(f"{name} has shape {x.shape}, expected ({n},) or ({n}, 1)")
        return x

    v, next_v = flat(v, "v"), flat(next_v, "next_v")
    if dones.shape != (n,):
        raise ValueError(f"dones has shape {dones.shape}, expected ({n},)")
    target = rewards + gamma * (1.0 - dones) * next_v
    return float(np.mean((v - target) ** 2))


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
    def tail_mean(key):
        series = np.asarray(log[key], dtype=float)
        return float(series[-max(1, len(series) // 4):].mean())

    if tail_mean("approx_kl") > KL_MAX:
        return "exploding_kl"
    if tail_mean("rel_entropy") < ENTROPY_MIN:
        return "entropy_collapse"
    scale = tail_mean("target_std")
    if scale > TARGET_STD_MAX or scale < TARGET_STD_MIN:
        return "reward_scale"
    if tail_mean("ev") < EV_MIN:
        return "dead_critic"
    return "healthy"


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
