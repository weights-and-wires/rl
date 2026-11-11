"""Credit assignment for language models: reference solution.

The toy reasoning task has T = 4 steps. Step j is correct with the known
probability STEP_P[j], independently of the other steps, and the final answer is
correct if and only if every step is correct. A prefix is therefore a 0/1 array
of step outcomes, and its exact value is a product of probabilities.
"""

import numpy as np

STEP_P = np.array([0.9, 0.8, 0.7, 0.6])   # chance that each step is correct
T = len(STEP_P)


def exact_value(prefix: np.ndarray, step_p: np.ndarray = STEP_P) -> float:
    """Exact probability that the final answer is correct, given a prefix.

    Args:
        prefix: the outcomes of the first k steps, shape (k,), 1 for a correct
            step and 0 for a wrong one. k runs from 0 to T.
        step_p: the chance that each of the T steps is correct, shape (T,).

    Returns:
        0.0 if any step in the prefix is wrong. Otherwise the product of
        step_p over the steps that have not been written yet. A complete
        prefix of correct steps is worth 1.0.
    """
    prefix = np.asarray(prefix)
    if not prefix.all():
        return 0.0
    return float(np.prod(step_p[len(prefix):]))


def mc_value(prefix, K: int, rng: np.random.Generator, step_p: np.ndarray = STEP_P):
    """Estimate the value of a prefix from K rollouts of the policy.

    A rollout writes the remaining steps, each correct with probability
    step_p[j], and is a success when the prefix and every remaining step are
    correct. Draw the outcomes with one call, rng.random((K, T - k)) < step_p[k:].

    Returns:
        (estimate, stderr). The estimate is the mean success over the K
        rollouts. The standard error is the sample standard deviation
        (ddof=1) divided by sqrt(K).
    """
    prefix = np.asarray(prefix)
    k = len(prefix)
    rest = rng.random((K, len(step_p) - k)) < step_p[k:]
    success = rest.all(axis=1) & bool(prefix.all())
    return float(success.mean()), float(success.std(ddof=1) / np.sqrt(K))


def vine_advantages(outcomes, K: int, rng: np.random.Generator, step_p: np.ndarray = STEP_P):
    """Per-step advantages from Monte Carlo values at every step boundary.

    Args:
        outcomes: the step outcomes of one sampled response, shape (T,).
        K: rollouts per boundary.

    Returns:
        (advantages, v_hat). v_hat has shape (T + 1,): v_hat[k] is the
        mc_value of outcomes[:k] for k < T, and v_hat[T] is the reward of the
        finished response, 1.0 if every step is correct, else 0.0. The
        advantage of step k is v_hat[k + 1] - v_hat[k], shape (T,).
    """
    outcomes = np.asarray(outcomes)
    v_hat = np.empty(len(step_p) + 1)
    for k in range(len(step_p)):
        v_hat[k] = mc_value(outcomes[:k], K, rng, step_p)[0]
    v_hat[-1] = float(outcomes.all())
    return np.diff(v_hat), v_hat


def math_shepherd_labels(rollout_correct: np.ndarray, mode: str) -> np.ndarray:
    """Automatic step labels from completion rollouts.

    Args:
        rollout_correct: shape (T, K). Entry [k, i] is 1 when rollout i, which
            starts from the prefix that ends with step k, reached a correct answer.
        mode: "hard" or "soft".

    Returns:
        shape (T,). "hard": 1.0 if any rollout from the prefix is correct, else
        0.0. "soft": the fraction of rollouts that are correct.
    """
    rollout_correct = np.asarray(rollout_correct, dtype=float)
    if mode == "hard":
        return rollout_correct.max(axis=1)
    if mode == "soft":
        return rollout_correct.mean(axis=1)
    raise ValueError(f"mode must be 'hard' or 'soft', got {mode!r}")


def implicit_prm_rewards(logp, logp_ref, step_ids, beta: float, n_steps: int) -> np.ndarray:
    """Process rewards read off an outcome-trained model's log ratio.

    Args:
        logp: log pi(token | prefix) for each token of one response, shape (L,).
        logp_ref: the same under the reference model, shape (L,).
        step_ids: the step each token belongs to, ints in [0, n_steps), shape (L,).
        beta: the reward scale of the implicit reward.
        n_steps: the number of steps.

    Returns:
        shape (n_steps,). Entry j is beta times the sum, over the tokens of
        step j, of logp - logp_ref.
    """
    per_token = beta * (np.asarray(logp) - np.asarray(logp_ref))
    return np.bincount(np.asarray(step_ids), weights=per_token, minlength=n_steps)


def credit_errors(step_p: np.ndarray, K: int, n_samples: int, seed: int):
    """Mean squared error of two per-step advantage estimates against the exact one.

    For each of n_samples responses (each step correct with probability
    step_p[j]), the exact advantage of step k is exact_value(outcomes[:k + 1])
    minus exact_value(outcomes[:k]). The uniform estimate is the sequence-level
    advantage R - V(root), with the exact root value, repeated at every step.
    The vine estimate is vine_advantages with K rollouts per boundary.

    Returns:
        (mse_uniform, mse_vine), each shape (T,): the mean over responses of
        the squared difference to the exact advantage, per step.
    """
    rng = np.random.default_rng(seed)
    T_ = len(step_p)
    se_uniform = np.zeros(T_)
    se_vine = np.zeros(T_)
    v_root = exact_value(np.array([], dtype=int), step_p)
    for _ in range(n_samples):
        outcomes = (rng.random(T_) < step_p).astype(int)
        values = [exact_value(outcomes[:k], step_p) for k in range(T_ + 1)]
        exact = np.diff(values)
        uniform = np.full(T_, outcomes.all() - v_root)
        vine, _ = vine_advantages(outcomes, K, rng, step_p)
        se_uniform += (uniform - exact) ** 2
        se_vine += (vine - exact) ** 2
    return se_uniform / n_samples, se_vine / n_samples


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    outcomes = np.array([1, 1, 0, 1])
    print("exact value of each prefix:",
          [round(exact_value(outcomes[:k]), 4) for k in range(T + 1)])
    for K in (4, 64, 1024):
        adv, v_hat = vine_advantages(outcomes, K, rng)
        print(f"K={K:<5d} vine advantages {np.round(adv, 3)}  sum {adv.sum():+.3f}")
    uniform, vine = credit_errors(STEP_P, K=16, n_samples=2000, seed=0)
    print("squared error per step, uniform:", np.round(uniform, 3))
    print("squared error per step, vine K=16:", np.round(vine, 3))
