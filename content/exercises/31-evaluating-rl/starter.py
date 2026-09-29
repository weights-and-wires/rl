"""Evaluating reinforcement learning: robust aggregates and honest intervals.

Every function takes `scores` of shape (n_runs, n_tasks): row n is one
training run (one seed), column m is one task, and scores[n, m] is that run's
normalised score on that task. This is the convention of the `rliable`
library (Agarwal et al., 2021).

Fill in the functions below. Use only numpy.
"""

import numpy as np


def iqm(scores: np.ndarray) -> float:
    """The interquartile mean of all n_runs * n_tasks scores pooled together.

    Sort the pooled scores, drop floor(0.25 * n) from each end, and average
    what remains. This matches scipy.stats.trim_mean(scores, 0.25, axis=None).
    """
    raise NotImplementedError("iqm")


def optimality_gap(scores: np.ndarray, threshold: float = 1.0) -> float:
    """threshold minus the mean over all runs and tasks of min(score, threshold)."""
    raise NotImplementedError("optimality_gap")


def performance_profile(scores: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """For each tau, the fraction of all runs whose score is strictly above tau.

    Args:
        scores: shape (n_runs, n_tasks).
        taus: thresholds, shape (T,).

    Returns:
        Shape (T,).
    """
    raise NotImplementedError("performance_profile")


def probability_of_improvement(scores_x: np.ndarray, scores_y: np.ndarray) -> float:
    """P(X > Y): the average over tasks of the chance a random run of X beats a random run of Y.

    On task m, compare every run of X with every run of Y: a win counts 1, a
    tie counts 1/2. Average over the pairs, then over tasks.

    Args:
        scores_x: shape (N, n_tasks).
        scores_y: shape (K, n_tasks). K can differ from N.
    """
    raise NotImplementedError("probability_of_improvement")


def stratified_resample(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One bootstrap resample that redraws runs within each task.

    Column m of the result holds n_runs draws, with replacement, from column m
    of scores. Draw every index at once with
    rng.integers(0, n_runs, size=(n_runs, n_tasks)).

    Returns:
        Shape (n_runs, n_tasks).
    """
    raise NotImplementedError("stratified_resample")


def bootstrap_replicates(scores: np.ndarray, statistic, n_boot: int, seed: int) -> np.ndarray:
    """statistic(resample) for n_boot stratified resamples.

    Make one generator, rng = np.random.default_rng(seed), and pass it to
    stratified_resample each time. `statistic` maps an (n_runs, n_tasks)
    array to a float.

    Returns:
        Shape (n_boot,).
    """
    raise NotImplementedError("bootstrap_replicates")


def percentile_ci(replicates: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """The percentile interval: the (1 - confidence) / 2 and (1 + confidence) / 2 quantiles.

    Use np.percentile, whose default linear interpolation the checks expect.

    Returns:
        (low, high) as floats.
    """
    raise NotImplementedError("percentile_ci")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_runs, n_tasks = 5, 8
    # Two algorithms with the same score distribution on every task.
    task_level = rng.uniform(0.2, 1.5, size=n_tasks)

    def run_paper(seed):
        g = np.random.default_rng(seed)
        a = task_level * g.gamma(4.0, 0.25, size=(n_runs, n_tasks))
        b = task_level * g.gamma(4.0, 0.25, size=(n_runs, n_tasks))
        return a, b

    a, b = run_paper(1)
    for name, stat in [("mean", lambda s: float(s.mean())),
                       ("median of task means", lambda s: float(np.median(s.mean(axis=0)))),
                       ("IQM", iqm)]:
        lo_a, hi_a = percentile_ci(bootstrap_replicates(a, stat, 2000, seed=0))
        lo_b, hi_b = percentile_ci(bootstrap_replicates(b, stat, 2000, seed=1))
        print(f"{name:21s} A: {stat(a):.3f} [{lo_a:.3f}, {hi_a:.3f}]   B: {stat(b):.3f} [{lo_b:.3f}, {hi_b:.3f}]")
    print(f"P(A > B) = {probability_of_improvement(a, b):.3f}, optimality gap of A = {optimality_gap(a):.3f}")

    taus = np.array([0.25, 0.5, 1.0, 1.5])
    for tau, frac in zip(taus, performance_profile(a, taus)):
        print(f"fraction of A's runs above {tau:.2f}: {frac:.3f}")

    # How often does the "better" algorithm by mean change, with 3 runs per task?
    flips = [bool(x[:3].mean() > y[:3].mean()) for x, y in (run_paper(s) for s in range(100, 300))]
    print(f"with 3 runs, A beats B by mean in {np.mean(flips):.0%} of 200 identical experiments")
