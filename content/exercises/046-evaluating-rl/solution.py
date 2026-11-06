"""Reference solution for the evaluating-RL exercise."""

import numpy as np


def iqm(scores: np.ndarray) -> float:
    x = np.sort(np.asarray(scores, dtype=float).ravel())
    cut = int(0.25 * x.size)
    return float(x[cut : x.size - cut].mean())


def performance_profile(scores: np.ndarray, taus: np.ndarray) -> np.ndarray:
    x = np.asarray(scores, dtype=float).ravel()
    taus = np.asarray(taus, dtype=float)
    return (x[None, :] > taus[:, None]).mean(axis=1)


def stratified_resample(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    n_runs, n_tasks = scores.shape
    idx = rng.integers(0, n_runs, size=(n_runs, n_tasks))
    return scores[idx, np.arange(n_tasks)[None, :]]


def bootstrap_replicates(scores: np.ndarray, statistic, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([statistic(stratified_resample(scores, rng)) for _ in range(n_boot)])


def percentile_ci(replicates: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    tail = 100.0 * (1.0 - confidence) / 2.0
    low, high = np.percentile(replicates, [tail, 100.0 - tail])
    return float(low), float(high)


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

    taus = np.array([0.25, 0.5, 1.0, 1.5])
    for tau, frac in zip(taus, performance_profile(a, taus)):
        print(f"fraction of A's runs above {tau:.2f}: {frac:.3f}")

    # How often does the "better" algorithm by mean change, with 3 runs per task?
    flips = [bool(x[:3].mean() > y[:3].mean()) for x, y in (run_paper(s) for s in range(100, 300))]
    print(f"with 3 runs, A beats B by mean in {np.mean(flips):.0%} of 200 identical experiments")
