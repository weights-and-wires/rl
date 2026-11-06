"""Checks for the evaluating-RL exercise."""

import numpy as np

from rlcheck import almost as _almost, check, close, equal, fail, finite, has, shape


def almost(got, want, tol, label=None):
    """rlcheck.almost, except that NaN or infinity fails instead of slipping past the comparison."""
    finite(got, label=label)
    _almost(got, want, tol, label=label)


# The lesson's worked example: 4 runs (rows) on 2 tasks (columns).
EXAMPLE = np.array([[0.2, 1.1], [0.9, 0.3], [0.4, 0.8], [0.5, 5.0]])


def _trimmed_mean(x):
    """The IQM by hand: sort the pooled scores, drop floor(n / 4) from each end."""
    x = np.sort(np.ravel(x))
    k = len(x) // 4
    return x[k : len(x) - k].mean()


@check("iqm pools every run and task, drops a quarter from each end, and ignores the outlier")
def test_iqm(sub):
    iqm = has(sub, "iqm")
    almost(iqm(np.arange(1, 9).reshape(4, 2) / 10), 0.45, tol=1e-12, label="IQM of 0.1, ..., 0.8")
    almost(iqm(EXAMPLE), 0.65, tol=1e-12, label="IQM of the worked example")
    spoiled = EXAMPLE.copy()
    spoiled[3, 1] = 500.0
    almost(iqm(spoiled), 0.65, tol=1e-12, label="IQM after raising the outlier from 5 to 500")
    rng = np.random.default_rng(0)
    for shp in [(5, 3), (3, 3), (7, 2), (10, 1), (6, 5)]:
        x = rng.gamma(2.0, size=shp)
        almost(iqm(x), _trimmed_mean(x), tol=1e-12, label=f"IQM of a {shp} matrix ({x.size} scores)")


@check("performance_profile is the fraction of runs strictly above each tau, and its area is the mean")
def test_performance_profile(sub):
    performance_profile = has(sub, "performance_profile")
    taus = np.array([0.0, 0.5, 1.0, 4.9, 5.0])
    got = performance_profile(EXAMPLE, taus)
    shape(got, (5,), label="profile")
    close(got, np.array([1.0, 0.5, 0.25, 0.125, 0.0]), atol=1e-12,
          label="profile of the worked example at taus 0, 0.5, 1, 4.9, 5 (strictly above)")
    rng = np.random.default_rng(1)
    x = rng.gamma(2.0, 0.5, size=(6, 4))
    grid = np.linspace(0.0, x.max() + 1.0, 400001)
    prof = performance_profile(x, grid)
    if np.any(np.diff(prof) > 1e-12):
        fail("the profile must never rise as tau grows")
    area = prof[:-1].sum() * (grid[1] - grid[0])
    almost(area, x.mean(), tol=1e-4, label="area under the profile of non-negative scores against their mean")


@check("stratified_resample draws each task's runs from that task only, with replacement")
def test_stratified_resample(sub):
    stratified_resample = has(sub, "stratified_resample")
    scores = np.array([[0.0, 10.0, 20.0], [1.0, 11.0, 21.0]])
    rng = np.random.default_rng(3)
    draws = np.stack([stratified_resample(scores, rng) for _ in range(4000)])
    shape(draws[0], (2, 3), label="one resample")
    for m in range(3):
        if not np.all(np.isin(draws[:, :, m], scores[:, m])):
            fail(f"task {m}'s resampled runs include scores from another task; resample within each column")
    frac = (draws == scores[1][None, None, :]).mean()
    almost(frac, 0.5, tol=0.02, label="how often the second run is drawn (should be half)")
    same = np.mean([np.all(d[0] == d[1]) for d in draws])
    almost(same, 1 / 8, tol=0.02, label="chance both rows of a resample match on all 3 tasks (1/8 if independent)")
    a = stratified_resample(scores, np.random.default_rng(9))
    b = stratified_resample(scores, np.random.default_rng(9))
    equal(a, b, label="two resamples from generators with the same seed")


@check("bootstrap_replicates of the mean match the exact stratified bootstrap mean and variance")
def test_bootstrap_moments(sub):
    bootstrap_replicates = has(sub, "bootstrap_replicates")
    reps = bootstrap_replicates(EXAMPLE, lambda s: float(np.mean(s)), 20000, seed=0)
    shape(reps, (20000,), label="replicates")
    n, m = EXAMPLE.shape
    # Each task's mean is resampled independently; its bootstrap variance is var(runs) / n with ddof 0.
    want_var = np.sum(EXAMPLE.var(axis=0) / n) / m**2
    almost(want_var, 0.2225, tol=1e-12, label="the lesson's hand value")
    se = np.sqrt(want_var / len(reps))
    almost(reps.mean(), EXAMPLE.mean(), tol=4 * se, label="mean of the replicates")
    almost(reps.var() / want_var, 1.0, tol=0.05, label="variance of the replicates over the exact value")


@check("bootstrap intervals collapse when runs agree within every task, even if tasks differ")
def test_bootstrap_stratified_width(sub):
    bootstrap_replicates = has(sub, "bootstrap_replicates")
    percentile_ci = has(sub, "percentile_ci")
    iqm = has(sub, "iqm")
    scores = np.tile(np.array([[0.1, 0.7, 1.3, 2.0]]), (5, 1))
    reps = bootstrap_replicates(scores, iqm, 500, seed=1)
    low, high = percentile_ci(reps)
    if not high - low <= 1e-12:
        fail(f"every run agrees within each task, so the interval should have zero width, got [{low}, {high}]. "
             "Resampling across tasks mixes them.")
    almost(low, iqm(scores), tol=1e-12, label="the collapsed interval")


@check("percentile_ci reads the two tail percentiles of the replicates")
def test_percentile_ci(sub):
    percentile_ci = has(sub, "percentile_ci")
    reps = np.arange(101, dtype=float)
    close(np.array(percentile_ci(reps, 0.95)), np.array([2.5, 97.5]), atol=1e-12, label="95% interval of 0..100")
    close(np.array(percentile_ci(reps, 0.9)), np.array([5.0, 95.0]), atol=1e-12, label="90% interval of 0..100")
    shuffled = np.random.default_rng(4).permutation(reps)
    close(np.array(percentile_ci(shuffled)), np.array([2.5, 97.5]), atol=1e-12, label="order must not matter")


@check("a 95% stratified bootstrap interval for the mean covers the true mean in most repeated experiments")
def test_coverage(sub):
    bootstrap_replicates = has(sub, "bootstrap_replicates")
    percentile_ci = has(sub, "percentile_ci")
    rng = np.random.default_rng(5)
    task_means = np.array([0.5, 1.0, 2.0])
    covered = 0
    trials = 200
    for t in range(trials):
        scores = task_means + rng.normal(scale=0.5, size=(10, 3))
        low, high = percentile_ci(bootstrap_replicates(scores, lambda s: float(s.mean()), 400, seed=t))
        covered += low <= task_means.mean() <= high
    if covered < 166:
        fail(f"the 95% interval covered the true mean in {covered} of {trials} experiments, expected at least 166")
