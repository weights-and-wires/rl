"""Checks for the debugging-RL-runs exercise."""

import numpy as np
import torch

from rlcheck import almost as _almost, check, close, equal, fail, finite, finite_difference, has

torch.set_num_threads(1)


def almost(got, want, tol, label=None):
    """rlcheck.almost, except that NaN or infinity fails instead of slipping past the comparison."""
    finite(got, label=label)
    _almost(got, want, tol, label=label)


# The lesson's worked example: four value targets and some predictions of them.
Y = np.array([0.0, 2.0, 4.0, 6.0])


# Named wrong implementation: 1 - Var(y_pred) / Var(y). It scores a perfect
# prediction 0 and a constant prediction 1, which is backwards.
@check("explained_variance gives 1, 0, 0.75, and -3 on the worked table, nan for constant targets; correlation ignores scale")
def test_explained_variance(sub):
    explained_variance = has(sub, "explained_variance")
    corr = has(sub, "value_target_correlation")
    almost(explained_variance(Y.copy(), Y), 1.0, tol=1e-12, label="a perfect prediction")
    almost(explained_variance(np.full(4, 3.0), Y), 0.0, tol=1e-12, label="the constant 3 (the mean of y)")
    almost(explained_variance(np.array([1.0, 2.0, 3.0, 4.0]), Y), 0.75, tol=1e-12,
           label="y_pred = [1, 2, 3, 4] (residual variance 1.25 over target variance 5)")
    rng = np.random.default_rng(0)
    y = rng.normal(2.0, 3.0, size=50)
    for c in (0.0, 2.0, -7.0, 100.0):
        almost(explained_variance(np.full(50, c), y), 0.0, tol=1e-12,
               label=f"the constant prediction {c} (a constant never explains variance)")
    almost(explained_variance(y.copy(), y), 1.0, tol=1e-12, label="a perfect prediction of random targets")
    almost(explained_variance(y + 5.0, y), 1.0, tol=1e-12, label="a perfect prediction shifted by a constant")
    almost(explained_variance(-Y, Y), -3.0, tol=1e-12, label="y_pred = -y (residual 2y has 4x the variance)")
    almost(explained_variance(3 * Y, Y), -3.0, tol=1e-12, label="y_pred = 3y (residual -2y has 4x the variance)")
    # The arguments are not symmetric: swapping them gives a different number.
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    y2 = np.array([1.0, 2.0, 3.0, 8.0])
    want = 1.0 - np.var(y2 - y_pred) / np.var(y2)
    almost(explained_variance(y_pred, y2), want, tol=1e-12, label="explained_variance(y_pred, y) with y_pred first (not swapped)")
    out = explained_variance(np.array([1.0, 2.0, 3.0]), np.full(3, 4.0))
    if not np.isnan(out):
        fail(f"constant targets have zero variance, so the result is undefined: expected nan, got {out}")
    # Correlation ignores scale: 3y + 1 correlates perfectly with y, which explained_variance scores -3.
    almost(corr(3 * Y + 1.0, Y), 1.0, tol=1e-12, label="correlation of 3y + 1 with y")
    almost(corr(-Y, Y), -1.0, tol=1e-12, label="correlation of -y with y")
    rng = np.random.default_rng(1)
    y = rng.normal(size=40)
    p = y + rng.normal(size=40)
    almost(corr(p, y), np.corrcoef(p, y)[0, 1], tol=1e-12, label="correlation of noisy predictions")
    if not np.isnan(corr(np.full(4, 2.0), Y)):
        fail("a constant prediction has no correlation with anything: expected nan")


@check("relative_entropy is 1 for uniform rows, 0 for one-hot rows, and 0.4690 for [0.9, 0.1]")
def test_relative_entropy(sub):
    relative_entropy = has(sub, "relative_entropy")
    almost(relative_entropy(np.full((3, 4), 0.25)), 1.0, tol=1e-12, label="uniform over 4 actions")
    almost(relative_entropy(np.eye(4)), 0.0, tol=1e-12, label="one-hot rows (exact zeros must not give nan)")
    h = -(0.9 * np.log(0.9) + 0.1 * np.log(0.1)) / np.log(2)
    almost(relative_entropy(np.array([[0.9, 0.1]])), h, tol=1e-12, label="[0.9, 0.1] (0.4690)")
    almost(relative_entropy(np.array([[0.9, 0.1], [0.5, 0.5]])), (h + 1.0) / 2, tol=1e-12,
           label="the mean over a batch of two rows")


@check("grad_norm is the global L2 norm over a list of arrays")
def test_grad_norm(sub):
    grad_norm = has(sub, "grad_norm")
    almost(grad_norm([np.array([3.0, 4.0])]), 5.0, tol=1e-12, label="norm of [3, 4]")
    almost(grad_norm([np.array([[3.0]]), np.array([4.0, 0.0])]), 5.0, tol=1e-12,
           label="norm of arrays [[3]] and [4, 0] taken together")
    rng = np.random.default_rng(2)
    parts = [rng.normal(size=(3, 2)), rng.normal(size=4)]
    flat = np.concatenate([p.ravel() for p in parts])
    almost(grad_norm(parts), np.linalg.norm(flat), tol=1e-12, label="norm of two random arrays")


def _batch(seed=0, n=12, k=3):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, k)), rng.integers(0, k, size=n), rng.normal(size=n)


@check("policy_loss_grad matches finite differences, and normalized advantages make it ignore the reward scale")
def test_policy_loss_grad(sub):
    policy_loss = has(sub, "policy_loss")
    policy_loss_grad = has(sub, "policy_loss_grad")
    logits, actions, adv = _batch()
    for normalize in (False, True):
        got = policy_loss_grad(logits, actions, adv, normalize)
        want = finite_difference(
            lambda z: policy_loss(z.reshape(logits.shape), actions, adv, normalize), logits.ravel()
        ).reshape(logits.shape)
        close(got, want, atol=1e-7, label=f"gradient with respect to the logits, normalize={normalize}")
    _scale_invariance(sub)


def _scale_invariance(sub):
    """Normalized: unchanged by reward scale. Unnormalized: grows in proportion to it."""
    policy_loss_grad = has(sub, "policy_loss_grad")
    grad_norm = has(sub, "grad_norm")
    for seed in range(3):
        logits, actions, adv = _batch(seed)
        base_n = policy_loss_grad(logits, actions, adv, True)
        base_u = policy_loss_grad(logits, actions, adv, False)
        for scale in (0.01, 100.0, 1e4):
            close(policy_loss_grad(logits, actions, scale * adv, True), base_n, rtol=1e-5, atol=1e-9,
                  label=f"normalized gradient at reward scale {scale} against scale 1 (seed {seed})")
            ratio = grad_norm([policy_loss_grad(logits, actions, scale * adv, False)]) / grad_norm([base_u])
            almost(ratio, scale, tol=1e-9 * scale, label=f"unnormalized gradient-norm ratio at reward scale {scale}")
    # A constant shift of the rewards changes the unnormalized gradient but not the normalized one.
    logits, actions, adv = _batch(5)
    close(policy_loss_grad(logits, actions, adv + 7.0, True), policy_loss_grad(logits, actions, adv, True),
          rtol=1e-5, atol=1e-9, label="normalized gradient after adding 7 to every advantage")


def _regression_batch(seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(16, 4, generator=g)
    y = torch.sin(x.sum(dim=1, keepdim=True)) + 0.3 * x[:, :1]
    return x, y


@check("overfit_one_batch drives a healthy network's loss near zero and exposes a dead one")
def test_overfit(sub):
    overfit_one_batch = has(sub, "overfit_one_batch")
    for seed in range(5):
        torch.manual_seed(seed)
        x, y = _regression_batch(seed)
        net = torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
        first, last = overfit_one_batch(net, x, y)
        if not last < 0.01 * first:
            fail(f"a healthy network should overfit one batch: expected final loss below {0.01 * first:.5f}, "
                 f"got {last:.5f} (initial {first:.5f}, seed {seed})")
    # A network whose first layer is dead (every ReLU is off) can only fit the mean of y.
    torch.manual_seed(0)
    x, y = _regression_batch(0)
    dead = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    with torch.no_grad():
        dead[0].bias.fill_(-1000.0)
    _, last = overfit_one_batch(dead, x, y)
    floor = float(y.var(unbiased=False))
    if not last > 0.8 * floor:
        fail(f"a dead network cannot beat the variance of y: expected final loss near {floor:.4f} or above, "
             f"got {last:.4f}")


def _td_batch(seed=0, n=8):
    rng = np.random.default_rng(seed)
    v = rng.normal(0.0, 2.0, size=n)
    next_v = rng.normal(0.0, 2.0, size=n)
    rewards = rng.normal(1.0, 1.0, size=n)
    dones = (rng.random(n) < 0.25).astype(float)
    return v, rewards, next_v, dones


# Named wrong implementation: the broadcast loss. With v of shape (N, 1) and a
# target of shape (N,), ((v - target) ** 2).mean() averages an (N, N) grid, so
# every state is pulled toward the mean target instead of its own target.
def _broadcast_loss(v, rewards, next_v, dones, gamma):
    target = rewards + gamma * (1.0 - dones) * next_v
    return float(((v.reshape(-1, 1) - target) ** 2).mean())


_V = np.array([1.0, 2.0, 3.0])
_R = np.array([2.0, 1.0, 5.0])
_NV = np.array([9.0, 2.0, 1.0])
_D = np.array([1.0, 0.0, 0.0])


@check("critic_td_loss gives 7.25/3 on a hand-computed batch (targets [2, 2, 5.5])")
def test_critic_td_loss_value(sub):
    critic_td_loss = has(sub, "critic_td_loss")
    # The targets are r + 0.5 * (1 - done) * next_v = [2, 2, 5.5], so the errors are [-1, 0, -2.5].
    almost(critic_td_loss(_V, _R, _NV, _D, 0.5), 7.25 / 3.0, tol=1e-12,
           label="mean squared TD error for v=[1, 2, 3], r=[2, 1, 5], next_v=[9, 2, 1], done=[1, 0, 0], gamma=0.5")


@check("critic_td_loss reshapes an (N, 1) critic output and raises ValueError for any other shape mismatch")
def test_critic_td_loss_shapes(sub):
    critic_td_loss = has(sub, "critic_td_loss")
    almost(critic_td_loss(_V.reshape(3, 1), _R, _NV.reshape(3, 1), _D, 0.5), 7.25 / 3.0, tol=1e-12,
           label="the same batch with v and next_v of shape (3, 1)")
    for bad in (np.ones((3, 2)), np.ones(4), np.ones((1, 3))):
        try:
            critic_td_loss(bad, _R, _NV, _D, 0.5)
        except ValueError:
            continue
        fail(f"a value array of shape {bad.shape} cannot match 3 targets: expected ValueError, got a number")


@check("critic_td_loss matches the correct loss on a seeded batch, where the broadcast loss is clearly different")
def test_critic_td_loss_not_broadcast(sub):
    critic_td_loss = has(sub, "critic_td_loss")
    v, r, nv, d = _td_batch(0)
    target = r + 0.9 * (1.0 - d) * nv
    correct = float(np.mean((v - target) ** 2))
    wrong = _broadcast_loss(v, r, nv, d, 0.9)
    if not abs(wrong - correct) > 0.5:
        fail(f"test setup: the broadcast loss {wrong:.4f} should differ clearly from the correct loss {correct:.4f}")
    got = critic_td_loss(v.reshape(-1, 1), r, nv, d, 0.9)
    almost(got, correct, tol=1e-12, label=f"seeded batch with (N, 1) input (the broadcast loss would give {wrong:.4f})")


def make_log(kind, seed=0, T=100):
    """A seeded training log. Each failure breaks one signal and leaves the others healthy."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, T)
    noise = lambda s: rng.normal(0.0, s, size=T)  # noqa: E731
    log = {
        "rel_entropy": np.clip(0.95 - 0.55 * t + noise(0.02), 0.0, 1.0),
        "approx_kl": np.abs(0.012 + noise(0.004)),
        "ev": 0.85 * (1.0 - np.exp(-4.0 * t)) + noise(0.03),
        "target_std": 5.0 + noise(0.5),
    }
    if kind == "entropy_collapse":
        log["rel_entropy"] = 0.95 * np.exp(-6.0 * t) + np.abs(noise(0.002))
    elif kind == "dead_critic":
        log["ev"] = noise(0.02)
    elif kind == "exploding_kl":
        log["approx_kl"] = 0.01 * np.exp(5.0 * t) * (1.0 + np.abs(noise(0.1)))
    elif kind == "reward_scale_large":
        log["target_std"] = 5000.0 + noise(500.0)
        log["ev"] = noise(0.02)          # a critic cannot fit targets this large, so its EV is also near 0
    elif kind == "reward_scale_small":
        log["target_std"] = 5e-4 + np.abs(noise(1e-4))
    return log


CASES = [
    ("healthy", "healthy"),
    ("entropy_collapse", "entropy_collapse"),
    ("dead_critic", "dead_critic"),
    ("exploding_kl", "exploding_kl"),
    ("reward_scale_large", "reward_scale"),
    ("reward_scale_small", "reward_scale"),
]


@check("diagnose names each injected failure and calls a healthy run healthy, over five seeds")
def test_diagnose_cases(sub):
    diagnose = has(sub, "diagnose")
    for kind, want in CASES:
        for seed in range(5):
            got = diagnose(make_log(kind, seed))
            if got != want:
                fail(f"a '{kind}' log (seed {seed}) should be diagnosed {want!r}, got {got!r}")


def _flat(**over):
    base = {"rel_entropy": 0.5, "approx_kl": 0.01, "ev": 0.8, "target_std": 5.0}
    base.update(over)
    return {k: np.full(20, v) for k, v in base.items()}


@check("diagnose uses the stated thresholds and the last quarter of the log, and reward scale outranks a dead critic")
def test_diagnose_thresholds(sub):
    diagnose = has(sub, "diagnose")
    table = [
        (dict(approx_kl=0.049), "healthy"), (dict(approx_kl=0.051), "exploding_kl"),
        (dict(rel_entropy=0.051), "healthy"), (dict(rel_entropy=0.049), "entropy_collapse"),
        (dict(ev=0.051), "healthy"), (dict(ev=0.049), "dead_critic"),
        (dict(target_std=99.0), "healthy"), (dict(target_std=101.0), "reward_scale"),
        (dict(target_std=0.011), "healthy"), (dict(target_std=0.009), "reward_scale"),
        (dict(target_std=500.0, ev=0.0), "reward_scale"),
        (dict(approx_kl=0.5, rel_entropy=0.01), "exploding_kl"),
    ]
    for over, want in table:
        got = diagnose(_flat(**over))
        if got != want:
            fail(f"with {over} on an otherwise healthy log, expected {want!r}, got {got!r}")
    # Only the last quarter of the log counts, so a critic that recovers is healthy.
    ev = np.concatenate([np.full(75, -1.0), np.full(25, 0.8)])   # whole-log mean -0.55, last quarter 0.8
    log = _flat()
    log["ev"] = ev
    got = diagnose(log)
    equal(got, "healthy", label="a log whose critic was bad for 75 updates and good for the last 25")
    ev = np.concatenate([np.full(75, 0.8), np.full(25, 0.0)])    # whole-log mean 0.6, last quarter 0
    log["ev"] = ev
    equal(diagnose(log), "dead_critic", label="a log whose critic was good for 75 updates and dead for the last 25")
    short = {k: np.array([v]) for k, v in dict(rel_entropy=0.5, approx_kl=0.01, ev=0.8, target_std=5.0).items()}
    equal(diagnose(short), "healthy", label="a log with a single update")
