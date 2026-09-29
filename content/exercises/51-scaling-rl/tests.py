"""Checks for the scaling reinforcement learning exercise."""

import numpy as np
import torch

from rlcheck import almost, check, close, equal, fail, has, shape

torch.set_num_threads(1)

R0, A, B, C_MID = 0.1, 0.6, 1.5, 1000.0
C_GRID = np.logspace(np.log10(C_MID / 8), np.log10(8 * C_MID), 16)
A_GRID = np.linspace(0.3, 1.0, 701)


@check("scalerl_curve: 0.35 at C_mid, 0.5444 at 4 C_mid, and R0 far below C_mid")
def test_curve(sub):
    scalerl_curve = has(sub, "scalerl_curve")
    got = np.asarray(scalerl_curve(np.array([C_MID, 4 * C_MID, 1e-6]), R0, A, B, C_MID), dtype=float)
    close(got, np.array([0.35, 0.1 + 0.5 / 1.125, 0.1]), atol=1e-9,
          label="R at C = C_mid, 4 C_mid, and 1e-6")


@check("fit_scalerl recovers A, B, and C_mid exactly from noise-free runs")
def test_fit_exact(sub):
    scalerl_curve = has(sub, "scalerl_curve")
    fit_scalerl = has(sub, "fit_scalerl")
    R = scalerl_curve(C_GRID, R0, A, B, C_MID)
    a, b, c = fit_scalerl(C_GRID, R, R0, A_GRID)
    almost(a, A, 1e-9, label="A")
    almost(b, B, 1e-6, label="B")
    almost(c, C_MID, 1e-3, label="C_mid")


@check("With noise, fits that pass the midpoint agree on A; fits that stop before it scatter")
def test_fit_noisy(sub):
    scalerl_curve = has(sub, "scalerl_curve")
    fit_scalerl = has(sub, "fit_scalerl")
    early = C_GRID <= C_MID / 2
    full_a, early_a = [], []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        R = scalerl_curve(C_GRID, R0, A, B, C_MID) + rng.normal(0.0, 0.01, C_GRID.size)
        a, _, c = fit_scalerl(C_GRID, R, R0, A_GRID)
        if abs(a - A) > 0.03 or abs(c / C_MID - 1) > 0.12:
            fail(f"noise seed {seed}: fitting all 16 runs gave A = {a:.3f} and C_mid = {c:.0f}; "
                 f"expected A within 0.03 of {A} and C_mid within 12% of {C_MID:.0f}")
        full_a.append(a)
        early_a.append(fit_scalerl(C_GRID[early], R[early], R0, A_GRID)[0])
    spread = np.ptp(early_a)
    if spread < 0.3:
        fail(f"fits on runs up to C_mid / 2 should disagree about A by at least 0.3 across "
             f"noise seeds, but ranged only {spread:.3f}: {np.round(early_a, 3)}")


@check("effective_rank counts singular values up to a 1 - delta share of their sum")
def test_effective_rank(sub):
    effective_rank = has(sub, "effective_rank")
    rng = np.random.default_rng(0)
    U, _ = np.linalg.qr(rng.normal(size=(50, 4)))
    V, _ = np.linalg.qr(rng.normal(size=(6, 4)))
    F = U @ np.diag([4.0, 3.0, 2.0, 1.0]) @ V.T  # shares 0.4, 0.7, 0.9, 1.0
    for delta, want in ((0.01, 4), (0.15, 3), (0.35, 2), (0.65, 1)):
        equal(int(effective_rank(F, delta=delta)), want, label=f"srank at delta = {delta}")
    equal(int(effective_rank(np.ones((10, 5)))), 1, label="srank of a rank-one matrix")


@check("dormant_fraction uses the mean absolute activation divided by its layer average")
def test_dormant(sub):
    dormant_fraction = has(sub, "dormant_fraction")
    acts = np.array([[0.0, 1.0, 1.0, 5.0], [0.0, 1.0, 3.0, 5.0]])  # scores 0, 0.5, 1, 2.5
    for tau, want in ((0.0, 0.25), (0.5, 0.5), (1.0, 0.75)):
        almost(dormant_fraction(acts, tau=tau), want, 1e-12, label=f"tau = {tau}")


@check("hidden_features returns both post-ReLU layers, and train_task fits one target")
def test_features_and_train(sub):
    make_mlp = has(sub, "make_mlp")
    hidden_features = has(sub, "hidden_features")
    train_task = has(sub, "train_task")
    net = make_mlp(0)
    x = torch.randn(64, 8, generator=torch.Generator().manual_seed(1))
    h1, h2 = hidden_features(net, x)
    shape(h1, (64, 64), label="first hidden layer")
    shape(h2, (64, 64), label="second hidden layer")
    with torch.no_grad():
        want = torch.relu(net[2](torch.relu(net[0](x))))
    close(h2, want.numpy(), atol=1e-6, label="second hidden layer against a manual forward pass")
    y = torch.randn(64, 1, generator=torch.Generator().manual_seed(2))
    mse = train_task(net, x, y, steps=150, lr=1e-2)
    if not mse < 0.05 * float(y.var()):
        fail(f"150 Adam steps on 64 points should fit a fresh target to under 5% of its "
             f"variance; the final loss was {mse:.4f} against a variance of {float(y.var()):.4f}")


@check("redo_reset revives dead units without changing the network's outputs on x")
def test_redo(sub):
    make_mlp = has(sub, "make_mlp")
    redo_reset = has(sub, "redo_reset")
    hidden_features = has(sub, "hidden_features")
    net = make_mlp(3)
    x = torch.randn(256, 8, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        net[0].bias[:10] = -100.0  # kill ten units in each hidden layer
        net[2].bias[5:15] = -100.0
        before = net(x).clone()
        live_weight = net[2].weight[20:, 10:].clone()
    h1, h2 = hidden_features(net, x)
    dead_before = [int((h.max(axis=0) <= 0).sum()) for h in (h1, h2)]
    if min(dead_before) < 10:
        fail("the test failed to kill units; hidden_features may not apply the ReLU")
    redo_reset(net, x, seed=5)
    with torch.no_grad():
        after = net(x)
        close(after.numpy(), before.numpy(), atol=1e-6,
              label="outputs on x after ReDo (outgoing weights of reset units must be zero)")
        close(net[2].weight[20:, 10:].numpy(), live_weight.numpy(), atol=0.0,
              label="weights between units that were alive")
    h1, h2 = hidden_features(net, x)
    for h, name, n_before in ((h1, "first", dead_before[0]), (h2, "second", dead_before[1])):
        # A freshly drawn unit is occasionally dead on x by chance, so allow a few.
        n_dead = int((h.max(axis=0) <= 0).sum())
        if n_dead > 0.2 * n_before:
            fail(f"the {name} hidden layer had {n_before} dead units before ReDo and still has "
                 f"{n_dead} after; expected at most a fifth of them")


@check("Without resets, later tasks fit worse while dead units rise and effective rank falls")
def test_plasticity_loss(sub):
    continual_run = has(sub, "continual_run")
    out = continual_run(seed=1, reset=False)
    loss, dead, rank = out["loss"], out["dead"], out["rank"]
    shape(loss, (20,), label="per-task losses")
    first, last = loss[:3].mean(), loss[-3:].mean()
    if last < 3 * first:
        fail(f"the mean normalised loss on the last three tasks, {last:.3f}, should be at least "
             f"three times that on the first three, {first:.3f}")
    if dead[-3:].mean() < dead[0] + 0.3:
        fail(f"the dead fraction should rise by at least 0.3; it went from {dead[0]:.2f} to "
             f"{dead[-3:].mean():.2f}")
    if rank[-3:].mean() > rank[0] - 20:
        fail(f"the effective rank should fall by at least 20; it went from {rank[0]} to "
             f"{rank[-3:].mean():.1f}")


@check("With ReDo resets between tasks, the network keeps fitting new targets")
def test_resets_restore(sub):
    continual_run = has(sub, "continual_run")
    plain = continual_run(seed=1, reset=False)
    reset = continual_run(seed=1, reset=True)
    last = reset["loss"][-3:].mean()
    if last > 0.1 or reset["loss"].max() > 0.15:
        fail(f"with resets every task should fit to a normalised loss under 0.15, and the last "
             f"three average under 0.1; got max {reset['loss'].max():.3f}, last three {last:.3f}")
    if last > 0.25 * plain["loss"][-3:].mean():
        fail(f"resets should cut the late-task loss to under a quarter of the no-reset run: "
             f"{last:.3f} with resets against {plain['loss'][-3:].mean():.3f} without")
