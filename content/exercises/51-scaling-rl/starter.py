"""Scaling reinforcement learning: sigmoid compute curves and plasticity loss.

Part one fits ScaleRL's sigmoidal compute-performance curve with numpy alone
and shows when its asymptote can't be trusted. Part two trains one small
network on a sequence of regression tasks, measures how its ability to fit
new targets decays, and restores that ability with ReDo resets.
"""

import numpy as np
import torch

torch.set_num_threads(1)


def make_mlp(seed, n_in=8, width=64):
    """A two-hidden-layer ReLU network: Linear, ReLU, Linear, ReLU, Linear.

    This helper is given. net[0] and net[2] are the hidden Linear layers,
    net[1] and net[3] their ReLUs, and net[4] the output layer.
    """
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(n_in, width),
        torch.nn.ReLU(),
        torch.nn.Linear(width, width),
        torch.nn.ReLU(),
        torch.nn.Linear(width, 1),
    )


def scalerl_curve(C, R0, A, B, C_mid):
    """ScaleRL's sigmoidal compute-performance curve.

    R(C) = R0 + (A - R0) / (1 + (C_mid / C) ** B).

    Args:
        C: compute, a float or an array of positive floats.
        R0: the reward before training.
        A: the asymptote, the reward as compute grows without limit.
        B: the efficiency exponent, positive.
        C_mid: the compute at which half the gain A - R0 is reached.

    Returns:
        R(C), with the same shape as C.
    """
    raise NotImplementedError("scalerl_curve")


def fit_scalerl(C, R, R0, A_grid):
    """Fit A, B, and C_mid to measured runs, with R0 known.

    For each candidate A in A_grid, keep the runs with R0 < R < A, fit a
    straight line to log((A - R0) / (R - R0) - 1) against log C by least
    squares, and read off B = -slope and C_mid = exp(intercept / B). Skip
    candidates that leave fewer than two runs or give B <= 0. Score each
    candidate by the sum of squared errors of scalerl_curve on all the runs,
    and return the best.

    Args:
        C: compute of each run, shape (m,).
        R: measured reward of each run, shape (m,).
        R0: the starting reward.
        A_grid: candidate asymptotes, shape (g,).

    Returns:
        A tuple (A, B, C_mid) of floats.
    """
    raise NotImplementedError("fit_scalerl")


def effective_rank(features, delta=0.01):
    """The effective rank srank_delta of a feature matrix (Kumar et al., 2020).

    With singular values s_1 >= s_2 >= ..., srank is the smallest k such that
    (s_1 + ... + s_k) / (s_1 + s_2 + ...) >= 1 - delta.

    Args:
        features: shape (n, h), one row per input.
        delta: the share of the singular-value sum you may leave out.

    Returns:
        An int. Return 0 for an all-zero matrix.
    """
    raise NotImplementedError("effective_rank")


def dormant_fraction(activations, tau=0.0):
    """The fraction of tau-dormant units in a layer (Sokar et al., 2023).

    Each unit's score is its mean absolute activation over the inputs,
    divided by the average of that mean over all units in the layer. A unit
    is dormant when its score is at most tau. With tau = 0 this counts dead
    units, which output zero on every input.

    Args:
        activations: shape (n, h), post-ReLU activations of one layer.
        tau: the threshold, at least 0.

    Returns:
        A float in [0, 1]. Return 1.0 when every activation is zero.
    """
    raise NotImplementedError("dormant_fraction")


def hidden_features(net, x):
    """The activations of both hidden layers, after their ReLUs.

    Args:
        net: a network from make_mlp.
        x: inputs, a float tensor of shape (n, n_in).

    Returns:
        A tuple (h1, h2) of numpy arrays, each shape (n, width). Don't track
        gradients.
    """
    raise NotImplementedError("hidden_features")


def train_task(net, x, y, steps=150, lr=1e-2):
    """Fit targets y from inputs x with a fresh Adam optimiser.

    Create torch.optim.Adam(net.parameters(), lr=lr) and take `steps` full-batch
    steps on the mean squared error.

    Args:
        net: a network from make_mlp. It's updated in place.
        x: inputs, shape (n, n_in).
        y: targets, shape (n, 1).
        steps: the number of Adam steps.
        lr: the learning rate.

    Returns:
        The mean squared error after the last step, a float.
    """
    raise NotImplementedError("train_task")


def redo_reset(net, x, seed):
    """ReDo: reinitialise every dead hidden unit and zero its outgoing weights.

    A unit is dead when its post-ReLU activation is zero on every row of x.
    For each dead unit in hidden layer 1 (net[0]) and hidden layer 2 (net[2]),
    copy that unit's incoming weights and bias from make_mlp(seed), a fresh
    network of the same shape. Then set the unit's outgoing weights, its
    column in the next Linear layer (net[2] or net[4]), to zero. Find the
    dead units of both layers before changing anything. Work under
    torch.no_grad().

    Args:
        net: a network from make_mlp. It's modified in place.
        x: the inputs that decide which units are dead, shape (n, n_in).
        seed: the seed for the fresh weights.

    Returns:
        net.
    """
    raise NotImplementedError("redo_reset")


def continual_run(seed, n_tasks=20, steps=150, n_points=512, reset=False):
    """Train one network on a sequence of random-target regression tasks.

    Draw x once, then a fresh target y for every task, all from
    gen = torch.Generator().manual_seed(seed):
        x = torch.randn(n_points, 8, generator=gen)
        y = torch.randn(n_points, 1, generator=gen)   # once per task
    Build the network with make_mlp(seed). Before every task after the
    first, and only when reset is True, call
    redo_reset(net, x, seed=10_000 * (seed + 1) + task). Then call
    train_task(net, x, y, steps=steps) and record three numbers.

    Args:
        seed: the seed for the data and the network.
        n_tasks: the number of tasks.
        steps: Adam steps per task.
        n_points: the number of inputs.
        reset: whether to apply ReDo between tasks.

    Returns:
        A dict of arrays, each shape (n_tasks,):
            "loss": the final mean squared error divided by y.var().
            "dead": dormant_fraction of the second hidden layer, tau = 0.
            "rank": effective_rank of the second hidden layer, delta = 0.01.
    """
    raise NotImplementedError("continual_run")


if __name__ == "__main__":
    R0, A, B, C_mid = 0.1, 0.6, 1.5, 1000.0
    C = np.logspace(np.log10(C_mid / 8), np.log10(8 * C_mid), 16)
    rng = np.random.default_rng(0)
    R = scalerl_curve(C, R0, A, B, C_mid) + rng.normal(0.0, 0.01, C.size)
    grid = np.linspace(0.3, 1.0, 701)
    print("fit on runs up to 8 x C_mid:   A = %.3f, B = %.2f, C_mid = %.0f"
          % fit_scalerl(C, R, R0, grid))
    early = C <= C_mid / 2
    print("fit on runs up to C_mid / 2:   A = %.3f, B = %.2f, C_mid = %.0f"
          % fit_scalerl(C[early], R[early], R0, grid))

    for reset in (False, True):
        out = continual_run(seed=0, reset=reset)
        print(f"\nReDo resets: {reset}")
        print("task  loss   dead  rank")
        for t in range(0, 20, 3):
            print(f"{t + 1:4d}  {out['loss'][t]:.3f}  {out['dead'][t]:.2f}  {out['rank'][t]:4d}")
