"""Reference solution for the scaling reinforcement learning exercise."""

import numpy as np
import torch

torch.set_num_threads(1)


def make_mlp(seed, n_in=8, width=64):
    """A two-hidden-layer ReLU network: Linear, ReLU, Linear, ReLU, Linear."""
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(n_in, width),
        torch.nn.ReLU(),
        torch.nn.Linear(width, width),
        torch.nn.ReLU(),
        torch.nn.Linear(width, 1),
    )


def scalerl_curve(C, R0, A, B, C_mid):
    """ScaleRL's sigmoid: R0 + (A - R0) / (1 + (C_mid / C) ** B)."""
    C = np.asarray(C, dtype=float)
    return R0 + (A - R0) / (1.0 + (C_mid / C) ** B)


def fit_scalerl(C, R, R0, A_grid):
    """Fit A, B, and C_mid by profiling over A and a line fit in log compute."""
    C = np.asarray(C, dtype=float)
    R = np.asarray(R, dtype=float)
    log_c = np.log(C)
    best = None
    for A in A_grid:
        keep = (R > R0) & (R < A)
        if keep.sum() < 2:
            continue
        # For fixed A: log((A - R0) / (R - R0) - 1) = B log C_mid - B log C.
        y = np.log((A - R0) / (R[keep] - R0) - 1.0)
        X = np.stack([np.ones(keep.sum()), log_c[keep]], axis=1)
        (intercept, slope), *_ = np.linalg.lstsq(X, y, rcond=None)
        B = -slope
        if B <= 0:
            continue
        C_mid = float(np.exp(intercept / B))
        sse = float(np.sum((scalerl_curve(C, R0, A, B, C_mid) - R) ** 2))
        if best is None or sse < best[0]:
            best = (sse, float(A), float(B), C_mid)
    if best is None:
        raise ValueError("no value in A_grid gives a valid fit")
    return best[1], best[2], best[3]


def effective_rank(features, delta=0.01):
    """srank: the fewest singular values that hold a 1 - delta share of their sum."""
    s = np.linalg.svd(np.asarray(features, dtype=float), compute_uv=False)
    total = s.sum()
    if total == 0:
        return 0
    share = np.cumsum(s) / total
    return int(np.searchsorted(share, 1.0 - delta - 1e-12) + 1)


def dormant_fraction(activations, tau=0.0):
    """The share of units whose normalised mean absolute activation is at most tau."""
    a = np.abs(np.asarray(activations, dtype=float)).mean(axis=0)
    mean = a.mean()
    if mean == 0:
        return 1.0
    return float(np.mean(a / mean <= tau))


def hidden_features(net, x):
    """The post-ReLU activations of both hidden layers, as numpy arrays."""
    with torch.no_grad():
        h1 = net[1](net[0](x))
        h2 = net[3](net[2](h1))
    return h1.numpy(), h2.numpy()


def train_task(net, x, y, steps=150, lr=1e-2):
    """Fit y from x with a fresh Adam optimiser. Returns the final mean squared error."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.mean((net(x) - y) ** 2)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(torch.mean((net(x) - y) ** 2))


def redo_reset(net, x, seed):
    """Reinitialise every dead hidden unit on x, and zero its outgoing weights."""
    fresh = make_mlp(seed, n_in=net[0].in_features, width=net[0].out_features)
    h1, h2 = hidden_features(net, x)
    with torch.no_grad():
        for layer, after, h, new in ((0, 2, h1, fresh[0]), (2, 4, h2, fresh[2])):
            dead = torch.from_numpy(h.max(axis=0) <= 0)
            net[layer].weight[dead] = new.weight[dead]
            net[layer].bias[dead] = new.bias[dead]
            net[after].weight[:, dead] = 0.0
    return net


def continual_run(seed, n_tasks=20, steps=150, n_points=512, reset=False):
    """Train one network on a sequence of random-target regression tasks."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n_points, 8, generator=gen)
    net = make_mlp(seed)
    losses, dead, rank = [], [], []
    for task in range(n_tasks):
        y = torch.randn(n_points, 1, generator=gen)
        if reset and task > 0:
            redo_reset(net, x, seed=10_000 * (seed + 1) + task)
        mse = train_task(net, x, y, steps=steps)
        losses.append(mse / float(y.var()))
        h1, h2 = hidden_features(net, x)
        dead.append(dormant_fraction(h2, tau=0.0))
        rank.append(effective_rank(h2))
    return {"loss": np.array(losses), "dead": np.array(dead), "rank": np.array(rank)}


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
