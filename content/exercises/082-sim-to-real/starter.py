"""Sim to real: randomize the physics, identify it, and adapt."""

import numpy as np

DT = 0.05
HORIZON = 100
U_MAX = 20.0
NOMINAL = np.array([1.0, 0.5])  # [mass, friction] of the simulator
RANGES = {"mass": (0.1, 2.0), "friction": (0.1, 1.5)}
ADAPT_GAINS = np.array([16.0, 8.0])  # (kx, kv): closed-loop x'' = -16 x - 8 x'
# The held-out "real" robots: a 6 x 5 grid, never used for training.
REAL_PARAMS = np.array(
    [[m, b] for m in np.linspace(0.15, 2.0, 6) for b in np.linspace(0.1, 1.5, 5)]
)


def step(state, u, params):
    """One step of a point mass pushed by force u: acceleration (u - friction * v) / mass.

    Args:
        state: `(..., 2)` array of `[position, velocity]`.
        u: force, shape `(...)`, clipped to `[-U_MAX, U_MAX]`.
        params: `(..., 2)` array of `[mass, friction]`, broadcast against `state`.

    Returns:
        The next state, shape `(..., 2)`. Velocity updates first, then
        position uses the new velocity.
    """
    state = np.asarray(state, dtype=float)
    params = np.asarray(params, dtype=float)
    u = np.clip(np.asarray(u, dtype=float), -U_MAX, U_MAX)
    x, v = state[..., 0], state[..., 1]
    mass, friction = params[..., 0], params[..., 1]
    v2 = v + DT * (u - friction * v) / mass
    x2 = x + DT * v2
    return np.stack([x2, v2], axis=-1)


def batch_cost(gains, params):
    """Mean per-step cost of the linear policy u = -kx * x - kv * v.

    Each rollout starts at x = 1, v = 0 and runs `HORIZON` steps. The step
    cost is x**2 + 0.1 * v**2 + 0.002 * u**2.

    Args:
        gains: `(K, 2)` array of `(kx, kv)`.
        params: `(P, 2)` array of `[mass, friction]`.

    Returns:
        A `(K, P)` array: the mean cost of each gain on each physics.
    """
    g = np.asarray(gains, dtype=float)[:, None, :]
    p = np.broadcast_to(np.asarray(params, dtype=float)[None], (g.shape[0], len(params), 2))
    s = np.zeros(p.shape)
    s[..., 0] = 1.0
    total = np.zeros(p.shape[:2])
    for _ in range(HORIZON):
        u = np.clip(-g[..., 0] * s[..., 0] - g[..., 1] * s[..., 1], -U_MAX, U_MAX)
        total += s[..., 0] ** 2 + 0.1 * s[..., 1] ** 2 + 0.002 * u**2
        s = step(s, u, p)
    return total / HORIZON


def collect_transitions(params, n, rng, noise=0.0):
    """Probe a system with random states and forces.

    Args:
        params: `[mass, friction]` of the system.
        n: number of transitions.
        rng: a `numpy.random.Generator`.
        noise: standard deviation of Gaussian noise on the measured next velocity.

    Returns:
        `(S, U, S2)` with shapes `(n, 2)`, `(n,)`, `(n, 2)`.
    """
    S = rng.uniform(-1.0, 1.0, size=(n, 2))
    U = rng.uniform(-10.0, 10.0, size=n)
    S2 = step(S, U, np.asarray(params, dtype=float))
    S2[:, 1] += noise * rng.standard_normal(n)
    return S, U, S2


def adaptive_cost(true_params, params_hat, gains=ADAPT_GAINS):
    """Mean per-step cost of `adaptive_action` on each true physics.

    Args:
        true_params: `(P, 2)` array, the real robots.
        params_hat: `(P, 2)` array, the estimate used for each robot.
        gains: `(kx, kv)`, the closed-loop gains to apply.

    Returns:
        A `(P,)` array of costs, with the same cost and start as `batch_cost`.
    """
    true_params = np.asarray(true_params, dtype=float)
    params_hat = np.asarray(params_hat, dtype=float)
    s = np.zeros((len(true_params), 2))
    s[:, 0] = 1.0
    total = np.zeros(len(true_params))
    for _ in range(HORIZON):
        u = np.clip(adaptive_action(s, params_hat, gains), -U_MAX, U_MAX)
        total += s[:, 0] ** 2 + 0.1 * s[:, 1] ** 2 + 0.002 * u**2
        s = step(s, u, true_params)
    return total / HORIZON


def randomized_params(rng, ranges, n):
    """Draw `n` physics from a distribution, log-uniform in each range.

    Args:
        rng: a `numpy.random.Generator`. Draw from it; never make a new one.
        ranges: `{"mass": (lo, hi), "friction": (lo, hi)}`.
        n: number of draws.

    Returns:
        An `(n, 2)` array of `[mass, friction]`, each row independent.
    """
    raise NotImplementedError("randomized_params")


def sysid_fit(S, U, S2):
    """Recover `(mass, friction)` from transitions by least squares.

    Args:
        S: `(N, 2)` states `[position, velocity]`.
        U: `(N,)` forces.
        S2: `(N, 2)` next states.

    Returns:
        `(mass, friction)` as floats.
    """
    raise NotImplementedError("sysid_fit")


def history_estimate(S, U, S2, window, prior):
    """Estimate physics from only the last `window` transitions.

    Args:
        S, U, S2: the transitions so far, in time order.
        window: how many of the most recent transitions to use.
        prior: `[mass, friction]` to return when the window holds fewer than
            three transitions or does not determine both parameters.

    Returns:
        A length-2 array `[mass, friction]`.
    """
    raise NotImplementedError("history_estimate")


def tune_gain(params, seed, iters=20, pop=60, elite=10):
    """Find the gain `(kx, kv)` with the lowest mean cost over `params` by CEM.

    Args:
        params: `(P, 2)` physics to average the cost over.
        seed: seed for the sampler.
        iters, pop, elite: CEM iterations, population, and elite count.

    Returns:
        The final mean of the sampling distribution, shape `(2,)`.
    """
    raise NotImplementedError("tune_gain")


def adaptive_action(state, params_hat, gains):
    """Force that makes the estimated system follow x'' = -kx * x - kv * v.

    Args:
        state: `(..., 2)` array of `[position, velocity]`.
        params_hat: `(..., 2)` estimated `[mass, friction]`.
        gains: `(kx, kv)`.

    Returns:
        The force, shape `(...)`. Not clipped.
    """
    raise NotImplementedError("adaptive_action")


def report(seed=0):
    """Tune on the simulator and on randomized physics, then score both on the real set."""
    rng = np.random.default_rng(seed)
    g_nom = tune_gain(NOMINAL[None], seed)
    g_rob = tune_gain(randomized_params(rng, RANGES, 128), seed)
    out = {}
    for name, g in (("nominal-tuned", g_nom), ("randomized", g_rob)):
        real = batch_cost(g[None], REAL_PARAMS)[0]
        out[name] = {"sim": float(batch_cost(g[None], NOMINAL[None])[0, 0]),
                     "real_mean": float(real.mean()), "real_worst": float(real.max())}
    est = np.array([sysid_fit(*collect_transitions(p, 200, rng, noise=0.002)) for p in REAL_PARAMS])
    a = adaptive_cost(REAL_PARAMS, est)
    out["adaptive"] = {"sim": float(adaptive_cost(NOMINAL[None], NOMINAL[None])[0]),
                       "real_mean": float(a.mean()), "real_worst": float(a.max())}
    return out


if __name__ == "__main__":
    for name, row in report().items():
        print(f"{name:14s} sim {row['sim']:.4f}   real mean {row['real_mean']:.4f}   real worst {row['real_worst']:.4f}")
