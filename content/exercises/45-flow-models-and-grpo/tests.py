"""Checks for the flow-models-and-GRPO exercise."""

import math

import torch

from rlcheck import almost, check, close, fail, has, shape

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

M, S = 2.0, 0.5


def _exact_velocity(x, t):
    return (t - (1 - t) * S**2) / ((1 - t) ** 2 * S**2 + t**2) * (x - (1 - t) * M) - M


def _marginal(t):
    """Mean and variance of x_t = (1 - t) x_0 + t eps for x_0 ~ N(M, S^2)."""
    return (1 - t) * M, (1 - t) ** 2 * S**2 + t**2


@check("score_from_velocity recovers the exact Gaussian score from the exact velocity")
def test_score_from_velocity(sub):
    score = has(sub, "score_from_velocity")
    x = torch.linspace(-3, 4, 15)
    for t in (0.1, 0.4, 0.75, 0.99):
        mean, var = _marginal(t)
        want = -(x - mean) / var
        close(score(x, t, _exact_velocity(x, t)), want, atol=1e-10, label=f"score at t = {t}")


@check("gaussian_velocity satisfies the continuity equation and the limits v(x, 0) = -x, v(x, 1) = x - m")
def test_gaussian_velocity(sub):
    vel = has(sub, "gaussian_velocity")
    x = torch.linspace(-2, 5, 11)
    close(vel(x, 0.0, M, S), -x, atol=1e-12, label="v at t = 0")
    close(vel(x, 1.0, M, S), x - M, atol=1e-12, label="v at t = 1")

    def density(x, t):
        mean, var = _marginal(t)
        return torch.exp(-0.5 * (x - mean) ** 2 / var) / math.sqrt(2 * math.pi * var)

    # dp/dt + d(p v)/dx = 0, checked by central differences.
    d = 1e-5
    for t in (0.2, 0.5, 0.8):
        dp_dt = (density(x, t + d) - density(x, t - d)) / (2 * d)
        flux = lambda y: density(y, t) * vel(y, t, M, S)
        dflux_dx = (flux(x + d) - flux(x - d)) / (2 * d)
        close(dp_dt + dflux_dx, torch.zeros_like(x), atol=1e-7,
              label=f"continuity residual at t = {t}")


@check("sde_step reduces to an Euler ODE step at a = 0, matches hand values, and caps sigma at t = 1")
def test_sde_step(sub):
    step = has(sub, "sde_step")
    x, v = torch.tensor([0.3, -1.2]), torch.tensor([0.5, 0.1])
    mean, std = step(x, 0.6, 0.1, v, 0.0, 0.9)
    close(mean, x - 0.1 * v, atol=1e-14, label="a = 0 mean")
    almost(float(std), 0.0, 1e-14, label="a = 0 standard deviation")

    # a = 0.7 at t = 0.6: sigma = 0.7 sqrt(1.5), std = sigma sqrt(0.1).
    sigma = 0.7 * math.sqrt(1.5)
    mean, std = step(x, 0.6, 0.1, v, 0.7, 0.9)
    want = x - 0.1 * (v + sigma**2 / 1.2 * (x + 0.4 * v))
    close(mean, want, atol=1e-12, label="a = 0.7 mean")
    almost(float(std), sigma * math.sqrt(0.1), 1e-12, label="a = 0.7 standard deviation")

    # At t = 1 the formula divides by zero; the cap puts t_cap in the denominator.
    mean, std = step(x, 1.0, 0.1, v, 0.7, 0.9)
    sigma = 0.7 * math.sqrt(1 / 0.1)
    almost(float(std), sigma * math.sqrt(0.1), 1e-12, label="capped standard deviation at t = 1")
    close(mean, x - 0.1 * (v + sigma**2 / 2 * x), atol=1e-12, label="capped mean at t = 1")


@check("flow_step_kl matches the Flow-GRPO closed form: 0.004218 on the worked example, and dmu^2 / (2 std^2) in general")
def test_flow_step_kl(sub):
    kl = has(sub, "flow_step_kl")
    step = has(sub, "sde_step")
    got = kl(torch.tensor([[0.2]]), torch.tensor([[0.0]]), 0.6, 0.1, 0.7, 0.9)
    shape(got, (1,), label="flow_step_kl sums the last axis")
    almost(float(got[0]), 0.004217755102040818, 1e-12, label="worked example")
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        t = float(torch.rand(1, generator=g)) * 0.9 + 0.05
        h = float(torch.rand(1, generator=g)) * 0.04 + 0.001
        a = float(torch.rand(1, generator=g)) * 1.5 + 0.1
        x = torch.randn(5, 2, generator=g)
        v1, v2 = torch.randn(5, 2, generator=g), torch.randn(5, 2, generator=g)
        m1, std = step(x, t, h, v1, a, 0.99)
        m2, _ = step(x, t, h, v2, a, 0.99)
        want = ((m1 - m2) ** 2).sum(-1) / (2 * float(std) ** 2)
        close(kl(v1, v2, t, h, a, 0.99), want, rtol=1e-10, atol=1e-14,
              label=f"KL at t = {t:.3f}, h = {h:.4f}, a = {a:.3f}")


@check("sample_sde keeps the data marginal N(2, 0.5^2) for a = 0, 0.3, 0.7, and sqrt 2 on a 250-step grid")
def test_marginals(sub):
    sample = has(sub, "sample_sde")
    grid = torch.linspace(1.0, 0.0, 251).tolist()
    gen = torch.Generator().manual_seed(1)
    x1 = torch.randn(40_000, generator=gen)
    corrs = []
    for a in (0.0, 0.3, 0.7, math.sqrt(2)):
        x0 = sample(_exact_velocity, x1, grid, a, gen)
        shape(x0, (40_000,), label="sample_sde output")
        almost(float(x0.mean()), M, 0.015, label=f"sample mean at a = {a:.3f}")
        almost(float(x0.std()), S, 0.01, label=f"sample standard deviation at a = {a:.3f}")
        corrs.append(float(torch.corrcoef(torch.stack([x1, x0]))[0, 1]))
    if corrs[0] < 0.999:
        fail(f"with a = 0 the sampler is an ODE, so the correlation with the noise should be 1, got {corrs[0]:.4f}")
    if abs(corrs[3]) > 0.02:
        fail(f"with a = sqrt 2 the sample should forget its starting noise, but the correlation is {corrs[3]:.4f}")
    if not corrs[0] > corrs[1] > corrs[2] > corrs[3]:
        fail(f"more noise should mean less memory; correlations for a = 0, 0.3, 0.7, sqrt 2 were "
             f"{[round(c, 3) for c in corrs]}")


@check("group_advantages normalises within each group and shows the illusory advantage")
def test_group_advantages(sub):
    adv = has(sub, "group_advantages")
    r = torch.tensor([[0.2, 0.5, 0.9, 0.4], [0.801, 0.802, 0.800, 0.803]])
    got = adv(r, 0.0)
    shape(got, (2, 4), label="advantages")
    close(got[0], torch.tensor([-1.1766968, 0.0, 1.5689291, -0.3922323]), atol=1e-6,
          label="rewards (0.2, 0.5, 0.9, 0.4), population std")
    close(got[1], torch.tensor([-0.4472136, 0.4472136, -1.3416408, 1.3416408]), atol=1e-6,
          label="rewards (0.801, 0.802, 0.800, 0.803)")
    floored = adv(r, 0.01)
    if float(floored[1].abs().max()) > 0.15:
        fail(f"a floor of 0.01 on the std should shrink the near-equal group below 0.15, got "
             f"{float(floored[1].abs().max()):.3f}")
    close(adv(torch.full((1, 5), 0.7), 1e-8), torch.zeros(1, 5), atol=0,
          label="a group of equal rewards")


@check("flow_logprob scores each stored step as a Gaussian and gives zero KL for the untrained policy")
def test_flow_logprob(sub):
    logprob = has(sub, "flow_logprob")
    sample = has(sub, "sample_sde")
    step = has(sub, "sde_step")
    Policy, ref_v = has(sub, "Policy"), has(sub, "reference_velocity")
    torch.manual_seed(2)
    policy = Policy()
    with torch.no_grad():
        for p in policy.net[-1].parameters():
            p.normal_(0.0, 0.3)
    grid = torch.linspace(1.0, 0.0, 6).tolist()
    c = torch.tensor([0, 1, 2, 3, 0, 1])
    gen = torch.Generator().manual_seed(3)
    with torch.no_grad():
        xs = sample(lambda x, t: policy(x, t, c), torch.randn(6, 2, generator=gen), grid, 0.7, gen,
                    return_path=True)
        shape(xs, (6, 6, 2), label="sample_sde path")
        logp, kl = logprob(policy, xs, c, grid, 0.7)
    shape(logp, (5, 6), label="flow_logprob log-probabilities")
    shape(kl, (5, 6), label="flow_logprob KLs")
    want_lp, want_kl = [], []
    with torch.no_grad():
        for k in range(5):
            t, h = grid[k], grid[k] - grid[k + 1]
            m_th, std = step(xs[k], t, h, policy(xs[k], t, c), 0.7, grid[1])
            m_ref, _ = step(xs[k], t, h, ref_v(xs[k], t), 0.7, grid[1])
            z = (xs[k + 1] - m_th) / std
            want_lp.append((-0.5 * z**2 - math.log(std) - 0.5 * math.log(2 * math.pi)).sum(-1))
            want_kl.append(((m_th - m_ref) ** 2).sum(-1) / (2 * float(std) ** 2))
    close(logp, torch.stack(want_lp), atol=1e-10, label="log-probabilities, summed over both coordinates")
    close(kl, torch.stack(want_kl), atol=1e-10, label="per-step KL to the reference flow")
    with torch.no_grad():
        _, kl0 = logprob(Policy(), xs, c, grid, 0.7)
    close(kl0, torch.zeros(5, 6), atol=1e-14, label="KL of the untrained policy")


@check("flow_grpo_loss clips per-step ratios, adds beta times the mean KL, and has gradient -A / (T n) at ratio 1")
def test_flow_grpo_loss(sub):
    loss_fn = has(sub, "flow_grpo_loss")
    logp_old = torch.zeros(2, 3)
    logp = torch.tensor([[math.log(1.1), math.log(1.5), math.log(1.5)],
                         [0.0, math.log(0.5), math.log(0.5)]])
    adv = torch.tensor([1.0, 1.0, -1.0])
    kl = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    terms = [1.1, 1.2, -1.5, 1.0, 0.5, -0.8]
    almost(float(loss_fn(logp, logp_old, adv, 0.2, kl, 0.5)), -sum(terms) / 6 + 0.5 * 0.35, 1e-12,
           label="hand example")
    g = torch.Generator().manual_seed(4)
    old = torch.randn(3, 4, generator=g)
    a = torch.randn(4, generator=g)
    new = old.clone().requires_grad_(True)
    loss_fn(new, old, a, 0.2, torch.zeros(3, 4), 0.1).backward()
    close(new.grad, (-a / 12).expand(3, 4), atol=1e-12, label="d loss / d logp at ratio 1")


@check("flow_grpo_train lifts the prompt reward from about 0.2 to above 0.85, and a larger beta holds the path KL down")
def test_flow_grpo_train(sub):
    train = has(sub, "flow_grpo_train")
    Policy = has(sub, "Policy")
    torch.manual_seed(5)
    history = train(Policy(), iterations=80, group_size=16, epochs=2, lr=3e-3, clip_eps=0.2,
                    beta=0.01, a=0.7, n_steps=10, generator=torch.Generator().manual_seed(5))
    if len(history) != 80:
        fail(f"flow_grpo_train should return 80 (reward, KL) pairs, got {len(history)}")
    first = sum(h[0] for h in history[:5]) / 5
    last = sum(h[0] for h in history[-5:]) / 5
    if last < 0.85:
        fail(f"the mean reward over the last 5 iterations should exceed 0.85, got {last:.3f} (started at {first:.3f})")
    if abs(history[0][1]) > 1e-12:
        fail(f"the path KL before any update must be 0, got {history[0][1]:.3g}")
    if max(h[1] for h in history) > 15:
        fail(f"with beta = 0.01 the path KL should stay below 15, reached {max(h[1] for h in history):.2f}")
    torch.manual_seed(5)
    held = train(Policy(), iterations=80, group_size=16, epochs=2, lr=3e-3, clip_eps=0.2,
                 beta=0.3, a=0.7, n_steps=10, generator=torch.Generator().manual_seed(5))
    if max(h[1] for h in held) > 1.5:
        fail(f"with beta = 0.3 the path KL should stay below 1.5, reached {max(h[1] for h in held):.2f}")
