"""Checks for the preference-tuning-for-diffusion exercise."""

import math

import torch

from rlcheck import almost, check, close, fail, finite_difference, has, shape

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

# The four-step linear-Gaussian chain from the lesson, in sampling order.
A_THETA = [0.90, 0.80, 0.85, 0.95]
B_THETA = [0.10, 0.20, 0.00, 0.30]
A_REF = [0.90, 0.70, 0.85, 1.00]
B_REF = [0.00, 0.20, 0.10, 0.00]
S = [0.50, 0.40, 0.30, 0.20]


def _perturbed(sub, w=0.08, b=-0.25):
    model = has(sub, "ToyDenoiser")()
    with torch.no_grad():
        model.w += w * torch.linspace(-1, 1, len(model.w))
        model.b += b
    return model


@check("gaussian_kl gives 0.1231 for N(0.5, 0.2^2) against N(0.4, 0.25^2), and 0 for equal Gaussians")
def test_gaussian_kl(sub):
    kl = has(sub, "gaussian_kl")
    want = math.log(0.25 / 0.2) + (0.2**2 + 0.1**2) / (2 * 0.25**2) - 0.5
    almost(float(kl(0.5, 0.2, 0.4, 0.25)), want, 1e-12, label="KL(N(0.5, 0.2^2) || N(0.4, 0.25^2))")
    almost(float(kl(0.3, 0.7, 0.3, 0.7)), 0.0, 1e-12, label="KL of a Gaussian with itself")
    g = torch.Generator().manual_seed(0)
    m1, m2 = torch.randn(30, generator=g), torch.randn(30, generator=g)
    s1, s2 = torch.rand(30, generator=g) + 0.2, torch.rand(30, generator=g) + 0.2
    want = torch.distributions.kl_divergence(torch.distributions.Normal(m1, s1),
                                             torch.distributions.Normal(m2, s2))
    close(kl(m1, s1, m2, s2), want, atol=1e-12, label="gaussian_kl on random inputs")


@check("chain_path_kl sums the per-step KLs to 1.16826 on the four-step chain, and 0 when theta = ref")
def test_chain_path_kl(sub):
    path = has(sub, "chain_path_kl")
    almost(float(path(A_THETA, B_THETA, A_REF, B_REF, S)), 1.1682551805555554, 1e-9,
           label="path KL of the lesson's chain")
    almost(float(path(A_REF, B_REF, A_REF, B_REF, S)), 0.0, 1e-12, label="path KL with theta = ref")


@check("chain_marginal_kl gives 0.05419 for the final samples, about 1/21 of the path KL")
def test_chain_marginal_kl(sub):
    marginal = has(sub, "chain_marginal_kl")
    path = has(sub, "chain_path_kl")
    got = float(marginal(A_THETA, B_THETA, A_REF, B_REF, S))
    almost(got, 0.05418571649466319, 1e-9, label="KL between the two final Gaussians")
    if got >= float(path(A_THETA, B_THETA, A_REF, B_REF, S)):
        fail("the final-sample KL must be smaller than the path KL")
    almost(float(marginal(A_REF, B_REF, A_REF, B_REF, S)), 0.0, 1e-12, label="marginal KL with theta = ref")


@check("chain_path_kl_mc averages summed log-ratios and lands within 3 standard errors of 1.16826")
def test_chain_path_kl_mc(sub):
    mc = has(sub, "chain_path_kl_mc")
    gen = torch.Generator().manual_seed(1)
    est, se = mc(A_THETA, B_THETA, A_REF, B_REF, S, 200_000, gen)
    est, se = float(est), float(se)
    if not 0.002 < se < 0.006:
        fail(f"with 200,000 samples the standard error should be about 0.0035, got {se:.5f}")
    if abs(est - 1.1682551805555554) > 3 * se:
        fail(f"Monte Carlo path KL {est:.4f} is more than 3 standard errors ({3 * se:.4f}) from 1.16826")


@check("ddpm_step_kl is (mu_theta - mu_ref)^2 / (2 sigma_t^2) at each stored state, and 0 for the pretrained model")
def test_ddpm_step_kl(sub):
    step_kl = has(sub, "ddpm_step_kl")
    sample = has(sub, "sample")
    T, BETAS, ALPHAS, ALPHA_BARS, SIGMAS = (has(sub, n) for n in
                                            ("T", "BETAS", "ALPHAS", "ALPHA_BARS", "SIGMAS"))
    eps_pretrained = has(sub, "eps_pretrained")
    model = _perturbed(sub)
    xs = sample(model, 50, torch.Generator().manual_seed(2))
    with torch.no_grad():
        got = step_kl(model, xs)
    shape(got, (T, 50), label="ddpm_step_kl output")
    want = []
    for k in range(T):
        t = T - k
        x = xs[k]
        coef = BETAS[t - 1] / torch.sqrt(1 - ALPHA_BARS[t - 1]) / torch.sqrt(ALPHAS[t - 1])
        dmu = -coef * (model.eps(x, t) - eps_pretrained(x, t))
        want.append(dmu**2 / (2 * SIGMAS[t - 1] ** 2))
    close(got, torch.stack(want).detach(), atol=1e-12, label="per-step KL against the mean gap")
    fresh = has(sub, "ToyDenoiser")()
    with torch.no_grad():
        close(step_kl(fresh, xs), torch.zeros(T, 50), atol=1e-14, label="KL of the untrained model")


@check("Summed step KLs and summed log-ratios estimate the same path KL on the toy model")
def test_chain_rule_on_toy(sub):
    step_kl = has(sub, "ddpm_step_kl")
    sample = has(sub, "sample")
    ddpm_mean = has(sub, "ddpm_mean")
    T, SIGMAS = has(sub, "T"), has(sub, "SIGMAS")
    eps_pretrained = has(sub, "eps_pretrained")
    model = _perturbed(sub)
    xs = sample(model, 40_000, torch.Generator().manual_seed(3))
    with torch.no_grad():
        kl_sum = step_kl(model, xs).sum(0)
        log_ratio = torch.zeros(xs.shape[1])
        for k in range(T):
            t = T - k
            m_theta = ddpm_mean(xs[k], model.eps(xs[k], t), t)
            m_ref = ddpm_mean(xs[k], eps_pretrained(xs[k], t), t)
            log_ratio += ((xs[k + 1] - m_ref) ** 2 - (xs[k + 1] - m_theta) ** 2) / (2 * SIGMAS[t - 1] ** 2)
    a, b = float(kl_sum.mean()), float(log_ratio.mean())
    se = math.sqrt(float(kl_sum.var() + log_ratio.var()) / xs.shape[1])
    if a < 0.05:
        fail(f"the perturbed model's path KL should be clearly positive, got {a:.4f}")
    if abs(a - b) > 4 * se:
        fail(f"mean summed step KL {a:.4f} and mean summed log-ratio {b:.4f} differ by more "
             f"than 4 standard errors ({4 * se:.4f})")


@check("denoising_errors noises x_0 once and scores the model and the pretrained network on the same x_t")
def test_denoising_errors(sub):
    errors = has(sub, "denoising_errors")
    ALPHA_BARS = has(sub, "ALPHA_BARS")
    eps_pretrained = has(sub, "eps_pretrained")
    model = _perturbed(sub)
    g = torch.Generator().manual_seed(4)
    x0 = torch.randn(12, generator=g)
    t = torch.randint(1, 11, (12,), generator=g)
    eps = torch.randn(12, generator=g)
    err, err_ref = errors(model, x0, t, eps)
    shape(err.detach(), (12,), label="model errors")
    abar = ALPHA_BARS[t - 1]
    x_t = torch.sqrt(abar) * x0 + torch.sqrt(1 - abar) * eps
    close(err.detach(), (eps - model.eps(x_t, t)).detach() ** 2, atol=1e-12, label="model error")
    close(err_ref.detach(), (eps - eps_pretrained(x_t, t)) ** 2, atol=1e-12, label="reference error")
    if not err.requires_grad:
        fail("the model's error must carry a gradient to the model's parameters")


@check("diffusion_dpo_loss is log 2 at the reference, 0.5130 on the worked example, and its gradient matches finite differences")
def test_diffusion_dpo_loss(sub):
    loss = has(sub, "diffusion_dpo_loss")
    e = torch.tensor([0.3, 0.7, 1.1])
    almost(float(loss(e, e, e + 0.2, e + 0.2, 5.0)), math.log(2), 1e-12, label="loss when policy = reference")
    almost(float(loss(torch.tensor([0.30]), torch.tensor([0.50]), torch.tensor([0.60]),
                      torch.tensor([0.40]), 1.0)),
           -math.log(1 / (1 + math.exp(-0.4))), 1e-10, label="worked example")
    g = torch.Generator().manual_seed(5)
    x0 = torch.rand(4, 4, generator=g).reshape(-1)

    def f(x):
        x = torch.as_tensor(x)
        return float(loss(x[0:4], x[4:8], x[8:12], x[12:16], 2.0))

    fd = finite_difference(f, x0.numpy().copy())
    x = x0.clone().requires_grad_(True)
    loss(x[0:4], x[4:8], x[8:12], x[12:16], 2.0).backward()
    close(x.grad, torch.as_tensor(fd), rtol=1e-5, atol=1e-8, label="gradient against finite differences")
    if not (x.grad[0:4] > 0).all():
        fail("raising the policy's error on the winner must raise the loss")


@check("diffusion_dpo_train on reference pairs lifts the true reward from about -6.3 to above -1")
def test_diffusion_dpo_train(sub):
    train = has(sub, "diffusion_dpo_train")
    sample = has(sub, "sample")
    true_reward = has(sub, "true_reward")
    ToyDenoiser = has(sub, "ToyDenoiser")
    gen = torch.Generator().manual_seed(6)
    ref = ToyDenoiser()
    x_a, x_b = sample(ref, 2000, gen)[-1], sample(ref, 2000, gen)[-1]
    a_wins = true_reward(x_a) >= true_reward(x_b)
    x0_w, x0_l = torch.where(a_wins, x_a, x_b), torch.where(a_wins, x_b, x_a)
    model = ToyDenoiser()
    losses = train(model, x0_w, x0_l, beta=5.0, steps=150, batch_size=256, lr=0.05, generator=gen)
    if len(losses) != 150:
        fail(f"diffusion_dpo_train should return 150 losses, got {len(losses)}")
    almost(losses[0], math.log(2), 1e-9, label="the first loss, before any update")
    after = float(true_reward(sample(model, 4000, torch.Generator().manual_seed(60))[-1]).mean())
    if after < -1.0:
        fail(f"after Diffusion-DPO the mean true reward should exceed -1, got {after:.3f}")


@check("Reward backprop on the proxy hacks it without a KL penalty, and a path-KL penalty prevents that")
def test_reward_hacking(sub):
    train = has(sub, "reward_backprop_train")
    proxy, true_reward = has(sub, "proxy_reward"), has(sub, "true_reward")
    ToyDenoiser = has(sub, "ToyDenoiser")
    x0s, kls = train(ToyDenoiser(), proxy, 0.0, steps=60, batch_size=256, lr=0.02,
                     generator=torch.Generator().manual_seed(7))
    shape(x0s, (60, 256), label="the returned samples")
    p = proxy(x0s).mean(1)
    tr = true_reward(x0s).mean(1)
    peak = int(torch.argmax(tr))
    if float(p[-5:].mean()) < float(p[:5].mean()) + 3.0:
        fail(f"the proxy reward should climb by more than 3, went {float(p[:5].mean()):.2f} -> {float(p[-5:].mean()):.2f}")
    if peak > 45 or float(tr[-5:].mean()) > float(tr[peak]) - 5.0:
        fail(f"without a KL penalty the true reward should peak early and then fall by more than 5; "
             f"it peaked at step {peak} at {float(tr[peak]):.2f} and ended at {float(tr[-5:].mean()):.2f}")
    x0s_kl, kls_kl = train(ToyDenoiser(), proxy, 0.3, steps=60, batch_size=256, lr=0.02,
                           generator=torch.Generator().manual_seed(8))
    final_kl = float(true_reward(x0s_kl[-5:]).mean())
    if final_kl < -1.5:
        fail(f"with kl_coef = 0.3 the final true reward should stay above -1.5, got {final_kl:.2f}")
    if max(kls_kl) > 3.0:
        fail(f"with kl_coef = 0.3 the path KL should stay below 3, reached {max(kls_kl):.2f}")
