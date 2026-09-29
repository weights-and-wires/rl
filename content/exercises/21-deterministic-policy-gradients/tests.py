"""Checks for the deterministic policy gradient exercise."""

import copy

import numpy as np
import torch

from rlcheck import almost, check, close, fail, has, shape, timed

torch.set_num_threads(1)


def _simulated_value(sub, theta, s0, lqr, steps=600):
    """Discounted return of a = theta * s from s0, by rolling the dynamics forward."""
    s, total, discount = s0, 0.0, 1.0
    for _ in range(steps):
        s, r = lqr.step(s, theta * s)
        total += discount * r
        discount *= lqr.gamma
    return total


def _params(module):
    return torch.cat([p.detach().flatten().clone() for p in module.parameters()])


def _agent(sub, seed=0):
    agent = sub.TD3Agent(obs_dim=3, act_dim=1, max_action=2.0, hidden=16, seed=seed)
    # Move the targets away from the online networks, so a mix-up shows.
    gen = torch.Generator().manual_seed(seed + 100)
    with torch.no_grad():
        for net in (agent.actor_target, agent.critic1_target, agent.critic2_target):
            for p in net.parameters():
                p.add_(0.3 * torch.randn(p.shape, generator=gen))
    return agent


def _batch(n=32, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return {
        "s": torch.randn(n, 3, generator=gen),
        "a": 2 * torch.rand(n, 1, generator=gen) - 1,
        "r": torch.randn(n, generator=gen),
        "s2": torch.randn(n, 3, generator=gen),
        "terminated": (torch.rand(n, generator=gen) < 0.25).float(),
    }


@check("lqr_value_coef gives V(s) = -P s^2 that matches a simulated discounted return")
def test_lqr_value(sub):
    lqr_value_coef = has(sub, "lqr_value_coef")
    lqr = sub.ScalarLQR()
    almost(lqr_value_coef(-0.5, lqr), 1.125 / 0.775, tol=1e-12, label="P at theta = -0.5")
    for theta, s0 in [(-0.5, 1.0), (-0.9, -2.0), (0.0, 0.5)]:
        want = _simulated_value(sub, theta, s0, lqr)
        almost(-lqr_value_coef(theta, lqr) * s0**2, want, tol=1e-9,
               label=f"V({s0}) under theta = {theta}")


@check("lqr_q equals one step of reward plus gamma times the value of the next state")
def test_lqr_q(sub):
    lqr_q = has(sub, "lqr_q")
    lqr = sub.ScalarLQR()
    almost(lqr_q(1.0, -0.5, -0.5, lqr), -1.125 / 0.775, tol=1e-12,
           label="Q(1, -0.5) under theta = -0.5, which is V(1) since a = mu(s)")
    for s, a, theta in [(1.0, 0.3, -0.5), (-1.5, 1.0, -0.8)]:
        s2, r = lqr.step(s, a)
        want = r + lqr.gamma * _simulated_value(sub, theta, s2, lqr)
        almost(lqr_q(s, a, theta, lqr), want, tol=1e-9, label=f"Q({s}, {a}) under theta = {theta}")


@check("lqr_dpg, the deterministic policy gradient, matches finite differences of J(theta)")
def test_lqr_dpg(sub):
    lqr_dpg = has(sub, "lqr_dpg")
    lqr = sub.ScalarLQR()
    almost(lqr_dpg(-0.5, lqr, 1.0), -2 * (-0.25 + 0.9 * (1.125 / 0.775) * 0.5) / 0.775, tol=1e-9,
           label="the lesson's worked example, theta = -0.5 and s0 = 1")
    for theta, s0 in [(-0.5, 1.0), (-0.3, 2.0), (-1.2, -1.0)]:
        h = 1e-5
        fd = (_simulated_value(sub, theta + h, s0, lqr) - _simulated_value(sub, theta - h, s0, lqr)) / (2 * h)
        almost(lqr_dpg(theta, lqr, s0), fd, tol=1e-5, label=f"dJ/dtheta at theta = {theta}, s0 = {s0}")
    other = sub.ScalarLQR(c=1.2, b=0.5, lam=0.1, gamma=0.8)
    h = 1e-5
    fd = (_simulated_value(sub, -1.5 + h, 1.0, other) - _simulated_value(sub, -1.5 - h, 1.0, other)) / (2 * h)
    almost(lqr_dpg(-1.5, other, 1.0), fd, tol=1e-5, label="dJ/dtheta on a second system")


@check("smoothed_target_action clips the noise to +-noise_clip, then the action to +-max_action")
def test_smoothed_target_action(sub):
    smooth = has(sub, "smoothed_target_action")
    mu = torch.tensor([[0.0], [1.8], [-1.0], [0.5]])
    eps = torch.tensor([[1.0], [2.0], [-10.0], [-1.0]])
    got = smooth(mu, eps, 0.4, 1.0, 2.0)
    shape(got, (4, 1), label="smoothed_target_action output")
    # Noise 0.4, 0.8, -4 clipped to -1, and -0.4. Then 1.8 + 0.8 = 2.6 is clipped to 2.
    close(got, [[0.4], [2.0], [-2.0], [0.1]], atol=1e-6, label="four hand-worked cases")
    gen = torch.Generator().manual_seed(0)
    mu = 2 * torch.rand(1000, 2, generator=gen) - 1
    eps = 5 * torch.randn(1000, 2, generator=gen)
    got = smooth(mu, eps, 0.2, 0.5, 1.0)
    if (got - mu).abs().max() > 0.5 + 1e-6:
        fail(f"the action moved by {(got - mu).abs().max():.3f}, more than noise_clip = 0.5")
    if got.abs().max() > 1.0 + 1e-6:
        fail(f"an action of {got.abs().max():.3f} left the bounds of +-1")


@check("td3_target is r + gamma (1 - terminated) min(Q1, Q2): 3.475 in the lesson's example")
def test_td3_target(sub):
    td3_target = has(sub, "td3_target")
    got = td3_target(torch.tensor([1.0, 1.0, 0.5]), torch.tensor([0.0, 1.0, 0.0]),
                     torch.tensor([3.0, 3.0, -1.0]), torch.tensor([2.5, 2.5, 4.0]), 0.99)
    shape(got, (3,), label="td3_target output")
    close(got, [1 + 0.99 * 2.5, 1.0, 0.5 + 0.99 * -1.0], atol=1e-6,
          label="the example, a terminal transition, and a case where Q1 is the smaller")


@check("critic_loss regresses both critics to the target from the TARGET networks, with no gradient into them")
def test_critic_loss(sub):
    critic_loss = has(sub, "critic_loss")
    agent = _agent(sub)
    batch = _batch()
    eps = torch.randn(32, 1, generator=torch.Generator().manual_seed(1))
    loss = critic_loss(agent, batch, eps)
    with torch.no_grad():
        noise = (agent.policy_noise * eps).clamp(-agent.noise_clip, agent.noise_clip)
        a2 = (agent.actor_target(batch["s2"]) + noise).clamp(-2.0, 2.0)
        q_next = torch.minimum(agent.critic1_target(batch["s2"], a2), agent.critic2_target(batch["s2"], a2))
        y = batch["r"] + agent.gamma * (1 - batch["terminated"]) * q_next
        want = ((agent.critic1(batch["s"], batch["a"]) - y) ** 2).mean() + (
            (agent.critic2(batch["s"], batch["a"]) - y) ** 2
        ).mean()
    almost(float(loss.detach()), float(want), tol=1e-5, label="the sum of both critics' mean squared errors")
    loss.backward()
    for name in ("actor", "actor_target", "critic1_target", "critic2_target"):
        if any(p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(agent, name).parameters()):
            fail(f"critic_loss sent gradient into agent.{name}. Compute the target under torch.no_grad().")
    for name in ("critic1", "critic2"):
        if all(p.grad is None for p in getattr(agent, name).parameters()):
            fail(f"critic_loss sent no gradient into agent.{name}")


@check("actor_loss is -mean Q1(s, mu(s)), and its gradient is the deterministic policy gradient")
def test_actor_loss(sub):
    actor_loss = has(sub, "actor_loss")
    agent = _agent(sub, seed=1)
    s = _batch(seed=2)["s"]
    loss = actor_loss(agent, s)
    with torch.no_grad():
        want = -agent.critic1(s, agent.actor(s)).mean()
        min_version = -torch.minimum(agent.critic1(s, agent.actor(s)), agent.critic2(s, agent.actor(s))).mean()
    if abs(float(want) - float(min_version)) > 1e-4 and abs(float(loss.detach()) - float(min_version)) < 1e-7:
        fail("actor_loss used min(Q1, Q2). TD3 trains the actor against Q1 alone.")
    almost(float(loss.detach()), float(want), tol=1e-6, label="-mean over the batch of Q1(s, mu(s))")

    params = list(agent.actor.parameters())
    got = torch.autograd.grad(loss, params)
    a = agent.actor(s)
    a_leaf = a.detach().requires_grad_(True)
    dq_da = torch.autograd.grad(agent.critic1(s, a_leaf).sum(), a_leaf)[0]
    # The DPG: -(1/N) sum_i grad_theta mu(s_i)^T grad_a Q1(s_i, a) at a = mu(s_i).
    dpg = torch.autograd.grad(a, params, grad_outputs=-dq_da / s.shape[0])
    for g, w in zip(got, dpg):
        close(g.numpy(), w.numpy(), atol=1e-6, label="actor_loss gradient against the chain rule grad_theta mu * grad_a Q1")


@check("soft_update moves each target parameter a fraction tau toward the source, leaving the source alone")
def test_soft_update(sub):
    soft_update = has(sub, "soft_update")
    torch.manual_seed(0)
    source = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    before_t, before_s = _params(target), _params(source)
    soft_update(target, source, 0.1)
    close(_params(target).numpy(), (0.9 * before_t + 0.1 * before_s).numpy(), atol=1e-7,
          label="target after one update with tau = 0.1")
    close(_params(source).numpy(), before_s.numpy(), atol=0, label="the source network")
    soft_update(target, source, 1.0)
    close(_params(target).numpy(), before_s.numpy(), atol=1e-7, label="target after tau = 1, a hard copy")


@check("td3_update steps the critics every call, but the actor and all three targets only when step % policy_delay == 0")
def test_td3_update_delay(sub):
    td3_update = has(sub, "td3_update")
    agent = _agent(sub, seed=3)
    names = ("critic1", "critic2", "actor", "actor_target", "critic1_target", "critic2_target")
    for step in range(1, 7):
        before = {n: _params(getattr(agent, n)) for n in names}
        batch = _batch(seed=10 + step)
        eps = torch.randn(32, 1, generator=torch.Generator().manual_seed(step))
        td3_update(agent, batch, eps, step)
        moved = {n: not torch.equal(before[n], _params(getattr(agent, n))) for n in names}
        for n in ("critic1", "critic2"):
            if not moved[n]:
                fail(f"at step {step}, agent.{n} didn't change. The critics update on every call.")
        expect = step % agent.policy_delay == 0
        for n in ("actor", "actor_target", "critic1_target", "critic2_target"):
            if moved[n] != expect:
                verb = "should" if expect else "shouldn't"
                fail(f"at step {step} with policy_delay = {agent.policy_delay}, agent.{n} {verb} change")


@check("TD3 with your update learns Pendulum in 8,000 steps: the noise-free actor averages above -800")
def test_td3_pendulum(sub):
    has(sub, "td3_update")
    with timed("8,000 steps of TD3"):
        agent, returns = sub.train_td3(steps=8000, seed=0)
    score = sub.evaluate(agent.actor)
    if not np.isfinite(score) or score < -800:
        fail(
            f"the trained actor averaged {score:.1f} over 10 episodes. A random policy averages about "
            "-1300, and the reference solution about -180. Check the sign of actor_loss, the "
            "(1 - terminated) mask, and that the critic target uses the target networks."
        )
