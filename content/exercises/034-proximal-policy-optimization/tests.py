"""Checks for the proximal policy optimisation exercise."""

import numpy as np
import torch

from rlcheck import check, close, fail, finite, has, shape

torch.set_num_threads(1)


def _t(x):
    return torch.tensor(x, dtype=torch.float64)


@check("clipped_surrogate reproduces the hand-computed table for both signs of the advantage")
def test_clipped_surrogate_table(sub):
    clipped_surrogate = has(sub, "clipped_surrogate")
    ratio = _t([1.3, 0.7, 1.3, 0.7, 1.0, 1.1])
    adv = _t([2.0, -1.0, -1.0, 2.0, 3.0, -2.0])
    logp_old = _t([-1.0, -0.5, -2.0, -1.5, -0.3, -1.2])
    logp_new = (logp_old + torch.log(ratio)).requires_grad_(True)
    got = clipped_surrogate(logp_new, logp_old, adv, 0.2)
    shape(got.detach(), (6,), label="clipped_surrogate output, one value per sample")
    want = np.array([2.4, -0.8, -1.3, 1.4, 3.0, -2.2])
    close(got.detach().numpy(), want, atol=1e-12, label="L^CLIP for each row")
    (grad,) = torch.autograd.grad(got.sum(), logp_new)
    # d/dlogp of r * A is r * A where the unclipped term is the minimum, else 0.
    close(grad.numpy(), np.array([0.0, 0.0, -1.3, 1.4, 3.0, -2.2]), atol=1e-12,
          label="gradient with respect to logp_new (zero exactly on the clipped rows)")


@check("At theta = theta_old, the gradient of the clipped surrogate is the vanilla policy gradient")
def test_clipped_surrogate_gradient_at_old(sub):
    clipped_surrogate = has(sub, "clipped_surrogate")
    torch.manual_seed(0)
    net = torch.nn.Linear(4, 3).double()
    obs = torch.randn(64, 4, dtype=torch.float64)
    actions = torch.randint(3, (64,))
    adv = torch.randn(64, dtype=torch.float64)
    logp = torch.log_softmax(net(obs), -1).gather(-1, actions[:, None]).squeeze(-1)
    surrogate = clipped_surrogate(logp, logp.detach(), adv, 0.2).mean()
    g1 = torch.autograd.grad(surrogate, list(net.parameters()))
    logp = torch.log_softmax(net(obs), -1).gather(-1, actions[:, None]).squeeze(-1)
    g2 = torch.autograd.grad((adv * logp).mean(), list(net.parameters()))
    for a, b in zip(g1, g2):
        close(a.numpy(), b.numpy(), atol=1e-12, label="gradient of L^CLIP against mean(A * grad log pi)")





@check("categorical_entropy is log k for a uniform policy and matches a hand value")
def test_categorical_entropy(sub):
    categorical_entropy = has(sub, "categorical_entropy")
    logits = torch.stack([torch.zeros(4, dtype=torch.float64), torch.log(_t([0.25, 0.75, 1e-300, 1e-300]))])
    got = categorical_entropy(logits)
    shape(got, (2,), label="one entropy per row")
    want = [np.log(4.0), -(0.25 * np.log(0.25) + 0.75 * np.log(0.75))]
    close(got.numpy(), want, atol=1e-10, label="entropy of [1/4] * 4 and of [0.25, 0.75, 0, 0]")
    close(categorical_entropy(_t([[1000.0, 0.0]])).numpy(), [0.0], atol=1e-10,
          label="entropy of a nearly deterministic policy (logits 1000 and 0)")


@check("approx_kl is zero at equal policies, and averages to the exact KL(old || new) over samples from old")
def test_approx_kl(sub):
    approx_kl = has(sub, "approx_kl")
    lp = _t([-0.5, -1.2, -2.0])
    close(float(approx_kl(lp, lp)), 0.0, atol=1e-12, label="approx_kl of identical log-probabilities")
    old = _t([0.5, 0.3, 0.2])
    new = _t([0.2, 0.5, 0.3])
    exact = float((old * torch.log(old / new)).sum())
    g = torch.Generator().manual_seed(0)
    a = torch.multinomial(old, 200000, replacement=True, generator=g)
    lo, ln = torch.log(old)[a], torch.log(new)[a]
    got = float(approx_kl(ln, lo))
    per = (torch.exp(ln - lo) - 1) - (ln - lo)
    if torch.any(per < -1e-12):
        fail("the k3 terms (r - 1) - log r should all be non-negative")
    se = float(per.std()) / np.sqrt(len(a))
    if not abs(got - exact) < 4 * se:
        fail(f"approx_kl over 200000 samples from old is {got:.5f}; the exact KL(old || new) "
             f"is {exact:.5f}, standard error {se:.5f}")


@check("clip_fraction counts the samples whose ratio left [1 - eps, 1 + eps]")
def test_clip_fraction(sub):
    clip_fraction = has(sub, "clip_fraction")
    ratio = _t([1.0, 1.25, 0.85, 0.7, 1.19, 1.3, 0.79, 1.0])
    logp_old = torch.zeros(8, dtype=torch.float64)
    got = clip_fraction(torch.log(ratio), logp_old, 0.2)
    close(float(got), 4 / 8, atol=1e-12, label="4 of 8 ratios outside [0.8, 1.2]")


@check("ppo_loss combines the policy, value, and entropy terms as specified")
def test_ppo_loss(sub):
    ppo_loss = has(sub, "ppo_loss")
    torch.manual_seed(1)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pi = torch.nn.Linear(4, 2)
            self.v = torch.nn.Linear(4, 1)

        def forward(self, obs):
            return self.pi(obs), self.v(obs).squeeze(-1)

    model = Tiny()
    n = 32
    obs = torch.randn(n, 4)
    actions = torch.randint(2, (n,))
    with torch.no_grad():
        logits, values = model(obs)
        logp_now = torch.log_softmax(logits, -1).gather(-1, actions[:, None]).squeeze(-1)
    batch = {
        "obs": obs,
        "actions": actions,
        "logp": logp_now + 0.3 * torch.randn(n),
        "values": values + 0.5 * torch.randn(n),
        "advantages": 2.0 + 3.0 * torch.randn(n),
        "returns": values + torch.randn(n),
    }
    ratio = torch.exp(logp_now - batch["logp"])
    adv = (batch["advantages"] - batch["advantages"].mean()) / (batch["advantages"].std() + 1e-8)
    pg = -torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv).mean()
    lp = torch.log_softmax(logits, -1)
    ent = -(lp.exp() * lp).sum(-1).mean()
    mse = ((values - batch["returns"]) ** 2).mean()

    got = ppo_loss(model, batch, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01)
    close(float(got), float(pg + 0.5 * mse - 0.01 * ent), rtol=1e-5, atol=1e-6,
          label="policy loss + 0.5 * value loss - 0.01 * entropy")
    if not got.requires_grad:
        fail("ppo_loss must return a tensor that can be backpropagated")


def _epochs_on_fixed_batch(sub, target_kl, epochs=10, lr=0.05):
    """Run ppo_epochs on one seeded batch with a large learning rate.

    Returns (kls, epochs_run, independent_kl), where independent_kl is the full-batch
    approx_kl of the updated model against the stored logp, computed here and not by
    the submission.
    """
    ActorCritic, VecCartPole = has(sub, "ActorCritic"), has(sub, "VecCartPole")
    collect_rollout, ppo_epochs = has(sub, "collect_rollout"), has(sub, "ppo_epochs")
    torch.manual_seed(0)
    model = ActorCritic()
    batch = collect_rollout(VecCartPole(8, seed=0), model, 128)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    torch.manual_seed(1)
    kls, ran = ppo_epochs(model, optimizer, batch, epochs=epochs, minibatches=4, target_kl=target_kl)
    with torch.no_grad():
        logits, _ = model(batch["obs"])
        logp = torch.log_softmax(logits, -1).gather(-1, batch["actions"][:, None]).squeeze(-1)
        log_ratio = logp - batch["logp"]
        independent = float((torch.exp(log_ratio) - 1 - log_ratio).mean())
    return list(kls), ran, independent


@check("ppo_epochs with target_kl=None runs every epoch and reports the full-batch approx_kl against the stored logp_old")
def test_ppo_epochs_reports_kl(sub):
    kls, ran, independent = _epochs_on_fixed_batch(sub, None)
    if ran != 10 or len(kls) != 10:
        fail(f"with target_kl=None and epochs=10, expected 10 epochs run and 10 KL values; "
             f"got epochs_run={ran} and {len(kls)} values")
    # Wrong implementation this catches: computing the KL against log-probs recomputed
    # from the current network instead of the stored batch["logp"]. Every ratio is
    # then 1, the KL is always 0, and no target_kl ever fires.
    close(kls[-1], independent, atol=1e-6,
          label="last per-epoch KL against an independent full-batch approx_kl of the updated model")


@check("ppo_epochs stops after the first epoch whose full-batch approx_kl exceeds target_kl")
def test_ppo_epochs_early_stop(sub):
    target = 0.015
    ref, _, _ = _epochs_on_fixed_batch(sub, None)
    over = [i for i, k in enumerate(ref) if k > target]
    if not over or over[0] == 0:
        fail(f"on this seeded batch (learning rate 0.05, 10 epochs) the KL should first pass {target} "
             f"after epoch 2 or later; the per-epoch KLs from target_kl=None were {np.round(ref, 4).tolist()}")
    want = over[0] + 1
    # Wrong implementations this catches: a KL that is always 0 (recomputed log-probs),
    # and a KL that is checked only after all epochs have run.
    kls, ran, _ = _epochs_on_fixed_batch(sub, target)
    if ran != want or len(kls) != want:
        fail(f"with target_kl={target}, expected the update to stop after epoch {want} "
             f"(the first KL above {target}; per-epoch KLs without a stop were {np.round(ref, 4).tolist()}); "
             f"got epochs_run={ran} and {len(kls)} KL values: {np.round(kls, 4).tolist()}")
    close(np.array(kls), np.array(ref[:want]), atol=1e-9,
          label="per-epoch KL values up to the stop, which must equal the unstopped run's")


@check("train_ppo learns to balance CartPole: the greedy policy averages at least 150 steps")
def test_train_ppo(sub):
    train_ppo = has(sub, "train_ppo")
    evaluate = has(sub, "evaluate")
    model, history = train_ppo(seed=0)
    if len(history) != 20:
        fail(f"train_ppo should return one history entry per iteration (20), got {len(history)}")
    finite(np.asarray(history, dtype=float), label="history")
    score = evaluate(model)
    if not score >= 150.0:
        fail(f"the greedy policy balanced the pole for {score:.1f} steps on average over 10 "
             f"episodes; expected at least 150. The training history was "
             f"{np.round(history, 1).tolist()}")
