"""Checks for the learning-from-pixels exercise."""

import numpy as np
import torch

from rlcheck import check, close, equal, fail, finite_difference, has, shape

torch.set_num_threads(1)


def _huber(u):
    return np.where(np.abs(u) <= 1, 0.5 * u ** 2, np.abs(u) - 0.5)


@check("preprocess takes the RGB maximum first, then luminance, scaled to [0, 1]")
def test_preprocess(sub):
    preprocess = has(sub, "preprocess")
    red = np.zeros((2, 3, 3), dtype=np.uint8)
    green = np.zeros((2, 3, 3), dtype=np.uint8)
    red[..., 0] = 255
    green[..., 1] = 255
    got = np.asarray(preprocess(red, green), dtype=float)
    shape(got, (2, 3), label="preprocess output")
    # The RGB maximum is (255, 255, 0), whose luminance is 0.299 + 0.587 = 0.886.
    # Named wrong implementation: gray each frame, then take the maximum.
    # That gives max(0.299, 0.587) = 0.587 and fails this check.
    close(got, np.full((2, 3), 0.886), atol=1e-9, label="red and green frames")
    white = np.full((2, 3, 3), 255, dtype=np.uint8)
    close(preprocess(white, np.zeros_like(white)), np.ones((2, 3)), atol=1e-9,
          label="white and black frames")


@check("frame_stack returns the last k frames, oldest first, and pads a short history")
def test_frame_stack(sub):
    frame_stack = has(sub, "frame_stack")
    frames = [np.full((2, 2), float(i)) for i in range(5)]
    got = np.asarray(frame_stack(frames, 3))
    shape(got, (3, 2, 2), label="stack of 3")
    equal(got[:, 0, 0].tolist(), [2.0, 3.0, 4.0], label="frame values in the stack")
    short = np.asarray(frame_stack(frames[:2], 4))
    equal(short[:, 0, 0].tolist(), [0.0, 0.0, 0.0, 1.0], label="stack from two frames")


@check("noise_f is sign(x) sqrt|x|, and factorized_noise is a rank-one outer product")
def test_factorized_noise(sub):
    noise_f = has(sub, "noise_f")
    factorized_noise = has(sub, "factorized_noise")
    close(noise_f(np.array([-4.0, 0.0, 0.25, 9.0])), [-2.0, 0.0, 0.5, 3.0], atol=1e-12,
          label="noise_f")
    eps_w, eps_b = factorized_noise(4, 3, np.random.default_rng(7))
    eps_w, eps_b = np.asarray(eps_w), np.asarray(eps_b)
    shape(eps_w, (3, 4), label="eps_w")
    shape(eps_b, (3,), label="eps_b")
    rng = np.random.default_rng(7)
    z_in = rng.standard_normal(4)
    z_out = rng.standard_normal(3)
    f_in = np.sign(z_in) * np.sqrt(np.abs(z_in))
    f_out = np.sign(z_out) * np.sqrt(np.abs(z_out))
    close(eps_w, np.outer(f_out, f_in), atol=1e-12, label="eps_w")
    close(eps_b, f_out, atol=1e-12, label="eps_b")
    rank = np.linalg.matrix_rank(eps_w)
    if rank != 1:
        fail(f"eps_w should have rank 1 (factorized), got rank {rank}")
    again, _ = factorized_noise(4, 3, np.random.default_rng(7))
    close(again, eps_w, atol=0, label="the same seed gives the same noise")


@check("NoisyLinear computes (mu + sigma eps) x + (mu_b + sigma_b eps_b), and the mean alone when noise is off")
def test_noisy_forward(sub):
    NoisyLinear = has(sub, "NoisyLinear")
    layer = NoisyLinear(5, 3, seed=1)
    layer.reset_noise(np.random.default_rng(3))
    x = torch.as_tensor(np.random.default_rng(0).normal(size=(4, 5)), dtype=torch.float32)
    w = layer.weight_mu + layer.weight_sigma * layer.eps_w
    b = layer.bias_mu + layer.bias_sigma * layer.eps_b
    want = (x @ w.T + b).detach().numpy()
    close(layer(x).detach().numpy(), want, atol=1e-5, label="noisy output")
    if np.allclose(layer.eps_w.numpy(), 0):
        fail("reset_noise left eps_w at zero")
    layer.noisy = False
    mean_want = (x @ layer.weight_mu.T + layer.bias_mu).detach().numpy()
    close(layer(x).detach().numpy(), mean_want, atol=1e-5, label="noise-free output")
    layer.noisy = True
    layer.reset_noise(np.random.default_rng(3))
    close(layer(x).detach().numpy(), want, atol=1e-5, label="same noise seed, same output")


@check("NoisyLinear gradients for mu and sigma match finite differences")
def test_noisy_gradient(sub):
    NoisyLinear = has(sub, "NoisyLinear")
    layer = NoisyLinear(4, 3, seed=2).double()
    layer.reset_noise(np.random.default_rng(5))
    rng = np.random.default_rng(1)
    x = torch.as_tensor(rng.normal(size=(6, 4)), dtype=torch.float64)
    g = torch.as_tensor(rng.normal(size=(6, 3)), dtype=torch.float64)
    # Loss: the sum over the batch of g * output. Its gradient for weight_sigma
    # is (g^T x) * eps_w, which is the gradient for weight_mu times the noise.
    (g * layer(x)).sum().backward()
    for name in ("weight_mu", "weight_sigma", "bias_mu", "bias_sigma"):
        param = getattr(layer, name)
        if param.grad is None:
            fail(f"no gradient reached {name}")
        original = param.detach().clone()

        def f(v, param=param, original=original):
            with torch.no_grad():
                param.copy_(torch.as_tensor(v))
                value = float((g * layer(x)).sum())
                param.copy_(original)
            return value

        numeric = finite_difference(f, original.numpy().copy(), eps=1e-6)
        close(param.grad.numpy(), numeric, rtol=1e-5, atol=1e-7, label=f"gradient of {name}")
    expected = (g.T @ x).detach().numpy() * layer.eps_w.numpy()
    close(layer.weight_sigma.grad.numpy(), expected, atol=1e-10, label="closed form for weight_sigma")


@check("random_shift translates each image with replicated edges, and a zero pad changes nothing")
def test_random_shift(sub):
    random_shift = has(sub, "random_shift")
    rng = np.random.default_rng(0)
    imgs = rng.normal(size=(6, 2, 5, 4)).astype(np.float32)
    zero = random_shift(imgs, 0, np.random.default_rng(1))
    equal(np.asarray(zero).tolist(), imgs.tolist(), label="pad=0 output")
    out = np.asarray(random_shift(imgs, 2, np.random.default_rng(1)))
    shape(out, imgs.shape, label="random_shift output")
    equal(str(out.dtype), str(imgs.dtype), label="dtype")
    padded = np.pad(imgs, ((0, 0), (0, 0), (2, 2), (2, 2)), mode="edge")
    for i in range(len(imgs)):
        crops = [padded[i, :, dy:dy + 5, dx:dx + 4] for dy in range(5) for dx in range(5)]
        if not any(np.array_equal(out[i], c) for c in crops):
            fail(f"image {i} is not a crop of the edge-padded image (replicate padding, window offsets 0 to 4)")
    again = np.asarray(random_shift(imgs, 2, np.random.default_rng(1)))
    equal(again.tolist(), out.tolist(), label="same seed, same shifts")
    other = np.asarray(random_shift(imgs, 2, np.random.default_rng(2)))
    if np.array_equal(other, out):
        fail("different seeds gave identical shifts for all six images")
    # With the documented draw, the offsets are rng.integers(0, 5, size=(6, 2)).
    shifts = np.random.default_rng(1).integers(0, 5, size=(6, 2))
    for i, (dy, dx) in enumerate(shifts):
        close(out[i], padded[i, :, dy:dy + 5, dx:dx + 4], atol=0,
              label=f"image {i} at offset ({dy}, {dx})")


def _setup_q(seed=0):
    rng = np.random.default_rng(seed)
    batch = {
        "obs": rng.normal(size=(4, 1, 3, 3)).astype(np.float32),
        "next_obs": rng.normal(size=(4, 1, 3, 3)).astype(np.float32),
        "actions": np.array([0, 1, 1, 0]),
        "rewards": np.array([1.0, 0.0, -1.0, 2.0], dtype=np.float32),
        "terminated": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    }
    wq = torch.as_tensor(rng.normal(size=(9, 2)), dtype=torch.float32).requires_grad_(True)
    wt = torch.as_tensor(rng.normal(size=(9, 2)), dtype=torch.float32).requires_grad_(True)
    return batch, wq, wt


@check("augmented_td_loss with pad 0 equals the plain Huber TD loss, with no gradient into the target")
def test_td_loss_plain(sub):
    augmented_td_loss = has(sub, "augmented_td_loss")
    batch, wq, wt = _setup_q()
    gamma = 0.9
    loss = augmented_td_loss(lambda x: x.flatten(1) @ wq, lambda x: x.flatten(1) @ wt,
                             batch, gamma, 0, np.random.default_rng(0))
    q = batch["obs"].reshape(4, -1) @ wq.detach().numpy()
    nxt = (batch["next_obs"].reshape(4, -1) @ wt.detach().numpy()).max(axis=1)
    y = batch["rewards"] + gamma * (1 - batch["terminated"]) * nxt
    u = q[np.arange(4), batch["actions"]] - y
    close(float(loss.detach()), float(_huber(u).mean()), atol=1e-5, label="loss value")
    loss.backward()
    if wt.grad is not None and float(wt.grad.abs().sum()) > 0:
        fail("the target network received a gradient; compute the target under torch.no_grad()")
    base = wq.detach().numpy().astype(np.float64).copy()
    obs_flat = batch["obs"].reshape(4, -1).astype(np.float64)

    def f(v):
        uv = (obs_flat @ v)[np.arange(4), batch["actions"]] - y
        return float(_huber(uv).mean())

    close(wq.grad.numpy(), finite_difference(f, base, eps=1e-6), rtol=1e-3, atol=1e-5,
          label="gradient of the loss with respect to the online weights")


@check("augmented_td_loss averages the target over k shifts and Q over m shifts, in that draw order")
def test_td_loss_augmented(sub):
    augmented_td_loss = has(sub, "augmented_td_loss")
    batch, wq, wt = _setup_q(1)
    gamma, pad, k, m = 0.9, 1, 2, 3

    def shift(imgs, rng):
        padded = np.pad(imgs, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="edge")
        s = rng.integers(0, 2 * pad + 1, size=(len(imgs), 2))
        return np.stack([padded[i, :, dy:dy + 3, dx:dx + 3] for i, (dy, dx) in enumerate(s)])

    rng = np.random.default_rng(4)
    best = [(shift(batch["next_obs"], rng).reshape(4, -1) @ wt.detach().numpy()).max(axis=1)
            for _ in range(k)]
    y = batch["rewards"] + gamma * (1 - batch["terminated"]) * np.mean(best, axis=0)
    terms = []
    for _ in range(m):
        q = shift(batch["obs"], rng).reshape(4, -1) @ wq.detach().numpy()
        terms.append(_huber(q[np.arange(4), batch["actions"]] - y).mean())
    want = float(np.mean(terms))
    got = float(augmented_td_loss(lambda x: x.flatten(1) @ wq, lambda x: x.flatten(1) @ wt,
                                  batch, gamma, pad, np.random.default_rng(4),
                                  k_target=k, m_q=m))
    # Named wrong implementation: build the target from the un-shifted next_obs.
    # It gets a different target, so this value differs.
    close(got, want, atol=1e-5, label=f"loss with k={k}, m={m}")
    plain = float(augmented_td_loss(lambda x: x.flatten(1) @ wq, lambda x: x.flatten(1) @ wt,
                                    batch, gamma, 0, np.random.default_rng(4),
                                    k_target=k, m_q=m))
    if abs(plain - got) < 1e-6:
        fail("shifting by pad=1 left the loss equal to the pad=0 loss; the shifts are not applied")


@check("a noisy DQN trained on pixel Catch, with and without shifts, beats a random policy's exact 0.20")
def test_catch(sub):
    train_catch, evaluate_catch = has(sub, "train_catch"), has(sub, "evaluate_catch")
    # A random paddle is independent of the ball's column, which is uniform
    # over 5 columns, so a random policy catches exactly 1/5 of the balls.
    random_rate = 1.0 / 5.0
    for augment in (False, True):
        for seed in (0, 1):
            rate = evaluate_catch(train_catch(seed=seed, augment=augment), episodes=300)
            if rate < 0.4:
                fail(f"augment={augment}, seed={seed}: caught {rate:.2f} of 300 balls; "
                     f"expected at least 0.40 (a random policy catches {random_rate:.2f})")
