"""Checks for the decision transformer exercise."""

import numpy as np
import torch

from rlcheck import almost, check, close, equal, fail, has

torch.set_num_threads(1)


def _tril(n):
    return np.tril(np.ones((n, n), dtype=bool))


@check("returns_to_go sums the reward at t and everything after it, with and without a discount")
def test_returns_to_go(sub):
    rtg = has(sub, "returns_to_go")
    # WRONG IMPLEMENTATION CAUGHT HERE: returns-to-go that are off by one, such as
    # np.cumsum(rewards[::-1])[::-1] shifted by one or a sum that includes the previous reward.
    # Then rtg[0] is not the whole return and rtg[-1] is not the last reward.
    close(rtg([1.0, 0.0, 2.0]), [3.0, 2.0, 2.0], atol=1e-12, label="returns_to_go([1, 0, 2])")
    close(rtg([-1.0, -1.0, -1.0]), [-3.0, -2.0, -1.0], atol=1e-12, label="returns_to_go([-1, -1, -1])")
    # gamma = 0.5: G_2 = 2, G_1 = 0 + 0.5 * 2 = 1, G_0 = 1 + 0.5 * 1 = 1.5.
    close(rtg([1.0, 0.0, 2.0], gamma=0.5), [1.5, 1.0, 2.0], atol=1e-12, label="returns_to_go([1, 0, 2], gamma=0.5)")
    close(rtg([4.0]), [4.0], atol=1e-12, label="returns_to_go([4])")


@check("build_tokens interleaves (G_t, s_t, a_t) and repeats each timestep three times")
def test_build_tokens(sub):
    build = has(sub, "build_tokens")
    kinds, steps, values = build([3.0, 2.0], [5, 6], [1, 0])
    equal(np.asarray(kinds), [0, 1, 2, 0, 1, 2], label="token kinds (0 = G, 1 = s, 2 = a)")
    equal(np.asarray(steps), [0, 0, 0, 1, 1, 1], label="timesteps")
    close(values, [3.0, 5.0, 1.0, 2.0, 6.0, 0.0], atol=1e-12, label="token values")
    # At test time the last action is not chosen yet: the sequence ends at s_{T-1}.
    kinds, steps, values = build([3.0, 2.0], [5, 6], [1])
    equal(np.asarray(kinds), [0, 1, 2, 0, 1], label="token kinds with the last action unknown")
    equal(np.asarray(steps), [0, 0, 0, 1, 1], label="timesteps with the last action unknown")
    close(values, [3.0, 5.0, 1.0, 2.0, 6.0], atol=1e-12, label="token values with the last action unknown")


@check("causal_mask lets token i see tokens j <= i, so s_t sees G_t and a_{t-1} but not a_t")
def test_causal_mask(sub):
    causal_mask = has(sub, "causal_mask")
    got = np.asarray(causal_mask(4), dtype=bool)
    equal(got, _tril(4), label="causal_mask(4)")
    # Two steps: tokens are G0 s0 a0 G1 s1 a1 at indexes 0 to 5.
    m = np.asarray(causal_mask(6), dtype=bool)
    equal(m[4], [True, True, True, True, True, False], label="row of s_1: sees G0 s0 a0 G1 s1, not a1")
    equal(m[1], [True, True, False, False, False, False], label="row of s_0: sees G0 s0, not a0")


@check("masked_attention gives zero weight to blocked tokens, and matches plain softmax attention with no mask")
def test_masked_attention(sub):
    attn = has(sub, "masked_attention")
    rng = np.random.default_rng(0)
    n, d = 7, 4
    x = rng.normal(size=(n, d))
    Wq, Wk, Wv = (rng.normal(size=(d, d)) for _ in range(3))
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    s = q @ k.T / np.sqrt(d)
    w = np.exp(s - s.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    close(attn(x, np.ones((n, n), dtype=bool), Wq, Wk, Wv), w @ v, atol=1e-10, label="attention with no mask")
    out = attn(x, _tril(n), Wq, Wk, Wv)
    close(out[0], v[0], atol=1e-10, label="token 0 sees only itself, so its output is its own value")
    x2 = x.copy()
    x2[4:] = rng.normal(size=(n - 4, d))
    out2 = attn(x2, _tril(n), Wq, Wk, Wv)
    close(out2[:4], out[:4], atol=1e-10, label="outputs 0 to 3 after changing tokens 4 to 6")


@check("with causal_mask, changing future tokens never changes an earlier output, and the s_t output ignores a_t")
def test_no_leak(sub):
    causal_mask, attn = has(sub, "causal_mask"), has(sub, "masked_attention")
    rng = np.random.default_rng(1)
    T, d = 4, 5
    n = 3 * T
    x = rng.normal(size=(n, d))
    Wq, Wk, Wv = (rng.normal(size=(d, d)) for _ in range(3))
    mask = np.asarray(causal_mask(n), dtype=bool)
    base = attn(x, mask, Wq, Wk, Wv)
    for cut in range(n - 1):
        x2 = x.copy()
        x2[cut + 1:] = rng.normal(size=(n - cut - 1, d))
        out = attn(x2, mask, Wq, Wk, Wv)
        close(out[: cut + 1], base[: cut + 1], atol=1e-10, label=f"outputs 0 to {cut} after changing tokens {cut + 1} to {n - 1}")
    # WRONG IMPLEMENTATION CAUGHT HERE: a mask such as np.tril(ones, k=1), which lets the state
    # token's output attend to the action token a_t it must predict. The action then leaks into its own prediction.
    for t in range(T):
        x2 = x.copy()
        x2[3 * t + 2] = rng.normal(size=d)
        out = attn(x2, mask, Wq, Wk, Wv)
        close(out[3 * t + 1], base[3 * t + 1], atol=1e-10, label=f"output at s_{t} after changing a_{t}")
    # The mask must still let information flow: the output at s_1 depends on G_1.
    x3 = x.copy()
    x3[3] = rng.normal(size=d)
    if np.allclose(attn(x3, mask, Wq, Wk, Wv)[4], base[4]):
        fail("changing G_1 did not change the output at s_1, so the mask blocks too much")


@check("rollout_with_target lowers the target by each observed reward, and discounting rescales it")
def test_rollout(sub):
    rollout = has(sub, "rollout_with_target")
    mdp = sub.make_world()
    path = [2, 2, 2, 2, 1, 1, 1, 1]  # down four times, right four times
    seen = []

    def act(rtgs, states, actions):
        seen.append((list(rtgs), list(states), list(actions)))
        return path[len(actions)]

    ret, trace = rollout(mdp, act, -8.0, 24, seed=0)
    almost(ret, -8.0, 1e-12, label="return of the 8-step path")
    close(trace, [-8, -7, -6, -5, -4, -3, -2, -1], atol=1e-12, label="target at each step (each -1 reward raises it by 1)")
    if [len(h[0]) for h in seen] != list(range(1, 9)) or [len(h[2]) for h in seen] != list(range(8)):
        fail("act must see t + 1 targets and states but only t actions at step t")
    # With gamma = 0.5 and the exact discounted target, the target at step t is the true return-to-go.
    gamma = 0.5
    target = -sum(gamma**k for k in range(8))
    ret, trace = rollout(mdp, lambda r, s, a: path[len(a)], target, 24, gamma=gamma, seed=0)
    want = [-sum(gamma**k for k in range(8 - t)) for t in range(8)]
    close(trace, want, atol=1e-12, label="discounted targets along the path")


@check("return-conditioned action frequencies pick the lucky arm: asking for 10 chooses an arm worth 1 over one worth 2")
def test_luck(sub):
    cond = has(sub, "conditional_action_probs")
    # Arm 0 pays 10 with probability 0.1 and 0 otherwise (mean 1). Arm 1 pays 2 always (mean 2).
    actions = np.array([0] * 100 + [1] * 100)
    returns = np.array([10.0] * 10 + [0.0] * 90 + [2.0] * 100)
    means = np.array([1.0, 2.0])
    p10 = np.asarray(cond(actions, returns, 10.0, 2))
    close(p10, [1.0, 0.0], atol=1e-12, label="P(a | G = 10)")
    close(np.asarray(cond(actions, returns, 2.0, 2)), [0.0, 1.0], atol=1e-12, label="P(a | G = 2)")
    close(np.asarray(cond(actions, returns, 7.0, 2)), [0.5, 0.5], atol=1e-12, label="P(a | G = 7), a return never seen")
    almost(float(p10 @ means), 1.0, 1e-12, label="expected return when conditioning on 10")
    if p10 @ means >= means.max():
        fail("conditioning on the highest return should pick a worse arm here")


@check("offline_q_iteration stitches two trajectories; the best single trajectory from the start is worse")
def test_stitching(sub):
    q_iter = has(sub, "offline_q_iteration")
    # A: s0 -> s1 -> end pays 0 then 1.  B: s3 -> s1 -> end pays 0 then 10.  Both pass through s1.
    transitions = [(0, 0, 0.0, 1, False), (1, 0, 1.0, 2, True), (3, 0, 0.0, 1, False), (1, 1, 10.0, 4, True)]
    trajs = [
        {"states": np.array([0, 1]), "actions": np.array([0, 0]), "rewards": np.array([0.0, 1.0])},
        {"states": np.array([3, 1]), "actions": np.array([0, 1]), "rewards": np.array([0.0, 10.0])},
    ]
    Q = np.asarray(q_iter(transitions, 5, 2, 1.0, 5))
    almost(Q[1, 1], 10.0, 1e-12, label="Q(s1, second action)")
    almost(Q[0, 0], 10.0, 1e-12, label="Q(s0, a0) stitches A's first step onto B's last step")
    if not np.isneginf(Q[0, 1]):
        fail(f"Q(s0, a1) was never seen in the data and must stay -inf, got {Q[0, 1]}")
    best = sub.max_return_from(trajs, 0)
    almost(best, 1.0, 1e-12, label="best return among trajectories that start at s0")
    if not Q[0].max() > best:
        fail(f"offline Q-iteration should beat the best trajectory from s0 ({best}), got {Q[0].max()}")
    Q9 = np.asarray(q_iter(transitions, 5, 2, 0.9, 5))
    almost(Q9[0, 0], 9.0, 1e-12, label="Q(s0, a0) with gamma = 0.9")


@check("a tiny decision transformer reaches a higher return when you ask for a higher return")
def test_dt_conditioning(sub):
    mdp = sub.make_world()
    trajs = sub.collect_mixed(mdp, 200, seed=0)
    model = sub.train_dt(trajs, mdp.n_states, mdp.n_actions, seed=0)
    act = sub.dt_policy(model, seed=0)
    high = sub.mean_return(mdp, act, -8.0, 20)
    low = sub.mean_return(mdp, act, -20.0, 20)
    if high < -12.0:
        fail(f"asked for -8, the mean return was {high:.1f}, expected above -12 (the best possible is -8)")
    if high - low < 5.0:
        fail(f"asked for -8 the mean return was {high:.1f}; asked for -20 it was {low:.1f}. Expected a gap of at least 5")
